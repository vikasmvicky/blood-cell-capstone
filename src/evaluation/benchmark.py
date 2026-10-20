"""Model size, parameter count, CPU latency and throughput.

    python -m src.evaluation.benchmark                    # uses the real checkpoint if present
    python -m src.evaluation.benchmark --architecture-only  # random weights; timing is weight-independent

Latency does not depend on weight values, so ``--architecture-only`` gives valid
timing/size numbers even without the checkpoint; the output is labelled so it can
never be mistaken for a trained-model result.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

from src.config import PROJECT_ROOT, load_config
from src.models.mobilenet import build_mobilenet_v2, count_parameters
from src.utils.seed import set_seed


def _time_calls(fn, repeats: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def benchmark(model: torch.nn.Module, image_size=(224, 224), repeats: int = 50, warmup: int = 10, batch_size: int = 32, threads: int | None = None) -> dict:
    if threads:
        torch.set_num_threads(threads)
    model.eval().cpu()
    x1 = torch.randn(1, 3, *image_size)
    xb = torch.randn(batch_size, 3, *image_size)
    with torch.inference_mode():
        lat = _time_calls(lambda: model(x1), repeats, warmup)
        bat = _time_calls(lambda: model(xb), max(5, repeats // 5), 2)
    lat_ms = sorted(v * 1000 for v in lat)
    return {
        "parameters": count_parameters(model),
        "parameters_millions": round(count_parameters(model) / 1e6, 3),
        "fp32_weights_mb": round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 2),
        "latency_ms_batch1": {
            "mean": round(statistics.mean(lat_ms), 2),
            "median": round(statistics.median(lat_ms), 2),
            "p95": round(lat_ms[int(0.95 * (len(lat_ms) - 1))], 2),
            "repeats": repeats,
        },
        "throughput_images_per_s": round(batch_size / statistics.mean(bat), 1),
        "throughput_batch_size": batch_size,
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--architecture-only", action="store_true")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "reports" / "benchmark.json"))
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if args.architecture_only:
        model, source, size = build_mobilenet_v2(cfg.raw["model"]["num_classes"]), "architecture-only (random weights; timing/size only)", None
        image_size = cfg.image_size
    else:
        from src.inference.errors import CheckpointError
        from src.inference.predictor import load_checkpoint

        try:
            loaded = load_checkpoint(cfg)
        except CheckpointError as exc:
            print(f"[benchmark unavailable] {exc}\nUse --architecture-only for timing without the checkpoint.", file=sys.stderr)
            return 2
        model, image_size = loaded.model, loaded.image_size
        source, size = f"checkpoint {loaded.checkpoint_path.name} (sha256 {loaded.checkpoint_sha256[:12]})", round(loaded.checkpoint_path.stat().st_size / 1e6, 2)

    result = benchmark(model, image_size, repeats=args.repeats, threads=args.threads)
    result["model_source"] = source
    result["checkpoint_file_mb"] = size
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
