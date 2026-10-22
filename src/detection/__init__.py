"""YOLO11 detection pipeline: dataset build, training, and HTTP-independent inference.

Modules:
- ``src.detection.dataset``  -- Stage 1: validate the CVAT pilot labels as a
  YOLO11 dataset (image/label pairs, class ids, normalized boxes, missing
  labels, duplicate hashes, split leakage); never fabricates boxes and is
  read-only on the source data. Building the Ultralytics dataset (data.yaml +
  split lists) from the validated labels comes in a later stage.
- ``src.detection.train``    -- reproducible training entry point (seed, imgsz,
  epochs, batch, unique run dirs, config + summary snapshots).
- ``src.detection.inference`` -- checkpoint loading, structured detections,
  annotated images, per-class counts. No FastAPI imports here on purpose: the
  HTTP layer must stay a thin wrapper so this code is testable in isolation.
"""
