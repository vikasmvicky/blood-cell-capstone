import json

import pytest

from src.config import PROJECT_ROOT, class_names_from_mapping, load_class_names, load_config


def test_config_loads_and_uses_relative_paths(real_config):
    assert real_config.image_size == (224, 224)
    assert len(real_config.mean) == len(real_config.std) == 3
    assert real_config.model_path.is_absolute()
    assert str(real_config.model_path).startswith(str(PROJECT_ROOT))
    assert "D:\\" not in json.dumps(real_config.raw)


def test_repository_class_mapping_matches_config(real_config):
    names = load_class_names(real_config.class_map_path)
    assert names == ["basophil", "eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"]
    assert set(real_config.display_names) == set(names)
    assert real_config.raw["model"]["num_classes"] == len(names)


def test_class_names_from_mapping_rejects_gaps():
    with pytest.raises(ValueError):
        class_names_from_mapping({"a": 0, "b": 2})
    with pytest.raises(ValueError):
        class_names_from_mapping({})


def test_env_override_model_path(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOOD_CELL_MODEL_PATH", str(tmp_path / "x.pt"))
    assert load_config().model_path == tmp_path / "x.pt"
