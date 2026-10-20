# Model checkpoints (not distributed in this archive)

Put the trained file here:

    outputs/models/best_mobilenet_finetuned.pt

It must be the file written by `python -m src.models.train_mobilenet_finetune`
(a dict with `model_state_dict`, `class_to_idx`, `image_size`, `normalization`, ...).
Alternatively point `BLOOD_CELL_MODEL_PATH` at it. See `../../ASSETS.md`.
