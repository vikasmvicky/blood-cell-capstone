"""Exception types with user-presentable messages."""


class InvalidImageError(ValueError):
    """The uploaded file cannot be used as a blood-cell image."""


class CheckpointError(RuntimeError):
    """Base class for model-checkpoint problems."""


class CheckpointNotFoundError(CheckpointError):
    """The trained checkpoint is not present at the expected location."""


class CheckpointIncompatibleError(CheckpointError):
    """The checkpoint exists but does not match the expected architecture/classes."""
