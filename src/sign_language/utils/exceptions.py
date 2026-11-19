"""Domain exceptions hierarchy for sign language recognition system."""


class SignLanguageError(Exception):
    """Base exception for all domain-specific errors in sign_language."""


class ConfigError(SignLanguageError):
    """Raised when configuration validation or loading fails."""


class CameraError(SignLanguageError):
    """Raised when the camera device cannot be opened, read, or configured."""


class DetectionError(SignLanguageError):
    """Raised when landmark or hand detection fails critically."""


class DatasetError(SignLanguageError):
    """Raised when dataset constraints, session counts, or schemas are violated."""


class TrainingError(SignLanguageError):
    """Raised when model candidate training or cross-validation fails."""


class ModelLoadError(SignLanguageError):
    """Raised when model artifacts cannot be loaded from disk."""


class FeatureVersionMismatchError(ModelLoadError):
    """Raised when loaded model feature_set_version differs from runtime features."""


class PredictionError(SignLanguageError):
    """Raised when real-time prediction encounters invalid inputs or states."""
