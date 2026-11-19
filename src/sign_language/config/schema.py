"""Type-safe, immutable dataclass schema for central config.yaml validation."""

from dataclasses import dataclass
from typing import Any
from sign_language.utils.exceptions import ConfigError


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Project-level metadata."""
    name: str
    short_name: str
    version: str
    description: str
    institution: str
    course: str


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig:
    """Random seed and determinism flags."""
    random_seed: int
    deterministic: bool


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Camera hardware and capture parameters."""
    index: int
    width: int
    height: int
    target_fps: int
    flip_horizontal: bool
    detect_before_flip: bool
    backend: str
    warmup_frames: int
    read_timeout_ms: int

    def validate(self) -> None:
        """Validate camera constraints."""
        if self.width <= 0 or self.height <= 0:
            raise ConfigError(f"Invalid camera resolution: {self.width}x{self.height}")
        if self.target_fps <= 0:
            raise ConfigError(f"Target FPS must be positive, got {self.target_fps}")


@dataclass(frozen=True, slots=True)
class HandDetectionConfig:
    """MediaPipe hand detection settings."""
    max_num_hands: int
    min_detection_confidence: float
    min_tracking_confidence: float
    model_complexity: int
    static_image_mode: bool
    multi_hand_policy: str

    def validate(self) -> None:
        """Validate detection settings."""
        if not (0.0 <= self.min_detection_confidence <= 1.0):
            raise ConfigError(f"Invalid min_detection_confidence: {self.min_detection_confidence}")
        if not (0.0 <= self.min_tracking_confidence <= 1.0):
            raise ConfigError(f"Invalid min_tracking_confidence: {self.min_tracking_confidence}")
        if self.max_num_hands < 1:
            raise ConfigError(f"max_num_hands must be >= 1, got {self.max_num_hands}")


@dataclass(frozen=True, slots=True)
class ExtendedFeaturesConfig:
    """Optional engineered geometric features."""
    fingertip_distances: bool
    inter_fingertip_distances: bool
    finger_curl_angles: bool
    extended_dim: int


@dataclass(frozen=True, slots=True)
class FeaturesConfig:
    """Feature definition and normalisation chain."""
    num_landmarks: int
    coords_per_landmark: int
    base_dim: int
    translation_invariant: bool
    scale_invariant: bool
    rotation_invariant: bool
    mirror_left_to_right: bool
    use_extended: bool
    extended: ExtendedFeaturesConfig
    feature_set_version: str

    def validate(self) -> None:
        """Validate feature dimensions and invariants."""
        expected_base = self.num_landmarks * self.coords_per_landmark
        if self.base_dim != expected_base:
            raise ConfigError(
                f"base_dim ({self.base_dim}) != num_landmarks ({self.num_landmarks}) * "
                f"coords_per_landmark ({self.coords_per_landmark})"
            )
        if self.rotation_invariant:
            raise ConfigError(
                "rotation_invariant must be False per ADR-004 to preserve palm/thumb orientation."
            )


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Data collection boundary validation rules."""
    require_all_landmarks: bool
    reject_out_of_bounds: bool
    coord_min: float
    coord_max: float
    min_hand_bbox_fraction: float
    max_hand_bbox_fraction: float
    max_duplicate_similarity: float

    def validate(self) -> None:
        """Validate bounding box and similarity parameters."""
        if not (0.0 < self.min_hand_bbox_fraction < self.max_hand_bbox_fraction <= 1.0):
            raise ConfigError("Invalid hand bounding box fraction thresholds")
        if not (0.0 < self.max_duplicate_similarity <= 1.0):
            raise ConfigError(f"Invalid max_duplicate_similarity: {self.max_duplicate_similarity}")


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Vocabulary, collection constraints, and targets."""
    classes: list[str]
    neutral_class: str
    samples_per_class_target: int
    samples_per_class_minimum: int
    sessions_per_class_minimum: int
    samples_per_session_cap: int
    capture_fps: int
    countdown_seconds: int
    save_debug_images: bool
    debug_image_every_n: int
    validation: ValidationConfig

    def validate(self) -> None:
        """Validate vocabulary and constraints."""
        if not self.classes:
            raise ConfigError("Dataset classes list cannot be empty")
        if self.neutral_class not in self.classes:
            raise ConfigError(
                f"neutral_class '{self.neutral_class}' must be in classes: {self.classes}"
            )
        if len(self.classes) != len(set(self.classes)):
            raise ConfigError("Duplicate class names found in dataset.classes")
        capacity = self.sessions_per_class_minimum * self.samples_per_session_cap
        if capacity < self.samples_per_class_target:
            raise ConfigError(
                f"Unsatisfiable collection target: {self.sessions_per_class_minimum} sessions * "
                f"{self.samples_per_session_cap} cap = {capacity} < target {self.samples_per_class_target}"
            )
        self.validation.validate()


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Grouped session splitting configuration."""
    strategy: str
    group_column: str
    test_size: float
    val_size: float
    cv_folds: int
    stratify: bool
    shuffle: bool
    report_leakage_comparison: bool

    def validate(self) -> None:
        """Validate split ratios and fold count."""
        if not (0.0 < self.test_size < 1.0):
            raise ConfigError(f"Invalid test_size: {self.test_size}")
        if not (0.0 <= self.val_size < 1.0):
            raise ConfigError(f"Invalid val_size: {self.val_size}")
        if self.cv_folds < 2:
            raise ConfigError(f"cv_folds must be >= 2, got {self.cv_folds}")


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Hyperparameter search options."""
    method: str
    n_iter: int
    scoring: str
    n_jobs: int
    verbose: int
    refit: bool


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Model candidates, metrics, search grids, and class weights."""
    candidates: list[str]
    primary_metric: str
    secondary_metrics: list[str]
    scaler: str
    search: SearchConfig
    grids: dict[str, Any]
    class_weight: str
    fail_on_class_imbalance_ratio: float

    def validate(self) -> None:
        """Validate candidate list and grids."""
        if not self.candidates:
            raise ConfigError("training.candidates list cannot be empty")
        for candidate in self.candidates:
            if candidate not in self.grids:
                raise ConfigError(f"Missing hyperparameter grid for candidate '{candidate}'")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model storage paths and version enforcement."""
    version: str
    classifier_file: str
    label_encoder_file: str
    metadata_file: str
    enforce_feature_version_match: bool


@dataclass(frozen=True, slots=True)
class StabilisationConfig:
    """Rolling-window prediction stabilization parameters."""
    enabled: bool
    window_size: int
    min_votes_in_window: int
    smoothing: str
    ema_alpha: float
    cooldown_seconds: float
    require_neutral_between_repeats: bool
    max_buffer_age_seconds: float

    def validate(self) -> None:
        """Validate stabilization invariants."""
        if self.min_votes_in_window > self.window_size:
            raise ConfigError(
                f"min_votes_in_window ({self.min_votes_in_window}) cannot exceed window_size ({self.window_size})"
            )
        if self.cooldown_seconds < 0:
            raise ConfigError(f"cooldown_seconds must be >= 0, got {self.cooldown_seconds}")


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Online prediction and stabilization."""
    confidence_threshold: float
    stabilisation: StabilisationConfig

    def validate(self) -> None:
        """Validate threshold and stabiliser."""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigError(f"Invalid confidence_threshold: {self.confidence_threshold}")
        self.stabilisation.validate()


@dataclass(frozen=True, slots=True)
class TextConfig:
    """Text composition formatting options."""
    separator: str
    max_length: int
    auto_capitalise: bool
    strip_underscores: bool
    enable_word_suggestions: bool
    enable_text_to_speech: bool

    def validate(self) -> None:
        """Validate text limits."""
        if self.max_length <= 0:
            raise ConfigError(f"max_length must be positive, got {self.max_length}")


@dataclass(frozen=True, slots=True)
class UIAppearanceConfig:
    """Visual theme colors."""
    accent: str
    success: str
    warning: str
    danger: str
    bg: str
    surface: str
    text_primary: str
    text_muted: str


@dataclass(frozen=True, slots=True)
class UIConfig:
    """UI settings and color palette."""
    mode: str
    window_title: str
    window_width: int
    window_height: int
    theme: str
    appearance: UIAppearanceConfig
    font_family: str
    show_landmarks: bool
    show_fps: bool
    show_confidence_bar: bool
    video_target_width: int


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging settings."""
    level: str
    console: bool
    file: bool
    dir: str
    filename: str
    max_bytes: int
    backup_count: int
    format: str


@dataclass(frozen=True, slots=True)
class PerformanceTargetsConfig:
    """Benchmarking threshold targets."""
    min_fps: float
    max_inference_latency_ms: float
    max_model_load_ms: float


@dataclass(frozen=True, slots=True)
class PerformanceConfig:
    """Performance benchmarks and recorded metrics."""
    targets: PerformanceTargetsConfig
    measured: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Relative filesystem paths."""
    data_raw: str
    data_interim: str
    data_processed: str
    data_features: str
    data_metadata: str
    models: str
    reports: str
    figures: str
    metrics: str
    logs: str


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Root configuration aggregating all sub-configurations."""
    project: ProjectConfig
    reproducibility: ReproducibilityConfig
    camera: CameraConfig
    hand_detection: HandDetectionConfig
    features: FeaturesConfig
    dataset: DatasetConfig
    split: SplitConfig
    training: TrainingConfig
    model: ModelConfig
    inference: InferenceConfig
    text: TextConfig
    ui: UIConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    paths: PathsConfig

    def validate(self) -> None:
        """Validate every configuration block."""
        self.camera.validate()
        self.hand_detection.validate()
        self.features.validate()
        self.dataset.validate()
        self.split.validate()
        self.training.validate()
        self.inference.validate()
        self.text.validate()
