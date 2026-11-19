"""Configuration loader with schema hydration and validation."""

from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml

from sign_language.config.schema import (
    AppConfig,
    CameraConfig,
    DatasetConfig,
    ExtendedFeaturesConfig,
    FeaturesConfig,
    HandDetectionConfig,
    InferenceConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    PerformanceConfig,
    PerformanceTargetsConfig,
    ProjectConfig,
    ReproducibilityConfig,
    SearchConfig,
    SplitConfig,
    StabilisationConfig,
    TextConfig,
    TrainingConfig,
    UIAppearanceConfig,
    UIConfig,
    ValidationConfig,
)
from sign_language.utils.exceptions import ConfigError
from sign_language.utils.paths import project_root


def _parse_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    """Retrieve a dictionary sub-tree or raise ConfigError."""
    val = data.get(key)
    if not isinstance(val, dict):
        raise ConfigError(f"Missing or invalid configuration section: '{key}'")
    return val


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load, parse, and validate config.yaml into an immutable AppConfig object.

    Args:
        path: Path to config.yaml. Defaults to project_root() / 'config.yaml'.

    Returns:
        AppConfig: Fully validated and frozen configuration object.

    Raises:
        ConfigError: If file not found, YAML invalid, or validation fails.
    """
    if path is None:
        cfg_path = project_root() / "config.yaml"
    else:
        cfg_path = Path(path)

    if not cfg_path.exists():
        raise ConfigError(f"Configuration file not found: {cfg_path}")

    try:
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigError(f"Failed to read or parse YAML from {cfg_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw)}")

    try:
        project_dict = _parse_dict(raw, "project")
        project = ProjectConfig(
            name=str(project_dict["name"]),
            short_name=str(project_dict["short_name"]),
            version=str(project_dict["version"]),
            description=str(project_dict["description"]).strip(),
            institution=str(project_dict["institution"]),
            course=str(project_dict["course"]),
        )

        repro_dict = _parse_dict(raw, "reproducibility")
        repro = ReproducibilityConfig(
            random_seed=int(repro_dict["random_seed"]),
            deterministic=bool(repro_dict["deterministic"]),
        )

        cam_dict = _parse_dict(raw, "camera")
        camera = CameraConfig(
            index=int(cam_dict["index"]),
            width=int(cam_dict["width"]),
            height=int(cam_dict["height"]),
            target_fps=int(cam_dict["target_fps"]),
            flip_horizontal=bool(cam_dict["flip_horizontal"]),
            detect_before_flip=bool(cam_dict["detect_before_flip"]),
            backend=str(cam_dict["backend"]),
            warmup_frames=int(cam_dict["warmup_frames"]),
            read_timeout_ms=int(cam_dict["read_timeout_ms"]),
        )

        det_dict = _parse_dict(raw, "hand_detection")
        hand_detection = HandDetectionConfig(
            max_num_hands=int(det_dict["max_num_hands"]),
            min_detection_confidence=float(det_dict["min_detection_confidence"]),
            min_tracking_confidence=float(det_dict["min_tracking_confidence"]),
            model_complexity=int(det_dict["model_complexity"]),
            static_image_mode=bool(det_dict["static_image_mode"]),
            multi_hand_policy=str(det_dict["multi_hand_policy"]),
        )

        feat_dict = _parse_dict(raw, "features")
        ext_dict = _parse_dict(feat_dict, "extended")
        extended = ExtendedFeaturesConfig(
            fingertip_distances=bool(ext_dict["fingertip_distances"]),
            inter_fingertip_distances=bool(ext_dict["inter_fingertip_distances"]),
            finger_curl_angles=bool(ext_dict["finger_curl_angles"]),
            extended_dim=int(ext_dict["extended_dim"]),
        )
        features = FeaturesConfig(
            num_landmarks=int(feat_dict["num_landmarks"]),
            coords_per_landmark=int(feat_dict["coords_per_landmark"]),
            base_dim=int(feat_dict["base_dim"]),
            translation_invariant=bool(feat_dict["translation_invariant"]),
            scale_invariant=bool(feat_dict["scale_invariant"]),
            rotation_invariant=bool(feat_dict["rotation_invariant"]),
            mirror_left_to_right=bool(feat_dict["mirror_left_to_right"]),
            use_extended=bool(feat_dict["use_extended"]),
            extended=extended,
            feature_set_version=str(feat_dict["feature_set_version"]),
        )

        ds_dict = _parse_dict(raw, "dataset")
        val_dict = _parse_dict(ds_dict, "validation")
        validation = ValidationConfig(
            require_all_landmarks=bool(val_dict["require_all_landmarks"]),
            reject_out_of_bounds=bool(val_dict["reject_out_of_bounds"]),
            coord_min=float(val_dict["coord_min"]),
            coord_max=float(val_dict["coord_max"]),
            min_hand_bbox_fraction=float(val_dict["min_hand_bbox_fraction"]),
            max_hand_bbox_fraction=float(val_dict["max_hand_bbox_fraction"]),
            max_duplicate_similarity=float(val_dict["max_duplicate_similarity"]),
        )
        dataset = DatasetConfig(
            classes=[str(c) for c in ds_dict["classes"]],
            neutral_class=str(ds_dict["neutral_class"]),
            samples_per_class_target=int(ds_dict["samples_per_class_target"]),
            samples_per_class_minimum=int(ds_dict["samples_per_class_minimum"]),
            sessions_per_class_minimum=int(ds_dict["sessions_per_class_minimum"]),
            samples_per_session_cap=int(ds_dict["samples_per_session_cap"]),
            capture_fps=int(ds_dict["capture_fps"]),
            countdown_seconds=int(ds_dict["countdown_seconds"]),
            save_debug_images=bool(ds_dict["save_debug_images"]),
            debug_image_every_n=int(ds_dict["debug_image_every_n"]),
            validation=validation,
        )

        split_dict = _parse_dict(raw, "split")
        split = SplitConfig(
            strategy=str(split_dict["strategy"]),
            group_column=str(split_dict["group_column"]),
            test_size=float(split_dict["test_size"]),
            val_size=float(split_dict["val_size"]),
            cv_folds=int(split_dict["cv_folds"]),
            stratify=bool(split_dict["stratify"]),
            shuffle=bool(split_dict["shuffle"]),
            report_leakage_comparison=bool(split_dict["report_leakage_comparison"]),
        )

        train_dict = _parse_dict(raw, "training")
        search_dict = _parse_dict(train_dict, "search")
        search = SearchConfig(
            method=str(search_dict["method"]),
            n_iter=int(search_dict["n_iter"]),
            scoring=str(search_dict["scoring"]),
            n_jobs=int(search_dict["n_jobs"]),
            verbose=int(search_dict["verbose"]),
            refit=bool(search_dict["refit"]),
        )
        training = TrainingConfig(
            candidates=[str(c) for c in train_dict["candidates"]],
            primary_metric=str(train_dict["primary_metric"]),
            secondary_metrics=[str(m) for m in train_dict["secondary_metrics"]],
            scaler=str(train_dict["scaler"]),
            search=search,
            grids=train_dict["grids"],
            class_weight=str(train_dict["class_weight"]),
            fail_on_class_imbalance_ratio=float(train_dict["fail_on_class_imbalance_ratio"]),
        )

        model_dict = _parse_dict(raw, "model")
        model = ModelConfig(
            version=str(model_dict["version"]),
            classifier_file=str(model_dict["classifier_file"]),
            label_encoder_file=str(model_dict["label_encoder_file"]),
            metadata_file=str(model_dict["metadata_file"]),
            enforce_feature_version_match=bool(model_dict["enforce_feature_version_match"]),
        )

        infer_dict = _parse_dict(raw, "inference")
        stab_dict = _parse_dict(infer_dict, "stabilisation")
        stabilisation = StabilisationConfig(
            enabled=bool(stab_dict["enabled"]),
            window_size=int(stab_dict["window_size"]),
            min_votes_in_window=int(stab_dict["min_votes_in_window"]),
            smoothing=str(stab_dict["smoothing"]),
            ema_alpha=float(stab_dict["ema_alpha"]),
            cooldown_seconds=float(stab_dict["cooldown_seconds"]),
            require_neutral_between_repeats=bool(stab_dict["require_neutral_between_repeats"]),
            max_buffer_age_seconds=float(stab_dict["max_buffer_age_seconds"]),
        )
        inference = InferenceConfig(
            confidence_threshold=float(infer_dict["confidence_threshold"]),
            stabilisation=stabilisation,
        )

        text_dict = _parse_dict(raw, "text")
        text = TextConfig(
            separator=str(text_dict["separator"]),
            max_length=int(text_dict["max_length"]),
            auto_capitalise=bool(text_dict["auto_capitalise"]),
            strip_underscores=bool(text_dict["strip_underscores"]),
            enable_word_suggestions=bool(text_dict["enable_word_suggestions"]),
            enable_text_to_speech=bool(text_dict["enable_text_to_speech"]),
        )

        ui_dict = _parse_dict(raw, "ui")
        app_dict = _parse_dict(ui_dict, "appearance")
        appearance = UIAppearanceConfig(
            accent=str(app_dict["accent"]),
            success=str(app_dict["success"]),
            warning=str(app_dict["warning"]),
            danger=str(app_dict["danger"]),
            bg=str(app_dict["bg"]),
            surface=str(app_dict["surface"]),
            text_primary=str(app_dict["text_primary"]),
            text_muted=str(app_dict["text_muted"]),
        )
        ui = UIConfig(
            mode=str(ui_dict["mode"]),
            window_title=str(ui_dict["window_title"]),
            window_width=int(ui_dict["window_width"]),
            window_height=int(ui_dict["window_height"]),
            theme=str(ui_dict["theme"]),
            appearance=appearance,
            font_family=str(ui_dict["font_family"]),
            show_landmarks=bool(ui_dict["show_landmarks"]),
            show_fps=bool(ui_dict["show_fps"]),
            show_confidence_bar=bool(ui_dict["show_confidence_bar"]),
            video_target_width=int(ui_dict["video_target_width"]),
        )

        log_dict = _parse_dict(raw, "logging")
        logging_cfg = LoggingConfig(
            level=str(log_dict["level"]),
            console=bool(log_dict["console"]),
            file=bool(log_dict["file"]),
            dir=str(log_dict["dir"]),
            filename=str(log_dict["filename"]),
            max_bytes=int(log_dict["max_bytes"]),
            backup_count=int(log_dict["backup_count"]),
            format=str(log_dict["format"]),
        )

        perf_dict = _parse_dict(raw, "performance")
        targets_dict = _parse_dict(perf_dict, "targets")
        targets = PerformanceTargetsConfig(
            min_fps=float(targets_dict["min_fps"]),
            max_inference_latency_ms=float(targets_dict["max_inference_latency_ms"]),
            max_model_load_ms=float(targets_dict["max_model_load_ms"]),
        )
        performance = PerformanceConfig(
            targets=targets,
            measured=perf_dict.get("measured", {}),
        )

        paths_dict = _parse_dict(raw, "paths")
        paths = PathsConfig(
            data_raw=str(paths_dict["data_raw"]),
            data_interim=str(paths_dict["data_interim"]),
            data_processed=str(paths_dict["data_processed"]),
            data_features=str(paths_dict["data_features"]),
            data_metadata=str(paths_dict["data_metadata"]),
            models=str(paths_dict["models"]),
            reports=str(paths_dict["reports"]),
            figures=str(paths_dict["figures"]),
            metrics=str(paths_dict["metrics"]),
            logs=str(paths_dict["logs"]),
        )

        config = AppConfig(
            project=project,
            reproducibility=repro,
            camera=camera,
            hand_detection=hand_detection,
            features=features,
            dataset=dataset,
            split=split,
            training=training,
            model=model,
            inference=inference,
            text=text,
            ui=ui,
            logging=logging_cfg,
            performance=performance,
            paths=paths,
        )

        config.validate()
        return config

    except KeyError as exc:
        raise ConfigError(f"Missing required configuration key: {exc}") from exc


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Retrieve the cached singleton application configuration.

    Returns:
        AppConfig: Validated application configuration.
    """
    return load_config()
