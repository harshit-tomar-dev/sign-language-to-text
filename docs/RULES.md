# RULES.md — Binding Development Standards

**Project:** Real-Time Sign Language to Text Conversion System Using Machine Learning
**Institution:** SRM Institute of Science and Technology — B.Tech CSE, Machine Learning (80 marks)
**Status of this document:** ACTIVE and BINDING as of Phase 0 completion.

---

## 1. Purpose, Scope, and How to Cite a Rule

### 1.1 Purpose

This document turns the project's engineering intent into **enforceable rules**. It exists because a
three-person team working in parallel across a 4–6 week window will diverge unless the conventions
are written down and citable. A rule here is not advice; it is the standard a change is measured
against during review.

### 1.2 Scope

| Applies to | Does not apply to |
| --- | --- |
| `src/sign_language/**` — all library code | `notebooks/**` (exploratory only, never imported) |
| `scripts/**` — CLI entry points | Generated artifacts in `data/`, `models/`, `logs/` |
| `tests/**` — all test code | Third-party code inside `.venv/` |
| `app.py` — desktop launcher | — |
| `docs/**`, `config.yaml`, dependency files | — |

Notebooks are exempt from style rules but are **never** an accepted home for logic that the
application depends on (see **PS-06**).

### 1.3 How to cite a rule in review

Every rule has a stable ID: a two-or-three letter section prefix plus a zero-padded number.
Quote the ID in the review comment, then the fix. Nothing else is needed.

```text
CF-01: confidence threshold is hard-coded at module level here.
       Move it to config.yaml -> inference.confidence_threshold and read it
       from the injected InferenceConfig object.

DS-03: this split uses train_test_split without groups. Reported numbers from a
       random split are not admissible (GR-02). Use StratifiedGroupKFold on
       session_id.
```

Rule IDs are **append-only**. If a rule is retired, mark it `WITHDRAWN` in place and never reuse the
number — old review comments must keep resolving to the right text.

### 1.4 Enforcement classes

| Class | Meaning |
| --- | --- |
| **AUTO** | Machine-checked. CI/pre-commit fails. The named tool is authoritative. |
| **REVIEW** | Human-checked at pull-request review time against the checklist in §17. |
| **DESIGN** | Architectural invariant. Violating it means the design changed — needs an ADR (**DOC-05**). |

### 1.5 Sibling documents

`PHASES.md` (roadmap and per-phase completion criteria) · `PRD.md` (requirements, team roster) ·
`ARCHITECTURE.md` (module design and ADR log) · `DATASET.md` (vocabulary and leakage rationale) ·
`MODEL.md` (feature and model design decisions) · `MEMORY.md` (running decision log) ·
`TROUBLESHOOTING.md` (environment failures).

---

## 2. Golden Rules — The Non-Negotiables

These eight override everything else in this document. If a later rule appears to permit a Golden
Rule violation, the later rule is wrong.

| ID | Rule | Class | Why |
| --- | --- | --- | --- |
| **GR-01** | **`config.yaml` is the only source of truth for tunables.** No tunable number, threshold, path, or class name is written in Python source. | AUTO + REVIEW | One place to change, one place to audit. A threshold duplicated in three modules will drift, and the version in the report will match none of them. |
| **GR-02** | **No data leakage.** Samples are split by `session_id` using `StratifiedGroupKFold`, never randomly, for any number that will be reported. | REVIEW | Frames captured milliseconds apart are near-duplicates. A random split scatters them across train and test, so the model is graded on data it memorised. It reads as ~99% and collapses live. |
| **GR-03** | **Never fabricate a result.** Every metric in every document is either produced by a script in this repo or written `TBD — measure in Phase N`. | REVIEW | This is an assessed academic project. An invented number is academic misconduct, and the viva will find it. |
| **GR-04** | **Never commit datasets or model binaries.** `.npz`, `.npy`, `.joblib`, images, and video stay local. Manifests and `model_metadata.json` are committed. | AUTO (`.gitignore`) | Size and privacy. The dataset is webcam footage of the three of us; pushing it publishes our likeness. |
| **GR-05** | **ML logic never imports UI.** `features/`, `preprocessing/`, `training/`, `inference/`, `text/` must never import `ui/`, `customtkinter`, or `tkinter`. | AUTO (import test) + DESIGN | The pipeline has to run headless from `scripts/`. One UI import in a feature module makes training depend on a display. |
| **GR-06** | **Every reported metric is reproducible from a named script.** No hand-computed numbers, no numbers that only exist in a notebook. | REVIEW | If a grader asks "how did you get that", the answer is a command, not a memory. |
| **GR-07** | **`TBD` is mandatory for unmeasured values.** Placeholders are never plausible-looking guesses. | REVIEW | A guess that looks like a measurement is indistinguishable from a lie once it is three commits old. |
| **GR-08** | **No hard-coded tunables — ever.** Restated deliberately because it is the rule most likely to be broken under deadline pressure. | AUTO + REVIEW | See **CF-01** for the enforcement detail and the canonical GOOD/BAD example. |

---

## 3. Coding Standards

Tooling is authoritative. Where a rule below is marked AUTO, the tool's verdict wins over opinion.
Settings come from `pyproject.toml`; do not pass conflicting flags on the command line.

| ID | Rule | Class | Enforced by |
| --- | --- | --- | --- |
| **CS-01** | PEP 8, with formatting delegated entirely to `black`. Never hand-format to fight the formatter. | AUTO | `black --check .` |
| **CS-02** | Line length is **100**. Set in `[tool.black] line-length = 100` and `[tool.ruff] line-length = 100`. Note that `E501` is in ruff's `ignore` list precisely because `black` owns wrapping — do not re-enable it. | AUTO | `black`, `ruff` |
| **CS-03** | Lint clean under the selected rule sets: `E, W, F, I, N, UP, B, C4, SIM, PTH, RET, ARG, D`. Ignored globally: `D203`, `D213`, `E501`. | AUTO | `ruff check .` |
| **CS-04** | Type hints on **every** public function, method, and module-level constant in `src/sign_language/`. `disallow_untyped_defs = true` is set, so an untyped def is a hard error. | AUTO | `mypy` |
| **CS-05** | Google-style docstrings on all public functions, classes, and modules. `convention = "google"` under `[tool.ruff.lint.pydocstyle]`. Tests are exempt (`per-file-ignores` for `tests/**`); `scripts/**` is exempt only from `D103`, so scripts still need a module docstring. | AUTO | `ruff` (D rules) |
| **CS-06** | f-strings for interpolation. No `%` formatting, no `.format()`. The one exception is logging — see **LG-06**. | AUTO | `ruff` (`UP`) |
| **CS-07** | `pathlib.Path` for all filesystem work. `os.path` is banned. `PTH` is selected in ruff for exactly this. | AUTO | `ruff` (`PTH`) |
| **CS-08** | No bare `except:` and no blanket `except Exception:` without re-raise or a logged, typed handler. | AUTO + REVIEW | `ruff` (`E722`, `B`) |
| **CS-09** | No mutable default arguments (`def f(x: list = [])`). Use `None` and construct inside. | AUTO | `ruff` (`B006`) |
| **CS-10** | No wildcard imports (`from x import *`) anywhere, including `__init__.py`. | AUTO | `ruff` (`F403`, `F405`) |
| **CS-11** | Module-level logger: `logger = logging.getLogger(__name__)`. Never a module-level `logging.basicConfig`, never a bare root-logger call. | REVIEW | §11 |
| **CS-12** | **`print()` is banned in `src/sign_language/**`.** It is permitted only in `scripts/**` and `app.py` for deliberate CLI output. Library code communicates through return values, exceptions, and the logger. | REVIEW | §11 |
| **CS-13** | No commented-out code blocks on `main`. Git history is the archive. | REVIEW | — |
| **CS-14** | `mypy` currently covers `files = ["src/sign_language"]` only. `scripts/` and `tests/` are therefore untyped by configuration — that is deliberate, but it means **CS-04** is only machine-enforced inside the package. Type hints in `scripts/` are REVIEW-class. | REVIEW | — |

### CS-12 / CS-11 — GOOD vs BAD

```python
# BAD — src/sign_language/inference/predictor.py
def predict(self, features):                 # CS-04: no type hints
    print("predicting...")                   # CS-12: print() in library code
    try:
        probs = self._pipeline.predict_proba([features])[0]
    except:                                  # CS-08: bare except
        print("failed")                      # CS-12 again, and the error is now invisible
        return None
    return probs
```

```python
# GOOD — src/sign_language/inference/predictor.py
"""Single-frame classification of hand-landmark feature vectors."""

import logging

import numpy as np

from sign_language.core.exceptions import PredictionError

logger = logging.getLogger(__name__)


def predict_proba(self, features: np.ndarray) -> np.ndarray:
    """Return class probabilities for one feature vector.

    Args:
        features: Feature vector of shape ``(n_features,)`` matching the
            ``feature_set_version`` the loaded model was trained on.

    Returns:
        Probability vector of shape ``(n_classes,)`` aligned to
        ``self.label_encoder.classes_``.

    Raises:
        PredictionError: If the feature vector width does not match the model.
    """
    if features.shape[0] != self._expected_dim:
        raise PredictionError(
            f"Feature width {features.shape[0]} does not match model "
            f"({self._expected_dim}). Retrain, or load the matching model version."
        )
    logger.debug("Predicting on feature vector of width %d", features.shape[0])
    return self._pipeline.predict_proba(features.reshape(1, -1))[0]
```

### CS-09 — GOOD vs BAD

```python
# BAD — the list is created once at import and shared by every caller
def collect(labels: list[str] = []) -> list[str]: ...

# GOOD
def collect(labels: list[str] | None = None) -> list[str]:
    labels = [] if labels is None else labels
    ...
```

---

## 4. Naming Conventions

| Entity | Convention | Example | Notes |
| --- | --- | --- | --- |
| Package | `lower_snake`, singular | `sign_language`, `features` | Enforced by `ruff N`. |
| Module | `lower_snake.py`, noun or noun-phrase | `landmark_extractor.py`, `stabiliser.py` | One responsibility (**PS-02**). |
| Class | `PascalCase` | `LandmarkExtractor`, `PredictionStabiliser` | No `Manager`/`Helper`/`Utils` suffixes. |
| Function / method | `lower_snake`, verb-first | `extract_landmarks()`, `build_feature_vector()` | |
| Boolean function | `is_`/`has_`/`should_` prefix | `is_hand_present()` | |
| Constant | `UPPER_SNAKE` at module scope | `MEDIAPIPE_NUM_LANDMARKS` | **Only** for library-fixed facts, never tunables (**CF-01**). |
| Private member | single leading underscore | `_pipeline`, `_validate_frame()` | Double underscore only for genuine name-mangling needs. |
| Test module | `test_<module_under_test>.py` | `tests/unit/test_normaliser.py` | Mirrors the `src` tree. |
| Test function | `test_<behaviour>_<condition>` | `test_scale_norm_rejects_zero_span()` | Reads as a sentence in `pytest -v`. |
| Fixture | `lower_snake`, noun | `dummy_landmarks`, `tmp_dataset_dir` | |
| Config key | `lower_snake`, nested by domain | `inference.stabilisation.window_size` | Mirrors the dataclass field name exactly (**CF-06**). |
| Class label | `UPPER_SNAKE` | `THANK_YOU`, `I_LOVE_YOU` | Multi-word labels joined with `_`. |
| Non-emitting label | **leading underscore reserved** | `_NEUTRAL` | A leading underscore means "this class produces no text output". Any future rejection class follows the same rule. |
| Session id | `s{NN}`, zero-padded, per-signer monotonic | `s01`, `s02`, `s07` | The grouping key for **GR-02**. Never reused across signers. |
| Signer id | `signer{N}` | `signer1`, `signer2`, `signer3` | Pseudonymous by design (**SEC-06**). Real names live only in `PRD.md`. |
| Model version | `v{N}` | `v1`, `v2` | Directory under `models/`. Bumped per **ML-07**. |
| Feature-set version | `v{N}-{name}-{dim}` | `v1-base-63` | Current value: `v1-base-63`. |
| Branch | `<type>/<short-kebab-desc>` | `feat/landmark-extractor` | Types in §12. |
| Report artifact | `lower_snake` with phase or model tag | `confusion_matrix_v1.png`, `leakage_comparison.json` | Lives in `reports/`. |

### NM-01 — Dataset file naming (mandatory, machine-parsed)

Interim per-sample arrays are written to:

```text
data/interim/{CLASS}/{CLASS}__{signer_id}__s{NN}__{YYYYMMDD-HHMMSS}.npz
```

Concrete example:

```text
data/interim/THANK_YOU/THANK_YOU__signer2__s03__20260901-153042.npz
```

| ID | Rule | Class |
| --- | --- | --- |
| **NM-01** | The four fields are separated by **double** underscores; words inside a field use single underscores. This is what makes `THANK_YOU` parseable. | REVIEW + AUTO (validator) |
| **NM-02** | `{CLASS}` in the filename must equal the parent directory name and must be a member of `dataset.classes` in `config.yaml`. | AUTO (`build_dataset.py`) |
| **NM-03** | Timestamp is local time, `YYYYMMDD-HHMMSS`, no separators inside the date or time parts. It is a tiebreaker, not the grouping key. | AUTO |
| **NM-04** | `session_id` for grouping is the pair `(signer_id, sNN)`. Two signers may both have `s01`; they are different groups. | DESIGN |
| **NM-05** | Filenames are never hand-edited to fix a mislabel. Delete the file and re-record (**DS-06**). | REVIEW |

---

## 5. Project Structure Rules

| ID | Rule | Class |
| --- | --- | --- |
| **PS-01** | **src-layout.** All importable code lives under `src/sign_language/`. Configured by `[tool.setuptools.packages.find] where = ["src"]` and `pytest pythonpath = ["src"]`. Never add a top-level `sign_language/` package. | AUTO |
| **PS-02** | **One responsibility per module.** A module that both reads the camera and computes features is split. | REVIEW |
| **PS-03** | **No module over ~400 lines** excluding docstrings. At 400 lines, split it. This is a smell threshold, not a hard gate. | REVIEW |
| **PS-04** | **No circular imports.** The dependency direction is fixed: `config` → `core` → `{features, preprocessing, text}` → `{training, inference}` → `ui`. Arrows never reverse. | AUTO (`ruff` cannot catch all; add `tests/unit/test_imports.py`) |
| **PS-05** | **`__init__.py` exports only the public surface** via explicit `from .module import Name` plus `__all__`. `F401` and `D104` are ignored in `__init__.py` precisely because re-exports there are intentional — that exemption does not license wildcard imports (**CS-10**). | REVIEW |
| **PS-06** | **`scripts/**` are thin CLI wrappers with NO business logic.** A script may parse arguments, load config, set up logging, call into `src/sign_language/`, print a summary, and set an exit code. Anything else belongs in the package. Rule of thumb: if you cannot delete a script and still reproduce the result from a three-line Python session, the logic is in the wrong place. | REVIEW |
| **PS-07** | `notebooks/` is exploratory. Nothing in `src/` or `scripts/` may import from a notebook, and no reported number may originate in one (**GR-06**). | REVIEW |
| **PS-08** | `app.py` at the repository root is a launcher only: build config, build logging, construct the desktop app, run it. No pipeline code. | REVIEW |
| **PS-09** | Tests mirror the source tree under `tests/unit/`, `tests/integration/`, `tests/model/`. | REVIEW |

### Canonical module map

```text
src/sign_language/
  config/          loader.py, schema.py            <- ONLY reader of config.yaml (CF-03)
  core/            exceptions.py, types.py, constants.py
  data_collection/ collector.py, validator.py, session.py
  features/        landmarks.py, normalise.py, extended.py, builder.py
  preprocessing/   dataset_builder.py, splitter.py, manifest.py
  training/        candidates.py, trainer.py, search.py, persist.py
  inference/       loader.py, predictor.py, stabiliser.py, pipeline.py
  text/            composer.py
  ui/              hud.py, desktop_app.py, widgets.py
  utils/           seeding.py, paths.py, logging_setup.py, timing.py
scripts/           collect_data.py, build_dataset.py, train_model.py,
                   evaluate_model.py, run_realtime.py, benchmark.py
tests/             unit/, integration/, model/
app.py
```

---

## 6. Configuration Rules

| ID | Rule | Class |
| --- | --- | --- |
| **CF-01** | **No hard-coded tunables.** Any value that a reasonable person might want to change without editing logic — threshold, window size, path, class list, colour, fps, seed — lives in `config.yaml` and reaches code as a field on a typed config object. | AUTO (review script + REVIEW) |
| **CF-02** | Library-fixed facts are the only permitted module constants: `MEDIAPIPE_NUM_LANDMARKS = 21`, `COORDS_PER_LANDMARK = 3`. Even these are cross-checked against `features.num_landmarks` / `features.coords_per_landmark` at startup and raise on mismatch. | REVIEW |
| **CF-03** | **Only `src/sign_language/config/loader.py` may open `config.yaml`.** No other module calls `yaml.safe_load` on it. Grep-checkable: `yaml.safe_load` appears in exactly one file. | AUTO (grep test) |
| **CF-04** | Config is **validated at startup and fails fast**. `loader.py` parses YAML into the dataclasses in `config/schema.py`, checks invariants, and raises `ConfigError` with the offending key path before any camera or model is touched. Never `dict.get(key, default)` at the point of use — a missing key is a config bug, not a runtime fallback. | REVIEW |
| **CF-05** | **Local overrides go in `config.local.yaml`**, which is gitignored. `loader.py` deep-merges it over `config.yaml` when present. Never commit a machine-specific `camera.index` into `config.yaml`. | REVIEW |
| **CF-06** | **Adding a tunable = two edits, same commit:** the key in `config.yaml` (with a `[FIXED]` or `[TUNE:Phase-N]` comment) **and** the field in the schema dataclass. A key with no dataclass field is silently ignored; a field with no key breaks startup. | REVIEW |
| **CF-07** | Config objects are **frozen** (`@dataclass(frozen=True)`) and passed by injection. No module mutates config at runtime and no module reaches for a global singleton. | REVIEW |
| **CF-08** | Values marked `[TUNE:Phase-N]` in `config.yaml` are **starting points, not results**. Changing one requires a line in `MEMORY.md` recording what experiment justified it. | REVIEW |
| **CF-09** | `performance.measured.*` holds YAML `null` (with a `# TBD — measure in Phase 13` comment), **not** the bare token `TBD` — a string where a number belongs breaks typed schema validation, which was audit defect 7. The schema field is therefore `float \| None`. Phase 13 replaces these with numbers written by `scripts/benchmark.py`, never by hand. A `null` here means "not yet measured" and must never be read as zero. | AUTO (schema) |
| **CF-10** | **Every class name in `config.yaml` is quoted, and the quotes are load-bearing.** YAML 1.1 resolves the bare tokens `YES`, `NO`, `ON`, `OFF`, `Y`, `N` to booleans, so `- YES` parses as `True` and `- NO` as `False`. Unquoted, two of the thirteen classes become booleans with **no error at load time** — producing a `data/raw/True/` directory, a label encoder fitted on mixed types, and log lines that print `True` where a sign name belongs. This was audit defect 12. `loader.py` asserts `all(isinstance(c, str) for c in classes)` and a unit test covers it. More generally: **validate config by loading it, not by reading it** — this defect was invisible to eleven passes of careful proofreading and obvious the first time the file was parsed. | AUTO (loader assertion + test) |

### CF-01 — GOOD vs BAD (the canonical example)

```python
# ============================ BAD ============================
# src/sign_language/inference/stabiliser.py

CONFIDENCE_THRESHOLD = 0.8      # CF-01 / GR-01 violation
WINDOW_SIZE = 10                # CF-01 violation
REQUIRED_CONSECUTIVE = 7        # CF-01 violation
COOLDOWN = 1.2                  # CF-01 violation


class PredictionStabiliser:
    def __init__(self):
        self._buffer = deque(maxlen=WINDOW_SIZE)

    def push(self, label, prob):
        if prob < CONFIDENCE_THRESHOLD:      # which 0.8? the one in this file,
            return None                      # or the one in config.yaml, or the
        ...                                  # one in the report? Nobody knows.
```

Why this is fatal rather than merely untidy: the number in the report, the number in
`config.yaml`, and the number the program actually used are now three independent facts. When the
viva asks "why 0.80?", there is no defensible answer, and any threshold sweep in Phase 9 silently
has no effect.

```python
# ============================ GOOD ===========================
# src/sign_language/config/schema.py

@dataclass(frozen=True)
class StabilisationConfig:
    """Prediction-stabilisation parameters from ``inference.stabilisation``."""

    enabled: bool
    window_size: int
    min_votes_in_window: int
    smoothing: str
    ema_alpha: float
    cooldown_seconds: float
    require_neutral_between_repeats: bool
    max_buffer_age_seconds: float


@dataclass(frozen=True)
class InferenceConfig:
    """Parameters from the ``inference`` block of config.yaml."""

    confidence_threshold: float
    stabilisation: StabilisationConfig


# src/sign_language/inference/stabiliser.py

class PredictionStabiliser:
    """Commits a label only once it is stable across a rolling window."""

    def __init__(self, config: InferenceConfig, neutral_label: str) -> None:
        """Initialise the stabiliser.

        Args:
            config: Inference configuration loaded from config.yaml.
            neutral_label: Label that must be seen between repeats, from
                ``dataset.neutral_class``.
        """
        self._config = config
        self._neutral_label = neutral_label
        self._buffer: deque[str] = deque(maxlen=config.stabilisation.window_size)
```

Now `inference.confidence_threshold: 0.80` in `config.yaml` is the single fact, the Phase 9 sweep
works by editing config, and the reported value is provably the value used.

---

## 7. Dataset Rules

| ID | Rule | Class |
| --- | --- | --- |
| **DS-01** | **Every sample carries `session_id` and `signer_id`.** Written into the `.npz` at capture time and into the session manifest. A sample without both is invalid and is dropped by `build_dataset.py` with a loud warning. | AUTO |
| **DS-02** | **Never split randomly for reported results.** `split.strategy` is `grouped_by_session` with `group_column: session_id`, `test_size: 0.20`, `val_size: 0.15`, `cv_folds: 5`. The `random` strategy exists **only** to demonstrate leakage in Phase 7 and its output is always labelled as such. | REVIEW |
| **DS-03** | **Never mix feature-set versions.** All samples in one dataset build share one `feature_set_version` (currently `v1-base-63`). Mixing 63-dim and 83-dim vectors is a silent corruption, not an error. | AUTO (`build_dataset.py`) |
| **DS-04** | **Validation rules are always on.** `require_all_landmarks`, `reject_out_of_bounds`, `min_hand_bbox_fraction: 0.04`, `max_hand_bbox_fraction: 0.90`, `max_duplicate_similarity: 0.9995`. Disabling a check to "get more samples" is forbidden — a rejected frame is a frame the model would have learned nothing from. | REVIEW |
| **DS-05** | **Never hand-edit a generated array.** `.npz`, `.npy`, and the processed splits are build outputs. Opening one in a script to "fix" a label is banned. | REVIEW |
| **DS-06** | **Regenerate rather than patch.** Wrong data means: delete the offending session directory, re-record, re-run `build_dataset.py`. The pipeline must be able to reproduce `data/processed/` from `data/interim/` with one command. | DESIGN |
| **DS-07** | **The dataset is never committed.** `data/raw/**`, `data/interim/**`, `data/processed/**`, `data/features/**` are gitignored (**GR-04**). | AUTO (`.gitignore`) |
| **DS-08** | **Manifests are always committed.** `data/metadata/**` (class registry, per-session manifests, counts) is force-included in `.gitignore` — with `.npz`/`.npy` re-excluded under it. Manifests are the evidence that the methodology was followed without shipping anyone's likeness. | AUTO |
| **DS-09** | Minimum viable dataset: **200 samples/class** (`samples_per_class_minimum`). `train_model.py` **refuses to run** below this. Target is **330/class**, at least **4 sessions/class**, at most **90 samples/session**. | AUTO (`train_model.py`) |
| **DS-10** | Class balance gate: abort training if `max_class_count / min_class_count > 3.0` (`fail_on_class_imbalance_ratio`). | AUTO |
| **DS-11** | Capture at `capture_fps: 8`, not 30. Holding a pose at 30 fps produces near-identical frames that inflate the count without adding information — and then get caught by `max_duplicate_similarity` anyway. | DESIGN |
| **DS-12** | `_NEUTRAL` is collected with the same rigour as every real sign. It is the rejection class; a weak `_NEUTRAL` forces the model to label a resting hand as a word. | REVIEW |

**Arithmetic note on DS-09 — resolved in Phase 0.** An earlier draft set
`samples_per_session_cap: 80`, so four sessions yielded at most 320 samples — 10 short of the 330
target. The documented minimum could not reach the documented target. The cap was raised to **90**,
giving 4 x 90 = 360 >= 330, and `AppConfig.validate()` now enforces the invariant
`sessions_per_class_minimum * samples_per_session_cap >= samples_per_class_target` at startup.
The planned schedule (3 signers x 2 sessions = 6 sessions/class, roughly 55 samples each) sits
comfortably inside this. Treat `sessions_per_class_minimum: 4` as the **abort floor, not the plan**.

---

## 8. Model Rules

| ID | Rule | Class |
| --- | --- | --- |
| **ML-01** | **Serialise the whole `Pipeline`**, scaler included, to `models/v1/sign_classifier.joblib`. Never persist a bare estimator and re-apply scaling by hand at inference — that is the classic train/serve skew bug. The scaler stays inside the Pipeline even for tree models so serialisation is uniform. | REVIEW |
| **ML-02** | **Always write `model_metadata.json`.** Minimum contents: `feature_set_version`, class list in encoder order, chosen candidate and hyperparameters, grouped-CV and test metrics, config hash, git SHA, UTC timestamp, and the versions of `python`, `numpy`, `scikit-learn`, `mediapipe`, `opencv-python`. | AUTO (`persist.py`) |
| **ML-03** | **`feature_set_version` guard enforced on load.** `enforce_feature_version_match: true`. If the model's version differs from `features.feature_set_version`, the loader raises `ModelVersionError` and refuses to predict. A 63-dim model fed 83-dim vectors must fail loudly, not quietly mispredict. | AUTO |
| **ML-04** | **The test set is touched exactly once**, in Phase 7, after model selection is final. Every touch is recorded in `MEMORY.md`. | REVIEW |
| **ML-05** | **Never tune on test.** Selection and hyperparameter search use grouped CV over train+val only (`cv_folds: 5`, `scoring: f1_macro`). | REVIEW |
| **ML-06** | **Never report a random-split score as the headline result.** Grouped macro-F1 is the headline. The random-split score appears only beside it, labelled "leaked baseline", with the gap quantified (`report_leakage_comparison: true`). | REVIEW + GR-03 |
| **ML-07** | **Bump `model.version` on any retrain that changes data or features.** New session added, validation rule changed, `use_extended` flipped → new version directory. Hyperparameter-only re-searches on identical data may overwrite, but must update `model_metadata.json`. | REVIEW |
| **ML-08** | Primary metric is **`macro_f1`** — macro, not weighted, so `_NEUTRAL` and rare signs count equally. Secondary: accuracy, macro precision, macro recall, balanced accuracy. Accuracy alone is never the headline. | REVIEW |
| **ML-09** | All six candidates in `training.candidates` are evaluated and reported, including the losers: `logistic_regression`, `knn`, `svm_rbf`, `random_forest`, `extra_trees`, `mlp`. The comparison table is a deliverable, not a means to an end. | REVIEW |
| **ML-10** | `mlp` is `sklearn.neural_network.MLPClassifier`. No TensorFlow, no PyTorch, no Keras. The project's stated scope is classical ML on landmark features. | DESIGN |
| **ML-11** | `class_weight: "balanced"` on every candidate that supports it. | REVIEW |
| **ML-12** | Grids stay small. On a ~4k-sample dataset an oversized grid overfits the CV folds and buys nothing. Expanding a grid requires a `MEMORY.md` note. | REVIEW |

---

## 9. Testing Rules

| ID | Rule | Class |
| --- | --- | --- |
| **TS-01** | **Every bug fix gets a regression test, written first.** The test must fail on the unfixed code. A fix with no test is not a fix; it is a coincidence. | REVIEW |
| **TS-02** | **Unit tests need neither a camera nor a trained model.** They use synthetic landmark arrays and fixtures. A unit test that opens `cv2.VideoCapture` is misclassified. | AUTO (marker discipline) |
| **TS-03** | Use the declared markers — `unit`, `integration`, `model`, `camera`, `slow`. `--strict-markers` is set, so an undeclared marker is a collection error. Adding a marker means adding it to `[tool.pytest.ini_options] markers`. | AUTO |
| **TS-04** | **Deterministic tests only.** Seed via `utils.seeding.set_global_seed()` with `reproducibility.random_seed: 42`. No dependence on wall-clock time, ordering, or unseeded randomness. A flaky test is deleted or fixed the day it flakes. | REVIEW |
| **TS-05** | Tests marked `model` skip cleanly when `models/v1/sign_classifier.joblib` is absent; tests marked `camera` skip cleanly with no webcam. A fresh clone must give a green `pytest -m unit`. | REVIEW |
| **TS-06** | **Coverage targets — whole-package `>=70%`, with `>=95%` for `src/sign_language/features/`, `preprocessing/splitter.py`, `inference/stabilizer.py` and `text/composer.py`**, the pure-logic modules where a silent numerical error is invisible in the UI but fatal to accuracy. The full per-module breakdown is the table in `TESTING.md` §10, which is authoritative. **Actual coverage: TBD — measure in Phase 12.** Note that `pyproject.toml` sets no `fail_under`, so this target is REVIEW-class, not AUTO; add `fail_under` in Phase 12 if the team wants it gated. | REVIEW |
| **TS-07** | Coverage config omits `*/ui/desktop_app.py` (GUI event loop) and `*/__init__.py`. `desktop_app.py` is therefore covered by the **manual UI test script** in Phase 12, which is a written, signed-off checklist — not an absence of testing. | REVIEW |
| **TS-08** | No network in tests. Ever. | REVIEW |
| **TS-09** | Assertions are specific. `assert result is not None` is not a test; assert the shape, the dtype, and a known value. | REVIEW |
| **TS-10** | `tests/**` is exempt from `D` and `ARG` ruff rules by `per-file-ignores`, but not from `black`, `F`, or `I`. | AUTO |

```bash
pytest -m unit                                 # fast gate, must be green on any machine
pytest -m "unit or integration"                # pre-merge gate
pytest -m model                                # after a model exists (Phase 6+)
pytest -m camera                               # local only, webcam attached
pytest --cov --cov-report=term-missing         # coverage, uses [tool.coverage] settings
```

---

## 10. Error Handling Rules

| ID | Rule | Class |
| --- | --- | --- |
| **EH-01** | All project exceptions derive from `SignLanguageError` in `src/sign_language/core/exceptions.py`. Callers can then catch one base type at the boundary. | DESIGN |
| **EH-02** | Defined subclasses: `ConfigError`, `CameraError`, `HandDetectionError`, `DatasetError`, `ValidationError`, `FeatureError`, `TrainingError`, `ModelLoadError`, `ModelVersionError`, `PredictionError`, `UIError`. Add to this list rather than raising bare `RuntimeError`/`ValueError` from library code. | REVIEW |
| **EH-03** | **Never swallow an exception silently.** `except X: pass` is banned. Handle it, log it, or re-raise it — with a reason in a comment if the handling is non-obvious. | AUTO (`ruff` `SIM105`/`B`) + REVIEW |
| **EH-04** | **User-facing messages state what happened AND what to do next.** Two clauses, no jargon, no tracebacks in the UI. | REVIEW |
| **EH-05** | **Log the full traceback at ERROR** (`logger.exception(...)` or `exc_info=True`) at the point where the exception is handled, exactly once. Do not log-and-re-raise at every level. | REVIEW |
| **EH-06** | **Resources are released via context managers.** `cv2.VideoCapture` and `mediapipe.solutions.hands.Hands` are wrapped so `release()`/`close()` runs in a `finally`, including on `KeyboardInterrupt`. A leaked camera handle means the next run cannot open the device and the user must reboot. | DESIGN |
| **EH-07** | **The app must never crash on a missing hand or a dropped frame.** Both are normal operating conditions, not errors: no hand → emit nothing, show "no hand detected"; failed `cap.read()` → increment a counter, skip the frame, and only surface `CameraError` after repeated failure inside `camera.read_timeout_ms: 2000`. | DESIGN |
| **EH-08** | Fail fast on startup-class problems (bad config, missing model, feature-version mismatch) and fail soft on frame-class problems (no hand, low confidence, one bad read). Never the reverse. | DESIGN |
| **EH-09** | Two hands in frame follows `multi_hand_policy: "highest_confidence"`: keep the highest-confidence hand, warn the user once, do not raise. Note `max_num_hands` is `2` precisely so this case is *detectable* — at `1` the branch would be unreachable. | REVIEW |

### EH-04 / EH-07 — GOOD vs BAD

```python
# BAD
if not cap.isOpened():
    raise Exception("error")                      # EH-02: bare Exception, EH-04: useless text

frame_ok, frame = cap.read()
if not frame_ok:
    sys.exit(1)                                   # EH-07: one dropped frame kills the app

try:
    landmarks = extract(frame)
except Exception:
    pass                                          # EH-03: silent swallow
```

```python
# GOOD
if not cap.isOpened():
    raise CameraError(
        f"Could not open camera index {cfg.camera.index}. "
        "Close any other app using the webcam, or set camera.index in "
        "config.local.yaml to 1 or 2. See docs/TROUBLESHOOTING.md."
    )

frame_ok, frame = cap.read()
if not frame_ok:
    self._consecutive_read_failures += 1
    if self._elapsed_ms() > cfg.camera.read_timeout_ms:
        raise CameraError(
            "Lost the camera feed. Reconnect the webcam and press Restart."
        )
    return None                                   # EH-07: skip this frame, keep running

try:
    landmarks = extract(frame)
except HandDetectionError:
    logger.exception("Landmark extraction failed on one frame; skipping")
    return None
```

---

## 11. Logging Rules

Configured once by `utils/logging_setup.py` from the `logging` block: `level: INFO`, console and
rotating file at `logs/app.log`, `max_bytes: 5242880`, `backup_count: 3`, and the format
`%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s`.

| Level | What belongs here | What must not |
| --- | --- | --- |
| `DEBUG` | Landmark coordinates, per-frame feature widths, buffer contents, per-frame confidences, resolved config values. | — |
| `INFO` | Lifecycle events only: app start/stop, config loaded, camera opened at WxH, model loaded with version and load time, session started/ended, sample counts at end of a capture, a **committed** sign. | Anything that fires per frame. |
| `WARNING` | Recoverable degradations: dropped frames, more hands than expected, rejected sample with reason, low FPS, missing optional file. | Conditions that are actually normal, such as "no hand in frame". |
| `ERROR` | Handled exceptions with full traceback, camera lost, model load failure, config validation failure. | Anything the user can fix by pressing a button, if it is not actually broken. |
| `CRITICAL` | Unused in this project. Reserved. | — |

| ID | Rule | Class |
| --- | --- | --- |
| **LG-01** | `logger = logging.getLogger(__name__)` at module top. Never `logging.info(...)` on the root logger. | REVIEW |
| **LG-02** | `logging.basicConfig` / handler setup happens **only** in `utils/logging_setup.py`, called once from `scripts/**` or `app.py`. Library modules never configure logging. | REVIEW |
| **LG-03** | **No per-frame logging at INFO.** At 30 fps one INFO line per frame is 1,800 lines/minute and rotates the 5 MB file away in minutes, destroying the record of the failure you were trying to find. Per-frame goes to DEBUG, or behind a counter that logs every N frames. | REVIEW |
| **LG-04** | **Landmark coordinates at DEBUG only.** They are biometric-adjacent data; at INFO they would be written to a file that lives in a OneDrive-synced folder. | REVIEW + SEC-04 |
| **LG-05** | **Never log a path containing a personal name at INFO.** The repo root is `C:\Users\harsh\OneDrive\ドキュメント\...`. Log paths relative to the project root, or log at DEBUG. | REVIEW |
| **LG-06** | Use `%`-style lazy formatting in log calls — `logger.debug("width %d", n)` — not f-strings. This is the one documented exception to **CS-06**: the f-string is evaluated even when the level is disabled, which matters on a per-frame DEBUG path. | REVIEW |
| **LG-07** | Log messages state facts, not moods. No `"oops"`, no emoji, no exclamation marks. | REVIEW |
| **LG-08** | `logs/` contents are gitignored except `.gitkeep`. Never commit a log file — it may contain paths and timings from a specific person's machine. | AUTO |

---

## 12. Git Rules

### 12.1 Branching

| ID | Rule | Class |
| --- | --- | --- |
| **GIT-01** | Branch names are `<type>/<short-kebab-description>`, where `<type>` is one of `feat`, `fix`, `docs`, `test`, `refactor`, `chore`. Examples: `feat/stabiliser-window`, `fix/camera-release-on-exit`, `docs/phase7-evaluation`, `test/feature-normalise-unit`, `refactor/split-dataset-builder`, `chore/lock-requirements`. | REVIEW |
| **GIT-02** | `main` is always runnable: `pytest -m unit` green, `ruff check .` clean, `black --check .` clean. | REVIEW |
| **GIT-03** | **Never force-push a shared branch**, `main` above all. Rewriting history that a teammate has pulled costs more time than any commit is worth. Force-push only your own unshared feature branch, and say so in the team chat. | REVIEW |
| **GIT-04** | One logical change per commit. "Add stabiliser + reformat 12 files + bump numpy" is three commits. | REVIEW |
| **GIT-05** | Never commit: `.env`, datasets (`data/raw|interim|processed|features`), model binaries (`*.joblib`, `*.pkl`), caches (`__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`), `.venv/`, `config.local.yaml`, log files, personal data of any kind. All are covered by `.gitignore`; verify with `git status --short` before every commit. | AUTO + REVIEW |
| **GIT-06** | `git add -A` is discouraged. Stage deliberately (`git add -p` or explicit paths) so a stray `.npz` never slips in. | REVIEW |

### 12.2 Conventional Commits

Format: `<type>(<scope>): <imperative summary>` — summary in lower case, no trailing period,
≤72 characters. Body (optional, wrapped at 72) explains **why**.

| Type | Use for |
| --- | --- |
| `feat` | New capability visible to a user or another module |
| `fix` | Bug fix in existing behaviour |
| `docs` | Documentation only |
| `test` | Adding or changing tests only |
| `refactor` | Behaviour-preserving restructuring |
| `perf` | Change made specifically for speed or memory |
| `chore` | Tooling, dependencies, config plumbing, gitignore |
| `data` | Dataset manifests and metadata (never the data itself) |
| `build` | Packaging, `pyproject.toml`, lock files |
| `revert` | Reverting a previous commit |

Scopes match the module map: `config`, `core`, `collect`, `features`, `preprocess`, `training`,
`inference`, `text`, `ui`, `utils`, `scripts`, `docs`, `deps`.

Ten real examples for this project's phases:

```text
chore(deps): pin numpy below 2.0 for mediapipe abi compatibility
build(deps): add requirements.lock.txt from verified phase 1 install
feat(config): add typed schema loader with startup validation
feat(collect): add session-aware capture script with countdown and validator
data(collect): add session manifests for signer1 sessions s01 and s02
feat(features): add wrist-relative translation and scale normalisation
feat(preprocess): split dataset with stratifiedgroupkfold on session_id
feat(training): compare six candidates on grouped cv macro f1
docs(training): record phase 6 candidate comparison and winner in MEMORY.md
fix(inference): release camera handle on keyboard interrupt
```

### 12.3 Pull request and review checklist

| ID | Rule | Class |
| --- | --- | --- |
| **GIT-07** | Every change to `src/` reaches `main` through a PR reviewed by at least one other team member. Three people, three reviewers available — no excuses. | REVIEW |
| **GIT-08** | PR description states: what changed, why, which phase in `PHASES.md` it advances, how it was verified (paste the command and its output), and which rules were consciously bent. | REVIEW |
| **GIT-09** | PRs that touch `config.yaml` must show the matching `config/schema.py` change in the same diff (**CF-06**). | REVIEW |
| **GIT-10** | PRs that change reported numbers must include the regenerated artifact under `reports/` and the command that produced it (**GR-06**). | REVIEW |
| **GIT-11** | No PR merges with a failing `ruff`, `black --check`, `mypy`, or `pytest -m unit`. | AUTO |

### 12.4 OneDrive warning (this repository specifically)

This repo lives inside a OneDrive-synced folder. OneDrive is not git-aware and will happily
interfere with git's own file operations.

| ID | Rule | Class |
| --- | --- | --- |
| **GIT-12** | **Pause OneDrive syncing before any git operation that rewrites many files** — `checkout`, `pull`, `merge`, `rebase`, `stash`. OneDrive can hold a lock on a file mid-operation and produce `error: unable to unlink` or a half-applied working tree. Right-click the OneDrive tray icon → *Pause syncing* → 2 hours. | REVIEW |
| **GIT-13** | **Watch for duplicate files.** When OneDrive detects a conflicting edit it silently creates `predictor (2).py`, `config (2).yaml`, or `MEMORY-DESKTOP-ABC123.md`. These are invisible to a quick glance, get picked up by `git add -A` (see **GIT-06**), and can shadow the real module. Before each commit run: | REVIEW |

```bash
git status --short
git ls-files --others --exclude-standard
```

```bash
# Find OneDrive conflict duplicates anywhere in the repo (Git Bash / PowerShell bash)
find . -name "* (2)*" -not -path "./.git/*"
find . -name "*-DESKTOP-*" -not -path "./.git/*"
```

| ID | Rule | Class |
| --- | --- | --- |
| **GIT-14** | Do not put `.venv/` inside the synced folder if avoidable — syncing tens of thousands of small files is slow and a frequent source of locks. If it must live there, exclude it in the OneDrive settings as well as in `.gitignore`. | REVIEW |
| **GIT-15** | `data/` must be excluded from OneDrive sync or relocated outside the synced tree. See **SEC-08** — this is a privacy rule, not just a performance one. | DESIGN |

---

## 13. Documentation Rules

| ID | Rule | Class |
| --- | --- | --- |
| **DOC-01** | **Docs are updated in the SAME commit as the code they describe.** A behaviour change with stale docs is an incomplete change. Reviewers reject on this. | REVIEW |
| **DOC-02** | **`MEMORY.md` is updated after every phase completion and every significant decision** — a tuned threshold, a rejected approach, a bug that cost more than an hour, a number that got measured. Each entry: date, author, what, why, what it supersedes. | REVIEW |
| **DOC-03** | **`TBD` is mandatory for unmeasured values**, written as `TBD — measure in Phase N` so the reader knows when it lands. Never a placeholder that looks like a measurement (**GR-03**, **GR-07**). | REVIEW |
| **DOC-04** | Google-style docstrings on all public functions, classes, and modules, with `Args`, `Returns`, and `Raises` where applicable. Machine-checked by ruff's `D` rules. | AUTO |
| **DOC-05** | **ADRs are appended to `ARCHITECTURE.md`** for any architectural change: context, decision, alternatives considered, consequences, date, author. Numbered `ADR-001`, `ADR-002`, … and never edited after acceptance — superseded ADRs get a new ADR pointing back. | REVIEW |
| **DOC-06** | Numbers appear in exactly one authoritative document and are cross-referenced elsewhere. Tunables live in `config.yaml`; measured results live in `reports/metrics/` and are quoted, not retyped, into prose. | REVIEW |
| **DOC-07** | Cross-reference sibling docs by relative filename (`DATASET.md`, `PHASES.md`), never by absolute path or bare title. | REVIEW |
| **DOC-08** | Every `[TUNE:Phase-N]` value that changes gets a `MEMORY.md` entry naming the experiment (**CF-08**). | REVIEW |
| **DOC-09** | `README.md` must always contain a working quickstart. If a command in it fails on a clean clone, that is a P1 bug. | REVIEW |

---

## 14. Security and Privacy Rules

The data subjects are the three of us. That makes this an ethics obligation, not a compliance
checkbox.

| ID | Rule | Class |
| --- | --- | --- |
| **SEC-01** | **The camera is opened only on explicit user action** — a CLI invocation or a button press — and **always released**, via the context manager in **EH-06**. No camera access at import time, at config load, or on app launch before the user asks. | DESIGN |
| **SEC-02** | **No network calls at runtime.** The application performs no HTTP requests, no model downloads, no update checks. MediaPipe's bundled model files ship with the wheel. If a code path needs the network, it is out of scope. | DESIGN |
| **SEC-03** | **No telemetry, no analytics, no crash reporting.** Nothing leaves the machine. | DESIGN |
| **SEC-04** | **Frames are never written to disk** except the consented audit trail: `save_debug_images: true` with `debug_image_every_n: 20` writes one JPEG per 20 samples to `data/raw/`. That is a deliberate, disclosed, sampled trail for the report — not silent recording. `data/raw/**` is gitignored. | REVIEW |
| **SEC-05** | **The dataset stays local.** It is never uploaded, never attached to an email, never put in a shared drive, never included in a submission ZIP. What ships is `data/metadata/**`. | REVIEW |
| **SEC-06** | **Consent is recorded.** Each signer's written consent to be recorded, and to the resulting landmarks being used in this project, is noted in `PRD.md` with a date. Signers are referred to as `signer1`/`signer2`/`signer3` in all filenames, manifests, and logs — the mapping to real names lives in `PRD.md` only. | REVIEW |
| **SEC-07** | Any signer may withdraw. On withdrawal, their sessions are deleted from `data/`, the dataset is rebuilt, and the model is retrained with a version bump (**ML-07**). No argument, no negotiation. | DESIGN |
| **SEC-08** | **OneDrive caveat.** This repo sits inside a OneDrive-synced folder, so by default `data/` — including debug JPEGs of our faces and hands — **would be uploaded to Microsoft's cloud**. Before Phase 3 collection starts, either exclude `data/` from OneDrive sync or relocate it outside the synced tree and point `paths.*` at the new location via `config.local.yaml`. This is a **Phase 3 blocker**. | DESIGN |
| **SEC-09** | **Never unpickle an untrusted `.joblib`.** `joblib.load` executes arbitrary code during deserialisation. Only load artifacts this repo produced, from `models/`. Never load a model file received over chat, email, or download. | REVIEW |
| **SEC-10** | No secrets in the repo. There are none today; if one ever appears it goes in `.env` (gitignored) with a committed `.env.example` listing key names only. | AUTO |
| **SEC-11** | No PII in error messages or the UI. Show `data/interim/HELLO/...`, never `C:\Users\<name>\OneDrive\...`. | REVIEW |

---

## 15. Dependency Rules

| ID | Rule | Class |
| --- | --- | --- |
| **DEP-01** | **`requirements.txt` is runtime; `requirements-dev.txt` is development.** A package needed only to test, lint, type-check, or explore never appears in `requirements.txt`. Install both with `pip install -r requirements.txt -r requirements-dev.txt`. | REVIEW |
| **DEP-02** | **Justify every new dependency in the PR:** what it does, why the standard library or an existing dependency cannot, its install size, and its Python-version constraints. Adding a package to save ten lines is usually a bad trade. `seaborn` is the model of an acceptable justification — it is documented in `requirements.txt` as being for the 13×13 confusion-matrix heatmap only. | REVIEW |
| **DEP-03** | **The `numpy>=1.24,<2.0` pin is load-bearing and must not be relaxed casually.** MediaPipe 0.10.x is compiled against the numpy 1.x C ABI; numpy 2.x produces `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x` at import. The opencv wheels of this era are also 1.x-ABI. Relaxing this pin requires verifying `import mediapipe, cv2` on a clean environment first, and a `MEMORY.md` entry. | DESIGN |
| **DEP-04** | **Never install two opencv variants in one environment.** `opencv-python` only — not `opencv-python-headless` (we need GUI windows for the Phase 8 HUD) and not `opencv-contrib-python` (no contrib modules are used). Mixing them breaks `import cv2` in ways that look like unrelated bugs. Check with `pip list | grep -i opencv` — exactly one line is correct. | REVIEW |
| **DEP-05** | **Freeze after a verified install:** `pip freeze > requirements.lock.txt`, committed. `requirements.txt` expresses intent with ranges; the lock file is the reproducible artifact. As of Phase 0, **the pins in `requirements.txt` have NOT been resolved against PyPI on the team's machine** — that is exactly what Phase 1 verifies. | REVIEW |
| **DEP-06** | Python **3.10 or 3.11** only, matching `requires-python = ">=3.10,<3.12"`. 3.13 has no MediaPipe wheels; 3.12 support is patchy. Verify with `python --version` before anything else. | REVIEW |
| **DEP-07** | `mediapipe>=0.10.9,<0.10.15` is pinned tightly because the legacy `mediapipe.solutions.hands` API this project uses is deprecated in newer builds. Do not widen it to chase a newer release. | DESIGN |
| **DEP-08** | `tkinter` is **not** a pip package — it ships with the CPython Windows installer. If `import tkinter` fails, re-run the installer with *tcl/tk and IDLE* enabled. It is therefore absent from `requirements.txt` by design, while `customtkinter` and `Pillow` (needed to convert BGR frames for Tk display) are present. | REVIEW |
| **DEP-09** | `pyproject.toml` deliberately declares **no** `[project.dependencies]`, to avoid competing with `requirements.txt`. Consequence: `pip install -e .` installs the package but **not** its runtime dependencies. Always install requirements explicitly first. | REVIEW |
| **DEP-10** | Dependency bumps are their own commit, typed `chore(deps)` or `build(deps)`, with the verifying import check pasted into the PR. | REVIEW |

---

## 16. Reproducibility Rules

| ID | Rule | Class |
| --- | --- | --- |
| **RP-01** | **Seed 42 everywhere**, from `reproducibility.random_seed`, applied through a single helper `utils/seeding.py::set_global_seed()` covering `random`, `numpy.random`, `PYTHONHASHSEED`, and every sklearn `random_state`. Never call `np.random.seed` directly in a module. | REVIEW |
| **RP-02** | Every script that splits, shuffles, samples, or searches calls `set_global_seed()` before doing so. `reproducibility.deterministic: true` is an assertion the code must honour. | REVIEW |
| **RP-03** | **Every reported number is regenerable by a named script**, and the script name is recorded next to the number. Mapping: dataset counts → `scripts/build_dataset.py`; candidate comparison → `scripts/train_model.py`; test metrics, confusion matrix, leakage comparison, leave-one-signer-out → `scripts/evaluate_model.py`; fps and latency → `scripts/benchmark.py`. | REVIEW |
| **RP-04** | **A config hash is recorded with every artifact.** A SHA-256 over the merged, resolved config goes into `model_metadata.json` and into each metrics JSON, so an artifact can be traced to the exact settings that produced it. | REVIEW |
| **RP-05** | **Library versions are recorded in `model_metadata.json`**: `python`, `numpy`, `scikit-learn`, `mediapipe`, `opencv-python`, `joblib`. A model that scores differently after an environment change must be diagnosable. | AUTO (`persist.py`) |
| **RP-06** | **No manual steps in the results path.** No "then open the notebook and run cell 7", no spreadsheet arithmetic, no screenshot-and-retype. `build_dataset.py → train_model.py → evaluate_model.py` reproduces every reported result from `data/interim/`. | DESIGN |
| **RP-07** | Artifacts are written, not printed. Metrics go to `reports/metrics/*.json`, figures to `reports/figures/*.png`. Console output is a convenience summary and is never the record. | REVIEW |
| **RP-08** | Any known source of non-determinism (thread count, BLAS version, `n_jobs: -1`) is documented next to the affected number. `n_jobs: -1` may reorder floating-point reductions; if a metric moves in the last decimal place, that is why. | REVIEW |

---

## 17. Code Review Checklist

Reviewer ticks every line. An unticked line is a blocking comment citing the rule ID.

**Correctness and scope**

- [ ] The change does what the PR says, and nothing else (**GIT-04**).
- [ ] It advances a phase named in `PHASES.md`.
- [ ] Edge cases handled: no hand in frame, dropped frame, empty buffer, single-class input.

**Golden Rules**

- [ ] No hard-coded tunables; every new number is in `config.yaml` **and** the schema (**GR-01**, **CF-06**).
- [ ] No random split feeding a reported number (**GR-02**).
- [ ] No metric present that a script in this repo cannot regenerate (**GR-03**, **GR-06**).
- [ ] `git status --short` shows no dataset, model binary, cache, or OneDrive duplicate (**GR-04**, **GIT-13**).
- [ ] Nothing under `features/`, `preprocessing/`, `training/`, `inference/`, `text/` imports `ui/`, `tkinter`, or `customtkinter` (**GR-05**).
- [ ] Unmeasured values written as `TBD — measure in Phase N` (**GR-07**).

**Code quality**

- [ ] `ruff check .` clean; `black --check .` clean; `mypy` clean (**CS-01**–**CS-05**).
- [ ] Type hints and Google docstrings on all new public functions.
- [ ] No `print()` in `src/` (**CS-12**); no bare `except` (**CS-08**); no mutable defaults (**CS-09**).
- [ ] `pathlib`, not `os.path` (**CS-07**).
- [ ] No module pushed past ~400 lines (**PS-03**).

**Errors and logging**

- [ ] New exceptions derive from `SignLanguageError` (**EH-01**).
- [ ] User-facing messages say what happened and what to do next (**EH-04**).
- [ ] Camera and MediaPipe resources released in `finally`/context manager (**EH-06**).
- [ ] No per-frame INFO logging (**LG-03**); no landmarks above DEBUG (**LG-04**); no personal paths at INFO (**LG-05**).

**Tests**

- [ ] New behaviour has tests; a bug fix has a regression test that fails without the fix (**TS-01**).
- [ ] Correct marker applied and declared (**TS-03**).
- [ ] `pytest -m unit` passes with no camera and no model (**TS-02**, **TS-05**).
- [ ] Tests are seeded and deterministic (**TS-04**).

**Data and models**

- [ ] `session_id` and `signer_id` preserved end to end (**DS-01**).
- [ ] `feature_set_version` consistent and guarded (**DS-03**, **ML-03**).
- [ ] Whole `Pipeline` serialised; `model_metadata.json` written (**ML-01**, **ML-02**).
- [ ] Test set not touched for tuning (**ML-04**, **ML-05**).

**Docs and privacy**

- [ ] Docs updated in this same commit (**DOC-01**); `MEMORY.md` entry added if a decision was made (**DOC-02**).
- [ ] ADR appended for architectural change (**DOC-05**).
- [ ] No frames, no PII, no personal paths introduced (**SEC-04**, **SEC-11**).

---

## 18. Definition of Done

### 18.1 A task is Done when

- [ ] Code written and self-reviewed against §17.
- [ ] `ruff check .`, `black --check .`, `mypy` all clean.
- [ ] `pytest -m unit` green; new tests added and passing.
- [ ] No hard-coded tunables introduced.
- [ ] Docstrings on new public functions.
- [ ] Docs touched in the same commit if behaviour or interface changed.
- [ ] Conventional Commit message, one logical change.
- [ ] PR reviewed and approved by another team member.
- [ ] Merged to `main` with `main` still runnable.

### 18.2 A phase is Done when

- [ ] Every completion-criteria checkbox for that phase in `PHASES.md` is ticked, with evidence.
- [ ] The phase's verification commands run clean, and their output is pasted into the PR.
- [ ] Artifacts exist at the exact paths the phase promised.
- [ ] Every number the phase was supposed to measure is measured — no leftover `TBD` that this phase owned.
- [ ] `MEMORY.md` updated with the phase's decisions, measurements, and surprises (**DOC-02**).
- [ ] `PHASES.md` progress tracker updated: status, owner, dates, notes.
- [ ] Any `[TUNE:Phase-N]` value this phase owned is either tuned with a recorded justification or explicitly deferred with a reason.
- [ ] `config.yaml` `project.version` bumped if the phase changed observable behaviour.

### 18.3 The project is Done when

- [ ] All 13 classes collected to at least `samples_per_class_minimum: 200`, ideally 330, across at least 4 sessions each — 3 signers represented.
- [ ] All six candidate models trained and compared on grouped CV macro-F1; winner selected and justified.
- [ ] Test set evaluated **once**; grouped macro-F1 reported as the headline, random-split score reported beside it as the quantified leakage.
- [ ] Leave-one-signer-out results reported honestly, including if they are poor.
- [ ] `models/v1/model_metadata.json` committed, complete, and consistent with `reports/metrics/`.
- [ ] Real-time app runs end to end: camera → landmarks → features → prediction → stabilisation → text.
- [ ] Measured performance meets `min_fps: 15`, `max_inference_latency_ms: 50`, `max_model_load_ms: 2000`, or the shortfall is documented with the reason.
- [ ] `pytest` green across `unit`, `integration`, and `model`; manual UI checklist signed off.
- [ ] Coverage for `features/` and `inference/` at or above the **TS-06** target, actual figure recorded.
- [ ] All documentation complete and internally consistent; **zero `TBD`s remain that a completed phase owned**.
- [ ] Demo rehearsed end to end on the presentation machine, with the fallback recording prepared.
- [ ] Viva question bank in `PHASES.md` Phase 16 walked through by all three members.
- [ ] Limitations stated plainly: static single-hand signs only, 13 classes, three signers, controlled lighting.

---

## 19. Revision History

| Version | Date | Author | Change |
| --- | --- | --- | --- |
| 1.0 | 2026-08-26 | HT | Initial standards issued at Phase 0 completion. Aligned to `config.yaml` and `pyproject.toml` as of this date. |

**Amending this document:** open a PR titled `docs(rules): <change>`, get agreement from all three
team members, add the row above, and never renumber an existing rule (§1.3).

**Known open items recorded at issue time**

| Item | Rule | Action |
| --- | --- | --- |
| ~~`sessions_per_class_minimum: 4` × `samples_per_session_cap: 80` = 320 < `samples_per_class_target: 330`.~~ | **DS-09** | **RESOLVED in Phase 0** — cap raised to 90 (4 x 90 = 360). Invariant now enforced at startup. |
| `performance.measured.*` parses as the string `"TBD"`, not a number. | **CF-09** | Schema must accept the sentinel; Phase 13 overwrites with floats. |
| `project.version` exists as `0.1.0-alpha` in `config.yaml` and `0.1.0a0` in `pyproject.toml`. | **DOC-06** | Pick one authority in Phase 1; the other should reference it. |
| No `feature_set_version` string is defined for the 83-dim extended set. | **DS-03** | Define it in Phase 5 before flipping `features.use_extended`. |
| `ui.mode` defaults to `desktop`, but the desktop app does not exist until Phase 11. | **CF-04** | Run with `hud` via `config.local.yaml` during Phases 8–10. |
| No `fail_under` in `[tool.coverage.report]`, so **TS-06** is not machine-enforced. | **TS-06** | Decide in Phase 12 whether to gate it. |
