# TESTING.md — Test Strategy and Plan

**Project:** Real-Time Sign Language to Text Conversion System
**Document version:** 1.0
**Status:** Strategy defined (Phase 0). **No tests written yet** — implemented alongside each phase and consolidated in Phase 12.
**Last updated:** 2026-08-26

> **Honesty statement.** Every results table in this document is empty. No test has been executed on
> this project. Coverage numbers, pass counts and timings all read `TBD` and will be filled in with
> real measured output from real runs. Nothing here is a claim about the system's current quality.

---

## 1. What we are trying to prove

Testing a machine learning system means proving two very different kinds of thing, and confusing them
is a common failure.

The first kind is ordinary software correctness: does the normaliser subtract the wrist landmark, does
the splitter keep sessions apart, does the text composer append a word. These are deterministic and
have exactly one right answer, so they are tested the way any code is tested — assert the expected
output.

The second kind is model behaviour: does the classifier reach acceptable macro-F1, does inference keep
up with the camera. These have no single right answer and depend on data that does not exist yet. They
are tested against **thresholds declared in advance** (`config.yaml` `performance.targets`, `PRD.md`
ACC-01…ACC-06) so that we cannot quietly move the goalposts after seeing the numbers. Declaring the
threshold before the measurement is the entire point.

A third category is easy to overlook: **structural properties that protect scientific validity**. That
the scaler is fitted inside the cross-validation fold and not before it. That train and test session
IDs are disjoint. That the feature-version guard actually refuses a mismatched model. These are unit
tests in form, but what they protect is the credibility of the results, so they are treated as the
highest-priority tests in the suite (§4).

---

## 2. Test taxonomy

Markers are declared in `pyproject.toml` `[tool.pytest.ini_options] markers`.

| Marker | Scope | Needs a camera? | Needs a trained model? | Typical runtime |
|---|---|---|---|---|
| `unit` | One function or class, no I/O | No | No | < 1 ms each |
| `integration` | Two or more modules together | No | No | < 1 s each |
| `model` | Trained artifacts, metric gates | No | **Yes** | seconds |
| `camera` | Real capture device | **Yes** | No | seconds |
| `slow` | Full training runs, benchmarks | No | Varies | minutes |

```bash
pytest -m unit                      # fast feedback loop, runs everywhere
pytest -m "unit or integration"     # the CI-equivalent default
pytest -m model                     # only after Phase 7 produces artifacts
pytest -m camera                    # only on a machine with a webcam
pytest                              # everything; skips what is unavailable
```

`camera` and `model` tests must **skip, not fail**, when their prerequisite is absent. A test suite
that goes red on a machine without a webcam trains people to ignore red, which is worse than having no
test.

```python
pytest.importorskip("cv2")
if not MODEL_PATH.exists():
    pytest.skip("no trained model — run scripts/train_model.py", allow_module_level=True)
```

---

## 3. Fixtures — `tests/conftest.py`

Real webcam frames and real trained models are unavailable during unit testing, so the suite is built
on **synthetic landmark data**. This is legitimate because the code under test is arithmetic on arrays;
it does not care where the arrays came from.

| Fixture | Provides | Why |
|---|---|---|
| `config` | The real `config.yaml`, loaded and validated | Catches config/code drift on every run |
| `synthetic_hand` | Deterministic `(21, 3)` float32 array, seed 42 | Reproducible input for feature tests |
| `translated_hand` | `synthetic_hand` + a constant offset | Proves translation invariance |
| `scaled_hand` | `synthetic_hand` × a constant factor | Proves scale invariance |
| `rotated_hand` | `synthetic_hand` rotated about the wrist | Proves rotation invariance is **absent** |
| `mirrored_hand` | `synthetic_hand` with x negated | Left/right handling |
| `degenerate_hand` | All 21 points identical | Divide-by-zero guard |
| `synthetic_dataset` | Small labelled frame with `session_id`, `signer_id`, 13 classes | Splitter and pipeline tests without real data |
| `fake_frame` | 720×1280×3 `uint8` array | Camera-free frame-path tests |
| `tmp_model_dir` | `tmp_path` with fake artifacts + metadata | Loader and version-guard tests |

Nothing in `tests/` may read from `data/` or `models/` outside a fixture. Tests that depend on the real
dataset are not tests, they are experiments, and they belong in `reports/`.

---

## 4. Priority-1 tests — the ones that protect the science

If everything else were deleted, these would remain. Each corresponds to a specific way the project
could produce impressive numbers that mean nothing.

### 4.1 Session groups do not leak across the split

```python
@pytest.mark.unit
def test_split_shares_no_session_between_train_and_test(synthetic_dataset, config):
    train, test = make_test_split(synthetic_dataset, config)
    assert set(train["session_id"]) & set(test["session_id"]) == set()
```

**What it prevents.** Near-duplicate frames from one recording appearing on both sides of the split,
which inflates accuracy without improving generalisation (`DATASET.md` §9). This is the single most
important assertion in the project. Extended to every CV fold, not just the outer split.

### 4.2 The scaler is fitted inside the pipeline, never before the split

```python
@pytest.mark.unit
def test_scaler_is_a_pipeline_step_not_a_preprocessing_step():
    pipe = build_candidate("svm_rbf", config)
    assert isinstance(pipe, Pipeline)
    assert "scaler" in dict(pipe.named_steps)
```

**What it prevents.** Preprocessing leakage — test-set statistics bleeding into training through a
globally-fitted `StandardScaler`. Structural rather than numerical, which is why it can be asserted
cheaply and reliably.

### 4.3 The feature-version guard actually refuses a mismatch

```python
@pytest.mark.unit
def test_load_rejects_metadata_with_wrong_feature_version(tmp_model_dir, config):
    write_metadata(tmp_model_dir, feature_set_version="v0-legacy-42")
    with pytest.raises(FeatureVersionMismatchError):
        SignPredictor.load(tmp_model_dir, config)
```

**What it prevents.** A model trained on one feature layout being served features in another order.
The failure mode is *confident nonsense rather than an exception* — the worst possible kind, because
nothing looks broken (`MODEL.md` §7.4).

### 4.4 Feature dimensionality is exactly what config declares

```python
@pytest.mark.unit
def test_feature_vector_dimension_matches_config(synthetic_hand, config):
    v = FeatureExtractor(config).extract(synthetic_hand)
    assert v.shape == (config.features.base_dim,)   # 63
```

### 4.5 Train and serve use the same extractor

```python
@pytest.mark.integration
def test_training_and_inference_import_the_same_extractor():
    import sign_language.preprocessing.dataset_builder as b
    import sign_language.inference.pipeline as i
    assert b.FeatureExtractor is i.FeatureExtractor
```

**What it prevents.** Train/serve skew from a duplicated feature implementation drifting out of sync —
the classic reason a model scores 0.95 offline and behaves badly live.

---

## 5. Unit tests by module

### 5.1 `config/`

Valid config loads; unknown key rejected; missing required key names the key; `confidence_threshold`
outside [0, 1] rejected; `min_votes_in_window` > `window_size` rejected; `samples_per_session_cap` ×
`sessions_per_class_minimum` ≥ `samples_per_class_target` asserted (90 × 4 = 360 ≥ 330); the 13 declared
classes include `neutral_class`; `base_dim` == `num_landmarks` × `coords_per_landmark`.

One further check that looks trivial and is not:

```python
@pytest.mark.unit
def test_every_class_name_loads_as_a_string(config):
    """YAML 1.1 resolves bare YES / NO / ON / OFF to booleans.

    Audit defect 12: unquoted, classes[3] was True and classes[4] was False,
    with no error at load time. Keep the quotes in config.yaml.
    """
    assert all(isinstance(c, str) for c in config.dataset.classes)
```

The arithmetic and type checks are unusual for a config test and are there because these exact
inconsistencies were found in the Phase 0 audit. Once found, a defect should become a test. The type
check in particular is the one that could not have been found by reading the file — only by loading it.

### 5.2 `features/normalizer.py`

| Test | Assertion |
|---|---|
| Translation invariance | `f(hand) == f(hand + offset)` within `1e-6` |
| Scale invariance | `f(hand) == f(hand * k)` within `1e-6` |
| Wrist becomes origin | landmark 0 == `(0, 0, 0)` exactly |
| Max distance normalised | `max(‖p‖) == 1.0` within `1e-6` |
| **Rotation invariance is ABSENT** | `f(hand) != f(rotate(hand))` |
| Degenerate hand | no `ZeroDivisionError`, no `NaN` |
| Left mirrored to right | `f(left) ≈ f(mirror(left))` |
| Dtype preserved | output is `float32` |
| Input not mutated | caller's array unchanged |

The rotation test asserts that a transformation **does not** hold, which looks like a mistake until you
know the reason: PLEASE and THANK_YOU differ only in palm orientation, so rotation-normalising would
make them mathematically identical (`MODEL.md` §5.5). Someone will eventually "improve" the normaliser
by adding rotation invariance; this test is the tripwire, and its docstring must explain why.

"Input not mutated" is included because in-place NumPy modification of a caller's array is a real and
very confusing bug class — the saved dataset would be silently altered.

### 5.3 `core/hand_detector.py`

Zero hands → `None`; one hand → that hand; two hands → the higher-confidence one under
`multi_hand_policy: "highest_confidence"`; the returned array is `(21, 3)` `float32`; handedness label
is propagated; malformed MediaPipe output raises `DetectionError` rather than `IndexError`.

### 5.4 `data_collection/validator.py`

Complete hand accepted; missing landmark rejected; coordinate outside `[-0.5, 1.5]` rejected; bbox
below `0.04` rejected (too far); above `0.90` rejected (too close); cosine similarity above `0.9995`
flagged as a duplicate; each rejection carries a machine-readable reason code so collection statistics
can be summarised per session.

### 5.5 `preprocessing/splitter.py`

Beyond §4.1: every fold is group-disjoint; all 13 classes appear in train; stratification holds
approximately given grouping (exact stratification is impossible with groups — the test asserts a
tolerance, not equality, which is the honest formulation); `random_seed: 42` gives byte-identical
splits across runs; a class with too few sessions raises `DatasetError` with a clear message instead of
producing a silently degenerate split.

### 5.6 `inference/stabilizer.py`

This module has the most behavioural surface in the project and the most ways to be subtly wrong.

| Test | Expected |
|---|---|
| 7 of last 10 above threshold | commit |
| 6 of last 10 | no commit |
| 7 non-consecutive with one gap | **commit** — a count, not a run |
| below-threshold frames | never counted as votes |
| same sign held after commit | **no second commit** (no `HELLO HELLO HELLO`) |
| repeat after passing through `_NEUTRAL` | commit |
| repeat without neutral | no commit |
| within `cooldown_seconds: 1.2` | no commit |
| buffer entries older than `2.0 s` | evicted |
| `_NEUTRAL` itself | never emitted as text |
| reset() | clears all state |

The third row is the design decision worth testing explicitly: one dropped frame costs one vote rather
than resetting all accumulated progress (`MODEL.md` §9.3). The fifth row is the requirement the user
stated in the brief in exactly those terms.

Time is injected as a parameter or a fake clock — never `time.sleep`. A suite that sleeps for cooldowns
becomes a suite nobody runs.

### 5.7 `text/composer.py`

Append; word-level backspace on an empty buffer is a no-op (not an exception); clear; underscores
stripped for display while stored labels stay canonical; `max_length: 500` trims from the front;
separator is `" "` from config, not a literal.

### 5.8 `utils/`

`paths.py` derives `models/v1` from `model.version` and creates directories idempotently; `seeding.py`
makes two runs produce identical arrays; every custom exception inherits `SignLanguageError`, so a
caller can catch the base class and be sure of catching everything the project raises.

---

## 6. Integration tests

| Test | Path exercised | Prerequisite |
|---|---|---|
| Landmarks → features → prediction | `features` + `inference` | fake model |
| Session files → built dataset | `data_collection` + `preprocessing` | synthetic `.npz` |
| Dataset → split → train → evaluate on a tiny synthetic set | full training path | none |
| Frame → detector → features → stabiliser → text | full inference path | fake frame, fake model |
| `ui/` is not imported by `inference/` | layering (goal G-08) | none |
| Config change propagates | e.g. `confidence_threshold` alters behaviour | none |

The tiny-synthetic-set training test is valuable out of proportion to its size: it runs the entire
training script end to end in seconds, so a crash in evaluation or artifact-writing is caught
immediately instead of forty minutes into a real grid search.

The layering test is a plain import check, and it is the only thing that will actually stop the
separation of UI from ML from eroding under deadline pressure.

---

## 7. Model tests — gated on real thresholds

Run only after Phase 7. Thresholds come from `PRD.md` ACC-01…ACC-06 and were fixed before any data
existed.

| ID | Gate | Threshold | Result |
|---|---|---|---|
| ACC-01 | Target macro-F1 (grouped CV) | ≥ 0.85 | TBD — Phase 7 |
| ACC-02 | Minimum acceptable macro-F1 | ≥ 0.75 | TBD — Phase 7 |
| ACC-03 | Per-class recall, every class | ≥ 0.70 | TBD — Phase 7 |
| ACC-04 | `_NEUTRAL` recall | ≥ 0.90 | TBD — Phase 7 |
| ACC-05 | Held-out test macro-F1 | ≥ 0.75 | TBD — Phase 7 |
| ACC-06 | LOSO macro-F1 | reported, not gated | TBD — Phase 7 |

ACC-06 is deliberately reported rather than gated. With three signers, leave-one-signer-out is
extremely noisy, and gating on a noisy statistic would either block progress for no reason or push us
toward tuning until the noise happened to favour us. Reporting it honestly is the better science.

Also checked: `model_metadata.json` exists and contains every required field; a metadata `TBD` in a
metric field is a **failure** at Phase 7 (the whole point of the placeholder is that it must be
replaced by a real number); the number of classes in the label encoder equals 13.

---

## 8. Performance tests

Measured, not asserted from imagination. Thresholds from `config.yaml` `performance.targets`.

| Metric | Target | Measured |
|---|---|---|
| End-to-end FPS | ≥ 15 | TBD — Phase 13 |
| Inference latency (feature → label), p50 | — | TBD — Phase 13 |
| Inference latency, p95 | ≤ 50 ms | TBD — Phase 13 |
| MediaPipe detection per frame | — | TBD — Phase 13 |
| Model load time | ≤ 2000 ms | TBD — Phase 13 |
| Memory (RSS) steady state | — | TBD — Phase 13 |
| Time from sign held to text appearing | — | TBD — Phase 13 |

`scripts/benchmark.py` writes these to `reports/metrics/`. Measurements are stated **with the hardware
they were taken on** — a latency number without a machine attached to it is not a result. p95 is gated
rather than the mean because a smooth-feeling real-time UI is determined by its worst frames, not its
average ones.

---

## 9. Manual test checklist

Some things cannot be automated: whether the video looks right, whether the UI feels responsive,
whether a stranger can operate it. Executed in Phase 14 and recorded with pass/fail plus notes.

**Camera and detection.** Camera opens within 3 s · feed is mirrored · skeleton tracks the hand · two
hands shows the warning and dims the ignored hand · covering the camera degrades gracefully · works in
dim light (note the degradation honestly) · stopping releases the device and the LED goes out.

**Recognition.** Each of the 12 signs recognised by the person who collected it · each attempted by
someone who did **not** collect it (expected to be worse — record it) · resting hand yields `_NEUTRAL`
and produces no text · a held sign produces exactly one word · the same sign repeats only after
returning to neutral.

**UI.** Every button works · every keyboard shortcut works · text wraps and scrolls · Clear confirms
above 5 words · window resizes without breaking layout · closing the window releases the camera.

**Error states.** Launch with no model → guidance screen, not a crash · unplug the camera mid-session →
error shown, text preserved · corrupt `config.yaml` → the offending key is named · run with the camera
already held by another app → the "in use" message appears.

**Presentation.** Readable from ~3 m · readable on a projector · no personal data on screen · the
disclaimer is visible.

---

## 10. Coverage

Target ≥ 70% overall on `src/sign_language/`, with `features/`, `preprocessing/splitter.py`,
`inference/stabilizer.py` and `text/composer.py` above 90% since they are pure logic. `ui/desktop_app.py`
is omitted from coverage in `pyproject.toml` — GUI event loops cannot be meaningfully unit-tested, and
pretending otherwise produces tests that assert widgets exist rather than that anything works.

```bash
pytest -m "unit or integration" --cov=sign_language --cov-report=term-missing --cov-report=html
```

| Module group | Target | Actual |
|---|---|---|
| `config/` | ≥ 85% | TBD — Phase 12 |
| `features/` | ≥ 95% | TBD — Phase 12 |
| `preprocessing/` | ≥ 90% | TBD — Phase 12 |
| `inference/stabilizer.py` | ≥ 95% | TBD — Phase 12 |
| `text/` | ≥ 95% | TBD — Phase 12 |
| `core/` | ≥ 70% | TBD — Phase 12 |
| `training/` | ≥ 60% | TBD — Phase 12 |
| **Overall** | **≥ 70%** | **TBD — Phase 12** |

Coverage is a diagnostic, not a goal. 70% with the §4 tests present is far stronger than 95% achieved
by asserting that constructors return objects.

---

## 11. Quality gates

```bash
ruff check src tests scripts      # lint
black --check src tests scripts   # formatting
mypy src                          # types
pytest -m "unit or integration"   # tests
```

All four must pass before a phase is marked complete. There is no CI service configured — this is a
local project — so the gate is a documented pre-commit habit rather than an enforced pipeline, and that
limitation is stated rather than glossed over.

---

## 12. Regression policy

Every bug found after a phase is closed gets a failing test **first**, then a fix. The test is named
for the behaviour, not the bug (`test_backspace_on_empty_buffer_is_noop`, not `test_bug_14`), so it
still communicates something in six months. The Phase 0 config audit is the model for this: twelve real
defects were found by inspection, and the arithmetic consistency checks in §5.1 exist so they cannot
recur.

---

## 13. Current test status

| Category | Written | Passing |
|---|---|---|
| `unit` | 0 | — |
| `integration` | 0 | — |
| `model` | 0 | — |
| `camera` | 0 | — |
| Manual checklist | not executed | — |

**No tests exist yet.** The suite is written incrementally from Phase 2 onward and consolidated in
Phase 12.

---

## 14. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-26 | Initial strategy: taxonomy, fixtures, priority-1 validity tests, per-module plans, model and performance gates, manual checklist, coverage targets. |

---

## 15. Related documents

`RULES.md` §TS-xx · `DATASET.md` §9 (leakage — what §4.1 defends) · `MODEL.md` §5.5, §7.6, §9.3 ·
`PRD.md` §11 (ACC gates) · `PHASES.md` Phase 12 · `DESIGN.md` §11 (screenshots) ·
`pyproject.toml` (markers, coverage config).
