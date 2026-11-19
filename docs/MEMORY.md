# MEMORY.md — Persistent Project State

**Project:** Real-Time Sign Language to Text Conversion System
**Last updated:** 2026-08-26
**Updated by:** Phase 0 completion

> **Purpose.** This is the project's working memory — the file to read first when picking the project up
> after a break, and the file to update at the end of every phase. It records what has actually happened,
> what was decided and why, what is still open, and what the real measured numbers are.
>
> **Rule.** Nothing enters this file speculatively. If a number has not been measured it reads `TBD` with
> the phase that will measure it. Every entry is dated. Entries are appended, not rewritten — a superseded
> decision is struck through and the reason recorded, so the reasoning history survives.
>
> This is distinct from the assistant's own memory directory. This file belongs to the repository and is
> committed.

---

## 1. Where the project stands right now

| | |
|---|---|
| **Current phase** | Phase 0 — Planning and documentation |
| **Phase 0 status** | ✅ Complete, pending user approval |
| **Next phase** | Phase 1 — Environment verification |
| **Blocked on** | User approval of the architecture, and resolution of OQ-06 (OneDrive) before Phase 3 |
| **Code written** | **None.** Zero source files exist under `src/`. |
| **Data collected** | **None.** `data/raw/` is empty. |
| **Model trained** | **None.** `models/v1/` is empty. |
| **Tests written** | **None.** |
| **Dependencies installed** | **Not verified.** No install has been run. |

Nothing in this project has been executed. Every artifact so far is documentation and configuration.

### 1.1 Resume here

**Session ended 2026-08-26 with Phase 0 complete and presented for approval.**

The next action is a decision, not a task: either approve the architecture and start **Phase 1
(environment verification)**, or revise the plan first. Phase 1 is the right entry point because it is
purely mechanical — create the venv, install, print real version numbers into §5.1 of this file — and it
converts the largest block of current uncertainty (OQ-01, the unverified dependency pins) into fact.

Before Phase 3 (collection), **OQ-06 must be closed** — see §4. It is the only genuinely blocking item.

To pick this project up cold, read in this order: this file → `PHASES.md` Phase 1 → `DATASET.md` §9.

---

## 2. Phase log

| Phase | Name | Status | Started | Completed | Notes |
|---|---|---|---|---|---|
| 0 | Planning and documentation | ✅ Complete | 2026-08-26 | 2026-08-26 | 14 documents, config, scaffold. 12 config defects found and fixed by self-audit. |
| 1 | Environment verification | ⬜ Not started | — | — | Gate: resolves all version uncertainty |
| 2 | Camera and hand detection | ⬜ Not started | — | — | Resolves the frame-ordering question empirically |
| 3 | Data collection | ⬜ Not started | — | — | Blocked on OQ-06 |
| 4 | Feature engineering | ⬜ Not started | — | — | |
| 5 | Dataset assembly | ⬜ Not started | — | — | |
| 6 | Model training | ⬜ Not started | — | — | |
| 7 | Evaluation | ⬜ Not started | — | — | First real metrics land here |
| 8 | Inference + HUD | ⬜ Not started | — | — | |
| 9 | Stabilisation | ⬜ Not started | — | — | |
| 10 | Text composition | ⬜ Not started | — | — | |
| 11 | Desktop application | ⬜ Not started | — | — | |
| 12 | Test consolidation | ⬜ Not started | — | — | |
| 13 | Performance measurement | ⬜ Not started | — | — | First real timings land here |
| 14 | Manual QA + screenshots | ⬜ Not started | — | — | |
| 15 | Documentation reconciliation | ⬜ Not started | — | — | Replace every TBD with measured values |
| 16 | Report and viva preparation | ⬜ Not started | — | — | |

---

## 3. Decisions made, and why

Each of these is expanded as an ADR in `ARCHITECTURE.md`. The one-line reason is what you need when
someone asks "why did you do it that way?" in a viva.

| # | Date | Decision | Reason in one line |
|---|---|---|---|
| D-01 | 2026-08-26 | Landmark-based classical ML (Option A), not image-based deep learning | 63 features instead of ~150k pixels makes a few-hundred-sample dataset viable |
| D-02 | 2026-08-26 | MediaPipe as a **frozen** feature extractor | We are doing transfer learning, and we say so rather than claiming to avoid deep learning |
| D-03 | 2026-08-26 | Store raw landmarks as primary; features are a build product | Lets us re-engineer features later without re-recording anyone |
| D-04 | 2026-08-26 | Sparse debug JPEGs (`every_n: 20`) as an audit trail | Enough to verify collection went right; not a photo archive of people |
| D-05 | 2026-08-26 | Translation + scale normalisation; **no rotation normalisation** | Rotation invariance would make PLEASE and THANK_YOU mathematically identical |
| D-06 | 2026-08-26 | Mirror left hands onto right | One canonical handedness halves the effective variation |
| D-07 | 2026-08-26 | Detect before flip (`detect_before_flip: true`) | A mirrored detection inverts handedness *and leaves the dataset looking valid* |
| D-08 | 2026-08-26 | Group the split by `session_id` (`StratifiedGroupKFold`) | Consecutive frames are near-duplicates; random splitting would inflate scores |
| D-09 | 2026-08-26 | `StandardScaler` **inside** the sklearn Pipeline | Refits per fold, so preprocessing leakage is structurally impossible |
| D-10 | 2026-08-26 | Report grouped **and** random scores side by side | Quantifies our own leakage instead of asserting there is none |
| D-11 | 2026-08-26 | LOSO reported, not gated | With 3 signers it is too noisy to gate on honestly |
| D-12 | 2026-08-26 | A `_NEUTRAL` rejection class | A closed-set classifier would otherwise force a resting hand into a real sign |
| D-13 | 2026-08-26 | Compare 6 candidates, select on macro-F1 | Macro-F1 weights all 13 classes equally regardless of sample count |
| D-14 | 2026-08-26 | Stabilise with 7 votes out of the last 10, not 7 consecutive | One dropped frame costs one vote instead of resetting all progress |
| D-15 | 2026-08-26 | Cooldown 1.2 s + must pass through `_NEUTRAL` to repeat | Prevents `HELLO HELLO HELLO` from one held sign |
| D-16 | 2026-08-26 | `feature_set_version` enforced at model load | A silent feature-order mismatch yields confident nonsense, not an error |
| D-17 | 2026-08-26 | CustomTkinter desktop app; **Streamlit rejected** | Live webcam in Streamlit needs WebRTC — more latency, more demo-day risk |
| D-18 | 2026-08-26 | OpenCV HUD first (Phase 8), full app later (Phase 11) | Lets inference be built and debugged without any GUI concerns |
| D-19 | 2026-08-26 | Team roster in `PRD.md` only, never in config or source | Personal data stays out of every runtime code path |
| D-20 | 2026-08-26 | `capture_fps: 8`, `sessions_per_class_minimum: 4` | Reduces frame redundancy at source and gives the grouped split real variety |
| D-21 | 2026-08-26 | `numpy<2` pinned | MediaPipe 0.10.x is built against the NumPy 1.x C ABI |
| D-22 | 2026-08-26 | Python `>=3.10,<3.12` | MediaPipe 0.10.x publishes no wheels for 3.12+ |
| D-23 | 2026-08-26 | No PyInstaller packaging | MediaPipe assets, ~400 MB bundle, antivirus — and it still needs data and a model |
| D-24 | 2026-08-26 | Descope vocabulary rather than redefine the metric if ACC-02 is missed | 8 signs with defensible numbers beats 12 with numbers we cannot defend |

### 3.1 Superseded decisions

None yet. When one is superseded, the row above is struck through and a dated replacement is added with
the reason for the change.

---

## 4. Open questions

Cross-referenced with `PRD.md` §21. Each must be closed before the phase named.

| ID | Question | Must resolve before | Status |
|---|---|---|---|
| OQ-01 | Do the pinned versions actually resolve together? | Phase 1 | Open |
| OQ-02 | Does `dshow` fix camera-open latency on this machine? | Phase 2 | Open |
| OQ-03 | Which frame-order convention does MediaPipe handedness actually follow here? | Phase 2 | Open |
| OQ-04 | Is 12 signs + `_NEUTRAL` separable enough, given the YES/NO/SORRY fist trio? | Phase 7 | Open |
| OQ-05 | Are 330 samples per class enough? | Phase 7 | Open |
| **OQ-06** | **How do we stop OneDrive syncing the dataset?** | **Phase 3 — blocking** | **Open** |
| OQ-07 | Is the roster in `PRD.md` §1 final and correctly spelled? | Phase 16 | Open — marked "proposed" |
| OQ-08 | Does the z coordinate help? (63-dim vs 42-dim ablation) | Phase 7 | Open |
| OQ-09 | Is `confidence_threshold: 0.80` the right value empirically? | Phase 9 | Open |
| OQ-10 | Can we cite a real deaf-population statistic, or leave it out? | Phase 16 | Open — deliberately uncited |

**OQ-06 is the only one that blocks the next real work.** The project directory sits inside OneDrive,
which means the dataset would be synchronised to Microsoft's cloud — directly contradicting the privacy
position in `DEPLOYMENT.md` §8. Until it is resolved, the claim "video never leaves this machine" is
aspirational rather than true and must not be presented as fact.

---

## 5. Measured results

**All empty. Every number below will be a real measurement or will stay marked TBD.**

### 5.1 Environment (Phase 1)

| Item | Value |
|---|---|
| Machine / CPU | TBD |
| RAM | TBD |
| OS build | TBD |
| Python | TBD |
| numpy / opencv / mediapipe / sklearn / pandas / customtkinter | TBD |
| Install succeeded first attempt? | TBD |
| Deviations from `requirements.txt` | TBD |

### 5.2 Camera (Phase 2)

| Item | Value |
|---|---|
| Working `camera.index` | TBD |
| Backend chosen | TBD |
| Open latency | TBD |
| Achieved capture FPS | TBD |
| Detection FPS at `model_complexity: 0` | TBD |
| Handedness convention confirmed | TBD |

### 5.3 Dataset (Phase 3–5)

| Item | Value |
|---|---|
| Classes collected | TBD (13 planned) |
| Total accepted samples | TBD |
| Samples per class (min / median / max) | TBD |
| Sessions per class (min) | TBD |
| Signers | TBD (3 planned: HT, BT, ST) |
| Rejection rate and top reason | TBD |
| Actual collection wall-clock | TBD (estimate: 2.5–5 h total) |
| Feature matrix shape | TBD (expect `(N, 63)`) |

### 5.4 Model comparison (Phase 6–7)

Grouped 5-fold CV, macro-F1 primary.

| Candidate | Macro-F1 | Accuracy | Fit time | Selected |
|---|---|---|---|---|
| logistic_regression | TBD | TBD | TBD | |
| knn | TBD | TBD | TBD | |
| svm_rbf | TBD | TBD | TBD | |
| random_forest | TBD | TBD | TBD | |
| extra_trees | TBD | TBD | TBD | |
| mlp | TBD | TBD | TBD | |

**Selected model:** TBD — Phase 7. **Best hyperparameters:** TBD.

### 5.5 Acceptance gates (Phase 7)

| ID | Gate | Threshold | Measured | Pass? |
|---|---|---|---|---|
| ACC-01 | Target macro-F1 (grouped CV) | ≥ 0.85 | TBD | — |
| ACC-02 | Minimum macro-F1 | ≥ 0.75 | TBD | — |
| ACC-03 | Per-class recall, all classes | ≥ 0.70 | TBD | — |
| ACC-04 | `_NEUTRAL` recall | ≥ 0.90 | TBD | — |
| ACC-05 | Held-out test macro-F1 | ≥ 0.75 | TBD | — |
| ACC-06 | LOSO macro-F1 | reported | TBD | n/a |

### 5.6 The leakage experiment (Phase 7)

The headline honest result of the project.

| Protocol | Macro-F1 |
|---|---|
| Random split (leaky — **for comparison only**) | TBD |
| Grouped by `session_id` (**reported result**) | TBD |
| Leave-one-signer-out | TBD |
| **Gap (random − grouped)** | **TBD** |

The gap is the quantity to be able to explain. A large gap does not mean something went wrong — it means
the grouped protocol is doing its job and the random number was never trustworthy.

### 5.7 Performance (Phase 13)

| Metric | Target | Measured |
|---|---|---|
| End-to-end FPS | ≥ 15 | TBD |
| Inference latency p50 / p95 | p95 ≤ 50 ms | TBD |
| MediaPipe detection per frame | — | TBD |
| Model load time | ≤ 2000 ms | TBD |
| Memory RSS | — | TBD |
| Sign-held to text-appears | — | TBD |
| Hardware these were measured on | — | TBD |

### 5.8 Test suite (Phase 12)

| Item | Value |
|---|---|
| Unit / integration / model / camera tests | 0 / 0 / 0 / 0 |
| Overall coverage | TBD (target ≥ 70%) |
| Manual checklist executed | No |

### 5.9 Live evaluation (Phase 14)

| Item | Value |
|---|---|
| Signs reliably recognised, own signer | TBD |
| Signs reliably recognised, unseen signer | TBD |
| Observed live accuracy vs test accuracy | TBD |
| Classes that failed in practice | TBD |

---

## 6. Phase 0 config audit — the twelve defects

A self-audit of `config.yaml` before any code was written found twelve real inconsistencies. Recorded
here because they are evidence that the single-source-of-truth discipline works, and because several
would have been expensive to find later. Full table in `ARCHITECTURE.md` §13.4.

The five most serious:

1. **`YES` and `NO` were parsed as booleans, not class names.** YAML 1.1 resolves the bare tokens
   `YES` / `NO` / `ON` / `OFF` / `Y` / `N` to booleans, so `dataset.classes[3]` was Python `True` and
   `[4]` was `False`. That would have produced `data/raw/True/` and `data/raw/False/` directories, a
   label encoder fitted on mixed types, and two of thirteen classes that could not be named in a log
   line — with **no error at load time**. Every class name is now quoted, and a test asserts they are
   all strings. Found by *parsing* the file, not reading it; the text `- YES` looks exactly like what
   it was meant to be.
2. **`samples_per_session_cap` did not reach the per-class target.** The cap multiplied by the session
   minimum came out below `samples_per_class_target`, so collection could satisfy every stated rule and
   still fall short. Fixed to `90` (4 × 90 = 360 ≥ 330). The arithmetic is now asserted by a config test.
3. **`min_votes_in_window` exceeded `window_size`** in an early draft — a stabiliser that could never
   commit anything. Fixed to 7 of 10, now validated.
4. **`extended_dim` did not match the sum of its declared components.** Fixed to 83. Harmless while
   `use_extended: false`, and a silent dimension mismatch the moment anyone enabled it.
5. **Coordinate validation bounds were `[0, 1]`,** which would have rejected every legitimate frame where
   a hand crossed the frame edge — MediaPipe extrapolates slightly outside the image. Widened to
   `[-0.5, 1.5]`.

Two lessons kept for later. **A found defect becomes a test** — `TESTING.md` §5.1 now includes the
consistency checks that make these impossible to reintroduce. And **validate config by loading it, not
by reading it**: eleven of these were visible to careful reading, and the worst one was not.

---

## 7. Things not to forget

Written for whoever picks this up cold — including us, in three weeks.

**Never random-split this dataset.** Consecutive frames from one session are near-duplicates. A random
split will produce a beautiful number that means nothing, and it is the first thing a knowledgeable
examiner will probe.

**The frame-ordering bug leaves no trace.** If detection ever runs on a mirrored frame, handedness
inverts, the left-hand mirroring step does the wrong thing, and the saved `.npz` files still look
perfectly valid. There is no crash. Verify before collecting, not after.

**`_NEUTRAL` is collected backwards from every other class.** High variance, not low. Resting hands,
transitions, random positions. Collected narrowly, it fails to reject the resting hands it will actually
see.

**Never disable `enforce_feature_version_match`.** The exception is the only thing between you and
confident, plausible, wrong predictions.

**The class names in `config.yaml` must stay quoted.** `YES` and `NO` are YAML booleans unquoted. This
was audit defect 12 and it produces no error — just `True` and `False` where two class names should be.

**Do not lower a minimum to make an error go away.** `sessions_per_class_minimum` failing means collect
more sessions, not lower the bar.

**Suspiciously high accuracy is a bug report, not a result.** Above 0.98 on grouped CV with this dataset
means check the split before celebrating.

**If accuracy misses the floor, cut signs — never redefine the metric.** The descoping plan in
`PHASES.md` exists precisely so this decision is made in advance rather than under pressure.

**Keep a reference to every tkinter `PhotoImage`.** Otherwise the video panel goes blank with no error.

**Never touch a tkinter widget from a worker thread.** Queue plus `after()`, every time.

**OneDrive is still syncing the dataset.** OQ-06. Unresolved.

---

## 8. Update protocol

At the end of every phase, update: §1 current phase · §2 the phase row with real dates · §3 any new
decision with its one-line reason · §4 questions closed, with the answer · §5 real measured numbers
replacing `TBD` · §7 anything learned the hard way. Then commit with
`docs(memory): record phase N results`.

Do not batch these updates. The value of this file is that it is current, and a memory file updated
retrospectively from recollection is a memory file containing guesses.

---

## 9. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-26 | Created at Phase 0 completion: state, phase log, 24 decisions, 10 open questions, empty results tables, the twelve-defect audit record, the do-not-forget list. |

---

## 10. Related documents

`PHASES.md` (the roadmap this tracks) · `PRD.md` §21 (open questions) · `ARCHITECTURE.md` §13.4 (audit
detail) and ADRs · `MODEL.md` §8 (evaluation protocols) · `DATASET.md` §9 (leakage) ·
`TESTING.md` (gates) · `DEPLOYMENT.md` §7.1 (OneDrive).
