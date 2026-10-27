# DEPLOYMENT.md — Setup, Running, and Distribution

**Project:** Real-Time Sign Language to Text Conversion System
**Document version:** 1.0
**Status:** Procedures written (Phase 0). **Not yet executed.** Every version number is unverified.
**Last updated:** 2026-08-26

> **Verification status: NOT VERIFIED.** No command in this document has been run on this machine.
> `requirements.txt` carries the same banner. Phase 1 exists specifically to execute these steps and
> correct whatever is wrong. Treat every pin and every timing as a proposal until then.

---

## 1. What "deployment" means here

This is not a web service and there is no server. "Deployment" means three things: getting the project
running on a fresh machine, running it reproducibly for a demonstration, and handing it to someone else
(an examiner, a teammate, a future maintainer) so they can run it too.

The architecture is deliberately **local-only**, and that is a design constraint rather than a
simplification. Camera frames are processed in-process and never transmitted (`PRD.md` NFR-11…NFR-14).
There is nothing to deploy to, no endpoint to secure, and no data in flight. This eliminates an entire
category of privacy risk at the cost of the system running on exactly one computer at a time — an
acceptable trade for a prototype whose subject matter is inherently sensitive video of people.

---

## 2. Target environment

| Item | Requirement | Notes |
|---|---|---|
| OS | Windows 10/11 (primary) | Developed and demonstrated here |
| OS | macOS / Linux | Should work; **untested** |
| Python | **3.10 or 3.11** | `pyproject.toml`: `>=3.10,<3.12` |
| RAM | 8 GB | 4 GB likely workable; untested |
| Webcam | 720p, 30 fps | Built-in laptop camera is sufficient |
| Disk | ~2 GB | Dependencies ~1.2 GB, dataset a few hundred MB |
| GPU | **Not required** | Nothing in the stack uses one |

### 2.1 Why the Python upper bound matters

Python 3.12 and 3.13 are excluded because **MediaPipe 0.10.x does not publish wheels for them**. On
3.13 `pip install mediapipe` fails outright; there is no workaround short of building from source,
which is not a reasonable ask.

There is a second, subtler constraint: MediaPipe 0.10.x is built against the **NumPy 1.x C ABI**, so
`requirements.txt` pins `numpy>=1.24,<2.0`. Installing NumPy 2.x alongside it produces an import-time
error about a module compiled against a different API version. Both of these are recorded in
`TROUBLESHOOTING.md` because they are the two most likely reasons a fresh install fails.

---

## 3. First-time setup

### 3.1 Verify Python

```bat
python --version
```

Expect `Python 3.10.x` or `3.11.x`. If Python is missing or the wrong version, install from
python.org and **check both "Add python.exe to PATH" and "tcl/tk and IDLE"** during installation. The
second box is easy to miss and is required: tkinter — and therefore CustomTkinter, and therefore the
entire UI — will not import without it.

### 3.2 Create and activate a virtual environment

```bat
cd "C:\Users\harsh\OneDrive\ドキュメント\ML PROJECT\sign-language-to-text"
python -m venv .venv
.venv\Scripts\activate
```

Prompt should now show `(.venv)`. On PowerShell use `.venv\Scripts\Activate.ps1`; if execution policy
blocks it, `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` for that session only.

A virtual environment is not optional here. The NumPy pin is strict enough that installing into a
global interpreter will eventually collide with some other project and produce an error that looks like
a bug in this one.

### 3.3 Install dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Roughly 1.2 GB and, on a first install over a normal connection, several minutes — MediaPipe and OpenCV
are large wheels.

### 3.4 Install the project itself

```bat
pip install -e .
```

Editable install, so `import sign_language` works from anywhere and edits take effect without
reinstalling. Without this step, scripts must be run from the project root and imports break in
confusing ways.

### 3.5 Verify the installation — Phase 1's actual job

```bat
python -c "import sys, numpy, cv2, mediapipe, sklearn, pandas, joblib, yaml; print(sys.version); print('numpy', numpy.__version__); print('cv2', cv2.__version__); print('mediapipe', mediapipe.__version__); print('sklearn', sklearn.__version__)"
python -c "import tkinter, customtkinter; print('tk ok', customtkinter.__version__)"
```

| Package | Expected | Actual |
|---|---|---|
| Python | 3.10.x / 3.11.x | TBD — Phase 1 |
| numpy | 1.24–1.26 | TBD — Phase 1 |
| opencv-python | 4.8–4.10 | TBD — Phase 1 |
| mediapipe | 0.10.9–0.10.14 | TBD — Phase 1 |
| scikit-learn | 1.3–1.5 | TBD — Phase 1 |
| pandas | 2.x | TBD — Phase 1 |
| customtkinter | 5.2.x | TBD — Phase 1 |

**These "expected" values are the pins we wrote, not versions we have confirmed resolve together.**
Phase 1 fills the Actual column from real output and, if resolution fails, `requirements.txt` is
corrected and the NOT-VERIFIED banner is replaced with the date and machine of verification.

### 3.6 Verify the camera

```bat
python scripts\check_camera.py
```

Opens the camera, reads a few frames, prints resolution and measured FPS, releases the device. Written
in Phase 1. Confirming the camera separately from the app means that when the app later fails, you
already know whether the camera is the cause.

### 3.7 Freeze the real environment

```bat
pip freeze > requirements-lock.txt
```

After successful verification only. `requirements.txt` stays human-readable with ranges;
`requirements-lock.txt` records the exact resolved set that actually worked, which is what makes the
result reproducible months later. Committed once real.

---

## 4. Running the system

Full pipeline order. Each step's real output is recorded in `MEMORY.md` as it happens.

```bat
:: 1. Collect data — repeat per class, per session (Phase 3)
python scripts\collect_data.py --class HELLO --signer HT --session 1

:: 2. Build the feature dataset from raw sessions (Phase 5)
python scripts\build_dataset.py

:: 3. Train and compare all six candidates (Phase 6-7)
python scripts\train_model.py

:: 4. Evaluate the selected model (Phase 7)
python scripts\evaluate_model.py

:: 5. Development HUD (Phase 8-9)
python scripts\run_realtime.py

:: 6. The deliverable application (Phase 11)
python app.py

:: 7. Performance measurement (Phase 13)
python scripts\benchmark.py
```

**Dependency order is real.** `train_model.py` fails with a clear `DatasetError` if `build_dataset.py`
has not run; `app.py` shows the designed no-model screen rather than crashing if training has not run
(`DESIGN.md` §6.1). Failing early with a message that names the missing prerequisite is a deliberate
design choice, not an accident of implementation.

Expected timings are all **TBD** — they will be measured, not guessed. Grid search over six candidates
is the long pole and is the one worth timing carefully.

---

## 5. Configuration management

`config.yaml` at the project root is the single source of truth and **is committed**. It contains no
secrets — no keys, no credentials, no personal data. The team roster deliberately lives in `PRD.md §1`
rather than in config, so that no source file needs to read personal information at runtime.

For machine-specific overrides that should not be committed, `config.local.yaml` is git-ignored and
shallow-merged over the base if present. The realistic use is `camera.index`, which differs between
machines with and without an external webcam, and is exactly the kind of change that otherwise gets
committed by accident and breaks a teammate's setup.

No `.env` file is used. There is nothing to put in one, and `.gitignore` excludes it anyway as a
safeguard against a future contributor adding one.

---

## 6. Model versioning

```text
models/
└── v1/
    ├── sign_classifier.joblib     ← git-ignored (binary)
    ├── label_encoder.joblib       ← git-ignored (binary)
    └── model_metadata.json        ← COMMITTED
```

The directory is derived from `model.version` by `utils/paths.py` — a new version means bumping config,
not editing paths in code, so `v2` can coexist with `v1` and be compared rather than overwriting it.

**Binaries are ignored; metadata is committed.** This is the important asymmetry. The `.joblib` files
are large, opaque, and regenerable from data plus config plus seed. The metadata JSON is small, text,
and diffable, and it records what actually happened: the chosen algorithm and hyperparameters, all
metrics, the class list, the training date, the dataset fingerprint (sample count, session count, class
distribution), the `feature_set_version`, the random seed, and library versions. Git history therefore
shows how model quality changed over time even though the models themselves are not in the repository.

The `feature_set_version` field is load-bearing rather than informational: `enforce_feature_version_match:
true` makes the predictor refuse a model whose version does not match the config, because a silent
feature-order mismatch produces confident nonsense rather than an error (`MODEL.md` §7.4).

A model is only ever promoted to `models/v1/` by `train_model.py`. Hand-copying artifacts breaks the
guarantee that metadata describes the binary sitting next to it.

---

## 7. What is committed and what is not

| Committed | Excluded | Why excluded |
|---|---|---|
| All source, tests, scripts | `data/raw/**` | Size, and it is video-derived personal data |
| `config.yaml` | `data/features/**` | Regenerable build product |
| `docs/**` | `models/**/*.joblib` | Large binaries, regenerable |
| `model_metadata.json` | `.venv/`, `__pycache__/` | Environment and cache |
| `data/metadata/` (schemas) | `logs/**` | Runtime output |
| `reports/figures/**` (small PNGs) | `config.local.yaml`, `.env` | Machine-specific / secrets by convention |
| `requirements*.txt` | `data/raw/**/debug/*.jpg` | **Images of people's hands** |

The last row deserves emphasis. Debug JPEGs are the only images the project stores, they are sampled
sparsely (`debug_image_every_n: 20`) as an audit trail, and they are the one artifact that could
identify a person. They are excluded from git for privacy, not size.

Commit style is conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`) with
scopes matching package names, e.g. `feat(features): add scale normalisation`.

### 7.1 The OneDrive problem — unresolved

The project lives under `C:\Users\harsh\OneDrive\ドキュメント\`, which means **OneDrive is
synchronising the dataset to Microsoft's cloud**. This directly contradicts the privacy position stated
in §8, and it is recorded honestly here and in `DATASET.md` §12 rather than quietly ignored.

Three options: exclude `data/` from sync via OneDrive's selective-sync settings; move the whole project
outside the OneDrive tree; or accept it and amend the privacy claim. **This is an open decision
(`PRD.md` OQ-06) and must be resolved before real collection starts in Phase 3.** Until then, the
statement "video never leaves this machine" is aspirational rather than true, and should not be
presented as fact.

There is a second, mundane risk in the same place: OneDrive can turn synced files into cloud-only
placeholders, which makes them appear present while failing to open, producing bizarre I/O errors
mid-training.

---

## 8. Privacy and security posture

| Requirement | How it is met | Status |
|---|---|---|
| Camera activates only on request | Opened on explicit Start, released on Stop and on window close | Design |
| No frames transmitted | No network code anywhere in `src/`; nothing to disable | Design |
| Local processing | MediaPipe and scikit-learn run in-process | Design |
| No unnecessary personal data | Only `signer_id` initials, never names | Design |
| Explicit dataset storage | Exact paths and schema in `DATASET.md` §6 | Documented |
| Camera-off is visible | Header indicator plus the OS camera LED | Design |
| Personal data not in source | Roster in `PRD.md` only | Enforced |
| Dataset not committed | `.gitignore` | Enforced |
| **Dataset not cloud-synced** | **Not resolved — see §7.1** | **OPEN** |

The strongest privacy property is structural: the project has **no HTTP client, no telemetry, and no
analytics**. There is no setting to disable because there is no code path to disable — which is a
far better guarantee than a promise, and it is verifiable by grepping the source for `requests`,
`urllib` and `socket`. Phase 12 includes exactly that check.

Consent for collection is informal but real: the three signers are the three project members, each
records their own data knowingly, and no third party is recorded. If the vocabulary were ever extended
by recording someone outside the team, written consent would be required — noted so the precedent is
not set casually.

---

## 9. Handing the project to someone else

For an examiner or a new teammate:

1. Clone or copy the repository (~a few MB without data and models).
2. Follow §3 setup.
3. **No dataset and no model are included.** They must collect their own data (§4 step 1) and train
   (§4 step 3), or the owner must transfer `data/` and `models/v1/` separately by USB or private
   transfer — never by committing them.
4. `README.md` is the entry point; `TROUBLESHOOTING.md` is the second stop.

This is a genuine limitation and worth stating plainly: a fresh clone **cannot recognise signs**. It
launches, shows the no-model guidance screen, and is otherwise inert. That is the correct behaviour for
a project whose data is personal video, but it means "clone and run" does not produce a working demo,
and anyone evaluating the project needs to know that in advance rather than discovering it.

### 9.1 Demonstration checklist

Run through this **before** the demo, not during it. Camera works and no other application is holding
it (Zoom and Teams are the usual culprits — they can hold the device even when apparently idle) ·
`.venv` activated · `python app.py` launches and the header shows the model loaded · lighting is
adequate and the background is uncluttered · a plain background behind the signing hand · window
readable from the audience's distance · `logs/app.log` writable · laptop on mains power, since
aggressive power-saving throttles the CPU and directly reduces FPS.

Have a fallback: recorded screen capture and the confusion-matrix figure from `reports/figures/`. If
the camera fails in the room, the evaluation can still proceed on evidence.

---

## 10. Packaging as an executable — considered and deferred

PyInstaller could produce a standalone `.exe`, and the reasons not to are concrete rather than
squeamish. MediaPipe ships binary model assets that PyInstaller does not detect automatically, so it
needs manual `--add-data` for files whose paths differ per version. The bundle would be roughly
300–500 MB. Antivirus software routinely quarantines PyInstaller output. And the resulting artifact
still needs a dataset and a trained model to be useful, so it would not actually make distribution
easier — the hard part of distribution here is the data, not the code.

**Decision: not packaged.** Distribution is source plus a documented setup. If packaging is attempted
later it belongs after Phase 16 as an extension, with `--add-data` for MediaPipe assets, `--noconsole`,
and a real test on a machine that has never had Python installed. Anything less is untested and should
not be claimed as a deliverable.

---

## 11. Logging and diagnostics

Console at `INFO`, rotating file at `logs/app.log` (5 MB × 3 backups) from `config.yaml logging:`.
Sensitive content is never logged: no frames, no landmark arrays, no file paths containing personal
data — only counts, timings, class names and status. Log lines are the first thing to read when
something misbehaves, and `TROUBLESHOOTING.md` maps common messages to causes.

For a fault report, the useful bundle is: `logs/app.log`, `python --version`, `pip freeze`, the OS
version, and `model_metadata.json`. Never the dataset.

---

## 12. Reproducibility

`random_seed: 42` and `deterministic: true` are applied to Python `random`, NumPy, and every
scikit-learn estimator that accepts `random_state`. Same data, same config, same seed, same library
versions should reproduce the same model and the same metrics.

The honest caveats: MediaPipe landmark output can vary very slightly across versions and platforms, so
the *dataset* is only reproducible if the raw session files are preserved rather than re-recorded;
`n_jobs: -1` parallelism can introduce tiny floating-point ordering differences; and thread counts in
BLAS can shift results in the last decimal place. These do not affect conclusions, but "bit-identical"
is a stronger claim than we can make, so we do not make it.

What genuinely makes results reproducible is the combination of the committed `config.yaml`, the
committed `model_metadata.json` recording the dataset fingerprint and library versions, and
`requirements-lock.txt`. Any of those three missing, and reproduction becomes guesswork.

---

## 13. Revision history

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-08-26 | Initial deployment documentation: environment constraints and their reasons, setup, run order, config and model versioning, git policy, privacy posture with the OneDrive issue recorded as open, handover, packaging decision, reproducibility caveats. |

---

## 14. Related documents

`README.md` (quick start) · `TROUBLESHOOTING.md` (failure catalogue) · `PHASES.md` Phase 1
(verification gate) · `DATASET.md` §12 (privacy and OneDrive) · `MODEL.md` §7 (artifacts and metadata)
· `RULES.md` §CF-xx · `config.yaml`.
