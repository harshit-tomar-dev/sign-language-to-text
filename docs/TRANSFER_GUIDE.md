# PROJECT TRANSFER GUIDE
## How to Run This Project on Another Laptop
**Created:** 2026-09-23

---

## Overview

You will:
1. **ZIP** the project on your laptop (exclude the 1 GB `.venv` folder)
2. **Transfer** the ZIP via AnyDesk to the other laptop
3. **Install Python 3.11** on the other laptop
4. **Install `uv`** (package manager)
5. **Create a new virtual environment** and install all packages
6. **Apply the sitecustomize fix** (if the path has special characters)
7. **Verify** everything works with the test suite

Estimated time: **30-45 minutes** (mostly download time)

---

## PART 1 — On YOUR Laptop (Sending Machine)

### Step 1A — Create a ZIP of the Project (Excluding .venv)

Open PowerShell in the project folder and run:

```powershell
cd "c:\Users\harsh\OneDrive\ドキュメント\ML PROJECT"

# Create a ZIP of the project folder, EXCLUDING the .venv directory
# (The .venv folder is ~1 GB and must be recreated on the target machine anyway)
$source = ".\sign-language-to-text"
$dest   = ".\sign-language-transfer.zip"

Add-Type -Assembly "System.IO.Compression.FileSystem"
[System.IO.Compression.ZipFile]::CreateFromDirectory($source, $dest)
```

> **If the above ZIP command is slow or fails**, use this alternative with PowerShell's Compress-Archive:
> ```powershell
> # Navigate INTO the project folder first, then zip everything EXCEPT .venv
> cd "c:\Users\harsh\OneDrive\ドキュメント\ML PROJECT\sign-language-to-text"
> 
> $items = Get-ChildItem -Exclude ".venv", "__pycache__", "*.pyc", ".pytest_cache"
> Compress-Archive -Path $items -DestinationPath "..\sign-language-transfer.zip" -Force
> ```

The resulting `sign-language-transfer.zip` will be in the `ML PROJECT` folder.

**Expected ZIP size:** ~5-15 MB (without `.venv` and without collected data in `data/`)

> 📌 **If you have already collected data** (samples in `data/features/`):  
> Include the `data/` folder in the ZIP — this transfers your entire dataset.  
> The ZIP will be larger (depends on dataset size).

---

### Step 1B — Transfer via AnyDesk

1. Open **AnyDesk** on both laptops
2. Connect to the other laptop
3. In AnyDesk, use **File Transfer** mode:
   - Click the three-bar menu → **File Transfer**
   - Or press `Ctrl+F` in the AnyDesk session
4. Navigate to `c:\Users\harsh\OneDrive\ドキュメント\ML PROJECT\` on the left panel
5. Find `sign-language-transfer.zip`
6. Drag it to a convenient location on the RIGHT panel (the other laptop)
   - Suggested destination: `C:\Users\<username>\Desktop\` or `C:\ML PROJECT\`

---

## PART 2 — On the OTHER Laptop (Receiving Machine)

### Step 2A — Extract the ZIP

1. Navigate to where you saved `sign-language-transfer.zip`
2. Right-click → **Extract All**
3. Choose destination: `C:\ML PROJECT\sign-language-to-text\`
4. Click Extract

You should now have the folder: `C:\ML PROJECT\sign-language-to-text\`

> ⚠️ **IMPORTANT — Path advice:**  
> Use a **simple path without spaces or special characters** on the new machine.  
> For example: `C:\ML PROJECT\sign-language-to-text\` is fine.  
> Avoid paths like `C:\Users\John's Laptop\Documents\` — the apostrophe can cause issues.  
> The `ドキュメント` (Japanese) path issue was specific to your original machine and was already fixed via the `sitecustomize.py` workaround. On the new machine, if you use a simple English path, that fix is NOT needed.

---

### Step 2B — Install Python 3.11

> ⚠️ **CRITICAL:** This project requires **Python 3.11 specifically**.  
> Python 3.12 and 3.13 will NOT work — MediaPipe has no wheels for those versions.  
> Python 3.10 also works if 3.11 is unavailable.

1. Open a browser and go to: **https://www.python.org/downloads/release/python-3110/**
2. Scroll down to **Files** section
3. Download: **Windows installer (64-bit)** → `python-3.11.0-amd64.exe`
   - (Or the latest 3.11.x — any 3.11.x works)
4. Run the installer
5. **⚠️ On the first screen:** Check **"Add Python 3.11 to PATH"** — this is critical
6. Click **"Install Now"**
7. Wait for installation to complete

**Verify Python installed correctly:**
```powershell
python --version
```
You should see: `Python 3.11.x`

If you see a different version, run:
```powershell
py -3.11 --version
```

---

### Step 2C — Install `uv` (Package Manager)

`uv` is the fast package manager used to create the virtual environment.

Open **PowerShell as Administrator** and run:
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Wait for it to finish, then **close and reopen PowerShell**.

Verify:
```powershell
uv --version
```
You should see something like: `uv 0.x.x`

---

### Step 2D — Create the Virtual Environment

Navigate to the project folder and create the venv:

```powershell
cd C:\ML PROJECT\sign-language-to-text

# Create a Python 3.11 virtual environment named .venv
uv venv --python 3.11 .venv
```

You should see: `Using Python 3.11.x` and `created virtual environment`.

---

### Step 2E — Install All Dependencies

Now install all required packages from the lock file:

```powershell
cd C:\ML PROJECT\sign-language-to-text

# Install ALL packages (this downloads ~500 MB of packages — takes 5-15 min)
uv pip install --python .venv\Scripts\python.exe `
    mediapipe==0.10.14 `
    opencv-python==4.11.0.86 `
    numpy==1.26.4 `
    scikit-learn==1.9.1 `
    pandas==2.3.3 `
    joblib==1.6.0 `
    PyYAML==6.0.3 `
    matplotlib==3.11.2 `
    seaborn==0.13.2 `
    customtkinter==5.2.2 `
    Pillow==11.3.0 `
    tqdm==4.70.1 `
    pytest==9.1.1 `
    scipy==1.17.1
```

> ⏱️ This step downloads packages from the internet. Connect to Wi-Fi before running.

**Alternative — install from requirements.txt directly:**
```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Wait until all packages are installed successfully.

---

### Step 2F — Check If Sitecustomize Fix Is Needed

The `sitecustomize.py` fix was created because your original machine had a Japanese character (`ドキュメント`) in the path. **If your new machine uses a simple English path like `C:\ML PROJECT\`, you do NOT need this fix.**

**Quick check:**
```powershell
# Run a quick import test
$env:PYTHONIOENCODING="utf-8"
& "C:\ML PROJECT\sign-language-to-text\.venv\Scripts\python.exe" -c "import mediapipe; print('MediaPipe OK')"
```

- If you see `MediaPipe OK` → ✅ No fix needed, skip to Step 2G
- If you see a `UnicodeEncodeError` or similar path error → ⚠️ See the fix below

**If fix IS needed** (only for non-ASCII paths):

Find your Python site-packages folder:
```powershell
& "C:\ML PROJECT\sign-language-to-text\.venv\Scripts\python.exe" -c "import site; print(site.getsitepackages())"
```

Create `sitecustomize.py` in that folder with this content:
```python
# Non-ASCII path fix for MediaPipe import
import sys
import ctypes

def _to_short_path(p):
    try:
        buf = ctypes.create_unicode_buffer(32768)
        ctypes.windll.kernel32.GetShortPathNameW(p, buf, 32768)
        return buf.value or p
    except Exception:
        return p

sys.path = [_to_short_path(p) if p else p for p in sys.path]

# Add src/ to sys.path
from pathlib import Path
_src = Path(__file__).resolve()
for _ in range(8):
    candidate = _src / "src"
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break
    _src = _src.parent
```

---

### Step 2G — Add `src/` to sys.path for Tests

The project uses `src/` layout. The venv needs to know where to find the `sign_language` package.

```powershell
# Find your site-packages directory
& "C:\ML PROJECT\sign-language-to-text\.venv\Scripts\python.exe" -c "import site; print(site.getsitepackages()[0])"
```

This prints something like: `C:\ML PROJECT\sign-language-to-text\.venv\Lib\site-packages`

Create a `.pth` file there so Python finds `src/`:
```powershell
# Replace the path below with what was printed above
$sitepkg = "C:\ML PROJECT\sign-language-to-text\.venv\Lib\site-packages"

# Create the .pth file pointing to src/
"C:\ML PROJECT\sign-language-to-text\src" | Out-File -FilePath "$sitepkg\sign_language_src.pth" -Encoding ascii
```

---

### Step 2H — Set PYTHONPATH Environment Variable (Backup Method)

As an additional backup, set `PYTHONIOENCODING` in the session:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "C:\ML PROJECT\sign-language-to-text\src"
```

> 💡 **Tip:** To make these permanent, add them via:  
> System Properties → Advanced → Environment Variables → New System Variable

---

## PART 3 — Verify the Installation

### Step 3A — Run the Automated Test Suite

```powershell
cd "C:\ML PROJECT\sign-language-to-text"
$env:PYTHONIOENCODING="utf-8"

& ".venv\Scripts\pytest.exe" -v
```

**Expected result:** `51 passed` (no failures)

If you see import errors, check that:
1. The `.pth` file was created correctly (Step 2G)
2. All packages installed (Step 2E)
3. Python version is 3.11 (Step 2B)

---

### Step 3B — Verify MediaPipe Imports

```powershell
$env:PYTHONIOENCODING="utf-8"
& ".venv\Scripts\python.exe" -c "
import cv2
import mediapipe as mp
import numpy as np
import sklearn
import yaml
print(f'OpenCV:      {cv2.__version__}')
print(f'MediaPipe:   {mp.__version__}')
print(f'NumPy:       {np.__version__}')
print(f'scikit-learn:{sklearn.__version__}')
print('All imports OK')
"
```

Expected output:
```
OpenCV:      4.11.0
MediaPipe:   0.10.14
NumPy:       1.26.4
scikit-learn:1.9.1
All imports OK
```

---

### Step 3C — Verify the Camera Works

```powershell
$env:PYTHONIOENCODING="utf-8"
& ".venv\Scripts\python.exe" scripts\check_camera.py
```

A window should open showing your webcam feed with detected hands.

> ⚠️ **Camera index may differ on the new machine.**  
> If the camera doesn't open, check the camera index in `config.yaml`:  
> ```yaml
> camera:
>   index: 0   # Try 0, 1, or 2 if camera doesn't open
> ```

---

### Step 3D — Verify Hand Detection

```powershell
$env:PYTHONIOENCODING="utf-8"
& ".venv\Scripts\python.exe" scripts\verify_hand_detection.py
```

Hold your right hand in front of the camera — the window should show skeleton landmarks and label it "RIGHT".

---

### Step 3E — Check Dataset Status

```powershell
$env:PYTHONIOENCODING="utf-8"
& ".venv\Scripts\python.exe" scripts\collect_data.py --status
```

If you transferred the `data/` folder with your collected samples, you should see your sample counts. If you transferred without data, all counts will be 0.

---

## PART 4 — Quick Reference Commands

Once installed, all commands to run on the new machine use this format:

```powershell
# Always set this first in a new terminal
cd "C:\ML PROJECT\sign-language-to-text"
$env:PYTHONIOENCODING="utf-8"

# Run tests
& ".venv\Scripts\pytest.exe" -v

# Collect data
& ".venv\Scripts\python.exe" scripts\collect_data.py --label HELLO

# Check dataset status
& ".venv\Scripts\python.exe" scripts\collect_data.py --status

# Verify camera + hand detection
& ".venv\Scripts\python.exe" scripts\verify_hand_detection.py
```

---

## PART 5 — Troubleshooting

### "ModuleNotFoundError: No module named 'sign_language'"
- The `src/` path was not added. Redo Step 2G (create the `.pth` file).
- Or set `$env:PYTHONPATH="C:\ML PROJECT\sign-language-to-text\src"` before running.

### "No module named 'mediapipe'"
- Dependencies were not installed. Redo Step 2E.
- Make sure you are using the `.venv\Scripts\python.exe` not the system Python.

### "Could not find a version that satisfies mediapipe"
- Your Python version is wrong. MediaPipe only works on Python 3.10 or 3.11.
- Run `& ".venv\Scripts\python.exe" --version` to check.

### Camera not opening (index error)
- Edit `config.yaml` → `camera.index: 0` → try 0, 1, or 2.
- Run `scripts\check_camera.py` to find the right index.

### Tests fail with import errors
- Check the `.pth` file content (Step 2G)
- Verify all packages installed: `uv pip list --python .venv\Scripts\python.exe`

### "UnicodeEncodeError" when running scripts
- Set `$env:PYTHONIOENCODING="utf-8"` and `$env:PYTHONUTF8="1"` in the PowerShell session.
- If the new path also has non-ASCII characters, apply the sitecustomize fix (Step 2F).

---

## PART 6 — What NOT to Transfer

❌ **Do NOT transfer the `.venv` folder.** It is machine-specific and ~1 GB. Always recreate it on the target machine using Steps 2D-2E.

❌ **Do NOT transfer `.pytest_cache`** — it is auto-regenerated.

❌ **Do NOT transfer `__pycache__`** folders — auto-regenerated.

✅ **DO transfer the `data/` folder** if you want to bring your collected dataset.

✅ **DO transfer `models/`** if you have trained models (after Phase 6-7).

---

*Transfer Guide v1.0 — 2026-09-23*
