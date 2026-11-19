# DATA COLLECTION — Standard Operating Procedure (SOP)
## Real-Time Sign Language to Text Conversion System
**Project:** SRM Institute of Science and Technology — Machine Learning (B.Tech CSE)  
**Version:** v1.0

---

## Before You Start — Essential Reading

Read `docs/SIGN_VOCABULARY.md` completely before collecting any data.  
You must know what each sign looks like, what distinguishes it from similar signs, and what counts as an invalid sample.

---

## PART A — UNDERSTANDING SESSIONS (Critical Concept)

### What is a Session?

A **session** is one continuous recording of a single sign class under one set of conditions.

> **Example:**  
> Session HELLO_01: You perform HELLO repeatedly for ~15 seconds in your room at 9 AM.  
> Session HELLO_02: You perform HELLO repeatedly for ~15 seconds in the same room at 6 PM.  
> Session HELLO_03: You perform HELLO from 30cm closer to the camera.  
> Session HELLO_04: You perform HELLO in a different room or with different background.

Each session is stored with a unique `session_id` (e.g., `HELLO_20260923_092345_a3f1`).

### Why Do We Need Multiple Sessions — NOT Just More Frames?

This is **the most important concept** in your project and you **will** be asked about it in your viva.

**The Wrong Mental Model:** "I'll just record 330 frames of HELLO in one go."

**The Problem — Frame-Level Leakage:**

Imagine you record 330 consecutive frames of HELLO at 8 fps. That's ~41 seconds of one continuous pose.

- Frame 47: your hand at position (0.42, 0.31)
- Frame 48: your hand at position (0.42, 0.32) — 1/8 second later

These frames are **nearly identical**. They are correlated.

Now you do a random 80/20 train/test split:
- Frame 47 → training set
- Frame 48 → test set

What has happened? The model is trained on frame 47, which is essentially **the same image** as test frame 48. When frame 48 is "tested," the model has already learned it in disguise. Your test accuracy is **98%** — but it's **fake**.

You collect a new session a week later → accuracy drops to **72%**. The 98% was an illusion.

**The Fix — Session-Level Grouping:**

All frames from Session 1 go to either training or test — **never split between them**.  
All frames from Session 2 go to the other.

Test accuracy is now **honest** — the test set contains frames the model has genuinely never encountered.

**Formal name:** Group-based splitting, implemented via `sklearn.model_selection.StratifiedGroupKFold` where `group = session_id`.

### Why 4+ Sessions Per Class (Not Just 1 Big Session)?

With session-level splitting:
- If you have only 1 session → you cannot put anything in the test set without losing all training data
- If you have 2 sessions → test set is one session (too small, noisy estimate)
- With 4+ sessions → you can hold out 1 session for test, use 3 for training → more statistically reliable

**Also:** Each session captures variation in lighting, distance, exact hand shape, and fatigue. More sessions = more generalization. The model doesn't just memorize one instance of your pose.

---

## PART B — SESSION MATH (What to Expect)

| Parameter | Value |
|---|---|
| Capture rate | 8 fps (configured) |
| Session cap | 90 samples per session |
| Countdown before capture | 3 seconds |
| Expected rejection rate | ~10–20% (coord bounds, duplicates, bbox) |
| Frames needed to reach 90 accepted | ~100–110 raw frames |
| Time at 8 fps for 100 frames | ~12–14 seconds |
| **Total session time (including countdown)** | **~15–20 seconds** |

**What this means for you:** One session takes about 20-25 seconds of total time including setup and countdown. For 13 classes × 5 sessions = 65 sessions = ~25–30 minutes of actual capture time (plus pauses to reposition).

**Practical plan:** Budget 2–3 hours across 2–3 days for complete data collection.

---

## PART C — WHAT TO VARY AND WHAT TO KEEP CONSISTENT

### ✅ VARY These (Helps the Model Generalize)

| What to Vary | How Much | Why |
|---|---|---|
| Distance from camera | ±10–15 cm from your standard position | Model should work at different hand distances |
| Hand position in frame | Left-center-right, up-down | Translation invariant, but helps with bbox validation range |
| Wrist tilt | ±15–20° from standard orientation | Natural variation within the sign's definition |
| Lighting | Morning vs evening, bright vs dim (but still readable) | Model should work in varying room lighting |
| Background | Different wall colors/textures (across sessions) | Robustness — MediaPipe handles most backgrounds |
| Natural micro-movements | Small finger adjustments, slight hand sway | Do NOT freeze rigidly — natural variation is better |
| Left vs right hand | The system handles both via mirror normalization | More variety in the canonical feature space |

### ❌ DO NOT VARY These (Changes the Sign Identity)

| What NOT to Vary | Why |
|---|---|
| Fundamental finger configuration | A HELLO with 4 fingers spread is wrong |
| Palm orientation (for PLEASE, THANK_YOU) | Up vs toward-camera is the ONLY distinguishing feature |
| Which fingers are extended vs curled | This IS the sign |
| Sign identity during a session | Don't mix two different signs in one session |

### The Balance

Natural variation is **good** — don't robotically freeze every muscle. A slight hand sway, micro-adjustment of wrist angle, breathing movement — these are all natural and help the model learn the sign's concept, not just one memorized exact configuration.

What makes variation **bad** is when it fundamentally changes the sign — e.g., accidentally curling one finger during HELLO, or letting the thumb drift up during SORRY.

---

## PART D — RECOMMENDED COLLECTION SCHEDULE

A practical **2-3 day schedule** for 13 classes × 4-6 sessions each:

### Day 1 — Morning (30-40 min)

| Session # | Class | Variation |
|---|---|---|
| 1-2 | HELLO | Normal distance, then 15cm closer |
| 3-4 | THANK_YOU | Normal distance, then different background |
| 5-6 | PLEASE | Normal, then slightly dimmer lighting |

### Day 1 — Evening (30-40 min)

| Session # | Class | Variation |
|---|---|---|
| 7-8 | YES | Normal, then left hand |
| 9-10 | NO | Normal, then different arm height |
| 11-12 | SORRY | Normal, then closer + different background |

### Day 2 — Morning (30-40 min)

| Session # | Class | Variation |
|---|---|---|
| 13-14 | HELP | Normal, then slight tilt variation |
| 15-16 | NAME | Normal, then different distance |
| 17-18 | WATER | Normal, emphasize spread on second session |

### Day 2 — Evening (30-40 min)

| Session # | Class | Variation |
|---|---|---|
| 19-20 | EAT | Normal, then slightly looser pinch |
| 21-22 | I_LOVE_YOU | Normal, then left hand |
| 23-24 | YOU | Normal, then different pointing angle |

### Day 3 — Any time (20-30 min)

| Session # | Class | Variation |
|---|---|---|
| 25-28 | _NEUTRAL | 4 sessions with VERY varied neutral poses |
| 29-30 | (Any class below 4 sessions) | Fill gaps |
| 31-32 | (Any class below target) | Top up sample counts |

**Priority:** Ensure every class has at least **4 sessions** and **200 valid samples**. Then aim for 6 sessions / 330 samples for the best model quality.

---

## PART E — ENVIRONMENT SETUP

### Lighting
- **Best:** Diffuse indoor lighting — overhead light or desk lamp that illuminates the hand evenly
- **Avoid:** Direct sunlight creating harsh shadows on fingers
- **Avoid:** Backlit conditions (camera pointing toward a bright window) — hand appears dark
- **Target:** Hand should be clearly visible, no blown highlights, no deep shadow between fingers

### Background
- **Best:** Plain/simple background (a wall, closed curtain)
- **Acceptable:** Slightly textured background (bookshelves, desk)
- **Avoid:** Moving backgrounds (TV, fan blades, people walking behind)
- **Why it matters:** MediaPipe hand detection works on all backgrounds, but complex moving backgrounds can occasionally cause detection failures, increasing your rejection rate

### Camera
- Your webcam is at Index 0 [DSHOW] — this was verified in Phase 2
- No changes needed to camera settings
- Sit or stand comfortably so you can hold the sign pose for 15-20 seconds

### Distance
- Start at a distance where your hand fills roughly **30-60% of the frame width**
- Use the verification script (`verify_hand_detection.py`) to check if your hand is being detected before starting collection
- Too close: hand exits the frame, bbox fraction too large → rejected
- Too far: landmarks become inaccurate → noisy features

### Clothing
- **Avoid:** Sleeves that cover the wrist — the wrist is landmark 0 and must be clearly visible
- **Acceptable:** Short sleeves, rolled-up sleeves
- **Best:** No sleeve or wrist visible

---

## PART F — STEP-BY-STEP COLLECTION PROCEDURE

Follow these steps **for each sign class, each session**:

### Step 0: Before Any Collection

```powershell
# Activate the environment
cd "c:\Users\harsh\OneDrive\ドキュメント\ML PROJECT\sign-language-to-text"
$env:PYTHONIOENCODING="utf-8"

# Check current dataset status
.venv\Scripts\python.exe scripts\collect_data.py --status
```

### Step 1: Select a Sign to Collect

1. Look at the status table — find a class that is incomplete (below 330 samples or fewer than 4 sessions)
2. Re-read the specification for that sign in `SIGN_VOCABULARY.md`
3. Stand in front of a mirror and practice the sign 10 times until it feels natural and consistent

### Step 2: Set Up the Environment

1. Position yourself in front of the camera
2. Check lighting — hand should be well-lit and clearly visible
3. Check background — keep it simple for this session
4. Decide your variation for this session (normal distance? closer? evening light?)
5. Note your planned variation before starting (you'll log this)

### Step 3: Start the Collection Script

For a specific sign:
```powershell
.venv\Scripts\python.exe scripts\collect_data.py --label HELLO
```

For all incomplete signs in sequence:
```powershell
.venv\Scripts\python.exe scripts\collect_data.py
```

### Step 4: During the Countdown (3 seconds)

- The window shows a large countdown number
- Get your hand into the sign position NOW
- Do NOT wait for "GO" to start positioning — be ready before the countdown hits 1

### Step 5: During Capture (~15-20 seconds)

**What to do:**
- Hold the sign naturally — do not freeze stiffly
- Every 5-8 seconds: **slightly vary** by moving your hand a few centimetres closer or farther, or tilting the wrist by 10-15 degrees
- Watch the progress bar fill at the bottom of the screen
- The accepted count should increase steadily — if it stalls, check if your hand is in frame and well-positioned

**What NOT to do:**
- Don't change to a different sign mid-session
- Don't let your pose drift so far that the sign is no longer valid
- Don't block the wrist with your sleeve

### Step 6: End the Session

- The script stops automatically when 90 samples are reached
- Or press **Q** or **ESC** to stop early if needed (samples collected so far are saved)
- Note: samples are saved immediately — you don't lose data by pressing Q

### Step 7: Check the Results

```powershell
.venv\Scripts\python.exe scripts\collect_data.py --status
```

- The new session should show in the counts
- If the session produced fewer than 30 samples, something was wrong (re-check lighting, distance, hand position)
- If the session produced 80-90 samples, it was successful

### Step 8: Decide on Another Session

At the prompt: "Collect another session for this class? [y/N]"

- Answer **y** if you want to collect another session NOW (perhaps with different lighting/distance)
- Answer **N** if you want to take a break or move to a different class
- You can always come back later

### Step 9: Log Your Session

After each session, record in your personal ML logbook:

```
Session log entry:
- Label: HELLO
- Session ID: (visible in terminal output)
- Date: 2026-09-23
- Time: 09:30 AM
- Lighting: Morning natural light
- Distance: Normal (50cm approx)
- Background: White wall
- Samples collected: 87
- Any issues: None / Hand was slightly too close in first 10 seconds
```

---

## PART G — PER-SIGN COLLECTION TIPS

### HELLO
- Actively spread all 5 fingers as wide as comfortable
- Don't let fingers close — they should be clearly spread throughout

### THANK_YOU
- Consciously keep fingers together (not spread)
- Ensure palm faces camera, not ceiling — this is what separates it from PLEASE

### PLEASE
- The palm-up orientation is the ONLY thing separating PLEASE from THANK_YOU
- Exaggerate the upward palm direction in early sessions; you'll naturally vary in later sessions

### YES and NO
- Exaggerate the thumb direction — clearly up for YES, clearly down for NO
- If you relax the thumb to a neutral side position, the sample may be ambiguous

### SORRY
- Make sure the thumb is clearly tucked or pressed against the fingers — no extension
- The knuckle-toward-camera orientation distinguishes it from YES/NO

### HELP
- The L-shape: thumb clearly up, index clearly sideways — be deliberate
- Keep pinky clearly folded — any extension starts to look like I_LOVE_YOU

### NAME and WATER
- NAME: two fingers TOGETHER, pointing up — consciously keep them touching
- WATER: three fingers SPREAD — consciously spread them wide
- These are the most confusable pair in the vocabulary

### EAT
- Practice the pinch/claw shape — all 5 fingertips toward the thumb tip
- A loose pinch (fingertips near but not touching) is fine — the shape is what matters

### I_LOVE_YOU
- Thumb + index + pinky all extended — consciously hold all three out
- Press middle and ring firmly against the palm — this is the critical folded position

### YOU
- Only the index finger — keep the thumb firmly tucked alongside the folded fingers
- Any thumb extension creeps toward HELP; any pinky extension creeps toward ILY

### _NEUTRAL (MOST IMPORTANT — TAKE YOUR TIME ON THIS CLASS)

Collect 4-6 sessions of VARIED neutral poses:

**Session 1:** Relaxed hanging hand (wrist dangling naturally, fingers slightly curled)  
**Session 2:** Flat hand at odd angles that don't match any sign  
**Session 3:** Mid-transition hand movements (slowly moving hand, random in-between positions)  
**Session 4:** Loosely open fist, not tight enough to be SORRY  
**Session 5:** Random idle hand states — let your hand just rest naturally  

The neutral class is your **reject boundary** — it must cover the space between signs so the model can say "no sign" rather than falsely predicting one.

---

## PART H — QUALITY CHECKS DURING COLLECTION

After every 2-3 sessions of the same class:

1. Check that the sample count increased by ~60-90 per session
2. If a session produced <30 samples: re-check your lighting and hand distance
3. Look at the debug images in `data/raw/<label>/` — do they show the correct pose?
4. Verify that MediaPipe is clearly detecting your hand during collection (the skeleton should appear)

---

## PART I — COMMANDS REFERENCE

```powershell
# Check dataset status
.venv\Scripts\python.exe scripts\collect_data.py --status

# Collect a specific class
.venv\Scripts\python.exe scripts\collect_data.py --label HELLO

# Collect all incomplete classes (interactive, class by class)
.venv\Scripts\python.exe scripts\collect_data.py

# View debug images for a class
explorer data\raw\HELLO

# Run all automated tests (should still be 51/51 passing)
.venv\Scripts\pytest.exe -v
```

---

## PART J — MINIMUM COMPLETION CRITERIA

Before proceeding to Phase 5 (training), verify:

| Check | Requirement |
|---|---|
| All 13 classes present in manifest | ✅ |
| Each class ≥ 200 samples | ✅ |
| Each class ≥ 4 distinct sessions | ✅ |
| No class with 0 samples | ✅ |
| Debug images look correct (right sign, hand visible) | ✅ (manual spot check) |

**Recommended target before training:**

| Check | Target |
|---|---|
| Each class samples | 300–330 |
| Each class sessions | 5–6 |
| Total dataset size | ~4,000–4,300 samples |

---

*Document version: v1.0 — Updated: 2026-09-23*
