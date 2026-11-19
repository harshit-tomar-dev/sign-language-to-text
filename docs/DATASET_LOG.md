# DATASET LOG
## Real-Time Sign Language to Text Conversion System

**Purpose:** Track every collection session — what was collected, when, under what conditions, and how many samples were accepted.

Update this file after every collection session.

---

## Dataset Version: v1

**Collection period:** TBD  
**Signer:** Harsh (single signer)  
**Feature version:** v1-base-63  
**Script used:** `scripts/collect_data.py`

---

## Session Log

Copy and fill one entry per collection session:

```
Session ID:        [copy from terminal output, e.g. HELLO_20260923_092345_a3f1]
Date:              YYYY-MM-DD
Time:              HH:MM AM/PM
Label:             [HELLO / THANK_YOU / ...]
Samples accepted:  [number from --status after session]
Session number:    [for this label, e.g., "session 2 of target 5"]
Lighting:          [e.g., morning natural light / desk lamp / evening dim]
Background:        [e.g., white wall / bookshelves]
Distance approx:   [e.g., 50cm / 60cm / 40cm (closer)]
Hand:              [Right / Left / alternating]
Special variation: [e.g., "moved closer halfway through" / "standard"]
Issues:            [e.g., "none" / "first 10 seconds hand too far"]
```

---

## Summary Table (Update After Each Session)

| Session ID | Date | Label | Samples | Lighting | Distance | Issues |
|---|---|---|---|---|---|---|
| *(pending first collection)* | | | | | | |

---

## Collection Statistics (Fill After Each Day)

### After Day 1

| Class | Sessions | Total Samples | Target Remaining |
|---|---|---|---|
| HELLO | — | — | 330 |
| THANK_YOU | — | — | 330 |
| ... | | | |

### After Day 2

*(Update similarly)*

### After Day 3 (Final)

*(Update similarly)*

---

## Quality Notes

Record any quality concerns observed during collection:

- *(Example)* SORRY sessions: found thumb was slightly visible in 3 sessions — may cause YES/NO confusion. Added 2 more sessions with emphasis on thumb tucked.
- *(Example)* PLEASE session 2: lighting was too dark — MediaPipe missed about 30% of frames. Session abandoned, re-collected.

---

## Dataset Audit Results (Fill After Phase 5)

| Check | Result | Pass/Fail |
|---|---|---|
| All 13 classes present | — | — |
| Min samples per class (200) | — | — |
| Min sessions per class (4) | — | — |
| No NaN features | — | — |
| No all-zero features | — | — |
| Class balance check | — | — |
| Total dataset size | — | — |

---

*Version 1.0 — 2026-09-23*
