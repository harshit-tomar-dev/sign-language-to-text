# SIGN VOCABULARY SPECIFICATION
## Real-Time Sign Language to Text Conversion System
**Project:** SRM Institute of Science and Technology — Machine Learning (B.Tech CSE)  
**Version:** v1.0 — 13-class vocabulary

---

## ⚠️ CRITICAL SCOPE DISCLAIMER — READ BEFORE COLLECTION

### What This System IS

This project implements a **Limited-Vocabulary Static Hand Gesture Classifier**. It:

- Recognises **13 predefined static hand poses** from a live webcam stream
- Uses **single-frame landmark snapshots** — no temporal/motion information
- Recognises **only the specific gestures it was trained on**
- Is trained on **one signer's data** (yours) — generalization to other signers is limited without multi-signer training data

### What This System is NOT

This system is **NOT**:
- A full Sign Language Recognition (SLR) system
- Capable of recognizing arbitrary ASL/BSL/ISL signs
- A grammar-aware sign language translator
- A temporal sequence recognizer (it does not track motion paths)

### Sign Language Recognition vs This Project

| Aspect | Full SLR System | This Project |
|---|---|---|
| Vocabulary | Thousands of signs | 13 specific gestures |
| Temporal info | Yes — motion trajectories | No — static snapshot only |
| Grammar | Yes — sign order, NMMs | No — isolated gesture |
| Generalization | Multi-signer datasets | Single-signer (yours) |
| Dynamic signs | Captured as sequences | Captured as static representative pose |
| Technique | RNNs, Transformers, 3D-CNNs | Classical ML on 63-D normalized landmarks |

### Why This Distinction Matters for Your Viva

When your examiner asks *"Does your system recognize sign language?"*, the correct answer is:

> *"My system is a limited-vocabulary hand gesture classifier that recognizes 13 predefined static poses. These poses are inspired by common sign language representations, but the model is learning my specific performance of each gesture. It is NOT a full sign language recognition system — it does not handle the full vocabulary, grammar, or movement sequences that real sign language requires."*

This is academically honest and technically defensible. Overclaiming would be a serious viva error.

---

## Vocabulary Design Rationale

The 13 classes were designed for **intrinsic separability** — each sign differs from all others based on **finger configuration and palm orientation**, not based on where the hand is in the frame.

This is critical because the feature pipeline removes position (translation invariance) and size (scale invariance). Two signs that are identical in finger configuration but differ only in hand location would be **unlearnable** by this system.

**Design principles:**
1. Each sign has a unique finger configuration (which fingers are extended/curled/spread)
2. Signs that look similar (HELLO vs THANK_YOU vs PLEASE) differ in palm orientation, which is preserved in the feature vector
3. The `_NEUTRAL` class is a **rejection class** — without it, the model is forced to label every idle hand as one of the 12 signs, producing false predictions when no sign is being made
4. Movement signs (like WATER in real ISL) are represented by their most distinctive **static representative pose**

---

## Verification Notice

The gesture descriptions below are broadly inspired by common representations. **They are the definitions YOUR model will learn from YOUR data.** Some differ from standard ASL/ISL for simplicity or separability reasons.

**Before finalizing your dataset, verify your specific poses by:**
- Consulting a trained sign language instructor at your institution
- Referencing a standardised sign language dictionary (e.g., ASL University, IndianSL.org)
- Noting in your EXPERIMENT_LOG.md which specific representation you used

---

## Sign Specifications

---

### 1. HELLO

| Property | Value |
|---|---|
| **Class label** | `HELLO` |
| **Meaning** | Greeting |
| **Static/Dynamic** | Dynamic in real sign language (waving motion), but captured as a **static open-palm pose** |
| **Hands required** | Single hand (right or left — system handles both) |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- All 5 fingers extended and spread apart (fan shape)
- Thumb also extended and spread outward
- No fingers curled

**Palm orientation:**
- Palm facing directly toward the camera (toward the viewer)

**Pose description:**  
Hold your hand with all five fingers open and spread as wide as comfortable, palm facing the camera directly. The hand should look like an open fan or a "stop" sign.

**Starting pose (for capture):**  
Open palm, fingers spread, palm facing camera. Hold this pose statically throughout the session.

**Natural variations (acceptable):**
- Slight tilt of wrist (±15°) while maintaining palm-toward-camera orientation
- Fingers slightly more or less spread — as long as all 5 are clearly extended

**Confusion risk:**
- **THANK_YOU** — also an open/flat palm facing camera, but THANK_YOU has fingers held **together** rather than spread. Be deliberate about spreading fingers for HELLO.
- **PLEASE** — palm orientation is different (PLEASE faces upward, not toward camera)

**Collection instructions:**
- Practice spreading all 5 fingers widely before starting
- During collection, don't freeze stiffly — small natural micro-movements are fine
- Every 10 seconds, slightly vary your wrist tilt or move the hand a few centimetres closer/farther

**INVALID samples for HELLO:**
- Hand partially closed (any finger curling)
- Fewer than 5 fingers visible as extended
- Hand sideways (not palm-to-camera)
- Two hands in frame

**Verification note:** The open-palm greeting is broadly consistent with ASL "Hello" (which also involves a wave from the forehead, but the static open-palm pose is used here). ⚠️ **Verify with an instructor if you want to claim ISL accuracy.**

---

### 2. THANK_YOU

| Property | Value |
|---|---|
| **Class label** | `THANK_YOU` |
| **Meaning** | Expression of gratitude |
| **Static/Dynamic** | Dynamic in real ASL (hand moves forward from chin), but captured as **static flat-palm pose** |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- All 5 fingers extended and held **together** (not spread)
- Fingers are flat and close together like a salute
- Thumb may be slightly tucked against the side of the index finger, or gently extended flat

**Palm orientation:**
- Palm facing the camera, fingers pointing **upward**

**Pose description:**  
Hold a flat hand (like a salute or a karate chop turned toward you), fingers together, palm facing the camera. Fingers point toward the ceiling.

**Natural variations (acceptable):**
- Slight angle of fingers (±10° from vertical)
- Minor separation between fingers (but NOT spread like HELLO)

**Confusion risk:**
- **HELLO** — spread vs. together fingers is the key distinction. HELLO fingers are widely spread; THANK_YOU fingers are flat and together.
- **PLEASE** — similar flat hand but PLEASE palm faces **upward** while THANK_YOU palm faces **toward camera**. This is the key separating feature.

**Collection instructions:**
- Practice the salute-like pose: flat hand, fingers together, palm facing you (the camera)
- The fingers should clearly point upward — not sideways
- The thumb position (slightly tucked vs slightly out) creates natural variation — let it vary naturally

**INVALID samples:**
- Fingers spread (that's HELLO)
- Palm facing up (that's PLEASE)
- Palm facing sideways or down

**Verification note:** Real ASL THANK_YOU moves the flat hand forward from the chin. Here we capture only the **static hand shape**. ⚠️ **Verify with an instructor.**

---

### 3. PLEASE

| Property | Value |
|---|---|
| **Class label** | `PLEASE` |
| **Meaning** | Polite request |
| **Static/Dynamic** | Dynamic in real ASL (circular motion on chest), but captured as **static palm-up flat hand** |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- All 5 fingers extended and **together** (flat hand)
- Fingers parallel to each other, no spreading

**Palm orientation:**
- Palm facing **upward** (ceiling) — this is the critical distinguishing feature from THANK_YOU and HELLO

**Pose description:**  
Hold your hand flat with fingers together, palm facing upward, like you are holding a tray or offering something. The back of the hand faces downward.

**Natural variations (acceptable):**
- Slight variation in how flat the hand is (minor finger curl is OK as long as intent is clear)
- Angle of palm from horizontal (±15°) — just ensure it is distinctly upward-facing, not toward-camera

**Confusion risk:**
- **THANK_YOU** — same flat hand, but THANK_YOU faces toward camera. The palm orientation is the ONLY distinguishing feature. Be very deliberate.
- **HELLO** — fingers are spread for HELLO; flat for PLEASE

**Collection instructions:**
- After setting the "palm up, fingers together" pose, hold it at various arm heights
- Slightly vary the exact angle (a little more tilted toward camera, then back to full palm-up)
- The key variation to capture: natural wrist micro-adjustments while keeping palm upward

**INVALID samples:**
- Palm facing camera (that's THANK_YOU territory)
- Fingers spread
- Fist or curled hand

**Verification note:** Real ASL PLEASE involves the flat hand rubbing the chest in a circular motion. ⚠️ **Verify with an instructor for ISL.**

---

### 4. YES

| Property | Value |
|---|---|
| **Class label** | `YES` |
| **Meaning** | Affirmation |
| **Static/Dynamic** | Dynamic in real sign (fist nodding), but captured as **static fist with thumb up** |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- All fingers **curled into a fist**
- Thumb extended **upward** (like a thumbs-up gesture)
- The thumb clearly points toward the ceiling

**Palm orientation:**
- Side of hand toward camera (thumb side facing up/forward)

**Pose description:**  
Make a fist, then extend your thumb straight upward. This is essentially a "thumbs up" gesture. The fist should be tight, thumb clearly up.

**Natural variations (acceptable):**
- Slight variation in fist tightness
- Thumb angle (straight up vs slightly toward camera) — as long as it's clearly pointing upward

**Confusion risk:**
- **NO** — same fist, but NO has thumb pointing **downward** instead of up. The thumb direction is the ONLY difference. Be very deliberate and exaggerated in the thumb direction.
- **SORRY** — similar fist but SORRY has thumb tucked IN and knuckles face camera

**Collection instructions:**
- Practice the clear thumbs-up pose — thumb should be unambiguously pointing up
- Vary slightly: move hand closer, farther, slight wrist rotation — but ALWAYS keep thumb up

**INVALID samples:**
- Thumb pointing sideways
- Thumb partially curled or ambiguous direction
- Open hand or any fingers extended besides thumb

**Verification note:** Thumbs-up for YES is a common simplified representation. In formal sign languages, YES is typically a nodding fist. ⚠️ **Verify with an instructor.**

---

### 5. NO

| Property | Value |
|---|---|
| **Class label** | `NO` |
| **Meaning** | Negation |
| **Static/Dynamic** | Captured as **static fist with thumb down** |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- All fingers **curled into a fist**
- Thumb extended **downward** (like a thumbs-down gesture)
- The thumb clearly points toward the floor

**Palm orientation:**
- Side of hand toward camera (thumb side facing down)

**Pose description:**  
Make a fist, then extend your thumb straight downward — a "thumbs down" gesture. Keep the fist tight and thumb clearly pointing down.

**Natural variations (acceptable):**
- Same as YES — slight variation in fist tightness and exact thumb angle, as long as it's clearly downward

**Confusion risk:**
- **YES** — only difference is thumb direction. Up = YES, Down = NO. The model must learn this from the Y coordinate of the thumb tip landmark. Make the direction unambiguous.
- **SORRY** — SORRY has thumb tucked in (no visible thumb extension)

**Collection instructions:**
- Be deliberate about the downward thumb — don't let it drift sideways
- Vary: different heights in frame, slightly closer/farther, slight wrist rotation

**INVALID samples:**
- Thumb sideways or ambiguous direction
- Any fingers extended other than thumb
- Thumb tucked in (that starts to look like SORRY)

**Verification note:** Thumbs-down for NO is a simplified representation. ⚠️ **Verify with an instructor.**

---

### 6. SORRY

| Property | Value |
|---|---|
| **Class label** | `SORRY` |
| **Meaning** | Apology |
| **Static/Dynamic** | Dynamic in real sign (circular motion over chest), captured as **static fist with knuckles facing camera** |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- All fingers **curled into a tight fist**
- Thumb tucked **inside** or pressed against the fingers (no extended thumb)
- The knuckles face toward the camera

**Palm orientation:**
- Knuckles toward camera (the back of the fist faces you)

**Pose description:**  
Make a tight fist with all fingers curled and thumb tucked in or pressed against the side. Rotate the fist so the knuckle row faces directly toward the camera (as if you would knock on a door facing the camera).

**Natural variations (acceptable):**
- Slight variation in how tight the fist is
- Thumb inside fist vs pressed against fingers (both are fine)

**Confusion risk:**
- **YES** — YES has thumb up, SORRY has thumb tucked. Make sure the thumb is clearly NOT extended.
- **NO** — same: NO has thumb down, SORRY has no extended thumb
- If the fist is rotated so the side (not knuckles) faces camera, it looks more like YES/NO territory

**Collection instructions:**
- Practice: make a fist → tuck thumb → rotate until knuckles face camera
- The knuckle-toward-camera orientation is what distinguishes this from YES/NO
- Vary: wrist angle, distance, height in frame

**INVALID samples:**
- Thumb visible and extended in any direction (that's YES or NO)
- Open palm or any fingers extended
- Side of fist facing camera rather than knuckles

**Verification note:** The specific SORRY gesture varies significantly across sign languages. This representation uses a closed fist (representing the circular motion's resting pose). ⚠️ **Verify with an instructor.**

---

### 7. HELP

| Property | Value |
|---|---|
| **Class label** | `HELP` |
| **Meaning** | Request for assistance |
| **Static/Dynamic** | Captured as **static L-shape** (thumb up + index pointing sideways) |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- **Thumb extended upward** (or slightly forward)
- **Index finger extended sideways** (pointing left or right — perpendicular to thumb)
- Middle, ring, pinky fingers **curled into palm**
- Thumb and index form an "L" shape

**Palm orientation:**
- The palm faces somewhat to the side (not directly toward camera)
- The "L" should be clearly visible as a right-angle shape

**Pose description:**  
Make an "L" shape with your hand: thumb pointing up, index finger pointing sideways (perpendicular to thumb), remaining three fingers curled in. This is the classic "L for loser" shape, which is also used as a simplified HELP representation.

**Natural variations (acceptable):**
- Exact angle of the L (thumb fully up vs slightly tilted)
- Index finger pointing to left or right depending on dominant hand
- Minor variation in how curled the three fingers are

**Confusion risk:**
- **YOU** — YOU has only the index finger extended. HELP has both thumb AND index. Make sure the thumb is clearly extended.
- **I_LOVE_YOU** — I_LOVE_YOU has thumb + index + pinky. HELP has only thumb + index (no pinky).

**Collection instructions:**
- Practice the L shape clearly — thumb and index forming a right angle
- Vary: different heights, closer/farther, slight wrist rotation

**INVALID samples:**
- Only index extended (that's YOU)
- Pinky also extended (approaches I_LOVE_YOU territory)
- Middle/ring/pinky not curled
- L shape ambiguous or unclear

**Verification note:** HELP in ASL involves lifting a closed fist on top of an open palm (two-hand sign). This project simplifies it to a one-hand static L-shape. ⚠️ **Verify with an instructor.**

---

### 8. NAME

| Property | Value |
|---|---|
| **Class label** | `NAME` |
| **Meaning** | Asking for or stating a name |
| **Static/Dynamic** | Captured as **static two-finger pose** |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- **Index finger extended**
- **Middle finger extended**
- Both held **together** (touching or nearly touching side by side)
- Ring and pinky **curled in**
- Thumb **tucked or slightly extended to side**

**Palm orientation:**
- Fingers pointing upward or slightly forward
- The two fingers should appear side by side, not spread into a "V"

**Pose description:**  
Extend your index and middle fingers held close together (not spread). The remaining fingers are curled. This is distinct from the peace/victory sign (which has fingers spread). Hold this pose with the paired fingers pointing upward.

**Natural variations (acceptable):**
- Slight natural separation between the two fingers (they don't need to touch)
- Minor angle variation of the finger pair

**Confusion risk:**
- **WATER** — WATER has THREE fingers (index + middle + ring) spread apart. NAME has TWO fingers held TOGETHER. Both count and spread are critical distinguishing features.
- Peace sign (fingers spread apart) — NAME has fingers TOGETHER; a spread peace-sign looks more like WATER

**Collection instructions:**
- Consciously hold the two fingers pressed or close together — don't let them spread
- Vary: different heights, angle of fingers, distance from camera

**INVALID samples:**
- Fingers spread apart (approaches WATER or a peace sign)
- Three fingers extended (that's WATER territory)
- Ring or pinky also extended

**Verification note:** NAME in ASL typically involves crossed index and middle fingers tapped. This project uses the two-finger static pose. ⚠️ **Verify with an instructor.**

---

### 9. WATER

| Property | Value |
|---|---|
| **Class label** | `WATER` |
| **Meaning** | Water / drinking request |
| **Static/Dynamic** | Captured as **static three-finger spread (W shape)** |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- **Index finger extended**
- **Middle finger extended**
- **Ring finger extended**
- All three clearly **spread apart** from each other (fan shape with 3 fingers)
- Pinky **curled in**
- Thumb **curled in or pressed to side**

**Palm orientation:**
- Fingers pointing upward or slightly forward
- The three fingers should form a clear "W" spread (not cramped together)

**Pose description:**  
Extend index, middle, and ring fingers and spread them apart into a fan or "W" shape. Pinky and thumb are folded down. The spread between the three fingers should be clearly visible.

**Natural variations (acceptable):**
- Exact degree of spread between fingers
- Slight angle of the hand (fingers pointing up vs slightly tilted)

**Confusion risk:**
- **NAME** — NAME has 2 fingers together. WATER has 3 fingers SPREAD. Both count (2 vs 3) AND spread (together vs apart) differ.
- Three-finger peace: if the pinky is accidentally extended, it starts to look like an open hand

**Collection instructions:**
- Actively spread the three fingers as wide as comfortable
- Make the spread unambiguous — this is the key distinguishing feature
- Keep thumb and pinky clearly curled

**INVALID samples:**
- Only 2 fingers extended (that's NAME territory)
- 4 or 5 fingers extended
- Fingers NOT spread (if three fingers are together, it's ambiguous with NAME)
- Pinky visible and extended

**Verification note:** WATER in ASL involves a "W" handshape (three fingers extended) tapped on the chin. This project captures the W handshape statically. ✓ The three-finger W handshape is broadly consistent with ASL WATER.

---

### 10. EAT

| Property | Value |
|---|---|
| **Class label** | `EAT` |
| **Meaning** | Eating / food / hungry |
| **Static/Dynamic** | Captured as **static pinch/claw pose** |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- All 5 fingertips **curled toward the thumb** (pinch gesture)
- Fingers form a "claw" or "pinch" shape
- Fingertips of index, middle, ring, pinky all touch or nearly touch the thumb tip
- No fingers straight or extended

**Palm orientation:**
- Side or front view — the pinch shape should be clearly visible
- Fingers pointing slightly forward or upward

**Pose description:**  
Bring all fingertips together toward the thumb as if pinching something or bringing food to your mouth. The hand should look like a cluster of fingertips meeting at a central point. This is sometimes called the "Italian hand" or "pinch" gesture.

**Natural variations (acceptable):**
- Exact tightness of the pinch (touching vs nearly touching)
- Slight variation in palm angle

**Confusion risk:**
- This is one of the most unique shapes in the vocabulary — minimal confusion risk
- Could potentially be confused with a partial fist (SORRY/YES/NO) if pinch is too tight; ensure the pinch shape (finger**tips** together, not knuckles forward) is distinct

**Collection instructions:**
- Practice: spread fingers → curl all fingertips toward thumb → form the pinch
- Vary: slightly looser/tighter pinch, different angles

**INVALID samples:**
- Full fist (all fingers curled but NOT meeting at tips — this looks like YES/NO/SORRY)
- Any fingers extended straight

**Verification note:** EAT in ASL uses this pinch shape moved toward the mouth. ✓ The hand shape is broadly standard.

---

### 11. I_LOVE_YOU

| Property | Value |
|---|---|
| **Class label** | `I_LOVE_YOU` |
| **Meaning** | "I love you" |
| **Static/Dynamic** | Static gesture — this is already a standard static sign in ASL |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- **Thumb extended** (to side)
- **Index finger extended** (upward)
- **Pinky finger extended** (upward or slightly outward)
- **Middle and ring fingers curled down** (pressed to palm)
- The three extended digits (thumb, index, pinky) form the ILY shape

**Palm orientation:**
- Palm facing toward camera
- Thumb pointing sideways, index and pinky pointing upward

**Pose description:**  
Extend your thumb, index finger, and pinky finger simultaneously, while folding your middle and ring fingers down. This is the universally recognized "I Love You" hand symbol, combining I (pinky), L (index+thumb), and Y (thumb+pinky).

**Natural variations (acceptable):**
- Slight variation in exact angle of the three extended digits
- Thumb may point slightly forward vs directly to the side

**Confusion risk:**
- **HELP** — HELP has thumb + index (no pinky). I_LOVE_YOU has thumb + index + pinky. The pinky extension is the key difference.
- **YOU** — YOU has only index finger. I_LOVE_YOU has three digits.
- If the middle/ring fingers are not clearly folded, may blend with HELLO or WATER

**Collection instructions:**
- Practice: make a fist → extend thumb, index, pinky simultaneously → press middle and ring down firmly
- Ensure middle and ring are clearly pressed down — ambiguity here causes confusions
- Vary: distance, slight wrist rotation, hand height

**INVALID samples:**
- Middle or ring also extended (becomes HELLO territory)
- Only index and pinky (no thumb) — different gesture
- Pinky not extended (becomes HELP)

**Verification note:** ✓ I Love You in ILY handshape is universally standardized in ASL and broadly recognized.

---

### 12. YOU

| Property | Value |
|---|---|
| **Class label** | `YOU` |
| **Meaning** | Pointing at another person |
| **Static/Dynamic** | Static — just a pointing gesture |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 |

**Finger configuration:**
- **Index finger extended and pointing** forward or slightly toward camera
- All other fingers (middle, ring, pinky) **curled in**
- Thumb **tucked alongside** the curled fingers or slightly raised alongside

**Palm orientation:**
- Side of hand or slightly angled, with index finger pointing forward/slightly toward camera

**Pose description:**  
Simply point your index finger forward as if pointing at someone. All other fingers are folded down. This is the most natural and intuitive gesture in the vocabulary.

**Natural variations (acceptable):**
- Index finger angle (pointing slightly up, forward, or to the side — as long as it's clearly the pointing gesture)
- Exact thumb position

**Confusion risk:**
- **HELP** — HELP has thumb ALSO extended. YOU has only index. Make sure the thumb is clearly tucked.
- **I_LOVE_YOU** — ILY has index + thumb + pinky. YOU has only index.
- Be deliberate about keeping the thumb down for YOU

**Collection instructions:**
- Practice: make a fist → extend only the index finger → point it forward
- Keep thumb clearly tucked/folded
- Vary: different pointing angles, hand heights, distances

**INVALID samples:**
- Thumb also extended (becomes HELP territory)
- Pinky also extended (approaches ILY territory)
- Two or more fingers extended

**Verification note:** ✓ Pointing with index finger is a near-universal deictic gesture and consistent with sign language conventions for YOU.

---

### 13. _NEUTRAL

| Property | Value |
|---|---|
| **Class label** | `_NEUTRAL` |
| **Meaning** | **Rejection class** — no sign is being made |
| **Static/Dynamic** | Various idle hand states |
| **Hands required** | Single hand |
| **Expected hands in frame** | 1 (or 0 if no hand present — but the system only records when a hand IS detected) |

**Why this class exists:**  
Without a neutral class, the model is a **closed-world classifier** — it must assign every frame to one of the 12 sign classes. If you hold your hand out idly between signs, the model would produce constant false predictions. The `_NEUTRAL` class allows the model to say *"a hand is present but no recognizable sign is being made."*

**What to capture for _NEUTRAL:**
- Relaxed, partly-curled hand (like when your hand hangs naturally)
- Flat hand at odd angles that don't match any sign
- Hand transitioning between poses (mid-movement)
- Hand at the side (if visible)
- Slightly open fist or loosely held hand
- Any hand pose that is clearly NOT one of the 12 defined signs

**Natural variations (MAXIMIZE variation here):**  
Unlike the other 11 classes where variation should be natural but controlled, **_NEUTRAL should be deliberately varied** to cover as many "not a sign" states as possible:
- Relaxed hanging hand
- Slightly curled fingers (not a fist, not a pinch)
- Flat hand tilted at odd angles
- Random intermediate states

**CRITICAL:** Do NOT include any of the 12 defined sign poses in _NEUTRAL. The boundary between a relaxed hand and SORRY (fist) or PLEASE (flat palm) should be maintained.

**Collection instructions:**
- Collect ~4-6 sessions of genuinely idle hand poses
- Try to cover diverse neutral states — not just one type of idle hand
- Include transitions between signs (hands moving, mid-pose)

**INVALID samples for _NEUTRAL:**
- Any pose that clearly matches one of the 12 sign definitions
- Completely straight rigid hand (might train the model to avoid a useful neutral state)

---

## Class Summary Table

| # | Label | Meaning | Key feature | Static pose? | Confusion with |
|---|---|---|---|---|---|
| 1 | `HELLO` | Greeting | 5 fingers spread, palm toward camera | ✅ | THANK_YOU |
| 2 | `THANK_YOU` | Gratitude | Flat hand together, palm toward camera | ✅ | HELLO, PLEASE |
| 3 | `PLEASE` | Request | Flat hand together, palm upward | ✅ | THANK_YOU |
| 4 | `YES` | Affirmation | Fist + thumb up | ✅ | NO, SORRY |
| 5 | `NO` | Negation | Fist + thumb down | ✅ | YES, SORRY |
| 6 | `SORRY` | Apology | Fist, thumb tucked, knuckles toward camera | ✅ | YES, NO |
| 7 | `HELP` | Assistance | L-shape (thumb up + index sideways) | ✅ | YOU, I_LOVE_YOU |
| 8 | `NAME` | Name | 2 fingers (index+middle) together, upright | ✅ | WATER |
| 9 | `WATER` | Water | 3 fingers (index+middle+ring) spread (W) | ✅ | NAME |
| 10 | `EAT` | Eating | All fingertips pinched toward thumb | ✅ | (minimal) |
| 11 | `I_LOVE_YOU` | Love | Thumb + index + pinky extended | ✅ | HELP, YOU |
| 12 | `YOU` | You (pointing) | Index only, extended | ✅ | HELP, I_LOVE_YOU |
| 13 | `_NEUTRAL` | No sign / idle | Any non-sign hand state | N/A | All 12 signs |

---

## Most Likely Confusion Pairs (Ranked by Risk)

| Rank | Pair | Why confusing | Prevention |
|---|---|---|---|
| 1 | YES ↔ NO | Identical fist, only thumb direction differs | Exaggerate thumb direction during collection |
| 2 | THANK_YOU ↔ PLEASE | Same flat hand, only palm direction differs | Be deliberate about palm-toward vs palm-up |
| 3 | HELLO ↔ THANK_YOU | Similar open hand, spread vs. together | Ensure fingers are maximally spread for HELLO |
| 4 | NAME ↔ WATER | 2 vs 3 fingers; together vs spread | 3-finger spread must be very clear for WATER |
| 5 | HELP ↔ I_LOVE_YOU | L-shape vs L+pinky | Keep pinky clearly down for HELP |
| 6 | HELP ↔ YOU | Thumb+index vs index only | Keep thumb clearly visible for HELP |
| 7 | YES/NO ↔ SORRY | Thumb up/down vs thumb tucked | Tuck thumb completely for SORRY |

---

*Document version: v1.0 — Updated: 2026-09-23*  
*All sign descriptions represent the specific gesture definitions used in this project's training data.*  
*These are NOT guaranteed to be universally standardized sign language representations.*
