# Five ways a webcam heart-rate monitor lies to you

*Bugs found in the `face-mesh` medical branch by pointing the code at a real
face instead of a synthetic test — and what each one teaches about building
medical-adjacent software.*

---

## TL;DR

The `medical` branch estimates heart rate, respiration and pupillary reflex from
nothing but a webcam (remote photoplethysmography, rPPG). It passed 181 synthetic
unit tests. Then I ran it on my own face and it reported a heart rate **double**
my real one, at 66% "confidence."

Five distinct bugs surfaced — none caught by the test suite, every one caught by
a single live recording. Two of them produced *confident, clinical-looking
numbers that were simply wrong*, which is the most dangerous failure mode a
health tool can have. This is the story of the five, in the order they had to be
fixed, because you cannot verify a fix against a polluted baseline.

The headline lesson: **synthetic tests prove the code does what you told it to.
They cannot tell you that what you told it to do is wrong.** Only real signal
does that.

---

## The setup

rPPG works because blood absorbs green light. Each heartbeat pushes blood into
the skin of the face, so the average green value of a cheek region rises and
falls at your pulse rate. Extract that region over time, band-pass it to plausible
heart-rate frequencies, take the spectrum, and the tallest peak *should* be your
pulse.

"Should" is doing a lot of work in that sentence.

---

## Bug #2 — the harness fed bad frames into its own estimator

*(Fixed first, because nothing downstream can be trusted until it is.)*

The pipeline computes a per-frame quality index and has a gate that is supposed
to drop low-quality frames before they reach the estimator. The harness computed
the gate value, logged it, and then handed the estimator **every** frame anyway.

On my recording, **479 of 931 frames** were below the quality threshold — more
than half — and all 479 went into the spectrum. The gate existed; it just wasn't
wired to anything.

**Why the tests missed it:** synthetic frames are uniformly clean, so the gate
never had a reason to fire, so nobody noticed the gate's output went nowhere.

**Fix:** actually filter on the gate before estimation, and record the drop count
so a session with mostly-rejected frames is visibly untrustworthy rather than
silently averaged.

**Lesson:** a quality gate you compute but don't enforce is worse than no gate —
it looks like a safeguard in code review while doing nothing.

---

## Bug #3 — no rejection of uneven lighting

Once bad frames were actually being dropped, a second class of garbage remained:
**406 frames with uneven lighting**. A shadow drifting across one cheek creates a
slow brightness ramp that lives in the same low frequencies as respiration and
drags the heart-rate band around with it. The estimator had no notion that a
lit-from-one-side face is unmeasurable.

**Fix:** a lighting-uniformity check that rejects frames where the region's
illumination gradient exceeds a threshold, sitting alongside the quality gate.

**Lesson:** "signal quality" is not one number. A frame can be sharp, in-focus,
well-tracked — and still be physically incapable of yielding a pulse because the
light is wrong.

---

## Bug #1 — harmonic locking (the double heart rate)

This is the one my own body exposed. I have a low resting heart rate. The
estimator reported roughly **twice** my true rate, confidently.

The cause is *harmonic locking*. A periodic pulse signal has energy not only at
its fundamental frequency `f` but also at `2f`, `3f`… For most people the
fundamental is the tallest peak and everything is fine. But when the fundamental
is low and the second harmonic falls near the centre of the search band, `2f` can
edge out `f` as the tallest peak — and the estimator happily reports `2f` as your
heart rate.

A subject with a 70 bpm pulse would **never** have surfaced this. Their
fundamental sits comfortably mid-band; their second harmonic is up near the band
edge and loses. It took an out-of-distribution subject — me — for the bug to
appear at all.

**Fix:** a sub-harmonic guard. Before accepting a peak, check whether a strong
peak exists at half its frequency; if it does and it's plausible, prefer the
fundamental. Combined with the now-enforced quality gate, the reported rate
matched my real pulse.

**Lesson:** your test population *is* your test coverage. Everything that passed
did so because the synthetic subject was average. The interesting failures live
in the tails, and if you never sample the tails you will ship those failures.

---

## Bug #4 — respiration reported at a subharmonic

Respiration is estimated from the slow amplitude modulation of the pulse signal.
The pipeline reported **8 breaths per minute**. Normal adult resting respiration
is 12–20.

8 is almost exactly **16 / 2**. The estimator had locked onto a subharmonic of
the true respiration rate — the mirror image of the heart-rate harmonic bug, one
octave down instead of up. A drifting baseline in the respiration band made every
*other* breath look like the dominant cycle.

**Fix:** the same family of correction — detrend the respiration band and add a
subharmonic sanity check against the physiological range, so 8/min gets re-examined
rather than reported.

**Lesson:** the same bug class recurs at every frequency band in a signal
pipeline. Fixing it once for heart rate does not fix it for respiration; each
band needs its own guard.

---

## Bug #5 — a pupillary reflex measured from a signal the method cannot see

The pupillary light reflex (PLR) is the pupil constricting when light hits the
eye — a fast response, latency **0.2–0.3 s**. The branch reported latencies of
**1.03 s and 1.54 s**, with the two eyes disagreeing by half a second.

Those numbers aren't a mismeasured reflex. They are **noise wearing the shape of
a reflex.** A webcam's spatial resolution on the iris, at normal sitting distance,
is too coarse to resolve sub-millimetre pupil-diameter changes on the timescale a
real PLR happens. The method physically cannot see the thing it was reporting a
number for — and it reported one anyway, formatted to two decimal places.

This is the most dangerous bug in the set, and the least like a bug. Everything
"worked": a signal went in, a plausible-looking latency came out. There was no
crash, no exception, no obviously-wrong value like a 300 bpm heart rate. Just a
confident measurement of nothing.

**Fix:** the PLR path now refuses to report a latency unless the pupil signal
clears a physical detectability bar (contrast, sampling rate, inter-eye
agreement). Below that bar it returns *no measurement* — which is the honest
answer — instead of a number.

**Lesson:** the failure that formats nicely is more dangerous than the one that
crashes. A tool that says "I can't measure this" is safe. A tool that fabricates
a clinical value from noise is exactly how bad medical software hurts people.

---

## What the whole thing taught me

1. **Synthetic tests validate logic, not premises.** All five bugs lived in the
   gap between "the code does what I specified" and "what I specified is
   physically true." Tests can't cross that gap; only real signal can.

2. **Out-of-distribution subjects are a feature.** My low heart rate wasn't an
   inconvenience — it was the single most valuable test case in the project.

3. **The scariest bug is the one that returns a pretty number.** Crashes get
   fixed because they're loud. Confident wrong answers ship because they're quiet.

4. **"I can't measure this" is a valid, and often the correct, output.** Half of
   fixing these bugs was teaching the code to *decline* rather than guess.

The five fixes landed as five separate commits, each closing its issue, each with
a regression test. The test suite grew accordingly. But the tests are not the
point of this writeup — the point is that all 181 of them were green while the
tool was reporting my heart rate as twice its real value. The bugs were only ever
going to be found by running the thing on a face.

> **Reminder, as everywhere in this branch:** this is not a medical device and
> nothing here is clinically validated. The value of the work is the failure
> analysis, not the measurements.
