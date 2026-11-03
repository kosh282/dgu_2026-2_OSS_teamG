# Clinical-style facial metrics

> ## ⚠️ Screening and educational use only
>
> **This is not a diagnostic tool and not a medical device.** None of these
> metrics are clinically validated. They reproduce the *geometry and signal
> processing* used in published research, not the validation, calibration, or
> regulatory work that a real clinical instrument requires.
>
> Published facial-landmark models are trained predominantly on healthy faces
> and are known to lose accuracy on faces affected by neuromuscular disease —
> which biases them against exactly the populations these metrics concern.
>
> Never use this to make a health decision. Consult a qualified clinician.

```bash
facemesh --medical --task rest      # all metrics, labelled task
facemesh --symmetry                 # asymmetry index only
facemesh --movement --task smile    # expressivity, interpretable
facemesh --heart-rate               # rPPG only
```

Results appear in an on-screen panel and, with `--export`, under a `medical`
key in the per-frame record.

---

## 0. Quality gating (applies to everything)

**Confidently wrong output is worse than no output.** Before any metric is
reported, `QualityGate` checks that the face is large enough, roughly frontal,
adequately and evenly lit, and not moving too fast. If any check fails, metrics
are **suppressed** and the panel says why (`Metrics paused: too dark`).

| Check | Default threshold |
|---|---|
| Face height | ≥ 18% of frame height |
| Head pose | ≤ 15° pitch/yaw/roll (needs the pose matrix) |
| Brightness | 0.15 – 0.92 of full scale |
| Left/right lighting imbalance | ≤ 0.25 |
| Landmark speed | ≤ 0.60 inter-ocular units/s |

Note motion is measured in *image* space, not the face-local frame — that frame
removes rigid motion by construction, so measuring there would always read zero.

---

## 1. Facial symmetry index

**What it measures.** Mirrored landmark pairs compared across the mid-sagittal
plane, per region (eye, brow, mouth, cheek). Perfect symmetry scores `1.0`.

**Why it matters.** Landmark asymmetry is the geometric core of facial-palsy
grading and post-stroke facial assessment.

**Three refinements over a naive score:**

**3D by default.** Deviation is measured in a 3D face-local frame. A purely
out-of-plane droop (a lip pulling *backwards*) reads `1.000` — completely
invisible — in a 2D score, but `0.468` in 3D. Pass `use_3d=False` to reproduce
the flat behaviour.

**Per-person baselines.** Most people have mild natural asymmetry, so an
absolute score is weakly informative. `SymmetryBaseline` records a person's own
resting asymmetry and reports *change from their normal*. In testing, a face
with natural asymmetry scores `0.878` absolute but `1.000` relative to itself,
while an acquired droop on the same face drops to `0.558` relative.

**Peak-on-movement.** Palsy often hides at rest and appears on movement, so
`DynamicSymmetry` reports the *peak* asymmetry over a window — a transient
one-sided smile that averages away is captured by the peak.

```python
from facemesh.medical import symmetry_metrics, SymmetryBaseline
m = symmetry_metrics(face_landmarks, w, h)          # 3D by default
# {'eye_asym':..., 'symmetry': 0.93, 'worst_region': 'mouth'}

bl = SymmetryBaseline()
for neutral_frame in resting_samples:
    bl.add_sample(symmetry_metrics(neutral_frame, w, h))
bl.fit()
bl.compare(m)   # {'relative_symmetry': 0.97, 'worst_delta_region': 'mouth'}
```

**Limits.** A low score means "look closer," not "pathology."

---

## 2. Facial movement / expressivity

**What it measures.** Over a rolling window: how far expressive landmarks range
(amplitude), how much they move frame-to-frame (jitter), and how much motion
power sits in the **4–6 Hz Parkinsonian tremor band**.

**Why it matters.** Reduced facial movement — *hypomimia*, the "masked face" —
is a hallmark of Parkinson's disease. Expression amplitude and tremor of small
muscle groups are the features used in the literature.

**Context is mandatory.** A resting face is legitimately still; passive
observation cannot distinguish "calm" from "hypomimic." Expressivity is only
meaningful relative to a *task*. Set one with `--task smile` (or
`analyzer.set_task(...)`); without it, results carry `interpretable: False` and
the panel says `(no task set)`. `task_summary()` reports peak expressivity per
task across a session.

**Tremor detection subtlety.** The spectrum is computed from the *signed*
landmark-centroid position series, **not** from jitter magnitude — rectifying a
5 Hz sinusoid doubles it to 10 Hz and would push real tremor straight out of the
band. A synthetic 5 Hz tremor is recovered at 5.04 Hz with ratio 1.00.

**Verified property.** A frozen face that translates, rotates 20° and changes
scale registers **zero** expressivity: it measures the face, not the head.

```python
from facemesh.medical import MovementAnalyzer
ma = MovementAnalyzer(window_s=5.0)
ma.set_task("smile")
m = ma.update(face_landmarks, w, h, now_s)
# {'expressivity': 0.71, 'tremor_hz': 0.0, 'interpretable': True, 'task': 'smile'}
```

---

## 3. Contactless heart rate, HRV and respiration (rPPG)

**What it measures.** Beats per minute, heart-rate variability (SDNN, RMSSD),
and breathing rate — from the camera alone.

**Why it works.** Blood volume changes with each heartbeat subtly alter how skin
reflects light. Remote photoplethysmography recovers that pulse from facial
video; the waveform carries heart rate, HRV, and (in its low-frequency envelope)
respiration. rPPG has been piloted in real clinical settings including
non-contact monitoring in neonatal intensive care.

**How it's computed.**

1. **Skin masking** — a filled polygon from the landmark contour (forehead, both
   cheeks), further restricted by a **YCrCb skin-chrominance test** and a
   luminance gate. This removes eyebrows, hair, glasses, shadows and
   blown-out highlights that a bounding box would include. In testing, dark hair
   over the forehead cuts the mask from 1770 to 792 px — those pixels carried no
   pulse and only added noise.
2. **Motion rejection** — frames whose landmark speed exceeds a threshold are
   discarded entirely, because motion artifacts dominate real-world rPPG error.
   Violent jitter rejects 59 of 60 frames; a still face rejects none.
3. **POS** (Plane-Orthogonal-to-Skin, Wang et al. 2017) projects normalized RGB
   onto a plane orthogonal to the skin-tone direction to isolate the pulse.
4. The dominant frequency in 42–180 bpm is the heart rate; spectral peak
   prominence gives a confidence value. HRV comes from peak-picking the
   bandpassed waveform; respiration from the 6–30 bpm band.

```python
from facemesh.medical import HeartRateEstimator
hr = HeartRateEstimator(window_s=10.0)
m = hr.update(frame_bgr, face_landmarks, now_s)
# {'bpm': 72.4, 'confidence': 0.61, 'reliable': True,
#  'sdnn_ms': 42.1, 'rmssd_ms': 38.7, 'respiration_rpm': 14.0}
```

### Accuracy — be honest about this

On **synthetic** signals with known ground truth, this implementation recovers
54–120 bpm to within ~2 bpm, HRV variability correctly, and respiration to
within 0.4 rpm. **That proves the signal processing is correct and says nothing
about real-world accuracy.**

Real-world error is unknown until *you* measure it. Use the harness:

```bash
python scripts/validate_rppg.py record --seconds 60 --out session.jsonl
python scripts/validate_rppg.py report session.jsonl --reference 68 71 70
```

It prints MAE, RMSE, bias and 95% limits of agreement against your reference
(pulse oximeter, smartwatch, or a manual 30-second count doubled). **Put the
measured number in the README.** A stated error beats an unstated one.

**Limits.** Motion artifacts dominate. Sit still, use even frontal lighting,
avoid backlighting. Respect `reliable: False` — that flag is the system telling
you the truth rather than a pretty number.

---

## Testing

The metrics are pure functions of landmark arrays and pixel buffers, so they run
without a camera:

```bash
pytest tests/test_medical.py tests/test_medical_quality.py -q   # 36 tests
```

Coverage includes: rigid-motion invariance in 2D and full 3D; 3D detecting a
depth droop that 2D provably misses; baselines separating natural from acquired
asymmetry; peak-vs-mean on transient movements; tremor-band isolation and the
signed-series requirement; task-context gating; skin-mask rejection of hair and
non-skin chrominance; HRV and respiration recovery; motion rejection; and every
quality-gate failure condition.

## References

Approaches follow published work on landmark-based facial palsy and stroke
assessment, facial expression analysis for Parkinson's disease, and remote
photoplethysmography (Wang et al., *Algorithmic Principles of Remote PPG*, 2017,
for POS). This is an educational reproduction of the underlying signal
processing, not of any validated clinical protocol.
