"""Remote photoplethysmography (rPPG) — contactless heart rate from the face.

The mesh locates skin regions (forehead + both cheeks). Instead of a crude
bounding box, an exact **polygon mask** is filled from the landmark contour and
further restricted by a **skin-tone test**, so eyebrows, hair, eyes, glasses and
background do not dilute the pulse signal.

Frames captured during rapid head motion are **rejected**, because motion
artifacts dominate real-world rPPG error. Remaining frames feed the POS
(Plane-Orthogonal-to-Skin, Wang et al. 2017) algorithm, which projects
normalized RGB onto a plane orthogonal to the skin-tone direction to isolate the
pulse. The dominant frequency in 42-180 bpm is the heart rate; spectral peak
prominence gives a confidence value.

SCREENING / EDUCATIONAL USE ONLY — not a medical device.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

# Closed contours (in landmark order) bounding each skin patch.
FOREHEAD = (67, 109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69)
LEFT_CHEEK = (330, 347, 346, 352, 411, 425, 266, 371)
RIGHT_CHEEK = (101, 118, 117, 123, 187, 205, 36, 142)

MIN_BPM, MAX_BPM = 36.0, 180.0
#: If the peak at f/2 carries at least this fraction of the peak at f, the
#: lower frequency is the fundamental and f is its second harmonic.
HARMONIC_RATIO = 0.25

#: Respiration search band (breaths per minute). Below ~9/min at rest is
#: implausible outside sleep or pathology, and that region is dominated by
#: postural/lighting wander rather than breathing.
MIN_RPM, MAX_RPM = 9.0, 30.0
#: High-pass cutoff applied before the respiration transform.
RESP_HIGHPASS_RPM = 11.0
#: If the peak at 2f carries at least this fraction of the peak at f, then f is
#: a subharmonic and 2f is the true breathing rate.
SUBHARMONIC_RATIO = 0.4


def skin_mask(frame_bgr, landmarks, contours=(FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK)) -> np.ndarray:
    """Boolean mask of skin pixels inside the given landmark contours.

    Combines a filled polygon mask with a YCrCb skin-chrominance test, which
    removes eyebrows, hair, shadows and specular highlights that fall inside
    the polygon.
    """
    h, w = frame_bgr.shape[:2]
    poly = np.zeros((h, w), np.uint8)
    for contour in contours:
        pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in contour],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(poly, cv2.convexHull(pts), 255)

    # Skin chrominance gate (standard Cr/Cb ranges).
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin = (cr >= 133) & (cr <= 180) & (cb >= 77) & (cb <= 127)

    # Drop very dark / blown-out pixels: they carry no usable pulse.
    lum = ycrcb[:, :, 0]
    usable = (lum > 40) & (lum < 245)

    return (poly > 0) & skin & usable


def skin_rgb(frame_bgr, landmarks) -> np.ndarray | None:
    """Mean RGB of masked skin pixels, or ``None`` if too few pixels survive."""
    mask = skin_mask(frame_bgr, landmarks)
    n = int(mask.sum())
    if n < 100:                       # not enough skin to average reliably
        return None
    pixels = frame_bgr[mask]          # (n, 3) BGR
    return pixels.mean(axis=0)[::-1]  # -> RGB


def lighting_quality(frame_bgr, landmarks) -> dict:
    """Brightness and left/right imbalance of the masked skin pixels.

    rPPG recovers a pulse from sub-percent changes in skin reflectance, so
    uneven or clipped illumination swamps the signal far more effectively than
    blood volume modulates it. Measuring on the *masked skin* (not a bounding
    box) keeps hair and background out of the statistic.
    """
    h, w = frame_bgr.shape[:2]
    mask = skin_mask(frame_bgr, landmarks)
    n = int(mask.sum())
    if n < 100:
        return {"brightness": 0.0, "imbalance": 1.0, "clipped": 1.0, "pixels": n}

    lum = frame_bgr.mean(axis=2)
    vals = lum[mask]
    brightness = float(vals.mean() / 255.0)
    clipped = float(((vals < 16) | (vals > 245)).mean())

    # Split on the facial midline (mean x of the masked pixels).
    xs = np.nonzero(mask)[1]
    mid = int(xs.mean())
    left = mask.copy()
    left[:, mid:] = False
    right = mask.copy()
    right[:, :mid] = False
    lm, rm = lum[left], lum[right]
    if lm.size < 20 or rm.size < 20:
        imbalance = 0.0
    else:
        a, b = float(lm.mean()), float(rm.mean())
        imbalance = abs(a - b) / max(a + b, 1e-6)
    return {"brightness": brightness, "imbalance": float(imbalance),
            "clipped": clipped, "pixels": n}


def pos_signal(rgb: np.ndarray) -> np.ndarray:
    """POS (Plane-Orthogonal-to-Skin) pulse signal from an ``(T, 3)`` RGB trace."""
    mean = rgb.mean(axis=0)
    mean[mean == 0] = 1e-9
    Cn = rgb / mean                          # temporal normalization
    S1 = Cn[:, 1] - Cn[:, 2]                 # G - B
    S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]  # G + B - 2R
    s2_std = S2.std()
    alpha = (S1.std() / s2_std) if s2_std > 1e-12 else 0.0
    h = S1 + alpha * S2
    return h - h.mean()


def _band_power(freqs: np.ndarray, spec: np.ndarray, centre: float,
                width: float = 4.0) -> float:
    """Total spectral power within +/- ``width`` bpm of ``centre``."""
    sel = (freqs >= centre - width) & (freqs <= centre + width)
    return float(spec[sel].sum()) if sel.any() else 0.0


def estimate_bpm(signal: np.ndarray, fps: float,
                 correct_harmonic: bool = True) -> tuple[float, float]:
    """Return ``(bpm, confidence)`` from a pulse signal via its spectrum.

    A physiological pulse has a sharp systolic upstroke, so its spectrum carries
    substantial energy at the **second harmonic**. When the harmonic is as
    strong as the fundamental — which happens most readily at low heart rates,
    where the fundamental sits near the low edge of the search band and is
    attenuated by windowing — a naive "take the biggest peak" estimator reports
    **double the true rate** with high confidence.

    That is not hypothetical: a real session from a subject with a resting rate
    of 46 bpm was reported as 83-84 bpm (issue #1). This function therefore
    checks whether significant power exists at ``f/2`` and, if so, prefers the
    fundamental.

    Args:
        correct_harmonic: Disable to reproduce the old (buggy) behaviour.
    """
    n = len(signal)
    if n < 32 or fps <= 0:
        return 0.0, 0.0
    windowed = signal * np.hanning(n)
    nfft = max(n, 1024)                    # finer resolution at the low edge
    spec = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps) * 60.0   # bpm

    band = (freqs >= MIN_BPM) & (freqs <= MAX_BPM)
    if not band.any() or spec[band].sum() <= 0:
        return 0.0, 0.0
    in_band = spec[band]
    band_freqs = freqs[band]
    peak = int(np.argmax(in_band))
    bpm = float(band_freqs[peak])

    if correct_harmonic:
        half = bpm / 2.0
        if half >= MIN_BPM:
            p_peak = _band_power(freqs, spec, bpm)
            p_half = _band_power(freqs, spec, half)
            if p_peak > 0 and (p_half / p_peak) >= HARMONIC_RATIO:
                # f/2 is the fundamental; the selected peak was its harmonic.
                bpm = half

    lo = int(np.argmin(np.abs(band_freqs - bpm)))
    a, b = max(lo - 2, 0), min(lo + 3, len(in_band))
    confidence = float(in_band[a:b].sum() / in_band.sum())
    return bpm, confidence


def inter_beat_intervals(signal: np.ndarray, fps: float, bpm: float) -> np.ndarray:
    """Inter-beat intervals (seconds) from the filtered pulse waveform."""
    if bpm <= 0 or len(signal) < 32:
        return np.array([])
    filtered = _bandpass(signal, fps, max(bpm - 25, MIN_BPM), min(bpm + 25, MAX_BPM))
    min_gap = int(fps * 60.0 / (bpm * 1.6))  # refractory: no beats closer than this
    peaks = []
    for i in range(1, len(filtered) - 1):
        if filtered[i] > filtered[i - 1] and filtered[i] >= filtered[i + 1] and filtered[i] > 0:
            if not peaks or (i - peaks[-1]) >= max(min_gap, 1):
                peaks.append(i)
    if len(peaks) < 3:
        return np.array([])
    return np.diff(np.array(peaks)) / fps


def hrv_metrics(ibis: np.ndarray) -> dict:
    """Standard time-domain HRV from inter-beat intervals (seconds)."""
    if len(ibis) < 2:
        return {"sdnn_ms": 0.0, "rmssd_ms": 0.0, "beats": int(len(ibis))}
    ms = ibis * 1000.0
    sdnn = float(np.std(ms, ddof=1))
    rmssd = float(np.sqrt(np.mean(np.diff(ms) ** 2)))
    return {"sdnn_ms": round(sdnn, 1), "rmssd_ms": round(rmssd, 1), "beats": int(len(ibis)) + 1}


def _hp_coefficients(fs: float, fc_hz: float, q: float = 0.7071):
    """RBJ cookbook second-order high-pass biquad coefficients."""
    w0 = 2 * np.pi * fc_hz / fs
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2 * q)
    b = np.array([(1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2])
    a = np.array([1 + alpha, -2 * cos_w0, 1 - alpha])
    return b / a[0], a / a[0]


def _biquad(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    for n in range(len(x)):
        x1 = x[n - 1] if n >= 1 else 0.0
        x2 = x[n - 2] if n >= 2 else 0.0
        y1 = y[n - 1] if n >= 1 else 0.0
        y2 = y[n - 2] if n >= 2 else 0.0
        y[n] = b[0] * x[n] + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
    return y


def highpass(signal: np.ndarray, fps: float, cutoff_rpm: float,
             order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth-style high-pass, pure NumPy.

    A brick-wall FFT high-pass is *not* sufficient here: at a 20 s window the
    bin spacing is ~3/min, so a strong drift component at ~8/min falls between
    bins and leaks a large tail past the band edge. Cascaded biquads applied
    forward and backward give real stopband attenuation (measured: gain 0.04 at
    7.8/min, 0.67 at 16/min) without pulling in SciPy.
    """
    x = np.asarray(signal, dtype=float)
    if len(x) < 9 or fps <= 0:
        return x - x.mean()
    b, a = _hp_coefficients(fps, cutoff_rpm / 60.0)
    y = x.copy()
    for _ in range(order):
        y = _biquad(y, b, a)
        y = _biquad(y[::-1], b, a)[::-1]
    return y


def _bandpass(signal: np.ndarray, fps: float, lo_bpm: float, hi_bpm: float) -> np.ndarray:
    """Zero-phase FFT bandpass, used before peak-picking for HRV."""
    n = len(signal)
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps) * 60.0
    spec[(freqs < lo_bpm) | (freqs > hi_bpm)] = 0
    return np.fft.irfft(spec, n=n)


def estimate_respiration(signal: np.ndarray, fps: float,
                         correct_subharmonic: bool = True) -> tuple[float, float]:
    """Breaths per minute from the low-frequency envelope.

    Two failure modes are corrected (issue #4):

    1. **Drift.** Slow postural or lighting wander (~0.1 Hz, i.e. ~6/min) sits
       inside a naive 6-30/min search band and can dominate the true breathing
       peak. The signal is high-passed before the transform, and the band now
       starts at :data:`MIN_RPM` = 9.
    2. **Subharmonics.** If every other breath is deeper, energy appears at
       ``f/2``. Preferring the larger peak then reports half the true rate. When
       significant power exists at ``2f``, that is taken as the fundamental.

    A real session reported 8 breaths/min for a healthy adult at rest, where
    12-20 is normal.

    Args:
        correct_subharmonic: Disable to reproduce the old (buggy) behaviour.
    """
    n = len(signal)
    if n < 64 or fps <= 0:
        return 0.0, 0.0

    filtered = highpass(signal, fps, RESP_HIGHPASS_RPM)
    windowed = (filtered - filtered.mean()) * np.hanning(n)
    nfft = max(n, 2048)
    spec = np.abs(np.fft.rfft(windowed, n=nfft)) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fps) * 60.0

    band = (freqs >= MIN_RPM) & (freqs <= MAX_RPM)
    if not band.any() or spec[band].sum() <= 0:
        return 0.0, 0.0
    in_band = spec[band]
    band_freqs = freqs[band]
    peak = int(np.argmax(in_band))
    rpm = float(band_freqs[peak])

    if correct_subharmonic:
        double = rpm * 2.0
        if double <= MAX_RPM:
            p_peak = _band_power(freqs, spec, rpm, width=1.5)
            p_double = _band_power(freqs, spec, double, width=1.5)
            if p_peak > 0 and (p_double / p_peak) >= SUBHARMONIC_RATIO:
                rpm = double

    lo = int(np.argmin(np.abs(band_freqs - rpm)))
    a, b = max(lo - 1, 0), min(lo + 2, len(in_band))
    conf = float(in_band[a:b].sum() / in_band.sum())
    return rpm, conf


class HeartRateEstimator:
    """Sliding-window rPPG estimator with skin masking and motion rejection.

    Args:
        window_s: Analysis window (longer = steadier, slower to react).
        min_seconds: Minimum data before reporting a value.
        min_confidence: Below this, the estimate is reported as unreliable.
        max_motion: Reject frames whose landmark speed exceeds this
            (inter-ocular units per second).
        with_hrv: Also compute HRV and respiration (a little more work).
    """

    def __init__(self, window_s: float = 10.0, min_seconds: float = 5.0,
                 min_confidence: float = 0.25, max_motion: float = 0.6,
                 with_hrv: bool = True, min_brightness: float = 0.20,
                 max_brightness: float = 0.92, max_imbalance: float = 0.25,
                 max_clipped: float = 0.10, max_illum_change: float = 0.06) -> None:
        self.window_s = window_s
        self.min_seconds = min_seconds
        self.min_confidence = min_confidence
        self.max_motion = max_motion
        self.with_hrv = with_hrv
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_imbalance = max_imbalance
        self.max_clipped = max_clipped
        self.max_illum_change = max_illum_change
        self._t: deque[float] = deque()
        self._rgb: deque[np.ndarray] = deque()
        self._prev_pts: np.ndarray | None = None
        self._prev_t: float | None = None
        self._prev_lum: float | None = None
        self.bpm = 0.0
        self.confidence = 0.0
        self.rejected = 0
        self.rejected_motion = 0
        self.rejected_lighting = 0

    def _lighting_bad(self, frame_bgr, landmarks) -> tuple[bool, list[str]]:
        """Reject frames whose illumination cannot carry a usable pulse."""
        q = lighting_quality(frame_bgr, landmarks)
        reasons = []
        if q["pixels"] < 100:
            # Note: the skin mask already drops pixels with luminance <=40, so a
            # very dark frame empties the mask and surfaces here rather than as
            # "too dark". Either way the frame is rejected.
            reasons.append("no skin")
        else:
            if q["brightness"] < self.min_brightness:
                reasons.append("too dark")
            elif q["brightness"] > self.max_brightness:
                reasons.append("overexposed")
            if q["imbalance"] > self.max_imbalance:
                reasons.append("uneven lighting")
            if q["clipped"] > self.max_clipped:
                reasons.append("clipped pixels")
            # A sudden illumination change corrupts the trace even when the
            # absolute level is fine (someone turns a lamp on, a cloud passes).
            if self._prev_lum is not None:
                delta = abs(q["brightness"] - self._prev_lum)
                if delta > self.max_illum_change:
                    reasons.append("illumination change")
            self._prev_lum = q["brightness"]
        return bool(reasons), reasons

    def _motion(self, landmarks, now_s: float) -> float:
        pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=float)
        speed = 0.0
        if self._prev_pts is not None and self._prev_t is not None:
            dt = now_s - self._prev_t
            if dt > 1e-6:
                # normalize by inter-ocular distance so it's scale-free
                iod = np.linalg.norm(pts[33] - pts[263]) + 1e-9
                speed = float(np.linalg.norm(pts - self._prev_pts, axis=1).mean() / iod / dt)
        self._prev_pts, self._prev_t = pts, now_s
        return speed

    def update(self, frame_bgr, landmarks, now_s: float) -> dict:
        moving = self._motion(landmarks, now_s) > self.max_motion
        bad_light, light_reasons = self._lighting_bad(frame_bgr, landmarks)
        if moving:
            self.rejected += 1
            self.rejected_motion += 1
        elif bad_light:
            self.rejected += 1
            self.rejected_lighting += 1
        else:
            rgb = skin_rgb(frame_bgr, landmarks)
            if rgb is not None:
                self._t.append(now_s)
                self._rgb.append(rgb)

        cutoff = now_s - self.window_s
        while self._t and self._t[0] < cutoff:
            self._t.popleft()
            self._rgb.popleft()

        span = (self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0
        ready = span >= self.min_seconds and len(self._t) >= 32
        out = {"bpm": 0.0, "confidence": 0.0, "ready": bool(ready),
               "reliable": False, "motion_rejected": bool(moving),
               "lighting_rejected": bool(bad_light and not moving),
               "lighting_reasons": light_reasons,
               "samples": len(self._t)}

        if not ready:
            out["bpm"] = round(self.bpm, 1) if self.bpm else 0.0
            out["confidence"] = round(self.confidence, 2)
            return out

        fps = (len(self._t) - 1) / span if span > 0 else 0.0
        sig = pos_signal(np.stack(self._rgb))
        bpm, conf = estimate_bpm(sig, fps)
        if conf >= self.min_confidence:
            self.bpm = bpm if self.bpm == 0 else 0.8 * self.bpm + 0.2 * bpm
        self.confidence = conf

        out["bpm"] = round(self.bpm, 1) if self.bpm else 0.0
        out["confidence"] = round(conf, 2)
        out["reliable"] = bool(conf >= self.min_confidence)

        if self.with_hrv and out["reliable"]:
            ibis = inter_beat_intervals(sig, fps, self.bpm)
            out.update(hrv_metrics(ibis))
            rpm, rconf = estimate_respiration(sig, fps)
            out["respiration_rpm"] = round(rpm, 1)
            out["respiration_confidence"] = round(rconf, 2)
            out["respiration_reliable"] = bool(rpm > 0 and rconf >= 0.2)
        return out
