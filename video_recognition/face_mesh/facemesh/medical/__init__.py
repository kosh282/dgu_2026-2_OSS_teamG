"""Medical / clinical-style facial metrics.

Analyses built on the same 478 landmarks:

- :mod:`quality`  — gate that suppresses metrics when capture conditions are bad.
- :mod:`symmetry` — 3D mirrored-pair asymmetry, per-person baselines, peak-on-movement.
- :mod:`movement` — expressivity amplitude/jitter and 4-6 Hz tremor, scored per task.
- :mod:`rppg`     — contactless heart rate, HRV and respiration, with skin masking
  and motion rejection.

.. warning::
   **Screening and educational use only. Not a diagnostic device and not a
   medical device.** These metrics are not clinically validated. Landmark
   detectors are trained predominantly on healthy faces and are known to lose
   accuracy on faces affected by neuromuscular disease, which biases exactly the
   populations these metrics concern. Always consult a qualified clinician.
"""

from .blink import Blink, BlinkAnalytics, EyeBlinkDetector
from .embeddings import EmbeddingBackend, GeometricBackend, describe_backends, get_backend
from .geometry import normalize, normalize_3d
from .longitudinal import SessionSummary, compare, compare_many, summarize_session, trend
from .identity import AppearanceTracker, FaceGallery, face_embedding, similarity
from .movement import MovementAnalyzer, tremor_power
from .pupil import GazeDynamics, detect_oscillation, iris_diameter
from .pdf import PdfBackendUnavailable, available_backends, write_pdf
from .plr import PLRSession, PLRTrace, analyze_plr
from .protocol import DEFAULT_PROTOCOL, Task, TaskProtocol, TaskResult
from .quality import QualityGate
from .stats import (
    Interval,
    autocorrelation_time,
    block_bootstrap_ci,
    bootstrap_ci,
    effective_n,
    suggest_block_size,
    summarize,
)
from .rppg import (
    HeartRateEstimator,
    estimate_bpm,
    estimate_respiration,
    hrv_metrics,
    inter_beat_intervals,
    pos_signal,
    skin_mask,
)
from .symmetry import (
    REGIONS,
    DynamicSymmetry,
    SymmetryBaseline,
    symmetry_metrics,
)

DISCLAIMER = (
    "Screening/educational use only - not a diagnostic or medical device."
)

__all__ = [
    "normalize",
    "normalize_3d",
    "QualityGate",
    "Interval",
    "PLRSession",
    "PLRTrace",
    "analyze_plr",
    "write_pdf",
    "available_backends",
    "PdfBackendUnavailable",
    "SessionSummary",
    "summarize_session",
    "compare",
    "compare_many",
    "trend",
    "bootstrap_ci",
    "block_bootstrap_ci",
    "effective_n",
    "autocorrelation_time",
    "suggest_block_size",
    "summarize",
    "BlinkAnalytics",
    "AppearanceTracker",
    "get_backend",
    "describe_backends",
    "EmbeddingBackend",
    "GeometricBackend",
    "FaceGallery",
    "face_embedding",
    "similarity",
    "GazeDynamics",
    "detect_oscillation",
    "iris_diameter",
    "EyeBlinkDetector",
    "Blink",
    "TaskProtocol",
    "Task",
    "TaskResult",
    "DEFAULT_PROTOCOL",
    "symmetry_metrics",
    "SymmetryBaseline",
    "DynamicSymmetry",
    "REGIONS",
    "MovementAnalyzer",
    "tremor_power",
    "HeartRateEstimator",
    "skin_mask",
    "pos_signal",
    "estimate_bpm",
    "inter_beat_intervals",
    "hrv_metrics",
    "estimate_respiration",
    "DISCLAIMER",
]
