"""Guided task protocol — prompt a movement sequence and score each one.

Facial palsy and hypomimia often hide at rest and reveal themselves on
*movement*. Clinical grading systems therefore ask the patient to perform a
fixed sequence of expressions and score each. This module reproduces that
structure: a scripted sequence of prompts, a countdown per step, and per-task
peak symmetry / expressivity collected while the prompt is active.

It does not reproduce any specific validated scale (House-Brackmann, Sunnybrook
etc.) and produces no clinical grade — only per-task measurements.

SCREENING / EDUCATIONAL USE ONLY — not a diagnostic device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .symmetry import DynamicSymmetry


@dataclass(frozen=True)
class Task:
    """One prompted movement."""

    name: str
    prompt: str
    seconds: float = 4.0
    settle_s: float = 1.0   # ignore the first moment while the face moves into place


#: The default sequence: a rest reference followed by the four movements that
#: most reliably expose unilateral weakness (upper face, eye, mid, lower face).
DEFAULT_PROTOCOL: tuple[Task, ...] = (
    Task("rest", "Relax your face. Look at the camera.", 5.0, 1.0),
    Task("raise-brows", "Raise your eyebrows as high as you can.", 4.0, 1.0),
    Task("close-eyes", "Close your eyes gently, then tightly.", 4.0, 1.0),
    Task("smile", "Smile as widely as you can, showing teeth.", 4.0, 1.0),
    Task("puff-cheeks", "Puff out both cheeks and hold.", 4.0, 1.0),
)


@dataclass
class TaskResult:
    """Measurements collected during a single task."""

    name: str
    frames: int = 0
    skipped_frames: int = 0
    peak_asymmetry: float = 0.0
    peak_symmetry: float = 1.0
    worst_region: str = ""
    peak_expressivity: float = 0.0
    mean_quality: float = 0.0
    _asym: list = field(default_factory=list, repr=False)
    _expr: list = field(default_factory=list, repr=False)
    _qual: list = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "task": self.name,
            "frames": self.frames,
            "skipped_frames": self.skipped_frames,
            "peak_asymmetry": round(self.peak_asymmetry, 4),
            "peak_symmetry": round(self.peak_symmetry, 3),
            "worst_region": self.worst_region,
            "peak_expressivity": round(self.peak_expressivity, 3),
            "mean_quality": round(self.mean_quality, 2),
        }


class TaskProtocol:
    """Drive a sequence of prompts and collect per-task measurements.

    Call :meth:`update` once per frame. It returns the current prompt state so
    the UI can render it, and silently records measurements while a task is
    active (outside its settle period, and only for good-quality frames).
    """

    def __init__(self, tasks=DEFAULT_PROTOCOL) -> None:
        self.tasks = tuple(tasks)
        self.index = 0
        self.finished = False
        self._start_t: float | None = None
        self._dyn = DynamicSymmetry(window_s=1e9)  # never expire within a task
        self.results: dict[str, TaskResult] = {}

    # --- state ------------------------------------------------------------
    @property
    def current(self) -> Task | None:
        return None if self.finished else self.tasks[self.index]

    def elapsed(self, now_s: float) -> float:
        return 0.0 if self._start_t is None else now_s - self._start_t

    def progress(self, now_s: float) -> float:
        cur = self.current
        if cur is None:
            return 1.0
        return min(self.elapsed(now_s) / cur.seconds, 1.0)

    def _advance(self) -> None:
        self.index += 1
        self._start_t = None
        self._dyn = DynamicSymmetry(window_s=1e9)
        if self.index >= len(self.tasks):
            self.finished = True

    def skip(self) -> None:
        """Skip the current task (user pressed 'n')."""
        if not self.finished:
            self._advance()

    def reset(self) -> None:
        self.index = 0
        self.finished = False
        self._start_t = None
        self.results = {}
        self._dyn = DynamicSymmetry(window_s=1e9)

    # --- per-frame --------------------------------------------------------
    def update(self, now_s: float, symmetry: dict | None = None,
               movement: dict | None = None, quality: dict | None = None) -> dict:
        """Advance the protocol and record a frame's measurements.

        Args:
            symmetry: output of :func:`symmetry_metrics`, or None.
            movement: output of ``MovementAnalyzer.update``, or None.
            quality: output of ``QualityGate.check``, or None (treated as good).

        Returns a dict describing what the UI should display.
        """
        if self.finished:
            return {"finished": True, "task": None, "prompt": "Assessment complete",
                    "progress": 1.0, "recording": False, "index": len(self.tasks),
                    "total": len(self.tasks)}

        task = self.tasks[self.index]
        if self._start_t is None:
            self._start_t = now_s

        elapsed = self.elapsed(now_s)
        settling = elapsed < task.settle_s
        good = quality is None or quality.get("ok", True)
        recording = (not settling) and good

        res = self.results.setdefault(task.name, TaskResult(name=task.name))
        if not settling:
            if not good:
                res.skipped_frames += 1
            else:
                res.frames += 1
                if quality is not None:
                    res._qual.append(quality.get("score", 1.0))
                if symmetry is not None:
                    res._asym.append((symmetry["asymmetry"], symmetry["symmetry"],
                                      symmetry["worst_region"]))
                if movement is not None:
                    res._expr.append(movement["expressivity"])

        if elapsed >= task.seconds:
            self._finalize(res)
            self._advance()

        return {
            "finished": self.finished,
            "task": task.name,
            "prompt": task.prompt,
            "progress": min(elapsed / task.seconds, 1.0),
            "settling": settling,
            "recording": recording,
            "blocked": (not good) and (not settling),
            "reason": "" if good else ", ".join(quality.get("reasons", [])),
            "index": self.index,
            "total": len(self.tasks),
        }

    @staticmethod
    def _finalize(res: TaskResult) -> None:
        if res._asym:
            peak_i = int(np.argmax([a for a, _, _ in res._asym]))
            res.peak_asymmetry = res._asym[peak_i][0]
            res.peak_symmetry = res._asym[peak_i][1]
            res.worst_region = res._asym[peak_i][2]
        if res._expr:
            res.peak_expressivity = max(res._expr)
        if res._qual:
            res.mean_quality = float(np.mean(res._qual))

    # --- summary ----------------------------------------------------------
    def summary(self) -> dict:
        """Per-task results plus a session-level roll-up.

        Any task still in progress is finalized first, so quitting mid-protocol
        (or reading results early) still yields that task's measurements.
        """
        for res in self.results.values():
            self._finalize(res)
        tasks = {name: r.to_dict() for name, r in self.results.items()}
        usable = [r for r in self.results.values() if r.frames > 0]
        movement_tasks = [r for r in usable if r.name != "rest"]

        roll: dict = {"tasks_completed": len(usable), "tasks_total": len(self.tasks)}
        if movement_tasks:
            worst = max(movement_tasks, key=lambda r: r.peak_asymmetry)
            roll["worst_task"] = worst.name
            roll["worst_task_asymmetry"] = round(worst.peak_asymmetry, 4)
            roll["worst_task_region"] = worst.worst_region
            roll["mean_peak_expressivity"] = round(
                float(np.mean([r.peak_expressivity for r in movement_tasks])), 3)
        rest = self.results.get("rest")
        if rest is not None and rest.frames:
            roll["rest_asymmetry"] = round(rest.peak_asymmetry, 4)
            if movement_tasks:
                # Asymmetry that only appears on movement is the interesting signal.
                roll["movement_excess_asymmetry"] = round(
                    roll["worst_task_asymmetry"] - rest.peak_asymmetry, 4)
        return {"tasks": tasks, "summary": roll}
