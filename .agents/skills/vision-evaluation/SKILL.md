---
name: vision-evaluation
description: Evaluate MediaPipe vision pipeline behavior, observation quality, runtime cost, and capture/model boundaries without overstating synthetic results.
---

# Vision evaluation

Use when modifying `video_recognition/`, assessing face observation behavior or frame scheduling, or measuring vision performance.

## Procedure

1. Read root `AGENTS.md`, `video_recognition/README.md`, and relevant `engagement_model/README.md` sections. Identify whether the change affects capture, face detection, postprocessing, calibration, feature windows, or publishing.
2. Run the hardware-free vision/model regression suites from the repository root:

   ```powershell
   video_recognition/.venv/Scripts/python.exe -m unittest video_recognition.test_window_enumeration video_recognition.test_layout_regions video_recognition.test_full_frame_schedule -v
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s experimental_vision -v
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s engagement_model/tests -v
   ```

3. Separate synthetic-frame/mock regression from real performance and accuracy evaluation. A real evaluation records device, OS, camera/window size, frame and face counts, duration, model/config versions, dropped frames, latency distribution, and unobservable intervals. Do not add raw face imagery or persistent participant identifiers to reports.
4. Only run webcam/window experiments when a device and consented sample are actually available. Label unrun cases explicitly.

## Pass criteria

Selected deterministic tests pass; missing or low-quality observations remain unavailable/partial rather than fabricated as zero; performance claims include measured conditions and are not inferred from mocks.

## Report

Separate code regression results, synthetic/mock conditions, real device measurements, observation limitations, and unverified accuracy claims.
