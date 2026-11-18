---
name: integration-check
description: Check the project's vision, model, STT, integration, frontend, and assistant connections after cross-module changes or when asked to verify the pipeline.
---

# Integration check

Use when a change crosses module boundaries, modifies event delivery or session persistence, or the user requests an end-to-end integration check.

## Procedure

1. Read the root `AGENTS.md`, affected module guides, contract README, and current `git status`.
2. Trace the changed path from input to consumer: vision `FeatureWindow` → engagement runtime/publisher → integration event; audio chunk → STT final event → integration; event snapshot → frontend/assistant; saved JSON → reload. Follow only affected branches and identify hardware/API assumptions.
3. Run deterministic suites from the repository root with the project's Python environment:

   ```powershell
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s integration/tests -t . -v
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s engagement_model/tests -v
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s frontend/tests -v
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s sherpa_stt/tests -t . -v
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s ai_agent/tests -t . -v
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s faster_whisper_stt -p "test_*.py" -v
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s experimental_vision -v
   video_recognition/.venv/Scripts/python.exe -m unittest video_recognition.test_window_enumeration video_recognition.test_layout_regions -v
   ```

   Run relevant suites when dependencies are unavailable; report omissions. GitHub Actions runs the dependency-free contract suite plus model, frontend, and hardware-free vision suites on Ubuntu using the declared integrated requirements.
4. Treat webcam, microphone, live window capture, downloaded model, and external API checks as separate environment checks. Never infer their success from mocks or synthetic fixtures.

## Pass criteria

All selected deterministic tests pass; event delivery, shared timestamps, and save/reload remain consistent; no unrequested contract changes or unreported environment gaps.

## Report

List changed path, executed commands and pass/fail counts, verified pipeline edges, hardware/API checks not run, and any known gaps.
