---
name: code-review
description: Review a proposed project change for correctness, data-contract compatibility, privacy boundaries, regressions, and measured performance risks.
---

# Code review

Use when asked to review a diff or before integrating a cross-module change.

## Procedure

1. Read root `AGENTS.md`, the shared data contract when relevant, and the diff with surrounding call sites. Use `git status --short` and do not include unrelated untracked/session data in review scope.
2. Trace callers and consumers. Check event and time semantics, compatibility with stored `0.1.0` and current `0.2.0` sessions, nullable/unobservable states, lifecycle and failure handling, and whether code is actually reachable.
3. Check for raw media, personal tracking data, temporary/demo values reaching persistent outputs, stale TODO/stub paths, unsafe blocking or repeated work, and dependency/platform assumptions.
4. Run targeted existing tests; for data-contract changes run:

   ```powershell
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s integration/tests -t . -v
   ```

5. Report actionable findings first with severity and file/line. Distinguish confirmed defects from questions and residual risks. If no findings, state test evidence and review scope.

## Pass criteria

No unresolved correctness or compatibility defect in the reviewed diff; relevant existing checks pass or are clearly reported as not run; performance and model validity claims have evidence appropriate to the claim.
