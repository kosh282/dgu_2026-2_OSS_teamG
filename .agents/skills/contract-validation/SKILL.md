---
name: contract-validation
description: Validate shared session JSON, event shapes, timestamps, quality, provenance, and producer-consumer compatibility against the team's existing contract.
---

# Contract validation

Use when changing event producers/consumers, session persistence, the shared example, or when asked whether data conforms to the shared contract.

## Procedure

1. Read `공유용_데이터_계약/README.md`, `examples/session.json`, `integration/contracts.py`, and session load/save code.
2. Treat the documented contract and existing validator as authoritative. Do not alter the contract to make an implementation pass. JSON Schema is not currently the source of truth; avoid duplicating its rules unless the team agrees to maintain both.
3. Check required fields and types, UUIDs, event type payloads, UTC timestamps, monotonic nonnegative relative millisecond ranges, quality/provenance, session identity, duration bounds, and forbidden raw-media/personal tracking data.
4. Run the contract suite and relevant storage tests from the repository root:

   ```powershell
   video_recognition/.venv/Scripts/python.exe -m unittest discover -s integration/tests -t . -v
   ```

   Dependency-free fallback: `python -m unittest discover -s integration/tests -t . -v`.
5. Use only the checked-in synthetic example or generated fixtures. Do not inspect or modify personal session files as test data.

## Pass criteria

The shared example loads through the real integration service; malformed required fields, types, or times are rejected; valid unknown optional fields remain compatible; no contract document or version is silently changed.

## Report

State fixture and validator used, checks run and results, compatibility impact, and any contract ambiguity needing team agreement.
