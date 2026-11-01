# Engineering decisions

## 2026-10-28 — Keep the current result inconclusive

**Context:** Supplied code includes hand-weighted DSP scores and browser-side learning, but the inspected project tree has no labeled dataset/checkpoint. The reports state benchmark runs were not completed.

**Decision:** Implement descriptive DSP, report no authenticity probability, and return `INCONCLUSIVE` until a model has a verified task, artifact provenance, evaluation, calibration, and threshold.

**Alternatives considered:** Reuse SignalGuard's score; reuse the HTML heuristic/IndexedDB model; call the paid explanation backend. Rejected because these do not satisfy the required validated, no-cost classifier path.

## 2026-10-28 — Keep a dependency-free WAV fallback

**Context:** The development host could not resolve PyPI and did not have FastAPI/SciPy/SoundFile installed.

**Decision:** Retain the requested FastAPI full path and provide an offline standard-library PCM WAV path so the current vertical slice remains locally runnable without a package download. The fallback has smaller limits and is labeled as such in status.

**Trade-off:** Two implementations increase maintenance cost. Both share the report shape and tests, but full-mode parity requires a clean installation and integration run.

## 2026-10-28 — Store metadata, never source recordings

**Context:** History is useful for review, but raw voice data is sensitive.

**Decision:** Keep only basename, digest, size, original metadata, and derived report. Deletion removes the record; no raw file retention feature is present.
