# Architecture and implemented vertical slice

## Runtime paths

The full local mode uses FastAPI, SQLAlchemy/SQLite, SoundFile/lib­sndfile, NumPy, SciPy, scikit-learn, joblib, and PyAV/FFmpeg. `app/main.py` exposes the API, imports/trains labeled features, applies quality gates, and serves the browser interface; `app/audio.py` validates, decodes, downmixes, and rationally resamples; `app/dsp.py` computes time/spectral/cepstral/prosodic/phase measurements; `app/learning.py` compares and trains five local classifier families on grouped holdouts; `app/database.py` creates schema version 2 for analysis and labeled feature records.

The dependency-free fallback is `app/local_server.py` plus `app/stdlib_dsp.py`. It uses CPython's HTTP server, `wave`, `sqlite3`, a bounded multipart reader, a finite windowed-sinc resampler, and an iterative radix-2 FFT. It supports PCM WAV only and cannot load or train the full-mode model. Both modes bind to loopback and store no raw audio. Full mode can emit an experimental uncalibrated score; no calibrated probability is available.

The browser source in `app/static/` uses same-origin calls for local analysis and training. It renders waveform and PSD arrays as SVG, requests a browser-safe PCM WAV preview, and explains measurements with a deterministic on-device text helper. The model output is mapped to a 1–11 synthetic-evidence index and compared with a user-selected cutoff. External neural checkpoints are not integrated.

## Data flow

1. Receive one multipart field named `file`; reject requests beyond the configured body and file limits.
2. Check extension, container, channel count, sample rate, frame count/duration, and finite decoded samples. Decode only in memory.
3. Record original sample rate/channel count/container subtype and a downmix RMS ratio. Average channels to mono; flag severe cancellation as a quality note.
4. Resample to 16 kHz. The full path uses `scipy.signal.resample_poly`; fallback uses a normalized windowed-sinc low-pass kernel.
5. Compute time-domain summary on the canonical waveform. Remove the sample mean from a separate analysis array before windowed spectral calculations. Preserve original amplitude in the waveform/quality fields.
6. Measure the audio and, when a compatible local checkpoint exists and blocking quality gates pass, compute its uncalibrated synthetic-class score and map it to 1–11. Otherwise return INCONCLUSIVE with an explanation. Lesser quality notices remain advisory.
7. Save sanitized filename, input SHA-256, byte count, metadata, timestamp, decision, and report JSON in SQLite. Never write source recording bytes.

## Persistent schema

Schema version 2 contains `analyses` and `training_examples`. Analysis rows store timestamps, sanitized filename, file hash, byte count, duration, original sample rate/channel count, decision, and complete report JSON. Training rows store the explicit label, recording group, dataset name, digest, and derived DSP feature vector. The checkpoint is saved separately under `data/models/`. Schema version is recorded in SQLite `PRAGMA user_version`.

## Failure behavior

Invalid/empty/unsupported/corrupt/oversized/overlong audio returns a stable request error and is not persisted. If DSP computation fails, the API returns a generic analysis error without a verdict. A missing classifier is a normal state represented in `/api/status` and each report. Database errors do not produce an analysis success response.
