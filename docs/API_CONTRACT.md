# API contract

The full API is served by FastAPI in `app.main`; the fallback uses the same route names where practical and advertises its smaller WAV-only scope at `/api/status`.

## Endpoints

| Method | Path | Behavior |
|---|---|---|
| GET | `/api/health` | Liveness and version. |
| GET | `/api/status` | Local mode, classifier availability, formats, limits, retention, and verdict policy. |
| POST | `/api/preview` | Decodes an accepted upload in memory and returns a temporary browser-safe mono PCM WAV for playback; does not create history. |
| GET | `/model-datasets` | In-app HTML catalog of local models, external research weights, and dataset sources/terms. |
| GET | `/model-datasets.md` | Plain-text project notes for the research catalog. |
| POST | `/api/analyze` | Multipart field `file`; bounded decode, DSP measurements, optional local classifier inference, quality-gated decision, and SQLite insert. |
| GET | `/api/training/status` | Full mode only; labeled counts by class/group and saved model evaluation metrics. |
| POST | `/api/training/samples` | Full mode only; multipart fields `file`, `label`, `group_id`, and `dataset_name`; stores derived DSP features and labels, not audio. |
| POST | `/api/model/compare` | Full mode only; compares Logistic Regression, RBF SVM, Random Forest, Gradient Boosting, and small MLP on the same grouped holdout. JSON fields: `algorithm` and `feature_set` (`basic` or `advanced`). |
| POST | `/api/model/train` | Full mode only; grouped 25% holdout and local save of selected model family and feature set. JSON fields: `algorithm` and `feature_set`. |
| POST | `/api/model/threshold` | Full mode only; updates the local model decision cutoff from 0.20 to 0.80. JSON field: `threshold`. Does not change model weights or calibrate scores. |
| DELETE | `/api/model` | Removes only the active model checkpoint; training labels remain. |
| POST | `/api/history/{analysis_id}/confirm` | Full mode only; explicit human/synthetic confirmation adds saved DSP features to training; `uncertain` excludes it. |
| GET | `/api/history?limit=20&offset=0` | Newest metadata summaries; full report is available by ID. `limit` is 1-100. |
| GET | `/api/history/{analysis_id}` | Persisted report. |
| DELETE | `/api/history/{analysis_id}` | Deletes the report row; raw audio was never stored. |
| GET | `/api/export/history.json` | JSON array of full saved records. |
| GET | `/api/export/history.csv` | Flat CSV of history metadata, model score, quality score, and key DSP values. |
| GET | `/api/export/model.json` | Model metadata, estimator parameters and available coefficients/importances, feature mapping, threshold, and evaluation metrics; no audio. Inspection JSON is not a loadable checkpoint. |
| GET | `/` | Browser UI. |

Full-mode analysis limits are 50 MB, 120 seconds, 4-192 kHz, and one or two channels. It supports WAV, FLAC, OGG/Opus, AIFF, MP3, M4A/MP4, AAC, WMA, and WebM using libsndfile and PyAV/FFmpeg. Training accepts files up to 100 MB and 600 seconds each. The fallback accepts PCM WAV, 4-192 kHz, one or two channels, 8 MB, and 30 seconds, but has no training or classifier support. The decoder validates content; the extension alone is not trusted.

Successful analysis returns an `analysis_id`, ISO timestamp, sanitized basename, SHA-256, byte count, duration, original rate/channels, decision, and report. The report includes deterministic DSP values, a 0-100 technical-quality indicator (not confidence), and, when inference is allowed, an experimental 1–11 synthetic-evidence index mapped from the model's uncalibrated output. Speech-quality blockers and strongly tonal DTMF-like inputs withhold inference; lesser warnings remain visible but advisory. Without a model, results are descriptive and INCONCLUSIVE. No calibrated probability is produced.

Errors use `{"error":{"code":"...","message":"..."}}`. Invalid audio is 400, request-size overflow is 413, unsupported handler operation is 404, database/model-independent DSP failures do not return a success report, and a busy local analysis/preview slot is 429. Validation responses do not contain a traceback.

The server binds to `127.0.0.1`; no CORS allowlist is enabled. Upload handlers reject a mismatched browser Origin when one is provided. The API is local-only and does not implement accounts or remote authentication.
