# Verification notes — 28 October 2026

## Confirmed in this workspace

| Area | Result | Evidence |
|---|---|---|
| Python source syntax | Pass | `.venv` Python 3.14.7: `python -m compileall -q app` completed. |
| Unit/workflow checks | Pass | `python -m unittest discover -s tests -v`: all 9 checks passed. |
| Schema compatibility | Pass | Existing SQLite `user_version=2` opens; both full app and fallback accept schema v2. |
| Saved model | Pass | `data/models/dsp_logistic.joblib` loads with matching 11-feature schema. |
| Supplied corpus import | Completed | 40 new files imported from the user-provided `Dataset.csv`; 2 exact Real-file duplicates skipped; no import errors. Total: 33 human / 25 synthetic records. |
| Current labels/holdout | Observed, weak evidence | 42 train / 16 holdout files across 29 / 10 groups; accuracy, balanced accuracy, and AUC are 1.000; confusion matrix `[[10,0],[0,6]]`. Labels are unverified and the small holdout may contain corpus-specific shortcuts; do not treat this as generalization evidence. |
| DTMF input | Decoder checked | Several supplied WAV tone files decoded through the full decoder. The quality gate now withholds classifier inference for strongly tonal, non-speech clips. |

## Not established by these checks

- The small holdout is not an independent benchmark and does not establish real-world accuracy. No calibrated probability, production threshold, or verified ground truth exists.
- A fresh user browser session and independent speaker/generator evaluation have not been completed. Refresh the page after the service starts to see the current `33 human / 25 AI` status.
- AASIST, RawNet2, RawGAT-ST, WavLM, and external research datasets are not integrated or downloaded into this project.

The `INCONCLUSIVE` result on DTMF/non-speech is intentional: those signals do not represent human or synthetic voice, so a voice classifier result would be misleading.
