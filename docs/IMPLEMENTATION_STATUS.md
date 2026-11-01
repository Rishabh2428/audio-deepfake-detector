# AegisVoice implementation status — 29 October 2026

## Available in full local mode

- Analysis and browser-safe preview for WAV, FLAC, OGG/Opus, AIFF, MP3, M4A/MP4, AAC, WMA, and WebM. Analysis uploads allow 50 MB and 120 seconds by default.
- Local waveform and power-spectrum views; time/frequency measures; STFT phase; MFCC, LFCC, CQT-style cepstra; autocorrelation F0/HNR and clearly labeled jitter/shimmer proxies; quality notices and a plain-English feature guide.
- Local labeled-data import from folders, separately chosen human/AI files, or CSV manifests. Audio is processed in memory and discarded. Folders/history can contribute only explicit labels.
- History controls to confirm Human/AI or mark a run unverified, with an advanced-feature readiness count for new measurements.
- Compare Logistic Regression, RBF SVM, Random Forest, Gradient Boosting, and small MLP on the same grouped holdout. Train one selected family using basic or extended features. Reset deletes the active checkpoint while retaining labels.
- Local plain-English explanation of derived DSP measurements; it cannot set or change the detector decision.
- Model output appears as a 1–11 synthetic-evidence index, with an adjustable 20–80% decision cutoff. Lesser input-quality warnings remain advisory; silence, strong channel cancellation, short clips, and tonal non-speech block a class result.
- External links and a catalog for AASIST/AASIST-L, ASVspoof 2021, In-the-Wild, and WaveFake.
- Offline standard-library fallback for PCM WAV inspection. It does not load/train a classifier or support compressed codecs.

## Current local artifacts

The `data/aegisvoice.sqlite3` database has 58 labeled examples (33 human / 25 synthetic) and the active saved model at `data/models/dsp_logistic.joblib` remains the prior basic-feature logistic model. Its recorded grouped holdout had 16 files / 10 groups and 100% accuracy, balanced accuracy, and AUC (confusion matrix `[[10,0],[0,6]]`). This is from unverified user-provided labels and a small corpus; it is not a measure of real-world accuracy or a calibrated probability. The 42 source-corpus clips have now been re-extracted locally with extended features: 21 human and 21 synthetic. They are ready for advanced-feature model comparison/training, while the active model was left unchanged.

A same-split comparison was run on those 42 clips and the 38-value extended features: Logistic Regression, RBF SVM, Random Forest, and small MLP each scored 90% accuracy; Gradient Boosting scored 100%; all showed AUC 1.000 on the same 10-file / 5-group holdout. This is a very small and unverified user corpus. No model was auto-selected from those results, and they should not be described as expected real-world accuracy.

The SignalGuard ZIP was reviewed as a reference. It contains Python heuristic DSP and optional text-explainer code, but no MATLAB `.m` files and no verified pretrained anti-spoof checkpoint. AegisVoice computes its own feature pipeline. The AASIST checkpoint is linked for research but has not been installed or integrated.

## Boundaries

- No pretrained deepfake detector is bundled. The implemented classifier learns from explicit local labels.
- Current user-provided labels and grouped holdout results are not independently verified; the 1–11 evidence index remains uncalibrated and experimental.
- The explanation is deterministic and runs in the browser from the displayed report; no hosted language service is used.
- XGBoost is not included as a model option because its package is not part of this project. Gradient Boosting is available locally.
- A proper independent dataset evaluation and probability calibration have not been performed.
