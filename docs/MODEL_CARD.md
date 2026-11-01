# Model card — experimental DSP-feature baseline

**Status:** this workspace contains `data/models/dsp_logistic.joblib`, trained on 58 local labeled feature records (33 human / 25 synthetic). It is an experimental user-data checkpoint, not a bundled or independently validated detector.

| Field | Current design |
|---|---|
| Intended use | Experimental human-vs-synthetic speech triage on user-labeled audio. Not proof or sole basis for consequential decisions. |
| Architecture | Current saved artifact is scikit-learn `StandardScaler` + class-balanced `LogisticRegression`. The app now also offers RBF SVM, Random Forest, Gradient Boosting, and a small MLP for comparison or later training. |
| Inputs | The current artifact uses 11 basic DSP values from a centered 10-second crop at 16 kHz mono. The app also supports a 38-value extended DSP/cepstral/prosodic/phase feature schema. |
| Labels | Explicit `human` / `synthetic`; folder labels and CSV are user-supplied and not independently verified. |
| Split | Deterministic 25% grouped holdout, seed 42; matched recording groups are kept together. |
| Training minimum | 16 recordings, both classes, at least 4 groups per class. |
| Current grouped holdout | 16 files / 10 held-out groups: 100% accuracy, balanced accuracy, and ROC AUC; confusion matrix `[[10, 0], [0, 6]]` (true rows human/synthetic, predicted columns human/synthetic). This small, user-labeled holdout is exploratory and may reflect corpus-specific shortcuts. |
| Additional evaluation | New training runs also report equal error rate (EER) and false-accept/false-reject rates at the fixed 0.50 evaluation point. The existing saved checkpoint predates these fields; retrain to calculate them. EER is not used to calibrate or set the saved threshold. |
| Score | Synthetic-class model output mapped to a 1–11 evidence index, uncalibrated. It is not a real-world probability. |
| Checkpoint | `data/models/dsp_logistic.joblib`, generated locally after training. |
| Explanation / hardware | Plain-English measurements are generated locally; local classifiers train on CPU. AASIST is not integrated. |
| Known limitations | Very small, imbalanced data; uncertain labels and provenance; generator/speaker/domain shift; and possible recording artifacts. Blocking quality checks withhold inference; lesser warnings remain advisory. |

The supplied Real/Fake corpus has 21 files per class and matching filename stems. The import added 19 Real and 21 Fake files; two Real files already existed and were deduplicated. Inspected Real files were stereo while Fake files were mono. Channel count is excluded from model features, and matching stems are kept together in the grouped split, but other dataset artifacts may still inflate holdout metrics. Independent external validation is required before making accuracy claims. The selected cutoff changes decisions but does not calibrate scores.
