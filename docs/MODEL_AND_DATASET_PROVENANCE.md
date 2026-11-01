# Model and dataset provenance

## Implemented model

- **Classifier:** The saved artifact is local scikit-learn `StandardScaler` + class-balanced `LogisticRegression`. The app supports comparison/training with Logistic Regression, RBF SVM, Random Forest, Gradient Boosting, or a small MLP.
- **Feature sets:** Basic 11-value DSP vector or 38-value extended DSP, MFCC/LFCC/CQT-style, prosodic, spectral, and phase vector. Prosodic values and CQT-style filters are approximate measurements and labeled as such.
- **Labels:** Only explicitly uploaded `human` and `synthetic` labels. Folders named Real/Human and Fake/AI are recognized; CSV columns `filename`/`path`, `label`/`class`, and optional `group_id` are supported.
- **Split:** Deterministic grouped 25% holdout (`GroupShuffleSplit`, seed 42); group IDs are kept together. At least 16 files and 4 distinct groups per class are required.
- **Features per recording:** Computed on a centered 10-second segment after canonical 16 kHz mono preprocessing. The database retains metadata, labels, and derived features, not original audio.
- **Artifact:** `data/models/dsp_logistic.joblib`.
- **Output:** The raw synthetic-class model output is mapped to a 1–11 index. A user-selected cutoff from 20% to 80% determines the label. Both are uncalibrated; neither is a real-world probability.

## Current local artifact (workspace snapshot)

`data/aegisvoice.sqlite3` contains 58 labeled feature records: 33 human and 25 synthetic. The current deterministic grouped holdout contains 16 clips / 10 groups and reports 1.000 accuracy, balanced accuracy, and ROC AUC, with confusion matrix `[[10, 0], [0, 6]]`. These small, user-labeled results are not evidence of broad accuracy. The supplied 42-clip corpus is user-provided and not independently verified.

## User-supplied corpus inspection
