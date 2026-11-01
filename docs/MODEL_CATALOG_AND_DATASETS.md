# Audio detector models and datasets

This catalog distinguishes components that run in AegisVoice from external research links.

## Components running locally

| Component | Status | Purpose and limit |
|---|---|---|
| MFCC, LFCC, CQT-style cepstra, STFT phase and spectrogram, pitch/HNR, jitter/shimmer proxies, RMS-envelope modulation, spectral and time-domain features | Implemented | Deterministic measurements and visualizations. The spectrogram and envelope-modulation descriptors are analysis-only, not model inputs. CQT-style uses geometrically spaced STFT filters; prosodic values are estimates. |
| Logistic Regression, RBF SVM, Random Forest, Gradient Boosting, small MLP | Implemented | Compare grouped holdout results and train a selected local model on explicitly labeled examples. Output scores are uncalibrated. |
| Held-out EER and error rates | Implemented on retraining | Equal error rate is estimated from the grouped holdout score curve. False-accept and false-reject rates are also shown at the fixed 0.50 evaluation cutoff. These are evaluation metrics only; they do not calibrate scores or choose the active threshold. |
| Confirmed history labels and imported folders/CSV | Implemented | Only explicit human/synthetic labels are used. Predictions do not become training labels automatically. |
| Synthetic-evidence index and decision threshold | Implemented | Raw model output is mapped onto 1–11 for display. A user-selected 20–80% cutoff determines the label. Neither is a calibrated probability. |
| Local measurement explanation | Implemented | Deterministically explains signal measurements on-device. It does not decide the label. |
| SignalGuard archive supplied with the project | Reviewed as reference | It contains Python DSP/heuristic modules for pitch, LPC, bispectrum, modulation, breath, phase, and temporal analysis, plus score fusion and threshold calibration helpers. The supplied ZIP has no top-level license file and no verified pretrained anti-spoof checkpoint; its source was not copied into AegisVoice. No MATLAB `.m` sources were found. |

## Pretrained research model

| Model | In-app status | Research link and use |
|---|---|---|
| AASIST / AASIST-L | Linked, not integrated | [Official AASIST repository](https://github.com/clovaai/aasist) and [checkpoint files](https://github.com/clovaai/aasist/tree/main/models/weights). It requires its own runtime, checkpoint, preprocessing, and independent evaluation. |
| WavLM / Wav2Vec2 frozen representations + lightweight classifier | Research direction, not integrated | A 2024 Interspeech study reports improved cross-dataset results with frozen self-supervised speech representations and a small back end. These require a larger runtime and must be independently tested on the intended domain before use. [Paper](https://www.isca-archive.org/interspeech_2024/pascu24_interspeech.pdf). |

No AASIST checkpoint is included in this workspace, and the local trainer does not load AASIST weights. Its benchmark results cannot be treated as this app's results.

## Datasets for training or evaluation

| Dataset | Use | Terms and caution |
|---|---|---|
| Supplied Human/AI corpus | User-provided training examples | Labels are not independently verified. Keep a source/group column and avoid putting related speakers or source clips in both train and holdout groups. |
| [ASVspoof 2021](https://www.asvspoof.org/index2021.html) | Controlled spoofing and deepfake benchmark | Review access requirements, terms, and challenge protocols before downloading. |
| [ASVspoof 5](https://zenodo.org/records/14498691) | More recent evaluation for varied devices, speakers, modern attacks, and adversarial conditions | The official record asks users to read its README/license and recommends its evaluation plan; registration may be required for data access. Prefer as an untouched external evaluation set, not a random split mixed into local training. |
| [Fraunhofer In-the-Wild](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | External evaluation on real-world online recordings | Review the dataset terms and collection context before use. |
| [WaveFake v1.2 on Zenodo](https://zenodo.org/records/5642694) | Generated-speech comparison | Review the record's attribution/share-alike terms and conditions on its source corpora. |

The app does not download or train from these external datasets automatically. Download from the linked source, review its terms, then import Human/AI folders or a CSV manifest plus the matching recordings. Training keeps extracted features and metadata, not the source recordings.

## Evaluation guidance

Keep speakers, source recordings, and generators grouped so related clips cannot land on both sides of the holdout. The local comparison uses the same grouped 25% holdout for each model family and reports accuracy, balanced accuracy, ROC AUC, equal error rate, false-accept/reject rates at the evaluation cutoff, confusion matrix, class counts, and group counts. It is an exploratory screening result, not a robust benchmark. Use an untouched external corpus for final evaluation and set decision thresholds for the intended error tradeoff. The single grouped split remains too small for reliable calibration; a threshold changes the label policy, it does not improve the classifier or turn its output into a probability.
