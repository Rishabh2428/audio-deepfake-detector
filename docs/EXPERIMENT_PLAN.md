# Experiment plan

No detector experiment has been run in this application. The plan below is prospective; no result is implied.

## Priority 1: reproduce the software path

Use generated sine, silence, impulse, and noise fixtures to verify numeric operations and edge cases only. Record expected values, sample rate, frame configuration, and tolerance. These fixtures must never enter detector accuracy metrics.

## Priority 2: labeled DSP baseline

1. Select an openly accessible corpus only after reviewing its official license, terms, label definitions, and redistribution constraints. Do not bundle data by default.
2. Preserve official train/development/evaluation partitions and speaker/generator identities. Deduplicate audio using hashes and investigate speaker, session, codec, and source leakage.
3. Define task labels before feature extraction. Keep bona fide, TTS, VC, replay, editing, and channel effects distinct when the source protocol distinguishes them.
4. Establish a simple baseline (e.g. logistic regression) before LightGBM. Fit normalization and any feature selection only on training data. Tune hyperparameters and thresholds on development data.
5. Evaluate once on the untouched evaluation partition. Report confusion matrix, ROC-AUC, EER, precision, recall, F1, balanced accuracy, class counts, uncertainty intervals, and abstention fraction; include the metric definition and positive class.
6. Perform generator-disjoint and cross-dataset testing. Report codec, noise, channel, language, duration, and quality slices where the protocol permits.

## Priority 3: neural models and fusion

Only integrate AASIST or a WavLM/XLS-R/wav2vec anti-spoofing checkpoint after confirming architecture/version, score direction, class mapping, preprocessing, checkpoint hash, license, intended task, and CPU feasibility. Compare each model separately on the same partitions before any fusion. Fit calibration/fusion on development data only and retain an inconclusive region chosen for explicit cost trade-offs.

## Reporting discipline

Use versioned manifests and save the exact command, environment inventory, source hashes, split manifest, feature schema, random seeds, model hash, calibration, threshold policy, and report. Do not report a benchmark metric until its evaluation command has completed and its evidence artifact exists.
