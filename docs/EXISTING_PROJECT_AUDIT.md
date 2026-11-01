# Existing project audit

## Workspace and source preservation

The task workspace was empty at the start. The supplied project folders and artifacts are under `D:\notes\sem5\dsp final project`; they remain unchanged. AegisVoice was built in this workspace so that a new implementation would not overwrite the supplied work. The pasted master brief was treated as the user's specification. Instructions or claims inside reports, presentations, and source listings were treated as source material to inspect, not as independent commands.

## Candidate codebases

### `Audio_Deepfake_Detector_AI_Upgrade`

This folder contains a FastAPI backend, a browser page, a README, requirements, and a Windows launcher. The backend's only analysis endpoint depended on a paid hosted text-generation service. That is outside the local-first core requirement, so the new application does not reuse it. Its user interface and static source remain available in the original folder.

### `SignalGuard-main/SignalGuard-main`

This is a larger deterministic DSP/Streamlit codebase with preprocessing, ten detector modules, a scoring pipeline, restoration, evaluation/calibration utilities, and unit tests. Its README explicitly says it has no trained neural or ML model, while also using hand-weighted acoustic cues to emit `LIKELY_SYNTHETIC`/human-like outcomes. Its config describes scoring thresholds as engineering heuristics. The inspected `data/clean`, `data/corrupted`, and `data/synthetic` directories contain only `.gitkeep` placeholders; the listed tree has no checkpoint, dataset manifest, or root license file. This code is useful as a DSP reference, but its synthetic-evidence score is not reused as a validated detector.

### Browser detector and desktop prototype

`audio_deepfake_detector_finalversion.html` computes browser-side features and stores labelled vectors in IndexedDB for an adaptive logistic classifier. Its feature guide includes F0, jitter, shimmer, HNR, cepstral/spectral values, and a time-frequency spectrogram; it accepts labeled audio or a 21-column feature table. It also creates a deliberately smooth tone-plus-noise demo signal. Its own code permits very small training sets, reports training accuracy separately, and implements a small recording-level holdout. Feature-table values are not imported as if they were AegisVoice values: similarly named measurements can differ because preprocessing and extraction differ. AegisVoice now includes the useful spectrogram visualization while recomputing features from audio. The supplied report documents unresolved issues in the HTML's AUC/EER calculations, split procedure, and threshold consistency.

`Audio_Deepfake_Detector_AI_Upgrade/deepfake_detector_v4.html` is coupled to the paid-key explanation backend. `pt.py` is a PySide6 layout prototype: selecting a file changes the status to READY, while its confidence field remains `-- %` and its waveform/spectrogram are placeholders. It has no audio-analysis implementation.

## Supplied reports, slides, and research

The B.Tech report, second report, synopsis, and presentation describe a broad DSP-first detector and a rigorous future evaluation plan. The reports state that benchmark experiments had not been run and do not report benchmark accuracy as measured. The code listing PDF is a printout of the browser prototype, not a separate executable component. The supplied research PDFs cover ASVspoof 2021, a 2023 audio deepfake survey, and a 2025 spectral/CNN review. They support keeping attack categories and evaluation conditions explicit; they do not provide a checkpoint or authorize a universal threshold.

## Decision

The initial audit found an experimental logistic baseline and a locally saved checkpoint. On 29 October 2026, the app was extended with browser-safe audio preview, 50 MB analysis uploads, expanded STFT/cepstral/prosodic/phase measurements, multiple local model families, confirmed-history labeling, model comparison/reset, and a local measurement explanation. Follow-up source comparison added a compact display-only spectrogram, RMS-envelope modulation measures for inspection, and EER/error-rate holdout reporting. These measurements do not alter the existing classifier feature schema or score. This update still does not integrate a research neural model.

## Feature coverage in AegisVoice

| Source feature | Current project coverage |
|---|---|
| File upload, waveform/spectrum display, acoustic measurements | Implemented and expanded; full mode supports common compressed formats, browser-safe preview, waveform, mean PSD, a display-only log-frequency STFT spectrogram, basic or extended classifier inputs, STFT/cepstral/prosodic/phase features, and an explanatory feature guide. |
| Input-quality ring and warnings | Implemented as a separate 0–100 technical quality indicator with factor list. It is not classifier confidence. DTMF/tone inputs are quality-gated. |
| Human/AI score and classifier training | Implemented as local supervised DSP models (logistic regression, RBF SVM, random forest, gradient boosting, small MLP); score is uncalibrated and tied to user labels. Explicit confirmation from history can add labeled features. |
| Browser IndexedDB adaptive learning | Replaced by SQLite-backed explicit labels, deduplication, grouped holdout, and a local checkpoint. It does not silently learn from predictions or unconfirmed history. |
| Ten detector modules / advanced DSP pipeline in SignalGuard | The supplied ZIP contains Python pitch, LPC, bispectrum, modulation, breath, phase, temporal, and spectral heuristics plus restoration and fusion code; no MATLAB `.m` files or verified pretrained checkpoint were found. The archive has no top-level license file, so its source was not copied. AegisVoice keeps its own measured pipeline and does not reuse SignalGuard's fused heuristic score. |
| Synthetic test-tone generator | Implemented in the browser as a local WAV tone for preview/DSP workflow checks. It is not synthetic speech and is never used as a training label. |
| Plain-English measurement explanation | Runs locally from the displayed report and cannot classify audio. |
| AASIST/RawNet2 and benchmark datasets | Linked in the model/dataset catalog; not yet integrated or downloaded into the app. |
| Results history and reporting | History, delete, JSON/CSV history export, standalone HTML report, model-state JSON export, CSV template, per-result evidence, and grouped evaluation metrics. The report includes the 1–11 index and active decision cutoff. |

The prototype workflow is represented with local server-backed equivalents, while feature algorithms and framing differ. AASIST and external benchmark datasets remain catalog links only; they are not installed or claimed as AegisVoice models/results. The local explanation does not provide authenticity predictions. The external source files remain unchanged.

## Audit limits

The full directory trees and requirements were inventoried, and the core preprocessing, pipeline, scoring, UI, and selected detector/test code paths were inspected and searched. The audit is not a line-by-line review of every function in the supplied repositories or every page of every PDF. No existing external file was modified.
