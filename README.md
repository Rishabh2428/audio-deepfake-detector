# AegisVoice

A local-first audio inspection and experimental deepfake-triage app. Full mode supports common audio formats, browser-safe playback, DSP measurements, labeled-data import, several local scikit-learn classifiers, history, and grouped holdout metrics. Its plain-English measurement explanation runs locally. The synthetic-evidence index is uncalibrated and is not a real-world probability.

## Start the full app on this computer

The project already has a full Python environment at `.venv`. Open PowerShell and run:

```powershell
Set-Location -LiteralPath "C:\Users\Asus\OneDrive\Documents\Codex\2026-10-28new"
.\start_aegisvoice.bat
```

The launcher uses the full app and selects an available port from 8000 through 8010. Open the exact `http://127.0.0.1:<port>` address it prints. To stop it, run `.\stop_aegisvoice.bat` from the project folder. For a foreground server instead, use `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8001` and press **Ctrl+C** in that PowerShell window.

Use the full command above for the saved classifier, training page, and compressed formats. `python -m app.local_server` is the smaller PCM-WAV-only fallback and does not load the classifier or provide the training API. The Windows launcher is also available: double-click `start_aegisvoice.bat`; it selects full mode when the dependencies are present.

On this workspace, the database contains **33 human and 25 AI-labeled recordings**. The supplied 42-clip `deepfake_sample_audio` corpus contributed 19 new human and 21 synthetic examples; two exact human clips were already present and were skipped. The saved grouped-holdout snapshot has 16 test clips: 100% accuracy, balanced accuracy, and ROC AUC (confusion matrix `[[10, 0], [0, 6]]`). Treat this as exploratory only: the CSV labels were not independently verified, the test set is small, and corpus-specific shortcuts may explain the result. This score is not a calibrated probability or a real-world accuracy claim. Reimporting the corpus skips exact duplicates.

## Lightweight offline PCM-WAV mode (fallback)

This fallback needs only Python 3.12 or later and works without package installation or external-service setup. It supports PCM WAV inspection only. It does not load the saved classifier, provide training, or decode compressed audio. Use the full mode above for the project features.

1. Open PowerShell in the `2026-10-28new` project folder. For this computer, paste:

   ```powershell
   Set-Location -LiteralPath "C:\Users\Asus\OneDrive\Documents\Codex\2026-10-28new"
   ```

2. Check Python is available:

   ```powershell
   python --version
   ```

   Python 3.12 or newer is required.

3. Start the WAV-only fallback only when full dependencies are unavailable:

```powershell
python -m app.local_server --host 127.0.0.1 --port 8000 --data-dir data
```

4. Leave that PowerShell window open. It should say `AegisVoice offline WAV mode running at http://127.0.0.1:8000`.
5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. Upload an uncompressed PCM WAV file (8, 16, 24, or 32 bit), up to 8 MB and 30 seconds.
6. To stop the server, return to PowerShell and press **Ctrl+C**.

If PowerShell says `No module named app`, first run the `Set-Location` step above. If it says the address or port is already in use, stop the other local server or change `--port 8000` to `--port 8001` and open `http://127.0.0.1:8001` instead.

## Set up the full local API on a new computer

The full path supports WAV, FLAC, OGG/Opus, AIFF, MP3, M4A/MP4, AAC, WMA, and WebM using SoundFile plus PyAV/FFmpeg. It also enables the supervised DSP-feature training page. Python 3.12 or later is required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then run `start_aegisvoice.bat` and open `http://127.0.0.1:8000`. The script selects full mode when dependencies are installed; otherwise it starts the WAV fallback. `stop_aegisvoice.bat` stops only the server recorded for this project. The scripts do not install packages automatically.

For a one-step dependency setup, double-click `install_aegisvoice.bat`, wait for it to finish, then double-click `start_aegisvoice.bat`. This requires internet access to PyPI the first time. The local model training and compressed-format decoders are available in full mode.

Dependency installation needs access to the Python package index. Audio decoding, analysis, and local explanations run on this computer. Optional text explanations support Google AI Studio/Gemini, NVIDIA NIM/Nemotron, OpenRouter (defaults to its free-model router), and OpenAI. After provider selection, an API key and explicit checkbox, AegisVoice sends only allowlisted numeric DSP measurements; no audio, filename, history ID, classifier score, label, or verdict is sent. Keys are not saved by AegisVoice and are cleared when the request completes; provider free tiers, access, and rate limits vary. Model names remain editable. Separately, optional Resemble Detect sends the analyzed audio file only after a distinct consent checkbox. The app requests Resemble's zero-retention mode, but the provider processes the audio under its terms, and API access or usage may cost money.

For online explanations, select Google AI Studio, OpenAI, NVIDIA NIM, OpenRouter, or Anthropic Claude and enter that provider's own API key. After a separate consent checkbox, **Load model catalog** asks the selected provider for its current text-model list; it sends no audio or DSP values. Select a model from the populated dropdown. Free and billing labels use provider catalog or pricing details where available; a model being listed does not guarantee account access, free quota, or a successful request. ChatGPT subscriptions are separate from OpenAI API access. Provider keys and account requirements are not interchangeable. See [docs/PROVIDER_API_ACCESS.md](docs/PROVIDER_API_ACCESS.md) for provider-by-provider setup, billing, workspace, and key requirements.

### Resemble Detect setup and use

1. In Resemble, open **Account → API** at [app.resemble.ai/hub/api](https://app.resemble.ai/hub/api) and create a Resemble API token. A Google AI Studio/Gemini key is a different credential and cannot be used in the Resemble field.
2. Confirm that the Resemble account and token have **Deepfake Detect API** access. A token page showing Synthesis endpoints alone does not confirm Detect access; ask Resemble support if Detect is unavailable to the account.
3. Analyze a supported audio file in AegisVoice, then use **Compare with Resemble Detect**. Paste the Resemble token, check the audio-upload consent, choose any optional report sections, and run the check. The token stays in this page for the check and optional report questions; it is not saved by AegisVoice. Select **Clear token** when finished.
4. If the result is still processing, leave the token in the field and select **Check result status**. The page displays the provider label and score, segment scores, structured Intelligence fields, watermark/SynthID, source tracing, and Content Credentials when returned by Resemble. Resemble's score is provider-specific and is not a calibrated probability or combined with the local classifier.
5. After detection finishes, ask a question under **Ask Intelligence about this report**. Check the separate question-consent box, enter a question, and select **Ask Intelligence**; use **Check answer** while Resemble prepares its response. This sends the question and report context to Resemble.
6. **Watermark / SynthID** can incur an additional provider charge. **Let a strong Intelligence finding escalate a non-fake verdict** requires Intelligence and changes Resemble's verdict logic. Zero-retention is requested by AegisVoice; Resemble still processes the upload under its own terms.

Common Resemble errors: **400** means the request or media/options were rejected; **401** means the token was rejected; **402** indicates payment or usage access is required; **403** means account/token access is missing. **520** and **503** are gateway/provider service errors and do not by themselves show that a key is invalid. Retry those later; if they persist, check provider status and contact support with the request time. The Detect [playground](https://app.resemble.ai/hub/detect-playground) and report pages require signing in to Resemble.

## What the measurements mean

The default DSP configuration is 16 kHz mono, 25 ms Hann frames, 10 ms hop, and a 512-point real FFT. The report shows waveform, mean spectrum, a display-only log-frequency STFT spectrogram, and frame-RMS envelope modulation measures, plus time/frequency measures, MFCC, LFCC and explicitly approximate CQT-style cepstra, phase-aware STFT summaries, autocorrelation pitch/HNR estimates, and jitter/shimmer proxies. The feature guide explains each value. The active model can use either the original 11 basic measurements or a 38-value extended DSP/cepstral/prosodic/phase vector; the spectrogram and modulation measures are not classifier inputs. A separate 0–100 indicator describes technical input quality, not authenticity confidence. The full mode resamples with a band-limited rational filter; the offline mode uses finite windowed-sinc interpolation.

With no trained model loaded, values are descriptive only and every decision is INCONCLUSIVE. The Train model view accepts labeled audio folders, individual human/synthetic files, and CSV manifests (`filename,label`, optional `group_id`). It extracts features and discards uploaded audio. Train on all imported and confirmed labels, or choose **Confirmed History results only** to train solely on results you explicitly labeled Human or AI. Compare or train Logistic Regression, RBF SVM, Random Forest, Gradient Boosting, or a small MLP on a grouped holdout; after measuring the holdout, the selected classifier is refit on all selected labeled examples and saved under `data/models/`. New training runs report equal error rate and false-accept / false-reject rates on the held-out split; EER is for evaluation only and does not set the active cutoff. The raw synthetic-class output is mapped to a 1–11 evidence index and the label follows a user-selectable 20–80% cutoff. Both remain uncalibrated and are not real-world probabilities. Independent external validation is needed before consequential use. Predictions are never used as labels. Reset removes the active model but retains labels.

Training files may be up to 100 MB and 10 minutes each. Organize folders with class names (`Real/`, `Fake/`) or map files in the CSV; matching recording stems are kept in the same evaluation group. Do not include uncertain labels or use the model's own predictions as ground truth.

To use the supplied sample corpus, select `D:\notes\sem5\dsp final project\deepfake_audio_demo_samples\deepfake_sample_audio` in the Train model view and choose its `Dataset.csv` manifest. The manifest maps the WAV filenames to Real/Fake labels. These labels are user-provided and not independently verified; resulting metrics are exploratory.

## Local storage and deletion

The database is `data/aegisvoice.sqlite3`. It stores the sanitized filename, SHA-256 of the uploaded bytes, byte count, timestamps, audio metadata, decision, DSP report, and labeled feature records. Raw recordings are processed in memory and not retained. Deleting a history row removes that metadata and report; it cannot delete or alter the source file on your device. History exports as JSON or CSV; each analysis can also be saved as HTML. Model metadata and coefficients/feature importance can be exported as JSON for inspection; this is not a portable checkpoint.

## Troubleshooting

- **Database schema version 2 is newer:** the current full app and WAV fallback both accept schema v2. Do not delete `data/aegisvoice.sqlite3`; stop the old server and start this build with the full command above.
- **Local service unavailable:** run `.\start_aegisvoice.bat` from the project folder, use the exact address printed by the launcher (it may choose 8001 or a later port if 8000 is occupied), and refresh with `Ctrl+F5`.
- **A local analysis is already running:** wait for the current upload to finish. The service allows one analysis upload at a time; the page disables its Analyze button during a request.
- **Add at least 16 labeled recordings:** training needs 16 total recordings, both classes, and at least four distinct groups in each class. In the full app, open Train model and read the live class/group counts. Re-importing the supplied folder skips exact duplicates.
- **Invalid or incomplete request fields:** the browser must send audio plus label, group, and dataset name for training. Reload the updated page with `Ctrl+F5`, select the dataset folder and its CSV manifest, then import from the Train model page.
- **Sample-rate error:** full mode accepts input from 4 kHz to 192 kHz and resamples to 16 kHz. The offline fallback accepts PCM WAV only and has narrower size/duration limits. Confirm full mode is running before analyzing compressed files.
- **DTMF tone files:** select a file to preview it with the built-in player. DTMF is a tone sequence, not speech; the full app reports its DSP measurements and withholds the voice-classifier result.
- **A user-trained classifier is active:** full mode reads `data/models/dsp_logistic.joblib` and the matching label counts from the local database. A score is experimental and uncalibrated. The WAV fallback never loads it.

## Tests

Run the dependency-free checks with:

```powershell
python -m unittest discover -v
```

Nine dependency-backed checks were recorded previously for tone/PSD behavior, silence, anti-alias attenuation, PCM WAV validation, stereo cancellation measurement, in-process request handlers, database persistence, JSON export, and deletion. Those checks predate the latest model-comparison, playback, and local-explanation changes. Start the full local app with the command above to use the current interface.

## Project state

See `docs/IMPLEMENTATION_STATUS.md`, `docs/EXISTING_PROJECT_AUDIT.md`, and `docs/END_TO_END_VERIFICATION.md` for the audited source material, implemented scope, test evidence, and remaining blockers. `LICENSES_AND_ATTRIBUTION.md` records direct dependencies; transitive package inventory should be generated after a clean full-mode install.
