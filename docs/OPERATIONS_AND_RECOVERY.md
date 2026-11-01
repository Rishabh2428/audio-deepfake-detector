# Operations and recovery

## Start and stop

For the trained model and common compressed audio formats, start the full FastAPI app. In PowerShell from the project folder run `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`; leave that window open and browse to `http://127.0.0.1:8000`. The Windows launcher also works: `start_aegisvoice.bat`; stop its background server with `stop_aegisvoice.bat`. It checks dependencies and may select the reduced WAV fallback if they are missing. No package is installed automatically.

The reduced fallback is useful only for PCM WAV DSP inspection; it does not load the model or offer training:

```powershell
python -m app.local_server --host 127.0.0.1 --port 8000 --data-dir data
```

Press Ctrl+C to stop. The full FastAPI command is in the root README. Set `AEGISVOICE_OPEN_BROWSER=1` to have the Windows script open a browser.

## Logs and status

The Windows launcher writes `data/server.log`, `data/server-error.log`, and `data/server.pid.json`. `/api/health` checks liveness; `/api/status` reports the current mode, classifier availability, accepted formats, and limits. `/api/training/status` shows saved class/group counts and model metrics in full mode.

## Database backup and restore

Stop the server before copying `data/aegisvoice.sqlite3` and `data/models/dsp_logistic.joblib`. Keep backups local and protect them like metadata containing filenames and file hashes. Restore by stopping AegisVoice, placing the files at the same paths, then starting the application. This manual process has not been exercised; no automated backup/restore utility is implemented.

## Troubleshooting

- Port in use: stop the existing AegisVoice process or choose a new port consistently in both startup configuration and browser URL.
- Full-mode packages missing: create a Python 3.12 virtual environment and install `requirements.txt`; startup does not install anything.
- Offline fallback refuses FLAC/OGG/AIFF: install the full requirements and confirm the startup script selected FastAPI full-format mode.
- Corrupt/unsupported file: use an uncompressed PCM WAV in fallback, or check full-mode format support and duration/rate/channel limits.
- Database schema: this build supports schema version 2. Stop an older server and use the current full command above; do not delete `data/aegisvoice.sqlite3` to clear the error.
- Training minimum: 16 labeled recordings total, both classes, and 4 distinct recording groups per class. Check the live counts on Train model; exact duplicate files are skipped.
- DTMF tone clips: they can be previewed and analyzed for DSP measurements, but the voice classifier intentionally withholds inference because DTMF is not speech.
