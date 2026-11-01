"""Dependency-free local server for a WAV-only first-run path.

The primary API is app.main (FastAPI). This fallback exists so the project can
be opened and exercised offline with only CPython installed. It accepts PCM WAV
only and uses a pure-Python windowed-sinc resampler and radix-2 FFT.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import math
import re
import sqlite3
import struct
import threading
import uuid
import wave
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from app.stdlib_dsp import analyze, resample_windowed_sinc


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
DOCS = ROOT.parent / "docs"
FALLBACK_MAX_BYTES = 8 * 1024 * 1024
FALLBACK_MAX_SECONDS = 30
FALLBACK_MAX_RATE = 192_000
_SLOTS = threading.BoundedSemaphore(1)


class InputProblem(ValueError):
    pass


@contextmanager
def db_connection(path: Path):
    """Commit successful work and always close the SQLite handle."""
    connection = sqlite3.connect(path, timeout=5)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def safe_name(value: str | None) -> str:
    name = Path((value or "recording").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9._ ()-]", "_", name).strip(" .")
    return (name or "recording")[:160]


def parse_multipart_file(body: bytes, content_type: str) -> tuple[str, bytes]:
    """Extract only the form field named file; never use a submitted path."""
    message = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    if not message.is_multipart():
        raise InputProblem("Send the recording as a multipart form field named 'file'.")
    part = next((p for p in message.iter_parts() if p.get_param("name", header="content-disposition") == "file"), None)
    if part is None:
        raise InputProblem("Choose an audio file first.")
    return safe_name(part.get_filename()), part.get_payload(decode=True) or b""


def decode_pcm_wav(payload: bytes, filename: str) -> tuple[list[float], dict]:
    if not payload:
        raise InputProblem("The uploaded file is empty.")
    if len(payload) > FALLBACK_MAX_BYTES:
        raise InputProblem("Offline WAV mode limits uploads to 8 MB.")
    if Path(filename).suffix.lower() != ".wav":
        raise InputProblem("Offline mode accepts PCM WAV files only. Install requirements.txt for FLAC, OGG, and AIFF support.")
    try:
        reader = wave.open(__import__("io").BytesIO(payload), "rb")
        channels = reader.getnchannels()
        rate = reader.getframerate()
        width = reader.getsampwidth()
        frames = reader.getnframes()
        if reader.getcomptype() != "NONE":
            raise InputProblem("Only uncompressed PCM WAV is supported in offline mode.")
        if rate < 4_000 or rate > FALLBACK_MAX_RATE:
            raise InputProblem("Offline WAV mode supports sample rates from 4 kHz through 192 kHz.")
        if channels < 1 or channels > 2:
            raise InputProblem("Offline WAV mode supports one or two channels.")
        if width not in (1, 2, 3, 4):
            raise InputProblem("Offline WAV mode supports 8, 16, 24, or 32 bit PCM.")
        if frames < 1:
            raise InputProblem("The WAV file contains no audio frames.")
        if frames / rate > FALLBACK_MAX_SECONDS:
            raise InputProblem("Offline WAV mode limits recordings to 30 seconds.")
        raw = reader.readframes(frames + 1)
        expected = frames * channels * width
        if len(raw) != expected:
            raise InputProblem("WAV data is truncated or malformed.")
    except InputProblem:
        raise
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        raise InputProblem("The file is not a valid, uncorrupted PCM WAV recording.") from exc

    mono: list[float] = []
    channel_square_sums = [0.0] * channels
    channel_peaks = [0.0] * channels
    offset = 0
    for _ in range(frames):
        channel_values = []
        for channel in range(channels):
            chunk = raw[offset : offset + width]
            offset += width
            if width == 1:
                value = (chunk[0] - 128) / 128.0
            else:
                integer = int.from_bytes(chunk, "little", signed=False)
                sign = 1 << (width * 8 - 1)
                if integer & sign:
                    integer -= 1 << (width * 8)
                value = integer / float(sign)
            channel_values.append(value)
            channel_square_sums[channel] += value * value
            channel_peaks[channel] = max(channel_peaks[channel], abs(value))
        mono.append(sum(channel_values) / channels)
    source_rms = math.sqrt(sum(total / frames for total in channel_square_sums) / channels)
    mono_rms = math.sqrt(sum(value * value for value in mono) / frames)
    downmix_ratio = mono_rms / source_rms if source_rms > 0 else 1.0
    output = resample_windowed_sinc(mono, rate, 16_000)
    if not output or any(not math.isfinite(value) for value in output):
        raise InputProblem("Audio preprocessing did not produce a finite signal.")
    return output, {
        "format": "WAV", "subtype": f"PCM_{width * 8}",
        "original_sample_rate_hz": rate, "analysis_sample_rate_hz": 16_000,
        "original_channels": channels, "original_frame_count": frames,
        "downmix_rms_ratio": downmix_ratio,
    }


def build_analysis_record(payload: bytes, filename: str) -> dict:
    """Perform the input-to-report computation without retaining audio bytes."""
    filename = safe_name(filename)
    samples, source = decode_pcm_wav(payload, filename)
    dsp = analyze(samples, 16_000)
    duration = dsp["time_domain"]["duration_seconds"]
    report = {
        "analysis_scope": "Descriptive signal measurements only; no trained authenticity classifier was run.",
        "source": source,
        "quality": {"warnings": quality_warnings(source, dsp)},
        "dsp": dsp,
        "decision": {"label": "INCONCLUSIVE", "reason": "No trained, evaluated classifier and validated decision threshold are configured. DSP measurements alone do not establish whether speech is genuine or synthetic.", "classifier_status": "unavailable", "score": None, "probability": None},
        "limitations": [
            "DSP measurements are descriptive and have no validated deepfake classification meaning in this build.",
            "The detector does not identify the speaker, verify truthfulness, or establish provenance.",
            "Unfamiliar generators, codecs, channels, languages, noise, and non-speech audio can affect measurements.",
            "Offline mode accepts uncompressed PCM WAV only and uses a pure-Python windowed-sinc resampler and FFT.",
        ],
    }
    return {
        "analysis_id": str(uuid.uuid4()), "created_at": datetime.now(timezone.utc).isoformat(),
        "filename": filename, "file_sha256": hashlib.sha256(payload).hexdigest(), "file_bytes": len(payload),
        "duration_seconds": duration, "original_sample_rate": source["original_sample_rate_hz"],
        "original_channels": source["original_channels"], "decision": "INCONCLUSIVE", "report": report,
    }


def persist_record(database_path: Path, record: dict) -> None:
    with db_connection(database_path) as connection:
        connection.execute("INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?)", (
            record["analysis_id"], record["created_at"], record["filename"], record["file_sha256"],
            record["file_bytes"], record["duration_seconds"], record["original_sample_rate"],
            record["original_channels"], record["decision"],
            json.dumps(record["report"], ensure_ascii=False, separators=(",", ":")),
        ))


def initialize_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with db_connection(path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > 2:
            raise RuntimeError(f"Database schema version {version} is newer than this AegisVoice build.")
        connection.execute("""CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY,
            created_at DATETIME NOT NULL,
            filename TEXT NOT NULL,
            file_sha256 TEXT NOT NULL,
            file_bytes INTEGER NOT NULL,
            duration_seconds REAL NOT NULL,
            original_sample_rate INTEGER NOT NULL,
            original_channels INTEGER NOT NULL,
            decision TEXT NOT NULL,
            report JSON NOT NULL
        )""")
        # Schema v2 is used by the full FastAPI app, which adds a separate
        # training_examples table. The WAV fallback only needs analyses.
        connection.execute("PRAGMA user_version = 2")


def quality_warnings(source: dict, dsp: dict) -> list[str]:
    time = dsp["time_domain"]
    warnings = []
    if time["duration_seconds"] < 1.0:
        warnings.append("Short recording: frame statistics may be unstable below one second.")
    if time["rms_dbfs"] is None or time["rms_dbfs"] < -50.0:
        warnings.append("Very low level or silence: spectral measurements may not describe speech.")
    if time["clipping_sample_fraction_at_0_999"] > 0:
        warnings.append("Samples reach the near-full-scale clipping check; recording quality may be affected.")
    if source["downmix_rms_ratio"] < 0.1 and source["original_channels"] > 1:
        warnings.append("Stereo fold-down substantially reduced level; anti-phase channel cancellation may be present.")
    return warnings


class Handler(BaseHTTPRequestHandler):
    server_version = "AegisVoiceLocal/0.1"

    @property
    def database_path(self) -> Path:
        return self.server.database_path

    def log_message(self, format, *args):
        # Avoid default logs containing user-controlled filenames or query values.
        print(f"AegisVoice local HTTP {args[1] if len(args) > 1 else ''}")

    def _send(self, status: int, body: bytes, content_type: str = "application/json", extra_headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value, extra_headers: dict | None = None):
        self._send(status, json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), extra_headers=extra_headers)

    def _error(self, status: int, code: str, message: str):
        self._json(status, {"error": {"code": code, "message": message}})

    def do_GET(self):
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/":
            return self._file("index.html", "text/html; charset=utf-8")
        if path == "/model-datasets":
            return self._file("model-datasets.html", "text/html; charset=utf-8")
        if path == "/model-datasets.md":
            catalog = DOCS / "MODEL_CATALOG_AND_DATASETS.md"
            if catalog.is_file():
                return self._send(200, catalog.read_bytes(), "text/markdown; charset=utf-8")
            return self._error(404, "not_found", "The local resource catalog is not available.")
        if path in ("/static/app.js", "/static/styles.css"):
            name = path.rsplit("/", 1)[-1]
            content = "text/javascript; charset=utf-8" if name.endswith(".js") else "text/css; charset=utf-8"
            return self._file(name, content)
        if path == "/api/health":
            return self._json(200, {"ok": True, "application": "AegisVoice", "version": "0.1.0"})
        if path == "/api/status":
            return self._json(200, {
                "application": "AegisVoice", "analysis_mode": "stdlib_wav_dsp_only",
                "classifier_available": False,
                "verdict_policy": "INCONCLUSIVE until a trained and evaluated model with verified label mapping is configured",
                "raw_audio_retention": "disabled; only upload metadata and derived report are stored",
                "accepted_formats": ["WAV (uncompressed PCM)"], "max_upload_mb": 8, "max_duration_seconds": 30,
            })
        if path == "/api/history":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", [20])[0]); offset = int(query.get("offset", [0])[0])
            except ValueError:
                return self._error(400, "request_rejected", "limit and offset must be integers.")
            if not 1 <= limit <= 100 or offset < 0:
                return self._error(400, "request_rejected", "limit must be 1-100 and offset must be non-negative.")
            with db_connection(self.database_path) as connection:
                total = connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
                rows = connection.execute("SELECT analysis_id,created_at,filename,file_sha256,file_bytes,duration_seconds,original_sample_rate,original_channels,decision FROM analyses ORDER BY created_at DESC,analysis_id LIMIT ? OFFSET ?", (limit, offset)).fetchall()
            return self._json(200, {"total": total, "items": [self._record_summary(row) for row in rows]})
        if path == "/api/export/history.json":
            with db_connection(self.database_path) as connection:
                rows = connection.execute("SELECT analysis_id,created_at,filename,file_sha256,file_bytes,duration_seconds,original_sample_rate,original_channels,decision,report FROM analyses ORDER BY created_at DESC").fetchall()
            items = [self._record(row) for row in rows]
            return self._json(200, items, {"Content-Disposition": "attachment; filename=aegisvoice-history.json"})
        if path.startswith("/api/history/"):
            analysis_id = unquote(path.rsplit("/", 1)[-1])
            with db_connection(self.database_path) as connection:
                row = connection.execute("SELECT analysis_id,created_at,filename,file_sha256,file_bytes,duration_seconds,original_sample_rate,original_channels,decision,report FROM analyses WHERE analysis_id=?", (analysis_id,)).fetchone()
            if row is None:
                return self._error(404, "not_found", "Analysis was not found.")
            return self._json(200, self._record(row))
        self._error(404, "not_found", "Route was not found.")

    def _file(self, name: str, content_type: str):
        path = STATIC / name
        if not path.is_file():
            return self._error(404, "not_found", "Application asset was not found.")
        self._send(200, path.read_bytes(), content_type)

    @staticmethod
    def _record_summary(row):
        analysis_id, created_at, filename, digest, file_bytes, duration, sample_rate, channels, decision = row
        return {"analysis_id": analysis_id, "created_at": created_at, "filename": filename,
            "file_sha256": digest, "file_bytes": file_bytes, "duration_seconds": duration,
            "original_sample_rate": sample_rate, "original_channels": channels, "decision": decision}

    @classmethod
    def _record(cls, row):
        result = cls._record_summary(row[:9])
        report = row[9]
        result["report"] = json.loads(report) if isinstance(report, str) else report
        return result

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in {"/api/analyze", "/api/preview"}:
            return self._error(404, "not_found", "Route was not found.")
        if not _SLOTS.acquire(blocking=False):
            return self._error(429, "busy", "A local analysis or preview is already running. Try again shortly.")
        try:
            if path == "/api/preview":
                self._preview_request()
            else:
                self._analyze_request()
        finally:
            _SLOTS.release()

    def _preview_request(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and urlsplit(origin).netloc.casefold() != host.casefold():
            return self._error(403, "origin_rejected", "Requests must come from this local AegisVoice page.")
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            return self._error(400, "invalid_content_length", "Invalid Content-Length header.")
        if length <= 0:
            return self._error(400, "empty_request", "Choose a PCM WAV file first.")
        if length > FALLBACK_MAX_BYTES + 1024 * 1024:
            return self._error(413, "request_too_large", "Offline WAV mode limits uploads to 8 MB.")
        body = self.rfile.read(length)
        if len(body) != length:
            return self._error(400, "incomplete_request", "The upload ended before the complete request was received.")
        try:
            filename, payload = parse_multipart_file(body, self.headers.get("Content-Type", ""))
            samples, _source = decode_pcm_wav(payload, filename)
        except InputProblem as exc:
            return self._error(400, "invalid_audio", str(exc))
        pcm = b"".join(struct.pack("<h", max(-32768, min(32767, round(value * 32767)))) for value in samples)
        output = io.BytesIO()
        with wave.open(output, "wb") as preview:
            preview.setnchannels(1)
            preview.setsampwidth(2)
            preview.setframerate(16_000)
            preview.writeframes(pcm)
        self._send(200, output.getvalue(), "audio/wav", {"Content-Disposition": "inline; filename=preview.wav"})

    def _analyze_request(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and urlsplit(origin).netloc.casefold() != host.casefold():
            return self._error(403, "origin_rejected", "Requests must come from this local AegisVoice page.")
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            return self._error(400, "invalid_content_length", "Invalid Content-Length header.")
        if length <= 0:
            return self._error(400, "empty_request", "Choose a PCM WAV file first.")
        max_body = FALLBACK_MAX_BYTES + 1024 * 1024
        if length > max_body:
            return self._error(413, "request_too_large", "Offline WAV mode limits uploads to 8 MB.")
        body = self.rfile.read(length)
        if len(body) != length:
            return self._error(400, "incomplete_request", "The upload ended before the complete request was received.")
        try:
            filename, payload = parse_multipart_file(body, self.headers.get("Content-Type", ""))
            record = build_analysis_record(payload, filename)
        except InputProblem as exc:
            return self._error(400, "invalid_audio", str(exc))
        except Exception:
            return self._error(422, "analysis_failed", "The WAV file could not be safely decoded or measured.")

        persist_record(self.database_path, record)
        return self._json(200, record)

    def do_DELETE(self):
        path = urlsplit(self.path).path
        if not path.startswith("/api/history/"):
            return self._error(404, "not_found", "Route was not found.")
        analysis_id = unquote(path.rsplit("/", 1)[-1])
        with db_connection(self.database_path) as connection:
            cursor = connection.execute("DELETE FROM analyses WHERE analysis_id=?", (analysis_id,))
        if cursor.rowcount == 0:
            return self._error(404, "not_found", "Analysis was not found.")
        return self._json(200, {"deleted": True, "analysis_id": analysis_id, "raw_audio_deleted": False})

    def log_error(self, status: int, message: str):
        self._error(status, "request_rejected", message)


class AegisHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, database_path: Path):
        self.database_path = database_path
        super().__init__(address, handler)


def serve(host: str = "127.0.0.1", port: int = 8000, data_dir: Path | None = None):
    if host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("AegisVoice is local-only. Bind to 127.0.0.1 or localhost.")
    data_dir = data_dir or Path("data")
    database_path = (data_dir / "aegisvoice.sqlite3").resolve()
    initialize_db(database_path)
    server = AegisHTTPServer((host, port), Handler, database_path)
    print(f"AegisVoice offline WAV mode running at http://{host}:{port}")
    print(f"History database: {database_path}")
    print("Press Ctrl+C to stop. Raw audio files are not saved.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="AegisVoice standard-library WAV server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    serve(args.host, args.port, args.data_dir)


if __name__ == "__main__":
    main()

