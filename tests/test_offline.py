from __future__ import annotations

import json
import io
import math
import sqlite3
import struct
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

from app.local_server import Handler, build_analysis_record, db_connection, decode_pcm_wav, initialize_db, parse_multipart_file, persist_record
from app.stdlib_dsp import analyze, resample_windowed_sinc


def make_wav(samples, rate=16_000, channels=1):
    buffer = __import__("io").BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        if channels == 1:
            values = samples
        else:
            values = [sample for value in samples for sample in ((value, -value) if channels == 2 else (value,) * channels)]
        writer.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, int(value * 32767)))) for value in values))
    return buffer.getvalue()


class DSPTests(unittest.TestCase):
    def test_fft_and_psd_peak_match_one_kilohertz_tone(self):
        rate = 16_000
        samples = [0.5 * math.sin(2 * math.pi * 1_000 * i / rate) for i in range(rate)]
        report = analyze(samples, rate)
        psd = report["power_spectral_density"]
        peak_frequency = psd["frequency_hz"][psd["power_db_per_hz"].index(max(psd["power_db_per_hz"]))]
        self.assertAlmostEqual(peak_frequency, 1_000, delta=35)
        self.assertAlmostEqual(report["time_domain"]["rms_amplitude"], 0.5 / math.sqrt(2), delta=0.002)
        self.assertAlmostEqual(report["frequency_domain"]["spectral_centroid_mean_hz"], 1_000, delta=80)
        self.assertTrue(all(math.isfinite(value) for value in psd["power_db_per_hz"]))

    def test_silence_has_no_fabricated_dbfs_value(self):
        report = analyze([0.0] * 8_000)
        self.assertIsNone(report["time_domain"]["rms_dbfs"])
        self.assertEqual(report["time_domain"]["low_energy_frame_fraction_relative_minus_40_db"], 1.0)
        self.assertEqual(report["frequency_domain"]["spectral_centroid_mean_hz"], 0.0)
        self.assertEqual(report["frequency_domain"]["spectral_flatness_mean"], 0.0)

    def test_dc_is_reported_and_removed_for_spectral_statistics(self):
        rate = 16_000
        samples = [0.08 + 0.4 * math.sin(2 * math.pi * 800 * i / rate) for i in range(rate)]
        report = analyze(samples, rate)
        self.assertAlmostEqual(report["time_domain"]["dc_offset_mean_amplitude"], 0.08, delta=1e-6)
        self.assertAlmostEqual(report["frequency_domain"]["spectral_centroid_mean_hz"], 800, delta=80)

    def test_windowed_sinc_resampling_preserves_duration_and_rejects_alias_band(self):
        source_rate = 48_000
        low = [0.4 * math.sin(2 * math.pi * 1_000 * i / source_rate) for i in range(source_rate // 4)]
        converted = resample_windowed_sinc(low, source_rate, 16_000)
        self.assertEqual(len(converted), 4_000)
        rms = math.sqrt(sum(value * value for value in converted) / len(converted))
        self.assertAlmostEqual(rms, 0.4 / math.sqrt(2), delta=0.02)
        high = [0.4 * math.sin(2 * math.pi * 12_000 * i / source_rate) for i in range(source_rate // 4)]
        rejected = resample_windowed_sinc(high, source_rate, 16_000)
        high_rms = math.sqrt(sum(value * value for value in rejected[100:-100]) / len(rejected[100:-100]))
        self.assertLess(high_rms, 0.04)


class AudioDecodeTests(unittest.TestCase):
    def test_pcm_wav_decode_and_stereo_cancellation_measure(self):
        samples = [0.3 * math.sin(2 * math.pi * 300 * i / 16_000) for i in range(8_000)]
        decoded, metadata = decode_pcm_wav(make_wav(samples, channels=2), "voice.wav")
        self.assertEqual(metadata["original_channels"], 2)
        self.assertEqual(metadata["original_sample_rate_hz"], 16_000)
        self.assertLess(metadata["downmix_rms_ratio"], 1e-6)
        self.assertEqual(len(decoded), 8_000)

    def test_rejects_corrupt_and_unsupported_inputs(self):
        with self.assertRaises(ValueError):
            decode_pcm_wav(b"not a wave", "voice.wav")
        with self.assertRaises(ValueError):
            decode_pcm_wav(b"audio", "voice.mp3")

    def test_size_limit_and_upload_name_are_bounded(self):
        with self.assertRaises(ValueError):
            decode_pcm_wav(b"x" * (8 * 1024 * 1024 + 1), "large.wav")
        body = (b'--b\r\nContent-Disposition: form-data; name="file"; filename="../../voice.wav"\r\n'
                b'Content-Type: audio/wav\r\n\r\nnot-a-wave\r\n--b--\r\n')
        filename, content = parse_multipart_file(body, "multipart/form-data; boundary=b")
        self.assertEqual(filename, "voice.wav")
        self.assertEqual(content, b"not-a-wave")


class PersistenceWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = Path.cwd() / "tests" / ".runtime"
        self.temp.mkdir(parents=True, exist_ok=True)
        self.database_path = self.temp / "aegisvoice.sqlite3"
        initialize_db(self.database_path)

    def tearDown(self):
        pass

    def test_measurements_history_export_restart_and_delete(self):
        source = [0.35 * math.sin(2 * math.pi * 440 * i / 16_000) for i in range(16_000)]
        analysis = build_analysis_record(make_wav(source), "test-tone.wav")
        self.assertEqual(analysis["decision"], "INCONCLUSIVE")
        self.assertIsNone(analysis["report"]["decision"]["probability"])
        self.assertEqual(analysis["report"]["dsp"]["time_domain"]["sample_count"], 16_000)
        self.assertEqual(len(analysis["file_sha256"]), 64)
        persist_record(self.database_path, analysis)
        with db_connection(self.database_path) as connection:
            row = connection.execute("SELECT report FROM analyses WHERE analysis_id=?", (analysis["analysis_id"],)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(json.loads(row[0])["decision"]["label"], "INCONCLUSIVE")
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0], 1)

        initialize_db(self.database_path)  # Simulate startup after a process restart.
        with db_connection(self.database_path) as connection:
            row = connection.execute("SELECT analysis_id,report FROM analyses").fetchone()
            self.assertEqual(row[0], analysis["analysis_id"])
            report = json.loads(row[1])
            self.assertEqual(report["dsp"]["time_domain"]["duration_seconds"], 1.0)
            exported = json.loads(json.dumps([{**analysis, "report": report}]))
            self.assertEqual(exported[0]["file_sha256"], analysis["file_sha256"])
            connection.execute("DELETE FROM analyses WHERE analysis_id=?", (analysis["analysis_id"],))
        with db_connection(self.database_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0], 0)

    def test_in_process_http_handlers_cover_upload_history_and_delete(self):
        class MemoryHandler(Handler):
            database_path = self.database_path

            def __init__(self, path, method, headers=None, body=b""):
                self.server = SimpleNamespace(database_path=type(self).database_path)
                self.path = path
                self.command = method
                self.headers = headers or {}
                self.rfile = io.BytesIO(body)
                self.wfile = io.BytesIO()
                self.status_code = None
                self.response_headers = {}

            def send_response(self, status, message=None):
                self.status_code = status

            def send_header(self, key, value):
                self.response_headers[key] = value

            def end_headers(self):
                pass

        page = MemoryHandler("/", "GET")
        page.do_GET()
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.response_headers["Content-Type"], "text/html; charset=utf-8")
        self.assertIn(b"AEGISVOICE", page.wfile.getvalue())
        self.assertIn("default-src 'self'", page.response_headers["Content-Security-Policy"])

        script = MemoryHandler("/static/app.js", "GET")
        script.do_GET()
        self.assertEqual(script.status_code, 200)
        self.assertEqual(script.response_headers["Content-Type"], "text/javascript; charset=utf-8")

        status = MemoryHandler("/api/status", "GET")
        status.do_GET()
        self.assertFalse(json.loads(status.wfile.getvalue())["classifier_available"])

        catalog = MemoryHandler("/model-datasets", "GET")
        catalog.do_GET()
        self.assertEqual(catalog.status_code, 200)
        self.assertIn(b"ASVspoof 2021", catalog.wfile.getvalue())

        tone = [0.25 * math.sin(2 * math.pi * 700 * i / 16_000) for i in range(16_000)]
        audio = make_wav(tone)
        boundary = "aegisvoice-boundary"
        body = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode()
            + audio
            + f"\r\n--{boundary}--\r\n".encode()
        )
        preview = MemoryHandler("/api/preview", "POST", {
            "Content-Length": str(len(body)), "Content-Type": f"multipart/form-data; boundary={boundary}"
        }, body)
        preview.do_POST()
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.response_headers["Content-Type"], "audio/wav")
        self.assertTrue(preview.wfile.getvalue().startswith(b"RIFF"))
        with wave.open(io.BytesIO(preview.wfile.getvalue()), "rb") as prepared:
            self.assertEqual(prepared.getframerate(), 16_000)
            self.assertEqual(prepared.getnchannels(), 1)

        request = MemoryHandler("/api/analyze", "POST", {
            "Content-Length": str(len(body)), "Content-Type": f"multipart/form-data; boundary={boundary}"
        }, body)
        rejected = MemoryHandler("/api/analyze", "POST", {
            "Content-Length": str(len(body)), "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Origin": "https://elsewhere.invalid", "Host": "127.0.0.1:8000"
        }, body)
        rejected.do_POST()
        self.assertEqual(rejected.status_code, 403)
        request.do_POST()
        self.assertEqual(request.status_code, 200)
        record = json.loads(request.wfile.getvalue())
        self.assertEqual(record["decision"], "INCONCLUSIVE")
        self.assertEqual(record["report"]["source"]["analysis_sample_rate_hz"], 16_000)

        history = MemoryHandler("/api/history?limit=20", "GET")
        history.do_GET()
        self.assertEqual(json.loads(history.wfile.getvalue())["items"][0]["analysis_id"], record["analysis_id"])

        export = MemoryHandler("/api/export/history.json", "GET")
        export.do_GET()
        self.assertEqual(json.loads(export.wfile.getvalue())[0]["file_sha256"], record["file_sha256"])

        initialize_db(self.database_path)
        detail = MemoryHandler(f"/api/history/{record['analysis_id']}", "GET")
        detail.do_GET()
        self.assertEqual(json.loads(detail.wfile.getvalue())["analysis_id"], record["analysis_id"])

        remove = MemoryHandler(f"/api/history/{record['analysis_id']}", "DELETE")
        remove.do_DELETE()
        self.assertEqual(remove.status_code, 200)
        self.assertTrue(json.loads(remove.wfile.getvalue())["deleted"])


if __name__ == "__main__":
    unittest.main()
