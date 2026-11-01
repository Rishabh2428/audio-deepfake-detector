"""Adapter for the optional Resemble Detect API."""

from __future__ import annotations

import json
import mimetypes
import re
import secrets
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://app.resemble.ai/api/v2/detect"
_UUID_PATTERN = re.compile(r"^[0-9a-fA-F-]{16,64}$")


def _as_float(value):
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _decode_response(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("Resemble returned an unreadable response.") from None
    if not isinstance(value, dict) or not isinstance(value.get("item"), dict):
        raise ValueError("Resemble returned an unexpected response.")

    item = value["item"]
    metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    label = str(metrics.get("label") or "").casefold()
    status = str(item.get("status") or "unknown").casefold()
    intelligence = item.get("intelligence")
    intelligence_results = intelligence if isinstance(intelligence, list) else ([intelligence] if isinstance(intelligence, dict) else [])
    intelligence_statuses = [str(result.get("status") or "unknown").casefold()
                             for result in intelligence_results if isinstance(result, dict)]
    watermark = item.get("watermark") if isinstance(item.get("watermark"), dict) else None
    watermark_status = str(watermark.get("status") or "unknown").casefold() if watermark else None

    # Detect, Intelligence, and Watermark can finish independently. Keep the
    # job available for another status check while a requested child is active.
    pending_statuses = {"pending", "queued", "processing", "running"}
    pending = status in pending_statuses or any(s in pending_statuses for s in intelligence_statuses)
    pending = pending or bool(watermark and watermark_status in pending_statuses)
    verdict = {"real": "HUMAN", "fake": "SYNTHETIC"}.get(label, "INCONCLUSIVE")
    if status == "failed":
        verdict = "INCONCLUSIVE"

    return {
        "provider": "Resemble Detect",
        "status": status,
        "pending": pending,
        "verdict": verdict,
        "provider_error": item.get("error_message") or item.get("failure_reason"),
        "provider_label": label or None,
        "aggregated_score": _as_float(metrics.get("aggregated_score")),
        "consistency": _as_float(metrics.get("consistency")),
        "chunk_scores": metrics.get("score"),
        "visualization": metrics.get("image") or item.get("visualization"),
        "detection_id": item.get("uuid"),
        "zero_retention_requested": bool(item.get("zero_retention_mode", False)),
        "file_deleted_at": item.get("file_deleted_at"),
        "intelligence": intelligence,
        "intelligence_status": intelligence_statuses[-1] if intelligence_statuses else None,
        "watermark": watermark,
        "audio_source_tracing": item.get("audio_source_tracing"),
        "c2pa_manifest": item.get("c2pa_manifest"),
        "report_url": item.get("url") if isinstance(item.get("url"), str) else None,
        "warning": "Resemble's label and score are provider-specific. Its score is not a calibrated probability or combined with the local model.",
    }


def _provider_error(exc: HTTPError) -> ValueError:
    if exc.code == 400:
        return ValueError(
            "Resemble rejected this request (HTTP 400). Check that the upload contains one supported media file "
            "and that the request options are valid. If the error is 403, ask Resemble to enable Deepfake Detect API access for the account."
        )
    if exc.code == 401:
        return ValueError("Resemble rejected the API token (HTTP 401). Check the token and its access.")
    if exc.code == 402:
        return ValueError("Resemble requires available API access or credits for this detection (HTTP 402).")
    if exc.code == 403:
        return ValueError("This Resemble account or token does not have Detect API access enabled (HTTP 403).")
    if exc.code == 429:
        return ValueError("Resemble rate limit reached (HTTP 429). Wait and try again.")
    if exc.code == 520:
        return ValueError(
            "Resemble's gateway returned HTTP 520. This is an upstream or edge-service error, "
            "not proof that the API token is wrong. Try again later; if it persists, contact Resemble support with the time of the request."
        )
    if exc.code in (500, 502, 503, 504):
        return ValueError(
            f"Resemble is temporarily unavailable (HTTP {exc.code}). Try again later or check Resemble's service status."
        )
    return ValueError(f"Resemble returned HTTP {exc.code}. Check the request and account access.")


def _validate_key(api_key: str) -> str:
    if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 500 or "\r" in api_key or "\n" in api_key:
        raise ValueError("Enter a valid Resemble API token for this request.")
    return api_key.strip()


def submit_audio(
    payload: bytes,
    filename: str,
    api_key: str,
    *,
    intelligence: bool = False,
    detect_watermark: bool = False,
    audio_source_tracing: bool = False,
    infer_from_intelligence: bool = False,
    visualize: bool = False,
) -> dict:
    if not payload:
        raise ValueError("Choose an audio file first.")
    key = _validate_key(api_key)
    if infer_from_intelligence and not intelligence:
        raise ValueError("Enable Intelligence before allowing its findings to affect the Detect verdict.")

    boundary = "aegisvoice-" + secrets.token_hex(16)
    safe_name = filename.replace("\r", "").replace("\n", "").replace('"', "_")[:120] or "audio.wav"
    mime_type = mimetypes.guess_type(safe_name)[0] or "application/octet-stream"
    fields = {
        "zero_retention_mode": "true",
        "intelligence": str(bool(intelligence)).lower(),
        "detect_watermark": str(bool(detect_watermark)).lower(),
        "audio_source_tracing": str(bool(audio_source_tracing)).lower(),
        "infer_from_intelligence": str(bool(infer_from_intelligence)).lower(),
        "visualize": str(bool(visualize)).lower(),
    }
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{safe_name}\"\r\n"
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8") + payload
    ]
    for name, field_value in fields.items():
        parts.append(
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{field_value}".encode("ascii")
        )
    body = b"".join(parts) + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = Request(BASE_URL, data=body, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Prefer": "wait",
        "Accept": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=120) as response:
            return _decode_response(response.read(2_000_001))
    except HTTPError as exc:
        raise _provider_error(exc) from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("Could not connect to Resemble. Check your internet connection and try again.") from None


def get_detection(detection_id: str, api_key: str) -> dict:
    if not _UUID_PATTERN.fullmatch(detection_id):
        raise ValueError("Invalid Resemble detection ID.")
    key = _validate_key(api_key)
    request = Request(f"{BASE_URL}/{detection_id}", headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json",
    }, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            return _decode_response(response.read(2_000_001))
    except HTTPError as exc:
        raise _provider_error(exc) from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("Could not reach Resemble to check the detection status.") from None


def _decode_question_response(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("Resemble returned an unreadable Intelligence answer.") from None
    item = value.get("item") if isinstance(value, dict) else None
    if not isinstance(item, dict):
        raise ValueError("Resemble returned an unexpected Intelligence response.")
    return {
        "question_id": item.get("uuid"),
        "detection_id": item.get("detect_uuid"),
        "query": item.get("query"),
        "answer": item.get("answer"),
        "status": str(item.get("status") or "unknown").casefold(),
        "error_message": item.get("error_message"),
    }


def submit_intelligence_question(detection_id: str, query: str, api_key: str) -> dict:
    if not _UUID_PATTERN.fullmatch(detection_id):
        raise ValueError("Invalid Resemble detection ID.")
    if not query.strip() or len(query) > 1000:
        raise ValueError("Enter a question of 1 to 1000 characters.")
    key = _validate_key(api_key)
    body = json.dumps({"query": query.strip()}, separators=(",", ":")).encode("utf-8")
    request = Request(f"{BASE_URL}s/{detection_id}/intelligence", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json",
    }, method="POST")
    try:
        with urlopen(request, timeout=35) as response:
            return _decode_question_response(response.read(1_000_001))
    except HTTPError as exc:
        if exc.code == 400:
            raise ValueError("Resemble rejected the question. Enter a question about this completed report and try again.") from None
        if exc.code == 422:
            raise ValueError("Resemble can answer questions only after this detection is complete.") from None
        if exc.code == 404:
            raise ValueError("Resemble could not find this detection. Run a new Detect check and try again.") from None
        raise _provider_error(exc) from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("Could not connect to Resemble Intelligence. Check your internet connection and try again.") from None


def get_intelligence_answer(detection_id: str, question_id: str, api_key: str) -> dict:
    if not _UUID_PATTERN.fullmatch(detection_id) or not _UUID_PATTERN.fullmatch(question_id):
        raise ValueError("Invalid Resemble detection or question ID.")
    key = _validate_key(api_key)
    request = Request(f"{BASE_URL}s/{detection_id}/intelligence/{question_id}", headers={
        "Authorization": f"Bearer {key}", "Accept": "application/json",
    }, method="GET")
    try:
        with urlopen(request, timeout=25) as response:
            return _decode_question_response(response.read(1_000_001))
    except HTTPError as exc:
        if exc.code == 404:
            raise ValueError("Resemble could not find this Intelligence question.") from None
        raise _provider_error(exc) from None
    except (URLError, TimeoutError, OSError):
        raise ValueError("Could not reach Resemble to check the Intelligence answer.") from None
