"""FastAPI entry point for the local AegisVoice DSP inspection slice."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
import csv
import io
import wave
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
import numpy as np

from app.audio import AudioInputError, decode_audio
from app.config import Settings
from app.database import AnalysisRecord, TrainingExample, initialize_database, make_engine, make_session_factory, serialize_record
from app.learning import ALGORITHMS, compare_models, load_model, score_report, set_model_threshold, stored_feature_vectors, train_model
from app.dsp import analyze
from app.schemas import AnalysisResponse, HealthResponse, HistoryResponse, StatusResponse
from app.llm_explainer import PROVIDER_NAMES, explain_measurements, list_provider_models
from app.resemble_detect import (
    get_detection as get_resemble_detection,
    get_intelligence_answer as get_resemble_intelligence_answer,
    submit_audio as submit_resemble_audio,
    submit_intelligence_question as submit_resemble_intelligence_question,
)


VERSION = "0.1.0"
STATIC_DIR = Path(__file__).parent / "static"
DOCS_DIR = Path(__file__).parent.parent / "docs"


class TrainModelRequest(BaseModel):
    algorithm: Literal["logistic_regression", "svm_rbf", "random_forest", "gradient_boosting", "small_neural_network"] = "logistic_regression"
    feature_set: Literal["basic", "advanced"] = "basic"
    source: Literal["all", "confirmed_history"] = "all"


class ExplainRequest(BaseModel):
    provider: Literal["gemini", "openai", "nvidia", "openrouter", "anthropic"]
    model: str
    api_key: str
    dsp: dict
    consent: bool
    workspace_id: str = ""


class ProviderModelsRequest(BaseModel):
    provider: Literal["gemini", "openai", "nvidia", "openrouter", "anthropic"]
    api_key: str
    consent: bool = False
    workspace_id: str = ""


class ResembleQuestionRequest(BaseModel):
    query: str
    consent: bool = False


class ThresholdRequest(BaseModel):
    threshold: float


def _safe_filename(value: str | None) -> str:
    candidate = Path((value or "recording").replace("\\", "/")).name
    candidate = re.sub(r"[^A-Za-z0-9._ ()-]", "_", candidate).strip(" .")
    return (candidate or "recording")[:160]


def _quality_messages(audio, dsp_report: dict) -> list[str]:
    values = dsp_report["time_domain"]
    messages: list[str] = []
    frequency = dsp_report["frequency_domain"]
    if values["duration_seconds"] < 1.0:
        messages.append("Short recording: frame statistics may be unstable below one second.")
    if values["rms_dbfs"] is None or values["rms_dbfs"] < -50.0:
        messages.append("Very low level or silence: spectral measurements may not describe speech.")
    if values["clipping_sample_fraction_at_0_999"] > 0:
        messages.append("Samples reach the near-full-scale clipping check; recording quality may be affected.")
    if audio.downmix_rms_ratio < 0.1 and audio.original_channels > 1:
        messages.append("Stereo fold-down substantially reduced level; anti-phase channel cancellation may be present.")
    if frequency["spectral_entropy_normalized_mean"] < 0.38 and frequency["spectral_flatness_mean"] < 0.005:
        messages.append("This signal is strongly tonal and may not be speech (for example, DTMF tones). The voice classifier was withheld.")
    return messages


def _classifier_block_reasons(audio, dsp_report: dict) -> list[str]:
    """Block non-speech or unusable input; keep lesser quality notices advisory."""
    time = dsp_report["time_domain"]
    frequency = dsp_report["frequency_domain"]
    reasons: list[str] = []
    if time["duration_seconds"] < 1.0:
        reasons.append("recording is shorter than one second")
    if time["rms_dbfs"] is None or time["rms_dbfs"] < -50.0:
        reasons.append("signal is silent or too quiet")
    if audio.downmix_rms_ratio < 0.1 and audio.original_channels > 1:
        reasons.append("stereo fold-down caused strong channel cancellation")
    if frequency["spectral_entropy_normalized_mean"] < 0.38 and frequency["spectral_flatness_mean"] < 0.005:
        reasons.append("signal is strongly tonal and does not resemble speech")
    return reasons


def _input_quality_score(audio, dsp_report: dict) -> dict:
    """Transparent technical-quality indicator; never a deepfake confidence."""
    time = dsp_report["time_domain"]
    score = 100
    reasons: list[str] = []
    duration = time["duration_seconds"]
    if duration < 1.0:
        score -= 25; reasons.append("very short clip")
    elif duration < 2.0:
        score -= 12; reasons.append("short clip")
    rms_dbfs = time["rms_dbfs"]
    if rms_dbfs is None or rms_dbfs < -50.0:
        score -= 25; reasons.append("very low level or silence")
    elif rms_dbfs < -35.0:
        score -= 10; reasons.append("quiet signal")
    clipping = time["clipping_sample_fraction_at_0_999"]
    if clipping > 0.01:
        score -= 20; reasons.append("clipping detected")
    elif clipping > 0:
        score -= 8; reasons.append("near-full-scale samples")
    low_energy = time["low_energy_frame_fraction_relative_minus_40_db"]
    if low_energy > 0.75:
        score -= 18; reasons.append("mostly low-energy frames")
    elif low_energy > 0.45:
        score -= 8; reasons.append("many low-energy frames")
    if audio.downmix_rms_ratio < 0.1 and audio.original_channels > 1:
        score -= 18; reasons.append("strong stereo cancellation")
    if audio.original_sample_rate < 8_000:
        score -= 12; reasons.append("low source sample rate")
    score = max(0, min(100, score))
    return {
        "score_0_to_100": score,
        "meaning": "Technical input-quality indicator only; not model confidence, authenticity probability, or evidence of synthesis.",
        "factors": reasons,
    }


class AnalyzeBodyLimitMiddleware:
    """Bound the complete multipart request body before Starlette parses it."""

    def __init__(self, app, *, limit: int):
        self.app = app
        self.limit = limit
        self.analysis_slots = threading.BoundedSemaphore(1)
        self.training_slots = threading.BoundedSemaphore(1)

    async def __call__(self, scope, receive, send):
        path = scope.get("path")
        if scope.get("type") != "http" or scope.get("method") != "POST" or path not in {"/api/analyze", "/api/preview", "/api/training/samples", "/api/external-detect/resemble"}:
            await self.app(scope, receive, send)
            return

        # Leave room for multipart boundaries and the file part headers while
        # the decoder independently enforces the exact 50 MB audio-byte cap.
        request_limit = self.limit + 1024 * 1024 if path in {"/api/analyze", "/api/preview", "/api/external-detect/resemble"} else 101 * 1024 * 1024

        slot = self.analysis_slots if path in {"/api/analyze", "/api/preview", "/api/external-detect/resemble"} else self.training_slots
        if not slot.acquire(blocking=False):
            message = "An audio analysis, preview, or external detection is already running. Try again shortly." if path in {"/api/analyze", "/api/preview", "/api/external-detect/resemble"} else "A dataset file is already being imported. Try again shortly."
            await self._json(send, 429, "busy", message)
            return
        try:
            await self._bounded_body(scope, receive, send, request_limit)
        finally:
            slot.release()

    async def _bounded_body(self, scope, receive, send, request_limit: int):

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        origin = headers.get(b"origin")
        host = headers.get(b"host", b"").decode("latin-1")
        if origin:
            try:
                origin_host = urlsplit(origin.decode("latin-1")).netloc
            except ValueError:
                origin_host = ""
            if not origin_host or origin_host.casefold() != host.casefold():
                await self._json(send, 403, "origin_rejected", "Requests must come from this local AegisVoice page.")
                return
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > request_limit:
                    await self._json(send, 413, "request_too_large", "Upload exceeds the configured request limit.")
                    return
            except ValueError:
                await self._json(send, 400, "invalid_content_length", "Invalid Content-Length header.")
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > request_limit:
                await self._json(send, 413, "request_too_large", "Upload exceeds the configured request limit.")
                return
            if not message.get("more_body", False):
                break

        sent = False

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _json(send, status: int, code: str, message: str):
        body = json.dumps({"error": {"code": code, "message": message}}, separators=(",", ":")).encode()
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def create_app(*, database_url: str | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    resolved_url = database_url or f"sqlite:///{(settings.data_dir / 'aegisvoice.sqlite3').resolve().as_posix()}"
    engine = make_engine(resolved_url)
    session_factory = make_session_factory(engine)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        initialize_database(engine)
        yield
        engine.dispose()

    app = FastAPI(
        title="AegisVoice",
        version=VERSION,
        description="Local DSP inspection with an optional experimental user-trained audio classifier.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.add_middleware(AnalyzeBodyLimitMiddleware, limit=settings.max_upload_bytes)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob: data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def get_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        missing = [str(item.get("loc", ["field"])[-1]) for item in exc.errors() if item.get("type") == "missing"]
        message = "Missing required fields: " + ", ".join(missing) + "." if missing else "One or more request fields have an invalid value. Check file type, label, recording group, and dataset name."
        return JSONResponse(status_code=422, content={"error": {"code": "invalid_request", "message": message}})

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        code = "not_found" if exc.status_code == 404 else "request_rejected"
        message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": code, "message": message}})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/model-datasets", include_in_schema=False)
    def model_dataset_catalog():
        return FileResponse(STATIC_DIR / "model-datasets.html", media_type="text/html; charset=utf-8")

    @app.get("/model-datasets.md", include_in_schema=False)
    def model_dataset_catalog_notes():
        return FileResponse(DOCS_DIR / "MODEL_CATALOG_AND_DATASETS.md", media_type="text/markdown; charset=utf-8")

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        return HealthResponse(ok=True, application="AegisVoice", version=VERSION)

    @app.get("/api/status", response_model=StatusResponse)
    def status():
        model_bundle = load_model(settings.data_dir / "models" / "dsp_logistic.joblib")
        return StatusResponse(
            application="AegisVoice",
            analysis_mode="experimental_dsp_ml" if model_bundle else "descriptive_dsp_only",
            classifier_available=bool(model_bundle),
            verdict_policy="Experimental uncalibrated model score; human review required" if model_bundle else "INCONCLUSIVE until a model is trained from labeled samples",
            raw_audio_retention="disabled; only upload metadata and derived report are stored",
            accepted_formats=["WAV", "FLAC", "OGG/Opus", "AIFF", "MP3", "M4A/MP4", "AAC", "WMA", "WebM"],
            max_upload_mb=settings.max_upload_bytes // (1024 * 1024),
            max_duration_seconds=settings.max_duration_seconds,
            model_algorithm=model_bundle.get("algorithm_name") if model_bundle else None,
            model_feature_set=model_bundle.get("feature_set", "basic") if model_bundle else None,
            model_feature_count=len(model_bundle.get("feature_names", [])) if model_bundle else 0,
            model_threshold=model_bundle.get("threshold", 0.5) if model_bundle else None,
        )

    @app.post("/api/preview")
    def prepare_browser_preview(file: Annotated[UploadFile, File(description="Audio to decode locally for browser playback")]):
        """Return a temporary browser-compatible PCM WAV; never save the source or preview."""
        filename = _safe_filename(file.filename)
        payload = file.file.read(settings.max_upload_bytes + 1)
        try:
            audio = decode_audio(payload, filename=filename, max_bytes=settings.max_upload_bytes,
                                 max_duration_seconds=settings.max_duration_seconds,
                                 target_sample_rate=settings.target_sample_rate)
        except AudioInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        pcm = np.clip(np.rint(audio.samples.astype(np.float64) * 32767.0), -32768, 32767).astype("<i2", copy=False)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(settings.target_sample_rate)
            wav.writeframes(pcm.tobytes())
        return Response(content=output.getvalue(), media_type="audio/wav",
                        headers={"Cache-Control": "no-store", "Content-Disposition": "inline; filename=preview.wav"})

    @app.post("/api/explain")
    def explain_report(request: ExplainRequest, http_request: Request):
        if request.consent is not True:
            raise HTTPException(status_code=400, detail="Opt in before sending DSP measurements to an online provider.")
        origin = http_request.headers.get("origin")
        if origin and urlsplit(origin).netloc.casefold() != http_request.headers.get("host", "").casefold():
            raise HTTPException(status_code=403, detail="Requests must come from this local AegisVoice page.")
        if len(request.api_key) > 500 or len(request.model) > 120:
            raise HTTPException(status_code=422, detail="The provider key or model name is too long.")
        # Strictly construct the external payload: no audio, file metadata,
        # history identifiers, classifier score, class label, or verdict.
        paths = {
            "duration_seconds": ("time_domain", "duration_seconds"),
            "rms_dbfs": ("time_domain", "rms_dbfs"),
            "peak_dbfs": ("time_domain", "peak_dbfs"),
            "spectral_centroid_mean_hz": ("frequency_domain", "spectral_centroid_mean_hz"),
            "spectral_rolloff_85_mean_hz": ("frequency_domain", "spectral_rolloff_85_mean_hz"),
            "spectral_flatness_mean": ("frequency_domain", "spectral_flatness_mean"),
            "spectral_entropy_normalized_mean": ("frequency_domain", "spectral_entropy_normalized_mean"),
            "spectral_flux_mean": ("frequency_domain", "spectral_flux_mean"),
            "mean_f0_hz": ("prosodic", "mean_f0_hz"),
            "jitter_local_proxy_percent": ("prosodic", "jitter_local_proxy_percent"),
            "shimmer_local_proxy_percent": ("prosodic", "shimmer_local_proxy_percent"),
            "autocorrelation_hnr_proxy_mean_db": ("prosodic", "autocorrelation_hnr_proxy_mean_db"),
            "voiced_frame_fraction": ("prosodic", "voiced_frame_fraction"),
            "spectral_slope_mean_db_per_octave": ("extended_spectral", "spectral_slope_mean_db_per_octave"),
            "spectral_crest_mean": ("extended_spectral", "spectral_crest_mean"),
            "mean_residual_phase_variation_radians": ("phase", "mean_residual_phase_variation_radians"),
            "residual_phase_jump_ratio": ("phase", "residual_phase_jump_ratio"),
            "group_delay_outlier_ratio": ("phase", "group_delay_outlier_ratio"),
        }
        safe_values = {}
        for name, path in paths.items():
            value = request.dsp
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and np.isfinite(value):
                safe_values[name] = round(float(value), 6)
        try:
            explanation = explain_measurements(request.provider, request.model.strip(), request.api_key.strip(), safe_values, request.workspace_id.strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"provider": PROVIDER_NAMES[request.provider], "model": request.model.strip(), "explanation": explanation,
                "sent_fields": list(safe_values), "retention": "The key and request are not saved by AegisVoice."}

    @app.post("/api/providers/models")
    def provider_models(request: ProviderModelsRequest, http_request: Request):
        if request.consent is not True:
            raise HTTPException(status_code=400, detail="Confirm that you want to check this key against the selected provider's model catalog.")
        origin = http_request.headers.get("origin")
        if origin and urlsplit(origin).netloc.casefold() != http_request.headers.get("host", "").casefold():
            raise HTTPException(status_code=403, detail="Requests must come from this local AegisVoice page.")
        if len(request.api_key) > 500:
            raise HTTPException(status_code=422, detail="The provider key is too long.")
        try:
            return list_provider_models(request.provider, request.api_key.strip(), request.workspace_id.strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/api/external-detect/resemble")
    def resemble_audio_detection(
        file: Annotated[UploadFile, File()],
        api_key: Annotated[str, Form()],
        consent: Annotated[str, Form()],
        intelligence: Annotated[bool, Form()] = False,
        detect_watermark: Annotated[bool, Form()] = False,
        audio_source_tracing: Annotated[bool, Form()] = False,
        infer_from_intelligence: Annotated[bool, Form()] = False,
        visualize: Annotated[bool, Form()] = False,
    ):
        if consent.casefold() != "true":
            raise HTTPException(status_code=400, detail="Confirm that you want to upload this audio to Resemble Detect.")
        filename = _safe_filename(file.filename)
        payload = file.file.read(settings.max_upload_bytes + 1)
        if not payload or len(payload) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"Choose an audio file no larger than {settings.max_upload_bytes // (1024 * 1024)} MB.")
        try:
            safe_suffix = Path(filename).suffix.lower()
            return submit_resemble_audio(
                payload,
                f"recording{safe_suffix}",
                api_key,
                intelligence=intelligence,
                detect_watermark=detect_watermark,
                audio_source_tracing=audio_source_tracing,
                infer_from_intelligence=infer_from_intelligence,
                visualize=visualize,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.get("/api/external-detect/resemble/{detection_id}")
    def resemble_detection_status(detection_id: str, http_request: Request):
        origin = http_request.headers.get("origin")
        if origin and urlsplit(origin).netloc.casefold() != http_request.headers.get("host", "").casefold():
            raise HTTPException(status_code=403, detail="Requests must come from this local AegisVoice page.")
        authorization = http_request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=400, detail="Enter the Resemble API key again to check this result.")
        try:
            return get_resemble_detection(detection_id, authorization[7:])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/api/external-detect/resemble/{detection_id}/intelligence")
    def ask_resemble_intelligence(detection_id: str, body: ResembleQuestionRequest, http_request: Request):
        origin = http_request.headers.get("origin")
        if origin and urlsplit(origin).netloc.casefold() != http_request.headers.get("host", "").casefold():
            raise HTTPException(status_code=403, detail="Requests must come from this local AegisVoice page.")
        if not body.consent:
            raise HTTPException(status_code=400, detail="Confirm that you want to send this question and report context to Resemble Intelligence.")
        authorization = http_request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=400, detail="Enter the Resemble API token to ask about this report.")
        try:
            return submit_resemble_intelligence_question(detection_id, body.query, authorization[7:])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.get("/api/external-detect/resemble/{detection_id}/intelligence/{question_id}")
    def resemble_intelligence_answer(detection_id: str, question_id: str, http_request: Request):
        origin = http_request.headers.get("origin")
        if origin and urlsplit(origin).netloc.casefold() != http_request.headers.get("host", "").casefold():
            raise HTTPException(status_code=403, detail="Requests must come from this local AegisVoice page.")
        authorization = http_request.headers.get("authorization", "")
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=400, detail="Enter the Resemble API token to check this answer.")
        try:
            return get_resemble_intelligence_answer(detection_id, question_id, authorization[7:])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/api/analyze", response_model=AnalysisResponse)
    def analyze_upload(
        file: Annotated[UploadFile, File(description="Common audio formats including WAV, FLAC, OGG, AIFF, MP3, M4A, AAC, WMA, and WebM")],
        session: Session = Depends(get_session),
    ):
        filename = _safe_filename(file.filename)
        if len(filename.encode("utf-8")) > 256:
            raise HTTPException(status_code=400, detail="Filename is too long.")
        if not file.filename:
            raise HTTPException(status_code=400, detail="Choose an audio file first.")
        payload = file.file.read(settings.max_upload_bytes + 1)
        try:
            audio = decode_audio(
                payload,
                filename=filename,
                max_bytes=settings.max_upload_bytes,
                max_duration_seconds=settings.max_duration_seconds,
                target_sample_rate=settings.target_sample_rate,
            )
            measurements = analyze(audio.samples, settings.target_sample_rate)
        except AudioInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ValueError, FloatingPointError, ArithmeticError) as exc:
            raise HTTPException(status_code=422, detail="Audio measurements could not be computed for this file.") from exc

        duration = measurements["time_domain"]["duration_seconds"]
        quality_messages = _quality_messages(audio, measurements)
        quality_score = _input_quality_score(audio, measurements)
        model_bundle = load_model(settings.data_dir / "models" / "dsp_logistic.joblib")
        classifier_block_reasons = _classifier_block_reasons(audio, measurements)
        quality_block = bool(classifier_block_reasons)
        # Minor warnings (for example some clipping)
        # remain visible but no longer suppress a trained model's score.
        score = score_report(measurements, model_bundle) if model_bundle and not quality_block else None
        threshold = float(model_bundle.get("threshold", 0.5)) if model_bundle else None
        score_index = min(11, max(1, int(1 + (score * 10) + 0.5))) if score is not None else None
        decision_label = "INCONCLUSIVE" if not model_bundle or quality_block else ("LIKELY_SYNTHETIC" if score >= threshold else "LIKELY_HUMAN")
        report = {
                "analysis_scope": f"Experimental user-trained {model_bundle.get('algorithm_name', 'DSP classifier')} plus descriptive signal measurements." if model_bundle and not quality_block else "Descriptive signal measurements only; classifier inference was withheld by a quality gate." if model_bundle else "Descriptive signal measurements only; no trained authenticity classifier was run.",
            "source": {
                "format": audio.format,
                "subtype": audio.subtype,
                "original_sample_rate_hz": audio.original_sample_rate,
                "analysis_sample_rate_hz": settings.target_sample_rate,
                "original_channels": audio.original_channels,
                "original_frame_count": audio.original_frames,
                "downmix_rms_ratio": audio.downmix_rms_ratio,
            },
            "quality": {**quality_score, "warnings": quality_messages},
            "dsp": measurements,
            "decision": {
                "label": decision_label,
                "reason": ("The classifier withheld a result because the " + ", ".join(classifier_block_reasons) + "." if model_bundle and quality_block else "Experimental local DSP model result. The 1–11 index is derived from an uncalibrated model score and is not proof of authenticity." if model_bundle else "No trained classifier is configured. DSP measurements alone do not establish whether speech is genuine or synthetic."),
                "classifier_status": "withheld_quality" if model_bundle and quality_block else "experimental_user_trained" if model_bundle else "unavailable",
                "model_algorithm": model_bundle.get("algorithm_name") if model_bundle else None,
                "feature_set": model_bundle.get("feature_set", "basic") if model_bundle else None,
                "feature_count": len(model_bundle.get("feature_names", [])) if model_bundle else 0,
                "score": score,
                "score_index_1_to_11": score_index,
                "threshold": threshold,
                "classifier_block_reasons": classifier_block_reasons,
                "probability": None,
                "score_meaning": "Uncalibrated synthetic-class model output mapped to an index from 1 to 11; not a calibrated probability." if model_bundle else None,
                "model_metrics": model_bundle.get("metrics") if model_bundle else None,
            },
            "limitations": [
                "DSP measurements and model cues are experimental; they do not establish a speaker's identity or prove authenticity.",
                "The detector does not identify the speaker, verify truthfulness, or establish provenance.",
                "Unfamiliar generators, codecs, channels, languages, noise, and non-speech audio can affect measurements.",
            ],
        }
        analysis_id = str(uuid.uuid4())
        record = AnalysisRecord(
            analysis_id=analysis_id,
            created_at=datetime.now(timezone.utc),
            filename=filename,
            file_sha256=hashlib.sha256(payload).hexdigest(),
            file_bytes=len(payload),
            duration_seconds=duration,
            original_sample_rate=audio.original_sample_rate,
            original_channels=audio.original_channels,
            decision=decision_label,
            report=report,
        )
        session.add(record)
        session.commit()
        return AnalysisResponse(**serialize_record(record))

    @app.get("/api/training/status")
    def training_status(session: Session = Depends(get_session)):
        rows = session.query(TrainingExample.label, func.count(TrainingExample.example_id)).group_by(TrainingExample.label).all()
        group_rows = session.query(TrainingExample.label, func.count(func.distinct(TrainingExample.group_id))).group_by(TrainingExample.label).all()
        examples = session.query(TrainingExample).all()
        advanced_examples = [row for row in examples if isinstance(row.features, dict) and "advanced" in row.features]
        advanced_count = len(advanced_examples)
        advanced_label_counts = {label: sum(1 for row in advanced_examples if row.label == label) for label in ("human", "synthetic")}
        advanced_groups = {label: len({row.group_id for row in advanced_examples if row.label == label}) for label in ("human", "synthetic")}
        confirmed_rows = [row for row in examples if row.dataset_name == "confirmed history"]
        confirmed_history_count = len(confirmed_rows)
        confirmed_by_label = {label: sum(1 for row in confirmed_rows if row.label == label) for label in ("human", "synthetic")}
        confirmed_groups = {label: len({row.group_id for row in confirmed_rows if row.label == label}) for label in ("human", "synthetic")}
        confirmed_advanced = [row for row in confirmed_rows if isinstance(row.features, dict) and "advanced" in row.features]
        confirmed_advanced_by_label = {label: sum(1 for row in confirmed_advanced if row.label == label) for label in ("human", "synthetic")}
        confirmed_advanced_groups = {label: len({row.group_id for row in confirmed_advanced if row.label == label}) for label in ("human", "synthetic")}
        bundle = load_model(settings.data_dir / "models" / "dsp_logistic.joblib")
        return {"examples": {label: count for label, count in rows}, "groups": {label: count for label, count in group_rows}, "model": bundle.get("metrics") if bundle else None,
                "model_ready": bool(bundle), "active_algorithm": bundle.get("algorithm_name") if bundle else None,
                "active_feature_set": bundle.get("feature_set", "basic") if bundle else None,
                "advanced_feature_examples": advanced_count, "advanced_feature_examples_by_label": advanced_label_counts,
                "advanced_feature_groups": advanced_groups,
                "confirmed_history_examples": confirmed_history_count,
                "confirmed_history": {"examples": confirmed_by_label, "groups": confirmed_groups,
                                      "advanced_examples": confirmed_advanced_by_label,
                                      "advanced_groups": confirmed_advanced_groups},
                "active_training_source": bundle.get("training_source", "all") if bundle else None,
                "warning": "Only explicitly supplied labels are used; scores are uncalibrated."}

    @app.post("/api/training/samples")
    def add_training_sample(file: Annotated[UploadFile, File()], label: Annotated[str, Form()], group_id: Annotated[str, Form()], dataset_name: Annotated[str, Form()], session: Session = Depends(get_session)):
        if label not in {"human", "synthetic"}:
            raise HTTPException(status_code=400, detail="Label must be human or synthetic.")
        if not group_id.strip() or len(group_id) > 160 or not dataset_name.strip() or len(dataset_name) > 160:
            raise HTTPException(status_code=400, detail="Dataset name and recording group are required.")
        payload = file.file.read(100 * 1024 * 1024 + 1)
        if not payload or len(payload) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Training clips must be under 100 MB each.")
        filename = _safe_filename(file.filename)
        try:
            audio = decode_audio(payload, filename=filename, max_bytes=100 * 1024 * 1024, max_duration_seconds=600)
            crop_size = settings.target_sample_rate * 10
            start = max(0, (len(audio.samples) - crop_size) // 2)
            measured = analyze(audio.samples[start:start + crop_size], settings.target_sample_rate)
        except AudioInputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if (measured["frequency_domain"]["spectral_entropy_normalized_mean"] < 0.38
                and measured["frequency_domain"]["spectral_flatness_mean"] < 0.005):
            raise HTTPException(status_code=422, detail="This clip appears strongly tonal rather than speech (such as DTMF). It was not added to the voice training set.")
        digest = hashlib.sha256(payload).hexdigest()
        exists = session.query(TrainingExample).filter_by(file_sha256=digest).first()
        if exists:
            if exists.label != label:
                raise HTTPException(status_code=409, detail="This same audio file already has a conflicting label in the training set.")
        vectors = stored_feature_vectors(measured)
        if exists:
            if exists.label != label:
                raise HTTPException(status_code=409, detail="This same audio file already has a conflicting label in the training set.")
            if not isinstance(exists.features, dict) or "advanced" not in exists.features:
                exists.features = vectors
                session.commit()
                return {"added": False, "duplicate": True, "upgraded_features": True, "example_id": exists.example_id}
            return {"added": False, "duplicate": True, "example_id": exists.example_id}
        row = TrainingExample(example_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc), filename=filename,
                              file_sha256=digest, label=label, group_id=group_id.strip(), dataset_name=dataset_name.strip(),
                              features=vectors)
        session.add(row); session.commit()
        return {"added": True, "example_id": row.example_id, "label": label, "group_id": row.group_id}

    @app.post("/api/model/train")
    def train_local_model(request: TrainModelRequest, session: Session = Depends(get_session)):
        rows = session.query(TrainingExample).all()
        if request.source == "confirmed_history":
            rows = [row for row in rows if row.dataset_name == "confirmed history"]
        try:
            metrics = train_model([{"label": r.label, "group_id": r.group_id, "features": r.features} for r in rows], settings.data_dir / "models" / "dsp_logistic.joblib", algorithm=request.algorithm, feature_set=request.feature_set, training_source=request.source)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"trained": True, "algorithm": metrics["algorithm_name"], "feature_set": metrics["feature_set"], "training_source": request.source,
                "metrics": metrics["metrics"], "warning": metrics["warning"]}

    @app.post("/api/model/compare")
    def compare_local_models(request: TrainModelRequest, session: Session = Depends(get_session)):
        rows = session.query(TrainingExample).all()
        if request.source == "confirmed_history":
            rows = [row for row in rows if row.dataset_name == "confirmed history"]
        try:
            return compare_models([{"label": r.label, "group_id": r.group_id, "features": r.features} for r in rows], feature_set=request.feature_set)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/model")
    def reset_local_model():
        model_path = settings.data_dir / "models" / "dsp_logistic.joblib"
        existed = model_path.is_file()
        model_path.unlink(missing_ok=True)
        return {"reset": existed, "training_labels_preserved": True}

    @app.post("/api/model/threshold")
    def update_model_threshold(request: ThresholdRequest):
        try:
            return set_model_threshold(settings.data_dir / "models" / "dsp_logistic.joblib", request.threshold)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/history", response_model=HistoryResponse)
    def history(limit: int = 20, offset: int = 0, session: Session = Depends(get_session)):
        if not 1 <= limit <= 100 or offset < 0:
            raise HTTPException(status_code=400, detail="limit must be 1-100 and offset must be non-negative.")
        total = session.scalar(select(func.count()).select_from(AnalysisRecord)) or 0
        records = session.scalars(
            select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc(), AnalysisRecord.analysis_id).offset(offset).limit(limit)
        ).all()
        return HistoryResponse(items=[serialize_record(record, include_report=False) for record in records], total=total)

    @app.get("/api/history/{analysis_id}", response_model=AnalysisResponse)
    def history_detail(analysis_id: str, session: Session = Depends(get_session)):
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis was not found.")
        return AnalysisResponse(**serialize_record(record))

    @app.post("/api/history/{analysis_id}/confirm")
    def confirm_history_label(analysis_id: str, label: Annotated[str, Body(embed=True)], session: Session = Depends(get_session)):
        if label not in {"human", "synthetic", "uncertain"}:
            raise HTTPException(status_code=400, detail="Choose human, synthetic, or uncertain.")
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis was not found.")
        digest = record.file_sha256
        existing = session.query(TrainingExample).filter_by(file_sha256=digest).all()
        if label == "uncertain":
            for row in existing:
                session.delete(row)
            session.commit()
            return {"confirmed": False, "label": "uncertain", "included_in_training": False}
        conflict = next((row for row in existing if row.label != label), None)
        if conflict:
            raise HTTPException(status_code=409, detail="This file already has a conflicting label in an imported dataset. Resolve the conflict there before training.")
        if existing:
            existing[0].dataset_name = "confirmed history"
            existing[0].features = stored_feature_vectors(record.report["dsp"])
            session.commit()
            return {"confirmed": True, "label": label, "included_in_training": True, "duplicate": True}
        row = TrainingExample(example_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc), filename=record.filename,
                              file_sha256=digest, label=label,
                              group_id=Path(record.filename).stem[:160] or digest[:16],
                              dataset_name="confirmed history", features=stored_feature_vectors(record.report["dsp"]))
        session.add(row); session.commit()
        return {"confirmed": True, "label": label, "included_in_training": True, "duplicate": False}

    @app.delete("/api/history/{analysis_id}")
    def delete_history(analysis_id: str, session: Session = Depends(get_session)):
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis was not found.")
        session.delete(record)
        session.commit()
        return {"deleted": True, "analysis_id": analysis_id, "raw_audio_deleted": False}

    @app.get("/api/export/history.json")
    def export_history(session: Session = Depends(get_session)):
        records = session.scalars(select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())).all()
        body = json.dumps([serialize_record(record) for record in records], ensure_ascii=False, separators=(",", ":"))
        return Response(content=body, media_type="application/json", headers={"Content-Disposition": "attachment; filename=aegisvoice-history.json"})

    @app.get("/api/export/history.csv")
    def export_history_csv(session: Session = Depends(get_session)):
        records = session.scalars(select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())).all()
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow([
            "analysis_id", "created_at", "filename", "sha256", "bytes", "duration_seconds",
            "original_sample_rate_hz", "channels", "decision", "synthetic_evidence_index_1_to_11",
            "synthetic_raw_output_uncalibrated", "decision_threshold",
            "input_quality_score_0_100", "quality_warnings", "rms_dbfs", "peak_dbfs",
            "spectral_centroid_hz", "spectral_flatness", "spectral_entropy", "spectral_rolloff_85_hz",
            "zero_crossing_rate", "spectral_flux",
        ])
        for record in records:
            report = record.report or {}
            dsp = report.get("dsp", {})
            time = dsp.get("time_domain", {})
            frequency = dsp.get("frequency_domain", {})
            decision = report.get("decision", {})
            quality = report.get("quality", {})
            writer.writerow([
                record.analysis_id, record.created_at.isoformat(), record.filename, record.file_sha256,
                record.file_bytes, record.duration_seconds, record.original_sample_rate, record.original_channels,
                record.decision, decision.get("score_index_1_to_11"), decision.get("score"), decision.get("threshold"), quality.get("score_0_to_100"),
                "; ".join(quality.get("warnings", [])), time.get("rms_dbfs"), time.get("peak_dbfs"),
                frequency.get("spectral_centroid_mean_hz"), frequency.get("spectral_flatness_mean"),
                frequency.get("spectral_entropy_normalized_mean"), frequency.get("spectral_rolloff_85_mean_hz"),
                frequency.get("zero_crossing_rate_mean"), frequency.get("spectral_flux_mean"),
            ])
        return Response(content=output.getvalue(), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=aegisvoice-history.csv"})

    @app.get("/api/export/model.json")
    def export_model_state():
        bundle = load_model(settings.data_dir / "models" / "dsp_logistic.joblib")
        if not bundle:
            raise HTTPException(status_code=404, detail="No trained local model is available to export.")
        classifier = bundle["classifier"]
        steps = getattr(classifier, "named_steps", {})
        estimator = list(steps.values())[-1] if steps else classifier
        state = {
            "format": "AegisVoice model metadata and coefficients",
            "version": bundle["version"],
            "algorithm": bundle.get("algorithm"),
            "algorithm_name": bundle.get("algorithm_name"),
            "training_source": bundle.get("training_source", "all imported and confirmed labels"),
            "trained_at_utc": bundle["trained_at_utc"],
            "feature_names": bundle["feature_names"],
            "feature_set": bundle.get("feature_set", "basic"),
            "class_order": [int(value) for value in estimator.classes_],
            "class_labels": {"0": "human", "1": "synthetic"},
            "decision_threshold": bundle.get("threshold", 0.5),
            "decision_threshold_range": [0.2, 0.8],
            "synthetic_index_scale": "1 to 11; mapped from raw uncalibrated synthetic-class model output",
            "score_is_calibrated_probability": bundle["score_is_calibrated_probability"],
            "estimator_parameters": estimator.get_params(deep=False),
            "evaluation_metrics": bundle["metrics"],
            "warning": bundle["warning"],
        }
        if "standardscaler" in steps:
            scaler = steps["standardscaler"]
            state["scaler_mean"] = scaler.mean_.tolist()
            state["scaler_scale"] = scaler.scale_.tolist()
        if hasattr(estimator, "coef_"):
            state["coefficients"] = estimator.coef_.tolist()
            state["intercept"] = estimator.intercept_.tolist()
        if hasattr(estimator, "feature_importances_"):
            state["feature_importances"] = estimator.feature_importances_.tolist()
        state["warning"] += " This JSON is an inspection export, not a loadable or signed model checkpoint."
        return Response(content=json.dumps(state, ensure_ascii=False, separators=(",", ":")),
                        media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=aegisvoice-model-state.json"})

    return app


app = create_app()

