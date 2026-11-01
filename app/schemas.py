"""Typed API response contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    ok: bool
    application: str
    version: str


class StatusResponse(BaseModel):
    application: str
    analysis_mode: str
    classifier_available: bool
    verdict_policy: str
    raw_audio_retention: str
    accepted_formats: list[str]
    max_upload_mb: int
    max_duration_seconds: int
    model_algorithm: str | None = None
    model_feature_set: str | None = None
    model_feature_count: int = 0
    model_threshold: float | None = None


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis_id: str
    created_at: str
    filename: str
    file_sha256: str
    file_bytes: int
    duration_seconds: float
    original_sample_rate: int
    original_channels: int
    decision: str
    report: dict


class HistoryResponse(BaseModel):
    items: list[dict]
    total: int

