"""Small, validated configuration surface for the local application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    max_upload_bytes: int = 50 * 1024 * 1024
    max_duration_seconds: int = 120
    target_sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    fft_size: int = 512

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("AEGISVOICE_DATA_DIR", "data")).expanduser()
        max_upload_mb = int(os.environ.get("AEGISVOICE_MAX_UPLOAD_MB", "50"))
        max_duration = int(os.environ.get("AEGISVOICE_MAX_DURATION_SECONDS", "120"))
        if not 1 <= max_upload_mb <= 100:
            raise ValueError("AEGISVOICE_MAX_UPLOAD_MB must be between 1 and 100")
        if not 1 <= max_duration <= 600:
            raise ValueError("AEGISVOICE_MAX_DURATION_SECONDS must be between 1 and 600")
        return cls(
            data_dir=data_dir,
            max_upload_bytes=max_upload_mb * 1024 * 1024,
            max_duration_seconds=max_duration,
        )

