"""Bounded audio decoding and deterministic canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SUPPORTED_FORMATS = {"WAV", "FLAC", "OGG", "AIFF", "MP3", "MPEG", "MPEG4", "AAC", "WMA", "WEBM", "MATROSKA"}


class AudioInputError(ValueError):
    """An input file is unsupported, malformed, or outside configured limits."""


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray
    original_sample_rate: int
    original_channels: int
    original_frames: int
    format: str
    subtype: str
    downmix_rms_ratio: float


def decode_audio(
    payload: bytes,
    *,
    filename: str,
    max_bytes: int,
    max_duration_seconds: int,
    target_sample_rate: int = 16_000,
) -> DecodedAudio:
    """Decode an accepted audio container without writing the recording to disk.

    The uploaded byte count is checked before invoking libsndfile. Container
    metadata is checked before sample allocation, and decoding is bounded to one
    frame beyond the configured duration so overlong streams are rejected.
    """
    if not payload:
        raise AudioInputError("The uploaded file is empty.")
    if len(payload) > max_bytes:
        raise AudioInputError(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in {"wav", "flac", "ogg", "oga", "aiff", "aif", "mp3", "m4a", "mp4", "aac", "wma", "webm", "opus", "mka"}:
        raise AudioInputError("Supported formats are WAV, FLAC, OGG/Opus, AIFF, MP3, M4A/MP4, AAC, WMA, and WebM.")

    try:
        stream = BytesIO(payload)
        try:
            info = sf.info(stream)
        except Exception:
            return _decode_with_pyav(payload, max_bytes=max_bytes, max_duration_seconds=max_duration_seconds, target_sample_rate=target_sample_rate)
        if info.format not in SUPPORTED_FORMATS:
            raise AudioInputError("The file container is not a supported audio format.")
        if info.samplerate < 4_000 or info.samplerate > 192_000:
            raise AudioInputError("Sample rate must be between 4 kHz and 192 kHz.")
        if info.channels < 1 or info.channels > 2:
            raise AudioInputError("Audio must contain one or two channels.")
        if info.frames <= 0:
            raise AudioInputError("The file contains no audio frames.")
        if info.frames / info.samplerate > max_duration_seconds:
            raise AudioInputError(f"Audio exceeds the {max_duration_seconds} second duration limit.")

        stream.seek(0)
        raw, sample_rate = sf.read(
            stream,
            dtype="float64",
            always_2d=True,
            frames=max_duration_seconds * info.samplerate + 1,
        )
    except AudioInputError:
        raise
    except (RuntimeError, OSError, ValueError) as exc:
        try:
            return _decode_with_pyav(payload, max_bytes=max_bytes, max_duration_seconds=max_duration_seconds, target_sample_rate=target_sample_rate)
        except AudioInputError:
            raise
        except Exception as fallback_exc:
            raise AudioInputError("The audio file could not be decoded. Install the full audio dependencies and check that it is valid.") from fallback_exc

    if raw.shape[0] == 0 or raw.shape[1] != info.channels:
        raise AudioInputError("The decoder returned an invalid audio shape.")
    if raw.shape[0] / sample_rate > max_duration_seconds:
        raise AudioInputError(f"Audio exceeds the {max_duration_seconds} second duration limit.")
    if not np.isfinite(raw).all():
        raise AudioInputError("Audio contains non-finite sample values.")

    raw = np.clip(raw, -1.0, 1.0)
    channel_rms = np.sqrt(np.mean(np.square(raw), axis=0))
    reference_rms = float(np.sqrt(np.mean(np.square(channel_rms))))
    mono = np.mean(raw, axis=1, dtype=np.float64)
    mono_rms = float(np.sqrt(np.mean(np.square(mono))))
    ratio = mono_rms / reference_rms if reference_rms > 0 else 1.0

    # Resample by a rational polyphase filter. The original bytes are not retained.
    factor = gcd(int(sample_rate), int(target_sample_rate))
    up, down = target_sample_rate // factor, sample_rate // factor
    canonical = resample_poly(mono, up, down).astype(np.float32, copy=False)
    if canonical.size == 0 or not np.isfinite(canonical).all():
        raise AudioInputError("Audio preprocessing failed to produce a finite signal.")

    return DecodedAudio(
        samples=canonical,
        original_sample_rate=int(sample_rate),
        original_channels=int(info.channels),
        original_frames=int(raw.shape[0]),
        format=str(info.format),
        subtype=str(info.subtype),
        downmix_rms_ratio=float(ratio),
    )


def _decode_with_pyav(payload: bytes, *, max_bytes: int, max_duration_seconds: int, target_sample_rate: int) -> DecodedAudio:
    """Decode common compressed formats with PyAV's bundled FFmpeg libraries."""
    if len(payload) > max_bytes:
        raise AudioInputError(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    try:
        import av
        import numpy as np
        from scipy.signal import resample_poly
        from io import BytesIO
        from math import gcd
        container = av.open(BytesIO(payload), mode="r")
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None or not stream.codec_context.sample_rate:
            raise AudioInputError("No decodable audio stream was found.")
        original_rate = int(stream.codec_context.sample_rate)
        original_channels = int(stream.codec_context.channels or 1)
        if not 4_000 <= original_rate <= 192_000 or not 1 <= original_channels <= 2:
            raise AudioInputError("Audio must be 4–192 kHz and contain one or two channels.")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_sample_rate)
        chunks = []
        frame_count = 0
        for frame in container.decode(stream):
            frame_count += frame.samples
            if frame_count > max_duration_seconds * original_rate:
                raise AudioInputError(f"Audio exceeds the {max_duration_seconds} second duration limit.")
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32, copy=False))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32, copy=False))
        container.close()
        if not chunks or frame_count == 0:
            raise AudioInputError("The audio stream contains no frames.")
        canonical = np.concatenate(chunks)
        if not np.isfinite(canonical).all():
            raise AudioInputError("Audio contains non-finite sample values.")
        return DecodedAudio(canonical, original_rate, original_channels, frame_count, "FFMPEG", "decoded", 1.0)
    except AudioInputError:
        raise
    except Exception as exc:
        raise AudioInputError("PyAV/FFmpeg could not decode this file. Check its codec and integrity.") from exc

