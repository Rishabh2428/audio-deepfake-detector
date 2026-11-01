"""Deterministic descriptive DSP measurements; no authenticity classifier."""

from __future__ import annotations

import math

import numpy as np
from scipy.fft import dct


def _dbfs(amplitude: float) -> float | None:
    return 20.0 * math.log10(amplitude) if amplitude > 0 else None


def _hz_to_mel(value: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + value / 700.0)


def _triangular_bank(frequencies: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bank = np.zeros((len(edges) - 2, frequencies.size), dtype=np.float64)
    for index in range(bank.shape[0]):
        left, center, right = edges[index:index + 3]
        rising = (frequencies - left) / max(center - left, np.finfo(float).tiny)
        falling = (right - frequencies) / max(right - center, np.finfo(float).tiny)
        bank[index] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


def _cepstral_summary(power: np.ndarray, frequencies: np.ndarray, sample_rate: int) -> dict:
    """Summarize mel, linear, and log-frequency filter-bank cepstra.

    The log-frequency representation is explicitly CQT-style: it uses
    geometrically spaced triangular filters over the STFT bins, not a full
    variable-window constant-Q transform.
    """
    nyquist = sample_rate / 2.0
    mel_edges = np.linspace(_hz_to_mel(np.asarray([0.0]))[0], _hz_to_mel(np.asarray([nyquist]))[0], 28)
    mel_hz = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)
    linear_edges = np.linspace(0.0, nyquist, 28)
    low_log = min(55.0, nyquist / 4.0)
    log_edges = np.geomspace(max(1.0, low_log), nyquist, 50)
    specifications = (
        ("mfcc", _triangular_bank(frequencies, mel_hz)),
        ("lfcc", _triangular_bank(frequencies, linear_edges)),
        ("cqcc_style", _triangular_bank(frequencies, log_edges)),
    )
    result = {}
    for name, bank in specifications:
        band_energy = power @ bank.T
        log_energy = np.log(np.maximum(band_energy, 1e-12))
        coefficients = dct(log_energy, type=2, axis=1, norm="ortho")[:, :13]
        result[name] = {
            "mean_coefficients": [float(value) for value in np.mean(coefficients, axis=0)],
            "std_coefficients": [float(value) for value in np.std(coefficients, axis=0)],
            "bands": int(bank.shape[0]),
            "method": "DCT-II of log filter-bank STFT power",
        }
    result["cqcc_style"]["method"] = "CQT-style geometrically spaced STFT filter bank + DCT-II; approximate, not a full CQT"
    return result


def _phase_summary(spectrum: np.ndarray, magnitude: np.ndarray, fft_size: int, hop_length: int) -> dict:
    phase = np.angle(spectrum)
    significant = magnitude >= (np.max(magnitude, axis=1, keepdims=True) * 0.01)
    if spectrum.shape[0] > 1:
        expected = 2.0 * np.pi * hop_length * np.arange(spectrum.shape[1], dtype=np.float64) / fft_size
        residual = (np.diff(phase, axis=0) - expected[None, :] + np.pi) % (2.0 * np.pi) - np.pi
        adjacent = significant[1:, :] & significant[:-1, :]
        residual_values = np.abs(residual[adjacent])
    else:
        residual_values = np.asarray([], dtype=np.float64)
    group_delay = np.diff(phase, axis=1)
    group_mask = significant[:, 1:] & significant[:, :-1]
    delay_values = group_delay[group_mask]
    if residual_values.size:
        phase_variation = float(np.mean(residual_values))
        jump_ratio = float(np.mean(residual_values >= 2.8))
    else:
        phase_variation, jump_ratio = None, None
    if delay_values.size:
        median = float(np.median(delay_values))
        mad = float(np.median(np.abs(delay_values - median)))
        delay_outliers = float(np.mean(np.abs(delay_values - median) > 3.5 * mad)) if mad > 1e-12 else 0.0
    else:
        delay_outliers = None
    return {
        "mean_residual_phase_variation_radians": phase_variation,
        "residual_phase_jump_ratio": jump_ratio,
        "group_delay_outlier_ratio": delay_outliers,
        "method": "STFT phase advance removed before frame-to-frame residual analysis; descriptive, recording-sensitive evidence",
    }


def analyze(samples: np.ndarray, sample_rate: int = 16_000) -> dict:
    """Compute reproducible time/frequency measurements for one mono waveform.

    Defaults follow the supplied project brief's engineering setup: 25 ms Hann
    frames, 10 ms hop, and a 512-point real FFT at 16 kHz. Measurements describe
    the uploaded signal and are not evidence of human or synthetic provenance.
    """
    raw = np.asarray(samples, dtype=np.float64).reshape(-1)
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("samples must be a non-empty finite one-dimensional signal")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")

    dc = float(np.mean(raw))
    rms = float(np.sqrt(np.mean(raw * raw)))
    peak = float(np.max(np.abs(raw)))
    clipping_fraction = float(np.mean(np.abs(raw) >= 0.999))
    x = raw - dc

    frame_length = round(sample_rate * 0.025)
    hop_length = round(sample_rate * 0.010)
    fft_size = max(512, 1 << (frame_length - 1).bit_length())
    if x.size < frame_length:
        x = np.pad(x, (0, frame_length - x.size))
    starts = np.arange(0, x.size - frame_length + 1, hop_length, dtype=np.int64)
    frames = np.stack([x[start : start + frame_length] for start in starts])
    window = np.hanning(frame_length).astype(np.float64)
    window_power = float(np.sum(window * window))
    spectrum = np.fft.rfft(frames * window[None, :], n=fft_size, axis=1)
    power = np.square(np.abs(spectrum))
    if fft_size % 2 == 0:
        power[:, 1:-1] *= 2.0
    else:
        power[:, 1:] *= 2.0
    psd = power / (sample_rate * window_power)
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    bin_hz = sample_rate / fft_size

    frame_energy = np.mean(frames * frames, axis=1)
    frame_rms = np.sqrt(frame_energy)
    magnitude = np.abs(spectrum)
    power_sum = np.sum(power, axis=1)
    safe_power = np.maximum(power, np.finfo(np.float64).tiny)
    centroid = np.divide(
        np.sum(power * frequencies[None, :], axis=1),
        power_sum,
        out=np.zeros_like(power_sum),
        where=power_sum > 0,
    )
    flatness = np.divide(
        np.exp(np.mean(np.log(safe_power), axis=1)),
        np.maximum(np.mean(power, axis=1), np.finfo(np.float64).tiny),
        out=np.zeros_like(power_sum),
        where=power_sum > 0,
    )
    normalized_power = np.divide(
        power,
        power_sum[:, None],
        out=np.zeros_like(power),
        where=power_sum[:, None] > 0,
    )
    entropy = -np.sum(
        np.where(normalized_power > 0, normalized_power * np.log(np.maximum(normalized_power, 1e-300)), 0.0),
        axis=1,
    ) / math.log(power.shape[1])
    cumulative = np.cumsum(power, axis=1)
    rolloff_index = np.argmax(cumulative >= 0.85 * power_sum[:, None], axis=1)
    rolloff = frequencies[rolloff_index]

    signs = np.signbit(frames)
    zero_crossing_rate = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    normalized_mag = magnitude / np.maximum(
        np.linalg.norm(magnitude, axis=1, keepdims=True), np.finfo(np.float64).tiny
    )
    flux = np.zeros(frames.shape[0], dtype=np.float64)
    if frames.shape[0] > 1:
        flux[1:] = np.sqrt(np.sum(np.square(normalized_mag[1:] - normalized_mag[:-1]), axis=1))

    # Summarize slow amplitude-envelope modulation as descriptive temporal
    # evidence. It is not a speech detector and is intentionally not a model
    # input until independently evaluated with labeled corpora.
    modulation_rate_hz = sample_rate / hop_length
    modulation_envelope = frame_rms - float(np.mean(frame_rms))
    modulation_power = np.square(np.abs(np.fft.rfft(modulation_envelope)))
    modulation_frequencies = np.fft.rfftfreq(frame_rms.size, d=1.0 / modulation_rate_hz)
    modulation_band = (modulation_frequencies >= 0.5) & (modulation_frequencies <= 20.0)
    speech_modulation_band = (modulation_frequencies >= 2.0) & (modulation_frequencies <= 8.0)
    available_modulation_power = float(np.sum(modulation_power[modulation_band]))
    speech_band_power = float(np.sum(modulation_power[speech_modulation_band]))
    if frame_rms.size >= 2 and np.any(speech_modulation_band):
        speech_band_indices = np.flatnonzero(speech_modulation_band)
        modulation_peak_hz = float(modulation_frequencies[speech_band_indices[np.argmax(modulation_power[speech_modulation_band])]]) if speech_band_power > 0 else None
        modulation_fraction = speech_band_power / available_modulation_power if available_modulation_power > 0 else 0.0
    else:
        modulation_peak_hz, modulation_fraction = None, None

    # Additional descriptive measurements are always reported. The learning
    # module selects a basic or extended feature schema independently.
    min_lag = max(1, round(sample_rate / 400.0))
    max_lag = min(round(sample_rate / 70.0), frame_length - 1)
    pitch_hz: list[float] = []
    harmonicity_values: list[float] = []
    hnr_values: list[float] = []
    voiced_tracks: list[tuple[int, float, float]] = []
    pitch_stride = 4
    sampled_frames = frames[::pitch_stride]
    for frame_index, (frame, frame_level) in enumerate(zip(sampled_frames, frame_rms[::pitch_stride])):
        centered = frame - float(np.mean(frame))
        energy = float(np.dot(centered, centered))
        if energy <= 1e-12 or frame_level <= 1e-5:
            continue
        best_correlation = -1.0
        best_lag = 0
        for lag in range(min_lag, max_lag + 1):
            correlation = float(np.dot(centered[:-lag], centered[lag:]) / energy)
            if correlation > best_correlation:
                best_correlation, best_lag = correlation, lag
        if best_lag and best_correlation > 0.32:
            normalized_correlation = min(0.999, max(0.0, best_correlation))
            pitch_hz.append(sample_rate / best_lag)
            harmonicity_values.append(normalized_correlation)
            hnr_values.append(10.0 * math.log10(normalized_correlation / (1.0 - normalized_correlation + 1e-6)))
            voiced_tracks.append((frame_index, float(best_lag), float(frame_level)))

    pitch_variation = [
        abs(pitch_hz[i] - pitch_hz[i - 1]) / ((pitch_hz[i] + pitch_hz[i - 1]) / 2.0) * 100.0
        for i in range(1, len(pitch_hz))
        if pitch_hz[i] > 0 and pitch_hz[i - 1] > 0
    ]
    frame_rms_values = np.asarray(frame_rms, dtype=np.float64)
    frame_level_variation = [
        abs(frame_rms_values[i] - frame_rms_values[i - 1])
        / ((frame_rms_values[i] + frame_rms_values[i - 1]) / 2.0) * 100.0
        for i in range(1, len(frame_rms_values))
        if frame_rms_values[i] > 1e-6 and frame_rms_values[i - 1] > 1e-6
    ]
    adjacent_voiced = [
        (voiced_tracks[index - 1], voiced_tracks[index])
        for index in range(1, len(voiced_tracks))
        if voiced_tracks[index][0] == voiced_tracks[index - 1][0] + 1
    ]
    jitter_proxy = (
        float(np.mean([abs(right[1] - left[1]) for left, right in adjacent_voiced]) / np.mean([item[1] for item in voiced_tracks]) * 100.0)
        if len(adjacent_voiced) >= 3 and voiced_tracks else None
    )
    shimmer_proxy = (
        float(np.mean([abs(right[2] - left[2]) for left, right in adjacent_voiced]) / np.mean([item[2] for item in voiced_tracks]) * 100.0)
        if len(adjacent_voiced) >= 3 and voiced_tracks and np.mean([item[2] for item in voiced_tracks]) > 1e-8 else None
    )
    spread = np.sqrt(
        np.divide(
            np.sum(power * np.square(frequencies[None, :] - centroid[:, None]), axis=1),
            power_sum,
            out=np.zeros_like(power_sum),
            where=power_sum > 0,
        )
    )
    positive_magnitude = np.maximum(magnitude[:, 1:], 1e-12)
    log_frequency = np.log2(frequencies[1:])
    log_frequency_centered = log_frequency - np.mean(log_frequency)
    log_magnitude_db = 20.0 * np.log10(positive_magnitude)
    slope = np.sum(log_magnitude_db * log_frequency_centered[None, :], axis=1) / max(
        float(np.sum(np.square(log_frequency_centered))), np.finfo(np.float64).tiny
    )
    spectral_crest = np.max(magnitude, axis=1) / np.maximum(np.mean(magnitude, axis=1), np.finfo(np.float64).tiny)

    max_frame_rms = float(np.max(frame_rms))
    low_energy_threshold = max_frame_rms * (10.0 ** (-40.0 / 20.0))
    low_energy_fraction = float(np.mean(frame_rms <= low_energy_threshold)) if max_frame_rms > 0 else 1.0

    mean_psd = np.mean(psd, axis=0)
    psd_db = 10.0 * np.log10(np.maximum(mean_psd, 1e-18))
    display_limit = 600
    stride = max(1, math.ceil(frames.shape[0] / display_limit))
    frame_times = (starts + frame_length / 2.0) / sample_rate
    # A compact, display-only log-frequency spectrogram. Quantized values keep
    # saved reports small; the model never consumes this visualization.
    spectrogram_frequencies = np.geomspace(max(bin_hz, 20.0), sample_rate / 2.0, 48)
    spectrogram_source_frequencies = frequencies[1:]
    spectrogram_log_source = np.log(spectrogram_source_frequencies)
    spectrogram_log_targets = np.log(spectrogram_frequencies)
    sampled_power_db = 10.0 * np.log10(np.maximum(power[::stride, 1:], 1e-18))
    spectrogram_db = np.stack([
        np.interp(spectrogram_log_targets, spectrogram_log_source, row)
        for row in sampled_power_db
    ])
    spectrogram_peak_db = float(np.max(spectrogram_db))
    spectrogram_u8 = np.rint((np.clip(spectrogram_db - spectrogram_peak_db, -80.0, 0.0) + 80.0) * (255.0 / 80.0)).astype(np.uint8)
    frame_records = [
        {
            "time_seconds": float(frame_times[i]),
            "rms_dbfs": _dbfs(float(frame_rms[i])),
            "spectral_centroid_hz": float(centroid[i]),
            "spectral_flatness": float(flatness[i]) if power_sum[i] > 0 else 0.0,
            "spectral_entropy_normalized": float(entropy[i]) if power_sum[i] > 0 else 0.0,
            "spectral_rolloff_85_hz": float(rolloff[i]),
            "zero_crossing_rate": float(zero_crossing_rate[i]),
            "spectral_flux": float(flux[i]),
        }
        for i in range(0, frames.shape[0], stride)
    ]

    waveform_stride = max(1, math.ceil(raw.size / 1_000))
    waveform = [float(value) for value in raw[::waveform_stride][:1_000]]
    total_spectral_power = float(np.sum(mean_psd) * bin_hz)
    high_band_start = 4_000.0
    high_band_power = float(np.sum(mean_psd[frequencies >= high_band_start]) * bin_hz)
    low_band_power = float(np.sum(mean_psd[frequencies < 500.0]) * bin_hz)
    mid_band_power = float(np.sum(mean_psd[(frequencies >= 500.0) & (frequencies < 2_000.0)]) * bin_hz)
    additional_spectral_power = float(np.sum(mean_psd) * bin_hz)
    band_denominator = additional_spectral_power if additional_spectral_power > 0 else 1.0

    return {
        "configuration": {
            "sample_rate_hz": int(sample_rate),
            "frame_length_samples": int(frame_length),
            "frame_duration_ms": 25.0,
            "hop_length_samples": int(hop_length),
            "hop_duration_ms": 10.0,
            "fft_size": int(fft_size),
            "window": "Hann",
            "resampling": "scipy.signal.resample_poly; rational polyphase filter",
        },
        "time_domain": {
            "duration_seconds": float(raw.size / sample_rate),
            "sample_count": int(raw.size),
            "rms_amplitude": rms,
            "rms_dbfs": _dbfs(rms),
            "peak_amplitude": peak,
            "peak_dbfs": _dbfs(peak),
            "dc_offset_mean_amplitude": dc,
            "clipping_sample_fraction_at_0_999": clipping_fraction,
            "low_energy_frame_fraction_relative_minus_40_db": low_energy_fraction,
        },
        "frequency_domain": {
            "spectral_centroid_mean_hz": float(np.mean(centroid)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "spectral_entropy_normalized_mean": float(np.mean(entropy)),
            "spectral_rolloff_85_mean_hz": float(np.mean(rolloff)),
            "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
            "spectral_flux_mean": float(np.mean(flux)),
            "power_below_4khz_fraction": float(max(0.0, min(1.0, 1.0 - high_band_power / total_spectral_power))) if total_spectral_power > 0 else 0.0,
        },
        "prosodic": {
            "mean_f0_hz": float(np.mean(pitch_hz)) if pitch_hz else None,
            "f0_std_hz": float(np.std(pitch_hz)) if pitch_hz else None,
            "pitch_variation_proxy_pct": float(np.mean(pitch_variation)) if pitch_variation else None,
            "jitter_local_proxy_percent": jitter_proxy,
            "shimmer_local_proxy_percent": shimmer_proxy,
            "frame_level_variation_proxy_pct": float(np.mean(frame_level_variation)) if frame_level_variation else None,
            "autocorrelation_harmonicity_proxy_mean": float(np.mean(harmonicity_values)) if harmonicity_values else None,
            "autocorrelation_hnr_proxy_mean_db": float(np.mean(hnr_values)) if hnr_values else None,
            "voiced_frame_fraction": float(len(pitch_hz) / sampled_frames.shape[0]),
            "measurement_note": "Jitter and shimmer are short-frame proxies from the autocorrelation pitch track and voiced-frame RMS; they are not clinical voice measurements.",
        },
        "extended_spectral": {
            "spectral_flux_std": float(np.std(flux)),
            "spectral_spread_mean_hz": float(np.mean(spread)),
            "spectral_slope_mean_db_per_octave": float(np.mean(slope)),
            "spectral_crest_mean": float(np.mean(spectral_crest)),
            "frame_rms_mean_amplitude": float(np.mean(frame_rms_values)),
            "frame_rms_std_amplitude": float(np.std(frame_rms_values)),
            "band_power_fractions": {
                "low_below_500_hz": float(low_band_power / band_denominator),
                "mid_500_to_2000_hz": float(mid_band_power / band_denominator),
                "high_at_or_above_2000_hz": float(max(0.0, (additional_spectral_power - low_band_power - mid_band_power) / band_denominator)),
            },
            "feature_scope_note": "Measurements are descriptive; the active model's selected feature schema is reported separately.",
        },
        "temporal_modulation": {
            "amplitude_modulation_index": float(np.std(frame_rms) / max(float(np.mean(frame_rms)), 1e-12)),
            "peak_modulation_frequency_2_to_8_hz": modulation_peak_hz,
            "power_fraction_2_to_8_hz_of_0_5_to_20_hz": float(max(0.0, min(1.0, modulation_fraction))) if modulation_fraction is not None else None,
            "frequency_resolution_hz": float(modulation_rate_hz / frame_rms.size),
            "method": "FFT of the centered 25 ms frame-RMS envelope sampled every 10 ms; temporal description only, not used by the saved model, sensitive to speech content and recording conditions.",
        },
        "cepstral": _cepstral_summary(power, frequencies, sample_rate),
        "phase": _phase_summary(spectrum, magnitude, fft_size, hop_length),
        "waveform": {"time_step_seconds": float(waveform_stride / sample_rate), "samples": waveform},
        "power_spectral_density": {
            "frequency_hz": [float(v) for v in frequencies],
            "power_db_per_hz": [float(v) for v in psd_db],
            "frequency_resolution_hz": float(bin_hz),
            "units": "relative full-scale amplitude squared per Hz",
        },
        "spectrogram": {
            "time_seconds": [float(value) for value in frame_times[::stride]],
            "frequency_hz": [float(value) for value in spectrogram_frequencies],
            "values_u8": spectrogram_u8.tolist(),
            "dynamic_range_db": 80,
            "frequency_scale": "logarithmic",
            "note": "Display-only STFT power relative to this recording's peak; not used by the classifier and not authenticity evidence by itself.",
        },
        "frame_measurements": frame_records,
    }

