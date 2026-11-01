"""Dependency-free WAV path for offline setup and numerical smoke checks."""

from __future__ import annotations

import math


def _fft(values: list[float], size: int) -> list[complex]:
    data = [complex(value, 0.0) for value in values] + [0j] * (size - len(values))
    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            data[i], data[j] = data[j], data[i]
    length = 2
    while length <= size:
        angle = -2.0 * math.pi / length
        step = complex(math.cos(angle), math.sin(angle))
        half = length // 2
        for start in range(0, size, length):
            weight = 1.0 + 0j
            for offset in range(half):
                even = data[start + offset]
                odd = data[start + offset + half] * weight
                data[start + offset] = even + odd
                data[start + offset + half] = even - odd
                weight *= step
        length *= 2
    return data[: size // 2 + 1]


def resample_windowed_sinc(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    """Band-limited windowed-sinc resampling for the standard-library fallback."""
    if source_rate == target_rate:
        return list(samples)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    output_count = round(len(samples) * target_rate / source_rate)
    if output_count <= 0:
        return []
    cutoff = min(1.0, target_rate / source_rate)
    radius = 24
    output: list[float] = []
    for output_index in range(output_count):
        position = output_index * source_rate / target_rate
        center = math.floor(position)
        numerator = 0.0
        denominator = 0.0
        for source_index in range(center - radius + 1, center + radius + 1):
            distance = position - source_index
            if abs(distance) >= radius:
                continue
            scaled = cutoff * distance
            sinc = 1.0 if scaled == 0 else math.sin(math.pi * scaled) / (math.pi * scaled)
            window = 0.5 + 0.5 * math.cos(math.pi * distance / radius)
            coefficient = cutoff * sinc * window
            if 0 <= source_index < len(samples):
                numerator += samples[source_index] * coefficient
            denominator += coefficient
        output.append(numerator / denominator if abs(denominator) > 1e-12 else 0.0)
    return output


def analyze(samples: list[float], sample_rate: int = 16_000) -> dict:
    """Compute the same measurements as the NumPy/SciPy application path."""
    if not samples or sample_rate < 1 or any(not math.isfinite(x) for x in samples):
        raise ValueError("samples must be non-empty, finite, and have a positive sample rate")
    raw = samples
    dc = sum(raw) / len(raw)
    rms = math.sqrt(sum(value * value for value in raw) / len(raw))
    peak = max(abs(value) for value in raw)
    clipping = sum(abs(value) >= 0.999 for value in raw) / len(raw)
    signal_centered = [value - dc for value in raw]
    frame_length = round(sample_rate * 0.025)
    hop = round(sample_rate * 0.010)
    fft_size = max(512, 1 << (frame_length - 1).bit_length())
    signal = signal_centered if len(signal_centered) >= frame_length else signal_centered + [0.0] * (frame_length - len(signal_centered))
    starts = list(range(0, len(signal) - frame_length + 1, hop))
    window = [0.5 - 0.5 * math.cos(2.0 * math.pi * index / (frame_length - 1)) for index in range(frame_length)]
    window_power = sum(value * value for value in window)
    frequencies = [index * sample_rate / fft_size for index in range(fft_size // 2 + 1)]
    psd_accumulator = [0.0] * len(frequencies)
    frame_records = []
    frame_rms_values = []
    frame_magnitudes = []
    frame_power_sums = []
    frame_centroids = []
    frame_flatnesses = []
    frame_entropies = []
    frame_rolloffs = []
    frame_zcr = []
    frame_flux = []

    for start in starts:
        frame = signal[start : start + frame_length]
        weighted = [frame[index] * window[index] for index in range(frame_length)]
        transform = _fft(weighted, fft_size)
        power = [abs(value) ** 2 for value in transform]
        for index in range(1, len(power) - 1):
            power[index] *= 2.0
        psd = [value / (sample_rate * window_power) for value in power]
        for index, value in enumerate(psd):
            psd_accumulator[index] += value
        total = sum(power)
        rms = math.sqrt(sum(value * value for value in frame) / len(frame))
        if total > 0:
            centroid = sum(p * f for p, f in zip(power, frequencies)) / total
            mean_power = total / len(power)
            flatness = math.exp(sum(math.log(max(p, 1e-300)) for p in power) / len(power)) / mean_power
            entropy = -sum((p / total) * math.log(max(p / total, 1e-300)) for p in power if p > 0) / math.log(len(power))
            cumulative = 0.0
            rolloff = frequencies[-1]
            for frequency, value in zip(frequencies, power):
                cumulative += value
                if cumulative >= 0.85 * total:
                    rolloff = frequency
                    break
        else:
            centroid = flatness = entropy = rolloff = 0.0
        zcr = sum((frame[i] < 0) != (frame[i - 1] < 0) for i in range(1, len(frame))) / max(1, len(frame) - 1)
        magnitude = [abs(value) for value in transform]
        norm = math.sqrt(sum(value * value for value in magnitude))
        unit_magnitude = [value / norm for value in magnitude] if norm else magnitude
        flux = math.sqrt(sum((a - b) ** 2 for a, b in zip(unit_magnitude, frame_magnitudes[-1]))) if frame_magnitudes else 0.0
        frame_rms_values.append(rms)
        frame_magnitudes.append(unit_magnitude)
        frame_power_sums.append(total)
        frame_centroids.append(centroid)
        frame_flatnesses.append(flatness)
        frame_entropies.append(entropy)
        frame_rolloffs.append(rolloff)
        frame_zcr.append(zcr)
        frame_flux.append(flux)

    stride = max(1, math.ceil(len(starts) / 600))
    for i in range(0, len(starts), stride):
        frame_records.append({
            "time_seconds": (starts[i] + frame_length / 2.0) / sample_rate,
            "rms_dbfs": 20.0 * math.log10(frame_rms_values[i]) if frame_rms_values[i] > 0 else None,
            "spectral_centroid_hz": frame_centroids[i],
            "spectral_flatness": frame_flatnesses[i],
            "spectral_entropy_normalized": frame_entropies[i],
            "spectral_rolloff_85_hz": frame_rolloffs[i],
            "zero_crossing_rate": frame_zcr[i],
            "spectral_flux": frame_flux[i],
        })

    max_frame_rms = max(frame_rms_values, default=0.0)
    floor = max_frame_rms * 0.01
    low_energy = sum(value <= floor for value in frame_rms_values) / len(frame_rms_values) if max_frame_rms else 1.0
    mean_psd = [value / len(starts) for value in psd_accumulator]
    psd_db = [10.0 * math.log10(max(value, 1e-18)) for value in mean_psd]
    bin_hz = sample_rate / fft_size
    total_psd = sum(mean_psd) * bin_hz
    high_psd = sum(value for value, frequency in zip(mean_psd, frequencies) if frequency >= 4_000.0) * bin_hz
    waveform_stride = max(1, math.ceil(len(raw) / 1_000))
    waveform = raw[::waveform_stride][:1_000]

    return {
        "configuration": {"sample_rate_hz": sample_rate, "frame_length_samples": frame_length,
            "frame_duration_ms": 25.0, "hop_length_samples": hop, "hop_duration_ms": 10.0,
            "fft_size": fft_size, "window": "Hann",
            "resampling": "windowed-sinc polyphase-compatible fallback; see implementation notes"},
        "time_domain": {"duration_seconds": len(raw) / sample_rate, "sample_count": len(raw),
            "rms_amplitude": rms, "rms_dbfs": 20.0 * math.log10(rms) if rms else None,
            "peak_amplitude": peak, "peak_dbfs": 20.0 * math.log10(peak) if peak else None,
            "dc_offset_mean_amplitude": dc, "clipping_sample_fraction_at_0_999": clipping,
            "low_energy_frame_fraction_relative_minus_40_db": low_energy},
        "frequency_domain": {"spectral_centroid_mean_hz": sum(frame_centroids) / len(frame_centroids),
            "spectral_flatness_mean": sum(frame_flatnesses) / len(frame_flatnesses),
            "spectral_entropy_normalized_mean": sum(frame_entropies) / len(frame_entropies),
            "spectral_rolloff_85_mean_hz": sum(frame_rolloffs) / len(frame_rolloffs),
            "zero_crossing_rate_mean": sum(frame_zcr) / len(frame_zcr),
            "spectral_flux_mean": sum(frame_flux) / len(frame_flux),
            "power_below_4khz_fraction": max(0.0, min(1.0, 1.0 - high_psd / total_psd)) if total_psd else 0.0},
        "waveform": {"time_step_seconds": waveform_stride / sample_rate, "samples": waveform},
        "power_spectral_density": {"frequency_hz": frequencies, "power_db_per_hz": psd_db,
            "frequency_resolution_hz": bin_hz, "units": "relative full-scale amplitude squared per Hz"},
        "frame_measurements": frame_records,
    }

