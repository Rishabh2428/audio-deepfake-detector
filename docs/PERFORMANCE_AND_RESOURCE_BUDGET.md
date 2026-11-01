# Performance and resource budget

## Configured limits

| Mode | File cap | Duration cap | Source rate | Channels | Concurrent analysis |
|---|---:|---:|---:|---:|---:|
| FastAPI full mode | 50 MB | 120 seconds | 4-192 kHz | 1-2 | 1 |
| Standard-library fallback | 8 MB | 30 seconds | 4-192 kHz | 1-2 | 1 |

The common DSP representation is 16 kHz mono. Frames are 400 samples at a 160-sample hop, producing at most about 12,000 frames for the 120-second full-mode limit. The full NumPy path materializes the 12,000 x 400 frame matrix (~38 MB float64), 12,000 x 257 complex FFT result (~49 MB complex128), and derived arrays. The upload body can include a 50 MB audio file plus multipart overhead. Peak memory is expected in the hundreds of MB per active request; it has not been measured. A single analysis/preview slot prevents the dominant allocations from multiplying.

The fallback stores Python float lists and computes a radix-2 FFT in Python. Its 30-second limit is intentionally lower. Runtime at the limit has not been measured. It is suitable for inspection, not batch throughput or real-time processing.

## Recovery and resource behavior

Oversized requests fail before decoder allocation. Inputs beyond configured duration or invalid metadata are rejected. No automatic large model or dataset downloads occur. If decoding or DSP fails, the request returns an error and no result row is committed. SQLite uses a five-second lock timeout in fallback; contention behavior has not been stress-tested.

## Measurement plan

After installing the pinned full environment on Windows 11, measure peak resident memory and elapsed time for 1-, 30-, and 120-second mono/stereo WAV/FLAC inputs at 8, 16, 48, and 96 kHz. Record machine CPU/RAM, exact package inventory, decode format, and run count. Keep these as performance measurements, not detector accuracy.
