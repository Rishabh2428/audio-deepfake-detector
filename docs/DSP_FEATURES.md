# DSP measurement specification

## Scope and defaults

These are descriptive signal statistics, not classifier features with validated decision meaning. Defaults in the supplied project brief are used for the common path: 16,000 Hz, 25 ms frames (`N = 400`), 10 ms hop (`H = 160`), 512 point real FFT, Hann window. This is an engineering configuration, not an empirically optimized detector setup. The full path uses `scipy.signal.resample_poly`; fallback uses a 49-tap-radius finite windowed-sinc low-pass interpolation kernel.

Input is finite normalized floating PCM from the decoder. For the canonical mono sequence `x[n]`, the measured mean is `μ = (1/Nx) Σ x[n]`, where `Nx` is the sample count. The spectral analysis array is `x_c[n] = x[n] - μ`. The original canonical samples remain the source for waveform and level measurements. No denoising or enhancement is applied.

## Time-domain values

- Duration: `Nx / fs` seconds; `fs` is the analysis sample rate.
- RMS amplitude: `sqrt((1/Nx) Σ x[n]^2)`; dBFS is `20 log10(RMS)`. For zero RMS, dBFS is returned as `null` rather than a fabricated finite value.
- Peak: `max_n |x[n]|`; peak dBFS uses `20 log10(peak)` and is null at zero.
- DC offset: `μ`, in normalized PCM amplitude units.
- Near-full-scale fraction: `(1/Nx) Σ 1(|x[n]| ≥ 0.999)`. `0.999` is an engineering indicator near full scale; it is not a clipping diagnosis or spoof threshold.
- Low-energy frame fraction: proportion of frames with centered frame RMS `≤ 10^(-40/20) * max_frame_RMS`. It is not VAD and does not identify silence or speech.
- Waveform: at most 1,000 evenly strided canonical samples, including their original DC level.

## Frame spectrum and PSD

For frame `t`, `x_t[n] = x_c[tH+n] w[n]`, where `w[n] = 0.5(1 - cos(2πn/(N-1)))` for `0 ≤ n < N`. Let `X_t[k]` be the 512-point real FFT and `K = 257` one-sided bins. The one-sided periodogram is `P_t[k] = c[k] |X_t[k]|² / (fs Σ w[n]²)`, with `c[k]=1` at DC and Nyquist and `2` at other bins. Mean PSD is the arithmetic mean of `P_t[k]` across frames. Units are relative full-scale amplitude squared per Hz; frequency spacing is `fs/512`.

- Power-weighted centroid: `Σ f_k P_t[k] / Σ P_t[k]`, averaged over nonzero frames, in Hz.
- Power spectral flatness: `exp((1/K)Σ ln(max(P_t[k], ε))) / ((1/K)Σ P_t[k])`, where `ε` is the smallest positive float used for numerical safety. Zero-power frames return 0.
- Normalized spectral entropy: `-Σ p_t[k] ln(p_t[k]) / ln(K)`, where `p_t[k] = P_t[k]/ΣP_t[k]`. Zero-power frames return 0.
- 85% power roll-off: smallest `f_k` for which cumulative `P_t[k]` reaches `0.85 ΣP_t[k]`.
- Zero-crossing rate: number of adjacent sign changes divided by `N-1` samples per frame; reported as a dimensionless ratio.
- Spectral flux: `||m_t - m_(t-1)||₂`, where `m_t = |X_t| / |||X_t|||₂`; first frame is 0. It is a magnitude-spectrum change measure, not a synthesis cue.
- Below-4-kHz PSD fraction: `1 - (integrated PSD for f ≥ 4,000 Hz)/(integrated total PSD)`. At 16 kHz its measured band ends at 8 kHz. It is descriptive and does not establish an artificial cutoff.
- Display spectrogram: power is sampled at 48 logarithmically spaced frequency points from the first usable FFT bin to Nyquist, for at most 600 evenly sampled frames. Each displayed value is quantized to 8 bits over an 80 dB range relative to the loudest bin in the recording. This visualization is not an active classifier input and colors are relative within a file, not comparable confidence values.
- Envelope modulation: the 25 ms frame-RMS sequence, sampled at the 10 ms hop rate, is mean-centered and transformed by an FFT. The report includes relative energy in 2–8 Hz compared with 0.5–20 Hz, its peak frequency, and the RMS-envelope coefficient of variation. This is temporal texture evidence, not voice activity detection; speech content, clip length, and recording conditions affect it. It is not used by the active classifier.

## Cepstral, prosodic, and phase summaries

- MFCC and LFCC: log energies from 27 mel-spaced or linear-spaced triangular filters over the STFT power, followed by a type-II DCT. The UI displays coefficient means; the advanced classifier uses mean coefficients C1-C4 from each family.
- CQT-style cepstra: DCT summaries of 48 geometrically spaced triangular STFT filters. This approximates log-frequency spacing; it is not a variable-window constant-Q transform and is not exact CQCC.
- Pitch and voicing: every fourth 25 ms frame is centered and searched for the strongest normalized autocorrelation peak in an approximate 70-400 Hz range. F0 and voiced-frame fraction are rough estimates, not a dedicated pitch tracker.
- HNR proxy: transforms that same autocorrelation peak into a harmonicity and HNR-like value. It is sensitive to speech content and noise and is not a calibrated voice-quality measure.
- Jitter proxy: average absolute change in the selected lag between adjacent sampled voiced frames, divided by the mean selected lag. Shimmer proxy: average absolute frame-RMS change divided by mean voiced-frame RMS. These are short-frame fluctuation proxies, not clinical or standard-cycle jitter/shimmer measurements.
- Phase-aware STFT summary: frame-to-frame phase change has the expected FFT-bin advance removed; mean wrapped residual and a large-residual ratio are reported. A group-delay outlier ratio summarizes within-frame frequency-bin phase differences. Phase cues are highly sensitive to codec and editing operations.
- Additional spectral summaries: spectral spread is the power-weighted standard deviation around the centroid; slope is a least-squares dB-per-octave trend; crest is peak magnitude divided by mean magnitude; flux/RMS variation and band fractions describe texture and level.

The extended feature vector contains 38 finite values. Unsupported or missing fields in older reports are not filled into the advanced feature schema, so only valid basic vectors from older history records can train the basic model.

Numerical arrays are checked for finite values; ratios with zero denominators produce 0 or null according to the field definition. Per-frame UI data are deterministically strided to no more than 600 records. PSD has 257 bins; waveform has no more than 1,000 points.

## Quality notices and parameter provenance

The UI warns below one second, below -50 dBFS RMS, on any sample at/above 0.999 full scale, and when stereo averaging leaves under 10% of the channel RMS reference. These are explicitly engineering warnings. They are not learned, published, or validated class thresholds. No measurement maps to human-like or synthetic-like evidence.

## Validation evidence

`tests/test_offline.py` uses programmatically generated tone/silence fixtures to check 1 kHz PSD peak and centroid, RMS, finite PSD values, silence handling, duration-preserving resampling, and attenuation above the 8 kHz target Nyquist. These software fixtures verify implementation arithmetic only; they are not a deepfake benchmark.
