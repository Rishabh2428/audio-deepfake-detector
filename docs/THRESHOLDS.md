# Thresholds and evidence interpretation

The saved model starts with a 0.5 synthetic-score cutoff. The user can adjust it from 0.20 to 0.80; this changes the label decision rule, not the classifier or its accuracy. The app maps its raw uncalibrated output onto a 1–11 synthetic-evidence index for display. This index is not a probability or scientifically validated. A result can receive a preliminary `LIKELY_HUMAN` or `LIKELY_SYNTHETIC` label only when the saved model is available and blocking quality checks pass. Silence, recordings shorter than one second, strong stereo cancellation, and strongly tonal non-speech remain `INCONCLUSIVE`; lesser quality warnings remain advisory.

The following values only create quality notices or define a DSP representation:

| Value | Use | Origin | Meaning |
|---|---|---|---|
| 0.999 normalized amplitude | Count near-full-scale samples | Engineering warning | Signal may have clipping; not proof of synthesis. |
| -40 dB relative to maximum frame RMS | Low-energy frame fraction | Engineering descriptor | Not VAD, not a pause label, not spoof evidence. |
| -50 dBFS RMS | Low-level notice | Engineering warning | Measurements may be dominated by noise/silence. |
| 1 second duration | Short-recording notice | Engineering warning | Frame aggregates may be unstable; not an authenticity rule. |
| 0.1 downmix RMS ratio | Stereo cancellation notice | Engineering warning | Averaging may have removed substantial energy; not an authenticity rule. |
| 0.85 cumulative frame power | Spectral roll-off feature | Descriptive convention | No classifier threshold is implied. |
| Entropy < 0.38 and flatness < 0.005 | Strongly tonal input gate | Heuristic quality check | Withholds voice-model inference for signals such as DTMF; it is not evidence of a deepfake. |

The user-trained model still needs independently verified labels, speaker/source/generator-disjoint evaluation, calibration data if probabilities are shown, uncertainty intervals, and a threshold selected for an explicitly defined use. A feature reference range, audio-quality gate, and decision threshold remain separate. No published benchmark score or cutoff is transferred to this build.
