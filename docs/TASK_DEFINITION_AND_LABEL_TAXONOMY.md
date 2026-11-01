# Task definition and label taxonomy

## Intended claim in this build

AegisVoice computes deterministic signal measurements and can run an experimental user-trained human-vs-synthetic classifier. Without a local checkpoint, or when blocking input-quality checks fire, its verdict is `INCONCLUSIVE`. When the model runs, its 1–11 synthetic-evidence index and threshold-based label are uncalibrated and depend on the supplied training data; they do not verify identity, provenance, or authenticity. Lesser quality warnings remain advisory.

## Canonical concepts for future datasets

| Concept | Definition | May be merged into a binary spoof class? |
|---|---|---|
| Bona fide speech | Speech treated as genuine under a source dataset's protocol; this does not authenticate identity or prove the recording is unedited. | Only when that dataset protocol defines it as the negative class. |
| Text-to-speech (TTS) | Speech generated from text by a synthesis system. | Preserve source attack label and generator when provided. |
| Voice conversion (VC) | Speech transformed to alter vocal identity/characteristics, often preserving linguistic content. | Preserve source attack label and method when provided. |
| Replay | Audio replayed through a physical loudspeaker/channel. | Do not equate with digital TTS/VC unless the benchmark explicitly defines one combined task. |
| Editing/splicing | Recording altered by editing, which may or may not include generated content. | Keep separate unless a named task justifies the mapping. |
| Codec/channel degradation | Compression, filtering, resampling, or transmission changes. | Not a synthetic label by itself. |
| Unknown/unrepresented | Condition outside the dataset and model's documented task. | Keep unknown; do not silently force a class. |

Future dataset manifests must preserve `source_label`, `canonical_name`, `task_family`, dataset/protocol version, speaker/generator identity where permitted, label source, and license. A normalized class order must be explicit in the dataset manifest and model artifact. Unknown, missing, contradictory, or unsupported labels must fail import or remain unlabelled; they must not be guessed from filenames.

## Future operating claim

Any later model card must say which dataset and protocol defined its classes, which domains and attacks were held out, and that measured performance does not prove identity, intent, truthfulness, consent, or legal authenticity. The UI wording must never exceed the validated task.
