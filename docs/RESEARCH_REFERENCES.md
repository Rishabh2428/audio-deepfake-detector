# Research and project references

## Supplied research papers

1. X. Liu et al., “ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, DOI `10.1109/TASLP.2023.3285283`. The supplied PDF distinguishes spoofing settings and highlights channel/codec and real-world conditions. It motivates reporting task and domain boundaries; it does not define a universal threshold for this application.
2. J. Yi et al., “Audio Deepfake Detection: A Survey,” 2023. The supplied survey reviews task types, datasets, features, classifiers, and evaluation, and identifies generalization to unknown attacks, in-the-wild data, and interpretability as open challenges.
3. A. O. M. Salih et al., “Deepfake Audio Detection in Voice Authentication: A Spectral and CNN-Based Comprehensive Review,” Engineering, Technology & Applied Science Research, 2025, DOI `10.48084/etasr.13400`. The supplied review is secondary context only. Its descriptive claims and citations are not used as measured performance or threshold values in AegisVoice.

## Supplied project documents

The B.Tech report and the second project report describe a proposed DSP-first pipeline and state that benchmark experiments had not yet been run; the second report also records issues in the browser prototype's AUC/EER, split, and threshold logic. The synopsis and presentation are proposal/communication artifacts rather than evaluation evidence. The code listing PDF prints the prototype's source and is not a separate dataset or model artifact.

## Evidence rule

No reported measurement in these references is represented as an AegisVoice result. This workspace contains an experimental user-trained DSP logistic checkpoint with a 16-file grouped holdout. Its perfect metrics on this small user-supplied corpus do not establish performance on independent speakers or generators. No research neural checkpoint or benchmark result is claimed for AegisVoice.
