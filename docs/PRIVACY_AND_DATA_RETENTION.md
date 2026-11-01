# Privacy and data retention

## Default behavior

The app runs locally and binds to loopback. It performs no network inference, telemetry, analytics, cloud storage, or account lookup. The optional-looking AI explanation in the supplied prototype is not part of this application.

Uploaded bytes are read into memory, decoded, and released after analysis or feature extraction. AegisVoice does not save the source audio. It saves the sanitized filename, SHA-256, byte count, timestamp, duration, original rate/channel count, quality notes, DSP values, decision, and report JSON in `data/aegisvoice.sqlite3`. The training table stores explicit labels, group/dataset names, file hashes, and derived feature vectors; the trained local checkpoint is stored under `data/models/`.

The filename can itself reveal personal information. SHA-256 can identify the same file across copies. Keep `data/` on a device and account you control; it is ignored by Git but not encrypted by this application. Do not include private recordings in support bundles. Logs are local and should still be reviewed before sharing.

## Deletion and retention

History deletion removes the corresponding database row. Because audio is never retained, deletion does not remove the original recording. No automatic expiry or database vacuum/backup policy is implemented. A future retention setting must separately specify report metadata, file bytes, and exported copies.

## User notice

Obtain appropriate permission before analyzing recordings. AegisVoice does not verify consent, identity, truthfulness, or legal authenticity. Do not use an INCONCLUSIVE DSP report as the sole basis for a high-impact decision.
