# Security threat model

## Deployment assumptions

The application is a single-user local tool. Both server paths bind to `127.0.0.1`; the documented full-mode command does not expose a LAN interface. No account, remote service, CORS permission, or secret is needed for local analysis. The measurement explanation is generated locally. Local processes and extensions with local-device access are inside the trust boundary.

## Threats and current controls

| Threat | Current control | Verification / remaining limit |
|---|---|---|
| Oversized request | FastAPI ASGI middleware bounds the complete analysis body; fallback checks Content-Length before reading; decoder has a separate file cap. | Decoder caps are unit-level; transport/body-limit HTTP test is blocked by socket restriction. |
| Malformed or mislabeled audio | Content is decoded by libsndfile or strict PCM `wave`; format, channel, sample rate, frames, finite values, and duration are checked. | Corrupt WAV and unsupported extension unit tests pass; other libsndfile codecs not installed/tested. |
| Path traversal / arbitrary file read | Upload path is never opened; only bytes are decoded. Stored filename is reduced to a safe basename and character set. | Multipart parser is in-process tested; fuzzing is not run. |
| Memory / CPU exhaustion | 50 MB/120-second full-mode caps, 8 MB/30-second fallback caps, two-channel limit, one concurrent analysis/preview slot, and bounded output arrays. | Worst-case resource profiling is not run. Third-party decode libraries still need clean-environment advisories. |
| Cross-site request / browser misuse | No CORS allowlist; analysis POST checks supplied Origin against Host; CSP restricts scripts/styles/connect to same origin. | Browser CORS/Origin behavior is not socket-tested. This is not authentication against local software. |
| Raw audio or identifier leak | No raw audio is written. Logs omit request paths/filenames in fallback. Measurement explanations use the locally stored report only. | Filename and SHA-256 are stored in local SQLite; filenames can contain personal information and hashes can identify duplicates. |
| SQL injection | SQLAlchemy expressions in full mode; parameterized SQL in fallback. | No dynamic SQL identifiers come from user input. |
| Stored script injection | UI inserts untrusted filenames/messages as text nodes; SVG charts use numeric arrays and fixed labels. | Manual browser security review remains. |
| Dependency supply chain | Direct packages are pinned in `requirements.txt`; package/license inventory script is provided. | No lock file or automated advisory scan was generated because package-index access failed. PyPI metadata/license is not a security audit. |
| Database corruption / future schema | SQLite schema version is checked; unknown future versions fail startup. | Backup/restore, migrations beyond v1, disk-full, and concurrent-reader tests remain. |

## Operational risks

The fallback has no authentication and is intended only on loopback. Do not change host to `0.0.0.0` or publish it through a reverse proxy without adding a threat review and authentication. Error responses are generic; server diagnostics go to local logs. The application is not represented as secure or forensic-grade based on these controls alone.
