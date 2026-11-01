# Licenses and attribution

## Project code

The AegisVoice source in this workspace is offered under the MIT License in `LICENSE`. No third-party model weights, audio datasets, fonts, icons, or research-paper files are bundled. The supplied materials remain in their original location and are not relicensed by this project.

## Direct full-mode Python dependencies

Versions were pinned on 28 October 2026 based on the current PyPI project metadata. These packages are installed by the user from PyPI; they are not vendored here. License identifiers below refer to package metadata and upstream licensing; review the installed distribution notices before redistribution.

| Package | Pinned version | Declared license | Role | Source |
|---|---:|---|---|---|
| FastAPI | 0.162.0 | MIT | Typed API and static file service | [PyPI](https://pypi.org/project/fastapi/) |
| Uvicorn | 0.54.0 | BSD-3-Clause | ASGI server | [PyPI](https://pypi.org/project/uvicorn/) |
| SQLAlchemy | 2.0.55 | MIT | SQLite persistence | [PyPI](https://pypi.org/project/SQLAlchemy/2.0.55/) |
| NumPy | 2.5.4 | BSD-3-Clause, 0BSD, MIT, Zlib, CC0-1.0 components | Numeric arrays and FFTs | [PyPI](https://pypi.org/project/numpy/) |
| SciPy | 1.18.1 | BSD | Rational polyphase resampling | [PyPI](https://pypi.org/project/scipy/1.18.1/) |
| SoundFile | 0.14.0 | BSD-3-Clause wrapper; wheel includes libsndfile | Audio decode | [PyPI](https://pypi.org/project/soundfile/0.14.0/) |
| python-multipart | 0.0.32 | Apache-2.0 | FastAPI multipart upload parser | [PyPI](https://pypi.org/project/python-multipart/) |

FastAPI also resolves transitive packages such as Starlette and Pydantic. Their exact versions and license metadata depend on the resolved installation. Run the inventory script in the clean virtual environment after installation:

```powershell
.\.venv\Scripts\python.exe scripts\create_sbom.py --output reports\software-inventory.json
```

The script lists every installed distribution visible to that interpreter. It is an inventory, not a security audit. A lock file and transitive license review have not been generated because the current development environment could not access PyPI.

## Standard-library fallback

Offline WAV mode uses CPython standard-library modules (`wave`, `sqlite3`, `http.server`, `email`, `math`, and related modules), covered by the Python Software Foundation License. No external package is needed for that mode.

## Datasets, models, and research sources

No dataset or checkpoint is downloaded or bundled. The supplied ASVspoof papers and the project PDFs are sources, not runtime dependencies. Before adding any dataset or model, verify the official terms for the specific version, intended research/commercial use, attribution, and redistribution. A free download is not assumed to grant redistribution rights.

## Security advisory status

No dependency advisory scan could be run in this environment. Before release, install from a clean environment, generate the installed-package inventory, check upstream advisories for the resolved versions, and record the result. Do not treat PyPI provenance or license metadata as proof that a package is secure.
