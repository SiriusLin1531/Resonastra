# Resonastra

Local Voice Training & Text-to-Speech
Fully offline. No cloud required.

本地语音训练与文本转语音
完全离线，无需云端服务。

Train Your Voice. Keep It Local.

## v1.0.0 Download

Download the binary from the **v1.0.0 release/mirror listed in release metadata** and verify it against the canonical SHA256 published here.

- Archive: `Resonastra_v1.0.0.zip`
- Size: `13,866,120,773` bytes
- SHA256: `420772a840fe496e175501085c20191c83b1433d366ea5bb060af1c5db093e41`
- Package manifest SHA256: `75661fd2fc02793f972866b26dda19e3e12352a22f2618194738f4a774d1a279`
- Checksum sidecar: `Resonastra_v1.0.0.zip.sha256`

The external binary URL is bound in release metadata after the clean public source commit is frozen. No placeholder download URL is published in this source snapshot.

## Canonical source baseline

- Canonical release-source commit: `905f8546406130ff2ea609ff8207b80d24f60549`
- Canonical source inventory SHA256: `2186ded0dfc7b4dc836e63e360b0c5c19661e379a585972b52b540fe9a9456bb`
- Canonical source correspondence: `247 / 247`

## System requirements

- Windows 10/11 x64
- Compatible NVIDIA GPU recommended for training/inference
- Microsoft Visual C++ Redistributable compatible with the bundled runtime
- Sufficient local storage for the extracted runtime, models, datasets, and outputs

The accepted v1.0.0 runtime baseline uses CPython 3.10.20, PyTorch 2.5.1, and CUDA runtime 11.8.

RTX 50-series / `sm_120` may emit an architecture-support warning. **Resonastra v1.0.0 does not claim native `sm_120` support.**

## Offline behavior

The distributed Windows package is designed to operate locally after extraction. The accepted DataFactory v1.0.0 ASR boundary uses local Chinese FunASR assets and does not download ASR models at runtime.

## Stable launchers

- `launch_check_env.bat`
- `launch_data_factory_ui.bat`
- `launch_training_ui.bat`
- `launch_infer_ui.bat`

## Public source vs binary payload

The public source repository intentionally excludes bundled runtime binaries, pretrained model weights, caches, user data, generated outputs, and private release-engineering evidence.

Those binary-distributed components are bound by the qualified v1.0.0 archive above.

## License and third-party software

Resonastra project-owned source is released under the MIT License. Third-party code and assets retain their own terms. See `THIRD_PARTY_NOTICES.md` and the license files included with selected third-party source.

## Security and contributions

See `SECURITY.md` and `CONTRIBUTING.md`.
