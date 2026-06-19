# Artifact and Platform Matrix

This matrix is the release-facing inventory for what `ordvec` publishes, what
is repo-local, and which platform expectations are checked by the release
workflow. It complements `RELEASING.md` and `docs/msrv-and-features.md`; the
workflow remains the source of truth for the exact build jobs. The matrix
documents packaging and distribution compatibility for release verification. It
is not a service support commitment, runtime SLA, or guarantee that every host
environment matching a platform family is supported.

## Published Artifacts

| Surface | Published where | Platform/build contract | Release verification |
| --- | --- | --- | --- |
| `ordvec` Rust crate | crates.io package `ordvec`; GitHub Release `.crate` asset | Rust 1.89 MSRV; default features empty; pure Rust, no BLAS/LAPACK/system numeric dependency | `cargo package --locked`; GitHub/Sigstore/SLSA provenance; pre-publish and post-publish byte identity against crates.io |
| `ordvec-manifest` Rust crate | crates.io package `ordvec-manifest`; GitHub Release `.crate` asset | Rust 1.89 MSRV; default features empty; optional `cli`, `sqlite`, and `sqlite-bundled` features | Built after matching `ordvec` exists; GitHub/Sigstore/SLSA provenance; byte identity against crates.io |
| Python `ordvec` | PyPI package `ordvec`; GitHub Release wheels and sdist | CPython 3.10+ abi3; `numpy>=2.2`; wheels for Linux x86_64 and Linux aarch64 are manylinux/glibc wheels; no musllinux/Alpine wheel is shipped yet; macOS aarch64 and Windows x64 wheels are also published; native extension modules are embedded in the wheel and do not load a separate `ordvec_ffi` library | Canonical wheel/sdist selection; linux/aarch64 native smoke; PyPI hash verification; PEP 740 attestation on fresh upload |
| Python `ordvec-manifest` | PyPI package `ordvec-manifest`; GitHub Release wheels and sdist | CPython 3.10+ abi3; Linux wheels are manylinux/glibc for x86_64 and aarch64; no musllinux/Alpine wheel is shipped yet; macOS aarch64 and Windows x64 wheels are also published; native extension modules are embedded in the wheel | Canonical wheel/sdist selection; linux/aarch64 native smoke; PyPI hash verification; PEP 740 attestation on fresh upload |
| Node/WASM | Not shipped; no npm package is published yet | Placeholder for issue #138; no JavaScript, TypeScript, or wasm package support is promised by this release | No release verification until a future packaging lane adds build jobs |
| JVM | Not shipped; no Maven/Gradle package is published yet | Placeholder for issue #139; no Java/Kotlin package support is promised by this release | No release verification until a future packaging lane adds build jobs |

The Python release currently expects exactly four wheels plus one sdist for
each Python package. There is no macOS x86_64 wheel leg in the current release
workflow. Linux users on musl-based distributions should build from source or
from the sdist unless a future release adds a `musllinux` wheel leg.

## Repo-Local Sidecars

| Surface | Published where | Platform/build contract | Release role |
| --- | --- | --- | --- |
| `ordvec-ffi` | Not published to crates.io; built from the repository | Rust 1.89; emits `rlib`, `cdylib`, and `staticlib`; ABI v1 header is committed under `ordvec-ffi/include/`; cdylibs are named `libordvec_ffi.so`, `libordvec_ffi.dylib`, or `ordvec_ffi.dll`; static archives are named such as `libordvec_ffi.a` | C ABI compatibility surface for embedders; CI checks header drift and C link smoke; embedders must pair header and native library from the same git tag, require `ordvec_abi_version() == 1`, and compare `ordvec_version_string()` with the packaged native library |
| `ordvec-go` | Not published as a Go module release from this repo | Thin cgo wrapper over `ordvec-ffi`; links the local Rust library from the same git tag and ABI version | Binding smoke and race/cgocheck coverage for the C ABI contract; consumers must not mix Go wrapper, generated header, and native library from different tags |
| `benchmarks/beir-bench` | Not shipped in the published crate or wheels | Workspace benchmark crate with `publish = false` | Release-adjacent benchmark harness only; not a shipped user dependency |
| `fuzz/` | Not a workspace member and not published | `cargo-fuzz` crate with its own lockfile | Loader and parser hardening gate; release workflow runs smoke fuzz jobs |

Loading/linking strategy for the repo-local native sidecars is part of the
release contract: C embedders load or link the `ordvec-ffi` dynamic/static
library that matches the checked-in ABI v1 header; Go uses cgo to link that
same local `ordvec-ffi` build; Python wheels embed their own native extension
modules and do not load the repo-local `ordvec_ffi` shared library. Keep the
header, Go wrapper, and native libraries on the same git tag and ABI version.

## Native Libraries and Version Alignment

- The Rust crates published to crates.io do not require a separately installed
  native numeric library.
- The repo-local `ordvec-ffi` crate is named `ordvec_ffi`; dynamic builds emit
  `libordvec_ffi.so` on Linux, `libordvec_ffi.dylib` on macOS, and
  `ordvec_ffi.dll` on Windows. Static builds emit archives such as
  `libordvec_ffi.a`.
- Python wheels embed their native extension modules inside the wheel. They do
  not load a separately installed `ordvec_ffi` shared library.
- The Go wrapper links through cgo against a local `ordvec-ffi` build. Use the
  Go package, generated header, and native library from the same git tag and ABI
  version.
- C and Go embedders should check `ordvec_abi_version() == 1` and
  `ordvec_version_string()` against the packaged native library. Do not mix
  headers, Go wrappers, and native libraries from different tags.

## SBOM Policy

The release workflow generates CycloneDX SBOMs for the Rust crate, manifest
crate, Python binding crate, and manifest Python binding crate as workflow
artifacts. Current PyPI distributions and GitHub Release assets do not embed or
attach those SBOM files. Published release assets are the canonical `.crate`,
wheel, and sdist files plus Sigstore and SLSA/in-toto provenance assets.

## Platform Notes

- SIMD dispatch in the core crate is not feature-gated. x86_64 dispatches
  AVX-512 and AVX2 at runtime where available, aarch64 uses NEON, wasm32 can
  use `simd128` when built with that target feature, and unsupported targets
  use scalar fallback paths.
- Native library naming, loading/linking strategy, and same-tag version
  alignment are documented above in "Native Libraries and Version Alignment";
  those rules are part of this platform matrix, not optional packaging notes.
- Published Python wheels are abi3, so one wheel per platform covers CPython
  3.10 and newer for that platform.
- The release workflow keeps the GitHub Release draft until both Rust crates
  and both Python packages have published successfully. A registry failure
  leaves the release draft unpublished.
- GitHub Release assets include the canonical crate, wheel, and sdist files
  plus provenance/attestation assets generated by the workflow. Verify
  downloaded assets with `gh attestation verify` and registry-served hash
  checks before treating them as deployment inputs.
