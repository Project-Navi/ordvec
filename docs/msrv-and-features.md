# MSRV and Feature Stability

This matrix is the release-facing build contract for downstream embedders,
packagers, and host systems. It complements the
[pre-1.0 compatibility policy](compatibility-policy.md), which defines how
compatibility-impacting changes are classified. The release artifact and wheel
target inventory lives in [artifact-platform-matrix.md](artifact-platform-matrix.md).

Current MSRV: Rust 1.89.

The MSRV applies to all Rust crates in this repository: `ordvec`,
`ordvec-manifest`, `ordvec-python`, `ordvec-manifest-python`, and
`ordvec-ffi`. The CI MSRV job, each `Cargo.toml` `rust-version`, and the
README MSRV badge/section must stay synchronized. Raising the MSRV is a
minor-version compatibility change and release notes must state the reason and
any migration note.

## Feature Matrix

| Surface | Default features | Stable default-off features | Optional dependency features | Experimental/internal features |
| --- | --- | --- | --- | --- |
| `ordvec` | none | none | none | `experimental` exposes `MultiBucketBitmap`; `test-utils` is repo-test-only and has no public stability promise. |
| `ordvec-manifest` | none | none | `cli`, `sqlite`, `sqlite-bundled` | none |
| `ordvec-python` | n/a | n/a | n/a | n/a |
| `ordvec-manifest-python` | n/a | n/a | n/a | n/a |
| `ordvec-ffi` | none | none | none | none |

SIMD dispatch in `ordvec` is not feature-gated. x86_64 dispatches AVX-512 and
AVX2 at runtime, aarch64 uses NEON, wasm32 can use `simd128` when the target is
built with that target feature, and other targets use the scalar fallback.
Host systems should not need BLAS, LAPACK, `ndarray`, `faer`, or a native graph
library to embed the core crate.

`ordvec-manifest` keeps its library default feature set empty. The `cli`
feature enables the `ordvec-manifest` binary and its `clap` dependency. The
`sqlite` feature enables the local cache/audit subcommands; `sqlite-bundled`
adds the bundled SQLite build through `rusqlite`.

## Change Policy

New feature flags must declare a stability class before merging:

- stable default feature;
- stable default-off feature;
- optional dependency feature;
- experimental/default-off feature;
- internal repo-test-only feature.

Changing the default feature set is compatibility-impacting and must be
classified in release notes. Adding a new required system dependency, changing
wheel platform expectations, or making an optional dependency effectively
required is also compatibility-impacting.

Experimental and internal features can change before 1.0, but releases should
still call out changes likely to affect known downstream users. Stable feature
changes should include examples or migration notes when the visible build or
API surface changes.

## Release Checks

`python tests/release_publish_invariants.py` keeps the following in sync:

- lockstep crate and Python package versions;
- Rust MSRV declarations, README badge text, and CI MSRV toolchain;
- crates.io metadata, PyPI metadata, docs.rs feature policy, and package
  contents;
- release workflow and registry preflight expectations.

Release review should also compare touched code against this matrix so host
systems can embed `ordvec` without hidden platform or feature surprises.
