# ordvec — Claude project context

`ordvec` is a **training-free ordinal & sign** vector-quantization crate for compressed
nearest-neighbour retrieval over high-dimensional embeddings. Pure Rust, **zero system
dependencies** (no BLAS / ndarray / faer). Original rank/sign retrieval work by Nelson
Spence, developed within the [turbovec](https://github.com/RyanCodrai/turbovec) project
(MIT, by Ryan Codrai) and factored out here as a standalone crate. Dual-licensed
**MIT OR Apache-2.0**.

It underpins the **OrdVec / RankQuant paper** (ordinal retrieval as a third category peer
to dense and sparse). It must stay **publishable-grade and fiction-free** — no fabricated
benchmarks or unverifiable perf claims.

## Status — 2026-05-23 (verify with `git log` / `gh pr list --repo Fieldnote-Echo/ordvec`)
- **Merged to main (PRs #1–7):** #1–5 = production hardening + CI gate + cargo-fuzz +
  de-fiction + dual-license; **#7 = OrdVec ontology rebrand (merged, `f6593cd`)**. main is
  now **v0.2.0** with the flat `src/` layout, `Rank/RankQuant/Bitmap/SignBitmap` names, and
  deprecated `*Index` *root* aliases (the `ordvec::rank_index::*` module path was removed).
- **Open issues:** #8 (perf: Spearman centre-drop optimization, deferred), #6 (fuzz:
  in-memory loader to drop per-iteration temp files).
- **Pre-publish polish (in flight):** 2 rustdoc warnings in `rank_io.rs` — `load_sign_bitmap`
  doc links to private `check_payload_bytes`/`check_dim`; small follow-up PR.
- **PUBLISH HELD** — never `cargo publish` for real without Nelson's explicit go (see Hard
  rules). **Publish-*prep* groundwork is now in progress** (Cargo.toml metadata / README /
  docs.rs / `--dry-run` audit); the actual publish stays gated.

## Public API (v0.2.0)
```rust
use ordvec::{Rank, RankQuant, Bitmap, SignBitmap};
```
- `Rank` — full-precision rank vectors (u16/coord). `RankQuant` — bucketed ranks, `bits`
  bits/coord (b ∈ {1,2,4}); symmetric + asymmetric (float-query LUT) scoring. `Bitmap` —
  top-bucket bitmap/doc, `popcount(Q AND D)`. `SignBitmap` — sign bitmap for sign-cosine
  candidate gen.
- `MultiBucketBitmap` — behind the `experimental` feature. `RankQuantFastscan` —
  `#[doc(hidden)]`, optional b=2 FastScan path.
- Deprecated `pub type *Index = *` aliases (e.g. `RankIndex` → `Rank`) exist for external
  back-compat only — **never use them internally** (the `-D warnings` build fails on the
  deprecation warning).

## Layout (flat, v0.2.0)
`src/`: `rank.rs` (rank-math primitives **and** the `Rank` index type), `quant.rs`,
`bitmap.rs`, `sign_bitmap.rs`, `multi_bucket.rs`, `fastscan.rs`, `quant_kernels.rs`,
`util.rs`, `rank_io.rs` (persistence), `lib.rs`. Tests: `tests/index/`, `tests/redteam_*.rs`,
`tests/deprecated_aliases.rs`. `fuzz/` = cargo-fuzz loader targets.

**Workspace (since the Python bindings):** the root manifest is a `[workspace]`
(`resolver = "2"`, `members = ["ordvec-python"]`, `default-members = ["."]`, `exclude = ["fuzz"]`).
`ordvec-python/` is the PyO3/maturin binding member (`publish = false`; ships to **PyPI** as
`ordvec`, not crates.io): `src/lib.rs` (the `_ordvec` module wrapping Rank/RankQuant/Bitmap/SignBitmap
via the `ordvec_core` alias, with `ensure_finite` + contiguity guards turning core panics into
clean `ValueError`s), `python/ordvec/__init__.py` (re-exports the 4 + undocumented `*Index`
back-compat aliases), `pyproject.toml`, `tests/` (pytest). `default-members = ["."]` keeps the
bare core gates scoped to the crate; the binding is gated by `.github/workflows/python.yml`.

## Hard rules — DO NOT break
- **PUBLISH HELD**: never `cargo publish` for real without Nelson's explicit go. CI only does
  `cargo publish -p ordvec --dry-run --locked`.
- **PyPI PUBLISH HELD**: `ordvec-python` ships to PyPI as `ordvec` via **maturin**, separately
  from the crate, and is `publish = false` (off crates.io). Never publish the wheel without
  Nelson's explicit go — it's coordinated with the crate publish + the paper's Zenodo release
  (the paper consumes the bindings for a final cold-repro run).
- **No system deps (core crate)**: no blas/openblas/faer/ndarray/statrs in `ordvec`. The `deps`
  CI job greps `cargo tree -p ordvec` (scoped to the core crate) and fails on them. NB: the
  `ordvec-python` binding legitimately pulls rust-numpy→`ndarray`; that's fine for the PyPI
  artifact and is exactly why the grep is `-p ordvec`-scoped, not workspace-wide.
- **File magics** `.tvr` / `.tvrq` / `.tvbm` / `.tvsb` — never rename (persistence formats).
- **Method names** (`new`/`add`/`search`/`search_asymmetric*`/`top_m_candidates*`/`write`/`load`)
  — stable; don't churn.
- **MSRV 1.89** (AVX-512 intrinsics floor): keep `rust-version = "1.89"`; don't use newer APIs.
- **No fiction**: the only benchmark is the reproducible in-repo `examples/bench_rank`
  synthetic run; real-corpus results are user-runnable + live in the paper. Keep the README
  **Provenance** section accurate (original work *developed within* turbovec, factored out;
  turbovec credited) — do not reintroduce "extracted from turbovec" framing.
- **Cargo.lock** stays in sync with the manifest (a version bump must update the lock — the
  `--locked` deps gate enforces this).

## The gate (run before pushing — mirrors CI)
```
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test                          # 88   +  --features experimental  # 95
cargo test --no-default-features
cargo +1.89.0 build                 # MSRV
cargo build --locked
RUSTFLAGS="-D warnings" cargo build  # rustc-warning-clean (CI sets this on every job)
cargo +nightly fuzz build           # fuzz targets compile
```
SIMD dispatches at runtime via `is_x86_feature_detected!` (AVX-512/AVX2 + scalar fallback).
x86-only items are `cfg(target_arch="x86_64")`-gated; the glue (`SimdTier`,
`BATCHED_AVX512_CHUNK`) carries `cfg_attr(not(x86_64), allow(dead_code))` so the crate builds
clean on aarch64 (CI's `macos-latest` is ARM). CI's `avx512` job runs the suite under Intel
SDE (Sapphire Rapids) so the AVX-512 kernels are actually exercised on hosted runners.

**Binding gate** (if `ordvec-python/` changed): `cargo fmt -p ordvec-python --check` +
`cargo clippy -p ordvec-python --all-targets -- -D warnings`, then build & test the wheel —
`maturin develop` + `pytest ordvec-python/tests` in a venv. Mirrors `.github/workflows/python.yml`.

## Roadmap (next, in order)
1. ~~Merge #7 (rebrand)~~ — **done** (main is v0.2.0).
2. **Publish prep** → crates.io (`ordvec` is available). **Groundwork in progress**; the
   `cargo publish` itself stays **GATED on Nelson's explicit go**.
3. **Python bindings** — `ordvec-python` (abi3) **IN PROGRESS** (branch `feat/ordvec-python-bindings`):
   workspace member added, the 4 classes ported to the ontology, pytest green (108 passed + 1 xfail).
   Remaining: open the PR (bot/Codex loop). **PyPI publish stays GATED** — coordinated with the crate
   publish + the paper's Zenodo release. (Discipline: extract → rebrand → publish → python.)

## Working conventions
- Commits `<type>: <desc>`; branches `<type>/<slug>`; PRs against `Fieldnote-Echo/ordvec`
  (Nelson's repo; `origin`). Never force-push main; never `git reset --hard`; stage specific
  files; commit/push only when asked.
- PR reviews come from copilot / gemini / qodo, plus a Codex stop-gate review. Pattern: triage
  every finding (fix / defer-to-issue / explain-as-non-issue), then resolve the remediated
  review threads (`gh api graphql` `resolveReviewThread`) and reply on deferred ones with the
  tracking issue.
- Durable project facts live in this repository's local Claude project memory
  (`MEMORY.md` index); do not commit machine-specific memory paths. The broader
  REO / paper / portfolio context lives in the turbovec project memory.
