# Contributing to ordvec

Thanks for your interest! `ordvec` is a training-free ordinal & sign
vector-quantization crate, and it underpins the OrdVec / RankQuant paper.
Contributions to the code, the docs, and the paper are all welcome.

## Ground rules

- **No system dependencies in the core crate.** No BLAS, OpenBLAS, faer,
  ndarray, or statrs in `ordvec`. CI enforces this (the `deps` job greps the
  dependency tree). The `ordvec-python` binding legitimately pulls `ndarray`
  via rust-numpy — that's fine; the guard is scoped to the core crate.
- **No fabricated benchmarks.** The only in-repo benchmark is the
  reproducible `examples/bench_rank` synthetic run. Real-corpus numbers are
  user-runnable and reported in the paper — please don't add unverifiable
  performance claims.
- **Keep theory claims bounded.** Cite theorem names or formalization docs for
  proof-backed statements, and preserve the finite in-model vs real-encoder
  caveat. The Lean bitmap theorem proves a constant-weight overlap admission
  model under explicit assumptions; it is not a blanket retrieval guarantee.
- **MSRV is Rust 1.89.** Don't use newer standard-library or language APIs.
- **Stable surface.** The persistence file magics (`.tvr` / `.tvrq` /
  `.tvbm` / `.tvsb`) and the public method names
  (`new` / `add` / `search` / `search_asymmetric*` / `top_m_candidates*` /
  `write` / `load`) are stable — please don't rename them.
- **Tests are required for new functionality.** As major new functionality
  is added, tests covering it MUST be added to the automated test suite
  (`cargo test`, plus `pytest` for the Python bindings). Changes that add
  capability without accompanying tests will be asked to add them before
  merge.

## The gate (run before opening a PR — mirrors CI)

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test                            # default surface
cargo test --features experimental
cargo test --no-default-features
cargo +1.89.0 build                   # MSRV
cargo build --locked
RUSTFLAGS="-D warnings" cargo build
cargo deny check                      # licenses / advisories / bans / sources
```

SIMD dispatches at runtime — AVX-512 / AVX2 on x86_64, NEON on aarch64,
simd128 on wasm32, and a scalar fallback elsewhere. If you change a SIMD
kernel, the AVX-512 path is exercised in CI under Intel SDE; locally, run on
an AVX-512 host or via SDE.

### Python bindings (`ordvec-python/`)

```sh
cargo fmt -p ordvec-python --check
cargo clippy -p ordvec-python --all-targets -- -D warnings
maturin develop && pytest ordvec-python/tests   # in a virtualenv
```

### Fuzzing

The loader, write→load round-trip, and FastScan paths have seven cargo-fuzz
targets in `fuzz/`. CI runs a bounded smoke on every PR (`fuzz.yml`); locally
you need a nightly toolchain and `cargo install cargo-fuzz`:

```sh
cargo +nightly fuzz build                 # compile all targets
cargo +nightly fuzz run load_rank         # one target, ad hoc
./fuzz/run_full_fuzz.sh                    # full deep campaign (all targets)
```

`run_full_fuzz.sh` runs each target in libFuzzer fork mode, persists the corpus
(resumable across runs), and collects crash artifacts. **It is heavy by
default** (~3h × 7 targets, `cores − 2` forks) — dial it down on a laptop via
env knobs, e.g. `SECS_PER_TARGET=120 FORKS=2 ./fuzz/run_full_fuzz.sh`. See the
script header for all knobs.

## Workflow

- **Branches:** `<type>/<slug>` (feat, fix, refactor, docs, test, chore,
  perf, ci).
- **Commits:** `<type>: <description>`.
- Update `CHANGELOG.md` under `[Unreleased]` for any user-facing change.
- Open a PR against `main` and fill in the PR template checklist.
- For larger changes, open an issue (or a Discussion) first so we can agree
  the approach before you invest time.

## Releases

Changelog and release notes are generated with
[git-cliff](https://git-cliff.org) from Conventional Commit history
(`cliff.toml`).

- **The whole release is automated except the two registry publishes.** Pushing
  a `vMAJOR.MINOR.PATCH` tag triggers `.github/workflows/release.yml`, which
  runs git-cliff for the GitHub Release notes, builds the crate + wheels +
  sdist, generates SLSA build provenance (`*.intoto.jsonl`) and a Sigstore
  bundle (`*.sigstore.json`), attaches everything to the GitHub Release, and
  un-drafts it — all without human intervention. The `crates.io` and `pypi`
  publishes wait at GitHub Environments with **Required reviewers** (the
  maintainer approves each in the Actions UI). Pre-release tags (e.g.
  `v0.3.0-rc.1`) do not trigger it.
- **`CHANGELOG.md` is curated by hand** — it is not auto-committed, because
  `main` is branch-protected. Keep adding entries under `[Unreleased]`; at
  release time promote that block to `## [X.Y.Z] - YYYY-MM-DD`. To draft the
  section from commits instead:

  ```sh
  cargo install git-cliff           # once
  git cliff --unreleased --tag vX.Y.Z   # preview the next section
  ```

## Licensing

By contributing, you agree that your contributions are dual-licensed under
**MIT OR Apache-2.0**, matching the project.

## Developer Certificate of Origin (DCO)

All contributions must be signed off under the
[Developer Certificate of Origin](./DCO) (DCO 1.1) — signing off certifies that
you wrote the change, or otherwise have the right to submit it under the
project's license. Add a sign-off line to every commit with:

```sh
git commit -s
```

which appends a trailer using your `git config` identity:

```
Signed-off-by: Your Name <you@example.com>
```

A DCO check runs on every pull request, so commits missing a valid
`Signed-off-by` will be flagged. To fix a commit you already made, use
`git commit --amend -s`; to sign off a range, `git rebase --signoff <base>`.
(This is separate from commit *signing* — `git commit -s -S` does both.)
