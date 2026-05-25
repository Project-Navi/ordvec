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
- **MSRV is Rust 1.89.** Don't use newer standard-library or language APIs.
- **Stable surface.** The persistence file magics (`.tvr` / `.tvrq` /
  `.tvbm` / `.tvsb`) and the public method names
  (`new` / `add` / `search` / `search_asymmetric*` / `top_m_candidates*` /
  `write` / `load`) are stable — please don't rename them.

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

- **GitHub Release notes are automated.** Pushing a `vMAJOR.MINOR.PATCH` tag
  triggers `.github/workflows/changelog.yml`, which runs git-cliff and opens a
  **draft** GitHub Release with the generated notes — review, then publish.
  Pre-release tags (e.g. `v0.3.0-rc.1`) do not trigger it.
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
