# Releasing `ordvec`

> **Publish is held.** A real `cargo publish` / `maturin publish` happens only
> on the maintainer's explicit go. CI never publishes for real — the crate job
> runs `cargo publish -p ordvec --dry-run --locked`, and the PyPI wheel is
> `publish = false` on crates.io and ships separately.

`ordvec` (the Rust crate) and `ordvec` on PyPI (the PyO3 wheel built from
`ordvec-python/`) are released by **manually dispatching** the release
workflows. Nothing ships on a tag push or a merge.

## Release pipeline controls

Both `release-crate.yml` and `release-python.yml`:

- are **`workflow_dispatch`-only** (no `push` / tag trigger);
- run a **`require-ci-green`** gate confirming `ci.yml` (and, for the wheel,
  `python.yml`) are green for the target commit on `main`;
- publish via **OIDC trusted publishing** (no long-lived crates.io / PyPI
  tokens in the repo);
- emit **SLSA build provenance** (`actions/attest-build-provenance`) **before**
  publishing — a failed attestation fails the release closed, so nothing ships
  without provenance recorded first;
- pin every third-party action by **commit SHA**, set
  `persist-credentials: false`, and default to `permissions: contents: read`.

`release-python.yml` additionally produces **PEP 740** attestations via the PyPI
Trusted Publishing step.

### Environment protection (configured in repo settings, not in code)

- **Required reviewer** — each environment (`crates-io`, `pypi`) requires
  maintainer (`Fieldnote-Echo`) approval before the publish job runs.
- **Deployment branch** — each environment is restricted to **`main`**, the
  only ref a release may be dispatched from. This makes "only `main` can
  publish" a configuration invariant rather than a manual check at approval
  time.

> These two settings are the supply-chain backstop the workflow code cannot
> express on its own (THREAT-SUPPLY-001 in [THREAT_MODEL.md](THREAT_MODEL.md)).

### Recommended (open)

- A **`v*` tag-protection ruleset** (block update + deletion) and a basic
  `main` ruleset, so a release tag cannot be force-moved and `main` cannot be
  force-pushed/deleted (THREAT-SUPPLY-002). Registries are already immutable
  (crates.io is yank-only; PyPI burns a version on delete), so this closes the
  remaining GitHub-side mutability surface.

## Checklist

1. Land everything on `main`; confirm the working tree and `Cargo.lock` are in
   sync (`cargo build --locked`).
2. Bump the version (crate `Cargo.toml`, and `ordvec-python` if the wheel
   changed) and update `CHANGELOG.md`. Commit on `main`.
3. Confirm CI is **green for that exact `main` SHA** (the dispatch ref must be
   `main` — the environment will refuse any other branch).
4. Get the maintainer's explicit go to publish.
5. Dispatch `release-crate.yml` (crate) and/or `release-python.yml` (wheel)
   from **`main`**.
6. Approve the environment deployment when prompted (required reviewer).
7. Verify the published artifact (crates.io / docs.rs / PyPI) and its
   provenance, and — for a coordinated release — the Zenodo deposit.

## Coordinated release note

The crate publish, the PyPI wheel, and the paper's Zenodo deposit are
coordinated (the paper consumes the bindings for a final cold-repro run). Do
not ship one leg in isolation without the maintainer's go.
