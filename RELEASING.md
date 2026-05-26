# Releasing `ordvec`

> **Publish is held.** A real `cargo publish` / PyPI publish happens only
> on the maintainer's explicit go. CI never publishes for real — the crate job
> runs `cargo publish -p ordvec --dry-run --locked`, and the PyPI wheel is
> `publish = false` on crates.io and ships separately.

`ordvec` (the Rust crate) and `ordvec` on PyPI (the PyO3 wheel built from
`ordvec-python/`) are released by **manually dispatching** the release
workflows. Nothing ships on a tag push or a merge.

## Release pipeline controls

Both `release-crate.yml` and `release-python.yml`:

- are **`workflow_dispatch`-only** (no `push` / tag trigger);
- run a **`require-ci-green`** gate confirming the per-commit CI is green for the
  target commit on `main` — `ci.yml`, `fuzz.yml`, and `codeql.yml` for the crate,
  plus `python.yml` for the wheel (a *successful* run for that exact SHA on `main`);
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

### Tag and branch protection

- **Immutable releases** is enabled, so a published release's `v*` tag cannot be
  force-moved or deleted and its assets cannot be replaced after publication.
  This closes the GitHub-side mutability surface the registries already close on
  their end (crates.io is yank-only; PyPI burns a version on delete).
- **`main` is a protected branch** — pull-request review is required and
  force-pushes and deletions are blocked, so the branch a release dispatches
  from cannot be rewritten (THREAT-SUPPLY-002).

## Checklist

1. Land everything on `main`; confirm the working tree and `Cargo.lock` are in
   sync (`cargo build --locked`).
2. Bump the version (crate `Cargo.toml`, and `ordvec-python` if the wheel
   changed) and update `CHANGELOG.md`. Commit on `main`.
3. Confirm CI is **green for current `main` HEAD**. A release dispatches from
   `main` (the environment refuses any other ref), so `require-ci-green` always
   checks `main` HEAD's SHA — which needs a **completed, successful** (not
   cancelled, not in-progress) run of `ci.yml`, `fuzz.yml`, `codeql.yml` (and
   `python.yml` for the wheel).
   - **Do not merge another PR between the release commit and the dispatch.**
     `ci.yml` / `python.yml` use `cancel-in-progress`, so merging again moves
     `main` HEAD and cancels the previous commit's in-flight CI. The superseded
     commit is no longer the release target: **release from the new HEAD once its
     own CI has completed green** — never from, or by re-validating, the older
     commit.
   - If HEAD's *own* run shows `cancelled` (superseded, but you have since
     stopped pushing), re-run **that HEAD run** from the Actions UI and wait for
     it to finish green before dispatching. The SHA you re-run must be the exact
     SHA you publish; do not hand-clear the gate on any other commit.
   - Release only from a commit on `main` with a **successful push-to-main run**
     of each gated workflow — in practice the tip the merge produced (a squash
     commit, a rebased tip, or a merge commit), whatever the merge strategy. An
     interior commit that exists in history only from a PR branch has no
     push-to-main run (its CI ran as a `pull_request` on the branch) and so is
     not releasable.
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
