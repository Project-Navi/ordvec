# Releasing `ordvec`

> **Publish is held.** A real crates.io / PyPI publish happens only after
> an eligible release approver explicitly approves the protected deployment.
> CI never publishes on its own — the unified release pipeline builds, attests,
> and attaches everything to the GitHub Release automatically on a tag push,
> then **waits at the `crates-io` and `pypi` environment gates** for the
> required-reviewer approval and 30-minute wait timer before any registry push.

`ordvec` (the Rust crate), `ordvec-manifest` (the lockstep manifest verifier
crate), `ordvec` on PyPI (the PyO3 wheel built from `ordvec-python/`), and
`ordvec-manifest` on PyPI (the PyO3 wheel built from
`ordvec-manifest-python/`) are released by **pushing a `vMAJOR.MINOR.PATCH` tag**
to current `main` HEAD. The release workflow handles build, canonical Python artifact selection,
attestation, SLSA provenance, Release-asset attach, and un-draft automatically;
only the registry environment approvals are manual.

## Release pipeline controls

The unified `release.yml`:

- triggers on **tag push** (`v[0-9]*.[0-9]*.[0-9]*`); a strict-SemVer guard
  step rejects pre-release / leading-zero / non-SemVer tags so they wake the
  workflow but skip every job below the gate;
- runs a **`require-ci-green`** gate confirming the tag points at current `main`
  HEAD and that per-commit CI is green on `main` for that SHA — `ci.yml`,
  `python.yml`, `fuzz.yml`, `codeql.yml`, `actionlint.yml`, `zizmor.yml`. The
  gate polls the latest exact-SHA push run every 30 seconds for at most 30
  minutes: missing, queued, and in-progress runs wait; a terminal non-success
  fails immediately; and a moved `main` HEAD fails immediately. Every draft-
  release and build-artifact job depends directly on this gate, so an unsettled
  or failed gate creates no release artifacts and no draft GitHub Release;
- publishes via **OIDC trusted publishing** (no long-lived crates.io / PyPI
  tokens in the repo) for both Rust crates and both Python distributions;
- canonicalizes each Python dist before attestation and release upload: for a
  new PyPI version it uses the current run's wheels/sdist; if PyPI already owns
  that immutable version during recovery, it downloads the exact PyPI-served
  files, verifies their SHA-256 digests from PyPI JSON, and uses those bytes as
  the GitHub Release assets;
- emits **GitHub SLSA build provenance** (`actions/attest-build-provenance`)
  and **SLSA-generator `*.intoto.jsonl`** assets attached to the GitHub Release
  **before** each gated publish — a failed attestation fails the release
  closed, so nothing ships without provenance recorded. In recovery mode where
  PyPI files already exist, the initial GitHub/SLSA subjects are deliberately
  limited to the Rust crate built by the current run; the Python files are
  verified immutable PyPI bytes from the earlier Trusted Publishing upload, not
  falsely claimed as rebuilt by the recovery run;
- stages the core **`.crate` file, canonical wheels, canonical sdist,
  `*.sigstore.json` bundle, and `*.intoto.jsonl` provenance** on the GitHub Release while it is still **a
  DRAFT** (`release-assets-draft` owns the core/Python Release uploads, and
  `release-manifest-assets-draft` later owns the manifest crate uploads; no
  manual attach, which is what v0.2.0's manual step missed);
- proves **byte-identity** in `publish-crate` and `publish-manifest-crate` on
  both sides of `cargo publish`:
    1. **pre-publish gate** — downloads the SLSA-attested `.crate` artifact,
       re-packages with `--locked`, and `sha256`-compares before minting the
       crates.io OIDC token. Defends against toolchain drift / deterministic-
       packaging regression; if they differ, fails closed **before** the
       token is minted (nothing reaches crates.io);
    2. **post-publish empirical proof** — downloads the just-published `.crate`
       from `crates.io/api/v1/crates/<crate>/<v>/download` and
       `sha256`-compares to the attested artifact. `cargo publish` runs its own
       internal packaging step the pre-publish gate cannot inspect; this is the
       only check that proves the bytes crates.io actually serves equal the
       SLSA-attested bytes. A mismatch fails closed, so
       `publish-github-release` never un-drafts the Release (the version is
       then yank-only on crates.io, but the failure is loudly observable);
- publishes `ordvec-manifest` only after the lockstep `ordvec` crate has
  published and passed its crates.io-served byte-identity readback. Cargo cannot
  package a fresh lockstep manifest version until the matching core crate exists
  on crates.io, so the workflow builds, attests, generates SLSA provenance for,
  and stages the manifest `.crate` on the draft GitHub Release after
  `publish-crate` succeeds; `publish-manifest-crate` then re-runs
  `cargo package -p ordvec-manifest --locked` and byte-compares that output to
  the attested artifact before minting its own OIDC token;
- **un-drafts the GitHub Release ONLY after `publish-crate`,
  `publish-manifest-crate`, `publish-pypi`, AND `publish-manifest-pypi` succeed**
  (`publish-github-release` is the sole un-draft point). If any publish fails
  or is skipped, the Release stays DRAFT — no public Release ever exists for a
  version the registries refused;
- pins every third-party action by **commit SHA** (the one mandated exception
  is the SLSA reusable workflow, tag-pinned per SLSA's trust model), sets
  `persist-credentials: false`, and defaults to `permissions: contents: read`.

The PyPI publish step additionally produces **PEP 740** attestations via
Trusted Publishing (served from PyPI's Integrity API) on a fresh upload. If the
version already exists on PyPI during recovery, the job skips upload and instead
verifies that PyPI-served wheel/sdist hashes match the canonical files staged on
the GitHub Release.

### Environment protection (configured in repo settings, not in code)

- **Required reviewers and self-review prevention** — each environment
  (`crates-io`, `pypi`) lists `Fieldnote-Echo` and `toadkicker` as required
  reviewers and has **prevent self-review** enabled. GitHub still requires one
  approving reviewer per deployment, but the account that triggered the
  deployment cannot approve it; a second listed release approver must clear the
  publish gate.
- **Wait timer** — each environment has a **30-minute wait timer**. No registry
  OIDC credential is minted until both the wait timer and the required-reviewer
  approval have passed.
- **Deployment branches and tags** — each environment's "Deployment branches
  and tags" policy is set to **Selected branches and tags** with a single
  **tag pattern**: **`v[0-9]*.[0-9]*.[0-9]*`** (matching the workflow's
  trigger glob). The release workflow runs on `refs/tags/vX.Y.Z`, NOT
  `refs/heads/main`, so a **branch-only** allowlist (the old setting under the
  dispatch model) would deadlock the publish — the environment would refuse
  every tag-triggered run. The "tag must point at a commit on `main`"
  guarantee is preserved by **`require-ci-green`**, which only passes if a
  successful push-event CI run exists for the exact SHA on `main` — a SHA
  that exists only via a PR merge to the protected branch. Optionally, a
  **tag ruleset** (Settings → Rules → Rulesets → New tag ruleset) can be added
  to restrict tag *creation* to refs on `main` as defence in depth.

> These environment settings are the supply-chain backstop the workflow code
> cannot express on its own (THREAT-SUPPLY-001 in
> [THREAT_MODEL.md](THREAT_MODEL.md)).

### Trusted-publisher configuration (one-time, in the registries)

The crates.io and PyPI Trusted Publisher records must point at this workflow
filename and GitHub repository identity. After the GitHub repo transfer, the
registry-side owner must be `Project-Navi` and the repository must be `ordvec`.
Until a record is updated, the corresponding gated publish fails **closed** at
the OIDC exchange (no risk of a bad publish; just a failed run).

- **crates.io** → `ordvec` → Settings → Trusted Publishing → GitHub publisher:
  `owner = Project-Navi`, `repository = ordvec`, `workflow = release.yml`,
  `environment = crates-io`.
- **crates.io** → `ordvec-manifest` → Settings → Trusted Publishing → GitHub
  publisher: `owner = Project-Navi`, `repository = ordvec`,
  `workflow = release.yml`, `environment = crates-io`. If crates.io requires an
  initial owner bootstrap before a new crate's Trusted Publisher can be
  configured, do that explicit maintainer-approved bootstrap before tagging.
- **PyPI** → `ordvec` → Publishing → GitHub publisher: `workflow = release.yml`,
  `owner = Project-Navi`, `repository = ordvec`, `environment = pypi`, project
  URL `https://pypi.org/p/ordvec`.
- **PyPI** → `ordvec-manifest` → Publishing → GitHub publisher:
  `workflow = release.yml`, `owner = Project-Navi`, `repository = ordvec`,
  `environment = pypi`, project URL `https://pypi.org/p/ordvec-manifest`.

### Tag and branch protection

- **Immutable releases** is enabled, so a published release's `v*` tag cannot
  be force-moved or deleted and its assets cannot be replaced after
  publication. This closes the GitHub-side mutability surface the registries
  already close on their end (crates.io is yank-only; PyPI burns a version on
  delete).
- **`main` is a protected branch** — pull-request review is required and
  force-pushes and deletions are blocked, so the branch a release tag points
  to cannot be rewritten (THREAT-SUPPLY-002).

## Checklist

1. Land everything on `main`; confirm the working tree and `Cargo.lock` are in
   sync (`cargo build --locked`).
2. Review the compatibility impact against
   [`docs/compatibility-policy.md`](docs/compatibility-policy.md):
   - classify the release as patch-compatible or minor-breaking;
   - identify touched stable Rust, Python, C ABI, Go, Manifest,
     persisted-format, examples/docs, feature, and MSRV surfaces;
   - for patch releases, run a SemVer compatibility check against the latest
     published crate when practical, or record why an equivalent check is not
     useful for this release;
   - distinguish `ordvec` primitive API/file compatibility from downstream
     application database behavior.
3. Bump the lockstep version (`Cargo.toml`,
   `ordvec-manifest/Cargo.toml` including its `ordvec` dependency,
   `ordvec-python/Cargo.toml`, `ordvec-python/pyproject.toml`,
   `ordvec-python/python/ordvec/__init__.py`,
   `ordvec-manifest-python/Cargo.toml`,
   `ordvec-manifest-python/pyproject.toml`,
   `ordvec-manifest-python/python/ordvec_manifest/__init__.py`, and
   `ordvec-ffi/Cargo.toml`). Also bump every internal path-dependency version:
   `ordvec-manifest` → `ordvec`, both Python binding aliases, `ordvec-ffi` →
   `ordvec`, and `benchmarks/beir-bench` → `ordvec`. Update `CHANGELOG.md`
   with migration notes for every intentional compatibility break. Commit on
   `main`.
   - Run `python tests/release_publish_invariants.py` after the bump; it checks
     lockstep versions, MSRV/docs drift, registry metadata parity, Python
     classifier/URL parity, docs.rs feature policy, package contents, and
     release workflow invariants.
   - **Downstream un-patch (one-time, 0.6.0):** OrdinalDB's root workspace and
     workspace-excluded standalone consumers carry `[patch.crates-io]` blocks
     for pre-release `ordvec` and `ordvec-manifest` git revisions. When 0.6.0
     publishes, remove every one of those overrides, regenerate each
     independent lockfile, and prove that OrdinalDB consumes the published
     crates.io releases instead of a pre-release git revision.
4. Confirm CI is **green for current `main` HEAD**. `require-ci-green` checks
   `main` HEAD's SHA — which needs a **completed, successful** (not
   `cancelled`) latest run of `ci.yml`, `python.yml`, `fuzz.yml`, `codeql.yml`,
   `actionlint.yml`, and `zizmor.yml`. A just-pushed tag may reach the release
   workflow before Actions exposes or completes those runs, so the gate waits
   up to 30 minutes and polls every 30 seconds. It fails immediately on a
   terminal non-success or if `main` advances, and fails closed on timeout.
   - Routine `ci.yml` / `coverage.yml` runs may warn and skip SDE-dependent
     steps when Intel's downloadmirror challenges GitHub-hosted runners. That
     keeps external mirror outages from holding `main` red, but it does **not**
     make a release shippable by itself: `release.yml` has a fail-closed
     `release-avx512` job that installs Intel SDE, runs the AVX-512 CPUID
     probe, and runs the AVX-512 test lane before assets can be staged.
     This release proof deliberately avoids writable workflow caches in the
     tag workflow; if Intel's download path is unavailable, wait, rerun, or land
     a reviewed SDE pin/update before tagging.
   - Before the final tag, spot-check `.github/actions/setup-intel-sde/action.yml`
     against Intel's SDE download page: version, Linux archive name, and SHA-256
     must match the currently accepted pin.
   - **Do not merge another PR between the release commit and the tag push.**
     `ci.yml` / `python.yml` use `cancel-in-progress`, so merging again moves
     `main` HEAD and cancels the previous commit's in-flight CI. The
     superseded commit is no longer the release target: **tag from the new
     HEAD once its own CI has completed green** — never from, or by
     re-validating, the older commit.
   - If HEAD's *own* run shows `cancelled` (superseded, but you have since
     stopped pushing), re-run **that HEAD run** from the Actions UI and wait
     for it to finish green before tagging. The SHA you re-run must be the
     exact SHA you publish; do not hand-clear the gate on any other commit.
   - Release only from a commit on `main` with a **successful push-to-main
     run** of each gated workflow — in practice the tip the merge produced (a
     squash commit, a rebased tip, or a merge commit), whatever the merge
     strategy. An interior commit that exists in history only from a PR branch
     has no push-to-main run (its CI ran as a `pull_request` on the branch)
     and so is not releasable.
5. Run the manual release-settings audit before creating the tag:

   ```sh
   bash tests/release_environment_settings.sh
   ```

   This verifies the GitHub Environments still require the expected reviewers,
   prevent self-review, apply the 30-minute wait timer, and accept only the
   stable release tag pattern. Separately verify the registry Trusted Publisher
   records by hand: crates.io must point both `ordvec` and `ordvec-manifest` to
   `release.yml` / `crates-io`, and PyPI must point both `ordvec` and
   `ordvec-manifest` to `release.yml` / `pypi`.
6. Get explicit maintainer agreement to publish.
7. Push the version tag from `main` (signed):

   ```sh
   git tag -s vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

   `release.yml` triggers automatically. It builds the core `.crate`, wheels,
   and sdist for both Python packages; selects the canonical Python dists
   (current build for a new PyPI version, verified PyPI bytes for an existing
   immutable version); attests the files this run can honestly attest (GitHub
   attestation store +
   `*.sigstore.json`); generates SLSA `*.intoto.jsonl`; and stages the core and
   Python assets on the GitHub Release — **as a DRAFT**. After `publish-crate`
   succeeds, it builds, attests, generates SLSA provenance for, and stages the
   lockstep `ordvec-manifest` `.crate`, then pauses at the manifest registry
   environment gate.
8. **Approve each publish environment pause** in the Actions UI. There are
   four registry publish jobs: `publish-crate`, `publish-manifest-crate`,
   `publish-pypi`, and `publish-manifest-pypi`. The two crates.io jobs use the
   same `crates-io` environment and may require separate approvals; the two PyPI
   jobs use the `pypi` environment and may also require separate approvals.
   Each job must also clear the 30-minute wait timer. Because self-review is
   blocked, the account that triggered the deployment cannot approve it; use the
   other listed release approver in the Actions UI. Required-reviewer approval is
   what authorises each registry push.
   - `publish-crate` and `publish-manifest-crate` first sha256-compare their
     repackaged `.crate` to the SLSA-attested artifact — if either diverges
     (toolchain drift, etc.) the job fails closed BEFORE the OIDC token is
     minted, so nothing reaches crates.io. Re-run / investigate.
   - Once **all** registry publish jobs succeed, `publish-github-release`
     un-drafts the GitHub Release automatically. If one gate fails, the Release
     stays DRAFT — investigate and re-run from a fixed workflow rather than
     approving another registry into a partial state.
   - `publish-pypi` and `publish-manifest-pypi` either upload their fresh
     canonical dist or, if PyPI already serves that version, skip upload and
     verify the existing files. In both modes they compare every PyPI-served
     wheel/sdist SHA-256 digest against the canonical `dist/` files before the
     GitHub Release can un-draft.
9. Verify each published artifact and its provenance:
   - crates.io / docs.rs for `ordvec` and `ordvec-manifest`;
   - PyPI (confirm the post-publish hash-verification log, optionally
     `pip download ordvec==X.Y.Z` and inspect, plus check the PEP 740 attestation
     at `GET https://pypi.org/integrity/ordvec/X.Y.Z/<file>/provenance`);
   - the GitHub Release page (`.crate`, wheels, sdist, `*.sigstore.json`,
     `*.intoto.jsonl` all present);
   - `gh attestation verify <file> -R Project-Navi/ordvec` on a downloaded
     artifact;
   - compare the observed release assets against
     [`docs/artifact-platform-matrix.md`](docs/artifact-platform-matrix.md);
   - for a coordinated release, the Zenodo deposit.

## Coordinated release note

The Rust crate publishes, the PyPI wheel, and the paper's Zenodo deposit are
coordinated (the paper consumes the bindings for a final cold-repro run). Do
not ship one leg in isolation without the maintainer's go.
