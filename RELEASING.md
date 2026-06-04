# Releasing `ordvec`

> **Publish is held.** A real `cargo publish` / PyPI publish happens only on
> the maintainer's explicit approval. CI never publishes — the unified release
> pipeline builds, attests, and attaches everything to the GitHub Release
> automatically on a tag push, then **waits at the `crates-io` and `pypi`
> environment gates** for a required-reviewer approval before either registry
> push.

`ordvec` (the Rust crate) and `ordvec` on PyPI (the PyO3 wheel built from
`ordvec-python/`) are released by **pushing a `vMAJOR.MINOR.PATCH` tag** to a
commit on `main`. The release workflow handles build, canonical Python artifact
selection, attestation, SLSA provenance, Release-asset attach, and un-draft
automatically; only the two registry gates are manual.

## Release pipeline controls

The unified `release.yml`:

- triggers on **tag push** (`v[0-9]*.[0-9]*.[0-9]*`); a strict-SemVer guard
  step rejects pre-release / leading-zero / non-SemVer tags so they wake the
  workflow but skip every job below the gate;
- runs a **`require-ci-green`** gate confirming the per-commit CI is green on
  `main` for the tagged SHA — `ci.yml`, `python.yml`, `fuzz.yml`, `codeql.yml`
  (a *successful* run for that exact SHA on `main`);
- publishes via **OIDC trusted publishing** (no long-lived crates.io / PyPI
  tokens in the repo);
- canonicalizes the Python dist before attestation and release upload: for a
  new PyPI version it uses the current run's wheels/sdist; if PyPI already owns
  that immutable version during recovery, it downloads the exact PyPI-served
  files, verifies their SHA-256 digests from PyPI JSON, and uses those bytes as
  the GitHub Release assets;
- emits **GitHub SLSA build provenance** (`actions/attest-build-provenance`)
  and a **SLSA-generator `*.intoto.jsonl`** attached to the GitHub Release
  **before** the gated publishes — a failed attestation fails the release
  closed, so nothing ships without provenance recorded. In recovery mode where
  PyPI files already exist, the GitHub/SLSA subjects are deliberately limited
  to the crate built by the current run; the Python files are verified immutable
  PyPI bytes from the earlier Trusted Publishing upload, not falsely claimed as
  rebuilt by the recovery run;
- stages the **`.crate`, canonical wheels, canonical sdist, `*.sigstore.json` bundle, and
  `*.intoto.jsonl` provenance** on the GitHub Release while it is still **a
  DRAFT** (`release-assets-draft` is the sole Release-asset writer — no manual
  attach, which is what v0.2.0's manual step missed);
- proves **byte-identity** in `publish-crate` on both sides of `cargo publish`:
    1. **pre-publish gate** — downloads the SLSA-attested `.crate` artifact,
       re-packages with `--locked`, and `sha256`-compares before minting the
       crates.io OIDC token. Defends against toolchain drift / deterministic-
       packaging regression; if they differ, fails closed **before** the
       token is minted (nothing reaches crates.io);
    2. **post-publish empirical proof** — downloads the just-published `.crate`
       from `crates.io/api/v1/crates/ordvec/<v>/download` and `sha256`-compares
       to the attested artifact. `cargo publish` runs its own internal
       packaging step the pre-publish gate cannot inspect; this is the only
       check that proves the bytes crates.io actually serves equal the SLSA-
       attested bytes. A mismatch fails closed, so `publish-github-release`
       never un-drafts the Release (the version is then yank-only on
       crates.io, but the failure is loudly observable);
- **un-drafts the GitHub Release ONLY after BOTH `publish-crate` AND
  `publish-pypi` succeed** (`publish-github-release` is the sole un-draft
  point). If either publish fails or is skipped, the Release stays DRAFT — no
  public Release ever exists for a version the registries refused;
- pins every third-party action by **commit SHA** (the one mandated exception
  is the SLSA reusable workflow, tag-pinned per SLSA's trust model), sets
  `persist-credentials: false`, and defaults to `permissions: contents: read`.

The PyPI publish step additionally produces **PEP 740** attestations via
Trusted Publishing (served from PyPI's Integrity API) on a fresh upload. If the
version already exists on PyPI during recovery, the job skips upload and instead
verifies that PyPI-served wheel/sdist hashes match the canonical files staged on
the GitHub Release.

### Environment protection (configured in repo settings, not in code)

- **Required reviewer** — each environment (`crates-io`, `pypi`) requires
  maintainer (`Fieldnote-Echo`) approval before its publish job runs.
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

> These two settings are the supply-chain backstop the workflow code cannot
> express on its own (THREAT-SUPPLY-001 in [THREAT_MODEL.md](THREAT_MODEL.md)).

### Trusted-publisher configuration (one-time, in the registries)

The crates.io and PyPI Trusted Publisher records must point at this workflow
filename. Until either is updated, the corresponding gated publish fails
**closed** at the OIDC exchange (no risk of a bad publish; just a failed run).

- **crates.io** → `ordvec` → Settings → Trusted Publishing → GitHub publisher:
  `workflow = release.yml`, `environment = crates-io`.
- **PyPI** → `ordvec` → Publishing → GitHub publisher: `workflow = release.yml`,
  `environment = pypi`.

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
3. Bump the version (crate `Cargo.toml`, `ordvec-manifest/Cargo.toml`, and
   `ordvec-python` if the wheel changed) and update `CHANGELOG.md` with
   migration notes for every intentional compatibility break. Commit on
   `main`.
4. Confirm CI is **green for current `main` HEAD**. `require-ci-green` checks
   `main` HEAD's SHA — which needs a **completed, successful** (not
   `cancelled`, not in-progress) run of `ci.yml`, `python.yml`, `fuzz.yml`, and
   `codeql.yml`.
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

   This verifies the GitHub Environments still require the expected reviewer
   and accept only the stable release tag pattern. Separately verify the
   registry Trusted Publisher records by hand: crates.io must point to
   `release.yml` / `crates-io`, and PyPI must point to `release.yml` / `pypi`.
6. Get the maintainer's explicit go to publish.
7. Push the version tag from `main` (signed):

   ```sh
   git tag -s vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

   `release.yml` triggers automatically. It builds the `.crate`, wheels, and
   sdist; selects the canonical Python dist (current build for a new PyPI
   version, verified PyPI bytes for an existing immutable version); attests the
   files this run can honestly attest (GitHub attestation store +
   `*.sigstore.json`); generates the SLSA `*.intoto.jsonl`; and stages every
   artifact, the attestation bundle, and the provenance on the GitHub Release
   — **as a DRAFT**. It then pauses at the two registry environment gates.
8. **Approve the two publish environments** when they pause in the Actions UI
   (one for `crates-io`, one for `pypi`). The required-reviewer approval is
   what authorises the registry push.
   - `publish-crate` first sha256-compares its repackaged `.crate` to the
     SLSA-attested artifact — if they diverge (toolchain drift, etc.) the job
     fails closed BEFORE the OIDC token is minted, so nothing reaches
     crates.io. Re-run / investigate.
   - Once **both** registry gates succeed, `publish-github-release` un-drafts
     the GitHub Release automatically. If one gate fails, the Release stays
     DRAFT — investigate and re-run from a fixed workflow rather than approving
     the other registry into another partial state.
   - `publish-pypi` either uploads the fresh canonical dist or, if PyPI already
     serves that version, skips upload and verifies the existing files. In both
     modes it compares every PyPI-served wheel/sdist SHA-256 digest against the
     canonical `dist/` files before the GitHub Release can un-draft.
9. Verify each published artifact and its provenance:
   - crates.io / docs.rs;
   - PyPI (confirm the post-publish hash-verification log, optionally
     `pip download ordvec==X.Y.Z` and inspect, plus check the PEP 740 attestation
     at `GET https://pypi.org/integrity/ordvec/X.Y.Z/<file>/provenance`);
   - the GitHub Release page (`.crate`, wheels, sdist, `*.sigstore.json`,
     `*.intoto.jsonl` all present);
   - `gh attestation verify <file> -R Fieldnote-Echo/ordvec` on a downloaded
     artifact;
   - for a coordinated release, the Zenodo deposit.

## Coordinated release note

The crate publish, the PyPI wheel, and the paper's Zenodo deposit are
coordinated (the paper consumes the bindings for a final cold-repro run). Do
not ship one leg in isolation without the maintainer's go.
