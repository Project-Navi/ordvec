# Releasing `ordvec`

> **Publish is held.** A real `cargo publish` / PyPI publish happens only on
> the maintainer's explicit approval. CI never publishes — the unified release
> pipeline builds, attests, and attaches everything to the GitHub Release
> automatically on a tag push, then **waits at the `crates-io` and `pypi`
> environment gates** for a required-reviewer approval before either registry
> push.

`ordvec` (the Rust crate) and `ordvec` on PyPI (the PyO3 wheel built from
`ordvec-python/`) are released by **pushing a `vMAJOR.MINOR.PATCH` tag** to a
commit on `main`. The release workflow handles build, attestation, SLSA
provenance, Release-asset attach, and un-draft automatically; only the two
registry pushes are manual.

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
- emits **GitHub SLSA build provenance** (`actions/attest-build-provenance`)
  and a **SLSA-generator `*.intoto.jsonl`** attached to the GitHub Release
  **before** the gated publishes — a failed attestation fails the release
  closed, so nothing ships without provenance recorded;
- stages the **`.crate`, wheels, sdist, `*.sigstore.json` bundle, and
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
Trusted Publishing (served from PyPI's Integrity API).

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
2. Bump the version (crate `Cargo.toml`, and `ordvec-python` if the wheel
   changed) and update `CHANGELOG.md`. Commit on `main`.
3. Confirm CI is **green for current `main` HEAD**. `require-ci-green` checks
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
4. Get the maintainer's explicit go to publish.
5. Push the version tag from `main` (signed):

   ```sh
   git tag -s vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

   `release.yml` triggers automatically. It builds the `.crate`, wheels, and
   sdist; attests them (GitHub attestation store + `*.sigstore.json`);
   generates the SLSA `*.intoto.jsonl`; and stages every artifact, the
   attestation bundle, and the provenance on the GitHub Release — **as a
   DRAFT**. It then pauses at the two registry environment gates.
6. **Approve the two publish environments** when they pause in the Actions UI
   (one for `crates-io`, one for `pypi`). The required-reviewer approval is
   what authorises the registry push.
   - `publish-crate` first sha256-compares its repackaged `.crate` to the
     SLSA-attested artifact — if they diverge (toolchain drift, etc.) the job
     fails closed BEFORE the OIDC token is minted, so nothing reaches
     crates.io. Re-run / investigate.
   - Once **both** publishes succeed, `publish-github-release` un-drafts the
     GitHub Release automatically. If one publish fails, the Release stays
     DRAFT — re-run the failed job, the un-draft then completes.
   - `publish-pypi` also queries PyPI after upload and compares every served
     wheel/sdist SHA-256 digest against the staged `dist/` files before the
     GitHub Release can un-draft.
7. Verify each published artifact and its provenance:
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
