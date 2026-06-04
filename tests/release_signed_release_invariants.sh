#!/usr/bin/env bash
#
# Signed-release / provenance invariants — pinned in CI.
#
# release.yml's signed-release graph attaches the .intoto.jsonl and Sigstore
# assets that OpenSSF Scorecard detects for Signed-Releases, while older
# unsigned releases may keep the score below 10 temporarily. The same graph
# keeps the build-attest-publish chain honest:
#
#     build-{crate,manifest-crate,wheels,sdist}   (raw artifacts)
#         |
#         +-> pypi-canonical-dist (current build, or verified immutable PyPI files)
#         |
#         +-> attest         (id-token + attestations + .sigstore.json;
#         |                   Rust-crates-only when PyPI files already exist)
#         +-> provenance     (slsa-github-generator @vX.Y.Z, .intoto.jsonl;
#         |                   Rust-crates-only when PyPI files already exist)
#         |
#         v
#     release-assets-draft   (uploads .crate/canonical .whl/.tar.gz/.sigstore.json/.intoto.jsonl to DRAFT release)
#         |
#         +--> publish-crate          (byte-identity check vs attested .crate, then cargo publish)
#         +--> publish-manifest-crate (after publish-crate; same byte-identity proof)
#         +--> publish-pypi           (Trusted Publishing, or existing-file verification)
#               |
#               v
#         publish-github-release (un-draft, ONLY after all registry publishes succeed)
#
# A regression in any of these edges (a future commit drops a needs:, renames
# the provenance file, lets release-assets-draft un-draft itself, forgets the
# byte-identity check, or moves the un-draft before the publishes) silently
# re-creates the v0.2.0 failure mode or weakens the chain. This script pins
# the graph so the regression fails on every push/PR, not at the next real
# release.
#
# It is intentionally a structural lint on release.yml (greps the YAML), not a
# runtime exercise of the pipeline — that's the fork dry-run's job.
set -euo pipefail
fail() { echo "::error::signed-release invariant violated: $*"; exit 1; }

wf=".github/workflows/release.yml"
[ -f "$wf" ] || fail "$wf: workflow file not found"

# Extract the body of a job (from `  <name>:` to the next 2-space-indented job key).
job_body() {
  local jobname="$1" start end
  start="$(grep -nE "^  ${jobname}:[[:space:]]*$" "$wf" | head -1 | cut -d: -f1)"
  [ -n "$start" ] || fail "$wf: no '${jobname}:' job found"
  end="$(awk -v s="$start" 'NR>s && /^  [A-Za-z0-9_-]+:/ {print NR-1; exit}' "$wf")"
  [ -n "$end" ] || end="$(awk 'END{print NR}' "$wf")"
  sed -n "${start},${end}p" "$wf"
}

# Accept both `needs: [a, b, c]` (inline) and `needs:\n  - a\n  - b` (block) forms.
job_needs() {
  local jobname="$1" needed="$2"
  local body
  body="$(job_body "$jobname")"
  printf '%s\n' "$body" | grep -qE "(^[[:space:]]+needs:.*\\b${needed}\\b|^[[:space:]]+-[[:space:]]+${needed}[[:space:]]*$)"
}

job_line() {
  local jobname="$1" pattern="$2"
  job_body "$jobname" | grep -nE "$pattern" | head -1 | cut -d: -f1
}

require_job_line() {
  local jobname="$1" pattern="$2" description="$3" line
  line="$(job_line "$jobname" "$pattern")"
  [ -n "$line" ] || fail "$jobname must contain $description"
  printf '%s\n' "$line"
}

job_downloads_artifact_to_path() {
  local jobname="$1" artifact="$2" expected_path="$3"
  job_body "$jobname" | awk -v artifact="$artifact" -v expected_path="$expected_path" '
    function flush_step() {
      if (has_download && has_name && has_path) {
        found = 1
      }
      has_download = has_name = has_path = 0
    }

    /^[[:space:]]+-[[:space:]]/ { flush_step() }
    $0 ~ "uses:[[:space:]]*actions/download-artifact" { has_download = 1 }
    $0 ~ "^[[:space:]]+name:[[:space:]]*" artifact "[[:space:]]*$" { has_name = 1 }
    $0 ~ "^[[:space:]]+path:[[:space:]]*" expected_path "[[:space:]]*$" { has_path = 1 }
    END { flush_step(); exit found ? 0 : 1 }
  '
}

# ----------------------------------------------------------------------
# (1) release-assets-draft needs attest + provenance + require-ci-green + notes
#     + exact linux/aarch64 wheel smoke
# ----------------------------------------------------------------------
for dep in attest provenance pypi-canonical-dist require-ci-green notes smoke-linux-aarch64-wheel; do
  job_needs release-assets-draft "$dep" \
    || fail "release-assets-draft must \`needs: $dep\` (fail-closed on missing provenance/CI)"
done

# ----------------------------------------------------------------------
# (2) release-assets-draft uploads every required asset class to the Release
# ----------------------------------------------------------------------
body_draft="$(job_body release-assets-draft)"
github_repo_env_re='^[[:space:]]+GH_REPO:[[:space:]]*"?\$\{\{[[:space:]]*github\.repository[[:space:]]*\}\}"?[[:space:]]*$'
for ext in '\.crate' '\.whl' '\.tar\.gz' '\.sigstore\.json' '\.intoto\.jsonl'; do
  printf '%s\n' "$body_draft" | grep -qE "dist/\*${ext}([^a-zA-Z]|$)" \
    || fail "release-assets-draft must \`gh release upload\` dist/*$(printf '%s' "$ext" | sed 's/\\//g')"
done
printf '%s\n' "$body_draft" | grep -qE 'name:[[:space:]]*pypi-canonical-dist' \
  || fail "release-assets-draft must upload canonical Python dist, not raw rebuilt wheel/sdist artifacts"
job_downloads_artifact_to_path release-assets-draft dist-crate dist \
  || fail "release-assets-draft must download the core dist-crate artifact into dist"
job_downloads_artifact_to_path release-assets-draft dist-manifest-crate dist \
  || fail "release-assets-draft must download the manifest dist-manifest-crate artifact into dist"
printf '%s\n' "$body_draft" | grep -qE "$github_repo_env_re" \
  || fail "release-assets-draft must set \`GH_REPO: \${{ github.repository }}\` (no checkout, so gh release upload needs explicit repo context)"

# ----------------------------------------------------------------------
# (3) release-assets-draft must NOT un-draft (the dedicated un-draft job owns
#     that; un-drafting here would re-introduce the public-release-before-
#     publish failure mode).
# ----------------------------------------------------------------------
if printf '%s\n' "$body_draft" | grep -qE 'gh release edit.*--draft=false'; then
  fail "release-assets-draft must NOT un-draft the Release (un-drafting belongs in publish-github-release, after all registry publishes succeed)"
fi

# ----------------------------------------------------------------------
# (4) provenance uses slsa-github-generator pinned to a SEMANTIC VERSION TAG
#     (NOT a SHA — SLSA trust model requires the tag for its self-verification)
# ----------------------------------------------------------------------
prov="$(job_body provenance)"
printf '%s\n' "$prov" | grep -qE 'uses:[[:space:]]*slsa-framework/slsa-github-generator/.+/generator_generic_slsa3\.yml@v[0-9]+\.[0-9]+\.[0-9]+' \
  || fail "provenance must \`uses: slsa-framework/slsa-github-generator/.../generator_generic_slsa3.yml@vX.Y.Z\` (tag-pinned per SLSA trust model)"

# ----------------------------------------------------------------------
# (5) provenance must have `upload-assets: false` — release-assets-draft is
#     the sole Release-asset writer; two concurrent writers would race.
# ----------------------------------------------------------------------
printf '%s\n' "$prov" | grep -qE '^[[:space:]]+upload-assets:[[:space:]]*false[[:space:]]*$' \
  || fail "provenance must set \`upload-assets: false\` (single Release-asset writer is release-assets-draft; the .intoto.jsonl flows through the workflow-artifact path)"

# ----------------------------------------------------------------------
# (6) provenance-name MUST end in `.intoto.jsonl` — Scorecard's provenance
#     probe is a pure filename-suffix match.
# ----------------------------------------------------------------------
printf '%s\n' "$prov" | grep -qE '^[[:space:]]+provenance-name:.*\.intoto\.jsonl[[:space:]]*$' \
  || fail "provenance must set \`provenance-name: <name>.intoto.jsonl\` (Scorecard Signed-Releases provenance probe matches this suffix only)"

# ----------------------------------------------------------------------
# (7) attest job grants id-token: write + attestations: write
# ----------------------------------------------------------------------
att="$(job_body attest)"
printf '%s\n' "$att" | grep -qE '^[[:space:]]+id-token:[[:space:]]*write' \
  || fail "attest job must grant \`id-token: write\` (Sigstore OIDC signing cert)"
printf '%s\n' "$att" | grep -qE '^[[:space:]]+attestations:[[:space:]]*write' \
  || fail "attest job must grant \`attestations: write\` (persist to the GitHub attestation store)"
job_needs attest build-manifest-crate \
  || fail "attest must \`needs: build-manifest-crate\` so the manifest .crate is an attestation subject"
job_downloads_artifact_to_path attest dist-manifest-crate dist \
  || fail "attest must download the dist-manifest-crate artifact into dist"

comb="$(job_body combine-hashes)"
job_needs combine-hashes build-manifest-crate \
  || fail "combine-hashes must \`needs: build-manifest-crate\` so the manifest .crate is a SLSA subject"
job_downloads_artifact_to_path combine-hashes dist-manifest-crate dist \
  || fail "combine-hashes must download the dist-manifest-crate artifact into dist"

build_manifest="$(job_body build-manifest-crate)"
printf '%s\n' "$build_manifest" | grep -qE 'cargo[[:space:]]+package[[:space:]]+-p[[:space:]]+ordvec-manifest[[:space:]]+--locked[[:space:]]+--no-verify' \
  || fail "build-manifest-crate must package with --no-verify before the lockstep core crate exists on crates.io"

# ----------------------------------------------------------------------
# (8) Registry publish jobs grant id-token: write AND need release-assets-draft.
# ----------------------------------------------------------------------
for pub in publish-crate publish-manifest-crate publish-pypi; do
  body="$(job_body "$pub")"
  printf '%s\n' "$body" | grep -qE '^[[:space:]]+id-token:[[:space:]]*write' \
    || fail "$pub must grant \`id-token: write\` (Trusted Publishing OIDC)"
  job_needs "$pub" release-assets-draft \
    || fail "$pub must \`needs: release-assets-draft\` (gated by attest + provenance via the draft-assets edge)"
done

# ----------------------------------------------------------------------
# (9) Rust crate publish jobs prove byte-identity vs the attested .crate on BOTH
#     sides of `cargo publish`:
#       (9a) pre-publish: download the attested .crate, re-run `cargo package`,
#            sha256-compare. Fail-closed BEFORE the OIDC token is minted.
#       (9b) post-publish: download the just-published .crate from crates.io
#            and sha256-compare to the attested artifact. cargo publish runs
#            its own internal packaging step that the pre-publish gate
#            cannot inspect — this is the empirical proof that the bytes
#            actually served by crates.io match the SLSA-attested artifact.
# ----------------------------------------------------------------------
check_crate_publish_job() {
  local jobname="$1" package="$2" artifact="$3" body pre_line oidc_line publish_line post_line
  body="$(job_body "$jobname")"
  printf '%s\n' "$body" | grep -qE 'uses:[[:space:]]*actions/download-artifact' \
    || fail "$jobname must download the attested $artifact artifact (byte-identity gate)"
  printf '%s\n' "$body" | grep -qE "name:[[:space:]]*${artifact}" \
    || fail "$jobname must download the artifact named \`$artifact\` (the attested .crate)"
  printf '%s\n' "$body" | grep -qE "cargo[[:space:]]+package[[:space:]]+-p[[:space:]]+${package}[[:space:]]+--locked" \
    || fail "$jobname must re-run \`cargo package -p $package --locked\` so it can sha256-compare to the attested .crate (pre-publish gate)"
  printf '%s\n' "$body" | grep -qE "cargo[[:space:]]+publish[[:space:]]+-p[[:space:]]+${package}[[:space:]]+--locked" \
    || fail "$jobname must run \`cargo publish -p $package --locked\`"
  printf '%s\n' "$body" | grep -qE 'sha256sum' \
    || fail "$jobname must sha256sum-compare the repackaged .crate vs the attested .crate before publishing"
  printf '%s\n' "$body" | grep -qE "crates\.io/api/v1/crates/${package}|static\.crates\.io/crates/${package}" \
    || fail "$jobname must download the just-published .crate from crates.io after \`cargo publish\` (post-publish byte-identity proof; pre-publish alone cannot inspect cargo publish's internal packaging)"

  pre_line="$(require_job_line "$jobname" '^[[:space:]]+- name:[[:space:]]*Verify byte-identity vs the attested \.crate' 'a pre-publish byte-identity verification step')"
  oidc_line="$(require_job_line "$jobname" '^[[:space:]]+- name:[[:space:]]*Mint a short-lived crates\.io credential' 'an OIDC credential mint step')"
  publish_line="$(require_job_line "$jobname" '^[[:space:]]+- name:[[:space:]]*cargo publish' 'a cargo publish step')"
  post_line="$(require_job_line "$jobname" '^[[:space:]]+- name:[[:space:]]*Post-publish byte-identity' 'a post-publish crates.io byte-identity step')"
  [ "$pre_line" -lt "$oidc_line" ] \
    || fail "$jobname must verify byte-identity BEFORE minting the crates.io OIDC credential"
  [ "$oidc_line" -lt "$publish_line" ] \
    || fail "$jobname must mint the crates.io OIDC credential BEFORE \`cargo publish\`"
  [ "$publish_line" -lt "$post_line" ] \
    || fail "$jobname must run the crates.io post-publish download/compare AFTER \`cargo publish\`"
}

check_crate_publish_job publish-crate ordvec dist-crate
check_crate_publish_job publish-manifest-crate ordvec-manifest dist-manifest-crate
job_needs publish-manifest-crate publish-crate \
  || fail "publish-manifest-crate must \`needs: publish-crate\` so the lockstep core crate publishes first"

pcd="$(job_body pypi-canonical-dist)"
printf '%s\n' "$pcd" | grep -qE 'release_pypi_canonical_dist\.py canonicalize' \
  || fail "pypi-canonical-dist must canonicalize Python artifacts before attestation/release upload"
printf '%s\n' "$pcd" | grep -qE 'name:[[:space:]]*pypi-canonical-dist' \
  || fail "pypi-canonical-dist must upload the canonical Python dist artifact"

ppb="$(job_body publish-pypi)"
job_needs publish-pypi pypi-canonical-dist \
  || fail "publish-pypi must \`needs: pypi-canonical-dist\` (publish/verify exactly the canonical files)"
printf '%s\n' "$ppb" | grep -qE 'name:[[:space:]]*pypi-canonical-dist' \
  || fail "publish-pypi must consume pypi-canonical-dist, not raw rebuilt wheel/sdist artifacts"
printf '%s\n' "$ppb" | grep -qE 'release_pypi_canonical_dist\.py verify' \
  || fail "publish-pypi must verify PyPI-served wheel/sdist hashes against canonical dist"
grep -q 'pypi.org/pypi' tests/release_pypi_canonical_dist.py \
  || fail "release_pypi_canonical_dist.py must query PyPI for served file hashes"

# ----------------------------------------------------------------------
# (10) publish-github-release un-drafts ONLY AFTER all registry publishes succeed.
# ----------------------------------------------------------------------
for dep in publish-crate publish-manifest-crate publish-pypi; do
  job_needs publish-github-release "$dep" \
    || fail "publish-github-release must \`needs: $dep\` (un-draft only after all registry publishes succeed)"
done
unp="$(job_body publish-github-release)"
printf '%s\n' "$unp" | grep -qE 'gh release edit.*--draft=false' \
  || fail "publish-github-release must \`gh release edit <tag> --draft=false\` (this is the sole un-draft point)"
printf '%s\n' "$unp" | grep -qE "$github_repo_env_re" \
  || fail "publish-github-release must set \`GH_REPO: \${{ github.repository }}\` (no checkout, so gh release edit needs explicit repo context)"

echo "OK: signed-release invariants hold."
