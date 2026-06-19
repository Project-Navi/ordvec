#!/usr/bin/env bash
#
# Signed-release / provenance invariants — pinned in CI.
#
# release.yml's signed-release graph attaches the .intoto.jsonl and Sigstore
# assets that OpenSSF Scorecard detects for Signed-Releases, while older
# unsigned releases may keep the score below 10 temporarily. The same graph
# keeps the build-attest-publish chain honest:
#
#     build-{crate,manifest-crate,wheels,manifest-wheels,sdist,manifest-sdist}
#         (raw artifacts)
#         |
#         +-> pypi-canonical-dist (current build, or verified immutable PyPI files)
#         +-> pypi-manifest-canonical-dist (same for ordvec-manifest)
#         |
#         +-> attest         (id-token + attestations + .sigstore.json;
#         |                   Rust-crates-only when PyPI files already exist)
#         +-> provenance     (slsa-github-generator @vX.Y.Z, .intoto.jsonl;
#         |                   Rust-crates-only when PyPI files already exist)
#         |
#         v
#     release-assets-draft   (uploads core .crate/canonical .whl/.tar.gz/.sigstore.json/.intoto.jsonl to DRAFT release)
#         |
#         +--> publish-crate (byte-identity check vs attested .crate, then cargo publish)
#         +--> publish-pypi  (Trusted Publishing, or existing-file verification)
#         |
#         +-> build/attest/provenance/release-manifest-assets-draft
#         |      (after publish-crate; uploads manifest .crate/.whl/.tar.gz/.sigstore.json/.intoto.jsonl)
#         |
#         +--> publish-manifest-crate (same byte-identity proof after manifest assets stage)
#         +--> publish-manifest-pypi  (Trusted Publishing, or existing-file verification)
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
fail() { echo "::error::signed-release invariant violated: $*" >&2; exit 1; }

wf=".github/workflows/release.yml"
[ -f "$wf" ] || fail "$wf: workflow file not found"

matches_regex() {
  local text="$1" pattern="$2"
  grep -qE -- "$pattern" <<<"$text"
}

contains_literal() {
  local text="$1" needle="$2"
  grep -Fq -- "$needle" <<<"$text"
}

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
  local body escaped
  body="$(job_body "$jobname")"
  escaped="$(printf '%s' "$needed" | sed 's/[][\\.^$*+?{}|()]/\\&/g')"
  matches_regex "$body" "(^[[:space:]]+needs:[[:space:]]*\\[[^]]*(^|[^A-Za-z0-9_-])${escaped}([^A-Za-z0-9_-]|$)|^[[:space:]]+-[[:space:]]+${escaped}[[:space:]]*$)"
}

job_line() {
  local jobname="$1" pattern="$2" body line
  body="$(job_body "$jobname")"
  line="$(grep -nE -m 1 -- "$pattern" <<<"$body" || true)"
  if [ -n "$line" ]; then
    printf '%s\n' "${line%%:*}"
  fi
  return 0
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
#     + fail-closed release AVX-512 proof + exact linux/aarch64 wheel smoke
# ----------------------------------------------------------------------
for dep in attest provenance pypi-canonical-dist require-ci-green release-avx512 notes smoke-linux-aarch64-wheel; do
  job_needs release-assets-draft "$dep" \
    || fail "release-assets-draft must \`needs: $dep\` (fail-closed on missing provenance/CI)"
done

# ----------------------------------------------------------------------
# (2) release-assets-draft uploads every required asset class to the Release
# ----------------------------------------------------------------------
body_draft="$(job_body release-assets-draft)"
github_repo_env_re='^[[:space:]]+GH_REPO:[[:space:]]*"?\$\{\{[[:space:]]*github\.repository[[:space:]]*\}\}"?[[:space:]]*$'
for ext in '\.crate' '\.whl' '\.tar\.gz' '\.sigstore\.json' '\.intoto\.jsonl'; do
  matches_regex "$body_draft" "dist/\*${ext}([^a-zA-Z]|$)" \
    || fail "release-assets-draft must \`gh release upload\` dist/*$(printf '%s' "$ext" | sed 's/\\//g')"
done
matches_regex "$body_draft" 'name:[[:space:]]*pypi-canonical-dist' \
  || fail "release-assets-draft must upload canonical Python dist, not raw rebuilt wheel/sdist artifacts"
job_downloads_artifact_to_path release-assets-draft dist-crate dist \
  || fail "release-assets-draft must download the core dist-crate artifact into dist"
matches_regex "$body_draft" "$github_repo_env_re" \
  || fail "release-assets-draft must set \`GH_REPO: \${{ github.repository }}\` (no checkout, so gh release upload needs explicit repo context)"

body_manifest_draft="$(job_body release-manifest-assets-draft)"
job_needs release-manifest-assets-draft build-manifest-crate \
  || fail "release-manifest-assets-draft must \`needs: build-manifest-crate\`"
job_needs release-manifest-assets-draft attest-manifest \
  || fail "release-manifest-assets-draft must \`needs: attest-manifest\`"
job_needs release-manifest-assets-draft manifest-provenance \
  || fail "release-manifest-assets-draft must \`needs: manifest-provenance\`"
job_needs release-manifest-assets-draft pypi-manifest-canonical-dist \
  || fail "release-manifest-assets-draft must \`needs: pypi-manifest-canonical-dist\`"
job_needs release-manifest-assets-draft smoke-linux-aarch64-manifest-wheel \
  || fail "release-manifest-assets-draft must \`needs: smoke-linux-aarch64-manifest-wheel\`"
job_downloads_artifact_to_path release-manifest-assets-draft dist-manifest-crate dist \
  || fail "release-manifest-assets-draft must download the manifest dist-manifest-crate artifact into dist"
job_downloads_artifact_to_path release-manifest-assets-draft sigstore-bundle-manifest dist \
  || fail "release-manifest-assets-draft must download the manifest Sigstore bundle into dist"
job_downloads_artifact_to_path release-manifest-assets-draft pypi-manifest-canonical-dist dist \
  || fail "release-manifest-assets-draft must download the canonical manifest Python dist into dist"
matches_regex "$body_manifest_draft" 'dist/\*\.crate([^a-zA-Z]|$)' \
  || fail "release-manifest-assets-draft must upload dist/*.crate"
matches_regex "$body_manifest_draft" 'dist/\*\.whl([^a-zA-Z]|$)' \
  || fail "release-manifest-assets-draft must upload dist/*.whl"
matches_regex "$body_manifest_draft" 'dist/\*\.tar\.gz([^a-zA-Z]|$)' \
  || fail "release-manifest-assets-draft must upload dist/*.tar.gz"
matches_regex "$body_manifest_draft" 'dist/\*\.sigstore\.json([^a-zA-Z]|$)' \
  || fail "release-manifest-assets-draft must upload dist/*.sigstore.json"
matches_regex "$body_manifest_draft" 'dist/\*\.intoto\.jsonl([^a-zA-Z]|$)' \
  || fail "release-manifest-assets-draft must upload dist/*.intoto.jsonl"
matches_regex "$body_manifest_draft" "$github_repo_env_re" \
  || fail "release-manifest-assets-draft must set \`GH_REPO: \${{ github.repository }}\`"

# ----------------------------------------------------------------------
# (3) release-assets-draft must NOT un-draft (the dedicated un-draft job owns
#     that; un-drafting here would re-introduce the public-release-before-
#     publish failure mode).
# ----------------------------------------------------------------------
if matches_regex "$body_draft" 'gh release edit.*--draft=false'; then
  fail "release-assets-draft must NOT un-draft the Release (un-drafting belongs in publish-github-release, after all registry publishes succeed)"
fi
if matches_regex "$body_manifest_draft" 'gh release edit.*--draft=false'; then
  fail "release-manifest-assets-draft must NOT un-draft the Release (un-drafting belongs in publish-github-release, after all registry publishes succeed)"
fi

# ----------------------------------------------------------------------
# (4) provenance uses slsa-github-generator pinned to a SEMANTIC VERSION TAG
#     (NOT a SHA — SLSA trust model requires the tag for its self-verification)
# ----------------------------------------------------------------------
prov="$(job_body provenance)"
matches_regex "$prov" 'uses:[[:space:]]*slsa-framework/slsa-github-generator/.+/generator_generic_slsa3\.yml@v[0-9]+\.[0-9]+\.[0-9]+' \
  || fail "provenance must \`uses: slsa-framework/slsa-github-generator/.../generator_generic_slsa3.yml@vX.Y.Z\` (tag-pinned per SLSA trust model)"
manifest_prov="$(job_body manifest-provenance)"
matches_regex "$manifest_prov" 'uses:[[:space:]]*slsa-framework/slsa-github-generator/.+/generator_generic_slsa3\.yml@v[0-9]+\.[0-9]+\.[0-9]+' \
  || fail "manifest-provenance must \`uses: slsa-framework/slsa-github-generator/.../generator_generic_slsa3.yml@vX.Y.Z\` (tag-pinned per SLSA trust model)"

# ----------------------------------------------------------------------
# (5) provenance must have `upload-assets: false` — asset-staging jobs, not
#     SLSA generator workflows, own Release uploads.
# ----------------------------------------------------------------------
matches_regex "$prov" '^[[:space:]]+upload-assets:[[:space:]]*false[[:space:]]*$' \
  || fail "provenance must set \`upload-assets: false\` (release-assets-draft uploads the collected .intoto.jsonl from the workflow-artifact path)"
matches_regex "$manifest_prov" '^[[:space:]]+upload-assets:[[:space:]]*false[[:space:]]*$' \
  || fail "manifest-provenance must set \`upload-assets: false\` (release-manifest-assets-draft uploads the collected .intoto.jsonl from the workflow-artifact path)"

# ----------------------------------------------------------------------
# (6) provenance-name MUST end in `.intoto.jsonl` — Scorecard's provenance
#     probe is a pure filename-suffix match.
# ----------------------------------------------------------------------
matches_regex "$prov" '^[[:space:]]+provenance-name:.*\.intoto\.jsonl[[:space:]]*$' \
  || fail "provenance must set \`provenance-name: <name>.intoto.jsonl\` (Scorecard Signed-Releases provenance probe matches this suffix only)"
matches_regex "$manifest_prov" '^[[:space:]]+provenance-name:.*ordvec-manifest-.*\.intoto\.jsonl[[:space:]]*$' \
  || fail "manifest-provenance must set \`provenance-name: ordvec-manifest-<version>.intoto.jsonl\`"

# ----------------------------------------------------------------------
# (7) attest job grants id-token: write + attestations: write
# ----------------------------------------------------------------------
att="$(job_body attest)"
matches_regex "$att" '^[[:space:]]+id-token:[[:space:]]*write' \
  || fail "attest job must grant \`id-token: write\` (Sigstore OIDC signing cert)"
matches_regex "$att" '^[[:space:]]+attestations:[[:space:]]*write' \
  || fail "attest job must grant \`attestations: write\` (persist to the GitHub attestation store)"
att_manifest="$(job_body attest-manifest)"
matches_regex "$att_manifest" '^[[:space:]]+id-token:[[:space:]]*write' \
  || fail "attest-manifest job must grant \`id-token: write\` (Sigstore OIDC signing cert)"
matches_regex "$att_manifest" '^[[:space:]]+attestations:[[:space:]]*write' \
  || fail "attest-manifest job must grant \`attestations: write\` (persist to the GitHub attestation store)"
job_needs attest-manifest build-manifest-crate \
  || fail "attest-manifest must \`needs: build-manifest-crate\`"
job_needs attest-manifest pypi-manifest-canonical-dist \
  || fail "attest-manifest must \`needs: pypi-manifest-canonical-dist\`"
job_downloads_artifact_to_path attest-manifest dist-manifest-crate dist \
  || fail "attest-manifest must download the dist-manifest-crate artifact into dist"
job_downloads_artifact_to_path attest-manifest pypi-manifest-canonical-dist dist \
  || fail "attest-manifest must download the canonical manifest Python dist into dist"

comb="$(job_body combine-hashes)"
comb_manifest="$(job_body combine-manifest-hash)"
job_needs combine-manifest-hash build-manifest-crate \
  || fail "combine-manifest-hash must \`needs: build-manifest-crate\` so the manifest .crate is a SLSA subject"
job_needs combine-manifest-hash pypi-manifest-canonical-dist \
  || fail "combine-manifest-hash must \`needs: pypi-manifest-canonical-dist\` so manifest Python dist can be SLSA subjects"
job_downloads_artifact_to_path combine-manifest-hash dist-manifest-crate dist \
  || fail "combine-manifest-hash must download the dist-manifest-crate artifact into dist"
job_downloads_artifact_to_path combine-manifest-hash pypi-manifest-canonical-dist dist \
  || fail "combine-manifest-hash must download the canonical manifest Python dist when it is built by this run"

build_manifest="$(job_body build-manifest-crate)"
job_needs build-manifest-crate publish-crate \
  || fail "build-manifest-crate must \`needs: publish-crate\` so lockstep ordvec exists on crates.io"
matches_regex "$build_manifest" 'cargo[[:space:]]+package[[:space:]]+-p[[:space:]]+ordvec-manifest[[:space:]]+--locked([^[:alnum:]_-]|$)' \
  || fail "build-manifest-crate must package ordvec-manifest with Cargo registry verification"
if contains_literal "$build_manifest" '--no-verify'; then
  fail "build-manifest-crate must not use --no-verify after publish-crate"
fi

# ----------------------------------------------------------------------
# (8) Registry publish jobs grant id-token: write AND need release-assets-draft.
# ----------------------------------------------------------------------
for pub in publish-crate publish-pypi; do
  body="$(job_body "$pub")"
  matches_regex "$body" '^[[:space:]]+id-token:[[:space:]]*write' \
    || fail "$pub must grant \`id-token: write\` (Trusted Publishing OIDC)"
  job_needs "$pub" release-assets-draft \
    || fail "$pub must \`needs: release-assets-draft\` (gated by attest + provenance via the draft-assets edge)"
done
body="$(job_body publish-manifest-crate)"
matches_regex "$body" '^[[:space:]]+id-token:[[:space:]]*write' \
  || fail "publish-manifest-crate must grant \`id-token: write\` (Trusted Publishing OIDC)"
job_needs publish-manifest-crate release-manifest-assets-draft \
  || fail "publish-manifest-crate must \`needs: release-manifest-assets-draft\`"
job_needs publish-manifest-crate publish-crate \
  || fail "publish-manifest-crate must \`needs: publish-crate\`"
body="$(job_body publish-manifest-pypi)"
matches_regex "$body" '^[[:space:]]+id-token:[[:space:]]*write' \
  || fail "publish-manifest-pypi must grant \`id-token: write\` (Trusted Publishing OIDC)"
job_needs publish-manifest-pypi release-manifest-assets-draft \
  || fail "publish-manifest-pypi must \`needs: release-manifest-assets-draft\`"
job_needs publish-manifest-pypi publish-manifest-crate \
  || fail "publish-manifest-pypi must \`needs: publish-manifest-crate\` (manifest crate publishes before manifest PyPI)"
job_needs publish-pypi publish-crate \
  || fail "publish-pypi must \`needs: publish-crate\` (core crate publishes before core PyPI)"

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
  matches_regex "$body" 'uses:[[:space:]]*actions/download-artifact' \
    || fail "$jobname must download the attested $artifact artifact (byte-identity gate)"
  matches_regex "$body" "name:[[:space:]]*${artifact}" \
    || fail "$jobname must download the artifact named \`$artifact\` (the attested .crate)"
  matches_regex "$body" "cargo[[:space:]]+package[[:space:]]+-p[[:space:]]+${package}[[:space:]]+--locked" \
    || fail "$jobname must re-run \`cargo package -p $package --locked\` so it can sha256-compare to the attested .crate (pre-publish gate)"
  matches_regex "$body" "cargo[[:space:]]+publish[[:space:]]+-p[[:space:]]+${package}[[:space:]]+--locked" \
    || fail "$jobname must run \`cargo publish -p $package --locked\`"
  matches_regex "$body" 'sha256sum' \
    || fail "$jobname must sha256sum-compare the repackaged .crate vs the attested .crate before publishing"
  matches_regex "$body" "crates\.io/api/v1/crates/${package}|static\.crates\.io/crates/${package}" \
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
manifest_pre_line="$(require_job_line publish-manifest-crate '^[[:space:]]+- name:[[:space:]]*Verify byte-identity vs the attested \.crate' 'a manifest pre-publish byte-identity verification step')"
manifest_dry_line="$(require_job_line publish-manifest-crate '^[[:space:]]+- name:[[:space:]]*Validate manifest publish dry-run' 'a manifest publish dry-run step')"
manifest_oidc_line="$(require_job_line publish-manifest-crate '^[[:space:]]+- name:[[:space:]]*Mint a short-lived crates\.io credential' 'a manifest OIDC credential mint step')"
awk '/cargo[[:space:]]+publish/ && /ordvec-manifest/ && /--dry-run/ && /--locked/ { found = 1 } END { exit found ? 0 : 1 }' \
  <<<"$(job_body publish-manifest-crate)" \
  || fail "publish-manifest-crate must dry-run \`cargo publish -p ordvec-manifest --dry-run --locked\` after byte-identity and before OIDC"
[ "$manifest_pre_line" -lt "$manifest_dry_line" ] \
  || fail "publish-manifest-crate must dry-run publish AFTER byte-identity verification"
[ "$manifest_dry_line" -lt "$manifest_oidc_line" ] \
  || fail "publish-manifest-crate must dry-run publish BEFORE minting the crates.io OIDC credential"

pcd="$(job_body pypi-canonical-dist)"
matches_regex "$pcd" 'release_pypi_canonical_dist\.py canonicalize' \
  || fail "pypi-canonical-dist must canonicalize Python artifacts before attestation/release upload"
matches_regex "$pcd" 'name:[[:space:]]*pypi-canonical-dist' \
  || fail "pypi-canonical-dist must upload the canonical Python dist artifact"
pcd_manifest="$(job_body pypi-manifest-canonical-dist)"
matches_regex "$pcd_manifest" 'release_pypi_canonical_dist\.py canonicalize' \
  || fail "pypi-manifest-canonical-dist must canonicalize manifest Python artifacts before attestation/release upload"
matches_regex "$pcd_manifest" '--project[[:space:]]+ordvec-manifest' \
  || fail "pypi-manifest-canonical-dist must canonicalize the ordvec-manifest PyPI project"
matches_regex "$pcd_manifest" 'name:[[:space:]]*pypi-manifest-canonical-dist' \
  || fail "pypi-manifest-canonical-dist must upload the canonical manifest Python dist artifact"

ppb="$(job_body publish-pypi)"
job_needs publish-pypi pypi-canonical-dist \
  || fail "publish-pypi must \`needs: pypi-canonical-dist\` (publish/verify exactly the canonical files)"
matches_regex "$ppb" 'name:[[:space:]]*pypi-canonical-dist' \
  || fail "publish-pypi must consume pypi-canonical-dist, not raw rebuilt wheel/sdist artifacts"
matches_regex "$ppb" 'release_pypi_canonical_dist\.py verify' \
  || fail "publish-pypi must verify PyPI-served wheel/sdist hashes against canonical dist"
mppb="$(job_body publish-manifest-pypi)"
job_needs publish-manifest-pypi pypi-manifest-canonical-dist \
  || fail "publish-manifest-pypi must \`needs: pypi-manifest-canonical-dist\` (publish/verify exactly the canonical manifest files)"
matches_regex "$mppb" 'name:[[:space:]]*pypi-manifest-canonical-dist' \
  || fail "publish-manifest-pypi must consume pypi-manifest-canonical-dist, not raw rebuilt wheel/sdist artifacts"
matches_regex "$mppb" 'release_pypi_canonical_dist\.py verify' \
  || fail "publish-manifest-pypi must verify PyPI-served manifest wheel/sdist hashes against canonical dist"
matches_regex "$mppb" '--project[[:space:]]+ordvec-manifest' \
  || fail "publish-manifest-pypi must verify the ordvec-manifest PyPI project"
grep -q 'pypi.org/pypi' tests/release_pypi_canonical_dist.py \
  || fail "release_pypi_canonical_dist.py must query PyPI for served file hashes"

# ----------------------------------------------------------------------
# (10) publish-github-release un-drafts ONLY AFTER all registry publishes succeed.
# ----------------------------------------------------------------------
for dep in publish-crate publish-manifest-crate publish-pypi publish-manifest-pypi; do
  job_needs publish-github-release "$dep" \
    || fail "publish-github-release must \`needs: $dep\` (un-draft only after all registry publishes succeed)"
done
unp="$(job_body publish-github-release)"
matches_regex "$unp" 'gh release edit.*--draft=false' \
  || fail "publish-github-release must \`gh release edit <tag> --draft=false\` (this is the sole un-draft point)"
matches_regex "$unp" "$github_repo_env_re" \
  || fail "publish-github-release must set \`GH_REPO: \${{ github.repository }}\` (no checkout, so gh release edit needs explicit repo context)"

echo "OK: signed-release invariants hold."
