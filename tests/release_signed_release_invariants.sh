#!/usr/bin/env bash
#
# Signed-release / provenance invariants — pinned in CI.
#
# release.yml's signed-release graph attaches the .intoto.jsonl and Sigstore
# assets that OpenSSF Scorecard detects for Signed-Releases, while older
# unsigned releases may keep the score below 10 temporarily. The same graph
# keeps the build-attest-publish chain honest:
#
#     build-{crate,wheels,sdist}                  (artifacts)
#         |
#         +-> attest         (id-token + attestations + .sigstore.json)
#         +-> provenance     (slsa-github-generator @vX.Y.Z, .intoto.jsonl)
#         |
#         v
#     release-assets-draft   (uploads .crate/.whl/.tar.gz/.sigstore.json/.intoto.jsonl to DRAFT release)
#         |
#         +--> publish-crate (byte-identity check vs attested .crate, then cargo publish)
#         +--> publish-pypi  (Trusted Publishing)
#               |
#               v
#         publish-github-release (un-draft, ONLY after both publishes succeed)
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

# ----------------------------------------------------------------------
# (1) release-assets-draft needs attest + provenance + require-ci-green + notes
#     + exact linux/aarch64 wheel smoke
# ----------------------------------------------------------------------
for dep in attest provenance require-ci-green notes smoke-linux-aarch64-wheel; do
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
printf '%s\n' "$body_draft" | grep -qE "$github_repo_env_re" \
  || fail "release-assets-draft must set \`GH_REPO: \${{ github.repository }}\` (no checkout, so gh release upload needs explicit repo context)"

# ----------------------------------------------------------------------
# (3) release-assets-draft must NOT un-draft (the dedicated un-draft job owns
#     that; un-drafting here would re-introduce the public-release-before-
#     publish failure mode).
# ----------------------------------------------------------------------
if printf '%s\n' "$body_draft" | grep -qE 'gh release edit.*--draft=false'; then
  fail "release-assets-draft must NOT un-draft the Release (un-drafting belongs in publish-github-release, after both publishes succeed)"
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

# ----------------------------------------------------------------------
# (8) Both publish jobs grant id-token: write AND need release-assets-draft.
# ----------------------------------------------------------------------
for pub in publish-crate publish-pypi; do
  body="$(job_body "$pub")"
  printf '%s\n' "$body" | grep -qE '^[[:space:]]+id-token:[[:space:]]*write' \
    || fail "$pub must grant \`id-token: write\` (Trusted Publishing OIDC)"
  job_needs "$pub" release-assets-draft \
    || fail "$pub must \`needs: release-assets-draft\` (gated by attest + provenance via the draft-assets edge)"
done

# ----------------------------------------------------------------------
# (9) publish-crate proves byte-identity vs the attested .crate on BOTH
#     sides of `cargo publish`:
#       (9a) pre-publish: download the attested .crate, re-run `cargo package`,
#            sha256-compare. Fail-closed BEFORE the OIDC token is minted.
#       (9b) post-publish: download the just-published .crate from crates.io
#            and sha256-compare to the attested artifact. cargo publish runs
#            its own internal packaging step that the pre-publish gate
#            cannot inspect — this is the empirical proof that the bytes
#            actually served by crates.io match the SLSA-attested artifact.
# ----------------------------------------------------------------------
pcb="$(job_body publish-crate)"
printf '%s\n' "$pcb" | grep -qE 'uses:[[:space:]]*actions/download-artifact' \
  || fail "publish-crate must download the attested dist-crate artifact (byte-identity gate)"
printf '%s\n' "$pcb" | grep -qE 'name:[[:space:]]*dist-crate' \
  || fail "publish-crate must download the artifact named \`dist-crate\` (the attested .crate)"
printf '%s\n' "$pcb" | grep -qE 'cargo[[:space:]]+package[[:space:]]+-p[[:space:]]+ordvec[[:space:]]+--locked' \
  || fail "publish-crate must re-run \`cargo package -p ordvec --locked\` so it can sha256-compare to the attested .crate (pre-publish gate)"
printf '%s\n' "$pcb" | grep -qE 'sha256sum' \
  || fail "publish-crate must sha256sum-compare the repackaged .crate vs the attested .crate before publishing"
printf '%s\n' "$pcb" | grep -qE 'crates\.io/api/v1/crates/ordvec|static\.crates\.io/crates/ordvec' \
  || fail "publish-crate must download the just-published .crate from crates.io after \`cargo publish\` (post-publish byte-identity proof; pre-publish alone cannot inspect cargo publish's internal packaging)"

pre_line="$(require_job_line publish-crate '^[[:space:]]+- name:[[:space:]]*Verify byte-identity vs the attested \.crate' 'a pre-publish byte-identity verification step')"
oidc_line="$(require_job_line publish-crate '^[[:space:]]+- name:[[:space:]]*Mint a short-lived crates\.io credential' 'an OIDC credential mint step')"
publish_line="$(require_job_line publish-crate '^[[:space:]]+- name:[[:space:]]*cargo publish' 'a cargo publish step')"
post_line="$(require_job_line publish-crate '^[[:space:]]+- name:[[:space:]]*Post-publish byte-identity' 'a post-publish crates.io byte-identity step')"
[ "$pre_line" -lt "$oidc_line" ] \
  || fail "publish-crate must verify byte-identity BEFORE minting the crates.io OIDC credential"
[ "$oidc_line" -lt "$publish_line" ] \
  || fail "publish-crate must mint the crates.io OIDC credential BEFORE \`cargo publish\`"
[ "$publish_line" -lt "$post_line" ] \
  || fail "publish-crate must run the crates.io post-publish download/compare AFTER \`cargo publish\`"

ppb="$(job_body publish-pypi)"
printf '%s\n' "$ppb" | grep -qE 'Post-publish PyPI hashes match staged dist' \
  || fail "publish-pypi must verify PyPI-served wheel/sdist hashes after publish"
printf '%s\n' "$ppb" | grep -qE 'pypi\.org/pypi/ordvec/.+/json|pypi\.org/pypi/ordvec/' \
  || fail "publish-pypi must query PyPI after publish for served file hashes"

# ----------------------------------------------------------------------
# (10) publish-github-release un-drafts ONLY AFTER both registry publishes succeed.
# ----------------------------------------------------------------------
for dep in publish-crate publish-pypi; do
  job_needs publish-github-release "$dep" \
    || fail "publish-github-release must \`needs: $dep\` (un-draft only after BOTH registry publishes succeed)"
done
unp="$(job_body publish-github-release)"
printf '%s\n' "$unp" | grep -qE 'gh release edit.*--draft=false' \
  || fail "publish-github-release must \`gh release edit <tag> --draft=false\` (this is the sole un-draft point)"
printf '%s\n' "$unp" | grep -qE "$github_repo_env_re" \
  || fail "publish-github-release must set \`GH_REPO: \${{ github.repository }}\` (no checkout, so gh release edit needs explicit repo context)"

echo "OK: signed-release invariants hold."
