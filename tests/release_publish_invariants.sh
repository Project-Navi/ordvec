#!/usr/bin/env bash
#
# Release-publish SBOM invariants — pinned in CI.
#
# release.yml is the unified tag-triggered release pipeline. A generated
# *.cdx.json SBOM once dirtied the release publish flow:
#   * crate — the untracked SBOM dirtied the git tree, so `cargo publish`
#     refused it;
#   * PyPI  — non-distribution artifacts in dist/ make the PyPI upload fail.
set -euo pipefail
fail() { echo "::error::release-publish invariant violated: $*"; exit 1; }

# Both generated SBOMs must be gitignored. A tracked/untracked *.cdx.json
#     makes `cargo publish` refuse the dirty tree and would otherwise bundle
#     the SBOM into the .crate.
for f in ordvec.cdx.json ordvec-python/ordvec-python.cdx.json; do
  git check-ignore -q -- "$f" || fail "$f is not gitignored (it is a generated SBOM artifact)"
done

python3 tests/release_publish_invariants.py

echo "OK: release-publish SBOM invariants hold."
