#!/usr/bin/env bash
#
# Release-publish SBOM invariants — pinned in CI.
#
# release-crate.yml / release-python.yml are workflow_dispatch-only, so their
# "generate a CycloneDX SBOM, then publish" flow never runs in push/PR CI. A
# generated *.cdx.json SBOM once broke BOTH publish paths and would only have
# surfaced at a manual release:
#   * crate — the untracked SBOM dirtied the git tree, so `cargo publish` refused
#     it (and would otherwise bundle it into the published .crate);
#   * PyPI  — the SBOM artifact was downloaded into dist/, which twine rejects.
# This pins the fixes so a regression fails here, on every push/PR, instead of
# silently passing CI and only breaking at manual release time.
set -euo pipefail
fail() { echo "::error::release-publish invariant violated: $*"; exit 1; }

# (1) Both generated SBOMs must be gitignored. A tracked/untracked *.cdx.json
#     makes `cargo publish` refuse the (dirty) tree and would otherwise bundle
#     the SBOM into the .crate. (Verified end-to-end when this guard was added:
#     `cargo publish --dry-run` is clean with the SBOM present iff it stays
#     gitignored — so this check is the durable pin.)
for f in ordvec.cdx.json ordvec-python/ordvec-python.cdx.json; do
  git check-ignore -q -- "$f" || fail "$f is not gitignored (it is a generated SBOM artifact)"
done

# (2) In the PyPI publish job the step order must be:
#       actions/download-artifact  (pulls the SBOM into dist/)
#         -> delete *.cdx.json from dist/
#           -> pypa/gh-action-pypi-publish upload.
#     twine rejects a stray .cdx.json in dist/, so the cleanup must run AFTER the
#     download (otherwise it is a no-op for the downloaded SBOM) and BEFORE the
#     upload. Match real step keys only: anchor on `uses:`/`run:`, skip YAML
#     comment lines, and key on the pinned action name (`pypa/gh-action-pypi-publish`)
#     rather than the bare string `pypi-publish`, which could match a job name.
wf=".github/workflows/release-python.yml"
[ -f "$wf" ] || fail "$wf: workflow file not found"

# Line number of the first real (non-comment) line matching the regex, if any.
step_line() { grep -nE "$1" "$wf" | grep -vE '^[0-9]+:[[:space:]]*#' | head -1 | cut -d: -f1; }

dl_line="$(step_line 'uses:[[:space:]]*actions/download-artifact' || true)"
clean_line="$(step_line 'run:.*(find|rm).*cdx\.json' || true)"
pub_line="$(step_line 'uses:[[:space:]]*pypa/gh-action-pypi-publish' || true)"

[ -n "$dl_line" ]    || fail "$wf: no actions/download-artifact step found"
[ -n "$clean_line" ] || fail "$wf: publish job has no run: step deleting *.cdx.json from dist/"
[ -n "$pub_line" ]   || fail "$wf: no pypa/gh-action-pypi-publish step found"

[ "$dl_line" -lt "$clean_line" ] \
  || fail "$wf: the *.cdx.json cleanup (line $clean_line) must run AFTER actions/download-artifact (line $dl_line), else it is a no-op for the downloaded SBOM"
[ "$clean_line" -lt "$pub_line" ] \
  || fail "$wf: the *.cdx.json cleanup (line $clean_line) must run BEFORE the pypa publish (line $pub_line)"

echo "OK: release-publish SBOM invariants hold."
