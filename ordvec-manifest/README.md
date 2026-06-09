# ordvec-manifest

Manifest verifier for ordvec index provenance and caller-owned sidecar
artifacts.

It verifies index bytes, probed header metadata, row identity, named auxiliary
artifacts, optional encoder distortion profile references, optional
calibration profile references, and attestation shape before a caller loads an
ordvec index. It does not sign artifacts, manage keys, call networks, mutate
index files, decide deployment trust policy, estimate encoder geometry, compute
calibration statistics, or change the C ABI.

`ordvec-manifest` is versioned in lockstep with the core `ordvec` crate. From a
workspace checkout, use the optional CLI with
`cargo run -p ordvec-manifest --features cli --`; from a published release,
install the binary with `cargo install ordvec-manifest --features cli`. The
library default feature set is empty and does not depend on `clap`.

```sh
ordvec-manifest create \
  --index path/to/index.tvrq \
  --row-id-is-identity \
  --aux ordinaldb.ids=path/to/ids.bin \
  --embedding-model bge-small-en-v1.5 \
  --out path/to/index.manifest.json

ordvec-manifest verify --manifest path/to/index.manifest.json
```

From a workspace checkout, prefix the same commands with
`cargo run -p ordvec-manifest --features cli --`.

The schema version is `ordvec.index_manifest.v1`. Relative paths resolve from
the manifest file's directory, absolute paths are rejected by default, and
relative paths may not escape the manifest directory unless explicitly allowed.
`create` follows the same policy: by default it emits only paths that should
verify with default settings. If an artifact or JSONL row map lives outside the
manifest directory, pass `--allow-path-escape` at create time and again at
verify time.

Library callers that need a verify-then-load sequence can use
`verify_for_load(manifest_path, VerifyOptions)` to obtain a `VerifiedLoadPlan`.
The helper verifies the manifest with the supplied options, fails closed by
returning `VerifiedLoadPlanError::VerificationFailed(report)` when report
errors exist, and otherwise returns the canonical primary artifact path, probed
metadata, row-identity summary, declared auxiliary artifact states, and the full
report. Callers that already hold a `ManifestDocument` can use
`verify_document_for_load(&document, VerifyOptions)` without re-reading the
manifest file. These helpers do not call `Rank::load`, `RankQuant::load`,
`Bitmap::load`, or `SignBitmap::load`, and they do not pin file descriptors or
lock mutable storage. Callers should load from the returned paths immediately on
storage they control, or re-verify if the backing files can change between
verification and load.

Controlled-storage load pattern:

```rust
let plan = ordvec_manifest::verify_for_load(&manifest_path, options)?;
let _ordinaldb_ids = plan.require_auxiliary("ordinaldb.ids")?;
let index = ordvec::RankQuant::load(plan.artifact_path())?;
```

Racy load pattern:

```rust
let plan = ordvec_manifest::verify_for_load(&manifest_path, options)?;
queue_for_later(plan);
// Another process may rewrite the artifact before the queued load runs.
```

If a manifest or artifact lives on shared, writable, or otherwise mutable
storage, re-run `verify_for_load` immediately before loading, load from
immutable storage, or use a caller-owned loading path that pins bytes.
`VerifiedLoadPlan` is not a byte pin.

Verification uses bounded parser/report defaults on both CLI and library paths.
Stable limit codes are part of the contract:

- manifest JSON: 1 MiB before JSON parsing
  (`manifest_file_too_large`);
- row-identity JSONL line: 64 KiB (`row_identity_line_too_large`);
- row-identity JSONL rows: 10,000,000
  (`row_identity_row_count_limit_exceeded`);
- row-identity duplicate-tracking `db_id` bytes: 64 MiB
  (`row_identity_duplicate_tracking_limit_exceeded`);
- auxiliary artifact declarations: 1,024
  (`auxiliary_artifact_count_limit_exceeded`);
- auxiliary artifact bytes per declared file: 64 MiB
  (`auxiliary_artifact_file_too_large`);
- encoder distortion profile artifact bytes: 64 MiB
  (`encoder_distortion_profile_too_large`);
- collected report issues: 1,024, after which a
  `verification_report_issue_limit_exceeded` issue is emitted;
- SQLite cached report JSON: 4 MiB (`sqlite_cached_report_too_large`).

The CLI exposes matching override flags on `inspect`, `verify`, `create`,
`sqlite verify`, and `sqlite activate`: `--max-manifest-bytes`,
`--max-row-map-line-bytes`, `--max-row-map-rows`,
`--max-row-map-tracked-id-bytes`, `--max-auxiliary-artifacts`,
`--max-auxiliary-artifact-bytes`,
`--max-encoder-distortion-profile-bytes`, `--max-report-issues`, and
`--max-cached-report-bytes`. Library callers can override the same ceilings
via `VerifyOptions::limits`.

Stable limit codes:

| Limit surface | Verification report code | `ManifestError::code()` |
| --- | --- | --- |
| manifest JSON bytes | n/a | `manifest_file_too_large` |
| row-identity JSONL line bytes | `row_identity_line_too_large` | `row_identity_line_too_large` |
| row-identity JSONL rows | `row_identity_row_count_limit_exceeded` | `row_identity_row_count_limit_exceeded` |
| row-identity duplicate-tracking `db_id` bytes | `row_identity_duplicate_tracking_limit_exceeded` | `row_identity_duplicate_tracking_limit_exceeded` |
| auxiliary artifact declarations | `auxiliary_artifact_count_limit_exceeded` | n/a |
| auxiliary artifact bytes per declared file | `auxiliary_artifact_file_too_large` | n/a |
| encoder distortion profile artifact bytes | `encoder_distortion_profile_too_large` | n/a |
| collected verification report issues | `verification_report_issue_limit_exceeded` | n/a |
| SQLite cached report JSON bytes | n/a | `sqlite_cached_report_too_large` |

Oversized byte-limit overrides that cannot be represented safely by the
bounded in-memory reader fail before reading with the same stable
`ManifestError::code()` as the corresponding byte limit. A
`max_report_issues` override of `0` suppresses detail issues and returns only
the `verification_report_issue_limit_exceeded` sentinel when any issue would
otherwise be reported. These limits bound metadata parsing and report/cache
growth; hashing an index or calibration profile is still proportional to the
artifact bytes being verified. SQLite cache-key construction treats an
over-limit encoder distortion profile as non-cacheable and reruns verification
instead of reusing a previously cached report.

Manifests may declare `auxiliary_artifacts` for caller-owned sidecars that
should be integrity-checked with the same path policy as the primary index.
Each entry has a stable `name`, relative `path`, lowercase SHA-256 digest,
`file_size_bytes`, and a `required` flag that defaults to `true`. Required
members fail verification when missing, tampered, size-mismatched, or rejected
by path policy. Optional members are reported as verified when present or as
`optional_absent` with a stable reason code when absent. The verifier checks
bytes only; application semantics remain with the caller.

`create` can declare sidecars while it hashes them:
`--aux NAME=PATH` creates a required declaration and
`--optional-aux NAME=PATH` creates an optional declaration. Library callers use
`CreateAuxiliaryArtifact { name, path, required }` through
`CreateManifestOptions::auxiliary_artifacts`. `VerifiedLoadPlan` offers
`auxiliary_by_name(name)` for inspection and `require_auxiliary(name)` for
callers that must fail if a named sidecar is not declared and verified.

For OrdinalDB v0.1, keep the ordvec row identity as
`RowIdentity::RowIdIdentity { row_count }` and declare the OrdinalDB `ids.bin`
file as required auxiliary artifact name `ordinaldb.ids`. That makes the vector
row count an ordvec invariant while leaving OrdinalDB's `u64` document IDs as a
caller-owned sidecar. Do not encode `ids.bin` as `RowIdentity::Jsonl`: v1 JSONL
row identity is UUID-oriented (`id_kind = "uuid"`), and generic row-map ID
formats are intentionally deferred.

The unified JSON report carries per-sidecar audit fields. A successful
auxiliary artifact verification includes the manifest path, resolved/canonical
paths, declared digest/length, and observed digest/length:

```json
{
  "ok": true,
  "checked_at": "2026-06-03T17:20:00Z",
  "manifest_id": "urn:uuid:11111111-1111-4111-8111-111111111111",
  "artifact": {
    "manifest_path": "index.tvrq",
    "observed_path": "index.tvrq",
    "canonical_path": "/srv/index/index.tvrq",
    "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "size_bytes": 4096,
    "metadata": null
  },
  "auxiliary_artifacts": [
    {
      "name": "ordgrep.sidecar",
      "manifest_path": "ordgrep.sidecar.json",
      "resolved_path": "/srv/index/ordgrep.sidecar.json",
      "canonical_path": "/srv/index/ordgrep.sidecar.json",
      "expected_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
      "expected_size_bytes": 128,
      "required": true,
      "state": "verified",
      "reason_code": null,
      "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
      "size_bytes": 128
    }
  ],
  "row_identity": {
    "kind": "row_id_identity",
    "manifest_path": null,
    "canonical_path": null,
    "sha256": null,
    "row_count": 1024,
    "validated_rows": 1024
  },
  "calibration": {
    "present": false,
    "schema_version": null,
    "profile_id": null,
    "calibrated_for_model": null,
    "ordinalization": null,
    "null_model": null,
    "profile_manifest_path": null,
    "profile_canonical_path": null,
    "profile_sha256": null,
    "profile_size_bytes": null
  },
  "attestation_shape_checks": [],
  "errors": [],
  "warnings": [],
  "skipped_checks": []
}
```

A tampered or missing sidecar fails closed while preserving declared fields for
audit logging. Observed digest/length fields are present when bytes could be
read and absent when the file is missing:

```json
{
  "ok": false,
  "checked_at": "2026-06-03T17:21:00Z",
  "manifest_id": "urn:uuid:11111111-1111-4111-8111-111111111111",
  "artifact": {
    "manifest_path": "index.tvrq",
    "observed_path": "index.tvrq",
    "canonical_path": "/srv/index/index.tvrq",
    "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "size_bytes": 4096,
    "metadata": null
  },
  "auxiliary_artifacts": [
    {
      "name": "ordgrep.sidecar",
      "manifest_path": "ordgrep.sidecar.json",
      "resolved_path": "/srv/index/ordgrep.sidecar.json",
      "canonical_path": "/srv/index/ordgrep.sidecar.json",
      "expected_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
      "expected_size_bytes": 128,
      "required": true,
      "state": "failed",
      "reason_code": "auxiliary_artifact_sha256_mismatch",
      "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
      "size_bytes": 128
    },
    {
      "name": "required-model-card",
      "manifest_path": "model-card.json",
      "resolved_path": "/srv/index/model-card.json",
      "expected_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
      "expected_size_bytes": 2048,
      "required": true,
      "state": "missing_required",
      "reason_code": "auxiliary_artifact_missing_required",
      "sha256": null,
      "size_bytes": null
    }
  ],
  "row_identity": {
    "kind": "row_id_identity",
    "manifest_path": null,
    "canonical_path": null,
    "sha256": null,
    "row_count": 1024,
    "validated_rows": 1024
  },
  "calibration": {
    "present": false,
    "schema_version": null,
    "profile_id": null,
    "calibrated_for_model": null,
    "ordinalization": null,
    "null_model": null,
    "profile_manifest_path": null,
    "profile_canonical_path": null,
    "profile_sha256": null,
    "profile_size_bytes": null
  },
  "attestation_shape_checks": [],
  "errors": [
    {
      "code": "auxiliary_artifact_sha256_mismatch",
      "message": "auxiliary artifact \"ordgrep.sidecar\" SHA-256 was 3333333333333333333333333333333333333333333333333333333333333333, manifest declares 2222222222222222222222222222222222222222222222222222222222222222"
    },
    {
      "code": "auxiliary_artifact_missing_required",
      "message": "required auxiliary artifact \"required-model-card\" is missing at /srv/index/model-card.json"
    }
  ],
  "warnings": [],
  "skipped_checks": []
}
```

With `--features sqlite`, the `sqlite verify` and `sqlite activate` subcommands
add a local cache/audit log plus one active-manifest pointer. This is not a
full named registry. `sqlite verify --use-cache` reuses only reports whose
manifest, verification options, artifact bytes, row-identity bytes,
calibration profile bytes, encoder distortion profile bytes, and declared
auxiliary artifact states/bytes still match; otherwise it runs fresh
verification and stores a new report. `sqlite activate --force` writes the
active pointer even when verification fails, emits a `sqlite_activation_forced`
warning in JSON output, and exits zero because it did mutate activation state.
