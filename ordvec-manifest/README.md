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
published release, install the binary with `cargo install ordvec-manifest`.
From a workspace checkout, use the CLI with `cargo run -p ordvec-manifest --`.
Library-only consumers that do not need the CLI can depend on the crate with
`default-features = false`.

```sh
ordvec-manifest create \
  --index path/to/index.ovrq \
  --row-id-is-identity \
  --aux app.ids=path/to/ids.bin \
  --embedding-model bge-small-en-v1.5 \
  --out path/to/index.manifest.json

ordvec-manifest verify --manifest path/to/index.manifest.json
```

From a workspace checkout, prefix the same commands with
`cargo run -p ordvec-manifest --`.

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
let _app_ids = plan.require_auxiliary("app.ids")?;
let index = ordvec::RankQuant::load(plan.artifact_path())?;
```

Concrete sidecar-backed bundle pattern:

```text
docs.odb/
  manifest.json
  index.ovrq
  ids.bin
```

The application writes the ordvec index and its own sidecar bytes first:

```rust
use std::{fs, path::Path};

fn write_bundle(
    index: &ordvec::RankQuant,
    doc_ids: &[u64],
    bundle: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(bundle)?;
    index.write(bundle.join("index.ovrq"))?;

    let mut id_bytes = Vec::with_capacity(doc_ids.len() * std::mem::size_of::<u64>());
    for id in doc_ids {
        id_bytes.extend_from_slice(&id.to_le_bytes());
    }
    fs::write(bundle.join("ids.bin"), id_bytes)?;
    Ok(())
}
```

Then create and verify a manifest that binds both files:

```sh
cargo run -p ordvec-manifest -- create \
  --index docs.odb/index.ovrq \
  --row-id-is-identity \
  --aux app.ids=docs.odb/ids.bin \
  --embedding-model bge-small-en-v1.5 \
  --out docs.odb/manifest.json

cargo run -p ordvec-manifest -- verify \
  --manifest docs.odb/manifest.json \
  --json
```

The load side verifies the bundle before any caller-owned sidecar parsing:

```rust
let plan = ordvec_manifest::verify_for_load(
    &bundle.join("manifest.json"),
    ordvec_manifest::VerifyOptions::default(),
)?;

let metadata = plan.metadata();
assert_eq!(metadata.vector_count, expected_rows);

let index = ordvec::RankQuant::load(plan.artifact_path())?;
let ids_path = plan.require_auxiliary("app.ids")?;
let ids_bytes = std::fs::read(ids_path)?;
let doc_ids = parse_caller_owned_ids(&ids_bytes)?;
```

`ordvec-manifest` owns the path, size, SHA-256, and index-metadata checks.
The application still owns the `ids.bin` schema, count check, duplicate policy,
endianness, and any database reconstruction rules. If `ids.bin` is modified
after manifest creation, verification fails with a stable auxiliary artifact
size or digest code before the caller parses those bytes.

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
- auxiliary artifact bytes per declared file: bounded by the
  manifest-declared `file_size_bytes` on verify and by the observed file
  size on create; the flat cap is an opt-in ceiling, unbounded by default
  (`auxiliary_artifact_file_too_large`);
- primary index artifact bytes: bounded by the manifest-declared
  `file_size_bytes` on verify; the flat cap is an opt-in ceiling, unbounded
  by default (`artifact_file_too_large`);
- calibration profile artifact bytes: bounded by the declared
  `file_size_bytes`; flat cap opt-in, unbounded by default
  (`calibration_profile_too_large`);
- encoder distortion profile artifact bytes: bounded by the declared
  `file_size_bytes`; flat cap opt-in, unbounded by default
  (`encoder_distortion_profile_too_large`);
- collected report issues: 1,024, after which a
  `verification_report_issue_limit_exceeded` issue is emitted;
- SQLite cached report JSON: 4 MiB (`sqlite_cached_report_too_large`).

The CLI exposes matching override flags on `inspect`, `verify`, `create`,
`sqlite verify`, and `sqlite activate`: `--max-manifest-bytes`,
`--max-row-map-line-bytes`, `--max-row-map-rows`,
`--max-row-map-tracked-id-bytes`, `--max-auxiliary-artifacts`,
`--max-auxiliary-artifact-bytes`, `--max-index-artifact-bytes`,
`--max-calibration-profile-bytes`,
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
| primary index artifact bytes | `artifact_file_too_large` | n/a |
| calibration profile artifact bytes | `calibration_profile_too_large` | n/a |
| encoder distortion profile artifact bytes | `encoder_distortion_profile_too_large` | n/a |
| collected verification report issues | `verification_report_issue_limit_exceeded` | n/a |
| SQLite cached report JSON bytes | n/a | `sqlite_cached_report_too_large` |

Oversized byte-limit overrides that cannot be represented safely by the
bounded in-memory reader fail before reading with the same stable
`ManifestError::code()` as the corresponding byte limit. A
`max_report_issues` override of `0` suppresses detail issues and returns only
the `verification_report_issue_limit_exceeded` sentinel when any issue would
otherwise be reported. These limits bound metadata parsing and report/cache
growth; hashing the primary index remains proportional to the artifact bytes
being verified. SQLite cache-key construction treats an over-limit calibration
or encoder distortion profile as non-cacheable and reruns verification instead
of reusing a previously cached report.

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

A consuming database can keep the ordvec row identity as
`RowIdentity::RowIdIdentity { row_count }` and declare its ID sidecar file as a
required auxiliary artifact (e.g. `app.ids`). That makes the vector row count an
ordvec invariant while leaving the caller's `u64` document IDs as caller-owned
sidecar bytes. Do not encode the ID sidecar as `RowIdentity::Jsonl`: v1 JSONL
row identity is UUID-oriented (`id_kind = "uuid"`), and generic row-map ID
formats are intentionally deferred to
[#145](https://github.com/Project-Navi/ordvec/issues/145). The reserved
`row_identity.db` metadata block is rejected in v1 because it is not byte-bound
or path-checked.

Stable row-identity boundary codes:

| Condition | Verification report code |
| --- | --- |
| JSONL row identity declares an ID kind other than `uuid` | `row_identity_id_kind_unsupported` |
| JSONL row identity includes reserved `row_identity.db` metadata | `row_identity_db_unsupported` |
| JSONL `db_id` / `parent_id` is empty | `row_identity_db_id_empty` / `row_identity_parent_id_empty` |
| JSONL `db_id` / `parent_id` contains NUL | `row_identity_db_id_contains_nul` / `row_identity_parent_id_contains_nul` |
| JSONL `db_id` / `parent_id` is not a UUID | `row_identity_db_id_invalid_uuid` / `row_identity_parent_id_invalid_uuid` |

The unified JSON report carries per-sidecar audit fields. A successful
auxiliary artifact verification includes the manifest path, resolved/canonical
paths, declared digest/length, and observed digest/length:

Stable sidecar report fields:

| Field | Meaning |
| --- | --- |
| `auxiliary_artifacts[].name` | Caller-owned sidecar name from the manifest. |
| `manifest_path` | Manifest-declared relative path. |
| `resolved_path` / `canonical_path` | Path used for verification and its canonical form when available. |
| `expected_sha256` / `expected_size_bytes` | Manifest-declared digest and byte length. |
| `sha256` / `size_bytes` | Observed digest and byte length when bytes could be read. |
| `required` | Whether absence is a verification error. |
| `state` | One of `verified`, `optional_absent`, `missing_required`, or `failed`. |
| `reason_code` | Stable null-or-string reason for any non-verified state, or the first failure reason. |

Stable sidecar states:

| `state` | `reason_code` | Report outcome |
| --- | --- | --- |
| `verified` | `null` | The declared sidecar was present and matched path policy, size, and digest. |
| `optional_absent` | `auxiliary_artifact_optional_absent` | The optional sidecar was absent; this is not an error. |
| `missing_required` | `auxiliary_artifact_missing_required` | A required sidecar was absent and verification fails. |
| `failed` | Code-specific | Path policy, hashing, size, digest, or limit validation failed. |

Common `failed` reason codes include `auxiliary_artifact_path_empty`,
`auxiliary_artifact_base_dir_unavailable`,
`auxiliary_artifact_path_unavailable`,
`auxiliary_artifact_path_escape_rejected`,
`auxiliary_artifact_file_too_large`,
`auxiliary_artifact_file_size_mismatch`, and
`auxiliary_artifact_sha256_mismatch`. `errors[].code` and `warnings[].code`
carry the same stable code namespace. `skipped_checks[]` is machine-readable
and records checks that were intentionally not run, such as
`attestations_absent`.

```json
{
  "ok": true,
  "checked_at": "2026-06-03T17:20:00Z",
  "manifest_id": "urn:uuid:11111111-1111-4111-8111-111111111111",
  "artifact": {
    "manifest_path": "index.ovrq",
    "observed_path": "index.ovrq",
    "canonical_path": "/srv/index/index.ovrq",
    "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "size_bytes": 4096,
    "metadata": null
  },
  "auxiliary_artifacts": [
    {
      "name": "app.sidecar",
      "manifest_path": "app.sidecar.json",
      "resolved_path": "/srv/index/app.sidecar.json",
      "canonical_path": "/srv/index/app.sidecar.json",
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
    "manifest_path": "index.ovrq",
    "observed_path": "index.ovrq",
    "canonical_path": "/srv/index/index.ovrq",
    "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "size_bytes": 4096,
    "metadata": null
  },
  "auxiliary_artifacts": [
    {
      "name": "app.sidecar",
      "manifest_path": "app.sidecar.json",
      "resolved_path": "/srv/index/app.sidecar.json",
      "canonical_path": "/srv/index/app.sidecar.json",
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
      "message": "auxiliary artifact \"app.sidecar\" SHA-256 was 3333333333333333333333333333333333333333333333333333333333333333, manifest declares 2222222222222222222222222222222222222222222222222222222222222222"
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
