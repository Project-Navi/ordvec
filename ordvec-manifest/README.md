# ordvec-manifest

Repo-local, publish=false sidecar verifier for ordvec index manifests.

It verifies index bytes, probed header metadata, row identity, optional
calibration profile references, and attestation shape before a caller loads an
ordvec index. It does not sign artifacts, manage keys, call networks, mutate
index files, decide deployment trust policy, compute calibration statistics, or
change the C ABI.

```sh
cargo run -p ordvec-manifest -- create \
  --index path/to/index.tvrq \
  --row-id-is-identity \
  --embedding-model bge-small-en-v1.5 \
  --out path/to/index.manifest.json

cargo run -p ordvec-manifest -- verify --manifest path/to/index.manifest.json
```

The schema version is `ordvec.index_manifest.v1`. Relative paths resolve from
the manifest file's directory, absolute paths are rejected by default, and
relative paths may not escape the manifest directory unless explicitly allowed.
`create` follows the same policy: by default it emits only paths that should
verify with default settings. If an artifact or JSONL row map lives outside the
manifest directory, pass `--allow-path-escape` at create time and again at
verify time.

Verification uses bounded parser/report defaults on both CLI and library paths:

- manifest JSON: 1 MiB before JSON parsing;
- row-identity JSONL line: 64 KiB;
- row-identity JSONL rows: 10,000,000;
- row-identity duplicate-tracking `db_id` bytes: 64 MiB;
- collected report issues: 1,024, after which a
  `verification_report_issue_limit_exceeded` issue is emitted;
- SQLite cached report JSON: 4 MiB.

The CLI exposes matching override flags on `inspect`, `verify`, `create`,
`sqlite verify`, and `sqlite activate`: `--max-manifest-bytes`,
`--max-row-map-line-bytes`, `--max-row-map-rows`,
`--max-row-map-tracked-id-bytes`, `--max-report-issues`, and
`--max-cached-report-bytes`. Library callers can override the same ceilings via
`VerifyOptions::limits`.

Stable limit codes:

| Limit surface | Verification report code | `ManifestError::code()` |
| --- | --- | --- |
| manifest JSON bytes | n/a | `manifest_file_too_large` |
| row-identity JSONL line bytes | `row_identity_line_too_large` | `row_identity_line_too_large` |
| row-identity JSONL rows | `row_identity_row_count_limit_exceeded` | `row_identity_row_count_limit_exceeded` |
| row-identity duplicate-tracking `db_id` bytes | `row_identity_duplicate_tracking_limit_exceeded` | `row_identity_duplicate_tracking_limit_exceeded` |
| collected verification report issues | `verification_report_issue_limit_exceeded` | n/a |
| SQLite cached report JSON bytes | n/a | `sqlite_cached_report_too_large` |

Oversized byte-limit overrides that cannot be represented safely by the
bounded in-memory reader fail before reading with the same stable
`ManifestError::code()` as the corresponding byte limit. These limits bound
metadata parsing and report/cache growth; hashing an index or calibration
profile is still proportional to the artifact bytes being verified.

With `--features sqlite`, the `sqlite verify` and `sqlite activate` subcommands
add a local cache/audit log plus one active-manifest pointer. This is not a
full named registry. `sqlite verify --use-cache` reuses only reports whose
manifest, verification options, artifact bytes, row-identity bytes, and
calibration profile bytes still match; otherwise it runs fresh verification and
stores a new report. `sqlite activate --force` writes the active pointer even
when verification fails, emits a `sqlite_activation_forced` warning in JSON
output, and exits zero because it did mutate activation state.
