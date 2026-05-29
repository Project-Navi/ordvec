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

With `--features sqlite`, the `sqlite verify` and `sqlite activate` subcommands
add a local cache/audit log plus one active-manifest pointer. This is not a
full named registry. `sqlite verify --use-cache` reuses only reports whose
manifest, verification options, artifact bytes, row-identity bytes, and
calibration profile bytes still match; otherwise it runs fresh verification and
stores a new report. `sqlite activate --force` writes the active pointer even
when verification fails, emits a `sqlite_activation_forced` warning in JSON
output, and exits zero because it did mutate activation state.
