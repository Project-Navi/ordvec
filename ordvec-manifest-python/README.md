# ordvec-manifest

Python bindings for the `ordvec-manifest` verifier.

Install from PyPI:

```bash
python -m pip install ordvec-manifest
```

Import as `ordvec_manifest`. The package exposes the Rust manifest verifier as
dict-returning Python functions:

```python
import ordvec_manifest

report = ordvec_manifest.verify_manifest("index.manifest.json")
if not report["ok"]:
    raise RuntimeError(report["errors"])
```

Create manifests with caller-owned sidecars by passing dictionaries with
`name`, `path`, and optional `required`:

```python
manifest = ordvec_manifest.create_manifest(
    "index.tvrq",
    "index.manifest.json",
    "bge-small-en-v1.5",
    row_id_is_identity=True,
    auxiliary_artifacts=[
        {"name": "ordinaldb.ids", "path": "ids.bin"},
        {"name": "optional.stats", "path": "stats.json", "required": False},
    ],
)
plan = ordvec_manifest.verify_for_load("index.manifest.json")
```

For OrdinalDB v0.1, keep `row_id_identity` for the ordvec row count and declare
`ids.bin` as required auxiliary artifact name `ordinaldb.ids`. Do not encode
`ids.bin` as JSONL row identity; the v1 JSONL row-map contract is UUID-only.

The verifier checks manifest shape, declared artifact digests and sizes, probed
ordvec index metadata, row identity, auxiliary artifact state, optional
calibration profiles, optional encoder-distortion profiles, and attestation
shape metadata. It does not sign artifacts, manage keys, call networks, mutate
index files, or decide deployment policy.
