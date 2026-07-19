# ordvec-manifest

Python bindings for the `ordvec-manifest` verifier.

## First verified index

```bash
python -m pip install --upgrade ordvec ordvec-manifest
```

```python
import numpy as np
from ordvec import RankQuant
import ordvec_manifest

documents = np.array([
    [8, 7, 6, 5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5, 6, 7, 8],
], dtype=np.float32)
index = RankQuant(dim=8, bits=1)
index.add(documents)
index.write("quickstart.ovrq")

ordvec_manifest.create_manifest(
    "quickstart.ovrq",
    "quickstart.manifest.json",
    "quickstart-embedding-v1",
    row_id_is_identity=True,
)
report = ordvec_manifest.verify_manifest("quickstart.manifest.json")
print(f"verified: {report['ok']}")
```

```text
verified: True
```

The package exposes the Rust manifest verifier as dict-returning Python
functions. To bind existing caller-owned sidecars, pass dictionaries with
`name`, `path`, and optional `required`:

```python
manifest = ordvec_manifest.create_manifest(
    "index.ovrq",
    "index.manifest.json",
    "bge-small-en-v1.5",
    row_id_is_identity=True,
    auxiliary_artifacts=[
        {"name": "app.ids", "path": "ids.bin"},
        {"name": "optional.stats", "path": "stats.json", "required": False},
    ],
)
plan = ordvec_manifest.verify_for_load("index.manifest.json")
```

A consuming database can keep `row_id_identity` for the ordvec row count and
declare its ID sidecar file as a required auxiliary artifact (e.g. `app.ids`).
Do not encode the ID sidecar as JSONL row identity; the v1 JSONL row-map contract is UUID-only.

The verifier checks manifest shape, declared artifact digests and sizes, probed
ordvec index metadata, row identity, auxiliary artifact state, optional
calibration profiles, optional encoder-distortion profiles, and attestation
shape metadata. It does not sign artifacts, manage keys, call networks, mutate
index files, or decide deployment policy.
