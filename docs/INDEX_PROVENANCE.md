# Index file provenance

`ordvec` persists indexes as `.tvr` / `.tvrq` / `.tvbm` / `.tvsb` files and
reloads them through `Rank::load`, `RankQuant::load`, `Bitmap::load`, and
`SignBitmap::load`. This note states exactly **what the loaders guarantee and
what they do not**, so you can decide whether an index file needs out-of-band
verification before you load it. For the byte layout and versioning of the
persisted formats themselves, see [`PERSISTED_FORMAT.md`](PERSISTED_FORMAT.md).
Format compatibility expectations are covered by the
[pre-1.0 compatibility policy](compatibility-policy.md).

## What the loaders validate

The loaders treat the byte stream as **untrusted input** and reject malformed
files without panicking, aborting, or silently accepting garbage:

- magic + version checks before any allocation;
- fallible allocation (`try_reserve_exact`) — an attacker-controlled length
  field returns `InvalidData`, never an OOM abort;
- all payload sizes computed with `checked_mul`; overflow is an error;
- a 128 GiB `MAX_PAYLOAD` cap plus `MAX_VECTORS` / `MAX_DIM` caps;
- an exact file-length match (trailing bytes or short files are rejected);
- per-row **structural** invariants: `Rank` rows must be a true permutation of
  `[0, dim)`, `RankQuant` rows must satisfy constant composition, `Bitmap` rows
  must have exactly `n_top` bits set.

A file that survives all of this is **structurally well-formed**. The four
loaders are exercised by `cargo fuzz` (the `load_*` targets).

## What the loaders do NOT validate

The loaders validate **structure, not origin or truth**:

- They do **not** authenticate who produced the file or whether it was modified
  in transit or at rest. There is no signature, MAC, or checksum in the format.
- A **structurally valid but semantically poisoned** index — one whose ranks,
  buckets, or bitmaps were crafted to bias retrieval — passes every check and
  returns attacker-influenced results. This is a *provenance* problem, not a
  parser problem (THREAT-DESER-002 / THREAT-POISON-\* in
  [../THREAT_MODEL.md](../THREAT_MODEL.md)).

## Guidance for deployments where index files cross a trust boundary

If you load index files that were produced elsewhere, transferred over a
network, or stored on shared/mutable infrastructure, verify them **before**
loading. The lockstep `ordvec-manifest` crate provides a sidecar verifier for
that pre-load step:

```sh
cargo run -p ordvec-manifest --features cli -- verify --manifest path/to/index.manifest.json
```

The `create` command emits default-verifiable manifests by default: artifact
and row-identity paths must resolve under the output manifest directory. If a
deployment intentionally keeps those files outside that directory, create with
`--allow-path-escape` and verify with the matching path-policy flag.
`create` can also bind caller-owned sidecars with `--aux NAME=PATH` for
required artifacts and `--optional-aux NAME=PATH` for optional artifacts.
Rust callers can use `verify_for_load(manifest_path, VerifyOptions)` to get a
`VerifiedLoadPlan` containing the canonical artifact path, probed metadata,
row-identity summary, auxiliary artifact states, and the full verification
report, then call `require_auxiliary(name)` for sidecars that must be present
before loading. Callers that already hold a `ManifestDocument` can use
`verify_document_for_load(&document, VerifyOptions)` without re-reading the
manifest file. The plan helpers do not call an ordvec loader, pin file
descriptors, or make mutable shared storage immutable; callers still own the
final policy decision and should load from the returned paths only while the
verified files remain under their control.
`ordvec-manifest/README.md` shows the intended verify-then-immediate-load
pattern. If another process can mutate the manifest, index, row map, or sidecar
between verification and load, re-run `verify_for_load` at the load boundary or
load from immutable storage or a caller-owned loading path that pins bytes.

The manifest verifier checks:

- the index bytes against the manifest's SHA-256 digest;
- the fixed index header metadata (`Rank`, `RankQuant`, `Bitmap`, or
  `SignBitmap`) without allocating the payload;
- declared dimension, vector count, bytes-per-vector, format version, and
  format parameters against the probed metadata;
- the `embedding` block as the encoder/vector representation that produced
  the index artifact;
- row identity, either explicit `row_id_identity` or a strict JSONL row map
  whose `row_id` equals the zero-based line number and whose `db_id` is
  non-empty, NUL-free, and unique by default;
- declared auxiliary artifacts, checking each caller-named sidecar's path,
  SHA-256 digest, byte length, and configured byte ceiling under the same
  default path policy as the primary index artifact;
- optional `encoder_distortion` profile references, checking scoped metric
  names, encoder identity, finite declared/estimated bounds, evidence kind,
  path/hash integrity for side artifacts, and optional calibration-profile
  linkage;
- optional `calibration` profile references, checking profile identity,
  path/hash integrity, encoder identity, and ordinalization compatibility;
- attestation **shape** only: predicate type, builder id when present, and at
  least one subject SHA-256 matching the artifact when attestations are
  supplied.

Auxiliary artifacts are for application-owned sidecars such as metadata,
secondary indexes, or stores that a caller intends to load together with the
ordvec index. The verifier does not interpret those bytes; it only reports
whether declared required members were verified, whether optional members were
present or absent, and whether any declared member failed path, size, or digest
checks or exceeded the configured auxiliary artifact byte limit. Callers should
load sidecars only after the relevant declaration is verified.

A consuming database can use `row_id_identity` for the ordvec vector row count
and declare its ID sidecar file as a required auxiliary artifact (e.g. `app.ids`).
The `u64` IDs remain caller-owned sidecar bytes. Do not model the ID sidecar
as JSONL row identity: v1 JSONL row identity is UUID-only, and generic row-map
ID formats are deferred until there is a separate schema contract for them. The
reserved `row_identity.db` block is rejected in v1 because it is not byte-bound
or path-checked.

When present, `encoder_distortion` records a scoped encoder geometry profile:
source metric, embedding metric, lower/upper distortion-style bounds when
declared, empirical violation statistics when available, evidence kind, and
hashes tying any profile side artifact to the manifest. This is deliberately
not a claim that the encoder is globally bi-Lipschitz over language. The
verifier checks that the declaration is finite, scoped, identity-compatible
with `embedding`, and byte-bound to any referenced profile; it does not estimate
the profile or promote empirical evidence into a theorem. If
`calibration_profile_id` is present, the verifier also checks that it names the
manifest's `calibration.profile_id`.

When present, `calibration` binds an index artifact to a hashed ordinal profile
used to interpret overlap, bucket, sign, or rank evidence under a calibrated
null. The verifier checks profile identity, path/hash integrity, encoder
identity, and ordinalization compatibility; it does not judge whether the null
model is scientifically adequate and does not compute likelihood ratios or tail
probabilities. Calibration profiles must match the encoder identity declared by
`embedding`; cross-encoder calibration is rejected by default. The
`uniform_hypergeometric` null is reserved for top-K overlap calibration and is
not accepted for bucket, sign, or rank-position ordinalizations.

Recipes that consume sidecar manifests should run the verifier or plan helper
first, then load/search/rerank only if verification succeeds.

You can also verify using whatever your deployment already trusts:

- a checksum manifest (e.g. SHA-256) recorded by the build that produced the
  index, verified at load time;
- your artifact store's integrity controls;
- a signature / attestation layer (e.g. Sigstore, GitHub artifact attestations)
  over the index files.

`ordvec-manifest` is not a trust oracle. It does **not** sign, manage keys,
call networks, mutate index files, change the C ABI, or decide whether a
builder or signer is trusted. `ordvec` deliberately ships **no** built-in
signing/MAC layer today: without a concrete deployment requiring it, an
in-format crypto layer would add key management with no clear owner.
