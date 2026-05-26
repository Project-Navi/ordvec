# Index file provenance

`ordvec` persists indexes as `.tvr` / `.tvrq` / `.tvbm` / `.tvsb` files and
reloads them through `Rank::load`, `RankQuant::load`, `Bitmap::load`, and
`SignBitmap::load`. This note states exactly **what the loaders guarantee and
what they do not**, so you can decide whether an index file needs out-of-band
verification before you load it.

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
loading using whatever your deployment already trusts:

- a checksum manifest (e.g. SHA-256) recorded by the build that produced the
  index, verified at load time;
- your artifact store's integrity controls;
- a signature / attestation layer (e.g. Sigstore, GitHub artifact attestations)
  over the index files.

`ordvec` deliberately ships **no** built-in signing/MAC layer today: without a
concrete deployment requiring it, an in-format crypto layer would add key
management with no clear owner. A sidecar verifier (e.g. an `ordvec verify`
utility, or an external HMAC/BLAKE3 manifest) can be added later **without a
file-format change** if a real deployment needs tamper-evidence.
