# Persisted Index Format

This document is the compatibility contract for ordvec persisted index files.
It covers the primitive index artifacts only: `.ovr`, `.ovrq`, `.ovbm`, and
`.ovsb`. It does not define a database, transaction log, replication protocol,
provenance system, checksum manifest, signature, or trust policy.

`RankQuantFastscan` can write and load `.ovfs` (magic `OVFS`) through its direct
API, but `.ovfs` is intentionally outside this v1 primitive-format,
`probe_index_metadata()`, and `ordvec-manifest` contract. Until metadata-probe
and manifest support are promoted, callers should treat `.ovfs` as a
specialized direct-load artifact and bind it with application-owned checksums or
attestations when it crosses a trust boundary. The direct `.ovfs` loader still
validates the payload before search: real document bytes must be 4-bit FastScan
codes, every row must satisfy b=2 constant composition, and block-tail padding
must be zero.

All integer fields are little-endian. Each format has one fixed header followed
by one contiguous payload. The payload must consume the rest of the file
exactly; v1 files have no footer, reserved trailing bytes, or extension block.

## Compatibility Policy

The current on-disk format version is `1` for every primitive family covered
here.
Within the v1 contract:

- loaders and `probe_index_metadata()` reject unknown magic, unsupported
  versions, malformed header fields, impossible dimensions, impossible row
  counts, payload-size overflow, short payloads, and trailing bytes;
- writers emit only v1 files matching the layouts below;
- `probe_index_metadata()` is the allocation-resistant preflight path for host
  stores and sidecar manifests;
- full loaders additionally validate payload row invariants before search or
  SIMD paths can observe the state.

A breaking persisted-format change requires one of:

- a new magic value;
- a format-version bump with documented rejection or migration behavior;
- a clearly documented migration tool that rewrites old bytes into the new
  layout.

Examples of breaking changes include changing endianness, changing fixed header
order or width, adding a trailing section, changing RankQuant packing order,
changing row-invariant interpretation, changing the primitive score assigned to
stored bytes, or assigning new semantics to an existing magic/version pair.
Strengthening rejection of malformed files is not a format break when valid v1
writer output still loads.

Rust API and release SemVer policies are tracked separately from this
byte-format contract.

## Metadata

`probe_index_metadata(path)` returns the segment descriptor host systems should
cache in their own manifests:

- `kind`: `Rank`, `RankQuant`, `Bitmap`, or `SignBitmap`;
- `format_version`: currently `1`;
- `dim`: vector dimension declared by the file;
- `vector_count`: number of stored documents;
- `bytes_per_vec`: payload bytes per stored document;
- `params`: format-specific parameters such as RankQuant `bits` or Bitmap
  `n_top`;
- `file_size_bytes`: total observed file size.

In v0.6.0, `probe_index_metadata(path)` rejects `OVFS` with an unsupported
metadata-probe error rather than returning a partial descriptor. Load `.ovfs`
only through `RankQuantFastscan::load` unless and until the FastScan metadata
contract is promoted in a later minor release; the direct loader rejects
invalid nibbles, non-canonical tail padding, and b=2 composition violations.

Example external segment entry:

```json
{
  "path": "segments/shard-0007/index.ovrq",
  "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "metadata": {
    "kind": "RankQuant",
    "format_version": 1,
    "dim": 1024,
    "vector_count": 1250000,
    "bytes_per_vec": 256,
    "params": { "bits": 2 },
    "file_size_bytes": 320000014
  }
}
```

The metadata describes byte shape, not trust. If an artifact crosses a trust
boundary, bind the file bytes to a checksum, signature, attestation, or
application-owned manifest before loading.

## Score Semantics

The `format_version` is also the primitive score-semantics version for the
bytes under that magic. A valid v1 artifact must keep the same interpretation
of stored rank, bucket, bitmap, or sign bytes when computing per-row primitive
scores. A future change that makes identical persisted bytes produce different
primitive scores requires a new magic, a version bump, or documented migration
or rejection behavior.

This contract does not freeze composed retrieval policy. Backend choice,
candidate-count selection, and ordering among equal scores are tracked outside
the byte-format contract unless they change the primitive score assigned to a
persisted row.

## Format Layouts

### Rank (`.ovr`, magic `OVR1`)

Current writers emit magic `OVR1`. Loaders also accept the legacy magic `TVR1`
(written by versions before the format rename).

Header:

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 4 | magic `OVR1` (or legacy `TVR1`) |
| 4 | 1 | format version `1` |
| 5 | 4 | `dim` as `u32` little-endian |
| 9 | 4 | `n_vectors` as `u32` little-endian |

Payload: `n_vectors * dim` `u16` values, each little-endian. Each row must be a
permutation of `[0, dim)`. `dim` must be in `[2, 65,535]`.

Probe metadata:

- `kind = Rank`
- `params = Rank`
- `bytes_per_vec = dim * 2`

### RankQuant (`.ovrq`, magic `OVRQ`)

Current writers emit magic `OVRQ`. Loaders also accept the legacy magic `TVRQ`
(written by versions before the format rename).

Header:

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 4 | magic `OVRQ` (or legacy `TVRQ`) |
| 4 | 1 | format version `1` |
| 5 | 1 | `bits` as `u8`, one of `1`, `2`, or `4` |
| 6 | 4 | `dim` as `u32` little-endian |
| 10 | 4 | `n_vectors` as `u32` little-endian |

Payload: `n_vectors * dim * bits / 8` packed bytes. Bucket codes are packed
MSB-first within each byte. For `bits = 2`, the first coordinate occupies bits
7..6 of the byte, the second coordinate bits 5..4, the third bits 3..2, and
the fourth bits 1..0.

`dim` must be in `[2, 65,535]` and divisible by both `1 << bits` and
`8 / bits`. Each row must have constant composition: exactly
`dim / (1 << bits)` coordinates in every bucket.

Probe metadata:

- `kind = RankQuant`
- `params = RankQuant { bits }`
- `bytes_per_vec = dim * bits / 8`

### Bitmap (`.ovbm`, magic `OVBM`)

Current writers emit magic `OVBM`. Loaders also accept the legacy magic `TVBM`
(written by versions before the format rename).

Header:

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 4 | magic `OVBM` (or legacy `TVBM`) |
| 4 | 1 | format version `1` |
| 5 | 4 | `dim` as `u32` little-endian |
| 9 | 4 | `n_top` as `u32` little-endian |
| 13 | 4 | `n_vectors` as `u32` little-endian |

Payload: `n_vectors * dim / 64` `u64` bitmap words, each little-endian. `dim`
must be in `[2, 65,535]` and a multiple of 64. Each row must have exactly
`n_top` bits set.

Probe metadata:

- `kind = Bitmap`
- `params = Bitmap { n_top }`
- `bytes_per_vec = dim / 8`

### SignBitmap (`.ovsb`, magic `OVSB`)

Current writers emit magic `OVSB`. Loaders also accept the legacy magic `TVSB`
(written by versions before the format rename).

Header:

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 4 | magic `OVSB` (or legacy `TVSB`) |
| 4 | 1 | format version `1` |
| 5 | 4 | `dim` as `u32` little-endian |
| 9 | 4 | `n_vectors` as `u32` little-endian |

Payload: `n_vectors * dim / 64` `u64` bitmap words, each little-endian. `dim`
must be a multiple of 64 and within `MAX_SIGN_BITMAP_DIM`. Any bit pattern is a
valid sign-bitmap row; there is no per-row popcount invariant.

Probe metadata:

- `kind = SignBitmap`
- `params = SignBitmap`
- `bytes_per_vec = dim / 8`

## Probe Versus Load

`probe_index_metadata()` validates fixed headers, parameter domains, checked
payload byte counts, and exact file length without reading payload rows. Use it
when a host system wants to inspect an artifact before allocation or before
choosing a loader.

The full loaders validate everything the probe validates and then inspect row
payload invariants:

- `Rank::load`: each row is a permutation of `[0, dim)`;
- `RankQuant::load`: each row has the required constant bucket composition;
- `Bitmap::load`: each row has exactly `n_top` bits set;
- `SignBitmap::load`: no additional row invariant exists.

`RankQuantFastscan::load` has its own direct loader path for `.ovfs`; it is not
covered by this probe-versus-load contract in v0.6.0.

Loader success is the primitive binary-safety boundary. It is not a provenance
or deployment-policy decision.
