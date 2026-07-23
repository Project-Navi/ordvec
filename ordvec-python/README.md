# ordvec (Python)

Python bindings for [`ordvec`](https://github.com/Project-Navi/ordvec) — a
training-free **ordinal & sign** vector-quantization library for compressed
nearest-neighbour retrieval over high-dimensional embeddings. Pure-Rust core,
zero system dependencies; SIMD-accelerated at runtime (AVX-512 / AVX2 / scalar).

## Quickstart

```bash
python -m pip install --upgrade ordvec
```

```python
import numpy as np
from ordvec import RankQuant

documents = np.array([
    [8, 7, 6, 5, 4, 3, 2, 1],
    [1, 2, 3, 4, 5, 6, 7, 8],
    [8, 1, 7, 2, 6, 3, 5, 4],
], dtype=np.float32)
query = np.array([[8, 7, 6, 5, 4, 3, 2, 1]], dtype=np.float32)

index = RankQuant(dim=8, bits=1)
index.add(documents)
scores, ids = index.search_asymmetric(query, k=1)
print(f"top document: {ids[0, 0]} (score {scores[0, 0]:.3f})")
```

```text
top document: 0 (score 0.396)
```

Persist and reopen without the original float corpus:

```python
index.write("quickstart.ovrq")
reopened = RankQuant.load("quickstart.ovrq")
_, reopened_ids = reopened.search_asymmetric(query, k=1)
assert reopened_ids[0, 0] == 0
```

## Classes

| Class | Purpose |
|-------|---------|
| `Rank` | Full-precision rank vectors (u16 per coordinate). |
| `RankQuant` | Bucketed ranks, `bits` ∈ {1, 2, 4}; symmetric + asymmetric (float-query LUT) scoring. |
| `Bitmap` | Constant-weight top-bucket bitmap per document; `popcount(Q AND D)` candidate scoring. |
| `SignBitmap` | Sign bitmap for sign-cosine candidate generation; separate from the constant-weight bitmap theorem. |

The Rust crate's `b = 8` RankQuant evidence/refinement width is not exposed
through the v0.6.0 Python `RankQuant` constructor and cannot be persisted to
`.ovrq`; use `bits` 1, 2, or 4 from Python.

## Two-stage retrieval (subset rerank)

A `Bitmap` / `SignBitmap` probe yields a candidate shortlist that
`RankQuant.search_asymmetric_subset(query, candidates, k)` reranks exactly.
Continuing from the quickstart's `documents` and `query`, tile the tiny rows to
the bitmap kernel's 64-coordinate minimum:

```python
from ordvec import Bitmap

documents64 = np.tile(documents, (1, 8))
query64 = np.tile(query, (1, 8))
reranker = RankQuant(dim=64, bits=1)
reranker.add(documents64)
probe = Bitmap(dim=64, n_top=16)
probe.add(documents64)
candidates = probe.top_m_candidates(query64[0], m=2)
scores, ids = reranker.search_asymmetric_subset(query64[0], candidates, k=1)
assert ids[0] == 0
```

Both returned arrays have length **`min(k, len(candidates))`**, not `k`. When
`k > len(candidates)` the result is silently capped to the candidate count — the
subset path never pads with sentinel rows. If you assemble a fixed-width
`(n_q, k)` result buffer, size each row by its candidate count rather than
assuming `k` rows back.

## Theory and calibration

`Bitmap` exposes the constant-weight top-bucket overlap statistic formalized in
[`ordvec-formalization`](https://github.com/Project-Navi/ordvec-formalization).
In that finite Lean model, literal bitmap overlap is the query-preserving
quotient statistic, an overlap threshold is Bayes-optimal under explicit
monotone-overlap assumptions, and the idealized uniform constant-weight null
calibrates that threshold by the hypergeometric upper tail.

This is not a deployment guarantee for every encoder or corpus. Real-corpus
recall, monotonicity, and null fit remain empirical diagnostics.

## Installation details

The v0.6.0 wheel matrix covers CPython 3.10+ (abi3) on manylinux/glibc
x86_64 and aarch64, macOS Apple Silicon, and Windows x64, and requires
`numpy>=2.2`. Intel Mac and musl/Alpine installations fall back to a source
build and require Rust 1.89 plus [maturin](https://www.maturin.rs/). See the
[artifact platform matrix](https://github.com/Project-Navi/ordvec/blob/v0.6.0/docs/artifact-platform-matrix.md)
for the complete release surface.

## Safety contract

The Python binding releases the GIL while Rust searches, scores, and mutates
indexes. Inputs that cross a GIL-released call are copied into Rust-owned
buffers first, so ordinary Python in-place NumPy mutation from another thread
cannot race the detached Rust scan. Large calls may temporarily require an
additional input-sized buffer.
The cross-language ownership and lifetime contract is maintained in
[`docs/bindings-safety.md`](https://github.com/Project-Navi/ordvec/blob/v0.6.0/docs/bindings-safety.md)
for this release line.

## Type stubs

The package ships hand-written type stubs (`_ordvec.pyi`) and a `py.typed`
marker, so editors and `mypy` get full signatures for the four index classes,
the module-level rank-math primitives, and the `MAX_*` constants — the abi3
native module is otherwise opaque to static analysis.

## Provenance & license

The `ordvec` Python package's active upstream, implementation history, issues,
releases, and governance live in `Project-Navi/ordvec`.

Courtesy note: ordvec was developed using the early
[turbovec](https://github.com/RyanCodrai/turbovec) project context as a
rapid-development scaffold, with thanks to that lineage. It is not a source
fork of turbovec.

Dual-licensed under either of
[MIT](https://github.com/Project-Navi/ordvec/blob/main/LICENSE-MIT) or
[Apache-2.0](https://github.com/Project-Navi/ordvec/blob/main/LICENSE-APACHE-2.0)
at your option.
