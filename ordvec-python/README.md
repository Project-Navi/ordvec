# ordvec (Python)

Python bindings for [`ordvec`](https://github.com/Fieldnote-Echo/ordvec) — a
training-free **ordinal & sign** vector-quantization library for compressed
nearest-neighbour retrieval over high-dimensional embeddings. Pure-Rust core,
zero system dependencies; SIMD-accelerated at runtime (AVX-512 / AVX2 / scalar).

```python
import numpy as np
import ordvec

q = ordvec.RankQuant(1024, 2)          # 1024-dim, 2 bits/coord
q.add(np.random.randn(10_000, 1024).astype(np.float32))
# asymmetric: full-precision float queries vs bucketed docs (recommended)
scores, ids = q.search_asymmetric(np.random.randn(8, 1024).astype(np.float32), k=10)
```

## Classes

| Class | Purpose |
|-------|---------|
| `Rank` | Full-precision rank vectors (u16 per coordinate). |
| `RankQuant` | Bucketed ranks, `bits` ∈ {1, 2, 4}; symmetric + asymmetric (float-query LUT) scoring. |
| `Bitmap` | Constant-weight top-bucket bitmap per document; `popcount(Q AND D)` candidate scoring. |
| `SignBitmap` | Sign bitmap for sign-cosine candidate generation; separate from the constant-weight bitmap theorem. |

## Two-stage retrieval (subset rerank)

A `Bitmap` / `SignBitmap` probe yields a candidate shortlist that
`RankQuant.search_asymmetric_subset(query, candidates, k)` reranks exactly:

```python
cands = bm.top_m_candidates(query, m=256)          # uint32 shortlist
scores, ids = rq.search_asymmetric_subset(query, cands, k=10)
```

Both returned arrays have length **`min(k, len(candidates))`**, not `k`. When
`k > len(candidates)` the result is silently capped to the candidate count — the
subset path never pads with sentinel rows. If you assemble a fixed-width
`(n_q, k)` result buffer, size each row by its candidate count rather than
assuming `k` rows back.

## Theory and calibration

`Bitmap` exposes the constant-weight top-bucket overlap statistic formalized in
[`ordvec-formalization`](https://github.com/Fieldnote-Echo/ordvec-formalization).
In that finite Lean model, literal bitmap overlap is the query-preserving
quotient statistic, an overlap threshold is Bayes-optimal under explicit
monotone-overlap assumptions, and the idealized uniform constant-weight null
calibrates that threshold by the hypergeometric upper tail.

This is not a deployment guarantee for every encoder or corpus. Real-corpus
recall, monotonicity, and null fit remain empirical diagnostics.

## Installation

```bash
pip install ordvec
```

Wheels target CPython 3.10+ (abi3) and require `numpy>=2.2`. Building from
source needs a Rust toolchain (MSRV 1.89) and
[maturin](https://www.maturin.rs/).

## Type stubs

The package ships hand-written type stubs (`_ordvec.pyi`) and a `py.typed`
marker, so editors and `mypy` get full signatures for the four index classes,
the module-level rank-math primitives, and the `MAX_*` constants — the abi3
native module is otherwise opaque to static analysis.

## Provenance & license

The `ordvec` Python bindings were developed within turbovec, factored out
into this standalone package. turbovec
([MIT](https://github.com/RyanCodrai/turbovec), by Ryan Codrai) is credited as
the origin project.

Dual-licensed under either of
[MIT](https://github.com/Fieldnote-Echo/ordvec/blob/main/LICENSE-MIT) or
[Apache-2.0](https://github.com/Fieldnote-Echo/ordvec/blob/main/LICENSE-APACHE-2.0)
at your option.
