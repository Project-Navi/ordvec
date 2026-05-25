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
| `Bitmap` | Top-bucket bitmap per document; `popcount(Q AND D)` candidate scoring. |
| `SignBitmap` | Sign bitmap for sign-cosine candidate generation. |

## Installation

```bash
pip install ordvec
```

Wheels target CPython 3.10+ (abi3). Building from source needs a Rust toolchain
(MSRV 1.89) and [maturin](https://www.maturin.rs/).

## Provenance & license

The `ordvec` Python bindings were developed within turbovec, factored out
into this standalone package. turbovec
([MIT](https://github.com/RyanCodrai/turbovec), by Ryan Codrai) is credited as
the origin project.

Dual-licensed under either of
[MIT](https://github.com/Fieldnote-Echo/ordvec/blob/main/LICENSE-MIT) or
[Apache-2.0](https://github.com/Fieldnote-Echo/ordvec/blob/main/LICENSE-APACHE)
at your option.
