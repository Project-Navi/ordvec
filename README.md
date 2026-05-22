# ordvec

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Training-free ordinal & sign quantization for vector retrieval.

`ordvec` is a small, dependency-light Rust crate for compressed
nearest-neighbour search over high-dimensional embeddings. It is
**data-oblivious**: no training, no rotation, no codebook. Norms are
analytical. There are **no system dependencies** — no BLAS, no
`ndarray`, no `faer`.

## Substrate families

- **`RankIndex`** — full-precision rank vectors (`u16` per coordinate).
- **`RankQuantIndex`** — ranks bucketed into `1 << bits` equal-width
  bins, `bits` bits per coordinate (`dim * bits / 8` bytes/doc). Both a
  symmetric (Spearman) and asymmetric (float-query LUT) scorer.
- **`BitmapIndex`** — a top-bucket bitmap per document (one bit per
  coordinate); scoring is `popcount(Q AND D)`, a coarsened rank overlap.
- **`SignBitmapIndex`** — a sign bitmap per document for sign-cosine
  candidate generation, feeding an exact rerank stage.

## Provenance

Extracted from [turbovec](https://github.com/RyanCodrai/turbovec) (MIT)
— the ordinal/sign retrieval substrate, lifted out as a standalone,
zero-system-dependency crate. The full development history of these
modules is preserved in this repository's git log. See
[LICENSE](LICENSE) for attribution.

## Reproducible benchmark

The head-to-head benchmark generates a seeded synthetic corpus
in-process, so the quality numbers (R@10, candidate-recall, bytes/vec)
are regenerable from a clean checkout with no external corpus file:

```sh
cargo run --release --example bench_rank
```

A committed capture of one run lives at
[`benchmarks/rank_modes_results.txt`](benchmarks/rank_modes_results.txt).

## License

MIT — see [LICENSE](LICENSE).
