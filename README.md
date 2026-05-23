# ordvec

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)

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

ordvec is an original ordinal/sign retrieval substrate that was
developed within the [turbovec](https://github.com/RyanCodrai/turbovec)
project — an MIT-licensed vector-quantization crate by Ryan Codrai —
and factored out here as a standalone, zero-system-dependency crate.
The rank/sign modules and their tests (including the red-team
regression suites) carry their full development history in this
repository's git log. With thanks to the turbovec project, where this
substrate was built.

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

Licensed under either of

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
