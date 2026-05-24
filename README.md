# ordvec

[![CI](https://github.com/Fieldnote-Echo/ordvec/actions/workflows/ci.yml/badge.svg)](https://github.com/Fieldnote-Echo/ordvec/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![MSRV](https://img.shields.io/badge/MSRV-1.89-blue.svg)](#minimum-supported-rust-version)

<!-- Add at the first crates.io release:
[![Crates.io](https://img.shields.io/crates/v/ordvec.svg)](https://crates.io/crates/ordvec)
[![docs.rs](https://docs.rs/ordvec/badge.svg)](https://docs.rs/ordvec)
-->

Training-free ordinal & sign quantization for vector retrieval.

`ordvec` is a small, dependency-light Rust crate for compressed
nearest-neighbour search over high-dimensional embeddings. It is
**data-oblivious**: no training, no rotation, no codebook. Norms are
analytical. There are **no system dependencies** — no BLAS, no
`ndarray`, no `faer`.

## Ordinal index family

- **`Rank`** — full-precision rank vectors (`u16` per coordinate).
- **`RankQuant`** — ranks bucketed into `1 << bits` equal-width
  bins, `bits` bits per coordinate (`dim * bits / 8` bytes/doc). Both a
  symmetric (Spearman) and asymmetric (float-query LUT) scorer.
- **`Bitmap`** — a top-bucket bitmap per document (one bit per
  coordinate); scoring is `popcount(Q AND D)`, a coarsened rank overlap.
- **`SignBitmap`** — a sign bitmap per document for sign-cosine
  candidate generation, feeding an exact rerank stage.

## Quickstart

The crate is being prepared for its first crates.io release. Until then,
add it as a git dependency:

```toml
[dependencies]
ordvec = { git = "https://github.com/Fieldnote-Echo/ordvec" }
```

```rust
use ordvec::RankQuant;

let dim = 1024;
let mut index = RankQuant::new(dim, 2);   // 2 bits/coord → 256 bytes/doc

// `add` takes a flat, row-major buffer of `dim * n_docs` f32s.
index.add(&doc_embeddings);               // &[f32], len = dim * n_docs

// Asymmetric scan: full-precision queries vs bucketed docs (recommended).
let results = index.search_asymmetric(&query_embeddings, 10); // len = dim * n_queries

let top_ids = results.indices_for_query(0);     // top-10 doc ids for query 0
let top_scores = results.scores_for_query(0);
```

For the sub-linear two-stage path (`Bitmap` / `SignBitmap` candidate
generation → `RankQuant` rerank) and the full mode comparison, see
[`docs/RANK_MODES.md`](docs/RANK_MODES.md).

## Documentation

- **Design deep-dive & reproducible benchmark tables:**
  [`docs/RANK_MODES.md`](docs/RANK_MODES.md)
- **API docs:** <https://docs.rs/ordvec> *(available after the first
  crates.io release)*
- **Paper (OrdVec / RankQuant):** _link TBD. Collaborators welcome (see
  [Contributing](#contributing))._

## Reproducible benchmark

The head-to-head benchmark generates a seeded synthetic corpus
in-process, so the quality numbers (R@10, candidate-recall, bytes/vec)
are regenerable from a clean checkout with no external corpus file:

```sh
cargo run --release --example bench_rank
```

A committed capture of one run lives at
[`benchmarks/rank_modes_results.txt`](benchmarks/rank_modes_results.txt).

## Provenance

ordvec is an original ordinal/sign retrieval substrate — all of its code
is the author's own work, developed within the
[turbovec](https://github.com/RyanCodrai/turbovec) project (an
MIT-licensed vector-quantization crate by Ryan Codrai) and factored out
here as a standalone, zero-system-dependency crate. The full development
history lives in this repository's git log. With thanks to the turbovec
project, where this substrate was built.

## Contributing

Contributions to the code, the docs, and the accompanying paper are all
welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). The crate is going
public specifically to invite collaboration on polishing the OrdVec /
RankQuant paper.

## Minimum supported Rust version

ordvec's MSRV is **Rust 1.89** — the release that stabilized the AVX-512
intrinsics the SIMD kernels rely on. Raising the MSRV is treated as a
minor-version change.

## License

Licensed under either of

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.
