# Roadmap

> A living document. ordvec is a small, paper-driven project; this captures
> **direction and scope**, not committed dates.

## Vision

**ordvec is a retrieval _primitive_, not a database.** The goal is to be a
universally embeddable building block for **edge-deployed RAG and on-device /
local AI retrieval** — settings where compute, memory, and storage are tight and
the hardware is heterogeneous (servers, ARM edge devices, browsers / WASM, and
eventually mobile and embedded targets).

Everything distinctive about ordvec serves that end:

- **Training-free encoding** — no codebook, no learned rotation, no fit step,
  and nothing to refit when the corpus drifts.
- **Zero system dependencies** — pure Rust, no BLAS / LAPACK / `ndarray` /
  `faer`; no native library to install; links and cross-compiles cleanly.
- **Predictable footprint** — exactly `dim × bits / 8` bytes per document, known
  before you see any data.
- **Runtime-dispatched SIMD with a scalar fallback** — the same retrieval core
  runs anywhere it compiles (AVX-512 / AVX2, NEON, WASM `simd128`, scalar).

## Non-goals

ordvec will **not** grow into a standalone vector database. It deliberately does
not pursue billion-scale navigable-graph ANN, distributed serving / sharding, a
query language, or persistence and transaction machinery beyond its simple file
formats. That is the territory of pgvector, Qdrant, Milvus, LanceDB, and full
Graph-RAG stacks. ordvec stays the **substrate** those systems — and bespoke
edge pipelines — compose, not a competitor to them.

## Direction

The throughline is **"be a good neighbour"**: ordvec should embed _natively_
into more hosts rather than forcing callers to adapt to it.

- **Publish.** A coordinated first release to crates.io (`ordvec`) and PyPI
  (`ordvec`), carrying SLSA build provenance and SBOMs (the release machinery is
  already in place). Unblocks `docs.rs` and the registry badges.
- **Cross-stack embedding via a C ABI.** A `cdylib` plus a generated C header so
  non-Rust / non-Python edge runtimes can link ordvec directly — the single
  largest reach multiplier beyond Rust and Python.
- **Adapters.** Thin integration layers for host retrieval / RAG systems. ordvec
  factored out of [turbovec](https://github.com/RyanCodrai/turbovec); natural
  next targets are mainstream RAG frameworks (via the Python binding) and
  Rust-native edge Graph-RAG stacks such as
  [EdgeQuake](https://github.com/raphaelmansuy/edgequake), where ordvec could
  back the vector-search layer in a Postgres-less / edge deployment —
  complementing, not replacing, the graph layer.
- **Broader platform reach.** musllinux wheels (Alpine / minimal containers);
  continued aarch64 and WASM coverage.
- **Toward 1.0.** An API-stability pass and a documented compatibility policy.
- **Supply-chain posture.** OpenSSF Best Practices _passing_ is earned;
  _silver_ is the next target (governance, this roadmap, coverage reporting).

## Out of scope here

Benchmarks and the intellectual framing live in the OrdVec / RankQuant paper
and the companion `ordvec-formalization` Lean repository; this roadmap concerns
the crate as an embeddable building block.
