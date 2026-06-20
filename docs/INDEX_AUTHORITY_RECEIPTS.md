# Index Authority Receipts for ordvec

Index Authority Receipts are CAIF-style evidence packets for ordvec benchmark results.

They make index-first retrieval evidence machine-readable.

Instead of only asking whether a retrieval mode is faster, a receipt asks whether the benchmark evidence supports using a compressed/index-first retrieval path within a stated workload scope.

## IFC

Index-First Compute means a cheaper index representation is evaluated before more expensive dense compute.

For ordvec, IFC can include RankQuant compressed scan, Bitmap candidate generation, SignBitmap candidate generation, or SignBitmap to RankQuant rerank.

## CAIF

Compute Authority Index Format describes whether a compute path is justified under a stated evidence envelope.

A receipt records baseline mode, candidate mode, quality delta, storage reduction, latency profile, scope, limitations, fallback conditions, and a deterministic receipt hash.

## Verify

Run:

    python3 tools/verify_index_authority.py examples/caif/trec-covid-sign-rq2.index-authority.json

Expected output:

    decision: ALLOW_INDEX_FIRST
    mode: sign_to_rq2
    baseline: flat_exact
    quality_within_bootstrap_noise: true
    storage_reduction: 10.6667x
    single_query_speedup: 105.6604x

## Non-goals

This does not change Rust code, Cargo.toml, CI, runtime behavior, signing, key management, or deployment trust policy.

It does not create new benchmark claims.

It preserves the stated benchmark scope and limitations.

## Principle

Benchmarks should not only report performance.

They should authorize compute paths within a defined evidence envelope.
