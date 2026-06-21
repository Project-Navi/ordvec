## Summary

Adds an optional CAIF-style Index Authority Receipt for ordvec benchmark evidence.

The goal is to make ordvec's index-first retrieval evidence machine-readable: quality delta, bytes/vector, latency regime, benchmark scope, limitations, fallback conditions, and a deterministic receipt hash.

## Why

ordvec already has a strong index-first compute story: compressed ordinal/sign retrieval can preserve retrieval quality under stated benchmark scopes while reducing storage and latency.

This PR adds a small evidence packet and verifier so downstream systems can answer:

> Is this compressed/index-first retrieval path evidence-supported before dense compute for this stated workload scope?

## What this includes

- `docs/INDEX_AUTHORITY_RECEIPTS.md`
- `examples/caif/trec-covid-sign-rq2.index-authority.json`
- `tools/verify_index_authority.py`

## What this does not do

- Does not change Rust code
- Does not change `Cargo.toml`
- Does not add runtime dependencies
- Does not add CI requirements
- Does not claim new benchmark results
- Does not add signing, key management, or deployment trust policy

## Verification

    python3 tools/verify_index_authority.py examples/caif/trec-covid-sign-rq2.index-authority.json

Expected output includes:

    decision: ALLOW_INDEX_FIRST
    quality_within_bootstrap_noise: true
    storage_reduction: 10.6667x
    single_query_speedup: 105.6604x

## Scope

The example uses existing public README benchmark values and preserves the stated limitations around dataset, encoder, corpus size, batch/threading regime, HNSW comparison, and larger-corpus claims.

## Framing

Benchmarks should not only report performance.

They should authorize compute paths within a defined evidence envelope.
