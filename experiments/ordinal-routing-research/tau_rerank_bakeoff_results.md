# Matched-bytes bake-off: tau-rerank vs b=4 — the deployment verdict

The decisive question for the density-collapse finding: does breaking dense-region
ties with permutation order (Kendall-tau rerank of b=2 survivors) beat simply
spending the bits on b=4? Source: `examples/tau_rerank_bakeoff.rs`. CEILING
experiment — tau uses the FULL stored rank order (best case for the method).

## Result — NO. b=4 wins decisively, even at the tau ceiling.

Real embeddings (nomic-embed-text, 8665 docs / 200 held-out queries, 768-d,
topk=128, M=50 — tau's best parameters):

| arm | bytes/vec | R@10 |
|-----|-----------|------|
| b2 asym | 192 | 0.8980 |
| b4 asym | 384 | 0.9420 |
| b2 + tau-rerank | 192* | 0.5965  (*ceiling: full stored ranks) |
| b2 + fp32-rerank | — | 1.0000  (absolute ceiling) |

Synthetic (dim 256, n 30k) tells the same story: b2 0.5785, b4 0.8095,
tau-rerank 0.22–0.57 (best case ties b2, never reaches b4), fp32-rerank 1.0000.

## Why the tau-rerank fails (the precise lesson)

1. **The candidate pool is fine.** fp32-rerank = 1.0000 proves b=2's top-M
   *contains* every FP32 true neighbour. Candidate generation is not the problem.
2. **Tau MISORDERS them.** tau-rerank (0.597) is worse than b=2's OWN ordering
   (0.898). The ~0.04 tau gap from density_collapse_results.md is real as a faint
   BINARY discriminator (true-neighbour vs far-lookalike) but far too weak to
   ORDER 50 candidates correctly — it actively destroys the ordering b=2's
   scores already provide.
3. **Even at the ceiling** (full stored ranks, best topk/M) tau-rerank only
   claws back to ≈ b=2; it never reaches b=4. No compact tau codec could do
   better than this ceiling, so codec work is NOT justified.

## Verdict

**The density-collapse tau-signal is a real-but-inert observation, not a usable
lever. "Just use b=4."** This closes the loop opened in
density_collapse_results.md: the signal exists (binary separation, CI > 0) but
does not convert to retrieval value. There is no ordvec code change here worth
making.

This is the honest negative that the whole research line was meant to resolve:
the prime/spectral/permutation ideas for improving dense-region retrieval do not
beat the boring baseline (spend the bits). The contribution of this branch is
therefore RESEARCH (a characterized mechanism + a clean negative), not a feature.

Reproduce:
```
cargo run --release --example tau_rerank_bakeoff                      # synthetic
cargo run --release --example tau_rerank_bakeoff -- \
    --corpus-npy repo_real.npy --queries-npy repo_q.npy --topk 128 --m 50
```
