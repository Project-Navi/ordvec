# Ordinal-routing research — reviewer's guide

Exploratory investigation into ordvec's **density behavior** and whether
prime/spectral structure can improve training-free routing. Everything here is
in `examples/` (runnable probes) and `benchmarks/` (findings + proofs) — **no
changes to the `ordvec` crate or its public API.**

Reviewed by three internal adversarial agents plus the PR bots; findings are
tiered below by **what survived scrutiny**. Read the tiers, not all 11 docs.

## 3-minute path

1. This file (the tiers).
2. **[density_collapse_results.md](density_collapse_results.md)** — the headline,
   real-embedding result (and its honest correction).
3. **[ADVERSARIAL_REVIEW.md](ADVERSARIAL_REVIEW.md)** — what was challenged,
   fixed, retracted, withdrawn. The integrity record.

## SOUND — proven or real-data confirmed

| doc | claim |
|-----|-------|
| [density_collapse_results.md](density_collapse_results.md) | **Headline.** RankQuant b=2 density collapse = Hamming-near codes the scorer can't separate. Among those lookalikes, true neighbours have lower intra-code Kendall-tau (gap ≈ 0.04, bootstrap CI > 0) on real `nomic-embed-text` embeddings. Modest but real; the lever is permutation order already in the `Rank` code. |
| [crt_seam_oracle_results.md](crt_seam_oracle_results.md) | CRT vernier seam theorem — exhaustive finite proof: lcm spacing, one coincidence/period, capped density `∏min(2t+1,m_i)/m_i`. Lean 4 skeleton in [lean/](lean/). |
| [twonn_id_results.md](twonn_id_results.md) | TwoNN intrinsic-dimension probe (chord-metric fix, sphere-validated). `nomic-embed-text` ID ≈ 13 / 768. Estimator-bias caveat noted. |
| [shard_recall_results.md](shard_recall_results.md) | Controlled ablation (post RNG-desync fix): random phase offsets add nothing vs aligned grids across R random directions. |

## THEORY — directionally right, restated honestly

| doc | status |
|-----|--------|
| [rigidity_impossibility_proofs.md](rigidity_impossibility_proofs.md) | The routing key is not number-variance-rigid (Thm 2/3, binomial `L(1-L/n)`). The over-broad "quantile optimal over all partitions" claim is **retracted** as a non-sequitur. |
| [conjecture_citation_audit.md](conjecture_citation_audit.md) | Citations verified by direct fetch (von Koch, Broughan-Barnett, Montgomery, Ethayarajh, etc.). |

## WITHDRAWN — see [withdrawn/](withdrawn/)

The number-variance "super-Poisson" finding ([withdrawn/spectral_probe_results.md](withdrawn/spectral_probe_results.md),
[withdrawn/corpus_zoo_results.md](withdrawn/corpus_zoo_results.md)) did not
survive: its unfold is uncalibrated (a salvage attempt inverted the result). The
*theory* above does not depend on it. Kept for the record, not as a claim.

## Conjecture verdict (the framing question)

Prime / Li(x) / Sacks-spiral constructions don't help retrieval: they act on the
index (ℕ) and carry no corpus information. The exploitable dense-region structure
lives on the permutohedron `S_D` — the data's own order — which is the
density-collapse result above. Detail across the theory docs + ADVERSARIAL_REVIEW.

## Reproduce

Per-doc commands are at the bottom of each file. Real-embedding pipeline (GPU via
ollama) is fully recorded in [density_collapse_results.md](density_collapse_results.md);
external-corpus recipe in [REAL_CORPUS_RUNBOOK.md](REAL_CORPUS_RUNBOOK.md).

## Open follow-up

The decisive deployment question is unanswered: does the ≈0.04 tau gap convert to
real recall gain vs simply using b=4 at matched bytes (R@10 vs FP32)? That, plus a
second corpus/encoder, is what would move density-collapse from "real effect" to
"ship it."
