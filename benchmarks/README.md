# Ordinal-routing research — reviewer's guide

Exploratory investigation into ordvec's **density behavior** and whether
prime/spectral structure can improve training-free routing. Everything here is
in `examples/` (runnable probes) and `benchmarks/` (findings + proofs) — **no
changes to the `ordvec` crate or its public API.**

Reviewed by three internal adversarial agents plus the PR bots; findings are
tiered below by **what survived scrutiny**. Read the tiers, not all 11 docs.

## 3-minute path

1. This file (the tiers + the verdict at the bottom).
2. **[density_collapse_results.md](density_collapse_results.md)** — the mechanism
   (real-embedding, with its honest correction), then
   **[tau_rerank_bakeoff_results.md](tau_rerank_bakeoff_results.md)** — the
   decisive negative: it doesn't beat b=4.
3. **[ADVERSARIAL_REVIEW.md](ADVERSARIAL_REVIEW.md)** — what was challenged,
   fixed, retracted, withdrawn. The integrity record.

## SOUND — proven or real-data confirmed

| doc | claim |
|-----|-------|
| [density_collapse_results.md](density_collapse_results.md) | **Mechanism.** RankQuant b=2 density collapse = Hamming-near codes the scorer can't separate. Among those lookalikes, true neighbours have lower intra-code Kendall-tau (gap ≈ 0.04, CI > 0). Real but small. |
| [tau_rerank_bakeoff_results.md](tau_rerank_bakeoff_results.md) | **The verdict.** Does that tau signal beat b=4? NO — b=4 wins even at the tau ceiling; tau scores below b=2's own ordering. Signal is real-but-inert; just use b=4. Closes the line: research, not a feature. |
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

## The deployment question — RESOLVED (negative)

[tau_rerank_bakeoff_results.md](tau_rerank_bakeoff_results.md): the decisive
matched-bytes experiment was run. **b=4 wins decisively, even at the tau ceiling**
(real embeddings: b4 0.942, b2 0.898, tau-rerank 0.597, fp32-rerank 1.000). The
b=2 candidate pool contains every true neighbour (fp32-rerank=1.0), but the ~0.04
tau gap is too weak to ORDER them — it scores below b=2's own ordering. The
density-collapse signal is **real but inert**: "just use b=4," no ordvec feature
follows.

This is the honest bottom line of the whole branch: a characterized mechanism and
a clean negative. **Research, not a feature** — the prime/spectral/permutation
ideas for dense-region retrieval do not beat the boring baseline (spend the bits).
