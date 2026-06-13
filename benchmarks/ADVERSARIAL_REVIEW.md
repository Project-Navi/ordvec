# Adversarial review findings (three independent hostile reviewers)

This branch was reviewed by three adversarial agents (Rust probes, math proofs,
Lean skeleton). Their findings are recorded here verbatim-in-summary so the PR
carries its own critique. Conclusions are tiered by what SURVIVED review.

## SOUND (ships as claimed)

- **CRT seam oracle** (`examples/crt_seam_oracle.rs`): exhaustive finite proof.
  Coincidence spacing = lcm; exactly one all-L coincidence per period; the
  honest negative result that phases cannot generically rescue a pointwise floor.
  All three reviewers independent-agree this file is honest.
- **TwoNN metric fix** (chord vs cosine): the squared-distance bug fix is correct
  and validated on sphere controls.

## CORRECTED IN THIS BRANCH (was wrong/loose, now fixed)

- **CRT density closed form**: must be `∏ min(2t+1, m_i)/m_i`, NOT `∏(2t+1)/m_i`.
  The literal product exceeds 1 at t=3 (period-3 grid saturates) — the SAME error
  class the doc attributed to the rejected `(2t)^L/M` form. The Rust code already
  caps with `.min(p)`; the markdown and Lean statements are corrected to match.
  Valid precondition for the uncapped form: `2t+1 ≤ min_i m_i` (t≤1 here).
- **Lean `crtEquiv`**: false as typed — `ZMod 0 = ℤ` breaks the cardinality
  match. Needs `[∀ i, NeZero (m i)]`. Signature corrected; `hcongr` lemma plan
  corrected to `Equiv.subtypeEquiv` + `Fintype.card_congr` (the cited
  `Equiv.card_filter_map` does not exist).

## DEMOTED TO EXPLORATORY (known confounds — NOT a settled finding)

- **"Routing keys are super-Poisson, never rigid"** (`spectral_probe.rs`,
  `withdrawn/corpus_zoo_results.md`): **WITHDRAWN — the probe does not measure what it
  claims.** Attempted salvage (added `--unfold-smooth K`, a K-knot empirical
  unfold) INVERTED the result: under the smooth unfold the isotropic corpus
  reads super-Poisson (Σ²/L 1.7→12) and clustered reads LOWER (1.4→6.8) — the
  opposite ordering from the Gaussian-unfold version. So the two unfolds
  disagree and NEITHER is validated against analytic ground truth. The
  Gaussian-unfold "isotropic = clean Poisson 0.99" was not a control passing —
  it was the single marginal the wrong unfold happened to fit. The smooth unfold
  has its own artifact (knot-scale + interpolation structure). CONCLUSION: the
  number-variance empirical finding is UNSUPPORTED in either direction. Fixing it
  requires rebuilding the estimator against a case with KNOWN analytic number
  variance (e.g. a stationary process with closed-form Σ²(L)) to calibrate the
  unfold + window estimator before trusting any corpus reading. This is open
  follow-up work, not a patch. The THEORY (rigidity_impossibility_proofs.md) is
  unaffected — it does not depend on this probe.
- **"Random offsets redundant / coprime adds nothing"** (`shard_recall.rs`):
  **Bug L FIXED.** `build_projs` now seeds direction and phase RNGs separately
  and identically across arms, so aligned vs random-offset share the same R
  directions and the ablation is clean. Re-run: aligned 0.9095 vs random-offset
  0.9080 (tied) — the "random offsets add nothing" claim now holds on a
  controlled comparison. Still-open caveat: the "fair envelope" undersells the
  coprime arms (they subdivide buckets and saturate below high budgets), and
  coprimality across R directions is the wrong geometry — the within-axis vernier
  harness remains unbuilt (theory is in crt_seam_oracle).
- **`gen_corpus` Bug O FIXED.** Corpus and queries now share one geometry
  (`A` + prototypes seeded from a dedicated geometry-only RNG keyed by cfg.seed),
  so query/corpus latent spaces match and shard_recall ground truth is valid.

## IMPOSSIBILITY PROOFS — repairable, currently overstated

- **Theorem 2**: conclusion (no rigidity) holds, but proof text is WRONG as
  written — n i.i.d. uniforms are a BINOMIAL process, Σ²(L) = L(1−L/n), not
  "Poisson, Σ²=L exactly, independent". Restate with the binomial value; the
  Θ(L) conclusion is unaffected.
- **Theorem 3**: correct under its hypothesis but the hypothesis is smuggled —
  "any fixed-distribution corpus" must be stated as "conditionally i.i.d. given
  latent θ (mixture of i.i.d.)". de Finetti is decorative; it's the law of total
  variance. Finite-without-replacement (the escape hatch) is excluded by
  ASSUMPTION, not proof.
- **NON-SEQUITUR (must retract)**: "quantile bucketing is optimal against the
  entire achievable class / prime-spectral structure provably is not there" does
  NOT follow from Σ²(L) ≥ L. Number variance and partition recall/load-balance
  are different figures of merit. Narrow the claim to: "the key is not
  number-variance-rigid" (true), and drop the optimality-over-all-partitions
  claim unless separately proved (likely false).

## Round 2 (real-embedding pipeline + post-PR code review)

After the real-embedding work and the PR was opened, a second hostile review
(plus the Gemini/qodo PR bots) hit the new material:

- **CRITICAL — density-collapse headline was an artifact, now corrected.** The
  win-rate climb 0.667→0.930 with top-k was an estimator-variance effect (M2),
  and tau was computed on the probe's own coords, coupling it to cosine (M1).
  FIXED: tau now uses the per-pair UNION of top coords (de-circularized), and we
  report the tau GAP (effect size) with a bootstrap 95% CI instead of win rate.
  Result survives but is MODEST and FLAT: gap ≈ 0.04, CI strictly > 0 at every
  top-k. The "sharpening / signature of a real effect" claim is RETRACTED; the
  small-but-real separation stands.
- **qodo: bucketing bug FIXED.** density_collapse reimplemented bucketing as
  `rank/(d/2^bits)` (panics at d/2^bits==0; wrong for non-divisible dims). Now
  uses `ordvec::rank::rank_to_bucket` — measures REAL RankQuant behavior.
- **embed_ollama.py hardened:** E2 (silent row misalignment if ollama returns
  wrong count) now aborts; E3 (empty corpus) guarded.
- **Reproducibility (E4/E5):** the repo-sentence extraction + embed procedure is
  now recorded verbatim in density_collapse_results.md (was unrecorded).
- **G1 overclaim:** body language softened; single-corpus/single-model
  generality explicitly NOT claimed.

## Net

The mathematically defensible deliverables are: the CRT vernier structure
(oracle + corrected density + Lean skeleton with corrected `crtEquiv` signature),
and the TwoNN metric fix. The empirical rigidity/routing findings are
exploratory with identified confounds and concrete salvage paths. The
impossibility theorems are directionally right but need restatement (binomial
not Poisson; mixture hypothesis explicit; optimality claim retracted).
