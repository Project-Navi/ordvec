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
  `corpus_zoo_results.md`): the fixed **Gaussian** unfold only fits the isotropic
  marginal. Non-isotropic corpora are unfolded by the wrong density, which can
  MANUFACTURE apparent clustering. The isotropic control and zoo are therefore
  partly circular. To salvage: unfold by each corpus's **empirical** smooth
  marginal (monotone spline on sorted keys — NOT the rank-overwrite currently
  mislabeled `--unfold-empirical`, which returns Σ²=0 by construction). UNTIL
  THEN this is exploratory, not a proven property of embedding keys.
- **"Random offsets redundant / coprime adds nothing"** (`shard_recall.rs`): the
  RandomOffset arm draws an extra RNG value per projection, desyncing the stream
  so arms get DIFFERENT projection directions — it was never the clean phase-only
  ablation claimed. Plus the "fair envelope" is not fair (coprime arms saturate
  below high budgets). The *retreat* ("needs a within-axis harness") is sound;
  the positive tie claim is withdrawn.

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

## Net

The mathematically defensible deliverables are: the CRT vernier structure
(oracle + corrected density + Lean skeleton with corrected `crtEquiv` signature),
and the TwoNN metric fix. The empirical rigidity/routing findings are
exploratory with identified confounds and concrete salvage paths. The
impossibility theorems are directionally right but need restatement (binomial
not Poisson; mixture hypothesis explicit; optimality claim retracted).
