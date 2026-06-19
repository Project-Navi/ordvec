> ⛔ WITHDRAWN — PROBE DOES NOT MEASURE WHAT IT CLAIMS. The salvage attempt
> (`--unfold-smooth K`) INVERTED the result: smooth unfold makes isotropic read
> super-Poisson (Σ²/L 1.7→12) and clustered read LOWER. The two unfolds disagree
> and neither is validated against analytic ground truth; the Gaussian "isotropic
> = 0.99 Poisson" control was just the one marginal that unfold happened to fit.
> The numbers below are RETAINED ONLY as the record of the failure. Fixing needs
> the estimator calibrated against a process with known closed-form Σ²(L). The
> theory (rigidity_impossibility_proofs.md) does NOT depend on this probe. Full
> account: ../ADVERSARIAL_REVIEW.md.

# Number-variance probe: is a 1-D routing key rigid or Poisson?

Experiment for the "prime mile-marker / spectral index" conjecture. The
conjecture needs the routing key to exhibit **rigidity** (sub-linear number
variance, Σ²(L) ~ log L, like GUE eigenvalue / zeta-zero spectra) for any
spectral/prime-gap partition structure to beat plain quantile bucketing.

Source: `examples/spectral_probe.rs`. Number variance Σ²(L) = variance of the
count of keys in a window of length L, after unfolding to unit mean density.
Poisson ⇒ Σ²(L) = L (ratio 1, flat). Rigid ⇒ Σ²(L) ~ log L (ratio → 0).

## Results (synthetic, n=200k, dim=256, 8 random-projection keys, seed=1)

| L | clustered Σ²/L | isotropic Σ²/L (control) | quantile-unfold Σ² |
|---|---|---|---|
| 2 | 1.28 | 1.00 | 0.0000 |
| 8 | 3.05 | 1.00 | 0.0000 |
| 32 | 3.26 | 1.00 | 0.0000 |
| 128 | 5.94 | 1.01 | 0.0000 |
| 512 | 17.93 | 1.01 | 0.0000 |

## Verdict

1. **Isotropic control = exact Poisson (ratio 1.00, flat).** Validates the
   probe: it reports "no structure" when there is none. The clustered signal is
   therefore real, not an unfold artifact.
2. **Clustered key is SUPER-Poissonian (ratio climbs 1.3→18, Σ² ≈ 0.037·L²).**
   Clustering, the *opposite* of the rigidity the conjecture requires. There is
   no sub-linear (log-L) regime for spectral/prime structure to grip.
3. **Quantile unfold = Σ² 0.0000 exactly.** Quantile (inverse-CDF) tiling
   balances the key perfectly by construction — empirical confirmation that it
   strictly dominates any fixed-density tiling (including Li(x)) on occupancy.

Conclusion: on this corpus the scalar routing key is Poisson-to-clustered, never
rigid. Quantile bucketing is the whole story for occupancy balance. The opening
the conjecture needed (rigidity) is absent. Re-run on real embeddings with
`--corpus-npy` to confirm; prior is strongly that real keys are also non-rigid.

Reproduce:
```
cargo run --release --example spectral_probe                    # clustered
cargo run --release --example spectral_probe -- --isotropic     # Poisson control
cargo run --release --example spectral_probe -- --unfold-empirical  # quantile
```
