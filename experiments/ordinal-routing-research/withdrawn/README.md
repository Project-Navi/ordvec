# Withdrawn results

These findings did NOT survive review and are kept only for the record — do not
cite them as results. See [../ADVERSARIAL_REVIEW.md](../ADVERSARIAL_REVIEW.md)
(Round 1) and [../README.md](../README.md) for the tiering.

- **spectral_probe_results.md** / **corpus_zoo_results.md** — the number-variance
  "embedding keys are super-Poisson, never rigid" finding. The probe's unfold is
  uncalibrated: a smooth-unfold salvage attempt INVERTED the isotropic/clustered
  ordering, proving neither unfold is validated against analytic ground truth.
  The rigidity THEORY (../rigidity_impossibility_proofs.md) does not depend on
  this probe and is unaffected.

The `examples/spectral_probe.rs` instrument itself is retained (its
`--rigid-selftest` is a valid estimator check); only the corpus *findings* are
withdrawn.
