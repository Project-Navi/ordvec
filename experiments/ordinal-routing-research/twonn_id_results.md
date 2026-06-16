> ⚠️ PARTIAL (adversarial review). The chord-vs-cosine METRIC FIX is correct and
> sphere-validated — that stands. BUT the estimator uses OLS-through-origin on the
> linearized CDF (F=(i+1)/N), the biased non-MLE variant; some of the "finite-
> sample deflation" attributed to sampling is actually estimator bias. Treat the
> low-tens ID as indicative, and prefer the MLE TwoNN estimator before quoting
> exact values. See benchmarks/ADVERSARIAL_REVIEW.md.

# TwoNN intrinsic dimension probe

TwoNN estimator (Facco et al. 2017) for measuring the intrinsic dimension of an
embedding corpus directly. The aim was to close the citation gap (no published
TwoNN figure for sentence-transformers) — but this branch records only the
synthetic sphere/cluster validations below; **no clean real sentence-transformer
ID is measured here** (the estimator is biased; see the caveat above). There is
no published TwoNN figure for sentence-transformers; deep vision-CNN intrinsic
dimension is low-tens (Ansuini, arXiv:1905.12784) but that is a cross-domain
analogy, not sentence-transformer evidence — so this branch establishes no
sentence-transformer ID.

Source: `examples/twonn_id.rs`. μ = r2/r1 (2nd/1st NN distance ratio), Pareto
fit d = Σ(log μ · -log(1-F)) / Σ(log μ)², top-10% tail trimmed. Anchors sampled
(≤3000), each searched exactly against the full corpus.

## CRITICAL: metric must be locally LINEAR in distance

First implementation used cosine distance (1 - cos θ ≈ θ²/2), a *squared*
distance — this HALVED every estimate (μ squared → d halved through log μ).
Fixed to chord/Euclidean distance between unit vectors: sqrt(2 - 2cos) ∝
sin(θ/2), locally linear in the angle. Anyone adapting this for cosine spaces
must use the chord metric, not cosine distance.

## Validation (isotropic sphere, true ID = ambient - 1)

| ambient dim | true ID | TwoNN (chord) | TwoNN (cosine bug) |
|-------------|---------|---------------|--------------------|
| 3  | 2  | 1.99  | 0.99 |
| 5  | 4  | 4.19  | 2.09 |
| 8  | 7  | 6.99  | 3.49 |
| 12 | 11 | 10.41 | 5.21 |
| 20 | 19 | 16.89 | — |
| 50 | 49 | 35.18 | — |

Verdict: **calibrated and trustworthy through ID ~12** (dense sampling at
n=20k). Above that, finite-sample deflation sets in (ID=20→16.9, ID=50→35) —
the curse of dimensionality floor on ALL NN-based ID estimators: ~exp(d) points
needed to populate a d-dim neighborhood. Reported IDs in the tens are therefore
LOWER BOUNDS; true ID may be higher. (The same finite-sample deflation would
also affect Ansuini's 12-25 vision-CNN last-layer figures — a methodological
note, not evidence those figures transfer to sentence-transformers.)

## Measurements

| corpus | n | ambient | TwoNN ID |
|--------|---|---------|----------|
| synthetic clustered (latent_dim=64) | 20k | 256 | 27.5 |
| isotropic dim=256 | 10k | 256 | (sampling-limited) |

The synthetic clustered corpus (built from a 64-D latent manifold) reads ID≈27
— deflated from 64 by finite-sample + the within-cluster local geometry being
lower-dim than the global latent space.

## For real embeddings

Run: `cargo run --release --example twonn_id -- --corpus-npy your_embeddings.npy`
Loads 2-D <f4 C-order .npy, L2-normalizes rows, reports chord-metric TwoNN ID.
A low-tens sentence-transformer ID is a HYPOTHESIS by analogy to vision-CNN final
layers (Ansuini's 12-25 are CNNs, not sentence encoders — see
conjecture_citation_audit.md), NOT established here; READ ANY ID AS A LOWER BOUND. This is the number that sizes the
routing-layer projection budget: R ≈ c·d_int ⇒ R∈{8,16} suffices.
