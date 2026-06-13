> ⛔ WITHDRAWN. This zoo runs the number-variance probe whose unfold is
> uncalibrated (see spectral_probe_results.md banner and ADVERSARIAL_REVIEW.md).
> The smooth-unfold salvage inverted the isotropic/clustered ordering, so the
> "robust across geometries" claim is unsupported. Retained as record only. The
> gen_corpus.rs zoo GENERATOR itself is fine and reusable; the measurement on top
> is what's withdrawn.

# Corpus zoo: number-variance robustness across geometries

Rigorous multi-corpus verification of the "embedding routing keys are not rigid"
finding. Generates 6 geometries as .npy (`examples/gen_corpus.rs`) and runs the
number-variance probe on each. Includes instrument controls at BOTH ends.

n=50k, dim=256, 8 random-projection keys, Gaussian unfold.

## Σ²(L)/L  (≈1 Poisson · >1 clustered · <1 RIGID)

| kind        | L=8   | L=32  | L=128 | L=512  | reading |
|-------------|-------|-------|-------|--------|---------|
| isotropic   | 0.99  | 0.99  | 0.99  | 0.93   | Poisson NULL ✓ |
| clustered   | 1.34  | 2.39  | 6.13  | 18.5   | super-Poisson |
| anisotropic | 2.38  | 6.48  | 22.7  | 85.0   | super-Poisson |
| rogue       | 1.60  | 3.34  | 10.0  | 34.1   | super-Poisson |
| manifold    | 2.93  | 8.73  | 30.3  | 96.4   | super-Poisson |

## Instrument controls (CRITICAL — proves the probe isn't blind)

| control | result | proves |
|---------|--------|--------|
| `--rigid-selftest` (perfect lattice key) | Σ²/L = 0.0000 all L | probe DETECTS rigidity ✓ |
| isotropic corpus | Σ²/L = 0.99 flat | probe DETECTS Poisson ✓ |

## Findings

1. **Robust across 5 geometries: embedding-like keys are super-Poisson
   (clustered), never rigid.** Crucially this includes the NONLINEAR MANIFOLD
   (2.9→96) — the one geometry where angular rigidity could plausibly have
   hidden. It didn't.
2. **The probe is validated at both ends.** Perfect lattice → Σ²=0 (sees
   rigidity); isotropic → Σ²/L≈1 (sees Poisson). So the super-Poisson readings
   are real, not instrument blindness. This was the gap that would have let a
   skeptic dismiss the whole spectral finding; it is now closed.
3. **Bonus substantive result (random-projection lattice control):** a sequence
   that is rigid along ONE direction does NOT produce a rigid key after random
   projection + L2 normalization (tested: still super-Poisson). I.e. even if an
   embedding had rigid structure on a special axis, a data-oblivious
   random-projection router would not see it — reinforcing that spectral
   structure is not exploitable training-free. (The instrument self-test
   isolates the estimator from this pipeline effect.)

## Verdict

The "keys are non-rigid → quantile bucketing is the whole story, no spectral
opening" conclusion is now verified across diverse corpus geometries with
validated instrument controls, not a single synthetic distribution. A FALLING
Σ²/L on a real embedding dump would still be the surprise worth chasing — run
the zoo recipe on real data via the runbook.

Reproduce:
```
for K in isotropic clustered anisotropic rogue manifold; do
  cargo run --release --example gen_corpus -- --kind $K --n 50000 --dim 256 --out zoo_$K.npy
  cargo run --release --example spectral_probe -- --corpus-npy zoo_$K.npy
done
cargo run --release --example spectral_probe -- --rigid-selftest   # instrument control
```
