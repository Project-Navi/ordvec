# Density collapse in ordvec: mechanism and a recoverable signal

What "density collapse" actually is in RankQuant, and whether the lost signal is
recoverable. Source: `examples/density_collapse.rs`. Ground truth = FP32 cosine,
so this probe cannot miscalibrate (unlike the withdrawn number-variance probe).

## Mechanism (corrected mid-investigation)

First model was wrong: I expected b=2 codes to COLLIDE (identical) in dense
regions. They don't — the rank transform is a permutation of 0..D, so a b=2 code
is a length-D sequence of bucket ids; two docs collide only if all D coordinates
bucket identically (astronomically rare on continuous data; measured collision
rate ≈ 0 even at noise=0.08).

The real mechanism is NEAR-collision: docs whose b=2 codes are HAMMING-CLOSE
(differ in few coordinate-buckets) are what the scorer cannot separate.
- noise=0.30: mean nearest-code Hamming = 59.9 / 256; ~123 docs within ~1.5×.
- noise=0.10: mean nearest-code Hamming = 18.4 / 256; ~58 docs within ~1.5×.
Denser clusters → tighter Hamming neighbourhoods → more docs the b=2 kernel
conflates. THAT is collapse: not identical codes, but codes too close for the
2-bit resolution to rank correctly.

## The recoverable signal (positive result)

Within each probe's b=2 "lookalike" set (M=40 Hamming-nearest), split by TRUE
FP32 cosine into the true-neighbour half vs the far-lookalike half. Compare the
top-k (k=16) Kendall-tau distance of the probe's coordinate ORDER to each:

| density | true-nbr cosine | true-nbr tau | far cosine | far tau | probes tau_true<tau_far |
|---------|-----------------|--------------|------------|---------|--------------------------|
| noise=0.30 | 0.9397 | 0.2616 | 0.9250 | 0.2810 | 262/300 = 0.873 |
| noise=0.10 | 0.9926 | 0.1250 | 0.9905 | 0.1388 | 272/300 = 0.907 |

**The FP32-true neighbours have systematically LOWER intra-code Kendall-tau than
the b2-lookalikes the code conflates them with — 87–91% of probes, both
densities.** The fine permutation order (which coordinate outranks which, WITHIN
the top bucket) separates true neighbours from false lookalikes.

## Why this matters

That fine order is ALREADY in the full `Rank` code (u16 per coord) and is
exactly what RankQuant b=2 discards. So the signal to break dense-region ties is
recoverable WITHOUT new storage — as a rerank stage on Kendall-tau of the top-k
coordinates of the b=2 survivors. This is the data-dependent, on-the-permutohedron
version of "uncover structure in dense regions": the exploitable combinatorial
structure is in S_D (the order the encoder induced), NOT on the integer line
(primes / Sacks spiral, which act on the index and carry no corpus information —
see the conjecture thread).

## Caveats / open

- Synthetic low-rank clustered corpus; confirm on real embeddings via a corpus
  dump (the probe has no .npy loader yet — add one to test real data).
- The test shows the signal EXISTS (separation in tau); it does not yet show a
  tau-rerank improves end-to-end recall vs simply using b=4. The honest next
  experiment: tau-rerank of b=2 survivors vs b=4 at matched bytes, R@10 vs FP32.
- Kendall-tau here is computed on FP32 values restricted to the probe's top-k
  coords; a deployable version computes it on the stored ranks of those coords.

Reproduce:
```
cargo run --release --example density_collapse                 # noise=0.30
cargo run --release --example density_collapse -- --noise 0.10 # denser
```
