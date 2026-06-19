# CRT seam oracle — corrected vernier theorem (exhaustive finite proof)

> Lean 4 formalization of this theorem lives in the companion repo:
> [ordvec-formalization#17](https://github.com/Fieldnote-Echo/ordvec-formalization/pull/17)
> (open PR, `sorry`-free).

`examples/crt_seam_oracle.rs` enumerates the full ring Z/M to verify the
within-axis vernier (coprime-grid) theorem exactly. This is the SPEC for the
vernier recall experiment and the corrected target for any Lean formalization.
Exhaustive enumeration over the finite ring IS the proof for these parameters.

Params: L=4 grids on one axis, coprime periods {3,5,7,11}, M=1155.

## CONFIRMED exactly

1. **Coincidence spacing = lcm.** First common seam of grids (m_i,m_j) at phi=0
   is exactly lcm(m_i,m_j). Coprime ⇒ lcm = m_i·m_j (maximal spacing: 3,5→15;
   5,7→35). Non-coprime coincide early (4,6→12; 6,9→18). This is WHY coprimality
   maximizes seam separation.
2. **Exactly one all-L coincidence per period M.** count = 1 over [0,M). ✓

## CORRECTED (the closed form is the CAPPED product)

3. **Joint near-seam density is ∏_i min(2t+1, m_i)/m_i.** The uncapped
   ∏(2t+1)/m_i is correct ONLY in the interior `2t+1 ≤ min_i m_i` (here t≤1);
   beyond that, bands saturate a grid and that grid's factor caps at 1. The
   uncapped form exceeds 1 at t=3 (period-3 grid: 7/3 > 1) — the SAME nonsense
   as the rejected continuous (2t)^L/M. (Caught in adversarial review; the Rust
   code already computes the capped product via `.min(p)`.)

   | t | measured | ∏min(2t+1,m_i)/m_i (capped) | ∏(2t+1)/m_i (uncapped) | (2t)^L/M |
   |---|----------|------------------------------|------------------------|----------|
   | 0 | 0.000866 | 0.000866                     | 0.000866               | 0.000000 |
   | 1 | 0.070130 | 0.070130                     | 0.070130               | 0.013853 |
   | 2 | 0.324675 | 0.324675                     | 1.039 (>1, wrong)      | 0.221645 |
   | 3 | 0.636364 | 0.636364                     | 2.078 (>1, wrong)      | 1.122    |

   Measured matches the CAPPED product to 6 d.p. at all t. The uncapped product
   is valid only for t≤1 (where 2t+1 ≤ min m_i = 3); it already exceeds 1 at t=2.
   The provable/Lean statement is the capped form, OR the uncapped form under the
   explicit precondition `2t+1 ≤ min_i m_i`.

## REFUTED (stronger than the earlier caveat)

4. **No generic pointwise seam-margin floor — phases cannot rescue it.**
   min_x max_i dist_to_seam = 0 with phi=0 AND with staggered phases [0,1,2,3].
   For L=4 with these small periods, some position always sits on every grid's
   seam regardless of phase. The "choose phases for a pointwise floor >= c"
   claim FAILS in this regime; the floor is not generic, it exists only for
   specific (periods, phases, L) and must be verified per-instance, not assumed.

## Proof of the density closed form (was: numerically confirmed only)

Claim: joint near-seam density = ∏_i min(2t+1, m_i)/m_i for pairwise-coprime
m_i. Equivalently = ∏_i (2t+1)/m_i under the precondition 2t+1 ≤ min_i m_i.

Proof. CRT: x ↦ (x mod m_1, …, x mod m_L) is a BIJECTION Z/M → ∏ Z/m_i (M=∏m_i,
pairwise coprime). Position x is within band t of a seam on grid i iff
(x − φ_i) mod m_i lies in the centered band {−t,…,+t} mod m_i. The number of
DISTINCT such residues is min(2t+1, m_i): exactly 2t+1 when the band fits
(2t+1 ≤ m_i), capping at m_i once it saturates (the band wraps to cover all
residues). The wraparound/min in the distance condition is INTRA-coordinate
(depends only on x mod m_i), so it does not couple grids — CRT independence
across coordinates is untouched. The joint count therefore factors:

    count = ∏_i min(2t+1, m_i),   density = count/M = ∏_i min(2t+1,m_i)/m_i.  ∎

(Even m_i have an extra antipodal-residue caveat at the exact half-period; the
chosen odd periods avoid it.) The independence is precisely what CRT supplies;
without coprimality the residue coordinates are coupled and the product fails
(matches Check 1: non-coprime grids coincide early). This is the provable form —
NOT the uncapped ∏(2t+1)/m_i (>1 for t≥2 here) nor the continuous (2t)^L/M.

## Consequence for the workplan

- The provable theorem is the DENSITY/COUNT statement (checks 1–3, CAPPED product
  ∏min(2t+1,m_i)/m_i), not a pointwise floor. A Lean proof should target the
  capped form (or the uncapped form under explicit 2t+1 ≤ min m_i) and
  "one coincidence per period", and should NOT claim a phase-tunable margin.
- The vernier RECALL experiment must therefore test whether the (small,
  bounded) joint blind-spot region actually costs recall — it cannot lean on a
  guaranteed per-query margin, because there isn't one.
- This is the ambush avoided: building the recall test on a phase-tunable-floor
  premise, then having Lean fail to prove it, would have invalidated the
  experiment after the fact. Caught by finite enumeration first.

Reproduce: `cargo run --release --example crt_seam_oracle`
