# Lean formalization — vernier CRT seam-density

`VernierSeamDensity.lean` formalizes the proved-on-paper seam-density theorem
(benchmarks/crt_seam_oracle_results.md) as a standalone module for the
`Fieldnote-Echo/ordvec-formalization` repo (Lean 4.28.0, Apache header,
`OrdvecFormalization` namespace).

(An earlier `CrtSeamDensity.lean` draft was superseded by this module and removed.)

## Placement note (corrected)

This module does NOT build on the repo's `Finite*Quotient` / `QuotientKernel`
stack. That layer proves fiber-invariance/sufficiency of abstract quotient maps
`Ω → Z` (`productMap = Prod.map`). The seam-density theorem is a CRT *bijection*
`ZMod M ≃ ∏ ZMod mᵢ` plus a *cardinality count* — a different toolkit (Mathlib
`ZMod` + `Fintype.piFinset`), which the formalization repo does not otherwise
import. Correct home (that repo), independent module.

## Status: convergent, not staged

There is no `0.5.0` branch/tag in any Fieldnote-Echo repo (verified with auth).
Nelson independently reached similar conclusions; "target 0.5.0" means this is
intended for the next release, not that a branch exists to diff against. When his
working branch surfaces, reconcile theorem statements (impossibility + CRT
density) — convergence is corroboration for the paper.

## What is established vs. open

| Statement | Lean status |
|---|---|
| (A) product-space count `∏(2t+1)`, NO coprimality | structurally complete, modulo name-checks ([VERIFY] `Fintype.mem_piFinset`, `Fintype.card_piFinset`) |
| (B) position-space count over `ZMod M` | depends on `crtEquiv` (sorry) + `hcongr` (sorry) |
| one-coincidence-per-period (t=0) | follows from (B) |
| `card_admissible` (per-grid = 2t+1) | sorry — routine `ZMod.val` window injectivity |

## The honest crux

Coprimality is used in EXACTLY ONE place: `crtEquiv`, the indexed bijection
`ZMod (∏ m i) ≃ ∏ i, ZMod (m i)`. The product-space half (A) is true for any
family and deliberately carries no `cop` hypothesis — stating it that way
prevents the classic self-deception of "proving" a CRT theorem that never
actually touches coprimality.

## Proof debt, ranked

1. `crtEquiv` — fold `ZMod.chineseRemainder` over `Fin L` by induction, or
   locate an existing indexed iso. Highest effort; the real mathematical content.
2. `card_admissible` — injectivity of `k ↦ (k : ZMod m) − t` on `{0..2t}` when
   `2t+1 ≤ m`. Routine.
3. `hcongr` — cardinality transport across `crtEquiv`. Standard once (1) exists.
4. Confirm [VERIFY] names against installed Mathlib.

## Build

No Lean toolchain was present at drafting; names carry inline confidence tags.
Drop into the ordvec-formalization Mathlib project, `lake build`, resolve the
four debts above. de Finetti / rigidity theorems (rigidity_impossibility_proofs.md)
are NOT formalized here — they need Mathlib `ProbabilityTheory` and de Finetti,
which is a separate, larger project (de Finetti is not currently in Mathlib).
