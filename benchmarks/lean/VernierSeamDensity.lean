/-
Copyright (c) 2026 ordvec authors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: (ordvec contributors)
-/

-- NOTE: does NOT build on the Finite*Quotient stack. That layer is about
-- fiber-invariance/sufficiency of quotient maps (Ω → Z); this theorem is a CRT
-- bijection + cardinality count, a different toolkit. Standalone module in the
-- OrdvecFormalization namespace, importing Mathlib ZMod directly (the repo does
-- not otherwise use ZMod — confirmed against its module list).
import Mathlib.Data.ZMod.Basic
import Mathlib.Data.Nat.GCD.Basic
import Mathlib.Data.Fintype.Pi
import Mathlib.Data.Fintype.BigOperators

open scoped BigOperators

namespace OrdvecFormalization

/-!
# Vernier seam density (within-axis coprime grids)

For pairwise-coprime grid periods `m : Fin L → ℕ` on a single routing axis, the
positions in `ZMod M` (`M = ∏ m i`) that lie within band `t` of a seam on EVERY
grid number exactly `∏ i, (2t+1)`, hence density `∏ i, (2t+1)/(m i)` when
`2t+1 ≤ m i`. Verified exhaustively in `examples/crt_seam_oracle.rs`; the closed
form is proved on paper in `benchmarks/crt_seam_oracle_results.md`.

Structure mirrors the repo's quotient idiom: the CRT residue map is a
`productMap`-style coordinate split, and pairwise coprimality is what makes the
position-space map a bijection (the only place coprimality is used).

DRAFT — drafted without a Lean toolchain. `sorry`s mark confirmed proof debt,
ranked at the file end. Lean 4.28.0 / Mathlib pinned by the repo `lake-manifest`.
-/

/-! ## Per-grid admissible residues -/

/-- Residues mod `m` within centered band `t` of `0`, as the image of the
    window `{0,…,2t}` shifted by `-t`. -/
def admissible (m t : ℕ) : Finset (ZMod m) :=
  (Finset.Icc 0 (2 * t)).image (fun k => (k : ZMod m) - (t : ZMod m))

/-- When the window fits (`2t+1 ≤ m`) the shift map is injective on it, so the
    admissible set has exactly `2t+1` elements. -/
theorem card_admissible {m t : ℕ} (h : 2 * t + 1 ≤ m) :
    (admissible m t).card = 2 * t + 1 := by
  -- card_image_of_injOn + injectivity of k ↦ (k:ZMod m) - t on a width-(2t+1)
  -- window (no wraparound since width < m). Routine ZMod.val arithmetic.
  sorry

/-! ## Product-space count (coprimality NOT used) -/

/-- The joint blind spot over the residue PRODUCT space has cardinality
    `∏ (2t+1)`. True for any family — carries no coprimality hypothesis, which
    is the honest statement of what this half proves (it is `Fintype.card_piFinset`,
    not CRT). -/
theorem blindspot_card_product
    {L : ℕ} (m : Fin L → ℕ) (t : ℕ) (hband : ∀ i, 2 * t + 1 ≤ m i) :
    (Finset.univ.filter
        (fun r : ∀ i, ZMod (m i) => ∀ i, r i ∈ admissible (m i) t)).card
      = ∏ i, (2 * t + 1) := by
  have hfilter :
      (Finset.univ.filter (fun r : ∀ i, ZMod (m i) => ∀ i, r i ∈ admissible (m i) t))
        = Fintype.piFinset (fun i => admissible (m i) t) := by
    ext r; simp [Fintype.mem_piFinset]   -- [VERIFY] Fintype.mem_piFinset
  rw [hfilter, Fintype.card_piFinset]    -- [VERIFY] Fintype.card_piFinset
  exact Finset.prod_congr rfl (fun i _ => card_admissible (hband i))

/-! ## CRT bijection (the only place coprimality is used) -/

/-- Indexed CRT equivalence `ZMod (∏ m i) ≃ ∏ i, ZMod (m i)` for a
    pairwise-coprime family. [CRUX] Construct by folding `ZMod.chineseRemainder`
    over `Fin L`, or via an existing indexed iso. This is the load-bearing
    object and the sole consumer of `cop`.

    `[∀ i, NeZero (m i)]` is REQUIRED, not cosmetic: without it some `m i = 0`
    gives `ZMod 0 = ℤ` (infinite), so the cardinalities of the two sides do not
    match and the Equiv is FALSE. (Caught in adversarial review.) Downstream this
    is discharged from `2t+1 ≤ m i ⇒ 1 ≤ m i ⇒ NeZero (m i)`. -/
def crtEquiv {L : ℕ} (m : Fin L → ℕ) [∀ i, NeZero (m i)]
    (cop : ∀ i j, i ≠ j → Nat.Coprime (m i) (m j)) :
    ZMod (∏ i, m i) ≃ (∀ i, ZMod (m i)) :=
  sorry

/-! ## Position-space count (the real theorem) -/

/-- Positions `x ∈ ZMod M` (M = ∏ m i) within band `t` of a seam on EVERY grid
    number exactly `∏ (2t+1)`. Transports `blindspot_card_product` across
    `crtEquiv`; coprimality is essential here (it is what makes `crtEquiv` a
    bijection). Phase offsets translate each per-grid set without changing its
    cardinality, so the count is phase-independent. -/
theorem blindspot_card_positions
    {L : ℕ} (m : Fin L → ℕ) [∀ i, NeZero (m i)]
    (cop : ∀ i j, i ≠ j → Nat.Coprime (m i) (m j))
    (t : ℕ) (hband : ∀ i, 2 * t + 1 ≤ m i) :
    (Finset.univ.filter
        (fun x : ZMod (∏ i, m i) =>
          ∀ i, (crtEquiv m cop x) i ∈ admissible (m i) t)).card
      = ∏ i, (2 * t + 1) := by
  have hcongr :
      (Finset.univ.filter
          (fun x : ZMod (∏ i, m i) => ∀ i, (crtEquiv m cop x) i ∈ admissible (m i) t)).card
        = (Finset.univ.filter
            (fun r : ∀ i, ZMod (m i) => ∀ i, r i ∈ admissible (m i) t)).card := by
    -- Pushforward of the filter along the bijection crtEquiv. The position-space
    -- predicate is literally (product-space predicate) ∘ crtEquiv, so transport
    -- by `Fintype.card_congr (Equiv.subtypeEquiv (crtEquiv m cop) (fun _ => Iff.rfl))`
    -- over the corresponding subtypes (or `Finset.card_nbij'` with crtEquiv /
    -- crtEquiv.symm). NOTE: `Equiv.card_filter_map` does NOT exist (review).
    sorry
  rw [hcongr]; exact blindspot_card_product m t hband

/-- Oracle Check 2: exactly one all-grids coincidence per period (`t = 0`).
    `NeZero (m i)` follows from `1 ≤ m i`; we take it as an instance argument
    consistent with `crtEquiv`'s requirement. -/
theorem unique_coincidence_per_period
    {L : ℕ} (m : Fin L → ℕ) [∀ i, NeZero (m i)]
    (cop : ∀ i j, i ≠ j → Nat.Coprime (m i) (m j))
    (hpos : ∀ i, 1 ≤ m i) :
    (Finset.univ.filter
        (fun x : ZMod (∏ i, m i) =>
          ∀ i, (crtEquiv m cop x) i ∈ admissible (m i) 0)).card = 1 := by
  simpa using blindspot_card_positions m cop 0 (by simpa using hpos)

/-! ### Proof debt, ranked (post-review)
  1. `crtEquiv` — indexed CRT bijection; fold `ZMod.chineseRemainder` over
     `Fin L`. THE crux, sole consumer of `cop`. Now carries the REQUIRED
     `[∀ i, NeZero (m i)]` (without it `ZMod 0 = ℤ` makes the Equiv false).
  2. `card_admissible` — window injectivity (routine `ZMod.val`). Statement
     confirmed correct by review (exact `2t+1` under `2t+1 ≤ m`).
  3. `hcongr` — cardinality transport via `Equiv.subtypeEquiv` + `Fintype.card_congr`
     (NOT `Equiv.card_filter_map`, which does not exist).
  4. [VERIFY] names: `Fintype.mem_piFinset`, `Fintype.card_piFinset` (both resolve
     per review). `Finset.Icc` is over ℕ then cast — no type error.
  Product half `blindspot_card_product` is the ONLY structurally-complete piece
  (modulo 4); the intended theorem (B) is entirely deferred to crux (1). Do not
  represent this as "most content proved." -/

end OrdvecFormalization
