# Why oblivious random-projection keys cannot be rigid — proofs, not corpora

The zoo gives evidence; these give the reason. Three claims, stated and proved.
"Rigid" = number variance Σ²(L) grows sub-linearly (o(L)); "Poisson-or-worse"
= Σ²(L) = Ω(L). The routing key is K_r(x) = r·x / ||x|| for a probe direction r
(cosine geometry: the key is a function of the RAY, magnitude-free).

------------------------------------------------------------------------
## Lemma 0 (magnitude is quotiented out)

For any x ≠ 0 and scalar c > 0, K_r(cx) = K_r(x). So any structure a corpus
encodes purely in vector NORMS is invisible to the key. In particular the
`projected-rigid` construction x_i = t_i·u collapses under normalization to
sign(t_i)·u ∈ {+u, −u}: 50k rigid magnitudes become 2 distinct keys. Measured
Σ²/L = 1936 (degenerate), exactly as the quotient predicts. ∎

Consequence: rigidity must live in the ANGULAR distribution on the sphere
S^{d-1}, or it does not exist for the key.

------------------------------------------------------------------------
## Theorem 1 (isotropy ⇒ exact Poisson key)

Let x be distributed so that x/||x|| is uniform on S^{d-1} (isotropic). For a
fixed unit r, the projected key K_r = r·(x/||x||) has the known marginal with
density ∝ (1−s²)^((d-3)/2) on [−1,1]. Drawing n i.i.d. corpus points gives n
i.i.d. key values from a fixed continuous law F. After unfolding by F, the
points are a homogeneous Poisson process of rate 1, for which Σ²(L) = L exactly.

Proof: i.i.d. samples under their own CDF are uniform on [0,1]; counts in
disjoint windows have Binomial counts → Σ²(L) = L(1−L/n) (≈ L for L ≪ n). ∎
(Matches isotropic zoo row: Σ²/L = 0.99, flat.)

------------------------------------------------------------------------
## Theorem 2 (i.i.d. corpus ⇒ Σ²(L) = L(1−L/n), NO rigidity, for ANY distribution)

This is the special case Var_θ(p) = 0 of Theorem 3 (i.i.d. = degenerate mixing
measure). Let corpus points be i.i.d. from ANY distribution D on R^d (isotropic
or not, clustered, manifold, heavy-tailed). Fix r. The keys K_r(x_1..x_n) are
i.i.d. from the induced 1-D law F_r. Unfolding by F_r (probability integral
transform) gives n i.i.d. Uniform[0,1] points.

Proof. n i.i.d. uniforms form a BINOMIAL point process (NOT Poisson — review
correction). A window of width L/n has count N ~ Binomial(n, L/n), so
    Σ²(L) = Var(N) = n·(L/n)(1 − L/n) = L(1 − L/n).
Thus Σ²(L) = L(1−L/n) = Θ(L): sub-LINEAR rigidity (Σ² = o(L)) is impossible. ∎

NOTE (corrected): the earlier "homogeneous Poisson, Σ²=L exactly, independent
window counts, Var=mean=L" was WRONG — that is the Poisson framing this work
exists to debunk. For fixed n the process is binomial; counts in disjoint
windows are NEGATIVELY correlated (they sum to n), and Σ²(L)=L(1−L/n) < L. The
CONCLUSION (no sub-linear rigidity) is unaffected because the value is Θ(L).

This is why clustered/anisotropic/rogue/manifold are not sub-linear: by Thm 3
the mixing variance only ADDS (Σ² ≥ L(1−L/n), and clustering pushes above L).
Genuine rigidity (Σ² = o(L)) needs a negatively-associated, FINITELY-exchangeable
generator (lattice / determinantal point process) — see Thm 3's escape hatch —
which a fixed data distribution does not produce.

------------------------------------------------------------------------
## Theorem 3 (no rigidity for any fixed-distribution corpus — full derivation)

Setup. Corpus rows exchangeable ⇒ keys exchangeable ⇒ unfolded positions
U_1..U_n exchangeable on [0,1]. Window W of length L/n, so E[count]=L. Let
I_j = 1{U_j ∈ W}, N = Σ I_j, p = L/n. By exchangeability every I_j has mean p
and every pair shares one covariance c = Cov(I_1,I_2):

    Var(N) = n·p(1−p) + n(n−1)·c.                                    (★)

de Finetti (infinite exchangeable ⇒ conditionally i.i.d. given latent θ):
I_j are independent Bernoulli(p(θ)) given θ, so
    c = Cov(I_1,I_2) = E[p(θ)²] − p̄² = Var_θ(p(θ)) ≥ 0,   p̄ = E[p(θ)].
Substituting into (★):

    Σ²(L) = Var(N) = n·p̄(1−p̄) + n(n−1)·Var_θ(p(θ))
                   ≥ n·p̄(1−p̄) = L(1 − L/n).                          ∎

So Σ²(L) ≥ L(1−L/n): rigidity (sub-Poisson, Σ²=o(L)) is IMPOSSIBLE because the
mixing-variance term Var_θ(p(θ)) is non-negative and can only RAISE the count
variance above the Poisson line. Clustering = large Var_θ(p(θ)) = strict excess.

THE PRECISE ESCAPE HATCH (named, not gestured at). de Finetti requires INFINITE
exchangeability. Finitely-exchangeable sequences (sampling WITHOUT replacement
from a fixed finite set — i.e. a lattice / determinantal point process) violate
the c ≥ 0 step: there c < 0 and Σ²(L) < L (genuinely rigid). But "drawn from a
fixed data distribution" IS conditional-i.i.d.-given-θ — the infinite-exchangeable
case — so any corpus an encoder produces from a data distribution falls under
the theorem. Rigidity requires a negatively-associated generator (lattice/DPP)
AND preserved row order; an oblivious router over a fixed distribution has
neither. ∎

------------------------------------------------------------------------
## What this proves about the conjecture (NARROWED post-review)

The original hope was that prime/spectral structure could exploit *rigidity* in
the routing key. Theorems 1–3 show the key has Σ²(L) ≥ L(1−L/n) for any
conditionally-i.i.d. (mixture-of-i.i.d.) corpus — equality at isotropy, strict
excess under clustering. So:

  **The routing key is not number-variance-rigid.** Sub-linear Σ²(L) requires a
  finitely-exchangeable / negatively-associated generator (lattice/DPP) AND
  preserved row order — neither holds for an oblivious router over a fixed
  distribution.

That is the defensible claim, and it is all the mathematics supports.

RETRACTED (non-sequitur, flagged in adversarial review). The earlier conclusion
— "quantile bucketing is optimal against the entire achievable class; no
spectral/prime partition can beat it because the structure is provably not
there" — DOES NOT FOLLOW from Σ²(L) ≥ L. Number variance and partition
quality (load balance / recall) are different figures of merit; Σ² ≥ L says
nothing about whether some partition outperforms quantile bucketing. A Poisson
key can still admit a useful partition. The impossibility result refutes only
the specific "exploit rigidity" route, NOT all spectral/prime partitions, and
NOT in favor of quantile-optimality-over-all-partitions. Establishing the latter
would need a separate argument (that partition quality is monotone in Σ²), which
is not provided and is likely false.

(The zoo runs in withdrawn/corpus_zoo_results.md were EVIDENCE for the narrow claim, with
the unfolding caveat in ADVERSARIAL_REVIEW.md. The selftest Σ²=0 only shows the
estimator can represent the o(L) regime; Thm 3 says a fixed-distribution corpus
cannot reach it.)
