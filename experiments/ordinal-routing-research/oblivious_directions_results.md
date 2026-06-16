# Data-oblivious structure vs learned-encoder geometry — the directions arc

Second investigation round (follows the density/CRT/tau work). Question: can a
**data-oblivious low-discrepancy structure** — golden-angle, Fibonacci, Sobol,
Kronecker — improve ordvec's training-free routing? The probes (`*.rs`) and this
doc live in this directory; to run a probe, copy it into the crate's `examples/`
and `cargo run --release --example <name>` (same convention as the sibling docs).
**No crate or public-API changes.** Every probe was **pre-registered** (pass/fail
thresholds fixed in the source header BEFORE running) to prevent goalpost-moving,
and run on real embeddings.

## Corpora (real, nomic-embed-text 768-d via ollama on GPU)

| corpus | n docs | queries | domain | mean-resultant-length R̄ |
|--------|--------|---------|--------|--------------------------|
| repo   | 8,777  | 200     | md+rust | ~0.07 raw-cluster fixture* |
| fiqa   | 57,600 | 1,000   | financial QA | 0.69 (strongly coned) |
| nq     | 74,129 | 1,000   | Wikipedia QA | 0.69 |
| quora  | 150,000| 1,000   | question pairs | ~0.7 |

Ground truth = FP32 cosine top-10. *The repo corpus is the prior round's; the
financial/QA corpora are the breadth/scale test. NOTE: intrinsic dimension is a
**corpus** property, not an encoder constant — the repo corpus reads TwoNN≈13 but
fiqa reads ≈24 with the SAME encoder. See the "ID≈13 anchor was a corpus artifact"
correction in the cross-encoder section before quoting any single ID number.

## VERDICT (pre-registered, replicated)

**CLASS-DEAD.** Data-oblivious low-discrepancy directions do **not** beat iid-random
directions for training-free routing — not in ambient space, not in the centered
intrinsic subspace. The lever is **not the directions**.

## The chain of probes

### 1. `fib_directions` — golden-angle directions, AMBIENT sphere → TIE (with caveat)
Shared-index triple-tiled golden-angle directions vs iid-Gaussian, recall@k at
matched candidates-scanned. Across 10 seeds the gap is inside random's own
seed-to-seed noise (±0.03); the eye-catching seed-1 +5.8pts **evaporated under
reseeding** (a caught false positive). CAVEAT registered at the time: ambient
768-sphere is the wrong regime — concentration of measure makes iid-Gaussian
already near-uniform, so the test was pre-doomed regardless of golden's merit.

### 2. `uniformity_lemma` — why golden is dead on BIN EDGES (not directions)
The rank transform whitens each coordinate's marginal to **exactly uniform**, so
equal-width bucketing is **entropy-optimal** (measured: entropy = B exactly at
B∈{1,2,4}; golden boundaries 3.92 < 4.00 — they *strictly waste bits*). BUT this
lemma governs the **marginal only**; it is blind to the **joint** correlation
structure that projection directions act on. So it kills golden on bin edges and
says nothing about directions — which set up probe #4.
Correction banked: constant-composition is intrinsic to *any* fixed boundaries
on ranks, NOT unique to equal-width (the original claim was too strong).

### 3. `overlap_decomp` — the cone is ~100% of the bitmap-overlap excess
ordvec's hypergeometric null `n_top²/D` is **miscalibrated on real data**: random
pairs overlap ~4× the textbook prediction, and `cone_frac ≈ 1.000` — essentially
**all** the excess is the shared cone (hubness), not pairwise similarity.
Per-coord **mean-centering removes it** (cone_base collapses to the uniform null)
AND amplifies the true-neighbour overlap gap 2–5×. Replicated on fiqa and nq.
This is the one continuous, geometry-aligned intervention that *converted*.

### 4. `subspace_directions` — the directions test done RIGHT → CLASS-DEAD
Center → project to the populated k≈13-dim PCA subspace (where joint structure is
legible) → place R directions via {random, Sobol, Kronecker} → route. Fair
envelope (recall@10 at matched candidates):

| corpus | budget | random | sobol | kronecker | pca-axes (ceiling) |
|--------|--------|--------|-------|-----------|--------------------|
| fiqa   | 8000   | 0.1967 | 0.1997| 0.2001    | 0.1946 |
| nq     | 8000   | 0.1836 | 0.1830| 0.1817    | 0.1877 |
| quora  | 8000   | 0.1004 | 0.1035| 0.1061    | 0.1073 |

Max margin anywhere **+0.006** (threshold for a win was +0.02), across all three
corpora. Decisive detail:
the **data-dependent ceiling (pca-axes) itself barely beats random** → the
directions are not a lever for *anyone* on this data. With ID≈13 and a cone, the
populated subspace is so dense that any ~13–128 directions cover it equally; there
is no discrepancy gap for quasi-random structure to close. The Kronecker/Sobol
**hybrid was NOT built** — it required distinct-regime component wins; there were
none.

## centering's recall verdict (`centering_recall`, pre-registered)

Centering as a RankQuant encoding change: full-scan R@10 vs FP32 cosine, raw vs
centered, matched bytes.

| corpus | Δb1 | Δb2 | Δb4 | verdict |
|--------|-----|-----|-----|---------|
| repo   | +0.038 | −0.044 | −0.108 | PARTIAL |
| fiqa   | +0.066 | −0.017 | −0.075 | PARTIAL |
| nq     | +0.051 | −0.016 | −0.083 | PARTIAL |
| quora  | +0.027 | −0.022 | −0.061 | PARTIAL |

Centering **helps at b=1, hurts at b≥2**, replicated across 3 corpora. It removes
the additive cone (good for top-bucket coverage / coarse routing) but rotates the
rank order away from raw-cosine (bad for full-order fidelity at higher bits). It
is a **low-bit-prefilter / coarse-routing tool, not a fine-scan tool** — and
explicitly does NOT survive at the incumbent b=4.

## The through-line

Every oblivious *discrete* structural prior tried across both rounds — golden,
Fibonacci, prime/spectral, Sobol, Kronecker, low-discrepancy directions — is
**inert against learned continuous embedding geometry**. The data is a smooth
~13-dim cone; discrete combinatorial structure has no purchase on it. The only
interventions that converted were **continuous and geometric** (centering / cone
removal), and only in the low-bit regime. "Just spend bits (b=4, raw ranks)"
remains the baseline that beats every oblivious-structure idea.

Even centering — the one continuous intervention that *partially* converted (b=1
prefilter, +0.03–0.07 R@10) — does not survive at b=4 and does not make a better
partition key. The cone is real and removable, but removing it trades cosine
fidelity for balance, and routing wants the fidelity.

## Scope & boundary of this negative (checkpoint)

This is a deliberate stopping point: the hypothesis space had one degree of
freedom — *which oblivious structure* on *which axis* (bin-edges / directions /
partition key) — and both ends of both axes have now been tested, **including the
data-dependent PCA ceiling**. The load-bearing result is that the ceiling itself
barely beats random: that kills the *axis*, so any further sequence is a
**derivative** that inherits the null. You cannot escape a null by swapping the
sequence when even the optimal member ties.

Every result here is conditioned on **off-the-shelf `nomic-embed-text`**: a
learned, anisotropic (R̄≈0.69), low-intrinsic-dim (≈13) encoder. The honest claim
is therefore scoped: *data-oblivious combinatorial structure is inert against
learned anisotropic embedding geometry of this kind.* The only **non-derivative**
next experiment is to change that conditioning variable — a different encoder
(different anisotropy / intrinsic dim), or a sparsity/ordinal-trained encoder
where ranks are the native code and discrete structure might finally seat.
Everything inside the current encoder is a derivative of what is already
falsified here.

## Partition balance (`partition_balance`, pre-registered) — BALANCE pass, PRUNING fail

The one round-on-round thread: does cone-removal make a balanced, prunable coarse
shard key? Sign-pattern key over the top-`bits_key` PCA axes (basis shared across
arms; only centering differs). Swept bits_key ∈ {6,10,14} on fiqa.

| sub-claim | threshold | result |
|-----------|-----------|--------|
| BALANCE — largest-cell-fraction reduction | ≥1.5× | **PASS** (1.6–2.2×; Gini 0.60→0.38 at bits_key=10) |
| PRUNING — candidates at matched recall≥0.90 | ≤0.80× | **FAIL** (same recall-vs-candidates envelope) |

Centering de-hubs the cells (more balanced, uses all cells) but at matched recall
scans the **same** candidates — better balance did NOT convert to better pruning.
Mechanism: balance is a *marginal* property of the key; pruning quality depends on
cell-to-neighbour *alignment*, and centering rotates the key away from cosine (the
same reason it fails recall at b≥2). Cells spread along the wrong axis don't prune
better. This was the last live thread; it closes for the session's recurring
reason.

## Cross-encoder extension (the non-derivative move) — IN PROGRESS

Every result above is conditioned on nomic (ID≈13, R̄≈0.69). The only
information-adding next step is changing the encoder. Geometry gate (8k fiqa
sample, TwoNN intrinsic dim + mean-resultant-length R̄) — run BEFORE any probe to
confirm the encoders genuinely differ from nomic, else a re-run is derivative:

First-pass gate (512-char chunks) measured: nomic 768d ID≈13 R̄0.69; all-minilm
384d ID21.7 R̄0.28; mxbai 1024d ID24.3 R̄0.69; snowflake-v1 1024d ID31.1 R̄0.74.

**Methodology correction (capacity confound).** 384-dim (all-minilm) cannot hold
meaning above ~400-char chunks, and our chunks were capped at 512 — so minilm's
geometry was capacity-corrupted, not a clean ID point, and does not belong on the
ladder. Floor set to **768-dim** encoders and chunk cap raised to **1100 chars**
(BGE-768 capacity ceiling). Revised capacity-honest set (all ≥1024-dim, all clear
the floor): **bge-m3, bge-large, snowflake-arctic-embed-v2**, plus
**harrier-oss-v1-0.6b** (the operator's actual daily-driver encoder). Model facts
verified against the ollama host + HF config (not recalled): harrier =
Qwen3-based, **1024-dim, 32k context** (Q8 GGUF); ingests 1100-char chunks with
zero truncation. Re-embedding fiqa at 1100-char chunks; ladder rebuilt on these.
(For code corpora, the right encoder is jina-code — noted for a future code-domain
run; not in this text-corpus ladder.)

PRE-REGISTERED prediction (fixed before the full-corpus probes run):
- The session's mechanism is "ID≈13 is so dense any directions cover it equally."
  If TRUE, then as ID climbs to 31 the subspace sparsifies and a discrepancy gap
  should OPEN → Sobol/Kronecker begin to beat random, and centering's b=4 penalty
  shrinks (less cone). Verdict per model by the SAME thresholds as above
  (directions: +0.02 at matched budget; centering: +0.02 at b4).
- If oblivious structure stays inert even at ID=31 / low-anisotropy minilm → the
  negative generalizes across the realistic ID range of text encoders (stronger).

### Ladder results (fiqa 57.6k, 1100-char chunks, capacity-honest)

| encoder | dim | intrinsic dim | centering Δb4 | sobol−random @8k | pca-axes ceiling |
|---------|-----|---------------|---------------|------------------|------------------|
| nomic@512 | 768 | ~13 | −0.075 | ~tie | ≈ random |
| bge-m3 | 1024 | 21.4 | −0.106 | +0.006 (flicker +0.024@4k) | leads (0.216) |
| bge-large | 1024 | 22.9 | −0.155 | **−0.036** | leads (0.255) |
| snowflake-arctic-v2 | 1024 | 18.0 | −0.088 | +0.010 | leads (0.237) |
| snowflake-arctic-v2 | 1024 | 18.0 | −0.088 | +0.010 | leads (0.237) |
| harrier-oss-0.6b | 1024 | 21.4 | −0.069 | +0.002 | leads (0.252) |
| nomic@1100 (same fiqa) | 768 | 23.1 | −0.082 | (n/a) | — |
| nomic@512 (same fiqa) | 768 | 24.3 | (≈ above) | — | — |

### CORRECTION — the "ID≈13" anchor was a CORPUS artifact, not an encoder fact

The "nomic intrinsic dim ≈ 13" that motivated the whole "dense low-dim subspace"
mechanism came from the **repo corpus** (8.7k md+rust sentences), NOT fiqa. On a
FIXED corpus (fiqa 57.6k), nomic's ID is **24.3 @512 / 23.1 @1100** — and *every*
encoder lands ID ~18–24. The apparent "ID ladder 13→31" was mostly **corpus
differences masquerading as encoder differences** (nomic-on-repo vs others-on-fiqa).
On one corpus the ladder is nearly flat. **Intrinsic dim is set more by the corpus
than the encoder**, and chunk length (512→1100) barely moved it (24.3→23.1) — the
opposite of the predicted large drop. Both my "ID≈13" anchor and my "longer chunks
→ much lower ID" prediction were wrong; recorded here rather than quietly dropped.

What SURVIVES the correction (now properly controlled — same corpus, varied encoder):
- **Directions CLASS-DEAD** across all 5 encoders at the ID range real text
  encoders actually produce (~18–24). The negative was never about low ID; it is
  general. No revival anywhere.
- **Centering b4 dead** across all 5 (−0.07 to −0.15); harrier + nomic@1100 both
  PARTIAL (b1 help, b4 fail) — same pattern as the original runs.
- **pca-axes (data-aligned) leads** at every encoder (0.24–0.25 vs random ~0.19):
  the lever is data-alignment, which training-free forbids. The robust positive.

**Directions: CLASS-DEAD, confirmed across the ID ladder.** sobol−random flips
sign per corpus (+0.006, −0.036, +0.010) — noise around zero, never ≥0.02 in ≥2
corpora; the bge-m3 flicker did NOT replicate at near-identical ID (bge-large 22.9).
The pre-registered ≥2-corpora rule holds the line.

**The one robust high-ID effect (a real, positive, scoped finding):** the
data-aligned ceiling (pca-axes) clearly LEADS random at every high-ID encoder
(0.22–0.26 vs ~0.18–0.20), unlike nomic where it barely beat random. So higher
intrinsic dim genuinely opens room on the directions axis — but ONLY
data-dependent directions exploit it; oblivious (sobol/kronecker) cannot. The
lever at high ID is **data-alignment**, which is exactly what training-free
forbids. This locates *why* oblivious structure fails rather than just restating
that it does.

**Centering b4: dead across all encoders, penalty SCALES with capacity** (−0.08 to
−0.15) — higher-capacity encoders carry more cosine order for centering to
corrupt. Most robust negative in the set.

harrier-oss-0.6b (operator's daily driver) + nomic@1100 (chunk-length control)
embedding; results to follow.

**Chunk length is a THIRD geometry axis (not a free parameter).** Longer chunks
average more tokens → vectors regress toward the corpus mean → R̄ rises (tighter
cone), intrinsic dim falls. So nomic@512 and nomic@1100 are *different geometries*,
and every 512-char result above is conditioned on that one chunk length. We
re-embed nomic@1100 on the SAME 57.6k fiqa docs to measure the shift directly
(two-point: ID and R̄ delta). Deployment consequence: a real enterprise lake has
documents at every chunk length at once → a *mixture* of geometries, not one — so
chunk-length heterogeneity is itself a lake pathology (folds into Path B).

## Reproduce
```
# embed real corpora via ollama nomic-embed-text -> .npy (see REAL_CORPUS_RUNBOOK.md)
cargo run --release --example uniformity_lemma    -- --corpus-npy CORPUS.npy
cargo run --release --example overlap_decomp      -- --corpus-npy CORPUS.npy --queries-npy Q.npy
cargo run --release --example centering_recall     -- --corpus-npy CORPUS.npy --queries-npy Q.npy
cargo run --release --example subspace_directions  -- --corpus-npy CORPUS.npy --queries-npy Q.npy --kdim 13
```
