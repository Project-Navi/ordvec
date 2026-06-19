# Path B — the chunk-length-mixture lake (deployment robustness, final arc)

The Phase B caveat split the unmodeled real-lake pathologies in two: OCR-garbage /
mixed-language (uncapturable from clean embeddings — needs a real dirty S3 sample)
and **chunk-length heterogeneity** (a *mixture of geometries* — capturable
synthetically). The directions writeup flagged the latter as **Path B**
(`oblivious_directions_results.md`, "chunk-length heterogeneity is itself a lake
pathology (folds into Path B)") and named chunk length a **third geometry axis**
(longer chunks → tokens average → vectors regress to the corpus mean → tighter cone,
lower ID). Path B tests both at once: does a real chunk-length *mixture* break b=4 raw
rank routing, and how big is the chunk-length geometry axis actually?

Domain is held constant (fiqa, 57.6k docs), only the chunk length varies — the clean
complement to Phase B, which held length constant and varied domain.

## Setup

Embed the SAME 57,600 fiqa docs (nomic-embed-text, gpu1) at four chunk-length caps
`{128, 256, 512, 1100}` chars, then union into one 230,388-doc lake
(`make_length_lake.py`). Queries: the full-length fiqa held-out set (1000) — a real
lake mixes *document* chunk lengths, not query lengths. Probe: the **same
`centering_recall.rs`** whose raw b=4 arm IS the incumbent that beat every
oblivious-structure idea. Metric: R@10 of b-bit rank codes vs **FP32-cosine top-10
recomputed on each corpus** — a routing-*fidelity* metric, so the 4× size difference
cancels (it hits the code arm and its own FP32 ground truth equally; no size confound).

## Pre-registration (locked before the lake run)

- **Path B fear CONFIRMED** (the first positive of the whole arc) if b=4 raw R@10 on
  the length mixture drops **≥ 0.02** vs the single-length fiqa@512 baseline.
- **FALSIFIED** (consistent with every prior lake fear) if flat to noise (|Δ| < 0.02).
- Secondary, pre-stated from the "third geometry axis" claim: longer chunks → tighter
  cone (higher R̄) and the strata cones point in *different* directions (a true mixture
  of geometries, not one).

## Result 1 — the chunk-length geometry axis is real but SMALL and CO-AXIAL

Per-stratum cone tightness R̄ (mean cosine to that stratum's centroid) and pairwise
centroid-axis cosine, all on the identical 57.6k docs:

| chunk cap | R̄ (cone tightness) | NaN/zero rows |
|-----------|--------------------|---------------|
| 128  | 0.7046 | 0 |
| 256  | 0.7151 | 0 |
| 512  | 0.7198 | 0 |
| 1100 | 0.7234 | 12 |

Cone-axis cosine between strata: **0.986–0.999** (128-vs-1100 = 0.9856; the rest tighter).

The predicted direction holds — longer chunks DO give a tighter cone (R̄ rises
monotonically 0.705→0.723) — but the magnitude is tiny: a 0.019 R̄ spread across an
8.6× chunk-length range, and the four cones are essentially **co-axial** (≥0.986). So
chunk length is a real third geometry axis but a *weak* one on this encoder/corpus: it
shifts cone tightness slightly along a shared axis, it does NOT produce the distinct,
differently-pointed cones the "mixture of geometries" framing imagined. (Caveat: fiqa
docs are short Q&A; a corpus with genuinely long source documents — where a 128-char
cap truly amputates content — could spread R̄ more. Established here only for
fiqa/nomic.)

## Result 2 — b=4 raw routing is IMMUNE to the length mixture (fear FALSIFIED)

`centering_recall` raw arm, 1000 fiqa queries, FP32-cosine top-10 ground truth:

| corpus | b=4 raw R@10 | b=4 raw CR@100 |
|--------|--------------|----------------|
| baseline fiqa@512 (57.6k) | 0.8230 | 1.0000 |
| **Path B length-mixture lake (230k, 4 lengths)** | **0.8253** | 1.0000 |
| **Δ** | **+0.0023** | 0.0000 |

Pre-registered ≥0.02 drop to confirm the fear. **Actual +0.002 — flat to noise, if
anything slightly up.** Candidate recall stays a perfect 1.0000: every FP32 top-10
neighbour is still captured in the top-100 by code distance on the 4×-larger mixed lake.

The centering signature is unchanged from single-length, too — b1 raw 0.516 → centered
+0.055, b2/b4 centered net-negative (b4 ΔR −0.079) — same PARTIAL pattern, same
capacity-scaling penalty as every prior corpus. The mixture changed neither the routing
fidelity nor the centering mechanism; it just reconfirmed both on 4× the data.

## Why the global bin edges survive a length mixture

`centering_recall`'s encoder is **per-dimension rank → equal-width bins** — the rank
transform is computed within each vector, so a slightly-tighter (long-chunk) or
slightly-looser (short-chunk) cone produces the same *ranks*; only the raw cosine
magnitudes differ, and ranks discard magnitude. The fixed-mass rank code has nothing
global for a tightness-shift to poison, exactly as the templated-hub immunity in Phase B
had nothing global (no IDF term) for a hub to poison. Training-free + rank-based is again
the property that confers the robustness.

## Verdict — Path B closes the synthetic-lake arc on the same negative

The last fear capturable without real dirty data is unfounded: a chunk-length mixture is
**not** a mixture of distinct geometries on fiqa/nomic (cones co-axial, R̄ spread 0.019),
and b=4 raw routing is **immune** to it (+0.002 R@10, CR 1.0). Combined with Phase B
(multi-domain union: lower-ID, globally centerable, hub-immune), every synthetic lake
pathology — multi-cone, templated-hub, and now multi-length — leaves the boring "spend
the bits, b=4, raw ranks" baseline intact.

CAVEAT (honest scope, unchanged): this is still *clean* embeddings of *curated* fiqa
text truncated to length. It models length-geometry heterogeneity; it does NOT model
OCR garbage, mixed-language, or broken-encoding sludge. The one remaining
non-derivative test is an actual dirty S3 sample (the OCR/multilingual half of the Phase
B caveat). Claim established: "a chunk-length mixture is benign for b=4 routing." Claim
NOT established: "raw dirty S3 sludge is benign."

## Reproduce
```
# 1. embed fiqa text truncated to each chunk-length cap (gpu1 nomic)
for N in 128 256; do python3 - <<PY
src=[l.rstrip('\n') for l in open('/tmp/corpora/fiqa_corpus.txt') if l.strip()]
open(f'/tmp/corpora/fiqa_$N.txt','w').writelines(l[:$N]+'\n' for l in src)
PY
  python3 embed_ollama.py --texts /tmp/corpora/fiqa_$N.txt \
     --out /tmp/corpora/fiqa_${N}_corpus.npy --host 100.67.101.76:11434 \
     --model nomic-embed-text
done
# (fiqa@512 = fiqa_corpus.npy and fiqa@1100 = fiqa_nomic1100_corpus.npy already exist)

# 2. union the four length strata into one lake
python3 make_length_lake.py \
   --parts fiqa_128_corpus.npy fiqa_256_corpus.npy fiqa_corpus.npy fiqa_nomic1100_corpus.npy \
   --out length_lake_corpus.npy --drop-nan

# 3. b=4 raw R@10 fidelity, baseline vs mixture (copy centering_recall.rs into examples/ first)
cargo run --release --example centering_recall -- --corpus-npy fiqa_corpus.npy        --queries-npy fiqa_q.npy
cargo run --release --example centering_recall -- --corpus-npy length_lake_corpus.npy --queries-npy fiqa_q.npy
```
