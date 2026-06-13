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

## REAL EMBEDDINGS (nomic-embed-text via ollama on RTX 5080, 768-d)

Corpus: 8665 real sentences extracted from this repo's markdown + Rust, embedded
with `nomic-embed-text` (GPU-resident via ollama). Generator:
`examples/embed_ollama.py`. Run: `density_collapse --corpus-npy repo_real.npy`.

First real-data facts:
- TwoNN intrinsic dimension of nomic-embed-text ≈ 13 (ambient 768) — our own
  measurement, squarely in the predicted low-tens range.
- Real embeddings are FAR more entangled at b=2 than synthetic clusters: mean
  nearest-code Hamming 314/768, and ~5083 of 8665 docs sit in each probe's
  b2-lookalike shell (vs ~60–120 synthetic). Real geometry, much denser collapse.

### CORRECTED after a second adversarial review (M1 + M2)

The first writeup reported a win-rate climb 0.667→0.930 with top-k as "the
signature of a real effect." TWO review findings invalidated that framing, and
the test was rewritten:

- **M2 (the win-rate climb was an artifact):** the per-probe tau GAP is flat
  across top-k; only the win-rate estimator's variance shrinks as k grows,
  mechanically pushing the win rate up. Win rate was the wrong statistic.
- **M1 (circularity):** tau was computed on the PROBE'S OWN top-k coords, which
  couples tau to cosine and makes the test near-tautological. Fixed to use the
  per-PAIR UNION of top coords, chosen independently of the cosine ranking.

Rewritten test reports the tau GAP (far − near) as an effect size, with a
bootstrap 95% CI over probes (de-circularized coords):

| top-k | tau gap (far − near) | 95% CI | verdict |
|-------|----------------------|--------|---------|
| 8  | 0.0420 | [0.0380, 0.0463] | signal |
| 16 | 0.0417 | [0.0381, 0.0454] | signal |
| 32 | 0.0440 | [0.0409, 0.0472] | signal |
| 64 | 0.0453 | [0.0424, 0.0483] | signal |

**Honest conclusion:** there IS a real effect — true neighbours have lower
intra-code Kendall-tau than the b2-lookalikes the code conflates them with, gap
≈ 0.04, CI strictly above 0 at every top-k. But it is MODEST and FLAT, not the
dramatic "sharpening" originally claimed. The win-rate monotonicity is retracted.
The lever (intra-code permutation order in the Rank code) shows a measurable but
small separation on this corpus/model; whether ~0.04 tau converts to useful
recall gain vs simply using b=4 is the unanswered deployment question.

## Caveats / open

- Real corpus here is repo-domain (md+rust), 8665 docs — narrow domain, modest
  size. Confirm on a larger, broader corpus (MS MARCO / Wikipedia passages).
- The test shows the signal EXISTS (separation in tau); it does not yet show a
  tau-rerank improves end-to-end recall vs simply using b=4. The honest next
  experiment: tau-rerank of b=2 survivors vs b=4 at matched bytes, R@10 vs FP32.
- Kendall-tau is computed on FP32 values restricted to the per-pair union of
  top-k coords; a deployable version computes it on the stored ranks.
- Single corpus, single model (nomic-embed-text), no cross-encoder check — do
  NOT read "confirmed" generality from one narrow corpus. Repeat on ≥1 more
  encoder before any strong claim.

Reproduce (synthetic):
```
cargo run --release --example density_collapse                 # noise=0.30
cargo run --release --example density_collapse -- --noise 0.10 # denser
```

Reproduce (real embeddings — full recorded procedure, was an E4 repro gap):
```
# 1. extract repo sentences (md + rust, 30..300 chars, >=4 words, deduped):
python - <<'PY'
import re, glob, os, tempfile
texts=[]
for f in glob.glob("**/*.md",recursive=True)+glob.glob("**/*.rs",recursive=True):
    if "target" in f.split(os.sep) or ".git" in f.split(os.sep): continue
    t=re.sub(r'```.*?```','',open(f,encoding='utf-8',errors='ignore').read(),flags=re.S)
    for line in re.split(r'(?<=[.!?])\s+|\n',t):
        s=line.strip(" #-*/>|`").strip()
        if 30<=len(s)<=300 and re.search(r'[a-zA-Z]{4}',s) and s.count(' ')>=4: texts.append(s)
seen=set(); out=[s for s in texts if not (s in seen or seen.add(s))]
open(os.path.join(tempfile.gettempdir(),"repo_texts.txt"),"w",encoding="utf-8").write("\n".join(out))
print(len(out),"sentences")
PY
# 2. embed via ollama nomic-embed-text (GPU) -> .npy:
python examples/embed_ollama.py --texts "$TMP/repo_texts.txt" --out "$TMP/repo_real.npy" --n 9000 --batch 128
# 3. run the probe:
cargo run --release --example density_collapse -- --corpus-npy "$TMP/repo_real.npy" --topk 32
```
