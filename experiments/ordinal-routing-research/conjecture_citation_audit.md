# Citation audit — prime/RH/embedding-geometry conjecture

Verified by direct WebFetch against primary sources (subagent sandbox had no web
egress; their "VERIFIED" tags were memory passes and are NOT relied on here).
WebSearch was backend-down; all checks below are WebFetch on known URLs.

## VERIFIED ✓ (fetched primary source)

| Claim | Verified value | Source |
|---|---|---|
| GPT-2 final-layer anisotropy | ≈0.99 ("almost perfect cosine similarity"); ~0.6 in layers 2–8, rising exponentially layers 8–12 | Ethayarajh 2019, arXiv:1909.00512 (via ar5iv) |
| Rogue dimensions dominate cosine | "a small number of rogue dimensions, often just 1–3, dominate these measures" (exact quote) | Timkey & van Schijndel 2021, arXiv:2109.04404 |
| Super-prime counting function | π_q(x) ~ x/(log x)²; q_n ~ n(log n)² | Broughan & Barnett 2009, J. Integer Seq. vol 12 |
| Mathlib CRT primitive | `ZMod.chineseRemainder : ZMod (m*n) ≃+* ZMod m × ZMod n` (exact signature) | mathlib4_docs ZMod/Basic |
| Discrepancy rates | uniform grid O(1/N); optimal LDS O(log N/N); random/Poisson O(√(log log N/N)) | Wikipedia Low-discrepancy_sequence |

## PARTIAL / QUALITATIVE (source confirms direction, not exact figure)

| Claim | Status |
|---|---|
| Mu & Viswanath D≈d/100 rule | Principle confirmed ("common mean + a few top directions"); the exact d/100 ratio NOT stated in fetched text. Cite as "a few top components", not d/100. arXiv:1702.01417 |
| Ansuini intrinsic dimension | NUMBERS CONFIRMED (ar5iv): last hidden layers ID ≈ 12–25; peak (early layers, rel.depth 0.2–0.4) ≈ 100–120; ID/ambient ≈ 2e-4. Hunchback = rise then contract. arXiv:1905.12784. NOTE: vision/CNN nets, not sentence encoders. |
| Valeriani et al. 2023 transformer ID | Qualitative confirmed: same expand–contract–stabilize ID profile in transformers (protein LM + image). Exact peak number not extracted. arXiv:2302.00294 |
| Routing implication | d_int ≈ low tens ⇒ JL R ≈ c·d_int ⇒ R∈{8,16} projections suffice (matches shard_recall envelope still rising at R=16). Projection budget sized by ~20, not ambient 256/1024. |

## CORRECTED (subagent confabulations caught)

- k-fold prime-cascade ~1/(log x)^k: **HEURISTIC, not proved.** Subagent falsely
  cited Goldston–Yıldırım (that paper is about small prime gaps — unrelated).
  Only the single iteration (Broughan&Barnett) is proved.
- Super-prime count "x/(ln x · ln ln x)": **wrong form**; paper proves x/(log x)².
- Prime-sequence discrepancy "O(n^-1/2)" (my own earlier claim): **unsupported**;
  no clean published rate for scaled primes. Verdict holds via uniform/quantile
  dominance regardless.

## STILL UNVERIFIED (needs working WebSearch or specific PDF)

- Exact intrinsic-dimension number for modern sentence embeddings
  (sentence-transformers / E5 / text-embedding-3). No confident published TwoNN
  figure located. This is the one empirical gap for the Level-2 anisotropy argument.
- Cai et al. 2021 "local isotropy" arXiv id unconfirmed.
