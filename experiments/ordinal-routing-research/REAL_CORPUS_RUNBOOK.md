# Real-corpus runbook for the conjecture probes

All four examples now consume the SAME `.npy` format `bench_rank` documents:
2-D little-endian float32 (`<f4`), C-order, shape `(n, dim)`. One corpus dump
flows through the entire investigation. No BLAS, no Python at run time.

Recommended public corpora (per bench_rank header): GloVe, OpenAI
text-embedding-3 dumps, or any sentence-transformer output saved as .npy.

## 1. Intrinsic dimension — sizes the projection budget
```
cargo run --release --example twonn_id -- --corpus-npy corpus.npy
```
Reports chord-metric TwoNN ID. Expect low tens (READ AS LOWER BOUND — finite
sample deflates above ~12). If ID≈d_int, the routing layer wants R≈c·d_int
projections, i.e. R∈{8,16}.

## 2. Number variance — is the routing key rigid or Poisson?
```
cargo run --release --example spectral_probe -- --corpus-npy corpus.npy
cargo run --release --example spectral_probe -- --corpus-npy corpus.npy --unfold-empirical
```
Σ²(L)/L flat≈1 ⇒ Poisson; climbing ⇒ clustered; falling ⇒ rigid (the only
result that would reopen the spectral conjecture). --unfold-empirical confirms
quantile tiling balances the key (Σ²→0).

## 3. Shard recall — does the oblivious router work; does coprime help?
```
cargo run --release --example shard_recall -- \
    --corpus-npy corpus.npy --queries-npy queries.npy
```
Needs BOTH files (real queries for honest recall). Fair envelope = recall@k at
equal candidates-scanned. Watch: does recall keep climbing R=1→16 (sets the
budget), and does coprime/random-offset beat plain aligned (predicted: no).

## 4. Headline retrieval quality (the existing bench)
```
cargo run --release --example bench_rank -- \
    --corpus-npy corpus.npy --queries-npy queries.npy --queries 200 --k 10
```

## Expected story on real embeddings (prior)

Consistent with synthetic + verified literature: ID low tens → R∈{8,16};
key Poisson/clustered not rigid → quantile bucketing; coprime adds nothing →
R shared-width random projections is the router. A FALLING Σ²(L)/L or a
coprime>random-offset gap that survives reseeding would be the surprise worth
chasing.
