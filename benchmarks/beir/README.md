# ordvec BEIR benchmark harness

Reproducible evaluation of ordvec's rank/sign retrieval on standard
[BEIR](https://github.com/beir-cellar/beir) datasets — quality (nDCG@10 vs
qrels) and latency (single-query / batched / threaded) — against an exact
inner-product baseline and a pure-Rust HNSW. The shared encoder is Microsoft
**Harrier** (`harrier-oss-v1-0.6b`, 1024-dim), run as GGUF `Q8_0`.

All latency is measured in **one Rust process** (`benchmarks/beir-bench`); Python
only embeds the corpus, scores nDCG against qrels, and renders the figures.

## Claims discipline

> **Benchmark numbers in this repository reflect synthetic or user-runnable
> real-corpus experiments only.  No numbers are fabricated or cherry-picked.
> Every result file produced by `make benchmark-beir` is fully reproducible
> from the commands documented here, using publicly available BEIR datasets and
> the pinned encoder revision recorded in `embeddings.manifest.json`.**

> **The `flat` baseline is an exact full-float inner-product search (identical
> retrieval to FAISS `IndexFlatIP`) used for comparison purposes — it is NOT
> ground truth.  nDCG@10 is computed against the official BEIR qrels
> (human-annotated relevance judgements), not against the `flat` results.
> Recall-vs-`flat` is an optional diagnostic only; it does not substitute for
> qrel-based evaluation.**

## Dataset suite

| Dataset    | Domain                        | #Queries | #Corpus |
|------------|-------------------------------|---------:|--------:|
| scifact    | Scientific claim verification | 300      | 5,183   |
| nfcorpus   | Biomedical IR                 | 323      | 3,633   |
| fiqa       | Financial QA                  | 648      | 57,638  |
| trec-covid | COVID-19 literature           | 50       | 171,332 |

Datasets are downloaded automatically on first run by a small vendored BEIR
reader (no `beir` PyPI package — it pulls an unbuildable `pytrec_eval`). The
default `make benchmark-beir` reproduces **scifact** (quality) + **trec-covid**
(scaling + latency); `nfcorpus`/`fiqa` are supported via `QUALITY_DATASETS=...`.

## Encoder

**Harrier (`harrier-oss-v1-0.6b`)** — a 600M-parameter bi-encoder producing
1024-dimensional L2-normalised float32 embeddings. The canonical lane runs the
**GGUF `Q8_0`** weights via `llama-cpp-python` (CUDA), last-token pooled.

- Documents receive no instruction prefix.
- Queries are prefixed with
  `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "`.
- The exact repo/file/quant + library versions are recorded in
  `embeddings.manifest.json` per cache directory.

Optional alternate encoder lanes (heavier; off by default): sentence-transformers
(`make bench-beir-prepare-st`) and Ollama (`make bench-beir-prepare-ollama`).

## Quick start

```bash
make bench-beir-setup       # Python deps + CUDA llama-cpp-python (built from source)
make benchmark-beir-smoke   # quick end-to-end sanity (scifact only)
make benchmark-beir         # quality (nDCG) + scaling sweep + three figures
```

`bench-beir-setup` installs `requirements.txt` and then builds `llama-cpp-python`
against the host CUDA toolkit (`CMAKE_ARGS="-DGGML_CUDA=on"`; override
`LLAMA_CMAKE_ARGS=` for a CPU-only build).

## Methods (all measured in the Rust harness)

| Method            | Bytes/vec | Description |
|-------------------|----------:|-------------|
| `flat`            | 4096      | Exact inner product (== FAISS `IndexFlatIP` math), pure-Rust SIMD GEMM. **Baseline, not ground truth.** |
| `hnsw`            | 4096 + graph | Pure-Rust HNSW (`hnsw_rs`, M=32, ef=128) — portable stand-in for C++ hnswlib. The graph is implementation-owned side storage, not included in the 4096-byte float-vector payload. |
| `rq2`             | 256       | RankQuant 2 bits/dim, asymmetric float-query LUT scan. |
| `rq4`             | 512       | RankQuant 4 bits/dim, asymmetric float-query LUT scan. |
| `bitmap-rq2`      | 384       | Two-stage: Bitmap candidate-gen → RankQuant-2 rerank. |
| `sign-rq2`        | 384       | Two-stage: SignBitmap candidate-gen → RankQuant-2 rerank. |

Thread/batch knobs (per `beir-bench`): `--threads N` pins query latency to a
rayon pool of N threads (index build still uses all cores); `--max-docs M`
sub-samples the corpus for the scaling sweep; `--batch` sets the matched batch.
The committed README figures use the default method set from the top-level
`Makefile`; they do not yet include the newer `sign-rq2-threaded` probe row.
Regenerate and review the public tables before using that probe for release
claims.

## Cache layout

One encoder run produces a directory per dataset/split:

```
.cache/ordvec-beir/<dataset>/<split>/encoder=<slug>/
    corpus.f32.npy           # float32 (n_docs, 1024), L2-normalised, C-order
    queries.f32.npy          # float32 (n_queries, 1024), L2-normalised, C-order
    corpus_ids.json          # list[str], sorted(corpus.keys())
    query_ids.json           # list[str], sorted(qrels.keys())
    qrels.json               # {qid: {doc_id: int_relevance}}
    texts.manifest.json      # raw-text provenance
    embeddings.manifest.json # encoder provider/model/quant/revision/dim/versions
    sha256s.json             # sha256 of each npy file
```

`prepare` skips re-embedding if these artefacts already exist (use `--force` to
re-embed).

## Results layout

```
results/beir/<dataset>/
    <method>.topk.jsonl   # one JSON line per query (full-corpus runs)
    <method>.summary.json # aggregate latency + provenance (full-corpus runs)
    timing.jsonl          # one record per (method, n_docs, threads) — drives the plots
results/beir/figures/     # scaling_curve / bars_single_thread / bars_threaded (.png/.svg)
```

Top-k JSONL row schema (emitted with `serde_json`, so IDs are always valid JSON):

```json
{"dataset":"scifact","split":"test","method":"ordvec-rq2",
 "qid_idx":0,"qid":"1","k":100,
 "doc_idxs":[42,7],"doc_ids":["abc","def"],"scores":[0.91,0.88]}
```

## `import ordvec` rule

This harness is an **external benchmark driver**. Python prepares embeddings,
evaluates qrels, and renders plots; the ordvec hot path is the Rust `beir-bench`
binary. The Python `ordvec` package is intentionally **not** imported — so the
latency numbers reflect the crate, not the bindings, and the harness does not
even require the wheel to be installed. The `bench-beir-guardrail` Make target
(run automatically by `benchmark-beir`) fails with a clear error if any
`benchmarks/beir/*.py` file contains `import ordvec` / `from ordvec`.

## Clean up

```bash
make bench-beir-clean         # remove result files + timing.jsonl, keep embedding cache
make bench-beir-clean-cache   # remove embedding cache (re-encoding required)
```
