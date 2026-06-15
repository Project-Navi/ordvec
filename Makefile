# ordvec-beir benchmark harness
#
# Reproduces, on a fresh CUDA machine, the ordvec retrieval story on standard
# BEIR datasets:
#   * quality  — nDCG@10 vs the official BEIR qrels, ordvec vs an exact
#                full-float baseline (`flat`, == FAISS IndexFlatIP math).
#   * scaling  — speedup-vs-corpus-size: brute force is O(n), ordvec sign/rank
#                candidate-gen is near-flat in n, so the gap widens with scale.
#   * graphics — three README figures (scaling curve + single-thread & threaded
#                latency bars).
#
# ALL latency is measured in ONE Rust process (`beir-bench`): ordvec vs an exact
# inner-product baseline vs a pure-Rust HNSW — same machine, batch, and thread
# count, no Python/FFI boundary. Python only embeds (GGUF Q8 via llama-cpp-python),
# scores nDCG, and renders the figures.
#
# Usage:
#   make bench-beir-setup        # install Python deps + CUDA llama-cpp-python
#   make benchmark-beir-smoke    # quick end-to-end sanity (scifact only)
#   make benchmark-beir          # full: quality + scaling + graphics

# ── interpreter ──────────────────────────────────────────────────────────────
PY ?= python3

# ── paths ─────────────────────────────────────────────────────────────────────
CACHE_DIR   := .cache/ordvec-beir
RESULTS_DIR := results/beir
FIG_DIR     := $(RESULTS_DIR)/figures

# ── datasets ──────────────────────────────────────────────────────────────────
# Quality (nDCG) datasets. PERF_DATASET drives the scaling curve + latency bars
# and must be large enough for the curve to bend (trec-covid ≈ 171K docs).
QUALITY_DATASETS := scifact
PERF_DATASET     := trec-covid
SPLIT            := test

# Smoke overrides (scifact is small + already cheap to embed).
SMOKE_QUALITY      := scifact
SMOKE_PERF_DATASET := scifact
SMOKE_SCALE_SIZES  := 500 1000 2000

# ── retrieval parameters ─────────────────────────────────────────────────────
TOPK       := 100
K_VALUES   := 10 100
BATCH      := 32
CANDIDATES := 500
SEED       := 1
NPROC      := $(shell nproc 2>/dev/null || echo 8)
# Batch regimes for the graphics: the scaling curve + single-thread bar use
# single-query (batch=1) — the latency-sensitive deployment where flat is
# memory-bound and ordvec wins ~100×; the threaded bar uses a batched throughput
# regime where flat amortizes its corpus stream across the batch.
SCALE_BATCH := 1
MULTI_BATCH := 32

# Corpus-size ladder for the scaling sweep (clamped to the real corpus size by
# the bench). Full-corpus points are added by the dedicated full runs.
SCALE_SIZES := 1000 3000 10000 30000 100000 170000

# ── methods (all measured in the single Rust process) ─────────────────────────
#   flat       exact inner product (== FAISS IndexFlatIP math), 4096 B/vec
#   hnsw       pure-Rust HNSW M=32 (Malkov–Yashunin), 4096 B/vec
#   rq2/rq4    ordvec RankQuant b=2 / b=4 (256 / 512 B/vec)
#   bitmap-rq2 ordvec Bitmap → RankQuant b=2 (two-stage)
#   sign-rq2   ordvec SignBitmap → RankQuant b=2 (two-stage)
BENCH_METHODS := flat,hnsw,rq2,rq4,bitmap-rq2,sign-rq2

# ── encoder (canonical: GGUF Q8_0 via llama-cpp-python / CUDA) ────────────────
HARRIER_GGUF_REPO := mradermacher/harrier-oss-v1-0.6b-GGUF
GGUF_FILE         := *Q8_0.gguf
N_GPU_LAYERS      := -1
N_CTX             := 2048
ENCODE_BATCH      := 16
# CUDA build flags for llama-cpp-python (override LLAMA_CMAKE_ARGS= for CPU).
LLAMA_CMAKE_ARGS  := -DGGML_CUDA=on

# ── phony ─────────────────────────────────────────────────────────────────────
.PHONY: benchmark-beir benchmark-beir-smoke bench-beir-setup bench-beir-build \
        bench-beir-guardrail bench-beir-quality bench-beir-scaling \
        bench-beir-plot bench-beir-clean bench-beir-clean-cache

# The pipeline is strictly sequential (prepare writes the cache the bench reads;
# eval/plot read run files). Steps are unordered prerequisites, so under a
# parallel make (-j, or an inherited MAKEFLAGS=-jN) they would race on a
# half-written cache. Force serial execution regardless.
.NOTPARALLEL:

# ── top-level targets ─────────────────────────────────────────────────────────

## Full run: quality (nDCG) + scaling sweep + three README graphics.
benchmark-beir: bench-beir-guardrail bench-beir-quality bench-beir-scaling bench-beir-plot

## Quick end-to-end sanity: everything on scifact, tiny scaling ladder.
benchmark-beir-smoke:
	$(MAKE) bench-beir-guardrail
	$(MAKE) bench-beir-quality QUALITY_DATASETS="$(SMOKE_QUALITY)"
	$(MAKE) bench-beir-scaling PERF_DATASET=$(SMOKE_PERF_DATASET) SCALE_SIZES="$(SMOKE_SCALE_SIZES)"
	$(MAKE) bench-beir-plot PERF_DATASET=$(SMOKE_PERF_DATASET)

# ── setup ─────────────────────────────────────────────────────────────────────

## Install Python deps (core wheels) + CUDA llama-cpp-python. The latter is built
## against the host CUDA toolkit; --no-cache-dir + --force-reinstall defeat pip's
## wheel cache (it ignores CMAKE_ARGS and would hand back a stale CPU build).
## CPU-only box: make bench-beir-setup LLAMA_CMAKE_ARGS=
bench-beir-setup:
	$(PY) -m pip install -r benchmarks/beir/requirements.txt
	CMAKE_ARGS="$(LLAMA_CMAKE_ARGS)" $(PY) -m pip install \
		--upgrade --force-reinstall --no-cache-dir llama-cpp-python

## Build the all-Rust comparison harness (release).
bench-beir-build:
	cargo build --release -p beir-bench

# ── guardrail ─────────────────────────────────────────────────────────────────

## Fail loudly if any harness *.py imports the ordvec Python package directly —
## the benchmark hot path is the Rust crate, not the Python bindings.
bench-beir-guardrail:
	@if grep -rnE "^[[:space:]]*(import ordvec|from ordvec)\b" benchmarks/beir --include='*.py' 2>/dev/null; then \
		echo "ERROR: a benchmarks/beir/*.py file imports the ordvec Python package."; \
		exit 1; \
	fi
	@echo "guardrail OK: no 'import ordvec' in benchmarks/beir/*.py"

# ── quality: nDCG@10 vs qrels (ordvec vs exact flat) ──────────────────────────

## Embed → run all methods (single-thread, full corpus) → score nDCG, per dataset.
bench-beir-quality: bench-beir-build
	@for d in $(QUALITY_DATASETS); do \
		echo "=== quality: $$d ==="; \
		$(PY) benchmarks/beir/beir_prepare.py --datasets $$d --split $(SPLIT) \
			--provider llamacpp --model "$(HARRIER_GGUF_REPO)" --gguf-file "$(GGUF_FILE)" \
			--n-gpu-layers $(N_GPU_LAYERS) --n-ctx $(N_CTX) --batch-size $(ENCODE_BATCH) \
			--cache-dir "$(CACHE_DIR)" --seed $(SEED) || exit 1; \
		$(CURDIR)/target/release/beir-bench --cache-dir "$(CACHE_DIR)" --dataset $$d \
			--split $(SPLIT) --top-k $(TOPK) --batch $(BATCH) --candidates $(CANDIDATES) \
			--threads 1 --methods $(BENCH_METHODS) --out-dir "$(RESULTS_DIR)" || exit 1; \
		$(PY) benchmarks/beir/beir_eval.py --datasets $$d --split $(SPLIT) \
			--cache-dir "$(CACHE_DIR)" --runs-dir "$(RESULTS_DIR)" --k-values $(K_VALUES) \
			--baseline flat --bootstrap-iters 1000 --seed $(SEED) --out-dir "$(RESULTS_DIR)" || exit 1; \
	done

# ── scaling: speedup-vs-corpus-size + single/threaded full-corpus points ───────

## Sweep the perf dataset over a corpus-size ladder (single-thread), then full
## corpus at 1 thread and at $(NPROC) threads. All append to timing.jsonl.
bench-beir-scaling: bench-beir-build
	@echo "=== scaling: $(PERF_DATASET) (sizes: $(SCALE_SIZES); threaded full = $(NPROC)t) ==="
	$(PY) benchmarks/beir/beir_prepare.py --datasets $(PERF_DATASET) --split $(SPLIT) \
		--provider llamacpp --model "$(HARRIER_GGUF_REPO)" --gguf-file "$(GGUF_FILE)" \
		--n-gpu-layers $(N_GPU_LAYERS) --n-ctx $(N_CTX) --batch-size $(ENCODE_BATCH) \
		--cache-dir "$(CACHE_DIR)" --seed $(SEED)
	rm -f "$(RESULTS_DIR)/$(PERF_DATASET)/timing.jsonl"
	@for n in $(SCALE_SIZES); do \
		echo "  -- n=$$n (1 thread, single-query batch=$(SCALE_BATCH)) --"; \
		$(CURDIR)/target/release/beir-bench --cache-dir "$(CACHE_DIR)" --dataset $(PERF_DATASET) \
			--split $(SPLIT) --top-k $(TOPK) --batch $(SCALE_BATCH) --candidates $(CANDIDATES) \
			--threads 1 --max-docs $$n --methods $(BENCH_METHODS) --out-dir "$(RESULTS_DIR)" || exit 1; \
	done
	@echo "  -- full corpus (1 thread, single-query batch=$(SCALE_BATCH); writes topk + nDCG inputs) --"
	$(CURDIR)/target/release/beir-bench --cache-dir "$(CACHE_DIR)" --dataset $(PERF_DATASET) \
		--split $(SPLIT) --top-k $(TOPK) --batch $(SCALE_BATCH) --candidates $(CANDIDATES) \
		--threads 1 --methods $(BENCH_METHODS) --out-dir "$(RESULTS_DIR)"
	@echo "  -- full corpus ($(NPROC) threads, batched batch=$(MULTI_BATCH)) --"
	$(CURDIR)/target/release/beir-bench --cache-dir "$(CACHE_DIR)" --dataset $(PERF_DATASET) \
		--split $(SPLIT) --top-k $(TOPK) --batch $(MULTI_BATCH) --candidates $(CANDIDATES) \
		--threads $(NPROC) --methods $(BENCH_METHODS) --out-dir "$(RESULTS_DIR)"
	$(PY) benchmarks/beir/beir_eval.py --datasets $(PERF_DATASET) --split $(SPLIT) \
		--cache-dir "$(CACHE_DIR)" --runs-dir "$(RESULTS_DIR)" --k-values $(K_VALUES) \
		--baseline flat --bootstrap-iters 1000 --seed $(SEED) --out-dir "$(RESULTS_DIR)"

# ── graphics ──────────────────────────────────────────────────────────────────

## Render the three README figures from the timing records.
bench-beir-plot:
	$(PY) benchmarks/beir/beir_plot.py --runs-dir "$(RESULTS_DIR)" \
		--scaling-dataset $(PERF_DATASET) --bar-dataset $(PERF_DATASET) \
		--scaling-threads 1 --scaling-batch $(SCALE_BATCH) \
		--bar-single-threads 1 --bar-single-batch $(SCALE_BATCH) \
		--bar-multi-threads $(NPROC) --bar-multi-batch $(MULTI_BATCH) \
		--out-dir "$(FIG_DIR)"

# ── cleanup ───────────────────────────────────────────────────────────────────

## Remove generated result files (keeps the embedding cache).
bench-beir-clean:
	find $(RESULTS_DIR) -name "*.topk.jsonl" -delete
	find $(RESULTS_DIR) -name "*.summary.json" -delete
	find $(RESULTS_DIR) -name "timing.jsonl" -delete

## Remove the embedding cache (re-encoding will be required).
bench-beir-clean-cache:
	rm -rf $(CACHE_DIR)
