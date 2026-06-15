"""
beir_prepare.py — Download and embed BEIR datasets for the ordvec-beir harness.

Responsibilities
----------------
1. Download (if absent) and load the BEIR dataset via GenericDataLoader.
2. Build stable orderings:
   - corpus_ids  = sorted(corpus.keys())
   - query_ids   = sorted(qrels.keys())
3. Construct document text:
   - ``title + "\\n" + text`` when title is non-empty, else ``text``.
4. Construct query text (raw query string, no prompt prepended to text — the
   prompt is baked in via ``prompt_name`` for ST or manually for Ollama).
5. Embed with the chosen provider (sentence-transformers or Ollama).
6. L2-normalise all rows; validate via ``validate_embeddings``.
7. Write the cache artefacts:
   - corpus.f32.npy, queries.f32.npy
   - corpus_ids.json, query_ids.json, qrels.json
   - texts.manifest.json, embeddings.manifest.json, sha256s.json

CLI
---
Run ``python beir_prepare.py --help`` for full usage.

Providers
---------
* **st** (canonical): sentence-transformers ``SentenceTransformer``.
  - Queries encoded with ``prompt_name="web_search_query"``.
  - Documents encoded with NO prompt.
  - Records sentence_transformers / transformers / torch versions + device.
* **ollama**: HTTP POST to ``<ollama-url>/api/embed``.
  - Queries prefixed with ``QUERY_PROMPT`` manually (documents unprefixed).
  - Rows normalised in Python after decoding.
  - Records ollama version, model digest, gguf_quant; ``canonical=false``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from typing import Any

import numpy as np
import requests

# Allow `from common import ...` when run as a script from the repo root
# (the Makefile invokes `python3 benchmarks/beir/<script>.py`).
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from common import (
    QUERY_PROMPT,
    dataset_cache_dir,
    encoder_slug,
    sha256_file,
    validate_embeddings,
)

# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _doc_text(entry: dict[str, str]) -> str:
    """Combine title and body according to the spec."""
    title = (entry.get("title") or "").strip()
    text = (entry.get("text") or "").strip()
    if title:
        return f"{title}\n{text}"
    return text


# ---------------------------------------------------------------------------
# Sentence-transformers encoder
# ---------------------------------------------------------------------------

def _embed_st(
    corpus_texts: list[str],
    query_texts: list[str],
    model_name: str,
    revision: str | None,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Embed with sentence-transformers; return (corpus_emb, query_emb, meta)."""
    import sentence_transformers
    import torch
    import transformers

    model = sentence_transformers.SentenceTransformer(
        model_name,
        revision=revision,
        device=device,
        trust_remote_code=False,
    )

    corpus_emb = model.encode(
        corpus_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    query_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        prompt_name="web_search_query",
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    corpus_emb = np.ascontiguousarray(corpus_emb, dtype=np.float32)
    query_emb = np.ascontiguousarray(query_emb, dtype=np.float32)

    meta: dict[str, Any] = {
        "sentence_transformers_version": sentence_transformers.__version__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": device,
    }
    return corpus_emb, query_emb, meta


# ---------------------------------------------------------------------------
# Ollama encoder
# ---------------------------------------------------------------------------

def _ollama_version(ollama_url: str) -> str:
    """Fetch Ollama server version string (best-effort)."""
    try:
        r = requests.get(f"{ollama_url.rstrip('/')}/api/version", timeout=10)
        r.raise_for_status()
        return r.json().get("version", "unknown")
    except Exception:
        return "unknown"


def _ollama_model_info(ollama_url: str, model: str) -> dict[str, str]:
    """Fetch model digest + gguf_quant (best-effort)."""
    try:
        r = requests.post(
            f"{ollama_url.rstrip('/')}/api/show",
            json={"name": model},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        digest = data.get("modelfile", {}) or {}
        # digest lives at top level in newer Ollama
        model_digest = data.get("digest", "unknown")
        details = data.get("details", {}) or {}
        gguf_quant = details.get("quantization_level", "unknown")
        return {"model_digest": model_digest, "gguf_quant": gguf_quant}
    except Exception:
        return {"model_digest": "unknown", "gguf_quant": "unknown"}


def _ollama_embed_batch(
    ollama_url: str,
    model: str,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Call Ollama /api/embed in batches; return stacked float32 array."""
    url = f"{ollama_url.rstrip('/')}/api/embed"
    all_vecs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = requests.post(
            url,
            json={"model": model, "input": batch},
            timeout=600,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ValueError(
                f"Ollama /api/embed returned no 'embeddings' key for batch "
                f"starting at index {i}. Response keys: {list(data.keys())}"
            )
        all_vecs.append(np.array(embeddings, dtype=np.float32))
    return np.vstack(all_vecs)


def _embed_ollama(
    corpus_texts: list[str],
    query_texts: list[str],
    model: str,
    ollama_url: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Embed with Ollama; return (corpus_emb, query_emb, meta)."""
    # Prepend the query prompt manually for queries
    prefixed_queries = [QUERY_PROMPT + q for q in query_texts]

    corpus_emb = _ollama_embed_batch(ollama_url, model, corpus_texts, batch_size)
    query_emb = _ollama_embed_batch(ollama_url, model, prefixed_queries, batch_size)

    # Manual L2-normalise
    corpus_norms = np.linalg.norm(corpus_emb, axis=1, keepdims=True)
    query_norms = np.linalg.norm(query_emb, axis=1, keepdims=True)
    corpus_norms = np.where(corpus_norms == 0, 1.0, corpus_norms)
    query_norms = np.where(query_norms == 0, 1.0, query_norms)
    corpus_emb = np.ascontiguousarray(corpus_emb / corpus_norms, dtype=np.float32)
    query_emb = np.ascontiguousarray(query_emb / query_norms, dtype=np.float32)

    version = _ollama_version(ollama_url)
    model_info = _ollama_model_info(ollama_url, model)

    meta: dict[str, Any] = {
        "ollama_version": version,
        "model_digest": model_info["model_digest"],
        "gguf_quant": model_info["gguf_quant"],
        "canonical": False,
    }
    return corpus_emb, query_emb, meta


# ---------------------------------------------------------------------------
# Cache-write helpers
# ---------------------------------------------------------------------------

def _write_json(path: pathlib.Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _write_npy(path: pathlib.Path, arr: np.ndarray) -> None:
    """Save a 2-D C-order float32 array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    np.save(str(path), arr)


# ---------------------------------------------------------------------------
# Vendored BEIR reader (no `beir` package dependency)
# ---------------------------------------------------------------------------

def _download_beir(dataset: str, cache_dir: pathlib.Path) -> pathlib.Path:
    """Download + unzip a BEIR dataset to ``<cache>/raw/<dataset>/`` (cached).

    Uses the public BEIR zip and stdlib ``zipfile`` so the harness does not
    depend on the ``beir`` package (which transitively pulls the unbuildable
    ``pytrec_eval``) — keeping ``pip install -r requirements.txt`` clean.
    """
    import zipfile

    from tqdm import tqdm

    data_path = cache_dir / "raw" / dataset
    if (data_path / "corpus.jsonl").exists():
        print(f"[prepare] Using cached raw data at {data_path}", flush=True)
        return data_path

    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = (
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
        f"{dataset}.zip"
    )
    print(f"[prepare] Downloading BEIR dataset: {url}", flush=True)
    zip_path = raw_dir / f"{dataset}.zip"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, desc=f"{dataset}.zip"
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    zip_path.unlink(missing_ok=True)
    if not (data_path / "corpus.jsonl").exists():
        raise FileNotFoundError(
            f"BEIR archive for {dataset!r} did not unzip to {data_path}"
        )
    return data_path


def _load_beir(
    data_path: pathlib.Path, split: str
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]]]:
    """Parse a BEIR dataset folder, mirroring ``beir.GenericDataLoader``:

    ``corpus = {cid: {"title", "text"}}``, ``queries = {qid: text}``,
    ``qrels = {qid: {cid: relevance}}``.
    """
    corpus: dict[str, dict[str, str]] = {}
    with open(data_path / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[str(d["_id"])] = {
                "title": d.get("title", "") or "",
                "text": d.get("text", "") or "",
            }
    queries: dict[str, str] = {}
    with open(data_path / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[str(d["_id"])] = d["text"]
    qrels: dict[str, dict[str, int]] = {}
    with open(data_path / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        header = f.readline()
        if "query-id" not in header:  # no header row → first line is data
            f.seek(0)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, cid, score = parts[0], parts[1], parts[2]
            qrels.setdefault(qid, {})[cid] = int(score)
    return corpus, queries, qrels


# ---------------------------------------------------------------------------
# llama.cpp GGUF embedder (exact Q8_0 weights, same llama.cpp as OrdinalDB)
# ---------------------------------------------------------------------------

def _embed_llamacpp(
    corpus_texts: list[str],
    query_texts: list[str],
    gguf_repo: str,
    gguf_file: str,
    n_gpu_layers: int,
    n_ctx: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Embed with the Q8_0 GGUF via llama-cpp-python (same llama.cpp + weights a
    native-Rust llama.cpp encoder uses), last-token pooled and L2-normalised.
    Returns ``(corpus_emb, query_emb, meta)``.
    """
    import llama_cpp
    from llama_cpp import Llama

    llm = Llama.from_pretrained(
        repo_id=gguf_repo,
        filename=gguf_file,
        embedding=True,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        n_batch=n_ctx,  # embeddings need the whole sequence in one batch
        n_ubatch=n_ctx,
        pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
        verbose=False,
    )

    def _embed_all(texts: list[str]) -> np.ndarray:
        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for e in llm.embed(chunk):
                arr = np.asarray(e, dtype=np.float32)
                if arr.ndim == 2:  # per-token (no pooling) → take last token
                    arr = arr[-1]
                vecs.append(arr)
        return np.vstack(vecs).astype(np.float32)

    prefixed_queries = [QUERY_PROMPT + q for q in query_texts]
    corpus_emb = _embed_all(corpus_texts)
    query_emb = _embed_all(prefixed_queries)

    def _l2(a: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n = np.where(n == 0, 1.0, n)
        return np.ascontiguousarray(a / n, dtype=np.float32)

    corpus_emb = _l2(corpus_emb)
    query_emb = _l2(query_emb)

    meta: dict[str, Any] = {
        "gguf_repo": gguf_repo,
        "gguf_file": gguf_file,
        "gguf_quant": "Q8_0",
        "llama_cpp_python_version": getattr(llama_cpp, "__version__", "unknown"),
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "canonical": True,
    }
    return corpus_emb, query_emb, meta


# ---------------------------------------------------------------------------
# Main prepare routine
# ---------------------------------------------------------------------------

def prepare_dataset(
    dataset: str,
    split: str,
    provider: str,
    model: str,
    revision: str | None,
    device: str,
    batch_size: int,
    ollama_url: str,
    cache_dir: pathlib.Path,
    gguf_file: str = "*Q8_0.gguf",
    n_gpu_layers: int = -1,
    n_ctx: int = 2048,
    force: bool = False,
) -> None:
    """Run the full prepare pipeline for one dataset."""
    # 0. Skip if this exact encoder's artefacts are already cached. Re-embedding
    #    a large corpus (e.g. trec-covid's 171K docs) is expensive, and several
    #    benchmark targets touch the same dataset; `--force` re-embeds.
    slug_revision = gguf_file if provider == "llamacpp" else revision
    slug = encoder_slug(provider, model, slug_revision)
    enc_dir = dataset_cache_dir(cache_dir, dataset, split, slug)
    required = [
        "corpus.f32.npy",
        "queries.f32.npy",
        "qrels.json",
        "corpus_ids.json",
        "query_ids.json",
        "embeddings.manifest.json",
        "sha256s.json",
    ]
    if not force and all((enc_dir / f).exists() for f in required):
        print(
            f"[prepare] {dataset}/{split}: cached encoder at {enc_dir} "
            "(use --force to re-embed); skipping",
            flush=True,
        )
        return

    # 1. Download + 2. Load via the vendored BEIR reader (no `beir` package
    #    dependency, so `pip install -r requirements.txt` stays clean on a fresh
    #    machine — `beir` would otherwise pull the unbuildable `pytrec_eval`).
    data_path = _download_beir(dataset, cache_dir)
    corpus, queries, qrels = _load_beir(data_path, split)

    # ------------------------------------------------------------------ #
    # 3. Stable ordering                                                   #
    # ------------------------------------------------------------------ #
    corpus_ids: list[str] = sorted(corpus.keys())
    query_ids: list[str] = sorted(qrels.keys())

    # ------------------------------------------------------------------ #
    # 4. Build text lists                                                  #
    # ------------------------------------------------------------------ #
    corpus_texts = [_doc_text(corpus[cid]) for cid in corpus_ids]
    query_texts = [queries[qid] for qid in query_ids]

    n_docs = len(corpus_ids)
    n_queries = len(query_ids)
    print(
        f"[prepare] {dataset}/{split}: {n_docs} docs, {n_queries} queries",
        flush=True,
    )

    # ------------------------------------------------------------------ #
    # 5. Embed                                                             #
    # ------------------------------------------------------------------ #
    # (slug / enc_dir computed above for the cache-skip check.)
    t0 = time.time()
    if provider == "st":
        corpus_emb, query_emb, enc_meta = _embed_st(
            corpus_texts,
            query_texts,
            model_name=model,
            revision=revision,
            device=device,
            batch_size=batch_size,
        )
    elif provider == "ollama":
        corpus_emb, query_emb, enc_meta = _embed_ollama(
            corpus_texts,
            query_texts,
            model=model,
            ollama_url=ollama_url,
            batch_size=batch_size,
        )
    elif provider == "llamacpp":
        corpus_emb, query_emb, enc_meta = _embed_llamacpp(
            corpus_texts,
            query_texts,
            gguf_repo=model,
            gguf_file=gguf_file,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}")
    embed_seconds = time.time() - t0

    # ------------------------------------------------------------------ #
    # 6. Validate (fail-closed)                                            #
    # ------------------------------------------------------------------ #
    validate_embeddings(corpus_emb)
    validate_embeddings(query_emb)

    dim = corpus_emb.shape[1]

    # ------------------------------------------------------------------ #
    # 7. Write artefacts                                                   #
    # ------------------------------------------------------------------ #
    corpus_npy = enc_dir / "corpus.f32.npy"
    query_npy = enc_dir / "queries.f32.npy"
    _write_npy(corpus_npy, corpus_emb)
    _write_npy(query_npy, query_emb)

    _write_json(enc_dir / "corpus_ids.json", corpus_ids)
    _write_json(enc_dir / "query_ids.json", query_ids)

    # qrels keyed by str qid → {str doc_id: int relevance}
    qrels_serialisable = {
        qid: {did: int(rel) for did, rel in doc_rels.items()}
        for qid, doc_rels in qrels.items()
    }
    _write_json(enc_dir / "qrels.json", qrels_serialisable)

    # texts.manifest.json
    texts_manifest = {
        "dataset": dataset,
        "split": split,
        "n_corpus": n_docs,
        "n_queries": n_queries,
        "corpus_id_order": "sorted(corpus.keys())",
        "query_id_order": "sorted(qrels.keys())",
        "doc_text_format": "title + '\\n' + text if title else text",
        "query_text_format": "raw query string (no prefix in text list)",
    }
    _write_json(enc_dir / "texts.manifest.json", texts_manifest)

    # sha256s (compute after writing npys)
    corpus_sha256 = sha256_file(corpus_npy)
    query_sha256 = sha256_file(query_npy)

    # embeddings.manifest.json
    if provider == "st":
        embeddings_manifest: dict[str, Any] = {
            "encoder_provider": "sentence-transformers",
            "encoder_model": model,
            "encoder_revision": revision,
            "encoder_slug": slug,
            "embedding_dim": dim,
            "dtype": "float32",
            "normalize_embeddings": True,
            "query_prompt_name": "web_search_query",
            "query_prompt_text": QUERY_PROMPT,
            "document_prompt_text": None,
            "n_corpus": n_docs,
            "n_queries": n_queries,
            "corpus_sha256": corpus_sha256,
            "query_sha256": query_sha256,
            "embed_seconds": embed_seconds,
            "device": enc_meta["device"],
            "sentence_transformers_version": enc_meta["sentence_transformers_version"],
            "transformers_version": enc_meta["transformers_version"],
            "torch_version": enc_meta["torch_version"],
        }
    elif provider == "llamacpp":
        embeddings_manifest = {
            "encoder_provider": "llama-cpp-python",
            "encoder_model": model,
            "encoder_revision": enc_meta["gguf_file"],
            "encoder_slug": slug,
            "embedding_dim": dim,
            "dtype": "float32",
            "normalize_embeddings": True,
            "query_prompt_name": None,
            "query_prompt_text": QUERY_PROMPT,
            "document_prompt_text": None,
            "n_corpus": n_docs,
            "n_queries": n_queries,
            "corpus_sha256": corpus_sha256,
            "query_sha256": query_sha256,
            "embed_seconds": embed_seconds,
            "gguf_repo": enc_meta["gguf_repo"],
            "gguf_file": enc_meta["gguf_file"],
            "gguf_quant": enc_meta["gguf_quant"],
            "llama_cpp_python_version": enc_meta["llama_cpp_python_version"],
            "n_gpu_layers": enc_meta["n_gpu_layers"],
            "n_ctx": enc_meta["n_ctx"],
            "pooling": "last_token",
            "canonical": enc_meta["canonical"],
        }
    else:  # ollama
        embeddings_manifest = {
            "encoder_provider": "ollama",
            "encoder_model": model,
            "encoder_revision": revision,
            "encoder_slug": slug,
            "embedding_dim": dim,
            "dtype": "float32",
            "normalize_embeddings": True,
            "query_prompt_name": None,
            "query_prompt_text": QUERY_PROMPT,
            "document_prompt_text": None,
            "n_corpus": n_docs,
            "n_queries": n_queries,
            "corpus_sha256": corpus_sha256,
            "query_sha256": query_sha256,
            "embed_seconds": embed_seconds,
            "ollama_version": enc_meta["ollama_version"],
            "model_digest": enc_meta["model_digest"],
            "gguf_quant": enc_meta["gguf_quant"],
            "canonical": False,
        }

    _write_json(enc_dir / "embeddings.manifest.json", embeddings_manifest)

    sha256s = {
        "corpus.f32.npy": corpus_sha256,
        "queries.f32.npy": query_sha256,
    }
    _write_json(enc_dir / "sha256s.json", sha256s)

    print(f"[prepare] Done. Wrote artefacts to {enc_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beir_prepare",
        description="Download and embed BEIR datasets for the ordvec-beir harness.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DATASET",
        help="One or more BEIR dataset names (e.g. msmarco nfcorpus).",
    )
    p.add_argument(
        "--split",
        default="test",
        help="Dataset split to use (default: test).",
    )
    p.add_argument(
        "--provider",
        choices=["st", "ollama", "llamacpp"],
        default="llamacpp",
        help=(
            "Encoder provider: 'llamacpp' (GGUF via llama-cpp-python, CUDA — the "
            "canonical lane), 'st' (sentence-transformers), or 'ollama'."
        ),
    )
    p.add_argument(
        "--model",
        default="mradermacher/harrier-oss-v1-0.6b-GGUF",
        help=(
            "Encoder identity: HuggingFace repo path. For 'llamacpp' this is the "
            "GGUF repo (paired with --gguf-file); for 'st' the model path; for "
            "'ollama' the model tag."
        ),
    )
    p.add_argument(
        "--gguf-file",
        default="*Q8_0.gguf",
        dest="gguf_file",
        help=(
            "GGUF filename (glob ok) within --model's repo for the 'llamacpp' "
            "lane (default: *Q8_0.gguf)."
        ),
    )
    p.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        dest="n_gpu_layers",
        help=(
            "Layers to offload to GPU for the 'llamacpp' lane; -1 = all "
            "(default: -1)."
        ),
    )
    p.add_argument(
        "--n-ctx",
        type=int,
        default=2048,
        dest="n_ctx",
        help="Context window for the 'llamacpp' lane (default: 2048).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Model revision / git commit SHA (for 'st'; ignored for 'ollama').",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Torch device for sentence-transformers (e.g. 'cuda', 'mps', 'cpu').",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        dest="batch_size",
        help="Encoding batch size (default: 64).",
    )
    p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        dest="ollama_url",
        help="Base URL for the Ollama server (default: http://localhost:11434).",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        dest="cache_dir",
        help=(
            "Root cache directory.  Defaults to "
            "<repo-root>/.cache/ordvec-beir."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-embed even if cached encoder artefacts already exist.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _set_seeds(args.seed)

    if args.cache_dir is None:
        # Resolve relative to this file's repo root: benchmarks/beir/../../
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        cache_dir = repo_root / ".cache" / "ordvec-beir"
    else:
        cache_dir = pathlib.Path(args.cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for dataset in args.datasets:
        print(f"\n{'='*60}", flush=True)
        print(f"[prepare] Processing: {dataset}", flush=True)
        try:
            prepare_dataset(
                dataset=dataset,
                split=args.split,
                provider=args.provider,
                model=args.model,
                revision=args.revision,
                device=args.device,
                batch_size=args.batch_size,
                ollama_url=args.ollama_url,
                cache_dir=cache_dir,
                gguf_file=args.gguf_file,
                n_gpu_layers=args.n_gpu_layers,
                n_ctx=args.n_ctx,
                force=args.force,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[prepare] ERROR for dataset {dataset!r}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            failed.append(dataset)

    if failed:
        print(
            f"\n[prepare] FAILED datasets: {failed}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
