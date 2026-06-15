"""
beir_eval.py — Evaluate ordvec-beir top-k runs against BEIR qrels.

Responsibilities (spec §9)
--------------------------
1. Discover every ``<runs-dir>/<dataset>/*.topk.jsonl`` file.
2. Build the run dict ``{qid: {doc_id: score}}`` for each method.
3. Evaluate against the cached BEIR qrels using ``pytrec_eval`` (the same
   engine BEIR's ``EvaluateRetrieval`` wraps).  Headline metric is nDCG@10;
   secondary metrics are MAP@10, Recall@100, MRR@10, Precision@10.
4. Pull systems columns (bytes/vector, total MiB, build seconds,
   p50/p95/p99 latency, queries/second) from each method's ``.summary.json``.
5. Run a *paired* bootstrap of every method vs the ``--baseline`` (faiss-flat):
   resample queries with replacement ``--bootstrap-iters`` times (seeded),
   compute the per-query metric delta (method - baseline), and report the
   mean delta + 95% CI + ``within_noise``.
6. (Diagnostic, behind ``--include-ann-diagnostics``) ANN recall@100 vs the
   baseline (overlap of top-100 doc sets).  Kept OUT of the headline summary.
7. Emit ``summary.csv``, ``summary.json``, ``comparison-matrix.md``,
   ``bootstrap.json`` and (via :mod:`beir_report`) ``summary.md``.

This harness is an *external consumer* of ordvec — it MUST NOT ``import
ordvec``.  It only reads cached artefacts and result files.

CLI
---
Run ``python beir_eval.py --help`` for full usage.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

# Allow `from common import ...` when run as a script from the repo root
# (the Makefile invokes `python3 benchmarks/beir/<script>.py`).
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from common import (
    find_encoder_dir,
    load_manifest,
    load_qrels,
    read_topk_jsonl,
)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

#: Headline metric reported as the lead column everywhere.
HEADLINE_METRIC = "ndcg@10"

#: Metric families pytrec_eval can compute per-query at a cut value ``k``.
#: Maps our metric prefix → (pytrec measure family, pytrec key template).
_PYTREC_FAMILIES: dict[str, tuple[str, str]] = {
    "ndcg": ("ndcg_cut", "ndcg_cut_{k}"),
    "map": ("map_cut", "map_cut_{k}"),
    "recall": ("recall", "recall_{k}"),
    "precision": ("P", "P_{k}"),
}


def _metric_label(prefix: str, k: int) -> str:
    """Public metric label, e.g. ``ndcg@10`` / ``recall@100``."""
    return f"{prefix}@{k}"


# ---------------------------------------------------------------------------
# Run / qrels loading
# ---------------------------------------------------------------------------

def discover_runs(
    runs_dir: pathlib.Path, dataset: str
) -> dict[str, pathlib.Path]:
    """Return ``{method_slug: topk_jsonl_path}`` for one dataset.

    The method slug is the JSONL filename with ``.topk.jsonl`` stripped.
    """
    ds_dir = runs_dir / dataset
    if not ds_dir.is_dir():
        raise FileNotFoundError(
            f"No run directory for dataset {dataset!r}: {ds_dir} does not exist."
        )
    out: dict[str, pathlib.Path] = {}
    for path in sorted(ds_dir.glob("*.topk.jsonl")):
        slug = path.name[: -len(".topk.jsonl")]
        out[slug] = path
    if not out:
        raise FileNotFoundError(
            f"No *.topk.jsonl run files found under {ds_dir}."
        )
    return out


def build_run_dict(topk_path: pathlib.Path) -> dict[str, dict[str, float]]:
    """Load a top-k JSONL file into ``{qid: {doc_id: score}}``.

    Later duplicate ``(qid, doc_id)`` pairs overwrite earlier ones, matching
    pytrec_eval's own last-wins semantics for run files.
    """
    run: dict[str, dict[str, float]] = {}
    for row in read_topk_jsonl(topk_path):
        qid = str(row["qid"])
        doc_ids = row["doc_ids"]
        scores = row["scores"]
        if len(doc_ids) != len(scores):
            raise ValueError(
                f"{topk_path}: qid={qid} has {len(doc_ids)} doc_ids but "
                f"{len(scores)} scores."
            )
        per_q = run.setdefault(qid, {})
        for did, score in zip(doc_ids, scores):
            per_q[str(did)] = float(score)
    return run


def load_summary(
    runs_dir: pathlib.Path, dataset: str, method_slug: str
) -> dict[str, Any] | None:
    """Load ``<method_slug>.summary.json`` if present, else ``None``."""
    path = runs_dir / dataset / f"{method_slug}.summary.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Per-query metrics (pytrec_eval + manual MRR)
# ---------------------------------------------------------------------------

def _require_pytrec_eval():
    try:
        import pytrec_eval  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise SystemExit(
            "pytrec_eval is required for BEIR evaluation but is not installed. "
            "Install it with `pip install pytrec_eval` (it is the same engine "
            "BEIR's EvaluateRetrieval wraps)."
        ) from exc
    return pytrec_eval


def per_query_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    k_values: list[int],
) -> dict[str, dict[str, float]]:
    """Compute every supported metric per query.

    Returns ``{metric_label: {qid: value}}`` covering, for each ``k`` in
    *k_values*: ``ndcg@k``, ``map@k``, ``recall@k``, ``precision@k`` (from
    pytrec_eval) and ``mrr@k`` (computed manually, matching BEIR semantics).

    Only qids present in *qrels* are scored — a method that omits a judged
    query is treated as scoring 0 for that query (pytrec_eval reports nothing,
    so we backfill zeros to keep the bootstrap paired).
    """
    pytrec_eval = _require_pytrec_eval()

    # Build the pytrec_eval measure set: one family entry per requested k.
    measures: set[str] = set()
    for k in k_values:
        for _prefix, (family, _tmpl) in _PYTREC_FAMILIES.items():
            measures.add(f"{family}.{k}")

    # pytrec_eval requires int relevances and string ids.
    clean_qrels = {
        qid: {str(did): int(rel) for did, rel in rels.items()}
        for qid, rels in qrels.items()
    }
    clean_run = {
        qid: {str(did): float(s) for did, s in docs.items()}
        for qid, docs in run.items()
    }

    evaluator = pytrec_eval.RelevanceEvaluator(clean_qrels, measures)
    raw = evaluator.evaluate(clean_run)  # {qid: {pytrec_key: value}}

    judged_qids = list(clean_qrels.keys())
    out: dict[str, dict[str, float]] = {}

    for k in k_values:
        for prefix, (_family, tmpl) in _PYTREC_FAMILIES.items():
            label = _metric_label(prefix, k)
            key = tmpl.format(k=k)
            out[label] = {
                qid: float(raw.get(qid, {}).get(key, 0.0))
                for qid in judged_qids
            }
        # MRR@k computed manually (pytrec_eval has no direct cut MRR).
        mrr_label = _metric_label("mrr", k)
        out[mrr_label] = {
            qid: _mrr_at_k(clean_qrels[qid], clean_run.get(qid, {}), k)
            for qid in judged_qids
        }
    return out


def _mrr_at_k(
    rels: dict[str, int], scored: dict[str, float], k: int
) -> float:
    """Reciprocal rank of the first relevant doc within the top-*k* (BEIR)."""
    relevant = {did for did, rel in rels.items() if rel > 0}
    if not relevant:
        return 0.0
    # Rank by score descending; ties broken by doc_id for determinism.
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    for rank, (did, _score) in enumerate(ranked[:k], start=1):
        if did in relevant:
            return 1.0 / rank
    return 0.0


def aggregate(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean each metric across its queries."""
    return {
        label: (float(np.mean(list(vals.values()))) if vals else 0.0)
        for label, vals in per_query.items()
    }


# ---------------------------------------------------------------------------
# ANN recall diagnostic (optional)
# ---------------------------------------------------------------------------

def ann_recall_at_k(
    method_run: dict[str, dict[str, float]],
    baseline_run: dict[str, dict[str, float]],
    k: int,
) -> float:
    """Mean fraction of the baseline top-*k* doc set recovered by *method_run*.

    Diagnostic only — overlap of doc-id sets, NOT a qrel-based metric.
    """
    def _topk_ids(docs: dict[str, float]) -> set[str]:
        ranked = sorted(docs.items(), key=lambda kv: (-kv[1], kv[0]))
        return {did for did, _ in ranked[:k]}

    overlaps: list[float] = []
    for qid, base_docs in baseline_run.items():
        base_top = _topk_ids(base_docs)
        if not base_top:
            continue
        meth_top = _topk_ids(method_run.get(qid, {}))
        overlaps.append(len(base_top & meth_top) / len(base_top))
    return float(np.mean(overlaps)) if overlaps else 0.0


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def paired_bootstrap(
    method_pq: dict[str, float],
    baseline_pq: dict[str, float],
    n_iters: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Paired bootstrap of (method - baseline) over a shared query set.

    *method_pq* / *baseline_pq* are ``{qid: per_query_value}`` for ONE metric.
    Resamples the common qid set with replacement *n_iters* times; the SAME
    resampled indices index both methods (paired).  Returns the observed mean
    delta plus the 2.5/97.5 percentiles of the bootstrap delta distribution
    and ``within_noise`` (the 95% CI straddles 0).
    """
    common = sorted(set(method_pq) & set(baseline_pq))
    n = len(common)
    if n == 0:
        return {
            "delta": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "within_noise": True,
        }
    m = np.array([method_pq[q] for q in common], dtype=np.float64)
    b = np.array([baseline_pq[q] for q in common], dtype=np.float64)
    diff = m - b
    observed_delta = float(diff.mean())

    boot = np.empty(n_iters, dtype=np.float64)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()

    ci_low = float(np.percentile(boot, 2.5))
    ci_high = float(np.percentile(boot, 97.5))
    within_noise = bool(ci_low <= 0.0 <= ci_high)
    return {
        "delta": observed_delta,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "within_noise": within_noise,
    }


# ---------------------------------------------------------------------------
# Encoder provenance
# ---------------------------------------------------------------------------

def load_encoder_meta(
    cache_dir: pathlib.Path, dataset: str, split: str
) -> dict[str, Any]:
    """Read the encoder manifest for a dataset; tolerate a missing cache."""
    try:
        enc_dir = find_encoder_dir(cache_dir, dataset, split)
        manifest = load_manifest(enc_dir)
    except (FileNotFoundError, ValueError):
        return {
            "encoder_provider": "unknown",
            "encoder_model": "unknown",
            "encoder_revision": None,
            "encoder_slug": "unknown",
        }
    return {
        "encoder_provider": manifest.get("encoder_provider", "unknown"),
        "encoder_model": manifest.get("encoder_model", "unknown"),
        "encoder_revision": manifest.get("encoder_revision"),
        "encoder_slug": manifest.get("encoder_slug", "unknown"),
    }


# ---------------------------------------------------------------------------
# Systems columns
# ---------------------------------------------------------------------------

_SYSTEMS_KEYS = (
    "bytes_per_vector",
    "index_total_mib",
    "build_seconds",
    "query_latency_ms_p50",
    "query_latency_ms_p95",
    "query_latency_ms_p99",
    "queries_per_second",
)


def systems_columns(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the systems columns from a method summary (None → all None)."""
    if summary is None:
        return {key: None for key in _SYSTEMS_KEYS}
    return {key: summary.get(key) for key in _SYSTEMS_KEYS}


# ---------------------------------------------------------------------------
# Core evaluation driver
# ---------------------------------------------------------------------------

def evaluate_dataset(
    dataset: str,
    split: str,
    cache_dir: pathlib.Path,
    runs_dir: pathlib.Path,
    k_values: list[int],
    baseline: str,
    bootstrap_iters: int,
    seed: int,
    include_ann: bool,
) -> dict[str, Any]:
    """Evaluate every method for one dataset.

    Returns a dict with ``rows`` (one per method, headline + systems +
    metrics), ``bootstrap`` (list of bootstrap entries) and ``encoder`` meta.
    """
    qrels = load_qrels(find_encoder_dir(cache_dir, dataset, split))
    encoder = load_encoder_meta(cache_dir, dataset, split)
    runs = discover_runs(runs_dir, dataset)

    # Per-method per-query metrics + summaries.
    pq_by_method: dict[str, dict[str, dict[str, float]]] = {}
    run_by_method: dict[str, dict[str, dict[str, float]]] = {}
    rows: list[dict[str, Any]] = []

    metric_labels: list[str] = []
    for k in k_values:
        for prefix in ("ndcg", "map", "recall", "precision", "mrr"):
            metric_labels.append(_metric_label(prefix, k))

    for method_slug, topk_path in runs.items():
        run = build_run_dict(topk_path)
        run_by_method[method_slug] = run
        pq = per_query_metrics(qrels, run, k_values)
        pq_by_method[method_slug] = pq
        means = aggregate(pq)
        summary = load_summary(runs_dir, dataset, method_slug)
        row: dict[str, Any] = {
            "dataset": dataset,
            "split": split,
            "method": method_slug,
            "encoder_provider": encoder["encoder_provider"],
            "encoder_model": encoder["encoder_model"],
            "encoder_slug": encoder["encoder_slug"],
            "n_queries_judged": len(qrels),
            "headline": means.get(HEADLINE_METRIC, 0.0),
        }
        row.update({label: means[label] for label in metric_labels})
        row.update(systems_columns(summary))
        rows.append(row)

    # ANN recall diagnostic (optional, never in headline rows by default).
    if include_ann and baseline in run_by_method:
        base_run = run_by_method[baseline]
        for row in rows:
            method = row["method"]
            row["ann_recall@100"] = ann_recall_at_k(
                run_by_method[method], base_run, 100
            )

    # Paired bootstrap vs baseline.
    bootstrap_entries: list[dict[str, Any]] = []
    if baseline in pq_by_method:
        base_pq = pq_by_method[baseline]
        for method_slug, pq in pq_by_method.items():
            if method_slug == baseline:
                continue
            for label in metric_labels:
                rng = np.random.default_rng(
                    _bootstrap_seed(seed, dataset, method_slug, label)
                )
                stats = paired_bootstrap(
                    pq[label], base_pq[label], bootstrap_iters, rng
                )
                bootstrap_entries.append(
                    {
                        "dataset": dataset,
                        "method": method_slug,
                        "baseline": baseline,
                        "metric": label,
                        "delta": stats["delta"],
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                        "within_noise": stats["within_noise"],
                    }
                )
    else:
        print(
            f"[eval] WARNING: baseline {baseline!r} not found for dataset "
            f"{dataset!r}; bootstrap deltas skipped.",
            file=sys.stderr,
        )

    return {
        "dataset": dataset,
        "split": split,
        "encoder": encoder,
        "rows": rows,
        "bootstrap": bootstrap_entries,
    }


def _bootstrap_seed(
    seed: int, dataset: str, method: str, metric: str
) -> int:
    """Derive a stable per-(dataset,method,metric) seed from the base seed."""
    import hashlib

    h = hashlib.sha256(f"{seed}|{dataset}|{method}|{metric}".encode())
    # 63-bit positive int for numpy's SeedSequence.
    return int.from_bytes(h.digest()[:8], "big") & ((1 << 63) - 1)


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def _csv_columns(k_values: list[int], include_ann: bool) -> list[str]:
    cols = [
        "dataset",
        "split",
        "method",
        "encoder_provider",
        "encoder_model",
        "encoder_slug",
        "n_queries_judged",
        "headline",
    ]
    for k in k_values:
        for prefix in ("ndcg", "map", "recall", "precision", "mrr"):
            cols.append(_metric_label(prefix, k))
    cols.extend(_SYSTEMS_KEYS)
    if include_ann:
        cols.append("ann_recall@100")
    return cols


def write_csv(
    path: pathlib.Path,
    all_rows: list[dict[str, Any]],
    k_values: list[int],
    include_ann: bool,
) -> None:
    import csv

    cols = _csv_columns(k_values, include_ann)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({c: row.get(c, "") for c in cols})


# ---------------------------------------------------------------------------
# Output: comparison matrix
# ---------------------------------------------------------------------------

#: Static family / implementation / search-type metadata for the matrix.
#: Keyed by the method-name *stem* (params stripped).
_METHOD_FAMILY: dict[str, dict[str, str]] = {
    "flat": {
        "family": "dense (float)",
        "implementation": "Exact inner product (== FAISS FlatIP math)",
        "search_type": "exact brute-force (SIMD GEMM)",
        "headline_role": "baseline (comparison, not ground truth)",
    },
    "hnsw": {
        "family": "dense (float) ANN",
        "implementation": "HNSW M=32 (pure-Rust hnsw_rs)",
        "search_type": "graph ANN (approximate)",
        "headline_role": "candidate",
    },
    # Back-compat: the older Python-baselines lane used these slugs.
    "faiss-flat": {
        "family": "dense (float)",
        "implementation": "FAISS FlatIP",
        "search_type": "exact brute-force",
        "headline_role": "baseline (comparison, not ground truth)",
    },
    "hnswlib": {
        "family": "dense (float) ANN",
        "implementation": "hnswlib M=32",
        "search_type": "graph ANN (approximate)",
        "headline_role": "candidate",
    },
    "ordvec-rq2": {
        "family": "ordvec rank-quant",
        "implementation": "RankQuant b=2",
        "search_type": "exact asymmetric LUT",
        "headline_role": "candidate",
    },
    "ordvec-rq4": {
        "family": "ordvec rank-quant",
        "implementation": "RankQuant b=4",
        "search_type": "exact asymmetric LUT",
        "headline_role": "candidate",
    },
    "ordvec-bitmap-rq2": {
        "family": "ordvec two-stage",
        "implementation": "Bitmap → RankQuant b=2",
        "search_type": "candidate-gen + rerank",
        "headline_role": "candidate",
    },
    "ordvec-sign-rq2": {
        "family": "ordvec two-stage",
        "implementation": "SignBitmap → RankQuant b=2",
        "search_type": "candidate-gen + rerank",
        "headline_role": "candidate",
    },
}


def method_stem(method_slug: str) -> str:
    """Strip ``-m<N>`` / ``-b<N>`` parameter suffixes from a method slug."""
    parts = method_slug.split("-")
    kept = [
        p
        for p in parts
        if not (p[:1] == "m" and p[1:].isdigit())
        and not (p[:1] == "b" and p[1:].isdigit())
    ]
    return "-".join(kept)


def family_meta(method_slug: str) -> dict[str, str]:
    """Return the comparison-matrix metadata for a method (best-effort)."""
    meta = _METHOD_FAMILY.get(method_stem(method_slug))
    if meta is not None:
        return dict(meta)
    return {
        "family": "unknown",
        "implementation": method_slug,
        "search_type": "unknown",
        "headline_role": "candidate",
    }


# ---------------------------------------------------------------------------
# Output: summary.json assembly
# ---------------------------------------------------------------------------

def assemble_summary_json(
    datasets: list[str],
    split: str,
    baseline: str,
    k_values: list[int],
    bootstrap_iters: int,
    seed: int,
    include_ann: bool,
    per_dataset: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the master summary.json structure consumed by beir_report."""
    all_rows: list[dict[str, Any]] = []
    all_bootstrap: list[dict[str, Any]] = []
    encoders: dict[str, dict[str, Any]] = {}
    for ds in per_dataset:
        all_rows.extend(ds["rows"])
        all_bootstrap.extend(ds["bootstrap"])
        encoders[ds["dataset"]] = ds["encoder"]

    return {
        "config": {
            "datasets": datasets,
            "split": split,
            "baseline": baseline,
            "k_values": k_values,
            "bootstrap_iters": bootstrap_iters,
            "seed": seed,
            "include_ann_diagnostics": include_ann,
            "headline_metric": HEADLINE_METRIC,
        },
        "encoders": encoders,
        "rows": all_rows,
        "bootstrap": all_bootstrap,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beir_eval",
        description=(
            "Evaluate ordvec-beir top-k runs against BEIR qrels "
            "(nDCG@10 headline; paired bootstrap vs the baseline)."
        ),
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DATASET",
        help="One or more BEIR dataset names to evaluate.",
    )
    p.add_argument("--split", default="test", help="Split (default: test).")
    p.add_argument(
        "--cache-dir",
        default=None,
        dest="cache_dir",
        help="Embedding cache root (default: <repo-root>/.cache/ordvec-beir).",
    )
    p.add_argument(
        "--runs-dir",
        default=None,
        dest="runs_dir",
        help="Results root (default: <repo-root>/results/beir).",
    )
    p.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[10, 100],
        dest="k_values",
        help="BEIR k-values for nDCG/MAP/Recall/Precision/MRR (default: 10 100).",
    )
    p.add_argument(
        "--baseline",
        default="faiss-flat",
        help="Method slug used as the paired-bootstrap baseline.",
    )
    p.add_argument(
        "--bootstrap-iters",
        type=int,
        default=10000,
        dest="bootstrap_iters",
        help="Bootstrap resamples per (dataset,method,metric) (default: 10000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed for the bootstrap (default: 42).",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        dest="out_dir",
        help="Output directory for summary artefacts (default: --runs-dir).",
    )
    p.add_argument(
        "--include-ann-diagnostics",
        action="store_true",
        dest="include_ann",
        help="Also compute ANN recall@100 vs the baseline (diagnostic only).",
    )
    return p


def _default_cache_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2] / ".cache" / "ordvec-beir"


def _default_runs_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2] / "results" / "beir"


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full evaluation and write all artefacts."""
    cache_dir = (
        pathlib.Path(args.cache_dir) if args.cache_dir else _default_cache_dir()
    )
    runs_dir = (
        pathlib.Path(args.runs_dir) if args.runs_dir else _default_runs_dir()
    )
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else runs_dir
    k_values = sorted(set(int(k) for k in args.k_values))

    per_dataset: list[dict[str, Any]] = []
    for dataset in args.datasets:
        print(f"[eval] Evaluating {dataset}/{args.split} ...", flush=True)
        per_dataset.append(
            evaluate_dataset(
                dataset=dataset,
                split=args.split,
                cache_dir=cache_dir,
                runs_dir=runs_dir,
                k_values=k_values,
                baseline=args.baseline,
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed,
                include_ann=args.include_ann,
            )
        )

    summary = assemble_summary_json(
        datasets=args.datasets,
        split=args.split,
        baseline=args.baseline,
        k_values=k_values,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
        include_ann=args.include_ann,
        per_dataset=per_dataset,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = out_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=False)
        fh.write("\n")

    bootstrap_path = out_dir / "bootstrap.json"
    with bootstrap_path.open("w", encoding="utf-8") as fh:
        json.dump(summary["bootstrap"], fh, indent=2)
        fh.write("\n")

    write_csv(out_dir / "summary.csv", summary["rows"], k_values, args.include_ann)

    # Render the markdown tables + summary.md via the report module.
    import beir_report

    beir_report.render_all(summary, out_dir)

    print(
        f"[eval] Wrote summary.json, summary.csv, bootstrap.json, "
        f"comparison-matrix.md and summary.md to {out_dir}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_eval(args)


if __name__ == "__main__":
    main()
