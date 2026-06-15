"""
beir_report.py — Render the public BEIR report tables from summary.json.

Responsibilities (spec §10)
---------------------------
Render three markdown tables plus the required-claims preamble:

1. **Comparison matrix** — family / method / implementation / search-type /
   bytes-per-vector-at-1024d / headline-role.
2. **Main per-dataset table** — dataset, encoder, method, nDCG@10, Δ vs FAISS,
   95% CI, MAP@10, Recall@100, bytes/vec, build s, p50, p95.
3. **Rollup table** — method, mean nDCG@10, mean Δ vs FAISS, datasets-within-CI,
   mean Recall@100, bytes/vec.

The report ALWAYS shows ``encoder_provider`` so HF and GGUF (Ollama) numbers are
never silently mixed.  All tables use ``tabulate`` (GitHub-flavoured markdown).

It writes ``comparison-matrix.md`` and ``summary.md`` (the latter embeds all
three tables plus the two required-claims paragraphs from spec §12, verbatim).

This module can be imported by :mod:`beir_eval` (``render_all``) or run on its
own against an existing ``summary.json`` (``python beir_report.py
results/beir/summary.json``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

# Allow `import beir_eval` when run as a script from the repo root
# (the Makefile invokes `python3 benchmarks/beir/<script>.py`).
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Required-claims paragraphs (spec §12) — reproduced VERBATIM.
# Editing these breaks the claims-discipline guarantee; keep them in sync with
# benchmarks/beir/README.md's "Claims discipline" section.
# ---------------------------------------------------------------------------

REQUIRED_CLAIM_REPRODUCIBILITY = (
    "**Benchmark numbers in this repository reflect synthetic or user-runnable "
    "real-corpus experiments only.  No numbers are fabricated or cherry-picked.  "
    "Every result file produced by `make benchmark-beir` is fully reproducible "
    "from the commands documented here, using publicly available BEIR datasets "
    "and the pinned encoder revision recorded in `embeddings.manifest.json`.**"
)

REQUIRED_CLAIM_FAISS_NOT_GROUND_TRUTH = (
    "**The `flat` baseline is an exact full-float inner-product search (identical "
    "retrieval to FAISS `IndexFlatIP`) used for comparison purposes — it is NOT "
    "ground truth.  nDCG@10 is computed against the official BEIR qrels "
    "(human-annotated relevance judgements), not against the `flat` results.  "
    "Recall-vs-`flat` (fraction of the exact top-k recovered by an approximate "
    "method) is an optional diagnostic metric only; it does not substitute for "
    "qrel-based evaluation.**"
)


# ---------------------------------------------------------------------------
# tabulate shim
# ---------------------------------------------------------------------------

def _tabulate(rows: list[list[Any]], headers: list[str]) -> str:
    """Render a GitHub-flavoured markdown table via ``tabulate``."""
    try:
        from tabulate import tabulate as _t
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise SystemExit(
            "tabulate is required for report rendering but is not installed. "
            "Install it with `pip install tabulate`."
        ) from exc
    return _t(rows, headers=headers, tablefmt="github")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_num(value: Any, places: int = 4) -> str:
    """Format a numeric value; ``None``/missing → ``"-"``."""
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _fmt_ci(low: Any, high: Any) -> str:
    if low is None or high is None:
        return "-"
    return f"[{float(low):+.4f}, {float(high):+.4f}]"


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.4f}"


def _encoder_label(encoder: dict[str, Any]) -> str:
    """Compact ``provider / model`` label (never silently drops provider)."""
    provider = encoder.get("encoder_provider", "unknown")
    model = encoder.get("encoder_model", "unknown")
    return f"{provider} / {model}"


# ---------------------------------------------------------------------------
# Bootstrap index
# ---------------------------------------------------------------------------

def _bootstrap_index(
    bootstrap: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index bootstrap entries by ``(dataset, method, metric)``."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in bootstrap:
        key = (entry["dataset"], entry["method"], entry["metric"])
        out[key] = entry
    return out


# ---------------------------------------------------------------------------
# Comparison matrix
# ---------------------------------------------------------------------------

def render_comparison_matrix(summary: dict[str, Any]) -> str:
    """Render the family/method/implementation/search-type matrix."""
    import beir_eval  # local import: family metadata + stem helper live there

    headers = [
        "Family",
        "Method",
        "Implementation",
        "Search type",
        "Bytes/vec @1024d",
        "Headline role",
    ]
    # One row per unique method slug, sorted with the baseline first.
    seen: dict[str, dict[str, Any]] = {}
    for row in summary["rows"]:
        seen.setdefault(row["method"], row)

    baseline = summary["config"]["baseline"]

    def _sort_key(method: str) -> tuple[int, str]:
        return (0 if method == baseline else 1, method)

    body: list[list[Any]] = []
    for method in sorted(seen, key=_sort_key):
        row = seen[method]
        meta = beir_eval.family_meta(method)
        body.append(
            [
                meta["family"],
                method,
                meta["implementation"],
                meta["search_type"],
                _fmt_int(row.get("bytes_per_vector")),
                meta["headline_role"],
            ]
        )
    return _tabulate(body, headers)


# ---------------------------------------------------------------------------
# Main per-dataset table
# ---------------------------------------------------------------------------

def render_main_table(summary: dict[str, Any]) -> str:
    """Render the per-(dataset, method) headline table."""
    boot = _bootstrap_index(summary["bootstrap"])
    baseline = summary["config"]["baseline"]
    encoders = summary.get("encoders", {})

    headers = [
        "Dataset",
        "Encoder (provider / model)",
        "Method",
        "nDCG@10",
        f"Δ vs {baseline}",
        "95% CI",
        "MAP@10",
        "Recall@100",
        "Bytes/vec",
        "Build s",
        "p50 ms",
        "p95 ms",
    ]

    # Group rows by dataset (preserve config order), method baseline-first.
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in summary["rows"]:
        by_dataset.setdefault(row["dataset"], []).append(row)

    body: list[list[Any]] = []
    for dataset in summary["config"]["datasets"]:
        rows = by_dataset.get(dataset, [])
        rows = sorted(
            rows, key=lambda r: (0 if r["method"] == baseline else 1, r["method"])
        )
        enc_label = _encoder_label(encoders.get(dataset, {}))
        for row in rows:
            method = row["method"]
            if method == baseline:
                delta_str = "(baseline)"
                ci_str = "-"
            else:
                b = boot.get((dataset, method, "ndcg@10"))
                if b is None:
                    delta_str = "-"
                    ci_str = "-"
                else:
                    delta_str = _fmt_delta(b["delta"])
                    if b.get("within_noise"):
                        delta_str += " *"
                    ci_str = _fmt_ci(b["ci95_low"], b["ci95_high"])
            body.append(
                [
                    dataset,
                    enc_label,
                    method,
                    _fmt_num(row.get("ndcg@10")),
                    delta_str,
                    ci_str,
                    _fmt_num(row.get("map@10")),
                    _fmt_num(row.get("recall@100")),
                    _fmt_int(row.get("bytes_per_vector")),
                    _fmt_num(row.get("build_seconds"), 2),
                    _fmt_num(row.get("query_latency_ms_p50"), 3),
                    _fmt_num(row.get("query_latency_ms_p95"), 3),
                ]
            )
    return _tabulate(body, headers)


# ---------------------------------------------------------------------------
# Rollup table
# ---------------------------------------------------------------------------

def render_rollup_table(summary: dict[str, Any]) -> str:
    """Render the cross-dataset rollup per method."""
    boot = _bootstrap_index(summary["bootstrap"])
    baseline = summary["config"]["baseline"]
    datasets = summary["config"]["datasets"]

    headers = [
        "Method",
        "Mean nDCG@10",
        f"Mean Δ vs {baseline}",
        "Datasets within CI",
        "Mean Recall@100",
        "Bytes/vec",
    ]

    # Accumulate per-method values across datasets.
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    for row in summary["rows"]:
        rows_by_method.setdefault(row["method"], []).append(row)

    def _mean(values: list[float]) -> float | None:
        clean = [v for v in values if v is not None]
        return (sum(clean) / len(clean)) if clean else None

    def _sort_key(method: str) -> tuple[int, str]:
        return (0 if method == baseline else 1, method)

    n_datasets = len(datasets)
    body: list[list[Any]] = []
    for method in sorted(rows_by_method, key=_sort_key):
        rows = rows_by_method[method]
        mean_ndcg = _mean([r.get("ndcg@10") for r in rows])
        mean_recall = _mean([r.get("recall@100") for r in rows])
        # Bytes/vec is fixed per method; take the first defined value.
        bytes_per_vec = next(
            (r.get("bytes_per_vector") for r in rows
             if r.get("bytes_per_vector") is not None),
            None,
        )

        if method == baseline:
            mean_delta_str = "(baseline)"
            within_str = "-"
        else:
            deltas = []
            within = 0
            counted = 0
            for ds in datasets:
                b = boot.get((ds, method, "ndcg@10"))
                if b is None:
                    continue
                counted += 1
                deltas.append(b["delta"])
                if b.get("within_noise"):
                    within += 1
            mean_delta = _mean(deltas)
            mean_delta_str = _fmt_delta(mean_delta) if mean_delta is not None else "-"
            denom = counted if counted else n_datasets
            within_str = f"{within}/{denom}"

        body.append(
            [
                method,
                _fmt_num(mean_ndcg),
                mean_delta_str,
                within_str,
                _fmt_num(mean_recall),
                _fmt_int(bytes_per_vec),
            ]
        )
    return _tabulate(body, headers)


# ---------------------------------------------------------------------------
# summary.md assembly
# ---------------------------------------------------------------------------

def render_summary_md(summary: dict[str, Any]) -> str:
    """Assemble the full summary.md document."""
    config = summary["config"]
    encoders = summary.get("encoders", {})

    # Provider audit line — surfaces a mixed-provider run loudly.
    providers = sorted(
        {enc.get("encoder_provider", "unknown") for enc in encoders.values()}
    )
    provider_note = ", ".join(providers) if providers else "unknown"
    mixed_warning = ""
    if len(providers) > 1:
        mixed_warning = (
            "\n> **Warning:** this report mixes more than one encoder provider "
            f"({provider_note}).  HF and GGUF numbers are NOT comparable; "
            "inspect the per-dataset encoder column before drawing conclusions.\n"
        )

    parts: list[str] = []
    parts.append("# ordvec BEIR evaluation summary\n")
    parts.append(
        f"- **Split:** `{config['split']}`\n"
        f"- **Datasets:** {', '.join(config['datasets'])}\n"
        f"- **Baseline:** `{config['baseline']}` "
        "(comparison only — not ground truth)\n"
        f"- **Headline metric:** {config['headline_metric']}\n"
        f"- **k-values:** {', '.join(str(k) for k in config['k_values'])}\n"
        f"- **Bootstrap iters:** {config['bootstrap_iters']} "
        f"(seed {config['seed']})\n"
        f"- **Encoder provider(s):** {provider_note}\n"
    )
    parts.append(mixed_warning)

    parts.append("\n## Claims discipline\n")
    parts.append("\n> " + REQUIRED_CLAIM_REPRODUCIBILITY + "\n")
    parts.append("\n> " + REQUIRED_CLAIM_FAISS_NOT_GROUND_TRUTH + "\n")

    parts.append("\n## Comparison matrix\n\n")
    parts.append(render_comparison_matrix(summary))
    parts.append("\n")

    parts.append("\n## Per-dataset results\n\n")
    parts.append(
        "`Δ vs FAISS` is the paired-bootstrap mean nDCG@10 delta "
        "(method - baseline); a trailing `*` marks deltas whose 95% CI "
        "straddles 0 (within noise).\n\n"
    )
    parts.append(render_main_table(summary))
    parts.append("\n")

    parts.append("\n## Rollup (mean across datasets)\n\n")
    parts.append(render_rollup_table(summary))
    parts.append("\n")

    if config.get("include_ann_diagnostics"):
        parts.append("\n## ANN recall diagnostic (vs baseline)\n\n")
        parts.append(render_ann_table(summary))
        parts.append("\n")

    return "".join(parts)


def render_ann_table(summary: dict[str, Any]) -> str:
    """Render the optional ANN-recall@100-vs-baseline diagnostic table."""
    headers = ["Dataset", "Method", "ANN recall@100 (vs baseline)"]
    baseline = summary["config"]["baseline"]
    body: list[list[Any]] = []
    for dataset in summary["config"]["datasets"]:
        for row in summary["rows"]:
            if row["dataset"] != dataset or row["method"] == baseline:
                continue
            if "ann_recall@100" not in row:
                continue
            body.append(
                [dataset, row["method"], _fmt_num(row["ann_recall@100"])]
            )
    if not body:
        return "_No ANN diagnostic data (run with `--include-ann-diagnostics`)._"
    return _tabulate(body, headers)


# ---------------------------------------------------------------------------
# Render-all entrypoint (called by beir_eval)
# ---------------------------------------------------------------------------

def render_all(summary: dict[str, Any], out_dir: pathlib.Path) -> None:
    """Write ``comparison-matrix.md`` and ``summary.md`` into *out_dir*."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_md = (
        "# ordvec BEIR comparison matrix\n\n"
        + render_comparison_matrix(summary)
        + "\n"
    )
    (out_dir / "comparison-matrix.md").write_text(matrix_md, encoding="utf-8")

    summary_md = render_summary_md(summary)
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI (standalone rendering from an existing summary.json)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="beir_report",
        description="Render BEIR report tables from a summary.json.",
    )
    p.add_argument(
        "summary_json",
        help="Path to summary.json produced by beir_eval.py.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        dest="out_dir",
        help="Output directory (default: directory of summary.json).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    summary_path = pathlib.Path(args.summary_json)
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    out_dir = (
        pathlib.Path(args.out_dir) if args.out_dir else summary_path.parent
    )
    render_all(summary, out_dir)
    print(f"[report] Wrote comparison-matrix.md and summary.md to {out_dir}")


if __name__ == "__main__":
    main()
