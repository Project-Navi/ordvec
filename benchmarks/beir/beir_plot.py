#!/usr/bin/env python3
"""Render the three README benchmark graphics from the Rust harness output.

Inputs (produced by `beir-bench`, written to `<runs-dir>/<dataset>/timing.jsonl`):
  * a SCALING sweep (one dataset swept over `--max-docs`, fixed `--threads 1`)
  * a SINGLE-THREAD full-corpus run (`--threads 1`)
  * a THREADED full-corpus run (`--threads N`)

Outputs (PNG + SVG) to `<out-dir>`:
  1. `scaling_curve.{png,svg}`   speedup-vs-`flat` as the corpus grows — the
     bands climb because exact brute force is O(n) while ordvec sign/rank
     candidate-gen is near-flat in n.
  2. `bars_single_thread.{png,svg}`  per-method query latency at 1 thread,
     full corpus — the controlled apples-to-apples bar.
  3. `bars_threaded.{png,svg}`   the same at N threads (matched batch).

No fabricated data: every point/bar is read straight from the harness records.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Presentation: stable method order, display labels, colours
# ---------------------------------------------------------------------------

# (slug, display label, colour). Order = legend / bar order.
METHOD_STYLE: list[tuple[str, str, str]] = [
    ("flat", "flat (exact IP, 4096 B)", "#444444"),
    ("hnsw", "HNSW M=32 (4096 B)", "#1f77b4"),
    ("ordvec-rq4", "ordvec RankQuant b=4 (512 B)", "#2ca02c"),
    ("ordvec-rq2", "ordvec RankQuant b=2 (256 B)", "#17becf"),
    ("ordvec-bitmap-rq2", "ordvec Bitmap→rq2 (384 B)", "#ff7f0e"),
    ("ordvec-sign-rq2", "ordvec Sign→rq2 (384 B)", "#d62728"),
]
LABEL = {s: lbl for s, lbl, _ in METHOD_STYLE}
COLOR = {s: c for s, _, c in METHOD_STYLE}
ORDER = [s for s, _, _ in METHOD_STYLE]


def _read_timing(path: pathlib.Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _dedupe_last(records: list[dict], key) -> list[dict]:
    """Keep the LAST record for each key (later runs overwrite earlier ones)."""
    out: dict = {}
    for r in records:
        out[key(r)] = r
    return list(out.values())


# ---------------------------------------------------------------------------
# Graphic 1: scaling curve (speedup vs flat, vs corpus size)
# ---------------------------------------------------------------------------

def plot_scaling(records: list[dict], dataset: str, threads: int, batch: int,
                 out_dir: pathlib.Path) -> None:
    recs = [r for r in records if r.get("threads") == threads and r.get("batch") == batch]
    recs = _dedupe_last(recs, lambda r: (r["method"], r["n_docs"]))

    # flat p50 at each n is the reference.
    flat_by_n = {r["n_docs"]: r["query_latency_ms_p50"] for r in recs if r["method"] == "flat"}
    if not flat_by_n:
        print("[plot] no 'flat' records in scaling sweep; skipping scaling_curve", file=sys.stderr)
        return

    mode = "single-query (batch=1)" if batch == 1 else f"batched (batch={batch})"
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for slug in ORDER:
        pts = sorted(
            ((r["n_docs"], r["query_latency_ms_p50"]) for r in recs if r["method"] == slug),
            key=lambda t: t[0],
        )
        xs = [n for n, _ in pts if n in flat_by_n]
        ys = [flat_by_n[n] / p for n, p in pts if n in flat_by_n and p > 0]
        if len(xs) < 2:
            continue
        if slug == "flat":
            ax.axhline(1.0, color=COLOR[slug], ls="--", lw=1.2, label=LABEL[slug])
        else:
            ax.plot(xs, ys, marker="o", lw=2.0, color=COLOR[slug], label=LABEL[slug])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size  (documents, log scale)")
    ax.set_ylabel("speedup vs exact flat  (×, log scale)")
    ax.set_title(
        f"ordvec scales: speedup over exact search grows with corpus size\n"
        f"{dataset}, {mode}, single-thread, Harrier-Q8 1024-d  (higher = faster than brute force)"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir, "scaling_curve")


# ---------------------------------------------------------------------------
# Graphics 2 & 3: per-method latency bars (single-thread / threaded)
# ---------------------------------------------------------------------------

def plot_bars(records: list[dict], dataset: str, threads: int, batch: int, n_docs: int,
              title: str, fname: str, out_dir: pathlib.Path) -> None:
    recs = _dedupe_last(
        [
            r for r in records
            if r.get("threads") == threads and r.get("batch") == batch and r.get("n_docs") == n_docs
        ],
        lambda r: r["method"],
    )
    by_method = {r["method"]: r for r in recs}
    slugs = [s for s in ORDER if s in by_method]
    if not slugs:
        print(f"[plot] no records for {fname} (threads={threads}, n={n_docs})", file=sys.stderr)
        return

    p50 = [by_method[s]["query_latency_ms_p50"] for s in slugs]
    qps = [by_method[s]["queries_per_second"] for s in slugs]
    colors = [COLOR[s] for s in slugs]
    labels = [LABEL[s].split(" (")[0] for s in slugs]

    flat_p50 = by_method.get("flat", {}).get("query_latency_ms_p50")

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    bars = ax.bar(range(len(slugs)), p50, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(slugs)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("query latency  p50 (ms/query, lower = better)")
    ax.set_title(title)
    ax.grid(True, axis="y", ls=":", alpha=0.4)

    for i, (b, ms, q) in enumerate(zip(bars, p50, qps)):
        spd = ""
        if flat_p50 and slugs[i] != "flat" and ms > 0:
            spd = f"\n{flat_p50 / ms:.1f}× vs flat"
        ax.text(
            b.get_x() + b.get_width() / 2, b.get_height(),
            f"{ms:.3f} ms\n{q:,.0f} q/s{spd}",
            ha="center", va="bottom", fontsize=7.5,
        )
    ax.set_ylim(0, max(p50) * 1.28)
    fig.tight_layout()
    _save(fig, out_dir, fname)


def _save(fig, out_dir: pathlib.Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_dir / stem}.png / .svg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="beir_plot", description="Render BEIR benchmark graphics.")
    p.add_argument("--runs-dir", default="results/beir")
    p.add_argument("--scaling-dataset", required=True,
                   help="Dataset swept over corpus size for the scaling curve (e.g. trec-covid).")
    p.add_argument("--bar-dataset", required=True,
                   help="Dataset for the single-thread / threaded latency bars.")
    p.add_argument("--scaling-threads", type=int, default=1)
    p.add_argument("--scaling-batch", type=int, default=1)
    p.add_argument("--bar-single-threads", type=int, default=1)
    p.add_argument("--bar-single-batch", type=int, default=1)
    p.add_argument("--bar-multi-threads", type=int, required=True,
                   help="Thread count for the threaded bar (must match a run).")
    p.add_argument("--bar-multi-batch", type=int, default=32)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    runs = pathlib.Path(args.runs_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else runs / "figures"

    # Scaling curve (single-query, single-thread by default).
    scaling_path = runs / args.scaling_dataset / "timing.jsonl"
    if scaling_path.is_file():
        plot_scaling(_read_timing(scaling_path), args.scaling_dataset,
                     args.scaling_threads, args.scaling_batch, out_dir)
    else:
        print(f"[plot] missing {scaling_path}; skipping scaling curve", file=sys.stderr)

    # Bars: pick the largest n_docs available for each (threads, batch) regime.
    bar_path = runs / args.bar_dataset / "timing.jsonl"
    if bar_path.is_file():
        bar_recs = _read_timing(bar_path)

        def _max_n(threads: int, batch: int) -> int:
            return max(
                (r["n_docs"] for r in bar_recs
                 if r.get("threads") == threads and r.get("batch") == batch),
                default=0,
            )

        n_single = _max_n(args.bar_single_threads, args.bar_single_batch)
        plot_bars(
            bar_recs, args.bar_dataset, args.bar_single_threads, args.bar_single_batch, n_single,
            f"Apples-to-apples, 1 thread, single-query — {args.bar_dataset} "
            f"({n_single:,} docs, Harrier-Q8 1024-d)\nall methods, one Rust process",
            "bars_single_thread", out_dir,
        )
        n_multi = _max_n(args.bar_multi_threads, args.bar_multi_batch)
        plot_bars(
            bar_recs, args.bar_dataset, args.bar_multi_threads, args.bar_multi_batch, n_multi,
            f"Apples-to-apples, {args.bar_multi_threads} threads, batched (batch={args.bar_multi_batch}) "
            f"— {args.bar_dataset}\n({n_multi:,} docs, Harrier-Q8 1024-d) — all methods, one Rust process",
            "bars_threaded", out_dir,
        )
    else:
        print(f"[plot] missing {bar_path}; skipping bars", file=sys.stderr)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
