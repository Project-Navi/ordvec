"""
common.py — shared contract module for the ordvec-beir harness.

All public names here are imported by other lanes (beir_prepare.py,
eval.py, etc.).  DO NOT rename or remove any public symbol.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUERY_PROMPT: str = (
    "Instruct: Given a web search query, retrieve relevant passages that answer"
    " the query\nQuery: "
)

# ---------------------------------------------------------------------------
# Encoder slug
# ---------------------------------------------------------------------------

_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def encoder_slug(provider: str, model: str, revision: str | None) -> str:
    """Return a deterministic, filesystem-safe slug for an encoder spec.

    Format:  <provider>__<model-safe>__<revision-or-norev>
    All ``/`` and ``:`` are replaced with ``__``; other unsafe chars are
    replaced with ``_``.

    Examples
    --------
    >>> encoder_slug("st", "microsoft/harrier-oss-v1-0.6b", "abc123")
    'st__microsoft__harrier-oss-v1-0.6b__abc123'
    """
    def _safe(s: str) -> str:
        s = s.replace("/", "__").replace(":", "__")
        s = _UNSAFE_RE.sub("_", s)
        return s

    rev_part = _safe(revision) if revision else "norev"
    return f"{_safe(provider)}__{_safe(model)}__{rev_part}"


# ---------------------------------------------------------------------------
# Cache / results path helpers
# ---------------------------------------------------------------------------

def dataset_cache_dir(
    cache_dir: str | pathlib.Path,
    dataset: str,
    split: str,
    slug: str,
) -> pathlib.Path:
    """Return (and create) the per-encoder cache directory.

    Layout: <cache_dir>/<dataset>/<split>/encoder=<slug>/
    """
    p = pathlib.Path(cache_dir) / dataset / split / f"encoder={slug}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def find_encoder_dir(
    cache_dir: str | pathlib.Path,
    dataset: str,
    split: str,
) -> pathlib.Path:
    """Resolve the single ``encoder=*`` sub-directory.

    Raises ``FileNotFoundError`` if zero matches, ``ValueError`` if >1.
    """
    base = pathlib.Path(cache_dir) / dataset / split
    matches = list(base.glob("encoder=*"))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No encoder directory found under {base}. "
            "Run beir_prepare.py first."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple encoder directories found under {base}: {matches}. "
            "Specify --encoder-slug to disambiguate."
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Manifest / metadata I/O
# ---------------------------------------------------------------------------

def load_manifest(enc_dir: str | pathlib.Path) -> dict:
    """Read ``embeddings.manifest.json`` from *enc_dir* and return its dict."""
    path = pathlib.Path(enc_dir) / "embeddings.manifest.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_ids(
    enc_dir: str | pathlib.Path,
) -> tuple[list[str], list[str]]:
    """Return ``(corpus_ids, query_ids)`` loaded from the cache directory."""
    enc_dir = pathlib.Path(enc_dir)
    with (enc_dir / "corpus_ids.json").open("r", encoding="utf-8") as fh:
        corpus_ids: list[str] = json.load(fh)
    with (enc_dir / "query_ids.json").open("r", encoding="utf-8") as fh:
        query_ids: list[str] = json.load(fh)
    return corpus_ids, query_ids


def load_qrels(
    enc_dir: str | pathlib.Path,
) -> dict[str, dict[str, int]]:
    """Return ``qrels`` dict ``{qid: {doc_id: relevance_int}}``."""
    path = pathlib.Path(enc_dir) / "qrels.json"
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Embedding array I/O
# ---------------------------------------------------------------------------

def load_npy_f32(path: str | pathlib.Path) -> "np.ndarray":
    """Load a 2-D C-order float32 ``.npy`` array and validate its shape.

    Raises
    ------
    ValueError
        If the array is not 2-D, not float32, or not C-contiguous.
    """
    import numpy as np  # local import — numpy may not be installed at import time

    arr = np.load(str(path))
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D array; got shape {arr.shape} from {path}"
        )
    if arr.dtype != np.float32:
        raise ValueError(
            f"Expected float32 array; got dtype={arr.dtype} from {path}"
        )
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
    return arr


# ---------------------------------------------------------------------------
# Embedding validation (fail-closed)
# ---------------------------------------------------------------------------

def validate_embeddings(arr: "np.ndarray") -> None:
    """Raise ``ValueError`` on any violation of the spec's fail-closed rules.

    Rules
    -----
    * 2-D array
    * dtype == float32
    * shape[1] == 1024
    * shape[1] % 16 == 0
    * every row L2 norm in [0.999, 1.001]
    """
    import numpy as np

    if arr.ndim != 2:
        raise ValueError(f"Embeddings must be 2-D; got shape {arr.shape}")
    if arr.dtype != np.float32:
        raise ValueError(
            f"Embeddings must be float32; got dtype={arr.dtype}"
        )
    dim = arr.shape[1]
    if dim != 1024:
        raise ValueError(
            f"Embedding dimension must be 1024; got {dim}"
        )
    if dim % 16 != 0:
        raise ValueError(
            f"Embedding dimension must be divisible by 16; got {dim}"
        )
    norms = np.linalg.norm(arr, axis=1)
    bad = np.where((norms < 0.999) | (norms > 1.001))[0]
    if bad.size > 0:
        raise ValueError(
            f"{bad.size} rows have L2 norm outside [0.999, 1.001]. "
            f"First offending row: index={bad[0]}, norm={norms[bad[0]]:.6f}"
        )


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_file(path: str | pathlib.Path) -> str:
    """Return the hex-encoded SHA-256 digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Results path helpers
# ---------------------------------------------------------------------------

def slug_for_method(
    method: str,
    candidate_m: int | None,
    batch: int | None,
) -> str:
    """Build a results slug from method name + optional params.

    Examples
    --------
    >>> slug_for_method("ordvec-bitmap-rq2", 500, 8)
    'ordvec-bitmap-rq2-m500-b8'
    >>> slug_for_method("dense-exact", None, None)
    'dense-exact'
    """
    parts = [method]
    if candidate_m is not None:
        parts.append(f"m{candidate_m}")
    if batch is not None:
        parts.append(f"b{batch}")
    return "-".join(parts)


def topk_jsonl_path(
    runs_dir: str | pathlib.Path,
    dataset: str,
    method_slug: str,
) -> pathlib.Path:
    """Return the path for the top-k JSONL results file."""
    p = pathlib.Path(runs_dir) / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{method_slug}.topk.jsonl"


def summary_json_path(
    runs_dir: str | pathlib.Path,
    dataset: str,
    method_slug: str,
) -> pathlib.Path:
    """Return the path for the summary JSON file."""
    p = pathlib.Path(runs_dir) / dataset
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{method_slug}.summary.json"


# ---------------------------------------------------------------------------
# Top-k JSONL I/O
# ---------------------------------------------------------------------------

def write_topk_jsonl(path: str | pathlib.Path, rows: list[dict]) -> None:
    """Write *rows* as newline-delimited JSON (one object per line).

    Each row must conform to the spec schema::

        {"dataset", "split", "method", "qid_idx", "qid",
         "k", "doc_idxs": [int], "doc_ids": [str], "scores": [float]}
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_topk_jsonl(path: str | pathlib.Path) -> list[dict]:
    """Read a top-k JSONL file and return a list of row dicts."""
    path = pathlib.Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
