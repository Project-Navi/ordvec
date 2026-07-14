"""ordvec — training-free ordinal & sign vector quantization (Python bindings).

ordvec was developed using the early turbovec project context as a
rapid-development scaffold, with thanks to that lineage. ordvec's implementation
history, active development, issues, releases, and governance live in
Project-Navi/ordvec. Dual-licensed MIT OR Apache-2.0.

Public API: the four index classes ``Rank``, ``RankQuant``, ``Bitmap``,
``SignBitmap``, plus the module-level rank-math primitives (``rank_transform``,
``rank_to_bucket``, ``bucket_ranks``, ``pack_buckets``, ``unpack_buckets``,
``rankquant_bytes_per_vec``, ``bucket_centre``, ``rank_norm``,
``rankquant_norm``), the eval-only arbitrary-width scorer
``rankquant_eval_search``, and the loader limit constants (``MAX_DIM``,
``MAX_SIGN_BITMAP_DIM``, ``MAX_VECTORS``). Together with the four classes'
methods this mirrors the headline Rust retrieval API. Rust-only metadata,
benchmark, and manifest-verification helpers remain available through the Rust
crates and the ``ordvec-manifest`` CLI; the low-level ``rank_io`` read/write
functions are reached through the classes' ``write()`` / ``load()`` methods
rather than exposed as standalone free functions. The specialized
``RankQuantFastscan`` b=2 fast path (and its ``.ovfs`` persistence) is a
Rust-only type and is intentionally not exposed in this binding.

``Bitmap`` exposes the constant-weight top-bucket overlap statistic formalized
in the companion ``ordvec-formalization`` Lean repo: under explicit finite
symmetry, quotient, and monotone-overlap assumptions, an overlap threshold is
Bayes-optimal and the idealized uniform constant-weight null gives that
threshold the hypergeometric upper tail. This is an in-model candidate-admission
claim, not a guarantee that real encoders or deployment corpora satisfy those
assumptions.

The ``*Index`` names are back-compat aliases for the pre-0.2 turbovec-python
rank-mode classes; they are kept only to ease script migration and are not part
of the documented surface — new code should use the OrdVec ontology names above.

Subset rerank result length: ``RankQuant.search_asymmetric_subset(query,
candidates, k)`` returns ``(scores, ids)`` of length ``min(k, len(candidates))``,
not ``k``. Passing ``k > len(candidates)`` yields arrays shorter than ``k`` (the
subset path does not pad with sentinel rows), so a caller building a fixed-width
``(n_q, k)`` buffer must size each row by its candidate count.

On-disk persistence: each class's ``write(path)`` / ``load(path)`` passes
``path`` straight to the filesystem with no normalisation or ``..`` / traversal
checks. Treat ``path`` as trusted input — in a service that derives it from
caller-supplied data, validate or sandbox the path first, exactly as you would
before a bare ``open()``.

Threading: the contract is read-concurrent, mutation-exclusive. ``search``,
``search_asymmetric``, ``search_asymmetric_subset``, and the dense scoring /
candidate generator methods release the GIL during the heavy Rust scan, so
other Python threads run concurrently. ``add`` also releases the GIL while
mutating an index, but mutable index operations must be treated as exclusive.
GIL-released search, candidate-generation, scoring, and ``add`` methods copy
NumPy inputs into Rust-owned buffers before detaching, so ordinary Python
in-place NumPy mutation in another thread cannot race detached Rust reads. This
intentionally trades zero-copy detached reads for race-free snapshots; large
calls may temporarily require an additional input-sized buffer. Callers still
own object-level scheduling: do not overlap mutable index operations such as
``add`` with searches on the same index unless the binding method explicitly
documents that pattern.
"""

from ._ordvec import (
    MAX_DIM,
    MAX_SIGN_BITMAP_DIM,
    MAX_VECTORS,
    Bitmap,
    Rank,
    RankQuant,
    SignBitmap,
    bucket_centre,
    bucket_ranks,
    pack_buckets,
    rank_norm,
    rank_to_bucket,
    rank_transform,
    rankquant_eval_search,
    rankquant_bytes_per_vec,
    rankquant_norm,
    unpack_buckets,
)

# Back-compat aliases for the pre-0.2 turbovec-python rank-mode names.
# Undocumented on purpose: present so existing scripts keep importing, while all
# docs/examples use the new ontology (Rank / RankQuant / Bitmap / SignBitmap).
RankIndex = Rank
RankQuantIndex = RankQuant
BitmapIndex = Bitmap
SignBitmapIndex = SignBitmap

__all__ = [
    # index classes
    "Rank",
    "RankQuant",
    "Bitmap",
    "SignBitmap",
    # rank-math primitives
    "rank_transform",
    "rank_to_bucket",
    "bucket_ranks",
    "pack_buckets",
    "unpack_buckets",
    "rankquant_bytes_per_vec",
    "bucket_centre",
    "rank_norm",
    "rankquant_norm",
    "rankquant_eval_search",
    # loader limit constants
    "MAX_DIM",
    "MAX_SIGN_BITMAP_DIM",
    "MAX_VECTORS",
    # back-compat aliases
    "RankIndex",
    "RankQuantIndex",
    "BitmapIndex",
    "SignBitmapIndex",
]

__version__ = "0.7.0"
