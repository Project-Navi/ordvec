"""ordvec — training-free ordinal & sign vector quantization (Python bindings).

Developed within the turbovec project
(MIT, by Ryan Codrai) and factored out. Dual-licensed MIT OR Apache-2.0.

Public API: the four index classes ``Rank``, ``RankQuant``, ``Bitmap``,
``SignBitmap``, plus the module-level rank-math primitives (``rank_transform``,
``rank_to_bucket``, ``bucket_ranks``, ``pack_buckets``, ``unpack_buckets``,
``rankquant_bytes_per_vec``, ``bucket_centre``, ``rank_norm``,
``rankquant_norm``), the eval-only arbitrary-width scorer
``rankquant_eval_search``, the byte-LUT scoring helper
``search_asymmetric_byte_lut``, and the loader limit constants (``MAX_DIM``,
``MAX_SIGN_BITMAP_DIM``, ``MAX_VECTORS``). Together with the four classes'
methods this mirrors the headline Rust retrieval API. Rust-only metadata
probing and manifest-verification helpers remain available through the Rust
crates and the ``ordvec-manifest`` CLI; the low-level ``rank_io`` read/write
functions are reached through the classes' ``write()`` / ``load()`` methods
rather than exposed as standalone free functions.

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
The input arrays are *read in place* (not copied) for that window — do not
mutate an array from another thread while a call that received it is in
progress, including subset candidate arrays, or the scan races the write and
may return inconsistent results. This is the standard contract for
GIL-releasing numeric extensions (NumPy itself behaves this way).
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
    search_asymmetric_byte_lut,
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
    "search_asymmetric_byte_lut",
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

__version__ = "0.4.0"
