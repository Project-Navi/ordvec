"""ordvec — training-free ordinal & sign vector quantization (Python bindings).

Developed within the turbovec project
(MIT, by Ryan Codrai) and factored out. Dual-licensed MIT OR Apache-2.0.

Public API: the four index classes ``Rank``, ``RankQuant``, ``Bitmap``,
``SignBitmap``, plus the module-level rank-math primitives (``rank_transform``,
``rank_to_bucket``, ``bucket_ranks``, ``pack_buckets``, ``unpack_buckets``,
``rankquant_bytes_per_vec``, ``bucket_centre``, ``rank_norm``,
``rankquant_norm``), the byte-LUT scoring helper ``search_asymmetric_byte_lut``,
and the loader limit constants (``MAX_DIM``, ``MAX_SIGN_BITMAP_DIM``,
``MAX_VECTORS``). Together with the four classes' methods this mirrors the Rust
crate's public API; the low-level ``rank_io`` read/write functions are reached
through the classes' ``write()`` / ``load()`` methods rather than exposed as
standalone free functions.

The ``*Index`` names are back-compat aliases for the pre-0.2 turbovec-python
rank-mode classes; they are kept only to ease script migration and are not part
of the documented surface — new code should use the OrdVec ontology names above.
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

__version__ = "0.2.0"
