"""ordvec — training-free ordinal & sign vector quantization (Python bindings).

Original work by Nelson Spence, developed within the turbovec project
(MIT, by Ryan Codrai) and factored out. Dual-licensed MIT OR Apache-2.0.

Public API: ``Rank``, ``RankQuant``, ``Bitmap``, ``SignBitmap``.

The ``*Index`` names are back-compat aliases for the pre-0.2 turbovec-python
rank-mode classes; they are kept only to ease script migration and are not part
of the documented surface — new code should use the OrdVec ontology names above.
"""

from ._ordvec import Rank, RankQuant, Bitmap, SignBitmap

# Back-compat aliases for the pre-0.2 turbovec-python rank-mode names.
# Undocumented on purpose: present so existing scripts keep importing, while all
# docs/examples use the new ontology (Rank / RankQuant / Bitmap / SignBitmap).
RankIndex = Rank
RankQuantIndex = RankQuant
BitmapIndex = Bitmap
SignBitmapIndex = SignBitmap

__all__ = [
    "Rank",
    "RankQuant",
    "Bitmap",
    "SignBitmap",
    # back-compat aliases
    "RankIndex",
    "RankQuantIndex",
    "BitmapIndex",
    "SignBitmapIndex",
]

__version__ = "0.2.0"
