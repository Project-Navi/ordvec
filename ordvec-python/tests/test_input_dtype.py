"""Embedding-input dtype/layout boundary contract (``as_f32_1d`` / ``as_f32_2d``).

ordvec normalises real-valued vector input to float32 at the FFI boundary — the
premise is *float vector in -> rank/sign transform*, so float32 is the internal
working dtype, not a contract the caller must pre-satisfy. The policy is uniform
across the four index types because every embedding entry point routes through
the same two choke-point helpers.

    Accepted:  float16 / float32 / float64, C-contiguous, finite after coercion
    Rejected:  bool, integers, complex, object, string        -> TypeError
               wrong ndim (scalar / 3-D)                       -> TypeError
               non-contiguous (transpose / stride)             -> ValueError
                 (never silently copied — the copy decision stays with the caller)
               non-finite after coercion (NaN / inf / f64 > f32::MAX)  -> ValueError

Why bool/int are rejected rather than coerced: a ``{0.0, 1.0}`` or narrow-integer
vector rank-transforms to an index-tie artefact (silent retrieval garbage), so
those are a deliberate usage-error guard, not an ergonomic gap. Candidate *IDs*
are a different boundary (labels, not measurements) and DO accept int64 — see
test_input_guards.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from ordvec import Bitmap, Rank, RankQuant, SignBitmap

INDEX_CLASSES = [Rank, RankQuant, Bitmap, SignBitmap]


def _make(index_cls):
    if index_cls is RankQuant:
        return RankQuant(dim=64, bits=2)
    if index_cls is Bitmap:
        return Bitmap(dim=64, n_top=16)
    return index_cls(dim=64)


def f32(n, dim=64, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


# -------------------------------------------------------------------
# Accepted: float dtypes, coerced to f32.
# -------------------------------------------------------------------


@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_float_dtypes_accepted(index_cls, dtype):
    idx = _make(index_cls)
    idx.add(np.ascontiguousarray(f32(8).astype(dtype)))
    assert len(idx) == 8


@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
def test_float64_coercion_is_faithful(index_cls):
    # f64 and f32 builds of the same values yield the same index — the rank/sign
    # transform is order/sign-only, and f64->f32 rounding is monotonic.
    v32 = f32(12)
    a = _make(index_cls)
    a.add(v32)
    b = _make(index_cls)
    b.add(v32.astype(np.float64))
    assert len(a) == len(b) == 12


# -------------------------------------------------------------------
# Rejected dtypes -> TypeError (bool/int/complex/object/string).
# -------------------------------------------------------------------


@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int32, np.int64, np.uint8, np.uint32, np.uint64, bool, np.complex64, np.complex128, object],
)
def test_nonfloat_dtypes_rejected(index_cls, dtype):
    with pytest.raises(TypeError):
        _make(index_cls).add(np.ones((8, 64), dtype=dtype))


@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
def test_string_dtype_rejected(index_cls):
    with pytest.raises(TypeError):
        _make(index_cls).add(np.full((8, 64), "x"))


# -------------------------------------------------------------------
# Rejected ndim -> TypeError (scalar / 3-D).
# -------------------------------------------------------------------


@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
def test_scalar_rejected(index_cls):
    with pytest.raises(TypeError):
        _make(index_cls).add(np.float32(1.0))


@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
def test_3d_rejected(index_cls):
    with pytest.raises(TypeError):
        _make(index_cls).add(np.zeros((2, 3, 64), dtype=np.float32))


# -------------------------------------------------------------------
# Rejected layout -> ValueError, checked BEFORE coercion so a float64 transpose
# is never silently laundered into a contiguous float32 (hidden copy).
# -------------------------------------------------------------------


@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_transpose_rejected_not_silently_copied(index_cls, dtype):
    v = np.asfortranarray(f32(8).astype(dtype))  # (8, 64) F-order -> non-C-contiguous
    assert not v.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        _make(index_cls).add(v)


# -------------------------------------------------------------------
# Rejected values -> ValueError, finite check AFTER coercion.
# -------------------------------------------------------------------


@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected(index_cls, bad):
    v = f32(8)
    v[2, 5] = bad
    with pytest.raises(ValueError, match="finite"):
        _make(index_cls).add(v)


@pytest.mark.filterwarnings("ignore:overflow encountered in cast")
@pytest.mark.parametrize("index_cls", INDEX_CLASSES)
def test_float64_overflow_to_inf_rejected(index_cls):
    # 1e300 is finite in float64 but rounds to +inf in float32 (NumPy's cast emits
    # a RuntimeWarning, ignored here) — the finite check runs on the POST-coercion
    # f32, so this is caught, not silently indexed.
    v = f32(8).astype(np.float64)
    v[0, 0] = 1e300
    with pytest.raises(ValueError, match="finite"):
        _make(index_cls).add(v)


# -------------------------------------------------------------------
# The 1-D query path (as_f32_1d) shares the contract.
# -------------------------------------------------------------------


def test_query_float64_accepted_and_faithful():
    corpus = f32(30)
    a = Bitmap(dim=64, n_top=16)
    a.add(corpus)
    q32 = f32(1, seed=7)[0]
    np.testing.assert_array_equal(
        a.top_m_candidates(q32, m=10),
        a.top_m_candidates(q32.astype(np.float64), m=10),
    )


def test_query_bool_rejected():
    a = SignBitmap(dim=64)
    a.add(f32(10))
    with pytest.raises(TypeError):
        a.top_m_candidates(np.ones(64, dtype=bool), m=5)


def test_query_noncontiguous_rejected():
    a = Bitmap(dim=64, n_top=16)
    a.add(f32(10))
    strided = np.ascontiguousarray(f32(1, dim=128)[0])[::2]  # len 64, non-contiguous
    assert not strided.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        a.top_m_candidates(strided, m=5)
