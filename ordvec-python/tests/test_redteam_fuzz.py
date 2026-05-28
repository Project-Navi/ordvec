"""Adversarial / red-team fuzz of the ordvec FFI boundary (callsign: Cipher).

Goal: feed the Python API garbage and confirm that *every* malformed input
surfaces as a clean, typed Python exception (``ValueError`` / ``IndexError`` /
``TypeError`` / ``IOError``) and never as a ``pyo3_runtime.PanicException``, a
hard interpreter abort / segfault, an OOM/hang, or a silently wrong result.

This file is the offensive complement to ``test_input_guards.py``: that suite
pins the three documented guard classes (non-finite, non-contiguous, OOR
subset ids); this one goes after the corners those tests do *not* touch —

* integer-scalar abuse for ``k`` / ``m`` / ``batch_size`` / ``idx`` /
  candidate ids: negative, ``2**63``, ``2**64`` (the wrap-to-giant-usize →
  OOM hypothesis);
* the ``from_shape_vec`` reshape and ``m_eff`` flatten invariants under
  adversarial ``m`` / ``k`` (the ``debug_assert_eq!`` in the batched flatten is
  compiled out in ``--release``, so a row-width mismatch would ``.expect()``-panic
  there — these tests assert it never does);
* the four on-disk loaders against truncated / extended / forged / corrupt
  files and a forged-huge-dim DoS-allocation header;
* exotic dtypes (bool / float16 / object / complex / int families) and NaN bit
  patterns (signaling + quiet) across every f32 entry point;
* type confusion on the ``search_asymmetric_byte_lut`` ``PyRef<RankQuant>`` arg
  and on every ``None`` / list / str argument;
* the documented PyO3 borrow-flag reentrancy contract (a ``__index__`` callback
  that re-enters a ``&mut self`` method on the object a ``&self`` method already
  borrowed → clean ``Already borrowed`` ``RuntimeError``, never a data race).

Abort-class probes (anything that *could* crash the interpreter rather than
raise) are run in a child process via ``_run_isolated`` so a hypothetical
segfault is observed as a non-zero child exit code instead of taking the whole
pytest session down with it.

Findings: the guards held on every vector probed — no genuine bug was found, so
every test here asserts CORRECT (guarded) behavior and passes. Any test that
*would* document a real defect is marked ``@pytest.mark.skip(reason="BUG: …")``;
there are currently none. See ``/tmp/cipher/py/findings.md`` for the full report.
"""
from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from ordvec import (
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


def unit_vectors(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    return v


# NaN bit patterns numpy/Rust must both treat as non-finite. Quiet NaN, signaling
# NaN, and a negative signaling NaN — `f32::is_finite()` must reject all three.
_QNAN = float(np.array([0x7FC00000], dtype=np.uint32).view(np.float32)[0])
_SNAN = float(np.array([0x7FA00000], dtype=np.uint32).view(np.float32)[0])
_NEG_SNAN = float(np.array([0xFFA00000], dtype=np.uint32).view(np.float32)[0])

# dtypes a float32 embedding param must reject. float16/float32/float64 are now
# coerced at the boundary (see test_input_dtype.py); integer and bool arrays are
# *deliberately* rejected (a {0,1} or narrow-int vector rank-transforms to a
# degenerate index artefact, not a meaningful ordinal signal), and complex/object
# would be a silent reinterpretation.
_WRONG_F32_DTYPES = [
    np.int32,
    np.int64,
    np.uint8,
    np.uint32,
    bool,
    np.complex64,
    object,
]

# Non-integer dtypes a candidate / doc-id param must reject. Integer dtypes are
# accepted (converted to u32 with checked bounds — see test_input_guards.py); a
# float / complex / object array is a clear TypeError, never a truncation.
_NON_INTEGER_ID_DTYPES = [
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    bool,
    object,
]

# Integer scalars that must NOT wrap to a giant usize / OOM. PyO3 maps a negative
# Python int and anything >= 2**64 to a clean OverflowError on usize conversion.
_BAD_INT_SCALARS = [-1, -(2**40), 2**64, 2**70]
# Huge-but-valid usize values that the core must CLAMP (to k<=n / m<=n), never
# allocate eagerly. usize::MAX is 2**64-1 on 64-bit but only 2**32-1 on 32-bit, so
# a 2**40+ literal would raise OverflowError at the PyO3 usize conversion on a
# 32-bit target (before reaching the clamp). Pick values that fit usize on each
# target so the clamp path is what's exercised everywhere.
_64BIT = sys.maxsize > 2**32
_HUGE_VALID_USIZE = [2**40, 2**62, 2**63] if _64BIT else [2**30, 2**31]
# m-sweep lists for the batched/chunked flatten-invariant tests (same rationale).
_HUGE_M = [0, 1, 1000, 2**40, 2**62] if _64BIT else [0, 1, 1000, 2**30, 2**31]
_HUGE_M_MID = [0, 1, 1000, 2**62] if _64BIT else [0, 1, 1000, 2**31]
_HUGE_M_SIMPLE = [0, 1, 2**62] if _64BIT else [0, 1, 2**31]


# =====================================================================
# Subprocess-isolation harness for abort-class probes.
# A segfault/abort cannot be caught by pytest.raises (it kills the process),
# so run the probe in a child and assert it exits 0 (clean) — a crash shows up
# as a negative return code (terminating signal) or a non-zero exit.
# =====================================================================

_CHILD_PREAMBLE = (
    "import numpy as np\n"
    "from ordvec import Rank, RankQuant, Bitmap, SignBitmap\n"
    "def uv(n,d,s=0):\n"
    "    rng=np.random.default_rng(s); v=rng.standard_normal((n,d)).astype(np.float32)\n"
    "    v/=np.linalg.norm(v,axis=1,keepdims=True)+1e-9; return v\n"
)


def _run_isolated(body: str) -> subprocess.CompletedProcess:
    """Run a probe body in a child interpreter; return the completed process.

    The child prints ``OK`` on a clean finish. The caller asserts
    ``returncode == 0`` so a hard abort (negative rc = killed by signal) or an
    uncaught ``PanicException`` (rc = 1, traceback on stderr) fails loudly here
    instead of crashing the pytest session.
    """
    src = _CHILD_PREAMBLE + body + "\nprint('OK')\n"
    return subprocess.run(
        [sys.executable, "-c", src],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_clean_child(proc: subprocess.CompletedProcess) -> None:
    assert proc.returncode == 0, (
        f"child crashed: rc={proc.returncode} "
        f"(negative = killed by signal {-proc.returncode} → abort/segfault)\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "PanicException" not in proc.stderr, (
        f"core panic leaked across the FFI boundary:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), (
        f"child did not finish cleanly:\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
    )


# =====================================================================
# KEY PROBE: integer-scalar abuse for k / m / batch_size / idx / candidate ids.
# A negative or >= 2**64 Python int must raise a clean OverflowError, never wrap
# to a giant usize that triggers an OOM or an OOB panic in the core.
# =====================================================================


@pytest.mark.parametrize("bad_k", _BAD_INT_SCALARS)
def test_rank_search_bad_int_k_raises_overflow(bad_k):
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(OverflowError):
        idx.search(unit_vectors(2, 64, seed=1), k=bad_k)


@pytest.mark.parametrize("bad_k", _BAD_INT_SCALARS)
def test_rankquant_search_asym_bad_int_k_raises_overflow(bad_k):
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(OverflowError):
        idx.search_asymmetric(unit_vectors(2, 64, seed=1), k=bad_k)


@pytest.mark.parametrize("bad_m", _BAD_INT_SCALARS)
def test_bitmap_top_m_bad_int_m_raises_overflow(bad_m):
    idx = Bitmap(dim=64, n_top=8)
    idx.add(unit_vectors(20, 64))
    with pytest.raises(OverflowError):
        idx.top_m_candidates(unit_vectors(1, 64, seed=1)[0], m=bad_m)


@pytest.mark.parametrize("bad", _BAD_INT_SCALARS)
def test_bitmap_chunked_bad_int_batch_size_raises_overflow(bad):
    idx = Bitmap(dim=64, n_top=8)
    idx.add(unit_vectors(20, 64))
    q = unit_vectors(2, 64, seed=1)
    with pytest.raises(OverflowError):
        idx.top_m_candidates_batched_chunked(q, m=5, batch_size=bad)


@pytest.mark.parametrize("bad_idx", _BAD_INT_SCALARS)
def test_swap_remove_bad_int_idx_raises_overflow(bad_idx):
    idx = Rank(dim=64)
    idx.add(unit_vectors(5, 64))
    with pytest.raises(OverflowError):
        idx.swap_remove(bad_idx)


@pytest.mark.parametrize("bad_k", _BAD_INT_SCALARS)
def test_subset_bad_int_k_raises_overflow(bad_k):
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    cand = np.array([0, 1, 2], dtype=np.uint32)
    with pytest.raises(OverflowError):
        idx.search_asymmetric_subset(unit_vectors(1, 64, seed=1)[0], cand, k=bad_k)


@pytest.mark.parametrize("huge_k", _HUGE_VALID_USIZE)
def test_rank_search_huge_valid_k_clamps_not_ooms(huge_k):
    # A huge-but-valid usize k must be CLAMPED to the index size — the result
    # has min(k, n) columns and is computed without eagerly allocating k slots.
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    scores, indices = idx.search(unit_vectors(2, 64, seed=1), k=huge_k)
    assert scores.shape == (2, 10)  # clamped to n=10
    assert indices.shape == (2, 10)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("huge_m", _HUGE_VALID_USIZE)
def test_bitmap_top_m_huge_valid_m_clamps_not_ooms(huge_m):
    idx = Bitmap(dim=64, n_top=8)
    idx.add(unit_vectors(10, 64))
    cands = idx.top_m_candidates(unit_vectors(1, 64, seed=1)[0], m=huge_m)
    assert cands.shape == (10,)  # m_eff = min(m, n)
    assert cands.dtype == np.uint32


def test_rank_search_k_zero_returns_empty_columns():
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    scores, indices = idx.search(unit_vectors(2, 64, seed=1), k=0)
    assert scores.shape == (2, 0)
    assert indices.shape == (2, 0)


# =====================================================================
# Integer-scalar dtype: numpy int scalars / bool must convert via __index__;
# a float scalar (even integral-valued) must be rejected as TypeError.
# =====================================================================


@pytest.mark.parametrize("k", [np.int64(3), np.uint64(3), np.int8(3), np.uint32(3), True])
def test_search_accepts_integer_scalar_k(k):
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    scores, indices = idx.search(unit_vectors(1, 64, seed=1), k=k)
    assert scores.shape == (1, int(k))


@pytest.mark.parametrize("k", [np.float32(3.0), np.float64(3.0), 3.0])
def test_search_rejects_float_scalar_k(k):
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(TypeError):
        idx.search(unit_vectors(1, 64, seed=1), k=k)


# =====================================================================
# dtype confusion: rust-numpy is strict — a wrong element dtype on any array
# param must be a clean TypeError (NOT a silent byte reinterpretation).
# =====================================================================


@pytest.mark.parametrize("dt", _WRONG_F32_DTYPES)
def test_rank_add_wrong_dtype_raises_type_error(dt):
    idx = Rank(dim=64)
    bad = np.ones((4, 64), dtype=dt)
    with pytest.raises(TypeError):
        idx.add(bad)


@pytest.mark.parametrize("dt", _NON_INTEGER_ID_DTYPES)
def test_subset_candidates_noninteger_dtype_raises_type_error(dt):
    # Candidate ids accept any *integer* dtype (converted to u32 by value, never
    # by byte reinterpretation — see test_input_guards.py); a non-integer dtype
    # must be a clean TypeError, not a silent truncation.
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    cand = np.array([0, 1, 2], dtype=dt)
    with pytest.raises(TypeError):
        idx.search_asymmetric_subset(unit_vectors(1, 64, seed=1)[0], cand, k=2)


@pytest.mark.parametrize("dt", [np.uint8, np.int8, np.int64, np.uint64])
def test_subset_candidates_integer_dtype_converted_by_value(dt):
    # Adversarial: a narrow/wide integer dtype is read as logical *values*, not
    # reinterpreted bytes. uint8 [1,2,3] -> ids 1,2,3, identical to uint32.
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    q = unit_vectors(1, 64, seed=1)[0]
    ref = np.array([1, 2, 3], dtype=np.uint32)
    s_ref, id_ref = idx.search_asymmetric_subset(q, ref, k=3)
    s, ids = idx.search_asymmetric_subset(q, ref.astype(dt), k=3)
    np.testing.assert_array_equal(ids, id_ref)
    np.testing.assert_array_equal(s, s_ref)


@pytest.mark.parametrize("dt", [np.uint32, np.int64, np.float64, np.uint8])
def test_body_overlap_q_bitmap_wrong_dtype_raises_type_error(dt):
    # q_bitmap must be uint64; a narrower/float dtype must be a clean TypeError.
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(10, 128))
    qb = idx.build_query_bitmap_fp32(unit_vectors(1, 128, seed=1)[0]).astype(dt)
    with pytest.raises(TypeError):
        idx.body_overlap_scores_subset(qb, np.array([0, 1], dtype=np.uint32))


@pytest.mark.parametrize("dt", _NON_INTEGER_ID_DTYPES)
def test_body_overlap_doc_ids_noninteger_dtype_raises_type_error(dt):
    # doc_ids accept any integer dtype (converted to u32 with checked bounds);
    # a non-integer dtype is a clean TypeError.
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(10, 128))
    qb = idx.build_query_bitmap_fp32(unit_vectors(1, 128, seed=1)[0])
    with pytest.raises(TypeError):
        idx.body_overlap_scores_subset(qb, np.array([0, 1, 2], dtype=dt))


def test_bucket_ranks_wrong_dtype_raises_type_error():
    # ranks must be uint16.
    with pytest.raises(TypeError):
        bucket_ranks(np.array([0, 1, 2, 3], dtype=np.int32), 2)


def test_pack_buckets_wrong_dtype_raises_type_error():
    with pytest.raises(TypeError):
        pack_buckets(np.array([0, 1, 2, 3], dtype=np.int8), 2)


# =====================================================================
# NaN encodings: signaling + quiet NaN bit patterns must both be rejected by
# the finite guard (f32::is_finite catches every NaN payload).
# =====================================================================


@pytest.mark.parametrize("nan_val", [_QNAN, _SNAN, _NEG_SNAN, math.nan])
def test_rank_add_all_nan_encodings_rejected(nan_val):
    assert not np.isfinite(nan_val)
    idx = Rank(dim=64)
    v = unit_vectors(4, 64)
    v[0, 0] = nan_val
    with pytest.raises(ValueError, match="finite"):
        idx.add(v)


@pytest.mark.parametrize("nan_val", [_QNAN, _SNAN, _NEG_SNAN])
def test_signbitmap_build_query_nan_encodings_rejected(nan_val):
    idx = SignBitmap(dim=64)
    q = unit_vectors(1, 64, seed=1)[0]
    q[3] = nan_val
    with pytest.raises(ValueError, match="finite"):
        idx.build_query_bitmap(q)


def test_rank_add_f32_extremes_are_accepted():
    # ±f32::MAX and the smallest subnormals are FINITE → must be accepted, not
    # rejected by the finite guard (regression guard against an over-eager check).
    big = np.finfo(np.float32).max
    tiny = np.finfo(np.float32).smallest_subnormal
    for fill in (big, -big, tiny, -tiny):
        idx = Rank(dim=64)
        v = unit_vectors(4, 64)
        v[0, :] = fill
        idx.add(v)  # must not raise
        assert len(idx) == 4


# =====================================================================
# Numeric-value correctness: sign threshold edge cases must match numpy exactly
# (a wrong-result bug would silently corrupt the sign-cosine candidate set).
# =====================================================================


def test_signbitmap_zero_and_neg_zero_set_no_bits():
    # bit j is set iff coord_j > 0. 0.0 and -0.0 are NOT > 0, so popcount == 0.
    idx = SignBitmap(dim=128)
    for fill in (0.0, -0.0):
        qb = idx.build_query_bitmap(np.full(128, fill, dtype=np.float32))
        popcount = sum(bin(int(w)).count("1") for w in qb)
        assert popcount == 0, f"fill={fill!r} set {popcount} bits, expected 0"


def test_signbitmap_subnormal_sets_all_bits():
    # The smallest positive subnormal IS > 0 → every bit set (matches numpy).
    idx = SignBitmap(dim=128)
    tiny = np.finfo(np.float32).smallest_subnormal
    qb = idx.build_query_bitmap(np.full(128, tiny, dtype=np.float32))
    popcount = sum(bin(int(w)).count("1") for w in qb)
    assert popcount == 128
    assert tiny > 0  # numpy agrees


def test_signbitmap_build_query_matches_numpy_sign():
    # General parity: bit set iff q[j] > 0, byte-for-byte against numpy.
    idx = SignBitmap(dim=128)
    q = (np.arange(128, dtype=np.float32) - 64.0)  # spans negative→positive→zero
    qb = idx.build_query_bitmap(q)
    popcount = sum(bin(int(w)).count("1") for w in qb)
    assert popcount == int((q > 0.0).sum())


def test_rank_all_equal_rows_tie_break_by_index():
    # Every coordinate equal → rank_transform ties broken by ascending index, so
    # a self-query still scores ~1.0 and the index column is the identity order.
    idx = Rank(dim=64)
    idx.add(np.ones((5, 64), dtype=np.float32))
    scores, indices = idx.search(np.ones((1, 64), dtype=np.float32), k=3)
    assert np.isfinite(scores).all()
    assert indices[0].tolist() == [0, 1, 2]


def test_rank_transform_all_equal_is_identity_permutation():
    out = rank_transform(np.ones(8, dtype=np.float32))
    np.testing.assert_array_equal(out, np.arange(8, dtype=np.uint16))


# =====================================================================
# Shape abuse: wrong ndim must be a clean TypeError (rust-numpy enforces ndim).
# =====================================================================


def test_rank_add_1d_where_2d_expected_raises_type_error():
    with pytest.raises(TypeError):
        Rank(dim=64).add(unit_vectors(1, 64)[0])  # 1-D into a 2-D param


def test_rank_search_1d_where_2d_expected_raises_type_error():
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(TypeError):
        idx.search(unit_vectors(1, 64)[0], k=3)  # 1-D query


def test_rank_add_0d_scalar_raises_type_error():
    with pytest.raises(TypeError):
        Rank(dim=64).add(np.float32(1.0))


def test_rank_add_3d_raises_type_error():
    with pytest.raises(TypeError):
        Rank(dim=64).add(np.zeros((2, 3, 64), dtype=np.float32))


def test_bitmap_top_m_2d_where_1d_expected_raises_type_error():
    idx = Bitmap(dim=64, n_top=8)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(TypeError):
        idx.top_m_candidates(unit_vectors(2, 64), m=5)  # 2-D into a 1-D param


def test_add_zero_rows_is_noop():
    idx = Rank(dim=64)
    idx.add(np.empty((0, 64), dtype=np.float32))
    assert len(idx) == 0


def test_add_zero_width_raises_value_error():
    # (64, 0): zero columns → width 0 != dim, caught by check_width.
    with pytest.raises(ValueError, match="dimension"):
        Rank(dim=64).add(np.empty((64, 0), dtype=np.float32))


# =====================================================================
# Non-contiguous views beyond the transpose/stride cases in test_input_guards:
# broadcast_to (0-stride), reversed, and a strided column-slice that has the
# RIGHT width but is non-contiguous — must hit the C-contiguity guard, never
# misread memory or return a wrong result.
# =====================================================================


def test_rank_add_broadcast_to_zero_stride_raises_value_error():
    base = unit_vectors(1, 64)[0]
    view = np.broadcast_to(base, (10, 64))  # 0-stride on axis 0, non-owning
    assert not view.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        Rank(dim=64).add(view)


def test_rank_add_reversed_view_raises_value_error():
    view = unit_vectors(4, 64)[::-1]
    assert not view.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        Rank(dim=64).add(view)


def test_rank_add_strided_columns_right_width_raises_value_error():
    # a[:, ::2] from a (4,128) array → shape (4,64): width MATCHES dim=64 but the
    # buffer is non-contiguous. Must be rejected on contiguity, not silently read.
    strided = unit_vectors(4, 128)[:, ::2]
    assert strided.shape == (4, 64)
    assert not strided.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        Rank(dim=64).add(strided)


def test_signbitmap_batched_fortran_order_raises_value_error():
    idx = SignBitmap(dim=64)
    idx.add(unit_vectors(10, 64))
    bad = np.asfortranarray(unit_vectors(4, 64))
    assert not bad.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        idx.top_m_candidates_batched(bad, m=5)


# =====================================================================
# Type confusion on non-array params: None / list / str must be a clean
# TypeError everywhere, including the search_asymmetric_byte_lut PyRef arg.
# =====================================================================


@pytest.mark.parametrize("bad_first", [None, [1, 2, 3], "rq", 42])
def test_byte_lut_wrong_index_type_raises_type_error(bad_first):
    q = unit_vectors(2, 64)
    with pytest.raises(TypeError):
        search_asymmetric_byte_lut(bad_first, q, k=3)


def test_byte_lut_rank_instead_of_rankquant_raises_type_error():
    # A Rank (wrong index type) where RankQuant is required → TypeError, not a
    # mis-cast that reads RankQuant fields off a Rank.
    rk = Rank(dim=64)
    rk.add(unit_vectors(10, 64))
    with pytest.raises(TypeError):
        search_asymmetric_byte_lut(rk, unit_vectors(2, 64), k=3)


@pytest.mark.parametrize("bad", [None, [[1.0] * 64] * 4, "hello"])
def test_rank_add_non_array_raises_type_error(bad):
    with pytest.raises(TypeError):
        Rank(dim=64).add(bad)


def test_subset_candidates_none_raises_type_error():
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    with pytest.raises(TypeError):
        idx.search_asymmetric_subset(unit_vectors(1, 64)[0], None, k=2)


@pytest.mark.parametrize(
    "ctor,args",
    [
        (Rank, (None,)),
        (Rank, (64.5,)),
        (RankQuant, (None, 2)),
        (RankQuant, (64, None)),
        (RankQuant, (64.0, 2)),
        (Bitmap, (64, None)),
        (SignBitmap, (None,)),
    ],
)
def test_constructors_reject_non_integer_args(ctor, args):
    with pytest.raises(TypeError):
        ctor(*args)


# =====================================================================
# Constructor domain edges: huge / out-of-range dim & n_top → clean ValueError
# (or OverflowError for >= 2**64), never a deferred panic.
# =====================================================================


def test_rank_dim_above_u16_value_error():
    with pytest.raises(ValueError, match=r"\[2, 65535\]"):
        Rank(dim=65_536)


def test_rank_dim_2pow63_value_error():
    # 2**63 fits usize but is > u16::MAX → ValueError (not OverflowError).
    with pytest.raises(ValueError, match=r"\[2, 65535\]"):
        Rank(dim=2**63)


def test_rank_dim_2pow64_overflow_error():
    with pytest.raises(OverflowError):
        Rank(dim=2**64)


def test_bitmap_huge_n_top_value_error():
    with pytest.raises(ValueError, match="n_top"):
        Bitmap(dim=64, n_top=2**63)


@pytest.mark.parametrize(
    "dim,bits,ok",
    [
        (64, 1, True),  # mult of 8 (codes_per_byte) — ok
        (60, 1, False),  # not a multiple of 8
        (2, 1, False),  # below the 8-divisor
        (4, 2, True),  # min RankQuant dim for bits=2 (mult of 4)
        (16, 4, True),  # mult of 16 (= 2^4)
        (8, 4, False),  # not a multiple of 16
    ],
)
def test_rankquant_dim_bits_divisor_domain(dim, bits, ok):
    if ok:
        idx = RankQuant(dim=dim, bits=bits)
        assert idx.dim == dim and idx.bits == bits
    else:
        with pytest.raises(ValueError, match="multiple"):
            RankQuant(dim=dim, bits=bits)


def test_rankquant_min_dim_search_is_finite_and_shaped():
    # The smallest valid RankQuant must still produce a correct, finite result.
    idx = RankQuant(dim=4, bits=2)
    idx.add(unit_vectors(5, 4))
    scores, indices = idx.search_asymmetric(unit_vectors(3, 4, seed=1), k=2)
    assert scores.shape == (3, 2)
    assert indices.shape == (3, 2)
    assert np.isfinite(scores).all()


# =====================================================================
# Module-primitive bits/d domain edges.
# =====================================================================


@pytest.mark.parametrize("bits", [8, 9, 255])
def test_rank_to_bucket_bits_above_7_value_error(bits):
    with pytest.raises(ValueError, match="bits"):
        rank_to_bucket(0, 1024, bits)


def test_rank_to_bucket_d_zero_value_error():
    with pytest.raises(ValueError, match="d must be"):
        rank_to_bucket(0, 0, 2)


def test_rank_to_bucket_rank_ge_d_value_error():
    # The core asserts rank < d (fail-loud, like the other bucket primitives);
    # the binding surfaces it as a clean ValueError, not a PanicException.
    with pytest.raises(ValueError, match="must be < d"):
        rank_to_bucket(8, 8, 2)


def test_bucket_ranks_out_of_range_value_error():
    # An entry >= len() would trip the core's `rank < d` assert; the binding
    # rejects it as a ValueError rather than letting it surface as a panic.
    with pytest.raises(ValueError, match="must be < d"):
        bucket_ranks(np.array([0, 5, 2, 3], dtype=np.uint16), 2)


@pytest.mark.parametrize("bits", [8, 255])
def test_bucket_centre_bits_above_7_value_error(bits):
    with pytest.raises(ValueError, match="bits"):
        bucket_centre(0, bits)


def test_bucket_centre_out_of_range_bucket_value_error():
    # bucket 128 at bits=7 is one past the [0, 128) alphabet.
    with pytest.raises(ValueError, match="out of range"):
        bucket_centre(128, 7)


@pytest.mark.parametrize("bits", [0, 3, 5, 6, 7])
def test_pack_buckets_non_124_bits_value_error(bits):
    with pytest.raises(ValueError, match="bits"):
        pack_buckets(np.zeros(8, dtype=np.uint8), bits)


def test_unpack_buckets_length_mismatch_value_error():
    with pytest.raises(ValueError, match="!= d"):
        unpack_buckets(np.array([0, 0], dtype=np.uint8), 5, 2)


def test_rank_transform_length_above_u16_value_error():
    with pytest.raises(ValueError, match="u16"):
        rank_transform(np.zeros(65_536, dtype=np.float32))


def test_rank_transform_exactly_u16_max_ok():
    out = rank_transform(np.zeros(65_535, dtype=np.float32))
    assert out.shape == (65_535,)
    assert out.dtype == np.uint16


def test_primitive_pure_math_huge_d_no_panic():
    # rank_norm / rankquant_bytes_per_vec are pure arithmetic on a usize; a huge
    # d must compute (or saturate) a value, never panic. Documents the surface.
    assert rank_norm(2**60) > 0.0
    assert rankquant_bytes_per_vec(2**40, 2) == (2**40) * 2 // 8
    assert rankquant_norm(1024, 2) > 0.0


# =====================================================================
# Empty / boundary state: search before add, empty candidate sets, m_eff at n=0.
# =====================================================================


def test_search_before_any_add_is_clean():
    # Each retrieval type must accept a search on an empty index and return a
    # (nq, 0) / (0,)-shaped result, never panic on the from_shape_vec reshape.
    q2 = unit_vectors(2, 64, seed=1)
    q1 = unit_vectors(1, 64, seed=1)[0]

    s, i = Rank(dim=64).search(q2, k=5)
    assert s.shape == (2, 0) and i.shape == (2, 0)

    s, i = RankQuant(dim=64, bits=2).search(q2, k=5)
    assert s.shape == (2, 0)
    s, i = RankQuant(dim=64, bits=2).search_asymmetric(q2, k=5)
    assert s.shape == (2, 0)

    s, i = Bitmap(dim=64, n_top=8).search(q2, k=5)
    assert s.shape == (2, 0)
    assert Bitmap(dim=64, n_top=8).top_m_candidates(q1, m=5).shape == (0,)


def test_subset_on_empty_index_oor_candidate_index_error():
    idx = RankQuant(dim=64, bits=2)  # n == 0 → every id is out of range
    cand = np.array([0], dtype=np.uint32)
    with pytest.raises(IndexError, match="out of range"):
        idx.search_asymmetric_subset(unit_vectors(1, 64, seed=1)[0], cand, k=2)


def test_subset_empty_candidates_is_clean():
    idx = RankQuant(dim=64, bits=2)
    idx.add(unit_vectors(10, 64))
    scores, ids = idx.search_asymmetric_subset(
        unit_vectors(1, 64, seed=1)[0], np.array([], dtype=np.uint32), k=2
    )
    assert scores.shape == (0,) and ids.shape == (0,)


def test_body_overlap_empty_doc_ids_is_clean():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(10, 128))
    qb = idx.build_query_bitmap_fp32(unit_vectors(1, 128, seed=1)[0])
    out = idx.body_overlap_scores_subset(qb, np.array([], dtype=np.uint32))
    assert out.shape == (0,)


def test_body_overlap_u32_max_doc_id_index_error():
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(10, 128))
    qb = idx.build_query_bitmap_fp32(unit_vectors(1, 128, seed=1)[0])
    with pytest.raises(IndexError, match="out of range"):
        idx.body_overlap_scores_subset(qb, np.array([2**32 - 1], dtype=np.uint32))


def test_body_overlap_sorted_duplicate_doc_ids_accepted():
    # Sorted ascending allows EQUAL adjacent ids (w[0] > w[1] is false for dups),
    # and equal ids must score identically (correctness check).
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(10, 128))
    qb = idx.build_query_bitmap_fp32(unit_vectors(1, 128, seed=1)[0])
    scores = idx.body_overlap_scores_subset(qb, np.array([1, 1, 2], dtype=np.uint32))
    assert scores.shape == (3,)
    assert int(scores[0]) == int(scores[1])  # same id → same score


# =====================================================================
# The m_eff flatten invariant under adversarial m (the debug_assert_eq! in the
# batched flatten is compiled out in --release; if a core row width ever != m_eff
# the from_shape_vec(...).expect(...) would panic). Sweep m around n on the
# DEBUG build here; the release build is swept separately via _run_isolated below.
# =====================================================================


@pytest.mark.parametrize("n", [0, 1, 7, 50])
@pytest.mark.parametrize("m", _HUGE_M)
def test_bitmap_batched_flatten_invariant_holds(n, m):
    idx = Bitmap(dim=64, n_top=8)
    if n:
        idx.add(unit_vectors(n, 64))
    out = idx.top_m_candidates_batched(unit_vectors(3, 64, seed=1), m=m)
    assert out.shape == (3, min(m, n))
    assert out.dtype == np.uint32


@pytest.mark.parametrize("n", [0, 1, 7, 50])
@pytest.mark.parametrize("m", _HUGE_M_MID)
def test_bitmap_chunked_flatten_invariant_holds(n, m):
    idx = Bitmap(dim=64, n_top=8)
    if n:
        idx.add(unit_vectors(n, 64))
    out = idx.top_m_candidates_batched_chunked(
        unit_vectors(3, 64, seed=1), m=m, batch_size=2
    )
    assert out.shape == (3, min(m, n))


@pytest.mark.parametrize("n", [0, 1, 7, 50])
@pytest.mark.parametrize("m", _HUGE_M_SIMPLE)
def test_signbitmap_batched_flatten_invariant_holds(n, m):
    idx = SignBitmap(dim=64)
    if n:
        idx.add(unit_vectors(n, 64))
    out = idx.top_m_candidates_batched(unit_vectors(3, 64, seed=1), m=m)
    assert out.shape == (3, min(m, n))


# =====================================================================
# Reentrancy: the documented PyO3 borrow-flag contract. A __index__ callback on
# an integer arg that re-enters a &mut self method (add / swap_remove) on the
# object a &self method (search / swap_remove) is already borrowing must raise a
# clean "Already borrowed" RuntimeError — NEVER a data race or panic.
# =====================================================================


def test_reentrant_add_during_search_k_conversion_is_blocked():
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))

    class ReentrantK:
        def __index__(self_inner):
            idx.add(unit_vectors(2, 64))  # &mut self re-entry while search holds &self
            return 3

    with pytest.raises(RuntimeError, match="[Bb]orrowed"):
        idx.search(unit_vectors(1, 64, seed=1), k=ReentrantK())
    assert len(idx) == 10  # the re-entrant mutation was cleanly blocked


def test_reentrant_add_during_swap_remove_idx_conversion_is_blocked():
    idx = Rank(dim=64)
    idx.add(unit_vectors(10, 64))

    class ReentrantIdx:
        def __index__(self_inner):
            idx.add(unit_vectors(2, 64))  # &mut self re-entry during swap_remove (&mut self)
            return 0

    with pytest.raises(RuntimeError, match="[Bb]orrowed"):
        idx.swap_remove(ReentrantIdx())
    assert len(idx) == 10


# =====================================================================
# Loader corruption: write a real index, then truncate / extend / forge / corrupt
# the file and confirm load() raises a clean IOError (== OSError), never a panic
# or a DoS allocation. NB: IOError is OSError in Python 3, so the loader's
# io::Error → pyo3 PyIOError surfaces as catchable OSError.
# =====================================================================


def _write_real_rank(path: str) -> bytes:
    idx = Rank(dim=128)
    idx.add(unit_vectors(20, 128))
    idx.write(path)
    with open(path, "rb") as f:
        return f.read()


def test_rank_load_header_only_truncated_io_error(tmp_path):
    data = _write_real_rank(str(tmp_path / "real.tvr"))
    p = str(tmp_path / "trunc.tvr")
    with open(p, "wb") as f:
        f.write(data[:13])  # header, zero payload
    with pytest.raises(IOError):
        Rank.load(p)


def test_rank_load_mid_payload_truncated_io_error(tmp_path):
    data = _write_real_rank(str(tmp_path / "real.tvr"))
    p = str(tmp_path / "half.tvr")
    with open(p, "wb") as f:
        f.write(data[: len(data) // 2])
    with pytest.raises(IOError):
        Rank.load(p)


def test_rank_load_trailing_bytes_io_error(tmp_path):
    # A structurally-valid file with extra trailing bytes is rejected (v1 has no
    # footer) — guards against record-smuggling past a smaller declared payload.
    data = _write_real_rank(str(tmp_path / "real.tvr"))
    p = str(tmp_path / "ext.tvr")
    with open(p, "wb") as f:
        f.write(data + b"\x00" * 64)
    with pytest.raises(IOError):
        Rank.load(p)


def test_rank_load_forged_huge_n_vectors_io_error_no_oom(tmp_path):
    # Forge n_vectors (bytes 9..13) to ~268M into a tiny file. The DoS-alloc
    # hypothesis: a naive loader allocates n_vectors*dim*2 up front. The loader
    # must reject (MAX_VECTORS / payload-mismatch) BEFORE allocating.
    data = bytearray(_write_real_rank(str(tmp_path / "real.tvr")))
    data[9:13] = struct.pack("<I", 0x0FFFFFFF)
    p = str(tmp_path / "forged.tvr")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError):
        Rank.load(p)


def test_rank_load_forged_huge_dim_io_error_no_oom(tmp_path):
    # Forge dim (bytes 5..9) to u16::MAX into a small file → declared payload
    # (n*dim*2) hugely exceeds the file; rejected by check_payload_matches_file
    # before any allocation.
    data = bytearray(_write_real_rank(str(tmp_path / "real.tvr")))
    data[5:9] = struct.pack("<I", 0xFFFF)
    p = str(tmp_path / "forged_dim.tvr")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError):
        Rank.load(p)


def test_rank_load_non_permutation_payload_io_error(tmp_path):
    # Zero the payload (every rank == 0) → valid shape but not a permutation of
    # [0, dim). The loader's per-row permutation check rejects it (a silently
    # wrong Spearman score would otherwise result).
    data = bytearray(_write_real_rank(str(tmp_path / "real.tvr")))
    for i in range(13, len(data)):
        data[i] = 0
    p = str(tmp_path / "nonperm.tvr")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError, match="permutation"):
        Rank.load(p)


def test_rank_load_wrong_magic_io_error(tmp_path):
    data = _write_real_rank(str(tmp_path / "real.tvr"))
    p = str(tmp_path / "magic.tvr")
    with open(p, "wb") as f:
        f.write(b"XXXX" + data[4:])
    with pytest.raises(IOError, match="magic"):
        Rank.load(p)


def test_rank_load_wrong_version_io_error(tmp_path):
    data = bytearray(_write_real_rank(str(tmp_path / "real.tvr")))
    data[4] = 99
    p = str(tmp_path / "ver.tvr")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError, match="version"):
        Rank.load(p)


def test_rank_load_zero_byte_file_io_error(tmp_path):
    p = str(tmp_path / "zero.tvr")
    open(p, "wb").close()
    with pytest.raises(IOError):
        Rank.load(p)


def test_rank_load_directory_io_error(tmp_path):
    with pytest.raises(IOError):
        Rank.load(str(tmp_path))


def test_rankquant_load_forged_bits_io_error(tmp_path):
    idx = RankQuant(dim=128, bits=2)
    idx.add(unit_vectors(20, 128))
    real = str(tmp_path / "real.tvrq")
    idx.write(real)
    data = bytearray(open(real, "rb").read())
    data[5] = 3  # bits byte → invalid {1,2,4} domain
    p = str(tmp_path / "bits3.tvrq")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError, match="bits"):
        RankQuant.load(p)


def test_rankquant_load_corrupt_composition_io_error(tmp_path):
    # Set the whole payload to 0xFF → every code is bucket 3 (b=2), violating the
    # constant-composition invariant the analytical norm depends on.
    idx = RankQuant(dim=128, bits=2)
    idx.add(unit_vectors(20, 128))
    real = str(tmp_path / "real.tvrq")
    idx.write(real)
    data = bytearray(open(real, "rb").read())
    for i in range(14, len(data)):
        data[i] = 0xFF
    p = str(tmp_path / "cc.tvrq")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError, match="composition"):
        RankQuant.load(p)


def test_bitmap_load_forged_n_top_io_error(tmp_path):
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(20, 128))
    real = str(tmp_path / "real.tvbm")
    idx.write(real)
    data = bytearray(open(real, "rb").read())
    data[9:13] = struct.pack("<I", 0)  # n_top = 0 is invalid
    p = str(tmp_path / "nt0.tvbm")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError, match="n_top"):
        Bitmap.load(p)


def test_bitmap_load_corrupt_popcount_io_error(tmp_path):
    # Zero the payload → each row popcount 0 != n_top, violating the per-row
    # popcount invariant the constant-weight bitmap-null / formal overlap model
    # assumes.
    idx = Bitmap(dim=128, n_top=32)
    idx.add(unit_vectors(20, 128))
    real = str(tmp_path / "real.tvbm")
    idx.write(real)
    data = bytearray(open(real, "rb").read())
    for i in range(17, len(data)):
        data[i] = 0
    p = str(tmp_path / "pop0.tvbm")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError, match="bits set"):
        Bitmap.load(p)


def test_signbitmap_load_forged_small_dim_io_error(tmp_path):
    idx = SignBitmap(dim=128)
    idx.add(unit_vectors(20, 128))
    real = str(tmp_path / "real.tvsb")
    idx.write(real)
    data = bytearray(open(real, "rb").read())
    data[5:9] = struct.pack("<I", 32)  # below the [64, MAX] range
    p = str(tmp_path / "d32.tvsb")
    with open(p, "wb") as f:
        f.write(bytes(data))
    with pytest.raises(IOError):
        SignBitmap.load(p)


def test_cross_format_load_wrong_magic_io_error(tmp_path):
    # Load a .tvr file through SignBitmap.load → wrong magic, clean IOError.
    real = str(tmp_path / "real.tvr")
    _write_real_rank(real)
    with pytest.raises(IOError, match="magic"):
        SignBitmap.load(real)


def test_load_nul_byte_path_io_error():
    # A path with an interior NUL cannot become a Rust CString → clean OSError,
    # not a panic. (write() guarded the same way.)
    with pytest.raises((IOError, ValueError)):
        Rank.load("/tmp/ordvec\x00evil.tvr")


def test_write_to_nonexistent_directory_io_error(tmp_path):
    idx = Rank(dim=64)
    idx.add(unit_vectors(3, 64))
    with pytest.raises(IOError):
        idx.write(str(tmp_path / "nonexistent_dir" / "idx.tvr"))


# =====================================================================
# Documented behavior: paths are forwarded UNMODIFIED — no `..` sanitisation.
# This is by-design (module + package docstring), not a vuln. The test confirms
# a `..` path behaves like an ordinary file op (round-trips) and does nothing
# *worse* than a normal write/read.
# =====================================================================


def test_dotdot_path_behaves_like_ordinary_file_op(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    # ../../ from a/b resolves back to tmp_path — an ordinary relative path.
    target = os.path.join(str(nested), "..", "..", "escaped.tvr")
    idx = Rank(dim=64)
    idx.add(unit_vectors(5, 64))
    idx.write(target)
    assert (tmp_path / "escaped.tvr").exists()  # normal fs resolution, nothing worse
    reloaded = Rank.load(target)
    assert len(reloaded) == 5


# =====================================================================
# Abort-class probes, isolated in a child process. A hard abort/segfault would
# kill the runner, so these run via _run_isolated and assert a clean exit; a
# crash surfaces as a non-zero/negative return code rather than a dead session.
# All of these are EXPECTED to finish cleanly (the probe catches the typed
# exception in-child and prints OK) — they are tripwires for a future regression
# into a real abort, not currently-failing cases.
# =====================================================================


def test_isolated_signaling_nan_add_no_abort():
    proc = _run_isolated(
        "v = uv(4, 64)\n"
        "v[0,0] = np.array([0x7FA00000], dtype=np.uint32).view(np.float32)[0]\n"
        "try:\n"
        "    Rank(64).add(v)\n"
        "    raise SystemExit('expected ValueError on signaling NaN')\n"
        "except ValueError:\n"
        "    pass\n"
    )
    _assert_clean_child(proc)


def test_isolated_forged_huge_dim_load_no_abort():
    proc = _run_isolated(
        "import struct, tempfile, os\n"
        "with tempfile.TemporaryDirectory() as td:\n"
        "    p = os.path.join(td, 'r.tvr')\n"
        "    idx = Rank(128); idx.add(uv(20, 128)); idx.write(p)\n"
        "    data = bytearray(open(p,'rb').read())\n"
        "    data[5:9] = struct.pack('<I', 0xFFFF)\n"  # huge dim
        "    fp = os.path.join(td, 'forged.tvr')\n"
        "    open(fp,'wb').write(bytes(data))\n"
        "    try:\n"
        "        Rank.load(fp)\n"
        "        raise SystemExit('expected IOError on forged dim')\n"
        "    except OSError:\n"
        "        pass\n"
    )
    _assert_clean_child(proc)


def test_isolated_huge_m_batched_no_abort_no_oom():
    # A huge m on a release build with debug_asserts off — the m_eff flatten must
    # not panic at .expect() and must not eagerly allocate that many slots. Use a
    # value that fits usize on the target (2**62 on 64-bit, 2**31 on 32-bit) so it
    # exercises the clamp rather than a PyO3 OverflowError.
    huge_m = 2**62 if sys.maxsize > 2**32 else 2**31
    proc = _run_isolated(
        "bm = Bitmap(64, 8); bm.add(uv(50, 64))\n"
        f"out = bm.top_m_candidates_batched(uv(3, 64), m={huge_m})\n"
        "assert out.shape == (3, 50), out.shape\n"
    )
    _assert_clean_child(proc)


def test_isolated_reentrant_borrow_no_abort():
    proc = _run_isolated(
        "idx = Rank(64); idx.add(uv(10, 64))\n"
        "class K:\n"
        "    def __index__(self):\n"
        "        idx.add(uv(2, 64)); return 3\n"
        "try:\n"
        "    idx.search(uv(1, 64), k=K())\n"
        "    raise SystemExit('expected Already borrowed')\n"
        "except RuntimeError:\n"
        "    pass\n"
        "assert len(idx) == 10\n"
    )
    _assert_clean_child(proc)
