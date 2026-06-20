"""Tests for the module-level rank-math primitives and limit constants.

These free functions mirror ``ordvec::rank::*`` and the ``ordvec::rank_io``
limit constants. Algorithmic correctness is proven in the crate's Rust tests;
these cover the FFI boundary, the numpy round-trips, and the argument guards
(bad input → typed exception, never a PanicException).
"""
from __future__ import annotations

import numpy as np
import pytest

import ordvec
from ordvec import (
    MAX_DIM,
    MAX_SIGN_BITMAP_DIM,
    MAX_VECTORS,
    bucket_centre,
    bucket_ranks,
    pack_buckets,
    rank_norm,
    rank_to_bucket,
    rank_transform,
    rankquant_bytes_per_vec,
    rankquant_norm,
    unpack_buckets,
)


def test_rank_transform_matches_numpy_argsort_argsort():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(256).astype(np.float32)
    r = rank_transform(v)
    assert r.dtype == np.uint16
    # Match the core's "ties broken by index" contract with a stable inner
    # argsort, so the test is robust even if the float32 cast produces ties
    # (the outer argsort runs on a permutation, so its sort kind is irrelevant).
    np.testing.assert_array_equal(
        r, np.argsort(np.argsort(v, kind="stable")).astype(np.uint16)
    )


def test_rank_transform_breaks_ties_by_index():
    # Equal values must rank by ascending index (stable), matching the core.
    v = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(
        rank_transform(v), np.array([0, 1, 2, 3], dtype=np.uint16)
    )


def test_rank_transform_rejects_oversize():
    v = np.zeros(70_000, dtype=np.float32)  # > u16::MAX
    with pytest.raises(ValueError, match="u16"):
        rank_transform(v)


def test_rank_transform_rejects_nonfinite():
    v = np.array([1.0, np.nan, 2.0], dtype=np.float32)
    with pytest.raises(ValueError):
        rank_transform(v)


def test_pack_unpack_round_trip():
    buckets = np.array([i % 4 for i in range(16)], dtype=np.uint8)
    packed = pack_buckets(buckets, 2)
    assert packed.dtype == np.uint8
    assert packed.shape == (4,)  # 16 codes * 2 bits / 8
    out = unpack_buckets(packed, 16, 2)
    np.testing.assert_array_equal(out, buckets)


def test_pack_buckets_rejects_out_of_range_codes():
    # Bucket codes must be in [0, 1<<bits); the core would silently mask
    # (b & mask), so the binding rejects out-of-range codes with a ValueError.
    with pytest.raises(ValueError, match="out of range"):
        pack_buckets(np.array([7, 7, 7, 7], dtype=np.uint8), 2)


def test_bucket_ranks_agrees_with_scalar_to_bucket():
    ranks = np.arange(1024, dtype=np.uint16)
    bk = bucket_ranks(ranks, 2)
    assert bk.dtype == np.uint8
    for r in (0, 255, 256, 1023):
        assert int(bk[r]) == rank_to_bucket(int(ranks[r]), 1024, 2)


def test_bucket_ranks_empty_returns_empty():
    # Verified non-panic: the core maps over an empty slice and never calls
    # rank_to_bucket, so its `d > 0` assert is unreachable for empty input
    # (and non-empty input always has d >= 1). Pin the empty -> empty contract.
    out = bucket_ranks(np.array([], dtype=np.uint16), 2)
    assert out.dtype == np.uint8
    assert out.shape == (0,)


def test_rank_to_bucket_partitions_uniformly():
    counts = [0, 0, 0, 0]
    for r in range(1024):
        counts[rank_to_bucket(r, 1024, 2)] += 1
    assert counts == [256, 256, 256, 256]


def test_rank_to_bucket_large_d_does_not_panic():
    # d is a free usize arg from Python; a d above u32::MAX must not
    # divide-by-zero inside the core (fixed with u64 math upstream). It must
    # return a value, not raise a PanicException.
    assert rank_to_bucket(0, 2**40, 2) == 0
    assert rank_to_bucket(65535, 2**40, 2) < 4


def test_bucket_centre_symmetric():
    assert [bucket_centre(b, 2) for b in range(4)] == [-1.5, -0.5, 0.5, 1.5]


def test_bucket_centre_rejects_out_of_range_bucket():
    # bucket 4 at bits=2 is outside [0, 4). The core hard-asserts this in every
    # build; the binding must surface it as a clean ValueError, never a
    # PanicException (mirrors pack_buckets' out-of-range guard).
    with pytest.raises(ValueError, match="out of range"):
        bucket_centre(4, 2)


def test_rank_norm_matches_direct():
    d = 1024
    mean = (d - 1) / 2
    direct = float(np.sqrt(sum((i - mean) ** 2 for i in range(d))))
    assert abs(rank_norm(d) - direct) / direct < 1e-5


def test_rankquant_bytes_per_vec():
    assert rankquant_bytes_per_vec(1024, 2) == 1024 * 2 // 8
    assert rankquant_bytes_per_vec(1024, 4) == 1024 * 4 // 8


def test_rankquant_norm_positive():
    assert rankquant_norm(1024, 2) > 0.0


def test_primitive_bits_guards():
    with pytest.raises(ValueError, match="bits"):
        pack_buckets(np.zeros(8, dtype=np.uint8), 3)
    with pytest.raises(ValueError, match="bits"):
        rankquant_bytes_per_vec(1024, 3)
    with pytest.raises(ValueError, match="bits"):
        rank_to_bucket(0, 1024, 8)


def test_constants_exposed():
    assert MAX_DIM == 65535
    assert MAX_SIGN_BITMAP_DIM == (1 << 24)
    assert MAX_VECTORS == 64 * 1024 * 1024
    assert ordvec.MAX_DIM == MAX_DIM
