"""Tests for the module-level rank-math primitives and limit constants.

These free functions mirror ``ordvec::rank::*``, the crate-root
``search_asymmetric_byte_lut``, and the ``ordvec::rank_io`` limit constants,
giving the Python package 1:1 parity with the Rust public surface. Algorithmic
correctness is proven in the crate's Rust tests; these cover the FFI boundary,
the numpy round-trips, and the argument guards (bad input → typed exception,
never a PanicException).
"""
from __future__ import annotations

import numpy as np
import pytest

import ordvec
from ordvec import (
    MAX_DIM,
    MAX_SIGN_BITMAP_DIM,
    MAX_VECTORS,
    RankQuant,
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


def test_rank_transform_matches_numpy_argsort_argsort():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(256).astype(np.float32)  # no ties w.p. 1
    r = rank_transform(v)
    assert r.dtype == np.uint16
    np.testing.assert_array_equal(r, np.argsort(np.argsort(v)).astype(np.uint16))


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


def test_bucket_ranks_agrees_with_scalar_to_bucket():
    ranks = np.arange(1024, dtype=np.uint16)
    bk = bucket_ranks(ranks, 2)
    assert bk.dtype == np.uint8
    for r in (0, 255, 256, 1023):
        assert int(bk[r]) == rank_to_bucket(int(ranks[r]), 1024, 2)


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


def test_search_asymmetric_byte_lut_self_retrieves_top1():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((40, 128)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    idx = RankQuant(dim=128, bits=2)
    idx.add(vectors)
    queries = vectors[:3]
    s_lut, i_lut = search_asymmetric_byte_lut(idx, queries, k=10)
    _, i_ref = idx.search_asymmetric(queries, k=10)
    assert s_lut.shape == (3, 10)
    # Both the byte-LUT and the production kernel are the asymmetric path, so a
    # self-query must self-rank at top-1 in both.
    for bi in range(3):
        assert int(i_lut[bi][0]) == bi
        assert int(i_ref[bi][0]) == bi


def test_search_asymmetric_byte_lut_rejects_b1():
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((10, 128)).astype(np.float32)
    idx = RankQuant(dim=128, bits=1)
    idx.add(vectors)
    with pytest.raises(ValueError, match="2, 4"):
        search_asymmetric_byte_lut(idx, vectors[:2], k=5)


def test_constants_exposed():
    assert MAX_DIM == 65535
    assert MAX_SIGN_BITMAP_DIM == (1 << 24)
    assert MAX_VECTORS == 64 * 1024 * 1024
    assert ordvec.MAX_DIM == MAX_DIM
