//! Python bindings for [`ordvec`](https://github.com/Fieldnote-Echo/ordvec) — the
//! training-free ordinal & sign vector-quantization crate.
//!
//! Exposes the four retrieval types under the OrdVec ontology — [`Rank`],
//! [`RankQuant`], [`Bitmap`], [`SignBitmap`] — as a single abi3 extension module
//! (`_ordvec`) wrapped by the `ordvec` Python package.
//!
//! The core crate is aliased as `ordvec_core` throughout, so the Rust namespace
//! never collides with the `ordvec` Python package name.
//!
//! Provenance: developed within turbovec
//! (MIT, by Ryan Codrai), factored out. Dual-licensed MIT OR Apache-2.0.
//!
//! Every FFI entry point validates its inputs at the boundary so the core's
//! `assert!`/`assert_all_finite` panics surface as typed Python exceptions, not
//! an opaque `PanicException`: constructors and `swap_remove` check their
//! arguments, `check_width` rejects shape mismatches, `ensure_finite` rejects
//! NaN/±Inf, and the inline guard rejects non-C-contiguous arrays.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::wrap_pyfunction;

/// `(scores, indices)` returned by a batched top-k search — `(nq, k)`-shaped each.
type SearchArrays<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i64>>);
/// `(scores, ids)` returned by a single-query subset rerank — 1-D arrays.
type SubsetArrays<'py> = (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>);

/// Reject NaN/±Inf at the FFI boundary.
///
/// ordvec enforces a strict all-finite input policy in the core (`assert_all_finite`),
/// which would otherwise *panic* on a non-finite embedding and surface across pyo3 as
/// an opaque `PanicException`. Validating here turns that into a clean, typed
/// `ValueError`, mirroring the C-contiguity guard used on every array input.
fn ensure_finite(xs: &[f32]) -> PyResult<()> {
    if xs.iter().any(|x| !x.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "input contains NaN or infinity; ordvec requires all-finite f32 embeddings",
        ));
    }
    Ok(())
}

/// Reject an array whose row width (2-D `ncols`) or length (1-D) does not match
/// the index dimension.
///
/// The core derives `n = slice.len() / dim` and only asserts divisibility, so a
/// wrong-but-divisible width (e.g. `(1, 128)` into a dim-64 index) would be
/// silently reinterpreted as a different vector count — or panic when the result
/// is reshaped to `(nrows, k)` via `from_shape_vec(...).unwrap()`. Validate the
/// width up front so the caller gets a clean `ValueError`.
fn check_width(got: usize, dim: usize) -> PyResult<()> {
    if got != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "array width {got} does not match index dimension {dim}"
        )));
    }
    Ok(())
}

/// Reject a `bits` value outside the `{1, 2, 4}` packing domain (used by the
/// RankQuant pack/unpack/norm primitives) as a clean `ValueError` rather than
/// letting the core `assert!` surface as a `PanicException`.
fn check_bits_124(bits: u8) -> PyResult<()> {
    if !matches!(bits, 1 | 2 | 4) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bits must be 1, 2, or 4",
        ));
    }
    Ok(())
}

/// Reject a `bits` value the bucket primitives can't represent: `rank_to_bucket`
/// / `bucket_centre` cap at 7 so `1 << bits` fits the result and never overflows
/// the shift. Mirrors the core asserts as a typed `ValueError`.
fn check_bits_max7(bits: u8) -> PyResult<()> {
    if bits > 7 {
        return Err(pyo3::exceptions::PyValueError::new_err("bits must be <= 7"));
    }
    Ok(())
}

// =====================================================================
// Rank-mode retrieval bindings: Rank, RankQuant, Bitmap, SignBitmap.
//
// API mirror of the core types so the OrdVec/RankQuant paper's Python pipeline
// can call the Rust kernels directly and verify parity. Ported from the rank/sign
// PyO3 bindings, renamed to the OrdVec ontology; the turbovec-specific TurboQuant
// and IdMap types are intentionally absent.
// =====================================================================

#[pyclass]
struct Rank {
    inner: ordvec_core::Rank,
}

#[pymethods]
impl Rank {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        if !(2..=u16::MAX as usize).contains(&dim) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dim must be in [2, {}]",
                u16::MAX
            )));
        }
        Ok(Self {
            inner: ordvec_core::Rank::new(dim),
        })
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = vectors.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        self.inner.add(slice);
        Ok(())
    }

    /// Symmetric rank-cosine search: rank-transforms the query, scores against
    /// stored rank vectors via Spearman correlation.
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let arr = queries.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        let results = self.inner.search(slice, k);
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .unwrap()
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Asymmetric rank-cosine search: queries stay as L2-normalised FP32, stored
    /// documents are integer ranks.
    fn search_asymmetric<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let arr = queries.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        let results = self.inner.search_asymmetric(slice, k);
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .unwrap()
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Serialise the rank index to a `.tvr` file.
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a `Rank` index from a `.tvr` file previously written by [`Rank.write`].
    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = ordvec_core::Rank::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    /// Order is not preserved. Returns the old index of the moved vector.
    /// Raises `IndexError` if `idx >= len(self)`.
    fn swap_remove(&mut self, idx: usize) -> PyResult<usize> {
        let n = self.inner.len();
        if idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {idx} out of range (index holds {n} vectors)"
            )));
        }
        Ok(self.inner.swap_remove(idx))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn bytes_per_vec(&self) -> usize {
        self.inner.bytes_per_vec()
    }

    #[getter]
    fn byte_size(&self) -> usize {
        self.inner.byte_size()
    }
}

#[pyclass]
struct RankQuant {
    inner: ordvec_core::RankQuant,
}

#[pymethods]
impl RankQuant {
    /// Construct a RankQuant index at the given bit width.
    /// Supported `bits` ∈ {1, 2, 4}; `dim` must be a multiple of `8/bits` and
    /// `2^bits`, in `[2, u16::MAX]`.
    #[new]
    fn new(dim: usize, bits: u8) -> PyResult<Self> {
        if !matches!(bits, 1 | 2 | 4) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "bits must be 1, 2, or 4",
            ));
        }
        if !(2..=u16::MAX as usize).contains(&dim) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dim must be in [2, {}]",
                u16::MAX
            )));
        }
        // Mirror the core asserts: dim must be a multiple of both 8/bits (codes
        // per byte) and 2^bits (so every bucket receives equal rank entries —
        // what keeps the analytical rankquant_norm exact per document).
        let codes_per_byte = (8 / bits) as usize;
        let n_buckets = 1usize << bits;
        if !dim.is_multiple_of(codes_per_byte) || !dim.is_multiple_of(n_buckets) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dim {dim} must be a multiple of {} for bits = {bits}",
                codes_per_byte.max(n_buckets)
            )));
        }
        Ok(Self {
            inner: ordvec_core::RankQuant::new(dim, bits),
        })
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = vectors.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        self.inner.add(slice);
        Ok(())
    }

    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let arr = queries.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        let results = self.inner.search(slice, k);
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .unwrap()
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Asymmetric search via the AVX-512 / AVX2 / scalar dispatch path.
    fn search_asymmetric<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let arr = queries.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        let results = self.inner.search_asymmetric(slice, k);
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .unwrap()
            .into_pyarray(py);
        Ok((scores, indices))
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = ordvec_core::RankQuant::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    /// Raises `IndexError` if `idx >= len(self)`.
    fn swap_remove(&mut self, idx: usize) -> PyResult<usize> {
        let n = self.inner.len();
        if idx >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {idx} out of range (index holds {n} vectors)"
            )));
        }
        Ok(self.inner.swap_remove(idx))
    }

    /// Asymmetric scoring restricted to a candidate subset (e.g. the top-M
    /// shortlist from a [`Bitmap`] or [`SignBitmap`] probe). Returns
    /// ``(scores, global_ids)`` where ``global_ids`` are the original doc
    /// indices (mapped from the local candidate slot); slots that could not be
    /// filled are returned as ``-1``. Uses the same AVX-512 → AVX2 → scalar
    /// dispatch as ``search_asymmetric``.
    fn search_asymmetric_subset<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        candidates: PyReadonlyArray1<u32>,
        k: usize,
    ) -> PyResult<SubsetArrays<'py>> {
        let q = query.as_array();
        check_width(q.len(), self.inner.dim())?;
        let q_slice = q.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(q_slice)?;
        let c = candidates.as_array();
        let c_slice = c.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        // Validate every candidate id against the index size *before* calling the
        // core. The core gathers `self.packed[di * bpv ..]` for each id and only
        // `assert!`s the bound, so an out-of-range id would panic inside Rust and
        // surface across pyo3 as a `PanicException` that leaks the internal buffer
        // geometry. Reject it here as a typed `IndexError` instead.
        let n = self.inner.len();
        if let Some(&bad) = c_slice.iter().find(|&&di| (di as usize) >= n) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "candidate id {bad} out of range (index holds {n} vectors)"
            )));
        }
        let (scores, ids) = self.inner.search_asymmetric_subset(q_slice, c_slice, k);
        Ok((scores.into_pyarray(py), ids.into_pyarray(py)))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn bits(&self) -> u8 {
        self.inner.bits()
    }

    #[getter]
    fn bytes_per_vec(&self) -> usize {
        self.inner.bytes_per_vec()
    }

    #[getter]
    fn byte_size(&self) -> usize {
        self.inner.byte_size()
    }
}

#[pyclass]
struct Bitmap {
    inner: ordvec_core::Bitmap,
}

#[pymethods]
impl Bitmap {
    /// Construct a top-bucket bitmap index. `dim` must be a positive multiple of
    /// 64; `n_top` (how many coordinates per document are flagged "top", e.g.
    /// dim/4 for the b=2-equivalent top quarter) must satisfy `0 < n_top < dim`.
    #[new]
    fn new(dim: usize, n_top: usize) -> PyResult<Self> {
        if dim == 0 || !dim.is_multiple_of(64) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dim must be a positive multiple of 64",
            ));
        }
        // Bitmap rank-transforms documents (u16 ranks) and indexes the query
        // side by u16 coordinate id, so it shares Rank/RankQuant's u16 dim cap.
        // Mirror the core `Bitmap::new` guard here so a too-large dim is a clean
        // ValueError, not a deferred PanicException on the first add/search.
        if dim > ordvec_core::rank_io::MAX_DIM {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dim {dim} exceeds the maximum bitmap dimension {} (u16 rank invariant)",
                ordvec_core::rank_io::MAX_DIM
            )));
        }
        if n_top == 0 || n_top >= dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "n_top must satisfy 0 < n_top < dim (got n_top = {n_top}, dim = {dim})"
            )));
        }
        Ok(Self {
            inner: ordvec_core::Bitmap::new(dim, n_top),
        })
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = vectors.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        self.inner.add(slice);
        Ok(())
    }

    /// Bitmap-overlap search: returns the top-`k` doc indices by
    /// popcount(query_top AND doc_top).
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let arr = queries.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        let results = self.inner.search(slice, k);
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .unwrap()
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .unwrap()
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Return top-`m` candidate doc IDs for a single query as a 1-D `uint32`
    /// array. Used as the candidate generator for two-stage retrieval (bitmap
    /// probe → exact RankQuant rerank).
    fn top_m_candidates<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        m: usize,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let arr = query.as_array();
        check_width(arr.len(), self.inner.dim())?;
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        let cands = self.inner.top_m_candidates(slice, m);
        Ok(cands.into_pyarray(py))
    }

    /// Build the query-side top-`n_top` bitmap from an FP32 query, returned as a
    /// 1-D `uint64` array of `dim / 64` words (the doc-side packing). Pairs with
    /// [`Bitmap.body_overlap_scores_subset`] for staged rescoring.
    fn build_query_bitmap_fp32<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let arr = query.as_array();
        check_width(arr.len(), self.inner.dim())?;
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        Ok(self.inner.build_query_bitmap_fp32(slice).into_pyarray(py))
    }

    /// Batched candidate generation: stream the bitmap corpus once and return
    /// top-`m` candidate doc IDs for each query. `queries` is a 2-D `(batch,
    /// dim)` f32 array; returns a 2-D `uint32` array of shape `(batch, m_eff)`
    /// where `m_eff = min(m, len(index))`. The column count is `m_eff`
    /// regardless of `batch`, so an empty `(0, dim)` input returns `(0, m_eff)`.
    fn top_m_candidates_batched<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        m: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let arr = queries.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let batch = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        // Guard the core's internal `batch * n` (scores) and `batch * qpv`
        // (query bitmaps) allocations BEFORE the call: an overflow there wraps
        // and then indexes out of bounds (a panic), so convert it to a clean
        // ValueError up front. `n.max(qpv)` bounds both core buffers.
        let n = self.inner.len();
        let qpv = self.inner.dim() / 64;
        batch.checked_mul(n.max(qpv)).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("batch * index size overflows usize")
        })?;
        let result = self.inner.top_m_candidates_batched(slice, m);
        let m_eff = m.min(n);
        let total = batch.checked_mul(m_eff).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("result size (batch * m) overflows usize")
        })?;
        let mut flat: Vec<u32> = Vec::with_capacity(total);
        for row in &result {
            debug_assert_eq!(row.len(), m_eff);
            flat.extend_from_slice(row);
        }
        Ok(numpy::ndarray::Array2::from_shape_vec((batch, m_eff), flat)
            .expect("internal: bitmap batched candidate flatten shape invariant")
            .into_pyarray(py))
    }

    /// Chunked batched candidate generation: like
    /// [`Bitmap.top_m_candidates_batched`] but processes `queries` in groups of
    /// `batch_size` rows in parallel — use when the full query workload is
    /// larger than one batch fits efficiently in cache. `queries` is a 2-D `(n,
    /// dim)` f32 array; returns a 2-D `uint32` array `(n, m_eff)`. `batch_size`
    /// must be > 0.
    fn top_m_candidates_batched_chunked<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        m: usize,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be > 0",
            ));
        }
        let arr = queries.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let n_queries = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        // Clamp batch_size to the query count so a very large value can't
        // overflow `batch_size * dim` inside the core (which fails loud with an
        // overflow panic). A batch larger than the workload is just one chunk,
        // so this is result-transparent — consistent with how the core clamps
        // `k`/`m`. The `.max(1)` keeps the core's `batch_size > 0` invariant
        // when there are no queries (the core then early-returns empty).
        let effective_batch = batch_size.min(n_queries.max(1));
        // Guard the core's per-chunk `effective_batch * n` / `* qpv` allocations
        // BEFORE the call (the chunked path calls top_m_candidates_batched once
        // per chunk), so an overflow is a clean ValueError rather than a
        // wrap -> OOB panic inside the chunk scan. `n.max(qpv)` bounds both.
        let n = self.inner.len();
        let qpv = self.inner.dim() / 64;
        effective_batch.checked_mul(n.max(qpv)).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("batch_size * index size overflows usize")
        })?;
        let result = self
            .inner
            .top_m_candidates_batched_chunked(slice, m, effective_batch);
        let m_eff = m.min(n);
        let total = n_queries.checked_mul(m_eff).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("result size (n_queries * m) overflows usize")
        })?;
        let mut flat: Vec<u32> = Vec::with_capacity(total);
        for row in &result {
            debug_assert_eq!(row.len(), m_eff);
            flat.extend_from_slice(row);
        }
        Ok(
            numpy::ndarray::Array2::from_shape_vec((n_queries, m_eff), flat)
                .expect("internal: bitmap chunked candidate flatten shape invariant")
                .into_pyarray(py),
        )
    }

    /// Compute bitmap-overlap scores for a subset of doc IDs against a pre-built
    /// query bitmap. `q_bitmap` is a 1-D `uint64` array of `dim / 64` words
    /// (e.g. from [`Bitmap.build_query_bitmap_fp32`]); `doc_ids` is a 1-D
    /// `uint32` array that must be in range. Returns a 1-D `uint32` array of
    /// overlap scores aligned to `doc_ids`.
    ///
    /// `doc_ids` must additionally be sorted ascending. This is a *Python-side
    /// ergonomic policy*, not a core requirement: the Rust core accepts unsorted
    /// ids and scores them correctly in input order, just with worse cache
    /// locality. The binding requires the sorted (cache-friendly) form so that
    /// is the only path Python callers take — pass `np.sort(doc_ids)` if your
    /// survivor set is unordered.
    fn body_overlap_scores_subset<'py>(
        &self,
        py: Python<'py>,
        q_bitmap: PyReadonlyArray1<u64>,
        doc_ids: PyReadonlyArray1<u32>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let qb = q_bitmap.as_array();
        let qb_slice = qb.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let qpv = self.inner.dim() / 64;
        if qb_slice.len() != qpv {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q_bitmap length {} does not match dim/64 = {qpv}",
                qb_slice.len()
            )));
        }
        let ids = doc_ids.as_array();
        let ids_slice = ids.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        // Bound-check every id before dispatch: the core hard-asserts ids are in
        // range (the AVX-512 path issues a raw load), so an OOB id would surface
        // as a PanicException. Reject it as a typed IndexError instead.
        let n = self.inner.len();
        if let Some(&bad) = ids_slice.iter().find(|&&di| (di as usize) >= n) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "doc id {bad} out of range (index holds {n} vectors)"
            )));
        }
        // Python-side ergonomic policy (NOT a core correctness requirement):
        // the Rust core scores unsorted ids correctly in input order, just with
        // worse cache locality. The binding requires the sorted, cache-friendly
        // form and returns a clean ValueError rather than silently running the
        // slow path.
        if ids_slice.windows(2).any(|w| w[0] > w[1]) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "doc_ids must be sorted in ascending order",
            ));
        }
        let mut out = vec![0u32; ids_slice.len()];
        self.inner
            .body_overlap_scores_subset(qb_slice, ids_slice, &mut out);
        Ok(out.into_pyarray(py))
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = ordvec_core::Bitmap::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn n_top(&self) -> usize {
        self.inner.n_top()
    }

    #[getter]
    fn bytes_per_vec(&self) -> usize {
        self.inner.bytes_per_vec()
    }

    #[getter]
    fn byte_size(&self) -> usize {
        self.inner.byte_size()
    }
}

#[pyclass]
struct SignBitmap {
    inner: ordvec_core::SignBitmap,
}

#[pymethods]
impl SignBitmap {
    /// 1-bit-per-coord sign-cosine retrieval substrate. `dim` must be a positive
    /// multiple of 64, at most `MAX_SIGN_BITMAP_DIM`. No `n_top` parameter — the
    /// threshold is data-independent (bit set iff coord > 0). Storage: `dim / 8`
    /// bytes per doc (128 B at D = 1024).
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        if dim == 0 || !dim.is_multiple_of(64) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dim must be a positive multiple of 64",
            ));
        }
        if dim > ordvec_core::rank_io::MAX_SIGN_BITMAP_DIM {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dim {dim} exceeds the maximum sign-bitmap dimension {}",
                ordvec_core::rank_io::MAX_SIGN_BITMAP_DIM
            )));
        }
        Ok(Self {
            inner: ordvec_core::SignBitmap::new(dim),
        })
    }

    fn add(&mut self, vectors: PyReadonlyArray2<f32>) -> PyResult<()> {
        let arr = vectors.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        self.inner.add(slice);
        Ok(())
    }

    /// Top-`m` candidate doc IDs for a single query, ranked by ascending Hamming
    /// distance (= descending sign agreement). Returns a 1-D `uint32` array of
    /// length `min(m, len(index))`.
    fn top_m_candidates<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
        m: usize,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let arr = query.as_array();
        check_width(arr.len(), self.inner.dim())?;
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        let cands = self.inner.top_m_candidates(slice, m);
        Ok(cands.into_pyarray(py))
    }

    /// Batched: streams the sign-bitmap corpus once per CHUNK=8 queries via the
    /// AVX-512 VPOPCNTDQ XOR-popcount kernel. `queries` is a flat `(batch, dim)`
    /// array. Returns a 2-D `uint32` array of shape `(batch, m_eff)` where
    /// `m_eff = min(m, len(index))`. The second dimension is `m_eff` regardless of
    /// `batch` — an empty queries array (`batch=0`) returns shape `(0, m_eff)`,
    /// not `(0, 0)`, so callers get a consistent column count across batched calls.
    fn top_m_candidates_batched<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        m: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let arr = queries.as_array();
        check_width(arr.ncols(), self.inner.dim())?;
        let batch = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        // Guard the core's internal `batch * n` (scores) and `batch * qpv`
        // (query bitmaps) allocations BEFORE the call: an overflow there wraps
        // and then indexes out of bounds (a panic), so convert it to a clean
        // ValueError up front. `n.max(qpv)` bounds both core buffers.
        let n = self.inner.len();
        let qpv = self.inner.dim() / 64;
        batch.checked_mul(n.max(qpv)).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("batch * index size overflows usize")
        })?;
        let result = self.inner.top_m_candidates_batched(slice, m);
        // m_eff is the per-row width the Rust impl guarantees for every non-empty
        // row; deriving it from `m` and the index size keeps the shape consistent
        // at `batch=0`.
        let m_eff = m.min(n);
        let total = batch.checked_mul(m_eff).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("result size (batch * m) overflows usize")
        })?;
        let mut flat: Vec<u32> = Vec::with_capacity(total);
        for row in &result {
            debug_assert_eq!(row.len(), m_eff);
            flat.extend_from_slice(row);
        }
        Ok(numpy::ndarray::Array2::from_shape_vec((batch, m_eff), flat)
            .expect("internal: batched candidate flatten shape invariant")
            .into_pyarray(py))
    }

    /// Build the query-side sign bitmap from an FP32 query, returned as a 1-D
    /// `uint64` array of `dim / 64` words (`bit j` set iff `q[j] > 0`).
    fn build_query_bitmap<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray1<f32>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let arr = query.as_array();
        check_width(arr.len(), self.inner.dim())?;
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        ensure_finite(slice)?;
        Ok(self.inner.build_query_bitmap(slice).into_pyarray(py))
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Persist the sign-bitmap payload to a `.tvsb` file. Format: 13-byte header
    /// (`TVSB` magic + version + dim + n_vectors) + LE u64 bitmaps.
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a `SignBitmap` from a `.tvsb` file previously written by
    /// [`SignBitmap.write`]. Raises `IOError` if the file is missing, malformed,
    /// or its payload length disagrees with the header-declared shape.
    #[classmethod]
    fn load(_cls: &Bound<PyType>, path: &str) -> PyResult<Self> {
        let inner = ordvec_core::SignBitmap::load(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    #[getter]
    fn bytes_per_vec(&self) -> usize {
        self.inner.bytes_per_vec()
    }

    #[getter]
    fn byte_size(&self) -> usize {
        self.inner.byte_size()
    }
}

// =====================================================================
// Module-level rank-math primitives.
//
// The four classes above give object-level parity with the Rust API; these
// free functions expose the `ordvec::rank` math primitives (the data-oblivious
// kernels the OrdVec/RankQuant paper's Python pipeline verifies against numpy)
// and the byte-LUT scoring path, so the crate's `pub` surface is fully
// reachable from Python. Each mirrors the core's argument asserts as a typed
// `ValueError` instead of letting them surface as a `PanicException`.
// =====================================================================

/// Dimension-wise rank transform: `out[k]` = rank of `v[k]` among `v` (ties
/// broken by index), equivalent to numpy `argsort(argsort(v))`. Returns a 1-D
/// `uint16` array; `len(v)` must be <= 65535 (the u16 rank invariant).
#[pyfunction]
fn rank_transform<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<f32>,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let arr = v.as_array();
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    if slice.len() > u16::MAX as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "length {} exceeds u16::MAX ({})",
            slice.len(),
            u16::MAX
        )));
    }
    ensure_finite(slice)?;
    Ok(ordvec_core::rank::rank_transform(slice).into_pyarray(py))
}

/// Bucket a single rank into one of `1 << bits` equal-width bins on `[0, d)`.
#[pyfunction]
fn rank_to_bucket(rank: u16, d: usize, bits: u8) -> PyResult<u8> {
    check_bits_max7(bits)?;
    if d == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("d must be > 0"));
    }
    Ok(ordvec_core::rank::rank_to_bucket(rank, d, bits))
}

/// Bucket every entry of a rank vector. Returns a 1-D `uint8` array.
#[pyfunction]
fn bucket_ranks<'py>(
    py: Python<'py>,
    ranks: PyReadonlyArray1<u16>,
    bits: u8,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    check_bits_max7(bits)?;
    let arr = ranks.as_array();
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    // Empty input maps to empty output. The core `bucket_ranks` already returns
    // an empty Vec here (its `map` never invokes `rank_to_bucket`, so the latter's
    // `d > 0` assert is unreachable on empty input), but make the empty -> empty
    // contract explicit at the boundary: the d == 0 case never reaches the core,
    // so the no-panic guarantee is local and obvious rather than relying on the
    // core's iterator short-circuit.
    if slice.is_empty() {
        return Ok(Vec::<u8>::new().into_pyarray(py));
    }
    Ok(ordvec_core::rank::bucket_ranks(slice, bits).into_pyarray(py))
}

/// Pack bucket indices (each in `[0, 1 << bits)`) into a dense byte stream.
/// `bits` ∈ {1, 2, 4}; `len(buckets)` must be a multiple of `8 / bits`.
#[pyfunction]
fn pack_buckets<'py>(
    py: Python<'py>,
    buckets: PyReadonlyArray1<u8>,
    bits: u8,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    check_bits_124(bits)?;
    let arr = buckets.as_array();
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    let codes_per_byte = (8 / bits) as usize;
    if !slice.len().is_multiple_of(codes_per_byte) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "len {} must be a multiple of {codes_per_byte} for bits = {bits}",
            slice.len()
        )));
    }
    // Reject out-of-range bucket codes rather than silently masking them: the
    // core packs `b & ((1 << bits) - 1)`, so a value with high bits set would be
    // truncated to a different bucket. The bucket alphabet is [0, 1 << bits).
    let max_code = (1u16 << bits) - 1;
    if let Some(&bad) = slice.iter().find(|&&b| b as u16 > max_code) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "bucket value {bad} out of range [0, {}) for bits = {bits}",
            1u16 << bits
        )));
    }
    Ok(ordvec_core::rank::pack_buckets(slice, bits).into_pyarray(py))
}

/// Unpack a `bits`-bit packed byte stream into `d` bucket indices (inverse of
/// `pack_buckets`). Returns a 1-D `uint8` array.
#[pyfunction]
fn unpack_buckets<'py>(
    py: Python<'py>,
    packed: PyReadonlyArray1<u8>,
    d: usize,
    bits: u8,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    check_bits_124(bits)?;
    let arr = packed.as_array();
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    let codes_per_byte = (8 / bits) as usize;
    if slice.len() * codes_per_byte != d {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "packed length {} * {codes_per_byte} codes/byte != d = {d}",
            slice.len()
        )));
    }
    Ok(ordvec_core::rank::unpack_buckets(slice, d, bits).into_pyarray(py))
}

/// Bytes per packed RankQuant document at dimension `d` and bit width `bits`.
#[pyfunction]
fn rankquant_bytes_per_vec(d: usize, bits: u8) -> PyResult<usize> {
    check_bits_124(bits)?;
    let codes_per_byte = (8 / bits) as usize;
    if !d.is_multiple_of(codes_per_byte) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "d {d} must be a multiple of {codes_per_byte} for bits = {bits}"
        )));
    }
    Ok(ordvec_core::rank::rankquant_bytes_per_vec(d, bits))
}

/// Mean-centred value of a bucket index for a `bits`-bit RankQuant scheme.
#[pyfunction]
fn bucket_centre(bucket: u8, bits: u8) -> PyResult<f32> {
    check_bits_max7(bits)?;
    // Mirror the core's bucket-range guard as a typed ValueError. The core
    // hard-asserts `bucket < 1 << bits` in every build, so without this
    // pre-check a Python caller would get a PanicException instead of a clean
    // error. Matches the analogous out-of-range guard in `pack_buckets`.
    let n_buckets = 1u16 << bits;
    if (bucket as u16) >= n_buckets {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "bucket {bucket} out of range [0, {n_buckets}) for bits = {bits}"
        )));
    }
    Ok(ordvec_core::rank::bucket_centre(bucket, bits))
}

/// Analytical L2 norm of a mean-centred rank vector of length `d`:
/// `sqrt(d * (d^2 - 1) / 12)`.
#[pyfunction]
fn rank_norm(d: usize) -> f32 {
    ordvec_core::rank::rank_norm(d)
}

/// Analytical L2 norm of a mean-centred `bits`-bit RankQuant vector of length
/// `d` (assumes uniform bucket composition, exact for permutation ranks).
#[pyfunction]
fn rankquant_norm(d: usize, bits: u8) -> PyResult<f32> {
    check_bits_124(bits)?;
    Ok(ordvec_core::rank::rankquant_norm(d, bits))
}

/// Asymmetric search via the byte-LUT scoring path (a benchmark/parity helper;
/// requires `bits ∈ {2, 4}`). Returns `(scores, indices)` matching
/// `RankQuant.search_asymmetric`.
#[pyfunction]
fn search_asymmetric_byte_lut<'py>(
    py: Python<'py>,
    index: PyRef<'_, RankQuant>,
    queries: PyReadonlyArray2<f32>,
    k: usize,
) -> PyResult<SearchArrays<'py>> {
    if index.inner.bits() == 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "search_asymmetric_byte_lut requires bits in {2, 4}; use RankQuant.search_asymmetric for b=1",
        ));
    }
    let arr = queries.as_array();
    check_width(arr.ncols(), index.inner.dim())?;
    let nq = arr.nrows();
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    ensure_finite(slice)?;
    let results = ordvec_core::search_asymmetric_byte_lut(&index.inner, slice, k);
    let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
        .unwrap()
        .into_pyarray(py);
    let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
        .unwrap()
        .into_pyarray(py);
    Ok((scores, indices))
}

/// The native extension module backing the `ordvec` Python package.
#[pymodule]
fn _ordvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Rank>()?;
    m.add_class::<RankQuant>()?;
    m.add_class::<Bitmap>()?;
    m.add_class::<SignBitmap>()?;

    // Module-level rank-math primitives (parity with `ordvec::rank::*` and the
    // crate-root `search_asymmetric_byte_lut`).
    m.add_function(wrap_pyfunction!(rank_transform, m)?)?;
    m.add_function(wrap_pyfunction!(rank_to_bucket, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_ranks, m)?)?;
    m.add_function(wrap_pyfunction!(pack_buckets, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_buckets, m)?)?;
    m.add_function(wrap_pyfunction!(rankquant_bytes_per_vec, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_centre, m)?)?;
    m.add_function(wrap_pyfunction!(rank_norm, m)?)?;
    m.add_function(wrap_pyfunction!(rankquant_norm, m)?)?;
    m.add_function(wrap_pyfunction!(search_asymmetric_byte_lut, m)?)?;

    // Loader/limit constants (parity with `ordvec::rank_io::*`).
    m.add("MAX_DIM", ordvec_core::rank_io::MAX_DIM)?;
    m.add(
        "MAX_SIGN_BITMAP_DIM",
        ordvec_core::rank_io::MAX_SIGN_BITMAP_DIM,
    )?;
    m.add("MAX_VECTORS", ordvec_core::rank_io::MAX_VECTORS)?;
    Ok(())
}
