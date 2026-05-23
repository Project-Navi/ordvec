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
//! Provenance: original work by Nelson Spence, developed within turbovec
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
        let result = self.inner.top_m_candidates_batched(slice, m);
        // m_eff is the per-row width the Rust impl guarantees for every non-empty
        // row; deriving it from `m` and the index size keeps the shape consistent
        // at `batch=0`.
        let m_eff = m.min(self.inner.len());
        let mut flat: Vec<u32> = Vec::with_capacity(batch * m_eff);
        for row in &result {
            debug_assert_eq!(row.len(), m_eff);
            flat.extend_from_slice(row);
        }
        Ok(numpy::ndarray::Array2::from_shape_vec((batch, m_eff), flat)
            .expect("internal: batched candidate flatten shape invariant")
            .into_pyarray(py))
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

/// The native extension module backing the `ordvec` Python package.
#[pymodule]
fn _ordvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Rank>()?;
    m.add_class::<RankQuant>()?;
    m.add_class::<Bitmap>()?;
    m.add_class::<SignBitmap>()?;
    Ok(())
}
