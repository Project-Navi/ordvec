//! Python bindings for [`ordvec`](https://github.com/Fieldnote-Echo/ordvec) — the
//! training-free ordinal & sign vector-quantization crate.
//!
//! Exposes the four retrieval types under the OrdVec ontology — `Rank`,
//! `RankQuant`, `Bitmap`, `SignBitmap` — as a single abi3 extension module
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
//! NaN/±Inf, and most array inputs reject non-C-contiguous layouts. Candidate
//! and doc-id arrays are the exception: contiguous `uint32` arrays are borrowed
//! zero-copy, non-contiguous `uint32` arrays are copied directly, and other
//! integer dtypes are copied through the checked `u32` conversion path.
//!
//! File paths passed to `write` / `load` are forwarded to the filesystem
//! unmodified — there is no `..` / traversal sanitisation — so callers must
//! treat the path as trusted input (see the `ordvec` package docstring).
//!
//! Threading: the search / candidate / `add` methods release the GIL
//! (`py.detach`) around the Rust scan and read the input arrays *in place*
//! (the `PyReadonlyArray` keeps the buffer alive and blocks rust-numpy-mediated
//! writes for the call's duration, but a raw Python in-place mutation from
//! another thread is not tracked). So a caller must not mutate an input array
//! from another thread while such a call is in progress — the released GIL lets
//! the write race the read and may yield inconsistent results. This is the
//! usual contract for GIL-releasing numeric extensions (NumPy behaves the same
//! way).

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
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

/// Mirror the core `add` capacity guard before releasing the GIL and entering
/// Rust core asserts. Public Python `add()` methods should raise `ValueError`
/// for over-capacity input, not a pyo3 `PanicException`.
fn check_add_capacity(
    current: usize,
    adding: usize,
    elems_per_vec: usize,
    elem_size: usize,
) -> PyResult<()> {
    let new_n = current
        .checked_add(adding)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("n_vectors overflows usize"))?;
    let max = ordvec_core::rank_io::MAX_VECTORS;
    if new_n > max {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "index would exceed MAX_VECTORS ({max}); had {current}, adding {adding}"
        )));
    }
    let total_elems = new_n.checked_mul(elems_per_vec).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "index buffer length (n_vectors * elems_per_vec) overflows usize",
        )
    })?;
    total_elems.checked_mul(elem_size).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("index buffer byte size overflows usize")
    })?;
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

/// Eval-only RankQuant scoring supports non-byte-aligned widths but still needs
/// at least two buckets and a bucket alphabet representable by `u8`.
fn check_bits_1_7(bits: u8) -> PyResult<()> {
    if !(1..=7).contains(&bits) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "bits must be in 1..=7",
        ));
    }
    Ok(())
}

fn not_contiguous_err() -> PyErr {
    pyo3::exceptions::PyValueError::new_err(
        "array must be C-contiguous; call np.ascontiguousarray() first",
    )
}

/// Candidate / doc-id slice obtained from a NumPy array, either borrowed
/// zero-copy (already `uint32` and contiguous) or owned (converted from another
/// integer dtype). The `Borrowed` variant keeps the `PyReadonlyArray` guard
/// alive so its slice stays valid across a GIL-released `py.detach` call.
enum CandidateIds<'py> {
    Borrowed(PyReadonlyArray1<'py, u32>),
    Owned(Vec<u32>),
}

impl CandidateIds<'_> {
    fn as_slice(&self) -> PyResult<&[u32]> {
        match self {
            CandidateIds::Borrowed(ro) => ro.as_slice().map_err(|_| not_contiguous_err()),
            CandidateIds::Owned(v) => Ok(v),
        }
    }
}

/// Coerce a NumPy candidate/doc-id array of *any* integer dtype to `u32`.
///
/// The core takes `&[u32]` doc ids (the corpus is capped at `MAX_VECTORS = 2^26`,
/// well below `u32::MAX`), so the natural binding type is `PyReadonlyArray1<u32>`.
/// But rust-numpy matches that dtype *exactly*, while NumPy index arrays are
/// `int64` by default (`np.arange`, `np.where()[0]`, `np.array([...])`, fancy
/// indexing, `np.argpartition`). Requiring `uint32` made the most natural ways to
/// build a candidate set raise an opaque `TypeError`, even though ordvec's own
/// candidate generators (`top_m_candidates*`) already emit `uint32`.
///
/// We accept any integer dtype and convert with **checked** bounds: a negative id
/// or one exceeding `u32::MAX` is a clean `ValueError`, never a silent wrap — note
/// `np.asarray(x, dtype=uint32)` would wrap `-1 -> 4294967295` and `2**32 -> 0`,
/// which would then score the wrong document. Already-`uint32` contiguous arrays
/// are borrowed zero-copy; every other dtype is copied once (candidate shortlists
/// are small relative to the scan; large-M FFI is tracked in issue #11). The
/// in-range (`< n`) check stays with the caller, which knows the corpus size.
fn as_u32_ids_1d<'py>(arr: &Bound<'py, PyAny>, what: &str) -> PyResult<CandidateIds<'py>> {
    // Fast path: already uint32. Borrow if contiguous; otherwise copy without
    // unnecessary bounds checks because every u32 value already fits.
    if let Ok(a) = arr.cast::<PyArray1<u32>>() {
        let ro = a.readonly();
        if ro.as_slice().is_ok() {
            return Ok(CandidateIds::Borrowed(ro));
        }
        let out = ro.as_array().to_vec();
        return Ok(CandidateIds::Owned(out));
    }

    macro_rules! try_int_dtype {
        ($t:ty) => {
            if let Ok(a) = arr.cast::<PyArray1<$t>>() {
                let ro = a.readonly();
                let view = ro.as_array();
                let mut out = Vec::with_capacity(view.len());
                for &x in view.iter() {
                    out.push(u32::try_from(x).map_err(|_| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "{what} {x} is out of range for a u32 index \
                             (must be in 0..=4294967295)"
                        ))
                    })?);
                }
                return Ok(CandidateIds::Owned(out));
            }
        };
    }
    // Order is irrelevant since each downcast is an exact dtype match.
    try_int_dtype!(i64);
    try_int_dtype!(u64);
    try_int_dtype!(i32);
    try_int_dtype!(i16);
    try_int_dtype!(u16);
    try_int_dtype!(i8);
    try_int_dtype!(u8);

    let got = arr
        .getattr("dtype")
        .map(|d| d.to_string())
        .unwrap_or_else(|_| "a non-array object".to_owned());
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{what} must be a 1-D integer NumPy array with values in [0, 2**32 - 1]; got {got} \
         (ordvec stores candidate ids as u32 — boolean and floating-point arrays are rejected)"
    )))
}

/// Reject any id `>= n` (out of the corpus) as a typed `IndexError`. The core
/// hard-asserts ids are in range (an AVX-512 path issues a raw gather load), so an
/// out-of-range id would otherwise surface as a `PanicException` that leaks the
/// internal buffer geometry.
fn check_ids_in_range(ids: &[u32], n: usize, what: &str) -> PyResult<()> {
    if let Some(&bad) = ids.iter().find(|&&di| (di as usize) >= n) {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{what} {bad} out of range (index holds {n} vectors)"
        )));
    }
    Ok(())
}

fn f32_dtype_error(arr: &Bound<'_, PyAny>) -> PyErr {
    let got = arr
        .getattr("dtype")
        .map(|d| d.to_string())
        .unwrap_or_else(|_| "a non-array object".to_owned());
    pyo3::exceptions::PyTypeError::new_err(format!(
        "expected a floating-point NumPy array (float16/float32/float64), got {got}; ordvec \
         rank/sign-transforms real vectors and converts them to float32 at the boundary — \
         boolean, integer, complex, object, and string arrays are rejected (a {{0, 1}} or \
         narrow-integer vector rank-transforms to a degenerate index artefact, not a meaningful \
         ordinal signal; call .astype(np.float32) to opt in deliberately)"
    ))
}

fn not_contiguous_f32_err() -> PyErr {
    pyo3::exceptions::PyValueError::new_err(
        "expected a C-contiguous NumPy array; got non-contiguous input. Use \
         np.ascontiguousarray(x, dtype=np.float32) if you intend to make a copy.",
    )
}

/// Reject a non-`float32` input whose dtype isn't a float kind, or whose `ndim`
/// doesn't match. Error types mirror the strict-extraction contract: a bad dtype
/// or rank is a `TypeError`, ordered so the dtype message wins. Layout is checked
/// separately by [`require_c_contiguous`] *after* this (a `ValueError`).
fn gate_float_ndim(arr: &Bound<'_, PyAny>, ndim: usize) -> PyResult<()> {
    let kind = arr
        .getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<char>());
    if !matches!(kind, Ok('f')) {
        return Err(f32_dtype_error(arr));
    }
    let nd = arr.getattr("ndim").and_then(|n| n.extract::<usize>());
    if !matches!(nd, Ok(n) if n == ndim) {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "expected a {ndim}-D float array"
        )));
    }
    Ok(())
}

/// Reject a non-`C`-contiguous original array *before* any dtype coercion, so a
/// transposed/strided float64 can't be silently laundered into a contiguous
/// float32 (that hidden copy can dominate runtime / poison benchmarks — the copy
/// decision stays with the caller).
fn require_c_contiguous(arr: &Bound<'_, PyAny>) -> PyResult<()> {
    let contiguous = arr
        .getattr("flags")
        .and_then(|f| f.getattr("c_contiguous"))
        .and_then(|c| c.extract::<bool>())
        .unwrap_or(false);
    if contiguous {
        Ok(())
    } else {
        Err(not_contiguous_f32_err())
    }
}

/// Length of `arr`'s axis `axis` from its `shape` tuple, read as cheap metadata so
/// width can be validated *before* any coercion copy — rejecting a wrong-shaped
/// large float64 array must not first allocate its float32 twin.
fn axis_len(arr: &Bound<'_, PyAny>, axis: usize) -> PyResult<usize> {
    arr.getattr("shape")?.get_item(axis)?.extract::<usize>()
}

fn infer_float_2d_width(arr: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(a) = arr.cast::<PyArray2<f32>>() {
        return Ok(a.readonly().as_array().ncols());
    }
    gate_float_ndim(arr, 2)?;
    axis_len(arr, 1)
}

/// Present an embedding vector as a 1-D `float32` `PyReadonlyArray`, converting at
/// the boundary. The premise of ordvec is *float vector in → rank/sign transform*,
/// so float32 is the internal working dtype, not a contract the caller must
/// pre-satisfy: `float64` (the default for `np.array([...])` and most API
/// embeddings) and `float16` are coerced. The transforms that consume the floats
/// are order-only (rank transform, top-bucket bitmap) or sign-only, and `f64→f32`
/// rounding is *monotonic* — it can never reorder two coordinates, only collapse a
/// near-tie at the f32 floor, strictly less perturbation than the rank/bucket
/// quantisation already applied. The asymmetric-query LUT keeps the floats but
/// scores against f32-quantised docs, so sub-`f32` query precision is meaningless
/// there too.
///
/// Rejected (matching exception type): non-float dtype — bool / integer / complex /
/// object / string — and wrong `ndim` (`TypeError`); a width that doesn't match the
/// index dimension, or a non-`C`-contiguous original (`ValueError`) — both checked
/// on the original *before* coercion, so a wrong-shaped large array is never copied
/// just to be rejected. Bool and narrow integers are
/// *deliberately* rejected: a `{0, 1}` or few-valued vector rank-transforms to an
/// index-tie artefact, i.e. silent retrieval garbage. The all-finite check runs on
/// the post-coercion f32 (an `f64 > f32::MAX` rounds to `+inf` — caught here, not
/// silently indexed). Already-`float32` contiguous arrays are borrowed zero-copy.
fn as_f32_1d<'py>(
    arr: &Bound<'py, PyAny>,
    expected_len: Option<usize>,
) -> PyResult<PyReadonlyArray1<'py, f32>> {
    let ro = if let Ok(a) = arr.cast::<PyArray1<f32>>() {
        let ro = a.readonly();
        if let Some(dim) = expected_len {
            check_width(ro.as_array().len(), dim)?;
        }
        ro
    } else {
        gate_float_ndim(arr, 1)?;
        if let Some(dim) = expected_len {
            check_width(axis_len(arr, 0)?, dim)?;
        }
        require_c_contiguous(arr)?;
        arr.py()
            .import("numpy")?
            .getattr("ascontiguousarray")?
            .call1((arr, "float32"))?
            .cast::<PyArray1<f32>>()
            .map(|a| a.readonly())
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("expected a 1-D float array"))?
    };
    ensure_finite(
        ro.as_array()
            .as_slice()
            .ok_or_else(not_contiguous_f32_err)?,
    )?;
    Ok(ro)
}

/// 2-D `(n, dim)` counterpart of [`as_f32_1d`] for the `add` / batched-query paths.
/// Same contract; see [`as_f32_1d`] for the full rationale.
fn as_f32_2d<'py>(arr: &Bound<'py, PyAny>, dim: usize) -> PyResult<PyReadonlyArray2<'py, f32>> {
    let ro = if let Ok(a) = arr.cast::<PyArray2<f32>>() {
        let ro = a.readonly();
        check_width(ro.as_array().ncols(), dim)?;
        ro
    } else {
        gate_float_ndim(arr, 2)?;
        check_width(axis_len(arr, 1)?, dim)?;
        require_c_contiguous(arr)?;
        arr.py()
            .import("numpy")?
            .getattr("ascontiguousarray")?
            .call1((arr, "float32"))?
            .cast::<PyArray2<f32>>()
            .map(|a| a.readonly())
            .map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "expected a 2-D float array of shape (n, dim)",
                )
            })?
    };
    ensure_finite(
        ro.as_array()
            .as_slice()
            .ok_or_else(not_contiguous_f32_err)?,
    )?;
    Ok(ro)
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

    fn __repr__(&self) -> String {
        format!("Rank(dim={}, n={})", self.inner.dim(), self.inner.len())
    }

    fn add<'py>(&mut self, py: Python<'py>, vectors: &Bound<'py, PyAny>) -> PyResult<()> {
        let vectors = as_f32_2d(vectors, self.inner.dim())?;
        let arr = vectors.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        check_add_capacity(
            self.inner.len(),
            arr.nrows(),
            self.inner.dim(),
            std::mem::size_of::<u16>(),
        )?;
        // Release the GIL around the parallel rank-transform / pack so other
        // Python threads run during a bulk add. `slice` (`&[f32]`) and
        // `&mut self.inner` are both `Ungil`, so no pointer juggling is needed.
        //
        // SAFETY (detaching on a `&mut self` method): `detach` drops the GIL
        // but NOT the `&mut self` exclusive borrow — PyO3 holds this object's
        // runtime borrow flag for the whole call, so another thread that
        // re-acquires the GIL and tries to touch the SAME object gets a clean
        // `Already borrowed` RuntimeError, never concurrent mutation. Distinct
        // objects run freely, which is the point of releasing the GIL.
        py.detach(|| self.inner.add(slice));
        Ok(())
    }

    /// Symmetric rank-cosine search: rank-transforms the query, scores against
    /// stored rank vectors via Spearman correlation.
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let results = py.detach(|| self.inner.search(slice, k));
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Asymmetric rank-cosine search: queries stay as L2-normalised FP32, stored
    /// documents are integer ranks.
    fn search_asymmetric<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let results = py.detach(|| self.inner.search_asymmetric(slice, k));
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Serialise the rank index to a `.tvr` file.
    ///
    /// `path` is forwarded to the filesystem unmodified — no `..` / traversal
    /// sanitisation — so treat it as trusted input (see the module docstring).
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a `Rank` index from a `.tvr` file previously written by [`Rank::write`].
    ///
    /// `path` is forwarded to the filesystem unmodified — no `..` / traversal
    /// sanitisation — so treat it as trusted input (see the module docstring).
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

    fn __repr__(&self) -> String {
        format!(
            "RankQuant(dim={}, bits={}, n={})",
            self.inner.dim(),
            self.inner.bits(),
            self.inner.len()
        )
    }

    fn add<'py>(&mut self, py: Python<'py>, vectors: &Bound<'py, PyAny>) -> PyResult<()> {
        let vectors = as_f32_2d(vectors, self.inner.dim())?;
        let arr = vectors.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        check_add_capacity(
            self.inner.len(),
            arr.nrows(),
            self.inner.bytes_per_vec(),
            std::mem::size_of::<u8>(),
        )?;
        // Release the GIL around the parallel rank-transform / pack so other
        // Python threads run during a bulk add. `slice` (`&[f32]`) and
        // `&mut self.inner` are both `Ungil`, so no pointer juggling is needed.
        //
        // SAFETY (detaching on a `&mut self` method): `detach` drops the GIL
        // but NOT the `&mut self` exclusive borrow — PyO3 holds this object's
        // runtime borrow flag for the whole call, so another thread that
        // re-acquires the GIL and tries to touch the SAME object gets a clean
        // `Already borrowed` RuntimeError, never concurrent mutation. Distinct
        // objects run freely, which is the point of releasing the GIL.
        py.detach(|| self.inner.add(slice));
        Ok(())
    }

    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let results = py.detach(|| self.inner.search(slice, k));
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Asymmetric search via the AVX-512 / AVX2 / scalar dispatch path.
    fn search_asymmetric<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let results = py.detach(|| self.inner.search_asymmetric(slice, k));
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Serialise the quantised index to a `.tvrq` file.
    ///
    /// `path` is forwarded to the filesystem unmodified — no `..` / traversal
    /// sanitisation — so treat it as trusted input (see the module docstring).
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a `RankQuant` index from a `.tvrq` file written by [`RankQuant::write`].
    ///
    /// `path` is forwarded to the filesystem unmodified — no `..` / traversal
    /// sanitisation — so treat it as trusted input (see the module docstring).
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
    /// indices (mapped from the local candidate slot). ``k`` is capped to the
    /// candidate-list length; the subset path does not add sentinel padding.
    /// Uses the same AVX-512 → AVX2 → scalar dispatch as ``search_asymmetric``.
    ///
    /// ``candidates`` may be unsorted and may contain duplicates. Duplicate
    /// candidate IDs are scored as separate entries and can produce duplicate
    /// hits; callers that require unique row IDs should deduplicate before
    /// calling.
    ///
    /// If the shortlist came from [`Bitmap`], this is the exact RankQuant
    /// rerank stage over that survivor set; it does not itself apply or
    /// calibrate a bitmap overlap threshold.
    ///
    /// ``candidates`` may be a 1-D array of any integer dtype — the ``uint32``
    /// emitted by ``top_m_candidates``/``top_m_candidates_batched`` or a plain
    /// ``int64`` index array (``np.arange``, ``np.where(...)[0]``, fancy-index
    /// results). Ids are converted to ``uint32``; a negative id, one ``>= 2**32``,
    /// or one ``>= len(self)`` raises a ``ValueError``/``IndexError``.
    fn search_asymmetric_subset<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
        candidates: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<SubsetArrays<'py>> {
        let query = as_f32_1d(query, Some(self.inner.dim()))?;
        let q = query.as_array();
        let q_slice = q.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        // Accept candidate ids of any integer dtype (NumPy index arrays are int64
        // by default) and convert to the core's u32 with checked bounds, then
        // reject any id outside the corpus before dispatch.
        let cands = as_u32_ids_1d(candidates, "candidate id")?;
        let c_slice = cands.as_slice()?;
        check_ids_in_range(c_slice, self.inner.len(), "candidate id")?;
        let (scores, ids) = py.detach(|| self.inner.search_asymmetric_subset(q_slice, c_slice, k));
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
    /// Every stored document and query bitmap has exactly `n_top` active
    /// coordinates, matching the constant-weight overlap model.
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

    fn __repr__(&self) -> String {
        format!(
            "Bitmap(dim={}, n_top={}, n={})",
            self.inner.dim(),
            self.inner.n_top(),
            self.inner.len()
        )
    }

    fn add<'py>(&mut self, py: Python<'py>, vectors: &Bound<'py, PyAny>) -> PyResult<()> {
        let vectors = as_f32_2d(vectors, self.inner.dim())?;
        let arr = vectors.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        check_add_capacity(
            self.inner.len(),
            arr.nrows(),
            self.inner.dim() / 64,
            std::mem::size_of::<u64>(),
        )?;
        // Release the GIL around the parallel rank-transform / pack so other
        // Python threads run during a bulk add. `slice` (`&[f32]`) and
        // `&mut self.inner` are both `Ungil`, so no pointer juggling is needed.
        //
        // SAFETY (detaching on a `&mut self` method): `detach` drops the GIL
        // but NOT the `&mut self` exclusive borrow — PyO3 holds this object's
        // runtime borrow flag for the whole call, so another thread that
        // re-acquires the GIL and tries to touch the SAME object gets a clean
        // `Already borrowed` RuntimeError, never concurrent mutation. Distinct
        // objects run freely, which is the point of releasing the GIL.
        py.detach(|| self.inner.add(slice));
        Ok(())
    }

    /// Bitmap-overlap search: returns the top-`k` doc indices by
    /// popcount(query_top AND doc_top). This uses the overlap statistic from
    /// the finite threshold theorem but returns a top-k ranking, not a
    /// calibrated threshold-admission set.
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<SearchArrays<'py>> {
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let nq = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let results = py.detach(|| self.inner.search(slice, k));
        let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        Ok((scores, indices))
    }

    /// Search one query against a caller-supplied subset of document IDs.
    ///
    /// `doc_ids` are global row ordinals. They may be unsorted and may contain
    /// duplicates; each entry is scored independently, so duplicate IDs can
    /// produce duplicate hits. Results are ordered by bitmap-overlap descending,
    /// then row ID ascending, matching the Rust core tie policy.
    fn search_subset<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
        doc_ids: &Bound<'py, PyAny>,
        k: usize,
    ) -> PyResult<SubsetArrays<'py>> {
        let query = as_f32_1d(query, Some(self.inner.dim()))?;
        let q = query.as_array();
        let q_slice = q.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let ids = as_u32_ids_1d(doc_ids, "doc id")?;
        let ids_slice = ids.as_slice()?;
        check_ids_in_range(ids_slice, self.inner.len(), "doc id")?;
        let (scores, out_ids) = py.detach(|| self.inner.search_subset(q_slice, ids_slice, k));
        Ok((scores.into_pyarray(py), out_ids.into_pyarray(py)))
    }

    /// Return top-`m` candidate doc IDs for a single query as a 1-D `uint32`
    /// array. Used as the candidate generator for two-stage retrieval (bitmap
    /// probe → exact RankQuant rerank). This is a fixed-budget shortlist over
    /// the formal overlap statistic; the theorem's admission rule is an
    /// explicit overlap threshold.
    fn top_m_candidates<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
        m: usize,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let query = as_f32_1d(query, Some(self.inner.dim()))?;
        let arr = query.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let cands = py.detach(|| self.inner.top_m_candidates(slice, m));
        Ok(cands.into_pyarray(py))
    }

    /// Build the query-side top-`n_top` bitmap from an FP32 query, returned as a
    /// 1-D `uint64` array of `dim / 64` words (the doc-side packing). Pairs with
    /// [`Bitmap::body_overlap_scores_subset`] for staged rescoring. The returned
    /// bitmap has exactly `n_top` active coordinates.
    fn build_query_bitmap_fp32<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let query = as_f32_1d(query, Some(self.inner.dim()))?;
        let arr = query.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        Ok(self.inner.build_query_bitmap_fp32(slice).into_pyarray(py))
    }

    /// Batched candidate generation: stream the bitmap corpus once and return
    /// top-`m` candidate doc IDs for each query. `queries` is a 2-D `(batch,
    /// dim)` f32 array; returns a 2-D `uint32` array of shape `(batch, m_eff)`
    /// where `m_eff = min(m, len(index))`. The column count is `m_eff`
    /// regardless of `batch`, so an empty `(0, dim)` input returns `(0, m_eff)`.
    /// Each row has the same fixed-budget semantics as
    /// [`Bitmap::top_m_candidates`].
    fn top_m_candidates_batched<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        m: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let batch = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        // Guard the core's internal `batch * n` (scores) and `batch * qpv`
        // (query bitmaps) allocations BEFORE the call: an overflow there wraps
        // and then indexes out of bounds (a panic), so convert it to a clean
        // ValueError up front. `n.max(qpv)` bounds both core buffers.
        let n = self.inner.len();
        let qpv = self.inner.dim() / 64;
        batch.checked_mul(n.max(qpv)).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("batch * index size overflows usize")
        })?;
        let result = py.detach(|| self.inner.top_m_candidates_batched(slice, m));
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
    /// [`Bitmap::top_m_candidates_batched`] but processes `queries` in groups of
    /// `batch_size` rows in parallel — use when the full query workload is
    /// larger than one batch fits efficiently in cache. `queries` is a 2-D `(n,
    /// dim)` f32 array; returns a 2-D `uint32` array `(n, m_eff)`. `batch_size`
    /// must be > 0. Each row has the same fixed-budget semantics as
    /// [`Bitmap::top_m_candidates`].
    fn top_m_candidates_batched_chunked<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        m: usize,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        if batch_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "batch_size must be > 0",
            ));
        }
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let n_queries = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
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
        let result = py.detach(|| {
            self.inner
                .top_m_candidates_batched_chunked(slice, m, effective_batch)
        });
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
    /// (e.g. from [`Bitmap::build_query_bitmap_fp32`]); `doc_ids` is a 1-D
    /// integer array of any dtype (converted to `uint32`) whose ids must be in
    /// range. Returns a 1-D `uint32` array of overlap scores aligned to `doc_ids`.
    /// These are the exact overlap values to compare against an explicit
    /// calibrated cutoff when using threshold admission.
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
        doc_ids: &Bound<'py, PyAny>,
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
        // Accept doc ids of any integer dtype (NumPy index arrays are int64 by
        // default) and convert to u32 with checked bounds, then reject any id
        // outside the corpus before dispatch.
        let doc_ids = as_u32_ids_1d(doc_ids, "doc id")?;
        let ids_slice = doc_ids.as_slice()?;
        check_ids_in_range(ids_slice, self.inner.len(), "doc id")?;
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
        py.detach(|| {
            self.inner
                .body_overlap_scores_subset(qb_slice, ids_slice, &mut out)
        });
        Ok(out.into_pyarray(py))
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Serialise the bitmap index to a `.tvbm` file.
    ///
    /// `path` is forwarded to the filesystem unmodified — no `..` / traversal
    /// sanitisation — so treat it as trusted input (see the module docstring).
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a `Bitmap` index from a `.tvbm` file written by [`Bitmap::write`].
    ///
    /// `path` is forwarded to the filesystem unmodified — no `..` / traversal
    /// sanitisation — so treat it as trusted input (see the module docstring).
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
    /// bytes per doc (128 B at D = 1024). This is separate from the
    /// constant-weight `Bitmap` theorem and its hypergeometric overlap null.
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

    fn __repr__(&self) -> String {
        format!(
            "SignBitmap(dim={}, n={})",
            self.inner.dim(),
            self.inner.len()
        )
    }

    fn add<'py>(&mut self, py: Python<'py>, vectors: &Bound<'py, PyAny>) -> PyResult<()> {
        let vectors = as_f32_2d(vectors, self.inner.dim())?;
        let arr = vectors.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        check_add_capacity(
            self.inner.len(),
            arr.nrows(),
            self.inner.dim() / 64,
            std::mem::size_of::<u64>(),
        )?;
        // Release the GIL around the parallel rank-transform / pack so other
        // Python threads run during a bulk add. `slice` (`&[f32]`) and
        // `&mut self.inner` are both `Ungil`, so no pointer juggling is needed.
        //
        // SAFETY (detaching on a `&mut self` method): `detach` drops the GIL
        // but NOT the `&mut self` exclusive borrow — PyO3 holds this object's
        // runtime borrow flag for the whole call, so another thread that
        // re-acquires the GIL and tries to touch the SAME object gets a clean
        // `Already borrowed` RuntimeError, never concurrent mutation. Distinct
        // objects run freely, which is the point of releasing the GIL.
        py.detach(|| self.inner.add(slice));
        Ok(())
    }

    /// Top-`m` candidate doc IDs for a single query, ranked by ascending Hamming
    /// distance (= descending sign agreement). Returns a 1-D `uint32` array of
    /// length `min(m, len(index))`.
    fn top_m_candidates<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
        m: usize,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let query = as_f32_1d(query, Some(self.inner.dim()))?;
        let arr = query.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let cands = py.detach(|| self.inner.top_m_candidates(slice, m));
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
        queries: &Bound<'py, PyAny>,
        m: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let batch = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        // Guard the core's internal `batch * n` (scores) and `batch * qpv`
        // (query bitmaps) allocations BEFORE the call: an overflow there wraps
        // and then indexes out of bounds (a panic), so convert it to a clean
        // ValueError up front. `n.max(qpv)` bounds both core buffers.
        let n = self.inner.len();
        let qpv = self.inner.dim() / 64;
        batch.checked_mul(n.max(qpv)).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("batch * index size overflows usize")
        })?;
        let result = py.detach(|| self.inner.top_m_candidates_batched(slice, m));
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

    /// Dense full-corpus sign-agreement scores for a single query. Returns a
    /// 1-D `uint32` array of length `len(index)`, aligned by document id.
    fn score_all<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u32>>> {
        let query = as_f32_1d(query, Some(self.inner.dim()))?;
        let arr = query.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let scores = py.detach(|| self.inner.score_all(slice));
        Ok(scores.into_pyarray(py))
    }

    /// Batched dense full-corpus sign-agreement scores. Returns a 2-D `uint32`
    /// array of shape `(batch, len(index))`, aligned by query row and document id.
    fn score_all_batched<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let queries = as_f32_2d(queries, self.inner.dim())?;
        let arr = queries.as_array();
        let batch = arr.nrows();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        let n = self.inner.len();
        let qpv = self.inner.dim() / 64;
        batch.checked_mul(n.max(qpv)).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("batch * index size overflows usize")
        })?;
        let scores = py.detach(|| self.inner.score_all_batched_flat(slice));
        Ok(numpy::ndarray::Array2::from_shape_vec((batch, n), scores)
            .expect("internal: batched dense score flatten shape invariant")
            .into_pyarray(py))
    }

    /// Build the query-side sign bitmap from an FP32 query, returned as a 1-D
    /// `uint64` array of `dim / 64` words (`bit j` set iff `q[j] > 0`).
    fn build_query_bitmap<'py>(
        &self,
        py: Python<'py>,
        query: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<u64>>> {
        let query = as_f32_1d(query, Some(self.inner.dim()))?;
        let arr = query.as_array();
        let slice = arr.as_slice().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "array must be C-contiguous; call np.ascontiguousarray() first",
            )
        })?;
        Ok(self.inner.build_query_bitmap(slice).into_pyarray(py))
    }

    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Persist the sign-bitmap payload to a `.tvsb` file. Format: 13-byte header
    /// (`TVSB` magic + version + dim + n_vectors) + LE u64 bitmaps.
    ///
    /// `path` is forwarded to the filesystem unmodified — no `..` / traversal
    /// sanitisation — so treat it as trusted input (see the module docstring).
    fn write(&self, path: &str) -> PyResult<()> {
        self.inner
            .write(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Load a `SignBitmap` from a `.tvsb` file previously written by
    /// [`SignBitmap::write`]. Raises `IOError` if the file is missing, malformed,
    /// or its payload length disagrees with the header-declared shape.
    ///
    /// `path` is forwarded to the filesystem unmodified — no `..` / traversal
    /// sanitisation — so treat it as trusted input (see the module docstring).
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
    v: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    let v = as_f32_1d(v, None)?;
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
    // The core `rank_to_bucket` now asserts `rank < d` (fail-loud, matching the
    // other bucket primitives); surface that as a clean `ValueError` rather
    // than letting the assert escape as a `PanicException`.
    if rank as usize >= d {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "rank ({rank}) must be < d ({d})"
        )));
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
    // `bucket_ranks` treats the input as a rank vector: each entry indexes into
    // `[0, len)`, and the core `rank_to_bucket` now asserts `rank < len`. Reject
    // an out-of-range entry here with a clean `ValueError` rather than letting
    // that assert surface as a `PanicException`. A valid rank vector (a
    // permutation of `[0, len)`) never trips this.
    let d = slice.len();
    if let Some(&bad) = slice.iter().find(|&&r| r as usize >= d) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "rank ({bad}) must be < d ({d})"
        )));
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
    // Reject out-of-range bucket codes here so the caller gets a clean
    // `ValueError`: the core `pack_buckets` now *asserts* every code is in
    // `[0, 1 << bits)` (it fails loud rather than masking), so an unchecked
    // out-of-range value would otherwise escape as a `PanicException`. The
    // bucket alphabet is [0, 1 << bits).
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
    queries: &Bound<'py, PyAny>,
    k: usize,
) -> PyResult<SearchArrays<'py>> {
    if index.inner.bits() == 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "search_asymmetric_byte_lut is a benchmark-only helper and does not support bits=1; use RankQuant.search_asymmetric instead",
        ));
    }
    let queries = as_f32_2d(queries, index.inner.dim())?;
    let arr = queries.as_array();
    let nq = arr.nrows();
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    // Deref the GIL-bound `PyRef` to a plain `&RankQuant` *before* the closure:
    // capturing `index` (a `PyRef`) directly would make the closure non-`Ungil`,
    // but a bare `&ordvec_core::RankQuant` is fine to carry across `detach`.
    let inner = &index.inner;
    let results = py.detach(|| ordvec_core::search_asymmetric_byte_lut(inner, slice, k));
    let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    Ok((scores, indices))
}

/// Eval-only symmetric RankQuant-style search for arbitrary `bits` in `1..=7`.
///
/// This rank-transforms and buckets the raw `corpus`/`queries` matrices on the
/// fly, so it supports non-byte-aligned widths such as `bits=3` without changing
/// `RankQuant` storage or `.tvrq` persistence. Returns `(scores, indices)` with
/// the same shape contract as `RankQuant.search`.
#[pyfunction]
fn rankquant_eval_search<'py>(
    py: Python<'py>,
    corpus: &Bound<'py, PyAny>,
    queries: &Bound<'py, PyAny>,
    bits: u8,
    k: usize,
) -> PyResult<SearchArrays<'py>> {
    check_bits_1_7(bits)?;
    let dim = infer_float_2d_width(corpus)?;
    if !(2..=u16::MAX as usize).contains(&dim) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "corpus width must be in [2, {}]",
            u16::MAX
        )));
    }
    let corpus = as_f32_2d(corpus, dim)?;
    let queries = as_f32_2d(queries, dim)?;
    let q_arr = queries.as_array();
    let nq = q_arr.nrows();
    let corpus_arr = corpus.as_array();
    let corpus_slice = corpus_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    let query_slice = q_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(
            "array must be C-contiguous; call np.ascontiguousarray() first",
        )
    })?;
    let results =
        py.detach(|| ordvec_core::rankquant_eval_search(corpus_slice, query_slice, dim, bits, k));
    let scores = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.scores)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        .into_pyarray(py);
    let indices = numpy::ndarray::Array2::from_shape_vec((nq, results.k), results.indices)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
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
    m.add_function(wrap_pyfunction!(rankquant_eval_search, m)?)?;

    // Loader/limit constants (parity with `ordvec::rank_io::*`).
    m.add("MAX_DIM", ordvec_core::rank_io::MAX_DIM)?;
    m.add(
        "MAX_SIGN_BITMAP_DIM",
        ordvec_core::rank_io::MAX_SIGN_BITMAP_DIM,
    )?;
    m.add("MAX_VECTORS", ordvec_core::rank_io::MAX_VECTORS)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::check_add_capacity;
    use ordvec_core::rank_io::MAX_VECTORS;

    #[test]
    fn add_capacity_allows_exact_ceiling() {
        check_add_capacity(MAX_VECTORS - 1, 1, 1, 1).unwrap();
        check_add_capacity(MAX_VECTORS, 0, 1, 1).unwrap();
    }

    #[test]
    fn add_capacity_rejects_vector_count_overflow() {
        let err = check_add_capacity(MAX_VECTORS, 1, 1, 1).unwrap_err();
        assert!(err.to_string().contains("MAX_VECTORS"));
    }

    #[test]
    fn add_capacity_rejects_buffer_length_overflow() {
        let err = check_add_capacity(0, MAX_VECTORS, usize::MAX, 1).unwrap_err();
        assert!(err.to_string().contains("buffer length"));
    }

    #[test]
    fn add_capacity_rejects_byte_size_overflow() {
        let err = check_add_capacity(0, MAX_VECTORS, usize::MAX / 2, 4).unwrap_err();
        assert!(err.to_string().contains("byte size"));
    }
}
