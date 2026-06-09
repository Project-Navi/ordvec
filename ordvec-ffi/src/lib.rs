#![allow(non_camel_case_types)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::io::{self, Read};
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::ptr;
use std::time::Instant;

use ordvec::{probe_index_metadata, Bitmap, IndexKind, IndexMetadata, IndexParams, RankQuant};

pub type ordvec_status_t = u32;
pub type ordvec_index_kind_t = u32;

pub const ORDVEC_ABI_VERSION: u32 = 1;

pub const ORDVEC_STATUS_OK: ordvec_status_t = 0;
pub const ORDVEC_STATUS_NULL_POINTER: ordvec_status_t = 1;
pub const ORDVEC_STATUS_BAD_ARGUMENT: ordvec_status_t = 2;
pub const ORDVEC_STATUS_BAD_STRUCT_SIZE: ordvec_status_t = 3;
pub const ORDVEC_STATUS_UNSUPPORTED_FORMAT: ordvec_status_t = 4;
pub const ORDVEC_STATUS_CORRUPT_INDEX: ordvec_status_t = 5;
pub const ORDVEC_STATUS_IO: ordvec_status_t = 6;
pub const ORDVEC_STATUS_DIM_MISMATCH: ordvec_status_t = 7;
pub const ORDVEC_STATUS_NONFINITE_QUERY: ordvec_status_t = 8;
pub const ORDVEC_STATUS_ROW_ID_OUT_OF_RANGE: ordvec_status_t = 9;
pub const ORDVEC_STATUS_BUFFER_TOO_SMALL: ordvec_status_t = 10;
pub const ORDVEC_STATUS_UNSUPPORTED_OPERATION: ordvec_status_t = 11;
pub const ORDVEC_STATUS_PANIC: ordvec_status_t = 12;
pub const ORDVEC_STATUS_INTERNAL: ordvec_status_t = 13;

pub const ORDVEC_INDEX_KIND_UNKNOWN: ordvec_index_kind_t = 0;
pub const ORDVEC_INDEX_KIND_RANK_QUANT: ordvec_index_kind_t = 1;
pub const ORDVEC_INDEX_KIND_BITMAP: ordvec_index_kind_t = 2;

pub const ORDVEC_CAP_FULL_SEARCH: u64 = 1 << 0;
pub const ORDVEC_CAP_SUBSET_SEARCH: u64 = 1 << 1;
/// Search statistics are supported. ABI v1 populates total time and
/// search-space counters; granular timing and byte counters are reserved and
/// remain zero until measured in a future ABI.
pub const ORDVEC_CAP_STATS: u64 = 1 << 2;
pub const ORDVEC_CAP_ID_EQUALS_ROW_ID: u64 = 1 << 3;

pub const ORDVEC_SEARCH_FLAG_NONE: u64 = 0;

pub struct ordvec_index_t {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ordvec_index_info_t {
    pub struct_size: u64,
    pub kind: u32,
    pub format_version: u32,
    pub dim: u64,
    pub bit_width: u32,
    pub n_top: u32,
    pub vector_count: u64,
    pub bytes_per_vec: u64,
    pub source_file_size_bytes: u64,
    pub capabilities: u64,
    pub reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ordvec_search_params_t {
    pub struct_size: u64,
    pub query: *const f32,
    pub dim: u64,
    pub k: u64,
    /// Optional subset rows. Rows are global row IDs, may be unsorted, and may
    /// contain duplicates; duplicate entries are scored independently.
    pub candidate_rows: *const u32,
    pub candidate_count: u64,
    pub flags: u64,
    pub user_tag: u64,
    pub reserved: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ordvec_hit_t {
    pub row_id: u64,
    pub id: u64,
    pub score: f32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Optional search statistics written when `stats_out` is non-null.
///
/// ABI v1 counts candidate entries rather than unique rows. Duplicate subset
/// candidates therefore increase both `candidate_count` and `vectors_scored`.
/// `prepare_ns`, `score_ns`, `select_ns`, and `bytes_read` are reserved and
/// currently written as zero.
pub struct ordvec_search_stats_t {
    pub struct_size: u64,
    pub abi_version: u32,
    pub kind: u32,
    pub dim: u64,
    pub bit_width: u32,
    pub n_top: u32,
    pub k: u64,
    pub user_tag: u64,
    pub vector_count: u64,
    pub candidate_count: u64,
    pub returned_count: u64,
    pub total_ns: u64,
    pub prepare_ns: u64,
    pub score_ns: u64,
    pub select_ns: u64,
    pub vectors_scored: u64,
    pub bytes_read: u64,
    pub reserved: [u64; 8],
}

enum LoadedIndex {
    RankQuant(RankQuant),
    Bitmap(Bitmap),
}

struct IndexHandle {
    index: LoadedIndex,
    source_file_size_bytes: u64,
}

impl IndexHandle {
    fn kind(&self) -> u32 {
        match self.index {
            LoadedIndex::RankQuant(_) => ORDVEC_INDEX_KIND_RANK_QUANT,
            LoadedIndex::Bitmap(_) => ORDVEC_INDEX_KIND_BITMAP,
        }
    }

    fn dim(&self) -> usize {
        match &self.index {
            LoadedIndex::RankQuant(index) => index.dim(),
            LoadedIndex::Bitmap(index) => index.dim(),
        }
    }

    fn len(&self) -> usize {
        match &self.index {
            LoadedIndex::RankQuant(index) => index.len(),
            LoadedIndex::Bitmap(index) => index.len(),
        }
    }

    fn bit_width(&self) -> u32 {
        match &self.index {
            LoadedIndex::RankQuant(index) => u32::from(index.bits()),
            LoadedIndex::Bitmap(_) => 0,
        }
    }

    fn n_top(&self) -> u32 {
        match &self.index {
            LoadedIndex::RankQuant(_) => 0,
            LoadedIndex::Bitmap(index) => index.n_top() as u32,
        }
    }

    fn bytes_per_vec(&self) -> usize {
        match &self.index {
            LoadedIndex::RankQuant(index) => index.bytes_per_vec(),
            LoadedIndex::Bitmap(index) => index.bytes_per_vec(),
        }
    }
}

#[derive(Debug)]
struct FfiError {
    status: ordvec_status_t,
    message: String,
}

impl FfiError {
    fn new(status: ordvec_status_t, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").expect("empty CString"));
}

fn set_last_error(message: &str) {
    let clean = message.replace('\0', "\\0");
    let cstr =
        CString::new(clean).unwrap_or_else(|_| CString::new("ordvec: invalid error").unwrap());
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = cstr;
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new("").expect("empty CString");
    });
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic crossing ordvec C ABI".to_owned()
    }
}

fn ffi_boundary(f: impl FnOnce() -> Result<(), FfiError>) -> ordvec_status_t {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => {
            clear_last_error();
            ORDVEC_STATUS_OK
        }
        Ok(Err(err)) => {
            set_last_error(&err.message);
            err.status
        }
        Err(payload) => {
            set_last_error(&panic_message(payload));
            ORDVEC_STATUS_PANIC
        }
    }
}

fn io_error_status(err: &io::Error) -> ordvec_status_t {
    match err.kind() {
        io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof => ORDVEC_STATUS_CORRUPT_INDEX,
        _ => ORDVEC_STATUS_IO,
    }
}

fn io_to_ffi(err: io::Error, context: &str) -> FfiError {
    FfiError::new(io_error_status(&err), format!("{context}: {err}"))
}

fn default_info() -> ordvec_index_info_t {
    ordvec_index_info_t {
        struct_size: std::mem::size_of::<ordvec_index_info_t>() as u64,
        kind: ORDVEC_INDEX_KIND_UNKNOWN,
        format_version: 0,
        dim: 0,
        bit_width: 0,
        n_top: 0,
        vector_count: 0,
        bytes_per_vec: 0,
        source_file_size_bytes: 0,
        capabilities: 0,
        reserved: [0; 8],
    }
}

fn default_params() -> ordvec_search_params_t {
    ordvec_search_params_t {
        struct_size: std::mem::size_of::<ordvec_search_params_t>() as u64,
        query: ptr::null(),
        dim: 0,
        k: 0,
        candidate_rows: ptr::null(),
        candidate_count: 0,
        flags: ORDVEC_SEARCH_FLAG_NONE,
        user_tag: 0,
        reserved: [0; 8],
    }
}

fn default_stats() -> ordvec_search_stats_t {
    ordvec_search_stats_t {
        struct_size: std::mem::size_of::<ordvec_search_stats_t>() as u64,
        abi_version: ORDVEC_ABI_VERSION,
        kind: ORDVEC_INDEX_KIND_UNKNOWN,
        dim: 0,
        bit_width: 0,
        n_top: 0,
        k: 0,
        user_tag: 0,
        vector_count: 0,
        candidate_count: 0,
        returned_count: 0,
        total_ns: 0,
        prepare_ns: 0,
        score_ns: 0,
        select_ns: 0,
        vectors_scored: 0,
        bytes_read: 0,
        reserved: [0; 8],
    }
}

fn check_exact_size(got: u64, expected: usize, what: &str) -> Result<(), FfiError> {
    if got != expected as u64 {
        return Err(FfiError::new(
            ORDVEC_STATUS_BAD_STRUCT_SIZE,
            format!("{what}.struct_size must be exactly {expected}, got {got}"),
        ));
    }
    Ok(())
}

fn check_reserved_zero(reserved: &[u64], what: &str) -> Result<(), FfiError> {
    if reserved.iter().any(|&x| x != 0) {
        return Err(FfiError::new(
            ORDVEC_STATUS_BAD_ARGUMENT,
            format!("{what}.reserved fields must be zero"),
        ));
    }
    Ok(())
}

fn duration_ns(start: Instant) -> u64 {
    let ns = start.elapsed().as_nanos();
    ns.min(u128::from(u64::MAX)) as u64
}

fn sniff_magic(path: &Path) -> Result<[u8; 4], FfiError> {
    let mut file = std::fs::File::open(path).map_err(|err| io_to_ffi(err, "open index"))?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|err| io_to_ffi(err, "read index magic"))?;
    Ok(magic)
}

unsafe fn handle_from_ptr<'a>(index: *const ordvec_index_t) -> Result<&'a IndexHandle, FfiError> {
    if index.is_null() {
        return Err(FfiError::new(
            ORDVEC_STATUS_NULL_POINTER,
            "index pointer is NULL",
        ));
    }
    // SAFETY: the caller supplied a non-null handle previously returned by
    // ordvec_index_load; use-after-free and double-free are undefined by the C
    // ABI contract.
    Ok(unsafe { &*(index as *const IndexHandle) })
}

fn info_for_handle(handle: &IndexHandle) -> ordvec_index_info_t {
    let mut info = default_info();
    info.kind = handle.kind();
    info.format_version = 1;
    info.dim = handle.dim() as u64;
    info.bit_width = handle.bit_width();
    info.n_top = handle.n_top();
    info.vector_count = handle.len() as u64;
    info.bytes_per_vec = handle.bytes_per_vec() as u64;
    info.source_file_size_bytes = handle.source_file_size_bytes;
    info.capabilities = ORDVEC_CAP_FULL_SEARCH
        | ORDVEC_CAP_SUBSET_SEARCH
        | ORDVEC_CAP_STATS
        | ORDVEC_CAP_ID_EQUALS_ROW_ID;
    info
}

fn info_for_metadata(meta: &IndexMetadata) -> Result<ordvec_index_info_t, FfiError> {
    let mut info = default_info();
    info.kind =
        match meta.kind {
            IndexKind::RankQuant => ORDVEC_INDEX_KIND_RANK_QUANT,
            IndexKind::Bitmap => ORDVEC_INDEX_KIND_BITMAP,
            IndexKind::Rank | IndexKind::SignBitmap => return Err(FfiError::new(
                ORDVEC_STATUS_UNSUPPORTED_FORMAT,
                "ABI v1 supports metadata probes only for TVRQ RankQuant and TVBM Bitmap indexes",
            )),
        };
    info.format_version = u32::from(meta.format_version);
    info.dim = meta.dim as u64;
    info.vector_count = meta.vector_count as u64;
    info.bytes_per_vec = meta.bytes_per_vec as u64;
    info.source_file_size_bytes = meta.file_size_bytes;
    match meta.params {
        IndexParams::RankQuant { bits } => {
            info.bit_width = u32::from(bits);
        }
        IndexParams::Bitmap { n_top } => {
            info.n_top = n_top as u32;
        }
        IndexParams::Rank | IndexParams::SignBitmap => {}
    }
    info.capabilities = ORDVEC_CAP_FULL_SEARCH
        | ORDVEC_CAP_SUBSET_SEARCH
        | ORDVEC_CAP_STATS
        | ORDVEC_CAP_ID_EQUALS_ROW_ID;
    Ok(info)
}

fn copy_hits(scores: &[f32], indices: &[i64], hits_out: *mut ordvec_hit_t) {
    debug_assert_eq!(scores.len(), indices.len());
    for (slot, (&score, &row)) in scores.iter().zip(indices).enumerate() {
        let row_id = u64::try_from(row).expect("ordvec core returned a negative row id");
        // SAFETY: validation proved the caller supplied at least scores.len()
        // writable hit slots.
        unsafe {
            *hits_out.add(slot) = ordvec_hit_t {
                row_id,
                id: row_id,
                score,
                reserved: 0,
            };
        }
    }
}

fn normalize_global_order(scores: Vec<f32>, indices: Vec<i64>, k: usize) -> (Vec<f32>, Vec<i64>) {
    let mut entries: Vec<(f32, i64)> = scores.into_iter().zip(indices).collect();
    entries.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    entries.truncate(k);
    entries.into_iter().unzip()
}

struct SearchValidation<'a> {
    query: &'a [f32],
    candidates: Option<&'a [u32]>,
    search_space: usize,
    required_hits: usize,
}

fn validate_search<'a>(
    handle: &'a IndexHandle,
    params: &'a ordvec_search_params_t,
    hits_out: *mut ordvec_hit_t,
    hits_capacity: u64,
    returned_out: *mut u64,
    stats_out: *mut ordvec_search_stats_t,
) -> Result<SearchValidation<'a>, FfiError> {
    check_exact_size(
        params.struct_size,
        std::mem::size_of::<ordvec_search_params_t>(),
        "ordvec_search_params_t",
    )?;
    if params.flags != ORDVEC_SEARCH_FLAG_NONE {
        return Err(FfiError::new(
            ORDVEC_STATUS_BAD_ARGUMENT,
            format!("unknown search flags: {}", params.flags),
        ));
    }
    check_reserved_zero(&params.reserved, "ordvec_search_params_t")?;
    if returned_out.is_null() {
        return Err(FfiError::new(
            ORDVEC_STATUS_NULL_POINTER,
            "returned_out pointer is NULL",
        ));
    }
    if !stats_out.is_null() {
        // SAFETY: stats_out is non-null; read only the leading struct_size
        // field without forming a reference to potentially uninitialized
        // output storage.
        let stats_size = unsafe { ptr::addr_of!((*stats_out).struct_size).read() };
        check_exact_size(
            stats_size,
            std::mem::size_of::<ordvec_search_stats_t>(),
            "ordvec_search_stats_t",
        )?;
    }
    if params.query.is_null() {
        return Err(FfiError::new(
            ORDVEC_STATUS_NULL_POINTER,
            "query pointer is NULL",
        ));
    }
    if params.dim != handle.dim() as u64 {
        return Err(FfiError::new(
            ORDVEC_STATUS_DIM_MISMATCH,
            format!(
                "query dim {} does not match index dim {}",
                params.dim,
                handle.dim()
            ),
        ));
    }
    let dim = handle.dim();
    // SAFETY: query is non-null and dim was validated against the loaded index.
    let query = unsafe { std::slice::from_raw_parts(params.query, dim) };
    if query.iter().any(|x| !x.is_finite()) {
        return Err(FfiError::new(
            ORDVEC_STATUS_NONFINITE_QUERY,
            "query contains NaN or infinity",
        ));
    }

    let candidates = match (params.candidate_count, params.candidate_rows.is_null()) {
        (0, true) => None,
        (0, false) => {
            return Err(FfiError::new(
                ORDVEC_STATUS_BAD_ARGUMENT,
                "candidate_count is zero but candidate_rows is non-NULL",
            ))
        }
        (_, true) => {
            return Err(FfiError::new(
                ORDVEC_STATUS_NULL_POINTER,
                "candidate_rows pointer is NULL but candidate_count is nonzero",
            ))
        }
        (count, false) => {
            let count = usize::try_from(count).map_err(|_| {
                FfiError::new(
                    ORDVEC_STATUS_BAD_ARGUMENT,
                    "candidate_count does not fit this platform",
                )
            })?;
            // SAFETY: candidate_rows is non-null and count was converted to usize.
            Some(unsafe { std::slice::from_raw_parts(params.candidate_rows, count) })
        }
    };

    if let Some(rows) = candidates {
        if let Some(&bad) = rows.iter().find(|&&row| (row as usize) >= handle.len()) {
            return Err(FfiError::new(
                ORDVEC_STATUS_ROW_ID_OUT_OF_RANGE,
                format!(
                    "candidate row {bad} is out of range for {} vectors",
                    handle.len()
                ),
            ));
        }
    }

    let search_space = candidates.map_or(handle.len(), <[u32]>::len);
    let required_hits = params.k.min(search_space as u64) as usize;
    if required_hits > 0 {
        if hits_out.is_null() {
            return Err(FfiError::new(
                ORDVEC_STATUS_NULL_POINTER,
                "hits_out pointer is NULL",
            ));
        }
        if hits_capacity < required_hits as u64 {
            return Err(FfiError::new(
                ORDVEC_STATUS_BUFFER_TOO_SMALL,
                format!("hits_capacity {hits_capacity} is smaller than required {required_hits}"),
            ));
        }
    }

    Ok(SearchValidation {
        query,
        candidates,
        search_space,
        required_hits,
    })
}

fn write_stats(
    handle: &IndexHandle,
    params: &ordvec_search_params_t,
    stats_out: *mut ordvec_search_stats_t,
    search_space: usize,
    returned_count: usize,
    total_ns: u64,
) {
    if stats_out.is_null() {
        return;
    }
    let mut stats = default_stats();
    stats.kind = handle.kind();
    stats.dim = handle.dim() as u64;
    stats.bit_width = handle.bit_width();
    stats.n_top = handle.n_top();
    stats.k = params.k;
    stats.user_tag = params.user_tag;
    stats.vector_count = handle.len() as u64;
    stats.candidate_count = search_space as u64;
    stats.returned_count = returned_count as u64;
    stats.total_ns = total_ns;
    stats.vectors_scored = search_space as u64;
    // SAFETY: validation checked stats_out is either null or points to a struct
    // with the exact ABI v1 size.
    unsafe {
        *stats_out = stats;
    }
}

#[no_mangle]
pub extern "C" fn ordvec_abi_version() -> u32 {
    ORDVEC_ABI_VERSION
}

#[no_mangle]
pub extern "C" fn ordvec_version_string() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn ordvec_status_name(status: ordvec_status_t) -> *const c_char {
    match status {
        ORDVEC_STATUS_OK => c"OK".as_ptr(),
        ORDVEC_STATUS_NULL_POINTER => c"NULL_POINTER".as_ptr(),
        ORDVEC_STATUS_BAD_ARGUMENT => c"BAD_ARGUMENT".as_ptr(),
        ORDVEC_STATUS_BAD_STRUCT_SIZE => c"BAD_STRUCT_SIZE".as_ptr(),
        ORDVEC_STATUS_UNSUPPORTED_FORMAT => c"UNSUPPORTED_FORMAT".as_ptr(),
        ORDVEC_STATUS_CORRUPT_INDEX => c"CORRUPT_INDEX".as_ptr(),
        ORDVEC_STATUS_IO => c"IO".as_ptr(),
        ORDVEC_STATUS_DIM_MISMATCH => c"DIM_MISMATCH".as_ptr(),
        ORDVEC_STATUS_NONFINITE_QUERY => c"NONFINITE_QUERY".as_ptr(),
        ORDVEC_STATUS_ROW_ID_OUT_OF_RANGE => c"ROW_ID_OUT_OF_RANGE".as_ptr(),
        ORDVEC_STATUS_BUFFER_TOO_SMALL => c"BUFFER_TOO_SMALL".as_ptr(),
        ORDVEC_STATUS_UNSUPPORTED_OPERATION => c"UNSUPPORTED_OPERATION".as_ptr(),
        ORDVEC_STATUS_PANIC => c"PANIC".as_ptr(),
        ORDVEC_STATUS_INTERNAL => c"INTERNAL".as_ptr(),
        _ => c"UNKNOWN".as_ptr(),
    }
}

#[no_mangle]
pub extern "C" fn ordvec_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| cell.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn ordvec_init() -> ordvec_status_t {
    ffi_boundary(|| Ok(()))
}

#[no_mangle]
/// Initialize an `ordvec_index_info_t` with ABI v1 defaults.
///
/// # Safety
///
/// If `info` is non-null, it must point to writable memory large enough for
/// `ordvec_index_info_t`.
pub unsafe extern "C" fn ordvec_index_info_init(info: *mut ordvec_index_info_t) {
    if !info.is_null() {
        // SAFETY: caller supplied a non-null writable pointer for initialization.
        unsafe {
            *info = default_info();
        }
    }
}

#[no_mangle]
/// Initialize an `ordvec_search_params_t` with ABI v1 defaults.
///
/// # Safety
///
/// If `params` is non-null, it must point to writable memory large enough for
/// `ordvec_search_params_t`.
pub unsafe extern "C" fn ordvec_search_params_init(params: *mut ordvec_search_params_t) {
    if !params.is_null() {
        // SAFETY: caller supplied a non-null writable pointer for initialization.
        unsafe {
            *params = default_params();
        }
    }
}

#[no_mangle]
/// Initialize an `ordvec_search_stats_t` with ABI v1 defaults.
///
/// # Safety
///
/// If `stats` is non-null, it must point to writable memory large enough for
/// `ordvec_search_stats_t`.
pub unsafe extern "C" fn ordvec_search_stats_init(stats: *mut ordvec_search_stats_t) {
    if !stats.is_null() {
        // SAFETY: caller supplied a non-null writable pointer for initialization.
        unsafe {
            *stats = default_stats();
        }
    }
}

#[no_mangle]
/// Load a `.tvrq` RankQuant or `.tvbm` Bitmap index.
///
/// # Safety
///
/// `path` must be a non-null, NUL-terminated, valid UTF-8 C string. `out`
/// must be non-null and point to writable memory for one `ordvec_index_t *`.
pub unsafe extern "C" fn ordvec_index_load(
    path: *const c_char,
    flags: u64,
    out: *mut *mut ordvec_index_t,
) -> ordvec_status_t {
    ffi_boundary(|| {
        if path.is_null() {
            return Err(FfiError::new(
                ORDVEC_STATUS_NULL_POINTER,
                "path pointer is NULL",
            ));
        }
        if out.is_null() {
            return Err(FfiError::new(
                ORDVEC_STATUS_NULL_POINTER,
                "out pointer is NULL",
            ));
        }
        // SAFETY: out is non-null and writable by contract.
        unsafe {
            *out = ptr::null_mut();
        }
        if flags != 0 {
            return Err(FfiError::new(
                ORDVEC_STATUS_BAD_ARGUMENT,
                format!("unknown load flags: {flags}"),
            ));
        }
        // SAFETY: path is a non-null NUL-terminated C string by caller contract.
        let path = unsafe { CStr::from_ptr(path) };
        let path = path.to_str().map_err(|_| {
            FfiError::new(
                ORDVEC_STATUS_BAD_ARGUMENT,
                "path must be valid UTF-8 in ABI v1",
            )
        })?;
        let path = Path::new(path);
        let magic = sniff_magic(path)?;
        let source_file_size_bytes = std::fs::metadata(path)
            .map_err(|err| io_to_ffi(err, "stat index"))?
            .len();

        let index = match &magic {
            b"TVRQ" => LoadedIndex::RankQuant(
                RankQuant::load(path).map_err(|err| io_to_ffi(err, "load TVRQ index"))?,
            ),
            b"TVBM" => LoadedIndex::Bitmap(
                Bitmap::load(path).map_err(|err| io_to_ffi(err, "load TVBM index"))?,
            ),
            b"TVR1" | b"TVSB" => {
                return Err(FfiError::new(
                    ORDVEC_STATUS_UNSUPPORTED_FORMAT,
                    "ABI v1 supports only TVRQ RankQuant and TVBM Bitmap indexes",
                ))
            }
            _ => {
                return Err(FfiError::new(
                    ORDVEC_STATUS_CORRUPT_INDEX,
                    "unrecognized ordvec index magic",
                ))
            }
        };

        let handle = Box::new(IndexHandle {
            index,
            source_file_size_bytes,
        });
        // SAFETY: out is non-null and writable; ownership moves to the caller.
        unsafe {
            *out = Box::into_raw(handle) as *mut ordvec_index_t;
        }
        Ok(())
    })
}

#[no_mangle]
/// Probe on-disk metadata for a `.tvrq` RankQuant or `.tvbm` Bitmap index
/// without loading payload rows into an index handle.
///
/// This validates the fixed header, declared dimensions, payload byte count,
/// and exact file length. Full row-invariant validation remains the job of
/// `ordvec_index_load`.
///
/// # Safety
///
/// `path` must be a non-null, NUL-terminated, valid UTF-8 C string. `info_out`
/// must be non-null, initialized with `ordvec_index_info_init`, and point to
/// writable memory for `ordvec_index_info_t`.
pub unsafe extern "C" fn ordvec_index_probe(
    path: *const c_char,
    flags: u64,
    info_out: *mut ordvec_index_info_t,
) -> ordvec_status_t {
    ffi_boundary(|| {
        if path.is_null() {
            return Err(FfiError::new(
                ORDVEC_STATUS_NULL_POINTER,
                "path pointer is NULL",
            ));
        }
        if info_out.is_null() {
            return Err(FfiError::new(
                ORDVEC_STATUS_NULL_POINTER,
                "info_out pointer is NULL",
            ));
        }
        if flags != 0 {
            return Err(FfiError::new(
                ORDVEC_STATUS_BAD_ARGUMENT,
                format!("unknown probe flags: {flags}"),
            ));
        }
        // SAFETY: info_out is non-null; read only the leading struct_size
        // field before overwriting the full output struct.
        let info_size = unsafe { ptr::addr_of!((*info_out).struct_size).read() };
        check_exact_size(
            info_size,
            std::mem::size_of::<ordvec_index_info_t>(),
            "ordvec_index_info_t",
        )?;
        // SAFETY: path is a non-null NUL-terminated C string by caller contract.
        let path = unsafe { CStr::from_ptr(path) };
        let path = path.to_str().map_err(|_| {
            FfiError::new(
                ORDVEC_STATUS_BAD_ARGUMENT,
                "path must be valid UTF-8 in ABI v1",
            )
        })?;
        let meta =
            probe_index_metadata(path).map_err(|err| io_to_ffi(err, "probe index metadata"))?;
        let info = info_for_metadata(&meta)?;
        // SAFETY: info_out is non-null and points to writable output storage.
        unsafe {
            ptr::write(info_out, info);
        }
        Ok(())
    })
}

#[no_mangle]
/// Copy metadata from a loaded index into `info_out`.
///
/// # Safety
///
/// `index` must be a live handle returned by `ordvec_index_load`. `info_out`
/// must be non-null and point to writable memory for `ordvec_index_info_t`.
pub unsafe extern "C" fn ordvec_index_info(
    index: *const ordvec_index_t,
    info_out: *mut ordvec_index_info_t,
) -> ordvec_status_t {
    ffi_boundary(|| {
        let handle = unsafe { handle_from_ptr(index) }?;
        if info_out.is_null() {
            return Err(FfiError::new(
                ORDVEC_STATUS_NULL_POINTER,
                "info_out pointer is NULL",
            ));
        }
        // SAFETY: info_out is non-null; read only the leading struct_size
        // field before overwriting the full output struct.
        let info_size = unsafe { ptr::addr_of!((*info_out).struct_size).read() };
        check_exact_size(
            info_size,
            std::mem::size_of::<ordvec_index_info_t>(),
            "ordvec_index_info_t",
        )?;
        let info = info_for_handle(handle);
        // SAFETY: info_out is non-null and points to writable output storage.
        unsafe {
            ptr::write(info_out, info);
        }
        Ok(())
    })
}

#[no_mangle]
/// Free a loaded index handle. `NULL` is accepted as a no-op.
///
/// # Safety
///
/// Non-null `index` must have been returned by `ordvec_index_load`, must not
/// have been freed before, and must not race with any other call.
pub unsafe extern "C" fn ordvec_index_free(index: *mut ordvec_index_t) {
    if index.is_null() {
        return;
    }
    // SAFETY: the pointer must have been returned by ordvec_index_load and not
    // freed before. Double-free/use-after-free is undefined by the ABI.
    unsafe {
        drop(Box::from_raw(index as *mut IndexHandle));
    }
}

#[no_mangle]
/// Run a synchronous single-query search.
///
/// When `params.candidate_rows` is supplied, those IDs are global row ordinals
/// and may be unsorted or duplicated. Duplicate candidates are scored as
/// separate entries and can produce duplicate hits; callers that need unique
/// output rows must deduplicate before calling.
///
/// # Safety
///
/// `index` must be a live handle returned by `ordvec_index_load`. All non-null
/// pointers in `params`, `hits_out`, `returned_out`, and `stats_out` must be
/// valid for the duration of the call according to the buffer sizes supplied by
/// the caller. No pointer is retained after return.
pub unsafe extern "C" fn ordvec_index_search(
    index: *const ordvec_index_t,
    params: *const ordvec_search_params_t,
    hits_out: *mut ordvec_hit_t,
    hits_capacity: u64,
    returned_out: *mut u64,
    stats_out: *mut ordvec_search_stats_t,
) -> ordvec_status_t {
    ffi_boundary(|| {
        let started = Instant::now();
        let handle = unsafe { handle_from_ptr(index) }?;
        if params.is_null() {
            return Err(FfiError::new(
                ORDVEC_STATUS_NULL_POINTER,
                "params pointer is NULL",
            ));
        }
        // SAFETY: params is non-null and only borrowed for this synchronous call.
        let params = unsafe { &*params };
        let validation = validate_search(
            handle,
            params,
            hits_out,
            hits_capacity,
            returned_out,
            stats_out,
        )?;

        if validation.required_hits == 0 {
            // SAFETY: returned_out was validated non-null.
            unsafe {
                *returned_out = 0;
            }
            write_stats(
                handle,
                params,
                stats_out,
                validation.search_space,
                0,
                duration_ns(started),
            );
            return Ok(());
        }

        let (scores, indices) = match (&handle.index, validation.candidates) {
            (LoadedIndex::RankQuant(index), None) => {
                let results = index.search_asymmetric(validation.query, validation.required_hits);
                (results.scores, results.indices)
            }
            (LoadedIndex::RankQuant(index), Some(rows)) => {
                // Ask the core for every candidate score, then normalize by the
                // ABI's global row-id tie policy before truncating. The core
                // subset helper uses global row IDs as score-tie keys; keeping
                // the ABI normalization centralized preserves duplicate and
                // boundary handling for caller-supplied candidate lists.
                let (scores, indices) =
                    index.search_asymmetric_subset(validation.query, rows, rows.len());
                normalize_global_order(scores, indices, validation.required_hits)
            }
            (LoadedIndex::Bitmap(index), None) => {
                let results = index.search(validation.query, validation.required_hits);
                (results.scores, results.indices)
            }
            (LoadedIndex::Bitmap(index), Some(rows)) => {
                index.search_subset(validation.query, rows, validation.required_hits)
            }
        };

        copy_hits(&scores, &indices, hits_out);
        // SAFETY: returned_out was validated non-null.
        unsafe {
            *returned_out = scores.len() as u64;
        }
        write_stats(
            handle,
            params,
            stats_out,
            validation.search_space,
            scores.len(),
            duration_ns(started),
        );
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordvec::{Bitmap, Rank, SignBitmap};
    use std::ffi::CString;
    use std::io::Write;

    fn temp_path(name: &str, ext: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "ordvec_ffi_{}_{}_{}.{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            ext
        ));
        p
    }

    fn make_rankquant_fixture() -> std::path::PathBuf {
        let path = temp_path("rankquant", "tvrq");
        let mut index = RankQuant::new(16, 2);
        let doc: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let mut corpus = Vec::new();
        for _ in 0..4 {
            corpus.extend_from_slice(&doc);
        }
        index.add(&corpus);
        index.write(&path).unwrap();
        path
    }

    fn make_bitmap_fixture() -> std::path::PathBuf {
        let path = temp_path("bitmap", "tvbm");
        let mut index = Bitmap::new(64, 4);
        let mut doc = vec![0.0f32; 64];
        for (j, value) in doc.iter_mut().take(4).enumerate() {
            *value = 10.0 + j as f32;
        }
        let mut corpus = Vec::new();
        for _ in 0..4 {
            corpus.extend_from_slice(&doc);
        }
        index.add(&corpus);
        index.write(&path).unwrap();
        path
    }

    unsafe fn load_handle(path: &Path) -> *mut ordvec_index_t {
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        let mut out = ptr::null_mut();
        let status = unsafe { ordvec_index_load(cpath.as_ptr(), 0, &mut out) };
        assert_eq!(status, ORDVEC_STATUS_OK);
        assert!(!out.is_null());
        out
    }

    #[test]
    fn layout_sizes_match_header_contract() {
        assert_eq!(std::mem::size_of::<ordvec_index_info_t>(), 128);
        assert_eq!(std::mem::size_of::<ordvec_search_params_t>(), 128);
        assert_eq!(std::mem::size_of::<ordvec_hit_t>(), 24);
        assert_eq!(std::mem::size_of::<ordvec_search_stats_t>(), 184);
    }

    #[test]
    fn load_info_and_free_rankquant() {
        let path = make_rankquant_fixture();
        unsafe {
            let handle = load_handle(&path);
            let mut info = default_info();
            assert_eq!(ordvec_index_info(handle, &mut info), ORDVEC_STATUS_OK);
            assert_eq!(info.kind, ORDVEC_INDEX_KIND_RANK_QUANT);
            assert_eq!(info.format_version, 1);
            assert_eq!(info.dim, 16);
            assert_eq!(info.bit_width, 2);
            assert_eq!(info.vector_count, 4);
            assert_eq!(
                info.capabilities & ORDVEC_CAP_SUBSET_SEARCH,
                ORDVEC_CAP_SUBSET_SEARCH
            );
            ordvec_index_free(handle);
            ordvec_index_free(ptr::null_mut());
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn probe_rankquant_metadata_without_loading() {
        let path = make_rankquant_fixture();
        let cpath = CString::new(path.to_str().unwrap()).unwrap();
        unsafe {
            let mut info = default_info();
            assert_eq!(
                ordvec_index_probe(cpath.as_ptr(), 0, &mut info),
                ORDVEC_STATUS_OK
            );
            assert_eq!(info.kind, ORDVEC_INDEX_KIND_RANK_QUANT);
            assert_eq!(info.format_version, 1);
            assert_eq!(info.dim, 16);
            assert_eq!(info.bit_width, 2);
            assert_eq!(info.n_top, 0);
            assert_eq!(info.vector_count, 4);
            assert_eq!(info.bytes_per_vec, 4);
            assert!(info.source_file_size_bytes > 0);
            assert_eq!(
                info.capabilities & ORDVEC_CAP_SUBSET_SEARCH,
                ORDVEC_CAP_SUBSET_SEARCH
            );
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn full_and_subset_search_rankquant() {
        let path = make_rankquant_fixture();
        unsafe {
            let handle = load_handle(&path);
            let query: Vec<f32> = (0..16).map(|x| x as f32).collect();
            let mut params = default_params();
            params.query = query.as_ptr();
            params.dim = 16;
            params.k = 2;
            params.user_tag = 77;
            let mut hits = vec![
                ordvec_hit_t {
                    row_id: 0,
                    id: 0,
                    score: 0.0,
                    reserved: 0
                };
                2
            ];
            let mut returned = 0;
            let mut stats = default_stats();
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &params,
                    hits.as_mut_ptr(),
                    2,
                    &mut returned,
                    &mut stats
                ),
                ORDVEC_STATUS_OK
            );
            assert_eq!(returned, 2);
            assert_eq!(hits[0].row_id, 0);
            assert_eq!(hits[0].id, hits[0].row_id);
            assert_eq!(stats.user_tag, 77);
            assert_eq!(stats.candidate_count, 4);
            assert_eq!(stats.vectors_scored, 4);

            let candidates = [3u32, 1, 2];
            params.candidate_rows = candidates.as_ptr();
            params.candidate_count = candidates.len() as u64;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &params,
                    hits.as_mut_ptr(),
                    2,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_OK
            );
            assert_eq!(returned, 2);
            assert_eq!([hits[0].row_id, hits[1].row_id], [1, 2]);

            let duplicate_candidates = [3u32, 1, 1, 2];
            params.k = 3;
            params.candidate_rows = duplicate_candidates.as_ptr();
            params.candidate_count = duplicate_candidates.len() as u64;
            let mut hits = vec![
                ordvec_hit_t {
                    row_id: 0,
                    id: 0,
                    score: 0.0,
                    reserved: 0
                };
                3
            ];
            let mut stats = default_stats();
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &params,
                    hits.as_mut_ptr(),
                    3,
                    &mut returned,
                    &mut stats
                ),
                ORDVEC_STATUS_OK
            );
            assert_eq!(returned, 3);
            assert_eq!([hits[0].row_id, hits[1].row_id, hits[2].row_id], [1, 1, 2]);
            assert_eq!(stats.candidate_count, 4);
            assert_eq!(stats.vectors_scored, 4);
            ordvec_index_free(handle);
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn full_and_subset_search_bitmap() {
        let path = make_bitmap_fixture();
        unsafe {
            let handle = load_handle(&path);
            let mut query = vec![0.0f32; 64];
            for (j, value) in query.iter_mut().take(4).enumerate() {
                *value = 10.0 + j as f32;
            }
            let candidates = [3u32, 1, 1, 2];
            let mut params = default_params();
            params.query = query.as_ptr();
            params.dim = 64;
            params.k = 3;
            params.candidate_rows = candidates.as_ptr();
            params.candidate_count = candidates.len() as u64;
            let mut hits = vec![
                ordvec_hit_t {
                    row_id: 0,
                    id: 0,
                    score: 0.0,
                    reserved: 0
                };
                3
            ];
            let mut returned = 0;
            let mut stats = default_stats();
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &params,
                    hits.as_mut_ptr(),
                    3,
                    &mut returned,
                    &mut stats
                ),
                ORDVEC_STATUS_OK
            );
            assert_eq!(returned, 3);
            assert_eq!([hits[0].row_id, hits[1].row_id, hits[2].row_id], [1, 1, 2]);
            assert_eq!(stats.candidate_count, 4);
            assert_eq!(stats.vectors_scored, 4);
            ordvec_index_free(handle);
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn search_validation_failures() {
        let path = make_rankquant_fixture();
        unsafe {
            let handle = load_handle(&path);
            let query = [1.0f32; 16];
            let mut params = default_params();
            params.query = query.as_ptr();
            params.dim = 16;
            params.k = 1;
            let mut hits = [ordvec_hit_t {
                row_id: 0,
                id: 0,
                score: 0.0,
                reserved: 0,
            }];
            let mut returned = 0;

            assert_eq!(
                ordvec_index_search(
                    ptr::null(),
                    &params,
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_NULL_POINTER
            );
            assert_eq!(
                ordvec_index_search(
                    handle,
                    ptr::null(),
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_NULL_POINTER
            );
            let mut bad = params;
            bad.struct_size -= 1;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_BAD_STRUCT_SIZE
            );
            bad = params;
            bad.flags = 1;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_BAD_ARGUMENT
            );
            bad = params;
            bad.reserved[0] = 1;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_BAD_ARGUMENT
            );
            bad = params;
            bad.query = ptr::null();
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_NULL_POINTER
            );
            bad = params;
            bad.dim = 15;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_DIM_MISMATCH
            );
            let mut nonfinite = query;
            nonfinite[0] = f32::NAN;
            bad = params;
            bad.query = nonfinite.as_ptr();
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_NONFINITE_QUERY
            );
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &params,
                    hits.as_mut_ptr(),
                    1,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_NULL_POINTER
            );
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &params,
                    ptr::null_mut(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_NULL_POINTER
            );
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &params,
                    hits.as_mut_ptr(),
                    0,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_BUFFER_TOO_SMALL
            );
            let cands = [99u32];
            bad = params;
            bad.candidate_rows = cands.as_ptr();
            bad.candidate_count = 1;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    hits.as_mut_ptr(),
                    1,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_ROW_ID_OUT_OF_RANGE
            );
            bad = params;
            bad.candidate_rows = cands.as_ptr();
            bad.candidate_count = 0;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    ptr::null_mut(),
                    0,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_BAD_ARGUMENT
            );
            bad = params;
            bad.candidate_rows = ptr::null();
            bad.candidate_count = 1;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &bad,
                    ptr::null_mut(),
                    0,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_NULL_POINTER
            );
            ordvec_index_free(handle);
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn zero_required_hits_allows_null_hits() {
        let path = make_rankquant_fixture();
        unsafe {
            let handle = load_handle(&path);
            let query = [1.0f32; 16];
            let mut params = default_params();
            params.query = query.as_ptr();
            params.dim = 16;
            params.k = 0;
            let mut returned = 42;
            assert_eq!(
                ordvec_index_search(
                    handle,
                    &params,
                    ptr::null_mut(),
                    0,
                    &mut returned,
                    ptr::null_mut()
                ),
                ORDVEC_STATUS_OK
            );
            assert_eq!(returned, 0);
            ordvec_index_free(handle);
        }
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn load_maps_unsupported_and_corrupt_formats() {
        let rank_path = temp_path("rank", "tvr");
        let mut rank = Rank::new(16);
        rank.add(&[0.0f32; 16]);
        rank.write(&rank_path).unwrap();

        let sign_path = temp_path("sign", "tvsb");
        let mut sign = SignBitmap::new(64);
        sign.add(&[0.0f32; 64]);
        sign.write(&sign_path).unwrap();

        let corrupt_path = temp_path("corrupt", "tvrq");
        std::fs::File::create(&corrupt_path)
            .unwrap()
            .write_all(b"TVRQ\x01")
            .unwrap();

        unsafe {
            for path in [&rank_path, &sign_path] {
                let cpath = CString::new(path.to_str().unwrap()).unwrap();
                let mut out = ptr::null_mut();
                assert_eq!(
                    ordvec_index_load(cpath.as_ptr(), 0, &mut out),
                    ORDVEC_STATUS_UNSUPPORTED_FORMAT
                );
                assert!(out.is_null());
            }
            let cpath = CString::new(corrupt_path.to_str().unwrap()).unwrap();
            let mut out = ptr::null_mut();
            assert_eq!(
                ordvec_index_load(cpath.as_ptr(), 0, &mut out),
                ORDVEC_STATUS_CORRUPT_INDEX
            );
            assert!(out.is_null());
        }
        std::fs::remove_file(rank_path).ok();
        std::fs::remove_file(sign_path).ok();
        std::fs::remove_file(corrupt_path).ok();
    }
}
