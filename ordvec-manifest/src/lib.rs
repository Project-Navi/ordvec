//! Manifest verification for ordvec index artifacts.
//!
//! This crate verifies JSON manifests that bind an ordvec index file to
//! SHA-256 digests, probed loader metadata, row identity, caller-owned
//! auxiliary artifacts, optional encoder-distortion profiles, optional
//! calibration profiles, and attestation-shape metadata. It is intentionally a
//! verifier, not a trust oracle: it does not sign artifacts, manage keys, call
//! networks, mutate index files, estimate model geometry, or decide deployment
//! policy.
//!
//! Library callers can use [`load_manifest_file_with_options`] and
//! [`verify_document_for_load`], or use [`verify_for_load`] when they need a
//! verified snapshot of the canonical artifact path and related load metadata.
//! The `ordvec-manifest` binary exposes the same bounded verification surfaces
//! for command-line use.
//!
//! ```no_run
//! use ordvec::RankQuant;
//! use ordvec_manifest::{verify_for_load, ManifestIndexKind, VerifyOptions};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let plan = verify_for_load("quickstart.manifest.json", VerifyOptions::default())?;
//! println!(
//!     "verified {} rows at {}",
//!     plan.metadata().vector_count,
//!     plan.artifact_path().display()
//! );
//! let index = plan.decode_primary_with(
//!     ManifestIndexKind::RankQuant,
//!     |reader, encoded_len| RankQuant::read_from_sized(reader, encoded_len),
//! )?;
//! assert_eq!(index.len(), plan.metadata().vector_count);
//! # Ok(())
//! # }
//! ```
//!
//! See the crate README for a copy-and-run index creation and CLI verification
//! path.

use chrono::{DateTime, SecondsFormat, Utc};
use ordvec::{
    probe_index_metadata, FormatSpec, IndexKind as CoreIndexKind,
    IndexMetadata as CoreIndexMetadata, IndexParams as CoreIndexParams, ManifestCoverage, FORMATS,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
#[cfg(any(unix, windows))]
use std::fs::OpenOptions;
use std::fs::{self, File, Metadata};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;

pub const SCHEMA_VERSION: &str = "ordvec.index_manifest.v2";
pub const CALIBRATION_SCHEMA_VERSION: &str = "ordvec.calibration.v1";
pub const ENCODER_DISTORTION_SCHEMA_VERSION: &str = "ordvec.encoder_distortion.v1";
pub const DEFAULT_MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
pub const DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_ROW_IDENTITY_ROWS: usize = 10_000_000;
pub const DEFAULT_MAX_ROW_IDENTITY_TRACKED_DB_ID_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_MAX_AUXILIARY_ARTIFACTS: usize = 1024;
/// Artifact-file reads are bounded by the manifest-declared size on verify
/// and by the observed file size on create; these flat caps are opt-in
/// ceilings and default to unbounded. Streaming hashing keeps memory
/// constant regardless of artifact size.
pub const DEFAULT_MAX_AUXILIARY_ARTIFACT_BYTES: u64 = u64::MAX;
pub const DEFAULT_MAX_INDEX_ARTIFACT_BYTES: u64 = u64::MAX;
pub const DEFAULT_MAX_CALIBRATION_PROFILE_BYTES: u64 = u64::MAX;
pub const DEFAULT_MAX_ENCODER_DISTORTION_PROFILE_BYTES: u64 = u64::MAX;
pub const DEFAULT_MAX_REPORT_ISSUES: usize = 1024;
pub const DEFAULT_MAX_CACHED_REPORT_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug)]
pub enum ManifestError {
    Io(io::Error),
    Json(serde_json::Error),
    Invalid(String),
    LimitExceeded { code: String, message: String },
}

impl ManifestError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub fn limit_exceeded(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::LimitExceeded {
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn code(&self) -> Option<&str> {
        match self {
            Self::LimitExceeded { code, .. } => Some(code.as_str()),
            _ => None,
        }
    }
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Json(err) => write!(f, "{err}"),
            Self::Invalid(message) => f.write_str(message),
            Self::LimitExceeded { code, message } => write!(f, "{code}: {message}"),
        }
    }
}

impl std::error::Error for ManifestError {}

impl From<io::Error> for ManifestError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for ManifestError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Clone, Debug)]
pub struct ManifestDocument {
    pub manifest: IndexManifest,
    pub source_path: Option<PathBuf>,
    pub base_dir: PathBuf,
}

pub fn load_manifest_file(path: impl AsRef<Path>) -> Result<ManifestDocument, ManifestError> {
    load_manifest_file_with_options(path, &VerifyOptions::default())
}

pub fn load_manifest_file_with_options(
    path: impl AsRef<Path>,
    options: &VerifyOptions,
) -> Result<ManifestDocument, ManifestError> {
    let path = path.as_ref();
    let manifest_bytes = read_manifest_bytes_bounded(path, options.limits.max_manifest_bytes)?;
    let manifest = parse_current_manifest_bytes(&manifest_bytes)?;
    let base_dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();
    Ok(ManifestDocument {
        manifest,
        source_path: Some(path.to_path_buf()),
        base_dir,
    })
}

/// Read a manifest through a bounded, forward-only, final-component-safe file handle.
///
/// The caller-selected parent directory remains trusted. The final component is
/// opened without following a Unix/WASI symlink or Windows reparse point and
/// must be a regular file; Windows additionally requires a disk handle. At
/// most `max_bytes + 1` bytes are read, permitting an exact limit check without
/// an unbounded allocation.
pub fn read_manifest_bytes_bounded(
    path: impl AsRef<Path>,
    max_bytes: u64,
) -> Result<Vec<u8>, ManifestError> {
    read_bounded_file(
        path.as_ref(),
        max_bytes,
        codes::MANIFEST_FILE_TOO_LARGE,
        "manifest file",
    )
}

/// Strictly parse bytes as the current manifest schema.
///
/// Older or newer schema generations are rejected with schema-version context;
/// this function performs no filesystem access and no compatibility fallback.
pub fn parse_current_manifest_bytes(bytes: &[u8]) -> Result<IndexManifest, ManifestError> {
    validate_manifest_json_number_tokens(bytes)?;
    let manifest: IndexManifest =
        serde_json::from_slice(bytes).map_err(|err| manifest_parse_error(bytes, err))?;
    if manifest.schema_version != SCHEMA_VERSION {
        return Err(ManifestError::invalid(format!(
            "manifest declares schema_version {:?} but this build supports \
             {SCHEMA_VERSION:?}; the manifest was written by an older or newer \
             manifest schema",
            manifest.schema_version
        )));
    }
    Ok(manifest)
}

/// Reject JSON number tokens that `serde_json` could only retain by changing
/// their exact decimal value.
///
/// OrdinalDB-compatible dependency graphs deliberately use
/// `serde_json/float_roundtrip` and reject representation-changing features
/// such as `arbitrary_precision`. Under that profile, an oversized or overly
/// precise token can otherwise be rounded before it reaches a nested
/// [`serde_json::Value`]. Call this on the original bytes before deserializing
/// a compatible legacy schema so schema dispatch cannot introduce aliases.
pub fn validate_manifest_json_number_tokens(bytes: &[u8]) -> Result<(), ManifestError> {
    let mut offset = 0usize;
    while offset < bytes.len() {
        match bytes[offset] {
            b'"' => {
                offset += 1;
                while offset < bytes.len() {
                    match bytes[offset] {
                        b'\\' => {
                            // Escapes consume the following byte. Full escape
                            // syntax remains the JSON parser's responsibility;
                            // this scan only keeps number-looking string bytes
                            // out of the numeric-token path.
                            offset = offset.saturating_add(2);
                        }
                        b'"' => {
                            offset += 1;
                            break;
                        }
                        _ => offset += 1,
                    }
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = offset;
                offset += 1;
                while offset < bytes.len()
                    && !matches!(bytes[offset], b',' | b']' | b'}')
                    && !bytes[offset].is_ascii_whitespace()
                {
                    offset += 1;
                }
                let token = std::str::from_utf8(&bytes[start..offset]).map_err(|_| {
                    ManifestError::invalid("manifest contains a non-UTF-8 JSON number token")
                })?;
                validate_lossless_json_number_token(token)?;
            }
            _ => offset += 1,
        }
    }
    Ok(())
}

fn validate_lossless_json_number_token(token: &str) -> Result<(), ManifestError> {
    let identity = decimal_identity(token).ok_or_else(|| {
        ManifestError::invalid(format!(
            "manifest contains JSON number {token:?} that cannot be represented canonically"
        ))
    })?;

    // Integer syntax within serde_json's native signed/unsigned domain is
    // retained exactly without going through f64. Decimal/exponent syntax is
    // always parsed as f64, so it must pass the round-trip check below even
    // when its mathematical value happens to be integral.
    let integer_syntax = !token.bytes().any(|byte| matches!(byte, b'.' | b'e' | b'E'));
    if integer_syntax && canonical_integer_from_identity(&identity).is_some() {
        return Ok(());
    }

    let value = token.parse::<f64>().map_err(|_| {
        ManifestError::invalid(format!(
            "manifest contains JSON number {token:?} that cannot be represented canonically"
        ))
    })?;
    let normalized = serde_json::Number::from_f64(value).ok_or_else(|| {
        ManifestError::invalid(format!(
            "manifest contains JSON number {token:?} that cannot be represented canonically"
        ))
    })?;
    if decimal_identity(&normalized.to_string()).as_ref() != Some(&identity) {
        return Err(ManifestError::invalid(format!(
            "manifest contains JSON number {token:?} outside the exact canonical i64/u64/f64 domain"
        )));
    }
    Ok(())
}

/// Wraps a manifest parse failure with schema-version context. Old or new
/// schema generations fail the strict `deny_unknown_fields` parse before the
/// `schema_version` field is ever validated, so a targeted probe of that one
/// field is needed to say *why* the document does not parse.
fn manifest_parse_error(manifest_bytes: &[u8], err: serde_json::Error) -> ManifestError {
    #[derive(Deserialize)]
    struct SchemaVersionProbe {
        schema_version: Option<String>,
    }
    if let Ok(SchemaVersionProbe {
        schema_version: Some(version),
    }) = serde_json::from_slice::<SchemaVersionProbe>(manifest_bytes)
    {
        if version != SCHEMA_VERSION {
            return ManifestError::invalid(format!(
                "manifest declares schema_version {version:?} but this build supports \
                 {SCHEMA_VERSION:?}; the manifest was written by an older or newer \
                 manifest schema: {err}"
            ));
        }
    }
    ManifestError::Json(err)
}

/// Filesystem operation at which a plan-verified artifact became inaccessible.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifiedArtifactAccessStage {
    Open,
    InitialMetadata,
    Read,
    FinalMetadata,
}

impl fmt::Display for VerifiedArtifactAccessStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Open => f.write_str("open"),
            Self::InitialMetadata => f.write_str("initial metadata"),
            Self::Read => f.write_str("read"),
            Self::FinalMetadata => f.write_str("final metadata"),
        }
    }
}

/// Why an opened final path component is not an acceptable artifact file.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifiedArtifactTypeRejection {
    FinalSymlinkOrReparsePoint,
    NonRegularFile,
    NonDiskFile,
    UnsupportedPlatform,
}

impl fmt::Display for VerifiedArtifactTypeRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FinalSymlinkOrReparsePoint => {
                f.write_str("the final component is a symlink or reparse point")
            }
            Self::NonRegularFile => f.write_str("the opened component is not a regular file"),
            Self::NonDiskFile => f.write_str("the opened handle is not a disk file"),
            Self::UnsupportedPlatform => {
                f.write_str("the target platform cannot provide a no-follow final-component open")
            }
        }
    }
}

enum OpenRegularFileError {
    Access {
        stage: VerifiedArtifactAccessStage,
        source: io::Error,
    },
    Type(VerifiedArtifactTypeRejection),
}

#[cfg(unix)]
fn is_final_symlink_open_error(source: &io::Error) -> bool {
    let Some(code) = source.raw_os_error() else {
        return false;
    };
    if code == libc::ELOOP {
        return true;
    }
    #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
    if code == libc::EMLINK {
        return true;
    }
    #[cfg(target_os = "netbsd")]
    if code == libc::EFTYPE {
        return true;
    }
    false
}

#[cfg(unix)]
fn open_regular_file_no_follow(path: &Path) -> Result<(File, Metadata), OpenRegularFileError> {
    use std::os::unix::fs::OpenOptionsExt;

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|source| {
            if is_final_symlink_open_error(&source) {
                OpenRegularFileError::Type(
                    VerifiedArtifactTypeRejection::FinalSymlinkOrReparsePoint,
                )
            } else {
                OpenRegularFileError::Access {
                    stage: VerifiedArtifactAccessStage::Open,
                    source,
                }
            }
        })?;
    let metadata = file
        .metadata()
        .map_err(|source| OpenRegularFileError::Access {
            stage: VerifiedArtifactAccessStage::InitialMetadata,
            source,
        })?;
    if !metadata.file_type().is_file() {
        return Err(OpenRegularFileError::Type(
            VerifiedArtifactTypeRejection::NonRegularFile,
        ));
    }
    Ok((file, metadata))
}

#[cfg(windows)]
fn open_regular_file_no_follow(path: &Path) -> Result<(File, Metadata), OpenRegularFileError> {
    use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{GetLastError, SetLastError, ERROR_SUCCESS};
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileType, FILE_ATTRIBUTE_REPARSE_POINT, FILE_FLAG_BACKUP_SEMANTICS,
        FILE_FLAG_OPEN_REPARSE_POINT, FILE_TYPE_DISK,
    };

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
        .map_err(|source| OpenRegularFileError::Access {
            stage: VerifiedArtifactAccessStage::Open,
            source,
        })?;
    let metadata = file
        .metadata()
        .map_err(|source| OpenRegularFileError::Access {
            stage: VerifiedArtifactAccessStage::InitialMetadata,
            source,
        })?;
    if metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
        return Err(OpenRegularFileError::Type(
            VerifiedArtifactTypeRejection::FinalSymlinkOrReparsePoint,
        ));
    }
    // `GetFileType` uses the last-error slot to disambiguate a legitimate
    // FILE_TYPE_UNKNOWN result, but it need not clear a stale value on
    // success. Clear it immediately before the query.
    // SAFETY: these calls only access the current thread's last-error slot;
    // `file` owns a live HANDLE for the duration of `GetFileType`.
    unsafe { SetLastError(ERROR_SUCCESS) };
    let file_type = unsafe { GetFileType(file.as_raw_handle().cast()) };
    if file_type == windows_sys::Win32::Storage::FileSystem::FILE_TYPE_UNKNOWN {
        // `FILE_TYPE_UNKNOWN` can be a legitimate handle classification or a
        // failed query. Windows requires consulting the thread-local last
        // error immediately to distinguish the two cases.
        // SAFETY: `GetLastError` reads the current thread's last-error value.
        let code = unsafe { GetLastError() };
        if code != ERROR_SUCCESS {
            return Err(OpenRegularFileError::Access {
                stage: VerifiedArtifactAccessStage::InitialMetadata,
                source: io::Error::from_raw_os_error(code as i32),
            });
        }
    }
    if file_type != FILE_TYPE_DISK {
        return Err(OpenRegularFileError::Type(
            VerifiedArtifactTypeRejection::NonDiskFile,
        ));
    }
    if !metadata.file_type().is_file() {
        return Err(OpenRegularFileError::Type(
            VerifiedArtifactTypeRejection::NonRegularFile,
        ));
    }
    Ok((file, metadata))
}

#[cfg(target_os = "wasi")]
fn open_regular_file_no_follow(path: &Path) -> Result<(File, Metadata), OpenRegularFileError> {
    use rustix::fs::{openat, Mode, OFlags, CWD};

    let descriptor = openat(
        CWD,
        path,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK,
        Mode::empty(),
    )
    .map_err(|error| {
        if error == rustix::io::Errno::LOOP {
            OpenRegularFileError::Type(VerifiedArtifactTypeRejection::FinalSymlinkOrReparsePoint)
        } else {
            OpenRegularFileError::Access {
                stage: VerifiedArtifactAccessStage::Open,
                source: io::Error::from_raw_os_error(error.raw_os_error()),
            }
        }
    })?;
    let file = File::from(descriptor);
    let metadata = file
        .metadata()
        .map_err(|source| OpenRegularFileError::Access {
            stage: VerifiedArtifactAccessStage::InitialMetadata,
            source,
        })?;
    if !metadata.file_type().is_file() {
        return Err(OpenRegularFileError::Type(
            VerifiedArtifactTypeRejection::NonRegularFile,
        ));
    }
    Ok((file, metadata))
}

#[cfg(not(any(unix, windows, target_os = "wasi")))]
fn open_regular_file_no_follow(_path: &Path) -> Result<(File, Metadata), OpenRegularFileError> {
    Err(OpenRegularFileError::Type(
        VerifiedArtifactTypeRejection::UnsupportedPlatform,
    ))
}

fn read_bounded_file(
    path: &Path,
    max_bytes: u64,
    code: &'static str,
    context: &'static str,
) -> Result<Vec<u8>, ManifestError> {
    let (mut file, _) = open_regular_file_no_follow(path).map_err(|err| match err {
        OpenRegularFileError::Access { source, .. } => ManifestError::Io(source),
        OpenRegularFileError::Type(rejection) => ManifestError::invalid(format!(
            "{} is not an acceptable regular manifest file: {rejection}",
            path.display()
        )),
    })?;
    let max_len = usize::try_from(max_bytes).map_err(|_| {
        ManifestError::limit_exceeded(
            code,
            format!(
                "{context} byte limit {max_bytes} is too large to enforce while reading {}",
                path.display()
            ),
        )
    })?;
    let read_limit = max_bytes.checked_add(1).ok_or_else(|| {
        ManifestError::limit_exceeded(
            code,
            format!(
                "{context} byte limit {max_bytes} is too large to enforce while reading {}",
                path.display()
            ),
        )
    })?;
    let mut bytes = Vec::new();
    let mut limited = Read::by_ref(&mut file).take(read_limit);
    limited.read_to_end(&mut bytes)?;
    if bytes.len() > max_len {
        return Err(ManifestError::limit_exceeded(
            code,
            format!(
                "{context} exceeds {max_bytes} bytes while reading {}",
                path.display()
            ),
        ));
    }
    Ok(bytes)
}

pub fn verify_manifest_with_base(
    manifest: IndexManifest,
    base_dir: impl Into<PathBuf>,
    options: VerifyOptions,
) -> VerificationReport {
    let document = ManifestDocument {
        manifest,
        source_path: None,
        base_dir: base_dir.into(),
    };
    verify_manifest(&document, options)
}

pub fn verify_index_manifest(
    index_path: impl Into<PathBuf>,
    manifest_path: impl AsRef<Path>,
    mut options: VerifyOptions,
) -> Result<VerificationReport, ManifestError> {
    let document = load_manifest_file_with_options(manifest_path, &options)?;
    options.index_override = Some(index_path.into());
    Ok(verify_manifest(&document, options))
}

/// Verifies a manifest file and returns a typed plan for caller-side loading.
///
/// The returned [`VerifiedLoadPlan`] is a verification snapshot: it contains
/// canonical paths, probed metadata, row identity, auxiliary artifact states,
/// and the full report for the bytes observed during this call. It is not a
/// lease, file lock, mmap, open descriptor, or durable byte pin. If backing
/// files can change between verification and load, re-verify immediately before
/// loading, load from immutable storage, or use a caller-owned loading path that
/// pins bytes.
pub fn verify_for_load(
    manifest_path: impl AsRef<Path>,
    options: VerifyOptions,
) -> Result<VerifiedLoadPlan, VerifiedLoadPlanError> {
    let document = load_manifest_file_with_options(manifest_path, &options)?;
    verify_document_for_load(&document, options)
}

/// Verifies an already-loaded manifest document and returns a typed load plan.
///
/// This has the same snapshot boundary as [`verify_for_load`]: it resolves and
/// verifies paths at call time, but it does not pin the verified bytes against
/// later mutation.
pub fn verify_document_for_load(
    document: &ManifestDocument,
    options: VerifyOptions,
) -> Result<VerifiedLoadPlan, VerifiedLoadPlanError> {
    let (report, paths) = verify_manifest_with_path_capture(document, options);
    VerifiedLoadPlan::from_report(document, report, paths)
}

pub fn verify_manifest(document: &ManifestDocument, options: VerifyOptions) -> VerificationReport {
    verify_manifest_with_path_capture(document, options).0
}

fn verify_manifest_with_path_capture(
    document: &ManifestDocument,
    options: VerifyOptions,
) -> (VerificationReport, VerificationPathCapture) {
    let mut paths = VerificationPathCapture::default();
    let mut report = VerificationReport::new();
    validate_manifest_shape(&document.manifest, &options, &mut report);

    let artifact_display_path = document.manifest.artifact.path.clone();
    report.artifact.manifest_path = Some(artifact_display_path.clone());
    let artifact_path = options
        .index_override
        .as_ref()
        .cloned()
        .unwrap_or_else(|| PathBuf::from(&document.manifest.artifact.path));
    report.artifact.observed_path = Some(path_to_display(&artifact_path));

    if let Some(resolved) = resolve_existing_path(
        &artifact_path,
        &document.base_dir,
        &options,
        &ARTIFACT_PATH_ISSUES,
        &mut report.errors,
    ) {
        paths.artifact_path = Some(resolved.canonical_path.clone());
        report.artifact.canonical_path = Some(path_to_display(&resolved.canonical_path));
        // Bound the read by the manifest-declared size: a primary artifact
        // larger than its declaration fails fast instead of being hashed in
        // full (the read was previously unbounded).
        match sha256_file_bounded(
            &resolved.canonical_path,
            document
                .manifest
                .artifact
                .file_size_bytes
                .min(options.limits.max_index_artifact_bytes),
            codes::ARTIFACT_FILE_TOO_LARGE,
            "index artifact",
        ) {
            Ok(hash) => {
                report.artifact.sha256 = Some(hash.sha256.clone());
                report.artifact.size_bytes = Some(hash.size_bytes);
                if !hex_digest_eq(&hash.sha256, &document.manifest.artifact.sha256) {
                    report.errors.push(
                        ReportIssue::new(
                            codes::ARTIFACT_SHA256_MISMATCH,
                            format!(
                                "artifact SHA-256 was {}, manifest declares {}",
                                hash.sha256, document.manifest.artifact.sha256
                            ),
                        )
                        .with_sha256_detail(
                            document.manifest.artifact.sha256.as_str(),
                            hash.sha256.as_str(),
                        ),
                    );
                }
                if hash.size_bytes != document.manifest.artifact.file_size_bytes {
                    report.errors.push(
                        ReportIssue::new(
                            codes::ARTIFACT_FILE_SIZE_MISMATCH,
                            format!(
                                "artifact size was {}, manifest declares {}",
                                hash.size_bytes, document.manifest.artifact.file_size_bytes
                            ),
                        )
                        .with_size_detail(
                            document.manifest.artifact.file_size_bytes,
                            hash.size_bytes,
                        ),
                    );
                }
            }
            Err(ManifestError::LimitExceeded { code, message }) => report.error(code, message),
            Err(err) => report.error(
                codes::ARTIFACT_HASH_FAILED,
                format!("failed to hash artifact: {err}"),
            ),
        }

        match probe_index_metadata(&resolved.canonical_path) {
            Ok(metadata) => {
                if let Ok(metadata_report) = MetadataReport::try_from_core(&metadata) {
                    report.artifact.metadata = Some(metadata_report);
                }
                compare_artifact_metadata(&document.manifest.artifact, &metadata, &mut report);
            }
            Err(err) => report.error(
                codes::ARTIFACT_PROBE_FAILED,
                format!("failed to probe artifact metadata: {err}"),
            ),
        }
    }

    verify_auxiliary_artifacts(document, &options, &mut report, &mut paths);
    verify_row_identity(document, &options, &mut report, &mut paths);
    verify_encoder_distortion(document, &options, &mut report);
    verify_calibration(document, &options, &mut report);
    verify_attestations(&document.manifest, &mut report);

    enforce_report_issue_limit(&mut report.errors, &options.limits);
    report.ok = report.errors.is_empty();
    (report, paths)
}

fn validate_manifest_shape(
    manifest: &IndexManifest,
    options: &VerifyOptions,
    report: &mut VerificationReport,
) {
    if manifest.schema_version != SCHEMA_VERSION {
        report.error(
            codes::SCHEMA_VERSION_UNSUPPORTED,
            format!(
                "schema_version must be {SCHEMA_VERSION}, got {}",
                manifest.schema_version
            ),
        );
    }
    if manifest.embedding.model.trim().is_empty() {
        report.error(
            codes::EMBEDDING_MODEL_EMPTY,
            "embedding.model must be non-empty",
        );
    }
    if manifest.embedding.dim == 0 {
        report.error(
            codes::EMBEDDING_DIM_ZERO,
            "embedding.dim must be greater than zero",
        );
    }
    if manifest.artifact.path.trim().is_empty() {
        report.error(
            codes::ARTIFACT_PATH_EMPTY,
            "artifact.path must be non-empty",
        );
    } else if !is_manifest_path_absolute(&manifest.artifact.path)
        && !is_canonical_manifest_path(&manifest.artifact.path, options.allow_path_escape)
    {
        report.error(
            codes::ARTIFACT_PATH_NOT_CANONICAL,
            "artifact.path must use forward slashes with no `.`, `..`, or empty segments",
        );
    }
    if !is_sha256_hex(&manifest.artifact.sha256) {
        report.error(
            codes::ARTIFACT_SHA256_INVALID,
            "artifact.sha256 must be a lowercase 64-character hex SHA-256 digest",
        );
    }
    if manifest.artifact.file_size_bytes == 0 {
        report.error(
            codes::ARTIFACT_FILE_SIZE_ZERO,
            "artifact.file_size_bytes must be greater than zero",
        );
    }
    if manifest.artifact.bytes_per_vec == 0 {
        report.error(
            codes::ARTIFACT_BYTES_PER_VEC_ZERO,
            "artifact.bytes_per_vec must be greater than zero",
        );
    }
    if manifest.artifact.dim != manifest.embedding.dim {
        report.error(
            codes::ARTIFACT_EMBEDDING_DIM_MISMATCH,
            format!(
                "artifact.dim {} does not match embedding.dim {}",
                manifest.artifact.dim, manifest.embedding.dim
            ),
        );
    }
    if !artifact_kind_matches_params(manifest.artifact.kind, &manifest.artifact.params) {
        report.error(
            codes::ARTIFACT_PARAMS_KIND_MISMATCH,
            "artifact.params discriminator does not match artifact.kind",
        );
    }

    let row_count = manifest.row_identity.row_count();
    if manifest.artifact.vector_count != row_count {
        report.error(
            codes::ARTIFACT_ROW_COUNT_MISMATCH,
            format!(
                "artifact.vector_count {} does not match row_identity.row_count {}",
                manifest.artifact.vector_count, row_count
            ),
        );
    }
    if let RowIdentity::Jsonl {
        path,
        sha256,
        id_kind,
        db,
        ..
    } = &manifest.row_identity
    {
        if path.trim().is_empty() {
            report.error(
                codes::ROW_IDENTITY_PATH_EMPTY,
                "row_identity.path must be non-empty",
            );
        } else if !is_manifest_path_absolute(path)
            && !is_canonical_manifest_path(path, options.allow_path_escape)
        {
            report.error(
                codes::ROW_IDENTITY_PATH_NOT_CANONICAL,
                "row_identity.path must use forward slashes with no `.`, `..`, or empty segments",
            );
        }
        if !is_sha256_hex(sha256) {
            report.error(
                codes::ROW_IDENTITY_SHA256_INVALID,
                "row_identity.sha256 must be a lowercase 64-character hex SHA-256 digest",
            );
        }
        if id_kind != "uuid" {
            report.error(
                codes::ROW_IDENTITY_ID_KIND_UNSUPPORTED,
                "row_identity.id_kind must be uuid in v1",
            );
        }
        if db.is_some() {
            report.error(
                codes::ROW_IDENTITY_DB_UNSUPPORTED,
                "row_identity.db is reserved for a future schema and is not verified in v1",
            );
        }
    }

    validate_auxiliary_artifact_shape(manifest, options, report);

    validate_optional_non_empty(
        codes::EMBEDDING_MODEL_REVISION_EMPTY,
        "embedding.model_revision must be non-empty when present",
        manifest.embedding.model_revision.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::EMBEDDING_TOKENIZER_REVISION_EMPTY,
        "embedding.tokenizer_revision must be non-empty when present",
        manifest.embedding.tokenizer_revision.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::EMBEDDING_POOLING_EMPTY,
        "embedding.pooling must be non-empty when present",
        manifest.embedding.pooling.as_deref(),
        report,
    );
    validate_optional_sha256(
        codes::EMBEDDING_CORPUS_DIGEST_INVALID,
        "embedding.corpus_digest must be a lowercase 64-character hex SHA-256 digest",
        manifest.embedding.corpus_digest.as_deref(),
        report,
    );
    validate_optional_sha256(
        codes::EMBEDDING_MATRIX_DIGEST_INVALID,
        "embedding.embedding_matrix_digest must be a lowercase 64-character hex SHA-256 digest",
        manifest.embedding.embedding_matrix_digest.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::EMBEDDING_NORMALIZATION_EMPTY,
        "embedding.normalization must be non-empty when present",
        manifest.embedding.normalization.as_deref(),
        report,
    );

    if let Some(build) = &manifest.build {
        if build.invocation_id.trim().is_empty() {
            report.error(
                codes::BUILD_INVOCATION_ID_EMPTY,
                "build.invocation_id must be non-empty",
            );
        }
        if build
            .builder_id
            .as_ref()
            .is_some_and(|builder_id| builder_id.trim().is_empty())
        {
            report.error(
                codes::BUILD_BUILDER_ID_EMPTY,
                "build.builder_id must be non-empty",
            );
        }
        validate_optional_non_empty(
            codes::BUILD_SOURCE_REPO_EMPTY,
            "build.source_repo must be non-empty when present",
            build.source_repo.as_deref(),
            report,
        );
        validate_optional_non_empty(
            codes::BUILD_SOURCE_COMMIT_EMPTY,
            "build.source_commit must be non-empty when present",
            build.source_commit.as_deref(),
            report,
        );
        validate_optional_non_empty(
            codes::BUILD_CI_PROVIDER_EMPTY,
            "build.ci_provider must be non-empty when present",
            build.ci_provider.as_deref(),
            report,
        );
        validate_optional_non_empty(
            codes::BUILD_CI_RUN_ID_EMPTY,
            "build.ci_run_id must be non-empty when present",
            build.ci_run_id.as_deref(),
            report,
        );
    }

    for key in manifest.extensions.keys() {
        if !extension_key_is_namespaced(key) {
            report.error(
                codes::EXTENSION_KEY_NOT_NAMESPACED,
                format!("extension key {key:?} must be namespaced"),
            );
        }
    }
}

fn validate_auxiliary_artifact_shape(
    manifest: &IndexManifest,
    options: &VerifyOptions,
    report: &mut VerificationReport,
) {
    if !check_auxiliary_artifact_count(manifest, &options.limits, report) {
        return;
    }
    let mut names = HashSet::new();
    for artifact in &manifest.auxiliary_artifacts {
        let name = artifact.name.trim();
        if name.is_empty() {
            report.error(
                codes::AUXILIARY_ARTIFACT_NAME_EMPTY,
                "auxiliary artifact name must be non-empty",
            );
        } else if artifact.name != name {
            report.error(
                codes::AUXILIARY_ARTIFACT_NAME_NOT_TRIMMED,
                format!(
                    "auxiliary artifact name {name:?} must not have leading or trailing whitespace"
                ),
            );
        } else if !names.insert(name.to_string()) {
            report.error(
                codes::AUXILIARY_ARTIFACT_NAME_DUPLICATE,
                format!("auxiliary artifact name {name:?} is duplicated"),
            );
        }

        if artifact.path.trim().is_empty() {
            report.error(
                codes::AUXILIARY_ARTIFACT_PATH_EMPTY,
                format!("auxiliary artifact {name:?} path must be non-empty"),
            );
        } else if !is_manifest_path_absolute(&artifact.path)
            && !is_canonical_manifest_path(&artifact.path, options.allow_path_escape)
        {
            report.error(
                codes::AUXILIARY_ARTIFACT_PATH_NOT_CANONICAL,
                format!(
                    "auxiliary artifact {name:?} path must use forward slashes with no `.`, `..`, or empty segments"
                ),
            );
        }
        if !is_sha256_hex(&artifact.sha256) {
            report.error(
                codes::AUXILIARY_ARTIFACT_SHA256_INVALID,
                format!(
                    "auxiliary artifact {name:?} sha256 must be a lowercase 64-character hex SHA-256 digest"
                ),
            );
        }
        // Optional artifacts may legitimately be declared absent with a
        // zero-size placeholder (see `AuxiliaryArtifactState::OptionalAbsent`);
        // only required declarations must carry a real size.
        if artifact.required && artifact.file_size_bytes == 0 {
            report.error(
                codes::AUXILIARY_ARTIFACT_FILE_SIZE_ZERO,
                format!(
                    "required auxiliary artifact {name:?} file_size_bytes must be greater than zero"
                ),
            );
        }
    }
}

fn validate_optional_non_empty(
    code: &str,
    message: &str,
    value: Option<&str>,
    report: &mut VerificationReport,
) {
    if value.is_some_and(|value| value.trim().is_empty()) {
        report.error(code, message);
    }
}

fn validate_optional_sha256(
    code: &str,
    message: &str,
    value: Option<&str>,
    report: &mut VerificationReport,
) {
    if value.is_some_and(|value| !is_sha256_hex(value)) {
        report.error(code, message);
    }
}

fn validate_optional_sha256_uri(
    code: &str,
    message: &str,
    value: Option<&str>,
    report: &mut VerificationReport,
) {
    let Some(value) = value else {
        return;
    };
    let Some(digest) = value.strip_prefix("sha256:") else {
        report.error(code, message);
        return;
    };
    if !is_sha256_hex(digest) {
        report.error(code, message);
    }
}

fn validate_optional_positive_f64(
    code: &str,
    message: &str,
    value: Option<f64>,
    report: &mut VerificationReport,
) {
    if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        report.error(code, message);
    }
}

fn validate_optional_nonnegative_f64(
    code: &str,
    message: &str,
    value: Option<f64>,
    report: &mut VerificationReport,
) {
    if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
        report.error(code, message);
    }
}

fn validate_optional_probability(
    code: &str,
    message: &str,
    value: Option<f64>,
    report: &mut VerificationReport,
) {
    if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        report.error(code, message);
    }
}

fn artifact_kind_matches_params(kind: ManifestIndexKind, params: &ManifestIndexParams) -> bool {
    matches!(
        (kind, params),
        (ManifestIndexKind::Rank, ManifestIndexParams::Rank)
            | (
                ManifestIndexKind::RankQuant,
                ManifestIndexParams::RankQuant { .. }
            )
            | (
                ManifestIndexKind::Bitmap,
                ManifestIndexParams::Bitmap { .. }
            )
            | (
                ManifestIndexKind::SignBitmap,
                ManifestIndexParams::SignBitmap
            )
    )
}

fn compare_artifact_metadata(
    artifact: &Artifact,
    metadata: &CoreIndexMetadata,
    report: &mut VerificationReport,
) {
    match ManifestIndexKind::try_from_core(metadata.kind) {
        Ok(observed_kind) => {
            if artifact.kind != observed_kind {
                report.error(
                    codes::ARTIFACT_KIND_MISMATCH,
                    format!(
                        "artifact kind was {:?}, manifest declares {:?}",
                        observed_kind, artifact.kind
                    ),
                );
            }
        }
        Err(err) => report.error(err.code(), err.message()),
    }
    match ManifestIndexParams::try_from_core(metadata.params) {
        Ok(observed_params) => {
            if artifact.params != observed_params {
                report.error(
                    codes::ARTIFACT_PARAMS_MISMATCH,
                    format!(
                        "artifact params were {:?}, manifest declares {:?}",
                        observed_params, artifact.params
                    ),
                );
            }
        }
        Err(err) => report.error(err.code(), err.message()),
    }
    if artifact.format_version != metadata.format_version {
        report.error(
            codes::ARTIFACT_FORMAT_VERSION_MISMATCH,
            format!(
                "artifact format_version was {}, manifest declares {}",
                metadata.format_version, artifact.format_version
            ),
        );
    }
    if artifact.dim != metadata.dim {
        report.error(
            codes::ARTIFACT_DIM_MISMATCH,
            format!(
                "artifact dim was {}, manifest declares {}",
                metadata.dim, artifact.dim
            ),
        );
    }
    if artifact.vector_count != metadata.vector_count {
        report.error(
            codes::ARTIFACT_VECTOR_COUNT_MISMATCH,
            format!(
                "artifact vector_count was {}, manifest declares {}",
                metadata.vector_count, artifact.vector_count
            ),
        );
    }
    if artifact.bytes_per_vec != metadata.bytes_per_vec {
        report.error(
            codes::ARTIFACT_BYTES_PER_VEC_MISMATCH,
            format!(
                "artifact bytes_per_vec was {}, manifest declares {}",
                metadata.bytes_per_vec, artifact.bytes_per_vec
            ),
        );
    }
    if artifact.file_size_bytes != metadata.file_size_bytes {
        report.error(
            codes::ARTIFACT_METADATA_FILE_SIZE_MISMATCH,
            format!(
                "artifact metadata file_size_bytes was {}, manifest declares {}",
                metadata.file_size_bytes, artifact.file_size_bytes
            ),
        );
    }
}

fn verify_row_identity(
    document: &ManifestDocument,
    options: &VerifyOptions,
    report: &mut VerificationReport,
    paths: &mut VerificationPathCapture,
) {
    match &document.manifest.row_identity {
        RowIdentity::RowIdIdentity { row_count } => {
            report.row_identity.kind = Some("row_id_identity".to_string());
            report.row_identity.row_count = Some(*row_count);
        }
        RowIdentity::Jsonl {
            path,
            sha256,
            row_count,
            ..
        } => {
            report.row_identity.kind = Some("jsonl".to_string());
            report.row_identity.manifest_path = Some(path.clone());
            report.row_identity.row_count = Some(*row_count);
            if *row_count > options.limits.max_row_identity_rows {
                report.error(
                    codes::ROW_IDENTITY_ROW_COUNT_LIMIT_EXCEEDED,
                    format!(
                        "row_identity.row_count {row_count} exceeds max_row_identity_rows={}",
                        options.limits.max_row_identity_rows
                    ),
                );
                return;
            }
            let row_path = PathBuf::from(path);
            if let Some(resolved) = resolve_existing_path(
                &row_path,
                &document.base_dir,
                options,
                &ROW_IDENTITY_PATH_ISSUES,
                &mut report.errors,
            ) {
                paths.row_identity_path = Some(resolved.canonical_path.clone());
                report.row_identity.canonical_path =
                    Some(path_to_display(&resolved.canonical_path));
                match validate_jsonl_rows(
                    &resolved.canonical_path,
                    options.allow_duplicate_db_ids,
                    &options.limits,
                    Some(*row_count),
                    &mut report.errors,
                ) {
                    Ok(stats) => {
                        report.row_identity.validated_rows = Some(stats.validated_rows);
                        if let Some(hash) = &stats.sha256 {
                            report.row_identity.sha256 = Some(hash.clone());
                            if !hex_digest_eq(hash, sha256) {
                                report.errors.push(
                                    ReportIssue::new(
                                        codes::ROW_IDENTITY_SHA256_MISMATCH,
                                        format!(
                                            "row_identity SHA-256 was {hash}, manifest declares {sha256}"
                                        ),
                                    )
                                    .with_sha256_detail(sha256.as_str(), hash.as_str()),
                                );
                            }
                        }
                        if stats.row_count != *row_count
                            && !report
                                .errors
                                .iter()
                                .any(|issue| issue.code == codes::ROW_IDENTITY_ROW_COUNT_MISMATCH)
                        {
                            let observed_rows = if stats.sha256.is_some() {
                                stats.row_count.to_string()
                            } else {
                                format!("at least {}", stats.row_count)
                            };
                            report.error(
                                codes::ROW_IDENTITY_ROW_COUNT_MISMATCH,
                                format!(
                                    "row identity file has {observed_rows} rows, manifest declares {row_count}"
                                ),
                            );
                        }
                    }
                    Err(err) => report.error(
                        codes::ROW_IDENTITY_READ_FAILED,
                        format!("failed to read row identity file: {err}"),
                    ),
                }
            }
        }
    }
}

fn verify_encoder_distortion(
    document: &ManifestDocument,
    options: &VerifyOptions,
    report: &mut VerificationReport,
) {
    let Some(profile) = &document.manifest.encoder_distortion else {
        return;
    };

    report.encoder_distortion.present = true;
    report.encoder_distortion.schema_version = Some(profile.schema_version.clone());
    report.encoder_distortion.profile_id = Some(profile.profile_id.clone());
    report.encoder_distortion.evidence_kind = Some(profile.evidence.kind.label().to_string());
    report.encoder_distortion.source_metric = Some(profile.source_metric.name.clone());
    report.encoder_distortion.embedding_metric = Some(profile.embedding_metric.name.clone());

    validate_encoder_distortion_shape(profile, report);
    validate_encoder_distortion_encoder(profile, &document.manifest.embedding, report);
    validate_encoder_distortion_metrics(profile, report);
    validate_encoder_distortion_bounds(&profile.bounds, report);
    validate_encoder_distortion_scope(&profile.scope, report);
    validate_encoder_distortion_evidence(profile, &document.base_dir, options, report);
    validate_encoder_distortion_calibration(
        profile,
        document.manifest.calibration.as_ref(),
        report,
    );
}

fn validate_encoder_distortion_shape(
    profile: &EncoderDistortionProfileRef,
    report: &mut VerificationReport,
) {
    if profile.schema_version != ENCODER_DISTORTION_SCHEMA_VERSION {
        report.error(
            codes::ENCODER_DISTORTION_SCHEMA_VERSION_UNSUPPORTED,
            format!(
                "encoder_distortion.schema_version must be {ENCODER_DISTORTION_SCHEMA_VERSION}, got {}",
                profile.schema_version
            ),
        );
    }
    if profile.profile_id.trim().is_empty() {
        report.error(
            codes::ENCODER_DISTORTION_PROFILE_ID_EMPTY,
            "encoder_distortion.profile_id must be non-empty",
        );
    }
    if profile
        .created_at
        .as_ref()
        .is_some_and(|created_at| DateTime::parse_from_rfc3339(created_at).is_err())
    {
        report.error(
            codes::ENCODER_DISTORTION_CREATED_AT_INVALID,
            "encoder_distortion.created_at must parse as RFC3339 when present",
        );
    }
    if profile.encoder.model.trim().is_empty() {
        report.error(
            codes::ENCODER_DISTORTION_ENCODER_MODEL_EMPTY,
            "encoder_distortion.encoder.model must be non-empty",
        );
    }
    if profile.encoder.dim == 0 {
        report.error(
            codes::ENCODER_DISTORTION_ENCODER_DIM_ZERO,
            "encoder_distortion.encoder.dim must be greater than zero",
        );
    }
    validate_optional_non_empty(
        codes::ENCODER_DISTORTION_ENCODER_MODEL_REVISION_EMPTY,
        "encoder_distortion.encoder.model_revision must be non-empty when present",
        profile.encoder.model_revision.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::ENCODER_DISTORTION_ENCODER_NORMALIZATION_EMPTY,
        "encoder_distortion.encoder.normalization must be non-empty when present",
        profile.encoder.normalization.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::ENCODER_DISTORTION_TOKENIZER_REVISION_EMPTY,
        "encoder_distortion.tokenizer_revision must be non-empty when present",
        profile.tokenizer_revision.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::ENCODER_DISTORTION_POOLING_EMPTY,
        "encoder_distortion.pooling must be non-empty when present",
        profile.pooling.as_deref(),
        report,
    );
}

fn validate_encoder_distortion_encoder(
    profile: &EncoderDistortionProfileRef,
    embedding: &Embedding,
    report: &mut VerificationReport,
) {
    if profile.encoder.model != embedding.model {
        report.error(
            codes::ENCODER_DISTORTION_ENCODER_MODEL_MISMATCH,
            format!(
                "encoder_distortion model {:?} does not match embedding.model {:?}",
                profile.encoder.model, embedding.model
            ),
        );
    }
    if profile.encoder.dim != embedding.dim {
        report.error(
            codes::ENCODER_DISTORTION_ENCODER_DIM_MISMATCH,
            format!(
                "encoder_distortion dim {} does not match embedding.dim {}",
                profile.encoder.dim, embedding.dim
            ),
        );
    }
    compare_optional_encoder_identity(
        codes::ENCODER_DISTORTION_ENCODER_MODEL_REVISION_MISMATCH,
        "encoder_distortion encoder",
        "model_revision",
        embedding.model_revision.as_deref(),
        profile.encoder.model_revision.as_deref(),
        report,
    );
    compare_optional_encoder_identity(
        codes::ENCODER_DISTORTION_ENCODER_NORMALIZATION_MISMATCH,
        "encoder_distortion encoder",
        "normalization",
        embedding.normalization.as_deref(),
        profile.encoder.normalization.as_deref(),
        report,
    );
    compare_optional_encoder_identity(
        codes::ENCODER_DISTORTION_TOKENIZER_REVISION_MISMATCH,
        "encoder_distortion",
        "tokenizer_revision",
        embedding.tokenizer_revision.as_deref(),
        profile.tokenizer_revision.as_deref(),
        report,
    );
    compare_optional_encoder_identity(
        codes::ENCODER_DISTORTION_POOLING_MISMATCH,
        "encoder_distortion",
        "pooling",
        embedding.pooling.as_deref(),
        profile.pooling.as_deref(),
        report,
    );
}

fn validate_encoder_distortion_metrics(
    profile: &EncoderDistortionProfileRef,
    report: &mut VerificationReport,
) {
    validate_metric_spec(
        "encoder_distortion_source_metric",
        &SOURCE_METRIC_ISSUES,
        &profile.source_metric,
        report,
    );
    validate_metric_spec(
        "encoder_distortion_embedding_metric",
        &EMBEDDING_METRIC_ISSUES,
        &profile.embedding_metric,
        report,
    );
}

/// Per-metric issue codes for [`validate_metric_spec`], so every emitted
/// code stays a named constant in [`codes`].
struct MetricSpecIssueCodes {
    name_empty: &'static str,
    version_empty: &'static str,
    digest_invalid: &'static str,
}

const SOURCE_METRIC_ISSUES: MetricSpecIssueCodes = MetricSpecIssueCodes {
    name_empty: codes::ENCODER_DISTORTION_SOURCE_METRIC_NAME_EMPTY,
    version_empty: codes::ENCODER_DISTORTION_SOURCE_METRIC_VERSION_EMPTY,
    digest_invalid: codes::ENCODER_DISTORTION_SOURCE_METRIC_DIGEST_INVALID,
};

const EMBEDDING_METRIC_ISSUES: MetricSpecIssueCodes = MetricSpecIssueCodes {
    name_empty: codes::ENCODER_DISTORTION_EMBEDDING_METRIC_NAME_EMPTY,
    version_empty: codes::ENCODER_DISTORTION_EMBEDDING_METRIC_VERSION_EMPTY,
    digest_invalid: codes::ENCODER_DISTORTION_EMBEDDING_METRIC_DIGEST_INVALID,
};

fn validate_metric_spec(
    prefix: &str,
    issue_codes: &MetricSpecIssueCodes,
    metric: &MetricSpec,
    report: &mut VerificationReport,
) {
    if metric.name.trim().is_empty() {
        report.error(
            issue_codes.name_empty,
            format!("{prefix}.name must be non-empty"),
        );
    }
    validate_optional_non_empty(
        issue_codes.version_empty,
        &format!("{prefix}.version must be non-empty when present"),
        metric.version.as_deref(),
        report,
    );
    validate_optional_sha256_uri(
        issue_codes.digest_invalid,
        &format!("{prefix}.digest must be sha256:<lowercase-hex> when present"),
        metric.digest.as_deref(),
        report,
    );
}

fn validate_encoder_distortion_bounds(bounds: &DistortionBounds, report: &mut VerificationReport) {
    if bounds.declared_lower_bound.is_none()
        && bounds.declared_upper_bound.is_none()
        && bounds.estimated_distortion.is_none()
        && bounds.violation_rate.is_none()
        && bounds.max_observed_violation.is_none()
        && bounds.quantile_observed_violation.is_none()
    {
        report.error(
            codes::ENCODER_DISTORTION_BOUNDS_EMPTY,
            "encoder_distortion.bounds must declare at least one bound or observed violation statistic",
        );
    }

    validate_optional_positive_f64(
        codes::ENCODER_DISTORTION_LOWER_BOUND_INVALID,
        "encoder_distortion.bounds.declared_lower_bound must be finite and greater than zero",
        bounds.declared_lower_bound,
        report,
    );
    validate_optional_positive_f64(
        codes::ENCODER_DISTORTION_UPPER_BOUND_INVALID,
        "encoder_distortion.bounds.declared_upper_bound must be finite and greater than zero",
        bounds.declared_upper_bound,
        report,
    );
    validate_optional_positive_f64(
        codes::ENCODER_DISTORTION_ESTIMATED_DISTORTION_INVALID,
        "encoder_distortion.bounds.estimated_distortion must be finite and greater than zero",
        bounds.estimated_distortion,
        report,
    );
    validate_optional_probability(
        codes::ENCODER_DISTORTION_VIOLATION_RATE_INVALID,
        "encoder_distortion.bounds.violation_rate must be finite and within [0, 1]",
        bounds.violation_rate,
        report,
    );
    validate_optional_nonnegative_f64(
        codes::ENCODER_DISTORTION_MAX_OBSERVED_VIOLATION_INVALID,
        "encoder_distortion.bounds.max_observed_violation must be finite and non-negative",
        bounds.max_observed_violation,
        report,
    );
    validate_optional_nonnegative_f64(
        codes::ENCODER_DISTORTION_QUANTILE_OBSERVED_VIOLATION_INVALID,
        "encoder_distortion.bounds.quantile_observed_violation must be finite and non-negative",
        bounds.quantile_observed_violation,
        report,
    );

    if let (Some(lower), Some(upper)) = (bounds.declared_lower_bound, bounds.declared_upper_bound) {
        if lower.is_finite() && upper.is_finite() && lower > upper {
            report.error(
                codes::ENCODER_DISTORTION_BOUNDS_ORDER_INVALID,
                "encoder_distortion.bounds.declared_lower_bound must be less than or equal to declared_upper_bound",
            );
        }
        if lower.is_finite() && upper.is_finite() && lower > 0.0 && upper > 0.0 {
            if let Some(estimated) = bounds.estimated_distortion {
                let expected = upper / lower;
                if !expected.is_finite() {
                    report.error(
                        codes::ENCODER_DISTORTION_DISTORTION_MISMATCH,
                        "encoder_distortion.bounds.declared_upper_bound / declared_lower_bound must be finite",
                    );
                } else {
                    let tolerance = 1e-9_f64.max(expected.abs() * 1e-9);
                    if estimated.is_finite() && (estimated - expected).abs() > tolerance {
                        report.error(
                            codes::ENCODER_DISTORTION_DISTORTION_MISMATCH,
                            format!(
                                "encoder_distortion.bounds.estimated_distortion {} does not match declared_upper_bound / declared_lower_bound {}",
                                estimated, expected
                            ),
                        );
                    }
                }
            }
        }
    }
}

fn validate_encoder_distortion_scope(scope: &DistortionScope, report: &mut VerificationReport) {
    validate_optional_sha256_uri(
        codes::ENCODER_DISTORTION_SCOPE_CORPUS_DIGEST_INVALID,
        "encoder_distortion.scope.corpus_digest must be sha256:<lowercase-hex> when present",
        scope.corpus_digest.as_deref(),
        report,
    );
    validate_optional_sha256_uri(
        codes::ENCODER_DISTORTION_SCOPE_QUERY_SET_DIGEST_INVALID,
        "encoder_distortion.scope.query_set_digest must be sha256:<lowercase-hex> when present",
        scope.query_set_digest.as_deref(),
        report,
    );
    validate_optional_sha256_uri(
        codes::ENCODER_DISTORTION_SCOPE_PAIR_SAMPLE_DIGEST_INVALID,
        "encoder_distortion.scope.pair_sample_digest must be sha256:<lowercase-hex> when present",
        scope.pair_sample_digest.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::ENCODER_DISTORTION_SCOPE_DOMAIN_EMPTY,
        "encoder_distortion.scope.domain must be non-empty when present",
        scope.domain.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::ENCODER_DISTORTION_SCOPE_ESTIMATOR_VERSION_EMPTY,
        "encoder_distortion.scope.estimator_version must be non-empty when present",
        scope.estimator_version.as_deref(),
        report,
    );
    if scope
        .sample_size
        .is_some_and(|sample_size| sample_size == 0)
    {
        report.error(
            codes::ENCODER_DISTORTION_SCOPE_SAMPLE_SIZE_ZERO,
            "encoder_distortion.scope.sample_size must be greater than zero when present",
        );
    }
    validate_optional_probability(
        codes::ENCODER_DISTORTION_SCOPE_CONFIDENCE_INVALID,
        "encoder_distortion.scope.confidence must be finite and within [0, 1]",
        scope.confidence,
        report,
    );
    validate_optional_probability(
        codes::ENCODER_DISTORTION_SCOPE_COVERAGE_INVALID,
        "encoder_distortion.scope.coverage must be finite and within [0, 1]",
        scope.coverage,
        report,
    );
}

fn validate_encoder_distortion_evidence(
    profile: &EncoderDistortionProfileRef,
    base_dir: &Path,
    options: &VerifyOptions,
    report: &mut VerificationReport,
) {
    validate_optional_non_empty(
        codes::ENCODER_DISTORTION_EVIDENCE_ESTIMATOR_ID_EMPTY,
        "encoder_distortion.evidence.estimator_id must be non-empty when present",
        profile.evidence.estimator_id.as_deref(),
        report,
    );
    validate_optional_sha256_uri(
        codes::ENCODER_DISTORTION_EVIDENCE_ESTIMATOR_HASH_INVALID,
        "encoder_distortion.evidence.estimator_hash must be sha256:<lowercase-hex> when present",
        profile.evidence.estimator_hash.as_deref(),
        report,
    );

    if profile.profile.is_none() && profile.evidence.kind != DistortionEvidenceKind::CallerAsserted
    {
        report.error(
            codes::ENCODER_DISTORTION_PROFILE_REQUIRED,
            "non-caller-asserted encoder distortion evidence requires a profile artifact",
        );
        return;
    }

    if let Some(artifact) = &profile.profile {
        validate_encoder_distortion_profile_artifact(artifact, base_dir, options, report);
    }
}

fn validate_encoder_distortion_profile_artifact(
    profile: &DistortionProfileArtifactRef,
    base_dir: &Path,
    options: &VerifyOptions,
    report: &mut VerificationReport,
) {
    report.encoder_distortion.profile_manifest_path = Some(profile.path.clone());
    if profile.path.trim().is_empty() {
        report.error(
            codes::ENCODER_DISTORTION_PROFILE_PATH_EMPTY,
            "encoder_distortion.profile.path must be non-empty",
        );
    } else if !is_manifest_path_absolute(&profile.path)
        && !is_canonical_manifest_path(&profile.path, options.allow_path_escape)
    {
        report.error(
            codes::ENCODER_DISTORTION_PROFILE_PATH_NOT_CANONICAL,
            "encoder_distortion.profile.path must use forward slashes with no `.`, `..`, or empty segments",
        );
    }
    if !is_sha256_hex(&profile.sha256) {
        report.error(
            codes::ENCODER_DISTORTION_PROFILE_SHA256_INVALID,
            "encoder_distortion.profile.sha256 must be a lowercase 64-character hex SHA-256 digest",
        );
    }
    if profile.file_size_bytes == 0 {
        report.error(
            codes::ENCODER_DISTORTION_PROFILE_FILE_SIZE_ZERO,
            "encoder_distortion.profile.file_size_bytes must be greater than zero",
        );
    }
    if profile.format.trim().is_empty() {
        report.error(
            codes::ENCODER_DISTORTION_PROFILE_FORMAT_EMPTY,
            "encoder_distortion.profile.format must be non-empty",
        );
    }
    validate_optional_sha256_uri(
        codes::ENCODER_DISTORTION_PROFILE_SOURCE_DIGEST_INVALID,
        "encoder_distortion.profile.source_digest must be sha256:<lowercase-hex> when present",
        profile.source_digest.as_deref(),
        report,
    );

    if !profile.path.trim().is_empty() {
        let path = PathBuf::from(&profile.path);
        if let Some(resolved) = resolve_existing_path(
            &path,
            base_dir,
            options,
            &ENCODER_DISTORTION_PROFILE_PATH_ISSUES,
            &mut report.errors,
        ) {
            report.encoder_distortion.profile_canonical_path =
                Some(path_to_display(&resolved.canonical_path));
            match sha256_file_bounded(
                &resolved.canonical_path,
                profile
                    .file_size_bytes
                    .min(options.limits.max_encoder_distortion_profile_bytes),
                codes::ENCODER_DISTORTION_PROFILE_TOO_LARGE,
                "encoder distortion profile",
            ) {
                Ok(hash) => {
                    report.encoder_distortion.profile_sha256 = Some(hash.sha256.clone());
                    report.encoder_distortion.profile_size_bytes = Some(hash.size_bytes);
                    if !hex_digest_eq(&hash.sha256, &profile.sha256) {
                        report.error(
                            codes::ENCODER_DISTORTION_PROFILE_SHA256_MISMATCH,
                            format!(
                                "encoder distortion profile SHA-256 was {}, manifest declares {}",
                                hash.sha256, profile.sha256
                            ),
                        );
                    }
                    if hash.size_bytes != profile.file_size_bytes {
                        report.error(
                            codes::ENCODER_DISTORTION_PROFILE_FILE_SIZE_MISMATCH,
                            format!(
                                "encoder distortion profile size was {}, manifest declares {}",
                                hash.size_bytes, profile.file_size_bytes
                            ),
                        );
                    }
                }
                Err(ManifestError::LimitExceeded { code, message }) => report.error(code, message),
                Err(err) => report.error(
                    codes::ENCODER_DISTORTION_PROFILE_HASH_FAILED,
                    format!("failed to hash encoder distortion profile: {err}"),
                ),
            }
        }
    }
}

fn validate_encoder_distortion_calibration(
    profile: &EncoderDistortionProfileRef,
    calibration: Option<&CalibrationProfileRef>,
    report: &mut VerificationReport,
) {
    let Some(calibration_profile_id) = &profile.calibration_profile_id else {
        return;
    };
    if calibration_profile_id.trim().is_empty() {
        report.error(
            codes::ENCODER_DISTORTION_CALIBRATION_PROFILE_ID_EMPTY,
            "encoder_distortion.calibration_profile_id must be non-empty when present",
        );
        return;
    }
    if calibration_profile_id.trim() != calibration_profile_id {
        report.error(
            codes::ENCODER_DISTORTION_CALIBRATION_PROFILE_ID_WHITESPACE,
            "encoder_distortion.calibration_profile_id must not contain leading or trailing whitespace",
        );
        return;
    }
    let Some(calibration) = calibration else {
        report.error(
            codes::ENCODER_DISTORTION_CALIBRATION_MISSING,
            "encoder_distortion.calibration_profile_id requires a calibration block",
        );
        return;
    };
    // Calibration profile ids are manifest identifiers; keep matching exact.
    if calibration.profile_id != *calibration_profile_id {
        report.error(
            codes::ENCODER_DISTORTION_CALIBRATION_PROFILE_MISMATCH,
            format!(
                "encoder_distortion.calibration_profile_id {:?} does not match calibration.profile_id {:?}",
                calibration_profile_id, calibration.profile_id
            ),
        );
    }
}

fn verify_calibration(
    document: &ManifestDocument,
    options: &VerifyOptions,
    report: &mut VerificationReport,
) {
    let Some(calibration) = &document.manifest.calibration else {
        return;
    };

    report.calibration.present = true;
    report.calibration.schema_version = Some(calibration.schema_version.clone());
    report.calibration.profile_id = Some(calibration.profile_id.clone());
    report.calibration.calibrated_for_model = Some(calibration.calibrated_for.model.clone());
    report.calibration.ordinalization = Some(calibration.ordinalization.label().to_string());
    report.calibration.null_model = Some(calibration.null_model.label().to_string());

    validate_calibration_shape(calibration, report);
    validate_calibration_encoder(calibration, &document.manifest.embedding, report);
    validate_calibration_ordinalization(calibration, &document.manifest.artifact, report);
    validate_calibration_null_model_ordinalization(calibration, report);
    validate_calibration_profile(
        calibration,
        &document.manifest.artifact,
        &document.base_dir,
        options,
        report,
    );
}

fn validate_calibration_shape(
    calibration: &CalibrationProfileRef,
    report: &mut VerificationReport,
) {
    if calibration.schema_version != CALIBRATION_SCHEMA_VERSION {
        report.error(
            codes::CALIBRATION_SCHEMA_VERSION_UNSUPPORTED,
            format!(
                "calibration.schema_version must be {CALIBRATION_SCHEMA_VERSION}, got {}",
                calibration.schema_version
            ),
        );
    }
    if calibration.profile_id.trim().is_empty() {
        report.error(
            codes::CALIBRATION_PROFILE_ID_EMPTY,
            "calibration.profile_id must be non-empty",
        );
    }
    if calibration
        .created_at
        .as_ref()
        .is_some_and(|created_at| DateTime::parse_from_rfc3339(created_at).is_err())
    {
        report.error(
            codes::CALIBRATION_CREATED_AT_INVALID,
            "calibration.created_at must parse as RFC3339 when present",
        );
    }
    if calibration.calibrated_for.model.trim().is_empty() {
        report.error(
            codes::CALIBRATION_ENCODER_MODEL_EMPTY,
            "calibration.calibrated_for.model must be non-empty",
        );
    }
    if calibration.calibrated_for.dim == 0 {
        report.error(
            codes::CALIBRATION_ENCODER_DIM_ZERO,
            "calibration.calibrated_for.dim must be greater than zero",
        );
    }
    validate_optional_non_empty(
        codes::CALIBRATION_ENCODER_MODEL_REVISION_EMPTY,
        "calibration.calibrated_for.model_revision must be non-empty when present",
        calibration.calibrated_for.model_revision.as_deref(),
        report,
    );
    validate_optional_non_empty(
        codes::CALIBRATION_ENCODER_NORMALIZATION_EMPTY,
        "calibration.calibrated_for.normalization must be non-empty when present",
        calibration.calibrated_for.normalization.as_deref(),
        report,
    );
    if calibration.ordinalization.dim() == 0 {
        report.error(
            codes::CALIBRATION_ORDINALIZATION_DIM_ZERO,
            "calibration.ordinalization.dim must be greater than zero",
        );
    }
    match &calibration.ordinalization {
        CalibrationOrdinalization::TopK { k, .. } if *k == 0 => {
            report.error(
                codes::CALIBRATION_ORDINALIZATION_ARTIFACT_MISMATCH,
                "calibration top_k.k must be greater than zero",
            );
        }
        CalibrationOrdinalization::Bucket { bits, .. } if !matches!(*bits, 1 | 2 | 4) => {
            report.error(
                codes::CALIBRATION_ORDINALIZATION_ARTIFACT_MISMATCH,
                "calibration bucket.bits must be 1, 2, or 4",
            );
        }
        CalibrationOrdinalization::CallerDefined { name, .. } if name.trim().is_empty() => {
            report.error(
                codes::CALIBRATION_ORDINALIZATION_ARTIFACT_MISMATCH,
                "calibration caller_defined.name must be non-empty",
            );
        }
        _ => {}
    }
    match &calibration.null_model {
        NullModelSpec::EmpiricalTailTable { statistic } if statistic.trim().is_empty() => {
            report.error(
                codes::CALIBRATION_NULL_STATISTIC_EMPTY,
                "calibration.null_model.statistic must be non-empty",
            );
        }
        NullModelSpec::CallerDefined {
            name,
            parameterization,
        } => {
            if name.trim().is_empty() {
                report.error(
                    codes::CALIBRATION_NULL_NAME_EMPTY,
                    "calibration.null_model.name must be non-empty",
                );
            }
            validate_optional_non_empty(
                codes::CALIBRATION_NULL_PARAMETERIZATION_EMPTY,
                "calibration.null_model.parameterization must be non-empty when present",
                parameterization.as_deref(),
                report,
            );
        }
        _ => {}
    }
}

fn validate_calibration_encoder(
    calibration: &CalibrationProfileRef,
    embedding: &Embedding,
    report: &mut VerificationReport,
) {
    if calibration.calibrated_for.model != embedding.model {
        report.error(
            codes::CALIBRATION_ENCODER_MODEL_MISMATCH,
            format!(
                "calibration model {:?} does not match embedding.model {:?}",
                calibration.calibrated_for.model, embedding.model
            ),
        );
    }
    if calibration.calibrated_for.dim != embedding.dim {
        report.error(
            codes::CALIBRATION_ENCODER_DIM_MISMATCH,
            format!(
                "calibration dim {} does not match embedding.dim {}",
                calibration.calibrated_for.dim, embedding.dim
            ),
        );
    }
    compare_optional_identity(
        codes::CALIBRATION_ENCODER_MODEL_REVISION_MISMATCH,
        "calibration encoder",
        "model_revision",
        embedding.model_revision.as_deref(),
        calibration.calibrated_for.model_revision.as_deref(),
        report,
    );
    compare_optional_identity(
        codes::CALIBRATION_ENCODER_NORMALIZATION_MISMATCH,
        "calibration encoder",
        "normalization",
        embedding.normalization.as_deref(),
        calibration.calibrated_for.normalization.as_deref(),
        report,
    );
}

fn compare_optional_identity(
    code: &str,
    subject: &str,
    field: &str,
    embedding_value: Option<&str>,
    calibration_value: Option<&str>,
    report: &mut VerificationReport,
) {
    compare_optional_encoder_identity(
        code,
        subject,
        field,
        embedding_value,
        calibration_value,
        report,
    );
}

fn compare_optional_encoder_identity(
    code: &str,
    subject: &str,
    field: &str,
    embedding_value: Option<&str>,
    observed_value: Option<&str>,
    report: &mut VerificationReport,
) {
    match (embedding_value, observed_value) {
        (Some(expected), Some(observed)) if expected == observed => {}
        (None, None) => {}
        _ => report.error(
            code,
            format!("{subject} {field} does not match embedding.{field}"),
        ),
    }
}

fn validate_calibration_ordinalization(
    calibration: &CalibrationProfileRef,
    artifact: &Artifact,
    report: &mut VerificationReport,
) {
    if calibration.ordinalization.dim() != artifact.dim {
        report.error(
            codes::CALIBRATION_ORDINALIZATION_DIM_MISMATCH,
            format!(
                "calibration ordinalization dim {} does not match artifact.dim {}",
                calibration.ordinalization.dim(),
                artifact.dim
            ),
        );
    }

    let compatible = match (artifact.kind, &artifact.params, &calibration.ordinalization) {
        (
            ManifestIndexKind::Bitmap,
            ManifestIndexParams::Bitmap { n_top },
            CalibrationOrdinalization::TopK { k, .. },
        ) => k == n_top,
        (
            ManifestIndexKind::RankQuant,
            ManifestIndexParams::RankQuant { bits },
            CalibrationOrdinalization::Bucket {
                bits: calibrated_bits,
                ..
            },
        ) => calibrated_bits == bits,
        (
            ManifestIndexKind::SignBitmap,
            ManifestIndexParams::SignBitmap,
            CalibrationOrdinalization::Sign { .. },
        ) => true,
        (
            ManifestIndexKind::Rank,
            ManifestIndexParams::Rank,
            CalibrationOrdinalization::RankPosition { .. }
            | CalibrationOrdinalization::CallerDefined { .. },
        ) => true,
        _ => false,
    };

    if !compatible {
        report.error(
            codes::CALIBRATION_ORDINALIZATION_ARTIFACT_MISMATCH,
            "calibration.ordinalization is incompatible with artifact.kind/artifact.params",
        );
    }
}

fn validate_calibration_null_model_ordinalization(
    calibration: &CalibrationProfileRef,
    report: &mut VerificationReport,
) {
    if matches!(
        (&calibration.null_model, &calibration.ordinalization),
        (
            NullModelSpec::UniformHypergeometric,
            CalibrationOrdinalization::TopK { .. }
        )
    ) {
        return;
    }
    if matches!(
        &calibration.null_model,
        NullModelSpec::UniformHypergeometric
    ) {
        report.error(
            codes::CALIBRATION_NULL_MODEL_ORDINALIZATION_MISMATCH,
            "uniform_hypergeometric calibration requires top_k ordinalization",
        );
    }
}

fn validate_calibration_profile(
    calibration: &CalibrationProfileRef,
    artifact: &Artifact,
    base_dir: &Path,
    options: &VerifyOptions,
    report: &mut VerificationReport,
) {
    if matches!(
        &calibration.null_model,
        NullModelSpec::UniformHypergeometric
    ) {
        if calibration.profile.is_some() {
            report.error(
                codes::CALIBRATION_PROFILE_UNEXPECTED,
                "uniform_hypergeometric calibration must not include a profile artifact",
            );
        }
        return;
    }

    let Some(profile) = &calibration.profile else {
        report.error(
            codes::CALIBRATION_PROFILE_REQUIRED,
            "non-uniform calibration requires a profile artifact",
        );
        return;
    };

    report.calibration.profile_manifest_path = Some(profile.path.clone());
    if profile.path.trim().is_empty() {
        report.error(
            codes::CALIBRATION_PROFILE_PATH_EMPTY,
            "calibration.profile.path must be non-empty",
        );
    } else if !is_manifest_path_absolute(&profile.path)
        && !is_canonical_manifest_path(&profile.path, options.allow_path_escape)
    {
        report.error(
            codes::CALIBRATION_PROFILE_PATH_NOT_CANONICAL,
            "calibration.profile.path must use forward slashes with no `.`, `..`, or empty segments",
        );
    }
    if !is_sha256_hex(&profile.sha256) {
        report.error(
            codes::CALIBRATION_PROFILE_SHA256_INVALID,
            "calibration.profile.sha256 must be a lowercase 64-character hex SHA-256 digest",
        );
    }
    if profile.file_size_bytes == 0 {
        report.error(
            codes::CALIBRATION_PROFILE_FILE_SIZE_ZERO,
            "calibration.profile.file_size_bytes must be greater than zero",
        );
    }
    if profile.dim != artifact.dim {
        report.error(
            codes::CALIBRATION_PROFILE_DIM_MISMATCH,
            format!(
                "calibration profile dim {} does not match artifact.dim {}",
                profile.dim, artifact.dim
            ),
        );
    }
    if profile.sample_count == 0 {
        report.error(
            codes::CALIBRATION_PROFILE_SAMPLE_COUNT_ZERO,
            "calibration.profile.sample_count must be greater than zero",
        );
    }
    validate_optional_source_digest(profile.source_digest.as_deref(), report);
    validate_calibration_parameterization(calibration, profile, report);
    validate_calibration_profile_shape(profile, &calibration.ordinalization, report);

    if !profile.path.trim().is_empty() {
        let path = PathBuf::from(&profile.path);
        if let Some(resolved) = resolve_existing_path(
            &path,
            base_dir,
            options,
            &CALIBRATION_PROFILE_PATH_ISSUES,
            &mut report.errors,
        ) {
            report.calibration.profile_canonical_path =
                Some(path_to_display(&resolved.canonical_path));
            match sha256_file_bounded(
                &resolved.canonical_path,
                profile
                    .file_size_bytes
                    .min(options.limits.max_calibration_profile_bytes),
                codes::CALIBRATION_PROFILE_TOO_LARGE,
                "calibration profile",
            ) {
                Ok(hash) => {
                    report.calibration.profile_sha256 = Some(hash.sha256.clone());
                    report.calibration.profile_size_bytes = Some(hash.size_bytes);
                    if !hex_digest_eq(&hash.sha256, &profile.sha256) {
                        report.error(
                            codes::CALIBRATION_PROFILE_SHA256_MISMATCH,
                            format!(
                                "calibration profile SHA-256 was {}, manifest declares {}",
                                hash.sha256, profile.sha256
                            ),
                        );
                    }
                    if hash.size_bytes != profile.file_size_bytes {
                        report.error(
                            codes::CALIBRATION_PROFILE_FILE_SIZE_MISMATCH,
                            format!(
                                "calibration profile size was {}, manifest declares {}",
                                hash.size_bytes, profile.file_size_bytes
                            ),
                        );
                    }
                }
                Err(ManifestError::LimitExceeded { code, message }) => report.error(code, message),
                Err(err) => report.error(
                    codes::CALIBRATION_PROFILE_HASH_FAILED,
                    format!("failed to hash calibration profile: {err}"),
                ),
            }
        }
    }
}

fn validate_optional_source_digest(value: Option<&str>, report: &mut VerificationReport) {
    let Some(value) = value else {
        return;
    };
    let Some(digest) = value.strip_prefix("sha256:") else {
        report.error(
            codes::CALIBRATION_PROFILE_SOURCE_DIGEST_INVALID,
            "calibration.profile.source_digest must be sha256:<lowercase-hex>",
        );
        return;
    };
    if !is_sha256_hex(digest) {
        report.error(
            codes::CALIBRATION_PROFILE_SOURCE_DIGEST_INVALID,
            "calibration.profile.source_digest must be sha256:<lowercase-hex>",
        );
    }
}

fn validate_calibration_parameterization(
    calibration: &CalibrationProfileRef,
    profile: &ProfileArtifactRef,
    report: &mut VerificationReport,
) {
    match &calibration.null_model {
        NullModelSpec::WeightedMarginalProfile { parameterization }
            if *parameterization != profile.parameterization =>
        {
            report.error(
                codes::CALIBRATION_NULL_PARAMETERIZATION_MISMATCH,
                format!(
                    "null_model parameterization {:?} does not match profile parameterization {:?}",
                    parameterization, profile.parameterization
                ),
            );
        }
        NullModelSpec::EmpiricalTailTable { .. }
            if profile.parameterization != ProfileParameterization::EmpiricalTailTable =>
        {
            report.error(
                codes::CALIBRATION_NULL_PARAMETERIZATION_MISMATCH,
                "empirical_tail_table null_model requires empirical_tail_table profile parameterization",
            );
        }
        _ => {}
    }
    if !profile_parameterization_matches_ordinalization(
        profile.parameterization,
        &calibration.ordinalization,
    ) {
        report.error(
            codes::CALIBRATION_PROFILE_PARAMETERIZATION_ORDINALIZATION_MISMATCH,
            "calibration profile parameterization is incompatible with calibration ordinalization",
        );
    }
}

fn profile_parameterization_matches_ordinalization(
    parameterization: ProfileParameterization,
    ordinalization: &CalibrationOrdinalization,
) -> bool {
    match ordinalization {
        CalibrationOrdinalization::TopK { .. } => matches!(
            parameterization,
            ProfileParameterization::MarginalTopKFrequency
                | ProfileParameterization::EmpiricalTailTable
        ),
        CalibrationOrdinalization::Bucket { .. } => matches!(
            parameterization,
            ProfileParameterization::BucketFrequency | ProfileParameterization::EmpiricalTailTable
        ),
        CalibrationOrdinalization::Sign { .. } => matches!(
            parameterization,
            ProfileParameterization::SignFrequency | ProfileParameterization::EmpiricalTailTable
        ),
        CalibrationOrdinalization::RankPosition { .. } => matches!(
            parameterization,
            ProfileParameterization::RankPositionFrequency
                | ProfileParameterization::EmpiricalTailTable
        ),
        CalibrationOrdinalization::CallerDefined { .. } => true,
    }
}

fn validate_calibration_profile_shape(
    profile: &ProfileArtifactRef,
    ordinalization: &CalibrationOrdinalization,
    report: &mut VerificationReport,
) {
    if profile.format.trim().is_empty() {
        report.error(
            codes::CALIBRATION_PROFILE_FORMAT_EMPTY,
            "calibration.profile.format must be non-empty",
        );
    }

    if profile.shape.is_empty() {
        return;
    }

    if let Some(expected) = expected_profile_shape(profile.parameterization, ordinalization) {
        if profile.shape != expected {
            report.error(
                codes::CALIBRATION_PROFILE_SHAPE_MISMATCH,
                format!(
                    "calibration profile shape {:?} does not match expected {:?}",
                    profile.shape, expected
                ),
            );
        }
    }

    let bytes_per_value = match profile.format.as_str() {
        "raw_f64_le" => Some(8u64),
        "raw_f32_le" => Some(4u64),
        _ => None,
    };
    let Some(bytes_per_value) = bytes_per_value else {
        return;
    };
    let Some(values) = profile
        .shape
        .iter()
        .try_fold(1u64, |acc, value| acc.checked_mul(*value as u64))
    else {
        report.error(
            codes::CALIBRATION_PROFILE_SHAPE_MISMATCH,
            "calibration.profile.shape product overflows u64",
        );
        return;
    };
    let Some(expected_bytes) = values.checked_mul(bytes_per_value) else {
        report.error(
            codes::CALIBRATION_PROFILE_SHAPE_MISMATCH,
            "calibration.profile.shape byte size overflows u64",
        );
        return;
    };
    if profile.file_size_bytes != expected_bytes {
        report.error(
            codes::CALIBRATION_PROFILE_FILE_SIZE_MISMATCH,
            format!(
                "calibration profile size {} does not match shape/format size {}",
                profile.file_size_bytes, expected_bytes
            ),
        );
    }
}

fn expected_profile_shape(
    parameterization: ProfileParameterization,
    ordinalization: &CalibrationOrdinalization,
) -> Option<Vec<usize>> {
    match parameterization {
        ProfileParameterization::MarginalTopKFrequency => Some(vec![ordinalization.dim()]),
        ProfileParameterization::SignFrequency => Some(vec![ordinalization.dim()]),
        ProfileParameterization::BucketFrequency => match ordinalization {
            CalibrationOrdinalization::Bucket { dim, bits } if matches!(*bits, 1 | 2 | 4) => {
                Some(vec![*dim, 1usize << *bits])
            }
            _ => None,
        },
        ProfileParameterization::RankPositionFrequency => {
            Some(vec![ordinalization.dim(), ordinalization.dim()])
        }
        ProfileParameterization::EmpiricalTailTable => None,
    }
}

fn verify_auxiliary_artifacts(
    document: &ManifestDocument,
    options: &VerifyOptions,
    report: &mut VerificationReport,
    paths: &mut VerificationPathCapture,
) {
    if !check_auxiliary_artifact_count(&document.manifest, &options.limits, report) {
        return;
    }
    let artifacts = auxiliary_artifacts_in_report_order(&document.manifest);
    let base_canonical = if options.allow_path_escape {
        None
    } else {
        match fs::canonicalize(&document.base_dir) {
            Ok(path) => Some(path),
            Err(err) => {
                for artifact in artifacts {
                    let mut entry = auxiliary_artifact_report_entry(artifact, &document.base_dir);
                    if artifact.path.trim().is_empty() {
                        mark_auxiliary_artifact_failed(
                            &mut entry,
                            codes::AUXILIARY_ARTIFACT_PATH_EMPTY,
                        );
                    } else {
                        report.error(
                            codes::AUXILIARY_ARTIFACT_BASE_DIR_UNAVAILABLE,
                            format!(
                                "failed to canonicalize base_dir {} for auxiliary artifact {:?}: {err}",
                                document.base_dir.display(),
                                artifact.name
                            ),
                        );
                        mark_auxiliary_artifact_failed(
                            &mut entry,
                            codes::AUXILIARY_ARTIFACT_BASE_DIR_UNAVAILABLE,
                        );
                    }
                    report.auxiliary_artifacts.push(entry);
                }
                return;
            }
        }
    };

    for artifact in artifacts {
        let mut entry = auxiliary_artifact_report_entry(artifact, &document.base_dir);
        let mut captured_path = None;

        if artifact.path.trim().is_empty() {
            mark_auxiliary_artifact_failed(&mut entry, codes::AUXILIARY_ARTIFACT_PATH_EMPTY);
            report.auxiliary_artifacts.push(entry);
            paths.auxiliary_artifact_paths.push(None);
            continue;
        }

        match resolve_auxiliary_artifact_path(
            artifact,
            &document.base_dir,
            base_canonical.as_deref(),
            options,
            report,
        ) {
            AuxiliaryPathResolution::Resolved(resolved) => {
                captured_path = Some(resolved.canonical_path.clone());
                entry.canonical_path = Some(path_to_display(&resolved.canonical_path));
                // Bound the read by the manifest-declared size (the manifest
                // is the trust anchor; the SHA-256 pins content). A flat
                // limit, when explicitly configured, remains a ceiling.
                match sha256_file_bounded(
                    &resolved.canonical_path,
                    artifact
                        .file_size_bytes
                        .min(options.limits.max_auxiliary_artifact_bytes),
                    codes::AUXILIARY_ARTIFACT_FILE_TOO_LARGE,
                    "auxiliary artifact",
                ) {
                    Ok(hash) => {
                        entry.sha256 = Some(hash.sha256.clone());
                        entry.size_bytes = Some(hash.size_bytes);
                        if !hex_digest_eq(&hash.sha256, &artifact.sha256) {
                            mark_auxiliary_artifact_failed(
                                &mut entry,
                                codes::AUXILIARY_ARTIFACT_SHA256_MISMATCH,
                            );
                            report.errors.push(
                                ReportIssue::new(
                                    codes::AUXILIARY_ARTIFACT_SHA256_MISMATCH,
                                    format!(
                                        "auxiliary artifact {:?} SHA-256 was {}, manifest declares {}",
                                        artifact.name, hash.sha256, artifact.sha256
                                    ),
                                )
                                .with_artifact_name(artifact.name.as_str())
                                .with_sha256_detail(
                                    artifact.sha256.as_str(),
                                    hash.sha256.as_str(),
                                ),
                            );
                        }
                        if hash.size_bytes != artifact.file_size_bytes {
                            mark_auxiliary_artifact_failed(
                                &mut entry,
                                codes::AUXILIARY_ARTIFACT_FILE_SIZE_MISMATCH,
                            );
                            report.errors.push(
                                ReportIssue::new(
                                    codes::AUXILIARY_ARTIFACT_FILE_SIZE_MISMATCH,
                                    format!(
                                        "auxiliary artifact {:?} size was {}, manifest declares {}",
                                        artifact.name, hash.size_bytes, artifact.file_size_bytes
                                    ),
                                )
                                .with_artifact_name(artifact.name.as_str())
                                .with_size_detail(artifact.file_size_bytes, hash.size_bytes),
                            );
                        }
                        if entry.reason_code.is_none() {
                            entry.state = AuxiliaryArtifactState::Verified;
                        }
                    }
                    Err(err) => {
                        let code = err.code().unwrap_or(codes::AUXILIARY_ARTIFACT_HASH_FAILED);
                        mark_auxiliary_artifact_failed(&mut entry, code);
                        let message = if err.code().is_some() {
                            err.to_string()
                        } else {
                            format!(
                                "failed to hash auxiliary artifact {:?}: {err}",
                                artifact.name
                            )
                        };
                        report.error(code, message);
                    }
                }
            }
            AuxiliaryPathResolution::OptionalAbsent => {
                entry.state = AuxiliaryArtifactState::OptionalAbsent;
                entry.reason_code = Some(codes::AUXILIARY_ARTIFACT_OPTIONAL_ABSENT.to_string());
            }
            AuxiliaryPathResolution::MissingRequired => {
                entry.state = AuxiliaryArtifactState::MissingRequired;
                entry.reason_code = Some(codes::AUXILIARY_ARTIFACT_MISSING_REQUIRED.to_string());
            }
            AuxiliaryPathResolution::Failed(code) => {
                entry.state = AuxiliaryArtifactState::Failed;
                entry.reason_code = Some(code);
            }
        }

        report.auxiliary_artifacts.push(entry);
        paths.auxiliary_artifact_paths.push(captured_path);
    }
}

fn auxiliary_artifact_report_entry(
    artifact: &AuxiliaryArtifact,
    base_dir: &Path,
) -> AuxiliaryArtifactReport {
    let resolved_path = if artifact.path.trim().is_empty() {
        None
    } else {
        Some(path_to_display(&auxiliary_artifact_resolved_path(
            artifact, base_dir,
        )))
    };
    AuxiliaryArtifactReport {
        name: artifact.name.clone(),
        manifest_path: artifact.path.clone(),
        resolved_path,
        canonical_path: None,
        expected_sha256: Some(artifact.sha256.clone()),
        expected_size_bytes: Some(artifact.file_size_bytes),
        required: artifact.required,
        state: AuxiliaryArtifactState::Failed,
        reason_code: None,
        sha256: None,
        size_bytes: None,
    }
}

fn check_auxiliary_artifact_count(
    manifest: &IndexManifest,
    limits: &ResourceLimits,
    report: &mut VerificationReport,
) -> bool {
    let count = manifest.auxiliary_artifacts.len();
    if count <= limits.max_auxiliary_artifacts {
        return true;
    }
    if !report
        .errors
        .iter()
        .any(|issue| issue.code == codes::AUXILIARY_ARTIFACT_COUNT_LIMIT_EXCEEDED)
    {
        push_report_issue_bounded(
            &mut report.errors,
            limits,
            codes::AUXILIARY_ARTIFACT_COUNT_LIMIT_EXCEEDED,
            format!(
                "auxiliary_artifacts has {count} entries, exceeding max_auxiliary_artifacts={}",
                limits.max_auxiliary_artifacts
            ),
        );
    }
    false
}

fn auxiliary_artifacts_in_report_order(manifest: &IndexManifest) -> Vec<&AuxiliaryArtifact> {
    let mut artifacts: Vec<_> = manifest.auxiliary_artifacts.iter().collect();
    artifacts.sort_by(|left, right| {
        left.name
            .cmp(&right.name)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.required.cmp(&right.required))
    });
    artifacts
}

enum AuxiliaryPathResolution {
    Resolved(ResolvedPath),
    OptionalAbsent,
    MissingRequired,
    Failed(String),
}

fn resolve_auxiliary_artifact_path(
    artifact: &AuxiliaryArtifact,
    base_dir: &Path,
    base_canonical: Option<&Path>,
    options: &VerifyOptions,
    report: &mut VerificationReport,
) -> AuxiliaryPathResolution {
    let path = Path::new(&artifact.path);
    // Same classification rule as `resolve_existing_path`: the policy gate
    // and the mismatch rejection below keep the platform-independent manifest
    // classification and this platform's resolution semantics aligned.
    let manifest_absolute = is_manifest_path_absolute(&artifact.path);
    if manifest_absolute && !options.allow_absolute_paths {
        report.error(
            codes::AUXILIARY_ARTIFACT_ABSOLUTE_PATH_REJECTED,
            format!(
                "absolute auxiliary artifact path {} for {:?} is rejected by default",
                path.display(),
                artifact.name
            ),
        );
        return AuxiliaryPathResolution::Failed(
            codes::AUXILIARY_ARTIFACT_ABSOLUTE_PATH_REJECTED.to_string(),
        );
    }

    if manifest_absolute != path.is_absolute() {
        report.error(
            codes::AUXILIARY_ARTIFACT_ABSOLUTE_PATH_UNRESOLVABLE,
            format!(
                "auxiliary artifact path {} for {:?} is classified absolute by manifest policy but cannot resolve as absolute on this platform; refusing to resolve it against the manifest base",
                path.display(),
                artifact.name
            ),
        );
        return AuxiliaryPathResolution::Failed(
            codes::AUXILIARY_ARTIFACT_ABSOLUTE_PATH_UNRESOLVABLE.to_string(),
        );
    }

    if !path.is_absolute() && !options.allow_path_escape && has_lexical_escape(path) {
        report.error(
            codes::AUXILIARY_ARTIFACT_PATH_ESCAPE_REJECTED,
            format!(
                "relative auxiliary artifact path {} for {:?} escapes the manifest base",
                path.display(),
                artifact.name
            ),
        );
        return AuxiliaryPathResolution::Failed(
            codes::AUXILIARY_ARTIFACT_PATH_ESCAPE_REJECTED.to_string(),
        );
    }

    let resolved_path = auxiliary_artifact_resolved_path(artifact, base_dir);
    let canonical_path = match fs::canonicalize(&resolved_path) {
        Ok(path) => path,
        Err(err) if err.kind() == io::ErrorKind::NotFound && !artifact.required => {
            return AuxiliaryPathResolution::OptionalAbsent;
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {
            report.errors.push(
                ReportIssue::new(
                    codes::AUXILIARY_ARTIFACT_MISSING_REQUIRED,
                    format!(
                        "required auxiliary artifact {:?} is missing at {}",
                        artifact.name,
                        resolved_path.display()
                    ),
                )
                .with_artifact_name(artifact.name.as_str()),
            );
            return AuxiliaryPathResolution::MissingRequired;
        }
        Err(err) => {
            report.error(
                codes::AUXILIARY_ARTIFACT_PATH_UNAVAILABLE,
                format!(
                    "failed to canonicalize auxiliary artifact {:?} at {}: {err}",
                    artifact.name,
                    resolved_path.display()
                ),
            );
            return AuxiliaryPathResolution::Failed(
                codes::AUXILIARY_ARTIFACT_PATH_UNAVAILABLE.to_string(),
            );
        }
    };

    if let Some(base_canonical) = base_canonical {
        if !canonical_path.starts_with(base_canonical) {
            report.error(
                codes::AUXILIARY_ARTIFACT_PATH_ESCAPE_REJECTED,
                format!(
                    "canonical auxiliary artifact path {} for {:?} is outside manifest base {}",
                    canonical_path.display(),
                    artifact.name,
                    base_canonical.display()
                ),
            );
            return AuxiliaryPathResolution::Failed(
                codes::AUXILIARY_ARTIFACT_PATH_ESCAPE_REJECTED.to_string(),
            );
        }
    }

    AuxiliaryPathResolution::Resolved(ResolvedPath { canonical_path })
}

fn auxiliary_artifact_resolved_path(artifact: &AuxiliaryArtifact, base_dir: &Path) -> PathBuf {
    let path = Path::new(&artifact.path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    }
}

fn mark_auxiliary_artifact_failed(entry: &mut AuxiliaryArtifactReport, code: &str) {
    entry.state = AuxiliaryArtifactState::Failed;
    if entry.reason_code.is_none() {
        entry.reason_code = Some(code.to_string());
    }
}

fn verify_attestations(manifest: &IndexManifest, report: &mut VerificationReport) {
    if manifest.attestations.is_empty() {
        report
            .skipped_checks
            .push("attestations_absent".to_string());
        return;
    }

    let artifact_sha = report
        .artifact
        .sha256
        .clone()
        .unwrap_or_else(|| manifest.artifact.sha256.clone());
    let mut any_subject_match = false;
    for (idx, attestation) in manifest.attestations.iter().enumerate() {
        let predicate_type = attestation
            .get("predicateType")
            .or_else(|| attestation.get("predicate_type"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned);
        if predicate_type.is_none() {
            report.error(
                codes::ATTESTATION_PREDICATE_TYPE_MISSING,
                format!("attestation {idx} has no predicateType"),
            );
        }

        let builder_id = attestation
            .pointer("/predicate/builder/id")
            .or_else(|| attestation.pointer("/predicate/runDetails/builder/id"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned);

        let subject_sha256_matched = attestation
            .get("subject")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|subjects| {
                subjects.iter().any(|subject| {
                    subject
                        .pointer("/digest/sha256")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|digest| hex_digest_eq(digest, &artifact_sha))
                })
            });
        any_subject_match |= subject_sha256_matched;
        report.attestation_shape_checks.push(AttestationShapeCheck {
            predicate_type,
            builder_id,
            subject_sha256_matched,
        });
    }

    if !any_subject_match {
        report.error(
            codes::ATTESTATION_SUBJECT_SHA256_MISMATCH,
            "no supplied attestation subject digest matches the artifact SHA-256",
        );
    }
}

#[derive(Clone, Debug, Default)]
pub struct VerifyOptions {
    pub allow_absolute_paths: bool,
    pub allow_path_escape: bool,
    pub allow_duplicate_db_ids: bool,
    pub index_override: Option<PathBuf>,
    pub limits: ResourceLimits,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_manifest_bytes: u64,
    pub max_row_identity_jsonl_line_bytes: usize,
    pub max_row_identity_rows: usize,
    pub max_row_identity_tracked_db_id_bytes: usize,
    pub max_auxiliary_artifacts: usize,
    pub max_auxiliary_artifact_bytes: u64,
    /// Opt-in ceiling for the primary index artifact read (unbounded by
    /// default; the manifest-declared size is always the effective bound).
    pub max_index_artifact_bytes: u64,
    pub max_calibration_profile_bytes: u64,
    pub max_encoder_distortion_profile_bytes: u64,
    pub max_report_issues: usize,
    pub max_cached_report_bytes: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: DEFAULT_MAX_MANIFEST_BYTES,
            max_row_identity_jsonl_line_bytes: DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES,
            max_row_identity_rows: DEFAULT_MAX_ROW_IDENTITY_ROWS,
            max_row_identity_tracked_db_id_bytes: DEFAULT_MAX_ROW_IDENTITY_TRACKED_DB_ID_BYTES,
            max_auxiliary_artifacts: DEFAULT_MAX_AUXILIARY_ARTIFACTS,
            max_auxiliary_artifact_bytes: DEFAULT_MAX_AUXILIARY_ARTIFACT_BYTES,
            max_index_artifact_bytes: DEFAULT_MAX_INDEX_ARTIFACT_BYTES,
            max_calibration_profile_bytes: DEFAULT_MAX_CALIBRATION_PROFILE_BYTES,
            max_encoder_distortion_profile_bytes: DEFAULT_MAX_ENCODER_DISTORTION_PROFILE_BYTES,
            max_report_issues: DEFAULT_MAX_REPORT_ISSUES,
            max_cached_report_bytes: DEFAULT_MAX_CACHED_REPORT_BYTES,
        }
    }
}

#[derive(Clone, Debug)]
struct ResolvedPath {
    canonical_path: PathBuf,
}

#[derive(Clone, Debug, Default)]
struct VerificationPathCapture {
    artifact_path: Option<PathBuf>,
    row_identity_path: Option<PathBuf>,
    auxiliary_artifact_paths: Vec<Option<PathBuf>>,
}

/// Per-context issue codes for [`resolve_existing_path`], so every emitted
/// code stays a named constant in [`codes`].
struct PathIssueCodes {
    absolute_path_rejected: &'static str,
    absolute_path_unresolvable: &'static str,
    base_dir_unavailable: &'static str,
    path_escape_rejected: &'static str,
    path_unavailable: &'static str,
    /// Code for a `NotFound` canonicalize error, when this context wants that
    /// distinguished from other I/O failures (permission denied, symlink
    /// loop, …). `None` keeps a single `path_unavailable` code for every
    /// error kind — used where a missing referenced file is not classified
    /// distinctly downstream.
    path_missing: Option<&'static str>,
}

const ARTIFACT_PATH_ISSUES: PathIssueCodes = PathIssueCodes {
    absolute_path_rejected: codes::ARTIFACT_ABSOLUTE_PATH_REJECTED,
    absolute_path_unresolvable: codes::ARTIFACT_ABSOLUTE_PATH_UNRESOLVABLE,
    base_dir_unavailable: codes::ARTIFACT_BASE_DIR_UNAVAILABLE,
    path_escape_rejected: codes::ARTIFACT_PATH_ESCAPE_REJECTED,
    path_unavailable: codes::ARTIFACT_PATH_UNAVAILABLE,
    path_missing: Some(codes::ARTIFACT_MISSING),
};

const ROW_IDENTITY_PATH_ISSUES: PathIssueCodes = PathIssueCodes {
    absolute_path_rejected: codes::ROW_IDENTITY_ABSOLUTE_PATH_REJECTED,
    absolute_path_unresolvable: codes::ROW_IDENTITY_ABSOLUTE_PATH_UNRESOLVABLE,
    base_dir_unavailable: codes::ROW_IDENTITY_BASE_DIR_UNAVAILABLE,
    path_escape_rejected: codes::ROW_IDENTITY_PATH_ESCAPE_REJECTED,
    path_unavailable: codes::ROW_IDENTITY_PATH_UNAVAILABLE,
    path_missing: Some(codes::ROW_IDENTITY_MISSING),
};

const ENCODER_DISTORTION_PROFILE_PATH_ISSUES: PathIssueCodes = PathIssueCodes {
    absolute_path_rejected: codes::ENCODER_DISTORTION_PROFILE_ABSOLUTE_PATH_REJECTED,
    absolute_path_unresolvable: codes::ENCODER_DISTORTION_PROFILE_ABSOLUTE_PATH_UNRESOLVABLE,
    base_dir_unavailable: codes::ENCODER_DISTORTION_PROFILE_BASE_DIR_UNAVAILABLE,
    path_escape_rejected: codes::ENCODER_DISTORTION_PROFILE_PATH_ESCAPE_REJECTED,
    path_unavailable: codes::ENCODER_DISTORTION_PROFILE_PATH_UNAVAILABLE,
    path_missing: None,
};

const CALIBRATION_PROFILE_PATH_ISSUES: PathIssueCodes = PathIssueCodes {
    absolute_path_rejected: codes::CALIBRATION_PROFILE_ABSOLUTE_PATH_REJECTED,
    absolute_path_unresolvable: codes::CALIBRATION_PROFILE_ABSOLUTE_PATH_UNRESOLVABLE,
    base_dir_unavailable: codes::CALIBRATION_PROFILE_BASE_DIR_UNAVAILABLE,
    path_escape_rejected: codes::CALIBRATION_PROFILE_PATH_ESCAPE_REJECTED,
    path_unavailable: codes::CALIBRATION_PROFILE_PATH_UNAVAILABLE,
    path_missing: None,
};

fn resolve_existing_path(
    path: &Path,
    base_dir: &Path,
    options: &VerifyOptions,
    issue_codes: &PathIssueCodes,
    errors: &mut Vec<ReportIssue>,
) -> Option<ResolvedPath> {
    // Policy classification uses the platform-independent manifest rule, not
    // `Path::is_absolute`, so an absolute-for-policy path can never dodge the
    // `allow_absolute_paths` gate on a platform whose native semantics would
    // read it as relative (e.g. `C:/...` or UNC on Unix, `/...` on Windows).
    let manifest_absolute = is_manifest_path_absolute(&path.to_string_lossy());
    if manifest_absolute && !options.allow_absolute_paths {
        errors.push(ReportIssue::new(
            issue_codes.absolute_path_rejected,
            format!("absolute path {} is rejected by default", path.display()),
        ));
        return None;
    }

    // A path classified absolute for policy purposes must never silently
    // resolve relative to the manifest base (or vice versa): when the
    // manifest classification and this platform's semantics disagree, reject
    // outright instead of resolving inconsistently across OSes.
    if manifest_absolute != path.is_absolute() {
        errors.push(ReportIssue::new(
            issue_codes.absolute_path_unresolvable,
            format!(
                "path {} is classified absolute by manifest policy but cannot resolve as absolute on this platform; refusing to resolve it against the manifest base",
                path.display()
            ),
        ));
        return None;
    }

    let base_canonical = match fs::canonicalize(base_dir) {
        Ok(path) => path,
        Err(err) => {
            errors.push(ReportIssue::new(
                issue_codes.base_dir_unavailable,
                format!(
                    "failed to canonicalize base_dir {}: {err}",
                    base_dir.display()
                ),
            ));
            return None;
        }
    };

    if !path.is_absolute() && !options.allow_path_escape && has_lexical_escape(path) {
        errors.push(ReportIssue::new(
            issue_codes.path_escape_rejected,
            format!("relative path {} escapes the manifest base", path.display()),
        ));
        return None;
    }

    let resolved_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    };
    let canonical_path = match fs::canonicalize(&resolved_path) {
        Ok(path) => path,
        Err(err) => {
            // Distinguish an absent file (NotFound) from other canonicalize
            // failures (permission denied, symlink loop, I/O) when this
            // context asks for it, so a downstream consumer branching on the
            // typed code never reads a permission error as a missing file.
            let code = match issue_codes.path_missing {
                Some(missing) if err.kind() == io::ErrorKind::NotFound => missing,
                _ => issue_codes.path_unavailable,
            };
            errors.push(ReportIssue::new(
                code,
                format!("failed to canonicalize {}: {err}", resolved_path.display()),
            ));
            return None;
        }
    };

    if !options.allow_path_escape && !canonical_path.starts_with(&base_canonical) {
        errors.push(ReportIssue::new(
            issue_codes.path_escape_rejected,
            format!(
                "canonical path {} is outside manifest base {}",
                canonical_path.display(),
                base_canonical.display()
            ),
        ));
        return None;
    }

    Some(ResolvedPath { canonical_path })
}

fn has_lexical_escape(path: &Path) -> bool {
    let mut depth = 0usize;
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(_) => depth += 1,
            Component::ParentDir => {
                if depth == 0 {
                    return true;
                }
                depth -= 1;
            }
            Component::Prefix(_) | Component::RootDir => return true,
        }
    }
    false
}

fn default_required() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexManifest {
    pub schema_version: String,
    pub artifact: Artifact,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub auxiliary_artifacts: Vec<AuxiliaryArtifact>,
    pub embedding: Embedding,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_distortion: Option<EncoderDistortionProfileRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationProfileRef>,
    pub row_identity: RowIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<BuildInfo>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attestations: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extensions: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: String,
    pub sha256: String,
    pub kind: ManifestIndexKind,
    pub format_version: u8,
    pub dim: usize,
    pub vector_count: usize,
    pub bytes_per_vec: usize,
    pub params: ManifestIndexParams,
    pub file_size_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuxiliaryArtifact {
    pub name: String,
    pub path: String,
    pub sha256: String,
    pub file_size_bytes: u64,
    #[serde(default = "default_required", skip_serializing_if = "is_true")]
    pub required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Embedding {
    pub model: String,
    pub dim: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooling: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_matrix_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationProfileRef {
    pub schema_version: String,
    pub profile_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    pub calibrated_for: EncoderSpec,
    pub ordinalization: CalibrationOrdinalization,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<ProfileArtifactRef>,
    pub null_model: NullModelSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderSpec {
    pub model: String,
    pub dim: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderDistortionProfileRef {
    pub schema_version: String,
    pub profile_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,
    pub encoder: EncoderSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooling: Option<String>,
    pub source_metric: MetricSpec,
    pub embedding_metric: MetricSpec,
    pub bounds: DistortionBounds,
    pub scope: DistortionScope,
    pub evidence: DistortionEvidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<DistortionProfileArtifactRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_profile_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistortionBounds {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_lower_bound: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_upper_bound: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_distortion: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub violation_rate: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_observed_violation: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantile_observed_violation: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistortionScope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corpus_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_set_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pair_sample_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_size: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator_version: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistortionEvidence {
    pub kind: DistortionEvidenceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator_hash: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistortionEvidenceKind {
    Certified,
    EmpiricalSample,
    BenchmarkEstimate,
    TeacherEstimate,
    CallerAsserted,
}

impl DistortionEvidenceKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Certified => "certified",
            Self::EmpiricalSample => "empirical_sample",
            Self::BenchmarkEstimate => "benchmark_estimate",
            Self::TeacherEstimate => "teacher_estimate",
            Self::CallerAsserted => "caller_asserted",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistortionProfileArtifactRef {
    pub path: String,
    pub sha256: String,
    pub file_size_bytes: u64,
    pub format: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CalibrationOrdinalization {
    TopK { dim: usize, k: usize },
    Bucket { dim: usize, bits: u8 },
    Sign { dim: usize },
    RankPosition { dim: usize },
    CallerDefined { dim: usize, name: String },
}

impl CalibrationOrdinalization {
    pub fn dim(&self) -> usize {
        match self {
            Self::TopK { dim, .. }
            | Self::Bucket { dim, .. }
            | Self::Sign { dim }
            | Self::RankPosition { dim }
            | Self::CallerDefined { dim, .. } => *dim,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::TopK { .. } => "top_k",
            Self::Bucket { .. } => "bucket",
            Self::Sign { .. } => "sign",
            Self::RankPosition { .. } => "rank_position",
            Self::CallerDefined { .. } => "caller_defined",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileArtifactRef {
    pub path: String,
    pub sha256: String,
    pub file_size_bytes: u64,
    pub dim: usize,
    pub sample_count: usize,
    pub parameterization: ProfileParameterization,
    pub format: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub shape: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_digest: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProfileParameterization {
    #[serde(rename = "marginal_topk_frequency")]
    MarginalTopKFrequency,
    BucketFrequency,
    SignFrequency,
    RankPositionFrequency,
    EmpiricalTailTable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NullModelSpec {
    UniformHypergeometric,
    WeightedMarginalProfile {
        parameterization: ProfileParameterization,
    },
    EmpiricalTailTable {
        statistic: String,
    },
    CallerDefined {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parameterization: Option<String>,
    },
}

impl NullModelSpec {
    pub fn label(&self) -> &'static str {
        match self {
            Self::UniformHypergeometric => "uniform_hypergeometric",
            Self::WeightedMarginalProfile { .. } => "weighted_marginal_profile",
            Self::EmpiricalTailTable { .. } => "empirical_tail_table",
            Self::CallerDefined { .. } => "caller_defined",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildInfo {
    pub invocation_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builder_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_repo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci_run_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RowIdentity {
    RowIdIdentity {
        row_count: usize,
    },
    Jsonl {
        path: String,
        sha256: String,
        row_count: usize,
        id_kind: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        db: Option<RowIdentityDb>,
    },
}

impl RowIdentity {
    pub fn row_count(&self) -> usize {
        match self {
            Self::RowIdIdentity { row_count } | Self::Jsonl { row_count, .. } => *row_count,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowIdentityDb {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub table: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_column: Option<String>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ManifestIndexKind {
    Rank,
    RankQuant,
    Bitmap,
    SignBitmap,
}

impl ManifestIndexKind {
    fn try_from_core(kind: CoreIndexKind) -> Result<Self, UnsupportedCoreMetadata> {
        require_manifest_coverage(kind)?;
        match kind {
            CoreIndexKind::Rank => Ok(Self::Rank),
            CoreIndexKind::RankQuant => Ok(Self::RankQuant),
            CoreIndexKind::Bitmap => Ok(Self::Bitmap),
            CoreIndexKind::SignBitmap => Ok(Self::SignBitmap),
            other => Err(UnsupportedCoreMetadata::Kind(other)),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum ManifestIndexParams {
    Rank,
    RankQuant { bits: u8 },
    Bitmap { n_top: usize },
    SignBitmap,
}

impl ManifestIndexParams {
    fn try_from_core(params: CoreIndexParams) -> Result<Self, UnsupportedCoreMetadata> {
        match params {
            CoreIndexParams::Rank => Ok(Self::Rank),
            CoreIndexParams::RankQuant { bits } => Ok(Self::RankQuant { bits }),
            CoreIndexParams::Bitmap { n_top } => Ok(Self::Bitmap { n_top }),
            CoreIndexParams::SignBitmap => Ok(Self::SignBitmap),
            other => Err(UnsupportedCoreMetadata::Params(other)),
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum UnsupportedCoreMetadata {
    Kind(CoreIndexKind),
    Params(CoreIndexParams),
    RegistryMissing(CoreIndexKind),
    ManifestNotCovered {
        kind: CoreIndexKind,
        reason: &'static str,
    },
}

impl UnsupportedCoreMetadata {
    fn code(self) -> &'static str {
        match self {
            Self::Kind(_) => codes::ARTIFACT_KIND_UNSUPPORTED,
            Self::Params(_) => codes::ARTIFACT_PARAMS_UNSUPPORTED,
            Self::RegistryMissing(_) => codes::ARTIFACT_FORMAT_REGISTRY_MISSING,
            Self::ManifestNotCovered { .. } => codes::ARTIFACT_MANIFEST_COVERAGE_UNSUPPORTED,
        }
    }

    fn message(self) -> String {
        match self {
            Self::Kind(kind) => {
                format!("artifact metadata kind {kind:?} is not supported by ordvec-manifest v1")
            }
            Self::Params(params) => format!(
                "artifact metadata params {params:?} are not supported by ordvec-manifest v1"
            ),
            Self::RegistryMissing(kind) => format!(
                "artifact metadata kind {kind:?} has no ordvec persisted-format registry entry"
            ),
            Self::ManifestNotCovered { kind, reason } => format!(
                "artifact metadata kind {kind:?} is not covered by ordvec-manifest v1: {reason}"
            ),
        }
    }
}

fn format_spec_for_kind(
    kind: CoreIndexKind,
) -> Result<&'static FormatSpec, UnsupportedCoreMetadata> {
    FORMATS
        .iter()
        .find(|spec| spec.kind == kind)
        .ok_or(UnsupportedCoreMetadata::RegistryMissing(kind))
}

fn require_manifest_coverage(kind: CoreIndexKind) -> Result<(), UnsupportedCoreMetadata> {
    match format_spec_for_kind(kind)?.manifest {
        ManifestCoverage::Covered => Ok(()),
        ManifestCoverage::NotCovered { reason, .. } => {
            Err(UnsupportedCoreMetadata::ManifestNotCovered { kind, reason })
        }
        _ => Err(UnsupportedCoreMetadata::ManifestNotCovered {
            kind,
            reason: "unsupported manifest coverage stance in ordvec persisted-format registry",
        }),
    }
}

/// Stable identity of one artifact covered by a verified load plan.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct VerifiedArtifactIdentity {
    pub name: String,
    pub kind: VerifiedArtifactKind,
    pub expected_size_bytes: u64,
    pub expected_sha256: String,
}

/// Role and, for a primary artifact, persisted index kind.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifiedArtifactKind {
    Primary(ManifestIndexKind),
    Auxiliary,
}

/// An observed departure from the bytes recorded in a verified load plan.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum VerifiedArtifactChange {
    InitialSize { expected: u64, observed: u64 },
    FinalSize { expected: u64, observed: u64 },
    Digest { expected: String, observed: String },
}

/// Failure while consuming an artifact recorded in a verified load plan.
///
/// Stale-content errors take precedence over decoder and incomplete-consumption
/// errors after all declared bytes have been drained. Filesystem access and type
/// failures take precedence over content classification.
#[derive(Debug)]
#[non_exhaustive]
pub enum VerifiedArtifactUseError<E> {
    KindMismatch {
        identity: VerifiedArtifactIdentity,
        expected: ManifestIndexKind,
        observed: ManifestIndexKind,
    },
    OptionalAbsent {
        name: String,
    },
    Access {
        identity: VerifiedArtifactIdentity,
        stage: VerifiedArtifactAccessStage,
        source: io::Error,
    },
    TypeRejected {
        identity: VerifiedArtifactIdentity,
        rejection: VerifiedArtifactTypeRejection,
    },
    Stale {
        identity: VerifiedArtifactIdentity,
        changes: Vec<VerifiedArtifactChange>,
    },
    Decoder {
        identity: VerifiedArtifactIdentity,
        source: E,
    },
    IncompleteConsumption {
        identity: VerifiedArtifactIdentity,
        consumed: u64,
        expected: u64,
    },
}

impl<E: fmt::Display> fmt::Display for VerifiedArtifactUseError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KindMismatch {
                identity,
                expected,
                observed,
            } => write!(
                f,
                "artifact {:?} has kind {observed:?}, expected {expected:?}",
                identity.name
            ),
            Self::OptionalAbsent { name } => {
                write!(f, "optional auxiliary artifact {name:?} is absent")
            }
            Self::Access {
                identity,
                stage,
                source,
            } => write!(
                f,
                "cannot use artifact {:?} during {stage}: {source}",
                identity.name
            ),
            Self::TypeRejected {
                identity,
                rejection,
            } => write!(f, "artifact {:?} was rejected: {rejection}", identity.name),
            Self::Stale { identity, changes } => write!(
                f,
                "artifact {:?} is stale relative to the verified plan ({changes:?})",
                identity.name
            ),
            Self::Decoder { identity, source } => {
                write!(f, "artifact {:?} failed to decode: {source}", identity.name)
            }
            Self::IncompleteConsumption {
                identity,
                consumed,
                expected,
            } => write!(
                f,
                "artifact {:?} decoder consumed {consumed} of {expected} encoded bytes",
                identity.name
            ),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for VerifiedArtifactUseError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Access { source, .. } => Some(source),
            Self::Decoder { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Verified paths and metadata for a caller-managed load.
///
/// A `VerifiedLoadPlan` means the manifest, primary artifact, row-identity
/// file, and declared auxiliary artifacts verified at the time verification
/// ran. It is not a durable capability over mutable storage: the plan does not
/// pin file descriptors, hold locks, buffer bytes, or guarantee that bytes at
/// the returned paths remain unchanged after verification. Treat it as proof of
/// the verification just performed, then load from controlled storage
/// immediately or re-verify if another actor may have changed the files.
#[derive(Clone, Debug)]
pub struct VerifiedLoadPlan {
    manifest_path: Option<PathBuf>,
    artifact_path: PathBuf,
    primary_identity: VerifiedArtifactIdentity,
    metadata: MetadataReport,
    row_identity: VerifiedRowIdentityPlan,
    auxiliary_artifacts: Vec<VerifiedAuxiliaryArtifactPlan>,
    report: VerificationReport,
}

impl VerifiedLoadPlan {
    fn from_report(
        document: &ManifestDocument,
        report: VerificationReport,
        paths: VerificationPathCapture,
    ) -> Result<Self, VerifiedLoadPlanError> {
        if !report.ok {
            return Err(VerifiedLoadPlanError::VerificationFailed(Box::new(report)));
        }

        let artifact_path =
            paths
                .artifact_path
                .clone()
                .ok_or_else(|| VerifiedLoadPlanError::IncompletePlan {
                    report: Box::new(report.clone()),
                    message: "verified report is missing the captured artifact path".to_string(),
                })?;
        let metadata = report.artifact.metadata.clone().ok_or_else(|| {
            VerifiedLoadPlanError::IncompletePlan {
                report: Box::new(report.clone()),
                message: "verified report is missing probed artifact metadata".to_string(),
            }
        })?;
        let primary_identity = VerifiedArtifactIdentity {
            name: "primary".to_string(),
            kind: VerifiedArtifactKind::Primary(metadata.kind),
            expected_size_bytes: report.artifact.size_bytes.ok_or_else(|| {
                VerifiedLoadPlanError::IncompletePlan {
                    report: Box::new(report.clone()),
                    message: "verified report is missing primary artifact size".to_string(),
                }
            })?,
            expected_sha256: report.artifact.sha256.clone().ok_or_else(|| {
                VerifiedLoadPlanError::IncompletePlan {
                    report: Box::new(report.clone()),
                    message: "verified report is missing primary artifact digest".to_string(),
                }
            })?,
        };
        let row_identity =
            VerifiedRowIdentityPlan::from_report(paths.row_identity_path.as_ref(), &report)?;
        let auxiliary_artifacts = report
            .auxiliary_artifacts
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                VerifiedAuxiliaryArtifactPlan::from_report(
                    entry,
                    paths
                        .auxiliary_artifact_paths
                        .get(idx)
                        .and_then(|path| path.as_ref()),
                    &report,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            manifest_path: document.source_path.clone(),
            artifact_path,
            primary_identity,
            metadata,
            row_identity,
            auxiliary_artifacts,
            report,
        })
    }

    pub fn manifest_path(&self) -> Option<&Path> {
        self.manifest_path.as_deref()
    }

    /// Canonical path of the primary index artifact observed during verification.
    ///
    /// This path is not a byte pin. Loading later from mutable/shared storage can
    /// still observe different bytes, so callers that cannot control mutation
    /// must re-verify immediately before loading.
    pub fn artifact_path(&self) -> &Path {
        &self.artifact_path
    }

    pub fn primary_identity(&self) -> &VerifiedArtifactIdentity {
        &self.primary_identity
    }

    /// Decode the primary artifact from one no-follow descriptor while checking
    /// its plan-recorded kind, size, and digest.
    ///
    /// The `u64` passed to `decoder` is the exact encoded byte count remaining
    /// at the reader's current position. The reader is forward-only and never
    /// accesses bytes beyond that boundary.
    pub fn decode_primary_with<T, E>(
        &self,
        expected_kind: ManifestIndexKind,
        decoder: impl FnOnce(&mut dyn Read, u64) -> Result<T, E>,
    ) -> Result<T, VerifiedArtifactUseError<E>> {
        if self.metadata.kind != expected_kind {
            return Err(VerifiedArtifactUseError::KindMismatch {
                identity: self.primary_identity.clone(),
                expected: expected_kind,
                observed: self.metadata.kind,
            });
        }
        decode_plan_verified(&self.artifact_path, self.primary_identity.clone(), decoder)
    }

    pub fn metadata(&self) -> &MetadataReport {
        &self.metadata
    }

    pub fn row_identity(&self) -> &VerifiedRowIdentityPlan {
        &self.row_identity
    }

    pub fn auxiliary_artifacts(&self) -> &[VerifiedAuxiliaryArtifactPlan] {
        &self.auxiliary_artifacts
    }

    pub fn auxiliary_by_name(&self, name: &str) -> Option<&VerifiedAuxiliaryArtifactPlan> {
        let name = name.trim();
        self.auxiliary_artifacts
            .iter()
            .find(|artifact| artifact.name().trim() == name)
    }

    pub fn require_auxiliary(&self, name: &str) -> Result<&Path, RequireAuxiliaryError> {
        let artifact = self.auxiliary_by_name(name).ok_or_else(|| {
            RequireAuxiliaryError::MissingDeclaration {
                name: name.to_string(),
            }
        })?;
        artifact
            .path()
            .ok_or_else(|| RequireAuxiliaryError::NotLoadable {
                name: name.to_string(),
                state: artifact.state(),
                reason_code: artifact.reason_code().map(ToOwned::to_owned),
            })
    }

    pub fn report(&self) -> &VerificationReport {
        &self.report
    }

    pub fn into_report(self) -> VerificationReport {
        self.report
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequireAuxiliaryError {
    MissingDeclaration {
        name: String,
    },
    NotLoadable {
        name: String,
        state: AuxiliaryArtifactState,
        reason_code: Option<String>,
    },
}

impl fmt::Display for RequireAuxiliaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDeclaration { name } => {
                write!(f, "required auxiliary artifact {name:?} is not declared")
            }
            Self::NotLoadable {
                name,
                state,
                reason_code,
            } => {
                write!(
                    f,
                    "required auxiliary artifact {name:?} is not loadable: state={state:?}"
                )?;
                if let Some(reason_code) = reason_code {
                    write!(f, ", reason_code={reason_code}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for RequireAuxiliaryError {}

#[derive(Clone, Debug)]
pub struct VerifiedRowIdentityPlan {
    kind: String,
    path: Option<PathBuf>,
    row_count: usize,
    validated_rows: Option<usize>,
    sha256: Option<String>,
}

impl VerifiedRowIdentityPlan {
    fn from_report(
        captured_path: Option<&PathBuf>,
        report: &VerificationReport,
    ) -> Result<Self, VerifiedLoadPlanError> {
        let kind = report.row_identity.kind.clone().ok_or_else(|| {
            VerifiedLoadPlanError::IncompletePlan {
                report: Box::new(report.clone()),
                message: "verified report is missing row identity kind".to_string(),
            }
        })?;
        let row_count =
            report
                .row_identity
                .row_count
                .ok_or_else(|| VerifiedLoadPlanError::IncompletePlan {
                    report: Box::new(report.clone()),
                    message: "verified report is missing row identity row count".to_string(),
                })?;
        let path = match kind.as_str() {
            "row_id_identity" => None,
            "jsonl" => Some(captured_path.cloned().ok_or_else(|| {
                VerifiedLoadPlanError::IncompletePlan {
                    report: Box::new(report.clone()),
                    message: "verified report is missing the captured row identity path"
                        .to_string(),
                }
            })?),
            _ => {
                return Err(VerifiedLoadPlanError::IncompletePlan {
                    report: Box::new(report.clone()),
                    message: format!("verified report has unsupported row identity kind {kind:?}"),
                });
            }
        };

        Ok(Self {
            kind,
            path,
            row_count,
            validated_rows: report.row_identity.validated_rows,
            sha256: report.row_identity.sha256.clone(),
        })
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    pub fn validated_rows(&self) -> Option<usize> {
        self.validated_rows
    }

    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedAuxiliaryArtifactPlan {
    name: String,
    path: Option<PathBuf>,
    required: bool,
    state: AuxiliaryArtifactState,
    reason_code: Option<String>,
    sha256: Option<String>,
    size_bytes: Option<u64>,
}

impl VerifiedAuxiliaryArtifactPlan {
    fn from_report(
        entry: &AuxiliaryArtifactReport,
        captured_path: Option<&PathBuf>,
        report: &VerificationReport,
    ) -> Result<Self, VerifiedLoadPlanError> {
        let path = match entry.state {
            AuxiliaryArtifactState::Verified => Some(captured_path.cloned().ok_or_else(|| {
                VerifiedLoadPlanError::IncompletePlan {
                    report: Box::new(report.clone()),
                    message: format!(
                        "verified auxiliary artifact {:?} is missing its captured path",
                        entry.name
                    ),
                }
            })?),
            AuxiliaryArtifactState::OptionalAbsent => None,
            AuxiliaryArtifactState::MissingRequired | AuxiliaryArtifactState::Failed => {
                return Err(VerifiedLoadPlanError::IncompletePlan {
                    report: Box::new(report.clone()),
                    message: format!(
                        "verified report contains non-loadable auxiliary artifact {:?}",
                        entry.name
                    ),
                });
            }
        };
        let (sha256, size_bytes) =
            match entry.state {
                AuxiliaryArtifactState::Verified => (
                    Some(entry.sha256.clone().ok_or_else(|| {
                        VerifiedLoadPlanError::IncompletePlan {
                            report: Box::new(report.clone()),
                            message: format!(
                                "verified auxiliary artifact {:?} is missing its digest",
                                entry.name
                            ),
                        }
                    })?),
                    Some(entry.size_bytes.ok_or_else(|| {
                        VerifiedLoadPlanError::IncompletePlan {
                            report: Box::new(report.clone()),
                            message: format!(
                                "verified auxiliary artifact {:?} is missing its size",
                                entry.name
                            ),
                        }
                    })?),
                ),
                AuxiliaryArtifactState::OptionalAbsent => (None, None),
                AuxiliaryArtifactState::MissingRequired | AuxiliaryArtifactState::Failed => {
                    unreachable!("non-loadable auxiliary states returned above")
                }
            };

        Ok(Self {
            name: entry.name.clone(),
            path,
            required: entry.required,
            state: entry.state,
            reason_code: entry.reason_code.clone(),
            sha256,
            size_bytes,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn required(&self) -> bool {
        self.required
    }

    pub fn state(&self) -> AuxiliaryArtifactState {
        self.state
    }

    pub fn reason_code(&self) -> Option<&str> {
        self.reason_code.as_deref()
    }

    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    pub fn size_bytes(&self) -> Option<u64> {
        self.size_bytes
    }

    pub fn identity(&self) -> Option<VerifiedArtifactIdentity> {
        Some(VerifiedArtifactIdentity {
            name: self.name.clone(),
            kind: VerifiedArtifactKind::Auxiliary,
            expected_size_bytes: self.size_bytes?,
            expected_sha256: self.sha256.clone()?,
        })
    }

    /// Decode this auxiliary artifact from one no-follow descriptor while
    /// checking its plan-recorded size and digest.
    ///
    /// Optional-absent artifacts return [`VerifiedArtifactUseError::OptionalAbsent`]
    /// before any path is opened. The `u64` passed to `decoder` is the exact
    /// encoded byte count remaining at the reader's current position.
    pub fn decode_verified_with<T, E>(
        &self,
        decoder: impl FnOnce(&mut dyn Read, u64) -> Result<T, E>,
    ) -> Result<T, VerifiedArtifactUseError<E>> {
        if self.state == AuxiliaryArtifactState::OptionalAbsent {
            return Err(VerifiedArtifactUseError::OptionalAbsent {
                name: self.name.clone(),
            });
        }
        let identity = self
            .identity()
            .expect("verified auxiliary plans always carry a size and digest");
        let path = self
            .path
            .as_deref()
            .expect("verified auxiliary plans always carry a captured path");
        decode_plan_verified(path, identity, decoder)
    }
}

#[derive(Debug)]
pub enum VerifiedLoadPlanError {
    Manifest(ManifestError),
    VerificationFailed(Box<VerificationReport>),
    IncompletePlan {
        report: Box<VerificationReport>,
        message: String,
    },
}

impl fmt::Display for VerifiedLoadPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(err) => write!(f, "{err}"),
            Self::VerificationFailed(report) => {
                write!(
                    f,
                    "manifest verification failed{}",
                    report_issue_summary(&report.errors)
                )
            }
            Self::IncompletePlan { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for VerifiedLoadPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Manifest(err) => Some(err),
            Self::VerificationFailed(_) | Self::IncompletePlan { .. } => None,
        }
    }
}

impl From<ManifestError> for VerifiedLoadPlanError {
    fn from(value: ManifestError) -> Self {
        Self::Manifest(value)
    }
}

const VERIFIED_DECODE_DRAIN_BYTES: usize = 64 * 1024;

struct DigestingBoundedReader<'a, R: Read + ?Sized> {
    reader: &'a mut R,
    remaining: u64,
    consumed: u64,
    hasher: Sha256,
    read_error: Option<io::Error>,
}

impl<'a, R: Read + ?Sized> DigestingBoundedReader<'a, R> {
    fn new(reader: &'a mut R, encoded_len: u64) -> Self {
        Self {
            reader,
            remaining: encoded_len,
            consumed: 0,
            hasher: Sha256::new(),
            read_error: None,
        }
    }

    fn finish(self) -> (String, Option<io::Error>) {
        (hex::encode(self.hasher.finalize()), self.read_error)
    }
}

impl<R: Read + ?Sized> Read for DigestingBoundedReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 || buf.is_empty() {
            return Ok(0);
        }
        let limit = usize::try_from(self.remaining.min(buf.len() as u64))
            .expect("bounded read length never exceeds the caller buffer");
        loop {
            match self.reader.read(&mut buf[..limit]) {
                Ok(read) => {
                    self.hasher.update(&buf[..read]);
                    self.remaining -= read as u64;
                    self.consumed += read as u64;
                    return Ok(read);
                }
                Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
                Err(source) => {
                    if self.read_error.is_none() {
                        self.read_error = Some(clone_io_error(&source));
                    }
                    return Err(source);
                }
            }
        }
    }
}

fn clone_io_error(source: &io::Error) -> io::Error {
    match source.raw_os_error() {
        Some(code) => io::Error::from_raw_os_error(code),
        None => io::Error::new(source.kind(), source.to_string()),
    }
}

fn decode_plan_verified<T, E>(
    path: &Path,
    identity: VerifiedArtifactIdentity,
    decoder: impl FnOnce(&mut dyn Read, u64) -> Result<T, E>,
) -> Result<T, VerifiedArtifactUseError<E>> {
    let (mut file, initial_metadata) = match open_regular_file_no_follow(path) {
        Ok(opened) => opened,
        Err(OpenRegularFileError::Access { stage, source }) => {
            return Err(VerifiedArtifactUseError::Access {
                identity,
                stage,
                source,
            });
        }
        Err(OpenRegularFileError::Type(rejection)) => {
            return Err(VerifiedArtifactUseError::TypeRejected {
                identity,
                rejection,
            });
        }
    };

    let expected_size = identity.expected_size_bytes;
    let initial_size = initial_metadata.len();
    if initial_size != expected_size {
        return Err(VerifiedArtifactUseError::Stale {
            identity,
            changes: vec![VerifiedArtifactChange::InitialSize {
                expected: expected_size,
                observed: initial_size,
            }],
        });
    }

    let mut reader = DigestingBoundedReader::new(&mut file, expected_size);
    let decoded = decoder(&mut reader, expected_size);
    let decoder_consumed = reader.consumed;

    if reader.remaining != 0 {
        // Keep the fixed verification scratch off small caller stacks, and
        // preserve the public recoverable-error contract if it cannot be
        // allocated. Fully consuming decoders do not need this allocation.
        let mut scratch = Vec::new();
        if scratch
            .try_reserve_exact(VERIFIED_DECODE_DRAIN_BYTES)
            .is_err()
        {
            return Err(VerifiedArtifactUseError::Access {
                identity,
                stage: VerifiedArtifactAccessStage::Read,
                source: io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    "verification read scratch allocation failed",
                ),
            });
        }
        scratch.resize(VERIFIED_DECODE_DRAIN_BYTES, 0);
        while reader.remaining != 0 {
            match reader.read(&mut scratch) {
                Ok(0) => break,
                Ok(_) => {}
                Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
    }
    let (observed_digest, read_error) = reader.finish();

    if let Some(source) = read_error {
        return Err(VerifiedArtifactUseError::Access {
            identity,
            stage: VerifiedArtifactAccessStage::Read,
            source,
        });
    }

    let final_size = file
        .metadata()
        .map_err(|source| VerifiedArtifactUseError::Access {
            identity: identity.clone(),
            stage: VerifiedArtifactAccessStage::FinalMetadata,
            source,
        })?
        .len();

    let mut changes = Vec::new();
    if final_size != expected_size {
        changes.push(VerifiedArtifactChange::FinalSize {
            expected: expected_size,
            observed: final_size,
        });
    }
    if observed_digest != identity.expected_sha256 {
        changes.push(VerifiedArtifactChange::Digest {
            expected: identity.expected_sha256.clone(),
            observed: observed_digest,
        });
    }
    if !changes.is_empty() {
        return Err(VerifiedArtifactUseError::Stale { identity, changes });
    }

    match decoded {
        Err(source) => Err(VerifiedArtifactUseError::Decoder { identity, source }),
        Ok(_) if decoder_consumed != expected_size => {
            Err(VerifiedArtifactUseError::IncompleteConsumption {
                identity,
                consumed: decoder_consumed,
                expected: expected_size,
            })
        }
        Ok(value) => Ok(value),
    }
}

fn report_issue_summary(errors: &[ReportIssue]) -> String {
    if errors.is_empty() {
        return String::new();
    }
    let codes = errors
        .iter()
        .take(3)
        .map(|issue| issue.code.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    if errors.len() > 3 {
        format!(": {codes}, ...")
    } else {
        format!(": {codes}")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationReport {
    pub ok: bool,
    pub checked_at: String,
    pub artifact: ArtifactReport,
    #[serde(default)]
    pub auxiliary_artifacts: Vec<AuxiliaryArtifactReport>,
    pub row_identity: RowIdentityReport,
    #[serde(default)]
    pub encoder_distortion: EncoderDistortionReport,
    pub calibration: CalibrationReport,
    pub attestation_shape_checks: Vec<AttestationShapeCheck>,
    pub errors: Vec<ReportIssue>,
    pub warnings: Vec<ReportIssue>,
    pub skipped_checks: Vec<String>,
}

impl VerificationReport {
    fn new() -> Self {
        Self {
            ok: false,
            checked_at: Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true),
            artifact: ArtifactReport::default(),
            auxiliary_artifacts: Vec::new(),
            row_identity: RowIdentityReport::default(),
            encoder_distortion: EncoderDistortionReport::default(),
            calibration: CalibrationReport::default(),
            attestation_shape_checks: Vec::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            skipped_checks: Vec::new(),
        }
    }

    fn error(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.errors.push(ReportIssue::new(code, message));
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ArtifactReport {
    pub manifest_path: Option<String>,
    pub observed_path: Option<String>,
    pub canonical_path: Option<String>,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
    pub metadata: Option<MetadataReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuxiliaryArtifactReport {
    pub name: String,
    pub manifest_path: String,
    #[serde(default)]
    pub resolved_path: Option<String>,
    #[serde(default)]
    pub canonical_path: Option<String>,
    #[serde(default)]
    pub expected_sha256: Option<String>,
    #[serde(default)]
    pub expected_size_bytes: Option<u64>,
    pub required: bool,
    pub state: AuxiliaryArtifactState,
    pub reason_code: Option<String>,
    pub sha256: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuxiliaryArtifactState {
    Verified,
    OptionalAbsent,
    MissingRequired,
    Failed,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RowIdentityReport {
    pub kind: Option<String>,
    pub manifest_path: Option<String>,
    pub canonical_path: Option<String>,
    pub sha256: Option<String>,
    pub row_count: Option<usize>,
    pub validated_rows: Option<usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EncoderDistortionReport {
    pub present: bool,
    pub schema_version: Option<String>,
    pub profile_id: Option<String>,
    pub evidence_kind: Option<String>,
    pub source_metric: Option<String>,
    pub embedding_metric: Option<String>,
    pub profile_manifest_path: Option<String>,
    pub profile_canonical_path: Option<String>,
    pub profile_sha256: Option<String>,
    pub profile_size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub present: bool,
    pub schema_version: Option<String>,
    pub profile_id: Option<String>,
    pub calibrated_for_model: Option<String>,
    pub ordinalization: Option<String>,
    pub null_model: Option<String>,
    pub profile_manifest_path: Option<String>,
    pub profile_canonical_path: Option<String>,
    pub profile_sha256: Option<String>,
    pub profile_size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetadataReport {
    pub kind: ManifestIndexKind,
    pub format_version: u8,
    pub dim: usize,
    pub vector_count: usize,
    pub bytes_per_vec: usize,
    pub params: ManifestIndexParams,
    pub file_size_bytes: u64,
}

impl MetadataReport {
    fn try_from_core(metadata: &CoreIndexMetadata) -> Result<Self, UnsupportedCoreMetadata> {
        Ok(Self {
            kind: ManifestIndexKind::try_from_core(metadata.kind)?,
            format_version: metadata.format_version,
            dim: metadata.dim,
            vector_count: metadata.vector_count,
            bytes_per_vec: metadata.bytes_per_vec,
            params: ManifestIndexParams::try_from_core(metadata.params)?,
            file_size_bytes: metadata.file_size_bytes,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttestationShapeCheck {
    pub predicate_type: Option<String>,
    pub builder_id: Option<String>,
    pub subject_sha256_matched: bool,
}

/// Stable machine-readable issue codes.
///
/// Every code emitted through [`crate::ReportIssue`] (including
/// [`crate::AuxiliaryArtifactReport`] reason codes and
/// [`crate::ManifestError::LimitExceeded`]) is named here so downstream
/// consumers can branch on constants instead of retyping string literals.
pub mod codes {
    pub const ARTIFACT_ABSOLUTE_PATH_REJECTED: &str = "artifact_absolute_path_rejected";
    pub const ARTIFACT_ABSOLUTE_PATH_UNRESOLVABLE: &str = "artifact_absolute_path_unresolvable";
    pub const ARTIFACT_BASE_DIR_UNAVAILABLE: &str = "artifact_base_dir_unavailable";
    pub const ARTIFACT_BYTES_PER_VEC_MISMATCH: &str = "artifact_bytes_per_vec_mismatch";
    pub const ARTIFACT_BYTES_PER_VEC_ZERO: &str = "artifact_bytes_per_vec_zero";
    pub const ARTIFACT_DIM_MISMATCH: &str = "artifact_dim_mismatch";
    pub const ARTIFACT_EMBEDDING_DIM_MISMATCH: &str = "artifact_embedding_dim_mismatch";
    pub const ARTIFACT_FILE_SIZE_MISMATCH: &str = "artifact_file_size_mismatch";
    pub const ARTIFACT_FILE_SIZE_ZERO: &str = "artifact_file_size_zero";
    pub const ARTIFACT_FILE_TOO_LARGE: &str = "artifact_file_too_large";
    pub const ARTIFACT_FORMAT_REGISTRY_MISSING: &str = "artifact_format_registry_missing";
    pub const ARTIFACT_FORMAT_VERSION_MISMATCH: &str = "artifact_format_version_mismatch";
    pub const ARTIFACT_HASH_FAILED: &str = "artifact_hash_failed";
    pub const ARTIFACT_KIND_MISMATCH: &str = "artifact_kind_mismatch";
    pub const ARTIFACT_KIND_UNSUPPORTED: &str = "artifact_kind_unsupported";
    pub const ARTIFACT_MANIFEST_COVERAGE_UNSUPPORTED: &str =
        "artifact_manifest_coverage_unsupported";
    pub const ARTIFACT_METADATA_FILE_SIZE_MISMATCH: &str = "artifact_metadata_file_size_mismatch";
    pub const ARTIFACT_MISSING: &str = "artifact_missing";
    pub const ARTIFACT_PARAMS_KIND_MISMATCH: &str = "artifact_params_kind_mismatch";
    pub const ARTIFACT_PARAMS_MISMATCH: &str = "artifact_params_mismatch";
    pub const ARTIFACT_PARAMS_UNSUPPORTED: &str = "artifact_params_unsupported";
    pub const ARTIFACT_PATH_EMPTY: &str = "artifact_path_empty";
    pub const ARTIFACT_PATH_ESCAPE_REJECTED: &str = "artifact_path_escape_rejected";
    pub const ARTIFACT_PATH_NOT_CANONICAL: &str = "artifact_path_not_canonical";
    pub const ARTIFACT_PATH_UNAVAILABLE: &str = "artifact_path_unavailable";
    pub const ARTIFACT_PROBE_FAILED: &str = "artifact_probe_failed";
    pub const ARTIFACT_ROW_COUNT_MISMATCH: &str = "artifact_row_count_mismatch";
    pub const ARTIFACT_SHA256_INVALID: &str = "artifact_sha256_invalid";
    pub const ARTIFACT_SHA256_MISMATCH: &str = "artifact_sha256_mismatch";
    pub const ARTIFACT_VECTOR_COUNT_MISMATCH: &str = "artifact_vector_count_mismatch";
    pub const ATTESTATION_PREDICATE_TYPE_MISSING: &str = "attestation_predicate_type_missing";
    pub const ATTESTATION_SUBJECT_SHA256_MISMATCH: &str = "attestation_subject_sha256_mismatch";
    pub const AUXILIARY_ARTIFACT_ABSOLUTE_PATH_REJECTED: &str =
        "auxiliary_artifact_absolute_path_rejected";
    pub const AUXILIARY_ARTIFACT_ABSOLUTE_PATH_UNRESOLVABLE: &str =
        "auxiliary_artifact_absolute_path_unresolvable";
    pub const AUXILIARY_ARTIFACT_BASE_DIR_UNAVAILABLE: &str =
        "auxiliary_artifact_base_dir_unavailable";
    pub const AUXILIARY_ARTIFACT_COUNT_LIMIT_EXCEEDED: &str =
        "auxiliary_artifact_count_limit_exceeded";
    pub const AUXILIARY_ARTIFACT_FILE_SIZE_MISMATCH: &str = "auxiliary_artifact_file_size_mismatch";
    pub const AUXILIARY_ARTIFACT_FILE_SIZE_ZERO: &str = "auxiliary_artifact_file_size_zero";
    pub const AUXILIARY_ARTIFACT_FILE_TOO_LARGE: &str = "auxiliary_artifact_file_too_large";
    pub const AUXILIARY_ARTIFACT_HASH_FAILED: &str = "auxiliary_artifact_hash_failed";
    pub const AUXILIARY_ARTIFACT_MISSING_REQUIRED: &str = "auxiliary_artifact_missing_required";
    pub const AUXILIARY_ARTIFACT_NAME_DUPLICATE: &str = "auxiliary_artifact_name_duplicate";
    pub const AUXILIARY_ARTIFACT_NAME_EMPTY: &str = "auxiliary_artifact_name_empty";
    pub const AUXILIARY_ARTIFACT_NAME_NOT_TRIMMED: &str = "auxiliary_artifact_name_not_trimmed";
    pub const AUXILIARY_ARTIFACT_OPTIONAL_ABSENT: &str = "auxiliary_artifact_optional_absent";
    pub const AUXILIARY_ARTIFACT_PATH_EMPTY: &str = "auxiliary_artifact_path_empty";
    pub const AUXILIARY_ARTIFACT_PATH_ESCAPE_REJECTED: &str =
        "auxiliary_artifact_path_escape_rejected";
    pub const AUXILIARY_ARTIFACT_PATH_NOT_CANONICAL: &str = "auxiliary_artifact_path_not_canonical";
    pub const AUXILIARY_ARTIFACT_PATH_UNAVAILABLE: &str = "auxiliary_artifact_path_unavailable";
    pub const AUXILIARY_ARTIFACT_SHA256_INVALID: &str = "auxiliary_artifact_sha256_invalid";
    pub const AUXILIARY_ARTIFACT_SHA256_MISMATCH: &str = "auxiliary_artifact_sha256_mismatch";
    pub const BUILD_BUILDER_ID_EMPTY: &str = "build_builder_id_empty";
    pub const BUILD_CI_PROVIDER_EMPTY: &str = "build_ci_provider_empty";
    pub const BUILD_CI_RUN_ID_EMPTY: &str = "build_ci_run_id_empty";
    pub const BUILD_INVOCATION_ID_EMPTY: &str = "build_invocation_id_empty";
    pub const BUILD_SOURCE_COMMIT_EMPTY: &str = "build_source_commit_empty";
    pub const BUILD_SOURCE_REPO_EMPTY: &str = "build_source_repo_empty";
    pub const CALIBRATION_CREATED_AT_INVALID: &str = "calibration_created_at_invalid";
    pub const CALIBRATION_ENCODER_DIM_MISMATCH: &str = "calibration_encoder_dim_mismatch";
    pub const CALIBRATION_ENCODER_DIM_ZERO: &str = "calibration_encoder_dim_zero";
    pub const CALIBRATION_ENCODER_MODEL_EMPTY: &str = "calibration_encoder_model_empty";
    pub const CALIBRATION_ENCODER_MODEL_MISMATCH: &str = "calibration_encoder_model_mismatch";
    pub const CALIBRATION_ENCODER_MODEL_REVISION_EMPTY: &str =
        "calibration_encoder_model_revision_empty";
    pub const CALIBRATION_ENCODER_MODEL_REVISION_MISMATCH: &str =
        "calibration_encoder_model_revision_mismatch";
    pub const CALIBRATION_ENCODER_NORMALIZATION_EMPTY: &str =
        "calibration_encoder_normalization_empty";
    pub const CALIBRATION_ENCODER_NORMALIZATION_MISMATCH: &str =
        "calibration_encoder_normalization_mismatch";
    pub const CALIBRATION_NULL_MODEL_ORDINALIZATION_MISMATCH: &str =
        "calibration_null_model_ordinalization_mismatch";
    pub const CALIBRATION_NULL_NAME_EMPTY: &str = "calibration_null_name_empty";
    pub const CALIBRATION_NULL_PARAMETERIZATION_EMPTY: &str =
        "calibration_null_parameterization_empty";
    pub const CALIBRATION_NULL_PARAMETERIZATION_MISMATCH: &str =
        "calibration_null_parameterization_mismatch";
    pub const CALIBRATION_NULL_STATISTIC_EMPTY: &str = "calibration_null_statistic_empty";
    pub const CALIBRATION_ORDINALIZATION_ARTIFACT_MISMATCH: &str =
        "calibration_ordinalization_artifact_mismatch";
    pub const CALIBRATION_ORDINALIZATION_DIM_MISMATCH: &str =
        "calibration_ordinalization_dim_mismatch";
    pub const CALIBRATION_ORDINALIZATION_DIM_ZERO: &str = "calibration_ordinalization_dim_zero";
    pub const CALIBRATION_PROFILE_ABSOLUTE_PATH_REJECTED: &str =
        "calibration_profile_absolute_path_rejected";
    pub const CALIBRATION_PROFILE_ABSOLUTE_PATH_UNRESOLVABLE: &str =
        "calibration_profile_absolute_path_unresolvable";
    pub const CALIBRATION_PROFILE_BASE_DIR_UNAVAILABLE: &str =
        "calibration_profile_base_dir_unavailable";
    pub const CALIBRATION_PROFILE_DIM_MISMATCH: &str = "calibration_profile_dim_mismatch";
    pub const CALIBRATION_PROFILE_FILE_SIZE_MISMATCH: &str =
        "calibration_profile_file_size_mismatch";
    pub const CALIBRATION_PROFILE_FILE_SIZE_ZERO: &str = "calibration_profile_file_size_zero";
    pub const CALIBRATION_PROFILE_FORMAT_EMPTY: &str = "calibration_profile_format_empty";
    pub const CALIBRATION_PROFILE_HASH_FAILED: &str = "calibration_profile_hash_failed";
    pub const CALIBRATION_PROFILE_ID_EMPTY: &str = "calibration_profile_id_empty";
    pub const CALIBRATION_PROFILE_PARAMETERIZATION_ORDINALIZATION_MISMATCH: &str =
        "calibration_profile_parameterization_ordinalization_mismatch";
    pub const CALIBRATION_PROFILE_PATH_EMPTY: &str = "calibration_profile_path_empty";
    pub const CALIBRATION_PROFILE_PATH_ESCAPE_REJECTED: &str =
        "calibration_profile_path_escape_rejected";
    pub const CALIBRATION_PROFILE_PATH_NOT_CANONICAL: &str =
        "calibration_profile_path_not_canonical";
    pub const CALIBRATION_PROFILE_PATH_UNAVAILABLE: &str = "calibration_profile_path_unavailable";
    pub const CALIBRATION_PROFILE_REQUIRED: &str = "calibration_profile_required";
    pub const CALIBRATION_PROFILE_SAMPLE_COUNT_ZERO: &str = "calibration_profile_sample_count_zero";
    pub const CALIBRATION_PROFILE_SHA256_INVALID: &str = "calibration_profile_sha256_invalid";
    pub const CALIBRATION_PROFILE_SHA256_MISMATCH: &str = "calibration_profile_sha256_mismatch";
    pub const CALIBRATION_PROFILE_SHAPE_MISMATCH: &str = "calibration_profile_shape_mismatch";
    pub const CALIBRATION_PROFILE_SOURCE_DIGEST_INVALID: &str =
        "calibration_profile_source_digest_invalid";
    pub const CALIBRATION_PROFILE_TOO_LARGE: &str = "calibration_profile_too_large";
    pub const CALIBRATION_PROFILE_UNEXPECTED: &str = "calibration_profile_unexpected";
    pub const CALIBRATION_SCHEMA_VERSION_UNSUPPORTED: &str =
        "calibration_schema_version_unsupported";
    pub const EMBEDDING_CORPUS_DIGEST_INVALID: &str = "embedding_corpus_digest_invalid";
    pub const EMBEDDING_DIM_ZERO: &str = "embedding_dim_zero";
    pub const EMBEDDING_MATRIX_DIGEST_INVALID: &str = "embedding_matrix_digest_invalid";
    pub const EMBEDDING_MODEL_EMPTY: &str = "embedding_model_empty";
    pub const EMBEDDING_MODEL_REVISION_EMPTY: &str = "embedding_model_revision_empty";
    pub const EMBEDDING_NORMALIZATION_EMPTY: &str = "embedding_normalization_empty";
    pub const EMBEDDING_POOLING_EMPTY: &str = "embedding_pooling_empty";
    pub const EMBEDDING_TOKENIZER_REVISION_EMPTY: &str = "embedding_tokenizer_revision_empty";
    pub const ENCODER_DISTORTION_BOUNDS_EMPTY: &str = "encoder_distortion_bounds_empty";
    pub const ENCODER_DISTORTION_BOUNDS_ORDER_INVALID: &str =
        "encoder_distortion_bounds_order_invalid";
    pub const ENCODER_DISTORTION_CALIBRATION_MISSING: &str =
        "encoder_distortion_calibration_missing";
    pub const ENCODER_DISTORTION_CALIBRATION_PROFILE_ID_EMPTY: &str =
        "encoder_distortion_calibration_profile_id_empty";
    pub const ENCODER_DISTORTION_CALIBRATION_PROFILE_ID_WHITESPACE: &str =
        "encoder_distortion_calibration_profile_id_whitespace";
    pub const ENCODER_DISTORTION_CALIBRATION_PROFILE_MISMATCH: &str =
        "encoder_distortion_calibration_profile_mismatch";
    pub const ENCODER_DISTORTION_CREATED_AT_INVALID: &str = "encoder_distortion_created_at_invalid";
    pub const ENCODER_DISTORTION_DISTORTION_MISMATCH: &str =
        "encoder_distortion_distortion_mismatch";
    pub const ENCODER_DISTORTION_EMBEDDING_METRIC_DIGEST_INVALID: &str =
        "encoder_distortion_embedding_metric_digest_invalid";
    pub const ENCODER_DISTORTION_EMBEDDING_METRIC_NAME_EMPTY: &str =
        "encoder_distortion_embedding_metric_name_empty";
    pub const ENCODER_DISTORTION_EMBEDDING_METRIC_VERSION_EMPTY: &str =
        "encoder_distortion_embedding_metric_version_empty";
    pub const ENCODER_DISTORTION_ENCODER_DIM_MISMATCH: &str =
        "encoder_distortion_encoder_dim_mismatch";
    pub const ENCODER_DISTORTION_ENCODER_DIM_ZERO: &str = "encoder_distortion_encoder_dim_zero";
    pub const ENCODER_DISTORTION_ENCODER_MODEL_EMPTY: &str =
        "encoder_distortion_encoder_model_empty";
    pub const ENCODER_DISTORTION_ENCODER_MODEL_MISMATCH: &str =
        "encoder_distortion_encoder_model_mismatch";
    pub const ENCODER_DISTORTION_ENCODER_MODEL_REVISION_EMPTY: &str =
        "encoder_distortion_encoder_model_revision_empty";
    pub const ENCODER_DISTORTION_ENCODER_MODEL_REVISION_MISMATCH: &str =
        "encoder_distortion_encoder_model_revision_mismatch";
    pub const ENCODER_DISTORTION_ENCODER_NORMALIZATION_EMPTY: &str =
        "encoder_distortion_encoder_normalization_empty";
    pub const ENCODER_DISTORTION_ENCODER_NORMALIZATION_MISMATCH: &str =
        "encoder_distortion_encoder_normalization_mismatch";
    pub const ENCODER_DISTORTION_ESTIMATED_DISTORTION_INVALID: &str =
        "encoder_distortion_estimated_distortion_invalid";
    pub const ENCODER_DISTORTION_EVIDENCE_ESTIMATOR_HASH_INVALID: &str =
        "encoder_distortion_evidence_estimator_hash_invalid";
    pub const ENCODER_DISTORTION_EVIDENCE_ESTIMATOR_ID_EMPTY: &str =
        "encoder_distortion_evidence_estimator_id_empty";
    pub const ENCODER_DISTORTION_LOWER_BOUND_INVALID: &str =
        "encoder_distortion_lower_bound_invalid";
    pub const ENCODER_DISTORTION_MAX_OBSERVED_VIOLATION_INVALID: &str =
        "encoder_distortion_max_observed_violation_invalid";
    pub const ENCODER_DISTORTION_POOLING_EMPTY: &str = "encoder_distortion_pooling_empty";
    pub const ENCODER_DISTORTION_POOLING_MISMATCH: &str = "encoder_distortion_pooling_mismatch";
    pub const ENCODER_DISTORTION_PROFILE_ABSOLUTE_PATH_REJECTED: &str =
        "encoder_distortion_profile_absolute_path_rejected";
    pub const ENCODER_DISTORTION_PROFILE_ABSOLUTE_PATH_UNRESOLVABLE: &str =
        "encoder_distortion_profile_absolute_path_unresolvable";
    pub const ENCODER_DISTORTION_PROFILE_BASE_DIR_UNAVAILABLE: &str =
        "encoder_distortion_profile_base_dir_unavailable";
    pub const ENCODER_DISTORTION_PROFILE_FILE_SIZE_MISMATCH: &str =
        "encoder_distortion_profile_file_size_mismatch";
    pub const ENCODER_DISTORTION_PROFILE_FILE_SIZE_ZERO: &str =
        "encoder_distortion_profile_file_size_zero";
    pub const ENCODER_DISTORTION_PROFILE_FORMAT_EMPTY: &str =
        "encoder_distortion_profile_format_empty";
    pub const ENCODER_DISTORTION_PROFILE_HASH_FAILED: &str =
        "encoder_distortion_profile_hash_failed";
    pub const ENCODER_DISTORTION_PROFILE_ID_EMPTY: &str = "encoder_distortion_profile_id_empty";
    pub const ENCODER_DISTORTION_PROFILE_PATH_EMPTY: &str = "encoder_distortion_profile_path_empty";
    pub const ENCODER_DISTORTION_PROFILE_PATH_ESCAPE_REJECTED: &str =
        "encoder_distortion_profile_path_escape_rejected";
    pub const ENCODER_DISTORTION_PROFILE_PATH_NOT_CANONICAL: &str =
        "encoder_distortion_profile_path_not_canonical";
    pub const ENCODER_DISTORTION_PROFILE_PATH_UNAVAILABLE: &str =
        "encoder_distortion_profile_path_unavailable";
    pub const ENCODER_DISTORTION_PROFILE_REQUIRED: &str = "encoder_distortion_profile_required";
    pub const ENCODER_DISTORTION_PROFILE_SHA256_INVALID: &str =
        "encoder_distortion_profile_sha256_invalid";
    pub const ENCODER_DISTORTION_PROFILE_SHA256_MISMATCH: &str =
        "encoder_distortion_profile_sha256_mismatch";
    pub const ENCODER_DISTORTION_PROFILE_SOURCE_DIGEST_INVALID: &str =
        "encoder_distortion_profile_source_digest_invalid";
    pub const ENCODER_DISTORTION_PROFILE_TOO_LARGE: &str = "encoder_distortion_profile_too_large";
    pub const ENCODER_DISTORTION_QUANTILE_OBSERVED_VIOLATION_INVALID: &str =
        "encoder_distortion_quantile_observed_violation_invalid";
    pub const ENCODER_DISTORTION_SCHEMA_VERSION_UNSUPPORTED: &str =
        "encoder_distortion_schema_version_unsupported";
    pub const ENCODER_DISTORTION_SCOPE_CONFIDENCE_INVALID: &str =
        "encoder_distortion_scope_confidence_invalid";
    pub const ENCODER_DISTORTION_SCOPE_CORPUS_DIGEST_INVALID: &str =
        "encoder_distortion_scope_corpus_digest_invalid";
    pub const ENCODER_DISTORTION_SCOPE_COVERAGE_INVALID: &str =
        "encoder_distortion_scope_coverage_invalid";
    pub const ENCODER_DISTORTION_SCOPE_DOMAIN_EMPTY: &str = "encoder_distortion_scope_domain_empty";
    pub const ENCODER_DISTORTION_SCOPE_ESTIMATOR_VERSION_EMPTY: &str =
        "encoder_distortion_scope_estimator_version_empty";
    pub const ENCODER_DISTORTION_SCOPE_PAIR_SAMPLE_DIGEST_INVALID: &str =
        "encoder_distortion_scope_pair_sample_digest_invalid";
    pub const ENCODER_DISTORTION_SCOPE_QUERY_SET_DIGEST_INVALID: &str =
        "encoder_distortion_scope_query_set_digest_invalid";
    pub const ENCODER_DISTORTION_SCOPE_SAMPLE_SIZE_ZERO: &str =
        "encoder_distortion_scope_sample_size_zero";
    pub const ENCODER_DISTORTION_SOURCE_METRIC_DIGEST_INVALID: &str =
        "encoder_distortion_source_metric_digest_invalid";
    pub const ENCODER_DISTORTION_SOURCE_METRIC_NAME_EMPTY: &str =
        "encoder_distortion_source_metric_name_empty";
    pub const ENCODER_DISTORTION_SOURCE_METRIC_VERSION_EMPTY: &str =
        "encoder_distortion_source_metric_version_empty";
    pub const ENCODER_DISTORTION_TOKENIZER_REVISION_EMPTY: &str =
        "encoder_distortion_tokenizer_revision_empty";
    pub const ENCODER_DISTORTION_TOKENIZER_REVISION_MISMATCH: &str =
        "encoder_distortion_tokenizer_revision_mismatch";
    pub const ENCODER_DISTORTION_UPPER_BOUND_INVALID: &str =
        "encoder_distortion_upper_bound_invalid";
    pub const ENCODER_DISTORTION_VIOLATION_RATE_INVALID: &str =
        "encoder_distortion_violation_rate_invalid";
    pub const EXTENSION_KEY_NOT_NAMESPACED: &str = "extension_key_not_namespaced";
    pub const MANIFEST_FILE_TOO_LARGE: &str = "manifest_file_too_large";
    pub const ROW_IDENTITY_ABSOLUTE_PATH_REJECTED: &str = "row_identity_absolute_path_rejected";
    pub const ROW_IDENTITY_ABSOLUTE_PATH_UNRESOLVABLE: &str =
        "row_identity_absolute_path_unresolvable";
    pub const ROW_IDENTITY_BASE_DIR_UNAVAILABLE: &str = "row_identity_base_dir_unavailable";
    pub const ROW_IDENTITY_DB_ID_CONTAINS_NUL: &str = "row_identity_db_id_contains_nul";
    pub const ROW_IDENTITY_DB_ID_EMPTY: &str = "row_identity_db_id_empty";
    pub const ROW_IDENTITY_DB_ID_INVALID_UUID: &str = "row_identity_db_id_invalid_uuid";
    pub const ROW_IDENTITY_DB_UNSUPPORTED: &str = "row_identity_db_unsupported";
    pub const ROW_IDENTITY_DUPLICATE_DB_ID: &str = "row_identity_duplicate_db_id";
    pub const ROW_IDENTITY_DUPLICATE_TRACKING_LIMIT_EXCEEDED: &str =
        "row_identity_duplicate_tracking_limit_exceeded";
    pub const ROW_IDENTITY_ID_KIND_UNSUPPORTED: &str = "row_identity_id_kind_unsupported";
    pub const ROW_IDENTITY_JSONL_INVALID_JSON: &str = "row_identity_jsonl_invalid_json";
    pub const ROW_IDENTITY_LINE_TOO_LARGE: &str = "row_identity_line_too_large";
    pub const ROW_IDENTITY_MISSING: &str = "row_identity_missing";
    pub const ROW_IDENTITY_PARENT_ID_CONTAINS_NUL: &str = "row_identity_parent_id_contains_nul";
    pub const ROW_IDENTITY_PARENT_ID_EMPTY: &str = "row_identity_parent_id_empty";
    pub const ROW_IDENTITY_PARENT_ID_INVALID_UUID: &str = "row_identity_parent_id_invalid_uuid";
    pub const ROW_IDENTITY_PATH_EMPTY: &str = "row_identity_path_empty";
    pub const ROW_IDENTITY_PATH_ESCAPE_REJECTED: &str = "row_identity_path_escape_rejected";
    pub const ROW_IDENTITY_PATH_NOT_CANONICAL: &str = "row_identity_path_not_canonical";
    pub const ROW_IDENTITY_PATH_UNAVAILABLE: &str = "row_identity_path_unavailable";
    pub const ROW_IDENTITY_READ_FAILED: &str = "row_identity_read_failed";
    pub const ROW_IDENTITY_ROW_COUNT_LIMIT_EXCEEDED: &str = "row_identity_row_count_limit_exceeded";
    pub const ROW_IDENTITY_ROW_COUNT_MISMATCH: &str = "row_identity_row_count_mismatch";
    pub const ROW_IDENTITY_ROW_ID_MISMATCH: &str = "row_identity_row_id_mismatch";
    pub const ROW_IDENTITY_SHA256_INVALID: &str = "row_identity_sha256_invalid";
    pub const ROW_IDENTITY_SHA256_MISMATCH: &str = "row_identity_sha256_mismatch";
    pub const SCHEMA_VERSION_UNSUPPORTED: &str = "schema_version_unsupported";
    pub const SQLITE_ACTIVATION_FORCED: &str = "sqlite_activation_forced";
    pub const SQLITE_CACHED_REPORT_TOO_LARGE: &str = "sqlite_cached_report_too_large";
    pub const VERIFICATION_REPORT_ISSUE_LIMIT_EXCEEDED: &str =
        "verification_report_issue_limit_exceeded";
}

/// Typed classification of [`ReportIssue::code`] values so downstream
/// security code can branch on enum variants instead of string compares.
///
/// The integrity-mismatch, missing-mandatory-file, schema-version, and
/// resource-limit code families are classified into typed variants. Every
/// other code maps to [`VerificationCode::Unknown`] — including the
/// path-policy rejections (absolute / escape / non-canonical) and the
/// diagnostic I/O failures (`*_path_unavailable`, `*_hash_failed`), which are
/// deliberately not distinguished as typed variants. The enum is
/// [`#[non_exhaustive]`](https://doc.rust-lang.org/reference/attributes/type_system.html),
/// so treat `Unknown` as the required catch-all. Manifest parse failures never
/// reach a report (they surface as [`ManifestError`]), so the schema family
/// covers only the in-report [`codes::SCHEMA_VERSION_UNSUPPORTED`] check.
#[non_exhaustive]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum VerificationCode {
    ArtifactSha256Mismatch,
    ArtifactFileSizeMismatch,
    ArtifactMissing,
    AuxiliarySha256Mismatch,
    AuxiliaryFileSizeMismatch,
    AuxiliaryMissingRequired,
    RowIdentitySha256Mismatch,
    RowIdentityRowCountMismatch,
    RowIdentityMissing,
    ManifestSchema,
    ResourceLimit,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReportIssue {
    pub code: String,
    pub message: String,
    /// Auxiliary artifact name the issue refers to, when one applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_name: Option<String>,
    /// Manifest-declared SHA-256 for mismatch issues.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_sha256: Option<String>,
    /// Observed SHA-256 for mismatch issues.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_sha256: Option<String>,
    /// Manifest-declared byte size for mismatch issues.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_size_bytes: Option<u64>,
    /// Observed byte size for mismatch issues.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_size_bytes: Option<u64>,
}

impl ReportIssue {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            artifact_name: None,
            expected_sha256: None,
            actual_sha256: None,
            expected_size_bytes: None,
            actual_size_bytes: None,
        }
    }

    pub fn with_artifact_name(mut self, name: impl Into<String>) -> Self {
        self.artifact_name = Some(name.into());
        self
    }

    pub fn with_sha256_detail(
        mut self,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        self.expected_sha256 = Some(expected.into());
        self.actual_sha256 = Some(actual.into());
        self
    }

    pub fn with_size_detail(mut self, expected: u64, actual: u64) -> Self {
        self.expected_size_bytes = Some(expected);
        self.actual_size_bytes = Some(actual);
        self
    }

    /// Maps this issue's code onto the typed [`VerificationCode`] families.
    pub fn classification(&self) -> VerificationCode {
        match self.code.as_str() {
            codes::ARTIFACT_SHA256_MISMATCH => VerificationCode::ArtifactSha256Mismatch,
            codes::ARTIFACT_FILE_SIZE_MISMATCH => VerificationCode::ArtifactFileSizeMismatch,
            codes::ARTIFACT_MISSING => VerificationCode::ArtifactMissing,
            codes::AUXILIARY_ARTIFACT_SHA256_MISMATCH => VerificationCode::AuxiliarySha256Mismatch,
            codes::AUXILIARY_ARTIFACT_FILE_SIZE_MISMATCH => {
                VerificationCode::AuxiliaryFileSizeMismatch
            }
            codes::AUXILIARY_ARTIFACT_MISSING_REQUIRED => {
                VerificationCode::AuxiliaryMissingRequired
            }
            codes::ROW_IDENTITY_SHA256_MISMATCH => VerificationCode::RowIdentitySha256Mismatch,
            codes::ROW_IDENTITY_ROW_COUNT_MISMATCH => VerificationCode::RowIdentityRowCountMismatch,
            codes::ROW_IDENTITY_MISSING => VerificationCode::RowIdentityMissing,
            codes::SCHEMA_VERSION_UNSUPPORTED => VerificationCode::ManifestSchema,
            codes::MANIFEST_FILE_TOO_LARGE
            | codes::ARTIFACT_FILE_TOO_LARGE
            | codes::AUXILIARY_ARTIFACT_FILE_TOO_LARGE
            | codes::AUXILIARY_ARTIFACT_COUNT_LIMIT_EXCEEDED
            | codes::CALIBRATION_PROFILE_TOO_LARGE
            | codes::ENCODER_DISTORTION_PROFILE_TOO_LARGE
            | codes::ROW_IDENTITY_LINE_TOO_LARGE
            | codes::ROW_IDENTITY_ROW_COUNT_LIMIT_EXCEEDED
            | codes::ROW_IDENTITY_DUPLICATE_TRACKING_LIMIT_EXCEEDED
            | codes::SQLITE_CACHED_REPORT_TOO_LARGE
            | codes::VERIFICATION_REPORT_ISSUE_LIMIT_EXCEEDED => VerificationCode::ResourceLimit,
            _ => VerificationCode::Unknown,
        }
    }
}

fn push_report_issue_bounded(
    errors: &mut Vec<ReportIssue>,
    limits: &ResourceLimits,
    code: impl Into<String>,
    message: impl Into<String>,
) {
    let limit = limits.max_report_issues;
    if errors.len() < limit {
        errors.push(ReportIssue::new(code, message));
        return;
    }
    if errors
        .iter()
        .any(|issue| issue.code == codes::VERIFICATION_REPORT_ISSUE_LIMIT_EXCEEDED)
    {
        return;
    }
    let detail_limit = limit.saturating_sub(1);
    errors.truncate(detail_limit);
    errors.push(ReportIssue::new(
        codes::VERIFICATION_REPORT_ISSUE_LIMIT_EXCEEDED,
        format!("verification report issue count exceeded max_report_issues={limit}"),
    ));
}

fn enforce_report_issue_limit(errors: &mut Vec<ReportIssue>, limits: &ResourceLimits) {
    let limit = limits.max_report_issues;
    if errors.len() <= limit {
        return;
    }
    errors.retain(|issue| issue.code != codes::VERIFICATION_REPORT_ISSUE_LIMIT_EXCEEDED);
    let detail_limit = limit.saturating_sub(1);
    errors.truncate(detail_limit);
    errors.push(ReportIssue::new(
        codes::VERIFICATION_REPORT_ISSUE_LIMIT_EXCEEDED,
        format!("verification report issue count exceeded max_report_issues={limit}"),
    ));
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileHash {
    pub sha256: String,
    pub size_bytes: u64,
}

pub fn sha256_file(path: impl AsRef<Path>) -> io::Result<FileHash> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut size_bytes = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = match file.read(&mut buf) {
            Ok(n) => n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        };
        if n == 0 {
            break;
        }
        size_bytes += n as u64;
        hasher.update(&buf[..n]);
    }
    Ok(FileHash {
        sha256: hex::encode(hasher.finalize()),
        size_bytes,
    })
}

/// Hashes an in-memory byte slice with the same digest form as [`sha256_file`].
pub fn sha256_bytes(bytes: &[u8]) -> FileHash {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    FileHash {
        sha256: hex::encode(hasher.finalize()),
        size_bytes: bytes.len() as u64,
    }
}

/// Hashes a reader, refusing inputs larger than `max_bytes`.
///
/// Exceeding the bound fails with [`io::ErrorKind::InvalidData`]; inputs of
/// exactly `max_bytes` succeed.
pub fn sha256_reader<R: Read>(mut reader: R, max_bytes: u64) -> io::Result<FileHash> {
    match sha256_read_bounded(&mut reader, max_bytes)? {
        Some(hash) => Ok(hash),
        None => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("input exceeds {max_bytes} bytes"),
        )),
    }
}

/// Bounded hashing core shared by [`sha256_file_bounded`] and
/// [`sha256_reader`]. Returns `Ok(None)` when the input exceeds `max_bytes`.
fn sha256_read_bounded<R: Read>(reader: &mut R, max_bytes: u64) -> io::Result<Option<FileHash>> {
    let mut hasher = Sha256::new();
    let mut size_bytes = 0u64;
    let mut buf = [0u8; 64 * 1024];
    loop {
        // Strict bound: never request bytes past max_bytes + 1 (the +1
        // detects exceedance), mirroring read_bounded_file's take() pattern.
        let allowance = max_bytes.saturating_add(1) - size_bytes;
        if allowance == 0 {
            break;
        }
        let want = allowance.min(buf.len() as u64) as usize;
        let n = match reader.read(&mut buf[..want]) {
            Ok(n) => n,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        };
        if n == 0 {
            break;
        }
        size_bytes += n as u64;
        if size_bytes > max_bytes {
            return Ok(None);
        }
        hasher.update(&buf[..n]);
    }
    Ok(Some(FileHash {
        sha256: hex::encode(hasher.finalize()),
        size_bytes,
    }))
}

pub fn sha256_file_bounded(
    path: impl AsRef<Path>,
    max_bytes: u64,
    code: &'static str,
    context: &'static str,
) -> Result<FileHash, ManifestError> {
    let path = path.as_ref();
    // Refuse non-regular files BEFORE opening: opening a FIFO read-only
    // blocks until a writer connects, and a device node would stream
    // forever under a large declared-size bound. Regular files terminate
    // at EOF and are post-checked against the declaration. (A path swapped
    // to a special file after this check is local-actor mutation, out of
    // scope per the threat model.)
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() {
        return Err(ManifestError::limit_exceeded(
            code,
            format!("{context} is not a regular file: {}", path.display()),
        ));
    }
    let mut file = File::open(path)?;
    match sha256_read_bounded(&mut file, max_bytes)? {
        Some(hash) => Ok(hash),
        None => Err(ManifestError::limit_exceeded(
            code,
            format!(
                "{context} exceeds {max_bytes} bytes while reading {}",
                path.display()
            ),
        )),
    }
}

#[derive(Clone, Debug)]
pub enum CreateRowIdentity {
    RowIdIdentity,
    Jsonl(PathBuf),
}

#[derive(Clone, Debug)]
pub struct CreateAuxiliaryArtifact {
    pub name: String,
    pub path: PathBuf,
    pub required: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CreateManifestOptions {
    pub allow_absolute_paths: bool,
    pub allow_path_escape: bool,
    pub limits: ResourceLimits,
    pub auxiliary_artifacts: Vec<CreateAuxiliaryArtifact>,
}

pub fn create_manifest_for_index(
    index_path: impl AsRef<Path>,
    row_identity: CreateRowIdentity,
    embedding_model: impl Into<String>,
    out_path: impl AsRef<Path>,
) -> Result<IndexManifest, ManifestError> {
    create_manifest_for_index_with_options(
        index_path,
        row_identity,
        embedding_model,
        out_path,
        CreateManifestOptions::default(),
    )
}

pub fn create_manifest_for_index_with_options(
    index_path: impl AsRef<Path>,
    row_identity: CreateRowIdentity,
    embedding_model: impl Into<String>,
    out_path: impl AsRef<Path>,
    options: CreateManifestOptions,
) -> Result<IndexManifest, ManifestError> {
    let index_path = index_path.as_ref();
    let out_path = out_path.as_ref();
    let out_base = out_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !out_base.exists() {
        fs::create_dir_all(out_base)?;
    }
    let metadata = probe_index_metadata(index_path)?;
    let index_hash = sha256_file_bounded(
        index_path,
        metadata
            .file_size_bytes
            .min(options.limits.max_index_artifact_bytes),
        codes::ARTIFACT_FILE_TOO_LARGE,
        "index artifact",
    )?;
    // One consistent snapshot: the manifest records the byte count that was
    // actually hashed, and any change between the metadata probe and the
    // hash (concurrent writer) fails loudly instead of embedding a
    // size/digest pair describing different bytes.
    if index_hash.size_bytes != metadata.file_size_bytes {
        return Err(ManifestError::invalid(format!(
            "index artifact changed during manifest creation: probed {} bytes, hashed {} bytes",
            metadata.file_size_bytes, index_hash.size_bytes
        )));
    }
    let kind = ManifestIndexKind::try_from_core(metadata.kind)
        .map_err(|err| ManifestError::invalid(err.message()))?;
    let params = ManifestIndexParams::try_from_core(metadata.params)
        .map_err(|err| ManifestError::invalid(err.message()))?;
    let artifact = Artifact {
        path: manifest_path_for_create(index_path, out_base, &options, "artifact")?,
        sha256: index_hash.sha256,
        kind,
        format_version: metadata.format_version,
        dim: metadata.dim,
        vector_count: metadata.vector_count,
        bytes_per_vec: metadata.bytes_per_vec,
        params,
        file_size_bytes: index_hash.size_bytes,
    };

    let row_identity = match row_identity {
        CreateRowIdentity::RowIdIdentity => RowIdentity::RowIdIdentity {
            row_count: metadata.vector_count,
        },
        CreateRowIdentity::Jsonl(path) => {
            let mut row_errors = Vec::new();
            let stats = validate_jsonl_rows(
                &path,
                false,
                &options.limits,
                Some(metadata.vector_count),
                &mut row_errors,
            )?;
            if !row_errors.is_empty() {
                if let Some(issue) = row_errors
                    .iter()
                    .find(|issue| is_limit_issue_code(&issue.code))
                {
                    return Err(ManifestError::limit_exceeded(
                        issue.code.clone(),
                        issue.message.clone(),
                    ));
                }
                let codes = row_errors
                    .iter()
                    .map(|issue| issue.code.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(ManifestError::invalid(format!(
                    "row map is invalid: {codes}"
                )));
            }
            if stats.row_count != metadata.vector_count {
                return Err(ManifestError::invalid(format!(
                    "row map has {} rows but index has {} vectors",
                    stats.row_count, metadata.vector_count
                )));
            }
            let row_sha256 = stats.sha256.ok_or_else(|| {
                ManifestError::invalid("row map hash unavailable after bounded validation")
            })?;
            RowIdentity::Jsonl {
                path: manifest_path_for_create(&path, out_base, &options, "row identity")?,
                sha256: row_sha256,
                row_count: stats.row_count,
                id_kind: "uuid".to_string(),
                db: None,
            }
        }
    };

    let auxiliary_artifacts =
        create_auxiliary_artifacts(&options.auxiliary_artifacts, out_base, &options)?;

    Ok(IndexManifest {
        schema_version: SCHEMA_VERSION.to_string(),
        artifact,
        auxiliary_artifacts,
        embedding: Embedding {
            model: embedding_model.into(),
            dim: metadata.dim,
            model_revision: None,
            tokenizer_revision: None,
            pooling: None,
            corpus_digest: None,
            embedding_matrix_digest: None,
            normalization: None,
        },
        encoder_distortion: None,
        calibration: None,
        row_identity,
        build: None,
        attestations: Vec::new(),
        extensions: BTreeMap::new(),
    })
}

fn create_auxiliary_artifacts(
    artifacts: &[CreateAuxiliaryArtifact],
    out_base: &Path,
    options: &CreateManifestOptions,
) -> Result<Vec<AuxiliaryArtifact>, ManifestError> {
    let count = artifacts.len();
    if count > options.limits.max_auxiliary_artifacts {
        return Err(ManifestError::limit_exceeded(
            codes::AUXILIARY_ARTIFACT_COUNT_LIMIT_EXCEEDED,
            format!(
                "auxiliary_artifacts has {count} entries, exceeding max_auxiliary_artifacts={}",
                options.limits.max_auxiliary_artifacts
            ),
        ));
    }

    let mut names = HashSet::new();
    let mut manifest_artifacts = Vec::with_capacity(artifacts.len());
    for artifact in artifacts {
        let name = artifact.name.trim();
        if name.is_empty() {
            return Err(ManifestError::invalid(
                "auxiliary artifact name must be non-empty",
            ));
        }
        if !names.insert(name.to_string()) {
            return Err(ManifestError::invalid(format!(
                "auxiliary artifact name {name:?} is duplicated"
            )));
        }
        // Create is a trusted context: bound the read by the artifact's own
        // observed size (catching mid-hash growth), not a flat cap. An
        // explicitly configured flat limit still applies as a ceiling.
        let observed_len = fs::metadata(&artifact.path)
            .map_err(ManifestError::from)?
            .len();
        let hash = sha256_file_bounded(
            &artifact.path,
            observed_len.min(options.limits.max_auxiliary_artifact_bytes),
            codes::AUXILIARY_ARTIFACT_FILE_TOO_LARGE,
            "auxiliary artifact",
        )?;
        manifest_artifacts.push(AuxiliaryArtifact {
            name: name.to_string(),
            path: manifest_path_for_create(
                &artifact.path,
                out_base,
                options,
                "auxiliary artifact",
            )?,
            sha256: hash.sha256,
            file_size_bytes: hash.size_bytes,
            required: artifact.required,
        });
    }
    // Deterministic manifest bytes: entry order must not depend on
    // declaration order.
    manifest_artifacts.sort_by(|a, b| {
        (a.name.as_str(), a.path.as_str()).cmp(&(b.name.as_str(), b.path.as_str()))
    });
    Ok(manifest_artifacts)
}

/// Writes the manifest in its single canonical serialization: serde_json
/// pretty-printing, struct-declaration field order, BTreeMap-sorted map keys.
/// Content hashing and signing operate on the stored bytes, so changing the
/// serializer or its settings changes every manifest's identity and is a
/// schema-version event, not a cosmetic change.
///
/// Nested [`serde_json::Value`] maps carried in `extensions` / `attestations`
/// are recursively key-sorted and their supplied numeric values are normalized
/// before serialization. File-loading policy must validate original number
/// tokens through [`parse_current_manifest_bytes`] (or
/// [`validate_manifest_json_number_tokens`] for a compatibility schema) before
/// deserialization. A writer cannot recover a source lexeme that some earlier,
/// caller-owned parser already rounded into a `Value`.
///
/// The destination is replaced atomically from a temporary file in the same
/// directory after the bytes and portable permission bits are synced. New
/// files use the same `0o666 & !umask` Unix mode as [`File::create`]; replacing
/// an existing file preserves its [`fs::Permissions`]. Platform-specific ACLs,
/// ownership, and extended attributes are outside this portable API contract.
/// Existing symlinks and other non-regular destinations observed before the
/// replacement are rejected. The destination is checked again immediately
/// before replacement and, on Unix, an existing file's device/inode identity
/// must still match. Callers must still serialize concurrent writers: portable
/// rename is atomic replacement, not a compare-and-swap primitive. A failed
/// validation or serialization never truncates an existing manifest.
pub fn write_manifest_file(
    manifest: &IndexManifest,
    path: impl AsRef<Path>,
) -> Result<(), ManifestError> {
    let canonical = canonical_manifest_for_write(manifest)?;
    let path = path.as_ref();
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let destination = snapshot_manifest_destination(path)?;
    let inherited_permissions = destination.permissions.clone();

    // Open the directory before writing or replacing anything. On Unix this
    // makes a predictable directory-handle permission failure happen before
    // the destination changes; the same handle is synced after the rename.
    #[cfg(unix)]
    let parent_directory = File::open(parent)?;

    let mut builder = tempfile::Builder::new();
    builder.prefix(".ordvec-manifest-").suffix(".tmp");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        builder.permissions(
            inherited_permissions
                .clone()
                .unwrap_or_else(|| fs::Permissions::from_mode(0o666)),
        );
    }
    #[cfg(not(unix))]
    if let Some(permissions) = inherited_permissions.as_ref() {
        builder.permissions(permissions.clone());
    }
    let mut temporary = builder.tempfile_in(parent)?;
    serde_json::to_writer_pretty(temporary.as_file_mut(), &canonical)?;
    temporary.as_file_mut().flush()?;
    if let Some(permissions) = inherited_permissions {
        temporary.as_file().set_permissions(permissions)?;
    }
    temporary.as_file().sync_all()?;
    persist_manifest_temporary(temporary, path, &destination)?;
    #[cfg(unix)]
    parent_directory.sync_all()?;
    Ok(())
}

struct ManifestDestinationSnapshot {
    permissions: Option<fs::Permissions>,
    #[cfg(unix)]
    identity: Option<(u64, u64)>,
}

fn snapshot_manifest_destination(
    path: &Path,
) -> Result<ManifestDestinationSnapshot, ManifestError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => {
            #[cfg(unix)]
            let identity = {
                use std::os::unix::fs::MetadataExt;
                Some((metadata.dev(), metadata.ino()))
            };
            Ok(ManifestDestinationSnapshot {
                permissions: Some(metadata.permissions()),
                #[cfg(unix)]
                identity,
            })
        }
        Ok(_) => Err(ManifestError::invalid(format!(
            "manifest destination {} exists and is not a regular file",
            path.display()
        ))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(ManifestDestinationSnapshot {
            permissions: None,
            #[cfg(unix)]
            identity: None,
        }),
        Err(error) => Err(ManifestError::Io(error)),
    }
}

fn persist_manifest_temporary(
    temporary: tempfile::NamedTempFile,
    path: &Path,
    initial: &ManifestDestinationSnapshot,
) -> Result<(), ManifestError> {
    let current = snapshot_manifest_destination(path)?;
    match (
        initial.permissions.is_some(),
        current.permissions.is_some(),
    ) {
        (false, true) => {
            return Err(ManifestError::invalid(format!(
                "manifest destination {} appeared during the write; coordinate concurrent writers and retry",
                path.display()
            )))
        }
        (true, false) => {
            return Err(ManifestError::invalid(format!(
                "manifest destination {} disappeared during the write; coordinate concurrent writers and retry",
                path.display()
            )))
        }
        _ => {}
    }
    #[cfg(unix)]
    if initial.identity != current.identity {
        return Err(ManifestError::invalid(format!(
            "manifest destination {} was replaced during the write; coordinate concurrent writers and retry",
            path.display()
        )));
    }
    temporary
        .persist(path)
        .map_err(|error| ManifestError::Io(error.error))?;
    Ok(())
}

fn canonical_manifest_for_write(manifest: &IndexManifest) -> Result<IndexManifest, ManifestError> {
    validate_manifest_numbers_for_write(manifest)?;
    let mut canonical = manifest.clone();
    normalize_typed_manifest_signed_zero(&mut canonical);
    for (index, attestation) in canonical.attestations.iter_mut().enumerate() {
        canonicalize_json_value(attestation, &format!("attestations[{index}]"))?;
    }
    for (name, extension) in &mut canonical.extensions {
        canonicalize_json_value(extension, &format!("extensions[{name:?}]"))?;
    }
    Ok(canonical)
}

fn normalize_typed_manifest_signed_zero(manifest: &mut IndexManifest) {
    let Some(profile) = manifest.encoder_distortion.as_mut() else {
        return;
    };

    // Keep this list aligned with every typed f64 field reachable from
    // IndexManifest. Validation runs on the caller-owned manifest first; only
    // this cloned canonical representation is changed.
    normalize_optional_signed_zero(&mut profile.bounds.declared_lower_bound);
    normalize_optional_signed_zero(&mut profile.bounds.declared_upper_bound);
    normalize_optional_signed_zero(&mut profile.bounds.estimated_distortion);
    normalize_optional_signed_zero(&mut profile.bounds.violation_rate);
    normalize_optional_signed_zero(&mut profile.bounds.max_observed_violation);
    normalize_optional_signed_zero(&mut profile.bounds.quantile_observed_violation);
    normalize_optional_signed_zero(&mut profile.scope.confidence);
    normalize_optional_signed_zero(&mut profile.scope.coverage);
}

fn normalize_optional_signed_zero(value: &mut Option<f64>) {
    if value.is_some_and(|value| value == 0.0) {
        *value = Some(0.0);
    }
}

fn canonicalize_json_value(
    value: &mut serde_json::Value,
    context: &str,
) -> Result<(), ManifestError> {
    match value {
        serde_json::Value::Array(values) => {
            for (index, nested) in values.iter_mut().enumerate() {
                canonicalize_json_value(nested, &format!("{context}[{index}]"))?;
            }
        }
        serde_json::Value::Object(values) => {
            for (key, nested) in values.iter_mut() {
                canonicalize_json_value(nested, &format!("{context}.{key}"))?;
            }
            values.sort_keys();
        }
        serde_json::Value::Number(number) => {
            let original = number.to_string();
            let identity = decimal_identity(&original).ok_or_else(|| {
                ManifestError::invalid(format!(
                    "{context} contains a JSON number that cannot be represented canonically"
                ))
            })?;
            let normalized = if let Some(integer) = canonical_integer_from_identity(&identity) {
                integer
            } else {
                let value = number.as_f64().ok_or_else(|| {
                    ManifestError::invalid(format!(
                        "{context} contains a JSON number that cannot be represented canonically"
                    ))
                })?;
                if !value.is_finite() {
                    return Err(ManifestError::invalid(format!(
                        "{context} contains a non-finite JSON number"
                    )));
                }
                let normalized = serde_json::Number::from_f64(value).ok_or_else(|| {
                    ManifestError::invalid(format!(
                        "{context} contains a JSON number that cannot be represented canonically"
                    ))
                })?;
                if decimal_identity(&normalized.to_string()).as_ref() != Some(&identity) {
                    return Err(ManifestError::invalid(format!(
                        "{context} contains a JSON number outside the exact canonical i64/u64/f64 domain"
                    )));
                }
                normalized
            };
            *number = normalized;
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::String(_) => {}
    }
    Ok(())
}

fn canonical_integer_from_identity(identity: &(bool, String, i64)) -> Option<serde_json::Number> {
    let (negative, digits, exponent) = identity;
    if *exponent < 0 {
        return None;
    }

    // Accumulate rather than materializing exponent zeroes. Any value outside
    // u64 overflows after at most twenty decimal digits, even for a hostile
    // arbitrary-precision exponent.
    let mut magnitude = 0_u64;
    for digit in digits.bytes() {
        magnitude = magnitude
            .checked_mul(10)?
            .checked_add(u64::from(digit - b'0'))?;
    }
    let mut remaining_exponent = *exponent;
    while remaining_exponent > 0 {
        magnitude = magnitude.checked_mul(10)?;
        remaining_exponent -= 1;
    }

    if *negative {
        let i64_min_magnitude = (i64::MAX as u64) + 1;
        if magnitude == i64_min_magnitude {
            Some(serde_json::Number::from(i64::MIN))
        } else {
            let signed = i64::try_from(magnitude).ok()?;
            Some(serde_json::Number::from(-signed))
        }
    } else {
        Some(serde_json::Number::from(magnitude))
    }
}

fn decimal_identity(value: &str) -> Option<(bool, String, i64)> {
    let (negative, unsigned) = if let Some(unsigned) = value.strip_prefix('-') {
        (true, unsigned)
    } else {
        (false, value)
    };
    let mut exponent_split = unsigned.split(['e', 'E']);
    let mantissa = exponent_split.next()?;
    let explicit_exponent = exponent_split
        .next()
        .map(str::parse::<i64>)
        .transpose()
        .ok()?
        .unwrap_or(0);
    if exponent_split.next().is_some() {
        return None;
    }
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let fraction_len = i64::try_from(fraction.len()).ok()?;
    let mut exponent = explicit_exponent.checked_sub(fraction_len)?;
    let mut digits = format!("{integer}{fraction}");
    let first_nonzero = digits.bytes().position(|byte| byte != b'0');
    let Some(first_nonzero) = first_nonzero else {
        return Some((false, "0".to_string(), 0));
    };
    digits.drain(..first_nonzero);
    while digits.ends_with('0') {
        digits.pop();
        exponent = exponent.checked_add(1)?;
    }
    Some((negative, digits, exponent))
}

fn validate_manifest_numbers_for_write(manifest: &IndexManifest) -> Result<(), ManifestError> {
    let Some(profile) = manifest.encoder_distortion.as_ref() else {
        return Ok(());
    };
    let values = [
        (
            "encoder_distortion.bounds.declared_lower_bound",
            profile.bounds.declared_lower_bound,
        ),
        (
            "encoder_distortion.bounds.declared_upper_bound",
            profile.bounds.declared_upper_bound,
        ),
        (
            "encoder_distortion.bounds.estimated_distortion",
            profile.bounds.estimated_distortion,
        ),
        (
            "encoder_distortion.bounds.violation_rate",
            profile.bounds.violation_rate,
        ),
        (
            "encoder_distortion.bounds.max_observed_violation",
            profile.bounds.max_observed_violation,
        ),
        (
            "encoder_distortion.bounds.quantile_observed_violation",
            profile.bounds.quantile_observed_violation,
        ),
        (
            "encoder_distortion.scope.confidence",
            profile.scope.confidence,
        ),
        ("encoder_distortion.scope.coverage", profile.scope.coverage),
    ];
    for (name, value) in values {
        if value.is_some_and(|value| !value.is_finite()) {
            return Err(ManifestError::invalid(format!(
                "{name} must be finite before the manifest can be written"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct JsonlStats {
    row_count: usize,
    validated_rows: usize,
    sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonlRow {
    row_id: usize,
    db_id: String,
    #[serde(default)]
    parent_id: Option<String>,
}

fn validate_jsonl_rows(
    path: &Path,
    allow_duplicate_db_ids: bool,
    limits: &ResourceLimits,
    expected_row_count: Option<usize>,
    errors: &mut Vec<ReportIssue>,
) -> io::Result<JsonlStats> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut seen = HashSet::new();
    let mut seen_db_id_bytes = 0usize;
    let mut row_count = 0usize;
    let mut validated_rows = 0usize;
    let mut line = Vec::new();
    let mut reached_eof = true;

    while let Some(too_long) = read_bounded_line(
        &mut reader,
        limits.max_row_identity_jsonl_line_bytes,
        &mut line,
        &mut hasher,
    )? {
        let line_idx = row_count;
        row_count += 1;
        if row_count > limits.max_row_identity_rows {
            reached_eof = false;
            push_report_issue_bounded(
                errors,
                limits,
                codes::ROW_IDENTITY_ROW_COUNT_LIMIT_EXCEEDED,
                format!(
                    "row identity file has more than max_row_identity_rows={} rows",
                    limits.max_row_identity_rows
                ),
            );
            break;
        }
        if let Some(expected_row_count) = expected_row_count {
            if row_count > expected_row_count {
                reached_eof = false;
                push_report_issue_bounded(
                    errors,
                    limits,
                    codes::ROW_IDENTITY_ROW_COUNT_MISMATCH,
                    format!(
                        "row identity file has more than declared row_count={expected_row_count}"
                    ),
                );
                break;
            }
        }
        if too_long {
            reached_eof = false;
            push_report_issue_bounded(
                errors,
                limits,
                codes::ROW_IDENTITY_LINE_TOO_LARGE,
                format!(
                    "line {line_idx} exceeds max_row_identity_jsonl_line_bytes={}",
                    limits.max_row_identity_jsonl_line_bytes
                ),
            );
            break;
        }
        trim_jsonl_terminator(&mut line);
        let row: JsonlRow = match serde_json::from_slice(&line) {
            Ok(row) => row,
            Err(err) => {
                push_report_issue_bounded(
                    errors,
                    limits,
                    codes::ROW_IDENTITY_JSONL_INVALID_JSON,
                    format!("line {line_idx} is not a strict row object: {err}"),
                );
                continue;
            }
        };
        if row.row_id != line_idx {
            push_report_issue_bounded(
                errors,
                limits,
                codes::ROW_IDENTITY_ROW_ID_MISMATCH,
                format!("line {line_idx} has row_id {}", row.row_id),
            );
        }
        validate_row_id_string("db_id", &DB_ID_ISSUES, &row.db_id, line_idx, limits, errors);
        if let Some(parent_id) = &row.parent_id {
            validate_row_id_string(
                "parent_id",
                &PARENT_ID_ISSUES,
                parent_id,
                line_idx,
                limits,
                errors,
            );
        }
        validated_rows += 1;
        if !allow_duplicate_db_ids {
            if seen.contains(&row.db_id) {
                push_report_issue_bounded(
                    errors,
                    limits,
                    codes::ROW_IDENTITY_DUPLICATE_DB_ID,
                    format!("line {line_idx} repeats db_id"),
                );
            } else {
                let next_seen_db_id_bytes = seen_db_id_bytes.saturating_add(row.db_id.len());
                if next_seen_db_id_bytes > limits.max_row_identity_tracked_db_id_bytes {
                    reached_eof = false;
                    push_report_issue_bounded(
                        errors,
                        limits,
                        codes::ROW_IDENTITY_DUPLICATE_TRACKING_LIMIT_EXCEEDED,
                        format!(
                            "tracked db_id bytes exceed max_row_identity_tracked_db_id_bytes={}",
                            limits.max_row_identity_tracked_db_id_bytes
                        ),
                    );
                    break;
                }
                seen_db_id_bytes = next_seen_db_id_bytes;
                seen.insert(row.db_id);
            }
        }
    }

    Ok(JsonlStats {
        row_count,
        validated_rows,
        sha256: reached_eof.then(|| hex::encode(hasher.finalize())),
    })
}

fn read_bounded_line<R: BufRead>(
    reader: &mut R,
    max_bytes: usize,
    out: &mut Vec<u8>,
    hasher: &mut Sha256,
) -> io::Result<Option<bool>> {
    out.clear();
    let max_bytes = max_bytes.max(1);

    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if out.is_empty() {
                Ok(None)
            } else {
                Ok(Some(false))
            };
        }

        let newline = available.iter().position(|byte| *byte == b'\n');
        let take_len = newline.map_or(available.len(), |pos| pos + 1);

        let remaining = max_bytes.saturating_sub(out.len());
        if take_len > remaining {
            let consume_len = remaining.saturating_add(1).min(take_len);
            if remaining > 0 {
                out.extend_from_slice(&available[..remaining]);
            }
            hasher.update(&available[..consume_len]);
            reader.consume(consume_len);
            return Ok(Some(true));
        }

        out.extend_from_slice(&available[..take_len]);
        hasher.update(&available[..take_len]);
        reader.consume(take_len);
        if newline.is_some() {
            return Ok(Some(false));
        }
    }
}

fn trim_jsonl_terminator(line: &mut Vec<u8>) {
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
}

/// Per-field issue codes for [`validate_row_id_string`], so every emitted
/// code stays a named constant in [`codes`].
struct RowIdIssueCodes {
    empty: &'static str,
    contains_nul: &'static str,
    invalid_uuid: &'static str,
}

const DB_ID_ISSUES: RowIdIssueCodes = RowIdIssueCodes {
    empty: codes::ROW_IDENTITY_DB_ID_EMPTY,
    contains_nul: codes::ROW_IDENTITY_DB_ID_CONTAINS_NUL,
    invalid_uuid: codes::ROW_IDENTITY_DB_ID_INVALID_UUID,
};

const PARENT_ID_ISSUES: RowIdIssueCodes = RowIdIssueCodes {
    empty: codes::ROW_IDENTITY_PARENT_ID_EMPTY,
    contains_nul: codes::ROW_IDENTITY_PARENT_ID_CONTAINS_NUL,
    invalid_uuid: codes::ROW_IDENTITY_PARENT_ID_INVALID_UUID,
};

fn validate_row_id_string(
    field: &str,
    issue_codes: &RowIdIssueCodes,
    value: &str,
    line_idx: usize,
    limits: &ResourceLimits,
    errors: &mut Vec<ReportIssue>,
) {
    let mut structurally_invalid = false;
    if value.is_empty() {
        structurally_invalid = true;
        push_report_issue_bounded(
            errors,
            limits,
            issue_codes.empty,
            format!("line {line_idx} has empty {field}"),
        );
    }
    if value.contains('\0') {
        structurally_invalid = true;
        push_report_issue_bounded(
            errors,
            limits,
            issue_codes.contains_nul,
            format!("line {line_idx} {field} contains NUL"),
        );
    }
    if !structurally_invalid && Uuid::parse_str(value).is_err() {
        push_report_issue_bounded(
            errors,
            limits,
            issue_codes.invalid_uuid,
            format!("line {line_idx} {field} must be a UUID in v1"),
        );
    }
}

fn is_limit_issue_code(code: &str) -> bool {
    matches!(
        code,
        codes::ROW_IDENTITY_LINE_TOO_LARGE
            | codes::ROW_IDENTITY_ROW_COUNT_LIMIT_EXCEEDED
            | codes::ROW_IDENTITY_DUPLICATE_TRACKING_LIMIT_EXCEEDED
            | codes::CALIBRATION_PROFILE_TOO_LARGE
            | codes::ENCODER_DISTORTION_PROFILE_TOO_LARGE
            | codes::VERIFICATION_REPORT_ISSUE_LIMIT_EXCEEDED
    )
}

fn manifest_path_for_create(
    path: &Path,
    base_dir: &Path,
    options: &CreateManifestOptions,
    context: &str,
) -> Result<String, ManifestError> {
    let canonical_path = fs::canonicalize(path)?;
    let canonical_base = fs::canonicalize(base_dir)?;
    let value = if let Ok(relative) = canonical_path.strip_prefix(&canonical_base) {
        if relative.as_os_str().is_empty() {
            ".".to_string()
        } else {
            path_to_manifest_string(relative)?
        }
    } else if !options.allow_path_escape {
        return Err(ManifestError::invalid(format!(
            "{context} path {} is outside manifest directory {}; use --allow-path-escape to create a manifest that requires non-default verification policy",
            canonical_path.display(),
            canonical_base.display()
        )));
    } else if let Some(relative) = relative_path_between(&canonical_base, &canonical_path) {
        path_to_manifest_string(&relative)?
    } else if options.allow_absolute_paths {
        path_to_manifest_string(&canonical_path)?
    } else {
        return Err(ManifestError::invalid(format!(
            "{context} path {} cannot be expressed relative to manifest directory {}; use --allow-absolute-paths with --allow-path-escape",
            canonical_path.display(),
            canonical_base.display()
        )));
    };

    // Never embed a path the manifest's own validation would reject: what
    // create writes, verify (under the same policy flags) must accept.
    if !is_manifest_path_absolute(&value)
        && !is_canonical_manifest_path(&value, options.allow_path_escape)
    {
        return Err(ManifestError::invalid(format!(
            "{context} path {value:?} cannot be embedded canonically (bundle-relative, forward slashes, no `.`, `..`, or empty segments); rename the file or move it into the manifest directory"
        )));
    }
    Ok(value)
}

fn relative_path_between(base: &Path, target: &Path) -> Option<PathBuf> {
    let base_components = base.components().collect::<Vec<_>>();
    let target_components = target.components().collect::<Vec<_>>();
    let mut common = 0usize;
    while common < base_components.len()
        && common < target_components.len()
        && base_components[common] == target_components[common]
    {
        common += 1;
    }

    if common == 0 {
        return None;
    }

    let mut relative = PathBuf::new();
    for component in &base_components[common..] {
        if matches!(component, Component::Normal(_)) {
            relative.push("..");
        }
    }
    for component in &target_components[common..] {
        match component {
            Component::Normal(part) => relative.push(part),
            Component::CurDir => {}
            Component::ParentDir => relative.push(".."),
            Component::Prefix(_) | Component::RootDir => return None,
        }
    }
    Some(relative)
}

/// A canonical manifest path is bundle-relative, uses forward slashes only,
/// and contains no `.`, `..`, or empty segments, so identical bundle content
/// always embeds identical path strings. `..` segments are only accepted
/// under `allow_path_escape`; absolute paths are excluded before this check
/// and remain governed by the `allow_absolute_paths` policy at resolution.
fn is_canonical_manifest_path(path: &str, allow_path_escape: bool) -> bool {
    if path.is_empty() || path.contains('\\') {
        return false;
    }
    path.split('/').all(|segment| {
        !segment.is_empty() && segment != "." && (segment != ".." || allow_path_escape)
    })
}

/// Detects absolute manifest path strings on any platform: POSIX (`/...`),
/// Windows drive (`C:/...` or `C:\...`), and UNC/verbatim (`\\...`, `//...`).
/// A single leading backslash is NOT absolute: on Unix it is an ordinary
/// file-name byte, so treating it as absolute here while resolution treats
/// it as relative would let `\evil` skip the canonical-form check and still
/// resolve inside the bundle. It stays non-absolute and is rejected by the
/// canonical-form check instead (backslashes are never canonical). The
/// canonical-form check skips absolute paths; the `allow_absolute_paths`
/// policy at path resolution accepts or rejects them.
fn is_manifest_path_absolute(path: &str) -> bool {
    if path.starts_with('/') || path.starts_with(r"\\") {
        return true;
    }
    let bytes = path.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'/' || bytes[2] == b'\\')
}

fn path_to_manifest_string(path: &Path) -> Result<String, ManifestError> {
    if path.is_absolute() {
        let display = path.to_str().ok_or_else(|| {
            ManifestError::invalid(format!(
                "path {:?} is not valid UTF-8 and cannot be embedded in a manifest",
                path.as_os_str()
            ))
        })?;
        // fs::canonicalize returns verbatim paths on Windows; strip the
        // verbatim prefix so the embedded string round-trips through
        // PathBuf::from at verification time.
        let display = if let Some(rest) = display.strip_prefix(r"\\?\UNC\") {
            format!(r"\\{rest}")
        } else if let Some(rest) = display.strip_prefix(r"\\?\") {
            rest.to_string()
        } else {
            display.to_string()
        };
        return Ok(display.replace('\\', "/"));
    }
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => parts.push(
                part.to_str()
                    .ok_or_else(|| {
                        ManifestError::invalid(format!(
                            "path {:?} is not valid UTF-8 and cannot be embedded in a manifest",
                            path.as_os_str()
                        ))
                    })?
                    .to_string(),
            ),
            Component::CurDir => parts.push(".".to_string()),
            Component::ParentDir => parts.push("..".to_string()),
            Component::Prefix(_) | Component::RootDir => {}
        }
    }
    if parts.is_empty() {
        Ok(".".to_string())
    } else {
        Ok(parts.join("/"))
    }
}

fn path_to_display(path: &Path) -> String {
    path.display().to_string()
}

fn extension_key_is_namespaced(key: &str) -> bool {
    if key.contains("://") || key.starts_with("urn:") {
        return true;
    }
    let mut parts = key.split('.');
    let Some(first) = parts.next() else {
        return false;
    };
    if !valid_extension_part(first) {
        return false;
    }
    let mut saw_second = false;
    for part in parts {
        saw_second = true;
        if !valid_extension_part(part) {
            return false;
        }
    }
    saw_second
}

fn valid_extension_part(part: &str) -> bool {
    !part.is_empty()
        && part
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        && part.bytes().any(|b| b.is_ascii_alphanumeric())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
}

fn hex_digest_eq(a: &str, b: &str) -> bool {
    a == b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[test]
    fn unix_nofollow_symlink_errnos_have_typed_classification() {
        assert!(is_final_symlink_open_error(&io::Error::from_raw_os_error(
            libc::ELOOP,
        )));
        assert!(!is_final_symlink_open_error(&io::Error::from_raw_os_error(
            libc::EACCES
        ),));

        #[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
        assert!(is_final_symlink_open_error(&io::Error::from_raw_os_error(
            libc::EMLINK,
        )));
        #[cfg(target_os = "netbsd")]
        assert!(is_final_symlink_open_error(&io::Error::from_raw_os_error(
            libc::EFTYPE,
        )));
    }

    #[test]
    fn digesting_bounded_reader_retries_interrupted_reads() {
        struct InterruptOnce<R> {
            inner: R,
            interrupted: bool,
        }

        impl<R: Read> Read for InterruptOnce<R> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                if !self.interrupted {
                    self.interrupted = true;
                    return Err(io::Error::from(io::ErrorKind::Interrupted));
                }
                self.inner.read(buffer)
            }
        }

        let bytes = b"plan-verified bytes survive an interrupted read";
        let mut source = InterruptOnce {
            inner: io::Cursor::new(bytes),
            interrupted: false,
        };
        let mut reader = DigestingBoundedReader::new(&mut source, bytes.len() as u64);
        let mut delivered = Vec::new();
        reader.read_to_end(&mut delivered).unwrap();
        assert_eq!(reader.remaining, 0);
        assert_eq!(reader.consumed, bytes.len() as u64);
        let (digest, read_error) = reader.finish();

        assert!(source.interrupted);
        assert_eq!(delivered, bytes);
        assert!(read_error.is_none());
        assert_eq!(digest, hex::encode(Sha256::digest(bytes)));
    }

    #[test]
    fn canonical_manifest_path_form_is_policy_aware() {
        assert!(is_canonical_manifest_path("index.ovrq", false));
        assert!(is_canonical_manifest_path("sub/dir/index.ovrq", false));
        assert!(!is_canonical_manifest_path("", false));
        assert!(!is_canonical_manifest_path("./index.ovrq", false));
        assert!(!is_canonical_manifest_path("a//b", false));
        assert!(!is_canonical_manifest_path("back\\slash.bin", false));
        assert!(!is_canonical_manifest_path("a/../index.ovrq", false));
        assert!(!is_canonical_manifest_path("../index.ovrq", false));
        assert!(is_canonical_manifest_path("a/../index.ovrq", true));
        assert!(is_canonical_manifest_path("../index.ovrq", true));
        assert!(!is_canonical_manifest_path("back\\slash.bin", true));
    }

    #[test]
    fn absolute_manifest_path_detection_covers_all_platform_forms() {
        assert!(is_manifest_path_absolute("/srv/index.ovrq"));
        assert!(is_manifest_path_absolute("C:/bundles/index.ovrq"));
        assert!(is_manifest_path_absolute("C:\\bundles\\index.ovrq"));
        assert!(is_manifest_path_absolute("\\\\server\\share\\index.ovrq"));
        assert!(is_manifest_path_absolute("//?/C:/bundles/index.ovrq"));
        assert!(is_manifest_path_absolute("\\\\?\\C:\\bundles\\index.ovrq"));
        assert!(!is_manifest_path_absolute("index.ovrq"));
        assert!(!is_manifest_path_absolute("sub/index.ovrq"));
        assert!(!is_manifest_path_absolute("C:"));
        // A single leading backslash is not absolute: it must fall through to
        // the canonical-form check (which rejects backslashes) instead of
        // skipping it and then resolving relative on Unix.
        assert!(!is_manifest_path_absolute("\\evil"));
        assert!(!is_manifest_path_absolute("\\"));
    }

    #[cfg(unix)]
    #[test]
    fn pre_persist_validation_rejects_a_destination_swapped_to_a_symlink() {
        use std::io::Write as _;
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("manifest.json");
        let target = directory.path().join("target.json");
        fs::write(&destination, b"original").unwrap();
        fs::write(&target, b"target").unwrap();
        let initial = snapshot_manifest_destination(&destination).unwrap();
        let mut temporary = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        temporary.write_all(b"replacement").unwrap();

        fs::remove_file(&destination).unwrap();
        symlink(&target, &destination).unwrap();
        let error = persist_manifest_temporary(temporary, &destination, &initial).unwrap_err();

        assert!(error.to_string().contains("not a regular file"), "{error}");
        assert!(fs::symlink_metadata(&destination)
            .unwrap()
            .file_type()
            .is_symlink());
        assert_eq!(fs::read(target).unwrap(), b"target");
    }

    #[cfg(unix)]
    #[test]
    fn pre_persist_validation_rejects_a_replaced_regular_inode() {
        use std::io::Write as _;

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("manifest.json");
        let held_original = directory.path().join("original-manifest-inode");
        fs::write(&destination, b"original").unwrap();
        // Keep the original inode live so the filesystem cannot recycle its
        // identity for the concurrent writer after `destination` is unlinked.
        fs::hard_link(&destination, &held_original).unwrap();
        let initial = snapshot_manifest_destination(&destination).unwrap();
        let mut temporary = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        temporary.write_all(b"replacement").unwrap();

        fs::remove_file(&destination).unwrap();
        fs::write(&destination, b"concurrent writer").unwrap();
        let error = persist_manifest_temporary(temporary, &destination, &initial).unwrap_err();

        assert!(
            error.to_string().contains("was replaced during the write"),
            "{error}"
        );
        assert_eq!(fs::read(destination).unwrap(), b"concurrent writer");
    }

    #[test]
    fn manifest_kind_conversion_uses_format_registry_coverage() {
        for spec in FORMATS {
            match spec.manifest {
                ManifestCoverage::Covered => {
                    ManifestIndexKind::try_from_core(spec.kind)
                        .expect("manifest-covered registry rows must map to manifest kinds");
                }
                ManifestCoverage::NotCovered {
                    tracking_issue,
                    reason,
                } => {
                    let err = ManifestIndexKind::try_from_core(spec.kind).unwrap_err();
                    assert_eq!(err.code(), "artifact_manifest_coverage_unsupported");
                    assert!(err.message().contains(reason));
                    assert!(tracking_issue > 0);
                }
                _ => panic!("unsupported manifest coverage stance in registry test"),
            }
        }
        assert!(matches!(
            ManifestIndexKind::try_from_core(CoreIndexKind::RankQuantFastscan)
                .unwrap_err()
                .code(),
            "artifact_manifest_coverage_unsupported"
        ));
    }
}

#[cfg(feature = "sqlite")]
pub mod sqlite;
