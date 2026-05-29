use chrono::{DateTime, SecondsFormat, Utc};
use ordvec::{
    probe_index_metadata, IndexKind as CoreIndexKind, IndexMetadata as CoreIndexMetadata,
    IndexParams as CoreIndexParams,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Component, Path, PathBuf};
use uuid::Uuid;

pub const SCHEMA_VERSION: &str = "ordvec.index_manifest.v1";

#[derive(Debug)]
pub enum ManifestError {
    Io(io::Error),
    Json(serde_json::Error),
    Invalid(String),
}

impl ManifestError {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Json(err) => write!(f, "{err}"),
            Self::Invalid(message) => f.write_str(message),
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
    let path = path.as_ref();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let manifest: IndexManifest = serde_json::from_reader(reader)?;
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

pub fn verify_manifest(document: &ManifestDocument, options: VerifyOptions) -> VerificationReport {
    let mut report = VerificationReport::new(Some(document.manifest.manifest_id.clone()));
    validate_manifest_shape(&document.manifest, &mut report);

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
        "artifact",
        &mut report.errors,
    ) {
        report.artifact.canonical_path = Some(path_to_display(&resolved.canonical_path));
        match sha256_file(&resolved.resolved_path) {
            Ok(hash) => {
                report.artifact.sha256 = Some(hash.sha256.clone());
                report.artifact.size_bytes = Some(hash.size_bytes);
                if !hex_digest_eq(&hash.sha256, &document.manifest.artifact.sha256) {
                    report.error(
                        "artifact_sha256_mismatch",
                        format!(
                            "artifact SHA-256 was {}, manifest declares {}",
                            hash.sha256, document.manifest.artifact.sha256
                        ),
                    );
                }
                if hash.size_bytes != document.manifest.artifact.file_size_bytes {
                    report.error(
                        "artifact_file_size_mismatch",
                        format!(
                            "artifact size was {}, manifest declares {}",
                            hash.size_bytes, document.manifest.artifact.file_size_bytes
                        ),
                    );
                }
            }
            Err(err) => report.error(
                "artifact_hash_failed",
                format!("failed to hash artifact: {err}"),
            ),
        }

        match probe_index_metadata(&resolved.resolved_path) {
            Ok(metadata) => {
                let metadata_report = MetadataReport::from_core(&metadata);
                compare_artifact_metadata(&document.manifest.artifact, &metadata, &mut report);
                report.artifact.metadata = Some(metadata_report);
            }
            Err(err) => report.error(
                "artifact_probe_failed",
                format!("failed to probe artifact metadata: {err}"),
            ),
        }
    }

    verify_row_identity(document, &options, &mut report);
    verify_attestations(&document.manifest, &mut report);

    report.ok = report.errors.is_empty();
    report
}

fn validate_manifest_shape(manifest: &IndexManifest, report: &mut VerificationReport) {
    if manifest.schema_version != SCHEMA_VERSION {
        report.error(
            "schema_version_unsupported",
            format!(
                "schema_version must be {SCHEMA_VERSION}, got {}",
                manifest.schema_version
            ),
        );
    }
    if manifest.manifest_id.trim().is_empty() {
        report.error("manifest_id_empty", "manifest_id must be non-empty");
    }
    if DateTime::parse_from_rfc3339(&manifest.created_at).is_err() {
        report.error("created_at_invalid", "created_at must parse as RFC3339");
    }
    if manifest.embedding.model.trim().is_empty() {
        report.error("embedding_model_empty", "embedding.model must be non-empty");
    }
    if manifest.embedding.dim == 0 {
        report.error(
            "embedding_dim_zero",
            "embedding.dim must be greater than zero",
        );
    }
    if manifest.artifact.path.trim().is_empty() {
        report.error("artifact_path_empty", "artifact.path must be non-empty");
    }
    if !is_sha256_hex(&manifest.artifact.sha256) {
        report.error(
            "artifact_sha256_invalid",
            "artifact.sha256 must be a 64-character hex SHA-256 digest",
        );
    }
    if manifest.artifact.bytes_per_vec == 0 {
        report.error(
            "artifact_bytes_per_vec_zero",
            "artifact.bytes_per_vec must be greater than zero",
        );
    }
    if manifest.artifact.dim != manifest.embedding.dim {
        report.error(
            "artifact_embedding_dim_mismatch",
            format!(
                "artifact.dim {} does not match embedding.dim {}",
                manifest.artifact.dim, manifest.embedding.dim
            ),
        );
    }
    if !artifact_kind_matches_params(manifest.artifact.kind, &manifest.artifact.params) {
        report.error(
            "artifact_params_kind_mismatch",
            "artifact.params discriminator does not match artifact.kind",
        );
    }

    let row_count = manifest.row_identity.row_count();
    if manifest.artifact.vector_count != row_count {
        report.error(
            "artifact_row_count_mismatch",
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
        ..
    } = &manifest.row_identity
    {
        if path.trim().is_empty() {
            report.error(
                "row_identity_path_empty",
                "row_identity.path must be non-empty",
            );
        }
        if !is_sha256_hex(sha256) {
            report.error(
                "row_identity_sha256_invalid",
                "row_identity.sha256 must be a 64-character hex SHA-256 digest",
            );
        }
        if id_kind != "uuid" {
            report.error(
                "row_identity_id_kind_unsupported",
                "row_identity.id_kind must be uuid in v1",
            );
        }
    }

    if let Some(build) = &manifest.build {
        if build.invocation_id.trim().is_empty() {
            report.error(
                "build_invocation_id_empty",
                "build.invocation_id must be non-empty",
            );
        }
        if build
            .builder_id
            .as_ref()
            .is_some_and(|builder_id| builder_id.trim().is_empty())
        {
            report.error(
                "build_builder_id_empty",
                "build.builder_id must be non-empty",
            );
        }
    }

    for key in manifest.extensions.keys() {
        if !extension_key_is_namespaced(key) {
            report.error(
                "extension_key_not_namespaced",
                format!("extension key {key:?} must be namespaced"),
            );
        }
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
    let observed_kind = ManifestIndexKind::from_core(metadata.kind);
    if artifact.kind != observed_kind {
        report.error(
            "artifact_kind_mismatch",
            format!(
                "artifact kind was {:?}, manifest declares {:?}",
                observed_kind, artifact.kind
            ),
        );
    }
    let observed_params = ManifestIndexParams::from_core(metadata.params);
    if artifact.params != observed_params {
        report.error(
            "artifact_params_mismatch",
            format!(
                "artifact params were {:?}, manifest declares {:?}",
                observed_params, artifact.params
            ),
        );
    }
    if artifact.format_version != metadata.format_version {
        report.error(
            "artifact_format_version_mismatch",
            format!(
                "artifact format_version was {}, manifest declares {}",
                metadata.format_version, artifact.format_version
            ),
        );
    }
    if artifact.dim != metadata.dim {
        report.error(
            "artifact_dim_mismatch",
            format!(
                "artifact dim was {}, manifest declares {}",
                metadata.dim, artifact.dim
            ),
        );
    }
    if artifact.vector_count != metadata.vector_count {
        report.error(
            "artifact_vector_count_mismatch",
            format!(
                "artifact vector_count was {}, manifest declares {}",
                metadata.vector_count, artifact.vector_count
            ),
        );
    }
    if artifact.bytes_per_vec != metadata.bytes_per_vec {
        report.error(
            "artifact_bytes_per_vec_mismatch",
            format!(
                "artifact bytes_per_vec was {}, manifest declares {}",
                metadata.bytes_per_vec, artifact.bytes_per_vec
            ),
        );
    }
    if artifact.file_size_bytes != metadata.file_size_bytes {
        report.error(
            "artifact_metadata_file_size_mismatch",
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
            let row_path = PathBuf::from(path);
            if let Some(resolved) = resolve_existing_path(
                &row_path,
                &document.base_dir,
                options,
                "row_identity",
                &mut report.errors,
            ) {
                report.row_identity.canonical_path =
                    Some(path_to_display(&resolved.canonical_path));
                match sha256_file(&resolved.resolved_path) {
                    Ok(hash) => {
                        report.row_identity.sha256 = Some(hash.sha256.clone());
                        if !hex_digest_eq(&hash.sha256, sha256) {
                            report.error(
                                "row_identity_sha256_mismatch",
                                format!(
                                    "row_identity SHA-256 was {}, manifest declares {}",
                                    hash.sha256, sha256
                                ),
                            );
                        }
                    }
                    Err(err) => report.error(
                        "row_identity_hash_failed",
                        format!("failed to hash row identity file: {err}"),
                    ),
                }

                match validate_jsonl_rows(
                    &resolved.resolved_path,
                    options.allow_duplicate_db_ids,
                    &mut report.errors,
                ) {
                    Ok(stats) => {
                        report.row_identity.validated_rows = Some(stats.row_count);
                        if stats.row_count != *row_count {
                            report.error(
                                "row_identity_row_count_mismatch",
                                format!(
                                    "row identity file has {} rows, manifest declares {}",
                                    stats.row_count, row_count
                                ),
                            );
                        }
                    }
                    Err(err) => report.error(
                        "row_identity_read_failed",
                        format!("failed to read row identity file: {err}"),
                    ),
                }
            }
        }
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
                "attestation_predicate_type_missing",
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
            "attestation_subject_sha256_mismatch",
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
}

#[derive(Clone, Debug)]
struct ResolvedPath {
    resolved_path: PathBuf,
    canonical_path: PathBuf,
}

fn resolve_existing_path(
    path: &Path,
    base_dir: &Path,
    options: &VerifyOptions,
    context: &str,
    errors: &mut Vec<ReportIssue>,
) -> Option<ResolvedPath> {
    if path.is_absolute() && !options.allow_absolute_paths {
        errors.push(ReportIssue::new(
            format!("{context}_absolute_path_rejected"),
            format!("absolute path {} is rejected by default", path.display()),
        ));
        return None;
    }

    let base_canonical = match fs::canonicalize(base_dir) {
        Ok(path) => path,
        Err(err) => {
            errors.push(ReportIssue::new(
                format!("{context}_base_dir_unavailable"),
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
            format!("{context}_path_escape_rejected"),
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
            errors.push(ReportIssue::new(
                format!("{context}_path_unavailable"),
                format!("failed to canonicalize {}: {err}", resolved_path.display()),
            ));
            return None;
        }
    };

    if !options.allow_path_escape && !canonical_path.starts_with(&base_canonical) {
        errors.push(ReportIssue::new(
            format!("{context}_path_escape_rejected"),
            format!(
                "canonical path {} is outside manifest base {}",
                canonical_path.display(),
                base_canonical.display()
            ),
        ));
        return None;
    }

    Some(ResolvedPath {
        resolved_path,
        canonical_path,
    })
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

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexManifest {
    pub schema_version: String,
    pub manifest_id: String,
    pub created_at: String,
    pub artifact: Artifact,
    pub embedding: Embedding,
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
pub struct Embedding {
    pub model: String,
    pub dim: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildInfo {
    pub invocation_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builder_id: Option<String>,
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
pub enum ManifestIndexKind {
    Rank,
    RankQuant,
    Bitmap,
    SignBitmap,
}

impl ManifestIndexKind {
    fn from_core(kind: CoreIndexKind) -> Self {
        match kind {
            CoreIndexKind::Rank => Self::Rank,
            CoreIndexKind::RankQuant => Self::RankQuant,
            CoreIndexKind::Bitmap => Self::Bitmap,
            CoreIndexKind::SignBitmap => Self::SignBitmap,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ManifestIndexParams {
    Rank,
    RankQuant { bits: u8 },
    Bitmap { n_top: usize },
    SignBitmap,
}

impl ManifestIndexParams {
    fn from_core(params: CoreIndexParams) -> Self {
        match params {
            CoreIndexParams::Rank => Self::Rank,
            CoreIndexParams::RankQuant { bits } => Self::RankQuant { bits },
            CoreIndexParams::Bitmap { n_top } => Self::Bitmap { n_top },
            CoreIndexParams::SignBitmap => Self::SignBitmap,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationReport {
    pub ok: bool,
    pub checked_at: String,
    pub manifest_id: Option<String>,
    pub artifact: ArtifactReport,
    pub row_identity: RowIdentityReport,
    pub attestation_shape_checks: Vec<AttestationShapeCheck>,
    pub errors: Vec<ReportIssue>,
    pub warnings: Vec<ReportIssue>,
    pub skipped_checks: Vec<String>,
}

impl VerificationReport {
    fn new(manifest_id: Option<String>) -> Self {
        Self {
            ok: false,
            checked_at: Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true),
            manifest_id,
            artifact: ArtifactReport::default(),
            row_identity: RowIdentityReport::default(),
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RowIdentityReport {
    pub kind: Option<String>,
    pub manifest_path: Option<String>,
    pub canonical_path: Option<String>,
    pub sha256: Option<String>,
    pub row_count: Option<usize>,
    pub validated_rows: Option<usize>,
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
    fn from_core(metadata: &CoreIndexMetadata) -> Self {
        Self {
            kind: ManifestIndexKind::from_core(metadata.kind),
            format_version: metadata.format_version,
            dim: metadata.dim,
            vector_count: metadata.vector_count,
            bytes_per_vec: metadata.bytes_per_vec,
            params: ManifestIndexParams::from_core(metadata.params),
            file_size_bytes: metadata.file_size_bytes,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttestationShapeCheck {
    pub predicate_type: Option<String>,
    pub builder_id: Option<String>,
    pub subject_sha256_matched: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReportIssue {
    pub code: String,
    pub message: String,
}

impl ReportIssue {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
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
        let n = file.read(&mut buf)?;
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

#[derive(Clone, Debug)]
pub enum CreateRowIdentity {
    RowIdIdentity,
    Jsonl(PathBuf),
}

pub fn create_manifest_for_index(
    index_path: impl AsRef<Path>,
    row_identity: CreateRowIdentity,
    embedding_model: impl Into<String>,
    out_path: impl AsRef<Path>,
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
    let index_hash = sha256_file(index_path)?;
    let artifact = Artifact {
        path: manifest_relative_path(index_path, out_base),
        sha256: index_hash.sha256,
        kind: ManifestIndexKind::from_core(metadata.kind),
        format_version: metadata.format_version,
        dim: metadata.dim,
        vector_count: metadata.vector_count,
        bytes_per_vec: metadata.bytes_per_vec,
        params: ManifestIndexParams::from_core(metadata.params),
        file_size_bytes: metadata.file_size_bytes,
    };

    let row_identity = match row_identity {
        CreateRowIdentity::RowIdIdentity => RowIdentity::RowIdIdentity {
            row_count: metadata.vector_count,
        },
        CreateRowIdentity::Jsonl(path) => {
            let row_hash = sha256_file(&path)?;
            let mut row_errors = Vec::new();
            let stats = validate_jsonl_rows(&path, false, &mut row_errors)?;
            if !row_errors.is_empty() {
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
            RowIdentity::Jsonl {
                path: manifest_relative_path(&path, out_base),
                sha256: row_hash.sha256,
                row_count: stats.row_count,
                id_kind: "uuid".to_string(),
                db: None,
            }
        }
    };

    let invocation_id = format!("urn:uuid:{}", Uuid::new_v4());
    Ok(IndexManifest {
        schema_version: SCHEMA_VERSION.to_string(),
        manifest_id: format!("urn:uuid:{}", Uuid::new_v4()),
        created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        artifact,
        embedding: Embedding {
            model: embedding_model.into(),
            dim: metadata.dim,
        },
        row_identity,
        build: Some(BuildInfo {
            invocation_id,
            builder_id: Some("ordvec-manifest".to_string()),
        }),
        attestations: Vec::new(),
        extensions: BTreeMap::new(),
    })
}

pub fn write_manifest_file(
    manifest: &IndexManifest,
    path: impl AsRef<Path>,
) -> Result<(), ManifestError> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, manifest)?;
    Ok(())
}

#[derive(Clone, Debug)]
struct JsonlStats {
    row_count: usize,
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
    errors: &mut Vec<ReportIssue>,
) -> io::Result<JsonlStats> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut seen = HashSet::new();
    let mut row_count = 0usize;

    for (line_idx, line) in reader.lines().enumerate() {
        let line = line?;
        row_count += 1;
        let row: JsonlRow = match serde_json::from_str(&line) {
            Ok(row) => row,
            Err(err) => {
                errors.push(ReportIssue::new(
                    "row_identity_jsonl_invalid_json",
                    format!("line {line_idx} is not a strict row object: {err}"),
                ));
                continue;
            }
        };
        if row.row_id != line_idx {
            errors.push(ReportIssue::new(
                "row_identity_row_id_mismatch",
                format!("line {line_idx} has row_id {}", row.row_id),
            ));
        }
        validate_row_id_string("db_id", &row.db_id, line_idx, errors);
        if let Some(parent_id) = &row.parent_id {
            validate_row_id_string("parent_id", parent_id, line_idx, errors);
        }
        if !allow_duplicate_db_ids && !seen.insert(row.db_id) {
            errors.push(ReportIssue::new(
                "row_identity_duplicate_db_id",
                format!("line {line_idx} repeats db_id"),
            ));
        }
    }

    Ok(JsonlStats { row_count })
}

fn validate_row_id_string(
    field: &str,
    value: &str,
    line_idx: usize,
    errors: &mut Vec<ReportIssue>,
) {
    if value.is_empty() {
        errors.push(ReportIssue::new(
            format!("row_identity_{field}_empty"),
            format!("line {line_idx} has empty {field}"),
        ));
    }
    if value.contains('\0') {
        errors.push(ReportIssue::new(
            format!("row_identity_{field}_contains_nul"),
            format!("line {line_idx} {field} contains NUL"),
        ));
    }
}

fn manifest_relative_path(path: &Path, base_dir: &Path) -> String {
    let canonical_path = fs::canonicalize(path);
    let canonical_base = fs::canonicalize(base_dir);
    if let (Ok(canonical_path), Ok(canonical_base)) = (canonical_path, canonical_base) {
        if let Ok(relative) = canonical_path.strip_prefix(&canonical_base) {
            if !relative.as_os_str().is_empty() {
                return path_to_manifest_string(relative);
            }
        }
    }
    path_to_manifest_string(path)
}

fn path_to_manifest_string(path: &Path) -> String {
    if path.is_absolute() {
        return path.display().to_string().replace('\\', "/");
    }
    let parts = path
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            Component::CurDir => Some(".".to_string()),
            Component::ParentDir => Some("..".to_string()),
            Component::Prefix(_) | Component::RootDir => None,
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.join("/")
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
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

fn hex_digest_eq(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

#[cfg(feature = "sqlite")]
pub mod sqlite;
