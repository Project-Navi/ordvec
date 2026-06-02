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
pub const CALIBRATION_SCHEMA_VERSION: &str = "ordvec.calibration.v1";
pub const DEFAULT_MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
pub const DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES: usize = 64 * 1024;
pub const DEFAULT_MAX_ROW_IDENTITY_ROWS: usize = 10_000_000;
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
    let manifest_bytes = read_bounded_file(
        path,
        options.limits.max_manifest_bytes,
        "manifest_file_too_large",
        "manifest file",
    )?;
    let manifest: IndexManifest = serde_json::from_slice(&manifest_bytes)?;
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

fn read_bounded_file(
    path: &Path,
    max_bytes: u64,
    code: &'static str,
    context: &'static str,
) -> Result<Vec<u8>, ManifestError> {
    let mut file = File::open(path)?;
    let max_len = usize::try_from(max_bytes).unwrap_or(usize::MAX);
    let mut bytes = Vec::new();
    let mut limited = file.by_ref().take(max_bytes.saturating_add(1));
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
    verify_calibration(document, &options, &mut report);
    verify_attestations(&document.manifest, &mut report);

    enforce_report_issue_limit(&mut report.errors, &options.limits);
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
            "artifact.sha256 must be a lowercase 64-character hex SHA-256 digest",
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
                "row_identity.sha256 must be a lowercase 64-character hex SHA-256 digest",
            );
        }
        if id_kind != "uuid" {
            report.error(
                "row_identity_id_kind_unsupported",
                "row_identity.id_kind must be uuid in v1",
            );
        }
    }

    validate_optional_non_empty(
        "embedding_model_revision_empty",
        "embedding.model_revision must be non-empty when present",
        manifest.embedding.model_revision.as_deref(),
        report,
    );
    validate_optional_sha256(
        "embedding_corpus_digest_invalid",
        "embedding.corpus_digest must be a lowercase 64-character hex SHA-256 digest",
        manifest.embedding.corpus_digest.as_deref(),
        report,
    );
    validate_optional_sha256(
        "embedding_matrix_digest_invalid",
        "embedding.embedding_matrix_digest must be a lowercase 64-character hex SHA-256 digest",
        manifest.embedding.embedding_matrix_digest.as_deref(),
        report,
    );
    validate_optional_non_empty(
        "embedding_normalization_empty",
        "embedding.normalization must be non-empty when present",
        manifest.embedding.normalization.as_deref(),
        report,
    );

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
        validate_optional_non_empty(
            "build_source_repo_empty",
            "build.source_repo must be non-empty when present",
            build.source_repo.as_deref(),
            report,
        );
        validate_optional_non_empty(
            "build_source_commit_empty",
            "build.source_commit must be non-empty when present",
            build.source_commit.as_deref(),
            report,
        );
        validate_optional_non_empty(
            "build_ci_provider_empty",
            "build.ci_provider must be non-empty when present",
            build.ci_provider.as_deref(),
            report,
        );
        validate_optional_non_empty(
            "build_ci_run_id_empty",
            "build.ci_run_id must be non-empty when present",
            build.ci_run_id.as_deref(),
            report,
        );
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
            if *row_count > options.limits.max_row_identity_rows {
                report.error(
                    "row_identity_row_count_limit_exceeded",
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
                "row_identity",
                &mut report.errors,
            ) {
                report.row_identity.canonical_path =
                    Some(path_to_display(&resolved.canonical_path));
                match validate_jsonl_rows(
                    &resolved.resolved_path,
                    options.allow_duplicate_db_ids,
                    &options.limits,
                    Some(*row_count),
                    &mut report.errors,
                ) {
                    Ok(stats) => {
                        report.row_identity.validated_rows = Some(stats.row_count);
                        if let Some(hash) = stats.sha256 {
                            report.row_identity.sha256 = Some(hash.clone());
                            if !hex_digest_eq(&hash, sha256) {
                                report.error(
                                    "row_identity_sha256_mismatch",
                                    format!(
                                        "row_identity SHA-256 was {hash}, manifest declares {sha256}"
                                    ),
                                );
                            }
                        }
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
            "calibration_schema_version_unsupported",
            format!(
                "calibration.schema_version must be {CALIBRATION_SCHEMA_VERSION}, got {}",
                calibration.schema_version
            ),
        );
    }
    if calibration.profile_id.trim().is_empty() {
        report.error(
            "calibration_profile_id_empty",
            "calibration.profile_id must be non-empty",
        );
    }
    if calibration
        .created_at
        .as_ref()
        .is_some_and(|created_at| DateTime::parse_from_rfc3339(created_at).is_err())
    {
        report.error(
            "calibration_created_at_invalid",
            "calibration.created_at must parse as RFC3339 when present",
        );
    }
    if calibration.calibrated_for.model.trim().is_empty() {
        report.error(
            "calibration_encoder_model_empty",
            "calibration.calibrated_for.model must be non-empty",
        );
    }
    if calibration.calibrated_for.dim == 0 {
        report.error(
            "calibration_encoder_dim_zero",
            "calibration.calibrated_for.dim must be greater than zero",
        );
    }
    validate_optional_non_empty(
        "calibration_encoder_model_revision_empty",
        "calibration.calibrated_for.model_revision must be non-empty when present",
        calibration.calibrated_for.model_revision.as_deref(),
        report,
    );
    validate_optional_non_empty(
        "calibration_encoder_normalization_empty",
        "calibration.calibrated_for.normalization must be non-empty when present",
        calibration.calibrated_for.normalization.as_deref(),
        report,
    );
    if calibration.ordinalization.dim() == 0 {
        report.error(
            "calibration_ordinalization_dim_zero",
            "calibration.ordinalization.dim must be greater than zero",
        );
    }
    match &calibration.ordinalization {
        CalibrationOrdinalization::TopK { k, .. } if *k == 0 => {
            report.error(
                "calibration_ordinalization_artifact_mismatch",
                "calibration top_k.k must be greater than zero",
            );
        }
        CalibrationOrdinalization::Bucket { bits, .. } if !matches!(*bits, 1 | 2 | 4) => {
            report.error(
                "calibration_ordinalization_artifact_mismatch",
                "calibration bucket.bits must be 1, 2, or 4",
            );
        }
        CalibrationOrdinalization::CallerDefined { name, .. } if name.trim().is_empty() => {
            report.error(
                "calibration_ordinalization_artifact_mismatch",
                "calibration caller_defined.name must be non-empty",
            );
        }
        _ => {}
    }
    match &calibration.null_model {
        NullModelSpec::EmpiricalTailTable { statistic } if statistic.trim().is_empty() => {
            report.error(
                "calibration_null_statistic_empty",
                "calibration.null_model.statistic must be non-empty",
            );
        }
        NullModelSpec::CallerDefined {
            name,
            parameterization,
        } => {
            if name.trim().is_empty() {
                report.error(
                    "calibration_null_name_empty",
                    "calibration.null_model.name must be non-empty",
                );
            }
            validate_optional_non_empty(
                "calibration_null_parameterization_empty",
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
            "calibration_encoder_model_mismatch",
            format!(
                "calibration model {:?} does not match embedding.model {:?}",
                calibration.calibrated_for.model, embedding.model
            ),
        );
    }
    if calibration.calibrated_for.dim != embedding.dim {
        report.error(
            "calibration_encoder_dim_mismatch",
            format!(
                "calibration dim {} does not match embedding.dim {}",
                calibration.calibrated_for.dim, embedding.dim
            ),
        );
    }
    compare_optional_identity(
        "calibration_encoder_model_revision_mismatch",
        "model_revision",
        embedding.model_revision.as_deref(),
        calibration.calibrated_for.model_revision.as_deref(),
        report,
    );
    compare_optional_identity(
        "calibration_encoder_normalization_mismatch",
        "normalization",
        embedding.normalization.as_deref(),
        calibration.calibrated_for.normalization.as_deref(),
        report,
    );
}

fn compare_optional_identity(
    code: &str,
    field: &str,
    embedding_value: Option<&str>,
    calibration_value: Option<&str>,
    report: &mut VerificationReport,
) {
    match (embedding_value, calibration_value) {
        (Some(expected), Some(observed)) if expected == observed => {}
        (None, None) => {}
        _ => report.error(
            code,
            format!("calibration encoder {field} does not match embedding.{field}"),
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
            "calibration_ordinalization_dim_mismatch",
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
            "calibration_ordinalization_artifact_mismatch",
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
            "calibration_null_model_ordinalization_mismatch",
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
                "calibration_profile_unexpected",
                "uniform_hypergeometric calibration must not include a profile artifact",
            );
        }
        return;
    }

    let Some(profile) = &calibration.profile else {
        report.error(
            "calibration_profile_required",
            "non-uniform calibration requires a profile artifact",
        );
        return;
    };

    report.calibration.profile_manifest_path = Some(profile.path.clone());
    if profile.path.trim().is_empty() {
        report.error(
            "calibration_profile_path_empty",
            "calibration.profile.path must be non-empty",
        );
    }
    if !is_sha256_hex(&profile.sha256) {
        report.error(
            "calibration_profile_sha256_invalid",
            "calibration.profile.sha256 must be a lowercase 64-character hex SHA-256 digest",
        );
    }
    if profile.file_size_bytes == 0 {
        report.error(
            "calibration_profile_file_size_zero",
            "calibration.profile.file_size_bytes must be greater than zero",
        );
    }
    if profile.dim != artifact.dim {
        report.error(
            "calibration_profile_dim_mismatch",
            format!(
                "calibration profile dim {} does not match artifact.dim {}",
                profile.dim, artifact.dim
            ),
        );
    }
    if profile.sample_count == 0 {
        report.error(
            "calibration_profile_sample_count_zero",
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
            "calibration_profile",
            &mut report.errors,
        ) {
            report.calibration.profile_canonical_path =
                Some(path_to_display(&resolved.canonical_path));
            match sha256_file(&resolved.resolved_path) {
                Ok(hash) => {
                    report.calibration.profile_sha256 = Some(hash.sha256.clone());
                    report.calibration.profile_size_bytes = Some(hash.size_bytes);
                    if !hex_digest_eq(&hash.sha256, &profile.sha256) {
                        report.error(
                            "calibration_profile_sha256_mismatch",
                            format!(
                                "calibration profile SHA-256 was {}, manifest declares {}",
                                hash.sha256, profile.sha256
                            ),
                        );
                    }
                    if hash.size_bytes != profile.file_size_bytes {
                        report.error(
                            "calibration_profile_file_size_mismatch",
                            format!(
                                "calibration profile size was {}, manifest declares {}",
                                hash.size_bytes, profile.file_size_bytes
                            ),
                        );
                    }
                }
                Err(err) => report.error(
                    "calibration_profile_hash_failed",
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
            "calibration_profile_source_digest_invalid",
            "calibration.profile.source_digest must be sha256:<lowercase-hex>",
        );
        return;
    };
    if !is_sha256_hex(digest) {
        report.error(
            "calibration_profile_source_digest_invalid",
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
                "calibration_null_parameterization_mismatch",
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
                "calibration_null_parameterization_mismatch",
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
            "calibration_profile_parameterization_ordinalization_mismatch",
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
            "calibration_profile_format_empty",
            "calibration.profile.format must be non-empty",
        );
    }

    if profile.shape.is_empty() {
        return;
    }

    if let Some(expected) = expected_profile_shape(profile.parameterization, ordinalization) {
        if profile.shape != expected {
            report.error(
                "calibration_profile_shape_mismatch",
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
            "calibration_profile_shape_mismatch",
            "calibration.profile.shape product overflows u64",
        );
        return;
    };
    let Some(expected_bytes) = values.checked_mul(bytes_per_value) else {
        report.error(
            "calibration_profile_shape_mismatch",
            "calibration.profile.shape byte size overflows u64",
        );
        return;
    };
    if profile.file_size_bytes != expected_bytes {
        report.error(
            "calibration_profile_file_size_mismatch",
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
            CalibrationOrdinalization::Bucket { dim, bits } => Some(vec![*dim, 1usize << *bits]),
            _ => None,
        },
        ProfileParameterization::RankPositionFrequency => {
            Some(vec![ordinalization.dim(), ordinalization.dim()])
        }
        ProfileParameterization::EmpiricalTailTable => None,
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
    pub limits: ResourceLimits,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_manifest_bytes: u64,
    pub max_row_identity_jsonl_line_bytes: usize,
    pub max_row_identity_rows: usize,
    pub max_report_issues: usize,
    pub max_cached_report_bytes: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_manifest_bytes: DEFAULT_MAX_MANIFEST_BYTES,
            max_row_identity_jsonl_line_bytes: DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES,
            max_row_identity_rows: DEFAULT_MAX_ROW_IDENTITY_ROWS,
            max_report_issues: DEFAULT_MAX_REPORT_ISSUES,
            max_cached_report_bytes: DEFAULT_MAX_CACHED_REPORT_BYTES,
        }
    }
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
pub struct Embedding {
    pub model: String,
    pub dim: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_revision: Option<String>,
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
    pub calibration: CalibrationReport,
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

fn push_report_issue_bounded(
    errors: &mut Vec<ReportIssue>,
    limits: &ResourceLimits,
    code: impl Into<String>,
    message: impl Into<String>,
) {
    let limit = limits.max_report_issues.max(1);
    if errors.len() < limit {
        errors.push(ReportIssue::new(code, message));
        return;
    }
    if errors
        .iter()
        .any(|issue| issue.code == "verification_report_issue_limit_exceeded")
    {
        return;
    }
    errors.truncate(limit.saturating_sub(1));
    errors.push(ReportIssue::new(
        "verification_report_issue_limit_exceeded",
        format!("verification report issue count exceeded max_report_issues={limit}"),
    ));
}

fn enforce_report_issue_limit(errors: &mut Vec<ReportIssue>, limits: &ResourceLimits) {
    let limit = limits.max_report_issues.max(1);
    if errors.len() <= limit {
        return;
    }
    errors.retain(|issue| issue.code != "verification_report_issue_limit_exceeded");
    errors.truncate(limit.saturating_sub(1));
    errors.push(ReportIssue::new(
        "verification_report_issue_limit_exceeded",
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

pub fn sha256_file_bounded(
    path: impl AsRef<Path>,
    max_bytes: u64,
    code: &'static str,
    context: &'static str,
) -> Result<FileHash, ManifestError> {
    let path = path.as_ref();
    let bytes = read_bounded_file(path, max_bytes, code, context)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(FileHash {
        sha256: hex::encode(hasher.finalize()),
        size_bytes: bytes.len() as u64,
    })
}

#[derive(Clone, Debug)]
pub enum CreateRowIdentity {
    RowIdIdentity,
    Jsonl(PathBuf),
}

#[derive(Clone, Debug, Default)]
pub struct CreateManifestOptions {
    pub allow_absolute_paths: bool,
    pub allow_path_escape: bool,
    pub limits: ResourceLimits,
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
    let index_hash = sha256_file(index_path)?;
    let artifact = Artifact {
        path: manifest_path_for_create(index_path, out_base, &options, "artifact")?,
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
            let mut row_errors = Vec::new();
            let stats = validate_jsonl_rows(
                &path,
                false,
                &options.limits,
                Some(metadata.vector_count),
                &mut row_errors,
            )?;
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

    let invocation_id = format!("urn:uuid:{}", Uuid::new_v4());
    Ok(IndexManifest {
        schema_version: SCHEMA_VERSION.to_string(),
        manifest_id: format!("urn:uuid:{}", Uuid::new_v4()),
        created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true),
        artifact,
        embedding: Embedding {
            model: embedding_model.into(),
            dim: metadata.dim,
            model_revision: None,
            corpus_digest: None,
            embedding_matrix_digest: None,
            normalization: None,
        },
        calibration: None,
        row_identity,
        build: Some(BuildInfo {
            invocation_id,
            builder_id: Some("ordvec-manifest".to_string()),
            source_repo: None,
            source_commit: None,
            ci_provider: None,
            ci_run_id: None,
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
    let mut row_count = 0usize;
    let mut line = Vec::new();
    let row_read_limit = expected_row_count
        .and_then(|row_count| row_count.checked_add(1))
        .unwrap_or(usize::MAX)
        .min(limits.max_row_identity_rows);
    let mut reached_eof = true;

    while let Some(too_long) = read_bounded_line(
        &mut reader,
        limits.max_row_identity_jsonl_line_bytes,
        &mut line,
        &mut hasher,
    )? {
        let line_idx = row_count;
        row_count += 1;
        if row_count > row_read_limit {
            reached_eof = false;
            push_report_issue_bounded(
                errors,
                limits,
                "row_identity_row_count_limit_exceeded",
                format!(
                    "row identity file has more than max_row_identity_rows={} rows",
                    limits.max_row_identity_rows
                ),
            );
            break;
        }
        if too_long {
            reached_eof = false;
            push_report_issue_bounded(
                errors,
                limits,
                "row_identity_line_too_large",
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
                    "row_identity_jsonl_invalid_json",
                    format!("line {line_idx} is not a strict row object: {err}"),
                );
                continue;
            }
        };
        if row.row_id != line_idx {
            push_report_issue_bounded(
                errors,
                limits,
                "row_identity_row_id_mismatch",
                format!("line {line_idx} has row_id {}", row.row_id),
            );
        }
        validate_row_id_string("db_id", &row.db_id, line_idx, limits, errors);
        if let Some(parent_id) = &row.parent_id {
            validate_row_id_string("parent_id", parent_id, line_idx, limits, errors);
        }
        if !allow_duplicate_db_ids && !seen.insert(row.db_id) {
            push_report_issue_bounded(
                errors,
                limits,
                "row_identity_duplicate_db_id",
                format!("line {line_idx} repeats db_id"),
            );
        }
    }

    Ok(JsonlStats {
        row_count,
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

fn validate_row_id_string(
    field: &str,
    value: &str,
    line_idx: usize,
    limits: &ResourceLimits,
    errors: &mut Vec<ReportIssue>,
) {
    if value.is_empty() {
        push_report_issue_bounded(
            errors,
            limits,
            format!("row_identity_{field}_empty"),
            format!("line {line_idx} has empty {field}"),
        );
    }
    if value.contains('\0') {
        push_report_issue_bounded(
            errors,
            limits,
            format!("row_identity_{field}_contains_nul"),
            format!("line {line_idx} {field} contains NUL"),
        );
    }
}

fn manifest_path_for_create(
    path: &Path,
    base_dir: &Path,
    options: &CreateManifestOptions,
    context: &str,
) -> Result<String, ManifestError> {
    let canonical_path = fs::canonicalize(path)?;
    let canonical_base = fs::canonicalize(base_dir)?;
    if let Ok(relative) = canonical_path.strip_prefix(&canonical_base) {
        if !relative.as_os_str().is_empty() {
            return Ok(path_to_manifest_string(relative));
        }
        return Ok(".".to_string());
    }

    if !options.allow_path_escape {
        return Err(ManifestError::invalid(format!(
            "{context} path {} is outside manifest directory {}; use --allow-path-escape to create a manifest that requires non-default verification policy",
            canonical_path.display(),
            canonical_base.display()
        )));
    }

    if let Some(relative) = relative_path_between(&canonical_base, &canonical_path) {
        return Ok(path_to_manifest_string(&relative));
    }

    if options.allow_absolute_paths {
        return Ok(path_to_manifest_string(&canonical_path));
    }

    Err(ManifestError::invalid(format!(
        "{context} path {} cannot be expressed relative to manifest directory {}; use --allow-absolute-paths with --allow-path-escape",
        canonical_path.display(),
        canonical_base.display()
    )))
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
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f'))
}

fn hex_digest_eq(a: &str, b: &str) -> bool {
    a == b
}

#[cfg(feature = "sqlite")]
pub mod sqlite;
