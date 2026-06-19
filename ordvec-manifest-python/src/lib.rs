//! Python bindings for the `ordvec-manifest` verifier crate.

use ordvec_manifest_core::{
    create_manifest_for_index_with_options, load_manifest_file_with_options,
    sha256_file as hash_file, write_manifest_file, CreateAuxiliaryArtifact, CreateManifestOptions,
    CreateRowIdentity, ManifestError, ResourceLimits, VerifiedLoadPlanError, VerifyOptions,
    CALIBRATION_SCHEMA_VERSION, DEFAULT_MAX_AUXILIARY_ARTIFACTS,
    DEFAULT_MAX_AUXILIARY_ARTIFACT_BYTES, DEFAULT_MAX_CACHED_REPORT_BYTES,
    DEFAULT_MAX_CALIBRATION_PROFILE_BYTES, DEFAULT_MAX_ENCODER_DISTORTION_PROFILE_BYTES,
    DEFAULT_MAX_MANIFEST_BYTES, DEFAULT_MAX_REPORT_ISSUES,
    DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES, DEFAULT_MAX_ROW_IDENTITY_ROWS,
    DEFAULT_MAX_ROW_IDENTITY_TRACKED_DB_ID_BYTES, ENCODER_DISTORTION_SCHEMA_VERSION,
    SCHEMA_VERSION,
};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use serde::Serialize;
use std::path::{Path, PathBuf};

fn manifest_error(err: ManifestError) -> PyErr {
    match err {
        ManifestError::Io(err) => pyo3::exceptions::PyOSError::new_err(err.to_string()),
        ManifestError::Json(err) => pyo3::exceptions::PyValueError::new_err(err.to_string()),
        ManifestError::Invalid(message) => pyo3::exceptions::PyValueError::new_err(message),
        ManifestError::LimitExceeded { code, message } => {
            pyo3::exceptions::PyValueError::new_err(format!("{code}: {message}"))
        }
    }
}

fn value_error(err: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

fn verified_load_plan_error(err: VerifiedLoadPlanError) -> PyErr {
    match err {
        VerifiedLoadPlanError::Manifest(err) => manifest_error(err),
        VerifiedLoadPlanError::VerificationFailed(_)
        | VerifiedLoadPlanError::IncompletePlan { .. } => value_error(err),
    }
}

fn json_to_py<T: Serialize>(py: Python<'_>, value: &T) -> PyResult<Py<PyAny>> {
    let text = serde_json::to_string(value).map_err(value_error)?;
    let json = PyModule::import(py, pyo3::intern!(py, "json"))?;
    let loads = json.getattr(pyo3::intern!(py, "loads"))?;
    Ok(loads.call1((text,))?.unbind())
}

fn path_to_string(path: &Path) -> String {
    strip_windows_verbatim_prefix(path.to_string_lossy().into_owned())
}

fn strip_windows_verbatim_prefix(path: String) -> String {
    if let Some(rest) = path.strip_prefix("\\\\?\\UNC\\") {
        format!("\\\\{rest}")
    } else if let Some(rest) = path.strip_prefix("\\\\?\\") {
        rest.to_string()
    } else {
        path
    }
}

#[cfg(test)]
mod tests {
    use super::strip_windows_verbatim_prefix;

    #[test]
    fn strips_windows_verbatim_drive_prefix() {
        assert_eq!(
            strip_windows_verbatim_prefix(r"\\?\C:\tmp\ids.bin".to_string()),
            r"C:\tmp\ids.bin"
        );
    }

    #[test]
    fn strips_windows_verbatim_unc_prefix() {
        assert_eq!(
            strip_windows_verbatim_prefix(r"\\?\UNC\server\share\ids.bin".to_string()),
            r"\\server\share\ids.bin"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn resource_limits(
    max_manifest_bytes: Option<u64>,
    max_row_map_line_bytes: Option<usize>,
    max_row_map_rows: Option<usize>,
    max_row_map_tracked_id_bytes: Option<usize>,
    max_auxiliary_artifacts: Option<usize>,
    max_auxiliary_artifact_bytes: Option<u64>,
    max_calibration_profile_bytes: Option<u64>,
    max_encoder_distortion_profile_bytes: Option<u64>,
    max_report_issues: Option<usize>,
    max_cached_report_bytes: Option<u64>,
) -> ResourceLimits {
    let mut limits = ResourceLimits::default();
    if let Some(value) = max_manifest_bytes {
        limits.max_manifest_bytes = value;
    }
    if let Some(value) = max_row_map_line_bytes {
        limits.max_row_identity_jsonl_line_bytes = value;
    }
    if let Some(value) = max_row_map_rows {
        limits.max_row_identity_rows = value;
    }
    if let Some(value) = max_row_map_tracked_id_bytes {
        limits.max_row_identity_tracked_db_id_bytes = value;
    }
    if let Some(value) = max_auxiliary_artifacts {
        limits.max_auxiliary_artifacts = value;
    }
    if let Some(value) = max_auxiliary_artifact_bytes {
        limits.max_auxiliary_artifact_bytes = value;
    }
    if let Some(value) = max_calibration_profile_bytes {
        limits.max_calibration_profile_bytes = value;
    }
    if let Some(value) = max_encoder_distortion_profile_bytes {
        limits.max_encoder_distortion_profile_bytes = value;
    }
    if let Some(value) = max_report_issues {
        limits.max_report_issues = value;
    }
    if let Some(value) = max_cached_report_bytes {
        limits.max_cached_report_bytes = value;
    }
    limits
}

#[derive(Serialize)]
struct PythonResourceLimits {
    max_manifest_bytes: u64,
    max_row_map_line_bytes: usize,
    max_row_map_rows: usize,
    max_row_map_tracked_id_bytes: usize,
    max_auxiliary_artifacts: usize,
    max_auxiliary_artifact_bytes: u64,
    max_calibration_profile_bytes: u64,
    max_encoder_distortion_profile_bytes: u64,
    max_report_issues: usize,
    max_cached_report_bytes: u64,
}

impl From<ResourceLimits> for PythonResourceLimits {
    fn from(limits: ResourceLimits) -> Self {
        Self {
            max_manifest_bytes: limits.max_manifest_bytes,
            max_row_map_line_bytes: limits.max_row_identity_jsonl_line_bytes,
            max_row_map_rows: limits.max_row_identity_rows,
            max_row_map_tracked_id_bytes: limits.max_row_identity_tracked_db_id_bytes,
            max_auxiliary_artifacts: limits.max_auxiliary_artifacts,
            max_auxiliary_artifact_bytes: limits.max_auxiliary_artifact_bytes,
            max_calibration_profile_bytes: limits.max_calibration_profile_bytes,
            max_encoder_distortion_profile_bytes: limits.max_encoder_distortion_profile_bytes,
            max_report_issues: limits.max_report_issues,
            max_cached_report_bytes: limits.max_cached_report_bytes,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_options(
    index: Option<PathBuf>,
    allow_absolute_paths: bool,
    allow_path_escape: bool,
    allow_duplicate_db_ids: bool,
    max_manifest_bytes: Option<u64>,
    max_row_map_line_bytes: Option<usize>,
    max_row_map_rows: Option<usize>,
    max_row_map_tracked_id_bytes: Option<usize>,
    max_auxiliary_artifacts: Option<usize>,
    max_auxiliary_artifact_bytes: Option<u64>,
    max_calibration_profile_bytes: Option<u64>,
    max_encoder_distortion_profile_bytes: Option<u64>,
    max_report_issues: Option<usize>,
    max_cached_report_bytes: Option<u64>,
) -> VerifyOptions {
    VerifyOptions {
        allow_absolute_paths,
        allow_path_escape,
        allow_duplicate_db_ids,
        index_override: index,
        limits: resource_limits(
            max_manifest_bytes,
            max_row_map_line_bytes,
            max_row_map_rows,
            max_row_map_tracked_id_bytes,
            max_auxiliary_artifacts,
            max_auxiliary_artifact_bytes,
            max_calibration_profile_bytes,
            max_encoder_distortion_profile_bytes,
            max_report_issues,
            max_cached_report_bytes,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn create_options(
    allow_absolute_paths: bool,
    allow_path_escape: bool,
    max_manifest_bytes: Option<u64>,
    max_row_map_line_bytes: Option<usize>,
    max_row_map_rows: Option<usize>,
    max_row_map_tracked_id_bytes: Option<usize>,
    max_auxiliary_artifacts: Option<usize>,
    max_auxiliary_artifact_bytes: Option<u64>,
    max_calibration_profile_bytes: Option<u64>,
    max_encoder_distortion_profile_bytes: Option<u64>,
    max_report_issues: Option<usize>,
    max_cached_report_bytes: Option<u64>,
    auxiliary_artifacts: Vec<CreateAuxiliaryArtifact>,
) -> CreateManifestOptions {
    CreateManifestOptions {
        allow_absolute_paths,
        allow_path_escape,
        limits: resource_limits(
            max_manifest_bytes,
            max_row_map_line_bytes,
            max_row_map_rows,
            max_row_map_tracked_id_bytes,
            max_auxiliary_artifacts,
            max_auxiliary_artifact_bytes,
            max_calibration_profile_bytes,
            max_encoder_distortion_profile_bytes,
            max_report_issues,
            max_cached_report_bytes,
        ),
        auxiliary_artifacts,
    }
}

fn parse_auxiliary_artifacts(
    py: Python<'_>,
    auxiliary_artifacts: Option<Py<PyAny>>,
) -> PyResult<Vec<CreateAuxiliaryArtifact>> {
    let Some(auxiliary_artifacts) = auxiliary_artifacts else {
        return Ok(Vec::new());
    };
    let auxiliary_artifacts = auxiliary_artifacts.bind(py);
    let mut parsed = Vec::new();
    for item in auxiliary_artifacts.try_iter()? {
        let item = item?;
        let name = item.get_item("name")?.extract::<String>()?;
        let path = item.get_item("path")?.extract::<PathBuf>()?;
        let required = match item.get_item("required") {
            Ok(value) => value.extract::<bool>()?,
            Err(err) if err.is_instance_of::<PyKeyError>(py) => true,
            Err(err) => return Err(err),
        };
        parsed.push(CreateAuxiliaryArtifact {
            name,
            path,
            required,
        });
    }
    Ok(parsed)
}

#[pyfunction]
fn default_resource_limits(py: Python<'_>) -> PyResult<Py<PyAny>> {
    json_to_py(py, &PythonResourceLimits::from(ResourceLimits::default()))
}

#[pyfunction]
fn sha256_file(py: Python<'_>, path: PathBuf) -> PyResult<Py<PyAny>> {
    let hash = py
        .detach(|| hash_file(path))
        .map_err(|err| pyo3::exceptions::PyOSError::new_err(err.to_string()))?;
    json_to_py(py, &hash)
}

#[pyfunction]
#[pyo3(signature = (
    manifest,
    *,
    max_manifest_bytes = None,
    max_row_map_line_bytes = None,
    max_row_map_rows = None,
    max_row_map_tracked_id_bytes = None,
    max_auxiliary_artifacts = None,
    max_auxiliary_artifact_bytes = None,
    max_calibration_profile_bytes = None,
    max_encoder_distortion_profile_bytes = None,
    max_report_issues = None,
    max_cached_report_bytes = None
))]
#[allow(clippy::too_many_arguments)]
fn inspect_manifest(
    py: Python<'_>,
    manifest: PathBuf,
    max_manifest_bytes: Option<u64>,
    max_row_map_line_bytes: Option<usize>,
    max_row_map_rows: Option<usize>,
    max_row_map_tracked_id_bytes: Option<usize>,
    max_auxiliary_artifacts: Option<usize>,
    max_auxiliary_artifact_bytes: Option<u64>,
    max_calibration_profile_bytes: Option<u64>,
    max_encoder_distortion_profile_bytes: Option<u64>,
    max_report_issues: Option<usize>,
    max_cached_report_bytes: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let options = verify_options(
        None,
        false,
        false,
        false,
        max_manifest_bytes,
        max_row_map_line_bytes,
        max_row_map_rows,
        max_row_map_tracked_id_bytes,
        max_auxiliary_artifacts,
        max_auxiliary_artifact_bytes,
        max_calibration_profile_bytes,
        max_encoder_distortion_profile_bytes,
        max_report_issues,
        max_cached_report_bytes,
    );
    let document = py
        .detach(|| load_manifest_file_with_options(manifest, &options))
        .map_err(manifest_error)?;
    json_to_py(py, &document.manifest)
}

#[pyfunction]
#[pyo3(signature = (
    manifest,
    *,
    index = None,
    allow_absolute_paths = false,
    allow_path_escape = false,
    allow_duplicate_db_ids = false,
    max_manifest_bytes = None,
    max_row_map_line_bytes = None,
    max_row_map_rows = None,
    max_row_map_tracked_id_bytes = None,
    max_auxiliary_artifacts = None,
    max_auxiliary_artifact_bytes = None,
    max_calibration_profile_bytes = None,
    max_encoder_distortion_profile_bytes = None,
    max_report_issues = None,
    max_cached_report_bytes = None
))]
#[allow(clippy::too_many_arguments)]
fn verify_manifest(
    py: Python<'_>,
    manifest: PathBuf,
    index: Option<PathBuf>,
    allow_absolute_paths: bool,
    allow_path_escape: bool,
    allow_duplicate_db_ids: bool,
    max_manifest_bytes: Option<u64>,
    max_row_map_line_bytes: Option<usize>,
    max_row_map_rows: Option<usize>,
    max_row_map_tracked_id_bytes: Option<usize>,
    max_auxiliary_artifacts: Option<usize>,
    max_auxiliary_artifact_bytes: Option<u64>,
    max_calibration_profile_bytes: Option<u64>,
    max_encoder_distortion_profile_bytes: Option<u64>,
    max_report_issues: Option<usize>,
    max_cached_report_bytes: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let options = verify_options(
        index,
        allow_absolute_paths,
        allow_path_escape,
        allow_duplicate_db_ids,
        max_manifest_bytes,
        max_row_map_line_bytes,
        max_row_map_rows,
        max_row_map_tracked_id_bytes,
        max_auxiliary_artifacts,
        max_auxiliary_artifact_bytes,
        max_calibration_profile_bytes,
        max_encoder_distortion_profile_bytes,
        max_report_issues,
        max_cached_report_bytes,
    );
    let report = py
        .detach(|| {
            let document = load_manifest_file_with_options(manifest, &options)?;
            Ok::<_, ManifestError>(ordvec_manifest_core::verify_manifest(&document, options))
        })
        .map_err(manifest_error)?;
    json_to_py(py, &report)
}

#[derive(Serialize)]
struct PythonVerifiedLoadPlan {
    manifest_path: Option<String>,
    artifact_path: String,
    metadata: ordvec_manifest_core::MetadataReport,
    row_identity: PythonVerifiedRowIdentityPlan,
    auxiliary_artifacts: Vec<PythonVerifiedAuxiliaryArtifactPlan>,
    report: ordvec_manifest_core::VerificationReport,
}

#[derive(Serialize)]
struct PythonVerifiedRowIdentityPlan {
    kind: String,
    path: Option<String>,
    row_count: usize,
    validated_rows: Option<usize>,
    sha256: Option<String>,
}

#[derive(Serialize)]
struct PythonVerifiedAuxiliaryArtifactPlan {
    name: String,
    path: Option<String>,
    required: bool,
    state: ordvec_manifest_core::AuxiliaryArtifactState,
    reason_code: Option<String>,
    sha256: Option<String>,
    size_bytes: Option<u64>,
}

#[pyfunction]
#[pyo3(signature = (
    manifest,
    *,
    index = None,
    allow_absolute_paths = false,
    allow_path_escape = false,
    allow_duplicate_db_ids = false,
    max_manifest_bytes = None,
    max_row_map_line_bytes = None,
    max_row_map_rows = None,
    max_row_map_tracked_id_bytes = None,
    max_auxiliary_artifacts = None,
    max_auxiliary_artifact_bytes = None,
    max_calibration_profile_bytes = None,
    max_encoder_distortion_profile_bytes = None,
    max_report_issues = None,
    max_cached_report_bytes = None
))]
#[allow(clippy::too_many_arguments)]
fn verify_for_load(
    py: Python<'_>,
    manifest: PathBuf,
    index: Option<PathBuf>,
    allow_absolute_paths: bool,
    allow_path_escape: bool,
    allow_duplicate_db_ids: bool,
    max_manifest_bytes: Option<u64>,
    max_row_map_line_bytes: Option<usize>,
    max_row_map_rows: Option<usize>,
    max_row_map_tracked_id_bytes: Option<usize>,
    max_auxiliary_artifacts: Option<usize>,
    max_auxiliary_artifact_bytes: Option<u64>,
    max_calibration_profile_bytes: Option<u64>,
    max_encoder_distortion_profile_bytes: Option<u64>,
    max_report_issues: Option<usize>,
    max_cached_report_bytes: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let options = verify_options(
        index,
        allow_absolute_paths,
        allow_path_escape,
        allow_duplicate_db_ids,
        max_manifest_bytes,
        max_row_map_line_bytes,
        max_row_map_rows,
        max_row_map_tracked_id_bytes,
        max_auxiliary_artifacts,
        max_auxiliary_artifact_bytes,
        max_calibration_profile_bytes,
        max_encoder_distortion_profile_bytes,
        max_report_issues,
        max_cached_report_bytes,
    );
    let plan = py
        .detach(|| ordvec_manifest_core::verify_for_load(manifest, options))
        .map_err(verified_load_plan_error)?;
    let row_identity = plan.row_identity();
    let value = PythonVerifiedLoadPlan {
        manifest_path: plan.manifest_path().map(path_to_string),
        artifact_path: path_to_string(plan.artifact_path()),
        metadata: plan.metadata().clone(),
        row_identity: PythonVerifiedRowIdentityPlan {
            kind: row_identity.kind().to_string(),
            path: row_identity.path().map(path_to_string),
            row_count: row_identity.row_count(),
            validated_rows: row_identity.validated_rows(),
            sha256: row_identity.sha256().map(str::to_string),
        },
        auxiliary_artifacts: plan
            .auxiliary_artifacts()
            .iter()
            .map(|artifact| PythonVerifiedAuxiliaryArtifactPlan {
                name: artifact.name().to_string(),
                path: artifact.path().map(path_to_string),
                required: artifact.required(),
                state: artifact.state(),
                reason_code: artifact.reason_code().map(str::to_string),
                sha256: artifact.sha256().map(str::to_string),
                size_bytes: artifact.size_bytes(),
            })
            .collect(),
        report: plan.report().clone(),
    };
    json_to_py(py, &value)
}

#[pyfunction]
#[pyo3(signature = (
    index,
    out,
    embedding_model,
    *,
    row_map = None,
    row_id_is_identity = false,
    auxiliary_artifacts = None,
    allow_absolute_paths = false,
    allow_path_escape = false,
    max_manifest_bytes = None,
    max_row_map_line_bytes = None,
    max_row_map_rows = None,
    max_row_map_tracked_id_bytes = None,
    max_auxiliary_artifacts = None,
    max_auxiliary_artifact_bytes = None,
    max_calibration_profile_bytes = None,
    max_encoder_distortion_profile_bytes = None,
    max_report_issues = None,
    max_cached_report_bytes = None
))]
#[allow(clippy::too_many_arguments)]
fn create_manifest(
    py: Python<'_>,
    index: PathBuf,
    out: PathBuf,
    embedding_model: String,
    row_map: Option<PathBuf>,
    row_id_is_identity: bool,
    auxiliary_artifacts: Option<Py<PyAny>>,
    allow_absolute_paths: bool,
    allow_path_escape: bool,
    max_manifest_bytes: Option<u64>,
    max_row_map_line_bytes: Option<usize>,
    max_row_map_rows: Option<usize>,
    max_row_map_tracked_id_bytes: Option<usize>,
    max_auxiliary_artifacts: Option<usize>,
    max_auxiliary_artifact_bytes: Option<u64>,
    max_calibration_profile_bytes: Option<u64>,
    max_encoder_distortion_profile_bytes: Option<u64>,
    max_report_issues: Option<usize>,
    max_cached_report_bytes: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let row_identity = match (row_map, row_id_is_identity) {
        (Some(_), true) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "use either row_map or row_id_is_identity, not both",
            ));
        }
        (Some(path), false) => CreateRowIdentity::Jsonl(path),
        (None, true) => CreateRowIdentity::RowIdIdentity,
        (None, false) => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "one of row_map or row_id_is_identity=True is required",
            ));
        }
    };
    let auxiliary_artifacts = parse_auxiliary_artifacts(py, auxiliary_artifacts)?;
    let options = create_options(
        allow_absolute_paths,
        allow_path_escape,
        max_manifest_bytes,
        max_row_map_line_bytes,
        max_row_map_rows,
        max_row_map_tracked_id_bytes,
        max_auxiliary_artifacts,
        max_auxiliary_artifact_bytes,
        max_calibration_profile_bytes,
        max_encoder_distortion_profile_bytes,
        max_report_issues,
        max_cached_report_bytes,
        auxiliary_artifacts,
    );
    let manifest = py
        .detach(|| {
            let manifest = create_manifest_for_index_with_options(
                index,
                row_identity,
                embedding_model,
                &out,
                options,
            )?;
            write_manifest_file(&manifest, out)?;
            Ok::<_, ManifestError>(manifest)
        })
        .map_err(manifest_error)?;
    json_to_py(py, &manifest)
}

#[pymodule]
fn _ordvec_manifest(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SCHEMA_VERSION", SCHEMA_VERSION)?;
    m.add("CALIBRATION_SCHEMA_VERSION", CALIBRATION_SCHEMA_VERSION)?;
    m.add(
        "ENCODER_DISTORTION_SCHEMA_VERSION",
        ENCODER_DISTORTION_SCHEMA_VERSION,
    )?;
    m.add("DEFAULT_MAX_MANIFEST_BYTES", DEFAULT_MAX_MANIFEST_BYTES)?;
    m.add(
        "DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES",
        DEFAULT_MAX_ROW_IDENTITY_JSONL_LINE_BYTES,
    )?;
    m.add(
        "DEFAULT_MAX_ROW_IDENTITY_ROWS",
        DEFAULT_MAX_ROW_IDENTITY_ROWS,
    )?;
    m.add(
        "DEFAULT_MAX_ROW_IDENTITY_TRACKED_DB_ID_BYTES",
        DEFAULT_MAX_ROW_IDENTITY_TRACKED_DB_ID_BYTES,
    )?;
    m.add(
        "DEFAULT_MAX_AUXILIARY_ARTIFACTS",
        DEFAULT_MAX_AUXILIARY_ARTIFACTS,
    )?;
    m.add(
        "DEFAULT_MAX_AUXILIARY_ARTIFACT_BYTES",
        DEFAULT_MAX_AUXILIARY_ARTIFACT_BYTES,
    )?;
    m.add(
        "DEFAULT_MAX_CALIBRATION_PROFILE_BYTES",
        DEFAULT_MAX_CALIBRATION_PROFILE_BYTES,
    )?;
    m.add(
        "DEFAULT_MAX_ENCODER_DISTORTION_PROFILE_BYTES",
        DEFAULT_MAX_ENCODER_DISTORTION_PROFILE_BYTES,
    )?;
    m.add("DEFAULT_MAX_REPORT_ISSUES", DEFAULT_MAX_REPORT_ISSUES)?;
    m.add(
        "DEFAULT_MAX_CACHED_REPORT_BYTES",
        DEFAULT_MAX_CACHED_REPORT_BYTES,
    )?;
    m.add_function(wrap_pyfunction!(default_resource_limits, m)?)?;
    m.add_function(wrap_pyfunction!(sha256_file, m)?)?;
    m.add_function(wrap_pyfunction!(inspect_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(verify_manifest, m)?)?;
    m.add_function(wrap_pyfunction!(verify_for_load, m)?)?;
    m.add_function(wrap_pyfunction!(create_manifest, m)?)?;
    Ok(())
}
