use crate::{
    codes, resolve_existing_path, sha256_bytes, sha256_file_bounded, validate_jsonl_rows,
    verify_auxiliary_artifacts, verify_manifest, AuxiliaryArtifactState, ManifestDocument,
    ManifestError, ReportIssue, ResourceLimits, RowIdentity, VerificationPathCapture,
    VerificationReport, VerifyOptions,
};
use chrono::{SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

pub fn verify_with_registry(
    db_path: impl AsRef<Path>,
    document: &ManifestDocument,
    manifest_path: impl AsRef<Path>,
    options: VerifyOptions,
    use_cache: bool,
) -> Result<VerificationReport, ManifestError> {
    let mut conn = Connection::open(db_path).map_err(sqlite_err)?;
    init(&mut conn)?;
    if use_cache {
        if let Some(cache_key) = current_cache_key(document, manifest_path.as_ref(), &options)? {
            if let Some(report) = load_cached_report(&conn, &cache_key, &options.limits)? {
                return Ok(report);
            }
        }
    }

    let store_options = options.clone();
    let report = verify_manifest(document, options);
    let cache_key =
        cache_key_from_report(manifest_path.as_ref(), &report, document, &store_options)?;
    store_report(
        &mut conn,
        manifest_path.as_ref(),
        &report,
        cache_key.as_ref(),
        &store_options.limits,
    )?;
    Ok(report)
}

pub fn activate(
    db_path: impl AsRef<Path>,
    document: &ManifestDocument,
    manifest_path: impl AsRef<Path>,
    options: VerifyOptions,
    force: bool,
) -> Result<VerificationReport, ManifestError> {
    let mut conn = Connection::open(db_path).map_err(sqlite_err)?;
    init(&mut conn)?;
    let store_options = options.clone();
    let mut report = verify_manifest(document, options);
    if !report.ok && force {
        report.warnings.push(ReportIssue::new(
            codes::SQLITE_ACTIVATION_FORCED,
            "sqlite activation was forced even though verification failed",
        ));
    }
    let cache_key = if !report.ok && force {
        None
    } else {
        cache_key_from_report(manifest_path.as_ref(), &report, document, &store_options)?
    };
    store_report(
        &mut conn,
        manifest_path.as_ref(),
        &report,
        cache_key.as_ref(),
        &store_options.limits,
    )?;
    if !report.ok && !force {
        return Ok(report);
    }

    conn.execute(
        "INSERT INTO active_manifest(id, manifest_path, activated_at, forced)
         VALUES(1, ?1, ?2, ?3)
         ON CONFLICT(id) DO UPDATE SET
           manifest_path=excluded.manifest_path,
           activated_at=excluded.activated_at,
           forced=excluded.forced",
        params![
            manifest_path.as_ref().display().to_string(),
            Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true),
            i64::from(force),
        ],
    )
    .map_err(sqlite_err)?;
    Ok(report)
}

fn init(conn: &mut Connection) -> Result<(), ManifestError> {
    if verification_reports_needs_migration(conn)? {
        // Migrate atomically. `execute_batch` runs each statement in its own
        // implicit transaction, so a crash between the RENAME and the DROP
        // (or two processes racing this path) would leave a stray
        // `verification_reports_old` behind — after which every future open
        // fails, because the RENAME target already exists. Run the whole
        // migration inside one IMMEDIATE transaction (write lock taken up
        // front), drop any leftover `_old` from a prior interrupted attempt,
        // and re-check need under the lock so a process that lost the race
        // commits a no-op instead of re-migrating an already-v2 table.
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(sqlite_err)?;
        if verification_reports_needs_migration(&tx)? {
            tx.execute_batch(
                "DROP TABLE IF EXISTS verification_reports_old;
                 ALTER TABLE verification_reports RENAME TO verification_reports_old;
                 CREATE TABLE verification_reports(
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    manifest_path TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    ok INTEGER NOT NULL,
                    manifest_location_sha256 TEXT,
                    manifest_sha256 TEXT,
                    options_sha256 TEXT,
                    artifact_sha256 TEXT,
                    row_identity_sha256 TEXT,
                    calibration_profile_sha256 TEXT,
                    auxiliary_artifacts_sha256 TEXT,
                    encoder_distortion_profile_sha256 TEXT,
                    report_json TEXT NOT NULL
                 );
                 INSERT INTO verification_reports(
                    manifest_path, checked_at, ok, report_json
                 )
                 SELECT manifest_path, checked_at, ok, report_json
                 FROM verification_reports_old;
                 DROP TABLE verification_reports_old;",
            )
            .map_err(sqlite_err)?;
        }
        tx.commit().map_err(sqlite_err)?;
    }
    // Schema v2 dropped active_manifest's manifest_id column, and `CREATE
    // TABLE IF NOT EXISTS` below would leave such a stale table in place,
    // making activate()'s INSERT fail at runtime on its NOT NULL column. The
    // registry is rebuildable cache/pointer state — cached verification
    // reports and the active-manifest pointer, never source of truth — so a
    // table whose live schema mismatches the current one is dropped and
    // recreated empty rather than migrated.
    drop_registry_table_on_schema_mismatch(
        conn,
        "active_manifest",
        &["id", "manifest_path", "activated_at", "forced"],
    )?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS verification_reports(
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            manifest_path TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            ok INTEGER NOT NULL,
            manifest_location_sha256 TEXT,
            manifest_sha256 TEXT,
            options_sha256 TEXT,
            artifact_sha256 TEXT,
            row_identity_sha256 TEXT,
            calibration_profile_sha256 TEXT,
            auxiliary_artifacts_sha256 TEXT,
            encoder_distortion_profile_sha256 TEXT,
            report_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS verification_reports_cache_idx
          ON verification_reports(
            manifest_location_sha256,
            manifest_sha256,
            options_sha256,
            artifact_sha256,
            row_identity_sha256,
            calibration_profile_sha256,
            auxiliary_artifacts_sha256,
            encoder_distortion_profile_sha256,
            report_id
          );
        CREATE TABLE IF NOT EXISTS active_manifest(
            id INTEGER PRIMARY KEY CHECK(id = 1),
            manifest_path TEXT NOT NULL,
            activated_at TEXT NOT NULL,
            forced INTEGER NOT NULL
        );",
    )
    .map_err(sqlite_err)?;
    Ok(())
}

fn store_report(
    conn: &mut Connection,
    manifest_path: &Path,
    report: &VerificationReport,
    cache_key: Option<&CacheKey>,
    limits: &ResourceLimits,
) -> Result<(), ManifestError> {
    let tx = conn.transaction().map_err(sqlite_err)?;
    let report_json = serde_json::to_string(report)?;
    if report_json.len() as u64 > limits.max_cached_report_bytes {
        return Err(ManifestError::limit_exceeded(
            codes::SQLITE_CACHED_REPORT_TOO_LARGE,
            format!(
                "cached report is {} bytes, exceeding max_cached_report_bytes={}",
                report_json.len(),
                limits.max_cached_report_bytes
            ),
        ));
    }
    tx.execute(
        "INSERT INTO verification_reports(
            manifest_path,
            checked_at,
            ok,
            manifest_location_sha256,
            manifest_sha256,
            options_sha256,
            artifact_sha256,
            row_identity_sha256,
            calibration_profile_sha256,
            auxiliary_artifacts_sha256,
            encoder_distortion_profile_sha256,
            report_json
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            manifest_path.display().to_string(),
            report.checked_at,
            i64::from(report.ok),
            cache_key.map(|key| key.manifest_location_sha256.as_str()),
            cache_key.map(|key| key.manifest_sha256.as_str()),
            cache_key.map(|key| key.options_sha256.as_str()),
            cache_key.map(|key| key.artifact_sha256.as_str()),
            cache_key.and_then(|key| key.row_identity_sha256.as_deref()),
            cache_key.and_then(|key| key.calibration_profile_sha256.as_deref()),
            cache_key.and_then(|key| key.auxiliary_artifacts_sha256.as_deref()),
            cache_key.and_then(|key| key.encoder_distortion_profile_sha256.as_deref()),
            report_json,
        ],
    )
    .map_err(sqlite_err)?;
    tx.commit().map_err(sqlite_err)?;
    Ok(())
}

fn load_cached_report(
    conn: &Connection,
    cache_key: &CacheKey,
    limits: &ResourceLimits,
) -> Result<Option<VerificationReport>, ManifestError> {
    let cached_row: Option<(i64, i64)> = conn
        .query_row(
            "SELECT report_id, length(CAST(report_json AS BLOB))
             FROM verification_reports
             WHERE manifest_location_sha256 = ?1
               AND manifest_sha256 = ?2
               AND options_sha256 = ?3
               AND artifact_sha256 = ?4
               AND (
                 (row_identity_sha256 IS NULL AND ?5 IS NULL)
                 OR row_identity_sha256 = ?5
               )
               AND (
                 (calibration_profile_sha256 IS NULL AND ?6 IS NULL)
                 OR calibration_profile_sha256 = ?6
               )
               AND (
                 (auxiliary_artifacts_sha256 IS NULL AND ?7 IS NULL)
                 OR auxiliary_artifacts_sha256 = ?7
               )
               AND (
                 (encoder_distortion_profile_sha256 IS NULL AND ?8 IS NULL)
                 OR encoder_distortion_profile_sha256 = ?8
               )
             ORDER BY report_id DESC
             LIMIT 1",
            params![
                cache_key.manifest_location_sha256.as_str(),
                cache_key.manifest_sha256.as_str(),
                cache_key.options_sha256.as_str(),
                cache_key.artifact_sha256.as_str(),
                cache_key.row_identity_sha256.as_deref(),
                cache_key.calibration_profile_sha256.as_deref(),
                cache_key.auxiliary_artifacts_sha256.as_deref(),
                cache_key.encoder_distortion_profile_sha256.as_deref(),
            ],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .map_err(sqlite_err)?;
    let Some((report_id, report_len)) = cached_row else {
        return Ok(None);
    };
    if report_len as u64 > limits.max_cached_report_bytes {
        return Err(ManifestError::limit_exceeded(
            codes::SQLITE_CACHED_REPORT_TOO_LARGE,
            format!(
                "cached report is {report_len} bytes, exceeding max_cached_report_bytes={}",
                limits.max_cached_report_bytes
            ),
        ));
    }

    let report_json: String = conn
        .query_row(
            "SELECT report_json FROM verification_reports WHERE report_id = ?1",
            params![report_id],
            |row| row.get(0),
        )
        .map_err(sqlite_err)?;
    serde_json::from_str(&report_json)
        .map(Some)
        .map_err(ManifestError::from)
}

#[derive(Clone, Debug)]
struct CacheKey {
    manifest_location_sha256: String,
    manifest_sha256: String,
    options_sha256: String,
    artifact_sha256: String,
    row_identity_sha256: Option<String>,
    calibration_profile_sha256: Option<String>,
    auxiliary_artifacts_sha256: Option<String>,
    encoder_distortion_profile_sha256: Option<String>,
}

#[derive(Serialize)]
struct CacheableVerifyOptions {
    allow_absolute_paths: bool,
    allow_path_escape: bool,
    allow_duplicate_db_ids: bool,
    index_override: Option<String>,
    limits: ResourceLimits,
}

#[derive(Serialize)]
struct CacheableManifestLocation {
    manifest_path: String,
    base_dir: String,
}

impl CacheableVerifyOptions {
    fn from_options(options: &VerifyOptions) -> Self {
        Self {
            allow_absolute_paths: options.allow_absolute_paths,
            allow_path_escape: options.allow_path_escape,
            allow_duplicate_db_ids: options.allow_duplicate_db_ids,
            index_override: options
                .index_override
                .as_ref()
                .map(|path| path.display().to_string().replace('\\', "/")),
            limits: options.limits.clone(),
        }
    }
}

fn manifest_location_sha256(
    manifest_path: &Path,
    document: &ManifestDocument,
) -> Result<Option<String>, ManifestError> {
    let manifest_path = match fs::canonicalize(manifest_path) {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    let base_dir = match fs::canonicalize(&document.base_dir) {
        Ok(path) => path,
        Err(_) => return Ok(None),
    };
    let material = CacheableManifestLocation {
        manifest_path: hex::encode(manifest_path.as_os_str().as_encoded_bytes()),
        base_dir: hex::encode(base_dir.as_os_str().as_encoded_bytes()),
    };
    let json = serde_json::to_vec(&material)?;
    Ok(Some(sha256_bytes(&json).sha256))
}

fn current_cache_key(
    document: &ManifestDocument,
    manifest_path: &Path,
    options: &VerifyOptions,
) -> Result<Option<CacheKey>, ManifestError> {
    let manifest_sha256 = match sha256_file_bounded(
        manifest_path,
        options.limits.max_manifest_bytes,
        codes::MANIFEST_FILE_TOO_LARGE,
        "manifest file",
    ) {
        Ok(hash) => hash.sha256,
        Err(_) => return Ok(None),
    };
    let Some(manifest_location_sha256) = manifest_location_sha256(manifest_path, document)? else {
        return Ok(None);
    };
    let options_json = serde_json::to_vec(&CacheableVerifyOptions::from_options(options))?;
    let options_sha256 = sha256_bytes(&options_json).sha256;

    let artifact_path = options
        .index_override
        .as_ref()
        .cloned()
        .unwrap_or_else(|| PathBuf::from(&document.manifest.artifact.path));
    let mut path_errors = Vec::<ReportIssue>::new();
    let Some(artifact) = resolve_existing_path(
        &artifact_path,
        &document.base_dir,
        options,
        &crate::ARTIFACT_PATH_ISSUES,
        &mut path_errors,
    ) else {
        return Ok(None);
    };
    // Bound the cache-key hash exactly like the verify path: declared size
    // with the opt-in ceiling. A bound violation just misses the cache.
    let artifact_sha256 = match sha256_file_bounded(
        &artifact.canonical_path,
        document
            .manifest
            .artifact
            .file_size_bytes
            .min(options.limits.max_index_artifact_bytes),
        codes::ARTIFACT_FILE_TOO_LARGE,
        "index artifact",
    ) {
        Ok(hash) => hash.sha256,
        Err(_) => return Ok(None),
    };

    let row_identity_sha256 = match &document.manifest.row_identity {
        RowIdentity::RowIdIdentity { .. } => None,
        RowIdentity::Jsonl {
            path, row_count, ..
        } => {
            if *row_count > options.limits.max_row_identity_rows {
                return Ok(None);
            }
            let row_path = PathBuf::from(path);
            let Some(row_identity) = resolve_existing_path(
                &row_path,
                &document.base_dir,
                options,
                &crate::ROW_IDENTITY_PATH_ISSUES,
                &mut path_errors,
            ) else {
                return Ok(None);
            };
            let mut row_errors = Vec::new();
            let stats = match validate_jsonl_rows(
                &row_identity.canonical_path,
                options.allow_duplicate_db_ids,
                &options.limits,
                Some(*row_count),
                &mut row_errors,
            ) {
                Ok(stats) => stats,
                Err(_) => return Ok(None),
            };
            if !row_errors.is_empty() || stats.row_count != *row_count {
                return Ok(None);
            }
            stats.sha256
        }
    };
    let calibration_profile_sha256 = current_calibration_profile_sha256(document, options)?;
    let auxiliary_artifacts_sha256 = current_auxiliary_artifacts_sha256(document, options)?;
    let encoder_distortion_profile_sha256 =
        current_encoder_distortion_profile_sha256(document, options)?;

    Ok(Some(CacheKey {
        manifest_location_sha256,
        manifest_sha256,
        options_sha256,
        artifact_sha256,
        row_identity_sha256,
        calibration_profile_sha256,
        auxiliary_artifacts_sha256,
        encoder_distortion_profile_sha256,
    }))
}

fn cache_key_from_report(
    manifest_path: &Path,
    report: &VerificationReport,
    document: &ManifestDocument,
    options: &VerifyOptions,
) -> Result<Option<CacheKey>, ManifestError> {
    let manifest_sha256 = match sha256_file_bounded(
        manifest_path,
        options.limits.max_manifest_bytes,
        codes::MANIFEST_FILE_TOO_LARGE,
        "manifest file",
    ) {
        Ok(hash) => hash.sha256,
        Err(_) => return Ok(None),
    };
    let Some(manifest_location_sha256) = manifest_location_sha256(manifest_path, document)? else {
        return Ok(None);
    };
    let options_json = serde_json::to_vec(&CacheableVerifyOptions::from_options(options))?;
    let options_sha256 = sha256_bytes(&options_json).sha256;
    let Some(artifact_sha256) = report.artifact.sha256.clone() else {
        return Ok(None);
    };
    let row_identity_sha256 = match &document.manifest.row_identity {
        RowIdentity::RowIdIdentity { .. } => None,
        RowIdentity::Jsonl { .. } => {
            let Some(sha256) = report.row_identity.sha256.clone() else {
                return Ok(None);
            };
            Some(sha256)
        }
    };
    let calibration_profile_sha256 = if document
        .manifest
        .calibration
        .as_ref()
        .and_then(|calibration| calibration.profile.as_ref())
        .is_some()
    {
        let Some(sha256) = report.calibration.profile_sha256.clone() else {
            return Ok(None);
        };
        Some(sha256)
    } else {
        None
    };
    let auxiliary_artifacts_sha256 = auxiliary_artifacts_sha256_from_report(document, report)?;
    let encoder_distortion_profile_sha256 = if document
        .manifest
        .encoder_distortion
        .as_ref()
        .and_then(|profile| profile.profile.as_ref())
        .is_some()
    {
        let Some(sha256) = report.encoder_distortion.profile_sha256.clone() else {
            return Ok(None);
        };
        Some(sha256)
    } else {
        None
    };
    Ok(Some(CacheKey {
        manifest_location_sha256,
        manifest_sha256,
        options_sha256,
        artifact_sha256,
        row_identity_sha256,
        calibration_profile_sha256,
        auxiliary_artifacts_sha256,
        encoder_distortion_profile_sha256,
    }))
}

fn current_auxiliary_artifacts_sha256(
    document: &ManifestDocument,
    options: &VerifyOptions,
) -> Result<Option<String>, ManifestError> {
    if document.manifest.auxiliary_artifacts.is_empty() {
        return Ok(None);
    }
    let mut report = VerificationReport::new();
    let mut paths = VerificationPathCapture::default();
    verify_auxiliary_artifacts(document, options, &mut report, &mut paths);
    auxiliary_artifacts_sha256_from_report(document, &report)
}

fn auxiliary_artifacts_sha256_from_report(
    document: &ManifestDocument,
    report: &VerificationReport,
) -> Result<Option<String>, ManifestError> {
    if document.manifest.auxiliary_artifacts.is_empty() {
        return Ok(None);
    }
    if report.auxiliary_artifacts.len() != document.manifest.auxiliary_artifacts.len() {
        return Ok(None);
    }

    let mut entries = Vec::with_capacity(report.auxiliary_artifacts.len());
    for entry in &report.auxiliary_artifacts {
        let state = match entry.state {
            AuxiliaryArtifactState::Verified => {
                let (Some(sha256), Some(size_bytes)) = (entry.sha256.as_ref(), entry.size_bytes)
                else {
                    return Ok(None);
                };
                ("verified", Some(sha256.clone()), Some(size_bytes))
            }
            AuxiliaryArtifactState::OptionalAbsent => ("optional_absent", None, None),
            AuxiliaryArtifactState::MissingRequired => ("missing_required", None, None),
            AuxiliaryArtifactState::Failed => ("failed", entry.sha256.clone(), entry.size_bytes),
        };
        entries.push(AuxiliaryArtifactCacheEntry {
            name: entry.name.clone(),
            path: entry.manifest_path.clone(),
            required: entry.required,
            state: state.0,
            reason_code: entry.reason_code.clone(),
            sha256: state.1,
            size_bytes: state.2,
        });
    }

    let json = serde_json::to_vec(&entries)?;
    Ok(Some(sha256_bytes(&json).sha256))
}

#[derive(Serialize)]
struct AuxiliaryArtifactCacheEntry {
    name: String,
    path: String,
    required: bool,
    state: &'static str,
    reason_code: Option<String>,
    sha256: Option<String>,
    size_bytes: Option<u64>,
}

fn current_calibration_profile_sha256(
    document: &ManifestDocument,
    options: &VerifyOptions,
) -> Result<Option<String>, ManifestError> {
    let Some(profile) = document
        .manifest
        .calibration
        .as_ref()
        .and_then(|calibration| calibration.profile.as_ref())
    else {
        return Ok(None);
    };
    let path = PathBuf::from(&profile.path);
    let mut path_errors = Vec::<ReportIssue>::new();
    let Some(resolved) = resolve_existing_path(
        &path,
        &document.base_dir,
        options,
        &crate::CALIBRATION_PROFILE_PATH_ISSUES,
        &mut path_errors,
    ) else {
        return Ok(None);
    };
    match sha256_file_bounded(
        &resolved.canonical_path,
        profile
            .file_size_bytes
            .min(options.limits.max_calibration_profile_bytes),
        codes::CALIBRATION_PROFILE_TOO_LARGE,
        "calibration profile",
    ) {
        Ok(hash) => Ok(Some(hash.sha256)),
        Err(_) => Ok(None),
    }
}

fn current_encoder_distortion_profile_sha256(
    document: &ManifestDocument,
    options: &VerifyOptions,
) -> Result<Option<String>, ManifestError> {
    let Some(profile) = document
        .manifest
        .encoder_distortion
        .as_ref()
        .and_then(|profile| profile.profile.as_ref())
    else {
        return Ok(None);
    };
    let path = PathBuf::from(&profile.path);
    let mut path_errors = Vec::<ReportIssue>::new();
    let Some(resolved) = resolve_existing_path(
        &path,
        &document.base_dir,
        options,
        &crate::ENCODER_DISTORTION_PROFILE_PATH_ISSUES,
        &mut path_errors,
    ) else {
        return Ok(None);
    };
    match sha256_file_bounded(
        &resolved.canonical_path,
        profile
            .file_size_bytes
            .min(options.limits.max_encoder_distortion_profile_bytes),
        codes::ENCODER_DISTORTION_PROFILE_TOO_LARGE,
        "encoder distortion profile",
    ) {
        Ok(hash) => Ok(Some(hash.sha256)),
        Err(_) => Ok(None),
    }
}

fn verification_reports_needs_migration(conn: &Connection) -> Result<bool, ManifestError> {
    let exists: Option<i64> = conn
        .query_row(
            "SELECT 1
             FROM sqlite_master
             WHERE type = 'table' AND name = 'verification_reports'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(sqlite_err)?;
    if exists.is_none() {
        return Ok(false);
    }

    let mut stmt = conn
        .prepare("PRAGMA table_info(verification_reports)")
        .map_err(sqlite_err)?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(sqlite_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sqlite_err)?;
    let current_required = [
        "report_id",
        "manifest_path",
        "checked_at",
        "ok",
        "manifest_location_sha256",
        "manifest_sha256",
        "options_sha256",
        "artifact_sha256",
        "row_identity_sha256",
        "calibration_profile_sha256",
        "auxiliary_artifacts_sha256",
        "encoder_distortion_profile_sha256",
        "report_json",
    ];
    // A stale manifest_id column carries a NOT NULL constraint the current
    // inserts no longer satisfy, so its presence always forces migration.
    let has_stale_manifest_id = columns.iter().any(|column| column == "manifest_id");
    if !has_stale_manifest_id && has_required_columns(&columns, &current_required) {
        return Ok(false);
    }

    let legacy_required = ["manifest_path", "checked_at", "ok", "report_json"];
    if has_required_columns(&columns, &legacy_required) {
        return Ok(true);
    }

    Err(ManifestError::invalid(format!(
        "unsupported verification_reports schema {:?}; refusing destructive migration",
        columns
    )))
}

fn has_required_columns(columns: &[String], required: &[&str]) -> bool {
    required
        .iter()
        .all(|required| columns.iter().any(|column| column == required))
}

/// Drops `table` when its live column set differs from `expected_columns`.
/// The sqlite registry is rebuildable cache/pointer state, so a stale-schema
/// table from an older build is dropped here and recreated empty by the
/// `CREATE TABLE IF NOT EXISTS` statements that `init` runs immediately
/// afterwards — legacy rows are never migrated. Idempotent: once the table
/// matches the current schema this is a no-op, and an absent table is left
/// for `CREATE TABLE IF NOT EXISTS` to create.
fn drop_registry_table_on_schema_mismatch(
    conn: &Connection,
    table: &str,
    expected_columns: &[&str],
) -> Result<(), ManifestError> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(sqlite_err)?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(sqlite_err)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(sqlite_err)?;
    if columns.is_empty() {
        return Ok(());
    }
    let matches_current =
        columns.len() == expected_columns.len() && has_required_columns(&columns, expected_columns);
    if !matches_current {
        conn.execute_batch(&format!("DROP TABLE {table}"))
            .map_err(sqlite_err)?;
    }
    Ok(())
}

fn sqlite_err(err: rusqlite::Error) -> ManifestError {
    ManifestError::invalid(format!("sqlite error: {err}"))
}
