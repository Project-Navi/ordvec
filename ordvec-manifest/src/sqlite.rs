use crate::{
    resolve_existing_path, sha256_file, verify_manifest, ManifestDocument, ManifestError,
    ReportIssue, RowIdentity, VerificationReport, VerifyOptions,
};
use chrono::{SecondsFormat, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub fn verify_with_registry(
    db_path: impl AsRef<Path>,
    document: &ManifestDocument,
    manifest_path: impl AsRef<Path>,
    options: VerifyOptions,
    use_cache: bool,
) -> Result<VerificationReport, ManifestError> {
    let mut conn = Connection::open(db_path).map_err(sqlite_err)?;
    init(&conn)?;
    if use_cache {
        if let Some(cache_key) = current_cache_key(document, manifest_path.as_ref(), &options)? {
            if let Some(report) =
                load_cached_report(&conn, &document.manifest.manifest_id, &cache_key)?
            {
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
        document,
        manifest_path.as_ref(),
        &report,
        cache_key.as_ref(),
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
    init(&conn)?;
    let store_options = options.clone();
    let report = verify_manifest(document, options);
    let cache_key =
        cache_key_from_report(manifest_path.as_ref(), &report, document, &store_options)?;
    store_report(
        &mut conn,
        document,
        manifest_path.as_ref(),
        &report,
        cache_key.as_ref(),
    )?;
    if !report.ok && !force {
        return Ok(report);
    }

    conn.execute(
        "INSERT INTO active_manifest(id, manifest_id, manifest_path, activated_at, forced)
         VALUES(1, ?1, ?2, ?3, ?4)
         ON CONFLICT(id) DO UPDATE SET
           manifest_id=excluded.manifest_id,
           manifest_path=excluded.manifest_path,
           activated_at=excluded.activated_at,
           forced=excluded.forced",
        params![
            document.manifest.manifest_id,
            manifest_path.as_ref().display().to_string(),
            Utc::now().to_rfc3339_opts(SecondsFormat::Nanos, true),
            i64::from(force),
        ],
    )
    .map_err(sqlite_err)?;
    Ok(report)
}

fn init(conn: &Connection) -> Result<(), ManifestError> {
    if verification_reports_needs_migration(conn)? {
        conn.execute_batch(
            "ALTER TABLE verification_reports RENAME TO verification_reports_old;
             CREATE TABLE verification_reports(
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                manifest_id TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                ok INTEGER NOT NULL,
                manifest_sha256 TEXT,
                options_sha256 TEXT,
                artifact_sha256 TEXT,
                row_identity_sha256 TEXT,
                report_json TEXT NOT NULL
             );
             INSERT INTO verification_reports(
                manifest_id, manifest_path, checked_at, ok, report_json
             )
             SELECT manifest_id, manifest_path, checked_at, ok, report_json
             FROM verification_reports_old;
             DROP TABLE verification_reports_old;",
        )
        .map_err(sqlite_err)?;
    }
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS verification_reports(
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            manifest_id TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            ok INTEGER NOT NULL,
            manifest_sha256 TEXT,
            options_sha256 TEXT,
            artifact_sha256 TEXT,
            row_identity_sha256 TEXT,
            report_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS verification_reports_cache_idx
          ON verification_reports(
            manifest_id,
            manifest_sha256,
            options_sha256,
            artifact_sha256,
            row_identity_sha256,
            report_id
          );
        CREATE TABLE IF NOT EXISTS active_manifest(
            id INTEGER PRIMARY KEY CHECK(id = 1),
            manifest_id TEXT NOT NULL,
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
    document: &ManifestDocument,
    manifest_path: &Path,
    report: &VerificationReport,
    cache_key: Option<&CacheKey>,
) -> Result<(), ManifestError> {
    let tx = conn.transaction().map_err(sqlite_err)?;
    let report_json = serde_json::to_string(report)?;
    tx.execute(
        "INSERT INTO verification_reports(
            manifest_id,
            manifest_path,
            checked_at,
            ok,
            manifest_sha256,
            options_sha256,
            artifact_sha256,
            row_identity_sha256,
            report_json
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            document.manifest.manifest_id,
            manifest_path.display().to_string(),
            report.checked_at,
            i64::from(report.ok),
            cache_key.map(|key| key.manifest_sha256.as_str()),
            cache_key.map(|key| key.options_sha256.as_str()),
            cache_key.map(|key| key.artifact_sha256.as_str()),
            cache_key.and_then(|key| key.row_identity_sha256.as_deref()),
            report_json,
        ],
    )
    .map_err(sqlite_err)?;
    tx.commit().map_err(sqlite_err)?;
    Ok(())
}

fn load_cached_report(
    conn: &Connection,
    manifest_id: &str,
    cache_key: &CacheKey,
) -> Result<Option<VerificationReport>, ManifestError> {
    let report_json: Option<String> = conn
        .query_row(
            "SELECT report_json
             FROM verification_reports
             WHERE manifest_id = ?1
               AND manifest_sha256 = ?2
               AND options_sha256 = ?3
               AND artifact_sha256 = ?4
               AND (
                 (row_identity_sha256 IS NULL AND ?5 IS NULL)
                 OR row_identity_sha256 = ?5
               )
             ORDER BY report_id DESC
             LIMIT 1",
            params![
                manifest_id,
                cache_key.manifest_sha256.as_str(),
                cache_key.options_sha256.as_str(),
                cache_key.artifact_sha256.as_str(),
                cache_key.row_identity_sha256.as_deref(),
            ],
            |row| row.get(0),
        )
        .optional()
        .map_err(sqlite_err)?;
    report_json
        .map(|json| serde_json::from_str(&json).map_err(ManifestError::from))
        .transpose()
}

#[derive(Clone, Debug)]
struct CacheKey {
    manifest_sha256: String,
    options_sha256: String,
    artifact_sha256: String,
    row_identity_sha256: Option<String>,
}

#[derive(Serialize)]
struct CacheableVerifyOptions {
    allow_absolute_paths: bool,
    allow_path_escape: bool,
    allow_duplicate_db_ids: bool,
    index_override: Option<String>,
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
        }
    }
}

fn current_cache_key(
    document: &ManifestDocument,
    manifest_path: &Path,
    options: &VerifyOptions,
) -> Result<Option<CacheKey>, ManifestError> {
    let manifest_sha256 = match sha256_file(manifest_path) {
        Ok(hash) => hash.sha256,
        Err(_) => return Ok(None),
    };
    let options_json = serde_json::to_vec(&CacheableVerifyOptions::from_options(options))?;
    let options_sha256 = sha256_bytes(&options_json);

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
        "artifact",
        &mut path_errors,
    ) else {
        return Ok(None);
    };
    let artifact_sha256 = match sha256_file(&artifact.resolved_path) {
        Ok(hash) => hash.sha256,
        Err(_) => return Ok(None),
    };

    let row_identity_sha256 = match &document.manifest.row_identity {
        RowIdentity::RowIdIdentity { .. } => None,
        RowIdentity::Jsonl { path, .. } => {
            let row_path = PathBuf::from(path);
            let Some(row_identity) = resolve_existing_path(
                &row_path,
                &document.base_dir,
                options,
                "row_identity",
                &mut path_errors,
            ) else {
                return Ok(None);
            };
            match sha256_file(&row_identity.resolved_path) {
                Ok(hash) => Some(hash.sha256),
                Err(_) => return Ok(None),
            }
        }
    };

    Ok(Some(CacheKey {
        manifest_sha256,
        options_sha256,
        artifact_sha256,
        row_identity_sha256,
    }))
}

fn cache_key_from_report(
    manifest_path: &Path,
    report: &VerificationReport,
    document: &ManifestDocument,
    options: &VerifyOptions,
) -> Result<Option<CacheKey>, ManifestError> {
    let manifest_sha256 = match sha256_file(manifest_path) {
        Ok(hash) => hash.sha256,
        Err(_) => return Ok(None),
    };
    let options_json = serde_json::to_vec(&CacheableVerifyOptions::from_options(options))?;
    let options_sha256 = sha256_bytes(&options_json);
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
    Ok(Some(CacheKey {
        manifest_sha256,
        options_sha256,
        artifact_sha256,
        row_identity_sha256,
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
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
    Ok(!columns.iter().any(|column| column == "report_id")
        || !columns.iter().any(|column| column == "manifest_sha256"))
}

fn sqlite_err(err: rusqlite::Error) -> ManifestError {
    ManifestError::invalid(format!("sqlite error: {err}"))
}
