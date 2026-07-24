use clap::{Args, Parser, Subcommand};
use ordvec_manifest::{
    create_manifest_for_index_with_options, load_manifest_file_with_options, sha256_file,
    verify_manifest, write_manifest_file, CreateAuxiliaryArtifact, CreateManifestOptions,
    CreateRowIdentity, ManifestDocument, ManifestError, NullModelSpec, ProfileParameterization,
    ResourceLimits, VerifyOptions,
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;

const EXIT_VERIFICATION_FAILED: i32 = 1;
const EXIT_USAGE_OR_CONFIG: i32 = 2;

#[derive(Parser)]
#[command(name = "ordvec-manifest")]
#[command(about = "Create and verify ordvec index manifests", version)]
#[command(after_help = "Run `ordvec-manifest <COMMAND> --help` for command options.")]
struct Cli {
    /// Manifest operation to run.
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute a file's SHA-256 digest and byte length.
    Hash {
        /// File to hash.
        path: PathBuf,
        /// Emit a machine-readable JSON object.
        #[arg(long)]
        json: bool,
    },
    /// Print a manifest summary without verifying its artifacts.
    Inspect {
        /// Manifest JSON to inspect.
        manifest: PathBuf,
        #[command(flatten)]
        limits: LimitArgs,
        /// Emit the parsed manifest as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Verify a manifest and every declared artifact.
    Verify {
        /// Manifest JSON to verify.
        #[arg(long)]
        manifest: PathBuf,
        /// Override the primary index path declared by the manifest.
        #[arg(long)]
        index: Option<PathBuf>,
        /// Permit absolute artifact paths (disabled by default).
        #[arg(long)]
        allow_absolute_paths: bool,
        /// Permit relative paths that escape the manifest directory.
        #[arg(long)]
        allow_path_escape: bool,
        /// Permit duplicate db_id values in a JSONL row map.
        #[arg(long)]
        allow_duplicate_db_ids: bool,
        #[command(flatten)]
        limits: LimitArgs,
        /// Emit the verification report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Create a deterministic manifest for an existing ordvec index.
    Create {
        /// Existing ordvec index to bind.
        #[arg(long)]
        index: PathBuf,
        /// JSONL row-identity map to bind.
        #[arg(long)]
        row_map: Option<PathBuf>,
        /// Declare row IDs as the zero-based index row numbers.
        #[arg(long)]
        row_id_is_identity: bool,
        /// Bind a required caller-owned sidecar as NAME=PATH (repeatable).
        #[arg(long = "aux", value_name = "NAME=PATH", value_parser = parse_auxiliary_artifact_arg)]
        auxiliary_artifacts: Vec<AuxiliaryArtifactArg>,
        /// Bind an optional caller-owned sidecar as NAME=PATH (repeatable).
        #[arg(long = "optional-aux", value_name = "NAME=PATH", value_parser = parse_auxiliary_artifact_arg)]
        optional_auxiliary_artifacts: Vec<AuxiliaryArtifactArg>,
        /// Identifier of the embedding model that produced the index.
        #[arg(long)]
        embedding_model: String,
        /// Destination manifest JSON path.
        #[arg(long)]
        out: PathBuf,
        /// Permit absolute artifact paths in the emitted manifest.
        #[arg(long)]
        allow_absolute_paths: bool,
        /// Permit emitted relative paths to escape the manifest directory.
        #[arg(long)]
        allow_path_escape: bool,
        #[command(flatten)]
        limits: LimitArgs,
    },
    #[cfg(feature = "sqlite")]
    /// Verify or activate manifests with a SQLite report cache.
    Sqlite {
        /// SQLite-backed operation to run.
        #[command(subcommand)]
        command: SqliteCommands,
    },
}

#[derive(Clone, Debug)]
struct AuxiliaryArtifactArg {
    name: String,
    path: PathBuf,
}

fn parse_auxiliary_artifact_arg(value: &str) -> Result<AuxiliaryArtifactArg, String> {
    let (name, path) = value
        .split_once('=')
        .ok_or_else(|| "expected NAME=PATH".to_string())?;
    if name.trim().is_empty() {
        return Err("auxiliary artifact name must be non-empty".to_string());
    }
    if path.trim().is_empty() {
        return Err("auxiliary artifact path must be non-empty".to_string());
    }
    Ok(AuxiliaryArtifactArg {
        name: name.trim().to_string(),
        path: PathBuf::from(path.trim()),
    })
}

#[cfg(test)]
mod tests {
    use super::{parse_auxiliary_artifact_arg, Cli, Commands, LimitArgs};
    use clap::{CommandFactory, Parser};
    use std::path::PathBuf;

    #[test]
    fn auxiliary_artifact_arg_trims_name_and_path() {
        let parsed = parse_auxiliary_artifact_arg(" app.ids = ids.bin ").unwrap();
        assert_eq!(parsed.name, "app.ids");
        assert_eq!(parsed.path, PathBuf::from("ids.bin"));
    }

    #[test]
    fn limit_args_wire_index_artifact_ceiling() {
        let args = LimitArgs {
            max_index_artifact_bytes: Some(42),
            ..LimitArgs::default()
        };
        assert_eq!(args.resource_limits().max_index_artifact_bytes, 42);
        // Unset flag leaves the library default (unbounded) untouched.
        assert_eq!(
            LimitArgs::default()
                .resource_limits()
                .max_index_artifact_bytes,
            ordvec_manifest::ResourceLimits::default().max_index_artifact_bytes
        );
    }

    #[test]
    fn verify_accepts_max_index_artifact_bytes_flag() {
        let cli = Cli::try_parse_from([
            "ordvec-manifest",
            "verify",
            "--manifest",
            "manifest.json",
            "--max-index-artifact-bytes",
            "8",
        ])
        .expect("flag must parse");
        match cli.command {
            Commands::Verify { limits, .. } => {
                assert_eq!(limits.max_index_artifact_bytes, Some(8));
                assert_eq!(limits.resource_limits().max_index_artifact_bytes, 8);
            }
            _ => panic!("expected verify command"),
        }
    }

    #[test]
    fn help_describes_commands_and_safety_relevant_options() {
        let mut root = Cli::command();
        let root_help = root.render_long_help().to_string();
        for expected in [
            "Compute a file's SHA-256 digest",
            "Print a manifest summary",
            "Verify a manifest and every declared artifact",
            "Create a deterministic manifest",
        ] {
            assert!(
                root_help.contains(expected),
                "missing help text: {expected}"
            );
        }

        let mut verify = Cli::command()
            .find_subcommand("verify")
            .expect("verify subcommand")
            .clone();
        let verify_help = verify.render_long_help().to_string();
        for expected in [
            "Manifest JSON to verify",
            "Permit absolute artifact paths",
            "Permit relative paths that escape",
            "Emit the verification report as JSON",
        ] {
            assert!(
                verify_help.contains(expected),
                "missing verify help text: {expected}"
            );
        }
    }
}

#[cfg(feature = "sqlite")]
#[derive(Subcommand)]
enum SqliteCommands {
    /// Verify a manifest, optionally reusing a valid cached report.
    Verify {
        /// SQLite cache database path.
        #[arg(long)]
        db: PathBuf,
        /// Manifest JSON to verify.
        #[arg(long)]
        manifest: PathBuf,
        /// Reuse a matching cached report when available.
        #[arg(long)]
        use_cache: bool,
        /// Override the primary index path declared by the manifest.
        #[arg(long)]
        index: Option<PathBuf>,
        /// Permit absolute artifact paths (disabled by default).
        #[arg(long)]
        allow_absolute_paths: bool,
        /// Permit relative paths that escape the manifest directory.
        #[arg(long)]
        allow_path_escape: bool,
        /// Permit duplicate db_id values in a JSONL row map.
        #[arg(long)]
        allow_duplicate_db_ids: bool,
        #[command(flatten)]
        limits: LimitArgs,
        /// Emit the verification report as JSON.
        #[arg(long)]
        json: bool,
    },
    /// Verify and mark a manifest active in the SQLite cache.
    Activate {
        /// SQLite cache database path.
        #[arg(long)]
        db: PathBuf,
        /// Manifest JSON to verify and activate.
        #[arg(long)]
        manifest: PathBuf,
        /// Activate even when verification reports errors.
        #[arg(long)]
        force: bool,
        /// Override the primary index path declared by the manifest.
        #[arg(long)]
        index: Option<PathBuf>,
        /// Permit absolute artifact paths (disabled by default).
        #[arg(long)]
        allow_absolute_paths: bool,
        /// Permit relative paths that escape the manifest directory.
        #[arg(long)]
        allow_path_escape: bool,
        /// Permit duplicate db_id values in a JSONL row map.
        #[arg(long)]
        allow_duplicate_db_ids: bool,
        #[command(flatten)]
        limits: LimitArgs,
        /// Emit the verification report as JSON.
        #[arg(long)]
        json: bool,
    },
}

#[derive(Args, Clone, Debug, Default)]
struct LimitArgs {
    /// Maximum manifest JSON bytes to read.
    #[arg(long)]
    max_manifest_bytes: Option<u64>,
    /// Maximum bytes in one JSONL row-map line.
    #[arg(long)]
    max_row_map_line_bytes: Option<usize>,
    /// Maximum JSONL row-map rows to inspect.
    #[arg(long)]
    max_row_map_rows: Option<usize>,
    /// Maximum bytes retained while checking duplicate db_id values.
    #[arg(long)]
    max_row_map_tracked_id_bytes: Option<usize>,
    /// Maximum number of declared auxiliary artifacts.
    #[arg(long)]
    max_auxiliary_artifacts: Option<usize>,
    /// Maximum bytes permitted for each auxiliary artifact.
    #[arg(long)]
    max_auxiliary_artifact_bytes: Option<u64>,
    /// Maximum bytes permitted for the primary index artifact.
    #[arg(long)]
    max_index_artifact_bytes: Option<u64>,
    /// Maximum bytes permitted for a calibration profile.
    #[arg(long)]
    max_calibration_profile_bytes: Option<u64>,
    /// Maximum bytes permitted for an encoder-distortion profile.
    #[arg(long)]
    max_encoder_distortion_profile_bytes: Option<u64>,
    /// Maximum detail issues retained in a verification report.
    #[arg(long)]
    max_report_issues: Option<usize>,
    /// Maximum bytes accepted for a cached SQLite report.
    #[arg(long)]
    max_cached_report_bytes: Option<u64>,
}

impl LimitArgs {
    fn resource_limits(&self) -> ResourceLimits {
        let mut limits = ResourceLimits::default();
        if let Some(value) = self.max_manifest_bytes {
            limits.max_manifest_bytes = value;
        }
        if let Some(value) = self.max_row_map_line_bytes {
            limits.max_row_identity_jsonl_line_bytes = value;
        }
        if let Some(value) = self.max_row_map_rows {
            limits.max_row_identity_rows = value;
        }
        if let Some(value) = self.max_row_map_tracked_id_bytes {
            limits.max_row_identity_tracked_db_id_bytes = value;
        }
        if let Some(value) = self.max_auxiliary_artifacts {
            limits.max_auxiliary_artifacts = value;
        }
        if let Some(value) = self.max_auxiliary_artifact_bytes {
            limits.max_auxiliary_artifact_bytes = value;
        }
        if let Some(value) = self.max_index_artifact_bytes {
            limits.max_index_artifact_bytes = value;
        }
        if let Some(value) = self.max_calibration_profile_bytes {
            limits.max_calibration_profile_bytes = value;
        }
        if let Some(value) = self.max_encoder_distortion_profile_bytes {
            limits.max_encoder_distortion_profile_bytes = value;
        }
        if let Some(value) = self.max_report_issues {
            limits.max_report_issues = value;
        }
        if let Some(value) = self.max_cached_report_bytes {
            limits.max_cached_report_bytes = value;
        }
        limits
    }
}

fn main() {
    std::process::exit(match run() {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err}");
            EXIT_USAGE_OR_CONFIG
        }
    });
}

fn run() -> Result<i32, ManifestError> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Hash {
            path,
            json: as_json,
        } => {
            let hash = sha256_file(&path)?;
            if as_json {
                print_json(&json!({
                    "path": path,
                    "sha256": hash.sha256,
                    "size_bytes": hash.size_bytes,
                }))?;
            } else {
                println!("{}  {}", hash.sha256, path.display());
            }
            Ok(0)
        }
        Commands::Inspect {
            manifest,
            limits,
            json: as_json,
        } => {
            let options = VerifyOptions {
                limits: limits.resource_limits(),
                ..VerifyOptions::default()
            };
            let document = load_manifest_file_with_options(&manifest, &options)?;
            if as_json {
                print_json(&document.manifest)?;
            } else {
                println!("schema_version: {}", document.manifest.schema_version);
                println!("artifact: {}", document.manifest.artifact.path);
                println!(
                    "auxiliary_artifacts: {}",
                    document.manifest.auxiliary_artifacts.len()
                );
                println!("row_identity: {}", row_identity_label(&document));
                println!("calibration: {}", calibration_label(&document));
            }
            Ok(0)
        }
        Commands::Verify {
            manifest,
            index,
            allow_absolute_paths,
            allow_path_escape,
            allow_duplicate_db_ids,
            limits,
            json: as_json,
        } => {
            let options = VerifyOptions {
                allow_absolute_paths,
                allow_path_escape,
                allow_duplicate_db_ids,
                index_override: index,
                limits: limits.resource_limits(),
            };
            let document = load_manifest_file_with_options(&manifest, &options)?;
            let report = verify_manifest(&document, options);
            emit_report(&report, as_json)?;
            Ok(if report.ok {
                0
            } else {
                EXIT_VERIFICATION_FAILED
            })
        }
        Commands::Create {
            index,
            row_map,
            row_id_is_identity,
            auxiliary_artifacts,
            optional_auxiliary_artifacts,
            embedding_model,
            out,
            allow_absolute_paths,
            allow_path_escape,
            limits,
        } => {
            let row_identity = match (row_map, row_id_is_identity) {
                (Some(_), true) => {
                    return Err(ManifestError::invalid(
                        "use either --row-map or --row-id-is-identity, not both",
                    ));
                }
                (Some(path), false) => CreateRowIdentity::Jsonl(path),
                (None, true) => CreateRowIdentity::RowIdIdentity,
                (None, false) => {
                    return Err(ManifestError::invalid(
                        "one of --row-map or --row-id-is-identity is required",
                    ));
                }
            };
            if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent)?;
            }
            let auxiliary_artifacts =
                create_auxiliary_options(auxiliary_artifacts, optional_auxiliary_artifacts);
            let manifest = create_manifest_for_index_with_options(
                &index,
                row_identity,
                embedding_model,
                &out,
                CreateManifestOptions {
                    allow_absolute_paths,
                    allow_path_escape,
                    limits: limits.resource_limits(),
                    auxiliary_artifacts,
                },
            )?;
            write_manifest_file(&manifest, &out)?;
            println!("{}", out.display());
            Ok(0)
        }
        #[cfg(feature = "sqlite")]
        Commands::Sqlite { command } => run_sqlite(command),
    }
}

fn create_auxiliary_options(
    required: Vec<AuxiliaryArtifactArg>,
    optional: Vec<AuxiliaryArtifactArg>,
) -> Vec<CreateAuxiliaryArtifact> {
    required
        .into_iter()
        .map(|artifact| CreateAuxiliaryArtifact {
            name: artifact.name,
            path: artifact.path,
            required: true,
        })
        .chain(
            optional
                .into_iter()
                .map(|artifact| CreateAuxiliaryArtifact {
                    name: artifact.name,
                    path: artifact.path,
                    required: false,
                }),
        )
        .collect()
}

#[cfg(feature = "sqlite")]
fn run_sqlite(command: SqliteCommands) -> Result<i32, ManifestError> {
    match command {
        SqliteCommands::Verify {
            db,
            manifest,
            use_cache,
            index,
            allow_absolute_paths,
            allow_path_escape,
            allow_duplicate_db_ids,
            limits,
            json: as_json,
        } => {
            let options = VerifyOptions {
                allow_absolute_paths,
                allow_path_escape,
                allow_duplicate_db_ids,
                index_override: index,
                limits: limits.resource_limits(),
            };
            let document = load_manifest_file_with_options(&manifest, &options)?;
            let report = ordvec_manifest::sqlite::verify_with_registry(
                &db, &document, &manifest, options, use_cache,
            )?;
            emit_report(&report, as_json)?;
            Ok(if report.ok {
                0
            } else {
                EXIT_VERIFICATION_FAILED
            })
        }
        SqliteCommands::Activate {
            db,
            manifest,
            force,
            index,
            allow_absolute_paths,
            allow_path_escape,
            allow_duplicate_db_ids,
            limits,
            json: as_json,
        } => {
            let options = VerifyOptions {
                allow_absolute_paths,
                allow_path_escape,
                allow_duplicate_db_ids,
                index_override: index,
                limits: limits.resource_limits(),
            };
            let document = load_manifest_file_with_options(&manifest, &options)?;
            let report =
                ordvec_manifest::sqlite::activate(&db, &document, &manifest, options, force)?;
            emit_report(&report, as_json)?;
            Ok(if report.ok || force {
                0
            } else {
                EXIT_VERIFICATION_FAILED
            })
        }
    }
}

fn emit_report(
    report: &ordvec_manifest::VerificationReport,
    as_json: bool,
) -> Result<(), ManifestError> {
    if as_json {
        print_json(report)?;
    } else if report.ok {
        println!("verified");
    } else {
        for issue in &report.errors {
            eprintln!("{}: {}", issue.code, issue.message);
        }
    }
    Ok(())
}

fn print_json(value: &impl serde::Serialize) -> Result<(), ManifestError> {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    serde_json::to_writer_pretty(&mut lock, value)?;
    use std::io::Write;
    lock.write_all(b"\n")?;
    Ok(())
}

fn row_identity_label(document: &ManifestDocument) -> &'static str {
    match &document.manifest.row_identity {
        ordvec_manifest::RowIdentity::RowIdIdentity { .. } => "row_id_identity",
        ordvec_manifest::RowIdentity::Jsonl { .. } => "jsonl",
    }
}

fn calibration_label(document: &ManifestDocument) -> String {
    let Some(calibration) = &document.manifest.calibration else {
        return "absent".to_string();
    };
    match &calibration.null_model {
        NullModelSpec::UniformHypergeometric => "uniform_hypergeometric".to_string(),
        NullModelSpec::WeightedMarginalProfile { parameterization } => {
            format!(
                "weighted_marginal_profile / {}",
                profile_parameterization_label(*parameterization)
            )
        }
        NullModelSpec::EmpiricalTailTable { .. } => "empirical_tail_table".to_string(),
        NullModelSpec::CallerDefined { name, .. } => format!("caller_defined / {name}"),
    }
}

fn profile_parameterization_label(parameterization: ProfileParameterization) -> &'static str {
    match parameterization {
        ProfileParameterization::MarginalTopKFrequency => "marginal_topk_frequency",
        ProfileParameterization::BucketFrequency => "bucket_frequency",
        ProfileParameterization::SignFrequency => "sign_frequency",
        ProfileParameterization::RankPositionFrequency => "rank_position_frequency",
        ProfileParameterization::EmpiricalTailTable => "empirical_tail_table",
    }
}
