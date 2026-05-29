use clap::{Parser, Subcommand};
use ordvec_manifest::{
    create_manifest_for_index_with_options, load_manifest_file, sha256_file, verify_manifest,
    write_manifest_file, CreateManifestOptions, CreateRowIdentity, ManifestDocument, ManifestError,
    NullModelSpec, ProfileParameterization, VerifyOptions,
};
use serde_json::json;
use std::fs;
use std::path::PathBuf;

const EXIT_VERIFICATION_FAILED: i32 = 1;
const EXIT_USAGE_OR_CONFIG: i32 = 2;

#[derive(Parser)]
#[command(name = "ordvec-manifest")]
#[command(about = "Verify ordvec index manifests", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Hash {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Inspect {
        manifest: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Verify {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        index: Option<PathBuf>,
        #[arg(long)]
        allow_absolute_paths: bool,
        #[arg(long)]
        allow_path_escape: bool,
        #[arg(long)]
        allow_duplicate_db_ids: bool,
        #[arg(long)]
        json: bool,
    },
    Create {
        #[arg(long)]
        index: PathBuf,
        #[arg(long)]
        row_map: Option<PathBuf>,
        #[arg(long)]
        row_id_is_identity: bool,
        #[arg(long)]
        embedding_model: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        allow_absolute_paths: bool,
        #[arg(long)]
        allow_path_escape: bool,
    },
    #[cfg(feature = "sqlite")]
    Sqlite {
        #[command(subcommand)]
        command: SqliteCommands,
    },
}

#[cfg(feature = "sqlite")]
#[derive(Subcommand)]
enum SqliteCommands {
    Verify {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        use_cache: bool,
        #[arg(long)]
        index: Option<PathBuf>,
        #[arg(long)]
        allow_absolute_paths: bool,
        #[arg(long)]
        allow_path_escape: bool,
        #[arg(long)]
        allow_duplicate_db_ids: bool,
        #[arg(long)]
        json: bool,
    },
    Activate {
        #[arg(long)]
        db: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        index: Option<PathBuf>,
        #[arg(long)]
        allow_absolute_paths: bool,
        #[arg(long)]
        allow_path_escape: bool,
        #[arg(long)]
        allow_duplicate_db_ids: bool,
        #[arg(long)]
        json: bool,
    },
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
            json: as_json,
        } => {
            let document = load_manifest_file(&manifest)?;
            if as_json {
                print_json(&document.manifest)?;
            } else {
                println!("manifest_id: {}", document.manifest.manifest_id);
                println!("schema_version: {}", document.manifest.schema_version);
                println!("artifact: {}", document.manifest.artifact.path);
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
            json: as_json,
        } => {
            let document = load_manifest_file(&manifest)?;
            let report = verify_manifest(
                &document,
                VerifyOptions {
                    allow_absolute_paths,
                    allow_path_escape,
                    allow_duplicate_db_ids,
                    index_override: index,
                },
            );
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
            embedding_model,
            out,
            allow_absolute_paths,
            allow_path_escape,
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
            let manifest = create_manifest_for_index_with_options(
                &index,
                row_identity,
                embedding_model,
                &out,
                CreateManifestOptions {
                    allow_absolute_paths,
                    allow_path_escape,
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
            json: as_json,
        } => {
            let document = load_manifest_file(&manifest)?;
            let report = ordvec_manifest::sqlite::verify_with_registry(
                &db,
                &document,
                &manifest,
                VerifyOptions {
                    allow_absolute_paths,
                    allow_path_escape,
                    allow_duplicate_db_ids,
                    index_override: index,
                },
                use_cache,
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
            json: as_json,
        } => {
            let document = load_manifest_file(&manifest)?;
            let report = ordvec_manifest::sqlite::activate(
                &db,
                &document,
                &manifest,
                VerifyOptions {
                    allow_absolute_paths,
                    allow_path_escape,
                    allow_duplicate_db_ids,
                    index_override: index,
                },
                force,
            )?;
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
        println!(
            "verified {}",
            report
                .manifest_id
                .as_deref()
                .unwrap_or("<missing manifest_id>")
        );
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
