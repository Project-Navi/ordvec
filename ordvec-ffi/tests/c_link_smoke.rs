use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn temp_path(prefix: &str, ext: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ordvec_ffi_{prefix}_{}_{}.{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        ext
    ));
    path
}

fn write_file(path: &Path, body: &[u8]) {
    std::fs::File::create(path)
        .unwrap()
        .write_all(body)
        .unwrap();
}

fn write_rankquant_fixture(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"TVRQ");
    bytes.push(1);
    bytes.push(2);
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    for _ in 0..4 {
        bytes.extend_from_slice(&[0x00, 0x55, 0xAA, 0xFF]);
    }
    write_file(path, &bytes);
}

fn c_string_literal(path: &Path) -> String {
    let raw = path.to_str().expect("temporary path must be UTF-8");
    let escaped = raw.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ordvec-ffi should live in the workspace root")
        .to_path_buf()
}

fn target_debug_dir() -> PathBuf {
    if let Ok(lib_dir) = std::env::var("ORDVEC_FFI_STATIC_LIB_DIR") {
        let lib_dir = PathBuf::from(lib_dir);
        return if lib_dir.is_absolute() {
            lib_dir
        } else {
            workspace_root().join(lib_dir)
        };
    }

    let target = std::env::var("ORDVEC_FFI_TARGET").ok();
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let target_dir = PathBuf::from(target_dir);
        return match target {
            Some(target) => target_dir.join(target).join("debug"),
            None => target_dir.join("debug"),
        };
    }

    let mut target_dir = workspace_root().join("target");
    if let Some(target) = target {
        target_dir.push(target);
    }
    target_dir.join("debug")
}

fn add_optional_sanitizer(cc: &mut Command) {
    match std::env::var("ORDVEC_FFI_CC_SANITIZER") {
        Ok(value) if value == "address" => {
            cc.arg("-fsanitize=address");
        }
        Ok(value) if value.trim().is_empty() => {}
        Ok(value) => panic!("unsupported ORDVEC_FFI_CC_SANITIZER={value}"),
        Err(_) => {}
    }
}

#[test]
#[cfg(unix)]
fn c_program_links_and_runs_against_static_library() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let include = manifest.join("include");
    let lib = target_debug_dir().join("libordvec_ffi.a");
    assert!(
        lib.exists(),
        "missing {}; run `cargo build -p ordvec-ffi` before the linked C smoke test",
        lib.display()
    );

    let fixture = temp_path("linked_fixture", "tvrq");
    write_rankquant_fixture(&fixture);

    let src = temp_path("linked_smoke", "c");
    let exe = temp_path("linked_smoke", "bin");
    let body = format!(
        r#"#include <stdint.h>
#include "ordvec.h"

int main(void) {{
    ordvec_index_t *idx = 0;
    ordvec_status_t st = ordvec_index_load({fixture}, 0, &idx);
    if (st != ORDVEC_STATUS_OK) return 1;

    ordvec_index_info_t info;
    ordvec_index_info_init(&info);
    if (ordvec_index_info(idx, &info) != ORDVEC_STATUS_OK) {{
        ordvec_index_free(idx);
        return 2;
    }}
    if (info.kind != ORDVEC_INDEX_KIND_RANK_QUANT || info.dim != 16 || info.vector_count != 4) {{
        ordvec_index_free(idx);
        return 3;
    }}

    float q[16] = {{0}};
    ordvec_search_params_t p;
    ordvec_search_params_init(&p);
    p.query = q;
    p.dim = 16;
    p.k = 2;

    ordvec_hit_t hits[2];
    uint64_t returned = 0;
    ordvec_search_stats_t stats;
    ordvec_search_stats_init(&stats);

    st = ordvec_index_search(idx, &p, hits, 2, &returned, &stats);
    ordvec_index_free(idx);
    if (st != ORDVEC_STATUS_OK) return 4;
    if (returned > 2) return 5;
    if (stats.returned_count != returned) return 6;
    return 0;
}}
"#,
        fixture = c_string_literal(&fixture)
    );
    write_file(&src, body.as_bytes());

    let mut cc = Command::new("cc");
    cc.arg("-std=c11")
        .arg("-I")
        .arg(&include)
        .arg(&src)
        .arg(&lib);
    if cfg!(target_os = "linux") {
        cc.args(["-ldl", "-lm", "-lpthread"]);
    } else if cfg!(target_os = "macos") {
        cc.args(["-lm", "-lpthread"]);
    }
    add_optional_sanitizer(&mut cc);
    let compile = cc.arg("-o").arg(&exe).output();
    match compile {
        Ok(output) => {
            assert!(
                output.status.success(),
                "linked C smoke did not compile\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            std::fs::remove_file(&fixture).ok();
            std::fs::remove_file(&src).ok();
            return;
        }
        Err(err) => panic!("failed to spawn C compiler: {err}"),
    }

    let run = Command::new(&exe)
        .status()
        .expect("linked C smoke failed to run");
    std::fs::remove_file(&fixture).ok();
    std::fs::remove_file(&src).ok();
    std::fs::remove_file(&exe).ok();
    assert!(run.success(), "linked C smoke exited with {run}");
}
