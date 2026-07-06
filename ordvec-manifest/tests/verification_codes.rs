use ordvec_manifest::codes;

/// Security-relevant code values are load-bearing for downstream consumers
/// (they branch on these strings via the consts). A silent rename must break
/// this test, not downstream security decisions.
#[test]
fn security_relevant_code_values_are_locked() {
    let locked: &[(&str, &str)] = &[
        (codes::ARTIFACT_SHA256_MISMATCH, "artifact_sha256_mismatch"),
        (
            codes::ARTIFACT_FILE_SIZE_MISMATCH,
            "artifact_file_size_mismatch",
        ),
        (
            codes::ARTIFACT_PATH_UNAVAILABLE,
            "artifact_path_unavailable",
        ),
        (
            codes::ARTIFACT_ABSOLUTE_PATH_REJECTED,
            "artifact_absolute_path_rejected",
        ),
        (
            codes::ARTIFACT_PATH_ESCAPE_REJECTED,
            "artifact_path_escape_rejected",
        ),
        (
            codes::AUXILIARY_ARTIFACT_SHA256_MISMATCH,
            "auxiliary_artifact_sha256_mismatch",
        ),
        (
            codes::AUXILIARY_ARTIFACT_FILE_SIZE_MISMATCH,
            "auxiliary_artifact_file_size_mismatch",
        ),
        (
            codes::AUXILIARY_ARTIFACT_MISSING_REQUIRED,
            "auxiliary_artifact_missing_required",
        ),
        (
            codes::AUXILIARY_ARTIFACT_ABSOLUTE_PATH_REJECTED,
            "auxiliary_artifact_absolute_path_rejected",
        ),
        (
            codes::AUXILIARY_ARTIFACT_PATH_ESCAPE_REJECTED,
            "auxiliary_artifact_path_escape_rejected",
        ),
        (
            codes::ROW_IDENTITY_SHA256_MISMATCH,
            "row_identity_sha256_mismatch",
        ),
        (
            codes::ROW_IDENTITY_ROW_COUNT_MISMATCH,
            "row_identity_row_count_mismatch",
        ),
        (
            codes::SCHEMA_VERSION_UNSUPPORTED,
            "schema_version_unsupported",
        ),
        (codes::MANIFEST_FILE_TOO_LARGE, "manifest_file_too_large"),
        (codes::ARTIFACT_FILE_TOO_LARGE, "artifact_file_too_large"),
        (
            codes::AUXILIARY_ARTIFACT_FILE_TOO_LARGE,
            "auxiliary_artifact_file_too_large",
        ),
        (
            codes::AUXILIARY_ARTIFACT_COUNT_LIMIT_EXCEEDED,
            "auxiliary_artifact_count_limit_exceeded",
        ),
        (
            codes::CALIBRATION_PROFILE_TOO_LARGE,
            "calibration_profile_too_large",
        ),
        (
            codes::ENCODER_DISTORTION_PROFILE_TOO_LARGE,
            "encoder_distortion_profile_too_large",
        ),
        (
            codes::ROW_IDENTITY_LINE_TOO_LARGE,
            "row_identity_line_too_large",
        ),
        (
            codes::ROW_IDENTITY_ROW_COUNT_LIMIT_EXCEEDED,
            "row_identity_row_count_limit_exceeded",
        ),
        (
            codes::ROW_IDENTITY_DUPLICATE_TRACKING_LIMIT_EXCEEDED,
            "row_identity_duplicate_tracking_limit_exceeded",
        ),
        (
            codes::VERIFICATION_REPORT_ISSUE_LIMIT_EXCEEDED,
            "verification_report_issue_limit_exceeded",
        ),
    ];
    for (actual, expected) in locked {
        assert_eq!(actual, expected);
    }
}

const LIB_RS: &str = include_str!("../src/lib.rs");

/// Emit sites must reference `codes::` constants, never bare string literals:
/// scans src/lib.rs for issue-emitting calls whose code argument is a literal.
#[test]
fn emit_sites_reference_code_consts_not_literals() {
    // (call pattern, zero-based index of the code argument)
    let emitters: &[(&str, usize)] = &[
        (".error(", 0),
        ("push_report_issue_bounded(", 2),
        ("mark_auxiliary_artifact_failed(", 1),
        ("ReportIssue::new(", 0),
    ];
    let mut violations = Vec::new();
    for (pattern, code_arg) in emitters {
        for line in literal_code_arg_lines(LIB_RS, pattern, *code_arg) {
            violations.push(format!("src/lib.rs:{line}: {pattern}\"...\""));
        }
    }
    assert!(
        violations.is_empty(),
        "bare string literals at emit sites (use ordvec_manifest::codes consts):\n{}",
        violations.join("\n")
    );
}

/// Returns 1-based line numbers of `pattern` call sites whose argument at
/// `code_arg` (zero-based, at call depth) is a bare string literal.
fn literal_code_arg_lines(src: &str, pattern: &str, code_arg: usize) -> Vec<usize> {
    let mut lines = Vec::new();
    let mut search_from = 0;
    while let Some(found) = src[search_from..].find(pattern) {
        let call_site = search_from + found;
        let args_start = call_site + pattern.len();
        if let Some(arg) = nth_call_arg(&src[args_start..], code_arg) {
            if arg.trim_start().starts_with('"') {
                lines.push(src[..call_site].bytes().filter(|b| *b == b'\n').count() + 1);
            }
        }
        search_from = args_start;
    }
    lines
}

/// Extracts the `n`th comma-separated argument at depth 1 of a call whose
/// opening parenthesis directly precedes `rest`. Tracks string literals and
/// nested brackets so embedded commas do not split arguments.
fn nth_call_arg(rest: &str, n: usize) -> Option<String> {
    let mut depth = 1usize;
    let mut arg_index = 0usize;
    let mut current = String::new();
    let mut in_string = false;
    let mut escaped = false;
    for c in rest.chars() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            current.push(c);
            continue;
        }
        match c {
            '"' => {
                in_string = true;
                current.push(c);
            }
            '(' | '[' | '{' => {
                depth += 1;
                current.push(c);
            }
            ')' | ']' | '}' => {
                depth -= 1;
                if depth == 0 {
                    return (arg_index == n).then_some(current);
                }
                current.push(c);
            }
            ',' if depth == 1 => {
                if arg_index == n {
                    return Some(current);
                }
                arg_index += 1;
                current.clear();
            }
            _ => current.push(c),
        }
    }
    None
}
