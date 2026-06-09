#!/usr/bin/env python3
"""Structural release publish invariants for registry upload jobs."""

from __future__ import annotations

import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None  # type: ignore[assignment]


WORKFLOW_PATH = os.environ.get("RELEASE_WORKFLOW_PATH", ".github/workflows/release.yml")
PYTHON_WORKFLOW_PATH = os.environ.get("PYTHON_WORKFLOW_PATH", ".github/workflows/python.yml")
CI_WORKFLOW_PATH = os.environ.get("CI_WORKFLOW_PATH", ".github/workflows/ci.yml")
STRICT_STABLE_TAG_PATTERN = r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
COVERAGE_WORKFLOW_PATH = os.environ.get("COVERAGE_WORKFLOW_PATH", ".github/workflows/coverage.yml")
SDE_ACTION_PATH = os.environ.get(
    "SDE_ACTION_PATH", ".github/actions/setup-intel-sde/action.yml"
)


def fail(message: str) -> None:
    print(f"::error::release-publish invariant violated: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_workflow(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        yq = shutil.which("yq")
        if yq is None:
            fail("PyYAML or yq is required to parse .github/workflows/release.yml")
        try:
            raw = subprocess.check_output([yq, "-o=json", ".", path], text=True)
        except subprocess.CalledProcessError as exc:
            fail(f"{path}: yq could not parse workflow YAML: {exc}")
        try:
            workflow = json.loads(raw)
        except json.JSONDecodeError as exc:
            fail(f"{path}: yq emitted invalid JSON: {exc}")
    else:
        try:
            with open(path, encoding="utf-8") as fh:
                workflow = yaml.safe_load(fh)
        except OSError as exc:
            fail(f"{path}: could not read workflow: {exc}")
        except Exception as exc:  # PyYAML exposes several parser exception types.
            fail(f"{path}: could not parse workflow YAML: {exc}")

    if not isinstance(workflow, dict):
        fail(f"{path}: workflow root must be a mapping")
    return workflow


def mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{context} must be a mapping")
    return value


def sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        fail(f"{context} must be a sequence")
    return value


def action_name(step: dict[str, Any]) -> str | None:
    uses = step.get("uses")
    if not isinstance(uses, str):
        return None
    return uses.split("@", 1)[0].lower()


def norm_path(value: Any) -> str:
    if value is None:
        return ""
    path = str(value).strip().replace("\\", "/")
    if not path:
        return ""
    normalized = posixpath.normpath(path)
    return "" if normalized == "." else normalized.rstrip("/")


def boolish_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() == "true")


def step_label(index: int, step: dict[str, Any]) -> str:
    name = step.get("name")
    if isinstance(name, str) and name:
        return f"step {index + 1} ({name!r})"
    return f"step {index + 1}"


def empty(value: Any) -> bool:
    return value is None or value == ""


def has_need(job: dict[str, Any], needed: str) -> bool:
    needs = job.get("needs")
    if isinstance(needs, str):
        return needs == needed
    if isinstance(needs, list):
        return needed in needs
    return False


def contains_text(value: Any, needle: str) -> bool:
    return isinstance(value, str) and needle in value


def contains_nested_text(value: Any, needle: str) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, dict):
        return any(contains_nested_text(inner, needle) for inner in value.values())
    if isinstance(value, list):
        return any(contains_nested_text(inner, needle) for inner in value)
    return False


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        fail(f"{path}: could not read workflow: {exc}")


def strip_toml_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {'"', "'"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line


def split_inline_table(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    bracket_depth = 0
    brace_depth = 0
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {'"', "'"}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "[" and quote is None:
            bracket_depth += 1
            continue
        if char == "]" and quote is None and bracket_depth > 0:
            bracket_depth -= 1
            continue
        if char == "{" and quote is None:
            brace_depth += 1
            continue
        if char == "}" and quote is None and brace_depth > 0:
            brace_depth -= 1
            continue
        if char == "," and quote is None and bracket_depth == 0 and brace_depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    parts.append(value[start:].strip())
    return [part for part in parts if part]


def parse_toml_value(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d+\.\d+", value):
        return float(value)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [parse_toml_value(part) for part in split_inline_table(inner)]
    if value.startswith("{") and value.endswith("}"):
        parsed: dict[str, Any] = {}
        for part in split_inline_table(value[1:-1]):
            key, separator, inner = part.partition("=")
            if not separator:
                raise ValueError(f"unsupported inline table entry {part!r}")
            parsed[key.strip()] = parse_toml_value(inner)
        return parsed
    raise ValueError(f"unsupported TOML value {value!r}")


def minimal_load_toml(path: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current: dict[str, Any] = data
    multiline_array: list[Any] | None = None
    for lineno, raw_line in enumerate(read_text(path).splitlines(), start=1):
        line = strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if multiline_array is not None:
            closes = line == "]" or (line.endswith("]") and line.count("]") > line.count("["))
            if closes:
                line = line[:-1].strip()
            if line.endswith(","):
                line = line[:-1].strip()
            if line:
                for part in split_inline_table(line):
                    multiline_array.append(parse_toml_value(part))
            if closes:
                multiline_array = None
            continue
        if line.startswith("[[") and line.endswith("]]"):
            current = {}
            continue
        if line.startswith("[") and line.endswith("]"):
            current = data
            for part in line[1:-1].split("."):
                current = current.setdefault(part.strip(), {})
                if not isinstance(current, dict):
                    raise ValueError(f"{path}:{lineno}: section conflicts with scalar value")
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError(f"{path}:{lineno}: unsupported TOML line {line!r}")
        if value.strip() == "[":
            multiline_array = []
            current[key.strip()] = multiline_array
            continue
        current[key.strip()] = parse_toml_value(value)
    if multiline_array is not None:
        raise ValueError(f"{path}: unterminated multiline array")
    return data


def read_toml_string_in_section(path: str, section: str, key: str) -> str:
    current_section: str | None = None
    for lineno, raw_line in enumerate(read_text(path).splitlines(), start=1):
        line = strip_toml_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            if line.startswith("[["):
                current_section = None
            else:
                current_section = line[1:-1].strip()
            continue
        if current_section != section:
            continue
        raw_key, separator, value = line.partition("=")
        if not separator or raw_key.strip() != key:
            continue
        parsed = parse_toml_value(value)
        if not isinstance(parsed, str):
            raise ValueError(f"{path}:{lineno}: {section}.{key} must be a string")
        return parsed
    raise ValueError(f"{path}: missing {section}.{key}")


def load_toml(path: str) -> dict[str, Any]:
    try:
        if tomllib is None:
            data = minimal_load_toml(path)
        else:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
    except OSError as exc:
        fail(f"{path}: could not read TOML: {exc}")
    except (tomllib.TOMLDecodeError if tomllib is not None else ValueError) as exc:
        fail(f"{path}: could not parse TOML: {exc}")
    if not isinstance(data, dict):
        fail(f"{path}: TOML root must be a mapping")
    return data


def package_manifest(path: str) -> dict[str, Any]:
    data = load_toml(path)
    package = mapping(data.get("package"), f"{path}: package")
    return package


def package_version(path: str) -> str:
    package = package_manifest(path)
    version = package.get("version")
    if not isinstance(version, str) or not version:
        fail(f"{path}: package.version must be a non-empty string")
    return version


def package_rust_version(path: str) -> str:
    package = package_manifest(path)
    rust_version = package.get("rust-version")
    if not isinstance(rust_version, str) or not rust_version:
        fail(f"{path}: package.rust-version must be a non-empty string")
    return rust_version


def package_publish_setting(path: str) -> bool:
    package = package_manifest(path)
    publish = package.get("publish", True)
    if not isinstance(publish, bool):
        fail(f"{path}: package.publish must be a boolean when present")
    return publish


def project_version(path: str) -> str:
    data = load_toml(path)
    project = mapping(data.get("project"), f"{path}: project")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        fail(f"{path}: project.version must be a non-empty string")
    return version


def python_init_version(path: str) -> str:
    text = read_text(path)
    matches = re.findall(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$", text, re.MULTILINE)
    if len(matches) != 1:
        fail(f"{path}: must contain exactly one literal __version__ assignment")
    return matches[0]


def semver_minor_requirement(version: str) -> str:
    match = re.fullmatch(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)", version)
    if match is None:
        fail(f"package.version {version!r} is not a strict MAJOR.MINOR.PATCH SemVer")
    return f"{match.group(1)}.{match.group(2)}"


def check_release_version_sync() -> None:
    core_version = package_version("Cargo.toml")
    expected = {
        "ordvec-python/Cargo.toml package.version": package_version("ordvec-python/Cargo.toml"),
        "ordvec-python/pyproject.toml project.version": project_version(
            "ordvec-python/pyproject.toml"
        ),
        "ordvec-python/python/ordvec/__init__.py __version__": python_init_version(
            "ordvec-python/python/ordvec/__init__.py"
        ),
        "ordvec-manifest/Cargo.toml package.version": package_version("ordvec-manifest/Cargo.toml"),
        "ordvec-ffi/Cargo.toml package.version": package_version("ordvec-ffi/Cargo.toml"),
    }
    for label, version in expected.items():
        if version != core_version:
            fail(f"{label} is {version}, expected lockstep version {core_version}")

    manifest = load_toml("ordvec-manifest/Cargo.toml")
    dependencies = mapping(manifest.get("dependencies"), "ordvec-manifest/Cargo.toml: dependencies")
    ordvec_dep = mapping(
        dependencies.get("ordvec"), "ordvec-manifest/Cargo.toml: dependencies.ordvec"
    )
    dep_version = ordvec_dep.get("version")
    if dep_version != core_version:
        fail(
            "ordvec-manifest/Cargo.toml: dependencies.ordvec.version "
            f"is {dep_version!r}, expected {core_version!r}"
        )

    changelog = read_text("CHANGELOG.md")
    if not re.search(rf"^## \[?{re.escape(core_version)}\]? - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.MULTILINE):
        fail(f"CHANGELOG.md must contain a dated section for {core_version}")

    threat_model = read_text("THREAT_MODEL.md")
    if not re.search(
        rf"^\>\s+\*\*Status:\*\*\s+v{re.escape(core_version)}\s+\(pre-1\.0\),",
        threat_model,
        re.MULTILINE,
    ):
        fail(f"THREAT_MODEL.md status must mention v{core_version}")

    fuzz_lock = read_text("fuzz/Cargo.lock")
    if not re.search(
        rf'(?ms)^\[\[package\]\]\nname = "ordvec"\nversion = "{re.escape(core_version)}"\n',
        fuzz_lock,
    ):
        fail(f"fuzz/Cargo.lock must lock the path dependency ordvec at {core_version}")


def check_release_compatibility_sync() -> None:
    core_version = package_version("Cargo.toml")
    core_msrv = package_rust_version("Cargo.toml")

    for path in (
        "ordvec-manifest/Cargo.toml",
        "ordvec-python/Cargo.toml",
        "ordvec-ffi/Cargo.toml",
    ):
        rust_version = package_rust_version(path)
        if rust_version != core_msrv:
            fail(f"{path}: package.rust-version is {rust_version}, expected {core_msrv}")

    readme = read_text("README.md")
    minor_req = semver_minor_requirement(core_version)
    quickstart = re.search(r"(?ms)^## Quickstart\b.*?```toml\n(?P<block>.*?)\n```", readme)
    if quickstart is None:
        fail("README.md must contain a Quickstart TOML dependency block")
    if f'ordvec = "{minor_req}"' not in quickstart.group("block"):
        fail(f"README.md quickstart must install ordvec = {minor_req!r}")
    if f"MSRV-{core_msrv}-blue.svg" not in readme:
        fail(f"README.md MSRV badge must mention {core_msrv}")
    if f"ordvec's MSRV is **Rust {core_msrv}**" not in readme:
        fail(f"README.md MSRV section must mention Rust {core_msrv}")

    compatibility = read_text("docs/compatibility-policy.md")
    if f"The Rust MSRV is Rust {core_msrv}." not in compatibility:
        fail(f"docs/compatibility-policy.md must mention Rust {core_msrv}")

    ci = read_text(".github/workflows/ci.yml")
    msrv_toolchain = f"{core_msrv}.0"
    if f"name: msrv ({msrv_toolchain})" not in ci:
        fail(f".github/workflows/ci.yml MSRV job name must mention {msrv_toolchain}")
    if f"toolchain: {msrv_toolchain}" not in ci:
        fail(f".github/workflows/ci.yml MSRV job must pin toolchain {msrv_toolchain}")


def check_publication_model() -> None:
    expected_publish = {
        "Cargo.toml": True,
        "ordvec-manifest/Cargo.toml": True,
        "ordvec-python/Cargo.toml": False,
        "ordvec-ffi/Cargo.toml": False,
        "fuzz/Cargo.toml": False,
    }
    for path, expected in expected_publish.items():
        actual = package_publish_setting(path)
        if actual != expected:
            wanted = "publishable" if expected else "publish = false"
            got = "publishable" if actual else "publish = false"
            fail(f"{path}: publication model is {got}, expected {wanted}")


def check_python_package_metadata() -> None:
    pyproject = load_toml("ordvec-python/pyproject.toml")
    project = mapping(pyproject.get("project"), "ordvec-python/pyproject.toml: project")
    if project.get("name") != "ordvec":
        fail("ordvec-python/pyproject.toml: project.name must be 'ordvec'")
    if project.get("requires-python") != ">=3.10":
        fail("ordvec-python/pyproject.toml: project.requires-python must be >=3.10")
    dependencies = sequence(
        project.get("dependencies"), "ordvec-python/pyproject.toml: project.dependencies"
    )
    if "numpy>=2.2" not in dependencies:
        fail("ordvec-python/pyproject.toml: project.dependencies must include numpy>=2.2")

    cargo = load_toml("ordvec-python/Cargo.toml")
    dependencies_table = mapping(cargo.get("dependencies"), "ordvec-python/Cargo.toml: dependencies")
    pyo3 = mapping(dependencies_table.get("pyo3"), "ordvec-python/Cargo.toml: dependencies.pyo3")
    pyo3_features = sequence(
        pyo3.get("features"), "ordvec-python/Cargo.toml: dependencies.pyo3.features"
    )
    for feature in ("extension-module", "abi3-py310"):
        if feature not in pyo3_features:
            fail(f"ordvec-python/Cargo.toml: pyo3 features must include {feature}")

    readme = read_text("README.md")
    py_readme = read_text("ordvec-python/README.md")
    for path, text in (("README.md", readme), ("ordvec-python/README.md", py_readme)):
        if "CPython 3.10+ (abi3)" not in text:
            fail(f"{path}: Python install docs must mention CPython 3.10+ (abi3)")
        if "`numpy>=2.2`" not in text:
            fail(f"{path}: Python install docs must mention numpy>=2.2")

    dependabot = read_text(".github/dependabot.yml")
    if "floor >=2.2" not in dependabot:
        fail(".github/dependabot.yml must keep the Python NumPy floor comment at >=2.2")


def check_strict_release_tag_patterns(workflow: dict[str, Any], path: str) -> None:
    try:
        tag_pattern = read_toml_string_in_section("cliff.toml", "git", "tag_pattern")
    except ValueError as exc:
        fail(str(exc))
    if tag_pattern != STRICT_STABLE_TAG_PATTERN:
        fail(
            "cliff.toml: git.tag_pattern must match release.yml's strict stable "
            "SemVer guard"
        )

    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    guard = mapping(jobs.get("guard"), f"{path}: jobs.guard")
    steps = sequence(guard.get("steps"), f"{path}: jobs.guard.steps")
    semver_runs: list[str] = []
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.guard.steps[{index}]")
        if step.get("id") != "semver":
            continue
        run = step.get("run")
        if not isinstance(run, str):
            fail(f"{path}: jobs.guard semver step must be a run step")
        semver_runs.append(run)
    if len(semver_runs) != 1:
        fail(f"{path}: jobs.guard must contain exactly one id: semver step")
    assignment = f"semver='{STRICT_STABLE_TAG_PATTERN}'"
    if assignment not in shell_logical_lines(semver_runs[0]):
        fail(f"{path}: jobs.guard semver step must execute the strict stable SemVer regex")

    compiled = re.compile(STRICT_STABLE_TAG_PATTERN)
    accepted = ("v0.4.0", "v1.2.3", "v10.20.30")
    rejected = ("v01.2.3", "v1.02.3", "v1.2.03", "v1.2.3-rc.1", "archive/v1.2.3")
    for tag in accepted:
        if compiled.fullmatch(tag) is None:
            fail(f"strict release tag regex must accept {tag}")
    for tag in rejected:
        if compiled.fullmatch(tag) is not None:
            fail(f"strict release tag regex must reject {tag}")


def shell_vars(name: str) -> set[str]:
    return {f"${name}", f"${{{name}}}"}


def shell_logical_lines(script: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        continued = line.endswith("\\")
        if continued:
            line = line[:-1].strip()
        current = f"{current} {line}".strip() if current else line
        if not continued:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines


def shell_curl_commands(script: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in shell_logical_lines(script):
        if "curl" not in line:
            continue
        if line.startswith("if "):
            line = line[3:].strip()
        if line.startswith("! "):
            line = line[2:].strip()
        line = line.split("; then", 1)[0].strip().rstrip(";")
        try:
            words = shlex.split(line)
        except ValueError:
            continue
        if words and words[0] == "curl":
            commands.append(words)
    return commands


def has_cargo_package_arg(words: list[str], package: str) -> bool:
    for index, word in enumerate(words):
        if word in {"-p", "--package"}:
            if index + 1 < len(words) and words[index + 1] == package:
                return True
        elif word.startswith("--package=") and word.split("=", 1)[1] == package:
            return True
        elif word.startswith("-p") and word != "-p" and word[2:] == package:
            return True
    return False


def has_cargo_command(run: str, subcommand: str, package: str) -> bool:
    for line in shell_logical_lines(run):
        if line.startswith("if "):
            line = line[3:].strip()
        line = line.split("; then", 1)[0].strip().rstrip(";")
        try:
            words = shlex.split(line)
        except ValueError:
            continue
        if len(words) < 3 or words[0] != "cargo" or words[1] != subcommand:
            continue
        if "--locked" in words and has_cargo_package_arg(words[2:], package):
            return True
    return False


def has_shell_arg(words: list[str], values: set[str]) -> bool:
    return any(word in values for word in words)


def has_shell_option_value(words: list[str], options: set[str], values: set[str]) -> bool:
    for index, word in enumerate(words):
        for option in options:
            if word == option and index + 1 < len(words) and words[index + 1] in values:
                return True
            if word.startswith(f"{option}=") and word.split("=", 1)[1] in values:
                return True
            if len(option) == 2 and word.startswith(option) and word != option and word[2:] in values:
                return True
    return False


def has_assignment(run: str, name: str, url_pattern: str) -> bool:
    quoted = rf"{name}=([\"']){url_pattern}\1"
    unquoted = rf"{name}={url_pattern}"
    return any(
        re.fullmatch(rf"(?:{quoted}|{unquoted})", line.strip()) for line in run.splitlines()
    )


def readback_curl_uses(words: list[str], url_var: str) -> bool:
    return (
        has_shell_arg(words, shell_vars(url_var))
        and has_shell_option_value(words, {"--user-agent", "-A"}, shell_vars("CRATES_IO_USER_AGENT"))
        and has_shell_option_value(words, {"--output", "-o"}, shell_vars("PUBLISHED"))
    )


def check_hash_requirement_temp_paths(paths: list[str]) -> None:
    for path in paths:
        workflow_text = read_text(path)
        if "/tmp/ordvec-" in workflow_text:
            fail(f"{path}: hash requirement files must be written under ${{RUNNER_TEMP}}, not /tmp")


def trigger_names(on_value: Any) -> set[str]:
    if isinstance(on_value, str):
        return {on_value}
    if isinstance(on_value, list):
        return {item for item in on_value if isinstance(item, str)}
    if isinstance(on_value, dict):
        return {key for key in on_value if isinstance(key, str)}
    return set()


def check_release_security_gates(workflow: dict[str, Any], path: str) -> None:
    blocked_triggers = {"pull_request_target", "workflow_run"}
    on_value = workflow.get("on", workflow.get(True))
    blocked = trigger_names(on_value) & blocked_triggers
    if blocked:
        fail(
            f"{path}: release workflow must not use trusted-publishing-blocked triggers: "
            f"{', '.join(sorted(blocked))}"
        )

    top_permissions = workflow.get("permissions")
    if top_permissions is not None and not isinstance(top_permissions, dict):
        fail(f"{path}: workflow permissions must be an explicit mapping, not {top_permissions!r}")
    if isinstance(top_permissions, dict) and top_permissions.get("id-token") == "write":
        fail(f"{path}: id-token: write must be scoped to explicit signing/publishing jobs, not workflow-wide")

    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    require_job = mapping(jobs.get("require-ci-green"), f"{path}: jobs.require-ci-green")
    steps = sequence(require_job.get("steps"), f"{path}: jobs.require-ci-green.steps")
    gated_workflows = ("ci.yml", "python.yml", "fuzz.yml", "codeql.yml", "actionlint.yml", "zizmor.yml")
    found_loop: tuple[str, ...] | None = None
    found_gate_run: str | None = None
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.require-ci-green.steps[{index}]")
        run = step.get("run")
        if not isinstance(run, str):
            continue
        match = re.search(r"(?m)^\s*for\s+wf\s+in\s+(.+?);\s+do\s*$", run)
        if match:
            found_loop = tuple(shlex.split(match.group(1)))
            found_gate_run = run
            break
    if found_loop is None:
        fail(f"{path}: require-ci-green must loop over the release-gated workflow filenames")
    if found_loop != gated_workflows:
        fail(
            f"{path}: require-ci-green gates {found_loop!r}; expected {gated_workflows!r}"
        )
    if found_gate_run is None or "event=push" not in found_gate_run or '.event == "push"' not in found_gate_run:
        fail(f"{path}: require-ci-green must require successful push workflow runs")

    allowed_id_token_jobs = {
        "attest",
        "provenance",
        "publish-crate",
        "attest-manifest",
        "manifest-provenance",
        "publish-manifest-crate",
        "publish-pypi",
    }
    for job_name, raw_job in jobs.items():
        if not isinstance(job_name, str):
            continue
        job = mapping(raw_job, f"{path}: jobs.{job_name}")
        permissions = job.get("permissions")
        if permissions is not None and not isinstance(permissions, dict):
            fail(f"{path}: jobs.{job_name}.permissions must be an explicit mapping, not {permissions!r}")
        if not isinstance(permissions, dict):
            continue
        id_token = permissions.get("id-token")
        if id_token == "write" and job_name not in allowed_id_token_jobs:
            fail(
                f"{path}: jobs.{job_name} grants id-token: write but is not an allowed "
                "release signing/publishing job"
            )

    for job_name, environment in (
        ("publish-crate", "crates-io"),
        ("publish-manifest-crate", "crates-io"),
        ("publish-pypi", "pypi"),
    ):
        job = mapping(jobs.get(job_name), f"{path}: jobs.{job_name}")
        raw_environment = job.get("environment")
        if isinstance(raw_environment, dict):
            actual = raw_environment.get("name")
        else:
            actual = raw_environment
        if actual != environment:
            fail(f"{path}: jobs.{job_name} must use environment {environment!r}; got {actual!r}")


def check_aarch64_smoke_selector(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get("smoke-linux-aarch64-wheel"), f"{path}: jobs.smoke-linux-aarch64-wheel")
    steps = sequence(job.get("steps"), f"{path}: jobs.smoke-linux-aarch64-wheel.steps")

    matching_steps: list[dict[str, Any]] = []
    for raw_step in steps:
        step = mapping(raw_step, f"{path}: jobs.smoke-linux-aarch64-wheel.steps[]")
        if step.get("name") == "Install exact wheel and run tiny RankQuant/Bitmap smoke":
            matching_steps.append(step)

    if len(matching_steps) != 1:
        fail(f"{path}: smoke-linux-aarch64-wheel must have exactly one install/smoke step")

    run = matching_steps[0].get("run")
    if not isinstance(run, str):
        fail(f"{path}: smoke-linux-aarch64-wheel install/smoke step must be a run step")
    if "manylinux_2_17_aarch64" in run:
        fail(f"{path}: linux/aarch64 wheel selector must not pin a specific manylinux policy tag")
    if not all(needle in run for needle in ('"aarch64"', '"manylinux"', '"musllinux"', "len(wheels) != 1")):
        fail(f"{path}: linux/aarch64 wheel selector must match architecture and assert exactly one wheel")


def check_pypi_canonical_dist(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get("pypi-canonical-dist"), f"{path}: jobs.pypi-canonical-dist")
    steps = sequence(job.get("steps"), f"{path}: jobs.pypi-canonical-dist.steps")

    for needed in ("build-wheels", "build-sdist"):
        if not has_need(job, needed):
            fail(f"{path}: pypi-canonical-dist must need {needed}")

    outputs = mapping(job.get("outputs"), f"{path}: jobs.pypi-canonical-dist.outputs")
    if outputs.get("source") != "${{ steps.canonicalize.outputs.source }}":
        fail(f"{path}: pypi-canonical-dist must expose the canonical source output")

    wheels_downloads: list[int] = []
    sdist_downloads: list[int] = []
    canonicalize_steps: list[dict[str, Any]] = []
    uploads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.pypi-canonical-dist.steps[{index}]")
        action = action_name(step)
        if action == "actions/download-artifact":
            with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
            artifact_path = norm_path(with_map.get("path"))
            if with_map.get("pattern") == "wheels-*" and boolish_true(with_map.get("merge-multiple")):
                if artifact_path != "built-dist":
                    fail(f"{path}: canonical wheel download must target built-dist")
                wheels_downloads.append(index)
            elif with_map.get("name") == "sdist":
                if artifact_path != "built-dist":
                    fail(f"{path}: canonical sdist download must target built-dist")
                sdist_downloads.append(index)
        elif action == "actions/upload-artifact":
            with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
            if with_map.get("name") == "pypi-canonical-dist":
                uploads.append((index, step, with_map))

        run = step.get("run")
        if contains_text(run, "tests/release_pypi_canonical_dist.py canonicalize"):
            canonicalize_steps.append(step)
            if "--built-dir built-dist" not in run or "--out-dir canonical-dist" not in run:
                fail(f"{path}: canonicalize step must read built-dist and write canonical-dist")

    if len(wheels_downloads) != 1:
        fail(f"{path}: pypi-canonical-dist must download exactly one wheels-* artifact set")
    if len(sdist_downloads) != 1:
        fail(f"{path}: pypi-canonical-dist must download exactly one sdist artifact")
    if len(canonicalize_steps) != 1:
        fail(f"{path}: pypi-canonical-dist must run release_pypi_canonical_dist.py canonicalize")
    if len(uploads) != 1:
        fail(f"{path}: pypi-canonical-dist must upload exactly one pypi-canonical-dist artifact")

    _, _, upload_with = uploads[0]
    upload_path = upload_with.get("path")
    if not (
        contains_text(upload_path, "canonical-dist/*.whl")
        and contains_text(upload_path, "canonical-dist/*.tar.gz")
    ):
        fail(f"{path}: pypi-canonical-dist upload must include canonical wheels and sdist")


def check_publish_pypi(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get("publish-pypi"), f"{path}: jobs.publish-pypi")
    steps = sequence(job.get("steps"), f"{path}: jobs.publish-pypi.steps")

    if not has_need(job, "pypi-canonical-dist"):
        fail(f"{path}: publish-pypi must need pypi-canonical-dist")

    publish_steps: list[tuple[int, dict[str, Any]]] = []
    canonical_downloads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    verify_steps: list[dict[str, Any]] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.publish-pypi.steps[{index}]")
        action = action_name(step)
        if action == "pypa/gh-action-pypi-publish":
            publish_steps.append((index, step))
        if action == "actions/download-artifact":
            with_block = step.get("with", {})
            with_map = mapping(with_block, f"{path}: {step_label(index, step)} with")
            if with_map.get("name") == "pypi-canonical-dist":
                canonical_downloads.append((index, step, with_map))
            elif norm_path(with_map.get("path")) == "dist":
                fail(f"{path}: {step_label(index, step)} downloads a non-canonical artifact into dist")

        run = step.get("run")
        if contains_text(run, "tests/release_pypi_canonical_dist.py verify"):
            verify_steps.append(step)
            if "--dist-dir dist" not in run:
                fail(f"{path}: PyPI verify step must verify dist")

    if len(publish_steps) != 1:
        fail(f"{path}: publish-pypi must have exactly one pypa/gh-action-pypi-publish step")

    publish_index, publish_step = publish_steps[0]
    if publish_step.get("if") != "needs.pypi-canonical-dist.outputs.source == 'build'":
        fail(f"{path}: PyPI publish step must only run when canonical source is the current build")
    publish_with = mapping(
        publish_step.get("with", {}), f"{path}: {step_label(publish_index, publish_step)} with"
    )
    if norm_path(publish_with.get("packages-dir")) != "dist":
        fail(f"{path}: PyPI publish step must upload packages-dir: dist")

    if len(canonical_downloads) != 1:
        fail(f"{path}: publish-pypi must download exactly one pypi-canonical-dist artifact")
    download_index, download_step, download_with = canonical_downloads[0]
    if download_index > publish_index:
        fail(f"{path}: {step_label(download_index, download_step)} must run before the PyPI publish step")
    if norm_path(download_with.get("path")) != "dist":
        fail(f"{path}: publish-pypi must download pypi-canonical-dist into dist")

    if len(verify_steps) != 1:
        fail(f"{path}: publish-pypi must run release_pypi_canonical_dist.py verify exactly once")

    for index, step in enumerate(steps):
        if action_name(step) != "actions/download-artifact":
            continue
        with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
        label = step_label(index, step)
        artifact_path = norm_path(with_map.get("path"))
        if artifact_path == "dist" and with_map.get("name") != "pypi-canonical-dist":
            fail(f"{path}: {label} must not place non-canonical artifacts in dist")


def check_publish_crate_job(
    workflow: dict[str, Any], path: str, job_name: str, package: str, artifact_name: str
) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get(job_name), f"{path}: jobs.{job_name}")
    steps = sequence(job.get("steps"), f"{path}: jobs.{job_name}.steps")

    crate_downloads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    package_runs: list[str] = []
    publish_runs: list[str] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        run = step.get("run")
        if isinstance(run, str):
            if has_cargo_command(run, "package", package):
                package_runs.append(run)
            if has_cargo_command(run, "publish", package):
                publish_runs.append(run)
        if action_name(step) == "actions/download-artifact":
            with_block = step.get("with", {})
            with_map = mapping(with_block, f"{path}: {step_label(index, step)} with")
            if with_map.get("name") == artifact_name:
                crate_downloads.append((index, step, with_map))

    if len(crate_downloads) != 1:
        fail(f"{path}: {job_name} must download exactly one {artifact_name} artifact")

    index, step, with_map = crate_downloads[0]
    label = step_label(index, step)
    artifact_path = norm_path(with_map.get("path"))
    if artifact_path != "${{ runner.temp }}/attested":
        fail(
            f"{path}: {label} downloads {artifact_name} to {artifact_path or 'the default path'!r}; "
            "it must use ${{ runner.temp }}/attested so cargo package sees a clean checkout"
        )
    if len(package_runs) != 1:
        fail(f"{path}: {job_name} must run exactly one `cargo package -p {package} --locked`")
    if len(publish_runs) != 1:
        fail(f"{path}: {job_name} must run exactly one `cargo publish -p {package} --locked`")

    verify_step_names = {
        "Verify byte-identity vs the attested .crate",
        "Post-publish byte-identity (download from crates.io == attested)",
    }
    verify_steps: list[dict[str, Any]] = []
    found_names: set[str] = set()
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        name = step.get("name")
        if name in verify_step_names:
            verify_steps.append(step)
            found_names.add(name)
    if found_names != verify_step_names:
        fail(f"{path}: {job_name} must have both attested .crate verification steps")

    attested_path = f"${{RUNNER_TEMP}}/attested/{package}-${{VERSION}}.crate"
    for step in verify_steps:
        name = step.get("name")
        run = step.get("run")
        if not isinstance(run, str):
            fail(f"{path}: {job_name} step {name!r} must be a run step")
        if attested_path not in run:
            fail(
                f"{path}: {job_name} step {name!r} must read the attested .crate "
                f"from {attested_path}"
            )
        if name == "Post-publish byte-identity (download from crates.io == attested)":
            if "/tmp/published.crate" in run:
                fail(f"{path}: {job_name} post-publish readback must not write to /tmp")
            if not (
                "${RUNNER_TEMP}/published.crate" in run or "$RUNNER_TEMP/published.crate" in run
            ):
                fail(f"{path}: {job_name} post-publish readback must write under ${{RUNNER_TEMP}}")
            version = r"\$(?:\{VERSION\}|VERSION)"
            if not has_assignment(
                run, "API_URL", rf"https://crates\.io/api/v1/crates/{package}/{version}/download"
            ):
                fail(f"{path}: {job_name} post-publish readback must define the crates.io API URL")
            if not has_assignment(
                run,
                "STATIC_URL",
                rf"https://static\.crates\.io/crates/{package}/{package}-{version}\.crate",
            ):
                fail(f"{path}: {job_name} post-publish readback must define the static.crates.io fallback")

            curl_commands = shell_curl_commands(run)
            if not any(readback_curl_uses(words, "API_URL") for words in curl_commands):
                fail(
                    f"{path}: {job_name} post-publish readback must curl $API_URL "
                    "with CRATES_IO_USER_AGENT into $PUBLISHED"
                )
            if not any(readback_curl_uses(words, "STATIC_URL") for words in curl_commands):
                fail(
                    f"{path}: {job_name} post-publish readback must curl $STATIC_URL "
                    "with CRATES_IO_USER_AGENT into $PUBLISHED"
                )


def check_publish_crates(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    manifest_job = mapping(jobs.get("publish-manifest-crate"), f"{path}: jobs.publish-manifest-crate")
    if not has_need(manifest_job, "publish-crate"):
        fail(f"{path}: publish-manifest-crate must need publish-crate so ordvec publishes first")
    build_manifest_job = mapping(jobs.get("build-manifest-crate"), f"{path}: jobs.build-manifest-crate")
    if not has_need(build_manifest_job, "publish-crate"):
        fail(f"{path}: build-manifest-crate must need publish-crate so lockstep ordvec exists")
    build_env = mapping(build_manifest_job.get("env"), f"{path}: jobs.build-manifest-crate.env")
    if build_env.get("VERSION") != "${{ needs.guard.outputs.version }}":
        fail(f"{path}: build-manifest-crate must expose the release VERSION to retry diagnostics")
    build_steps = sequence(
        build_manifest_job.get("steps"), f"{path}: jobs.build-manifest-crate.steps"
    )
    build_manifest_packages = []
    for index, raw_step in enumerate(build_steps):
        step = mapping(raw_step, f"{path}: jobs.build-manifest-crate.steps[{index}]")
        run = step.get("run")
        if isinstance(run, str) and has_cargo_command(run, "package", "ordvec-manifest"):
            build_manifest_packages.append(run)
    if len(build_manifest_packages) != 1:
        fail(f"{path}: build-manifest-crate must package ordvec-manifest after publish-crate")
    build_run = build_manifest_packages[0]
    for fragment in ("for i in {1..12}", "sleep 10", "ordvec ${VERSION}"):
        if fragment not in build_run:
            fail(f"{path}: build-manifest-crate package step must retry crates.io propagation")
    check_publish_crate_job(workflow, path, "publish-crate", "ordvec", "dist-crate")
    check_publish_crate_job(
        workflow,
        path,
        "publish-manifest-crate",
        "ordvec-manifest",
        "dist-manifest-crate",
    )


def check_ci_manifest_package_defer(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    deps_job = mapping(jobs.get("deps"), f"{path}: jobs.deps")
    steps = sequence(deps_job.get("steps"), f"{path}: jobs.deps.steps")
    manifest_package_runs = []
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.deps.steps[{index}]")
        run = step.get("run")
        if isinstance(run, str) and has_cargo_command(run, "package", "ordvec-manifest"):
            manifest_package_runs.append(run)
    if len(manifest_package_runs) != 1:
        fail(f"{path}: deps job must run exactly one deferred ordvec-manifest package check")
    run = manifest_package_runs[0]
    if "grep" in run or "failed to select a version for the requirement" in run:
        fail(f"{path}: deferred ordvec-manifest package check must not grep cargo errors")
    required_fragments = (
        "cargo metadata --no-deps --format-version 1",
        "https://crates.io/api/v1/crates/ordvec/${core_version}",
        '--write-out "%{http_code}"',
        '[ "${status}" = "404" ]',
        "not deferring a real packaging failure",
    )
    for fragment in required_fragments:
        if fragment not in run:
            fail(f"{path}: deferred ordvec-manifest package check must include {fragment!r}")


def check_sde_setup_action(path: str) -> None:
    action = load_workflow(path)
    runs = mapping(action.get("runs"), f"{path}: runs")
    steps = sequence(runs.get("steps"), f"{path}: runs.steps")
    run_scripts = []
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: runs.steps[{index}]")
        run = step.get("run")
        if isinstance(run, str):
            run_scripts.append(run)
    curl_commands = [words for run in run_scripts for words in shell_curl_commands(run)]
    download_curl_commands = [
        words for words in curl_commands if any(word in shell_vars("download_url") for word in words)
    ]
    if len(download_curl_commands) != 1:
        fail(f"{path}: Intel SDE setup action must have exactly one curl command for $download_url")
    download_curl = download_curl_commands[0]
    for option in ("--connect-timeout", "--max-time", "--retry-max-time"):
        if option not in download_curl:
            fail(f"{path}: Intel SDE download curl must set {option} on the download command")
    action_text = read_text(path)
    if 'rm -f "${archive}" "${cached_archive}" || true' not in action_text:
        fail(f"{path}: unreadable or invalid cached Intel SDE archives must be purged")
    if "continuing without updating cache" not in action_text:
        fail(f"{path}: Intel SDE cache population failures must be best-effort warnings")
    required_outage_fragments = (
        "allow-unavailable",
        "sde-available=false",
        "downloadmirror-challenge",
        "x-amzn-waf-action",
        "sha256sum -c -",
    )
    for fragment in required_outage_fragments:
        if fragment not in action_text:
            fail(f"{path}: Intel SDE outage softening must include {fragment!r}")


def check_sde_cache_job(workflow: dict[str, Any], path: str, job_name: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get(job_name), f"{path}: jobs.{job_name}")
    job_env = mapping(job.get("env"), f"{path}: jobs.{job_name}.env")
    if not job_env.get("SDE_VERSION"):
        fail(f"{path}: jobs.{job_name} must define SDE_VERSION")
    if not job_env.get("SDE_SHA256"):
        fail(f"{path}: jobs.{job_name} must define SDE_SHA256")

    steps = sequence(job.get("steps"), f"{path}: jobs.{job_name}.steps")
    cache_steps: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    setup_steps: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        action = action_name(step)
        if action == "actions/cache":
            with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
            if norm_path(with_map.get("path")) == "~/.cache/ordvec-intel-sde":
                cache_steps.append((index, step, with_map))
        elif action == "./.github/actions/setup-intel-sde":
            with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
            setup_steps.append((index, step, with_map))

    if len(cache_steps) != 1:
        fail(f"{path}: jobs.{job_name} must restore exactly one Intel SDE archive cache")
    _, _, cache_with = cache_steps[0]
    key = cache_with.get("key")
    expected_key = (
        "intel-sde-${{ runner.os }}-${{ runner.arch }}-"
        "${{ env.SDE_VERSION }}-${{ env.SDE_SHA256 }}"
    )
    if key != expected_key:
        fail(
            f"{path}: jobs.{job_name} Intel SDE cache key must be version+sha pinned, "
            "not action-file-hash based"
        )
    restore_keys = str(cache_with.get("restore-keys") or "")
    expected_restore_key = "intel-sde-${{ runner.os }}-${{ runner.arch }}-"
    if expected_restore_key not in {line.strip() for line in restore_keys.splitlines()}:
        fail(
            f"{path}: jobs.{job_name} Intel SDE cache restore-keys must include "
            "the runner OS/arch prefix"
        )
    if contains_text(key, "hashFiles") or contains_text(key, "setup-intel-sde/action.yml"):
        fail(f"{path}: jobs.{job_name} Intel SDE cache key must not hash the action file")

    if len(setup_steps) != 1:
        fail(f"{path}: jobs.{job_name} must use exactly one setup-intel-sde action")
    _, _, setup_with = setup_steps[0]
    if setup_with.get("version") != "${{ env.SDE_VERSION }}":
        fail(f"{path}: jobs.{job_name} setup-intel-sde must receive env.SDE_VERSION")
    if setup_with.get("sha256") != "${{ env.SDE_SHA256 }}":
        fail(f"{path}: jobs.{job_name} setup-intel-sde must receive env.SDE_SHA256")
    if not boolish_true(setup_with.get("allow-unavailable")):
        fail(f"{path}: jobs.{job_name} must explicitly opt into the temporary SDE outage valve")

    available_if = "steps.sde.outputs.sde-available == 'true'"
    unavailable_if = "steps.sde.outputs.sde-available != 'true'"
    outage_notice_steps = []
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        if step.get("if") == unavailable_if and contains_text(
            step.get("run"), "Intel SDE archive unavailable"
        ):
            outage_notice_steps.append(step)
    if len(outage_notice_steps) != 1:
        fail(f"{path}: jobs.{job_name} must emit exactly one notice when Intel SDE is unavailable")

    sde_guarded_names = {
        "Install cargo-llvm-cov (pinned)",
        "Sanity-check AVX-512 detection under SDE",
        "sanity-check AVX-512 detection under SDE",
        "Generate coverage (lcov) + enforce floor",
        "Upload coverage to Codecov",
        "cargo test under SDE (AVX-512 kernels)",
    }
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        name = step.get("name")
        if (
            name in sde_guarded_names
            or contains_nested_text(step.get("env"), "steps.sde.outputs.sde-path")
            or contains_text(step.get("run"), "SDE_PATH")
        ):
            if step.get("if") != available_if:
                fail(
                    f"{path}: {step_label(index, step)} must be guarded by "
                    "steps.sde.outputs.sde-available"
                )


def check_sde_cache_invariants() -> None:
    check_sde_setup_action(SDE_ACTION_PATH)
    check_sde_cache_job(load_workflow(CI_WORKFLOW_PATH), CI_WORKFLOW_PATH, "avx512")
    check_sde_cache_job(load_workflow(COVERAGE_WORKFLOW_PATH), COVERAGE_WORKFLOW_PATH, "coverage")


def main() -> None:
    workflow = load_workflow(WORKFLOW_PATH)
    check_release_version_sync()
    check_release_compatibility_sync()
    check_publication_model()
    check_python_package_metadata()
    check_strict_release_tag_patterns(workflow, WORKFLOW_PATH)
    check_hash_requirement_temp_paths(
        [WORKFLOW_PATH, PYTHON_WORKFLOW_PATH, CI_WORKFLOW_PATH, COVERAGE_WORKFLOW_PATH]
    )
    check_release_security_gates(workflow, WORKFLOW_PATH)
    check_aarch64_smoke_selector(workflow, WORKFLOW_PATH)
    check_pypi_canonical_dist(workflow, WORKFLOW_PATH)
    check_publish_crates(workflow, WORKFLOW_PATH)
    check_ci_manifest_package_defer(load_workflow(CI_WORKFLOW_PATH), CI_WORKFLOW_PATH)
    check_publish_pypi(workflow, WORKFLOW_PATH)
    check_sde_cache_invariants()


if __name__ == "__main__":
    main()
