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
from fnmatch import fnmatchcase
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None  # type: ignore[assignment]


WORKFLOW_PATH = os.environ.get("RELEASE_WORKFLOW_PATH", ".github/workflows/release.yml")
CI_WORKFLOW_PATH = os.environ.get("CI_WORKFLOW_PATH", ".github/workflows/ci.yml")
PYTHON_WORKFLOW_PATH = os.environ.get("PYTHON_WORKFLOW_PATH", ".github/workflows/python.yml")
STRICT_STABLE_TAG_PATTERN = r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
COVERAGE_WORKFLOW_PATH = os.environ.get("COVERAGE_WORKFLOW_PATH", ".github/workflows/coverage.yml")
SDE_ACTION_PATH = os.environ.get(
    "SDE_ACTION_PATH", ".github/actions/setup-intel-sde/action.yml"
)
ROUTINE_CI_SDE_ALLOW_UNAVAILABLE = "${{ github.event_name != 'workflow_dispatch' }}"
RELEASE_SDE_ALLOW_UNAVAILABLE = "false"
SDE_AVAILABLE_IF = "${{ steps.sde.outputs.sde-available == 'true' }}"
SDE_UNAVAILABLE_NOTICE_IF = "${{ steps.sde.outputs.sde-available != 'true' }}"
PYPI_CANONICAL_EXPECTED_ARGS = (
    "--expected-wheels 4",
    "--expected-sdists 1",
    "--required-wheel-tag x86_64",
    "--required-wheel-tag aarch64",
    "--required-wheel-tag macosx",
    "--required-wheel-tag win_amd64",
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


def boolish_false(value: Any) -> bool:
    return value is False or (isinstance(value, str) and value.lower() == "false")


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


def string_sequence(value: Any, context: str) -> list[str]:
    items = sequence(value, context)
    if not all(isinstance(item, str) for item in items):
        fail(f"{context} must contain only strings")
    return items


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
        "ordvec-manifest-python/Cargo.toml package.version": package_version(
            "ordvec-manifest-python/Cargo.toml"
        ),
        "ordvec-manifest-python/pyproject.toml project.version": project_version(
            "ordvec-manifest-python/pyproject.toml"
        ),
        "ordvec-manifest-python/python/ordvec_manifest/__init__.py __version__": python_init_version(
            "ordvec-manifest-python/python/ordvec_manifest/__init__.py"
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
        "ordvec-manifest-python/Cargo.toml",
        "ordvec-ffi/Cargo.toml",
    ):
        rust_version = package_rust_version(path)
        if rust_version != core_msrv:
            fail(f"{path}: package.rust-version is {rust_version}, expected {core_msrv}")

    fuzz_rust_version = package_rust_version("fuzz/Cargo.toml")
    if fuzz_rust_version != core_msrv:
        fail(f"fuzz/Cargo.toml: package.rust-version is {fuzz_rust_version}, expected {core_msrv}")

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

    msrv_features = read_text("docs/msrv-and-features.md")
    if f"Current MSRV: Rust {core_msrv}." not in msrv_features:
        fail(f"docs/msrv-and-features.md must mention Rust {core_msrv}")
    for required in (
        "`Cargo.toml` `rust-version`",
        "README MSRV badge/section",
        "New feature flags must declare a stability class before merging",
        "`experimental` exposes `MultiBucketBitmap`",
        "`test-utils` is repo-test-only",
        "`cli`, `sqlite`, `sqlite-bundled`",
        "without hidden platform or feature surprises",
    ):
        if required not in msrv_features:
            fail(f"docs/msrv-and-features.md must mention {required!r}")

    ci = read_text(".github/workflows/ci.yml")
    msrv_toolchain = f"{core_msrv}.0"
    if f"name: msrv ({msrv_toolchain})" not in ci:
        fail(f".github/workflows/ci.yml MSRV job name must mention {msrv_toolchain}")
    if f"toolchain: {msrv_toolchain}" not in ci:
        fail(f".github/workflows/ci.yml MSRV job must pin toolchain {msrv_toolchain}")


def check_registry_metadata_parity() -> None:
    expected_crates = {
        "Cargo.toml": {
            "license": "MIT OR Apache-2.0",
            "repository": "https://github.com/Project-Navi/ordvec",
            "homepage": "https://github.com/Project-Navi/ordvec",
            "documentation": "https://docs.rs/ordvec",
            "readme": "README.md",
            "keywords": ["vector-search", "quantization", "nearest-neighbor", "ann", "simd"],
            "categories": ["algorithms", "science", "compression"],
        },
        "ordvec-manifest/Cargo.toml": {
            "license": "MIT OR Apache-2.0",
            "repository": "https://github.com/Project-Navi/ordvec",
            "homepage": "https://github.com/Project-Navi/ordvec",
            "documentation": "https://docs.rs/ordvec-manifest",
            "readme": "README.md",
            "keywords": ["vector-search", "manifest", "provenance", "verification", "quantization"],
            "categories": ["algorithms", "command-line-utilities", "data-structures"],
        },
    }

    for path, expected in expected_crates.items():
        data = load_toml(path)
        package = mapping(data.get("package"), f"{path}: package")
        for key in ("license", "repository", "homepage", "documentation", "readme"):
            if package.get(key) != expected[key]:
                fail(f"{path}: package.{key} is {package.get(key)!r}, expected {expected[key]!r}")
        for key in ("keywords", "categories"):
            actual = string_sequence(package.get(key), f"{path}: package.{key}")
            if actual != expected[key]:
                fail(f"{path}: package.{key} is {actual!r}, expected {expected[key]!r}")

        metadata = mapping(package.get("metadata"), f"{path}: package.metadata")
        docs = mapping(metadata.get("docs"), f"{path}: package.metadata.docs")
        docs_rs = mapping(docs.get("rs"), f"{path}: package.metadata.docs.rs")
        if docs_rs.get("all-features") is not False:
            fail(f"{path}: package.metadata.docs.rs.all-features must be false")
        if path == "ordvec-manifest/Cargo.toml":
            features = string_sequence(
                docs_rs.get("features"), f"{path}: package.metadata.docs.rs.features"
            )
            if features != ["cli", "sqlite-bundled"]:
                fail(
                    f"{path}: package.metadata.docs.rs.features is {features!r}, "
                    "expected ['cli', 'sqlite-bundled']"
                )
        elif "features" in docs_rs:
            fail(f"{path}: package.metadata.docs.rs.features must not be set")


def check_manifest_cli_defaults() -> None:
    manifest = load_toml("ordvec-manifest/Cargo.toml")
    features = mapping(manifest.get("features"), "ordvec-manifest/Cargo.toml: features")
    default_features = string_sequence(
        features.get("default"), "ordvec-manifest/Cargo.toml: features.default"
    )
    if default_features != ["cli"]:
        fail("ordvec-manifest/Cargo.toml: default features must be ['cli']")
    cli_features = string_sequence(
        features.get("cli"), "ordvec-manifest/Cargo.toml: features.cli"
    )
    if "dep:clap" not in cli_features:
        fail("ordvec-manifest/Cargo.toml: features.cli must enable dep:clap")

    text = read_text("ordvec-manifest/Cargo.toml")
    if 'name = "ordvec-manifest"' not in text or 'required-features = ["cli"]' not in text:
        fail("ordvec-manifest/Cargo.toml: binary must remain gated on the cli feature")

    readme = read_text("ordvec-manifest/README.md")
    if "cargo install ordvec-manifest --features cli" in readme:
        fail("ordvec-manifest/README.md: install instructions must not require --features cli")
    if "cargo install ordvec-manifest" not in readme:
        fail("ordvec-manifest/README.md: must document default cargo install")


def check_publication_model() -> None:
    expected_publish = {
        "Cargo.toml": True,
        "ordvec-manifest/Cargo.toml": True,
        "ordvec-python/Cargo.toml": False,
        "ordvec-manifest-python/Cargo.toml": False,
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
    license_table = mapping(project.get("license"), "ordvec-python/pyproject.toml: project.license")
    if license_table.get("text") != "MIT OR Apache-2.0":
        fail("ordvec-python/pyproject.toml: project.license.text must be MIT OR Apache-2.0")
    classifiers = set(
        string_sequence(
            project.get("classifiers"), "ordvec-python/pyproject.toml: project.classifiers"
        )
    )
    for classifier in (
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Rust",
    ):
        if classifier not in classifiers:
            fail(f"ordvec-python/pyproject.toml: missing classifier {classifier!r}")
    urls = mapping(project.get("urls"), "ordvec-python/pyproject.toml: project.urls")
    for key, expected in {
        "Homepage": "https://github.com/Project-Navi/ordvec",
        "Repository": "https://github.com/Project-Navi/ordvec",
        "Issues": "https://github.com/Project-Navi/ordvec/issues",
        "Formalization": "https://github.com/Fieldnote-Echo/ordvec-formalization",
    }.items():
        if urls.get(key) != expected:
            fail(f"ordvec-python/pyproject.toml: project.urls.{key} must be {expected!r}")

    cargo = load_toml("ordvec-python/Cargo.toml")
    dependencies_table = mapping(cargo.get("dependencies"), "ordvec-python/Cargo.toml: dependencies")
    pyo3 = mapping(dependencies_table.get("pyo3"), "ordvec-python/Cargo.toml: dependencies.pyo3")
    pyo3_features = sequence(
        pyo3.get("features"), "ordvec-python/Cargo.toml: dependencies.pyo3.features"
    )
    for feature in ("extension-module", "abi3-py310"):
        if feature not in pyo3_features:
            fail(f"ordvec-python/Cargo.toml: pyo3 features must include {feature}")

    manifest_pyproject = load_toml("ordvec-manifest-python/pyproject.toml")
    manifest_project = mapping(
        manifest_pyproject.get("project"),
        "ordvec-manifest-python/pyproject.toml: project",
    )
    if manifest_project.get("name") != "ordvec-manifest":
        fail("ordvec-manifest-python/pyproject.toml: project.name must be 'ordvec-manifest'")
    if manifest_project.get("requires-python") != ">=3.10":
        fail("ordvec-manifest-python/pyproject.toml: project.requires-python must be >=3.10")
    manifest_license = mapping(
        manifest_project.get("license"),
        "ordvec-manifest-python/pyproject.toml: project.license",
    )
    if manifest_license.get("text") != "MIT OR Apache-2.0":
        fail(
            "ordvec-manifest-python/pyproject.toml: project.license.text must be MIT OR Apache-2.0"
        )
    manifest_classifiers = set(
        string_sequence(
            manifest_project.get("classifiers"),
            "ordvec-manifest-python/pyproject.toml: project.classifiers",
        )
    )
    for classifier in (
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Rust",
    ):
        if classifier not in manifest_classifiers:
            fail(f"ordvec-manifest-python/pyproject.toml: missing classifier {classifier!r}")
    manifest_urls = mapping(
        manifest_project.get("urls"),
        "ordvec-manifest-python/pyproject.toml: project.urls",
    )
    for key, expected in {
        "Homepage": "https://github.com/Project-Navi/ordvec",
        "Repository": "https://github.com/Project-Navi/ordvec",
        "Issues": "https://github.com/Project-Navi/ordvec/issues",
    }.items():
        if manifest_urls.get(key) != expected:
            fail(
                "ordvec-manifest-python/pyproject.toml: "
                f"project.urls.{key} must be {expected!r}"
            )

    manifest_cargo = load_toml("ordvec-manifest-python/Cargo.toml")
    manifest_dependencies = mapping(
        manifest_cargo.get("dependencies"),
        "ordvec-manifest-python/Cargo.toml: dependencies",
    )
    manifest_pyo3 = mapping(
        manifest_dependencies.get("pyo3"),
        "ordvec-manifest-python/Cargo.toml: dependencies.pyo3",
    )
    manifest_pyo3_features = sequence(
        manifest_pyo3.get("features"),
        "ordvec-manifest-python/Cargo.toml: dependencies.pyo3.features",
    )
    for feature in ("extension-module", "abi3-py310"):
        if feature not in manifest_pyo3_features:
            fail(f"ordvec-manifest-python/Cargo.toml: pyo3 features must include {feature}")

    manifest_tool = mapping(
        manifest_pyproject.get("tool"),
        "ordvec-manifest-python/pyproject.toml: tool",
    )
    manifest_maturin = mapping(
        manifest_tool.get("maturin"),
        "ordvec-manifest-python/pyproject.toml: tool.maturin",
    )
    if manifest_maturin.get("module-name") != "ordvec_manifest._ordvec_manifest":
        fail("ordvec-manifest-python/pyproject.toml: tool.maturin.module-name must target ordvec_manifest._ordvec_manifest")

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


def check_release_docs_include_manifest_pypi_lane() -> None:
    releasing = read_text("RELEASING.md")
    normalized_releasing = " ".join(releasing.split())
    for required in (
        "`ordvec-manifest` on PyPI",
        "`publish-manifest-pypi`",
        "four registry publish jobs",
        "PyPI must point both `ordvec` and `ordvec-manifest`",
        "https://pypi.org/p/ordvec-manifest",
    ):
        if " ".join(required.split()) not in normalized_releasing:
            fail(f"RELEASING.md must mention {required!r}")

    threat_model = read_text("THREAT_MODEL.md")
    normalized_threat_model = " ".join(threat_model.split())
    for required in ("`publish-manifest-pypi`", "two **`pypi`** publish jobs"):
        if " ".join(required.split()) not in normalized_threat_model:
            fail(f"THREAT_MODEL.md must mention {required!r}")


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


def cargo_package_files(package: str) -> set[str]:
    cmd = ["cargo", "package", "-p", package, "--list", "--locked", "--allow-dirty"]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        fail(f"{' '.join(cmd)} failed while checking package contents: {stderr}")
    return {line.strip() for line in output.splitlines() if line.strip()}


def check_required_package_files(package: str, files: set[str], required: set[str]) -> None:
    missing = sorted(required - files)
    if missing:
        fail(f"{package}: package is missing required files: {', '.join(missing)}")


def check_forbidden_package_prefixes(
    package: str, files: set[str], forbidden_prefixes: tuple[str, ...]
) -> None:
    forbidden = sorted(
        file for file in files if any(file == prefix.rstrip("/") or file.startswith(prefix) for prefix in forbidden_prefixes)
    )
    if forbidden:
        fail(f"{package}: package includes forbidden files: {', '.join(forbidden)}")


def check_packaged_readme_links(package: str, files: set[str], readme_path: str) -> None:
    readme = read_text(readme_path)
    for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", readme):
        raw_target = match.group(1).strip()
        if not raw_target or raw_target.startswith("#"):
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", raw_target):
            continue
        target = raw_target.split("#", 1)[0].split("?", 1)[0].strip()
        if not target:
            continue
        if target.startswith("/") or target.startswith("../") or "/../" in target:
            fail(f"{package}: README link {raw_target!r} escapes the packaged crate")
        normalized = posixpath.normpath(target)
        if normalized not in files and not any(file.startswith(normalized + "/") for file in files):
            fail(f"{package}: README link {raw_target!r} points to a file or directory not packaged")


def check_package_contents() -> None:
    core_files = cargo_package_files("ordvec")
    check_required_package_files(
        "ordvec",
        core_files,
        {
            "Cargo.lock",
            "Cargo.toml",
            "Cargo.toml.orig",
            "CHANGELOG.md",
            "LICENSE-APACHE-2.0",
            "LICENSE-MIT",
            "README.md",
            "benchmarks/rank_modes_results.txt",
            "docs/PERSISTED_FORMAT.md",
            "docs/RANK_MODES.md",
            "docs/compatibility-policy.md",
            "docs/determinism.md",
            "examples/bench_rank.rs",
            "src/lib.rs",
            "tests/index/main.rs",
            "tests/persistence_compat.rs",
        },
    )
    check_forbidden_package_prefixes(
        "ordvec",
        core_files,
        (
            ".agents/",
            ".claude/",
            ".codex/",
            ".github/",
            ".playwright-mcp/",
            "fuzz/",
            "ordvec-ffi/",
            "ordvec-go/",
            "ordvec-manifest/",
            "ordvec-manifest-python/",
            "ordvec-python/",
            "target/",
            "tests/release_",
        ),
    )
    check_packaged_readme_links("ordvec", core_files, "README.md")

    manifest_files = cargo_package_files("ordvec-manifest")
    check_required_package_files(
        "ordvec-manifest",
        manifest_files,
        {
            "Cargo.lock",
            "Cargo.toml",
            "Cargo.toml.orig",
            "LICENSE-APACHE-2.0",
            "LICENSE-MIT",
            "README.md",
            "src/lib.rs",
            "src/main.rs",
            "src/sqlite.rs",
            "tests/manifest.rs",
        },
    )
    check_forbidden_package_prefixes(
        "ordvec-manifest",
        manifest_files,
        (
            ".agents/",
            ".claude/",
            ".codex/",
            ".github/",
            ".playwright-mcp/",
            "docs/",
            "fuzz/",
            "ordvec-ffi/",
            "ordvec-go/",
            "ordvec-manifest/",
            "ordvec-manifest-python/",
            "ordvec-python/",
            "target/",
            "tests/release_",
        ),
    )
    check_packaged_readme_links("ordvec-manifest", manifest_files, "ordvec-manifest/README.md")


def check_ci_package_guards(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    deps = mapping(jobs.get("deps"), f"{path}: jobs.deps")
    steps = sequence(deps.get("steps"), f"{path}: jobs.deps.steps")

    core_dry_runs: list[str] = []
    manifest_deferred_runs: list[str] = []
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.deps.steps[{index}]")
        run = step.get("run")
        if not isinstance(run, str):
            continue
        for words in cargo_command_words(run, "publish", "ordvec"):
            if "--dry-run" in words:
                core_dry_runs.append(run)
        if cargo_command_words(run, "package", "ordvec-manifest"):
            manifest_deferred_runs.append(run)

    if len(core_dry_runs) != 1:
        fail(f"{path}: deps job must run exactly one `cargo publish -p ordvec --dry-run --locked`")
    if len(manifest_deferred_runs) != 1:
        fail(f"{path}: deps job must run exactly one deferred ordvec-manifest package check")

    manifest_run = manifest_deferred_runs[0]
    if "grep" in manifest_run or "failed to select a version for the requirement" in manifest_run:
        fail(f"{path}: deferred ordvec-manifest package check must not grep cargo errors")
    required_fragments = (
        "cargo metadata --no-deps --format-version 1",
        "https://crates.io/api/v1/crates/ordvec/${core_version}",
        '--write-out "%{http_code}"',
        '[ "${status}" = "404" ]',
        "ordvec-manifest package check is deferred",
        "not deferring a real packaging failure",
    )
    for fragment in required_fragments:
        if fragment not in manifest_run:
            fail(f"{path}: deferred ordvec-manifest package check must include {fragment!r}")


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


def cargo_command_words(run: str, subcommand: str, package: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in shell_logical_lines(run):
        for part in re.split(r"&&|\|\||;", line):
            part = part.strip()
            for prefix in ("if ", "then ", "! "):
                if part.startswith(prefix):
                    part = part[len(prefix):].strip()
            if not part:
                continue
            try:
                words = shlex.split(part)
            except ValueError:
                continue
            cmd_idx = 0
            while cmd_idx < len(words) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[cmd_idx]):
                cmd_idx += 1
            cmd = words[cmd_idx:]
            if len(cmd) < 3 or cmd[0] != "cargo" or cmd[1] != subcommand:
                continue
            if "--locked" in cmd and has_cargo_package_arg(cmd[2:], package):
                commands.append(cmd)
    return commands


def has_cargo_command(run: str, subcommand: str, package: str) -> bool:
    return bool(cargo_command_words(run, subcommand, package))


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


def recovery_curl_uses(words: list[str], url_var: str) -> bool:
    return (
        has_shell_arg(words, shell_vars(url_var))
        and has_shell_option_value(words, {"--user-agent", "-A"}, shell_vars("CRATES_IO_USER_AGENT"))
        and has_shell_option_value(words, {"--output", "-o"}, shell_vars("EXISTING"))
        and has_shell_option_value(words, {"--write-out", "-w"}, {"%{http_code}"})
        and "--retry" in words
        and "--retry-all-errors" in words
    )


def metadata_curl_uses(words: list[str]) -> bool:
    return (
        has_shell_arg(words, shell_vars("METADATA_URL"))
        and has_shell_option_value(words, {"--user-agent", "-A"}, shell_vars("CRATES_IO_USER_AGENT"))
        and has_shell_option_value(words, {"--header", "-H"}, {"Accept: application/json"})
        and has_shell_option_value(words, {"--output", "-o"}, shell_vars("METADATA"))
        and has_shell_option_value(words, {"--write-out", "-w"}, {"%{http_code}"})
        and "--retry" in words
        and "--retry-all-errors" in words
    )


def recovery_curl_is_bounded(words: list[str], url_var: str) -> bool:
    return (
        recovery_curl_uses(words, url_var)
        and has_shell_option_value(words, {"--retry"}, {"0", "1"})
        and has_shell_option_value(words, {"--connect-timeout"}, {"5", "10"})
        and has_shell_option_value(words, {"--max-time", "-m"}, {"10", "15", "20"})
    )


def has_recovery_retry_loop(run: str, package: str) -> bool:
    bounded_loop = re.search(
        r"\bfor\s+[A-Za-z_][A-Za-z0-9_]*\s+in\s+"
        r"(?:1\s+2\s+3\s+4\s+5\s+6\s+7\s+8\s+9\s+10\s+11\s+12|\{1\.\.12\}|\$\(seq\s+1\s+12\));\s*do",
        run,
    )
    return bool(
        bounded_loop
        and re.search(r"\bsleep\s+5\b", run)
        and re.search(
            r'\[\s*"\$API_CURL_EXIT"\s+-eq\s+0\s*\]\s*&&\s*'
            r'\[\s*"\$API_STATUS"\s*=\s*200\s*\]',
            run,
        )
        and re.search(
            r'\[\s*"\$STATIC_CURL_EXIT"\s+-eq\s+0\s*\]\s*&&\s*'
            r'\[\s*"\$STATIC_STATUS"\s*=\s*200\s*\]',
            run,
        )
        and f"waiting for crates.io to serve {package}" in run
    )


def check_crate_recovery_status_handling(
    recovery_run: str, path: str, job_name: str, package: str
) -> None:
    required_fragments = (
        "METADATA_CURL_EXIT=0",
        'if [ "$METADATA_CURL_EXIT" -ne 0 ]; then',
        'case "$METADATA_STATUS" in',
        f"crates.io metadata does not list {package}",
        f"crates.io metadata lists {package}",
        "could not determine crates.io metadata",
        "unexpected crates.io metadata status",
        "API_CURL_EXIT=0",
        "STATIC_CURL_EXIT=0",
        "curl exit ${API_CURL_EXIT}",
        "curl exit ${STATIC_CURL_EXIT}",
        "recovery endpoints did not serve the .crate after retries",
        "200)",
        "404)",
    )
    for fragment in required_fragments:
        if fragment not in recovery_run:
            fail(f"{path}: {job_name} recovery step must contain {fragment!r}")
    if not has_recovery_retry_loop(recovery_run, package):
        fail(
            f"{path}: {job_name} recovery step must use a bounded propagation loop "
            "with sleep and API/static success checks"
        )
    curl_commands = shell_curl_commands(recovery_run)
    for url_var in ("API_URL", "STATIC_URL"):
        if not any(recovery_curl_is_bounded(words, url_var) for words in curl_commands):
            fail(
                f"{path}: {job_name} recovery step must keep ${url_var} curl probes "
                "short inside the outer propagation loop"
            )
    if "Both crates.io recovery endpoints returned 404" in recovery_run:
        fail(
            f"{path}: {job_name} recovery step must use crates.io metadata, "
            "not download-endpoint 404s, to decide first-publish absence"
        )
    if "VERSION_PRESENT_FILE" in recovery_run or "python3 -" in recovery_run:
        fail(
            f"{path}: {job_name} recovery step must use the per-version metadata endpoint, "
            "not inline JSON parsing"
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
    if (
        found_gate_run is None
        or "repos/${REPO}/commits/main" not in found_gate_run
        or "MAIN_SHA" not in found_gate_run
    ):
        fail(f"{path}: require-ci-green must verify the release tag points at current main")

    allowed_id_token_jobs = {
        "attest",
        "provenance",
        "publish-crate",
        "attest-manifest",
        "manifest-provenance",
        "publish-manifest-crate",
        "publish-pypi",
        "publish-manifest-pypi",
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
        ("publish-manifest-pypi", "pypi"),
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


def check_pypi_canonical_dist(
    workflow: dict[str, Any],
    path: str,
    *,
    job_name: str = "pypi-canonical-dist",
    wheel_build_job: str = "build-wheels",
    sdist_build_job: str = "build-sdist",
    wheel_artifact_pattern: str = "wheels-*",
    sdist_artifact_name: str = "sdist",
    canonical_artifact_name: str = "pypi-canonical-dist",
    project: str | None = None,
    required_license_files: tuple[str, ...] = (),
) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get(job_name), f"{path}: jobs.{job_name}")
    steps = sequence(job.get("steps"), f"{path}: jobs.{job_name}.steps")

    for needed in (wheel_build_job, sdist_build_job):
        if not has_need(job, needed):
            fail(f"{path}: {job_name} must need {needed}")

    wheel_job = mapping(jobs.get(wheel_build_job), f"{path}: jobs.{wheel_build_job}")
    wheel_steps = sequence(wheel_job.get("steps"), f"{path}: jobs.{wheel_build_job}.steps")
    wheel_upload_names: list[str] = []
    for index, raw_step in enumerate(wheel_steps):
        step = mapping(raw_step, f"{path}: jobs.{wheel_build_job}.steps[{index}]")
        if action_name(step) != "actions/upload-artifact":
            continue
        with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
        name = with_map.get("name")
        if isinstance(name, str) and fnmatchcase(name, wheel_artifact_pattern):
            wheel_upload_names.append(name)
    if len(wheel_upload_names) != 1:
        fail(
            f"{path}: {wheel_build_job} must upload exactly one artifact matching "
            f"{wheel_artifact_pattern}; got {wheel_upload_names!r}"
        )

    outputs = mapping(job.get("outputs"), f"{path}: jobs.{job_name}.outputs")
    if outputs.get("source") != "${{ steps.canonicalize.outputs.source }}":
        fail(f"{path}: {job_name} must expose the canonical source output")

    wheels_downloads: list[int] = []
    sdist_downloads: list[int] = []
    canonicalize_steps: list[dict[str, Any]] = []
    uploads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        action = action_name(step)
        if action == "actions/download-artifact":
            with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
            artifact_path = norm_path(with_map.get("path"))
            if with_map.get("pattern") == wheel_artifact_pattern and boolish_true(with_map.get("merge-multiple")):
                if artifact_path != "built-dist":
                    fail(f"{path}: {job_name} canonical wheel download must target built-dist")
                wheels_downloads.append(index)
            elif with_map.get("name") == sdist_artifact_name:
                if artifact_path != "built-dist":
                    fail(f"{path}: {job_name} canonical sdist download must target built-dist")
                sdist_downloads.append(index)
        elif action == "actions/upload-artifact":
            with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
            if with_map.get("name") == canonical_artifact_name:
                uploads.append((index, step, with_map))

        run = step.get("run")
        if contains_text(run, "tests/release_pypi_canonical_dist.py canonicalize"):
            canonicalize_steps.append(step)
            if "--built-dir built-dist" not in run or "--out-dir canonical-dist" not in run:
                fail(f"{path}: {job_name} canonicalize step must read built-dist and write canonical-dist")
            if project is not None and f"--project {project}" not in run:
                fail(f"{path}: {job_name} canonicalize step must pass --project {project}")
            for required_arg in PYPI_CANONICAL_EXPECTED_ARGS:
                if required_arg not in run:
                    fail(f"{path}: {job_name} canonicalize step must pass {required_arg}")
            for license_file in required_license_files:
                if f"--require-license-file {license_file}" not in run:
                    fail(
                        f"{path}: {job_name} canonicalize step must pass "
                        f"--require-license-file {license_file}"
                    )

    if len(wheels_downloads) != 1:
        fail(f"{path}: {job_name} must download exactly one {wheel_artifact_pattern} artifact set")
    if len(sdist_downloads) != 1:
        fail(f"{path}: {job_name} must download exactly one {sdist_artifact_name} artifact")
    if len(canonicalize_steps) != 1:
        fail(f"{path}: {job_name} must run release_pypi_canonical_dist.py canonicalize")
    if len(uploads) != 1:
        fail(f"{path}: {job_name} must upload exactly one {canonical_artifact_name} artifact")

    _, _, upload_with = uploads[0]
    upload_path = upload_with.get("path")
    if not (
        contains_text(upload_path, "canonical-dist/*.whl")
        and contains_text(upload_path, "canonical-dist/*.tar.gz")
    ):
        fail(f"{path}: {job_name} upload must include canonical wheels and sdist")


def check_publish_pypi(
    workflow: dict[str, Any],
    path: str,
    *,
    job_name: str = "publish-pypi",
    canonical_job: str = "pypi-canonical-dist",
    canonical_artifact_name: str = "pypi-canonical-dist",
    project: str | None = None,
    crate_publish_job: str = "publish-crate",
    required_license_files: tuple[str, ...] = (),
) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get(job_name), f"{path}: jobs.{job_name}")
    steps = sequence(job.get("steps"), f"{path}: jobs.{job_name}.steps")

    if not has_need(job, canonical_job):
        fail(f"{path}: {job_name} must need {canonical_job}")
    if not has_need(job, crate_publish_job):
        fail(f"{path}: {job_name} must need {crate_publish_job} to avoid a partial PyPI-first release")

    publish_steps: list[tuple[int, dict[str, Any]]] = []
    canonical_downloads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    verify_steps: list[dict[str, Any]] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        action = action_name(step)
        if action == "pypa/gh-action-pypi-publish":
            publish_steps.append((index, step))
        if action == "actions/download-artifact":
            with_block = step.get("with", {})
            with_map = mapping(with_block, f"{path}: {step_label(index, step)} with")
            if with_map.get("name") == canonical_artifact_name:
                canonical_downloads.append((index, step, with_map))
            elif norm_path(with_map.get("path")) == "dist":
                fail(f"{path}: {step_label(index, step)} downloads a non-canonical artifact into dist")

        run = step.get("run")
        if contains_text(run, "tests/release_pypi_canonical_dist.py verify"):
            verify_steps.append(step)
            if "--dist-dir dist" not in run:
                fail(f"{path}: {job_name} PyPI verify step must verify dist")
            if project is not None and f"--project {project}" not in run:
                fail(f"{path}: {job_name} PyPI verify step must pass --project {project}")
            for required_arg in PYPI_CANONICAL_EXPECTED_ARGS:
                if required_arg not in run:
                    fail(f"{path}: {job_name} PyPI verify step must pass {required_arg}")
            for license_file in required_license_files:
                if f"--require-license-file {license_file}" not in run:
                    fail(
                        f"{path}: {job_name} PyPI verify step must pass "
                        f"--require-license-file {license_file}"
                    )

    if len(publish_steps) != 1:
        fail(f"{path}: {job_name} must have exactly one pypa/gh-action-pypi-publish step")

    publish_index, publish_step = publish_steps[0]
    if publish_step.get("if") != f"needs.{canonical_job}.outputs.source == 'build'":
        fail(f"{path}: {job_name} PyPI publish step must only run when canonical source is the current build")
    publish_with = mapping(
        publish_step.get("with", {}), f"{path}: {step_label(publish_index, publish_step)} with"
    )
    if norm_path(publish_with.get("packages-dir")) != "dist":
        fail(f"{path}: {job_name} PyPI publish step must upload packages-dir: dist")

    if len(canonical_downloads) != 1:
        fail(f"{path}: {job_name} must download exactly one {canonical_artifact_name} artifact")
    download_index, download_step, download_with = canonical_downloads[0]
    if download_index > publish_index:
        fail(f"{path}: {step_label(download_index, download_step)} must run before the PyPI publish step")
    if norm_path(download_with.get("path")) != "dist":
        fail(f"{path}: {job_name} must download {canonical_artifact_name} into dist")

    if len(verify_steps) != 1:
        fail(f"{path}: {job_name} must run release_pypi_canonical_dist.py verify exactly once")

    for index, step in enumerate(steps):
        if action_name(step) != "actions/download-artifact":
            continue
        with_map = mapping(step.get("with", {}), f"{path}: {step_label(index, step)} with")
        label = step_label(index, step)
        artifact_path = norm_path(with_map.get("path"))
        if artifact_path == "dist" and with_map.get("name") != canonical_artifact_name:
            fail(f"{path}: {label} must not place non-canonical artifacts in dist")


def check_publish_crate_job(
    workflow: dict[str, Any],
    path: str,
    job_name: str,
    package: str,
    artifact_name: str,
    *,
    require_publish_dry_run: bool = False,
) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get(job_name), f"{path}: jobs.{job_name}")
    steps = sequence(job.get("steps"), f"{path}: jobs.{job_name}.steps")

    crate_downloads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    package_runs: list[tuple[int, str]] = []
    publish_runs: list[tuple[int, str]] = []
    publish_dry_runs: list[tuple[int, str]] = []
    auth_steps: list[int] = []
    recovery_steps: list[tuple[int, dict[str, Any]]] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        run = step.get("run")
        if isinstance(run, str):
            if has_cargo_command(run, "package", package):
                package_runs.append((index, run))
            for words in cargo_command_words(run, "publish", package):
                if "--dry-run" in words:
                    publish_dry_runs.append((index, run))
                else:
                    publish_runs.append((index, run))
        if action_name(step) == "rust-lang/crates-io-auth-action":
            auth_steps.append(index)
        if step.get("name") == f"Check for existing {package} .crate recovery":
            recovery_steps.append((index, step))
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
    if require_publish_dry_run and len(publish_dry_runs) != 1:
        fail(
            f"{path}: {job_name} must run exactly one "
            f"`cargo publish -p {package} --dry-run --locked` before minting OIDC"
        )

    verify_step_names = {
        "Verify byte-identity vs the attested .crate",
        "Post-publish byte-identity (download from crates.io == attested)",
    }
    verify_steps: list[dict[str, Any]] = []
    verify_step_indices: dict[str, int] = {}
    found_names: set[str] = set()
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        name = step.get("name")
        if name in verify_step_names:
            verify_steps.append(step)
            verify_step_indices[name] = index
            found_names.add(name)
    if found_names != verify_step_names:
        fail(f"{path}: {job_name} must have both attested .crate verification steps")

    recovery_id = "crate_recovery" if package == "ordvec" else "manifest_crate_recovery"
    if len(recovery_steps) != 1:
        fail(f"{path}: {job_name} must have exactly one first-publish recovery check")
    recovery_index, recovery_step = recovery_steps[0]
    if recovery_step.get("id") != recovery_id:
        fail(f"{path}: {job_name} recovery step must have id {recovery_id}")
    recovery_run = recovery_step.get("run")
    if not isinstance(recovery_run, str):
        fail(f"{path}: {job_name} recovery step must be a run step")
    for required in (
        "already_published=true",
        "already_published=false",
        "Refusing recovery",
        f"crates.io already serves byte-identical {package}",
    ):
        if required not in recovery_run:
            fail(f"{path}: {job_name} recovery step must contain {required!r}")
    check_crate_recovery_status_handling(recovery_run, path, job_name, package)
    version = r"\$(?:\{VERSION\}|VERSION)"
    if not has_assignment(
        recovery_run, "METADATA_URL", rf"https://crates\.io/api/v1/crates/{package}/{version}"
    ):
        fail(f"{path}: {job_name} recovery step must define the per-version crates.io metadata URL")
    if not any(metadata_curl_uses(words) for words in shell_curl_commands(recovery_run)):
        fail(
            f"{path}: {job_name} recovery step must query $METADATA_URL "
            "with CRATES_IO_USER_AGENT and Accept: application/json into $METADATA"
        )
    for url_var in ("API_URL", "STATIC_URL"):
        if not any(
            recovery_curl_uses(words, url_var) for words in shell_curl_commands(recovery_run)
        ):
            fail(
                f"{path}: {job_name} recovery step must curl ${url_var} "
                "with CRATES_IO_USER_AGENT into $EXISTING, capture HTTP status, and retry"
            )

    protected_step_names = {
        "Mint a short-lived crates.io credential (OIDC)",
        "cargo publish",
    }
    if require_publish_dry_run:
        protected_step_names.add("Validate manifest publish dry-run")
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        name = step.get("name")
        if name in protected_step_names:
            if index < recovery_index:
                fail(f"{path}: {name} must run after the {package} crate recovery check")
            if step.get("if") != f"steps.{recovery_id}.outputs.already_published != 'true'":
                fail(
                    f"{path}: {name} must be skipped when {package} crate recovery found "
                    "byte-identical existing bytes"
                )

    if require_publish_dry_run:
        dry_run_index = publish_dry_runs[0][0]
        byte_identity_index = verify_step_indices["Verify byte-identity vs the attested .crate"]
        if dry_run_index < byte_identity_index:
            fail(f"{path}: {job_name} dry-run publish must run after byte-identity verification")
        if auth_steps and dry_run_index > min(auth_steps):
            fail(f"{path}: {job_name} dry-run publish must run before OIDC token minting")

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
    build_manifest_job = mapping(jobs.get("build-manifest-crate"), f"{path}: jobs.build-manifest-crate")
    if not has_need(build_manifest_job, "publish-crate"):
        fail(f"{path}: build-manifest-crate must need publish-crate so lockstep ordvec exists")
    build_manifest_steps = sequence(
        build_manifest_job.get("steps"), f"{path}: jobs.build-manifest-crate.steps"
    )
    build_manifest_packages = 0
    for index, raw_step in enumerate(build_manifest_steps):
        step = mapping(raw_step, f"{path}: jobs.build-manifest-crate.steps[{index}]")
        run = step.get("run")
        if isinstance(run, str) and has_cargo_command(run, "package", "ordvec-manifest"):
            build_manifest_packages += 1
    if build_manifest_packages != 1:
        fail(f"{path}: build-manifest-crate must package ordvec-manifest after publish-crate")

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
        require_publish_dry_run=True,
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


def check_sde_cache_job(
    workflow: dict[str, Any],
    path: str,
    job_name: str,
    *,
    expected_allow_unavailable: str,
    expected_notice_if: str | None,
    require_cache: bool,
    require_guarded_sde_steps: bool,
) -> None:
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

    if require_cache:
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
    else:
        if cache_steps:
            fail(f"{path}: jobs.{job_name} must not restore workflow caches in release context")
        for index, step in enumerate(steps):
            action = action_name(step)
            if action in {"actions/cache", "swatinem/rust-cache"}:
                fail(
                    f"{path}: {step_label(index, step)} must not use workflow caches "
                    "in the release fail-closed SDE proof"
                )

    if len(setup_steps) != 1:
        fail(f"{path}: jobs.{job_name} must use exactly one setup-intel-sde action")
    _, _, setup_with = setup_steps[0]
    if setup_with.get("version") != "${{ env.SDE_VERSION }}":
        fail(f"{path}: jobs.{job_name} setup-intel-sde must receive env.SDE_VERSION")
    if setup_with.get("sha256") != "${{ env.SDE_SHA256 }}":
        fail(f"{path}: jobs.{job_name} setup-intel-sde must receive env.SDE_SHA256")
    if setup_with.get("allow-unavailable") != expected_allow_unavailable:
        fail(
            f"{path}: jobs.{job_name} setup-intel-sde allow-unavailable must be "
            f"{expected_allow_unavailable!r}"
        )

    outage_notice_steps = [
        mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]")
        for index, raw_step in enumerate(steps)
        if contains_text(
            mapping(raw_step, f"{path}: jobs.{job_name}.steps[{index}]").get("run"),
            "Intel SDE archive unavailable",
        )
    ]
    if expected_notice_if is None:
        if outage_notice_steps:
            fail(f"{path}: jobs.{job_name} must not contain a soft-skip Intel SDE outage notice")
    else:
        matching_notices = [step for step in outage_notice_steps if step.get("if") == expected_notice_if]
        if len(matching_notices) != 1:
            fail(
                f"{path}: jobs.{job_name} must emit exactly one Intel SDE outage notice "
                f"guarded by {expected_notice_if!r}"
            )

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
            if require_guarded_sde_steps and step.get("if") != SDE_AVAILABLE_IF:
                fail(
                    f"{path}: {step_label(index, step)} must run after SDE setup succeeds, "
                    "and may be skipped only when SDE setup reports unavailable"
                )
            if not require_guarded_sde_steps and step.get("if") is not None:
                fail(
                    f"{path}: {step_label(index, step)} is in a release fail-closed SDE proof "
                    "and must not be guarded behind a green-skip condition"
                )


def check_sde_cache_invariants() -> None:
    check_sde_setup_action(SDE_ACTION_PATH)
    check_sde_cache_job(
        load_workflow(CI_WORKFLOW_PATH),
        CI_WORKFLOW_PATH,
        "avx512",
        expected_allow_unavailable=ROUTINE_CI_SDE_ALLOW_UNAVAILABLE,
        expected_notice_if=SDE_UNAVAILABLE_NOTICE_IF,
        require_cache=True,
        require_guarded_sde_steps=True,
    )
    check_sde_cache_job(
        load_workflow(COVERAGE_WORKFLOW_PATH),
        COVERAGE_WORKFLOW_PATH,
        "coverage",
        expected_allow_unavailable=ROUTINE_CI_SDE_ALLOW_UNAVAILABLE,
        expected_notice_if=SDE_UNAVAILABLE_NOTICE_IF,
        require_cache=True,
        require_guarded_sde_steps=True,
    )
    release_workflow = load_workflow(WORKFLOW_PATH)
    check_sde_cache_job(
        release_workflow,
        WORKFLOW_PATH,
        "release-avx512",
        expected_allow_unavailable=RELEASE_SDE_ALLOW_UNAVAILABLE,
        expected_notice_if=None,
        require_cache=False,
        require_guarded_sde_steps=False,
    )
    jobs = mapping(release_workflow.get("jobs"), f"{WORKFLOW_PATH}: jobs")
    draft_job = mapping(
        jobs.get("release-assets-draft"), f"{WORKFLOW_PATH}: jobs.release-assets-draft"
    )
    if not has_need(draft_job, "release-avx512"):
        fail(f"{WORKFLOW_PATH}: release-assets-draft must need release-avx512")


def main() -> None:
    workflow = load_workflow(WORKFLOW_PATH)
    ci_workflow = load_workflow(CI_WORKFLOW_PATH)
    check_release_version_sync()
    check_release_compatibility_sync()
    check_registry_metadata_parity()
    check_manifest_cli_defaults()
    check_publication_model()
    check_python_package_metadata()
    check_release_docs_include_manifest_pypi_lane()
    check_strict_release_tag_patterns(workflow, WORKFLOW_PATH)
    check_package_contents()
    check_ci_package_guards(ci_workflow, CI_WORKFLOW_PATH)
    check_hash_requirement_temp_paths(
        [WORKFLOW_PATH, PYTHON_WORKFLOW_PATH, CI_WORKFLOW_PATH, COVERAGE_WORKFLOW_PATH]
    )
    check_release_security_gates(workflow, WORKFLOW_PATH)
    check_aarch64_smoke_selector(workflow, WORKFLOW_PATH)
    check_pypi_canonical_dist(workflow, WORKFLOW_PATH)
    check_pypi_canonical_dist(
        workflow,
        WORKFLOW_PATH,
        job_name="pypi-manifest-canonical-dist",
        wheel_build_job="build-manifest-wheels",
        sdist_build_job="build-manifest-sdist",
        wheel_artifact_pattern="manifest-wheels-*",
        sdist_artifact_name="sdist-manifest",
        canonical_artifact_name="pypi-manifest-canonical-dist",
        project="ordvec-manifest",
        required_license_files=("LICENSE-MIT", "LICENSE-APACHE-2.0"),
    )
    check_publish_crates(workflow, WORKFLOW_PATH)
    check_ci_manifest_package_defer(load_workflow(CI_WORKFLOW_PATH), CI_WORKFLOW_PATH)
    check_publish_pypi(workflow, WORKFLOW_PATH)
    check_publish_pypi(
        workflow,
        WORKFLOW_PATH,
        job_name="publish-manifest-pypi",
        canonical_job="pypi-manifest-canonical-dist",
        canonical_artifact_name="pypi-manifest-canonical-dist",
        project="ordvec-manifest",
        crate_publish_job="publish-manifest-crate",
        required_license_files=("LICENSE-MIT", "LICENSE-APACHE-2.0"),
    )
    check_sde_cache_invariants()


if __name__ == "__main__":
    main()
