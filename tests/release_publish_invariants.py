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


WORKFLOW_PATH = os.environ.get("RELEASE_WORKFLOW_PATH", ".github/workflows/release.yml")
PYTHON_WORKFLOW_PATH = os.environ.get("PYTHON_WORKFLOW_PATH", ".github/workflows/python.yml")


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


def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        fail(f"{path}: could not read workflow: {exc}")


def shell_vars(name: str) -> set[str]:
    return {f"${name}", f"${{{name}}}"}


def shell_curl_commands(script: str) -> list[list[str]]:
    commands: list[list[str]] = []
    for line in shell_logical_lines(script):
        if "curl" not in line:
            continue
        if line.startswith("if "):
            line = line[3:].strip()
        line = line.split("; then", 1)[0].strip().rstrip(";")
        try:
            words = shlex.split(line)
        except ValueError:
            continue
        if words and words[0] == "curl":
            commands.append(words)
    return commands


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
    check_publish_crate_job(workflow, path, "publish-crate", "ordvec", "dist-crate")
    check_publish_crate_job(
        workflow,
        path,
        "publish-manifest-crate",
        "ordvec-manifest",
        "dist-manifest-crate",
    )


def main() -> None:
    workflow = load_workflow(WORKFLOW_PATH)
    check_hash_requirement_temp_paths([WORKFLOW_PATH, PYTHON_WORKFLOW_PATH])
    check_aarch64_smoke_selector(workflow, WORKFLOW_PATH)
    check_pypi_canonical_dist(workflow, WORKFLOW_PATH)
    check_publish_crates(workflow, WORKFLOW_PATH)
    check_publish_pypi(workflow, WORKFLOW_PATH)


if __name__ == "__main__":
    main()
