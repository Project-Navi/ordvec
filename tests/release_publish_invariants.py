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
CI_WORKFLOW_PATH = os.environ.get("CI_WORKFLOW_PATH", ".github/workflows/ci.yml")
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


def shell_vars(name: str) -> set[str]:
    return {f"${name}", f"${{{name}}}"}


def shell_logical_lines(script: str) -> list[str]:
    lines: list[str] = []
    current = ""
    for raw_line in script.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            current += line[:-1].strip() + " "
            continue
        current += line
        if current:
            lines.append(current.strip())
        current = ""
    if current:
        lines.append(current.strip())
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


def check_publish_crate(workflow: dict[str, Any], path: str) -> None:
    jobs = mapping(workflow.get("jobs"), f"{path}: jobs")
    job = mapping(jobs.get("publish-crate"), f"{path}: jobs.publish-crate")
    steps = sequence(job.get("steps"), f"{path}: jobs.publish-crate.steps")

    crate_downloads: list[tuple[int, dict[str, Any], dict[str, Any]]] = []

    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.publish-crate.steps[{index}]")
        if action_name(step) != "actions/download-artifact":
            continue
        with_block = step.get("with", {})
        with_map = mapping(with_block, f"{path}: {step_label(index, step)} with")
        if with_map.get("name") == "dist-crate":
            crate_downloads.append((index, step, with_map))

    if len(crate_downloads) != 1:
        fail(f"{path}: publish-crate must download exactly one dist-crate artifact")

    index, step, with_map = crate_downloads[0]
    label = step_label(index, step)
    artifact_path = norm_path(with_map.get("path"))
    if artifact_path != "${{ runner.temp }}/attested":
        fail(
            f"{path}: {label} downloads dist-crate to {artifact_path or 'the default path'!r}; "
            "it must use ${{ runner.temp }}/attested so cargo package sees a clean checkout"
        )

    verify_step_names = {
        "Verify byte-identity vs the attested .crate",
        "Post-publish byte-identity (download from crates.io == attested)",
    }
    verify_steps: list[dict[str, Any]] = []
    found_names: set[str] = set()
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"{path}: jobs.publish-crate.steps[{index}]")
        name = step.get("name")
        if name in verify_step_names:
            verify_steps.append(step)
            found_names.add(name)
    if found_names != verify_step_names:
        fail(f"{path}: publish-crate must have both attested .crate verification steps")

    for step in verify_steps:
        name = step.get("name")
        run = step.get("run")
        if not isinstance(run, str):
            fail(f"{path}: publish-crate step {name!r} must be a run step")
        if "${RUNNER_TEMP}/attested/ordvec-${VERSION}.crate" not in run:
            fail(
                f"{path}: publish-crate step {name!r} must read the attested .crate "
                "from ${RUNNER_TEMP}/attested"
            )
        if name == "Post-publish byte-identity (download from crates.io == attested)":
            if "/tmp/published.crate" in run:
                fail(f"{path}: publish-crate post-publish readback must not write to /tmp")
            if not (
                "${RUNNER_TEMP}/published.crate" in run or "$RUNNER_TEMP/published.crate" in run
            ):
                fail(f"{path}: publish-crate post-publish readback must write under ${{RUNNER_TEMP}}")
            version = r"\$(?:\{VERSION\}|VERSION)"
            if not has_assignment(
                run, "API_URL", rf"https://crates\.io/api/v1/crates/ordvec/{version}/download"
            ):
                fail(f"{path}: publish-crate post-publish readback must define the crates.io API URL")
            if not has_assignment(
                run, "STATIC_URL", rf"https://static\.crates\.io/crates/ordvec/ordvec-{version}\.crate"
            ):
                fail(f"{path}: publish-crate post-publish readback must define the static.crates.io fallback")

            curl_commands = shell_curl_commands(run)
            if not any(readback_curl_uses(words, "API_URL") for words in curl_commands):
                fail(
                    f"{path}: publish-crate post-publish readback must curl $API_URL "
                    "with CRATES_IO_USER_AGENT into $PUBLISHED"
                )
            if not any(readback_curl_uses(words, "STATIC_URL") for words in curl_commands):
                fail(
                    f"{path}: publish-crate post-publish readback must curl $STATIC_URL "
                    "with CRATES_IO_USER_AGENT into $PUBLISHED"
                )


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
    check_hash_requirement_temp_paths(
        [WORKFLOW_PATH, PYTHON_WORKFLOW_PATH, CI_WORKFLOW_PATH, COVERAGE_WORKFLOW_PATH]
    )
    check_aarch64_smoke_selector(workflow, WORKFLOW_PATH)
    check_pypi_canonical_dist(workflow, WORKFLOW_PATH)
    check_publish_crate(workflow, WORKFLOW_PATH)
    check_publish_pypi(workflow, WORKFLOW_PATH)
    check_sde_cache_invariants()


if __name__ == "__main__":
    main()
