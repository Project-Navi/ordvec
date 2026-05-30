#!/usr/bin/env python3
"""Structural release publish invariants for registry upload jobs."""

from __future__ import annotations

import json
import os
import posixpath
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


def main() -> None:
    workflow = load_workflow(WORKFLOW_PATH)
    check_hash_requirement_temp_paths([WORKFLOW_PATH, PYTHON_WORKFLOW_PATH])
    check_aarch64_smoke_selector(workflow, WORKFLOW_PATH)
    check_pypi_canonical_dist(workflow, WORKFLOW_PATH)
    check_publish_crate(workflow, WORKFLOW_PATH)
    check_publish_pypi(workflow, WORKFLOW_PATH)


if __name__ == "__main__":
    main()
