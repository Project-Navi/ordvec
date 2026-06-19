---
name: custom-codereview-guide
description: ordvec-specific code review guidance for OpenHands PR reviews.
triggers:
  - /codereview
---

# ordvec Code Review Guidance

Prioritize correctness, security, release-contract drift, and behavioral
regressions. Avoid spending review budget on style nits unless they hide a real
maintenance or correctness risk.

For benchmark and documentation changes, verify that performance, memory, and
storage claims match the implementation and checked artifacts. A passing build
does not prove a benchmark claim.

For loaders, persisted formats, and manifest verification, check malformed-input
handling, exact length validation, resource limits, path confinement, and
cross-dispatch consistency. Safe Rust panics from externally supplied artifacts
should be treated as bugs.

For GitHub Actions and release changes, check least-privilege permissions,
pinned third-party actions, OIDC subject drift, required release invariants, and
whether a green workflow can hide skipped release-critical coverage.

When reviewing generated or agent-authored changes, verify the final code and
tests directly. Do not treat PR prose, bot summaries, or previous review
comments as proof that the issue is fixed.
