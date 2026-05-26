# Governance

ordvec is a small, actively maintained open-source project. This document
describes how it is run.

## Roles

- **Maintainer.** ordvec is currently maintained by Nelson Spence
  ([@Fieldnote-Echo](https://github.com/Fieldnote-Echo)), the project lead and
  final decision-maker on technical direction, releases, and scope.
- **Code owners.** Listed in [`.github/CODEOWNERS`](https://github.com/Fieldnote-Echo/ordvec/blob/main/.github/CODEOWNERS); they
  review and approve changes.

## Decision-making

- All changes land via **pull request** — direct pushes to `main` are blocked by
  branch protection.
- Every PR requires **passing CI** and at least **one approving review from a
  code owner other than the author**, with review conversations resolved before
  merge.
- Routine changes are decided by maintainer / code-owner review. Larger,
  direction-setting changes (scope, public API, dependencies) are discussed in
  an issue or pull request first; the maintainer makes the final call,
  consistent with the [roadmap](ROADMAP.md).
- Decisions favour the project's stated scope — ordvec is a retrieval
  _primitive_ for edge / local AI retrieval, **not** a standalone vector
  database (see [ROADMAP.md](ROADMAP.md)).

## Becoming a code owner

There is no formal membership process yet. Contributors who provide sustained,
high-quality contributions and reviews may be invited to become code owners.
The project is being opened specifically to grow this group — including
collaborators on the accompanying OrdVec / RankQuant paper.

## Contributing, conduct, and security

- **Contributing:** see [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are
  dual-licensed **MIT OR Apache-2.0**, matching the project license.
- **Code of conduct:** see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
  (Contributor Covenant).
- **Security:** report vulnerabilities privately per [SECURITY.md](SECURITY.md).
