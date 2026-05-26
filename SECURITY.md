# Security Policy

## Supported versions

`ordvec` is pre-1.0; security fixes land on `main` and the latest published
release.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

Use GitHub's private vulnerability reporting:
**Security → Report a vulnerability**
(<https://github.com/Fieldnote-Echo/ordvec/security/advisories/new>).

We aim to acknowledge reports within a few business days.

`ordvec` parses serialized index files (`.tvr` / `.tvrq` / `.tvbm` /
`.tvsb`); the loaders are fuzzed (`cargo +nightly fuzz`), so
parsing-robustness reports against the deserialization paths are especially
welcome. Reports are also welcome against the `unsafe` SIMD kernels (shape /
bounds invariants), the Python FFI contract (buffer handling, GIL discipline),
and the release pipeline.

## Threat model

See [`THREAT_MODEL.md`](THREAT_MODEL.md) for the full attack-surface analysis —
existing defenses, known residual risks, and the library-owned vs
deployment-owned split.
