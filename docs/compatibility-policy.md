# Pre-1.0 Compatibility Policy

`ordvec` is still pre-1.0, so minor releases may carry intentional breaking
changes. The project nevertheless treats downstream embedders as real users:
patch releases should be safe for stable surfaces, and any intentional break
must be visible in release notes before users discover it at build or load time.

This policy covers the published Rust crate, the PyPI bindings, repo-local
sidecars (C ABI, Go, and Manifest), the primitive persisted index formats, and
project examples and documentation. It does not promise a database or
application-store lifecycle outside `ordvec` itself.

## Versioning Rules

- **Patch releases (`0.x.y`)** do not intentionally break stable public
  surfaces in the same minor series. Bug fixes, safety hardening, deterministic
  tie-break fixes, and malformed-input rejection are allowed, but release notes
  must call out any user-visible behavior change.
- **Minor releases (`0.x.0`)** may make pre-1.0 breaking changes when they are
  documented in `CHANGELOG.md` with migration notes.
- **Experimental or hidden APIs** may change before 1.0 without a patch-stable
  contract. The release notes should still mention changes that are likely to
  affect known downstream users.
- **Security or corruption fixes** may reject inputs that older versions
  accepted when those inputs are malformed, ambiguous, or outside the documented
  contract. These are compatibility-impacting fixes, not stable-format breaks.

## Stability Buckets

### Published Rust Crate

The stable Rust surface is the default-feature API published as the `ordvec`
crate, excluding items marked `#[doc(hidden)]`, deprecated aliases, and APIs
behind explicitly experimental features.

Patch-stable APIs include the headline primitives and their documented
construction, mutation, search, metadata, and persistence methods:

- `Rank`, `RankQuant`, `Bitmap`, and `SignBitmap`;
- `new`, `add`, `search`, `search_asymmetric`, subset/candidate search helpers,
  `len`, `is_empty`, `dim`, `bytes_per_vec`, and `byte_size` methods where
  present;
- `RankQuant::bits`, `RankQuant::search_asymmetric_subset`, and
  `SignBitmap::top_m_candidates`, which are part of the current downstream
  `ordgrep` integration surface;
- `write` and `load` for the primitive index files;
- `SearchResults` fields and per-query accessors;
- `rank_io::probe_index_metadata`, `IndexKind`, `IndexParams`, and
  `IndexMetadata`.

Low-level rank helpers such as `rank_transform`, `rank_to_bucket`,
`pack_buckets`, and `rankquant_eval_search` are public and should not receive
source-breaking signature changes in patch releases. Numeric or ordering
semantics changes must be called out when user-visible.

Deprecated root aliases such as `RankQuantIndex` remain available until a
documented minor release removes them.

### Experimental and Hidden Rust APIs

The `experimental` feature is a default-off research surface. Today it exposes
`MultiBucketBitmap`; it is not patch-stable before 1.0.

`#[doc(hidden)]` exports such as `RankQuantFastscan` and
`search_asymmetric_byte_lut` are reachable for internal benchmarks and parity
tests, but they are not part of the stable default API.

New feature flags must declare their stability class before merging:

- stable default feature;
- stable default-off feature;
- optional dependency feature;
- experimental/default-off feature.

Changing the default feature set is a compatibility-impacting minor-release
decision unless the change is strictly additive and documented.

### Examples and Documentation

Examples and documentation are compatibility guidance, not executable API
surface. Patch releases should keep examples buildable and avoid changing
documented stable-surface behavior without release-note context. Corrections,
clarifications, and security warnings are allowed in patch releases when they
make the documented contract more accurate.

### Python Package

The PyPI package exposes the same four headline classes: `Rank`, `RankQuant`,
`Bitmap`, and `SignBitmap`. Patch releases should not intentionally break their
constructors, `add`, search methods, or returned `(scores, ids)` shapes.

Python, NumPy, or wheel-platform floor changes are minor-release changes unless
a supported upstream version has reached end-of-life or a security issue makes
the old floor unsafe. Such changes require release-note migration text.

### Repo-Local Sidecars (C ABI, Go, and Manifest)

`ordvec-ffi`, `ordvec-go`, and `ordvec-manifest` are repo-local sidecars, not
part of the published core `.crate`. They are still consumed by embedders from
the GitHub checkout, so their compatibility must be reviewed before releases.

The C ABI is versioned by `ORDVEC_ABI_VERSION`. ABI v1 currently supports
loading persisted `RankQuant` and `Bitmap` files, metadata inspection, and
synchronous single-query search. It does not expose builders, mutating index
APIs, `Rank`, or `SignBitmap`. Patch releases should preserve ABI v1 struct
layouts, init helpers, status values, capability bits, and documented
pointer/lifetime rules. Breaking ABI changes require a minor release and either
a new ABI version or clear migration notes.

The Go wrapper follows the C ABI. Source-breaking Go API changes require the
same compatibility classification in release notes.

The `ordvec-manifest` CLI and its v1 JSON schema are also treated as stable
repo-local surfaces. Patch releases should not introduce breaking changes to
the CLI arguments, emitted error codes, or JSON schema structure. Minor
releases may introduce schema or CLI updates with documented migration steps.

### Primitive Persisted Formats

The primitive index formats are the files written and loaded by the core index
types:

- `.tvr` / `TVR1` for `Rank`;
- `.tvrq` / `TVRQ` for `RankQuant`;
- `.tvbm` / `TVBM` for `Bitmap`;
- `.tvsb` / `TVSB` for `SignBitmap`.

Patch releases should keep valid files from the same minor series loadable.
Loader hardening may reject malformed files, forged sizes, trailing bytes, bad
dimensions, unsupported bit widths, or files outside documented capacity
limits. This bucket tracks the format-compatibility requirements from
[#118](https://github.com/Fieldnote-Echo/ordvec/issues/118).

Minor releases may introduce new format versions or new sidecar conventions.
When they do, release notes must say whether older files remain readable and
what migration, if any, a downstream store should perform. Older library
versions are not expected to be forward-compatible with newer format versions
and should safely reject them.

This is a primitive file-format promise. It does not define an application
database lifecycle, `.ordgrep` store schema, cache invalidation policy,
manifest trust policy, or migration framework for downstream systems.
Deployment-side provenance guidance lives in
[`INDEX_PROVENANCE.md`](https://github.com/Fieldnote-Echo/ordvec/blob/main/docs/INDEX_PROVENANCE.md).

## MSRV and Build Features

The Rust MSRV is Rust 1.89. Raising it is a minor-version compatibility change
and requires a reason in release notes. Keep `Cargo.toml` `rust-version`, the
README MSRV badge/section, and the CI MSRV job synchronized.

The core crate has no required system or numerical dependencies. Adding one, or
adding an optional dependency feature that changes build expectations for
embedders, requires explicit release-note classification.

## Release Review

Every release should include a compatibility-impact review:

- Identify whether the release is patch-compatible or minor-breaking.
- List touched stable Rust, Python, C ABI, Go, Manifest, persisted-format,
  examples/docs, feature, and MSRV surfaces.
- Add changelog migration notes for every intentional break.
- For patch releases, run a SemVer compatibility check against the latest
  published crate when practical, or record why an equivalent check was not
  useful for that release.
- Distinguish `ordvec` primitive API/file compatibility from downstream
  application database behavior.
