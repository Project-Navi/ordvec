# Threat Model — `ordvec`

> **Status:** v0.5.0 (pre-1.0), 2026-06-15. This is the maintained threat model
> for the `ordvec` Rust crate, C ABI, Go wrapper, PyO3/maturin Python bindings,
> and the `ordvec-manifest` sidecar verifier. It is reviewed when the
> attack surface changes (new persistence formats, new `unsafe` kernels, new
> FFI surface, or release-pipeline changes).
>
> Scope discipline: `ordvec` is a **pure computational library** — no network
> surface, no authentication/authorization, no secrets handling, no
> multi-tenancy of its own. This document deliberately does **not** enumerate
> web-application threats (SQLi/XSS/CSRF/session) that do not apply. It covers
> the surfaces that actually exist: untrusted-input parsing, `unsafe` SIMD, the
> C/Python FFI boundaries, the supply chain, and resource use under untrusted
> callers. Deployment-owned risks (corpus trust, co-tenancy, admission control)
> are documented as *context* for integrators, not as library action items.

See also: [`SECURITY.md`](SECURITY.md) (reporting), [`RELEASING.md`](RELEASING.md)
(release controls), [`docs/INDEX_PROVENANCE.md`](docs/INDEX_PROVENANCE.md)
(what the loaders do and do not guarantee).

---

## Scope and security ownership

**`ordvec` owns:**

- Memory safety of all safe public APIs.
- Robust rejection of malformed serialized index files — no panic, no OOM
  abort, no silent data corruption, no trailing-data acceptance.
- Deterministic, finite-input behavior for valid embeddings.
- Clear, documented failure contracts for invalid caller input (non-finite
  floats, dimension mismatches, shape errors) — panic in Rust, typed status
  codes in C/Go, `ValueError` in Python.
- Supply-chain hygiene for the published crate and Python wheels.

**`ordvec` does not own:**

- Trustworthiness of the upstream embedding model.
- Corpus provenance or document-level poisoning.
- Authorization over which documents may be indexed or retrieved.
- Tenant isolation or microarchitectural isolation on a hosting platform.
- Cryptographic verification of index-file origin (callers add this externally
  — see [`docs/INDEX_PROVENANCE.md`](docs/INDEX_PROVENANCE.md)).

> A structurally valid index file can still be semantically malicious. The
> loaders validate format invariants — not truth, authorization, or corpus
> integrity.

## Maintenance budget

`ordvec` has one project lead plus an additional maintainer / release
approver. Mitigations are prioritized when they are (1) low-maintenance once
merged, (2) enforceable by tests or CI, (3) local to the library boundary, and
(4) unlikely to add operational burden downstream. Heavyweight controls
(mandatory index signing, long-running fuzz farms, service-level admission
control) are documented as **deployment guidance** unless the project has
maintainer capacity to own them. Release publication requires a non-triggering
approver through protected GitHub Environments; the residual release
supply-chain risk is approver account compromise / collusion, not a
single-owner project structure (see THREAT-SUPPLY-001).

---

## 1. Architecture and trust boundaries

### 1.1 Component map

| Layer | Components | Trust boundary |
|---|---|---|
| **Deserialization** | `rank_io.rs` — `.ovr` / `.ovrq` / `.ovbm` / `.ovsb` / `.ovfs` loaders (`.ovfs`/`OVFS` is the FastScan format and has no legacy magic; the other four also accept the legacy `.tvr` / `.tvrq` / `.tvbm` / `.tvsb` magics) | Untrusted filesystem / network byte stream |
| **Manifest verification** | `ordvec-manifest` — JSON sidecar verifier | Manifest + index + optional row-map files before load |
| **Compute kernels** | `fastscan.rs`, `quant_kernels.rs`, `bitmap.rs`, `sign_bitmap.rs` | Trust established after format validation |
| **Index API** | `rank.rs`, `quant.rs`, `bitmap.rs`, `sign_bitmap.rs` | Caller-controlled query embeddings |
| **C ABI** | `ordvec-ffi` (`include/ordvec.h`) | C caller ↔ Rust boundary; raw pointers and opaque handles |
| **Go FFI** | `ordvec-go` (cgo over `ordvec-ffi`) | Go slices ↔ synchronous C ABI calls |
| **Python FFI** | `ordvec-python` (PyO3 / maturin) | Python ↔ Rust boundary; NumPy buffers |
| **CI / supply chain** | GitHub Actions workflows; `Cargo.lock`; crates.io + PyPI | GitHub OIDC, crates.io, PyPI trust chains |

The `fuzz/` directory holds **nine** cargo-fuzz targets: `load_rank`,
`load_rankquant`, `load_bitmap`, `load_sign_bitmap`, `load_fastscan`
(deserialization — the last drives the `.ovfs`/`OVFS` FastScan loader via
`RankQuantFastscan::load`); `roundtrip_rankquant` (write→load round-trip);
`search_rankquant` (the single-rate ingest + asymmetric-search compute path);
`fastscan_b2` (the FastScan b=2 block-32 kernel — the one `unsafe`-heavy scan
path the others do not reach); and `signbitmap_rankquant_twostage` (sign
candidate generation followed by RankQuant subset reranking).

### 1.2 Deployment contexts (for integrators)

- **Offline / batch indexing** — a trusted operator encodes a corpus and writes
  index files. Low risk unless files later cross a trust boundary.
- **Serving pipeline** — an index loaded at startup, then queried by
  user-controlled embeddings. Query vectors cross the trust boundary on every
  search call (see §6).
- **RAG substrate** — `ordvec` retrieves the *k* nearest documents fed to an
  LLM. The retrieval layer becomes a target for corpus-level poisoning; this is
  a **deployment risk**, not a parser risk (see §7).
- **Multi-tenant / cloud** — tenants sharing one process share SIMD execution
  units. Microarchitectural isolation is a hosting-platform responsibility
  (see THREAT-SIMD-002).

---

## 2. Deserialization threats (THREAT-DESER) — library-owned

### 2.1 Existing defenses (code-verified)

`rank_io.rs` implements layered parser hardening:

- Magic + version checks before any allocation.
- Fallible allocation via `try_reserve_exact` — an attacker-controlled length
  field returns `InvalidData`, never an OOM abort.
- All payload sizes computed with `usize::checked_mul`; overflow returns `Err`.
- A 128 GiB `MAX_PAYLOAD` cap and `MAX_VECTORS` (64 Mi) / `MAX_DIM` caps,
  enforced on **both** the load and write paths (the write-side cap runs
  *before* `File::create`, so a rejected write cannot truncate an existing
  file).
- Exact file-length match (`check_payload_matches_file`): trailing bytes or
  short files are rejected.
- Per-row **structural** invariants: `Rank` rows must be a true permutation of
  `[0, dim)` (verified by bound + duplicate checks ⇒ pigeonhole);
  `RankQuant` rows must satisfy constant composition (uniform per-bucket
  histogram); `Bitmap` rows must have exactly `n_top` bits set;
  `RankQuantFastscan` `.ovfs` rows must use valid FastScan nibbles, satisfy
  b=2 constant composition, and zero block-tail padding.
- No `panic!` on malformed data — all validation returns
  `io::Error(InvalidData)`.
- The raw `rank_io` read/write functions are `pub(crate)`; the only public
  persistence API is the index types' `write()` / `load()`, making the
  write→load round-trip a type-level guarantee.

The five loaders are covered by cargo-fuzz targets (the `load_*` targets,
including `load_fastscan` for the `.ovfs` FastScan format).

### 2.2 Index-file risk classes

**THREAT-DESER-001 (library-owned, P4): Malformed index file.**
The loader must reject corrupt/invalid files without panic, OOM, or
trailing-data acceptance. The current implementation satisfies this for all
five formats. *Residual:* `file.metadata()?.len()` is sampled at open time;
on NFS/FUSE mounts with concurrent writers a TOCTOU window exists between
`metadata()` and the reads. On writable shared mounts the practical outcome is
a read error or `InvalidData`, not an exploit. *Likelihood:* Very Low.
*Impact:* error surfaced. 

**THREAT-DESER-002 (deployment-owned, P3 docs): Malicious-but-valid index.**
A structurally valid index with semantically poisoned contents passes every
parser check and returns attacker-influenced results. This is a *provenance*
problem, not a parser problem. *Mitigation (no format change):*
[`docs/INDEX_PROVENANCE.md`](docs/INDEX_PROVENANCE.md) documents that `ordvec`
validates structure, not origin, and lists verification options (checksum
manifest, artifact-store integrity, Sigstore / GitHub artifact attestation)
for deployments where index files cross trust boundaries. The repo now includes
`ordvec-manifest`, a sidecar verifier that binds an index file to
JSON manifest metadata by SHA-256, allocation-resistant header probing, strict
row identity checks, and attestation shape checks. It deliberately does **not**
sign, manage keys, call networks, mutate index files, change the C ABI, or
decide trust policy; an in-format crypto layer is still not shipped because it
would add unowned key management.

---

## 3. Unsafe SIMD and memory-safety threats (THREAT-SIMD) — library-owned

### 3.1 What the FastScan kernel does

`scan_b2_fastscan_avx512` uses unaligned loads (`_mm256_loadu_si256`),
byte-shuffle LUT lookups (`_mm256_shuffle_epi8` / VPSHUFB), broadcast, widen
(`_mm256_cvtepu8_epi16`, `_mm512_cvtepu16_epi32`), and accumulate
(`_mm512_add_epi16/epi32`, `_mm512_storeu_si512`). It is a load/shuffle/widen/
accumulate sequence with **no gather instructions**. The Intel DOWNFALL (GDS)
vulnerability is specific to gather-based data sampling and does **not** apply
to this kernel.

### 3.2 Risks

**THREAT-SIMD-001 (P1, mitigated this cycle):
Unsafe-kernel invariant preservation under future refactors.**
`scan_b2_fastscan_avx512` safety depends on caller-established invariants —
`packed_fs.len() == n_blocks * pairs * 32` (formed via `checked_mul`, overflow
⇒ caller panics) and `lut_u8.len() == pairs * 16`. These are asserted by the
`pub(crate)` entry point `search_asymmetric_fastscan_b2` before dispatch, and
`RankQuantFastscan::search` is the type-level safe wrapper that owns the shape
by construction. A future refactor calling the inner function directly could
bypass the asserts. *Mitigations:* the runtime asserts + the type wrapper are
the primary boundary; the scalar-vs-SIMD equivalence test
(`fastscan_b2_top10_matches_avx512_kernel`) guards behavior; and
**`#![deny(unsafe_op_in_unsafe_fn)]` is now enforced crate-wide** (at the crate
root in `lib.rs`), so every unsafe operation in every SIMD kernel —
`fastscan.rs`, `bitmap.rs`, `sign_bitmap.rs`, `quant_kernels.rs`, and the
`util.rs` NEON popcount — sits in an explicit `unsafe {}` block and stays
visible to future edits. (The lone exception, `horizontal_sum_avx2`, is
register-only with no memory access, so its intrinsics are safe under the
`#[target_feature]` gate and an explicit block would be `unused_unsafe`.)

**THREAT-SIMD-002 (P4, deployment note): Microarchitectural side channels in
co-tenancy.** `ordvec` does not claim protection against microarchitectural
side channels under hostile multi-tenant co-residency. The kernel uses no
gather instructions (ruling out DOWNFALL/GDS), but SIMD execution units are
shared across SMT threads, and port-contention timing channels remain
theoretically possible on vulnerable hardware. Sensitive deployments should
avoid sharing physical cores across trust domains and rely on the
OS/hypervisor side-channel posture. Not a library action item.

**THREAT-SIMD-003 (P3): FastScan approximation is not CPU-dependent
divergence.** The 8-bit global-affine LUT in `build_fastscan_b2_query`
introduces `O(span/255)` per-pair approximation error — an intentional
trade-off matching FAISS FastScan semantics, documented in the code. The
scalar and AVX-512 paths agree on the same quantized inputs (equivalence test),
and `TopK` uses `total_cmp` for deterministic tie-breaking across all paths.
This is approximate *scoring*, not a CPU oracle. FastScan is a stable
specialized pre-ranker; callers needing exact scores use
`RankQuant::search_asymmetric`.

**THREAT-SIMD-004 (mitigated this cycle): Native sanitizer coverage for
unsafe kernels.** `.github/workflows/sanitizers.yml` runs nightly
AddressSanitizer tests with `-Zsanitizer=address` and `-Z build-std` on
native x86_64 and Linux/aarch64. The x86_64 leg instruments the scalar/AVX2
surfaces plus the repo-local C ABI tests; the aarch64 leg instruments the
NEON path on a native ARM runner. This deliberately does not claim AVX-512
sanitizer coverage: GitHub-hosted runners still need Intel SDE to execute
those kernels, and layering ASAN onto the existing SDE leg remains a follow-up.

---

## 4. FFI threats (THREAT-FFI) — binding-owned

### 4.1 C ABI defenses (code-verified)

`ordvec-ffi` exposes only loaded `.ovrq` `RankQuant` and `.ovbm` `Bitmap`
indexes (legacy `.tvrq` / `.tvbm` files also load) through one opaque handle. The ABI checks raw pointer nullness and
caller-supplied lengths before use, requires exact v1 `struct_size` values for
input structs, rejects unknown flags and nonzero reserved input fields,
validates query dimension and finiteness before entering core search,
bounds-checks every candidate row before any subset scorer runs, and requires
caller-owned output buffers large enough for `min(k, search_space_size)`.

Every fallible entry point is wrapped in `catch_unwind`, maps panics to
`ORDVEC_STATUS_PANIC`, and stores a thread-local error detail for the caller.
Successful fallible calls clear that thread-local error. The ABI does not log
queries, row IDs, paths, stats, or errors; stats are local output structs only.
Concurrent search/info calls may share a handle, but `ordvec_index_free` must
not race with any other call.

The C ABI is designed for thin higher-level wrappers that preserve the same
lifetime contract. In the stacked Go-wrapper PR, the repo-local wrapper
serializes `Search`/`Info` against `Close`, copies C-owned results into Go
values, treats `Close` as idempotent, returns `ErrClosed` after close, and uses
the C ABI only synchronously. Those wrapper-specific mitigations are
code-verified in that PR.

**THREAT-FFI-001 (P1, mitigated): Panic or invalid input crossing the C ABI.**
Malformed C calls must return status codes rather than unwind into C or read
past caller buffers. *Mitigations:* exact-size input structs, pointer/order
validation, row bounds checks, output-capacity checks, `catch_unwind`, Rust ABI
tests for failure paths, and C/C++ header compile smoke tests. *Residual:*
passing an invalid non-null pointer is still undefined behavior, as in any C
ABI; the library can validate nullness and sizes, not pointer provenance.

**THREAT-FFI-002 (P2, documented): Handle lifetime misuse.**
`ordvec_index_free(NULL)` is a no-op, but double free, use after free, or
freeing a handle while another thread is searching are undefined behavior.
*Mitigation:* documented contract in `docs/c-api.md`. The stacked Go wrapper PR
serializes `Close` against `Search`/`Info` and adds a finalizer safety net,
while still requiring explicit `Close`.

**THREAT-FFI-003 (P3, mitigated): Accidental telemetry through ABI stats.**
Search stats could become a logging side channel if the library emitted them
globally. *Mitigation:* ABI v1 has no callbacks or global logging; stats are
written only to caller-provided memory and contain aggregate counters/timings,
not query values or hit contents.

### 4.2 Python defenses (code-verified)

The binding takes `PyReadonlyArray`, rejects non-C-contiguous arrays with a
clear `ValueError`, validates finiteness (`ensure_finite`), maps shape errors
to `ValueError`, and releases the GIL (`py.detach`) around the pure-Rust
(Rayon-parallel) compute in every heavy method while reading the input arrays
in place. PyO3's `&mut self` borrow tracking means a second thread re-entering
the **same** index object during a released-GIL call gets a clean
`Already borrowed` `RuntimeError`, never concurrent mutation.

### 4.3 Python risks (documented contracts, implemented)

**THREAT-FFI-004 (P2, documented): Concurrent input-array mutation during a
released-GIL call.** `PyReadonlyArray` keeps the input buffer alive and blocks
`rust-numpy`-mediated writes for the call's duration, but it cannot stop
another thread or native extension from mutating the *same backing memory*
through a reference obtained before the call. This can yield numerically
inconsistent results — a numeric-extension contract issue, not a UAF. *Status:*
documented in the module docstring and the per-method docs ("do not mutate an
input array from another thread while an `ordvec` call is in progress"),
matching the standard contract for GIL-releasing NumPy extensions. An optional
`safe_copy=True` hard-isolation parameter remains a possible future ergonomic.

**THREAT-FFI-005 (P2, documented): Unsanitized filesystem-path forwarding.**
`write()` / `load()` forward the path to the filesystem unmodified (no `..` /
traversal sanitization). A service exposing these path arguments to user input
could enable traversal or arbitrary-file overwrite. This is a **caller
responsibility**. *Status:* documented in the module docstring and on every
`write`/`load` method ("treat the path as trusted input; web/multi-user
applications must validate paths before calling"). 

---

## 5. Supply-chain threats (THREAT-SUPPLY)

### 5.1 Existing controls (verified)

**Workflow code (all workflows):** third-party actions pinned by commit SHA
(the one mandated exception is the SLSA reusable workflow, which the SLSA
trust model requires be pinned by version *tag*); `persist-credentials: false`
on every checkout; `permissions: contents: read` default. The **release
workflow** (`release.yml`) is tag-triggered with a strict-SemVer guard; build,
GitHub attestation, SLSA provenance, Release-asset attach, and un-draft all
run automatically, while the two **`crates.io`** publish jobs (`publish-crate`
for `ordvec` first, then `publish-manifest-crate` for lockstep
`ordvec-manifest`) and the two **`pypi`** publish jobs (`publish-pypi` and
`publish-manifest-pypi`) are gated behind GitHub Environments with **Required
reviewers** (the only manual step). It runs a `require-ci-green` gate against
current `main` HEAD, publishes via **OIDC trusted publishing** (no long-lived
registry tokens), and emits **SLSA build
provenance** (`actions/attest-build-provenance` + a `slsa-github-generator`
`*.intoto.jsonl` attached to the GitHub Release) **before** publish — a failed
attestation fails the release closed. Each Rust publish job proves pre- and
post-publish crates.io byte identity against the attested `.crate`; PyPI
additionally gets **PEP 740** attestations via Trusted Publishing.

**Static / supply-chain analysis:** **CodeQL** scans Rust, Python, and Actions
(no-build databases); **OpenSSF Scorecard** publishes SARIF to code scanning
and the score badge; **zizmor** audits workflow hardening (pinned); a
`cargo-deny` / audit job gates advisories and licenses. The core crate has near
zero non-Rust dependencies by design (the `deps` gate greps `cargo tree -p
ordvec`); the Python binding's larger tree (numpy → ndarray) is intentional and
scoped to the wheel.

### 5.2 Risks

**THREAT-SUPPLY-001 (mitigated; residual = release-approver account
compromise / collusion): Release configuration and ownership.** The release
**environments** (`pypi`, `crates-io`) list `Fieldnote-Echo` and `toadkicker` as
required reviewers, enable **prevent self-review**, enforce a **30-minute wait
timer**, and restrict deployment to the **release-tag pattern
`v[0-9]*.[0-9]*.[0-9]*`** (the tag-triggered workflow runs on
`refs/tags/...`, not `refs/heads/main`, so a branch-only allowlist would
deadlock publishing — see RELEASING.md). The `require-ci-green` gate
independently verifies the tag SHA has a successful push-event CI run on `main`,
and `main` itself is branch-protected (PR review, no force-push) — so a release
cannot be cut from an unmerged or attacker branch, and no publish runs without
an explicit human approval by a listed release approver who did not trigger the
deployment. The remaining residual is compromise or misuse of an eligible
approver account, or collusion between release participants. *Mitigations:*
strong 2FA / passkeys on both approver accounts, a small reviewed approver list,
and the 30-minute deployment window for the non-triggering approver to inspect
or cancel a bad release. See [`RELEASING.md`](RELEASING.md).

**THREAT-SUPPLY-002 (mitigated): Release immutability and tag integrity.**
Published artifacts are **immutable by registry design** — crates.io is
yank-only (a published version's bytes can never be overwritten) and PyPI burns
a version on delete (no different artifact may be re-uploaded under the same
version). So post-publish "silent replacement" of a version is not possible on
either registry, and consumers can verify artifacts against the SLSA / PEP 740
provenance above. The GitHub-side mutability surface is now closed too:
`release.yml` cuts tagged GitHub Releases, and **GitHub immutable releases is
enabled**, so a published release's `v*` tag cannot be force-moved or deleted
and its assets cannot be replaced after publication; the **`main` branch is
protected** (pull-request review required, force-pushes and deletions blocked)
and is the **only branch a release-tag commit can reside on**: each release
environment (`pypi`, `crates-io`) policies "Deployment branches and tags" to
the tag pattern `v[0-9]*.[0-9]*.[0-9]*`, and `require-ci-green` independently
verifies the tag SHA has a successful push-event CI run on `main` — a SHA
that only exists via a PR merge to the protected branch. *Residual:* draft / non-release tags are not covered by
release immutability, and — as with the registries — these GitHub controls
ultimately trust the release approver set; that residual folds into
THREAT-SUPPLY-001.

**THREAT-SUPPLY-003 (P3): Typosquatting adjacent names.** Namespace-adjacent
crate/package names (`ord-vec`, `ordvecs`, `order-vec`) could be registered to
typosquat dependents. *Mitigation:* publish the first functional release
promptly; optionally register adjacent names.

---

## 6. Query and resource-exhaustion threats (THREAT-QUERY) — library-adjacent

These arise from correct behavior on large-but-valid inputs from untrusted
callers, not from parser or unsafe bugs.

**THREAT-QUERY-001 (P2, deployment docs): Caller-controlled batch / `k`
exhaustion.** `result_buffer_len(nq, k)` checks `nq * k` overflow and panics
loudly rather than under-allocating; `k` is clamped to `n_vectors`. But a
serving application can still be CPU/memory-exhausted by large query batches
(`nq`), large `k`, or concurrent scans over a large corpus. `ordvec` does not
enforce service-level quotas — by design (it is a library, not a server).
*Mitigation:* callers exposing search over a network must independently bound
batch size, `k`, request rate, and corpus size; a configurable `max_nq` /
`max_k` at the binding level is a possible future convenience.

**THREAT-QUERY-002 (P3): Panic on contract violation in Rust server contexts.**
Rust APIs fail fast on invalid contract input (non-finite floats, dimension /
shape violations) via `assert!` / `expect`. In a Rust-native server an
unhandled panic crashes the thread/process; the Python bindings convert these
to typed `ValueError`. *Mitigation:* Rust service callers must validate
untrusted input before calling, or catch panics at the request boundary.

---

## 7. Corpus and embedding poisoning (THREAT-POISON) — deployment-owned

These sit **outside** the library's security perimeter; they are documented as
context for integrators using `ordvec` as a RAG substrate. Corpus poisoning of
embedding retrievers is a documented attack class (see PoisonedRAG and OWASP
LLM08:2025 in the references); the mitigations are corpus provenance, ingestion
access control, and (where applicable) hybrid lexical + vector retrieval — all
deployment concerns. The points below are the `ordvec`-specific shape of that
class.

**THREAT-POISON-001: Ordinal rank inversion.** Because `ordvec` is
training-free, the rank transform is deterministic and invertible. An attacker
who controls the embedding pipeline can engineer an embedding whose ordinal
(Spearman) correlation with target queries is maximized — the ordinal analogue
of embedding-inversion attacks. `ordvec` has no codebook to protect and cannot
prevent construction of maximally correlated embeddings; mitigation requires
access control and provenance on the embedding source.

**THREAT-POISON-002: Top-`n_top` overlap poisoning.** `Bitmap` scores documents
by `popcount(Q AND D)`. The loader enforces exactly `n_top` bits per row, so an
injected document cannot set arbitrary bits — the realistic attack is crafting a
document whose top-`n_top` coordinates maximally overlap the most-queried
coordinates. Requires knowledge of the query distribution and corpus write
access.

**THREAT-POISON-003: RankQuant boundary exploitation.** `RankQuant` uses
equal-width bucket quantization; documents near bucket boundaries can be crafted
to score highly under the coarse pre-filter yet differ under exact reranking,
exploiting quantization information loss to pass the coarse stage. Requires
knowledge of quantization parameters and the document distribution.

---

## 8. Fuzzing coverage (THREAT-FUZZ)

Nine targets cover the five loaders, the write→load round-trip, the
single-rate compute path, the FastScan kernel, and the composed
SignBitmap→RankQuant retrieval path.

**THREAT-FUZZ-001 (closed this cycle): FastScan path was unfuzzed.** The
`fastscan_b2` target now drives `RankQuantFastscan` (`pack_fastscan_b2` +
`search_asymmetric_fastscan_b2` + the scalar/AVX-512 kernel), crossing the
32-doc block boundary so tail-padding blocks are exercised. On
non-AVX-512 CI runners it exercises the scalar reference kernel; under Intel SDE
it exercises the AVX-512 kernel. The `load_fastscan` target also follows every
successful `.ovfs` load with a safe `search()` call so loader-accepted bytes
must survive the public scan path.

**THREAT-FUZZ-002 (mitigated this cycle): CI-bound fuzzing for continuous
regression.** A `fuzz.yml` workflow now runs a bounded smoke on every pull
request and push to `main` (`-max_total_time=60` over `load_rank`,
`load_rankquant`, `fastscan_b2`, and `signbitmap_rankquant_twostage`) plus a
weekly full sweep (`-max_total_time=300` over all nine targets), so a
regression that
reintroduces a loader panic / OOM, breaks the write→load round-trip, or
destabilises the FastScan kernel or composed sign→RankQuant path surfaces in CI
rather than only at the next manual campaign. cargo-fuzz is version-pinned and
the actions are SHA-pinned, matching the repo's scheduled-workflow hardening.

*Note on `load_sign_bitmap`:* all bit patterns are structurally valid for sign
bitmaps (no per-row invariant), so that target is correctly scoped to parser
robustness — no OOM, no panic, no trailing-data acceptance.

---

## 9. CI/CD pipeline threats (THREAT-CICD)

**THREAT-CICD-001 (P3, mitigated by control): Workflow injection via PR
metadata.** If a `run:` step interpolated user-controlled context (PR title,
branch name) into a shell expression via `${{ ... }}` without an `env:` hop, a
script-injection could run in the runner. *Mitigation:* `zizmor` audits exactly
this class of issue and runs in CI; pass user-controlled context through `env:`
rather than inline `${{ }}` in `run:` blocks. SHA-pinned actions bound the
blast radius of a compromised dependency separately.

---

## 10. Threat register

| ID | Category | Owner | Description | Likelihood | Impact | Status / priority |
|---|---|---|---|---|---|---|
| THREAT-SIMD-001 | Memory safety | Library | Unsafe-kernel invariant bypass on refactor | Medium | High | **Mitigated** — `unsafe_op_in_unsafe_fn` denied crate-wide + type wrapper + equivalence test |
| THREAT-SIMD-004 | Memory safety | Library | Native sanitizer coverage for unsafe kernels | Medium | High | **Mitigated** — ASAN on x86_64 scalar/AVX2 + aarch64 NEON; AVX-512 SDE+ASAN deferred |
| THREAT-FFI-001 | FFI | Binding | Panic or invalid input crossing C ABI | Medium | High | **Mitigated** — status codes, validation, `catch_unwind` |
| THREAT-FFI-002 | FFI | Caller | Handle lifetime misuse | Medium | High | **P2** — documented contract; stacked Go wrapper serializes `Close` |
| THREAT-FFI-003 | FFI | Binding | Accidental telemetry through ABI stats | Low | Low | **Mitigated** — caller-owned stats, no logging |
| THREAT-FFI-004 | FFI | Binding | Concurrent input mutation during released-GIL call | Medium | Medium | **P2** — documented contract |
| THREAT-FFI-005 | FFI | Binding | Unsanitized path forwarding | Medium | Medium | **P2** — documented contract |
| THREAT-SUPPLY-001 | Supply chain | Config | Release config / dual-approver gate | Low | Critical | **Mitigated** (two approvers, self-review blocked, 30-minute wait timer, `require-ci-green` main-SHA gate); residual = approver compromise / collusion |
| THREAT-SUPPLY-002 | Supply chain | Config | Release immutability / tag integrity | Low | High | **Mitigated** — registries immutable; GitHub immutable releases on + `main` protected |
| THREAT-SUPPLY-003 | Supply chain | Config | Typosquatting adjacent names | Medium | Medium | P3 |
| THREAT-QUERY-001 | Resource | Deployment | Batch / `k` exhaustion in serving | Medium | Medium | **P2** — deployment docs |
| THREAT-QUERY-002 | Resource | Deployment | Panic on contract violation (Rust servers) | Low | Medium | P3 |
| THREAT-FUZZ-001 | Fuzzing | Library | FastScan path unfuzzed | Medium | High | **Closed** (`fastscan_b2` added) |
| THREAT-FUZZ-002 | Fuzzing | Library | No CI-bound fuzzing | Medium | Medium | **Mitigated** — `fuzz.yml` PR smoke + weekly sweep |
| THREAT-DESER-001 | Deserialization | Library | TOCTOU on shared mounts | Very Low | Low | P4 |
| THREAT-DESER-002 | Provenance | Deployment | Malicious-but-valid index | Medium | High | P3 (docs — `INDEX_PROVENANCE.md`) |
| THREAT-CICD-001 | CI/CD | Library | Workflow injection via PR metadata | Low | High | P3 — mitigated by `zizmor` |
| THREAT-SIMD-002 | Side channel | Deployment | Microarchitectural co-tenancy (no gather) | Low | Medium | P4 |
| THREAT-SIMD-003 | Semantic | Library | FastScan approximation (doc clarity) | Low | Low | P3 |
| THREAT-POISON-001 | Index poisoning | Deployment | Ordinal rank inversion | Medium | High | Deployment |
| THREAT-POISON-002 | Index poisoning | Deployment | Top-`n_top` overlap poisoning | Low | Medium | Deployment |
| THREAT-POISON-003 | Index poisoning | Deployment | RankQuant boundary exploitation | Low | Low | Deployment |

---

## 11. Open mitigations

**Done this cycle:** `#![deny(unsafe_op_in_unsafe_fn)]` enforced **crate-wide**
across all SIMD modules (SIMD-001); the `fastscan_b2` fuzz target (FUZZ-001)
plus a CI `fuzz.yml` — PR smoke + weekly sweep (FUZZ-002); the `rank_to_bucket`
primitive made fail-loud (`rank < d`) to match the rest of the bucket API, with
matching binding guards; reviewer-gated release-tag deployment plus the
`require-ci-green` main-SHA gate (SUPPLY-001); **GitHub immutable releases
enabled + `main` branch protection**
(SUPPLY-002); [`docs/INDEX_PROVENANCE.md`](docs/INDEX_PROVENANCE.md) (DESER-002);
[`RELEASING.md`](RELEASING.md) (SUPPLY-001); ASAN coverage for native
x86_64/aarch64 unsafe paths (SIMD-004).

**Open, low cost:**

1. Document recommended `nq` / `k` / corpus bounds for single-process serving
   in the Rust and Python API docs (THREAT-QUERY-001).

**Later (not release blockers):** stronger deployment-specific manifest
trust-policy UX such as external signatures/HMACs if a deployment requires
tamper-evidence beyond `ordvec-manifest`'s hash-bound sidecar verification
(DESER-002); a `safe_copy=True` FFI isolation option (FFI-001); layering ASAN
onto the Intel SDE AVX-512 leg.

---

## References

Only load-bearing, verifiable sources are listed.

- **PoisonedRAG** — *Knowledge Corruption Attacks to Retrieval-Augmented
  Generation of Large Language Models* (arXiv:2402.07867). Establishes that
  injecting a small number of poisoned passages into a retriever corpus
  achieves high attack-success rates — context for §7.
- **OWASP LLM08:2025 — Vector and Embedding Weaknesses.** Retrieval-layer risk
  class (poisoning, embedding inversion, access-control bypass) — context for
  §7 / scope.
- **"Memory-Safety Challenge Considered Solved? An In-Depth Study with All Rust
  CVEs"** (arXiv:2003.03296). Real-world Rust memory-safety bugs require
  `unsafe` code — the rationale for the §3 focus on the SIMD kernels.
- **GitHub Security Lab — preventing pwn-requests.** Expression-injection in
  `run:` steps and untrusted-context handling — basis for THREAT-CICD-001.
