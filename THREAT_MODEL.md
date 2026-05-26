# Threat Model — `ordvec`

> **Status:** v0.2.0 (pre-1.0), 2026-05-25. This is the maintained threat model
> for the `ordvec` Rust crate and the `ordvec` PyO3/maturin Python bindings. It
> is reviewed when the attack surface changes (new persistence formats, new
> `unsafe` kernels, new FFI surface, or release-pipeline changes).
>
> Scope discipline: `ordvec` is a **pure computational library** — no network
> surface, no authentication/authorization, no secrets handling, no
> multi-tenancy of its own. This document deliberately does **not** enumerate
> web-application threats (SQLi/XSS/CSRF/session) that do not apply. It covers
> the surfaces that actually exist: untrusted-input parsing, `unsafe` SIMD, the
> Python FFI boundary, the supply chain, and resource use under untrusted
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
  floats, dimension mismatches, shape errors) — panic in Rust, `ValueError`
  in Python.
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

`ordvec` is maintained by a single primary contributor. Mitigations are
prioritized when they are (1) low-maintenance once merged, (2) enforceable by
tests or CI, (3) local to the library boundary, and (4) unlikely to add
operational burden downstream. Heavyweight controls (mandatory index signing,
long-running fuzz farms, service-level admission control) are documented as
**deployment guidance** until there is maintainer capacity to own them. The
absence of a second maintainer is itself a tracked supply-chain residual
(see THREAT-SUPPLY-001).

---

## 1. Architecture and trust boundaries

### 1.1 Component map

| Layer | Components | Trust boundary |
|---|---|---|
| **Deserialization** | `rank_io.rs` — `.tvr` / `.tvrq` / `.tvbm` / `.tvsb` loaders | Untrusted filesystem / network byte stream |
| **Compute kernels** | `fastscan.rs`, `quant_kernels.rs`, `bitmap.rs`, `sign_bitmap.rs` | Trust established after format validation |
| **Index API** | `rank.rs`, `quant.rs`, `bitmap.rs`, `sign_bitmap.rs` | Caller-controlled query embeddings |
| **Python FFI** | `ordvec-python` (PyO3 / maturin) | Python ↔ Rust boundary; NumPy buffers |
| **CI / supply chain** | 12 GitHub Actions workflows; `Cargo.lock`; crates.io + PyPI | GitHub OIDC, crates.io, PyPI trust chains |

The `fuzz/` directory holds **seven** cargo-fuzz targets: `load_rank`,
`load_rankquant`, `load_bitmap`, `load_sign_bitmap` (deserialization);
`roundtrip_rankquant` (write→load round-trip); `search_rankquant` (the
single-rate ingest + asymmetric-search compute path); and `fastscan_b2` (the
FastScan b=2 block-32 kernel — the one `unsafe`-heavy scan path the others do
not reach).

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
  histogram); `Bitmap` rows must have exactly `n_top` bits set.
- No `panic!` on malformed data — all validation returns
  `io::Error(InvalidData)`.
- The raw `rank_io` read/write functions are `pub(crate)`; the only public
  persistence API is the index types' `write()` / `load()`, making the
  write→load round-trip a type-level guarantee.

The four loaders are continuously fuzzed (`cargo fuzz`, the `load_*` targets).

### 2.2 Index-file risk classes

**THREAT-DESER-001 (library-owned, P4): Malformed index file.**
The loader must reject corrupt/invalid files without panic, OOM, or
trailing-data acceptance. The current implementation satisfies this for all
four formats. *Residual:* `file.metadata()?.len()` is sampled at open time;
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
for deployments where index files cross trust boundaries. An optional sidecar
verifier (HMAC / BLAKE3) can be added later without a format bump; it is
deliberately **not** shipped now (no concrete deployment requires it, and an
in-format crypto layer would add unowned key management).

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

**THREAT-SIMD-001 (P1, mitigated this cycle; crate-wide rollout tracked):
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
**`#![deny(unsafe_op_in_unsafe_fn)]` is now enforced in `fastscan.rs`**, so
every unsafe operation in the kernel sits in an explicit `unsafe {}` block and
stays visible to future edits. *Open:* roll the lint out crate-wide to the
other SIMD modules (`bitmap.rs`, `sign_bitmap.rs`, `quant_kernels.rs`,
`util.rs` NEON) — tracked as a follow-up.

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
This is approximate *scoring*, not a CPU oracle. FastScan is a `#[doc(hidden)]`
pre-ranker; callers needing exact scores use `RankQuant::search_asymmetric`.

---

## 4. Python FFI threats (THREAT-FFI) — binding-owned

### 4.1 Existing defenses (code-verified)

The binding takes `PyReadonlyArray`, rejects non-C-contiguous arrays with a
clear `ValueError`, validates finiteness (`ensure_finite`), maps shape errors
to `ValueError`, and releases the GIL (`py.detach`) around the pure-Rust
(Rayon-parallel) compute in every heavy method while reading the input arrays
in place. PyO3's `&mut self` borrow tracking means a second thread re-entering
the **same** index object during a released-GIL call gets a clean
`Already borrowed` `RuntimeError`, never concurrent mutation.

### 4.2 Risks (documented contracts, implemented)

**THREAT-FFI-001 (P2, documented): Concurrent input-array mutation during a
released-GIL call.** `PyReadonlyArray` keeps the input buffer alive and blocks
`rust-numpy`-mediated writes for the call's duration, but it cannot stop
another thread or native extension from mutating the *same backing memory*
through a reference obtained before the call. This can yield numerically
inconsistent results — a numeric-extension contract issue, not a UAF. *Status:*
documented in the module docstring and the per-method docs ("do not mutate an
input array from another thread while an `ordvec` call is in progress"),
matching the standard contract for GIL-releasing NumPy extensions. An optional
`safe_copy=True` hard-isolation parameter remains a possible future ergonomic.

**THREAT-FFI-002 (P2, documented): Unsanitized filesystem-path forwarding.**
`write()` / `load()` forward the path to the filesystem unmodified (no `..` /
traversal sanitization). A service exposing these path arguments to user input
could enable traversal or arbitrary-file overwrite. This is a **caller
responsibility**. *Status:* documented in the module docstring and on every
`write`/`load` method ("treat the path as trusted input; web/multi-user
applications must validate paths before calling"). 

---

## 5. Supply-chain threats (THREAT-SUPPLY)

### 5.1 Existing controls (verified)

**Workflow code (all 12 workflows):** third-party actions pinned by commit
SHA; `persist-credentials: false` on every checkout; `permissions: contents:
read` default. **Release workflows** (`release-crate.yml`, `release-python.yml`)
are `workflow_dispatch`-only (no tag/push trigger), run a `require-ci-green`
gate against `main`, publish via **OIDC trusted publishing** (no long-lived
registry tokens), and emit **SLSA build provenance**
(`actions/attest-build-provenance`) **before** publish — a failed attestation
fails the release closed. `release-python` additionally gets **PEP 740**
attestations via Trusted Publishing.

**Static / supply-chain analysis:** **CodeQL** scans Rust, Python, and Actions
(no-build databases); **OpenSSF Scorecard** publishes SARIF to code scanning
and the score badge; **zizmor** audits workflow hardening (pinned); a
`cargo-deny` / audit job gates advisories and licenses. The core crate has near
zero non-Rust dependencies by design (the `deps` gate greps `cargo tree -p
ordvec`); the Python binding's larger tree (numpy → ndarray) is intentional and
scoped to the wheel.

### 5.2 Risks

**THREAT-SUPPLY-001 (mitigated; residual = single-maintainer account
compromise): Release configuration and ownership.** The release **environments**
(`pypi`, `crates-io`) now require **approval by the maintainer** and restrict
deployment to the **`main`** branch only — so a release cannot be dispatched
from an unmerged or attacker branch, and no publish runs without an explicit
human approval. The remaining residual is *maintainer-account compromise*: a
single owner is both dispatcher and approver, so account takeover (or social
engineering) is not caught by a second human. *Mitigations:* strong 2FA /
passkeys on the maintainer account; recruiting a **second owner/maintainer**
(also an open OpenSSF Best-Practices item) — which would additionally make a
deployment **wait timer** worthwhile (a second party able to cancel a bad
release during the window). See [`RELEASING.md`](RELEASING.md).

**THREAT-SUPPLY-002 (P3): Release immutability and tag integrity.** Published
artifacts are **immutable by registry design** — crates.io is yank-only (a
published version's bytes can never be overwritten) and PyPI burns a version on
delete (no different artifact may be re-uploaded under the same version). So
post-publish "silent replacement" of a version is not possible on either
registry, and consumers can verify artifacts against the SLSA / PEP 740
provenance above. *Residual (GitHub-side):* `changelog.yml` cuts tagged GitHub
Releases, but the repo currently has **no tag-protection ruleset and no `main`
ruleset**, so a tag could be force-moved or a release asset replaced.
*Mitigation:* add a `v*` **tag ruleset** (block update + deletion) and a basic
`main` ruleset; optionally enable GitHub immutable releases.

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

Seven targets cover the four loaders, the write→load round-trip, the
single-rate compute path, and (new) the FastScan kernel.

**THREAT-FUZZ-001 (closed this cycle): FastScan path was unfuzzed.** The
`fastscan_b2` target now drives `RankQuantFastscan` (`pack_fastscan_b2` +
`search_asymmetric_fastscan_b2` + the scalar/AVX-512 kernel), crossing the
32-doc block boundary so tail-padding blocks are exercised. On
non-AVX-512 CI runners it exercises the scalar reference kernel; under Intel SDE
it exercises the AVX-512 kernel.

**THREAT-FUZZ-002 (P3): No CI-bound fuzzing for continuous regression.** Fuzzing
is run manually; there is no CI gate. A bounded weekly smoke job (e.g.
`-runs=50000` on `load_rank`, `load_rankquant`, and `fastscan_b2`) would catch
regressions between manual runs. (Low overhead; weighed against maintenance
budget.)

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
| THREAT-SIMD-001 | Memory safety | Library | Unsafe-kernel invariant bypass on refactor | Medium | High | **P1** — lint enforced in `fastscan.rs`; crate-wide rollout tracked |
| THREAT-FFI-001 | FFI | Binding | Concurrent input mutation during released-GIL call | Medium | Medium | **P2** — documented contract |
| THREAT-FFI-002 | FFI | Binding | Unsanitized path forwarding | Medium | Medium | **P2** — documented contract |
| THREAT-SUPPLY-001 | Supply chain | Config | Release config / single-owner | Low | Critical | **Mitigated** (reviewer + main-only); residual = account compromise / 2nd owner |
| THREAT-SUPPLY-002 | Supply chain | Config | Release immutability / tag integrity | Low | High | **P3** — registries immutable; add tag ruleset |
| THREAT-SUPPLY-003 | Supply chain | Config | Typosquatting adjacent names | Medium | Medium | P3 |
| THREAT-QUERY-001 | Resource | Deployment | Batch / `k` exhaustion in serving | Medium | Medium | **P2** — deployment docs |
| THREAT-QUERY-002 | Resource | Deployment | Panic on contract violation (Rust servers) | Low | Medium | P3 |
| THREAT-FUZZ-001 | Fuzzing | Library | FastScan path unfuzzed | Medium | High | **Closed** (`fastscan_b2` added) |
| THREAT-FUZZ-002 | Fuzzing | Library | No CI-bound fuzzing | Medium | Medium | P3 |
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

**Done this cycle:** `#![deny(unsafe_op_in_unsafe_fn)]` in `fastscan.rs`
(SIMD-001); `fastscan_b2` fuzz target (FUZZ-001); release-environment reviewers
+ main-only deployment (SUPPLY-001); [`docs/INDEX_PROVENANCE.md`](docs/INDEX_PROVENANCE.md)
(DESER-002); [`RELEASING.md`](RELEASING.md) (SUPPLY-001).

**Open, low cost:**

1. Add a `v*` tag-protection ruleset (+ basic `main` ruleset) and optionally
   enable GitHub immutable releases (THREAT-SUPPLY-002).
2. Roll `#![deny(unsafe_op_in_unsafe_fn)]` out crate-wide across the remaining
   SIMD modules (THREAT-SIMD-001).
3. Add a bounded weekly CI fuzz smoke job (THREAT-FUZZ-002).
4. Document recommended `nq` / `k` / corpus bounds for single-process serving
   in the Rust and Python API docs (THREAT-QUERY-001).

**Later (not release blockers):** a second maintainer/owner (then a release
wait timer becomes meaningful); an optional sidecar index verifier
(`ordvec verify` / external HMAC/BLAKE3 manifest) if a deployment requires
tamper-evidence (DESER-002); a `safe_copy=True` FFI isolation option
(FFI-001).

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
