# Bindings Safety and Ownership

This is the cross-language contract for embedders using the Rust crate, Python
package, C ABI, or Go wrapper. It consolidates the binding notes that otherwise
live near each implementation. It does not add a new runtime policy: callers
still own scheduling, path trust, input mutability, and deployment provenance.

## Concurrency

`ordvec` is read-concurrent and mutation-exclusive.

- Rust index values can be searched concurrently through shared references.
  Mutation methods such as `add` require exclusive access.
- Python search, candidate-generation, scoring, and `add` methods release the
  GIL while Rust performs the heavy work. PyO3 still enforces object borrow
  rules, and the binding copies NumPy input arrays into Rust-owned buffers
  before releasing the GIL.
- The C ABI permits concurrent `ordvec_index_search`,
  `ordvec_index_probe`, and `ordvec_index_info` calls on one loaded handle.
  `ordvec_index_free` must not race with any other call on that handle.
- The Go wrapper serializes `Close` against `Search` and `Info`; after
  `Close`, both methods return `ErrClosed`.

## Borrowed Inputs

Caller-provided Rust slices, C buffers, and Go slices are borrowed for the
duration of the call and are not retained after the function returns. Python
NumPy inputs that cross a GIL-released call are copied before the GIL is
released.

- Do not mutate Rust slices, C buffers, or Go slices while a call that received
  them is in progress.
- Query, corpus, candidate, output, hit, and stats buffers remain caller-owned
  unless a specific API says otherwise.
- Candidate lists are entry lists, not sets. Duplicate candidate IDs are scored
  independently, count toward candidate and vector-scored statistics, and can
  produce duplicate hits. Deduplicate before calling when unique row IDs or
  waste-free scoring matter.

## Returned Memory

Current C ABI search calls do not return heap-owned result buffers. Callers
allocate and retain ownership of `hits_out`, `returned_out`, and `stats_out`.
`ordvec_index_load` returns an opaque handle that must be freed exactly once
with `ordvec_index_free` after all concurrent calls using that handle have
finished; ABI v1 has no general `ordvec_free` for result memory.

The Go wrapper copies C search hits into Go-owned slices. It frees temporary
`C.CString` values internally and releases the C index handle through `Close`
or its finalizer; Go callers do not free C result buffers.

## Rows and External IDs

Core search results use internal row ordinals. The primitive persisted formats
do not carry an application ID map.

`ordvec-manifest` can bind an application-owned ID sidecar as a required
auxiliary artifact, but the primitive Rust, C, Go, and Python search paths still
return row ordinals. Host systems should maintain their own row-to-application
ID map and verify it together with the index when crossing a trust boundary.

## Paths and Trust

`write` and `load` paths are trusted input. The core crate, Python binding, C
ABI, and Go wrapper forward paths to the filesystem without path traversal
sanitization or sandboxing.

Services that derive paths from user input should canonicalize and constrain
paths before calling `ordvec`, or use an application storage layer that never
exposes raw path choice to callers. Resolve paths against an allowed base
directory after symlink resolution, then reject any resolved path outside that
base. In Rust, use `std::fs::canonicalize`; in Python, use `pathlib.Path.resolve`;
in Go, combine lexical cleanup such as `filepath.Clean` with symlink-aware
resolution such as `filepath.EvalSymlinks`. For artifact integrity and sidecar
binding, use `ordvec-manifest`; it verifies hashes, declared metadata,
auxiliary artifacts, and attestation shape, but it does not sign files or
decide key policy.

## Errors and Panics

- The Rust crate keeps fail-loud panicking constructors and methods where that
  is the documented API. Existing `try_*` helpers return `OrdvecError` only
  where explicitly provided.
- Python validates dimensions, dtypes, contiguity where required, finite
  values, candidate ranges, and capacities at the boundary so common invalid
  inputs raise typed Python exceptions instead of surfacing opaque Rust panics.
- The C ABI catches Rust panics at status-returning FFI boundaries and returns
  `ORDVEC_STATUS_PANIC`; no Rust unwind crosses the ABI boundary. The same
  thread's `ordvec_last_error()` is set to the panic payload when it is a string
  or to fallback panic text otherwise. Successful status-returning calls clear
  that thread-local error. Non-status helpers such as `ordvec_last_error()`,
  version/status accessors, and `ordvec_index_free` do not report fallible
  status.
- The Go wrapper maps C status values to Go errors and preserves the C ABI
  pointer and lifetime rules.

## Release Review Checklist

When a change touches a binding, review these questions before release:

- Does the change preserve read-concurrent, mutation-exclusive behavior?
- Are borrowed buffers still borrowed only for the documented call duration?
- Are path-trust assumptions unchanged or documented?
- Are row ordinals, duplicate candidates, and result shapes still described
  consistently across Rust, Python, C, and Go?
- If a validation rule changes, is it a documented hardening fix rather than a
  silent compatibility break?
