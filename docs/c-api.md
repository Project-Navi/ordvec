# C API

`ordvec-ffi` exposes a small ABI v1 for loading persisted `.tvrq`
`RankQuant` and `.tvbm` `Bitmap` indexes and running synchronous single-query
searches. The public header is [`../ordvec-ffi/include/ordvec.h`](../ordvec-ffi/include/ordvec.h).

## Build and Link

Build the native library from the workspace:

```sh
cargo build -p ordvec-ffi --release
```

Compile C or C++ callers with the committed header and link either the shared
or static library from `target/release`:

```sh
cc -I ordvec-ffi/include app.c -L target/release -lordvec_ffi -o app
```

When linking dynamically, make sure your platform's loader can find
`libordvec_ffi.so`, `libordvec_ffi.dylib`, or `ordvec_ffi.dll`.

## Minimal Example

```c
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ordvec.h"

int main(void) {
    ordvec_index_t *index = NULL;
    ordvec_status_t st = ordvec_index_load("index.tvrq", 0, &index);
    if (st != ORDVEC_STATUS_OK) {
        fprintf(stderr, "load failed: %s\n", ordvec_last_error());
        return 1;
    }

    ordvec_index_info_t info;
    ordvec_index_info_init(&info);
    st = ordvec_index_info(index, &info);
    if (st != ORDVEC_STATUS_OK) {
        fprintf(stderr, "info failed: %s\n", ordvec_last_error());
        ordvec_index_free(index);
        return 1;
    }

    if (info.dim > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "index dimension is too large\n");
        ordvec_index_free(index);
        return 1;
    }

    float *query = calloc((size_t)info.dim, sizeof *query);
    if (query == NULL) {
        fprintf(stderr, "query allocation failed\n");
        ordvec_index_free(index);
        return 1;
    }

    ordvec_search_params_t params;
    ordvec_search_params_init(&params);
    params.query = query;
    params.dim = info.dim;
    params.k = 10;

    ordvec_hit_t hits[10];
    uint64_t returned = 0;
    ordvec_search_stats_t stats;
    ordvec_search_stats_init(&stats);

    st = ordvec_index_search(index, &params, hits, 10, &returned, &stats);
    if (st != ORDVEC_STATUS_OK) {
        fprintf(stderr, "search failed: %s\n", ordvec_last_error());
        free(query);
        ordvec_index_free(index);
        return 1;
    }

    for (uint64_t i = 0; i < returned; i++) {
        printf("row=%" PRIu64 " id=%" PRIu64 " score=%f\n",
               hits[i].row_id,
               hits[i].id,
               hits[i].score);
    }

    free(query);
    ordvec_index_free(index);
    return 0;
}
```

## ABI Contracts

All fallible functions return an `ordvec_status_t`. On success, they clear the
calling thread's `ordvec_last_error()` string. On failure, they set it to a
human-readable detail string. The pointer returned by `ordvec_last_error()` is
thread-local and valid until the next fallible `ordvec` C call on that same
thread.

Panics are caught and returned as `ORDVEC_STATUS_PANIC`; no Rust unwind crosses
the C ABI. The library does not install a global panic hook, so the Rust
default hook may still write panic diagnostics to stderr before the status is
returned.

Input structs must be initialized with their init helper and must have
`struct_size == sizeof(type)`. ABI v1 rejects larger forward-compatible structs
with `ORDVEC_STATUS_BAD_STRUCT_SIZE`. Unknown flags and nonzero reserved input
fields return `ORDVEC_STATUS_BAD_ARGUMENT`.

Search is synchronous. Caller pointers are borrowed only for the duration of
`ordvec_index_search`; no query, candidate, hit, stats, or path pointer is
retained after the function returns.

Rows are internal row ordinals. ABI v1 has no external ID map:
`ordvec_hit_t.id` is always equal to `ordvec_hit_t.row_id` widened to
`uint64_t`.

Hits are ordered by score descending, then row ID ascending. Candidate rows are
internal row ordinals and may be unsorted or duplicated. Duplicates are scored
as separate candidate entries and can produce duplicate hits.

## Search Modes

Full search requires:

- `candidate_count == 0`
- `candidate_rows == NULL`

Subset search requires:

- `candidate_count > 0`
- `candidate_rows != NULL`

`candidate_count == 0 && candidate_rows != NULL` returns
`ORDVEC_STATUS_BAD_ARGUMENT`. `candidate_count > 0 && candidate_rows == NULL`
returns `ORDVEC_STATUS_NULL_POINTER`.

Let `search_space_size` be the vector count for full search, or
`candidate_count` for subset search. `required_hits = min(k, search_space_size)`.
If `required_hits == 0`, `hits_out` may be `NULL` and `hits_capacity` may be
zero, but `returned_out` is still required and receives zero. If
`required_hits > 0`, `hits_out` must be non-null and `hits_capacity >=
required_hits`.

## Stats

If `stats_out` is non-null, it must be initialized with
`ordvec_search_stats_init`. On successful search, ABI v1 fills:

- `abi_version`, `kind`, `dim`, `bit_width`, `n_top`
- `k`, `user_tag`
- `vector_count`
- `candidate_count`
- `returned_count`
- `total_ns`
- `vectors_scored`

`candidate_count` and `vectors_scored` count search-space entries, not unique
rows. For full search this is the index vector count; for subset search this is
the candidate entry count, including duplicates. `prepare_ns`, `score_ns`,
`select_ns`, and byte/counter fields are reserved and currently zero.

## Threading

Concurrent searches and info calls on one loaded handle are allowed.
`ordvec_index_free` must not race with any other call on the same handle.
`ordvec_index_free(NULL)` is a no-op. Use after free and double free are
undefined behavior.

## V1 Exclusions

ABI v1 intentionally excludes `Rank`, `SignBitmap`, external IDs, ID maps,
builders, mutating index APIs, logging callbacks, custom allocators, async
search, batched search, richer measured timing breakdowns, and release
packaging. Those can be added in later ABI versions without changing the v1
struct-size rule.
