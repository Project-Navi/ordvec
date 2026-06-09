# ordvec-go

Thin cgo wrapper over the local `ordvec-ffi` C ABI.

Build the Rust library before running Go tests or linking a Go program:

```sh
cargo build -p ordvec-ffi --release
cd ordvec-go
go test ./...
go test -race ./...
GOEXPERIMENT=cgocheck2 go test ./...
```

`Index.Close` should be called explicitly. A finalizer is installed as a safety
net, but it is not a resource-management strategy.

Search with `nil` options or `nil` `SearchOptions.Candidates` performs a full
search. An empty, non-nil `Candidates` slice is treated as an explicit empty
subset and returns a typed `StatusBadArgument`, matching the C ABI v1
pointer/count contract.

Subset candidates are global row IDs. They may be unsorted and may contain
duplicates; duplicate entries are scored independently and can produce duplicate
hits. Deduplicate `SearchOptions.Candidates` before calling `Search` when unique
hits are required.
