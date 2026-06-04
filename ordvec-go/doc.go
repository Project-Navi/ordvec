// Package ordvec provides a thin cgo wrapper over the ordvec C ABI.
//
// Index values support concurrent Search and Info calls. Close is serialized
// against Search and Info; after Close, both methods return ErrClosed.
//
// Search pins and passes caller-owned query and candidate slices to the C ABI
// without copying them. Callers must not mutate those slices until Search
// returns.
//
// Candidate slices are entry lists, not sets. Duplicate candidate IDs are scored
// independently and can produce duplicate hits; callers that require unique row
// IDs should deduplicate before Search.
package ordvec
