package ordvec

/*
#cgo CFLAGS: -I${SRCDIR}/../ordvec-ffi/include
#cgo linux LDFLAGS: ${SRCDIR}/../target/release/libordvec_ffi.a -ldl -lm -lpthread
#cgo darwin LDFLAGS: ${SRCDIR}/../target/release/libordvec_ffi.a -lm -lpthread
#cgo windows LDFLAGS: -L${SRCDIR}/../target/release -lordvec_ffi -lws2_32 -lbcrypt -luserenv
#include <stdlib.h>
#include "ordvec.h"

static ordvec_status_t ordvec_go_index_search(
    const ordvec_index_t* index,
    const float* query,
    uint64_t dim,
    uint64_t k,
    const uint32_t* candidate_rows,
    uint64_t candidate_count,
    uint64_t user_tag,
    ordvec_hit_t* hits_out,
    uint64_t hits_capacity,
    uint64_t* returned_out,
    ordvec_search_stats_t* stats_out
) {
    ordvec_search_params_t params;
    ordvec_search_params_init(&params);
    params.query = query;
    params.dim = dim;
    params.k = k;
    params.candidate_rows = candidate_rows;
    params.candidate_count = candidate_count;
    params.user_tag = user_tag;
    return ordvec_index_search(index, &params, hits_out, hits_capacity, returned_out, stats_out);
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

type Status uint32

const (
	StatusOK                   Status = C.ORDVEC_STATUS_OK
	StatusNullPointer          Status = C.ORDVEC_STATUS_NULL_POINTER
	StatusBadArgument          Status = C.ORDVEC_STATUS_BAD_ARGUMENT
	StatusBadStructSize        Status = C.ORDVEC_STATUS_BAD_STRUCT_SIZE
	StatusUnsupportedFormat    Status = C.ORDVEC_STATUS_UNSUPPORTED_FORMAT
	StatusCorruptIndex         Status = C.ORDVEC_STATUS_CORRUPT_INDEX
	StatusIO                   Status = C.ORDVEC_STATUS_IO
	StatusDimMismatch          Status = C.ORDVEC_STATUS_DIM_MISMATCH
	StatusNonfiniteQuery       Status = C.ORDVEC_STATUS_NONFINITE_QUERY
	StatusRowIDOutOfRange      Status = C.ORDVEC_STATUS_ROW_ID_OUT_OF_RANGE
	StatusBufferTooSmall       Status = C.ORDVEC_STATUS_BUFFER_TOO_SMALL
	StatusUnsupportedOperation Status = C.ORDVEC_STATUS_UNSUPPORTED_OPERATION
	StatusPanic                Status = C.ORDVEC_STATUS_PANIC
	StatusInternal             Status = C.ORDVEC_STATUS_INTERNAL
)

type Kind uint32

const (
	KindUnknown   Kind = C.ORDVEC_INDEX_KIND_UNKNOWN
	KindRankQuant Kind = C.ORDVEC_INDEX_KIND_RANK_QUANT
	KindBitmap    Kind = C.ORDVEC_INDEX_KIND_BITMAP
)

const (
	CapFullSearch    uint64 = C.ORDVEC_CAP_FULL_SEARCH
	CapSubsetSearch  uint64 = C.ORDVEC_CAP_SUBSET_SEARCH
	CapStats         uint64 = C.ORDVEC_CAP_STATS
	CapIDEqualsRowID uint64 = C.ORDVEC_CAP_ID_EQUALS_ROW_ID
)

var ErrClosed = errors.New("ordvec: index closed")

type StatusError struct {
	Status  Status
	Message string
}

func (e *StatusError) Error() string {
	if e.Message == "" {
		return fmt.Sprintf("ordvec: %s", e.Status)
	}
	return fmt.Sprintf("ordvec: %s: %s", e.Status, e.Message)
}

func (s Status) String() string {
	return C.GoString(C.ordvec_status_name(C.ordvec_status_t(s)))
}

func ABIVersion() uint32 {
	return uint32(C.ordvec_abi_version())
}

func Version() string {
	ptr := C.ordvec_version_string()
	if ptr == nil {
		return ""
	}
	return C.GoString(ptr)
}

type Info struct {
	Kind                Kind
	FormatVersion       uint32
	Dim                 uint64
	BitWidth            uint32
	NTop                uint32
	VectorCount         uint64
	BytesPerVec         uint64
	SourceFileSizeBytes uint64
	Capabilities        uint64
}

type Hit struct {
	RowID uint64
	ID    uint64
	Score float32
}

type Stats struct {
	ABIVersion     uint32
	Kind           Kind
	Dim            uint64
	BitWidth       uint32
	NTop           uint32
	K              uint64
	UserTag        uint64
	VectorCount    uint64
	CandidateCount uint64
	ReturnedCount  uint64
	TotalNS        uint64
	PrepareNS      uint64
	ScoreNS        uint64
	SelectNS       uint64
	VectorsScored  uint64
	BytesRead      uint64
}

type SearchOptions struct {
	// Candidates is an optional subset of global row IDs. Entries may be
	// unsorted and may contain duplicates; duplicate entries are scored
	// independently and can produce duplicate hits.
	Candidates []uint32
	UserTag    uint64
}

type Index struct {
	mu   sync.RWMutex
	ptr  *C.ordvec_index_t
	info Info
}

var emptyCandidateSentinel uint32

func statusError(st C.ordvec_status_t) error {
	status := Status(st)
	if status == StatusOK {
		return nil
	}
	msg := C.GoString(C.ordvec_last_error())
	if msg == "" {
		msg = status.String()
	}
	return &StatusError{Status: status, Message: msg}
}

func callStatus(fn func() C.ordvec_status_t) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	st := fn()
	return statusError(st)
}

func Probe(path string) (Info, error) {
	if strings.IndexByte(path, 0) >= 0 {
		return Info{}, errors.New("ordvec: path contains null byte")
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var ci C.ordvec_index_info_t
	C.ordvec_index_info_init(&ci)
	err := callStatus(func() C.ordvec_status_t {
		return C.ordvec_index_probe(cpath, 0, &ci)
	})
	if err != nil {
		return Info{}, err
	}
	return infoFromC(ci), nil
}

func Load(path string) (*Index, error) {
	if strings.IndexByte(path, 0) >= 0 {
		return nil, errors.New("ordvec: path contains null byte")
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	var out *C.ordvec_index_t
	err := callStatus(func() C.ordvec_status_t {
		return C.ordvec_index_load(cpath, 0, &out)
	})
	if err != nil {
		return nil, err
	}
	idx := &Index{ptr: out}
	info, err := idx.infoLocked()
	if err != nil {
		C.ordvec_index_free(out)
		return nil, err
	}
	idx.info = info
	runtime.SetFinalizer(idx, (*Index).finalize)
	return idx, nil
}

func (idx *Index) finalize() {
	_ = idx.Close()
}

func (idx *Index) Close() error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.ptr == nil {
		return nil
	}
	C.ordvec_index_free(idx.ptr)
	idx.ptr = nil
	runtime.SetFinalizer(idx, nil)
	return nil
}

func (idx *Index) Info() (Info, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.ptr == nil {
		return Info{}, ErrClosed
	}
	return idx.info, nil
}

func (idx *Index) infoLocked() (Info, error) {
	var ci C.ordvec_index_info_t
	C.ordvec_index_info_init(&ci)
	err := callStatus(func() C.ordvec_status_t {
		return C.ordvec_index_info(idx.ptr, &ci)
	})
	runtime.KeepAlive(idx)
	if err != nil {
		return Info{}, err
	}
	return infoFromC(ci), nil
}

func infoFromC(ci C.ordvec_index_info_t) Info {
	return Info{
		Kind:                Kind(ci.kind),
		FormatVersion:       uint32(ci.format_version),
		Dim:                 uint64(ci.dim),
		BitWidth:            uint32(ci.bit_width),
		NTop:                uint32(ci.n_top),
		VectorCount:         uint64(ci.vector_count),
		BytesPerVec:         uint64(ci.bytes_per_vec),
		SourceFileSizeBytes: uint64(ci.source_file_size_bytes),
		Capabilities:        uint64(ci.capabilities),
	}
}

func (idx *Index) Search(query []float32, k uint64, opts *SearchOptions) ([]Hit, Stats, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.ptr == nil {
		return nil, Stats{}, ErrClosed
	}

	searchSpace := idx.info.VectorCount
	if opts != nil && opts.Candidates != nil {
		searchSpace = uint64(len(opts.Candidates))
	}
	required := k
	if searchSpace < required {
		required = searchSpace
	}
	if required > uint64(int(^uint(0)>>1)) {
		return nil, Stats{}, fmt.Errorf("ordvec: required hit count %d overflows int", required)
	}

	var pinner runtime.Pinner
	defer pinner.Unpin()

	var queryPtr *C.float
	if len(query) > 0 {
		pinner.Pin(&query[0])
		queryPtr = (*C.float)(unsafe.Pointer(&query[0]))
	}
	var candidateRows *C.uint32_t
	var candidateCount C.uint64_t
	var userTag C.uint64_t
	if opts != nil {
		userTag = C.uint64_t(opts.UserTag)
		if opts.Candidates != nil {
			candidateCount = C.uint64_t(len(opts.Candidates))
			if len(opts.Candidates) > 0 {
				pinner.Pin(&opts.Candidates[0])
				candidateRows = (*C.uint32_t)(unsafe.Pointer(&opts.Candidates[0]))
			} else {
				pinner.Pin(&emptyCandidateSentinel)
				candidateRows = (*C.uint32_t)(unsafe.Pointer(&emptyCandidateSentinel))
			}
		}
	}

	chits := make([]C.ordvec_hit_t, int(required))
	var hitsPtr *C.ordvec_hit_t
	if len(chits) > 0 {
		hitsPtr = &chits[0]
	}
	var returned C.uint64_t
	var cstats C.ordvec_search_stats_t
	C.ordvec_search_stats_init(&cstats)
	err := callStatus(func() C.ordvec_status_t {
		return C.ordvec_go_index_search(
			idx.ptr,
			queryPtr,
			C.uint64_t(len(query)),
			C.uint64_t(k),
			candidateRows,
			candidateCount,
			userTag,
			hitsPtr,
			C.uint64_t(len(chits)),
			&returned,
			&cstats,
		)
	})
	runtime.KeepAlive(query)
	if opts != nil {
		runtime.KeepAlive(opts.Candidates)
	}
	runtime.KeepAlive(idx)
	if err != nil {
		return nil, Stats{}, err
	}

	hits := make([]Hit, int(returned))
	for i := range hits {
		hits[i] = Hit{
			RowID: uint64(chits[i].row_id),
			ID:    uint64(chits[i].id),
			Score: float32(chits[i].score),
		}
	}
	stats := Stats{
		ABIVersion:     uint32(cstats.abi_version),
		Kind:           Kind(cstats.kind),
		Dim:            uint64(cstats.dim),
		BitWidth:       uint32(cstats.bit_width),
		NTop:           uint32(cstats.n_top),
		K:              uint64(cstats.k),
		UserTag:        uint64(cstats.user_tag),
		VectorCount:    uint64(cstats.vector_count),
		CandidateCount: uint64(cstats.candidate_count),
		ReturnedCount:  uint64(cstats.returned_count),
		TotalNS:        uint64(cstats.total_ns),
		PrepareNS:      uint64(cstats.prepare_ns),
		ScoreNS:        uint64(cstats.score_ns),
		SelectNS:       uint64(cstats.select_ns),
		VectorsScored:  uint64(cstats.vectors_scored),
		BytesRead:      uint64(cstats.bytes_read),
	}
	return hits, stats, nil
}
