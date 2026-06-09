package ordvec

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
)

func writeRankQuantFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.tvrq")
	var b []byte
	b = append(b, []byte("TVRQ")...)
	b = append(b, 1) // version
	b = append(b, 2) // bits
	b = binary.LittleEndian.AppendUint32(b, 16)
	b = binary.LittleEndian.AppendUint32(b, 4)
	row := []byte{0x00, 0x55, 0xAA, 0xFF}
	for i := 0; i < 4; i++ {
		b = append(b, row...)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeBitmapFixture(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.tvbm")
	var b []byte
	b = append(b, []byte("TVBM")...)
	b = append(b, 1) // version
	b = binary.LittleEndian.AppendUint32(b, 64)
	b = binary.LittleEndian.AppendUint32(b, 4)
	b = binary.LittleEndian.AppendUint32(b, 4)
	row := uint64(0)
	for i := 0; i < 4; i++ {
		row |= 1 << i
	}
	for i := 0; i < 4; i++ {
		b = binary.LittleEndian.AppendUint64(b, row)
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func query16() []float32 {
	q := make([]float32, 16)
	for i := range q {
		q[i] = float32(i)
	}
	return q
}

func query64() []float32 {
	q := make([]float32, 64)
	for i := 0; i < 4; i++ {
		q[i] = 10 + float32(i)
	}
	return q
}

func TestVersionAccessors(t *testing.T) {
	if ABIVersion() != 1 {
		t.Fatalf("unexpected ABI version: %d", ABIVersion())
	}
	manifest, err := os.ReadFile(filepath.Join("..", "ordvec-ffi", "Cargo.toml"))
	if err != nil {
		t.Fatal(err)
	}
	match := regexp.MustCompile(`(?m)^version = "([^"]+)"$`).FindSubmatch(manifest)
	if match == nil {
		t.Fatal("ordvec-ffi/Cargo.toml missing package version")
	}
	if got, want := Version(), string(match[1]); got != want {
		t.Fatalf("unexpected library version: got %q want %q", got, want)
	}
}

func TestLoadInfoSearchRankQuant(t *testing.T) {
	idx, err := Load(writeRankQuantFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	info, err := idx.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.Kind != KindRankQuant || info.Dim != 16 || info.BitWidth != 2 || info.VectorCount != 4 {
		t.Fatalf("unexpected info: %+v", info)
	}

	hits, stats, err := idx.Search(query16(), 2, &SearchOptions{UserTag: 99})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits", len(hits))
	}
	if hits[0].RowID != 0 || hits[0].ID != hits[0].RowID {
		t.Fatalf("unexpected first hit: %+v", hits[0])
	}
	if stats.UserTag != 99 || stats.CandidateCount != 4 || stats.VectorsScored != 4 || stats.ReturnedCount != 2 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestRankQuantSubsetSearchOrdersByRowID(t *testing.T) {
	idx, err := Load(writeRankQuantFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	hits, stats, err := idx.Search(query16(), 2, &SearchOptions{
		Candidates: []uint32{3, 1, 2},
		UserTag:    7,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := []uint64{hits[0].RowID, hits[1].RowID}; got[0] != 1 || got[1] != 2 {
		t.Fatalf("unexpected row order: %v", got)
	}
	if stats.UserTag != 7 || stats.CandidateCount != 3 || stats.VectorsScored != 3 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestRankQuantSubsetSearchAllowsDuplicateHits(t *testing.T) {
	idx, err := Load(writeRankQuantFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	hits, stats, err := idx.Search(query16(), 3, &SearchOptions{
		Candidates: []uint32{3, 1, 1, 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := []uint64{hits[0].RowID, hits[1].RowID, hits[2].RowID}
	if got[0] != 1 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("unexpected row order: %v", got)
	}
	if stats.Kind != KindRankQuant || stats.CandidateCount != 4 || stats.VectorsScored != 4 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestBitmapSubsetSearchAllowsDuplicateHits(t *testing.T) {
	idx, err := Load(writeBitmapFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	hits, stats, err := idx.Search(query64(), 3, &SearchOptions{
		Candidates: []uint32{3, 1, 1, 2},
	})
	if err != nil {
		t.Fatal(err)
	}
	got := []uint64{hits[0].RowID, hits[1].RowID, hits[2].RowID}
	if got[0] != 1 || got[1] != 1 || got[2] != 2 {
		t.Fatalf("unexpected row order: %v", got)
	}
	if stats.Kind != KindBitmap || stats.NTop != 4 || stats.CandidateCount != 4 {
		t.Fatalf("unexpected stats: %+v", stats)
	}
}

func TestNilAndEmptySubsetDistinction(t *testing.T) {
	idx, err := Load(writeRankQuantFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	if _, _, err := idx.Search(query16(), 1, nil); err != nil {
		t.Fatalf("nil options should mean full search: %v", err)
	}
	if _, _, err := idx.Search(query16(), 1, &SearchOptions{}); err != nil {
		t.Fatalf("nil Candidates should mean full search: %v", err)
	}
	_, _, err = idx.Search(query16(), 1, &SearchOptions{Candidates: []uint32{}})
	var statusErr *StatusError
	if !errors.As(err, &statusErr) || statusErr.Status != StatusBadArgument {
		t.Fatalf("empty nonnil candidates should be BAD_ARGUMENT, got %T %[1]v", err)
	}
}

func TestTypedStatusErrors(t *testing.T) {
	idx, err := Load(writeRankQuantFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Close()

	q := query16()
	q[0] = float32(math.NaN())
	_, _, err = idx.Search(q, 1, nil)
	var statusErr *StatusError
	if !errors.As(err, &statusErr) {
		t.Fatalf("expected StatusError, got %T %[1]v", err)
	}
	if statusErr.Status != StatusNonfiniteQuery {
		t.Fatalf("unexpected status: %v", statusErr.Status)
	}

	_, err = Load(filepath.Join(t.TempDir(), "missing.tvrq"))
	if !errors.As(err, &statusErr) || statusErr.Status != StatusIO {
		t.Fatalf("missing file should be IO status, got %T %[1]v", err)
	}
}

func TestLoadRejectsNullBytePath(t *testing.T) {
	_, err := Load("bad\x00path.tvrq")
	if err == nil || !strings.Contains(err.Error(), "null byte") {
		t.Fatalf("Load should reject null byte paths, got %v", err)
	}
	var statusErr *StatusError
	if errors.As(err, &statusErr) {
		t.Fatalf("null byte path should be rejected before C call, got %v", err)
	}
}

func TestCloseIsIdempotentAndErrClosed(t *testing.T) {
	idx, err := Load(writeRankQuantFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
	if err := idx.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := idx.Info(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Info after Close should return ErrClosed, got %v", err)
	}
	if _, _, err := idx.Search(query16(), 1, nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("Search after Close should return ErrClosed, got %v", err)
	}
}

func TestConcurrentSearchInfoAndClose(t *testing.T) {
	idx, err := Load(writeRankQuantFixture(t))
	if err != nil {
		t.Fatal(err)
	}

	const workers = 8
	const iterations = 64

	start := make(chan struct{})
	searchReady := make(chan struct{})
	infoReady := make(chan struct{})
	errCh := make(chan error, workers*iterations)
	var searchReadyOnce sync.Once
	var infoReadyOnce sync.Once
	var wg sync.WaitGroup

	run := func(name string, fn func() error, markReady func()) {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				errCh <- fmt.Errorf("%s panic: %v", name, r)
			}
		}()
		<-start
		for i := 0; i < iterations; i++ {
			err := fn()
			if err == nil {
				markReady()
			}
			if err != nil && !errors.Is(err, ErrClosed) {
				errCh <- fmt.Errorf("%s returned unexpected error: %w", name, err)
			}
		}
	}

	query := query16()
	for i := 0; i < workers/2; i++ {
		wg.Add(1)
		go run(
			"Search",
			func() error {
				_, _, err := idx.Search(query, 2, nil)
				return err
			},
			func() { searchReadyOnce.Do(func() { close(searchReady) }) },
		)
		wg.Add(1)
		go run(
			"Info",
			func() error {
				_, err := idx.Info()
				return err
			},
			func() { infoReadyOnce.Do(func() { close(infoReady) }) },
		)
	}

	close(start)
	<-searchReady
	<-infoReady
	if err := idx.Close(); err != nil {
		t.Errorf("Close returned unexpected error: %v", err)
	}
	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Error(err)
	}

	if _, err := idx.Info(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Info after Close should return ErrClosed, got %v", err)
	}
	if _, _, err := idx.Search(query16(), 1, nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("Search after Close should return ErrClosed, got %v", err)
	}
}
