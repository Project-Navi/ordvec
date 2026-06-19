//! Counting-allocator proof that the caller-owned batched rerank `_into` form
//! performs ZERO heap allocations in steady state — i.e. after the
//! `SubsetScratch` has warmed to the batch shape. This is the strong form of
//! the capacity-stability proxy in `tests/index/two_stage.rs`
//! (`batched_into_is_allocation_free_after_warmup`): a capacity check can miss
//! an alloc-then-free-to-same-capacity, an allocation counter cannot.
//!
//! Lives in its own test binary so the `#[global_allocator]` only governs this
//! file's measurement and never perturbs the rest of the suite.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(feature = "test-utils")]
use ordvec::{RankQuant, SubsetScratch};
#[cfg(feature = "test-utils")]
use rand::{RngExt, SeedableRng};
#[cfg(feature = "test-utils")]
use rand_chacha::ChaCha8Rng;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// System allocator that counts allocating operations (alloc / zeroed /
/// realloc). Dealloc is not counted — we assert on *allocations* in a window.
struct Counting;

// SAFETY: every method forwards to the System allocator with the identical
// pointer/layout, only incrementing a relaxed counter first; this preserves
// all of System's safety contract.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[cfg(feature = "test-utils")]
#[test]
fn batched_into_is_truly_allocation_free_after_warmup() {
    let dim = 132usize;
    let n = 2_000usize;
    let nq = 8usize;
    let m = 64usize;
    let k = 10usize;
    let bits = 2u8;
    assert!(
        !ordvec::subset_rerank_uses_simd(dim, bits),
        "test shape must force the scalar rerank fallback"
    );

    let mut rng = ChaCha8Rng::seed_from_u64(2024);
    let corpus: Vec<f32> = (0..n * dim).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut rq = RankQuant::new(dim, bits);
    rq.add(&corpus);
    let queries: Vec<f32> = (0..nq * dim).map(|_| rng.random_range(-1.0..1.0)).collect();

    let mut offsets = Vec::with_capacity(nq + 1);
    let mut candidates = Vec::with_capacity(nq * m);
    for _ in 0..nq {
        offsets.push(candidates.len());
        candidates.extend(0..m as u32);
    }
    offsets.push(candidates.len());
    let out_k = k.min(rq.len());
    let mut out_scores = vec![f32::NEG_INFINITY; nq * out_k];
    let mut out_indices = vec![-1i64; nq * out_k];
    let mut scratch = SubsetScratch::new();

    // Warm the scratch to this exact batch shape.
    rq.search_asymmetric_subset_batched_serial_into(
        &queries,
        &offsets,
        &candidates,
        k,
        &mut scratch,
        &mut out_scores,
        &mut out_indices,
    );

    // Steady state: an identical second call (same shape, warmed scratch,
    // caller-owned output buffers reused) must allocate nothing.
    let before = ALLOCS.load(Ordering::Relaxed);
    rq.search_asymmetric_subset_batched_serial_into(
        &queries,
        &offsets,
        &candidates,
        k,
        &mut scratch,
        &mut out_scores,
        &mut out_indices,
    );
    let after = ALLOCS.load(Ordering::Relaxed);

    assert_eq!(
        after - before,
        0,
        "steady-state _into allocated {} time(s) (expected 0)",
        after - before
    );
}
