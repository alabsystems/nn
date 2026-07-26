// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Metal GPU dispatch infrastructure: dispatch plan construction,
//! GpuScope state machine, ActivationArena planning logic, buffer validation,
//! and error handling.
//!
//! These tests exercise the planning and state-management layers WITHOUT
//! requiring actual Metal GPU access. Pure-function dispatch plan construction,
//! cache behavior, error paths, and type properties are all testable offline.

use std::collections::HashSet;

use crate::dispatch_plan::{
    clear_dispatch_plan_cache, dispatch_plan_cache_len, plan_elementwise, plan_grid_2d,
    plan_grid_3d, plan_reduction, threadgroup_width_1d, DispatchMode,
};
use crate::dispatch_stats::{dispatch_stats, record_gpu_event, reset_counters, TOTAL_ENCODINGS};
use crate::error::MetalError;
use crate::gpu_scope::ScopeExitMode;

// ===========================================================================
// Section 1: Dispatch plan construction from DispatchMode
// ===========================================================================

/// Elementwise plan with u32::MAX: the threadgroup width should cap at 64.
#[test]
fn test_dispatch_plan_elementwise_u32_max() {
    let plan = DispatchMode::Elementwise { total: u32::MAX }
        .plan()
        .unwrap();
    assert_eq!(plan.grid(), [u32::MAX, 1, 1]);
    assert_eq!(plan.threads(), [64, 1, 1]);
    assert_eq!(plan.output_elems(), u32::MAX as usize);
    assert_eq!(plan.constants(), &[u32::MAX]);
    assert!(!plan.use_threadgroups());
    assert!(plan.threadgroup_memory_bytes().is_none());
}

/// Internal plan_elementwise with total=0 produces a zero-grid plan, not an error.
#[test]
fn test_dispatch_plan_elementwise_zero_no_error() {
    let plan = plan_elementwise(0).unwrap();
    assert_eq!(plan.grid(), [0, 1, 1]);
    assert_eq!(plan.output_elems(), 0);
    assert_eq!(plan.constants(), &[0]);
}

/// Grid2D with minimal dimensions (1x1) produces correct plan.
#[test]
fn test_dispatch_plan_grid2d_minimal() {
    let plan = plan_grid_2d([1, 1], [1, 1]).unwrap();
    assert_eq!(plan.grid(), [1, 1, 1]);
    assert_eq!(plan.threads(), [1, 1, 1]);
    assert_eq!(plan.output_elems(), 1);
    assert_eq!(plan.constants(), &[1, 1]);
    assert!(!plan.use_threadgroups());
}

/// Grid3D with minimal dimensions (1x1x1) produces correct plan.
#[test]
fn test_dispatch_plan_grid3d_minimal() {
    let plan = plan_grid_3d([1, 1, 1], [1, 1, 1]).unwrap();
    assert_eq!(plan.grid(), [1, 1, 1]);
    assert_eq!(plan.threads(), [1, 1, 1]);
    assert_eq!(plan.output_elems(), 1);
    assert_eq!(plan.constants(), &[1, 1, 1]);
}

/// Reduction plan with outer=1 (single slice) is valid.
#[test]
fn test_dispatch_plan_reduction_single_slice() {
    let plan = plan_reduction(1, 1024, 256, 4096).unwrap();
    assert_eq!(plan.grid(), [1, 1, 1]);
    assert_eq!(plan.threads(), [256, 1, 1]);
    assert_eq!(plan.output_elems(), 1);
    assert_eq!(plan.constants(), &[1, 1024]);
    assert_eq!(plan.threadgroup_memory_bytes(), Some(4096));
    assert!(plan.use_threadgroups());
}

/// Reduction plan with large outer count.
#[test]
fn test_dispatch_plan_reduction_large_outer() {
    let plan = plan_reduction(65536, 512, 128, 2048).unwrap();
    assert_eq!(plan.grid(), [65536, 1, 1]);
    assert_eq!(plan.output_elems(), 65536);
}

// ===========================================================================
// Section 2: Dispatch plan error paths
// ===========================================================================

/// Grid2D with zero in second grid dimension is rejected.
#[test]
fn test_dispatch_plan_grid2d_zero_second_grid_dim() {
    let err = plan_grid_2d([10, 0], [8, 4]).unwrap_err();
    assert!(
        err.to_string().contains("grid"),
        "expected grid error, got: {err}"
    );
}

/// Grid2D with zero in first thread dimension is rejected.
#[test]
fn test_dispatch_plan_grid2d_zero_first_thread_dim() {
    let err = plan_grid_2d([10, 20], [0, 4]).unwrap_err();
    assert!(
        err.to_string().contains("threadgroup"),
        "expected threadgroup error, got: {err}"
    );
}

/// Grid2D with zero in second thread dimension is rejected.
#[test]
fn test_dispatch_plan_grid2d_zero_second_thread_dim() {
    let err = plan_grid_2d([10, 20], [8, 0]).unwrap_err();
    assert!(
        err.to_string().contains("threadgroup"),
        "expected threadgroup error, got: {err}"
    );
}

/// Grid3D with zero in first grid dimension is rejected.
#[test]
fn test_dispatch_plan_grid3d_zero_first_grid_dim() {
    let err = plan_grid_3d([0, 8, 4], [4, 4, 2]).unwrap_err();
    assert!(err.to_string().contains("grid"));
}

/// Grid3D with zero in third grid dimension is rejected.
#[test]
fn test_dispatch_plan_grid3d_zero_third_grid_dim() {
    let err = plan_grid_3d([4, 8, 0], [4, 4, 2]).unwrap_err();
    assert!(err.to_string().contains("grid"));
}

/// Grid3D with zero in first thread dimension is rejected.
#[test]
fn test_dispatch_plan_grid3d_zero_first_thread_dim() {
    let err = plan_grid_3d([4, 8, 2], [0, 4, 1]).unwrap_err();
    assert!(err.to_string().contains("threadgroup"));
}

/// Grid3D with zero in third thread dimension is rejected.
#[test]
fn test_dispatch_plan_grid3d_zero_third_thread_dim() {
    let err = plan_grid_3d([4, 8, 2], [2, 4, 0]).unwrap_err();
    assert!(err.to_string().contains("threadgroup"));
}

/// Reduction with zero outer is rejected.
#[test]
fn test_dispatch_plan_reduction_zero_outer() {
    let err = plan_reduction(0, 256, 64, 512).unwrap_err();
    assert!(err.to_string().contains("outer"));
}

/// Reduction with zero reduce is rejected.
#[test]
fn test_dispatch_plan_reduction_zero_reduce() {
    let err = plan_reduction(64, 0, 64, 512).unwrap_err();
    assert!(err.to_string().contains("reduce"));
}

/// Reduction with zero threads is rejected.
#[test]
fn test_dispatch_plan_reduction_zero_threads() {
    let err = plan_reduction(64, 256, 0, 512).unwrap_err();
    assert!(err.to_string().contains("threads_per_group"));
}

// ===========================================================================
// Section 3: DispatchMode Hash/Eq for cache correctness
// ===========================================================================

/// DispatchMode variants with different parameters hash to different values.
#[test]
fn test_dispatch_mode_hash_distinct_elementwise() {
    let mut set = HashSet::new();
    set.insert(DispatchMode::Elementwise { total: 100 });
    set.insert(DispatchMode::Elementwise { total: 200 });
    set.insert(DispatchMode::Elementwise { total: 100 }); // duplicate
    assert_eq!(set.len(), 2, "duplicate should not increase set size");
}

/// Different DispatchMode variants are distinct in a HashSet.
#[test]
fn test_dispatch_mode_hash_distinct_variants() {
    let mut set = HashSet::new();
    set.insert(DispatchMode::Elementwise { total: 64 });
    set.insert(DispatchMode::Grid2D {
        grid: [64, 1],
        threads: [64, 1],
    });
    set.insert(DispatchMode::Grid3D {
        grid: [64, 1, 1],
        threads: [64, 1, 1],
    });
    set.insert(DispatchMode::PerSliceReduction {
        outer: 64,
        reduce: 1,
        threads: 64,
        shared_bytes: 0,
    });
    assert_eq!(set.len(), 4, "all four variant types should be distinct");
}

/// DispatchMode::Grid2D with swapped grid dimensions are distinct.
#[test]
fn test_dispatch_mode_grid2d_order_matters() {
    let a = DispatchMode::Grid2D {
        grid: [10, 20],
        threads: [4, 4],
    };
    let b = DispatchMode::Grid2D {
        grid: [20, 10],
        threads: [4, 4],
    };
    assert_ne!(a, b, "grid dimension order must matter for equality");
}

// ===========================================================================
// Section 4: Dispatch plan cache behavior
// ===========================================================================

/// plan_cached produces identical results for all DispatchMode variants.
#[test]
fn test_dispatch_plan_cache_all_variants() {
    clear_dispatch_plan_cache();

    let modes: Vec<DispatchMode> = vec![
        DispatchMode::Elementwise { total: 512 },
        DispatchMode::Grid2D {
            grid: [32, 16],
            threads: [8, 4],
        },
        DispatchMode::Grid3D {
            grid: [8, 4, 2],
            threads: [4, 2, 1],
        },
        DispatchMode::PerSliceReduction {
            outer: 32,
            reduce: 256,
            threads: 64,
            shared_bytes: 1024,
        },
    ];

    for mode in &modes {
        let uncached = mode.plan().unwrap();
        let cached = mode.plan_cached().unwrap();
        assert_eq!(
            uncached, cached,
            "cached plan must match uncached for {mode:?}"
        );
    }
    assert_eq!(dispatch_plan_cache_len(), modes.len());
}

/// Cache eviction: filling beyond DISPATCH_PLAN_CACHE_MAX triggers a clear.
#[test]
fn test_dispatch_plan_cache_eviction_on_overflow() {
    clear_dispatch_plan_cache();

    // Fill the cache with 512 unique entries (the max).
    for i in 1..=512u32 {
        let _ = DispatchMode::Elementwise { total: i }
            .plan_cached()
            .unwrap();
    }
    assert_eq!(dispatch_plan_cache_len(), 512);

    // The 513th entry should trigger eviction (clear), then insert itself.
    let _ = DispatchMode::Elementwise { total: 513 }
        .plan_cached()
        .unwrap();

    // After eviction: cache was cleared, then the new entry was inserted.
    assert_eq!(
        dispatch_plan_cache_len(),
        1,
        "cache should have 1 entry after eviction + insert"
    );
}

/// plan_cached with an error mode does not populate the cache.
#[test]
fn test_dispatch_plan_cache_error_does_not_populate() {
    clear_dispatch_plan_cache();

    // Grid2D with zero dimension returns error.
    let result = DispatchMode::Grid2D {
        grid: [0, 10],
        threads: [4, 4],
    }
    .plan_cached();
    assert!(result.is_err());
    assert_eq!(
        dispatch_plan_cache_len(),
        0,
        "error should not populate cache"
    );
}

// ===========================================================================
// Section 5: threadgroup_width_1d boundary coverage
// ===========================================================================

#[test]
fn test_threadgroup_width_1d_boundaries() {
    assert_eq!(threadgroup_width_1d(0), 0, "zero stays zero");
    assert_eq!(threadgroup_width_1d(1), 1);
    assert_eq!(threadgroup_width_1d(63), 63);
    assert_eq!(threadgroup_width_1d(64), 64, "boundary at 64");
    assert_eq!(threadgroup_width_1d(65), 64, "capped at 64");
    assert_eq!(threadgroup_width_1d(u32::MAX), 64, "large value capped at 64");
}

// ===========================================================================
// Section 6: DispatchPlan builder method chaining
// ===========================================================================

/// Builder methods are independent: each only changes the targeted field.
#[test]
fn test_dispatch_plan_builder_independence() {
    let base = DispatchMode::Elementwise { total: 100 }
        .plan()
        .unwrap();
    let original_grid = base.grid();
    let original_threads = base.threads();

    let modified = base.with_output_elems(42);
    assert_eq!(modified.output_elems(), 42);
    assert_eq!(modified.grid(), original_grid, "grid unchanged");
    assert_eq!(modified.threads(), original_threads, "threads unchanged");
    assert!(
        modified.threadgroup_memory_bytes().is_none(),
        "threadgroup memory unchanged"
    );
    assert!(!modified.use_threadgroups(), "use_threadgroups unchanged");
}

/// Builder chain: all modifiers applied produce correct combined result.
#[test]
fn test_dispatch_plan_full_builder_chain() {
    let plan = DispatchMode::Elementwise { total: 10 }
        .plan()
        .unwrap()
        .with_output_elems(500)
        .with_constants(vec![1, 2, 3, 4])
        .with_threadgroup_memory_bytes(Some(8192))
        .with_use_threadgroups(true);

    assert_eq!(plan.output_elems(), 500);
    assert_eq!(plan.constants(), &[1, 2, 3, 4]);
    assert_eq!(plan.threadgroup_memory_bytes(), Some(8192));
    assert!(plan.use_threadgroups());
    // Grid and threads from Elementwise(10) are preserved.
    assert_eq!(plan.grid(), [10, 1, 1]);
    assert_eq!(plan.threads(), [10, 1, 1]);
}

// ===========================================================================
// Section 7: GpuScope ScopeExitMode enum properties
// ===========================================================================

#[test]
fn test_scope_exit_mode_debug() {
    let flush = ScopeExitMode::Flush;
    let submit = ScopeExitMode::Submit;
    assert!(format!("{flush:?}").contains("Flush"));
    assert!(format!("{submit:?}").contains("Submit"));
}

#[test]
fn test_scope_exit_mode_clone_eq() {
    let a = ScopeExitMode::Flush;
    let b = a;
    assert_eq!(a, b);

    let c = ScopeExitMode::Submit;
    assert_ne!(a, c, "Flush != Submit");
}

#[test]
fn test_scope_exit_mode_copy_semantics() {
    let a = ScopeExitMode::Submit;
    let b = a; // Copy
    let c = a; // Copy again -- `a` still usable
    assert_eq!(b, c);
    assert_eq!(a, ScopeExitMode::Submit); // `a` not moved
}

// ===========================================================================
// Section 8: Dispatch stats counter isolation
// ===========================================================================

/// Each counter is independent: setting one does not affect others.
#[test]
fn test_dispatch_stats_counter_independence() {
    reset_counters();
    TOTAL_ENCODINGS.with(|c| c.set(10));
    let stats = dispatch_stats();
    assert_eq!(stats.compute_encodings, 10);
    assert_eq!(stats.blits, 0, "blits should be unaffected");
    assert_eq!(stats.flushes, 0, "flushes should be unaffected");
    assert_eq!(stats.submits, 0, "submits should be unaffected");
}

/// record_gpu_event correctly tracks monotonically increasing values.
#[test]
fn test_record_gpu_event_monotonic() {
    let counter = std::cell::Cell::new(0);
    let mut prev = 0;
    for _ in 0..100 {
        let n = record_gpu_event(&counter, "test-mono", 0);
        assert!(
            n > prev,
            "counter should be monotonically increasing: prev={prev}, n={n}"
        );
        prev = n;
    }
    assert_eq!(counter.get(), 100);
}

// ===========================================================================
// Section 9: MetalError variant coverage for dispatch infrastructure
// ===========================================================================

/// Zero-size buffer creation error.
#[test]
fn test_metal_error_buffer_create_zero() {
    let err = MetalError::BufferCreate(0);
    assert!(err.to_string().contains("size=0"));
}

/// Arena overflow error includes capacity and remaining info.
#[test]
fn test_metal_error_arena_overflow() {
    let err = MetalError::ArenaOverflow {
        requested: 2048,
        remaining: 512,
        capacity: 4096,
    };
    let msg = err.to_string();
    assert!(msg.contains("2048"), "should include requested: {msg}");
    assert!(msg.contains("512"), "should include remaining: {msg}");
    assert!(msg.contains("4096"), "should include capacity: {msg}");
}

/// Invalid arena alignment error.
#[test]
fn test_metal_error_invalid_arena_alignment() {
    let err = MetalError::InvalidArenaAlignment { alignment: 3 };
    let msg = err.to_string();
    assert!(
        msg.contains("3") && msg.contains("power of two"),
        "error message: {msg}"
    );
}

/// Arena checkpoint error when saved > current.
#[test]
fn test_metal_error_arena_checkpoint() {
    let err = MetalError::ArenaCheckpoint {
        saved: 1000,
        current: 500,
    };
    let msg = err.to_string();
    assert!(msg.contains("1000"), "should include saved: {msg}");
    assert!(msg.contains("500"), "should include current: {msg}");
}

/// Buffer bounds exceeded error for blit operations.
#[test]
fn test_metal_error_buffer_bounds_exceeded() {
    let err = MetalError::BufferBoundsExceeded {
        buffer_len: 1024,
        offset: 900,
        size: 200,
        role: "source",
    };
    let msg = err.to_string();
    assert!(msg.contains("900"), "should include offset: {msg}");
    assert!(msg.contains("200"), "should include size: {msg}");
    assert!(msg.contains("1024"), "should include buffer_len: {msg}");
    assert!(msg.contains("source"), "should include role: {msg}");
}

/// Buffer byte overflow error.
#[test]
fn test_metal_error_buffer_byte_overflow() {
    let err = MetalError::BufferByteOverflow {
        elems: usize::MAX,
        elem_size: 4,
    };
    let msg = err.to_string();
    assert!(msg.contains("overflows"), "error message: {msg}");
}

/// Dispatch size overflow error.
#[test]
fn test_metal_error_dispatch_size_overflow() {
    let err = MetalError::DispatchSizeOverflow(5_000_000_000);
    assert!(err.to_string().contains("exceeds u32::MAX"));
}

/// Invalid grid dimension error.
#[test]
fn test_metal_error_invalid_grid_dimension() {
    let err = MetalError::InvalidGridDimension {
        dimension: "outer",
        value: 0,
    };
    assert!(err.to_string().contains("outer"));
    assert!(err.to_string().contains("non-zero"));
}

/// Pending flush required error includes count.
#[test]
fn test_metal_error_pending_flush_required() {
    let err = MetalError::PendingFlushRequired { pending_count: 42 };
    let msg = err.to_string();
    assert!(
        msg.contains("42") && msg.contains("flush"),
        "error message: {msg}"
    );
}

/// Stale arena read error includes generation numbers.
#[test]
fn test_metal_error_stale_arena_read() {
    let err = MetalError::StaleArenaRead {
        alloc_gen: 2,
        current_gen: 5,
    };
    let msg = err.to_string();
    assert!(msg.contains("2"), "should include alloc_gen: {msg}");
    assert!(msg.contains("5"), "should include current_gen: {msg}");
}

// ===========================================================================
// Section 10: MetalError -> TensorError conversion for dispatch errors
// ===========================================================================

/// DispatchFailed maps to BackendErrorKind::DispatchFailed.
#[test]
fn test_metal_error_dispatch_failed_conversion() {
    let metal_err = MetalError::DispatchFailed("test error".into());
    let tensor_err: nn_core::TensorError = metal_err.into();
    let msg = format!("{tensor_err}");
    assert!(
        msg.contains("test error"),
        "converted error should include original message: {msg}"
    );
}

/// BufferCreate maps to BackendErrorKind::OutOfMemory.
#[test]
fn test_metal_error_buffer_create_conversion() {
    let metal_err = MetalError::BufferCreate(1024);
    let tensor_err: nn_core::TensorError = metal_err.into();
    let msg = format!("{tensor_err}");
    assert!(
        msg.contains("1024"),
        "converted error should include buffer size: {msg}"
    );
}

/// ArenaOverflow maps to BackendErrorKind::OutOfMemory.
#[test]
fn test_metal_error_arena_overflow_conversion() {
    let metal_err = MetalError::ArenaOverflow {
        requested: 8192,
        remaining: 100,
        capacity: 4096,
    };
    let tensor_err: nn_core::TensorError = metal_err.into();
    let msg = format!("{tensor_err}");
    assert!(
        msg.contains("8192"),
        "converted error should include requested size: {msg}"
    );
}

/// GpuTimeout maps to BackendErrorKind::DispatchFailed.
#[test]
fn test_metal_error_gpu_timeout_conversion() {
    let metal_err = MetalError::GpuTimeout(std::time::Duration::from_secs(60));
    let tensor_err: nn_core::TensorError = metal_err.into();
    let msg = format!("{tensor_err}");
    assert!(
        msg.contains("timed out"),
        "converted error should include timeout info: {msg}"
    );
}

// ===========================================================================
// Section 11: Dispatch plan output_elems computation correctness
// ===========================================================================

/// Verify output_elems matches product of grid dimensions for all plan types.
#[test]
fn test_dispatch_plan_output_elems_consistency() {
    // Elementwise: output_elems = total
    let plan = plan_elementwise(42).unwrap();
    assert_eq!(plan.output_elems(), 42);

    // Grid2D: output_elems = grid[0] * grid[1]
    let plan = plan_grid_2d([10, 20], [4, 4]).unwrap();
    assert_eq!(plan.output_elems(), 10 * 20);

    // Grid3D: output_elems = grid[0] * grid[1] * grid[2]
    let plan = plan_grid_3d([5, 10, 20], [2, 4, 4]).unwrap();
    assert_eq!(plan.output_elems(), 5 * 10 * 20);

    // Reduction: output_elems = outer (one output per slice)
    let plan = plan_reduction(32, 256, 64, 512).unwrap();
    assert_eq!(plan.output_elems(), 32);
}

// ===========================================================================
// Section 12: DispatchPlan grid/threads accessor symmetry
// ===========================================================================

/// Grid2D plan pads to 3D with depth=1.
#[test]
fn test_dispatch_plan_grid2d_pads_to_3d() {
    let plan = plan_grid_2d([100, 200], [8, 8]).unwrap();
    assert_eq!(plan.grid()[2], 1, "depth must be 1 for 2D");
    assert_eq!(plan.threads()[2], 1, "thread depth must be 1 for 2D");
}

/// Elementwise plan pads to 3D with height=depth=1.
#[test]
fn test_dispatch_plan_elementwise_pads_to_3d() {
    let plan = plan_elementwise(256).unwrap();
    assert_eq!(plan.grid()[1], 1, "height must be 1 for elementwise");
    assert_eq!(plan.grid()[2], 1, "depth must be 1 for elementwise");
    assert_eq!(plan.threads()[1], 1);
    assert_eq!(plan.threads()[2], 1);
}

/// Reduction plan grid is [outer, 1, 1] -- strictly 1D in outer count.
#[test]
fn test_dispatch_plan_reduction_grid_shape() {
    let plan = plan_reduction(100, 512, 128, 2048).unwrap();
    assert_eq!(plan.grid()[0], 100);
    assert_eq!(plan.grid()[1], 1);
    assert_eq!(plan.grid()[2], 1);
    assert_eq!(plan.threads()[0], 128);
    assert_eq!(plan.threads()[1], 1);
    assert_eq!(plan.threads()[2], 1);
}
