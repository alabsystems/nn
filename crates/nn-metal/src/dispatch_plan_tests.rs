// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;

// --- Elementwise ---

#[test]
fn test_elementwise_zero_total() {
    let plan = DispatchMode::Elementwise { total: 0 }.plan().unwrap();
    assert_eq!(plan.output_elems, 0);
    assert_eq!(plan.grid, [0, 1, 1]);
    assert!(!plan.use_threadgroups);
}

#[test]
fn test_elementwise_small() {
    let plan = DispatchMode::Elementwise { total: 10 }.plan().unwrap();
    assert_eq!(plan.grid, [10, 1, 1]);
    assert_eq!(plan.threads, [10, 1, 1]);
    assert_eq!(plan.output_elems, 10);
    assert_eq!(plan.constants, vec![10]);
    assert!(plan.threadgroup_memory_bytes.is_none());
    assert!(!plan.use_threadgroups);
}

#[test]
fn test_elementwise_large() {
    let plan = DispatchMode::Elementwise { total: 1024 }.plan().unwrap();
    assert_eq!(plan.grid, [1024, 1, 1]);
    assert_eq!(plan.threads, [64, 1, 1]);
}

// --- Grid2D ---

#[test]
fn test_grid_2d_valid() {
    let plan = DispatchMode::Grid2D {
        grid: [128, 64],
        threads: [16, 8],
    }
    .plan()
    .unwrap();
    assert_eq!(plan.grid, [128, 64, 1]);
    assert_eq!(plan.threads, [16, 8, 1]);
    assert_eq!(plan.output_elems, 128 * 64);
    assert_eq!(plan.constants, vec![128, 64]);
    assert!(!plan.use_threadgroups);
}

#[test]
fn test_grid_2d_zero_grid_dim_rejected() {
    let err = DispatchMode::Grid2D {
        grid: [0, 64],
        threads: [16, 8],
    }
    .plan()
    .unwrap_err();
    assert!(err.to_string().contains("grid"));
}

#[test]
fn test_grid_2d_zero_thread_dim_rejected() {
    let err = DispatchMode::Grid2D {
        grid: [128, 64],
        threads: [0, 8],
    }
    .plan()
    .unwrap_err();
    assert!(err.to_string().contains("threadgroup"));
}

// --- Grid3D ---

#[test]
fn test_grid_3d_valid() {
    let plan = DispatchMode::Grid3D {
        grid: [32, 16, 8],
        threads: [8, 4, 2],
    }
    .plan()
    .unwrap();
    assert_eq!(plan.grid, [32, 16, 8]);
    assert_eq!(plan.threads, [8, 4, 2]);
    assert_eq!(plan.output_elems, 32 * 16 * 8);
    assert_eq!(plan.constants, vec![32, 16, 8]);
}

#[test]
fn test_grid_3d_zero_dim_rejected() {
    let err = DispatchMode::Grid3D {
        grid: [32, 0, 8],
        threads: [8, 4, 2],
    }
    .plan()
    .unwrap_err();
    assert!(err.to_string().contains("grid"));
}

// --- PerSliceReduction ---

#[test]
fn test_reduction_valid() {
    let plan = DispatchMode::PerSliceReduction {
        outer: 128,
        reduce: 768,
        threads: 256,
        shared_bytes: 1024,
    }
    .plan()
    .unwrap();
    assert_eq!(plan.grid, [128, 1, 1]);
    assert_eq!(plan.threads, [256, 1, 1]);
    assert_eq!(plan.output_elems, 128);
    assert_eq!(plan.constants, vec![128, 768]);
    assert_eq!(plan.threadgroup_memory_bytes, Some(1024));
    assert!(plan.use_threadgroups);
}

#[test]
fn test_reduction_zero_outer_rejected() {
    let err = DispatchMode::PerSliceReduction {
        outer: 0,
        reduce: 768,
        threads: 256,
        shared_bytes: 1024,
    }
    .plan()
    .unwrap_err();
    assert!(err.to_string().contains("outer"));
}

#[test]
fn test_reduction_zero_reduce_rejected() {
    let err = DispatchMode::PerSliceReduction {
        outer: 128,
        reduce: 0,
        threads: 256,
        shared_bytes: 1024,
    }
    .plan()
    .unwrap_err();
    assert!(err.to_string().contains("reduce"));
}

#[test]
fn test_reduction_zero_threads_rejected() {
    let err = DispatchMode::PerSliceReduction {
        outer: 128,
        reduce: 768,
        threads: 0,
        shared_bytes: 1024,
    }
    .plan()
    .unwrap_err();
    assert!(err.to_string().contains("threads_per_group"));
}

// --- threadgroup_width_1d ---

#[test]
fn test_threadgroup_width_small() {
    assert_eq!(threadgroup_width_1d(1), 1);
    assert_eq!(threadgroup_width_1d(32), 32);
    assert_eq!(threadgroup_width_1d(63), 63);
}

#[test]
fn test_threadgroup_width_boundary() {
    assert_eq!(threadgroup_width_1d(64), 64);
    assert_eq!(threadgroup_width_1d(65), 64);
    assert_eq!(threadgroup_width_1d(10000), 64);
}

/// Regression: 3D output_elems overflow is caught by checked multiplication.
///
/// The cube root of u64::MAX ≈ 2.6M, so grids where all three dimensions
/// exceed that threshold would overflow. `plan_grid_3d` uses checked_mul
/// and returns `DispatchSizeOverflow` instead of silent wrap.
#[test]
fn test_grid_3d_output_elems_overflow_returns_error() {
    // Three dimensions each > cube_root(u64::MAX) ≈ 2,642,245
    let large: u32 = 3_000_000;
    let err = DispatchMode::Grid3D {
        grid: [large, large, large],
        threads: [8, 8, 8],
    }
    .plan()
    .unwrap_err();
    assert!(
        err.to_string().contains("exceeds u32::MAX"),
        "expected DispatchSizeOverflow, got: {err}"
    );
}

/// 3D grids with small dimensions that fit in usize must still succeed.
#[test]
fn test_grid_3d_no_overflow_small_dims() {
    let plan = DispatchMode::Grid3D {
        grid: [100, 200, 300],
        threads: [8, 4, 2],
    }
    .plan()
    .unwrap();
    assert_eq!(plan.output_elems, 100 * 200 * 300);
}

/// Regression: 2D output_elems uses checked_mul (consistency with 3D).
///
/// On 64-bit platforms, u32 * u32 always fits in usize, so this validates
/// the guard exists without triggering it. On 32-bit targets, large grid
/// dimensions would actually overflow.
#[test]
fn test_grid_2d_large_dims_checked_mul() {
    let plan = DispatchMode::Grid2D {
        grid: [u32::MAX, 1],
        threads: [64, 1],
    }
    .plan()
    .unwrap();
    assert_eq!(plan.output_elems, u32::MAX as usize);
}

/// 2D grid with both dimensions at u32::MAX succeeds on 64-bit (product fits usize).
#[test]
fn test_grid_2d_max_both_dims() {
    let plan = DispatchMode::Grid2D {
        grid: [u32::MAX, u32::MAX],
        threads: [64, 64],
    }
    .plan();
    // On 64-bit: (2^32-1)^2 < 2^64-1, so this fits.
    // On 32-bit: this would overflow → Err.
    if cfg!(target_pointer_width = "64") {
        let plan = plan.unwrap();
        assert_eq!(plan.output_elems, u32::MAX as usize * u32::MAX as usize);
    } else {
        assert!(plan.is_err());
    }
}

// --- DispatchPlan builder methods ---

#[test]
fn test_plan_with_output_elems() {
    let plan = DispatchMode::Elementwise { total: 10 }
        .plan()
        .unwrap()
        .with_output_elems(42);
    assert_eq!(plan.output_elems(), 42, "with_output_elems overrides");
    // Other fields unchanged.
    assert_eq!(plan.grid(), [10, 1, 1]);
    assert_eq!(plan.threads(), [10, 1, 1]);
}

#[test]
fn test_plan_with_constants() {
    let plan = DispatchMode::Elementwise { total: 10 }
        .plan()
        .unwrap()
        .with_constants(vec![1, 2, 3]);
    assert_eq!(plan.constants(), &[1, 2, 3]);
    assert_eq!(plan.output_elems(), 10, "output_elems unchanged");
}

#[test]
fn test_plan_with_threadgroup_memory_bytes() {
    let plan = DispatchMode::Elementwise { total: 10 }
        .plan()
        .unwrap()
        .with_threadgroup_memory_bytes(Some(2048));
    assert_eq!(plan.threadgroup_memory_bytes(), Some(2048));
}

#[test]
fn test_plan_with_threadgroup_memory_bytes_none() {
    // Start with a reduction (has threadgroup memory), then clear it.
    let plan = DispatchMode::PerSliceReduction {
        outer: 4,
        reduce: 256,
        threads: 64,
        shared_bytes: 512,
    }
    .plan()
    .unwrap()
    .with_threadgroup_memory_bytes(None);
    assert_eq!(plan.threadgroup_memory_bytes(), None);
}

#[test]
fn test_plan_with_use_threadgroups() {
    let plan = DispatchMode::Elementwise { total: 10 }
        .plan()
        .unwrap()
        .with_use_threadgroups(true);
    assert!(plan.use_threadgroups());
}

#[test]
fn test_plan_builder_chain() {
    let plan = DispatchMode::Grid2D {
        grid: [32, 16],
        threads: [8, 4],
    }
    .plan()
    .unwrap()
    .with_output_elems(999)
    .with_constants(vec![7])
    .with_threadgroup_memory_bytes(Some(4096))
    .with_use_threadgroups(true);

    assert_eq!(plan.output_elems(), 999);
    assert_eq!(plan.constants(), &[7]);
    assert_eq!(plan.threadgroup_memory_bytes(), Some(4096));
    assert!(plan.use_threadgroups());
    // Grid and threads unchanged by builder.
    assert_eq!(plan.grid(), [32, 16, 1]);
    assert_eq!(plan.threads(), [8, 4, 1]);
}

// --- DispatchPlan accessor coverage ---

#[test]
fn test_reduction_plan_accessors() {
    let plan = DispatchMode::PerSliceReduction {
        outer: 64,
        reduce: 512,
        threads: 128,
        shared_bytes: 2048,
    }
    .plan()
    .unwrap();
    assert_eq!(plan.grid(), [64, 1, 1]);
    assert_eq!(plan.threads(), [128, 1, 1]);
    assert_eq!(plan.output_elems(), 64);
    assert_eq!(plan.constants(), &[64, 512]);
    assert_eq!(plan.threadgroup_memory_bytes(), Some(2048));
    assert!(plan.use_threadgroups());
}

#[test]
fn test_elementwise_plan_no_threadgroup_memory() {
    let plan = DispatchMode::Elementwise { total: 256 }
        .plan()
        .unwrap();
    assert!(plan.threadgroup_memory_bytes().is_none());
    assert!(!plan.use_threadgroups());
}

// --- DispatchPlan Debug and Clone ---

#[test]
fn test_dispatch_plan_debug() {
    let plan = DispatchMode::Elementwise { total: 5 }
        .plan()
        .unwrap();
    let dbg = format!("{plan:?}");
    assert!(dbg.contains("DispatchPlan"));
}

#[test]
fn test_dispatch_plan_clone_eq() {
    let plan = DispatchMode::Grid3D {
        grid: [4, 8, 2],
        threads: [2, 4, 1],
    }
    .plan()
    .unwrap();
    let cloned = plan.clone();
    assert_eq!(plan, cloned);
}

// --- DispatchMode edge cases ---

#[test]
fn test_grid_3d_zero_thread_dim_rejected() {
    let err = DispatchMode::Grid3D {
        grid: [4, 8, 2],
        threads: [2, 0, 1],
    }
    .plan()
    .unwrap_err();
    assert!(err.to_string().contains("threadgroup"));
}

#[test]
fn test_reduction_zero_shared_bytes_allowed() {
    // shared_bytes = 0 is valid (kernel may not use shared memory).
    let plan = DispatchMode::PerSliceReduction {
        outer: 4,
        reduce: 32,
        threads: 16,
        shared_bytes: 0,
    }
    .plan()
    .unwrap();
    assert_eq!(plan.threadgroup_memory_bytes(), Some(0));
}

#[test]
fn test_elementwise_single_element() {
    let plan = DispatchMode::Elementwise { total: 1 }
        .plan()
        .unwrap();
    assert_eq!(plan.grid(), [1, 1, 1]);
    assert_eq!(plan.threads(), [1, 1, 1]);
    assert_eq!(plan.output_elems(), 1);
    assert_eq!(plan.constants(), &[1]);
}

#[test]
fn test_elementwise_exactly_64() {
    let plan = DispatchMode::Elementwise { total: 64 }
        .plan()
        .unwrap();
    assert_eq!(plan.threads(), [64, 1, 1], "64 threads at boundary");
}

// --- DispatchMode Debug and Clone ---

#[test]
fn test_dispatch_mode_debug() {
    let mode = DispatchMode::Elementwise { total: 42 };
    let dbg = format!("{mode:?}");
    assert!(dbg.contains("Elementwise"));
    assert!(dbg.contains("42"));
}

#[test]
fn test_dispatch_mode_clone_eq() {
    let mode = DispatchMode::Grid2D {
        grid: [10, 20],
        threads: [8, 4],
    };
    let cloned = mode.clone();
    assert_eq!(mode, cloned);
}

// --- Dispatch plan cache ---

#[test]
fn test_plan_cached_returns_same_as_plan() {
    clear_dispatch_plan_cache();
    let mode = DispatchMode::Elementwise { total: 256 };
    let uncached = mode.plan().unwrap();
    let cached = mode.plan_cached().unwrap();
    assert_eq!(uncached, cached);
}

#[test]
fn test_plan_cached_populates_cache() {
    clear_dispatch_plan_cache();
    assert_eq!(dispatch_plan_cache_len(), 0);
    let _ = DispatchMode::Elementwise { total: 100 }.plan_cached().unwrap();
    assert_eq!(dispatch_plan_cache_len(), 1, "cache should have 1 entry after first call");
}

#[test]
fn test_plan_cached_second_call_is_cache_hit() {
    clear_dispatch_plan_cache();
    let mode = DispatchMode::Grid2D {
        grid: [32, 16],
        threads: [8, 4],
    };
    let first = mode.plan_cached().unwrap();
    assert_eq!(dispatch_plan_cache_len(), 1);
    let second = mode.plan_cached().unwrap();
    // Cache size should not increase on the second call (same key).
    assert_eq!(dispatch_plan_cache_len(), 1);
    assert_eq!(first, second);
}

#[test]
fn test_plan_cached_different_mode_is_cache_miss() {
    clear_dispatch_plan_cache();
    let _ = DispatchMode::Elementwise { total: 100 }.plan_cached().unwrap();
    assert_eq!(dispatch_plan_cache_len(), 1);
    let _ = DispatchMode::Elementwise { total: 200 }.plan_cached().unwrap();
    assert_eq!(dispatch_plan_cache_len(), 2, "different mode should be a new entry");
}

#[test]
fn test_plan_cached_different_dispatch_types() {
    clear_dispatch_plan_cache();
    let _ = DispatchMode::Elementwise { total: 64 }.plan_cached().unwrap();
    let _ = DispatchMode::Grid3D {
        grid: [4, 8, 2],
        threads: [2, 4, 1],
    }
    .plan_cached()
    .unwrap();
    let _ = DispatchMode::PerSliceReduction {
        outer: 16,
        reduce: 256,
        threads: 128,
        shared_bytes: 512,
    }
    .plan_cached()
    .unwrap();
    assert_eq!(dispatch_plan_cache_len(), 3, "three different modes = three entries");
}

#[test]
fn test_clear_dispatch_plan_cache() {
    clear_dispatch_plan_cache();
    let _ = DispatchMode::Elementwise { total: 50 }.plan_cached().unwrap();
    let _ = DispatchMode::Elementwise { total: 60 }.plan_cached().unwrap();
    assert_eq!(dispatch_plan_cache_len(), 2);
    clear_dispatch_plan_cache();
    assert_eq!(dispatch_plan_cache_len(), 0, "cache should be empty after clear");
}

#[test]
fn test_plan_cached_reduction_matches_plan() {
    clear_dispatch_plan_cache();
    let mode = DispatchMode::PerSliceReduction {
        outer: 64,
        reduce: 512,
        threads: 128,
        shared_bytes: 2048,
    };
    let uncached = mode.plan().unwrap();
    let cached = mode.plan_cached().unwrap();
    assert_eq!(uncached, cached, "cached reduction must match uncached");
}

#[test]
fn test_plan_cached_builder_does_not_mutate_cache() {
    clear_dispatch_plan_cache();
    let mode = DispatchMode::Elementwise { total: 10 };
    let base = mode.plan_cached().unwrap();
    let modified = mode.plan_cached().unwrap().with_output_elems(999);
    // The cached plan should still match the original base plan.
    let refetched = mode.plan_cached().unwrap();
    assert_eq!(base, refetched, "cache entry must not be mutated by builder");
    assert_ne!(modified.output_elems(), refetched.output_elems());
}
