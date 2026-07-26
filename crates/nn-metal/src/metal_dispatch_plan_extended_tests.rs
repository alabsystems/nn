// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for Metal dispatch plan infrastructure: plan construction,
//! GPU buffer size calculations, pipeline cache configuration, NativeOpKind
//! variant coverage, Metal threadgroup sizing, dispatch profiler types,
//! KernelSource construction, SizeClassAllocator statistics, and error types.
//!
//! All 71 tests are structure/config tests only — no live GPU required.
//! Part of #4563.

use std::collections::HashSet;

use nn_core::DType;
use nn_dsl::ir::ScalarType;
use nn_dsl::PeepholeConfig;

use crate::dispatch_plan::{
    clear_dispatch_plan_cache, dispatch_plan_cache_len, plan_elementwise, plan_grid_2d,
    plan_grid_3d, plan_reduction, threadgroup_width_1d, DispatchMode,
};
use crate::dispatch_profiler::{DispatchProfileEntry, DispatchType, FusionOpportunity, TopEntry, TypeBreakdown};
use crate::dispatch_stats::{dispatch_stats, reset_counters};
use crate::error::MetalError;
use crate::kernel_source::KernelSource;
use crate::simdgroup_tile_select::{
    is_scalar_fallback, select_gemm_tiles, TileConfig, SIMDGROUP_ALIGN, SMALL_K_THRESHOLD,
    STANDARD_MN_THRESHOLD, TALL_SKINNY_RATIO, TINY_THRESHOLD, WIDE_RATIO,
};

// ═══════════════════════════════════════════════════════════════════════
// 1. DispatchMode::plan — basic plan construction for each variant
// ═══════════════════════════════════════════════════════════════════════

/// Elementwise mode with 1 element produces a valid 1-element plan.
#[test]
fn plan_elementwise_single_element() {
    let plan = plan_elementwise(1).unwrap();
    assert_eq!(plan.grid(), [1, 1, 1]);
    assert_eq!(plan.threads(), [1, 1, 1]);
    assert_eq!(plan.output_elems(), 1);
    assert!(!plan.use_threadgroups());
    assert_eq!(plan.constants(), &[1]);
}

/// Elementwise mode with 0 elements yields an empty plan (not an error).
#[test]
fn plan_elementwise_zero_is_valid_empty() {
    let plan = plan_elementwise(0).unwrap();
    assert_eq!(plan.grid(), [0, 1, 1]);
    assert_eq!(plan.output_elems(), 0);
    assert_eq!(plan.constants(), &[0]);
}

/// Elementwise mode with large element count uses 64-wide threadgroups.
#[test]
fn plan_elementwise_large_uses_64_width() {
    let plan = plan_elementwise(100_000).unwrap();
    assert_eq!(plan.threads()[0], 64);
    assert_eq!(plan.grid()[0], 100_000);
    assert_eq!(plan.output_elems(), 100_000);
}

/// Grid2D with zero grid dimension returns error.
#[test]
fn plan_grid2d_zero_grid_is_error() {
    assert!(plan_grid_2d([0, 10], [8, 8]).is_err());
    assert!(plan_grid_2d([10, 0], [8, 8]).is_err());
}

/// Grid2D with zero thread dimension returns error.
#[test]
fn plan_grid2d_zero_threads_is_error() {
    assert!(plan_grid_2d([10, 10], [0, 8]).is_err());
    assert!(plan_grid_2d([10, 10], [8, 0]).is_err());
}

/// Grid2D valid construction sets correct grid and thread dims.
#[test]
fn plan_grid2d_valid_construction() {
    let plan = plan_grid_2d([64, 32], [8, 4]).unwrap();
    assert_eq!(plan.grid(), [64, 32, 1]);
    assert_eq!(plan.threads(), [8, 4, 1]);
    assert_eq!(plan.output_elems(), 64 * 32);
    assert_eq!(plan.constants(), &[64, 32]);
    assert!(!plan.use_threadgroups());
}

/// Grid3D with zero grid dimension returns error.
#[test]
fn plan_grid3d_zero_grid_is_error() {
    assert!(plan_grid_3d([0, 10, 10], [4, 4, 4]).is_err());
    assert!(plan_grid_3d([10, 0, 10], [4, 4, 4]).is_err());
    assert!(plan_grid_3d([10, 10, 0], [4, 4, 4]).is_err());
}

/// Grid3D with zero threads returns error.
#[test]
fn plan_grid3d_zero_threads_is_error() {
    assert!(plan_grid_3d([8, 8, 8], [0, 4, 4]).is_err());
    assert!(plan_grid_3d([8, 8, 8], [4, 0, 4]).is_err());
    assert!(plan_grid_3d([8, 8, 8], [4, 4, 0]).is_err());
}

/// Grid3D valid construction sets correct output_elems = product of grid.
#[test]
fn plan_grid3d_valid_construction() {
    let plan = plan_grid_3d([4, 8, 16], [2, 4, 2]).unwrap();
    assert_eq!(plan.grid(), [4, 8, 16]);
    assert_eq!(plan.threads(), [2, 4, 2]);
    assert_eq!(plan.output_elems(), 4 * 8 * 16);
    assert_eq!(plan.constants(), &[4, 8, 16]);
    assert!(!plan.use_threadgroups());
}

/// Reduction with zero outer returns error.
#[test]
fn plan_reduction_zero_outer_is_error() {
    assert!(plan_reduction(0, 256, 128, 2048).is_err());
}

/// Reduction with zero reduce dim returns error.
#[test]
fn plan_reduction_zero_reduce_is_error() {
    assert!(plan_reduction(64, 0, 128, 2048).is_err());
}

/// Reduction with zero threads returns error.
#[test]
fn plan_reduction_zero_threads_is_error() {
    assert!(plan_reduction(64, 256, 0, 2048).is_err());
}

/// Reduction valid construction uses threadgroups and has shared memory.
#[test]
fn plan_reduction_valid_construction() {
    let plan = plan_reduction(128, 768, 256, 4096).unwrap();
    assert_eq!(plan.grid(), [128, 1, 1]);
    assert_eq!(plan.threads(), [256, 1, 1]);
    assert_eq!(plan.output_elems(), 128);
    assert!(plan.use_threadgroups());
    assert_eq!(plan.threadgroup_memory_bytes(), Some(4096));
    assert_eq!(plan.constants(), &[128, 768]);
}

// ═══════════════════════════════════════════════════════════════════════
// 2. DispatchPlan builder pattern — with_* methods
// ═══════════════════════════════════════════════════════════════════════

/// with_output_elems overrides the plan's output element count.
#[test]
fn plan_with_output_elems_override() {
    let plan = plan_elementwise(512).unwrap().with_output_elems(999);
    assert_eq!(plan.output_elems(), 999);
}

/// with_constants overrides the plan's constant values.
#[test]
fn plan_with_constants_override() {
    let plan = plan_elementwise(512).unwrap().with_constants(vec![42, 99]);
    assert_eq!(plan.constants(), &[42, 99]);
}

/// with_threadgroup_memory_bytes overrides shared memory.
#[test]
fn plan_with_threadgroup_memory_bytes_override() {
    let plan = plan_elementwise(512)
        .unwrap()
        .with_threadgroup_memory_bytes(Some(8192));
    assert_eq!(plan.threadgroup_memory_bytes(), Some(8192));
}

/// with_use_threadgroups overrides dispatch mode.
#[test]
fn plan_with_use_threadgroups_override() {
    let plan = plan_elementwise(512).unwrap().with_use_threadgroups(true);
    assert!(plan.use_threadgroups());
}

// ═══════════════════════════════════════════════════════════════════════
// 3. threadgroup_width_1d — boundary cases
// ═══════════════════════════════════════════════════════════════════════

/// threadgroup_width_1d(0) returns 0 (no threads).
#[test]
fn threadgroup_width_1d_zero() {
    assert_eq!(threadgroup_width_1d(0), 0);
}

/// threadgroup_width_1d(1) returns 1.
#[test]
fn threadgroup_width_1d_one() {
    assert_eq!(threadgroup_width_1d(1), 1);
}

/// threadgroup_width_1d for values < 64 returns the value itself.
#[test]
fn threadgroup_width_1d_small_values() {
    for v in [2, 7, 16, 31, 32, 63] {
        assert_eq!(threadgroup_width_1d(v), v, "small value {v} should be identity");
    }
}

/// threadgroup_width_1d at 64 returns 64.
#[test]
fn threadgroup_width_1d_exactly_64() {
    assert_eq!(threadgroup_width_1d(64), 64);
}

/// threadgroup_width_1d above 64 clamps to 64.
#[test]
fn threadgroup_width_1d_large_values() {
    for v in [65, 128, 1024, u32::MAX] {
        assert_eq!(threadgroup_width_1d(v), 64, "large value {v} should clamp to 64");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 4. Dispatch plan cache — thread-local caching behavior
// ═══════════════════════════════════════════════════════════════════════

/// Clear cache resets length to zero.
#[test]
fn cache_clear_resets_to_zero() {
    clear_dispatch_plan_cache();
    assert_eq!(dispatch_plan_cache_len(), 0);
}

/// plan_cached populates cache.
#[test]
fn cache_plan_cached_populates() {
    clear_dispatch_plan_cache();
    let _ = DispatchMode::Elementwise { total: 77 }.plan_cached().unwrap();
    assert_eq!(dispatch_plan_cache_len(), 1);
}

/// plan_cached for same mode doesn't duplicate.
#[test]
fn cache_same_mode_no_duplicate() {
    clear_dispatch_plan_cache();
    let _ = DispatchMode::Elementwise { total: 200 }.plan_cached().unwrap();
    let _ = DispatchMode::Elementwise { total: 200 }.plan_cached().unwrap();
    assert_eq!(dispatch_plan_cache_len(), 1);
}

/// plan_cached for different modes each get an entry.
#[test]
fn cache_different_modes_separate_entries() {
    clear_dispatch_plan_cache();
    let _ = DispatchMode::Elementwise { total: 100 }.plan_cached().unwrap();
    let _ = DispatchMode::Elementwise { total: 200 }.plan_cached().unwrap();
    let _ = DispatchMode::Grid2D {
        grid: [8, 8],
        threads: [4, 4],
    }
    .plan_cached()
    .unwrap();
    assert_eq!(dispatch_plan_cache_len(), 3);
}

/// plan_cached returns same result as plan (no corruption).
#[test]
fn cache_matches_uncached() {
    clear_dispatch_plan_cache();
    let mode = DispatchMode::PerSliceReduction {
        outer: 32,
        reduce: 256,
        threads: 64,
        shared_bytes: 1024,
    };
    let uncached = mode.plan().unwrap();
    let cached = mode.plan_cached().unwrap();
    assert_eq!(uncached, cached);
}

/// DispatchMode::Eq distinguishes Grid2D from Grid3D.
#[test]
fn dispatch_mode_eq_distinguishes_variants() {
    let a = DispatchMode::Grid2D {
        grid: [8, 8],
        threads: [4, 4],
    };
    let b = DispatchMode::Grid3D {
        grid: [8, 8, 1],
        threads: [4, 4, 1],
    };
    assert_ne!(a, b, "Grid2D and Grid3D should not be equal");
}

/// DispatchMode::Hash produces different hashes for different totals.
#[test]
fn dispatch_mode_hash_varies() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let hash_of = |mode: &DispatchMode| {
        let mut h = DefaultHasher::new();
        mode.hash(&mut h);
        h.finish()
    };

    let m1 = DispatchMode::Elementwise { total: 100 };
    let m2 = DispatchMode::Elementwise { total: 200 };
    // Hash collision is technically possible but extremely unlikely for these values.
    assert_ne!(hash_of(&m1), hash_of(&m2));
}

// ═══════════════════════════════════════════════════════════════════════
// 5. GPU buffer size calculations — element count * dtype size
// ═══════════════════════════════════════════════════════════════════════

/// Buffer bytes for a [B, C, T] tensor shape at different dtypes.
#[test]
fn buffer_bytes_for_3d_shape() {
    let shape = [1_usize, 256, 1024];
    let elems: usize = shape.iter().product();
    assert_eq!(elems, 262_144);

    assert_eq!(elems * DType::F32.size_bytes(), 1_048_576); // 1 MB
    assert_eq!(elems * DType::F16.size_bytes(), 524_288);   // 512 KB
    assert_eq!(elems * DType::BF16.size_bytes(), 524_288);  // 512 KB
}

/// Buffer bytes for embedding table [vocab, dim].
#[test]
fn buffer_bytes_embedding_table() {
    let vocab_size = 32_000_usize;
    let dim = 768;
    let total = vocab_size * dim;
    assert_eq!(total * DType::F32.size_bytes(), 98_304_000);
    assert_eq!(total * DType::F16.size_bytes(), 49_152_000);
}

/// Buffer bytes: checked_mul catches overflow for unreasonable sizes.
#[test]
fn buffer_bytes_overflow_detection_checked_mul() {
    let elems = usize::MAX / 3;
    let result = elems.checked_mul(DType::F32.size_bytes());
    assert!(result.is_none(), "should overflow for usize::MAX/3 * 4");
}

/// Buffer bytes for scalar (rank-0) is exactly one element.
#[test]
fn buffer_bytes_scalar_tensor() {
    let elems = 1_usize; // rank-0 shape []
    assert_eq!(elems * DType::F32.size_bytes(), 4);
    assert_eq!(elems * DType::F64.size_bytes(), 8);
}

/// Buffer bytes for bool dtype is 1 byte per element.
#[test]
fn buffer_bytes_bool_dtype() {
    assert_eq!(100 * DType::Bool.size_bytes(), 100);
}

// ═══════════════════════════════════════════════════════════════════════
// 6. Metal threadgroup sizing — stays within device limits
// ═══════════════════════════════════════════════════════════════════════

const METAL_MAX_THREADS: u32 = 1024;

/// All elementwise plans have threadgroup product <= 1024.
#[test]
fn threadgroup_elementwise_within_limits() {
    for total in [1, 16, 64, 128, 512, 1024, 65536, u32::MAX] {
        let plan = plan_elementwise(total).unwrap();
        let product: u32 = plan.threads().iter().product();
        assert!(
            product <= METAL_MAX_THREADS,
            "total={total}: product={product} exceeds {METAL_MAX_THREADS}"
        );
    }
}

/// Grid3D with small threadgroup dims stays within Metal limits.
#[test]
fn threadgroup_grid3d_small_within_limits() {
    let plan = plan_grid_3d([100, 50, 25], [8, 8, 2]).unwrap();
    let product: u32 = plan.threads().iter().product();
    assert!(product <= METAL_MAX_THREADS);
    assert_eq!(product, 128);
}

/// Reduction threadgroup width stays within Metal limits.
#[test]
fn threadgroup_reduction_within_limits() {
    for threads in [32, 64, 128, 256, 512, 1024] {
        let plan = plan_reduction(16, 1024, threads, 4096).unwrap();
        let product: u32 = plan.threads().iter().product();
        assert!(
            product <= METAL_MAX_THREADS,
            "threads={threads}: product={product} exceeds Metal max"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 7. Simdgroup tile selection — shape-based routing
// ═══════════════════════════════════════════════════════════════════════

/// Tiny matrices (M*N < 1024) fall back to scalar.
#[test]
fn tile_select_tiny_scalar_fallback() {
    assert!(select_gemm_tiles(1, 640, 256).is_none());
    assert!(select_gemm_tiles(4, 8, 16).is_none());
    assert!(select_gemm_tiles(31, 32, 31).is_none());
}

/// is_scalar_fallback is consistent with select_gemm_tiles returning None.
#[test]
fn tile_select_scalar_fallback_consistent() {
    assert!(is_scalar_fallback(1, 256));
    assert!(is_scalar_fallback(4, 16));
    assert!(!is_scalar_fallback(64, 64)); // 4096 >= TINY_THRESHOLD
}

/// Large square matrices get the SQUARE tile config.
#[test]
fn tile_select_large_square() {
    let cfg = select_gemm_tiles(256, 768, 768).unwrap();
    assert_eq!(cfg.tile_m, 32);
    assert_eq!(cfg.tile_n, 32);
    assert_eq!(cfg.tile_k, 32);
}

/// Small K (<= 32) clamps tile_k to rounded-up K.
#[test]
fn tile_select_small_k_clamp() {
    let cfg = select_gemm_tiles(64, 16, 64).unwrap();
    assert_eq!(cfg.tile_m, 32);
    assert_eq!(cfg.tile_n, 32);
    // 16 rounded up to next multiple of SIMDGROUP_ALIGN(8) = 16
    assert_eq!(cfg.tile_k, 16);
}

/// Tall-skinny (M >> N) gets TALL_SKINNY tile.
#[test]
fn tile_select_tall_skinny() {
    // M=256, N=32, ratio=8 >= TALL_SKINNY_RATIO(4), both aligned to 8
    let cfg = select_gemm_tiles(256, 64, 32).unwrap();
    assert_eq!(cfg.tile_m, TileConfig::TALL_SKINNY.tile_m);
    assert_eq!(cfg.tile_n, TileConfig::TALL_SKINNY.tile_n);
}

/// Wide (N >> M) gets WIDE tile.
#[test]
fn tile_select_wide() {
    // M=32, N=256, ratio=8 >= WIDE_RATIO(4), both aligned to 8
    let cfg = select_gemm_tiles(32, 64, 256).unwrap();
    assert_eq!(cfg.tile_m, TileConfig::WIDE.tile_m);
    assert_eq!(cfg.tile_n, TileConfig::WIDE.tile_n);
}

/// TileConfig::output_per_threadgroup is tile_m * tile_n.
#[test]
fn tile_config_output_per_threadgroup() {
    assert_eq!(TileConfig::SQUARE.output_per_threadgroup(), 32 * 32);
    assert_eq!(TileConfig::TALL_SKINNY.output_per_threadgroup(), 64 * 16);
    assert_eq!(TileConfig::WIDE.output_per_threadgroup(), 16 * 64);
}

/// TileConfig::threadgroup_count for a given M x N.
#[test]
fn tile_config_threadgroup_count() {
    let cfg = TileConfig::SQUARE;
    // 128 / 32 * 128 / 32 = 4 * 4 = 16 threadgroups
    assert_eq!(cfg.threadgroup_count(128, 128), 16);
    // 33 / 32 = 2, 65 / 32 = 3 -> 6 threadgroups (ceil division)
    assert_eq!(cfg.threadgroup_count(33, 65), 6);
}

/// TileConfig::threads_per_threadgroup is product of threadgroup_size.
#[test]
fn tile_config_threads_per_threadgroup() {
    assert_eq!(TileConfig::SQUARE.threads_per_threadgroup(), 128);
    assert_eq!(TileConfig::TALL_SKINNY.threads_per_threadgroup(), 128);
    assert_eq!(TileConfig::WIDE.threads_per_threadgroup(), 128);
}

/// Tile selection constants have expected values.
#[test]
fn tile_selection_constants() {
    assert_eq!(SIMDGROUP_ALIGN, 8);
    assert_eq!(TINY_THRESHOLD, 1024);
    assert_eq!(STANDARD_MN_THRESHOLD, 16_384);
    assert_eq!(SMALL_K_THRESHOLD, 32);
    assert_eq!(TALL_SKINNY_RATIO, 4);
    assert_eq!(WIDE_RATIO, 4);
}

// ═══════════════════════════════════════════════════════════════════════
// 8. KernelSource construction and accessors
// ═══════════════════════════════════════════════════════════════════════

/// KernelSource::new creates source with expected defaults.
#[test]
fn kernel_source_new_defaults() {
    let ks = KernelSource::new("kernel void foo() {}", "foo");
    assert_eq!(ks.entry_point(), "foo");
    assert!(!ks.fast_math());
    assert!(ks.function_constants().is_empty());
    assert!(ks.msl_source().contains("kernel void foo"));
}

/// KernelSource with_fast_math builder sets the flag.
#[test]
fn kernel_source_with_fast_math() {
    let ks = KernelSource::new("void k() {}", "k").with_fast_math(true);
    assert!(ks.fast_math());
}

/// KernelSource with_function_constant adds constants.
#[test]
fn kernel_source_with_function_constants() {
    let ks = KernelSource::new("void k() {}", "k")
        .with_function_constant(0, 32)
        .with_function_constant(1, 64);
    assert_eq!(ks.function_constants().len(), 2);
    assert_eq!(ks.function_constants()[0], (0, 32));
    assert_eq!(ks.function_constants()[1], (1, 64));
}

/// KernelSource implements Eq/Hash (used as PipelineCache key).
#[test]
fn kernel_source_eq_and_hash() {
    let a = KernelSource::new("void k() {}", "k").with_fast_math(true);
    let b = KernelSource::new("void k() {}", "k").with_fast_math(true);
    let c = KernelSource::new("void k() {}", "k").with_fast_math(false);
    assert_eq!(a, b);
    assert_ne!(a, c);

    let mut set = HashSet::new();
    set.insert(a);
    assert!(set.contains(&b));
    assert!(!set.contains(&c));
}

// ═══════════════════════════════════════════════════════════════════════
// 9. MetalError — display messages and variant coverage
// ═══════════════════════════════════════════════════════════════════════

/// MetalError::InvalidGridDimension formats correctly.
#[test]
fn metal_error_invalid_grid_dimension() {
    let err = MetalError::InvalidGridDimension {
        dimension: "reduce",
        value: 0,
    };
    assert!(err.to_string().contains("reduce"));
    assert!(err.to_string().contains("non-zero"));
}

/// MetalError::DispatchSizeOverflow formats with the offending count.
#[test]
fn metal_error_dispatch_size_overflow() {
    let err = MetalError::DispatchSizeOverflow(5_000_000_000);
    assert!(err.to_string().contains("5000000000"));
    assert!(err.to_string().contains("u32::MAX"));
}

/// MetalError::BufferByteOverflow includes both elems and elem_size.
#[test]
fn metal_error_buffer_byte_overflow() {
    let err = MetalError::BufferByteOverflow {
        elems: 999,
        elem_size: 8,
    };
    let msg = err.to_string();
    assert!(msg.contains("999"));
    assert!(msg.contains("8"));
}

/// MetalError::ArenaOverflow shows all three size values.
#[test]
fn metal_error_arena_overflow() {
    let err = MetalError::ArenaOverflow {
        requested: 10_000,
        remaining: 5_000,
        capacity: 50_000,
    };
    let msg = err.to_string();
    assert!(msg.contains("10000"));
    assert!(msg.contains("5000"));
    assert!(msg.contains("50000"));
}

/// MetalError converts into nn_core::TensorError.
#[test]
fn metal_error_converts_to_tensor_error() {
    let metal_err = MetalError::NoDevice;
    let tensor_err: nn_core::TensorError = metal_err.into();
    assert!(tensor_err.to_string().contains("unavailable"));
}

// ═══════════════════════════════════════════════════════════════════════
// 10. DispatchProfileEntry — profiler data types
// ═══════════════════════════════════════════════════════════════════════

/// DispatchProfileEntry::duration_ns subtracts start from end.
#[test]
fn profile_entry_duration_ns() {
    let entry = DispatchProfileEntry {
        step_idx: 0,
        op_name: "matmul".into(),
        dispatch_type: DispatchType::StandardOp("matmul".into()),
        start_ns: 1000,
        end_ns: 5000,
        input_bytes: 4096,
        output_bytes: 2048,
    };
    assert_eq!(entry.duration_ns(), 4000);
    assert!((entry.duration_us() - 4.0).abs() < 1e-9);
}

/// DispatchProfileEntry::total_bytes sums input and output.
#[test]
fn profile_entry_total_bytes() {
    let entry = DispatchProfileEntry {
        step_idx: 1,
        op_name: "softmax".into(),
        dispatch_type: DispatchType::NativeOp("softmax".into()),
        start_ns: 0,
        end_ns: 1000,
        input_bytes: 8192,
        output_bytes: 8192,
    };
    assert_eq!(entry.total_bytes(), 16384);
}

/// DispatchProfileEntry::bandwidth_gbps returns 0.0 for zero duration.
#[test]
fn profile_entry_bandwidth_zero_duration() {
    let entry = DispatchProfileEntry {
        step_idx: 0,
        op_name: "relu".into(),
        dispatch_type: DispatchType::FusedKernel("relu".into()),
        start_ns: 100,
        end_ns: 100, // zero duration
        input_bytes: 4096,
        output_bytes: 4096,
    };
    assert_eq!(entry.bandwidth_gbps(), 0.0);
}

/// DispatchType categories are distinct.
#[test]
fn dispatch_type_categories() {
    let a = DispatchType::NativeOp("lstm".into());
    let b = DispatchType::FusedKernel("fused_relu_add".into());
    let c = DispatchType::StandardOp("conv1d".into());

    assert_eq!(a.category(), "native_op");
    assert_eq!(b.category(), "fused_kernel");
    assert_eq!(c.category(), "standard_op");

    assert_eq!(a.name(), "lstm");
    assert_eq!(b.name(), "fused_relu_add");
    assert_eq!(c.name(), "conv1d");
}

/// DispatchType Display format includes category and name.
#[test]
fn dispatch_type_display() {
    let dt = DispatchType::NativeOp("FlashAttention".into());
    let s = format!("{dt}");
    assert!(s.contains("NativeOp"));
    assert!(s.contains("FlashAttention"));
}

/// FusionOpportunity fields accessible.
#[test]
fn fusion_opportunity_construction() {
    let opp = FusionOpportunity {
        first_idx: 3,
        second_idx: 4,
        first_op: "relu".into(),
        second_op: "add".into(),
        combined_ns: 500,
        saved_bytes: 16384,
    };
    assert_eq!(opp.first_idx, 3);
    assert_eq!(opp.second_idx, 4);
    assert_eq!(opp.saved_bytes, 16384);
}

/// TypeBreakdown default values are zero.
#[test]
fn type_breakdown_default() {
    let tb = TypeBreakdown::default();
    assert_eq!(tb.total_ns, 0);
    assert_eq!(tb.count, 0);
    assert_eq!(tb.total_bytes, 0);
}

/// TopEntry fields accessible.
#[test]
fn top_entry_construction() {
    let entry = TopEntry {
        step_idx: 7,
        op_name: "gemm".into(),
        dispatch_type: "native_op".into(),
        duration_ns: 25_000,
        total_bytes: 1_048_576,
        bandwidth_gbps: 42.0,
    };
    assert_eq!(entry.step_idx, 7);
    assert_eq!(entry.duration_ns, 25_000);
}

// ═══════════════════════════════════════════════════════════════════════
// 11. DispatchStats — counters start at zero and are snapshot-stable
// ═══════════════════════════════════════════════════════════════════════

/// After reset_counters, all stats are zero.
#[test]
fn dispatch_stats_after_reset_all_zero() {
    reset_counters();
    let s = dispatch_stats();
    assert_eq!(s.compute_encodings, 0);
    assert_eq!(s.blits, 0);
    assert_eq!(s.flushes, 0);
    assert_eq!(s.submits, 0);
}

/// Consecutive stat reads return the same values.
#[test]
fn dispatch_stats_consecutive_reads_stable() {
    reset_counters();
    let s1 = dispatch_stats();
    let s2 = dispatch_stats();
    assert_eq!(s1, s2);
}

// ═══════════════════════════════════════════════════════════════════════
// 12. F16AutocastConfig — builder patterns
// ═══════════════════════════════════════════════════════════════════════

/// F16AutocastConfig::all enables all 8 segments.
#[test]
fn f16_autocast_config_all_enabled() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    let cfg = crate::F16AutocastConfig::all(MixedPrecisionPolicy::apple_silicon_default());
    assert!(cfg.plbert);
    assert!(cfg.text);
    assert!(cfg.prosody);
    assert!(cfg.f0);
    assert!(cfg.generator);
    assert!(cfg.regulate);
    assert!(cfg.sinegen_pre);
    assert!(cfg.sinegen_post);
}

/// F16AutocastConfig::none disables all segments.
#[test]
fn f16_autocast_config_none_disabled() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    let cfg = crate::F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default());
    assert!(!cfg.plbert);
    assert!(!cfg.text);
    assert!(!cfg.prosody);
    assert!(!cfg.f0);
    assert!(!cfg.generator);
    assert!(!cfg.regulate);
    assert!(!cfg.sinegen_pre);
    assert!(!cfg.sinegen_post);
}

/// F16AutocastConfig::recommended enables compute-heavy segments.
#[test]
fn f16_autocast_config_recommended() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    let cfg = crate::F16AutocastConfig::recommended(MixedPrecisionPolicy::apple_silicon_default());
    assert!(cfg.plbert);
    assert!(cfg.text);
    assert!(cfg.prosody);
    assert!(cfg.f0);
    assert!(cfg.generator);
    assert!(!cfg.regulate);
    assert!(!cfg.sinegen_pre);
    assert!(cfg.sinegen_post);
}

/// F16AutocastConfig::generator_only enables only the generator.
#[test]
fn f16_autocast_config_generator_only() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    let cfg = crate::F16AutocastConfig::generator_only(MixedPrecisionPolicy::apple_silicon_default());
    assert!(!cfg.plbert);
    assert!(!cfg.text);
    assert!(cfg.generator);
    assert!(!cfg.regulate);
}

/// F16AutocastConfig builder pattern chains.
#[test]
fn f16_autocast_config_builder_chain() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    let cfg = crate::F16AutocastConfig::none(MixedPrecisionPolicy::apple_silicon_default())
        .with_plbert(true)
        .with_generator(true);
    assert!(cfg.plbert);
    assert!(cfg.generator);
    assert!(!cfg.text);
    assert!(!cfg.prosody);
}

// ═══════════════════════════════════════════════════════════════════════
// 13. PeepholeConfig defaults and field count
// ═══════════════════════════════════════════════════════════════════════

/// PeepholeConfig::default() has all passes enabled.
#[test]
fn peephole_config_default_all_enabled() {
    let cfg = PeepholeConfig::default();
    assert!(cfg.norm_activ_conv1d);
    assert!(cfg.fused_resblock);
    assert!(cfg.linear_activation);
    assert!(cfg.add_layer_norm);
    assert!(cfg.norm_linear);
    assert!(cfg.attention_transpose);
    assert!(cfg.flip_lstm);
    assert!(cfg.batched_linear_projection);
    assert!(cfg.channels_first_layer_norm);
    assert!(cfg.silu_mul);
    assert!(cfg.auto_fuse_elementwise);
    assert!(cfg.bilstm_cat);
    assert!(cfg.add_norm_linear);
    assert!(cfg.fuse_adain_snake);
    assert!(cfg.fuse_upsample_conv1d);
}

// ═══════════════════════════════════════════════════════════════════════
// 14. SizeClassAllocator stats types
// ═══════════════════════════════════════════════════════════════════════

/// SizeClassStats default is all zeros.
#[test]
fn size_class_stats_default_zero() {
    let s = crate::SizeClassStats::default();
    assert_eq!(s.hits, 0);
    assert_eq!(s.misses, 0);
    assert_eq!(s.free_count, 0);
    assert_eq!(s.in_use_count, 0);
    assert_eq!(s.peak_in_use, 0);
    assert_eq!(s.free_bytes, 0);
    assert_eq!(s.total_allocs(), 0);
    assert_eq!(s.hit_rate(), 0.0);
}

/// SizeClassStats::total_allocs is hits + misses.
#[test]
fn size_class_stats_total_allocs() {
    let s = crate::SizeClassStats {
        hits: 10,
        misses: 5,
        ..Default::default()
    };
    assert_eq!(s.total_allocs(), 15);
}

/// SizeClassStats::hit_rate computes correctly.
#[test]
fn size_class_stats_hit_rate() {
    let s = crate::SizeClassStats {
        hits: 7,
        misses: 3,
        ..Default::default()
    };
    assert!((s.hit_rate() - 0.7).abs() < 1e-9);
}

/// BufferPoolSizeClassStats default has zero fragmentation.
#[test]
fn buffer_pool_stats_default() {
    let s = crate::BufferPoolSizeClassStats::default();
    assert_eq!(s.oversized_allocs, 0);
    assert_eq!(s.total_free_bytes, 0);
    assert_eq!(s.total_used_bytes, 0);
    assert_eq!(s.fragmentation_ratio, 0.0);
    assert_eq!(s.hit_rate, 0.0);
}

/// SIZE_CLASS_BOUNDARIES has 8 entries in ascending order.
#[test]
fn size_class_boundaries_ascending() {
    use crate::buffer_pool_size_class::{NUM_SIZE_CLASSES, SIZE_CLASS_BOUNDARIES};
    assert_eq!(NUM_SIZE_CLASSES, 8);
    for i in 1..NUM_SIZE_CLASSES {
        assert!(
            SIZE_CLASS_BOUNDARIES[i] > SIZE_CLASS_BOUNDARIES[i - 1],
            "class {i} should be > class {}",
            i - 1
        );
    }
    // First class is 4KB, last is 64MB.
    assert_eq!(SIZE_CLASS_BOUNDARIES[0], 4 * 1024);
    assert_eq!(SIZE_CLASS_BOUNDARIES[7], 64 * 1024 * 1024);
}

// ═══════════════════════════════════════════════════════════════════════
// 15. ScalarType ↔ DType mapping for Metal dispatch
// ═══════════════════════════════════════════════════════════════════════

/// ScalarType::F32 MSL string is "float" with byte_size 4.
#[test]
fn scalar_type_f32_msl() {
    assert_eq!(ScalarType::F32.msl_str(), "float");
    assert_eq!(ScalarType::F32.byte_size(), 4);
}

/// ScalarType::F16 MSL string is "half" with byte_size 2.
#[test]
fn scalar_type_f16_msl() {
    assert_eq!(ScalarType::F16.msl_str(), "half");
    assert_eq!(ScalarType::F16.byte_size(), 2);
}

/// dtype_to_msl converts F32 correctly.
#[test]
fn dtype_to_msl_f32() {
    let (msl_str, byte_size) = crate::dtype_to_msl(DType::F32).unwrap();
    assert_eq!(msl_str, "float");
    assert_eq!(byte_size, 4);
}

/// dtype_to_msl converts F16 correctly.
#[test]
fn dtype_to_msl_f16() {
    let (msl_str, byte_size) = crate::dtype_to_msl(DType::F16).unwrap();
    assert_eq!(msl_str, "half");
    assert_eq!(byte_size, 2);
}

/// dtype_to_msl converts BF16 correctly.
#[test]
fn dtype_to_msl_bf16() {
    let (msl_str, byte_size) = crate::dtype_to_msl(DType::BF16).unwrap();
    // BF16 uses bfloat MSL type with 2-byte size
    assert_eq!(byte_size, 2);
    assert!(!msl_str.is_empty());
}

/// dtype_to_msl rejects non-float types.
#[test]
fn dtype_to_msl_rejects_i32() {
    assert!(crate::dtype_to_msl(DType::I32).is_err());
}

// ═══════════════════════════════════════════════════════════════════════
// 16. to_u32 helper — safe usize → u32 conversion
// ═══════════════════════════════════════════════════════════════════════

/// to_u32 converts valid values.
#[test]
fn to_u32_valid_values() {
    assert_eq!(crate::to_u32(0, "test").unwrap(), 0);
    assert_eq!(crate::to_u32(1024, "test").unwrap(), 1024);
    assert_eq!(
        crate::to_u32(u32::MAX as usize, "test").unwrap(),
        u32::MAX
    );
}

/// to_u32 rejects values above u32::MAX.
#[test]
fn to_u32_rejects_overflow() {
    let val = u32::MAX as usize + 1;
    assert!(crate::to_u32(val, "test").is_err());
}
