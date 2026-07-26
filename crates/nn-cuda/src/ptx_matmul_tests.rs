// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for PTX matmul kernel generation.
//!
//! Tests cover `PtxMatmulConfig` construction, `emit_ptx_matmul` output
//! structure, `ptx_matmul_launch_config` grid/block coverage, tile size
//! constants, and edge cases (sub-tile dimensions, large matrices,
//! non-square shapes).

use super::*;

// -----------------------------------------------------------------------
// PtxMatmulConfig construction
// -----------------------------------------------------------------------

#[test]
fn test_config_new_sets_defaults() {
    let c = PtxMatmulConfig::new("nn_kernel");
    assert_eq!(c.kernel_name, "nn_kernel");
    assert_eq!(c.tile_size, PTX_MATMUL_TILE_SIZE);
    assert_eq!(c.sm_target, "sm_80");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_builder_chaining() {
    let c = PtxMatmulConfig::new("k")
        .with_tile_size(8)
        .with_sm_target("sm_70");
    assert_eq!(c.tile_size, 8);
    assert_eq!(c.sm_target, "sm_70");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_all_valid_tile_sizes() {
    // Every tile size from MIN to MAX should validate
    for tile in PTX_MATMUL_MIN_TILE..=PTX_MATMUL_MAX_TILE {
        let c = PtxMatmulConfig::new("k").with_tile_size(tile);
        assert!(c.validate().is_ok(), "tile_size={tile} should be valid");
    }
}

#[test]
fn test_config_invalid_tile_sizes() {
    for &tile in &[0, 1, 2, 3, 33, 64, 128, 256, 1024, usize::MAX] {
        let c = PtxMatmulConfig::new("k").with_tile_size(tile);
        assert!(c.validate().is_err(), "tile_size={tile} should be invalid");
    }
}

#[test]
fn test_config_empty_kernel_name_is_invalid() {
    let c = PtxMatmulConfig {
        kernel_name: String::new(),
        tile_size: 16,
        sm_target: "sm_80".into(),
    };
    assert!(c.validate().is_err());
}

#[test]
fn test_config_shared_memory_bytes_all_tiles() {
    // shared_memory_bytes = 2 * tile^2 * 4
    for tile in PTX_MATMUL_MIN_TILE..=PTX_MATMUL_MAX_TILE {
        let c = PtxMatmulConfig::new("k").with_tile_size(tile);
        let expected = 2 * tile * tile * 4;
        assert_eq!(
            c.shared_memory_bytes(),
            expected,
            "shared memory mismatch for tile={tile}"
        );
    }
}

#[test]
fn test_config_threads_per_block_all_tiles() {
    for tile in PTX_MATMUL_MIN_TILE..=PTX_MATMUL_MAX_TILE {
        let c = PtxMatmulConfig::new("k").with_tile_size(tile);
        assert_eq!(
            c.threads_per_block(),
            tile * tile,
            "threads_per_block mismatch for tile={tile}"
        );
    }
}

#[test]
fn test_config_default_impl() {
    let c = PtxMatmulConfig::default();
    assert_eq!(c.kernel_name, "ptx_matmul_f32");
    assert_eq!(c.tile_size, 16);
    assert_eq!(c.sm_target, "sm_80");
}

// -----------------------------------------------------------------------
// Tile size constants
// -----------------------------------------------------------------------

#[test]
fn test_tile_constants_ordering() {
    assert!(
        PTX_MATMUL_MIN_TILE <= PTX_MATMUL_TILE_SIZE,
        "default tile must be >= min"
    );
    assert!(
        PTX_MATMUL_TILE_SIZE <= PTX_MATMUL_MAX_TILE,
        "default tile must be <= max"
    );
    assert!(
        PTX_MATMUL_MIN_TILE < PTX_MATMUL_MAX_TILE,
        "min must be strictly less than max"
    );
}

#[test]
fn test_tile_default_is_power_of_2() {
    assert!(
        PTX_MATMUL_TILE_SIZE.is_power_of_two(),
        "default tile size {PTX_MATMUL_TILE_SIZE} is not a power of 2"
    );
}

#[test]
fn test_tile_min_is_power_of_2() {
    assert!(
        PTX_MATMUL_MIN_TILE.is_power_of_two(),
        "min tile size {PTX_MATMUL_MIN_TILE} is not a power of 2"
    );
}

#[test]
fn test_tile_max_is_power_of_2() {
    assert!(
        PTX_MATMUL_MAX_TILE.is_power_of_two(),
        "max tile size {PTX_MATMUL_MAX_TILE} is not a power of 2"
    );
}

#[test]
fn test_max_threads_per_block_within_gpu_limit() {
    // Most NVIDIA GPUs support up to 1024 threads per block
    let max_threads = PTX_MATMUL_MAX_TILE * PTX_MATMUL_MAX_TILE;
    assert!(
        max_threads <= 1024,
        "max tile {PTX_MATMUL_MAX_TILE}x{PTX_MATMUL_MAX_TILE} = {max_threads} threads exceeds GPU limit of 1024"
    );
}

// -----------------------------------------------------------------------
// emit_ptx_matmul: output non-empty and structurally valid
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_output_is_nonempty() {
    let ptx = emit_ptx_matmul(&PtxMatmulConfig::default()).unwrap();
    assert!(!ptx.is_empty(), "PTX output must not be empty");
    // A real PTX matmul kernel should be substantial (> 500 chars)
    assert!(
        ptx.len() > 500,
        "PTX output suspiciously short: {} chars",
        ptx.len()
    );
}

#[test]
fn test_emit_ptx_matmul_rejects_invalid_config() {
    let c = PtxMatmulConfig::new("k").with_tile_size(2);
    assert!(emit_ptx_matmul(&c).is_err());

    let c = PtxMatmulConfig::new("").with_tile_size(16);
    assert!(emit_ptx_matmul(&c).is_err());

    let c = PtxMatmulConfig::new("k").with_tile_size(64);
    assert!(emit_ptx_matmul(&c).is_err());
}

#[test]
fn test_emit_ptx_matmul_contains_kernel_name() {
    for name in &["gemm_f32", "nn_matmul", "custom_kernel_123"] {
        let ptx = emit_ptx_matmul(&PtxMatmulConfig::new(name)).unwrap();
        assert!(
            ptx.contains(&format!(".entry {name}")),
            "PTX must contain entry point for kernel '{name}'"
        );
    }
}

#[test]
fn test_emit_ptx_matmul_ptx_header() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    assert!(
        ptx.contains(".version"),
        "must contain PTX version directive"
    );
    assert!(ptx.contains(".target"), "must contain target directive");
    assert!(
        ptx.contains(".address_size 64"),
        "must declare 64-bit addressing"
    );
}

#[test]
fn test_emit_ptx_matmul_shared_memory_declarations() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    assert!(ptx.contains(".shared .align 4 .f32 As["));
    assert!(ptx.contains(".shared .align 4 .f32 Bs["));
}

#[test]
fn test_emit_ptx_matmul_kernel_parameters() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    // A, B, C pointers are .u64; M, N, K are .u32
    assert!(ptx.contains(".param .u64 param_A"));
    assert!(ptx.contains(".param .u64 param_B"));
    assert!(ptx.contains(".param .u64 param_C"));
    assert!(ptx.contains(".param .u32 param_M"));
    assert!(ptx.contains(".param .u32 param_N"));
    assert!(ptx.contains(".param .u32 param_K"));
}

#[test]
fn test_emit_ptx_matmul_thread_indices() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    // PTX thread index registers
    assert!(ptx.contains("%tid.x"), "must read threadIdx.x");
    assert!(ptx.contains("%tid.y"), "must read threadIdx.y");
    assert!(ptx.contains("%ctaid.x"), "must read blockIdx.x");
    assert!(ptx.contains("%ctaid.y"), "must read blockIdx.y");
}

#[test]
fn test_emit_ptx_matmul_tiling_loop_structure() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    // Tile loop labels
    assert!(
        ptx.contains("TILE_LOOP:"),
        "must have tile loop entry label"
    );
    assert!(ptx.contains("TILE_DONE:"), "must have tile loop exit label");
    // Inner dot product loop labels
    assert!(
        ptx.contains("DOT_LOOP:"),
        "must have dot product loop label"
    );
    assert!(
        ptx.contains("DOT_DONE:"),
        "must have dot product exit label"
    );
}

#[test]
fn test_emit_ptx_matmul_bounds_check_labels() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    assert!(
        ptx.contains("SKIP_LOAD_A:"),
        "must have bounds-check skip for A tile"
    );
    assert!(
        ptx.contains("SKIP_LOAD_B:"),
        "must have bounds-check skip for B tile"
    );
    assert!(
        ptx.contains("KERNEL_EXIT:"),
        "must have kernel exit label for out-of-bounds threads"
    );
}

#[test]
fn test_emit_ptx_matmul_barrier_sync() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    let barrier_count = ptx.matches("bar.sync").count();
    assert!(
        barrier_count >= 2,
        "need at least 2 barriers (post-load, post-dot), got {barrier_count}"
    );
}

#[test]
fn test_emit_ptx_matmul_fma_instruction() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    assert!(
        ptx.contains("fma.rn.f32"),
        "must use fused multiply-add for accumulation"
    );
}

#[test]
fn test_emit_ptx_matmul_global_memory_access() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    assert!(
        ptx.contains("ld.global.f32"),
        "must load from global memory"
    );
    assert!(ptx.contains("st.global.f32"), "must store to global memory");
}

#[test]
fn test_emit_ptx_matmul_shared_memory_access() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    assert!(
        ptx.contains("ld.shared.f32"),
        "must load from shared memory"
    );
    assert!(ptx.contains("st.shared.f32"), "must store to shared memory");
}

#[test]
fn test_emit_ptx_matmul_is_pure_ptx() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    // Must NOT contain CUDA C++ keywords as actual code (comments are OK)
    assert!(!ptx.contains("__global__"));
    assert!(!ptx.contains("__shared__"));
    assert!(!ptx.contains("__syncthreads"));
    assert!(!ptx.contains("#include"));
    // PTX uses %tid.x / %ctaid.x, not CUDA C++ threadIdx / blockIdx
    // (comments may mention threadIdx for documentation, which is fine)
    assert!(
        !ptx.contains("blockDim."),
        "must not contain CUDA C++ blockDim references"
    );
}

#[test]
fn test_emit_ptx_matmul_returns_and_braces() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    assert!(ptx.contains("ret;"), "kernel must end with ret");
    // PTX entry must have matching braces
    assert!(ptx.contains('{'), "must have opening brace");
    assert!(ptx.contains('}'), "must have closing brace");
}

// -----------------------------------------------------------------------
// emit_ptx_matmul with various tile sizes
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_tile_4_shared_memory_size() {
    let c = PtxMatmulConfig::new("gemm4").with_tile_size(4);
    let ptx = emit_ptx_matmul(&c).unwrap();
    assert!(ptx.contains("As[16]"), "4x4 tile -> As[16]");
    assert!(ptx.contains("Bs[16]"), "4x4 tile -> Bs[16]");
    assert!(ptx.contains(".reqntid 4, 4"));
}

#[test]
fn test_emit_ptx_matmul_tile_8() {
    let c = PtxMatmulConfig::new("gemm8").with_tile_size(8);
    let ptx = emit_ptx_matmul(&c).unwrap();
    assert!(ptx.contains("As[64]"), "8x8 tile -> As[64]");
    assert!(ptx.contains("Bs[64]"));
    assert!(ptx.contains(".reqntid 8, 8"));
    assert!(ptx.contains(".entry gemm8"));
}

#[test]
fn test_emit_ptx_matmul_tile_16() {
    let ptx = emit_ptx_matmul_default("gemm16").unwrap();
    assert!(ptx.contains("As[256]"), "16x16 tile -> As[256]");
    assert!(ptx.contains("Bs[256]"));
    assert!(ptx.contains(".reqntid 16, 16"));
}

#[test]
fn test_emit_ptx_matmul_tile_32() {
    let c = PtxMatmulConfig::new("gemm32").with_tile_size(32);
    let ptx = emit_ptx_matmul(&c).unwrap();
    assert!(ptx.contains("As[1024]"), "32x32 tile -> As[1024]");
    assert!(ptx.contains("Bs[1024]"));
    assert!(ptx.contains(".reqntid 32, 32"));
}

#[test]
fn test_emit_ptx_matmul_all_valid_power_of_2_tiles() {
    // Test every power-of-2 tile in the valid range
    for tile in [4, 8, 16, 32] {
        if !(PTX_MATMUL_MIN_TILE..=PTX_MATMUL_MAX_TILE).contains(&tile) {
            continue;
        }
        let name = format!("gemm_t{tile}");
        let c = PtxMatmulConfig::new(&name).with_tile_size(tile);
        let ptx = emit_ptx_matmul(&c).unwrap();
        let expected_shared = tile * tile;
        assert!(
            ptx.contains(&format!("As[{expected_shared}]")),
            "tile={tile}: expected As[{expected_shared}]"
        );
        assert!(
            ptx.contains(&format!(".reqntid {tile}, {tile}")),
            "tile={tile}: expected .reqntid {tile}, {tile}"
        );
    }
}

#[test]
fn test_emit_ptx_matmul_non_power_of_2_tiles() {
    // Non-power-of-2 tiles in [4, 32] should also work
    for tile in [5, 6, 7, 9, 10, 12, 15, 20, 24, 30] {
        let c = PtxMatmulConfig::new("k").with_tile_size(tile);
        let ptx = emit_ptx_matmul(&c).unwrap();
        let expected_shared = tile * tile;
        assert!(
            ptx.contains(&format!("As[{expected_shared}]")),
            "non-pow2 tile={tile}: expected As[{expected_shared}]"
        );
    }
}

// -----------------------------------------------------------------------
// emit_ptx_matmul_default
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_default_various_names() {
    for name in &["kernel_a", "matmul_f32", "gemm_sm80", "k"] {
        let ptx = emit_ptx_matmul_default(name).unwrap();
        assert!(
            ptx.contains(&format!(".entry {name}")),
            "default emit must produce entry for '{name}'"
        );
    }
}

#[test]
fn test_emit_ptx_matmul_default_matches_explicit_config() {
    let default_ptx = emit_ptx_matmul_default("test").unwrap();
    let config_ptx = emit_ptx_matmul(&PtxMatmulConfig::new("test")).unwrap();
    assert_eq!(
        default_ptx, config_ptx,
        "default helper must produce identical PTX to explicit config"
    );
}

// -----------------------------------------------------------------------
// Custom SM targets
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_sm_targets() {
    for target in &["sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90"] {
        let c = PtxMatmulConfig::new("k").with_sm_target(target);
        let ptx = emit_ptx_matmul(&c).unwrap();
        assert!(
            ptx.contains(&format!(".target {target}")),
            "PTX must target {target}"
        );
    }
}

// -----------------------------------------------------------------------
// ptx_matmul_launch_config: grid/block coverage
// -----------------------------------------------------------------------

#[test]
fn test_launch_config_square_powers_of_2() {
    for &size in &[16, 32, 64, 128, 256, 512, 1024, 2048] {
        let tile = 16;
        let (grid, block) = ptx_matmul_launch_config(size, size, tile);
        assert_eq!(block, [tile, tile, 1]);
        let expected_grid_dim = size.div_ceil(tile);
        assert_eq!(grid[0], expected_grid_dim, "grid_x for {size}x{size}");
        assert_eq!(grid[1], expected_grid_dim, "grid_y for {size}x{size}");
        assert_eq!(grid[2], 1);
    }
}

#[test]
fn test_launch_config_non_square() {
    // M != N != K (K doesn't affect grid dims)
    let (grid, block) = ptx_matmul_launch_config(100, 200, 16);
    assert_eq!(block, [16, 16, 1]);
    // grid_x = ceil(N/tile) = ceil(200/16) = 13
    assert_eq!(grid[0], 13);
    // grid_y = ceil(M/tile) = ceil(100/16) = 7
    assert_eq!(grid[1], 7);
    assert_eq!(grid[2], 1);
}

#[test]
fn test_launch_config_large_matrices() {
    let (grid, block) = ptx_matmul_launch_config(2048, 2048, 16);
    assert_eq!(grid, [128, 128, 1]);
    assert_eq!(block, [16, 16, 1]);

    let (grid, block) = ptx_matmul_launch_config(4096, 4096, 32);
    assert_eq!(grid, [128, 128, 1]);
    assert_eq!(block, [32, 32, 1]);
}

#[test]
fn test_launch_config_grid_covers_all_output_elements() {
    // For various M, N, tile combinations, verify grid * block >= dimension
    let test_cases: Vec<(usize, usize, usize)> = vec![
        (1, 1, 4),
        (5, 5, 4),
        (15, 15, 16),
        (16, 16, 16),
        (17, 17, 16),
        (31, 33, 8),
        (100, 200, 16),
        (127, 255, 32),
        (128, 256, 32),
        (1000, 2000, 16),
        (2048, 4096, 32),
    ];

    for (m, n, tile) in test_cases {
        let (grid, block) = ptx_matmul_launch_config(m, n, tile);

        let total_x = grid[0] * block[0];
        let total_y = grid[1] * block[1];

        assert!(
            total_x >= n,
            "grid_x * block_x = {total_x} must cover N={n} (tile={tile})"
        );
        assert!(
            total_y >= m,
            "grid_y * block_y = {total_y} must cover M={m} (tile={tile})"
        );

        // Over-coverage should be at most one tile minus one
        assert!(
            total_x < n + tile,
            "grid_x * block_x = {total_x} over-covers N={n} by more than tile-1"
        );
        assert!(
            total_y < m + tile,
            "grid_y * block_y = {total_y} over-covers M={m} by more than tile-1"
        );
    }
}

#[test]
fn test_launch_config_exact_multiples() {
    // When M and N are exact multiples of tile, no over-coverage
    let (grid, block) = ptx_matmul_launch_config(256, 512, 16);
    assert_eq!(grid[0] * block[0], 512); // exactly N
    assert_eq!(grid[1] * block[1], 256); // exactly M
}

#[test]
fn test_launch_config_z_dim_always_1() {
    for &(m, n, tile) in &[(64, 64, 8), (100, 200, 16), (4096, 4096, 32)] {
        let (grid, block) = ptx_matmul_launch_config(m, n, tile);
        assert_eq!(grid[2], 1, "grid z must always be 1");
        assert_eq!(block[2], 1, "block z must always be 1");
    }
}

// -----------------------------------------------------------------------
// Edge case: dimensions smaller than tile size
// -----------------------------------------------------------------------

#[test]
fn test_launch_config_m_smaller_than_tile() {
    // M=5 with tile=16: needs 1 block in y dimension
    let (grid, block) = ptx_matmul_launch_config(5, 64, 16);
    assert_eq!(grid[1], 1, "ceil(5/16) = 1");
    assert_eq!(grid[0], 4, "ceil(64/16) = 4");
    assert_eq!(block, [16, 16, 1]);
}

#[test]
fn test_launch_config_n_smaller_than_tile() {
    // N=3 with tile=16: needs 1 block in x dimension
    let (grid, block) = ptx_matmul_launch_config(128, 3, 16);
    assert_eq!(grid[0], 1, "ceil(3/16) = 1");
    assert_eq!(grid[1], 8, "ceil(128/16) = 8");
    assert_eq!(block, [16, 16, 1]);
}

#[test]
fn test_launch_config_both_smaller_than_tile() {
    // Both M and N smaller than tile: single block
    let (grid, block) = ptx_matmul_launch_config(1, 1, 16);
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [16, 16, 1]);
}

#[test]
fn test_launch_config_m_equals_1_vector_row() {
    // Row vector: M=1 (single row)
    let (grid, _block) = ptx_matmul_launch_config(1, 1024, 16);
    assert_eq!(grid[1], 1, "single row -> 1 y block");
    assert_eq!(grid[0], 64, "ceil(1024/16) = 64");
}

#[test]
fn test_launch_config_n_equals_1_vector_col() {
    // Column vector: N=1 (single column)
    let (grid, _block) = ptx_matmul_launch_config(1024, 1, 16);
    assert_eq!(grid[0], 1, "single column -> 1 x block");
    assert_eq!(grid[1], 64, "ceil(1024/16) = 64");
}

// -----------------------------------------------------------------------
// PTX generation for non-square matrix dimensions
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_works_for_any_runtime_dimensions() {
    // The PTX kernel handles M, N, K at runtime via parameters.
    // Verify the generated PTX is structurally valid regardless of
    // what dimensions we intend to use it with.
    let ptx = emit_ptx_matmul_default("nonsquare_gemm").unwrap();

    // Runtime dimension params must exist
    assert!(ptx.contains("param_M"));
    assert!(ptx.contains("param_N"));
    assert!(ptx.contains("param_K"));

    // Bounds checking ensures correct behavior for any M, N, K
    assert!(ptx.contains("setp.lt.u32"), "must bounds-check rows/cols");
    assert!(ptx.contains("and.pred"), "must combine row/col predicates");
}

// -----------------------------------------------------------------------
// PTX generation: comment and documentation quality
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_contains_algorithm_comment() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    // Should contain a comment describing the algorithm
    assert!(
        ptx.contains("Tiled f32 GEMM") || ptx.contains("C[M,N] = A[M,K] * B[K,N]"),
        "PTX should contain algorithm description comment"
    );
}

#[test]
fn test_emit_ptx_matmul_contains_shared_memory_size_comment() {
    let c = PtxMatmulConfig::new("k").with_tile_size(16);
    let ptx = emit_ptx_matmul(&c).unwrap();
    let expected_bytes = c.shared_memory_bytes();
    assert!(
        ptx.contains(&format!("{expected_bytes} bytes")),
        "PTX should document shared memory usage ({expected_bytes} bytes)"
    );
}

// -----------------------------------------------------------------------
// PTX instruction completeness
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_all_required_instructions() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    let required = [
        ("ld.param", "parameter loading"),
        ("mov.u32", "register moves"),
        ("mov.f32", "float register init"),
        ("mad.lo.u32", "index multiply-add"),
        ("mul.wide.u32", "widening multiply for byte offsets"),
        ("add.u32", "integer addition (loop increment)"),
        ("add.u64", "64-bit pointer arithmetic"),
        ("div.u32", "integer division (tile count)"),
        ("ld.global.f32", "global memory load"),
        ("st.global.f32", "global memory store"),
        ("ld.shared.f32", "shared memory load"),
        ("st.shared.f32", "shared memory store"),
        ("fma.rn.f32", "fused multiply-add"),
        ("setp.lt.u32", "predicate: less than"),
        ("setp.ge.u32", "predicate: greater-or-equal"),
        ("and.pred", "predicate AND"),
        ("bar.sync", "thread barrier"),
        ("bra", "branch"),
        ("ret", "return"),
    ];
    for (instr, desc) in &required {
        assert!(
            ptx.contains(instr),
            "PTX missing required instruction '{instr}' ({desc})"
        );
    }
}

// -----------------------------------------------------------------------
// Register declarations
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_register_declarations() {
    let ptx = emit_ptx_matmul_default("k").unwrap();
    // Must declare all 4 register types
    assert!(ptx.contains(".reg .u32"), "must declare u32 registers");
    assert!(ptx.contains(".reg .f32"), "must declare f32 registers");
    assert!(ptx.contains(".reg .u64"), "must declare u64 registers");
    assert!(
        ptx.contains(".reg .pred"),
        "must declare predicate registers"
    );
}

// -----------------------------------------------------------------------
// Deterministic output
// -----------------------------------------------------------------------

#[test]
fn test_emit_ptx_matmul_deterministic() {
    let c = PtxMatmulConfig::new("det_test").with_tile_size(8);
    let ptx1 = emit_ptx_matmul(&c).unwrap();
    let ptx2 = emit_ptx_matmul(&c).unwrap();
    assert_eq!(ptx1, ptx2, "PTX generation must be deterministic");
}

#[test]
fn test_emit_ptx_matmul_different_configs_differ() {
    let ptx_16 = emit_ptx_matmul(&PtxMatmulConfig::new("k").with_tile_size(16)).unwrap();
    let ptx_8 = emit_ptx_matmul(&PtxMatmulConfig::new("k").with_tile_size(8)).unwrap();
    assert_ne!(
        ptx_16, ptx_8,
        "different tile sizes must produce different PTX"
    );
}

#[test]
fn test_emit_ptx_matmul_different_names_differ() {
    let ptx_a = emit_ptx_matmul_default("kernel_a").unwrap();
    let ptx_b = emit_ptx_matmul_default("kernel_b").unwrap();
    assert_ne!(
        ptx_a, ptx_b,
        "different kernel names must produce different PTX"
    );
}

// -----------------------------------------------------------------------
// CPU reference: matmul_reference
// -----------------------------------------------------------------------

#[test]
fn test_matmul_reference_identity_2x2() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let identity = vec![1.0, 0.0, 0.0, 1.0];
    let c = matmul_reference(&a, &identity, 2, 2, 2);
    assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_matmul_reference_identity_3x3() {
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let c = matmul_reference(&identity, &b, 3, 3, 3);
    assert_eq!(c, b);
}

#[test]
fn test_matmul_reference_known_2x2() {
    // [1 2] * [5 6] = [19 22]
    // [3 4]   [7 8]   [43 50]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];
    let c = matmul_reference(&a, &b, 2, 2, 2);
    assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matmul_reference_non_square() {
    // A[2,3] * B[3,2] = C[2,2]
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c = matmul_reference(&a, &b, 2, 3, 2);
    assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matmul_reference_dot_product() {
    // [1,3] * [3,1] = scalar
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let c = matmul_reference(&a, &b, 1, 3, 1);
    assert_eq!(c, vec![32.0]);
}

#[test]
fn test_matmul_reference_zeros() {
    let a = vec![0.0; 4];
    let b = vec![1.0, 2.0, 3.0, 4.0];
    let c = matmul_reference(&a, &b, 2, 2, 2);
    assert_eq!(c, vec![0.0; 4]);
}

#[test]
fn test_matmul_reference_wide_output() {
    // A[1,2] * B[2,4] = C[1,4]
    let a = vec![1.0, 2.0];
    let b = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let c = matmul_reference(&a, &b, 1, 2, 4);
    assert_eq!(c, vec![1.0, 2.0, 0.0, 0.0]);
}

#[test]
fn test_matmul_reference_tall_output() {
    // A[4,1] * B[1,1] = C[4,1]
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0];
    let c = matmul_reference(&a, &b, 4, 1, 1);
    assert_eq!(c, vec![2.0, 4.0, 6.0, 8.0]);
}

// -----------------------------------------------------------------------
// Naive matmul PTX generation: generate_matmul_ptx
// -----------------------------------------------------------------------

#[test]
fn test_generate_matmul_ptx_entry_point() {
    let ptx = generate_matmul_ptx(4, 4, 4);
    assert!(ptx.contains(".entry naive_matmul_f32"));
}

#[test]
fn test_generate_matmul_ptx_sm70_target() {
    let ptx = generate_matmul_ptx(8, 8, 8);
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_generate_matmul_ptx_all_params() {
    let ptx = generate_matmul_ptx(4, 4, 4);
    assert!(ptx.contains("param_A"));
    assert!(ptx.contains("param_B"));
    assert!(ptx.contains("param_C"));
    assert!(ptx.contains("param_M"));
    assert!(ptx.contains("param_N"));
    assert!(ptx.contains("param_K"));
}

#[test]
fn test_generate_matmul_ptx_no_shared_memory() {
    let ptx = generate_matmul_ptx(16, 16, 16);
    assert!(
        !ptx.contains(".shared"),
        "naive kernel must not use shared memory"
    );
}

#[test]
fn test_generate_matmul_ptx_has_fma() {
    let ptx = generate_matmul_ptx(4, 4, 4);
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn test_generate_matmul_ptx_dimension_comments() {
    let ptx = generate_matmul_ptx(32, 64, 16);
    assert!(ptx.contains("C[32,16]"));
    assert!(ptx.contains("A[32,64]"));
    assert!(ptx.contains("B[64,16]"));
}

// -----------------------------------------------------------------------
// Tiled matmul wrapper: generate_matmul_tiled_ptx
// -----------------------------------------------------------------------

#[test]
fn test_generate_matmul_tiled_ptx_entry() {
    let ptx = generate_matmul_tiled_ptx(4, 4, 4, 4);
    assert!(ptx.contains(".entry tiled_matmul_f32"));
}

#[test]
fn test_generate_matmul_tiled_ptx_uses_shared_memory() {
    let ptx = generate_matmul_tiled_ptx(16, 16, 16, 16);
    assert!(ptx.contains(".shared .align 4"));
}

#[test]
fn test_generate_matmul_tiled_ptx_dimension_header() {
    let ptx = generate_matmul_tiled_ptx(128, 256, 64, 16);
    assert!(ptx.contains("M=128"));
    assert!(ptx.contains("K=256"));
    assert!(ptx.contains("N=64"));
    assert!(ptx.contains("tile=16"));
}

#[test]
fn test_generate_matmul_tiled_ptx_tile8_shared_size() {
    let ptx = generate_matmul_tiled_ptx(8, 8, 8, 8);
    assert!(ptx.contains("As[64]"));
    assert!(ptx.contains("Bs[64]"));
}

#[test]
fn test_generate_matmul_tiled_ptx_has_barrier() {
    let ptx = generate_matmul_tiled_ptx(16, 16, 16, 16);
    assert!(ptx.contains("bar.sync"));
}

// -----------------------------------------------------------------------
// MATMUL_BLOCK_SIZE constant
// -----------------------------------------------------------------------

#[test]
fn test_matmul_block_size_value() {
    assert_eq!(MATMUL_BLOCK_SIZE, 16);
}
