// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for MoE grouped GEMM kernel emission.

use super::*;

// -- Grouped GEMM tests -------------------------------------------------------

#[test]
fn test_grouped_gemm_basic_generation() {
    let src = emit_grouped_gemm_kernel("moe_gemm_test", 8, 7168, 2048, 32768).unwrap();

    assert!(
        src.contains("#include <hip/hip_runtime.h>"),
        "missing HIP include"
    );
    assert!(src.contains("extern \"C\" __global__ void moe_gemm_test("));
    assert!(src.contains("const half* __restrict__ input"));
    assert!(src.contains("const half* __restrict__ weights"));
    assert!(src.contains("half* __restrict__ output"));
    assert!(src.contains("const unsigned int* __restrict__ expert_offsets"));
}

#[test]
fn test_grouped_gemm_dimensions_in_source() {
    let src = emit_grouped_gemm_kernel("k1", 32, 7168, 2048, 65536).unwrap();

    assert!(src.contains("N_EXPERTS = 32"), "wrong n_experts");
    assert!(src.contains("IN_DIM = 7168"), "wrong in_dim");
    assert!(src.contains("OUT_DIM = 2048"), "wrong out_dim");
}

#[test]
fn test_grouped_gemm_expert_offset_lookup() {
    let src = emit_grouped_gemm_kernel("k_off", 8, 2048, 7168, 8192).unwrap();

    // Must scan expert_offsets to find which expert owns the tile row.
    assert!(
        src.contains("expert_offsets[e]"),
        "missing expert offset lookup"
    );
    assert!(
        src.contains("expert_offsets[e + 1u]"),
        "missing expert end offset"
    );
    assert!(
        src.contains("expert_id * OUT_IN"),
        "missing per-expert weight indexing"
    );
}

#[test]
fn test_grouped_gemm_tile_gemm_structure() {
    let src = emit_grouped_gemm_kernel("k_tile", 8, 2048, 2048, 8192).unwrap();

    // Verify tiled GEMM with shared memory.
    assert!(
        src.contains("__shared__ float As["),
        "missing A shared tile"
    );
    assert!(
        src.contains("__shared__ float Bs["),
        "missing B shared tile"
    );
    assert!(
        src.contains("__half2float("),
        "missing half-to-float conversion"
    );
    assert!(
        src.contains("__float2half("),
        "missing float-to-half conversion"
    );
}

#[test]
fn test_grouped_gemm_alignment_error() {
    // in_dim not a multiple of 32.
    let result = emit_grouped_gemm_kernel("bad", 8, 7168, 2000, 8192);
    assert!(result.is_err(), "out_dim=2000 should fail alignment");
    let err = result.unwrap_err().to_string();
    assert!(err.contains("multiples of 32"), "{err}");
}

#[test]
fn test_grouped_gemm_launch_config() {
    // 32 experts, top-4 from 8192 tokens = 32768 total, out_dim = 2048.
    let cfg = grouped_gemm_launch_config(32768, 2048);

    assert_eq!(cfg.grid.x, 2048 / 32); // 64 col tiles
    assert_eq!(cfg.grid.y, 32768 / 32); // 1024 row tiles
    assert_eq!(cfg.grid.z, 1);
    assert_eq!(cfg.block.x, 256);
    assert!(cfg.shared_mem_bytes > 0, "should use shared memory");
}

// -- SwiGLU kernel tests ------------------------------------------------------

#[test]
fn test_swiglu_basic_generation() {
    let src = emit_moe_swiglu_kernel("moe_swiglu_test", 32, 7168, 2048).unwrap();

    assert!(src.contains("extern \"C\" __global__ void moe_swiglu_test("));
    assert!(src.contains("const half* __restrict__ gate_weights"));
    assert!(src.contains("const half* __restrict__ up_weights"));
}

#[test]
fn test_swiglu_silu_computation() {
    let src = emit_moe_swiglu_kernel("k_silu", 8, 2048, 256).unwrap();

    // SwiGLU = silu(gate) * up = gate * sigmoid(gate) * up.
    assert!(src.contains("sigmoid"), "missing sigmoid");
    assert!(src.contains("expf(-gate_val)"), "missing exp for sigmoid");
    assert!(
        src.contains("gate_val * sigmoid_gate * up_val"),
        "missing SwiGLU computation"
    );
}

#[test]
fn test_swiglu_expert_offset_lookup() {
    let src = emit_moe_swiglu_kernel("k_exp", 32, 7168, 2048).unwrap();

    assert!(
        src.contains("expert_offsets[e]"),
        "missing expert offset scan"
    );
    assert!(
        src.contains("expert_id * EXP_HID"),
        "missing per-expert weight offset"
    );
}

#[test]
fn test_swiglu_dimensions() {
    let src = emit_moe_swiglu_kernel("k_dim", 32, 7168, 2048).unwrap();

    assert!(src.contains("D_HIDDEN = 7168"), "wrong d_hidden");
    assert!(src.contains("D_EXPERT = 2048"), "wrong d_expert");
    assert!(src.contains("N_EXPERTS = 32"), "wrong n_experts");
}

#[test]
fn test_swiglu_launch_config() {
    let cfg = moe_swiglu_launch_config(32768, 2048);

    assert_eq!(cfg.grid.x, 32768); // one block per token
    assert_eq!(cfg.grid.y, 8); // ceil(2048/256)
    assert_eq!(cfg.block.x, 256);
}

// -- Permute kernel tests -----------------------------------------------------

#[test]
fn test_permute_basic_generation() {
    let src = emit_moe_permute_kernel("moe_perm", 7168).unwrap();

    assert!(src.contains("extern \"C\" __global__ void moe_perm("));
    assert!(src.contains("const half* __restrict__ input"));
    assert!(src.contains("half* __restrict__ permuted"));
    assert!(
        src.contains("source_token_ids[perm_row]"),
        "missing source token lookup"
    );
}

#[test]
fn test_permute_dimension() {
    let src = emit_moe_permute_kernel("k_dim", 7168).unwrap();
    assert!(src.contains("D_HIDDEN = 7168"), "wrong d_hidden");
}

#[test]
fn test_permute_launch_config() {
    // 32768 tokens × 7168 features = 234_881_024 elements.
    let n_elements = 32768 * 7168;
    let cfg = moe_permute_launch_config(n_elements);

    let expected_blocks = n_elements.div_ceil(256);
    assert_eq!(cfg.grid.x, expected_blocks as u32);
    assert_eq!(cfg.block.x, 256);
}

// -- Un-permute kernel tests --------------------------------------------------

#[test]
fn test_unpermute_basic_generation() {
    let src = emit_moe_unpermute_kernel("moe_unperm", 7168, 4).unwrap();

    assert!(src.contains("extern \"C\" __global__ void moe_unperm("));
    assert!(src.contains("const half* __restrict__ expert_output"));
    // Output is float* for safe atomicAdd accumulation (no half/float mismatch).
    assert!(
        src.contains("float* __restrict__ output"),
        "output must be float for atomicAdd"
    );
    assert!(src.contains("const float* __restrict__ routing_weights"));
}

#[test]
fn test_unpermute_atomic_accumulation() {
    let src = emit_moe_unpermute_kernel("k_atom", 7168, 4).unwrap();

    // Must use atomicAdd to accumulate across k experts for same token.
    assert!(src.contains("atomicAdd"), "missing atomic accumulation");
}

#[test]
fn test_unpermute_routing_weight_application() {
    let src = emit_moe_unpermute_kernel("k_wt", 7168, 4).unwrap();

    assert!(
        src.contains("routing_weights[dst_token * K + k_idx]"),
        "missing routing weight lookup"
    );
    assert!(src.contains("K = 4"), "wrong experts_per_tok");
}

// -- Thread mapping correctness (#2933) ---------------------------------------

/// Verify the GEMM kernel has unique thread-to-output mapping (no redundant work).
///
/// The bug (#2933): `threadIdx.x % TILE` mapped 8 threads to the same row,
/// causing 87.5% wasted compute and a benign write race.
///
/// Fixed: `threadIdx.x / THREADS_PER_ROW` for row, `threadIdx.x % THREADS_PER_ROW`
/// for column group. Each of 256 threads writes 4 unique output elements.
#[test]
fn test_grouped_gemm_unique_thread_mapping() {
    let src = emit_grouped_gemm_kernel("k_thr", 8, 2048, 2048, 8192).unwrap();

    // Verify thread mapping constants are present.
    assert!(
        src.contains("THREADS_PER_ROW"),
        "missing THREADS_PER_ROW constant"
    );
    assert!(
        src.contains("COLS_PER_THREAD"),
        "missing COLS_PER_THREAD constant"
    );

    // Row assignment uses division (not modulo — modulo was the bug).
    assert!(
        src.contains("threadIdx.x / THREADS_PER_ROW"),
        "row_in_tile should use division, not modulo"
    );

    // Column group uses modulo of THREADS_PER_ROW (not TILE).
    assert!(
        src.contains("threadIdx.x % THREADS_PER_ROW"),
        "col_group should use modulo of THREADS_PER_ROW"
    );

    // Accumulator sized to COLS_PER_THREAD (4), not TILE (32).
    assert!(
        src.contains("acc[COLS_PER_THREAD]"),
        "accumulator should be COLS_PER_THREAD elements, not TILE"
    );

    // Write phase uses col_start offset (each thread writes unique columns).
    assert!(
        src.contains("tile_col + col_start + j"),
        "write phase must use col_start for unique output columns"
    );
}

/// Simulate the thread mapping for a 32×32 tile with 256 threads and verify
/// every output element is assigned to exactly one thread.
#[test]
fn test_grouped_gemm_coverage_simulation() {
    let tile: usize = 32;
    let block_size: usize = 256;
    let threads_per_row = block_size / tile; // 8
    let cols_per_thread = tile / threads_per_row; // 4

    // Track which (row, col) each thread is responsible for.
    let mut coverage = vec![0u32; tile * tile]; // tile * tile = 1024

    for tid in 0..block_size {
        let row = tid / threads_per_row;
        let col_group = tid % threads_per_row;
        let col_start = col_group * cols_per_thread;

        for j in 0..cols_per_thread {
            let col = col_start + j;
            assert!(row < tile, "row {row} out of tile range");
            assert!(col < tile, "col {col} out of tile range");
            coverage[row * tile + col] += 1;
        }
    }

    // Every element must be covered exactly once (no gaps, no duplicates).
    for row in 0..tile {
        for col in 0..tile {
            let count = coverage[row * tile + col];
            assert_eq!(
                count, 1,
                "output[{row}][{col}] covered {count} times (expected 1)"
            );
        }
    }
}

// -- Competition dimensions ---------------------------------------------------

#[test]
fn test_deepseek_v3_competition_dimensions() {
    // Competition benchmark: 32 experts, d_hidden=7168, d_expert=2048.
    let gemm_src = emit_grouped_gemm_kernel(
        "dsv3_gemm",
        32,
        7168,
        2048,
        32768, // 8192 tokens × top-4
    )
    .unwrap();
    assert!(gemm_src.contains("N_EXPERTS = 32"));
    assert!(gemm_src.contains("IN_DIM = 7168"));
    assert!(gemm_src.contains("OUT_DIM = 2048"));

    let swiglu_src = emit_moe_swiglu_kernel("dsv3_swiglu", 32, 7168, 2048).unwrap();
    assert!(swiglu_src.contains("D_HIDDEN = 7168"));
    assert!(swiglu_src.contains("D_EXPERT = 2048"));

    // Balanced braces check.
    for (name, src) in [("gemm", &gemm_src), ("swiglu", &swiglu_src)] {
        let opens = src.matches('{').count();
        let closes = src.matches('}').count();
        assert_eq!(
            opens, closes,
            "unbalanced braces in {name}: opens={opens}, closes={closes}"
        );
    }
}
