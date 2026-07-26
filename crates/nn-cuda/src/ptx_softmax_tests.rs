// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for PTX softmax kernel generation.
//!
//! Covers config construction, PTX structural correctness, softmax and
//! log_softmax variants, launch configuration, edge cases, and instruction
//! coverage.

use super::*;

// =====================================================================
// Config construction
// =====================================================================

#[test]
fn test_config_basic() {
    let c = PtxSoftmaxConfig::new("softmax", 128);
    assert_eq!(c.dim, 128);
    assert_eq!(c.kernel_name, "softmax");
    assert_eq!(c.sm_target, "sm_80");
    assert!(!c.log_mode);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_new_log() {
    let c = PtxSoftmaxConfig::new_log("log_sm", 64);
    assert!(c.log_mode);
    assert_eq!(c.dim, 64);
    assert_eq!(c.kernel_name, "log_sm");
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_with_log_mode_toggle() {
    let c = PtxSoftmaxConfig::new("sm", 64).with_log_mode(true);
    assert!(c.log_mode);

    let c2 = PtxSoftmaxConfig::new_log("sm", 64).with_log_mode(false);
    assert!(!c2.log_mode);
}

#[test]
fn test_config_with_sm_target() {
    let c = PtxSoftmaxConfig::new("sm", 64).with_sm_target("sm_90");
    assert_eq!(c.sm_target, "sm_90");
}

#[test]
fn test_config_dim_zero_rejected() {
    let c = PtxSoftmaxConfig::new("softmax", 0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_empty_name_rejected() {
    let c = PtxSoftmaxConfig::new("", 128);
    assert!(c.validate().is_err());
}

// ---- Config: block_size for various row sizes ----

#[test]
fn test_config_block_size_dim_1() {
    // dim=1 rounds up to 32 (one warp)
    let c = PtxSoftmaxConfig::new("s", 1);
    assert_eq!(c.block_size(), 32);
    assert_eq!(c.num_warps(), 1);
    assert!(c.is_warp_only());
    assert_eq!(c.shared_memory_bytes(), 0);
}

#[test]
fn test_config_block_size_dim_16() {
    let c = PtxSoftmaxConfig::new("s", 16);
    assert_eq!(c.block_size(), 32);
    assert_eq!(c.num_warps(), 1);
    assert!(c.is_warp_only());
}

#[test]
fn test_config_block_size_dim_32() {
    let c = PtxSoftmaxConfig::new("s", 32);
    assert_eq!(c.block_size(), 32);
    assert!(c.is_warp_only());
}

#[test]
fn test_config_block_size_dim_33() {
    // dim=33 rounds up to 64 (2 warps)
    let c = PtxSoftmaxConfig::new("s", 33);
    assert_eq!(c.block_size(), 64);
    assert_eq!(c.num_warps(), 2);
    assert!(!c.is_warp_only());
}

#[test]
fn test_config_block_size_dim_64() {
    let c = PtxSoftmaxConfig::new("s", 64);
    assert_eq!(c.block_size(), 64);
    assert_eq!(c.num_warps(), 2);
    assert!(!c.is_warp_only());
}

#[test]
fn test_config_block_size_dim_128() {
    let c = PtxSoftmaxConfig::new("s", 128);
    assert_eq!(c.block_size(), 128);
    assert_eq!(c.num_warps(), 4);
}

#[test]
fn test_config_block_size_dim_256() {
    let c = PtxSoftmaxConfig::new("s", 256);
    assert_eq!(c.block_size(), 256);
    assert_eq!(c.num_warps(), 8);
}

#[test]
fn test_config_block_size_capped_at_256() {
    // dim=512 -> would be 512 threads but capped at 256
    let c = PtxSoftmaxConfig::new("s", 512);
    assert_eq!(c.block_size(), 256);
    assert_eq!(c.num_warps(), 8);
}

#[test]
fn test_config_block_size_dim_1024() {
    let c = PtxSoftmaxConfig::new("s", 1024);
    assert_eq!(c.block_size(), 256);
    assert_eq!(c.num_warps(), 8);
}

#[test]
fn test_config_block_size_very_large_dim() {
    // dim=50257 (GPT-2 vocab size) -- still capped at 256
    let c = PtxSoftmaxConfig::new("s", 50257);
    assert_eq!(c.block_size(), 256);
    assert_eq!(c.num_warps(), 8);
    assert!(!c.is_warp_only());
}

// ---- Config: shared memory ----

#[test]
fn test_config_shared_memory_warp_only() {
    let c = PtxSoftmaxConfig::new("s", 16);
    assert_eq!(c.shared_memory_bytes(), 0);
}

#[test]
fn test_config_shared_memory_2_warps() {
    let c = PtxSoftmaxConfig::new("s", 64);
    assert_eq!(c.num_warps(), 2);
    assert_eq!(c.shared_memory_bytes(), 8); // 2 * 4 bytes
}

#[test]
fn test_config_shared_memory_4_warps() {
    let c = PtxSoftmaxConfig::new("s", 128);
    assert_eq!(c.num_warps(), 4);
    assert_eq!(c.shared_memory_bytes(), 16); // 4 * 4 bytes
}

#[test]
fn test_config_shared_memory_8_warps() {
    let c = PtxSoftmaxConfig::new("s", 256);
    assert_eq!(c.num_warps(), 8);
    assert_eq!(c.shared_memory_bytes(), 32); // 8 * 4 bytes
}

// =====================================================================
// PTX generation: emit_ptx_softmax produces valid output
// =====================================================================

#[test]
fn test_emit_ptx_softmax_non_empty() {
    let config = PtxSoftmaxConfig::new("test_kernel", 128);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(!ptx.is_empty(), "PTX output must not be empty");
    assert!(
        ptx.len() > 500,
        "PTX output must be substantial (got {} bytes)",
        ptx.len()
    );
}

#[test]
fn test_emit_ptx_softmax_default_produces_valid_output() {
    let ptx = emit_ptx_softmax_default("test_default", 64).unwrap();
    assert!(!ptx.is_empty());
    assert!(ptx.contains(".entry test_default"));
}

#[test]
fn test_emit_ptx_softmax_rejects_invalid_config() {
    let result = emit_ptx_softmax(&PtxSoftmaxConfig::new("", 128));
    assert!(result.is_err());

    let result = emit_ptx_softmax(&PtxSoftmaxConfig::new("k", 0));
    assert!(result.is_err());
}

// =====================================================================
// PTX structural correctness: version, target, entry point
// =====================================================================

#[test]
fn test_ptx_contains_version_and_target() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains(".version"),
        "must contain PTX version directive"
    );
    assert!(ptx.contains(".target sm_80"), "must contain SM target");
    assert!(
        ptx.contains(".address_size 64"),
        "must declare 64-bit addressing"
    );
}

#[test]
fn test_ptx_contains_visible_entry() {
    // In PTX, `.visible .entry` is the equivalent of `__global__` in CUDA C++.
    let config = PtxSoftmaxConfig::new("nn_softmax", 64);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(
        ptx.contains(".visible .entry nn_softmax"),
        "must declare visible entry point (PTX equivalent of __global__)"
    );
}

#[test]
fn test_ptx_contains_thread_block_indices() {
    // In PTX, `%tid.x` is threadIdx.x, `%ctaid.x` is blockIdx.x.
    let ptx = generate_softmax_ptx(false, 64);
    assert!(
        ptx.contains("%tid.x"),
        "must reference threadIdx.x (PTX: %tid.x)"
    );
    assert!(
        ptx.contains("%ctaid.x"),
        "must reference blockIdx.x (PTX: %ctaid.x)"
    );
}

#[test]
fn test_ptx_contains_shared_memory_for_multi_warp() {
    // In PTX, `.shared` is the equivalent of `__shared__` in CUDA C++.
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains(".shared .align 4 .f32 warp_scratch["),
        "multi-warp must declare shared memory (PTX equivalent of __shared__)"
    );
}

#[test]
fn test_ptx_contains_kernel_params() {
    let ptx = generate_softmax_ptx(false, 64);
    assert!(ptx.contains("param_input"), "must have input pointer param");
    assert!(
        ptx.contains("param_output"),
        "must have output pointer param"
    );
    assert!(ptx.contains("param_row_size"), "must have row_size param");
    assert!(ptx.contains("param_num_rows"), "must have num_rows param");
}

#[test]
fn test_ptx_contains_reqntid() {
    // .reqntid declares required thread count per block
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains(".reqntid 128"),
        "dim=128 should require 128 threads"
    );

    let ptx = generate_softmax_ptx(false, 32);
    assert!(
        ptx.contains(".reqntid 32"),
        "dim=32 should require 32 threads"
    );

    let ptx = generate_softmax_ptx(false, 1024);
    assert!(
        ptx.contains(".reqntid 256"),
        "dim=1024 should require 256 threads (capped)"
    );
}

// =====================================================================
// Softmax algorithm: numerical stability via max subtraction
// =====================================================================

#[test]
fn test_softmax_has_max_reduction() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("max.f32"),
        "must use max.f32 for finding row maximum"
    );
    assert!(
        ptx.contains("Phase 1: find row max"),
        "must have phase 1 max-finding comment"
    );
}

#[test]
fn test_softmax_has_max_subtraction_for_numerical_stability() {
    // After finding max, must subtract it before exp: sub.f32 %f4, %f3, %f0
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("sub.f32"),
        "must subtract max from each element for numerical stability"
    );
    assert!(
        ptx.contains("val - max"),
        "must comment the max subtraction (diff = val - max)"
    );
}

#[test]
fn test_softmax_initializes_max_to_neg_infinity() {
    let ptx = generate_softmax_ptx(false, 64);
    // Should initialize local_max to -inf for correct max reduction
    assert!(
        ptx.contains("local_max = -inf"),
        "must initialize local max to negative infinity"
    );
}

// =====================================================================
// Softmax algorithm: exp() calls
// =====================================================================

#[test]
fn test_softmax_has_exp_via_ex2() {
    // PTX uses ex2.approx.f32 (base-2 exponential) with log2(e) prescale
    // instead of a direct exp() -- this is the standard fast-exp pattern.
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("ex2.approx.f32"),
        "must use fast base-2 exponential for exp()"
    );
}

#[test]
fn test_softmax_has_log2e_prescale() {
    // The log2(e) prescale converts natural-base to base-2: exp(x) = 2^(x * log2e)
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("log2(e)"),
        "must mention log2(e) prescale in comments"
    );
}

// =====================================================================
// Softmax algorithm: sum + division (normalization)
// =====================================================================

#[test]
fn test_softmax_has_sum_accumulation() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(ptx.contains("add.f32"), "must accumulate sum via add.f32");
    assert!(ptx.contains("local_sum"), "must track local sum variable");
}

#[test]
fn test_softmax_has_normalization_division() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("div.approx.f32"),
        "must normalize via division (1.0 / sum)"
    );
    assert!(
        ptx.contains("inv_sum"),
        "must compute inverse sum for multiplication-based normalization"
    );
}

#[test]
fn test_softmax_has_warp_shuffle_reduction() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("shfl.down.sync"),
        "must use warp shuffle for parallel reduction"
    );
    assert!(
        ptx.contains("shfl.idx.sync"),
        "must use shuffle broadcast to share result across warp"
    );
}

// =====================================================================
// log_softmax variant
// =====================================================================

#[test]
fn test_log_softmax_config_creation() {
    let c = PtxSoftmaxConfig::new_log("log_sm", 64);
    assert!(c.log_mode);
    assert_eq!(c.dim, 64);
    assert!(c.validate().is_ok());
}

#[test]
fn test_log_softmax_uses_lg2_not_div() {
    let ptx = generate_softmax_ptx(true, 128);
    assert!(
        ptx.contains("lg2.approx.f32"),
        "log_softmax must use lg2 for computing log(sum)"
    );
    // log_softmax subtracts log(sum) instead of dividing by sum
    assert!(
        !ptx.contains("div.approx.f32"),
        "log_softmax should not use div.approx (uses subtraction instead)"
    );
}

#[test]
fn test_log_softmax_header_label() {
    let ptx = generate_softmax_ptx(true, 64);
    assert!(
        ptx.contains("LogSoftmax f32"),
        "log_softmax header must say LogSoftmax"
    );
}

#[test]
fn test_log_softmax_phase4_label() {
    let config = PtxSoftmaxConfig::new_log("log_sm", 64);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(
        ptx.contains("log_softmax normalize"),
        "phase 4 must be labeled as log_softmax normalize"
    );
}

#[test]
fn test_log_softmax_differs_from_softmax() {
    let sm = generate_softmax_ptx(false, 128);
    let lsm = generate_softmax_ptx(true, 128);
    assert_ne!(
        sm, lsm,
        "softmax and log_softmax must produce different PTX"
    );
}

#[test]
fn test_log_softmax_convenience_name() {
    let ptx = generate_softmax_ptx(true, 64);
    assert!(
        ptx.contains("ptx_log_softmax_f32"),
        "log_softmax convenience must use log kernel name"
    );
}

#[test]
fn test_log_softmax_still_has_max_subtraction() {
    // log_softmax still subtracts max for numerical stability
    let ptx = generate_softmax_ptx(true, 128);
    assert!(ptx.contains("max.f32"), "log_softmax must find row max");
    assert!(ptx.contains("sub.f32"), "log_softmax must subtract max");
    assert!(
        ptx.contains("ex2.approx.f32"),
        "log_softmax must compute exp for sum"
    );
}

// =====================================================================
// Launch config for various sequence lengths
// =====================================================================

#[test]
fn test_launch_config_basic() {
    let (grid, block) = ptx_softmax_launch_config(100, 128);
    assert_eq!(grid, [100, 1, 1], "grid.x = num_rows");
    assert_eq!(block, [128, 1, 1], "block.x = block_size for 4-warp config");
}

#[test]
fn test_launch_config_small_dim() {
    let (grid, block) = ptx_softmax_launch_config(10, 16);
    assert_eq!(grid, [10, 1, 1]);
    assert_eq!(block, [32, 1, 1], "rounds up to one warp");
}

#[test]
fn test_launch_config_large_dim() {
    let (grid, block) = ptx_softmax_launch_config(64, 2048);
    assert_eq!(grid, [64, 1, 1]);
    assert_eq!(block, [256, 1, 1], "capped at 256 threads");
}

#[test]
fn test_launch_config_single_row() {
    let (grid, block) = ptx_softmax_launch_config(1, 128);
    assert_eq!(grid, [1, 1, 1], "single row = single block");
    assert_eq!(block, [128, 1, 1]);
}

#[test]
fn test_launch_config_many_rows() {
    let (grid, block) = ptx_softmax_launch_config(100_000, 64);
    assert_eq!(grid, [100_000, 1, 1], "one block per row");
    assert_eq!(block, [64, 1, 1]);
}

#[test]
fn test_launch_config_dim_1() {
    let (grid, block) = ptx_softmax_launch_config(50, 1);
    assert_eq!(grid, [50, 1, 1]);
    assert_eq!(block, [32, 1, 1], "dim=1 rounds up to one warp");
}

#[test]
fn test_launch_config_vocab_size() {
    // Common vocab sizes: 32000 (LLaMA), 50257 (GPT-2), 151936 (Qwen)
    let (grid, block) = ptx_softmax_launch_config(8, 50257);
    assert_eq!(grid, [8, 1, 1]);
    assert_eq!(block, [256, 1, 1], "large vocab capped at 256");
}

// =====================================================================
// Different block sizes (warp-only vs multi-warp)
// =====================================================================

#[test]
fn test_small_dim_warp_only_no_shared_memory() {
    let ptx = generate_softmax_ptx(false, 16);
    assert!(
        !ptx.contains("warp_scratch"),
        "dim<=32 should not declare shared memory scratch"
    );
    assert!(
        !ptx.contains(".shared"),
        "dim<=32 should not use shared memory"
    );
    assert!(
        ptx.contains("shfl.down.sync"),
        "warp-only still uses shuffles"
    );
    assert!(
        !ptx.contains("bar.sync"),
        "warp-only should not need barriers"
    );
}

#[test]
fn test_dim_32_warp_only() {
    let ptx = generate_softmax_ptx(false, 32);
    assert!(!ptx.contains("warp_scratch"), "dim=32 should be warp-only");
    assert!(!ptx.contains("bar.sync"), "dim=32 should not need barriers");
}

#[test]
fn test_multi_warp_uses_shared_memory_and_barriers() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains(".shared .align 4 .f32 warp_scratch["),
        "dim>32 must declare shared memory for cross-warp reduction"
    );
    assert!(
        ptx.contains("shfl.down.sync"),
        "large dim still uses warp shuffles within each warp"
    );
    assert!(
        ptx.contains("bar.sync"),
        "cross-warp reduction requires barrier"
    );
}

#[test]
fn test_shared_memory_size_matches_warp_count() {
    // dim=64 -> 2 warps -> warp_scratch[2]
    let ptx = generate_softmax_ptx(false, 64);
    assert!(
        ptx.contains("warp_scratch[2]"),
        "64-dim should have 2-warp scratch"
    );

    // dim=128 -> 4 warps -> warp_scratch[4]
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("warp_scratch[4]"),
        "128-dim should have 4-warp scratch"
    );

    // dim=256 -> 8 warps -> warp_scratch[8]
    let ptx = generate_softmax_ptx(false, 256);
    assert!(
        ptx.contains("warp_scratch[8]"),
        "256-dim should have 8-warp scratch"
    );
}

#[test]
fn test_different_dims_produce_different_ptx() {
    let ptx_32 = generate_softmax_ptx(false, 32);
    let ptx_128 = generate_softmax_ptx(false, 128);
    let ptx_1024 = generate_softmax_ptx(false, 1024);

    assert_ne!(ptx_32, ptx_128, "dim=32 and dim=128 should differ");
    assert_ne!(ptx_128, ptx_1024, "dim=128 and dim=1024 should differ");
    assert_ne!(ptx_32, ptx_1024, "dim=32 and dim=1024 should differ");
}

// =====================================================================
// Edge case: row_size = 1 (trivial softmax = 1.0)
// =====================================================================

#[test]
fn test_dim_1_generates_valid_ptx() {
    let config = PtxSoftmaxConfig::new("softmax_1", 1);
    assert_eq!(config.block_size(), 32);
    assert!(config.is_warp_only());
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry softmax_1"), "entry point present");
    assert!(ptx.contains("ret;"), "kernel has return");
    assert!(!ptx.is_empty());
}

#[test]
fn test_dim_1_warp_only_no_shared() {
    let ptx = generate_softmax_ptx(false, 1);
    assert!(!ptx.contains(".shared"), "dim=1 must be warp-only");
    assert!(!ptx.contains("bar.sync"), "dim=1 no barriers needed");
}

#[test]
fn test_dim_1_log_softmax() {
    // log_softmax of a single element should also generate valid PTX
    let config = PtxSoftmaxConfig::new_log("log_sm_1", 1);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry log_sm_1"));
    assert!(
        ptx.contains("lg2.approx.f32"),
        "log_softmax still needs lg2"
    );
}

// =====================================================================
// Edge case: very large row_size
// =====================================================================

#[test]
fn test_very_large_dim_generates_valid_ptx() {
    // dim=50257 (GPT-2 vocab), threads loop over elements with stride
    let config = PtxSoftmaxConfig::new("softmax_gpt2_vocab", 50257);
    assert_eq!(config.block_size(), 256);
    assert_eq!(config.num_warps(), 8);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry softmax_gpt2_vocab"));
    assert!(ptx.contains("warp_scratch[8]"));
    assert!(ptx.contains("ret;"));
}

#[test]
fn test_very_large_dim_151936() {
    // dim=151936 (Qwen2.5 vocab size)
    let config = PtxSoftmaxConfig::new("softmax_qwen_vocab", 151_936);
    assert_eq!(config.block_size(), 256);
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry softmax_qwen_vocab"));
    assert!(
        ptx.contains("dim=151936"),
        "comment should reflect actual dim"
    );
}

#[test]
fn test_large_dim_stride_loop() {
    // For dim > block_size, threads loop with stride = block_size.
    // The PTX loops: `add.u32 %r7, %r7, {block_size}` then branches back.
    let ptx = generate_softmax_ptx(false, 1024);
    // block_size is 256, so stride is 256
    assert!(
        ptx.contains("256"),
        "stride should appear as block_size in the loop increment"
    );
}

// =====================================================================
// Custom SM target
// =====================================================================

#[test]
fn test_custom_sm_target_sm70() {
    let config = PtxSoftmaxConfig::new("sm70_kernel", 64).with_sm_target("sm_70");
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_custom_sm_target_sm90() {
    let config = PtxSoftmaxConfig::new("sm90_kernel", 64).with_sm_target("sm_90");
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".target sm_90"));
}

// =====================================================================
// Convenience wrappers
// =====================================================================

#[test]
fn test_generate_softmax_ptx_matches_emit() {
    let direct = generate_softmax_ptx(false, 128);
    let config = PtxSoftmaxConfig::new("ptx_softmax_f32", 128);
    let via_config = emit_ptx_softmax(&config).unwrap();
    assert_eq!(
        direct, via_config,
        "convenience wrapper must match direct emission"
    );
}

#[test]
fn test_generate_softmax_ptx_log_matches_emit() {
    let direct = generate_softmax_ptx(true, 128);
    let config = PtxSoftmaxConfig::new("ptx_log_softmax_f32", 128).with_log_mode(true);
    let via_config = emit_ptx_softmax(&config).unwrap();
    assert_eq!(
        direct, via_config,
        "log convenience wrapper must match direct emission"
    );
}

#[test]
fn test_emit_ptx_softmax_default_uses_sm80() {
    let ptx = emit_ptx_softmax_default("default_kernel", 64).unwrap();
    assert!(ptx.contains(".target sm_80"), "default must use sm_80");
    assert!(ptx.contains(".entry default_kernel"));
}

// =====================================================================
// PTX is pure PTX assembly, not CUDA C++
// =====================================================================

#[test]
fn test_ptx_is_not_cuda_cpp() {
    let ptx = generate_softmax_ptx(false, 128);
    // PTX assembly must not contain CUDA C++ syntax (keywords, preprocessor).
    // Note: comments may reference threadIdx/blockIdx for documentation purposes,
    // so we check for the actual CUDA C++ keywords with decorators.
    assert!(
        !ptx.contains("__global__"),
        "must not contain CUDA C++ __global__"
    );
    assert!(!ptx.contains("#include"), "must not contain C++ includes");
    assert!(
        !ptx.contains("__shared__"),
        "must not contain CUDA C++ __shared__"
    );
    assert!(
        !ptx.contains("__syncthreads"),
        "must not contain CUDA C++ __syncthreads"
    );
    // PTX uses %tid.x and %ctaid.x instead of threadIdx.x and blockIdx.x.
    // Comments may reference CUDA names for clarity, but the actual instructions
    // must use the PTX register names.
    assert!(
        ptx.contains("%tid.x"),
        "PTX must use %tid.x (not threadIdx.x) in instructions"
    );
    assert!(
        ptx.contains("%ctaid.x"),
        "PTX must use %ctaid.x (not blockIdx.x) in instructions"
    );
}

// =====================================================================
// PTX instruction completeness
// =====================================================================

#[test]
fn test_ptx_instruction_set_coverage() {
    let ptx = generate_softmax_ptx(false, 128);
    let expected_instructions = [
        "ld.param",       // parameter loading
        "mov.u32",        // register moves
        "mul.wide.u32",   // widening multiply for byte offsets
        "add.u64",        // 64-bit pointer arithmetic
        "ld.global.f32",  // global memory load
        "st.global.f32",  // global memory store
        "max.f32",        // float max for phase 1
        "sub.f32",        // subtract max in phase 2
        "ex2.approx.f32", // fast exp via base-2
        "add.f32",        // sum accumulation
        "div.approx.f32", // normalization division
        "mul.f32",        // scalar multiplication
        "shfl.down.sync", // warp shuffle reduction
        "shfl.idx.sync",  // warp shuffle broadcast
        "setp.ge.u32",    // loop termination
        "bar.sync",       // thread barrier (cross-warp, dim=128)
        "bra",            // branch
        "ret",            // return
    ];
    for instr in &expected_instructions {
        assert!(ptx.contains(instr), "PTX must contain instruction: {instr}");
    }
}

#[test]
fn test_ptx_register_declarations() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(ptx.contains(".reg .u32"), "must declare u32 registers");
    assert!(ptx.contains(".reg .f32"), "must declare f32 registers");
    assert!(ptx.contains(".reg .u64"), "must declare u64 registers");
    assert!(
        ptx.contains(".reg .pred"),
        "must declare predicate registers"
    );
}

#[test]
fn test_ptx_has_kernel_exit_label() {
    let ptx = generate_softmax_ptx(false, 64);
    assert!(ptx.contains("KERNEL_EXIT:"), "must have KERNEL_EXIT label");
    assert!(ptx.contains("ret;"), "must return at kernel exit");
}

#[test]
fn test_ptx_has_all_four_phases() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(ptx.contains("Phase 1"), "must have phase 1 (find max)");
    assert!(ptx.contains("Phase 2"), "must have phase 2 (exp)");
    assert!(ptx.contains("Phase 3"), "must have phase 3 (sum)");
    assert!(ptx.contains("Phase 4"), "must have phase 4 (normalize)");
}

// =====================================================================
// Non-power-of-2 dimensions
// =====================================================================

#[test]
fn test_non_power_of_2_dim_50() {
    let config = PtxSoftmaxConfig::new("softmax_50", 50);
    assert_eq!(config.block_size(), 64); // rounds up to 2 warps
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry softmax_50"));
    assert!(ptx.contains("shfl.down.sync"));
    assert!(ptx.contains("warp_scratch[2]"), "50-dim rounds to 2 warps");
}

#[test]
fn test_non_power_of_2_dim_100() {
    let config = PtxSoftmaxConfig::new("softmax_100", 100);
    assert_eq!(config.block_size(), 128); // rounds up to 4 warps
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry softmax_100"));
    assert!(ptx.contains("warp_scratch[4]"));
}

#[test]
fn test_non_power_of_2_dim_768() {
    // Common transformer hidden dim
    let config = PtxSoftmaxConfig::new("softmax_768", 768);
    assert_eq!(config.block_size(), 256); // capped
    let ptx = emit_ptx_softmax(&config).unwrap();
    assert!(ptx.contains(".entry softmax_768"));
    assert!(ptx.contains("dim=768"), "comment should reflect actual dim");
}

// =====================================================================
// Softmax header comment reflects mode and parameters
// =====================================================================

#[test]
fn test_softmax_header_comment() {
    let ptx = generate_softmax_ptx(false, 64);
    assert!(ptx.contains("Softmax f32"), "header must label as Softmax");
    assert!(ptx.contains("dim=64"), "header must include dim");
    assert!(
        ptx.contains("block_size=64"),
        "header must include block_size"
    );
    assert!(ptx.contains("warps=2"), "header must include warp count");
}

#[test]
fn test_softmax_warp_only_reduction_comment() {
    let ptx = generate_softmax_ptx(false, 32);
    assert!(
        ptx.contains("warp-only (no shared memory)"),
        "warp-only must be labeled in comments"
    );
}

#[test]
fn test_softmax_multi_warp_reduction_comment() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("warp shuffle + shared memory cross-warp"),
        "multi-warp must be labeled in comments"
    );
}

// =====================================================================
// Cross-warp reduction structure
// =====================================================================

#[test]
fn test_cross_warp_max_reduction_labels() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("CROSS_MAX_LOAD:"),
        "must have cross-warp max label"
    );
    assert!(
        ptx.contains("CROSS_MAX_DONE:"),
        "must have cross-warp max done label"
    );
    assert!(
        ptx.contains("BCAST_MAX_LOAD:"),
        "must have max broadcast label"
    );
}

#[test]
fn test_cross_warp_sum_reduction_labels() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(
        ptx.contains("CROSS_SUM_LOAD:"),
        "must have cross-warp sum label"
    );
    assert!(
        ptx.contains("CROSS_SUM_DONE:"),
        "must have cross-warp sum done label"
    );
    assert!(
        ptx.contains("BCAST_SUM_LOAD:"),
        "must have sum broadcast label"
    );
}

#[test]
fn test_warp_only_no_cross_warp_labels() {
    let ptx = generate_softmax_ptx(false, 32);
    assert!(
        !ptx.contains("CROSS_MAX_LOAD"),
        "warp-only should not have cross-warp labels"
    );
    assert!(
        !ptx.contains("CROSS_SUM_LOAD"),
        "warp-only should not have cross-warp labels"
    );
    assert!(
        !ptx.contains("BCAST_MAX_LOAD"),
        "warp-only should not have broadcast labels"
    );
}

// =====================================================================
// Phase loop structure
// =====================================================================

#[test]
fn test_phase_loop_labels() {
    let ptx = generate_softmax_ptx(false, 128);
    assert!(ptx.contains("PHASE1_LOOP:"), "must have phase 1 loop label");
    assert!(
        ptx.contains("PHASE1_REDUCE:"),
        "must have phase 1 reduce label"
    );
    assert!(ptx.contains("PHASE2_LOOP:"), "must have phase 2 loop label");
    assert!(
        ptx.contains("PHASE3_REDUCE:"),
        "must have phase 3 reduce label"
    );
    assert!(ptx.contains("PHASE4_LOOP:"), "must have phase 4 loop label");
}

// =====================================================================
// Reference computation: softmax known values
// =====================================================================

#[test]
fn test_softmax_reference_uniform_input() {
    // All equal inputs -> uniform output: 1/N each
    let input = vec![1.0f32; 4];
    let output = softmax_reference(&input);
    assert_eq!(output.len(), 4);
    for &v in &output {
        assert!(
            (v - 0.25).abs() < 1e-6,
            "uniform input should give uniform output 0.25, got {v}"
        );
    }
}

#[test]
fn test_softmax_reference_sums_to_one() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = softmax_reference(&input);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax must sum to 1.0, got {sum}"
    );
}

#[test]
fn test_softmax_reference_all_positive() {
    let input = vec![-10.0, -5.0, 0.0, 5.0, 10.0];
    let output = softmax_reference(&input);
    for &v in &output {
        assert!(v > 0.0, "all softmax outputs must be positive, got {v}");
    }
}

#[test]
fn test_softmax_reference_monotonic() {
    // Larger inputs should produce larger softmax outputs
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = softmax_reference(&input);
    for i in 0..output.len() - 1 {
        assert!(
            output[i] < output[i + 1],
            "softmax must be monotonically increasing with input: {} >= {}",
            output[i],
            output[i + 1]
        );
    }
}

#[test]
fn test_softmax_reference_zeros() {
    // input = [0, 0, 0] -> [1/3, 1/3, 1/3]
    let input = vec![0.0, 0.0, 0.0];
    let output = softmax_reference(&input);
    for &v in &output {
        assert!(
            (v - 1.0 / 3.0).abs() < 1e-6,
            "softmax of zeros should be uniform, got {v}"
        );
    }
}

#[test]
fn test_softmax_reference_dominant_element() {
    // One very large element dominates
    let input = vec![0.0, 0.0, 100.0, 0.0];
    let output = softmax_reference(&input);
    assert!(
        output[2] > 0.999,
        "dominant element should get almost all probability, got {}",
        output[2]
    );
    for (i, &v) in output.iter().enumerate() {
        if i != 2 {
            assert!(v < 0.001, "non-dominant should be near zero, got {v}");
        }
    }
}

#[test]
fn test_softmax_reference_negative_inputs() {
    let input = vec![-1.0, -2.0, -3.0, -4.0];
    let output = softmax_reference(&input);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "softmax of negatives must sum to 1.0, got {sum}"
    );
    // Least negative -> largest probability
    assert!(output[0] > output[1]);
    assert!(output[1] > output[2]);
    assert!(output[2] > output[3]);
}

#[test]
fn test_softmax_reference_translation_invariance() {
    // softmax(x) == softmax(x + c) for any constant c
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let shifted = vec![101.0, 102.0, 103.0, 104.0];
    let out_orig = softmax_reference(&input);
    let out_shifted = softmax_reference(&shifted);
    for (a, b) in out_orig.iter().zip(out_shifted.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "softmax must be translation invariant: {a} vs {b}"
        );
    }
}

#[test]
fn test_softmax_reference_empty_input() {
    let output = softmax_reference(&[]);
    assert!(output.is_empty());
}

#[test]
fn test_softmax_reference_single_element() {
    let output = softmax_reference(&[42.0]);
    assert_eq!(output.len(), 1);
    assert!(
        (output[0] - 1.0).abs() < 1e-6,
        "single-element softmax must be 1.0"
    );
}

#[test]
fn test_softmax_reference_large_values_numerical_stability() {
    // Large values that would overflow naive exp()
    let input = vec![1000.0, 1001.0, 1002.0];
    let output = softmax_reference(&input);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax with large values must still sum to 1.0, got {sum}"
    );
    for &v in &output {
        assert!(v.is_finite(), "softmax output must be finite, got {v}");
        assert!(v > 0.0, "softmax output must be positive, got {v}");
    }
}

#[test]
fn test_softmax_reference_large_negative_values() {
    let input = vec![-1000.0, -1001.0, -1002.0];
    let output = softmax_reference(&input);
    let sum: f32 = output.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax with large negatives must still sum to 1.0, got {sum}"
    );
}

#[test]
fn test_softmax_reference_mixed_sign() {
    let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let output = softmax_reference(&input);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // Monotonically increasing
    for i in 0..output.len() - 1 {
        assert!(output[i] < output[i + 1]);
    }
}

// =====================================================================
// Reference computation: log_softmax known values
// =====================================================================

#[test]
fn test_log_softmax_reference_uniform() {
    // Equal inputs -> log(1/N) = -log(N)
    let input = vec![0.0f32; 4];
    let output = log_softmax_reference(&input);
    let expected = -(4.0f32).ln();
    for &v in &output {
        assert!(
            (v - expected).abs() < 1e-6,
            "log_softmax of uniform should be -ln(4), got {v}, expected {expected}"
        );
    }
}

#[test]
fn test_log_softmax_reference_exp_sums_to_one() {
    // exp(log_softmax(x)) should sum to 1
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = log_softmax_reference(&input);
    let exp_sum: f32 = output.iter().map(|&v| v.exp()).sum();
    assert!(
        (exp_sum - 1.0).abs() < 1e-5,
        "exp(log_softmax) must sum to 1.0, got {exp_sum}"
    );
}

#[test]
fn test_log_softmax_reference_all_non_positive() {
    // log_softmax values are always <= 0
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = log_softmax_reference(&input);
    for &v in &output {
        assert!(v <= 0.0, "log_softmax values must be <= 0, got {v}");
    }
}

#[test]
fn test_log_softmax_reference_matches_log_of_softmax() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let sm = softmax_reference(&input);
    let log_sm = log_softmax_reference(&input);
    for (ls, s) in log_sm.iter().zip(sm.iter()) {
        let expected = s.ln();
        assert!(
            (ls - expected).abs() < 1e-5,
            "log_softmax should equal log(softmax): got {ls}, expected {expected}"
        );
    }
}

#[test]
fn test_log_softmax_reference_translation_invariance() {
    let input = vec![1.0, 2.0, 3.0];
    let shifted = vec![1001.0, 1002.0, 1003.0];
    let out_orig = log_softmax_reference(&input);
    let out_shifted = log_softmax_reference(&shifted);
    for (a, b) in out_orig.iter().zip(out_shifted.iter()) {
        assert!(
            (a - b).abs() < 1e-4,
            "log_softmax must be translation invariant: {a} vs {b}"
        );
    }
}

#[test]
fn test_log_softmax_reference_empty_input() {
    let output = log_softmax_reference(&[]);
    assert!(output.is_empty());
}

#[test]
fn test_log_softmax_reference_single_element() {
    let output = log_softmax_reference(&[42.0]);
    assert_eq!(output.len(), 1);
    assert!(
        output[0].abs() < 1e-6,
        "single-element log_softmax must be 0.0, got {}",
        output[0]
    );
}

#[test]
fn test_log_softmax_reference_dominant_element() {
    let input = vec![0.0, 0.0, 100.0, 0.0];
    let output = log_softmax_reference(&input);
    // The dominant element's log_softmax should be near 0
    assert!(
        output[2].abs() < 1e-3,
        "dominant element log_softmax should be near 0, got {}",
        output[2]
    );
    // Non-dominant should be very negative
    for (i, &v) in output.iter().enumerate() {
        if i != 2 {
            assert!(
                v < -90.0,
                "non-dominant log_softmax should be very negative, got {v}"
            );
        }
    }
}

// =========================================================================
// SOFTMAX_BLOCK_SIZE constant
// =========================================================================

#[test]
fn test_softmax_block_size_constant_value() {
    assert_eq!(SOFTMAX_BLOCK_SIZE, 256);
}

#[test]
fn test_softmax_block_size_matches_max_block_size() {
    // The public constant should match the internal MAX_BLOCK_SIZE
    let config = PtxSoftmaxConfig::new("test", 512);
    assert_eq!(config.block_size(), SOFTMAX_BLOCK_SIZE as usize);
}

// =========================================================================
// generate_log_softmax_ptx convenience function
// =========================================================================

#[test]
fn test_generate_log_softmax_ptx_produces_valid_ptx() {
    let ptx = generate_log_softmax_ptx(64);
    assert!(
        ptx.contains(".version"),
        "PTX must contain .version directive"
    );
    assert!(
        ptx.contains(".target"),
        "PTX must contain .target directive"
    );
    assert!(
        ptx.contains(".visible .entry"),
        "PTX must contain .visible .entry for kernel"
    );
}

#[test]
fn test_generate_log_softmax_ptx_contains_log_kernel_name() {
    let ptx = generate_log_softmax_ptx(128);
    assert!(
        ptx.contains("ptx_log_softmax_f32"),
        "log_softmax PTX must use the log_softmax kernel name"
    );
}

#[test]
fn test_generate_log_softmax_ptx_contains_log_instruction() {
    let ptx = generate_log_softmax_ptx(32);
    // log_softmax uses lg2.approx.f32 for the log(sum) computation
    assert!(
        ptx.contains("lg2.approx.f32"),
        "log_softmax PTX must contain lg2 instruction for log computation"
    );
}

#[test]
fn test_generate_log_softmax_ptx_differs_from_softmax() {
    let log_ptx = generate_log_softmax_ptx(64);
    let softmax_ptx = generate_softmax_ptx(false, 64);
    assert_ne!(
        log_ptx, softmax_ptx,
        "log_softmax and softmax PTX should differ"
    );
}

#[test]
fn test_generate_log_softmax_ptx_various_dims() {
    for dim in [1, 16, 32, 64, 128, 256, 512] {
        let ptx = generate_log_softmax_ptx(dim);
        assert!(
            !ptx.is_empty(),
            "generate_log_softmax_ptx({dim}) should produce non-empty PTX"
        );
        assert!(
            ptx.contains(".visible .entry"),
            "generate_log_softmax_ptx({dim}) must contain kernel entry"
        );
    }
}
