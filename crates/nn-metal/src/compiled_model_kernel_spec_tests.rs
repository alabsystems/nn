// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for KernelSpec builders.
//!
//! Extracted from `compiled_model_kernel_spec.rs` per 500-line limit.
//! Part of #3503 (KernelSpec unification).

use super::*;
use nn_dsl::ir::ScalarType;

#[test]
fn test_spec_instance_norm_f32_basic() {
    let spec = spec_instance_norm(ScalarType::F32, 1e-5, &[1, 64, 256]).unwrap();
    assert_eq!(spec.kernel_name, "fused_instance_norm_float");
    assert_eq!(spec.grid, [64, 1, 1]); // 1 * 64 rows
    assert_eq!(spec.threadgroup, [NORM_TG_SIZE, 1, 1]);
    assert_eq!(spec.output_bytes, 1 * 64 * 256 * 4);
    assert_eq!(spec.bindings.len(), 4);
    assert_eq!(spec.param_count, 1);
}

#[test]
fn test_spec_instance_norm_rank_too_low() {
    let err = spec_instance_norm(ScalarType::F32, 1e-5, &[1, 64]).unwrap_err();
    assert!(err.contains("rank >= 3"));
}

#[test]
fn test_spec_layer_norm_f32_basic() {
    let spec = spec_layer_norm(ScalarType::F32, 1e-5, &[1, 32, 768], 768).unwrap();
    assert_eq!(spec.kernel_name, "fused_layer_norm_float");
    assert_eq!(spec.grid, [32, 1, 1]); // 1 * 32 rows
    assert_eq!(spec.output_bytes, 1 * 32 * 768 * 4);
    assert_eq!(spec.bindings.len(), 6);
    assert_eq!(spec.param_count, 3);
}

#[test]
fn test_spec_layer_norm_rank_too_low() {
    let err = spec_layer_norm(ScalarType::F32, 1e-5, &[768], 768).unwrap_err();
    assert!(err.contains("rank >= 2"));
}

#[test]
fn test_spec_add_layer_norm_f32_basic() {
    let spec =
        spec_add_layer_norm(ScalarType::F32, 1e-5, &[1, 32, 768], 768).unwrap();
    assert_eq!(spec.kernel_name, "fused_add_layer_norm_float");
    assert_eq!(spec.grid, [32, 1, 1]);
    assert_eq!(spec.output_bytes, 1 * 32 * 768 * 4);
    assert_eq!(spec.bindings.len(), 7); // 2 edges + 2 weights + output + 2 constants
    assert_eq!(spec.param_count, 4);
}

#[test]
fn test_spec_channels_first_layer_norm_basic() {
    let spec = spec_channels_first_layer_norm(
        ScalarType::F32,
        1e-5,
        &[1, 512, 100],
        512,
        None,
    )
    .unwrap();
    assert_eq!(spec.kernel_name, "fused_channels_first_layer_norm_float");
    assert_eq!(spec.grid, [100, 1, 1]); // B*T = 1*100
    assert_eq!(spec.output_bytes, 1 * 512 * 100 * 4);
    assert_eq!(spec.bindings.len(), 7); // edge + 2 weights + output + 3 constants
}

#[test]
fn test_spec_channels_first_layer_norm_with_leaky_relu() {
    let spec = spec_channels_first_layer_norm(
        ScalarType::F32,
        1e-5,
        &[1, 512, 100],
        512,
        Some(0.2),
    )
    .unwrap();
    assert_eq!(
        spec.kernel_name,
        "fused_channels_first_ln_leaky_relu_float"
    );
    assert_eq!(spec.bindings.len(), 8); // +1 for slope constant
}

#[test]
fn test_spec_channels_first_layer_norm_wrong_rank() {
    let err = spec_channels_first_layer_norm(
        ScalarType::F32,
        1e-5,
        &[1, 512],
        512,
        None,
    )
    .unwrap_err();
    assert!(err.contains("rank == 3"));
}

#[test]
fn test_spec_adain_snake_f32_basic() {
    let spec =
        spec_adain_snake(ScalarType::F32, 1e-5, &[1, 256, 100], 256, true).unwrap();
    assert_eq!(spec.kernel_name, "fused_adain_snake_float");
    assert_eq!(spec.grid, [256, 1, 1]); // 1 * 256 rows
    assert_eq!(spec.output_bytes, 1 * 256 * 100 * 4);
    assert_eq!(spec.bindings.len(), 8); // 3 edges + 1 weight + output + 3 constants
    assert_eq!(spec.param_count, 4);
}

#[test]
fn test_spec_adain_snake_rank_too_low() {
    let err =
        spec_adain_snake(ScalarType::F32, 1e-5, &[1, 64], 64, true).unwrap_err();
    assert!(err.contains("rank >= 3"));
}

#[test]
fn test_spec_adain_leaky_relu_f32_basic() {
    let spec =
        spec_adain_leaky_relu(ScalarType::F32, 1e-5, 0.2, &[1, 128, 50]).unwrap();
    assert_eq!(spec.kernel_name, "fused_adain_leaky_relu_float");
    assert_eq!(spec.grid, [128, 1, 1]); // 1 * 128 rows
    assert_eq!(spec.output_bytes, 1 * 128 * 50 * 4);
    assert_eq!(spec.bindings.len(), 7); // 3 edges + output + 3 constants
    assert_eq!(spec.param_count, 3);
}

#[test]
fn test_spec_ada_layer_norm_f32_basic() {
    let spec =
        spec_ada_layer_norm(ScalarType::F32, 1e-5, &[1, 32, 256], 256).unwrap();
    assert_eq!(spec.kernel_name, "fused_ada_layer_norm_float");
    assert_eq!(spec.grid, [32, 1, 1]); // 1 * 32 rows
    assert_eq!(spec.output_bytes, 1 * 32 * 256 * 4);
    assert_eq!(spec.bindings.len(), 9); // 3 edges + 2 weights + output + 3 constants
    assert_eq!(spec.param_count, 5);
}

#[test]
fn test_spec_ada_layer_norm_rank_too_low() {
    let err =
        spec_ada_layer_norm(ScalarType::F32, 1e-5, &[1, 256], 256).unwrap_err();
    assert!(err.contains("rank >= 3"));
}

#[test]
fn test_kernel_binding_constant_u32() {
    let binding = KernelBinding::constant_u32(42);
    match binding {
        KernelBinding::Constant(bytes) => {
            assert_eq!(bytes.len(), 4);
            let val = u32::from_ne_bytes(bytes.try_into().unwrap());
            assert_eq!(val, 42);
        }
        _ => panic!("expected Constant"),
    }
}

#[test]
fn test_kernel_binding_constant_f32() {
    let binding = KernelBinding::constant_f32(1e-5);
    match binding {
        KernelBinding::Constant(bytes) => {
            assert_eq!(bytes.len(), 4);
            let val = f32::from_ne_bytes(bytes.try_into().unwrap());
            assert!((val - 1e-5).abs() < f32::EPSILON);
        }
        _ => panic!("expected Constant"),
    }
}

#[test]
fn test_spec_dispatch_mode_into_native() {
    // Verify conversion produces the expected variant (Debug format check).
    let threads = SpecDispatchMode::Threads.into_native();
    assert!(format!("{threads:?}").contains("Threads"));
    let threadgroups = SpecDispatchMode::Threadgroups.into_native();
    assert!(format!("{threadgroups:?}").contains("Threadgroups"));
}

// =========================================================================
// GroupNorm tests (#3503 D3)
// =========================================================================

#[test]
fn test_spec_group_norm_f32_basic() {
    // Input [1, 32, 64], 8 groups → channels_per_group=4, spatial=64
    let spec = spec_group_norm(ScalarType::F32, 1e-5, &[1, 32, 64], 8).unwrap();
    assert_eq!(spec.kernel_name, "fused_group_norm_float");
    // flat_rows = B*G = 1*8 = 8
    assert_eq!(spec.grid, [8, 1, 1]);
    assert_eq!(spec.threadgroup, [NORM_TG_SIZE, 1, 1]);
    // total = 1*32*64 = 2048 elems × 4 bytes = 8192
    assert_eq!(spec.output_bytes, 1 * 32 * 64 * 4);
    // bindings: edge + 2 weights + output + 5 constants = 9
    assert_eq!(spec.bindings.len(), 9);
    assert_eq!(spec.param_count, 3);
}

#[test]
fn test_spec_group_norm_rank_too_low() {
    let err = spec_group_norm(ScalarType::F32, 1e-5, &[32], 8).unwrap_err();
    assert!(err.contains("rank >= 2"));
}

#[test]
fn test_spec_group_norm_channels_not_divisible() {
    let err = spec_group_norm(ScalarType::F32, 1e-5, &[1, 30, 64], 8).unwrap_err();
    assert!(err.contains("not divisible"));
}

// =========================================================================
// RmsNorm tests (#3503 D3)
// =========================================================================

#[test]
fn test_spec_rms_norm_f32_basic() {
    // Input [1, 32, 4096], hidden_dim=4096
    let spec = spec_rms_norm(ScalarType::F32, 1e-6, &[1, 32, 4096], 4096).unwrap();
    assert_eq!(spec.kernel_name, "fused_rms_norm_float");
    // flat_rows = 1*32 = 32
    assert_eq!(spec.grid, [32, 1, 1]);
    assert_eq!(spec.threadgroup, [NORM_TG_SIZE, 1, 1]);
    assert_eq!(spec.output_bytes, 1 * 32 * 4096 * 4);
    // bindings: edge + weight + output + 2 constants = 5
    assert_eq!(spec.bindings.len(), 5);
    assert_eq!(spec.param_count, 2);
}

#[test]
fn test_spec_rms_norm_rank1() {
    // Rank 1: [4096] → flat_rows = 1
    let spec = spec_rms_norm(ScalarType::F32, 1e-6, &[4096], 4096).unwrap();
    assert_eq!(spec.grid, [1, 1, 1]);
    assert_eq!(spec.output_bytes, 4096 * 4);
}

#[test]
fn test_spec_rms_norm_empty() {
    let err = spec_rms_norm(ScalarType::F32, 1e-6, &[], 0).unwrap_err();
    assert!(err.contains("rank >= 1"));
}

// =========================================================================
// Snake tests (#3503 D3)
// =========================================================================

#[test]
fn test_spec_snake_f32_basic() {
    // Input [1, 256, 100], channels=256
    let spec = spec_snake(ScalarType::F32, &[1, 256, 100], 256).unwrap();
    assert_eq!(spec.kernel_name, "fused_snake_float");
    // total = 25600, threadgroups = ceil(25600/256) = 100
    assert_eq!(spec.grid, [100, 1, 1]);
    assert_eq!(spec.threadgroup, [NORM_TG_SIZE, 1, 1]);
    assert_eq!(spec.output_bytes, 1 * 256 * 100 * 4);
    // bindings: edge + weight + output + 3 constants = 6
    assert_eq!(spec.bindings.len(), 6);
    assert_eq!(spec.param_count, 2);
}

#[test]
fn test_spec_snake_empty() {
    let err = spec_snake(ScalarType::F32, &[0], 0).unwrap_err();
    assert!(err.contains("empty"));
}

// =========================================================================
// FlashAttention tests (#3503 D3)
// =========================================================================

#[test]
fn test_spec_flash_attention_f32_basic() {
    use nn_dsl::AttentionLayout;
    // Q [1, 8, 32, 64], K [1, 8, 32, 64]
    let spec = spec_flash_attention(
        ScalarType::F32,
        0.125,
        false,
        &[1, 8, 32, 64],
        &[1, 8, 32, 64],
        AttentionLayout::HeadsFirst,
    )
    .unwrap();
    assert_eq!(spec.kernel_name, "flash_attn_f32");
    // grid_x = ceil(32/32) = 1, grid_y = B*H = 8
    assert_eq!(spec.grid, [1, 8, 1]);
    assert_eq!(spec.threadgroup, [32, 1, 1]);
    // total = 1*8*32*64 = 16384 × 4 bytes
    assert_eq!(spec.output_bytes, 16384 * 4);
    // bindings: 3 edges + output + 7 constants = 11
    assert_eq!(spec.bindings.len(), 11);
    assert_eq!(spec.param_count, 3);
}

#[test]
fn test_spec_flash_attention_gqa() {
    use nn_dsl::AttentionLayout;
    // Q [1, 8, 32, 64], K [1, 2, 32, 64] → group_size=4
    let spec = spec_flash_attention(
        ScalarType::F32,
        0.125,
        false,
        &[1, 8, 32, 64],
        &[1, 2, 32, 64],
        AttentionLayout::HeadsFirst,
    )
    .unwrap();
    // grid_y = B*H_q = 8
    assert_eq!(spec.grid[1], 8);
}

#[test]
fn test_spec_flash_attention_h_mismatch() {
    use nn_dsl::AttentionLayout;
    // H_q=7, H_kv=3 → not divisible
    let err = spec_flash_attention(
        ScalarType::F32,
        0.125,
        false,
        &[1, 7, 32, 64],
        &[1, 3, 32, 64],
        AttentionLayout::HeadsFirst,
    )
    .unwrap_err();
    assert!(err.contains("multiple"));
}

#[test]
fn test_spec_flash_attention_wrong_rank() {
    use nn_dsl::AttentionLayout;
    let err = spec_flash_attention(
        ScalarType::F32,
        0.125,
        false,
        &[1, 8, 32],
        &[1, 8, 32, 64],
        AttentionLayout::HeadsFirst,
    )
    .unwrap_err();
    assert!(err.contains("rank 4"));
}

#[test]
fn test_spec_flash_attention_seq_first() {
    use nn_dsl::AttentionLayout;
    // SeqFirst: Q [1, 32, 8, 64], K [1, 32, 8, 64]
    let spec = spec_flash_attention(
        ScalarType::F32,
        0.125,
        false,
        &[1, 32, 8, 64],
        &[1, 32, 8, 64],
        AttentionLayout::SeqFirst,
    )
    .unwrap();
    assert_eq!(spec.kernel_name, "flash_attn_f32_seq_first");
}

// =========================================================================
// LinearActivation tests (#3503 D3)
// =========================================================================

#[test]
fn test_spec_linear_activation_f32_naive() {
    use nn_dsl::GemmActivation;
    // Small shape that won't route to simdgroup: [1, 16] → in=16, out=32
    let spec = spec_linear_activation(
        ScalarType::F32,
        &GemmActivation::Relu,
        16,
        32,
        true,
        &[1, 16],
    )
    .unwrap();
    // Naive path
    assert!(spec.kernel_name.starts_with("la_"));
    assert!(spec.kernel_name.contains("relu"));
    assert_eq!(spec.output_bytes, 1 * 32 * 4);
    // With bias: input + weight + bias + output = 4 bindings
    assert_eq!(spec.bindings.len(), 4);
    assert_eq!(spec.param_count, 3);
}

#[test]
fn test_spec_linear_activation_no_bias() {
    use nn_dsl::GemmActivation;
    let spec = spec_linear_activation(
        ScalarType::F32,
        &GemmActivation::Gelu,
        16,
        32,
        false,
        &[1, 16],
    )
    .unwrap();
    assert!(spec.kernel_name.contains("gelu"));
    // No bias: input + weight + output = 3 bindings
    assert_eq!(spec.bindings.len(), 3);
    assert_eq!(spec.param_count, 2);
}

// =========================================================================
// NormLinear tests (#3503 D3)
// =========================================================================

#[test]
fn test_spec_norm_linear_layer_norm_small() {
    use nn_dsl::trace_compile::FusedNormKind;
    // Small shape (won't hit simdgroup): input [1, 64], hidden=64, out=32
    let spec = spec_norm_linear(
        ScalarType::F32,
        FusedNormKind::LayerNorm,
        1e-5,
        &[1, 64],
        64,
        32,
        true,
    )
    .unwrap();
    assert_eq!(spec.kernel_name, "fused_norm_linear_ln_float_b1");
    // flat_rows = 1
    assert_eq!(spec.grid, [1, 1, 1]);
    assert_eq!(spec.output_bytes, 1 * 32 * 4);
    // LN + bias bindings: input + norm_w + norm_b + weight + bias + output + 4 consts = 10
    assert_eq!(spec.bindings.len(), 10);
    assert_eq!(spec.param_count, 5);
    // Threadgroup memory for normalized values
    assert_eq!(spec.threadgroup_memory_bytes, 64 * 4);
}

#[test]
fn test_spec_norm_linear_rms_norm_small() {
    use nn_dsl::trace_compile::FusedNormKind;
    let spec = spec_norm_linear(
        ScalarType::F32,
        FusedNormKind::RmsNorm,
        1e-6,
        &[1, 64],
        64,
        32,
        false,
    )
    .unwrap();
    assert_eq!(spec.kernel_name, "fused_norm_linear_rms_float_b0");
    // RMS, no bias: input + norm_w + weight + output + 4 consts = 8
    assert_eq!(spec.bindings.len(), 8);
    assert_eq!(spec.param_count, 3);
}

#[test]
fn test_spec_norm_linear_simdgroup_rejected() {
    use nn_dsl::trace_compile::FusedNormKind;
    // Large dimensions that would route to simdgroup: [32, 768], out=768
    let result = spec_norm_linear(
        ScalarType::F32,
        FusedNormKind::LayerNorm,
        1e-5,
        &[32, 768],
        768,
        768,
        true,
    );
    // Should be rejected (simdgroup path needs MultiKernelSpec)
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("simdgroup"));
}

#[test]
fn test_spec_norm_linear_zero_dim() {
    use nn_dsl::trace_compile::FusedNormKind;
    let err = spec_norm_linear(
        ScalarType::F32,
        FusedNormKind::LayerNorm,
        1e-5,
        &[0, 64],
        64,
        32,
        true,
    )
    .unwrap_err();
    assert!(err.contains("zero-size"));
}
