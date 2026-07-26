// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4186.
//!
//! Extended tests for PTX kernel generation covering activations, reductions,
//! matmul, elementwise binary/unary, normalization, softmax, linear, and
//! reference CPU implementations.

use crate::{
    // Elementwise binary/unary
    add_reference,
    // Reductions
    argmax_reference,
    argmin_reference,
    div_reference,
    // CUDA C++ emission (ptx_emit)
    emit_activation_kernels,
    emit_elementwise_kernel,
    emit_matmul_kernel,
    // Activations
    emit_ptx_activation,
    emit_ptx_activation_default,
    // LayerNorm
    emit_ptx_layernorm,
    // Matmul
    emit_ptx_matmul,
    emit_ptx_matmul_default,
    // RMSNorm
    emit_ptx_rmsnorm,
    emit_reduction_kernel,
    emit_softmax_kernel,
    exp_reference,
    gelu_fast_reference,
    gelu_reference,
    generate_add_ptx,
    generate_all_activation_ptx,
    generate_argmax_ptx,
    generate_argmin_ptx,
    generate_div_ptx,
    generate_exp_ptx,
    generate_layernorm_ptx,
    // Linear
    generate_linear_no_bias_ptx,
    generate_linear_ptx,
    generate_linear_relu_ptx,
    generate_log_ptx,
    // Softmax
    generate_log_softmax_ptx,
    generate_matmul_ptx,
    generate_matmul_tiled_ptx,
    generate_max_ptx,
    generate_mean_ptx,
    generate_mul_ptx,
    generate_neg_ptx,
    generate_rmsnorm_ptx,
    generate_scalar_mul_ptx,
    generate_softmax_ptx,
    generate_sqrt_ptx,
    generate_sub_ptx,
    generate_sum_ptx,
    layernorm_reference,
    linear_reference,
    log_reference,
    log_softmax_reference,
    matmul_reference,
    max_reference,
    mean_reference,
    mish_reference,
    mul_reference,
    neg_reference,
    ptx_activation_launch_config,
    ptx_elementwise_launch_config,
    ptx_layernorm_launch_config,
    ptx_matmul_launch_config,
    // Codegen helpers
    ptx_prelude,
    ptx_reduce_launch_config,
    ptx_rmsnorm_launch_config,
    ptx_softmax_launch_config,
    rmsnorm_reference,
    scalar_mul_reference,
    silu_reference,
    snake_reference,
    softmax_reference,
    sqrt_reference,
    sub_reference,
    sum_reference,
    PtxActivation,
    PtxActivationConfig,
    PtxLayerNormConfig,
    PtxMatmulConfig,
    PtxRmsNormConfig,
    ReductionOp,
    ELEMENTWISE_BLOCK_SIZE,
    LINEAR_BLOCK_SIZE,
    MATMUL_BLOCK_SIZE,
    PTX_MATMUL_MAX_TILE,
    PTX_MATMUL_MIN_TILE,
    PTX_MATMUL_TILE_SIZE,
    PTX_VERSION,
    REDUCE_BLOCK_SIZE,
    SOFTMAX_BLOCK_SIZE,
    WARP_SIZE,
};

// =========================================================================
// Section 1: PTX Output Structure
// =========================================================================

#[test]
fn test_ptx_prelude_contains_version_directive() {
    let prelude = ptx_prelude("sm_80");
    assert!(prelude.contains(&format!(".version {PTX_VERSION}")));
    assert!(prelude.contains(".target sm_80"));
    assert!(prelude.contains(".address_size 64"));
}

#[test]
fn test_ptx_prelude_custom_sm_target() {
    let prelude = ptx_prelude("sm_90");
    assert!(prelude.contains(".target sm_90"));
    assert!(prelude.contains(".version"));
}

// =========================================================================
// Section 2: Activation PTX Kernels
// =========================================================================

#[test]
fn test_activation_gelu_ptx_entry_and_version() {
    let config = PtxActivationConfig::new("gelu_kernel", PtxActivation::Gelu);
    let ptx = emit_ptx_activation(&config).unwrap();
    assert!(ptx.contains(".entry gelu_kernel"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".target sm_80"));
    assert!(ptx.contains(".param .u64 param_input"));
    assert!(ptx.contains(".param .u64 param_output"));
    assert!(ptx.contains(".param .u32 param_n"));
}

#[test]
fn test_activation_gelu_ptx_contains_erf_approximation() {
    let ptx = emit_ptx_activation_default("gelu_f32", PtxActivation::Gelu).unwrap();
    // GELU uses erf approximation via Horner + exp
    assert!(ptx.contains("ex2.approx.f32"));
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains("rcp.approx.f32"));
}

#[test]
fn test_activation_gelu_fast_ptx_entry() {
    let ptx = emit_ptx_activation_default("gelu_fast_f32", PtxActivation::GeluFast).unwrap();
    assert!(ptx.contains(".entry gelu_fast_f32"));
    assert!(ptx.contains("GELU fast"));
}

#[test]
fn test_activation_silu_ptx_contains_sigmoid_computation() {
    let ptx = emit_ptx_activation_default("silu_f32", PtxActivation::Silu).unwrap();
    assert!(ptx.contains(".entry silu_f32"));
    assert!(ptx.contains("SiLU"));
    assert!(ptx.contains("neg.f32"));
    assert!(ptx.contains("ex2.approx.f32"));
    assert!(ptx.contains("rcp.approx.f32"));
    assert!(ptx.contains("mul.f32"));
}

#[test]
fn test_activation_mish_ptx_contains_tanh_softplus() {
    let ptx = emit_ptx_activation_default("mish_f32", PtxActivation::Mish).unwrap();
    assert!(ptx.contains(".entry mish_f32"));
    assert!(ptx.contains("Mish"));
    assert!(ptx.contains("lg2.approx.f32"));
    assert!(ptx.contains("ex2.approx.f32"));
}

#[test]
fn test_activation_snake_ptx_has_alpha_parameter() {
    let ptx = emit_ptx_activation_default("snake_f32", PtxActivation::Snake).unwrap();
    assert!(ptx.contains(".entry snake_f32"));
    assert!(ptx.contains(".param .f32 param_alpha"));
    assert!(ptx.contains("sin.approx.f32"));
    assert!(ptx.contains("rcp.approx.f32"));
}

#[test]
fn test_activation_snake_requires_alpha() {
    assert!(PtxActivation::Snake.requires_alpha());
    assert!(!PtxActivation::Gelu.requires_alpha());
    assert!(!PtxActivation::Silu.requires_alpha());
    assert!(!PtxActivation::Mish.requires_alpha());
    assert!(!PtxActivation::GeluFast.requires_alpha());
}

#[test]
fn test_activation_config_validation_empty_name_rejected() {
    let config = PtxActivationConfig::new("", PtxActivation::Silu);
    assert!(config.validate().is_err());
}

#[test]
fn test_activation_config_validation_zero_block_rejected() {
    let config = PtxActivationConfig::new("k", PtxActivation::Silu).with_block_size(0);
    assert!(config.validate().is_err());
}

#[test]
fn test_activation_custom_sm_target() {
    let config = PtxActivationConfig::new("test_k", PtxActivation::Gelu).with_sm_target("sm_70");
    let ptx = emit_ptx_activation(&config).unwrap();
    assert!(ptx.contains(".target sm_70"));
}

#[test]
fn test_generate_all_activation_ptx_returns_all_five() {
    let all = generate_all_activation_ptx();
    assert_eq!(all.len(), 5);
    let names: Vec<&str> = all.iter().map(|(name, _)| *name).collect();
    assert!(names.contains(&"gelu"));
    assert!(names.contains(&"gelu_fast"));
    assert!(names.contains(&"silu"));
    assert!(names.contains(&"mish"));
    assert!(names.contains(&"snake"));
    // Each should contain valid PTX
    for (_, ptx) in &all {
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".entry"));
    }
}

#[test]
fn test_activation_launch_config_standard() {
    let (grid, block) = ptx_activation_launch_config(1024, 256);
    assert_eq!(grid, [4, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_activation_launch_config_not_multiple() {
    let (grid, block) = ptx_activation_launch_config(1000, 256);
    assert_eq!(grid, [4, 1, 1]); // ceil(1000/256) = 4
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_activation_launch_config_zero_block_uses_default() {
    let (grid, block) = ptx_activation_launch_config(512, 0);
    assert_eq!(block, [256, 1, 1]); // falls back to 256
    assert_eq!(grid, [2, 1, 1]);
}

// =========================================================================
// Section 3: Activation Reference Functions
// =========================================================================

#[test]
fn test_silu_reference_at_zero_returns_zero() {
    assert!((silu_reference(0.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_silu_reference_positive_input() {
    let result = silu_reference(1.0);
    // silu(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ~= 0.7311
    assert!((result - 0.7311).abs() < 0.01);
}

#[test]
fn test_silu_reference_negative_input() {
    let result = silu_reference(-1.0);
    // silu(-1) = -1 * sigmoid(-1) = -1 / (1 + exp(1)) ~= -0.2689
    assert!((result - (-0.2689)).abs() < 0.01);
}

#[test]
fn test_gelu_reference_at_zero_returns_zero() {
    assert!((gelu_reference(0.0) - 0.0).abs() < 1e-5);
}

#[test]
fn test_gelu_reference_large_positive() {
    // For large x, gelu(x) -> x
    let x = 5.0;
    assert!((gelu_reference(x) - x).abs() < 0.01);
}

#[test]
fn test_gelu_reference_large_negative() {
    // For large negative x, gelu(x) -> 0
    assert!((gelu_reference(-5.0)).abs() < 0.01);
}

#[test]
fn test_gelu_fast_reference_at_zero_returns_zero() {
    assert!((gelu_fast_reference(0.0) - 0.0).abs() < 1e-5);
}

#[test]
fn test_gelu_fast_reference_positive() {
    let result = gelu_fast_reference(1.0);
    // gelu_fast(1) = 1 * sigmoid(1.702) ~= 0.846
    assert!(result > 0.8 && result < 0.9);
}

#[test]
fn test_mish_reference_at_zero_returns_zero() {
    assert!((mish_reference(0.0) - 0.0).abs() < 1e-5);
}

#[test]
fn test_mish_reference_positive() {
    let result = mish_reference(1.0);
    // mish(1) = 1 * tanh(softplus(1)) = tanh(ln(1+e)) ~= 0.8651
    assert!((result - 0.8651).abs() < 0.01);
}

#[test]
fn test_snake_reference_at_zero_returns_zero() {
    let result = snake_reference(0.0, 1.0);
    // snake(0, alpha) = 0 + (1/alpha) * sin(0)^2 = 0
    assert!((result - 0.0).abs() < 1e-6);
}

#[test]
fn test_snake_reference_preserves_identity_component() {
    // snake(x, alpha) >= x for all x since sin^2 >= 0 and 1/alpha > 0
    let x = 2.0;
    let alpha = 1.5;
    let result = snake_reference(x, alpha);
    assert!(result >= x);
}

// =========================================================================
// Section 4: Elementwise Binary PTX Kernels
// =========================================================================

#[test]
fn test_add_ptx_entry_and_instruction() {
    let ptx = generate_add_ptx(1024);
    assert!(ptx.contains(".entry ptx_add_f32"));
    assert!(ptx.contains("add.f32"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".target"));
    assert!(ptx.contains(".param .u64 param_a"));
    assert!(ptx.contains(".param .u64 param_b"));
    assert!(ptx.contains(".param .u64 param_output"));
    assert!(ptx.contains(".param .u32 param_n"));
}

#[test]
fn test_sub_ptx_entry_and_instruction() {
    let ptx = generate_sub_ptx(512);
    assert!(ptx.contains(".entry ptx_sub_f32"));
    assert!(ptx.contains("sub.f32"));
}

#[test]
fn test_mul_ptx_entry_and_instruction() {
    let ptx = generate_mul_ptx(256);
    assert!(ptx.contains(".entry ptx_mul_f32"));
    assert!(ptx.contains("mul.f32"));
}

#[test]
fn test_div_ptx_entry_and_instruction() {
    let ptx = generate_div_ptx(128);
    assert!(ptx.contains(".entry ptx_div_f32"));
    assert!(ptx.contains("div.approx.f32"));
}

#[test]
fn test_exp_ptx_entry_and_instruction() {
    let ptx = generate_exp_ptx(1024);
    assert!(ptx.contains(".entry ptx_exp_f32"));
    assert!(ptx.contains("ex2.approx.f32"));
}

#[test]
fn test_log_ptx_entry_and_instruction() {
    let ptx = generate_log_ptx(512);
    assert!(ptx.contains(".entry ptx_log_f32"));
    assert!(ptx.contains("lg2.approx.f32"));
}

#[test]
fn test_sqrt_ptx_entry_and_instruction() {
    let ptx = generate_sqrt_ptx(256);
    assert!(ptx.contains(".entry ptx_sqrt_f32"));
    assert!(ptx.contains("sqrt.approx.f32"));
}

#[test]
fn test_neg_ptx_entry_and_instruction() {
    let ptx = generate_neg_ptx(128);
    assert!(ptx.contains(".entry ptx_neg_f32"));
    assert!(ptx.contains("neg.f32"));
}

#[test]
fn test_scalar_mul_ptx_entry_and_scalar_param() {
    let ptx = generate_scalar_mul_ptx(1024);
    assert!(ptx.contains(".entry ptx_scalar_mul_f32"));
    assert!(ptx.contains(".param .f32 param_scalar"));
    assert!(ptx.contains("mul.f32"));
}

#[test]
fn test_elementwise_all_have_grid_stride_loop() {
    for ptx in [
        generate_add_ptx(64),
        generate_sub_ptx(64),
        generate_mul_ptx(64),
        generate_div_ptx(64),
        generate_exp_ptx(64),
        generate_log_ptx(64),
        generate_sqrt_ptx(64),
        generate_neg_ptx(64),
    ] {
        assert!(ptx.contains("nctaid.x"), "Missing grid stride loop");
        assert!(ptx.contains("ret;"));
    }
}

#[test]
fn test_elementwise_launch_config_exact() {
    let (grid, block) = ptx_elementwise_launch_config(512);
    assert_eq!(block, [ELEMENTWISE_BLOCK_SIZE, 1, 1]);
    assert_eq!(grid, [2, 1, 1]); // 512 / 256 = 2
}

#[test]
fn test_elementwise_launch_config_remainder() {
    let (grid, block) = ptx_elementwise_launch_config(300);
    assert_eq!(block, [256, 1, 1]);
    assert_eq!(grid, [2, 1, 1]); // ceil(300/256) = 2
}

// =========================================================================
// Section 5: Elementwise Reference Functions
// =========================================================================

#[test]
fn test_add_reference_basic() {
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    let result = add_reference(&a, &b);
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_sub_reference_basic() {
    let a = vec![10.0, 20.0, 30.0];
    let b = vec![1.0, 2.0, 3.0];
    let result = sub_reference(&a, &b);
    assert_eq!(result, vec![9.0, 18.0, 27.0]);
}

#[test]
fn test_mul_reference_basic() {
    let a = vec![2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0];
    let result = mul_reference(&a, &b);
    assert_eq!(result, vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_div_reference_basic() {
    let a = vec![10.0, 20.0, 30.0];
    let b = vec![2.0, 5.0, 10.0];
    let result = div_reference(&a, &b);
    assert_eq!(result, vec![5.0, 4.0, 3.0]);
}

#[test]
fn test_exp_reference_known_values() {
    let input = vec![0.0, 1.0];
    let result = exp_reference(&input);
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - std::f32::consts::E).abs() < 1e-5);
}

#[test]
fn test_log_reference_known_values() {
    let input = vec![1.0, std::f32::consts::E];
    let result = log_reference(&input);
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_sqrt_reference_known_values() {
    let input = vec![4.0, 9.0, 16.0];
    let result = sqrt_reference(&input);
    assert!((result[0] - 2.0).abs() < 1e-6);
    assert!((result[1] - 3.0).abs() < 1e-6);
    assert!((result[2] - 4.0).abs() < 1e-6);
}

#[test]
fn test_neg_reference_basic() {
    let input = vec![1.0, -2.0, 0.0];
    let result = neg_reference(&input);
    assert_eq!(result, vec![-1.0, 2.0, -0.0]);
}

#[test]
fn test_scalar_mul_reference_basic() {
    let input = vec![1.0, 2.0, 3.0];
    let result = scalar_mul_reference(&input, 3.0);
    assert_eq!(result, vec![3.0, 6.0, 9.0]);
}

// =========================================================================
// Section 6: Reduction PTX Kernels
// =========================================================================

#[test]
fn test_sum_ptx_structure() {
    let ptx = generate_sum_ptx(1024);
    assert!(ptx.contains(".entry ptx_sum_f32"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains(".shared"));
    assert!(ptx.contains("add.f32"));
    assert!(ptx.contains("bar.sync"));
    assert!(ptx.contains("st.global.f32"));
}

#[test]
fn test_max_ptx_structure() {
    let ptx = generate_max_ptx(512);
    assert!(ptx.contains(".entry ptx_max_f32"));
    assert!(ptx.contains("max.f32"));
    assert!(ptx.contains(".shared"));
    assert!(ptx.contains("bar.sync"));
}

#[test]
fn test_mean_ptx_structure() {
    let ptx = generate_mean_ptx(256);
    assert!(ptx.contains(".entry ptx_mean_f32"));
    assert!(ptx.contains("add.f32"));
    assert!(ptx.contains("div.approx.f32"));
    assert!(ptx.contains("cvt.rn.f32.u32"));
}

#[test]
fn test_argmax_ptx_dual_shared_memory() {
    let ptx = generate_argmax_ptx(512);
    assert!(ptx.contains(".entry ptx_argmax_f32"));
    assert!(ptx.contains("smem_val"));
    assert!(ptx.contains("smem_idx"));
    assert!(ptx.contains("setp.gt.f32"));
}

#[test]
fn test_argmin_ptx_uses_lt_compare() {
    let ptx = generate_argmin_ptx(512);
    assert!(ptx.contains(".entry ptx_argmin_f32"));
    assert!(ptx.contains("setp.lt.f32"));
}

#[test]
fn test_all_reduce_kernels_have_proper_target() {
    for ptx in [
        generate_sum_ptx(64),
        generate_max_ptx(64),
        generate_mean_ptx(64),
        generate_argmax_ptx(64),
        generate_argmin_ptx(64),
    ] {
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".version"));
        assert!(ptx.contains("ret;"));
    }
}

#[test]
fn test_reduce_launch_config_single_block() {
    let (grid, block) = ptx_reduce_launch_config();
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [REDUCE_BLOCK_SIZE as usize, 1, 1]);
}

// =========================================================================
// Section 7: Reduction Reference Functions
// =========================================================================

#[test]
fn test_sum_reference_multiple_values() {
    assert!((sum_reference(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 15.0).abs() < 1e-6);
}

#[test]
fn test_sum_reference_negative_values() {
    assert!((sum_reference(&[-1.0, -2.0, 3.0]) - 0.0).abs() < 1e-6);
}

#[test]
fn test_max_reference_with_negatives() {
    assert!((max_reference(&[-10.0, -5.0, -20.0]) - (-5.0)).abs() < 1e-6);
}

#[test]
fn test_max_reference_empty_returns_neg_inf() {
    assert_eq!(max_reference(&[]), f32::NEG_INFINITY);
}

#[test]
fn test_mean_reference_uniform() {
    assert!((mean_reference(&[5.0, 5.0, 5.0, 5.0]) - 5.0).abs() < 1e-6);
}

#[test]
fn test_mean_reference_mixed() {
    assert!((mean_reference(&[0.0, 10.0]) - 5.0).abs() < 1e-6);
}

#[test]
fn test_argmax_reference_tie_first_wins() {
    assert_eq!(argmax_reference(&[5.0, 5.0, 5.0]), 0);
}

#[test]
fn test_argmin_reference_empty_returns_zero() {
    assert_eq!(argmin_reference(&[]), 0);
}

#[test]
fn test_argmin_reference_single_element() {
    assert_eq!(argmin_reference(&[42.0]), 0);
}

// =========================================================================
// Section 8: Matmul PTX Kernels
// =========================================================================

#[test]
fn test_matmul_naive_ptx_structure() {
    let ptx = generate_matmul_ptx(64, 32, 48);
    assert!(ptx.contains(".entry naive_matmul_f32"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains("fma.rn.f32"));
    assert!(ptx.contains(".param .u64 param_A"));
    assert!(ptx.contains(".param .u64 param_B"));
    assert!(ptx.contains(".param .u64 param_C"));
    assert!(ptx.contains(".param .u32 param_M"));
    assert!(ptx.contains(".param .u32 param_N"));
    assert!(ptx.contains(".param .u32 param_K"));
}

#[test]
fn test_matmul_tiled_ptx_contains_shared_memory() {
    let ptx = generate_matmul_tiled_ptx(128, 64, 96, 16);
    assert!(ptx.contains(".entry tiled_matmul_f32"));
    assert!(ptx.contains(".shared .align 4"));
    assert!(ptx.contains("bar.sync"));
    assert!(ptx.contains("fma.rn.f32"));
}

#[test]
fn test_matmul_tiled_ptx_tile_8() {
    let ptx = generate_matmul_tiled_ptx(32, 32, 32, 8);
    assert!(ptx.contains(".shared .align 4 .f32 As[64]")); // 8*8=64
    assert!(ptx.contains(".shared .align 4 .f32 Bs[64]"));
}

#[test]
fn test_emit_ptx_matmul_config_validation() {
    // Tile too small
    let config = PtxMatmulConfig::new("k").with_tile_size(2);
    assert!(emit_ptx_matmul(&config).is_err());

    // Tile too large
    let config = PtxMatmulConfig::new("k").with_tile_size(64);
    assert!(emit_ptx_matmul(&config).is_err());

    // Empty name
    let config = PtxMatmulConfig {
        kernel_name: String::new(),
        tile_size: 16,
        sm_target: "sm_80".to_string(),
    };
    assert!(emit_ptx_matmul(&config).is_err());
}

#[test]
fn test_matmul_config_shared_memory_bytes() {
    let config = PtxMatmulConfig::new("test");
    // Default tile=16: 2 * 16 * 16 * 4 = 2048
    assert_eq!(config.shared_memory_bytes(), 2048);

    let config = PtxMatmulConfig::new("test").with_tile_size(32);
    assert_eq!(config.shared_memory_bytes(), 8192);
}

#[test]
fn test_matmul_config_threads_per_block() {
    let config = PtxMatmulConfig::new("test");
    assert_eq!(config.threads_per_block(), 256); // 16*16
}

#[test]
fn test_matmul_launch_config() {
    let (grid, block) = ptx_matmul_launch_config(128, 64, 16);
    assert_eq!(grid, [4, 8, 1]); // [ceil(64/16), ceil(128/16)]
    assert_eq!(block, [16, 16, 1]);
}

#[test]
fn test_matmul_constants() {
    assert_eq!(MATMUL_BLOCK_SIZE, 16);
    assert_eq!(PTX_MATMUL_TILE_SIZE, 16);
    assert_eq!(PTX_MATMUL_MIN_TILE, 4);
    assert_eq!(PTX_MATMUL_MAX_TILE, 32);
}

#[test]
fn test_emit_ptx_matmul_default_uses_16_tile() {
    let ptx = emit_ptx_matmul_default("gemm_f32").unwrap();
    assert!(ptx.contains(".entry gemm_f32"));
    assert!(ptx.contains(".shared .align 4"));
    // Default tile = 16, so 16*16=256 entries per shared tile
    assert!(ptx.contains(".f32 As[256]"));
    assert!(ptx.contains(".f32 Bs[256]"));
}

// =========================================================================
// Section 9: Matmul Reference Function
// =========================================================================

#[test]
fn test_matmul_reference_identity() {
    // Multiply by identity matrix
    let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
    let b = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let c = matmul_reference(&a, &b, 2, 2, 2);
    assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_matmul_reference_1x1() {
    let a = vec![3.0];
    let b = vec![4.0];
    let c = matmul_reference(&a, &b, 1, 1, 1);
    assert!((c[0] - 12.0).abs() < 1e-6);
}

#[test]
fn test_matmul_reference_rectangular() {
    // A: 2x3, B: 3x2 => C: 2x2
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let c = matmul_reference(&a, &b, 2, 3, 2);
    // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 154
    assert!((c[0] - 58.0).abs() < 1e-4);
    assert!((c[1] - 64.0).abs() < 1e-4);
    assert!((c[2] - 139.0).abs() < 1e-4);
    assert!((c[3] - 154.0).abs() < 1e-4);
}

// =========================================================================
// Section 10: LayerNorm PTX Kernels
// =========================================================================

#[test]
fn test_layernorm_ptx_entry_and_params() {
    let config = PtxLayerNormConfig::new("ln_768", 768, 1e-5);
    let ptx = emit_ptx_layernorm(&config).unwrap();
    assert!(ptx.contains(".entry ln_768"));
    assert!(ptx.contains(".param .u64 param_input"));
    assert!(ptx.contains(".param .u64 param_output"));
    assert!(ptx.contains(".param .u64 param_gamma"));
    assert!(ptx.contains(".param .u64 param_beta"));
    assert!(ptx.contains("rsqrt.approx.f32"));
}

#[test]
fn test_layernorm_ptx_small_dim_warp_only() {
    let config = PtxLayerNormConfig::new("ln_16", 16, 1e-5);
    assert!(config.is_warp_only());
    assert_eq!(config.shared_memory_bytes(), 0);
    let ptx = emit_ptx_layernorm(&config).unwrap();
    assert!(ptx.contains("shfl.down.sync"));
    // No shared memory for warp-only
    assert!(!ptx.contains("warp_scratch"));
}

#[test]
fn test_layernorm_ptx_large_dim_uses_shared() {
    let config = PtxLayerNormConfig::new("ln_256", 256, 1e-5);
    assert!(!config.is_warp_only());
    assert!(config.shared_memory_bytes() > 0);
    let ptx = emit_ptx_layernorm(&config).unwrap();
    assert!(ptx.contains("warp_scratch"));
}

#[test]
fn test_layernorm_config_validation_rejects_zero_shape() {
    let config = PtxLayerNormConfig::new("k", 0, 1e-5);
    assert!(config.validate().is_err());
}

#[test]
fn test_layernorm_config_validation_rejects_nan_eps() {
    let config = PtxLayerNormConfig::new("k", 64, f32::NAN);
    assert!(config.validate().is_err());
}

#[test]
fn test_layernorm_config_validation_rejects_negative_eps() {
    let config = PtxLayerNormConfig::new("k", 64, -1e-5);
    assert!(config.validate().is_err());
}

#[test]
fn test_generate_layernorm_ptx_convenience() {
    let ptx = generate_layernorm_ptx(512);
    assert!(ptx.contains(".entry ptx_layernorm_f32"));
    assert!(ptx.contains(".version"));
}

#[test]
fn test_layernorm_launch_config() {
    let (grid, block) = ptx_layernorm_launch_config(32, 768);
    assert_eq!(grid, [32, 1, 1]); // one block per row
    assert!(block[0] > 0);
    assert_eq!(block[1], 1);
    assert_eq!(block[2], 1);
}

// =========================================================================
// Section 11: LayerNorm Reference Function
// =========================================================================

#[test]
fn test_layernorm_reference_uniform_input() {
    let input = vec![5.0, 5.0, 5.0, 5.0];
    let gamma = vec![1.0, 1.0, 1.0, 1.0];
    let beta = vec![0.0, 0.0, 0.0, 0.0];
    let result = layernorm_reference(&input, &gamma, &beta, 1e-5);
    // mean=5, var=0, so (x-mean)/sqrt(var+eps) ~= 0 for all
    for &v in &result {
        assert!(v.abs() < 0.01);
    }
}

#[test]
fn test_layernorm_reference_with_affine() {
    let input = vec![1.0, 3.0];
    let gamma = vec![2.0, 2.0];
    let beta = vec![1.0, 1.0];
    let result = layernorm_reference(&input, &gamma, &beta, 1e-5);
    // mean=2, var=1, inv_std=1/sqrt(1+1e-5)~=1
    // y[0] = 2*(1-2)*1 + 1 = -1
    // y[1] = 2*(3-2)*1 + 1 = 3
    assert!((result[0] - (-1.0)).abs() < 0.01);
    assert!((result[1] - 3.0).abs() < 0.01);
}

// =========================================================================
// Section 12: RMSNorm PTX Kernels
// =========================================================================

#[test]
fn test_rmsnorm_ptx_entry_and_params() {
    let config = PtxRmsNormConfig::new("rms_4096", 4096, 1e-5);
    let ptx = emit_ptx_rmsnorm(&config).unwrap();
    assert!(ptx.contains(".entry rms_4096"));
    assert!(ptx.contains(".param .u64 param_input"));
    assert!(ptx.contains(".param .u64 param_output"));
    assert!(ptx.contains(".param .u64 param_weight"));
    // RMSNorm should NOT have param_beta
    assert!(!ptx.contains("param_beta"));
    assert!(ptx.contains("rsqrt.approx.f32"));
}

#[test]
fn test_rmsnorm_ptx_small_dim_warp_only() {
    let config = PtxRmsNormConfig::new("rms_16", 16, 1e-6);
    assert!(config.is_warp_only());
    let ptx = emit_ptx_rmsnorm(&config).unwrap();
    assert!(ptx.contains("shfl.down.sync"));
}

#[test]
fn test_rmsnorm_config_validation_rejects_zero_dim() {
    let config = PtxRmsNormConfig::new("k", 0, 1e-5);
    assert!(config.validate().is_err());
}

#[test]
fn test_rmsnorm_config_validation_rejects_inf_eps() {
    let config = PtxRmsNormConfig::new("k", 64, f32::INFINITY);
    assert!(config.validate().is_err());
}

#[test]
fn test_generate_rmsnorm_ptx_convenience() {
    let ptx = generate_rmsnorm_ptx(128, 1e-5);
    assert!(ptx.contains(".entry ptx_rmsnorm_f32"));
}

#[test]
fn test_rmsnorm_launch_config() {
    let (grid, block) = ptx_rmsnorm_launch_config(64, 4096);
    assert_eq!(grid, [64, 1, 1]);
    assert!(block[0] > 0);
}

// =========================================================================
// Section 13: RMSNorm Reference Function
// =========================================================================

#[test]
fn test_rmsnorm_reference_unit_weight() {
    let input = vec![3.0, 4.0]; // rms = sqrt((9+16)/2) = sqrt(12.5) ~= 3.536
    let weight = vec![1.0, 1.0];
    let result = rmsnorm_reference(&input, &weight, 1e-5);
    let mean_sq = f32::midpoint(9.0, 16.0);
    let inv_rms = 1.0 / (mean_sq + 1e-5_f32).sqrt();
    assert!((result[0] - 3.0 * inv_rms).abs() < 1e-5);
    assert!((result[1] - 4.0 * inv_rms).abs() < 1e-5);
}

#[test]
fn test_rmsnorm_reference_preserves_zero() {
    let input = vec![0.0, 0.0, 1.0];
    let weight = vec![1.0, 1.0, 1.0];
    let result = rmsnorm_reference(&input, &weight, 1e-6);
    // Zero inputs should remain zero after normalization
    assert!(result[0].abs() < 1e-6);
    assert!(result[1].abs() < 1e-6);
}

// =========================================================================
// Section 14: Softmax PTX Kernels
// =========================================================================

#[test]
fn test_softmax_ptx_entry_and_structure() {
    let ptx = generate_softmax_ptx(false, 512);
    assert!(ptx.contains(".entry ptx_softmax_f32"));
    assert!(ptx.contains(".version"));
    assert!(ptx.contains("ex2.approx.f32"));
    // Normalization uses inv_sum = 1.0 / sum via `div.approx.f32`, the
    // reciprocal idiom emitted here (not a standalone `rcp` instruction).
    assert!(ptx.contains("div.approx.f32"));
}

#[test]
fn test_log_softmax_ptx_entry() {
    let ptx = generate_log_softmax_ptx(256);
    assert!(ptx.contains(".entry ptx_log_softmax_f32"));
}

#[test]
fn test_softmax_launch_config() {
    let (grid, block) = ptx_softmax_launch_config(32, 512);
    assert_eq!(grid, [32, 1, 1]);
    assert!(block[0] > 0);
}

#[test]
fn test_softmax_block_size_constant() {
    assert_eq!(SOFTMAX_BLOCK_SIZE, 256);
}

// =========================================================================
// Section 15: Softmax Reference Functions
// =========================================================================

#[test]
fn test_softmax_reference_sums_to_one() {
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let result = softmax_reference(&input);
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_softmax_reference_monotonic() {
    let input = vec![1.0, 2.0, 3.0];
    let result = softmax_reference(&input);
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
}

#[test]
fn test_softmax_reference_empty() {
    assert!(softmax_reference(&[]).is_empty());
}

#[test]
fn test_softmax_reference_single_element() {
    let result = softmax_reference(&[42.0]);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn test_log_softmax_reference_known_property() {
    let input = vec![1.0, 2.0, 3.0];
    let result = log_softmax_reference(&input);
    // exp(log_softmax) should equal softmax
    let sm = softmax_reference(&input);
    for (ls, s) in result.iter().zip(sm.iter()) {
        assert!((ls.exp() - s).abs() < 1e-5);
    }
}

#[test]
fn test_log_softmax_reference_empty() {
    assert!(log_softmax_reference(&[]).is_empty());
}

// =========================================================================
// Section 16: Linear PTX Kernels
// =========================================================================

#[test]
fn test_linear_ptx_entry_and_params() {
    let ptx = generate_linear_ptx(768, 3072);
    assert!(ptx.contains(".entry linear_bias_f32"));
    assert!(ptx.contains(".param .u64 param_input"));
    assert!(ptx.contains(".param .u64 param_weight"));
    assert!(ptx.contains(".param .u64 param_bias"));
    assert!(ptx.contains(".param .u64 param_output"));
}

#[test]
fn test_linear_no_bias_ptx_no_bias_param() {
    let ptx = generate_linear_no_bias_ptx(512, 1024);
    assert!(ptx.contains(".entry linear_no_bias_f32"));
    // Should not reference a bias load in the main computation
    assert!(!ptx.contains("param_bias"));
}

#[test]
fn test_linear_relu_ptx_has_relu_logic() {
    let ptx = generate_linear_relu_ptx(256, 512);
    assert!(ptx.contains(".entry linear_relu_f32"));
    // Should contain a max/relu comparison
    assert!(ptx.contains("max.f32") || ptx.contains("setp.gt.f32") || ptx.contains("selp.f32"));
}

#[test]
fn test_linear_block_size_constant() {
    assert_eq!(LINEAR_BLOCK_SIZE, 256);
}

// =========================================================================
// Section 17: Linear Reference Function
// =========================================================================

#[test]
fn test_linear_reference_with_bias() {
    // input: [1, 2] (1x2), weight: [2, 3] (2x3 = in_f x out_f), bias: [3]
    let input = vec![1.0, 2.0];
    let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let bias = vec![0.1, 0.2, 0.3];
    let result = linear_reference(&input, &weight, Some(&bias), 2, 3);
    // out[0] = 1*1 + 2*4 + 0.1 = 9.1
    // out[1] = 1*2 + 2*5 + 0.2 = 12.2
    // out[2] = 1*3 + 2*6 + 0.3 = 15.3
    assert!((result[0] - 9.1).abs() < 1e-5);
    assert!((result[1] - 12.2).abs() < 1e-5);
    assert!((result[2] - 15.3).abs() < 1e-5);
}

#[test]
fn test_linear_reference_no_bias() {
    let input = vec![2.0, 3.0];
    let weight = vec![1.0, 0.0, 0.0, 1.0]; // identity 2x2
    let result = linear_reference(&input, &weight, None, 2, 2);
    assert!((result[0] - 2.0).abs() < 1e-6);
    assert!((result[1] - 3.0).abs() < 1e-6);
}

// =========================================================================
// Section 18: CUDA C++ Emission (ptx_emit)
// =========================================================================

#[test]
fn test_emit_activation_kernels_all_five() {
    let src = emit_activation_kernels();
    assert!(src.contains("relu_kernel"));
    assert!(src.contains("silu_kernel"));
    assert!(src.contains("sigmoid_kernel"));
    assert!(src.contains("tanh_kernel"));
    assert!(src.contains("gelu_kernel"));
    assert_eq!(src.matches("__global__").count(), 5);
}

#[test]
fn test_emit_elementwise_kernel_custom_op() {
    let src = emit_elementwise_kernel("abs_kernel", "x > 0.0f ? x : -x", 512).unwrap();
    assert!(src.contains("abs_kernel"));
    assert!(src.contains("__global__"));
    assert!(src.contains("x > 0.0f ? x : -x"));
}

#[test]
fn test_emit_elementwise_kernel_zero_rejected() {
    assert!(emit_elementwise_kernel("k", "x", 0).is_err());
}

#[test]
fn test_emit_softmax_kernel_structure() {
    let src = emit_softmax_kernel(128).unwrap();
    assert!(src.contains("softmax_kernel"));
    assert!(src.contains("__shared__"));
    assert!(src.contains("expf"));
}

#[test]
fn test_emit_reduction_kernel_all_ops() {
    for op in [
        ReductionOp::Sum,
        ReductionOp::Max,
        ReductionOp::Min,
        ReductionOp::Mean,
    ] {
        let src = emit_reduction_kernel("test_reduce", op, 256).unwrap();
        assert!(src.contains("test_reduce"));
        assert!(src.contains("__shared__"));
        assert!(src.contains("__syncthreads"));
    }
}

#[test]
fn test_emit_reduction_kernel_zero_rejected() {
    assert!(emit_reduction_kernel("k", ReductionOp::Sum, 0).is_err());
}

#[test]
fn test_emit_matmul_kernel_tiled() {
    let src = emit_matmul_kernel("tiled_gemm", 16).unwrap();
    assert!(src.contains("tiled_gemm"));
    assert!(src.contains("#define TILE_SIZE 16"));
    assert!(src.contains("__shared__"));
}

#[test]
fn test_emit_matmul_kernel_invalid_tile_rejected() {
    assert!(emit_matmul_kernel("k", 0).is_err());
    assert!(emit_matmul_kernel("k", 64).is_err());
}

// =========================================================================
// Section 19: Cross-cutting PTX structural invariants
// =========================================================================

#[test]
fn test_all_ptx_generators_produce_version_directive() {
    let kernels: Vec<String> = vec![
        generate_add_ptx(64),
        generate_sub_ptx(64),
        generate_mul_ptx(64),
        generate_sum_ptx(64),
        generate_max_ptx(64),
        generate_mean_ptx(64),
        generate_matmul_ptx(8, 8, 8),
        generate_layernorm_ptx(64),
        generate_rmsnorm_ptx(64, 1e-5),
        generate_softmax_ptx(false, 64),
        generate_log_softmax_ptx(64),
        generate_linear_ptx(32, 64),
    ];
    for (i, ptx) in kernels.iter().enumerate() {
        assert!(
            ptx.contains(".version"),
            "Kernel {i} missing .version directive"
        );
        assert!(
            ptx.contains(".target"),
            "Kernel {i} missing .target directive"
        );
        assert!(
            ptx.contains(".entry"),
            "Kernel {i} missing .entry directive"
        );
        assert!(ptx.contains("ret;"), "Kernel {i} missing ret instruction");
    }
}

#[test]
fn test_all_ptx_generators_have_address_size_64() {
    let kernels: Vec<String> = vec![
        generate_add_ptx(128),
        generate_sum_ptx(128),
        generate_matmul_ptx(16, 16, 16),
        generate_layernorm_ptx(128),
        generate_rmsnorm_ptx(128, 1e-5),
    ];
    for (i, ptx) in kernels.iter().enumerate() {
        assert!(
            ptx.contains(".address_size 64"),
            "Kernel {i} missing .address_size 64"
        );
    }
}

#[test]
fn test_all_ptx_generators_contain_param_directives() {
    let kernels: Vec<String> = vec![
        generate_add_ptx(64),
        generate_sum_ptx(64),
        generate_matmul_ptx(8, 8, 8),
        generate_layernorm_ptx(64),
    ];
    for (i, ptx) in kernels.iter().enumerate() {
        assert!(
            ptx.contains(".param"),
            "Kernel {i} missing .param directives"
        );
    }
}

#[test]
fn test_warp_size_constant() {
    assert_eq!(WARP_SIZE, 32);
}
