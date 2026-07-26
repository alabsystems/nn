// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for PTX activation kernel generation.
//!
//! Covers config validation, PTX structural checks, reference computation
//! verification, edge cases for all supported activations (GELU, SiLU, Mish,
//! Snake, GELU Fast).

use super::*;

// =========================================================================
// Config construction and validation
// =========================================================================

#[test]
fn test_config_basic() {
    let c = PtxActivationConfig::new("silu_kernel", PtxActivation::Silu);
    assert_eq!(c.kernel_name, "silu_kernel");
    assert_eq!(c.activation, PtxActivation::Silu);
    assert_eq!(c.sm_target, "sm_80");
    assert_eq!(c.block_size, 256);
    assert!(c.validate().is_ok());
}

#[test]
fn test_config_empty_name_rejected() {
    let c = PtxActivationConfig::new("", PtxActivation::Gelu);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_zero_block_size_rejected() {
    let c = PtxActivationConfig::new("act", PtxActivation::Gelu).with_block_size(0);
    assert!(c.validate().is_err());
}

#[test]
fn test_config_custom_sm_target() {
    let c = PtxActivationConfig::new("act", PtxActivation::Gelu).with_sm_target("sm_70");
    assert_eq!(c.sm_target, "sm_70");
}

#[test]
fn test_config_custom_block_size() {
    let c = PtxActivationConfig::new("act", PtxActivation::Gelu).with_block_size(128);
    assert_eq!(c.block_size, 128);
}

// =========================================================================
// PtxActivation enum
// =========================================================================

#[test]
fn test_activation_names() {
    assert_eq!(PtxActivation::Gelu.name(), "gelu");
    assert_eq!(PtxActivation::GeluFast.name(), "gelu_fast");
    assert_eq!(PtxActivation::Silu.name(), "silu");
    assert_eq!(PtxActivation::Mish.name(), "mish");
    assert_eq!(PtxActivation::Snake.name(), "snake");
}

#[test]
fn test_activation_requires_alpha() {
    assert!(!PtxActivation::Gelu.requires_alpha());
    assert!(!PtxActivation::GeluFast.requires_alpha());
    assert!(!PtxActivation::Silu.requires_alpha());
    assert!(!PtxActivation::Mish.requires_alpha());
    assert!(PtxActivation::Snake.requires_alpha());
}

// =========================================================================
// PTX structural validation -- SiLU
// =========================================================================

#[test]
fn test_silu_ptx_contains_version_and_target() {
    let ptx = emit_ptx_activation_default("silu_f32", PtxActivation::Silu).unwrap();
    assert!(ptx.contains(".version 6.5"));
    assert!(ptx.contains(".target sm_80"));
    assert!(ptx.contains(".address_size 64"));
}

#[test]
fn test_silu_ptx_contains_entry() {
    let ptx = emit_ptx_activation_default("silu_f32", PtxActivation::Silu).unwrap();
    assert!(ptx.contains(".visible .entry silu_f32"));
}

#[test]
fn test_silu_ptx_has_sigmoid_ops() {
    let ptx = emit_ptx_activation_default("silu_f32", PtxActivation::Silu).unwrap();
    assert!(ptx.contains("ex2.approx.f32"), "SiLU needs exp for sigmoid");
    assert!(
        ptx.contains("rcp.approx.f32"),
        "SiLU needs reciprocal for 1/(1+exp)"
    );
}

#[test]
fn test_silu_ptx_params() {
    let ptx = emit_ptx_activation_default("silu_f32", PtxActivation::Silu).unwrap();
    assert!(ptx.contains("param_input"));
    assert!(ptx.contains("param_output"));
    assert!(ptx.contains("param_n"));
    assert!(
        !ptx.contains("param_alpha"),
        "SiLU should not have alpha param"
    );
}

#[test]
fn test_silu_ptx_grid_stride_loop() {
    let ptx = emit_ptx_activation_default("silu_f32", PtxActivation::Silu).unwrap();
    assert!(ptx.contains("ACT_LOOP"));
    assert!(ptx.contains("ACT_EXIT"));
}

// =========================================================================
// PTX structural validation -- GELU
// =========================================================================

#[test]
fn test_gelu_ptx_contains_erf_approx() {
    let ptx = emit_ptx_activation_default("gelu_f32", PtxActivation::Gelu).unwrap();
    assert!(ptx.contains("erf"), "GELU should mention erf in comments");
    assert!(
        ptx.contains("fma.rn.f32"),
        "GELU uses Horner evaluation via fma"
    );
}

#[test]
fn test_gelu_ptx_entry() {
    let ptx = emit_ptx_activation_default("gelu_f32", PtxActivation::Gelu).unwrap();
    assert!(ptx.contains(".visible .entry gelu_f32"));
}

// =========================================================================
// PTX structural validation -- GELU Fast
// =========================================================================

#[test]
fn test_gelu_fast_ptx_entry() {
    let ptx = emit_ptx_activation_default("gelu_fast_f32", PtxActivation::GeluFast).unwrap();
    assert!(ptx.contains(".visible .entry gelu_fast_f32"));
    assert!(ptx.contains("GELU fast"));
}

#[test]
fn test_gelu_fast_ptx_has_sigmoid() {
    let ptx = emit_ptx_activation_default("gelu_fast_f32", PtxActivation::GeluFast).unwrap();
    assert!(ptx.contains("ex2.approx.f32"));
    assert!(ptx.contains("rcp.approx.f32"));
}

// =========================================================================
// PTX structural validation -- Mish
// =========================================================================

#[test]
fn test_mish_ptx_entry() {
    let ptx = emit_ptx_activation_default("mish_f32", PtxActivation::Mish).unwrap();
    assert!(ptx.contains(".visible .entry mish_f32"));
}

#[test]
fn test_mish_ptx_has_softplus_and_tanh() {
    let ptx = emit_ptx_activation_default("mish_f32", PtxActivation::Mish).unwrap();
    assert!(
        ptx.contains("lg2.approx.f32"),
        "Mish needs log for softplus"
    );
    assert!(
        ptx.contains("ex2.approx.f32"),
        "Mish needs exp for softplus/tanh"
    );
    assert!(
        ptx.contains("softplus"),
        "Mish comment should mention softplus"
    );
    assert!(ptx.contains("tanh"), "Mish comment should mention tanh");
}

// =========================================================================
// PTX structural validation -- Snake
// =========================================================================

#[test]
fn test_snake_ptx_entry() {
    let ptx = emit_ptx_activation_default("snake_f32", PtxActivation::Snake).unwrap();
    assert!(ptx.contains(".visible .entry snake_f32"));
}

#[test]
fn test_snake_ptx_has_alpha_param() {
    let ptx = emit_ptx_activation_default("snake_f32", PtxActivation::Snake).unwrap();
    assert!(
        ptx.contains("param_alpha"),
        "Snake requires alpha parameter"
    );
}

#[test]
fn test_snake_ptx_has_sin() {
    let ptx = emit_ptx_activation_default("snake_f32", PtxActivation::Snake).unwrap();
    assert!(ptx.contains("sin.approx.f32"), "Snake needs sin");
}

#[test]
fn test_snake_ptx_has_reciprocal() {
    let ptx = emit_ptx_activation_default("snake_f32", PtxActivation::Snake).unwrap();
    assert!(ptx.contains("rcp.approx.f32"), "Snake needs 1/alpha");
}

// =========================================================================
// PTX not CUDA C++
// =========================================================================

#[test]
fn test_ptx_not_cuda_cpp() {
    for act in [
        PtxActivation::Gelu,
        PtxActivation::Silu,
        PtxActivation::Mish,
        PtxActivation::Snake,
    ] {
        let ptx = emit_ptx_activation_default(&format!("{}_f32", act.name()), act).unwrap();
        assert!(
            !ptx.contains("__global__"),
            "{} should be PTX, not CUDA C++",
            act.name()
        );
    }
}

// =========================================================================
// Generate all activations
// =========================================================================

#[test]
fn test_generate_all_activation_ptx() {
    let all = generate_all_activation_ptx();
    assert_eq!(all.len(), 5);

    let names: Vec<&str> = all.iter().map(|(n, _)| *n).collect();
    assert!(names.contains(&"gelu"));
    assert!(names.contains(&"gelu_fast"));
    assert!(names.contains(&"silu"));
    assert!(names.contains(&"mish"));
    assert!(names.contains(&"snake"));

    // All should be valid PTX with version and entry
    for (name, ptx) in &all {
        assert!(ptx.contains(".version 6.5"), "{name} missing PTX version");
        assert!(ptx.contains(".entry"), "{name} missing entry point");
    }
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_launch_config_basic() {
    let (grid, block) = ptx_activation_launch_config(1024, 256);
    assert_eq!(grid, [4, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_non_aligned() {
    let (grid, block) = ptx_activation_launch_config(1000, 256);
    assert_eq!(grid, [4, 1, 1]); // ceil(1000/256) = 4
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_single_element() {
    let (grid, block) = ptx_activation_launch_config(1, 256);
    assert_eq!(grid, [1, 1, 1]);
    assert_eq!(block, [256, 1, 1]);
}

#[test]
fn test_launch_config_zero_block_uses_default() {
    let (grid, block) = ptx_activation_launch_config(512, 0);
    assert_eq!(grid, [2, 1, 1]); // ceil(512/256) = 2
    assert_eq!(block, [256, 1, 1]);
}

// =========================================================================
// Reference implementation: SiLU
// =========================================================================

#[test]
fn test_silu_reference_zero() {
    assert!((silu_reference(0.0) - 0.0).abs() < 1e-6);
}

#[test]
fn test_silu_reference_positive() {
    // SiLU(1) = 1 * sigmoid(1) ~= 0.7311
    let result = silu_reference(1.0);
    assert!((result - 0.7311).abs() < 0.001, "got {result}");
}

#[test]
fn test_silu_reference_negative() {
    // SiLU(-1) = -1 * sigmoid(-1) ~= -0.2689
    let result = silu_reference(-1.0);
    assert!((result - (-0.2689)).abs() < 0.001, "got {result}");
}

#[test]
fn test_silu_reference_large_positive() {
    // sigmoid(10) ~= 1, so SiLU(10) ~= 10
    let result = silu_reference(10.0);
    assert!((result - 10.0).abs() < 0.01, "got {result}");
}

#[test]
fn test_silu_reference_large_negative() {
    // sigmoid(-10) ~= 0, so SiLU(-10) ~= 0
    let result = silu_reference(-10.0);
    assert!(result.abs() < 0.01, "got {result}");
}

// =========================================================================
// Reference implementation: GELU
// =========================================================================

#[test]
fn test_gelu_reference_zero() {
    assert!(gelu_reference(0.0).abs() < 1e-6);
}

#[test]
fn test_gelu_reference_positive() {
    // GELU(1) ~= 0.8413
    let result = gelu_reference(1.0);
    assert!((result - 0.8413).abs() < 0.01, "got {result}");
}

#[test]
fn test_gelu_reference_negative() {
    // GELU(-1) ~= -0.1587
    let result = gelu_reference(-1.0);
    assert!((result - (-0.1587)).abs() < 0.01, "got {result}");
}

#[test]
fn test_gelu_reference_large_positive() {
    // GELU(5) ~= 5 (erf(5/sqrt(2)) ~= 1)
    let result = gelu_reference(5.0);
    assert!((result - 5.0).abs() < 0.01, "got {result}");
}

// =========================================================================
// Reference implementation: GELU Fast
// =========================================================================

#[test]
fn test_gelu_fast_reference_zero() {
    assert!(gelu_fast_reference(0.0).abs() < 1e-6);
}

#[test]
fn test_gelu_fast_reference_positive() {
    let result = gelu_fast_reference(1.0);
    // Should be close to exact GELU
    let exact = gelu_reference(1.0);
    assert!((result - exact).abs() < 0.1, "fast={result}, exact={exact}");
}

// =========================================================================
// Reference implementation: Mish
// =========================================================================

#[test]
fn test_mish_reference_zero() {
    // Mish(0) = 0 * tanh(ln(2)) = 0
    assert!(mish_reference(0.0).abs() < 1e-6);
}

#[test]
fn test_mish_reference_positive() {
    // Mish(1) = 1 * tanh(softplus(1)) = tanh(ln(1+e)) ~= 0.8651
    let result = mish_reference(1.0);
    assert!((result - 0.8651).abs() < 0.01, "got {result}");
}

#[test]
fn test_mish_reference_negative() {
    // Mish(-1) ~= -0.3034
    let result = mish_reference(-1.0);
    assert!((result - (-0.3034)).abs() < 0.01, "got {result}");
}

#[test]
fn test_mish_reference_large_positive() {
    // Mish(10) ~= 10 (tanh(softplus(10)) ~= 1)
    let result = mish_reference(10.0);
    assert!((result - 10.0).abs() < 0.01, "got {result}");
}

// =========================================================================
// Reference implementation: Snake
// =========================================================================

#[test]
fn test_snake_reference_zero() {
    // Snake(0, alpha) = 0 + (1/alpha) * sin(0)^2 = 0
    assert!(snake_reference(0.0, 1.0).abs() < 1e-6);
}

#[test]
fn test_snake_reference_identity_component() {
    // At multiples of pi/alpha, sin(alpha*x) = 0, so Snake(x, alpha) = x
    let alpha = 1.0;
    let x = std::f32::consts::PI;
    let result = snake_reference(x, alpha);
    assert!((result - x).abs() < 1e-5, "got {result}, expected {x}");
}

#[test]
fn test_snake_reference_always_geq_x() {
    // Snake(x, alpha) = x + non_negative_term >= x
    for x_int in -10..=10 {
        let x = x_int as f32 * 0.5;
        let result = snake_reference(x, 1.0);
        assert!(
            result >= x - 1e-6,
            "Snake({x}, 1.0) = {result} should be >= {x}"
        );
    }
}

#[test]
fn test_snake_reference_different_alphas() {
    let x = 1.0;
    let s1 = snake_reference(x, 0.5);
    let s2 = snake_reference(x, 2.0);
    // Different alphas should give different results
    assert!((s1 - s2).abs() > 0.01, "alpha=0.5: {s1}, alpha=2.0: {s2}");
}

// =========================================================================
// PTX size sanity
// =========================================================================

#[test]
fn test_all_activations_ptx_reasonable_size() {
    for act in [
        PtxActivation::Gelu,
        PtxActivation::GeluFast,
        PtxActivation::Silu,
        PtxActivation::Mish,
        PtxActivation::Snake,
    ] {
        let ptx = emit_ptx_activation_default(&format!("{}_f32", act.name()), act).unwrap();
        assert!(
            ptx.len() > 200,
            "{} PTX too small: {} bytes",
            act.name(),
            ptx.len()
        );
        assert!(
            ptx.len() < 20_000,
            "{} PTX too large: {} bytes",
            act.name(),
            ptx.len()
        );
    }
}

// =========================================================================
// Config Clone/Debug/Eq
// =========================================================================

#[test]
fn test_config_clone() {
    let c = PtxActivationConfig::new("act", PtxActivation::Gelu);
    let c2 = c.clone();
    assert_eq!(c.kernel_name, c2.kernel_name);
    assert_eq!(c.activation, c2.activation);
}

#[test]
fn test_config_debug() {
    let c = PtxActivationConfig::new("act", PtxActivation::Silu);
    let debug = format!("{c:?}");
    assert!(debug.contains("PtxActivationConfig"));
    assert!(debug.contains("Silu"));
}

#[test]
fn test_activation_eq() {
    assert_eq!(PtxActivation::Gelu, PtxActivation::Gelu);
    assert_ne!(PtxActivation::Gelu, PtxActivation::Silu);
}
