// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for direct NativeOp dispatch (DynTensor bridge elimination).
//!
//! Part of #3472.

use nn_dsl::ir::ScalarType;
use nn_dsl::NativeOpKind;

use super::{
    can_use_direct_dispatch, generate_fused_geglu_msl, generate_fused_mul_add_msl,
    generate_fused_siglu_msl, generate_silu_mul_msl, DirectDispatch, FusedGeGLUDirect,
    FusedMulAddDirect, FusedSiGLUDirect, SiluMulDirect,
};

// --- Unit tests (no GPU required) ---

#[test]
fn test_can_dispatch_direct_silu_mul() {
    let op = NativeOpKind::SiluMul {
        input_shape: vec![2, 4],
    };
    assert!(can_use_direct_dispatch(&op));
}

#[test]
fn test_can_dispatch_direct_not_available_for_lstm() {
    let op = NativeOpKind::LstmSequence {
        hidden_size: 256,
        input_shape: vec![10, 1, 640],
        h_shape: vec![1, 1, 256],
        reverse: false,
    };
    assert!(!can_use_direct_dispatch(&op));
}

#[test]
fn test_can_dispatch_direct_not_available_for_layer_norm() {
    let op = NativeOpKind::LayerNorm {
        eps: 1e-5,
        input_shape: vec![1, 10, 768],
        hidden_dim: 768,
    };
    assert!(!can_use_direct_dispatch(&op));
}

#[test]
fn test_silu_mul_direct_output_bytes_f32() {
    let direct = SiluMulDirect;
    assert_eq!(direct.output_bytes(1024, ScalarType::F32), 4096);
}

#[test]
fn test_silu_mul_direct_output_bytes_f16() {
    let direct = SiluMulDirect;
    assert_eq!(direct.output_bytes(1024, ScalarType::F16), 2048);
}

#[test]
fn test_silu_mul_direct_output_bytes_zero() {
    let direct = SiluMulDirect;
    assert_eq!(direct.output_bytes(0, ScalarType::F32), 0);
}

#[test]
fn test_silu_mul_direct_supports_f32() {
    let direct = SiluMulDirect;
    assert!(direct.supports_scalar_type(ScalarType::F32));
}

#[test]
fn test_silu_mul_direct_supports_f16() {
    let direct = SiluMulDirect;
    assert!(direct.supports_scalar_type(ScalarType::F16));
}

#[test]
fn test_silu_mul_direct_unsupported_bf16() {
    let direct = SiluMulDirect;
    // BF16 is not supported by the SiluMul direct path (Metal 'half' covers
    // F16; BF16 needs bfloat which is Apple Silicon M4+ only).
    assert!(!direct.supports_scalar_type(ScalarType::BF16));
}

#[test]
fn test_silu_mul_msl_f32_contains_kernel() {
    let msl = generate_silu_mul_msl("silu_mul_direct_float", ScalarType::F32);
    assert!(
        msl.contains("kernel void silu_mul_direct_float"),
        "MSL should contain the kernel entry point"
    );
    assert!(
        msl.contains("sigmoid"),
        "MSL should compute sigmoid"
    );
    assert!(
        msl.contains("silu_g * u"),
        "MSL should compute silu(gate) * up"
    );
    // F32 path should NOT have float() casts on loads.
    assert!(
        !msl.contains("float(gate[tid])"),
        "F32 path should not cast float inputs"
    );
}

#[test]
fn test_silu_mul_msl_f16_contains_upcast() {
    let msl = generate_silu_mul_msl("silu_mul_direct_half", ScalarType::F16);
    assert!(
        msl.contains("kernel void silu_mul_direct_half"),
        "MSL should contain the kernel entry point"
    );
    // F16 path should upcast to float for precision.
    assert!(
        msl.contains("float(gate[tid])"),
        "F16 path should cast half inputs to float"
    );
    assert!(
        msl.contains("half("),
        "F16 path should cast back to half for output"
    );
}

#[test]
fn test_silu_mul_msl_includes_metal_header() {
    let msl = generate_silu_mul_msl("test_kernel", ScalarType::F32);
    assert!(
        msl.contains("#include <metal_stdlib>"),
        "MSL should include metal stdlib"
    );
    assert!(
        msl.contains("using namespace metal"),
        "MSL should use metal namespace"
    );
}

#[test]
fn test_silu_mul_msl_has_bounds_check() {
    let msl = generate_silu_mul_msl("test_kernel", ScalarType::F32);
    assert!(
        msl.contains("if (tid >= count) return"),
        "MSL should have bounds check"
    );
}

// --- Fused kernel structure tests (#3537) ---

#[test]
fn test_silu_mul_msl_single_kernel_entry() {
    // The fused MSL must have exactly one kernel entry point — confirming
    // this is a single-dispatch kernel, not two separate dispatches.
    let msl = generate_silu_mul_msl("silu_mul_fused", ScalarType::F32);
    let kernel_count = msl.matches("kernel void").count();
    assert_eq!(
        kernel_count, 1,
        "fused MSL should have exactly 1 kernel entry (single dispatch), got {kernel_count}"
    );
}

#[test]
fn test_silu_mul_msl_reads_both_inputs() {
    // The fused kernel must read from both gate and up buffers.
    let msl = generate_silu_mul_msl("silu_mul_fused", ScalarType::F32);
    assert!(
        msl.contains("gate[tid]"),
        "fused MSL must read from gate buffer"
    );
    assert!(
        msl.contains("up[tid]"),
        "fused MSL must read from up buffer"
    );
}

#[test]
fn test_silu_mul_msl_computes_fused_formula() {
    // The fused kernel should compute sigmoid in-line and multiply in one step.
    let msl = generate_silu_mul_msl("silu_mul_fused", ScalarType::F32);
    assert!(
        msl.contains("1.0f / (1.0f + exp(-g))"),
        "fused MSL must compute sigmoid inline"
    );
    assert!(
        msl.contains("g * sigmoid_g"),
        "fused MSL must compute silu = gate * sigmoid"
    );
    assert!(
        msl.contains("silu_g * u"),
        "fused MSL must compute fused output = silu * up"
    );
}

#[test]
fn test_silu_mul_msl_two_input_buffers() {
    // The fused kernel must have exactly 2 input buffers (gate, up)
    // plus output buffer and count constant.
    let msl = generate_silu_mul_msl("silu_mul_fused", ScalarType::F32);
    assert!(
        msl.contains("[[buffer(0)]]") && msl.contains("[[buffer(1)]]"),
        "fused MSL must have buffer(0) for gate and buffer(1) for up"
    );
    assert!(
        msl.contains("[[buffer(2)]]"),
        "fused MSL must have buffer(2) for output"
    );
    assert!(
        msl.contains("[[buffer(3)]]"),
        "fused MSL must have buffer(3) for element count"
    );
}

// --- FusedMulAdd tests (#4252, #4431) ---

#[test]
fn test_can_dispatch_direct_fused_mul_add() {
    let op = NativeOpKind::FusedMulAdd {
        input_shape: vec![2, 4],
    };
    assert!(can_use_direct_dispatch(&op));
}

#[test]
fn test_fused_mul_add_direct_output_bytes() {
    let direct = FusedMulAddDirect;
    assert_eq!(direct.output_bytes(1024, ScalarType::F32), 4096);
    assert_eq!(direct.output_bytes(1024, ScalarType::F16), 2048);
    assert_eq!(direct.output_bytes(0, ScalarType::F32), 0);
}

#[test]
fn test_fused_mul_add_direct_supports_f32_f16() {
    let direct = FusedMulAddDirect;
    assert!(direct.supports_scalar_type(ScalarType::F32));
    assert!(direct.supports_scalar_type(ScalarType::F16));
    assert!(!direct.supports_scalar_type(ScalarType::BF16));
}

#[test]
fn test_fused_mul_add_msl_f32_structure() {
    let msl = generate_fused_mul_add_msl("fused_mul_add_direct_float", ScalarType::F32);
    assert!(
        msl.contains("kernel void fused_mul_add_direct_float"),
        "MSL should contain kernel entry point"
    );
    assert!(
        msl.contains("fma(va, vb, vc)"),
        "MSL should use hardware FMA"
    );
    let kernel_count = msl.matches("kernel void").count();
    assert_eq!(kernel_count, 1, "single-dispatch kernel expected");
}

#[test]
fn test_fused_mul_add_msl_three_input_buffers() {
    let msl = generate_fused_mul_add_msl("fma_test", ScalarType::F32);
    assert!(msl.contains("[[buffer(0)]]"), "buffer(0) for a");
    assert!(msl.contains("[[buffer(1)]]"), "buffer(1) for b");
    assert!(msl.contains("[[buffer(2)]]"), "buffer(2) for c");
    assert!(msl.contains("[[buffer(3)]]"), "buffer(3) for output");
    assert!(msl.contains("[[buffer(4)]]"), "buffer(4) for count");
}

#[test]
fn test_fused_mul_add_msl_f16_upcast() {
    let msl = generate_fused_mul_add_msl("fma_half", ScalarType::F16);
    assert!(
        msl.contains("float(a[tid])"),
        "F16 path should upcast a to float"
    );
    assert!(msl.contains("half("), "F16 path should cast back to half");
}

// --- FusedSiGLU tests (#4252, #4431) ---

#[test]
fn test_can_dispatch_direct_fused_siglu() {
    let op = NativeOpKind::FusedSiGLU {
        input_shape: vec![2, 4],
    };
    assert!(can_use_direct_dispatch(&op));
}

#[test]
fn test_fused_siglu_direct_output_bytes() {
    let direct = FusedSiGLUDirect;
    assert_eq!(direct.output_bytes(512, ScalarType::F32), 2048);
    assert_eq!(direct.output_bytes(512, ScalarType::F16), 1024);
}

#[test]
fn test_fused_siglu_direct_supports_f32_f16() {
    let direct = FusedSiGLUDirect;
    assert!(direct.supports_scalar_type(ScalarType::F32));
    assert!(direct.supports_scalar_type(ScalarType::F16));
    assert!(!direct.supports_scalar_type(ScalarType::BF16));
}

#[test]
fn test_fused_siglu_msl_f32_structure() {
    let msl = generate_fused_siglu_msl("fused_siglu_direct_float", ScalarType::F32);
    assert!(
        msl.contains("kernel void fused_siglu_direct_float"),
        "MSL should contain kernel entry point"
    );
    assert!(
        msl.contains("sigmoid_val"),
        "MSL should compute sigmoid"
    );
    assert!(
        msl.contains("val * sigmoid_val"),
        "MSL should compute x * sigmoid(x)"
    );
    let kernel_count = msl.matches("kernel void").count();
    assert_eq!(kernel_count, 1, "single-dispatch kernel expected");
}

#[test]
fn test_fused_siglu_msl_one_input_buffer() {
    let msl = generate_fused_siglu_msl("siglu_test", ScalarType::F32);
    assert!(msl.contains("[[buffer(0)]]"), "buffer(0) for x");
    assert!(msl.contains("[[buffer(1)]]"), "buffer(1) for output");
    assert!(msl.contains("[[buffer(2)]]"), "buffer(2) for count");
}

#[test]
fn test_fused_siglu_msl_f16_upcast() {
    let msl = generate_fused_siglu_msl("siglu_half", ScalarType::F16);
    assert!(
        msl.contains("float(x[tid])"),
        "F16 path should upcast x to float"
    );
    assert!(msl.contains("half("), "F16 path should cast back to half");
}

// --- FusedGeGLU tests (#4252, #4431) ---

#[test]
fn test_can_dispatch_direct_fused_geglu() {
    let op = NativeOpKind::FusedGeGLU {
        input_shape: vec![2, 4],
    };
    assert!(can_use_direct_dispatch(&op));
}

#[test]
fn test_fused_geglu_direct_output_bytes() {
    let direct = FusedGeGLUDirect;
    assert_eq!(direct.output_bytes(256, ScalarType::F32), 1024);
    assert_eq!(direct.output_bytes(256, ScalarType::F16), 512);
}

#[test]
fn test_fused_geglu_direct_supports_f32_f16() {
    let direct = FusedGeGLUDirect;
    assert!(direct.supports_scalar_type(ScalarType::F32));
    assert!(direct.supports_scalar_type(ScalarType::F16));
    assert!(!direct.supports_scalar_type(ScalarType::BF16));
}

#[test]
fn test_fused_geglu_msl_f32_structure() {
    let msl = generate_fused_geglu_msl("fused_geglu_direct_float", ScalarType::F32);
    assert!(
        msl.contains("kernel void fused_geglu_direct_float"),
        "MSL should contain kernel entry point"
    );
    // GELU fast approximation uses tanh.
    assert!(
        msl.contains("tanh(k *"),
        "MSL should compute fast GELU via tanh"
    );
    assert!(
        msl.contains("0.044715f"),
        "MSL should include GELU cubic coefficient"
    );
    assert!(
        msl.contains("gelu_g * u"),
        "MSL should compute gelu(gate) * up"
    );
    let kernel_count = msl.matches("kernel void").count();
    assert_eq!(kernel_count, 1, "single-dispatch kernel expected");
}

#[test]
fn test_fused_geglu_msl_two_input_buffers() {
    let msl = generate_fused_geglu_msl("geglu_test", ScalarType::F32);
    assert!(msl.contains("[[buffer(0)]]"), "buffer(0) for gate");
    assert!(msl.contains("[[buffer(1)]]"), "buffer(1) for up");
    assert!(msl.contains("[[buffer(2)]]"), "buffer(2) for output");
    assert!(msl.contains("[[buffer(3)]]"), "buffer(3) for count");
}

#[test]
fn test_fused_geglu_msl_f16_upcast() {
    let msl = generate_fused_geglu_msl("geglu_half", ScalarType::F16);
    assert!(
        msl.contains("float(gate[tid])"),
        "F16 path should upcast gate to float"
    );
    assert!(
        msl.contains("float(up[tid])"),
        "F16 path should upcast up to float"
    );
    assert!(msl.contains("half("), "F16 path should cast back to half");
}
