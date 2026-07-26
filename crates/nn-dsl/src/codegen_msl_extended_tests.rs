// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended MSL codegen tests for Metal compute shader generation.
//!
//! Covers: elementwise activations (ReLU, GELU, SiLU, sigmoid, tanh),
//! reduction MSL, matmul/simdgroup dispatch, softmax two-pass, conv1d/conv2d,
//! MSL header validation, push constant handling, and workgroup size config.
//!
//! Part of #4186.

use crate::codegen_msl::{emit_msl, MSL_PRELUDE};
use crate::codegen_msl_tensor::{build_dispatch_plan, DispatchStep, REDUCE_THREADGROUP_SIZE};
use crate::codegen_msl_tensor_emit::{emit_reduce_kernel, emit_tensor_msl};
use crate::ir::ScalarType;
use crate::precision::{PrecisionContract, PrecisionTier};
use crate::tensor_ir::{ReduceOp, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};
use crate::test_kernels::parse_kernel;

// ===========================================================================
// 1. Elementwise MSL generation — ReLU, GELU, SiLU, sigmoid, tanh
// ===========================================================================

#[test]
fn test_relu_elementwise_msl_signature() {
    let kernel = parse_kernel("fn relu(x: f32) -> f32 { x.max(0.0) }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(
        msl.contains("relu_kernel"),
        "entry point must be relu_kernel, got:\n{msl}"
    );
    assert!(
        msl.contains("float _nn_relu(float x)"),
        "scalar helper must have correct signature, got:\n{msl}"
    );
    assert!(msl.contains("max("), "ReLU must emit max()");
}

#[test]
fn test_gelu_elementwise_msl_tanh_approx() {
    let def = TensorKernelDef::new(
        "gelu_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![4, 16],
                },
                vec![4, 16],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Gelu {
                    input: TensorNodeId::new(0),
                },
                vec![4, 16],
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("GELU MSL emission");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(
        msl.contains("0.044715"),
        "GELU tanh approximation must include 0.044715 constant"
    );
    // The tanh approximation is emitted via its mathematically-equivalent
    // exp-based form (`0.5 * x * (2 - 2 / (exp(2*inner) + 1))`), not a literal
    // `tanh()` call, so it bit-matches the scalar `gelu.rs` reference (see
    // emit_gelu_kernel docs, #679). It is characterized by the sqrt(2/pi)
    // coefficient 0.7978845608028654 and an exp() call (the erf form has
    // neither).
    assert!(
        msl.contains("0.7978845608028654"),
        "GELU tanh approximation must include the sqrt(2/pi) coefficient"
    );
    assert!(
        msl.contains("metal::precise::exp") || msl.contains("exp("),
        "GELU tanh approximation must use the exp-based tanh-equivalent form"
    );
}

#[test]
fn test_silu_elementwise_msl_sigmoid_mul() {
    // SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    // Build via Sigmoid dispatch then multiply — verify sigmoid kernel is emitted
    let def = TensorKernelDef::new(
        "silu_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 8],
                },
                vec![2, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Sigmoid {
                    input: TensorNodeId::new(0),
                },
                vec![2, 8],
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Sigmoid MSL");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(
        msl.contains("metal::precise::exp"),
        "sigmoid must use exp, got:\n{msl}"
    );
    assert!(
        msl.contains("device const float*"),
        "f32 sigmoid must use float buffers"
    );
}

#[test]
fn test_sigmoid_elementwise_dispatch_step() {
    let def = TensorKernelDef::new(
        "sig_dispatch",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![3, 8],
                },
                vec![3, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Sigmoid {
                    input: TensorNodeId::new(0),
                },
                vec![3, 8],
            ),
        ],
        TensorNodeId::new(1),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(
            &plan[0],
            DispatchStep::Sigmoid {
                total_elements: 24,
                ..
            }
        ),
        "expected Sigmoid step with 24 elements, got {:?}",
        &plan[0]
    );
}

#[test]
fn test_tanh_elementwise_msl() {
    let def = TensorKernelDef::new(
        "tanh_test",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Tanh {
                    input: TensorNodeId::new(0),
                },
                vec![4, 8],
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Tanh MSL emission");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(
        msl.contains("metal::precise::tanh"),
        "tanh must use metal::precise::tanh, got:\n{msl}"
    );
}

#[test]
fn test_relu_dispatch_plan() {
    let def = TensorKernelDef::new(
        "relu_dp",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![2, 4],
                },
                vec![2, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Relu {
                    input: TensorNodeId::new(0),
                },
                vec![2, 4],
            ),
        ],
        TensorNodeId::new(1),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(
            &plan[0],
            DispatchStep::Relu {
                total_elements: 8,
                ..
            }
        ),
        "expected Relu step with 8 elements, got {:?}",
        &plan[0]
    );
}

// ===========================================================================
// 2. Reduction MSL generation — threadgroup-based reduction
// ===========================================================================

#[test]
fn test_reduce_sum_threadgroup_structure() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let msl = emit_reduce_kernel("reduce_sum_tg", ReduceOp::Sum, ScalarType::F32, contract);
    // Must have threadgroup shared memory declaration
    let expected_shared = format!("threadgroup float shared[{REDUCE_THREADGROUP_SIZE}]");
    assert!(
        msl.contains(&expected_shared),
        "must declare shared memory of size {REDUCE_THREADGROUP_SIZE}, got:\n{msl}"
    );
    // Must have tree reduction phase
    assert!(
        msl.contains("stride >>= 1"),
        "must have tree reduction loop"
    );
    assert!(
        msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "must have threadgroup barrier"
    );
}

#[test]
fn test_reduce_max_uses_fmax() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let msl = emit_reduce_kernel("reduce_max", ReduceOp::Max, ScalarType::F32, contract);
    assert!(
        msl.contains("fmax"),
        "Max reduce must use fmax, got:\n{msl}"
    );
    assert!(
        msl.contains("-INFINITY"),
        "Max reduce identity must be -INFINITY, got:\n{msl}"
    );
}

#[test]
fn test_reduce_min_uses_fmin() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let msl = emit_reduce_kernel("reduce_min", ReduceOp::Min, ScalarType::F32, contract);
    assert!(
        msl.contains("fmin"),
        "Min reduce must use fmin, got:\n{msl}"
    );
    assert!(
        msl.contains("INFINITY"),
        "Min reduce identity must be INFINITY, got:\n{msl}"
    );
}

#[test]
fn test_reduce_mean_divides_by_reduce_dim() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let msl = emit_reduce_kernel("reduce_mean", ReduceOp::Mean, ScalarType::F32, contract);
    assert!(
        msl.contains("/ float(reduce_dim)"),
        "Mean reduce must divide by reduce_dim, got:\n{msl}"
    );
}

#[test]
fn test_reduce_via_tensor_def_produces_threadgroup_code() {
    let out_shape = vec![4, 32];
    let def = TensorKernelDef::new(
        "tg_reduce",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![4, 32, 64],
                },
                vec![4, 32, 64],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Sum,
                    input: TensorNodeId::new(0),
                    axis: 2,
                    keepdim: false,
                },
                out_shape,
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("reduction MSL");
    assert!(
        msl.contains("threadgroup float shared"),
        "must contain threadgroup shared memory"
    );
    assert!(msl.contains("threadgroup_barrier"), "must contain barrier");
}

// ===========================================================================
// 3. MatMul MSL generation — simdgroup_matrix for supported sizes
// ===========================================================================

#[test]
fn test_matmul_dispatch_plan_small_naive() {
    // Small matmul (4x4 @ 4x4) should use naive MatMul, not simdgroup
    let def = TensorKernelDef::new(
        "mm_small",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![4, 4],
                },
                vec![4, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![4, 4],
                },
                vec![4, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right: false,
                    scale: None,
                },
                vec![4, 4],
            ),
        ],
        TensorNodeId::new(2),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("matmul plan");
    assert_eq!(plan.len(), 1, "should have 1 dispatch step");
    // Small matmul uses naive or tiled, not simdgroup (M*N < 16384 or K < 128)
    assert!(
        !matches!(&plan[0], DispatchStep::SimdgroupMatMul(..)),
        "4x4 matmul should NOT use simdgroup (too small)"
    );
}

#[test]
fn test_matmul_simdgroup_eligible_shape() {
    // M=128, K=128, N=128: all % 8 == 0, M*N = 16384, K >= 128 -> simdgroup
    let def = TensorKernelDef::new(
        "mm_simd",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![128, 128],
                },
                vec![128, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![128, 128],
                },
                vec![128, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right: false,
                    scale: None,
                },
                vec![128, 128],
            ),
        ],
        TensorNodeId::new(2),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("simdgroup matmul plan");
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(&plan[0], DispatchStep::SimdgroupMatMul(..)),
        "128x128 matmul should use simdgroup, got {:?}",
        &plan[0]
    );
}

#[test]
fn test_matmul_simdgroup_msl_contains_simdgroup_matrix() {
    let def = TensorKernelDef::new(
        "mm_simd_msl",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![128, 128],
                },
                vec![128, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![128, 128],
                },
                vec![128, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right: false,
                    scale: None,
                },
                vec![128, 128],
            ),
        ],
        TensorNodeId::new(2),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("simdgroup matmul MSL");
    assert!(
        msl.contains("#include <metal_simdgroup_matrix>"),
        "simdgroup matmul must include metal_simdgroup_matrix, got:\n{msl}"
    );
    assert!(
        msl.contains("simdgroup_matrix"),
        "simdgroup matmul must use simdgroup_matrix type, got:\n{msl}"
    );
}

#[test]
fn test_matmul_with_scale_dispatch() {
    // Matmul with scale (attention pattern: Q@K^T / sqrt(d_k))
    let def = TensorKernelDef::new(
        "mm_scale",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "q".into(),
                    shape: vec![8, 16],
                },
                vec![8, 16],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "k".into(),
                    shape: vec![16, 8],
                },
                vec![16, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right: false,
                    scale: Some(0.25),
                },
                vec![8, 8],
            ),
        ],
        TensorNodeId::new(2),
    );
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("scaled matmul plan");
    assert_eq!(plan.len(), 1);
    // Verify scale is propagated to the dispatch step
    match &plan[0] {
        DispatchStep::MatMul { scale, m, k, n, .. } => {
            assert_eq!(scale, &Some(0.25), "scale must propagate");
            assert_eq!(*m, 8);
            assert_eq!(*k, 16);
            assert_eq!(*n, 8);
        }
        DispatchStep::TiledMatMul(p) => {
            assert_eq!(p.scale, Some(0.25), "scale must propagate");
        }
        other => panic!("expected MatMul or TiledMatMul step, got: {other:?}"),
    }
}

// ===========================================================================
// 4. Softmax MSL generation — two-pass online softmax
// ===========================================================================

#[test]
fn test_softmax_two_pass_structure() {
    use crate::softmax::build_softmax;
    let def = build_softmax("sm_twopass", &[8, 32], -1).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Softmax MSL");
    // Phase 1: find max
    assert!(
        msl.contains("shared_max"),
        "softmax must have shared_max for max reduction, got:\n{msl}"
    );
    // Phase 2: compute exp and sum
    assert!(
        msl.contains("shared_sum"),
        "softmax must have shared_sum for sum reduction, got:\n{msl}"
    );
    assert!(
        msl.contains("metal::precise::exp"),
        "softmax must use precise exp, got:\n{msl}"
    );
    // Phase 3: normalize
    assert!(
        msl.contains("/ sum_val"),
        "softmax must normalize by sum_val, got:\n{msl}"
    );
}

#[test]
fn test_softmax_dispatch_step_shape() {
    use crate::softmax::build_softmax;
    let def = build_softmax("sm_shape", &[2, 4, 16], -1).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(
            &plan[0],
            DispatchStep::Softmax {
                axis_size: 16,
                outer_size: 8,
                ..
            }
        ),
        "expected axis_size=16, outer_size=2*4=8, got {:?}",
        &plan[0]
    );
}

#[test]
fn test_softmax_f16_uses_float_accumulator() {
    use crate::softmax::build_softmax;
    let def = build_softmax("sm_f16", &[4, 8], -1).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F16).expect("F16 softmax MSL");
    // Shared memory must use float accumulator for precision
    assert!(
        msl.contains("threadgroup float"),
        "F16 softmax shared memory must use float accumulator, got:\n{msl}"
    );
    assert!(
        msl.contains("device const half*"),
        "F16 softmax buffers must use half, got:\n{msl}"
    );
}

// ===========================================================================
// 5. Conv1d/Conv2d MSL generation
// ===========================================================================

#[test]
fn test_conv1d_msl_padding_stride() {
    use crate::conv1d::build_conv1d;
    // stride=2, padding=1
    let def = build_conv1d("conv_ps", 4, 8, 3, 16, 2, 1, false).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Conv1d MSL");
    assert!(msl.contains("STRIDE = 2"), "stride must be baked in");
    assert!(msl.contains("PADDING = 1"), "padding must be baked in");
    assert!(
        msl.contains("IN_CH_PER_GROUP = 4"),
        "in_channels must be baked in"
    );
    assert!(
        msl.contains("OUT_CHANNELS = 8"),
        "out_channels must be baked in"
    );
    assert!(
        msl.contains("KERNEL_SIZE = 3"),
        "kernel_size must be baked in"
    );
}

#[test]
fn test_conv1d_dispatch_params_correct() {
    use crate::conv1d::build_conv1d;
    // in_ch=2, out_ch=4, kernel=3, in_len=10, stride=1, pad=0
    let def = build_conv1d("conv_dp", 2, 4, 3, 10, 1, 0, false).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("plan");
    assert_eq!(plan.len(), 1);
    match &plan[0] {
        DispatchStep::Conv1d(p) => {
            assert_eq!(p.in_channels, 2);
            assert_eq!(p.out_channels, 4);
            assert_eq!(p.kernel_size, 3);
            assert_eq!(p.stride, 1);
            assert_eq!(p.padding, 0);
            // out_len = (10 - 3) / 1 + 1 = 8; total = 4 * 8 = 32
            assert_eq!(p.total_elements, 32);
        }
        other => panic!("expected Conv1d step, got: {other:?}"),
    }
}

#[test]
fn test_conv2d_dispatch_params() {
    use crate::conv2d::build_conv2d;
    let def = build_conv2d("conv2d_dp", 3, 16, 3, 3, 32, 32, 1, 1, 1, 1, false).expect("build");
    let (plan, _) = build_dispatch_plan(&def, ScalarType::F32).expect("Conv2d plan");
    assert_eq!(plan.len(), 1);
    match &plan[0] {
        DispatchStep::Conv2d(p) => {
            assert_eq!(p.in_channels, 3);
            assert_eq!(p.out_channels, 16);
            assert_eq!(p.kernel_h, 3);
            assert_eq!(p.kernel_w, 3);
            assert_eq!(p.stride_h, 1);
            assert_eq!(p.stride_w, 1);
            assert_eq!(p.padding_h, 1);
            assert_eq!(p.padding_w, 1);
            // out_h = (32 + 2*1 - 3) / 1 + 1 = 32
            // out_w = (32 + 2*1 - 3) / 1 + 1 = 32
            // total = 16 * 32 * 32 = 16384
            assert_eq!(p.total_elements, 16384);
        }
        other => panic!("expected Conv2d step, got: {other:?}"),
    }
}

#[test]
fn test_conv2d_msl_emission() {
    use crate::conv2d::build_conv2d;
    let def = build_conv2d("conv2d_em", 1, 4, 3, 3, 8, 8, 1, 1, 0, 0, true).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Conv2d MSL");
    assert!(msl.contains("[[kernel]]"), "must have kernel attribute");
    assert!(
        msl.contains("conv2d_em_conv2d"),
        "kernel name must contain conv2d"
    );
    assert!(msl.contains("bias"), "must reference bias buffer");
}

#[test]
fn test_conv2d_stride_2_msl() {
    use crate::conv2d::build_conv2d;
    let def = build_conv2d("conv2d_s2", 3, 8, 3, 3, 16, 16, 2, 2, 1, 1, false).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("Conv2d MSL");
    assert!(msl.contains("STRIDE_H = 2"), "stride_h must be baked in");
    assert!(msl.contains("STRIDE_W = 2"), "stride_w must be baked in");
}

// ===========================================================================
// 6. MSL header validation — all generated MSL starts with #include
// ===========================================================================

#[test]
fn test_scalar_msl_starts_with_include() {
    let kernel = parse_kernel("fn add(x: f32, y: f32) -> f32 { x + y }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "scalar MSL must start with #include <metal_stdlib>, got:\n{}",
        &msl[..msl.len().min(80)]
    );
}

#[test]
fn test_tensor_reduce_msl_starts_with_include() {
    let def = TensorKernelDef::new(
        "hdr_reduce",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![8, 64],
                },
                vec![8, 64],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Reduce {
                    op: ReduceOp::Sum,
                    input: TensorNodeId::new(0),
                    axis: 1,
                    keepdim: false,
                },
                vec![8],
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL");
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "tensor MSL must start with #include <metal_stdlib>"
    );
}

#[test]
fn test_tensor_activation_msl_starts_with_include() {
    let def = TensorKernelDef::new(
        "hdr_act",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "x".into(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Relu {
                    input: TensorNodeId::new(0),
                },
                vec![4, 8],
            ),
        ],
        TensorNodeId::new(1),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL");
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "tensor activation MSL must start with #include <metal_stdlib>"
    );
}

#[test]
fn test_softmax_msl_starts_with_include() {
    use crate::softmax::build_softmax;
    let def = build_softmax("hdr_sm", &[4, 8], -1).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL");
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "softmax MSL must start with #include <metal_stdlib>"
    );
}

#[test]
fn test_conv1d_msl_starts_with_include() {
    use crate::conv1d::build_conv1d;
    let def = build_conv1d("hdr_conv", 2, 4, 3, 8, 1, 0, false).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL");
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "conv1d MSL must start with #include <metal_stdlib>"
    );
}

#[test]
fn test_matmul_msl_starts_with_include() {
    let def = TensorKernelDef::new(
        "hdr_mm",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![4, 8],
                },
                vec![4, 8],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![8, 4],
                },
                vec![8, 4],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right: false,
                    scale: None,
                },
                vec![4, 4],
            ),
        ],
        TensorNodeId::new(2),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL");
    assert!(
        msl.starts_with("#include <metal_stdlib>"),
        "matmul MSL must start with #include <metal_stdlib>"
    );
}

#[test]
fn test_msl_prelude_constant_has_metal_stdlib() {
    assert!(
        MSL_PRELUDE.contains("#include <metal_stdlib>"),
        "MSL_PRELUDE must contain metal_stdlib include"
    );
    assert!(
        MSL_PRELUDE.contains("using namespace metal;"),
        "MSL_PRELUDE must contain using namespace metal"
    );
}

// ===========================================================================
// 7. Push constant handling — buffer bindings and push constant struct
// ===========================================================================

#[test]
fn test_reduce_buffer_bindings() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let msl = emit_reduce_kernel("buf_test", ReduceOp::Sum, ScalarType::F32, contract);
    // Reduce kernel uses explicit buffer bindings:
    // buffer(0) = input, buffer(1) = output, buffer(2) = reduce_dim, buffer(3) = outer_size
    assert!(msl.contains("[[buffer(0)]]"), "input must be at buffer(0)");
    assert!(msl.contains("[[buffer(1)]]"), "output must be at buffer(1)");
    assert!(
        msl.contains("[[buffer(2)]]"),
        "reduce_dim must be at buffer(2)"
    );
    assert!(
        msl.contains("[[buffer(3)]]"),
        "outer_size must be at buffer(3)"
    );
}

#[test]
fn test_reduce_push_constants_type() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let msl = emit_reduce_kernel("push_const", ReduceOp::Mean, ScalarType::F32, contract);
    // Push constants (reduce_dim, outer_size) are passed as `constant uint&`
    assert!(
        msl.contains("constant uint& reduce_dim"),
        "reduce_dim must be constant uint&, got:\n{msl}"
    );
    assert!(
        msl.contains("constant uint& outer_size"),
        "outer_size must be constant uint&, got:\n{msl}"
    );
}

#[test]
fn test_scalar_kernel_buffer_bindings() {
    // Scalar kernel with 2 params: buffer(0)=x, buffer(1)=y, buffer(2)=out, buffer(3)=N
    let kernel = parse_kernel("fn add(x: f32, y: f32) -> f32 { x + y }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("[[buffer(0)]]"),
        "first param must be at buffer(0)"
    );
    assert!(
        msl.contains("[[buffer(1)]]"),
        "second param must be at buffer(1)"
    );
    assert!(msl.contains("[[buffer(2)]]"), "output must be at buffer(2)");
    assert!(
        msl.contains("[[buffer(3)]]"),
        "element count must be at buffer(3)"
    );
}

#[test]
fn test_conv1d_buffer_bindings() {
    use crate::conv1d::build_conv1d;
    let def = build_conv1d("conv_buf", 2, 4, 3, 8, 1, 0, true).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL");
    // Conv1d with bias: buffer(0)=input, buffer(1)=weight, buffer(2)=bias, buffer(3)=output
    assert!(msl.contains("buffer(0)"), "input must be bound");
    assert!(msl.contains("buffer(1)"), "weight must be bound");
    assert!(msl.contains("buffer(2)"), "bias must be bound");
    assert!(msl.contains("buffer(3)"), "output must be bound");
}

// ===========================================================================
// 8. Workgroup size configuration — threadgroup_size declarations
// ===========================================================================

#[test]
fn test_reduce_threadgroup_size_is_256() {
    // Verify the constant value used for reduction threadgroup sizing
    assert_eq!(
        REDUCE_THREADGROUP_SIZE, 256,
        "reduction threadgroup size must be 256"
    );
}

#[test]
fn test_reduce_kernel_uses_threadgroup_attributes() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let msl = emit_reduce_kernel("tg_attr", ReduceOp::Sum, ScalarType::F32, contract);
    // Reduction kernel must use threadgroup position attributes
    assert!(
        msl.contains("[[threadgroup_position_in_grid]]"),
        "must use threadgroup_position_in_grid"
    );
    assert!(
        msl.contains("[[thread_position_in_threadgroup]]"),
        "must use thread_position_in_threadgroup"
    );
    assert!(
        msl.contains("[[threads_per_threadgroup]]"),
        "must use threads_per_threadgroup"
    );
}

#[test]
fn test_elementwise_kernel_uses_thread_position() {
    let kernel = parse_kernel("fn double(x: f32) -> f32 { x + x }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("[[thread_position_in_grid]]"),
        "elementwise kernel must use thread_position_in_grid, got:\n{msl}"
    );
}

#[test]
fn test_softmax_uses_threadgroup_attributes() {
    use crate::softmax::build_softmax;
    let def = build_softmax("sm_tg", &[4, 8], -1).expect("build");
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL");
    assert!(
        msl.contains("[[threadgroup_position_in_grid]]"),
        "softmax must use threadgroup_position_in_grid"
    );
    assert!(
        msl.contains("[[thread_position_in_threadgroup]]"),
        "softmax must use thread_position_in_threadgroup"
    );
}

#[test]
fn test_reduce_threadgroup_size_power_of_two() {
    // Compile-time assertion exists in the source, but verify at test time too
    assert!(
        REDUCE_THREADGROUP_SIZE.is_power_of_two(),
        "threadgroup size must be power of two for tree reduction"
    );
}

#[test]
fn test_simdgroup_matmul_msl_threadgroup_config() {
    // Simdgroup matmul uses 128 threads per threadgroup (4 simdgroups of 32)
    let def = TensorKernelDef::new(
        "mm_tg_cfg",
        vec![
            TensorNode::new(
                TensorNodeId::new(0),
                TensorOpKind::Input {
                    name: "a".into(),
                    shape: vec![128, 128],
                },
                vec![128, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(1),
                TensorOpKind::Input {
                    name: "b".into(),
                    shape: vec![128, 128],
                },
                vec![128, 128],
            ),
            TensorNode::new(
                TensorNodeId::new(2),
                TensorOpKind::MatMul {
                    left: TensorNodeId::new(0),
                    right: TensorNodeId::new(1),
                    transpose_right: false,
                    scale: None,
                },
                vec![128, 128],
            ),
        ],
        TensorNodeId::new(2),
    );
    let msl = emit_tensor_msl(&def, ScalarType::F32).expect("MSL");
    // Simdgroup kernels use [[threadgroup_position_in_grid]] for tile dispatch
    assert!(
        msl.contains("[[threadgroup_position_in_grid]]"),
        "simdgroup matmul must use threadgroup_position_in_grid"
    );
}
