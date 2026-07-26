// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for ICB + autocast composition.
//!
//! Tests the pure logic in `compiled_model_icb_autocast.rs`:
//! dtype resolution, cast point detection, F16 safety classification.
//!
//! Part of #3499.

use std::collections::HashMap;

use nn_core::DType;
use nn_dsl::ir::ScalarType;
use nn_dsl::NativeOpKind;

use super::{is_f16_safe, resolve_autocast};

// ── resolve_autocast: basic scenarios ──────────────────────────────────

#[test]
fn test_simple_elementwise_chain_all_f16() {
    // A chain of elementwise Dispatch steps should all resolve to F16.
    let steps = vec![
        make_dispatch_step("add"),
        make_dispatch_step("mul"),
        make_dispatch_step("tanh"),
    ];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0]), f32_meta(vec![1])];
    let gemm = no_gemm(3);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");

    assert_eq!(plan.dtype_per_step.len(), 3);
    assert_eq!(plan.dtype_per_step[0], DType::F16);
    assert_eq!(plan.dtype_per_step[1], DType::F16);
    assert_eq!(plan.dtype_per_step[2], DType::F16);
    assert_eq!(plan.total_f16_steps, 3);
    assert_eq!(plan.total_f32_steps, 0);
    // No cast points — all same dtype.
    assert!(plan.cast_points.is_empty());
}

#[test]
fn test_lstm_stays_f32() {
    // LSTM NativeOp must stay F32 — never assigned F16.
    let steps = vec![make_lstm_step()];
    let metas = vec![f32_meta(vec![])];
    let gemm = no_gemm(1);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");

    assert_eq!(plan.dtype_per_step[0], DType::F32);
    assert_eq!(plan.total_f32_steps, 1);
    assert_eq!(plan.total_f16_steps, 0);
}

#[test]
fn test_matmul_uses_f16() {
    // Matmul (linear) Dispatch step should be F16-safe.
    let steps = vec![make_dispatch_step("linear")];
    let metas = vec![f32_meta(vec![])];
    let gemm = no_gemm(1);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");

    assert_eq!(plan.dtype_per_step[0], DType::F16);
    assert_eq!(plan.total_f16_steps, 1);
}

#[test]
fn test_cast_points_at_boundaries() {
    // F16 dispatch → F32 LSTM → F16 dispatch: cast points at LSTM boundaries.
    let steps = vec![
        make_dispatch_step("linear"),
        make_lstm_step(),
        make_dispatch_step("add"),
    ];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0]), f32_meta(vec![1])];
    let gemm = no_gemm(3);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");

    assert_eq!(plan.dtype_per_step[0], DType::F16); // linear → F16
    assert_eq!(plan.dtype_per_step[1], DType::F32); // LSTM → F32
    assert_eq!(plan.dtype_per_step[2], DType::F16); // add → F16

    // Cast point at step 1 (LSTM reads F16 input, needs F32).
    assert!(plan.needs_cast(1), "LSTM should need a cast from F16 input");
    // Cast point at step 2 (add reads F32 from LSTM, needs F16).
    assert!(plan.needs_cast(2), "add should need a cast from F32 LSTM output");
    assert_eq!(plan.cast_points.len(), 2);

    // Verify cast directions.
    let (idx, from, to) = plan.cast_points[0];
    assert_eq!(idx, 1);
    assert_eq!(from, DType::F16);
    assert_eq!(to, DType::F32);

    let (idx, from, to) = plan.cast_points[1];
    assert_eq!(idx, 2);
    assert_eq!(from, DType::F32);
    assert_eq!(to, DType::F16);
}

#[test]
fn test_mixed_chain_correct_plan() {
    // linear(F16) → softmax(F32) → mul(F16) → layer_norm(F32) → linear(F16)
    let steps = vec![
        make_dispatch_step("linear"),
        make_dispatch_step("softmax"),
        make_dispatch_step("mul"),
        make_dispatch_step("layer_norm"),
        make_dispatch_step("linear"),
    ];
    let metas = vec![
        f32_meta(vec![]),
        f32_meta(vec![0]),
        f32_meta(vec![1]),
        f32_meta(vec![2]),
        f32_meta(vec![3]),
    ];
    let gemm = no_gemm(5);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");

    assert_eq!(plan.dtype_per_step[0], DType::F16); // linear
    assert_eq!(plan.dtype_per_step[1], DType::F32); // softmax (reduce)
    assert_eq!(plan.dtype_per_step[2], DType::F16); // mul
    assert_eq!(plan.dtype_per_step[3], DType::F32); // layer_norm (reduce)
    assert_eq!(plan.dtype_per_step[4], DType::F16); // linear

    assert_eq!(plan.total_f16_steps, 3);
    assert_eq!(plan.total_f32_steps, 2);
    // 4 cast points: F16→F32 at softmax, F32→F16 at mul, F16→F32 at layer_norm, F32→F16 at linear
    assert_eq!(plan.cast_points.len(), 4);
}

#[test]
fn test_empty_model() {
    let plan = resolve_autocast(&[], &[], &[]).expect("empty should succeed");

    assert!(plan.dtype_per_step.is_empty());
    assert!(plan.cast_points.is_empty());
    assert_eq!(plan.total_f16_steps, 0);
    assert_eq!(plan.total_f32_steps, 0);
    assert!((plan.f16_ratio() - 0.0).abs() < f64::EPSILON);
}

// ── resolve_autocast: passthrough propagation ─────────────────────────

#[test]
fn test_passthrough_inherits_source_dtype() {
    // F16 dispatch → passthrough should inherit F16.
    let steps = vec![
        make_dispatch_step("add"),
        make_passthrough_step(),
        make_dispatch_step("mul"),
    ];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0]), f32_meta(vec![1])];
    let gemm = no_gemm(3);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");

    assert_eq!(plan.dtype_per_step[0], DType::F16);
    assert_eq!(plan.dtype_per_step[1], DType::F16); // inherited from step 0
    assert_eq!(plan.dtype_per_step[2], DType::F16);
    // No cast points — all F16.
    assert!(plan.cast_points.is_empty());
}

#[test]
fn test_passthrough_inherits_f32_from_lstm() {
    // LSTM(F32) → passthrough should inherit F32.
    let steps = vec![
        make_lstm_step(),
        make_passthrough_step(),
        make_dispatch_step("add"),
    ];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0]), f32_meta(vec![1])];
    let gemm = no_gemm(3);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");

    assert_eq!(plan.dtype_per_step[0], DType::F32); // LSTM
    assert_eq!(plan.dtype_per_step[1], DType::F32); // passthrough inherits F32
    assert_eq!(plan.dtype_per_step[2], DType::F16); // add
    // Cast at step 2: F32→F16.
    assert!(plan.needs_cast(2));
}

// ── is_f16_safe: individual op classification ──────────────────────────

#[test]
fn test_f16_safe_elementwise_dispatch() {
    let step = make_dispatch_step("add");
    assert!(is_f16_safe(&step, None));
}

#[test]
fn test_f16_unsafe_softmax() {
    let step = make_dispatch_step("softmax");
    assert!(!is_f16_safe(&step, None));
}

#[test]
fn test_f16_unsafe_mixed_gemm() {
    let step = make_dispatch_step("linear");
    let gemm = crate::compiled_model::MixedGemmInfo {
        m: 8,
        k: 256,
        n: 256,
        batch_count: 1,
        transpose_b: true,
        broadcast_b: false,
        has_bias: false,
        activation: None,
    };
    assert!(!is_f16_safe(&step, Some(&gemm)));
}

#[test]
fn test_f16_safe_conv1d_gemm() {
    let step = nn_dsl::CompiledStep::NativeOp {
        op: NativeOpKind::Conv1dGemm {
            input_shape: vec![1, 128, 256],
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            dilation: 1,
            groups: 1,
            has_bias: true,
        },
        weight_data: HashMap::new(),
    };
    assert!(is_f16_safe(&step, None));
}

#[test]
fn test_f16_unsafe_runtime_op() {
    let step = nn_dsl::CompiledStep::RuntimeOp {
        op: nn_dsl::trace_compile::RuntimeOpKind::RepeatInterleave {
            dim: 0,
            input_shape: vec![4],
            counts_shape: vec![4],
        },
    };
    assert!(!is_f16_safe(&step, None));
}

// ── IcbAutocastPlan methods ─────────────────────────────────────────────

#[test]
fn test_plan_icb_compatible_steps() {
    // linear(F16) → LSTM(F32) → add(F16): step 0 ICB-ok, step 1 not dispatch, step 2 needs cast.
    let steps = vec![
        make_dispatch_step("linear"),
        make_lstm_step(),
        make_dispatch_step("add"),
    ];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0]), f32_meta(vec![1])];
    let gemm = no_gemm(3);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");
    let compatible = plan.icb_compatible_steps(&steps);

    assert!(compatible[0]); // Dispatch, no cast needed
    assert!(!compatible[1]); // NativeOp, not a Dispatch
    assert!(!compatible[2]); // Dispatch, but needs cast (F32→F16)
}

#[test]
fn test_plan_scalar_type_conversion() {
    let steps = vec![make_dispatch_step("add"), make_lstm_step()];
    let metas = vec![f32_meta(vec![]), f32_meta(vec![0])];
    let gemm = no_gemm(2);

    let plan = resolve_autocast(&steps, &metas, &gemm).expect("resolve should succeed");

    assert_eq!(plan.step_scalar_type(0), ScalarType::F16);
    assert_eq!(plan.step_scalar_type(1), ScalarType::F32);
}

// ── Error handling ──────────────────────────────────────────────────────

#[test]
fn test_mismatched_lengths_error() {
    let steps = vec![make_dispatch_step("add")];
    let metas = vec![]; // mismatch
    let gemm = no_gemm(1);

    let result = resolve_autocast(&steps, &metas, &gemm);
    assert!(result.is_err());
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn f32_meta(edges: Vec<usize>) -> crate::compiled_model::StepMeta {
    crate::compiled_model::StepMeta {
        edges,
        scalar_type: ScalarType::F32,
        numel: 1,
    }
}

fn no_gemm(n: usize) -> Vec<Option<crate::compiled_model::MixedGemmInfo>> {
    vec![None; n]
}

fn make_dispatch_step(name: &str) -> nn_dsl::CompiledStep {
    use nn_dsl::{CompiledKernel, CompiledStep, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind};

    let node = TensorNode::new(
        TensorNodeId::new(0),
        TensorOpKind::Input {
            name: "x".into(),
            shape: vec![1],
        },
        vec![1],
    );
    let def = TensorKernelDef::new(name, vec![node], TensorNodeId::new(0));
    CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data: HashMap::new(),
        external_node_ids: None,
    }
}

fn make_passthrough_step() -> nn_dsl::CompiledStep {
    nn_dsl::CompiledStep::Passthrough {
        op_name: "reshape".into(),
        output_shape: vec![1],
    }
}

fn make_lstm_step() -> nn_dsl::CompiledStep {
    nn_dsl::CompiledStep::NativeOp {
        op: NativeOpKind::LstmSequence {
            hidden_size: 256,
            input_shape: vec![8, 1, 640],
            h_shape: vec![1, 256],
            reverse: false,
        },
        weight_data: HashMap::new(),
    }
}
