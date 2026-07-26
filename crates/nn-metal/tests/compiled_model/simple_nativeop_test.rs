// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end compiled model parity tests for simple NativeOp variants:
//! LayerNorm, MaxPool1d, and ConstantWeight (Arange).
//!
//! Each test builds a trace graph → compiles to CompiledModel (NativeOp) →
//! executes on GPU → verifies against a CPU reference implementation.
//!
//! These were identified as untested NativeOp paths in the P10 prover audit.
//! Part of #2218.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use super::helpers::{assert_close, compile_and_run, create_input_buffer, input_node};

// -- CPU reference: LayerNorm -------------------------------------------------

/// CPU LayerNorm: normalize over hidden dim, apply weight and bias.
///
/// Input x: `[B, T, C]` row-major. weight/bias: `[C]`.
/// For each (b, t):
///   normed[c] = (x[b,t,c] - mean) / sqrt(var + eps) * weight[c] + bias[c]
fn cpu_layer_norm(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    batch: usize,
    time: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch * time * hidden];
    for b in 0..batch {
        for t in 0..time {
            let offset = (b * time + t) * hidden;
            let row = &x[offset..offset + hidden];
            let mean: f32 = row.iter().sum::<f32>() / hidden as f32;
            let var: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / hidden as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for c in 0..hidden {
                output[offset + c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
            }
        }
    }
    output
}

// -- CPU reference: MaxPool1d -------------------------------------------------

/// CPU MaxPool1d on `[B, C, T]` input with padding.
fn cpu_max_pool1d(
    x: &[f32],
    batch: usize,
    channels: usize,
    time: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let out_len = (time + 2 * padding - kernel_size) / stride + 1;
    let mut output = vec![f32::NEG_INFINITY; batch * channels * out_len];
    for b in 0..batch {
        for c in 0..channels {
            let in_offset = (b * channels + c) * time;
            let out_offset = (b * channels + c) * out_len;
            for o in 0..out_len {
                let start = o * stride;
                let mut max_val = f32::NEG_INFINITY;
                for k in 0..kernel_size {
                    let pos = start + k;
                    if pos >= padding && pos < time + padding {
                        let t = pos - padding;
                        let val = x[in_offset + t];
                        if val > max_val {
                            max_val = val;
                        }
                    }
                }
                output[out_offset + o] = max_val;
            }
        }
    }
    output
}

// -- Tests: LayerNorm NativeOp ------------------------------------------------

/// [1, 4, 16] -> LayerNorm(eps=1e-5): fused GPU kernel via NativeOp.
#[test]
fn test_compiled_layer_norm_nativeop() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (1, 4, 16);
    let eps = 1e-5_f64;

    let x_data = super::test_utils::rand_f32_vec(0x1A0E_0001, batch * time * hidden, -1.0, 1.0);
    let w_data = super::test_utils::rand_f32_vec(0x1A0E_0002, hidden, 0.8, 1.2);
    let b_data = super::test_utils::rand_f32_vec(0x1A0E_0003, hidden, -0.1, 0.1);

    let weight = WeightRef::new(w_data.clone(), vec![hidden]).expect("weight");
    let bias = WeightRef::new(b_data.clone(), vec![hidden]).expect("bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        TraceNode::new(
            1,
            "layer_norm_0".into(),
            TraceOp::LayerNorm { eps, weight, bias },
            vec![0],
            vec![batch, time, hidden],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let result = compile_and_run(&cache, graph, &[&x_buf], batch * time * hidden);

    let expected = cpu_layer_norm(&x_data, &w_data, &b_data, batch, time, hidden, eps as f32);
    assert_close("layer_norm_nativeop", &result, &expected, 1e-4);
}

/// [2, 8, 32] -> LayerNorm: batched case with larger dimensions.
#[test]
fn test_compiled_layer_norm_nativeop_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (2, 8, 32);
    let eps = 1e-5_f64;

    let x_data = super::test_utils::rand_f32_vec(0x1A0E_0010, batch * time * hidden, -2.0, 2.0);
    let w_data = super::test_utils::rand_f32_vec(0x1A0E_0011, hidden, 0.5, 1.5);
    let b_data = super::test_utils::rand_f32_vec(0x1A0E_0012, hidden, -0.2, 0.2);

    let weight = WeightRef::new(w_data.clone(), vec![hidden]).expect("weight");
    let bias = WeightRef::new(b_data.clone(), vec![hidden]).expect("bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        TraceNode::new(
            1,
            "layer_norm_0".into(),
            TraceOp::LayerNorm { eps, weight, bias },
            vec![0],
            vec![batch, time, hidden],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let result = compile_and_run(&cache, graph, &[&x_buf], batch * time * hidden);

    let expected = cpu_layer_norm(&x_data, &w_data, &b_data, batch, time, hidden, eps as f32);
    assert_close("layer_norm_nativeop_batched", &result, &expected, 1e-4);
}

// -- Tests: MaxPool1d NativeOp ------------------------------------------------

/// [1, 4, 20] -> MaxPool1d(kernel=3, stride=2, padding=1): standard case.
#[test]
fn test_compiled_max_pool1d_nativeop() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (1, 4, 20);
    let (kernel_size, stride, padding) = (3, 2, 1);
    let out_len = (time + 2 * padding - kernel_size) / stride + 1; // 10

    let x_data = super::test_utils::rand_f32_vec(0xA900_0001, batch * channels * time, -2.0, 2.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "max_pool1d_0".into(),
            TraceOp::MaxPool1d {
                kernel_size,
                stride,
                padding,
            },
            vec![0],
            vec![batch, channels, out_len],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let result = compile_and_run(&cache, graph, &[&x_buf], batch * channels * out_len);

    let expected = cpu_max_pool1d(&x_data, batch, channels, time, kernel_size, stride, padding);
    assert_close("max_pool1d_nativeop", &result, &expected, 1e-6);
}

/// [2, 8, 50] -> MaxPool1d(kernel=5, stride=3, padding=0): no padding, batched.
#[test]
fn test_compiled_max_pool1d_nativeop_batched() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, channels, time) = (2, 8, 50);
    let (kernel_size, stride, padding) = (5, 3, 0);
    let out_len = (time + 2 * padding - kernel_size) / stride + 1; // 16

    let x_data = super::test_utils::rand_f32_vec(0xA900_0010, batch * channels * time, -3.0, 3.0);

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, channels, time]),
        TraceNode::new(
            1,
            "max_pool1d_0".into(),
            TraceOp::MaxPool1d {
                kernel_size,
                stride,
                padding,
            },
            vec![0],
            vec![batch, channels, out_len],
            DType::F32,
        ),
    ]);

    let x_buf = create_input_buffer(&cache, &x_data);
    let result = compile_and_run(&cache, graph, &[&x_buf], batch * channels * out_len);

    let expected = cpu_max_pool1d(&x_data, batch, channels, time, kernel_size, stride, padding);
    assert_close("max_pool1d_nativeop_batched", &result, &expected, 1e-6);
}

// -- Tests: ConstantWeight (Arange) NativeOp ----------------------------------

/// Arange(0, 10, 1) -> ConstantWeight: precomputed constant embedded as weight.
#[test]
fn test_compiled_arange_constant_weight() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let n = 10;

    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "arange_0".into(),
        TraceOp::Arange {
            start: 0.0,
            end: 10.0,
            step: 1.0,
        },
        vec![],
        vec![n],
        DType::F32,
    )]);

    // No external inputs — arange is fully precomputed.
    let result = compile_and_run(&cache, graph, &[], n);

    let expected: Vec<f32> = (0..n).map(|i| i as f32).collect();
    assert_close("arange_constant_weight", &result, &expected, 0.0);
}

/// Arange(0.5, 5.0, 0.5) -> ConstantWeight: fractional step.
#[test]
fn test_compiled_arange_constant_weight_fractional() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let n = 9; // ceil((5.0 - 0.5) / 0.5) = 9

    let graph = ComputationGraph::from_nodes(vec![TraceNode::new(
        0,
        "arange_0".into(),
        TraceOp::Arange {
            start: 0.5,
            end: 5.0,
            step: 0.5,
        },
        vec![],
        vec![n],
        DType::F32,
    )]);

    let result = compile_and_run(&cache, graph, &[], n);

    let expected: Vec<f32> = (0..n).map(|i| 0.5 + i as f32 * 0.5).collect();
    assert_close(
        "arange_constant_weight_fractional",
        &result,
        &expected,
        1e-6,
    );
}

// -- NormLinear: LayerNorm + Linear fusion (#3089) ----------------------------

/// LayerNorm(16) → Linear(16, 8): peephole should fuse into NormLinear.
///
/// Verifies both: (a) the compiled model contains a NormLinear NativeOp,
/// and (b) the fused GPU output matches the CPU reference.
#[test]
fn test_compiled_norm_linear_layernorm() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, hidden, out_f) = (4, 16, 8);
    let eps = 1e-5_f32;

    let ln_w = super::test_utils::rand_f32_vec(0xF089_0001, hidden, 0.5, 1.5);
    let ln_b = super::test_utils::rand_f32_vec(0xF089_0002, hidden, -0.1, 0.1);
    let w = super::test_utils::rand_f32_vec(0xF089_0003, out_f * hidden, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0xF089_0004, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xF089_0005, batch * hidden, -1.0, 1.0);

    fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
        WeightRef::new(data, shape).expect("weight")
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, hidden]),
        TraceNode::new(
            1,
            "layernorm_0".into(),
            TraceOp::LayerNorm {
                eps: f64::from(eps),
                weight: weight(ln_w.clone(), vec![hidden]),
                bias: weight(ln_b.clone(), vec![hidden]),
            },
            vec![0],
            vec![batch, hidden],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, hidden]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    // Compile and verify NormLinear NativeOp is present.
    let compiled = nn_metal::compiled_model::CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile ln+linear");
    let has_norm_linear = compiled.steps().iter().any(|s| {
        matches!(
            s,
            nn_dsl::CompiledStep::NativeOp {
                op: nn_dsl::NativeOpKind::NormLinear { .. },
                ..
            }
        )
    });
    assert!(
        has_norm_linear,
        "peephole should fuse LayerNorm+Linear into NormLinear"
    );

    // Execute on GPU.
    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    // CPU reference: LayerNorm → Linear.
    let normed = cpu_layer_norm(&input_data, &ln_w, &ln_b, batch, 1, hidden, eps);
    let expected = super::test_utils::linear_ref(&normed, &w, Some(&b), batch, hidden, out_f);
    assert_close("norm_linear_ln", &result, &expected, 1e-3);
}

// -- NormLinear: RmsNorm + Linear fusion (#3089) ------------------------------

/// RmsNorm(16) → Linear(16, 8): peephole should fuse into NormLinear.
#[test]
fn test_compiled_norm_linear_rmsnorm() {
    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, hidden, out_f) = (4, 16, 8);
    let eps = 1e-5_f32;

    let rms_w = super::test_utils::rand_f32_vec(0xF089_0011, hidden, 0.5, 1.5);
    let w = super::test_utils::rand_f32_vec(0xF089_0012, out_f * hidden, -0.5, 0.5);
    let b = super::test_utils::rand_f32_vec(0xF089_0013, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xF089_0014, batch * hidden, -1.0, 1.0);

    fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
        WeightRef::new(data, shape).expect("weight")
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, hidden]),
        TraceNode::new(
            1,
            "rmsnorm_0".into(),
            TraceOp::RmsNorm {
                eps: f64::from(eps),
                weight: weight(rms_w.clone(), vec![hidden]),
            },
            vec![0],
            vec![batch, hidden],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w.clone(), vec![out_f, hidden]),
                bias: Some(weight(b.clone(), vec![out_f])),
            },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    // Compile and verify NormLinear NativeOp is present.
    let compiled = nn_metal::compiled_model::CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile rms+linear");
    let has_norm_linear = compiled.steps().iter().any(|s| {
        matches!(
            s,
            nn_dsl::CompiledStep::NativeOp {
                op: nn_dsl::NativeOpKind::NormLinear { .. },
                ..
            }
        )
    });
    assert!(
        has_norm_linear,
        "peephole should fuse RmsNorm+Linear into NormLinear"
    );

    // Execute on GPU.
    let result = compile_and_run(
        &cache,
        graph,
        &[&create_input_buffer(&cache, &input_data)],
        batch * out_f,
    );

    // CPU reference: RmsNorm → Linear.
    let mut normed = vec![0.0_f32; batch * hidden];
    for row in 0..batch {
        let row_data = &input_data[row * hidden..(row + 1) * hidden];
        let ms: f32 = row_data.iter().map(|v| v * v).sum::<f32>() / hidden as f32;
        let rms = (ms + eps).sqrt();
        for col in 0..hidden {
            normed[row * hidden + col] = row_data[col] / rms * rms_w[col];
        }
    }
    let expected = super::test_utils::linear_ref(&normed, &w, Some(&b), batch, hidden, out_f);
    assert_close("norm_linear_rms", &result, &expected, 1e-3);
}

// -- NormLinear autocast: F16 parity (#3287) ----------------------------------

/// LayerNorm(16) → Linear(16, 8) in autocast mode: NormLinear should run F16
/// because it has F32 accumulators (threadgroup + dot product).
///
/// Verifies:
/// 1. Peephole fuses LayerNorm+Linear into NormLinear NativeOp
/// 2. Autocast classifies NormLinear as compute-dominant (F16)
/// 3. GPU output matches F32 baseline within F16 tolerance
///
/// Part of #3287.
#[test]
fn test_autocast_norm_linear_layernorm_parity() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    use nn_metal::compiled_model::CompiledModel;

    use super::helpers::read_output_n;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, hidden, out_f) = (4, 16, 8);
    let eps = 1e-5_f32;

    let ln_w = super::test_utils::rand_f32_vec(0xAC87_0001, hidden, 0.5, 1.5);
    let ln_b = super::test_utils::rand_f32_vec(0xAC87_0002, hidden, -0.1, 0.1);
    let w = super::test_utils::rand_f32_vec(0xAC87_0003, out_f * hidden, -0.3, 0.3);
    let b = super::test_utils::rand_f32_vec(0xAC87_0004, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xAC87_0005, batch * hidden, -1.0, 1.0);

    fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
        WeightRef::new(data, shape).expect("weight")
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, hidden]),
        TraceNode::new(
            1,
            "layernorm_0".into(),
            TraceOp::LayerNorm {
                eps: f64::from(eps),
                weight: weight(ln_w, vec![hidden]),
                bias: weight(ln_b, vec![hidden]),
            },
            vec![0],
            vec![batch, hidden],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w, vec![out_f, hidden]),
                bias: Some(weight(b, vec![out_f])),
            },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let has_norm_linear = f32_model.steps().iter().any(|s| {
        matches!(
            s,
            nn_dsl::CompiledStep::NativeOp {
                op: nn_dsl::NativeOpKind::NormLinear { .. },
                ..
            }
        )
    });
    assert!(
        has_norm_linear,
        "peephole should fuse LayerNorm+Linear into NormLinear"
    );

    let buf = create_input_buffer(&cache, &input_data);
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, batch * out_f);

    // Autocast: NormLinear should be classified as compute-dominant → F16.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast(), "model should be autocast");
    assert!(
        ac_model.num_autocast_f16_steps() > 0,
        "NormLinear should be classified as F16 compute-dominant step"
    );

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, batch * out_f);

    // F16 NormLinear: norm uses F32 threadgroup accumulators, GEMM uses F32 dot.
    // Precision loss from F16 I/O quantization only.
    assert_close("autocast_norm_linear_ln", &ac_result, &f32_result, 2e-2);
}

/// RmsNorm(16) → Linear(16, 8) in autocast mode: same as LayerNorm variant.
///
/// Part of #3287.
#[test]
fn test_autocast_norm_linear_rmsnorm_parity() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;
    use nn_metal::compiled_model::CompiledModel;

    use super::helpers::read_output_n;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, hidden, out_f) = (4, 16, 8);
    let eps = 1e-5_f32;

    let rms_w = super::test_utils::rand_f32_vec(0xAC87_0011, hidden, 0.5, 1.5);
    let w = super::test_utils::rand_f32_vec(0xAC87_0012, out_f * hidden, -0.3, 0.3);
    let b = super::test_utils::rand_f32_vec(0xAC87_0013, out_f, -0.1, 0.1);
    let input_data = super::test_utils::rand_f32_vec(0xAC87_0014, batch * hidden, -1.0, 1.0);

    fn weight(data: Vec<f32>, shape: Vec<usize>) -> WeightRef {
        WeightRef::new(data, shape).expect("weight")
    }

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, hidden]),
        TraceNode::new(
            1,
            "rmsnorm_0".into(),
            TraceOp::RmsNorm {
                eps: f64::from(eps),
                weight: weight(rms_w, vec![hidden]),
            },
            vec![0],
            vec![batch, hidden],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "linear_0".into(),
            TraceOp::Linear {
                weight: weight(w, vec![out_f, hidden]),
                bias: Some(weight(b, vec![out_f])),
            },
            vec![1],
            vec![batch, out_f],
            DType::F32,
        ),
    ]);

    // F32 baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let buf = create_input_buffer(&cache, &input_data);
    let f32_out = f32_model.execute(&cache, &[&buf]).expect("f32 exec");
    let f32_result = read_output_n(&f32_out, batch * out_f);

    // Autocast.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    assert!(
        ac_model.num_autocast_f16_steps() > 0,
        "NormLinear (RmsNorm) should be F16 in autocast"
    );

    let ac_out = ac_model.execute(&cache, &[&buf]).expect("autocast exec");
    let ac_result = read_output_n(&ac_out, batch * out_f);

    assert_close("autocast_norm_linear_rms", &ac_result, &f32_result, 2e-2);
}

// -- Tests: NativeOp planned-buffer redirect (#3448) --------------------------

/// Verify that NativeOp LayerNorm uses the planned-buffer redirect to
/// eliminate blit relocations. The redirect arms before NativeOp execution
/// so `arena_alloc_or_create` returns the planned buffer region directly.
///
/// After execute, `dispatch_stats().blits` should be 0 for a single
/// NativeOp step because the output writes directly into the planned buffer.
#[test]
fn test_nativeop_redirect_eliminates_blits() {
    use nn_metal::compiled_model::CompiledModel;

    use super::helpers::read_output_n;

    super::test_utils::gpu_init();
    let cache = super::test_utils::metal_setup();

    let (batch, time, hidden) = (1, 4, 16);
    let eps = 1e-5_f64;

    let x_data = super::test_utils::rand_f32_vec(0x3448_0001, batch * time * hidden, -1.0, 1.0);
    let w_data = super::test_utils::rand_f32_vec(0x3448_0002, hidden, 0.8, 1.2);
    let b_data = super::test_utils::rand_f32_vec(0x3448_0003, hidden, -0.1, 0.1);

    let weight = WeightRef::new(w_data.clone(), vec![hidden]).expect("weight");
    let bias = WeightRef::new(b_data.clone(), vec![hidden]).expect("bias");

    let graph = ComputationGraph::from_nodes(vec![
        input_node(0, &[batch, time, hidden]),
        TraceNode::new(
            1,
            "layer_norm_redirect".into(),
            TraceOp::LayerNorm { eps, weight, bias },
            vec![0],
            vec![batch, time, hidden],
            DType::F32,
        ),
    ]);

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile");

    let x_buf = create_input_buffer(&cache, &x_data);

    // Reset counters before execute to isolate this model's blits.
    nn_metal::reset_counters();
    let out_buf = compiled.execute(&cache, &[&x_buf]).expect("execute");
    let stats = nn_metal::dispatch_stats();

    eprintln!(
        "[#3448] NativeOp redirect: compute={}, blits={}, flushes={}",
        stats.compute_encodings, stats.blits, stats.flushes,
    );

    // With the redirect, the NativeOp writes directly into the planned buffer.
    // No blit relocation should be needed.
    assert_eq!(
        stats.blits, 0,
        "NativeOp planned-buffer redirect should eliminate blits, got {}",
        stats.blits,
    );

    // Verify correctness too — redirect must not corrupt output.
    let result = read_output_n(&out_buf, batch * time * hidden);
    let expected = cpu_layer_norm(&x_data, &w_data, &b_data, batch, time, hidden, eps as f32);
    assert_close("nativeop_redirect_parity", &result, &expected, 1e-4);
}
