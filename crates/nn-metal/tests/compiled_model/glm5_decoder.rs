// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLM-4/5 decoder CompiledModel integration test.
//!
//! Uses `forward_from_embeddings()` (pre-computed embeddings, no KV cache)
//! to produce a clean computation graph for `trace_graph()` and `CompiledModel`.
//! The embedding layer is excluded because `forward()` takes `&[usize]` which
//! is not traceable as a DynTensor input.
//!
//! Single input: hidden states `[1, seq_len, hidden_size]` F32.
//! Positions are `&[usize]` (baked into graph as RoPE constants).

use nn_core::dyn_tensor::trace;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_core::{DType, Device, VarBuilder};
use nn_glm5::{Glm5Config, Glm5Model};
use nn_metal::compiled_model::CompiledModel;

fn init() -> nn_metal::PipelineCache {
    super::test_utils::gpu_init();
    super::test_utils::metal_setup()
}

/// Minimal GLM-5 config for fast testing: 2 layers, 4 heads, 2 kv groups.
/// Matches nn-glm5 internal tiny_config() values.
fn test_config() -> Glm5Config {
    Glm5Config::new(
        256,      // hidden_size
        512,      // ffn_hidden_size
        2,        // num_layers
        4,        // num_attention_heads
        2,        // multi_query_group_num
        100,      // padded_vocab_size
        64,       // kv_channels
        1e-5,     // layernorm_epsilon
        64,       // seq_length
        true,     // rmsnorm
        true,     // add_qkv_bias
        false,    // add_bias_linear
        10_000.0, // rope_theta
    )
}

/// Create test input: pre-computed embeddings [1, seq_len, hidden_size] on GPU.
fn test_input(config: &Glm5Config, dev: &Device) -> DynTensor {
    let seq_len = 4;
    DynTensor::zeros(&[1, seq_len, config.hidden_size], DType::F32, dev)
        .expect("hidden states tensor")
}

/// Positions matching seq_len=4.
fn test_positions() -> Vec<usize> {
    vec![0, 1, 2, 3]
}

/// AC1: GLM5 decoder produces valid ComputationGraph via trace_graph().
#[test]
fn test_glm5_decoder_trace_graph() {
    let _cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = Glm5Model::load(&vb, config.clone()).expect("model load");

    let hidden = test_input(&config, &dev);
    let positions = test_positions();

    let (output, graph) = trace::trace_graph(|| {
        let mut traced = hidden.clone();
        if let Some(id) = trace::record_input(traced.dims(), DType::F32) {
            traced.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.forward_from_embeddings(&traced, &positions, None)
        })
    })
    .expect("trace_graph should succeed");

    // Verify output shape: [1, seq_len, padded_vocab_size].
    assert_eq!(output.rank(), 3);
    assert_eq!(output.dim(0).unwrap(), 1);
    assert_eq!(output.dim(1).unwrap(), 4); // seq_len
    assert_eq!(output.dim(2).unwrap(), config.padded_vocab_size);

    let node_count = graph.nodes().len();
    assert!(
        node_count > 10,
        "decoder graph should have substantial node count, got {node_count}"
    );

    let input_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::Input))
        .count();
    assert!(
        input_count >= 1,
        "graph should have at least 1 input (hidden states), got {input_count}"
    );

    eprintln!("GLM5 decoder (2-layer, d=256): {node_count} nodes, {input_count} inputs");
}

/// AC2: CompiledModel executes decoder via trace+builder, matches eager output.
#[test]
fn test_glm5_decoder_compiled_vs_eager() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = Glm5Model::load(&vb, config.clone()).expect("model load");

    let hidden = test_input(&config, &dev);
    let positions = test_positions();

    // Eager forward — extract to CPU immediately.
    let eager_output = with_nan_check_policy(NanCheckPolicy::Skip, || {
        model.forward_from_embeddings(&hidden, &positions, None)
    })
    .expect("eager forward");
    let eager_shape = eager_output.dims().to_vec();
    let eager_data = eager_output.to_flat_vec::<f32>().unwrap();

    // Trace + compile via builder (fresh model to avoid state).
    let model_ref = Glm5Model::load(&vb, config).expect("model load for trace");
    let positions_ref = test_positions();

    let (_trace_out, graph) = trace::trace_graph(|| {
        let mut traced = hidden.clone();
        if let Some(id) = trace::record_input(traced.dims(), DType::F32) {
            traced.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model_ref.forward_from_embeddings(&traced, &positions_ref, None)
        })
    })
    .expect("trace_graph for compile");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("builder.build() should succeed");

    let compiled_output = compiled
        .execute_dyn(&cache, &[&hidden])
        .expect("compiled execute");
    let compiled_data = compiled_output.to_flat_vec::<f32>().unwrap();

    assert_eq!(eager_shape, compiled_output.dims());

    let max_error = eager_data
        .iter()
        .zip(compiled_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // GLM5 has fused QKV, partial RoPE, SwiGLU MLP — allow same tolerance
    // as Whisper decoder (1e-4).
    assert!(
        max_error < 1e-4,
        "max error between eager and compiled should be < 1e-4, got {max_error}"
    );

    eprintln!("GLM5 decoder compiled vs eager: shape={eager_shape:?}, max_error={max_error:.2e}");
}

/// AC3: Dispatch count — compiled should reduce steps vs graph nodes.
#[test]
fn test_glm5_decoder_dispatch_count() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = Glm5Model::load(&vb, config.clone()).expect("model load");

    let hidden = test_input(&config, &dev);
    let positions = test_positions();

    let (_output, graph) = trace::trace_graph(|| {
        let mut traced = hidden.clone();
        if let Some(id) = trace::record_input(traced.dims(), DType::F32) {
            traced.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.forward_from_embeddings(&traced, &positions, None)
        })
    })
    .expect("trace_graph should succeed");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile should succeed");

    let step_count = compiled.num_steps();
    let dispatch_count = compiled.num_dispatches();
    let node_count = graph.nodes().len();

    assert!(
        dispatch_count < node_count,
        "compiled dispatches ({dispatch_count}) should be fewer than \
         graph nodes ({node_count})"
    );

    eprintln!(
        "GLM5 decoder (2-layer, d=256): {node_count} graph nodes -> \
         {step_count} compiled steps, {dispatch_count} dispatches"
    );
}

/// AC4: Autocast compiled decoder matches F32 compiled decoder within F16 tolerance.
///
/// Exercises: fused QKV(F16) → partial RoPE → GQA SDPA(F16) → SwiGLU MLP(F16)
/// with boundary casts. Uses forward_from_embeddings for clean graph.
/// Part of #3260.
#[test]
fn test_glm5_decoder_autocast_vs_f32() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;

    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = Glm5Model::load(&vb, config.clone()).expect("model load");

    let hidden = test_input(&config, &dev);
    let positions = test_positions();

    // Trace decoder forward (no cache — clean graph).
    let (_output, graph) = trace::trace_graph(|| {
        let mut traced = hidden.clone();
        if let Some(id) = trace::record_input(traced.dims(), DType::F32) {
            traced.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.forward_from_embeddings(&traced, &positions, None)
        })
    })
    .expect("trace_graph");

    // F32 compiled baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute_dyn(&cache, &[&hidden]).expect("f32 exec");
    let f32_data = f32_out.to_flat_vec::<f32>().unwrap();

    // Autocast compiled.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    let ac_out = ac_model
        .execute_dyn(&cache, &[&hidden])
        .expect("autocast exec");
    let ac_data = ac_out.to_flat_vec::<f32>().unwrap();

    assert_eq!(f32_out.dims(), ac_out.dims());

    let max_error = f32_data
        .iter()
        .zip(ac_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // GLM5: fused QKV + partial RoPE + SwiGLU + GQA, all F16.
    // Zero-init weights minimize divergence. Same tolerance as Whisper.
    assert!(
        max_error < 0.1,
        "GLM5 autocast vs f32 max error: {max_error:.2e} (expected < 0.1)"
    );
    eprintln!("GLM5 decoder autocast vs F32: max_error={max_error:.2e}");
}
