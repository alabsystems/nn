// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper decoder CompiledModel integration test.
//!
//! Completes the Whisper compiled model story (encoder: `whisper_encoder.rs`).
//! The decoder uses `forward_no_cache()` (teacher-forcing, no KV cache) to
//! produce a clean computation graph for `trace_graph()` and `CompiledModel`.
//!
//! Two inputs: token IDs (F32 with integer values) and encoder output (F32).
//! Token IDs are F32 because the compiled model embedding MSL kernel reads
//! `float*` buffers and casts to `uint` internally.

use nn_core::dyn_tensor::trace;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_core::{DType, Device, VarBuilder};
use nn_metal::compiled_model::CompiledModel;
use nn_whisper::{WhisperConfig, WhisperModel};

fn init() -> nn_metal::PipelineCache {
    super::test_utils::gpu_init();
    super::test_utils::metal_setup()
}

/// Whisper-tiny-like config for fast testing: 2 heads, 1 decoder layer, d=16.
fn test_config() -> WhisperConfig {
    WhisperConfig::whisper_tiny()
        .with_num_mel_bins(4)
        .with_max_source_positions(16)
        .with_d_model(16)
        .with_encoder_attention_heads(2)
        .with_encoder_layers(2)
        .with_encoder_ffn_dim(32)
        .with_vocab_size(32)
        .with_max_target_positions(16)
        .with_decoder_attention_heads(2)
        .with_decoder_layers(1)
        .with_decoder_ffn_dim(32)
}

/// Create test inputs: F32 token IDs and F32 encoder output on GPU.
fn test_inputs(config: &WhisperConfig, dev: &Device) -> (DynTensor, DynTensor) {
    let seq_len = 4;
    let audio_len = 8;

    // Token IDs as F32 (integer values 0..seq_len). Compiled model embedding
    // kernel reads float* buffers and casts to uint.
    let token_values: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let tokens = DynTensor::from_vec(token_values, &[1, seq_len], dev).expect("token tensor");

    // Encoder output: [1, audio_len, d_model].
    let encoder_output =
        DynTensor::zeros(&[1, audio_len, config.d_model], DType::F32, dev).expect("encoder output");

    (tokens, encoder_output)
}

/// AC1: Whisper TextDecoder produces valid ComputationGraph via trace_graph().
#[test]
fn test_whisper_decoder_trace_graph() {
    let _cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let (tokens, encoder_output) = test_inputs(&config, &dev);

    let (output, graph) = trace::trace_graph(|| {
        let mut traced_tokens = tokens.clone();
        let mut traced_enc = encoder_output.clone();
        if let Some(id) = trace::record_input(traced_tokens.dims(), DType::F32) {
            traced_tokens.set_trace_id(id);
        }
        if let Some(id) = trace::record_input(traced_enc.dims(), DType::F32) {
            traced_enc.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model
                .decoder()
                .forward_no_cache(&traced_tokens, &traced_enc)
        })
    })
    .expect("trace_graph should succeed");

    // Verify output shape: [1, seq_len, vocab_size].
    assert_eq!(output.rank(), 3);
    assert_eq!(output.dim(0).unwrap(), 1);
    assert_eq!(output.dim(1).unwrap(), 4); // seq_len
    assert_eq!(output.dim(2).unwrap(), config.vocab_size);

    // Verify graph has nodes.
    let node_count = graph.nodes().len();
    assert!(
        node_count > 10,
        "decoder graph should have substantial node count, got {node_count}"
    );

    // Verify at least 2 input nodes (tokens + encoder output).
    let input_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::Input))
        .count();
    assert!(
        input_count >= 2,
        "graph should have at least 2 inputs (tokens + encoder output), got {input_count}"
    );

    eprintln!(
        "Whisper decoder (1-layer, d=16): {node_count} nodes, \
         {input_count} inputs"
    );
}

/// AC2: CompiledModel executes decoder via trace+builder, matches eager output.
#[test]
fn test_whisper_decoder_compiled_vs_eager() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let (tokens, encoder_output) = test_inputs(&config, &dev);

    // Eager forward (no cache) — extract to CPU immediately to avoid arena staleness.
    let eager_output = with_nan_check_policy(NanCheckPolicy::Skip, || {
        model.decoder().forward_no_cache(&tokens, &encoder_output)
    })
    .expect("eager decode");
    let eager_shape = eager_output.dims().to_vec();
    let eager_data = eager_output.to_flat_vec::<f32>().unwrap();

    // Trace + compile via builder (fresh model to avoid state).
    let model_ref = WhisperModel::load(&vb, config).expect("model load for trace");

    let (_trace_out, graph) = trace::trace_graph(|| {
        let mut traced_tokens = tokens.clone();
        let mut traced_enc = encoder_output.clone();
        if let Some(id) = trace::record_input(traced_tokens.dims(), DType::F32) {
            traced_tokens.set_trace_id(id);
        }
        if let Some(id) = trace::record_input(traced_enc.dims(), DType::F32) {
            traced_enc.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model_ref
                .decoder()
                .forward_no_cache(&traced_tokens, &traced_enc)
        })
    })
    .expect("trace_graph for compile");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("builder.build() should succeed");

    // Execute compiled model.
    let compiled_output = compiled
        .execute_dyn(&cache, &[&tokens, &encoder_output])
        .expect("compiled execute should succeed");
    let compiled_data = compiled_output.to_flat_vec::<f32>().unwrap();

    // Verify shapes match.
    assert_eq!(eager_shape, compiled_output.dims());

    let max_error = eager_data
        .iter()
        .zip(compiled_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Decoder has more ops than encoder (cross-attention, embedding), so
    // allow slightly higher tolerance than encoder's 1e-5.
    assert!(
        max_error < 1e-4,
        "max error between eager and compiled should be < 1e-4, got {max_error}"
    );

    eprintln!(
        "Whisper decoder compiled vs eager: shape={eager_shape:?}, max_error={max_error:.2e}"
    );
}

/// AC4: Autocast compiled decoder matches F32 compiled decoder within F16 tolerance.
///
/// Exercises: Embedding(F16) → SelfAttention+SDPA(F16) → CrossAttention+SDPA(F16)
/// → FFN(F16) with boundary casts. Uses forward_no_cache for clean graph.
/// Part of #2981.
#[test]
fn test_whisper_decoder_autocast_vs_f32() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;

    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let (tokens, encoder_output) = test_inputs(&config, &dev);

    // Trace decoder forward (no cache — clean graph).
    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_tokens = tokens.clone();
        let mut traced_enc = encoder_output.clone();
        if let Some(id) = trace::record_input(traced_tokens.dims(), DType::F32) {
            traced_tokens.set_trace_id(id);
        }
        if let Some(id) = trace::record_input(traced_enc.dims(), DType::F32) {
            traced_enc.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model
                .decoder()
                .forward_no_cache(&traced_tokens, &traced_enc)
        })
    })
    .expect("trace_graph");

    // F32 compiled baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model
        .execute_dyn(&cache, &[&tokens, &encoder_output])
        .expect("f32 exec");
    let f32_data = f32_out.to_flat_vec::<f32>().unwrap();

    // Autocast compiled.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    let ac_out = ac_model
        .execute_dyn(&cache, &[&tokens, &encoder_output])
        .expect("autocast exec");
    let ac_data = ac_out.to_flat_vec::<f32>().unwrap();

    assert_eq!(f32_out.dims(), ac_out.dims());

    let max_error = f32_data
        .iter()
        .zip(ac_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Decoder: cross-attention + self-attention + FFN + embedding, all F16.
    // Zero-init weights minimize divergence. Same tolerance as encoder.
    assert!(
        max_error < 0.1,
        "decoder autocast vs f32 max error: {max_error:.2e} (expected < 0.1)"
    );
    eprintln!("Whisper decoder autocast vs F32: max_error={max_error:.2e}");
}

/// AC3: Dispatch count comparison — compiled should reduce steps vs graph nodes.
#[test]
fn test_whisper_decoder_dispatch_count() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let (tokens, encoder_output) = test_inputs(&config, &dev);

    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_tokens = tokens.clone();
        let mut traced_enc = encoder_output.clone();
        if let Some(id) = trace::record_input(traced_tokens.dims(), DType::F32) {
            traced_tokens.set_trace_id(id);
        }
        if let Some(id) = trace::record_input(traced_enc.dims(), DType::F32) {
            traced_enc.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model
                .decoder()
                .forward_no_cache(&traced_tokens, &traced_enc)
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
        "Whisper decoder (1-layer, d=16): {node_count} graph nodes -> \
         {step_count} compiled steps, {dispatch_count} dispatches"
    );
}
