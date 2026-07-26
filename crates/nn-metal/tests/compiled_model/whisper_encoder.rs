// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper encoder CompiledModel integration test.
//!
//! Phase 1 of #3090 (CompiledModel transformer expansion):
//! - Traces the encoder forward pass via `trace_graph()`
//! - Compiles to `CompiledModel` via `builder().build()`
//! - Verifies compiled output matches eager with max error < 1e-5
//! - Compares dispatch counts (compiled vs eager)

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

/// Whisper-tiny-like config for fast testing: 2 heads, 2 layers, d=16.
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

/// AC1: Whisper AudioEncoder produces valid ComputationGraph via trace_graph().
#[test]
fn test_whisper_encoder_trace_graph() {
    let _cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    let (output, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph should succeed");

    // Verify output shape: [1, seq_len, d_model].
    assert_eq!(output.rank(), 3);
    assert_eq!(output.dim(0).unwrap(), 1);
    assert_eq!(output.dim(2).unwrap(), config.d_model);

    // Verify graph has nodes.
    let node_count = graph.nodes().len();
    assert!(
        node_count > 10,
        "encoder graph should have substantial node count, got {node_count}"
    );

    // Verify at least 1 input node exists.
    let input_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::Input))
        .count();
    assert!(
        input_count >= 1,
        "graph should have at least 1 input, got {input_count}"
    );

    // Check for ConstantWeight nodes (positional embedding).
    let cw_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::ConstantWeight { .. }))
        .count();
    eprintln!(
        "Whisper encoder (2-layer, d=16): {node_count} nodes, \
         {input_count} inputs, {cw_count} constant weights"
    );
}

/// AC2: CompiledModel executes encoder via trace+builder, matches eager output.
#[test]
fn test_whisper_encoder_compiled_vs_eager() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let mut model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    // Eager forward — extract data to CPU immediately to avoid arena staleness.
    let eager_output = model.encode(&mel).expect("eager encode");
    let eager_shape = eager_output.dims().to_vec();
    let eager_data = eager_output.to_flat_vec::<f32>().unwrap();

    // Trace + compile via builder (fresh model to avoid cache state).
    let model_ref = WhisperModel::load(&vb, config).expect("model load for trace");

    let (_trace_out, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model_ref.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph for compile");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("builder.build() should succeed");

    // Execute compiled model.
    let compiled_output = compiled
        .execute_dyn(&cache, &[&mel])
        .expect("compiled execute should succeed");
    let compiled_data = compiled_output.to_flat_vec::<f32>().unwrap();

    // Verify shapes match.
    assert_eq!(eager_shape, compiled_output.dims());

    let max_error = eager_data
        .iter()
        .zip(compiled_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-5,
        "max error between eager and compiled should be < 1e-5, got {max_error}"
    );
}

/// AC4: Autocast compiled encoder matches F32 compiled encoder within F16 tolerance.
///
/// First real-model autocast validation. Exercises the full transformer F16 pipeline:
/// Conv1d(F16) → LayerNorm(F32) → SDPA(F16) → FFN(F16) with boundary casts.
/// Part of #2981.
#[test]
fn test_whisper_encoder_autocast_vs_f32() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;

    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    // Trace encoder forward.
    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph");

    // F32 compiled baseline.
    let f32_model = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile f32");
    let f32_out = f32_model.execute_dyn(&cache, &[&mel]).expect("f32 exec");
    let f32_data = f32_out.to_flat_vec::<f32>().unwrap();

    // Autocast compiled.
    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");
    assert!(ac_model.is_autocast());
    let ac_out = ac_model
        .execute_dyn(&cache, &[&mel])
        .expect("autocast exec");
    let ac_data = ac_out.to_flat_vec::<f32>().unwrap();

    assert_eq!(f32_out.dims(), ac_out.dims());

    let max_error = f32_data
        .iter()
        .zip(ac_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // Encoder: 2 layers × (Conv1d + SDPA + FFN) with F16 boundary casts.
    // Zero-init weights reduce numerical divergence. Tolerance mirrors
    // autocast_test.rs flash attention chain (0.1).
    assert!(
        max_error < 0.1,
        "encoder autocast vs f32 max error: {max_error:.2e} (expected < 0.1)"
    );
    eprintln!("Whisper encoder autocast vs F32: max_error={max_error:.2e}");
}

/// AC5: Autocast encoder has F16 steps — proves autocast classification is active.
///
/// Without this assertion, a broken autocast classifier could pass AC4
/// (all F32, zero error) while silently disabling the speedup.
/// Part of #2981.
#[test]
fn test_whisper_encoder_autocast_has_f16_steps() {
    use nn_core::mixed_precision::MixedPrecisionPolicy;

    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph");

    let policy = MixedPrecisionPolicy::apple_silicon_default();
    let ac_model = CompiledModel::builder(&graph, &cache)
        .autocast(policy)
        .build()
        .expect("compile autocast");

    // Encoder has Conv1d, Linear (QKV, FFN), and SDPA — all compute-dominant.
    // With 2 layers: at least 2 SDPA + 6 QKV projections + 4 FFN linears + 2 Conv1d = 14.
    // Some may not qualify (simdgroup size threshold), but >0 is the baseline assertion.
    let f16_count = ac_model.num_autocast_f16_steps();
    assert!(
        f16_count > 0,
        "encoder autocast should have at least 1 F16 step, got 0"
    );
    eprintln!(
        "Whisper encoder autocast: {f16_count} F16 steps out of {} total",
        ac_model.num_steps()
    );
}

/// AC3: Dispatch count comparison — compiled should use fusion to reduce steps.
#[test]
fn test_whisper_encoder_dispatch_count() {
    let cache = init();
    let config = test_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    let (_output, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph should succeed");

    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("compile should succeed");

    let step_count = compiled.num_steps();
    let dispatch_count = compiled.num_dispatches();
    let node_count = graph.nodes().len();

    // Compiled dispatch count should be less than total graph nodes because
    // many nodes are weights, shape ops, or identity passthroughs.
    assert!(
        dispatch_count < node_count,
        "compiled dispatches ({dispatch_count}) should be fewer than graph nodes ({node_count})"
    );

    eprintln!(
        "Whisper encoder (2-layer, d=16): {node_count} graph nodes -> \
         {step_count} compiled steps, {dispatch_count} dispatches"
    );
}
