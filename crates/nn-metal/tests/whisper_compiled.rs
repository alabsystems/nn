// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Whisper encoder + decoder trace_graph -> CompiledModel integration tests.
//!
//! Proves the trace-compile pipeline generalizes beyond Kokoro by running
//! both the Whisper encoder and decoder through:
//!   1. `trace_graph()` — records computation graph
//!   2. `compile_trace_to_plan_with_fusion()` — compiles to dispatch plan
//!   3. `CompiledModel::builder().build()` — builds executable model
//!   4. `execute_dyn()` — runs on GPU, compares against eager path
//!
//! Tests 1-3: Encoder (conv stem + self-attention + FFN + LayerNorm).
//! Tests 4-5: Decoder (embedding + cross-attention + causal mask + tied projection).
//! Test 6: Peephole pass comparison documenting Whisper vs Kokoro patterns.
//!
//! Uses whisper-tiny config (d=384, 4 layers, 6 heads) for real-dimension
//! tests and a mini config (d=32, 2 layers, 2 heads) for fast CI.
//!
//! Part of #3516.

#![cfg(target_os = "macos")]

mod test_utils;

use nn_core::dyn_tensor::trace;
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{with_nan_check_policy, NanCheckPolicy};
use nn_core::{DType, Device, VarBuilder};
use nn_dsl::trace_compile::compile_trace_to_plan_with_fusion;
use nn_metal::compiled_model::CompiledModel;
use nn_whisper::{WhisperConfig, WhisperModel};

fn init() -> nn_metal::PipelineCache {
    test_utils::gpu_init();
    test_utils::metal_setup()
}

/// Mini Whisper config for fast CI tests (d=32, 2 encoder layers, 1 decoder layer).
fn mini_whisper_config() -> WhisperConfig {
    WhisperConfig::whisper_tiny()
        .with_num_mel_bins(8)
        .with_max_source_positions(32)
        .with_d_model(32)
        .with_encoder_attention_heads(2)
        .with_encoder_layers(2)
        .with_encoder_ffn_dim(64)
        .with_vocab_size(32)
        .with_max_target_positions(16)
        .with_decoder_attention_heads(2)
        .with_decoder_layers(1)
        .with_decoder_ffn_dim(64)
}

// -------------------------------------------------------------------------
// Test 1: Whisper-tiny encoder traces and compiles successfully
// -------------------------------------------------------------------------

/// Trace the whisper-tiny encoder (d=384, 4 layers, 6 heads) and compile
/// to a CompiledPlan. Reports graph node count, step breakdown, fusion
/// stats, and peephole stats.
#[test]
fn test_whisper_tiny_encoder_trace_and_plan() {
    let _cache = init();
    let config = WhisperConfig::whisper_tiny();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    // Use a short mel (32 frames -> 16 seq after stride-2 conv).
    let n_frames = 32;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    // Step 1: trace_graph
    let (output, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph should succeed for whisper-tiny encoder");

    // Verify output shape: [1, seq_len, d_model]
    assert_eq!(output.rank(), 3, "encoder output should be rank 3");
    assert_eq!(output.dim(0).unwrap(), 1, "batch = 1");
    assert_eq!(
        output.dim(2).unwrap(),
        config.d_model,
        "last dim = d_model = {}",
        config.d_model
    );

    let node_count = graph.nodes().len();
    let input_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::Input))
        .count();
    let weight_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::ConstantWeight { .. }))
        .count();

    eprintln!("--- Whisper-tiny encoder trace_graph ---");
    eprintln!("  Graph nodes: {node_count}");
    eprintln!("  Input nodes: {input_count}");
    eprintln!("  ConstantWeight nodes: {weight_count}");

    // Whisper-tiny has 4 encoder layers. Each layer has self-attention (4
    // linear projections) + FFN (2 linear layers) + 2 layer norms + residual
    // connections. Plus the 2 conv stems and final layer norm.
    // Expect a substantial graph.
    assert!(
        node_count > 50,
        "whisper-tiny encoder should have >50 graph nodes, got {node_count}"
    );
    assert_eq!(input_count, 1, "should have exactly 1 input (mel)");

    // Step 2: compile_trace_to_plan_with_fusion
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("compile_trace_to_plan_with_fusion should succeed for whisper-tiny encoder");

    let summary = plan.summary_with_graph(&graph);
    eprintln!("\n{summary}");

    // Verify plan has reasonable structure.
    assert!(
        !plan.steps.is_empty(),
        "compiled plan should have at least 1 step"
    );
    assert_eq!(plan.input_shapes.len(), 1, "plan should have 1 input shape");

    // Report step type breakdown.
    eprintln!("--- Step type breakdown ---");
    for (name, count) in &summary.step_counts {
        eprintln!("  {name}: {count}");
    }
    if !summary.native_op_variants.is_empty() {
        eprintln!("--- NativeOp variants ---");
        for (name, count) in &summary.native_op_variants {
            eprintln!("  {name}: {count}");
        }
    }
    eprintln!(
        "--- Fusion stats ---\n  Chains: {}, Ops fused: {}, Dispatches saved: {}",
        summary.fusion.fused_chains, summary.fusion.fused_ops, summary.fusion.dispatches_saved
    );
    eprintln!(
        "--- Peephole stats ---\n  Native ops: {}, Native dispatches: {}, Passthroughs: {}",
        summary.peephole.native_ops,
        summary.peephole.native_dispatches,
        summary.peephole.passthrough_count
    );
}

// -------------------------------------------------------------------------
// Test 2: CompiledModel executes and matches eager output
// -------------------------------------------------------------------------

/// Build CompiledModel from whisper-tiny encoder trace, execute on GPU,
/// and verify output matches the eager (uncompiled) forward pass.
#[test]
fn test_whisper_tiny_encoder_compiled_vs_eager() {
    let cache = init();
    let config = WhisperConfig::whisper_tiny();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let mut model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let n_frames = 32;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    // Eager forward — extract to CPU immediately to avoid arena staleness.
    let eager_output = model.encode(&mel).expect("eager encode");
    let eager_shape = eager_output.dims().to_vec();
    let eager_data = eager_output.to_flat_vec::<f32>().unwrap();

    // Trace + compile (fresh model to avoid cache state).
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
        .expect("CompiledModel::builder().build() should succeed");

    // Execute compiled model.
    let compiled_output = compiled
        .execute_dyn(&cache, &[&mel])
        .expect("compiled execute should succeed");
    let compiled_shape = compiled_output.dims().to_vec();
    let compiled_data = compiled_output.to_flat_vec::<f32>().unwrap();

    // Verify shapes match.
    assert_eq!(
        eager_shape, compiled_shape,
        "eager shape {eager_shape:?} != compiled shape {compiled_shape:?}"
    );

    // Verify values match within tolerance.
    let max_error = eager_data
        .iter()
        .zip(compiled_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-5,
        "max error between eager and compiled should be < 1e-5, got {max_error}"
    );

    let num_steps = compiled.num_steps();
    let num_dispatches = compiled.num_dispatches();
    let node_count = graph.nodes().len();

    eprintln!("--- Whisper-tiny encoder compiled vs eager ---");
    eprintln!("  Output shape: {eager_shape:?}");
    eprintln!("  Max error: {max_error:.2e}");
    eprintln!("  Graph nodes: {node_count}");
    eprintln!("  Compiled steps: {num_steps}");
    eprintln!("  GPU dispatches: {num_dispatches}");
    eprintln!(
        "  Dispatch reduction: {:.0}% ({node_count} nodes -> {num_dispatches} dispatches)",
        (1.0 - num_dispatches as f64 / node_count as f64) * 100.0
    );
}

// -------------------------------------------------------------------------
// Test 3: Mini-config encoder (fast CI) with full pipeline + diagnostics
// -------------------------------------------------------------------------

/// Uses a small config (d=32, 2 layers, 2 heads) for fast iteration.
/// Exercises the full pipeline and reports all diagnostics.
#[test]
fn test_whisper_mini_encoder_full_pipeline() {
    let cache = init();
    let config = mini_whisper_config();

    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    // Trace.
    let (trace_output, graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("trace_graph");

    // Extract trace output to CPU immediately to avoid arena staleness.
    let trace_shape = trace_output.dims().to_vec();
    let trace_data = trace_output.to_flat_vec::<f32>().unwrap();

    // Verify trace output shape.
    assert_eq!(trace_shape.len(), 3);
    assert_eq!(trace_shape[2], config.d_model);

    // Compile to plan (with fusion) and inspect.
    let plan = compile_trace_to_plan_with_fusion(&graph).expect("compile plan");
    let summary = plan.summary_with_graph(&graph);

    eprintln!("--- Mini encoder plan summary ---");
    eprintln!("{summary}");

    // Build CompiledModel.
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("build compiled model");

    // Execute.
    let compiled_output = compiled
        .execute_dyn(&cache, &[&mel])
        .expect("execute compiled");

    // Verify shape.
    assert_eq!(trace_shape, compiled_output.dims());

    // Compare values.
    let compiled_data = compiled_output.to_flat_vec::<f32>().unwrap();
    let max_error = trace_data
        .iter()
        .zip(compiled_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-5,
        "mini encoder max error: {max_error:.2e} (expected < 1e-5)"
    );

    // Collect TraceOp distribution from graph for gap analysis.
    // Use Debug formatting and extract the variant name (first word before
    // any '{' or '(' or whitespace).
    let mut op_counts = std::collections::BTreeMap::<String, usize>::new();
    for node in graph.nodes() {
        let debug = format!("{:?}", node.op());
        let name = debug
            .split(|c: char| c == '{' || c == '(' || c.is_whitespace())
            .next()
            .unwrap_or("Unknown")
            .to_string();
        *op_counts.entry(name).or_insert(0) += 1;
    }

    eprintln!("\n--- TraceOp distribution ---");
    for (name, count) in &op_counts {
        eprintln!("  {name}: {count}");
    }

    eprintln!(
        "\n--- Result ---\n  Shape: {:?}\n  Max error: {max_error:.2e}\n  Steps: {}\n  Dispatches: {}",
        compiled_output.dims(),
        compiled.num_steps(),
        compiled.num_dispatches(),
    );
}

// -------------------------------------------------------------------------
// Test 4: Mini decoder trace + compile + execute (cross-attention path)
// -------------------------------------------------------------------------

/// Exercises the decoder through the trace-compile pipeline.
///
/// The decoder exercises ops NOT present in the encoder:
/// - Embedding lookup (U32 token input)
/// - Cross-attention (Q from decoder, K/V from encoder output)
/// - sdpa_causal (fused causal masking in self-attention)
/// - Tied output projection (matmul with transposed embedding weight)
///
/// This is the key generalization test: if the pipeline handles both the
/// encoder (conv + self-attention) AND decoder (embedding + cross-attention +
/// causal mask + tied projection), it generalizes to full transformer
/// encoder-decoder architectures.
///
/// Part of #3516.
#[test]
fn test_whisper_mini_decoder_trace_compile_execute() {
    let cache = init();
    let config = mini_whisper_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    // Simulate encoder output: [1, audio_len, d_model].
    // Use 8 audio frames (after stride-2 conv on 16-frame mel).
    let audio_len = 8;
    let encoder_output =
        DynTensor::zeros(&[1, audio_len, config.d_model], DType::F32, &dev).expect("enc output");

    // Token sequence: 4 tokens (batch=1).
    let seq_len = 4;
    let tokens = DynTensor::from_vec_u32(vec![1u32, 2, 3, 4], &[1, seq_len], &dev).expect("tokens");

    // Step 1: trace_graph on decoder forward_no_cache.
    let (trace_output, graph) = trace::trace_graph(|| {
        // Decoder takes two inputs: tokens (U32) and encoder_output (F32).
        let mut traced_tokens = tokens.clone();
        if let Some(id) = trace::record_input(traced_tokens.dims(), DType::U32) {
            traced_tokens.set_trace_id(id);
        }
        let mut traced_enc = encoder_output.clone();
        if let Some(id) = trace::record_input(traced_enc.dims(), DType::F32) {
            traced_enc.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model
                .decoder()
                .forward_no_cache(&traced_tokens, &traced_enc)
        })
    })
    .expect("trace_graph should succeed for whisper mini decoder");

    // Verify output shape: [1, seq_len, vocab_size].
    assert_eq!(trace_output.rank(), 3, "decoder output should be rank 3");
    assert_eq!(trace_output.dim(0).unwrap(), 1, "batch = 1");
    assert_eq!(trace_output.dim(1).unwrap(), seq_len, "seq dim = {seq_len}");
    assert_eq!(
        trace_output.dim(2).unwrap(),
        config.vocab_size,
        "last dim = vocab_size = {}",
        config.vocab_size
    );

    let trace_data = trace_output.to_flat_vec::<f32>().unwrap();
    let trace_shape = trace_output.dims().to_vec();

    // Inspect graph.
    let node_count = graph.nodes().len();
    let input_count = graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), trace::TraceOp::Input))
        .count();

    eprintln!("--- Mini decoder trace_graph ---");
    eprintln!("  Graph nodes: {node_count}");
    eprintln!("  Input nodes: {input_count}");
    assert_eq!(
        input_count, 2,
        "decoder should have 2 inputs (tokens + encoder_output)"
    );

    // Collect TraceOp distribution for gap analysis.
    let mut op_counts = std::collections::BTreeMap::<String, usize>::new();
    for node in graph.nodes() {
        let debug = format!("{:?}", node.op());
        let name = debug
            .split(|c: char| c == '{' || c == '(' || c.is_whitespace())
            .next()
            .unwrap_or("Unknown")
            .to_string();
        *op_counts.entry(name).or_insert(0) += 1;
    }

    eprintln!("\n--- Decoder TraceOp distribution ---");
    for (name, count) in &op_counts {
        eprintln!("  {name}: {count}");
    }

    // Verify decoder-specific ops are present.
    assert!(
        op_counts.contains_key("Embedding"),
        "decoder graph should contain Embedding op"
    );
    assert!(
        op_counts.contains_key("SdpaCausal"),
        "decoder graph should contain SdpaCausal op (self-attention)"
    );

    // Step 2: compile_trace_to_plan_with_fusion.
    let plan = compile_trace_to_plan_with_fusion(&graph)
        .expect("compile_trace should succeed for whisper mini decoder");

    let summary = plan.summary_with_graph(&graph);
    eprintln!("\n{summary}");

    // Step 3: CompiledModel::builder().build().
    let compiled = CompiledModel::builder(&graph, &cache)
        .build()
        .expect("CompiledModel build should succeed for decoder");

    // Step 4: execute_dyn with 2 inputs (tokens + encoder_output).
    let compiled_output = compiled
        .execute_dyn(&cache, &[&tokens, &encoder_output])
        .expect("compiled decoder execute should succeed");

    let compiled_shape = compiled_output.dims().to_vec();
    let compiled_data = compiled_output.to_flat_vec::<f32>().unwrap();

    // Verify shapes match.
    assert_eq!(
        trace_shape, compiled_shape,
        "trace shape {trace_shape:?} != compiled shape {compiled_shape:?}"
    );

    // Verify values match within tolerance.
    let max_error = trace_data
        .iter()
        .zip(compiled_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_error < 1e-4,
        "decoder max error between trace and compiled should be < 1e-4, got {max_error:.2e}"
    );

    eprintln!(
        "\n--- Result ---\n  Shape: {:?}\n  Max error: {max_error:.2e}\n  Steps: {}\n  Dispatches: {}",
        compiled_output.dims(),
        compiled.num_steps(),
        compiled.num_dispatches(),
    );
}

// -------------------------------------------------------------------------
// Test 5: Full encode + decode pipeline (encoder compiled + decoder compiled)
// -------------------------------------------------------------------------

/// Demonstrates the full Whisper pipeline through the compiled path:
/// 1. Trace + compile encoder
/// 2. Execute compiled encoder to get audio features
/// 3. Trace + compile decoder (using compiled encoder output)
/// 4. Execute compiled decoder to get logits
///
/// This is the end-to-end generalization test. If both encoder and decoder
/// compile and execute correctly with matching outputs, the trace-compile
/// pipeline handles the full Whisper architecture.
///
/// Part of #3516.
#[test]
fn test_whisper_mini_full_pipeline_compiled() {
    let cache = init();
    let config = mini_whisper_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    // -- Encoder --
    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    let (_enc_trace, enc_graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("encoder trace");

    let enc_compiled = CompiledModel::builder(&enc_graph, &cache)
        .build()
        .expect("encoder compile");

    let encoder_output = enc_compiled
        .execute_dyn(&cache, &[&mel])
        .expect("encoder execute");

    eprintln!("  Encoder output shape: {:?}", encoder_output.dims());
    eprintln!(
        "  Encoder steps: {}, dispatches: {}",
        enc_compiled.num_steps(),
        enc_compiled.num_dispatches()
    );

    // -- Decoder --
    let seq_len = 3;
    let tokens = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, seq_len], &dev).expect("tokens");

    let (_dec_trace, dec_graph) = trace::trace_graph(|| {
        let mut traced_tokens = tokens.clone();
        if let Some(id) = trace::record_input(traced_tokens.dims(), DType::U32) {
            traced_tokens.set_trace_id(id);
        }
        let mut traced_enc = encoder_output.clone();
        if let Some(id) = trace::record_input(traced_enc.dims(), DType::F32) {
            traced_enc.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model
                .decoder()
                .forward_no_cache(&traced_tokens, &traced_enc)
        })
    })
    .expect("decoder trace");

    let dec_compiled = CompiledModel::builder(&dec_graph, &cache)
        .build()
        .expect("decoder compile");

    let logits = dec_compiled
        .execute_dyn(&cache, &[&tokens, &encoder_output])
        .expect("decoder execute");

    // Verify logits shape: [1, seq_len, vocab_size].
    assert_eq!(logits.rank(), 3);
    assert_eq!(logits.dim(0).unwrap(), 1);
    assert_eq!(logits.dim(1).unwrap(), seq_len);
    assert_eq!(logits.dim(2).unwrap(), config.vocab_size);

    // Verify logits are finite.
    let logit_data = logits.to_flat_vec::<f32>().unwrap();
    let non_finite = logit_data.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "logits should be all finite, found {non_finite} non-finite values"
    );

    eprintln!("  Decoder output shape: {:?}", logits.dims());
    eprintln!(
        "  Decoder steps: {}, dispatches: {}",
        dec_compiled.num_steps(),
        dec_compiled.num_dispatches()
    );
    eprintln!(
        "\n--- Full pipeline summary ---\n  Encoder: {} dispatches\n  Decoder: {} dispatches\n  Total: {} dispatches",
        enc_compiled.num_dispatches(),
        dec_compiled.num_dispatches(),
        enc_compiled.num_dispatches() + dec_compiled.num_dispatches(),
    );
}

// -------------------------------------------------------------------------
// Test 6: Peephole pass comparison (Whisper encoder vs Kokoro patterns)
// -------------------------------------------------------------------------

/// Documents which peephole passes fire for Whisper encoder and decoder,
/// comparing against the Kokoro patterns. This is diagnostic-only — the
/// test always passes, but the output is the deliverable.
///
/// Part of #3516 acceptance criteria: "Document which peephole passes fire
/// for Whisper vs Kokoro."
#[test]
fn test_whisper_peephole_pass_comparison() {
    let _cache = init();
    let config = mini_whisper_config();
    let dev = Device::metal();
    let vb = VarBuilder::zeros(DType::F32, &dev);
    let model = WhisperModel::load(&vb, config.clone()).expect("model load");

    // -- Encoder --
    let n_frames = 16;
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, n_frames], DType::F32, &dev)
        .expect("mel tensor");

    let (_enc_out, enc_graph) = trace::trace_graph(|| {
        let mut traced_mel = mel.clone();
        if let Some(id) = trace::record_input(traced_mel.dims(), DType::F32) {
            traced_mel.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model.encoder().forward_no_cache(&traced_mel)
        })
    })
    .expect("encoder trace");

    let enc_plan = compile_trace_to_plan_with_fusion(&enc_graph).expect("encoder compile");
    let enc_summary = enc_plan.summary_with_graph(&enc_graph);

    // -- Decoder --
    let audio_len = 8;
    let encoder_output =
        DynTensor::zeros(&[1, audio_len, config.d_model], DType::F32, &dev).expect("enc output");
    let tokens = DynTensor::from_vec_u32(vec![1u32, 2, 3], &[1, 3], &dev).expect("tokens");

    let (_dec_out, dec_graph) = trace::trace_graph(|| {
        let mut traced_tokens = tokens.clone();
        if let Some(id) = trace::record_input(traced_tokens.dims(), DType::U32) {
            traced_tokens.set_trace_id(id);
        }
        let mut traced_enc = encoder_output.clone();
        if let Some(id) = trace::record_input(traced_enc.dims(), DType::F32) {
            traced_enc.set_trace_id(id);
        }
        with_nan_check_policy(NanCheckPolicy::Skip, || {
            model
                .decoder()
                .forward_no_cache(&traced_tokens, &traced_enc)
        })
    })
    .expect("decoder trace");

    let dec_plan = compile_trace_to_plan_with_fusion(&dec_graph).expect("decoder compile");
    let dec_summary = dec_plan.summary_with_graph(&dec_graph);

    eprintln!("=== Whisper Peephole Pass Comparison ===");
    eprintln!();
    eprintln!("--- Encoder ---");
    eprintln!("  Graph nodes: {}", enc_graph.nodes().len());
    eprintln!("  Compiled steps: {}", enc_plan.steps.len());
    eprintln!(
        "  Peephole: native_ops={}, native_dispatches={}, passthroughs={}",
        enc_summary.peephole.native_ops,
        enc_summary.peephole.native_dispatches,
        enc_summary.peephole.passthrough_count,
    );
    eprintln!(
        "  Fusion: chains={}, ops_fused={}, dispatches_saved={}",
        enc_summary.fusion.fused_chains,
        enc_summary.fusion.fused_ops,
        enc_summary.fusion.dispatches_saved,
    );
    eprintln!("  Step types:");
    for (name, count) in &enc_summary.step_counts {
        eprintln!("    {name}: {count}");
    }
    if !enc_summary.native_op_variants.is_empty() {
        eprintln!("  NativeOp variants:");
        for (name, count) in &enc_summary.native_op_variants {
            eprintln!("    {name}: {count}");
        }
    }

    eprintln!();
    eprintln!("--- Decoder ---");
    eprintln!("  Graph nodes: {}", dec_graph.nodes().len());
    eprintln!("  Compiled steps: {}", dec_plan.steps.len());
    eprintln!(
        "  Peephole: native_ops={}, native_dispatches={}, passthroughs={}",
        dec_summary.peephole.native_ops,
        dec_summary.peephole.native_dispatches,
        dec_summary.peephole.passthrough_count,
    );
    eprintln!(
        "  Fusion: chains={}, ops_fused={}, dispatches_saved={}",
        dec_summary.fusion.fused_chains,
        dec_summary.fusion.fused_ops,
        dec_summary.fusion.dispatches_saved,
    );
    eprintln!("  Step types:");
    for (name, count) in &dec_summary.step_counts {
        eprintln!("    {name}: {count}");
    }
    if !dec_summary.native_op_variants.is_empty() {
        eprintln!("  NativeOp variants:");
        for (name, count) in &dec_summary.native_op_variants {
            eprintln!("    {name}: {count}");
        }
    }

    eprintln!();
    eprintln!("--- Comparison with Kokoro patterns ---");
    eprintln!("  Kokoro uses: FusedResBlock, LstmSequence, NormLinear, AddLayerNorm,");
    eprintln!("    Conv1dGemm, ChannelsFirstLayerNorm, BatchedStyleProjection.");
    eprintln!("  Whisper encoder uses: conv stem + self-attention (SDPA) + FFN + LayerNorm.");
    eprintln!("  Whisper decoder adds: Embedding, SdpaCausal, cross-attention SDPA,");
    eprintln!("    tied output projection (matmul with transposed embedding weight).");
    eprintln!("  Key difference: Whisper has no LSTM, no FusedResBlock, no Snake activation.");
    eprintln!("  Key similarity: Both use Linear, LayerNorm, GELU, Softmax via SDPA.");
}
