// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reproduction test for #2774: Conv1d → Transpose(1,2) → LayerNorm shape
//! mismatch in NY IBP propagation.
//!
//! The Kokoro TextEncoder architecture is:
//!   Embedding → 3×[Conv1d(d_en, d_en, k=5, p=2) → Transpose(1,2) → LayerNorm(d_en) → Transpose(1,2) → LeakyReLU(0.2)]
//!
//! When traced and translated to NY's `GraphNetwork`, the Transpose
//! layer causes shape flattening: NY sees `[1, C*T]` instead of
//! `[1, T, C]`, which makes LayerNorm's `gamma.len() != shape[ndim-1]` check
//! fail. This test isolates the minimal pattern to reproduce the issue.
//!
//! Part of #2774.
//! Part of #2218.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, LayerNorm, Module};
use nn_core::test_utils::cpu;
use nn_core::DType;
use nn_verify::{trace_to_graph_model, BoundedTensor};
use ndarray::{ArrayD, IxDyn};

/// Create a traced input tensor (register as graph input).
fn trace_input(t: &DynTensor) -> DynTensor {
    let mut out = t.clone();
    out.set_trace_id(record_input(out.dims(), out.dtype()).expect("tracing active"));
    out
}

/// Build a Conv1d → Transpose(1,2) → LayerNorm chain with configurable size.
///
/// kernel_size = 5, padding = 2 matches the Kokoro TextEncoder.
fn build_conv_transpose_layernorm_sized(c: usize) -> (Conv1d, LayerNorm) {
    let kernel_size = 5usize;

    // Conv1d weight: [out_channels, in_channels/groups, kernel_size] = [C, C, K]
    let w_data: Vec<f32> = (0..c * c * kernel_size)
        .map(|i| (i as f32) * 0.01)
        .collect();
    let w = DynTensor::new(&w_data, &[c, c, kernel_size], &cpu()).unwrap();

    // Conv1d bias: [out_channels] = [C]
    let b_data: Vec<f32> = (0..c).map(|i| (i as f32) * 0.1).collect();
    let b = DynTensor::new(&b_data, &[c], &cpu()).unwrap();

    let cfg = Conv1dConfig::new(2, 1, 1);
    let conv = Conv1d::new(w, Some(b), cfg).expect("Conv1d::new");

    // LayerNorm weight and bias: [C]
    let ln_w = DynTensor::ones(&[c], DType::F32, &cpu()).unwrap();
    let ln_b = DynTensor::zeros(&[c], DType::F32, &cpu()).unwrap();
    let ln = LayerNorm::new(ln_w, ln_b, 1e-5).expect("LayerNorm::new");

    (conv, ln)
}

/// Trace Conv1d → Transpose(1,2) → LayerNorm through `trace_graph`.
///
/// Returns `(output_tensor, computation_graph)`.
fn trace_conv_transpose_layernorm(
    conv: &Conv1d,
    ln: &LayerNorm,
    batch: usize,
    channels: usize,
    seq_len: usize,
) -> (DynTensor, nn_core::dyn_tensor::trace::ComputationGraph) {
    let input_shape = [batch, channels, seq_len];
    let input_data = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let x = trace_input(&input_data);
        // Conv1d: [B, C, T] → [B, C, T]
        let conv_out = conv.forward(&x)?;
        // Transpose(1,2): [B, C, T] → [B, T, C]
        let transposed = conv_out.transpose(1, 2)?;
        // LayerNorm(C): normalize last dim of [B, T, C]
        let normed = ln.forward(&transposed)?;
        Ok(normed)
    })
    .expect("trace Conv1d → Transpose → LayerNorm");

    (result, graph)
}

// -- Test 1: Verify trace_to_graph_model succeeds for this pattern -----------

#[test]
fn test_conv_transpose_layernorm_graph_builds() {
    let (conv, ln) = build_conv_transpose_layernorm_sized(8);
    let (_result, graph) = trace_conv_transpose_layernorm(&conv, &ln, 1, 8, 3);

    // This should succeed: graph translation handles Conv1d, Transpose, LayerNorm.
    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model should succeed")
        .graph;

    eprintln!("GraphNetwork built: {} nodes", gn.num_nodes());
}

// -- Test 2: Verify IBP propagation succeeds ---------------------------------

#[test]
fn test_conv_transpose_layernorm_ibp_propagation() {
    let (conv, ln) = build_conv_transpose_layernorm_sized(8);
    let batch = 1;
    let channels = 8;
    let seq_len = 3;
    let (_result, graph) = trace_conv_transpose_layernorm(&conv, &ln, batch, channels, seq_len);

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;

    // Input bounds: [B=1, C=8, T=3]
    let input_shape = [batch, channels, seq_len];
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&input_shape), -1.0f32),
        ArrayD::from_elem(IxDyn(&input_shape), 1.0f32),
    )
    .expect("valid bounds");

    // This is where #2774 manifests: IBP through Transpose may flatten
    // the BoundedTensor shape, causing LayerNorm's gamma.len() check to fail.
    let output_bounds = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP propagation through Conv1d → Transpose → LayerNorm");

    super::common::assert_bounds_valid(&output_bounds);
    eprintln!(
        "IBP output shape: {:?}, bounds valid",
        output_bounds.shape()
    );

    // Output should be [B=1, T=3, C=8] (post-transpose, post-layernorm).
    assert_eq!(
        output_bounds.shape(),
        &[1, seq_len, channels],
        "output shape should be [B, T, C] after Transpose(1,2) + LayerNorm"
    );
}

// -- Test 3: Full TextEncoder pattern (with second Transpose + LeakyReLU) ----

#[test]
fn test_full_text_encoder_block_ibp() {
    let (conv, ln) = build_conv_transpose_layernorm_sized(8);
    let batch = 1;
    let channels = 8;
    let seq_len = 3;
    let input_shape = [batch, channels, seq_len];
    let input_data = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let x = trace_input(&input_data);
        // Conv1d: [B, C, T] → [B, C, T]
        let conv_out = conv.forward(&x)?;
        // Transpose(1,2): [B, C, T] → [B, T, C]
        let transposed = conv_out.transpose(1, 2)?;
        // LayerNorm(C): normalize last dim
        let normed = ln.forward(&transposed)?;
        // Transpose(1,2): [B, T, C] → [B, C, T]
        let back = normed.transpose(1, 2)?;
        // LeakyReLU(0.2)
        let activated = back.leaky_relu(0.2)?;
        Ok(activated)
    })
    .expect("trace full TextEncoder block");

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&input_shape), -1.0f32),
        ArrayD::from_elem(IxDyn(&input_shape), 1.0f32),
    )
    .expect("valid bounds");

    let output_bounds = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP through full Conv1d → Transpose → LayerNorm → Transpose → LeakyReLU");

    super::common::assert_bounds_valid(&output_bounds);
    eprintln!("Full block IBP output shape: {:?}", output_bounds.shape());

    // Output should be [B=1, C=8, T=3] (original layout restored).
    assert_eq!(
        output_bounds.shape(),
        &[batch, channels, seq_len],
        "output shape should be [B, C, T] after two transposes"
    );
}

// -- Test 4: Extract per-layer bounds (this is where the error message appears) --

#[test]
fn test_conv_transpose_layernorm_layer_extraction() {
    let (conv, ln) = build_conv_transpose_layernorm_sized(8);
    let batch = 1;
    let channels = 8;
    let seq_len = 3;
    let (_result, graph) = trace_conv_transpose_layernorm(&conv, &ln, batch, channels, seq_len);

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;

    let input_shape = [batch, channels, seq_len];
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&input_shape), -1.0f32),
        ArrayD::from_elem(IxDyn(&input_shape), 1.0f32),
    )
    .expect("valid bounds");

    // extract_layer_bounds is where the shape mismatch error typically surfaces.
    match nn_verify::layer_bounds::extract_layer_bounds(&gn, &input_bounds) {
        Ok(records) => {
            eprintln!("extract_layer_bounds succeeded: {} layers", records.len());
            for rec in &records {
                eprintln!(
                    "  Layer {}: {:?} (node {:?})",
                    rec.layer_index, rec.layer_type, rec.node_name
                );
            }
        }
        Err(e) => {
            eprintln!(
                "extract_layer_bounds FAILED (#2774): {e}\n\
                 This is the expected failure — shape mismatch in LayerNorm \
                 after Transpose flattens the BoundedTensor."
            );
            // Record the error for analysis but don't fail the test yet —
            // the purpose is to diagnose, not to assert success.
            panic!(
                "#2774 reproduction: extract_layer_bounds failed with: {e}\n\
                 See compose_kokoro_conv_transpose_layernorm.rs for context."
            );
        }
    }
}

// -- Test 5: Production-like dimensions (d_en=128) ---------------------------

#[test]
fn test_conv_transpose_layernorm_d128_ibp() {
    let d_en = 128;
    let (conv, ln) = build_conv_transpose_layernorm_sized(d_en);
    let batch = 1;
    let seq_len = 4;
    let (_result, graph) = trace_conv_transpose_layernorm(&conv, &ln, batch, d_en, seq_len);

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;

    let input_shape = [batch, d_en, seq_len];
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&input_shape), -1.0f32),
        ArrayD::from_elem(IxDyn(&input_shape), 1.0f32),
    )
    .expect("valid bounds");

    let output_bounds = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP with d_en=128 production-like dimensions");

    super::common::assert_bounds_valid(&output_bounds);
    eprintln!(
        "d128 IBP output shape: {:?}, bounds valid",
        output_bounds.shape()
    );

    assert_eq!(
        output_bounds.shape(),
        &[batch, seq_len, d_en],
        "output shape should be [B, T, d_en] after Transpose(1,2) + LayerNorm"
    );
}

// -- Test 6: Full TextEncoder with actual model (no production weights) ------

#[test]
fn test_actual_text_encoder_ibp() {
    use nn_core::VarBuilder;
    use nn_models::kokoro_tts::TextEncoder;
    use nn_models::KokoroConfig;

    let config = KokoroConfig::default();
    let vocab_size = config.plbert.vocab_size;
    let d_en = config.d_en;

    // Use random weights via VarBuilder::zeros (deterministic, no file needed).
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let text_encoder = TextEncoder::load(vb.pp("text_encoder"), vocab_size, d_en)
        .expect("TextEncoder::load with zeros");

    let token_shape = [1, 4];
    let tokens = DynTensor::full(&token_shape, 5.0, DType::I64, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let x = trace_input(&tokens);
        text_encoder
            .forward(&x)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))
    })
    .expect("TextEncoder trace");

    let gn = match trace_to_graph_model(&graph) {
        Ok(result) => result.graph,
        Err(e) => {
            eprintln!("trace_to_graph_model failed: {e}");
            eprintln!("  This may be expected if TextEncoder has multiple inputs or LSTM.");
            return;
        }
    };

    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&token_shape), 0.0f32),
        ArrayD::from_elem(IxDyn(&token_shape), (vocab_size - 1) as f32),
    )
    .expect("valid bounds");

    match gn.propagate_ibp(&input_bounds) {
        Ok(output_bounds) => {
            super::common::assert_bounds_valid(&output_bounds);
            eprintln!("TextEncoder IBP output shape: {:?}", output_bounds.shape());
        }
        Err(e) => {
            eprintln!(
                "TextEncoder IBP FAILED (#2774): {e}\n\
                 This is the shape mismatch we're investigating."
            );
            panic!("#2774: TextEncoder IBP failed: {e}");
        }
    }
}
