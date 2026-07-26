// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro encoder (TextEncoder) convert tests: BiLSTM + Linear with transposes.

use std::collections::HashMap;
use std::path::Path;

use nn_core::dyn_tensor::trace::{TraceNode, TraceOp};

use crate::graph_build::ImportedGraph;
use crate::import_model;

#[cfg(all(feature = "metal", target_os = "macos"))]
use crate::convert;

#[cfg(all(feature = "metal", feature = "verify", target_os = "macos"))]
use crate::convert::ConvertResult;

/// Write synthetic Kokoro encoder weights to a safetensors file.
///
/// BiLSTM (d_en=8, hidden=4, 1 layer bidirectional):
///   Forward:  w_ih [16, 8]=128, w_hh [16, 4]=64, b_ih [16], b_hh [16]
///   Backward: same with _reverse suffix
/// Linear: weight [8, 8]=64, bias [8]
fn write_kokoro_encoder_weights(dir: &Path) -> std::path::PathBuf {
    let mut tensors = HashMap::new();

    let w_ih: Vec<u8> = (0..128)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let w_hh: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32) * 0.01).to_le_bytes())
        .collect();
    let bias: Vec<u8> = [0.0f32; 16].iter().flat_map(|f| f.to_le_bytes()).collect();

    for (name, shape, data) in [
        ("lstm.weight_ih_l0", vec![16, 8], w_ih.as_slice()),
        ("lstm.weight_hh_l0", vec![16, 4], w_hh.as_slice()),
        ("lstm.bias_ih_l0", vec![16], bias.as_slice()),
        ("lstm.bias_hh_l0", vec![16], bias.as_slice()),
    ] {
        tensors.insert(
            name.to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, data).unwrap(),
        );
    }

    let w_ih_rev: Vec<u8> = (0..128)
        .flat_map(|i| ((i as f32) * -0.01).to_le_bytes())
        .collect();
    let w_hh_rev: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32) * -0.01).to_le_bytes())
        .collect();

    for (name, shape, data) in [
        (
            "lstm.weight_ih_l0_reverse",
            vec![16, 8],
            w_ih_rev.as_slice(),
        ),
        (
            "lstm.weight_hh_l0_reverse",
            vec![16, 4],
            w_hh_rev.as_slice(),
        ),
        ("lstm.bias_ih_l0_reverse", vec![16], bias.as_slice()),
        ("lstm.bias_hh_l0_reverse", vec![16], bias.as_slice()),
    ] {
        tensors.insert(
            name.to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape, data).unwrap(),
        );
    }

    let lin_w: Vec<u8> = (0..64)
        .flat_map(|i| ((i as f32) * 0.02).to_le_bytes())
        .collect();
    let lin_b: Vec<u8> = [0.1f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();

    tensors.insert(
        "linear.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 8], &lin_w).unwrap(),
    );
    tensors.insert(
        "linear.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &lin_b).unwrap(),
    );

    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

fn import_kokoro_encoder_fixture() -> ImportedGraph {
    let dir = std::env::temp_dir().join(format!("nn_import_enc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/kokoro_encoder_mini.json"),
    )
    .unwrap();
    let weights_path = write_kokoro_encoder_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

/// Kokoro encoder imports with correct structure: BiLSTM expands to 9 nodes,
/// plus 4 Transpose + 1 Linear.
#[test]
fn test_import_kokoro_encoder_mini_structure() {
    let imported = import_kokoro_encoder_fixture();

    assert_eq!(imported.num_user_inputs, 1);

    let ops: Vec<&TraceOp> = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op(), TraceOp::Input | TraceOp::Constant { .. }))
        .map(TraceNode::op)
        .collect();

    // BiLSTM zero constants are TraceOp::Constant (filtered out).
    // Remaining: 4 Transpose + fwd_lstm + flip + bwd_lstm + flip + cat + 1 Linear = 10.
    let counts = [
        (
            "Transpose",
            ops.iter()
                .filter(|op| matches!(op, TraceOp::Transpose { .. }))
                .count(),
        ),
        (
            "Lstm",
            ops.iter()
                .filter(|op| matches!(op, TraceOp::Lstm { .. }))
                .count(),
        ),
        (
            "Flip",
            ops.iter()
                .filter(|op| matches!(op, TraceOp::Flip { .. }))
                .count(),
        ),
        (
            "Cat",
            ops.iter()
                .filter(|op| matches!(op, TraceOp::Cat { .. }))
                .count(),
        ),
        (
            "Linear",
            ops.iter()
                .filter(|op| matches!(op, TraceOp::Linear { .. }))
                .count(),
        ),
    ];

    assert_eq!(counts[0].1, 4, "expected 4 Transpose ops");
    assert_eq!(counts[1].1, 2, "expected 2 LSTM ops (fwd + bwd)");
    assert_eq!(counts[2].1, 2, "expected 2 Flip ops");
    assert_eq!(counts[3].1, 1, "expected 1 Cat op (BiLSTM merge)");
    assert_eq!(counts[4].1, 1, "expected 1 Linear op");

    eprintln!(
        "[Kokoro encoder] {} total ops (excl Input/Constant)",
        ops.len()
    );
}

/// Kokoro encoder IBP bounds: import → trace_to_graph → NY IBP.
///
/// Part of #2306 (nn::convert() one-function pipeline).
#[test]
#[cfg(feature = "verify")]
fn test_import_kokoro_encoder_ibp_bounds() {
    use ndarray::{ArrayD, IxDyn};

    let imported = import_kokoro_encoder_fixture();

    let gn = nn_verify::trace_to_graph_model(&imported.graph)
        .expect("trace_to_graph_model must succeed for Kokoro encoder")
        .graph;

    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .expect("imported graph must have an Input node");
    let shape = input_node.output_shape();
    assert_eq!(shape, &[1, 8, 4], "Kokoro encoder input shape");

    let lower = ArrayD::from_elem(IxDyn(shape), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 1.0_f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP propagation must succeed for Kokoro encoder");

    let (out_lo, out_hi) = output.lower_upper();

    for (idx, (&lo, &hi)) in out_lo.iter().zip(out_hi.iter()).enumerate() {
        assert!(lo.is_finite(), "lower bound at idx {idx} not finite: {lo}");
        assert!(hi.is_finite(), "upper bound at idx {idx} not finite: {hi}");
        assert!(lo <= hi, "bounds inverted at idx {idx}: lo={lo} > hi={hi}");
    }

    let max_width = out_hi
        .iter()
        .zip(out_lo.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    assert!(
        max_width > 0.0,
        "IBP bounds should be non-trivial, max_width={max_width}"
    );

    eprintln!(
        "[Kokoro encoder IBP] {} output elements, max_width={max_width:.4}",
        out_lo.len()
    );
}

/// convert() on Kokoro encoder succeeds: LSTM decomposed to primitives (#2306).
///
/// LSTM expansion decomposes to Linear+Sigmoid+Tanh+BinaryMul+BinaryAdd,
/// all of which have working MSL codegen. convert() produces a CompiledModel
/// with L1 (Kani) and L2 (IBP) proof layers.
///
/// Part of #2306 (nn::convert() one-function pipeline).
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_convert_kokoro_encoder_lstm_decomposed() {
    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let dir = std::env::temp_dir().join(format!("nn_convert_enc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/kokoro_encoder_mini.json"),
    )
    .unwrap();
    let weights_path = write_kokoro_encoder_weights(&dir);

    let result = convert(&graph_path, &weights_path, None, &cache);
    let _ = std::fs::remove_dir_all(&dir);

    let convert_result = result.expect("convert() should succeed with LSTM decomposition");

    // L1: Kani not run during convert() (populated by Prover separately).
    assert!(convert_result.proof.kernel_safety.is_none());
    // L3: No reference trace provided.
    assert!(convert_result.proof.reference_parity.is_none());

    eprintln!("[Kokoro encoder] convert() succeeded with LSTM decomposition");
}

/// Convert encoder fixture and return result with L2 bounds verified.
#[cfg(all(feature = "metal", feature = "verify", target_os = "macos"))]
fn convert_encoder_with_l2(cache: &nn_metal::PipelineCache) -> ConvertResult {
    let dir = std::env::temp_dir().join(format!("nn_composed_enc_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("graph.json"),
        include_str!("../test_data/kokoro_encoder_mini.json"),
    )
    .unwrap();
    let weights = write_kokoro_encoder_weights(&dir);
    let result =
        convert(&dir.join("graph.json"), &weights, None, cache).expect("encoder convert()");
    let _ = std::fs::remove_dir_all(&dir);
    let l2 = result
        .proof
        .composition_bounds
        .as_ref()
        .expect("encoder L2");
    assert!(l2.propagation_ok, "encoder IBP must propagate");
    result
}

/// Convert decoder fixture and return result with L2 bounds verified.
#[cfg(all(feature = "metal", feature = "verify", target_os = "macos"))]
fn convert_decoder_with_l2(cache: &nn_metal::PipelineCache) -> ConvertResult {
    let dir = std::env::temp_dir().join(format!("nn_composed_dec_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("graph.json"),
        include_str!("../test_data/kokoro_decoder_mini.json"),
    )
    .unwrap();
    let weights = super::write_kokoro_decoder_weights(&dir);
    let result =
        convert(&dir.join("graph.json"), &weights, None, cache).expect("decoder convert()");
    let _ = std::fs::remove_dir_all(&dir);
    let l2 = result
        .proof
        .composition_bounds
        .as_ref()
        .expect("decoder L2");
    assert!(l2.propagation_ok, "decoder IBP must propagate");
    result
}

/// Compose encoder output bounds into decoder input bounds via per-channel reduction.
///
/// Encoder output: `[1, C, T_enc]`. Decoder input: `[1, C, T_dec]`.
/// Per-channel min/max across time is a sound over-approximation (length_regulate
/// only replicates/selects values within the same channel).
#[cfg(all(feature = "metal", feature = "verify", target_os = "macos"))]
fn compose_bounds_per_channel(
    enc_output: &nn_verify::BoundedTensor,
    dec_time: usize,
) -> nn_verify::BoundedTensor {
    use ndarray::{ArrayD, IxDyn};

    let (enc_lo, enc_hi) = enc_output.lower_upper();
    let channels = enc_lo.shape()[1];
    let enc_time = enc_lo.shape()[2];

    let mut ch_lo = vec![f32::INFINITY; channels];
    let mut ch_hi = vec![f32::NEG_INFINITY; channels];
    for c in 0..channels {
        for t in 0..enc_time {
            ch_lo[c] = ch_lo[c].min(enc_lo[[0, c, t]]);
            ch_hi[c] = ch_hi[c].max(enc_hi[[0, c, t]]);
        }
    }

    let mut lo = ArrayD::zeros(IxDyn(&[1, channels, dec_time]));
    let mut hi = ArrayD::zeros(IxDyn(&[1, channels, dec_time]));
    for c in 0..channels {
        for t in 0..dec_time {
            lo[[0, c, t]] = ch_lo[c];
            hi[[0, c, t]] = ch_hi[c];
        }
    }
    nn_verify::BoundedTensor::new(lo, hi).expect("valid composed bounds")
}

/// Composed Kokoro pipeline: encoder convert() + decoder convert() + bounds composition.
///
/// Demonstrates the full `convert()` one-function pipeline on both halves of Kokoro TTS,
/// then composes their IBP bounds: encoder output bounds → decoder input bounds.
///
/// Part of #2306 (nn::convert() one-function pipeline).
#[test]
#[cfg(all(feature = "metal", feature = "verify", target_os = "macos"))]
fn test_convert_kokoro_composed_encoder_decoder() {
    use ndarray::{ArrayD, IxDyn};

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let enc_result = convert_encoder_with_l2(&cache);
    let dec_result = convert_decoder_with_l2(&cache);

    // Propagate encoder IBP to get actual output bounds.
    let enc_gn = nn_verify::trace_to_graph_model(&enc_result.graph.graph)
        .expect("encoder trace_to_graph_model")
        .graph;
    let enc_input = {
        let lo = ArrayD::from_elem(IxDyn(&[1, 8, 4]), -1.0_f32);
        let hi = ArrayD::from_elem(IxDyn(&[1, 8, 4]), 1.0_f32);
        nn_verify::BoundedTensor::new(lo, hi).unwrap()
    };
    let enc_output = enc_gn.propagate_ibp(&enc_input).expect("encoder IBP");

    // Compose: encoder per-channel output bounds → decoder input bounds [1, 8, 16].
    let composed_input = compose_bounds_per_channel(&enc_output, 16);

    // Propagate decoder with encoder-derived bounds.
    let dec_gn = nn_verify::trace_to_graph_model(&dec_result.graph.graph)
        .expect("decoder trace_to_graph_model")
        .graph;
    let composed_output = dec_gn
        .propagate_ibp(&composed_input)
        .expect("decoder IBP with encoder-derived bounds");

    let (comp_lo, comp_hi) = composed_output.lower_upper();
    let composed_width = comp_hi
        .iter()
        .zip(comp_lo.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);

    for (idx, (&lo, &hi)) in comp_lo.iter().zip(comp_hi.iter()).enumerate() {
        assert!(lo.is_finite(), "composed lower[{idx}] not finite: {lo}");
        assert!(hi.is_finite(), "composed upper[{idx}] not finite: {hi}");
        assert!(
            lo <= hi,
            "composed bounds inverted at {idx}: lo={lo} > hi={hi}"
        );
    }

    // Composed bounds should be tighter than default [-1, 1] decoder bounds.
    let default_width = dec_result
        .proof
        .composition_bounds
        .as_ref()
        .and_then(|l2| l2.output_width)
        .unwrap_or(f32::INFINITY);
    assert!(
        composed_width <= default_width + 1e-6,
        "composed ({composed_width:.4}) should be <= default ({default_width:.4})"
    );

    eprintln!(
        "[Kokoro composed] encoder width: {:.4}, decoder default: {default_width:.4}, \
         composed: {composed_width:.4}",
        enc_result
            .proof
            .composition_bounds
            .as_ref()
            .and_then(|l2| l2.output_width)
            .unwrap_or(0.0),
    );
}
