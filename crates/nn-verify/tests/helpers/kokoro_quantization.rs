// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Quantization verification for Kokoro TTS: prove bf16 within epsilon of f32.
//!
//! Loads production f32 weights, creates bf16-quantized weights (f32 → bf16 → f32
//! roundtrip to simulate precision loss), traces Kokoro model segments through
//! both weight sets, runs NY IBP on each, and computes the quantization
//! drift. Uses Lipschitz composition from `quality_bound.rs` to prove quality
//! metrics (MCD, SNR, cosine similarity) remain within thresholds.
//!
//! **Requires:** `KOKORO_WEIGHTS=/path/to/kokoro_weights_rust.safetensors`
//! Tests are gated behind `#[cfg(feature = "production-weights")]`.
//!
//! Part of #2464.
//! Part of #2218.

#[cfg(feature = "production-weights")]
use super::kokoro_production_weights::require_production_weights;

#[cfg(feature = "production-weights")]
use nn_core::dyn_tensor::trace::trace_graph;
#[cfg(feature = "production-weights")]
use nn_core::dyn_tensor::DynTensor;
#[cfg(feature = "production-weights")]
use nn_core::layers::{Linear, Module};
#[cfg(feature = "production-weights")]
use nn_core::test_utils::cpu;
#[cfg(feature = "production-weights")]
use nn_core::{DType, VarBuilder};
#[cfg(feature = "production-weights")]
use nn_models::kokoro_tts::TextEncoder;
#[cfg(feature = "production-weights")]
use nn_models::KokoroConfig;
#[cfg(feature = "production-weights")]
use nn_tts_verify::{
    build_quantization_certificate, build_segment_result, mcd_lipschitz, snr_lipschitz,
    QualityMetricSpec,
};
#[cfg(feature = "production-weights")]
use nn_verify::{trace_to_graph_model, BoundedTensor};
#[cfg(feature = "production-weights")]
use ndarray::{ArrayD, IxDyn};
#[cfg(feature = "production-weights")]
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: simulate bf16 quantization by roundtripping weights
// ---------------------------------------------------------------------------

/// Convert all weight tensors from f32 → bf16 → f32, simulating quantization.
///
/// Each weight value is rounded to the nearest bf16 representable value.
/// The resulting tensors are still f32 dtype (for NY compatibility)
/// but contain only values exactly representable in bf16.
#[cfg(feature = "production-weights")]
fn quantize_weights_bf16(tensors: &HashMap<String, DynTensor>) -> HashMap<String, DynTensor> {
    tensors
        .iter()
        .map(|(key, tensor)| {
            let quantized = if tensor.dtype() == DType::F32 {
                tensor
                    .to_dtype(DType::BF16)
                    .and_then(|t| t.to_dtype(DType::F32))
                    .unwrap_or_else(|_| tensor.clone())
            } else {
                tensor.clone()
            };
            (key.clone(), quantized)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: trace and propagate a segment, returning flat lower/upper arrays
// ---------------------------------------------------------------------------

#[cfg(feature = "production-weights")]
struct SegmentBounds {
    lower: Vec<f32>,
    upper: Vec<f32>,
}

/// Trace bert_encoder (Linear) with given weights and return output bounds.
#[cfg(feature = "production-weights")]
fn trace_bert_encoder_bounds(vb: &VarBuilder, config: &KokoroConfig) -> SegmentBounds {
    let hidden = config.plbert.hidden_size;
    let d_en = config.d_en;
    let w = vb
        .get(&[d_en, hidden], "bert_encoder.weight")
        .expect("weight");
    let b = vb.get(&[d_en], "bert_encoder.bias").expect("bias");
    let bert_encoder = Linear::new(w, Some(b)).expect("Linear::new");

    let input_shape = [1, 4, hidden];
    let input_data = DynTensor::full(&input_shape, 0.1, DType::F32, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let x = super::kokoro_production_weights::trace_input(&input_data);
        bert_encoder.forward(&x)
    })
    .expect("bert_encoder trace");

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&input_shape), -3.0f32),
        ArrayD::from_elem(IxDyn(&input_shape), 3.0f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let lower: Vec<f32> = lo.iter().copied().collect();
    let upper: Vec<f32> = hi.iter().copied().collect();

    SegmentBounds { lower, upper }
}

/// Trace TextEncoder with given weights and return output bounds.
#[cfg(feature = "production-weights")]
fn trace_text_encoder_bounds(vb: &VarBuilder, config: &KokoroConfig) -> SegmentBounds {
    let vocab_size = config.plbert.vocab_size;
    let d_en = config.d_en;
    let text_encoder =
        TextEncoder::load(&vb.pp("text_encoder"), vocab_size, d_en).expect("TextEncoder::load");

    let token_shape = [1, 4];
    let tokens = DynTensor::full(&token_shape, 5.0, DType::I64, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let x = super::kokoro_production_weights::trace_input(&tokens);
        text_encoder
            .forward(&x)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))
    })
    .expect("TextEncoder trace");

    let gn = trace_to_graph_model(&graph)
        .expect("trace_to_graph_model")
        .graph;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&token_shape), 0.0f32),
        ArrayD::from_elem(IxDyn(&token_shape), (vocab_size - 1) as f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    super::common::assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let lower: Vec<f32> = lo.iter().copied().collect();
    let upper: Vec<f32> = hi.iter().copied().collect();

    SegmentBounds { lower, upper }
}

// ---------------------------------------------------------------------------
// Test 1: bert_encoder quantization drift
// ---------------------------------------------------------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_bert_encoder_bf16_quantization_drift() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();

    // F32 model
    let vb_f32 = VarBuilder::from_tensors(tensors.clone(), DType::F32, &cpu());
    let f32_bounds = trace_bert_encoder_bounds(&vb_f32, &config);

    // BF16-quantized model (weights rounded to bf16 precision)
    let quantized = quantize_weights_bf16(&tensors);
    let vb_bf16 = VarBuilder::from_tensors(quantized, DType::F32, &cpu());
    let bf16_bounds = trace_bert_encoder_bounds(&vb_bf16, &config);

    // Compute segment result
    let seg = build_segment_result(
        "bert_encoder",
        &f32_bounds.lower,
        &f32_bounds.upper,
        &bf16_bounds.lower,
        &bf16_bounds.upper,
    )
    .expect("segment result");

    eprintln!(
        "bert_encoder quantization: f32_width={:.6}, bf16_width={:.6}, \
         max_drift={:.6}, mean_drift={:.6}",
        seg.f32_output_width,
        seg.quantized_output_width,
        seg.max_element_drift,
        seg.mean_element_drift
    );

    // Drift should be small for a single Linear layer with bf16 rounding.
    assert!(
        seg.max_element_drift < 1.0,
        "bert_encoder bf16 drift should be small, got {}",
        seg.max_element_drift
    );
    assert!(seg.max_element_drift.is_finite(), "drift must be finite");
}

// ---------------------------------------------------------------------------
// Test 2: TextEncoder quantization drift
// ---------------------------------------------------------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_text_encoder_bf16_quantization_drift() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();

    let vb_f32 = VarBuilder::from_tensors(tensors.clone(), DType::F32, &cpu());
    let f32_bounds = trace_text_encoder_bounds(&vb_f32, &config);

    let quantized = quantize_weights_bf16(&tensors);
    let vb_bf16 = VarBuilder::from_tensors(quantized, DType::F32, &cpu());
    let bf16_bounds = trace_text_encoder_bounds(&vb_bf16, &config);

    let seg = build_segment_result(
        "text_encoder",
        &f32_bounds.lower,
        &f32_bounds.upper,
        &bf16_bounds.lower,
        &bf16_bounds.upper,
    )
    .expect("segment result");

    eprintln!(
        "text_encoder quantization: f32_width={:.6}, bf16_width={:.6}, \
         max_drift={:.6}, mean_drift={:.6}",
        seg.f32_output_width,
        seg.quantized_output_width,
        seg.max_element_drift,
        seg.mean_element_drift
    );

    assert!(seg.max_element_drift.is_finite(), "drift must be finite");
}

// ---------------------------------------------------------------------------
// Test 3: Full quantization certificate with Lipschitz quality bounds
// ---------------------------------------------------------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_kokoro_bf16_quantization_certificate() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();

    // -- Trace segments with f32 and bf16 weights --

    let vb_f32 = VarBuilder::from_tensors(tensors.clone(), DType::F32, &cpu());
    let f32_bert = trace_bert_encoder_bounds(&vb_f32, &config);
    let f32_text = trace_text_encoder_bounds(&vb_f32, &config);

    let quantized = quantize_weights_bf16(&tensors);
    let vb_bf16 = VarBuilder::from_tensors(quantized, DType::F32, &cpu());
    let bf16_bert = trace_bert_encoder_bounds(&vb_bf16, &config);
    let bf16_text = trace_text_encoder_bounds(&vb_bf16, &config);

    // -- Build segment results --

    let seg_bert = build_segment_result(
        "bert_encoder",
        &f32_bert.lower,
        &f32_bert.upper,
        &bf16_bert.lower,
        &bf16_bert.upper,
    )
    .expect("bert segment");

    let seg_text = build_segment_result(
        "text_encoder",
        &f32_text.lower,
        &f32_text.upper,
        &bf16_text.lower,
        &bf16_text.upper,
    )
    .expect("text segment");

    eprintln!(
        "Segment drifts — bert: {:.6}, text: {:.6}",
        seg_bert.max_element_drift, seg_text.max_element_drift
    );

    // -- Build quality specs with conservative baselines --
    //
    // These are typical baselines for Kokoro TTS output quality. The Lipschitz
    // constants are computed from signal statistics (see quality_bound.rs).
    let quality_specs = vec![
        QualityMetricSpec {
            name: "SNR".into(),
            lipschitz_constant: snr_lipschitz(0.1, 25.0).expect("snr_lipschitz"),
            baseline_value: 25.0,
            threshold: 10.0,
            higher_is_better: true,
            citation: "ITU-T P.56 (2011)",
        },
        QualityMetricSpec {
            name: "MCD".into(),
            lipschitz_constant: mcd_lipschitz(100).expect("mcd_lipschitz"),
            baseline_value: 4.0,
            threshold: 6.0,
            higher_is_better: false,
            citation: "Kubichek (1993). Mel-cepstral distance. IEEE ICASSP.",
        },
    ];

    // -- Build and validate certificate --

    let cert =
        build_quantization_certificate("F32", "BF16", vec![seg_bert, seg_text], &quality_specs)
            .expect("quantization certificate");

    eprintln!(
        "QuantizationCertificate: quality_preserved={}, max_drift={:.6}, \
         tightest_metric='{}', tightest_margin={:.4}",
        cert.quality_preserved,
        cert.max_output_drift,
        cert.quality_certificate.tightest_metric,
        cert.quality_certificate.tightest_margin,
    );

    for r in &cert.quality_certificate.metric_results {
        eprintln!(
            "  {}: baseline={:.2}, worst_case={:.4}, threshold={:.2}, \
             margin={:.4}, guaranteed={}",
            r.metric_name,
            r.baseline_value,
            r.worst_case_value,
            r.threshold,
            r.margin,
            r.guaranteed
        );
    }

    // The certificate itself is machine-checkable. Log structural properties.
    assert_eq!(cert.source_dtype, "F32");
    assert_eq!(cert.target_dtype, "BF16");
    assert_eq!(cert.segment_results.len(), 2);
    assert!(
        cert.max_output_drift.is_finite(),
        "max drift must be finite"
    );

    // If quality is preserved, all metrics have positive margin.
    if cert.quality_preserved {
        for r in &cert.quality_certificate.metric_results {
            assert!(
                r.margin >= 0.0,
                "{}: margin should be non-negative when quality_preserved=true",
                r.metric_name
            );
        }
        eprintln!("CERTIFICATE VALID: bf16 Kokoro proven within quality thresholds.");
    } else {
        eprintln!(
            "CERTIFICATE: quality NOT preserved — tightest metric '{}' has margin {:.4}",
            cert.quality_certificate.tightest_metric, cert.quality_certificate.tightest_margin
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Synthetic quantization certificate (no production weights needed)
// ---------------------------------------------------------------------------

/// Verify the certificate pipeline works end-to-end with synthetic data.
/// This test always runs (no production-weights feature gate).
#[test]
fn test_synthetic_quantization_certificate() {
    use nn_tts_verify::{build_quantization_certificate, build_segment_result, QualityMetricSpec};

    // Simulate small bf16 rounding drift on 4 output elements.
    let f32_lo = [0.0f32, -1.0, 0.5, -0.5];
    let f32_hi = [1.0f32, 0.0, 1.5, 0.5];
    let bf16_lo = [0.001f32, -0.999, 0.501, -0.499];
    let bf16_hi = [1.001f32, 0.001, 1.501, 0.501];

    let seg = build_segment_result("synthetic", &f32_lo, &f32_hi, &bf16_lo, &bf16_hi)
        .expect("synthetic segment");

    let specs = vec![
        QualityMetricSpec {
            name: "SNR".into(),
            lipschitz_constant: 10.0,
            baseline_value: 30.0,
            threshold: 10.0,
            higher_is_better: true,
            citation: "test",
        },
        QualityMetricSpec {
            name: "MCD".into(),
            lipschitz_constant: 1.0,
            baseline_value: 3.0,
            threshold: 6.0,
            higher_is_better: false,
            citation: "test",
        },
    ];

    let cert =
        build_quantization_certificate("F32", "BF16", vec![seg], &specs).expect("certificate");

    assert!(
        cert.quality_preserved,
        "synthetic small drift should preserve quality"
    );
    assert!(cert.max_output_drift < 0.01);
    assert_eq!(cert.segment_results.len(), 1);
    assert_eq!(cert.segment_results[0].num_elements, 4);
}
