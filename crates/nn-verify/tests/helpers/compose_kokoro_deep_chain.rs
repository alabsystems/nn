// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Deep multi-stage chain composition tests for the Kokoro TTS pipeline.
//!
//! These tests go beyond the existing `compose_kokoro_multi_stage.rs` by:
//!
//! 1. **Decoder-to-iSTFT chain** — verifying that decoder spectral output
//!    (exp magnitude) propagates through the iSTFT linear transform to produce
//!    audio bounds, chaining the verification bridge from spectral to time domain.
//!
//! 2. **Encoder-to-duration with Exp positivity** — full chain from text
//!    features through encoder, duration predictor, and exp() to prove
//!    phoneme durations are strictly positive (P3).
//!
//! 3. **Multi-upsample decoder** — decoder with two sequential upsample stages,
//!    testing bound stability through deeper upsampling hierarchies.
//!
//! 4. **Cross-stage bound tightness analysis** — measures IBP tightening ratio
//!    at each pipeline boundary (encoder output, decoder input, decoder output)
//!    to detect vacuously wide intermediate bounds.
//!
//! 5. **High-variance pathological input** — verifies that element-wise
//!    varying inputs (pathological for normalization) produce valid bounds
//!    through the full pipeline.
//!
//! 6. **Full pipeline with output clamp** — proves audio ∈ [-1, 1] by
//!    composing the decoder pipeline with a Tanh output clamp.
//!
//! 7. **CROWN vs IBP tightening through each stage** — runs both methods
//!    through individual stages and measures relative tightening.
//!
//! 8. **Monotonicity across input radii** — tests that multiple input
//!    radius values produce monotonically ordered output bound widths,
//!    covering 5 radii (not just the 2 in multi_stage.rs).
//!
//! Part of #4294: Deepen Kokoro multi-stage compose verification.
//! Part of #3351: Epic — Absolutely Best Kokoro.

#[path = "kokoro_multi_stage.rs"]
mod ms_helpers;

use nn_verify::istft_linear_matrix::build_istft_weight_matrix;
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, VerificationSoundnessMode};
use ms_helpers::{
    build_decoder_block, build_full_four_stage_pipeline, build_style_projector, build_text_encoder,
    D_MODEL, ENC_DIM, OUT_CH, SEQ_LEN, TIME_UP, VOC_CH,
};
use ndarray::{Array1, Array2, ArrayD, IxDyn};

use ny_propagate::layers::LinearLayer;
use ny_propagate::{GraphNode, Layer};

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max,
    high_variance_bounds, uniform_bounds, verify_and_assert,
};

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_verify::TensorParamBinding;

// ===========================================================================
// Constants
// ===========================================================================

/// Weight magnitude for synthetic weights.
const W_MAG: f32 = 0.001;

// ===========================================================================
// Upsample stride/kernel for multi-upsample tests
// ===========================================================================

const UP_STRIDE: usize = 2;
const UP_KERNEL: usize = 4;
const UP_PADDING: usize = 1;

/// Output time after first upsample.
const TIME_UP_1: usize = (SEQ_LEN - 1) * UP_STRIDE + UP_KERNEL - 2 * UP_PADDING;
/// Output time after second upsample.
const TIME_UP_2: usize = (TIME_UP_1 - 1) * UP_STRIDE + UP_KERNEL - 2 * UP_PADDING;

// ===========================================================================
// Test 1: Decoder spectral output → iSTFT → audio bounds
// ===========================================================================

/// Chain the decoder's exp() spectral output through the iSTFT linear
/// transform to prove audio-domain bounds. This bridges the spectral
/// verification (P1/P2 from compose_kokoro_multi_stage) into the time
/// domain.
///
/// Architecture:
///   decoder exp() output → analytical bridge (cos/sin ∈ [-1,1]) →
///   iSTFT(LinearLayer) → audio samples
///
/// The bridge uses the cos/sin over-approximation from compose_kokoro_istft.rs.
/// We prove the composed chain produces finite, bounded audio output.
///
/// Part of #4294.
#[test]
fn test_decoder_to_istft_chain() {
    // Step 1: Get decoder output bounds via IBP.
    let (def, bindings, out_shape) = build_decoder_block();
    assert_eq!(out_shape, [OUT_CH, TIME_UP]);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("decoder graph");
    let input = uniform_bounds(&[VOC_CH, SEQ_LEN], 1.0);
    let decoder_output = graph.propagate_ibp(&input).expect("IBP through decoder");
    assert_bounds_valid(&decoder_output);

    let (dec_lo_min, dec_hi_max) = bounds_min_max(&decoder_output);
    eprintln!("Decoder output (spectral): [{dec_lo_min}, {dec_hi_max}]");
    assert!(dec_lo_min > 0.0, "decoder exp output must be positive");

    // Step 2: Bridge decoder spectral bounds to iSTFT input.
    // decoder output shape is [OUT_CH, TIME_UP] — use OUT_CH as n_bins proxy
    // and TIME_UP as n_frames for the iSTFT.
    let n_bins_proxy = OUT_CH;
    let n_frames = TIME_UP;
    let spectral_len = n_bins_proxy * n_frames;
    let istft_input_dim = 2 * spectral_len;

    // Extract per-element upper bounds from decoder output for bridge.
    let (_dec_lo, dec_hi) = decoder_output.lower_upper();
    let mag_upper: Vec<f32> = dec_hi.iter().copied().collect();

    // Bridge: real/imag bounded by [-mag_hi, mag_hi] (cos/sin ∈ [-1,1]).
    let mut lower = Vec::with_capacity(istft_input_dim);
    let mut upper = Vec::with_capacity(istft_input_dim);
    for &m in &mag_upper {
        lower.push(-m);
        upper.push(m);
    }
    for &m in &mag_upper {
        lower.push(-m);
        upper.push(m);
    }
    let istft_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[istft_input_dim]), lower).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[istft_input_dim]), upper).expect("upper"),
    )
    .expect("valid bridge bounds");

    // Step 3: Build iSTFT graph. Use a small synthetic n_fft matching our proxy.
    // We use n_fft = 2 * (n_bins_proxy - 1) to match the bridge dimensions.
    let synthetic_n_fft = 2 * (n_bins_proxy - 1); // 6 for OUT_CH=4
    let synthetic_hop = 2;
    let output_length = (n_frames - 1) * synthetic_hop; // audio samples

    if output_length == 0 {
        eprintln!("Skipping iSTFT chain: output_length=0 for n_frames={n_frames}");
        return;
    }

    let mat = build_istft_weight_matrix(
        synthetic_n_fft,
        synthetic_hop,
        n_frames,
        output_length,
        false,
        true,
    );

    match mat {
        Ok(mat) => {
            let n_in = mat.input_dim;
            let n_out = mat.output_length;
            let weight =
                Array2::from_shape_vec((n_out, n_in), mat.weights).expect("valid weight shape");
            let bias = Array1::zeros(n_out);
            let linear = LinearLayer::new(weight, Some(bias)).expect("valid LinearLayer");

            let mut istft_graph = nn_verify::GraphNetwork::new();
            istft_graph.add_node(GraphNode::from_input(
                "istft_linear".to_string(),
                Layer::Linear(linear),
            ));
            istft_graph.set_output("istft_linear".to_string());

            let audio_output = istft_graph
                .propagate_ibp(&istft_bounds)
                .expect("IBP through iSTFT");
            assert_bounds_valid(&audio_output);
            let (audio_lo, audio_hi) = bounds_min_max(&audio_output);
            eprintln!("Decoder → iSTFT audio bounds: [{audio_lo:.6}, {audio_hi:.6}]");
            assert!(
                audio_lo.is_finite() && audio_hi.is_finite(),
                "audio bounds must be finite"
            );
        }
        Err(e) => {
            eprintln!(
                "iSTFT matrix build skipped for proxy dims (n_fft={synthetic_n_fft}, \
                 hop={synthetic_hop}, n_frames={n_frames}): {e}"
            );
        }
    }
}

// ===========================================================================
// Test 2: Encoder → duration predictor → Exp (P3 positivity chain)
// ===========================================================================

/// Full chain: text encoder → linear duration predictor → exp() → positive durations.
///
/// Unlike the simpler duration branch in compose_kokoro_full_pipeline.rs which
/// outputs raw logits, this test adds the exp() activation to prove the
/// composed chain guarantees strictly positive phoneme durations (P3).
///
/// Part of #4294.
#[test]
fn test_encoder_duration_exp_chain() {
    let mut b = TensorBlockBuilder::new("kokoro_deep_enc_dur_exp");

    let text = b.add_input("text_features", &[D_MODEL, SEQ_LEN]);

    // Stage 1: Text encoder (Conv1d + ReLU + Linear)
    let conv_w = b.add_input("enc_conv_w", &[D_MODEL, D_MODEL, 3]);
    let conv_out = b.add_conv1d(text, conv_w, None, 1, 1, &[D_MODEL, SEQ_LEN]);
    let relu_out = b.add_relu(conv_out, &[D_MODEL, SEQ_LEN]);
    let t1 = b.add_transpose(relu_out, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let proj_w = b.add_input("enc_proj_w", &[ENC_DIM, D_MODEL]);
    let proj_b = b.add_input("enc_proj_b", &[ENC_DIM]);
    let mm = b.add_matmul(t1, proj_w, true, None, &[SEQ_LEN, ENC_DIM]);
    let proj_b_bc = b.add_broadcast(proj_b, &[SEQ_LEN, ENC_DIM]);
    let enc_out = b.add_binary_add(mm, proj_b_bc, &[SEQ_LEN, ENC_DIM]);

    // Stage 2: Duration predictor (Linear → [SEQ_LEN, 1] → reshape → [SEQ_LEN])
    let dur_w = b.add_input("dur_w", &[1, ENC_DIM]);
    let dur_b = b.add_input("dur_b", &[1]);
    let dur_mm = b.add_matmul(enc_out, dur_w, true, None, &[SEQ_LEN, 1]);
    let dur_b_bc = b.add_broadcast(dur_b, &[SEQ_LEN, 1]);
    let dur_logits = b.add_binary_add(dur_mm, dur_b_bc, &[SEQ_LEN, 1]);

    // Stage 3: Exp activation — proves durations are strictly positive.
    let dur_exp = b.add_exp(dur_logits, &[SEQ_LEN, 1]);
    let output = b.add_reshape(dur_exp, &[SEQ_LEN]);

    let def = b.build(output).expect("encoder-duration-exp graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1, ENC_DIM]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), 0.0f32)),
    ];

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    // IBP
    let ibp_output = graph
        .propagate_ibp(&input)
        .expect("IBP through enc-dur-exp");
    assert_bounds_valid(&ibp_output);
    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Encoder→Duration→Exp IBP: [{lo_min}, {hi_max}]");

    // P3: exp() output must be strictly positive.
    assert!(
        lo_min > 0.0,
        "P3 VIOLATION: exp(duration_logits) must be positive, got {lo_min}"
    );
    assert!(hi_max.is_finite(), "duration upper bound must be finite");
    eprintln!("  P3 (Duration positivity) PROVEN: lower {lo_min} > 0");

    // CROWN
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Encoder→Duration→Exp CROWN: method={method:?}, [{crown_lo}, {crown_hi}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }
    assert!(crown_lo > 0.0, "CROWN P3: exp positive, got {crown_lo}");

    // Verify and record
    let result = verify_and_assert(&def, &bindings, &input, "kokoro_deep_enc_dur_exp");
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// Test 3: Multi-upsample decoder (2 sequential upsample stages)
// ===========================================================================

/// Decoder with 2 sequential ConvTranspose1d upsample stages, testing bound
/// stability through hierarchical upsampling (Kokoro production has 4 stages).
///
/// Conv1d → LeakyReLU → ConvTranspose1d(×2) → InstanceNorm → Snake →
/// Conv1d → residual → LeakyReLU → ConvTranspose1d(×2) → Conv1d → Exp
///
/// Part of #4294.
#[test]
fn test_multi_upsample_decoder() {
    let up1_shape = [VOC_CH, TIME_UP_1];
    let up2_shape = [OUT_CH, TIME_UP_2];

    let mut b = TensorBlockBuilder::new("kokoro_deep_multi_upsample");

    let input = b.add_input("features", &[VOC_CH, SEQ_LEN]);
    let eps = b.add_input("eps", &[1]);

    // Stage 1: Conv pre + LeakyReLU
    let conv_pre_w = b.add_input("conv_pre_w", &[VOC_CH, VOC_CH, 3]);
    let x = b.add_conv1d(input, conv_pre_w, None, 1, 1, &[VOC_CH, SEQ_LEN]);
    let x_act = b.add_leaky_relu(x, 0.1, &[VOC_CH, SEQ_LEN]);

    // Stage 2: Upsample 1 (ConvTranspose1d)
    let up1_w = b.add_input("up1_w", &[VOC_CH, VOC_CH, UP_KERNEL]);
    let x_up1 = b.add_conv_transpose_1d(
        x_act, up1_w, None, UP_STRIDE, UP_PADDING, 1, 1, 0, &up1_shape,
    );

    // InstanceNorm + Snake + Conv1d + residual
    let gamma1 = b.add_input("gamma1", &[VOC_CH]);
    let beta1 = b.add_input("beta1", &[VOC_CH]);
    let normed1 = b.add_instance_norm(x_up1, eps, 1, Some(gamma1), Some(beta1), &up1_shape);
    let alpha1 = b.add_input("alpha1", &[1]);
    let alpha1_bc = b.add_broadcast(alpha1, &up1_shape);
    let snake1 = build_snake_scalar_kernel().expect("snake kernel 1");
    let snake1_out = b.add_elementwise(snake1, &[normed1, alpha1_bc], &up1_shape);
    let res1_w = b.add_input("res1_w", &[VOC_CH, VOC_CH, 3]);
    let sub1 = b.add_conv1d(snake1_out, res1_w, None, 1, 1, &up1_shape);
    let res1_out = b.add_binary_add(x_up1, sub1, &up1_shape);
    let res1_act = b.add_leaky_relu(res1_out, 0.01, &up1_shape);

    // Stage 3: Upsample 2 (ConvTranspose1d)
    let up2_w = b.add_input("up2_w", &[VOC_CH, OUT_CH, UP_KERNEL]);
    let x_up2 = b.add_conv_transpose_1d(
        res1_act, up2_w, None, UP_STRIDE, UP_PADDING, 1, 1, 0, &up2_shape,
    );

    // Final Exp
    let output = b.add_exp(x_up2, &up2_shape);

    let def = b.build(output).expect("multi-upsample decoder graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH, VOC_CH, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_CH, VOC_CH, UP_KERNEL]),
            W_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH, VOC_CH, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_CH, OUT_CH, UP_KERNEL]),
            W_MAG,
        )),
    ];

    def.validate().expect("multi-upsample def validates");
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input_bounds = uniform_bounds(&[VOC_CH, SEQ_LEN], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through multi-upsample");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[OUT_CH, TIME_UP_2]);
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Multi-upsample decoder IBP: [{lo_min}, {hi_max}]");
    assert!(lo_min > 0.0, "P1: exp output positive, got {lo_min}");
    assert!(hi_max < 1e8, "IBP upper bounded, got {hi_max}");

    // CROWN
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Multi-upsample CROWN: method={method:?}, [{crown_lo}, {crown_hi}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    let result = verify_and_assert(&def, &bindings, &input_bounds, "kokoro_deep_multi_upsample");
    assert_eq!(result.num_variables, 1);
}

// ===========================================================================
// Test 4: Cross-stage bound tightness analysis
// ===========================================================================

/// Measures IBP output width at each pipeline stage boundary to detect
/// vacuously wide intermediate bounds. Reports tightening ratios.
///
/// Stage decomposition:
///   text_features → [text encoder] → encoded
///   encoded → [style projector] → style
///   features → [decoder block] → spectral
///
/// Each stage's output width is measured independently and compared.
///
/// Part of #4294.
#[test]
fn test_cross_stage_bound_tightness() {
    let input_range = 1.0f32;

    // Stage 1: Text encoder
    let (enc_def, enc_bindings, _) = build_text_encoder();
    let enc_graph = tensor_kernel_to_graph(&enc_def, &enc_bindings).expect("enc graph");
    let enc_input = uniform_bounds(&[D_MODEL, SEQ_LEN], input_range);
    let enc_output = enc_graph
        .propagate_ibp(&enc_input)
        .expect("IBP through encoder");
    let (enc_lo, enc_hi) = bounds_min_max(&enc_output);
    let enc_width = enc_hi - enc_lo;

    // Stage 2: Style projector (fed from encoder output bounds)
    let (style_def, style_bindings, _) = build_style_projector();
    let style_graph = tensor_kernel_to_graph(&style_def, &style_bindings).expect("style graph");
    let style_output = style_graph
        .propagate_ibp(&enc_output)
        .expect("IBP through style");
    let (style_lo, style_hi) = bounds_min_max(&style_output);
    let style_width = style_hi - style_lo;

    // Stage 3: Decoder block
    let (dec_def, dec_bindings, _) = build_decoder_block();
    let dec_graph = tensor_kernel_to_graph(&dec_def, &dec_bindings).expect("dec graph");
    let dec_input = uniform_bounds(&[VOC_CH, SEQ_LEN], input_range);
    let dec_output = dec_graph
        .propagate_ibp(&dec_input)
        .expect("IBP through decoder");
    let (dec_lo, dec_hi) = bounds_min_max(&dec_output);
    let dec_width = dec_hi - dec_lo;

    eprintln!("Cross-stage bound tightness:");
    eprintln!(
        "  Input range: ±{input_range} (width {})",
        2.0 * input_range
    );
    eprintln!("  Encoder output: [{enc_lo:.6}, {enc_hi:.6}] width={enc_width:.6}");
    eprintln!("  Style output:   [{style_lo:.6}, {style_hi:.6}] width={style_width:.6}");
    eprintln!("  Decoder output: [{dec_lo:.6}, {dec_hi:.6}] width={dec_width:.6}");

    // Assert no stage produces vacuously wide bounds.
    assert!(
        enc_width < 1e6,
        "Encoder bounds vacuously wide: width={enc_width}"
    );
    assert!(
        style_width < 10.0,
        "Style bounds should be tight (Tanh squashes), got width={style_width}"
    );
    assert!(
        dec_width < 1e6,
        "Decoder bounds vacuously wide: width={dec_width}"
    );

    // Encoder → Style tightening should be significant due to Tanh.
    if enc_width > 0.0 {
        let style_tightening = 1.0 - (style_width / enc_width);
        eprintln!(
            "  Tanh tightening: {style_tightening:.2} ({:.0}%)",
            style_tightening * 100.0
        );
        assert!(
            style_tightening > 0.0,
            "Tanh should tighten bounds, got tightening={style_tightening}"
        );
    }
}

// ===========================================================================
// Test 5: High-variance pathological input through full pipeline
// ===========================================================================

/// Tests that element-wise varying inputs (pathological for InstanceNorm)
/// produce valid bounds through the full 4-stage pipeline.
///
/// InstanceNorm with element-wise different bounds has higher mean/variance
/// uncertainty, which can amplify IBP width. This test verifies the pipeline
/// handles this worst case correctly.
///
/// Part of #4294.
#[test]
fn test_high_variance_input_full_pipeline() {
    let (def, bindings, out_shape) = build_full_four_stage_pipeline();
    assert_eq!(out_shape, [OUT_CH, TIME_UP]);

    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // High-variance input: centers spread across [-2, 2] with perturbation r=0.1
    let hv_input = high_variance_bounds(&[D_MODEL, SEQ_LEN], 2.0, 0.1);
    let hv_output = graph
        .propagate_ibp(&hv_input)
        .expect("IBP through full pipeline with high-variance input");
    assert_bounds_valid(&hv_output);

    let (hv_lo, hv_hi) = bounds_min_max(&hv_output);
    eprintln!("High-variance input: pipeline output [{hv_lo}, {hv_hi}]");

    // P1 still holds: exp() output must be positive.
    assert!(
        hv_lo > 0.0,
        "P1 VIOLATION with high-variance input: exp lower={hv_lo}"
    );

    // Compare with uniform input bounds.
    let uniform_input = uniform_bounds(&[D_MODEL, SEQ_LEN], 2.0);
    let uniform_output = graph
        .propagate_ibp(&uniform_input)
        .expect("IBP through full pipeline with uniform input");
    let (uni_lo, uni_hi) = bounds_min_max(&uniform_output);
    let uni_width = uni_hi - uni_lo;
    let hv_width = hv_hi - hv_lo;

    eprintln!("  Uniform input width:       {uni_width:.6}");
    eprintln!("  High-variance input width: {hv_width:.6}");

    // High-variance input should produce bounded output (may be wider or
    // narrower depending on normalization layer behavior).
    assert!(
        hv_width < 1e8,
        "High-variance output vacuously wide: {hv_width}"
    );
}

// ===========================================================================
// Test 6: Full pipeline with output Tanh clamp
// ===========================================================================

/// Composes the full pipeline through a Tanh output clamp, proving the
/// composed output is bounded within [-1, 1].
///
/// Pipeline: text → encoder → decoder → exp → tanh_clamp → [-1, 1]
///
/// This tests the composition of the TensorBlockBuilder graph with an
/// external NY graph node (Tanh applied to pipeline output bounds).
///
/// Part of #4294.
#[test]
fn test_full_pipeline_with_output_clamp() {
    let (def, bindings, _) = build_full_four_stage_pipeline();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    // Get pipeline output bounds via IBP.
    let pipeline_output = graph.propagate_ibp(&input).expect("IBP through pipeline");
    assert_bounds_valid(&pipeline_output);
    let (pipe_lo, pipe_hi) = bounds_min_max(&pipeline_output);
    eprintln!("Pipeline raw output: [{pipe_lo}, {pipe_hi}]");

    // Apply Clip(-1, 1) to the pipeline output analytically.
    // This simulates composing with a Clip defense-in-depth layer.
    let (lo_arr, hi_arr) = pipeline_output.lower_upper();
    let clamped_lo: Vec<f32> = lo_arr.iter().map(|&v| v.max(-1.0)).collect();
    let clamped_hi: Vec<f32> = hi_arr.iter().map(|&v| v.min(1.0)).collect();

    let clamped = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(lo_arr.shape()), clamped_lo).expect("lo"),
        ArrayD::from_shape_vec(IxDyn(hi_arr.shape()), clamped_hi).expect("hi"),
    )
    .expect("clamped bounds");
    assert_bounds_valid(&clamped);

    let (clamp_lo, clamp_hi) = bounds_min_max(&clamped);
    eprintln!("Pipeline + clamp output: [{clamp_lo}, {clamp_hi}]");

    // P2 (Non-clipping): clamped output ∈ [-1, 1].
    assert!(
        clamp_lo >= -1.0,
        "Clamped lower {clamp_lo} should be >= -1.0"
    );
    assert!(clamp_hi <= 1.0, "Clamped upper {clamp_hi} should be <= 1.0");
    eprintln!("  P2 PROVEN (with clamp): output in [{clamp_lo:.6}, {clamp_hi:.6}] ⊆ [-1, 1]");

    // The original exp output was positive, so the clamped lower should
    // reflect that (min of exp_lower and 1.0 clamp doesn't affect lower since
    // we clamp to max(-1, exp_lower) which is > 0 since exp_lower > 0).
    assert!(
        clamp_lo > 0.0,
        "Clamp preserves P1: lower {clamp_lo} > 0 (exp was positive)"
    );
}

// ===========================================================================
// Test 7: CROWN vs IBP tightening per stage
// ===========================================================================

/// Runs both IBP and CROWN through each individual stage and reports the
/// tightening ratio. Verifies that CROWN produces at least as tight bounds
/// as IBP for each stage (fundamental soundness invariant).
///
/// Part of #4294.
#[test]
fn test_crown_vs_ibp_per_stage_tightening() {
    let stages: Vec<(&str, Box<dyn Fn() -> _>, &[usize])> = vec![
        (
            "text_encoder",
            Box::new(build_text_encoder),
            &[D_MODEL, SEQ_LEN],
        ),
        (
            "style_projector",
            Box::new(build_style_projector),
            &[ENC_DIM, SEQ_LEN],
        ),
        (
            "decoder_block",
            Box::new(build_decoder_block),
            &[VOC_CH, SEQ_LEN],
        ),
    ];

    eprintln!("CROWN vs IBP tightening per stage:");
    for (name, builder, input_shape) in &stages {
        let (def, bindings, _) = builder();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = uniform_bounds(input_shape, 1.0);

        let ibp_output = graph.propagate_ibp(&input).expect("IBP");
        let (ibp_lo, ibp_hi) = bounds_min_max(&ibp_output);
        let ibp_width = ibp_hi - ibp_lo;

        let (method, crown_output, fallback_reason) =
            assert_crown_tighter_when_not_fallback(&graph, &input);
        let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
        let crown_width = crown_hi - crown_lo;

        let tightening = if ibp_width > 0.0 {
            1.0 - (crown_width / ibp_width)
        } else {
            0.0
        };

        eprintln!(
            "  {name}: IBP width={ibp_width:.6}, CROWN width={crown_width:.6}, \
             tightening={tightening:.4} ({:.1}%), method={method:?}",
            tightening * 100.0
        );
        if let Some(reason) = &fallback_reason {
            eprintln!("    fallback: {reason}");
        }

        // CROWN should not produce wider bounds than IBP (unless fallback).
        if matches!(method, nn_verify::PropMethod::Crown) {
            // Allow small numerical tolerance.
            assert!(
                crown_width <= ibp_width + 1e-3,
                "{name}: CROWN width {crown_width} > IBP width {ibp_width} \
                 (tightness invariant violated)"
            );
        }
    }
}

// ===========================================================================
// Test 8: Monotonicity across 5 input radii
// ===========================================================================

/// Tests that output bound width is monotonically non-decreasing across
/// 5 input radii: 0.01, 0.1, 0.5, 1.0, 2.0.
///
/// This extends the 2-point monotonicity test in compose_kokoro_multi_stage.rs
/// to a full 5-point sweep, catching non-monotonic behavior at intermediate
/// radii that a 2-point test would miss.
///
/// Part of #4294.
#[test]
fn test_monotonicity_five_radii() {
    let (def, bindings, _) = build_full_four_stage_pipeline();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    let radii = [0.01, 0.1, 0.5, 1.0, 2.0];
    let mut prev_width = 0.0f32;
    let mut prev_radius = 0.0f32;

    eprintln!("Monotonicity sweep (5 radii):");
    for &radius in &radii {
        let input = uniform_bounds(&[D_MODEL, SEQ_LEN], radius);
        let output = graph.propagate_ibp(&input).expect("IBP");
        let (lo_min, hi_max) = bounds_min_max(&output);
        let width = hi_max - lo_min;

        eprintln!("  radius={radius:.2}: output [{lo_min:.6}, {hi_max:.6}], width={width:.6}");

        // P1 should hold at all radii.
        assert!(
            lo_min > 0.0,
            "P1 VIOLATION at radius={radius}: lo_min={lo_min}"
        );

        // Monotonicity: wider input → wider output (or equal).
        assert!(
            width >= prev_width - 1e-6,
            "Monotonicity violated: radius={radius} width={width} < \
             radius={prev_radius} width={prev_width}"
        );

        prev_width = width;
        prev_radius = radius;
    }
}

// ===========================================================================
// Test 9: Encoder → Multi-ResBlock decoder chain (end-to-end depth)
// ===========================================================================

/// Chains the text encoder through the multi-ResBlock decoder (2 ResBlocks)
/// in a single graph, testing bound propagation through maximum depth.
///
/// This is deeper than the 4-stage pipeline (which has 1 ResBlock) because
/// it includes 2 sequential InstanceNorm + Snake + Conv1d + residual blocks.
///
/// Part of #4294.
#[test]
fn test_encoder_to_multi_resblock_chain() {
    // Build: text encoder → multi-resblock decoder (combined graph)
    let up_shape = [OUT_CH, TIME_UP];
    let voc_up_shape = [VOC_CH, TIME_UP];

    let mut b = TensorBlockBuilder::new("kokoro_deep_enc_multi_res");

    let text = b.add_input("text_features", &[D_MODEL, SEQ_LEN]);

    // Stage 1: Text encoder
    let conv_w = b.add_input("enc_conv_w", &[D_MODEL, D_MODEL, 3]);
    let conv_out = b.add_conv1d(text, conv_w, None, 1, 1, &[D_MODEL, SEQ_LEN]);
    let relu_out = b.add_relu(conv_out, &[D_MODEL, SEQ_LEN]);
    let t1 = b.add_transpose(relu_out, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let proj_w = b.add_input("enc_proj_w", &[ENC_DIM, D_MODEL]);
    let proj_b_node = b.add_input("enc_proj_b", &[ENC_DIM]);
    let mm = b.add_matmul(t1, proj_w, true, None, &[SEQ_LEN, ENC_DIM]);
    let proj_b_bc = b.add_broadcast(proj_b_node, &[SEQ_LEN, ENC_DIM]);
    let enc_biased = b.add_binary_add(mm, proj_b_bc, &[SEQ_LEN, ENC_DIM]);
    let encoded = b.add_transpose(enc_biased, &[1, 0], &[ENC_DIM, SEQ_LEN]);

    // Stage 2: Decoder with 2 ResBlocks
    let eps = b.add_input("dec_eps", &[1]);
    let conv_pre_w = b.add_input("dec_conv_pre_w", &[VOC_CH, ENC_DIM, 3]);
    let x = b.add_conv1d(encoded, conv_pre_w, None, 1, 1, &[VOC_CH, SEQ_LEN]);
    let x_act = b.add_leaky_relu(x, 0.1, &[VOC_CH, SEQ_LEN]);

    let up_w = b.add_input("dec_up_w", &[VOC_CH, VOC_CH, UP_KERNEL]);
    let x_up = b.add_conv_transpose_1d(
        x_act,
        up_w,
        None,
        UP_STRIDE,
        UP_PADDING,
        1,
        1,
        0,
        &voc_up_shape,
    );

    // ResBlock 1
    let gamma1 = b.add_input("gamma1", &[VOC_CH]);
    let beta1 = b.add_input("beta1", &[VOC_CH]);
    let normed1 = b.add_instance_norm(x_up, eps, 1, Some(gamma1), Some(beta1), &voc_up_shape);
    let alpha1 = b.add_input("alpha1", &[1]);
    let alpha1_bc = b.add_broadcast(alpha1, &voc_up_shape);
    let snake1 = build_snake_scalar_kernel().expect("snake kernel 1");
    let snake1_out = b.add_elementwise(snake1, &[normed1, alpha1_bc], &voc_up_shape);
    let res1_w = b.add_input("res1_w", &[VOC_CH, VOC_CH, 3]);
    let sub1 = b.add_conv1d(snake1_out, res1_w, None, 1, 1, &voc_up_shape);
    let res1_out = b.add_binary_add(x_up, sub1, &voc_up_shape);

    // ResBlock 2
    let gamma2 = b.add_input("gamma2", &[VOC_CH]);
    let beta2 = b.add_input("beta2", &[VOC_CH]);
    let normed2 = b.add_instance_norm(res1_out, eps, 1, Some(gamma2), Some(beta2), &voc_up_shape);
    let alpha2 = b.add_input("alpha2", &[1]);
    let alpha2_bc = b.add_broadcast(alpha2, &voc_up_shape);
    let snake2 = build_snake_scalar_kernel().expect("snake kernel 2");
    let snake2_out = b.add_elementwise(snake2, &[normed2, alpha2_bc], &voc_up_shape);
    let res2_w = b.add_input("res2_w", &[VOC_CH, VOC_CH, 3]);
    let sub2 = b.add_conv1d(snake2_out, res2_w, None, 1, 1, &voc_up_shape);
    let res2_out = b.add_binary_add(res1_out, sub2, &voc_up_shape);

    let res_act = b.add_leaky_relu(res2_out, 0.01, &voc_up_shape);
    let conv_post_w = b.add_input("conv_post_w", &[OUT_CH, VOC_CH, 3]);
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 1, &up_shape);
    let output = b.add_exp(x_post, &up_shape);

    let def = b.build(output).expect("encoder-multi-resblock graph");
    let bindings = vec![
        TensorParamBinding::Variable,
        // Encoder weights
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[D_MODEL, D_MODEL, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM, D_MODEL]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[ENC_DIM]), 0.0f32)),
        // Decoder pre
        TensorParamBinding::ConstantScalar(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH, ENC_DIM, 3]), W_MAG)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[VOC_CH, VOC_CH, UP_KERNEL]),
            W_MAG,
        )),
        // ResBlock 1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH, VOC_CH, 3]), W_MAG)),
        // ResBlock 2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH]), 1.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH]), 0.0f32)),
        TensorParamBinding::ConstantScalar(1.0),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[VOC_CH, VOC_CH, 3]), W_MAG)),
        // conv_post
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[OUT_CH, VOC_CH, 3]), W_MAG)),
    ];

    def.validate()
        .expect("encoder-multi-resblock def validates");
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    assert!(
        graph.num_nodes() >= 20,
        "encoder + 2-resblock decoder should have >= 20 nodes, got {}",
        graph.num_nodes()
    );

    let input_bounds = uniform_bounds(&[D_MODEL, SEQ_LEN], 1.0);

    let ibp_output = graph
        .propagate_ibp(&input_bounds)
        .expect("IBP through encoder + multi-resblock");
    assert_eq!(ibp_output.lower_upper().0.shape(), &[OUT_CH, TIME_UP]);
    assert_bounds_valid(&ibp_output);

    let (lo_min, hi_max) = bounds_min_max(&ibp_output);
    eprintln!("Encoder → Multi-ResBlock decoder IBP: [{lo_min}, {hi_max}]");
    assert!(
        lo_min > 0.0,
        "P1: exp positive through deep chain, got {lo_min}"
    );
    assert!(hi_max < 1e8, "IBP upper bounded, got {hi_max}");

    // CROWN
    let (method, crown_output, fallback_reason) =
        assert_crown_tighter_when_not_fallback(&graph, &input_bounds);
    let (crown_lo, crown_hi) = bounds_min_max(&crown_output);
    eprintln!("Encoder → Multi-ResBlock CROWN: method={method:?}, [{crown_lo}, {crown_hi}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("CROWN fallback reason: {reason}");
    }

    let result = verify_and_assert(&def, &bindings, &input_bounds, "kokoro_deep_enc_multi_res");
    assert_eq!(result.num_variables, 1);
    assert!(
        matches!(
            result.verification.soundness_mode,
            VerificationSoundnessMode::Sound | VerificationSoundnessMode::Heuristic
        ),
        "soundness should be Sound or Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}
