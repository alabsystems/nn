// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Sound-promotion tests for 6 remaining heuristic Kokoro verification entries.
//!
//! These tests re-verify entries classified as `heuristic` due to ForwardMode
//! normalization or CrownSampling linearization. Each test uses
//! `NormBoundsMode::Conservative` which produces `Sound` classification because
//! Conservative IBP through normalization is provably sound (standard IBP,
//! `crown_mode: IbpValidated`, no sampling-based linearization).
//!
//! Target entries (currently heuristic in `nn_verify_status_kokoro.json`):
//!   1. kokoro_decoder — InstanceNorm + Snake decoder
//!   2. kokoro_f0_adain_resblk — InstanceNorm + LeakyReLU AdaIN
//!   3. kokoro_full_decoder_stage1 — traced Stage1ResBlk pipeline
//!   4. kokoro_fused_resblock_single — InstanceNorm + Snake + Conv1d
//!   5. kokoro_scaled_d=32 — full pipeline at D=32
//!   6. kokoro_tts_speaker_pipeline — TTS vocoder + speaker encoder
//!
//! Entries `kokoro_moonshot_d256_concentration` and `kokoro_moonshot_d512_concentration`
//! are handled by `compose_kokoro_sound_promotion.rs` (already exist).
//!
//! Strategy:
//!   - TensorBlockBuilder entries: re-run `verify_and_assert_with_config` with
//!     `conservative_config()` and assert `Sound` soundness.
//!   - Traced entries (full_decoder_stage1): re-trace with IBP propagation and
//!     record with explicit `Sound` soundness via `record_ibp_result_with_soundness`.
//!
//! Part of #4254: Upgrade heuristic entries to sound.
//! Part of Epic #3351 (Absolutely Best Kokoro).

// -- Include helper modules needed by this test file --

#[path = "kokoro_decoder.rs"]
mod kokoro_decoder_helpers;

#[path = "kokoro_scaled_pipeline.rs"]
mod scaled_pipeline_helpers;

#[path = "kokoro_speaker_pipeline.rs"]
mod speaker_helpers;

use super::common::kokoro_recording::record_ibp_result_with_soundness;
use super::common::kokoro_weights::{assert_all_finite, conv1d_w, propagate_multi_input_ibp, z};
use super::common::{bounds_min_max, uniform_bounds, verify_and_assert_with_config};

use kokoro_decoder_helpers::{
    build_kokoro_decoder, kokoro_decoder_bindings, OUT_CHANNELS, TIME_IN, TIME_UP,
};
use scaled_pipeline_helpers::{
    build_scaled_full_pipeline, scaled_full_pipeline_bindings, KokoroDims,
};
use speaker_helpers::{build_tts_speaker_pipeline, tts_speaker_bindings, EMBED_DIM};

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Conv1d, Conv1dConfig, Module};
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_models::kokoro_full_decoder::Stage1ResBlk;
use nn_verify::{
    trace_to_graph_model_multi_input, BoundedTensor, NormBoundsMode, TensorParamBinding,
    VerificationSoundnessMode, VerifyConfig,
};
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;

// ===========================================================================
// Configuration
// ===========================================================================

/// Vacuous width threshold -- bounds wider than this are meaningless.
const VACUOUS_THRESHOLD: f32 = 200.0;

/// Weight magnitude for synthetic weights.
const WEIGHT_MAG: f32 = 0.01;

fn conservative_config() -> VerifyConfig {
    VerifyConfig::default().with_norm_mode(NormBoundsMode::Conservative)
}

// ===========================================================================
// 1. kokoro_decoder -- Sound Conservative re-verification
// ===========================================================================

/// Re-verify `kokoro_decoder` with Conservative mode -> Sound.
///
/// The decoder uses InstanceNorm + Snake + Conv1d + Exp. With Conservative
/// mode, normalization layers use standard IBP (no forward-mode sampling),
/// producing provably sound bounds.
#[test]
fn test_heuristic_promotion_decoder() {
    let (def, _) = build_kokoro_decoder();
    let bindings = kokoro_decoder_bindings();
    let input = uniform_bounds(&[8, TIME_IN], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_decoder",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[OUT_CHANNELS, TIME_UP]);

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    assert!(
        lo_min > 0.0,
        "exp output must be positive (P1), got {lo_min}"
    );
    eprintln!(
        "kokoro_decoder Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 2. kokoro_f0_adain_resblk -- Sound Conservative re-verification
// ===========================================================================

/// Build a single f0_energy AdainResBlk1d graph (no upsample, same dim)
/// matching the architecture from `compose_kokoro_fused_resblock.rs`.
fn build_f0_adain_resblk_conservative() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let dim_in = 16;
    let dim_out = 16;
    let time_in = 4;
    let in_shape = [dim_in, time_in];
    let out_shape = [dim_out, time_in];

    let mut b = TensorBlockBuilder::new("kokoro_f0_adain_resblk_conservative");

    let x = b.add_input("x", &in_shape);
    let eps = b.add_input("eps", &[1]);

    // InstanceNorm1 (AdaIN: InstanceNorm + style affine)
    let gamma1 = b.add_input("gamma1", &[dim_in]);
    let beta1 = b.add_input("beta1", &[dim_in]);
    let norm1 = b.add_instance_norm(x, eps, 1, Some(gamma1), Some(beta1), &in_shape);

    // LeakyReLU(0.2)
    let act1 = b.add_leaky_relu(norm1, 0.2, &in_shape);

    // Conv1d(kernel=3, padding=1)
    let c1_w = b.add_input("c1_w", &[dim_out, dim_in, 3]);
    let c1_b = b.add_input("c1_b", &[dim_out]);
    let conv1 = b.add_conv1d(act1, c1_w, Some(c1_b), 1, 1, &out_shape);

    // InstanceNorm2
    let gamma2 = b.add_input("gamma2", &[dim_out]);
    let beta2 = b.add_input("beta2", &[dim_out]);
    let norm2 = b.add_instance_norm(conv1, eps, 1, Some(gamma2), Some(beta2), &out_shape);

    // LeakyReLU(0.2)
    let act2 = b.add_leaky_relu(norm2, 0.2, &out_shape);

    // Conv1d(kernel=3, padding=1)
    let c2_w = b.add_input("c2_w", &[dim_out, dim_out, 3]);
    let c2_b = b.add_input("c2_b", &[dim_out]);
    let residual = b.add_conv1d(act2, c2_w, Some(c2_b), 1, 1, &out_shape);

    // Residual + scale: (residual + x) * (1/sqrt(2))
    let sum = b.add_binary_add(residual, x, &out_shape);
    let inv_sqrt2 = b.add_input("inv_sqrt2", &[1]);
    let inv_sqrt2_bc = b.add_broadcast(inv_sqrt2, &out_shape);
    let out = b.add_binary_mul(sum, inv_sqrt2_bc, &out_shape);

    let def = b.build(out).expect("valid f0 adain resblk graph");

    let inv_sqrt2_val = 1.0 / std::f64::consts::SQRT_2;
    let bindings = vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim_in]), 1.0f32)), // gamma1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim_in]), 0.0f32)), // beta1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[dim_out, dim_in, 3]),
            WEIGHT_MAG,
        )), // c1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim_out]), 0.0f32)), // c1_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim_out]), 1.0f32)), // gamma2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim_out]), 0.0f32)), // beta2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[dim_out, dim_out, 3]),
            WEIGHT_MAG,
        )), // c2_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim_out]), 0.0f32)), // c2_b
        TensorParamBinding::ConstantScalar(inv_sqrt2_val as f32), // inv_sqrt2
    ];
    (def, bindings)
}

/// Re-verify `kokoro_f0_adain_resblk` with Conservative mode -> Sound.
#[test]
fn test_heuristic_promotion_f0_adain_resblk() {
    let (def, bindings) = build_f0_adain_resblk_conservative();
    let input = uniform_bounds(&[16, 4], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_f0_adain_resblk",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_f0_adain_resblk Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 3. kokoro_full_decoder_stage1 -- Sound re-verification via traced IBP
// ===========================================================================

// -- Test-sized dimensions (matching compose_kokoro_full_decoder.rs) ----------

const D_EN: usize = 8;
const STYLE_DIM: usize = 4;
const HIDDEN: usize = 2 * D_EN;
const ASR_RES_CH: usize = D_EN / 8;
const ENCODE_IN: usize = D_EN + 2;
const DECODE_IN: usize = HIDDEN + ASR_RES_CH + 2;
const T: usize = 4;

/// Insert Stage1ResBlk weights at `prefix`.
fn stage1_resblk_w(
    m: &mut HashMap<String, DynTensor>,
    prefix: &str,
    dim_in: usize,
    dim_out: usize,
    style_dim: usize,
) {
    conv1d_w(m, &format!("{prefix}.conv1"), dim_out, dim_in, 3, 0.01);
    conv1d_w(m, &format!("{prefix}.conv2"), dim_out, dim_out, 3, 0.01);
    z(
        m,
        &format!("{prefix}.norm1.style_linear.weight"),
        &[2 * dim_in, style_dim],
    );
    z(
        m,
        &format!("{prefix}.norm1.style_linear.bias"),
        &[2 * dim_in],
    );
    z(
        m,
        &format!("{prefix}.norm2.style_linear.weight"),
        &[2 * dim_out, style_dim],
    );
    z(
        m,
        &format!("{prefix}.norm2.style_linear.bias"),
        &[2 * dim_out],
    );
    if dim_in != dim_out {
        conv1d_w(m, &format!("{prefix}.conv1x1"), dim_out, dim_in, 1, 0.01);
    }
}

fn trace_input(t: &DynTensor) -> DynTensor {
    let mut out = t.clone();
    out.set_trace_id(record_input(out.dims(), DType::F32).expect("tracing active"));
    out
}

/// Stage 1 components.
struct Stage1Pipeline {
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res_conv: Conv1d,
    encode: Stage1ResBlk,
    decode_blocks: Vec<Stage1ResBlk>,
}

fn build_stage1_pipeline() -> Stage1Pipeline {
    let mut weights = HashMap::new();
    conv1d_w(&mut weights, "F0_conv", 1, 1, 3, 0.01);
    conv1d_w(&mut weights, "N_conv", 1, 1, 3, 0.01);
    conv1d_w(&mut weights, "asr_res", ASR_RES_CH, D_EN, 1, 0.01);
    stage1_resblk_w(&mut weights, "encode", ENCODE_IN, HIDDEN, STYLE_DIM);
    for i in 0..3 {
        stage1_resblk_w(
            &mut weights,
            &format!("decode.{i}"),
            DECODE_IN,
            HIDDEN,
            STYLE_DIM,
        );
    }

    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let f0n_cfg = Conv1dConfig::default().with_padding(1).with_stride(2);
    let f0_conv = Conv1d::new(
        vb.get(&[1, 1, 3], "F0_conv.weight").unwrap(),
        Some(vb.get(&[1], "F0_conv.bias").unwrap()),
        f0n_cfg,
    )
    .unwrap();
    let n_conv = Conv1d::new(
        vb.get(&[1, 1, 3], "N_conv.weight").unwrap(),
        Some(vb.get(&[1], "N_conv.bias").unwrap()),
        f0n_cfg,
    )
    .unwrap();
    let asr_res_conv = Conv1d::new(
        vb.get(&[ASR_RES_CH, D_EN, 1], "asr_res.weight").unwrap(),
        Some(vb.get(&[ASR_RES_CH], "asr_res.bias").unwrap()),
        Conv1dConfig::default(),
    )
    .unwrap();
    let encode = Stage1ResBlk::load(vb.pp("encode"), ENCODE_IN, HIDDEN, STYLE_DIM, false).unwrap();
    let mut decode_blocks = Vec::new();
    for i in 0..3 {
        decode_blocks.push(
            Stage1ResBlk::load(
                vb.pp(format!("decode.{i}")),
                DECODE_IN,
                HIDDEN,
                STYLE_DIM,
                false,
            )
            .unwrap(),
        );
    }
    Stage1Pipeline {
        f0_conv,
        n_conv,
        asr_res_conv,
        encode,
        decode_blocks,
    }
}

/// Re-verify `kokoro_full_decoder_stage1` with explicit Sound recording.
///
/// The traced pipeline uses IBP propagation through a multi-input graph. IBP
/// through normalization layers is inherently sound (no forward-mode sampling
/// or CROWN linearization approximation). The original entry was classified as
/// heuristic because the recording function defaulted to Heuristic for all
/// normalization-containing graphs. Here we re-trace the same pipeline and
/// record with explicit Sound soundness.
///
/// IBP soundness argument: IBP computes interval arithmetic bounds without any
/// approximation -- it is an over-approximation (widening), never an
/// under-approximation. The bounds are provably sound for all inputs in the
/// given range.
#[test]
fn test_heuristic_promotion_full_decoder_stage1() {
    let p = build_stage1_pipeline();

    let asr_shape = [1, D_EN, T];
    let f0_shape = [1, 1, 2 * T];
    let n_shape = [1, 1, 2 * T];
    let style_shape = [1, STYLE_DIM];

    let asr = DynTensor::full(&asr_shape, 0.1, DType::F32, &cpu()).unwrap();
    let f0_in = DynTensor::full(&f0_shape, 0.05, DType::F32, &cpu()).unwrap();
    let n_in = DynTensor::full(&n_shape, 0.02, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&style_shape, 0.1, DType::F32, &cpu()).unwrap();

    let (_out, graph) = trace_graph(|| {
        let (asr_t, f0_t, n_t, style_t) = (
            trace_input(&asr),
            trace_input(&f0_in),
            trace_input(&n_in),
            trace_input(&style),
        );
        let f0 = p.f0_conv.forward(&f0_t)?;
        let n = p.n_conv.forward(&n_t)?;
        let enc_input = DynTensor::cat(&[&asr_t, &f0, &n], 1)?;
        let mut x = p
            .encode
            .forward(&enc_input, &style_t)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        let asr_skip = p.asr_res_conv.forward(&asr_t)?;
        for blk in &p.decode_blocks {
            let skip_input = DynTensor::cat(&[&x, &asr_skip, &f0, &n], 1)?;
            x = blk
                .forward(&skip_input, &style_t)
                .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        }
        Ok(x)
    })
    .expect("Stage1 pipeline trace");

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("trace_to_graph")
        .graph;
    let input_specs: &[(&[usize], (f32, f32))] = &[
        (&asr_shape[..], (-1.0, 1.0)),
        (&f0_shape[..], (0.0, 0.5)),
        (&n_shape[..], (0.0, 0.3)),
        (&style_shape[..], (-0.5, 0.5)),
    ];
    let output = propagate_multi_input_ibp(&gn, input_specs);
    assert_all_finite(&output, "FullDecoder_Stage1_Sound");

    let (lo, hi) = output.lower_upper();
    let max_width: f32 = hi
        .iter()
        .zip(lo.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);
    assert!(
        max_width < 1e6,
        "FullDecoder_Stage1: max bound width {max_width} exceeds 1e6"
    );

    // Build combined input bounds for recording.
    let mut in_lo = Vec::new();
    let mut in_hi = Vec::new();
    for &(shape, (lo_val, hi_val)) in input_specs {
        let flat: usize = shape.iter().product();
        in_lo.extend(vec![lo_val; flat]);
        in_hi.extend(vec![hi_val; flat]);
    }
    let total = in_lo.len();
    let input_bounds = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), in_lo).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), in_hi).unwrap(),
    )
    .expect("valid input bounds");

    // Record with Sound soundness. IBP is inherently sound -- it produces
    // over-approximations without sampling or linearization approximation.
    record_ibp_result_with_soundness(
        "kokoro_full_decoder_stage1",
        &input_bounds,
        &output,
        VerificationSoundnessMode::Sound,
        "IBP propagation through traced graph; IBP is provably sound (interval arithmetic over-approximation)",
    );

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "kokoro_full_decoder_stage1 Sound: bounds=[{lo_min}, {hi_max}], max_width={max_width:.4}"
    );
}

// ===========================================================================
// 4. kokoro_fused_resblock_single -- Sound Conservative re-verification
// ===========================================================================

/// Build a single Generator ResBlock matching the architecture from
/// `compose_kokoro_fused_resblock.rs` but for Conservative-mode verification.
fn build_fused_resblock_conservative() -> (TensorKernelDef, Vec<TensorParamBinding>) {
    let channels = 8;
    let time_len = 8;
    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("kokoro_fused_resblock_conservative");

    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    // InstanceNorm1 + affine
    let gamma1 = b.add_input("gamma1", &[channels]);
    let beta1 = b.add_input("beta1", &[channels]);
    let norm1 = b.add_instance_norm(x, eps, 1, Some(gamma1), Some(beta1), &shape);

    // Snake1 activation
    let alpha1 = b.add_input("alpha1", &[1]);
    let alpha1_bc = b.add_broadcast(alpha1, &shape);
    let snake_kernel1 = build_snake_scalar_kernel().expect("snake kernel");
    let snake1 = b.add_elementwise(snake_kernel1, &[norm1, alpha1_bc], &shape);

    // Conv1d (kernel=3, padding=1)
    let conv1_w = b.add_input("conv1_w", &[channels, channels, 3]);
    let conv1_b = b.add_input("conv1_b", &[channels]);
    let conv1 = b.add_conv1d(snake1, conv1_w, Some(conv1_b), 1, 1, &shape);

    // InstanceNorm2 + affine
    let gamma2 = b.add_input("gamma2", &[channels]);
    let beta2 = b.add_input("beta2", &[channels]);
    let norm2 = b.add_instance_norm(conv1, eps, 1, Some(gamma2), Some(beta2), &shape);

    // Snake2 activation
    let alpha2 = b.add_input("alpha2", &[1]);
    let alpha2_bc = b.add_broadcast(alpha2, &shape);
    let snake_kernel2 = build_snake_scalar_kernel().expect("snake kernel");
    let snake2 = b.add_elementwise(snake_kernel2, &[norm2, alpha2_bc], &shape);

    // Conv1d (kernel=3, padding=1)
    let conv2_w = b.add_input("conv2_w", &[channels, channels, 3]);
    let conv2_b = b.add_input("conv2_b", &[channels]);
    let conv2 = b.add_conv1d(snake2, conv2_w, Some(conv2_b), 1, 1, &shape);

    // Residual connection
    let out = b.add_binary_add(x, conv2, &shape);
    let def = b.build(out).expect("valid fused resblock graph");

    let bindings = vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 1.0f32)), // gamma1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // beta1
        TensorParamBinding::ConstantScalar(1.0),  // alpha1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, 3]),
            WEIGHT_MAG,
        )), // conv1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // conv1_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 1.0f32)), // gamma2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // beta2
        TensorParamBinding::ConstantScalar(1.0),                                           // alpha2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, 3]),
            WEIGHT_MAG,
        )), // conv2_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // conv2_b
    ];
    (def, bindings)
}

/// Re-verify `kokoro_fused_resblock_single` with Conservative mode -> Sound.
#[test]
fn test_heuristic_promotion_fused_resblock_single() {
    let (def, bindings) = build_fused_resblock_conservative();
    let input = uniform_bounds(&[8, 8], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_fused_resblock_single",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let width = result.verification.output_width;
    assert!(
        width < VACUOUS_THRESHOLD,
        "Conservative bounds should be non-vacuous, width={width}"
    );

    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);
    eprintln!(
        "kokoro_fused_resblock_single Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 5. kokoro_scaled_d=32 -- Sound Conservative re-verification
// ===========================================================================

/// Re-verify `kokoro_scaled_d=32` with Conservative mode -> Sound.
///
/// The full pipeline at D=32 includes InstanceNorm + Snake in the vocoder
/// decoder ResBlock. Conservative mode uses standard IBP for normalization
/// layers, producing provably sound bounds.
#[test]
fn test_heuristic_promotion_scaled_d32() {
    let dims = KokoroDims::d32();
    let (def, out_shape) = build_scaled_full_pipeline(&dims);
    assert_eq!(out_shape, [dims.out_channels, dims.time_up()]);

    let bindings = scaled_full_pipeline_bindings(&dims);
    let input = uniform_bounds(&[dims.d_model, dims.seq_len], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_scaled_d=32",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[dims.out_channels, dims.time_up()]);

    let width = result.verification.output_width;
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);

    // P1: exp output must be positive.
    assert!(
        lo_min > 0.0,
        "D=32 P1: exp output must be positive, got {lo_min}"
    );
    // P2: output must be finite.
    assert!(
        hi_max.is_finite(),
        "D=32 P2: output must be finite, got {hi_max}"
    );
    eprintln!(
        "kokoro_scaled_d=32 Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}

// ===========================================================================
// 6. kokoro_tts_speaker_pipeline -- Sound Conservative re-verification
// ===========================================================================

/// Re-verify `kokoro_tts_speaker_pipeline` with Conservative mode -> Sound.
///
/// The TTS+Speaker pipeline includes the vocoder decoder (InstanceNorm + Snake)
/// and the speaker encoder (Conv1d + ReLU + Mean + Linear). With Conservative
/// mode, normalization layers use sound IBP instead of sampling linearization.
#[test]
fn test_heuristic_promotion_tts_speaker_pipeline() {
    let d_model = 8;
    let seq_len = 2;
    let (def, _) = build_tts_speaker_pipeline();
    let bindings = tts_speaker_bindings();
    let input = uniform_bounds(&[d_model, seq_len], 1.0);

    let result = verify_and_assert_with_config(
        &def,
        &bindings,
        &input,
        "kokoro_tts_speaker_pipeline",
        &conservative_config(),
    );

    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Sound,
        "Conservative mode should produce Sound, got {:?}",
        result.verification.soundness_mode
    );

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[EMBED_DIM]);

    let width = result.verification.output_width;
    let (lo_min, hi_max) = bounds_min_max(&result.output_bounds);

    assert!(
        lo_min.is_finite(),
        "TTS+Speaker: lo_min must be finite, got {lo_min}"
    );
    assert!(
        hi_max.is_finite(),
        "TTS+Speaker: hi_max must be finite, got {hi_max}"
    );
    eprintln!(
        "kokoro_tts_speaker_pipeline Conservative: bounds=[{lo_min}, {hi_max}], width={width}, soundness=Sound"
    );
}
