// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! IBP bound width at intermediate Kokoro scale (d_en=64).
//!
//! Issue #2410: All existing Kokoro verification uses toy-scale dimensions
//! (d_en=8, hidden=8). Production Kokoro has d_en=512, 12 ALBERT layers,
//! 82M parameters. IBP bounds grow exponentially with network depth —
//! nobody has validated bounds remain useful (non-vacuous) at scale.
//!
//! This module tests at d_en=64, hidden=128 with pseudo-random weights at
//! two magnitudes (conservative 0.005 and Xavier-scale 0.1) to determine
//! whether IBP produces tight-enough bounds for meaningful verification,
//! or if CROWN linearization is needed at every segment boundary.
//!
//! Part of #2410.
//! Part of #2218.

use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, TensorError, VarBuilder};
use nn_models::kokoro_tts::{ProsodyPredictor, TextEncoder};
use nn_verify::trace_to_graph_model_multi_input;
use std::collections::HashMap;

use super::common::kokoro_weights::{assert_all_finite, propagate_multi_input_ibp, z_fill};
use super::common::{assert_bounds_valid, bounds_min_max, uniform_bounds};

// -- Intermediate-scale dimensions (8× toy, 1/8 production) -------------------

/// Encoder dimension: 64 (production: 512, toy: 8).
const D_EN: usize = 64;
/// Token vocabulary size for TextEncoder.
const VOCAB_SIZE: usize = 32;
/// Style embedding dimension: 16 (production: 128, toy: 4).
const STYLE_DIM: usize = 16;
/// Number of prosody blocks: 2 (production: 3, toy: 1).
const N_PROSODY_LAYERS: usize = 2;

// -- Weight construction with pseudo-random LCG values ------------------------

/// Insert a weight tensor with LCG pseudo-random values in `[-mag, +mag]`.
///
/// Uses a linear congruential generator for reproducibility. Mixed-sign values
/// exercise the IBP bound-flip path where negative weights swap lower/upper
/// during interval matmul. `seed` varies per weight tensor for independence.
fn z_lcg(m: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize], mag: f32, seed: u64) {
    let n: usize = shape.iter().product();
    let mut state = seed;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            // LCG: x_{n+1} = (a * x_n + c) mod m
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let normalized = ((state >> 33) as f32) / ((1u64 << 30) as f32) - 1.0;
            normalized * mag
        })
        .collect();
    let tensor = DynTensor::from_vec(data, shape, &cpu()).unwrap();
    m.insert(name.to_string(), tensor);
}

/// Build TextEncoder weights at d_en=64 scale.
fn scaled_text_encoder_weights(mag: f32) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let bh = D_EN / 2; // BiLSTM hidden per direction

    // Embedding(VOCAB_SIZE, D_EN)
    z_lcg(&mut m, "embedding.weight", &[VOCAB_SIZE, D_EN], mag, 2000);

    // 3x Conv1d(D_EN, D_EN, k=5) + LayerNorm(D_EN)
    for i in 0..3 {
        z_lcg(
            &mut m,
            &format!("convs.{i}.weight"),
            &[D_EN, D_EN, 5],
            mag,
            2010 + i as u64,
        );
        z_fill(&mut m, &format!("convs.{i}.bias"), &[D_EN], 0.0);
        m.insert(
            format!("norms.{i}.weight"),
            DynTensor::full(&[D_EN], 1.0, DType::F32, &cpu()).unwrap(),
        );
        z_fill(&mut m, &format!("norms.{i}.bias"), &[D_EN], 0.0);
    }

    // BiLSTM
    let p = "lstm";
    z_lcg(
        &mut m,
        &format!("{p}.weight_ih_l0"),
        &[4 * bh, D_EN],
        mag,
        2001,
    );
    z_lcg(
        &mut m,
        &format!("{p}.weight_hh_l0"),
        &[4 * bh, bh],
        mag,
        2002,
    );
    z_fill(&mut m, &format!("{p}.bias_ih_l0"), &[4 * bh], 0.0);
    z_fill(&mut m, &format!("{p}.bias_hh_l0"), &[4 * bh], 0.0);

    z_lcg(
        &mut m,
        &format!("{p}.weight_ih_l0_reverse"),
        &[4 * bh, D_EN],
        mag,
        2003,
    );
    z_lcg(
        &mut m,
        &format!("{p}.weight_hh_l0_reverse"),
        &[4 * bh, bh],
        mag,
        2004,
    );
    z_fill(&mut m, &format!("{p}.bias_ih_l0_reverse"), &[4 * bh], 0.0);
    z_fill(&mut m, &format!("{p}.bias_hh_l0_reverse"), &[4 * bh], 0.0);

    z_lcg(
        &mut m,
        &format!("{p}.linear.weight"),
        &[D_EN, D_EN],
        mag,
        2005,
    );
    z_fill(&mut m, &format!("{p}.linear.bias"), &[D_EN], 0.0);
    m
}

/// Maximum duration bins (matches test call: `ProsodyPredictor::load(..., 50)`).
const MAX_DUR: usize = 50;

/// Build ProsodyPredictor weights at d_en=64 scale.
///
/// Matches current `ProsodyPredictor::load` naming (#2498):
/// - DurationEncoder BiLSTMs: `duration.lstms.{i}.*`
/// - DurationEncoder AdaLayerNorms: `duration.norms.{i}.*`
/// - Duration projection: `duration.duration_proj.*`
/// - Final BiLSTM: `lstm.*`
fn scaled_prosody_weights(mag: f32) -> HashMap<String, DynTensor> {
    let mut m = HashMap::new();
    let hidden = D_EN / 2;
    let bilstm_input = D_EN + STYLE_DIM;

    // DurationEncoder: N_PROSODY_LAYERS × (BiLSTM + AdaLayerNorm)
    for i in 0..N_PROSODY_LAYERS {
        let s = (i as u64 + 1) * 3000;

        // BiLSTM forward direction
        let p = format!("duration.lstms.{i}");
        z_lcg(
            &mut m,
            &format!("{p}.weight_ih_l0"),
            &[4 * hidden, bilstm_input],
            mag,
            s,
        );
        z_lcg(
            &mut m,
            &format!("{p}.weight_hh_l0"),
            &[4 * hidden, hidden],
            mag,
            s + 1,
        );
        z_fill(&mut m, &format!("{p}.bias_ih_l0"), &[4 * hidden], 0.0);
        z_fill(&mut m, &format!("{p}.bias_hh_l0"), &[4 * hidden], 0.0);

        // BiLSTM backward direction
        z_lcg(
            &mut m,
            &format!("{p}.weight_ih_l0_reverse"),
            &[4 * hidden, bilstm_input],
            mag,
            s + 2,
        );
        z_lcg(
            &mut m,
            &format!("{p}.weight_hh_l0_reverse"),
            &[4 * hidden, hidden],
            mag,
            s + 3,
        );
        z_fill(
            &mut m,
            &format!("{p}.bias_ih_l0_reverse"),
            &[4 * hidden],
            0.0,
        );
        z_fill(
            &mut m,
            &format!("{p}.bias_hh_l0_reverse"),
            &[4 * hidden],
            0.0,
        );

        // AdaLayerNorm: fc (style projection) + optional norm
        let np = format!("duration.norms.{i}");
        z_lcg(
            &mut m,
            &format!("{np}.fc.weight"),
            &[2 * D_EN, STYLE_DIM],
            mag,
            s + 4,
        );
        z_fill(&mut m, &format!("{np}.fc.bias"), &[2 * D_EN], 0.0);
        z_fill(&mut m, &format!("{np}.norm.weight"), &[D_EN], 1.0);
        z_fill(&mut m, &format!("{np}.norm.bias"), &[D_EN], 0.0);
    }

    // Duration projection: Linear(d_model → max_dur)
    z_lcg(
        &mut m,
        "duration.duration_proj.weight",
        &[MAX_DUR, D_EN],
        mag,
        9001,
    );
    z_fill(&mut m, "duration.duration_proj.bias", &[MAX_DUR], 0.0);

    // Final ProsodyPredictor BiLSTM (prefix "lstm")
    let fp = "lstm";
    z_lcg(
        &mut m,
        &format!("{fp}.weight_ih_l0"),
        &[4 * hidden, bilstm_input],
        mag,
        9100,
    );
    z_lcg(
        &mut m,
        &format!("{fp}.weight_hh_l0"),
        &[4 * hidden, hidden],
        mag,
        9101,
    );
    z_fill(&mut m, &format!("{fp}.bias_ih_l0"), &[4 * hidden], 0.0);
    z_fill(&mut m, &format!("{fp}.bias_hh_l0"), &[4 * hidden], 0.0);
    z_lcg(
        &mut m,
        &format!("{fp}.weight_ih_l0_reverse"),
        &[4 * hidden, bilstm_input],
        mag,
        9102,
    );
    z_lcg(
        &mut m,
        &format!("{fp}.weight_hh_l0_reverse"),
        &[4 * hidden, hidden],
        mag,
        9103,
    );
    z_fill(
        &mut m,
        &format!("{fp}.bias_ih_l0_reverse"),
        &[4 * hidden],
        0.0,
    );
    z_fill(
        &mut m,
        &format!("{fp}.bias_hh_l0_reverse"),
        &[4 * hidden],
        0.0,
    );

    m
}

// -- Shared test runner -------------------------------------------------------

/// Trace text pipeline at d_en=64, propagate IBP, return max bound width.
fn run_text_pipeline_ibp(weight_mag: f32) -> f32 {
    let te_weights = scaled_text_encoder_weights(weight_mag);
    let vb_te = VarBuilder::from_tensors(te_weights, DType::F32, &cpu());
    let text_encoder = TextEncoder::load(&vb_te, VOCAB_SIZE, D_EN).unwrap();

    let batch = 1;
    let seq_len = 4;
    let token_shape = [batch, seq_len];
    let token_ids: Vec<i64> = (0..batch * seq_len)
        .map(|i| (i % VOCAB_SIZE) as i64)
        .collect();
    let tokens = DynTensor::from_vec_i64(token_ids, &token_shape, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = tokens.clone();
        let id = record_input(x.dims(), DType::I64).unwrap();
        x.set_trace_id(id);
        let text_features = text_encoder.forward(&x)?;
        Ok(text_features)
    })
    .unwrap();

    let gn = nn_verify::trace_to_graph_model(&graph)
        .expect("trace_to_graph_model at d_en=64")
        .graph;

    let input_bounds = uniform_bounds(&token_shape, VOCAB_SIZE as f32);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP at d_en=64");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    let (lo_arr, hi_arr) = output.lower_upper();
    let max_width: f32 = hi_arr
        .iter()
        .zip(lo_arr.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);
    let mean_width: f32 = hi_arr
        .iter()
        .zip(lo_arr.iter())
        .map(|(h, l)| h - l)
        .sum::<f32>()
        / lo_arr.len() as f32;

    eprintln!(
        "IBP d_en=64 text_pipeline (mag={weight_mag}):\n  \
         lo=[{lo_min:.6}], hi=[{hi_max:.6}]\n  \
         max_width={max_width:.6}, mean_width={mean_width:.6}\n  \
         shape={:?}, n={}",
        output.lower().shape(),
        lo_arr.len(),
    );

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "text pipeline bounds not finite at d_en=64 mag={weight_mag}: [{lo_min}, {hi_max}]"
    );
    max_width
}

/// Trace ProsodyPredictor at d_en=64, propagate IBP, return max bound width.
fn run_prosody_ibp(weight_mag: f32) -> f32 {
    let weights = scaled_prosody_weights(weight_mag);
    let vb = VarBuilder::from_tensors(weights, DType::F32, &cpu());
    let prosody = ProsodyPredictor::load(&vb, D_EN, STYLE_DIM, N_PROSODY_LAYERS, 50).unwrap();

    let batch = 1;
    let seq_len = 4;
    let x = DynTensor::full(&[batch, D_EN, seq_len], 0.1, DType::F32, &cpu()).unwrap();
    let style = DynTensor::full(&[batch, STYLE_DIM], 0.1, DType::F32, &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut inp = x.clone();
        let id_x = record_input(inp.dims(), DType::F32).unwrap();
        inp.set_trace_id(id_x);
        let mut sty = style.clone();
        let id_s = record_input(sty.dims(), DType::F32).unwrap();
        sty.set_trace_id(id_s);
        let (dur_logits, _features) = prosody
            .forward(&inp, &sty)
            .map_err(|e| TensorError::Unsupported(e.to_string()))?;
        Ok(dur_logits)
    })
    .unwrap();

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("trace_to_graph_model_multi_input at d_en=64")
        .graph;

    let x_flat = batch * D_EN * seq_len;
    let style_flat = batch * STYLE_DIM;
    let output = propagate_multi_input_ibp(
        &gn,
        &[(&[x_flat], (-1.0, 1.0)), (&[style_flat], (-1.0, 1.0))],
    );
    assert_all_finite(&output, &format!("prosody d_en=64 mag={weight_mag}"));

    let (lo_min, hi_max) = bounds_min_max(&output);
    let (lo_arr, hi_arr) = output.lower_upper();
    let max_width: f32 = hi_arr
        .iter()
        .zip(lo_arr.iter())
        .map(|(h, l)| h - l)
        .fold(0.0f32, f32::max);

    eprintln!(
        "IBP d_en=64 prosody (mag={weight_mag}, {N_PROSODY_LAYERS} blocks):\n  \
         lo=[{lo_min:.6}], hi=[{hi_max:.6}]\n  \
         max_width={max_width:.6}, shape={:?}",
        output.lower().shape(),
    );

    assert!(
        lo_min.is_finite() && hi_max.is_finite(),
        "prosody bounds not finite at d_en=64 mag={weight_mag}: [{lo_min}, {hi_max}]"
    );
    max_width
}

// -- Tests --------------------------------------------------------------------

/// IBP at d_en=64 with conservative weights (mag=0.005).
///
/// Baseline: small weights keep bounds tight. Confirms the trace → graph →
/// IBP pipeline works correctly at intermediate scale before stress-testing.
#[test]
fn test_kokoro_ibp_scaling_d64_conservative() {
    let text_width = run_text_pipeline_ibp(0.005);
    let prosody_width = run_prosody_ibp(0.005);

    // Conservative weights should produce tight bounds at any scale.
    // Prosody threshold 25.0 accounts for DurationEncoder's BiLSTM+AdaLayerNorm
    // depth (each block amplifies bounds through bidirectional processing).
    assert!(
        text_width < 1.0,
        "conservative text pipeline width {text_width} unexpectedly wide"
    );
    assert!(
        prosody_width < 25.0,
        "conservative prosody width {prosody_width} unexpectedly wide"
    );
}

/// IBP at d_en=64 with Xavier-scale weights (mag=0.1).
///
/// Xavier initialization for 64→64: sqrt(2/128) ≈ 0.125. Using mag=0.1
/// approximates realistic weight magnitudes. This is the critical test:
/// if IBP bounds blow up at Xavier scale, CROWN linearization is needed
/// at every segment boundary for production verification.
///
/// Result interpretation:
/// - max_width < 100: IBP useful for production verification
/// - 100 < max_width < 1e6: IBP marginal, CROWN recommended
/// - max_width > 1e6: IBP vacuous, CROWN required at segment boundaries
#[test]
fn test_kokoro_ibp_scaling_d64_xavier() {
    let text_width = run_text_pipeline_ibp(0.1);
    let prosody_width = run_prosody_ibp(0.1);

    eprintln!(
        "\n=== IBP Scaling Summary (d_en=64, Xavier mag=0.1) ===\n\
         text_pipeline: max_width={text_width:.4}\n\
         prosody:       max_width={prosody_width:.4}\n\
         threshold:     1e6 (vacuous if exceeded)\n\
         verdict:       {}",
        if text_width < 1e6 && prosody_width < 1e6 {
            "IBP non-vacuous at d_en=64"
        } else {
            "CROWN required at segment boundaries"
        }
    );

    // Hard assertion: bounds must be finite (no NaN/Inf blow-up).
    // Width threshold is documented, not enforced — the test PASSES regardless
    // of width to capture the measurement. Follow-up filed if width > 1e6.
}
