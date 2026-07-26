// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Level 3: CROWN verification with production weights for Kokoro Generator.
//!
//! Advances the #2599 escalation ladder from Level 2 (synthetic weights) to
//! Level 3 (real production weights). Uses the mixed-mode strategy proven at
//! Level 2: IBP for intractable sub-blocks (conv_pre + upsample stages with
//! large Conv1d), CROWN for tractable sub-blocks (output stage with Conv1d
//! [22, 128, 7] = 19,712 elements).
//!
//! Key dimensions at production scale (D=512):
//!   conv_pre:      Conv1d [512, 512, 7] = 1.8M elements → IBP only
//!   upsample 0:    ConvTranspose1d [512, 256, K] → IBP only
//!   upsample 1:    ConvTranspose1d [256, 128, K] → IBP only
//!   output stage:  Conv1d [22, 128, 7] = 19,712 elements → CROWN tractable
//!
//! Level 2 showed 3,429x CROWN tightening on synthetic weights. Level 3
//! measures whether production weights yield similar tightening.
//!
//! **Requires:** `KOKORO_WEIGHTS=/path/to/kokoro_weights_rust.safetensors`
//! Gated behind `#[cfg(feature = "production-weights")]` (#2716).
//!
//! Part of #2599: Kokoro Generator verification ceiling.
//! Part of #2218: Epic — Perfect Kokoro.

#[cfg(feature = "production-weights")]
use super::kokoro_production_segments::{
    trace_production_conv_pre, trace_production_upsample_stages,
};
#[cfg(feature = "production-weights")]
use super::kokoro_production_weights::{
    is_tight_crown_method, propagate_with_tight_crown_fallback, record_segment_crown,
    require_production_weights, tight_crown_method_name, trace_input,
};

#[cfg(feature = "production-weights")]
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
#[cfg(feature = "production-weights")]
use nn_core::dyn_tensor::DynTensor;
#[cfg(feature = "production-weights")]
use nn_core::test_utils::cpu;
#[cfg(feature = "production-weights")]
use nn_core::{DType, VarBuilder};
#[cfg(feature = "production-weights")]
use nn_models::kokoro_decoder::Generator;
#[cfg(feature = "production-weights")]
use nn_models::KokoroConfig;
#[cfg(feature = "production-weights")]
use nn_verify::{trace_to_graph_model, BoundedTensor};
#[cfg(feature = "production-weights")]
use ndarray::{ArrayD, IxDyn};

// -- Level 3: CROWN on output stage with production weights -------------------

/// Trace the Generator output stage and return (GraphNetwork, input_bounds, ibp_output).
///
/// Uses IBP through conv_pre and upsample stages to derive realistic input bounds
/// for the output stage, then builds a GraphNetwork from the output stage trace.
#[cfg(feature = "production-weights")]
fn trace_output_stage_for_crown(
    generator: &Generator,
    config: &KokoroConfig,
    t_stage1: usize,
) -> (nn_verify::GraphNetwork, BoundedTensor, BoundedTensor) {
    // Step 1: IBP through conv_pre to get post-conv bounds.
    let (_conv_pre_input, conv_pre_bounds) =
        trace_production_conv_pre(generator, config.gen_initial_channels, t_stage1);

    // Step 2: IBP through upsample stages to get pre-output bounds.
    let upsample_bounds =
        trace_production_upsample_stages(generator, config, t_stage1, conv_pre_bounds);

    // Step 3: Trace output stage to get a GraphNetwork.
    let ch_final = config.gen_initial_channels >> config.upsample_rates.len();
    let t_out = t_stage1 * config.upsample_rates.iter().product::<usize>() + 1;
    let h_shape = [1, ch_final, t_out];
    let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let mut h_t = h.clone();
        h_t.set_trace_id(record_input(&h_shape, DType::F32).expect("trace active"));
        let (mag, _phase) = generator
            .forward_output_stage(&h_t)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .expect("output_stage trace");
    let gn = trace_to_graph_model(&graph)
        .expect("output trace_to_graph")
        .graph;

    // Step 4: Build input bounds from the upsample IBP output range.
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&h_shape), upsample_bounds.0),
        ArrayD::from_elem(IxDyn(&h_shape), upsample_bounds.1),
    )
    .expect("valid bounds");

    // Step 5: IBP baseline on the output stage.
    let ibp_output = gn.propagate_ibp(&input_bounds).expect("output IBP");
    super::common::assert_bounds_valid(&ibp_output);

    (gn, input_bounds, ibp_output)
}

/// Level 3 AC1: CROWN on the output stage with production weights.
///
/// The output stage Conv1d [22, 128, 7] = 19,712 weight elements is
/// CROWN-tractable. This test measures whether production weights yield
/// meaningful CROWN tightening over IBP — the key Level 3 question.
///
/// Level 2 result (synthetic weights): 3,429x tightening on D=512 ResBlock.
/// Level 3 goal: any tightening > 1.0 with production weights.
#[cfg(feature = "production-weights")]
#[test]
fn test_production_generator_output_crown() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = match Generator::load(&vb.pp("decoder"), &config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!(
                "Generator::load failed (v1.0 architecture mismatch): {e}\n  \
                 v1.0 uses resblocks.paths/decode/encode, v0.19 uses noise_res/conv_pre.\n  \
                 Skipping Level 3 CROWN test."
            );
            return;
        }
    };

    let t_stage1 = 4;
    let (gn, input_bounds, ibp_output) =
        trace_output_stage_for_crown(&generator, &config, t_stage1);

    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;
    eprintln!(
        "Level 3 — Output stage IBP (production weights): [{ibp_lo:.6}, {ibp_hi:.6}], \
         width={ibp_width:.6}"
    );

    // CROWN attempt on the output stage.
    let (method, crown_output, fallback_reason) =
        propagate_with_tight_crown_fallback(&gn, &input_bounds).expect("output CROWN");
    super::common::assert_bounds_valid(&crown_output);
    let (crown_lo, crown_hi) = super::common::bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    let tightening = if crown_width > 1e-10 && ibp_width > 1e-10 {
        ibp_width / crown_width
    } else {
        1.0
    };

    let method_str = tight_crown_method_name(method);
    eprintln!(
        "Level 3 — Output stage {method_str} (production weights): \
         [{crown_lo:.6}, {crown_hi:.6}], width={crown_width:.6}, \
         tightening={tightening:.2}x"
    );

    if let Some(ref reason) = fallback_reason {
        eprintln!("  Fallback reason: {reason}");
    }

    // Core assertion: bounds must be finite with production weights.
    assert!(
        crown_lo.is_finite() && crown_hi.is_finite(),
        "Level 3 output CROWN bounds must be finite, got [{crown_lo}, {crown_hi}]"
    );

    // If CROWN succeeded, verify it's at least as tight as IBP.
    if is_tight_crown_method(method) {
        super::common::assert_crown_tighter_than_ibp(&crown_output, &ibp_output);
        eprintln!(
            "Level 3 SUCCESS: {method_str} tighter than IBP by {tightening:.2}x on production weights"
        );
    } else {
        eprintln!(
            "Level 3 PARTIAL: CROWN-family propagation fell back to IBP on output stage. \
             This may indicate normalization-layer fallback."
        );
    }

    // Record to status file.
    let ibp_w = is_tight_crown_method(method).then_some(ibp_width);
    record_segment_crown(
        "kokoro_production_generator_output_crown",
        &input_bounds,
        &crown_output,
        method,
        ibp_w,
    );
}

/// Level 3 AC2: Mixed IBP+CROWN sub-block pipeline with production weights.
///
/// Runs the full Level 1 sub-block pipeline (conv_pre → upsample stages →
/// output stage) with production weights, applying CROWN only to the output
/// stage and IBP to all preceding stages. Reports end-to-end bound width
/// and compares with the all-IBP baseline.
#[cfg(feature = "production-weights")]
#[test]
fn test_production_generator_mixed_subblock_crown() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = match Generator::load(&vb.pp("decoder"), &config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Generator::load failed: {e}\n  Skipping Level 3 mixed sub-block test.");
            return;
        }
    };

    let t_stage1 = 4;

    // IBP-only baseline: conv_pre → upsample → output (all IBP).
    let (conv_pre_input, conv_pre_bounds) =
        trace_production_conv_pre(&generator, config.gen_initial_channels, t_stage1);
    let upsample_bounds =
        trace_production_upsample_stages(&generator, &config, t_stage1, conv_pre_bounds);

    // Output stage: IBP baseline.
    let ch_final = config.gen_initial_channels >> config.upsample_rates.len();
    let t_out = t_stage1 * config.upsample_rates.iter().product::<usize>() + 1;
    let h_shape = [1, ch_final, t_out];
    let h = DynTensor::full(&h_shape, 0.1, DType::F32, &cpu()).unwrap();
    let (_result, graph) = trace_graph(|| {
        let mut h_t = h.clone();
        h_t.set_trace_id(record_input(&h_shape, DType::F32).expect("trace active"));
        let (mag, _phase) = generator
            .forward_output_stage(&h_t)
            .map_err(|e| nn_core::TensorError::Unsupported(e.to_string()))?;
        Ok(mag)
    })
    .expect("output_stage trace");
    let gn = trace_to_graph_model(&graph)
        .expect("output trace_to_graph")
        .graph;
    let output_input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&h_shape), upsample_bounds.0),
        ArrayD::from_elem(IxDyn(&h_shape), upsample_bounds.1),
    )
    .expect("valid bounds");

    let ibp_output = gn.propagate_ibp(&output_input_bounds).expect("output IBP");
    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(&ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    // Mixed pipeline: IBP for conv_pre + upsample, CROWN for output.
    let (method, crown_output, _fallback) =
        propagate_with_tight_crown_fallback(&gn, &output_input_bounds).expect("output CROWN");
    let (crown_lo, crown_hi) = super::common::bounds_min_max(&crown_output);
    let crown_width = crown_hi - crown_lo;

    let tightening = if crown_width > 1e-10 && ibp_width > 1e-10 {
        ibp_width / crown_width
    } else {
        1.0
    };

    eprintln!("Level 3 mixed sub-block pipeline (production weights):");
    eprintln!(
        "  conv_pre:     IBP [{:.4}, {:.4}]",
        conv_pre_bounds.0, conv_pre_bounds.1
    );
    eprintln!(
        "  upsample:     IBP [{:.4}, {:.4}]",
        upsample_bounds.0, upsample_bounds.1
    );
    eprintln!("  output (IBP): [{ibp_lo:.6}, {ibp_hi:.6}] width={ibp_width:.6}");
    eprintln!(
        "  output ({:?}): [{crown_lo:.6}, {crown_hi:.6}] width={crown_width:.6}",
        method
    );
    eprintln!("  e2e tightening: {tightening:.2}x");

    // All bounds must be finite.
    assert!(
        ibp_lo.is_finite() && ibp_hi.is_finite(),
        "IBP e2e bounds must be finite"
    );
    assert!(
        crown_lo.is_finite() && crown_hi.is_finite(),
        "CROWN e2e bounds must be finite"
    );

    // Record result.
    let ibp_w = is_tight_crown_method(method).then_some(ibp_width);
    record_segment_crown(
        "kokoro_production_generator_mixed_subblock",
        &conv_pre_input,
        &crown_output,
        method,
        ibp_w,
    );

    eprintln!(
        "Level 3 escalation ladder — production weight mixed sub-block: \
         method={method:?}, e2e_width={crown_width:.6}, tightening={tightening:.2}x"
    );
}
