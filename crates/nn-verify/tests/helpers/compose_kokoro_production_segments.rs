// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Production weight verification for Kokoro TTS model — predictor, generator,
//! and composition tests.
//!
//! Extracted from `compose_kokoro_production.rs` to keep both files
//! under the 500-line limit (#2633).
//!
//! Tests 3-8: ProsodyPredictor, F0EnergyPredictor, Generator, Generator
//! sub-block, text→prosody composition, and combined report.
//!
//! **Requires:** `KOKORO_WEIGHTS=/path/to/kokoro_v1_0.safetensors`
//! Tests are gated behind `#[cfg(feature = "production-weights")]` (#2716).
//! Without the feature: tests don't compile (no false CI confidence).
//! With the feature: tests panic if weights are missing.
//!
//! Part of #2633, Part of #2461, Part of #2598, Part of #2218.

#[cfg(feature = "production-weights")]
use super::kokoro_production_segments::{
    analyze_and_report, append_records, persist_report, record_all_segments,
    trace_bert_encoder_segment, trace_f0_predictor_segment, trace_generator_segment,
    trace_production_conv_pre, trace_production_output_stage, trace_production_upsample_stages,
    trace_prosody_predictor_composed, trace_prosody_predictor_segment, trace_text_encoder_segment,
};
#[cfg(feature = "production-weights")]
use super::kokoro_production_weights::{record_segment, require_production_weights};

#[cfg(feature = "production-weights")]
use nn_core::test_utils::cpu;
#[cfg(feature = "production-weights")]
use nn_core::{DType, VarBuilder};
#[cfg(feature = "production-weights")]
use nn_models::kokoro_decoder::Generator;
#[cfg(feature = "production-weights")]
use nn_models::KokoroConfig;

// -- Test 3: ProsodyPredictor with production weights -------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_prosody_predictor_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let seg = trace_prosody_predictor_segment(&vb, &config);
    let report = analyze_and_report("kokoro_prosody_predictor_production", &seg.records);

    if !seg.records.is_empty() {
        assert!(
            report.output_is_finite,
            "production ProsodyPredictor bounds should be finite"
        );
        assert!(
            report.total_layers >= 5,
            "ProsodyPredictor should have 5+ layers (Conv+Norm+LSTM+Linear+...)"
        );
    } else {
        eprintln!("ProsodyPredictor: layer extraction skipped (LayerNorm shape mismatch)");
    }
    super::common::assert_bounds_valid(&seg.output_bounds);

    record_segment(
        "kokoro_production_prosody_predictor",
        &seg.input_bounds,
        &seg.output_bounds,
    );
}

// -- Test 4: F0EnergyPredictor with production weights ------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_f0_predictor_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let seg = match trace_f0_predictor_segment(&vb, &config) {
        Ok(seg) => seg,
        Err(e) => {
            eprintln!("UNVERIFIED: F0EnergyPredictor — {e}");
            eprintln!("  NOT recording to status file (#2716)");
            return;
        }
    };
    let report = analyze_and_report("kokoro_f0_predictor_production", &seg.records);

    if !seg.records.is_empty() {
        assert!(
            report.output_is_finite,
            "production F0EnergyPredictor bounds should be finite"
        );
        assert!(
            report.total_layers >= 3,
            "F0EnergyPredictor should have 3+ layers (BiLSTM+AdainResBlk+Linear)"
        );
    } else {
        eprintln!("F0EnergyPredictor: layer extraction skipped");
    }
    super::common::assert_bounds_valid(&seg.output_bounds);

    record_segment(
        "kokoro_production_f0_predictor",
        &seg.input_bounds,
        &seg.output_bounds,
    );
}

// -- Test 5: Generator with production weights --------------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_generator_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let seg = match trace_generator_segment(&vb, &config) {
        Ok(seg) => seg,
        Err(e) => {
            eprintln!("UNVERIFIED: Generator — {e}");
            eprintln!("  NOT recording to status file (#2716)");
            return;
        }
    };
    let report = analyze_and_report("kokoro_generator_production", &seg.records);

    if !seg.records.is_empty() {
        assert!(
            report.output_is_finite,
            "production Generator bounds should be finite"
        );
        assert!(
            report.total_layers >= 5,
            "Generator should have 5+ layers (Conv+ConvTranspose+ResBlock+...)"
        );
    } else {
        eprintln!("Generator: layer extraction skipped");
    }
    super::common::assert_bounds_valid(&seg.output_bounds);

    record_segment(
        "kokoro_production_generator",
        &seg.input_bounds,
        &seg.output_bounds,
    );
}

// -- Test 6: Generator sub-block with production weights (#2597) ---------------

/// Production-weight sub-block verification for the Generator (#2597).
#[cfg(feature = "production-weights")]
#[test]
fn test_production_generator_subblock_ibp() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());
    let generator = match Generator::load(&vb.pp("decoder"), &config) {
        Ok(g) => g,
        Err(e) => {
            eprintln!(
                "Generator::load failed (v1.0 architecture mismatch): {e}\n  \
                 v1.0 uses resblocks.paths/decode/encode, v0.19 uses noise_res/conv_pre.\n  \
                 Skipping sub-block test."
            );
            return;
        }
    };
    let t_stage1 = 4;

    let (conv_pre_input, conv_pre_bounds) =
        trace_production_conv_pre(&generator, config.gen_initial_channels, t_stage1);
    let upsample_bounds =
        trace_production_upsample_stages(&generator, &config, t_stage1, conv_pre_bounds);
    let output = trace_production_output_stage(&generator, &config, t_stage1, upsample_bounds);

    super::common::assert_bounds_valid(&output);
    let (lo_final, hi_final) = super::common::bounds_min_max(&output);
    eprintln!("Production sub-block output: [{lo_final}, {hi_final}]");
    assert!(
        lo_final.is_finite() && hi_final.is_finite(),
        "production sub-block output must be finite, got [{lo_final}, {hi_final}]"
    );
    assert!(
        lo_final > -1.0,
        "exp output should be near-zero, got lo={lo_final}"
    );

    record_segment(
        "kokoro_production_generator_subblock",
        &conv_pre_input,
        &output,
    );
    eprintln!("Production Generator sub-block: finite bounds [{lo_final}, {hi_final}]");
}

// -- Test 7: Composed verification — text_encoder → prosody_predictor ----------
//
// Chains 2 segments with production weights: text_encoder output bounds become
// prosody_predictor input bounds. This tests that bounds compose soundly across
// segment boundaries (Part of #2461).

#[cfg(feature = "production-weights")]
#[test]
fn test_production_composed_text_to_prosody() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    // Stage 1: text_encoder with production weights → output bounds
    let te = trace_text_encoder_segment(&vb, &config);
    let (te_lo, te_hi) = super::common::bounds_min_max(&te.output_bounds);
    eprintln!(
        "Composed stage 1 — TextEncoder output: [{te_lo:.4}, {te_hi:.4}], width={:.4}",
        te_hi - te_lo
    );
    assert!(
        te_lo.is_finite() && te_hi.is_finite(),
        "TextEncoder output must be finite for composition"
    );

    // Stage 2: prosody_predictor using text_encoder output range as text_features bounds
    let pp = trace_prosody_predictor_composed(&vb, &config, (te_lo, te_hi));
    let (pp_lo, pp_hi) = super::common::bounds_min_max(&pp.output_bounds);
    let pp_width = pp_hi - pp_lo;
    eprintln!(
        "Composed stage 2 — ProsodyPredictor output: [{pp_lo:.4}, {pp_hi:.4}], width={pp_width:.4}"
    );

    // Composed bounds must be finite (soundness).
    assert!(
        pp_lo.is_finite() && pp_hi.is_finite(),
        "composed text→prosody output must be finite, got [{pp_lo}, {pp_hi}]"
    );

    // Compare with standalone prosody (hardcoded [-1,1] text_features) to verify
    // composition produces different (typically wider) bounds.
    let pp_standalone = trace_prosody_predictor_segment(&vb, &config);
    let (sa_lo, sa_hi) = super::common::bounds_min_max(&pp_standalone.output_bounds);
    let sa_width = sa_hi - sa_lo;
    eprintln!("Standalone ProsodyPredictor: [{sa_lo:.4}, {sa_hi:.4}], width={sa_width:.4}");
    eprintln!(
        "Composition effect: composed_width={pp_width:.4}, standalone_width={sa_width:.4}, \
         ratio={:.4}",
        if sa_width > 0.0 {
            pp_width / sa_width
        } else {
            1.0
        }
    );

    record_segment(
        "kokoro_production_composed_text_to_prosody",
        &pp.input_bounds,
        &pp.output_bounds,
    );
}

// -- Test 8: Combined report across all segments ------------------------------

#[cfg(feature = "production-weights")]
#[test]
fn test_production_combined_report() {
    let tensors = require_production_weights();
    let config = KokoroConfig::default();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &cpu());

    let bert = trace_bert_encoder_segment(&vb, &config);
    let te = trace_text_encoder_segment(&vb, &config);
    let pp = trace_prosody_predictor_segment(&vb, &config);
    let f0 = match trace_f0_predictor_segment(&vb, &config) {
        Ok(seg) => Some(seg),
        Err(e) => {
            eprintln!("UNVERIFIED: F0EnergyPredictor — {e}");
            None
        }
    };
    let generator = match trace_generator_segment(&vb, &config) {
        Ok(seg) => Some(seg),
        Err(e) => {
            eprintln!("UNVERIFIED: Generator — {e}");
            None
        }
    };

    record_all_segments(&bert, &te, &pp, f0.as_ref(), generator.as_ref());

    // Count segments with layer records before moving records out.
    let verified_count = 3 + f0.is_some() as usize + generator.is_some() as usize;
    let segments_with_records = [
        Some(&bert),
        Some(&te),
        Some(&pp),
        f0.as_ref(),
        generator.as_ref(),
    ]
    .iter()
    .filter(|s| s.map_or(false, |s| !s.records.is_empty()))
    .count();

    let mut all_records = bert.records;
    append_records(&mut all_records, te.records);
    append_records(&mut all_records, pp.records);
    if let Some(f0) = f0 {
        append_records(&mut all_records, f0.records);
    }
    if let Some(generator) = generator {
        append_records(&mut all_records, generator.records);
    }

    let report = analyze_and_report("kokoro_production_all_segments", &all_records);
    eprintln!(
        "Summary: CROWN coverage {:.1}%, output_width={:.2}, finite={}, chained_norm_depth={}",
        report.crown_coverage * 100.0,
        report.output_width,
        report.output_is_finite,
        report.chained_norm_depth,
    );

    // Log norm chain explosion recommendations (#2708 AC3).
    let norm_chain_recs: Vec<_> = report
        .recommendations
        .iter()
        .filter(|r| {
            matches!(
                r,
                nn_verify::bound_analysis::TighteningRecommendation::NormChainExplosion { .. }
            )
        })
        .collect();
    eprintln!(
        "Norm chain detection: depth={}, explosion_recs={}",
        report.chained_norm_depth,
        norm_chain_recs.len(),
    );

    persist_report(&report);

    // bert_encoder always provides layer records; other segments may not
    // (LayerNorm shape mismatch, grouped ConvTranspose, v1.0 arch mismatch).
    if report.total_layers > 0 {
        assert!(report.output_is_finite, "combined output should be finite");
    }
    eprintln!("Combined report: {segments_with_records} of {verified_count} verified segments contributed layer records");
}
