// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Junction contract validation and pipeline certificate tests for the Kokoro
//! Generator sub-block segmented verification.
//!
//! Extracted from `compose_kokoro_generator_subblock.rs` to keep both files
//! under the 500-line limit (#2633).
//!
//! - AC5: Junction contracts (J3_MAGNITUDE, J3B_PHASE) from sub-block bounds
//! - AC6: Pipeline certificate with sub-block Generator bounds
//!
//! Part of #2633, Part of #2597, Part of #2218.

use nn_tts_verify::kokoro_contracts::{
    all_contracts, bounds_within_contract, contract_stage, max_contract_violation,
    JunctionContract, J3B_PHASE_LOWER, J3B_PHASE_UPPER, J3_MAGNITUDE_LOWER, J3_MAGNITUDE_UPPER,
};
use nn_tts_verify::{verify_pipeline, VerifiedStage};

use super::common::bounds_min_max;
use super::common::kokoro_weights::assert_all_finite;
use super::compose_kokoro_generator_subblock::{
    build_test_generator, record_subblock, trace_conv_pre, trace_output_conv_post_log_mag,
    trace_output_conv_post_phase, trace_upsample_stage,
};

// -- Helpers ------------------------------------------------------------------

/// Build a VerifiedStage for the Generator using actual sub-block IBP bounds.
///
/// Uses conv_post pre-activation bounds (log_magnitude + phase_raw) as the
/// Generator's output bounds, since junction contracts are defined at the
/// pre-activation boundary.
fn build_generator_stage_from_subblocks(
    generator: &nn_models::kokoro_decoder::Generator,
) -> VerifiedStage {
    let conv_pre_out = trace_conv_pre(generator);
    let (lo0, hi0) = bounds_min_max(&conv_pre_out);

    let up_out = trace_upsample_stage(generator, 0, (lo0, hi0));
    let (lo1, hi1) = bounds_min_max(&up_out);

    // Get conv_post bounds for both channels
    let log_mag = trace_output_conv_post_log_mag(generator, (lo1, hi1));
    let phase = trace_output_conv_post_phase(generator, (lo1, hi1));
    let (mag_lo, mag_hi) = bounds_min_max(&log_mag);
    let (phase_lo, phase_hi) = bounds_min_max(&phase);

    // Use the wider of the two channels as the uniform output bound
    let out_lo = f64::from(mag_lo.min(phase_lo));
    let out_hi = f64::from(mag_hi.max(phase_hi));

    let shape = [1, 256]; // representative shape
    let n = shape.iter().product::<usize>();
    let contracts = all_contracts();
    let j3_mag = &contracts[2]; // J3_MAGNITUDE — Generator input contract

    VerifiedStage::new(
        "kokoro_generator_subblock",
        shape.to_vec(),
        shape.to_vec(),
        vec![j3_mag.lower; n],
        vec![j3_mag.upper; n],
        vec![out_lo; n],
        vec![out_hi; n],
        "IBP-subblock",
        true, // IBP through standard layers (Conv1d/LeakyReLU/ConvTranspose/clamp/exp/sin) is sound
    )
}

// -- Tests --------------------------------------------------------------------

/// AC5: Junction contract validation — sub-block conv_post bounds satisfy
/// J3_MAGNITUDE and J3B_PHASE contracts from dvoice (#2597 AC3).
///
/// Traces the raw conv_post output (before clamp/exp/sin) and checks:
/// - log_magnitude bounds within J3_MAGNITUDE [-80, 80]
/// - phase_raw bounds within J3B_PHASE [-6283.2, 6283.2]
#[test]
fn test_generator_subblock_junction_contracts() {
    let generator = build_test_generator();

    // Run sub-block pipeline to get hidden state bounds at output stage input
    let conv_pre_out = trace_conv_pre(&generator);
    let (lo0, hi0) = bounds_min_max(&conv_pre_out);
    let up_out = trace_upsample_stage(&generator, 0, (lo0, hi0));
    let (lo1, hi1) = bounds_min_max(&up_out);
    eprintln!("Upsample output bounds for conv_post input: [{lo1}, {hi1}]");

    // Trace raw conv_post output (pre-activation) for both channels
    let log_mag_bounds = trace_output_conv_post_log_mag(&generator, (lo1, hi1));
    assert_all_finite(&log_mag_bounds, "conv_post/log_mag");
    let (log_mag_lo, log_mag_hi) = bounds_min_max(&log_mag_bounds);
    eprintln!("Conv_post log_magnitude IBP: [{log_mag_lo}, {log_mag_hi}]");

    let phase_bounds = trace_output_conv_post_phase(&generator, (lo1, hi1));
    assert_all_finite(&phase_bounds, "conv_post/phase_raw");
    let (phase_lo, phase_hi) = bounds_min_max(&phase_bounds);
    eprintln!("Conv_post phase_raw IBP: [{phase_lo}, {phase_hi}]");

    // Check J3_MAGNITUDE contract: pre-exp log magnitude within [-80, 80]
    let j3_mag = JunctionContract::new(
        "J3_MAGNITUDE",
        "Generator post_conv",
        J3_MAGNITUDE_LOWER,
        J3_MAGNITUDE_UPPER,
    );
    let (log_lo_arr, log_hi_arr) = log_mag_bounds.lower_upper();
    let log_lo_f64: Vec<f64> = log_lo_arr.iter().map(|&v| f64::from(v)).collect();
    let log_hi_f64: Vec<f64> = log_hi_arr.iter().map(|&v| f64::from(v)).collect();
    let mag_contained = bounds_within_contract(&j3_mag, &log_lo_f64, &log_hi_f64);
    let mag_violation = max_contract_violation(&j3_mag, &log_lo_f64, &log_hi_f64);
    eprintln!("J3_MAGNITUDE: contained={mag_contained}, max_violation={mag_violation:.6}");

    // Check J3B_PHASE contract: raw phase within [-6283.2, 6283.2]
    let j3b_phase = JunctionContract::new(
        "J3B_PHASE",
        "Generator post_conv",
        J3B_PHASE_LOWER,
        J3B_PHASE_UPPER,
    );
    let (phase_lo_arr, phase_hi_arr) = phase_bounds.lower_upper();
    let phase_lo_f64: Vec<f64> = phase_lo_arr.iter().map(|&v| f64::from(v)).collect();
    let phase_hi_f64: Vec<f64> = phase_hi_arr.iter().map(|&v| f64::from(v)).collect();
    let phase_contained = bounds_within_contract(&j3b_phase, &phase_lo_f64, &phase_hi_f64);
    let phase_violation = max_contract_violation(&j3b_phase, &phase_lo_f64, &phase_hi_f64);
    eprintln!("J3B_PHASE: contained={phase_contained}, max_violation={phase_violation:.6}");

    // At synthetic weight scale, both contracts should pass. With production
    // weights, bounds will be wider but still finite (unlike monolithic [-inf,inf]).
    assert!(
        mag_contained,
        "J3_MAGNITUDE contract violated: log_mag bounds [{log_mag_lo}, {log_mag_hi}] \
         exceed [{J3_MAGNITUDE_LOWER}, {J3_MAGNITUDE_UPPER}], violation={mag_violation:.6}"
    );
    assert!(
        phase_contained,
        "J3B_PHASE contract violated: phase_raw bounds [{phase_lo}, {phase_hi}] \
         exceed [{J3B_PHASE_LOWER}, {J3B_PHASE_UPPER}], violation={phase_violation:.6}"
    );

    // Record contract check result
    record_subblock("kokoro_generator_subblock_j3_check", &log_mag_bounds);
    eprintln!("Junction contract validation passed for synthetic weights");
}

/// AC6: Pipeline certificate is_valid with sub-block Generator bounds (#2597 AC3).
///
/// Composes Decoder -> Generator(sub-block) -> iSTFT and verifies the full
/// pipeline certificate is valid. This is the test that was structurally
/// impossible with monolithic Generator IBP bounds of [-inf, inf].
#[test]
fn test_generator_subblock_pipeline_certificate() {
    let contracts = all_contracts();

    // Stage 1: Decoder (from contracts)
    let decoder = contract_stage(
        "kokoro_decoder",
        &[1, 256],
        &[1, 256],
        &contracts[4], // J4_BF16 input
        &contracts[1], // J2_ENERGY output [-50, 50]
        "CROWN",
        true,
    );

    // Stage 2: Generator (from actual sub-block IBP bounds)
    let generator = build_test_generator();
    let gen_stage = build_generator_stage_from_subblocks(&generator);
    let (gen_lo, gen_hi) = (gen_stage.output_lower[0], gen_stage.output_upper[0]);
    eprintln!("Generator sub-block output bounds: [{gen_lo}, {gen_hi}]");

    // Stage 3: iSTFT (from contracts)
    let istft = contract_stage(
        "kokoro_istft",
        &[1, 256],
        &[1, 24000],
        &contracts[3], // J3B_PHASE input [-6283.2, 6283.2]
        &contracts[5], // J5_AUDIO output [-1, 1]
        "CROWN",
        true,
    );

    let cert = verify_pipeline(&[decoder, gen_stage, istft]).expect("valid pipeline");
    eprintln!("{}", cert.report());

    // Junction 0: Decoder output [-50, 50] ⊆ Generator input [-80, 80]
    assert!(
        cert.junctions[0].bounds_contained,
        "J0 decoder->generator violation: {}",
        cert.junctions[0].max_violation,
    );

    // Junction 1: Generator output (actual IBP) ⊆ iSTFT input [-6283.2, 6283.2]
    assert!(
        cert.junctions[1].bounds_contained,
        "J1 generator->istft violation: gen output [{gen_lo}, {gen_hi}], {}",
        cert.junctions[1].max_violation,
    );

    assert!(
        cert.is_valid,
        "Pipeline certificate must be valid with sub-block bounds"
    );
    eprintln!("Pipeline certificate VALID: sub-block Generator resolves [-inf, inf] (#2597 AC3)");
}
