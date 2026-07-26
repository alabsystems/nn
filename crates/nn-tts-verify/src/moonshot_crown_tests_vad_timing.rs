// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Silero VAD timing certificate tests (#1739 Phase 37).
//!
//! Wires the **real** Silero VAD dispatch plan (24 production steps from
//! `build_silero_vad_dispatch_plan_default()`) through the cost propagation
//! pipeline to produce a `TimingCertificate` and verify moonshot property P5
//! (temporal boundedness).
//!
//! This is the first real-model (not synthetic) end-to-end temporal
//! boundedness proof in the moonshot infrastructure. The Silero VAD model
//! architecture:
//!
//! - 4 Conv1d+ReLU encoder blocks (129→128→64→64→128 channels)
//! - Temporal mean pooling (128×T → 128)
//! - LSTM cell decomposed into 12 primitives (2 Linear + activations + state ops)
//! - Output stage: ReLU + Linear(128→1) + Sigmoid
//!
//! At 33 STFT frames (standard 512-sample chunk at 16kHz), worst-case inference
//! is well under 1ms on M4 Max — proving Silero VAD satisfies real-time
//! voice activity detection requirements.

use super::*;

use crate::cost_model::{
    estimate_peak_memory, profile_dispatch_plan, total_estimated_time_us, total_flops,
    total_memory_bytes, HardwareCostModel,
};
use crate::pipeline::verify_pipeline_with_timing;
use crate::silero_vad_dispatch::{
    build_silero_vad_dispatch_plan, build_silero_vad_dispatch_plan_default, TOTAL_EXPECTED_STEPS,
};

use super::vad_helpers::silero_vad_verified_stages;

// ---------------------------------------------------------------------------
// Silero VAD timing certificate tests
// ---------------------------------------------------------------------------

/// Default Silero VAD dispatch plan: verify_pipeline_with_timing produces a
/// TimingCertificate with worst-case < 1ms on M4 Max conservative.
#[test]
fn test_vad_timing_default_within_1ms() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    assert_eq!(dispatch_plan.len(), TOTAL_EXPECTED_STEPS);

    let encoder_dim = 128;
    let stages = silero_vad_verified_stages(encoder_dim);
    let hw = HardwareCostModel::m4_max_conservative();
    let timing_bound_us = 1_000.0; // 1ms

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, timing_bound_us)
        .expect("pipeline must verify");

    assert!(
        cert.timing_bound_met,
        "Silero VAD must complete in <1ms on M4 Max conservative, \
         got {:.1} us ({:.3} ms), bound={:.0} us",
        cert.worst_case_time_us,
        cert.worst_case_time_us / 1000.0,
        cert.timing_bound_us,
    );
    assert!(cert.bounds_cert.is_valid, "pipeline bounds must be valid");
    assert!(cert.overall_passed, "overall must pass");
}

/// Verify P5 (temporal boundedness) check produces CrownProven for the
/// Silero VAD timing certificate with a 1ms bound.
#[test]
fn test_vad_p5_crown_proven() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    let stages = silero_vad_verified_stages(128);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, 1_000.0)
        .expect("pipeline must verify");

    let result = check_temporal_boundedness(&cert);
    assert!(result.proven, "P5 must be proven: {}", result.explanation);
    assert_eq!(
        result.level,
        VerificationLevel::CrownProven,
        "P5 must be CrownProven for sound pipeline"
    );
    assert_eq!(
        result.property_index, 4,
        "temporal boundedness is P5 (index 4)"
    );
    assert!(result.is_sound, "pipeline should be sound");
}

/// Verify that total FLOPs are in the expected range for the small Silero VAD.
///
/// Silero VAD is tiny: 4 Conv1d layers + LSTM + Linear. Expected: 10K-10M FLOPs.
#[test]
fn test_vad_flop_count_realistic() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    let hw = HardwareCostModel::m4_max_conservative();
    let profiles = profile_dispatch_plan(&dispatch_plan, &hw);
    let flops = total_flops(&profiles);

    assert!(
        flops >= 10_000,
        "Silero VAD should have >=10K FLOPs, got {flops}"
    );
    assert!(
        flops <= 100_000_000,
        "Silero VAD should have <=100M FLOPs (it's tiny), got {flops}"
    );
}

/// Verify memory traffic is realistic for the tiny Silero VAD model.
///
/// Silero VAD has ~200K parameters. Expected: 100KB-100MB total memory traffic.
#[test]
fn test_vad_memory_traffic_realistic() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    let hw = HardwareCostModel::m4_max_conservative();
    let profiles = profile_dispatch_plan(&dispatch_plan, &hw);
    let mem = total_memory_bytes(&profiles);

    let mem_kb = mem as f64 / 1024.0;
    assert!(
        mem_kb >= 100.0,
        "Silero VAD should use >=100 KB memory traffic, got {mem_kb:.1} KB"
    );
    let mem_mb = mem_kb / 1024.0;
    assert!(
        mem_mb <= 100.0,
        "Silero VAD should use <=100 MB memory traffic, got {mem_mb:.1} MB"
    );
}

/// Peak memory should be well under 1MB for the tiny Silero VAD.
#[test]
fn test_vad_peak_memory_small() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    let mem = estimate_peak_memory(&dispatch_plan);

    assert!(
        mem.peak_total_mb() < 1.0,
        "Silero VAD peak memory {:.3} MB should be < 1 MB",
        mem.peak_total_mb()
    );
}

/// Silero VAD with more STFT frames (65 frames = ~1 second audio).
/// Should still easily be under 1ms.
#[test]
fn test_vad_timing_65_frames_within_1ms() {
    let dispatch_plan = build_silero_vad_dispatch_plan(65);
    let stages = silero_vad_verified_stages(128);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, 1_000.0)
        .expect("pipeline must verify");

    assert!(
        cert.timing_bound_met,
        "65-frame VAD must be <1ms, got {:.1} us ({:.3} ms)",
        cert.worst_case_time_us,
        cert.worst_case_time_us / 1000.0,
    );
    assert!(cert.overall_passed);
}

/// Boundary test: extremely tight timing bound (1μs) should fail.
/// Proves the timing bound is meaningful and not trivially satisfied.
#[test]
fn test_vad_timing_tight_bound_fails() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    let stages = silero_vad_verified_stages(128);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, 1.0) // 1μs
        .expect("pipeline must verify");

    assert!(
        !cert.timing_bound_met,
        "1μs bound should fail for Silero VAD: estimated={:.1} us",
        cert.worst_case_time_us,
    );
    assert!(!cert.overall_passed, "overall should fail with tight bound");

    // P5 should NOT be proven with a failed timing bound.
    let result = check_temporal_boundedness(&cert);
    assert!(
        !result.proven,
        "P5 must NOT be proven when timing exceeds bound"
    );
    assert_eq!(result.level, VerificationLevel::Empirical);
}

/// Theoretical M4 Max model should be faster than conservative model.
#[test]
fn test_vad_theoretical_faster_than_conservative() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    let hw_theoretical = HardwareCostModel::m4_max();
    let hw_conservative = HardwareCostModel::m4_max_conservative();

    let profiles_t = profile_dispatch_plan(&dispatch_plan, &hw_theoretical);
    let profiles_c = profile_dispatch_plan(&dispatch_plan, &hw_conservative);

    let time_t = total_estimated_time_us(&profiles_t);
    let time_c = total_estimated_time_us(&profiles_c);

    assert!(
        time_c > time_t,
        "conservative ({time_c:.1} us) should be slower than theoretical ({time_t:.1} us)"
    );
}

/// Boundary test: timing at the exact estimated bound should pass (<=).
#[test]
fn test_vad_timing_at_exact_bound() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    let stages = silero_vad_verified_stages(128);
    let hw = HardwareCostModel::m4_max_conservative();

    let profiles = profile_dispatch_plan(&dispatch_plan, &hw);
    let actual_time = total_estimated_time_us(&profiles);

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, actual_time)
        .expect("pipeline must verify");

    assert!(
        cert.timing_bound_met,
        "exact-bound should pass (<=): time={actual_time:.1} us"
    );
}

/// Report output includes hardware and timing information.
#[test]
fn test_vad_timing_report_format() {
    let dispatch_plan = build_silero_vad_dispatch_plan_default();
    let stages = silero_vad_verified_stages(128);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, 1_000.0)
        .expect("pipeline must verify");

    let report = cert.report();
    assert!(
        report.contains("Timing Verification Report"),
        "should have report header"
    );
    assert!(report.contains("PASS"), "should pass");
    assert!(report.contains("2.8 TFLOPS"), "conservative model");
    assert!(report.contains("200 GB/s"), "conservative bandwidth");
    assert!(report.contains("Total FLOPs:"), "should show FLOPs");
}
