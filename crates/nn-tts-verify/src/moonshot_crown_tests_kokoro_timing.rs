// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kokoro-scale timing certificate tests (#1739 AC6, #1741 P5).
//!
//! Uses the production-accurate dispatch plan from `kokoro_dispatch.rs`
//! with actual `DispatchStep` variants from `KokoroConfig::default()`, profiles
//! through `HardwareCostModel::m4_max_conservative()`, and verifies the
//! `TimingCertificate` proves worst-case inference < 50ms on M4 Max.
//!
//! Architecture modeled (Kokoro-82M Generator / ISTFTNet vocoder):
//! - conv_pre: Conv1d(512, 512, k=7, pad=3)
//! - 2 upsample stages: LeakyReLU → ConvTranspose1d → noise injection → 3 ResBlocks
//! - ResBlock per dilation: AdaIN(Linear) → Snake(Sigmoid) → Conv1d → repeat → add
//! - conv_post: Conv1d(128, 22, k=7) → exp(Tanh) + sin(Tanh)
//!
//! Total: 181 steps. For 100 tokens → 6000 audio frames (10×6 upsample).

use super::*;

use nn_dsl::DispatchStep;

use crate::cost_model::{
    profile_dispatch_plan, total_estimated_time_us, total_flops, total_memory_bytes,
    HardwareCostModel,
};
use crate::kokoro_dispatch::{build_kokoro_dispatch_plan, TOTAL_EXPECTED_STEPS};
use crate::pipeline::verify_pipeline_with_timing;

use super::kokoro_helpers::kokoro_verified_stages;

// ---------------------------------------------------------------------------
// Kokoro timing certificate tests
// ---------------------------------------------------------------------------

/// Kokoro vocoder dispatch plan at 100 tokens: verify_pipeline_with_timing
/// produces a TimingCertificate with worst-case < 50ms on M4 Max conservative.
#[test]
fn test_kokoro_timing_100_tokens_within_50ms() {
    let seq_len = 100;
    let dim = 192; // CROWN verification dimension

    let (dispatch_plan, final_t) = build_kokoro_dispatch_plan(seq_len);
    let stages = kokoro_verified_stages(dim);
    let hw = HardwareCostModel::m4_max_conservative();
    let timing_bound_us = 50_000.0; // 50ms

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, timing_bound_us)
        .expect("pipeline must verify");

    assert!(
        cert.timing_bound_met,
        "Kokoro 100-token vocoder must complete in <50ms on M4 Max conservative, \
         got {:.1} us ({:.2} ms), bound={:.0} us",
        cert.worst_case_time_us,
        cert.worst_case_time_us / 1000.0,
        cert.timing_bound_us,
    );
    assert!(cert.bounds_cert.is_valid, "pipeline bounds must be valid");
    assert!(cert.overall_passed, "overall must pass");

    // Verify final time dimension: 100 * 10 * 6 = 6000 audio frames
    assert_eq!(final_t, 6000, "100 tokens x 10 x 6 upsample = 6000 frames");
}

/// Verify production dispatch plan step counts match architecture constants.
///
/// Production plan: conv_pre(1) + 2×stage(88) + output(4) = 181 steps.
/// Step types: 52 Conv1d, 2 ConvTranspose1d, 48 Linear, 51 Sigmoid, 2 Tanh, 26 Add.
#[test]
fn test_kokoro_dispatch_plan_step_count() {
    let (plan, _) = build_kokoro_dispatch_plan(100);

    // Count step types
    let conv1d_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Conv1d(_)))
        .count();
    let conv_t_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::ConvTranspose1d(_)))
        .count();
    let linear_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Linear { .. }))
        .count();
    let tanh_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Tanh { .. }))
        .count();
    let sigmoid_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Sigmoid { .. }))
        .count();
    let add_count = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::BinaryAdd { .. }))
        .count();

    // Production architecture exact counts (validated against kokoro_dispatch_tests.rs)
    assert_eq!(
        conv1d_count, 52,
        "52 Conv1d (conv_pre + noise + ResBlock + conv_post)"
    );
    assert_eq!(conv_t_count, 2, "2 ConvTranspose1d upsample stages");
    assert_eq!(linear_count, 48, "48 AdaIN Linear projections");
    assert_eq!(sigmoid_count, 51, "51 Sigmoid (Snake + LeakyReLU proxies)");
    assert_eq!(tanh_count, 2, "2 Tanh (exp_magnitude + sin_phase)");
    assert_eq!(add_count, 26, "26 BinaryAdd (residual + noise)");

    // Total must match the production constant
    assert_eq!(
        plan.len(),
        TOTAL_EXPECTED_STEPS,
        "production plan must have {TOTAL_EXPECTED_STEPS} steps"
    );
}

/// Kokoro total FLOPs are in the expected range for an 82M-parameter TTS model.
///
/// At 100 tokens with 60x total upsampling (10x6), the vocoder processes
/// 6000 audio frames through 128-channel ResBlocks. Expected: 100M-10B FLOPs.
#[test]
fn test_kokoro_flop_count_realistic() {
    let (plan, _) = build_kokoro_dispatch_plan(100);
    let hw = HardwareCostModel::m4_max_conservative();
    let profiles = profile_dispatch_plan(&plan, &hw);
    let flops = total_flops(&profiles);

    assert!(
        flops >= 100_000_000,
        "Kokoro should have >=100M FLOPs, got {flops}"
    );
    assert!(
        flops <= 100_000_000_000,
        "Kokoro should have <=100B FLOPs, got {flops}"
    );
}

/// Kokoro memory traffic is in the expected range.
///
/// Model weights (~82M params x 4 bytes) + activations. Expected: 10MB-10GB.
#[test]
fn test_kokoro_memory_traffic_realistic() {
    let (plan, _) = build_kokoro_dispatch_plan(100);
    let hw = HardwareCostModel::m4_max_conservative();
    let profiles = profile_dispatch_plan(&plan, &hw);
    let mem = total_memory_bytes(&profiles);

    let mem_mb = mem as f64 / (1024.0 * 1024.0);
    assert!(
        mem_mb >= 10.0,
        "Kokoro should use >=10 MB memory traffic, got {mem_mb:.1} MB"
    );
    assert!(
        mem_mb <= 10_000.0,
        "Kokoro should use <=10 GB memory traffic, got {mem_mb:.1} MB"
    );
}

/// Short utterance (10 tokens) -- fastest case, well within timing bound.
#[test]
fn test_kokoro_timing_10_tokens_fast() {
    let seq_len = 10;
    let dim = 192;

    let (dispatch_plan, final_t) = build_kokoro_dispatch_plan(seq_len);
    let stages = kokoro_verified_stages(dim);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, 50_000.0)
        .expect("pipeline must verify");

    assert!(cert.timing_bound_met, "10-token Kokoro must be fast");
    assert!(cert.overall_passed);
    assert_eq!(final_t, 600, "10 tokens x 60 = 600 frames");

    // Short utterance should be significantly faster than 50ms
    assert!(
        cert.worst_case_time_us < 20_000.0,
        "10-token should be <20ms, got {:.1} ms",
        cert.worst_case_time_us / 1000.0
    );
}

/// Boundary test: timing exactly at bound.
#[test]
fn test_kokoro_timing_at_exact_bound() {
    let (dispatch_plan, _) = build_kokoro_dispatch_plan(100);
    let stages = kokoro_verified_stages(192);
    let hw = HardwareCostModel::m4_max_conservative();

    // Profile to find actual estimated time, then use that as the bound.
    let profiles = profile_dispatch_plan(&dispatch_plan, &hw);
    let actual_time = total_estimated_time_us(&profiles);

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, actual_time)
        .expect("pipeline must verify");

    assert!(
        cert.timing_bound_met,
        "exact-bound should pass (<=): time={actual_time:.1} us"
    );
}

/// Verify P5 (temporal boundedness) check produces CrownProven for the
/// Kokoro timing certificate.
#[test]
fn test_kokoro_p5_crown_proven() {
    let (dispatch_plan, _) = build_kokoro_dispatch_plan(100);
    let stages = kokoro_verified_stages(192);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, 50_000.0)
        .expect("pipeline must verify");

    let result = check_temporal_boundedness(&cert);
    assert!(result.proven, "P5 must be proven: {}", result.explanation);
    assert_eq!(
        result.level,
        VerificationLevel::CrownProven,
        "P5 must be CrownProven"
    );
}

/// Verify tight timing bound (1ms) causes failure -- proves bound is meaningful.
#[test]
fn test_kokoro_timing_tight_bound_fails() {
    let (dispatch_plan, _) = build_kokoro_dispatch_plan(100);
    let stages = kokoro_verified_stages(192);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, 1_000.0) // 1ms
        .expect("pipeline must verify");

    assert!(
        !cert.timing_bound_met,
        "1ms bound should fail for Kokoro 100-token: estimated={:.1} us",
        cert.worst_case_time_us,
    );
    assert!(!cert.overall_passed, "overall should fail with tight bound");
}

/// Report output includes realistic hardware and timing information.
#[test]
fn test_kokoro_timing_report_format() {
    let (dispatch_plan, _) = build_kokoro_dispatch_plan(50);
    let stages = kokoro_verified_stages(192);
    let hw = HardwareCostModel::m4_max_conservative();

    let cert = verify_pipeline_with_timing(&stages, &dispatch_plan, &hw, 50_000.0)
        .expect("pipeline must verify");

    let report = cert.report();
    assert!(report.contains("Timing Verification Report"));
    assert!(report.contains("PASS"));
    assert!(report.contains("2.8 TFLOPS")); // conservative model
    assert!(report.contains("200 GB/s"));
    assert!(report.contains("Total FLOPs:"));
    assert!(report.contains("MB"));
}
