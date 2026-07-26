// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Silero VAD dispatch plan builder.
//!
//! Part of #1739 AC4.

use super::*;
use crate::cost_model::{
    estimate_peak_memory, profile_dispatch_plan, total_estimated_time_us, total_flops,
    total_memory_bytes, HardwareCostModel,
};

// --- Step count ---

#[test]
fn test_dispatch_plan_step_count() {
    let plan = build_silero_vad_dispatch_plan_default();
    assert_eq!(
        plan.len(),
        TOTAL_EXPECTED_STEPS,
        "expected {} steps, got {}",
        TOTAL_EXPECTED_STEPS,
        plan.len()
    );
}

#[test]
fn test_step_count_constant_consistency() {
    // Verify the constant decomposition adds up correctly.
    let expected = ENCODER_STEP_PAIRS * 2 + 1 + LSTM_DECOMPOSED_STEPS + OUTPUT_STAGE_STEPS;
    assert_eq!(expected, 24);
    assert_eq!(TOTAL_EXPECTED_STEPS, 24);
}

// --- Encoder block structure ---

#[test]
fn test_encoder_blocks_are_conv1d_relu_pairs() {
    let plan = build_silero_vad_dispatch_plan_default();
    // First 8 steps should alternate Conv1d and Relu.
    for i in 0..4 {
        let conv_step = &plan[i * 2];
        let relu_step = &plan[i * 2 + 1];
        assert!(
            matches!(conv_step, DispatchStep::Conv1d(_)),
            "step {} should be Conv1d, got {:?}",
            i * 2,
            conv_step
        );
        assert!(
            matches!(relu_step, DispatchStep::Relu { .. }),
            "step {} should be Relu",
            i * 2 + 1
        );
    }
}

#[test]
fn test_encoder_conv1d_channel_progression() {
    let plan = build_silero_vad_dispatch_plan_default();
    // Block 0: 129 → 128
    if let DispatchStep::Conv1d(p) = &plan[0] {
        assert_eq!(p.in_channels, 129);
        assert_eq!(p.out_channels, 128);
    } else {
        panic!("step 0 not Conv1d");
    }
    // Block 1: 128 → 64
    if let DispatchStep::Conv1d(p) = &plan[2] {
        assert_eq!(p.in_channels, 128);
        assert_eq!(p.out_channels, 64);
    } else {
        panic!("step 2 not Conv1d");
    }
    // Block 2: 64 → 64 (previously untested — P1-269 audit)
    if let DispatchStep::Conv1d(p) = &plan[4] {
        assert_eq!(p.in_channels, 64);
        assert_eq!(p.out_channels, 64);
    } else {
        panic!("step 4 not Conv1d");
    }
    // Block 3: 64 → 128
    if let DispatchStep::Conv1d(p) = &plan[6] {
        assert_eq!(p.in_channels, 64);
        assert_eq!(p.out_channels, 128);
    } else {
        panic!("step 6 not Conv1d");
    }
}

// --- Temporal pool ---

#[test]
fn test_temporal_pool_after_encoder() {
    let plan = build_silero_vad_dispatch_plan_default();
    let pool_idx = ENCODER_STEP_PAIRS * 2; // step 8
    assert!(
        matches!(&plan[pool_idx], DispatchStep::Reduce { .. }),
        "step {pool_idx} should be Reduce (temporal pool)"
    );
}

// --- LSTM decomposed structure ---

#[test]
fn test_lstm_linear_gates() {
    let plan = build_silero_vad_dispatch_plan_default();
    let lstm_start = ENCODER_STEP_PAIRS * 2 + 1; // step 9
                                                 // First 2 LSTM steps are Linear (ih and hh)
    assert!(
        matches!(&plan[lstm_start], DispatchStep::Linear { .. }),
        "LSTM step 0 should be Linear (ih)"
    );
    assert!(
        matches!(&plan[lstm_start + 1], DispatchStep::Linear { .. }),
        "LSTM step 1 should be Linear (hh)"
    );
    // ih: 128 → 512 (4*128)
    if let DispatchStep::Linear {
        in_features,
        out_features,
        ..
    } = &plan[lstm_start]
    {
        assert_eq!(*in_features, 128);
        assert_eq!(*out_features, 512);
    }
}

#[test]
fn test_lstm_has_12_decomposed_steps() {
    let plan = build_silero_vad_dispatch_plan_default();
    let lstm_start = ENCODER_STEP_PAIRS * 2 + 1;
    let lstm_end = lstm_start + LSTM_DECOMPOSED_STEPS;
    assert_eq!(lstm_end - lstm_start, 12);
    // Verify the last LSTM step is BinaryMul (h_new = o * tanh(c))
    assert!(
        matches!(&plan[lstm_end - 1], DispatchStep::BinaryMul { .. }),
        "last LSTM step should be BinaryMul (h_new)"
    );
}

// --- Output stage ---

#[test]
fn test_output_stage_structure() {
    let plan = build_silero_vad_dispatch_plan_default();
    let out_start = TOTAL_EXPECTED_STEPS - OUTPUT_STAGE_STEPS;
    assert!(matches!(&plan[out_start], DispatchStep::Relu { .. }));
    assert!(matches!(&plan[out_start + 1], DispatchStep::Linear { .. }));
    assert!(matches!(&plan[out_start + 2], DispatchStep::Sigmoid { .. }));
    // Final sigmoid produces 1 element (VAD probability)
    if let DispatchStep::Sigmoid { total_elements, .. } = &plan[out_start + 2] {
        assert_eq!(*total_elements, 1);
    }
}

// --- Profiling integration ---

#[test]
fn test_profile_silero_vad_plan() {
    let plan = build_silero_vad_dispatch_plan_default();
    let model = HardwareCostModel::m4_max();
    let profiles = profile_dispatch_plan(&plan, &model);
    assert_eq!(profiles.len(), plan.len());
    let total_time = total_estimated_time_us(&profiles);
    let total_f = total_flops(&profiles);
    let total_mem = total_memory_bytes(&profiles);
    // Silero VAD is a small model — total time should be < 1ms on M4 Max
    assert!(
        total_time < 1000.0,
        "Silero VAD should be < 1ms on M4 Max, got {total_time:.1} μs"
    );
    // Silero VAD has 4 conv blocks + LSTM + output linear — FLOPs must be
    // at least 100K (even the smallest conv1d 64→64×9×k3 > 1K FLOPs).
    // The prior `> 0` assertion would pass for any non-zero value (P1-269 audit).
    assert!(
        total_f > 100_000,
        "Silero VAD total FLOPs should be > 100K, got {total_f}"
    );
    // 24 dispatch steps each producing at least some output — memory must exceed 1 KB.
    assert!(
        total_mem > 1024,
        "Silero VAD total memory should be > 1 KB, got {total_mem}"
    );
}

#[test]
fn test_conservative_model_upper_bounds_theoretical() {
    let plan = build_silero_vad_dispatch_plan_default();
    let theoretical = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();
    let t_profiles = profile_dispatch_plan(&plan, &theoretical);
    let c_profiles = profile_dispatch_plan(&plan, &conservative);
    let t_time = total_estimated_time_us(&t_profiles);
    let c_time = total_estimated_time_us(&c_profiles);
    assert!(
        c_time > t_time,
        "conservative ({c_time:.1}) should > theoretical ({t_time:.1})"
    );
}

// --- Peak memory ---

#[test]
fn test_peak_memory_silero_vad() {
    let plan = build_silero_vad_dispatch_plan_default();
    let mem = estimate_peak_memory(&plan);
    // Silero VAD is tiny — peak memory should be well under 1 MB
    assert!(
        mem.peak_total_mb() < 1.0,
        "peak memory {:.3} MB should be < 1 MB",
        mem.peak_total_mb()
    );
    assert_eq!(mem.per_step_output_bytes.len(), plan.len());
}

// --- Parameterized temporal dim ---

#[test]
fn test_different_stft_frame_counts() {
    let plan_33 = build_silero_vad_dispatch_plan(33);
    let plan_17 = build_silero_vad_dispatch_plan(17);
    // Both should have the same number of steps (topology unchanged)
    assert_eq!(plan_33.len(), plan_17.len());
    // But different total elements in encoder Conv1d steps
    let conv0_33 = match &plan_33[0] {
        DispatchStep::Conv1d(p) => p.total_elements,
        _ => panic!("expected Conv1d"),
    };
    let conv0_17 = match &plan_17[0] {
        DispatchStep::Conv1d(p) => p.total_elements,
        _ => panic!("expected Conv1d"),
    };
    assert!(
        conv0_33 > conv0_17,
        "33 frames should produce more elements"
    );
}

#[test]
fn test_encoder_temporal_downsampling() {
    // Blocks 1 and 2 have stride=2, halving temporal dim each time.
    // Input: T=33 → block0(s=1): 33 → block1(s=2): 17 → block2(s=2): 9 → block3(s=1): 9
    let plan = build_silero_vad_dispatch_plan(33);
    let expected_t = [33, 17, 9, 9];
    for (i, &expected) in expected_t.iter().enumerate() {
        if let DispatchStep::Conv1d(p) = &plan[i * 2] {
            let actual_t = p.total_elements / p.out_channels;
            assert_eq!(
                actual_t, expected,
                "block {i} output temporal dim: expected {expected}, got {actual_t}"
            );
        }
    }
}
