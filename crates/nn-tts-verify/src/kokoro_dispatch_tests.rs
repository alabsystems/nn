// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Kokoro TTS vocoder dispatch plan builder.
//!
//! Part of #1739 AC3.

use super::*;
use crate::cost_model::{
    estimate_peak_memory, profile_dispatch_plan, total_estimated_time_us, total_flops,
    total_memory_bytes, HardwareCostModel,
};

// --- Step count ---

#[test]
fn test_dispatch_plan_step_count() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
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
    assert_eq!(STEPS_PER_DILATION, 7, "7 steps per dilation layer");
    assert_eq!(DILATIONS_PER_RESBLOCK, 3, "3 dilations per ResBlock");
    assert_eq!(STEPS_PER_RESBLOCK, 21, "21 steps per ResBlock");
    assert_eq!(RESBLOCKS_PER_STAGE, 3, "3 ResBlocks per stage");
    assert_eq!(NOISE_STEPS_PER_STAGE, 23, "23 noise steps per stage");
    assert_eq!(STEPS_PER_STAGE, 88, "88 steps per upsample stage");
    assert_eq!(CONV_PRE_STEPS, 1, "1 conv_pre step");
    assert_eq!(OUTPUT_STAGE_STEPS, 4, "4 output stage steps");
    assert_eq!(TOTAL_EXPECTED_STEPS, 181, "1 + 2×88 + 4 = 181 total steps");
}

// --- conv_pre structure ---

#[test]
fn test_conv_pre_is_first_step() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    if let DispatchStep::Conv1d(p) = &plan[0] {
        assert_eq!(p.in_channels, 512, "conv_pre in_channels");
        assert_eq!(p.out_channels, 512, "conv_pre out_channels");
        assert_eq!(p.kernel_size, 7, "conv_pre kernel_size");
        assert_eq!(p.padding, 3, "conv_pre padding");
        assert_eq!(p.kernel_name, "conv_pre");
    } else {
        panic!("step 0 should be Conv1d (conv_pre)");
    }
}

// --- Upsample stage structure ---

#[test]
fn test_stage0_leaky_relu_and_upsample() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    // Stage 0 starts after conv_pre (step 1)
    let stage0_start = CONV_PRE_STEPS; // 1

    // LeakyReLU (modeled as Sigmoid)
    assert!(
        matches!(&plan[stage0_start], DispatchStep::Sigmoid { .. }),
        "stage 0 should start with LeakyReLU (Sigmoid proxy)"
    );

    // ConvTranspose1d upsample
    if let DispatchStep::ConvTranspose1d(p) = &plan[stage0_start + 1] {
        assert_eq!(p.in_channels, 512, "upsample_0 in_channels");
        assert_eq!(p.out_channels, 256, "upsample_0 out_channels");
        assert_eq!(p.kernel_size, 20, "upsample_0 kernel_size");
        assert_eq!(p.stride, 10, "upsample_0 stride");
    } else {
        panic!("step {} should be ConvTranspose1d", stage0_start + 1);
    }
}

#[test]
fn test_stage1_upsample_channels() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    // Stage 1 starts after stage 0: conv_pre(1) + stage_steps(88)
    let stage1_start = CONV_PRE_STEPS + STEPS_PER_STAGE; // 89

    // ConvTranspose1d upsample (after LeakyReLU)
    if let DispatchStep::ConvTranspose1d(p) = &plan[stage1_start + 1] {
        assert_eq!(p.in_channels, 256, "upsample_1 in_channels");
        assert_eq!(p.out_channels, 128, "upsample_1 out_channels");
        assert_eq!(p.kernel_size, 12, "upsample_1 kernel_size");
        assert_eq!(p.stride, 6, "upsample_1 stride");
    } else {
        panic!("step {} should be ConvTranspose1d", stage1_start + 1);
    }
}

// --- Output stage ---

#[test]
fn test_output_stage_structure() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    let out_start = TOTAL_EXPECTED_STEPS - OUTPUT_STAGE_STEPS; // 177

    // LeakyReLU (Sigmoid proxy)
    assert!(
        matches!(&plan[out_start], DispatchStep::Sigmoid { .. }),
        "output stage starts with LeakyReLU"
    );

    // conv_post: Conv1d(128 → 22, k=7, pad=3)
    if let DispatchStep::Conv1d(p) = &plan[out_start + 1] {
        assert_eq!(p.in_channels, 128, "conv_post in_channels");
        assert_eq!(p.out_channels, 22, "conv_post out_channels = 2*n_bins");
        assert_eq!(p.kernel_size, 7, "conv_post kernel_size");
    } else {
        panic!("step {} should be Conv1d (conv_post)", out_start + 1);
    }

    // exp(magnitude) and sin(phase) — both Tanh proxy
    assert!(
        matches!(&plan[out_start + 2], DispatchStep::Tanh { .. }),
        "exp_magnitude should be Tanh proxy"
    );
    assert!(
        matches!(&plan[out_start + 3], DispatchStep::Tanh { .. }),
        "sin_phase should be Tanh proxy"
    );
}

// --- Temporal dimension ---

#[test]
fn test_temporal_dimension_100_tokens() {
    let (_, t_final) = build_kokoro_dispatch_plan(100);
    // 100 × 10 × 6 = 6000
    assert_eq!(t_final, 6000, "100 tokens → 6000 audio frames");
}

#[test]
fn test_temporal_dimension_10_tokens() {
    let (_, t_final) = build_kokoro_dispatch_plan(10);
    // 10 × 10 × 6 = 600
    assert_eq!(t_final, 600, "10 tokens → 600 audio frames");
}

#[test]
fn test_temporal_dimension_1_token() {
    let (_, t_final) = build_kokoro_dispatch_plan(1);
    // 1 × 10 × 6 = 60
    assert_eq!(t_final, 60, "1 token → 60 audio frames");
}

#[test]
fn test_default_is_100_tokens() {
    let (plan_default, t_default) = build_kokoro_dispatch_plan_default();
    let (plan_100, t_100) = build_kokoro_dispatch_plan(100);
    assert_eq!(t_default, t_100);
    assert_eq!(plan_default.len(), plan_100.len());
}

// --- Parameterized topology invariance ---

#[test]
fn test_different_seq_lens_same_step_count() {
    let (plan_10, _) = build_kokoro_dispatch_plan(10);
    let (plan_200, _) = build_kokoro_dispatch_plan(200);
    assert_eq!(
        plan_10.len(),
        plan_200.len(),
        "topology unchanged by seq_len"
    );
    assert_eq!(plan_10.len(), TOTAL_EXPECTED_STEPS);
}

#[test]
fn test_larger_seq_len_produces_larger_elements() {
    let (plan_10, _) = build_kokoro_dispatch_plan(10);
    let (plan_200, _) = build_kokoro_dispatch_plan(200);
    // conv_pre total_elements should scale with seq_len
    let elem_10 = match &plan_10[0] {
        DispatchStep::Conv1d(p) => p.total_elements,
        _ => panic!("expected Conv1d"),
    };
    let elem_200 = match &plan_200[0] {
        DispatchStep::Conv1d(p) => p.total_elements,
        _ => panic!("expected Conv1d"),
    };
    assert!(
        elem_200 > elem_10,
        "200 tokens should produce more elements than 10"
    );
    // Should scale linearly: elem_200 / elem_10 ≈ 20
    let ratio = elem_200 as f64 / elem_10 as f64;
    assert!(
        (ratio - 20.0).abs() < 0.01,
        "element scaling should be ~20x, got {ratio:.2}"
    );
}

// --- Step type counts ---

#[test]
fn test_step_type_distribution() {
    let (plan, _) = build_kokoro_dispatch_plan_default();

    let n_conv1d = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Conv1d(_)))
        .count();
    let n_conv_t = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::ConvTranspose1d(_)))
        .count();
    let n_linear = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Linear { .. }))
        .count();
    let n_sigmoid = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Sigmoid { .. }))
        .count();
    let n_tanh = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::Tanh { .. }))
        .count();
    let n_add = plan
        .iter()
        .filter(|s| matches!(s, DispatchStep::BinaryAdd { .. }))
        .count();

    // Per stage: 2 upsample stages
    // Conv1d: conv_pre(1) + per stage [noise_conv(1) + 4 ResBlocks × 3 dil × 2 conv = 24 + 1]
    //   = 1 + 2 × (1 + 4×3×2) = 1 + 2 × 25 = 51  ... wait let me recalculate.
    // Per dilation: 2 Conv1d (dilated + d=1)
    // Per ResBlock: 3 dilations × 2 = 6 Conv1d
    // Per stage: noise_conv(1) + noise_resblock(6) + 3 main ResBlocks(6 each = 18) = 25
    // conv_pre(1) + conv_post(1) + 2 stages × 25 = 52
    assert_eq!(n_conv1d, 52, "Conv1d count");

    // 2 ConvTranspose1d (one per upsample stage)
    assert_eq!(n_conv_t, 2, "ConvTranspose1d count");

    // Per dilation: 2 AdaIN Linears
    // Per ResBlock: 3 dil × 2 = 6
    // Per stage: noise_resblock(6) + 3 main(6 each = 18) = 24
    // Total: 2 × 24 = 48
    assert_eq!(n_linear, 48, "Linear (AdaIN) count");

    // Per dilation: 2 Snake (Sigmoid proxy)
    // Per ResBlock: 3 dil × 2 = 6
    // Per stage: noise_resblock(6) + 3 main(6 each = 18) = 24
    //   + stage LeakyReLU(1) = 25
    // Total: 2 × 25 + output LeakyReLU(1) = 51
    assert_eq!(n_sigmoid, 51, "Sigmoid (Snake/LeakyReLU proxy) count");

    // 2 Tanh (exp_magnitude + sin_phase)
    assert_eq!(n_tanh, 2, "Tanh (exp/sin proxy) count");

    // Per dilation: 1 residual add
    // Per ResBlock: 3 dilations = 3
    // Per stage: noise_add(1) + noise_resblock(3) + 3 main(3 each = 9) = 13
    // Total: 2 × 13 = 26
    assert_eq!(n_add, 26, "BinaryAdd (residual) count");

    // Total should match
    assert_eq!(
        n_conv1d + n_conv_t + n_linear + n_sigmoid + n_tanh + n_add,
        TOTAL_EXPECTED_STEPS,
        "all step types should sum to total"
    );
}

// --- Profiling integration ---

#[test]
fn test_profile_kokoro_plan() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    let model = HardwareCostModel::m4_max();
    let profiles = profile_dispatch_plan(&plan, &model);
    assert_eq!(profiles.len(), plan.len());
    let total_time = total_estimated_time_us(&profiles);
    let total_f = total_flops(&profiles);
    let total_mem = total_memory_bytes(&profiles);
    // Kokoro 82M vocoder should run in < 50ms on M4 Max for 100 tokens
    assert!(
        total_time < 50_000.0,
        "Kokoro should be < 50ms on M4 Max, got {total_time:.1} μs"
    );
    assert!(total_f > 0, "total FLOPs should be positive");
    assert!(total_mem > 0, "total memory should be positive");
}

#[test]
fn test_conservative_model_upper_bounds_theoretical() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
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
fn test_peak_memory_kokoro() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    let mem = estimate_peak_memory(&plan);
    // Kokoro vocoder at 100 tokens — peak memory should be under 100 MB
    assert!(
        mem.peak_total_mb() < 100.0,
        "peak memory {:.3} MB should be < 100 MB",
        mem.peak_total_mb()
    );
    assert_eq!(mem.per_step_output_bytes.len(), plan.len());
}

// --- Noise injection structure ---

#[test]
fn test_noise_conv_is_strided() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    // Noise conv for stage 0 is at: conv_pre(1) + LeakyReLU(1) + ConvTranspose1d(1) = step 3
    let noise_conv_idx = CONV_PRE_STEPS + 2; // step 3
    if let DispatchStep::Conv1d(p) = &plan[noise_conv_idx] {
        assert_eq!(p.in_channels, 22, "noise conv in_channels = 2*n_bins");
        assert_eq!(
            p.out_channels, 256,
            "noise conv out_channels = stage channels"
        );
        assert_eq!(p.kernel_size, 1, "noise conv k=1");
        assert!(
            p.stride > 1,
            "noise conv should be strided (stride={})",
            p.stride
        );
        assert!(
            p.kernel_name.contains("noise"),
            "kernel name should contain 'noise'"
        );
    } else {
        panic!("step {noise_conv_idx} should be Conv1d (noise_conv_0)");
    }
}

// --- ResBlock Conv1d field validation ---

#[test]
fn test_resblock_conv1d_params_validated() {
    let (plan, _) = build_kokoro_dispatch_plan_default();
    // Stage 0, noise ResBlock, first dilation (dil_idx=0, dilation=1):
    // Starts at noise_conv(1) + dilation_0 offset:
    //   noise_conv is at step 3 (CONV_PRE_STEPS + 2)
    //   dilation 0: AdaIN1(4) + Snake(5) + Conv1d(6)
    let first_rb_conv_idx = CONV_PRE_STEPS + 2 + 1 + 2; // step 6: first ResBlock Conv1d
                                                        // Stage 0: channels = 512/2 = 256, kernel_size = RESBLOCK_KERNELS[0] = 3, dilation = 1
    if let DispatchStep::Conv1d(p) = &plan[first_rb_conv_idx] {
        assert_eq!(
            p.in_channels, 256,
            "resblock conv1 in_channels = stage channels"
        );
        assert_eq!(
            p.out_channels, 256,
            "resblock conv1 out_channels = stage channels"
        );
        assert_eq!(
            p.kernel_size, 3,
            "resblock conv1 kernel_size = RESBLOCK_KERNELS[0]"
        );
        assert_eq!(p.dilation, 1, "resblock dil_0 dilation = 1");
        assert!(
            p.kernel_name.contains("rb_0_0_d0_conv1"),
            "kernel name should be rb_0_0_d0_conv1, got {}",
            p.kernel_name
        );
    } else {
        panic!(
            "step {} should be Conv1d (resblock dilation conv), got {:?}",
            first_rb_conv_idx,
            std::mem::discriminant(&plan[first_rb_conv_idx])
        );
    }

    // Also validate the second Conv1d in the same dilation (d=1 conv):
    // AdaIN2(7) + Snake2(8) + Conv1d(9)
    let second_rb_conv_idx = first_rb_conv_idx + 3; // step 9
    if let DispatchStep::Conv1d(p) = &plan[second_rb_conv_idx] {
        assert_eq!(p.in_channels, 256, "resblock conv2 in_channels");
        assert_eq!(p.out_channels, 256, "resblock conv2 out_channels");
        assert_eq!(p.kernel_size, 3, "resblock conv2 kernel_size");
        assert_eq!(p.dilation, 1, "resblock conv2 dilation = 1 (non-dilated)");
        assert!(
            p.kernel_name.contains("rb_0_0_d0_conv2"),
            "kernel name should be rb_0_0_d0_conv2, got {}",
            p.kernel_name
        );
    } else {
        panic!(
            "step {} should be Conv1d (resblock d=1 conv), got {:?}",
            second_rb_conv_idx,
            std::mem::discriminant(&plan[second_rb_conv_idx])
        );
    }
}
