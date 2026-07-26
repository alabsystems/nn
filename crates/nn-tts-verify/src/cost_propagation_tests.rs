// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for coupled CROWN + cost propagation.
//!
//! Part of #1739 Phase 2 — AC5.

use super::*;
use crate::cost_model::LayerCostProfile;
use crate::pipeline::{PipelineCertificate, VerifiedStage};

fn dummy_stage(name: &str) -> VerifiedStage {
    VerifiedStage {
        name: name.to_string(),
        input_lower: vec![-1.0, -1.0],
        input_upper: vec![1.0, 1.0],
        output_lower: vec![-0.5, -0.5],
        output_upper: vec![0.5, 0.5],
        input_shape: vec![2],
        output_shape: vec![2],
        method: "CROWN".to_string(),
        is_sound: true,
    }
}

fn dummy_cost(name: &str, flops: u64, mem: u64, time_us: f64) -> LayerCostProfile {
    LayerCostProfile {
        layer_name: name.to_string(),
        flops,
        memory_bytes: mem,
        estimated_time_us: time_us,
        measured_time_us: None,
    }
}

// --- aggregate_layer_cost tests ---

#[test]
fn test_aggregate_layer_cost_single_step() {
    let profiles = vec![dummy_cost("matmul_0", 1000, 2000, 5.0)];
    let result = aggregate_layer_cost(&profiles, "layer_0");
    assert_eq!(result.layer_name, "layer_0");
    assert_eq!(result.flops, 1000);
    assert_eq!(result.memory_bytes, 2000);
    assert!((result.estimated_time_us - 5.0).abs() < 1e-10);
    assert!(result.measured_time_us.is_none());
}

#[test]
fn test_aggregate_layer_cost_multiple_steps() {
    let profiles = vec![
        dummy_cost("matmul_0", 1000, 2000, 5.0),
        dummy_cost("relu_0", 100, 400, 1.0),
        dummy_cost("bias_add_0", 200, 600, 0.5),
    ];
    let result = aggregate_layer_cost(&profiles, "layer_0");
    assert_eq!(result.flops, 1300);
    assert_eq!(result.memory_bytes, 3000);
    assert!((result.estimated_time_us - 6.5).abs() < 1e-10);
}

#[test]
fn test_aggregate_layer_cost_empty() {
    let profiles: Vec<LayerCostProfile> = vec![];
    let result = aggregate_layer_cost(&profiles, "layer_0");
    assert_eq!(result.flops, 0);
    assert_eq!(result.memory_bytes, 0);
    assert!((result.estimated_time_us).abs() < 1e-10);
}

// --- CoupledLayerResult tests ---

#[test]
fn test_coupled_layer_result_display() {
    let result = CoupledLayerResult {
        stage: dummy_stage("layer_0"),
        cost_profile: dummy_cost("layer_0", 5000, 10000, 2.5),
        dispatch_step_count: 3,
    };
    assert_eq!(result.dispatch_step_count, 3);
    assert_eq!(result.cost_profile.flops, 5000);
}

// --- CoupledTimingCertificate tests ---

fn make_coupled_cert(coupled: bool) -> CoupledTimingCertificate {
    let stages = vec![dummy_stage("layer_0"), dummy_stage("layer_1")];
    let cost_profiles = vec![
        dummy_cost("layer_0", 1000, 2000, 5.0),
        dummy_cost("layer_1", 2000, 3000, 7.0),
    ];

    let bounds_cert = PipelineCertificate {
        stages: stages.clone(),
        junctions: vec![],
        e2e_input_lower: vec![-1.0],
        e2e_input_upper: vec![1.0],
        e2e_output_lower: vec![-0.5],
        e2e_output_upper: vec![0.5],
        is_valid: true,
        is_sound: true,
    };

    let timing = TimingCertificate {
        bounds_cert,
        cost_profiles: cost_profiles.clone(),
        worst_case_time_us: 12.0,
        total_flops: 3000,
        total_memory_bytes: 5000,
        hardware_name: "test".to_string(),
        timing_bound_us: 100.0,
        timing_bound_met: true,
        overall_passed: true,
        peak_memory: None,
    };

    let step_count = if coupled { 3 } else { 0 };

    CoupledTimingCertificate {
        timing,
        coupled_layers: vec![
            CoupledLayerResult {
                stage: stages[0].clone(),
                cost_profile: cost_profiles[0].clone(),
                dispatch_step_count: step_count,
            },
            CoupledLayerResult {
                stage: stages[1].clone(),
                cost_profile: cost_profiles[1].clone(),
                dispatch_step_count: step_count,
            },
        ],
        total_dispatch_steps: step_count * 2,
    }
}

#[test]
fn test_all_layers_coupled_true() {
    let cert = make_coupled_cert(true);
    assert!(cert.all_layers_coupled());
    assert_eq!(cert.total_dispatch_steps, 6);
}

#[test]
fn test_all_layers_coupled_false_zero_steps() {
    let cert = make_coupled_cert(false);
    assert!(!cert.all_layers_coupled());
    assert_eq!(cert.total_dispatch_steps, 0);
}

#[test]
fn test_all_layers_coupled_false_empty() {
    let cert = CoupledTimingCertificate {
        timing: make_coupled_cert(true).timing,
        coupled_layers: vec![],
        total_dispatch_steps: 0,
    };
    assert!(!cert.all_layers_coupled());
}

#[test]
fn test_coupled_cert_display() {
    let cert = make_coupled_cert(true);
    let display = format!("{cert}");
    assert!(display.contains("CoupledTimingCertificate"));
    assert!(display.contains("2 layers"));
    assert!(display.contains("coupled=true"));
}

#[test]
fn test_coupled_cert_report() {
    let cert = make_coupled_cert(true);
    let report = cert.report();
    assert!(report.contains("Per-Layer Coupled Verification"));
    assert!(report.contains("layer_0"));
    assert!(report.contains("layer_1"));
    assert!(report.contains("Dispatch steps: 3"));
    assert!(report.contains("All layers coupled: true"));
}

#[test]
fn test_coupled_cert_report_uncoupled() {
    let cert = make_coupled_cert(false);
    let report = cert.report();
    assert!(report.contains("All layers coupled: false"));
    assert!(report.contains("Dispatch steps: 0"));
}

// ============================================================================
// Kokoro-scale timing certificate tests (AC6 of #1739)
// ============================================================================

use crate::cost_model::{
    profile_dispatch_plan, total_estimated_time_us, total_flops, total_memory_bytes,
    HardwareCostModel,
};
use nn_dsl::{DispatchStep, ScalarType, TensorNodeId};

// Kokoro dimension constants for dispatch plan construction.
const KOKORO_MAX_TOKENS: usize = 100;
const KOKORO_EMBED_DIM: usize = 256;
const KOKORO_HIDDEN_DIM: usize = 512;
const KOKORO_FFN_DIM: usize = 2048;
const KOKORO_VOCODER_DIM: usize = 1024;
/// ~256 audio samples per token at 24kHz.
const KOKORO_AUDIO_SAMPLES: usize = KOKORO_MAX_TOKENS * 256;
/// Vocoder frame count after 4x downsampling.
const KOKORO_VOCODER_FRAMES: usize = KOKORO_AUDIO_SAMPLES / 4;

/// Helper: linear dispatch step with bias.
fn linear_step(
    name: &str,
    input: usize,
    output: usize,
    weight: usize,
    bias: usize,
    in_f: usize,
    out_f: usize,
    batch: usize,
) -> DispatchStep {
    DispatchStep::Linear {
        kernel_name: name.to_string(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(input),
        weight: TensorNodeId::new(weight),
        bias: Some(TensorNodeId::new(bias)),
        output: TensorNodeId::new(output),
        in_features: in_f,
        out_features: out_f,
        batch_size: batch,
        total_elements: batch * out_f,
    }
}

/// Encoder portion: embedding + projection + FFN + duration predictor (6 steps).
fn kokoro_encoder_steps() -> Vec<DispatchStep> {
    vec![
        // 1. Embedding lookup: [100, 256]
        DispatchStep::Embedding {
            kernel_name: "text_embedding".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(1),
            weight: TensorNodeId::new(0),
            output: TensorNodeId::new(2),
            embedding_dim: KOKORO_EMBED_DIM,
            num_indices: KOKORO_MAX_TOKENS,
            total_elements: KOKORO_MAX_TOKENS * KOKORO_EMBED_DIM,
        },
        // 2. Linear projection: [100, 256] → [100, 512]
        linear_step(
            "text_projection",
            2,
            5,
            3,
            4,
            KOKORO_EMBED_DIM,
            KOKORO_HIDDEN_DIM,
            KOKORO_MAX_TOKENS,
        ),
        // 3. FFN up-projection: [100, 512] → [100, 2048]
        linear_step(
            "ffn_up",
            5,
            8,
            6,
            7,
            KOKORO_HIDDEN_DIM,
            KOKORO_FFN_DIM,
            KOKORO_MAX_TOKENS,
        ),
        // 4. GELU activation: [100, 2048]
        DispatchStep::Gelu {
            kernel_name: "ffn_gelu".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(8),
            output: TensorNodeId::new(9),
            total_elements: KOKORO_MAX_TOKENS * KOKORO_FFN_DIM,
        },
        // 5. FFN down-projection: [100, 2048] → [100, 512]
        linear_step(
            "ffn_down",
            9,
            12,
            10,
            11,
            KOKORO_FFN_DIM,
            KOKORO_HIDDEN_DIM,
            KOKORO_MAX_TOKENS,
        ),
        // 6. Duration predictor: [100, 512] → [100, 1]
        linear_step(
            "duration_pred",
            12,
            15,
            13,
            14,
            KOKORO_HIDDEN_DIM,
            1,
            KOKORO_MAX_TOKENS,
        ),
    ]
}

/// Vocoder portion: up-projection + sigmoid gate + down-projection (3 steps).
fn kokoro_vocoder_steps() -> Vec<DispatchStep> {
    vec![
        // 7. Vocoder up-projection: [6400, 512] → [6400, 1024]
        linear_step(
            "vocoder_up",
            16,
            19,
            17,
            18,
            KOKORO_HIDDEN_DIM,
            KOKORO_VOCODER_DIM,
            KOKORO_VOCODER_FRAMES,
        ),
        // 8. Vocoder activation (sigmoid for gating)
        DispatchStep::Sigmoid {
            kernel_name: "vocoder_gate".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(19),
            output: TensorNodeId::new(20),
            total_elements: KOKORO_VOCODER_FRAMES * KOKORO_VOCODER_DIM,
        },
        // 9. Vocoder down-projection: [6400, 1024] → [6400, 1]
        linear_step(
            "vocoder_out",
            20,
            23,
            21,
            22,
            KOKORO_VOCODER_DIM,
            1,
            KOKORO_VOCODER_FRAMES,
        ),
    ]
}

/// Build a representative Kokoro TTS dispatch plan (9 steps).
///
/// Kokoro architecture (simplified):
///   1-6. Encoder: Embedding → projection → FFN → duration predictor
///   7-9. Vocoder: up-projection → sigmoid gate → down-projection
fn kokoro_dispatch_plan() -> Vec<DispatchStep> {
    let mut plan = kokoro_encoder_steps();
    plan.extend(kokoro_vocoder_steps());
    plan
}

#[test]
fn test_kokoro_timing_certificate_conservative() {
    // AC6 of #1739: End-to-end timing certificate for Kokoro TTS.
    //
    // Target claim: "For any English text up to 100 tokens, Kokoro synthesis
    // completes in < 100 ms on M4 Max."
    let conservative = HardwareCostModel::m4_max_conservative();
    let plan = kokoro_dispatch_plan();
    let profiles = profile_dispatch_plan(&plan, &conservative);

    let worst_case = total_estimated_time_us(&profiles);
    let flops = total_flops(&profiles);
    let mem = total_memory_bytes(&profiles);

    // Build the timing certificate.
    let stages = vec![dummy_stage("kokoro_encoder"), dummy_stage("kokoro_vocoder")];
    let bounds_cert = PipelineCertificate {
        stages,
        junctions: vec![],
        e2e_input_lower: vec![-1.0],
        e2e_input_upper: vec![1.0],
        e2e_output_lower: vec![-1.0],
        e2e_output_upper: vec![1.0],
        is_valid: true,
        is_sound: true,
    };

    let timing_bound_us = 100_000.0; // 100 ms target
    let timing_met = worst_case <= timing_bound_us;

    let cert = TimingCertificate {
        bounds_cert,
        cost_profiles: profiles.clone(),
        worst_case_time_us: worst_case,
        total_flops: flops,
        total_memory_bytes: mem,
        hardware_name: format!(
            "M4 Max (conservative): peak={:.1} TFLOPS, bw={:.0} GB/s",
            conservative.peak_tflops_f32, conservative.peak_bandwidth_gbs,
        ),
        timing_bound_us,
        timing_bound_met: timing_met,
        overall_passed: timing_met,
        peak_memory: None,
    };

    // The simplified Kokoro plan should complete well within 100ms.
    // Conservative model with 9 dispatch steps:
    //   Main cost: FFN matmuls (100 × 512→2048→512) + vocoder (6400 × 512→1024→1)
    assert!(
        cert.timing_bound_met,
        "Kokoro 100-token worst-case ({:.1} μs = {:.1} ms) exceeds 100ms target",
        cert.worst_case_time_us,
        cert.worst_case_time_us / 1000.0,
    );
    assert!(cert.overall_passed);

    // Sanity: FLOPs should be in the billions (realistic for a small TTS model).
    assert!(flops > 1_000_000_000, "FLOPs unexpectedly low: {flops}");

    // Report should include all 9 dispatch steps.
    assert_eq!(profiles.len(), 9);
}

#[test]
fn test_kokoro_theoretical_vs_conservative_gap() {
    // Document the gap between theoretical and conservative estimates
    // for a realistic Kokoro dispatch plan.
    let theoretical = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();
    let plan = kokoro_dispatch_plan();

    let t_profiles = profile_dispatch_plan(&plan, &theoretical);
    let c_profiles = profile_dispatch_plan(&plan, &conservative);

    let t_total = total_estimated_time_us(&t_profiles);
    let c_total = total_estimated_time_us(&c_profiles);

    // Conservative should be >= 3x the theoretical (given 5x compute + 2x BW factors).
    let ratio = c_total / t_total;
    assert!(
        ratio >= 3.0,
        "conservative/theoretical ratio too small: {ratio:.2}x"
    );
    assert!(
        ratio <= 10.0,
        "conservative/theoretical ratio too large: {ratio:.2}x"
    );
}

#[test]
fn test_kokoro_certificate_report_contents() {
    let conservative = HardwareCostModel::m4_max_conservative();
    let plan = kokoro_dispatch_plan();
    let profiles = profile_dispatch_plan(&plan, &conservative);
    let worst_case = total_estimated_time_us(&profiles);

    let stages = vec![dummy_stage("kokoro")];
    let bounds_cert = PipelineCertificate {
        stages,
        junctions: vec![],
        e2e_input_lower: vec![-1.0],
        e2e_input_upper: vec![1.0],
        e2e_output_lower: vec![-1.0],
        e2e_output_upper: vec![1.0],
        is_valid: true,
        is_sound: true,
    };

    let cert = TimingCertificate {
        bounds_cert,
        cost_profiles: profiles,
        worst_case_time_us: worst_case,
        total_flops: total_flops(&profile_dispatch_plan(&plan, &conservative)),
        total_memory_bytes: total_memory_bytes(&profile_dispatch_plan(&plan, &conservative)),
        hardware_name: "M4 Max (conservative)".to_string(),
        timing_bound_us: 100_000.0,
        timing_bound_met: worst_case <= 100_000.0,
        overall_passed: worst_case <= 100_000.0,
        peak_memory: None,
    };

    let report = cert.report();
    assert!(report.contains("M4 Max (conservative)"));
    assert!(report.contains("100000")); // timing bound
}
