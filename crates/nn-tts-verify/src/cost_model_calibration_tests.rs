// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::cost_model::{
    profile_dispatch_plan, step_flops, step_memory_bytes, step_name, HardwareCostModel,
    LayerCostProfile,
};
use nn_dsl::{DispatchStep, ScalarType, TensorNodeId};

// ============================================================================
// Helper: build known dispatch steps with real dimensions
// ============================================================================

/// FFN matmul [512, 768] x [768, 3072] — production Kokoro/Whisper dimensions.
/// Measured on M4 Max: ~784 μs via simdgroup GEMM (W2 benchmark, #1518 AC3).
fn ffn_matmul_step() -> DispatchStep {
    DispatchStep::MatMul {
        kernel_name: "ffn_matmul".to_string(),
        dtype: ScalarType::F32,
        left: TensorNodeId::new(0),
        right: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        m: 512,
        k: 768,
        n: 3072,
        batch_size: 1,
        transpose_right: false,
        broadcast_right: false,
        scale: None,
        total_elements: 512 * 3072,
    }
}

/// Small matmul [128, 128] x [128, 128].
/// Measured on M4 Max: ~10-50 μs for naive kernel at this scale.
fn small_matmul_step() -> DispatchStep {
    DispatchStep::MatMul {
        kernel_name: "small_matmul".to_string(),
        dtype: ScalarType::F32,
        left: TensorNodeId::new(0),
        right: TensorNodeId::new(1),
        output: TensorNodeId::new(2),
        m: 128,
        k: 128,
        n: 128,
        batch_size: 1,
        transpose_right: false,
        broadcast_right: false,
        scale: None,
        total_elements: 128 * 128,
    }
}

/// Linear layer [1, 4] -> [1, 8] — tiny fixture for unit tests.
fn tiny_linear_step() -> DispatchStep {
    DispatchStep::Linear {
        kernel_name: "tiny_linear".to_string(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(0),
        weight: TensorNodeId::new(1),
        bias: None,
        output: TensorNodeId::new(2),
        in_features: 4,
        out_features: 8,
        batch_size: 1,
        total_elements: 8,
    }
}

/// ReLU with 8 elements.
fn tiny_relu_step() -> DispatchStep {
    DispatchStep::Relu {
        kernel_name: "tiny_relu".to_string(),
        dtype: ScalarType::F32,
        input: TensorNodeId::new(2),
        output: TensorNodeId::new(3),
        total_elements: 8,
    }
}

// ============================================================================
// Tests: calibrate_profiles
// ============================================================================

#[test]
fn test_calibrate_exact_match() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: profiles[0].estimated_time_us / 2.0, // estimate is 2x measured
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    assert_eq!(report.steps.len(), 1);
    assert_eq!(report.unmatched_steps.len(), 0);
    assert!(report.all_conservative());
    assert!((report.steps[0].conservatism_ratio - 2.0).abs() < 1e-10);
}

#[test]
fn test_calibrate_underestimate_detected() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    // Measured takes longer than estimated — this is an underestimate (unsafe).
    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: profiles[0].estimated_time_us * 3.0,
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    assert_eq!(report.underestimate_count, 1);
    assert!(!report.all_conservative());
    assert!(report.steps[0].conservatism_ratio < 1.0);
}

#[test]
fn test_calibrate_unmatched_steps() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step(), tiny_relu_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    // Only provide measurement for linear, not relu.
    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: 10.0,
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    assert_eq!(report.steps.len(), 1);
    assert_eq!(report.unmatched_steps.len(), 1);
    assert_eq!(report.unmatched_steps[0], "tiny_relu");
}

#[test]
fn test_calibrate_empty_profiles() {
    let report = calibrate_profiles(&[], &[]);
    assert!(report.steps.is_empty());
    assert!(!report.all_conservative()); // empty is not conservative
    assert_eq!(report.max_conservatism, 0.0);
}

#[test]
fn test_calibrate_zero_measurement_skipped() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: 0.0, // zero measurement → treated as unmatched
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    assert_eq!(report.steps.len(), 0);
    assert_eq!(report.unmatched_steps.len(), 1);
}

#[test]
fn test_calibrate_nan_measurement_skipped() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: f64::NAN,
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    assert_eq!(report.steps.len(), 0);
    assert_eq!(report.unmatched_steps.len(), 1);
}

#[test]
fn test_calibrate_negative_measurement_skipped() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: -5.0,
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    assert_eq!(report.steps.len(), 0);
    assert_eq!(report.unmatched_steps.len(), 1);
}

// ============================================================================
// Tests: within_factor
// ============================================================================

#[test]
fn test_within_factor_tight() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step()];
    let profiles = profile_dispatch_plan(&plan, &model);
    let est = profiles[0].estimated_time_us;

    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: est / 1.5, // 1.5x conservatism
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    assert!(report.within_factor(2.0)); // 1.5 < 2.0
    assert!(!report.within_factor(1.2)); // 1.5 > 1.2
}

// ============================================================================
// Tests: fill_measured
// ============================================================================

#[test]
fn test_fill_measured_basic() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step(), tiny_relu_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: 42.0,
    }];

    let filled = fill_measured(&profiles, &measurements);
    assert_eq!(filled.len(), 2);
    assert_eq!(filled[0].measured_time_us, Some(42.0));
    assert_eq!(filled[1].measured_time_us, None); // relu not measured
}

#[test]
fn test_fill_measured_preserves_existing() {
    let profiles = vec![LayerCostProfile {
        layer_name: "existing".to_string(),
        flops: 100,
        memory_bytes: 200,
        estimated_time_us: 5.0,
        measured_time_us: Some(3.0), // already has measurement
    }];

    // No matching measurement → keep existing.
    let filled = fill_measured(&profiles, &[]);
    assert_eq!(filled[0].measured_time_us, Some(3.0));
}

// ============================================================================
// Tests: conservatism analysis with production-scale dimensions
// ============================================================================

#[test]
fn test_ffn_matmul_roofline_conservatism() {
    // FFN matmul [512, 768] x [768, 3072]:
    //   FLOPs = 2 * 512 * 768 * 3072 = 2,415,919,104
    //   Memory = (512*768 + 768*3072 + 512*3072) * 4 bytes
    //
    // M4 Max measured: ~784 μs (simdgroup GEMM, W2 benchmark #1518 AC3).
    let model = HardwareCostModel::m4_max();
    let plan = vec![ffn_matmul_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    // Verify FLOP count matches expected.
    let expected_flops: u64 = 2 * 512 * 768 * 3072;
    assert_eq!(profiles[0].flops, expected_flops);

    // Measured from #1518 AC3 benchmark: 0.784ms = 784 μs.
    let measurements = vec![Measurement {
        step_name: "ffn_matmul".to_string(),
        measured_time_us: 784.0,
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    assert_eq!(report.steps.len(), 1);

    let step = &report.steps[0];

    // The roofline estimate for this matmul at M4 Max:
    //   compute_time = 2,415,919,104 / (14.2 * 1e6) ≈ 170 μs
    //   memory_time = (512*768 + 768*3072 + 512*3072) * 4 / (400 * 1e3) ≈ 43 μs
    //   total = max(170, 43) + 5 ≈ 175 μs
    //
    // This is < 784 μs measured — the roofline model UNDERestimates because
    // it assumes peak utilization, while real matmul achieves <30% of peak
    // on complex shapes. This is expected and documented.
    //
    // The roofline model is a theoretical lower bound, not a conservative
    // upper bound. For a conservative bound, we need a calibrated model
    // with an empirical correction factor.
    assert!(step.estimated_time_us > 100.0); // sanity: estimate is reasonable
    assert!(step.estimated_time_us < 1000.0); // sanity: not vacuously large

    // Document the actual ratio for future calibration.
    // Expected: ratio ≈ 0.22 (roofline significantly underestimates).
    assert!(step.conservatism_ratio > 0.1);
    assert!(step.conservatism_ratio < 5.0);
}

#[test]
fn test_small_matmul_dispatch_overhead_dominated() {
    // Small [128,128]x[128,128]: FLOPs = 2 * 128^3 = 4,194,304
    // At M4 Max: compute_time ≈ 0.3 μs, dispatch_overhead = 5.0 μs.
    // Total roofline ≈ 5.3 μs — dispatch-overhead dominated.
    let model = HardwareCostModel::m4_max();
    let plan = vec![small_matmul_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    // For small ops, dispatch overhead dominates the estimate.
    assert!(profiles[0].estimated_time_us >= model.dispatch_overhead_us);
    assert!(profiles[0].estimated_time_us < 2.0 * model.dispatch_overhead_us);
}

// ============================================================================
// Tests: report generation
// ============================================================================

#[test]
fn test_calibration_report_format() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step(), tiny_relu_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    let measurements = vec![Measurement {
        step_name: "tiny_linear".to_string(),
        measured_time_us: 3.0,
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    let text = report.report();
    assert!(text.contains("Roofline Calibration Report"));
    assert!(text.contains("Steps matched: 1"));
    assert!(text.contains("tiny_linear"));
    assert!(text.contains("Unmatched steps"));
    assert!(text.contains("tiny_relu"));
}

// ============================================================================
// Tests: multi-step calibration
// ============================================================================

#[test]
fn test_multi_step_statistics() {
    let model = HardwareCostModel::m4_max();
    let plan = vec![tiny_linear_step(), tiny_relu_step()];
    let profiles = profile_dispatch_plan(&plan, &model);

    let measurements = vec![
        Measurement {
            step_name: "tiny_linear".to_string(),
            measured_time_us: profiles[0].estimated_time_us / 2.0, // 2x conservative
        },
        Measurement {
            step_name: "tiny_relu".to_string(),
            measured_time_us: profiles[1].estimated_time_us / 3.0, // 3x conservative
        },
    ];

    let report = calibrate_profiles(&profiles, &measurements);
    assert_eq!(report.steps.len(), 2);
    assert!(report.all_conservative());
    assert!((report.min_conservatism - 2.0).abs() < 1e-10);
    assert!((report.max_conservatism - 3.0).abs() < 1e-10);
    assert!((report.mean_conservatism - 2.5).abs() < 1e-10);
    assert!(report.within_factor(4.0));
    assert!(!report.within_factor(2.5));
}

// ============================================================================
// Tests: conservatism factor for timing certificates
// ============================================================================

#[test]
fn test_conservative_model_is_conservative_for_ffn_matmul() {
    // The conservative model must produce estimates >= measured (conservatism >= 1.0)
    // for the FFN matmul benchmark measured at 784 μs on M4 Max (#1518 AC3).
    let conservative = HardwareCostModel::m4_max_conservative();
    let plan = vec![ffn_matmul_step()];
    let profiles = profile_dispatch_plan(&plan, &conservative);

    let measurements = vec![Measurement {
        step_name: "ffn_matmul".to_string(),
        measured_time_us: 784.0,
    }];

    let report = calibrate_profiles(&profiles, &measurements);
    let step = &report.steps[0];

    // Conservative model MUST be conservative (ratio >= 1.0).
    assert!(
        step.is_conservative,
        "conservative model must produce estimate >= measured for FFN matmul: \
         estimated={:.1}μs, measured={:.1}μs, ratio={:.3}",
        step.estimated_time_us, step.measured_time_us, step.conservatism_ratio,
    );

    // Must not be vacuously large (< 5x measured).
    assert!(
        report.within_factor(5.0),
        "conservative model must not be vacuously wide: ratio={:.3}",
        step.conservatism_ratio,
    );
}

#[test]
fn test_conservative_model_validate_passes() {
    let conservative = HardwareCostModel::m4_max_conservative();
    assert!(conservative.validate().is_ok());
    assert!(conservative.peak_tflops_f32 > 0.0);
    assert!(conservative.peak_bandwidth_gbs > 0.0);
    assert!(conservative.dispatch_overhead_us > 0.0);
}

#[test]
fn test_conservative_strictly_slower_than_theoretical() {
    // Conservative model should always produce larger estimates than theoretical.
    let theoretical = HardwareCostModel::m4_max();
    let conservative = HardwareCostModel::m4_max_conservative();

    let steps = vec![ffn_matmul_step(), small_matmul_step()];
    for step in &steps {
        let flops = step_flops(step);
        let mem = step_memory_bytes(step);
        let t_est = theoretical.estimate_time_us(flops, mem);
        let c_est = conservative.estimate_time_us(flops, mem);
        assert!(
            c_est > t_est,
            "conservative estimate must exceed theoretical for {}: \
             conservative={:.1}μs, theoretical={:.1}μs",
            step_name(step),
            c_est,
            t_est,
        );
    }
}
