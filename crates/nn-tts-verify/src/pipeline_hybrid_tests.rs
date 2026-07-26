// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::pipeline::VerifiedStage;
use nn_dsl::{DispatchStep, ScalarType, TensorNodeId};

fn two_stage_pipeline() -> Vec<VerifiedStage> {
    vec![
        VerifiedStage {
            name: "encoder".to_string(),
            input_lower: vec![-1.0; 4],
            input_upper: vec![1.0; 4],
            output_lower: vec![-0.5; 8],
            output_upper: vec![0.5; 8],
            input_shape: vec![1, 4],
            output_shape: vec![1, 8],
            method: "CROWN".to_string(),
            is_sound: true,
        },
        VerifiedStage {
            name: "decoder".to_string(),
            input_lower: vec![-1.0; 8],
            input_upper: vec![1.0; 8],
            output_lower: vec![-0.3; 2],
            output_upper: vec![0.3; 2],
            input_shape: vec![1, 8],
            output_shape: vec![1, 2],
            method: "CROWN".to_string(),
            is_sound: true,
        },
    ]
}

fn small_dispatch_plan() -> Vec<DispatchStep> {
    vec![
        DispatchStep::Linear {
            kernel_name: "encoder_linear".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: None,
            output: TensorNodeId::new(2),
            in_features: 4,
            out_features: 8,
            batch_size: 1,
            total_elements: 8,
        },
        DispatchStep::Relu {
            kernel_name: "encoder_relu".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(2),
            output: TensorNodeId::new(3),
            total_elements: 8,
        },
        DispatchStep::Linear {
            kernel_name: "decoder_linear".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(3),
            weight: TensorNodeId::new(4),
            bias: None,
            output: TensorNodeId::new(5),
            in_features: 8,
            out_features: 2,
            batch_size: 1,
            total_elements: 2,
        },
    ]
}

#[test]
fn test_timing_cert_bounds_and_timing_pass() {
    let stages = two_stage_pipeline();
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    // 1 second bound — easily met for a tiny plan.
    let cert = verify_pipeline_with_timing(&stages, &plan, &model, 1_000_000.0).unwrap();
    assert!(cert.bounds_cert.is_valid);
    assert!(cert.timing_bound_met);
    assert!(cert.overall_passed);
    assert!(cert.worst_case_time_us > 0.0);
    assert!(cert.total_flops > 0);
    assert!(cert.total_memory_bytes > 0);
}

#[test]
fn test_timing_cert_timing_exceeds_bound() {
    let stages = two_stage_pipeline();
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    // 0.001 μs bound — impossible even for tiny plan (dispatch overhead alone is 5μs).
    let cert = verify_pipeline_with_timing(&stages, &plan, &model, 0.001).unwrap();
    assert!(cert.bounds_cert.is_valid);
    assert!(!cert.timing_bound_met);
    assert!(!cert.overall_passed);
}

#[test]
fn test_timing_cert_bounds_fail() {
    let mut stages = two_stage_pipeline();
    // Make bounds incompatible: encoder output exceeds decoder input.
    stages[0].output_upper = vec![2.0; 8]; // output > input upper of 1.0
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    let cert = verify_pipeline_with_timing(&stages, &plan, &model, 1_000_000.0).unwrap();
    assert!(!cert.bounds_cert.is_valid);
    assert!(cert.timing_bound_met);
    assert!(!cert.overall_passed); // bounds fail → overall fails
}

#[test]
fn test_timing_cert_invalid_bound_nan() {
    let stages = two_stage_pipeline();
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    let result = verify_pipeline_with_timing(&stages, &plan, &model, f64::NAN);
    assert!(result.is_err());
}

#[test]
fn test_timing_cert_invalid_bound_zero() {
    let stages = two_stage_pipeline();
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    let result = verify_pipeline_with_timing(&stages, &plan, &model, 0.0);
    assert!(result.is_err());
}

#[test]
fn test_timing_cert_invalid_bound_negative() {
    let stages = two_stage_pipeline();
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    let result = verify_pipeline_with_timing(&stages, &plan, &model, -100.0);
    assert!(result.is_err());
}

#[test]
fn test_timing_cert_report_contains_key_info() {
    let stages = two_stage_pipeline();
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    let cert = verify_pipeline_with_timing(&stages, &plan, &model, 1_000_000.0).unwrap();
    let report = cert.report();
    assert!(report.contains("Timing Verification Report"));
    assert!(report.contains("Bounds: PASS"));
    assert!(report.contains("Timing: PASS"));
    assert!(report.contains("TFLOPS"));
    assert!(report.contains("PASSED"));
}

#[test]
fn test_timing_cert_display() {
    let stages = two_stage_pipeline();
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    let cert = verify_pipeline_with_timing(&stages, &plan, &model, 1_000_000.0).unwrap();
    let s = format!("{cert}");
    assert!(s.contains("TimingCertificate"));
    assert!(s.contains("bounds=pass"));
    assert!(s.contains("timing=pass"));
    assert!(s.contains("overall=PASS"));
}

#[test]
fn test_timing_cert_empty_plan() {
    let stages = two_stage_pipeline();
    let plan: Vec<DispatchStep> = vec![];
    let model = HardwareCostModel::m4_max();
    // Empty dispatch plan → 0 time → passes any positive bound.
    let cert = verify_pipeline_with_timing(&stages, &plan, &model, 1_000_000.0).unwrap();
    assert!(cert.timing_bound_met);
    assert_eq!(cert.total_flops, 0);
    assert_eq!(cert.total_memory_bytes, 0);
    assert_eq!(cert.worst_case_time_us, 0.0);
}

#[test]
fn test_timing_cert_insufficient_stages() {
    let stages = vec![two_stage_pipeline().remove(0)]; // only 1 stage
    let plan = small_dispatch_plan();
    let model = HardwareCostModel::m4_max();
    let result = verify_pipeline_with_timing(&stages, &plan, &model, 1_000_000.0);
    assert!(result.is_err());
}

// --- Existing HybridCertificate (formal+statistical) tests ---

#[test]
fn test_strong_evidence_all_criteria_met() {
    let cert = HybridCertificate {
        formal_dim: 64,
        formal_property: "output_bounded".to_string(),
        formal_is_sound: true,
        statistical_dim: 512,
        n_samples: 1000,
        p_value: 0.001,
        effect_size: 1.2,
        property_holds: true,
    };
    assert!(cert.is_strong_evidence());
}

#[test]
fn test_weak_evidence_ibp_fallback() {
    let cert = HybridCertificate {
        formal_dim: 64,
        formal_property: "output_bounded".to_string(),
        formal_is_sound: false, // IBP fallback
        statistical_dim: 512,
        n_samples: 1000,
        p_value: 0.001,
        effect_size: 1.2,
        property_holds: true,
    };
    assert!(!cert.is_strong_evidence());
}

#[test]
fn test_weak_evidence_high_p_value() {
    let cert = HybridCertificate {
        formal_dim: 64,
        formal_property: "output_bounded".to_string(),
        formal_is_sound: true,
        statistical_dim: 512,
        n_samples: 50,
        p_value: 0.15, // not significant
        effect_size: 1.2,
        property_holds: true,
    };
    assert!(!cert.is_strong_evidence());
}

#[test]
fn test_weak_evidence_small_effect() {
    let cert = HybridCertificate {
        formal_dim: 64,
        formal_property: "output_bounded".to_string(),
        formal_is_sound: true,
        statistical_dim: 512,
        n_samples: 1000,
        p_value: 0.001,
        effect_size: 0.3, // small effect
        property_holds: true,
    };
    assert!(!cert.is_strong_evidence());
}

#[test]
fn test_weak_evidence_property_fails() {
    let cert = HybridCertificate {
        formal_dim: 64,
        formal_property: "output_bounded".to_string(),
        formal_is_sound: true,
        statistical_dim: 512,
        n_samples: 1000,
        p_value: 0.001,
        effect_size: 1.2,
        property_holds: false,
    };
    assert!(!cert.is_strong_evidence());
}

#[test]
fn test_display_format() {
    let cert = HybridCertificate {
        formal_dim: 64,
        formal_property: "output_bounded".to_string(),
        formal_is_sound: true,
        statistical_dim: 512,
        n_samples: 1000,
        p_value: 0.0012,
        effect_size: 1.5,
        property_holds: true,
    };
    let s = format!("{cert}");
    assert!(s.contains("formal_dim=64"));
    assert!(s.contains("stat_dim=512"));
    assert!(s.contains("n=1000"));
    assert!(s.contains("p=0.0012"));
    assert!(s.contains("d=1.50"));
    assert!(s.contains("holds=true"));
}

// --- verify_layerwise_with_timing tests (NY gated) ---

#[cfg(feature = "ny")]
mod crown_timing {
    use crate::cost_model::HardwareCostModel;
    use crate::monotonicity::propagation_mode_is_sound_crown_family;
    use crate::pipeline::{verify_layerwise_with_timing, verify_pipeline_with_timing};
    use nn_dsl::{DispatchStep, ScalarType, TensorBlockBuilder, TensorNodeId};
    use nn_verify::{BoundedTensor, TensorParamBinding};
    use ndarray::{ArrayD, IxDyn};

    /// Build a minimal 2-layer pipeline (Linear + ReLU) for testing.
    ///
    /// Layer 0: Linear [4] -> [8] with fixed weight
    /// Layer 1: ReLU [8] -> [8]
    fn two_layer_crown_pipeline() -> (
        Vec<(nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>)>,
        BoundedTensor,
        Vec<DispatchStep>,
    ) {
        let dim_in = 4;
        let dim_out = 8;

        // Build Linear layer kernel def.
        let mut builder = TensorBlockBuilder::new("linear_layer");
        let input = builder.add_input("input", &[dim_in]);
        let weight = builder.add_input("weight", &[dim_out, dim_in]);
        let linear_out = builder.add_linear(input, weight, None, &[dim_out]);
        let linear_def = builder.build(linear_out).expect("valid graph");
        let weight_data = ArrayD::from_elem(IxDyn(&[dim_out, dim_in]), 0.1_f32);
        let linear_bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::ConstantTensor(weight_data),
        ];

        // Build ReLU layer kernel def.
        let mut builder2 = TensorBlockBuilder::new("relu_layer");
        let relu_input = builder2.add_input("input", &[dim_out]);
        let relu_output = builder2.add_relu(relu_input, &[dim_out]);
        let relu_def = builder2.build(relu_output).expect("valid graph");
        let relu_bindings = vec![TensorParamBinding::Variable];

        let layers = vec![(linear_def, linear_bindings), (relu_def, relu_bindings)];

        // Input bounds: [-1, 1] for each input dimension.
        let input_bounds = BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[dim_in]), -1.0_f32),
            ArrayD::from_elem(IxDyn(&[dim_in]), 1.0_f32),
        )
        .expect("valid bounds");

        // Matching dispatch plan.
        let dispatch_plan = vec![
            DispatchStep::Linear {
                kernel_name: "linear_layer".to_string(),
                dtype: ScalarType::F32,
                input: TensorNodeId::new(0),
                weight: TensorNodeId::new(1),
                bias: None,
                output: TensorNodeId::new(2),
                in_features: dim_in,
                out_features: dim_out,
                batch_size: 1,
                total_elements: dim_out,
            },
            DispatchStep::Relu {
                kernel_name: "relu_layer".to_string(),
                dtype: ScalarType::F32,
                input: TensorNodeId::new(2),
                output: TensorNodeId::new(3),
                total_elements: dim_out,
            },
        ];

        (layers, input_bounds, dispatch_plan)
    }

    #[test]
    fn test_layerwise_timing_pass() {
        let (layers, bounds, plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        let cert =
            verify_layerwise_with_timing(&layers, &bounds, &plan, &model, 1_000_000.0).unwrap();

        assert!(cert.bounds_cert.is_valid);
        assert!(cert.timing_bound_met);
        assert!(cert.overall_passed);
        assert!(cert.worst_case_time_us > 0.0);
        assert!(cert.total_flops > 0);
        assert!(cert.total_memory_bytes > 0);
        assert_eq!(cert.bounds_cert.stages.len(), 2);
    }

    #[test]
    fn test_layerwise_timing_exceeds_bound() {
        let (layers, bounds, plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        // Impossible bound: 0.001 μs (dispatch overhead alone is 5 μs per step).
        let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, 0.001).unwrap();

        assert!(cert.bounds_cert.is_valid);
        assert!(!cert.timing_bound_met);
        assert!(!cert.overall_passed);
    }

    #[test]
    fn test_layerwise_timing_invalid_bound_nan() {
        let (layers, bounds, plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        let result = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, f64::NAN);
        assert!(result.is_err());
    }

    #[test]
    fn test_layerwise_timing_invalid_bound_negative() {
        let (layers, bounds, plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        let result = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, -100.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_layerwise_timing_invalid_bound_zero() {
        let (layers, bounds, plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        let result = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_layerwise_timing_stages_are_crown_verified() {
        let (layers, bounds, plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        let cert =
            verify_layerwise_with_timing(&layers, &bounds, &plan, &model, 1_000_000.0).unwrap();

        // Each stage should have been verified by sound CROWN-family or IBP.
        for stage in &cert.bounds_cert.stages {
            assert!(
                propagation_mode_is_sound_crown_family(&stage.method) || stage.method == "IBP",
                "expected sound CROWN-family or IBP, got: {}",
                stage.method
            );
        }
    }

    #[test]
    fn test_timing_pipeline_preserves_alpha_beta_crown_stage_provenance() {
        let mut stages = super::two_stage_pipeline();
        stages[0].method = "AlphaCrown".to_string();
        stages[1].method = "BetaCrown".to_string();

        let plan = super::small_dispatch_plan();
        let model = HardwareCostModel::m4_max();
        let cert = verify_pipeline_with_timing(&stages, &plan, &model, 1_000_000.0).unwrap();

        assert!(cert.bounds_cert.is_valid);
        assert!(cert.bounds_cert.is_sound);
        assert!(cert.overall_passed);
        assert_eq!(cert.bounds_cert.stages[0].method, "AlphaCrown");
        assert_eq!(cert.bounds_cert.stages[1].method, "BetaCrown");
    }

    #[test]
    fn test_layerwise_timing_cost_profiles_match_plan() {
        let (layers, bounds, plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        let cert =
            verify_layerwise_with_timing(&layers, &bounds, &plan, &model, 1_000_000.0).unwrap();

        // Cost profiles should match the dispatch plan length.
        assert_eq!(cert.cost_profiles.len(), plan.len());

        // Each profile should have a non-empty name and positive timing.
        for profile in &cert.cost_profiles {
            assert!(!profile.layer_name.is_empty());
            assert!(profile.estimated_time_us > 0.0);
        }
    }

    #[test]
    fn test_layerwise_timing_empty_dispatch_plan() {
        let (layers, bounds, _plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        // Empty dispatch plan → 0 time → passes any positive bound.
        let cert =
            verify_layerwise_with_timing(&layers, &bounds, &[], &model, 1_000_000.0).unwrap();

        assert!(cert.timing_bound_met);
        assert_eq!(cert.total_flops, 0);
        assert_eq!(cert.total_memory_bytes, 0);
        assert_eq!(cert.worst_case_time_us, 0.0);
        // Bounds cert should still be valid from CROWN propagation.
        assert!(cert.bounds_cert.is_valid);
    }

    #[test]
    fn test_layerwise_timing_overall_requires_sound() {
        let (layers, bounds, plan) = two_layer_crown_pipeline();
        let model = HardwareCostModel::m4_max();
        let cert =
            verify_layerwise_with_timing(&layers, &bounds, &plan, &model, 1_000_000.0).unwrap();

        // overall_passed requires bounds_cert.is_sound (CROWN, not IBP).
        // With small dimensions, CROWN should succeed, making overall_passed = true.
        if cert.bounds_cert.is_sound {
            assert!(cert.overall_passed);
        } else {
            // If IBP fallback occurred, overall should fail even with valid timing.
            assert!(!cert.overall_passed);
        }
    }
}
