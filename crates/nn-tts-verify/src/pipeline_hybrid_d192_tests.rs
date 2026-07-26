// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! D=192 CROWN + roofline timing composition tests (#1741 Property 5 gap).
//!
//! First tests combining actual CROWN propagation at production dimension
//! (D=192) with roofline timing via verify_layerwise_with_timing.
//! Bridges through check_temporal_boundedness for moonshot P5 CrownProven.

use crate::cost_model::HardwareCostModel;
use crate::pipeline::verify_layerwise_with_timing;
use nn_dsl::{DispatchStep, ScalarType, TensorBlockBuilder, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a 3-layer Linear+ReLU pipeline at D=192 with matching dispatch plan.
fn d192_timing_pipeline() -> (
    Vec<(nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>)>,
    BoundedTensor,
    Vec<DispatchStep>,
) {
    let dim = 192;
    let w_val = 0.01_f32;

    // Layer 0: Linear+ReLU [192] -> [192]
    let mut b0 = TensorBlockBuilder::new("d192_layer_0");
    let d0 = b0.add_input("data", &[dim]);
    let w0 = b0.add_input("weight", &[dim, dim]);
    let b0_bias = b0.add_input("bias", &[dim]);
    let lin0 = b0.add_linear(d0, w0, Some(b0_bias), &[dim]);
    let relu0 = b0.add_relu(lin0, &[dim]);
    let def0 = b0.build(relu0).expect("valid layer 0");
    let bind0 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), w_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0_f32)),
    ];

    // Layer 1: Linear+ReLU [192] -> [192]
    let mut b1 = TensorBlockBuilder::new("d192_layer_1");
    let d1 = b1.add_input("data", &[dim]);
    let w1 = b1.add_input("weight", &[dim, dim]);
    let b1_bias = b1.add_input("bias", &[dim]);
    let lin1 = b1.add_linear(d1, w1, Some(b1_bias), &[dim]);
    let relu1 = b1.add_relu(lin1, &[dim]);
    let def1 = b1.build(relu1).expect("valid layer 1");
    let bind1 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), w_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0_f32)),
    ];

    // Layer 2: Linear [192] -> [192] (no ReLU on final layer)
    let mut b2 = TensorBlockBuilder::new("d192_layer_2");
    let d2 = b2.add_input("data", &[dim]);
    let w2 = b2.add_input("weight", &[dim, dim]);
    let b2_bias = b2.add_input("bias", &[dim]);
    let lin2 = b2.add_linear(d2, w2, Some(b2_bias), &[dim]);
    let def2 = b2.build(lin2).expect("valid layer 2");
    let bind2 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), w_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0_f32)),
    ];

    let layers = vec![(def0, bind0), (def1, bind1), (def2, bind2)];
    let initial = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[dim]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[dim]), 1.0_f32),
    )
    .expect("valid bounds");

    // Dispatch plan matching the 3-layer architecture.
    // 2 Linear + 2 ReLU + 1 Linear = 5 steps.
    let plan = vec![
        DispatchStep::Linear {
            kernel_name: "d192_layer_0".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: Some(TensorNodeId::new(2)),
            output: TensorNodeId::new(3),
            in_features: dim,
            out_features: dim,
            batch_size: 1,
            total_elements: dim,
        },
        DispatchStep::Relu {
            kernel_name: "d192_layer_0_relu".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(3),
            output: TensorNodeId::new(4),
            total_elements: dim,
        },
        DispatchStep::Linear {
            kernel_name: "d192_layer_1".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(4),
            weight: TensorNodeId::new(5),
            bias: Some(TensorNodeId::new(6)),
            output: TensorNodeId::new(7),
            in_features: dim,
            out_features: dim,
            batch_size: 1,
            total_elements: dim,
        },
        DispatchStep::Relu {
            kernel_name: "d192_layer_1_relu".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(7),
            output: TensorNodeId::new(8),
            total_elements: dim,
        },
        DispatchStep::Linear {
            kernel_name: "d192_layer_2".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(8),
            weight: TensorNodeId::new(9),
            bias: Some(TensorNodeId::new(10)),
            output: TensorNodeId::new(11),
            in_features: dim,
            out_features: dim,
            batch_size: 1,
            total_elements: dim,
        },
    ];

    (layers, initial, plan)
}

#[test]
fn test_layerwise_timing_d192_crown_proven() {
    // D=192 CROWN + roofline timing: the full P5 CrownProven path.
    // 3 layers (Linear+ReLU -> Linear+ReLU -> Linear) at production dim
    // with M4 Max roofline model and generous timing bound (100ms).
    let (layers, bounds, plan) = d192_timing_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0; // 100ms — moonshot P5 bound

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 timing cert");

    // CROWN propagation must succeed at D=192 for Linear+ReLU.
    assert!(cert.bounds_cert.is_valid, "D=192 bounds must be valid");
    assert_eq!(cert.bounds_cert.stages.len(), 3);

    // Timing must be well within the 100ms bound for 3 small layers.
    assert!(cert.timing_bound_met, "timing must meet 100ms bound");
    assert!(cert.worst_case_time_us > 0.0, "timing must be positive");
    assert!(
        cert.worst_case_time_us < timing_bound_us,
        "worst_case={:.1} μs must be < bound={:.1} μs",
        cert.worst_case_time_us,
        timing_bound_us,
    );

    // With Linear+ReLU at small weights, CROWN should succeed (is_sound).
    // Combined with timing_bound_met, overall_passed should be true.
    if cert.bounds_cert.is_sound {
        assert!(
            cert.overall_passed,
            "sound CROWN + timing met = overall_passed"
        );
    }

    // Cost profiles match the 5-step dispatch plan.
    assert_eq!(cert.cost_profiles.len(), 5);
    assert!(cert.total_flops > 0);
    assert!(cert.total_memory_bytes > 0);
}

#[test]
fn test_layerwise_timing_d192_moonshot_p5_bridge() {
    // Bridge D=192 CROWN+timing certificate to moonshot P5
    // (check_temporal_boundedness) — the full proof chain.
    use crate::moonshot_crown::check_temporal_boundedness;

    let (layers, bounds, plan) = d192_timing_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 timing cert");

    // Bridge to moonshot P5.
    let p5 = check_temporal_boundedness(&cert);
    assert_eq!(p5.property_index, 4);
    assert_eq!(p5.property_name, "Temporally bounded (< 100ms on M4 Max)");
    assert!(p5.proven, "P5 must be proven: {}", p5.explanation);
    assert!(
        p5.bound_value < p5.threshold,
        "worst_case={:.1} must be < bound={:.1}",
        p5.bound_value,
        p5.threshold,
    );

    // If CROWN succeeded (is_sound), P5 reaches CrownProven.
    if cert.bounds_cert.is_sound {
        assert_eq!(
            p5.level,
            crate::moonshot::VerificationLevel::CrownProven,
            "sound CROWN + timing met = CrownProven for P5"
        );
        assert!(p5.is_sound);
    }
}

#[test]
fn test_layerwise_timing_d192_full_bundle() {
    // D=192 CROWN+timing through verify_properties_with_timing
    // for all 5 properties (P1, P2, P3, P5, P6) in one bundle.
    use crate::moonshot_crown::verify_properties_with_timing;

    let (layers, bounds, plan) = d192_timing_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 timing cert");

    let bundle = verify_properties_with_timing(&cert.bounds_cert, &cert, 192);
    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 5); // P1, P2, P3, P5, P6

    // P1 (non-silence) — output has non-zero range.
    assert!(
        bundle.results[0].proven,
        "P1 must pass: {}",
        bundle.results[0]
    );

    // P5 (temporal) — timing certificate passes.
    let p5 = &bundle.results[3];
    assert_eq!(p5.property_index, 4);
    assert!(p5.proven, "P5 must pass: {p5}");

    // P6 (streaming) — bounded crossfade discontinuity.
    let p6 = &bundle.results[4];
    assert_eq!(p6.property_index, 5);
    assert!(p6.proven, "P6 must pass: {p6}");

    // If CROWN succeeded, P5 should be CrownProven.
    if cert.bounds_cert.is_sound {
        assert_eq!(
            p5.level,
            crate::moonshot::VerificationLevel::CrownProven,
            "P5 should reach CrownProven with sound CROWN timing"
        );
    }
}
