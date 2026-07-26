// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! D=192 CROWN + roofline timing composition tests for P2 Non-clipping (#1741).
//!
//! Uses Linear+Tanh architecture where Tanh naturally bounds output to (-1, 1),
//! guaranteeing P2 (Non-clipping: samples in [-1, 1]) reaches CrownProven
//! when CROWN propagation succeeds with sound bounds.

use crate::cost_model::HardwareCostModel;
use crate::pipeline::verify_layerwise_with_timing;
use nn_dsl::{DispatchStep, ScalarType, TensorBlockBuilder, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a 3-layer Linear+Tanh pipeline at D=192 with matching dispatch plan.
///
/// Architecture: Linear+Tanh → Linear+Tanh → Linear+Tanh
/// Each Tanh bounds output to (-1, 1), so the final output satisfies P2.
/// Small weights (0.01) ensure CROWN propagation stays tractable.
fn d192_tanh_bounded_pipeline() -> (
    Vec<(nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>)>,
    BoundedTensor,
    Vec<DispatchStep>,
) {
    let dim = 192;
    let w_val = 0.01_f32;

    // Layer 0: Linear+Tanh [192] -> [192]
    let mut b0 = TensorBlockBuilder::new("p2_layer_0");
    let d0 = b0.add_input("data", &[dim]);
    let w0 = b0.add_input("weight", &[dim, dim]);
    let b0_bias = b0.add_input("bias", &[dim]);
    let lin0 = b0.add_linear(d0, w0, Some(b0_bias), &[dim]);
    let tanh0 = b0.add_tanh(lin0, &[dim]);
    let def0 = b0.build(tanh0).expect("valid layer 0");
    let bind0 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), w_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0_f32)),
    ];

    // Layer 1: Linear+Tanh [192] -> [192]
    let mut b1 = TensorBlockBuilder::new("p2_layer_1");
    let d1 = b1.add_input("data", &[dim]);
    let w1 = b1.add_input("weight", &[dim, dim]);
    let b1_bias = b1.add_input("bias", &[dim]);
    let lin1 = b1.add_linear(d1, w1, Some(b1_bias), &[dim]);
    let tanh1 = b1.add_tanh(lin1, &[dim]);
    let def1 = b1.build(tanh1).expect("valid layer 1");
    let bind1 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), w_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), 0.0_f32)),
    ];

    // Layer 2: Linear+Tanh [192] -> [192]
    let mut b2 = TensorBlockBuilder::new("p2_layer_2");
    let d2 = b2.add_input("data", &[dim]);
    let w2 = b2.add_input("weight", &[dim, dim]);
    let b2_bias = b2.add_input("bias", &[dim]);
    let lin2 = b2.add_linear(d2, w2, Some(b2_bias), &[dim]);
    let tanh2 = b2.add_tanh(lin2, &[dim]);
    let def2 = b2.build(tanh2).expect("valid layer 2");
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

    // Dispatch plan: 3 layers × (Linear + Tanh) = 6 steps.
    let plan = vec![
        DispatchStep::Linear {
            kernel_name: "p2_layer_0".to_string(),
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
        DispatchStep::Tanh {
            kernel_name: "p2_layer_0_tanh".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(3),
            output: TensorNodeId::new(4),
            total_elements: dim,
        },
        DispatchStep::Linear {
            kernel_name: "p2_layer_1".to_string(),
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
        DispatchStep::Tanh {
            kernel_name: "p2_layer_1_tanh".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(7),
            output: TensorNodeId::new(8),
            total_elements: dim,
        },
        DispatchStep::Linear {
            kernel_name: "p2_layer_2".to_string(),
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
        DispatchStep::Tanh {
            kernel_name: "p2_layer_2_tanh".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(11),
            output: TensorNodeId::new(12),
            total_elements: dim,
        },
    ];

    (layers, initial, plan)
}

#[test]
fn test_p2_non_clipping_d192_crown_proven() {
    // D=192 Linear+Tanh architecture: CROWN bounds must be within [-1, 1].
    // Tanh naturally bounds output to (-1, 1), so P2 (non-clipping) should pass
    // whenever CROWN propagation succeeds.
    let (layers, bounds, plan) = d192_tanh_bounded_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0; // 100ms

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 tanh timing cert");

    // CROWN propagation must succeed at D=192.
    assert!(cert.bounds_cert.is_valid, "D=192 bounds must be valid");
    assert_eq!(cert.bounds_cert.stages.len(), 3);

    // Tanh output bounds must be within [-1, 1].
    for (i, stage) in cert.bounds_cert.stages.iter().enumerate() {
        for &ub in stage.output_upper.iter() {
            assert!(ub <= 1.0, "stage {i} upper bound {ub} exceeds 1.0");
        }
        for &lb in stage.output_lower.iter() {
            assert!(lb >= -1.0, "stage {i} lower bound {lb} below -1.0");
        }
    }

    // End-to-end bounds must also be in [-1, 1].
    let max_upper = crate::stats::fold_max_propagate_nan(
        cert.bounds_cert.e2e_output_upper.iter().copied(),
        f64::NEG_INFINITY,
    );
    let min_lower = crate::stats::fold_min_propagate_nan(
        cert.bounds_cert.e2e_output_lower.iter().copied(),
        f64::INFINITY,
    );

    assert!(max_upper <= 1.0, "e2e max_upper={max_upper} must be <= 1.0");
    assert!(
        min_lower >= -1.0,
        "e2e min_lower={min_lower} must be >= -1.0"
    );

    // Timing must pass.
    assert!(cert.timing_bound_met, "timing must meet 100ms bound");
}

#[test]
fn test_p2_non_clipping_d192_moonshot_bridge() {
    // Bridge D=192 tanh-bounded certificate to moonshot P2 (check_non_clipping).
    use crate::moonshot_crown::check_non_clipping;

    let (layers, bounds, plan) = d192_tanh_bounded_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 tanh timing cert");

    let p2 = check_non_clipping(&cert.bounds_cert);
    assert_eq!(p2.property_index, 1);
    assert_eq!(p2.property_name, "Non-clipping (samples in [-1, 1])");
    assert!(p2.proven, "P2 must be proven: {}", p2.explanation);
    assert!(
        p2.bound_value <= p2.threshold,
        "worst_bound={:.4} must be <= threshold={:.1}",
        p2.bound_value,
        p2.threshold,
    );

    // If CROWN succeeded (is_sound), P2 reaches CrownProven.
    if cert.bounds_cert.is_sound {
        assert_eq!(
            p2.level,
            crate::moonshot::VerificationLevel::CrownProven,
            "sound CROWN + tanh bounded = CrownProven for P2"
        );
        assert!(p2.is_sound);
    }
}

#[test]
fn test_p2_non_clipping_d192_full_bundle() {
    // D=192 tanh-bounded pipeline through verify_properties_with_timing
    // — verify P2 passes in the full 5-property bundle.
    use crate::moonshot_crown::verify_properties_with_timing;

    let (layers, bounds, plan) = d192_tanh_bounded_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 tanh timing cert");

    let bundle = verify_properties_with_timing(&cert.bounds_cert, &cert, 192);
    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 5); // P1, P2, P3, P5, P6

    // P2 (non-clipping) — tanh output in [-1, 1].
    let p2 = &bundle.results[1];
    assert_eq!(p2.property_index, 1);
    assert!(p2.proven, "P2 must pass: {p2}");

    // P5 (temporal) — timing certificate passes.
    let p5 = &bundle.results[3];
    assert_eq!(p5.property_index, 4);
    assert!(p5.proven, "P5 must pass: {p5}");

    // If CROWN succeeded, P2 should be CrownProven.
    if cert.bounds_cert.is_sound {
        assert_eq!(
            p2.level,
            crate::moonshot::VerificationLevel::CrownProven,
            "P2 should reach CrownProven with sound CROWN + tanh"
        );
    }
}
