// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! D=192 CROWN composition tests for P3 Intelligibility proxy (#1741).
//!
//! P3 (Intelligibility proxy) requires: output_range / input_range < max_range_ratio.
//! A Linear+Tanh pipeline naturally bounds the range ratio because Tanh compresses
//! output to (-1, 1). With input bounds [-1, 1] (range=2.0) and Tanh output
//! bounds within (-1, 1) (range <= 2.0), the ratio is <= 1.0, well below the
//! 100.0 threshold. This proves the model produces "informative" (non-vacuous)
//! bounds — a proxy for intelligibility.
//!
//! Uses the same 3-layer Linear+Tanh architecture as the D=192 P2 tests,
//! since that architecture naturally satisfies P3 (bounded range ratio).

use crate::cost_model::HardwareCostModel;
use crate::pipeline::verify_layerwise_with_timing;
use nn_dsl::{DispatchStep, ScalarType, TensorBlockBuilder, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a 3-layer Linear+Tanh pipeline at D=192 with matching dispatch plan.
///
/// Architecture: Linear+Tanh -> Linear+Tanh -> Linear+Tanh
/// Each Tanh bounds output to (-1, 1), so the range ratio stays <= 1.0.
/// This satisfies P3's max_range_ratio=100.0 threshold by a wide margin.
fn d192_range_bounded_pipeline() -> (
    Vec<(nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>)>,
    BoundedTensor,
    Vec<DispatchStep>,
) {
    let dim = 192;
    let w_val = 0.01_f32;

    // Layer 0: Linear+Tanh [192] -> [192]
    let mut b0 = TensorBlockBuilder::new("p3_layer_0");
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
    let mut b1 = TensorBlockBuilder::new("p3_layer_1");
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
    let mut b2 = TensorBlockBuilder::new("p3_layer_2");
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

    // Dispatch plan: 3 layers x (Linear + Tanh) = 6 steps.
    let plan = vec![
        DispatchStep::Linear {
            kernel_name: "p3_layer_0".to_string(),
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
            kernel_name: "p3_layer_0_tanh".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(3),
            output: TensorNodeId::new(4),
            total_elements: dim,
        },
        DispatchStep::Linear {
            kernel_name: "p3_layer_1".to_string(),
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
            kernel_name: "p3_layer_1_tanh".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(7),
            output: TensorNodeId::new(8),
            total_elements: dim,
        },
        DispatchStep::Linear {
            kernel_name: "p3_layer_2".to_string(),
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
            kernel_name: "p3_layer_2_tanh".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(11),
            output: TensorNodeId::new(12),
            total_elements: dim,
        },
    ];

    (layers, initial, plan)
}

#[test]
fn test_p3_intelligibility_d192_crown_proven() {
    // D=192 Linear+Tanh architecture: CROWN bounds must have bounded range ratio.
    // Tanh output in (-1, 1) with input in [-1, 1] gives range_ratio <= 1.0,
    // well below the 100.0 threshold for P3.
    let (layers, bounds, plan) = d192_range_bounded_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 tanh timing cert");

    // CROWN propagation must succeed at D=192.
    assert!(cert.bounds_cert.is_valid, "D=192 bounds must be valid");
    assert_eq!(cert.bounds_cert.stages.len(), 3);

    // Compute range ratio: output_range / input_range.
    let input_range = cert
        .bounds_cert
        .e2e_input_upper
        .iter()
        .zip(cert.bounds_cert.e2e_input_lower.iter())
        .map(|(u, l)| u - l)
        .fold(0.0_f64, f64::max);

    let output_range = cert
        .bounds_cert
        .e2e_output_upper
        .iter()
        .zip(cert.bounds_cert.e2e_output_lower.iter())
        .map(|(u, l)| u - l)
        .fold(0.0_f64, f64::max);

    let range_ratio = if input_range > 0.0 {
        output_range / input_range
    } else {
        f64::INFINITY
    };

    eprintln!(
        "P3 D=192: input_range={input_range:.6}, output_range={output_range:.6}, \
         ratio={range_ratio:.6}, stages={}, is_sound={}",
        cert.bounds_cert.stages.len(),
        cert.bounds_cert.is_sound
    );

    // Tanh compresses range: ratio should be <= 1.0.
    assert!(
        range_ratio < 100.0,
        "range_ratio={range_ratio:.6} must be < 100.0"
    );
    assert!(range_ratio.is_finite(), "range_ratio must be finite");
}

#[test]
fn test_p3_intelligibility_d192_moonshot_bridge() {
    // Bridge D=192 tanh-bounded certificate to moonshot P3 (check_intelligibility_proxy).
    use crate::moonshot_crown::check_intelligibility_proxy;

    let (layers, bounds, plan) = d192_range_bounded_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 tanh timing cert");

    let p3 = check_intelligibility_proxy(&cert.bounds_cert, 100.0);
    assert_eq!(p3.property_index, 2);
    assert_eq!(p3.property_name, "Intelligible (attention monotonic)");
    assert!(p3.proven, "P3 must be proven: {}", p3.explanation);
    assert!(
        p3.bound_value < p3.threshold,
        "range_ratio={:.6} must be < threshold={:.1}",
        p3.bound_value,
        p3.threshold,
    );

    eprintln!(
        "P3 bridge: bound_value={:.6}, threshold={:.6}, level={:?}",
        p3.bound_value, p3.threshold, p3.level,
    );

    // If CROWN succeeded (is_sound), P3 reaches CrownPartial (proxy, not full monotonicity).
    if cert.bounds_cert.is_sound {
        assert_eq!(
            p3.level,
            crate::moonshot::VerificationLevel::CrownPartial,
            "sound CROWN + range-bounded = CrownPartial for P3"
        );
    }
}

#[test]
fn test_p3_intelligibility_d192_full_bundle() {
    // D=192 tanh-bounded pipeline through verify_properties_with_timing
    // -- verify P3 passes in the full 5-property bundle.
    use crate::moonshot_crown::verify_properties_with_timing;

    let (layers, bounds, plan) = d192_range_bounded_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 tanh timing cert");

    let bundle = verify_properties_with_timing(&cert.bounds_cert, &cert, 192);
    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 5); // P1, P2, P3, P5, P6

    // P3 (intelligibility proxy) -- range ratio within threshold.
    let p3 = &bundle.results[2];
    assert_eq!(p3.property_index, 2);
    assert!(p3.proven, "P3 must pass in bundle: {p3}");

    eprintln!(
        "P3 in bundle: proven={}, level={:?}, bound={:.6}",
        p3.proven, p3.level, p3.bound_value,
    );

    // If CROWN succeeded, P3 should be CrownPartial (proxy level).
    if cert.bounds_cert.is_sound {
        assert_eq!(
            p3.level,
            crate::moonshot::VerificationLevel::CrownPartial,
            "P3 should reach CrownPartial with sound CROWN + range bounded"
        );
    }
}
