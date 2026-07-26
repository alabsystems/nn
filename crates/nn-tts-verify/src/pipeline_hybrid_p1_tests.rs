// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! D=192 CROWN composition tests for P1 Non-silence (#1741).
//!
//! P1 (Non-silence) requires: max(|output_lower|, |output_upper|) > threshold.
//! A Linear+ReLU pipeline with non-zero weights and bias produces output whose
//! absolute bounds are above the RMS threshold (0.01), proving the model
//! generates non-trivial (non-silent) output.
//!
//! Uses the same 3-layer Linear+ReLU architecture as the D=192 P5 timing tests,
//! since that architecture naturally satisfies P1 (non-zero output bounds).

use crate::cost_model::HardwareCostModel;
use crate::pipeline::verify_layerwise_with_timing;
use nn_dsl::{DispatchStep, ScalarType, TensorBlockBuilder, TensorNodeId};
use nn_verify::{BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

/// Build a 3-layer Linear+ReLU pipeline at D=192 with bias to ensure
/// non-zero output bounds (P1 non-silence).
///
/// The key for P1 is that the bias shifts the output away from zero,
/// guaranteeing max(|output|) > threshold even after CROWN bound widening.
fn d192_non_silence_pipeline() -> (
    Vec<(nn_dsl::tensor_ir::TensorKernelDef, Vec<TensorParamBinding>)>,
    BoundedTensor,
    Vec<DispatchStep>,
) {
    let dim = 192;
    let w_val = 0.01_f32;
    let bias_val = 0.1_f32; // Non-zero bias ensures non-silence

    // Layer 0: Linear(bias=0.1)+ReLU [192] -> [192]
    let mut b0 = TensorBlockBuilder::new("p1_layer_0");
    let d0 = b0.add_input("data", &[dim]);
    let w0 = b0.add_input("weight", &[dim, dim]);
    let b0_bias = b0.add_input("bias", &[dim]);
    let lin0 = b0.add_linear(d0, w0, Some(b0_bias), &[dim]);
    let relu0 = b0.add_relu(lin0, &[dim]);
    let def0 = b0.build(relu0).expect("valid layer 0");
    let bind0 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), w_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), bias_val)),
    ];

    // Layer 1: Linear(bias=0.1)+ReLU [192] -> [192]
    let mut b1 = TensorBlockBuilder::new("p1_layer_1");
    let d1 = b1.add_input("data", &[dim]);
    let w1 = b1.add_input("weight", &[dim, dim]);
    let b1_bias = b1.add_input("bias", &[dim]);
    let lin1 = b1.add_linear(d1, w1, Some(b1_bias), &[dim]);
    let relu1 = b1.add_relu(lin1, &[dim]);
    let def1 = b1.build(relu1).expect("valid layer 1");
    let bind1 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), w_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), bias_val)),
    ];

    // Layer 2: Linear(bias=0.1) [192] -> [192] (no ReLU on final layer)
    let mut b2 = TensorBlockBuilder::new("p1_layer_2");
    let d2 = b2.add_input("data", &[dim]);
    let w2 = b2.add_input("weight", &[dim, dim]);
    let b2_bias = b2.add_input("bias", &[dim]);
    let lin2 = b2.add_linear(d2, w2, Some(b2_bias), &[dim]);
    let def2 = b2.build(lin2).expect("valid layer 2");
    let bind2 = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim, dim]), w_val)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim]), bias_val)),
    ];

    let layers = vec![(def0, bind0), (def1, bind1), (def2, bind2)];
    let initial = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[dim]), -1.0_f32),
        ArrayD::from_elem(IxDyn(&[dim]), 1.0_f32),
    )
    .expect("valid bounds");

    // Dispatch plan: 2 × (Linear + ReLU) + 1 × Linear = 5 steps.
    let plan = vec![
        DispatchStep::Linear {
            kernel_name: "p1_layer_0".to_string(),
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
            kernel_name: "p1_layer_0_relu".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(3),
            output: TensorNodeId::new(4),
            total_elements: dim,
        },
        DispatchStep::Linear {
            kernel_name: "p1_layer_1".to_string(),
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
            kernel_name: "p1_layer_1_relu".to_string(),
            dtype: ScalarType::F32,
            input: TensorNodeId::new(7),
            output: TensorNodeId::new(8),
            total_elements: dim,
        },
        DispatchStep::Linear {
            kernel_name: "p1_layer_2".to_string(),
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
fn test_p1_non_silence_d192_crown_proven() {
    // D=192 CROWN propagation for P1: output has non-zero absolute bounds.
    // 3-layer Linear(bias=0.1)+ReLU pipeline at production dimension.
    let (layers, bounds, plan) = d192_non_silence_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 P1 timing cert");

    // CROWN propagation must succeed at D=192.
    assert!(cert.bounds_cert.is_valid, "D=192 bounds must be valid");
    assert_eq!(cert.bounds_cert.stages.len(), 3);

    // Output bounds must be finite and non-zero.
    let last_stage = &cert.bounds_cert.stages[2];
    let max_abs_upper = last_stage
        .output_upper
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);
    let max_abs_lower = last_stage
        .output_lower
        .iter()
        .map(|x| x.abs())
        .fold(0.0_f64, f64::max);
    let max_abs = max_abs_upper.max(max_abs_lower);

    eprintln!(
        "P1 D=192: max_abs_output = {max_abs:.6}, \
         stages={}, is_sound={}",
        cert.bounds_cert.stages.len(),
        cert.bounds_cert.is_sound
    );

    // Non-zero bias + non-zero weights → non-zero output bounds.
    assert!(
        max_abs > 0.0,
        "output must have non-zero absolute bounds, got {max_abs}"
    );
}

#[test]
fn test_p1_non_silence_d192_moonshot_bridge() {
    // Bridge D=192 CROWN bounds to moonshot P1 (check_non_silence).
    use crate::moonshot_crown::check_non_silence;

    let (layers, bounds, plan) = d192_non_silence_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 timing cert");

    let p1 = check_non_silence(&cert.bounds_cert, 0.01);
    assert_eq!(p1.property_index, 0);
    assert_eq!(p1.property_name, "Non-silent (RMS > 0.01)");
    assert!(p1.proven, "P1 must be proven: {}", p1.explanation);
    assert!(
        p1.bound_value > p1.threshold,
        "max_abs={:.6} must exceed threshold={:.6}",
        p1.bound_value,
        p1.threshold,
    );

    eprintln!(
        "P1 bridge: bound_value={:.6}, threshold={:.6}, level={:?}",
        p1.bound_value, p1.threshold, p1.level,
    );

    // If CROWN succeeded (is_sound), P1 reaches CrownProven.
    if cert.bounds_cert.is_sound {
        assert_eq!(
            p1.level,
            crate::moonshot::VerificationLevel::CrownProven,
            "sound CROWN + non-silence proven = CrownProven for P1"
        );
        assert!(p1.is_sound);
    }
}

#[test]
fn test_p1_non_silence_d192_full_bundle() {
    // D=192 P1 through verify_properties_with_timing — full 5-property bundle.
    use crate::moonshot_crown::verify_properties_with_timing;

    let (layers, bounds, plan) = d192_non_silence_pipeline();
    let model = HardwareCostModel::m4_max();
    let timing_bound_us = 100_000.0;

    let cert = verify_layerwise_with_timing(&layers, &bounds, &plan, &model, timing_bound_us)
        .expect("D=192 timing cert");

    let bundle = verify_properties_with_timing(&cert.bounds_cert, &cert, 192);
    assert_eq!(bundle.verification_dim, 192);
    assert_eq!(bundle.results.len(), 5); // P1, P2, P3, P5, P6

    // P1 (non-silence) — the primary target of this test.
    let p1 = &bundle.results[0];
    assert_eq!(p1.property_index, 0);
    assert!(p1.proven, "P1 must pass in bundle: {p1}");

    eprintln!(
        "P1 in bundle: proven={}, level={:?}, bound={:.6}",
        p1.proven, p1.level, p1.bound_value,
    );

    // If CROWN succeeded, P1 should be CrownProven.
    if cert.bounds_cert.is_sound {
        assert_eq!(
            p1.level,
            crate::moonshot::VerificationLevel::CrownProven,
            "P1 should reach CrownProven with sound CROWN bounds"
        );
    }
}
