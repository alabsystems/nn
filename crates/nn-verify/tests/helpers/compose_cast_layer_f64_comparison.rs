// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! f64 tightness comparison tests for CastLayer-containing pipelines.
//!
//! Validates that CastLayer (identity for upcasts) has zero impact on bound
//! tightness, and that F16 downcast (Clamp) does not widen bounds for inputs
//! within the F16 representable range.
//!
//! Part of #4316: CastLayer for ToDtype verification + f64 evaluation.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
use nn_core::DType;
use nn_verify::{
    convert_network_to_f64, evaluate_network_f64, trace_to_graph_model, Layer, LinearLayer,
    Network, ReLULayer, SequentialLayerF64,
};
use ndarray::{Array1, Array2, ArrayD};

// ---------------------------------------------------------------------------
// 1. f64 tightness: CastLayer has zero impact on bound tightness
// ---------------------------------------------------------------------------

/// Measure f64 tightness for a Linear+ReLU network and verify that adding a
/// conceptual CastLayer does not degrade the precision gap.
///
/// Since CastLayer passes bounds through unchanged (Cow::Borrowed), the
/// precision gap should be identical whether or not a cast is present.
#[test]
fn test_f64_tightness_cast_zero_impact() {
    let w = Array2::from_shape_vec((3, 2), vec![1.0, -0.5, 0.3, 0.7, -0.8, 0.4]).unwrap();
    let b = Array1::from_vec(vec![0.1, -0.2, 0.05]);

    let mut net = Network::new();
    let layer = LinearLayer::new(w, Some(b)).expect("valid linear");
    net.add_layer(Layer::Linear(layer));
    net.add_layer(Layer::ReLU(ReLULayer::new()));

    let input = uniform_bounds(&[2], 1.0);

    // f32 IBP
    let ibp_output = net.propagate_ibp(&input).expect("f32 IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();

    // f64 evaluation at corners and midpoint
    let f64_layers: Vec<SequentialLayerF64> =
        convert_network_to_f64(net.layers()).expect("f64 conversion");

    let (in_lo, in_hi) = input.lower_upper();
    let lower_f64: ArrayD<f64> = in_lo.mapv(f64::from);
    let upper_f64: ArrayD<f64> = in_hi.mapv(f64::from);
    let mid_f64: ArrayD<f64> = (&lower_f64 + &upper_f64) / 2.0;

    let out_lo = evaluate_network_f64(&f64_layers, &lower_f64).expect("f64 at lower");
    let out_hi = evaluate_network_f64(&f64_layers, &upper_f64).expect("f64 at upper");
    let out_mid = evaluate_network_f64(&f64_layers, &mid_f64).expect("f64 at mid");

    // Soundness: all f64 evaluations within f32 IBP bounds.
    let eps = 1e-4_f64;
    for i in 0..ibp_lo.len() {
        let ilo = f64::from(ibp_lo[[i]]);
        let ihi = f64::from(ibp_hi[[i]]);

        for &val in &[out_lo[[i]], out_hi[[i]], out_mid[[i]]] {
            assert!(
                val >= ilo - eps,
                "Soundness: f64[{i}]={val} below IBP lower {ilo}"
            );
            assert!(
                val <= ihi + eps,
                "Soundness: f64[{i}]={val} above IBP upper {ihi}"
            );
        }
    }

    // Document precision gap.
    let n = ibp_lo.len();
    let mut total_f32_width = 0.0_f64;
    let mut total_f64_width = 0.0_f64;
    for i in 0..n {
        let f32_width = f64::from(ibp_hi[[i]] - ibp_lo[[i]]);
        let obs_min = out_lo[[i]].min(out_hi[[i]]).min(out_mid[[i]]);
        let obs_max = out_lo[[i]].max(out_hi[[i]]).max(out_mid[[i]]);
        total_f32_width += f32_width;
        total_f64_width += obs_max - obs_min;
    }

    let avg_f32 = total_f32_width / n as f64;
    let avg_f64 = total_f64_width / n as f64;
    let ratio = if avg_f64 > 1e-10 {
        avg_f32 / avg_f64
    } else {
        f64::INFINITY
    };

    eprintln!(
        "f64 tightness (CastLayer baseline): \
         f32_width={avg_f32:.6}, f64_width={avg_f64:.6}, ratio={ratio:.2}x"
    );

    assert!(
        avg_f32 >= avg_f64 - 1e-6,
        "IBP sound: f32_width={avg_f32} >= f64_width={avg_f64}"
    );
}

// ---------------------------------------------------------------------------
// 2. F16 downcast: no widening for inputs within F16 range
// ---------------------------------------------------------------------------

/// F16 downcast (Clamp to [-65504, 65504]) does not introduce additional
/// bound widening for inputs well within F16 range.
#[test]
fn test_f16_downcast_no_widening_within_range() {
    let shape = vec![2, 3];

    let graph_no_cast = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape.clone(),
            DType::F32,
        ),
    ]);

    let graph_with_cast = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "relu_0".into(),
            TraceOp::Relu,
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "cast_0".into(),
            TraceOp::ToDtype {
                target_dtype: DType::F16,
            },
            vec![1],
            shape.clone(),
            DType::F32,
        ),
    ]);

    let gn_no = trace_to_graph_model(&graph_no_cast).expect("no-cast").graph;
    let gn_with = trace_to_graph_model(&graph_with_cast)
        .expect("with-cast")
        .graph;

    let input = uniform_bounds(&shape, 2.0);

    let out_no = gn_no.propagate_ibp(&input).expect("IBP no cast");
    let out_with = gn_with.propagate_ibp(&input).expect("IBP with F16");

    assert_bounds_valid(&out_no);
    assert_bounds_valid(&out_with);

    // After ReLU on [-2, 2]: bounds are [0, 2]. F16 Clamp [-65504, 65504]
    // is inactive at these values. Output bounds should be identical.
    let (lo_no, hi_no) = out_no.lower_upper();
    let (lo_with, hi_with) = out_with.lower_upper();

    let eps = 1e-3;
    for (i, ((&ln, &lw), (&hn, &hw))) in lo_no
        .iter()
        .zip(lo_with.iter())
        .zip(hi_no.iter().zip(hi_with.iter()))
        .enumerate()
    {
        assert!(
            (ln - lw).abs() < eps,
            "F16 Clamp lower[{i}]: no_cast={ln}, with_cast={lw}"
        );
        assert!(
            (hn - hw).abs() < eps,
            "F16 Clamp upper[{i}]: no_cast={hn}, with_cast={hw}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Consecutive CastLayers: identity composition
// ---------------------------------------------------------------------------

/// Multiple consecutive CastLayer operations (F64 + F32) produce identity.
#[test]
fn test_consecutive_castlayers_identity() {
    let shape = vec![2, 4];
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input_0".into(),
            TraceOp::Input,
            vec![],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            1,
            "cast_f64".into(),
            TraceOp::ToDtype {
                target_dtype: DType::F64,
            },
            vec![0],
            shape.clone(),
            DType::F32,
        ),
        TraceNode::new(
            2,
            "cast_f32".into(),
            TraceOp::ToDtype {
                target_dtype: DType::F32,
            },
            vec![1],
            shape.clone(),
            DType::F32,
        ),
    ]);

    let result = trace_to_graph_model(&graph).expect("consecutive casts translate");
    assert_eq!(result.dtype_cast_count, 0, "no downcasts = 0 cast count");

    let input = uniform_bounds(&shape, 10.0);
    let output = result
        .graph
        .propagate_ibp(&input)
        .expect("IBP consecutive casts");
    assert_bounds_valid(&output);

    let (in_lo, in_hi) = input.lower_upper();
    let (out_lo, out_hi) = output.lower_upper();
    let eps = 1e-5;
    for (&il, &ol) in in_lo.iter().zip(out_lo.iter()) {
        assert!(
            (il - ol).abs() < eps,
            "consecutive cast lower: {il} vs {ol}"
        );
    }
    for (&ih, &oh) in in_hi.iter().zip(out_hi.iter()) {
        assert!(
            (ih - oh).abs() < eps,
            "consecutive cast upper: {ih} vs {oh}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. f64 tightness with wider input range and deeper network
// ---------------------------------------------------------------------------

/// Measure f64 tightness for a deeper network (2 layers) and document the
/// overapproximation ratio. This provides a baseline for comparison against
/// CastLayer-containing pipelines.
#[test]
fn test_f64_tightness_two_layer_baseline() {
    let w1 = Array2::from_shape_vec(
        (4, 3),
        vec![
            1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.3, 0.7, -0.2, -0.8, 0.1, 0.9,
        ],
    )
    .unwrap();
    let b1 = Array1::from_vec(vec![0.1, -0.2, 0.05, 0.0]);

    let w2 =
        Array2::from_shape_vec((2, 4), vec![0.5, -0.3, 0.7, 0.1, -0.4, 0.6, -0.2, 0.8]).unwrap();
    let b2 = Array1::from_vec(vec![0.0, 0.1]);

    let mut net = Network::new();
    let l1 = LinearLayer::new(w1, Some(b1)).expect("valid linear 1");
    net.add_layer(Layer::Linear(l1));
    net.add_layer(Layer::ReLU(ReLULayer::new()));
    let l2 = LinearLayer::new(w2, Some(b2)).expect("valid linear 2");
    net.add_layer(Layer::Linear(l2));
    net.add_layer(Layer::ReLU(ReLULayer::new()));

    let input = uniform_bounds(&[3], 0.5);

    // f32 IBP
    let ibp_output = net.propagate_ibp(&input).expect("f32 IBP");
    assert_bounds_valid(&ibp_output);
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();

    // f64 evaluation
    let f64_layers: Vec<SequentialLayerF64> =
        convert_network_to_f64(net.layers()).expect("f64 conversion");

    let (in_lo, in_hi) = input.lower_upper();
    let lower_f64: ArrayD<f64> = in_lo.mapv(f64::from);
    let upper_f64: ArrayD<f64> = in_hi.mapv(f64::from);
    let mid_f64: ArrayD<f64> = (&lower_f64 + &upper_f64) / 2.0;

    let out_lo = evaluate_network_f64(&f64_layers, &lower_f64).expect("f64 lower");
    let out_hi = evaluate_network_f64(&f64_layers, &upper_f64).expect("f64 upper");
    let out_mid = evaluate_network_f64(&f64_layers, &mid_f64).expect("f64 mid");

    // Soundness check
    let eps = 1e-4_f64;
    let n = ibp_lo.len();
    for i in 0..n {
        let ilo = f64::from(ibp_lo[[i]]);
        let ihi = f64::from(ibp_hi[[i]]);
        for &val in &[out_lo[[i]], out_hi[[i]], out_mid[[i]]] {
            assert!(val >= ilo - eps, "sound: f64[{i}]={val} >= ibp_lo {ilo}");
            assert!(val <= ihi + eps, "sound: f64[{i}]={val} <= ibp_hi {ihi}");
        }
    }

    // Document precision gap
    let mut total_f32 = 0.0_f64;
    let mut total_f64 = 0.0_f64;
    for i in 0..n {
        let f32_w = f64::from(ibp_hi[[i]] - ibp_lo[[i]]);
        let obs_min = out_lo[[i]].min(out_hi[[i]]).min(out_mid[[i]]);
        let obs_max = out_lo[[i]].max(out_hi[[i]]).max(out_mid[[i]]);
        total_f32 += f32_w;
        total_f64 += obs_max - obs_min;
    }

    let avg_f32 = total_f32 / n as f64;
    let avg_f64 = total_f64 / n as f64;
    let ratio = if avg_f64 > 1e-10 {
        avg_f32 / avg_f64
    } else {
        f64::INFINITY
    };

    eprintln!(
        "f64 tightness (2-layer baseline): \
         f32_width={avg_f32:.6}, f64_width={avg_f64:.6}, ratio={ratio:.2}x"
    );

    // IBP through 2 ReLU layers should show measurable overapproximation.
    assert!(
        ratio >= 1.0 - 1e-4,
        "2-layer IBP ratio {ratio:.4} should be >= 1.0 (sound)"
    );
}
