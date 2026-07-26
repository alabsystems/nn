// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for parameterized activation trace-to-graph translation:
//! `TraceOp::Elu { alpha }` and `TraceOp::LeakyRelu { slope }`.
//!
//! These dedicated `TraceOp` variants carry their parameters (alpha, slope)
//! directly, unlike the generic `TraceOp::Activation { name }` which uses
//! default values. The trace-to-graph-model path must propagate the actual
//! parameter values through to NY's `EluLayer` / `LeakyReLULayer`.
//!
//! Part of #2246 (dedicated Elu/LeakyRelu TraceOp variants).

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Activation, Module};
use nn_core::{DType, Device};
use nn_verify::{propagate_with_crown_fallback, trace_to_graph_model, PropMethod};

fn cpu() -> Device {
    Device::Cpu
}

// ---------------------------------------------------------------------------
// Elu IBP — default alpha = 1.0
// ---------------------------------------------------------------------------

#[test]
fn test_model_trace_elu_ibp_alpha_1() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();
    let elu = Activation::Elu(1.0);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = elu.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Elu alpha=1.0 translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // elu(x, 1.0) = x for x >= 0, alpha*(exp(x)-1) for x < 0
    // For x in [-2, 2]: lower bound = 1.0*(exp(-2)-1) ≈ -0.865
    for &v in lo.iter() {
        assert!(
            v >= -0.87,
            "elu(alpha=1) lower bound should be >= ~-0.865, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.01,
            "elu(alpha=1) upper bound should be <= 2 for x in [-2,2], got {v}"
        );
    }
    // Verify output is non-trivial (not all zeros)
    for &v in hi.iter() {
        assert!(v >= 1.9, "elu upper bound should reach near 2.0, got {v}");
    }
}

// ---------------------------------------------------------------------------
// Elu IBP — custom alpha = 0.5
// ---------------------------------------------------------------------------

#[test]
fn test_model_trace_elu_ibp_alpha_half() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();
    let elu = Activation::Elu(0.5);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = elu.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Elu alpha=0.5 translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // elu(x, 0.5) for x < 0: 0.5*(exp(x)-1)
    // For x = -2: 0.5*(exp(-2)-1) ≈ -0.432
    for &v in lo.iter() {
        assert!(
            v >= -0.45,
            "elu(alpha=0.5) lower bound should be >= ~-0.432, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.01,
            "elu(alpha=0.5) upper bound should be <= 2 for x in [-2,2], got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// LeakyRelu IBP — default slope = 0.01
// ---------------------------------------------------------------------------

#[test]
fn test_model_trace_leaky_relu_ibp_slope_001() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();
    let leaky = Activation::LeakyRelu(0.01);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = leaky.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("LeakyRelu slope=0.01 translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // leaky_relu(x, 0.01) for x < 0: 0.01*x
    // For x = -2: 0.01*(-2) = -0.02
    for &v in lo.iter() {
        assert!(
            v >= -0.03,
            "leaky_relu(slope=0.01) lower bound should be >= -0.02, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.01,
            "leaky_relu(slope=0.01) upper bound should be <= 2, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v >= 1.9,
            "leaky_relu upper bound should reach near 2.0, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// LeakyRelu IBP — Kokoro-style slope = 0.1
// ---------------------------------------------------------------------------

#[test]
fn test_model_trace_leaky_relu_ibp_slope_01() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();
    let leaky = Activation::LeakyRelu(0.1);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = leaky.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("LeakyRelu slope=0.1 translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    // leaky_relu(x, 0.1) for x < 0: 0.1*x
    // For x = -2: 0.1*(-2) = -0.2
    for &v in lo.iter() {
        assert!(
            v >= -0.21,
            "leaky_relu(slope=0.1) lower bound should be >= -0.2, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.01,
            "leaky_relu(slope=0.1) upper bound should be <= 2, got {v}"
        );
    }
}

// ---------------------------------------------------------------------------
// LeakyRelu CROWN — piecewise linear, CROWN should succeed
// ---------------------------------------------------------------------------

#[test]
fn test_model_trace_leaky_relu_crown() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();
    let leaky = Activation::LeakyRelu(0.01);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = leaky.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&output);

    // LeakyRelu is piecewise linear — CROWN should handle it without fallback.
    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for LeakyRelu (piecewise linear). Error: {crown_err:?}"
    );

    // CROWN bounds should be at least as tight as IBP.
    let ibp_output = gn.propagate_ibp(&input_bounds).expect("IBP baseline");
    let (crown_lo, crown_hi) = output.lower_upper();
    let (ibp_lo, ibp_hi) = ibp_output.lower_upper();
    let eps = 1e-4;
    for (&cl, &il) in crown_lo.iter().zip(ibp_lo.iter()) {
        assert!(
            cl >= il - eps,
            "CROWN lower {cl} should be >= IBP lower {il}"
        );
    }
    for (&cu, &iu) in crown_hi.iter().zip(ibp_hi.iter()) {
        assert!(
            cu <= iu + eps,
            "CROWN upper {cu} should be <= IBP upper {iu}"
        );
    }
}

// ---------------------------------------------------------------------------
// Elu CROWN — nonlinear, CROWN may or may not succeed (test robustness)
// ---------------------------------------------------------------------------

#[test]
fn test_model_trace_elu_crown_or_ibp_fallback() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();
    let elu = Activation::Elu(1.0);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = elu.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("propagation");
    assert_bounds_valid(&output);

    // Whether CROWN succeeds or falls back to IBP, bounds must be valid.
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.87, "elu lower bound should be >= ~-0.865, got {v}");
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.01,
            "elu upper bound should be <= 2 for x in [-2,2], got {v}"
        );
    }
}
