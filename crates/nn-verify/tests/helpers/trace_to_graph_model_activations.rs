// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for unary activation trace-to-graph translation
//! via the `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Mirrors `trace_to_graph_activations.rs` (old `trace_to_graph_network` path)
//! to ensure equivalent coverage on the new path.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Dropout, Module};
use nn_core::{DType, Device};
use nn_verify::{propagate_with_crown_fallback, trace_to_graph_model, BoundedTensor, PropMethod};
use ndarray::{ArrayD, IxDyn};

fn cpu() -> Device {
    Device::Cpu
}

// -- Tanh IBP -----------------------------------------------------------------

#[test]
fn test_model_trace_tanh_ibp() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.tanh()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Tanh translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v >= -1.0 - 1e-5,
            "tanh lower bound should be >= -1, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(v <= 1.0 + 1e-5, "tanh upper bound should be <= 1, got {v}");
    }
    for &v in lo.iter() {
        assert!(
            v <= -0.9,
            "tanh lower bound should reach near -0.964, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v >= 0.9,
            "tanh upper bound should reach near 0.964, got {v}"
        );
    }
}

// -- Exp IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_exp_ibp() {
    let x = DynTensor::new(&[0.0, 0.5, -0.5, 1.0, -1.0, 0.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.exp()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("Exp translation").graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "exp lower bound should be >= 0, got {v}");
    }
    for &v in lo.iter() {
        assert!(v >= 0.35, "exp lower bound should be >= exp(-1), got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.8, "exp upper bound should be <= exp(1), got {v}");
    }
}

// -- Gelu IBP -----------------------------------------------------------------

#[test]
fn test_model_trace_gelu_ibp() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.gelu()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Gelu translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.2, "gelu lower bound too negative: {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.1, "gelu upper bound too high for x in [-2,2]: {v}");
    }
}

// -- GeluErf IBP --------------------------------------------------------------

#[test]
fn test_model_trace_gelu_erf_ibp() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.gelu_erf()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("GeluErf translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.2, "gelu_erf lower bound too negative: {v}");
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.1,
            "gelu_erf upper bound too high for x in [-2,2]: {v}"
        );
    }
}

// -- SiLU IBP -----------------------------------------------------------------

#[test]
fn test_model_trace_silu_ibp() {
    let x = DynTensor::new(&[0.5, -0.5, 1.0, -1.0, 2.0, -2.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.silu()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Silu translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.3, "silu lower bound too negative: {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.1, "silu upper bound too high for x in [-2,2]: {v}");
    }
}

// -- Sigmoid IBP --------------------------------------------------------------

#[test]
fn test_model_trace_sigmoid_ibp() {
    let x = DynTensor::new(&[0.0, 1.0, -1.0, 2.0, -2.0, 0.5], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.sigmoid()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Sigmoid translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "sigmoid lower bound should be >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "sigmoid upper bound should be <= 1, got {v}");
    }
    for &v in lo.iter() {
        assert!(
            v <= 0.13,
            "sigmoid lower bound should reach near 0.119, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v >= 0.87,
            "sigmoid upper bound should reach near 0.881, got {v}"
        );
    }
}

// -- Sqrt IBP -----------------------------------------------------------------

#[test]
fn test_model_trace_sqrt_ibp() {
    let x = DynTensor::new(&[1.0, 4.0, 9.0, 16.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.sqrt()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Sqrt translation")
        .graph;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 2]), 0.5f32),
        ArrayD::from_elem(IxDyn(&[2, 2]), 4.0f32),
    )
    .expect("positive bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v >= 0.70,
            "sqrt lower bound should be >= sqrt(0.5), got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.01,
            "sqrt upper bound should be <= sqrt(4.0), got {v}"
        );
    }
}

// -- Abs IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_abs_ibp() {
    let x = DynTensor::new(&[-1.0, 2.0, -3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.abs()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("Abs translation").graph;
    let input_bounds = uniform_bounds(&[2, 2], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "abs lower bound should be >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.0 + 0.1,
            "abs upper bound should be <= 2 for x in [-2,2], got {v}"
        );
    }
}

// -- Neg IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_neg_ibp() {
    let x = DynTensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.neg()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("Neg translation").graph;
    let input_bounds = uniform_bounds(&[2, 2], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -1.0 - 1e-5, "neg lower bound too low: {l}");
        assert!(u <= 1.0 + 1e-5, "neg upper bound too high: {u}");
        assert!(l <= -1.0 + 0.01, "neg lower bound should reach -1, got {l}");
        assert!(u >= 1.0 - 0.01, "neg upper bound should reach 1, got {u}");
    }
}

// -- Neg CROWN ----------------------------------------------------------------

#[test]
fn test_model_trace_neg_crown() {
    let x = DynTensor::new(&[1.0, -2.0, 3.0, -4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.neg()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[2, 2], 1.0);

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&output);

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for Neg (linear). Error: {crown_err:?}"
    );
}

// -- Sqr IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_sqr_ibp() {
    let x = DynTensor::new(&[-1.0, 0.5, 2.0, -0.5], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.sqr()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("Sqr translation").graph;
    let input_bounds = uniform_bounds(&[2, 2], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "sqr lower bound should be >= 0, got {v}");
    }
    for &v in hi.iter() {
        assert!(
            v <= 1.0 + 0.1,
            "sqr upper bound should be <= 1 for x in [-1,1], got {v}"
        );
    }
}

// -- Recip IBP ----------------------------------------------------------------

#[test]
fn test_model_trace_recip_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 4.0, 0.5], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.recip()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Recip translation")
        .graph;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 2]), 0.5f32),
        ArrayD::from_elem(IxDyn(&[2, 2]), 4.0f32),
    )
    .expect("positive bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v >= 0.24,
            "recip lower bound should be >= 1/4.0 = 0.25, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.01,
            "recip upper bound should be <= 1/0.5 = 2.0, got {v}"
        );
    }
}

// -- Log IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_log_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.log()?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("Log translation").graph;
    let input_bounds = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[2, 2]), 0.5f32),
        ArrayD::from_elem(IxDyn(&[2, 2]), 4.0f32),
    )
    .expect("positive bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.70, "log lower bound should be >= ln(0.5), got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.40, "log upper bound should be <= ln(4.0), got {v}");
    }
}

// -- Dropout IBP (identity at inference) --------------------------------------

#[test]
fn test_model_trace_dropout_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let dropout = Dropout::new(0.5);

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = dropout.forward(&x)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Dropout translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -1.0 - 1e-5, "dropout lower bound changed: {l}");
        assert!(u <= 1.0 + 1e-5, "dropout upper bound changed: {u}");
        assert!(
            l <= -1.0 + 0.01,
            "dropout lower bound should reach -1, got {l}"
        );
        assert!(
            u >= 1.0 - 0.01,
            "dropout upper bound should reach 1, got {u}"
        );
    }
}
