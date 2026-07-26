// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for shape ops and Softmax trace-to-graph translation
//! via the `trace_to_graph_model` (LayerSpec → build_graph_network) path.
//!
//! Mirrors `trace_to_graph_shape_softmax.rs` (old `trace_to_graph_network` path)
//! to ensure equivalent coverage on the new path.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{record_input, trace_graph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_verify::{
    propagate_with_crown_fallback, trace_to_graph_model, trace_to_graph_model_multi_input,
    PropMethod,
};

fn cpu() -> Device {
    Device::Cpu
}

// -- Softmax IBP --------------------------------------------------------------

#[test]
fn test_model_trace_softmax_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.softmax(1)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Softmax translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "softmax lower bound should be ~0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.01, "softmax upper bound should be ~1, got {v}");
    }
}

// -- Softmax CROWN ------------------------------------------------------------

#[test]
fn test_model_trace_softmax_crown() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.softmax(1)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&output);
}

// -- Reshape IBP --------------------------------------------------------------

#[test]
fn test_model_trace_reshape_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.reshape([3, 2])?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[3, 2]);

    let gn = trace_to_graph_model(&graph)
        .expect("Reshape translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.len(), 6, "output should have 6 elements");
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -1.0 - 1e-6, "lower bound should be >= -1.0, got {l}");
        assert!(u <= 1.0 + 1e-6, "upper bound should be <= 1.0, got {u}");
    }
}

// -- Reshape CROWN ------------------------------------------------------------

#[test]
fn test_model_trace_reshape_crown() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.reshape([6])?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&output);

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for Reshape. Error: {crown_err:?}"
    );
}

// -- Narrow IBP ---------------------------------------------------------------

#[test]
fn test_model_trace_narrow_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.narrow(1, 1, 2)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 2]);

    let gn = trace_to_graph_model(&graph)
        .expect("Narrow translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 4], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.len(), 2, "output should have 2 elements");
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(
            l >= -1.0 - 1e-5,
            "narrow lower bound should be >= -1, got {l}"
        );
        assert!(
            u <= 1.0 + 1e-5,
            "narrow upper bound should be <= 1, got {u}"
        );
    }
}

// -- Narrow CROWN -------------------------------------------------------------

#[test]
fn test_model_trace_narrow_crown() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.narrow(1, 1, 2)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[1, 4], 1.0);

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&output);

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for Narrow/Slice. Error: {crown_err:?}"
    );
}

// -- Unsqueeze IBP ------------------------------------------------------------

#[test]
fn test_model_trace_unsqueeze_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.unsqueeze(0)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 2, 3]);

    let gn = trace_to_graph_model(&graph)
        .expect("Unsqueeze translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -1.0 - 1e-5, "unsqueeze lower bound changed: {l}");
        assert!(u <= 1.0 + 1e-5, "unsqueeze upper bound changed: {u}");
        assert!(
            l <= -1.0 + 0.01,
            "unsqueeze should preserve lower bound, got {l}"
        );
        assert!(
            u >= 1.0 - 0.01,
            "unsqueeze should preserve upper bound, got {u}"
        );
    }
}

// -- Squeeze IBP --------------------------------------------------------------

#[test]
fn test_model_trace_squeeze_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3, 1], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.squeeze(2)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 3]);

    let gn = trace_to_graph_model(&graph)
        .expect("Squeeze translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3, 1], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -1.0 - 1e-5, "squeeze lower bound changed: {l}");
        assert!(u <= 1.0 + 1e-5, "squeeze upper bound changed: {u}");
    }
}

// -- Clamp IBP ----------------------------------------------------------------

#[test]
fn test_model_trace_clamp_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.clamp(0.0, 3.0)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 3]);

    let gn = trace_to_graph_model(&graph)
        .expect("Clamp translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 5.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -0.01, "clamped lower bound should be ~0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 3.01, "clamped upper bound should be ~3, got {v}");
    }
}

// -- Clamp CROWN --------------------------------------------------------------

#[test]
fn test_model_trace_clamp_crown() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.clamp(0.0, 3.0)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[2, 3], 5.0);

    let (_method, output, _crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&output);
}

// -- ReduceSum IBP ------------------------------------------------------------

#[test]
fn test_model_trace_reduce_sum_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.sum_keepdim(1)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 1]);

    let gn = trace_to_graph_model(&graph)
        .expect("ReduceSum translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -3.0 - 1e-5, "sum lower bound too low: {l}");
        assert!(u <= 3.0 + 1e-5, "sum upper bound too high: {u}");
    }
}

// -- ReduceMean IBP -----------------------------------------------------------

#[test]
fn test_model_trace_reduce_mean_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.mean_keepdim(1)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 1]);

    let gn = trace_to_graph_model(&graph)
        .expect("ReduceMean translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -1.0 - 1e-5, "mean lower bound too low: {l}");
        assert!(u <= 1.0 + 1e-5, "mean upper bound too high: {u}");
    }
}

// -- Cat IBP ------------------------------------------------------------------

#[test]
fn test_model_trace_cat_ibp() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let b = DynTensor::new(&[4.0, 5.0, 6.0], &[1, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let mut b = b.clone();
        let id_a = record_input(&[1, 3], DType::F32).unwrap();
        let id_b = record_input(&[1, 3], DType::F32).unwrap();
        a.set_trace_id(id_a);
        b.set_trace_id(id_b);
        let y = DynTensor::cat(&[&a, &b], 1)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 6]);

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("Cat translation")
        .graph;
    // Multi-input: two [1,3] inputs stacked = [6]
    let input_bounds = uniform_bounds(&[6], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -1.0 - 1e-5, "cat lower bound should be >= -1, got {l}");
        assert!(u <= 1.0 + 1e-5, "cat upper bound should be <= 1, got {u}");
    }
}

// -- Transpose CROWN ----------------------------------------------------------

#[test]
fn test_model_trace_transpose_crown() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.transpose(1, 2)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph).expect("translation").graph;
    let input_bounds = uniform_bounds(&[1, 2, 3], 1.0);

    let (method, output, crown_err) =
        propagate_with_crown_fallback(&gn, &input_bounds).expect("CROWN propagation");
    assert_bounds_valid(&output);

    assert_eq!(
        method,
        PropMethod::Crown,
        "CROWN should succeed for Transpose. Error: {crown_err:?}"
    );
}

// -- Permute IBP --------------------------------------------------------------

#[test]
fn test_model_trace_permute_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.permute([0, 2, 1])?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 3, 2]);

    let gn = trace_to_graph_model(&graph)
        .expect("Permute translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 2, 3], 1.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for (&l, &u) in lo.iter().zip(hi.iter()) {
        assert!(l >= -1.0 - 1e-5, "permute lower bound changed: {l}");
        assert!(u <= 1.0 + 1e-5, "permute upper bound changed: {u}");
    }
}
// -- LogSoftmax IBP -----------------------------------------------------------

#[test]
fn test_model_trace_log_softmax_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.log_softmax(1)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 3]);

    let gn = trace_to_graph_model(&graph)
        .expect("LogSoftmax translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, _hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v < 0.0,
            "log_softmax lower bound should be negative, got {v}"
        );
        assert!(
            v >= -20.0,
            "log_softmax lower bound is unexpectedly low: {v}"
        );
    }
}

// -- Expand: full trace → verify pipeline (Part of #2329) ---------------------

#[test]
fn test_model_trace_expand_ibp() {
    // [1, 4] → [3, 4]: expand dim 0 from 1 to 3.
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.expand([3, 4])?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[3, 4]);

    let gn = trace_to_graph_model(&graph)
        .expect("Expand translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 4], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Expand duplicates elements — each output element has same bounds as source.
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v >= -2.0 - 1e-6,
            "expand lower bound should be >= -2.0, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v <= 2.0 + 1e-6,
            "expand upper bound should be <= 2.0, got {v}"
        );
    }
}

#[test]
fn test_model_trace_expand_both_paths_accept() {
    // Round-trip: same graph through BOTH compile and verify paths.
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.expand([3, 4])?;
        Ok(y)
    })
    .unwrap();

    // Verify path: trace_to_graph_model
    let gn = trace_to_graph_model(&graph)
        .expect("Expand: verify path should succeed")
        .graph;
    assert!(
        gn.num_nodes() >= 2,
        "graph should have at least input + expand nodes"
    );

    // Compile path: compile_trace
    let steps = nn_dsl::compile_trace(&graph).expect("Expand: compile path should succeed");
    assert!(
        !steps.is_empty(),
        "compile_trace should produce dispatch steps"
    );
}

// -- RepeatInterleave: compile path round-trip (Part of #2329) ----------------
//
// The verify-path IBP test lives in trace_to_graph_model_decompose_ops.rs
// (test_repeat_interleave_verify_path_ibp). Constant node fix: #2399.

#[test]
fn test_model_trace_repeat_interleave_compile_path() {
    // Verify the compile path accepts a RepeatInterleave trace graph.
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let repeats = DynTensor::full(&[3], 2.0, DType::F32, &cpu())?;
        let y = x.repeat_interleave(1, &repeats)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 6]);

    let steps =
        nn_dsl::compile_trace(&graph).expect("RepeatInterleave: compile path should succeed");
    assert!(
        !steps.is_empty(),
        "compile_trace should produce dispatch steps"
    );
}
