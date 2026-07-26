// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for lstm() free function and BiLstm::load().
//! Split from nn_var_builder_tests_free_fns.rs for 500-line limit.

use super::super::super::*;
use crate::dyn_tensor::test_helpers::cpu;
use crate::layers::BiLstm;
use crate::var_builder::VarBuilder;
use crate::{DType, DynTensor};
use std::collections::HashMap;

fn map_vb(tensors: HashMap<String, DynTensor>) -> VarBuilder {
    VarBuilder::from_tensors(tensors, DType::F32, &cpu())
}

// -- lstm() free function -----------------------------------------------------

#[test]
fn test_lstm_fn_loads_weights() {
    let input_size = 3;
    let hidden_size = 2;
    let four_h = 4 * hidden_size;
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight_ih_l0".into(),
        DynTensor::zeros(&[four_h, input_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "weight_hh_l0".into(),
        DynTensor::zeros(&[four_h, hidden_size], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let l = lstm(input_size, hidden_size, &vb).unwrap();
    assert_eq!(l.hidden_size(), hidden_size);
}

#[test]
fn test_lstm_fn_with_bias() {
    let input_size = 4;
    let hidden_size = 3;
    let four_h = 4 * hidden_size;
    let mut tensors = HashMap::new();
    tensors.insert(
        "weight_ih_l0".into(),
        DynTensor::zeros(&[four_h, input_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "weight_hh_l0".into(),
        DynTensor::zeros(&[four_h, hidden_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias_ih_l0".into(),
        DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "bias_hh_l0".into(),
        DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let l = lstm(input_size, hidden_size, &vb).unwrap();
    // Forward pass with zero weights: gates all sigmoid(0)=0.5, tanh(0)=0
    let x = DynTensor::ones(&[1, input_size], DType::F32, &cpu()).unwrap();
    let (out, _state) = l.forward(&x, None).unwrap();
    assert_eq!(out.dims(), &[1, hidden_size]);
}

// -- BiLstm::load() -----------------------------------------------------------

#[test]
fn test_bilstm_load_weights() {
    let input_size = 3;
    let hidden_size = 2;
    let four_h = 4 * hidden_size;
    let mut tensors = HashMap::new();
    // Forward direction
    tensors.insert(
        "weight_ih_l0".into(),
        DynTensor::zeros(&[four_h, input_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "weight_hh_l0".into(),
        DynTensor::zeros(&[four_h, hidden_size], DType::F32, &cpu()).unwrap(),
    );
    // Backward direction
    tensors.insert(
        "weight_ih_l0_reverse".into(),
        DynTensor::zeros(&[four_h, input_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "weight_hh_l0_reverse".into(),
        DynTensor::zeros(&[four_h, hidden_size], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let bi = BiLstm::load(&vb, input_size, hidden_size).unwrap();
    assert_eq!(bi.hidden_size(), hidden_size);
}

#[test]
fn test_bilstm_load_with_bias() {
    let input_size = 4;
    let hidden_size = 2;
    let four_h = 4 * hidden_size;
    let mut tensors = HashMap::new();
    for suffix in &["", "_reverse"] {
        tensors.insert(
            format!("weight_ih_l0{suffix}"),
            DynTensor::zeros(&[four_h, input_size], DType::F32, &cpu()).unwrap(),
        );
        tensors.insert(
            format!("weight_hh_l0{suffix}"),
            DynTensor::zeros(&[four_h, hidden_size], DType::F32, &cpu()).unwrap(),
        );
        tensors.insert(
            format!("bias_ih_l0{suffix}"),
            DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
        );
        tensors.insert(
            format!("bias_hh_l0{suffix}"),
            DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
        );
    }
    let vb = map_vb(tensors);
    let bi = BiLstm::load(&vb, input_size, hidden_size).unwrap();
    // Forward pass over a 2-step sequence
    let x = DynTensor::ones(&[2, 1, input_size], DType::F32, &cpu()).unwrap();
    let (out, _fwd, _bwd) = bi.forward_seq(&x, None, None).unwrap();
    // Output is [seq=2, batch=1, 2*hidden=4]
    assert_eq!(out.dims(), &[2, 1, 2 * hidden_size]);
}

/// Verify BiLstm::load() accepts dvoice decomposed naming (#2741).
#[test]
fn test_bilstm_load_decomposed_naming() {
    let input_size = 3;
    let hidden_size = 2;
    let four_h = 4 * hidden_size;
    let mut tensors = HashMap::new();
    // Forward direction: decomposed naming
    tensors.insert(
        "forward.weight_ih.weight".into(),
        DynTensor::zeros(&[four_h, input_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "forward.weight_hh.weight".into(),
        DynTensor::zeros(&[four_h, hidden_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "forward.weight_ih.bias".into(),
        DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "forward.weight_hh.bias".into(),
        DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
    );
    // Backward direction: decomposed naming
    tensors.insert(
        "backward.weight_ih.weight".into(),
        DynTensor::zeros(&[four_h, input_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "backward.weight_hh.weight".into(),
        DynTensor::zeros(&[four_h, hidden_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "backward.weight_ih.bias".into(),
        DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "backward.weight_hh.bias".into(),
        DynTensor::zeros(&[four_h], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    let bi = BiLstm::load(&vb, input_size, hidden_size).unwrap();
    assert_eq!(bi.hidden_size(), hidden_size);
    // Verify forward pass works
    let x = DynTensor::ones(&[2, 1, input_size], DType::F32, &cpu()).unwrap();
    let (out, _fwd, _bwd) = bi.forward_seq(&x, None, None).unwrap();
    assert_eq!(out.dims(), &[2, 1, 2 * hidden_size]);
}

#[test]
fn test_bilstm_load_missing_reverse_fails() {
    let input_size = 3;
    let hidden_size = 2;
    let four_h = 4 * hidden_size;
    let mut tensors = HashMap::new();
    // Forward only — missing reverse weights
    tensors.insert(
        "weight_ih_l0".into(),
        DynTensor::zeros(&[four_h, input_size], DType::F32, &cpu()).unwrap(),
    );
    tensors.insert(
        "weight_hh_l0".into(),
        DynTensor::zeros(&[four_h, hidden_size], DType::F32, &cpu()).unwrap(),
    );
    let vb = map_vb(tensors);
    assert!(BiLstm::load(&vb, input_size, hidden_size).is_err());
}
