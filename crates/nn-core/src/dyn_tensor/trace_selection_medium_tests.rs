// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trace recording tests for selection/indexing ops (#2347 HIGH severity)
//! and raw conv_transpose / upsample ops (#2347 MEDIUM severity).

use super::*;
use crate::dyn_tensor::DynTensor;
use crate::{DType, Device};

fn cpu() -> Device {
    Device::Cpu
}

// -- HIGH severity: topk (#2347) ----------------------------------------------

#[test]
fn test_trace_topk() {
    let a = DynTensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0], &[1, 5], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[1, 5], DType::F32).unwrap();
        a.set_trace_id(id);
        let (values, _indices) = a.topk(1, 2)?;
        Ok(values)
    })
    .unwrap();

    assert_eq!(result.shape().dims(), &[1, 2]);
    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2); // input + topk
    assert!(matches!(nodes[1].op(), TraceOp::Topk { k: 2, dim: 1 }));
}

// -- HIGH severity: argmax (#2347) --------------------------------------------

#[test]
fn test_trace_argmax() {
    let a = DynTensor::new(&[1.0, 3.0, 2.0], &[1, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[1, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.argmax(1)?;
        Ok(b)
    })
    .unwrap();

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::Argmax { dim: 1 }));
}

// -- HIGH severity: triu (#2347) ----------------------------------------------

#[test]
fn test_trace_triu() {
    let a = DynTensor::new(&[1.0; 9], &[3, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.triu(0)?;
        Ok(b)
    })
    .unwrap();

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::Triu { diagonal: 0 }));
}

// -- HIGH severity: tril (#2347) ----------------------------------------------

#[test]
fn test_trace_tril() {
    let a = DynTensor::new(&[1.0; 9], &[3, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3, 3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.tril(0)?;
        Ok(b)
    })
    .unwrap();

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::Tril { diagonal: 0 }));
}

// -- HIGH severity: compare eq (#2347) ----------------------------------------

#[test]
fn test_trace_compare_eq() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0], &[3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.eq(2.0)?;
        Ok(b)
    })
    .unwrap();

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::Compare { .. }));
}

// -- HIGH severity: arg_sort (#2347) ------------------------------------------

#[test]
fn test_trace_arg_sort() {
    let a = DynTensor::new(&[3.0, 1.0, 2.0], &[3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[3], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.arg_sort(0, true)?;
        Ok(b)
    })
    .unwrap();

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(
        nodes[1].op(),
        TraceOp::ArgSort {
            dim: 0,
            descending: false
        }
    ));
}

// -- MEDIUM severity: upsample_nearest_2d (#2347) -----------------------------

#[test]
fn test_trace_upsample_nearest_2d() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.upsample_nearest_2d(2, 2)?;
        Ok(b)
    })
    .unwrap();

    assert_eq!(result.shape().dims(), &[1, 1, 4, 4]);
    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::Upsample2d { .. }));
}

// -- MEDIUM severity: upsample_bilinear_2d (#2347) ----------------------------

#[test]
fn test_trace_upsample_bilinear_2d() {
    let a = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut a = a.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        a.set_trace_id(id);
        let b = a.upsample_bilinear_2d(2.0, 2.0, false)?;
        Ok(b)
    })
    .unwrap();

    assert_eq!(result.shape().dims(), &[1, 1, 4, 4]);
    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::Upsample2d { .. }));
}

// -- MEDIUM severity: conv_transpose1d raw method (#2347) ---------------------

#[test]
fn test_trace_conv_transpose1d_raw() {
    // [batch=1, in_ch=1, len=3], kernel [in_ch=1, out_ch=1, k=2]
    let input = DynTensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3], &cpu()).unwrap();
    let kernel = DynTensor::new(&[1.0, 1.0], &[1, 1, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut input = input.clone();
        let id = record_input(&[1, 1, 3], DType::F32).unwrap();
        input.set_trace_id(id);
        let out = input.conv_transpose1d(&kernel, 0, 0, 1, 1, 1)?;
        Ok(out)
    })
    .unwrap();

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::ConvTranspose1d { .. }));
}

// -- MEDIUM severity: conv_transpose2d raw method (#2347) ---------------------

#[test]
fn test_trace_conv_transpose2d_raw() {
    // [batch=1, in_ch=1, h=2, w=2], kernel [in_ch=1, out_ch=1, kh=2, kw=2]
    let input = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();
    let kernel = DynTensor::new(&[1.0; 4], &[1, 1, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut input = input.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        input.set_trace_id(id);
        let out = input.conv_transpose2d(&kernel, [0, 0], [0, 0], [1, 1], [1, 1], 1)?;
        Ok(out)
    })
    .unwrap();

    let nodes = graph.nodes();
    assert_eq!(nodes.len(), 2);
    assert!(matches!(nodes[1].op(), TraceOp::ConvTranspose2d { .. }));
}
