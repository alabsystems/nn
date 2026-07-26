// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pooling, reduction, and concatenation convert tests.
//!
//! Exercises the dpdf-critical ops that appear in DocLayout-YOLO and
//! Granite-Docling model conversion:
//!
//! - MaxPool2d with kernel=3, stride=1, padding=1 (SPPF pattern in YOLO)
//! - AdaptiveAvgPool2d to [1,1] (global average pooling in SigLIP2)
//! - Cat along channel dim (feature pyramid concatenation)
//! - Mean reduction along spatial dim (SigLIP2 mean pooling)
//! - Sum reduction (DFL head in YOLO)
//!
//! Part of dpdf model import coverage.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use nn_core::dyn_tensor::trace::TraceOp;

use crate::graph_build::ImportedGraph;
use crate::import_model;

static POOL_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ---------------------------------------------------------------------------
// SPPF-style fixture: Conv2d -> MaxPool2d -> MaxPool2d -> Cat -> AdaptiveAvgPool2d
// ---------------------------------------------------------------------------

/// Write synthetic conv weights for the pool/reduce/cat fixture.
///
/// Conv2d [8, 3, 3, 3] = 216 elements, bias [8]
fn write_pool_reduce_cat_weights(dir: &Path) -> std::path::PathBuf {
    let mut tensors = HashMap::new();

    let conv_w: Vec<u8> = (0..216)
        .flat_map(|i| ((i as f32) * 0.001).to_le_bytes())
        .collect();
    let conv_b: Vec<u8> = [0.0f32; 8].iter().flat_map(|f| f.to_le_bytes()).collect();

    tensors.insert(
        "conv.weight".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8, 3, 3, 3], &conv_w)
            .unwrap(),
    );
    tensors.insert(
        "conv.bias".to_string(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![8], &conv_b).unwrap(),
    );

    let weights_path = dir.join("weights.safetensors");
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();
    weights_path
}

/// Import the pool/reduce/cat fixture from disk.
fn import_pool_reduce_cat_fixture() -> ImportedGraph {
    let id = POOL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nn_import_pool_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/pool_reduce_cat.json"),
    )
    .unwrap();
    let weights_path = write_pool_reduce_cat_weights(&dir);
    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

// ---------------------------------------------------------------------------
// Graph structure tests (no Metal required)
// ---------------------------------------------------------------------------

/// E2E: SPPF-style graph imports with correct structure.
///
/// Conv2d -> MaxPool2d(k=3,s=1,p=1) -> MaxPool2d(k=3,s=1,p=1) ->
/// Cat(dim=1, 3 tensors) -> AdaptiveAvgPool2d([1,1]) -> View
#[test]
fn test_import_pool_reduce_cat_structure() {
    let imported = import_pool_reduce_cat_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["output"]);

    // 1 Input + 2 params + 6 compute ops = 9 total nodes.
    assert_eq!(imported.graph.len(), 9);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::Reshape { .. }),
        "expected Reshape (view) as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 24]);
}

/// E2E: all pool/reduce/cat ops map to correct TraceOp variants.
#[test]
fn test_import_pool_reduce_cat_op_counts() {
    let imported = import_pool_reduce_cat_fixture();
    let nodes = imported.graph.nodes();
    let count = |pred: fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    assert_eq!(count(|op| matches!(op, TraceOp::Conv2d { .. })), 1);
    assert_eq!(count(|op| matches!(op, TraceOp::MaxPool2d { .. })), 2);
    assert_eq!(count(|op| matches!(op, TraceOp::Cat { .. })), 1);
    assert_eq!(
        count(|op| matches!(op, TraceOp::AdaptiveAvgPool2d { .. })),
        1
    );
    assert_eq!(count(|op| matches!(op, TraceOp::Reshape { .. })), 1);
}

/// E2E: MaxPool2d parameters (kernel=3, stride=1, padding=1) survive import (SPPF pattern).
#[test]
fn test_import_pool_maxpool2d_params() {
    let imported = import_pool_reduce_cat_fixture();
    let nodes = imported.graph.nodes();

    let pool_nodes: Vec<_> = nodes
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::MaxPool2d { .. }))
        .collect();
    assert_eq!(pool_nodes.len(), 2);

    for pool in &pool_nodes {
        if let TraceOp::MaxPool2d {
            kernel_size,
            stride,
            padding,
        } = pool.op()
        {
            assert_eq!(*kernel_size, [3, 3], "SPPF uses kernel_size=3");
            assert_eq!(*stride, [1, 1], "SPPF uses stride=1 (no downsampling)");
            assert_eq!(*padding, [1, 1], "SPPF uses padding=1 (same spatial)");
        }
    }
}

/// E2E: Cat has correct dim and num_inputs for 3-way SPPF concatenation.
#[test]
fn test_import_pool_cat_params() {
    let imported = import_pool_reduce_cat_fixture();
    let nodes = imported.graph.nodes();

    let cat_node = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Cat { .. }))
        .unwrap();

    if let TraceOp::Cat { dim, num_inputs } = cat_node.op() {
        assert_eq!(*dim, 1, "cat along channel dim");
        assert_eq!(*num_inputs, 3, "SPPF concatenates conv_out + pool1 + pool2");
    }
    // Cat output: 3 * 8 channels = 24 channels, spatial preserved.
    assert_eq!(cat_node.output_shape(), &[1, 24, 8, 8]);
}

/// E2E: AdaptiveAvgPool2d has correct output_size [1,1] (global average pooling).
#[test]
fn test_import_pool_adaptive_avg_pool2d_params() {
    let imported = import_pool_reduce_cat_fixture();
    let nodes = imported.graph.nodes();

    let gap_node = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::AdaptiveAvgPool2d { .. }))
        .unwrap();

    if let TraceOp::AdaptiveAvgPool2d { output_size } = gap_node.op() {
        assert_eq!(*output_size, [1, 1], "global average pooling to [1,1]");
    }
    assert_eq!(gap_node.output_shape(), &[1, 24, 1, 1]);
}

/// E2E: intermediate shapes propagate correctly through SPPF pipeline.
///
/// Input [1,3,8,8] -> Conv2d(pad=1) -> [1,8,8,8]
/// -> MaxPool2d(k=3,s=1,p=1) -> [1,8,8,8] (same spatial)
/// -> MaxPool2d(k=3,s=1,p=1) -> [1,8,8,8]
/// -> Cat(dim=1) -> [1,24,8,8]
/// -> AdaptiveAvgPool2d([1,1]) -> [1,24,1,1]
/// -> View -> [1,24]
#[test]
fn test_import_pool_reduce_cat_shapes() {
    let imported = import_pool_reduce_cat_fixture();
    let nodes = imported.graph.nodes();

    let input = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input.output_shape(), &[1, 3, 8, 8]);

    let conv = nodes.iter().find(|n| n.name() == "conv_out").unwrap();
    assert_eq!(conv.output_shape(), &[1, 8, 8, 8]);

    // Both MaxPool2d nodes preserve spatial (stride=1, padding=1).
    let pool1 = nodes.iter().find(|n| n.name() == "pool1").unwrap();
    assert_eq!(pool1.output_shape(), &[1, 8, 8, 8]);

    let pool2 = nodes.iter().find(|n| n.name() == "pool2").unwrap();
    assert_eq!(pool2.output_shape(), &[1, 8, 8, 8]);

    // Cat: 8 + 8 + 8 = 24 channels.
    let cat = nodes.iter().find(|n| n.name() == "cat_out").unwrap();
    assert_eq!(cat.output_shape(), &[1, 24, 8, 8]);

    // AdaptiveAvgPool2d: spatial -> [1,1].
    let gap = nodes.iter().find(|n| n.name() == "gap_out").unwrap();
    assert_eq!(gap.output_shape(), &[1, 24, 1, 1]);

    // View: flatten to [1, 24].
    let output = nodes.iter().find(|n| n.name() == "output").unwrap();
    assert_eq!(output.output_shape(), &[1, 24]);
}

// ---------------------------------------------------------------------------
// Mean/Sum reduction fixture
// ---------------------------------------------------------------------------

/// Import the mean+sum reduction fixture (no weights needed).
fn import_mean_sum_reduce_fixture() -> ImportedGraph {
    let id = POOL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("nn_import_reduce_{}_{}", std::process::id(), id));
    std::fs::create_dir_all(&dir).unwrap();
    let graph_path = dir.join("graph.json");
    std::fs::write(
        &graph_path,
        include_str!("../test_data/mean_sum_reduce.json"),
    )
    .unwrap();
    // Write empty safetensors (no weights in this graph).
    let weights_path = dir.join("weights.safetensors");
    let tensors: HashMap<String, safetensors::tensor::TensorView<'_>> = HashMap::new();
    let serialized = safetensors::serialize(&tensors, None).unwrap();
    std::fs::write(&weights_path, serialized).unwrap();

    let imported = import_model(&graph_path, &weights_path).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    imported
}

/// E2E: mean+sum reduction graph imports with correct structure.
///
/// Input [1,4,8] -> mean(dim=2, keepdim=true) -> [1,4,1]
///               -> sum(dim=1, keepdim=false) -> [1,1]
#[test]
fn test_import_mean_sum_reduce_structure() {
    let imported = import_mean_sum_reduce_fixture();

    assert_eq!(imported.num_user_inputs, 1);
    assert_eq!(imported.user_input_names, vec!["x"]);
    assert_eq!(imported.output_names, vec!["output"]);

    // 1 Input + 0 params + 2 compute ops = 3 total nodes.
    assert_eq!(imported.graph.len(), 3);

    let output = imported.graph.output_node().unwrap();
    assert!(
        matches!(output.op(), TraceOp::ReduceSum { .. }),
        "expected ReduceSum as output, got: {:?}",
        output.op()
    );
    assert_eq!(output.output_shape(), &[1, 1]);
}

/// E2E: mean and sum reduction ops map to correct TraceOp variants with params.
#[test]
fn test_import_mean_sum_reduce_op_params() {
    let imported = import_mean_sum_reduce_fixture();
    let nodes = imported.graph.nodes();

    let mean_node = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::ReduceMean { .. }))
        .unwrap();
    if let TraceOp::ReduceMean { dim, keepdim } = mean_node.op() {
        assert_eq!(*dim, 2, "mean along dim 2 (spatial)");
        assert!(*keepdim, "keepdim=true for SigLIP2-style mean pooling");
    }
    assert_eq!(mean_node.output_shape(), &[1, 4, 1]);

    let sum_node = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::ReduceSum { .. }))
        .unwrap();
    if let TraceOp::ReduceSum { dim, keepdim } = sum_node.op() {
        assert_eq!(*dim, 1, "sum along dim 1 (DFL-style channel reduction)");
        assert!(!*keepdim, "keepdim=false for DFL sum");
    }
    assert_eq!(sum_node.output_shape(), &[1, 1]);
}

/// E2E: reduction shapes propagate correctly.
#[test]
fn test_import_mean_sum_reduce_shapes() {
    let imported = import_mean_sum_reduce_fixture();
    let nodes = imported.graph.nodes();

    let input = nodes
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .unwrap();
    assert_eq!(input.output_shape(), &[1, 4, 8]);

    let mean = nodes.iter().find(|n| n.name() == "mean_out").unwrap();
    assert_eq!(mean.output_shape(), &[1, 4, 1], "mean(dim=2, keepdim=true)");

    let sum = nodes.iter().find(|n| n.name() == "output").unwrap();
    assert_eq!(sum.output_shape(), &[1, 1], "sum(dim=1, keepdim=false)");
}

// ---------------------------------------------------------------------------
// Verification bridge tests (require `verify` feature)
// ---------------------------------------------------------------------------

/// E2E: SPPF-style graph -> NY IBP via shaped bounds.
///
/// Exercises Conv2d -> MaxPool2d -> Cat -> AdaptiveAvgPool2d through
/// the trace_to_graph -> IBP propagation pipeline. This validates that
/// pooling and concatenation ops translate correctly for verification.
#[test]
#[cfg(feature = "verify")]
fn test_import_pool_reduce_cat_ibp_bounds() {
    use ndarray::{ArrayD, IxDyn};

    let imported = import_pool_reduce_cat_fixture();

    let gn = nn_verify::trace_to_graph_model(&imported.graph)
        .expect("trace_to_graph_model must succeed for pool/reduce/cat")
        .graph;

    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .expect("must have Input node");
    let shape = input_node.output_shape();
    assert_eq!(shape, &[1, 3, 8, 8]);

    let lower = ArrayD::from_elem(IxDyn(shape), -1.0_f32);
    let upper = ArrayD::from_elem(IxDyn(shape), 1.0_f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper).expect("valid bounds");

    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP propagation must succeed for pool/reduce/cat graph");

    let (out_lo, out_hi) = output.lower_upper();
    let max_width = out_hi
        .iter()
        .zip(out_lo.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    assert!(
        max_width.is_finite() && max_width > 0.0,
        "pool/reduce/cat IBP bound width should be finite and positive, got {max_width}"
    );

    eprintln!(
        "[pool/reduce/cat IBP] {} output elements, max_width={max_width:.4}",
        out_lo.len()
    );
}

// ---------------------------------------------------------------------------
// Metal GPU execution tests (require `metal` feature + macOS)
// ---------------------------------------------------------------------------

/// Import -> CompiledModel -> execute_dyn on Metal GPU for the SPPF-style graph.
///
/// Validates MaxPool2d (stride=1) + Cat + AdaptiveAvgPool2d all compile and
/// execute correctly on Metal GPU.
#[test]
#[cfg(all(feature = "metal", target_os = "macos"))]
fn test_import_compile_execute_pool_reduce_cat_gpu() {
    use nn_core::{DType, Device};
    use nn_metal::compiled_model::CompiledModel;

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let imported = import_pool_reduce_cat_fixture();

    let compiled = CompiledModel::builder(&imported.graph, &cache)
        .build()
        .expect("compile pool/reduce/cat to Metal");

    assert_eq!(compiled.num_inputs(), 1);
    assert_eq!(compiled.output_shape(), &[1, 24]);
    assert_eq!(compiled.output_dtype(), DType::F32);

    let nd = compiled.num_dispatches();
    assert!(nd >= 1, "expected at least 1 dispatch, got {nd}");
    eprintln!(
        "[pool/reduce/cat GPU] steps={}, dispatches={nd}",
        compiled.num_steps()
    );

    // Create GPU input: [1, 3, 8, 8] with deterministic values.
    let input_size = 3 * 8 * 8;
    let input_data: Vec<f32> = (0..input_size).map(|i| (i as f32) * 0.01 - 0.5).collect();
    let input_cpu = nn_core::DynTensor::from_vec(input_data, &[1, 3, 8, 8], &Device::Cpu).unwrap();
    let input_gpu = input_cpu.to_device(&Device::metal()).unwrap();

    let output = compiled
        .execute_dyn(&cache, &[&input_gpu])
        .expect("GPU execution must succeed");

    assert_eq!(output.dims(), &[1, 24]);
    assert_eq!(output.dtype(), DType::F32);

    let output_cpu = output.to_device(&Device::Cpu).unwrap();
    let vals = output_cpu.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 24);
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "pool/reduce/cat GPU output[{i}] is not finite: {v}"
        );
    }

    let any_nonzero = vals.iter().any(|&v| v.abs() > 1e-10);
    assert!(
        any_nonzero,
        "GPU output is all zeros -- likely weight loading issue"
    );

    eprintln!(
        "[pool/reduce/cat GPU] output: {} elements, range=[{:.4}, {:.4}]",
        vals.len(),
        vals.iter().copied().fold(f32::INFINITY, f32::min),
        vals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    );
}
