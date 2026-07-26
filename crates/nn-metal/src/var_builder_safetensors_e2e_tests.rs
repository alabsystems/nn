#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! E2E tests for mmap-based safetensors loading and model import pipeline.
//! Extracted from `var_builder_safetensors_tests.rs` (500-line limit).
//!
//! Silero VAD architecture test is in `var_builder_safetensors_e2e_silero_tests.rs`.

#[path = "var_builder_safetensors_e2e_silero_tests.rs"]
mod silero;

use std::path::Path;

use nn_core::{DType, Device, VarBuilder};

use crate::context::MetalContext;
use crate::var_builder_safetensors::{
    from_mmaped_safetensors, from_mmaped_safetensors_with_ctx, MetalVarBuilderExt,
};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_vb_st_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn create_single_tensor_file(path: &Path, name: &str, shape: &[usize], values: &[f32]) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let bytes = bytemuck::cast_slice::<f32, u8>(values);
    let view = TensorView::new(StDtype::F32, shape.to_vec(), bytes).expect("valid view");
    let data = serialize(vec![(name.to_string(), view)], None).expect("serialize");
    std::fs::write(path, data).expect("write safetensors");
}

fn create_multi_tensor_file(path: &Path, tensors: &[(&str, &[usize], &[f32])]) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, shape, values)| {
            let bytes = bytemuck::cast_slice::<f32, u8>(values);
            (
                name.to_string(),
                TensorView::new(StDtype::F32, shape.to_vec(), bytes).expect("valid view"),
            )
        })
        .collect();
    std::fs::write(path, serialize(views, None).expect("serialize")).expect("write");
}

#[test]
fn test_from_mmaped_safetensors_loads_and_reads() {
    let dir = temp_dir("mmaped_load");
    let path = dir.join("model.safetensors");
    create_multi_tensor_file(
        &path,
        &[
            ("encoder.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("encoder.bias", &[2], &[0.5, 0.6]),
        ],
    );

    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: Test file is not modified during the test.
    let vb = unsafe {
        from_mmaped_safetensors_with_ctx(&[path.as_path()], DType::F32, &Device::Cpu, &ctx)
            .expect("load should succeed")
    };

    let enc = vb.pp("encoder");
    let w = enc.get(&[2, 3], "weight").expect("load weight");
    assert_eq!(w.dims(), &[2, 3]);
    let data = w.to_flat_vec::<f32>().expect("readback");
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let b = enc.get(&[2], "bias").expect("load bias");
    let bias_data = b.to_flat_vec::<f32>().expect("readback");
    assert_eq!(bias_data, vec![0.5, 0.6]);

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_from_mmaped_safetensors_empty_paths_rejected() {
    let empty: &[&Path] = &[];
    // SAFETY: No file is opened (empty paths), so the mmap contract is trivially satisfied.
    let result = unsafe { from_mmaped_safetensors(empty, DType::F32, &Device::Cpu) };
    assert!(result.is_err(), "empty paths should be rejected");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("empty paths"),
        "error should mention empty paths, got: {err_msg}"
    );
}

#[test]
fn test_from_mmaped_safetensors_multi_file_loads_all_shards() {
    let dir = temp_dir("mmaped_multi");
    let p1 = dir.join("shard1.safetensors");
    let p2 = dir.join("shard2.safetensors");
    create_single_tensor_file(&p1, "encoder.weight", &[2], &[1.0, 2.0]);
    create_single_tensor_file(&p2, "decoder.weight", &[3], &[3.0, 4.0, 5.0]);

    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: Test files are not modified during the test.
    let vb = unsafe {
        from_mmaped_safetensors_with_ctx(
            &[p1.as_path(), p2.as_path()],
            DType::F32,
            &Device::Cpu,
            &ctx,
        )
        .expect("multi-file load should succeed")
    };

    // Tensor from shard 1
    let enc = vb.get(&[2], "encoder.weight").expect("load from shard 1");
    let enc_data = enc.to_flat_vec::<f32>().expect("readback");
    assert_eq!(enc_data, vec![1.0, 2.0]);

    // Tensor from shard 2
    let dec = vb.get(&[3], "decoder.weight").expect("load from shard 2");
    let dec_data = dec.to_flat_vec::<f32>().expect("readback");
    assert_eq!(dec_data, vec![3.0, 4.0, 5.0]);

    // Nonexistent tensor across all shards
    assert!(
        vb.get(&[1], "nonexistent").is_err(),
        "missing tensor should fail"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_from_mmaped_safetensors_nonexistent_file_rejected() {
    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: File does not exist, so the mmap contract is trivially satisfied
    // (load will fail before any mapping is created).
    let result = unsafe {
        from_mmaped_safetensors_with_ctx(
            &[std::env::temp_dir().join("definitely_does_not_exist_nn_test.safetensors")],
            DType::F32,
            &Device::Cpu,
            &ctx,
        )
    };
    assert!(result.is_err(), "nonexistent file should be rejected");
}

#[test]
fn test_from_mmaped_safetensors_constructs_linear() {
    let dir = temp_dir("mmaped_linear");
    let path = dir.join("model.safetensors");
    // Linear layer: weight [4, 3] + bias [4]
    create_multi_tensor_file(
        &path,
        &[
            (
                "linear.weight",
                &[4, 3],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ),
            ("linear.bias", &[4], &[0.1, 0.2, 0.3, 0.4]),
        ],
    );

    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: Test file is not modified during the test.
    let vb = unsafe {
        from_mmaped_safetensors_with_ctx(&[path.as_path()], DType::F32, &Device::Cpu, &ctx)
            .expect("load should succeed")
    };

    let lin_vb = vb.pp("linear");
    let w = lin_vb.get(&[4, 3], "weight").expect("load weight");
    let b = lin_vb.get(&[4], "bias").expect("load bias");

    // Construct Linear layer and run forward pass
    let linear = nn_core::layers::Linear::new(w, Some(b)).unwrap();
    let input =
        nn_core::DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &Device::Cpu).expect("input");
    let output = nn_core::layers::Module::forward(&linear, &input).expect("forward");
    assert_eq!(output.dims(), &[1, 4]);
    let out_data = output.to_flat_vec::<f32>().expect("readback");
    // Expected: matmul([1,2,3], W^T) + bias
    // row0: 1*1+2*0+3*0 + 0.1 = 1.1
    // row1: 1*0+2*1+3*0 + 0.2 = 2.2
    // row2: 1*0+2*0+3*1 + 0.3 = 3.3
    // row3: 1*1+2*1+3*1 + 0.4 = 6.4
    assert!((out_data[0] - 1.1).abs() < 1e-5, "got {}", out_data[0]);
    assert!((out_data[1] - 2.2).abs() < 1e-5, "got {}", out_data[1]);
    assert!((out_data[2] - 3.3).abs() < 1e-5, "got {}", out_data[2]);
    assert!((out_data[3] - 6.4).abs() < 1e-5, "got {}", out_data[3]);

    std::fs::remove_dir_all(&dir).ok();
}

/// Demonstrates the full PyTorch model import pipeline:
///   safetensors file → VarBuilder → layers::Linear layers → forward pass
///
/// This is the pattern dvoice uses for candle→nn migration:
///   1. Convert PyTorch weights to safetensors (Python converter script)
///   2. Load via `from_mmaped_safetensors` into a `VarBuilder`
///   3. Construct nn layers using `Layer::load()` / free functions
///   4. Run inference via `Module::forward()`
#[test]
fn test_e2e_safetensors_to_inference_pipeline() {
    let dir = temp_dir("e2e_pipeline");
    let path = dir.join("two_layer_model.safetensors");

    // Simulate a 2-layer MLP: Linear(3→4) → Linear(4→2)
    // Weight names follow PyTorch convention: "layer.weight", "layer.bias"
    create_multi_tensor_file(
        &path,
        &[
            // Layer 1: W=[4,3], b=[4] (identity-ish + ones row)
            (
                "layer1.weight",
                &[4, 3],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            ),
            ("layer1.bias", &[4], &[0.0, 0.0, 0.0, 0.0]),
            // Layer 2: W=[2,4], b=[2]
            (
                "layer2.weight",
                &[2, 4],
                &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ),
            ("layer2.bias", &[2], &[0.5, -0.5]),
        ],
    );

    let ctx = MetalContext::new().expect("Metal context");
    // Step 1: Load safetensors → VarBuilder
    // SAFETY: Test file is not modified during the test.
    let vb = unsafe {
        from_mmaped_safetensors_with_ctx(&[path.as_path()], DType::F32, &Device::Cpu, &ctx)
            .expect("load safetensors")
    };

    // Step 2: Construct nn layers using candle-compatible free functions
    let linear1 = nn_core::layers::linear(3, 4, vb.pp("layer1")).expect("load layer1");
    let linear2 = nn_core::layers::linear(4, 2, vb.pp("layer2")).expect("load layer2");

    // Step 3: Run inference
    let input =
        nn_core::DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &Device::Cpu).expect("input");
    let hidden = nn_core::layers::Module::forward(&linear1, &input).expect("layer1 forward");
    assert_eq!(hidden.dims(), &[1, 4]);
    let output = nn_core::layers::Module::forward(&linear2, &hidden).expect("layer2 forward");
    assert_eq!(output.dims(), &[1, 2]);

    let out_data = output.to_flat_vec::<f32>().expect("readback");
    // Layer1: [1,2,3] @ W1^T = [1, 2, 3, 6]
    // Layer2: [1,2,3,6] @ W2^T + b = [1+2+0.5, 3+6-0.5] = [3.5, 8.5]
    assert!(
        (out_data[0] - 3.5).abs() < 1e-5,
        "output[0]: expected 3.5, got {}",
        out_data[0]
    );
    assert!(
        (out_data[1] - 8.5).abs() < 1e-5,
        "output[1]: expected 8.5, got {}",
        out_data[1]
    );

    std::fs::remove_dir_all(&dir).ok();
}

/// Verify the candle-compatible associated function syntax works:
///   `VarBuilder::from_mmaped_safetensors(&[path], dtype, &device)`
#[test]
fn test_ext_from_mmaped_safetensors_single_file() {
    let dir = temp_dir("ext_single");
    let path = dir.join("model.safetensors");
    create_multi_tensor_file(
        &path,
        &[
            ("encoder.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("encoder.bias", &[2], &[0.5, 0.6]),
        ],
    );

    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: Test file is not modified during the test.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors_with_ctx(
            &[path.as_path()],
            DType::F32,
            &Device::Cpu,
            &ctx,
        )
        .expect("extension trait load should succeed")
    };

    let w = vb
        .pp("encoder")
        .get(&[2, 3], "weight")
        .expect("load weight");
    assert_eq!(w.dims(), &[2, 3]);
    let data = w.to_flat_vec::<f32>().expect("readback");
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    std::fs::remove_dir_all(&dir).ok();
}

/// Verify multi-file (sharded) loading works through the extension trait.
#[test]
fn test_ext_from_mmaped_safetensors_multi_file() {
    let dir = temp_dir("ext_multi");
    let p1 = dir.join("shard1.safetensors");
    let p2 = dir.join("shard2.safetensors");
    create_single_tensor_file(&p1, "layer1.weight", &[3], &[1.0, 2.0, 3.0]);
    create_single_tensor_file(&p2, "layer2.weight", &[2], &[4.0, 5.0]);

    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: Test files are not modified during the test.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors_with_ctx(
            &[p1.as_path(), p2.as_path()],
            DType::F32,
            &Device::Cpu,
            &ctx,
        )
        .expect("multi-file extension trait load should succeed")
    };

    // Shard 1
    let t1 = vb.get(&[3], "layer1.weight").expect("shard 1");
    assert_eq!(
        t1.to_flat_vec::<f32>().expect("readback"),
        vec![1.0, 2.0, 3.0]
    );

    // Shard 2
    let t2 = vb.get(&[2], "layer2.weight").expect("shard 2");
    assert_eq!(t2.to_flat_vec::<f32>().expect("readback"), vec![4.0, 5.0]);

    std::fs::remove_dir_all(&dir).ok();
}

/// Verify empty paths are rejected through the extension trait.
#[test]
fn test_ext_from_mmaped_safetensors_empty_paths_rejected() {
    let empty: &[&Path] = &[];
    // SAFETY: No file is opened (empty paths), so the mmap contract is trivially satisfied.
    let result = unsafe { VarBuilder::from_mmaped_safetensors(empty, DType::F32, &Device::Cpu) };
    assert!(result.is_err(), "empty paths should be rejected");
}
