// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for safetensors zero-copy loading via nn-metal.
//!
//! # Safety
//!
//! `WeightMap::load` is `unsafe` because it creates an mmap-backed Metal buffer
//! from a file path. Callers must ensure: (1) the file is a valid safetensors
//! file, (2) the Metal context is initialized, (3) the file outlives the
//! WeightMap. All tests here create temp files via `create_test_safetensors` /
//! `create_multi_tensor_file` and clean up after assertions, satisfying these
//! preconditions.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use nn_core::{DType, Tensor};
use nn_metal::{MetalBackend, MetalContext, MetalTensorExt, WeightError, WeightMap};

/// Create a minimal safetensors file with a single f32 tensor.
fn create_test_safetensors(path: &Path, name: &str, values: &[f32]) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let bytes = bytemuck::cast_slice::<f32, u8>(values);
    let shape = vec![values.len()];
    let tensors = vec![(
        name.to_string(),
        TensorView::new(StDtype::F32, shape, bytes).unwrap(),
    )];
    let serialized = serialize(tensors, None).unwrap();
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&serialized).unwrap();
}

/// Create a safetensors file with multiple tensors of varying dtypes.
fn create_multi_tensor_file(path: &Path) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;

    let w1: Vec<f32> = vec![1.0, 2.0, 3.0];
    let w2: Vec<f32> = vec![4.0, 5.0];
    let b1 = bytemuck::cast_slice::<f32, u8>(&w1);
    let b2 = bytemuck::cast_slice::<f32, u8>(&w2);
    let tensors = vec![
        (
            "encoder.weight".to_string(),
            TensorView::new(StDtype::F32, vec![3], b1).unwrap(),
        ),
        (
            "encoder.bias".to_string(),
            TensorView::new(StDtype::F32, vec![2], b2).unwrap(),
        ),
    ];
    let serialized = serialize(tensors, None).unwrap();
    std::fs::write(path, &serialized).unwrap();
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_metal_st_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_load_single_tensor() {
    let dir = temp_dir("single");
    let path = dir.join("model.safetensors");
    create_test_safetensors(&path, "weights", &[1.0, 2.0, 3.0, 4.0]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    assert_eq!(wm.tensor_count(), 1);
    assert!(wm.total_bytes() > 0);

    let info = wm.tensor_info("weights").unwrap();
    assert_eq!(info.dtype, DType::F32);
    assert_eq!(info.shape, vec![4]);
    assert_eq!(info.numel().unwrap(), 4);
    assert_eq!(info.byte_len, 16); // 4 floats * 4 bytes

    // Read tensor data back via CPU-accessible shared memory.
    let data = wm.tensor_data("weights").unwrap();
    let readback: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(readback, &[1.0, 2.0, 3.0, 4.0]);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_multiple_tensors() {
    let dir = temp_dir("multi");
    let path = dir.join("multi.safetensors");
    create_multi_tensor_file(&path);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    assert_eq!(wm.tensor_count(), 2);

    let info_w = wm.tensor_info("encoder.weight").unwrap();
    assert_eq!(info_w.shape, vec![3]);
    assert_eq!(info_w.byte_len, 12);
    assert_eq!(info_w.dtype, DType::F32);

    let info_b = wm.tensor_info("encoder.bias").unwrap();
    assert_eq!(info_b.shape, vec![2]);
    assert_eq!(info_b.byte_len, 8);

    // Verify data integrity.
    let dw = wm.tensor_data("encoder.weight").unwrap();
    let readback_w: Vec<f32> = dw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(readback_w, &[1.0, 2.0, 3.0]);

    let db = wm.tensor_data("encoder.bias").unwrap();
    let readback_b: Vec<f32> = db
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(readback_b, &[4.0, 5.0]);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_tensor_not_found() {
    let dir = temp_dir("notfound");
    let path = dir.join("model.safetensors");
    create_test_safetensors(&path, "weights", &[1.0]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    let err = wm.tensor_info("nonexistent").unwrap_err();
    assert!(matches!(err, WeightError::TensorNotFound(_)));

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_tensor_names_iteration() {
    let dir = temp_dir("names");
    let path = dir.join("model.safetensors");
    create_multi_tensor_file(&path);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    let mut names: Vec<&str> = wm.tensor_names().collect();
    names.sort_unstable();
    assert_eq!(names, vec!["encoder.bias", "encoder.weight"]);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_buffer_is_page_aligned() {
    let dir = temp_dir("pagealign");
    let path = dir.join("model.safetensors");
    create_test_safetensors(&path, "w", &[1.0, 2.0, 3.0, 4.0, 5.0]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    // Metal buffer len should be page-aligned and >= file size.
    assert!(wm.buffer().len() >= wm.total_bytes());
    assert_eq!(wm.buffer().len() % 4096, 0);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_arc_sharing() {
    let dir = temp_dir("arc");
    let path = dir.join("shared.safetensors");
    create_test_safetensors(&path, "w", &[42.0]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = Arc::new(unsafe { WeightMap::load(&path, &ctx).unwrap() });

    let wm2 = Arc::clone(&wm);
    let wm3 = Arc::clone(&wm);

    assert_eq!(wm.tensor_count(), 1);
    assert_eq!(wm2.tensor_count(), 1);
    assert_eq!(wm3.tensor_count(), 1);
    assert_eq!(wm.buffer().len(), wm2.buffer().len());

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_cross_thread_read() {
    let dir = temp_dir("xthread");
    let path = dir.join("model.safetensors");
    create_test_safetensors(&path, "w", &[10.0, 20.0, 30.0]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = Arc::new(unsafe { WeightMap::load(&path, &ctx).unwrap() });

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let wm = Arc::clone(&wm);
            std::thread::spawn(move || {
                // autoreleasepool: Metal buffer reads may create ObjC temporaries.
                objc::rc::autoreleasepool(|| {
                    let data = wm.tensor_data("w").unwrap();
                    let readback: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    assert_eq!(readback, &[10.0, 20.0, 30.0]);
                    wm.tensor_count()
                })
            })
        })
        .collect();

    for h in handles {
        assert_eq!(h.join().unwrap(), 1);
    }

    std::fs::remove_dir_all(&dir).unwrap();
}

/// Compile-time assertion: WeightMap must be Send + Sync.
#[test]
fn test_weightmap_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WeightMap>();
}

/// Verify drop-order safety: WeightMap uses ManuallyDrop fields and an
/// explicit Drop impl that drops buffer before mmap (#522).
///
/// Previous approach checked field declaration order; now we verify the
/// structural `ManuallyDrop` + `impl Drop` pattern in source.
#[test]
fn test_drop_order_buffer_before_mmap() {
    let source = include_str!("../../src/safetensors.rs");

    // Both fields must be wrapped in ManuallyDrop.
    assert!(
        source.contains("buffer: ManuallyDrop<MetalBuffer>"),
        "buffer field must be ManuallyDrop<MetalBuffer>"
    );
    assert!(
        source.contains("mmap: ManuallyDrop<Mmap>"),
        "mmap field must be ManuallyDrop<Mmap>"
    );

    // Explicit Drop impl must drop buffer before mmap.
    let drop_buffer = source
        .find("ManuallyDrop::drop(&mut self.buffer)")
        .expect("Drop impl must call ManuallyDrop::drop on buffer");
    let drop_mmap = source
        .find("ManuallyDrop::drop(&mut self.mmap)")
        .expect("Drop impl must call ManuallyDrop::drop on mmap");
    assert!(
        drop_buffer < drop_mmap,
        "UNSOUND: buffer must be dropped before mmap in Drop impl. \
         buffer drop at byte {drop_buffer}, mmap drop at byte {drop_mmap}"
    );
}

// --- load_tensor (Direction 3: #748 AC4) ---

#[test]
fn test_load_tensor_f32() {
    let _backend = MetalBackend::init().expect("Metal init");
    let dir = temp_dir("load_tensor");
    let path = dir.join("model.safetensors");
    create_test_safetensors(&path, "weights", &[1.0, 2.0, 3.0, 4.0]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    let tensor: Tensor<1, f32, MetalBackend> = wm.load_tensor("weights", [4], &ctx).unwrap();
    assert_eq!(tensor.dims(), &[4]);
    assert_eq!(tensor.numel(), 4);

    // Verify data via to_cpu roundtrip.
    let cpu = tensor.to_cpu().unwrap();
    assert_eq!(cpu.as_ndarray().as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_tensor_shape_mismatch() {
    let dir = temp_dir("load_tensor_shape");
    let path = dir.join("model.safetensors");
    create_test_safetensors(&path, "w", &[1.0, 2.0, 3.0]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    // Stored shape is [3], but we request [2, 2]
    let result: Result<Tensor<2, f32, MetalBackend>, _> = wm.load_tensor("w", [2, 2], &ctx);
    assert!(result.is_err(), "shape mismatch should return error");
    let err = result.unwrap_err();
    assert!(
        matches!(err, WeightError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {err:?}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_tensor_dtype_mismatch() {
    let dir = temp_dir("load_tensor_dtype");
    let path = dir.join("model.safetensors");
    create_test_safetensors(&path, "w", &[1.0_f32]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    // Stored dtype is F32, but we request i32
    let result: Result<Tensor<1, i32, MetalBackend>, _> = wm.load_tensor("w", [1], &ctx);
    assert!(result.is_err(), "dtype mismatch should return error");
    let err = result.unwrap_err();
    assert!(
        matches!(err, WeightError::DtypeMismatch { .. }),
        "expected DtypeMismatch, got: {err:?}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_tensor_not_found() {
    let dir = temp_dir("load_tensor_notfound");
    let path = dir.join("model.safetensors");
    create_test_safetensors(&path, "w", &[1.0]);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    let result: Result<Tensor<1, f32, MetalBackend>, _> = wm.load_tensor("nonexistent", [1], &ctx);
    assert!(result.is_err(), "missing tensor should return error");
    let err = result.unwrap_err();
    assert!(
        matches!(err, WeightError::TensorNotFound(_)),
        "expected TensorNotFound, got: {err:?}"
    );

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn test_load_tensor_multi_tensor_file() {
    let _backend = MetalBackend::init().expect("Metal init");
    let dir = temp_dir("load_tensor_multi");
    let path = dir.join("multi.safetensors");
    create_multi_tensor_file(&path);

    let ctx = MetalContext::new().unwrap();
    // SAFETY: see module-level safety documentation.
    let wm = unsafe { WeightMap::load(&path, &ctx).unwrap() };

    // Load weight tensor [3]
    let weight: Tensor<1, f32, MetalBackend> = wm.load_tensor("encoder.weight", [3], &ctx).unwrap();
    let cpu_w = weight.to_cpu().unwrap();
    assert_eq!(cpu_w.as_ndarray().as_slice().unwrap(), &[1.0, 2.0, 3.0]);

    // Load bias tensor [2]
    let bias: Tensor<1, f32, MetalBackend> = wm.load_tensor("encoder.bias", [2], &ctx).unwrap();
    let cpu_b = bias.to_cpu().unwrap();
    assert_eq!(cpu_b.as_ndarray().as_slice().unwrap(), &[4.0, 5.0]);

    std::fs::remove_dir_all(&dir).unwrap();
}
