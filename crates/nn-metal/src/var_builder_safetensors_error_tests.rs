// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error path tests for SafeTensorsBackend (#1082).
//!
//! Covers untested error paths in `load_tensor_from_weight_map()`:
//! - F32 NaN/Inf rejection (NonFiniteData guard)
//! - F16 NaN rejection
//! - Unsupported stored dtype (I64 → DTypeMismatch)
//! - Requested dtype != F32 (DTypeMismatch)
//! - Data length mismatch (via direct call)
//! - Shape overflow (via direct call)

use std::io::Write;
use std::path::Path;

use nn_core::var_builder::TensorBackend;
use nn_core::{DType, Device, TensorError};

use crate::context::MetalContext;
use crate::safetensors::WeightMap;
use crate::var_builder_safetensors::{load_tensor_from_weight_map, SafeTensorsBackend};

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("nn_vb_err_{name}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn load_weight_map(path: &Path) -> WeightMap {
    let ctx = MetalContext::new().expect("Metal context");
    // SAFETY: Test file is written by the test and not modified during loading.
    unsafe { WeightMap::load(path, &ctx).expect("load weight map") }
}

fn create_raw_tensor_file(
    path: &Path,
    name: &str,
    shape: &[usize],
    dtype: safetensors::Dtype,
    bytes: &[u8],
) {
    use safetensors::tensor::{serialize, TensorView};
    let view = TensorView::new(dtype, shape.to_vec(), bytes).expect("valid view");
    let tensors = vec![(name.to_string(), view)];
    let serialized = serialize(tensors, None).expect("serialize");
    let mut file = std::fs::File::create(path).expect("create file");
    file.write_all(&serialized).expect("write file");
}

// -- AC1: F32 NaN rejection ---------------------------------------------------

#[test]
fn test_get_f32_nan_rejected() {
    let dir = temp_dir("f32_nan");
    let path = dir.join("model.safetensors");
    let values = [1.0f32, f32::NAN, 3.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&values);
    create_raw_tensor_file(&path, "w", &[3], safetensors::Dtype::F32, bytes);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);
    let err = backend
        .get(&[3], "w", DType::F32, &Device::Cpu)
        .unwrap_err();
    assert!(
        matches!(err, TensorError::NonFiniteData { count: 1, .. }),
        "expected NonFiniteData count=1, got: {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_f32_inf_rejected() {
    let dir = temp_dir("f32_inf");
    let path = dir.join("model.safetensors");
    let values = [1.0f32, f32::INFINITY, f32::NEG_INFINITY, 4.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&values);
    create_raw_tensor_file(&path, "w", &[4], safetensors::Dtype::F32, bytes);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);
    let err = backend
        .get(&[4], "w", DType::F32, &Device::Cpu)
        .unwrap_err();
    assert!(
        matches!(err, TensorError::NonFiniteData { count: 2, .. }),
        "expected NonFiniteData count=2, got: {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_f16_nan_rejected() {
    let dir = temp_dir("f16_nan");
    let path = dir.join("model.safetensors");
    let f16_bytes: Vec<u8> = [
        half::f16::from_f32(1.0),
        half::f16::NAN,
        half::f16::from_f32(3.0),
    ]
    .iter()
    .flat_map(|v| v.to_le_bytes())
    .collect();
    create_raw_tensor_file(&path, "w", &[3], safetensors::Dtype::F16, &f16_bytes);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);
    let err = backend
        .get(&[3], "w", DType::F32, &Device::Cpu)
        .unwrap_err();
    assert!(
        matches!(err, TensorError::NonFiniteData { count: 1, .. }),
        "expected NonFiniteData count=1, got: {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// -- AC2: Unsupported stored dtype (I64) → DTypeMismatch ----------------------

#[test]
fn test_get_stored_i64_dtype_rejected() {
    let dir = temp_dir("dtype_i64");
    let path = dir.join("model.safetensors");
    let values = [1i64, 2, 3];
    let bytes = bytemuck::cast_slice::<i64, u8>(&values);
    create_raw_tensor_file(&path, "w", &[3], safetensors::Dtype::I64, bytes);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);
    let err = backend
        .get(&[3], "w", DType::F32, &Device::Cpu)
        .unwrap_err();
    assert!(
        matches!(err, TensorError::DTypeMismatch { .. }),
        "expected DTypeMismatch for I64, got: {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_get_requested_bf16_produces_bf16() {
    // After #1646: BF16 requested dtype produces native bf16 storage.
    let dir = temp_dir("req_bf16");
    let path = dir.join("model.safetensors");
    let values = [1.0f32, 2.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&values);
    create_raw_tensor_file(&path, "w", &[2], safetensors::Dtype::F32, bytes);

    let wm = load_weight_map(&path);
    let backend = SafeTensorsBackend::new(wm);
    let t = backend
        .get(&[2], "w", DType::BF16, &Device::Cpu)
        .expect("BF16 request should succeed");
    assert_eq!(t.dtype(), DType::BF16, "native bf16 storage (#1646)");
    assert_eq!(t.dims(), &[2]);
    std::fs::remove_dir_all(&dir).ok();
}

// -- AC3: Data length mismatch (direct call) ----------------------------------

#[test]
fn test_data_length_mismatch_via_direct_call() {
    let dir = temp_dir("data_len");
    let path = dir.join("model.safetensors");
    // Create a [2] f32 tensor (8 bytes of data)
    let values = [1.0f32, 2.0];
    let bytes = bytemuck::cast_slice::<f32, u8>(&values);
    create_raw_tensor_file(&path, "w", &[2], safetensors::Dtype::F32, bytes);

    let wm = load_weight_map(&path);
    // Call load_tensor_from_weight_map directly with wrong shape [4]
    // The stored data is 8 bytes but shape [4] expects 16 bytes for F32
    let err = load_tensor_from_weight_map(&wm, "w", &[4], DType::F32, DType::F32, &Device::Cpu)
        .unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::DataLengthMismatch {
                expected: 4,
                actual: 2
            }
        ),
        "expected DataLengthMismatch(4,2), got: {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

// -- AC4: Shape overflow (direct call) ----------------------------------------

#[test]
fn test_shape_overflow_via_direct_call() {
    let dir = temp_dir("shape_ovf");
    let path = dir.join("model.safetensors");
    let values = [1.0f32];
    let bytes = bytemuck::cast_slice::<f32, u8>(&values);
    create_raw_tensor_file(&path, "w", &[1], safetensors::Dtype::F32, bytes);

    let wm = load_weight_map(&path);
    // Shape [usize::MAX, 2] overflows checked_mul
    let err = load_tensor_from_weight_map(
        &wm,
        "w",
        &[usize::MAX, 2],
        DType::F32,
        DType::F32,
        &Device::Cpu,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            TensorError::InvalidShape(_) | TensorError::DimensionOverflow { .. }
        ),
        "expected InvalidShape or DimensionOverflow for overflow, got: {err:?}"
    );
    std::fs::remove_dir_all(&dir).ok();
}
