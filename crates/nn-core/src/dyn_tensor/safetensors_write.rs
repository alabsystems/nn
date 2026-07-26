// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Safetensors serialization for [`DynTensor`].
//!
//! Writes named tensors to the safetensors format (same as PyTorch,
//! HuggingFace, candle). Uses little-endian byte conversion, not
//! `bytemuck::cast_slice`, to avoid alignment issues with mmap'd data.

use std::collections::HashMap;
use std::path::Path;

use crate::dyn_tensor::DynTensor;
use crate::{Device, Result, TensorError};

/// Convert f32 slice to little-endian bytes.
fn f32_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

/// Serialize named tensors to safetensors format and write to a file.
///
/// All tensors are converted to CPU f32 before serialization (matching
/// the DynTensor dtype/storage invariant: all float data is f32 internally).
///
/// # Errors
///
/// Returns an error if any tensor is on GPU and cannot be read, or if
/// file I/O fails.
pub fn save_safetensors(
    tensors: &HashMap<String, DynTensor>,
    path: impl AsRef<Path>,
) -> Result<()> {
    let bytes = tensors_to_safetensors_bytes(tensors)?;
    std::fs::write(path.as_ref(), bytes)?;
    Ok(())
}

/// Serialize named tensors to safetensors bytes (for in-memory use).
///
/// # Errors
///
/// Returns an error if any tensor is on GPU and cannot be read.
pub fn tensors_to_safetensors_bytes(tensors: &HashMap<String, DynTensor>) -> Result<Vec<u8>> {
    // Collect byte buffers first so they outlive the TensorView borrows.
    let mut entries: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let cpu_tensor = tensor.to_device(&Device::Cpu)?;
        let arr = cpu_tensor.to_f32_array()?;
        let shape = tensor.dims().to_vec();
        let data: Vec<f32> = arr.iter().copied().collect();
        entries.push((name.clone(), shape, f32_to_le_bytes(&data)));
    }

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = entries
        .iter()
        .map(|(name, shape, bytes)| {
            let view =
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .map_err(|e| TensorError::InvalidShape(format!("safetensors view: {e}")))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<Vec<_>>>()?;

    safetensors::tensor::serialize(views, None)
        .map_err(|e| TensorError::Unsupported(format!("safetensors serialize: {e}")))
}

/// Load named tensors from a safetensors file.
///
/// Returns a map of tensor name → DynTensor (CPU). Supports F32, BF16, and F16
/// dtypes — BF16/F16 tensors are stored natively without f32 conversion.
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed, or if the file
/// contains tensors with unsupported dtypes.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, DynTensor>> {
    let bytes = std::fs::read(path.as_ref())?;
    load_safetensors_from_bytes(&bytes)
}

/// Load named tensors from safetensors bytes.
///
/// Supports F32, BF16, and F16 dtypes. BF16/F16 tensors are stored natively
/// (no f32 conversion) via [`DynTensor::from_vec_bf16`] / [`DynTensor::from_vec_f16`].
pub fn load_safetensors_from_bytes(bytes: &[u8]) -> Result<HashMap<String, DynTensor>> {
    let st = safetensors::SafeTensors::deserialize(bytes)
        .map_err(|e| TensorError::InvalidShape(format!("safetensors deserialize: {e}")))?;

    let mut result = HashMap::new();
    for (name, view) in st.tensors() {
        let shape = view.shape();
        let data_bytes = view.data();
        let tensor = match view.dtype() {
            safetensors::Dtype::F32 => {
                if data_bytes.len() % 4 != 0 {
                    return Err(TensorError::InvalidShape(format!(
                        "tensor '{name}': F32 byte length {} not divisible by 4",
                        data_bytes.len()
                    )));
                }
                let values: Vec<f32> = data_bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                DynTensor::from_vec(values, shape, &Device::Cpu)?
            }
            safetensors::Dtype::BF16 => {
                if data_bytes.len() % 2 != 0 {
                    return Err(TensorError::InvalidShape(format!(
                        "tensor '{name}': BF16 byte length {} not divisible by 2",
                        data_bytes.len()
                    )));
                }
                let values: Vec<half::bf16> = data_bytes
                    .chunks_exact(2)
                    .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                DynTensor::from_vec_bf16(values, shape, &Device::Cpu)?
            }
            safetensors::Dtype::F16 => {
                if data_bytes.len() % 2 != 0 {
                    return Err(TensorError::InvalidShape(format!(
                        "tensor '{name}': F16 byte length {} not divisible by 2",
                        data_bytes.len()
                    )));
                }
                let values: Vec<half::f16> = data_bytes
                    .chunks_exact(2)
                    .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                DynTensor::from_vec_f16(values, shape, &Device::Cpu)?
            }
            other => {
                return Err(TensorError::Unsupported(format!(
                    "tensor '{name}': unsupported safetensors dtype {other:?}"
                )));
            }
        };
        result.insert(name.clone(), tensor);
    }
    Ok(result)
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_safetensors_bytes() {
        let t1 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[3], &Device::Cpu).unwrap();
        let t2 = DynTensor::from_vec(vec![4.0, 5.0], &[1, 2], &Device::Cpu).unwrap();
        let mut map = HashMap::new();
        map.insert("w1".to_string(), t1);
        map.insert("w2".to_string(), t2);

        let bytes = tensors_to_safetensors_bytes(&map).unwrap();
        let loaded = load_safetensors_from_bytes(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(
            loaded["w1"].to_flat_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0]
        );
        assert_eq!(loaded["w1"].dims(), &[3]);
        assert_eq!(loaded["w2"].to_flat_vec::<f32>().unwrap(), vec![4.0, 5.0]);
        assert_eq!(loaded["w2"].dims(), &[1, 2]);
    }

    #[test]
    fn test_roundtrip_safetensors_file() {
        let t = DynTensor::from_vec(vec![1.5, -2.5, 3.5], &[3], &Device::Cpu).unwrap();
        let mut map = HashMap::new();
        map.insert("param".to_string(), t);

        let dir = std::env::temp_dir().join(format!("nn_st_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");

        save_safetensors(&map, &path).unwrap();
        let loaded = load_safetensors(&path).unwrap();

        assert_eq!(
            loaded["param"].to_flat_vec::<f32>().unwrap(),
            vec![1.5, -2.5, 3.5]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_empty_tensors() {
        let map: HashMap<String, DynTensor> = HashMap::new();
        let bytes = tensors_to_safetensors_bytes(&map).unwrap();
        let loaded = load_safetensors_from_bytes(&bytes).unwrap();
        assert!(loaded.is_empty());
    }

    /// Build a safetensors byte buffer with a single tensor of the given dtype.
    fn build_safetensors_single(
        name: &str,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        data: &[u8],
    ) -> Vec<u8> {
        let view = safetensors::tensor::TensorView::new(dtype, shape, data).unwrap();
        safetensors::tensor::serialize(vec![(name.to_string(), view)], None).unwrap()
    }

    #[test]
    fn test_load_bf16_safetensors() {
        let values = [1.0f32, -2.5, 3.75];
        let bf16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let bytes = build_safetensors_single("w", safetensors::Dtype::BF16, vec![3], &bf16_bytes);
        let loaded = load_safetensors_from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        let t = &loaded["w"];
        assert_eq!(t.dims(), &[3]);
        assert_eq!(t.dtype(), crate::DType::BF16);
        // BF16 roundtrip: convert back to f32 and check values are close.
        let f32_vals = t.to_f32_array().unwrap();
        let f32_vec: Vec<f32> = f32_vals.iter().copied().collect();
        for (got, expected) in f32_vec.iter().zip(values.iter()) {
            assert!(
                (got - expected).abs() < 0.1,
                "BF16 roundtrip: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_load_f16_safetensors() {
        let values = [1.0f32, -2.5, 3.75];
        let f16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let bytes = build_safetensors_single("w", safetensors::Dtype::F16, vec![3], &f16_bytes);
        let loaded = load_safetensors_from_bytes(&bytes).unwrap();
        assert_eq!(loaded.len(), 1);
        let t = &loaded["w"];
        assert_eq!(t.dims(), &[3]);
        assert_eq!(t.dtype(), crate::DType::F16);
        let f32_vals = t.to_f32_array().unwrap();
        let f32_vec: Vec<f32> = f32_vals.iter().copied().collect();
        for (got, expected) in f32_vec.iter().zip(values.iter()) {
            assert!(
                (got - expected).abs() < 0.01,
                "F16 roundtrip: got {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_load_unsupported_dtype_errors() {
        // I64 is not supported — should return an error.
        let data = 42i64.to_le_bytes();
        let bytes = build_safetensors_single("x", safetensors::Dtype::I64, vec![1], &data);
        let err = load_safetensors_from_bytes(&bytes).unwrap_err();
        assert!(
            format!("{err}").contains("unsupported"),
            "expected unsupported dtype error, got: {err}"
        );
    }
}
