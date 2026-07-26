// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Safetensors weight loading utilities for the import pipeline.
//!
//! Extracted from `convert.rs` (Wave 4 D3a). Self-contained I/O utilities
//! with no cross-references to the pipeline logic.

use std::collections::HashMap;
use std::path::Path;

use crate::error::ImportError;

/// Load safetensors weights into a map from tensor name (FQN) to (f32 data, shape).
pub(crate) fn load_safetensors_weights(
    path: &Path,
) -> Result<HashMap<String, (Vec<f32>, Vec<usize>)>, ImportError> {
    let data = std::fs::read(path).map_err(|e| ImportError::Io {
        path: path.display().to_string(),
        detail: e.to_string(),
    })?;
    let tensors = safetensors::SafeTensors::deserialize(&data).map_err(|e| ImportError::Io {
        path: path.display().to_string(),
        detail: format!("safetensors parse: {e}"),
    })?;

    let mut result = HashMap::new();
    for (name, view) in tensors.tensors() {
        let shape: Vec<usize> = view.shape().to_vec();
        let f32_data = tensor_view_to_f32(&view, &name)?;
        result.insert(name, (f32_data, shape));
    }
    Ok(result)
}

/// Convert a safetensors tensor view to f32 data.
///
/// Returns an error for dtypes that cannot be converted to f32.
pub(super) fn tensor_view_to_f32(
    view: &safetensors::tensor::TensorView<'_>,
    name: &str,
) -> Result<Vec<f32>, ImportError> {
    use safetensors::Dtype;
    let raw = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        Dtype::F16 => Ok(raw
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::BF16 => Ok(raw
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        Dtype::F64 => Ok(raw
            .chunks_exact(8)
            .map(|c| {
                let bytes: [u8; 8] = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                f64::from_le_bytes(bytes) as f32
            })
            .collect()),
        Dtype::I64 => Ok(raw
            .chunks_exact(8)
            .map(|c| {
                let bytes: [u8; 8] = [c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]];
                i64::from_le_bytes(bytes) as f32
            })
            .collect()),
        Dtype::U8 => Ok(raw.iter().map(|&b| f32::from(b)).collect()),
        Dtype::I8 => Ok(raw.iter().map(|&b| f32::from(b as i8)).collect()),
        other => Err(ImportError::UnsupportedDtype {
            name: name.to_string(),
            dtype: format!("{other:?}"),
        }),
    }
}
