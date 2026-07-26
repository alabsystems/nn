// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NumPy `.npy` file writing functions.
//!
//! Extracted from `npy.rs` to keep files under the 500-line maintenance
//! threshold.

use std::path::Path;

use super::{NpyError, NPY_MAGIC};

/// Write f32 data to a `.npy` v1.0 file (little-endian, C order).
///
/// The file uses dtype `<f4` (little-endian float32).
///
/// # Errors
///
/// Returns [`NpyError::DataLengthMismatch`] if `data.len()` does not equal the
/// product of `shape`, or [`NpyError::ShapeOverflow`] on shape overflow.
pub fn write_npy(path: impl AsRef<Path>, data: &[f32], shape: &[usize]) -> Result<(), NpyError> {
    let bytes = write_npy_to_bytes(data, shape)?;
    std::fs::write(path.as_ref(), bytes)?;
    Ok(())
}

/// Serialize f32 data to an in-memory `.npy` v1.0 byte buffer.
///
/// The output uses dtype `<f4` (little-endian float32), C order.
pub fn write_npy_to_bytes(data: &[f32], shape: &[usize]) -> Result<Vec<u8>, NpyError> {
    let numel: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| NpyError::ShapeOverflow(shape.to_vec()))?;

    if data.len() != numel {
        return Err(NpyError::DataLengthMismatch {
            shape: shape.to_vec(),
            expected: numel,
            actual: data.len(),
        });
    }

    // Build the Python dict header.
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        format!("({})", dims.join(", "))
    };

    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}",
    );

    // NPY v1.0: 6 (magic) + 2 (version) + 2 (header_len) = 10 byte prefix.
    let prefix_len: usize = 10;
    let total_header = header.len() + 1; // +1 for trailing newline
                                         // Pad to 64-byte alignment.
    let padded_len = (prefix_len + total_header).div_ceil(64) * 64 - prefix_len;
    let padding = padded_len - header.len() - 1;

    let data_bytes = numel * 4; // 4 bytes per f32
    let mut buf = Vec::with_capacity(prefix_len + padded_len + data_bytes);

    // Magic + version.
    buf.extend_from_slice(NPY_MAGIC);
    buf.push(1); // major
    buf.push(0); // minor

    // Header length (u16 LE for v1.0).
    let header_len = padded_len as u16;
    buf.extend_from_slice(&header_len.to_le_bytes());

    // Header string + padding + newline.
    buf.extend_from_slice(header.as_bytes());
    buf.extend(std::iter::repeat_n(b' ', padding));
    buf.push(b'\n');

    // Raw f32 data in little-endian.
    for &val in data {
        buf.extend_from_slice(&val.to_le_bytes());
    }

    Ok(buf)
}
