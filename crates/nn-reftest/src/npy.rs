// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Load and write reference tensors in NumPy `.npy` format.
//!
//! The `.npy` format is NumPy's native binary tensor format. This module
//! provides zero-dependency parsing of `.npy` v1.0/v2.0 files with automatic
//! conversion to f32, matching the safetensors loader semantics.
//!
//! Dtype conversion functions live in `npy_convert.rs`.
//!
//! # Reading
//!
//! Two APIs are available for reading `.npy` files:
//!
//! - [`read_npy`] returns a standalone [`NpyTensor`] with dtype metadata.
//! - [`load_npy`] returns a [`ReferenceTrace`] for trace-based comparison.
//!
//! ```rust,no_run
//! use nn_reftest::npy::{read_npy, write_npy};
//!
//! let tensor = read_npy("reference/encoder_output.npy").expect("failed to read");
//! println!("shape: {:?}, dtype: {:?}", tensor.shape, tensor.dtype);
//! ```
//!
//! # Writing
//!
//! [`write_npy`] writes f32 data to a valid `.npy` v1.0 file:
//!
//! ```rust,no_run
//! use nn_reftest::npy::write_npy;
//!
//! write_npy("output.npy", &[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("write failed");
//! ```
//!
//! # Round-trip
//!
//! ```rust,no_run
//! use nn_reftest::npy::{read_npy, write_npy};
//!
//! write_npy("/tmp/test.npy", &[1.0, 2.0, 3.0], &[3]).unwrap();
//! let t = read_npy("/tmp/test.npy").unwrap();
//! assert_eq!(t.data, vec![1.0, 2.0, 3.0]);
//! ```
//!
//! For loading a directory of `.npy` files as a multi-checkpoint trace:
//!
//! ```rust,no_run
//! use nn_reftest::load_npy_dir;
//!
//! // Each .npy file becomes a checkpoint, sorted by filename.
//! let trace = load_npy_dir("reference/checkpoints/").expect("failed to load");
//! ```

use std::path::Path;

use crate::error::ReftestError;
use crate::trace::{NamedTensor, ReferenceTrace};

/// NumPy dtype descriptor — the element type stored in a `.npy` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NpyDType {
    /// 16-bit IEEE 754 float (`<f2` / `>f2`).
    F16,
    /// 32-bit IEEE 754 float (`<f4` / `>f4`).
    F32,
    /// 64-bit IEEE 754 float (`<f8` / `>f8`).
    F64,
    /// 32-bit signed integer (`<i4`).
    I32,
    /// 64-bit signed integer (`<i8`).
    I64,
    /// 8-bit unsigned integer (`|u1`).
    U8,
}

impl NpyDType {
    /// Parse a NumPy dtype descriptor string into an [`NpyDType`].
    ///
    /// Returns `None` for unrecognised descriptors.
    #[must_use]
    pub fn from_descr(descr: &str) -> Option<Self> {
        match descr {
            "<f2" | ">f2" => Some(Self::F16),
            "<f4" | ">f4" => Some(Self::F32),
            "<f8" | ">f8" => Some(Self::F64),
            "<i4" | "=i4" => Some(Self::I32),
            "<i8" | "=i8" => Some(Self::I64),
            "<u1" | "|u1" => Some(Self::U8),
            // Also accept the narrower integer types that convert_npy_to_f32 handles,
            // but they don't map to a first-class NpyDType variant yet.
            _ => None,
        }
    }

    /// Return the NumPy descriptor string for this dtype (little-endian).
    #[must_use]
    pub fn to_descr(self) -> &'static str {
        match self {
            Self::F16 => "<f2",
            Self::F32 => "<f4",
            Self::F64 => "<f8",
            Self::I32 => "<i4",
            Self::I64 => "<i8",
            Self::U8 => "|u1",
        }
    }
}

impl std::fmt::Display for NpyDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.to_descr())
    }
}

/// A single tensor loaded from a `.npy` file, with dtype metadata.
///
/// Data is always converted to `f32` for uniformity, but `dtype` records the
/// original on-disk element type.
#[derive(Debug, Clone)]
pub struct NpyTensor {
    /// Flattened f32 element data (row-major / C order).
    pub data: Vec<f32>,
    /// Tensor dimensions (e.g. `[2, 3]` for a 2x3 matrix).
    pub shape: Vec<usize>,
    /// The original NumPy dtype of the file.
    pub dtype: NpyDType,
}

impl NpyTensor {
    /// Number of elements in this tensor.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Errors specific to `.npy` file I/O.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum NpyError {
    /// The file does not start with the `\x93NUMPY` magic bytes.
    #[error("invalid NPY magic bytes (expected \\x93NUMPY)")]
    BadMagic,

    /// The file uses an unsupported NPY format version.
    #[error("unsupported NPY version {major}.{minor} (only 1.0 and 2.0 supported)")]
    UnsupportedVersion { major: u8, minor: u8 },

    /// The NPY header could not be parsed.
    #[error("NPY header parse error: {0}")]
    HeaderParse(String),

    /// The file uses a dtype that is not supported.
    #[error("unsupported NPY dtype: {0}")]
    UnsupportedDtype(String),

    /// The file uses Fortran (column-major) order.
    #[error("NPY Fortran order not supported (only C order)")]
    FortranOrder,

    /// The shape product overflows `usize`.
    #[error("shape product overflow: {0:?}")]
    ShapeOverflow(Vec<usize>),

    /// Data length does not match the expected element count.
    #[error("data length mismatch: shape {shape:?} requires {expected} elements, got {actual}")]
    DataLengthMismatch {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },

    /// Filesystem I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Wrapped reftest error from the conversion layer.
    #[error("{0}")]
    Reftest(#[from] ReftestError),
}

/// Read a single `.npy` file and return an [`NpyTensor`] with dtype metadata.
///
/// The data is always converted to `f32`. The original dtype is recorded in
/// [`NpyTensor::dtype`].
pub fn read_npy(path: impl AsRef<Path>) -> Result<NpyTensor, NpyError> {
    let bytes = std::fs::read(path.as_ref())?;
    read_npy_from_bytes(&bytes)
}

/// Parse an in-memory `.npy` buffer into an [`NpyTensor`].
pub fn read_npy_from_bytes(data: &[u8]) -> Result<NpyTensor, NpyError> {
    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(NpyError::BadMagic);
    }

    let major = data[6];
    let minor = data[7];

    let (header_len, header_start) = match (major, minor) {
        (1, 0) => {
            let len = u16::from_le_bytes([data[8], data[9]]) as usize;
            (len, 10usize)
        }
        (2, 0) => {
            if data.len() < 12 {
                return Err(NpyError::BadMagic);
            }
            let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            (len, 12usize)
        }
        _ => {
            return Err(NpyError::UnsupportedVersion { major, minor });
        }
    };

    let data_start = header_start
        .checked_add(header_len)
        .ok_or_else(|| NpyError::HeaderParse("header_len overflow".into()))?;
    if data.len() < data_start {
        return Err(NpyError::HeaderParse(
            "file truncated before data section".into(),
        ));
    }

    let header_bytes = &data[header_start..data_start];
    let header = std::str::from_utf8(header_bytes)
        .map_err(|e| NpyError::HeaderParse(format!("invalid UTF-8: {e}")))?;

    let (descr, shape, fortran_order) = parse_npy_header(header)?;

    if fortran_order {
        return Err(NpyError::FortranOrder);
    }

    let dtype = NpyDType::from_descr(&descr).unwrap_or(NpyDType::F32); // fallback; convert_npy_to_f32 will reject truly unknown dtypes

    let raw = &data[data_start..];
    let numel: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| NpyError::ShapeOverflow(shape.clone()))?;

    let f32_data = convert_npy_to_f32(raw, &descr, numel)?;

    Ok(NpyTensor {
        data: f32_data,
        shape,
        dtype,
    })
}

#[path = "npy_convert.rs"]
pub(crate) mod convert;
use convert::convert_npy_to_f32;

#[path = "npy_write.rs"]
mod write;
pub use write::{write_npy, write_npy_to_bytes};

/// NPY magic bytes: `\x93NUMPY`.
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Load a single `.npy` file as a one-checkpoint [`ReferenceTrace`].
///
/// The checkpoint name is derived from the filename stem (e.g.,
/// `encoder_output.npy` → `"encoder_output"`).
#[must_use = "returns a Result that may contain an error"]
pub fn load_npy(path: impl AsRef<Path>) -> Result<ReferenceTrace, ReftestError> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("tensor")
        .to_string();
    let tensor = parse_npy(&bytes, name)?;
    Ok(ReferenceTrace::from_checkpoints(vec![tensor]))
}

/// Load a single `.npy` file from in-memory bytes.
///
/// The checkpoint is named `tensor_name`.
#[must_use = "returns a Result that may contain an error"]
pub fn load_npy_from_bytes(
    data: &[u8],
    tensor_name: impl Into<String>,
) -> Result<ReferenceTrace, ReftestError> {
    let tensor = parse_npy(data, tensor_name.into())?;
    Ok(ReferenceTrace::from_checkpoints(vec![tensor]))
}

/// Load a directory of `.npy` files as a multi-checkpoint [`ReferenceTrace`].
///
/// Each `.npy` file becomes a checkpoint named after the file stem. Files are
/// sorted alphabetically by filename for deterministic ordering.
///
/// Non-`.npy` files are silently skipped.
#[must_use = "returns a Result that may contain an error"]
pub fn load_npy_dir(path: impl AsRef<Path>) -> Result<ReferenceTrace, ReftestError> {
    let dir = path.as_ref();
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|entry| {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    let _ = std::io::Write::write_fmt(
                        &mut std::io::stderr(),
                        format_args!(
                            "nn-reftest: load_npy_dir: failed to read directory entry in {}: {e}\n",
                            dir.display()
                        ),
                    );
                    return None;
                }
            };
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("npy") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    // Sort by filename for deterministic ordering.
    entries.sort();

    let mut checkpoints = Vec::with_capacity(entries.len());
    for entry_path in &entries {
        let bytes = std::fs::read(entry_path)?;
        let name = entry_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("tensor")
            .to_string();
        checkpoints.push(parse_npy(&bytes, name)?);
    }

    Ok(ReferenceTrace::from_checkpoints(checkpoints))
}

/// Parse a `.npy` byte buffer into a [`NamedTensor`].
pub(crate) fn parse_npy(data: &[u8], name: String) -> Result<NamedTensor, ReftestError> {
    // Minimum size: 6 (magic) + 2 (version) + 2 (header_len v1) = 10
    if data.len() < 10 || &data[..6] != NPY_MAGIC {
        return Err(ReftestError::NpyBadMagic);
    }

    let major = data[6];
    let minor = data[7];

    let (header_len, header_start) = match (major, minor) {
        (1, 0) => {
            let len = u16::from_le_bytes([data[8], data[9]]) as usize;
            (len, 10usize)
        }
        (2, 0) => {
            if data.len() < 12 {
                return Err(ReftestError::NpyBadMagic);
            }
            let len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
            (len, 12usize)
        }
        _ => {
            return Err(ReftestError::NpyUnsupportedVersion { major, minor });
        }
    };

    let data_start = header_start
        .checked_add(header_len)
        .ok_or_else(|| ReftestError::NpyHeaderParse("header_len overflow".into()))?;
    if data.len() < data_start {
        return Err(ReftestError::NpyHeaderParse(
            "file truncated before data section".into(),
        ));
    }

    let header_bytes = &data[header_start..data_start];
    let header = std::str::from_utf8(header_bytes)
        .map_err(|e| ReftestError::NpyHeaderParse(format!("invalid UTF-8: {e}")))?;

    let (dtype, shape, fortran_order) = parse_npy_header(header)?;

    if fortran_order {
        return Err(ReftestError::NpyFortranOrder);
    }

    let raw = &data[data_start..];
    let numel: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| ReftestError::ShapeProductOverflow(shape.clone()))?;

    let f32_data = convert_npy_to_f32(raw, &dtype, numel)?;

    NamedTensor::new(name, shape, f32_data)
}

/// Parse the NumPy header dict string to extract dtype, shape, and order.
///
/// The header is a Python literal dict like:
/// `{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }`
pub(crate) fn parse_npy_header(header: &str) -> Result<(String, Vec<usize>, bool), ReftestError> {
    let header = header.trim().trim_matches(|c| c == '{' || c == '}');

    let dtype = extract_string_value(header, "descr")
        .ok_or_else(|| ReftestError::NpyHeaderParse("missing 'descr' field".into()))?;

    let fortran_order = extract_bool_value(header, "fortran_order").unwrap_or(false);

    let shape = extract_shape(header)
        .ok_or_else(|| ReftestError::NpyHeaderParse("missing or invalid 'shape' field".into()))?;

    Ok((dtype, shape, fortran_order))
}

/// Extract a string value from a Python dict literal.
/// Looks for `'key': 'value'` or `'key': "value"`.
pub(crate) fn extract_string_value(header: &str, key: &str) -> Option<String> {
    // Find 'key':
    let pattern = format!("'{key}'");
    let key_pos = header.find(&pattern)?;
    let after_key = &header[key_pos + pattern.len()..];

    // Skip colon and whitespace.
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let value_start = after_colon.trim_start();

    // Extract quoted string.
    let quote = value_start.as_bytes().first()?;
    if *quote != b'\'' && *quote != b'"' {
        return None;
    }
    let quote_char = *quote as char;
    let inner = &value_start[1..];
    let end = inner.find(quote_char)?;
    Some(inner[..end].to_string())
}

/// Extract a boolean value from a Python dict literal.
pub(crate) fn extract_bool_value(header: &str, key: &str) -> Option<bool> {
    let pattern = format!("'{key}'");
    let key_pos = header.find(&pattern)?;
    let after_key = &header[key_pos + pattern.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();

    if after_colon.starts_with("True") {
        Some(true)
    } else if after_colon.starts_with("False") {
        Some(false)
    } else {
        None
    }
}

/// Extract shape tuple from a Python dict literal.
/// Handles `(3, 4)`, `(3,)` (1-D), and `()` (scalar).
pub(crate) fn extract_shape(header: &str) -> Option<Vec<usize>> {
    let pattern = "'shape'";
    let key_pos = header.find(pattern)?;
    let after_key = &header[key_pos + pattern.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();

    let paren_start = after_colon.find('(')?;
    let paren_end = after_colon.find(')')?;
    let inner = after_colon[paren_start + 1..paren_end].trim();

    if inner.is_empty() {
        return Some(vec![]); // scalar
    }

    let dims: Option<Vec<usize>> = inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().ok())
        .collect();

    dims
}

#[cfg(test)]
#[path = "npy_tests.rs"]
mod tests;
