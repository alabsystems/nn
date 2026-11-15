// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! First-party structural validation of a mapped safetensors file.
//!
//! [`WeightMap::load`](super::WeightMap::load) (and through it
//! `CompiledKokoro::load`, `load_with_config` and `load_with_hard_bounds`)
//! runs [`validate_safetensors_bytes`] on the mmap before anything else reads
//! it: before the foreign `safetensors` parser, before the Metal no-copy
//! buffer is created, and before any tensor range is indexed. A truncated or
//! crafted file is refused with [`WeightError::InvalidFile`]; nothing is read
//! out of bounds and nothing panics.
//!
//! Layout checked (the safetensors format):
//!
//! ```text
//! [ u64 LE header_len ][ header_len bytes of JSON, starting with '{' ][ data ]
//! ```
//!
//! * the file holds the 8-byte length prefix;
//! * `header_len` is at most [`MAX_HEADER_LEN`] (the reference parser's
//!   limit) and the header lies inside the file (checked add);
//! * the header starts with `{` (the format's magic) and is a UTF-8 JSON
//!   object;
//! * every tensor entry has a string `dtype`, an integer `shape` and a
//!   two-integer `data_offsets` `[start, end]` with `start <= end <= data_len`;
//! * for the dtypes nn loads, `end - start` equals `numel * dtype_size`
//!   (checked multiply);
//! * the entries, ordered by `start`, tile the data region exactly: they
//!   start at 0, never overlap or leave a gap, and end at `data_len`.
//!
//! Every one of these except the `{` check is something the reference
//! `safetensors` 0.8 parser also enforces, so a file it accepts passes here
//! unless its header begins with JSON whitespace instead of `{`. The format
//! specification requires the header to begin with `{` and every known
//! writer (the reference Rust and Python serializers pad only at the end) emits
//! it, so no realistic weight file is affected. This
//! validator exists so the bounds are checked in first-party code (where the
//! verifier sees them) and so a file with offsets near `u64::MAX` is refused
//! before reaching the reference parser's unchecked `buffer_end + 8 + n`.
//!
//! Decision recorded 2026-10-02: made by the orchestrator under the owner's
//! instruction ("decide for yourself ... prefer reversible steps").

use super::WeightError;

/// Largest header the loader accepts, in bytes (the reference `safetensors`
/// parser's `MAX_HEADER_SIZE`).
pub(super) const MAX_HEADER_LEN: u64 = 100_000_000;

/// Size of the little-endian `u64` header-length prefix.
const LEN_PREFIX: usize = 8;

/// Key of the optional free-form metadata entry in the header.
const METADATA_KEY: &str = "__metadata__";

/// Structural facts established by [`validate_safetensors_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SafetensorsLayout {
    /// Length of the JSON header in bytes.
    pub header_len: usize,
    /// Byte offset of the data region (`8 + header_len`).
    pub data_start: usize,
    /// Length of the data region (`file_len - data_start`).
    pub data_len: usize,
    /// Number of tensor entries in the header.
    pub tensor_count: usize,
}

fn invalid(reason: impl Into<String>) -> WeightError {
    WeightError::InvalidFile {
        reason: reason.into(),
    }
}

/// Byte size of one element for the dtypes nn loads; `None` for the rest
/// (they are refused later by `convert_dtype`, and their sub-byte sizes are
/// left to the reference parser).
fn dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "BOOL" | "U8" => Some(1),
        "F16" | "BF16" => Some(2),
        "F32" | "I32" => Some(4),
        "F64" | "I64" => Some(8),
        _ => None,
    }
}

fn as_usize(value: &serde_json::Value, what: &str, name: &str) -> Result<usize, WeightError> {
    let Some(raw) = value.as_u64() else {
        return Err(invalid(format!(
            "tensor `{name}`: {what} is not a non-negative integer"
        )));
    };
    usize::try_from(raw)
        .map_err(|_| invalid(format!("tensor `{name}`: {what} {raw} overflows usize")))
}

/// Validate the structure of a whole safetensors file held in `bytes`.
///
/// # Errors
///
/// [`WeightError::InvalidFile`] naming the first violated rule (see the
/// module documentation).
pub fn validate_safetensors_bytes(bytes: &[u8]) -> Result<SafetensorsLayout, WeightError> {
    let file_len = bytes.len();
    let Some(prefix) = bytes.get(..LEN_PREFIX) else {
        return Err(invalid(format!(
            "file is {file_len} bytes, shorter than the {LEN_PREFIX}-byte header-length prefix"
        )));
    };
    let mut raw = [0u8; LEN_PREFIX];
    raw.copy_from_slice(prefix);
    let header_len_u64 = u64::from_le_bytes(raw);
    if header_len_u64 > MAX_HEADER_LEN {
        return Err(invalid(format!(
            "header length {header_len_u64} exceeds the {MAX_HEADER_LEN}-byte limit"
        )));
    }
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| invalid(format!("header length {header_len_u64} overflows usize")))?;
    let Some(data_start) = LEN_PREFIX.checked_add(header_len) else {
        return Err(invalid("header end overflows usize"));
    };
    let Some(header) = bytes.get(LEN_PREFIX..data_start) else {
        return Err(invalid(format!(
            "header of {header_len} bytes runs past the end of the {file_len}-byte file (truncated?)"
        )));
    };
    if header.first() != Some(&b'{') {
        return Err(invalid(
            "header does not start with '{' (not a safetensors file)",
        ));
    }
    let Some(data_len) = file_len.checked_sub(data_start) else {
        return Err(invalid("header end lies past the end of the file"));
    };
    let text =
        std::str::from_utf8(header).map_err(|e| invalid(format!("header is not UTF-8: {e}")))?;
    let parsed: serde_json::Value =
        serde_json::from_str(text).map_err(|e| invalid(format!("header is not JSON: {e}")))?;
    let Some(entries) = parsed.as_object() else {
        return Err(invalid("header is not a JSON object"));
    };

    let mut ranges: Vec<(usize, usize, &str)> = Vec::with_capacity(entries.len());
    for (name, entry) in entries {
        if name == METADATA_KEY {
            continue;
        }
        let Some(fields) = entry.as_object() else {
            return Err(invalid(format!("tensor `{name}`: entry is not an object")));
        };
        let Some(dtype) = fields.get("dtype").and_then(serde_json::Value::as_str) else {
            return Err(invalid(format!("tensor `{name}`: missing string `dtype`")));
        };
        let Some(shape) = fields.get("shape").and_then(serde_json::Value::as_array) else {
            return Err(invalid(format!("tensor `{name}`: missing array `shape`")));
        };
        let Some(offsets) = fields
            .get("data_offsets")
            .and_then(serde_json::Value::as_array)
        else {
            return Err(invalid(format!(
                "tensor `{name}`: missing array `data_offsets`"
            )));
        };
        let [start_v, end_v] = offsets.as_slice() else {
            return Err(invalid(format!(
                "tensor `{name}`: `data_offsets` has {} elements, expected 2",
                offsets.len()
            )));
        };
        let start = as_usize(start_v, "data_offsets start", name)?;
        let end = as_usize(end_v, "data_offsets end", name)?;
        if start > end {
            return Err(invalid(format!(
                "tensor `{name}`: data_offsets start {start} > end {end}"
            )));
        }
        if end > data_len {
            return Err(invalid(format!(
                "tensor `{name}`: data_offsets end {end} exceeds the {data_len}-byte data region (truncated?)"
            )));
        }
        let mut numel: usize = 1;
        for dim in shape {
            let d = as_usize(dim, "shape dimension", name)?;
            numel = numel.checked_mul(d).ok_or_else(|| {
                invalid(format!("tensor `{name}`: shape product overflows usize"))
            })?;
        }
        if let Some(size) = dtype_size(dtype) {
            let Some(bytes_needed) = numel.checked_mul(size) else {
                return Err(invalid(format!(
                    "tensor `{name}`: byte size overflows usize"
                )));
            };
            // `start <= end` was checked above, so this cannot underflow.
            let span = end - start;
            if span != bytes_needed {
                return Err(invalid(format!(
                    "tensor `{name}`: {span} data bytes for {numel} {dtype} elements ({bytes_needed} expected)"
                )));
            }
        }
        ranges.push((start, end, name.as_str()));
    }

    ranges.sort_unstable();
    let mut cursor: usize = 0;
    for &(start, end, name) in &ranges {
        if start != cursor {
            return Err(invalid(format!(
                "tensor `{name}`: data starts at {start}, expected {cursor} (overlap or gap)"
            )));
        }
        cursor = end;
    }
    if cursor != data_len {
        return Err(invalid(format!(
            "tensors cover {cursor} of {data_len} data bytes (file not fully described)"
        )));
    }

    Ok(SafetensorsLayout {
        header_len,
        data_start,
        data_len,
        tensor_count: ranges.len(),
    })
}

#[cfg(test)]
#[path = "safetensors_validate_tests.rs"]
mod tests;
