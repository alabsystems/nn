// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Security validation for GGUF tensor layouts, alignment, and metadata.
//!
//! These functions provide defense-in-depth checks beyond basic parsing:
//! - Tensor data regions fit within the data section
//! - No unintentional overlapping tensor byte ranges
//! - Tensor offsets are aligned for their dtype
//! - Tensor names contain only valid characters
//! - Metadata values are internally consistent

use std::collections::HashSet;

use crate::dequant::GgufDType;
use crate::error::GgufError;
use crate::metadata::GgufMetadata;
use crate::tensor_info::GgufTensorInfo;

#[cfg(test)]
#[path = "security_validation_tests.rs"]
mod tests;

/// Byte range occupied by a tensor within the data section.
#[derive(Debug, Clone)]
struct TensorRange {
    name: String,
    start: u64,
    end: u64,
}

/// Validate that all tensors fit within the data section.
///
/// Checks each tensor's `[offset .. offset + byte_size)` region against the
/// total `data_size`. This catches crafted GGUF files where tensor offsets
/// or shapes claim more space than the file provides.
///
/// Returns `Ok(())` if all tensors fit. Returns the first out-of-bounds
/// tensor as an error.
pub fn validate_tensor_layout(tensors: &[GgufTensorInfo], data_size: u64) -> Result<(), GgufError> {
    for t in tensors {
        let byte_size = t.checked_byte_size()?;
        let end = t
            .offset
            .checked_add(byte_size)
            .ok_or_else(|| GgufError::DataOffsetOverflow {
                name: t.name.clone(),
                data_offset: 0,
                tensor_offset: t.offset,
            })?;
        if end > data_size {
            return Err(GgufError::TensorExceedsDataSection {
                name: t.name.clone(),
                start: t.offset,
                end,
                data_size,
            });
        }
    }
    Ok(())
}

/// Return the natural alignment requirement (in bytes) for a given dtype.
///
/// Non-quantized types require alignment matching their element size (e.g.,
/// F32 requires 4-byte alignment, F16 requires 2-byte). Quantized block
/// types use their `type_size` as the natural alignment unit, capped at 32
/// to match GGUF's 32-byte file alignment.
///
/// Returns 1 for unknown/unsupported types (no alignment constraint).
pub fn dtype_alignment(dtype: GgufDType) -> u64 {
    match dtype {
        GgufDType::F32 | GgufDType::I32 => 4,
        GgufDType::F16 | GgufDType::BF16 | GgufDType::I16 => 2,
        GgufDType::F64 | GgufDType::I64 => 8,
        GgufDType::I8 => 1,
        // Quantized block types: aligned to the lesser of type_size and 32.
        _ => {
            let ts = dtype.type_size() as u64;
            if ts == 0 {
                1
            } else {
                ts.min(32)
            }
        }
    }
}

/// Validate that a tensor's data offset is properly aligned for its dtype.
///
/// Misaligned offsets can cause undefined behavior on architectures that
/// require aligned memory access (e.g., SIMD loads). Even on x86 where
/// misaligned access is allowed, it incurs performance penalties.
pub fn validate_alignment(name: &str, offset: u64, dtype: GgufDType) -> Result<(), GgufError> {
    let required = dtype_alignment(dtype);
    if required > 1 && !offset.is_multiple_of(required) {
        return Err(GgufError::MisalignedTensorOffset {
            name: name.to_string(),
            offset,
            required,
            dtype,
        });
    }
    Ok(())
}

/// Detect overlapping byte ranges among tensors.
///
/// While GGUF allows shared/aliased weights (same offset, same data), this
/// function detects *partial* overlaps which are almost always indicative of
/// corruption or a crafted malicious file.
///
/// Returns the first pair of overlapping tensors found, or `Ok(())` if none.
/// Tensors with identical `[start..end)` ranges are considered aliases and
/// are NOT flagged as overlaps.
pub fn detect_overlapping_tensors(tensors: &[GgufTensorInfo]) -> Result<(), GgufError> {
    // Collect byte ranges, skipping tensors with zero byte size.
    let mut ranges: Vec<TensorRange> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let byte_size = match t.checked_byte_size() {
            Ok(s) => s,
            Err(_) => continue, // Skip tensors with invalid sizes.
        };
        if byte_size == 0 {
            continue;
        }
        let end = match t.offset.checked_add(byte_size) {
            Some(e) => e,
            None => continue,
        };
        ranges.push(TensorRange {
            name: t.name.clone(),
            start: t.offset,
            end,
        });
    }

    // Sort by start offset for efficient overlap detection.
    ranges.sort_by_key(|r| (r.start, r.end));

    for i in 1..ranges.len() {
        let prev = &ranges[i - 1];
        let curr = &ranges[i];

        // If previous range ends after current starts, they overlap.
        if prev.end > curr.start {
            // Allow exact aliases (identical ranges).
            if prev.start == curr.start && prev.end == curr.end {
                continue;
            }
            return Err(GgufError::OverlappingTensors {
                name_a: prev.name.clone(),
                name_b: curr.name.clone(),
                start_a: prev.start,
                end_a: prev.end,
                start_b: curr.start,
                end_b: curr.end,
            });
        }
    }

    Ok(())
}

/// Validate that all tensor names are unique.
///
/// Duplicate tensor names in a GGUF file indicate corruption or a crafted
/// file attempting to confuse consumers that store tensors in a hashmap
/// (where the second entry silently overwrites the first).
pub fn validate_unique_tensor_names(tensors: &[GgufTensorInfo]) -> Result<(), GgufError> {
    let mut seen = HashSet::with_capacity(tensors.len());
    for t in tensors {
        if !seen.insert(&t.name) {
            return Err(GgufError::DuplicateTensorName {
                name: t.name.clone(),
            });
        }
    }
    Ok(())
}

/// Validate that tensor names contain only safe characters.
///
/// Rejects names containing:
/// - Null bytes (truncation attacks)
/// - Control characters (U+0000..U+001F, U+007F) except tab
/// - Lone surrogates (invalid UTF-8 would already be rejected, but this
///   catches edge cases in valid-but-suspicious strings)
///
/// Note: Rust strings are always valid UTF-8, so the name has already passed
/// UTF-8 validation. This checks for semantically suspicious content.
pub fn validate_tensor_name(name: &str) -> Result<(), GgufError> {
    for (i, byte) in name.bytes().enumerate() {
        if byte == 0 {
            return Err(GgufError::InvalidTensorName {
                name: name.to_string(),
                byte_index: i,
                description: "null byte".to_string(),
            });
        }
        // Reject ASCII control characters except tab (0x09), which some
        // GGUF producers may legitimately use in tensor names.
        if byte < 0x20 && byte != 0x09 {
            return Err(GgufError::InvalidTensorName {
                name: name.to_string(),
                byte_index: i,
                description: format!("control character 0x{byte:02x}"),
            });
        }
        if byte == 0x7F {
            return Err(GgufError::InvalidTensorName {
                name: name.to_string(),
                byte_index: i,
                description: "DEL character (0x7F)".to_string(),
            });
        }
    }
    Ok(())
}

/// Cross-check metadata values for internal consistency.
///
/// Validates relationships between metadata entries that should be
/// consistent. Currently checks:
/// - `general.architecture` is a non-empty string if present
/// - `*.vocab_size` is positive if present
/// - `*.embedding_length` is positive if present
/// - `*.block_count` is positive if present
///
/// Returns `Ok(())` if all checks pass. This is advisory validation --
/// the parser will still load files that fail these checks, but callers
/// can use this to detect potentially corrupt metadata.
pub fn validate_metadata_consistency(metadata: &GgufMetadata) -> Result<Vec<String>, GgufError> {
    let mut warnings = Vec::new();

    // Check architecture is non-empty if present.
    if let Some(arch) = metadata.get_str("general.architecture") {
        if arch.trim().is_empty() {
            warnings.push("general.architecture is present but empty".to_string());
        }
    }

    // Check numeric metadata that should be positive.
    let positive_keys = [
        "llama.embedding_length",
        "llama.block_count",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.vocab_size",
        "qwen2.embedding_length",
        "qwen2.block_count",
        "qwen2.attention.head_count",
        "qwen2.attention.head_count_kv",
        "qwen2.vocab_size",
    ];
    for key in &positive_keys {
        if let Some(val) = metadata.get_u64(key) {
            if val == 0 {
                warnings.push(format!("{key} is 0 (expected positive)"));
            }
        }
    }

    // Check context_length is reasonable if present.
    for prefix in &["llama", "qwen2"] {
        let key = format!("{prefix}.context_length");
        if let Some(val) = metadata.get_u64(&key) {
            if val > 1_000_000 {
                warnings.push(format!("{key} = {val} (unusually large, >1M tokens)"));
            }
        }
    }

    Ok(warnings)
}

/// Run all security validations on a set of tensors and metadata.
///
/// Convenience function that runs the full suite of validation checks:
/// 1. Unique tensor names
/// 2. Tensor name character validation
/// 3. Tensor layout within data section
/// 4. Alignment checking
/// 5. Overlapping tensor detection
/// 6. Metadata consistency
///
/// Returns the first hard error encountered, or a list of metadata
/// consistency warnings if all hard checks pass.
pub fn validate_all(
    tensors: &[GgufTensorInfo],
    data_size: u64,
    metadata: &GgufMetadata,
) -> Result<Vec<String>, GgufError> {
    validate_unique_tensor_names(tensors)?;

    for t in tensors {
        validate_tensor_name(&t.name)?;
        validate_alignment(&t.name, t.offset, t.dtype)?;
    }

    validate_tensor_layout(tensors, data_size)?;
    detect_overlapping_tensors(tensors)?;
    validate_metadata_consistency(metadata)
}
