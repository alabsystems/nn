// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GGUF security validation: tensor layout, alignment, overlap
//! detection, name validation, and metadata consistency.

use std::collections::HashMap;

use crate::dequant::GgufDType;
use crate::error::GgufError;
use crate::metadata::{GgufMetadata, GgufMetadataValue};
use crate::tensor_info::GgufTensorInfo;

use super::*;

/// Helper to build a `GgufTensorInfo` concisely.
fn make_tensor(name: &str, shape: &[u64], dtype: GgufDType, offset: u64) -> GgufTensorInfo {
    GgufTensorInfo {
        name: name.to_string(),
        n_dims: shape.len() as u32,
        shape: shape.to_vec(),
        dtype,
        offset,
    }
}

/// Helper to build metadata from key-value pairs.
fn make_metadata(entries: Vec<(&str, GgufMetadataValue)>) -> GgufMetadata {
    let entries = entries
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect::<HashMap<_, _>>();
    GgufMetadata { entries }
}

// ---------------------------------------------------------------
// validate_tensor_layout
// ---------------------------------------------------------------

#[test]
fn test_layout_valid_tensors_fit_in_data_section() {
    let tensors = vec![
        make_tensor("a", &[4], GgufDType::F32, 0), // 16 bytes at [0..16)
        make_tensor("b", &[8], GgufDType::F32, 16), // 32 bytes at [16..48)
    ];
    validate_tensor_layout(&tensors, 48).expect("should fit");
}

#[test]
fn test_layout_tensor_extends_past_data_section() {
    let tensors = vec![
        make_tensor("big", &[1024], GgufDType::F32, 0), // 4096 bytes
    ];
    let err = validate_tensor_layout(&tensors, 100).unwrap_err();
    assert!(
        matches!(err, GgufError::TensorExceedsDataSection { .. }),
        "expected TensorExceedsDataSection, got: {err}"
    );
}

#[test]
fn test_layout_tensor_offset_plus_size_overflows_u64() {
    let tensors = vec![
        make_tensor("evil", &[4], GgufDType::F32, u64::MAX - 8), // offset + 16 overflows
    ];
    let err = validate_tensor_layout(&tensors, u64::MAX).unwrap_err();
    assert!(
        matches!(err, GgufError::DataOffsetOverflow { .. }),
        "expected DataOffsetOverflow, got: {err}"
    );
}

#[test]
fn test_layout_empty_tensors_list_is_valid() {
    validate_tensor_layout(&[], 0).expect("empty list should be valid");
}

#[test]
fn test_layout_tensor_exactly_at_boundary() {
    let tensors = vec![
        make_tensor("exact", &[4], GgufDType::F32, 0), // 16 bytes at [0..16)
    ];
    validate_tensor_layout(&tensors, 16).expect("should fit exactly");
}

#[test]
fn test_layout_tensor_one_byte_past_boundary() {
    let tensors = vec![
        make_tensor("over", &[4], GgufDType::F32, 0), // 16 bytes
    ];
    let err = validate_tensor_layout(&tensors, 15).unwrap_err();
    assert!(matches!(err, GgufError::TensorExceedsDataSection { .. }));
}

// ---------------------------------------------------------------
// validate_alignment
// ---------------------------------------------------------------

#[test]
fn test_alignment_f32_aligned() {
    validate_alignment("t", 0, GgufDType::F32).expect("0 is aligned");
    validate_alignment("t", 4, GgufDType::F32).expect("4 is aligned");
    validate_alignment("t", 128, GgufDType::F32).expect("128 is aligned");
}

#[test]
fn test_alignment_f32_misaligned() {
    let err = validate_alignment("t", 1, GgufDType::F32).unwrap_err();
    assert!(
        matches!(err, GgufError::MisalignedTensorOffset { required: 4, .. }),
        "expected MisalignedTensorOffset, got: {err}"
    );
}

#[test]
fn test_alignment_f32_misaligned_at_3() {
    let err = validate_alignment("t", 3, GgufDType::F32).unwrap_err();
    assert!(matches!(err, GgufError::MisalignedTensorOffset { .. }));
}

#[test]
fn test_alignment_f16_aligned() {
    validate_alignment("t", 0, GgufDType::F16).expect("0 is aligned");
    validate_alignment("t", 2, GgufDType::F16).expect("2 is aligned");
    validate_alignment("t", 64, GgufDType::F16).expect("64 is aligned");
}

#[test]
fn test_alignment_f16_misaligned() {
    let err = validate_alignment("t", 1, GgufDType::F16).unwrap_err();
    assert!(matches!(
        err,
        GgufError::MisalignedTensorOffset { required: 2, .. }
    ));
}

#[test]
fn test_alignment_f64_aligned() {
    validate_alignment("t", 0, GgufDType::F64).expect("0 is aligned");
    validate_alignment("t", 8, GgufDType::F64).expect("8 is aligned");
}

#[test]
fn test_alignment_f64_misaligned() {
    let err = validate_alignment("t", 4, GgufDType::F64).unwrap_err();
    assert!(matches!(
        err,
        GgufError::MisalignedTensorOffset { required: 8, .. }
    ));
}

#[test]
fn test_alignment_i8_always_aligned() {
    // I8 has 1-byte alignment, so any offset is valid.
    validate_alignment("t", 0, GgufDType::I8).expect("aligned");
    validate_alignment("t", 1, GgufDType::I8).expect("aligned");
    validate_alignment("t", 7, GgufDType::I8).expect("aligned");
}

#[test]
fn test_alignment_bf16_aligned() {
    validate_alignment("t", 0, GgufDType::BF16).expect("aligned");
    validate_alignment("t", 2, GgufDType::BF16).expect("aligned");
}

#[test]
fn test_alignment_bf16_misaligned() {
    let err = validate_alignment("t", 3, GgufDType::BF16).unwrap_err();
    assert!(matches!(err, GgufError::MisalignedTensorOffset { .. }));
}

#[test]
fn test_alignment_q4_0_quantized() {
    // Q4_0: type_size=18, capped to min(18, 32) = 18.
    validate_alignment("t", 0, GgufDType::Q4_0).expect("0 is aligned");
    validate_alignment("t", 18, GgufDType::Q4_0).expect("18 is aligned");
    validate_alignment("t", 36, GgufDType::Q4_0).expect("36 is aligned");
}

#[test]
fn test_alignment_q4_0_misaligned() {
    let err = validate_alignment("t", 1, GgufDType::Q4_0).unwrap_err();
    assert!(matches!(err, GgufError::MisalignedTensorOffset { .. }));
}

// ---------------------------------------------------------------
// detect_overlapping_tensors
// ---------------------------------------------------------------

#[test]
fn test_overlap_no_tensors() {
    detect_overlapping_tensors(&[]).expect("empty is valid");
}

#[test]
fn test_overlap_single_tensor() {
    let tensors = vec![make_tensor("a", &[4], GgufDType::F32, 0)];
    detect_overlapping_tensors(&tensors).expect("single tensor is valid");
}

#[test]
fn test_overlap_adjacent_tensors_no_overlap() {
    let tensors = vec![
        make_tensor("a", &[4], GgufDType::F32, 0),  // [0..16)
        make_tensor("b", &[4], GgufDType::F32, 16), // [16..32)
    ];
    detect_overlapping_tensors(&tensors).expect("adjacent is valid");
}

#[test]
fn test_overlap_partial_overlap_detected() {
    let tensors = vec![
        make_tensor("a", &[8], GgufDType::F32, 0),  // [0..32)
        make_tensor("b", &[4], GgufDType::F32, 16), // [16..32) — partial overlap
    ];
    let err = detect_overlapping_tensors(&tensors).unwrap_err();
    assert!(
        matches!(err, GgufError::OverlappingTensors { .. }),
        "expected OverlappingTensors, got: {err}"
    );
}

#[test]
fn test_overlap_exact_alias_allowed() {
    // Exact same range = shared/aliased weight. Should be allowed.
    let tensors = vec![
        make_tensor("a", &[4], GgufDType::F32, 0), // [0..16)
        make_tensor("b", &[4], GgufDType::F32, 0), // [0..16) — exact alias
    ];
    detect_overlapping_tensors(&tensors).expect("exact alias should be allowed");
}

#[test]
fn test_overlap_contained_tensor_detected() {
    // Tensor B is entirely contained within tensor A.
    let tensors = vec![
        make_tensor("a", &[16], GgufDType::F32, 0), // [0..64)
        make_tensor("b", &[4], GgufDType::F32, 16), // [16..32) — contained
    ];
    let err = detect_overlapping_tensors(&tensors).unwrap_err();
    assert!(matches!(err, GgufError::OverlappingTensors { .. }));
}

#[test]
fn test_overlap_three_tensors_second_and_third_overlap() {
    let tensors = vec![
        make_tensor("a", &[4], GgufDType::F32, 0),  // [0..16)
        make_tensor("b", &[8], GgufDType::F32, 32), // [32..64)
        make_tensor("c", &[4], GgufDType::F32, 48), // [48..64) — overlaps b
    ];
    let err = detect_overlapping_tensors(&tensors).unwrap_err();
    assert!(matches!(err, GgufError::OverlappingTensors { .. }));
}

#[test]
fn test_overlap_unsorted_offsets_still_detected() {
    // Provide tensors in reverse offset order.
    let tensors = vec![
        make_tensor("b", &[4], GgufDType::F32, 8), // [8..24)
        make_tensor("a", &[8], GgufDType::F32, 0), // [0..32) — overlaps b
    ];
    let err = detect_overlapping_tensors(&tensors).unwrap_err();
    assert!(matches!(err, GgufError::OverlappingTensors { .. }));
}

// ---------------------------------------------------------------
// validate_unique_tensor_names
// ---------------------------------------------------------------

#[test]
fn test_unique_names_all_unique() {
    let tensors = vec![
        make_tensor("a", &[4], GgufDType::F32, 0),
        make_tensor("b", &[4], GgufDType::F32, 16),
        make_tensor("c", &[4], GgufDType::F32, 32),
    ];
    validate_unique_tensor_names(&tensors).expect("all unique");
}

#[test]
fn test_unique_names_duplicate_detected() {
    let tensors = vec![
        make_tensor("weight", &[4], GgufDType::F32, 0),
        make_tensor("bias", &[4], GgufDType::F32, 16),
        make_tensor("weight", &[8], GgufDType::F32, 32),
    ];
    let err = validate_unique_tensor_names(&tensors).unwrap_err();
    assert!(
        matches!(err, GgufError::DuplicateTensorName { ref name } if name == "weight"),
        "expected DuplicateTensorName, got: {err}"
    );
}

#[test]
fn test_unique_names_empty_list() {
    validate_unique_tensor_names(&[]).expect("empty list is valid");
}

// ---------------------------------------------------------------
// validate_tensor_name
// ---------------------------------------------------------------

#[test]
fn test_name_valid_standard() {
    validate_tensor_name("blk.0.attn_q.weight").expect("standard name");
}

#[test]
fn test_name_valid_with_dots_underscores_digits() {
    validate_tensor_name("model.layers.31.self_attn.q_proj.weight").expect("valid");
}

#[test]
fn test_name_null_byte_rejected() {
    let name = "evil\0name";
    let err = validate_tensor_name(name).unwrap_err();
    assert!(
        matches!(err, GgufError::InvalidTensorName { byte_index: 4, .. }),
        "expected InvalidTensorName at byte 4, got: {err}"
    );
}

#[test]
fn test_name_control_char_rejected() {
    let name = "evil\x01name";
    let err = validate_tensor_name(name).unwrap_err();
    assert!(matches!(err, GgufError::InvalidTensorName { .. }));
}

#[test]
fn test_name_newline_rejected() {
    let name = "evil\nname";
    let err = validate_tensor_name(name).unwrap_err();
    assert!(matches!(err, GgufError::InvalidTensorName { .. }));
}

#[test]
fn test_name_carriage_return_rejected() {
    let name = "evil\rname";
    let err = validate_tensor_name(name).unwrap_err();
    assert!(matches!(err, GgufError::InvalidTensorName { .. }));
}

#[test]
fn test_name_del_character_rejected() {
    let name = "evil\x7fname";
    let err = validate_tensor_name(name).unwrap_err();
    assert!(matches!(err, GgufError::InvalidTensorName { .. }));
}

#[test]
fn test_name_tab_allowed() {
    // Tab (0x09) is explicitly allowed.
    validate_tensor_name("name\twith_tab").expect("tab should be allowed");
}

#[test]
fn test_name_unicode_allowed() {
    // Non-ASCII UTF-8 characters should be allowed.
    validate_tensor_name("tensor_\u{00E9}").expect("valid unicode");
    validate_tensor_name("\u{4E2D}\u{6587}").expect("CJK characters allowed");
}

#[test]
fn test_name_empty_string_valid() {
    // An empty string has no invalid characters.
    validate_tensor_name("").expect("empty string is technically valid");
}

#[test]
fn test_name_bell_character_rejected() {
    let name = "name\x07bell";
    let err = validate_tensor_name(name).unwrap_err();
    assert!(matches!(err, GgufError::InvalidTensorName { .. }));
}

#[test]
fn test_name_escape_character_rejected() {
    let name = "name\x1besc";
    let err = validate_tensor_name(name).unwrap_err();
    assert!(matches!(err, GgufError::InvalidTensorName { .. }));
}

// ---------------------------------------------------------------
// validate_metadata_consistency
// ---------------------------------------------------------------

#[test]
fn test_metadata_empty_is_consistent() {
    let metadata = make_metadata(vec![]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert!(warnings.is_empty());
}

#[test]
fn test_metadata_valid_llama_config() {
    let metadata = make_metadata(vec![
        (
            "general.architecture",
            GgufMetadataValue::String("llama".to_string()),
        ),
        ("llama.embedding_length", GgufMetadataValue::U32(4096)),
        ("llama.block_count", GgufMetadataValue::U32(32)),
        ("llama.attention.head_count", GgufMetadataValue::U32(32)),
        ("llama.vocab_size", GgufMetadataValue::U32(32000)),
    ]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert!(warnings.is_empty(), "got warnings: {warnings:?}");
}

#[test]
fn test_metadata_empty_architecture_warned() {
    let metadata = make_metadata(vec![(
        "general.architecture",
        GgufMetadataValue::String(String::new()),
    )]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("empty"));
}

#[test]
fn test_metadata_whitespace_only_architecture_warned() {
    let metadata = make_metadata(vec![(
        "general.architecture",
        GgufMetadataValue::String("   ".to_string()),
    )]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert_eq!(warnings.len(), 1);
}

#[test]
fn test_metadata_zero_embedding_length_warned() {
    let metadata = make_metadata(vec![("llama.embedding_length", GgufMetadataValue::U32(0))]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("embedding_length"));
}

#[test]
fn test_metadata_zero_block_count_warned() {
    let metadata = make_metadata(vec![("llama.block_count", GgufMetadataValue::U32(0))]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("block_count"));
}

#[test]
fn test_metadata_zero_vocab_size_warned() {
    let metadata = make_metadata(vec![("llama.vocab_size", GgufMetadataValue::U32(0))]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("vocab_size"));
}

#[test]
fn test_metadata_huge_context_length_warned() {
    let metadata = make_metadata(vec![(
        "llama.context_length",
        GgufMetadataValue::U64(2_000_000),
    )]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert_eq!(warnings.len(), 1);
    assert!(warnings[0].contains("unusually large"));
}

#[test]
fn test_metadata_reasonable_context_length_ok() {
    let metadata = make_metadata(vec![(
        "llama.context_length",
        GgufMetadataValue::U64(131072),
    )]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert!(warnings.is_empty());
}

#[test]
fn test_metadata_multiple_zero_values_all_warned() {
    let metadata = make_metadata(vec![
        ("llama.embedding_length", GgufMetadataValue::U32(0)),
        ("llama.block_count", GgufMetadataValue::U32(0)),
        ("llama.vocab_size", GgufMetadataValue::U32(0)),
    ]);
    let warnings = validate_metadata_consistency(&metadata).expect("should succeed");
    assert_eq!(warnings.len(), 3);
}

// ---------------------------------------------------------------
// validate_all (integration)
// ---------------------------------------------------------------

#[test]
fn test_validate_all_valid_file() {
    let tensors = vec![
        make_tensor("weight", &[4, 3], GgufDType::F32, 0), // 48 bytes
        make_tensor("bias", &[4], GgufDType::F32, 48),     // 16 bytes
    ];
    let metadata = make_metadata(vec![(
        "general.architecture",
        GgufMetadataValue::String("llama".to_string()),
    )]);
    let warnings = validate_all(&tensors, 64, &metadata).expect("should pass");
    assert!(warnings.is_empty());
}

#[test]
fn test_validate_all_catches_duplicate_names() {
    let tensors = vec![
        make_tensor("dup", &[4], GgufDType::F32, 0),
        make_tensor("dup", &[4], GgufDType::F32, 16),
    ];
    let metadata = make_metadata(vec![]);
    let err = validate_all(&tensors, 32, &metadata).unwrap_err();
    assert!(matches!(err, GgufError::DuplicateTensorName { .. }));
}

#[test]
fn test_validate_all_catches_invalid_name() {
    let tensors = vec![make_tensor("evil\x00name", &[4], GgufDType::F32, 0)];
    let metadata = make_metadata(vec![]);
    let err = validate_all(&tensors, 16, &metadata).unwrap_err();
    assert!(matches!(err, GgufError::InvalidTensorName { .. }));
}

#[test]
fn test_validate_all_catches_misalignment() {
    let tensors = vec![
        make_tensor("t", &[4], GgufDType::F32, 3), // offset 3 not aligned to 4
    ];
    let metadata = make_metadata(vec![]);
    let err = validate_all(&tensors, 64, &metadata).unwrap_err();
    assert!(matches!(err, GgufError::MisalignedTensorOffset { .. }));
}

#[test]
fn test_validate_all_catches_out_of_bounds() {
    let tensors = vec![
        make_tensor("t", &[1024], GgufDType::F32, 0), // 4096 bytes
    ];
    let metadata = make_metadata(vec![]);
    let err = validate_all(&tensors, 100, &metadata).unwrap_err();
    assert!(matches!(err, GgufError::TensorExceedsDataSection { .. }));
}

#[test]
fn test_validate_all_catches_overlap() {
    let tensors = vec![
        make_tensor("a", &[8], GgufDType::F32, 0),  // [0..32)
        make_tensor("b", &[4], GgufDType::F32, 16), // [16..32) — overlap
    ];
    let metadata = make_metadata(vec![]);
    let err = validate_all(&tensors, 64, &metadata).unwrap_err();
    assert!(matches!(err, GgufError::OverlappingTensors { .. }));
}

// ---------------------------------------------------------------
// dtype_alignment
// ---------------------------------------------------------------

#[test]
fn test_dtype_alignment_values() {
    assert_eq!(dtype_alignment(GgufDType::F32), 4);
    assert_eq!(dtype_alignment(GgufDType::F16), 2);
    assert_eq!(dtype_alignment(GgufDType::BF16), 2);
    assert_eq!(dtype_alignment(GgufDType::F64), 8);
    assert_eq!(dtype_alignment(GgufDType::I8), 1);
    assert_eq!(dtype_alignment(GgufDType::I16), 2);
    assert_eq!(dtype_alignment(GgufDType::I32), 4);
    assert_eq!(dtype_alignment(GgufDType::I64), 8);
    // Quantized: min(type_size, 32).
    assert_eq!(dtype_alignment(GgufDType::Q4_0), 18);
    assert_eq!(dtype_alignment(GgufDType::Q8_0), 32); // type_size=34, capped to 32
}

// ---------------------------------------------------------------
// Edge cases: maximum u64 values
// ---------------------------------------------------------------

#[test]
fn test_layout_max_u64_offset_overflow() {
    let tensors = vec![make_tensor("evil", &[1], GgufDType::I8, u64::MAX)];
    let err = validate_tensor_layout(&tensors, u64::MAX).unwrap_err();
    assert!(
        matches!(err, GgufError::DataOffsetOverflow { .. }),
        "expected overflow, got: {err}"
    );
}

#[test]
fn test_layout_max_u64_data_size_with_valid_tensor() {
    let tensors = vec![
        make_tensor("t", &[1], GgufDType::I8, 0), // 1 byte at [0..1)
    ];
    validate_tensor_layout(&tensors, u64::MAX).expect("should fit in max data section");
}

#[test]
fn test_overlap_max_offset_tensors() {
    // Two tensors at the very end of the u64 range. The overlap check should
    // not panic on checked_add.
    let tensors = vec![
        make_tensor("a", &[1], GgufDType::I8, u64::MAX - 2),
        make_tensor("b", &[1], GgufDType::I8, u64::MAX - 1),
    ];
    // These would be invalid in validate_tensor_layout (would overflow),
    // but detect_overlapping_tensors should not panic.
    let result = detect_overlapping_tensors(&tensors);
    // Either Ok or an error is fine, just no panic.
    let _ = result;
}
