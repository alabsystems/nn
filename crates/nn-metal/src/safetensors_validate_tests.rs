// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Regression tests for first-party safetensors structural validation:
//! truncated and crafted files are refused with `WeightError::InvalidFile`,
//! and every file the reference serializer writes is accepted.

use super::{validate_safetensors_bytes, MAX_HEADER_LEN};
use crate::safetensors::WeightError;

fn valid_file() -> Vec<u8> {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype as StDtype;
    let a: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b: Vec<u16> = vec![1, 2, 3];
    let views = vec![
        (
            "a".to_string(),
            TensorView::new(StDtype::F32, vec![2, 3], bytemuck::cast_slice(&a)).expect("view"),
        ),
        (
            "b".to_string(),
            TensorView::new(StDtype::F16, vec![3], bytemuck::cast_slice(&b)).expect("view"),
        ),
    ];
    serialize(views, None).expect("serialize")
}

/// Assemble a file from a raw header string and a data region.
fn file_with_header(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

fn reason(bytes: &[u8]) -> String {
    match validate_safetensors_bytes(bytes) {
        Err(WeightError::InvalidFile { reason }) => reason,
        Err(other) => panic!("expected InvalidFile, got {other}"),
        Ok(layout) => panic!("expected InvalidFile, got Ok({layout:?})"),
    }
}

#[test]
fn test_valid_serialized_file_accepted() {
    let bytes = valid_file();
    let layout = validate_safetensors_bytes(&bytes).expect("valid file");
    assert_eq!(layout.tensor_count, 2);
    assert_eq!(layout.data_len, 6 * 4 + 3 * 2);
    assert_eq!(layout.data_start + layout.data_len, bytes.len());
    // The reference parser agrees.
    safetensors::SafeTensors::deserialize(&bytes).expect("reference parser accepts");
}

#[test]
fn test_valid_file_with_metadata_and_empty_tensor_accepted() {
    let header = r#"{"__metadata__":{"format":"pt"},"e":{"dtype":"F32","shape":[0],"data_offsets":[0,0]},"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
    let bytes = file_with_header(header, &[7, 8]);
    let layout = validate_safetensors_bytes(&bytes).expect("valid");
    assert_eq!(layout.tensor_count, 2);
    safetensors::SafeTensors::deserialize(&bytes).expect("reference parser accepts");
}

#[test]
fn test_empty_header_object_accepted() {
    let bytes = file_with_header("{}", &[]);
    assert_eq!(
        validate_safetensors_bytes(&bytes)
            .expect("valid")
            .tensor_count,
        0
    );
}

/// Every proper prefix of a valid file (a truncation at any byte) is refused.
#[test]
fn test_every_truncation_refused() {
    let bytes = valid_file();
    for cut in 0..bytes.len() {
        let r = reason(&bytes[..cut]);
        assert!(
            !r.is_empty(),
            "truncation at {cut} refused without a reason"
        );
    }
}

#[test]
fn test_empty_file_refused() {
    assert!(reason(&[]).contains("header-length prefix"));
}

#[test]
fn test_trailing_garbage_refused() {
    let mut bytes = valid_file();
    bytes.push(0);
    assert!(reason(&bytes).contains("not fully described"));
}

#[test]
fn test_header_length_past_end_refused() {
    let mut bytes = valid_file();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    bytes[..8].copy_from_slice(&(n + 1_000).to_le_bytes());
    assert!(reason(&bytes).contains("runs past the end"));
}

#[test]
fn test_huge_header_length_refused() {
    let mut bytes = valid_file();
    bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(reason(&bytes).contains("limit"));
    bytes[..8].copy_from_slice(&(MAX_HEADER_LEN + 1).to_le_bytes());
    assert!(reason(&bytes).contains("limit"));
}

#[test]
fn test_bad_magic_refused() {
    let mut bytes = valid_file();
    bytes[8] = b'[';
    assert!(reason(&bytes).contains("does not start with '{'"));
}

#[test]
fn test_non_utf8_and_non_json_header_refused() {
    let mut bytes = valid_file();
    bytes[9] = 0xff;
    assert!(reason(&bytes).contains("UTF-8"));
    let bytes = file_with_header("{not json", &[]);
    assert!(reason(&bytes).contains("not JSON"));
}

#[test]
fn test_offsets_past_data_region_refused() {
    let header = r#"{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
    assert!(reason(&file_with_header(header, &[1, 2])).contains("exceeds"));
}

/// Offsets near `u64::MAX` are refused in first-party code (the reference
/// parser adds `buffer_end + 8 + n` unchecked).
#[test]
fn test_huge_offsets_refused() {
    let header = format!(
        r#"{{"x":{{"dtype":"U8","shape":[1],"data_offsets":[{},{}]}}}}"#,
        u64::MAX - 1,
        u64::MAX
    );
    let r = reason(&file_with_header(&header, &[1]));
    assert!(r.contains("exceeds") || r.contains("overflows"), "{r}");
}

#[test]
fn test_reversed_offsets_refused() {
    let header = r#"{"x":{"dtype":"U8","shape":[0],"data_offsets":[2,1]}}"#;
    assert!(reason(&file_with_header(header, &[1, 2])).contains("start 2 > end 1"));
}

#[test]
fn test_overlapping_tensors_refused() {
    let header = r#"{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"y":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#;
    assert!(reason(&file_with_header(header, &[1, 2, 3])).contains("overlap or gap"));
}

#[test]
fn test_gap_between_tensors_refused() {
    let header = r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"y":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#;
    assert!(reason(&file_with_header(header, &[1, 2, 3])).contains("overlap or gap"));
}

#[test]
fn test_shape_size_mismatch_refused() {
    let header = r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
    assert!(reason(&file_with_header(header, &[0; 4])).contains("expected"));
}

#[test]
fn test_shape_overflow_refused() {
    let header = format!(
        r#"{{"x":{{"dtype":"F32","shape":[{},{}],"data_offsets":[0,4]}}}}"#,
        u64::MAX,
        u64::MAX
    );
    assert!(reason(&file_with_header(&header, &[0; 4])).contains("overflows"));
}

#[test]
fn test_malformed_entries_refused() {
    for header in [
        r#"{"x":1}"#,
        r#"{"x":{"shape":[1],"data_offsets":[0,1]}}"#,
        r#"{"x":{"dtype":"U8","data_offsets":[0,1]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1,2]}}"#,
        r#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[-1,1]}}"#,
        r#"{"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,1]}}"#,
    ] {
        reason(&file_with_header(header, &[1]));
    }
}

/// Single-byte corruption anywhere in the header never panics: the file is
/// either refused or still a structurally valid file.
#[test]
fn test_header_byte_flips_never_panic() {
    let bytes = valid_file();
    let header_end = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    for i in 0..header_end {
        for flip in [0x01u8, 0x80, 0xff] {
            let mut corrupted = bytes.clone();
            corrupted[i] ^= flip;
            let _ = validate_safetensors_bytes(&corrupted);
        }
    }
}

/// The one rule stricter than the reference 0.8 parser: a header that begins
/// with JSON whitespace instead of `{` (which the format spec forbids and no
/// known writer emits) is refused, although `serde_json` in the reference
/// parser would skip the whitespace. Trailing whitespace padding, which the
/// reference serializer does emit, stays accepted (`valid_file` above).
#[test]
fn test_leading_whitespace_header_refused_deliberately() {
    let header = r#" {"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
    let bytes = file_with_header(header, &[7, 8]);
    safetensors::SafeTensors::deserialize(&bytes).expect("reference parser tolerates it");
    assert!(reason(&bytes).contains("does not start with '{'"));
    let padded = r#"{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}    "#;
    let bytes = file_with_header(padded, &[7, 8]);
    validate_safetensors_bytes(&bytes).expect("trailing padding accepted");
    safetensors::SafeTensors::deserialize(&bytes).expect("reference parser accepts");
}
