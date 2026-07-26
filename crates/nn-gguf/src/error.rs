// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for GGUF parsing.

use thiserror::Error;

use crate::dequant::GgufDType;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("not a GGUF file: invalid magic (expected 0x46475547, got {found:#010x})")]
    InvalidMagic { found: u32 },

    #[error("unsupported GGUF version {version} (only v3 supported)")]
    UnsupportedVersion { version: u32 },

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid UTF-8 in string at offset {offset}")]
    InvalidUtf8 { offset: u64 },

    #[error("unsupported quantization type: {dtype:?}")]
    UnsupportedDType { dtype: GgufDType },

    #[error("tensor {name} has unexpected size: expected {expected} bytes, got {actual}")]
    TensorSizeMismatch {
        name: String,
        expected: u64,
        actual: u64,
    },

    #[error("unknown metadata value type: {type_id}")]
    UnknownMetadataType { type_id: u32 },

    #[error("missing required metadata key: {key}")]
    MissingMetadata { key: String },

    #[error("missing required tensor: {name}")]
    MissingTensor { name: String },

    #[error("unsupported architecture: expected \"{expected}\", got \"{found}\"")]
    ArchitectureMismatch { expected: String, found: String },

    #[error("reshape element count mismatch: source has {from_count} elements, target shape has {to_count}")]
    ReshapeElementMismatch { from_count: usize, to_count: usize },

    #[error("cannot compute byte size for unsupported quantization type: {dtype:?}")]
    UnsupportedByteSizeComputation { dtype: GgufDType },

    #[error("quantized tensor reshape requires element count divisible by block size {block_size}, got {element_count}")]
    QuantBlockAlignment {
        block_size: usize,
        element_count: usize,
    },

    #[error("GGUF header declares {count} tensors, exceeding maximum of {max}")]
    TensorCountExceeded { count: u64, max: u64 },

    #[error("GGUF header declares {count} metadata entries, exceeding maximum of {max}")]
    MetadataCountExceeded { count: u64, max: u64 },

    #[error("GGUF string length {len} exceeds maximum of {max} bytes")]
    StringLengthExceeded { len: u64, max: u64 },

    #[error("GGUF metadata array declares {count} elements, exceeding maximum of {max}")]
    ArrayLengthExceeded { count: u64, max: u64 },

    #[error("tensor {name}: shape causes element count overflow")]
    ElementCountOverflow { name: String },

    #[error("tensor {name}: byte size computation overflows (elements={elements}, type_size={type_size}, block_size={block_size})")]
    ByteSizeOverflow {
        name: String,
        elements: u64,
        type_size: u64,
        block_size: u64,
    },

    #[error("tensor {name}: data offset overflow (data_offset={data_offset}, tensor_offset={tensor_offset})")]
    DataOffsetOverflow {
        name: String,
        data_offset: u64,
        tensor_offset: u64,
    },

    #[error("tensor dimensions {n_dims} exceeds maximum of {max}")]
    DimensionCountExceeded { n_dims: u32, max: u32 },

    #[error("tensor {name}: byte size {byte_size} exceeds maximum allowed {max} bytes")]
    TensorTooLarge {
        name: String,
        byte_size: u64,
        max: u64,
    },

    #[error("tensor {name}: dimension {dim_index} has size 0, which is not allowed")]
    ZeroDimension { name: String, dim_index: usize },

    #[error(
        "tensor {name}: data region [{start}..{end}) extends beyond file bounds ({file_len} bytes)"
    )]
    DataOutOfBounds {
        name: String,
        start: u64,
        end: u64,
        file_len: u64,
    },

    #[error("tensor {name}: offset {offset} is not aligned to {required}-byte boundary for dtype {dtype:?}")]
    MisalignedTensorOffset {
        name: String,
        offset: u64,
        required: u64,
        dtype: GgufDType,
    },

    #[error("tensors {name_a} and {name_b} have overlapping data regions: [{start_a}..{end_a}) and [{start_b}..{end_b})")]
    OverlappingTensors {
        name_a: String,
        name_b: String,
        start_a: u64,
        end_a: u64,
        start_b: u64,
        end_b: u64,
    },

    #[error("tensor {name}: data region [{start}..{end}) extends beyond data section of {data_size} bytes")]
    TensorExceedsDataSection {
        name: String,
        start: u64,
        end: u64,
        data_size: u64,
    },

    #[error("duplicate tensor name: {name}")]
    DuplicateTensorName { name: String },

    #[error("tensor name contains invalid character at byte {byte_index}: {description}")]
    InvalidTensorName {
        name: String,
        byte_index: usize,
        description: String,
    },
}
