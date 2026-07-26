// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GGUF model format parser for nn.
//!
//! Parses GGUF v3 files (the standard model distribution format for
//! llama.cpp and the broader GGML ecosystem). Supports dequantization
//! of Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, and Q8_0
//! quantized tensors to f32.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_gguf::GgufFile;
//!
//! // Open with mmap for zero-copy access:
//! let gguf = GgufFile::open("model.gguf")?;
//! println!("Architecture: {:?}", gguf.architecture());
//! println!("Tensors: {}", gguf.tensor_names().len());
//!
//! // Zero-copy raw tensor bytes:
//! let raw = gguf.tensor_data("token_embd.weight").unwrap()?;
//!
//! // Dequantize to f32:
//! let (data, shape) = gguf.dequantize_tensor("token_embd.weight")?;
//! ```

mod arch_llama;
mod arch_qwen3;
mod architecture;
mod dequant;
mod error;
mod header;
mod metadata;
mod reader;
pub mod security_validation;
mod tensor_info;
pub mod tensor_layout;

#[cfg(test)]
#[path = "security_tests.rs"]
mod security_tests;

#[cfg(test)]
#[path = "gguf_security_hardening_tests.rs"]
mod gguf_security_hardening_tests;

pub use arch_llama::{build_llama_graph, build_llama_graph_with_weights, LlamaConfig};
pub use arch_qwen3::{gguf_to_hf_name, load_qwen3_tensors, Qwen3GgufConfig};
pub use architecture::ModelArchitecture;
pub use dequant::{
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_0, dequantize_q4_1, dequantize_q4_k,
    dequantize_q5_0, dequantize_q5_1, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0, GgufDType,
};
pub use error::GgufError;
pub use header::GgufHeader;
pub use metadata::{GgufMetadata, GgufMetadataValue};
pub use reader::GgufFile;
pub use security_validation::{
    detect_overlapping_tensors, dtype_alignment, validate_alignment, validate_all,
    validate_metadata_consistency, validate_tensor_layout, validate_tensor_name,
    validate_unique_tensor_names,
};
pub use tensor_info::GgufTensorInfo;
pub use tensor_layout::{
    compute_byte_size, flatten_shape, reshape_tensor_data, transpose_shape, transpose_tensor_data,
    unflatten_shape, LayoutMap, TensorLayout,
};
