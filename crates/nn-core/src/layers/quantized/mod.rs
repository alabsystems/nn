// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Quantized layers: GGML Q4K, INT8 per-channel, MXFP4, INT4/INT8 per-group,
//! GPTQ/AWQ INT4, and residual vector quantization.
//!
//! - [`QLinear`] / [`BlockQ4K`] — GGML Q4_K_S quantized linear layer (~4.5 bits/weight)
//! - [`Int8Linear`] / [`Int8QuantParams`] — INT8 per-channel quantized linear (W8A16)
//! - [`Mxfp4Block`] / [`Mxfp4Tensor`] — MXFP4 block-quantized 4-bit (OCP MX spec)
//! - [`QuantizedTensor`] / [`QuantizationConfig`] — INT4/INT8 per-group weight quantization
//! - [`GptqLinear`] / [`GptqFormat`] — GPTQ INT4 group-quantized linear
//! - [`AwqFormat`] — AWQ INT4 group-quantized linear (bit-compatible with GPTQ)
//! - [`Rvq`] / [`VqCodebook`] — residual vector quantization

// -- GGML Quantized Linear (Q4K for memory-efficient LLM inference) -----------
mod linear;
pub use linear::{BlockQ4K, GgmlDType, QLinear, QuantizedWeight};

// -- Residual Vector Quantization (RVQ) ---------------------------------------
mod rvq;
pub use rvq::{Rvq, VqCodebook};

// -- INT8 per-channel quantization (W8A16) ------------------------------------
mod int8;
pub use int8::{
    dequantize_per_channel, max_quantization_error, quantize_per_channel, Int8Mode, Int8QuantParams,
};
mod int8_linear;
pub use int8_linear::Int8Linear;

// -- MXFP4 block-quantized 4-bit (OCP Microscaling) --------------------------
mod mxfp4;
pub use mxfp4::{
    dequantize_block as mxfp4_dequantize_block, dequantize_tensor as mxfp4_dequantize_tensor,
    quantize_block as mxfp4_quantize_block, quantize_tensor as mxfp4_quantize_tensor, Mxfp4Block,
    Mxfp4Tensor, BLOCK_STORAGE_BYTES as MXFP4_BLOCK_STORAGE_BYTES, MXFP4_BLOCK_SIZE,
};

// -- INT4/INT8 per-group weight quantization (VLM deployment) ----------------
mod weight_quant;
pub use weight_quant::{
    dequantize as weight_dequantize, quantize_per_group as weight_quantize_per_group,
    quantized_matmul, QuantDtype, QuantizationConfig, QuantizedTensor,
};

// -- GPTQ INT4 group-quantized (AutoGPTQ / HuggingFace) ----------------------
pub(crate) mod gptq_loader;
pub use gptq_loader::{
    load_gptq_linear, unpack_gptq_qweight, unpack_gptq_qzeros, GptqFormat, GptqLinear,
};

// -- AWQ INT4 group-quantized (bit-compatible with GPTQ) ---------------------
mod awq_loader;
pub use awq_loader::{load_awq_linear, unpack_awq_qweight, AwqFormat};

#[cfg(kani)]
#[path = "kani_mxfp4_extended_proofs.rs"]
mod kani_mxfp4_extended_proofs;

#[cfg(kani)]
#[path = "kani_gptq_awq_proofs.rs"]
mod kani_gptq_awq_proofs;

#[cfg(kani)]
#[path = "kani_quantized_extra.rs"]
mod kani_quantized_extra;

#[cfg(kani)]
#[path = "kani_weight_quant_proofs.rs"]
mod kani_weight_quant_proofs;

#[cfg(kani)]
#[path = "kani_rvq_proofs.rs"]
mod kani_rvq_proofs;

#[cfg(kani)]
#[path = "kani_int8_extended_proofs.rs"]
mod kani_int8_extended_proofs;

#[cfg(test)]
mod rvq_tests;

#[cfg(test)]
#[path = "gptq_awq_tests.rs"]
mod gptq_awq_tests;
