// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Metal GPU dispatch for quantized matmul (INT4/INT8 W4A16/W8A16).
//!
//! Phase 1 (current): CPU fallback path.
//!   - Reads FP32 input from GPU to CPU
//!   - Calls `quantized_matmul` from nn-core (dequantize + BLAS matmul)
//!   - Uploads FP32 result back to GPU
//!
//! Phase 2 (future): Native MSL kernel with on-the-fly dequantization.
//!   - Pack quantized weights, scales, zero_points into Metal buffers at load time
//!   - Per-tile dequantize during GEMM (one thread-group loads, dequantizes, MACs)
//!   - ~8x memory savings for INT4, ~4x for INT8 vs FP32 weight storage
//!
//! # MSL Kernel Design (Phase 2)
//!
//! ```text
//! Buffer layout:
//!   buffer(0): A — FP32 activations [M, K]
//!   buffer(1): W — packed INT4/INT8 weights [N, K/pack_factor] as uint8/uint32
//!   buffer(2): scales — FP32 per-group scales [N, K/group_size]
//!   buffer(3): zeros  — I32 per-group zero-points [N, K/group_size]
//!   buffer(4): C — FP32 output [M, N]
//!
//! Kernel structure (32x32 tiled):
//!   1. Each threadgroup handles a TILE_M x TILE_N output tile
//!   2. For each K-tile:
//!      a. Load activation tile A[tile_m..tile_m+TILE_M][k..k+TILE_K] to threadgroup
//!      b. Load weight tile: unpack INT4/INT8 → half, apply (q - zp) * scale
//!      c. simdgroup_matrix MAC into float accumulators
//!   3. Write accumulated output tile to C
//! ```
//!
//! Part of #3869

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::quantized::{quantized_matmul, QuantizedTensor};
use nn_core::{Device, Result, TensorError};

impl super::MetalDynBackend {
    /// Quantized matmul on Metal: `input @ weight^T` where `weight` is quantized.
    ///
    /// Input: FP32 DynTensor on GPU with shape `[*, K]`.
    /// Weight: [`QuantizedTensor`] (INT4 or INT8, per-group scales + zeros).
    /// Output: FP32 DynTensor on GPU with shape `[*, N]`.
    ///
    /// Phase 1: CPU fallback — reads input to CPU, calls `quantized_matmul`,
    /// uploads result. Correct but not memory-optimal for large VLMs.
    ///
    /// Phase 2 (future): Native MSL kernel with on-the-fly dequantization.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Input's last dimension doesn't match `weight.in_features()`
    /// - Input is not F32 dtype
    /// - GPU transfer fails
    pub(crate) fn gpu_quantized_matmul(
        input: &DynTensor,
        weight: &QuantizedTensor,
    ) -> Result<DynTensor> {
        // Validate input is F32 (quantized matmul operates in W4A16/W8A16 mode)
        if input.dtype() != nn_core::DType::F32 {
            return Err(TensorError::dtype_mismatch(
                nn_core::DType::F32,
                input.dtype(),
            ));
        }

        // Validate shape compatibility
        let input_dims = input.dims();
        if input_dims.is_empty() {
            return Err(TensorError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        let x_last = input_dims[input_dims.len() - 1];
        if x_last != weight.in_features() {
            return Err(TensorError::shape_mismatch(
                vec![weight.in_features()],
                vec![x_last],
            ));
        }

        // Phase 1: CPU fallback path.
        // Flush pending GPU work before CPU readback (#2009).
        crate::gpu_scope::flush()?;

        // Transfer input to CPU for quantized matmul
        let input_cpu = input.to_device(&Device::Cpu)?;

        // Compute quantized matmul on CPU (dequantize-then-matmul)
        let result_cpu = quantized_matmul(&input_cpu, weight)?;

        // Transfer result back to GPU
        result_cpu.to_device(&Device::metal())
    }
}

#[cfg(test)]
#[path = "dyn_tensor_metal_quantized_matmul_tests.rs"]
mod tests;
