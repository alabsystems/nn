// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! SageAttention GPU dispatch for Metal.
//!
//! Provides GPU-accelerated SageAttention (Zhang et al., 2024; arXiv:2410.02367)
//! for document VLM inference workloads. The current implementation uses a CPU
//! fallback path that leverages the verified CPU reference in `nn-core`. A
//! future native MSL implementation will use a 4-kernel strategy:
//!
//! - **Kernel 1:** INT8 per-head absmax quantization of Q and K
//! - **Kernel 2:** INT8 GEMM for attention scores (reuse `int8_gemm_msl`)
//! - **Kernel 3:** Softmax over scores (reuse existing fused softmax MSL)
//! - **Kernel 4:** FP32 PV accumulation via matmul (reuse simdgroup matmul)
//!
//! The CPU fallback path reads Q/K/V from GPU to CPU, runs the verified
//! `SageAttention::forward`, and uploads the result back to GPU. This ensures
//! correctness while the native MSL kernels are developed.
//!
//! Part of #3871 — Metal GPU SageAttention kernel for document VLM inference.

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::attention::{SageAttention, SageAttentionConfig};
use nn_core::{Device, Result, TensorError};

/// GPU-accelerated SageAttention dispatch.
///
/// Wraps [`SageAttentionConfig`] and provides a `forward()` method that
/// dispatches on Metal GPU tensors. Currently uses CPU fallback; native MSL
/// kernels are planned (see module docs).
#[derive(Debug, Clone)]
pub(crate) struct SageAttentionGpu {
    config: SageAttentionConfig,
    /// CPU reference implementation for fallback path.
    cpu_impl: SageAttention,
}

impl SageAttentionGpu {
    /// Create a new GPU SageAttention instance. Validates configuration.
    pub(crate) fn new(config: SageAttentionConfig) -> Result<Self> {
        let cpu_impl = SageAttention::new(config)?;
        Ok(Self { config, cpu_impl })
    }

    /// Access the underlying configuration.
    pub(crate) fn config(&self) -> &SageAttentionConfig {
        &self.config
    }

    /// Run SageAttention on GPU tensors.
    ///
    /// # Arguments
    ///
    /// - `q`: query tensor `[B, num_heads, S_q, head_dim]` on Metal GPU
    /// - `k`: key tensor `[B, num_kv_heads, S_kv, head_dim]` on Metal GPU
    /// - `v`: value tensor `[B, num_kv_heads, S_kv, head_dim]` on Metal GPU
    ///
    /// # Returns
    ///
    /// Attention output `[B, num_heads, S_q, head_dim]` on Metal GPU.
    ///
    /// # Implementation
    ///
    /// Currently uses CPU fallback: GPU → CPU → SageAttention::forward → GPU.
    /// Native MSL dispatch is planned via the 4-kernel strategy documented in
    /// the module-level docs.
    pub(crate) fn forward(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        v: &DynTensor,
    ) -> Result<DynTensor> {
        // Validate that inputs are on GPU.
        if !q.device().is_gpu() || !k.device().is_gpu() || !v.device().is_gpu() {
            return Err(TensorError::InvalidShape(
                "SageAttentionGpu: all inputs must be on GPU device".to_string(),
            ));
        }

        // CPU fallback path: read GPU tensors to CPU, run verified reference,
        // upload result back to GPU.
        //
        // GpuScope must NOT wrap this function — it does CPU readback.
        // (nn_engineering.md: "GpuScope must NOT wrap functions that do CPU readback.")
        let q_cpu = q.to_device(&Device::Cpu)?;
        let k_cpu = k.to_device(&Device::Cpu)?;
        let v_cpu = v.to_device(&Device::Cpu)?;

        let output_cpu = self.cpu_impl.forward(&q_cpu, &k_cpu, &v_cpu)?;

        // Upload result back to GPU.
        output_cpu.to_device(&Device::metal())
    }
}

#[cfg(test)]
#[path = "dyn_tensor_metal_sage_attn_tests.rs"]
mod tests;
