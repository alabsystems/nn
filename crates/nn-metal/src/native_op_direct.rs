// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Direct GPU dispatch for NativeOp steps — bypasses DynTensor intermediates.
//!
//! The current NativeOp execution path converts `GpuSlice → DynTensor`
//! (via `slice_to_dyn`), runs eager ops, then converts back (`dyn_to_slice`).
//! Each conversion allocates an `Arc<MetalTensorData>` + `Vec<usize>` shape
//! and extracts the buffer back. For simple elementwise ops like SiluMul,
//! this bridge overhead dominates the actual GPU work.
//!
//! `DirectDispatch` dispatches Metal kernels directly on `GpuSlice` buffers:
//! no DynTensor wrapping, no shape allocation, no gpu_data extraction.
//!
//! Part of #3472 (NativeOp DynTensor bridge elimination).

use nn_core::{Result, TensorError};
use nn_dsl::ir::ScalarType;
use nn_dsl::NativeOpKind;

use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::compiled_model::CompiledModelError;
use crate::dispatch_plan::DispatchMode;
use crate::gpu_slice::GpuSlice;
use crate::kernel_dispatch::KernelPipeline;

/// Trait for NativeOp variants that can dispatch directly on GPU buffers
/// without DynTensor conversion overhead.
///
/// Implementors generate/cache an MSL kernel and dispatch it with raw
/// buffer + offset bindings. The trait is `pub(crate)` — only used
/// internally by the compiled model executor.
pub(crate) trait DirectDispatch {
    /// Dispatch the operation directly on GPU buffers.
    ///
    /// # Arguments
    ///
    /// * `inputs` — Input `GpuSlice` buffers (gate, up, etc.)
    /// * `output` — Pre-allocated output `GpuSlice` (buffer + offset)
    /// * `num_elements` — Total element count for the dispatch
    /// * `scalar_type` — Element type (F32/F16/BF16)
    /// * `cache` — Pipeline cache for MSL compilation
    ///
    /// # Errors
    ///
    /// Returns `Err` if MSL compilation or Metal dispatch fails.
    fn dispatch_direct(
        &self,
        inputs: &[&GpuSlice],
        output: &GpuSlice,
        num_elements: usize,
        scalar_type: ScalarType,
        cache: &PipelineCache,
    ) -> Result<()>;

    /// Number of output bytes for this operation.
    fn output_bytes(&self, num_elements: usize, scalar_type: ScalarType) -> usize;

    /// Whether this implementation can handle the given scalar type.
    fn supports_scalar_type(&self, scalar_type: ScalarType) -> bool;
}

/// Check whether a `NativeOpKind` has a direct dispatch implementation.
///
/// Returns `true` for ops where the DynTensor bridge can be bypassed.
/// Used by the executor to decide between direct and bridge paths.
pub(crate) fn can_use_direct_dispatch(op: &NativeOpKind) -> bool {
    matches!(
        op,
        NativeOpKind::SiluMul { .. }
            | NativeOpKind::FusedMulAdd { .. }
            | NativeOpKind::FusedSiGLU { .. }
            | NativeOpKind::FusedGeGLU { .. }
    )
}

/// Direct dispatch for `NativeOpKind::SiluMul`.
///
/// Generates a fused `silu(gate) * up` MSL kernel and dispatches it
/// directly on the input GpuSlice buffers. Eliminates:
/// - 2x `slice_to_dyn` (Arc<MetalTensorData> + Vec<usize> alloc each)
/// - `silu()` dispatch (intermediate buffer + DynTensor)
/// - `mul()` dispatch (second intermediate buffer + DynTensor)
/// - 1x `dyn_to_slice` (gpu_data extraction)
///
/// Replaces 3 DynTensor allocations + 2 Metal dispatches with
/// 0 DynTensor allocations + 1 Metal dispatch.
pub(crate) struct SiluMulDirect;

impl DirectDispatch for SiluMulDirect {
    fn dispatch_direct(
        &self,
        inputs: &[&GpuSlice],
        output: &GpuSlice,
        num_elements: usize,
        scalar_type: ScalarType,
        cache: &PipelineCache,
    ) -> Result<()> {
        if inputs.len() != 2 {
            return Err(TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: 0,
                reason: format!(
                    "SiluMulDirect: expected 2 inputs (gate, up), got {}",
                    inputs.len()
                ),
            }));
        }
        if num_elements == 0 {
            return Ok(());
        }

        let gate = inputs[0];
        let up = inputs[1];

        // Generate MSL kernel name and source keyed by scalar type.
        // PipelineCache deduplicates by MSL source hash, so identical
        // scalar types share a single compiled pipeline.
        let type_str = scalar_type.msl_str();
        let kernel_name = format!("silu_mul_direct_{type_str}");
        let msl_source = generate_silu_mul_msl(&kernel_name, scalar_type);

        // Compile or retrieve cached pipeline (2 input buffers: gate, up).
        let pipeline = KernelPipeline::from_msl(cache, &msl_source, &kernel_name, 2, false)
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("SiluMulDirect pipeline: {e}"),
                })
            })?;

        // Build elementwise dispatch plan.
        let total_u32 = u32::try_from(num_elements).map_err(|_| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: 0,
                reason: "SiluMulDirect: element count exceeds u32".into(),
            })
        })?;
        let plan = DispatchMode::Elementwise { total: total_u32 }
            .plan()
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("SiluMulDirect plan: {e}"),
                })
            })?;

        // Bind buffers with offsets and dispatch.
        let input_bufs: Vec<&MetalBuffer> = vec![gate.buffer(), up.buffer()];
        let input_offsets = vec![gate.byte_offset(), up.byte_offset()];

        pipeline
            .dispatch_buffers_with_all_offsets(
                cache.context(),
                &input_bufs,
                &input_offsets,
                output.buffer(),
                output.byte_offset(),
                &plan,
            )
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("SiluMulDirect dispatch: {e}"),
                })
            })?;

        Ok(())
    }

    fn output_bytes(&self, num_elements: usize, scalar_type: ScalarType) -> usize {
        num_elements.saturating_mul(scalar_type.byte_size())
    }

    fn supports_scalar_type(&self, scalar_type: ScalarType) -> bool {
        matches!(scalar_type, ScalarType::F32 | ScalarType::F16)
    }
}

/// Generate MSL source for a fused `silu(gate) * up` elementwise kernel.
///
/// Layout follows the standard nn-dsl kernel convention:
/// - `buffer(0)`: gate input (read-only)
/// - `buffer(1)`: up input (read-only)
/// - `buffer(2)`: output (write-only)
/// - `buffer(3)`: element count (constant uint)
fn generate_silu_mul_msl(kernel_name: &str, scalar_type: ScalarType) -> String {
    let st = scalar_type.msl_str();
    // F16 inputs accumulate in F32 for precision, then cast back.
    let needs_upcast = matches!(scalar_type, ScalarType::F16 | ScalarType::BF16);
    let (load_cast_open, load_cast_close) = if needs_upcast {
        ("float(", ")")
    } else {
        ("", "")
    };
    let store_cast = if needs_upcast {
        format!("{st}(")
    } else {
        String::new()
    };
    let store_close = if needs_upcast { ")" } else { "" };

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const {st}* gate [[buffer(0)]],
    device const {st}* up   [[buffer(1)]],
    device {st}* output      [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= count) return;
    float g = {load_cast_open}gate[tid]{load_cast_close};
    float u = {load_cast_open}up[tid]{load_cast_close};
    float sigmoid_g = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sigmoid_g;
    output[tid] = {store_cast}silu_g * u{store_close};
}}
"#
    )
}

// ---------------------------------------------------------------------------
// FusedMulAdd: a * b + c in a single Metal dispatch.
// ---------------------------------------------------------------------------

/// Direct dispatch for `NativeOpKind::FusedMulAdd`.
///
/// Generates a fused `a * b + c` MSL kernel using hardware FMA. 3 inputs
/// (a, b, c), 1 output. Replaces 2 dispatches (Mul + Add) with 1.
///
/// Part of #4252, #4431.
pub(crate) struct FusedMulAddDirect;

impl DirectDispatch for FusedMulAddDirect {
    fn dispatch_direct(
        &self,
        inputs: &[&GpuSlice],
        output: &GpuSlice,
        num_elements: usize,
        scalar_type: ScalarType,
        cache: &PipelineCache,
    ) -> Result<()> {
        if inputs.len() != 3 {
            return Err(TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: 0,
                reason: format!(
                    "FusedMulAddDirect: expected 3 inputs (a, b, c), got {}",
                    inputs.len()
                ),
            }));
        }
        if num_elements == 0 {
            return Ok(());
        }

        let a = inputs[0];
        let b = inputs[1];
        let c = inputs[2];

        let type_str = scalar_type.msl_str();
        let kernel_name = format!("fused_mul_add_direct_{type_str}");
        let msl_source = generate_fused_mul_add_msl(&kernel_name, scalar_type);

        // 3 input buffers: a, b, c.
        let pipeline = KernelPipeline::from_msl(cache, &msl_source, &kernel_name, 3, false)
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedMulAddDirect pipeline: {e}"),
                })
            })?;

        let total_u32 = u32::try_from(num_elements).map_err(|_| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: 0,
                reason: "FusedMulAddDirect: element count exceeds u32".into(),
            })
        })?;
        let plan = DispatchMode::Elementwise { total: total_u32 }
            .plan()
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedMulAddDirect plan: {e}"),
                })
            })?;

        let input_bufs: Vec<&MetalBuffer> = vec![a.buffer(), b.buffer(), c.buffer()];
        let input_offsets = vec![a.byte_offset(), b.byte_offset(), c.byte_offset()];

        pipeline
            .dispatch_buffers_with_all_offsets(
                cache.context(),
                &input_bufs,
                &input_offsets,
                output.buffer(),
                output.byte_offset(),
                &plan,
            )
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedMulAddDirect dispatch: {e}"),
                })
            })?;

        Ok(())
    }

    fn output_bytes(&self, num_elements: usize, scalar_type: ScalarType) -> usize {
        num_elements.saturating_mul(scalar_type.byte_size())
    }

    fn supports_scalar_type(&self, scalar_type: ScalarType) -> bool {
        matches!(scalar_type, ScalarType::F32 | ScalarType::F16)
    }
}

/// Generate MSL source for fused `a * b + c` using hardware FMA.
///
/// - `buffer(0)`: a (read-only)
/// - `buffer(1)`: b (read-only)
/// - `buffer(2)`: c (read-only)
/// - `buffer(3)`: output (write-only)
/// - `buffer(4)`: element count (constant uint)
fn generate_fused_mul_add_msl(kernel_name: &str, scalar_type: ScalarType) -> String {
    let st = scalar_type.msl_str();
    let needs_upcast = matches!(scalar_type, ScalarType::F16 | ScalarType::BF16);
    let (lc_o, lc_c) = if needs_upcast {
        ("float(", ")")
    } else {
        ("", "")
    };
    let store_cast = if needs_upcast {
        format!("{st}(")
    } else {
        String::new()
    };
    let store_close = if needs_upcast { ")" } else { "" };

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const {st}* a    [[buffer(0)]],
    device const {st}* b    [[buffer(1)]],
    device const {st}* c    [[buffer(2)]],
    device {st}* output      [[buffer(3)]],
    constant uint& count     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= count) return;
    float va = {lc_o}a[tid]{lc_c};
    float vb = {lc_o}b[tid]{lc_c};
    float vc = {lc_o}c[tid]{lc_c};
    output[tid] = {store_cast}fma(va, vb, vc){store_close};
}}
"#
    )
}

// ---------------------------------------------------------------------------
// FusedSiGLU: sigmoid(x) * x (SiLU/Swish) on a single input.
// ---------------------------------------------------------------------------

/// Direct dispatch for `NativeOpKind::FusedSiGLU`.
///
/// Generates a fused `x * sigmoid(x)` MSL kernel. 1 input, 1 output.
/// Replaces 2 dispatches (Sigmoid + Mul) with 1.
///
/// Part of #4252, #4431.
pub(crate) struct FusedSiGLUDirect;

impl DirectDispatch for FusedSiGLUDirect {
    fn dispatch_direct(
        &self,
        inputs: &[&GpuSlice],
        output: &GpuSlice,
        num_elements: usize,
        scalar_type: ScalarType,
        cache: &PipelineCache,
    ) -> Result<()> {
        if inputs.len() != 1 {
            return Err(TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: 0,
                reason: format!(
                    "FusedSiGLUDirect: expected 1 input (x), got {}",
                    inputs.len()
                ),
            }));
        }
        if num_elements == 0 {
            return Ok(());
        }

        let x = inputs[0];

        let type_str = scalar_type.msl_str();
        let kernel_name = format!("fused_siglu_direct_{type_str}");
        let msl_source = generate_fused_siglu_msl(&kernel_name, scalar_type);

        // 1 input buffer.
        let pipeline = KernelPipeline::from_msl(cache, &msl_source, &kernel_name, 1, false)
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedSiGLUDirect pipeline: {e}"),
                })
            })?;

        let total_u32 = u32::try_from(num_elements).map_err(|_| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: 0,
                reason: "FusedSiGLUDirect: element count exceeds u32".into(),
            })
        })?;
        let plan = DispatchMode::Elementwise { total: total_u32 }
            .plan()
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedSiGLUDirect plan: {e}"),
                })
            })?;

        let input_bufs: Vec<&MetalBuffer> = vec![x.buffer()];
        let input_offsets = vec![x.byte_offset()];

        pipeline
            .dispatch_buffers_with_all_offsets(
                cache.context(),
                &input_bufs,
                &input_offsets,
                output.buffer(),
                output.byte_offset(),
                &plan,
            )
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedSiGLUDirect dispatch: {e}"),
                })
            })?;

        Ok(())
    }

    fn output_bytes(&self, num_elements: usize, scalar_type: ScalarType) -> usize {
        num_elements.saturating_mul(scalar_type.byte_size())
    }

    fn supports_scalar_type(&self, scalar_type: ScalarType) -> bool {
        matches!(scalar_type, ScalarType::F32 | ScalarType::F16)
    }
}

/// Generate MSL source for fused `x * sigmoid(x)` (SiLU/Swish).
///
/// - `buffer(0)`: x (read-only)
/// - `buffer(1)`: output (write-only)
/// - `buffer(2)`: element count (constant uint)
fn generate_fused_siglu_msl(kernel_name: &str, scalar_type: ScalarType) -> String {
    let st = scalar_type.msl_str();
    let needs_upcast = matches!(scalar_type, ScalarType::F16 | ScalarType::BF16);
    let (lc_o, lc_c) = if needs_upcast {
        ("float(", ")")
    } else {
        ("", "")
    };
    let store_cast = if needs_upcast {
        format!("{st}(")
    } else {
        String::new()
    };
    let store_close = if needs_upcast { ")" } else { "" };

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const {st}* x    [[buffer(0)]],
    device {st}* output      [[buffer(1)]],
    constant uint& count     [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= count) return;
    float val = {lc_o}x[tid]{lc_c};
    float sigmoid_val = 1.0f / (1.0f + exp(-val));
    output[tid] = {store_cast}val * sigmoid_val{store_close};
}}
"#
    )
}

// ---------------------------------------------------------------------------
// FusedGeGLU: gelu(gate) * up in a single Metal dispatch.
// ---------------------------------------------------------------------------

/// Direct dispatch for `NativeOpKind::FusedGeGLU`.
///
/// Generates a fused `gelu(gate) * up` MSL kernel. 2 inputs (gate, up),
/// 1 output. Replaces 2 dispatches (GELU + Mul) with 1. Used in Qwen3/GLM5
/// MLP blocks.
///
/// Part of #4252, #4431.
pub(crate) struct FusedGeGLUDirect;

impl DirectDispatch for FusedGeGLUDirect {
    fn dispatch_direct(
        &self,
        inputs: &[&GpuSlice],
        output: &GpuSlice,
        num_elements: usize,
        scalar_type: ScalarType,
        cache: &PipelineCache,
    ) -> Result<()> {
        if inputs.len() != 2 {
            return Err(TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: 0,
                reason: format!(
                    "FusedGeGLUDirect: expected 2 inputs (gate, up), got {}",
                    inputs.len()
                ),
            }));
        }
        if num_elements == 0 {
            return Ok(());
        }

        let gate = inputs[0];
        let up = inputs[1];

        let type_str = scalar_type.msl_str();
        let kernel_name = format!("fused_geglu_direct_{type_str}");
        let msl_source = generate_fused_geglu_msl(&kernel_name, scalar_type);

        // 2 input buffers: gate, up.
        let pipeline = KernelPipeline::from_msl(cache, &msl_source, &kernel_name, 2, false)
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedGeGLUDirect pipeline: {e}"),
                })
            })?;

        let total_u32 = u32::try_from(num_elements).map_err(|_| {
            TensorError::from(CompiledModelError::DispatchFailed {
                step_idx: 0,
                reason: "FusedGeGLUDirect: element count exceeds u32".into(),
            })
        })?;
        let plan = DispatchMode::Elementwise { total: total_u32 }
            .plan()
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedGeGLUDirect plan: {e}"),
                })
            })?;

        let input_bufs: Vec<&MetalBuffer> = vec![gate.buffer(), up.buffer()];
        let input_offsets = vec![gate.byte_offset(), up.byte_offset()];

        pipeline
            .dispatch_buffers_with_all_offsets(
                cache.context(),
                &input_bufs,
                &input_offsets,
                output.buffer(),
                output.byte_offset(),
                &plan,
            )
            .map_err(|e| {
                TensorError::from(CompiledModelError::DispatchFailed {
                    step_idx: 0,
                    reason: format!("FusedGeGLUDirect dispatch: {e}"),
                })
            })?;

        Ok(())
    }

    fn output_bytes(&self, num_elements: usize, scalar_type: ScalarType) -> usize {
        num_elements.saturating_mul(scalar_type.byte_size())
    }

    fn supports_scalar_type(&self, scalar_type: ScalarType) -> bool {
        matches!(scalar_type, ScalarType::F32 | ScalarType::F16)
    }
}

/// Generate MSL source for fused `gelu(gate) * up`.
///
/// Uses the fast GELU approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
///
/// - `buffer(0)`: gate (read-only)
/// - `buffer(1)`: up (read-only)
/// - `buffer(2)`: output (write-only)
/// - `buffer(3)`: element count (constant uint)
fn generate_fused_geglu_msl(kernel_name: &str, scalar_type: ScalarType) -> String {
    let st = scalar_type.msl_str();
    let needs_upcast = matches!(scalar_type, ScalarType::F16 | ScalarType::BF16);
    let (lc_o, lc_c) = if needs_upcast {
        ("float(", ")")
    } else {
        ("", "")
    };
    let store_cast = if needs_upcast {
        format!("{st}(")
    } else {
        String::new()
    };
    let store_close = if needs_upcast { ")" } else { "" };

    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const {st}* gate [[buffer(0)]],
    device const {st}* up   [[buffer(1)]],
    device {st}* output      [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {{
    if (tid >= count) return;
    float g = {lc_o}gate[tid]{lc_c};
    float u = {lc_o}up[tid]{lc_c};
    // Fast GELU approximation (Hendrycks & Gimpel, 2016).
    float k = 0.7978845608f; // sqrt(2/pi)
    float gelu_g = 0.5f * g * (1.0f + tanh(k * (g + 0.044715f * g * g * g)));
    output[tid] = {store_cast}gelu_g * u{store_close};
}}
"#
    )
}

// Direct dispatch is wired into `CompiledModel::execute_native_op` in
// `compiled_model_execute_native.rs`. The executor resolves input slices
// (which requires `pub(super)` access to `resolve_input_slice`) and then
// calls the DirectDispatch implementation directly. Part of #3537, #4252.

#[cfg(test)]
#[path = "native_op_direct_tests.rs"]
mod tests;
