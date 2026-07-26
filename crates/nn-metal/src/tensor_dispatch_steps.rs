// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-step dispatch execution for tensor Metal pipeline.
//!
//! Extracted from `tensor_dispatch.rs` (#827 Direction 3) to keep the
//! executor under 400 lines. Contains the match dispatch over
//! `DispatchStep` variants.

use std::collections::HashMap;

use nn_dsl::{DispatchStep, TensorKernelDef, TensorNodeId, MAX_DIRECT_BINDING_INPUTS};

use super::helpers::{
    encode_elementwise_step, encode_reduce, encode_tiled_transpose_step, output_elems,
    EncodeContext,
};
use super::TensorDispatchError;
use crate::buffer::MetalBuffer;
use crate::cache::PipelineCache;
use crate::context::MetalContext;
use crate::dispatch::CommandBatch;

#[path = "tensor_dispatch_steps_conv.rs"]
mod conv;

#[path = "tensor_dispatch_steps_ops.rs"]
mod ops;

/// Execute a single dispatch step: encode it into the command batch.
///
/// `elem_size` is the byte size of the element type being dispatched
/// (4 for f32, 2 for f16/bf16). Passed through to helpers for buffer
/// allocation sizing.
///
/// Returns `Ok(())` on success. Buffer allocations and GPU encoding are
/// performed as side effects via `buffers` and `batch`.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_one_step(
    step: &DispatchStep,
    cache: &PipelineCache,
    batch: &CommandBatch,
    ctx: &MetalContext,
    combined_msl: &str,
    kernel: &TensorKernelDef,
    elem_size: usize,
    buffers: &mut HashMap<TensorNodeId, MetalBuffer>,
    offsets: &mut HashMap<TensorNodeId, usize>,
) -> Result<(), TensorDispatchError> {
    let mut enc = EncodeContext {
        cache,
        batch,
        ctx,
        elem_size,
        buffers,
        offsets,
    };

    match step {
        DispatchStep::Reduce {
            kernel_name,
            input,
            output,
            reduce_dim,
            outer_size,
            ..
        } => {
            encode_reduce(
                &mut enc,
                combined_msl,
                kernel_name,
                *input,
                *output,
                *outer_size,
                *reduce_dim,
            )?;
        }

        DispatchStep::Elementwise {
            kernel_name,
            inputs: step_inputs,
            output,
            total_elements,
            ..
        } => {
            if step_inputs.len() > MAX_DIRECT_BINDING_INPUTS {
                // Packed dispatch: all inputs in one buffer via offsets.
                // The packed kernel name has `_packed_kernel` suffix instead of
                // `_kernel`. Part of #1649.
                let packed_name = kernel_name.replace("_kernel", "_packed_kernel");
                super::packed::encode_packed_elementwise_step(
                    cache,
                    batch,
                    ctx,
                    combined_msl,
                    &packed_name,
                    step_inputs,
                    *output,
                    *total_elements,
                    elem_size,
                    enc.buffers,
                    enc.offsets,
                )?;
            } else {
                encode_elementwise_step(
                    &mut enc,
                    combined_msl,
                    kernel_name,
                    step_inputs,
                    *output,
                    *total_elements,
                )?;
            }
        }

        DispatchStep::Broadcast {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        } => {
            encode_elementwise_step(
                &mut enc,
                combined_msl,
                kernel_name,
                &[*input],
                *output,
                *total_elements,
            )?;
        }

        DispatchStep::Reshape { input, output } => {
            // Reshape is a logical shape change — the data is unchanged.
            // Use buffer aliasing (ARC refcount increment) instead of CPU
            // clone_buffer, because clone_buffer does a CPU-side memcpy that
            // reads stale zeros from not-yet-committed GPU output buffers
            // in the same command batch.
            let src = enc
                .buffers
                .get(input)
                .ok_or(TensorDispatchError::MissingBuffer(*input))?;
            let buf = src.alias();
            enc.buffers.insert(*output, buf);
            // Propagate byte offset through reshape aliases (#1945).
            if let Some(&off) = enc.offsets.get(input) {
                enc.offsets.insert(*output, off);
            }
        }

        // Shape-inferred unary ops: compute output elements from node shape.
        DispatchStep::AxisSelect {
            kernel_name,
            input,
            output,
            ..
        }
        | DispatchStep::Narrow {
            kernel_name,
            input,
            output,
            ..
        } => {
            let elems = output_elems(kernel, *output)?;
            encode_elementwise_step(
                &mut enc,
                combined_msl,
                kernel_name,
                &[*input],
                *output,
                elems,
            )?;
        }

        // Transpose: tiled shared-memory dispatch for qualifying 2D transposes,
        // fallback to naive elementwise for other permutations. Part of #3230 (Gap 4).
        DispatchStep::Transpose {
            kernel_name,
            input,
            output,
            ..
        } => {
            if let Some((batch_size, rows, cols)) = step.tiled_transpose_params() {
                encode_tiled_transpose_step(
                    &mut enc,
                    combined_msl,
                    kernel_name,
                    *input,
                    *output,
                    batch_size,
                    rows,
                    cols,
                )?;
            } else {
                let elems = output_elems(kernel, *output)?;
                encode_elementwise_step(
                    &mut enc,
                    combined_msl,
                    kernel_name,
                    &[*input],
                    *output,
                    elems,
                )?;
            }
        }

        DispatchStep::Stack {
            kernel_name,
            inputs: step_inputs,
            output,
            input_shape,
            ..
        } => {
            let elems = output_elems(kernel, *output)?;
            if step_inputs.len() > MAX_DIRECT_BINDING_INPUTS {
                super::packed::encode_packed_stack_step(
                    cache,
                    batch,
                    ctx,
                    combined_msl,
                    kernel_name,
                    step_inputs,
                    *output,
                    input_shape,
                    elems,
                    elem_size,
                    enc.buffers,
                    enc.offsets,
                )?;
            } else {
                encode_elementwise_step(
                    &mut enc,
                    combined_msl,
                    kernel_name,
                    step_inputs,
                    *output,
                    elems,
                )?;
            }
        }

        DispatchStep::Concat {
            kernel_name,
            inputs: step_inputs,
            output,
            first_input_shape,
            input_axis_sizes,
            axis,
            ..
        } => {
            let elems = output_elems(kernel, *output)?;
            if step_inputs.len() > MAX_DIRECT_BINDING_INPUTS {
                super::packed::encode_packed_concat_step(
                    cache,
                    batch,
                    ctx,
                    combined_msl,
                    kernel_name,
                    step_inputs,
                    *output,
                    first_input_shape,
                    input_axis_sizes,
                    *axis,
                    elems,
                    elem_size,
                    enc.buffers,
                    enc.offsets,
                )?;
            } else {
                encode_elementwise_step(
                    &mut enc,
                    combined_msl,
                    kernel_name,
                    step_inputs,
                    *output,
                    elems,
                )?;
            }
        }

        // Conv-like ops, Linear, Embedding, and selection: delegated to
        // tensor_dispatch_steps_conv.rs.
        DispatchStep::Conv1d(..)
        | DispatchStep::Conv2d(..)
        | DispatchStep::ConvTranspose1d(..)
        | DispatchStep::Linear { .. }
        | DispatchStep::Embedding { .. }
        | DispatchStep::IndexSelect { .. }
        | DispatchStep::Gather { .. } => {
            return conv::dispatch_conv_or_embedding(step, &mut enc, combined_msl).ok_or(
                TensorDispatchError::UnsupportedStep {
                    step: format!("{step:?}"),
                },
            )?;
        }

        // Simdgroup-tiled GEMM: uses 3D threadgroup dispatch. Part of #2275.
        DispatchStep::SimdgroupLinear(ref p) => {
            let mut step_inputs = vec![p.input, p.weight];
            if let Some(b) = p.bias {
                step_inputs.push(b);
            }
            super::helpers::encode_simdgroup_gemm(
                &mut enc,
                combined_msl,
                &p.kernel_name,
                &step_inputs,
                p.output,
                p.batch_size,
                p.out_features,
                1, // batch_count=1, batch dims folded into M
            )?;
        }

        DispatchStep::SimdgroupMatMul(ref p) => {
            super::helpers::encode_simdgroup_gemm(
                &mut enc,
                combined_msl,
                &p.kernel_name,
                &[p.left, p.right],
                p.output,
                p.m,
                p.n,
                p.batch_size,
            )?;
        }

        // Tiled shared-memory GEMM: uses 3D threadgroup dispatch. Part of #3230 (Gap 1).
        DispatchStep::TiledLinear(ref p) => {
            let mut step_inputs = vec![p.input, p.weight];
            if let Some(b) = p.bias {
                step_inputs.push(b);
            }
            super::helpers::encode_tiled_gemm(
                &mut enc,
                combined_msl,
                &p.kernel_name,
                &step_inputs,
                p.output,
                p.batch_size,
                p.out_features,
                1, // batch_count=1, batch dims folded into M
            )?;
        }

        DispatchStep::TiledMatMul(ref p) => {
            super::helpers::encode_tiled_gemm(
                &mut enc,
                combined_msl,
                &p.kernel_name,
                &[p.left, p.right],
                p.output,
                p.m,
                p.n,
                p.batch_size,
            )?;
        }

        // Binary ops, unary activations, ZeroPad1d, Softmax:
        // delegated to tensor_dispatch_steps_ops.rs.
        step => {
            if let Some(result) =
                ops::dispatch_binary_unary_or_misc(step, &mut enc, combined_msl, kernel)
            {
                result?;
            } else {
                return Err(TensorDispatchError::UnsupportedStep {
                    step: format!("{step:?}"),
                });
            }
        }
    }

    Ok(())
}
