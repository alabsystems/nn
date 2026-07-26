// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv-like and embedding dispatch steps for tensor Metal pipeline.
//!
//! Extracted from `tensor_dispatch_steps.rs` (#1575) to keep files under 400 lines.
//! Contains dispatch arms for Conv1d, Conv2d, ConvTranspose1d, Linear, and Embedding.

use nn_dsl::{DispatchStep, TensorNodeId};

use super::super::helpers::{
    encode_conv_like_step, encode_elementwise_step, to_dispatch_u32, EncodeContext,
};
use super::super::TensorDispatchError;
use crate::arena::arena_alloc_or_create;
use crate::dispatch_plan::DispatchMode;
use crate::error::MetalError;

/// Dispatch a conv-like or embedding step.
///
/// Returns `Some(Ok(()))` if the step was handled, `Some(Err(...))` on error,
/// or `None` if the step is not a conv/linear/embedding variant.
pub(super) fn dispatch_conv_or_embedding(
    step: &DispatchStep,
    enc: &mut EncodeContext<'_>,
    combined_msl: &str,
) -> Option<Result<(), TensorDispatchError>> {
    match step {
        DispatchStep::Conv1d(ref p) => Some(encode_conv_like_step(
            enc,
            combined_msl,
            &p.kernel_name,
            p.input,
            p.weight,
            p.bias,
            p.output,
            p.total_elements,
        )),
        DispatchStep::Conv2d(ref p) => Some(encode_conv_like_step(
            enc,
            combined_msl,
            &p.kernel_name,
            p.input,
            p.weight,
            p.bias,
            p.output,
            p.total_elements,
        )),
        DispatchStep::ConvTranspose1d(ref p) => Some(encode_conv_like_step(
            enc,
            combined_msl,
            &p.kernel_name,
            p.input,
            p.weight,
            p.bias,
            p.output,
            p.total_elements,
        )),
        DispatchStep::Linear {
            kernel_name,
            input,
            weight,
            bias,
            output,
            total_elements,
            ..
        } => Some(encode_conv_like_step(
            enc,
            combined_msl,
            kernel_name,
            *input,
            *weight,
            *bias,
            *output,
            *total_elements,
        )),
        DispatchStep::Embedding {
            kernel_name,
            input,
            weight,
            output,
            total_elements,
            ..
        } => Some(encode_elementwise_step(
            enc,
            combined_msl,
            kernel_name,
            &[*input, *weight],
            *output,
            *total_elements,
        )),
        DispatchStep::IndexSelect {
            kernel_name,
            input,
            indices,
            output,
            total_elements,
            ..
        }
        | DispatchStep::Gather {
            kernel_name,
            input,
            indices,
            output,
            total_elements,
            ..
        } => Some(encode_index_select_with_conv(
            enc,
            combined_msl,
            kernel_name,
            *input,
            *indices,
            *output,
            *total_elements,
        )),
        _ => None,
    }
}

/// Encode an IndexSelect/Gather step with f32→u32 index conversion.
///
/// The MSL kernel reads `uint*` indices for precision (Part of #2278).
/// The compiled pipeline stores indices as f32 buffers. This function:
/// 1. Runs the `{name}_f32_to_u32` conversion kernel on the indices
/// 2. Runs the main IndexSelect/Gather kernel with the u32 indices buffer
fn encode_index_select_with_conv(
    enc: &mut EncodeContext<'_>,
    msl: &str,
    kernel_name: &str,
    input: TensorNodeId,
    indices: TensorNodeId,
    output: TensorNodeId,
    total_elements: usize,
) -> Result<(), TensorDispatchError> {
    // Step 1: Convert f32 indices to u32 via the companion kernel.
    let conv_name = format!("{kernel_name}_f32_to_u32");
    let (idx_buf, idx_offset) = enc.input(indices)?;

    // Index buffer element count: byte_length / sizeof(f32).
    let idx_numel = idx_buf.len().saturating_sub(idx_offset) / 4;
    let idx_u32 = to_dispatch_u32(idx_numel)?;

    // Allocate u32 output buffer (same byte size — both u32 and f32 are 4 bytes).
    // Cannot use enc.alloc_output because this uses 4-byte elements, not enc.elem_size.
    let u32_bytes = idx_numel
        .checked_mul(4)
        .ok_or(MetalError::BufferByteOverflow {
            elems: idx_numel,
            elem_size: 4,
        })?;
    let (u32_buf, u32_offset) = arena_alloc_or_create(enc.ctx, u32_bytes)?;

    // Run conversion kernel: 1 input (f32 indices) → 1 output (u32 indices).
    let conv_pipeline = enc.pipeline(msl, &conv_name, 1)?;
    let conv_plan = DispatchMode::Elementwise { total: idx_u32 }.plan_cached()?;
    let e = enc.batch.new_encoder()?;
    conv_pipeline.encode_into(
        e,
        &[idx_buf],
        &[idx_offset],
        &u32_buf,
        u32_offset,
        &conv_plan,
    )?;
    // idx_buf borrow released here (NLL last use).

    // Step 2: Run the main IndexSelect/Gather kernel with u32 indices.
    // buffer(0) = input (float), buffer(1) = u32 indices, buffer(2) = output, buffer(3) = total.
    let main_pipeline = enc.pipeline(msl, kernel_name, 2)?;
    let total_u32 = to_dispatch_u32(total_elements)?;
    let main_plan = DispatchMode::Elementwise { total: total_u32 }.plan_cached()?;

    let (out_buf, out_offset) = enc.alloc_output(total_elements)?;
    let (in_buf, in_offset) = enc.input(input)?;

    let e2 = enc.batch.new_encoder()?;
    main_pipeline.encode_into(
        e2,
        &[in_buf, &u32_buf],
        &[in_offset, u32_offset],
        &out_buf,
        out_offset,
        &main_plan,
    )?;
    // in_buf borrow released here (NLL last use).

    enc.insert_output(output, out_buf, out_offset);
    Ok(())
}
