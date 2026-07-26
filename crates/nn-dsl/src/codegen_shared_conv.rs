// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic Conv1d kernel emitter using [`CodegenSyntax`].
//!
//! The loop body and indexing math are written once; the GPU syntax
//! (type keywords, cast syntax, kernel wrapper) comes from the trait.
//!
//! Part of #3338 D3.

use crate::codegen_shared::conv_output_len;
use crate::codegen_syntax::CodegenSyntax;
use crate::ir::ScalarType;

/// Parameters for a Conv1d kernel, validated before emission.
pub struct Conv1dParams {
    pub name: String,
    pub dtype: ScalarType,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub in_length: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub groups: usize,
    pub has_bias: bool,
}

/// Emit the Conv1d loop body (shared across MSL and HIP).
///
/// Returns the body lines between the thread-guard `if (tid >= total) return;`
/// and the final output write. The caller provides the kernel wrapper.
///
/// This is called by both `emit_conv1d_msl` and the HIP equivalent.
pub fn emit_conv1d_body<S: CodegenSyntax>(
    s: &S,
    p: &Conv1dParams,
) -> Result<Conv1dEmission, S::Error> {
    if p.groups == 0 {
        return Err(s.invalid_parameter_error("Conv1d groups must be > 0".into()));
    }
    let in_ch_per_group = p.in_channels / p.groups;
    let out_len = conv_output_len(p.in_length, p.kernel_size, p.stride, p.padding, p.dilation)
        .ok_or_else(|| {
            s.invalid_parameter_error(format!(
                "Conv1d output length overflow: in_length={}, kernel_size={}, \
                 stride={}, padding={}, dilation={}",
                p.in_length, p.kernel_size, p.stride, p.padding, p.dilation
            ))
        })?;

    let t = s.type_name(p.dtype)?;
    let acc = s.accum_type(p.dtype);
    let needs_cast = t != acc;

    let stride_s = s.safe_uint(p.stride)?;
    let padding_s = s.safe_uint(p.padding)?;
    let dilation_s = s.safe_uint(p.dilation)?;
    let in_length_s = s.safe_uint(p.in_length)?;
    let kernel_size_s = s.safe_uint(p.kernel_size)?;
    let in_ch_per_group_s = s.safe_uint(in_ch_per_group)?;
    let in_channels_s = s.safe_uint(p.in_channels)?;
    let out_len_s = s.safe_uint(out_len)?;
    let groups_s = s.safe_uint(p.groups)?;
    let out_channels_s = s.safe_uint(p.out_channels)?;

    let uint = s.uint_keyword();

    // Build the bias addition line.
    let bias_line = if p.has_bias {
        if needs_cast {
            format!("    sum += {};\n", s.cast_expr(acc, "bias[oc_local]"))
        } else {
            "    sum += bias[oc_local];\n".to_string()
        }
    } else {
        String::new()
    };

    // Build the MAC (multiply-accumulate) expression.
    let mac_expr = if needs_cast {
        format!(
            "sum += {} * {};",
            s.cast_expr(acc, "input[in_idx]"),
            s.cast_expr(acc, "weight[w_idx]")
        )
    } else {
        "sum += input[in_idx] * weight[w_idx];".to_string()
    };

    // Build the store expression.
    let store_expr = if needs_cast {
        s.cast_expr(t, "sum")
    } else {
        "sum".to_string()
    };

    // Build the constant declarations and loop body.
    let body = format!(
        r#"    {cst} STRIDE = {stride_s};
    {cst} PADDING = {padding_s};
    {cst} DILATION = {dilation_s};
    {cst} IN_LENGTH = {in_length_s};
    {cst} KERNEL_SIZE = {kernel_size_s};
    {cst} IN_CH_PER_GROUP = {in_ch_per_group_s};
    {cst} IN_CHANNELS = {in_channels_s};
    {cst} OUT_LEN = {out_len_s};
    {cst} GROUPS = {groups_s};
    {cst} OUT_CHANNELS = {out_channels_s};

    {uint} oc = tid / OUT_LEN;
    {uint} ot = tid % OUT_LEN;
    // For batched inputs the flat tid covers batch * out_channels * out_len
    // elements. oc may exceed OUT_CHANNELS for batch > 0. The input buffer
    // is contiguous [batch, channels, length] so abs_ic naturally indexes
    // into the correct batch. Weights are shared across batches so
    // use oc_local = oc % OUT_CHANNELS.
    {uint} oc_local = oc % OUT_CHANNELS;
    {uint} g = oc_local / (OUT_CHANNELS / GROUPS);

    {acc} sum = 0;
    for ({uint} ic = 0; ic < IN_CH_PER_GROUP; ic++) {{
        {uint} abs_ic = g * IN_CH_PER_GROUP + ic;
        // For batch > 0, offset abs_ic by (oc / OUT_CHANNELS) * in_channels
        // so we read from the correct batch in the flat input buffer.
        {uint} batch_ic_offset = (oc / OUT_CHANNELS) * IN_CHANNELS;
        for ({uint} k = 0; k < KERNEL_SIZE; k++) {{
            {uint} it = ot * STRIDE + k * DILATION;
            if (it >= PADDING && it - PADDING < IN_LENGTH) {{
                {uint} in_idx = (batch_ic_offset + abs_ic) * IN_LENGTH + (it - PADDING);
                {uint} w_idx = oc_local * IN_CH_PER_GROUP * KERNEL_SIZE + ic * KERNEL_SIZE + k;
                {mac_expr}
            }}
        }}
    }}
{bias_line}    output[tid] = {store_expr};"#,
        cst = format!("const {uint}"),
    );

    Ok(Conv1dEmission {
        body,
        storage_type: t.to_string(),
        accum_type: acc.to_string(),
        needs_cast,
        out_len,
    })
}

/// Result of emitting the Conv1d body. Backends use this to wrap in their
/// kernel header/parameter syntax.
pub struct Conv1dEmission {
    /// The loop body (constants + indexing + MAC + store).
    pub body: String,
    /// Storage type name (e.g., `"float"`, `"half"`).
    pub storage_type: String,
    /// Accumulator type name (e.g., `"float"`).
    pub accum_type: String,
    /// Whether input loads need casts (f16/bf16 → f32).
    pub needs_cast: bool,
    /// Computed output length.
    pub out_len: usize,
}
