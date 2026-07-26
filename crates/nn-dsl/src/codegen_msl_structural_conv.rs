// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL code generation for convolution tensor operations (Conv1d, ConvTranspose1d).
//!
//! Extracted from `codegen_msl_structural` to keep both files under the 500-line
//! limit. The parent module owns structural ops (AxisSelect, Stack, Broadcast);
//! this module owns convolution MSL emission.
//!
//! Conv2d codegen is in the `conv2d` submodule (`codegen_msl_structural_conv2d.rs`).

#[path = "codegen_msl_structural_conv2d.rs"]
mod conv2d;
pub(crate) use conv2d::emit_conv2d_kernel;

use crate::codegen_msl;
use crate::codegen_msl_structural::safe_msl_uint;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;

/// Emit MSL for a Conv1d kernel.
///
/// Computes a 1D convolution with configurable stride, padding, dilation, and
/// groups. Each output thread computes one element: the dot product of the
/// relevant input window with the weight kernel, plus optional bias.
///
/// Buffer layout:
/// - `buffer(0)`: input data `[in_channels, in_length]`
/// - `buffer(1)`: weight data `[out_channels, in_ch_per_group, kernel_size]`
/// - `buffer(2)`: bias (if `has_bias`) or output (if no bias)
/// - `buffer(3)`: output (if `has_bias`) or total elements (if no bias)
/// - `buffer(4)`: total elements (if `has_bias`)
#[must_use = "returns a Result that may contain an error"]
#[cfg_attr(not(test), allow(dead_code))] // Used in codegen_msl_structural_tests.rs
pub(crate) fn emit_conv1d_kernel(
    name: &str,
    dtype: ScalarType,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    has_bias: bool,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    // Accumulate dot product in f32 for f16/bf16 precision (#2557, #1352).
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;
    if groups == 0 {
        return Err(TensorMSLCodegenError::InvalidParameter(
            "Conv1d groups must be > 0".into(),
        ));
    }
    let in_ch_per_group = in_channels / groups;
    // Checked output-length formula: out = (in + 2*pad - dil*(ks-1) - 1) / stride + 1
    let out_len = kernel_size
        .checked_sub(1)
        .and_then(|ks_m1| dilation.checked_mul(ks_m1))
        .and_then(|dilated| dilated.checked_add(1))
        .and_then(|sub_term| {
            in_length
                .checked_add(2usize.checked_mul(padding)?)
                .and_then(|padded| padded.checked_sub(sub_term))
        })
        .and_then(|numerator| numerator.checked_div(stride).map(|quotient| quotient + 1))
        .ok_or_else(|| {
            TensorMSLCodegenError::InvalidParameter(format!(
                "Conv1d output length overflow: in_length={in_length}, kernel_size={kernel_size}, \
                 stride={stride}, padding={padding}, dilation={dilation}"
            ))
        })?;

    let (bias_buf, out_buf, total_buf) = if has_bias {
        ("2", "3", "4")
    } else {
        ("", "2", "3")
    };

    let mut params = format!(
        "    device const {t}* input  [[buffer(0)]],\n\
         \x20   device const {t}* weight [[buffer(1)]],\n"
    );
    if has_bias {
        params.push_str(&format!(
            "    device const {t}* bias   [[buffer({bias_buf})]],\n"
        ));
    }
    params.push_str(&format!(
        "    device {t}* output         [[buffer({out_buf})]],\n\
         \x20   constant uint& total      [[buffer({total_buf})]],\n\
         \x20   uint tid [[thread_position_in_grid]]"
    ));

    let stride_s = safe_msl_uint(stride)?;
    let padding_s = safe_msl_uint(padding)?;
    let dilation_s = safe_msl_uint(dilation)?;
    let in_length_s = safe_msl_uint(in_length)?;
    let kernel_size_s = safe_msl_uint(kernel_size)?;
    let in_ch_per_group_s = safe_msl_uint(in_ch_per_group)?;
    let out_len_s = safe_msl_uint(out_len)?;
    let groups_s = safe_msl_uint(groups)?;
    let out_channels_s = safe_msl_uint(out_channels)?;

    let in_channels_s = safe_msl_uint(in_channels)?;
    let bias_line = if has_bias {
        if needs_cast {
            format!("    sum += {acc}(bias[oc_local]);\n")
        } else {
            "    sum += bias[oc_local];\n".to_string()
        }
    } else {
        String::new()
    };

    // Cast accumulator back to storage type on final write.
    let store_expr = if needs_cast {
        format!("{t}(sum)")
    } else {
        "sum".to_string()
    };

    // Cast loads to accumulator type for half-precision inputs.
    let mac_expr = if needs_cast {
        format!("sum += {acc}(input[in_idx]) * {acc}(weight[w_idx]);")
    } else {
        "sum += input[in_idx] * weight[w_idx];".to_string()
    };

    Ok(format!(
        r#"[[kernel]] void {name}(
{params}
) {{
    if (tid >= total) return;
    const uint STRIDE = {stride_s};
    const uint PADDING = {padding_s};
    const uint DILATION = {dilation_s};
    const uint IN_LENGTH = {in_length_s};
    const uint KERNEL_SIZE = {kernel_size_s};
    const uint IN_CH_PER_GROUP = {in_ch_per_group_s};
    const uint IN_CHANNELS = {in_channels_s};
    const uint OUT_LEN = {out_len_s};
    const uint GROUPS = {groups_s};
    const uint OUT_CHANNELS = {out_channels_s};

    uint oc = tid / OUT_LEN;
    uint ot = tid % OUT_LEN;
    // For batched inputs the flat tid covers batch * out_channels * out_len
    // elements. oc may exceed OUT_CHANNELS for batch > 0. The input buffer
    // is contiguous [batch, channels, length] so abs_ic naturally indexes
    // into the correct batch. Weight and bias are shared across batches so
    // use oc_local = oc % OUT_CHANNELS.
    uint oc_local = oc % OUT_CHANNELS;
    uint g = oc_local / (OUT_CHANNELS / GROUPS);

    {acc} sum = 0;
    for (uint ic = 0; ic < IN_CH_PER_GROUP; ic++) {{
        uint abs_ic = g * IN_CH_PER_GROUP + ic;
        // For batch > 0, offset abs_ic by (oc / OUT_CHANNELS) * in_channels
        // so we read from the correct batch in the flat input buffer.
        uint batch_ic_offset = (oc / OUT_CHANNELS) * IN_CHANNELS;
        for (uint k = 0; k < KERNEL_SIZE; k++) {{
            uint it = ot * STRIDE + k * DILATION;
            if (it >= PADDING && it - PADDING < IN_LENGTH) {{
                uint in_idx = (batch_ic_offset + abs_ic) * IN_LENGTH + (it - PADDING);
                uint w_idx = oc_local * IN_CH_PER_GROUP * KERNEL_SIZE + ic * KERNEL_SIZE + k;
                {mac_expr}
            }}
        }}
    }}
{bias_line}    output[tid] = {store_expr};
}}"#,
    ))
}

/// Emit MSL for a ConvTranspose1d (transposed convolution / upsampling) kernel.
///
/// For each output element, accumulates contributions from input positions that
/// map to it through the transposition relationship:
///   `output[ot] += input[it] * weight[abs_ic, oc_in_group, k]`
///   where `ot = it * stride + k * dilation - padding`
///
/// Weight layout: `[in_channels, out_channels/groups, kernel_size]` (note: in/out
/// swapped relative to Conv1d).
///
/// Supports dilation > 1 and groups > 1, matching PyTorch's ConvTranspose1d.
///
/// Buffer layout:
/// - `buffer(0)`: input data `[in_channels, in_length]`
/// - `buffer(1)`: weight data `[in_channels, out_channels/groups, kernel_size]`
/// - `buffer(2)`: bias (if `has_bias`) or output (if no bias)
/// - `buffer(3)`: output (if `has_bias`) or total elements (if no bias)
/// - `buffer(4)`: total elements (if `has_bias`)
#[must_use = "returns a Result that may contain an error"]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_conv_transpose_1d_kernel(
    name: &str,
    dtype: ScalarType,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_padding: usize,
    has_bias: bool,
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    // Accumulate dot product in f32 for f16/bf16 precision (#2557, #1352).
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;
    if groups == 0 {
        return Err(TensorMSLCodegenError::InvalidParameter(
            "ConvTranspose1d groups must be > 0".into(),
        ));
    }
    let in_ch_per_group = in_channels / groups;

    let (bias_buf, out_buf, total_buf) = if has_bias {
        ("2", "3", "4")
    } else {
        ("", "2", "3")
    };

    let mut params = format!(
        "    device const {t}* input  [[buffer(0)]],\n\
         \x20   device const {t}* weight [[buffer(1)]],\n"
    );
    if has_bias {
        params.push_str(&format!(
            "    device const {t}* bias   [[buffer({bias_buf})]],\n"
        ));
    }
    params.push_str(&format!(
        "    device {t}* output         [[buffer({out_buf})]],\n\
         \x20   constant uint& total      [[buffer({total_buf})]],\n\
         \x20   uint tid [[thread_position_in_grid]]"
    ));

    let stride_s = safe_msl_uint(stride)?;
    let padding_s = safe_msl_uint(padding)?;
    let dilation_s = safe_msl_uint(dilation)?;
    let in_length_s = safe_msl_uint(in_length)?;
    let kernel_size_s = safe_msl_uint(kernel_size)?;
    let in_channels_s = safe_msl_uint(in_channels)?;
    let out_channels_s = safe_msl_uint(out_channels)?;
    let in_ch_per_group_s = safe_msl_uint(in_ch_per_group)?;
    let groups_s = safe_msl_uint(groups)?;

    // Checked output-length formula (PyTorch):
    // out = (in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    let out_length = in_length
        .checked_sub(1)
        .and_then(|im1| im1.checked_mul(stride))
        .and_then(|base| {
            dilation
                .checked_mul(kernel_size.checked_sub(1)?)
                .and_then(|dk| base.checked_add(dk))
        })
        .and_then(|sum| sum.checked_add(output_padding))
        .and_then(|sum| sum.checked_add(1))
        .and_then(|sum| sum.checked_sub(2usize.checked_mul(padding)?))
        .ok_or_else(|| {
            TensorMSLCodegenError::InvalidParameter(format!(
                "ConvTranspose1d output length overflow: in_length={in_length}, \
                 kernel_size={kernel_size}, stride={stride}, padding={padding}, \
                 dilation={dilation}, output_padding={output_padding}"
            ))
        })?;
    let out_length_s = safe_msl_uint(out_length)?;

    let bias_line = if has_bias {
        if needs_cast {
            format!("    sum += {acc}(bias[oc_local]);\n")
        } else {
            "    sum += bias[oc_local];\n".to_string()
        }
    } else {
        String::new()
    };

    // Cast accumulator back to storage type on final write.
    let store_expr = if needs_cast {
        format!("{t}(sum)")
    } else {
        "sum".to_string()
    };

    // Cast loads to accumulator type for half-precision inputs.
    let mac_expr = if needs_cast {
        format!("sum += {acc}(input[in_idx]) * {acc}(weight[w_idx]);")
    } else {
        "sum += input[in_idx] * weight[w_idx];".to_string()
    };

    // ConvTranspose1d gather pattern with dilation and groups:
    // For output position ot, output channel oc:
    //   g = oc / (OUT_CHANNELS / GROUPS)
    //   sum over (ic in group, k) of input[abs_ic, it] * weight[abs_ic, oc_in_group, k]
    //   where dk = k * dilation, it = (ot + padding - dk) / stride,
    //   when (ot + padding) >= dk
    //   and (ot + padding - dk) % stride == 0 and it < in_length
    Ok(format!(
        r#"[[kernel]] void {name}(
{params}
) {{
    if (tid >= total) return;
    const uint STRIDE = {stride_s};
    const uint PADDING = {padding_s};
    const uint DILATION = {dilation_s};
    const uint IN_LENGTH = {in_length_s};
    const uint KERNEL_SIZE = {kernel_size_s};
    const uint IN_CHANNELS = {in_channels_s};
    const uint IN_CH_PER_GROUP = {in_ch_per_group_s};
    const uint OUT_CHANNELS = {out_channels_s};
    const uint OUT_LENGTH = {out_length_s};
    const uint GROUPS = {groups_s};
    const uint OUT_CH_PER_GROUP = OUT_CHANNELS / GROUPS;

    uint oc = tid / OUT_LENGTH;
    uint ot = tid % OUT_LENGTH;
    // For batched inputs oc may exceed OUT_CHANNELS for batch > 0.
    // Weight and bias are shared across batches.
    uint oc_local = oc % OUT_CHANNELS;
    uint g = oc_local / OUT_CH_PER_GROUP;
    uint oc_in_group = oc_local % OUT_CH_PER_GROUP;
    uint batch_ic_offset = (oc / OUT_CHANNELS) * IN_CHANNELS;

    {acc} sum = 0;
    for (uint ic = 0; ic < IN_CH_PER_GROUP; ic++) {{
        uint abs_ic = g * IN_CH_PER_GROUP + ic;
        for (uint k = 0; k < KERNEL_SIZE; k++) {{
            uint dk = k * DILATION;
            uint ot_pad = ot + PADDING;
            if (ot_pad >= dk && (ot_pad - dk) % STRIDE == 0) {{
                uint it = (ot_pad - dk) / STRIDE;
                if (it < IN_LENGTH) {{
                    uint in_idx = (batch_ic_offset + abs_ic) * IN_LENGTH + it;
                    uint w_idx = abs_ic * OUT_CH_PER_GROUP * KERNEL_SIZE + oc_in_group * KERNEL_SIZE + k;
                    {mac_expr}
                }}
            }}
        }}
    }}
{bias_line}    output[tid] = {store_expr};
}}"#,
    ))
}
