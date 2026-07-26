// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! HIP C++ emission for convolution ops: Conv1d, Conv2d, ConvTranspose1d.
//!
//! Parallel to `nn-dsl::codegen_msl_structural_conv` — each function generates
//! a HIP `__global__` kernel. Uses f32 accumulation for f16/bf16 inputs.
//! Batched input support via `oc_local = oc % OUT_CHANNELS` for weight/bias
//! indexing (critical invariant: weight/bias buffers are not batched).

use crate::codegen_hip::{hip_accumulator_type, hip_type, safe_hip_uint};
use crate::HipCodegenError;
use nn_dsl::ScalarType;

/// Checked output-length formula for Conv1d:
/// `out = (in_length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
fn conv1d_out_len(
    in_length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, HipCodegenError> {
    kernel_size
        .checked_sub(1)
        .and_then(|ks_m1| dilation.checked_mul(ks_m1))
        .and_then(|dilated| dilated.checked_add(1))
        .and_then(|sub_term| {
            in_length
                .checked_add(2usize.checked_mul(padding)?)
                .and_then(|padded| padded.checked_sub(sub_term))
        })
        .and_then(|numerator| numerator.checked_div(stride).map(|q| q + 1))
        .ok_or_else(|| {
            HipCodegenError::InvalidParameter(format!(
                "Conv1d output length overflow: in_length={in_length}, kernel_size={kernel_size}, \
                 stride={stride}, padding={padding}, dilation={dilation}"
            ))
        })
}

/// Emit HIP kernel for 1D convolution.
///
/// Each thread computes one output element: the dot product of the relevant
/// input window with the weight kernel, plus optional bias. Supports stride,
/// padding, dilation, and groups. Batched via `oc_local = oc % OUT_CHANNELS`.
#[allow(clippy::too_many_arguments)]
pub fn emit_conv1d_kernel(
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
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let acc = hip_accumulator_type(dtype);
    let needs_cast = t != acc;

    if groups == 0 {
        return Err(HipCodegenError::InvalidParameter(
            "Conv1d groups must be > 0".into(),
        ));
    }
    let in_ch_per_group = in_channels / groups;
    let out_len = conv1d_out_len(in_length, kernel_size, stride, padding, dilation)?;

    let stride_s = safe_hip_uint(stride)?;
    let padding_s = safe_hip_uint(padding)?;
    let dilation_s = safe_hip_uint(dilation)?;
    let in_length_s = safe_hip_uint(in_length)?;
    let kernel_size_s = safe_hip_uint(kernel_size)?;
    let in_ch_per_group_s = safe_hip_uint(in_ch_per_group)?;
    let in_channels_s = safe_hip_uint(in_channels)?;
    let out_len_s = safe_hip_uint(out_len)?;
    let groups_s = safe_hip_uint(groups)?;
    let out_channels_s = safe_hip_uint(out_channels)?;

    let bias_param = if has_bias {
        format!("    const {t}* __restrict__ bias,\n")
    } else {
        String::new()
    };

    let bias_line = if has_bias {
        if needs_cast {
            format!("    sum += ({acc})bias[oc_local];\n")
        } else {
            "    sum += bias[oc_local];\n".to_string()
        }
    } else {
        String::new()
    };

    let mac_expr = if needs_cast {
        format!("sum += ({acc})input[in_idx] * ({acc})weight[w_idx];")
    } else {
        "sum += input[in_idx] * weight[w_idx];".to_string()
    };

    let store_expr = if needs_cast {
        format!("({t})sum")
    } else {
        "sum".to_string()
    };

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    const {t}* __restrict__ weight,
{bias_param}    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const unsigned int STRIDE = {stride_s};
    const unsigned int PADDING = {padding_s};
    const unsigned int DILATION = {dilation_s};
    const unsigned int IN_LENGTH = {in_length_s};
    const unsigned int KERNEL_SIZE = {kernel_size_s};
    const unsigned int IN_CH_PER_GROUP = {in_ch_per_group_s};
    const unsigned int IN_CHANNELS = {in_channels_s};
    const unsigned int OUT_LEN = {out_len_s};
    const unsigned int GROUPS = {groups_s};
    const unsigned int OUT_CHANNELS = {out_channels_s};

    unsigned int oc = tid / OUT_LEN;
    unsigned int ot = tid % OUT_LEN;
    unsigned int oc_local = oc % OUT_CHANNELS;
    unsigned int g = oc_local / (OUT_CHANNELS / GROUPS);

    {acc} sum = 0;
    for (unsigned int ic = 0; ic < IN_CH_PER_GROUP; ic++) {{
        unsigned int abs_ic = g * IN_CH_PER_GROUP + ic;
        unsigned int batch_ic_offset = (oc / OUT_CHANNELS) * IN_CHANNELS;
        for (unsigned int k = 0; k < KERNEL_SIZE; k++) {{
            unsigned int it = ot * STRIDE + k * DILATION;
            if (it >= PADDING && it - PADDING < IN_LENGTH) {{
                unsigned int in_idx = (batch_ic_offset + abs_ic) * IN_LENGTH + (it - PADDING);
                unsigned int w_idx = oc_local * IN_CH_PER_GROUP * KERNEL_SIZE + ic * KERNEL_SIZE + k;
                {mac_expr}
            }}
        }}
    }}
{bias_line}    output[tid] = {store_expr};
}}"#,
    ))
}

/// Emit HIP kernel for 2D convolution.
///
/// Each thread computes one output element. Supports stride, padding, dilation,
/// and groups. Batched via `oc_local = oc % OUT_CHANNELS`.
#[allow(clippy::too_many_arguments)]
pub fn emit_conv2d_kernel(
    name: &str,
    dtype: ScalarType,
    in_channels: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    in_height: usize,
    in_width: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    has_bias: bool,
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let acc = hip_accumulator_type(dtype);
    let needs_cast = t != acc;

    if groups == 0 {
        return Err(HipCodegenError::InvalidParameter(
            "Conv2d groups must be > 0".into(),
        ));
    }
    let in_ch_per_group = in_channels / groups;

    // Output spatial dimensions (checked arithmetic).
    let out_h = conv1d_out_len(in_height, kernel_h, stride_h, padding_h, dilation_h)?;
    let out_w = conv1d_out_len(in_width, kernel_w, stride_w, padding_w, dilation_w)?;
    let out_hw = out_h
        .checked_mul(out_w)
        .ok_or_else(|| HipCodegenError::ShapeProductOverflow {
            shape: vec![out_h, out_w],
        })?;

    let stride_h_s = safe_hip_uint(stride_h)?;
    let stride_w_s = safe_hip_uint(stride_w)?;
    let padding_h_s = safe_hip_uint(padding_h)?;
    let padding_w_s = safe_hip_uint(padding_w)?;
    let dilation_h_s = safe_hip_uint(dilation_h)?;
    let dilation_w_s = safe_hip_uint(dilation_w)?;
    let in_height_s = safe_hip_uint(in_height)?;
    let in_width_s = safe_hip_uint(in_width)?;
    let kernel_h_s = safe_hip_uint(kernel_h)?;
    let kernel_w_s = safe_hip_uint(kernel_w)?;
    let in_ch_per_group_s = safe_hip_uint(in_ch_per_group)?;
    let in_channels_s = safe_hip_uint(in_channels)?;
    let out_channels_s = safe_hip_uint(out_channels)?;
    let out_h_s = safe_hip_uint(out_h)?;
    let out_w_s = safe_hip_uint(out_w)?;
    let out_hw_s = safe_hip_uint(out_hw)?;
    let groups_s = safe_hip_uint(groups)?;

    let bias_param = if has_bias {
        format!("    const {t}* __restrict__ bias,\n")
    } else {
        String::new()
    };

    let bias_line = if has_bias {
        if needs_cast {
            format!("    sum += ({acc})bias[oc_local];\n")
        } else {
            "    sum += bias[oc_local];\n".to_string()
        }
    } else {
        String::new()
    };

    let mac_expr = if needs_cast {
        format!("sum += ({acc})input[in_idx] * ({acc})weight[w_idx];")
    } else {
        "sum += input[in_idx] * weight[w_idx];".to_string()
    };

    let store_expr = if needs_cast {
        format!("({t})sum")
    } else {
        "sum".to_string()
    };

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    const {t}* __restrict__ weight,
{bias_param}    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const unsigned int STRIDE_H = {stride_h_s};
    const unsigned int STRIDE_W = {stride_w_s};
    const unsigned int PADDING_H = {padding_h_s};
    const unsigned int PADDING_W = {padding_w_s};
    const unsigned int DILATION_H = {dilation_h_s};
    const unsigned int DILATION_W = {dilation_w_s};
    const unsigned int IN_HEIGHT = {in_height_s};
    const unsigned int IN_WIDTH = {in_width_s};
    const unsigned int KERNEL_H = {kernel_h_s};
    const unsigned int KERNEL_W = {kernel_w_s};
    const unsigned int IN_CH_PER_GROUP = {in_ch_per_group_s};
    const unsigned int IN_CHANNELS = {in_channels_s};
    const unsigned int OUT_CHANNELS = {out_channels_s};
    const unsigned int OUT_H = {out_h_s};
    const unsigned int OUT_W = {out_w_s};
    const unsigned int OUT_HW = {out_hw_s};
    const unsigned int GROUPS = {groups_s};

    unsigned int oc = tid / OUT_HW;
    unsigned int spatial = tid % OUT_HW;
    unsigned int oh = spatial / OUT_W;
    unsigned int ow = spatial % OUT_W;
    unsigned int oc_local = oc % OUT_CHANNELS;
    unsigned int g = oc_local / (OUT_CHANNELS / GROUPS);

    {acc} sum = 0;
    for (unsigned int ic = 0; ic < IN_CH_PER_GROUP; ic++) {{
        unsigned int abs_ic = g * IN_CH_PER_GROUP + ic;
        unsigned int batch_ic_offset = (oc / OUT_CHANNELS) * IN_CHANNELS;
        for (unsigned int kh = 0; kh < KERNEL_H; kh++) {{
            unsigned int ih = oh * STRIDE_H + kh * DILATION_H;
            if (ih >= PADDING_H && ih - PADDING_H < IN_HEIGHT) {{
                for (unsigned int kw = 0; kw < KERNEL_W; kw++) {{
                    unsigned int iw = ow * STRIDE_W + kw * DILATION_W;
                    if (iw >= PADDING_W && iw - PADDING_W < IN_WIDTH) {{
                        unsigned int in_idx = (batch_ic_offset + abs_ic) * IN_HEIGHT * IN_WIDTH + (ih - PADDING_H) * IN_WIDTH + (iw - PADDING_W);
                        unsigned int w_idx = oc_local * IN_CH_PER_GROUP * KERNEL_H * KERNEL_W + ic * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw;
                        {mac_expr}
                    }}
                }}
            }}
        }}
    }}
{bias_line}    output[tid] = {store_expr};
}}"#,
    ))
}

/// Emit HIP kernel for transposed 1D convolution (ConvTranspose1d).
///
/// Gather pattern: for each output position, accumulates contributions from
/// input positions that map to it through the transposition relationship.
/// Weight layout: `[in_channels, out_channels/groups, kernel_size]`.
#[allow(clippy::too_many_arguments)]
pub fn emit_conv_transpose1d_kernel(
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
) -> Result<String, HipCodegenError> {
    let t = hip_type(dtype)?;
    let acc = hip_accumulator_type(dtype);
    let needs_cast = t != acc;

    if groups == 0 {
        return Err(HipCodegenError::InvalidParameter(
            "ConvTranspose1d groups must be > 0".into(),
        ));
    }
    let in_ch_per_group = in_channels / groups;

    // PyTorch output length:
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
            HipCodegenError::InvalidParameter(format!(
                "ConvTranspose1d output length overflow: in_length={in_length}, \
                 kernel_size={kernel_size}, stride={stride}, padding={padding}, \
                 dilation={dilation}, output_padding={output_padding}"
            ))
        })?;

    let stride_s = safe_hip_uint(stride)?;
    let padding_s = safe_hip_uint(padding)?;
    let dilation_s = safe_hip_uint(dilation)?;
    let in_length_s = safe_hip_uint(in_length)?;
    let kernel_size_s = safe_hip_uint(kernel_size)?;
    let in_channels_s = safe_hip_uint(in_channels)?;
    let out_channels_s = safe_hip_uint(out_channels)?;
    let in_ch_per_group_s = safe_hip_uint(in_ch_per_group)?;
    let groups_s = safe_hip_uint(groups)?;
    let out_length_s = safe_hip_uint(out_length)?;

    let bias_param = if has_bias {
        format!("    const {t}* __restrict__ bias,\n")
    } else {
        String::new()
    };

    let bias_line = if has_bias {
        if needs_cast {
            format!("    sum += ({acc})bias[oc_local];\n")
        } else {
            "    sum += bias[oc_local];\n".to_string()
        }
    } else {
        String::new()
    };

    let mac_expr = if needs_cast {
        format!("sum += ({acc})input[in_idx] * ({acc})weight[w_idx];")
    } else {
        "sum += input[in_idx] * weight[w_idx];".to_string()
    };

    let store_expr = if needs_cast {
        format!("({t})sum")
    } else {
        "sum".to_string()
    };

    Ok(format!(
        r#"extern "C" __global__ void {name}(
    const {t}* __restrict__ input,
    const {t}* __restrict__ weight,
{bias_param}    {t}* __restrict__ output,
    const unsigned int total
) {{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    const unsigned int STRIDE = {stride_s};
    const unsigned int PADDING = {padding_s};
    const unsigned int DILATION = {dilation_s};
    const unsigned int IN_LENGTH = {in_length_s};
    const unsigned int KERNEL_SIZE = {kernel_size_s};
    const unsigned int IN_CHANNELS = {in_channels_s};
    const unsigned int IN_CH_PER_GROUP = {in_ch_per_group_s};
    const unsigned int OUT_CHANNELS = {out_channels_s};
    const unsigned int OUT_LENGTH = {out_length_s};
    const unsigned int GROUPS = {groups_s};
    const unsigned int OUT_CH_PER_GROUP = OUT_CHANNELS / GROUPS;

    unsigned int oc = tid / OUT_LENGTH;
    unsigned int ot = tid % OUT_LENGTH;
    unsigned int oc_local = oc % OUT_CHANNELS;
    unsigned int g = oc_local / OUT_CH_PER_GROUP;
    unsigned int oc_in_group = oc_local % OUT_CH_PER_GROUP;
    unsigned int batch_ic_offset = (oc / OUT_CHANNELS) * IN_CHANNELS;

    {acc} sum = 0;
    for (unsigned int ic = 0; ic < IN_CH_PER_GROUP; ic++) {{
        unsigned int abs_ic = g * IN_CH_PER_GROUP + ic;
        for (unsigned int k = 0; k < KERNEL_SIZE; k++) {{
            unsigned int dk = k * DILATION;
            unsigned int ot_pad = ot + PADDING;
            if (ot_pad >= dk && (ot_pad - dk) % STRIDE == 0) {{
                unsigned int it = (ot_pad - dk) / STRIDE;
                if (it < IN_LENGTH) {{
                    unsigned int in_idx = (batch_ic_offset + abs_ic) * IN_LENGTH + it;
                    unsigned int w_idx = abs_ic * OUT_CH_PER_GROUP * KERNEL_SIZE + oc_in_group * KERNEL_SIZE + k;
                    {mac_expr}
                }}
            }}
        }}
    }}
{bias_line}    output[tid] = {store_expr};
}}"#,
    ))
}

#[cfg(test)]
#[path = "codegen_hip_tensor_tests_conv.rs"]
mod tests;
