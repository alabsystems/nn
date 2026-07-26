// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL code generation for Conv2d tensor operations.
//!
//! Extracted from `codegen_msl_structural_conv` to keep the parent module under
//! the 500-line limit. Part of #1410 Direction 3.

use crate::codegen_msl;
use crate::codegen_msl_structural::safe_msl_uint;
use crate::codegen_msl_tensor::TensorMSLCodegenError;
use crate::ir::ScalarType;

/// Emit MSL for a Conv2d kernel.
///
/// 2D generalization of `emit_conv1d_kernel`. Each output thread computes one
/// element at position `(oc, oh, ow)`: the dot product of the relevant 2D
/// input patch with the weight kernel, plus optional bias.
///
/// Input layout: `[in_channels, height, width]` (row-major).
/// Weight layout: `[out_channels, in_channels/groups, kernel_h, kernel_w]`.
/// Output layout: `[out_channels, out_h, out_w]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_conv2d_kernel(
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
) -> Result<String, TensorMSLCodegenError> {
    let t = codegen_msl::msl_type(dtype);
    // Accumulate dot product in f32 for f16/bf16 precision (#2557, #1352).
    let acc = codegen_msl::msl_accumulator_type(dtype);
    let needs_cast = t != acc;
    let in_ch_per_group = in_channels / groups;

    // Output spatial dimensions.
    let eff_kh = dilation_h * (kernel_h - 1) + 1;
    let eff_kw = dilation_w * (kernel_w - 1) + 1;
    let padded_h = in_height + 2 * padding_h;
    let padded_w = in_width + 2 * padding_w;
    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(TensorMSLCodegenError::InvalidParameter(format!(
            "Conv2d kernel too large: padded=({padded_h},{padded_w}), eff_kernel=({eff_kh},{eff_kw})"
        )));
    }
    let out_h = (padded_h - eff_kh) / stride_h + 1;
    let out_w = (padded_w - eff_kw) / stride_w + 1;

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

    let stride_h_s = safe_msl_uint(stride_h)?;
    let stride_w_s = safe_msl_uint(stride_w)?;
    let padding_h_s = safe_msl_uint(padding_h)?;
    let padding_w_s = safe_msl_uint(padding_w)?;
    let dilation_h_s = safe_msl_uint(dilation_h)?;
    let dilation_w_s = safe_msl_uint(dilation_w)?;
    let in_height_s = safe_msl_uint(in_height)?;
    let in_width_s = safe_msl_uint(in_width)?;
    let kernel_h_s = safe_msl_uint(kernel_h)?;
    let kernel_w_s = safe_msl_uint(kernel_w)?;
    let in_ch_per_group_s = safe_msl_uint(in_ch_per_group)?;
    let out_h_s = safe_msl_uint(out_h)?;
    let out_w_s = safe_msl_uint(out_w)?;
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
    const uint STRIDE_H = {stride_h_s};
    const uint STRIDE_W = {stride_w_s};
    const uint PADDING_H = {padding_h_s};
    const uint PADDING_W = {padding_w_s};
    const uint DILATION_H = {dilation_h_s};
    const uint DILATION_W = {dilation_w_s};
    const uint IN_HEIGHT = {in_height_s};
    const uint IN_WIDTH = {in_width_s};
    const uint KERNEL_H = {kernel_h_s};
    const uint KERNEL_W = {kernel_w_s};
    const uint IN_CH_PER_GROUP = {in_ch_per_group_s};
    const uint IN_CHANNELS = {in_channels_s};
    const uint OUT_H = {out_h_s};
    const uint OUT_W = {out_w_s};
    const uint GROUPS = {groups_s};
    const uint OUT_CHANNELS = {out_channels_s};

    uint oc = tid / (OUT_H * OUT_W);
    uint rem = tid % (OUT_H * OUT_W);
    uint oh = rem / OUT_W;
    uint ow = rem % OUT_W;
    // For batched inputs the flat tid covers batch * out_channels * out_h * out_w
    // elements. oc may exceed OUT_CHANNELS for batch > 0. Weight and bias are
    // shared across batches so use oc_local = oc % OUT_CHANNELS.
    uint oc_local = oc % OUT_CHANNELS;
    uint g = oc_local / (OUT_CHANNELS / GROUPS);

    {acc} sum = 0;
    for (uint ic = 0; ic < IN_CH_PER_GROUP; ic++) {{
        uint abs_ic = g * IN_CH_PER_GROUP + ic;
        uint batch_ic_offset = (oc / OUT_CHANNELS) * IN_CHANNELS;
        for (uint kh = 0; kh < KERNEL_H; kh++) {{
            uint ih = oh * STRIDE_H + kh * DILATION_H;
            if (ih >= PADDING_H && ih - PADDING_H < IN_HEIGHT) {{
                for (uint kw = 0; kw < KERNEL_W; kw++) {{
                    uint iw = ow * STRIDE_W + kw * DILATION_W;
                    if (iw >= PADDING_W && iw - PADDING_W < IN_WIDTH) {{
                        uint in_idx = (batch_ic_offset + abs_ic) * IN_HEIGHT * IN_WIDTH + (ih - PADDING_H) * IN_WIDTH + (iw - PADDING_W);
                        uint w_idx = oc_local * IN_CH_PER_GROUP * KERNEL_H * KERNEL_W + ic * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw;
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
