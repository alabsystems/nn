// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convolution shape inference for the tensor IR.
//!
//! Extracted from `tensor_ir_validate_shape.rs` to stay under the 500-line
//! limit. Contains output shape computation for Conv1d, Conv2d, and
//! ConvTranspose1d operations.

use super::super::super::{TensorIRConvError, TensorIRError};

/// Compute Conv1d output shape.
///
/// Conv1d: `[*, C_in, L_in]` → `[*, C_out, L_out]`
/// where `L_out = (L_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1`.
pub(crate) fn conv1d_output_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Vec<usize>, TensorIRError> {
    if input_shape.len() < 2 {
        return Err(TensorIRConvError::Conv1dInputRankTooLow {
            rank: input_shape.len(),
        }
        .into());
    }
    if weight_shape.len() < 3 {
        return Err(TensorIRConvError::Conv1dWeightShape {
            shape: weight_shape.to_vec(),
        }
        .into());
    }
    let in_len = input_shape[input_shape.len() - 1];
    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];
    if stride == 0 {
        return Err(TensorIRConvError::Conv1dZeroStride.into());
    }
    if kernel_size == 0 {
        return Err(TensorIRConvError::Conv1dZeroKernelSize.into());
    }
    // kernel_size >= 1 guaranteed above, so kernel_size - 1 is safe.
    let effective_kernel = dilation
        .checked_mul(kernel_size - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv1dArithmeticOverflow {
                context: format!(
                    "effective_kernel: dilation={dilation} * (kernel_size={kernel_size} - 1) + 1"
                ),
            })
        })?;
    let padded = padding
        .checked_mul(2)
        .and_then(|p2| in_len.checked_add(p2))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv1dArithmeticOverflow {
                context: format!("padded: in_len={in_len} + 2 * padding={padding}"),
            })
        })?;
    if padded < effective_kernel {
        return Err(TensorIRConvError::Conv1dKernelTooLarge {
            kernel_size: effective_kernel,
            padded_len: padded,
            in_len,
            padding,
        }
        .into());
    }
    let out_len = (padded - effective_kernel) / stride + 1;
    // Preserve leading dimensions (batch etc), replace last two with [out_ch, out_len]
    let mut output_shape = input_shape[..input_shape.len() - 2].to_vec();
    output_shape.push(out_channels);
    output_shape.push(out_len);
    Ok(output_shape)
}

/// Compute Conv2d output shape.
///
/// Conv2d: `[*, C_in, H, W]` → `[*, C_out, H_out, W_out]`
/// where `H_out = (H + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1`.
pub(crate) fn conv2d_output_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<Vec<usize>, TensorIRError> {
    if input_shape.len() < 3 {
        return Err(TensorIRConvError::Conv2dInputRankTooLow {
            rank: input_shape.len(),
        }
        .into());
    }
    if weight_shape.len() != 4 {
        return Err(TensorIRConvError::Conv2dWeightShape {
            shape: weight_shape.to_vec(),
        }
        .into());
    }
    let in_h = input_shape[input_shape.len() - 2];
    let in_w = input_shape[input_shape.len() - 1];
    let out_channels = weight_shape[0];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];
    if stride_h == 0 || stride_w == 0 {
        return Err(TensorIRConvError::Conv2dZeroStride { stride_h, stride_w }.into());
    }
    if kernel_h == 0 || kernel_w == 0 {
        return Err(TensorIRConvError::Conv2dZeroKernelSize { kernel_h, kernel_w }.into());
    }
    // out_h = (in_h + 2*pad_h - dilation_h*(kH-1) - 1) / stride_h + 1
    let eff_kh = dilation_h
        .checked_mul(kernel_h - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!(
                    "effective_kernel_h: dilation_h={dilation_h} * (kH={kernel_h} - 1) + 1"
                ),
            })
        })?;
    let padded_h = padding_h
        .checked_mul(2)
        .and_then(|p2| in_h.checked_add(p2))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!("padded_h: in_h={in_h} + 2 * padding_h={padding_h}"),
            })
        })?;
    let out_h = padded_h
        .checked_sub(eff_kh)
        .map(|v| v / stride_h + 1)
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!("out_h: padded_h={padded_h} < eff_kh={eff_kh}"),
            })
        })?;
    let eff_kw = dilation_w
        .checked_mul(kernel_w - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!(
                    "effective_kernel_w: dilation_w={dilation_w} * (kW={kernel_w} - 1) + 1"
                ),
            })
        })?;
    let padded_w = padding_w
        .checked_mul(2)
        .and_then(|p2| in_w.checked_add(p2))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!("padded_w: in_w={in_w} + 2 * padding_w={padding_w}"),
            })
        })?;
    let out_w = padded_w
        .checked_sub(eff_kw)
        .map(|v| v / stride_w + 1)
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv2dArithmeticOverflow {
                context: format!("out_w: padded_w={padded_w} < eff_kw={eff_kw}"),
            })
        })?;
    // Preserve leading dimensions, replace last 3 with [out_ch, out_h, out_w]
    let mut output_shape = input_shape[..input_shape.len() - 3].to_vec();
    output_shape.push(out_channels);
    output_shape.push(out_h);
    output_shape.push(out_w);
    Ok(output_shape)
}

/// Compute ConvTranspose1d output shape.
///
/// ConvTranspose1d: `[*, C_in, L_in]` → `[*, C_out, L_out]`
/// where `L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1`.
/// `C_out = weight_shape[1] * groups`.
pub(crate) fn conv_transpose1d_output_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    output_padding: usize,
) -> Result<Vec<usize>, TensorIRError> {
    if input_shape.len() < 2 {
        return Err(TensorIRConvError::ConvTranspose1dInputRankTooLow {
            rank: input_shape.len(),
        }
        .into());
    }
    if weight_shape.len() != 3 {
        return Err(TensorIRConvError::ConvTranspose1dWeightShape {
            shape: weight_shape.to_vec(),
        }
        .into());
    }
    let in_len = input_shape[input_shape.len() - 1];
    let out_ch_per_group = weight_shape[1]; // Note: swapped vs Conv1d
    let out_channels = out_ch_per_group.checked_mul(groups).ok_or_else(|| {
        TensorIRError::from(TensorIRConvError::ConvTranspose1dArithmeticOverflow {
            context: format!("out_ch_per_group={out_ch_per_group} * groups={groups}"),
        })
    })?;
    let kernel_size = weight_shape[2];
    if stride == 0 {
        return Err(TensorIRConvError::ConvTranspose1dZeroStride.into());
    }
    // out_length = (in_len - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    let expanded = in_len
        .checked_sub(1)
        .and_then(|v| v.checked_mul(stride))
        .and_then(|base| {
            dilation
                .checked_mul(kernel_size.checked_sub(1)?)
                .and_then(|dk| base.checked_add(dk))
        })
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::ConvTranspose1dArithmeticOverflow {
                context: format!(
                    "(in_len={in_len} - 1) * stride={stride} + dilation={dilation} \
                     * (kernel_size={kernel_size} - 1) + 1"
                ),
            })
        })?;
    let double_pad = padding.checked_mul(2).ok_or_else(|| {
        TensorIRError::from(TensorIRConvError::ConvTranspose1dArithmeticOverflow {
            context: format!("2 * padding={padding}"),
        })
    })?;
    let base_len = expanded.saturating_sub(double_pad);
    let out_len = base_len.checked_add(output_padding).ok_or_else(|| {
        TensorIRError::from(TensorIRConvError::ConvTranspose1dArithmeticOverflow {
            context: format!("base_len={base_len} + output_padding={output_padding}"),
        })
    })?;
    if out_len == 0 {
        return Err(TensorIRConvError::ConvTranspose1dOutputNonPositive {
            out_length: if expanded >= double_pad {
                (expanded - double_pad) as isize
            } else {
                expanded as isize - double_pad as isize
            },
            in_length: in_len,
            stride,
            kernel_size,
            padding,
        }
        .into());
    }
    let mut output_shape = input_shape[..input_shape.len() - 2].to_vec();
    output_shape.push(out_channels);
    output_shape.push(out_len);
    Ok(output_shape)
}

/// Compute ConvTranspose2d output shape.
///
/// ConvTranspose2d: `[*, C_in, H_in, W_in]` → `[*, C_out, H_out, W_out]`
/// where `D_out = (D_in - 1) * stride - 2 * padding + dilation * (K - 1) + output_padding + 1`.
/// `C_out = weight_shape[1] * groups`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_transpose2d_output_shape(
    input_shape: &[usize],
    weight_shape: &[usize],
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    output_padding_h: usize,
    output_padding_w: usize,
) -> Result<Vec<usize>, TensorIRError> {
    if input_shape.len() < 3 {
        return Err(TensorIRConvError::ConvTranspose2dInputRankTooLow {
            rank: input_shape.len(),
        }
        .into());
    }
    if weight_shape.len() != 4 {
        return Err(TensorIRConvError::ConvTranspose2dWeightShape {
            shape: weight_shape.to_vec(),
        }
        .into());
    }
    let in_h = input_shape[input_shape.len() - 2];
    let in_w = input_shape[input_shape.len() - 1];
    let out_ch_per_group = weight_shape[1];
    let out_channels = out_ch_per_group.checked_mul(groups).ok_or_else(|| {
        TensorIRError::from(TensorIRConvError::ConvTranspose2dArithmeticOverflow {
            context: format!("out_ch_per_group={out_ch_per_group} * groups={groups}"),
        })
    })?;
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];
    if stride_h == 0 || stride_w == 0 {
        return Err(TensorIRConvError::ConvTranspose2dZeroStride { stride_h, stride_w }.into());
    }

    // Helper: compute one spatial dimension of transposed conv output.
    // out = (in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
    let compute_dim = |in_dim: usize,
                       s: usize,
                       p: usize,
                       d: usize,
                       k: usize,
                       op: usize,
                       label: &str|
     -> Result<usize, TensorIRError> {
        let expanded = in_dim
            .checked_sub(1)
            .and_then(|v| v.checked_mul(s))
            .and_then(|base| {
                d.checked_mul(k.checked_sub(1)?)
                    .and_then(|dk| base.checked_add(dk))
            })
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| {
                TensorIRError::from(TensorIRConvError::ConvTranspose2dArithmeticOverflow {
                    context: format!(
                        "{label}: (in={in_dim}-1)*stride={s} + dilation={d}*(k={k}-1) + 1"
                    ),
                })
            })?;
        let double_pad = p.checked_mul(2).ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::ConvTranspose2dArithmeticOverflow {
                context: format!("{label}: 2 * padding={p}"),
            })
        })?;
        Ok(expanded.saturating_sub(double_pad) + op)
    };

    let out_h: usize = compute_dim(
        in_h,
        stride_h,
        padding_h,
        dilation_h,
        kernel_h,
        output_padding_h,
        "h",
    )?;
    let out_w: usize = compute_dim(
        in_w,
        stride_w,
        padding_w,
        dilation_w,
        kernel_w,
        output_padding_w,
        "w",
    )?;

    let mut output_shape = input_shape[..input_shape.len() - 3].to_vec();
    output_shape.push(out_channels);
    output_shape.push(out_h);
    output_shape.push(out_w);
    Ok(output_shape)
}

/// Compute Pool2d (AvgPool2d / MaxPool2d) output shape.
///
/// Pool2d: `[*, C, H, W]` → `[*, C, H_out, W_out]`
/// where `H_out = (H + 2*pad_h - kernel_h) / stride_h + 1`.
pub(crate) fn pool2d_output_shape(
    input_shape: &[usize],
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
) -> Result<Vec<usize>, TensorIRError> {
    if input_shape.len() < 3 {
        return Err(TensorIRConvError::Pool2dInputRankTooLow {
            rank: input_shape.len(),
        }
        .into());
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(TensorIRConvError::Pool2dZeroStride { stride_h, stride_w }.into());
    }
    if kernel_h == 0 || kernel_w == 0 {
        return Err(TensorIRConvError::Pool2dZeroKernelSize { kernel_h, kernel_w }.into());
    }
    let in_h = input_shape[input_shape.len() - 2];
    let in_w = input_shape[input_shape.len() - 1];
    let padded_h = padding_h
        .checked_mul(2)
        .and_then(|p2| in_h.checked_add(p2))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Pool2dArithmeticOverflow {
                context: format!("padded_h: in_h={in_h} + 2 * padding_h={padding_h}"),
            })
        })?;
    let padded_w = padding_w
        .checked_mul(2)
        .and_then(|p2| in_w.checked_add(p2))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Pool2dArithmeticOverflow {
                context: format!("padded_w: in_w={in_w} + 2 * padding_w={padding_w}"),
            })
        })?;
    if padded_h < kernel_h || padded_w < kernel_w {
        return Err(TensorIRConvError::Pool2dKernelTooLarge {
            kernel_h,
            kernel_w,
            padded_h,
            padded_w,
        }
        .into());
    }
    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    // Preserve leading dims including channels, replace last 2 (H, W) with pooled dims.
    let mut output_shape = input_shape[..input_shape.len() - 2].to_vec();
    output_shape.push(out_h);
    output_shape.push(out_w);
    Ok(output_shape)
}
