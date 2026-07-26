// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Backward rules for pooling operations (MaxPool1d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d).

use nn_core::dyn_tensor::DynTensor;
use nn_core::tensor::checked_dim_product;

use crate::error::{AutodiffError, Result};
use crate::grad::GradStore;
use crate::op::Op;

use super::accumulate;

/// Dispatch backward rule for pooling operations.
pub(super) fn backward_pool(op: &Op, grad: &DynTensor, grads: &mut GradStore) -> Result<()> {
    match op {
        Op::MaxPool1d {
            input,
            indices,
            kernel_size: _,
            stride: _,
            padding: _,
        }
        | Op::MaxPool2d {
            input,
            indices,
            kernel_size: _,
            stride: _,
            padding: _,
        } => {
            // MaxPool1d/MaxPool2d backward: scatter gradient to argmax positions.
            // Device-agnostic via scatter_add on flattened tensors.
            // The backward rule is identical for 1D and 2D — both use flat argmax indices.
            let input_dims = input.tensor().dims();
            let numel = checked_dim_product(input_dims)?;
            let device = grad.device();

            // Flatten grad and indices to 1D for scatter_add along dim 0
            let grad_flat = grad.flatten_all()?;
            let idx_flat = indices.flatten_all()?.to_dtype(nn_core::DType::U32)?;

            // scatter_add_into: zeros[indices[i]] += grad[i]
            // _into avoids cloning the zeros tensor (refcount == 1).
            let zeros = DynTensor::zeros(&[numel], grad.dtype(), &device)?;
            let grad_t = zeros.scatter_add_into(0, &idx_flat, &grad_flat)?;
            let grad_t = grad_t.reshape(input_dims)?;

            accumulate(input, &grad_t, grads)
        }
        Op::AvgPool2d {
            input,
            kernel_size,
            stride,
            padding,
        } => backward_avg_pool2d(input, grad, *kernel_size, *stride, *padding, grads),
        Op::AdaptiveAvgPool2d {
            input,
            output_h,
            output_w,
        } => backward_adaptive_avg_pool2d(input, grad, *output_h, *output_w, grads),
        other => Err(super::unsupported(other)),
    }
}

/// AvgPool2d backward: device-agnostic via conv_transpose2d.
///
/// Three steps:
/// 1. Compute per-window valid element counts via `conv2d(ones, ones_kernel)`.
///    This correctly handles `count_include_pad=false` with padding — each
///    output position gets the exact count of valid (non-padded) input elements
///    in its window. (Note: `avg_pool2d(ones) * K²` does NOT work because
///    `avg_pool2d` with `count_include_pad=false` returns `count/count = 1.0`
///    for all-ones input, giving K² everywhere regardless of padding.)
/// 2. Normalize gradient by per-window count via element-wise division.
/// 3. Spread normalized gradient back to input via depthwise `conv_transpose2d`
///    with an all-ones kernel (groups=channels).
fn backward_avg_pool2d(
    input: &std::sync::Arc<crate::tracked::TrackedTensor>,
    grad: &DynTensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    grads: &mut GradStore,
) -> Result<()> {
    let in_dims = input.tensor().dims();
    let (_, channels, in_h, in_w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    let device = grad.device();

    // Step 1: Compute per-window valid element counts.
    // conv2d(ones_input, ones_kernel) sums 1s in each window, giving
    // the exact count of valid (non-padded) positions per output element.
    let ones_input = DynTensor::ones(&[1, 1, in_h, in_w], grad.dtype(), &device)?;
    let ones_kernel = DynTensor::ones(&[1, 1, kernel_size, kernel_size], grad.dtype(), &device)?;
    let counts = ones_input.conv2d(&ones_kernel, padding, stride, 1, 1)?;

    // Step 2: Normalize gradient by count (broadcast across batch and channels).
    let grad_norm = grad.div(&counts)?;

    // Step 3: Spread via depthwise conv_transpose2d with ones kernel.
    // groups=channels makes each channel independent (depthwise).
    let spread_kernel = DynTensor::ones(
        &[channels, 1, kernel_size, kernel_size],
        grad.dtype(),
        &device,
    )?;

    // output_padding restores exact input spatial dims when stride > 1.
    let base_h = in_h + 2 * padding;
    let base_w = in_w + 2 * padding;
    let output_padding_h = if base_h >= kernel_size {
        (base_h - kernel_size) % stride
    } else {
        0
    };
    let output_padding_w = if base_w >= kernel_size {
        (base_w - kernel_size) % stride
    } else {
        0
    };
    // conv_transpose2d accepts a single output_padding for both H and W.
    // Using max() may inflate one dimension, but values in the kept region
    // are unaffected because the extra rows/columns fall beyond the trim
    // boundary. narrow() below restores exact input spatial dims.
    let output_padding = output_padding_h.max(output_padding_w);

    let pad2 = [padding, padding];
    let stride2 = [stride, stride];
    let grad_input = grad_norm.conv_transpose2d(
        &spread_kernel,
        pad2,
        [output_padding, output_padding],
        stride2,
        [1, 1],
        channels,
    )?;

    // Trim if spatial dimensions don't match exactly (rare edge case).
    let gi_dims = grad_input.dims();
    let grad_input = if gi_dims[2] != in_h || gi_dims[3] != in_w {
        grad_input.narrow(2, 0, in_h)?.narrow(3, 0, in_w)?
    } else {
        grad_input
    };

    accumulate(input, &grad_input, grads)
}

/// AdaptiveAvgPool2d backward: distribute gradient per window.
///
/// Three tiers for device-agnostic coverage:
/// 1. Global pooling (output 1×1): `expand` + scalar division — fully device-agnostic.
/// 2. Uniform windows (in_h % out_h == 0): `upsample_nearest_2d` + scalar division.
/// 3. General (variable windows): CPU loop fallback with explicit device transfer.
fn backward_adaptive_avg_pool2d(
    input: &std::sync::Arc<crate::tracked::TrackedTensor>,
    grad: &DynTensor,
    output_h: usize,
    output_w: usize,
    grads: &mut GradStore,
) -> Result<()> {
    let in_dims = input.tensor().dims();
    let (_, _, in_h, in_w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);

    // Tier 1: Global average pooling (output 1×1) — most common case.
    if output_h == 1 && output_w == 1 {
        let window_size = (in_h * in_w) as f64;
        let grad_t = grad.mul_scalar(1.0 / window_size)?.expand(in_dims)?;
        return accumulate(input, &grad_t, grads);
    }

    // Tier 2: Uniform windows (exact division) — upsample_nearest_2d.
    if in_h % output_h == 0 && in_w % output_w == 0 {
        let scale_h = in_h / output_h;
        let scale_w = in_w / output_w;
        let window_size = (scale_h * scale_w) as f64;
        let grad_t = grad
            .mul_scalar(1.0 / window_size)?
            .upsample_nearest_2d(scale_h, scale_w)?;
        return accumulate(input, &grad_t, grads);
    }

    // Tier 3: General case — CPU loop for variable window sizes.
    backward_adaptive_avg_pool2d_cpu(input, grad, output_h, output_w, in_dims, grads)
}

/// CPU fallback for AdaptiveAvgPool2d backward with non-uniform window sizes.
fn backward_adaptive_avg_pool2d_cpu(
    input: &std::sync::Arc<crate::tracked::TrackedTensor>,
    grad: &DynTensor,
    output_h: usize,
    output_w: usize,
    in_dims: &[usize],
    grads: &mut GradStore,
) -> Result<()> {
    let (batch, channels, in_h, in_w) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    let device = grad.device();
    let grad_cpu = if device.is_gpu() {
        grad.to_device(&nn_core::Device::Cpu)?
    } else {
        grad.clone()
    };
    let numel = checked_dim_product(in_dims)?;
    let mut grad_input = vec![0.0f32; numel];

    let grad_c = grad_cpu.contiguous()?;
    let grad_flat = grad_c.to_f32_array()?;
    let g = grad_flat.as_slice().ok_or(AutodiffError::NotContiguous {
        op: "AdaptiveAvgPool2d backward",
    })?;

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..output_h {
                let start_h = (oh * in_h) / output_h;
                let end_h = ((oh + 1) * in_h).div_ceil(output_h);
                for ow in 0..output_w {
                    let start_w = (ow * in_w) / output_w;
                    let end_w = ((ow + 1) * in_w).div_ceil(output_w);
                    let window_size = (end_h - start_h) * (end_w - start_w);
                    if window_size == 0 {
                        continue;
                    }
                    let out_idx = b * channels * output_h * output_w
                        + c * output_h * output_w
                        + oh * output_w
                        + ow;
                    let g_val = g[out_idx] / window_size as f32;
                    for ih in start_h..end_h {
                        for iw in start_w..end_w {
                            let in_idx =
                                b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                            grad_input[in_idx] += g_val;
                        }
                    }
                }
            }
        }
    }

    let mut grad_t = DynTensor::from_vec(grad_input, in_dims, &nn_core::Device::Cpu)?;
    if device.is_gpu() {
        grad_t = grad_t.to_device(&device)?;
    }
    accumulate(input, &grad_t, grads)
}
