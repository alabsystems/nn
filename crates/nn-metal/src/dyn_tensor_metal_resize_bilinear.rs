// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native bilinear interpolation resize for [`MetalDynBackend`].
//!
//! Implements `resize_bilinear(target_h, target_w)` as a single Metal
//! compute dispatch, replacing the CPU round-trip path. Each GPU thread
//! computes one output pixel using the half-pixel-center coordinate
//! mapping: `src = (dst + 0.5) * (in_size / out_size) - 0.5`, clamped
//! to `[0, in_size - 1]`.
//!
//! Input: `[..., H_in, W_in]` (rank >= 3). Output: `[..., target_h, target_w]`.
//!
//! Part of #3535.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Result;

use super::MetalTensorData;

/// MSL kernel source for bilinear resize.
///
/// One thread per output element. Half-pixel-center coordinate mapping
/// matches PyTorch `F.interpolate(mode='bilinear', align_corners=False)`.
const RESIZE_BILINEAR_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void resize_bilinear_f32(
    device const float* input       [[buffer(0)]],
    device float* output            [[buffer(1)]],
    device const uint& total_els    [[buffer(2)]],
    device const uint& in_h         [[buffer(3)]],
    device const uint& in_w         [[buffer(4)]],
    device const uint& out_h        [[buffer(5)]],
    device const uint& out_w        [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= total_els) return;

    // Decompose tid into (batch_channel, oh, ow).
    uint out_hw = out_h * out_w;
    uint bc = tid / out_hw;
    uint rem = tid % out_hw;
    uint oh = rem / out_w;
    uint ow = rem % out_w;

    // Half-pixel-center coordinate mapping:
    // src = (dst + 0.5) * (in_size / out_size) - 0.5, clamped to [0, in_size - 1].
    float scale_y = float(in_h) / float(out_h);
    float scale_x = float(in_w) / float(out_w);

    float src_y = (float(oh) + 0.5f) * scale_y - 0.5f;
    float src_x = (float(ow) + 0.5f) * scale_x - 0.5f;

    src_y = clamp(src_y, 0.0f, float(in_h - 1));
    src_x = clamp(src_x, 0.0f, float(in_w - 1));

    uint y0 = min(uint(floor(src_y)), in_h - 1);
    uint y1 = min(y0 + 1, in_h - 1);
    uint x0 = min(uint(floor(src_x)), in_w - 1);
    uint x1 = min(x0 + 1, in_w - 1);

    float wy = src_y - float(y0);
    float wx = src_x - float(x0);

    uint base = bc * (in_h * in_w);
    float v00 = input[base + y0 * in_w + x0];
    float v01 = input[base + y0 * in_w + x1];
    float v10 = input[base + y1 * in_w + x0];
    float v11 = input[base + y1 * in_w + x1];

    float val = v00 * (1.0f - wy) * (1.0f - wx)
              + v01 * (1.0f - wy) * wx
              + v10 * wy * (1.0f - wx)
              + v11 * wy * wx;

    output[tid] = val;
}
"#;

impl super::MetalDynBackend {
    /// GPU-native bilinear interpolation resize to absolute target dimensions.
    ///
    /// Returns `None` for non-F32 dtypes (fall back to CPU).
    pub(super) fn gpu_resize_bilinear(
        x: &DynTensor,
        target_h: usize,
        target_w: usize,
    ) -> Option<Result<DynTensor>> {
        if Self::validate_f32_buffer(x, "gpu_resize_bilinear").is_err() {
            return crate::gpu_fallback("resize_bilinear", "non-f32 dtype not supported on Metal");
        }

        let shape = x.dims();
        let rank = shape.len();
        if rank < 3 {
            return Some(Err(nn_core::TensorError::RankMismatch {
                expected: 3,
                actual: rank,
            }));
        }

        Some(Self::gpu_resize_bilinear_inner(x, target_h, target_w))
    }

    /// Inner dispatch that returns `Result<DynTensor>` so `?` works.
    fn gpu_resize_bilinear_inner(
        x: &DynTensor,
        target_h: usize,
        target_w: usize,
    ) -> Result<DynTensor> {
        let shape = x.dims();
        let rank = shape.len();
        let in_h = shape[rank - 2];
        let in_w = shape[rank - 1];

        let x_data = x.gpu_data::<MetalTensorData>()?;

        // Compute outer (batch * channels) product.
        let outer = crate::metal_backend::checked_dim_product(&shape[..rank - 2])?;

        let total_elems = outer
            .checked_mul(target_h)
            .and_then(|v| v.checked_mul(target_w))
            .ok_or_else(|| nn_core::TensorError::DimensionOverflow {
                dims: shape.to_vec(),
            })?;

        let mut out_shape = shape.to_vec();
        out_shape[rank - 2] = target_h;
        out_shape[rank - 1] = target_w;

        Self::dispatch_raw_msl(
            RESIZE_BILINEAR_MSL,
            "resize_bilinear_f32",
            1, // param_count: 1 input buffer
            &[&x_data.buffer],
            &[x_data.byte_offset],
            total_elems,
            &out_shape,
            x.dtype(),
            vec![
                crate::to_u32(total_elems, "resize_bilinear total")?,
                crate::to_u32(in_h, "resize_bilinear in_h")?,
                crate::to_u32(in_w, "resize_bilinear in_w")?,
                crate::to_u32(target_h, "resize_bilinear out_h")?,
                crate::to_u32(target_w, "resize_bilinear out_w")?,
            ],
        )
    }
}
