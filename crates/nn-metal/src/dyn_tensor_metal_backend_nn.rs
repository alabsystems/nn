// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! [`GpuNnOps`] implementation for `MetalDynBackend`.
//!
//! 24+ NN methods: softmax, log_softmax, conv1d, conv2d, conv3d, conv_transpose1d,
//! layer_norm, group_norm, rms_norm, instance_norm, snake_tensor,
//! adain_snake, adain_leaky_relu, rope, lstm_cell, lstm_sequence,
//! clamp, clamp_min, clamp_max, sdpa, sdpa_causal, max_pool2d, avg_pool2d,
//! adaptive_avg_pool2d.
//! Extracted from `dyn_tensor_metal_backend_impl.rs` (#1917).

use nn_core::dyn_tensor::{BinaryOp, DynTensor, GpuNnOps};
use nn_core::{DType, Result};

use super::helpers::{
    ensure_matching_dtype, gpu_norm_with_dtype_promotion, promote_to_f32, FUSED_NORM_MIN_REDUCTION,
};
use super::MetalDynBackend;

impl GpuNnOps for MetalDynBackend {
    fn softmax(&self, x: &DynTensor, dim: usize) -> Option<Result<DynTensor>> {
        // Metal MSL softmax only supports last-axis; return None for other axes
        // so the caller falls back to CPU decomposition.
        if dim + 1 != x.rank() {
            return crate::gpu_fallback("softmax", "non-last axis not supported on Metal");
        }
        Some(Self::gpu_softmax(x, dim))
    }

    fn log_softmax(&self, x: &DynTensor, dim: usize) -> Option<Result<DynTensor>> {
        // Metal MSL softmax only supports last-axis; return None for other axes.
        if dim + 1 != x.rank() {
            return crate::gpu_fallback("log_softmax", "non-last axis not supported on Metal");
        }
        // BF16/F16: promote → F32 → kernel → cast back (#2981 D2 regression).
        // The native log_softmax kernel (softmax → log multi-step composition)
        // produces incorrect results with BF16 intermediate buffers.
        Some(gpu_norm_with_dtype_promotion(x, |x32| {
            Self::gpu_log_softmax(&x32, dim)
        }))
    }

    fn conv1d(
        &self,
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Option<Result<DynTensor>> {
        Some(Self::gpu_conv1d(
            input, kernel, bias, padding, stride, dilation, groups,
        ))
    }

    fn conv2d(
        &self,
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Option<Result<DynTensor>> {
        Some(Self::gpu_conv2d(
            input, kernel, bias, padding, stride, dilation, groups,
        ))
    }

    fn conv3d(
        &self,
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: [usize; 3],
        stride: [usize; 3],
        dilation: [usize; 3],
        groups: usize,
    ) -> Option<Result<DynTensor>> {
        Some(Self::gpu_conv3d(
            input, kernel, bias, padding, stride, dilation, groups,
        ))
    }

    fn conv_transpose1d(
        &self,
        input: &DynTensor,
        kernel: &DynTensor,
        bias: Option<&DynTensor>,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Option<Result<DynTensor>> {
        Some(Self::gpu_conv_transpose1d(
            input,
            kernel,
            bias,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        ))
    }

    // -- Fused normalization ops (#1290, #3294, #3348) -----------------------
    //
    // Two paths based on reduction dimension size (#3348 D5/D7):
    //
    // 1. Fused MSL path (reduction_dim >= FUSED_NORM_MIN_REDUCTION):
    //    Hand-written MSL kernel, native half I/O with float accumulators.
    //    Saves 2 dtype-cast dispatches for F16/BF16 inputs.
    //
    // 2. Decomposed IR path (reduction_dim < FUSED_NORM_MIN_REDUCTION):
    //    TensorBlockBuilder IR compiled to MSL, F32 promotion required.
    //    Better threadgroup occupancy for small reduction dimensions.
    //
    // The fused kernel uses rsqrt (no per-op finiteness check); model-level
    // guards (#941) catch non-finite values at stage boundaries.

    fn layer_norm(
        &self,
        x: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
    ) -> Option<Result<DynTensor>> {
        // LayerNorm always uses the fused MSL kernel (has its own dedicated file).
        Some(gpu_norm_with_dtype_promotion(x, |x32| {
            let w32 = promote_to_f32(weight)?;
            let b32 = promote_to_f32(bias)?;
            Self::gpu_layer_norm_fused(&x32, &w32, &b32, eps)
        }))
    }

    fn rms_norm(&self, x: &DynTensor, weight: &DynTensor, eps: f64) -> Option<Result<DynTensor>> {
        // RmsNorm reduces over last dim. Route to fused MSL for large dims (#3294),
        // decomposed IR for small dims (#3348 regression fix).
        let hidden_dim = x.dims().last().copied().unwrap_or(0);
        if hidden_dim >= FUSED_NORM_MIN_REDUCTION {
            // Fused MSL kernel: native half I/O, single dispatch, float accumulators.
            let dtype = x.dtype();
            Some((|| -> Result<DynTensor> {
                let w = ensure_matching_dtype(weight, dtype)?;
                Self::gpu_rms_norm_fused(x, &w, eps)
            })())
        } else {
            // Decomposed IR path: F32 promotion + TensorBlockBuilder compiled kernel.
            Some(gpu_norm_with_dtype_promotion(x, |x32| {
                let w32 = promote_to_f32(weight)?;
                Self::gpu_rms_norm(&x32, &w32, eps)
            }))
        }
    }

    fn group_norm(
        &self,
        x: &DynTensor,
        num_groups: usize,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
    ) -> Option<Result<DynTensor>> {
        // GroupNorm reduces over (channels_per_group * spatial). Route to fused MSL
        // when reduction is large enough (#3294), decomposed IR otherwise (#3348).
        let dims = x.dims();
        let channels = if dims.len() >= 2 { dims[1] } else { 0 };
        let channels_per_group = if num_groups > 0 {
            channels / num_groups.max(1)
        } else {
            0
        };
        let spatial: usize = dims.get(2..).map_or(1, |s| s.iter().product());
        let reduction_dim = channels_per_group.saturating_mul(spatial.max(1));

        if reduction_dim >= FUSED_NORM_MIN_REDUCTION {
            // Fused MSL kernel: native half I/O, single dispatch, float accumulators.
            let dtype = x.dtype();
            Some((|| -> Result<DynTensor> {
                let w = ensure_matching_dtype(weight, dtype)?;
                let b = ensure_matching_dtype(bias, dtype)?;
                Self::gpu_group_norm_fused(x, num_groups, &w, &b, eps)
            })())
        } else {
            // Decomposed IR path: F32 promotion + TensorBlockBuilder compiled kernel.
            Some(gpu_norm_with_dtype_promotion(x, |x32| {
                let w32 = promote_to_f32(weight)?;
                let b32 = promote_to_f32(bias)?;
                Self::gpu_group_norm(&x32, num_groups, &w32, &b32, eps)
            }))
        }
    }

    fn instance_norm(&self, x: &DynTensor, eps: f64) -> Option<Result<DynTensor>> {
        Some(gpu_norm_with_dtype_promotion(x, |x32| {
            // Use fused single-dispatch kernel (#2472) instead of 7-dispatch
            // IR decomposition. Falls back to decomposed path for rank < 3.
            if x32.rank() >= 3 {
                Self::gpu_instance_norm_fused(&x32, eps)
            } else {
                Self::gpu_instance_norm(&x32, eps)
            }
        }))
    }

    // -- Fused BatchNorm (#4324) ------------------------------------------
    //
    // BatchNorm inference uses precomputed running statistics -- no reduction
    // needed. Single fused kernel replaces ~6 separate dispatches.

    fn batch_norm(
        &self,
        x: &DynTensor,
        running_mean: &DynTensor,
        running_var: &DynTensor,
        weight: Option<&DynTensor>,
        bias: Option<&DynTensor>,
        eps: f64,
    ) -> Option<Result<DynTensor>> {
        // BF16/F16: promote input → F32 → fused kernel → cast back.
        // running_mean and running_var are always F32 (PyTorch convention).
        Some(gpu_norm_with_dtype_promotion(x, |x32| {
            let rm32 = promote_to_f32(running_mean)?;
            let rv32 = promote_to_f32(running_var)?;
            let w32 = weight.map(promote_to_f32).transpose()?;
            let b32 = bias.map(promote_to_f32).transpose()?;
            Self::gpu_batch_norm_fused(&x32, &rm32, &rv32, w32.as_ref(), b32.as_ref(), eps)
        }))
    }

    // -- Fused Snake (#2226, #3294) ----------------------------------------

    fn snake_tensor(&self, x: &DynTensor, alpha: &DynTensor) -> Option<Result<DynTensor>> {
        // Snake is elementwise (no reduction), so the fused MSL kernel is always
        // better — 1 dispatch instead of ~6, and native half I/O saves 2 casts.
        let dtype = x.dtype();
        Some((|| -> Result<DynTensor> {
            let a = ensure_matching_dtype(alpha, dtype)?;
            Self::gpu_snake_tensor_fused(x, &a)
        })())
    }

    // -- Fused AdaIN+Snake (#2227) ----------------------------------------

    fn adain_snake(
        &self,
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        alpha: &DynTensor,
        eps: f64,
    ) -> Option<Result<DynTensor>> {
        Some(gpu_norm_with_dtype_promotion(x, |x32| {
            let g32 = promote_to_f32(gamma)?;
            let b32 = promote_to_f32(beta)?;
            let a32 = promote_to_f32(alpha)?;
            Self::gpu_adain_snake(&x32, &g32, &b32, &a32, eps, true)
        }))
    }

    // -- Fused AdaIN+LeakyRelu (#2472) ------------------------------------

    fn adain_leaky_relu(
        &self,
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        eps: f64,
        slope: f64,
    ) -> Option<Result<DynTensor>> {
        Some(gpu_norm_with_dtype_promotion(x, |x32| {
            let g32 = promote_to_f32(gamma)?;
            let b32 = promote_to_f32(beta)?;
            Self::gpu_adain_leaky_relu_fused(&x32, &g32, &b32, eps, slope)
        }))
    }

    // -- Fused RoPE (#1363) -----------------------------------------------

    fn rope(&self, x: &DynTensor, cos: &DynTensor, sin: &DynTensor) -> Option<Result<DynTensor>> {
        Some(Self::gpu_rope(x, cos, sin))
    }

    // -- Fused LSTM cell (#1373) ------------------------------------------

    fn lstm_cell(
        &self,
        input: &DynTensor,
        hidden: &DynTensor,
        cell: &DynTensor,
        w_ih: &DynTensor,
        w_hh: &DynTensor,
        bias: Option<&DynTensor>,
        hidden_size: usize,
    ) -> Option<Result<(DynTensor, DynTensor)>> {
        Some(Self::gpu_lstm_cell(
            input,
            hidden,
            cell,
            w_ih,
            w_hh,
            bias,
            hidden_size,
        ))
    }

    // -- Fused LSTM sequence (#1805) --------------------------------------

    fn lstm_sequence(
        &self,
        input: &DynTensor,
        w_ih: &DynTensor,
        w_hh: &DynTensor,
        bias: Option<&DynTensor>,
        h0: &DynTensor,
        c0: &DynTensor,
        hidden_size: usize,
    ) -> Option<Result<(DynTensor, DynTensor, DynTensor)>> {
        Self::gpu_lstm_sequence(input, w_ih, w_hh, bias, h0, c0, hidden_size, false)
    }

    // -- Fused Clamp (#1815 D2a) --------------------------------------------

    fn clamp(&self, x: &DynTensor, min: f64, max: f64) -> Option<Result<DynTensor>> {
        // BF16/F16 supported: dispatch_def emits `half` buffers with `float`
        // accumulators. Scalar constants baked into MSL (#3230 Gap 2).
        match x.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {}
            _ => return None,
        }
        Some(Self::gpu_clamp(x, min, max))
    }

    fn clamp_min(&self, x: &DynTensor, min: f64) -> Option<Result<DynTensor>> {
        match x.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {}
            _ => return None,
        }
        Some(Self::gpu_clamp_min(x, min))
    }

    fn clamp_max(&self, x: &DynTensor, max: f64) -> Option<Result<DynTensor>> {
        match x.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {}
            _ => return None,
        }
        Some(Self::gpu_clamp_max(x, max))
    }

    // -- Fused SDPA / Flash Attention (#2434) ------------------------------

    fn sdpa(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        v: &DynTensor,
        mask: Option<&DynTensor>,
        scale: f64,
    ) -> Option<Result<DynTensor>> {
        // GPU tensors only — CPU tensors fall through to decomposed path (#2567).
        if !q.device().is_gpu() || !k.device().is_gpu() || !v.device().is_gpu() {
            return None;
        }
        // Float dtypes only (F32, BF16, F16), 4D inputs, head_dim <= 128.
        if !matches!(q.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return None;
        }
        if mask.is_some() {
            // Explicit mask tensors fall back to decomposed path.
            // Use sdpa_causal() for fused causal masking.
            return None;
        }
        if q.rank() != 4 || k.rank() != 4 {
            return None;
        }
        let d = q.dims()[3];
        if d > 128 || d == 0 {
            return None;
        }
        // GQA: H_q must be a multiple of H_kv.
        let h_q = q.dims()[1];
        let h_kv = k.dims()[1];
        if h_kv == 0 || !h_q.is_multiple_of(h_kv) {
            return None;
        }
        Some(Self::gpu_flash_attention(q, k, v, scale, false))
    }

    fn sdpa_causal(
        &self,
        q: &DynTensor,
        k: &DynTensor,
        v: &DynTensor,
        scale: f64,
    ) -> Option<Result<DynTensor>> {
        // GPU tensors only — CPU tensors fall through to decomposed path (#2567).
        if !q.device().is_gpu() || !k.device().is_gpu() || !v.device().is_gpu() {
            return None;
        }
        // Float dtypes only (F32, BF16, F16), 4D inputs, head_dim <= 128.
        if !matches!(q.dtype(), DType::F32 | DType::BF16 | DType::F16) {
            return None;
        }
        if q.rank() != 4 || k.rank() != 4 {
            return None;
        }
        let d = q.dims()[3];
        if d > 128 || d == 0 {
            return None;
        }
        // Causal requires S_q == S_kv.
        if q.dims()[2] != k.dims()[2] {
            return None;
        }
        // GQA: H_q must be a multiple of H_kv.
        let h_q = q.dims()[1];
        let h_kv = k.dims()[1];
        if h_kv == 0 || !h_q.is_multiple_of(h_kv) {
            return None;
        }
        Some(Self::gpu_flash_attention(q, k, v, scale, true))
    }

    fn scalar_binary_op(
        &self,
        op: BinaryOp,
        x: &DynTensor,
        scalar: f64,
    ) -> Option<Result<DynTensor>> {
        // BF16/F16 supported: dispatch_def emits `half` buffers with `float`
        // accumulators. Same codepath as gpu_binary and gpu_unary (#3230 Gap 2).
        if !x.device().is_gpu() {
            return None;
        }
        match x.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {}
            _ => return None,
        }
        Some(Self::gpu_scalar_binary(op, x, scalar))
    }

    // -- Pool2d GPU kernels (#4323) ----------------------------------------------

    fn max_pool2d(
        &self,
        x: &DynTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Option<Result<DynTensor>> {
        match x.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {}
            _ => return None,
        }
        Some(Self::gpu_max_pool2d(x, kernel_size, stride, padding))
    }

    fn avg_pool2d(
        &self,
        x: &DynTensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Option<Result<DynTensor>> {
        match x.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {}
            _ => return None,
        }
        Some(Self::gpu_avg_pool2d(x, kernel_size, stride, padding))
    }

    fn adaptive_avg_pool2d(
        &self,
        x: &DynTensor,
        out_h: usize,
        out_w: usize,
    ) -> Option<Result<DynTensor>> {
        match x.dtype() {
            DType::F32 | DType::BF16 | DType::F16 => {}
            _ => return None,
        }
        Some(Self::gpu_adaptive_avg_pool2d(x, out_h, out_w))
    }

    // -- Bilinear resize (#3535) -----------------------------------------------

    fn resize_bilinear(
        &self,
        x: &DynTensor,
        target_h: usize,
        target_w: usize,
    ) -> Option<Result<DynTensor>> {
        Self::gpu_resize_bilinear(x, target_h, target_w)
    }

    // -- MoE scatter-gather (#3547) -------------------------------------------

    fn moe_scatter_gather(
        &self,
        hidden: &DynTensor,
        indices: &DynTensor,
        weights: &DynTensor,
        expert_gate_weights: &[DynTensor],
        expert_up_weights: &[DynTensor],
        expert_down_weights: &[DynTensor],
        num_experts: usize,
    ) -> Option<Result<DynTensor>> {
        Self::gpu_moe_scatter_gather(
            hidden,
            indices,
            weights,
            expert_gate_weights,
            expert_up_weights,
            expert_down_weights,
            num_experts,
        )
    }
}
