// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused GPU normalization kernels for [`MetalDynBackend`].
//!
//! Provides fused LayerNorm, RmsNorm, GroupNorm, and InstanceNorm dispatch that composes
//! the full normalization (reduce→center→variance→rsqrt→scale→affine) into
//! a single `TensorKernelDef` dispatch, avoiding the 5-8 separate kernel
//! launches from the decomposed nn layer path.
//!
//! Part of #1290 (fused normalization GPU kernels).
//!
//! All norm kernels dispatch with `PrecisionTier::Strict` to enable
//! Kahan-compensated Mean reductions, improving numerical precision
//! for large hidden dimensions (#1814 D3).
//!
//! GroupNorm and InstanceNorm (grouped/per-channel normalization) are in
//! `dyn_tensor_metal_norm_ops_grouped.rs`.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError};

use crate::gpu_slice::GpuSlice;
use nn_dsl::ir::{BinOpKind, ScalarType, UnaryFnKind};
use nn_dsl::tensor_ir::ReduceOp;
use nn_dsl::{PrecisionContract, PrecisionTier, TensorBlockBuilder};

use super::kernels::{make_binop_kernel, make_sqr_kernel, make_unary_kernel};
use super::MetalTensorData;

#[path = "dyn_tensor_metal_norm_ops_grouped.rs"]
mod grouped;

impl super::MetalDynBackend {
    /// Create a Metal buffer containing a single f32 eps value.
    fn make_eps_buffer(eps: f64) -> Result<crate::buffer::MetalBuffer> {
        let ctx = Self::ctx()?;
        ctx.create_buffer(&[eps as f32]).map_err(|e| {
            TensorError::backend_failure(
                nn_core::BackendDomain::Metal,
                nn_core::BackendErrorKind::OutOfMemory,
                format!("eps buffer: {e}"),
            )
        })
    }

    /// Decomposed LayerNorm GPU kernel: `(x - mean) / sqrt(var + eps) * weight + bias`
    ///
    /// Normalizes over the last dimension. Composes the full normalization
    /// (mean, center, var, rsqrt, scale, affine) into a single dispatch graph.
    /// Production path uses `gpu_layer_norm_fused` instead; this decomposed version
    /// is retained for test comparison with the fused kernel.
    #[cfg(test)]
    pub(super) fn gpu_layer_norm(
        x: &DynTensor,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
    ) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_layer_norm")?;
        Self::validate_f32(weight, "gpu_layer_norm(weight)")?;
        Self::validate_f32(bias, "gpu_layer_norm(bias)")?;

        let shape = x.dims();
        let rank = shape.len();
        if rank == 0 {
            return Err(TensorError::InvalidShape(
                "gpu_layer_norm requires rank >= 1".into(),
            ));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let w_data = weight.gpu_data::<MetalTensorData>()?;
        let b_data = bias.gpu_data::<MetalTensorData>()?;
        let eps_buf = Self::make_eps_buffer(eps)?;

        // Reduced shape after removing last axis (IR Reduce removes the axis).
        // e.g. [B, S, D] → [B, S]
        let reduced_shape: Vec<usize> = shape[..rank - 1].to_vec();
        // For rank-1 input, reduce produces scalar [1] per IR convention.
        let reduced_shape = if reduced_shape.is_empty() {
            vec![1]
        } else {
            reduced_shape
        };

        let def = crate::kernel_def_cache::get_or_build(
            "layer_norm",
            &[shape, weight.dims(), bias.dims()],
            &[eps.to_bits()],
            x.dtype(),
            || {
                let mut bld = TensorBlockBuilder::new("dyn_layer_norm");
                let input = bld.add_input("data", shape);
                let w_node = bld.add_input("weight", weight.dims());
                let b_node = bld.add_input("bias", bias.dims());
                let eps_node = bld.add_input("eps", &[1]);
                let eps_bc = bld.add_broadcast(eps_node, &reduced_shape);

                let mean = bld.add_reduce(input, ReduceOp::Mean, rank - 1, false, &reduced_shape);
                let mean_bc = bld.add_broadcast_left(mean, shape);
                let centered = bld.add_elementwise(
                    make_binop_kernel("sub", BinOpKind::Sub),
                    &[input, mean_bc],
                    shape,
                );
                let sq = bld.add_elementwise(make_sqr_kernel(), &[centered], shape);
                let var = bld.add_reduce(sq, ReduceOp::Mean, rank - 1, false, &reduced_shape);
                let var_eps = bld.add_elementwise(
                    make_binop_kernel("add", BinOpKind::Add),
                    &[var, eps_bc],
                    &reduced_shape,
                );
                let rsqrt = bld.add_elementwise(
                    make_unary_kernel("rsqrt", UnaryFnKind::Rsqrt),
                    &[var_eps],
                    &reduced_shape,
                );
                let rsqrt_bc = bld.add_broadcast_left(rsqrt, shape);
                let normed = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[centered, rsqrt_bc],
                    shape,
                );
                let w_bc = bld.add_broadcast(w_node, shape);
                let scaled = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[normed, w_bc],
                    shape,
                );
                let b_bc = bld.add_broadcast(b_node, shape);
                let out = bld.add_elementwise(
                    make_binop_kernel("add", BinOpKind::Add),
                    &[scaled, b_bc],
                    shape,
                );

                crate::build_kernel(bld, out)
            },
        )?;

        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        Self::dispatch_def_with_contract(
            &def,
            &[
                ("data", x_data.as_gpu_slice()),
                ("weight", w_data.as_gpu_slice()),
                ("bias", b_data.as_gpu_slice()),
                ("eps", GpuSlice::zero_offset(eps_buf)),
            ],
            shape,
            x.dtype(),
            contract,
        )
    }

    /// Fused RmsNorm GPU kernel: `x / sqrt(mean(x^2) + eps) * weight`
    ///
    /// Normalizes over the last dimension. No mean-centering (unlike LayerNorm).
    pub(super) fn gpu_rms_norm(x: &DynTensor, weight: &DynTensor, eps: f64) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_rms_norm")?;
        Self::validate_f32(weight, "gpu_rms_norm(weight)")?;

        let shape = x.dims();
        let rank = shape.len();
        if rank == 0 {
            return Err(TensorError::InvalidShape(
                "gpu_rms_norm requires rank >= 1".into(),
            ));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let w_data = weight.gpu_data::<MetalTensorData>()?;
        let eps_buf = Self::make_eps_buffer(eps)?;

        // Reduced shape after removing last axis (IR Reduce removes the axis).
        let reduced_shape: Vec<usize> = shape[..rank - 1].to_vec();
        let reduced_shape = if reduced_shape.is_empty() {
            vec![1]
        } else {
            reduced_shape
        };

        let def = crate::kernel_def_cache::get_or_build(
            "rms_norm",
            &[shape, weight.dims()],
            &[eps.to_bits()],
            x.dtype(),
            || {
                let mut bld = TensorBlockBuilder::new("dyn_rms_norm");
                let input = bld.add_input("data", shape);
                let w_node = bld.add_input("weight", weight.dims());
                let eps_node = bld.add_input("eps", &[1]);
                let eps_bc = bld.add_broadcast(eps_node, &reduced_shape);

                let sq = bld.add_elementwise(make_sqr_kernel(), &[input], shape);
                let mean_sq = bld.add_reduce(sq, ReduceOp::Mean, rank - 1, false, &reduced_shape);
                let rms_sq = bld.add_elementwise(
                    make_binop_kernel("add", BinOpKind::Add),
                    &[mean_sq, eps_bc],
                    &reduced_shape,
                );
                let rsqrt = bld.add_elementwise(
                    make_unary_kernel("rsqrt", UnaryFnKind::Rsqrt),
                    &[rms_sq],
                    &reduced_shape,
                );
                let rsqrt_bc = bld.add_broadcast_left(rsqrt, shape);
                let normed = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[input, rsqrt_bc],
                    shape,
                );
                let w_bc = bld.add_broadcast(w_node, shape);
                let out = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[normed, w_bc],
                    shape,
                );

                crate::build_kernel(bld, out)
            },
        )?;

        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        Self::dispatch_def_with_contract(
            &def,
            &[
                ("data", x_data.as_gpu_slice()),
                ("weight", w_data.as_gpu_slice()),
                ("eps", GpuSlice::zero_offset(eps_buf)),
            ],
            shape,
            x.dtype(),
            contract,
        )
    }
}
