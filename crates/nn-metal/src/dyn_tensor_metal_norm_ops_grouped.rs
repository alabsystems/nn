// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]

//! Fused GPU GroupNorm, InstanceNorm, and AdaIN+Snake kernels.
//!
//! Extracted from `dyn_tensor_metal_norm_ops.rs` for 500-line compliance.
//! These norm variants operate on grouped/per-channel dimensions (reshape to
//! `[batch*groups, features]` before normalizing), unlike LayerNorm/RmsNorm
//! which normalize over the last dimension directly.
//!
//! AdaIN+Snake (#2227) fuses InstanceNorm → affine(gamma, beta) → Snake(alpha)
//! into a single dispatch graph for the 36 Kokoro ResBlock invocations.

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError};

use crate::gpu_slice::GpuSlice;
use nn_dsl::ir::{BinOpKind, ScalarType, UnaryFnKind};
use nn_dsl::tensor_ir::ReduceOp;
use nn_dsl::{PrecisionContract, PrecisionTier, TensorBlockBuilder};

use super::super::kernels::{make_binop_kernel, make_sqr_kernel, make_unary_kernel};
use super::super::MetalTensorData;

impl super::super::MetalDynBackend {
    /// Fused GroupNorm GPU kernel.
    ///
    /// Input `[batch, channels, *spatial]`. Reshapes to
    /// `[batch * num_groups, channels_per_group * spatial]`, normalizes over
    /// the last dim, reshapes back, applies per-channel affine.
    pub(in super::super) fn gpu_group_norm(
        x: &DynTensor,
        num_groups: usize,
        weight: &DynTensor,
        bias: &DynTensor,
        eps: f64,
    ) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_group_norm")?;
        Self::validate_f32(weight, "gpu_group_norm(weight)")?;
        Self::validate_f32(bias, "gpu_group_norm(bias)")?;

        let dims = x.dims();
        if dims.len() < 2 {
            return Err(TensorError::InvalidShape(
                "gpu_group_norm requires rank >= 2".into(),
            ));
        }
        let batch = dims[0];
        let channels = dims[1];
        if !channels.is_multiple_of(num_groups) {
            return Err(TensorError::ValueOutOfRange {
                description: "gpu_group_norm: channels not divisible by num_groups",
            });
        }
        let channels_per_group = channels / num_groups;
        let spatial = crate::metal_backend::checked_dim_product(&dims[2..])?;

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let w_data = weight.gpu_data::<MetalTensorData>()?;
        let b_data = bias.gpu_data::<MetalTensorData>()?;
        let eps_buf = Self::make_eps_buffer(eps)?;

        // Reshape to [batch * num_groups, channels_per_group * spatial]
        let flat_rows = batch * num_groups;
        let flat_cols = channels_per_group * spatial;
        let flat_shape = [flat_rows, flat_cols];
        // Reduce over axis 1 removes it: [flat_rows, flat_cols] → [flat_rows]
        let reduced_shape = [flat_rows];

        let def = crate::kernel_def_cache::get_or_build(
            "group_norm",
            &[dims, weight.dims(), bias.dims()],
            &[num_groups as u64, eps.to_bits()],
            x.dtype(),
            || {
                let mut bld = TensorBlockBuilder::new("dyn_group_norm");
                let input = bld.add_input("data", &flat_shape);
                let w_node = bld.add_input("weight", weight.dims());
                let b_node = bld.add_input("bias", bias.dims());
                let eps_node = bld.add_input("eps", &[1]);
                let eps_bc = bld.add_broadcast(eps_node, &reduced_shape);

                let mean = bld.add_reduce(input, ReduceOp::Mean, 1, false, &reduced_shape);
                let mean_bc = bld.add_broadcast_left(mean, &flat_shape);
                let centered = bld.add_elementwise(
                    make_binop_kernel("sub", BinOpKind::Sub),
                    &[input, mean_bc],
                    &flat_shape,
                );
                let sq = bld.add_elementwise(make_sqr_kernel(), &[centered], &flat_shape);
                let var = bld.add_reduce(sq, ReduceOp::Mean, 1, false, &reduced_shape);
                let var_eps = bld.add_elementwise(
                    make_binop_kernel("add", BinOpKind::Add),
                    &[var, eps_bc],
                    &reduced_shape,
                );
                let rsqrt_val = bld.add_elementwise(
                    make_unary_kernel("rsqrt", UnaryFnKind::Rsqrt),
                    &[var_eps],
                    &reduced_shape,
                );
                let rsqrt_bc = bld.add_broadcast_left(rsqrt_val, &flat_shape);
                let normed = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[centered, rsqrt_bc],
                    &flat_shape,
                );

                let normed_full = bld.add_reshape(normed, dims);

                let mut wb_shape = vec![1usize; dims.len()];
                wb_shape[1] = channels;
                let w_reshaped = bld.add_reshape(w_node, &wb_shape);
                let w_bc = bld.add_broadcast(w_reshaped, dims);
                let scaled = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[normed_full, w_bc],
                    dims,
                );
                let b_reshaped = bld.add_reshape(b_node, &wb_shape);
                let b_bc = bld.add_broadcast(b_reshaped, dims);
                let out = bld.add_elementwise(
                    make_binop_kernel("add", BinOpKind::Add),
                    &[scaled, b_bc],
                    dims,
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
                ("eps", GpuSlice::from_ref(&eps_buf, 0)),
            ],
            dims,
            x.dtype(),
            contract,
        )
    }

    /// Fused InstanceNorm GPU kernel: `(x - mean) / sqrt(var + eps)`
    ///
    /// Input `[B, C, *spatial]`. Reshapes to `[B*C, spatial_flat]`, normalizes
    /// over the last dim, reshapes back. No learnable affine parameters.
    ///
    /// This is structurally identical to GroupNorm with `num_groups = channels`
    /// and no weight/bias affine, but without the affine overhead.
    pub(in super::super) fn gpu_instance_norm(x: &DynTensor, eps: f64) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_instance_norm")?;

        let dims = x.dims();
        if dims.len() < 3 {
            return Err(TensorError::InvalidShape(
                "gpu_instance_norm requires rank >= 3".into(),
            ));
        }
        let batch = dims[0];
        let channels = dims[1];
        let spatial = crate::metal_backend::checked_dim_product(&dims[2..])?;

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let eps_buf = Self::make_eps_buffer(eps)?;

        // Reshape to [B*C, spatial_flat] — each (batch, channel) pair is one row.
        let flat_rows =
            batch
                .checked_mul(channels)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;
        let flat_shape = [flat_rows, spatial];
        // Reduce over axis 1: [flat_rows, spatial] → [flat_rows]
        let reduced_shape = [flat_rows];

        let def = crate::kernel_def_cache::get_or_build(
            "instance_norm",
            &[dims],
            &[eps.to_bits()],
            x.dtype(),
            || {
                let mut bld = TensorBlockBuilder::new("dyn_instance_norm");
                let input = bld.add_input("data", &flat_shape);
                let eps_node = bld.add_input("eps", &[1]);
                let eps_bc = bld.add_broadcast(eps_node, &reduced_shape);

                let mean = bld.add_reduce(input, ReduceOp::Mean, 1, false, &reduced_shape);
                let mean_bc = bld.add_broadcast_left(mean, &flat_shape);
                let centered = bld.add_elementwise(
                    make_binop_kernel("sub", BinOpKind::Sub),
                    &[input, mean_bc],
                    &flat_shape,
                );
                let sq = bld.add_elementwise(make_sqr_kernel(), &[centered], &flat_shape);
                let var = bld.add_reduce(sq, ReduceOp::Mean, 1, false, &reduced_shape);
                let var_eps = bld.add_elementwise(
                    make_binop_kernel("add", BinOpKind::Add),
                    &[var, eps_bc],
                    &reduced_shape,
                );
                let rsqrt_val = bld.add_elementwise(
                    make_unary_kernel("rsqrt", UnaryFnKind::Rsqrt),
                    &[var_eps],
                    &reduced_shape,
                );
                let rsqrt_bc = bld.add_broadcast_left(rsqrt_val, &flat_shape);
                let normed = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[centered, rsqrt_bc],
                    &flat_shape,
                );

                // Reshape back to original dims.
                let out = bld.add_reshape(normed, dims);

                crate::build_kernel(bld, out)
            },
        )?;

        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        Self::dispatch_def_with_contract(
            &def,
            &[
                ("data", x_data.as_gpu_slice()),
                ("eps", GpuSlice::from_ref(&eps_buf, 0)),
            ],
            dims,
            x.dtype(),
            contract,
        )
    }

    /// Fused per-channel Snake GPU kernel (#2226).
    ///
    /// `x + (1/alpha) * sin²(alpha * x)` in a single dispatch graph.
    /// Replaces 6 separate GPU dispatches per snake_tensor call.
    /// Input shape is arbitrary; alpha broadcasts left-aligned over x.
    pub(in super::super) fn gpu_snake_tensor(
        x: &DynTensor,
        alpha: &DynTensor,
    ) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_snake_tensor")?;
        Self::validate_f32(alpha, "gpu_snake_tensor(alpha)")?;

        let dims = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let alpha_data = alpha.gpu_data::<MetalTensorData>()?;

        let snake_kernel = nn_dsl::adain::build_snake_scalar_kernel()
            .map_err(|e| TensorError::InvalidShape(format!("snake scalar kernel build: {e}")))?;

        let def = crate::kernel_def_cache::get_or_build(
            "snake_tensor",
            &[dims, alpha.dims()],
            &[],
            x.dtype(),
            || {
                let mut bld = TensorBlockBuilder::new("dyn_snake_tensor");
                let input = bld.add_input("data", dims);
                let alpha_node = bld.add_input("alpha", alpha.dims());
                let alpha_bc = bld.add_broadcast_left(alpha_node, dims);
                let out = bld.add_elementwise(snake_kernel, &[input, alpha_bc], dims);
                crate::build_kernel(bld, out)
            },
        )?;

        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        Self::dispatch_def_with_contract(
            &def,
            &[
                ("data", x_data.as_gpu_slice()),
                ("alpha", alpha_data.as_gpu_slice()),
            ],
            dims,
            x.dtype(),
            contract,
        )
    }

    /// Fused AdaIN+Snake GPU kernel (#2227).
    ///
    /// InstanceNorm(x) → affine((1+gamma)*normed + beta) → Snake(alpha).
    /// Single dispatch graph replaces ~12 separate GPU dispatches in the
    /// decomposed path. Input `[B, C, T]`, gamma/beta `[B, C, 1]`,
    /// alpha `[1, C, 1]` (must match input rank; bare `[C]` would broadcast
    /// incorrectly via right-aligned `add_broadcast`).
    pub(in super::super) fn gpu_adain_snake(
        x: &DynTensor,
        gamma: &DynTensor,
        beta: &DynTensor,
        alpha: &DynTensor,
        eps: f64,
        residual_gamma: bool,
    ) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_adain_snake")?;
        Self::validate_f32(gamma, "gpu_adain_snake(gamma)")?;
        Self::validate_f32(beta, "gpu_adain_snake(beta)")?;
        Self::validate_f32(alpha, "gpu_adain_snake(alpha)")?;

        let dims = x.dims();
        if dims.len() < 3 {
            return Err(TensorError::InvalidShape(
                "gpu_adain_snake requires rank >= 3".into(),
            ));
        }
        // Alpha must match input rank for right-aligned broadcast correctness.
        // Bare [C] would broadcast against T (wrong); require [1, C, 1].
        if alpha.rank() != dims.len() {
            return Err(TensorError::InvalidShape(format!(
                "gpu_adain_snake: alpha rank {} must match input rank {} (use [1, C, 1] not [C])",
                alpha.rank(),
                dims.len(),
            )));
        }
        let batch = dims[0];
        let channels = dims[1];
        let spatial = crate::metal_backend::checked_dim_product(&dims[2..])?;

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let gamma_data = gamma.gpu_data::<MetalTensorData>()?;
        let beta_data = beta.gpu_data::<MetalTensorData>()?;
        let alpha_data = alpha.gpu_data::<MetalTensorData>()?;
        let eps_buf = Self::make_eps_buffer(eps)?;

        // Reshape to [B*C, spatial_flat] for InstanceNorm reduction.
        let flat_rows =
            batch
                .checked_mul(channels)
                .ok_or_else(|| TensorError::DimensionOverflow {
                    dims: dims.to_vec(),
                })?;
        let flat_shape = [flat_rows, spatial];
        let reduced_shape = [flat_rows];

        let snake_kernel = nn_dsl::adain::build_snake_scalar_kernel()
            .map_err(|e| TensorError::InvalidShape(format!("snake scalar kernel build: {e}")))?;

        // Cache key includes residual_gamma to avoid mixing affine variants.
        let cache_name = if residual_gamma {
            "adain_snake_rg"
        } else {
            "adain_snake"
        };
        let def = crate::kernel_def_cache::get_or_build(
            cache_name,
            &[dims, gamma.dims(), beta.dims(), alpha.dims()],
            &[eps.to_bits()],
            x.dtype(),
            || {
                let mut bld = TensorBlockBuilder::new("dyn_adain_snake");
                let input = bld.add_input("data", &flat_shape);
                let gamma_node = bld.add_input("gamma", gamma.dims());
                let beta_node = bld.add_input("beta", beta.dims());
                let alpha_node = bld.add_input("alpha", alpha.dims());
                let eps_node = bld.add_input("eps", &[1]);
                let eps_bc = bld.add_broadcast(eps_node, &reduced_shape);

                // --- InstanceNorm: (x - mean) * rsqrt(var + eps) ---
                let mean = bld.add_reduce(input, ReduceOp::Mean, 1, false, &reduced_shape);
                let mean_bc = bld.add_broadcast_left(mean, &flat_shape);
                let centered = bld.add_elementwise(
                    make_binop_kernel("sub", BinOpKind::Sub),
                    &[input, mean_bc],
                    &flat_shape,
                );
                let sq = bld.add_elementwise(make_sqr_kernel(), &[centered], &flat_shape);
                let var = bld.add_reduce(sq, ReduceOp::Mean, 1, false, &reduced_shape);
                let var_eps = bld.add_elementwise(
                    make_binop_kernel("add", BinOpKind::Add),
                    &[var, eps_bc],
                    &reduced_shape,
                );
                let rsqrt_val = bld.add_elementwise(
                    make_unary_kernel("rsqrt", UnaryFnKind::Rsqrt),
                    &[var_eps],
                    &reduced_shape,
                );
                let rsqrt_bc = bld.add_broadcast_left(rsqrt_val, &flat_shape);
                let normed = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[centered, rsqrt_bc],
                    &flat_shape,
                );

                // Reshape normed back to [B, C, T] for affine + snake.
                let normed_full = bld.add_reshape(normed, dims);

                // --- Affine ---
                let gamma_bc = bld.add_broadcast(gamma_node, dims);
                let scaled_by_gamma = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[normed_full, gamma_bc],
                    dims,
                );
                let beta_bc = bld.add_broadcast(beta_node, dims);
                let adain_out = if residual_gamma {
                    // Kokoro convention: (1 + gamma) * normed + beta
                    // = normed + gamma * normed + beta
                    let normed_plus_scaled = bld.add_elementwise(
                        make_binop_kernel("add", BinOpKind::Add),
                        &[normed_full, scaled_by_gamma],
                        dims,
                    );
                    bld.add_elementwise(
                        make_binop_kernel("add", BinOpKind::Add),
                        &[normed_plus_scaled, beta_bc],
                        dims,
                    )
                } else {
                    // Standard AdaIN: gamma * normed + beta (#3251)
                    bld.add_elementwise(
                        make_binop_kernel("add", BinOpKind::Add),
                        &[scaled_by_gamma, beta_bc],
                        dims,
                    )
                };

                // --- Snake: y + (1/alpha) * sin²(alpha * y) ---
                let alpha_bc = bld.add_broadcast(alpha_node, dims);
                let out = bld.add_elementwise(snake_kernel, &[adain_out, alpha_bc], dims);

                crate::build_kernel(bld, out)
            },
        )?;

        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        Self::dispatch_def_with_contract(
            &def,
            &[
                ("data", x_data.as_gpu_slice()),
                ("gamma", gamma_data.as_gpu_slice()),
                ("beta", beta_data.as_gpu_slice()),
                ("alpha", alpha_data.as_gpu_slice()),
                ("eps", GpuSlice::from_ref(&eps_buf, 0)),
            ],
            dims,
            x.dtype(),
            contract,
        )
    }
}
