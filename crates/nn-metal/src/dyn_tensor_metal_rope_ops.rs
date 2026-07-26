// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused GPU RoPE (Rotary Position Embedding) kernel for [`MetalDynBackend`].
//!
//! Replaces the 11-dispatch decomposed path (4 narrow + 4 broadcast_mul +
//! 1 broadcast_sub + 1 broadcast_add + 1 cat) with a single dispatch graph.
//!
//! For Qwen3-8B (36 layers), this reduces RoPE GPU dispatches from 792 to 72
//! per forward pass.
//!
//! Part of #1363 (fused GPU RoPE kernel).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError};

use crate::metal_backend::checked_dim_product;

use nn_dsl::ir::BinOpKind;
use nn_dsl::TensorBlockBuilder;

use super::kernels::make_binop_kernel;
use super::MetalTensorData;

impl super::MetalDynBackend {
    /// Fused RoPE GPU kernel.
    ///
    /// Applies rotary position embedding in a single dispatch graph:
    /// ```text
    /// y[..., 2i]   = x[..., 2i] * cos[..., i] - x[..., 2i+1] * sin[..., i]
    /// y[..., 2i+1] = x[..., 2i] * sin[..., i] + x[..., 2i+1] * cos[..., i]
    /// ```
    ///
    /// Input shapes:
    /// - `x`: `[..., S, D]` where D = head_dim (must be even)
    /// - `cos`: `[S, D/2]` (precomputed cosines for positions)
    /// - `sin`: `[S, D/2]` (precomputed sines for positions)
    ///
    /// Output: same shape as `x`.
    pub(super) fn gpu_rope(x: &DynTensor, cos: &DynTensor, sin: &DynTensor) -> Result<DynTensor> {
        Self::validate_same_float_dtype(x, cos, "gpu_rope")?;
        Self::validate_same_float_dtype(x, sin, "gpu_rope")?;

        let x_dims = x.dims();
        let rank = x_dims.len();
        if rank < 2 {
            return Err(TensorError::InvalidShape(
                "gpu_rope requires rank >= 2".into(),
            ));
        }
        let head_dim = x_dims[rank - 1];
        let seq_len = x_dims[rank - 2];
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(TensorError::ValueOutOfRange {
                description: "gpu_rope: head_dim must be a positive even number",
            });
        }
        let half_dim = head_dim / 2;

        // Validate cos/sin shapes match [seq_len, half_dim].
        let cos_dims = cos.dims();
        let sin_dims = sin.dims();
        if cos_dims != [seq_len, half_dim] {
            return Err(TensorError::shape_mismatch(
                vec![seq_len, half_dim],
                cos_dims.to_vec(),
            ));
        }
        if sin_dims != [seq_len, half_dim] {
            return Err(TensorError::shape_mismatch(
                vec![seq_len, half_dim],
                sin_dims.to_vec(),
            ));
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;
        let cos_data = cos.gpu_data::<MetalTensorData>()?;
        let sin_data = sin.gpu_data::<MetalTensorData>()?;

        // Flatten batch dimensions for the IR graph.
        // x: [..., S, D] → flat [B, S, D] where B = product(batch_dims)
        let batch: usize = checked_dim_product(&x_dims[..rank - 2])?;
        let flat_shape = [batch, seq_len, head_dim];
        // cos/sin: [S, D/2]
        let cs_shape = [seq_len, half_dim];

        let def = crate::kernel_def_cache::get_or_build(
            "rope",
            &[x_dims, cos_dims, sin_dims],
            &[batch as u64],
            x.dtype(),
            || {
                // pairs: [B, S, D/2, 2]
                let pairs_shape = [batch, seq_len, half_dim, 2];
                // half: [B, S, D/2]
                let half_shape = [batch, seq_len, half_dim];
                // Narrow output from pairs dim 3: [B, S, D/2, 1]
                let narrow_shape = [batch, seq_len, half_dim, 1];

                let mut bld = TensorBlockBuilder::new("dyn_rope");

                // Inputs (IR sees flattened shapes; GPU buffers have the same flat data).
                let data = bld.add_input("data", &flat_shape);
                let cos_node = bld.add_input("cos", &cs_shape);
                let sin_node = bld.add_input("sin", &cs_shape);

                // Reshape to pairs: [B, S, D] → [B, S, D/2, 2]
                let pairs = bld.add_reshape(data, &pairs_shape);

                // Split even/odd: narrow along axis=3, then reshape to remove the trailing 1.
                let even_raw = bld.add_narrow(pairs, 3, 0, 1, &narrow_shape);
                let odd_raw = bld.add_narrow(pairs, 3, 1, 1, &narrow_shape);
                let even = bld.add_reshape(even_raw, &half_shape);
                let odd = bld.add_reshape(odd_raw, &half_shape);

                // Broadcast cos/sin from [S, D/2] → [B, S, D/2]
                // Use right-aligned (NumPy-style) broadcast: [S, D/2] aligns to trailing
                // dims of [B, S, D/2], prepending batch dim.
                let cos_bc = bld.add_broadcast(cos_node, &half_shape);
                let sin_bc = bld.add_broadcast(sin_node, &half_shape);

                // y_even = even * cos - odd * sin
                let even_cos = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[even, cos_bc],
                    &half_shape,
                );
                let odd_sin = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[odd, sin_bc],
                    &half_shape,
                );
                let y_even = bld.add_elementwise(
                    make_binop_kernel("sub", BinOpKind::Sub),
                    &[even_cos, odd_sin],
                    &half_shape,
                );

                // y_odd = even * sin + odd * cos
                let even_sin = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[even, sin_bc],
                    &half_shape,
                );
                let odd_cos = bld.add_elementwise(
                    make_binop_kernel("mul", BinOpKind::Mul),
                    &[odd, cos_bc],
                    &half_shape,
                );
                let y_odd = bld.add_elementwise(
                    make_binop_kernel("add", BinOpKind::Add),
                    &[even_sin, odd_cos],
                    &half_shape,
                );

                // Reshape back to [B, S, D/2, 1] for interleaving.
                let y_even_exp = bld.add_reshape(y_even, &narrow_shape);
                let y_odd_exp = bld.add_reshape(y_odd, &narrow_shape);

                // Interleave: concat along axis=3 → [B, S, D/2, 2]
                let interleaved = bld.add_concat(&[y_even_exp, y_odd_exp], 3, &pairs_shape);

                // Reshape back to flat: [B, S, D]
                let out = bld.add_reshape(interleaved, &flat_shape);

                crate::build_kernel(bld, out)
            },
        )?;

        Self::dispatch_def(
            &def,
            &[
                ("data", x_data.as_gpu_slice()),
                ("cos", cos_data.as_gpu_slice()),
                ("sin", sin_data.as_gpu_slice()),
            ],
            // Output shape is the original (non-flattened) shape.
            // dispatch_def uses this for creating the output DynTensor.
            // The GPU buffer layout is flat regardless.
            x_dims,
            x.dtype(),
        )
    }
}
