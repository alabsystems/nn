// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kahan compensated GPU reduce dispatch (#1814).
//!
//! Extracted from `dyn_tensor_metal_ops_reduce.rs` for 500-line compliance.
//! Provides `gpu_reduce_compensated` (any axis) and `gpu_reduce_compensated_last`
//! (last axis) with `PrecisionTier::Strict` for near-f64 precision from f32
//! arithmetic.

use nn_core::dyn_tensor::{DynTensor, ReduceOp};
use nn_core::{check_dim, Result, TensorError};

use nn_dsl::tensor_ir::ReduceOp as DslReduceOp;
use nn_dsl::{PrecisionContract, PrecisionTier, TensorBlockBuilder};

impl super::MetalDynBackend {
    /// Kahan compensated reduce for Sum/Mean (#1814).
    ///
    /// Dispatches with `PrecisionTier::Strict` to emit Kahan-compensated MSL
    /// kernels providing ~2x mantissa bits of precision from f32 arithmetic.
    /// Non-last axes use the transpose→reduce(last)→transpose pattern.
    pub(super) fn gpu_reduce_compensated(
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_reduce_compensated")?;
        let shape = x.dims();
        check_dim(dim, shape.len())?;

        // Guard: zero-length axis — compensated reduce has same identity semantics
        // as standard reduce. Sum→0.0, Mean→error. (Max/Min are not supported
        // in compensated path and are rejected below.)
        if shape[dim] == 0 {
            if matches!(op, ReduceOp::Mean) {
                return Err(TensorError::ZeroLengthDimension {
                    axis: dim,
                    operation: "mean_compensated",
                });
            }
            // Only Sum reaches here (Max/Min rejected below).
            return Self::zero_length_reduce_identity(ReduceOp::Sum, x, dim, keepdim);
        }

        let dsl_op = match op {
            ReduceOp::Sum => DslReduceOp::Sum,
            ReduceOp::Mean => DslReduceOp::Mean,
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_reduce_compensated: only Sum/Mean supported, got {op:?}"
                )))
            }
        };

        // Non-last axis: transpose → compensated reduce(last) → transpose back.
        let rank = shape.len();
        if rank > 1 && dim + 1 != rank {
            let last = rank - 1;
            let transposed = Self::gpu_transpose(x, dim, last)?;
            let reduced =
                Self::gpu_reduce_compensated_last(dsl_op, op, &transposed, last, keepdim)?;
            if keepdim {
                return Self::gpu_transpose(&reduced, dim, last);
            }
            if reduced.rank() <= 1 {
                return Ok(reduced);
            }
            let new_rank = reduced.rank();
            if dim >= new_rank {
                return Ok(reduced);
            }
            let mut perm: Vec<usize> = (0..new_rank).collect();
            perm.remove(dim);
            perm.push(dim);
            return Self::gpu_permute(&reduced, &perm);
        }

        Self::gpu_reduce_compensated_last(dsl_op, op, x, dim, keepdim)
    }

    /// Kahan compensated last-axis reduce dispatch.
    fn gpu_reduce_compensated_last(
        dsl_op: DslReduceOp,
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        let shape = x.dims();
        let x_data = x.gpu_data::<super::MetalTensorData>()?;

        let mut reduce_shape: Vec<usize> = shape.to_vec();
        reduce_shape.remove(dim);

        let op_tag: u64 = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Mean => 1,
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_reduce_compensated_last: unsupported op {op:?}"
                )));
            }
        };
        let keepdim_tag = u64::from(keepdim);

        let contract =
            PrecisionContract::bootstrap(PrecisionTier::Strict, nn_dsl::ir::ScalarType::F32);

        // Scalar output edge case: GPU cannot produce rank-0 tensors.
        if reduce_shape.is_empty() {
            let out_shape = vec![1usize];
            let def = crate::kernel_def_cache::get_or_build(
                "reduce_compensated_scalar",
                &[shape],
                &[op_tag, dim as u64, keepdim_tag],
                x.dtype(),
                || {
                    let mut b = TensorBlockBuilder::new("dyn_reduce_compensated");
                    let input = b.add_input("data", shape);
                    let reduced = b.add_reduce(input, dsl_op, dim, false, &out_shape);
                    crate::build_kernel(b, reduced)
                },
            )?;
            let result = Self::dispatch_def_with_contract(
                &def,
                &[("data", x_data.as_gpu_slice())],
                &out_shape,
                x.dtype(),
                contract,
            )?;
            if keepdim {
                return Ok(result);
            }
            return result.reshape([]);
        }

        let def = crate::kernel_def_cache::get_or_build(
            "reduce_compensated",
            &[shape],
            &[op_tag, dim as u64, keepdim_tag],
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_reduce_compensated");
                let input = b.add_input("data", shape);
                let reduced = b.add_reduce(input, dsl_op, dim, false, &reduce_shape);

                if keepdim {
                    let mut keepdim_shape: Vec<usize> = shape.to_vec();
                    keepdim_shape[dim] = 1;
                    let out = b.add_reshape(reduced, &keepdim_shape);
                    crate::build_kernel(b, out)
                } else {
                    crate::build_kernel(b, reduced)
                }
            },
        )?;

        if keepdim {
            let mut keepdim_shape: Vec<usize> = shape.to_vec();
            keepdim_shape[dim] = 1;
            Self::dispatch_def_with_contract(
                &def,
                &[("data", x_data.as_gpu_slice())],
                &keepdim_shape,
                x.dtype(),
                contract,
            )
        } else {
            Self::dispatch_def_with_contract(
                &def,
                &[("data", x_data.as_gpu_slice())],
                &reduce_shape,
                x.dtype(),
                contract,
            )
        }
    }
}
