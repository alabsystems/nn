// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU reduce op dispatch for [`MetalDynBackend`].
//!
//! Extracted from `dyn_tensor_metal_ops.rs` (#1276) for 500-line compliance.
//! Contains `gpu_reduce` (last-axis) and `gpu_reduce_via_transpose` (any axis).

use nn_core::dyn_tensor::{DynTensor, ReduceOp};
use nn_core::{check_dim, Result, TensorError};

use nn_dsl::tensor_ir::ReduceOp as DslReduceOp;
use nn_dsl::TensorBlockBuilder;

impl super::MetalDynBackend {
    /// GPU-native reduce op dispatch for Sum and Mean (keepdim).
    pub(super) fn gpu_reduce(
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_reduce")?;
        let shape = x.dims();
        check_dim(dim, shape.len())?;

        // Guard: zero-length axis. Mean is undefined (0/0). For Sum/Max/Min,
        // the GPU kernel would read 0 elements and return uninitialized memory.
        // Return mathematically correct identity values matching CPU behavior:
        //   Sum → 0.0, Max → NEG_INFINITY, Min → INFINITY, Mean → error.
        if shape[dim] == 0 {
            if matches!(op, ReduceOp::Mean) {
                return Err(TensorError::ZeroLengthDimension {
                    axis: dim,
                    operation: "mean",
                });
            }
            return Self::zero_length_reduce_identity(op, x, dim, keepdim);
        }

        let x_data = x.gpu_data::<super::MetalTensorData>()?;

        let mut reduce_shape: Vec<usize> = shape.to_vec();
        reduce_shape.remove(dim);

        let dsl_op = match op {
            ReduceOp::Sum => DslReduceOp::Sum,
            ReduceOp::Mean => DslReduceOp::Mean,
            ReduceOp::Max => DslReduceOp::Max,
            ReduceOp::Min => DslReduceOp::Min,
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_reduce: unsupported op {op:?}"
                )))
            }
        };

        // Encode reduce op as u64 for cache key discrimination.
        // Must mirror the dsl_op match above — no catch-all to prevent
        // two different ops silently sharing cache key 4 if a new variant
        // is added to the first match but not this one.
        let op_tag = match op {
            ReduceOp::Sum => 0u64,
            ReduceOp::Mean => 1,
            ReduceOp::Max => 2,
            ReduceOp::Min => 3,
            // unreachable: the dsl_op match above returns Err for unknown ops.
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_reduce: unsupported op tag {op:?}"
                )))
            }
        };
        let keepdim_tag = u64::from(keepdim);

        // GPU dispatch cannot produce rank-0 tensors (empty shape). When
        // reducing the last dimension of a rank-1 tensor (→ scalar), produce
        // [1] via kernel, then reshape to [] if keepdim=false.
        if reduce_shape.is_empty() {
            let out_shape = vec![1usize];
            let def = crate::kernel_def_cache::get_or_build(
                "reduce_scalar",
                &[shape],
                &[op_tag, dim as u64, keepdim_tag],
                x.dtype(),
                || {
                    let mut b = TensorBlockBuilder::new("dyn_reduce");
                    let input = b.add_input("data", shape);
                    let reduced = b.add_reduce(input, dsl_op, dim, false, &out_shape);
                    crate::build_kernel(b, reduced)
                },
            )?;
            let result = Self::dispatch_def(
                &def,
                &[("data", x_data.as_gpu_slice())],
                &out_shape,
                x.dtype(),
            )?;
            if keepdim {
                return Ok(result); // [1] is correct for keepdim=true
            }
            return result.reshape([]); // [1] → scalar [] for keepdim=false
        }

        let def = crate::kernel_def_cache::get_or_build(
            "reduce",
            &[shape],
            &[op_tag, dim as u64, keepdim_tag],
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_reduce");
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
            Self::dispatch_def(
                &def,
                &[("data", x_data.as_gpu_slice())],
                &keepdim_shape,
                x.dtype(),
            )
        } else {
            Self::dispatch_def(
                &def,
                &[("data", x_data.as_gpu_slice())],
                &reduce_shape,
                x.dtype(),
            )
        }
    }

    /// GPU-native non-last-axis reduce via transpose→reduce(last)→transpose.
    ///
    /// The MSL reduce kernel only supports last-axis reductions. For non-last
    /// axes, we transpose `dim` to the last position, run `gpu_reduce` on the
    /// last axis, then transpose the result back to restore the original axis
    /// ordering.
    pub(super) fn gpu_reduce_via_transpose(
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        let rank = x.rank();
        if rank == 0 || dim + 1 == rank {
            return Err(TensorError::InvalidShape(format!(
                "gpu_reduce_via_transpose requires non-last axis: rank={rank}, dim={dim}"
            )));
        }

        let last = rank - 1;

        // Step 1: Transpose dim ↔ last to put the reduce axis last.
        let transposed = Self::gpu_transpose(x, dim, last)?;

        // Step 2: Reduce the (now-last) axis.
        let reduced = Self::gpu_reduce(op, &transposed, last, keepdim)?;

        // Step 3: Transpose back to restore original ordering.
        if keepdim {
            // reduced shape: original shape with dim and last swapped, last=1
            // e.g. [A,B,C] reduce dim=0 keepdim → transpose(0,2) → [C,B,A] →
            //   reduce(2,keepdim) → [C,B,1] → transpose(0,2) → [1,B,C] ✓
            Self::gpu_transpose(&reduced, dim, last)
        } else {
            // reduced shape: transposed shape with last axis removed
            // e.g. [A,B,C] reduce dim=0 → transpose(0,2) → [C,B,A] →
            //   reduce(2) → [C,B]
            // We need [B,C]. The dim=0 was removed, so dimensions that were at
            // indices 1..last are now at 0..last-1, and original-last is at
            // position dim (since dim < last and we removed last).
            //
            // Build a permutation to restore original order:
            // Original axes minus `dim`: e.g. for [A,B,C] minus dim=0 → [B,C]
            // After transpose(0,2)+reduce: axes are [last, 1..last-1] = [C,B]
            // We need permutation that maps [C,B] → [B,C]
            if reduced.rank() <= 1 {
                // Scalar or 1D result — no reordering needed.
                return Ok(reduced);
            }
            // The transposed-then-reduced tensor has axes arranged as:
            //   [orig_last, orig_1, ..., orig_{dim-1}, orig_{dim+1}, ..., orig_{last-1}]
            // We want:
            //   [orig_0, ..., orig_{dim-1}, orig_{dim+1}, ..., orig_last]
            // but orig_dim is removed, so the target is:
            //   [orig_0, ..., orig_{dim-1}, orig_{dim+1}, ..., orig_last]
            //
            // In the reduced tensor, original axis `last` is at position 0 (if dim < last),
            // original axes [0..dim) map to positions [1..dim+1) (shifted +1 because
            // orig_last took position 0), and original axes [dim+1..last) map to
            // positions [dim+1..last).
            //
            // Actually: after transpose(dim,last), the axes are
            //   [0, ..., dim-1, LAST, dim+1, ..., last-1, DIM]
            // After removing the last axis (DIM), we have rank-1 axes:
            //   [0, ..., dim-1, LAST, dim+1, ..., last-1]
            // We want the natural order with dim removed:
            //   [0, ..., dim-1, dim+1, ..., last-1, LAST]
            // So we need to move axis `dim` (where LAST sits) to the end.
            let new_rank = reduced.rank();
            if dim >= new_rank {
                // dim was the position of LAST in the reduced tensor;
                // if dim >= new_rank, LAST is already at the end.
                return Ok(reduced);
            }
            let mut perm: Vec<usize> = (0..new_rank).collect();
            // Move position `dim` to the end.
            perm.remove(dim);
            perm.push(dim);
            Self::gpu_permute(&reduced, &perm)
        }
    }

    /// Return the mathematically correct identity value for a reduce op over a
    /// zero-length axis. CPU ndarray returns these naturally; the GPU kernel
    /// would read 0 elements and leave uninitialized memory without this guard.
    ///
    /// Identity values:
    /// - Sum → 0.0 (additive identity)
    /// - Max → f32::NEG_INFINITY (maximum identity)
    /// - Min → f32::INFINITY (minimum identity)
    /// - Mean → unreachable (caller must reject before calling this)
    pub(super) fn zero_length_reduce_identity(
        op: ReduceOp,
        x: &DynTensor,
        dim: usize,
        keepdim: bool,
    ) -> Result<DynTensor> {
        let fill_value = match op {
            ReduceOp::Sum => 0.0f32,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Mean => {
                // Mean over zero-length axis is undefined (0/0).
                // Caller must check this before calling.
                return Err(TensorError::ZeroLengthDimension {
                    axis: dim,
                    operation: "mean",
                });
            }
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "zero_length_reduce_identity: unsupported op {op:?}"
                )));
            }
        };

        let shape = x.dims();
        let out_shape: Vec<usize> = if keepdim {
            let mut s = shape.to_vec();
            s[dim] = 1;
            s
        } else {
            shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != dim)
                .map(|(_, &d)| d)
                .collect()
        };

        DynTensor::full(&out_shape, f64::from(fill_value), x.dtype(), &x.device())
    }
}
