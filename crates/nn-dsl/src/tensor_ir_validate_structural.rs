// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Structural tensor IR validators: reshape, axis-select, stack, reduce,
//! elementwise, broadcast.
//!
//! Extracted from `tensor_ir_validate.rs` per #619.

use super::super::{
    validate_broadcast_alignment, BroadcastAlignment, TensorIRError, TensorIRLayerError,
    TensorKernelDef, TensorNodeId,
};
use super::validate_shape;
use crate::ir::KernelDef;

impl TensorKernelDef {
    pub(super) fn validate_reshape(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        target_shape: &[usize],
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        validate_shape(target_shape)?;
        let input_shape = &self.nodes[input.index()].shape;
        let input_product: usize = input_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                TensorIRError::Layer(TensorIRLayerError::ShapeProductOverflow {
                    shape: input_shape.clone(),
                })
            })?;
        let target_product: usize = target_shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| {
                TensorIRError::Layer(TensorIRLayerError::ShapeProductOverflow {
                    shape: target_shape.to_vec(),
                })
            })?;
        if input_product != target_product {
            return Err(TensorIRError::ReshapeProductMismatch {
                input_product,
                target_product,
            });
        }
        Ok(())
    }

    pub(super) fn validate_axis_select(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        axis: usize,
        index: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        let input_shape = &self.nodes[input.index()].shape;
        if axis >= input_shape.len() {
            return Err(TensorIRError::AxisSelectOutOfBounds {
                axis,
                shape: input_shape.clone(),
            });
        }
        if axis == 0 {
            return Err(TensorIRError::AxisZeroReserved { op: "axis_select" });
        }
        if index >= input_shape[axis] {
            return Err(TensorIRError::AxisSelectIndexOutOfBounds {
                index,
                dim: input_shape[axis],
                axis,
            });
        }
        Ok(())
    }

    pub(super) fn validate_stack(
        &self,
        current: TensorNodeId,
        inputs: &[TensorNodeId],
        axis: usize,
    ) -> Result<(), TensorIRError> {
        if inputs.is_empty() {
            return Err(TensorIRError::EmptyStack);
        }
        for input_id in inputs {
            self.check_ref(current, *input_id)?;
        }
        let first_shape = &self.nodes[inputs[0].index()].shape;
        for input_id in &inputs[1..] {
            let input_shape = &self.nodes[input_id.index()].shape;
            if input_shape != first_shape {
                return Err(TensorIRError::StackShapeMismatch {
                    expected: first_shape.clone(),
                    found: input_shape.clone(),
                });
            }
        }
        // Note: axis 0 is allowed for Stack. NY's UnsqueezeLayer
        // and ConcatLayer both accept axis 0. The axis_offset mechanism in
        // graph_tensor_structural.rs handles multi-variable stacking correctly
        // (user axis 0 + offset 1 = NY axis 1). This enables
        // [2, batch, H] output from LSTM dual builder for zero-copy dim-0
        // narrow regardless of batch size.
        if axis > first_shape.len() {
            return Err(TensorIRError::StackAxisOutOfBounds {
                axis,
                rank: first_shape.len(),
            });
        }
        Ok(())
    }

    pub(super) fn validate_concat(
        &self,
        current: TensorNodeId,
        inputs: &[TensorNodeId],
        axis: usize,
    ) -> Result<(), TensorIRError> {
        if inputs.len() < 2 {
            return Err(TensorIRLayerError::EmptyConcat.into());
        }
        for input_id in inputs {
            self.check_ref(current, *input_id)?;
        }
        let first_shape = &self.nodes[inputs[0].index()].shape;
        let rank = first_shape.len();
        // Note: axis 0 is allowed for Concat, mirroring validate_stack above.
        // In user-space TensorIR axis 0 is a legitimate data axis (e.g. channel
        // for [C,H,W] backbone/SPPF/C2f kernels, sequence/token for [T,D]
        // attention/KV-cache kernels). The framework's batch/variable-stacking
        // axis is NOT user-space: it is injected *below* user axes at
        // translation time via ctx.axis_offset (graph_tensor.rs: axis_offset =
        // num_variables>1 ? 1 : 0; graph_tensor_structural.rs translate_concat
        // emits ConcatLayer::new(axis + ctx.axis_offset)). So user axis 0 either
        // maps to NY axis 0 (single-variable kernels) or is shifted to NY axis 1
        // (multi-variable), never aliasing the packing axis. NY's ConcatLayer
        // and its nary IBP path support concat at any axis, and concat is a pure
        // layout op (lower-with-lower, upper-with-upper) so axis-0 IBP/CROWN
        // bounds are exact. The real soundness guards (axis < rank, matching
        // rank, matching non-concat dims) below are preserved.
        if axis >= rank {
            return Err(TensorIRLayerError::ConcatAxisOutOfBounds { axis, rank }.into());
        }
        for input_id in &inputs[1..] {
            let input_shape = &self.nodes[input_id.index()].shape;
            if input_shape.len() != rank {
                return Err(TensorIRLayerError::ConcatRankMismatch {
                    expected: rank,
                    found: input_shape.len(),
                }
                .into());
            }
            for (dim, (a, b)) in first_shape.iter().zip(input_shape.iter()).enumerate() {
                if dim != axis && a != b {
                    return Err(TensorIRLayerError::ConcatShapeMismatch {
                        axis: dim,
                        expected: *a,
                        found: *b,
                    }
                    .into());
                }
            }
        }
        Ok(())
    }

    pub(super) fn validate_reduce(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        axis: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        let input_shape = &self.nodes[input.index()].shape;
        if axis >= input_shape.len() {
            return Err(TensorIRError::ReduceAxisOutOfBounds {
                axis,
                shape: input_shape.clone(),
            });
        }
        Ok(())
    }

    pub(super) fn validate_elementwise(
        &self,
        current: TensorNodeId,
        kernel: &KernelDef,
        inputs: &[TensorNodeId],
    ) -> Result<(), TensorIRError> {
        if kernel.params.len() != inputs.len() {
            return Err(TensorIRError::ElementwiseParamMismatch {
                expected: kernel.params.len(),
                got: inputs.len(),
            });
        }
        for input_id in inputs {
            self.check_ref(current, *input_id)?;
        }
        // Enforce shape equality across all elementwise inputs (MVP:
        // exact match; broadcast-compatible inputs should use explicit
        // Broadcast nodes before feeding into Elementwise).
        if inputs.len() > 1 {
            let first_shape = &self.nodes[inputs[0].index()].shape;
            for (i, input_id) in inputs[1..].iter().enumerate() {
                let input_shape = &self.nodes[input_id.index()].shape;
                if input_shape != first_shape {
                    return Err(TensorIRError::ElementwiseShapeMismatch {
                        expected: first_shape.clone(),
                        found: input_shape.clone(),
                        index: i + 1,
                    });
                }
            }
        }
        kernel.validate()?;
        Ok(())
    }

    pub(super) fn validate_binary_add(
        &self,
        current: TensorNodeId,
        left: TensorNodeId,
        right: TensorNodeId,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, left)?;
        self.check_ref(current, right)?;
        let left_shape = &self.nodes[left.index()].shape;
        let right_shape = &self.nodes[right.index()].shape;
        if left_shape != right_shape {
            return Err(TensorIRLayerError::BinaryAddShapeMismatch {
                left: left_shape.clone(),
                right: right_shape.clone(),
            }
            .into());
        }
        Ok(())
    }

    pub(super) fn validate_binary_mul(
        &self,
        current: TensorNodeId,
        left: TensorNodeId,
        right: TensorNodeId,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, left)?;
        self.check_ref(current, right)?;
        let left_shape = &self.nodes[left.index()].shape;
        let right_shape = &self.nodes[right.index()].shape;
        if left_shape != right_shape {
            return Err(TensorIRLayerError::BinaryMulShapeMismatch {
                left: left_shape.clone(),
                right: right_shape.clone(),
            }
            .into());
        }
        Ok(())
    }

    pub(super) fn validate_narrow(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        axis: usize,
        start: usize,
        length: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        let input_shape = &self.nodes[input.index()].shape;
        if axis >= input_shape.len() {
            return Err(TensorIRLayerError::NarrowAxisOutOfBounds {
                axis,
                shape: input_shape.clone(),
            }
            .into());
        }
        if length == 0 {
            return Err(TensorIRLayerError::NarrowZeroLength { axis }.into());
        }
        let dim = input_shape[axis];
        let end = start.checked_add(length).ok_or(TensorIRError::Layer(
            TensorIRLayerError::NarrowOutOfBounds {
                start,
                length,
                dim,
                axis,
            },
        ))?;
        if end > dim {
            return Err(TensorIRLayerError::NarrowOutOfBounds {
                start,
                length,
                dim,
                axis,
            }
            .into());
        }
        Ok(())
    }

    pub(super) fn validate_transpose(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        axes: &[usize],
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        let input_shape = &self.nodes[input.index()].shape;
        let rank = input_shape.len();
        if axes.len() != rank {
            return Err(TensorIRLayerError::TransposeAxesLengthMismatch {
                axes_len: axes.len(),
                rank,
            }
            .into());
        }
        // Check that axes is a valid permutation of [0..rank).
        let mut seen = vec![false; rank];
        for &a in axes {
            if a >= rank {
                return Err(TensorIRLayerError::TransposeAxisOutOfBounds { axis: a, rank }.into());
            }
            if seen[a] {
                return Err(TensorIRLayerError::TransposeDuplicateAxis { axis: a }.into());
            }
            seen[a] = true;
        }
        Ok(())
    }

    pub(super) fn validate_broadcast(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        target_shape: &[usize],
        alignment: BroadcastAlignment,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        validate_shape(target_shape)?;
        let input_shape = &self.nodes[input.index()].shape;
        validate_broadcast_alignment(input_shape, target_shape, alignment)?;
        Ok(())
    }

    pub(super) fn validate_zero_pad_1d(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        pad_left: usize,
        pad_right: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.is_empty() {
            return Err(TensorIRLayerError::ZeroPad1dScalarInput.into());
        }
        // Output length must not overflow
        let last_dim = input_shape[input_shape.len() - 1];
        last_dim
            .checked_add(pad_left)
            .and_then(|v| v.checked_add(pad_right))
            .ok_or(TensorIRError::Layer(
                TensorIRLayerError::ZeroPad1dOverflow {
                    in_length: last_dim,
                    pad_left,
                    pad_right,
                },
            ))?;
        Ok(())
    }
}
