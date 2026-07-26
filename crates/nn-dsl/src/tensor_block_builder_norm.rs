// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Normalization builder methods for `TensorBlockBuilder`.
//!
//! Extracted from `tensor_block_builder.rs` to stay under the 500-line limit.
//! Contains `add_instance_norm` and `add_group_norm_g1` (decomposed GroupNorm).

use crate::ir::{BinOpKind, UnaryFnKind};
use crate::tensor_builders::{binop_kernel, square_kernel, unary_kernel};
use crate::tensor_ir::{ReduceOp, TensorNode, TensorNodeId, TensorOpKind};

use super::TensorBlockBuilder;

impl TensorBlockBuilder {
    /// Add an InstanceNorm1d op. Returns output node ID.
    ///
    /// Pass `gamma` and `beta` for affine InstanceNorm (learnable scale/shift).
    /// Pass `None, None` for non-affine normalization.
    ///
    /// # Panics
    ///
    /// Debug-asserts that the spatial dimension (at `axis`) is > 1.
    /// InstanceNorm on a single element is degenerate: mean=value, var=0,
    /// so the output is always the bias term regardless of input. Bounds
    /// computed at spatial dim=1 cannot be extrapolated to production
    /// dimensions. See #2637.
    pub fn add_instance_norm(
        &mut self,
        input: TensorNodeId,
        eps: TensorNodeId,
        axis: usize,
        gamma: Option<TensorNodeId>,
        beta: Option<TensorNodeId>,
        out_shape: &[usize],
    ) -> TensorNodeId {
        debug_assert!(
            axis < out_shape.len() && out_shape[axis] > 1,
            "InstanceNorm spatial dimension at axis {axis} is {} — degenerate (need > 1). \
             Bounds at spatial dim=1 are meaningless for normalization. See #2637.",
            out_shape.get(axis).copied().unwrap_or(0),
        );
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::InstanceNorm1d {
                input,
                eps,
                axis,
                gamma,
                beta,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a GroupNorm(groups=1) decomposition using only dispatchable primitives.
    ///
    /// Decomposes into: Reshape [C, T] → [1, C*T] → decomposed InstanceNorm
    /// (Reduce/Broadcast/Elementwise) → Reshape [C, T].
    /// With optional affine: broadcast(gamma) * x + broadcast(beta).
    ///
    /// Unlike the InstanceNorm1d native op, this produces only Reduce, Broadcast,
    /// and Elementwise nodes that `build_dispatch_plan` can handle directly.
    ///
    /// Returns the final output node ID with shape [C, T].
    ///
    /// # Panics
    ///
    /// Debug-asserts that `channels * time_len > 1`. GroupNorm(g=1)
    /// normalizes over the full C*T dimension — a single element makes
    /// mean=value, var=0, so the output is always the bias term. Bounds
    /// computed at C*T=1 are degenerate and cannot be extrapolated to
    /// production dimensions. See #2637.
    pub fn add_group_norm_g1(
        &mut self,
        input: TensorNodeId,
        eps: TensorNodeId,
        gamma: Option<TensorNodeId>,
        beta: Option<TensorNodeId>,
        channels: usize,
        time_len: usize,
    ) -> TensorNodeId {
        debug_assert!(
            channels * time_len > 1,
            "GroupNorm(g=1) normalization dimension C*T is {} — degenerate (need > 1). \
             Bounds at C*T=1 are meaningless for normalization. See #2637.",
            channels * time_len,
        );
        let flat_shape = vec![1, channels * time_len];
        let reduced_shape = vec![1];
        let out_shape = [channels, time_len];

        // Step 1: Reshape [C, T] → [1, C*T]
        let reshaped = self.add_reshape(input, &flat_shape);

        // Step 2: Decomposed InstanceNorm on [1, C*T] (normalizes over axis 1)
        // mean(x) over axis 1
        let mean = self.add_reduce(reshaped, ReduceOp::Mean, 1, false, &reduced_shape);
        // broadcast mean → [1, C*T]
        let mean_bc = self.add_broadcast_left(mean, &flat_shape);
        // x - mean
        let centered = self.add_elementwise(
            binop_kernel("sub", BinOpKind::Sub),
            &[reshaped, mean_bc],
            &flat_shape,
        );
        // (x - mean)^2
        let sq = self.add_elementwise(square_kernel(), &[centered], &flat_shape);
        // var = mean((x - mean)^2)
        let var = self.add_reduce(sq, ReduceOp::Mean, 1, false, &reduced_shape);
        // broadcast var → [1, C*T]
        let var_bc = self.add_broadcast_left(var, &flat_shape);
        // broadcast eps → [1, C*T]
        let eps_bc = self.add_broadcast_left(eps, &flat_shape);
        // var + eps
        let var_eps = self.add_elementwise(
            binop_kernel("add", BinOpKind::Add),
            &[var_bc, eps_bc],
            &flat_shape,
        );
        // rsqrt(var + eps)
        let rsqrt = self.add_elementwise(
            unary_kernel("rsqrt", UnaryFnKind::Rsqrt),
            &[var_eps],
            &flat_shape,
        );
        // (x - mean) * rsqrt(var + eps)
        let normed = self.add_elementwise(
            binop_kernel("mul", BinOpKind::Mul),
            &[centered, rsqrt],
            &flat_shape,
        );

        // Step 3: Reshape [1, C*T] → [C, T]
        let mut output = self.add_reshape(normed, &out_shape);

        // Step 4: Optional per-channel affine (gamma * x + beta)
        // gamma/beta are [C], broadcast left-aligned to [C, T].
        if let Some(gamma_id) = gamma {
            let gamma_bc = self.add_broadcast_left(gamma_id, &out_shape);
            output = self.add_binary_mul(output, gamma_bc, &out_shape);
        }
        if let Some(beta_id) = beta {
            let beta_bc = self.add_broadcast_left(beta_id, &out_shape);
            output = self.add_binary_add(output, beta_bc, &out_shape);
        }

        output
    }

    /// Add a BatchNorm op using frozen running statistics. Returns output node ID.
    ///
    /// BatchNorm (inference): `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
    /// NY pre-computes `scale = gamma / sqrt(var + eps)` and
    /// `bias = beta - mean * scale` internally.
    pub fn add_batch_norm(
        &mut self,
        input: TensorNodeId,
        running_mean: TensorNodeId,
        running_var: TensorNodeId,
        weight: TensorNodeId,
        bias: TensorNodeId,
        eps: TensorNodeId,
        out_shape: &[usize],
    ) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::BatchNorm {
                input,
                running_mean,
                running_var,
                weight,
                bias,
                eps,
            },
            out_shape.to_vec(),
        ));
        id
    }

    /// Add a reshape op. Returns output node ID.
    ///
    /// Validation enforces `product(input.shape) == product(target_shape)`.
    pub fn add_reshape(&mut self, input: TensorNodeId, target_shape: &[usize]) -> TensorNodeId {
        let id = self.alloc_id();
        self.nodes.push(TensorNode::new(
            id,
            TensorOpKind::Reshape {
                input,
                target_shape: target_shape.to_vec(),
            },
            target_shape.to_vec(),
        ));
        id
    }
}
