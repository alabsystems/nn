// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Composite tracked tensor operations: conv, softmax, layer_norm, embedding,
//! cross-entropy loss, dropout.
//!
//! Normalization ops (rms_norm, group_norm, batch_norm, instance_norm) are in
//! `tracked_composite_ops_norm.rs`.
//! Pool ops (max_pool2d, avg_pool2d, adaptive_avg_pool2d) are in
//! `tracked_pool_ops.rs`.
//!
//! Extracted from `tracked.rs` for 500-line compliance.

#[path = "tracked_pool_ops.rs"]
mod pool_ops;

use super::TrackedTensor;
use crate::error::Result;
use crate::op::Op;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{Conv1dParams, Conv2dParams, ConvTranspose1dParams};
use std::sync::Arc;

#[path = "tracked_composite_ops_norm.rs"]
mod norm_ops;

impl TrackedTensor {
    /// 1-D convolution.
    pub fn conv1d(
        self: &Arc<Self>,
        kernel: &Arc<Self>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Arc<Self>> {
        let data = self
            .data
            .conv1d(kernel.tensor(), padding, stride, dilation, groups)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Conv1d {
                input: Arc::clone(self),
                kernel: Arc::clone(kernel),
                padding,
                stride,
                dilation,
                groups,
            },
        )))
    }

    /// 2-D convolution.
    pub fn conv2d(
        self: &Arc<Self>,
        kernel: &Arc<Self>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Arc<Self>> {
        let data = self
            .data
            .conv2d(kernel.tensor(), padding, stride, dilation, groups)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Conv2d {
                input: Arc::clone(self),
                kernel: Arc::clone(kernel),
                padding,
                stride,
                dilation,
                groups,
            },
        )))
    }

    /// 1-D convolution with named parameter struct.
    ///
    /// Prevents parameter-order mistakes. See [`Conv1dParams`].
    pub fn conv1d_with(
        self: &Arc<Self>,
        kernel: &Arc<Self>,
        params: Conv1dParams,
    ) -> Result<Arc<Self>> {
        self.conv1d(
            kernel,
            params.padding,
            params.stride,
            params.dilation,
            params.groups,
        )
    }

    /// 2-D convolution with named parameter struct.
    ///
    /// Prevents parameter-order mistakes. See [`Conv2dParams`].
    pub fn conv2d_with(
        self: &Arc<Self>,
        kernel: &Arc<Self>,
        params: Conv2dParams,
    ) -> Result<Arc<Self>> {
        self.conv2d(
            kernel,
            params.padding,
            params.stride,
            params.dilation,
            params.groups,
        )
    }

    /// Concatenate multiple tensors along a dimension.
    pub fn cat(inputs: &[&Arc<Self>], dim: usize) -> Result<Arc<Self>> {
        let tensor_refs: Vec<&DynTensor> = inputs.iter().map(|t| t.tensor()).collect();
        let data = DynTensor::cat(&tensor_refs, dim)?;
        let arcs: Vec<Arc<Self>> = inputs.iter().map(|t| Arc::clone(t)).collect();
        Ok(Arc::new(Self::from_op(data, Op::Cat(arcs, dim))))
    }

    /// Softmax over a dimension.
    pub fn softmax(self: &Arc<Self>, dim: usize) -> Result<Arc<Self>> {
        let data = self.data.softmax(dim)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Softmax(Arc::clone(self), dim),
        )))
    }

    /// Layer normalization.
    pub fn layer_norm(
        self: &Arc<Self>,
        weight: &Arc<Self>,
        bias: &Arc<Self>,
        eps: f64,
    ) -> Result<Arc<Self>> {
        let normalized_shape = weight.tensor().dims()[0];
        let mean = self.data.mean_keepdim(self.data.rank() - 1)?;
        let diff = self.data.sub(&mean)?;
        let var = diff.sqr()?.mean_keepdim(diff.rank() - 1)?;
        let inv_std = var.add_scalar(eps)?.sqrt()?.recip()?;
        let normed = diff.mul(&inv_std)?;
        let data = normed.mul(weight.tensor())?.add(bias.tensor())?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::LayerNorm {
                input: Arc::clone(self),
                weight: Arc::clone(weight),
                bias: Arc::clone(bias),
                eps,
                normalized_shape,
            },
        )))
    }

    /// Embedding lookup.
    ///
    /// Supports multi-dimensional indices: flattens to 1D for index_select,
    /// then reshapes output to `[*input_dims, embed_dim]` matching PyTorch
    /// `nn.Embedding` behavior.
    pub fn embedding(weight: &Arc<Self>, indices: &Arc<Self>) -> Result<Arc<Self>> {
        let w_dims = weight.tensor().dims();
        if w_dims.len() < 2 {
            return Err(crate::AutodiffError::InvalidConfig {
                op: "embedding",
                reason: format!(
                    "weight must be 2D [vocab, embed_dim], got {}D",
                    w_dims.len()
                ),
            });
        }
        let embed_dim = w_dims[w_dims.len() - 1];
        if embed_dim == 0 {
            return Err(crate::AutodiffError::InvalidConfig {
                op: "embedding",
                reason: "embed_dim must be > 0".into(),
            });
        }
        let idx = indices.tensor();
        let input_dims = idx.dims().to_vec();
        // Flatten multi-dimensional indices to 1D for index_select
        let flat_ids = idx.reshape([idx.elem_count()])?;
        let flat_result = weight.tensor().index_select(&flat_ids, 0)?;
        // Reshape to [*input_dims, embed_dim]
        let mut out_shape = input_dims;
        out_shape.push(embed_dim);
        let data = flat_result.reshape(&out_shape)?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::Embedding(Arc::clone(weight), Arc::clone(indices)),
        )))
    }

    /// Dropout with explicit training mode control.
    ///
    /// When `training` is false, returns the input unchanged (identity).
    /// When `training` is true, applies inverted dropout with probability `p`.
    ///
    /// Matches PyTorch's `F.dropout(x, p, training)` semantics.
    pub fn dropout_train(self: &Arc<Self>, p: f64, training: bool) -> Result<Arc<Self>> {
        if !training {
            return Ok(Arc::clone(self));
        }
        self.dropout(p)
    }

    /// Dropout: randomly zero elements with probability `p` during training.
    ///
    /// Uses inverted dropout: surviving elements are scaled by `1/(1-p)` so the
    /// expected output magnitude is unchanged. The binary mask is stored in the
    /// Op for the backward pass.
    ///
    /// # Errors
    ///
    /// Returns `AutodiffError::Dropout` if `p` is not in `[0, 1)`.
    pub fn dropout(self: &Arc<Self>, p: f64) -> Result<Arc<Self>> {
        use crate::error::AutodiffError;

        if !(0.0..1.0).contains(&p) {
            return Err(AutodiffError::Dropout { p });
        }
        if p == 0.0 {
            // No dropout — identity pass-through with no op recorded
            return Ok(Arc::clone(self));
        }
        let scale = 1.0 / (1.0 - p);
        // Generate mask: 1.0 where rand >= p (keep), 0.0 where rand < p (drop)
        let rand_vals = DynTensor::rand(0.0, 1.0, self.data.dims(), &self.data.device())?;
        let keep = rand_vals.ge(p)?;
        let ones = DynTensor::ones(self.data.dims(), self.data.dtype(), &self.data.device())?;
        let zeros = DynTensor::zeros(self.data.dims(), self.data.dtype(), &self.data.device())?;
        let mask = keep.where_cond(&ones, &zeros)?;
        // Apply: output = input * mask * scale
        let data = self.data.mul(&mask)?.mul_scalar(scale)?;
        let mask_tracked = Arc::new(Self::from_tensor(mask));
        Ok(Arc::new(Self::from_op(
            data,
            Op::Dropout(Arc::clone(self), mask_tracked, scale),
        )))
    }

    /// 1-D transposed convolution.
    pub fn conv_transpose1d(
        self: &Arc<Self>,
        kernel: &Arc<Self>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        output_padding: usize,
    ) -> Result<Arc<Self>> {
        let data = self.data.conv_transpose1d(
            kernel.tensor(),
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        )?;
        Ok(Arc::new(Self::from_op(
            data,
            Op::ConvTranspose1d {
                input: Arc::clone(self),
                kernel: Arc::clone(kernel),
                padding,
                stride,
                dilation,
                groups,
                output_padding,
            },
        )))
    }

    /// 1-D transposed convolution with named parameter struct.
    ///
    /// Prevents the parameter-order bug that caused P1 #1484
    /// (`stride ↔ output_padding` swap). See [`ConvTranspose1dParams`].
    pub fn conv_transpose1d_with(
        self: &Arc<Self>,
        kernel: &Arc<Self>,
        params: ConvTranspose1dParams,
    ) -> Result<Arc<Self>> {
        self.conv_transpose1d(
            kernel,
            params.padding,
            params.stride,
            params.dilation,
            params.groups,
            params.output_padding,
        )
    }

    /// Cross-entropy loss with fused log-softmax.
    ///
    /// Computes `-log_softmax(logits, dim)[targets].mean()` as a single
    /// operation with a numerically stable fused backward rule.
    /// `targets` should contain class indices (u32 tensor).
    pub fn cross_entropy_loss(
        self: &Arc<Self>,
        targets: &Arc<Self>,
        dim: usize,
    ) -> Result<Arc<Self>> {
        let log_sm = self.data.log_softmax(dim)?;
        let target_log_probs = log_sm.gather(targets.tensor(), dim)?;
        let loss_per_sample = target_log_probs.neg()?;
        // Mean over all dimensions to get scalar loss
        let mut data = loss_per_sample;
        for d in (0..data.rank()).rev() {
            data = data.mean_keepdim(d)?;
        }
        for _ in 0..data.rank() {
            data = data.squeeze(0)?;
        }
        Ok(Arc::new(Self::from_op(
            data,
            Op::CrossEntropyLoss(Arc::clone(self), Arc::clone(targets), dim),
        )))
    }

    /// Mean squared error loss: `mean((input - target)^2)`.
    ///
    /// Produces a scalar output. Both input and target must have the same shape.
    pub fn mse_loss(self: &Arc<Self>, target: &Arc<Self>) -> Result<Arc<Self>> {
        let diff = self.data.sub(target.tensor())?;
        let sq = diff.sqr()?;
        let mut data = sq;
        for d in (0..data.rank()).rev() {
            data = data.mean_keepdim(d)?;
        }
        for _ in 0..data.rank() {
            data = data.squeeze(0)?;
        }
        Ok(Arc::new(Self::from_op(
            data,
            Op::MseLoss(Arc::clone(self), Arc::clone(target)),
        )))
    }

    /// L1 loss: `mean(|input - target|)`.
    ///
    /// Produces a scalar output. Both input and target must have the same shape.
    pub fn l1_loss(self: &Arc<Self>, target: &Arc<Self>) -> Result<Arc<Self>> {
        let diff = self.data.sub(target.tensor())?;
        let abs_diff = diff.abs()?;
        let mut data = abs_diff;
        for d in (0..data.rank()).rev() {
            data = data.mean_keepdim(d)?;
        }
        for _ in 0..data.rank() {
            data = data.squeeze(0)?;
        }
        Ok(Arc::new(Self::from_op(
            data,
            Op::L1Loss(Arc::clone(self), Arc::clone(target)),
        )))
    }

    /// Huber (smooth L1) loss with transition point `delta`.
    ///
    /// Quadratic for `|x| < delta`, linear for `|x| >= delta`:
    /// ```text
    /// loss(x) = 0.5 * x^2 / delta       if |x| < delta
    ///         = |x| - 0.5 * delta         if |x| >= delta
    /// ```
    ///
    /// Produces a scalar output. Both input and target must have the same shape.
    pub fn huber_loss(self: &Arc<Self>, target: &Arc<Self>, delta: f64) -> Result<Arc<Self>> {
        let diff = self.data.sub(target.tensor())?;
        let abs_diff = diff.abs()?;
        // Quadratic region: 0.5 * diff^2 / delta
        let quadratic = diff.sqr()?.mul_scalar(0.5 / delta)?;
        // Linear region: |diff| - 0.5 * delta
        let linear = abs_diff.add_scalar(-0.5 * delta)?;
        // Select: where |diff| < delta use quadratic, else linear
        let mask = abs_diff.lt(delta)?;
        let result = mask.where_cond(&quadratic, &linear)?;
        let mut data = result;
        for d in (0..data.rank()).rev() {
            data = data.mean_keepdim(d)?;
        }
        for _ in 0..data.rank() {
            data = data.squeeze(0)?;
        }
        Ok(Arc::new(Self::from_op(
            data,
            Op::HuberLoss(Arc::clone(self), Arc::clone(target), delta),
        )))
    }

    // Pool ops (max_pool2d, avg_pool2d, adaptive_avg_pool2d) are in tracked_pool_ops.rs.
}

#[cfg(test)]
#[path = "loss_tests.rs"]
mod loss_tests;
