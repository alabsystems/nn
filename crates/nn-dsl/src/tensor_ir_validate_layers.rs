// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Layer-specific tensor IR validators: InstanceNorm1d, Conv1d,
//! ConvTranspose1d, RmsNorm, AdaIN1d, Softmax.
//!
//! Extracted from `tensor_ir_validate.rs` per #619.
//! Linear, MatMul, Embedding, Attention, LayerNorm live in
//! `tensor_ir_validate_layers_ext.rs`.

use super::super::{TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNodeId};

impl TensorKernelDef {
    pub(super) fn validate_instance_norm(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        eps: TensorNodeId,
        axis: usize,
        gamma: Option<TensorNodeId>,
        beta: Option<TensorNodeId>,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, eps)?;
        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 2 {
            return Err(TensorIRLayerError::InstanceNormRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }
        let eps_shape = &self.nodes[eps.index()].shape;
        if eps_shape != &[1] {
            return Err(TensorIRLayerError::InstanceNormEpsNotScalar {
                shape: eps_shape.clone(),
            }
            .into());
        }
        if axis >= input_shape.len() {
            return Err(TensorIRError::ReduceAxisOutOfBounds {
                axis,
                shape: input_shape.clone(),
            });
        }
        // InstanceNorm1d normalizes over the last axis only.
        // Reject non-last-axis early at IR validation (defense-in-depth;
        // graph_tensor.rs also checks this at NY translation time).
        if axis + 1 != input_shape.len() {
            return Err(TensorIRLayerError::InstanceNormAxisNotLast {
                axis,
                rank: input_shape.len(),
            }
            .into());
        }
        // Validate optional affine parameters: gamma and beta must both
        // be present or both absent, and must have shape [C] where C is
        // the channel dimension (second-to-last axis).
        match (gamma, beta) {
            (Some(g), Some(b)) => {
                self.check_ref(current, g)?;
                self.check_ref(current, b)?;
                let num_channels = input_shape[input_shape.len() - 2];
                let gamma_shape = &self.nodes[g.index()].shape;
                if gamma_shape != &[num_channels] {
                    return Err(TensorIRLayerError::InstanceNormAffineShapeMismatch {
                        param: "gamma",
                        expected_channels: num_channels,
                        got_shape: gamma_shape.clone(),
                    }
                    .into());
                }
                let beta_shape = &self.nodes[b.index()].shape;
                if beta_shape != &[num_channels] {
                    return Err(TensorIRLayerError::InstanceNormAffineShapeMismatch {
                        param: "beta",
                        expected_channels: num_channels,
                        got_shape: beta_shape.clone(),
                    }
                    .into());
                }
            }
            (None, None) => {} // Non-affine mode, OK.
            _ => {
                return Err(TensorIRLayerError::InstanceNormAffineMismatch.into());
            }
        }
        Ok(())
    }

    // Conv validators (validate_conv1d, validate_conv2d, validate_conv_transpose_1d)
    // extracted to tensor_ir_validate_layers_conv.rs (#827 Direction 4).

    pub(super) fn validate_rms_norm(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        eps: TensorNodeId,
        axis: usize,
        weight: TensorNodeId,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, eps)?;
        self.check_ref(current, weight)?;
        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 2 {
            return Err(TensorIRLayerError::RmsNormRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }
        let eps_shape = &self.nodes[eps.index()].shape;
        if eps_shape != &[1] {
            return Err(TensorIRLayerError::RmsNormEpsNotScalar {
                shape: eps_shape.clone(),
            }
            .into());
        }
        if axis + 1 != input_shape.len() {
            return Err(TensorIRLayerError::RmsNormAxisNotLast {
                axis,
                rank: input_shape.len(),
            }
            .into());
        }
        let hidden = input_shape[input_shape.len() - 1];
        let weight_shape = &self.nodes[weight.index()].shape;
        if weight_shape != &[hidden] {
            return Err(TensorIRLayerError::RmsNormWeightShape {
                expected_hidden: hidden,
                got_shape: weight_shape.clone(),
            }
            .into());
        }
        Ok(())
    }

    pub(super) fn validate_adain1d(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        eps: TensorNodeId,
        axis: usize,
        style_gamma: TensorNodeId,
        style_beta: TensorNodeId,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, eps)?;
        self.check_ref(current, style_gamma)?;
        self.check_ref(current, style_beta)?;
        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 2 {
            return Err(TensorIRLayerError::AdaIN1dRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }
        let eps_shape = &self.nodes[eps.index()].shape;
        if eps_shape != &[1] {
            return Err(TensorIRLayerError::AdaIN1dEpsNotScalar {
                shape: eps_shape.clone(),
            }
            .into());
        }
        if axis + 1 != input_shape.len() {
            return Err(TensorIRLayerError::AdaIN1dAxisNotLast {
                axis,
                rank: input_shape.len(),
            }
            .into());
        }
        let num_channels = input_shape[input_shape.len() - 2];
        let sg_shape = &self.nodes[style_gamma.index()].shape;
        if sg_shape != &[num_channels] {
            return Err(TensorIRLayerError::AdaIN1dStyleShapeMismatch {
                param: "style_gamma",
                expected_channels: num_channels,
                got_shape: sg_shape.clone(),
            }
            .into());
        }
        let sb_shape = &self.nodes[style_beta.index()].shape;
        if sb_shape != &[num_channels] {
            return Err(TensorIRLayerError::AdaIN1dStyleShapeMismatch {
                param: "style_beta",
                expected_channels: num_channels,
                got_shape: sb_shape.clone(),
            }
            .into());
        }
        Ok(())
    }

    pub(super) fn validate_batch_norm(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        running_mean: TensorNodeId,
        running_var: TensorNodeId,
        weight: TensorNodeId,
        bias: TensorNodeId,
        eps: TensorNodeId,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, running_mean)?;
        self.check_ref(current, running_var)?;
        self.check_ref(current, weight)?;
        self.check_ref(current, bias)?;
        self.check_ref(current, eps)?;

        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 2 {
            return Err(TensorIRLayerError::BatchNormRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }

        let eps_shape = &self.nodes[eps.index()].shape;
        if eps_shape != &[1] {
            return Err(TensorIRLayerError::BatchNormEpsNotScalar {
                shape: eps_shape.clone(),
            }
            .into());
        }

        // Infer the channel axis from the BN param length rather than
        // hardcoding it. The four affine/stat params (running_mean/_var,
        // weight, bias) are per-channel rank-1 tensors `[C]`, so the
        // running_mean length is the authoritative channel count. We then
        // check that *some* plausible channel axis of the input equals that
        // length. This mirrors NY's runtime `detect_input_layout`, which for
        // rank>=3 picks axis 0 when `shape[0] == expected_channels` (channels-
        // first, no batch dim) else axis 1 (NCHW). Hardcoding dim 1 for rank>=3
        // wrongly rejected valid channels-first rank-3 inputs (e.g. [C,S,S]
        // with [C] params), where it read the spatial dim S as the channel
        // count.
        //
        // Soundness: this is strictly a false-positive removal. We still
        // require running_mean to be rank-1 (else no authoritative C exists),
        // still require all four params to be exactly `[C]`, and still reject
        // any BN whose param length matches no candidate channel axis. The
        // candidate axes are exactly those NY would consider, so we never
        // accept a layout NY cannot then resolve. For rank==2 either axis 0
        // ([C, L]) or axis 1 ([L, C]) may carry the channels; for rank>=3 the
        // channel is axis 0 (channels-first) or axis 1 (batched NCHW).
        let mean_shape = &self.nodes[running_mean.index()].shape;
        if mean_shape.len() != 1 {
            return Err(TensorIRLayerError::BatchNormParamShape {
                param: "running_mean",
                // No authoritative channel count yet; report the input's
                // last dim as a best-effort expected size for diagnostics.
                expected_channels: input_shape[input_shape.len() - 1],
                got_shape: mean_shape.clone(),
            }
            .into());
        }
        let num_channels = mean_shape[0];

        // Accept if num_channels matches axis 0 or axis 1 of the input. (Both
        // are valid channel positions under NY's heuristic; for rank>=3 axis 0
        // is channels-first and axis 1 is batched NCHW. The square-spatial
        // ambiguity where shape[0]==shape[1] is inherent to NY's own heuristic
        // and resolves identically there.)
        let axis_matches = input_shape[0] == num_channels || input_shape[1] == num_channels;
        if !axis_matches {
            return Err(TensorIRLayerError::BatchNormParamShape {
                param: "running_mean",
                // Report axis 0's size as the expected channel count: with no
                // matching axis, channels-first is the convention we try first.
                expected_channels: input_shape[0],
                got_shape: mean_shape.clone(),
            }
            .into());
        }

        // All four params must be exactly rank-1 `[num_channels]`.
        for &(param_id, param_name) in &[
            (running_mean, "running_mean"),
            (running_var, "running_var"),
            (weight, "weight"),
            (bias, "bias"),
        ] {
            let shape = &self.nodes[param_id.index()].shape;
            if shape != &[num_channels] {
                return Err(TensorIRLayerError::BatchNormParamShape {
                    param: param_name,
                    expected_channels: num_channels,
                    got_shape: shape.clone(),
                }
                .into());
            }
        }

        Ok(())
    }

    pub(super) fn validate_softmax(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        axis: i32,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.is_empty() {
            return Err(TensorIRLayerError::SoftmaxInputScalar.into());
        }
        let rank = input_shape.len();
        // Validate axis with Python-style negative indexing.
        // Valid range: [-rank, rank)
        let rank_i32 =
            i32::try_from(rank).map_err(|_| TensorIRLayerError::SoftmaxAxisOutOfBounds {
                axis,
                rank,
                neg_rank: i32::MIN,
            })?;
        if axis < -rank_i32 || axis >= rank_i32 {
            return Err(TensorIRLayerError::SoftmaxAxisOutOfBounds {
                axis,
                rank,
                neg_rank: -rank_i32,
            }
            .into());
        }
        Ok(())
    }
}
