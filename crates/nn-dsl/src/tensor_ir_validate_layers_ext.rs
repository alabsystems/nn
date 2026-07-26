// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended layer-specific tensor IR validators: Embedding, Attention,
//! LayerNorm, Linear, MatMul.
//!
//! Extracted from `tensor_ir_validate_layers.rs` to stay under 500-line limit.

use super::super::{TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNodeId};

impl TensorKernelDef {
    pub(super) fn validate_embedding(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        weight: TensorNodeId,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, weight)?;

        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.is_empty() {
            return Err(TensorIRLayerError::EmbeddingInputScalar.into());
        }

        let weight_shape = &self.nodes[weight.index()].shape;
        if weight_shape.len() != 2 {
            return Err(TensorIRLayerError::EmbeddingWeightNotMatrix {
                shape: weight_shape.clone(),
            }
            .into());
        }

        Ok(())
    }

    /// Validate Attention: Q/K rank >= 2, Q[-1] == K[-1], K[-2] == V[-2], scale valid.
    pub(super) fn validate_attention(
        &self,
        current: TensorNodeId,
        q: TensorNodeId,
        k: TensorNodeId,
        v: TensorNodeId,
        scale: Option<f32>,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, q)?;
        self.check_ref(current, k)?;
        self.check_ref(current, v)?;

        let q_shape = &self.nodes[q.index()].shape;
        if q_shape.len() < 2 {
            return Err(TensorIRLayerError::AttentionRankTooLow {
                side: "Q",
                rank: q_shape.len(),
            }
            .into());
        }

        let k_shape = &self.nodes[k.index()].shape;
        if k_shape.len() < 2 {
            return Err(TensorIRLayerError::AttentionRankTooLow {
                side: "K",
                rank: k_shape.len(),
            }
            .into());
        }

        let v_shape = &self.nodes[v.index()].shape;
        if v_shape.len() < 2 {
            return Err(TensorIRLayerError::AttentionRankTooLow {
                side: "V",
                rank: v_shape.len(),
            }
            .into());
        }

        // Q head dim (D) must match K head dim
        let q_d = q_shape[q_shape.len() - 1];
        let k_d = k_shape[k_shape.len() - 1];
        if q_d != k_d {
            return Err(TensorIRLayerError::AttentionHeadDimMismatch { q_d, k_d }.into());
        }

        // K sequence length (T_kv) must match V sequence length
        let k_t = k_shape[k_shape.len() - 2];
        let v_t = v_shape[v_shape.len() - 2];
        if k_t != v_t {
            return Err(TensorIRLayerError::AttentionKvSeqMismatch { k_t, v_t }.into());
        }

        // Scale must be finite and positive if present
        if let Some(s) = scale {
            if !s.is_finite() || s <= 0.0 {
                return Err(TensorIRLayerError::AttentionScaleInvalid { value: s }.into());
            }
        }

        Ok(())
    }

    pub(super) fn validate_layer_norm(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        eps: TensorNodeId,
        axis: usize,
        weight: TensorNodeId,
        bias: TensorNodeId,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, eps)?;
        self.check_ref(current, weight)?;
        self.check_ref(current, bias)?;
        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 2 {
            return Err(TensorIRLayerError::LayerNormRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }
        let eps_shape = &self.nodes[eps.index()].shape;
        if eps_shape != &[1] {
            return Err(TensorIRLayerError::LayerNormEpsNotScalar {
                shape: eps_shape.clone(),
            }
            .into());
        }
        if axis + 1 != input_shape.len() {
            return Err(TensorIRLayerError::LayerNormAxisNotLast {
                axis,
                rank: input_shape.len(),
            }
            .into());
        }
        let hidden = input_shape[input_shape.len() - 1];
        let weight_shape = &self.nodes[weight.index()].shape;
        if weight_shape != &[hidden] {
            return Err(TensorIRLayerError::LayerNormWeightShape {
                expected_hidden: hidden,
                got_shape: weight_shape.clone(),
            }
            .into());
        }
        let bias_shape = &self.nodes[bias.index()].shape;
        if bias_shape != &[hidden] {
            return Err(TensorIRLayerError::LayerNormBiasShape {
                expected_hidden: hidden,
                got_shape: bias_shape.clone(),
            }
            .into());
        }
        Ok(())
    }

    pub(super) fn validate_linear(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        weight: TensorNodeId,
        bias: Option<TensorNodeId>,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, weight)?;

        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.is_empty() {
            return Err(TensorIRLayerError::LinearInputScalar.into());
        }

        let weight_shape = &self.nodes[weight.index()].shape;
        if weight_shape.len() != 2 {
            return Err(TensorIRLayerError::LinearWeightNotMatrix {
                shape: weight_shape.clone(),
            }
            .into());
        }

        let in_features = input_shape[input_shape.len() - 1];
        let weight_in = weight_shape[1];
        if in_features != weight_in {
            return Err(TensorIRLayerError::LinearFeatureMismatch {
                input_features: in_features,
                weight_in,
            }
            .into());
        }

        let out_features = weight_shape[0];
        if let Some(bias_id) = bias {
            self.check_ref(current, bias_id)?;
            let bias_shape = &self.nodes[bias_id.index()].shape;
            if bias_shape != &[out_features] {
                return Err(TensorIRLayerError::LinearBiasShape {
                    expected: out_features,
                    got_shape: bias_shape.clone(),
                }
                .into());
            }
        }

        Ok(())
    }

    /// Validate a MatMul op: both inputs rank >= 2, contracted dims match, scale valid.
    pub(super) fn validate_matmul(
        &self,
        current: TensorNodeId,
        left: TensorNodeId,
        right: TensorNodeId,
        transpose_right: bool,
        scale: Option<f32>,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, left)?;
        self.check_ref(current, right)?;

        let left_shape = &self.nodes[left.index()].shape;
        if left_shape.len() < 2 {
            return Err(TensorIRLayerError::MatMulRankTooLow {
                side: "left".to_string(),
                rank: left_shape.len(),
            }
            .into());
        }

        let right_shape = &self.nodes[right.index()].shape;
        if right_shape.len() < 2 {
            return Err(TensorIRLayerError::MatMulRankTooLow {
                side: "right".to_string(),
                rank: right_shape.len(),
            }
            .into());
        }

        let left_k = left_shape[left_shape.len() - 1];
        let right_k = if transpose_right {
            right_shape[right_shape.len() - 1]
        } else {
            right_shape[right_shape.len() - 2]
        };
        if left_k != right_k {
            return Err(TensorIRLayerError::MatMulDimMismatch { left_k, right_k }.into());
        }

        if let Some(s) = scale {
            if !s.is_finite() || s == 0.0 {
                return Err(TensorIRLayerError::MatMulScaleInvalid { value: s }.into());
            }
        }

        Ok(())
    }
}
