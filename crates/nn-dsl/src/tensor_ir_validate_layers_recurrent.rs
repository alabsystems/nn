// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Recurrent and gated layer validators: GatedDeltaNet, LSTM.
//!
//! Extracted from `tensor_ir_validate_layers_ext.rs` (#1575) to keep files
//! under 400 lines.

use super::super::{TensorIRError, TensorIRLayerError, TensorKernelDef, TensorNodeId};

impl TensorKernelDef {
    /// Validate a Gated DeltaNet cell: Q/K/V/state/gate/beta shapes consistent, scale valid.
    pub(super) fn validate_gated_delta_net(
        &self,
        current: TensorNodeId,
        q: TensorNodeId,
        k: TensorNodeId,
        v: TensorNodeId,
        state: TensorNodeId,
        gate: TensorNodeId,
        beta: TensorNodeId,
        scale: f32,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, q)?;
        self.check_ref(current, k)?;
        self.check_ref(current, v)?;
        self.check_ref(current, state)?;
        self.check_ref(current, gate)?;
        self.check_ref(current, beta)?;

        let q_shape = &self.nodes[q.index()].shape;
        let k_shape = &self.nodes[k.index()].shape;

        // Q and K must have rank >= 2 (at least [H, K])
        if q_shape.len() < 2 {
            return Err(TensorIRLayerError::AttentionRankTooLow {
                side: "Q",
                rank: q_shape.len(),
            }
            .into());
        }
        if k_shape.len() < 2 {
            return Err(TensorIRLayerError::AttentionRankTooLow {
                side: "K",
                rank: k_shape.len(),
            }
            .into());
        }

        // Q and K last dim must match (head dim K)
        let q_dim = q_shape[q_shape.len() - 1];
        let k_dim = k_shape[k_shape.len() - 1];
        if q_dim != k_dim {
            return Err(TensorIRLayerError::GatedDeltaNetQkDimMismatch { q_dim, k_dim }.into());
        }

        // State must have rank >= 3 (at least [H, K, V])
        let state_shape = &self.nodes[state.index()].shape;
        if state_shape.len() < 3 {
            return Err(TensorIRLayerError::GatedDeltaNetStateShape {
                shape: state_shape.clone(),
            }
            .into());
        }

        // Scale must be finite and positive
        if !scale.is_finite() || scale <= 0.0 {
            return Err(TensorIRLayerError::GatedDeltaNetScaleInvalid { value: scale }.into());
        }

        Ok(())
    }

    /// Validate an LSTM cell: weight shapes consistent with input/hidden sizes, bias optional.
    pub(super) fn validate_lstm(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        hidden_state: TensorNodeId,
        cell_state: TensorNodeId,
        weight_ih: TensorNodeId,
        weight_hh: TensorNodeId,
        bias: Option<TensorNodeId>,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;
        self.check_ref(current, hidden_state)?;
        self.check_ref(current, cell_state)?;
        self.check_ref(current, weight_ih)?;
        self.check_ref(current, weight_hh)?;
        if let Some(b) = bias {
            self.check_ref(current, b)?;
        }

        let input_shape = &self.nodes[input.index()].shape;
        let hidden_shape = &self.nodes[hidden_state.index()].shape;
        let cell_shape = &self.nodes[cell_state.index()].shape;
        let wih_shape = &self.nodes[weight_ih.index()].shape;
        let whh_shape = &self.nodes[weight_hh.index()].shape;

        // Input must not be scalar.
        if input_shape.is_empty() {
            return Err(TensorIRLayerError::LstmInputScalar.into());
        }

        // Hidden and cell state must have the same shape.
        if hidden_shape != cell_shape {
            return Err(TensorIRLayerError::LstmHiddenCellMismatch {
                hidden: hidden_shape.clone(),
                cell: cell_shape.clone(),
            }
            .into());
        }

        // weight_ih must be 2-D: [4*H, I]
        if wih_shape.len() != 2 {
            return Err(TensorIRLayerError::LstmWeightIhNotMatrix {
                shape: wih_shape.clone(),
            }
            .into());
        }

        // weight_hh must be 2-D: [4*H, H]
        if whh_shape.len() != 2 {
            return Err(TensorIRLayerError::LstmWeightHhNotMatrix {
                shape: whh_shape.clone(),
            }
            .into());
        }

        let input_features = input_shape[input_shape.len() - 1];
        let hidden_size = hidden_shape[hidden_shape.len() - 1];
        let four_h = 4 * hidden_size;

        // weight_ih must be [4*H, I]
        if wih_shape[0] != four_h || wih_shape[1] != input_features {
            return Err(TensorIRLayerError::LstmWeightIhShape {
                expected_rows: four_h,
                expected_cols: input_features,
                got_shape: wih_shape.clone(),
            }
            .into());
        }

        // weight_hh must be [4*H, H]
        if whh_shape[0] != four_h || whh_shape[1] != hidden_size {
            return Err(TensorIRLayerError::LstmWeightHhShape {
                expected_rows: four_h,
                expected_cols: hidden_size,
                got_shape: whh_shape.clone(),
            }
            .into());
        }

        // Optional bias must be [4*H]
        if let Some(b) = bias {
            let bias_shape = &self.nodes[b.index()].shape;
            if bias_shape != &[four_h] {
                return Err(TensorIRLayerError::LstmBiasShape {
                    expected: four_h,
                    got_shape: bias_shape.clone(),
                }
                .into());
            }
        }

        Ok(())
    }
}
