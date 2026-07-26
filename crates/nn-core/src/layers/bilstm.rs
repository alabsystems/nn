// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional LSTM extracted from `lstm.rs` (#1442).

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};

use super::{Lstm, LstmState};

/// Bidirectional LSTM: runs a forward LSTM and a backward LSTM, concatenating outputs.
///
/// Output hidden size is `2 * hidden_size` (forward + backward concatenated along feature dim).
///
/// Used by Kokoro TTS TextEncoder and ProsodyPredictor where bidirectional context improves
/// text feature encoding.
#[derive(Debug, Clone)]
pub struct BiLstm {
    forward_lstm: Lstm,
    backward_lstm: Lstm,
    hidden_size: usize,
}

impl BiLstm {
    /// Create a bidirectional LSTM from forward and backward weight tensors.
    ///
    /// Both LSTMs must have the same `hidden_size`.
    pub fn new(forward_lstm: Lstm, backward_lstm: Lstm) -> Result<Self> {
        if forward_lstm.hidden_size() != backward_lstm.hidden_size() {
            return Err(TensorError::shape_mismatch(
                vec![forward_lstm.hidden_size()],
                vec![backward_lstm.hidden_size()],
            ));
        }
        let hidden_size = forward_lstm.hidden_size();
        Ok(Self {
            forward_lstm,
            backward_lstm,
            hidden_size,
        })
    }

    /// Create from PyTorch-style weight names (layer 0 bidirectional).
    ///
    /// Expects weights named: `weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`, `bias_hh_l0`
    /// for forward, and `weight_ih_l0_reverse`, `weight_hh_l0_reverse`, etc. for backward.
    pub fn from_weights(
        w_ih_fwd: DynTensor,
        w_hh_fwd: DynTensor,
        b_ih_fwd: Option<DynTensor>,
        b_hh_fwd: Option<DynTensor>,
        w_ih_rev: DynTensor,
        w_hh_rev: DynTensor,
        b_ih_rev: Option<DynTensor>,
        b_hh_rev: Option<DynTensor>,
        hidden_size: usize,
    ) -> Result<Self> {
        let forward_lstm = Lstm::new(w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd, hidden_size)?;
        let backward_lstm = Lstm::new(w_ih_rev, w_hh_rev, b_ih_rev, b_hh_rev, hidden_size)?;
        Ok(Self {
            forward_lstm,
            backward_lstm,
            hidden_size,
        })
    }

    /// Hidden size of each direction (total output is `2 * hidden_size`).
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Forward LSTM reference.
    #[must_use]
    pub fn forward_lstm(&self) -> &Lstm {
        &self.forward_lstm
    }

    /// Backward LSTM reference.
    #[must_use]
    pub fn backward_lstm(&self) -> &Lstm {
        &self.backward_lstm
    }

    /// Run bidirectional LSTM over a sequence (time-first layout).
    ///
    /// # Arguments
    /// - `input`: shape `[seq_len, batch, input_size]`
    /// - `state_fwd`: optional initial state for forward direction
    /// - `state_bwd`: optional initial state for backward direction
    ///
    /// # Returns
    /// `(outputs, final_state_fwd, final_state_bwd)` where
    /// `outputs` is `[seq_len, batch, 2 * hidden_size]` (forward + backward concatenated).
    pub fn forward_seq(
        &self,
        input: &DynTensor,
        state_fwd: Option<&LstmState>,
        state_bwd: Option<&LstmState>,
    ) -> Result<(DynTensor, LstmState, LstmState)> {
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }
        // Forward direction: process sequence as-is.
        let (fwd_outputs, fwd_final) = self.forward_lstm.forward_seq(input, state_fwd)?;

        // Backward direction: reverse the sequence along dim 0, run LSTM, reverse output.
        let reversed_input = input.flip(0)?;
        let (bwd_outputs_rev, bwd_final) =
            self.backward_lstm.forward_seq(&reversed_input, state_bwd)?;
        let bwd_outputs = bwd_outputs_rev.flip(0)?;

        // Concatenate forward and backward outputs along feature dim (dim 2).
        // fwd_outputs: [seq_len, batch, hidden_size]
        // bwd_outputs: [seq_len, batch, hidden_size]
        // result: [seq_len, batch, 2 * hidden_size]
        // Single cat along dim=2 replaces per-timestep narrow+cat O(n²) loop.
        let outputs = DynTensor::cat(&[&fwd_outputs, &bwd_outputs], 2)?;

        Ok((outputs, fwd_final, bwd_final))
    }

    /// Run bidirectional LSTM over a sequence (batch-first layout).
    ///
    /// Convenience wrapper: accepts `[batch, seq_len, input_size]`, transposes to
    /// time-first internally, runs the BiLSTM, and transposes the output back to
    /// `[batch, seq_len, 2 * hidden_size]`.
    ///
    /// Eliminates explicit transpose pairs in callers where data is naturally
    /// batch-first (e.g., ProsodyPredictor, DurationEncoder).
    ///
    /// Part of #2492.
    pub fn forward_seq_batch_first(
        &self,
        input: &DynTensor,
        state_fwd: Option<&LstmState>,
        state_bwd: Option<&LstmState>,
    ) -> Result<(DynTensor, LstmState, LstmState)> {
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }
        // [batch, seq_len, input_size] → [seq_len, batch, input_size]
        let time_first = input.transpose(0, 1)?;
        let (output, fwd_state, bwd_state) = self.forward_seq(&time_first, state_fwd, state_bwd)?;
        // [seq_len, batch, 2*hidden] → [batch, seq_len, 2*hidden]
        let batch_first = output.transpose(0, 1)?;
        Ok((batch_first, fwd_state, bwd_state))
    }
}
