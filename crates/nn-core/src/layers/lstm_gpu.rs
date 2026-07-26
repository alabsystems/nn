// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused GPU LSTM cell dispatch for [`Lstm`].
//!
//! Extracted from `lstm.rs` for the 500-line limit.
//! Part of #1373 (fused GPU LSTM cell).

use crate::dyn_tensor::gpu::{gpu_backend_dispatch_pair, gpu_backend_dispatch_triple};
use crate::dyn_tensor::DynTensor;
use crate::layers::check_output_finite;
use crate::{DType, Result};

use super::{Lstm, LstmState};

impl Lstm {
    /// Try fused GPU LSTM cell dispatch. Returns `None` to fall back to CPU.
    pub(super) fn try_gpu_lstm_cell(
        &self,
        input: &DynTensor,
        state: Option<&LstmState>,
    ) -> Option<Result<(DynTensor, LstmState)>> {
        let (batch, _) = match input.dims2() {
            Ok(d) => d,
            Err(e) => return Some(Err(e)),
        };

        // Validate caller-provided state finiteness (#R1-1151 F1).
        if let Some(s) = state {
            if let Err(e) = s.validate_finiteness() {
                return Some(Err(e));
            }
        }

        let device = input.device();
        let h = match state {
            Some(s) => s.h.clone(),
            None => match DynTensor::zeros(&[batch, self.hidden_size], DType::F32, &device) {
                Ok(z) => z,
                Err(e) => return Some(Err(e)),
            },
        };
        let c = match state {
            Some(s) => s.c.clone(),
            None => match DynTensor::zeros(&[batch, self.hidden_size], DType::F32, &device) {
                Ok(z) => z,
                Err(e) => return Some(Err(e)),
            },
        };

        // Combine biases into single [4*H] tensor for the fused kernel.
        let combined_bias = match (&self.b_ih, &self.b_hh) {
            (Some(bih), Some(bhh)) => match bih.add(bhh) {
                Ok(b) => Some(b),
                Err(e) => return Some(Err(e)),
            },
            (Some(b), None) | (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };

        let pair = gpu_backend_dispatch_pair(|backend| {
            backend.lstm_cell(
                input,
                &h,
                &c,
                &self.w_ih,
                &self.w_hh,
                combined_bias.as_ref(),
                self.hidden_size,
            )
        })?;

        // Backend now returns (h_new, c_new) directly — no narrow() needed.
        let (h_new, c_new) = match pair {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };

        // Tier 1 finiteness check.
        if let Err(e) = check_output_finite(&h_new, "LSTM h (fused)") {
            return Some(Err(e));
        }
        if let Err(e) = check_output_finite(&c_new, "LSTM c (fused)") {
            return Some(Err(e));
        }

        let output = h_new.clone();
        Some(match LstmState::new(h_new, c_new) {
            Ok(state) => Ok((output, state)),
            Err(e) => Err(e),
        })
    }

    /// Try fused GPU LSTM sequence dispatch over full `[seq_len, batch, input_size]`.
    ///
    /// Returns the full output `[seq_len, batch, hidden_size]` and the final
    /// `LstmState` `(h_n, c_n)` each `[batch, hidden_size]`, all computed in a
    /// single Metal dispatch. Returns `None` to fall back to per-timestep loop.
    ///
    /// Part of #1805 (fused LSTM sequence Metal kernel).
    pub(super) fn try_gpu_lstm_sequence(
        &self,
        input: &DynTensor,
        state: Option<&LstmState>,
    ) -> Option<Result<(DynTensor, LstmState)>> {
        let dims = input.dims();
        if dims.len() != 3 {
            return None; // Fall back — rank validation handled by caller.
        }
        let batch = dims[1];
        let device = input.device();

        // Validate caller-provided state finiteness (#R1-1151 F1).
        if let Some(s) = state {
            if let Err(e) = s.validate_finiteness() {
                return Some(Err(e));
            }
        }

        // Create zero initial states if not provided.
        let h0 = match state {
            Some(s) => s.h.clone(),
            None => match DynTensor::zeros(&[batch, self.hidden_size], DType::F32, &device) {
                Ok(z) => z,
                Err(e) => return Some(Err(e)),
            },
        };
        let c0 = match state {
            Some(s) => s.c.clone(),
            None => match DynTensor::zeros(&[batch, self.hidden_size], DType::F32, &device) {
                Ok(z) => z,
                Err(e) => return Some(Err(e)),
            },
        };

        // Combine biases into single [4*H] tensor for the fused kernel.
        let combined_bias = match (&self.b_ih, &self.b_hh) {
            (Some(bih), Some(bhh)) => match bih.add(bhh) {
                Ok(b) => Some(b),
                Err(e) => return Some(Err(e)),
            },
            (Some(b), None) | (None, Some(b)) => Some(b.clone()),
            (None, None) => None,
        };

        let triple = gpu_backend_dispatch_triple(|backend| {
            backend.lstm_sequence(
                input,
                &self.w_ih,
                &self.w_hh,
                combined_bias.as_ref(),
                &h0,
                &c0,
                self.hidden_size,
            )
        })?;

        let (output, h_n, c_n) = match triple {
            Ok(t) => t,
            Err(e) => return Some(Err(e)),
        };

        // Tier 1 finiteness check on final states.
        if let Err(e) = check_output_finite(&h_n, "LSTM h_n (fused seq)") {
            return Some(Err(e));
        }
        if let Err(e) = check_output_finite(&c_n, "LSTM c_n (fused seq)") {
            return Some(Err(e));
        }

        Some(match LstmState::new(h_n, c_n) {
            Ok(final_state) => Ok((output, final_state)),
            Err(e) => Err(e),
        })
    }
}
