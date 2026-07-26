// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Trainable LSTM cell with gradient flow through recurrent connections.
//!
//! Decomposes the LSTM cell into existing tracked ops (matmul, sigmoid, tanh,
//! mul, add, narrow) so gradients flow through the standard backward rules
//! without a dedicated `Op::Lstm` variant.
//!
//! Gate order follows PyTorch convention: input (i), forget (f), cell (g), output (o).
//!
//! Required for dvoice model fine-tuning (Silero VAD, HTDemucs both use LSTM).

use crate::error::Result;
use crate::tracked::TrackedTensor;
use crate::var::Var;
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use std::sync::Arc;

use super::TrainableModule;

/// Tracked LSTM hidden/cell state for gradient flow through recurrent steps.
///
/// Both `h` and `c` are `Arc<TrackedTensor>` so gradients propagate backward
/// through multiple timesteps (truncated BPTT).
#[derive(Debug, Clone)]
pub struct TrackedLstmState {
    /// Hidden state `[batch, hidden_size]`.
    pub h: Arc<TrackedTensor>,
    /// Cell state `[batch, hidden_size]`.
    pub c: Arc<TrackedTensor>,
}

/// An LSTM cell with trainable `Var` weights.
///
/// Decomposes the LSTM computation into:
/// ```text
/// gates = input @ w_ih^T + h @ w_hh^T + b_ih + b_hh
/// i, f, g, o = split(gates, 4)
/// i = sigmoid(i), f = sigmoid(f), g = tanh(g), o = sigmoid(o)
/// c_new = f * c + i * g
/// h_new = o * tanh(c_new)
/// ```
///
/// All operations are tracked on the gradient tape via existing `Op` variants.
/// No dedicated `Op::Lstm` is needed — gradients flow correctly through the
/// composition of matmul, sigmoid, tanh, narrow, mul, and add backward rules.
///
/// Matches `layers::Lstm` semantics (weight layout, gate order, bias handling).
#[derive(Debug, Clone)]
pub struct TrainableLstm {
    w_ih: Var,         // [4 * hidden_size, input_size]
    w_hh: Var,         // [4 * hidden_size, hidden_size]
    b_ih: Option<Var>, // [4 * hidden_size]
    b_hh: Option<Var>, // [4 * hidden_size]
    hidden_size: usize,
}

impl TrainableLstm {
    /// Create a new LSTM cell with uniform initialization.
    ///
    /// Matches PyTorch's `nn.LSTMCell` default: U(-k, k) where k = 1/sqrt(hidden_size).
    ///
    /// `input_size`: dimension of the input vector.
    /// `hidden_size`: dimension of the hidden/cell state.
    /// `bias`: whether to include bias terms.
    pub fn new(input_size: usize, hidden_size: usize, bias: bool) -> Result<Self> {
        let k = 1.0 / (hidden_size as f64).sqrt();
        let w_ih = Var::rand(&[4 * hidden_size, input_size], -k, k, &Device::Cpu)?;
        let w_hh = Var::rand(&[4 * hidden_size, hidden_size], -k, k, &Device::Cpu)?;
        let (b_ih, b_hh) = if bias {
            (
                Some(Var::rand(&[4 * hidden_size], -k, k, &Device::Cpu)?),
                Some(Var::rand(&[4 * hidden_size], -k, k, &Device::Cpu)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
        })
    }

    /// Create from existing `Var`s.
    ///
    /// `w_ih`: `[4*hidden_size, input_size]`
    /// `w_hh`: `[4*hidden_size, hidden_size]`
    /// `b_ih`, `b_hh`: `[4*hidden_size]` (optional)
    pub fn from_vars(
        w_ih: Var,
        w_hh: Var,
        b_ih: Option<Var>,
        b_hh: Option<Var>,
        hidden_size: usize,
    ) -> Self {
        Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
        }
    }

    /// Create from `DynTensor`s (wraps each in a new `Var`).
    pub fn from_tensors(
        w_ih: DynTensor,
        w_hh: DynTensor,
        b_ih: Option<DynTensor>,
        b_hh: Option<DynTensor>,
        hidden_size: usize,
    ) -> Self {
        Self {
            w_ih: Var::new(w_ih),
            w_hh: Var::new(w_hh),
            b_ih: b_ih.map(Var::new),
            b_hh: b_hh.map(Var::new),
            hidden_size,
        }
    }

    /// Hidden size of this LSTM cell.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Reference to `w_ih` weight.
    #[must_use]
    pub fn w_ih(&self) -> &Var {
        &self.w_ih
    }

    /// Reference to `w_hh` weight.
    #[must_use]
    pub fn w_hh(&self) -> &Var {
        &self.w_hh
    }

    /// Single-step LSTM cell forward with tracked state.
    ///
    /// `input`: `[batch, input_size]`
    /// `state`: previous `(h, c)` or `None` for zero-init.
    ///
    /// Returns `(h_new, TrackedLstmState)` where `h_new` is the output and
    /// the state contains both `h_new` and `c_new` for the next timestep.
    pub fn forward_cell(
        &self,
        input: &Arc<TrackedTensor>,
        state: Option<&TrackedLstmState>,
    ) -> Result<(Arc<TrackedTensor>, TrackedLstmState)> {
        let batch = input.tensor().dims()[0];
        let h_size = self.hidden_size;

        // Get or zero-init previous state.
        let (h_prev, c_prev) = match state {
            Some(s) => (s.h.clone(), s.c.clone()),
            None => {
                let h0 = DynTensor::zeros(&[batch, h_size], DType::F32, &Device::Cpu)?;
                let c0 = DynTensor::zeros(&[batch, h_size], DType::F32, &Device::Cpu)?;
                (
                    Arc::new(TrackedTensor::from_tensor(h0)),
                    Arc::new(TrackedTensor::from_tensor(c0)),
                )
            }
        };

        // Wrap weights as tracked tensors for gradient flow.
        let w_ih = Arc::new(TrackedTensor::from_var(&self.w_ih)?);
        let w_hh = Arc::new(TrackedTensor::from_var(&self.w_hh)?);

        // Transpose weights: [4H, I] -> [I, 4H] and [4H, H] -> [H, 4H]
        let w_ih_t = w_ih.transpose(0, 1)?;
        let w_hh_t = w_hh.transpose(0, 1)?;

        // gates = input @ w_ih^T + h_prev @ w_hh^T
        let gates = input.matmul(&w_ih_t)?;
        let gates = gates.add(&h_prev.matmul(&w_hh_t)?)?;

        // Add biases.
        let gates = if let Some(ref b) = self.b_ih {
            let b_tracked = Arc::new(TrackedTensor::from_var(b)?);
            gates.add(&b_tracked)?
        } else {
            gates
        };
        let gates = if let Some(ref b) = self.b_hh {
            let b_tracked = Arc::new(TrackedTensor::from_var(b)?);
            gates.add(&b_tracked)?
        } else {
            gates
        };

        // Split gates: [batch, 4*H] -> i, f, g, o each [batch, H]
        let i_gate = gates.narrow(1, 0, h_size)?;
        let f_gate = gates.narrow(1, h_size, h_size)?;
        let g_gate = gates.narrow(1, 2 * h_size, h_size)?;
        let o_gate = gates.narrow(1, 3 * h_size, h_size)?;

        // Gate activations.
        let i_gate = i_gate.sigmoid()?;
        let f_gate = f_gate.sigmoid()?;
        let g_gate = g_gate.tanh()?;
        let o_gate = o_gate.sigmoid()?;

        // State update: c_new = f * c + i * g
        let c_new = f_gate.mul(&c_prev)?.add(&i_gate.mul(&g_gate)?)?;

        // Output: h_new = o * tanh(c_new)
        let h_new = o_gate.mul(&c_new.tanh()?)?;

        let new_state = TrackedLstmState {
            h: h_new.clone(),
            c: c_new,
        };
        Ok((h_new, new_state))
    }

    /// Multi-step LSTM forward over a sequence.
    ///
    /// `input_seq`: `[batch, seq_len, input_size]`
    /// `state`: initial state or `None` for zero-init.
    ///
    /// Returns `(outputs, final_state)` where `outputs` is a Vec of per-step
    /// hidden states. Use `TrackedTensor::cat` on the outputs to get the full
    /// sequence tensor `[batch, seq_len, hidden_size]`.
    pub fn forward_seq(
        &self,
        input_seq: &Arc<TrackedTensor>,
        state: Option<&TrackedLstmState>,
    ) -> Result<(Vec<Arc<TrackedTensor>>, TrackedLstmState)> {
        let seq_len = input_seq.tensor().dims()[1];
        let mut current_state = state.cloned();
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Slice timestep: [batch, 1, input_size] -> squeeze -> [batch, input_size]
            let x_t = input_seq.narrow(1, t, 1)?;
            let x_t = x_t.squeeze(1)?;

            let (h_new, new_state) = self.forward_cell(&x_t, current_state.as_ref())?;
            outputs.push(h_new);
            current_state = Some(new_state);
        }

        let final_state = current_state.ok_or(crate::error::AutodiffError::EmptySequence {
            op: "LSTM forward_seq",
        })?;
        Ok((outputs, final_state))
    }
}

impl TrainableModule for TrainableLstm {
    /// Single-step forward for the `TrainableModule` trait.
    ///
    /// Input shape: `[batch, input_size]`.
    /// Output shape: `[batch, hidden_size]`.
    ///
    /// Always uses zero-initialized state. For recurrent usage across
    /// timesteps, use `forward_cell` or `forward_seq` directly.
    fn forward(&self, x: &Arc<TrackedTensor>) -> Result<Arc<TrackedTensor>> {
        let (h_new, _state) = self.forward_cell(x, None)?;
        Ok(h_new)
    }

    fn vars(&self) -> Vec<&Var> {
        let mut v = vec![&self.w_ih, &self.w_hh];
        if let Some(ref b) = self.b_ih {
            v.push(b);
        }
        if let Some(ref b) = self.b_hh {
            v.push(b);
        }
        v
    }
}
