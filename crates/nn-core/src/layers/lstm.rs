// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM layer for [`DynTensor`].
//!
//! Provides [`Lstm`] matching candle-nn's `LSTM` API for drop-in replacement.
//! Implements the standard LSTM cell equations:
//!
//! ```text
//! gates = x @ w_ih^T + h @ w_hh^T + b_ih + b_hh
//! i, f, g, o = split(gates, 4)  — input, forget, cell, output gates
//! c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
//! h_new = sigmoid(o) * tanh(c_new)
//! ```

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::layers::check_output_finite;
use crate::layers::validation::validate_weight_finite;
use crate::layers::{nan_check_policy, NanCheckPolicy};
use crate::{DType, Result, TensorError};

#[path = "lstm_gpu.rs"]
mod lstm_gpu;

#[path = "lstm_seq.rs"]
mod lstm_seq;

/// LSTM state: hidden state `h` and cell state `c`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LstmState {
    /// Hidden state, shape `[batch, hidden_size]`.
    pub h: DynTensor,
    /// Cell state, shape `[batch, hidden_size]`.
    pub c: DynTensor,
}

impl LstmState {
    /// Create a new LSTM state from hidden and cell tensors.
    ///
    /// Validates that `h` and `c` have identical shapes.
    pub fn new(h: DynTensor, c: DynTensor) -> Result<Self> {
        if h.dims() != c.dims() {
            return Err(TensorError::shape_mismatch(
                h.dims().to_vec(),
                c.dims().to_vec(),
            ));
        }
        Ok(Self { h, c })
    }

    /// Validate that h and c contain no NaN/Inf values.
    ///
    /// Called at LSTM forward entry when caller provides initial state.
    /// Zero-initialized states skip this check (they are finite by construction).
    /// Uses `any_non_finite()` fast path (no GPU→CPU round-trip on the happy path).
    pub fn validate_finiteness(&self) -> Result<()> {
        if nan_check_policy() == NanCheckPolicy::Skip {
            return Ok(());
        }
        if self.h.any_non_finite()? {
            return Err(TensorError::NonFiniteData {
                name: "LSTM initial state h0".to_string(),
                count: 1, // at least 1; exact count requires CPU readback
            });
        }
        if self.c.any_non_finite()? {
            return Err(TensorError::NonFiniteData {
                name: "LSTM initial state c0".to_string(),
                count: 1,
            });
        }
        Ok(())
    }
}

/// Single-layer LSTM cell.
///
/// Matches candle-nn `LSTM` API. Weight layout follows PyTorch convention:
/// - `w_ih`: `[4 * hidden_size, input_size]` (input-hidden weights)
/// - `w_hh`: `[4 * hidden_size, hidden_size]` (hidden-hidden weights)
/// - `b_ih`: `[4 * hidden_size]` (input-hidden bias, optional)
/// - `b_hh`: `[4 * hidden_size]` (hidden-hidden bias, optional)
///
/// Gate order (PyTorch convention): input, forget, cell (g), output.
#[derive(Debug, Clone)]
pub struct Lstm {
    w_ih: DynTensor,
    w_hh: DynTensor,
    b_ih: Option<DynTensor>,
    b_hh: Option<DynTensor>,
    hidden_size: usize,
}

impl Lstm {
    /// Create an LSTM cell from weight tensors.
    ///
    /// - `w_ih`: shape `[4 * hidden_size, input_size]`
    /// - `w_hh`: shape `[4 * hidden_size, hidden_size]`
    /// - `b_ih`, `b_hh`: shape `[4 * hidden_size]` (optional)
    /// - `hidden_size`: size of the hidden state
    pub fn new(
        w_ih: DynTensor,
        w_hh: DynTensor,
        b_ih: Option<DynTensor>,
        b_hh: Option<DynTensor>,
        hidden_size: usize,
    ) -> Result<Self> {
        if hidden_size == 0 {
            return Err(TensorError::ValueOutOfRange {
                description: "LSTM hidden_size must be > 0",
            });
        }
        // Validate w_ih shape: [4*H, input_size]
        let (wih_rows, _) = w_ih.dims2().map_err(|_| TensorError::RankMismatch {
            expected: 2,
            actual: w_ih.dims().len(),
        })?;
        if wih_rows != 4 * hidden_size {
            return Err(TensorError::shape_mismatch(
                vec![4 * hidden_size, wih_rows],
                w_ih.dims().to_vec(),
            ));
        }
        // Validate w_hh shape: [4*H, H]
        let (whh_rows, whh_cols) = w_hh.dims2().map_err(|_| TensorError::RankMismatch {
            expected: 2,
            actual: w_hh.dims().len(),
        })?;
        if whh_rows != 4 * hidden_size || whh_cols != hidden_size {
            return Err(TensorError::shape_mismatch(
                vec![4 * hidden_size, hidden_size],
                vec![whh_rows, whh_cols],
            ));
        }
        // Validate bias shapes: must be 1D with length 4*hidden_size.
        let four_h = 4 * hidden_size;
        if let Some(ref b) = b_ih {
            let b_dims = b.dims();
            if b_dims != [four_h] {
                return Err(TensorError::shape_mismatch(vec![four_h], b_dims.to_vec()));
            }
        }
        if let Some(ref b) = b_hh {
            let b_dims = b.dims();
            if b_dims != [four_h] {
                return Err(TensorError::shape_mismatch(vec![four_h], b_dims.to_vec()));
            }
        }

        // Validate weight finiteness: reject NaN/Inf at construction (#2064).
        // LSTM gates (sigmoid/tanh) amplify non-finite values through recurrence,
        // making construction-time rejection essential.
        validate_weight_finite(&w_ih, "LSTM w_ih")?;
        validate_weight_finite(&w_hh, "LSTM w_hh")?;
        if let Some(ref b) = b_ih {
            validate_weight_finite(b, "LSTM b_ih")?;
        }
        if let Some(ref b) = b_hh {
            validate_weight_finite(b, "LSTM b_hh")?;
        }

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_size,
        })
    }

    /// Hidden size.
    #[must_use]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Input-hidden weight reference.
    #[must_use]
    pub fn w_ih(&self) -> &DynTensor {
        &self.w_ih
    }

    /// Hidden-hidden weight reference.
    #[must_use]
    pub fn w_hh(&self) -> &DynTensor {
        &self.w_hh
    }

    /// Run one LSTM step.
    ///
    /// # Arguments
    /// - `input`: shape `[batch, input_size]`
    /// - `state`: optional `(h, c)` each `[batch, hidden_size]`.
    ///   If `None`, zero-initialized.
    ///
    /// # Returns
    /// `(output, new_state)` where output is `h_new` `[batch, hidden_size]`.
    /// Matches candle-nn `LSTM::forward()` and nn's `GatedDeltaNet::forward()`.
    pub fn forward(
        &self,
        input: &DynTensor,
        state: Option<&LstmState>,
    ) -> Result<(DynTensor, LstmState)> {
        // Auto-upcast BF16/F16 to F32 for numerical stability (#1990).
        // LSTM gate computation (sigmoid, tanh, matmul) compounds precision errors
        // across recurrent timesteps in half-precision. Matches softmax pattern (#1813).
        let original_dtype = input.dtype();
        if matches!(original_dtype, DType::BF16 | DType::F16) {
            let f32_input = input.to_dtype(DType::F32)?;
            let (output, new_state) = self.forward(&f32_input, state)?;
            let output = output.to_dtype(original_dtype)?;
            let h = new_state.h.to_dtype(original_dtype)?;
            let c = new_state.c.to_dtype(original_dtype)?;
            return Ok((output, LstmState::new(h, c)?));
        }
        let tracing = trace::is_tracing();
        // Suppress decomposed ops (matmul, sigmoid, narrow, etc.) during tracing —
        // only the composite LSTM op should appear in the trace graph.
        // Matches Linear/Conv1d/Embedding pattern.
        let compute = |slf: &Self| -> Result<(DynTensor, LstmState)> {
            // GPU fast-path: fused LSTM cell avoids ~18 separate dispatches.
            // Uses non-transposed weights — add_linear handles transpose internally.
            if input.device().is_gpu() {
                if let Some(result) = slf.try_gpu_lstm_cell(input, state) {
                    return result;
                }
            }
            let w_ih_t = slf.w_ih.transpose(0, 1)?;
            let w_hh_t = slf.w_hh.transpose(0, 1)?;
            slf.forward_with_transposed(input, state, &w_ih_t, &w_hh_t)
        };
        let (mut output, new_state) = if tracing {
            trace::with_trace_suppressed(|| compute(self))?
        } else {
            compute(self)?
        };

        // Record composite LSTM op for trace-to-graph verification pipeline.
        // LSTM requires 3 trace inputs: x, h, c. Resolve h/c from caller
        // state or create zero-valued graph inputs when state is None.
        if tracing {
            let (mut h_trace, mut c_trace) = match state {
                Some(s) => (s.h.clone(), s.c.clone()),
                None => {
                    let batch = input.dims()[0];
                    let device = input.device();
                    (
                        DynTensor::zeros(&[batch, self.hidden_size], DType::F32, &device)?,
                        DynTensor::zeros(&[batch, self.hidden_size], DType::F32, &device)?,
                    )
                }
            };
            // Ensure h and c have trace IDs. If they came from outside the
            // trace context (user-supplied initial state or freshly created
            // zeros), register them as graph Input nodes.
            if h_trace.trace_id().is_none() {
                if let Some(id) = trace::record_input(h_trace.dims(), h_trace.dtype()) {
                    h_trace.set_trace_id(id);
                }
            }
            if c_trace.trace_id().is_none() {
                if let Some(id) = trace::record_input(c_trace.dims(), c_trace.dtype()) {
                    c_trace.set_trace_id(id);
                }
            }
            let input_ids = DynTensor::trace_input_ids(&[input, &h_trace, &c_trace])?;
            // Record initial state node IDs when caller provided non-zero state.
            // None = zero-initialized (sound for first timestep only, see #2401).
            let (init_h, init_c) = if state.is_some() {
                (h_trace.trace_id(), c_trace.trace_id())
            } else {
                (None, None)
            };
            if let Some(id) = trace::record_op(
                TraceOp::Lstm {
                    weight_ih: self.w_ih.to_weight_ref()?,
                    weight_hh: self.w_hh.to_weight_ref()?,
                    bias_ih: self
                        .b_ih
                        .as_ref()
                        .map(DynTensor::to_weight_ref)
                        .transpose()?,
                    bias_hh: self
                        .b_hh
                        .as_ref()
                        .map(DynTensor::to_weight_ref)
                        .transpose()?,
                    hidden_size: self.hidden_size,
                    initial_hidden: init_h,
                    initial_cell: init_c,
                },
                &input_ids,
                output.dims(),
                output.dtype(),
            ) {
                output.set_trace_id(id);
            }
        }

        Ok((output, new_state))
    }

    /// LSTM step with pre-transposed weights (avoids recomputing per timestep).
    ///
    /// All computation is F32, matching PyTorch's MKL path. GPU path uses the
    /// fused Metal LSTM kernel when available.
    fn forward_with_transposed(
        &self,
        input: &DynTensor,
        state: Option<&LstmState>,
        w_ih_t: &DynTensor,
        w_hh_t: &DynTensor,
    ) -> Result<(DynTensor, LstmState)> {
        let (batch, _input_size) = input.dims2().map_err(|_| TensorError::RankMismatch {
            expected: 2,
            actual: input.dims().len(),
        })?;

        let device = input.device();

        // Validate caller-provided state finiteness (#R1-1151 F1).
        // NaN/Inf in h0/c0 propagates through LSTM gates — sigmoid(Inf)=1.0
        // silently absorbs Inf, producing finite-looking but incorrect output.
        if let Some(s) = state {
            s.validate_finiteness()?;
        }

        let h = match state {
            Some(s) => s.h.clone(),
            None => DynTensor::zeros(&[batch, self.hidden_size], DType::F32, &device)?,
        };
        let c = match state {
            Some(s) => s.c.clone(),
            None => DynTensor::zeros(&[batch, self.hidden_size], DType::F32, &device)?,
        };

        // gates = input @ w_ih^T + h @ w_hh^T
        let mut gates = input.matmul(w_ih_t)?.add(&h.matmul(w_hh_t)?)?;

        // Add biases if present.
        if let Some(ref b) = self.b_ih {
            gates = gates.broadcast_add(b)?;
        }
        if let Some(ref b) = self.b_hh {
            gates = gates.broadcast_add(b)?;
        }

        // Split gates: [batch, 4*H] -> i, f, g, o each [batch, H]
        let h_size = self.hidden_size;
        let i_gate = gates.narrow(1, 0, h_size)?;
        let f_gate = gates.narrow(1, h_size, h_size)?;
        let g_gate = gates.narrow(1, 2 * h_size, h_size)?;
        let o_gate = gates.narrow(1, 3 * h_size, h_size)?;

        // Apply activations.
        let i_gate = i_gate.sigmoid()?;
        let f_gate = f_gate.sigmoid()?;
        let g_gate = g_gate.tanh()?;
        let o_gate = o_gate.sigmoid()?;

        // c_new = f * c + i * g
        let c_new = f_gate.mul(&c)?.add(&i_gate.mul(&g_gate)?)?;

        // h_new = o * tanh(c_new)
        let h_new = o_gate.mul(&c_new.tanh()?)?;

        // Tier 1 finiteness check (#1209): LSTM gates use sigmoid/tanh (exp-based).
        check_output_finite(&h_new, "LSTM h")?;
        check_output_finite(&c_new, "LSTM c")?;

        let output = h_new.clone();
        Ok((output, LstmState::new(h_new, c_new)?))
    }
}

/// Alias for candle-nn API compatibility (`candle_nn::rnn::LSTMCell`).
pub type LstmCell = Lstm;

// -- Bidirectional LSTM (extracted to bilstm.rs) ------------------------------

#[path = "bilstm.rs"]
mod bilstm_impl;
pub use bilstm_impl::BiLstm;

#[cfg(kani)]
#[path = "kani_lstm_proofs.rs"]
mod kani_lstm_proofs;

#[cfg(test)]
#[path = "lstm_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bilstm_tests.rs"]
mod bilstm_tests;
