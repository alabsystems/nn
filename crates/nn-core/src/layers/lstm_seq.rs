// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM sequence processing for [`Lstm`].
//!
//! Extracted from `lstm.rs` for 500-line compliance.
//! Contains [`Lstm::forward_seq`] — the time-major sequence path
//! with fused GPU dispatch (#1805) and per-timestep fallbacks.

use crate::dyn_tensor::trace::{self, TraceOp};
use crate::dyn_tensor::DynTensor;
use crate::layers::{check_output_finite, with_nan_check_policy, NanCheckPolicy};
use crate::{DType, Result, TensorError};

use super::{Lstm, LstmState};

impl Lstm {
    /// Run LSTM over a sequence (time-major).
    ///
    /// # Arguments
    /// - `input`: shape `[seq_len, batch, input_size]`
    /// - `state`: optional initial `(h, c)` each `[batch, hidden_size]`
    ///
    /// # Returns
    /// `(outputs, final_state)` where `outputs` is `[seq_len, batch, hidden_size]`.
    pub fn forward_seq(
        &self,
        input: &DynTensor,
        state: Option<&LstmState>,
    ) -> Result<(DynTensor, LstmState)> {
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(TensorError::RankMismatch {
                expected: 3,
                actual: dims.len(),
            });
        }
        let seq_len = dims[0];
        if seq_len == 0 {
            return Err(TensorError::ZeroLengthDimension {
                axis: 0,
                operation: "LSTM forward_seq",
            });
        }

        // Auto-upcast BF16/F16 to F32 for numerical stability (#1990).
        // For sequences, upcast once at the top to avoid per-timestep dtype conversion.
        let original_dtype = input.dtype();
        if matches!(original_dtype, DType::BF16 | DType::F16) {
            let f32_input = input.to_dtype(DType::F32)?;
            let (output, final_state) = self.forward_seq(&f32_input, state)?;
            let output = output.to_dtype(original_dtype)?;
            let h = final_state.h.to_dtype(original_dtype)?;
            let c = final_state.c.to_dtype(original_dtype)?;
            return Ok((output, LstmState::new(h, c)?));
        }

        let use_gpu = input.device().is_gpu();
        let tracing = trace::is_tracing();

        // Try fused GPU sequence path first (#1805): single Metal dispatch for
        // the entire sequence, eliminating per-timestep commit_and_wait() barriers.
        // Skip during tracing (#2369): fused path returns untraced tensors.
        if use_gpu && !tracing {
            if let Some(result) = self.try_gpu_lstm_sequence(input, state) {
                return result;
            }
        }

        // Tracing path (#2224): record the full forward_seq as a single
        // composite LSTM trace op. Suppresses per-timestep trace recording
        // to avoid dangling h/c state Input nodes — per-cell forward()
        // would create new Input nodes for each timestep's state because
        // new_state.h/.c don't carry trace_ids (computed under suppression).
        // The translator decomposes the single LSTM into per-gate primitives
        // with zero-initialized state (conservative bounds).
        if tracing {
            // Compute the full sequence under trace suppression.
            // with_trace_suppressed makes is_tracing() return false,
            // so forward() won't record per-cell trace ops.
            let (mut stacked, final_state) = trace::with_trace_suppressed(|| {
                let mut current_state: Option<LstmState> = None;
                let mut outputs = Vec::with_capacity(seq_len);
                for t in 0..seq_len {
                    let x_t = input.narrow(0, t, 1)?.squeeze(0)?;
                    let st_ref = current_state.as_ref().or(state);
                    let (h_out, new_state) = self.forward(&x_t, st_ref)?;
                    outputs.push(h_out.unsqueeze(0)?);
                    current_state = Some(new_state);
                }
                let refs: Vec<&DynTensor> = outputs.iter().collect();
                let stacked = DynTensor::cat(&refs, 0)?;
                let fs = current_state.ok_or_else(|| {
                    TensorError::InvalidShape("LSTM forward_seq: no state after loop".into())
                })?;
                Ok::<_, TensorError>((stacked, fs))
            })?;

            // Record single composite LSTM op for the full sequence.
            // Only the data input is referenced — the translator creates
            // zero-initialized h/c state internally (inject_zero_state).
            if let Some(input_id) = input.trace_id() {
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
                        initial_hidden: None,
                        initial_cell: None,
                    },
                    &[input_id],
                    stacked.dims(),
                    stacked.dtype(),
                ) {
                    stacked.set_trace_id(id);
                }
            }
            return Ok((stacked, final_state));
        }

        // Non-tracing fallback: per-timestep loop.
        let mut current_state: Option<LstmState> = None;
        let mut outputs = Vec::with_capacity(seq_len);

        if use_gpu {
            // GPU per-timestep fallback (hidden_size > 512): skip per-step NaN
            // checks to prevent arena stale-read errors (#2328).
            //
            // Each check_output_finite() on GPU tensors calls flush(), which
            // resets the arena and advances its generation counter. After seq_len
            // iterations, accumulated generation gaps make the final state
            // tensors stale. Skipping per-step checks keeps all loop tensors in
            // one arena generation. Model-boundary validation (#941, #958) and
            // caller-level checks serve as the NaN/Inf backstop (#1939).
            with_nan_check_policy(NanCheckPolicy::Skip, || -> Result<()> {
                for t in 0..seq_len {
                    let x_t = input.narrow(0, t, 1)?.squeeze(0)?;
                    let st_ref = current_state.as_ref().or(state);
                    let (h_out, new_state) = self.forward(&x_t, st_ref)?;
                    outputs.push(h_out.unsqueeze(0)?);
                    current_state = Some(new_state);
                }
                Ok(())
            })?;
        } else {
            // CPU path: batch input-to-gate matmul outside the loop (#2679).
            // Before: seq_len tiny [1,inp]×[inp,4H] matmuls with high dispatch overhead.
            // After: 1 large [seq_len*batch,inp]×[inp,4H] matmul, then per-timestep
            // hidden matmul only. Hidden matmul remains sequential (h depends on t-1).
            let w_ih_t = self.w_ih.transpose(0, 1)?;
            let w_hh_t = self.w_hh.transpose(0, 1)?;

            let (s, b) = (dims[0], dims[1]);
            let four_h = 4 * self.hidden_size();
            let flat_input = input.reshape([s * b, dims[2]])?;
            let all_input_gates = flat_input.matmul(&w_ih_t)?;
            let all_input_gates = all_input_gates.reshape([s, b, four_h])?;

            // Add input bias once (not per-timestep).
            let all_input_gates = match &self.b_ih {
                Some(b_ih) => all_input_gates.broadcast_add(b_ih)?,
                None => all_input_gates,
            };

            for t in 0..seq_len {
                let input_gates_t = all_input_gates.narrow(0, t, 1)?.squeeze(0)?;
                let st_ref = current_state.as_ref().or(state);
                let (h_out, new_state) = self.forward_seq_step(&input_gates_t, st_ref, &w_hh_t)?;
                outputs.push(h_out.unsqueeze(0)?);
                current_state = Some(new_state);
            }
        }

        // Stack outputs along dim 0 -> [seq_len, batch, hidden_size]
        let output_refs: Vec<&DynTensor> = outputs.iter().collect();
        let stacked = DynTensor::cat(&output_refs, 0)?;

        let final_state = current_state.ok_or_else(|| {
            TensorError::InvalidShape("LSTM forward_seq: no state after loop (seq_len > 0)".into())
        })?;

        Ok((stacked, final_state))
    }

    /// LSTM step with pre-computed input gates (#2679).
    ///
    /// Used by `forward_seq` CPU path: input-to-gate matmul is batched outside
    /// the per-timestep loop, so this method only computes hidden-to-gate.
    ///
    /// `input_gates`: `[batch, 4*hidden_size]` — already `input @ w_ih^T + b_ih`
    fn forward_seq_step(
        &self,
        input_gates: &DynTensor,
        state: Option<&LstmState>,
        w_hh_t: &DynTensor,
    ) -> Result<(DynTensor, LstmState)> {
        let batch = input_gates.dims()[0];
        let device = input_gates.device();

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

        // gates = pre_computed_input_gates + h @ w_hh^T + b_hh
        let mut gates = input_gates.add(&h.matmul(w_hh_t)?)?;
        if let Some(ref b) = self.b_hh {
            gates = gates.broadcast_add(b)?;
        }

        // Split into i, f, g, o — same as forward_with_transposed
        let h_size = self.hidden_size;
        let i_gate = gates.narrow(1, 0, h_size)?.sigmoid()?;
        let f_gate = gates.narrow(1, h_size, h_size)?.sigmoid()?;
        let g_gate = gates.narrow(1, 2 * h_size, h_size)?.tanh()?;
        let o_gate = gates.narrow(1, 3 * h_size, h_size)?.sigmoid()?;

        let c_new = f_gate.mul(&c)?.add(&i_gate.mul(&g_gate)?)?;
        let h_new = o_gate.mul(&c_new.tanh()?)?;

        check_output_finite(&h_new, "LSTM h")?;
        check_output_finite(&c_new, "LSTM c")?;

        Ok((h_new.clone(), LstmState::new(h_new, c_new)?))
    }
}
