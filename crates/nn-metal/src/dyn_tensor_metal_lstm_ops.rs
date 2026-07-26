// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused GPU LSTM cell kernel for [`MetalDynBackend`].
//!
//! Replaces the ~18-dispatch decomposed path (2 matmul + 2 broadcast_add +
//! 4 narrow + 3 sigmoid + 1 tanh + 3 mul + 2 add + 1 tanh) with a single
//! dispatch graph via [`build_lstm_cell_decomposed_dual`].
//!
//! Output is `(h_new, c_new)` each `[batch, hidden_size]`, split from the
//! internal `[2, batch, hidden_size]` stacked output.
//!
//! Part of #1373 (fused GPU LSTM cell).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError};

use nn_dsl::input_names;

use super::MetalTensorData;

impl super::MetalDynBackend {
    /// Fused LSTM cell GPU kernel.
    ///
    /// Builds a single dispatch graph containing all LSTM gate operations
    /// (2 linear + 4 narrow + 3 sigmoid + 1 tanh + 3 mul + 2 add + 1 tanh
    /// + stack), then dispatches in one GPU command.
    ///
    /// # Arguments
    /// - `input`: `[batch, input_size]`
    /// - `hidden`: `[batch, hidden_size]`
    /// - `cell`: `[batch, hidden_size]`
    /// - `w_ih`: `[4*hidden_size, input_size]` (PyTorch layout, non-transposed)
    /// - `w_hh`: `[4*hidden_size, hidden_size]` (PyTorch layout, non-transposed)
    /// - `bias`: `[4*hidden_size]` (optional combined b_ih + b_hh)
    /// - `hidden_size`: hidden dimension
    ///
    /// # Returns
    /// `(h_new, c_new)` each `[batch, hidden_size]`.
    pub(super) fn gpu_lstm_cell(
        input: &DynTensor,
        hidden: &DynTensor,
        cell: &DynTensor,
        w_ih: &DynTensor,
        w_hh: &DynTensor,
        bias: Option<&DynTensor>,
        hidden_size: usize,
    ) -> Result<(DynTensor, DynTensor)> {
        Self::validate_same_float_dtype(input, hidden, "gpu_lstm_cell")?;
        Self::validate_same_float_dtype(input, cell, "gpu_lstm_cell")?;
        Self::validate_same_float_dtype(input, w_ih, "gpu_lstm_cell")?;
        Self::validate_same_float_dtype(input, w_hh, "gpu_lstm_cell")?;
        if let Some(b) = bias {
            Self::validate_same_float_dtype(input, b, "gpu_lstm_cell")?;
        }

        // Validate weight finiteness before kernel launch.
        // GPU fused kernel applies sigmoid(Inf)=1.0 and tanh(Inf)=1.0,
        // which silently absorbs Inf values making output appear finite.
        // CPU path catches this via check_output_finite on gate values.
        // Defense-in-depth: reject non-finite weights before dispatch.
        if w_ih.any_non_finite()? {
            return Err(TensorError::NonFiniteData {
                name: "gpu_lstm_cell: w_ih".into(),
                count: 1, // at least 1; exact count requires CPU readback
            });
        }
        if w_hh.any_non_finite()? {
            return Err(TensorError::NonFiniteData {
                name: "gpu_lstm_cell: w_hh".into(),
                count: 1,
            });
        }
        if let Some(b) = bias {
            if b.any_non_finite()? {
                return Err(TensorError::NonFiniteData {
                    name: "gpu_lstm_cell: bias".into(),
                    count: 1,
                });
            }
        }

        // Validate shapes.
        let input_dims = input.dims();
        if input_dims.len() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: input_dims.len(),
            });
        }
        let batch = input_dims[0];
        let input_size = input_dims[1];

        let h_dims = hidden.dims();
        if h_dims != [batch, hidden_size] {
            return Err(TensorError::shape_mismatch(
                vec![batch, hidden_size],
                h_dims.to_vec(),
            ));
        }
        let c_dims = cell.dims();
        if c_dims != [batch, hidden_size] {
            return Err(TensorError::shape_mismatch(
                vec![batch, hidden_size],
                c_dims.to_vec(),
            ));
        }

        let four_h = 4 * hidden_size;
        let wih_dims = w_ih.dims();
        if wih_dims != [four_h, input_size] {
            return Err(TensorError::shape_mismatch(
                vec![four_h, input_size],
                wih_dims.to_vec(),
            ));
        }
        let whh_dims = w_hh.dims();
        if whh_dims != [four_h, hidden_size] {
            return Err(TensorError::shape_mismatch(
                vec![four_h, hidden_size],
                whh_dims.to_vec(),
            ));
        }
        if let Some(b) = bias {
            let b_dims = b.dims();
            if b_dims != [four_h] {
                return Err(TensorError::shape_mismatch(vec![four_h], b_dims.to_vec()));
            }
        }

        // Extract GPU buffer handles.
        let input_data = input.gpu_data::<MetalTensorData>()?;
        let hidden_data = hidden.gpu_data::<MetalTensorData>()?;
        let cell_data = cell.gpu_data::<MetalTensorData>()?;
        let wih_data = w_ih.gpu_data::<MetalTensorData>()?;
        let whh_data = w_hh.gpu_data::<MetalTensorData>()?;

        // Build fused LSTM dispatch graph via the existing decomposed builder,
        // using KernelDefCache to avoid rebuilding on every call.
        let has_bias = bias.is_some();
        let def = crate::kernel_def_cache::get_or_build(
            "lstm_cell",
            &[input.dims(), hidden.dims(), w_ih.dims(), w_hh.dims()],
            &[u64::from(has_bias)],
            input.dtype(),
            || {
                let d = nn_dsl::build_lstm_cell_decomposed_dual(
                    input_size,
                    hidden_size,
                    batch,
                    has_bias,
                )
                .map_err(|e| TensorError::InvalidShape(format!("lstm kernel build: {e}")))?;
                Ok(d)
            },
        )?;

        // Map GPU buffers to the builder's input names.
        let mut inputs: Vec<(&str, crate::gpu_slice::GpuSlice)> = vec![
            (input_names::DATA, input_data.as_gpu_slice()),
            (input_names::HIDDEN_STATE, hidden_data.as_gpu_slice()),
            (input_names::CELL_STATE, cell_data.as_gpu_slice()),
            (input_names::WEIGHT_IH, wih_data.as_gpu_slice()),
            (input_names::WEIGHT_HH, whh_data.as_gpu_slice()),
        ];

        let bias_data;
        if let Some(b) = bias {
            bias_data = b.gpu_data::<MetalTensorData>()?;
            inputs.push((input_names::BIAS, bias_data.as_gpu_slice()));
        }

        // Output shape: [2, batch, hidden_size] from the builder.
        // Axis-0 stacking means dim-0 narrow is always zero-copy (contiguous
        // byte region) regardless of batch size — no GPU kernel dispatch.
        let out_shape = &[2, batch, hidden_size];

        let stacked = Self::dispatch_def(&def, &inputs, out_shape, input.dtype())?;

        // Dim-0 narrow: zero-copy byte-offset view via MetalBuffer::alias().
        // narrow(0, 0, 1) → [1, batch, H] → reshape → [batch, H]
        // narrow(0, 1, 1) → [1, batch, H] → reshape → [batch, H]
        let h_new = stacked.narrow(0, 0, 1)?.reshape([batch, hidden_size])?;
        let c_new = stacked.narrow(0, 1, 1)?.reshape([batch, hidden_size])?;
        Ok((h_new, c_new))
    }
}
