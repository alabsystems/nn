// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU forward pass and dispatch helpers for `DemucsTransformer`.
//!
//! Extracted from `demucs_transformer.rs` (#833) to keep files under the
//! 500-line limit. Contains the CPU inference-time methods: `forward()`,
//! `dispatch_single()`, `dispatch_cross()`.
//!
//! GPU-resident forward pass (`forward_gpu()`) and GPU dispatch helpers are
//! in `demucs_transformer_forward_gpu.rs`.

use std::collections::HashMap;

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::ScalarType;

use crate::tensor_dispatch::execute_tensor_dispatch;
use crate::PipelineCache;

use nn_models::TransformerBuildError;

use super::{
    helpers, DemucsTransformer, DemucsTransformerError, BOTTLENECK_DIM, NUM_LAYERS, TRANSFORMER_DIM,
};

#[path = "demucs_transformer_forward_gpu.rs"]
mod gpu_forward;

impl DemucsTransformer {
    /// Run the cross-domain transformer forward pass.
    ///
    /// `temporal`: flattened `[BOTTLENECK_DIM, T]` — temporal encoder output.
    /// `spectral`: flattened `[BOTTLENECK_DIM, seq_s]` — spectral encoder output
    ///   (where `seq_s = F * T` after flattening spatial dims).
    ///
    /// Returns `(temporal_out, spectral_out)` both at `[BOTTLENECK_DIM, ...]`.
    pub fn forward(
        &self,
        cache: &PipelineCache,
        temporal: &[f32],
        spectral: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), DemucsTransformerError> {
        let expected_t = BOTTLENECK_DIM * self.temporal_seq_len;
        if temporal.len() != expected_t {
            return Err(TransformerBuildError::DimMismatch {
                stage: "temporal input".to_string(),
                expected: expected_t,
                actual: temporal.len(),
            }
            .into());
        }
        let expected_s = BOTTLENECK_DIM * self.spectral_seq_len;
        if spectral.len() != expected_s {
            return Err(TransformerBuildError::DimMismatch {
                stage: "spectral input".to_string(),
                expected: expected_s,
                actual: spectral.len(),
            }
            .into());
        }

        // Step 1: Channel upsample (384 → 512).
        let mut t_data = self.dispatch_single(
            cache,
            &self.upsample_t_def,
            &self.upsample_t_weights,
            temporal,
        )?;
        let mut s_data = self.dispatch_single(
            cache,
            &self.upsample_s_def,
            &self.upsample_s_weights,
            spectral,
        )?;

        // Step 2: Transpose [C, T] → [T, C] (CPU-side for both branches).
        t_data = helpers::transpose_ct_to_tc(&t_data, TRANSFORMER_DIM, self.temporal_seq_len);
        s_data = helpers::transpose_ct_to_tc(&s_data, TRANSFORMER_DIM, self.spectral_seq_len);

        // Step 3: Input LayerNorm (on [T, D] sequences).
        t_data =
            self.dispatch_single(cache, &self.norm_in_t_def, &self.norm_in_t_weights, &t_data)?;
        s_data =
            self.dispatch_single(cache, &self.norm_in_s_def, &self.norm_in_s_weights, &s_data)?;

        // Step 4: Add sinusoidal positional embeddings (CPU-side).
        helpers::add_sinusoidal_1d(&mut t_data, self.temporal_seq_len, TRANSFORMER_DIM);
        helpers::add_sinusoidal_1d(&mut s_data, self.spectral_seq_len, TRANSFORMER_DIM);

        // Step 5: 5 transformer layers (alternating self/cross).
        for i in 0..NUM_LAYERS {
            // Validate finiteness (single-pass count, no redundant .any() scan).
            crate::check_non_finite_err(&t_data, |count| DemucsTransformerError::NonFiniteInput {
                layer: i,
                count,
            })?;
            crate::check_non_finite_err(&s_data, |count| DemucsTransformerError::NonFiniteInput {
                layer: i,
                count,
            })?;

            let is_cross = i % 2 == 1;

            if is_cross {
                // Cross-attention: both branches read previous iteration's outputs.
                let t_old = t_data.clone();
                let s_old = s_data.clone();

                // Spectral attends to temporal: Q from s_old, KV from t_old.
                s_data = self.dispatch_cross(
                    cache,
                    &self.spectral_layer_defs[i],
                    &self.spectral_layer_weights[i],
                    &s_old,
                    &t_old,
                )?;

                // Temporal attends to spectral: Q from t_old, KV from s_old.
                t_data = self.dispatch_cross(
                    cache,
                    &self.temporal_layer_defs[i],
                    &self.temporal_layer_weights[i],
                    &t_old,
                    &s_old,
                )?;
            } else {
                // Self-attention: each branch processes independently.
                t_data = self.dispatch_single(
                    cache,
                    &self.temporal_layer_defs[i],
                    &self.temporal_layer_weights[i],
                    &t_data,
                )?;
                s_data = self.dispatch_single(
                    cache,
                    &self.spectral_layer_defs[i],
                    &self.spectral_layer_weights[i],
                    &s_data,
                )?;
            }
        }

        // Step 6: Transpose back [T, C] → [C, T].
        t_data = helpers::transpose_tc_to_ct(&t_data, self.temporal_seq_len, TRANSFORMER_DIM);
        s_data = helpers::transpose_tc_to_ct(&s_data, self.spectral_seq_len, TRANSFORMER_DIM);

        // Step 7: Channel downsample (512 → 384).
        t_data = self.dispatch_single(
            cache,
            &self.downsample_t_def,
            &self.downsample_t_weights,
            &t_data,
        )?;
        s_data = self.dispatch_single(
            cache,
            &self.downsample_s_def,
            &self.downsample_s_weights,
            &s_data,
        )?;

        Ok((t_data, s_data))
    }

    /// Dispatch a def with a single "data" input.
    fn dispatch_single(
        &self,
        cache: &PipelineCache,
        def: &TensorKernelDef,
        weight_map: &HashMap<String, Vec<f32>>,
        data: &[f32],
    ) -> Result<Vec<f32>, DemucsTransformerError> {
        let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
        inputs.insert(nn_dsl::input_names::DATA, data);
        for (name, w) in weight_map {
            inputs.insert(name.as_str(), w.as_slice());
        }
        Ok(execute_tensor_dispatch(
            cache,
            def,
            ScalarType::F32,
            &inputs,
        )?)
    }

    /// Dispatch a cross-attention def with "data" (Q source) and "cross" (KV source) inputs.
    fn dispatch_cross(
        &self,
        cache: &PipelineCache,
        def: &TensorKernelDef,
        weight_map: &HashMap<String, Vec<f32>>,
        data: &[f32],
        cross: &[f32],
    ) -> Result<Vec<f32>, DemucsTransformerError> {
        let mut inputs: HashMap<&str, &[f32]> = HashMap::new();
        inputs.insert(nn_dsl::input_names::DATA, data);
        inputs.insert("cross", cross);
        for (name, w) in weight_map {
            inputs.insert(name.as_str(), w.as_slice());
        }
        Ok(execute_tensor_dispatch(
            cache,
            def,
            ScalarType::F32,
            &inputs,
        )?)
    }
}
