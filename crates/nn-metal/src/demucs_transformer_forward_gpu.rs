// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident forward pass and dispatch helpers for `DemucsTransformer`.
//!
//! Extracted from `demucs_transformer_forward.rs` to keep files under 400 lines.
//! `forward_gpu()` (#1372) keeps all intermediates on Metal GPU buffers,
//! eliminating 15 of 16 CPU round-trips. GPU transpose (Steps 2, 6),
//! GPU sinusoidal embedding (Step 4), and GPU downsample (Step 7) are
//! all buffer-to-buffer. Only the final output reads back to CPU.

use std::collections::HashMap;

use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::ScalarType;

use crate::buffer::MetalBuffer;
use crate::element::MetalElement;
use crate::gpu_scope::with_gpu_scope;
use crate::gpu_slice::GpuSlice;
use crate::tensor_dispatch::{execute_tensor_dispatch_to_buffer, output_elems, DispatchInput};
use crate::PipelineCache;

use nn_models::TransformerBuildError;

use super::super::{DemucsTransformer, DemucsTransformerError, BOTTLENECK_DIM, NUM_LAYERS};

/// Run a closure under [`with_gpu_scope`], converting errors through
/// `DemucsTransformerError` → `TensorError` for the scope boundary, then
/// mapping any scope-infrastructure error back to `DemucsTransformerError`.
///
/// All GPU dispatches inside `f` share one `CommandBatch` — a single
/// `commit_and_wait` at scope exit replaces N per-dispatch barriers.
fn scoped_dispatch<T>(
    f: impl FnOnce() -> Result<T, DemucsTransformerError>,
) -> Result<T, DemucsTransformerError> {
    with_gpu_scope(|| f().map_err(Into::into)).map_err(|te| {
        DemucsTransformerError::Dispatch(crate::tensor_dispatch::TensorDispatchError::Metal(
            crate::error::MetalError::DispatchFailed(format!("gpu_scope: {te}")),
        ))
    })
}

impl DemucsTransformer {
    /// GPU-optimized forward pass — keeps all intermediates on Metal GPU buffers.
    ///
    /// Same semantics as [`forward`](Self::forward), but every step (upsample,
    /// transpose, norm, sinusoidal embedding, transformer layers, downsample) is
    /// GPU buffer-to-buffer. Only the final output reads back to CPU.
    ///
    /// NaN/Inf safety checks at transformer layers 0, 2, 4 require temporary
    /// CPU readback (6 total: 3 layers × 2 branches).
    ///
    /// Uses `with_gpu_scope` (#1854) to batch GPU dispatches into fewer
    /// `commit_and_wait` barriers. Dispatch-only segments share a single
    /// `CommandBatch`; NaN checks between scopes force commits so that
    /// `read_buffer` sees committed data. Reduces ~19 barriers to ~8.
    pub fn forward_gpu(
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

        // Upload all weights to GPU on first call; subsequent calls use cached buffers.
        let gw = self.ensure_gpu_weights(cache)?;
        let empty_gpu_weights: HashMap<String, MetalBuffer> = HashMap::new();

        // ── Scope 1: Steps 1–4 (upsample, transpose, norm, sinusoidal) ──
        // 8 dispatches batched into 1 commit_and_wait.
        let (mut t_slice, mut s_slice) = scoped_dispatch(|| {
            let t_slice = self.dispatch_single_gpu_cpu(
                cache,
                &self.upsample_t_def,
                &gw.upsample_t,
                temporal,
            )?;
            let s_slice = self.dispatch_single_gpu_cpu(
                cache,
                &self.upsample_s_def,
                &gw.upsample_s,
                spectral,
            )?;
            let t_slice = self.dispatch_single_gpu(
                cache,
                &self.transpose_ct_tc_t_def,
                &empty_gpu_weights,
                &t_slice,
            )?;
            let s_slice = self.dispatch_single_gpu(
                cache,
                &self.transpose_ct_tc_s_def,
                &empty_gpu_weights,
                &s_slice,
            )?;
            let t_slice =
                self.dispatch_single_gpu(cache, &self.norm_in_t_def, &gw.norm_in_t, &t_slice)?;
            let s_slice =
                self.dispatch_single_gpu(cache, &self.norm_in_s_def, &gw.norm_in_s, &s_slice)?;
            let t_slice = self.dispatch_single_gpu(
                cache,
                &self.sinusoidal_t_def,
                &gw.sinusoidal_t,
                &t_slice,
            )?;
            let s_slice = self.dispatch_single_gpu(
                cache,
                &self.sinusoidal_s_def,
                &gw.sinusoidal_s,
                &s_slice,
            )?;
            Ok((t_slice, s_slice))
        })?;

        // ── Step 5: 5 transformer layers with NaN checks between scopes ──
        for i in 0..NUM_LAYERS {
            // NaN/Inf check requires CPU readback — only check every other layer
            // to balance safety vs performance. Layer 0 catches bad input from
            // positional embedding; layers 2, 4 catch propagation.
            // Runs outside any scope so read_buffer sees committed data.
            if i % 2 == 0 {
                // Determine element count from the def that produced the current slices.
                // Layer 0: sinusoidal defs; layers 2,4: previous layer's defs.
                let (t_def, s_def) = if i == 0 {
                    (&self.sinusoidal_t_def, &self.sinusoidal_s_def)
                } else {
                    (
                        &self.temporal_layer_defs[i - 1],
                        &self.spectral_layer_defs[i - 1],
                    )
                };
                let t_elems = output_elems(t_def, t_def.output)?;
                let s_elems = output_elems(s_def, s_def.output)?;
                let t_data =
                    f32::read_buffer_at_offset(t_slice.buffer(), t_slice.byte_offset(), t_elems)
                        .map_err(crate::tensor_dispatch::TensorDispatchError::from)?;
                let s_data =
                    f32::read_buffer_at_offset(s_slice.buffer(), s_slice.byte_offset(), s_elems)
                        .map_err(crate::tensor_dispatch::TensorDispatchError::from)?;
                crate::check_non_finite_err(&t_data, |count| {
                    DemucsTransformerError::NonFiniteInput { layer: i, count }
                })?;
                crate::check_non_finite_err(&s_data, |count| {
                    DemucsTransformerError::NonFiniteInput { layer: i, count }
                })?;
            }

            let is_cross = i % 2 == 1;

            // Each layer's dispatches (2 or 4) batched into 1 commit_and_wait.
            let (new_t, new_s) = scoped_dispatch(|| {
                if is_cross {
                    let t_old = t_slice.alias();
                    let s_old = s_slice.alias();
                    let new_s = self.dispatch_cross_gpu(
                        cache,
                        &self.spectral_layer_defs[i],
                        &gw.spectral_layers[i],
                        &s_old,
                        &t_old,
                    )?;
                    let new_t = self.dispatch_cross_gpu(
                        cache,
                        &self.temporal_layer_defs[i],
                        &gw.temporal_layers[i],
                        &t_old,
                        &s_old,
                    )?;
                    Ok((new_t, new_s))
                } else {
                    let new_t = self.dispatch_single_gpu(
                        cache,
                        &self.temporal_layer_defs[i],
                        &gw.temporal_layers[i],
                        &t_slice,
                    )?;
                    let new_s = self.dispatch_single_gpu(
                        cache,
                        &self.spectral_layer_defs[i],
                        &gw.spectral_layers[i],
                        &s_slice,
                    )?;
                    Ok((new_t, new_s))
                }
            })?;
            t_slice = new_t;
            s_slice = new_s;
        }

        // ── Scope 7: Steps 6–7 (transpose + downsample) ──
        // 4 dispatches batched into 1 commit_and_wait.
        let (t_slice, s_slice) = scoped_dispatch(|| {
            let t_slice = self.dispatch_single_gpu(
                cache,
                &self.transpose_tc_ct_t_def,
                &empty_gpu_weights,
                &t_slice,
            )?;
            let s_slice = self.dispatch_single_gpu(
                cache,
                &self.transpose_tc_ct_s_def,
                &empty_gpu_weights,
                &s_slice,
            )?;
            let t_slice = self.dispatch_single_gpu(
                cache,
                &self.downsample_t_def,
                &gw.downsample_t,
                &t_slice,
            )?;
            let s_slice = self.dispatch_single_gpu(
                cache,
                &self.downsample_s_def,
                &gw.downsample_s,
                &s_slice,
            )?;
            Ok((t_slice, s_slice))
        })?;

        // Final CPU readback — the only round-trip in the entire forward pass.
        // Use offset-aware readback for arena-allocated buffers (#2207).
        let t_elems = output_elems(&self.downsample_t_def, self.downsample_t_def.output)?;
        let s_elems = output_elems(&self.downsample_s_def, self.downsample_s_def.output)?;
        let t_data = f32::read_buffer_at_offset(t_slice.buffer(), t_slice.byte_offset(), t_elems)
            .map_err(crate::tensor_dispatch::TensorDispatchError::from)?;
        let s_data = f32::read_buffer_at_offset(s_slice.buffer(), s_slice.byte_offset(), s_elems)
            .map_err(crate::tensor_dispatch::TensorDispatchError::from)?;

        Ok((t_data, s_data))
    }

    /// Dispatch a def with CPU "data" input and persistent GPU weights, returning a GpuSlice.
    fn dispatch_single_gpu_cpu(
        &self,
        cache: &PipelineCache,
        def: &TensorKernelDef,
        gpu_weight_map: &HashMap<String, MetalBuffer>,
        data: &[f32],
    ) -> Result<GpuSlice, DemucsTransformerError> {
        let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
        inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Cpu(data));
        for (name, buf) in gpu_weight_map {
            inputs.insert(
                name.as_str(),
                DispatchInput::Gpu(GpuSlice::from_ref(buf, 0)),
            );
        }
        Ok(execute_tensor_dispatch_to_buffer::<f32>(
            cache,
            def,
            ScalarType::F32,
            &inputs,
        )?)
    }

    /// Dispatch a def with a GpuSlice "data" input and persistent GPU weights, returning a GpuSlice.
    fn dispatch_single_gpu(
        &self,
        cache: &PipelineCache,
        def: &TensorKernelDef,
        gpu_weight_map: &HashMap<String, MetalBuffer>,
        data: &GpuSlice,
    ) -> Result<GpuSlice, DemucsTransformerError> {
        let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
        inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Gpu(data.alias()));
        for (name, buf) in gpu_weight_map {
            inputs.insert(
                name.as_str(),
                DispatchInput::Gpu(GpuSlice::from_ref(buf, 0)),
            );
        }
        Ok(execute_tensor_dispatch_to_buffer::<f32>(
            cache,
            def,
            ScalarType::F32,
            &inputs,
        )?)
    }

    /// Dispatch a cross-attention def with GpuSlice inputs and persistent GPU weights, returning a GpuSlice.
    fn dispatch_cross_gpu(
        &self,
        cache: &PipelineCache,
        def: &TensorKernelDef,
        gpu_weight_map: &HashMap<String, MetalBuffer>,
        data: &GpuSlice,
        cross: &GpuSlice,
    ) -> Result<GpuSlice, DemucsTransformerError> {
        let mut inputs: HashMap<&str, DispatchInput<'_, f32>> = HashMap::new();
        inputs.insert(nn_dsl::input_names::DATA, DispatchInput::Gpu(data.alias()));
        inputs.insert("cross", DispatchInput::Gpu(cross.alias()));
        for (name, buf) in gpu_weight_map {
            inputs.insert(
                name.as_str(),
                DispatchInput::Gpu(GpuSlice::from_ref(buf, 0)),
            );
        }
        Ok(execute_tensor_dispatch_to_buffer::<f32>(
            cache,
            def,
            ScalarType::F32,
            &inputs,
        )?)
    }
}
