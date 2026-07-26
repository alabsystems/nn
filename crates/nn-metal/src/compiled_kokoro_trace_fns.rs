// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared trace functions for Kokoro pipeline segments.
//!
//! Each function traces a single pipeline segment and returns
//! `(primary_output, graph)` with multi-output marking applied.
//!
//! Used by both `compiled_kokoro_segments.rs` (runtime compilation) and
//! `compiled_kokoro_precompile_segments.rs` (ahead-of-time MSL).
//! Eliminates near-identical trace closures across both files.
//!
//! Part of #2218.

use std::cell::Cell;

use nn_core::dyn_tensor::trace::{trace_graph, ComputationGraph};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::{DType, Result, TensorError};
use nn_models::kokoro_error::KokoroError;
use nn_models::kokoro_source::{interp_downsample_gpu, interp_upsample_gpu};

use super::{cpu, model_device, set_last_output, trace_input, CompiledKokoro};

/// Trace segment 0: PlBert + bert_encoder → bert_features.
///
/// Pre-computes position/token-type embeddings, traces PlBert forward,
/// folds bert_encoder. Returns `(bert_features, graph)`.
pub(super) fn trace_seg_plbert(
    kokoro: &CompiledKokoro,
    input_ids: &DynTensor,
) -> Result<(DynTensor, ComputationGraph)> {
    let model = kokoro.shared.model()?;
    let plbert = model.plbert();
    let bert_encoder = model.bert_encoder();
    let seq_len = input_ids.dims()[1];
    let dev = model_device(Some(model));

    let seq_len_u32 = u32::try_from(seq_len).map_err(|_| TensorError::DimensionOverflow {
        dims: vec![seq_len],
    })?;
    let position_ids = DynTensor::arange_u32(0, seq_len_u32, &cpu())?.to_device(&dev)?;
    let pos_emb = plbert
        .position_embeddings()
        .forward(&position_ids)?
        .unsqueeze(0)?;

    let token_type_ids = DynTensor::zeros(&[seq_len], DType::U32, &cpu())?.to_device(&dev)?;
    let type_emb = plbert
        .token_type_embeddings()
        .forward(&token_type_ids)?
        .unsqueeze(0)?;

    let (_out, mut graph) = trace_graph(|| {
        let ids = trace_input(input_ids)?;
        let pe = trace_input(&pos_emb)?;
        let te = trace_input(&type_emb)?;
        let word_emb = plbert.word_embeddings().forward(&ids)?;
        let combined = word_emb.broadcast_add(&pe)?.broadcast_add(&te)?;
        let bert_output = plbert.forward_core(&combined)?;
        let bert_features = bert_encoder.forward(&bert_output)?.transpose(1, 2)?;
        Ok(bert_features)
    })?;

    set_last_output(&mut graph)?;
    Ok((_out, graph))
}

/// Trace segment 1: TextEncoder (token IDs → features).
pub(super) fn trace_seg_text(
    kokoro: &CompiledKokoro,
    input_ids: &DynTensor,
) -> Result<(DynTensor, ComputationGraph)> {
    let text_enc = kokoro.shared.model()?.text_encoder();

    let (_out, mut graph) = trace_graph(|| {
        let ids = trace_input(input_ids)?;
        text_enc
            .forward(&ids)
            .map_err(KokoroError::into_tensor_error)
    })?;

    set_last_output(&mut graph)?;
    Ok((_out, graph))
}

/// Trace segment 2: ProsodyPredictor (multi-output: dur, features).
///
/// Returns `(dur_output, graph)` with both outputs marked.
pub(super) fn trace_seg_prosody(
    kokoro: &CompiledKokoro,
    bert_features: &DynTensor,
    style: &DynTensor,
) -> Result<(DynTensor, ComputationGraph)> {
    let prosody = kokoro.shared.model()?.prosody_predictor();
    let feat_id: Cell<Option<u64>> = Cell::new(None);

    let (dur_out, mut graph) = trace_graph(|| {
        let inp = trace_input(bert_features)?;
        let sty = trace_input(style)?;
        let (dur, feat) = prosody
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        feat_id.set(feat.trace_id());
        Ok(dur)
    })?;

    mark_multi_output(&mut graph, dur_out.trace_id(), feat_id.get(), "prosody")?;
    Ok((dur_out, graph))
}

/// Trace segment 3: F0EnergyPredictor (multi-output: f0, energy).
///
/// Returns `(f0_output, graph)` with both outputs marked.
pub(super) fn trace_seg_f0(
    kokoro: &CompiledKokoro,
    aligned: &DynTensor,
    style: &DynTensor,
) -> Result<(DynTensor, ComputationGraph)> {
    let f0_pred = kokoro.shared.model()?.f0_predictor();
    let energy_id: Cell<Option<u64>> = Cell::new(None);

    let (f0_out, mut graph) = trace_graph(|| {
        let inp = trace_input(aligned)?;
        let sty = trace_input(style)?;
        let (f0, en) = f0_pred
            .forward(&inp, &sty)
            .map_err(KokoroError::into_tensor_error)?;
        energy_id.set(en.trace_id());
        Ok(f0)
    })?;

    mark_multi_output(&mut graph, f0_out.trace_id(), energy_id.get(), "f0")?;
    Ok((f0_out, graph))
}

/// Trace segment 4: Generator (multi-output: magnitude, phase).
///
/// Returns `(magnitude, graph)` with both outputs marked.
pub(super) fn trace_seg_generator(
    kokoro: &CompiledKokoro,
    regulated: &DynTensor,
    f0: &DynTensor,
    energy: &DynTensor,
    decoder_style: &DynTensor,
    har_source: &DynTensor,
) -> Result<(DynTensor, ComputationGraph)> {
    let generator = kokoro.shared.model()?.decoder();
    let phase_id: Cell<Option<u64>> = Cell::new(None);

    let (mag_out, mut graph) = trace_graph(|| {
        let inp = trace_input(regulated)?;
        let f0_t = trace_input(f0)?;
        let en_t = trace_input(energy)?;
        let sty = trace_input(decoder_style)?;
        let h = trace_input(har_source)?;
        let (mag, phase) = generator.forward(&inp, &f0_t, &en_t, &sty, &h)?;
        phase_id.set(phase.trace_id());
        Ok(mag)
    })?;

    mark_multi_output(&mut graph, mag_out.trace_id(), phase_id.get(), "generator")?;
    Ok((mag_out, graph))
}

/// Trace segment 5: Regulate pre-readback elementwise chain.
///
/// Pure elementwise ops — no model weights. Speed is a runtime input
/// (not baked as constant) so the segment doesn't need recompilation
/// when speed changes.
///
/// Returns `(counts_gpu, graph)` with both outputs marked:
/// - Primary: `counts_gpu` `[T]` — integer frame counts for prefix_sum.
/// - Secondary: `durations` `[B, T]` — float durations for return value.
///
/// Part of #1815 Tier 6 D2b.
pub(super) fn trace_seg_regulate(
    dur_logits: &DynTensor,
    speed_inv: &DynTensor,
    max_dur: f64,
) -> Result<(DynTensor, ComputationGraph)> {
    let dur_id: Cell<Option<u64>> = Cell::new(None);

    let (counts_out, mut graph) = trace_graph(|| {
        let logits = trace_input(dur_logits)?;
        let si = trace_input(speed_inv)?;

        // sigmoid(logits).sum(last_dim) * (1/speed), clamped to [1, max_dur].
        let durations = logits
            .sigmoid()?
            .sum(2)?
            .broadcast_mul(&si)?
            .clamp(1.0, max_dur)?;

        dur_id.set(durations.trace_id());

        // Round to integer counts: add(0.5) + floor, clamp_min(1).
        let counts = durations
            .squeeze(0)?
            .add_scalar(0.5)?
            .floor()?
            .clamp_min(1.0)?;

        Ok(counts)
    })?;

    mark_multi_output(&mut graph, counts_out.trace_id(), dur_id.get(), "regulate")?;
    Ok((counts_out, graph))
}

/// Trace segment 5a: SineGen pre-cumsum (single-output: rad_frames).
///
/// Traces steps 1-4 of SineGen. The cumsum (step 5) is eager because
/// `cumsum_kahan` is a custom Metal kernel not in TraceOp.
///
/// Voiced mask is computed eagerly in `build_harmonic_source` because the
/// `.gt()` compare op is not yet supported by the trace compiler (#3213).
///
/// Returns `(rad_frames, graph)`:
/// - `rad_frames` `[B, T_frames, n_ch]` — fractional phase at frame rate.
///
/// U32 index tensors from `interp_downsample_gpu` are auto-registered as
/// ConstantWeight with f32 data via `to_weight_ref()` U32 path. The compiled
/// pipeline converts f32→u32 at dispatch time (lossless for indices < 2^24).
///
/// Part of #1815 Tier 6 D2.
pub(super) fn trace_seg_sinegen_pre(
    f0: &DynTensor,
    upp: usize,
    sr: f32,
    n_ch: usize,
) -> Result<(DynTensor, ComputationGraph)> {
    let batch = f0.dim(0)?;
    let t_frames = f0.dim(1)?;
    let t_audio = t_frames * upp;
    let device = f0.device();

    let (rad_out, mut graph) = trace_graph(|| {
        let f0_in = trace_input(f0)?;

        // Steps 1-2: upsample F0 to audio rate, expand to harmonics.
        let f0_audio = f0_in
            .unsqueeze(2)?
            .expand([batch, t_frames, upp, 1])?
            .reshape([batch, t_audio, 1])?;
        let harmonics_data: Vec<f32> = (1..=n_ch).map(|h| h as f32).collect();
        let harmonics = DynTensor::from_vec(harmonics_data, &[1, 1, n_ch], &device)?;
        let freq = f0_audio.broadcast_mul(&harmonics)?;

        // Step 3: normalize and fract.
        let rad_audio = freq.mul_scalar(1.0 / f64::from(sr))?.fract()?;

        // Step 4: downsample to frame rate via interp helper.
        // CPU loop (index computation) is not traced. U32 index tensors
        // auto-register as ConstantWeight with f32 data. Traced DynTensor ops
        // (index_select, broadcast_mul, broadcast_add) form the compiled graph.
        let rad_frames = interp_downsample_gpu(&rad_audio, t_frames)?;

        Ok(rad_frames)
    })?;

    set_last_output(&mut graph)?;
    Ok((rad_out, graph))
}

/// Trace segment 5b: SineGen post-cumsum (single-output: excitation).
///
/// Traces steps 6-8 of SineGen + SourceModule (linear → tanh).
/// Input is the cumulative sum from the eager `cumsum_kahan` step.
///
/// Returns `(excitation, graph)` — single output `[B, T_audio, 1]`.
///
/// Part of #1815 Tier 6 D3.
pub(super) fn trace_seg_sinegen_post(
    cum_gpu: &DynTensor,
    f0_gpu: &DynTensor,
    l_linear: &Linear,
    upp: usize,
    sine_amp: f32,
    voiced_threshold: f64,
) -> Result<(DynTensor, ComputationGraph)> {
    let t_frames = cum_gpu.dim(1)?;
    let batch = cum_gpu.dim(0)?;
    let t_audio = t_frames * upp;

    let (excitation, mut graph) = trace_graph(|| {
        let cum_in = trace_input(cum_gpu)?;
        let f0_in = trace_input(f0_gpu)?;

        // Voiced mask: fold from eager into compiled segment.
        // f0_gpu is [B, T_frames, 1] → expand to [B, T_audio, 1] → gt(threshold).
        // Previously computed eagerly (4+ dispatches). Now traced and compiled.
        let voiced = f0_in
            .unsqueeze(2)?
            .expand([batch, t_frames, upp, 1])?
            .reshape([batch, t_audio, 1])?
            .gt(voiced_threshold)?
            .to_dtype(DType::F32)?;

        // Step 6: scale by 2π × upp.
        let phase_frames = cum_in.mul_scalar(std::f64::consts::TAU * upp as f64)?;

        // Step 7: upsample phase to audio rate via interp helper.
        // Same pattern as D2: CPU loop not traced, U32 indices captured
        // as f32 ConstantWeight, DynTensor ops form the compiled graph.
        let phase_audio = interp_upsample_gpu(&phase_frames, t_audio)?;

        // Step 8: sin(phase) × sine_amp.
        let sines = phase_audio.sin()?.mul_scalar(f64::from(sine_amp))?;

        // SourceModule: sines × voiced → linear → tanh → transpose.
        // Transpose [B, T_audio, 1] → [B, 1, T_audio] is folded into the
        // compiled segment, saving 1 eager dispatch in build_harmonic_source.
        let sine_wavs = sines.broadcast_mul(&voiced)?;
        let projected = l_linear.forward(&sine_wavs)?;
        projected.tanh()?.transpose(1, 2)
    })?;

    set_last_output(&mut graph)?;
    Ok((excitation, graph))
}

/// Mark primary + secondary output on a multi-output computation graph.
fn mark_multi_output(
    graph: &mut ComputationGraph,
    primary_id: Option<u64>,
    secondary_id: Option<u64>,
    segment: &str,
) -> Result<()> {
    if let Some(id) = primary_id {
        if !graph.set_primary_output(id) {
            return Err(TensorError::InvalidShape(format!(
                "trace bug: {segment} primary output node not found in graph"
            )));
        }
    }
    if let Some(id) = secondary_id {
        if !graph.mark_output(id) {
            return Err(TensorError::InvalidShape(format!(
                "trace bug: {segment} secondary output node not found in graph"
            )));
        }
    }
    Ok(())
}
