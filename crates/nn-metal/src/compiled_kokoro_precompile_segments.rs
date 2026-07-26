// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Per-segment precompile trace functions for Kokoro MSL pre-compilation.
//!
//! Each function creates dummy inputs at representative shapes, delegates to
//! [`trace_fns`] for the trace closure, then calls `super::export_plan_msl()`
//! to compile and write MSL.
//!
//! Extracted from `compiled_kokoro_precompile.rs` (code structure wave 9b).
//! Trace dedup: #2218.

use nn_core::dyn_tensor::DynTensor;
use nn_core::DType;

use super::super::{model_device, CompiledKokoro, CompiledKokoroError};

/// Pre-compile segment 0: PlBert + bert_encoder.
pub(super) fn precompile_segment_plbert(
    kokoro: &CompiledKokoro,
    output_dir: &std::path::Path,
    seq_lens: &[usize],
) -> Result<(usize, usize), CompiledKokoroError> {
    let dev = model_device(kokoro.shared.model.as_ref());
    let mut total = (0, 0);

    for &seq_len in seq_lens {
        let input_ids = DynTensor::zeros(&[1, seq_len], DType::F32, &dev)?;
        let (_out, graph) =
            super::super::trace_fns::trace_seg_plbert(kokoro, &input_ids).map_err(|e| {
                CompiledKokoroError::PrecompileFailed {
                    segment: "plbert",
                    shape_key: seq_len,
                    source: Box::new(e),
                }
            })?;
        let (f, b) = super::export_plan_msl(&graph, "plbert", seq_len, output_dir)?;
        total.0 += f;
        total.1 += b;
    }

    Ok(total)
}

/// Pre-compile segment 1: TextEncoder.
pub(super) fn precompile_segment_text(
    kokoro: &CompiledKokoro,
    output_dir: &std::path::Path,
    seq_lens: &[usize],
) -> Result<(usize, usize), CompiledKokoroError> {
    let dev = model_device(kokoro.shared.model.as_ref());
    let mut total = (0, 0);

    for &seq_len in seq_lens {
        let input_ids = DynTensor::zeros(&[1, seq_len], DType::F32, &dev)?;
        let (_out, graph) =
            super::super::trace_fns::trace_seg_text(kokoro, &input_ids).map_err(|e| {
                CompiledKokoroError::PrecompileFailed {
                    segment: "text",
                    shape_key: seq_len,
                    source: Box::new(e),
                }
            })?;
        let (f, b) = super::export_plan_msl(&graph, "text", seq_len, output_dir)?;
        total.0 += f;
        total.1 += b;
    }

    Ok(total)
}

/// Pre-compile segment 2: ProsodyPredictor.
pub(super) fn precompile_segment_prosody(
    kokoro: &CompiledKokoro,
    output_dir: &std::path::Path,
    seq_lens: &[usize],
) -> Result<(usize, usize), CompiledKokoroError> {
    let dev = model_device(kokoro.shared.model.as_ref());
    let d_en = kokoro.shared.config.d_en;
    let style_dim = kokoro.shared.config.style_dim;
    let mut total = (0, 0);

    for &seq_len in seq_lens {
        let bert_features = DynTensor::zeros(&[1, d_en, seq_len], DType::F32, &dev)?;
        let style = DynTensor::zeros(&[1, style_dim], DType::F32, &dev)?;
        let (_out, graph) =
            super::super::trace_fns::trace_seg_prosody(kokoro, &bert_features, &style).map_err(
                |e| CompiledKokoroError::PrecompileFailed {
                    segment: "prosody",
                    shape_key: seq_len,
                    source: Box::new(e),
                },
            )?;
        let (f, b) = super::export_plan_msl(&graph, "prosody", seq_len, output_dir)?;
        total.0 += f;
        total.1 += b;
    }

    Ok(total)
}

/// Pre-compile segment 3: F0EnergyPredictor.
pub(super) fn precompile_segment_f0(
    kokoro: &CompiledKokoro,
    output_dir: &std::path::Path,
    t_mels: &[usize],
) -> Result<(usize, usize), CompiledKokoroError> {
    let dev = model_device(kokoro.shared.model.as_ref());
    let config = &kokoro.shared.config;
    let prosody_dim = config.d_en + config.style_dim;
    let style_dim = config.style_dim;
    let mut total = (0, 0);

    for &t_mel in t_mels {
        let aligned = DynTensor::zeros(&[1, prosody_dim, t_mel], DType::F32, &dev)?;
        let style = DynTensor::zeros(&[1, style_dim], DType::F32, &dev)?;
        let (_out, graph) = super::super::trace_fns::trace_seg_f0(kokoro, &aligned, &style)
            .map_err(|e| CompiledKokoroError::PrecompileFailed {
                segment: "f0",
                shape_key: t_mel,
                source: Box::new(e),
            })?;
        let (f, b) = super::export_plan_msl(&graph, "f0", t_mel, output_dir)?;
        total.0 += f;
        total.1 += b;
    }

    Ok(total)
}

/// Pre-compile segment 4: Generator (FullDecoder).
pub(super) fn precompile_segment_generator(
    kokoro: &CompiledKokoro,
    output_dir: &std::path::Path,
    t_mels: &[usize],
) -> Result<(usize, usize), CompiledKokoroError> {
    let dev = model_device(kokoro.shared.model.as_ref());
    let config = &kokoro.shared.config;
    let d_en = config.d_en;
    let style_dim = config.style_dim;
    let n_fft = config.n_fft;
    let n_bins = n_fft / 2 + 1;
    let hop_length = n_fft / 4;
    let upsample_factor: usize = config.upsample_rates.iter().product();
    let source_upsample = upsample_factor * hop_length;
    let mut total = (0, 0);

    for &t_mel in t_mels {
        let t_f0 = 2 * t_mel;
        let t_audio = t_f0 * source_upsample;
        let t_stft = t_audio / hop_length + 1;

        let regulated = DynTensor::zeros(&[1, d_en, t_mel], DType::F32, &dev)?;
        let f0 = DynTensor::zeros(&[1, 1, t_f0], DType::F32, &dev)?;
        let energy = DynTensor::zeros(&[1, 1, t_f0], DType::F32, &dev)?;
        let decoder_style = DynTensor::zeros(&[1, style_dim], DType::F32, &dev)?;
        let har_source = DynTensor::zeros(&[1, 2 * n_bins, t_stft], DType::F32, &dev)?;

        let (_out, graph) = super::super::trace_fns::trace_seg_generator(
            kokoro,
            &regulated,
            &f0,
            &energy,
            &decoder_style,
            &har_source,
        )
        .map_err(|e| CompiledKokoroError::PrecompileFailed {
            segment: "generator",
            shape_key: t_mel,
            source: Box::new(e),
        })?;
        let (f, b) = super::export_plan_msl(&graph, "generator", t_mel, output_dir)?;
        total.0 += f;
        total.1 += b;
    }

    Ok(total)
}

/// Pre-compile segment 5: Regulate (elementwise, no model weights).
///
/// Pure elementwise ops (sigmoid, sum, clamp, floor). No model weights needed.
/// Cache key: `seq_len` (phoneme count T from dur_logits).
pub(super) fn precompile_segment_regulate(
    kokoro: &CompiledKokoro,
    output_dir: &std::path::Path,
    seq_lens: &[usize],
) -> Result<(usize, usize), CompiledKokoroError> {
    let dev = model_device(kokoro.shared.model.as_ref());
    let max_dur = kokoro.shared.config.max_dur;
    let max_dur_f64 = max_dur as f64;
    let mut total = (0, 0);

    for &seq_len in seq_lens {
        let dur_logits = DynTensor::zeros(&[1, seq_len, max_dur], DType::F32, &dev)?;
        let speed_inv = DynTensor::full(&[1], 1.0, DType::F32, &dev)?;
        let (_out, graph) =
            super::super::trace_fns::trace_seg_regulate(&dur_logits, &speed_inv, max_dur_f64)
                .map_err(|e| CompiledKokoroError::PrecompileFailed {
                    segment: "regulate",
                    shape_key: seq_len,
                    source: Box::new(e),
                })?;
        let (f, b) = super::export_plan_msl(&graph, "regulate", seq_len, output_dir)?;
        total.0 += f;
        total.1 += b;
    }

    Ok(total)
}

/// Pre-compile segment 5a: SineGen pre-cumsum.
///
/// Traces steps 1-4 of SineGen (upsample F0 → expand harmonics → normalize →
/// downsample). No model weights needed. Cache key: `t_frames`.
pub(super) fn precompile_segment_sinegen_pre(
    kokoro: &CompiledKokoro,
    output_dir: &std::path::Path,
    t_frames_list: &[usize],
) -> Result<(usize, usize), CompiledKokoroError> {
    let sm = kokoro.shared.source_module.as_ref().ok_or_else(|| {
        CompiledKokoroError::PrecompileFailed {
            segment: "sinegen_pre",
            shape_key: 0,
            source: Box::new(nn_core::TensorError::Unsupported(
                "SourceModule not loaded".into(),
            )),
        }
    })?;
    let dev = sm.linear().weight().device();
    let sg = sm.sine_gen();
    let config = &kokoro.shared.config;
    let upsample_factor: usize = config.upsample_rates.iter().product();
    let hop_length = config.n_fft / 4;
    let source_upsample = upsample_factor * hop_length;
    let mut total = (0, 0);

    for &t_frames in t_frames_list {
        let f0 = DynTensor::zeros(&[1, t_frames, 1], DType::F32, &dev)?;
        let (_out, graph) = super::super::trace_fns::trace_seg_sinegen_pre(
            &f0,
            source_upsample,
            sg.sampling_rate(),
            sg.n_channels(),
        )
        .map_err(|e| CompiledKokoroError::PrecompileFailed {
            segment: "sinegen_pre",
            shape_key: t_frames,
            source: Box::new(e),
        })?;
        let (f, b) = super::export_plan_msl(&graph, "sinegen_pre", t_frames, output_dir)?;
        total.0 += f;
        total.1 += b;
    }

    Ok(total)
}

/// Pre-compile segment 5b: SineGen post-cumsum.
///
/// Traces steps 6-8 of SineGen + SourceModule linear → tanh → transpose.
/// Uses SourceModule's Linear layer (model weights). Cache key: `t_frames`.
pub(super) fn precompile_segment_sinegen_post(
    kokoro: &CompiledKokoro,
    output_dir: &std::path::Path,
    t_frames_list: &[usize],
) -> Result<(usize, usize), CompiledKokoroError> {
    let sm = kokoro.shared.source_module.as_ref().ok_or_else(|| {
        CompiledKokoroError::PrecompileFailed {
            segment: "sinegen_post",
            shape_key: 0,
            source: Box::new(nn_core::TensorError::Unsupported(
                "SourceModule not loaded".into(),
            )),
        }
    })?;
    let dev = sm.linear().weight().device();
    let sg = sm.sine_gen();
    let n_ch = sg.n_channels();
    let config = &kokoro.shared.config;
    let upsample_factor: usize = config.upsample_rates.iter().product();
    let hop_length = config.n_fft / 4;
    let source_upsample = upsample_factor * hop_length;
    let mut total = (0, 0);

    let voiced_threshold = f64::from(sg.voiced_threshold());
    for &t_frames in t_frames_list {
        let cum_gpu = DynTensor::zeros(&[1, t_frames, n_ch], DType::F32, &dev)?;
        let f0_gpu = DynTensor::zeros(&[1, t_frames, 1], DType::F32, &dev)?;
        let (_out, graph) = super::super::trace_fns::trace_seg_sinegen_post(
            &cum_gpu,
            &f0_gpu,
            sm.linear(),
            source_upsample,
            sg.sine_amp(),
            voiced_threshold,
        )
        .map_err(|e| CompiledKokoroError::PrecompileFailed {
            segment: "sinegen_post",
            shape_key: t_frames,
            source: Box::new(e),
        })?;
        let (f, b) = super::export_plan_msl(&graph, "sinegen_post", t_frames, output_dir)?;
        total.0 += f;
        total.1 += b;
    }

    Ok(total)
}
