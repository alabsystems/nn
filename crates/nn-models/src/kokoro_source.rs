// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SineGen + SourceModule: multi-harmonic excitation source for Kokoro TTS.
//!
//! Matches the dvoice/PyTorch SineGen phase algorithm exactly:
//!   1. Upsample F0 to audio rate (nearest-neighbor)
//!   2. Expand to harmonics at audio rate
//!   3. Normalize: (fn_hz / sr) % 1 at audio rate (handles aliasing)
//!   4. Downsample to frame rate (linear interpolation — smoothing step)
//!   5. Cumulative sum at frame rate (O(T_frames))
//!   6. Scale by 2π × upsample_factor
//!   7. Upsample to audio rate (linear interpolation)
//!   8. sines = sin(phase) × sine_amp
//!
//! See `designs/archive/2026-03-16-kokoro-architecture-correction.md`, correction #2.

use std::sync::Mutex;

use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::{Linear, Module};
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device, Result};

use crate::kokoro_error::KokoroError;

// -- Phase precision analysis (#2649) -----------------------------------------

/// Maximum safe number of frames for SineGen cumsum.
///
/// **Kahan f32 error analysis:** For n terms each in [0, 1), Kahan summation
/// has error bound `(2ε + O(nε²)) × Σ|a_i|` where ε = 2^-24 ≈ 5.96e-8.
///
/// Worst case: harmonic 9 at max F0 ~1600Hz → phase_inc = 9*1600/24000 = 0.6
/// per frame. At n=8000 frames: cumsum value ≈ 4800, error ≈ 2ε × 4800 ≈ 5.7e-4.
/// Scaled by 2π × upp(300): phase error ≈ 1.08 rad ≈ 62°.
///
/// At n=4000 frames (~33 seconds): error ≈ 0.54 rad ≈ 31° (harmonic 9 only).
/// This is well below the perceptible threshold for speech synthesis because:
/// 1. SourceModule's learned linear projection averages across harmonics
/// 2. Human hearing is phase-insensitive for non-tonal speech
/// 3. Per-sample *frequency* error is ~2ε ≈ 1.2e-7 Hz (inaudible)
///
/// Limit: 8000 frames ≈ 67 seconds at 24kHz/300upp. Provides ~2× headroom
/// over Kokoro's 512-token maximum (~40 seconds).
pub const MAX_SINEGEN_FRAMES: usize = 8000;

// -- SineGen ------------------------------------------------------------------

/// Multi-harmonic sine generator with voiced/unvoiced masking and noise.
///
/// Given F0 pitch contour `[B, T_frames, 1]` and an upsample factor,
/// produces `(sines [B, T_audio, 9], voiced [B, T_audio, 1], noise [B, T_audio, 9])`.
///
/// The 9 channels correspond to harmonics 1-9 (fundamental + 8 overtones).
pub struct SineGen {
    sampling_rate: f32,
    harmonic_num: usize,
    sine_amp: f32,
    /// Noise standard deviation for voiced regions. Currently unused in inference
    /// (deterministic zero noise). Retained for future training mode support.
    #[allow(dead_code)]
    noise_std: f32,
    voiced_threshold: f32,
    /// Terminal cumulative phase from the previous streaming chunk.
    ///
    /// Shape: `[1, 1, n_channels]` (one value per harmonic). When `Some`, the
    /// next `forward()` call adds this offset to the cumulative sum before
    /// scaling by `2*pi*upp`, ensuring phase continuity across chunk boundaries.
    ///
    /// Uses `Mutex` for interior mutability so `forward()` can remain `&self`
    /// (required by `SourceModule` and `KokoroModel` which hold `SineGen`
    /// immutably). `Mutex` (not `RefCell`) because `SineGen` lives inside
    /// `Arc<SharedKokoroState>` which must be `Send + Sync` for downstream
    /// thread pool usage. Lock contention is impossible in practice since
    /// `forward()` is called sequentially per voice.
    last_cumphase: Mutex<Option<Vec<f32>>>,
}

impl Default for SineGen {
    fn default() -> Self {
        Self::new()
    }
}

impl SineGen {
    /// Create with Kokoro defaults: 24kHz, 8 overtones, amp=0.1, threshold=10.
    pub fn new() -> Self {
        Self {
            sampling_rate: 24000.0,
            harmonic_num: 8,
            sine_amp: 0.1,
            noise_std: 0.003,
            voiced_threshold: 10.0,
            last_cumphase: Mutex::new(None),
        }
    }

    /// Number of output channels (fundamental + harmonics).
    pub fn n_channels(&self) -> usize {
        self.harmonic_num + 1
    }

    /// Sampling rate in Hz (default: 24000).
    pub fn sampling_rate(&self) -> f32 {
        self.sampling_rate
    }

    /// Sine amplitude scaling factor (default: 0.1).
    pub fn sine_amp(&self) -> f32 {
        self.sine_amp
    }

    /// F0 threshold below which frames are treated as unvoiced (default: 10.0 Hz).
    pub fn voiced_threshold(&self) -> f32 {
        self.voiced_threshold
    }

    /// Reset the carried-over cumulative phase to zero.
    ///
    /// Call this at the start of each new utterance / streaming session.
    /// Within a session, phase carries across chunks automatically via
    /// `forward()`.
    pub fn reset_phase(&self) {
        // SAFETY invariant: no panic while holding lock, so poisoning is impossible.
        *self.last_cumphase.lock().unwrap() = None;
    }

    /// Forward: F0 → (sines, voiced_mask, noise).
    ///
    /// `f0`: `[B, T_frames, 1]` — fundamental frequency in Hz.
    /// `upp`: upsample factor (product of Generator upsample rates, e.g. 300).
    ///
    /// Returns `(sines [B, T_audio, n_ch], voiced [B, T_audio, 1], noise [B, T_audio, n_ch])`
    /// where `T_audio = T_frames * upp` and `n_ch = harmonic_num + 1`.
    ///
    /// Hybrid GPU/CPU: steps 1-4 and 6-8 run as DynTensor GPU ops; step 5
    /// (cumsum) reads back to CPU for f64 precision (#2691). Step 7
    /// (audio-rate upsample) uses decomposed GPU ops (#2909).
    pub fn forward(
        &self,
        f0: &DynTensor,
        upp: usize,
    ) -> std::result::Result<(DynTensor, DynTensor, DynTensor), KokoroError> {
        let device = f0.device();
        let batch = f0.dim(0)?;
        let t_frames = f0.dim(1)?;
        if t_frames > MAX_SINEGEN_FRAMES {
            return Err(KokoroError::InvalidInput(format!(
                "SineGen: t_frames={t_frames} exceeds MAX_SINEGEN_FRAMES={MAX_SINEGEN_FRAMES} \
                 (~{} seconds). Phase precision degrades beyond this limit.",
                MAX_SINEGEN_FRAMES * upp / 24000
            )));
        }
        let t_audio = t_frames * upp;
        let n_ch = self.n_channels();
        let sr = self.sampling_rate;

        // Steps 1-3: GPU — upsample → harmonics → normalize → fmod
        let f0_audio = f0
            .unsqueeze(2)?
            .expand([batch, t_frames, upp, 1])?
            .reshape([batch, t_audio, 1])?;
        let harmonics_data: Vec<f32> = (1..=n_ch).map(|h| h as f32).collect();
        let harmonics = DynTensor::from_vec(harmonics_data, &[1, 1, n_ch], &device)?;
        let freq = f0_audio.broadcast_mul(&harmonics)?;
        // fract(freq/sr) = (freq/sr) - floor(freq/sr). mul_scalar + fract = 2
        // dispatches vs affine(1/sr,0) + floor + sub = 4 (affine with bias=0
        // still dispatches add-zero). Saves 2 GPU dispatches total. (#1815)
        let rad_audio = freq.mul_scalar(1.0 / f64::from(sr))?.fract()?;

        // Step 4: GPU — downsample to frame rate (linear interpolation).
        // 3-op decomposition; frame-rate error bounded by 1/(2*upp) (#2909).
        let rad_frames = interp_downsample_gpu(&rad_audio, t_frames)?;

        // Step 5: GPU — Kahan-compensated cumulative sum along dim 1 (time frames).
        // Kahan f32 achieves O(nε) error vs O(n²ε) for naive f32. Worst-case
        // phase error ~0.014 rad (vs ~2.3 rad naive, ~0 rad f64). Eliminates
        // the CPU f64 round-trip that forced a GPU flush. (#2909, #2691)
        let cum_gpu = rad_frames.cumsum_kahan(1)?;

        // Step 5b: Phase continuity — add carried-over terminal phase from
        // previous streaming chunk. Without this, cumsum resets to zero at each
        // chunk boundary, creating audible clicking/popping artifacts.
        // The offset is in normalized-frequency units (cycles), matching cumsum
        // output. fract() prevents unbounded growth over many chunks since
        // sin(2*pi*x) is periodic with period 1 in the scaled domain.
        let cum_gpu = {
            let prev = self.last_cumphase.lock().unwrap();
            if let Some(ref prev_phase) = *prev {
                let offset = DynTensor::from_vec(prev_phase.clone(), &[1, 1, n_ch], &device)?;
                cum_gpu.broadcast_add(&offset)?
            } else {
                cum_gpu
            }
        };

        // Save terminal cumulative phase for the next streaming chunk.
        // Extract the last frame: cum_gpu[:, -1, :] → [n_ch] values.
        // fract() keeps values in [0, 1) to prevent precision loss over many
        // chunks (sin is periodic, so fract preserves phase continuity).
        {
            let last_frame_idx =
                DynTensor::from_vec_u32(vec![(t_frames - 1) as u32], &[1], &device)?;
            let last_frame = cum_gpu.index_select(&last_frame_idx, 1)?;
            let last_frame_frac = last_frame.fract()?;
            let last_vals = last_frame_frac.to_flat_vec::<f32>()?;
            *self.last_cumphase.lock().unwrap() = Some(last_vals);
        }

        // Step 6: GPU — scale by 2π × upp (single dispatch).
        // Folded from 3 mul_scalar calls to 1 — saves 2 GPU dispatches (#1815).
        let phase_frames = cum_gpu.mul_scalar(std::f64::consts::TAU * upp as f64)?;

        // Step 7: GPU — upsample phase to audio rate. Decomposed DynTensor
        // ops (index_select + broadcast_mul + broadcast_add) matching
        // half-pixel coordinate mapping. Replaces CPU interp loop (#2909).
        let phase_audio = interp_upsample_gpu(&phase_frames, t_audio)?;

        // Step 8: GPU — sin(phase) × sine_amp
        let sines = phase_audio.sin()?.mul_scalar(f64::from(self.sine_amp))?;

        // Voiced mask: GPU — f0 > threshold at audio rate.
        let voiced = f0_audio
            .gt(f64::from(self.voiced_threshold))?
            .to_dtype(DType::F32)?;

        // Noise: deterministic zeros for inference.
        let noise = DynTensor::zeros(&[batch, t_audio, n_ch], DType::F32, &device)?;

        Ok((sines, voiced, noise))
    }
}

// -- GPU interpolation helpers ------------------------------------------------

/// GPU-compatible linear interpolation downsample via DynTensor ops.
///
/// `[B, T_in, C]` → `[B, t_out, C]` using index_select + broadcast_mul + add.
/// Matches half-pixel coordinate mapping: `src = (dst + 0.5) * scale - 0.5`.
///
/// Index tensors are U32 (for eager dispatch compatibility with `index_select`).
/// Trace-compiled paths that need f32 indices should compute them inline.
pub fn interp_downsample_gpu(x: &DynTensor, t_out: usize) -> Result<DynTensor> {
    let t_in = x.dim(1)?;
    if t_in == t_out {
        return x.contiguous();
    }
    let device = x.device();
    let scale = t_in as f32 / t_out as f32;
    let max_lo = t_in.saturating_sub(2) as f32;
    let t_in_m1 = t_in.saturating_sub(1) as f32;

    let mut lo_vec = Vec::with_capacity(t_out);
    let mut frac_vec = Vec::with_capacity(t_out);
    let mut one_m_frac_vec = Vec::with_capacity(t_out);
    for dst in 0..t_out {
        let src = ((dst as f32 + 0.5) * scale - 0.5).clamp(0.0, t_in_m1);
        let lo = src.floor().min(max_lo);
        lo_vec.push(lo as u32);
        let f = src - lo;
        frac_vec.push(f);
        one_m_frac_vec.push(1.0 - f);
    }

    let hi_vec: Vec<u32> = lo_vec.iter().map(|&l| l + 1).collect();
    let lo_ids = DynTensor::from_vec_u32(lo_vec, &[t_out], &device)?;
    let hi_ids = DynTensor::from_vec_u32(hi_vec, &[t_out], &device)?;
    let lo_vals = x.index_select(&lo_ids, 1)?;
    let hi_vals = x.index_select(&hi_ids, 1)?;

    let frac = DynTensor::from_vec(frac_vec, &[1, t_out, 1], &device)?;
    // Precompute 1-frac on CPU (free) instead of GPU affine(-1,1) which
    // dispatches 2 kernels (broadcast_mul + broadcast_add). Saves 2 GPU
    // dispatches per interp call. (#1815)
    let one_m_frac = DynTensor::from_vec(one_m_frac_vec, &[1, t_out, 1], &device)?;
    lo_vals
        .broadcast_mul(&one_m_frac)?
        .broadcast_add(&hi_vals.broadcast_mul(&frac)?)
}

/// GPU-compatible linear interpolation upsample via DynTensor ops.
///
/// `[B, T_in, C]` → `[B, t_out, C]` using index_select + broadcast_mul + add.
/// Same half-pixel coordinate mapping as `interp_downsample_gpu`. (#2909)
///
/// Index tensors are U32 (for eager dispatch compatibility with `index_select`).
/// Trace-compiled paths that need f32 indices should compute them inline.
pub fn interp_upsample_gpu(x: &DynTensor, t_out: usize) -> Result<DynTensor> {
    let t_in = x.dim(1)?;
    let batch = x.dim(0)?;
    let channels = x.dim(2)?;
    // Single-frame input has no hi neighbor — broadcast instead.
    if t_in <= 1 {
        return x.expand([batch, t_out, channels]);
    }
    if t_in == t_out {
        return x.contiguous();
    }
    let device = x.device();
    let scale = t_in as f32 / t_out as f32;
    let max_lo = t_in.saturating_sub(2) as f32;
    let t_in_m1 = t_in.saturating_sub(1) as f32;

    let mut lo_vec = Vec::with_capacity(t_out);
    let mut frac_vec = Vec::with_capacity(t_out);
    let mut one_m_frac_vec = Vec::with_capacity(t_out);
    for dst in 0..t_out {
        let src = ((dst as f32 + 0.5) * scale - 0.5).clamp(0.0, t_in_m1);
        let lo = src.floor().min(max_lo);
        lo_vec.push(lo as u32);
        let f = src - lo;
        frac_vec.push(f);
        one_m_frac_vec.push(1.0 - f);
    }

    let hi_vec: Vec<u32> = lo_vec.iter().map(|&l| l + 1).collect();
    let lo_ids = DynTensor::from_vec_u32(lo_vec, &[t_out], &device)?;
    let hi_ids = DynTensor::from_vec_u32(hi_vec, &[t_out], &device)?;
    let lo_vals = x.index_select(&lo_ids, 1)?;
    let hi_vals = x.index_select(&hi_ids, 1)?;

    let frac = DynTensor::from_vec(frac_vec, &[1, t_out, 1], &device)?;
    // Precompute 1-frac on CPU — saves 2 GPU dispatches. (#1815)
    let one_m_frac = DynTensor::from_vec(one_m_frac_vec, &[1, t_out, 1], &device)?;
    lo_vals
        .broadcast_mul(&one_m_frac)?
        .broadcast_add(&hi_vals.broadcast_mul(&frac)?)
}

// -- SourceModule -------------------------------------------------------------

/// Learned source module: SineGen → linear projection → tanh.
///
/// Wraps SineGen with a `Linear(n_channels, 1)` to combine harmonics into
/// a single excitation signal.
///
/// Weight prefix: `decoder.generator.m_source.` in safetensors.
pub struct SourceModule {
    sine_gen: SineGen,
    l_linear: Linear,
}

impl SourceModule {
    /// Load from VarBuilder.
    ///
    /// Expects weights: `l_linear.weight` `[1, 9]`, `l_linear.bias` `[1]`.
    pub fn load(vb: impl AsRef<VarBuilder>) -> Result<Self> {
        let vb = vb.as_ref();
        let sine_gen = SineGen::new();
        let n_ch = sine_gen.n_channels();
        let l_linear = {
            let w = vb.get(&[1, n_ch], "l_linear.weight")?;
            let b = vb.get(&[1], "l_linear.bias")?;
            Linear::new(w, Some(b))?
        };
        Ok(Self { sine_gen, l_linear })
    }

    /// Transfer `l_linear` weights to the given device.
    ///
    /// Returns a new `SourceModule` with weights on `device`. No-op if already
    /// on the correct device (DynTensor::to_device clones cheaply via Arc).
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        let w = self.l_linear.weight().to_device(device)?;
        let b = self
            .l_linear
            .bias()
            .map(|b| b.to_device(device))
            .transpose()?;
        Ok(Self {
            sine_gen: SineGen::new(),
            l_linear: Linear::new(w, b)?,
        })
    }

    /// Access to the learned linear projection layer.
    ///
    /// Used by trace-compiled SineGen segments to wire `l_linear` weights
    /// into the computation graph. See `compiled_kokoro_trace_fns.rs`.
    pub fn linear(&self) -> &Linear {
        &self.l_linear
    }

    /// Access to the internal SineGen configuration.
    ///
    /// Used by trace-compiled segments to read sampling_rate, harmonic_num, etc.
    pub fn sine_gen(&self) -> &SineGen {
        &self.sine_gen
    }

    /// Reset the SineGen cumulative phase for a new utterance.
    ///
    /// Call at the start of each streaming session. Within a session, phase
    /// continuity is maintained automatically across chunks.
    pub fn reset_phase(&self) {
        self.sine_gen.reset_phase();
    }

    /// Forward: F0 → excitation signal.
    ///
    /// `f0`: `[B, T_frames, 1]` — fundamental frequency in Hz.
    /// `upp`: upsample factor (product of Generator upsample rates).
    ///
    /// Returns `[B, T_audio, 1]` excitation signal (tanh-bounded).
    pub fn forward(
        &self,
        f0: &DynTensor,
        upp: usize,
    ) -> std::result::Result<DynTensor, KokoroError> {
        let (sines, voiced, _noise) = self.sine_gen.forward(f0, upp)?;
        // broadcast_mul handles [B,T,9] × [B,T,1] natively — no expand
        // needed. Saves 1 GPU dispatch (expand kernel). (#1815)
        let sine_wavs = sines.broadcast_mul(&voiced)?;
        let projected = self.l_linear.forward(&sine_wavs)?;
        Ok(projected.tanh()?)
    }
}

#[cfg(test)]
#[path = "kokoro_source_tests.rs"]
mod tests;

#[cfg(kani)]
#[path = "kokoro_source_kani_tests.rs"]
mod kani_proofs;
