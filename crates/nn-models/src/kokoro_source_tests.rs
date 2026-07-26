// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for SineGen and SourceModule.

use super::*;
use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};

fn ones(shape: &[usize]) -> DynTensor {
    DynTensor::ones(shape, DType::F32, &Device::Cpu).unwrap()
}

// -- Test-only helpers (extracted from kokoro_source.rs) -----------------------

/// CPU linear interpolation upsample — reference implementation.
///
/// Retained for testing parity against `interp_upsample_gpu` (#2909).
fn interp_upsample_cpu(
    phase_frames: &DynTensor,
    batch: usize,
    t_out: usize,
    channels: usize,
) -> Result<DynTensor> {
    let t_in = phase_frames.dim(1)?;
    // Guard: single-frame input has no hi neighbor — broadcast instead.
    if t_in <= 1 {
        return phase_frames.expand([batch, t_out, channels]);
    }
    let data = phase_frames.to_flat_vec::<f32>()?;
    let scale = t_in as f32 / t_out as f32;
    let max_idx = t_in.saturating_sub(2) as f32;
    let t_in_m1 = t_in.saturating_sub(1) as f32;

    let mut out = vec![0.0f32; batch * t_out * channels];
    for b in 0..batch {
        let base_in = b * t_in * channels;
        let base_out = b * t_out * channels;
        for dst in 0..t_out {
            let src = ((dst as f32 + 0.5) * scale - 0.5).clamp(0.0, t_in_m1);
            let lo = src.floor().min(max_idx) as usize;
            let hi = lo + 1;
            let frac = src - lo as f32;
            let one_m_frac = 1.0f32 - frac;
            for c in 0..channels {
                out[base_out + dst * channels + c] = one_m_frac * data[base_in + lo * channels + c]
                    + frac * data[base_in + hi * channels + c];
            }
        }
    }
    DynTensor::from_vec(out, &[batch, t_out, channels], &Device::Cpu)
}

/// Upsample tensor along dim 1 via nearest-neighbor.
///
/// `x`: `[B, T, C]` → `[B, T*factor, C]`.
fn upsample_nearest_dim1(x: &DynTensor, factor: usize) -> Result<DynTensor> {
    if factor == 1 {
        return x.contiguous();
    }
    let (batch, t, c) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
    let expanded = x.unsqueeze(2)?.expand([batch, t, factor, c])?;
    expanded.reshape([batch, t * factor, c])
}

/// Fractional part: x - floor(x). Equivalent to Python `x % 1`.
///
/// Prevents floating-point overflow in cumsum by wrapping phase increments
/// when harmonics exceed the Nyquist frequency.
fn fmod_one(x: &DynTensor) -> Result<DynTensor> {
    let floored = x.floor()?;
    x.broadcast_sub(&floored)
}

/// Linear interpolation along dim 1: `[B, T_in, C]` → `[B, T_out, C]`.
///
/// Equivalent to PyTorch `F.interpolate(mode="linear", align_corners=False)`.
/// Uses half-pixel coordinate mapping: `src = (dst + 0.5) * scale - 0.5`.
///
/// Single-pass raw loop to match PyTorch's single-pass vectorized kernel.
/// SineGen phase values reach ~80,000 radians; even 1 ULP interpolation
/// difference cascades through sin() to produce completely different outputs.
/// The 3-pass DynTensor approach (index_select → broadcast_mul → broadcast_add)
/// rounds differently from PyTorch's fused `(1-f)*lo + f*hi` (#2691).
fn linear_interpolate_1d(x: &DynTensor, t_out: usize) -> Result<DynTensor> {
    let (batch, t_in, channels) = (x.dim(0)?, x.dim(1)?, x.dim(2)?);
    if t_in == t_out {
        return x.contiguous();
    }
    if t_in <= 1 {
        return x.expand([batch, t_out, channels]);
    }
    let device = x.device();
    let data = x.to_flat_vec::<f32>()?;
    let scale = t_in as f32 / t_out as f32;
    let max_idx = (t_in - 2) as f32;
    let t_in_m1 = (t_in - 1) as f32;

    let total = batch * t_out * channels;
    let mut out = vec![0.0f32; total];

    for b in 0..batch {
        let base_in = b * t_in * channels;
        let base_out = b * t_out * channels;
        for dst in 0..t_out {
            let src = ((dst as f32 + 0.5) * scale - 0.5).clamp(0.0, t_in_m1);
            let lo = src.floor().min(max_idx) as usize;
            let hi = lo + 1;
            let frac = src - lo as f32;
            let one_m_frac = 1.0f32 - frac;
            let out_offset = base_out + dst * channels;
            let lo_offset = base_in + lo * channels;
            let hi_offset = base_in + hi * channels;
            for c in 0..channels {
                out[out_offset + c] = one_m_frac * data[lo_offset + c] + frac * data[hi_offset + c];
            }
        }
    }

    DynTensor::from_vec(out, &[batch, t_out, channels], &device)
}

/// Voiced mask: 1.0 where f0 > threshold, 0.0 otherwise.
///
/// `f0`: `[B, T, 1]`. Returns `[B, T, 1]`.
fn f0_voiced_mask(f0: &DynTensor, threshold: f32) -> Result<DynTensor> {
    let mask_u32 = f0.gt(f64::from(threshold))?;
    mask_u32.to_dtype(DType::F32)
}

#[test]
fn test_sinegen_output_shape() {
    let sg = SineGen::new();
    let f0 = DynTensor::full(&[1, 10, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
    let upp = 120;
    let (sines, voiced, noise) = sg.forward(&f0, upp).unwrap();
    // T_audio = 10 * 120 = 1200, n_ch = 9
    assert_eq!(sines.dims(), &[1, 1200, 9]);
    assert_eq!(voiced.dims(), &[1, 1200, 1]);
    assert_eq!(noise.dims(), &[1, 1200, 9]);
}

#[test]
fn test_sinegen_voiced_mask() {
    // F0 = 0 → unvoiced (below threshold 10)
    let f0_zero = DynTensor::full(&[1, 5, 1], 0.0, DType::F32, &Device::Cpu).unwrap();
    let mask = f0_voiced_mask(&f0_zero, 10.0).unwrap();
    let vals = mask.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v == 0.0), "zero F0 should be unvoiced");

    // F0 = 200 → voiced (above threshold)
    let f0_200 = DynTensor::full(&[1, 5, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
    let mask = f0_voiced_mask(&f0_200, 10.0).unwrap();
    let vals = mask.to_flat_vec::<f32>().unwrap();
    assert!(vals.iter().all(|&v| v == 1.0), "200Hz F0 should be voiced");
}

#[test]
fn test_upsample_nearest_dim1() {
    let x = DynTensor::from_vec(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 3, 2],
        &Device::Cpu,
    )
    .unwrap();
    let up = upsample_nearest_dim1(&x, 2).unwrap();
    assert_eq!(up.dims(), &[1, 6, 2]);
    let vals = up.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals,
        &[1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0]
    );
}

#[test]
fn test_upsample_nearest_dim1_factor_1() {
    let x = ones(&[2, 4, 3]);
    let up = upsample_nearest_dim1(&x, 1).unwrap();
    assert_eq!(up.dims(), &[2, 4, 3]);
}

#[test]
fn test_fmod_one() {
    let x = DynTensor::from_vec(
        vec![0.0_f32, 0.5, 1.0, 1.7, 2.3, 3.0],
        &[1, 6, 1],
        &Device::Cpu,
    )
    .unwrap();
    let result = fmod_one(&x).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    let expected = [0.0, 0.5, 0.0, 0.7, 0.3, 0.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "fmod_one[{i}]: got {got}, expected {exp}"
        );
    }
}

#[test]
fn test_linear_interpolate_1d_identity() {
    let x = DynTensor::from_vec(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[1, 3, 2],
        &Device::Cpu,
    )
    .unwrap();
    let result = linear_interpolate_1d(&x, 3).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    let expected: Vec<f32> = x.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        vals, expected,
        "identity interpolation should not change values"
    );
}

#[test]
fn test_linear_interpolate_1d_upsample() {
    let x = DynTensor::from_vec(vec![0.0_f32, 10.0], &[1, 2, 1], &Device::Cpu).unwrap();
    let result = linear_interpolate_1d(&x, 4).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    // align_corners=False: [0, 2.5, 7.5, 10]
    let expected = [0.0, 2.5, 7.5, 10.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 0.1,
            "linear_interp[{i}]: got {got:.2}, expected {exp:.2}"
        );
    }
}

#[test]
fn test_linear_interpolate_1d_downsample() {
    // Downsample from 12 to 4 (audio→frame rate, step 4 in SineGen)
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let x = DynTensor::from_vec(data, &[1, 12, 1], &Device::Cpu).unwrap();
    let result = linear_interpolate_1d(&x, 4).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals.len(), 4);
    // Output should be monotonically increasing
    for i in 1..vals.len() {
        assert!(
            vals[i] >= vals[i - 1] - 1e-5,
            "downsample not monotonic at {i}: {} < {}",
            vals[i],
            vals[i - 1]
        );
    }
}

#[test]
fn test_linear_interpolate_1d_monotonic() {
    let x = DynTensor::from_vec(vec![0.0_f32, 10.0, 30.0, 60.0], &[1, 4, 1], &Device::Cpu).unwrap();
    let result = linear_interpolate_1d(&x, 40).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    for i in 1..vals.len() {
        assert!(
            vals[i] >= vals[i - 1] - 1e-5,
            "not monotonic at {i}: {} < {}",
            vals[i],
            vals[i - 1]
        );
    }
}

#[test]
fn test_source_module_loads() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let sm = SourceModule::load(&vb);
    assert!(sm.is_ok(), "SourceModule should load: {:?}", sm.err());
}

#[test]
fn test_source_module_output_shape() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let sm = SourceModule::load(&vb).unwrap();
    let f0 = DynTensor::full(&[1, 8, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
    let out = sm.forward(&f0, 60).unwrap();
    assert_eq!(out.dims(), &[1, 480, 1]);
    let vals = out.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| (-1.0..=1.0).contains(&v)),
        "tanh should bound output"
    );
}

#[test]
fn test_sinegen_sine_amplitude() {
    let sg = SineGen::new();
    let f0 = DynTensor::full(&[1, 4, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
    let (sines, _, _) = sg.forward(&f0, 10).unwrap();
    let vals = sines.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| v.abs() <= 0.1 + 1e-6),
        "sines should be bounded by amplitude 0.1"
    );
}

/// Prove: nearest-neighbor upsampling creates instantaneous frequency discontinuity.
///
/// With two frames at different F0 values, the phase increment jumps abruptly
/// at the frame boundary. The reference algorithm uses linear interpolation to
/// glide smoothly. See #2680.
#[test]
fn test_nearest_upsample_frequency_discontinuity() {
    let sr = 24000.0_f32;
    let upp = 10_usize;
    let f0_1 = 200.0_f32;
    let f0_2 = 300.0_f32;
    let f0 = DynTensor::from_vec(vec![f0_1, f0_2], &[1, 2, 1], &Device::Cpu).unwrap();
    let phase_inc = f0.mul_scalar(1.0 / f64::from(sr)).unwrap();
    let phase_inc_up = upsample_nearest_dim1(&phase_inc, upp).unwrap();
    let vals = phase_inc_up.to_flat_vec::<f32>().unwrap();
    let before = vals[upp - 1];
    let after = vals[upp];
    let freq_jump = (after - before).abs() * sr;
    assert!(
        freq_jump > 99.0,
        "nearest-neighbor should jump ~100Hz at boundary, got {freq_jump}Hz"
    );
}

/// Verify: constant F0 produces smooth, continuous sine output.
#[test]
fn test_sinegen_constant_f0_smooth() {
    let sg = SineGen::new();
    let f0 = DynTensor::full(&[1, 4, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
    let (sines, _, _) = sg.forward(&f0, 10).unwrap();
    let vals = sines.to_flat_vec::<f32>().unwrap();
    let n_ch = 9;
    for i in 1..(4 * 10) {
        let prev = vals[(i - 1) * n_ch];
        let curr = vals[i * n_ch];
        let diff = (curr - prev).abs();
        assert!(
            diff < 0.01,
            "fundamental should be smooth: sample {i} diff={diff}"
        );
    }
}

/// Verify: fmod_one handles aliasing (harmonics above Nyquist).
#[test]
fn test_fmod_one_aliasing() {
    // 9th harmonic of 3kHz = 27kHz, sr=24kHz → phase_inc = 27/24 = 1.125
    // After fmod_one: 0.125 (aliased to 3kHz)
    let x = DynTensor::from_vec(vec![1.125_f32], &[1, 1, 1], &Device::Cpu).unwrap();
    let result = fmod_one(&x).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert!(
        (vals[0] - 0.125).abs() < 1e-5,
        "fmod_one(1.125) should be 0.125, got {}",
        vals[0]
    );
}

/// Verify: noise has correct conditional amplitude.
#[test]
fn test_sinegen_noise_amplitude() {
    let sg = SineGen::new();
    // All voiced (F0=200 > threshold=10)
    let f0_voiced = DynTensor::full(&[1, 4, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
    let (_, _, noise_v) = sg.forward(&f0_voiced, 10).unwrap();
    // Noise is zeros * amp, so all zeros regardless of amplitude
    let vals = noise_v.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| v.abs() < 1e-10),
        "deterministic noise should be zero"
    );

    // All unvoiced (F0=0 < threshold=10)
    let f0_unvoiced = DynTensor::full(&[1, 4, 1], 0.0, DType::F32, &Device::Cpu).unwrap();
    let (_, _, noise_u) = sg.forward(&f0_unvoiced, 10).unwrap();
    let vals = noise_u.to_flat_vec::<f32>().unwrap();
    assert!(
        vals.iter().all(|&v| v.abs() < 1e-10),
        "zeros * unvoiced_amp should still be zero"
    );
}

/// Verify `linear_interpolate_1d` produces exact PyTorch `F.interpolate(align_corners=False)` values.
///
/// PyTorch coordinate formula: `src = (dst + 0.5) * (T_in / T_out) - 0.5`
/// For input `[0, 10]`, output size 4:
///   dst=0: src=-0.25 → clamp=0.0 → val=0.0
///   dst=1: src= 0.25 → val=2.5
///   dst=2: src= 0.75 → val=7.5
///   dst=3: src= 1.25 → clamp=1.0 → val=10.0
#[test]
fn test_linear_interpolate_1d_exact_pytorch_values() {
    let x = DynTensor::from_vec(vec![0.0_f32, 10.0], &[1, 2, 1], &Device::Cpu).unwrap();
    let result = linear_interpolate_1d(&x, 4).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    let expected = [0.0_f32, 2.5, 7.5, 10.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "linear_interp[{i}]: got {got}, expected {exp} (diff={:.2e})",
            (got - exp).abs()
        );
    }
}

/// Verify `linear_interpolate_1d` with asymmetric input produces correct coordinate mapping.
///
/// PyTorch: `F.interpolate(tensor([1, 4, 9]).view(1,3,1), size=5, mode='linear', align_corners=False)`
/// scale = 3/5 = 0.6
///   dst=0: src=-0.20 → clamp=0.0 → val=1.0
///   dst=1: src= 0.40 → val=1*(1-0.4)+4*0.4 = 2.2
///   dst=2: src= 1.00 → val=4.0
///   dst=3: src= 1.60 → val=4*0.4+9*0.6 = 7.0
///   dst=4: src= 2.20 → clamp=2.0 → val=9.0
#[test]
fn test_linear_interpolate_1d_asymmetric_exact() {
    let x = DynTensor::from_vec(vec![1.0_f32, 4.0, 9.0], &[1, 3, 1], &Device::Cpu).unwrap();
    let result = linear_interpolate_1d(&x, 5).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    let expected = [1.0_f32, 2.2, 4.0, 7.0, 9.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "asymmetric[{i}]: got {got}, expected {exp} (diff={:.2e})",
            (got - exp).abs()
        );
    }
}

/// Document that `linear_interpolate_1d` with t_in=0 hits expand() error.
///
/// The `t_in <= 1` guard routes to `expand()`, but expand can't map dim 0→4.
/// SineGen always has t_in >= 1, so this is not reachable in production.
#[test]
fn test_linear_interpolate_1d_empty_input_errors() {
    let x = DynTensor::zeros(&[1, 0, 2], DType::F32, &Device::Cpu).unwrap();
    let result = linear_interpolate_1d(&x, 4);
    assert!(
        result.is_err(),
        "t_in=0 should error via expand() — expand can't map 0→4"
    );
}

/// Performance proof: `linear_interpolate_1d` runtime scales O(t_out), not O(t_out²).
///
/// The function has a single CPU loop O(t_out) for index computation, then
/// two `index_select` GPU/CPU ops O(B × t_out × C). Doubling t_out should
/// roughly double runtime. An O(n²) bug would cause 4× slowdown.
///
/// Part of #2218.
#[test]
fn test_linear_interpolate_1d_linear_scaling() {
    use std::time::Instant;

    let channels = 9;
    let t_in = 100;
    let data: Vec<f32> = (0..t_in * channels).map(|i| (i as f32) * 0.001).collect();
    let x = DynTensor::from_vec(data, &[1, t_in, channels], &Device::Cpu).unwrap();

    // Small output: 3000
    let t_small = 3000;
    // Large output: 4× bigger = 12000 (typical audio for 100-frame SineGen with upp=120)
    let t_large = t_small * 4;

    // Warm up
    let _ = linear_interpolate_1d(&x, t_small).unwrap();

    // Measure small (5 runs)
    let start = Instant::now();
    for _ in 0..5 {
        let out = linear_interpolate_1d(&x, t_small).unwrap();
        assert_eq!(out.dims(), &[1, t_small, channels]);
    }
    let time_small = start.elapsed();

    // Measure large (5 runs)
    let start = Instant::now();
    for _ in 0..5 {
        let out = linear_interpolate_1d(&x, t_large).unwrap();
        assert_eq!(out.dims(), &[1, t_large, channels]);
    }
    let time_large = start.elapsed();

    let ratio = time_large.as_secs_f64() / time_small.as_secs_f64();
    // O(n): expect ~4× ratio. O(n²): would be ~16×.
    // Use generous 8× bound to avoid timing flakiness.
    assert!(
        ratio < 8.0,
        "linear_interpolate_1d scaling ratio {ratio:.1}× for 4× output increase — \
         expected <8× (O(n)), got >8× suggesting O(n²). \
         small={:.3}ms, large={:.3}ms",
        time_small.as_secs_f64() * 1000.0 / 5.0,
        time_large.as_secs_f64() * 1000.0 / 5.0,
    );
}

/// Performance proof: `SineGen::forward` output sizes scale linearly with T_frames × upp.
///
/// Verifies no hidden quadratic allocation: output tensor element count is
/// exactly `B × T_audio × n_ch` where T_audio = T_frames × upp. No
/// intermediate tensors should grow faster than this.
///
/// Part of #2218.
#[test]
fn test_sinegen_output_size_linear_in_input() {
    let sg = SineGen::new();
    let n_ch = 9; // harmonic_num(8) + 1

    for &(t_frames, upp) in &[(10, 60), (20, 60), (10, 120), (40, 120)] {
        let f0 = DynTensor::full(&[1, t_frames, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
        let (sines, voiced, noise) = sg.forward(&f0, upp).unwrap();
        let t_audio = t_frames * upp;
        assert_eq!(
            sines.dims(),
            &[1, t_audio, n_ch],
            "sines shape mismatch for t_frames={t_frames}, upp={upp}"
        );
        assert_eq!(
            voiced.dims(),
            &[1, t_audio, 1],
            "voiced shape mismatch for t_frames={t_frames}, upp={upp}"
        );
        assert_eq!(
            noise.dims(),
            &[1, t_audio, n_ch],
            "noise shape mismatch for t_frames={t_frames}, upp={upp}"
        );
    }
}

/// Verify `interp_upsample_gpu` matches `interp_upsample_cpu` on CPU tensors.
///
/// Both functions implement the same half-pixel coordinate mapping:
/// `src = (dst + 0.5) * scale - 0.5`. The GPU variant uses decomposed
/// DynTensor ops (index_select + broadcast_mul + broadcast_add); the CPU
/// variant uses a fused `(1-f)*lo + f*hi` loop. On CPU device they should
/// produce identical results since there's no GPU rounding difference.
#[test]
fn test_interp_upsample_gpu_cpu_parity() {
    let data: Vec<f32> = (0..20 * 9).map(|i| (i as f32) * 0.1).collect();
    let x = DynTensor::from_vec(data, &[1, 20, 9], &Device::Cpu).unwrap();
    let t_out = 600; // 30× upsample (similar to SineGen's 120-300× in production)

    let gpu_result = interp_upsample_gpu(&x, t_out).unwrap();
    let cpu_result = interp_upsample_cpu(&x, 1, t_out, 9).unwrap();

    let gpu_vals = gpu_result.to_flat_vec::<f32>().unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_eq!(gpu_vals.len(), cpu_vals.len());
    for (i, (&g, &c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        assert!(
            (g - c).abs() < 1e-5,
            "upsample parity mismatch at {i}: gpu={g}, cpu={c} (diff={:.2e})",
            (g - c).abs()
        );
    }
}

// -- Phase precision tests (#2649) --------------------------------------------

/// Verify SineGen phase accuracy at T=24000 audio samples (1 second).
///
/// At T_frames=80 (24000/300), Kahan cumsum error is negligible.
/// CPU uses f64 accumulation; GPU uses Kahan f32 (tested separately in
/// dyn_tensor_metal_cumsum_tests). This test verifies no NaN/Inf propagation
/// and output bounded by sine_amp=0.1.
#[test]
fn test_sinegen_phase_accuracy_1_second() {
    let sg = SineGen::new();
    let t_frames = 80; // 1 second at 24kHz / 300upp
    let upp = 300;
    let t_audio = t_frames * upp; // 24000

    let f0 = DynTensor::full(&[1, t_frames, 1], 440.0, DType::F32, &Device::Cpu).unwrap();
    let (sines, voiced, _) = sg.forward(&f0, upp).unwrap();

    assert_eq!(sines.dims(), &[1, t_audio, 9]);
    let vals = sines.to_flat_vec::<f32>().unwrap();

    // No NaN or Inf.
    let non_finite = vals.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "NaN/Inf found in 1-second SineGen output");

    // All values bounded by amplitude 0.1.
    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs <= 0.1 + 1e-6,
        "max amplitude {max_abs} exceeds 0.1 for 1-second output"
    );

    // Voiced mask should be all 1.0 (F0=440 > threshold=10).
    let voiced_vals = voiced.to_flat_vec::<f32>().unwrap();
    assert!(
        voiced_vals.iter().all(|&v| v == 1.0),
        "440Hz should be fully voiced"
    );
}

/// Verify SineGen phase accuracy at T=240000 audio samples (10 seconds).
///
/// Formal analysis (#2649): at T_frames=800, Kahan f32 cumsum error
/// for harmonic 9 at 3960Hz: ~2ε × 800 × 0.165 × 2π × 300 ≈ 0.03 rad.
/// CPU f64 fallback error: effectively zero. This test verifies no NaN/Inf
/// and bounded output at 10-second durations.
#[test]
fn test_sinegen_phase_accuracy_10_seconds() {
    let sg = SineGen::new();
    let t_frames = 800; // 10 seconds at 24kHz / 300upp
    let upp = 300;
    let t_audio = t_frames * upp; // 240000

    let f0 = DynTensor::full(&[1, t_frames, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
    let (sines, _, _) = sg.forward(&f0, upp).unwrap();

    assert_eq!(sines.dims(), &[1, t_audio, 9]);
    let vals = sines.to_flat_vec::<f32>().unwrap();

    let non_finite = vals.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "NaN/Inf found in 10-second SineGen output");

    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs <= 0.1 + 1e-6,
        "max amplitude {max_abs} exceeds 0.1 for 10-second output"
    );
}

/// Verify MAX_SINEGEN_FRAMES runtime guard rejects overlong sequences.
#[test]
fn test_sinegen_max_frames_guard() {
    let sg = SineGen::new();
    let t_frames = MAX_SINEGEN_FRAMES + 1;
    let f0 = DynTensor::full(&[1, t_frames, 1], 200.0, DType::F32, &Device::Cpu).unwrap();
    let err = sg.forward(&f0, 300).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("MAX_SINEGEN_FRAMES"),
        "error should mention MAX_SINEGEN_FRAMES: {msg}"
    );
}

/// Verify MAX_SINEGEN_FRAMES allows the 512-token Kokoro maximum (~40 seconds).
#[test]
fn test_sinegen_max_frames_allows_512_tokens() {
    // 512 tokens × ~80ms average duration = ~41 seconds
    // At 24kHz / 300upp: t_frames = 24000 * 41 / 300 ≈ 3280
    let t_frames = 3280;
    assert!(
        t_frames <= MAX_SINEGEN_FRAMES,
        "MAX_SINEGEN_FRAMES ({MAX_SINEGEN_FRAMES}) must accommodate 512-token max ({t_frames} frames)"
    );
}

// -- fmod_one precision tests -------------------------------------------------

/// Verify `fmod_one` produces x - floor(x) for positive, negative, and boundary values.
#[test]
fn test_fmod_one_precise() {
    let x = DynTensor::from_vec(
        vec![0.0_f32, 0.5, 1.0, 1.7, -0.3, 2.999, 0.001],
        &[1, 7, 1],
        &Device::Cpu,
    )
    .unwrap();
    let result = fmod_one(&x).unwrap();
    let vals = result.to_flat_vec::<f32>().unwrap();
    let expected = [0.0_f32, 0.5, 0.0, 0.7, 0.7, 0.999, 0.001];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "fmod_one[{i}]: got {got}, expected {exp}"
        );
    }
}
