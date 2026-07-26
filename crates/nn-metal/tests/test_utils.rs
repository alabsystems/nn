#![allow(deprecated)]
// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities for nn-metal integration tests.
//!
//! Provides deterministic PRNG, Metal setup helpers, precision assertion,
//! and CPU reference implementations for differential testing.

#![allow(dead_code, unreachable_pub)]

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;
use nn_dsl::{within_differential_budget, PrecisionContract, PrecisionTier, ScalarType};
use nn_metal::{register_metal_dyn_backend, MetalBackend, PipelineCache};
use nn_models::kokoro_streaming::KokoroStreamConfig;

// ---------------------------------------------------------------------------
// Deterministic PRNG — re-exported from nn_core::test_prng (#1411)
// ---------------------------------------------------------------------------

pub use nn_core::test_prng::rand_f32_vec;

/// Derive a smaller streaming overlap for short synthetic Kokoro test chunks.
///
/// The mini Kokoro fixtures emit much shorter PCM than production, so the
/// production default crossfade can exceed the generated chunk length.
pub(crate) fn short_stream_config_for_pcm_len(pcm_len: usize) -> KokoroStreamConfig {
    // Preserve the production target (20ms / 480 samples) when possible, and
    // clamp only when short fixtures would violate the chunk-length guard.
    let default_cf = KokoroStreamConfig::default().crossfade_samples;
    let max_safe_cf = pcm_len.saturating_sub(1).max(1);
    let crossfade_samples = default_cf.min(max_safe_cf);
    KokoroStreamConfig::new(crossfade_samples).expect("valid short-chunk crossfade")
}

/// Generate a deterministic vector of `half::f16` values in `[lo, hi]`.
///
/// Uses the same PRNG as `rand_f32_vec` then converts to f16. Values are
/// clamped to the f16 representable range before conversion.
pub(crate) fn rand_f16_vec(seed: u64, count: usize, lo: f32, hi: f32) -> Vec<half::f16> {
    rand_f32_vec(seed, count, lo, hi)
        .into_iter()
        .map(half::f16::from_f32)
        .collect()
}

// ---------------------------------------------------------------------------
// Metal setup
// ---------------------------------------------------------------------------

pub(crate) fn metal_setup() -> PipelineCache {
    // Initialize the global Metal backend so gpu_scope::get_or_create_batch()
    // can find the global context for lazy command buffer batching (#2424).
    let _ = MetalBackend::init().expect("Metal device required");
    PipelineCache::new_global().expect("Metal global cache")
}

/// Initialize Metal backend and register DynTensor GPU dispatch.
///
/// Call once at the start of each GPU forward test. Idempotent.
pub(crate) fn gpu_init() {
    let _ = MetalBackend::init();
    register_metal_dyn_backend();
}

/// Compare GPU and CPU DynTensor results element-wise within tolerance.
///
/// Moves the GPU tensor to CPU, extracts flat f32 vectors, and asserts
/// that every element pair is within `tol`. Panics with a diagnostic
/// message showing the first violation.
pub(crate) fn assert_gpu_cpu_close(
    gpu_result: &DynTensor,
    cpu_result: &DynTensor,
    tol: f32,
    label: &str,
) {
    let gpu_vals = gpu_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let cpu_vals = cpu_result.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        gpu_vals.len(),
        cpu_vals.len(),
        "{label}: length mismatch (gpu={}, cpu={})",
        gpu_vals.len(),
        cpu_vals.len()
    );
    for (i, (g, c)) in gpu_vals.iter().zip(cpu_vals.iter()).enumerate() {
        let diff = (g - c).abs();
        assert!(
            diff <= tol,
            "{label}[{i}]: gpu={g} cpu={c} diff={diff} > {tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Precision assertion
// ---------------------------------------------------------------------------

/// Per-element check using precision contract.
pub(crate) fn assert_within_budget(name: &str, gpu: &[f32], cpu: &[f32]) {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    for (i, (&r, &g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            within_differential_budget(r, g, contract),
            "{name}[{i}]: out of budget — cpu={r}, gpu={g}, delta={:.6e}",
            (r - g).abs(),
        );
    }
}

/// Per-element check for f16 dispatch results.
///
/// Converts f16 GPU output to f32 for comparison against f32 CPU reference,
/// using the F16 precision contract (wider tolerance: 1e-2 abs for Normal tier).
pub(crate) fn assert_within_budget_f16(name: &str, gpu: &[half::f16], cpu_f32: &[f32]) {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F16);
    for (i, (&g_f16, &r)) in gpu.iter().zip(cpu_f32.iter()).enumerate() {
        let g = g_f16.to_f32();
        assert!(
            within_differential_budget(r, g, contract),
            "{name}[{i}]: out of budget — cpu={r}, gpu_f16={g}, delta={:.6e}",
            (r - g).abs(),
        );
    }
}

// ---------------------------------------------------------------------------
// Bounds assertion (NY proved bounds)
// ---------------------------------------------------------------------------

/// Assert every GPU output element falls within proved bounds (with ULP margin).
pub(crate) fn assert_gpu_within_bounds(
    label: &str,
    gpu_out: &[f32],
    proved_lo: &ndarray::ArrayD<f32>,
    proved_hi: &ndarray::ArrayD<f32>,
) {
    let lo_slice = proved_lo.as_slice().expect("contiguous lower");
    let hi_slice = proved_hi.as_slice().expect("contiguous upper");
    assert_eq!(
        gpu_out.len(),
        lo_slice.len(),
        "{label}: GPU output length {} != bounds length {}",
        gpu_out.len(),
        lo_slice.len()
    );
    for (i, &g) in gpu_out.iter().enumerate() {
        let lo = lo_slice[i];
        let hi = hi_slice[i];
        let ulp_margin = (hi - lo).abs() * f32::EPSILON;
        assert!(
            g >= lo - ulp_margin && g <= hi + ulp_margin,
            "{label} GPU output[{i}] violates proved bounds: \
             gpu={g}, proved=[{lo}, {hi}]",
        );
    }
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU reference for 1D convolution (groups=1, dilation=1).
///
/// Input layout: `[in_channels, in_length]`
/// Weight layout: `[out_channels, in_channels, kernel_size]`
/// Output layout: `[out_channels, out_length]`
pub(crate) fn conv1d_ref(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    let mut output = vec![0.0_f32; out_channels * out_length];
    for oc in 0..out_channels {
        for ot in 0..out_length {
            let mut sum = 0.0_f32;
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let it = ot * stride + k;
                    if it >= padding && it - padding < in_length {
                        let in_idx = ic * in_length + (it - padding);
                        let w_idx = (oc * in_channels + ic) * kernel_size + k;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
            if let Some(b) = bias {
                sum += b[oc];
            }
            output[oc * out_length + ot] = sum;
        }
    }
    output
}

/// CPU reference for transposed 1D convolution (groups=1, no dilation).
///
/// Input layout: `[in_channels, in_length]`
/// Weight layout: `[in_channels, out_channels, kernel_size]`
/// Output layout: `[out_channels, out_length]`
/// where `out_length = (in_length - 1) * stride + kernel_size - 2 * padding`
pub(crate) fn conv_transpose_1d_ref(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    let out_length = (in_length - 1) * stride + kernel_size - 2 * padding;
    let mut output = vec![0.0_f32; out_channels * out_length];
    for ic in 0..in_channels {
        for il in 0..in_length {
            let x = input[ic * in_length + il];
            for oc in 0..out_channels {
                for k in 0..kernel_size {
                    let out_pos = il * stride + k;
                    if out_pos >= padding && out_pos - padding < out_length {
                        let w_idx = (ic * out_channels + oc) * kernel_size + k;
                        output[oc * out_length + (out_pos - padding)] += x * weight[w_idx];
                    }
                }
            }
        }
    }
    if let Some(b) = bias {
        for oc in 0..out_channels {
            for ol in 0..out_length {
                output[oc * out_length + ol] += b[oc];
            }
        }
    }
    output
}

/// CPU reference for zero-padding a 1D signal on left and right.
///
/// Input layout: `[channels, in_length]`
/// Output layout: `[channels, in_length + pad_left + pad_right]`
pub(crate) fn zero_pad_1d_ref(
    input: &[f32],
    channels: usize,
    in_length: usize,
    pad_left: usize,
    pad_right: usize,
) -> Vec<f32> {
    let out_length = in_length + pad_left + pad_right;
    let mut output = vec![0.0_f32; channels * out_length];
    for c in 0..channels {
        for t in 0..in_length {
            output[c * out_length + pad_left + t] = input[c * in_length + t];
        }
    }
    output
}

/// CPU reference for causal Conv1d (zero-pad-left then conv1d with padding=0).
///
/// Equivalent to `F.pad(x, (pad_left, 0))` + `conv1d(x, weight, padding=0)`.
pub(crate) fn causal_conv1d_ref(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    dilation: usize,
) -> Vec<f32> {
    let pad_left = (kernel_size - 1) * dilation;
    let padded = zero_pad_1d_ref(input, in_channels, in_length, pad_left, 0);
    let padded_length = in_length + pad_left;
    // Dilated conv1d reference: effective_kernel = dilation*(kernel_size-1)+1
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let out_length = (padded_length - effective_kernel) / stride + 1;
    let mut output = vec![0.0_f32; out_channels * out_length];
    for oc in 0..out_channels {
        for ot in 0..out_length {
            let mut sum = 0.0_f32;
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let it = ot * stride + k * dilation;
                    if it < padded_length {
                        let in_idx = ic * padded_length + it;
                        let w_idx = (oc * in_channels + ic) * kernel_size + k;
                        sum += padded[in_idx] * weight[w_idx];
                    }
                }
            }
            if let Some(b) = bias {
                sum += b[oc];
            }
            output[oc * out_length + ot] = sum;
        }
    }
    output
}

/// CPU reference for GLU: split input along axis 0, apply sigmoid to gate half,
/// multiply data * sigmoid(gate).
pub(crate) fn glu_ref(input: &[f32], channels_2x: usize, time: usize) -> Vec<f32> {
    let half = channels_2x / 2;
    let mut output = vec![0.0_f32; half * time];
    for c in 0..half {
        for t in 0..time {
            let data = input[c * time + t];
            let gate = input[(c + half) * time + t];
            let sig = 1.0 / (1.0 + (-gate).exp());
            output[c * time + t] = data * sig;
        }
    }
    output
}

/// CPU reference for matrix multiplication: `C = A @ B` or `C = A @ B^T`.
///
/// Left layout: `[M, K]` (row-major).
/// Right layout: `[K, N]` (no transpose) or `[N, K]` (transpose_right).
/// Output layout: `[M, N]` (row-major).
/// Optional scale factor multiplied into the result.
pub(crate) fn matmul_ref(
    left: &[f32],
    right: &[f32],
    m: usize,
    k: usize,
    n: usize,
    transpose_right: bool,
    scale: Option<f32>,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for kk in 0..k {
                let right_val = if transpose_right {
                    right[j * k + kk]
                } else {
                    right[kk * n + j]
                };
                sum += left[i * k + kk] * right_val;
            }
            if let Some(s) = scale {
                sum *= s;
            }
            output[i * n + j] = sum;
        }
    }
    output
}

/// CPU reference for linear (fully-connected) layer: `y = x @ W^T + b`.
///
/// Input layout: `[batch_size, in_features]` (row-major).
/// Weight layout: `[out_features, in_features]` (row-major).
/// Output layout: `[batch_size, out_features]` (row-major).
pub(crate) fn linear_ref(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    batch_size: usize,
    in_features: usize,
    out_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; batch_size * out_features];
    for row in 0..batch_size {
        for col in 0..out_features {
            let mut sum = 0.0_f32;
            for k in 0..in_features {
                sum += input[row * in_features + k] * weight[col * in_features + k];
            }
            if let Some(b) = bias {
                sum += b[col];
            }
            output[row * out_features + col] = sum;
        }
    }
    output
}
