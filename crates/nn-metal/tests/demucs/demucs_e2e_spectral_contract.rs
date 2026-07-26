// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Synthetic-weight HTDemucs spectral (dual-branch) contract tests.
//!
//! Exercises the full dual-branch forward pipeline (temporal encoder + spectral
//! encoder → transformer → temporal decoder + spectral decoder → iSTFT → sum)
//! with deterministic synthetic weights. Validates:
//! - Spectral branch construction succeeds with `new_with_spectral()`
//! - `forward_with_stft()` produces finite, correct-shape output
//! - `forward_gpu_with_stft()` CPU/GPU parity
//! - Spectral branch contributes non-zero output (not degenerate)
//!
//! Complements `demucs_e2e_contract.rs` which tests temporal-only mode.
//!
//! Part of #1745 — HTDemucs dual-branch spectral e2e test gap.

use super::demucs_e2e_spectral_helpers::{deterministic_noise, make_htdemucs_spectral_weights};
use nn_metal::{HTDemucs, MetalBackend, PipelineCache};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AUDIO_CHANNELS: usize = 2;
const OUTPUT_CHANNELS: usize = 8;
const SPECTRAL_INPUT_CHANNELS: usize = 4;

/// Use stft_f=2048 (production default) to match hardcoded SPECTRAL_BOTTLENECK_F=8.
/// Frequency downsample: 2048 → 512 → 128 → 32 → 8 (bottleneck).
const STFT_F: usize = 2048;

/// Audio temporal dimension — use 1024 for a more reasonable bottleneck_t=4.
const AUDIO_T: usize = 1024;

/// Compute bottleneck_t matching htdemucs_helpers.rs logic.
fn compute_bottleneck_t(audio_t: usize) -> usize {
    let mut t = audio_t;
    for _ in 0..4 {
        if !t.is_multiple_of(4) {
            t += 4 - (t % 4);
        }
        t = (t + 2 * 2 - 8) / 4 + 1;
    }
    t
}

// ---------------------------------------------------------------------------
// Contract tests
// ---------------------------------------------------------------------------

/// AC1: Construction with `new_with_spectral()` succeeds.
#[test]
fn contract_spectral_construction() {
    let stft_t = compute_bottleneck_t(AUDIO_T);
    let weights = make_htdemucs_spectral_weights();
    let model =
        HTDemucs::new_with_spectral(weights, AUDIO_T, STFT_F, stft_t).expect("construction");

    assert!(model.has_spectral(), "spectral branch should be enabled");
    assert_eq!(model.audio_t(), AUDIO_T);
}

/// AC2: `forward_with_stft()` produces finite, correctly-shaped output.
#[test]
fn contract_spectral_forward_with_stft() {
    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());

    let stft_t = compute_bottleneck_t(AUDIO_T);
    let weights = make_htdemucs_spectral_weights();
    let model =
        HTDemucs::new_with_spectral(weights, AUDIO_T, STFT_F, stft_t).expect("construction");

    let audio = deterministic_noise(AUDIO_CHANNELS * AUDIO_T);
    let stft_mag = deterministic_noise(SPECTRAL_INPUT_CHANNELS * STFT_F * stft_t);

    let output = model
        .forward_with_stft(&cache, &audio, &stft_mag)
        .expect("forward_with_stft should succeed");

    // Output shape: [OUTPUT_CHANNELS, audio_t] = [8, 1024] flattened.
    let expected_len = OUTPUT_CHANNELS * AUDIO_T;
    assert_eq!(
        output.len(),
        expected_len,
        "output length mismatch: expected {expected_len}, got {}",
        output.len()
    );

    let non_finite = output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "{non_finite} non-finite values in spectral forward output"
    );

    let all_zero = output.iter().all(|&v| v == 0.0);
    assert!(
        !all_zero,
        "spectral forward should produce non-trivial output"
    );
}

/// AC3: GPU/CPU parity for `forward_gpu_with_stft()`.
#[test]
fn contract_spectral_gpu_cpu_parity() {
    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());

    let stft_t = compute_bottleneck_t(AUDIO_T);
    let weights = make_htdemucs_spectral_weights();
    let model =
        HTDemucs::new_with_spectral(weights, AUDIO_T, STFT_F, stft_t).expect("construction");

    let audio = deterministic_noise(AUDIO_CHANNELS * AUDIO_T);
    let stft_mag = deterministic_noise(SPECTRAL_INPUT_CHANNELS * STFT_F * stft_t);

    let cpu_output = model
        .forward_with_stft(&cache, &audio, &stft_mag)
        .expect("CPU forward_with_stft");

    let gpu_output = model
        .forward_gpu_with_stft(&cache, &audio, &stft_mag)
        .expect("GPU forward_gpu_with_stft");

    assert_eq!(
        cpu_output.len(),
        gpu_output.len(),
        "CPU/GPU output length mismatch"
    );

    let cpu_non_finite = cpu_output.iter().filter(|v| !v.is_finite()).count();
    let gpu_non_finite = gpu_output.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        cpu_non_finite, 0,
        "{cpu_non_finite} non-finite in CPU output"
    );
    assert_eq!(
        gpu_non_finite, 0,
        "{gpu_non_finite} non-finite in GPU output"
    );

    let max_abs_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    // Allow slightly larger tolerance than temporal-only tests because the
    // spectral branch has more computation stages (iSTFT, spectral decoder).
    assert!(
        max_abs_diff < 1e-2,
        "CPU/GPU max absolute difference {max_abs_diff} exceeds 1e-2"
    );
}

/// AC4: Spectral branch contributes non-zero output.
///
/// Compares temporal-only forward (no STFT input) with dual-branch forward
/// (with STFT input). The difference should be non-zero, proving the spectral
/// branch actually contributes to the final result.
#[test]
fn contract_spectral_branch_contributes() {
    let backend = match MetalBackend::init() {
        Ok(b) => b,
        Err(_) => return,
    };
    let cache = PipelineCache::new(backend.context().clone());

    let stft_t = compute_bottleneck_t(AUDIO_T);
    let weights = make_htdemucs_spectral_weights();
    let model =
        HTDemucs::new_with_spectral(weights, AUDIO_T, STFT_F, stft_t).expect("construction");

    let audio = deterministic_noise(AUDIO_CHANNELS * AUDIO_T);
    let stft_mag = deterministic_noise(SPECTRAL_INPUT_CHANNELS * STFT_F * stft_t);

    // Temporal-only forward (no STFT data — zeros fed to transformer spectral branch).
    let temporal_only = model
        .forward(&cache, &audio)
        .expect("temporal-only forward");

    // Dual-branch forward (STFT data fed through spectral encoder + decoder + iSTFT).
    let dual_branch = model
        .forward_with_stft(&cache, &audio, &stft_mag)
        .expect("dual-branch forward");

    assert_eq!(temporal_only.len(), dual_branch.len());

    // The spectral branch should add a non-zero contribution.
    let max_diff = temporal_only
        .iter()
        .zip(dual_branch.iter())
        .map(|(t, d)| (t - d).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff > 1e-10,
        "spectral branch should contribute non-zero output, but max diff = {max_diff}"
    );
}
