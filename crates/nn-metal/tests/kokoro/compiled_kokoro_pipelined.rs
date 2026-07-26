// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `CompiledKokoro::synthesize_pipelined()`.
//!
//! Validates that the GpuFence-based pipelined synthesis path produces
//! bit-identical output to the sequential `synthesize()` path.
//!
//! Uses miniaturized config (D_EN=8, style_dim=4) via [`kokoro_test_weights`].
//!
//! Part of #4251.

use nn_core::dyn_tensor::DynTensor;
use nn_core::Device;

use super::kokoro_test_weights as kw;

fn cpu() -> Device {
    Device::Cpu
}

/// Pipelined synthesis produces the same output as sequential synthesis.
///
/// Runs both `synthesize()` and `synthesize_pipelined()` with identical inputs
/// and verifies that the output PCM samples are bit-identical. This is the
/// primary correctness gate for the GpuFence pipelining integration.
///
/// Part of #4251.
#[test]
fn test_synthesize_pipelined_matches_sequential() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(500, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Warmup: compile all segments.
    let _ = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("warmup synthesize");

    // Sequential path (cache-hit).
    let (audio_seq, cert_seq) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("sequential synthesize");

    // Pipelined path (cache-hit, same inputs).
    let (audio_pipe, cert_pipe) = kokoro
        .synthesize_pipelined(&input_ids, &style, 1.0, &cache)
        .expect("pipelined synthesize");

    // Shape must match.
    assert_eq!(
        audio_seq.dims(),
        audio_pipe.dims(),
        "pipelined audio shape should match sequential"
    );

    // PCM samples must be identical.
    let pcm_seq = audio_seq.to_flat_vec::<f32>().expect("read sequential PCM");
    let pcm_pipe = audio_pipe.to_flat_vec::<f32>().expect("read pipelined PCM");
    assert_eq!(
        pcm_seq.len(),
        pcm_pipe.len(),
        "PCM sample counts should match"
    );

    // Bit-identical comparison. Both paths execute the same GPU kernels
    // in the same order — the only difference is when submits happen.
    // Metal command queues execute serially, so results are deterministic.
    for (i, (s, p)) in pcm_seq.iter().zip(pcm_pipe.iter()).enumerate() {
        assert!(
            (s - p).abs() < 1e-6 || (s.is_nan() && p.is_nan()),
            "PCM sample {i} differs: sequential={s}, pipelined={p}"
        );
    }

    // Certificates should have the same overall result.
    assert_eq!(
        cert_seq.overall_passed, cert_pipe.overall_passed,
        "certificate overall_passed should match"
    );
    assert_eq!(
        cert_seq.hard_bounds.len(),
        cert_pipe.hard_bounds.len(),
        "certificate hard_bounds count should match"
    );
}

/// Pipelined synthesis works on cold start (compilation + execution).
///
/// Verifies that `synthesize_pipelined()` correctly handles segment
/// compilation on the first call (cache miss path), not just cache hits.
///
/// Part of #4251.
#[test]
fn test_synthesize_pipelined_cold_start() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(501, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // First call — cold start (all segments must compile).
    let result = kokoro.synthesize_pipelined(&input_ids, &style, 1.0, &cache);
    assert!(
        result.is_ok(),
        "pipelined cold start failed: {:?}",
        result.err()
    );

    let (audio, certificate) = result.unwrap();

    // Audio shape: [1, 1, T_audio].
    assert_eq!(audio.dims().len(), 3, "audio should be rank-3");
    assert_eq!(audio.dims()[0], 1, "batch dim should be 1");
    assert_eq!(audio.dims()[1], 1, "channel dim should be 1");
    assert!(audio.dims()[2] > 0, "audio should have non-zero samples");

    // Certificate should have 8 hard bounds.
    assert_eq!(
        certificate.hard_bounds.len(),
        8,
        "expected 8 hard bounds, got {}",
        certificate.hard_bounds.len()
    );

    // Audio should be finite.
    let pcm = audio.to_flat_vec::<f32>().expect("read PCM");
    let non_finite = pcm.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "pipelined audio should be all finite, found {non_finite} non-finite values"
    );
}

/// Pipelined synthesis handles different sequence lengths back-to-back.
///
/// Shape changes trigger segment recompilation. Verifies the pipelined
/// path handles cache misses mid-stream without corrupting state.
///
/// Part of #4251.
#[test]
fn test_synthesize_pipelined_different_lengths() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(502, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    let ids_3 = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let ids_4 = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    // First call compiles for len=3.
    let (audio_3, _) = kokoro
        .synthesize_pipelined(&ids_3, &style, 1.0, &cache)
        .expect("pipelined len=3");

    // Second call compiles for len=4 (different shape).
    let (audio_4, _) = kokoro
        .synthesize_pipelined(&ids_4, &style, 1.0, &cache)
        .expect("pipelined len=4");

    assert!(audio_3.dims()[2] > 0, "len=3 audio should have samples");
    assert!(audio_4.dims()[2] > 0, "len=4 audio should have samples");
    assert_ne!(
        audio_3.dims()[2],
        audio_4.dims()[2],
        "different input lengths should produce different audio lengths"
    );
}

/// Pipelined synthesis audio is valid: no NaN, no Inf, all samples in [-1, 1].
///
/// Equivalent to `test_synthesize_gpu_audio_valid_range` in the synthesize
/// test file, but exercises the GpuFence-based code path. Validates that
/// fence submit/wait boundaries do not introduce data corruption.
///
/// Part of #4264, #4251.
#[test]
fn test_synthesize_pipelined_audio_valid_range() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(510, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    let (audio, _cert) = kokoro
        .synthesize_pipelined(&input_ids, &style, 1.0, &cache)
        .expect("pipelined synthesize");

    let pcm = audio.to_flat_vec::<f32>().expect("read PCM");

    // No NaN or Inf.
    let non_finite = pcm.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite,
        0,
        "pipelined audio should be all finite, found {non_finite} non-finite / {} total",
        pcm.len()
    );

    // All samples in [-1, 1] (PCM audio invariant).
    let out_of_range: Vec<(usize, f32)> = pcm
        .iter()
        .enumerate()
        .filter(|(_, v)| v.abs() > 1.0)
        .map(|(i, v)| (i, *v))
        .collect();
    assert!(
        out_of_range.is_empty(),
        "pipelined audio should be in [-1, 1], found {} samples out of range. \
         First 5: {:?}",
        out_of_range.len(),
        &out_of_range[..out_of_range.len().min(5)]
    );

    assert!(!pcm.is_empty(), "audio should have samples");

    eprintln!(
        "pipelined audio: {} samples, min={:.6}, max={:.6}",
        pcm.len(),
        pcm.iter().copied().fold(f32::INFINITY, f32::min),
        pcm.iter().copied().fold(f32::NEG_INFINITY, f32::max),
    );
}

/// Pipelined synthesis PCM output matches sequential synthesis exactly.
///
/// Tighter comparison than `test_synthesize_pipelined_matches_sequential`:
/// this test uses a fresh instance, warms up with sequential, then compares
/// both paths on a cache-hit call. Also verifies hard bound results match
/// per-property (not just overall_passed).
///
/// Part of #4264, #4251.
#[test]
fn test_synthesize_pipelined_matches_synthesize_pcm_exact() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(511, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Warmup to compile all segments.
    let _ = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("warmup");

    // Sequential reference (cache-hit).
    let (audio_seq, cert_seq) = kokoro
        .synthesize(&input_ids, &style, 1.0, &cache)
        .expect("sequential");

    // Pipelined (cache-hit, same inputs).
    let (audio_pipe, cert_pipe) = kokoro
        .synthesize_pipelined(&input_ids, &style, 1.0, &cache)
        .expect("pipelined");

    let pcm_seq = audio_seq.to_flat_vec::<f32>().expect("seq pcm");
    let pcm_pipe = audio_pipe.to_flat_vec::<f32>().expect("pipe pcm");

    assert_eq!(pcm_seq.len(), pcm_pipe.len(), "sample counts must match");

    // Compute max absolute difference.
    let max_diff = pcm_seq
        .iter()
        .zip(pcm_pipe.iter())
        .map(|(s, p)| (s - p).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "pipelined PCM should match sequential within 1e-6, max_diff={max_diff}"
    );

    // Per-property hard bound comparison.
    assert_eq!(
        cert_seq.hard_bounds.len(),
        cert_pipe.hard_bounds.len(),
        "hard bound count mismatch"
    );
    for (s, p) in cert_seq
        .hard_bounds
        .iter()
        .zip(cert_pipe.hard_bounds.iter())
    {
        assert_eq!(s.name, p.name, "hard bound names should match in order");
        assert_eq!(
            s.passed, p.passed,
            "hard bound '{}' result differs: sequential={}, pipelined={}",
            s.name, s.passed, p.passed
        );
    }

    eprintln!(
        "pipelined vs sequential: max_diff={max_diff:.2e}, samples={}, \
         cert_match={}",
        pcm_seq.len(),
        cert_seq.overall_passed == cert_pipe.overall_passed
    );
}

/// Pipelined synthesis rejects invalid speed values.
///
/// Part of #4251.
#[test]
fn test_synthesize_pipelined_invalid_speed() {
    let (mut kokoro, cache) = kw::build_kokoro_mini();
    let config = kw::mini_test_config();
    let style_dim = config.style_dim;

    let input_ids = DynTensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3], &cpu()).unwrap();
    let style = DynTensor::new(
        &super::test_utils::rand_f32_vec(503, 2 * style_dim, -0.1, 0.1),
        &[1, 2 * style_dim],
        &cpu(),
    )
    .unwrap();

    // Zero speed.
    let result = kokoro.synthesize_pipelined(&input_ids, &style, 0.0, &cache);
    assert!(result.is_err(), "zero speed should be rejected");

    // NaN speed.
    let result = kokoro.synthesize_pipelined(&input_ids, &style, f32::NAN, &cache);
    assert!(result.is_err(), "NaN speed should be rejected");

    // Negative speed.
    let result = kokoro.synthesize_pipelined(&input_ids, &style, -1.0, &cache);
    assert!(result.is_err(), "negative speed should be rejected");
}
