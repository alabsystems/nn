// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Encoder-focused tests for Whisper AudioEncoder.
//!
//! Covers: forward pass with random weights, output shape verification,
//! multi-head attention, positional encoding, layer normalization,
//! mel-to-encoder integration, edge cases, and config validation.

use crate::config::WhisperConfig;
use crate::encoder::AudioEncoder;
use crate::positional::sinusoidal_embedding;
use crate::test_utils::tiny_config;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helper: random tensor via rand crate + DynTensor::from_vec
// (DynTensor::rand requires the `training` feature, unavailable in tests)
// ---------------------------------------------------------------------------

fn rand_tensor(lo: f32, hi: f32, dims: &[usize]) -> DynTensor {
    use rand::RngExt;
    let numel: usize = dims.iter().product();
    let mut rng = rand::rng();
    let data: Vec<f32> = (0..numel)
        .map(|_| {
            let u: f32 = rng.random();
            u * (hi - lo) + lo
        })
        .collect();
    DynTensor::from_vec(data, dims, &cpu()).expect("rand_tensor")
}

/// Insert a random tensor into a HashMap under the given key.
fn insert_rand(tensors: &mut HashMap<String, DynTensor>, key: &str, shape: &[usize]) {
    tensors.insert(key.to_string(), rand_tensor(-0.02, 0.02, shape));
}

/// Insert LN weight=1.0 and bias=0.0 tensors into the map.
fn insert_ln(tensors: &mut HashMap<String, DynTensor>, prefix: &str, d: usize) {
    let dev = &cpu();
    tensors.insert(
        format!("{prefix}.weight"),
        DynTensor::ones(&[d], DType::F32, dev).expect("ones"),
    );
    tensors.insert(
        format!("{prefix}.bias"),
        DynTensor::zeros(&[d], DType::F32, dev).expect("zeros"),
    );
}

// ---------------------------------------------------------------------------
// Helper: build a VarBuilder with small random weights for non-degenerate tests.
// ---------------------------------------------------------------------------

fn random_encoder_vb(config: &WhisperConfig) -> VarBuilder {
    let d = config.d_model;
    let mel = config.num_mel_bins;
    let ffn = config.encoder_ffn_dim;
    let dev = &cpu();
    let p = "model.encoder";

    let mut t: HashMap<String, DynTensor> = HashMap::new();

    // Conv1d stem.
    insert_rand(&mut t, &format!("{p}.conv1.weight"), &[d, mel, 3]);
    insert_rand(&mut t, &format!("{p}.conv1.bias"), &[d]);
    insert_rand(&mut t, &format!("{p}.conv2.weight"), &[d, d, 3]);
    insert_rand(&mut t, &format!("{p}.conv2.bias"), &[d]);

    // Final LayerNorm.
    insert_ln(&mut t, &format!("{p}.layer_norm"), d);

    // Encoder transformer blocks.
    for i in 0..config.encoder_layers {
        let bp = format!("{p}.layers.{i}");

        // Self-attention projections.
        insert_rand(&mut t, &format!("{bp}.self_attn.q_proj.weight"), &[d, d]);
        insert_rand(&mut t, &format!("{bp}.self_attn.q_proj.bias"), &[d]);
        insert_rand(&mut t, &format!("{bp}.self_attn.k_proj.weight"), &[d, d]);
        // k_proj bias: MultiHeadAttention::load uses get_or_zeros, skip.
        insert_rand(&mut t, &format!("{bp}.self_attn.v_proj.weight"), &[d, d]);
        insert_rand(&mut t, &format!("{bp}.self_attn.v_proj.bias"), &[d]);
        insert_rand(&mut t, &format!("{bp}.self_attn.out_proj.weight"), &[d, d]);
        insert_rand(&mut t, &format!("{bp}.self_attn.out_proj.bias"), &[d]);

        // Block layer norms.
        insert_ln(&mut t, &format!("{bp}.self_attn_layer_norm"), d);
        insert_ln(&mut t, &format!("{bp}.final_layer_norm"), d);

        // FFN.
        insert_rand(&mut t, &format!("{bp}.fc1.weight"), &[ffn, d]);
        insert_rand(&mut t, &format!("{bp}.fc1.bias"), &[ffn]);
        insert_rand(&mut t, &format!("{bp}.fc2.weight"), &[d, ffn]);
        insert_rand(&mut t, &format!("{bp}.fc2.bias"), &[d]);
    }

    VarBuilder::from_tensors(t, DType::F32, dev)
}

// ===========================================================================
// Section 1: Encoder forward pass with random weights
// ===========================================================================

#[test]
fn test_encoder_forward_random_weights_nonzero() {
    let config = tiny_config();
    let vb = random_encoder_vb(&config);
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = rand_tensor(-1.0, 1.0, &[1, config.num_mel_bins, 16]);
    let out = encoder.forward(&mel).unwrap();

    // With random weights, output should not be all zeros.
    let flat = out.to_flat_vec::<f32>().unwrap();
    let nonzero_count = flat.iter().filter(|v| v.abs() > 1e-8).count();
    assert!(
        nonzero_count > 0,
        "encoder with random weights should produce non-zero output"
    );
}

#[test]
fn test_encoder_forward_random_weights_finite() {
    let config = tiny_config();
    let vb = random_encoder_vb(&config);
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = rand_tensor(-1.0, 1.0, &[1, config.num_mel_bins, 16]);
    let out = encoder.forward(&mel).unwrap();

    let flat = out.to_flat_vec::<f32>().unwrap();
    for (i, &v) in flat.iter().enumerate() {
        assert!(
            v.is_finite(),
            "encoder output element {i} is not finite: {v}"
        );
    }
}

#[test]
fn test_encoder_forward_multiple_layers() {
    // Test with 3 encoder layers to exercise stacked transformer blocks.
    let config = tiny_config().with_encoder_layers(3);
    let vb = random_encoder_vb(&config);
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = rand_tensor(-1.0, 1.0, &[1, config.num_mel_bins, 16]);
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.rank(), 3);
    assert_eq!(out.dim(0).unwrap(), 1);
    assert_eq!(out.dim(2).unwrap(), config.d_model);

    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

// ===========================================================================
// Section 2: Output shape verification for various input lengths
// ===========================================================================

/// Conv1d stride-2 output length formula for encoder conv2:
/// conv1: stride=1, pad=1, kernel=3 => output_len = mel_len (preserves length)
/// conv2: stride=2, pad=1, kernel=3 => output_len = (mel_len + 2*1 - 3)/2 + 1
fn expected_seq_len(mel_len: usize) -> usize {
    (mel_len + 2 - 3) / 2 + 1
}

#[test]
fn test_encoder_output_shape_mel_len_4() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 4], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.dims(), &[1, expected_seq_len(4), config.d_model]);
}

#[test]
fn test_encoder_output_shape_mel_len_8() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.dims(), &[1, expected_seq_len(8), config.d_model]);
}

#[test]
fn test_encoder_output_shape_mel_len_16() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 16], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.dims(), &[1, expected_seq_len(16), config.d_model]);
}

#[test]
fn test_encoder_output_shape_odd_mel_len() {
    // Odd-length mel input: conv stride-2 rounds down.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 7], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.dims(), &[1, expected_seq_len(7), config.d_model]);
}

#[test]
fn test_encoder_output_shape_systematic() {
    // Verify the shape formula across a range of mel lengths.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());

    for mel_len in [2, 3, 4, 5, 6, 8, 10, 12, 15, 16] {
        let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();
        let mel =
            DynTensor::zeros(&[1, config.num_mel_bins, mel_len], DType::F32, &cpu()).unwrap();
        let out = encoder.forward(&mel).unwrap();

        let expected = expected_seq_len(mel_len);
        assert_eq!(
            out.dim(1).unwrap(),
            expected,
            "mel_len={mel_len}: expected seq_len={expected}, got {}",
            out.dim(1).unwrap()
        );
    }
}

#[test]
fn test_encoder_output_d_model_consistency() {
    // Output dim(2) always matches config.d_model regardless of mel length.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());

    for mel_len in [4, 8, 16] {
        let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();
        let mel =
            DynTensor::zeros(&[1, config.num_mel_bins, mel_len], DType::F32, &cpu()).unwrap();
        let out = encoder.forward(&mel).unwrap();
        assert_eq!(out.dim(2).unwrap(), config.d_model);
    }
}

// ===========================================================================
// Section 3: Multi-head attention within encoder layers
// ===========================================================================

#[test]
fn test_encoder_different_head_counts() {
    // 2 heads (default tiny config).
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

#[test]
fn test_encoder_four_heads() {
    // 4 heads: d_model=16, head_dim=4.
    let config = tiny_config().with_encoder_attention_heads(4);
    assert_eq!(config.encoder_head_dim(), 4);

    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

#[test]
fn test_encoder_no_cache_vs_cached_equivalence() {
    // forward_no_cache should produce the same output as forward for fresh encoder.
    let config = tiny_config();
    let vb = random_encoder_vb(&config);

    let mut cached_encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();
    let no_cache_encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = rand_tensor(-1.0, 1.0, &[1, config.num_mel_bins, 8]);

    let cached_out = cached_encoder.forward(&mel).unwrap();
    let no_cache_out = no_cache_encoder.forward_no_cache(&mel).unwrap();

    assert_eq!(cached_out.dims(), no_cache_out.dims());

    let cached_data = cached_out.to_flat_vec::<f32>().unwrap();
    let no_cache_data = no_cache_out.to_flat_vec::<f32>().unwrap();

    let max_err = cached_data
        .iter()
        .zip(no_cache_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-5,
        "forward vs forward_no_cache max error: {max_err}"
    );
}

// ===========================================================================
// Section 4: Positional encoding correctness
// ===========================================================================

#[test]
fn test_positional_embedding_shape_matches_config() {
    let config = tiny_config();
    let emb =
        sinusoidal_embedding(config.max_source_positions, config.d_model, DType::F32, &cpu())
            .unwrap();
    assert_eq!(
        emb.dims(),
        &[config.max_source_positions, config.d_model]
    );
}

#[test]
fn test_positional_embedding_bounded() {
    // All sinusoidal values must be in [-1, 1].
    let emb = sinusoidal_embedding(100, 16, DType::F32, &cpu()).unwrap();
    let flat = emb.to_flat_vec::<f32>().unwrap();
    for &v in &flat {
        assert!(v.is_finite(), "pos embedding must be finite");
        assert!(
            (-1.0..=1.0).contains(&v),
            "sin/cos values must be in [-1, 1], got {v}"
        );
    }
}

#[test]
fn test_positional_embedding_position_zero() {
    // At position 0: sin(0)=0 for first half, cos(0)=1 for second half.
    let channels = 16;
    let half = channels / 2;
    let emb = sinusoidal_embedding(4, channels, DType::F32, &cpu()).unwrap();
    let flat = emb.to_flat_vec::<f32>().unwrap();

    for (i, &v) in flat.iter().take(half).enumerate() {
        assert!(
            v.abs() < 1e-6,
            "sin at pos 0 dim {i} should be 0, got {v}"
        );
    }
    for i in 0..half {
        assert!(
            (flat[half + i] - 1.0).abs() < 1e-6,
            "cos at pos 0 dim {i} should be 1, got {}",
            flat[half + i]
        );
    }
}

#[test]
fn test_positional_embedding_different_positions_differ() {
    // Embeddings at different positions should differ.
    let channels = 16;
    let emb = sinusoidal_embedding(10, channels, DType::F32, &cpu()).unwrap();
    let flat = emb.to_flat_vec::<f32>().unwrap();

    let pos0 = &flat[0..channels];
    let pos1 = &flat[channels..2 * channels];

    let diff: f32 = pos0
        .iter()
        .zip(pos1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff > 1e-4,
        "positions 0 and 1 should have different embeddings, total diff = {diff}"
    );
}

#[test]
fn test_positional_embedding_narrow_for_encoder() {
    // Encoder narrows the positional embedding to seq_len < max_source_positions.
    let config = tiny_config(); // max_source_positions = 8
    let emb = sinusoidal_embedding(
        config.max_source_positions,
        config.d_model,
        DType::F32,
        &cpu(),
    )
    .unwrap();

    // Narrow to 4 positions (simulating conv output of 4 frames).
    let narrow = emb.narrow(0, 0, 4).unwrap();
    assert_eq!(narrow.dims(), &[4, config.d_model]);
}

// ===========================================================================
// Section 5: Layer normalization in encoder
// ===========================================================================

#[test]
fn test_encoder_layer_norm_normalizes_output() {
    // With random weights and LN weight=1/bias=0, the final LN should
    // produce approximately zero-mean unit-variance along d_model.
    let config = tiny_config();
    let vb = random_encoder_vb(&config);
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = rand_tensor(-1.0, 1.0, &[1, config.num_mel_bins, 8]);
    let out = encoder.forward(&mel).unwrap();

    let flat = out.to_flat_vec::<f32>().unwrap();
    let seq_len = out.dim(1).unwrap();
    let d = config.d_model;

    // Check each position: mean should be near 0, variance near 1.
    for t in 0..seq_len {
        let row: Vec<f32> = (0..d).map(|j| flat[t * d + j]).collect();
        let mean: f32 = row.iter().sum::<f32>() / d as f32;
        let var: f32 = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / d as f32;

        // Loose tolerance because d_model=16 is small.
        assert!(
            mean.abs() < 0.2,
            "position {t}: LN output mean should be near 0, got {mean}"
        );
        assert!(
            (var - 1.0).abs() < 0.5,
            "position {t}: LN output variance should be near 1, got {var}"
        );
    }
}

#[test]
fn test_encoder_zero_weights_produces_finite_output() {
    // Zero-weight encoder: all matmuls produce zero, but LN of zero-vector
    // should still produce finite output (LN handles zero-variance gracefully).
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();

    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

// ===========================================================================
// Section 6: Mel spectrogram feature extraction (integration)
// ===========================================================================

#[test]
fn test_mel_to_encoder_pipeline() {
    // Generate mel from PCM audio, then feed through encoder.
    use crate::audio::{mel_filterbank, pcm_to_mel};

    let config = tiny_config();
    let n_fft = 400;
    let hop = 160;
    let sr = 16_000;

    // Generate 1 second of sine wave at 440 Hz.
    let samples: usize = sr;
    let audio: Vec<f32> = (0..samples)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
        .collect();

    let filters = mel_filterbank(config.num_mel_bins, n_fft, sr);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, config.num_mel_bins).unwrap();

    // mel shape: [1, num_mel_bins, n_frames]
    assert_eq!(mel.dim(0).unwrap(), 1);
    assert_eq!(mel.dim(1).unwrap(), config.num_mel_bins);

    // Feed through encoder. Adjust max_source_positions to fit.
    let n_frames = mel.dim(2).unwrap();
    let encoder_positions = expected_seq_len(n_frames) + 10;
    let test_config = config.with_max_source_positions(encoder_positions);

    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &test_config).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.rank(), 3);
    assert_eq!(out.dim(0).unwrap(), 1);
    assert_eq!(out.dim(2).unwrap(), test_config.d_model);
}

#[test]
fn test_whisper_mel_spectrogram_standard_params() {
    // Verify standard mel spectrogram has expected shape.
    use crate::audio::whisper_mel_spectrogram_for_config;

    let config = tiny_config();
    // Generate 0.5 seconds of audio (padded to 30s by whisper_mel_spectrogram).
    let audio: Vec<f32> = (0..8000).map(|i| (i as f32 * 0.01).sin()).collect();
    let mel = whisper_mel_spectrogram_for_config(&audio, config.num_mel_bins).unwrap();

    assert_eq!(mel.dim(0).unwrap(), 1);
    assert_eq!(mel.dim(1).unwrap(), config.num_mel_bins);
    assert_eq!(mel.dim(2).unwrap(), crate::config::N_FRAMES); // 3000
}

// ===========================================================================
// Section 7: Edge cases
// ===========================================================================

#[test]
fn test_encoder_minimum_mel_length() {
    // mel_len=2 should produce at least 1 frame after stride-2 conv.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 2], DType::F32, &cpu()).unwrap();
    let result = encoder.forward(&mel);
    assert!(
        result.is_ok(),
        "encoder should handle mel_len=2: {result:?}"
    );
    let out = result.unwrap();
    assert!(out.dim(1).unwrap() >= 1, "should produce at least 1 frame");
}

#[test]
fn test_encoder_mel_len_3() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 3], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.dim(1).unwrap(), expected_seq_len(3));
}

#[test]
fn test_encoder_max_boundary() {
    // max_source_positions = 8 in tiny_config. Produce exactly 8 frames.
    // (mel_len + 2 - 3)/2 + 1 = 8 => mel_len = 15.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 15], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.dim(1).unwrap(), 8);
}

#[test]
fn test_encoder_very_short_pcm() {
    // Very short audio: 100 samples at 16 kHz (6.25 ms).
    use crate::audio::{mel_filterbank, pcm_to_mel};

    let config = tiny_config();
    let n_fft = 400;
    let hop = 160;
    let sr = 16_000;

    let audio: Vec<f32> = (0..100).map(|i| (i as f32 * 0.05).sin()).collect();
    let filters = mel_filterbank(config.num_mel_bins, n_fft, sr);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, config.num_mel_bins).unwrap();

    let n_frames = mel.dim(2).unwrap();
    let encoder_positions = expected_seq_len(n_frames) + 5;
    let test_config = config.with_max_source_positions(encoder_positions);

    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &test_config).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.rank(), 3);
    assert!(out.dim(1).unwrap() >= 1);
}

#[test]
fn test_encoder_one_second_audio() {
    // 1 second of audio at 16 kHz produces many mel frames.
    use crate::audio::{mel_filterbank, pcm_to_mel};

    let config = tiny_config();
    let n_fft = 400;
    let hop = 160;
    let sr = 16_000;

    let audio: Vec<f32> = (0..sr).map(|i| (i as f32 * 0.01).sin()).collect();
    let filters = mel_filterbank(config.num_mel_bins, n_fft, sr);
    let mel = pcm_to_mel(&audio, &filters, n_fft, hop, config.num_mel_bins).unwrap();

    let n_frames = mel.dim(2).unwrap();
    assert!(n_frames > 50, "1s audio should produce many frames");

    let encoder_positions = expected_seq_len(n_frames) + 5;
    let test_config = config.with_max_source_positions(encoder_positions);

    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &test_config).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.rank(), 3);
    assert_eq!(out.dim(2).unwrap(), test_config.d_model);
}

#[test]
fn test_encoder_batch_size_two() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    // Batch of 2 mel spectrograms.
    let mel = DynTensor::zeros(&[2, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.dim(0).unwrap(), 2);
    assert_eq!(out.dim(1).unwrap(), expected_seq_len(8));
    assert_eq!(out.dim(2).unwrap(), config.d_model);
}

// ===========================================================================
// Section 8: Config validation (invalid params should error)
// ===========================================================================

#[test]
fn test_config_zero_encoder_layers_loads() {
    // Zero encoder_layers is not rejected by validate(). Model should still
    // load and produce output (just conv + pos + LN, no transformer blocks).
    let config = tiny_config().with_encoder_layers(0);
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.rank(), 3);
}

#[test]
fn test_config_zero_encoder_heads_rejected() {
    let config = tiny_config().with_encoder_attention_heads(0);
    let result = config.validate();
    assert!(
        result.is_err(),
        "zero encoder_attention_heads should fail validation"
    );
}

#[test]
fn test_config_d_model_not_divisible_by_heads() {
    // d_model=16 not divisible by 3 heads.
    let config = tiny_config().with_encoder_attention_heads(3);
    let result = config.validate();
    assert!(
        result.is_err(),
        "d_model not divisible by heads should fail validation"
    );
}

#[test]
fn test_config_head_dim_calculation() {
    let config = tiny_config();
    assert_eq!(config.encoder_head_dim(), 8); // 16 / 2
    let config4 = tiny_config().with_encoder_attention_heads(4);
    assert_eq!(config4.encoder_head_dim(), 4); // 16 / 4
}

#[test]
fn test_config_zero_d_model_rejected() {
    let config = tiny_config().with_d_model(0);
    let result = config.validate();
    assert!(result.is_err(), "zero d_model should fail validation");
}

#[test]
fn test_config_zero_mel_bins_rejected() {
    let config = tiny_config().with_num_mel_bins(0);
    let result = config.validate();
    assert!(result.is_err(), "zero num_mel_bins should fail validation");
}

#[test]
fn test_config_zero_encoder_ffn_dim_rejected() {
    let config = tiny_config().with_encoder_ffn_dim(0);
    let result = config.validate();
    assert!(
        result.is_err(),
        "zero encoder_ffn_dim should fail validation"
    );
}

// ===========================================================================
// Additional: determinism, dtype handling, cache reset
// ===========================================================================

#[test]
fn test_encoder_deterministic_with_same_weights() {
    // Running forward twice on the same encoder with same input should
    // produce identical results (encoder resets cache each forward call).
    let config = tiny_config();
    let vb = random_encoder_vb(&config);
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = rand_tensor(-1.0, 1.0, &[1, config.num_mel_bins, 8]);

    let out1 = encoder.forward(&mel).unwrap();
    let out2 = encoder.forward(&mel).unwrap();

    let d1 = out1.to_flat_vec::<f32>().unwrap();
    let d2 = out2.to_flat_vec::<f32>().unwrap();

    let max_err = d1
        .iter()
        .zip(d2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-6,
        "encoder should be deterministic, max error: {max_err}"
    );
}

#[test]
fn test_encoder_bf16_dtype() {
    // BF16 model should load and produce finite output.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::BF16, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::BF16, &cpu()).unwrap();
    let out = encoder.forward(&mel).unwrap();

    assert_eq!(out.rank(), 3);
    let flat = out.to_flat_vec::<f32>().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()));
}

#[test]
fn test_encoder_reset_cache() {
    // reset_cache should not panic on a fresh encoder.
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut encoder = AudioEncoder::load(vb.pp("model.encoder"), &config).unwrap();

    encoder.reset_cache(); // Should not panic.

    // Forward, then reset, then forward again should work.
    let mel = DynTensor::zeros(&[1, config.num_mel_bins, 8], DType::F32, &cpu()).unwrap();
    let _ = encoder.forward(&mel).unwrap();
    encoder.reset_cache();
    let out = encoder.forward(&mel).unwrap();
    assert_eq!(out.rank(), 3);
}
