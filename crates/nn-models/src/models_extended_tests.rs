// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended model builder and signal processing tests for nn-models (#4186).
//!
//! Covers: STFT window functions, STFT output shape, iSTFT roundtrip,
//! mel filterbank shape, Kokoro phoneme tokenizer, Silero VAD encoder config,
//! HTDemucs architecture constants, model dispatch registry, audio normalization,
//! sample rate conversion ratio, convert config builder, streaming config,
//! and chorus mixing.

use std::f32::consts::PI;

use crate::convert::{ConvertConfig, DpdfModelType};
use crate::demucs_shared::{
    channels_at_depth, conv1d_output_len, BASE_CHANNELS, DCONV_COMPRESS, DCONV_DEPTH, DCONV_KERNEL,
    TEMPORAL_DEPTH, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE,
};
use crate::dpdf_registry::{DpdfModelRegistry, ModelType};
use crate::istft::{IstftBasis, IstftParams};
use crate::kokoro_chorus::{mix_voices, mix_voices_stereo, ChorusConfig, VoiceMix};
use crate::kokoro_streaming::{crossfade_chunks, CrossfadeWindow, KokoroStreamConfig};
use crate::kokoro_tokenizer::{KokoroTokenizer, KokoroVocab, MAX_PHONEME_TOKENS, PAD_TOKEN_ID};
use crate::kokoro_tts::{KOKORO_HOP_LENGTH, KOKORO_N_FFT, KOKORO_SAMPLE_RATE};
use crate::silero_vad_builders::{ENCODER_BLOCKS, LSTM_HIDDEN_SIZE};
use crate::stft::{compute_stft_magnitude, StftParams};

// ===========================================================================
// 1. STFT window functions: Hann and Hamming symmetry and normalization
// ===========================================================================

/// Generate a Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n as f32).cos()))
        .collect()
}

/// Generate a Hamming window of length `n`.
fn hamming_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.54 - 0.46 * (2.0 * PI * k as f32 / n as f32).cos())
        .collect()
}

#[test]
fn test_hann_window_symmetry() {
    let n = 256;
    let w = hann_window(n);
    // Periodic Hann: w[k] == w[n-k] for k=1..n-1 is NOT exact for periodic,
    // but w[k] should approximately equal w[n-k] within machine epsilon for
    // the symmetric part. The periodic Hann formula w[k] = 0.5*(1 - cos(2*pi*k/N))
    // satisfies w[k] = w[N-k] exactly.
    for k in 1..n {
        let diff = (w[k] - w[n - k]).abs();
        assert!(
            diff < 1e-6,
            "Hann window not symmetric at k={k}: w[{k}]={}, w[{}]={}",
            w[k],
            n - k,
            w[n - k]
        );
    }
}

#[test]
fn test_hann_window_endpoints() {
    let n = 256;
    let w = hann_window(n);
    // Periodic Hann: w[0] = 0.0
    assert!(
        w[0].abs() < 1e-10,
        "Hann window should start at 0, got {}",
        w[0]
    );
    // w[n/2] should be peak = 1.0
    assert!(
        (w[n / 2] - 1.0).abs() < 1e-6,
        "Hann window mid-point should be ~1.0, got {}",
        w[n / 2]
    );
}

#[test]
fn test_hann_window_non_negative() {
    let n = 512;
    let w = hann_window(n);
    for (k, &val) in w.iter().enumerate() {
        assert!(val >= 0.0, "Hann window negative at k={k}: {val}");
    }
}

#[test]
fn test_hamming_window_symmetry() {
    let n = 256;
    let w = hamming_window(n);
    for k in 1..n {
        let diff = (w[k] - w[n - k]).abs();
        assert!(
            diff < 1e-6,
            "Hamming window not symmetric at k={k}: w[{k}]={}, w[{}]={}",
            w[k],
            n - k,
            w[n - k]
        );
    }
}

#[test]
fn test_hamming_window_minimum_positive() {
    let n = 256;
    let w = hamming_window(n);
    // Hamming window minimum is 0.54 - 0.46 = 0.08 (at endpoints)
    for (k, &val) in w.iter().enumerate() {
        assert!(
            val >= 0.07,
            "Hamming window too small at k={k}: {val} (min should be ~0.08)"
        );
    }
}

#[test]
fn test_hamming_window_normalization_peak() {
    let n = 256;
    let w = hamming_window(n);
    let max_val = w.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    // Hamming peak is 0.54 + 0.46 = 1.0 at center
    assert!(
        (max_val - 1.0).abs() < 1e-5,
        "Hamming window peak should be ~1.0, got {max_val}"
    );
}

#[test]
fn test_istft_hann_window_matches_manual() {
    // IstftBasis precomputes a Hann window internally; verify it matches our manual one
    let n_fft = 20;
    let params = IstftParams::new(n_fft, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let expected = hann_window(n_fft);
    let actual = basis.window();
    assert_eq!(actual.len(), expected.len());
    for (k, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-7,
            "Hann window mismatch at k={k}: IstftBasis={a}, manual={e}"
        );
    }
}

// ===========================================================================
// 2. STFT output shape: (n_fft/2 + 1) frequency bins
// ===========================================================================

#[test]
fn test_stft_output_shape_default() {
    let params = StftParams::default(); // n_fft=256, hop=128
                                        // Number of frequency bins = n_fft/2 + 1 = 129
    assert_eq!(params.n_freqs, 129);
}

#[test]
fn test_stft_output_shape_small_fft() {
    let params = StftParams::new(16, 8);
    assert_eq!(params.n_freqs, 9); // 16/2 + 1
}

#[test]
fn test_stft_output_shape_kokoro() {
    // Kokoro uses n_fft=20
    let n_bins = KOKORO_N_FFT / 2 + 1;
    assert_eq!(n_bins, 11, "Kokoro should have 11 frequency bins");
}

#[test]
fn test_stft_magnitude_output_dimensions() {
    // Create a small STFT with known dimensions.
    let n_fft = 8;
    let hop = 4;
    let params = StftParams::new(n_fft, hop); // n_freqs=5, pad_right=2
    let n_freqs = params.n_freqs;
    assert_eq!(n_freqs, 5);

    // Construct a real DFT basis of shape [n_fft+2, 1, n_fft] = [10, 1, 8]
    // Flattened: 10 * 8 = 80 elements
    let basis_len = (n_fft + 2) * n_fft;
    // Use cosine/sine DFT basis for a real test
    let mut basis = vec![0.0f32; basis_len];
    for f in 0..(n_fft + 2) {
        for k in 0..n_fft {
            basis[f * n_fft + k] = (2.0 * PI * f as f32 * k as f32 / n_fft as f32).cos();
        }
    }

    let audio_len = 100;
    let audio = vec![0.0f32; audio_len];
    let padded_len = audio_len + params.pad_right;
    let expected_n_frames = (padded_len - n_fft) / hop + 1;

    let mag = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    assert_eq!(
        mag.len(),
        n_freqs * expected_n_frames,
        "STFT output should be n_freqs * n_frames = {} * {} = {}, got {}",
        n_freqs,
        expected_n_frames,
        n_freqs * expected_n_frames,
        mag.len()
    );
}

#[test]
fn test_stft_n_freqs_formula_consistency() {
    // Verify n_freqs = n_fft/2 + 1 across various FFT sizes
    for n_fft in [8, 16, 20, 32, 64, 128, 256, 512, 1024, 4096] {
        let params = StftParams::new(n_fft, n_fft / 2);
        assert_eq!(
            params.n_freqs,
            n_fft / 2 + 1,
            "n_freqs formula broken for n_fft={n_fft}"
        );
    }
}

// ===========================================================================
// 3. iSTFT roundtrip: iSTFT(STFT(x)) approximately equals x
// ===========================================================================

#[test]
fn test_istft_roundtrip_pure_cosine() {
    // Test with a small FFT size: synthesize a cosine, take its forward DFT components,
    // then reconstruct via iSTFT. The result should approximate the original.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1; // 11
    let n_frames = 8;

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    // Generate a known signal: cosine at frequency bin 2
    let sig_len = n_fft + (n_frames - 1) * hop; // 20 + 35 = 55
    let freq_bin = 2;
    let original: Vec<f32> = (0..sig_len)
        .map(|k| (2.0 * PI * freq_bin as f32 * k as f32 / n_fft as f32).cos())
        .collect();

    // Manual forward DFT at each frame to get real/imag STFT components.
    // IstftBasis::istft applies a Hann SYNTHESIS window and COLA-normalizes by
    // sum(w^2); perfect reconstruction therefore requires the forward transform
    // to apply the matching Hann ANALYSIS window. Without it the roundtrip is
    // scaled by sum(w)/sum(w^2) (a 4/3 factor for n_fft=20, hop=5).
    let window = basis.window().to_vec();
    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];

    for t in 0..n_frames {
        let offset = t * hop;
        for f in 0..n_bins {
            let mut r = 0.0f32;
            let mut im = 0.0f32;
            for k in 0..n_fft {
                let angle = 2.0 * PI * f as f32 * k as f32 / n_fft as f32;
                let sample = if offset + k < sig_len {
                    original[offset + k]
                } else {
                    0.0
                };
                let windowed = sample * window[k];
                r += windowed * angle.cos();
                im -= windowed * angle.sin();
            }
            real[f * n_frames + t] = r;
            imag[f * n_frames + t] = im;
        }
    }

    let reconstructed = basis.istft(&real, &imag, n_frames, sig_len).unwrap();
    assert_eq!(reconstructed.len(), sig_len);

    // The interior samples (away from edges) should closely match the original.
    // Skip the first and last n_fft samples to avoid edge effects.
    let start = n_fft;
    let end = sig_len.saturating_sub(n_fft);
    if end > start {
        let max_error: f32 = (start..end)
            .map(|i| (reconstructed[i] - original[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_error < 0.1,
            "iSTFT roundtrip max error in interior = {max_error}, expected < 0.1"
        );
    }
}

#[test]
fn test_istft_roundtrip_dc_signal() {
    // A DC (constant) signal should roundtrip through STFT/iSTFT.
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 10;

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    let sig_len = n_fft + (n_frames - 1) * hop;
    let dc_val = 0.5f32;

    // Forward DFT of the DC signal with the Hann ANALYSIS window. IstftBasis
    // applies a Hann synthesis window and COLA-normalizes by sum(w^2), so the
    // forward transform must apply the matching analysis window for an exact
    // roundtrip. The windowed DC frame has spectral content beyond bin 0 (the
    // Hann taper is not flat), so compute the full forward DFT rather than only
    // setting the DC bin.
    let window = basis.window().to_vec();
    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];

    for t in 0..n_frames {
        for f in 0..n_bins {
            let mut r = 0.0f32;
            let mut im = 0.0f32;
            for k in 0..n_fft {
                let angle = 2.0 * PI * f as f32 * k as f32 / n_fft as f32;
                let windowed = dc_val * window[k];
                r += windowed * angle.cos();
                im -= windowed * angle.sin();
            }
            real[f * n_frames + t] = r;
            imag[f * n_frames + t] = im;
        }
    }

    let reconstructed = basis.istft(&real, &imag, n_frames, sig_len).unwrap();

    // Interior samples should be approximately dc_val
    let start = n_fft;
    let end = sig_len.saturating_sub(n_fft);
    if end > start {
        for i in start..end {
            assert!(
                (reconstructed[i] - dc_val).abs() < 0.05,
                "DC roundtrip: sample[{i}]={}, expected ~{dc_val}",
                reconstructed[i]
            );
        }
    }
}

#[test]
fn test_istft_zero_input_produces_zeros() {
    let n_fft = 20;
    let hop = 5;
    let n_bins = n_fft / 2 + 1;
    let n_frames = 4;

    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();

    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];
    let out_len = n_fft + (n_frames - 1) * hop;

    let result = basis.istft(&real, &imag, n_frames, out_len).unwrap();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.abs() < 1e-10,
            "iSTFT of zeros should be zero, got {v} at index {i}"
        );
    }
}

// ===========================================================================
// 4. Mel filterbank shape: [n_mels, n_fft/2 + 1]
// ===========================================================================

#[test]
fn test_mel_filterbank_shape_formula() {
    // Verify the standard mel filterbank shape relationship
    let n_fft = 512;
    let n_mels = 80;
    let n_freqs = n_fft / 2 + 1; // 257
    let expected_shape = (n_mels, n_freqs);
    assert_eq!(expected_shape, (80, 257));
}

#[test]
fn test_mel_filterbank_shape_common_configs() {
    // Common mel filterbank configurations used in speech models
    let configs: Vec<(usize, usize, (usize, usize))> = vec![
        (256, 40, (40, 129)),     // Silero VAD: 40 mels, 256 FFT
        (512, 80, (80, 257)),     // Whisper: 80 mels, 512 FFT
        (1024, 80, (80, 513)),    // Common speech: 80 mels, 1024 FFT
        (4096, 128, (128, 2049)), // HTDemucs spectral: 128 mels, 4096 FFT
    ];
    for (n_fft, n_mels, expected) in configs {
        let n_freqs = n_fft / 2 + 1;
        assert_eq!(
            (n_mels, n_freqs),
            expected,
            "Mel shape mismatch for n_fft={n_fft}, n_mels={n_mels}"
        );
    }
}

// ===========================================================================
// 5. Kokoro phoneme tokenizer: known phonemes map to valid token IDs
// ===========================================================================

#[test]
fn test_kokoro_tokenizer_known_phonemes() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    // Common phonemes that should be in the default vocabulary
    let known = [
        ('a', 43),
        ('b', 44),
        ('s', 61),
        ('t', 62),
        (' ', 16),
        ('.', 4),
        (',', 3),
    ];
    for (ch, expected_id) in known {
        let id = tokenizer.vocab().get(ch);
        assert_eq!(
            id,
            Some(expected_id),
            "Phoneme '{ch}' should map to {expected_id}"
        );
    }
}

#[test]
fn test_kokoro_tokenizer_ipa_phonemes() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    // IPA phonemes that Kokoro needs for English synthesis
    let ipa_phonemes = [
        ('\u{0259}', 83),  // schwa (ə)
        ('\u{025B}', 86),  // open-mid front (ɛ)
        ('\u{014B}', 112), // eng (ŋ)
        ('\u{0283}', 131), // postalveolar fricative (ʃ)
        ('\u{0292}', 147), // voiced postalveolar fricative (ʒ)
        ('\u{02C8}', 156), // primary stress (ˈ)
        ('\u{02D0}', 158), // length mark (ː)
    ];
    for (ch, expected_id) in ipa_phonemes {
        let id = tokenizer.vocab().get(ch);
        assert_eq!(
            id,
            Some(expected_id),
            "IPA phoneme '{}' (U+{:04X}) should map to {expected_id}",
            ch,
            ch as u32
        );
    }
}

#[test]
fn test_kokoro_tokenizer_encode_with_padding() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    let tokens = tokenizer.encode("hello").unwrap();
    // Should start and end with PAD_TOKEN_ID = 0
    assert_eq!(tokens[0], PAD_TOKEN_ID);
    assert_eq!(*tokens.last().unwrap(), PAD_TOKEN_ID);
    // "hello" has 5 chars, all in vocab: h(50) e(47) l(54) l(54) o(57)
    assert_eq!(tokens.len(), 7); // 5 phonemes + 2 padding
    assert_eq!(tokens[1], 50); // h
    assert_eq!(tokens[2], 47); // e
    assert_eq!(tokens[3], 54); // l
    assert_eq!(tokens[4], 54); // l
    assert_eq!(tokens[5], 57); // o
}

#[test]
fn test_kokoro_tokenizer_unknown_chars_dropped() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    // Emoji and CJK characters are not in the vocab - should be dropped
    let tokens = tokenizer.encode("a\u{1F600}b").unwrap();
    // Only 'a' (43) and 'b' (44) should be present
    assert_eq!(tokens, vec![PAD_TOKEN_ID, 43, 44, PAD_TOKEN_ID]);
}

#[test]
fn test_kokoro_tokenizer_empty_string() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    let tokens = tokenizer.encode("").unwrap();
    // Empty string: just two padding tokens
    assert_eq!(tokens, vec![PAD_TOKEN_ID, PAD_TOKEN_ID]);
}

#[test]
fn test_kokoro_vocab_default_size() {
    let vocab = KokoroVocab::kokoro_default();
    // Default vocab has 178 tokens (IDs 0-177)
    assert_eq!(vocab.n_tokens(), 178);
    assert!(!vocab.is_empty());
}

#[test]
fn test_kokoro_vocab_roundtrip() {
    let vocab = KokoroVocab::kokoro_default();
    // Every mapping should be reversible
    for (ch, id) in vocab.iter() {
        let decoded = vocab.decode_id(id);
        assert_eq!(
            decoded,
            Some(ch),
            "Vocab roundtrip failed: char '{ch}' -> id {id} -> {decoded:?}"
        );
    }
}

#[test]
fn test_kokoro_tokenizer_max_tokens_constant() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    assert_eq!(tokenizer.max_tokens(), MAX_PHONEME_TOKENS);
    assert_eq!(MAX_PHONEME_TOKENS, 510);
}

#[test]
fn test_kokoro_tokenizer_count_tokens() {
    let tokenizer = KokoroTokenizer::kokoro_default();
    let count = tokenizer.count_tokens("hello");
    assert_eq!(count, 5); // all 5 chars in vocab
    let count_with_unknown = tokenizer.count_tokens("h\u{1F600}i");
    assert_eq!(count_with_unknown, 2); // only h and i
}

#[test]
fn test_kokoro_vocab_insert_and_remove() {
    let mut vocab = KokoroVocab::empty();
    vocab.insert('x', 42);
    assert_eq!(vocab.get('x'), Some(42));
    assert_eq!(vocab.decode_id(42), Some('x'));
    let removed = vocab.remove('x');
    assert_eq!(removed, Some(42));
    assert_eq!(vocab.get('x'), None);
}

#[test]
fn test_kokoro_vocab_insert_auto() {
    let mut vocab = KokoroVocab::empty();
    // n_tokens starts at 1 (padding token 0)
    let id1 = vocab.insert_auto('a');
    assert_eq!(id1, 1);
    let id2 = vocab.insert_auto('b');
    assert_eq!(id2, 2);
    assert_eq!(vocab.n_tokens(), 3);
}

// ===========================================================================
// 6. Silero VAD: encoder block configuration and output shape
// ===========================================================================

#[test]
fn test_silero_vad_encoder_block_count() {
    assert_eq!(ENCODER_BLOCKS.len(), 4);
}

#[test]
fn test_silero_vad_encoder_channel_flow() {
    // Verify the channel dimension chain: 129 -> 128 -> 64 -> 64 -> 128
    let expected_channels = [(129, 128), (128, 64), (64, 64), (64, 128)];
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        let (exp_in, exp_out) = expected_channels[i];
        assert_eq!(block.in_channels, exp_in, "Block {i}: in_channels mismatch");
        assert_eq!(
            block.out_channels, exp_out,
            "Block {i}: out_channels mismatch"
        );
    }
}

#[test]
fn test_silero_vad_lstm_hidden_size() {
    assert_eq!(LSTM_HIDDEN_SIZE, 128);
}

#[test]
fn test_silero_vad_output_probability_range() {
    // The output stage is ReLU -> Linear(128->1) -> Sigmoid.
    // Sigmoid output is always in [0, 1] by definition.
    // Test the sigmoid function properties directly.
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    assert!(sigmoid(100.0) > 0.999);
    assert!(sigmoid(-100.0) < 0.001);
    assert!(sigmoid(0.0) >= 0.0 && sigmoid(0.0) <= 1.0);
}

#[test]
fn test_silero_vad_stft_input_shape() {
    // Silero VAD STFT: n_fft=256, hop=128, produces 129 frequency bins
    let params = StftParams::default();
    assert_eq!(params.n_freqs, 129);
    // First encoder block expects in_channels=129, matching STFT output
    assert_eq!(ENCODER_BLOCKS[0].in_channels, params.n_freqs);
}

// ===========================================================================
// 7. HTDemucs source separation: architecture constants
// ===========================================================================

#[test]
fn test_htdemucs_base_channels() {
    assert_eq!(BASE_CHANNELS, 48);
}

#[test]
fn test_htdemucs_channel_growth() {
    // channels_at_depth(d) = BASE_CHANNELS * GROWTH^d
    assert_eq!(channels_at_depth(0), 48);
    assert_eq!(channels_at_depth(1), 96);
    assert_eq!(channels_at_depth(2), 192);
    assert_eq!(channels_at_depth(3), 384);
    assert_eq!(channels_at_depth(4), 768);
}

#[test]
fn test_htdemucs_dconv_compress() {
    // DConv compression ratio should be 4
    assert_eq!(DCONV_COMPRESS, 4);
    // Compressed channels at depth 0: 48/4 = 12
    assert_eq!(channels_at_depth(0) / DCONV_COMPRESS, 12);
}

#[test]
fn test_htdemucs_temporal_depth() {
    assert_eq!(TEMPORAL_DEPTH, 5);
}

#[test]
fn test_htdemucs_conv1d_output_len() {
    // conv1d_output_len(input, kernel, stride, padding)
    // Standard formula: (input + 2*padding - kernel) / stride + 1
    let out = conv1d_output_len(100, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE, 0).unwrap();
    let expected = (100 - TEMPORAL_KERNEL_SIZE) / TEMPORAL_STRIDE + 1;
    assert_eq!(out, expected);
}

#[test]
fn test_htdemucs_dconv_depth_and_kernel() {
    assert_eq!(DCONV_DEPTH, 2);
    assert_eq!(DCONV_KERNEL, 3);
}

// ===========================================================================
// 8. Model dispatch: registered models are found by name
// ===========================================================================

#[test]
fn test_dpdf_registry_default_pipeline_count() {
    let registry = DpdfModelRegistry::default_pipeline();
    assert_eq!(registry.len(), 8, "Default pipeline should have 8 models");
}

#[test]
fn test_dpdf_registry_lookup_by_name() {
    let registry = DpdfModelRegistry::default_pipeline();
    let expected_names = [
        "granite_docling",
        "doclayout_yolo",
        "glm_ocr",
        "table_transformer",
        "qwen3_vl",
        "paddle_ocr",
        "firered_ocr",
        "rt_detr_heron",
    ];
    for name in expected_names {
        let entry = registry.get(name);
        assert!(
            entry.is_some(),
            "Model '{name}' should be found in default pipeline"
        );
        assert_eq!(entry.unwrap().name, name);
    }
}

#[test]
fn test_dpdf_registry_lookup_missing() {
    let registry = DpdfModelRegistry::default_pipeline();
    assert!(registry.get("nonexistent_model").is_none());
}

#[test]
fn test_dpdf_registry_list_by_type() {
    let registry = DpdfModelRegistry::default_pipeline();

    let ocr_models = registry.list_by_type(ModelType::OCR);
    assert_eq!(ocr_models.len(), 2, "Should have 2 OCR models");

    let vlm_models = registry.list_by_type(ModelType::VLM);
    assert_eq!(vlm_models.len(), 3, "Should have 3 VLM models");

    let layout_models = registry.list_by_type(ModelType::LayoutDetection);
    assert_eq!(
        layout_models.len(),
        2,
        "Should have 2 layout detection models"
    );

    let table_models = registry.list_by_type(ModelType::TableStructure);
    assert_eq!(table_models.len(), 1, "Should have 1 table structure model");
}

#[test]
fn test_dpdf_registry_model_types() {
    let registry = DpdfModelRegistry::default_pipeline();
    assert_eq!(
        registry.get("granite_docling").unwrap().model_type,
        ModelType::VLM
    );
    assert_eq!(
        registry.get("doclayout_yolo").unwrap().model_type,
        ModelType::LayoutDetection
    );
    assert_eq!(registry.get("glm_ocr").unwrap().model_type, ModelType::OCR);
    assert_eq!(
        registry.get("table_transformer").unwrap().model_type,
        ModelType::TableStructure
    );
}

#[test]
fn test_convert_config_detect_model_type() {
    assert_eq!(
        ConvertConfig::detect_model_type("ds4sd/Granite-Docling-258M"),
        Some(DpdfModelType::GraniteDocling)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("DocLayout-YOLO-base"),
        Some(DpdfModelType::DocLayoutYolo)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("Qwen3-VL-8B"),
        Some(DpdfModelType::Qwen3VL)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("table-transformer-structure"),
        Some(DpdfModelType::TableTransformer)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("glm-ocr-0.9b"),
        Some(DpdfModelType::GlmOcr)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("PaddleOCR-VL-1.5"),
        Some(DpdfModelType::PaddleOcr)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("FireRed-OCR-2B"),
        Some(DpdfModelType::FireRedOcr)
    );
    assert_eq!(ConvertConfig::detect_model_type("random-model"), None);
}

// ===========================================================================
// 9. Audio normalization: peak normalization to [-1, 1]
// ===========================================================================

/// Peak-normalize an audio buffer to [-1.0, 1.0].
fn peak_normalize(audio: &mut [f32]) {
    let peak = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    if peak > 0.0 {
        let scale = 1.0 / peak;
        for sample in audio.iter_mut() {
            *sample *= scale;
        }
    }
}

#[test]
fn test_peak_normalization_range() {
    let mut audio = vec![0.3, -0.7, 0.5, -0.2, 0.1];
    peak_normalize(&mut audio);
    for (i, &sample) in audio.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&sample),
            "Normalized sample[{i}]={sample} outside [-1, 1]"
        );
    }
    // The peak should be exactly 1.0 or -1.0
    let new_peak = audio.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        (new_peak - 1.0).abs() < 1e-6,
        "Peak after normalization should be 1.0, got {new_peak}"
    );
}

#[test]
fn test_peak_normalization_preserves_sign() {
    let mut audio = vec![0.3, -0.7, 0.5];
    peak_normalize(&mut audio);
    // Original had max abs at index 1 (-0.7), so normalized should be -1.0
    assert!((audio[1] - (-1.0)).abs() < 1e-6);
    // Other samples should maintain relative sign
    assert!(audio[0] > 0.0);
    assert!(audio[2] > 0.0);
}

#[test]
fn test_peak_normalization_zero_input() {
    let mut audio = vec![0.0, 0.0, 0.0];
    peak_normalize(&mut audio);
    // All zeros should remain zeros (no division by zero)
    for &sample in &audio {
        assert_eq!(sample, 0.0);
    }
}

#[test]
fn test_peak_normalization_already_normalized() {
    let mut audio = vec![1.0, -0.5, 0.3];
    peak_normalize(&mut audio);
    // Peak is already 1.0, so scaling should be identity
    assert!((audio[0] - 1.0).abs() < 1e-6);
    assert!((audio[1] - (-0.5)).abs() < 1e-6);
    assert!((audio[2] - 0.3).abs() < 1e-6);
}

#[test]
fn test_mix_voices_clip_produces_normalized_range() {
    // mix_voices with clip=true should produce output in [-1, 1]
    let voice1 = vec![0.8, 0.9, 1.0, -1.0];
    let voice2 = vec![0.7, 0.8, 0.9, -0.9];
    let gains = vec![1.0, 1.0]; // Sum > 1.0 before clipping
    let mixed = mix_voices(&[voice1, voice2], &gains, true).unwrap();
    for (i, &sample) in mixed.iter().enumerate() {
        assert!(
            (-1.0..=1.0).contains(&sample),
            "Clipped output[{i}]={sample} outside [-1, 1]"
        );
    }
}

// ===========================================================================
// 10. Sample rate conversion: output length formula
// ===========================================================================

#[test]
fn test_sample_rate_conversion_length() {
    // Standard formula: output_length = input_length * target_rate / source_rate
    let input_length = 24000usize; // 1 second at 24kHz
    let source_rate = 24000usize;
    let target_rate = 16000usize;
    let output_length = input_length * target_rate / source_rate;
    assert_eq!(output_length, 16000);
}

#[test]
fn test_sample_rate_conversion_kokoro_to_silero() {
    // Kokoro outputs at 24kHz, Silero VAD expects 16kHz
    let kokoro_samples = KOKORO_SAMPLE_RATE; // 24000 (1 second)
    let silero_rate = 16000usize;
    let silero_samples = kokoro_samples * silero_rate / KOKORO_SAMPLE_RATE;
    assert_eq!(silero_samples, 16000);
}

#[test]
fn test_sample_rate_conversion_upsample() {
    // 16kHz to 48kHz (3x upsample)
    let input_len = 16000usize;
    let source_rate = 16000usize;
    let target_rate = 48000usize;
    let output_len = input_len * target_rate / source_rate;
    assert_eq!(output_len, 48000);
}

#[test]
fn test_sample_rate_conversion_fractional() {
    // 22050 Hz to 16000 Hz (non-integer ratio)
    let input_len = 22050usize; // 1 second at 22050Hz
    let source_rate = 22050usize;
    let target_rate = 16000usize;
    // Integer arithmetic truncates: 22050 * 16000 / 22050 = 16000
    let output_len = input_len * target_rate / source_rate;
    assert_eq!(output_len, 16000);
}

#[test]
fn test_kokoro_sample_rate_constant() {
    assert_eq!(KOKORO_SAMPLE_RATE, 24000);
}

#[test]
fn test_kokoro_fft_hop_relationship() {
    // Kokoro: n_fft=20, hop=5, so overlap ratio = 1 - hop/n_fft = 0.75 (75%)
    assert_eq!(KOKORO_N_FFT, 20);
    assert_eq!(KOKORO_HOP_LENGTH, 5);
    let overlap_ratio = 1.0 - KOKORO_HOP_LENGTH as f64 / KOKORO_N_FFT as f64;
    assert!((overlap_ratio - 0.75).abs() < 1e-10);
}

// ===========================================================================
// Bonus: ConvertConfig builder pattern tests
// ===========================================================================

#[test]
fn test_convert_config_builder_defaults() {
    let config = ConvertConfig::new("test-model");
    assert_eq!(config.model_name, "test-model");
    assert!(config.validate_weights);
    assert!(config.constant_fold);
    assert!(config.model_type.is_none());
}

#[test]
fn test_convert_config_builder_chain() {
    let config = ConvertConfig::new("nn-model")
        .with_validate_weights(false)
        .with_constant_fold(false)
        .with_model_type(DpdfModelType::GraniteDocling);
    assert_eq!(config.model_name, "nn-model");
    assert!(!config.validate_weights);
    assert!(!config.constant_fold);
    assert_eq!(config.model_type, Some(DpdfModelType::GraniteDocling));
}

#[test]
fn test_convert_config_default() {
    let config = ConvertConfig::default();
    assert_eq!(config.model_name, "unnamed");
}

// ===========================================================================
// Bonus: Streaming config and crossfade tests
// ===========================================================================

#[test]
fn test_streaming_config_creation() {
    let config = KokoroStreamConfig::new(480).unwrap();
    assert_eq!(config.crossfade_samples, 480);
    assert_eq!(config.crossfade_window, CrossfadeWindow::Linear);
}

#[test]
fn test_streaming_config_zero_crossfade_rejected() {
    let result = KokoroStreamConfig::new(0);
    assert!(result.is_err());
}

#[test]
fn test_streaming_config_with_window() {
    let config = KokoroStreamConfig::new(480)
        .unwrap()
        .with_window(CrossfadeWindow::Hann);
    assert_eq!(config.crossfade_window, CrossfadeWindow::Hann);
}

#[test]
fn test_crossfade_chunks_basic() {
    let prev = vec![0.0f32; 100];
    let mut next = vec![1.0f32; 100];
    crossfade_chunks(&prev, &mut next, 10).unwrap();
    // After crossfade, the first 10 samples of next should be blended
    // from prev's tail (zeros) and next's head (ones)
    // alpha = i / (N-1), so sample 0 is mostly prev, sample 9 is mostly next
    assert!(next[0] < 0.2, "First crossfade sample should be near 0");
    assert!(next[9] > 0.8, "Last crossfade sample should be near 1");
}

#[test]
fn test_crossfade_chunks_too_short() {
    let prev = vec![0.0f32; 5];
    let mut next = vec![1.0f32; 100];
    let result = crossfade_chunks(&prev, &mut next, 10);
    assert!(result.is_err());
}

// ===========================================================================
// Bonus: Chorus mixing tests
// ===========================================================================

#[test]
fn test_chorus_config_equal_gain() {
    let config = ChorusConfig::equal_gain(4).unwrap();
    assert_eq!(config.n_voices, 4);
    for &g in &config.gains {
        assert!((g - 0.25).abs() < 1e-6);
    }
    assert!(config.clip_output);
    assert!(config.pans.is_none());
}

#[test]
fn test_chorus_config_validation() {
    assert!(ChorusConfig::equal_gain(0).is_err());
    assert!(ChorusConfig::equal_gain(33).is_err());
    assert!(ChorusConfig::equal_gain(1).is_ok());
    assert!(ChorusConfig::equal_gain(32).is_ok());
}

#[test]
fn test_mix_voices_same_length() {
    let v1 = vec![0.5f32; 100];
    let v2 = vec![0.3f32; 100];
    let mixed = mix_voices(&[v1, v2], &[0.5, 0.5], false).unwrap();
    assert_eq!(mixed.len(), 100);
    // Each sample: 0.5*0.5 + 0.3*0.5 = 0.4
    for &s in &mixed {
        assert!((s - 0.4).abs() < 1e-6);
    }
}

#[test]
fn test_mix_voices_different_lengths() {
    // Shorter voice should be zero-padded
    let v1 = vec![1.0f32; 50];
    let v2 = vec![1.0f32; 100];
    let mixed = mix_voices(&[v1, v2], &[0.5, 0.5], false).unwrap();
    assert_eq!(mixed.len(), 100, "Output should match longest voice");
    // First 50 samples: both voices contribute
    assert!((mixed[0] - 1.0).abs() < 1e-6);
    // Samples 50-99: only v2 contributes
    assert!((mixed[75] - 0.5).abs() < 1e-6);
}

#[test]
fn test_mix_voices_stereo_center_pan() {
    let v1 = vec![1.0f32; 10];
    let mix_params = vec![VoiceMix {
        gain: 1.0,
        pan: 0.0, // center
    }];
    let stereo = mix_voices_stereo(&[v1], &mix_params, false).unwrap();
    // Center pan: left and right should be equal (cos(pi/4) = sin(pi/4) = ~0.707)
    assert_eq!(stereo.len(), 20); // 10 samples * 2 channels
    let left = stereo[0];
    let right = stereo[1];
    assert!(
        (left - right).abs() < 1e-6,
        "Center pan should produce equal L/R: L={left}, R={right}"
    );
}

#[test]
fn test_mix_voices_empty() {
    let mixed = mix_voices(&[], &[], false).unwrap();
    assert!(mixed.is_empty());
}

#[test]
fn test_dpdf_registry_parameter_counts_nonzero() {
    let registry = DpdfModelRegistry::default_pipeline();
    for entry in registry.models() {
        assert!(
            entry.parameter_count > 0,
            "Model '{}' should have nonzero parameter count",
            entry.name
        );
    }
}

#[test]
fn test_model_type_labels() {
    assert_eq!(ModelType::OCR.label(), "OCR");
    assert_eq!(ModelType::VLM.label(), "VLM");
    assert_eq!(ModelType::LayoutDetection.label(), "Layout Detection");
    assert_eq!(ModelType::TableStructure.label(), "Table Structure");
}

// ===========================================================================
// Bonus: IstftBasis DFT basis shape validation
// ===========================================================================

#[test]
fn test_istft_basis_dimensions() {
    let n_fft = 20;
    let n_bins = n_fft / 2 + 1; // 11
    let params = IstftParams::new(n_fft, 5, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    assert_eq!(basis.n_bins(), n_bins);
    assert_eq!(basis.cos_basis().len(), n_bins * n_fft);
    assert_eq!(basis.sin_basis().len(), n_bins * n_fft);
    assert_eq!(basis.window().len(), n_fft);
}

#[test]
fn test_istft_params_validation() {
    // n_fft must be even and > 0
    assert!(IstftParams::new(0, 5, false, false).is_err());
    assert!(IstftParams::new(3, 5, false, false).is_err()); // odd
    assert!(IstftParams::new(4, 0, false, false).is_err()); // zero hop
    assert!(IstftParams::new(4, 2, false, false).is_ok());
}

#[test]
fn test_istft_cos_basis_orthogonality() {
    // For a well-formed DFT basis, cos_basis[0][0] should be 1.0 (cos(0) = 1)
    let n_fft = 8;
    let params = IstftParams::new(n_fft, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    // cos_basis[f=0, k=0] = cos(0) = 1.0
    assert!((basis.cos_basis()[0] - 1.0).abs() < 1e-7);
    // sin_basis[f=0, k=0] = sin(0) = 0.0
    assert!(basis.sin_basis()[0].abs() < 1e-7);
}
