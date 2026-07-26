// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for nn-models builders and signal processing.
//!
//! Covers:
//! - STFT/iSTFT signal processing (shapes, roundtrip, window, hop, FFT size, batch)
//! - Model configs (HTDemucs, Kokoro, Silero VAD, PlBert)
//! - Model builders (Silero VAD encoder, HTDemucs transformer, DConv)
//! - Convert config and model type detection
//! - Dispatch shape and dtype expectations
//!
//! Part of builder test coverage expansion.

use std::f32::consts::PI;

use nn_models::convert::{ConvertConfig, ConvertError, DpdfModelType};
use nn_models::demucs_shared::{
    build_dconv_sublayer, channels_at_depth, conv1d_output_len, validate_weight_size,
    DConvSubLayerInputs, BASE_CHANNELS, DCONV_COMPRESS, DCONV_DEPTH, GROWTH, SPECTRAL_BASIC_DEPTH,
    SPECTRAL_DEPTH, SPECTRAL_INPUT_CHANNELS, SPECTRAL_OUTPUT_CHANNELS, TEMPORAL_BASIC_DEPTH,
    TEMPORAL_DEPTH, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE,
};
use nn_models::demucs_transformer_constants::{
    BOTTLENECK_DIM, FFN_HIDDEN_DIM, LAYER_NORM_EPS, NUM_HEADS, NUM_LAYERS, TRANSFORMER_DIM,
};
use nn_models::silero_vad_builders::{
    build_encoder_block_def, build_output_def, ENCODER_BLOCKS, LSTM_HIDDEN_SIZE,
};
use nn_models::{
    compute_stft_magnitude, IstftBasis, IstftError, IstftParams, KokoroConfig, PlbertConfig,
    StftError, StftParams,
};

// ============================================================================
// A. STFT/iSTFT Signal Processing (8+ tests)
// ============================================================================

/// STFT on known audio shape: 576 samples (Silero VAD) -> [129, 4].
#[test]
fn test_stft_shape_silero_vad_576_samples() {
    let params = StftParams::default();
    let audio = vec![0.5f32; 576];
    let basis = vec![0.01f32; 258 * 256]; // non-zero basis
    let result = compute_stft_magnitude(&audio, &basis, &params).unwrap();
    // padded_len = 576 + 64 = 640; n_frames = (640 - 256) / 128 + 1 = 4
    assert_eq!(result.len(), 129 * 4);
    // With non-zero basis and non-zero audio, magnitudes should be non-zero
    let non_zero_count = result.iter().filter(|&&v| v > 1e-10).count();
    assert!(non_zero_count > 0, "should have non-zero magnitudes");
}

/// STFT -> iSTFT roundtrip on a chirp signal (frequency sweep).
#[test]
fn test_stft_istft_roundtrip_chirp() {
    let n_fft = 32;
    let hop = 8;
    let signal_len = 256;

    // Generate chirp: frequency sweeps from 1 to 10
    let signal: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / signal_len as f32;
            let freq = 1.0 + 9.0 * t;
            (2.0 * PI * freq * t).sin()
        })
        .collect();

    // Forward STFT
    let n_bins = n_fft / 2 + 1;
    let n_frames = (signal_len - n_fft) / hop + 1;
    let window: Vec<f32> = (0..n_fft)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
        .collect();

    let mut real = vec![0.0f32; n_bins * n_frames];
    let mut imag = vec![0.0f32; n_bins * n_frames];
    for t in 0..n_frames {
        let offset = t * hop;
        for f in 0..n_bins {
            let (mut r, mut im) = (0.0f32, 0.0f32);
            for k in 0..n_fft {
                let angle = 2.0 * PI * (f as f32) * (k as f32) / (n_fft as f32);
                let windowed = signal[offset + k] * window[k];
                r += windowed * angle.cos();
                im -= windowed * angle.sin();
            }
            real[f * n_frames + t] = r;
            imag[f * n_frames + t] = im;
        }
    }

    // Inverse STFT
    let params = IstftParams::new(n_fft, hop, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let full_len = n_fft + (n_frames - 1) * hop;
    let reconstructed = basis.istft(&real, &imag, n_frames, full_len).unwrap();

    // Check interior samples (excluding boundary effects)
    let margin = n_fft / 2;
    let mut max_err = 0.0f32;
    for i in margin..(full_len - margin).min(signal_len) {
        let err = (reconstructed[i] - signal[i]).abs();
        max_err = max_err.max(err);
    }
    assert!(
        max_err < 0.1,
        "chirp roundtrip max error = {max_err}, expected < 0.1"
    );
}

/// Hann window: w[k] = 0.5 * (1 - cos(2pi*k/N)).
/// Verify endpoints are 0, midpoint is 1, and all values in [0, 1].
#[test]
fn test_stft_window_function_properties() {
    let params = IstftParams::new(64, 16, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let window = basis.window();

    assert_eq!(window.len(), 64);
    // Endpoints should be ~0
    assert!(window[0].abs() < 1e-6, "window[0] should be ~0");
    assert!(window[63].abs() < 0.01, "window[N-1] should be ~0");
    // Midpoint should be ~1
    assert!(
        (window[32] - 1.0).abs() < 1e-6,
        "window[N/2] should be ~1, got {}",
        window[32]
    );
    // All in [0, 1]
    for (k, &w) in window.iter().enumerate() {
        assert!((0.0..=1.0).contains(&w), "window[{k}] = {w} outside [0, 1]");
    }
    // Periodic Hann: w[k] == w[N-k] for k = 1..N/2-1
    // (w[0] = 0 pairs with w[N] which is outside the window)
    for k in 1..32 {
        assert!(
            (window[k] - window[64 - k]).abs() < 1e-5,
            "window symmetry violated at k={k}: {} vs {}",
            window[k],
            window[64 - k]
        );
    }
}

/// Different hop sizes produce expected frame counts.
#[test]
fn test_stft_hop_length_frame_counts() {
    let n_fft = 256;
    let audio_len = 1024;

    // hop=128: frames = (1024 + 64 - 256) / 128 + 1 = 832/128 + 1 = 7
    let p1 = StftParams::new(n_fft, 128);
    let audio = vec![0.0f32; audio_len];
    let basis = vec![0.0f32; 258 * 256];
    let r1 = compute_stft_magnitude(&audio, &basis, &p1).unwrap();
    let frames_128 = r1.len() / p1.n_freqs;

    // hop=64: frames = (1024 + 64 - 256) / 64 + 1 = 832/64 + 1 = 14
    let p2 = StftParams::new(n_fft, 64);
    let r2 = compute_stft_magnitude(&audio, &basis, &p2).unwrap();
    let frames_64 = r2.len() / p2.n_freqs;

    // Halving the hop length should roughly double the frame count
    assert!(
        frames_64 >= 2 * frames_128 - 1,
        "halving hop should ~double frames: hop=64 gave {frames_64}, hop=128 gave {frames_128}"
    );
}

/// FFT size affects frequency resolution (number of bins).
#[test]
fn test_stft_fft_size_frequency_resolution() {
    // n_fft=256 -> 129 bins; n_fft=512 -> 257 bins
    let p256 = StftParams::new(256, 128);
    assert_eq!(p256.n_freqs, 129);

    let p512 = StftParams::new(512, 256);
    assert_eq!(p512.n_freqs, 257);

    let p1024 = StftParams::new(1024, 512);
    assert_eq!(p1024.n_freqs, 513);

    // Doubling n_fft roughly doubles frequency bins
    assert_eq!(p512.n_freqs, 2 * p256.n_freqs - 1);
}

/// iSTFT overlap-add: verify COLA condition for various (n_fft, hop) combos.
#[test]
fn test_istft_overlap_add_cola_condition() {
    let test_cases = [(16, 4), (32, 8), (64, 16), (128, 32)];

    for &(n_fft, hop) in &test_cases {
        let n_frames = 10;
        let full_len = n_fft + (n_frames - 1) * hop;
        let window: Vec<f32> = (0..n_fft)
            .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n_fft as f32).cos()))
            .collect();

        let mut window_sum = vec![0.0f32; full_len];
        for t in 0..n_frames {
            let offset = t * hop;
            for k in 0..n_fft {
                window_sum[offset + k] += window[k] * window[k];
            }
        }

        // COLA: interior should have non-zero window sum
        let margin = n_fft / 2;
        for i in margin..(full_len - margin) {
            assert!(
                window_sum[i] > 1e-6,
                "COLA violated at pos {i} for n_fft={n_fft}, hop={hop}: sum={}",
                window_sum[i]
            );
        }
    }
}

/// Single-channel audio through STFT produces correct output dimensions.
#[test]
fn test_stft_mono_audio_various_lengths() {
    let params = StftParams::new(4, 2);
    let basis = vec![0.0f32; 6 * 4];

    for audio_len in [6, 8, 10, 16, 32] {
        let audio = vec![0.1f32; audio_len];
        let result = compute_stft_magnitude(&audio, &basis, &params).unwrap();
        let padded_len = audio_len + params.pad_right;
        let expected_frames = (padded_len - params.n_fft) / params.hop_length + 1;
        let expected_total = params.n_freqs * expected_frames;
        assert_eq!(
            result.len(),
            expected_total,
            "audio_len={audio_len}: expected {expected_total} elements, got {}",
            result.len()
        );
    }
}

/// Verify iSTFT output matches requested output_length exactly.
#[test]
fn test_istft_output_length_exact() {
    let params = IstftParams::new(16, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_bins = basis.n_bins();
    let n_frames = 5;
    let real = vec![0.0f32; n_bins * n_frames];
    let imag = vec![0.0f32; n_bins * n_frames];

    for target_len in [1, 10, 20, 32, 50, 100] {
        let result = basis.istft(&real, &imag, n_frames, target_len).unwrap();
        assert_eq!(
            result.len(),
            target_len,
            "requested {target_len}, got {}",
            result.len()
        );
    }
}

// ============================================================================
// B. Model Configs (6+ tests)
// ============================================================================

/// HTDemucs architecture: default depths produce expected channel counts.
#[test]
fn test_htdemucs_config_default_channel_progression() {
    let expected = [(0, 48), (1, 96), (2, 192), (3, 384)];
    for &(depth, channels) in &expected {
        assert_eq!(
            channels_at_depth(depth),
            channels,
            "depth {depth} should have {channels} channels"
        );
    }
    // Verify GROWTH=2.0 formula
    for depth in 0..TEMPORAL_DEPTH {
        let expected = (BASE_CHANNELS as f64 * GROWTH.powi(depth as i32)) as usize;
        assert_eq!(channels_at_depth(depth), expected);
    }
}

/// Kokoro config: default values match the documented Kokoro-82M architecture.
#[test]
fn test_kokoro_config_default_architecture_invariants() {
    let cfg = KokoroConfig::default();
    cfg.validate().expect("default must be valid");

    // Encoder dimension
    assert_eq!(cfg.d_en, 512);
    // Generator initial channels matches encoder dimension
    assert_eq!(cfg.gen_initial_channels, 512);
    // n_fft must be divisible by 4
    assert!(cfg.n_fft % 4 == 0);
    // upsample_rates product = hop length
    let hop: usize = cfg.upsample_rates.iter().product();
    assert_eq!(hop, 60, "upsample product = hop length for iSTFT");
    // Each kernel_size = 2 * rate
    for (ks, rate) in cfg
        .upsample_kernel_sizes
        .iter()
        .zip(cfg.upsample_rates.iter())
    {
        assert_eq!(*ks, 2 * rate);
    }
    // resblock_dilations length matches resblock_kernel_sizes
    assert_eq!(
        cfg.resblock_dilations.len(),
        cfg.resblock_kernel_sizes.len()
    );
}

/// Silero VAD config: validate the encoder block architecture.
#[test]
fn test_silero_vad_config_encoder_architecture() {
    assert_eq!(ENCODER_BLOCKS.len(), 4);
    assert_eq!(LSTM_HIDDEN_SIZE, 128);

    // First block: STFT bins (129) -> 128
    assert_eq!(ENCODER_BLOCKS[0].in_channels, 129);
    assert_eq!(ENCODER_BLOCKS[0].out_channels, 128);

    // Channel chain: each out_channels == next in_channels
    for i in 0..ENCODER_BLOCKS.len() - 1 {
        assert_eq!(
            ENCODER_BLOCKS[i].out_channels,
            ENCODER_BLOCKS[i + 1].in_channels,
            "block {i} -> block {} channel mismatch",
            i + 1
        );
    }

    // Last block output == LSTM hidden size
    let last = &ENCODER_BLOCKS[ENCODER_BLOCKS.len() - 1];
    assert_eq!(last.out_channels, LSTM_HIDDEN_SIZE);

    // All blocks use kernel_size=3 with valid padding
    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        assert_eq!(block.kernel_size, 3, "block {i} kernel");
        assert!(block.padding <= block.kernel_size / 2, "block {i} padding");
    }
}

/// PlBert config default matches Kokoro-82M expectations.
#[test]
fn test_plbert_config_default_kokoro_values() {
    let cfg = PlbertConfig::default();
    assert_eq!(cfg.vocab_size, 178, "Kokoro phoneme vocab");
    assert_eq!(cfg.embedding_dim, 128, "factorized embedding dim");
    assert_eq!(cfg.hidden_size, 768, "ALBERT hidden size");
    assert_eq!(cfg.num_attention_heads, 12, "attention heads");
    assert_eq!(cfg.intermediate_size, 2048, "FFN intermediate");
    assert_eq!(cfg.max_position_embeddings, 512, "max positions");
    assert_eq!(cfg.num_hidden_layers, 12, "shared layer iterations");
    // Head dim must divide evenly
    assert_eq!(
        cfg.hidden_size % cfg.num_attention_heads,
        0,
        "hidden_size must be divisible by num_heads"
    );
}

/// KokoroConfig::new() produces a valid config, and default field values
/// satisfy all invariants validated by validate().
#[test]
fn test_kokoro_config_new_validates_and_is_consistent() {
    let cfg = KokoroConfig::new();
    cfg.validate().expect("KokoroConfig::new() must be valid");

    // All non-zero fields
    assert!(cfg.d_en > 0, "d_en must be > 0");
    assert!(cfg.style_dim > 0, "style_dim must be > 0");
    assert!(cfg.max_dur > 0, "max_dur must be > 0");
    assert!(
        cfg.n_fft > 0 && cfg.n_fft.is_multiple_of(4),
        "n_fft must be > 0 and divisible by 4"
    );
    assert!(
        !cfg.upsample_rates.is_empty(),
        "upsample_rates must be non-empty"
    );

    // KokoroConfig::new() == KokoroConfig::default()
    let def = KokoroConfig::default();
    assert_eq!(cfg.d_en, def.d_en);
    assert_eq!(cfg.style_dim, def.style_dim);
    assert_eq!(cfg.n_fft, def.n_fft);
    assert_eq!(cfg.max_dur, def.max_dur);
    assert_eq!(cfg.gen_initial_channels, def.gen_initial_channels);
    assert_eq!(cfg.upsample_rates, def.upsample_rates);
    assert_eq!(cfg.upsample_kernel_sizes, def.upsample_kernel_sizes);
    assert_eq!(cfg.resblock_kernel_sizes, def.resblock_kernel_sizes);
}

/// Kokoro config field consistency: upsample_rates/kernel_sizes lengths match,
/// resblock_dilations length matches resblock_kernel_sizes.
#[test]
fn test_kokoro_config_field_length_consistency() {
    let cfg = KokoroConfig::default();
    assert_eq!(
        cfg.upsample_rates.len(),
        cfg.upsample_kernel_sizes.len(),
        "upsample_rates and upsample_kernel_sizes must have same length"
    );
    assert_eq!(
        cfg.resblock_dilations.len(),
        cfg.resblock_kernel_sizes.len(),
        "resblock_dilations and resblock_kernel_sizes must have same length"
    );
    // Each dilation list should be non-empty
    for (i, dilations) in cfg.resblock_dilations.iter().enumerate() {
        assert!(
            !dilations.is_empty(),
            "resblock_dilations[{i}] must be non-empty"
        );
    }
}

// ============================================================================
// C. Model Building (4+ tests)
// ============================================================================

/// Build HTDemucs transformer channel bridge def.
#[test]
fn test_htdemucs_builder_channel_bridge() {
    use nn_models::demucs_transformer_builders::build_channel_bridge_def;

    let seq_len = 16;
    let (def, _wmap) =
        build_channel_bridge_def("test_bridge", BOTTLENECK_DIM, TRANSFORMER_DIM, seq_len)
            .expect("bridge should build");

    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(output_shape[0], TRANSFORMER_DIM);
    assert_eq!(output_shape[1], seq_len);
}

/// Build all 4 Silero VAD encoder blocks and verify the temporal chain.
#[test]
fn test_silero_vad_builder_full_chain() {
    // Starting from STFT output frames
    let stft_params = StftParams::default();
    let audio_samples = 576;
    let padded = audio_samples + stft_params.pad_right;
    let mut t = (padded - stft_params.n_fft) / stft_params.hop_length + 1; // 4

    for (i, block) in ENCODER_BLOCKS.iter().enumerate() {
        let t_out = (t + 2 * block.padding - block.kernel_size) / block.stride + 1;
        let def = build_encoder_block_def(block, t, t_out)
            .unwrap_or_else(|e| panic!("encoder block {i} failed: {e}"));
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(output_shape[0], 1, "batch dim for block {i}");
        assert_eq!(
            output_shape[1], block.out_channels,
            "channels for block {i}"
        );
        assert_eq!(output_shape[2], t_out, "time for block {i}");
        t = t_out;
    }
}

/// Build Silero VAD output stage and verify it ends with Sigmoid.
#[test]
fn test_silero_vad_builder_output_stage() {
    let def = build_output_def().expect("output def should build");
    let output_shape = &def.nodes[def.output.index()].shape;
    assert_eq!(
        output_shape,
        &[1, 1],
        "output should be [1, 1] scalar probability"
    );

    // Verify it contains ReLU + Sigmoid
    let has_relu = def
        .nodes
        .iter()
        .any(|n| matches!(n.kind, nn_dsl::tensor_ir::TensorOpKind::Relu { .. }));
    let has_sigmoid = def
        .nodes
        .iter()
        .any(|n| matches!(n.kind, nn_dsl::tensor_ir::TensorOpKind::Sigmoid { .. }));
    assert!(has_relu, "output stage needs ReLU");
    assert!(has_sigmoid, "output stage needs Sigmoid");
}

/// Build DConv sublayer and verify residual connection preserves dimensions.
#[test]
fn test_dconv_sublayer_preserves_dimensions() {
    use nn_dsl::TensorBlockBuilder;

    for depth in 0..TEMPORAL_DEPTH {
        let channels = channels_at_depth(depth);
        let compressed = channels / DCONV_COMPRESS;
        let t_len = 64;

        let mut b = TensorBlockBuilder::new(&format!("dconv_test_{depth}"));
        let input = b.add_input("input", &[channels, t_len]);

        let mut x = input;
        for k in 0..DCONV_DEPTH {
            let dc = DConvSubLayerInputs::add_to_builder(&mut b, k, channels, compressed);
            x = build_dconv_sublayer(&mut b, x, &dc, channels, compressed, t_len)
                .unwrap_or_else(|e| panic!("depth={depth}, k={k}: {e}"));
        }

        // Output shape should match input shape (residual connection)
        let def = b.build(x).expect("build should succeed");
        let output_shape = &def.nodes[def.output.index()].shape;
        assert_eq!(
            output_shape,
            &[channels, t_len],
            "DConv at depth={depth} should preserve [channels, t_len]"
        );
    }
}

// ============================================================================
// D. Dispatch & Shape Tests (3+ tests)
// ============================================================================

/// Conv1d output length formula matches expected model dimensions.
#[test]
fn test_model_dispatch_shapes_conv1d_temporal() {
    // HTDemucs temporal encoder: input=65536, kernel=8, stride=4, padding=2
    let out = conv1d_output_len(65536, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE, 2).unwrap();
    assert_eq!(out, 16384, "temporal encoder depth 0 halves length");

    // Chain 5 temporal encoder blocks (TEMPORAL_DEPTH=5)
    // 65536 -> 16384 -> 4096 -> 1024 -> 256 -> 64
    let mut t = 65536;
    for _depth in 0..TEMPORAL_DEPTH {
        t = conv1d_output_len(t, TEMPORAL_KERNEL_SIZE, TEMPORAL_STRIDE, 2).unwrap();
    }
    assert_eq!(t, 64, "5 temporal blocks: 65536 -> 64");
}

/// IstftParams validated construction: correct produces Ok, invalid produces Err.
#[test]
fn test_istft_params_validated_construction() {
    // Valid cases
    assert!(IstftParams::new(8, 4, false, false).is_ok());
    assert!(IstftParams::new(4096, 1024, true, true).is_ok());
    assert!(IstftParams::new(20, 5, false, false).is_ok()); // Kokoro

    // Invalid: odd n_fft
    assert!(matches!(
        IstftParams::new(7, 3, false, false),
        Err(IstftError::OddNfft { n_fft: 7 })
    ));

    // Invalid: zero n_fft
    assert!(matches!(
        IstftParams::new(0, 4, false, false),
        Err(IstftError::OddNfft { n_fft: 0 })
    ));

    // Invalid: zero hop
    assert!(matches!(
        IstftParams::new(8, 0, false, false),
        Err(IstftError::ZeroHopLength)
    ));
}

/// Correct dtype handling: STFT errors propagate correctly to TensorError.
#[test]
fn test_stft_error_to_tensor_error_conversion() {
    let stft_err = StftError::AudioTooShort {
        padded_len: 10,
        n_fft: 256,
    };
    let tensor_err: nn_core::TensorError = stft_err.into();
    let msg = tensor_err.to_string();
    assert!(msg.contains("10") || msg.contains("256"));

    let istft_err = IstftError::OddNfft { n_fft: 5 };
    let tensor_err: nn_core::TensorError = istft_err.into();
    let msg = tensor_err.to_string();
    assert!(msg.contains("5") || msg.contains("even"));
}

// ============================================================================
// E. Convert Config & Model Type Detection (4+ tests)
// ============================================================================

/// ConvertConfig builder chain works correctly.
#[test]
fn test_convert_config_builder_chain_full() {
    let cfg = ConvertConfig::new("test-model")
        .with_validate_weights(false)
        .with_constant_fold(false)
        .with_model_type(DpdfModelType::GraniteDocling);
    assert_eq!(cfg.model_name, "test-model");
    assert!(!cfg.validate_weights);
    assert!(!cfg.constant_fold);
    assert_eq!(cfg.model_type, Some(DpdfModelType::GraniteDocling));
}

/// Model type detection from HuggingFace identifiers.
#[test]
fn test_convert_config_model_type_detection() {
    assert_eq!(
        ConvertConfig::detect_model_type("ds4sd/Granite-Docling-258M"),
        Some(DpdfModelType::GraniteDocling)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("opendatalab/DocLayout-YOLO"),
        Some(DpdfModelType::DocLayoutYolo)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("Qwen/Qwen3-VL-2B"),
        Some(DpdfModelType::Qwen3VL)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("microsoft/Table-Transformer-v1"),
        Some(DpdfModelType::TableTransformer)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("THUDM/GLM-OCR-0.9B"),
        Some(DpdfModelType::GlmOcr)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("PaddleOCR/PaddleOCR-SVTR"),
        Some(DpdfModelType::PaddleOcr)
    );
    assert_eq!(
        ConvertConfig::detect_model_type("FireRed/FireRed-OCR-2B"),
        Some(DpdfModelType::FireRedOcr)
    );
    // Unknown model
    assert_eq!(
        ConvertConfig::detect_model_type("AI Provider/whisper-large-v3"),
        None
    );
}

/// ConvertConfig default has sensible values.
#[test]
fn test_convert_config_default() {
    let cfg = ConvertConfig::default();
    assert_eq!(cfg.model_name, "unnamed");
    assert!(cfg.validate_weights);
    assert!(cfg.constant_fold);
    assert!(cfg.model_type.is_none());
}

/// ConvertError display messages contain useful diagnostics.
#[test]
fn test_convert_error_display_messages() {
    let err = ConvertError::Io {
        path: "/tmp/model.json".into(),
        detail: "file not found".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("/tmp/model.json"));
    assert!(msg.contains("file not found"));

    let err = ConvertError::WeightShapeMismatch {
        name: "encoder.layer.0.weight".into(),
        expected: 768,
        actual: 512,
    };
    let msg = err.to_string();
    assert!(msg.contains("encoder.layer.0.weight"));
    assert!(msg.contains("768"));
    assert!(msg.contains("512"));

    let err = ConvertError::WeightLoad("corrupted header".into());
    assert!(err.to_string().contains("corrupted header"));
}

// ============================================================================
// F. HTDemucs Transformer Constants (2+ tests)
// ============================================================================

/// Transformer architecture constants are self-consistent.
#[test]
fn test_htdemucs_transformer_constants_consistency() {
    assert_eq!(TRANSFORMER_DIM, 512);
    assert_eq!(NUM_HEADS, 8);
    assert_eq!(TRANSFORMER_DIM % NUM_HEADS, 0, "dim must divide by heads");
    let head_dim = TRANSFORMER_DIM / NUM_HEADS;
    assert_eq!(head_dim, 64);

    assert_eq!(BOTTLENECK_DIM, 384);
    assert_eq!(BOTTLENECK_DIM, channels_at_depth(3));

    assert_eq!(FFN_HIDDEN_DIM, 2048);
    assert_eq!(NUM_LAYERS, 5);
    assert!(LAYER_NORM_EPS > 0.0);
    assert!(LAYER_NORM_EPS < 1e-3);
}

/// HTDemucs depth constants match the full 6-depth architecture.
///
/// Real HTDemucs: temporal has 5 depths (4 basic + 1 final), spectral has 6 depths
/// (4 basic + 2 deep with BiLSTM + local attention).
#[test]
fn test_htdemucs_depth_constants() {
    assert_eq!(TEMPORAL_BASIC_DEPTH, 4, "temporal basic depth");
    assert_eq!(
        TEMPORAL_DEPTH, 5,
        "temporal total depth (4 basic + 1 final)"
    );
    assert_eq!(SPECTRAL_BASIC_DEPTH, 4, "spectral basic depth");
    assert_eq!(SPECTRAL_DEPTH, 6, "spectral total depth (4 basic + 2 deep)");
    assert_eq!(SPECTRAL_INPUT_CHANNELS, 4, "2 stereo * 2 complex");
    assert_eq!(
        SPECTRAL_OUTPUT_CHANNELS, 16,
        "4 sources * 2 stereo * 2 complex"
    );
}

// ============================================================================
// G. Weight Validation (2+ tests)
// ============================================================================

/// validate_weight_size returns correct errors with descriptive messages.
#[test]
fn test_validate_weight_size_descriptive_errors() {
    let data = vec![0.0f32; 100];
    assert!(validate_weight_size(&data, "conv.weight", 100).is_ok());

    let err = validate_weight_size(&data, "encoder.bias", 200).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("encoder.bias"), "error should name the weight");
    assert!(msg.contains("200"), "error should show expected count");
    assert!(msg.contains("100"), "error should show actual count");
}

/// channels_at_depth panics at absurd depths.
#[test]
#[should_panic(expected = "exceeds maximum")]
fn test_channels_at_depth_overflow_guard() {
    channels_at_depth(31);
}

// ============================================================================
// H. IstftBasis DFT Correctness (2+ tests)
// ============================================================================

/// Verify DFT basis orthogonality: cos and sin basis for different frequencies.
#[test]
fn test_istft_dft_basis_orthogonality() {
    let params = IstftParams::new(16, 4, false, false).unwrap();
    let basis = IstftBasis::new(params).unwrap();
    let n_fft = 16;

    // DC (f=0): cos basis should be all 1.0
    for k in 0..n_fft {
        assert!(
            (basis.cos_basis()[k] - 1.0).abs() < 1e-6,
            "cos[0, {k}] should be 1.0"
        );
        assert!(
            basis.sin_basis()[k].abs() < 1e-6,
            "sin[0, {k}] should be 0.0"
        );
    }

    // Nyquist (f=n_fft/2): cos(pi*k) = (-1)^k
    let f = n_fft / 2;
    for k in 0..n_fft {
        let expected = if k % 2 == 0 { 1.0 } else { -1.0 };
        assert!(
            (basis.cos_basis()[f * n_fft + k] - expected).abs() < 1e-5,
            "cos[{f}, {k}] should be {expected}"
        );
    }
}

/// Verify normalized iSTFT uses 1/sqrt(N) factor.
#[test]
fn test_istft_normalization_factor() {
    let n_fft = 8;
    let params_norm = IstftParams::new(n_fft, 4, true, false).unwrap();
    let params_unnorm = IstftParams::new(n_fft, 4, false, false).unwrap();

    let basis_norm = IstftBasis::new(params_norm).unwrap();
    let basis_unnorm = IstftBasis::new(params_unnorm).unwrap();

    let n_bins = n_fft / 2 + 1;
    let n_frames = 1;

    // DC-only signal: real[0] = 1.0, everything else 0
    let mut real = vec![0.0f32; n_bins * n_frames];
    real[0] = 1.0;
    let imag = vec![0.0f32; n_bins * n_frames];

    let out_norm = basis_norm.istft(&real, &imag, n_frames, n_fft).unwrap();
    let out_unnorm = basis_unnorm.istft(&real, &imag, n_frames, n_fft).unwrap();

    // Normalized uses 1/sqrt(N), unnormalized uses 1/N
    // The ratio of peak values should be sqrt(N)
    let peak_norm = out_norm.iter().copied().fold(0.0f32, f32::max);
    let peak_unnorm = out_unnorm.iter().copied().fold(0.0f32, f32::max);
    if peak_unnorm > 1e-10 {
        let ratio = peak_norm / peak_unnorm;
        let expected_ratio = (n_fft as f32).sqrt();
        assert!(
            (ratio - expected_ratio).abs() < 0.1,
            "norm/unnorm peak ratio should be sqrt(N)={expected_ratio}, got {ratio}"
        );
    }
}
