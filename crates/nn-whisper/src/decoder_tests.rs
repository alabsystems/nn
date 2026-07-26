// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for Whisper text decoder and KV cache integration.

use crate::config::WhisperConfig;
use crate::decode::{DecodeConfig, DEFAULT_AVG_LOGPROB_THRESHOLD, DEFAULT_COMPRESSION_RATIO_THRESHOLD, MAX_DECODE_LENGTH};
use crate::positional::causal_mask;
use crate::test_utils::{tiny_config, tiny_encoder_output, tiny_model};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::kv_cache::{KvCache, KvCacheLayer};
use nn_core::test_utils::cpu;
use nn_core::{DType, VarBuilder};

// ---------------------------------------------------------------------------
// KvCache creation and layer count
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_creation_with_correct_layer_count() {
    for n in [0, 1, 4, 32] {
        let cache = KvCache::new(n);
        assert_eq!(cache.num_layers(), n, "KvCache::new({n}) must create {n} layers");
    }
}

#[test]
fn test_kv_cache_new_all_layers_empty() {
    let cache = KvCache::new(6);
    assert!(cache.is_empty(), "freshly created cache must be empty");
    assert_eq!(cache.seq_len(), 0, "empty cache seq_len must be 0");
    for i in 0..6 {
        let layer = cache.layer(i).expect("valid layer index");
        assert!(layer.is_empty(), "layer {i} must be empty after creation");
        assert_eq!(layer.seq_len(), 0);
    }
}

// ---------------------------------------------------------------------------
// KvCacheLayer seq_len tracking
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_seq_len_after_append() {
    let mut layer = KvCacheLayer::empty();
    assert_eq!(layer.seq_len(), 0);
    assert_eq!(layer.current_seq_len(), 0); // candle compat alias

    // Append a [1, 2, 3, 4] shaped tensor (batch=1, heads=2, seq=3, head_dim=4).
    let k = DynTensor::zeros(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let v = DynTensor::zeros(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let (_fk, _fv) = layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 3, "seq_len after first append of 3 positions");
    assert_eq!(layer.current_seq_len(), 3);
    assert!(!layer.is_empty());

    // Append a single new position.
    let k2 = DynTensor::zeros(&[1, 2, 1, 4], DType::F32, &cpu()).unwrap();
    let v2 = DynTensor::zeros(&[1, 2, 1, 4], DType::F32, &cpu()).unwrap();
    let (_fk2, _fv2) = layer.append(&k2, &v2).unwrap();
    assert_eq!(layer.seq_len(), 4, "seq_len after appending 1 more position");
}

// ---------------------------------------------------------------------------
// KvCacheLayer access via layer / layer_mut
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_access_valid_index() {
    let mut cache = KvCache::new(4);
    // Immutable access.
    for i in 0..4 {
        assert!(cache.layer(i).is_ok(), "layer({i}) must be Ok");
    }
    // Mutable access.
    for i in 0..4 {
        assert!(cache.layer_mut(i).is_ok(), "layer_mut({i}) must be Ok");
    }
}

#[test]
fn test_kv_cache_layer_access_out_of_bounds() {
    let cache = KvCache::new(4);
    assert!(cache.layer(4).is_err(), "layer(4) must be Err for 4-layer cache");
    assert!(cache.layer(100).is_err());
}

#[test]
fn test_kv_cache_layer_mut_out_of_bounds() {
    let mut cache = KvCache::new(4);
    assert!(cache.layer_mut(4).is_err(), "layer_mut(4) must be Err");
    assert!(cache.layer_mut(usize::MAX).is_err());
}

#[test]
fn test_kv_cache_layer_mut_append() {
    let mut cache = KvCache::new(2);
    let k = DynTensor::zeros(&[1, 2, 5, 4], DType::F32, &cpu()).unwrap();
    let v = DynTensor::zeros(&[1, 2, 5, 4], DType::F32, &cpu()).unwrap();
    // Append to layer 0 only.
    cache.layer_mut(0).unwrap().append(&k, &v).unwrap();
    assert_eq!(cache.layer(0).unwrap().seq_len(), 5);
    assert_eq!(cache.layer(1).unwrap().seq_len(), 0);
    // seq_len() on the full cache returns first non-empty layer's seq_len.
    assert_eq!(cache.seq_len(), 5);
}

// ---------------------------------------------------------------------------
// WhisperConfig construction and validation
// ---------------------------------------------------------------------------

#[test]
fn test_whisper_config_large_v3_turbo_validates() {
    let config = WhisperConfig::large_v3_turbo();
    config.validate().expect("large_v3_turbo config must validate");
}

#[test]
fn test_whisper_config_tiny_validates() {
    let config = WhisperConfig::whisper_tiny();
    config.validate().expect("whisper_tiny config must validate");
}

#[test]
fn test_whisper_config_test_tiny_validates() {
    let config = tiny_config();
    config.validate().expect("tiny test config must validate");
}

#[test]
fn test_whisper_config_zero_d_model_rejected() {
    let config = tiny_config().with_d_model(0);
    assert!(config.validate().is_err(), "d_model=0 must fail validation");
}

#[test]
fn test_whisper_config_zero_heads_rejected() {
    let config = tiny_config().with_encoder_attention_heads(0);
    assert!(config.validate().is_err(), "encoder_attention_heads=0 must fail");
    let config2 = tiny_config().with_decoder_attention_heads(0);
    assert!(config2.validate().is_err(), "decoder_attention_heads=0 must fail");
}

#[test]
fn test_whisper_config_d_model_not_divisible_by_heads() {
    let config = tiny_config().with_d_model(15).with_encoder_attention_heads(4);
    assert!(config.validate().is_err(), "15 not divisible by 4 must fail");
}

#[test]
fn test_whisper_config_head_dim_computation() {
    let config = tiny_config(); // d_model=16, heads=2
    assert_eq!(config.encoder_head_dim(), 8);
    assert_eq!(config.decoder_head_dim(), 8);
}

#[test]
fn test_whisper_config_builder_chainable() {
    let config = WhisperConfig::whisper_tiny()
        .with_d_model(64)
        .with_encoder_attention_heads(4)
        .with_decoder_attention_heads(4)
        .with_encoder_layers(2)
        .with_decoder_layers(2)
        .with_encoder_ffn_dim(128)
        .with_decoder_ffn_dim(128)
        .with_vocab_size(100)
        .with_num_mel_bins(16)
        .with_max_source_positions(32)
        .with_max_target_positions(24);
    config.validate().expect("chained config must validate");
    assert_eq!(config.d_model, 64);
    assert_eq!(config.encoder_layers, 2);
    assert_eq!(config.decoder_layers, 2);
    assert_eq!(config.vocab_size, 100);
    assert_eq!(config.max_target_positions, 24);
}

// ---------------------------------------------------------------------------
// Decoder layer count matches config
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_block_count_matches_config() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let mut model = crate::WhisperModel::load(&vb, config.clone()).unwrap();

    // Verify decode produces output shaped [1, seq_len, vocab_size].
    // Decoder must have `config.decoder_layers` blocks internally.
    // We cannot directly count blocks (private field), but we can verify
    // that the model loads and forwards with the expected shapes.
    let tokens = DynTensor::from_vec_u32(vec![0; 2], &[1, 2], &cpu()).unwrap();
    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(
        logits.dims(),
        &[1, 2, config.vocab_size],
        "decoder output shape must be [batch, seq_len, vocab_size]"
    );
}

// ---------------------------------------------------------------------------
// Cache reset clears cached data
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_reset_clears_all_layers() {
    let mut cache = KvCache::new(4);
    let k = DynTensor::zeros(&[1, 2, 5, 4], DType::F32, &cpu()).unwrap();
    let v = DynTensor::zeros(&[1, 2, 5, 4], DType::F32, &cpu()).unwrap();

    // Populate all layers.
    for i in 0..4 {
        cache.layer_mut(i).unwrap().append(&k, &v).unwrap();
    }
    assert_eq!(cache.seq_len(), 5);
    assert!(!cache.is_empty());

    // Reset.
    cache.reset();
    assert!(cache.is_empty(), "all layers must be empty after reset");
    assert_eq!(cache.seq_len(), 0);
    for i in 0..4 {
        assert!(cache.layer(i).unwrap().is_empty());
    }
}

#[test]
fn test_kv_cache_layer_reset_clears_key_and_value() {
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::zeros(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let v = DynTensor::zeros(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 3);

    layer.reset();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
    assert!(layer.key().unwrap().is_none(), "key must be None after reset");
    assert!(layer.value().unwrap().is_none(), "value must be None after reset");
}

#[test]
fn test_kv_cache_clear_preserves_capacity() {
    let mut layer = KvCacheLayer::empty();
    let k = DynTensor::zeros(&[1, 2, 10, 4], DType::F32, &cpu()).unwrap();
    let v = DynTensor::zeros(&[1, 2, 10, 4], DType::F32, &cpu()).unwrap();
    layer.append(&k, &v).unwrap();
    let cap_before = layer.buffer_capacity();
    assert!(cap_before >= 10);

    layer.clear();
    assert!(layer.is_empty());
    assert_eq!(layer.seq_len(), 0);
    assert_eq!(layer.buffer_capacity(), cap_before, "clear must preserve capacity");
}

// ---------------------------------------------------------------------------
// Model-level cache reset
// ---------------------------------------------------------------------------

#[test]
fn test_model_reset_kv_cache() {
    let mut model = tiny_model();
    let enc_out = tiny_encoder_output();
    let tokens = DynTensor::from_vec_u32(vec![0; 3], &[1, 3], &cpu()).unwrap();

    // Run one decode step to populate caches.
    model.decode(&tokens, &enc_out, true, 0).unwrap();

    // Reset and run again — should succeed without error.
    model.reset_kv_cache();
    let logits = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits.dims()[0], 1);
}

// ---------------------------------------------------------------------------
// Position computation from cache state
// ---------------------------------------------------------------------------

#[test]
fn test_position_offset_multi_step_decode() {
    let mut model = tiny_model();
    let config = tiny_config();
    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();

    // Step 1: 4 initial tokens at position 0.
    let tokens = DynTensor::from_vec_u32(vec![0; 4], &[1, 4], &cpu()).unwrap();
    let logits1 = model.decode(&tokens, &enc_out, true, 0).unwrap();
    assert_eq!(logits1.dims(), &[1, 4, config.vocab_size]);

    // Step 2: 1 new token at position_offset=4.
    let token2 = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();
    let logits2 = model.decode(&token2, &enc_out, false, 4).unwrap();
    assert_eq!(logits2.dims(), &[1, 1, config.vocab_size]);

    // Step 3: another token at position_offset=5.
    let token3 = DynTensor::from_vec_u32(vec![0], &[1, 1], &cpu()).unwrap();
    let logits3 = model.decode(&token3, &enc_out, false, 5).unwrap();
    assert_eq!(logits3.dims(), &[1, 1, config.vocab_size]);
}

// ---------------------------------------------------------------------------
// Causal mask generation with offset
// ---------------------------------------------------------------------------

#[test]
fn test_causal_mask_generation() {
    let mask = causal_mask(8, DType::F32, &cpu()).unwrap();
    assert_eq!(mask.dims(), &[8, 8]);
    let flat = mask.to_flat_vec::<f32>().unwrap();
    for i in 0..8 {
        for j in 0..8 {
            let val = flat[i * 8 + j];
            if j <= i {
                assert_eq!(val, 0.0, "mask[{i}][{j}] must be 0.0 (attend)");
            } else {
                assert!(val.is_infinite() && val < 0.0, "mask[{i}][{j}] must be -inf (block)");
            }
        }
    }
}

#[test]
fn test_causal_mask_slice_with_offset() {
    // Simulates how the decoder slices the causal mask with a position offset.
    let max_positions = 16;
    let mask = causal_mask(max_positions, DType::F32, &cpu()).unwrap();

    // At offset=4, seq_len=1, total_kv_len=5:
    // mask.narrow(0, 4, 1) -> row 4
    // mask.narrow(1, 0, 5) -> columns 0..5
    // Row 4: [0, 0, 0, 0, 0, -inf, -inf, ...] -> first 5 are [0,0,0,0,0]
    let offset = 4;
    let seq_len = 1;
    let total_kv = offset + seq_len;
    let sliced = mask.narrow(0, offset, seq_len).unwrap();
    let sliced = sliced.narrow(1, 0, total_kv).unwrap();
    assert_eq!(sliced.dims(), &[1, 5]);
    let flat = sliced.to_flat_vec::<f32>().unwrap();
    // All 5 values should be 0.0 (position 4 attends to positions 0..=4).
    for (j, &val) in flat.iter().enumerate() {
        assert_eq!(val, 0.0, "sliced_mask[0][{j}] must be 0.0 at offset=4");
    }
}

#[test]
fn test_causal_mask_initial_prompt_slice() {
    // Initial prompt with seq_len=4 at offset=0.
    let mask = causal_mask(16, DType::F32, &cpu()).unwrap();
    let sliced = mask.narrow(0, 0, 4).unwrap();
    let sliced = sliced.narrow(1, 0, 4).unwrap();
    assert_eq!(sliced.dims(), &[4, 4]);
    let flat = sliced.to_flat_vec::<f32>().unwrap();
    // Lower-triangular 4x4.
    for i in 0..4 {
        for j in 0..4 {
            let val = flat[i * 4 + j];
            if j <= i {
                assert_eq!(val, 0.0);
            } else {
                assert!(val.is_infinite() && val < 0.0);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DecodeConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn test_decode_config_defaults() {
    let config = DecodeConfig::default();
    assert_eq!(config.max_length, MAX_DECODE_LENGTH);
    assert_eq!(config.max_length, 224);
    assert_eq!(config.compression_ratio_threshold, DEFAULT_COMPRESSION_RATIO_THRESHOLD);
    assert_eq!(config.avg_logprob_threshold, DEFAULT_AVG_LOGPROB_THRESHOLD);
    assert!(config.suppress_tokens.is_empty());
    assert!(!config.initial_tokens.is_empty(), "default must have initial tokens");
    assert_eq!(config.initial_tokens, vec![50258, 50259, 50360, 50364]);
    assert!(config.seed.is_none(), "default seed must be None (greedy)");
}

#[test]
fn test_decode_config_validates_ok() {
    let config = DecodeConfig::default();
    config.validate().expect("default config must validate");
}

#[test]
fn test_decode_config_zero_max_length_rejected() {
    let config = DecodeConfig::default().with_max_length(0);
    assert!(config.validate().is_err(), "max_length=0 must fail validation");
}

#[test]
fn test_decode_config_nan_threshold_rejected() {
    let config = DecodeConfig::default().with_compression_ratio_threshold(f64::NAN);
    assert!(config.validate().is_err(), "NaN compression_ratio_threshold must fail");
    let config2 = DecodeConfig::default().with_avg_logprob_threshold(f64::INFINITY);
    assert!(config2.validate().is_err(), "Inf avg_logprob_threshold must fail");
}

#[test]
fn test_decode_config_empty_initial_tokens_rejected() {
    let config = DecodeConfig::default().with_initial_tokens(Vec::new());
    assert!(config.validate().is_err(), "empty initial_tokens must fail");
}

#[test]
fn test_decode_config_builder_methods() {
    let config = DecodeConfig::default()
        .with_max_length(100)
        .with_seed(Some(42))
        .with_suppress_tokens(vec![1, 2, 3])
        .with_compression_ratio_threshold(3.0)
        .with_avg_logprob_threshold(-0.5);
    config.validate().expect("custom config must validate");
    assert_eq!(config.max_length, 100);
    assert_eq!(config.seed, Some(42));
    assert_eq!(config.suppress_tokens, vec![1, 2, 3]);
    assert!((config.compression_ratio_threshold - 3.0).abs() < f64::EPSILON);
    assert!((config.avg_logprob_threshold - (-0.5)).abs() < f64::EPSILON);
}

// ---------------------------------------------------------------------------
// KvCacheLayer dim constraint
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_new_requires_dim_2() {
    assert!(KvCacheLayer::new(2, 100).is_ok(), "dim=2 must succeed");
    assert!(KvCacheLayer::new(0, 100).is_err(), "dim=0 must fail");
    assert!(KvCacheLayer::new(1, 100).is_err(), "dim=1 must fail");
    assert!(KvCacheLayer::new(3, 100).is_err(), "dim=3 must fail");
}

// ---------------------------------------------------------------------------
// KvCacheLayer key/value accessors
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_key_value_accessors() {
    let mut layer = KvCacheLayer::empty();
    // Before any append, key/value are None.
    assert!(layer.key().unwrap().is_none());
    assert!(layer.value().unwrap().is_none());
    assert!(layer.k().unwrap().is_none()); // candle compat
    assert!(layer.v().unwrap().is_none());

    // After append, key/value are Some with correct shape.
    let k = DynTensor::zeros(&[1, 2, 3, 8], DType::F32, &cpu()).unwrap();
    let v = DynTensor::zeros(&[1, 2, 3, 8], DType::F32, &cpu()).unwrap();
    layer.append(&k, &v).unwrap();
    let cached_k = layer.key().unwrap().expect("key must be Some after append");
    let cached_v = layer.value().unwrap().expect("value must be Some after append");
    assert_eq!(cached_k.dims(), &[1, 2, 3, 8]);
    assert_eq!(cached_v.dims(), &[1, 2, 3, 8]);
}

// ---------------------------------------------------------------------------
// KvCacheLayer weight generation / invalidate
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_invalidate_increments_gen() {
    let mut layer = KvCacheLayer::empty();
    assert_eq!(layer.weight_generation(), 0);

    let k = DynTensor::zeros(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let v = DynTensor::zeros(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    layer.append(&k, &v).unwrap();
    assert_eq!(layer.seq_len(), 3);

    layer.invalidate();
    assert_eq!(layer.weight_generation(), 1);
    assert_eq!(layer.seq_len(), 0, "invalidate must clear seq_len");
    // Buffer capacity is preserved.
    assert!(layer.buffer_capacity() > 0, "invalidate preserves buffer capacity");
}

// ---------------------------------------------------------------------------
// KvCacheLayer append returns full accumulated K/V
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_append_returns_full_kv() {
    let mut layer = KvCacheLayer::empty();
    let k1 = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let v1 = DynTensor::ones(&[1, 2, 3, 4], DType::F32, &cpu()).unwrap();
    let (fk1, fv1) = layer.append(&k1, &v1).unwrap();
    assert_eq!(fk1.dims(), &[1, 2, 3, 4], "first append: full K shape");
    assert_eq!(fv1.dims(), &[1, 2, 3, 4], "first append: full V shape");

    // Drop views before next append to avoid COW copy.
    drop(fk1);
    drop(fv1);

    let k2 = DynTensor::ones(&[1, 2, 2, 4], DType::F32, &cpu()).unwrap();
    let v2 = DynTensor::ones(&[1, 2, 2, 4], DType::F32, &cpu()).unwrap();
    let (fk2, fv2) = layer.append(&k2, &v2).unwrap();
    assert_eq!(fk2.dims(), &[1, 2, 5, 4], "second append: full K = 3+2 = 5 along seq dim");
    assert_eq!(fv2.dims(), &[1, 2, 5, 4]);
}

// ---------------------------------------------------------------------------
// Decoder forward_no_cache
// ---------------------------------------------------------------------------

#[test]
fn test_decoder_forward_no_cache_shape() {
    let config = tiny_config();
    let vb = VarBuilder::zeros(DType::F32, &cpu());
    let model = crate::WhisperModel::load(&vb, config.clone()).unwrap();
    let decoder = model.decoder();

    let tokens = DynTensor::from_vec_u32(vec![0; 4], &[1, 4], &cpu()).unwrap();
    let enc_out = DynTensor::zeros(
        &[1, config.max_source_positions, config.d_model],
        DType::F32,
        &cpu(),
    )
    .unwrap();
    let logits = decoder.forward_no_cache(&tokens, &enc_out).unwrap();
    assert_eq!(logits.dims(), &[1, 4, config.vocab_size]);
}

// ---------------------------------------------------------------------------
// KvCache dim accessor
// ---------------------------------------------------------------------------

#[test]
fn test_kv_cache_layer_dim_is_always_2() {
    let layer = KvCacheLayer::empty();
    assert_eq!(layer.dim(), 2, "KvCacheLayer dim is always 2 (sequence dimension)");
}

// ---------------------------------------------------------------------------
// Multiple config presets validate
// ---------------------------------------------------------------------------

#[test]
fn test_all_config_presets_validate() {
    let presets: Vec<(&str, WhisperConfig)> = vec![
        ("large_v3_turbo", WhisperConfig::large_v3_turbo()),
        ("whisper_tiny", WhisperConfig::whisper_tiny()),
        ("whisper_base", WhisperConfig::whisper_base()),
        ("whisper_small", WhisperConfig::whisper_small()),
        ("whisper_medium", WhisperConfig::whisper_medium()),
        ("whisper_large_v2", WhisperConfig::whisper_large_v2()),
    ];
    for (name, config) in presets {
        config.validate().unwrap_or_else(|e| panic!("{name} config must validate: {e}"));
    }
}
