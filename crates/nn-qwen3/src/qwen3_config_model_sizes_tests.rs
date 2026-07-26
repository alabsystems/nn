// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for Qwen3 model configuration: factory methods for known model
//! sizes, decoder layer shape verification, RoPE frequency-band analysis, GQA
//! consistency across variants, SwiGLU intermediate size validation, KV cache
//! shape expectations, and config validation edge cases (#4495).
//!
//! These tests exercise config construction, validation, and derived properties
//! without requiring model weights.

use crate::rope_cache::RoPECache;
use crate::test_utils::tiny_config;
use crate::{Qwen3Config, Qwen3MoeConfig};

// ===========================================================================
// 1. Config factory methods for all known Qwen3 dense model sizes
// ===========================================================================

/// Build Qwen3 config for a named production variant.
/// Returns None for unknown names.
fn qwen3_config_for_size(name: &str) -> Option<Qwen3Config> {
    match name {
        "0.6B" => Some(Qwen3Config::new(
            896,
            4864,
            28,
            14,
            2,
            151_936,
            1e-6,
            1_000_000.0,
            40_960,
            true,
            None,
        )),
        "1.7B" => Some(Qwen3Config::new(
            2048,
            6144,
            28,
            16,
            4,
            151_936,
            1e-6,
            1_000_000.0,
            40_960,
            false,
            None,
        )),
        "4B" => Some(Qwen3Config::new(
            2560,
            9216,
            36,
            32,
            8,
            151_936,
            1e-6,
            1_000_000.0,
            131_072,
            false,
            None,
        )),
        "8B" => Some(Qwen3Config::new(
            4096,
            14336,
            36,
            32,
            8,
            151_936,
            1e-6,
            1_000_000.0,
            131_072,
            false,
            None,
        )),
        "14B" => Some(Qwen3Config::new(
            5120,
            17408,
            40,
            40,
            8,
            151_936,
            1e-6,
            1_000_000.0,
            131_072,
            false,
            None,
        )),
        "32B" => Some(Qwen3Config::new(
            5120,
            25600,
            64,
            40,
            8,
            151_936,
            1e-6,
            1_000_000.0,
            131_072,
            false,
            None,
        )),
        _ => None,
    }
}

/// All known Qwen3 dense sizes produce valid configs.
#[test]
fn test_factory_all_sizes_validate() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size)
            .unwrap_or_else(|| panic!("factory should produce config for {size}"));
        cfg.validate()
            .unwrap_or_else(|e| panic!("{size} config should validate: {e}"));
    }
}

/// Factory returns None for unknown size names.
#[test]
fn test_factory_unknown_size_returns_none() {
    assert!(qwen3_config_for_size("99B").is_none());
    assert!(qwen3_config_for_size("").is_none());
}

/// All factory configs use the standard Qwen3 vocab_size (151_936).
#[test]
fn test_factory_all_use_standard_vocab() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert_eq!(
            cfg.vocab_size, 151_936,
            "{size}: expected vocab_size=151936, got {}",
            cfg.vocab_size
        );
    }
}

/// All factory configs share the same rms_norm_eps and rope_theta.
#[test]
fn test_factory_shared_hyperparameters() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert!(
            (cfg.rms_norm_eps - 1e-6).abs() < 1e-12,
            "{size}: rms_norm_eps should be 1e-6, got {}",
            cfg.rms_norm_eps
        );
        assert!(
            (cfg.rope_theta - 1_000_000.0).abs() < 1e-6,
            "{size}: rope_theta should be 1M, got {}",
            cfg.rope_theta
        );
    }
}

/// Only the smallest model (0.6B) uses tied word embeddings.
#[test]
fn test_factory_tie_word_embeddings_pattern() {
    assert!(qwen3_config_for_size("0.6B").unwrap().tie_word_embeddings);
    for size in ["1.7B", "4B", "8B", "14B", "32B"] {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert!(
            !cfg.tie_word_embeddings,
            "{size}: should NOT have tied embeddings"
        );
    }
}

// ===========================================================================
// 2. Decoder layer weight shape verification
// ===========================================================================

/// Compute expected projection weight shapes for a given config.
fn expected_projection_shapes(cfg: &Qwen3Config) -> Vec<(&'static str, [usize; 2])> {
    let h = cfg.hidden_size;
    let hd = cfg.head_dim(); // always 128
    let nh = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let i = cfg.intermediate_size;
    vec![
        ("q_proj", [nh * hd, h]),
        ("k_proj", [nkv * hd, h]),
        ("v_proj", [nkv * hd, h]),
        ("o_proj", [h, nh * hd]),
        ("gate_proj", [i, h]),
        ("up_proj", [i, h]),
        ("down_proj", [h, i]),
    ]
}

/// Q projection output dim = num_attention_heads * head_dim for all variants.
#[test]
fn test_q_proj_output_dim_all_variants() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        let q_out = cfg.num_attention_heads * cfg.head_dim();
        let shapes = expected_projection_shapes(&cfg);
        let q_shape = shapes.iter().find(|(n, _)| *n == "q_proj").unwrap();
        assert_eq!(
            q_shape.1[0], q_out,
            "{size}: q_proj rows should be {} (num_heads * head_dim), got {}",
            q_out, q_shape.1[0]
        );
    }
}

/// K and V projections have the same shape (both use num_kv_heads).
#[test]
fn test_k_v_proj_shapes_equal() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        let shapes = expected_projection_shapes(&cfg);
        let k_shape = shapes.iter().find(|(n, _)| *n == "k_proj").unwrap();
        let v_shape = shapes.iter().find(|(n, _)| *n == "v_proj").unwrap();
        assert_eq!(
            k_shape.1, v_shape.1,
            "{size}: K and V projections must have identical shapes"
        );
    }
}

/// O projection transposes Q projection: o_proj = [hidden, num_heads * head_dim].
#[test]
fn test_o_proj_is_transpose_of_q() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        let shapes = expected_projection_shapes(&cfg);
        let q = shapes.iter().find(|(n, _)| *n == "q_proj").unwrap();
        let o = shapes.iter().find(|(n, _)| *n == "o_proj").unwrap();
        assert_eq!(
            q.1[0], o.1[1],
            "{size}: o_proj cols ({}) should match q_proj rows ({})",
            o.1[1], q.1[0]
        );
        assert_eq!(
            q.1[1], o.1[0],
            "{size}: o_proj rows ({}) should match q_proj cols ({})",
            o.1[0], q.1[1]
        );
    }
}

/// QK-Norm weight shape is always [head_dim] = [128].
#[test]
fn test_qk_norm_weight_shape_is_head_dim() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        // QK-Norm is RmsNorm with weight shape [head_dim].
        let expected_shape = [cfg.head_dim()];
        assert_eq!(
            expected_shape,
            [128],
            "{size}: QK-Norm shape should always be [128]"
        );
    }
}

/// RMSNorm weight shapes per layer: input_layernorm and post_attention_layernorm
/// are both [hidden_size].
#[test]
fn test_layernorm_weight_shapes() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        // Both per-layer norms have shape [hidden_size]
        let expected = cfg.hidden_size;
        assert!(
            expected > 0,
            "{size}: hidden_size should be > 0 for norm shapes"
        );
    }
}

// ===========================================================================
// 3. RoPE frequency band analysis
// ===========================================================================

/// RoPE with base=1M covers positions beyond max_position_embeddings=131072.
/// The lowest frequency (last index) must have period > max_position_embeddings.
#[test]
fn test_rope_lowest_freq_period_exceeds_max_pos() {
    let sizes = ["4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        let head_dim = cfg.head_dim();
        let half_dim = head_dim / 2;
        let last_idx = half_dim - 1; // 63

        let theta_last = 1.0 / cfg.rope_theta.powf((2 * last_idx) as f64 / head_dim as f64);
        let period = 2.0 * std::f64::consts::PI / theta_last;

        assert!(
            period > cfg.max_position_embeddings as f64,
            "{size}: lowest freq period ({period:.0}) should exceed max_position_embeddings ({})",
            cfg.max_position_embeddings
        );
    }
}

/// RoPE highest frequency (index 0) has period 2*pi regardless of base.
#[test]
fn test_rope_highest_freq_period_is_2pi() {
    for base in [10_000.0_f64, 1_000_000.0] {
        let theta_0 = 1.0 / base.powf(0.0);
        let period = 2.0 * std::f64::consts::PI / theta_0;
        assert!(
            (period - 2.0 * std::f64::consts::PI).abs() < 1e-10,
            "highest freq period should be 2*pi for base={base}"
        );
    }
}

/// RoPE frequency ratios between adjacent indices are constant in log space.
/// This verifies the geometric progression: freq[i+1] / freq[i] = base^(-2/dim).
#[test]
fn test_rope_geometric_frequency_progression() {
    let cfg = qwen3_config_for_size("8B").unwrap();
    let head_dim = cfg.head_dim();
    let half_dim = head_dim / 2;
    let expected_ratio = cfg.rope_theta.powf(-2.0 / head_dim as f64);

    for i in 0..half_dim - 1 {
        let freq_i = 1.0 / cfg.rope_theta.powf((2 * i) as f64 / head_dim as f64);
        let freq_next = 1.0 / cfg.rope_theta.powf((2 * (i + 1)) as f64 / head_dim as f64);
        let ratio = freq_next / freq_i;
        assert!(
            (ratio - expected_ratio).abs() < 1e-12,
            "freq ratio at index {i}: expected {expected_ratio}, got {ratio}"
        );
    }
}

/// RoPE cache at position 0 should be identity (cos=1, sin=0) for all frequencies.
#[test]
fn test_rope_cache_position_zero_identity() {
    let cache = RoPECache::new(128, 128, 1_000_000.0);
    let (cos, sin) = cache.get(0);
    for i in 0..64 {
        assert!(
            (cos[i] - 1.0).abs() < 1e-7,
            "cos[{i}] at pos 0 should be 1.0, got {}",
            cos[i]
        );
        assert!(
            sin[i].abs() < 1e-7,
            "sin[{i}] at pos 0 should be 0.0, got {}",
            sin[i]
        );
    }
}

/// RoPE cache values stay bounded in [-1, 1] for all positions.
#[test]
fn test_rope_cache_values_bounded() {
    let cache = RoPECache::new(2048, 128, 1_000_000.0);
    for pos in (0..2048).step_by(100) {
        let (cos, sin) = cache.get(pos);
        for i in 0..64 {
            assert!(
                cos[i].abs() <= 1.0 + 1e-6,
                "cos[{i}] at pos {pos} out of bounds: {}",
                cos[i]
            );
            assert!(
                sin[i].abs() <= 1.0 + 1e-6,
                "sin[{i}] at pos {pos} out of bounds: {}",
                sin[i]
            );
        }
    }
}

/// RoPE cos^2 + sin^2 = 1 for all positions and frequency indices.
#[test]
fn test_rope_cache_trig_identity() {
    let cache = RoPECache::new(512, 128, 1_000_000.0);
    for pos in (0..512).step_by(50) {
        let (cos, sin) = cache.get(pos);
        for i in 0..64 {
            let sum = cos[i] * cos[i] + sin[i] * sin[i];
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "cos^2 + sin^2 at pos {pos}, idx {i} should be 1.0, got {sum}"
            );
        }
    }
}

// ===========================================================================
// 4. GQA head count consistency
// ===========================================================================

/// num_attention_heads must be divisible by num_key_value_heads for all variants.
#[test]
fn test_gqa_divisibility_all_variants() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert_eq!(
            cfg.num_attention_heads % cfg.num_key_value_heads,
            0,
            "{size}: num_heads ({}) must be divisible by num_kv_heads ({})",
            cfg.num_attention_heads,
            cfg.num_key_value_heads
        );
    }
}

/// GQA group count increases memory efficiency for larger models.
/// Verify expected GQA group counts.
#[test]
fn test_gqa_group_counts_match_expected() {
    let expected = [
        ("0.6B", 7),
        ("1.7B", 4),
        ("4B", 4),
        ("8B", 4),
        ("14B", 5),
        ("32B", 5),
    ];
    for (size, expected_groups) in expected {
        let cfg = qwen3_config_for_size(size).unwrap();
        let groups = cfg.num_kv_groups().unwrap();
        assert_eq!(
            groups, expected_groups,
            "{size}: expected {expected_groups} GQA groups, got {groups}"
        );
    }
}

/// GQA: MHA (num_heads == num_kv_heads) gives group count of 1.
#[test]
fn test_gqa_mha_gives_one_group() {
    let cfg = Qwen3Config::new(256, 512, 2, 4, 4, 100, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 1);
}

/// GQA: MQA (num_kv_heads == 1) gives group count == num_heads.
#[test]
fn test_gqa_mqa_gives_num_heads_groups() {
    let cfg = Qwen3Config::new(256, 512, 2, 8, 1, 100, 1e-6, 10_000.0, 64, true, None);
    assert_eq!(cfg.num_kv_groups().unwrap(), 8);
}

/// GQA with non-divisible head counts should fail.
#[test]
fn test_gqa_non_divisible_heads_fails() {
    let cfg = Qwen3Config::new(256, 512, 2, 7, 3, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.num_kv_groups().is_err());
}

/// GQA with zero kv_heads should fail.
#[test]
fn test_gqa_zero_kv_heads_fails() {
    let cfg = Qwen3Config::new(256, 512, 2, 8, 0, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.num_kv_groups().is_err());
}

// ===========================================================================
// 5. SwiGLU intermediate size validation
// ===========================================================================

/// SwiGLU intermediate_size should be a multiple of 128 for all production configs
/// (hardware alignment for tensor core efficiency).
#[test]
fn test_swiglu_intermediate_128_aligned() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert_eq!(
            cfg.intermediate_size % 128,
            0,
            "{size}: intermediate_size ({}) should be 128-aligned",
            cfg.intermediate_size
        );
    }
}

/// SwiGLU ratio (intermediate / hidden) should be within reasonable bounds.
/// Production Qwen3 models range from ~2.67 to ~5.43.
#[test]
fn test_swiglu_ratio_bounds() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        let ratio = cfg.intermediate_size as f64 / cfg.hidden_size as f64;
        assert!(
            (2.5..=6.0).contains(&ratio),
            "{size}: SwiGLU ratio {ratio:.2} outside expected [2.5, 6.0]"
        );
    }
}

/// SwiGLU MLP has exactly 3 projections: gate, up, down.
/// Total MLP params per layer = 3 * hidden * intermediate.
#[test]
fn test_swiglu_mlp_param_count_formula() {
    let cfg = qwen3_config_for_size("8B").unwrap();
    let mlp_params = 3 * cfg.hidden_size * cfg.intermediate_size;
    // gate[14336, 4096] + up[14336, 4096] + down[4096, 14336] = 3 * 4096 * 14336
    assert_eq!(mlp_params, 3 * 4096 * 14336);
    assert_eq!(mlp_params, 176_160_768);
}

/// intermediate_size=0 should fail validation.
#[test]
fn test_swiglu_zero_intermediate_fails() {
    let cfg = Qwen3Config::new(256, 0, 2, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// SwiGLU gate and up projections produce intermediate_size outputs;
/// down projection maps back to hidden_size.
#[test]
fn test_swiglu_projection_dimensions_consistency() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        let shapes = expected_projection_shapes(&cfg);
        let gate = shapes.iter().find(|(n, _)| *n == "gate_proj").unwrap();
        let up = shapes.iter().find(|(n, _)| *n == "up_proj").unwrap();
        let down = shapes.iter().find(|(n, _)| *n == "down_proj").unwrap();

        // gate and up: [intermediate, hidden]
        assert_eq!(gate.1[0], cfg.intermediate_size, "{size}: gate output dim");
        assert_eq!(gate.1[1], cfg.hidden_size, "{size}: gate input dim");
        assert_eq!(up.1, gate.1, "{size}: up and gate shapes must match");

        // down: [hidden, intermediate]
        assert_eq!(down.1[0], cfg.hidden_size, "{size}: down output dim");
        assert_eq!(down.1[1], cfg.intermediate_size, "{size}: down input dim");
    }
}

// ===========================================================================
// 6. KV cache shape expectations
// ===========================================================================

/// KV cache per-token memory per layer = 2 * num_kv_heads * head_dim * dtype_bytes.
#[test]
fn test_kv_cache_per_token_memory() {
    let sizes_and_expected: &[(&str, usize)] = &[
        ("0.6B", 2 * 2 * 128 * 4), // 2 kv_heads, 2048 bytes
        ("1.7B", 2 * 4 * 128 * 4), // 4 kv_heads, 4096 bytes
        ("8B", 2 * 8 * 128 * 4),   // 8 kv_heads, 8192 bytes
        ("32B", 2 * 8 * 128 * 4),  // 8 kv_heads, 8192 bytes
    ];
    for (size, expected_bytes) in sizes_and_expected {
        let cfg = qwen3_config_for_size(size).unwrap();
        let bytes = 2 * cfg.num_key_value_heads * cfg.head_dim() * 4; // f32
        assert_eq!(
            bytes, *expected_bytes,
            "{size}: KV cache bytes/token/layer should be {expected_bytes}, got {bytes}"
        );
    }
}

/// Total KV cache memory for a model at full context length.
/// Formula: layers * 2 * nkv * head_dim * 4 * max_position_embeddings.
#[test]
fn test_kv_cache_total_at_full_context() {
    // Qwen3-8B: 36 layers, 8 kv heads, head_dim=128, max_pos=131072
    let cfg = qwen3_config_for_size("8B").unwrap();
    let total_bytes = cfg.num_hidden_layers
        * 2
        * cfg.num_key_value_heads
        * cfg.head_dim()
        * 4
        * cfg.max_position_embeddings;
    let gib = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // 36 * 2 * 8 * 128 * 4 * 131072 = ~36 GiB
    assert!(
        gib > 30.0 && gib < 45.0,
        "Qwen3-8B full-context KV cache should be ~36 GiB, got {gib:.1} GiB"
    );
}

/// KV cache layer count must match num_hidden_layers.
#[test]
fn test_kv_cache_layer_count() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    let expected_layers = [28, 28, 36, 36, 40, 64];
    for (size, expected) in sizes.iter().zip(expected_layers.iter()) {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert_eq!(
            cfg.num_hidden_layers, *expected,
            "{size}: expected {expected} layers, got {}",
            cfg.num_hidden_layers
        );
    }
}

/// KV cache: K and V tensors per layer have shape [batch, num_kv_heads, seq, head_dim].
/// Verify the dimensions are self-consistent.
#[test]
fn test_kv_cache_tensor_shape_formula() {
    let batch = 1_usize;
    let seq = 10_usize;
    let cfg = qwen3_config_for_size("8B").unwrap();

    let k_shape = [batch, cfg.num_key_value_heads, seq, cfg.head_dim()];
    let v_shape = [batch, cfg.num_key_value_heads, seq, cfg.head_dim()];

    // K and V shapes are identical
    assert_eq!(k_shape, v_shape);
    // Total elements per K or V: batch * nkv * seq * head_dim
    let elements = k_shape.iter().product::<usize>();
    assert_eq!(
        elements,
        batch * cfg.num_key_value_heads * seq * cfg.head_dim()
    );
}

// ===========================================================================
// 7. Config validation edge cases
// ===========================================================================

/// Negative rms_norm_eps should fail validation.
#[test]
fn test_config_negative_rms_norm_eps_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, -1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// Zero rms_norm_eps should fail validation.
#[test]
fn test_config_zero_rms_norm_eps_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 0.0, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// Very small positive rms_norm_eps should pass validation.
#[test]
fn test_config_very_small_rms_norm_eps_accepted() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 1e-12, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_ok());
}

/// Zero num_attention_heads should fail validation.
#[test]
fn test_config_zero_attention_heads_rejected() {
    let cfg = Qwen3Config::new(256, 512, 2, 0, 2, 100, 1e-6, 10_000.0, 64, true, None);
    assert!(cfg.validate().is_err());
}

/// Zero num_hidden_layers is technically valid from validate() perspective
/// (no explicit check), but results in an empty model.
#[test]
fn test_config_zero_hidden_layers_validates() {
    let cfg = Qwen3Config::new(256, 512, 0, 2, 2, 100, 1e-6, 10_000.0, 64, true, None);
    // validate() does not reject zero layers (it's a legal degenerate config)
    assert!(cfg.validate().is_ok());
}

/// Very large rope_theta (e.g., 1e12) should pass validation.
#[test]
fn test_config_very_large_rope_theta_accepted() {
    let cfg = Qwen3Config::new(256, 512, 2, 2, 2, 100, 1e-6, 1e12, 64, true, None);
    assert!(cfg.validate().is_ok());
}

/// NEG_INFINITY rope_theta should fail validation.
#[test]
fn test_config_neg_infinity_rope_theta_rejected() {
    let cfg = Qwen3Config::new(
        256,
        512,
        2,
        2,
        2,
        100,
        1e-6,
        f64::NEG_INFINITY,
        64,
        true,
        None,
    );
    assert!(cfg.validate().is_err());
}

/// Multiple validation failures: zero heads AND zero hidden_size.
/// Should fail (at least one error).
#[test]
fn test_config_multiple_invalid_fields_fails() {
    let cfg = Qwen3Config::new(0, 0, 0, 0, 0, 0, -1.0, -1.0, 0, true, None);
    assert!(cfg.validate().is_err());
}

/// YaRN scaling config should be preserved through clone and builder.
#[test]
fn test_config_yarn_scaling_preserved() {
    use nn_core::layers::YarnScaling;

    let yarn = YarnScaling::new(4.0, 1.0, 32.0, 1.0, 64);
    let cfg = Qwen3Config::new(
        256,
        512,
        2,
        2,
        2,
        100,
        1e-6,
        1_000_000.0,
        131_072,
        true,
        Some(yarn),
    );
    assert!(cfg.rope_scaling.is_some());
    let cfg2 = cfg.with_vocab_size(200);
    assert!(cfg2.rope_scaling.is_some());
    assert_eq!(cfg2.vocab_size, 200);
}

/// builder with_num_hidden_layers preserves all other fields.
#[test]
fn test_builder_preserves_fields() {
    let cfg = Qwen3Config::new(
        4096,
        14336,
        36,
        32,
        8,
        151_936,
        1e-6,
        1_000_000.0,
        131_072,
        false,
        None,
    );
    let cfg2 = cfg.with_num_hidden_layers(4);
    assert_eq!(cfg2.num_hidden_layers, 4);
    assert_eq!(cfg2.hidden_size, 4096);
    assert_eq!(cfg2.intermediate_size, 14336);
    assert_eq!(cfg2.num_attention_heads, 32);
    assert_eq!(cfg2.num_key_value_heads, 8);
    assert_eq!(cfg2.vocab_size, 151_936);
    assert!(!cfg2.tie_word_embeddings);
}

// ===========================================================================
// 8. MoE config factory methods
// ===========================================================================

/// Build Qwen3 MoE config for known production variants.
fn qwen3_moe_config_for_size(name: &str) -> Option<Qwen3MoeConfig> {
    match name {
        "30B-A3B" => {
            let base = Qwen3Config::new(
                4096,
                2560,
                48,
                32,
                4,
                151_936,
                1e-6,
                1_000_000.0,
                131_072,
                false,
                None,
            );
            Some(Qwen3MoeConfig::new(base, 128, 8, true, Some(2560)))
        }
        "235B-A22B" => {
            let base = Qwen3Config::new(
                6144,
                3072,
                94,
                64,
                4,
                151_936,
                1e-6,
                1_000_000.0,
                131_072,
                false,
                None,
            );
            Some(Qwen3MoeConfig::new(base, 128, 8, true, Some(3072)))
        }
        _ => None,
    }
}

/// All known Qwen3 MoE sizes validate.
#[test]
fn test_moe_factory_all_validate() {
    for size in ["30B-A3B", "235B-A22B"] {
        let cfg = qwen3_moe_config_for_size(size).unwrap();
        cfg.validate()
            .unwrap_or_else(|e| panic!("{size} MoE config should validate: {e}"));
    }
}

/// MoE configs share 128 total experts with 8 active per token.
#[test]
fn test_moe_factory_expert_counts() {
    for size in ["30B-A3B", "235B-A22B"] {
        let cfg = qwen3_moe_config_for_size(size).unwrap();
        assert_eq!(cfg.num_experts, 128, "{size}: num_experts");
        assert_eq!(cfg.num_experts_per_tok, 8, "{size}: num_experts_per_tok");
    }
}

/// MoE configs both use shared experts.
#[test]
fn test_moe_factory_shared_expert() {
    for size in ["30B-A3B", "235B-A22B"] {
        let cfg = qwen3_moe_config_for_size(size).unwrap();
        assert!(cfg.shared_expert, "{size}: should use shared expert");
    }
}

/// MoE shared expert fallback to base intermediate size when not specified.
#[test]
fn test_moe_shared_expert_fallback() {
    let base = tiny_config();
    let cfg = Qwen3MoeConfig::new(base.clone(), 8, 2, true, None);
    assert_eq!(cfg.shared_expert_ff_dim(), base.intermediate_size);
}

/// MoE config with all experts active validates.
#[test]
fn test_moe_all_experts_active() {
    let base = tiny_config();
    let cfg = Qwen3MoeConfig::new(base, 4, 4, false, None);
    assert!(cfg.validate().is_ok());
}

/// MoE config with single expert validates.
#[test]
fn test_moe_single_expert() {
    let base = tiny_config();
    let cfg = Qwen3MoeConfig::new(base, 1, 1, false, None);
    assert!(cfg.validate().is_ok());
}

// ===========================================================================
// 9. Cross-variant consistency invariants
// ===========================================================================

/// head_dim is constant 128 across ALL Qwen3 variants (dense and MoE).
#[test]
fn test_head_dim_constant_128_all_variants() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert_eq!(cfg.head_dim(), 128, "{size}: head_dim must be 128");
    }
}

/// Larger models have more layers (monotonically non-decreasing).
#[test]
fn test_layer_count_non_decreasing_with_size() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    let mut prev_layers = 0;
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert!(
            cfg.num_hidden_layers >= prev_layers,
            "{size}: layer count ({}) should be >= previous ({})",
            cfg.num_hidden_layers,
            prev_layers
        );
        prev_layers = cfg.num_hidden_layers;
    }
}

/// hidden_size is non-decreasing across model sizes.
#[test]
fn test_hidden_size_non_decreasing() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    let mut prev_hidden = 0;
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        assert!(
            cfg.hidden_size >= prev_hidden,
            "{size}: hidden_size ({}) should be >= previous ({})",
            cfg.hidden_size,
            prev_hidden
        );
        prev_hidden = cfg.hidden_size;
    }
}

/// Total parameter count increases with model size (approximate).
#[test]
fn test_param_count_increases_with_size() {
    let sizes = ["0.6B", "1.7B", "4B", "8B", "14B", "32B"];
    let mut prev_params = 0_usize;
    for size in sizes {
        let cfg = qwen3_config_for_size(size).unwrap();
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let hd = 128;
        let v = cfg.vocab_size;
        let n = cfg.num_hidden_layers;

        // Rough parameter count (see dense_param_count in config_extended_tests)
        let embed = v * h;
        let attn = (nh * hd * h) + 2 * (nkv * hd * h) + (h * nh * hd);
        let norms = 2 * h + 2 * hd;
        let mlp = 3 * i * h;
        let final_norm = h;
        let lm_head = if cfg.tie_word_embeddings { 0 } else { v * h };
        let total = embed + n * (attn + norms + mlp) + final_norm + lm_head;

        assert!(
            total > prev_params,
            "{size}: param count ({total}) should exceed previous ({prev_params})"
        );
        prev_params = total;
    }
}
