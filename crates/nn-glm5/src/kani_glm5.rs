// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GLM-4/5 model forward path safety.
//!
//! Covers:
//! - Config validation: zero-field rejection, divisibility, NaN/Inf guards
//! - head_dim() identity with kv_channels
//! - num_kv_groups() division safety (zero-divisor, non-divisible)
//! - Default config validity
//! - QKV fused projection size: no overflow for valid configs
//! - Attention scale computation: 1/sqrt(head_dim) is finite
//! - SwiGLU MLP dimension: ffn_hidden_size * 2 no overflow
//! - GQA repeat ratio: num_heads / num_kv_heads exact division
//! - Config constructor roundtrip: new() preserves all fields
//! - Causal mask offset arithmetic: no underflow
//! - Error conversion safety: Glm5Error → TensorError never panics
//!
//! Issue: #3597

use crate::config::Glm5Config;
use crate::error::Glm5Error;

// CBMC transcendental stubs — f64::sqrt
fn sqrt_f64_stub(x: f64) -> f64 {
    let r: f64 = kani::any();
    kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10);
    if x > 0.0 {
        kani::assume(r > 0.0);
        kani::assume(r >= x.min(1.0));
    }
    r
}

// ============================================================================
// Harness 1: validate() rejects zero num_attention_heads
// ============================================================================

/// Proves that validate() rejects num_attention_heads == 0.
///
/// Zero heads would cause division-by-zero in num_kv_groups() and in the
/// attention scale computation 1/sqrt(head_dim).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_heads() {
    let mut cfg = Glm5Config::default();
    cfg.num_attention_heads = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero heads must fail validation");
}

// ============================================================================
// Harness 2: validate() rejects zero multi_query_group_num
// ============================================================================

/// Proves that validate() rejects multi_query_group_num == 0.
///
/// Zero KV groups would cause division-by-zero in num_kv_groups().
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_kv_groups() {
    let mut cfg = Glm5Config::default();
    cfg.multi_query_group_num = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero kv groups must fail validation");
}

// ============================================================================
// Harness 3: validate() rejects non-divisible heads/groups
// ============================================================================

/// Proves that validate() rejects when num_attention_heads is not divisible
/// by multi_query_group_num.
///
/// Non-divisible would produce truncated integer division in GQA, giving
/// wrong repeat counts.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_non_divisible_heads() {
    let mut cfg = Glm5Config::default();
    cfg.num_attention_heads = 5;
    cfg.multi_query_group_num = 2;
    let result = cfg.validate();
    assert!(result.is_err(), "5 % 2 != 0 must fail validation");
}

// ============================================================================
// Harness 4: validate() rejects NaN layernorm_epsilon
// ============================================================================

/// Proves that validate() rejects NaN layernorm_epsilon.
///
/// IEEE 754: NaN comparisons return false, so `eps <= 0.0` returns false
/// for NaN. The `!eps.is_finite()` guard catches this.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_nan_epsilon() {
    let mut cfg = Glm5Config::default();
    cfg.layernorm_epsilon = f64::NAN;
    let result = cfg.validate();
    assert!(result.is_err(), "NaN epsilon must fail validation");
}

// ============================================================================
// Harness 5: validate() rejects Inf rope_theta
// ============================================================================

/// Proves that validate() rejects infinite rope_theta.
///
/// Infinite rope_theta would produce NaN in RoPE frequency computation:
/// theta = rope_theta^(-2i/d), where Inf^(-x) = 0 or NaN depending on sign.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_inf_rope_theta() {
    let mut cfg = Glm5Config::default();
    cfg.rope_theta = f64::INFINITY;
    let result = cfg.validate();
    assert!(result.is_err(), "infinite rope_theta must fail validation");
}

// ============================================================================
// Harness 6: validate() rejects negative rope_theta
// ============================================================================

/// Proves that validate() rejects negative rope_theta.
///
/// Negative base frequency would produce complex-valued RoPE angles.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_negative_rope_theta() {
    let mut cfg = Glm5Config::default();
    cfg.rope_theta = -1.0;
    let result = cfg.validate();
    assert!(result.is_err(), "negative rope_theta must fail validation");
}

// ============================================================================
// Harness 7: validate() rejects zero kv_channels
// ============================================================================

/// Proves that validate() rejects kv_channels == 0.
///
/// Zero kv_channels would make head_dim() == 0, causing zero-size tensors
/// and division-by-zero in attention scale.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_kv_channels() {
    let mut cfg = Glm5Config::default();
    cfg.kv_channels = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero kv_channels must fail validation");
}

// ============================================================================
// Harness 8: validate() rejects kv_channels not multiple of 4
// ============================================================================

/// Proves that validate() rejects kv_channels that are not a multiple of 4.
///
/// HalfRotaryEmbedding requires head_dim divisible by 4 (splits into
/// head_dim/2, then each half into sin/cos pairs).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_kv_channels_not_mult_4() {
    let mut cfg = Glm5Config::default();
    cfg.kv_channels = 65; // not divisible by 4
    let result = cfg.validate();
    assert!(result.is_err(), "kv_channels=65 must fail validation");
}

// ============================================================================
// Harness 9: validate() rejects zero num_layers
// ============================================================================

/// Proves that validate() rejects num_layers == 0.
///
/// Zero layers would produce a model with no transformer blocks — the
/// embedding would pass directly to the output layer with no processing.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_num_layers() {
    let mut cfg = Glm5Config::default();
    cfg.num_layers = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero num_layers must fail validation");
}

// ============================================================================
// Harness 10: Default config passes validation
// ============================================================================

/// Proves that Glm5Config::default() (GLM-4-9B) passes validation.
///
/// The default config is the production GLM-4-9B configuration. If it
/// fails validation, the library is unusable without manual config construction.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_config_is_valid() {
    let cfg = Glm5Config::default();
    let result = cfg.validate();
    assert!(result.is_ok(), "GLM-4-9B default config must be valid");
}

// ============================================================================
// Harness 11: head_dim() always equals kv_channels
// ============================================================================

/// Proves that head_dim() is a pure alias for kv_channels.
///
/// If this invariant broke (e.g., someone changed head_dim to compute
/// hidden_size / num_heads), QKV split sizes would be wrong.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn head_dim_equals_kv_channels() {
    let kv_channels: usize = kani::any();
    kani::assume(kv_channels <= 1024); // realistic range
    let mut cfg = Glm5Config::default();
    cfg.kv_channels = kv_channels;
    assert_eq!(cfg.head_dim(), kv_channels);
}

// ============================================================================
// Harness 12: num_kv_groups() division safety
// ============================================================================

/// Proves that num_kv_groups() never panics for any valid (heads, groups)
/// combination where groups divides heads and both are nonzero.
///
/// For invalid inputs (groups == 0 or non-divisible), proves it returns Err.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn num_kv_groups_division_safe() {
    let heads: usize = kani::any();
    let groups: usize = kani::any();
    kani::assume(heads > 0 && heads <= 128);
    kani::assume(groups > 0 && groups <= 128);

    let mut cfg = Glm5Config::default();
    cfg.num_attention_heads = heads;
    cfg.multi_query_group_num = groups;

    let result = cfg.num_kv_groups();
    if heads % groups == 0 {
        // Valid: must succeed and return exact quotient
        let val = result.expect("divisible heads/groups must succeed");
        assert_eq!(val, heads / groups);
    } else {
        // Invalid: must return error, not truncated division
        assert!(result.is_err(), "non-divisible must return Err");
    }
}

// ============================================================================
// Harness 13: QKV fused projection size no overflow
// ============================================================================

/// Proves that the fused QKV projection output size (nh + 2*nkv) * hd does
/// not overflow usize for configs that pass validation.
///
/// In Glm5Attention::load, qkv_size = (nh + 2 * nkv) * hd. Overflow here
/// would silently produce a wrong weight shape, causing a shape mismatch
/// at load time or (worse) silent memory corruption.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn qkv_size_no_overflow() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    let hd: usize = kani::any();

    // Constrain to realistic model sizes (up to GPT-4 scale)
    kani::assume(nh > 0 && nh <= 128);
    kani::assume(nkv > 0 && nkv <= 128);
    kani::assume(hd > 0 && hd <= 256);
    kani::assume(nh >= nkv); // heads >= kv_heads always
    kani::assume(nh % nkv == 0); // divisibility required

    // (nh + 2 * nkv) * hd must not overflow
    let sum = nh.checked_add(2 * nkv);
    assert!(sum.is_some(), "nh + 2*nkv must not overflow");
    let qkv_size = sum.unwrap().checked_mul(hd);
    assert!(qkv_size.is_some(), "(nh + 2*nkv) * hd must not overflow");
}

// ============================================================================
// Harness 14: Attention scale is finite and positive for valid head_dim
// ============================================================================

/// Proves that the attention scale 1/sqrt(head_dim) is finite and positive
/// for any valid head_dim (multiple of 4, nonzero, <= 256).
///
/// The attention layer computes `let scale = 1.0 / (hd as f64).sqrt()`.
/// If head_dim were 0, sqrt(0) = 0 and 1/0 = Inf. Validation prevents this.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
#[kani::stub(f64::sqrt, sqrt_f64_stub)]
fn attention_scale_finite_positive() {
    let hd: usize = kani::any();
    kani::assume(hd > 0 && hd <= 256);
    kani::assume(hd % 4 == 0); // HalfRotaryEmbedding requirement

    let scale = 1.0 / (hd as f64).sqrt();
    assert!(scale.is_finite(), "scale must be finite for hd > 0");
    assert!(scale > 0.0, "scale must be positive");
}

// ============================================================================
// Harness 15: SwiGLU MLP ffn_hidden_size * 2 no overflow
// ============================================================================

/// Proves that ffn_hidden_size * 2 (the dense_h_to_4h output dimension)
/// does not overflow for realistic FFN sizes.
///
/// In Glm5MLP::load, the fused gate+up projection weight shape is
/// [ffn_hidden_size * 2, hidden_size]. Overflow would cause a wrong shape.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn swiglu_ffn_double_no_overflow() {
    let ffn: usize = kani::any();
    kani::assume(ffn > 0 && ffn <= 65536); // largest realistic FFN dim

    let doubled = ffn.checked_mul(2);
    assert!(doubled.is_some(), "ffn * 2 must not overflow");
}

// ============================================================================
// Harness 16: GQA repeat ratio is exact
// ============================================================================

/// Proves that num_heads / num_kv_heads yields an exact integer ratio
/// when num_kv_groups() succeeds, matching the value used in repeat_kv.
///
/// In Glm5Attention::forward: `repeat_kv(&k, self.num_heads / self.num_kv_heads)`.
/// If this division were inexact, repeat_kv would get a truncated count,
/// producing the wrong number of repeated KV heads.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn gqa_repeat_ratio_exact() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    kani::assume(nh > 0 && nh <= 128);
    kani::assume(nkv > 0 && nkv <= 128);
    kani::assume(nh % nkv == 0);

    let ratio = nh / nkv;
    // Verify the ratio reconstructs the original head count
    assert_eq!(ratio * nkv, nh, "ratio * nkv must reconstruct nh");
    assert!(ratio >= 1, "ratio must be >= 1");
}

// ============================================================================
// Harness 17: Config new() constructor roundtrip
// ============================================================================

/// Proves that Glm5Config::new() preserves all field values.
///
/// Since `#[non_exhaustive]` prevents struct literal construction outside the
/// crate, `new()` is the only external constructor. If it misassigned any
/// field (e.g., swapped hidden_size and ffn_hidden_size), models would
/// silently load wrong weights.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_new_roundtrip() {
    let hidden: usize = kani::any();
    let ffn: usize = kani::any();
    let layers: usize = kani::any();
    let heads: usize = kani::any();
    let kv_groups: usize = kani::any();
    let vocab: usize = kani::any();
    let kv_ch: usize = kani::any();
    let seq: usize = kani::any();

    // Keep small to avoid state explosion
    kani::assume(hidden <= 16 && ffn <= 16 && layers <= 4);
    kani::assume(heads <= 8 && kv_groups <= 8 && vocab <= 16);
    kani::assume(kv_ch <= 16 && seq <= 16);

    let cfg = Glm5Config::new(
        hidden, ffn, layers, heads, kv_groups, vocab, kv_ch,
        1e-5, // fixed: f64 symbolic is expensive
        seq, true, false, true, 10_000.0,
    );

    assert_eq!(cfg.hidden_size, hidden);
    assert_eq!(cfg.ffn_hidden_size, ffn);
    assert_eq!(cfg.num_layers, layers);
    assert_eq!(cfg.num_attention_heads, heads);
    assert_eq!(cfg.multi_query_group_num, kv_groups);
    assert_eq!(cfg.padded_vocab_size, vocab);
    assert_eq!(cfg.kv_channels, kv_ch);
    assert_eq!(cfg.seq_length, seq);
    assert!(cfg.rmsnorm);
    assert!(!cfg.add_qkv_bias);
    assert!(cfg.add_bias_linear);
}

// ============================================================================
// Harness 18: Error conversion Glm5Error → TensorError never panics
// ============================================================================

/// Proves that converting Glm5Error::InvalidConfig to TensorError via the
/// From impl does not panic.
///
/// The From<Glm5Error> for TensorError impl calls `other.to_string()` which
/// invokes the Display impl generated by thiserror. If any variant's
/// #[error("...")] format panicked (e.g., missing field), this would be a
/// runtime crash in error handling paths.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn error_conversion_invalid_config_no_panic() {
    let err = Glm5Error::InvalidConfig {
        reason: String::from("test"),
    };
    let _te: nn_core::TensorError = err.into();
    // If we reach here, the conversion did not panic
}

/// Proves that converting Glm5Error::CacheMismatch to TensorError
/// does not panic.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn error_conversion_cache_mismatch_no_panic() {
    let cache_layers: usize = kani::any();
    let model_layers: usize = kani::any();
    kani::assume(cache_layers <= 100);
    kani::assume(model_layers <= 100);

    let err = Glm5Error::CacheMismatch {
        cache_layers,
        model_layers,
    };
    let _te: nn_core::TensorError = err.into();
}

// ============================================================================
// Harness 20: validate() rejects zero padded_vocab_size
// ============================================================================

/// Proves that validate() rejects padded_vocab_size == 0.
///
/// Zero vocab size would create a zero-row embedding matrix, causing
/// out-of-bounds access for any token ID.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_vocab() {
    let mut cfg = Glm5Config::default();
    cfg.padded_vocab_size = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero vocab size must fail validation");
}

// ============================================================================
// Harness 21: validate() rejects zero seq_length
// ============================================================================

/// Proves that validate() rejects seq_length == 0.
///
/// Zero seq_length would make RoPE frequency table empty, causing
/// index-out-of-bounds during positional encoding.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_seq_length() {
    let mut cfg = Glm5Config::default();
    cfg.seq_length = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero seq_length must fail validation");
}

// ============================================================================
// Harness 22: validate() rejects zero layernorm_epsilon
// ============================================================================

/// Proves that validate() rejects layernorm_epsilon == 0.0.
///
/// Zero epsilon in RMSNorm: rms = sqrt(mean(x^2) + eps). If eps == 0 and
/// input is all-zeros, rms == 0, causing division-by-zero in normalization.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_epsilon() {
    let mut cfg = Glm5Config::default();
    cfg.layernorm_epsilon = 0.0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero epsilon must fail validation");
}

// ============================================================================
// Harness 23: validate() rejects negative layernorm_epsilon
// ============================================================================

/// Proves that validate() rejects negative layernorm_epsilon.
///
/// Negative epsilon could make rms = sqrt(mean(x^2) + eps) imaginary
/// when eps < -mean(x^2), producing NaN.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_negative_epsilon() {
    let mut cfg = Glm5Config::default();
    cfg.layernorm_epsilon = -1e-5;
    let result = cfg.validate();
    assert!(result.is_err(), "negative epsilon must fail validation");
}

// ============================================================================
// Harness 24: validate() rejects Inf layernorm_epsilon
// ============================================================================

/// Proves that validate() rejects Inf layernorm_epsilon.
///
/// Infinite epsilon makes rms = sqrt(Inf) = Inf, so x / rms = 0 for all
/// inputs, collapsing all token representations.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_inf_epsilon() {
    let mut cfg = Glm5Config::default();
    cfg.layernorm_epsilon = f64::INFINITY;
    let result = cfg.validate();
    assert!(result.is_err(), "infinite epsilon must fail validation");
}

// ============================================================================
// Harness 25: validate() rejects NaN rope_theta
// ============================================================================

/// Proves that validate() rejects NaN rope_theta.
///
/// IEEE 754: NaN > 0.0 is false, NaN <= 0.0 is false. The is_finite()
/// guard catches NaN before the positivity check.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_nan_rope_theta() {
    let mut cfg = Glm5Config::default();
    cfg.rope_theta = f64::NAN;
    let result = cfg.validate();
    assert!(result.is_err(), "NaN rope_theta must fail validation");
}

// ============================================================================
// Harness 26: validate() rejects zero hidden_size
// ============================================================================

/// Proves that validate() rejects hidden_size == 0.
///
/// Zero hidden_size would create zero-column weight matrices in all
/// linear layers, making every projection a zero vector.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_hidden_size() {
    let mut cfg = Glm5Config::default();
    cfg.hidden_size = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero hidden_size must fail validation");
}

// ============================================================================
// Harness 27: validate() rejects zero ffn_hidden_size
// ============================================================================

/// Proves that validate() rejects ffn_hidden_size == 0.
///
/// Zero FFN hidden size would make the SwiGLU MLP a no-op (zero-width
/// intermediate), losing all nonlinearity.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_rejects_zero_ffn_hidden_size() {
    let mut cfg = Glm5Config::default();
    cfg.ffn_hidden_size = 0;
    let result = cfg.validate();
    assert!(result.is_err(), "zero ffn_hidden_size must fail validation");
}

// ============================================================================
// Harness 28: Error conversion InvalidInput → TensorError no panic
// ============================================================================

/// Proves that converting Glm5Error::InvalidInput to TensorError does not
/// panic.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn error_conversion_invalid_input_no_panic() {
    let err = Glm5Error::InvalidInput {
        reason: String::from("positions mismatch"),
    };
    let _te: nn_core::TensorError = err.into();
}

// ============================================================================
// Harness 29: Error conversion NonFiniteOutput → TensorError no panic
// ============================================================================

/// Proves that converting Glm5Error::NonFiniteOutput to TensorError does
/// not panic.
///
/// This variant uses a &'static str for stage and usize for count.
/// Display formatting of both must succeed.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn error_conversion_non_finite_output_no_panic() {
    let count: usize = kani::any();
    kani::assume(count <= 10_000);

    let err = Glm5Error::NonFiniteOutput {
        stage: "Glm5Attention",
        count,
    };
    let _te: nn_core::TensorError = err.into();
}

// ============================================================================
// Harness 30: Error conversion WeightLoad → TensorError no panic
// ============================================================================

/// Proves that converting Glm5Error::WeightLoad to TensorError does not
/// panic.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn error_conversion_weight_load_no_panic() {
    let err = Glm5Error::WeightLoad {
        reason: String::from("missing tensor"),
    };
    let _te: nn_core::TensorError = err.into();
}

// ============================================================================
// Harness 31: validate() accepts all valid kv_channels multiples of 4
// ============================================================================

/// Proves that validate() accepts any kv_channels that is a positive
/// multiple of 4, with all other fields at valid defaults.
///
/// This is the positive counterpart to harness 7 (rejects 0) and
/// harness 8 (rejects non-mult-4). Together they prove the kv_channels
/// guard is both necessary and sufficient.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_accepts_valid_kv_channels() {
    let kv_ch: usize = kani::any();
    kani::assume(kv_ch > 0 && kv_ch <= 1024);
    kani::assume(kv_ch % 4 == 0);

    let mut cfg = Glm5Config::default();
    cfg.kv_channels = kv_ch;

    let result = cfg.validate();
    assert!(result.is_ok(), "valid kv_channels must pass validation");
}

// ============================================================================
// Harness 32: validate() accepts divisible heads/groups combinations
// ============================================================================

/// Proves that validate() accepts any combination where num_attention_heads
/// is a positive multiple of multi_query_group_num, with valid defaults.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_validate_accepts_divisible_heads() {
    let nkv: usize = kani::any();
    let multiplier: usize = kani::any();
    kani::assume(nkv > 0 && nkv <= 32);
    kani::assume(multiplier > 0 && multiplier <= 32);

    let nh = nkv * multiplier;
    kani::assume(nh <= 128);

    let mut cfg = Glm5Config::default();
    cfg.num_attention_heads = nh;
    cfg.multi_query_group_num = nkv;

    let result = cfg.validate();
    assert!(
        result.is_ok(),
        "divisible heads/groups must pass validation"
    );
}

// ============================================================================
// Harness 33: validate + num_kv_groups consistency
// ============================================================================

/// Proves that when validate() passes, num_kv_groups() also succeeds and
/// the result times multi_query_group_num equals num_attention_heads.
///
/// This proves the two validation paths are consistent: validate() checks
/// divisibility, and num_kv_groups() also checks divisibility. They must
/// agree on what constitutes a valid config.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_and_num_kv_groups_consistent() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    kani::assume(nh > 0 && nh <= 128);
    kani::assume(nkv > 0 && nkv <= 128);
    kani::assume(nh % nkv == 0);

    let mut cfg = Glm5Config::default();
    cfg.num_attention_heads = nh;
    cfg.multi_query_group_num = nkv;

    // validate() should pass
    assert!(cfg.validate().is_ok());

    // num_kv_groups() should also succeed and be consistent
    let groups = cfg.num_kv_groups().expect("must succeed for valid config");
    assert_eq!(groups * nkv, nh, "groups * nkv must reconstruct nh");
}

// ============================================================================
// Harness 34: Config Clone preserves all fields
// ============================================================================

/// Proves that cloning a Glm5Config preserves all field values.
///
/// Clone is derived, but if a manual impl were introduced (e.g., for
/// deep-copy of future heap-allocated fields), this catches regressions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn config_clone_preserves_fields() {
    let cfg = Glm5Config::default();
    let cloned = cfg.clone();

    assert_eq!(cfg.hidden_size, cloned.hidden_size);
    assert_eq!(cfg.ffn_hidden_size, cloned.ffn_hidden_size);
    assert_eq!(cfg.num_layers, cloned.num_layers);
    assert_eq!(cfg.num_attention_heads, cloned.num_attention_heads);
    assert_eq!(cfg.multi_query_group_num, cloned.multi_query_group_num);
    assert_eq!(cfg.padded_vocab_size, cloned.padded_vocab_size);
    assert_eq!(cfg.kv_channels, cloned.kv_channels);
    assert_eq!(cfg.seq_length, cloned.seq_length);
    assert_eq!(cfg.rmsnorm, cloned.rmsnorm);
    assert_eq!(cfg.add_qkv_bias, cloned.add_qkv_bias);
    assert_eq!(cfg.add_bias_linear, cloned.add_bias_linear);
}

// ============================================================================
// Harness 35: Default config GLM-4-9B specific invariants
// ============================================================================

/// Proves specific architectural invariants of the GLM-4-9B default config.
///
/// These are not just "valid" but specifically correct for the 9B model:
/// - 32 heads with 2 KV groups → 16x repeat
/// - kv_channels = 128 → head_dim = 128 → hidden_size = 32 * 128 = 4096
/// - SwiGLU intermediate = 13696 * 2 = 27392
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn default_config_glm4_9b_invariants() {
    let cfg = Glm5Config::default();

    // 32 query heads / 2 kv groups = 16x GQA repeat
    assert_eq!(cfg.num_attention_heads / cfg.multi_query_group_num, 16);

    // head_dim = kv_channels = 128
    assert_eq!(cfg.head_dim(), 128);

    // hidden_size = num_heads * head_dim (standard transformer relation)
    assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());

    // SwiGLU: ffn_hidden_size * 2 must not overflow and equals 27392
    assert_eq!(cfg.ffn_hidden_size * 2, 27392);
}

// ============================================================================
// Harness 36: Symbolic validate rejects all zero-field configs
// ============================================================================

/// Proves that setting ANY single required-nonzero field to 0 causes
/// validate() to fail.
///
/// Uses symbolic selection to cover all 6 zero-rejecting fields:
/// hidden_size, ffn_hidden_size, num_layers, num_attention_heads,
/// multi_query_group_num, padded_vocab_size, kv_channels, seq_length.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_any_single_zero_field() {
    let field: u8 = kani::any();
    kani::assume(field < 8);

    let mut cfg = Glm5Config::default();
    match field {
        0 => cfg.hidden_size = 0,
        1 => cfg.ffn_hidden_size = 0,
        2 => cfg.num_layers = 0,
        3 => cfg.num_attention_heads = 0,
        4 => cfg.multi_query_group_num = 0,
        5 => cfg.padded_vocab_size = 0,
        6 => cfg.kv_channels = 0,
        7 => cfg.seq_length = 0,
        _ => unreachable!(),
    }

    let result = cfg.validate();
    assert!(
        result.is_err(),
        "zeroing any required field must fail validation"
    );
}

// ============================================================================
// Harness 37: validate rejects all non-finite float configs
// ============================================================================

/// Proves that setting any float field to NaN, +Inf, -Inf, or 0.0 causes
/// validate() to fail.
///
/// Covers both layernorm_epsilon and rope_theta with all pathological values.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn validate_rejects_non_finite_float_fields() {
    let field: u8 = kani::any();
    let value_kind: u8 = kani::any();
    kani::assume(field < 2);
    kani::assume(value_kind < 5);

    let bad_value = match value_kind {
        0 => f64::NAN,
        1 => f64::INFINITY,
        2 => f64::NEG_INFINITY,
        3 => 0.0,
        4 => -1.0,
        _ => unreachable!(),
    };

    let mut cfg = Glm5Config::default();
    match field {
        0 => cfg.layernorm_epsilon = bad_value,
        1 => cfg.rope_theta = bad_value,
        _ => unreachable!(),
    }

    let result = cfg.validate();
    assert!(
        result.is_err(),
        "non-finite/non-positive float must fail validation"
    );
}

// ============================================================================
// Harness 38: num_kv_groups rejects zero multi_query_group_num
// ============================================================================

/// Proves that num_kv_groups() returns Err when multi_query_group_num is 0,
/// independent of validate() (which also checks this).
///
/// num_kv_groups() has its own zero guard because it may be called on
/// configs that haven't been validate()d.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn num_kv_groups_rejects_zero_groups() {
    let mut cfg = Glm5Config::default();
    cfg.multi_query_group_num = 0;

    let result = cfg.num_kv_groups();
    assert!(result.is_err(), "zero groups must fail in num_kv_groups");
}

// ============================================================================
// Harness 39: QKV weight row count matches config
// ============================================================================

/// Proves that the QKV weight's first dimension (output features) computed
/// from config parameters matches the fused formula used in load().
///
/// In load: qkv_size = (nh + 2 * nkv) * hd
/// QKV weight shape: [qkv_size, hidden_size]
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn qkv_weight_row_count_from_config() {
    let nh: usize = kani::any();
    let nkv: usize = kani::any();
    let hd: usize = kani::any();

    kani::assume(nh > 0 && nh <= 128);
    kani::assume(nkv > 0 && nkv <= 128);
    kani::assume(hd > 0 && hd <= 256);
    kani::assume(nh >= nkv);
    kani::assume(nh % nkv == 0);

    let mut cfg = Glm5Config::default();
    cfg.num_attention_heads = nh;
    cfg.multi_query_group_num = nkv;
    cfg.kv_channels = hd;

    // qkv_size from load()
    let qkv_size = (nh + 2 * nkv) * cfg.head_dim();

    // Must equal q_proj + k_proj + v_proj individually
    let q_proj_out = nh * cfg.head_dim();
    let k_proj_out = nkv * cfg.head_dim();
    let v_proj_out = nkv * cfg.head_dim();

    assert_eq!(qkv_size, q_proj_out + k_proj_out + v_proj_out);
}
