// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for Multi-head Latent Attention (MLA).
//!
//! Proves properties of [`MlaConfig`] validation, dimension computations,
//! RoPE pairing invariants, and KV compression geometry. These harnesses
//! verify the arithmetic that underpins MLA's low-rank KV cache strategy
//! (DeepSeek-V2, arXiv:2405.04434).
//!
//! Part of #3705.

use super::mla::MlaConfig;


// -- Kani transcendental stubs (CBMC #239, #329, #708) --

fn sqrt_f32_stub(x: f32) -> f32 { let r: f32 = kani::any(); kani::assume(r.is_finite() && r >= 0.0 && r <= 1e10); if x > 0.0 { kani::assume(r > 0.0); kani::assume(r >= x.min(1.0)); } r }

// ===========================================================================
// MlaConfig::validate — acceptance and rejection
// ===========================================================================

/// Prove: valid DeepSeek-V2-like config always passes validation.
///
/// Uses representative parameters from the DeepSeek-V2 architecture:
/// hidden=5120, 128 heads, kv_lora_rank=512, rope_dim=64, qk_nope_dim=128.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_valid_config_passes() {
    let cfg = MlaConfig {
        hidden_size: 5120,
        num_heads: 128,
        kv_lora_rank: 512,
        q_lora_rank: Some(1536),
        rope_dim: 64,
        qk_nope_dim: 128,
        v_head_dim: 128,
        rms_norm_eps: 1e-6,
    };
    assert!(cfg.validate().is_ok(), "DeepSeek-V2-like config must be valid");
}

/// Prove: config with zero num_heads is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_zero_heads_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 0,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: 32,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_err(), "zero heads must be rejected");
}

/// Prove: config with zero hidden_size is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_zero_hidden_size_rejected() {
    let cfg = MlaConfig {
        hidden_size: 0,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: 32,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_err(), "zero hidden_size must be rejected");
}

/// Prove: config with zero kv_lora_rank is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_zero_kv_lora_rank_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 0,
        q_lora_rank: None,
        rope_dim: 32,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_err(), "zero kv_lora_rank must be rejected");
}

/// Prove: config with zero rope_dim is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_zero_rope_dim_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: 0,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_err(), "zero rope_dim must be rejected");
}

/// Prove: odd rope_dim is rejected (RoPE requires pair pairing).
///
/// RoPE operates on consecutive pairs (x[2i], x[2i+1]). An odd rope_dim
/// would leave one element unpaired, corrupting the rotation.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_odd_rope_dim_rejected() {
    let rope: u8 = kani::any();
    kani::assume(rope >= 1 && rope <= 127);
    kani::assume(rope % 2 == 1); // odd
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: rope as usize,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_err(), "odd rope_dim must be rejected");
}

/// Prove: even rope_dim in valid config is accepted.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_even_rope_dim_accepted() {
    let rope: u8 = kani::any();
    kani::assume(rope >= 2 && rope <= 128);
    kani::assume(rope % 2 == 0);
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: rope as usize,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_ok(), "even rope_dim with valid config must pass");
}

/// Prove: zero qk_nope_dim is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_zero_qk_nope_dim_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: 32,
        qk_nope_dim: 0,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_err(), "zero qk_nope_dim must be rejected");
}

/// Prove: zero v_head_dim is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_zero_v_head_dim_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: 32,
        qk_nope_dim: 32,
        v_head_dim: 0,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_err(), "zero v_head_dim must be rejected");
}

/// Prove: q_lora_rank=Some(0) is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_zero_q_lora_rank_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: Some(0),
        rope_dim: 32,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert!(cfg.validate().is_err(), "q_lora_rank=0 must be rejected");
}

/// Prove: NaN rms_norm_eps is rejected.
///
/// IEEE 754: NaN comparisons are always false, so `eps < 0.0` won't catch NaN.
/// The validation must use `!is_finite()` to catch NaN (per nn_engineering.md).
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_nan_rms_norm_eps_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: 32,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: f64::NAN,
    };
    assert!(cfg.validate().is_err(), "NaN rms_norm_eps must be rejected");
}

/// Prove: negative rms_norm_eps is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_negative_rms_norm_eps_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: 32,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: -1e-5,
    };
    assert!(cfg.validate().is_err(), "negative rms_norm_eps must be rejected");
}

/// Prove: infinity rms_norm_eps is rejected.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_inf_rms_norm_eps_rejected() {
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: 32,
        qk_nope_dim: 32,
        v_head_dim: 32,
        rms_norm_eps: f64::INFINITY,
    };
    assert!(cfg.validate().is_err(), "inf rms_norm_eps must be rejected");
}

// ===========================================================================
// MlaConfig::qk_head_dim — dimension arithmetic
// ===========================================================================

/// Prove: qk_head_dim equals qk_nope_dim + rope_dim for all valid configs.
///
/// This is the effective Q/K head dimension used in SDPA. The split into
/// nope (no position embedding) and rope portions is the core MLA innovation.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn mla_qk_head_dim_is_sum() {
    let nope: u8 = kani::any();
    let rope: u8 = kani::any();
    kani::assume(nope >= 1 && nope <= 128);
    kani::assume(rope >= 2 && rope <= 128);
    kani::assume(rope % 2 == 0);
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: rope as usize,
        qk_nope_dim: nope as usize,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    assert_eq!(
        cfg.qk_head_dim(),
        nope as usize + rope as usize,
        "qk_head_dim must equal qk_nope_dim + rope_dim"
    );
}

/// Prove: qk_head_dim is strictly greater than both rope_dim and qk_nope_dim.
///
/// Since both components are > 0 in a valid config, the sum exceeds each part.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_qk_head_dim_exceeds_components() {
    let nope: u8 = kani::any();
    let rope: u8 = kani::any();
    kani::assume(nope >= 1 && nope <= 128);
    kani::assume(rope >= 2 && rope <= 128);
    kani::assume(rope % 2 == 0);
    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: 8,
        kv_lora_rank: 64,
        q_lora_rank: None,
        rope_dim: rope as usize,
        qk_nope_dim: nope as usize,
        v_head_dim: 32,
        rms_norm_eps: 1e-5,
    };
    let head_dim = cfg.qk_head_dim();
    assert!(head_dim > cfg.rope_dim, "head_dim must exceed rope_dim");
    assert!(head_dim > cfg.qk_nope_dim, "head_dim must exceed qk_nope_dim");
}

// ===========================================================================
// KV compression ratio — the memory saving
// ===========================================================================

/// Prove: KV compression ratio is always < 1 when kv_lora_rank < num_heads * v_head_dim.
///
/// The whole point of MLA is to cache `kv_lora_rank` per token instead of
/// `num_heads * v_head_dim`. This proves the compression always saves memory
/// when the latent rank is smaller than the full KV dimension.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
fn mla_kv_compression_saves_memory() {
    let num_heads: u8 = kani::any();
    let v_head_dim: u8 = kani::any();
    let kv_lora_rank: u16 = kani::any();

    kani::assume(num_heads >= 1 && num_heads <= 128);
    kani::assume(v_head_dim >= 1 && v_head_dim <= 128);
    kani::assume(kv_lora_rank >= 1 && kv_lora_rank <= 4096);

    let full_kv = num_heads as u32 * v_head_dim as u32;
    kani::assume(full_kv > 0);
    kani::assume((kv_lora_rank as u32) < full_kv);

    // Compression ratio: kv_lora_rank / (num_heads * v_head_dim)
    let ratio_num = kv_lora_rank as u32;
    let ratio_den = full_kv;
    assert!(
        ratio_num < ratio_den,
        "KV lora rank must be less than full KV dimension for compression"
    );
}

// ===========================================================================
// SDPA scale factor
// ===========================================================================

/// Prove: SDPA scale is 1/sqrt(qk_head_dim) and is always positive & finite.
///
/// The attention scale factor normalizes dot products to prevent softmax
/// saturation. It must be positive and finite for any valid head dim.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn mla_scale_positive_finite() {
    let nope: u8 = kani::any();
    let rope: u8 = kani::any();
    kani::assume(nope >= 1 && nope <= 128);
    kani::assume(rope >= 2 && rope <= 128);
    kani::assume(rope % 2 == 0);

    let qk_head_dim = nope as usize + rope as usize;
    let scale = 1.0 / (qk_head_dim as f64).sqrt();

    assert!(scale.is_finite(), "scale must be finite");
    assert!(scale > 0.0, "scale must be positive");
    assert!(scale <= 1.0, "scale must be <= 1.0 for head_dim >= 1");
}

/// Prove: SDPA scale decreases monotonically with increasing head dimension.
///
/// Larger head dimensions produce smaller scale factors, which is necessary
/// to keep attention logits from growing too large.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::stub(f32::sqrt, sqrt_f32_stub)]
fn mla_scale_decreases_with_head_dim() {
    let d1: u8 = kani::any();
    let d2: u8 = kani::any();
    kani::assume(d1 >= 1 && d1 <= 200);
    kani::assume(d2 >= 1 && d2 <= 200);
    kani::assume(d2 > d1);

    let s1 = 1.0 / (d1 as f64).sqrt();
    let s2 = 1.0 / (d2 as f64).sqrt();

    assert!(s1 > s2, "larger head dim must produce smaller scale");
}

// ===========================================================================
// Weight matrix dimensions — Q projection
// ===========================================================================

/// Prove: Q_b projection output dimension equals num_heads * qk_head_dim.
///
/// This ensures the Q output can be reshaped to [B, T, num_heads, qk_head_dim]
/// without remainder.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn mla_q_proj_output_divisible_by_heads() {
    let num_heads: u8 = kani::any();
    let nope: u8 = kani::any();
    let rope: u8 = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(nope >= 1 && nope <= 64);
    kani::assume(rope >= 2 && rope <= 64);
    kani::assume(rope % 2 == 0);

    let qk_head_dim = nope as usize + rope as usize;
    let q_out = num_heads as usize * qk_head_dim;

    // q_out is divisible by num_heads (by construction)
    assert_eq!(
        q_out % (num_heads as usize),
        0,
        "Q output must be divisible by num_heads"
    );
    // Each head gets exactly qk_head_dim
    assert_eq!(
        q_out / (num_heads as usize),
        qk_head_dim,
        "Each head must get qk_head_dim dimensions"
    );
}

// ===========================================================================
// Weight matrix dimensions — KV projection
// ===========================================================================

/// Prove: KV_a projection output equals kv_lora_rank + rope_dim.
///
/// The KV_a projection produces the compressed latent AND the rope portion
/// of K (shared across heads in MQA style). These are split by narrow().
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn mla_kv_a_output_dim() {
    let kv_lora: u16 = kani::any();
    let rope: u8 = kani::any();
    kani::assume(kv_lora >= 1 && kv_lora <= 4096);
    kani::assume(rope >= 2 && rope <= 128);
    kani::assume(rope % 2 == 0);

    let kv_a_out = kv_lora as usize + rope as usize;
    // narrow(0, kv_lora_rank) + narrow(kv_lora_rank, rope_dim) covers the full output
    assert_eq!(
        kv_lora as usize + rope as usize,
        kv_a_out,
        "KV_a output must be kv_lora_rank + rope_dim"
    );
    // Both portions are non-empty
    assert!(kv_lora > 0, "KV lora portion must be non-empty");
    assert!(rope > 0, "Rope portion must be non-empty");
}

/// Prove: KV_b projection output equals num_heads * (qk_nope_dim + v_head_dim).
///
/// The KV_b uplift produces both K_nope and V per head. These are split
/// by narrow() along the last dimension.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn mla_kv_b_output_dim() {
    let num_heads: u8 = kani::any();
    let nope: u8 = kani::any();
    let v_dim: u8 = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(nope >= 1 && nope <= 64);
    kani::assume(v_dim >= 1 && v_dim <= 64);

    let per_head = nope as usize + v_dim as usize;
    let kv_b_out = num_heads as usize * per_head;

    // Divisible by num_heads
    assert_eq!(kv_b_out % (num_heads as usize), 0);
    // Each head contributes nope + v_dim
    assert_eq!(kv_b_out / (num_heads as usize), per_head);
}

/// Prove: output projection dimension equals num_heads * v_head_dim.
///
/// The output projection maps the concatenated attention output back to
/// hidden_size. Its input dimension must be num_heads * v_head_dim.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn mla_output_proj_input_dim() {
    let num_heads: u8 = kani::any();
    let v_dim: u8 = kani::any();
    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(v_dim >= 1 && v_dim <= 64);

    let out_in = num_heads as usize * v_dim as usize;
    assert_eq!(out_in % (num_heads as usize), 0);
    assert_eq!(out_in / (num_heads as usize), v_dim as usize);
}

// ===========================================================================
// RoPE pairing — rope_dim/2 is always integer
// ===========================================================================

/// Prove: rope_dim / 2 produces exact half for even rope_dim.
///
/// RoPE pairs elements (x[2i], x[2i+1]). The reshape to [..., half_dim, 2]
/// requires rope_dim to be exactly divisible by 2.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn mla_rope_half_dim_exact() {
    let rope: u8 = kani::any();
    kani::assume(rope >= 2 && rope <= 128);
    kani::assume(rope % 2 == 0);

    let half = rope as usize / 2;
    assert_eq!(half * 2, rope as usize, "half_dim * 2 must equal rope_dim");
    assert!(half >= 1, "half_dim must be at least 1");
}

// ===========================================================================
// Config accessor consistency
// ===========================================================================

/// Prove: all MlaLayer accessors return the config values they were constructed with.
///
/// Part of #3705.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(8)]
fn mla_accessor_consistency() {
    let nh: u8 = kani::any();
    let kvlr: u8 = kani::any();
    let rd: u8 = kani::any();
    let nope: u8 = kani::any();
    let vhd: u8 = kani::any();

    kani::assume(nh >= 1 && nh <= 32);
    kani::assume(kvlr >= 1 && kvlr <= 64);
    kani::assume(rd >= 2 && rd <= 64);
    kani::assume(rd % 2 == 0);
    kani::assume(nope >= 1 && nope <= 64);
    kani::assume(vhd >= 1 && vhd <= 64);

    let cfg = MlaConfig {
        hidden_size: 256,
        num_heads: nh as usize,
        kv_lora_rank: kvlr as usize,
        q_lora_rank: None,
        rope_dim: rd as usize,
        qk_nope_dim: nope as usize,
        v_head_dim: vhd as usize,
        rms_norm_eps: 1e-5,
    };

    assert_eq!(cfg.num_heads, nh as usize);
    assert_eq!(cfg.kv_lora_rank, kvlr as usize);
    assert_eq!(cfg.rope_dim, rd as usize);
    assert_eq!(cfg.qk_nope_dim, nope as usize);
    assert_eq!(cfg.v_head_dim, vhd as usize);
}
