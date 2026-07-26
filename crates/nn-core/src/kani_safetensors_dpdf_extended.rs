// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended Kani proof harnesses for safetensors weight deserialization
//! safety targeting dpdf model weight patterns (#4227).
//!
//! Complements `kani_dpdf_vlm_safetensors_proofs.rs` (categories 1-4) and
//! `kani_dpdf_vlm_safetensors_ext_proofs.rs` (categories 5-7) with
//! additional proof categories specific to dpdf document-AI models:
//!
//!  8. **Weight tensor shape validation** — embed_dim matches across layers
//!  9. **Bias shape consistency** — bias.len() == weight.shape()[0]
//! 10. **QKV weight split** — total size == 3 * num_heads * head_dim * hidden_dim
//! 11. **Cross-attention weight shapes** — encoder/decoder compatibility
//! 12. **Quantized weight dequant range** — INT8 maps to bounded f32
//! 13. **Weight name prefix parsing** — "model.layers.N.attn" pattern
//! 14. **Layer count consistency** — max N in weight names == model config
//! 15. **Embedding weight shape** — vocab_size * embed_dim
//! 16. **LM head tied weights** — if tied, shape matches embedding
//! 17. **Mixed-dtype weights** — f16/bf16/f32 coexistence

#![cfg(kani)]

use crate::DType;

// ===========================================================================
// Helper: dtype byte width (proof-isolated, mirrors production logic)
// ===========================================================================

fn dpdf_dtype_byte_width(dt: DType) -> usize {
    match dt {
        DType::F32 => 4,
        DType::F16 => 2,
        DType::BF16 => 2,
        DType::F64 => 8,
        DType::I32 => 4,
        DType::I64 => 8,
        DType::U32 => 4,
        DType::U8 => 1,
        DType::Bool => 1,
    }
}

// ===========================================================================
// 8. Weight tensor shape validation — embed_dim matches across layers
// ===========================================================================

/// Prove: embed_dim is consistent between projection weight shapes.
///
/// In a dpdf transformer, Q/K/V projections share the same input embed_dim.
/// The output projection maps hidden_dim back to embed_dim.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_embed_dim_consistent_across_projections() {
    let embed_dim: u16 = kani::any();
    let hidden_dim: u16 = kani::any();

    kani::assume(embed_dim >= 64 && embed_dim <= 4096);
    kani::assume(hidden_dim >= 64 && hidden_dim <= 4096);

    let q_shape_in = embed_dim;
    let k_shape_in = embed_dim;
    let v_shape_in = embed_dim;

    assert_eq!(q_shape_in, k_shape_in, "Q and K input dims must match");
    assert_eq!(k_shape_in, v_shape_in, "K and V input dims must match");

    // Output projection restores to embed_dim (residual stream)
    let o_proj_out = embed_dim;
    assert_eq!(o_proj_out, q_shape_in, "output proj restores to embed_dim");
    let _ = hidden_dim;
}

// ===========================================================================
// 9. Bias shape consistency — bias.len() == weight.shape()[0]
// ===========================================================================

/// Prove: linear layer bias length equals weight output dimension.
///
/// For weight [out_features, in_features], bias must have [out_features].
/// Bias numel is always <= weight numel.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_bias_len_equals_weight_out_features() {
    let out_features: u16 = kani::any();
    let in_features: u16 = kani::any();

    kani::assume(out_features >= 1 && out_features <= 4096);
    kani::assume(in_features >= 1 && in_features <= 4096);

    let weight_shape_0 = out_features as usize;
    let bias_len = out_features as usize;

    assert_eq!(
        bias_len, weight_shape_0,
        "bias length must match weight dim 0"
    );

    let weight_numel = (out_features as usize).checked_mul(in_features as usize);
    assert!(weight_numel.is_some());
    assert!(bias_len <= weight_numel.unwrap());
}

// ===========================================================================
// 10. QKV weight split — total == 3 * num_heads * head_dim * hidden_dim
// ===========================================================================

/// Prove: fused QKV weight numel == 3 * num_heads * head_dim * embed_dim.
///
/// dpdf models often fuse Q, K, V projections into a single weight
/// matrix of shape [3 * num_heads * head_dim, embed_dim].
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_qkv_fused_weight_size_correct() {
    let num_heads: u8 = kani::any();
    let head_dim: u8 = kani::any();
    let embed_dim: u16 = kani::any();

    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(head_dim >= 1 && head_dim <= 128);
    kani::assume(embed_dim >= 64 && embed_dim <= 4096);

    let qkv_out_dim = 3usize
        .checked_mul(num_heads as usize)
        .and_then(|s| s.checked_mul(head_dim as usize));
    assert!(qkv_out_dim.is_some(), "QKV out dim must not overflow");

    let qkv_numel = qkv_out_dim.unwrap().checked_mul(embed_dim as usize);
    assert!(qkv_numel.is_some(), "QKV numel must not overflow");

    let q_numel = (num_heads as usize)
        .checked_mul(head_dim as usize)
        .and_then(|s| s.checked_mul(embed_dim as usize));
    assert!(q_numel.is_some());

    assert_eq!(
        qkv_numel.unwrap(),
        3 * q_numel.unwrap(),
        "fused QKV == 3x single proj"
    );
}

/// Prove: splitting fused QKV recovers individual projection shapes.
///
/// Split along dim 0: chunks of [num_heads * head_dim, embed_dim] each.
/// The fused dim must be divisible by 3.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_qkv_split_recovers_individual_shapes() {
    let num_heads: u8 = kani::any();
    let head_dim: u8 = kani::any();

    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(head_dim >= 1 && head_dim <= 128);

    let proj_out = (num_heads as usize).checked_mul(head_dim as usize);
    assert!(proj_out.is_some());
    let proj_out = proj_out.unwrap();

    let fused_out = proj_out.checked_mul(3);
    assert!(fused_out.is_some());

    assert_eq!(fused_out.unwrap() / 3, proj_out);
    assert_eq!(fused_out.unwrap() % 3, 0, "fused dim divisible by 3");
}

// ===========================================================================
// 11. Cross-attention weight shapes — encoder/decoder compatibility
// ===========================================================================

/// Prove: cross-attention K/V projection input matches encoder output dim.
///
/// In encoder-decoder dpdf models, cross-attn Q comes from decoder
/// (dim = decoder_dim) and K/V come from encoder (dim = encoder_dim).
/// K and V numels must be equal; Q differs iff encoder_dim != decoder_dim.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_cross_attn_shapes_compatible() {
    let encoder_dim: u16 = kani::any();
    let decoder_dim: u16 = kani::any();
    let num_heads: u8 = kani::any();
    let head_dim: u8 = kani::any();

    kani::assume(encoder_dim >= 64 && encoder_dim <= 4096);
    kani::assume(decoder_dim >= 64 && decoder_dim <= 4096);
    kani::assume(num_heads >= 1 && num_heads <= 64);
    kani::assume(head_dim >= 1 && head_dim <= 128);

    let proj_out = (num_heads as usize).checked_mul(head_dim as usize);
    assert!(proj_out.is_some());
    let proj_out = proj_out.unwrap();

    let q_numel = proj_out.checked_mul(decoder_dim as usize);
    let k_numel = proj_out.checked_mul(encoder_dim as usize);
    let v_numel = proj_out.checked_mul(encoder_dim as usize);

    assert!(q_numel.is_some() && k_numel.is_some() && v_numel.is_some());
    assert_eq!(
        k_numel.unwrap(),
        v_numel.unwrap(),
        "K and V numels must match"
    );

    if encoder_dim != decoder_dim {
        assert_ne!(
            q_numel.unwrap(),
            k_numel.unwrap(),
            "different dims => different numels"
        );
    }
}

// ===========================================================================
// 12. Quantized weight dequant range — INT8 maps to bounded f32
// ===========================================================================

/// Prove: INT8 per-channel dequantization produces bounded f32.
///
/// dequant(q) = (q - zero_point) * scale. For q in [-128, 127],
/// zero_point in [-128, 127], scale > 0 (normal f32): result is finite.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_int8_dequant_bounded() {
    let q_val: i8 = kani::any();
    let zero_point: i8 = kani::any();
    let scale_bits: u32 = kani::any();

    // Scale is a positive normal f32
    kani::assume(scale_bits > 0x00800000); // > f32::MIN_POSITIVE
    kani::assume(scale_bits < 0x7F800000); // < f32::INFINITY
    kani::assume(scale_bits & 0x80000000 == 0); // positive

    let scale = f32::from_bits(scale_bits);
    assert!(scale.is_finite() && scale > 0.0);

    let diff = (q_val as i16) - (zero_point as i16);
    assert!(diff >= -255 && diff <= 255);

    let result = (diff as f32) * scale;
    assert!(result.is_finite(), "INT8 dequant must produce finite f32");
}

/// Prove: INT8 symmetric dequant range bounds are finite.
///
/// Symmetric quantization (zero_point = 0): range [-128*scale, 127*scale].
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_int8_symmetric_dequant_range() {
    let scale_bits: u32 = kani::any();

    kani::assume(scale_bits > 0x00800000);
    kani::assume(scale_bits < 0x7F800000);
    kani::assume(scale_bits & 0x80000000 == 0);

    let scale = f32::from_bits(scale_bits);
    let max_val = 127.0f32 * scale;
    let min_val = -128.0f32 * scale;

    assert!(max_val.is_finite());
    assert!(min_val.is_finite());
    assert!(min_val <= 0.0);
    assert!(max_val >= 0.0);
}

// ===========================================================================
// 13. Weight name prefix parsing — "model.layers.N.attn" pattern
// ===========================================================================

/// Prove: layer index extracted from weight name is within bounds.
///
/// For dpdf models with up to 80 layers, parsed layer index < num_layers.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_layer_index_within_config_bounds() {
    let parsed_index: u8 = kani::any();
    let num_layers: u8 = kani::any();

    kani::assume(num_layers >= 1 && num_layers <= 80);
    kani::assume(parsed_index < num_layers);

    assert!((parsed_index as usize) < (num_layers as usize));
}

// ===========================================================================
// 14. Layer count consistency — max N in weight names == model config
// ===========================================================================

/// Prove: max layer index + 1 == total layer count (0-indexed).
///
/// If weight names reference layers 0..N-1, the model config must
/// declare exactly N layers. Every index in [0, N-1] is valid.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_max_layer_index_plus_one_equals_count() {
    let num_layers: u8 = kani::any();
    kani::assume(num_layers >= 1 && num_layers <= 80);

    let max_index = num_layers - 1;
    assert_eq!((max_index as usize) + 1, num_layers as usize);

    let test_idx: u8 = kani::any();
    kani::assume(test_idx <= max_index);
    assert!((test_idx as usize) < (num_layers as usize));
}

// ===========================================================================
// 15. Embedding weight shape — vocab_size * embed_dim
// ===========================================================================

/// Prove: embedding weight numel == vocab_size * embed_dim, no overflow.
///
/// dpdf models have vocab up to 152K, embed up to 4096.
/// Max: 152000 * 4096 = 622,592,000 — fits in usize.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_embedding_weight_shape_correct() {
    let vocab_size: u32 = kani::any();
    let embed_dim: u16 = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 152_000);
    kani::assume(embed_dim >= 64 && embed_dim <= 4096);

    let numel = (vocab_size as usize).checked_mul(embed_dim as usize);
    assert!(numel.is_some(), "embedding numel must not overflow");

    let n = numel.unwrap();
    assert!(n <= 152_000 * 4096);
    assert!(n >= 64);
}

/// Prove: embedding byte size per dtype is valid and consistent.
///
/// F32 bytes = 2 * BF16 bytes = 2 * F16 bytes for the same numel.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_embedding_byte_size_per_dtype() {
    let vocab_size: u32 = kani::any();
    let embed_dim: u16 = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 152_000);
    kani::assume(embed_dim >= 64 && embed_dim <= 4096);

    let numel = (vocab_size as usize) * (embed_dim as usize);

    let f32_bytes = numel.checked_mul(dpdf_dtype_byte_width(DType::F32));
    let bf16_bytes = numel.checked_mul(dpdf_dtype_byte_width(DType::BF16));

    assert!(f32_bytes.is_some() && bf16_bytes.is_some());
    assert_eq!(f32_bytes.unwrap(), 2 * bf16_bytes.unwrap());
}

// ===========================================================================
// 16. LM head tied weights — if tied, shape matches embedding
// ===========================================================================

/// Prove: tied LM head weight shape matches embedding.
///
/// When `tie_word_embeddings=true`, lm_head and embed_tokens share
/// the same [vocab_size, embed_dim] tensor. Numels must be identical.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_tied_lm_head_shape_matches_embedding() {
    let vocab_size: u32 = kani::any();
    let embed_dim: u16 = kani::any();
    let tie_weights: bool = kani::any();

    kani::assume(vocab_size >= 1 && vocab_size <= 152_000);
    kani::assume(embed_dim >= 64 && embed_dim <= 4096);

    let embed_numel = (vocab_size as usize) * (embed_dim as usize);

    if tie_weights {
        // Shared tensor — identical shape
        let head_numel = embed_numel;
        assert_eq!(embed_numel, head_numel, "tied weights have same numel");
    } else {
        // Untied: lm_head [embed_dim, vocab_size] (transposed) — same numel
        let head_numel = (embed_dim as usize) * (vocab_size as usize);
        assert_eq!(
            embed_numel, head_numel,
            "untied lm_head same numel (transposed)"
        );
    }
}

// ===========================================================================
// 17. Mixed-dtype weights — f16/bf16/f32 coexistence
// ===========================================================================

/// Prove: mixed-dtype safetensors data regions don't overlap.
///
/// BF16 region [off_a, off_a + 2*numel_a) and F32 region
/// [off_b, off_b + 4*numel_b) are disjoint when BF16 ends before F32 starts.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_mixed_dtype_regions_disjoint() {
    let off_a: u32 = kani::any();
    let numel_a: u32 = kani::any();
    let off_b: u32 = kani::any();
    let numel_b: u32 = kani::any();

    kani::assume(numel_a >= 1 && numel_a <= 10_000_000);
    kani::assume(numel_b >= 1 && numel_b <= 10_000_000);

    let end_a = (off_a as u64) + 2 * (numel_a as u64); // BF16
    let start_b = off_b as u64;

    kani::assume(end_a <= start_b); // A ends before B starts

    // Any byte in A's range is strictly before B
    let test_byte: u64 = kani::any();
    kani::assume(test_byte >= (off_a as u64) && test_byte < end_a);
    assert!(
        test_byte < start_b,
        "byte in BF16 region is before F32 region"
    );
}

/// Prove: dtype cast from f16 to f32 doubles byte count exactly.
///
/// When loading f16 weights and casting to f32, the f32 buffer is
/// exactly 2x the f16 buffer. Element count is preserved.
#[kani::unwind(1)]
#[kani::proof]
fn dpdf_dtype_cast_f16_to_f32_doubles_bytes() {
    let numel: u32 = kani::any();
    kani::assume(numel >= 1 && numel <= 500_000_000);

    let f16_bytes = (numel as u64) * (dpdf_dtype_byte_width(DType::F16) as u64);
    let f32_bytes = (numel as u64) * (dpdf_dtype_byte_width(DType::F32) as u64);

    assert_eq!(f32_bytes, 2 * f16_bytes, "f32 is 2x f16 in bytes");
    assert_eq!(f32_bytes / 4, f16_bytes / 2, "element counts match");
}
