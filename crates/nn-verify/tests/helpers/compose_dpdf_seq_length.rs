// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose tests for sequence length and position extrapolation bounds.
//!
//! Verifies IBP and CROWN bound propagation across varying sequence lengths
//! and position encoding extrapolation scenarios relevant to dpdf document
//! understanding models (Qwen3-VL, GLM-OCR, Granite-Docling, FireRed-OCR).
//!
//! Long-document PDF processing requires handling variable-length sequences:
//! multi-page documents produce longer token sequences, KV-cache grows during
//! autoregressive generation, and position encodings must extrapolate beyond
//! training-time maximums.
//!
//! 1.  **Variable sequence length attention IBP**: MHA bounds at len=4, 8, 16
//! 2.  **Position encoding extrapolation IBP**: sinusoidal PE beyond training max
//! 3.  **RoPE extended positions IBP**: RoPE cos/sin at 2x training positions
//! 4.  **Causal mask length effects IBP**: causal attention at different lengths
//! 5.  **Encoder sequence scaling IBP**: linear proj + norm at different seq_lens
//! 6.  **Decoder generation length IBP**: stacked FFN layers at growing lengths
//! 7.  **KV-cache growing context IBP**: attention with increasing KV-cache size
//! 8.  **Sequence truncation bounds IBP**: truncated vs full sequence bounds
//! 9.  **Padding effects on bounds IBP**: zero-padded vs unpadded attention
//! 10. **CROWN tightness at different lengths**: CROWN vs IBP at seq_len=4, 8
//! 11. **Monotone tightening across lengths IBP**: narrower eps -> tighter bounds
//! 12. **Full transformer block at multiple lengths IBP + CROWN**: end-to-end
//! 13. **RoPE frequency decay at long positions IBP**: high-freq dims decay
//! 14. **Cross-attention variable-length IBP**: different Q and KV lengths
//! 15. **Bound width scaling with sequence length IBP**: empirical width growth
//!
//! Dimensions (small for fast verification, structurally representative):
//! - HIDDEN_DIM=32, NUM_HEADS=4, HEAD_DIM=8, FFN_DIM=64
//!
//! Part of #4055: Compose tests for sequence length and position extrapolation bounds.

use super::common::{
    assert_bounds_valid, assert_crown_tighter_when_not_fallback, bounds_min_max, uniform_bounds,
};
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::{AttentionMask, TensorKernelDef};
use nn_verify::{tensor_kernel_to_graph, BoundedTensor, TensorParamBinding};
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions -- small for fast verification, structurally representative
// ---------------------------------------------------------------------------

const HIDDEN_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = HIDDEN_DIM / NUM_HEADS; // 8
const FFN_DIM: usize = 64;
const WEIGHT_MAG: f32 = 0.02;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Constant-weight binding helper.
fn weight_binding(shape: &[usize]) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(shape), WEIGHT_MAG))
}

/// Scalar constant binding.
fn scalar_binding(val: f32) -> TensorParamBinding {
    TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[1]), val))
}

/// Build RoPE cos/sin constant tensors for `[seq, dim]`.
fn rope_cos_sin(seq: usize, d: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut cos_data = vec![0.0f32; seq * d];
    let mut sin_data = vec![0.0f32; seq * d];
    for t in 0..seq {
        for i in 0..d / 2 {
            let freq = (t as f64) / 10000.0_f64.powf(2.0 * i as f64 / d as f64);
            let c = freq.cos() as f32;
            let s = freq.sin() as f32;
            cos_data[t * d + 2 * i] = c;
            cos_data[t * d + 2 * i + 1] = c;
            sin_data[t * d + 2 * i] = s;
            sin_data[t * d + 2 * i + 1] = s;
        }
    }
    (
        ArrayD::from_shape_vec(IxDyn(&[seq, d]), cos_data).expect("cos"),
        ArrayD::from_shape_vec(IxDyn(&[seq, d]), sin_data).expect("sin"),
    )
}

/// Build SiLU activation: SiLU(x) = x * sigmoid(x).
fn add_silu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    shape: &[usize],
) -> nn_dsl::TensorNodeId {
    let sig = b.add_sigmoid(input, shape);
    b.add_binary_mul(input, sig, shape)
}

/// Build a SwiGLU FFN block: gate_proj -> SiLU -> mul(up_proj) -> down_proj.
fn build_swiglu(
    b: &mut TensorBlockBuilder,
    input: nn_dsl::TensorNodeId,
    prefix: &str,
    seq_len: usize,
    hidden: usize,
    ffn: usize,
) -> nn_dsl::TensorNodeId {
    let ffn_shape = [seq_len, ffn];
    let out_shape = [seq_len, hidden];
    let gate_w = b.add_input(&format!("{prefix}_gate_w"), &[ffn, hidden]);
    let up_w = b.add_input(&format!("{prefix}_up_w"), &[ffn, hidden]);
    let down_w = b.add_input(&format!("{prefix}_down_w"), &[hidden, ffn]);

    let gate = b.add_linear(input, gate_w, None, &ffn_shape);
    let gate_act = add_silu(b, gate, &ffn_shape);
    let up = b.add_linear(input, up_w, None, &ffn_shape);
    let h = b.add_binary_mul(gate_act, up, &ffn_shape);
    b.add_linear(h, down_w, None, &out_shape)
}

/// Push SwiGLU weight bindings (gate_w, up_w, down_w).
fn push_swiglu_bindings(bindings: &mut Vec<TensorParamBinding>, hidden: usize, ffn: usize) {
    bindings.push(weight_binding(&[ffn, hidden]));
    bindings.push(weight_binding(&[ffn, hidden]));
    bindings.push(weight_binding(&[hidden, ffn]));
}

/// Build a standard MHA kernel at given sequence length.
fn build_mha_kernel(seq_len: usize, mask: AttentionMask) -> TensorKernelDef {
    let mut b = TensorBlockBuilder::new(&format!("dpdf_seq_mha_{seq_len}"));
    let input = b.add_input("x", &[seq_len, HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

    let out = b
        .add_multi_head_attention(
            input,
            q_w,
            k_w,
            v_w,
            o_w,
            NUM_HEADS,
            mask,
            &[seq_len, HIDDEN_DIM],
        )
        .expect("valid MHA");
    b.build(out).expect("valid MHA kernel")
}

/// Standard MHA bindings (Variable + 4 weight matrices).
fn mha_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
    ]
}

/// Compute output bound width from a `BoundedTensor`.
fn bound_width(bounds: &BoundedTensor) -> f32 {
    let (lo_min, hi_max) = bounds_min_max(bounds);
    hi_max - lo_min
}

// ===========================================================================
// 1. Variable sequence length attention IBP
// ===========================================================================

fn test_mha_at_length(seq_len: usize) -> f32 {
    let def = build_mha_kernel(seq_len, AttentionMask::Standard);
    let bindings = mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[seq_len, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("MHA seq_len={seq_len} IBP: width={width:.6}");
    assert!(
        width.is_finite(),
        "MHA width must be finite at len={seq_len}"
    );
    width
}

#[test]
fn test_seq_length_mha_len4_ibp() {
    test_mha_at_length(4);
}

#[test]
fn test_seq_length_mha_len8_ibp() {
    test_mha_at_length(8);
}

#[test]
fn test_seq_length_mha_len16_ibp() {
    test_mha_at_length(16);
}

// ===========================================================================
// 2. Position encoding extrapolation IBP
// ===========================================================================

/// Verify that sinusoidal PE at positions beyond training max produces
/// bounded attention outputs. PE values remain in [-1, 1] by construction.
#[test]
fn test_seq_length_pe_extrapolation_ibp() {
    let train_max = 8;
    let extrap_len = 16; // 2x training max

    // Build attention with PE added to input
    let mut b = TensorBlockBuilder::new("dpdf_seq_pe_extrap");
    let input = b.add_input("x", &[extrap_len, HIDDEN_DIM]);
    let pe = b.add_input("pe", &[extrap_len, HIDDEN_DIM]);
    let shape = [extrap_len, HIDDEN_DIM];

    // x + PE (additive position encoding)
    let positioned = b.add_binary_add(input, pe, &shape);

    // Linear projection to verify bounds propagate
    let w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(positioned, w, None, &shape);
    let def = b.build(out).expect("valid PE extrap kernel");

    // Sinusoidal PE at extrapolated positions (still in [-1, 1])
    let pe_data = super::common::sinusoidal_pe(extrap_len, HIDDEN_DIM);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(pe_data),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[extrap_len, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "PE extrapolation (train={train_max}, extrap={extrap_len}) IBP: \
         bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "lower must be finite");
    assert!(hi_max.is_finite(), "upper must be finite");
}

// ===========================================================================
// 3. RoPE at extended positions IBP
// ===========================================================================

/// RoPE cos/sin at 2x training positions still produces bounded rotations.
#[test]
fn test_seq_length_rope_extended_positions_ibp() {
    let extended_len = 16;

    let mut b = TensorBlockBuilder::new("dpdf_seq_rope_extended");
    let input = b.add_input("x", &[extended_len, HIDDEN_DIM]);
    let cos_pe = b.add_input("cos_pe", &[extended_len, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin_pe", &[extended_len, HIDDEN_DIM]);
    let shape = [extended_len, HIDDEN_DIM];

    // RoPE: x * cos + rotate_half(x) * sin
    // Approximate rotate_half as: x * cos - x * sin (simplified)
    let x_cos = b.add_binary_mul(input, cos_pe, &shape);
    let x_sin = b.add_binary_mul(input, sin_pe, &shape);
    // Simplified rotation: cos_part + sin_part (bound-equivalent)
    let out = b.add_binary_add(x_cos, x_sin, &shape);
    let def = b.build(out).expect("valid RoPE extended kernel");

    let (cos_data, sin_data) = rope_cos_sin(extended_len, HIDDEN_DIM);

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_data),
        TensorParamBinding::ConstantTensor(sin_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[extended_len, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("RoPE extended (len={extended_len}) IBP: width={width:.6}");
    assert!(width.is_finite(), "RoPE extended width must be finite");
}

// ===========================================================================
// 4. Causal mask length effects IBP
// ===========================================================================

/// Causal attention at different sequence lengths should produce valid bounds.
/// Longer sequences have more masked positions proportionally.
#[test]
fn test_seq_length_causal_mask_effects_ibp() {
    for &seq_len in &[4, 8, 16] {
        let def = build_mha_kernel(seq_len, AttentionMask::Causal);
        let bindings = mha_bindings();
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = uniform_bounds(&[seq_len, HIDDEN_DIM], 1.0);
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Causal MHA seq_len={seq_len} IBP: width={width:.6}");
        assert!(
            width.is_finite(),
            "causal attention width must be finite at len={seq_len}"
        );
    }
}

// ===========================================================================
// 5. Encoder sequence scaling IBP
// ===========================================================================

/// Encoder block (Linear + RMSNorm) at different sequence lengths.
/// Bounds should remain finite regardless of sequence length.
fn test_encoder_at_length(seq_len: usize) -> f32 {
    let mut b = TensorBlockBuilder::new(&format!("dpdf_seq_encoder_{seq_len}"));
    let input = b.add_input("x", &[seq_len, HIDDEN_DIM]);
    let eps = b.add_input("eps", &[1]);
    let norm_w = b.add_input("norm_w", &[HIDDEN_DIM]);
    let proj_w = b.add_input("proj_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let shape = [seq_len, HIDDEN_DIM];

    let normed = b.add_rms_norm(input, eps, 1, norm_w, &shape);
    let out = b.add_linear(normed, proj_w, None, &shape);
    let def = b.build(out).expect("valid encoder kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        scalar_binding(1e-5),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)),
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[seq_len, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Encoder seq_len={seq_len} IBP: width={width:.6}");
    assert!(width.is_finite(), "encoder width must be finite");
    width
}

#[test]
fn test_seq_length_encoder_scaling_ibp() {
    let w4 = test_encoder_at_length(4);
    let w8 = test_encoder_at_length(8);
    let w16 = test_encoder_at_length(16);
    // All must be finite (already asserted inside helper)
    eprintln!("Encoder scaling: len=4 w={w4:.6}, len=8 w={w8:.6}, len=16 w={w16:.6}");
}

// ===========================================================================
// 6. Decoder generation length IBP
// ===========================================================================

/// Stacked SwiGLU FFN at growing generation lengths simulates autoregressive
/// decoding where the sequence grows token-by-token.
#[test]
fn test_seq_length_decoder_generation_ibp() {
    for &seq_len in &[2, 4, 8] {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_seq_decoder_gen_{seq_len}"));
        let input = b.add_input("x", &[seq_len, HIDDEN_DIM]);

        let h = build_swiglu(&mut b, input, "ffn0", seq_len, HIDDEN_DIM, FFN_DIM);
        let out = build_swiglu(&mut b, h, "ffn1", seq_len, HIDDEN_DIM, FFN_DIM);
        let def = b.build(out).expect("valid decoder gen kernel");

        let mut bindings = vec![TensorParamBinding::Variable];
        push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);
        push_swiglu_bindings(&mut bindings, HIDDEN_DIM, FFN_DIM);

        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
        let input = uniform_bounds(&[seq_len, HIDDEN_DIM], 1.0);
        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Decoder gen seq_len={seq_len} IBP: width={width:.6}");
        assert!(width.is_finite(), "decoder gen width must be finite");
    }
}

// ===========================================================================
// 7. KV-cache growing context IBP
// ===========================================================================

/// Simulate KV-cache growing context by building attention with increasing
/// KV sequence length while Q length stays fixed (autoregressive generation).
/// Uses cross-attention pattern: Q=[1, D] attending to KV=[ctx, D].
#[test]
fn test_seq_length_kv_cache_growing_context_ibp() {
    for &ctx_len in &[4, 8, 16] {
        let q_len = 1; // Single new token query

        let mut b = TensorBlockBuilder::new(&format!("dpdf_seq_kv_cache_{ctx_len}"));
        let q_input = b.add_input("q", &[q_len, HIDDEN_DIM]);
        let kv_input = b.add_input("kv", &[ctx_len, HIDDEN_DIM]);

        let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

        let out = b
            .add_multi_head_cross_attention(
                q_input,
                kv_input,
                q_w,
                k_w,
                v_w,
                o_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &[q_len, HIDDEN_DIM],
            )
            .expect("valid cross attention");
        let def = b.build(out).expect("valid KV-cache kernel");

        let bindings = vec![
            TensorParamBinding::Variable, // q
            TensorParamBinding::Variable, // kv (also variable -- separate input)
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

        // Combined input: Q tokens + KV tokens as a single flattened BoundedTensor
        let total_seq = q_len + ctx_len;
        let input = uniform_bounds(&[total_seq, HIDDEN_DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("KV-cache ctx={ctx_len} IBP: width={width:.6}");
        assert!(
            width.is_finite(),
            "KV-cache width must be finite at ctx={ctx_len}"
        );
    }
}

// ===========================================================================
// 8. Sequence truncation bounds IBP
// ===========================================================================

/// Truncated sequence (shorter) should produce valid bounds.
/// A linear projection at length=4 vs length=8 with same weights.
#[test]
fn test_seq_length_truncation_ibp() {
    let full_len = 8;
    let trunc_len = 4;

    let build_linear_kernel = |len: usize| -> TensorKernelDef {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_seq_trunc_{len}"));
        let input = b.add_input("x", &[len, HIDDEN_DIM]);
        let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let out = b.add_linear(input, w, None, &[len, HIDDEN_DIM]);
        b.build(out).expect("valid linear kernel")
    };

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];

    // Full sequence
    let full_def = build_linear_kernel(full_len);
    let full_graph = tensor_kernel_to_graph(&full_def, &bindings).expect("full graph");
    let full_input = uniform_bounds(&[full_len, HIDDEN_DIM], 1.0);
    let full_output = full_graph.propagate_ibp(&full_input).expect("full IBP");
    assert_bounds_valid(&full_output);

    // Truncated sequence
    let trunc_def = build_linear_kernel(trunc_len);
    let trunc_graph = tensor_kernel_to_graph(&trunc_def, &bindings).expect("trunc graph");
    let trunc_input = uniform_bounds(&[trunc_len, HIDDEN_DIM], 1.0);
    let trunc_output = trunc_graph.propagate_ibp(&trunc_input).expect("trunc IBP");
    assert_bounds_valid(&trunc_output);

    let full_width = bound_width(&full_output);
    let trunc_width = bound_width(&trunc_output);
    eprintln!(
        "Truncation IBP: full(len={full_len})={full_width:.6}, trunc(len={trunc_len})={trunc_width:.6}"
    );
    // Both must be finite
    assert!(full_width.is_finite(), "full width must be finite");
    assert!(trunc_width.is_finite(), "trunc width must be finite");
}

// ===========================================================================
// 9. Padding effects on bounds IBP
// ===========================================================================

/// Zero-padded input (simulating batch padding) vs unpadded.
/// Padding zeros widen the effective input range, which may widen bounds.
#[test]
fn test_seq_length_padding_effects_ibp() {
    let real_len = 4;
    let padded_len = 8;

    let mut b = TensorBlockBuilder::new("dpdf_seq_padding");
    let input = b.add_input("x", &[padded_len, HIDDEN_DIM]);
    let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let out = b.add_linear(input, w, None, &[padded_len, HIDDEN_DIM]);
    let def = b.build(out).expect("valid padded kernel");

    let bindings = vec![
        TensorParamBinding::Variable,
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    // Build padded bounds: real tokens in [-1, 1], padding in [0, 0]
    let n = padded_len * HIDDEN_DIM;
    let real_n = real_len * HIDDEN_DIM;
    let mut lower = vec![0.0f32; n];
    let mut upper = vec![0.0f32; n];
    for i in 0..real_n {
        lower[i] = -1.0;
        upper[i] = 1.0;
    }
    // Padding positions stay at [0, 0]
    let padded_input = BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[padded_len, HIDDEN_DIM]), lower).expect("lower"),
        ArrayD::from_shape_vec(IxDyn(&[padded_len, HIDDEN_DIM]), upper).expect("upper"),
    )
    .expect("valid padded bounds");

    let output = graph.propagate_ibp(&padded_input).expect("IBP");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "Padding (real={real_len}, padded={padded_len}) IBP: bounds=[{lo_min:.6}, {hi_max:.6}]"
    );
    assert!(lo_min.is_finite(), "padded lower must be finite");
    assert!(hi_max.is_finite(), "padded upper must be finite");
}

// ===========================================================================
// 10. CROWN tightness at different lengths
// ===========================================================================

fn test_crown_at_length(seq_len: usize) {
    let def = build_mha_kernel(seq_len, AttentionMask::Standard);
    let bindings = mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[seq_len, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("CROWN MHA seq_len={seq_len}: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

#[test]
fn test_seq_length_crown_tightness_len4() {
    test_crown_at_length(4);
}

#[test]
fn test_seq_length_crown_tightness_len8() {
    test_crown_at_length(8);
}

// ===========================================================================
// 11. Monotone tightening across lengths IBP
// ===========================================================================

/// Smaller input range should produce tighter output bounds at any length.
#[test]
fn test_seq_length_monotone_tightening_ibp() {
    let seq_len = 8;
    let def = build_mha_kernel(seq_len, AttentionMask::Standard);
    let bindings = mha_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

    let wide_input = uniform_bounds(&[seq_len, HIDDEN_DIM], 1.0);
    let wide_output = graph.propagate_ibp(&wide_input).expect("wide IBP");
    assert_bounds_valid(&wide_output);
    let wide_width = bound_width(&wide_output);

    let tight_input = uniform_bounds(&[seq_len, HIDDEN_DIM], 0.1);
    let tight_output = graph.propagate_ibp(&tight_input).expect("tight IBP");
    assert_bounds_valid(&tight_output);
    let tight_width = bound_width(&tight_output);

    eprintln!(
        "Monotone tightening (len={seq_len}): eps=1.0 w={wide_width:.6}, eps=0.1 w={tight_width:.6}"
    );
    assert!(
        tight_width <= wide_width + 1e-6,
        "tighter input should produce tighter output: wide={wide_width}, tight={tight_width}"
    );
}

// ===========================================================================
// 12. Full transformer block at multiple lengths IBP + CROWN
// ===========================================================================

fn build_transformer_block_kernel(seq_len: usize) -> TensorKernelDef {
    use nn_dsl::{TransformerBlockConfig, TransformerBlockWeights};

    let mut b = TensorBlockBuilder::new(&format!("dpdf_seq_transformer_{seq_len}"));
    let input = b.add_input("x", &[seq_len, HIDDEN_DIM]);

    let ln1_w = b.add_input("ln1_w", &[HIDDEN_DIM]);
    let ln1_b = b.add_input("ln1_b", &[HIDDEN_DIM]);
    let ln2_w = b.add_input("ln2_w", &[HIDDEN_DIM]);
    let ln2_b = b.add_input("ln2_b", &[HIDDEN_DIM]);
    let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);
    let ffn1_w = b.add_input("ffn1_w", &[FFN_DIM, HIDDEN_DIM]);
    let ffn2_w = b.add_input("ffn2_w", &[HIDDEN_DIM, FFN_DIM]);
    let eps = b.add_input("eps", &[1]);

    let config = TransformerBlockConfig {
        num_heads: NUM_HEADS,
        mask: AttentionMask::Standard,
        ffn_hidden_dim: FFN_DIM,
    };
    let weights = TransformerBlockWeights {
        ln1_weight: ln1_w,
        ln1_bias: ln1_b,
        ln2_weight: ln2_w,
        ln2_bias: ln2_b,
        q_weight: q_w,
        k_weight: k_w,
        v_weight: v_w,
        out_weight: o_w,
        ffn1_weight: ffn1_w,
        ffn2_weight: ffn2_w,
        eps,
    };

    let out = b
        .add_transformer_block(input, &weights, &config)
        .expect("valid transformer");
    b.build(out).expect("valid transformer kernel")
}

fn transformer_block_bindings() -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)), // ln1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)), // ln1_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 1.0f32)), // ln2_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[HIDDEN_DIM]), 0.0f32)), // ln2_b
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),                                           // q_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),                                           // k_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),                                           // v_w
        weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),                                           // o_w
        weight_binding(&[FFN_DIM, HIDDEN_DIM]), // ffn1_w
        weight_binding(&[HIDDEN_DIM, FFN_DIM]), // ffn2_w
        scalar_binding(1e-5),                   // eps
    ]
}

#[test]
fn test_seq_length_transformer_block_len4_ibp() {
    let def = build_transformer_block_kernel(4);
    let bindings = transformer_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[4, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("Transformer block len=4 IBP: width={width:.6}");
    assert!(width.is_finite(), "transformer width must be finite");
}

#[test]
fn test_seq_length_transformer_block_len8_ibp() {
    let def = build_transformer_block_kernel(8);
    let bindings = transformer_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[8, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);
    let width = bound_width(&output);
    eprintln!("Transformer block len=8 IBP: width={width:.6}");
    assert!(width.is_finite(), "transformer width must be finite");
}

#[test]
fn test_seq_length_transformer_block_crown() {
    let def = build_transformer_block_kernel(4);
    let bindings = transformer_block_bindings();
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[4, HIDDEN_DIM], 0.5);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("Transformer block CROWN: method={method:?}, width={width:.6}");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

// ===========================================================================
// 13. RoPE frequency decay at long positions IBP
// ===========================================================================

/// High-frequency RoPE dimensions decay faster at long positions.
/// Verify that bounds remain finite even when cos/sin cycle rapidly.
#[test]
fn test_seq_length_rope_frequency_decay_ibp() {
    let long_len = 32;

    let mut b = TensorBlockBuilder::new("dpdf_seq_rope_freq_decay");
    let input = b.add_input("x", &[long_len, HIDDEN_DIM]);
    let cos_pe = b.add_input("cos_pe", &[long_len, HIDDEN_DIM]);
    let sin_pe = b.add_input("sin_pe", &[long_len, HIDDEN_DIM]);
    let shape = [long_len, HIDDEN_DIM];

    let x_cos = b.add_binary_mul(input, cos_pe, &shape);
    let x_sin = b.add_binary_mul(input, sin_pe, &shape);
    let out = b.add_binary_add(x_cos, x_sin, &shape);
    let def = b.build(out).expect("valid freq decay kernel");

    let (cos_data, sin_data) = rope_cos_sin(long_len, HIDDEN_DIM);
    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(cos_data),
        TensorParamBinding::ConstantTensor(sin_data),
    ];
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");
    let input = uniform_bounds(&[long_len, HIDDEN_DIM], 1.0);
    let output = graph.propagate_ibp(&input).expect("IBP");
    assert_bounds_valid(&output);

    let width = bound_width(&output);
    eprintln!("RoPE freq decay (len={long_len}) IBP: width={width:.6}");
    assert!(width.is_finite(), "RoPE freq decay width must be finite");
}

// ===========================================================================
// 14. Cross-attention variable-length IBP
// ===========================================================================

/// Cross-attention with different Q and KV sequence lengths.
/// Models encoder-decoder architectures where encoder and decoder have
/// different token counts (e.g., multi-page document encoder attending
/// to a shorter summary decoder).
#[test]
fn test_seq_length_cross_attention_variable_ibp() {
    for &(q_len, kv_len) in &[(4, 8), (8, 4), (4, 16)] {
        let mut b = TensorBlockBuilder::new(&format!("dpdf_seq_crossattn_{q_len}_{kv_len}"));
        let q_input = b.add_input("q", &[q_len, HIDDEN_DIM]);
        let kv_input = b.add_input("kv", &[kv_len, HIDDEN_DIM]);

        let q_w = b.add_input("q_w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let k_w = b.add_input("k_w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let v_w = b.add_input("v_w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let o_w = b.add_input("o_w", &[HIDDEN_DIM, HIDDEN_DIM]);

        let out = b
            .add_multi_head_cross_attention(
                q_input,
                kv_input,
                q_w,
                k_w,
                v_w,
                o_w,
                NUM_HEADS,
                AttentionMask::Standard,
                &[q_len, HIDDEN_DIM],
            )
            .expect("valid cross attention");
        let def = b.build(out).expect("valid cross-attn kernel");

        let bindings = vec![
            TensorParamBinding::Variable,
            TensorParamBinding::Variable,
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
        ];
        let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph");

        // Combined input: Q tokens + KV tokens as a single flattened BoundedTensor
        let total_seq = q_len + kv_len;
        let input = uniform_bounds(&[total_seq, HIDDEN_DIM], 1.0);

        let output = graph.propagate_ibp(&input).expect("IBP");
        assert_bounds_valid(&output);

        let width = bound_width(&output);
        eprintln!("Cross-attn (q={q_len}, kv={kv_len}) IBP: width={width:.6}");
        assert!(
            width.is_finite(),
            "cross-attn width must be finite (q={q_len}, kv={kv_len})"
        );
    }
}

// ===========================================================================
// 15. Bound width scaling with sequence length IBP
// ===========================================================================

/// Empirically measure how bound width scales with sequence length.
/// For a linear projection, width should be independent of sequence length
/// (each position is processed independently). For attention, width may grow
/// due to softmax over more positions.
#[test]
fn test_seq_length_bound_width_scaling_ibp() {
    let lengths = [4, 8, 16];
    let mut linear_widths = Vec::new();
    let mut attn_widths = Vec::new();

    for &seq_len in &lengths {
        // Linear projection (length-independent)
        let mut b = TensorBlockBuilder::new(&format!("dpdf_seq_scale_linear_{seq_len}"));
        let input = b.add_input("x", &[seq_len, HIDDEN_DIM]);
        let w = b.add_input("w", &[HIDDEN_DIM, HIDDEN_DIM]);
        let out = b.add_linear(input, w, None, &[seq_len, HIDDEN_DIM]);
        let lin_def = b.build(out).expect("valid linear");

        let lin_bindings = vec![
            TensorParamBinding::Variable,
            weight_binding(&[HIDDEN_DIM, HIDDEN_DIM]),
        ];
        let lin_graph = tensor_kernel_to_graph(&lin_def, &lin_bindings).expect("graph");
        let lin_input = uniform_bounds(&[seq_len, HIDDEN_DIM], 1.0);
        let lin_output = lin_graph.propagate_ibp(&lin_input).expect("IBP");
        assert_bounds_valid(&lin_output);
        linear_widths.push(bound_width(&lin_output));

        // MHA (length-dependent via softmax)
        let attn_def = build_mha_kernel(seq_len, AttentionMask::Standard);
        let attn_bindings = mha_bindings();
        let attn_graph = tensor_kernel_to_graph(&attn_def, &attn_bindings).expect("graph");
        let attn_input = uniform_bounds(&[seq_len, HIDDEN_DIM], 1.0);
        let attn_output = attn_graph.propagate_ibp(&attn_input).expect("IBP");
        assert_bounds_valid(&attn_output);
        attn_widths.push(bound_width(&attn_output));
    }

    eprintln!("Bound width scaling:");
    for (i, &len) in lengths.iter().enumerate() {
        eprintln!(
            "  len={len}: linear_width={:.6}, attn_width={:.6}",
            linear_widths[i], attn_widths[i]
        );
    }

    // Linear widths should be approximately equal across lengths
    for w in &linear_widths {
        assert!(w.is_finite(), "linear width must be finite");
    }
    // Attention widths should all be finite
    for w in &attn_widths {
        assert!(w.is_finite(), "attention width must be finite");
    }
    // Linear width should be roughly constant (tolerance for numerical noise)
    if linear_widths.len() >= 2 {
        let max_lin = linear_widths
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let min_lin = linear_widths.iter().copied().fold(f32::INFINITY, f32::min);
        let ratio = if min_lin > 0.0 {
            max_lin / min_lin
        } else {
            1.0
        };
        eprintln!("Linear width ratio (max/min): {ratio:.4}");
        assert!(
            ratio < 1.1,
            "linear projection width should be length-independent, ratio={ratio}"
        );
    }
}
