// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `buffer_planner_bytes.rs` — the [`native_op_output_bytes`]
//! function and supporting helpers.
//!
//! Proves:
//! - `checked_shape_bytes` overflow protection: large dims produce 0 (not panic/overflow).
//! - BiLstmCat output bytes = seq_len * batch * 2 * hidden_size * 4.
//! - LstmSequence output bytes consistency with shape.
//! - FlashAttention output bytes = product of output_shape * 4.
//! - NormActivConv1d output bytes respects padding/dilation formula.
//! - MaxPool1d output bytes respects kernel/stride formula.
//! - LinearActivation output bytes = batch * out_features * 4.
//! - Conv1dGemm output bytes formula correctness.
//! - ConstantWeight always returns 0 bytes.
//! - FusedResBlock output bytes matches phase1.input_shape.
//! - step_output_bytes_typed scales correctly from F32 to F16 (half the bytes).

use super::bytes::native_op_output_bytes;
use super::bytes::step_output_bytes_typed;
use crate::ir::ScalarType;
use crate::trace_compile::{CompiledStep, NativeOpKind};
use crate::trace_compile::{GemmActivation, NormActivConv1dParams, NormActivation};
use std::collections::HashMap;

// ============================================================================
// 1. checked_shape_bytes overflow protection (tested indirectly via FlashAttention)
// ============================================================================

/// Proves: large dimensions produce 0 bytes (overflow protection), not panic.
///
/// SUBSTANTIVE: `checked_shape_bytes` uses `checked_mul` to catch overflow.
/// This harness verifies that product overflow returns 0 via an op that
/// delegates to `checked_shape_bytes` (FlashAttention uses it on output_shape).
#[kani::proof]
#[kani::unwind(8)]
fn proof_checked_shape_bytes_overflow_returns_zero() {
    // Use dimensions large enough to overflow usize when multiplied together.
    // usize::MAX / 4 will overflow when multiplied by F32_BYTES (4).
    let big = usize::MAX / 2;
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![big, 2, 1, 1],
        k_shape: vec![1, 1, 1, 1],
        output_shape: vec![big, 2, 1, 1],
        input_layout: crate::trace_compile::AttentionLayout::HeadsFirst,
    };
    let bytes = native_op_output_bytes(&op);
    // big * 2 overflows usize, so checked_mul returns None → 0
    assert_eq!(bytes, 0, "overflow must produce 0, not panic");
}

/// Proves: zero-dimensional shape produces F32_BYTES (product of empty = 1).
///
/// SUBSTANTIVE: An empty output_shape has product 1, so bytes = 1 * 4 = 4.
/// Verifies the fold identity element is correct.
#[kani::proof]
#[kani::unwind(8)]
fn proof_checked_shape_bytes_empty_shape() {
    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![],
        k_shape: vec![],
        output_shape: vec![],
        input_layout: crate::trace_compile::AttentionLayout::HeadsFirst,
    };
    let bytes = native_op_output_bytes(&op);
    // Empty shape → product = 1 → 1 * 4 = 4
    assert_eq!(bytes, 4);
}

// ============================================================================
// 2. BiLstmCat output bytes = seq_len * batch * 2 * hidden_size * 4
// ============================================================================

/// Proves: BiLstmCat output bytes follow the formula
/// `seq_len * batch * 2 * hidden_size * F32_BYTES`.
///
/// SUBSTANTIVE: BiLstmCat concatenates forward and reverse LSTM outputs
/// along the hidden dimension. Output shape is [seq_len, batch, 2*hidden].
/// Buffer planner must allocate exactly this many bytes.
#[kani::proof]
#[kani::unwind(8)]
fn proof_bilstm_cat_output_bytes_consistent() {
    let hidden: usize = kani::any();
    kani::assume(hidden > 0 && hidden <= 512);
    let seq_len: usize = kani::any();
    kani::assume(seq_len > 0 && seq_len <= 64);
    let batch: usize = kani::any();
    kani::assume(batch > 0 && batch <= 8);

    let op = NativeOpKind::BiLstmCat {
        hidden_size: hidden,
        input_shape: vec![seq_len, batch, hidden * 2],
        h_shape: vec![batch, hidden],
        fwd_lstm_step: 0,
        rev_lstm_step: 1,
    };
    let bytes = native_op_output_bytes(&op);
    // Should be seq_len * batch * 2 * hidden * 4
    let expected = seq_len
        .checked_mul(batch)
        .and_then(|v| v.checked_mul(2 * hidden))
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(0);
    assert_eq!(bytes, expected);
}

// ============================================================================
// 3. LstmSequence output bytes consistency with shape
// ============================================================================

/// Proves: LstmSequence output bytes = seq_len * batch * hidden_size * 4.
///
/// SUBSTANTIVE: LSTM output shape is [seq_len, batch, hidden_size].
/// The buffer planner must allocate exactly this many bytes.
#[kani::proof]
#[kani::unwind(8)]
fn proof_lstm_sequence_output_bytes() {
    let hidden: usize = kani::any();
    kani::assume(hidden > 0 && hidden <= 512);
    let seq_len: usize = kani::any();
    kani::assume(seq_len > 0 && seq_len <= 64);
    let batch: usize = kani::any();
    kani::assume(batch > 0 && batch <= 8);
    let input_size: usize = kani::any();
    kani::assume(input_size > 0 && input_size <= 256);

    let op = NativeOpKind::LstmSequence {
        hidden_size: hidden,
        input_shape: vec![seq_len, batch, input_size],
        h_shape: vec![batch, hidden],
        reverse: false,
    };
    let bytes = native_op_output_bytes(&op);
    let expected = seq_len
        .checked_mul(batch)
        .and_then(|v| v.checked_mul(hidden))
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(0);
    assert_eq!(bytes, expected);
}

// ============================================================================
// 4. FlashAttention output bytes = product of output_shape * 4
// ============================================================================

/// Proves: FlashAttention output bytes = B * H * S * D * 4.
///
/// SUBSTANTIVE: FlashAttention output is same shape as Q: [B, H_q, S_q, D].
/// The buffer planner delegates to checked_shape_bytes(output_shape).
#[kani::proof]
#[kani::unwind(8)]
fn proof_flash_attention_output_bytes() {
    let b: usize = kani::any();
    kani::assume(b > 0 && b <= 4);
    let h: usize = kani::any();
    kani::assume(h > 0 && h <= 8);
    let s: usize = kani::any();
    kani::assume(s > 0 && s <= 64);
    let d: usize = kani::any();
    kani::assume(d > 0 && d <= 128);

    let op = NativeOpKind::FlashAttention {
        scale: 0.125,
        causal: false,
        q_shape: vec![b, h, s, d],
        k_shape: vec![b, h, s, d],
        output_shape: vec![b, h, s, d],
        input_layout: crate::trace_compile::AttentionLayout::HeadsFirst,
    };
    let bytes = native_op_output_bytes(&op);
    let expected = b
        .checked_mul(h)
        .and_then(|v| v.checked_mul(s))
        .and_then(|v| v.checked_mul(d))
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(0);
    assert_eq!(bytes, expected);
}

// ============================================================================
// 5. NormActivConv1d output bytes respects padding/dilation formula
// ============================================================================

/// Proves: NormActivConv1d output bytes = B * C_out * T_out * 4
/// where T_out = T_in + 2*padding - dilation*(K-1).
///
/// SUBSTANTIVE: The NormActivConv1d stride is always 1, so the output
/// temporal dimension follows the formula above. Buffer planner must
/// allocate exactly B * C_out * T_out * F32_BYTES.
#[kani::proof]
#[kani::unwind(8)]
fn proof_norm_activ_conv1d_output_bytes() {
    let batch: usize = kani::any();
    kani::assume(batch > 0 && batch <= 4);
    let c_in: usize = kani::any();
    kani::assume(c_in > 0 && c_in <= 64);
    let c_out: usize = kani::any();
    kani::assume(c_out > 0 && c_out <= 64);
    let t_in: usize = kani::any();
    kani::assume(t_in >= 8 && t_in <= 128);
    let kernel_size: usize = kani::any();
    kani::assume(kernel_size >= 1 && kernel_size <= 7);
    let dilation: usize = kani::any();
    kani::assume(dilation >= 1 && dilation <= 3);
    let padding: usize = kani::any();
    kani::assume(padding <= 16);

    // Ensure T_out > 0: t_in + 2*padding >= dilation*(kernel_size-1)
    let kernel_span = dilation * (kernel_size - 1);
    let padded = t_in + 2 * padding;
    kani::assume(padded >= kernel_span);

    let t_out = padded - kernel_span;

    let op = NativeOpKind::NormActivConv1d {
        activation: NormActivation::Snake,
        eps: 1e-5,
        conv_dilation: dilation,
        conv_padding: padding,
        input_shape: vec![batch, c_in, t_in],
        output_channels: c_out,
        kernel_size,
        external_node_ids: None,
    };
    let bytes = native_op_output_bytes(&op);
    let expected = batch
        .checked_mul(c_out)
        .and_then(|v| v.checked_mul(t_out))
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(0);
    assert_eq!(bytes, expected);
}

// ============================================================================
// 6. MaxPool1d output bytes respects kernel/stride formula
// ============================================================================

/// Proves: MaxPool1d output bytes = B * C * floor((L + 2*P - K) / S + 1) * 4.
///
/// SUBSTANTIVE: MaxPool1d output length follows the standard pooling formula.
/// Buffer planner must allocate exactly the right amount.
#[kani::proof]
#[kani::unwind(8)]
fn proof_max_pool1d_output_bytes() {
    let batch: usize = kani::any();
    kani::assume(batch > 0 && batch <= 4);
    let channels: usize = kani::any();
    kani::assume(channels > 0 && channels <= 64);
    let length: usize = kani::any();
    kani::assume(length >= 4 && length <= 128);
    let kernel_size: usize = kani::any();
    kani::assume(kernel_size >= 1 && kernel_size <= 8);
    let stride: usize = kani::any();
    kani::assume(stride >= 1 && stride <= 4);
    let padding: usize = kani::any();
    kani::assume(padding <= 4);

    let padded = length.saturating_add(2 * padding);
    kani::assume(padded >= kernel_size);
    let out_len = (padded - kernel_size) / stride + 1;

    let op = NativeOpKind::MaxPool1d {
        kernel_size,
        stride,
        padding,
        input_shape: vec![batch, channels, length],
    };
    let bytes = native_op_output_bytes(&op);
    let expected = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(out_len))
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(0);
    assert_eq!(bytes, expected);
}

// ============================================================================
// 7. LinearActivation output bytes = batch * out_features * 4
// ============================================================================

/// Proves: LinearActivation output bytes = product(input_shape[:-1]) * out_features * 4.
///
/// SUBSTANTIVE: LinearActivation replaces the last dimension with out_features.
/// The batch dimensions are the product of all but the last dim.
#[kani::proof]
#[kani::unwind(8)]
fn proof_linear_activation_output_bytes() {
    let batch: usize = kani::any();
    kani::assume(batch > 0 && batch <= 8);
    let in_features: usize = kani::any();
    kani::assume(in_features > 0 && in_features <= 256);
    let out_features: usize = kani::any();
    kani::assume(out_features > 0 && out_features <= 256);

    let op = NativeOpKind::LinearActivation {
        activation: GemmActivation::Relu,
        in_features,
        out_features,
        has_bias: true,
        input_shape: vec![batch, in_features],
    };
    let bytes = native_op_output_bytes(&op);
    let expected = batch
        .checked_mul(out_features)
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(0);
    assert_eq!(bytes, expected);
}

// ============================================================================
// 8. Conv1dGemm output bytes formula correctness
// ============================================================================

/// Proves: Conv1dGemm output bytes = B * out_channels * L_out * 4
/// where L_out = (L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1.
///
/// SUBSTANTIVE: Conv1dGemm uses the standard convolution output length formula.
/// Buffer planner must allocate the exact amount for the im2col + GEMM output.
#[kani::proof]
#[kani::unwind(8)]
fn proof_conv1d_gemm_output_bytes() {
    let batch: usize = kani::any();
    kani::assume(batch > 0 && batch <= 4);
    let c_in: usize = kani::any();
    kani::assume(c_in > 0 && c_in <= 64);
    let c_out: usize = kani::any();
    kani::assume(c_out > 0 && c_out <= 64);
    let l_in: usize = kani::any();
    kani::assume(l_in >= 8 && l_in <= 128);
    let kernel_size: usize = kani::any();
    kani::assume(kernel_size >= 1 && kernel_size <= 7);
    let stride: usize = kani::any();
    kani::assume(stride >= 1 && stride <= 4);
    let padding: usize = kani::any();
    kani::assume(padding <= 8);
    let dilation: usize = kani::any();
    kani::assume(dilation >= 1 && dilation <= 3);

    let effective_k = dilation * (kernel_size - 1) + 1;
    let padded_in = l_in + 2 * padding;
    kani::assume(padded_in >= effective_k);
    let l_out = (padded_in - effective_k) / stride + 1;

    let op = NativeOpKind::Conv1dGemm {
        input_shape: vec![batch, c_in, l_in],
        out_channels: c_out,
        kernel_size,
        stride,
        padding,
        dilation,
        groups: 1,
        has_bias: true,
    };
    let bytes = native_op_output_bytes(&op);
    let expected = batch
        .checked_mul(c_out)
        .and_then(|v| v.checked_mul(l_out))
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(0);
    assert_eq!(bytes, expected);
}

// ============================================================================
// 9. ConstantWeight always returns 0 bytes
// ============================================================================

/// Proves: ConstantWeight always returns 0 bytes regardless of shape.
///
/// SUBSTANTIVE: ConstantWeight aliases a pre-uploaded buffer — the buffer
/// planner must NOT allocate any new memory for it.
#[kani::proof]
#[kani::unwind(8)]
fn proof_constant_weight_always_zero_bytes() {
    let d0: usize = kani::any();
    kani::assume(d0 <= 1024);
    let d1: usize = kani::any();
    kani::assume(d1 <= 1024);

    let op = NativeOpKind::ConstantWeight {
        name: "test_weight".into(),
        shape: vec![d0, d1],
    };
    let bytes = native_op_output_bytes(&op);
    assert_eq!(bytes, 0, "ConstantWeight must always return 0 bytes");
}

// ============================================================================
// 10. FusedResBlock output bytes matches phase1.input_shape
// ============================================================================

/// Proves: FusedResBlock output bytes = checked_shape_bytes(phase1.input_shape).
///
/// SUBSTANTIVE: FusedResBlock preserves the block input shape (residual add
/// produces same shape as x). The buffer planner uses phase1.input_shape
/// to determine output size.
#[kani::proof]
#[kani::unwind(8)]
fn proof_fused_resblock_output_bytes_matches_phase1_input() {
    let batch: usize = kani::any();
    kani::assume(batch > 0 && batch <= 4);
    let channels: usize = kani::any();
    kani::assume(channels > 0 && channels <= 64);
    let t_len: usize = kani::any();
    kani::assume(t_len > 0 && t_len <= 128);

    let shape = vec![batch, channels, t_len];

    let params1 = NormActivConv1dParams::new(
        NormActivation::Snake,
        1e-5,
        1,
        1,
        shape.clone(),
        channels,
        3,
    );
    let params2 = NormActivConv1dParams::new(
        NormActivation::Snake,
        1e-5,
        1,
        1,
        // Phase2 may have a different T due to padding/dilation, but
        // the output uses phase1.input_shape (the block input).
        vec![batch, channels, t_len],
        channels,
        3,
    );

    let op = NativeOpKind::FusedResBlock {
        phase1: params1,
        phase2: params2,
        input_steps: vec![0, 1, 2, 3, 4],
        residual_scale: 1.0,
        style_proj: None,
        shortcut_step: None,
        pool_step: None,
        style_batch_offset: None,
    };
    let bytes = native_op_output_bytes(&op);
    // Expected: product(phase1.input_shape) * 4
    let expected = batch
        .checked_mul(channels)
        .and_then(|v| v.checked_mul(t_len))
        .and_then(|v| v.checked_mul(4))
        .unwrap_or(0);
    assert_eq!(bytes, expected);
}

// ============================================================================
// 11. step_output_bytes_typed scales correctly from F32 to F16
// ============================================================================

/// Proves: step_output_bytes_typed with F16 dtype produces half the bytes
/// compared to the default F32 path for NativeOp steps.
///
/// SUBSTANTIVE: Mixed-precision executors cast NativeOp outputs to F16.
/// The buffer planner must allocate half the bytes when dtype is F16.
/// Formula: f32_bytes / 4 * 2 = f32_bytes / 2.
#[kani::proof]
#[kani::unwind(8)]
fn proof_step_output_bytes_typed_f16_half_of_f32() {
    let batch: usize = kani::any();
    kani::assume(batch > 0 && batch <= 8);
    let out_features: usize = kani::any();
    kani::assume(out_features > 0 && out_features <= 256);

    let op = NativeOpKind::LinearActivation {
        activation: GemmActivation::Silu,
        in_features: 64,
        out_features,
        has_bias: false,
        input_shape: vec![batch, 64],
    };

    let step = CompiledStep::NativeOp {
        op: op.clone(),
        weight_data: HashMap::new(),
    };

    let f32_bytes = step_output_bytes_typed(&step, Some(ScalarType::F32));
    let f16_bytes = step_output_bytes_typed(&step, Some(ScalarType::F16));
    let default_bytes = step_output_bytes_typed(&step, None);

    // F32 typed == default (None defaults to F32)
    assert_eq!(f32_bytes, default_bytes, "None must default to F32");
    // F16 must be exactly half of F32
    assert_eq!(
        f16_bytes * 2,
        f32_bytes,
        "F16 bytes must be half of F32 bytes"
    );
    // Verify absolute value
    let expected_f32 = batch * out_features * 4;
    assert_eq!(f32_bytes, expected_f32);
    let expected_f16 = batch * out_features * 2;
    assert_eq!(f16_bytes, expected_f16);
}
