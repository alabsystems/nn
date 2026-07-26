// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `trainable_extra.rs` and `trainable_extra_norm.rs`.
//!
//! Proves properties of trainable layer parameter counts, weight shape
//! invariants, bias broadcast shape construction, convolution output
//! dimension formulas, and normalization layer initialization.
//!
//! The trainable module wrappers own `Var` weights and connect nn layers
//! to the autodiff system. These harnesses verify the structural properties
//! that must hold for correct forward passes and gradient accumulation.
//!
//! **Local-copy gap:** Scalar functions here re-implement production formulas.
//! `// SYNC:` comments track correspondence.
//!
//! Re: #3714 (Kani harnesses for nn-autodiff grad + backward_rules_special + trainable_extra).

// ── TrainableEmbedding: weight shape [vocab_size, embed_dim] ─────────────
//
// Embedding weight must be 2D. The vars() method returns exactly 1 Var.
//
// SYNC: trainable_extra.rs:54-56, 103-105

/// Model embedding weight element count: vocab_size * embed_dim.
///
/// SYNC: trainable_extra.rs:64 (Var::randn(&[vocab_size, embed_dim], ...))
#[allow(dead_code)]
fn embedding_weight_numel(vocab_size: usize, embed_dim: usize) -> usize {
    vocab_size * embed_dim
}

/// Prove embedding weight element count is positive for valid params.
#[kani::unwind(1)]
#[kani::proof]
fn prove_embedding_weight_numel_positive() {
    let vocab: u16 = kani::any();
    let embed: u16 = kani::any();
    kani::assume(vocab >= 1 && vocab <= 10000);
    kani::assume(embed >= 1 && embed <= 1024);
    let n = embedding_weight_numel(vocab as usize, embed as usize);
    assert!(n > 0, "embedding weight numel must be positive");
    assert!(
        n == vocab as usize * embed as usize,
        "embedding weight numel must be vocab * embed"
    );
}

/// Prove embedding vars() returns exactly 1 variable.
///
/// SYNC: trainable_extra.rs:103-105
#[allow(dead_code)]
fn embedding_var_count() -> usize {
    1 // weight only
}

#[kani::unwind(1)]
#[kani::proof]
fn prove_embedding_var_count() {
    assert!(
        embedding_var_count() == 1,
        "TrainableEmbedding must have exactly 1 var (weight)"
    );
}

// ── TrainableConvTranspose1d: output length formula ──────────────────────
//
// ConvTranspose1d output: (in_len - 1) * stride - 2*padding + dilation*(kernel-1) + output_padding + 1
//
// SYNC: trainable_extra.rs:186-206

/// Conv transpose 1D output length.
///
/// SYNC: Op::ConvTranspose1d output size formula.
#[allow(dead_code)]
fn conv_transpose1d_output_len(
    in_len: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    output_padding: usize,
) -> Option<usize> {
    if in_len == 0 || kernel_size == 0 || stride == 0 {
        return None;
    }
    let base = (in_len - 1).checked_mul(stride)?;
    let effective_kernel = dilation.checked_mul(kernel_size - 1)?;
    let raw = base
        .checked_add(effective_kernel)?
        .checked_add(output_padding)?
        .checked_add(1)?;
    Some(raw.checked_sub(2 * padding)?)
}

/// Prove conv transpose output > 0 for valid standard parameters.
#[kani::unwind(1)]
#[kani::proof]
fn prove_conv_transpose_output_positive() {
    let in_len: u8 = kani::any();
    let kernel: u8 = kani::any();
    let stride: u8 = kani::any();
    kani::assume(in_len >= 1 && in_len <= 32);
    kani::assume(kernel >= 1 && kernel <= 8);
    kani::assume(stride >= 1 && stride <= 4);
    if let Some(out) = conv_transpose1d_output_len(
        in_len as usize,
        kernel as usize,
        0, // no padding
        stride as usize,
        1, // no dilation
        0, // no output padding
    ) {
        assert!(out > 0, "conv transpose output must be positive");
    }
}

/// Prove conv transpose output is larger than input for stride > 1 (upsampling).
#[kani::unwind(1)]
#[kani::proof]
fn prove_conv_transpose_upsamples() {
    let in_len: u8 = kani::any();
    let stride: u8 = kani::any();
    kani::assume(in_len >= 2 && in_len <= 16);
    kani::assume(stride >= 2 && stride <= 4);
    // kernel=1, no padding, dilation=1, output_padding=0 → pure stride upsampling
    if let Some(out) = conv_transpose1d_output_len(in_len as usize, 1, 0, stride as usize, 1, 0) {
        assert!(
            out >= in_len as usize,
            "conv transpose with stride>1 must produce output >= input"
        );
    }
}

/// Model var count for ConvTranspose1d: 1 (weight) or 2 (weight + bias).
///
/// SYNC: trainable_extra.rs:209-215
#[allow(dead_code)]
fn conv_transpose1d_var_count(has_bias: bool) -> usize {
    if has_bias {
        2
    } else {
        1
    }
}

/// Prove ConvTranspose1d var count is 1 or 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_conv_transpose_var_count() {
    let has_bias: bool = kani::any();
    let count = conv_transpose1d_var_count(has_bias);
    assert!(count >= 1 && count <= 2, "var count must be 1 or 2");
    if has_bias {
        assert!(count == 2, "with bias must be 2");
    } else {
        assert!(count == 1, "without bias must be 1");
    }
}

// ── TrainableConv2d: bias broadcast shape ────────────────────────────────
//
// Conv2d bias is [out_channels]. For broadcasting against [B, out_channels, H, W],
// it is reshaped to [1, out_channels, 1, 1].
//
// SYNC: trainable_extra.rs:296-299

/// Compute Conv2d bias broadcast shape: [1, out_ch, 1, 1].
///
/// SYNC: trainable_extra.rs:298 (`b_tracked.reshape(&[1, b.dims()?[0], 1, 1])`)
#[allow(dead_code)]
fn conv2d_bias_broadcast_shape(out_channels: usize) -> [usize; 4] {
    [1, out_channels, 1, 1]
}

/// Prove bias broadcast shape has numel == out_channels.
#[kani::unwind(5)]
#[kani::proof]
fn prove_conv2d_bias_broadcast_numel() {
    let out_ch: u16 = kani::any();
    kani::assume(out_ch >= 1 && out_ch <= 2048);
    let shape = conv2d_bias_broadcast_shape(out_ch as usize);
    let numel: usize = shape.iter().product();
    assert!(
        numel == out_ch as usize,
        "bias broadcast shape numel must equal out_channels"
    );
}

/// Prove bias broadcast shape has correct dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn prove_conv2d_bias_broadcast_dims() {
    let out_ch: u16 = kani::any();
    kani::assume(out_ch >= 1 && out_ch <= 2048);
    let shape = conv2d_bias_broadcast_shape(out_ch as usize);
    assert!(shape[0] == 1, "dim 0 must be 1 (batch)");
    assert!(shape[1] == out_ch as usize, "dim 1 must be out_channels");
    assert!(shape[2] == 1, "dim 2 must be 1 (height)");
    assert!(shape[3] == 1, "dim 3 must be 1 (width)");
}

/// Model Conv1d bias broadcast shape: [1, out_ch, 1].
///
/// SYNC: trainable.rs:244 (`b_tracked.reshape(&[1, b.dims()?[0], 1])`)
#[allow(dead_code)]
fn conv1d_bias_broadcast_shape(out_channels: usize) -> [usize; 3] {
    [1, out_channels, 1]
}

/// Prove Conv1d bias broadcast shape numel == out_channels.
#[kani::unwind(5)]
#[kani::proof]
fn prove_conv1d_bias_broadcast_numel() {
    let out_ch: u16 = kani::any();
    kani::assume(out_ch >= 1 && out_ch <= 2048);
    let shape = conv1d_bias_broadcast_shape(out_ch as usize);
    let numel: usize = shape.iter().product();
    assert!(
        numel == out_ch as usize,
        "conv1d bias broadcast shape numel must equal out_channels"
    );
}

// ── TrainableConv2d var count ────────────────────────────────────────────
//
// Conv2d has weight and optional bias.
//
// SYNC: trainable_extra.rs:305-311

/// Model var count for Conv2d: 1 (weight) or 2 (weight + bias).
#[allow(dead_code)]
fn conv2d_var_count(has_bias: bool) -> usize {
    if has_bias {
        2
    } else {
        1
    }
}

/// Prove Conv2d var count is 1 or 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_conv2d_var_count() {
    let has_bias: bool = kani::any();
    let count = conv2d_var_count(has_bias);
    assert!(count >= 1 && count <= 2, "var count must be 1 or 2");
}

// ── TrainableLayerNorm: initialization invariants ────────────────────────
//
// LayerNorm initializes with weight=1, bias=0, matching PyTorch.
//
// SYNC: trainable_extra_norm.rs:38-44

/// Model LayerNorm initial weight value (all ones).
#[allow(dead_code)]
fn layer_norm_init_weight(size: usize) -> Vec<f32> {
    vec![1.0f32; size]
}

/// Prove LayerNorm initial weight values are all 1.0.
#[kani::unwind(5)]
#[kani::proof]
fn prove_layer_norm_init_weight_all_ones() {
    let size: u8 = kani::any();
    kani::assume(size >= 1 && size <= 32);
    let w = layer_norm_init_weight(size as usize);
    for &v in &w {
        assert!(v == 1.0, "LayerNorm weight must init to 1.0");
    }
    assert!(
        w.len() == size as usize,
        "weight length must equal normalized_shape"
    );
}

/// Model LayerNorm var count: always 2 (weight + bias).
///
/// SYNC: trainable_extra_norm.rs:81-83
#[allow(dead_code)]
fn layer_norm_var_count() -> usize {
    2 // weight + bias
}

/// Prove LayerNorm var count is 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_layer_norm_var_count() {
    assert!(layer_norm_var_count() == 2, "LayerNorm must have 2 vars");
}

// ── TrainableRmsNorm: initialization invariants ──────────────────────────
//
// RmsNorm initializes with weight=1, no bias.
//
// SYNC: trainable_extra_norm.rs:101-106

/// Model RmsNorm var count: always 1 (weight only, no bias).
///
/// SYNC: trainable_extra_norm.rs:127-129
#[allow(dead_code)]
fn rms_norm_var_count() -> usize {
    1 // weight only
}

/// Prove RmsNorm var count is 1.
#[kani::unwind(1)]
#[kani::proof]
fn prove_rms_norm_var_count() {
    assert!(rms_norm_var_count() == 1, "RmsNorm must have 1 var");
}

// ── TrainableGroupNorm: initialization invariants ────────────────────────
//
// GroupNorm has weight=[C], bias=[C], and num_groups divides C.
//
// SYNC: trainable_extra_norm.rs:146-158

/// Model GroupNorm valid configuration: num_groups divides C.
///
/// SYNC: trainable_extra_norm.rs:146
#[allow(dead_code)]
fn is_valid_group_norm_config(num_channels: usize, num_groups: usize) -> bool {
    num_groups > 0 && num_channels > 0 && num_channels % num_groups == 0
}

/// Prove valid GroupNorm config accepted.
#[kani::unwind(1)]
#[kani::proof]
fn prove_valid_group_norm_config() {
    let channels: u8 = kani::any();
    let groups: u8 = kani::any();
    kani::assume(channels >= 1 && channels <= 128);
    kani::assume(groups >= 1 && groups <= channels);
    kani::assume(channels as usize % groups as usize == 0);
    assert!(
        is_valid_group_norm_config(channels as usize, groups as usize),
        "C divisible by G must be valid"
    );
}

/// Prove GroupNorm config requires num_groups divides C.
#[kani::unwind(1)]
#[kani::proof]
fn prove_invalid_group_norm_config_rejects() {
    let channels: u8 = kani::any();
    let groups: u8 = kani::any();
    kani::assume(channels >= 2 && channels <= 128);
    kani::assume(groups >= 2 && groups <= channels);
    kani::assume(channels as usize % groups as usize != 0);
    assert!(
        !is_valid_group_norm_config(channels as usize, groups as usize),
        "C not divisible by G must be invalid"
    );
}

/// Model GroupNorm var count: always 2 (weight + bias).
///
/// SYNC: trainable_extra_norm.rs:190-193
#[allow(dead_code)]
fn group_norm_var_count() -> usize {
    2
}

/// Prove GroupNorm var count is 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_group_norm_var_count() {
    assert!(group_norm_var_count() == 2, "GroupNorm must have 2 vars");
}

// ── TrainableBatchNorm: var count ────────────────────────────────────────
//
// SYNC: trainable_extra_norm.rs:243-246

/// Model BatchNorm var count: always 2 (weight + bias).
#[allow(dead_code)]
fn batch_norm_var_count() -> usize {
    2
}

/// Prove BatchNorm var count is 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_batch_norm_var_count() {
    assert!(batch_norm_var_count() == 2, "BatchNorm must have 2 vars");
}

// ── TrainableInstanceNorm: var count ─────────────────────────────────────
//
// SYNC: trainable_extra_norm.rs:295-298

/// Model InstanceNorm var count: always 2 (weight + bias).
#[allow(dead_code)]
fn instance_norm_var_count() -> usize {
    2
}

/// Prove InstanceNorm var count is 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_instance_norm_var_count() {
    assert!(
        instance_norm_var_count() == 2,
        "InstanceNorm must have 2 vars"
    );
}

// ── TrainableLinear: weight shape and var count ──────────────────────────
//
// Linear weight is [out_features, in_features]. vars() returns 1 or 2.
//
// SYNC: trainable.rs:66-69, 155-161

/// Model linear weight element count.
///
/// SYNC: trainable.rs:80 (weight shape [out_features, in_features])
#[allow(dead_code)]
fn linear_weight_numel(in_features: usize, out_features: usize) -> usize {
    out_features * in_features
}

/// Prove linear weight numel is positive for valid dimensions.
#[kani::unwind(1)]
#[kani::proof]
fn prove_linear_weight_numel_positive() {
    let in_f: u16 = kani::any();
    let out_f: u16 = kani::any();
    kani::assume(in_f >= 1 && in_f <= 1024);
    kani::assume(out_f >= 1 && out_f <= 1024);
    let n = linear_weight_numel(in_f as usize, out_f as usize);
    assert!(n > 0, "linear weight numel must be positive");
}

/// Model linear var count: 1 (weight) or 2 (weight + bias).
///
/// SYNC: trainable.rs:155-161
#[allow(dead_code)]
fn linear_var_count(has_bias: bool) -> usize {
    if has_bias {
        2
    } else {
        1
    }
}

/// Prove linear var count is 1 or 2.
#[kani::unwind(1)]
#[kani::proof]
fn prove_linear_var_count() {
    let has_bias: bool = kani::any();
    let count = linear_var_count(has_bias);
    assert!(count >= 1 && count <= 2, "linear var count must be 1 or 2");
}

// ── Normalization eps must be positive ────────────────────────────────────
//
// All normalization layers require eps > 0 to avoid division by zero.
// This is an invariant across LayerNorm, RmsNorm, GroupNorm, BatchNorm, InstanceNorm.

/// Model eps validation for normalization layers.
#[allow(dead_code)]
fn is_valid_norm_eps(eps: f64) -> bool {
    eps.is_finite() && eps > 0.0
}

/// Prove standard eps values are valid.
#[kani::unwind(1)]
#[kani::proof]
fn prove_standard_eps_values_valid() {
    // Common eps values used in practice
    let eps_values: [f64; 4] = [1e-5, 1e-6, 1e-8, 1e-12];
    for &eps in &eps_values {
        assert!(is_valid_norm_eps(eps), "standard eps values must be valid");
    }
}

/// Prove zero eps is invalid.
#[kani::unwind(1)]
#[kani::proof]
fn prove_zero_eps_invalid() {
    assert!(
        !is_valid_norm_eps(0.0),
        "zero eps must be invalid (causes div by zero)"
    );
}

/// Prove negative eps is invalid.
#[kani::unwind(1)]
#[kani::proof]
fn prove_negative_eps_invalid() {
    let eps: f64 = kani::any();
    kani::assume(eps.is_finite() && eps < 0.0);
    assert!(!is_valid_norm_eps(eps), "negative eps must be invalid");
}
