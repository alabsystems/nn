// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Builder helpers for Kokoro F0EnergyPredictor NY composition.
//!
//! Builds a simplified F0EnergyPredictor graph for disentanglement verification.
//! The production F0EnergyPredictor (`crates/nn-models/src/kokoro_f0.rs`) has:
//!
//!   aligned [B, d_model, T] + style [B, style_dim]
//!   → BiLSTM → [B, 2*H, T]
//!   → F0 head:     3 × AdainResBlk1d (block 1 upsamples 2x) → Linear → [B, 1, 2T]
//!   → Energy head:  3 × AdainResBlk1d (block 1 upsamples 2x) → Linear → [B, 1, 2T]
//!
//! Simplifications for NY tractability:
//! - BiLSTM replaced with shared Conv1d (avoids sequence unrolling)
//! - 1 AdainResBlk1d per head (no upsample) instead of 3
//! - AdaIN decomposed as InstanceNorm + style affine (gamma*x + beta)
//! - Style influence modeled as pre-computed gamma/beta vectors (not learned projection)
//! - LeakyReLU replaced with ReLU (NY native support)
//! - Output is concatenated [f0..., energy...] for single-output graph
//!
//! **Single-variable approach:** text_features [D_MODEL * SEQ_LEN] and style
//! [STYLE_DIM] are packed into flat_input [FLAT_INPUT_SIZE].
//! Narrow+Reshape in the IR splits them. The style dimension represents
//! pre-computed AdaIN gamma/beta parameters that modulate normalization.
//!
//! Part of #1738: Compositional Verification of Prosody Controls.

use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimensions (small-scale for NY tractability)
// ---------------------------------------------------------------------------

/// Model dimension (production Kokoro: 512).
pub(super) const D_MODEL: usize = 8;

/// Style dimension: 2 * SHARED_DIM for gamma + beta per head.
/// In the production model, style_dim=128 is projected to 2*channels per AdaIN.
/// Here we use the pre-projected AdaIN parameters directly.
pub(super) const STYLE_DIM: usize = 2 * D_MODEL;

/// Sequence length. T=4 gives InstanceNorm meaningful statistics (mean/var
/// over 4 elements per channel). T=1 was degenerate: InstanceNorm normalizes
/// a single element to zero regardless of input, making bounds vacuous. #2637.
pub(super) const SEQ_LEN: usize = 4;

/// Total flat input size: text_features + style packed into one vector.
pub(super) const FLAT_INPUT_SIZE: usize = D_MODEL * SEQ_LEN + STYLE_DIM;

/// F0 projection output dimension.
const F0_OUT_DIM: usize = 1;

/// Energy projection output dimension.
const ENERGY_OUT_DIM: usize = 1;

/// Total output size: [f0, energy] concatenated.
pub(super) const OUTPUT_SIZE: usize = F0_OUT_DIM * SEQ_LEN + ENERGY_OUT_DIM * SEQ_LEN;

/// F0 output starts at index 0.
#[allow(dead_code)]
pub(super) const F0_OUTPUT_START: usize = 0;
/// F0 output ends at this index (exclusive).
#[allow(dead_code)]
pub(super) const F0_OUTPUT_END: usize = F0_OUT_DIM * SEQ_LEN;

/// Energy output starts after F0.
#[allow(dead_code)]
pub(super) const ENERGY_OUTPUT_START: usize = F0_OUT_DIM * SEQ_LEN;
/// Energy output end (exclusive).
#[allow(dead_code)]
pub(super) const ENERGY_OUTPUT_END: usize = OUTPUT_SIZE;

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.01;

// ---------------------------------------------------------------------------
// Input splitting
// ---------------------------------------------------------------------------

/// Flat input layout:
///   [0 .. D_MODEL*SEQ_LEN)         = text_features
///   [D_MODEL*SEQ_LEN .. FLAT_INPUT_SIZE) = style (gamma + beta for AdaIN)
fn add_input_splitting(b: &mut TensorBlockBuilder) -> (TensorNodeId, TensorNodeId, TensorNodeId) {
    let text_size = D_MODEL * SEQ_LEN;

    let flat_input = b.add_input("flat_input", &[FLAT_INPUT_SIZE]);

    // text_features: [D_MODEL * SEQ_LEN] → reshape to [D_MODEL, SEQ_LEN]
    let text_flat = b.add_narrow(flat_input, 0, 0, text_size, &[text_size]);
    let text_input = b.add_reshape(text_flat, &[D_MODEL, SEQ_LEN]);

    // style: [STYLE_DIM] = [2 * D_MODEL] (pre-computed gamma + beta)
    let style_input = b.add_narrow(flat_input, 0, text_size, STYLE_DIM, &[STYLE_DIM]);

    // eps constant
    let eps = b.add_input("eps", &[1]);

    (text_input, style_input, eps)
}

// ---------------------------------------------------------------------------
// Shared feature extraction (simplified BiLSTM → Conv1d)
// ---------------------------------------------------------------------------

/// Conv1d: [D_MODEL, SEQ_LEN] → [D_MODEL, SEQ_LEN].
/// Kernel=1 so no temporal mixing, just a learned linear transform per channel.
fn add_shared_conv(b: &mut TensorBlockBuilder, text_input: TensorNodeId) -> TensorNodeId {
    let shared_w = b.add_input("shared_conv_w", &[D_MODEL, D_MODEL, 1]);
    let shared_b = b.add_input("shared_conv_b", &[D_MODEL]);
    let shared_b_bc = b.add_broadcast_left(shared_b, &[D_MODEL, SEQ_LEN]);
    let shared_out = b.add_conv1d(text_input, shared_w, None, 1, 0, &[D_MODEL, SEQ_LEN]);
    b.add_binary_add(shared_out, shared_b_bc, &[D_MODEL, SEQ_LEN])
}

// ---------------------------------------------------------------------------
// AdainResBlk1d (simplified)
// ---------------------------------------------------------------------------

/// Add one simplified AdainResBlk1d.
///
/// InstanceNorm(x, no_affine) → gamma * x + beta → ReLU → Conv1d → residual + scale
///
/// gamma and beta come from the style input (pre-computed AdaIN parameters).
/// This decomposition avoids passing variable-derived gamma/beta to NY's
/// InstanceNorm layer (which requires constant affine parameters).
fn add_adain_resblk(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    style_input: TensorNodeId,
    eps: TensorNodeId,
    prefix: &str,
) -> TensorNodeId {
    let shape = [D_MODEL, SEQ_LEN];

    // Split style into gamma [D_MODEL] and beta [D_MODEL]
    let gamma = b.add_narrow(style_input, 0, 0, D_MODEL, &[D_MODEL]);
    let beta = b.add_narrow(style_input, 0, D_MODEL, D_MODEL, &[D_MODEL]);

    // InstanceNorm without affine parameters
    let normed = b.add_instance_norm(input, eps, 1, None, None, &shape);

    // Broadcast gamma/beta from [D_MODEL] to [D_MODEL, SEQ_LEN]
    let gamma_bc = b.add_broadcast_left(gamma, &shape);
    let beta_bc = b.add_broadcast_left(beta, &shape);

    // AdaIN: gamma * normed + beta
    let scaled = b.add_binary_mul(normed, gamma_bc, &shape);
    let affined = b.add_binary_add(scaled, beta_bc, &shape);

    // ReLU
    let activated = b.add_relu(affined, &shape);

    // Conv1d: [D_MODEL, SEQ_LEN] → [D_MODEL, SEQ_LEN]
    let conv_w = b.add_input(&format!("{prefix}_conv_w"), &[D_MODEL, D_MODEL, 3]);
    let conv_b = b.add_input(&format!("{prefix}_conv_b"), &[D_MODEL]);
    let conv_b_bc = b.add_broadcast_left(conv_b, &shape);
    let conv_out = b.add_conv1d(activated, conv_w, None, 1, 1, &shape);
    let conv_biased = b.add_binary_add(conv_out, conv_b_bc, &shape);

    // Residual: (input + conv_biased) * (1/√2)
    let sum = b.add_binary_add(input, conv_biased, &shape);
    let inv_sqrt2 = b.add_input(&format!("{prefix}_inv_sqrt2"), &[1]);
    let inv_sqrt2_bc = b.add_broadcast(inv_sqrt2, &shape);
    b.add_binary_mul(sum, inv_sqrt2_bc, &shape)
}

// ---------------------------------------------------------------------------
// F0/Energy projection heads
// ---------------------------------------------------------------------------

/// Linear: [D_MODEL, SEQ_LEN] → transpose → [SEQ_LEN, D_MODEL] → matmul → flatten.
fn add_projection_head(
    b: &mut TensorBlockBuilder,
    input: TensorNodeId,
    out_dim: usize,
    prefix: &str,
) -> TensorNodeId {
    let transposed = b.add_transpose(input, &[1, 0], &[SEQ_LEN, D_MODEL]);
    let proj_w = b.add_input(&format!("{prefix}_proj_w"), &[out_dim, D_MODEL]);
    let proj_b = b.add_input(&format!("{prefix}_proj_b"), &[out_dim]);
    let projected = b.add_matmul(transposed, proj_w, true, None, &[SEQ_LEN, out_dim]);
    let proj_b_bc = b.add_broadcast(proj_b, &[SEQ_LEN, out_dim]);
    let biased = b.add_binary_add(projected, proj_b_bc, &[SEQ_LEN, out_dim]);
    b.add_reshape(biased, &[out_dim * SEQ_LEN])
}

// ---------------------------------------------------------------------------
// Full F0EnergyPredictor builder
// ---------------------------------------------------------------------------

/// Build a simplified Kokoro F0EnergyPredictor graph.
///
/// Architecture:
///   flat_input [FLAT_INPUT_SIZE] → split → text [D_MODEL, SEQ_LEN], style [STYLE_DIM]
///   → shared Conv1d → [D_MODEL, SEQ_LEN]
///   → F0 head:     AdaIN(style) → ReLU → Conv1d → residual → Linear → [F0_OUT_DIM]
///   → Energy head:  AdaIN(style) → ReLU → Conv1d → residual → Linear → [ENERGY_OUT_DIM]
///   → Concat → [OUTPUT_SIZE]
pub(super) fn build_kokoro_f0_energy() -> (TensorKernelDef, [usize; 1]) {
    // Compile-time guard: InstanceNorm spatial dim must be > 1 (#2637).
    const _: () = assert!(SEQ_LEN > 1);
    let mut b = TensorBlockBuilder::new("kokoro_f0_energy_verify");

    let (text_input, style_input, eps) = add_input_splitting(&mut b);
    let shared = add_shared_conv(&mut b, text_input);

    let f0_resblk = add_adain_resblk(&mut b, shared, style_input, eps, "f0");
    let f0_out = add_projection_head(&mut b, f0_resblk, F0_OUT_DIM, "f0");

    let energy_resblk = add_adain_resblk(&mut b, shared, style_input, eps, "energy");
    let energy_out = add_projection_head(&mut b, energy_resblk, ENERGY_OUT_DIM, "energy");

    // Concat F0 + energy. Axis 0 is reserved, so reshape to 2D first.
    let f0_2d = b.add_reshape(f0_out, &[1, F0_OUT_DIM * SEQ_LEN]);
    let energy_2d = b.add_reshape(energy_out, &[1, ENERGY_OUT_DIM * SEQ_LEN]);
    let concat_2d = b.add_concat(&[f0_2d, energy_2d], 1, &[1, OUTPUT_SIZE]);
    let output = b.add_reshape(concat_2d, &[OUTPUT_SIZE]);

    (
        b.build(output)
            .expect("valid kokoro F0EnergyPredictor graph"),
        [OUTPUT_SIZE],
    )
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

/// Push bindings for one AdainResBlk1d (conv + inv_sqrt2 only; no style projection).
fn push_adain_resblk_bindings(bindings: &mut Vec<TensorParamBinding>) {
    // conv_w [D_MODEL, D_MODEL, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL, 3]),
        WEIGHT_MAG,
    )));
    // conv_b [D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));
    // inv_sqrt2 [1]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        1.0 / std::f64::consts::SQRT_2 as f32,
    )));
}

/// Push bindings for a projection head (Linear).
fn push_projection_bindings(bindings: &mut Vec<TensorParamBinding>, out_dim: usize) {
    // proj_w [out_dim, D_MODEL]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_dim, D_MODEL]),
        WEIGHT_MAG,
    )));
    // proj_b [out_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out_dim]),
        0.0f32,
    )));
}

/// Build parameter bindings for the F0EnergyPredictor graph.
///
/// Input order matches `add_input` call order:
/// 1. flat_input [FLAT_INPUT_SIZE] — Variable
/// 2. eps [1] — ConstantScalar
/// 3. shared_conv_w [D_MODEL, D_MODEL, 1] — ConstantTensor
/// 4. shared_conv_b [D_MODEL] — ConstantTensor
/// 5. f0_conv_w, f0_conv_b, f0_inv_sqrt2 — ConstantTensor (AdainResBlk)
/// 6. f0_proj_w, f0_proj_b — ConstantTensor (projection head)
/// 7. energy_conv_w, energy_conv_b, energy_inv_sqrt2 — ConstantTensor
/// 8. energy_proj_w, energy_proj_b — ConstantTensor
#[allow(clippy::vec_init_then_push)]
pub(super) fn kokoro_f0_energy_bindings() -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // 1. flat_input — Variable
    bindings.push(TensorParamBinding::Variable);

    // 2. eps
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // 3. Shared Conv1d
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL, D_MODEL, 1]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[D_MODEL]),
        0.0f32,
    )));

    // 4. F0 head: AdainResBlk + projection
    push_adain_resblk_bindings(&mut bindings);
    push_projection_bindings(&mut bindings, F0_OUT_DIM);

    // 5. Energy head: AdainResBlk + projection
    push_adain_resblk_bindings(&mut bindings);
    push_projection_bindings(&mut bindings, ENERGY_OUT_DIM);

    bindings
}
