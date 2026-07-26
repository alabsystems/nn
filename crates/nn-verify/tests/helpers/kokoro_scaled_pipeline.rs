// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

// Helpers are shared across multiple test binaries; not all binaries use all functions.
#![allow(dead_code, clippy::duplicated_attributes)]

//! Parameterized builder for scaled Kokoro pipeline composition.
//!
//! Unlike `kokoro_full_pipeline.rs` which uses fixed D_MODEL=8, this module
//! accepts dimensions as parameters so NY composition can be tested
//! at D=16, D=32, D=64 — stepping toward production D=512.
//!
//! Architecture (audio path):
//! ```text
//!   text_features [d_model, seq_len] (Variable)
//!   → TextEncoder: Conv1d + ReLU + Linear → encoded [enc_dim, seq_len]
//!   → Vocoder: Conv1d → LeakyReLU → ConvTranspose1d → ResBlock(InstanceNorm + Snake)
//!     → LeakyReLU → Conv1d → Exp → audio [out_ch, time_up]
//! ```
//!
//! Architecture (duration path):
//! ```text
//!   text_features [d_model, seq_len] (Variable)
//!   → TextEncoder: Conv1d + ReLU + Linear → encoded [enc_dim, seq_len]
//!   → DurationPredictor: Linear → dur_logits [seq_len]
//! ```
//!
//! Part of #1741: THE MOONSHOT — scaling composition proofs toward production.

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_dsl::TensorNodeId;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimension configuration
// ---------------------------------------------------------------------------

/// Pipeline dimensions — parameterized for scaling studies.
#[derive(Debug, Clone, Copy)]
pub(super) struct KokoroDims {
    /// Model dimension (production: 512).
    pub(super) d_model: usize,
    /// Encoder output dimension (typically == d_model).
    pub(super) enc_dim: usize,
    /// Vocoder internal channels (production: 512).
    pub(super) voc_channels: usize,
    /// Vocoder upsampled channels (production: 256).
    pub(super) voc_up_channels: usize,
    /// Vocoder output channels (production: 2 * n_bins).
    pub(super) out_channels: usize,
    /// Sequence length (phoneme tokens).
    pub(super) seq_len: usize,
    /// ConvTranspose1d upsample stride.
    pub(super) upsample_stride: usize,
    /// ConvTranspose1d upsample kernel size.
    pub(super) upsample_kernel: usize,
}

impl KokoroDims {
    /// D=8 scale (current baseline, matches kokoro_full_pipeline.rs).
    pub(super) fn d8() -> Self {
        Self {
            d_model: 8,
            enc_dim: 8,
            voc_channels: 4,
            voc_up_channels: 4,
            out_channels: 4,
            seq_len: 2,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=16 scale — first scaling step.
    pub(super) fn d16() -> Self {
        Self {
            d_model: 16,
            enc_dim: 16,
            voc_channels: 8,
            voc_up_channels: 8,
            out_channels: 4,
            seq_len: 4,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=32 scale — meaningful step toward production.
    pub(super) fn d32() -> Self {
        Self {
            d_model: 32,
            enc_dim: 32,
            voc_channels: 16,
            voc_up_channels: 16,
            out_channels: 8,
            seq_len: 4,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=64 scale — requires per-layer CROWN (#1762).
    pub(super) fn d64() -> Self {
        Self {
            d_model: 64,
            enc_dim: 64,
            voc_channels: 32,
            voc_up_channels: 32,
            out_channels: 16,
            seq_len: 8,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=128 scale — uses per-layer CROWN composition (#1762).
    pub(super) fn d128() -> Self {
        Self {
            d_model: 128,
            enc_dim: 128,
            voc_channels: 64,
            voc_up_channels: 64,
            out_channels: 32,
            seq_len: 8,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=256 scale — approaching production dimensions (512/2).
    ///
    /// This is the highest scaled dimension before production D=512.
    /// Uses per-layer CROWN composition (#1762) to avoid vacuously wide
    /// bounds that monolithic IBP produces at this scale.
    ///
    /// seq_len reduced to 2 (from D=128's 8) to keep CROWN propagation
    /// tractable — D=256 with seq_len=4 still exceeds 20min per layer
    /// (Conv1d weight [256,256,3] = 196K elements creates enormous graph).
    /// The key scaling axis is d_model, not seq_len.
    pub(super) fn d256() -> Self {
        Self {
            d_model: 256,
            enc_dim: 256,
            voc_channels: 128,
            voc_up_channels: 128,
            out_channels: 64,
            seq_len: 2,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=512 scale — production dimensions.
    ///
    /// Kokoro production: d_model=512, voc_channels=512, voc_up_channels=256.
    /// seq_len=2 keeps verification tractable at production width — the key
    /// scaling axis is d_model, not seq_len.
    pub(super) fn d512() -> Self {
        Self {
            d_model: 512,
            enc_dim: 512,
            voc_channels: 256,
            voc_up_channels: 256,
            out_channels: 128,
            seq_len: 2,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=768 scale — intermediate between production D=512 and D=1024.
    pub(super) fn d768() -> Self {
        Self {
            d_model: 768,
            enc_dim: 768,
            voc_channels: 384,
            voc_up_channels: 384,
            out_channels: 192,
            seq_len: 2,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=1024 scale — 2× production, stress-tests provability frontier.
    pub(super) fn d1024() -> Self {
        Self {
            d_model: 1024,
            enc_dim: 1024,
            voc_channels: 512,
            voc_up_channels: 512,
            out_channels: 256,
            seq_len: 2,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// Upsample padding: (kernel - stride) / 2.
    pub(super) fn upsample_padding(&self) -> usize {
        (self.upsample_kernel - self.upsample_stride) / 2
    }

    /// Output time dimension after vocoder upsampling.
    /// conv_transpose1d: (in-1)*stride + kernel - 2*padding
    pub(super) fn time_up(&self) -> usize {
        (self.seq_len - 1) * self.upsample_stride + self.upsample_kernel
            - 2 * self.upsample_padding()
    }

    /// Assert normalization spatial dimensions are non-degenerate (#2637).
    ///
    /// InstanceNorm operates on `[voc_up_channels, time_up]` in the vocoder —
    /// the spatial dimension is `time_up`. At time_up=1, InstanceNorm has
    /// mean=value, var=0, making bounds vacuous (output = bias regardless
    /// of input). Bounds at spatial dim=1 cannot be extrapolated to
    /// production dimensions.
    pub(super) fn assert_norm_dims_valid(&self) {
        let time_up = self.time_up();
        assert!(
            time_up > 1,
            "KokoroDims: vocoder InstanceNorm spatial dim (time_up={time_up}) is \
             degenerate — need > 1. seq_len={}, stride={}, kernel={}. See #2637.",
            self.seq_len,
            self.upsample_stride,
            self.upsample_kernel,
        );
        assert!(
            self.seq_len > 1,
            "KokoroDims: seq_len={} is degenerate for any normalization that \
             reduces over the time dimension. See #2637.",
            self.seq_len,
        );
    }
}

/// Weight magnitude for synthetic test weights.
const WEIGHT_MAG: f32 = 0.001;

// ---------------------------------------------------------------------------
// Parameterized text encoder
// ---------------------------------------------------------------------------

/// Add text encoder: Conv1d + ReLU + Linear projection.
///
/// Input: `[d_model, seq_len]` → Output: `[enc_dim, seq_len]`
fn add_text_encoder_scaled(
    b: &mut TensorBlockBuilder,
    text_input: TensorNodeId,
    dims: &KokoroDims,
) -> TensorNodeId {
    let d = dims.d_model;
    let enc = dims.enc_dim;
    let s = dims.seq_len;

    // Conv1d: [d_model, seq_len] → [d_model, seq_len]
    let enc_conv_w = b.add_input("enc_conv_w", &[d, d, 3]);
    let enc_conv_out = b.add_conv1d(text_input, enc_conv_w, None, 1, 1, &[d, s]);

    // ReLU
    let enc_relu = b.add_relu(enc_conv_out, &[d, s]);

    // Linear: [d_model, seq_len] → [enc_dim, seq_len]
    let enc_t = b.add_transpose(enc_relu, &[1, 0], &[s, d]);
    let enc_proj_w = b.add_input("enc_proj_w", &[enc, d]);
    let enc_proj_b = b.add_input("enc_proj_b", &[enc]);
    let enc_projected = b.add_matmul(enc_t, enc_proj_w, true, None, &[s, enc]);
    let enc_proj_b_bc = b.add_broadcast(enc_proj_b, &[s, enc]);
    let enc_biased = b.add_binary_add(enc_projected, enc_proj_b_bc, &[s, enc]);
    b.add_transpose(enc_biased, &[1, 0], &[enc, s])
}

// ---------------------------------------------------------------------------
// Parameterized vocoder decoder
// ---------------------------------------------------------------------------

/// Add vocoder decoder: Conv1d → LeakyReLU → ConvTranspose1d → ResBlock → LeakyReLU → Conv1d → Exp.
///
/// Input: `[enc_dim, seq_len]` → Output: `[out_channels, time_up]`
fn add_vocoder_decoder_scaled(
    b: &mut TensorBlockBuilder,
    encoded: TensorNodeId,
    dims: &KokoroDims,
) -> TensorNodeId {
    let enc = dims.enc_dim;
    let vc = dims.voc_channels;
    let vup = dims.voc_up_channels;
    let out = dims.out_channels;
    let s = dims.seq_len;
    let t = dims.time_up();
    let up_shape = [vup, t];

    // Shared epsilon for InstanceNorm
    let eps = b.add_input("voc_eps", &[1]);

    // Conv pre: [enc_dim, seq_len] → [voc_channels, seq_len]
    let conv_pre_w = b.add_input("voc_conv_pre_w", &[vc, enc, 3]);
    let x = b.add_conv1d(encoded, conv_pre_w, None, 1, 1, &[vc, s]);

    // LeakyReLU(0.1) before upsample
    let x_act = b.add_leaky_relu(x, 0.1, &[vc, s]);

    // ConvTranspose1d upsample: [voc_channels, seq_len] → [voc_up_channels, time_up]
    let upsample_w = b.add_input("voc_upsample_w", &[vc, vup, dims.upsample_kernel]);
    let x_up = b.add_conv_transpose_1d(
        x_act,
        upsample_w,
        None,
        dims.upsample_stride,
        dims.upsample_padding(),
        1, // dilation
        1, // groups
        0, // output_padding
        &up_shape,
    );

    // ResBlock: InstanceNorm + Snake + Conv1d + residual
    let style_gamma = b.add_input("voc_style_gamma", &[vup]);
    let style_beta = b.add_input("voc_style_beta", &[vup]);
    let normed = b.add_instance_norm(
        x_up,
        eps,
        1, // axis=1 (time)
        Some(style_gamma),
        Some(style_beta),
        &up_shape,
    );

    // Snake activation
    let alpha = b.add_input("voc_alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);

    // Conv1d in ResBlock
    let res_conv_w = b.add_input("voc_res_conv_w", &[vup, vup, 3]);
    let sublayer_out = b.add_conv1d(snake_out, res_conv_w, None, 1, 1, &up_shape);

    // Residual connection
    let res_out = b.add_binary_add(x_up, sublayer_out, &up_shape);

    // LeakyReLU(0.01) before conv_post
    let res_act = b.add_leaky_relu(res_out, 0.01, &up_shape);

    // Conv post: [voc_up_channels, time_up] → [out_channels, time_up]
    let conv_post_w = b.add_input("voc_conv_post_w", &[out, vup, 3]);
    let x_post = b.add_conv1d(res_act, conv_post_w, None, 1, 1, &[out, t]);

    // Exp activation: log-magnitude → magnitude (always positive)
    b.add_exp(x_post, &[out, t])
}

// ---------------------------------------------------------------------------
// Parameterized duration predictor
// ---------------------------------------------------------------------------

/// Add duration predictor: Linear → dur_logits.
///
/// Input: `[enc_dim, seq_len]` → Output: `[seq_len]`
fn add_duration_predictor_scaled(
    b: &mut TensorBlockBuilder,
    encoded: TensorNodeId,
    dims: &KokoroDims,
) -> TensorNodeId {
    let enc = dims.enc_dim;
    let s = dims.seq_len;

    let enc_t = b.add_transpose(encoded, &[1, 0], &[s, enc]);
    let dur_proj_w = b.add_input("dur_proj_w", &[1, enc]);
    let dur_proj_b = b.add_input("dur_proj_b", &[1]);
    let dur_projected = b.add_matmul(enc_t, dur_proj_w, true, None, &[s, 1]);
    let dur_proj_b_bc = b.add_broadcast(dur_proj_b, &[s, 1]);
    let dur_biased = b.add_binary_add(dur_projected, dur_proj_b_bc, &[s, 1]);
    b.add_reshape(dur_biased, &[s])
}

// ---------------------------------------------------------------------------
// Public builders
// ---------------------------------------------------------------------------

/// Build the full Kokoro audio pipeline at the given scale.
///
/// Returns `(TensorKernelDef, [out_channels, time_up])`.
pub(super) fn build_scaled_full_pipeline(dims: &KokoroDims) -> (TensorKernelDef, [usize; 2]) {
    dims.assert_norm_dims_valid();
    let mut b = TensorBlockBuilder::new("kokoro_scaled_pipeline");
    let text_input = b.add_input("text_features", &[dims.d_model, dims.seq_len]);
    let encoded = add_text_encoder_scaled(&mut b, text_input, dims);
    let audio = add_vocoder_decoder_scaled(&mut b, encoded, dims);
    let out_shape = [dims.out_channels, dims.time_up()];
    (
        b.build(audio).expect("valid scaled kokoro pipeline graph"),
        out_shape,
    )
}

/// Build the duration branch at the given scale.
///
/// Returns `(TensorKernelDef, seq_len)`.
pub(super) fn build_scaled_duration_branch(dims: &KokoroDims) -> (TensorKernelDef, usize) {
    let mut b = TensorBlockBuilder::new("kokoro_scaled_duration");
    let text_input = b.add_input("text_features", &[dims.d_model, dims.seq_len]);
    let encoded = add_text_encoder_scaled(&mut b, text_input, dims);
    let dur_logits = add_duration_predictor_scaled(&mut b, encoded, dims);
    (
        b.build(dur_logits)
            .expect("valid scaled duration branch graph"),
        dims.seq_len,
    )
}

// ---------------------------------------------------------------------------
// Bindings builders
// ---------------------------------------------------------------------------

/// Push text encoder weight bindings for the given dimensions.
fn push_text_encoder_bindings_scaled(bindings: &mut Vec<TensorParamBinding>, dims: &KokoroDims) {
    let d = dims.d_model;
    let enc = dims.enc_dim;

    // enc_conv_w [d_model, d_model, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[d, d, 3]),
        WEIGHT_MAG,
    )));
    // enc_proj_w [enc_dim, d_model]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[enc, d]),
        WEIGHT_MAG,
    )));
    // enc_proj_b [enc_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[enc]),
        0.0f32,
    )));
}

/// Push vocoder decoder weight bindings for the given dimensions.
fn push_vocoder_bindings_scaled(bindings: &mut Vec<TensorParamBinding>, dims: &KokoroDims) {
    let enc = dims.enc_dim;
    let vc = dims.voc_channels;
    let vup = dims.voc_up_channels;
    let out = dims.out_channels;

    // voc_eps
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));
    // voc_conv_pre_w [voc_channels, enc_dim, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[vc, enc, 3]),
        WEIGHT_MAG,
    )));
    // voc_upsample_w [voc_channels, voc_up_channels, upsample_kernel]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[vc, vup, dims.upsample_kernel]),
        WEIGHT_MAG,
    )));
    // voc_style_gamma [voc_up_channels]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[vup]),
        1.0f32,
    )));
    // voc_style_beta [voc_up_channels]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[vup]),
        0.0f32,
    )));
    // voc_alpha (scalar for Snake activation)
    bindings.push(TensorParamBinding::ConstantScalar(1.0));
    // voc_res_conv_w [voc_up_channels, voc_up_channels, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[vup, vup, 3]),
        WEIGHT_MAG,
    )));
    // voc_conv_post_w [out_channels, voc_up_channels, 3]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[out, vup, 3]),
        WEIGHT_MAG,
    )));
}

/// Push duration predictor weight bindings for the given dimensions.
fn push_duration_predictor_bindings_scaled(
    bindings: &mut Vec<TensorParamBinding>,
    dims: &KokoroDims,
) {
    let enc = dims.enc_dim;
    // dur_proj_w [1, enc_dim]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1, enc]),
        WEIGHT_MAG,
    )));
    // dur_proj_b [1]
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[1]),
        0.0f32,
    )));
}

/// Build full pipeline bindings at the given scale.
///
/// `text_features` = Variable. All weights = ConstantTensor.
#[allow(clippy::vec_init_then_push)]
pub(super) fn scaled_full_pipeline_bindings(dims: &KokoroDims) -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable);
    push_text_encoder_bindings_scaled(&mut bindings, dims);
    push_vocoder_bindings_scaled(&mut bindings, dims);
    bindings
}

/// Build duration branch bindings at the given scale.
#[allow(clippy::vec_init_then_push)]
pub(super) fn scaled_duration_branch_bindings(dims: &KokoroDims) -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();
    bindings.push(TensorParamBinding::Variable);
    push_text_encoder_bindings_scaled(&mut bindings, dims);
    push_duration_predictor_bindings_scaled(&mut bindings, dims);
    bindings
}
