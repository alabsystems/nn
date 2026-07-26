// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Parameterized Kokoro decoder builder for scaled compose tests.
//!
//! Unlike `kokoro_decoder.rs` which uses fixed D=8 dimensions, this module
//! accepts dimension parameters so CROWN vs IBP tightness can be tested
//! at D=32 and D=64 — scales where CROWN's linear relaxation provides
//! meaningful tightening over IBP's interval arithmetic.
//!
//! The decoder architecture matches `kokoro_decoder.rs`:
//!   features [in_ch, time_in] (Variable)
//!   → Conv1d conv_pre [ch, time_in]
//!   → LeakyReLU(0.1)
//!   → ConvTranspose1d upsample [up_ch, time_up]
//!   → ResBlock(InstanceNorm + Snake + Conv1d) + residual
//!   → LeakyReLU(0.01)
//!   → Conv1d conv_post [out_ch, time_up]
//!   → Exp
//!
//! Part of #2239: Scale compose dimensions for tighter CROWN bounds.

use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::TensorParamBinding;
use ndarray::{ArrayD, IxDyn};

// ---------------------------------------------------------------------------
// Dimension configuration
// ---------------------------------------------------------------------------

/// Decoder dimensions — parameterized for scaling studies.
#[derive(Debug, Clone, Copy)]
pub(super) struct DecoderDims {
    /// Input channels (production: 512).
    pub(super) in_channels: usize,
    /// Internal channels after conv_pre (production: 512).
    pub(super) channels: usize,
    /// Channels after upsample (production: 256).
    pub(super) up_channels: usize,
    /// Output channels (production: 2 * n_bins).
    pub(super) out_channels: usize,
    /// Input time length.
    pub(super) time_in: usize,
    /// ConvTranspose1d upsample stride.
    pub(super) upsample_stride: usize,
    /// ConvTranspose1d upsample kernel.
    pub(super) upsample_kernel: usize,
}

impl DecoderDims {
    /// D=8 scale (matches kokoro_decoder.rs baseline).
    pub(super) fn d8() -> Self {
        Self {
            in_channels: 8,
            channels: 8,
            up_channels: 4,
            out_channels: 4,
            time_in: 4,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=32 scale — first meaningful dimension for CROWN tightening.
    pub(super) fn d32() -> Self {
        Self {
            in_channels: 32,
            channels: 32,
            up_channels: 16,
            out_channels: 8,
            time_in: 4,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// D=64 scale — demonstrates substantial CROWN tightening over IBP.
    pub(super) fn d64() -> Self {
        Self {
            in_channels: 64,
            channels: 64,
            up_channels: 32,
            out_channels: 16,
            time_in: 4,
            upsample_stride: 2,
            upsample_kernel: 4,
        }
    }

    /// Upsample padding: (kernel - stride) / 2.
    pub(super) fn upsample_padding(&self) -> usize {
        (self.upsample_kernel - self.upsample_stride) / 2
    }

    /// Output time after upsampling.
    /// conv_transpose1d: (in-1)*stride + kernel - 2*padding
    pub(super) fn time_up(&self) -> usize {
        (self.time_in - 1) * self.upsample_stride + self.upsample_kernel
            - 2 * self.upsample_padding()
    }

    /// Assert normalization spatial dimensions are non-degenerate (#2637).
    ///
    /// InstanceNorm operates on `[up_channels, time_up]` — at time_up=1,
    /// mean=value, var=0, making bounds vacuous.
    pub(super) fn assert_norm_dims_valid(&self) {
        let time_up = self.time_up();
        assert!(
            time_up > 1,
            "DecoderDims: InstanceNorm spatial dim (time_up={time_up}) is \
             degenerate — need > 1. See #2637.",
        );
    }
}

/// Weight magnitude for synthetic test weights.
///
/// Small enough that exp(accumulated_value) stays finite through the pipeline,
/// large enough to create meaningful inter-channel variation.
const WEIGHT_MAG: f32 = 0.001;

/// Generate deterministic non-uniform weights for a given shape.
///
/// Uniform weights make all output channels identical, so IBP is already
/// exact — CROWN has no correlations to exploit. Non-uniform weights create
/// channel-dependent values where CROWN's linear relaxation can track
/// inter-channel correlations and produce tighter bounds.
///
/// Uses a simple LCG-based deterministic pattern (not random — reproducible
/// across runs). Values scaled to `[-WEIGHT_MAG, +WEIGHT_MAG]` to keep
/// pre-exp values small (preventing Exp overflow at larger dimensions).
fn nonuniform_weights(shape: &[usize], seed: u64) -> ArrayD<f32> {
    let n_elements: usize = shape.iter().product();
    let scale = WEIGHT_MAG;

    let mut data = Vec::with_capacity(n_elements);
    let mut state = seed;
    for _ in 0..n_elements {
        // Simple LCG: state = (state * 6364136223846793005 + 1) mod 2^64
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        // Map to [-1, 1] then scale
        let val = ((state >> 33) as f32 / (u32::MAX >> 1) as f32) * 2.0 - 1.0;
        data.push(val * scale);
    }
    ArrayD::from_shape_vec(IxDyn(shape), data).expect("valid weight shape")
}

// ---------------------------------------------------------------------------
// Parameterized builder
// ---------------------------------------------------------------------------

/// Build Kokoro decoder with LeakyReLU at given dimensions.
///
/// Returns `(TensorKernelDef, [out_channels, time_up])`.
pub(super) fn build_scaled_decoder(dims: &DecoderDims) -> (TensorKernelDef, [usize; 2]) {
    dims.assert_norm_dims_valid();
    let mut b = TensorBlockBuilder::new("kokoro_decoder_scaled");
    let up_shape = [dims.up_channels, dims.time_up()];

    // Variable input: encoder features
    let input = b.add_input("features", &[dims.in_channels, dims.time_in]);

    // Shared epsilon
    let eps = b.add_input("eps", &[1]);

    // Conv pre: [in_ch, time_in] → [ch, time_in]
    let conv_pre_w = b.add_input("conv_pre_w", &[dims.channels, dims.in_channels, 7]);
    let x = b.add_conv1d(
        input,
        conv_pre_w,
        None,
        1,
        3,
        &[dims.channels, dims.time_in],
    );

    // LeakyReLU(0.1) before upsample
    let x_act = b.add_leaky_relu(x, 0.1, &[dims.channels, dims.time_in]);

    // ConvTranspose1d upsample
    let upsample_w = b.add_input(
        "upsample_w",
        &[dims.channels, dims.up_channels, dims.upsample_kernel],
    );
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
    let style_gamma = b.add_input("res_style_gamma", &[dims.up_channels]);
    let style_beta = b.add_input("res_style_beta", &[dims.up_channels]);
    let normed = b.add_instance_norm(
        x_up,
        eps,
        1, // axis=1 (time)
        Some(style_gamma),
        Some(style_beta),
        &up_shape,
    );

    // Snake activation (scalar alpha for NY tractability)
    let alpha = b.add_input("res_alpha", &[1]);
    let alpha_bc = b.add_broadcast(alpha, &up_shape);
    let snake_kernel = build_snake_scalar_kernel().expect("snake kernel");
    let snake_out = b.add_elementwise(snake_kernel, &[normed, alpha_bc], &up_shape);

    // Conv1d in ResBlock (same-padding)
    let res_conv_w = b.add_input("res_conv_w", &[dims.up_channels, dims.up_channels, 3]);
    let sublayer_out = b.add_conv1d(snake_out, res_conv_w, None, 1, 1, &up_shape);

    // Residual connection
    let res_out = b.add_binary_add(x_up, sublayer_out, &up_shape);

    // LeakyReLU(0.01) before conv_post
    let res_act = b.add_leaky_relu(res_out, 0.01, &up_shape);

    // Conv post
    let conv_post_w = b.add_input("conv_post_w", &[dims.out_channels, dims.up_channels, 7]);
    let x_post = b.add_conv1d(
        res_act,
        conv_post_w,
        None,
        1,
        3,
        &[dims.out_channels, dims.time_up()],
    );

    // Exp activation
    let output = b.add_exp(x_post, &[dims.out_channels, dims.time_up()]);

    let out_shape = [dims.out_channels, dims.time_up()];
    (
        b.build(output).expect("valid scaled kokoro decoder graph"),
        out_shape,
    )
}

/// Build parameter bindings for the scaled decoder with non-uniform weights.
///
/// Non-uniform weights create channel-dependent correlations
/// that CROWN's linear relaxation can exploit for tighter bounds. Uniform
/// weights make all channels identical, causing CROWN to degenerate to IBP.
#[allow(clippy::vec_init_then_push)]
pub(super) fn scaled_decoder_bindings(dims: &DecoderDims) -> Vec<TensorParamBinding> {
    let mut bindings = Vec::new();

    // features: Variable
    bindings.push(TensorParamBinding::Variable);

    // eps: Constant scalar
    bindings.push(TensorParamBinding::ConstantScalar(1e-5));

    // conv_pre weight — non-uniform
    bindings.push(TensorParamBinding::ConstantTensor(nonuniform_weights(
        &[dims.channels, dims.in_channels, 7],
        42,
    )));

    // upsample weight — non-uniform
    bindings.push(TensorParamBinding::ConstantTensor(nonuniform_weights(
        &[dims.channels, dims.up_channels, dims.upsample_kernel],
        137,
    )));

    // ResBlock: style_gamma — non-uniform around 1.0
    let gamma_shape = [dims.up_channels];
    let mut gamma = nonuniform_weights(&gamma_shape, 271);
    gamma.mapv_inplace(|v| 1.0 + v * 0.1); // centered at 1.0
    bindings.push(TensorParamBinding::ConstantTensor(gamma));

    // ResBlock: style_beta — non-uniform around 0.0
    bindings.push(TensorParamBinding::ConstantTensor(nonuniform_weights(
        &[dims.up_channels],
        314,
    )));

    // ResBlock: alpha (scalar)
    bindings.push(TensorParamBinding::ConstantScalar(1.0));

    // ResBlock: conv weight — non-uniform
    bindings.push(TensorParamBinding::ConstantTensor(nonuniform_weights(
        &[dims.up_channels, dims.up_channels, 3],
        577,
    )));

    // conv_post weight — non-uniform
    bindings.push(TensorParamBinding::ConstantTensor(nonuniform_weights(
        &[dims.out_channels, dims.up_channels, 7],
        691,
    )));

    bindings
}
