// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compose verification for Kokoro FusedResBlock patterns.
//!
//! FusedResBlock is the most dispatched pattern in the Kokoro pipeline (54 Metal
//! dispatches per synthesis). This module adds NY compose tests for:
//!
//! 1. **Single Generator ResBlock** (from `kokoro_resblock.rs`):
//!    `InstanceNorm -> Snake -> Conv1d -> InstanceNorm -> Snake -> Conv1d + residual`
//!
//! 2. **f0_energy 3-block AdainResBlk1d chain** (from `kokoro_f0.rs`):
//!    `InstanceNorm -> LeakyReLU -> Conv1d -> InstanceNorm -> LeakyReLU -> Conv1d + residual`
//!    with block 1 applying 2x ConvTranspose1d upsample. Scaled by `1/sqrt(2)`.
//!
//! 3. **noise_res** (Conv1d -> LeakyReLU chain):
//!    Simple pattern used in the Kokoro noise module.
//!
//! All tests use small dims (C<=16, T<=8) and IbpValidated soundness mode for
//! normalization paths, per the issue specification.
//!
//! Part of #3574: Compose verification for Kokoro FusedResBlock chains.

use super::common::{
    assert_bounds_valid, assert_bounds_width, assert_crown_tighter_when_not_fallback,
    assert_norm_spatial_non_degenerate, bounds_min_max, uniform_bounds, verify_and_assert,
};
use nn_dsl::build_snake_scalar_kernel;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::{tensor_kernel_to_graph, TensorParamBinding, VerificationSoundnessMode};
use ndarray::{ArrayD, IxDyn};

// ===========================================================================
// Constants
// ===========================================================================

/// Small weight magnitude for bounded verification.
const WEIGHT_MAG: f32 = 0.01;

/// Channels for single ResBlock tests.
const RB_CH: usize = 8;
/// Temporal dimension for ResBlock tests (must be > 1 for InstanceNorm).
const RB_T: usize = 8;

/// Channels for f0_energy tests (BiLSTM output = 2*hidden).
const F0_CH: usize = 16;
/// Reduced channels after upsample block (hidden dim).
const F0_CH_HALF: usize = 8;
/// Temporal dimension for f0_energy input.
const F0_T: usize = 4;
/// Temporal dimension after 2x upsample.
const F0_T_UP: usize = F0_T * 2;

/// Channels for noise_res tests.
const NR_CH: usize = 8;
/// Temporal dimension for noise_res.
const NR_T: usize = 8;

// ===========================================================================
// Builder: Single Generator ResBlock
// ===========================================================================

/// Build a single Generator ResBlock graph.
///
/// Architecture (from `kokoro_resblock.rs` `ResBlock::forward`):
///   InstanceNorm1 -> Snake1 -> Conv1d(dilated) -> InstanceNorm2 -> Snake2 -> Conv1d(1) + residual
///
/// Input: `[C, T]` (Variable).
/// Output: `[C, T]`.
fn build_single_resblock(channels: usize, time_len: usize) -> TensorKernelDef {
    assert_norm_spatial_non_degenerate(time_len, "resblock");
    let shape = [channels, time_len];

    let mut b = TensorBlockBuilder::new("kokoro_single_resblock");

    // Inputs
    let x = b.add_input("x", &shape);
    let eps = b.add_input("eps", &[1]);

    // InstanceNorm1 (affine via style projection)
    let gamma1 = b.add_input("gamma1", &[channels]);
    let beta1 = b.add_input("beta1", &[channels]);
    let norm1 = b.add_instance_norm(x, eps, 1, Some(gamma1), Some(beta1), &shape);

    // Snake1 activation
    let alpha1 = b.add_input("alpha1", &[1]);
    let alpha1_bc = b.add_broadcast(alpha1, &shape);
    let snake_kernel1 = build_snake_scalar_kernel().expect("snake kernel");
    let snake1 = b.add_elementwise(snake_kernel1, &[norm1, alpha1_bc], &shape);

    // Conv1d (dilated, kernel=3, dilation=1, padding=1)
    let conv1_w = b.add_input("conv1_w", &[channels, channels, 3]);
    let conv1_b = b.add_input("conv1_b", &[channels]);
    let conv1 = b.add_conv1d(snake1, conv1_w, Some(conv1_b), 1, 1, &shape);

    // InstanceNorm2
    let gamma2 = b.add_input("gamma2", &[channels]);
    let beta2 = b.add_input("beta2", &[channels]);
    let norm2 = b.add_instance_norm(conv1, eps, 1, Some(gamma2), Some(beta2), &shape);

    // Snake2 activation
    let alpha2 = b.add_input("alpha2", &[1]);
    let alpha2_bc = b.add_broadcast(alpha2, &shape);
    let snake_kernel2 = build_snake_scalar_kernel().expect("snake kernel");
    let snake2 = b.add_elementwise(snake_kernel2, &[norm2, alpha2_bc], &shape);

    // Conv1d (no dilation, kernel=3, padding=1)
    let conv2_w = b.add_input("conv2_w", &[channels, channels, 3]);
    let conv2_b = b.add_input("conv2_b", &[channels]);
    let conv2 = b.add_conv1d(snake2, conv2_w, Some(conv2_b), 1, 1, &shape);

    // Residual connection: x + conv2
    let out = b.add_binary_add(x, conv2, &shape);

    b.build(out).expect("valid single resblock graph")
}

/// Bindings for a single Generator ResBlock.
fn single_resblock_bindings(channels: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 1.0f32)), // gamma1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // beta1
        TensorParamBinding::ConstantScalar(1.0),  // alpha1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, 3]),
            WEIGHT_MAG,
        )), // conv1_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // conv1_b
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 1.0f32)), // gamma2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // beta2
        TensorParamBinding::ConstantScalar(1.0),                                           // alpha2
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, 3]),
            WEIGHT_MAG,
        )), // conv2_w
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)), // conv2_b
    ]
}

// ===========================================================================
// Builder: f0_energy AdainResBlk1d (single block)
// ===========================================================================

/// Build a single f0_energy AdainResBlk1d graph.
///
/// Architecture (from `kokoro_f0.rs` `AdainResBlk1d::forward_impl`):
///   Residual: InstanceNorm1 -> LeakyReLU(0.2) -> [ConvTranspose1d 2x if upsample]
///     -> Conv1d(k=3,p=1) -> InstanceNorm2 -> LeakyReLU(0.2) -> Conv1d(k=3,p=1)
///   Shortcut: [Conv1d(k=1) if dim change or upsample]
///   Output: (residual + shortcut) * (1/sqrt(2))
///
/// Input: `[C_in, T]` (Variable).
/// Output: `[C_out, T_out]`.
fn build_f0_adain_resblk(
    dim_in: usize,
    dim_out: usize,
    time_in: usize,
    upsample: bool,
) -> (TensorKernelDef, [usize; 2]) {
    let time_out = if upsample { time_in * 2 } else { time_in };
    let in_shape = [dim_in, time_in];
    let out_shape = [dim_out, time_out];

    assert_norm_spatial_non_degenerate(time_in, "f0_adain_resblk");
    if upsample {
        assert_norm_spatial_non_degenerate(time_out, "f0_adain_resblk_upsample");
    }

    let mut b = TensorBlockBuilder::new("kokoro_f0_adain_resblk");

    // Inputs
    let x = b.add_input("x", &in_shape);
    let eps = b.add_input("eps", &[1]);

    // --- Residual path ---

    // InstanceNorm1 (AdaIN: InstanceNorm + style affine)
    let gamma1 = b.add_input("gamma1", &[dim_in]);
    let beta1 = b.add_input("beta1", &[dim_in]);
    let norm1 = b.add_instance_norm(x, eps, 1, Some(gamma1), Some(beta1), &in_shape);

    // LeakyReLU(0.2)
    let act1 = b.add_leaky_relu(norm1, 0.2, &in_shape);

    // Optional ConvTranspose1d 2x upsample on residual path
    let after_up = if upsample {
        let up_shape = [dim_in, time_out];
        // ConvTranspose1d: stride=2, kernel=3, padding=1, output_padding=1,
        // groups=dim_in (depthwise)
        let pool_w = b.add_input("pool_w", &[dim_in, 1, 3]);
        let pool_b = b.add_input("pool_b", &[dim_in]);
        b.add_conv_transpose_1d(act1, pool_w, Some(pool_b), 2, 1, 1, dim_in, 1, &up_shape)
    } else {
        act1
    };

    // Conv1d(kernel=3, padding=1): [C_in, T_out] -> [C_out, T_out]
    let c1_w = b.add_input("c1_w", &[dim_out, dim_in, 3]);
    let c1_b = b.add_input("c1_b", &[dim_out]);
    let conv1 = b.add_conv1d(after_up, c1_w, Some(c1_b), 1, 1, &out_shape);

    // InstanceNorm2
    let gamma2 = b.add_input("gamma2", &[dim_out]);
    let beta2 = b.add_input("beta2", &[dim_out]);
    let norm2 = b.add_instance_norm(conv1, eps, 1, Some(gamma2), Some(beta2), &out_shape);

    // LeakyReLU(0.2)
    let act2 = b.add_leaky_relu(norm2, 0.2, &out_shape);

    // Conv1d(kernel=3, padding=1): [C_out, T_out] -> [C_out, T_out]
    let c2_w = b.add_input("c2_w", &[dim_out, dim_out, 3]);
    let c2_b = b.add_input("c2_b", &[dim_out]);
    let residual = b.add_conv1d(act2, c2_w, Some(c2_b), 1, 1, &out_shape);

    // --- Shortcut path ---
    // If dim changes or upsampling, apply 1x1 Conv1d on upsampled input.
    let shortcut = if dim_in != dim_out || upsample {
        let skip_w = b.add_input("skip_w", &[dim_out, dim_in, 1]);
        if upsample {
            // Depthwise ConvTranspose1d on shortcut path
            let skip_pool_w = b.add_input("skip_pool_w", &[dim_in, 1, 3]);
            let skip_pool_b = b.add_input("skip_pool_b", &[dim_in]);
            let shortcut_up = b.add_conv_transpose_1d(
                x,
                skip_pool_w,
                Some(skip_pool_b),
                2,
                1,
                1,
                dim_in,
                1,
                &[dim_in, time_out],
            );
            b.add_conv1d(shortcut_up, skip_w, None, 1, 0, &out_shape)
        } else {
            b.add_conv1d(x, skip_w, None, 1, 0, &out_shape)
        }
    } else {
        x
    };

    // (residual + shortcut) * (1/sqrt(2))
    let sum = b.add_binary_add(residual, shortcut, &out_shape);
    let inv_sqrt2 = b.add_input("inv_sqrt2", &[1]);
    let inv_sqrt2_bc = b.add_broadcast(inv_sqrt2, &out_shape);
    let out = b.add_binary_mul(sum, inv_sqrt2_bc, &out_shape);

    (
        b.build(out).expect("valid f0 adain resblk graph"),
        out_shape,
    )
}

/// Bindings for a single f0_energy AdainResBlk1d.
fn f0_adain_resblk_bindings(
    dim_in: usize,
    dim_out: usize,
    upsample: bool,
) -> Vec<TensorParamBinding> {
    let inv_sqrt2_val = 1.0 / std::f64::consts::SQRT_2;
    let mut bindings = vec![
        TensorParamBinding::Variable,             // x
        TensorParamBinding::ConstantScalar(1e-5), // eps
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim_in]), 1.0f32)), // gamma1
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[dim_in]), 0.0f32)), // beta1
    ];

    if upsample {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[dim_in, 1, 3]),
            WEIGHT_MAG,
        ))); // pool_w
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[dim_in]),
            0.0f32,
        ))); // pool_b
    }

    // Conv1 weights
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[dim_out, dim_in, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[dim_out]),
        0.0f32,
    )));

    // InstanceNorm2
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[dim_out]),
        1.0f32,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[dim_out]),
        0.0f32,
    )));

    // Conv2 weights
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[dim_out, dim_out, 3]),
        WEIGHT_MAG,
    )));
    bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[dim_out]),
        0.0f32,
    )));

    // Skip conv (when dim changes or upsampling)
    if dim_in != dim_out || upsample {
        bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[dim_out, dim_in, 1]),
            WEIGHT_MAG,
        )));
        if upsample {
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[dim_in, 1, 3]),
                WEIGHT_MAG,
            )));
            bindings.push(TensorParamBinding::ConstantTensor(ArrayD::from_elem(
                IxDyn(&[dim_in]),
                0.0f32,
            )));
        }
    }

    // Scale factor: 1/sqrt(2)
    bindings.push(TensorParamBinding::ConstantScalar(inv_sqrt2_val as f32));

    bindings
}

// ===========================================================================
// Builder: noise_res (Conv1d -> LeakyReLU chain)
// ===========================================================================

/// Build a noise_res graph: Conv1d -> LeakyReLU -> Conv1d -> LeakyReLU.
///
/// Simple two-layer pattern used in Kokoro noise residual blocks.
///
/// Input: `[C, T]` (Variable).
/// Output: `[C, T]`.
fn build_noise_res(channels: usize, time_len: usize) -> TensorKernelDef {
    let shape = [channels, time_len];
    let mut b = TensorBlockBuilder::new("kokoro_noise_res");

    let x = b.add_input("x", &shape);

    // Conv1d(kernel=3, padding=1) -> LeakyReLU(0.2)
    let conv1_w = b.add_input("conv1_w", &[channels, channels, 3]);
    let conv1_b = b.add_input("conv1_b", &[channels]);
    let conv1 = b.add_conv1d(x, conv1_w, Some(conv1_b), 1, 1, &shape);
    let act1 = b.add_leaky_relu(conv1, 0.2, &shape);

    // Conv1d(kernel=3, padding=1) -> LeakyReLU(0.2)
    let conv2_w = b.add_input("conv2_w", &[channels, channels, 3]);
    let conv2_b = b.add_input("conv2_b", &[channels]);
    let conv2 = b.add_conv1d(act1, conv2_w, Some(conv2_b), 1, 1, &shape);
    let out = b.add_leaky_relu(conv2, 0.2, &shape);

    b.build(out).expect("valid noise_res graph")
}

/// Bindings for a noise_res block.
fn noise_res_bindings(channels: usize) -> Vec<TensorParamBinding> {
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[channels, channels, 3]),
            WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[channels]), 0.0f32)),
    ]
}

// ===========================================================================
// Tests: Single Generator ResBlock
// ===========================================================================

/// AC1: Single ResBlock TensorKernelDef validates.
#[test]
fn test_fused_resblock_single_def_validates() {
    let def = build_single_resblock(RB_CH, RB_T);
    def.validate().expect("single resblock should validate");
}

/// AC1: Single ResBlock graph builds with expected depth.
#[test]
fn test_fused_resblock_single_graph_builds() {
    let def = build_single_resblock(RB_CH, RB_T);
    let bindings = single_resblock_bindings(RB_CH);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // `x` is the only Variable (NETWORK_INPUT sentinel); all weights/eps/alpha
    // bind as constants and fold into their layers. The Snake activations fuse to
    // a single native Snake node each, and InstanceNorm fuses to one node each.
    // Translated ops: InstanceNorm (1) + Snake (1) + Conv (1) + InstanceNorm (1)
    // + Snake (1) + Conv (1) + residual add (1) = 7 nodes.
    assert!(
        graph.num_nodes() >= 7,
        "single resblock should have >= 7 nodes, got {}",
        graph.num_nodes()
    );
}

/// AC1: IBP propagates through single ResBlock with finite bounds.
#[test]
fn test_fused_resblock_single_ibp() {
    let def = build_single_resblock(RB_CH, RB_T);
    let bindings = single_resblock_bindings(RB_CH);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[RB_CH, RB_T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through single resblock");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[RB_CH, RB_T],
        "resblock output shape must be [C, T]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Single ResBlock IBP: bounds=[{lo_min}, {hi_max}]");
    assert!(lo_min.is_finite(), "lower bound must be finite");
    assert!(hi_max.is_finite(), "upper bound must be finite");

    // Residual connection ensures width stays bounded.
    assert_bounds_width(&output, 200.0, "single_resblock_ibp");
}

/// AC4: CROWN propagation through single ResBlock (Conv1d -> norm -> activation).
///
/// InstanceNorm requires IbpValidated mode (heuristic CROWN linearization).
#[test]
fn test_fused_resblock_single_crown() {
    let def = build_single_resblock(RB_CH, RB_T);
    let bindings = single_resblock_bindings(RB_CH);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[RB_CH, RB_T], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[RB_CH, RB_T]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("Single ResBlock CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// AC1: Verify and record single ResBlock under status key.
#[test]
fn test_fused_resblock_single_verify_and_record() {
    let def = build_single_resblock(RB_CH, RB_T);
    let bindings = single_resblock_bindings(RB_CH);
    let input = uniform_bounds(&[RB_CH, RB_T], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_fused_resblock_single");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[RB_CH, RB_T]);

    // InstanceNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "ResBlock with InstanceNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: f0_energy AdainResBlk1d blocks
// ===========================================================================

/// AC2: Single f0 AdainResBlk1d (no upsample, same dim) validates.
#[test]
fn test_f0_adain_resblk_no_upsample_def_validates() {
    let (def, out_shape) = build_f0_adain_resblk(F0_CH, F0_CH, F0_T, false);
    def.validate()
        .expect("f0 adain resblk (no upsample) should validate");
    assert_eq!(out_shape, [F0_CH, F0_T]);
}

/// AC2: Single f0 AdainResBlk1d (with upsample + dim change) validates.
#[test]
fn test_f0_adain_resblk_upsample_def_validates() {
    let (def, out_shape) = build_f0_adain_resblk(F0_CH, F0_CH_HALF, F0_T, true);
    def.validate()
        .expect("f0 adain resblk (upsample) should validate");
    assert_eq!(out_shape, [F0_CH_HALF, F0_T_UP]);
}

/// AC2: IBP propagates through f0 AdainResBlk1d (no upsample).
#[test]
fn test_f0_adain_resblk_no_upsample_ibp() {
    let (def, _) = build_f0_adain_resblk(F0_CH, F0_CH, F0_T, false);
    let bindings = f0_adain_resblk_bindings(F0_CH, F0_CH, false);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[F0_CH, F0_T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through f0 adain resblk");
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("f0 AdainResBlk1d (no upsample) IBP: bounds=[{lo_min}, {hi_max}]");
    assert_bounds_width(&output, 200.0, "f0_adain_no_upsample_ibp");
}

/// AC2: IBP propagates through f0 AdainResBlk1d (with 2x upsample).
#[test]
fn test_f0_adain_resblk_upsample_ibp() {
    let (def, out_shape) = build_f0_adain_resblk(F0_CH, F0_CH_HALF, F0_T, true);
    let bindings = f0_adain_resblk_bindings(F0_CH, F0_CH_HALF, true);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[F0_CH, F0_T], 1.0);

    let output = graph
        .propagate_ibp(&input)
        .expect("IBP through f0 adain resblk (upsample)");
    assert_bounds_valid(&output);
    assert_eq!(output.lower_upper().0.shape(), &out_shape);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("f0 AdainResBlk1d (upsample) IBP: bounds=[{lo_min}, {hi_max}]");
    assert_bounds_width(&output, 500.0, "f0_adain_upsample_ibp");
}

/// AC2: 3-block f0_energy chain with upsample -- IBP through full chain.
///
/// Block 0: [F0_CH, F0_T] -> [F0_CH, F0_T]               (same dim, no upsample)
/// Block 1: [F0_CH, F0_T] -> [F0_CH_HALF, F0_T_UP]       (dim reduction + 2x up)
/// Block 2: [F0_CH_HALF, F0_T_UP] -> [F0_CH_HALF, F0_T_UP] (same dim, no upsample)
///
/// Each block's output bounds become the next block's input bounds.
#[test]
fn test_f0_energy_3block_chain_ibp() {
    // Block 0: same dim, no upsample
    let (def0, _) = build_f0_adain_resblk(F0_CH, F0_CH, F0_T, false);
    let bindings0 = f0_adain_resblk_bindings(F0_CH, F0_CH, false);
    let graph0 = tensor_kernel_to_graph(&def0, &bindings0).expect("block 0 graph");
    let input0 = uniform_bounds(&[F0_CH, F0_T], 1.0);

    let out0 = graph0.propagate_ibp(&input0).expect("block 0 IBP");
    assert_bounds_valid(&out0);
    let (lo0, hi0) = bounds_min_max(&out0);
    eprintln!("f0_energy block 0: bounds=[{lo0}, {hi0}]");

    // Block 1: dim reduction + 2x upsample
    let (def1, _) = build_f0_adain_resblk(F0_CH, F0_CH_HALF, F0_T, true);
    let bindings1 = f0_adain_resblk_bindings(F0_CH, F0_CH_HALF, true);
    let graph1 = tensor_kernel_to_graph(&def1, &bindings1).expect("block 1 graph");

    // Junction: use block 0 output range as block 1 input.
    let input1 = uniform_bounds(&[F0_CH, F0_T], hi0.max(-lo0));
    let out1 = graph1.propagate_ibp(&input1).expect("block 1 IBP");
    assert_bounds_valid(&out1);
    let (lo1, hi1) = bounds_min_max(&out1);
    eprintln!("f0_energy block 1 (upsample): bounds=[{lo1}, {hi1}]");

    // Block 2: same dim, no upsample (at upsampled temporal)
    let (def2, _) = build_f0_adain_resblk(F0_CH_HALF, F0_CH_HALF, F0_T_UP, false);
    let bindings2 = f0_adain_resblk_bindings(F0_CH_HALF, F0_CH_HALF, false);
    let graph2 = tensor_kernel_to_graph(&def2, &bindings2).expect("block 2 graph");

    let input2 = uniform_bounds(&[F0_CH_HALF, F0_T_UP], hi1.max(-lo1));
    let out2 = graph2.propagate_ibp(&input2).expect("block 2 IBP");
    assert_bounds_valid(&out2);
    let (lo2, hi2) = bounds_min_max(&out2);
    eprintln!("f0_energy block 2: bounds=[{lo2}, {hi2}]");

    // All 3 blocks must produce finite, bounded output.
    assert_bounds_width(&out2, 1000.0, "f0_energy_3block_chain");

    // Per-block expansion should not explode.
    let width0 = hi0 - lo0;
    let width2 = hi2 - lo2;
    let total_expansion = if width0 > 1e-10 { width2 / width0 } else { 1.0 };
    eprintln!(
        "f0_energy 3-block chain: width0={width0:.3}, width2={width2:.3}, \
         total_expansion={total_expansion:.2}x"
    );
    assert!(
        total_expansion < 100.0,
        "3-block chain expansion {total_expansion:.2}x exceeds 100x threshold"
    );
}

/// AC2: CROWN propagation through f0 AdainResBlk1d (no upsample).
#[test]
fn test_f0_adain_resblk_no_upsample_crown() {
    let (def, _) = build_f0_adain_resblk(F0_CH, F0_CH, F0_T, false);
    let bindings = f0_adain_resblk_bindings(F0_CH, F0_CH, false);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[F0_CH, F0_T], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[F0_CH, F0_T]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!(
        "f0 AdainResBlk1d (no upsample) CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]"
    );
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// AC2: Verify and record f0 AdainResBlk1d under status key.
#[test]
fn test_f0_adain_resblk_verify_and_record() {
    let (def, _) = build_f0_adain_resblk(F0_CH, F0_CH, F0_T, false);
    let bindings = f0_adain_resblk_bindings(F0_CH, F0_CH, false);
    let input = uniform_bounds(&[F0_CH, F0_T], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_f0_adain_resblk");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[F0_CH, F0_T]);

    // InstanceNorm uses heuristic normalization approximation.
    assert_eq!(
        result.verification.soundness_mode,
        VerificationSoundnessMode::Heuristic,
        "f0 AdainResBlk1d with InstanceNorm should produce Heuristic, got {:?}",
        result.verification.soundness_mode
    );
}

// ===========================================================================
// Tests: noise_res (Conv1d -> LeakyReLU chain)
// ===========================================================================

/// AC3: noise_res TensorKernelDef validates.
#[test]
fn test_noise_res_def_validates() {
    let def = build_noise_res(NR_CH, NR_T);
    def.validate().expect("noise_res should validate");
}

/// AC3: noise_res graph builds with expected depth.
#[test]
fn test_noise_res_graph_builds() {
    let def = build_noise_res(NR_CH, NR_T);
    let bindings = noise_res_bindings(NR_CH);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");

    // x + Conv1d + LeakyReLU + Conv1d + LeakyReLU = many nodes
    assert!(
        graph.num_nodes() >= 5,
        "noise_res should have >= 5 nodes, got {}",
        graph.num_nodes()
    );
}

/// AC3: IBP propagates through noise_res with finite, tight bounds.
#[test]
fn test_noise_res_ibp() {
    let def = build_noise_res(NR_CH, NR_T);
    let bindings = noise_res_bindings(NR_CH);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NR_CH, NR_T], 1.0);

    let output = graph.propagate_ibp(&input).expect("IBP through noise_res");

    assert_eq!(
        output.lower_upper().0.shape(),
        &[NR_CH, NR_T],
        "noise_res output shape must be [C, T]"
    );
    assert_bounds_valid(&output);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("noise_res IBP: bounds=[{lo_min}, {hi_max}]");

    // Conv + LeakyReLU chain without normalization: bounds should stay tight.
    assert_bounds_width(&output, 50.0, "noise_res_ibp");
}

/// AC3: CROWN propagation through noise_res (no normalization layers).
///
/// noise_res has no InstanceNorm, so CROWN should succeed without fallback.
#[test]
fn test_noise_res_crown() {
    let def = build_noise_res(NR_CH, NR_T);
    let bindings = noise_res_bindings(NR_CH);
    let graph = tensor_kernel_to_graph(&def, &bindings).expect("graph translation");
    let input = uniform_bounds(&[NR_CH, NR_T], 1.0);

    let (method, output, fallback_reason) = assert_crown_tighter_when_not_fallback(&graph, &input);
    assert_eq!(output.lower_upper().0.shape(), &[NR_CH, NR_T]);

    let (lo_min, hi_max) = bounds_min_max(&output);
    eprintln!("noise_res CROWN: method={method:?}, bounds=[{lo_min}, {hi_max}]");
    if let Some(reason) = &fallback_reason {
        eprintln!("Fallback reason: {reason}");
    }
}

/// AC3: Verify and record noise_res under status key.
#[test]
fn test_noise_res_verify_and_record() {
    let def = build_noise_res(NR_CH, NR_T);
    let bindings = noise_res_bindings(NR_CH);
    let input = uniform_bounds(&[NR_CH, NR_T], 1.0);

    let result = verify_and_assert(&def, &bindings, &input, "kokoro_noise_res");
    assert_eq!(result.num_variables, 1, "single Variable input");

    let (lo, _) = result.output_bounds.lower_upper();
    assert_eq!(lo.shape(), &[NR_CH, NR_T]);

    // noise_res has no normalization -- may be Sound or IbpValidated.
    eprintln!(
        "noise_res soundness_mode={:?}",
        result.verification.soundness_mode
    );
}
