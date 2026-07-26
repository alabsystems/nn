// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fused AdaIN residual block compilation.
//!
//! Compiles `TraceOp::FusedAdainResBlock` into a single `TensorKernelDef`
//! dispatch. Each Generator ResBlock (Snake) or F0 AdainResBlk1d (LeakyRelu)
//! becomes one GPU dispatch instead of ~11-14 decomposed dispatches.
//!
//! See `designs/2026-03-15-dilated-path-fusion.md`.

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::{ComputationGraph, ResBlockActivation, TraceNode, WeightRef};

use crate::ir::BinOpKind;
use crate::tensor_block_builder::TensorBlockBuilder;
use crate::tensor_builders::binop_kernel;
use crate::tensor_ir::TensorIRError;

use super::{resolve_input_shape, CompiledKernel, CompiledStep};

/// Compile a fused AdaIN residual block into a single dispatch.
///
/// Tensor inputs from the trace graph: `[x, style]`.
/// All weights are provided as `WeightRef` fields from the `TraceOp`.
///
/// The compiled graph implements:
/// ```text
/// proj1 = Linear(style, adain1_w, adain1_b)
/// gamma1, beta1 = Narrow(proj1) → Reshape to [B, C, 1]
/// h = InstanceNorm(x) → affine(gamma1, beta1) → activation
/// h = Conv1d(h, conv1_w, conv1_b, dilation, padding)
/// proj2 = Linear(style, adain2_w, adain2_b)
/// gamma2, beta2 = Narrow(proj2) → Reshape to [B, C, 1]
/// h = InstanceNorm(h) → affine(gamma2, beta2) → activation
/// h = Conv1d(h, conv2_w, conv2_b, padding)
/// output = (x + h) * residual_scale
/// ```
#[allow(clippy::too_many_arguments)]
pub(in crate::trace_compile) fn compile_fused_adain_resblock(
    node: &TraceNode,
    graph: &ComputationGraph,
    activation: &ResBlockActivation,
    adain1_weight: &WeightRef,
    adain1_bias: &WeightRef,
    adain2_weight: &WeightRef,
    adain2_bias: &WeightRef,
    conv1_weight: &WeightRef,
    conv1_bias: &WeightRef,
    conv1_dilation: usize,
    conv1_padding: usize,
    conv2_weight: &WeightRef,
    conv2_bias: &WeightRef,
    conv2_padding: usize,
    eps: f64,
    residual_scale: f64,
) -> Result<CompiledStep, TensorIRError> {
    let eps_f32 = eps as f32;
    if !eps_f32.is_finite() || eps_f32 < 0.0 {
        return Err(TensorIRError::NonFiniteConstant {
            name: "FusedAdainResBlock eps".into(),
            value: eps,
        });
    }

    let x_shape = resolve_input_shape(node, 0, graph)?;
    let style_shape = resolve_input_shape(node, 1, graph)?;

    if x_shape.len() != 3 {
        return Err(TensorIRError::UnsupportedTraceOp {
            name: format!(
                "FusedAdainResBlock requires rank-3 input [B, C, T], got rank {}",
                x_shape.len()
            ),
        });
    }

    let batch = x_shape[0];
    let c_in = x_shape[1];
    let t_in = x_shape[2];

    // Derive output channels from conv1 weight shape [C_out, C_in, K]
    let c_out = conv1_weight.shape()[0];
    let k1 = conv1_weight.shape()[2];
    let k2 = conv2_weight.shape()[2];

    // Conv1d output length: (L + 2*pad - dilation*(K-1) - 1) / stride + 1
    let conv1_out_len = conv1d_out_len(t_in, k1, conv1_padding, conv1_dilation);
    let conv2_out_len = conv1d_out_len(conv1_out_len, k2, conv2_padding, 1);

    let mut b = TensorBlockBuilder::new("fused_adain_resblock");
    let mut weight_data = HashMap::new();

    // Graph inputs from trace
    let x = b.add_input("input_0", x_shape);
    let style = b.add_input("input_1", style_shape);

    // Scalar weights
    let eps_node = add_scalar_weight(&mut b, &mut weight_data, "eps", eps_f32);
    let ones_node = add_scalar_weight(&mut b, &mut weight_data, "ones", 1.0);

    // =========================================================================
    // Phase 1: AdaIN1 + activation1 + Conv1
    // =========================================================================

    // Style projection 1: Linear(style, w, b) -> [B, 2*C_in]
    let adain1_w = add_weight(&mut b, &mut weight_data, "adain1_w", adain1_weight);
    let adain1_b = add_weight(&mut b, &mut weight_data, "adain1_b", adain1_bias);
    let proj1_shape = [batch, 2 * c_in];
    let proj1 = b.add_linear(style, adain1_w, Some(adain1_b), &proj1_shape);

    // Narrow gamma1, beta1 -> Reshape to [B, C, 1]
    let gamma1_2d_shape = [batch, c_in];
    let gamma1_2d = b.add_narrow(proj1, 1, 0, c_in, &gamma1_2d_shape);
    let beta1_2d = b.add_narrow(proj1, 1, c_in, c_in, &gamma1_2d_shape);
    let affine1_3d = [batch, c_in, 1];
    let gamma1 = b.add_reshape(gamma1_2d, &affine1_3d);
    let beta1 = b.add_reshape(beta1_2d, &affine1_3d);

    // InstanceNorm(x) + affine: (1 + gamma) * normed + beta
    let x_out_shape = &[batch, c_in, t_in];
    let normed1 = b.add_instance_norm(x, eps_node, 2, None, None, x_out_shape);
    let ones1_bc = b.add_broadcast(ones_node, &affine1_3d);
    let scale1 = b.add_binary_add(ones1_bc, gamma1, &affine1_3d);
    let scale1_bc = b.add_broadcast(scale1, x_out_shape);
    let scaled1 = b.add_binary_mul(normed1, scale1_bc, x_out_shape);
    let beta1_bc = b.add_broadcast(beta1, x_out_shape);
    let adain1_out = b.add_binary_add(scaled1, beta1_bc, x_out_shape);

    // Activation 1
    let activated1 = add_activation(
        &mut b,
        &mut weight_data,
        activation,
        true,
        adain1_out,
        x_out_shape,
    )?;

    // Conv1 (dilated)
    let conv1_w = add_weight(&mut b, &mut weight_data, "conv1_w", conv1_weight);
    let conv1_b_node = add_weight(&mut b, &mut weight_data, "conv1_b", conv1_bias);
    let conv1_out_shape = [batch, c_out, conv1_out_len];
    let conv1_out = b.add_conv1d_full(
        activated1,
        conv1_w,
        Some(conv1_b_node),
        1, // stride
        conv1_padding,
        conv1_dilation,
        1, // groups
        &conv1_out_shape,
    );

    // =========================================================================
    // Phase 2: AdaIN2 + activation2 + Conv2
    // =========================================================================

    // Style projection 2: Linear(style, w, b) -> [B, 2*C_out]
    let adain2_w = add_weight(&mut b, &mut weight_data, "adain2_w", adain2_weight);
    let adain2_b = add_weight(&mut b, &mut weight_data, "adain2_b", adain2_bias);
    let proj2_shape = [batch, 2 * c_out];
    let proj2 = b.add_linear(style, adain2_w, Some(adain2_b), &proj2_shape);

    // Narrow gamma2, beta2 -> Reshape to [B, C_out, 1]
    let gamma2_2d_shape = [batch, c_out];
    let gamma2_2d = b.add_narrow(proj2, 1, 0, c_out, &gamma2_2d_shape);
    let beta2_2d = b.add_narrow(proj2, 1, c_out, c_out, &gamma2_2d_shape);
    let affine2_3d = [batch, c_out, 1];
    let gamma2 = b.add_reshape(gamma2_2d, &affine2_3d);
    let beta2 = b.add_reshape(beta2_2d, &affine2_3d);

    // InstanceNorm(conv1_out) + affine
    let normed2 = b.add_instance_norm(conv1_out, eps_node, 2, None, None, &conv1_out_shape);
    let ones2_bc = b.add_broadcast(ones_node, &affine2_3d);
    let scale2 = b.add_binary_add(ones2_bc, gamma2, &affine2_3d);
    let scale2_bc = b.add_broadcast(scale2, &conv1_out_shape);
    let scaled2 = b.add_binary_mul(normed2, scale2_bc, &conv1_out_shape);
    let beta2_bc = b.add_broadcast(beta2, &conv1_out_shape);
    let adain2_out = b.add_binary_add(scaled2, beta2_bc, &conv1_out_shape);

    // Activation 2
    let activated2 = add_activation(
        &mut b,
        &mut weight_data,
        activation,
        false,
        adain2_out,
        &conv1_out_shape,
    )?;

    // Conv2 (no dilation)
    let conv2_w = add_weight(&mut b, &mut weight_data, "conv2_w", conv2_weight);
    let conv2_b_node = add_weight(&mut b, &mut weight_data, "conv2_b", conv2_bias);
    let conv2_out_shape = [batch, c_out, conv2_out_len];
    let conv2_out = b.add_conv1d_full(
        activated2,
        conv2_w,
        Some(conv2_b_node),
        1, // stride
        conv2_padding,
        1, // dilation
        1, // groups
        &conv2_out_shape,
    );

    // =========================================================================
    // Residual: output = (x + conv2_out) * residual_scale
    // =========================================================================
    let output_shape = node.output_shape();
    let residual = b.add_binary_add(x, conv2_out, output_shape);

    let output = if (residual_scale - 1.0).abs() > f64::EPSILON {
        let scale_val = residual_scale as f32;
        let scale_node = add_scalar_weight(&mut b, &mut weight_data, "residual_scale", scale_val);
        let scale_bc = b.add_broadcast(scale_node, output_shape);
        b.add_binary_mul(residual, scale_bc, output_shape)
    } else {
        residual
    };

    let def = b.build(output)?;

    Ok(CompiledStep::Dispatch {
        kernel: CompiledKernel::new(def),
        weight_data,
        external_node_ids: super::graph_input_ids(node, 2),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Add a weight tensor as a builder input and record it in `weight_data`.
fn add_weight(
    b: &mut TensorBlockBuilder,
    weight_data: &mut HashMap<String, WeightRef>,
    name: &str,
    w: &WeightRef,
) -> crate::tensor_ir::TensorNodeId {
    let id = b.add_input(name, w.shape());
    weight_data.insert(name.to_string(), w.clone());
    id
}

/// Add a scalar constant as a weight input.
fn add_scalar_weight(
    b: &mut TensorBlockBuilder,
    weight_data: &mut HashMap<String, WeightRef>,
    name: &str,
    value: f32,
) -> crate::tensor_ir::TensorNodeId {
    let id = b.add_input(name, &[1]);
    weight_data.insert(
        name.to_string(),
        WeightRef::new(vec![value], vec![1]).expect("valid scalar"),
    );
    id
}

/// Add the appropriate activation (Snake or LeakyRelu) after an AdaIN output.
///
/// `is_first`: whether this is the first (alpha1) or second (alpha2) activation
/// in the residual block.
fn add_activation(
    b: &mut TensorBlockBuilder,
    weight_data: &mut HashMap<String, WeightRef>,
    activation: &ResBlockActivation,
    is_first: bool,
    input: crate::tensor_ir::TensorNodeId,
    shape: &[usize],
) -> Result<crate::tensor_ir::TensorNodeId, TensorIRError> {
    match activation {
        ResBlockActivation::Snake { alpha1, alpha2 } => {
            let alpha = if is_first { alpha1 } else { alpha2 };
            let name = if is_first { "alpha1" } else { "alpha2" };
            let alpha_node = add_weight(b, weight_data, name, alpha);
            let alpha_bc = b.add_broadcast_left(alpha_node, shape);
            let snake_kernel = crate::adain::build_snake_scalar_kernel()
                .map_err(|e| TensorIRError::ScalarKernelBuild(e.to_string()))?;
            Ok(b.add_elementwise(snake_kernel, &[input, alpha_bc], shape))
        }
        ResBlockActivation::LeakyRelu { slope } => {
            // Decompose: leaky_relu(x) = relu(x) - slope * relu(-x)
            // Uses only Relu, Sub, Mul, Broadcast — all have MSL codegen support.
            // (TensorOpKind::LeakyRelu does NOT have MSL codegen — #2472 note.)
            #[allow(clippy::cast_possible_truncation)]
            let slope_f32 = *slope as f32;
            let suffix = if is_first { "1" } else { "2" };
            let zero_name = format!("leaky_zero_{suffix}");
            let slope_name = format!("leaky_slope_{suffix}");
            let zero_node = add_scalar_weight(b, weight_data, &zero_name, 0.0);
            let slope_node = add_scalar_weight(b, weight_data, &slope_name, slope_f32);

            let relu_x = b.add_relu(input, shape);
            let zero_bc = b.add_broadcast(zero_node, shape);
            let sub_kernel = binop_kernel("sub", BinOpKind::Sub);
            let neg_x = b.add_elementwise(sub_kernel.clone(), &[zero_bc, input], shape);
            let relu_neg = b.add_relu(neg_x, shape);
            let slope_bc = b.add_broadcast(slope_node, shape);
            let slope_relu = b.add_binary_mul(slope_bc, relu_neg, shape);
            let output = b.add_elementwise(sub_kernel, &[relu_x, slope_relu], shape);
            Ok(output)
        }
    }
}

/// Compute Conv1d output length (stride=1).
fn conv1d_out_len(input_len: usize, kernel_size: usize, padding: usize, dilation: usize) -> usize {
    // output_len = (input_len + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1
    // With stride=1: input_len + 2*padding - dilation*(kernel_size - 1)
    input_len + 2 * padding - dilation * (kernel_size - 1)
}

#[cfg(test)]
#[path = "trace_compile_resblock_tests.rs"]
mod resblock_tests;
