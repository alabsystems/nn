// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Cross-backend parity tests: CPU vs Metal.
//!
//! Validates that the same operations produce identical results on CPU and
//! Metal backends within floating-point tolerance. Covers:
//!
//! - **Elementwise ops:** relu, gelu, silu, sigmoid, tanh, softmax, exp, log
//! - **Unary ops:** sqrt, abs, recip, sin, cos, neg, sqr, floor, round, fract, gelu_erf
//! - **Binary ops:** add, sub, mul, div, maximum, minimum, broadcast, scalar
//! - **Linear algebra:** matmul (various shapes), linear, batched matmul, transpose, cat
//! - **Normalization:** LayerNorm, BatchNorm, GroupNorm, RmsNorm
//! - **Convolution:** Conv1d, Conv2d, ConvTranspose1d
//! - **Reduce ops:** sum, mean, max, min with keepdim
//! - **Compare/where:** scalar compare, tensor compare, where_cond
//! - **Clamp ops:** clamp, clamp_min, clamp_max
//! - **Shape ops:** permute, transpose 3D, narrow, pad, reshape, repeat, squeeze/unsqueeze
//! - **Selection ops:** embedding, index_select, gather, scatter_add, index_add
//! - **Softmax axes:** softmax/log_softmax along different axes and ranks
//! - **Full model:** MLP, attention block

mod test_utils;

#[path = "cross_backend_parity/binary_ops.rs"]
mod binary_ops;
#[path = "cross_backend_parity/clamp_ops.rs"]
mod clamp_ops;
#[path = "cross_backend_parity/compare_where_ops.rs"]
mod compare_where_ops;
#[path = "cross_backend_parity/convolution.rs"]
mod convolution;
#[path = "cross_backend_parity/elementwise.rs"]
mod elementwise;
#[path = "cross_backend_parity/full_model.rs"]
mod full_model;
#[path = "cross_backend_parity/linear_algebra.rs"]
mod linear_algebra;
#[path = "cross_backend_parity/matmul_shapes.rs"]
mod matmul_shapes;
#[path = "cross_backend_parity/normalization.rs"]
mod normalization;
#[path = "cross_backend_parity/reduce_ops.rs"]
mod reduce_ops;
#[path = "cross_backend_parity/selection_ops.rs"]
mod selection_ops;
#[path = "cross_backend_parity/shape_ops.rs"]
mod shape_ops;
#[path = "cross_backend_parity/softmax_axes.rs"]
mod softmax_axes;
#[path = "cross_backend_parity/unary_ops.rs"]
mod unary_ops;
