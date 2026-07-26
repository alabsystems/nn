// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "macos")]
#![allow(dead_code, unreachable_pub)]

//! Consolidated GPU op tests: nn forward pass GPU dispatch,
//! flash attention, op coverage, and polar/rect conversion.

mod test_utils;

#[path = "gpu_ops/fused_norm_shape_benchmark.rs"]
mod fused_norm_shape_benchmark;
#[path = "gpu_ops/gpu_flash_attn.rs"]
mod gpu_flash_attn;
#[path = "gpu_ops/gpu_op_coverage.rs"]
mod gpu_op_coverage;
#[path = "gpu_ops/gpu_polar_to_rect.rs"]
mod gpu_polar_to_rect;
#[path = "gpu_ops/gpu_sage_attn.rs"]
mod gpu_sage_attn;
#[path = "gpu_ops/nn_gpu_conv1d_gemm.rs"]
mod nn_gpu_conv1d_gemm;
#[path = "gpu_ops/nn_gpu_elementwise_f16.rs"]
mod nn_gpu_elementwise_f16;
#[path = "gpu_ops/nn_gpu_forward.rs"]
mod nn_gpu_forward;
#[path = "gpu_ops/nn_gpu_forward_div.rs"]
mod nn_gpu_forward_div;
#[path = "gpu_ops/nn_gpu_forward_lstm.rs"]
mod nn_gpu_forward_lstm;
#[path = "gpu_ops/nn_gpu_forward_ops.rs"]
mod nn_gpu_forward_ops;
#[path = "gpu_ops/nn_gpu_scalar_f16.rs"]
mod nn_gpu_scalar_f16;
