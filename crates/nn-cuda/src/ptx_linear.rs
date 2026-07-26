// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for fused Linear (matmul + bias) kernels.
//!
//! Generates raw PTX for the Linear layer: `output = input * weight^T + bias`.
//! This is the fused matmul+bias operation used in every transformer layer
//! (attention projections, feed-forward networks, output heads).
//!
//! ## Variants
//!
//! - [`generate_linear_ptx`] — Standard Linear with bias: `Y = X * W^T + b`
//! - [`generate_linear_no_bias_ptx`] — Without bias: `Y = X * W^T`
//! - [`generate_linear_relu_ptx`] — Fused Linear + ReLU: `Y = max(0, X * W^T + b)`
//!
//! ## Weight layout convention
//!
//! Weights are `[out_features, in_features]` (row-major), matching PyTorch
//! `nn.Linear`. The kernel computes `X[batch, in_f] * W^T[in_f, out_f]`,
//! which is equivalent to `X[batch, in_f] * W[in_f, out_f]` when W is
//! stored transposed. For simplicity, the PTX kernel takes W as
//! `[in_features, out_features]` (pre-transposed).
//!
//! ## Thread block configuration
//!
//! Block: `(LINEAR_BLOCK_SIZE, 1, 1)` = `(256, 1, 1)`.
//! Each thread computes one output element.
//! Grid: `(ceil(batch * out_features / 256), 1, 1)`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Thread block size for Linear kernels (256 threads).
pub const LINEAR_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// PTX generation: Linear with bias
// ---------------------------------------------------------------------------

/// Generate PTX for fused Linear (matmul + bias): `Y = X * W + bias`.
///
/// # Layout
///
/// - `input`: `[batch, in_features]` row-major
/// - `weight`: `[in_features, out_features]` row-major (pre-transposed)
/// - `bias`: `[out_features]`
/// - `output`: `[batch, out_features]` row-major
///
/// # Arguments
///
/// * `in_features` — input dimension (K in matmul terms)
/// * `out_features` — output dimension (N in matmul terms)
///
/// Batch size is passed as a runtime parameter.
pub fn generate_linear_ptx(in_features: u32, out_features: u32) -> String {
    let block = LINEAR_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude("sm_70"));
    ptx.push_str(&format!(
        "// Linear (matmul + bias): Y[batch,{out_features}] = \
         X[batch,{in_features}] * W[{in_features},{out_features}] + b[{out_features}]\n\
         // Block: {block}, 1D grid over batch * out_features\n\n"
    ));

    // Kernel entry: input, weight, bias, output, batch_size, in_features, out_features
    ptx.push_str(
        ".visible .entry linear_bias_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_weight,\n\
         \x20   .param .u64 param_bias,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_batch,\n\
         \x20   .param .u32 param_in_features,\n\
         \x20   .param .u32 param_out_features\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block}\n{{\n"));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<16>;\n\
         \x20   .reg .f32  %f<6>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_weight];\n\
         \x20   ld.param.u64  %rd2, [param_bias];\n\
         \x20   ld.param.u64  %rd3, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_batch];\n\
         \x20   ld.param.u32  %r1,  [param_in_features];\n\
         \x20   ld.param.u32  %r2,  [param_out_features];\n\n",
    );

    // Global thread index = blockIdx.x * blockDim.x + threadIdx.x
    ptx.push_str(&format!(
        "\x20   mov.u32       %r3, %tid.x;\n\
         \x20   mov.u32       %r4, %ctaid.x;\n\
         \x20   mad.lo.u32    %r5, %r4, {block}, %r3;  // global_idx\n\n"
    ));

    // total_outputs = batch * out_features
    ptx.push_str(
        "\x20   mul.lo.u32    %r6, %r0, %r2;           // total = batch * out_features\n\
         \x20   setp.ge.u32   %p0, %r5, %r6;           // global_idx >= total?\n\
         \x20   @%p0 bra      LINEAR_EXIT;\n\n",
    );

    // row = global_idx / out_features, col = global_idx % out_features
    ptx.push_str(
        "\x20   div.u32       %r7, %r5, %r2;           // row = global_idx / out_f\n\
         \x20   rem.u32       %r8, %r5, %r2;           // col = global_idx % out_f\n\n",
    );

    // acc = 0.0, loop over in_features
    ptx.push_str(&format!(
        "\x20   mov.f32       %f0, {zero};\n\
         \x20   mov.u32       %r9, 0;                   // i = 0\n\n"
    ));

    // Loop: acc += input[row*in_f + i] * weight[i*out_f + col]
    ptx.push_str(
        "LINEAR_DOT:\n\
         \x20   setp.ge.u32   %p1, %r9, %r1;           // i >= in_features?\n\
         \x20   @%p1 bra      LINEAR_BIAS;\n\
         \x20   // input[row * in_f + i]\n\
         \x20   mad.lo.u32    %r10, %r7, %r1, %r9;\n\
         \x20   mul.wide.u32  %rd4, %r10, 4;\n\
         \x20   add.u64       %rd5, %rd0, %rd4;\n\
         \x20   ld.global.f32 %f1, [%rd5];\n\
         \x20   // weight[i * out_f + col]\n\
         \x20   mad.lo.u32    %r11, %r9, %r2, %r8;\n\
         \x20   mul.wide.u32  %rd6, %r11, 4;\n\
         \x20   add.u64       %rd7, %rd1, %rd6;\n\
         \x20   ld.global.f32 %f2, [%rd7];\n\
         \x20   fma.rn.f32    %f0, %f1, %f2, %f0;\n\
         \x20   add.u32       %r9, %r9, 1;\n\
         \x20   bra           LINEAR_DOT;\n\n",
    );

    // Add bias: acc += bias[col]
    ptx.push_str(
        "LINEAR_BIAS:\n\
         \x20   mul.wide.u32  %rd8, %r8, 4;\n\
         \x20   add.u64       %rd9, %rd2, %rd8;\n\
         \x20   ld.global.f32 %f3, [%rd9];\n\
         \x20   add.f32       %f0, %f0, %f3;\n\n",
    );

    // Store output[row * out_f + col]
    ptx.push_str(
        "\x20   // Store output\n\
         \x20   mad.lo.u32    %r12, %r7, %r2, %r8;\n\
         \x20   mul.wide.u32  %rd10, %r12, 4;\n\
         \x20   add.u64       %rd11, %rd3, %rd10;\n\
         \x20   st.global.f32 [%rd11], %f0;\n\
         LINEAR_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// PTX generation: Linear without bias
// ---------------------------------------------------------------------------

/// Generate PTX for Linear without bias: `Y = X * W`.
///
/// Same as [`generate_linear_ptx`] but without the bias addition step.
///
/// # Layout
///
/// - `input`: `[batch, in_features]` row-major
/// - `weight`: `[in_features, out_features]` row-major (pre-transposed)
/// - `output`: `[batch, out_features]` row-major
pub fn generate_linear_no_bias_ptx(in_features: u32, out_features: u32) -> String {
    let block = LINEAR_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude("sm_70"));
    ptx.push_str(&format!(
        "// Linear (no bias): Y[batch,{out_features}] = \
         X[batch,{in_features}] * W[{in_features},{out_features}]\n\
         // Block: {block}, 1D grid over batch * out_features\n\n"
    ));

    // Kernel entry: input, weight, output, batch_size, in_features, out_features
    ptx.push_str(
        ".visible .entry linear_no_bias_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_weight,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_batch,\n\
         \x20   .param .u32 param_in_features,\n\
         \x20   .param .u32 param_out_features\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block}\n{{\n"));

    ptx.push_str(
        "\x20   .reg .u32  %r<16>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_weight];\n\
         \x20   ld.param.u64  %rd2, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_batch];\n\
         \x20   ld.param.u32  %r1,  [param_in_features];\n\
         \x20   ld.param.u32  %r2,  [param_out_features];\n\n",
    );

    // Global thread index
    ptx.push_str(&format!(
        "\x20   mov.u32       %r3, %tid.x;\n\
         \x20   mov.u32       %r4, %ctaid.x;\n\
         \x20   mad.lo.u32    %r5, %r4, {block}, %r3;\n\n"
    ));

    // Bounds check
    ptx.push_str(
        "\x20   mul.lo.u32    %r6, %r0, %r2;\n\
         \x20   setp.ge.u32   %p0, %r5, %r6;\n\
         \x20   @%p0 bra      LNB_EXIT;\n\n",
    );

    // Decompose global_idx -> row, col
    ptx.push_str(
        "\x20   div.u32       %r7, %r5, %r2;\n\
         \x20   rem.u32       %r8, %r5, %r2;\n\n",
    );

    // Dot product loop
    ptx.push_str(&format!(
        "\x20   mov.f32       %f0, {zero};\n\
         \x20   mov.u32       %r9, 0;\n\n"
    ));

    ptx.push_str(
        "LNB_DOT:\n\
         \x20   setp.ge.u32   %p1, %r9, %r1;\n\
         \x20   @%p1 bra      LNB_STORE;\n\
         \x20   mad.lo.u32    %r10, %r7, %r1, %r9;\n\
         \x20   mul.wide.u32  %rd3, %r10, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;\n\
         \x20   ld.global.f32 %f1, [%rd4];\n\
         \x20   mad.lo.u32    %r11, %r9, %r2, %r8;\n\
         \x20   mul.wide.u32  %rd5, %r11, 4;\n\
         \x20   add.u64       %rd6, %rd1, %rd5;\n\
         \x20   ld.global.f32 %f2, [%rd6];\n\
         \x20   fma.rn.f32    %f0, %f1, %f2, %f0;\n\
         \x20   add.u32       %r9, %r9, 1;\n\
         \x20   bra           LNB_DOT;\n\n",
    );

    // Store
    ptx.push_str(
        "LNB_STORE:\n\
         \x20   mad.lo.u32    %r12, %r7, %r2, %r8;\n\
         \x20   mul.wide.u32  %rd7, %r12, 4;\n\
         \x20   add.u64       %rd8, %rd2, %rd7;\n\
         \x20   st.global.f32 [%rd8], %f0;\n\
         LNB_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// PTX generation: Linear + ReLU fusion
// ---------------------------------------------------------------------------

/// Generate PTX for fused Linear + ReLU: `Y = max(0, X * W + bias)`.
///
/// Fuses the bias addition and ReLU activation into the matmul output
/// store, eliminating one global memory round-trip compared to separate
/// Linear + ReLU kernels.
///
/// # Layout
///
/// Same as [`generate_linear_ptx`].
pub fn generate_linear_relu_ptx(in_features: u32, out_features: u32) -> String {
    let block = LINEAR_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude("sm_70"));
    ptx.push_str(&format!(
        "// Fused Linear+ReLU: Y[batch,{out_features}] = \
         max(0, X[batch,{in_features}] * W[{in_features},{out_features}] + b[{out_features}])\n\
         // Block: {block}, 1D grid over batch * out_features\n\n"
    ));

    ptx.push_str(
        ".visible .entry linear_relu_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_weight,\n\
         \x20   .param .u64 param_bias,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_batch,\n\
         \x20   .param .u32 param_in_features,\n\
         \x20   .param .u32 param_out_features\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block}\n{{\n"));

    ptx.push_str(
        "\x20   .reg .u32  %r<16>;\n\
         \x20   .reg .f32  %f<6>;\n\
         \x20   .reg .u64  %rd<12>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_weight];\n\
         \x20   ld.param.u64  %rd2, [param_bias];\n\
         \x20   ld.param.u64  %rd3, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_batch];\n\
         \x20   ld.param.u32  %r1,  [param_in_features];\n\
         \x20   ld.param.u32  %r2,  [param_out_features];\n\n",
    );

    // Global thread index
    ptx.push_str(&format!(
        "\x20   mov.u32       %r3, %tid.x;\n\
         \x20   mov.u32       %r4, %ctaid.x;\n\
         \x20   mad.lo.u32    %r5, %r4, {block}, %r3;\n\n"
    ));

    // Bounds check
    ptx.push_str(
        "\x20   mul.lo.u32    %r6, %r0, %r2;\n\
         \x20   setp.ge.u32   %p0, %r5, %r6;\n\
         \x20   @%p0 bra      LR_EXIT;\n\n",
    );

    // Decompose global_idx -> row, col
    ptx.push_str(
        "\x20   div.u32       %r7, %r5, %r2;\n\
         \x20   rem.u32       %r8, %r5, %r2;\n\n",
    );

    // Dot product loop
    ptx.push_str(&format!(
        "\x20   mov.f32       %f0, {zero};\n\
         \x20   mov.u32       %r9, 0;\n\n"
    ));

    ptx.push_str(
        "LR_DOT:\n\
         \x20   setp.ge.u32   %p1, %r9, %r1;\n\
         \x20   @%p1 bra      LR_BIAS;\n\
         \x20   mad.lo.u32    %r10, %r7, %r1, %r9;\n\
         \x20   mul.wide.u32  %rd4, %r10, 4;\n\
         \x20   add.u64       %rd5, %rd0, %rd4;\n\
         \x20   ld.global.f32 %f1, [%rd5];\n\
         \x20   mad.lo.u32    %r11, %r9, %r2, %r8;\n\
         \x20   mul.wide.u32  %rd6, %r11, 4;\n\
         \x20   add.u64       %rd7, %rd1, %rd6;\n\
         \x20   ld.global.f32 %f2, [%rd7];\n\
         \x20   fma.rn.f32    %f0, %f1, %f2, %f0;\n\
         \x20   add.u32       %r9, %r9, 1;\n\
         \x20   bra           LR_DOT;\n\n",
    );

    // Add bias
    ptx.push_str(
        "LR_BIAS:\n\
         \x20   mul.wide.u32  %rd8, %r8, 4;\n\
         \x20   add.u64       %rd9, %rd2, %rd8;\n\
         \x20   ld.global.f32 %f3, [%rd9];\n\
         \x20   add.f32       %f0, %f0, %f3;\n",
    );

    // ReLU: max(0.0, acc)
    ptx.push_str(&format!(
        "\x20   // ReLU: max(0, acc)\n\
         \x20   mov.f32       %f4, {zero};\n\
         \x20   max.f32       %f0, %f0, %f4;\n\n"
    ));

    // Store
    ptx.push_str(
        "\x20   mad.lo.u32    %r12, %r7, %r2, %r8;\n\
         \x20   mul.wide.u32  %rd10, %r12, 4;\n\
         \x20   add.u64       %rd11, %rd3, %rd10;\n\
         \x20   st.global.f32 [%rd11], %f0;\n\
         LR_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/// Compute Linear on CPU for reference/testing.
///
/// `output[b, j] = sum_i(input[b, i] * weight[i, j]) + bias[j]`
///
/// - `input`: `[batch, in_features]` row-major
/// - `weight`: `[in_features, out_features]` row-major (pre-transposed)
/// - `bias`: optional `[out_features]`
///
/// Returns `[batch, out_features]` row-major.
///
/// # Panics
///
/// Panics if dimensions are inconsistent.
pub fn linear_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    in_f: usize,
    out_f: usize,
) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        in_f * out_f,
        "weight must have in_f*out_f={} elements, got {}",
        in_f * out_f,
        weight.len()
    );
    assert_eq!(
        input.len() % in_f,
        0,
        "input length {} must be divisible by in_features={}",
        input.len(),
        in_f
    );
    if let Some(b) = bias {
        assert_eq!(
            b.len(),
            out_f,
            "bias must have out_f={} elements, got {}",
            out_f,
            b.len()
        );
    }

    let batch = input.len() / in_f;
    let mut output = vec![0.0f32; batch * out_f];

    for row in 0..batch {
        for col in 0..out_f {
            let mut sum = 0.0f32;
            for i in 0..in_f {
                sum += input[row * in_f + i] * weight[i * out_f + col];
            }
            if let Some(b) = bias {
                sum += b[col];
            }
            output[row * out_f + col] = sum;
        }
    }
    output
}

/// Compute the launch configuration for a Linear kernel.
///
/// Grid: `(ceil(batch * out_features / LINEAR_BLOCK_SIZE), 1, 1)`.
/// Block: `(LINEAR_BLOCK_SIZE, 1, 1)`.
#[must_use]
pub fn ptx_linear_launch_config(batch: usize, out_features: usize) -> ([usize; 3], [usize; 3]) {
    let total = batch * out_features;
    let block = LINEAR_BLOCK_SIZE as usize;
    let grid_x = total.div_ceil(block);
    ([grid_x, 1, 1], [block, 1, 1])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_linear_tests.rs"]
mod ptx_linear_tests;
