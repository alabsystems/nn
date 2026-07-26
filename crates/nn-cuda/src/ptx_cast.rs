// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for dtype casting kernels.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for type conversion
//! between f32, f16, and bf16. These are fundamental operations for mixed-precision
//! inference and training pipelines, where model weights and activations may be
//! stored in reduced precision for memory/bandwidth savings but computed in f32.
//!
//! ## Supported conversions
//!
//! | Source | Target | PTX instruction            |
//! |--------|--------|----------------------------|
//! | f32    | f16    | `cvt.rn.f16.f32`           |
//! | f16    | f32    | `cvt.f32.f16`              |
//! | f32    | bf16   | `cvt.rn.bf16.f32`          |
//! | bf16   | f32    | `cvt.f32.bf16`             |
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)` -- standard elementwise block size.
//! Grid: `(ceil(n / 256), 1, 1)` -- one thread per element.
//!
//! ## Kernel interface
//!
//! Each kernel takes:
//! - `param_input`  -- pointer to source tensor
//! - `param_output` -- pointer to destination tensor
//! - `param_n`      -- u32, number of elements

use crate::codegen_ptx::DEFAULT_SM_TARGET;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Block size for cast kernels (256 threads).
pub const CAST_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// PTX generation: f32 <-> f16
// ---------------------------------------------------------------------------

/// Generate PTX for f32 to f16 conversion.
///
/// Uses `cvt.rn.f16.f32` (round-to-nearest-even) for each element.
/// Input buffer is f32 (4 bytes/elem), output buffer is f16 (2 bytes/elem).
///
/// # Arguments
///
/// * `n` -- number of elements to convert
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_cast::generate_f32_to_f16_ptx;
/// let ptx = generate_f32_to_f16_ptx(1024);
/// assert!(ptx.contains("cvt.rn.f16.f32"));
/// ```
pub fn generate_f32_to_f16_ptx(n: u32) -> String {
    let block_size = CAST_BLOCK_SIZE;

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&format!(
        ".version 7.0\n.target {DEFAULT_SM_TARGET}\n.address_size 64\n\n"
    ));
    ptx.push_str(&format!(
        "// Cast f32 -> f16: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry cast_f32_to_f16(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .b16  %h<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str(&format!(
        "\x20   // Compute global index\n\
         \x20   mov.u32       %r1, %ctaid.x;\n\
         \x20   mov.u32       %r2, %tid.x;\n\
         \x20   mad.lo.u32    %r3, %r1, {block_size}, %r2;\n\n\
         \x20   // Bounds check\n\
         \x20   setp.ge.u32   %p0, %r3, %r0;\n\
         \x20   @%p0 bra      CAST_F32_F16_EXIT;\n\n"
    ));

    ptx.push_str(
        "\x20   // Load f32 input[idx]\n\
         \x20   mul.wide.u32  %rd2, %r3, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\n\
         \x20   // Convert f32 -> f16 (round to nearest even)\n\
         \x20   cvt.rn.f16.f32 %h0, %f0;\n\n\
         \x20   // Store f16 output[idx]\n\
         \x20   mul.wide.u32  %rd4, %r3, 2;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.b16 [%rd5], %h0;\n\n",
    );

    ptx.push_str(
        "CAST_F32_F16_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// Generate PTX for f16 to f32 conversion.
///
/// Uses `cvt.f32.f16` for each element.
/// Input buffer is f16 (2 bytes/elem), output buffer is f32 (4 bytes/elem).
///
/// # Arguments
///
/// * `n` -- number of elements to convert
pub fn generate_f16_to_f32_ptx(n: u32) -> String {
    let block_size = CAST_BLOCK_SIZE;

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&format!(
        ".version 7.0\n.target {DEFAULT_SM_TARGET}\n.address_size 64\n\n"
    ));
    ptx.push_str(&format!(
        "// Cast f16 -> f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry cast_f16_to_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .b16  %h<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str(&format!(
        "\x20   // Compute global index\n\
         \x20   mov.u32       %r1, %ctaid.x;\n\
         \x20   mov.u32       %r2, %tid.x;\n\
         \x20   mad.lo.u32    %r3, %r1, {block_size}, %r2;\n\n\
         \x20   // Bounds check\n\
         \x20   setp.ge.u32   %p0, %r3, %r0;\n\
         \x20   @%p0 bra      CAST_F16_F32_EXIT;\n\n"
    ));

    ptx.push_str(
        "\x20   // Load f16 input[idx]\n\
         \x20   mul.wide.u32  %rd2, %r3, 2;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.b16 %h0, [%rd3];\n\n\
         \x20   // Convert f16 -> f32\n\
         \x20   cvt.f32.f16   %f0, %h0;\n\n\
         \x20   // Store f32 output[idx]\n\
         \x20   mul.wide.u32  %rd4, %r3, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.f32 [%rd5], %f0;\n\n",
    );

    ptx.push_str(
        "CAST_F16_F32_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// PTX generation: f32 <-> bf16
// ---------------------------------------------------------------------------

/// Generate PTX for f32 to bf16 conversion.
///
/// Uses `cvt.rn.bf16.f32` (round-to-nearest-even) for each element.
/// Input buffer is f32 (4 bytes/elem), output buffer is bf16 (2 bytes/elem).
///
/// # Arguments
///
/// * `n` -- number of elements to convert
pub fn generate_f32_to_bf16_ptx(n: u32) -> String {
    let block_size = CAST_BLOCK_SIZE;

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&format!(
        ".version 7.0\n.target {DEFAULT_SM_TARGET}\n.address_size 64\n\n"
    ));
    ptx.push_str(&format!(
        "// Cast f32 -> bf16: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry cast_f32_to_bf16(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .b16  %h<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str(&format!(
        "\x20   // Compute global index\n\
         \x20   mov.u32       %r1, %ctaid.x;\n\
         \x20   mov.u32       %r2, %tid.x;\n\
         \x20   mad.lo.u32    %r3, %r1, {block_size}, %r2;\n\n\
         \x20   // Bounds check\n\
         \x20   setp.ge.u32   %p0, %r3, %r0;\n\
         \x20   @%p0 bra      CAST_F32_BF16_EXIT;\n\n"
    ));

    ptx.push_str(
        "\x20   // Load f32 input[idx]\n\
         \x20   mul.wide.u32  %rd2, %r3, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\n\
         \x20   // Convert f32 -> bf16 (round to nearest even)\n\
         \x20   cvt.rn.bf16.f32 %h0, %f0;\n\n\
         \x20   // Store bf16 output[idx]\n\
         \x20   mul.wide.u32  %rd4, %r3, 2;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.b16 [%rd5], %h0;\n\n",
    );

    ptx.push_str(
        "CAST_F32_BF16_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// Generate PTX for bf16 to f32 conversion.
///
/// Uses `cvt.f32.bf16` for each element.
/// Input buffer is bf16 (2 bytes/elem), output buffer is f32 (4 bytes/elem).
///
/// # Arguments
///
/// * `n` -- number of elements to convert
pub fn generate_bf16_to_f32_ptx(n: u32) -> String {
    let block_size = CAST_BLOCK_SIZE;

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&format!(
        ".version 7.0\n.target {DEFAULT_SM_TARGET}\n.address_size 64\n\n"
    ));
    ptx.push_str(&format!(
        "// Cast bf16 -> f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry cast_bf16_to_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .b16  %h<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str(&format!(
        "\x20   // Compute global index\n\
         \x20   mov.u32       %r1, %ctaid.x;\n\
         \x20   mov.u32       %r2, %tid.x;\n\
         \x20   mad.lo.u32    %r3, %r1, {block_size}, %r2;\n\n\
         \x20   // Bounds check\n\
         \x20   setp.ge.u32   %p0, %r3, %r0;\n\
         \x20   @%p0 bra      CAST_BF16_F32_EXIT;\n\n"
    ));

    ptx.push_str(
        "\x20   // Load bf16 input[idx]\n\
         \x20   mul.wide.u32  %rd2, %r3, 2;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.b16 %h0, [%rd3];\n\n\
         \x20   // Convert bf16 -> f32\n\
         \x20   cvt.f32.bf16  %f0, %h0;\n\n\
         \x20   // Store f32 output[idx]\n\
         \x20   mul.wide.u32  %rd4, %r3, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.f32 [%rd5], %f0;\n\n",
    );

    ptx.push_str(
        "CAST_BF16_F32_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_cast_tests.rs"]
mod ptx_cast_tests;
