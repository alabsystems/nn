// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for 1D padding operations.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 1D padding.
//! Two modes are supported:
//!
//! ## Constant Padding
//!
//! Pads with a fixed value (typically 0.0):
//! - `Output[i] = pad_value`          if `i < pad_left`
//! - `Output[i] = Input[i - pad_left]` if `pad_left <= i < pad_left + n`
//! - `Output[i] = pad_value`          if `i >= pad_left + n`
//!
//! ## Reflect Padding
//!
//! Pads by reflecting the input signal (excluding the boundary element):
//! - `Output[pad_left - 1] = Input[1]`, `Output[pad_left - 2] = Input[2]`, etc.
//! - `Output[pad_left + n] = Input[n - 2]`, etc.
//!
//! ## Kernel interface
//!
//! Constant padding parameters:
//! - `param_input`    -- pointer to input tensor (f32), length `n`
//! - `param_output`   -- pointer to output tensor (f32), length `n + pad_left + pad_right`
//! - `param_n`        -- u32, number of input elements
//!
//! Reflect padding parameters (same layout):
//! - `param_input`    -- pointer to input tensor (f32), length `n`
//! - `param_output`   -- pointer to output tensor (f32), length `n + pad_left + pad_right`
//! - `param_n`        -- u32, number of input elements
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)`.
//! Grid: `(ceil(output_length / 256), 1, 1)`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for padding kernels (256 threads).
pub const PAD_BLOCK_SIZE: u32 = 256;

/// SM target for padding kernels.
const SM_TARGET: &str = "sm_70";

// ---------------------------------------------------------------------------
// Constant padding: PTX generation
// ---------------------------------------------------------------------------

/// Generate PTX for 1D constant padding.
///
/// Each thread computes one output element of the padded tensor:
/// - Elements in the left pad region are filled with `pad_value`.
/// - Elements in the right pad region are filled with `pad_value`.
/// - Elements in between are copied from the input.
///
/// # Arguments
/// * `n` -- number of input elements
/// * `pad_left` -- number of padding elements on the left
/// * `pad_right` -- number of padding elements on the right
/// * `pad_value` -- constant fill value for padded regions
///
/// # Example
/// ```
/// use nn_cuda::ptx_pad::generate_pad1d_ptx;
/// let ptx = generate_pad1d_ptx(10, 2, 3, 0.0);
/// assert!(ptx.contains(".entry ptx_pad1d_const_f32"));
/// ```
#[must_use]
pub fn generate_pad1d_ptx(n: u32, pad_left: u32, pad_right: u32, pad_value: f32) -> String {
    let output_len = n + pad_left + pad_right;
    let block_size = PAD_BLOCK_SIZE;
    let pv = format_ptx_float(pad_value);

    let mut ptx = String::with_capacity(2048);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Constant pad1d f32: n={n}, pad_left={pad_left}, pad_right={pad_right}, \
         pad_value={pad_value}, output_len={output_len}, block={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_pad1d_const_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<12>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Compute global index + grid stride
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;    // global_idx\n\
         \x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;          // grid_stride\n\n",
    );

    // Total output elements
    ptx.push_str(&format!(
        "\x20   mov.u32       %r7, {output_len};      // total output elements\n\n"
    ));

    // Grid-stride loop
    ptx.push_str(
        "PAD1D_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r7;\n\
         \x20   @%p0 bra      PAD1D_EXIT;\n\n",
    );

    // Check if in left pad region: i < pad_left
    ptx.push_str(&format!(
        "\x20   // Check left pad: i < pad_left\n\
         \x20   setp.lt.u32   %p1, %r4, {pad_left};\n\
         \x20   @%p1 bra      PAD1D_FILL;\n\n"
    ));

    // Check if in right pad region: i >= pad_left + n
    ptx.push_str(&format!(
        "\x20   // Check right pad: i >= pad_left + n\n\
         \x20   add.u32       %r8, %r0, {pad_left};   // pad_left + n\n\
         \x20   setp.ge.u32   %p2, %r4, %r8;\n\
         \x20   @%p2 bra      PAD1D_FILL;\n\n"
    ));

    // Copy from input: input_idx = i - pad_left
    ptx.push_str(&format!(
        "\x20   // Copy from input: input[i - pad_left]\n\
         \x20   sub.u32       %r9, %r4, {pad_left};   // input_idx\n\
         \x20   mul.wide.u32  %rd2, %r9, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   bra           PAD1D_STORE;\n\n"
    ));

    // Fill with pad_value
    ptx.push_str(&format!(
        "PAD1D_FILL:\n\
         \x20   mov.f32       %f0, {pv};\n\n"
    ));

    // Store output[i]
    ptx.push_str(
        "PAD1D_STORE:\n\
         \x20   mul.wide.u32  %rd4, %r4, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.f32 [%rd5], %f0;\n\n",
    );

    // Grid-stride advance
    ptx.push_str(
        "\x20   add.u32       %r4, %r4, %r6;\n\
         \x20   bra           PAD1D_LOOP;\n\n",
    );

    ptx.push_str(
        "PAD1D_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Reflect padding: PTX generation
// ---------------------------------------------------------------------------

/// Generate PTX for 1D reflect padding.
///
/// Reflect padding mirrors the input signal at the boundaries, excluding
/// the boundary element itself:
/// - Left pad:  `Output[pad_left - 1 - j] = Input[1 + j]` for `j = 0..pad_left-1`
/// - Right pad: `Output[pad_left + n + j] = Input[n - 2 - j]` for `j = 0..pad_right-1`
///
/// # Arguments
/// * `n` -- number of input elements (must be >= 2 for reflect padding)
/// * `pad_left` -- number of padding elements on the left (must be < n)
/// * `pad_right` -- number of padding elements on the right (must be < n)
///
/// # Example
/// ```
/// use nn_cuda::ptx_pad::generate_reflect_pad1d_ptx;
/// let ptx = generate_reflect_pad1d_ptx(10, 2, 3);
/// assert!(ptx.contains(".entry ptx_reflect_pad1d_f32"));
/// ```
#[must_use]
pub fn generate_reflect_pad1d_ptx(n: u32, pad_left: u32, pad_right: u32) -> String {
    let output_len = n + pad_left + pad_right;
    let block_size = PAD_BLOCK_SIZE;

    let mut ptx = String::with_capacity(3072);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Reflect pad1d f32: n={n}, pad_left={pad_left}, pad_right={pad_right}, \
         output_len={output_len}, block={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_reflect_pad1d_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<16>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Compute global index + grid stride
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;    // global_idx\n\
         \x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;          // grid_stride\n\n",
    );

    // Total output elements
    ptx.push_str(&format!(
        "\x20   mov.u32       %r7, {output_len};      // total output elements\n\n"
    ));

    // Grid-stride loop
    ptx.push_str(
        "RPAD1D_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r7;\n\
         \x20   @%p0 bra      RPAD1D_EXIT;\n\n",
    );

    // Check if in left pad region: i < pad_left
    // Reflect index: input_idx = pad_left - i  (for i in 0..pad_left)
    ptx.push_str(&format!(
        "\x20   // Check left pad: i < pad_left\n\
         \x20   setp.lt.u32   %p1, %r4, {pad_left};\n\
         \x20   @!%p1 bra     RPAD1D_CHECK_RIGHT;\n\
         \x20   // Left reflect: input_idx = pad_left - i\n\
         \x20   mov.u32       %r8, {pad_left};\n\
         \x20   sub.u32       %r9, %r8, %r4;          // pad_left - i\n\
         \x20   bra           RPAD1D_LOAD;\n\n"
    ));

    // Check if in right pad region: i >= pad_left + n
    // Reflect index: input_idx = n - 2 - (i - pad_left - n) = 2*n - 2 - (i - pad_left)
    ptx.push_str(&format!(
        "RPAD1D_CHECK_RIGHT:\n\
         \x20   add.u32       %r10, %r0, {pad_left};  // pad_left + n\n\
         \x20   setp.lt.u32   %p2, %r4, %r10;\n\
         \x20   @!%p2 bra     RPAD1D_RIGHT_REFLECT;\n\
         \x20   // Middle region: input_idx = i - pad_left\n\
         \x20   sub.u32       %r9, %r4, {pad_left};\n\
         \x20   bra           RPAD1D_LOAD;\n\n"
    ));

    // Right reflect region
    ptx.push_str(&format!(
        "RPAD1D_RIGHT_REFLECT:\n\
         \x20   // Right reflect: input_idx = 2*n - 2 - (i - pad_left)\n\
         \x20   sub.u32       %r11, %r4, {pad_left};  // i - pad_left\n\
         \x20   mul.lo.u32    %r12, %r0, 2;           // 2*n\n\
         \x20   sub.u32       %r12, %r12, 2;          // 2*n - 2\n\
         \x20   sub.u32       %r9, %r12, %r11;        // 2*n - 2 - (i - pad_left)\n\n"
    ));

    // Load input[input_idx] and store output[i]
    ptx.push_str(
        "RPAD1D_LOAD:\n\
         \x20   mul.wide.u32  %rd2, %r9, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   // Store output[i]\n\
         \x20   mul.wide.u32  %rd4, %r4, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.f32 [%rd5], %f0;\n\n",
    );

    // Grid-stride advance
    ptx.push_str(
        "\x20   add.u32       %r4, %r4, %r6;\n\
         \x20   bra           RPAD1D_LOOP;\n\n",
    );

    ptx.push_str(
        "RPAD1D_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

/// Reference implementation of 1D constant padding.
///
/// Output has length `input.len() + pad_left + pad_right`.
/// Padded positions are filled with `pad_value`.
#[must_use]
pub fn pad1d_reference(
    input: &[f32],
    pad_left: usize,
    pad_right: usize,
    pad_value: f32,
) -> Vec<f32> {
    let n = input.len();
    let output_len = n + pad_left + pad_right;
    let mut output = vec![pad_value; output_len];
    for (i, &val) in input.iter().enumerate() {
        output[pad_left + i] = val;
    }
    output
}

/// Reference implementation of 1D reflect padding.
///
/// Output has length `input.len() + pad_left + pad_right`.
/// Reflected elements exclude the boundary (e.g., reflecting `[a,b,c,d]`
/// left by 2 gives `[c, b, a, b, c, d, ...]`).
///
/// # Panics
///
/// Panics if `input.len() < 2` or `pad_left >= input.len()`
/// or `pad_right >= input.len()`.
#[must_use]
pub fn reflect_pad1d_reference(input: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let n = input.len();
    assert!(n >= 2, "reflect padding requires input length >= 2");
    assert!(
        pad_left < n,
        "pad_left ({pad_left}) must be < input length ({n})"
    );
    assert!(
        pad_right < n,
        "pad_right ({pad_right}) must be < input length ({n})"
    );

    let output_len = n + pad_left + pad_right;
    let mut output = Vec::with_capacity(output_len);

    // Left reflection: output position i (0..pad_left) maps to input[pad_left - i]
    for i in 0..pad_left {
        output.push(input[pad_left - i]);
    }

    // Middle (copy)
    output.extend_from_slice(input);

    // Right reflection
    for j in 0..pad_right {
        // reflect index: n - 2 - j
        output.push(input[n - 2 - j]);
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_pad_tests.rs"]
mod ptx_pad_tests;
