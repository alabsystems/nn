// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for quantization kernels.
//!
//! Generates raw PTX for f32-to-int8 per-tensor quantization and the inverse
//! int8-to-f32 dequantization. These kernels implement the standard
//! affine quantization scheme:
//!
//! ```text
//! quantize:   q = clamp(round(x / scale) + zero_point, -128, 127)
//! dequantize: x = (q - zero_point) * scale
//! ```
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)` -- standard elementwise block size.
//! Grid: `(ceil(n / 256), 1, 1)` -- one thread per element.
//!
//! ## Kernel interfaces
//!
//! **`quantize_f32_to_int8`:**
//! - `param_input`  -- pointer to f32 source tensor
//! - `param_output` -- pointer to s8 destination tensor
//! - `param_n`      -- u32, number of elements
//!
//! Scale and zero_point are baked as PTX constants for maximum performance.
//!
//! **`dequantize_int8_to_f32`:**
//! - `param_input`  -- pointer to s8 source tensor
//! - `param_output` -- pointer to f32 destination tensor
//! - `param_n`      -- u32, number of elements

use crate::codegen_ptx::{format_ptx_float, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Block size for quantization kernels (256 threads).
pub const QUANTIZE_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// PTX generation: f32 -> int8 quantization
// ---------------------------------------------------------------------------

/// Generate PTX for f32-to-int8 per-tensor quantization.
///
/// Implements: `q = clamp(round(x / scale) + zero_point, -128, 127)`
///
/// Scale and zero_point are baked as immediate constants in the PTX.
/// The rounding mode is round-to-nearest-even (`cvt.rni`).
///
/// # Arguments
///
/// * `n` -- number of elements to quantize
/// * `scale` -- quantization scale factor (must be > 0)
/// * `zero_point` -- integer zero point offset
pub fn generate_quantize_f32_to_int8_ptx(n: u32, scale: f32, zero_point: i32) -> String {
    let block_size = QUANTIZE_BLOCK_SIZE;
    let inv_scale = format_ptx_float(1.0 / scale);
    let zp_float = format_ptx_float(zero_point as f32);
    let min_val = format_ptx_float(-128.0);
    let max_val = format_ptx_float(127.0);

    let mut ptx = String::with_capacity(4096);

    // Module header
    ptx.push_str(&format!(
        ".version 7.0\n.target {DEFAULT_SM_TARGET}\n.address_size 64\n\n"
    ));
    ptx.push_str(&format!(
        "// Quantize f32 -> int8: n={n}, scale={scale}, zero_point={zero_point}, \
         block_size={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry quantize_f32_to_int8(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Register declarations
    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<8>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\
         \x20   .reg .s32  %rs<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Global index
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;     // global_idx\n\
         \x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;           // grid_stride\n\n",
    );

    // Grid-stride loop
    ptx.push_str(
        "Q_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      Q_EXIT;\n\n",
    );

    // Load input[idx] as f32
    ptx.push_str(
        "\x20   mul.wide.u32  %rd2, %r4, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\n",
    );

    // Compute: x * inv_scale
    ptx.push_str(&format!(
        "\x20   mov.f32       %f1, {inv_scale};        // 1.0 / scale\n\
         \x20   mul.rn.f32    %f2, %f0, %f1;           // x / scale\n\n"
    ));

    // Round to nearest integer
    ptx.push_str("\x20   cvt.rni.f32.f32 %f3, %f2;              // round to nearest int\n\n");

    // Add zero point
    ptx.push_str(&format!(
        "\x20   mov.f32       %f4, {zp_float};         // zero_point as float\n\
         \x20   add.f32       %f3, %f3, %f4;           // + zero_point\n\n"
    ));

    // Clamp to [-128, 127]
    ptx.push_str(&format!(
        "\x20   mov.f32       %f5, {min_val};           // -128.0\n\
         \x20   mov.f32       %f6, {max_val};           // 127.0\n\
         \x20   max.f32       %f3, %f3, %f5;           // clamp lower\n\
         \x20   min.f32       %f3, %f3, %f6;           // clamp upper\n\n"
    ));

    // Convert to s32 then store as s8
    ptx.push_str(
        "\x20   cvt.rni.s32.f32 %rs0, %f3;             // float -> s32\n\
         \x20   mul.wide.u32  %rd4, %r4, 1;            // byte offset (1 byte per s8)\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.s8  [%rd5], %rs0;            // store as s8\n\n",
    );

    // Grid-stride advance
    ptx.push_str(
        "\x20   add.u32       %r4, %r4, %r6;           // global_idx += grid_stride\n\
         \x20   bra           Q_LOOP;\n\n\
         Q_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// PTX generation: int8 -> f32 dequantization
// ---------------------------------------------------------------------------

/// Generate PTX for int8-to-f32 dequantization.
///
/// Implements: `x = (q - zero_point) * scale`
///
/// # Arguments
///
/// * `n` -- number of elements to dequantize
/// * `scale` -- quantization scale factor
/// * `zero_point` -- integer zero point offset
pub fn generate_dequantize_int8_to_f32_ptx(n: u32, scale: f32, zero_point: i32) -> String {
    let block_size = QUANTIZE_BLOCK_SIZE;
    let scale_ptx = format_ptx_float(scale);
    let zp_float = format_ptx_float(zero_point as f32);

    let mut ptx = String::with_capacity(4096);

    // Module header
    ptx.push_str(&format!(
        ".version 7.0\n.target {DEFAULT_SM_TARGET}\n.address_size 64\n\n"
    ));
    ptx.push_str(&format!(
        "// Dequantize int8 -> f32: n={n}, scale={scale}, zero_point={zero_point}, \
         block_size={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry dequantize_int8_to_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Register declarations
    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<6>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\
         \x20   .reg .s32  %rs<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Global index
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;     // global_idx\n\
         \x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;           // grid_stride\n\n",
    );

    // Grid-stride loop
    ptx.push_str(
        "DQ_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      DQ_EXIT;\n\n",
    );

    // Load input[idx] as s8, sign-extend to s32
    ptx.push_str(
        "\x20   mul.wide.u32  %rd2, %r4, 1;            // byte offset\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.s8  %rs0, [%rd3];            // load as s8 -> s32\n\n",
    );

    // Convert s32 -> f32
    ptx.push_str("\x20   cvt.rn.f32.s32 %f0, %rs0;              // q as float\n\n");

    // Subtract zero point
    ptx.push_str(&format!(
        "\x20   mov.f32       %f1, {zp_float};         // zero_point as float\n\
         \x20   sub.f32       %f2, %f0, %f1;           // q - zero_point\n\n"
    ));

    // Multiply by scale
    ptx.push_str(&format!(
        "\x20   mov.f32       %f3, {scale_ptx};        // scale\n\
         \x20   mul.rn.f32    %f4, %f2, %f3;           // (q - zero_point) * scale\n\n"
    ));

    // Store output[idx] as f32
    ptx.push_str(
        "\x20   mul.wide.u32  %rd4, %r4, 4;            // f32 byte offset\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.f32 [%rd5], %f4;\n\n",
    );

    // Grid-stride advance
    ptx.push_str(
        "\x20   add.u32       %r4, %r4, %r6;           // global_idx += grid_stride\n\
         \x20   bra           DQ_LOOP;\n\n\
         DQ_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

/// CPU reference for f32-to-int8 per-tensor quantization.
///
/// `q = clamp(round(x / scale) + zero_point, -128, 127)`
pub fn quantize_reference(input: &[f32], scale: f32, zero_point: i32) -> Vec<i8> {
    input
        .iter()
        .map(|&x| {
            let q = (x / scale).round() as i32 + zero_point;
            q.clamp(-128, 127) as i8
        })
        .collect()
}

/// CPU reference for int8-to-f32 dequantization.
///
/// `x = (q - zero_point) * scale`
pub fn dequantize_reference(input: &[i8], scale: f32, zero_point: i32) -> Vec<f32> {
    input
        .iter()
        .map(|&q| (i32::from(q) - zero_point) as f32 * scale)
        .collect()
}
