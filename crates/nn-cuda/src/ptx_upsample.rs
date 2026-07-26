// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for nearest-neighbor upsampling (1D and 2D).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 nearest
//! neighbor upsampling. Each thread computes one output element by reading
//! the corresponding input element at `floor(output_idx / scale)`.
//!
//! ## 1D Upsampling
//!
//! `Output[i] = Input[i / scale]` (integer division).
//! Output length = `n * scale`.
//!
//! Example: `[1, 2, 3]` with scale=2 becomes `[1, 1, 2, 2, 3, 3]`.
//!
//! ## 2D Upsampling
//!
//! For input `[H, W]` and scales `(scale_h, scale_w)`:
//! `Output[oh, ow] = Input[oh / scale_h, ow / scale_w]`.
//! Output shape = `[H * scale_h, W * scale_w]`.
//!
//! ## Kernel interface
//!
//! 1D parameters:
//! - `param_input`  -- pointer to input tensor (f32), length `n`
//! - `param_output` -- pointer to output tensor (f32), length `n * scale`
//! - `param_n`      -- u32, number of input elements
//!
//! 2D parameters:
//! - `param_input`  -- pointer to input tensor (f32), `[h, w]` row-major
//! - `param_output` -- pointer to output tensor (f32), `[h*scale_h, w*scale_w]`
//! - `param_h`      -- u32, input height
//! - `param_w`      -- u32, input width
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)`.
//! Grid: `(ceil(total_output_elements / 256), 1, 1)`.

use crate::codegen_ptx::ptx_prelude;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for upsample kernels (256 threads).
pub const UPSAMPLE_BLOCK_SIZE: u32 = 256;

/// SM target for upsample kernels.
const SM_TARGET: &str = "sm_70";

// ---------------------------------------------------------------------------
// 1D Nearest-Neighbor Upsample: PTX generation
// ---------------------------------------------------------------------------

/// Generate PTX for 1D nearest-neighbor upsampling.
///
/// Each output element `Output[i] = Input[i / scale]`.
/// Output length = `n * scale`.
///
/// # Arguments
/// * `n` -- number of input elements
/// * `scale` -- integer upsampling factor (must be >= 1)
///
/// # Example
/// ```
/// use nn_cuda::ptx_upsample::generate_upsample_nearest1d_ptx;
/// let ptx = generate_upsample_nearest1d_ptx(10, 2);
/// assert!(ptx.contains(".entry ptx_upsample_nearest1d_f32"));
/// ```
#[must_use]
pub fn generate_upsample_nearest1d_ptx(n: u32, scale: u32) -> String {
    let output_len = n * scale;
    let block_size = UPSAMPLE_BLOCK_SIZE;

    let mut ptx = String::with_capacity(2048);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Nearest 1D upsample f32: n={n}, scale={scale}, \
         output_len={output_len}, block={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_upsample_nearest1d_f32(\n\
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
         \x20   .reg .pred %p<2>;\n\n",
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
        "UP1D_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r7;\n\
         \x20   @%p0 bra      UP1D_EXIT;\n\n",
    );

    // Compute input index: input_idx = i / scale
    ptx.push_str(&format!(
        "\x20   // input_idx = i / scale\n\
         \x20   div.u32       %r8, %r4, {scale};\n\n"
    ));

    // Load input[input_idx]
    ptx.push_str(
        "\x20   mul.wide.u32  %rd2, %r8, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\n",
    );

    // Store output[i]
    ptx.push_str(
        "\x20   mul.wide.u32  %rd4, %r4, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.f32 [%rd5], %f0;\n\n",
    );

    // Grid-stride advance
    ptx.push_str(
        "\x20   add.u32       %r4, %r4, %r6;\n\
         \x20   bra           UP1D_LOOP;\n\n",
    );

    ptx.push_str(
        "UP1D_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// 2D Nearest-Neighbor Upsample: PTX generation
// ---------------------------------------------------------------------------

/// Generate PTX for 2D nearest-neighbor upsampling.
///
/// For input `[H, W]` and scales `(scale_h, scale_w)`:
/// `Output[oh, ow] = Input[oh / scale_h, ow / scale_w]`.
/// Output shape = `[H * scale_h, W * scale_w]`.
///
/// # Arguments
/// * `h` -- input height
/// * `w` -- input width
/// * `scale_h` -- vertical upsampling factor
/// * `scale_w` -- horizontal upsampling factor
///
/// # Example
/// ```
/// use nn_cuda::ptx_upsample::generate_upsample_nearest2d_ptx;
/// let ptx = generate_upsample_nearest2d_ptx(4, 4, 2, 2);
/// assert!(ptx.contains(".entry ptx_upsample_nearest2d_f32"));
/// ```
#[must_use]
pub fn generate_upsample_nearest2d_ptx(h: u32, w: u32, scale_h: u32, scale_w: u32) -> String {
    let out_h = h * scale_h;
    let out_w = w * scale_w;
    let output_len = out_h * out_w;
    let block_size = UPSAMPLE_BLOCK_SIZE;

    let mut ptx = String::with_capacity(2048);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Nearest 2D upsample f32: h={h}, w={w}, scale_h={scale_h}, scale_w={scale_w}, \
         out_h={out_h}, out_w={out_w}, output_len={output_len}, block={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_upsample_nearest2d_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_h,\n\
         \x20   .param .u32 param_w\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<20>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_h];\n\
         \x20   ld.param.u32  %r1,  [param_w];\n\n",
    );

    // Compute global index + grid stride
    ptx.push_str(
        "\x20   mov.u32       %r2, %tid.x;\n\
         \x20   mov.u32       %r3, %ctaid.x;\n\
         \x20   mov.u32       %r4, %ntid.x;\n\
         \x20   mad.lo.u32    %r5, %r3, %r4, %r2;    // global_idx\n\
         \x20   mov.u32       %r6, %nctaid.x;\n\
         \x20   mul.lo.u32    %r7, %r6, %r4;          // grid_stride\n\n",
    );

    // Total output elements
    ptx.push_str(&format!(
        "\x20   mov.u32       %r8, {output_len};      // total output elements\n\
         \x20   mov.u32       %r9, {out_w};            // output width\n\n"
    ));

    // Grid-stride loop
    ptx.push_str(
        "UP2D_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r5, %r8;\n\
         \x20   @%p0 bra      UP2D_EXIT;\n\n",
    );

    // Decompose output index into (oh, ow)
    // oh = global_idx / out_w, ow = global_idx % out_w
    ptx.push_str(
        "\x20   div.u32       %r10, %r5, %r9;         // oh = idx / out_w\n\
         \x20   rem.u32       %r11, %r5, %r9;         // ow = idx % out_w\n\n",
    );

    // Compute input indices: ih = oh / scale_h, iw = ow / scale_w
    ptx.push_str(&format!(
        "\x20   div.u32       %r12, %r10, {scale_h};  // ih = oh / scale_h\n\
         \x20   div.u32       %r13, %r11, {scale_w};  // iw = ow / scale_w\n\n"
    ));

    // Compute input offset: ih * w + iw
    ptx.push_str("\x20   mad.lo.u32    %r14, %r12, %r1, %r13;  // ih * w + iw\n\n");

    // Load input[ih, iw]
    ptx.push_str(
        "\x20   mul.wide.u32  %rd2, %r14, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\n",
    );

    // Store output[oh, ow]
    ptx.push_str(
        "\x20   mul.wide.u32  %rd4, %r5, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;\n\
         \x20   st.global.f32 [%rd5], %f0;\n\n",
    );

    // Grid-stride advance
    ptx.push_str(
        "\x20   add.u32       %r5, %r5, %r7;\n\
         \x20   bra           UP2D_LOOP;\n\n",
    );

    ptx.push_str(
        "UP2D_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

/// Reference implementation of 1D nearest-neighbor upsampling.
///
/// Each input element is repeated `scale` times.
/// Output length = `input.len() * scale`.
#[must_use]
pub fn upsample_nearest1d_reference(input: &[f32], scale: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len() * scale);
    for &val in input {
        for _ in 0..scale {
            output.push(val);
        }
    }
    output
}

/// Reference implementation of 2D nearest-neighbor upsampling.
///
/// Input is row-major `[h, w]`. Output is `[h * scale_h, w * scale_w]`.
/// Each input pixel is expanded to a `scale_h x scale_w` block.
#[must_use]
pub fn upsample_nearest2d_reference(
    input: &[f32],
    h: usize,
    w: usize,
    scale_h: usize,
    scale_w: usize,
) -> Vec<f32> {
    let out_h = h * scale_h;
    let out_w = w * scale_w;
    let mut output = Vec::with_capacity(out_h * out_w);
    for oh in 0..out_h {
        for ow in 0..out_w {
            let ih = oh / scale_h;
            let iw = ow / scale_w;
            output.push(input[ih * w + iw]);
        }
    }
    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_upsample_tests.rs"]
mod ptx_upsample_tests;
