// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for tensor manipulation operations.
//!
//! Generates raw PTX for structural tensor ops that rearrange or fill data
//! without arithmetic transformation. Each kernel uses a grid-stride loop
//! over the output elements.
//!
//! ## Operations
//!
//! | Op      | Description                              | Parameters                    |
//! |---------|------------------------------------------|-------------------------------|
//! | Concat  | Concatenate two 1D tensors               | `param_a`, `param_b`, `param_output`, `param_n_a`, `param_n_b` |
//! | Slice   | Extract contiguous sub-range             | `param_input`, `param_output`, `param_start`, `param_len` |
//! | Repeat  | Repeat each element N times              | `param_input`, `param_output`, `param_n`, `param_repeats` |
//! | Fill    | Fill output with a constant value        | `param_output`, `param_n`, `param_value` |
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)`.
//! Grid: `(ceil(output_len / 256), 1, 1)`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for tensor manipulation kernels (256 threads).
pub const TENSOR_OPS_BLOCK_SIZE: u32 = 256;

/// SM target for tensor manipulation kernels.
const SM_TARGET: &str = "sm_70";

// ---------------------------------------------------------------------------
// Concat: concatenate two 1D tensors
// ---------------------------------------------------------------------------

/// Generate PTX for concatenating two 1D f32 tensors.
///
/// Output length is `n_a + n_b`. Thread `idx` reads from `a` if `idx < n_a`,
/// otherwise from `b[idx - n_a]`.
///
/// # Arguments
/// * `n_a` -- number of elements in tensor a
/// * `n_b` -- number of elements in tensor b
///
/// # Example
/// ```
/// use nn_cuda::ptx_tensor_ops::generate_concat_ptx;
/// let ptx = generate_concat_ptx(512, 256);
/// assert!(ptx.contains(".entry ptx_concat_f32"));
/// ```
#[must_use]
pub fn generate_concat_ptx(n_a: u32, n_b: u32) -> String {
    let block_size = TENSOR_OPS_BLOCK_SIZE;
    let total = n_a + n_b;
    let _ = total; // used in comment only

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Concat f32: n_a={n_a}, n_b={n_b}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_concat_f32(\n\
         \x20   .param .u64 param_a,\n\
         \x20   .param .u64 param_b,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n_a,\n\
         \x20   .param .u32 param_n_b\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<10>;\n\
         \x20   .reg .f32  %f<2>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_a];\n\
         \x20   ld.param.u64  %rd1, [param_b];\n\
         \x20   ld.param.u64  %rd2, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n_a];\n\
         \x20   ld.param.u32  %r1,  [param_n_b];\n\n",
    );

    // Total = n_a + n_b
    ptx.push_str("\x20   add.u32       %r2, %r0, %r1;          // total = n_a + n_b\n\n");

    // Global thread index
    ptx.push_str(
        "\x20   mov.u32       %r3, %tid.x;\n\
         \x20   mov.u32       %r4, %ctaid.x;\n\
         \x20   mov.u32       %r5, %ntid.x;\n\
         \x20   mad.lo.u32    %r6, %r4, %r5, %r3;     // idx\n\n",
    );

    // Grid-stride loop
    ptx.push_str(
        "\x20   mov.u32       %r7, %nctaid.x;\n\
         \x20   mul.lo.u32    %r8, %r7, %r5;           // grid_stride\n\
         CONCAT_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r6, %r2;           // idx >= total?\n\
         \x20   @%p0 bra      CONCAT_EXIT;\n\
         \x20   // Branch: idx < n_a -> read from a, else read from b\n\
         \x20   setp.lt.u32   %p1, %r6, %r0;           // idx < n_a?\n\
         \x20   @%p1 bra      CONCAT_FROM_A;\n\
         \x20   // Read from b[idx - n_a]\n\
         \x20   sub.u32       %r9, %r6, %r0;           // offset = idx - n_a\n\
         \x20   mul.wide.u32  %rd3, %r9, 4;\n\
         \x20   add.u64       %rd4, %rd1, %rd3;        // &b[offset]\n\
         \x20   ld.global.f32 %f0, [%rd4];\n\
         \x20   bra           CONCAT_STORE;\n\
         CONCAT_FROM_A:\n\
         \x20   mul.wide.u32  %rd3, %r6, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;        // &a[idx]\n\
         \x20   ld.global.f32 %f0, [%rd4];\n\
         CONCAT_STORE:\n\
         \x20   mul.wide.u32  %rd5, %r6, 4;\n\
         \x20   add.u64       %rd6, %rd2, %rd5;        // &output[idx]\n\
         \x20   st.global.f32 [%rd6], %f0;\n\
         \x20   add.u32       %r6, %r6, %r8;           // idx += grid_stride\n\
         \x20   bra           CONCAT_LOOP;\n\
         CONCAT_EXIT:\n\
         \x20   ret;\n\
         }}\n",
    );

    ptx
}

/// CPU reference for 1D tensor concatenation.
#[must_use]
pub fn concat_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    result.extend_from_slice(a);
    result.extend_from_slice(b);
    result
}

// ---------------------------------------------------------------------------
// Slice: extract a contiguous sub-range from a 1D tensor
// ---------------------------------------------------------------------------

/// Generate PTX for slicing a 1D f32 tensor.
///
/// Copies `len` elements starting at `start` from the input tensor to the
/// output tensor: `output[i] = input[start + i]` for `i in 0..len`.
///
/// # Arguments
/// * `n`     -- total number of elements in input (for documentation)
/// * `start` -- start index of the slice
/// * `len`   -- number of elements to extract
///
/// # Example
/// ```
/// use nn_cuda::ptx_tensor_ops::generate_slice_ptx;
/// let ptx = generate_slice_ptx(1024, 100, 200);
/// assert!(ptx.contains(".entry ptx_slice_f32"));
/// ```
#[must_use]
pub fn generate_slice_ptx(n: u32, start: u32, len: u32) -> String {
    let block_size = TENSOR_OPS_BLOCK_SIZE;

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Slice f32: n={n}, start={start}, len={len}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_slice_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_start,\n\
         \x20   .param .u32 param_len\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<2>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_start];\n\
         \x20   ld.param.u32  %r1,  [param_len];\n\n",
    );

    // Global thread index
    ptx.push_str(
        "\x20   mov.u32       %r2, %tid.x;\n\
         \x20   mov.u32       %r3, %ctaid.x;\n\
         \x20   mov.u32       %r4, %ntid.x;\n\
         \x20   mad.lo.u32    %r5, %r3, %r4, %r2;     // idx\n\n",
    );

    // Grid-stride loop
    ptx.push_str(
        "\x20   mov.u32       %r6, %nctaid.x;\n\
         \x20   mul.lo.u32    %r7, %r6, %r4;           // grid_stride\n\
         SLICE_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r5, %r1;           // idx >= len?\n\
         \x20   @%p0 bra      SLICE_EXIT;\n\
         \x20   // Load input[start + idx]\n\
         \x20   add.u32       %r2, %r0, %r5;           // src_idx = start + idx\n\
         \x20   mul.wide.u32  %rd2, %r2, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;        // &input[src_idx]\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   // Store output[idx]\n\
         \x20   mul.wide.u32  %rd4, %r5, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;        // &output[idx]\n\
         \x20   st.global.f32 [%rd5], %f0;\n\
         \x20   add.u32       %r5, %r5, %r7;           // idx += grid_stride\n\
         \x20   bra           SLICE_LOOP;\n\
         SLICE_EXIT:\n\
         \x20   ret;\n\
         }}\n",
    );

    ptx
}

/// CPU reference for 1D tensor slice.
#[must_use]
pub fn slice_reference(input: &[f32], start: usize, len: usize) -> Vec<f32> {
    input[start..start + len].to_vec()
}

// ---------------------------------------------------------------------------
// Repeat: repeat each element N times
// ---------------------------------------------------------------------------

/// Generate PTX for repeating each element of a 1D f32 tensor.
///
/// Output has `n * repeats` elements. Element `output[i] = input[i / repeats]`.
///
/// # Arguments
/// * `n`       -- number of elements in input
/// * `repeats` -- number of times to repeat each element
///
/// # Example
/// ```
/// use nn_cuda::ptx_tensor_ops::generate_repeat_ptx;
/// let ptx = generate_repeat_ptx(256, 4);
/// assert!(ptx.contains(".entry ptx_repeat_f32"));
/// ```
#[must_use]
pub fn generate_repeat_ptx(n: u32, repeats: u32) -> String {
    let block_size = TENSOR_OPS_BLOCK_SIZE;
    let total = n * repeats;
    let _ = total; // used in comment only

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Repeat f32: n={n}, repeats={repeats}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_repeat_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n,\n\
         \x20   .param .u32 param_repeats\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<10>;\n\
         \x20   .reg .f32  %f<2>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\
         \x20   ld.param.u32  %r1,  [param_repeats];\n\n",
    );

    // Compute total output length = n * repeats
    ptx.push_str("\x20   mul.lo.u32    %r2, %r0, %r1;           // total = n * repeats\n\n");

    // Global thread index
    ptx.push_str(
        "\x20   mov.u32       %r3, %tid.x;\n\
         \x20   mov.u32       %r4, %ctaid.x;\n\
         \x20   mov.u32       %r5, %ntid.x;\n\
         \x20   mad.lo.u32    %r6, %r4, %r5, %r3;     // idx\n\n",
    );

    // Grid-stride loop
    ptx.push_str(
        "\x20   mov.u32       %r7, %nctaid.x;\n\
         \x20   mul.lo.u32    %r8, %r7, %r5;           // grid_stride\n\
         REPEAT_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r6, %r2;           // idx >= total?\n\
         \x20   @%p0 bra      REPEAT_EXIT;\n\
         \x20   // src_idx = idx / repeats\n\
         \x20   div.u32       %r9, %r6, %r1;           // src_idx\n\
         \x20   mul.wide.u32  %rd2, %r9, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;        // &input[src_idx]\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   // Store output[idx]\n\
         \x20   mul.wide.u32  %rd4, %r6, 4;\n\
         \x20   add.u64       %rd5, %rd1, %rd4;        // &output[idx]\n\
         \x20   st.global.f32 [%rd5], %f0;\n\
         \x20   add.u32       %r6, %r6, %r8;           // idx += grid_stride\n\
         \x20   bra           REPEAT_LOOP;\n\
         REPEAT_EXIT:\n\
         \x20   ret;\n\
         }}\n",
    );

    ptx
}

/// CPU reference for repeating each element of a 1D tensor.
#[must_use]
pub fn repeat_reference(input: &[f32], repeats: usize) -> Vec<f32> {
    input
        .iter()
        .flat_map(|&x| std::iter::repeat_n(x, repeats))
        .collect()
}

// ---------------------------------------------------------------------------
// Fill: fill a tensor with a constant value
// ---------------------------------------------------------------------------

/// Generate PTX for filling a 1D f32 tensor with a constant value.
///
/// Every element `output[i] = value` for `i in 0..n`.
///
/// # Arguments
/// * `n`     -- number of output elements
/// * `value` -- constant fill value (encoded as IEEE 754 hex literal)
///
/// # Example
/// ```
/// use nn_cuda::ptx_tensor_ops::generate_fill_ptx;
/// let ptx = generate_fill_ptx(1024, 0.0);
/// assert!(ptx.contains(".entry ptx_fill_f32"));
/// ```
#[must_use]
pub fn generate_fill_ptx(n: u32, value: f32) -> String {
    let block_size = TENSOR_OPS_BLOCK_SIZE;
    let value_hex = format_ptx_float(value);

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Fill f32: n={n}, value={value} ({value_hex}), block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_fill_f32(\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n,\n\
         \x20   .param .f32 param_value\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<2>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\
         \x20   ld.param.f32  %f0,  [param_value];\n\n",
    );

    // Global thread index
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;     // idx\n\n",
    );

    // Grid-stride loop
    ptx.push_str("\x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;           // grid_stride\n\
         FILL_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;           // idx >= n?\n\
         \x20   @%p0 bra      FILL_EXIT;\n\
         \x20   // Store value at output[idx]\n\
         \x20   mul.wide.u32  %rd1, %r4, 4;\n\
         \x20   add.u64       %rd2, %rd0, %rd1;        // &output[idx]\n\
         \x20   st.global.f32 [%rd2], %f0;\n\
         \x20   add.u32       %r4, %r4, %r6;           // idx += grid_stride\n\
         \x20   bra           FILL_LOOP;\n\
         FILL_EXIT:\n\
         \x20   ret;\n\
         }\n");

    ptx
}

/// CPU reference for filling a tensor with a constant.
#[must_use]
pub fn fill_reference(n: usize, value: f32) -> Vec<f32> {
    vec![value; n]
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Compute grid and block dimensions for tensor manipulation kernels.
///
/// Grid: `(ceil(output_len / 256), 1, 1)`.
/// Block: `(256, 1, 1)`.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_tensor_ops_launch_config(output_len: u32) -> ([u32; 3], [u32; 3]) {
    let grid_x = output_len.div_ceil(TENSOR_OPS_BLOCK_SIZE);
    ([grid_x, 1, 1], [TENSOR_OPS_BLOCK_SIZE, 1, 1])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_tensor_ops_tests.rs"]
mod ptx_tensor_ops_tests;
