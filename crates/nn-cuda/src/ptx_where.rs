// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for conditional selection (`where`) and clamp
//! operations.
//!
//! ## Where (conditional select)
//!
//! Element-wise ternary: `output[i] = cond[i] ? a[i] : b[i]`
//!
//! Parameters:
//! - `param_cond`   -- pointer to condition tensor (u32, 0 = false, nonzero = true)
//! - `param_a`      -- pointer to "true" branch tensor (f32)
//! - `param_b`      -- pointer to "false" branch tensor (f32)
//! - `param_output` -- pointer to output tensor (f32)
//! - `param_n`      -- u32, total number of elements
//!
//! ## Clamp
//!
//! Element-wise clamp: `output[i] = min(max(input[i], min_val), max_val)`
//!
//! Parameters:
//! - `param_input`  -- pointer to input tensor (f32)
//! - `param_output` -- pointer to output tensor (f32)
//! - `param_n`      -- u32, total number of elements
//!
//! Min/max values are baked into the PTX as immediate constants.
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)`.
//! Grid: `(ceil(n / 256), 1, 1)`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for where/clamp kernels (256 threads).
pub const WHERE_BLOCK_SIZE: u32 = 256;

/// SM target for where/clamp kernels.
const SM_TARGET: &str = "sm_70";

// ---------------------------------------------------------------------------
// Where: PTX generation
// ---------------------------------------------------------------------------

/// Generate PTX assembly for element-wise conditional selection.
///
/// `output[i] = cond[i] ? a[i] : b[i]`
///
/// The condition buffer uses `u32` elements (0 = false, nonzero = true).
/// The `a`, `b`, and `output` buffers are `f32`.
#[must_use]
pub fn generate_where_ptx(n: u32) -> String {
    let block_size = WHERE_BLOCK_SIZE;
    let mut ptx = String::with_capacity(2048);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Elementwise where f32: n={n}, block_size={block_size}\n\
         // output[i] = cond[i] ? a[i] : b[i]\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_where_f32(\n\
         \x20   .param .u64 param_cond,\n\
         \x20   .param .u64 param_a,\n\
         \x20   .param .u64 param_b,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_cond];\n\
         \x20   ld.param.u64  %rd1, [param_a];\n\
         \x20   ld.param.u64  %rd2, [param_b];\n\
         \x20   ld.param.u64  %rd3, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Thread index: tid = blockIdx.x * blockDim.x + threadIdx.x
    ptx.push_str("\x20   mov.u32       %r1, %ctaid.x;\n\
         \x20   mov.u32       %r2, %ntid.x;\n\
         \x20   mad.lo.u32    %r3, %r1, %r2, %tid.x;\n\n");

    // Grid-stride loop
    ptx.push_str("\x20   // Grid-stride loop\n\
         \x20   mov.u32       %r4, %nctaid.x;\n\
         \x20   mul.lo.u32    %r5, %r4, %r2;       // stride = gridDim.x * blockDim.x\n\
         LOOP:\n\
         \x20   setp.ge.u32   %p0, %r3, %r0;       // if idx >= n, exit\n\
         \x20   @%p0 bra      DONE;\n\n");

    // Load condition, a, b
    ptx.push_str(
        "\x20   // Compute addresses\n\
         \x20   mul.wide.u32  %rd4, %r3, 4;         // byte offset (f32 = 4 bytes)\n\
         \x20   add.u64       %rd5, %rd0, %rd4;     // &cond[idx]\n\
         \x20   add.u64       %rd6, %rd1, %rd4;     // &a[idx]\n\
         \x20   add.u64       %rd7, %rd2, %rd4;     // &b[idx]\n\
         \x20   add.u64       %rd8, %rd3, %rd4;     // &output[idx]\n\n\
         \x20   // Load values\n\
         \x20   ld.global.u32 %r6, [%rd5];          // cond[idx]\n\
         \x20   ld.global.f32 %f0, [%rd6];          // a[idx]\n\
         \x20   ld.global.f32 %f1, [%rd7];          // b[idx]\n\n",
    );

    // Conditional select: if cond != 0 then a else b
    ptx.push_str(
        "\x20   // Select: cond ? a : b\n\
         \x20   setp.ne.u32   %p1, %r6, 0;\n\
         \x20   selp.f32      %f2, %f0, %f1, %p1;  // f2 = p1 ? f0 : f1\n\n\
         \x20   // Store result\n\
         \x20   st.global.f32 [%rd8], %f2;\n\n",
    );

    // Advance and loop
    ptx.push_str(
        "\x20   add.u32       %r3, %r3, %r5;       // idx += stride\n\
         \x20   bra           LOOP;\n\
         DONE:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Where: CPU reference
// ---------------------------------------------------------------------------

/// CPU reference for element-wise conditional selection.
///
/// `output[i] = condition[i] != 0 ? a[i] : b[i]`
///
/// # Panics
///
/// Panics if `condition`, `a`, and `b` do not have the same length.
#[must_use]
pub fn where_reference(condition: &[u32], a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(
        condition.len(),
        a.len(),
        "condition and a must have the same length"
    );
    assert_eq!(a.len(), b.len(), "a and b must have the same length");

    condition
        .iter()
        .zip(a.iter())
        .zip(b.iter())
        .map(|((&c, &av), &bv)| if c != 0 { av } else { bv })
        .collect()
}

// ---------------------------------------------------------------------------
// Clamp: PTX generation
// ---------------------------------------------------------------------------

/// Generate PTX assembly for element-wise clamp.
///
/// `output[i] = min(max(input[i], min_val), max_val)`
///
/// The `min_val` and `max_val` are baked into the PTX as immediate
/// float constants.
#[must_use]
pub fn generate_clamp_ptx(n: u32, min_val: f32, max_val: f32) -> String {
    let block_size = WHERE_BLOCK_SIZE;
    let min_str = format_ptx_float(min_val);
    let max_str = format_ptx_float(max_val);
    let mut ptx = String::with_capacity(2048);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Elementwise clamp f32: n={n}, block_size={block_size}\n\
         // output[i] = min(max(input[i], {min_val}), {max_val})\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry ptx_clamp_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Thread index
    ptx.push_str("\x20   mov.u32       %r1, %ctaid.x;\n\
         \x20   mov.u32       %r2, %ntid.x;\n\
         \x20   mad.lo.u32    %r3, %r1, %r2, %tid.x;\n\n");

    // Grid-stride loop
    ptx.push_str("\x20   // Grid-stride loop\n\
         \x20   mov.u32       %r4, %nctaid.x;\n\
         \x20   mul.lo.u32    %r5, %r4, %r2;\n\
         LOOP:\n\
         \x20   setp.ge.u32   %p0, %r3, %r0;\n\
         \x20   @%p0 bra      DONE;\n\n");

    // Load, clamp, store
    ptx.push_str(&format!(
        "\x20   mul.wide.u32  %rd2, %r3, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;     // &input[idx]\n\
         \x20   add.u64       %rd4, %rd1, %rd2;     // &output[idx]\n\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\n\
         \x20   // clamp: max(input, min_val) then min(result, max_val)\n\
         \x20   max.f32       %f1, %f0, {min_str};\n\
         \x20   min.f32       %f2, %f1, {max_str};\n\n\
         \x20   st.global.f32 [%rd4], %f2;\n\n"
    ));

    // Advance and loop
    ptx.push_str(
        "\x20   add.u32       %r3, %r3, %r5;\n\
         \x20   bra           LOOP;\n\
         DONE:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Clamp: CPU reference
// ---------------------------------------------------------------------------

/// CPU reference for element-wise clamp.
///
/// `output[i] = min(max(input[i], min_val), max_val)`
#[must_use]
pub fn clamp_reference(input: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
    input.iter().map(|&x| x.max(min_val).min(max_val)).collect()
}

// ---------------------------------------------------------------------------
// Launch config helper
// ---------------------------------------------------------------------------

/// Compute the CUDA launch configuration for where/clamp kernels.
///
/// Grid: `(ceil(n / block_size), 1, 1)`.
/// Block: `(block_size, 1, 1)`.
#[must_use]
pub fn ptx_where_launch_config(n: u32) -> crate::cuda_ffi::CudaLaunchConfig {
    use crate::cuda_ffi::{CudaDim3, CudaLaunchConfig};

    let grid_x = n.div_ceil(WHERE_BLOCK_SIZE);
    CudaLaunchConfig {
        grid: CudaDim3::d1(grid_x),
        block: CudaDim3::d1(WHERE_BLOCK_SIZE),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_where_tests.rs"]
mod tests;
