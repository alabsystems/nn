// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for reduction operations (sum, max, mean, argmax, argmin).
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 reductions
//! over a contiguous 1-D buffer. Each kernel uses a tree-style reduction within
//! a single thread block using shared memory.
//!
//! ## Algorithm
//!
//! Two phases:
//! 1. **Grid-stride accumulate** -- each thread accumulates a partial result
//!    across multiple elements (if n > block_size).
//! 2. **Shared-memory tree reduction** -- partial results are reduced in shared
//!    memory with sequential addressing (no bank conflicts), halving the active
//!    thread count each step.
//!
//! ## Kernel interface
//!
//! Sum / Max / Mean parameters:
//! - `param_input`  -- pointer to input tensor (f32)
//! - `param_output` -- pointer to output scalar (f32)
//! - `param_n`      -- u32, total number of elements
//!
//! Argmax / Argmin parameters:
//! - `param_input`  -- pointer to input tensor (f32)
//! - `param_out_idx` -- pointer to output index (u32)
//! - `param_n`      -- u32, total number of elements
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)`.
//! Grid:  `(1, 1, 1)` -- single-block reduction (sufficient for n <= ~65 K in
//! production; multi-block cascading is a follow-up).

use crate::codegen_ptx::{format_ptx_float, ptx_prelude};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for reduction kernels (256 threads = 8 warps).
pub const REDUCE_BLOCK_SIZE: u32 = 256;

/// SM target for reduction kernels.
const SM_TARGET: &str = "sm_70";

// ---------------------------------------------------------------------------
// Sum reduction
// ---------------------------------------------------------------------------

/// Generate PTX for sum reduction: `output[0] = sum(input[0..n])`.
///
/// Single-block tree reduction in shared memory.
///
/// # Arguments
/// * `n` -- total number of elements
///
/// # Example
/// ```
/// use nn_cuda::ptx_reduce::generate_sum_ptx;
/// let ptx = generate_sum_ptx(1024);
/// assert!(ptx.contains(".entry ptx_sum_f32"));
/// assert!(ptx.contains("add.f32"));
/// ```
#[must_use]
pub fn generate_sum_ptx(n: u32) -> String {
    let block_size = REDUCE_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Sum reduction f32: n={n}, block_size={block_size}\n\n"
    ));

    // Shared memory for tree reduction
    ptx.push_str(&format!(".shared .align 4 .f32 smem[{block_size}];\n\n"));

    ptx.push_str(&format!(
        ".visible .entry ptx_sum_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<10>;\n\
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

    // Thread index
    ptx.push_str("\x20   mov.u32       %r1, %tid.x;           // tid\n\n");

    // Phase 1: grid-stride accumulation
    ptx.push_str(&format!(
        "\x20   // Phase 1: grid-stride partial sum\n\
         \x20   mov.f32       %f0, {zero};            // acc = 0.0\n\
         \x20   mov.u32       %r2, %r1;               // i = tid\n\
         SUM_ACCUM:\n\
         \x20   setp.ge.u32   %p0, %r2, %r0;          // i >= n?\n\
         \x20   @%p0 bra      SUM_STORE;\n\
         \x20   mul.wide.u32  %rd2, %r2, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f1, [%rd3];\n\
         \x20   add.f32       %f0, %f0, %f1;\n\
         \x20   add.u32       %r2, %r2, {block_size};\n\
         \x20   bra           SUM_ACCUM;\n\
         SUM_STORE:\n\n"
    ));

    // Store partial to shared memory
    ptx.push_str(
        "\x20   // Store partial to shared memory\n\
         \x20   mul.wide.u32  %rd4, %r1, 4;\n\
         \x20   mov.u64       %rd5, smem;\n\
         \x20   add.u64       %rd4, %rd5, %rd4;\n\
         \x20   st.shared.f32 [%rd4], %f0;\n\
         \x20   bar.sync      0;\n\n",
    );

    // Phase 2: tree reduction in shared memory
    emit_tree_reduce_sum(&mut ptx, block_size);

    // Thread 0 writes result
    ptx.push_str(
        "\x20   // Thread 0 writes result\n\
         \x20   setp.ne.u32   %p2, %r1, 0;\n\
         \x20   @%p2 bra      SUM_EXIT;\n\
         \x20   mov.u64       %rd6, smem;\n\
         \x20   ld.shared.f32 %f2, [%rd6];\n\
         \x20   st.global.f32 [%rd1], %f2;\n\
         SUM_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Max reduction
// ---------------------------------------------------------------------------

/// Generate PTX for max reduction: `output[0] = max(input[0..n])`.
///
/// # Arguments
/// * `n` -- total number of elements
///
/// # Example
/// ```
/// use nn_cuda::ptx_reduce::generate_max_ptx;
/// let ptx = generate_max_ptx(512);
/// assert!(ptx.contains(".entry ptx_max_f32"));
/// assert!(ptx.contains("max.f32"));
/// ```
#[must_use]
pub fn generate_max_ptx(n: u32) -> String {
    let block_size = REDUCE_BLOCK_SIZE;
    let neg_inf = format_ptx_float(f32::NEG_INFINITY);

    let mut ptx = String::with_capacity(4096);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Max reduction f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(".shared .align 4 .f32 smem[{block_size}];\n\n"));

    ptx.push_str(&format!(
        ".visible .entry ptx_max_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<10>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str("\x20   mov.u32       %r1, %tid.x;\n\n");

    // Phase 1: grid-stride max accumulation
    ptx.push_str(&format!(
        "\x20   // Phase 1: grid-stride partial max\n\
         \x20   mov.f32       %f0, {neg_inf};          // acc = -inf\n\
         \x20   mov.u32       %r2, %r1;               // i = tid\n\
         MAX_ACCUM:\n\
         \x20   setp.ge.u32   %p0, %r2, %r0;\n\
         \x20   @%p0 bra      MAX_STORE;\n\
         \x20   mul.wide.u32  %rd2, %r2, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f1, [%rd3];\n\
         \x20   max.f32       %f0, %f0, %f1;\n\
         \x20   add.u32       %r2, %r2, {block_size};\n\
         \x20   bra           MAX_ACCUM;\n\
         MAX_STORE:\n\n"
    ));

    // Store partial to shared memory
    ptx.push_str(
        "\x20   mul.wide.u32  %rd4, %r1, 4;\n\
         \x20   mov.u64       %rd5, smem;\n\
         \x20   add.u64       %rd4, %rd5, %rd4;\n\
         \x20   st.shared.f32 [%rd4], %f0;\n\
         \x20   bar.sync      0;\n\n",
    );

    // Phase 2: tree reduction
    emit_tree_reduce_max(&mut ptx, block_size);

    ptx.push_str(
        "\x20   setp.ne.u32   %p2, %r1, 0;\n\
         \x20   @%p2 bra      MAX_EXIT;\n\
         \x20   mov.u64       %rd6, smem;\n\
         \x20   ld.shared.f32 %f2, [%rd6];\n\
         \x20   st.global.f32 [%rd1], %f2;\n\
         MAX_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Mean reduction
// ---------------------------------------------------------------------------

/// Generate PTX for mean reduction: `output[0] = sum(input[0..n]) / n`.
///
/// Same as sum reduction but divides by n at the end.
///
/// # Arguments
/// * `n` -- total number of elements
///
/// # Example
/// ```
/// use nn_cuda::ptx_reduce::generate_mean_ptx;
/// let ptx = generate_mean_ptx(256);
/// assert!(ptx.contains(".entry ptx_mean_f32"));
/// ```
#[must_use]
pub fn generate_mean_ptx(n: u32) -> String {
    let block_size = REDUCE_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Mean reduction f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(".shared .align 4 .f32 smem[{block_size}];\n\n"));

    ptx.push_str(&format!(
        ".visible .entry ptx_mean_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<10>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str("\x20   mov.u32       %r1, %tid.x;\n\n");

    // Phase 1: grid-stride sum
    ptx.push_str(&format!(
        "\x20   // Phase 1: grid-stride partial sum\n\
         \x20   mov.f32       %f0, {zero};\n\
         \x20   mov.u32       %r2, %r1;\n\
         MEAN_ACCUM:\n\
         \x20   setp.ge.u32   %p0, %r2, %r0;\n\
         \x20   @%p0 bra      MEAN_STORE;\n\
         \x20   mul.wide.u32  %rd2, %r2, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f1, [%rd3];\n\
         \x20   add.f32       %f0, %f0, %f1;\n\
         \x20   add.u32       %r2, %r2, {block_size};\n\
         \x20   bra           MEAN_ACCUM;\n\
         MEAN_STORE:\n\n"
    ));

    // Store partial to shared memory
    ptx.push_str(
        "\x20   mul.wide.u32  %rd4, %r1, 4;\n\
         \x20   mov.u64       %rd5, smem;\n\
         \x20   add.u64       %rd4, %rd5, %rd4;\n\
         \x20   st.shared.f32 [%rd4], %f0;\n\
         \x20   bar.sync      0;\n\n",
    );

    // Phase 2: tree reduction (sum)
    emit_tree_reduce_sum(&mut ptx, block_size);

    // Thread 0 divides by n and writes result
    ptx.push_str(
        "\x20   setp.ne.u32   %p2, %r1, 0;\n\
         \x20   @%p2 bra      MEAN_EXIT;\n\
         \x20   mov.u64       %rd6, smem;\n\
         \x20   ld.shared.f32 %f2, [%rd6];\n\
         \x20   cvt.rn.f32.u32 %f3, %r0;             // n_f = (float)n\n\
         \x20   div.approx.f32 %f2, %f2, %f3;         // sum / n\n\
         \x20   st.global.f32 [%rd1], %f2;\n\
         MEAN_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Argmax reduction
// ---------------------------------------------------------------------------

/// Generate PTX for argmax: `out_idx[0] = argmax(input[0..n])`.
///
/// Tracks both value and index. In shared memory, stores interleaved
/// `(value, index)` pairs for the tree reduction.
///
/// # Arguments
/// * `n` -- total number of elements
///
/// # Example
/// ```
/// use nn_cuda::ptx_reduce::generate_argmax_ptx;
/// let ptx = generate_argmax_ptx(512);
/// assert!(ptx.contains(".entry ptx_argmax_f32"));
/// ```
#[must_use]
pub fn generate_argmax_ptx(n: u32) -> String {
    emit_argext_ptx(n, "ptx_argmax_f32", "argmax", true)
}

// ---------------------------------------------------------------------------
// Argmin reduction
// ---------------------------------------------------------------------------

/// Generate PTX for argmin: `out_idx[0] = argmin(input[0..n])`.
///
/// # Arguments
/// * `n` -- total number of elements
///
/// # Example
/// ```
/// use nn_cuda::ptx_reduce::generate_argmin_ptx;
/// let ptx = generate_argmin_ptx(512);
/// assert!(ptx.contains(".entry ptx_argmin_f32"));
/// ```
#[must_use]
pub fn generate_argmin_ptx(n: u32) -> String {
    emit_argext_ptx(n, "ptx_argmin_f32", "argmin", false)
}

/// Emit PTX for argmax or argmin. `is_max=true` => argmax, else argmin.
fn emit_argext_ptx(n: u32, kernel_name: &str, op_name: &str, is_max: bool) -> String {
    let block_size = REDUCE_BLOCK_SIZE;
    let identity = if is_max {
        format_ptx_float(f32::NEG_INFINITY)
    } else {
        format_ptx_float(f32::INFINITY)
    };
    let cmp_op = if is_max { "gt" } else { "lt" };

    let mut ptx = String::with_capacity(4096);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// {op_name} reduction f32: n={n}, block_size={block_size}\n\n"
    ));

    // Shared memory: values[block_size] + indices[block_size]
    ptx.push_str(&format!(
        ".shared .align 4 .f32 smem_val[{block_size}];\n\
         .shared .align 4 .u32 smem_idx[{block_size}];\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry {kernel_name}(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_out_idx,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<12>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_out_idx];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str("\x20   mov.u32       %r1, %tid.x;           // tid\n\n");

    // Phase 1: grid-stride accumulation tracking value + index
    ptx.push_str(&format!(
        "\x20   // Phase 1: grid-stride partial {op_name}\n\
         \x20   mov.f32       %f0, {identity};         // best_val = identity\n\
         \x20   mov.u32       %r3, 0;                  // best_idx = 0\n\
         \x20   mov.u32       %r2, %r1;               // i = tid\n\
         ARG_ACCUM:\n\
         \x20   setp.ge.u32   %p0, %r2, %r0;          // i >= n?\n\
         \x20   @%p0 bra      ARG_STORE;\n\
         \x20   mul.wide.u32  %rd2, %r2, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f1, [%rd3];\n\
         \x20   setp.{cmp_op}.f32 %p1, %f1, %f0;      // val better than best_val?\n\
         \x20   @!%p1 bra     ARG_SKIP;\n\
         \x20   mov.f32       %f0, %f1;               // best_val = val\n\
         \x20   mov.u32       %r3, %r2;               // best_idx = i\n\
         ARG_SKIP:\n\
         \x20   add.u32       %r2, %r2, {block_size};\n\
         \x20   bra           ARG_ACCUM;\n\
         ARG_STORE:\n\n"
    ));

    // Store partial val+idx to shared memory
    ptx.push_str(
        "\x20   // Store partial to shared memory\n\
         \x20   mul.wide.u32  %rd4, %r1, 4;\n\
         \x20   mov.u64       %rd5, smem_val;\n\
         \x20   add.u64       %rd6, %rd5, %rd4;\n\
         \x20   st.shared.f32 [%rd6], %f0;\n\
         \x20   mov.u64       %rd7, smem_idx;\n\
         \x20   add.u64       %rd8, %rd7, %rd4;\n\
         \x20   st.shared.u32 [%rd8], %r3;\n\
         \x20   bar.sync      0;\n\n",
    );

    // Phase 2: tree reduction in shared memory
    emit_tree_reduce_argext(&mut ptx, block_size, cmp_op);

    // Thread 0 writes result index
    ptx.push_str(
        "\x20   setp.ne.u32   %p2, %r1, 0;\n\
         \x20   @%p2 bra      ARG_EXIT;\n\
         \x20   mov.u64       %rd5, smem_idx;\n\
         \x20   ld.shared.u32 %r4, [%rd5];\n\
         \x20   st.global.u32 [%rd1], %r4;\n\
         ARG_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

// ---------------------------------------------------------------------------
// Tree reduction helpers
// ---------------------------------------------------------------------------

/// Emit shared-memory tree reduction for sum (add.f32).
fn emit_tree_reduce_sum(ptx: &mut String, block_size: u32) {
    ptx.push_str("\x20   // Phase 2: tree reduction (sum) in shared memory\n");
    let mut stride = block_size / 2;
    let mut step = 0u32;
    while stride > 0 {
        ptx.push_str(&format!(
            "\x20   setp.lt.u32   %p1, %r1, {stride};    // tid < stride?\n\
             \x20   @!%p1 bra     SUM_SYNC_{step};\n\
             \x20   // Load smem[tid] and smem[tid + stride]\n\
             \x20   mul.wide.u32  %rd4, %r1, 4;\n\
             \x20   mov.u64       %rd5, smem;\n\
             \x20   add.u64       %rd4, %rd5, %rd4;\n\
             \x20   ld.shared.f32 %f2, [%rd4];\n\
             \x20   mov.u32       %r8, %r1;\n\
             \x20   add.u32       %r8, %r8, {stride};\n\
             \x20   mul.wide.u32  %rd6, %r8, 4;\n\
             \x20   add.u64       %rd6, %rd5, %rd6;\n\
             \x20   ld.shared.f32 %f3, [%rd6];\n\
             \x20   add.f32       %f2, %f2, %f3;\n\
             \x20   st.shared.f32 [%rd4], %f2;\n\
             SUM_SYNC_{step}:\n\
             \x20   bar.sync      0;\n\n"
        ));
        stride /= 2;
        step += 1;
    }
}

/// Emit shared-memory tree reduction for max (max.f32).
fn emit_tree_reduce_max(ptx: &mut String, block_size: u32) {
    ptx.push_str("\x20   // Phase 2: tree reduction (max) in shared memory\n");
    let mut stride = block_size / 2;
    let mut step = 0u32;
    while stride > 0 {
        ptx.push_str(&format!(
            "\x20   setp.lt.u32   %p1, %r1, {stride};\n\
             \x20   @!%p1 bra     MAX_SYNC_{step};\n\
             \x20   mul.wide.u32  %rd4, %r1, 4;\n\
             \x20   mov.u64       %rd5, smem;\n\
             \x20   add.u64       %rd4, %rd5, %rd4;\n\
             \x20   ld.shared.f32 %f2, [%rd4];\n\
             \x20   mov.u32       %r8, %r1;\n\
             \x20   add.u32       %r8, %r8, {stride};\n\
             \x20   mul.wide.u32  %rd6, %r8, 4;\n\
             \x20   add.u64       %rd6, %rd5, %rd6;\n\
             \x20   ld.shared.f32 %f3, [%rd6];\n\
             \x20   max.f32       %f2, %f2, %f3;\n\
             \x20   st.shared.f32 [%rd4], %f2;\n\
             MAX_SYNC_{step}:\n\
             \x20   bar.sync      0;\n\n"
        ));
        stride /= 2;
        step += 1;
    }
}

/// Emit shared-memory tree reduction for argmax/argmin (compare + index swap).
fn emit_tree_reduce_argext(ptx: &mut String, block_size: u32, cmp_op: &str) {
    ptx.push_str("\x20   // Phase 2: tree reduction (arg) in shared memory\n");
    let mut stride = block_size / 2;
    let mut step = 0u32;
    while stride > 0 {
        ptx.push_str(&format!(
            "\x20   setp.lt.u32   %p1, %r1, {stride};\n\
             \x20   @!%p1 bra     ARG_SYNC_{step};\n\
             \x20   // Load smem_val/idx[tid] and smem_val/idx[tid+stride]\n\
             \x20   mul.wide.u32  %rd4, %r1, 4;\n\
             \x20   mov.u64       %rd5, smem_val;\n\
             \x20   add.u64       %rd6, %rd5, %rd4;\n\
             \x20   ld.shared.f32 %f2, [%rd6];            // val_a = smem_val[tid]\n\
             \x20   mov.u64       %rd7, smem_idx;\n\
             \x20   add.u64       %rd8, %rd7, %rd4;\n\
             \x20   ld.shared.u32 %r4, [%rd8];            // idx_a = smem_idx[tid]\n\
             \x20   mov.u32       %r8, %r1;\n\
             \x20   add.u32       %r8, %r8, {stride};\n\
             \x20   mul.wide.u32  %rd9, %r8, 4;\n\
             \x20   add.u64       %rd5, smem_val, %rd9;\n\
             \x20   ld.shared.f32 %f3, [%rd5];            // val_b = smem_val[tid+stride]\n\
             \x20   add.u64       %rd7, smem_idx, %rd9;\n\
             \x20   ld.shared.u32 %r5, [%rd7];            // idx_b = smem_idx[tid+stride]\n\
             \x20   // Compare: val_b {cmp_op} val_a?\n\
             \x20   setp.{cmp_op}.f32 %p3, %f3, %f2;\n\
             \x20   @!%p3 bra     ARG_NOSW_{step};\n\
             \x20   // Swap: write val_b, idx_b to tid slot\n\
             \x20   st.shared.f32 [%rd6], %f3;\n\
             \x20   st.shared.u32 [%rd8], %r5;\n\
             ARG_NOSW_{step}:\n\
             ARG_SYNC_{step}:\n\
             \x20   bar.sync      0;\n\n"
        ));
        stride /= 2;
        step += 1;
    }
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

/// Compute grid and block dimensions for a reduction kernel.
///
/// Single block reduction: Grid `(1, 1, 1)`, Block `(256, 1, 1)`.
///
/// # Returns
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_reduce_launch_config() -> ([usize; 3], [usize; 3]) {
    ([1, 1, 1], [REDUCE_BLOCK_SIZE as usize, 1, 1])
}

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

/// CPU reference: sum of all elements.
#[must_use]
pub fn sum_reference(input: &[f32]) -> f32 {
    input.iter().sum()
}

/// CPU reference: max of all elements. Returns `NEG_INFINITY` for empty input.
#[must_use]
pub fn max_reference(input: &[f32]) -> f32 {
    input.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// CPU reference: mean of all elements. Returns `0.0` for empty input.
#[must_use]
pub fn mean_reference(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let sum: f32 = input.iter().sum();
    sum / input.len() as f32
}

/// CPU reference: index of the maximum element. Returns `0` for empty input.
#[must_use]
pub fn argmax_reference(input: &[f32]) -> u32 {
    if input.is_empty() {
        return 0;
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in input.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// CPU reference: index of the minimum element. Returns `0` for empty input.
#[must_use]
pub fn argmin_reference(input: &[f32]) -> u32 {
    if input.is_empty() {
        return 0;
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::INFINITY;
    for (i, &v) in input.iter().enumerate() {
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Sum PTX structure --

    #[test]
    fn test_sum_ptx_contains_entry() {
        let ptx = generate_sum_ptx(1024);
        assert!(ptx.contains(".entry ptx_sum_f32"));
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".target sm_70"));
    }

    #[test]
    fn test_sum_ptx_contains_add_instruction() {
        let ptx = generate_sum_ptx(256);
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("bar.sync"));
    }

    #[test]
    fn test_sum_ptx_single_element() {
        let ptx = generate_sum_ptx(1);
        assert!(ptx.contains(".entry ptx_sum_f32"));
        // Should still produce valid PTX even for n=1
        assert!(ptx.contains("bar.sync"));
    }

    // -- Max PTX structure --

    #[test]
    fn test_max_ptx_contains_entry() {
        let ptx = generate_max_ptx(512);
        assert!(ptx.contains(".entry ptx_max_f32"));
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn test_max_ptx_single_element() {
        let ptx = generate_max_ptx(1);
        assert!(ptx.contains(".entry ptx_max_f32"));
    }

    // -- Mean PTX structure --

    #[test]
    fn test_mean_ptx_contains_entry() {
        let ptx = generate_mean_ptx(256);
        assert!(ptx.contains(".entry ptx_mean_f32"));
        assert!(ptx.contains("div.approx.f32"));
    }

    #[test]
    fn test_mean_ptx_has_cvt_for_n() {
        let ptx = generate_mean_ptx(128);
        assert!(ptx.contains("cvt.rn.f32.u32"));
    }

    // -- Argmax PTX structure --

    #[test]
    fn test_argmax_ptx_contains_entry() {
        let ptx = generate_argmax_ptx(512);
        assert!(ptx.contains(".entry ptx_argmax_f32"));
        assert!(ptx.contains("smem_val"));
        assert!(ptx.contains("smem_idx"));
    }

    #[test]
    fn test_argmax_ptx_uses_gt_compare() {
        let ptx = generate_argmax_ptx(256);
        assert!(ptx.contains("setp.gt.f32"));
    }

    // -- Argmin PTX structure --

    #[test]
    fn test_argmin_ptx_contains_entry() {
        let ptx = generate_argmin_ptx(512);
        assert!(ptx.contains(".entry ptx_argmin_f32"));
    }

    #[test]
    fn test_argmin_ptx_uses_lt_compare() {
        let ptx = generate_argmin_ptx(256);
        assert!(ptx.contains("setp.lt.f32"));
    }

    // -- Reference implementation tests --

    #[test]
    fn test_sum_reference_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((sum_reference(&data) - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_reference_single() {
        assert!((sum_reference(&[42.0]) - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_sum_reference_empty() {
        assert!((sum_reference(&[]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_reference_basic() {
        let data = vec![1.0, 5.0, 3.0, 2.0];
        assert!((max_reference(&data) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_reference_single() {
        assert!((max_reference(&[7.0]) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_reference_all_same() {
        let data = vec![3.0; 100];
        assert!((max_reference(&data) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_reference_negative() {
        let data = vec![-5.0, -2.0, -10.0, -1.0];
        assert!((max_reference(&data) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_mean_reference_basic() {
        let data = vec![2.0, 4.0, 6.0, 8.0];
        assert!((mean_reference(&data) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_reference_single() {
        assert!((mean_reference(&[3.0]) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_reference_empty() {
        assert!((mean_reference(&[]) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmax_reference_basic() {
        let data = vec![1.0, 5.0, 3.0, 2.0];
        assert_eq!(argmax_reference(&data), 1);
    }

    #[test]
    fn test_argmax_reference_first_element() {
        let data = vec![10.0, 5.0, 3.0];
        assert_eq!(argmax_reference(&data), 0);
    }

    #[test]
    fn test_argmax_reference_last_element() {
        let data = vec![1.0, 2.0, 3.0, 100.0];
        assert_eq!(argmax_reference(&data), 3);
    }

    #[test]
    fn test_argmax_reference_all_same() {
        let data = vec![5.0; 8];
        // First occurrence wins
        assert_eq!(argmax_reference(&data), 0);
    }

    #[test]
    fn test_argmax_reference_single() {
        assert_eq!(argmax_reference(&[42.0]), 0);
    }

    #[test]
    fn test_argmin_reference_basic() {
        let data = vec![3.0, 1.0, 5.0, 2.0];
        assert_eq!(argmin_reference(&data), 1);
    }

    #[test]
    fn test_argmin_reference_negative() {
        let data = vec![0.0, -5.0, 3.0, -2.0];
        assert_eq!(argmin_reference(&data), 1);
    }

    #[test]
    fn test_argmin_reference_all_same() {
        let data = vec![5.0; 8];
        assert_eq!(argmin_reference(&data), 0);
    }

    // -- Launch config --

    #[test]
    fn test_reduce_launch_config() {
        let (grid, block) = ptx_reduce_launch_config();
        assert_eq!(grid, [1, 1, 1]);
        assert_eq!(block, [256, 1, 1]);
    }

    // -- PTX version and target --

    #[test]
    fn test_all_reduce_kernels_target_sm_70() {
        for ptx in [
            generate_sum_ptx(64),
            generate_max_ptx(64),
            generate_mean_ptx(64),
            generate_argmax_ptx(64),
            generate_argmin_ptx(64),
        ] {
            assert!(ptx.contains(".target sm_70"), "Missing sm_70 target");
        }
    }
}
