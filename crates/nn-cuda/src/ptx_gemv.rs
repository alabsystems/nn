// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for GEMV (matrix-vector), dot product, and outer product.
//!
//! Three related vector/matrix operations:
//!
//! - **GEMV:** `y[M] = A[M, N] @ x[N]` — matrix-vector multiply.
//!   Each thread block cooperatively loads `x` into shared memory, then
//!   each thread computes one element of `y` via a dot product of its row
//!   of `A` with the cached `x` vector.
//!
//! - **Dot product:** `result = sum(a[i] * b[i])` — parallel reduction in
//!   shared memory, yielding a scalar.
//!
//! - **Outer product:** `C[i, j] = a[i] * b[j]` — each thread computes one
//!   output element; no reduction required.
//!
//! All kernels emit raw PTX assembly (no CUDA C++), loadable via
//! `cuModuleLoadData` or assemblable via `ptxas`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for GEMV kernels: 256 threads.
///
/// One thread per output element in y. Shared memory: `N * 4` bytes
/// for the cached x vector (bounded by N, not block size).
pub const GEMV_BLOCK_SIZE: u32 = 256;

// ---------------------------------------------------------------------------
// GEMV: y = A @ x
// ---------------------------------------------------------------------------

/// Generate PTX for matrix-vector multiply: `y[M] = A[M, N] @ x[N]`.
///
/// Each thread computes one element of `y`. The `x` vector is cooperatively
/// loaded into shared memory by all threads in a block, then each thread
/// walks its row of `A` multiplied by the shared `x`.
///
/// # Arguments
///
/// * `m` — number of rows in A / length of y (for PTX comments only)
/// * `n` — number of columns in A / length of x (for PTX comments only)
///
/// Dimensions are passed as runtime kernel parameters.
///
/// # Thread configuration
///
/// Block: `(GEMV_BLOCK_SIZE, 1, 1)`. Grid: `(ceil(M / GEMV_BLOCK_SIZE), 1, 1)`.
#[must_use]
pub fn generate_gemv_ptx(m: u32, n: u32) -> String {
    let block = GEMV_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude("sm_70"));
    ptx.push_str(&format!(
        "// GEMV f32: y[{m}] = A[{m},{n}] @ x[{n}]\n\
         // Block: {block} threads, shared memory for x vector\n\n"
    ));

    // Shared memory for x vector (N elements, loaded cooperatively)
    ptx.push_str(&format!(".shared .align 4 .f32 xs[{n}];\n\n"));

    // Kernel entry
    ptx.push_str(
        ".visible .entry gemv_f32(\n\
         \x20   .param .u64 param_A,\n\
         \x20   .param .u64 param_x,\n\
         \x20   .param .u64 param_y,\n\
         \x20   .param .u32 param_M,\n\
         \x20   .param .u32 param_N\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block}\n{{\n"));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<16>;\n\
         \x20   .reg .f32  %f<6>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_A];\n\
         \x20   ld.param.u64  %rd1, [param_x];\n\
         \x20   ld.param.u64  %rd2, [param_y];\n\
         \x20   ld.param.u32  %r0,  [param_M];\n\
         \x20   ld.param.u32  %r1,  [param_N];\n\n",
    );

    // Compute global thread index: row = blockIdx.x * blockDim.x + threadIdx.x
    ptx.push_str(&format!(
        "\x20   // Thread index\n\
         \x20   mov.u32       %r2, %tid.x;\n\
         \x20   mov.u32       %r3, %ctaid.x;\n\
         \x20   mad.lo.u32    %r4, %r3, {block}, %r2;  // row = blockIdx.x * block + tid.x\n\n"
    ));

    // Cooperatively load x into shared memory
    // Each thread loads elements: i = tid.x, tid.x + blockDim.x, ...
    ptx.push_str(&format!(
        "\x20   // Cooperatively load x into shared memory\n\
         \x20   mov.u32       %r5, %r2;               // i = tid.x\n\
         LOAD_X_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r5, %r1;          // i >= N?\n\
         \x20   @%p0 bra      LOAD_X_DONE;\n\
         \x20   mul.wide.u32  %rd3, %r5, 4;            // byte offset\n\
         \x20   add.u64       %rd4, %rd1, %rd3;        // &x[i]\n\
         \x20   ld.global.f32 %f0, [%rd4];\n\
         \x20   mov.u64       %rd5, xs;                // shared mem base\n\
         \x20   add.u64       %rd6, %rd5, %rd3;        // &xs[i]\n\
         \x20   st.shared.f32 [%rd6], %f0;\n\
         \x20   add.u32       %r5, %r5, {block};       // i += blockDim.x\n\
         \x20   bra           LOAD_X_LOOP;\n\
         LOAD_X_DONE:\n\n"
    ));

    // Barrier after loading x
    ptx.push_str(
        "\x20   // Synchronize after x load\n\
         \x20   bar.sync      0;\n\n",
    );

    // Bounds check: row < M
    ptx.push_str(
        "\x20   // Bounds check\n\
         \x20   setp.ge.u32   %p1, %r4, %r0;          // row >= M?\n\
         \x20   @%p1 bra      GEMV_EXIT;\n\n",
    );

    // Compute dot product: y[row] = sum over j of A[row*N + j] * xs[j]
    ptx.push_str(&format!(
        "\x20   // Dot product: y[row] = A[row,:] . x\n\
         \x20   mov.f32       %f1, {zero};             // acc = 0.0\n\
         \x20   mov.u32       %r6, 0;                  // j = 0\n\
         GEMV_DOT:\n\
         \x20   setp.ge.u32   %p2, %r6, %r1;          // j >= N?\n\
         \x20   @%p2 bra      GEMV_STORE;\n\
         \x20   // Load A[row * N + j]\n\
         \x20   mad.lo.u32    %r7, %r4, %r1, %r6;     // row * N + j\n\
         \x20   mul.wide.u32  %rd3, %r7, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;\n\
         \x20   ld.global.f32 %f2, [%rd4];\n\
         \x20   // Load xs[j]\n\
         \x20   mul.wide.u32  %rd5, %r6, 4;\n\
         \x20   mov.u64       %rd6, xs;\n\
         \x20   add.u64       %rd7, %rd6, %rd5;\n\
         \x20   ld.shared.f32 %f3, [%rd7];\n\
         \x20   // acc += A[row,j] * x[j]\n\
         \x20   fma.rn.f32    %f1, %f2, %f3, %f1;\n\
         \x20   add.u32       %r6, %r6, 1;\n\
         \x20   bra           GEMV_DOT;\n\n"
    ));

    // Store y[row]
    ptx.push_str(
        "GEMV_STORE:\n\
         \x20   // Store y[row] = acc\n\
         \x20   mul.wide.u32  %rd3, %r4, 4;\n\
         \x20   add.u64       %rd4, %rd2, %rd3;\n\
         \x20   st.global.f32 [%rd4], %f1;\n\
         GEMV_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// CPU reference: `y[M] = A[M, N] @ x[N]`.
///
/// # Panics
///
/// Panics if `a.len() != m * n` or `x.len() != n`.
#[must_use]
pub fn gemv_reference(a: &[f32], x: &[f32], m: usize, n: usize) -> Vec<f32> {
    assert_eq!(
        a.len(),
        m * n,
        "A must have m*n={} elements, got {}",
        m * n,
        a.len()
    );
    assert_eq!(x.len(), n, "x must have n={n} elements, got {}", x.len());

    let mut y = vec![0.0f32; m];
    for row in 0..m {
        let mut sum = 0.0f32;
        for j in 0..n {
            sum += a[row * n + j] * x[j];
        }
        y[row] = sum;
    }
    y
}

// ---------------------------------------------------------------------------
// Dot product: result = sum(a[i] * b[i])
// ---------------------------------------------------------------------------

/// Generate PTX for dot product: `result = sum(a[i] * b[i])`.
///
/// Uses parallel reduction in shared memory. A single thread block
/// processes the entire vector, with each thread accumulating a partial
/// sum over strided elements, then reducing in shared memory via
/// sequential halving.
///
/// # Arguments
///
/// * `n` — vector length (for PTX comments only; runtime parameter)
///
/// # Thread configuration
///
/// Block: `(GEMV_BLOCK_SIZE, 1, 1)`. Grid: `(1, 1, 1)`.
#[must_use]
pub fn generate_dot_ptx(n: u32) -> String {
    let block = GEMV_BLOCK_SIZE;
    let zero = format_ptx_float(0.0);

    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude("sm_70"));
    ptx.push_str(&format!(
        "// Dot product f32: result = sum(a[i] * b[i]), N={n}\n\
         // Block: {block} threads, shared memory reduction\n\n"
    ));

    // Shared memory for partial sums
    ptx.push_str(&format!(".shared .align 4 .f32 partial[{block}];\n\n"));

    // Kernel entry
    ptx.push_str(
        ".visible .entry dot_f32(\n\
         \x20   .param .u64 param_a,\n\
         \x20   .param .u64 param_b,\n\
         \x20   .param .u64 param_result,\n\
         \x20   .param .u32 param_N\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {block}\n{{\n"));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<16>;\n\
         \x20   .reg .f32  %f<6>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_a];\n\
         \x20   ld.param.u64  %rd1, [param_b];\n\
         \x20   ld.param.u64  %rd2, [param_result];\n\
         \x20   ld.param.u32  %r0,  [param_N];\n\n",
    );

    // Thread index
    ptx.push_str(
        "\x20   // Thread index\n\
         \x20   mov.u32       %r1, %tid.x;\n\n",
    );

    // Strided accumulation: each thread sums elements at i, i+blockDim.x, ...
    ptx.push_str(&format!(
        "\x20   // Strided partial sum\n\
         \x20   mov.f32       %f0, {zero};             // partial = 0.0\n\
         \x20   mov.u32       %r2, %r1;                // i = tid.x\n\
         DOT_ACC_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r2, %r0;          // i >= N?\n\
         \x20   @%p0 bra      DOT_ACC_DONE;\n\
         \x20   // Load a[i]\n\
         \x20   mul.wide.u32  %rd3, %r2, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;\n\
         \x20   ld.global.f32 %f1, [%rd4];\n\
         \x20   // Load b[i]\n\
         \x20   add.u64       %rd5, %rd1, %rd3;\n\
         \x20   ld.global.f32 %f2, [%rd5];\n\
         \x20   // partial += a[i] * b[i]\n\
         \x20   fma.rn.f32    %f0, %f1, %f2, %f0;\n\
         \x20   add.u32       %r2, %r2, {block};       // i += blockDim.x\n\
         \x20   bra           DOT_ACC_LOOP;\n\
         DOT_ACC_DONE:\n\n"
    ));

    // Store partial sum to shared memory
    ptx.push_str(
        "\x20   // Store partial sum to shared memory\n\
         \x20   mul.wide.u32  %rd3, %r1, 4;            // tid.x * 4\n\
         \x20   mov.u64       %rd4, partial;\n\
         \x20   add.u64       %rd5, %rd4, %rd3;\n\
         \x20   st.shared.f32 [%rd5], %f0;\n\
         \x20   bar.sync      0;\n\n",
    );

    // Tree reduction in shared memory
    // stride = blockDim.x / 2, stride /= 2, ...
    ptx.push_str(&format!(
        "\x20   // Tree reduction\n\
         \x20   mov.u32       %r3, {half_block};       // stride = block / 2\n\
         DOT_REDUCE:\n\
         \x20   setp.eq.u32   %p1, %r3, 0;            // stride == 0?\n\
         \x20   @%p1 bra      DOT_REDUCE_DONE;\n\
         \x20   setp.ge.u32   %p2, %r1, %r3;          // tid >= stride?\n\
         \x20   @%p2 bra      DOT_REDUCE_SKIP;\n\
         \x20   // partial[tid] += partial[tid + stride]\n\
         \x20   add.u32       %r4, %r1, %r3;           // tid + stride\n\
         \x20   mul.wide.u32  %rd3, %r1, 4;\n\
         \x20   mov.u64       %rd4, partial;\n\
         \x20   add.u64       %rd5, %rd4, %rd3;        // &partial[tid]\n\
         \x20   ld.shared.f32 %f3, [%rd5];\n\
         \x20   mul.wide.u32  %rd6, %r4, 4;\n\
         \x20   add.u64       %rd7, %rd4, %rd6;        // &partial[tid + stride]\n\
         \x20   ld.shared.f32 %f4, [%rd7];\n\
         \x20   add.f32       %f3, %f3, %f4;\n\
         \x20   st.shared.f32 [%rd5], %f3;\n\
         DOT_REDUCE_SKIP:\n\
         \x20   bar.sync      0;\n\
         \x20   shr.u32       %r3, %r3, 1;             // stride /= 2\n\
         \x20   bra           DOT_REDUCE;\n\
         DOT_REDUCE_DONE:\n\n",
        half_block = block / 2
    ));

    // Thread 0 writes result
    ptx.push_str(
        "\x20   // Thread 0 writes final result\n\
         \x20   setp.ne.u32   %p3, %r1, 0;\n\
         \x20   @%p3 bra      DOT_EXIT;\n\
         \x20   mov.u64       %rd3, partial;\n\
         \x20   ld.shared.f32 %f5, [%rd3];\n\
         \x20   st.global.f32 [%rd2], %f5;\n\
         DOT_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// CPU reference: `result = sum(a[i] * b[i])`.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_reference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "a and b must have equal length: {} vs {}",
        a.len(),
        b.len()
    );

    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ---------------------------------------------------------------------------
// Outer product: C[i, j] = a[i] * b[j]
// ---------------------------------------------------------------------------

/// Generate PTX for outer product: `C[M, N]` where `C[i, j] = a[i] * b[j]`.
///
/// Each thread computes one element of the output matrix. No reduction
/// or shared memory needed — pure elementwise.
///
/// # Arguments
///
/// * `m` — length of vector a / rows of C (for PTX comments only)
/// * `n` — length of vector b / columns of C (for PTX comments only)
///
/// # Thread configuration
///
/// Block: `(16, 16, 1)`. Grid: `(ceil(N/16), ceil(M/16), 1)`.
#[must_use]
pub fn generate_outer_ptx(m: u32, n: u32) -> String {
    let tile: u32 = 16;
    let mut ptx = String::with_capacity(4096);

    ptx.push_str(&ptx_prelude("sm_70"));
    ptx.push_str(&format!(
        "// Outer product f32: C[{m},{n}] = a[{m}] (x) b[{n}]\n\
         // Block: {tile}x{tile} threads\n\n"
    ));

    // Kernel entry
    ptx.push_str(
        ".visible .entry outer_f32(\n\
         \x20   .param .u64 param_a,\n\
         \x20   .param .u64 param_b,\n\
         \x20   .param .u64 param_C,\n\
         \x20   .param .u32 param_M,\n\
         \x20   .param .u32 param_N\n\
         )\n",
    );

    ptx.push_str(&format!(".reqntid {tile}, {tile}\n{{\n"));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<12>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_a];\n\
         \x20   ld.param.u64  %rd1, [param_b];\n\
         \x20   ld.param.u64  %rd2, [param_C];\n\
         \x20   ld.param.u32  %r0,  [param_M];\n\
         \x20   ld.param.u32  %r1,  [param_N];\n\n",
    );

    // Compute row and col
    ptx.push_str(&format!(
        "\x20   // Thread indices\n\
         \x20   mov.u32       %r2, %tid.x;\n\
         \x20   mov.u32       %r3, %tid.y;\n\
         \x20   mov.u32       %r4, %ctaid.x;\n\
         \x20   mov.u32       %r5, %ctaid.y;\n\
         \x20   mad.lo.u32    %r6, %r5, {tile}, %r3;   // row = blockIdx.y * tile + tid.y\n\
         \x20   mad.lo.u32    %r7, %r4, {tile}, %r2;   // col = blockIdx.x * tile + tid.x\n\n"
    ));

    // Bounds check
    ptx.push_str(
        "\x20   // Bounds check\n\
         \x20   setp.ge.u32   %p0, %r6, %r0;          // row >= M?\n\
         \x20   setp.ge.u32   %p1, %r7, %r1;          // col >= N?\n\
         \x20   or.pred        %p2, %p0, %p1;\n\
         \x20   @%p2 bra      OUTER_EXIT;\n\n",
    );

    // Load a[row] and b[col]
    ptx.push_str(
        "\x20   // Load a[row]\n\
         \x20   mul.wide.u32  %rd3, %r6, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;\n\
         \x20   ld.global.f32 %f0, [%rd4];\n\
         \x20   // Load b[col]\n\
         \x20   mul.wide.u32  %rd5, %r7, 4;\n\
         \x20   add.u64       %rd6, %rd1, %rd5;\n\
         \x20   ld.global.f32 %f1, [%rd6];\n\n",
    );

    // Compute and store C[row, col] = a[row] * b[col]
    ptx.push_str(
        "\x20   // C[row, col] = a[row] * b[col]\n\
         \x20   mul.f32       %f2, %f0, %f1;\n\
         \x20   // Store C[row * N + col]\n\
         \x20   mad.lo.u32    %r8, %r6, %r1, %r7;     // row * N + col\n\
         \x20   mul.wide.u32  %rd7, %r8, 4;\n\
         \x20   add.u64       %rd8, %rd2, %rd7;\n\
         \x20   st.global.f32 [%rd8], %f2;\n\
         OUTER_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    ptx
}

/// CPU reference: outer product `C[i, j] = a[i] * b[j]`.
///
/// Returns row-major `C` of shape `[a.len(), b.len()]`.
#[must_use]
pub fn outer_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    let m = a.len();
    let n = b.len();
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] = a[i] * b[j];
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_gemv_tests.rs"]
mod ptx_gemv_tests;
