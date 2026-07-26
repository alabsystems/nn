// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for elementwise binary and unary operations.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 elementwise
//! operations. Each kernel is a simple grid-stride loop over all elements --
//! no reduction or shared memory needed.
//!
//! ## Binary Operations
//!
//! | Op   | Formula       | PTX instruction |
//! |------|---------------|-----------------|
//! | Add  | `a_i + b_i`   | `add.f32`       |
//! | Sub  | `a_i - b_i`   | `sub.f32`       |
//! | Mul  | `a_i * b_i`   | `mul.f32`       |
//! | Div  | `a_i / b_i`   | `div.approx.f32`|
//!
//! ## Unary Operations
//!
//! | Op       | Formula       | PTX instruction(s)                     |
//! |----------|---------------|----------------------------------------|
//! | Exp      | `exp(x_i)`    | `mul.f32` (log2e prescale) + `ex2.approx.f32` |
//! | Log      | `log(x_i)`    | `lg2.approx.f32` + `mul.f32` (ln2 postscale) |
//! | Sqrt     | `sqrt(x_i)`   | `sqrt.approx.f32`                     |
//! | Neg      | `-x_i`        | `neg.f32`                              |
//!
//! ## Scalar Operations
//!
//! | Op         | Formula         | PTX instruction |
//! |------------|-----------------|-----------------|
//! | ScalarMul  | `x_i * scalar`  | `mul.f32`       |
//!
//! ## Kernel interface
//!
//! Binary ops parameters:
//! - `param_a`      -- pointer to first input tensor (f32)
//! - `param_b`      -- pointer to second input tensor (f32)
//! - `param_output` -- pointer to output tensor (f32)
//! - `param_n`      -- u32, total number of elements
//!
//! Unary ops parameters:
//! - `param_input`  -- pointer to input tensor (f32)
//! - `param_output` -- pointer to output tensor (f32)
//! - `param_n`      -- u32, total number of elements
//!
//! ScalarMul parameters:
//! - `param_input`  -- pointer to input tensor (f32)
//! - `param_output` -- pointer to output tensor (f32)
//! - `param_scalar` -- f32, the scalar multiplier
//! - `param_n`      -- u32, total number of elements
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)`.
//! Grid: `(ceil(n / 256), 1, 1)`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for elementwise kernels (256 threads).
pub const ELEMENTWISE_BLOCK_SIZE: u32 = 256;

/// SM target for elementwise kernels.
const SM_TARGET: &str = "sm_70";

/// log2(e) as f32 -- prescale factor for `ex2.approx.f32`.
const LOG2_E: f32 = std::f32::consts::LOG2_E;

/// ln(2) as f32 -- postscale factor for `lg2.approx.f32`.
const LN_2: f32 = std::f32::consts::LN_2;

// ---------------------------------------------------------------------------
// Binary ops: Add, Sub, Mul, Div
// ---------------------------------------------------------------------------

/// Emit a PTX kernel for elementwise binary operation.
///
/// The `op_name` is used in comments and the kernel name.
/// The `ptx_instruction` is the PTX instruction to apply (e.g., `add.f32`).
fn emit_binary_op_ptx(n: u32, kernel_name: &str, op_name: &str, ptx_instruction: &str) -> String {
    let block_size = ELEMENTWISE_BLOCK_SIZE;
    let mut ptx = String::with_capacity(2048);

    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Elementwise {op_name} f32: n={n}, block_size={block_size}\n\n"
    ));

    // Kernel entry
    ptx.push_str(&format!(
        ".visible .entry {kernel_name}(\n\
         \x20   .param .u64 param_a,\n\
         \x20   .param .u64 param_b,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n"
    ));
    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_a];\n\
         \x20   ld.param.u64  %rd1, [param_b];\n\
         \x20   ld.param.u64  %rd2, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Compute global thread index: idx = blockIdx.x * blockDim.x + threadIdx.x
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;  // idx = blockIdx.x * blockDim.x + threadIdx.x\n\n",
    );

    // Grid-stride loop
    ptx.push_str(&format!(
        "\x20   // Grid-stride loop\n\
         \x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;        // grid_stride = gridDim.x * blockDim.x\n\
         EW_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;        // idx >= n?\n\
         \x20   @%p0 bra      EW_EXIT;\n\
         \x20   // Load a[idx] and b[idx]\n\
         \x20   mul.wide.u32  %rd3, %r4, 4;          // byte offset\n\
         \x20   add.u64       %rd4, %rd0, %rd3;      // &a[idx]\n\
         \x20   add.u64       %rd5, %rd1, %rd3;      // &b[idx]\n\
         \x20   ld.global.f32 %f0, [%rd4];\n\
         \x20   ld.global.f32 %f1, [%rd5];\n\
         \x20   // Compute result\n\
         \x20   {ptx_instruction} %f2, %f0, %f1;\n\
         \x20   // Store output[idx]\n\
         \x20   add.u64       %rd6, %rd2, %rd3;      // &output[idx]\n\
         \x20   st.global.f32 [%rd6], %f2;\n\
         \x20   add.u32       %r4, %r4, %r6;         // idx += grid_stride\n\
         \x20   bra           EW_LOOP;\n\
         EW_EXIT:\n\
         \x20   ret;\n\
         }}\n"
    ));

    ptx
}

/// Generate PTX for element-wise addition: `output[i] = a[i] + b[i]`.
///
/// # Arguments
/// * `n` -- total number of elements
///
/// # Example
/// ```
/// use nn_cuda::ptx_elementwise::generate_add_ptx;
/// let ptx = generate_add_ptx(1024);
/// assert!(ptx.contains(".entry ptx_add_f32"));
/// assert!(ptx.contains("add.f32"));
/// ```
#[must_use]
pub fn generate_add_ptx(n: u32) -> String {
    emit_binary_op_ptx(n, "ptx_add_f32", "add", "add.f32")
}

/// Generate PTX for element-wise subtraction: `output[i] = a[i] - b[i]`.
#[must_use]
pub fn generate_sub_ptx(n: u32) -> String {
    emit_binary_op_ptx(n, "ptx_sub_f32", "sub", "sub.f32")
}

/// Generate PTX for element-wise multiplication: `output[i] = a[i] * b[i]`.
#[must_use]
pub fn generate_mul_ptx(n: u32) -> String {
    emit_binary_op_ptx(n, "ptx_mul_f32", "mul", "mul.f32")
}

/// Generate PTX for element-wise division: `output[i] = a[i] / b[i]`.
#[must_use]
pub fn generate_div_ptx(n: u32) -> String {
    emit_binary_op_ptx(n, "ptx_div_f32", "div", "div.approx.f32")
}

// ---------------------------------------------------------------------------
// Unary ops: Exp, Log, Sqrt, Neg
// ---------------------------------------------------------------------------

/// Generate PTX for element-wise exp: `output[i] = exp(input[i])`.
///
/// Uses `ex2.approx.f32` with log2(e) prescale: `exp(x) = 2^(x * log2(e))`.
#[must_use]
pub fn generate_exp_ptx(n: u32) -> String {
    let block_size = ELEMENTWISE_BLOCK_SIZE;
    let log2e = format_ptx_float(LOG2_E);

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Elementwise exp f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_exp_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    // Global thread index
    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;\n\n",
    );

    // Grid-stride loop
    ptx.push_str(&format!(
        "\x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;\n\
         EXP_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      EXP_EXIT;\n\
         \x20   mul.wide.u32  %rd2, %r4, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   // exp(x) = 2^(x * log2(e))\n\
         \x20   mul.f32       %f1, %f0, {log2e};\n\
         \x20   ex2.approx.f32 %f2, %f1;\n\
         \x20   add.u64       %rd4, %rd1, %rd2;\n\
         \x20   st.global.f32 [%rd4], %f2;\n\
         \x20   add.u32       %r4, %r4, %r6;\n\
         \x20   bra           EXP_LOOP;\n\
         EXP_EXIT:\n\
         \x20   ret;\n\
         }}\n"
    ));

    ptx
}

/// Generate PTX for element-wise log: `output[i] = ln(input[i])`.
///
/// Uses `lg2.approx.f32` with ln(2) postscale: `ln(x) = lg2(x) * ln(2)`.
#[must_use]
pub fn generate_log_ptx(n: u32) -> String {
    let block_size = ELEMENTWISE_BLOCK_SIZE;
    let ln2 = format_ptx_float(LN_2);

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Elementwise log f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_log_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;\n\n",
    );

    ptx.push_str(&format!(
        "\x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;\n\
         LOG_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      LOG_EXIT;\n\
         \x20   mul.wide.u32  %rd2, %r4, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   // ln(x) = lg2(x) * ln(2)\n\
         \x20   lg2.approx.f32 %f1, %f0;\n\
         \x20   mul.f32       %f2, %f1, {ln2};\n\
         \x20   add.u64       %rd4, %rd1, %rd2;\n\
         \x20   st.global.f32 [%rd4], %f2;\n\
         \x20   add.u32       %r4, %r4, %r6;\n\
         \x20   bra           LOG_LOOP;\n\
         LOG_EXIT:\n\
         \x20   ret;\n\
         }}\n"
    ));

    ptx
}

/// Generate PTX for element-wise sqrt: `output[i] = sqrt(input[i])`.
#[must_use]
pub fn generate_sqrt_ptx(n: u32) -> String {
    let block_size = ELEMENTWISE_BLOCK_SIZE;

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Elementwise sqrt f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_sqrt_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;\n\n",
    );

    ptx.push_str("\x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;\n\
         SQRT_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      SQRT_EXIT;\n\
         \x20   mul.wide.u32  %rd2, %r4, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   sqrt.approx.f32 %f1, %f0;\n\
         \x20   add.u64       %rd4, %rd1, %rd2;\n\
         \x20   st.global.f32 [%rd4], %f1;\n\
         \x20   add.u32       %r4, %r4, %r6;\n\
         \x20   bra           SQRT_LOOP;\n\
         SQRT_EXIT:\n\
         \x20   ret;\n\
         }\n");

    ptx
}

/// Generate PTX for element-wise negation: `output[i] = -input[i]`.
#[must_use]
pub fn generate_neg_ptx(n: u32) -> String {
    let block_size = ELEMENTWISE_BLOCK_SIZE;

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Elementwise neg f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_neg_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;\n\n",
    );

    ptx.push_str("\x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;\n\
         NEG_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      NEG_EXIT;\n\
         \x20   mul.wide.u32  %rd2, %r4, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   neg.f32       %f1, %f0;\n\
         \x20   add.u64       %rd4, %rd1, %rd2;\n\
         \x20   st.global.f32 [%rd4], %f1;\n\
         \x20   add.u32       %r4, %r4, %r6;\n\
         \x20   bra           NEG_LOOP;\n\
         NEG_EXIT:\n\
         \x20   ret;\n\
         }\n");

    ptx
}

// ---------------------------------------------------------------------------
// Scalar ops
// ---------------------------------------------------------------------------

/// Generate PTX for scalar multiplication: `output[i] = input[i] * scalar`.
#[must_use]
pub fn generate_scalar_mul_ptx(n: u32) -> String {
    let block_size = ELEMENTWISE_BLOCK_SIZE;

    let mut ptx = String::with_capacity(2048);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Elementwise scalar_mul f32: n={n}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_scalar_mul_f32(\n\
         \x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .f32 param_scalar,\n\
         \x20   .param .u32 param_n\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    ptx.push_str(
        "\x20   .reg .u32  %r<8>;\n\
         \x20   .reg .f32  %f<4>;\n\
         \x20   .reg .u64  %rd<6>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.f32  %f3,  [param_scalar];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\n",
    );

    ptx.push_str(
        "\x20   mov.u32       %r1, %tid.x;\n\
         \x20   mov.u32       %r2, %ctaid.x;\n\
         \x20   mov.u32       %r3, %ntid.x;\n\
         \x20   mad.lo.u32    %r4, %r2, %r3, %r1;\n\n",
    );

    ptx.push_str("\x20   mov.u32       %r5, %nctaid.x;\n\
         \x20   mul.lo.u32    %r6, %r5, %r3;\n\
         SMUL_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r4, %r0;\n\
         \x20   @%p0 bra      SMUL_EXIT;\n\
         \x20   mul.wide.u32  %rd2, %r4, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;\n\
         \x20   ld.global.f32 %f0, [%rd3];\n\
         \x20   mul.f32       %f1, %f0, %f3;\n\
         \x20   add.u64       %rd4, %rd1, %rd2;\n\
         \x20   st.global.f32 [%rd4], %f1;\n\
         \x20   add.u32       %r4, %r4, %r6;\n\
         \x20   bra           SMUL_LOOP;\n\
         SMUL_EXIT:\n\
         \x20   ret;\n\
         }\n");

    ptx
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Compute grid and block dimensions for elementwise kernels.
///
/// Grid: `(ceil(n / 256), 1, 1)`.
/// Block: `(256, 1, 1)`.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_elementwise_launch_config(n: u32) -> ([u32; 3], [u32; 3]) {
    let grid_x = n.div_ceil(ELEMENTWISE_BLOCK_SIZE);
    ([grid_x, 1, 1], [ELEMENTWISE_BLOCK_SIZE, 1, 1])
}

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

/// CPU reference for element-wise add.
#[must_use]
pub fn add_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

/// CPU reference for element-wise sub.
#[must_use]
pub fn sub_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// CPU reference for element-wise mul.
#[must_use]
pub fn mul_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// CPU reference for element-wise div.
#[must_use]
pub fn div_reference(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect()
}

/// CPU reference for element-wise exp.
#[must_use]
pub fn exp_reference(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.exp()).collect()
}

/// CPU reference for element-wise log.
#[must_use]
pub fn log_reference(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.ln()).collect()
}

/// CPU reference for element-wise sqrt.
#[must_use]
pub fn sqrt_reference(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| x.sqrt()).collect()
}

/// CPU reference for element-wise neg.
#[must_use]
pub fn neg_reference(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| -x).collect()
}

/// CPU reference for scalar multiplication.
#[must_use]
pub fn scalar_mul_reference(input: &[f32], scalar: f32) -> Vec<f32> {
    input.iter().map(|&x| x * scalar).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_elementwise_tests.rs"]
mod ptx_elementwise_tests;
