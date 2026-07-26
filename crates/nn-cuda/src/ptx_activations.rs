// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for activation function kernels.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 elementwise
//! activation functions. Each activation is a simple grid-stride loop over
//! all elements -- no reduction or shared memory needed.
//!
//! ## Supported Activations
//!
//! | Activation | Formula                                  | Used in            |
//! |------------|------------------------------------------|--------------------|
//! | GELU       | `x * 0.5 * (1 + erf(x / sqrt(2)))`      | BERT, GPT, Whisper |
//! | GELU Fast  | `x * sigmoid(1.702 * x)`                 | Approximation      |
//! | SiLU/Swish | `x * sigmoid(x)`                         | Llama, Qwen3, GLM  |
//! | Mish       | `x * tanh(softplus(x))`                  | YOLOv4, detection  |
//! | Snake      | `x + (1/alpha) * sin(alpha*x)^2`         | Kokoro TTS         |
//!
//! ## PTX implementation notes
//!
//! - **sigmoid** is computed as `1 / (1 + exp(-x))` using `ex2.approx.f32`
//!   with a log2(e) prescale.
//! - **tanh** is computed as `2 * sigmoid(2x) - 1`.
//! - **GELU exact** uses the Abramowitz & Stegun erf approximation:
//!   `erf(x) ~= 1 - (a1*t + a2*t^2 + a3*t^3) * exp(-x^2)` where `t = 1/(1+0.47047*|x|)`.
//! - **Snake** takes an extra `alpha` parameter passed as a kernel argument.
//!
//! ## Kernel interface
//!
//! Parameters (common to all activations except Snake):
//! - `param_input`  -- pointer to input tensor (f32)
//! - `param_output` -- pointer to output tensor (f32)
//! - `param_n`      -- u32, total number of elements
//!
//! Snake adds:
//! - `param_alpha`  -- f32, the alpha frequency parameter
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)`.
//! Grid: `(ceil(n / 256), 1, 1)`.

use crate::codegen_ptx::{format_ptx_float, ptx_prelude, PtxCodegenError, DEFAULT_SM_TARGET};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for activation kernels (256 threads).
const ACTIVATION_BLOCK_SIZE: usize = 256;

/// log2(e) as f32 -- prescale factor for `ex2.approx.f32`.
const LOG2_E: f32 = std::f32::consts::LOG2_E;

// ---------------------------------------------------------------------------
// Activation enum
// ---------------------------------------------------------------------------

/// Supported activation functions for PTX kernel generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxActivation {
    /// Gaussian Error Linear Unit (exact via erf approximation).
    Gelu,
    /// GELU fast approximation: `x * sigmoid(1.702 * x)`.
    GeluFast,
    /// Sigmoid Linear Unit / Swish: `x * sigmoid(x)`.
    Silu,
    /// Mish: `x * tanh(softplus(x))` where softplus(x) = ln(1 + exp(x)).
    Mish,
    /// Snake: `x + (1/alpha) * sin(alpha*x)^2`. Requires alpha parameter.
    Snake,
}

impl PtxActivation {
    /// Human-readable name for comments and kernel naming.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gelu => "gelu",
            Self::Silu => "silu",
            Self::GeluFast => "gelu_fast",
            Self::Mish => "mish",
            Self::Snake => "snake",
        }
    }

    /// Whether this activation requires an extra alpha parameter.
    #[must_use]
    pub fn requires_alpha(&self) -> bool {
        matches!(self, Self::Snake)
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for PTX activation kernel generation.
#[derive(Debug, Clone)]
pub struct PtxActivationConfig {
    /// Kernel function name in the PTX module.
    pub kernel_name: String,
    /// Which activation function to generate.
    pub activation: PtxActivation,
    /// SM target for the PTX prelude (e.g., "sm_80").
    pub sm_target: String,
    /// Block size (number of threads per block). Default: 256.
    pub block_size: usize,
}

impl PtxActivationConfig {
    /// Create an activation config with default sm_80 target and 256-thread blocks.
    pub fn new(kernel_name: &str, activation: PtxActivation) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            activation,
            sm_target: DEFAULT_SM_TARGET.to_string(),
            block_size: ACTIVATION_BLOCK_SIZE,
        }
    }

    /// Set the SM target.
    #[must_use]
    pub fn with_sm_target(mut self, target: &str) -> Self {
        self.sm_target = target.to_string();
        self
    }

    /// Set the block size.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.kernel_name.is_empty() {
            return Err(PtxCodegenError::InvalidParameter(
                "kernel_name must not be empty".into(),
            ));
        }
        if self.block_size == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "block_size must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PTX generation
// ---------------------------------------------------------------------------

/// Emit a complete PTX module for an f32 activation function.
///
/// Generates a grid-stride loop kernel that applies the specified activation
/// elementwise. No shared memory or warp reduction needed -- pure elementwise.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_activations::{emit_ptx_activation, PtxActivation, PtxActivationConfig};
/// let config = PtxActivationConfig::new("silu_kernel", PtxActivation::Silu);
/// let ptx = emit_ptx_activation(&config).unwrap();
/// assert!(ptx.contains(".entry silu_kernel"));
/// ```
pub fn emit_ptx_activation(config: &PtxActivationConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let name = &config.kernel_name;
    let block_size = config.block_size;
    let act = config.activation;

    let mut ptx = String::with_capacity(4096);

    // -- Module header --
    ptx.push_str(&ptx_prelude(&config.sm_target));
    ptx.push_str(&format!(
        "// Activation: {} f32, block_size={block_size}\n\n",
        act.name()
    ));

    // -- Kernel entry point --
    ptx.push_str(&format!(".visible .entry {name}(\n"));
    ptx.push_str(
        "\x20   .param .u64 param_input,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n",
    );
    if act.requires_alpha() {
        ptx.push_str(",\n\x20   .param .f32 param_alpha");
    }
    ptx.push_str("\n)\n");
    ptx.push_str(&format!(".reqntid {block_size}\n{{\n"));

    // -- Register declarations --
    ptx.push_str(
        "\x20   // Register declarations\n\
         \x20   .reg .u32  %r<10>;\n\
         \x20   .reg .f32  %f<16>;\n\
         \x20   .reg .u64  %rd<8>;\n\
         \x20   .reg .pred %p<4>;\n\n",
    );

    // -- Load parameters --
    ptx.push_str(
        "\x20   // Load kernel parameters\n\
         \x20   ld.param.u64  %rd0, [param_input];\n\
         \x20   ld.param.u64  %rd1, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n",
    );
    if act.requires_alpha() {
        ptx.push_str("\x20   ld.param.f32  %f14, [param_alpha];\n");
    }
    ptx.push('\n');

    // -- Compute global thread index --
    ptx.push_str(&format!(
        "\x20   // Global thread index = blockIdx.x * blockDim.x + threadIdx.x\n\
         \x20   mov.u32       %r1, %ctaid.x;\n\
         \x20   mov.u32       %r2, %tid.x;\n\
         \x20   mad.lo.u32    %r3, %r1, {block_size}, %r2; // global_idx\n\n"
    ));

    // -- Grid-stride loop --
    ptx.push_str(&format!(
        "\x20   // Grid-stride loop\n\
         \x20   mov.u32       %r4, %nctaid.x;         // gridDim.x\n\
         \x20   mul.lo.u32    %r5, %r4, {block_size}; // stride = gridDim.x * blockDim.x\n\
         ACT_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r3, %r0;          // idx >= n?\n\
         \x20   @%p0 bra      ACT_EXIT;\n\
         \x20   // Load input[idx]\n\
         \x20   mul.wide.u32  %rd2, %r3, 4;\n\
         \x20   add.u64       %rd3, %rd0, %rd2;       // &input[idx]\n\
         \x20   ld.global.f32 %f0, [%rd3];            // x = input[idx]\n\n"
    ));

    // -- Compute activation --
    match act {
        PtxActivation::Silu => emit_silu_body(&mut ptx),
        PtxActivation::Gelu => emit_gelu_body(&mut ptx),
        PtxActivation::GeluFast => emit_gelu_fast_body(&mut ptx),
        PtxActivation::Mish => emit_mish_body(&mut ptx),
        PtxActivation::Snake => emit_snake_body(&mut ptx),
    }

    // -- Store output[idx] and advance --
    ptx.push_str("\x20   // Store output[idx]\n\
         \x20   add.u64       %rd4, %rd1, %rd2;       // &output[idx]\n\
         \x20   st.global.f32 [%rd4], %f1;            // output[idx] = result\n\
         \x20   add.u32       %r3, %r3, %r5;          // idx += stride\n\
         \x20   bra           ACT_LOOP;\n\n");

    // -- Kernel exit --
    ptx.push_str(
        "ACT_EXIT:\n\
         \x20   ret;\n\
         }\n",
    );

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Activation body emitters
// ---------------------------------------------------------------------------

/// SiLU/Swish: `result = x * sigmoid(x)` where sigmoid(x) = 1/(1+exp(-x))
fn emit_silu_body(ptx: &mut String) {
    let log2e = format_ptx_float(LOG2_E);
    let one = format_ptx_float(1.0);
    ptx.push_str(&format!(
        "\x20   // SiLU: x * sigmoid(x)\n\
         \x20   // sigmoid(x) = 1 / (1 + exp(-x))\n\
         \x20   // exp(-x) = 2^(-x * log2(e))\n\
         \x20   neg.f32       %f2, %f0;               // -x\n\
         \x20   mul.f32       %f3, %f2, {log2e};      // -x * log2(e)\n\
         \x20   ex2.approx.f32 %f4, %f3;              // exp(-x)\n\
         \x20   add.f32       %f5, %f4, {one};        // 1 + exp(-x)\n\
         \x20   rcp.approx.f32 %f6, %f5;              // 1 / (1 + exp(-x)) = sigmoid(x)\n\
         \x20   mul.f32       %f1, %f0, %f6;          // x * sigmoid(x)\n\n"
    ));
}

/// GELU exact: `result = x * 0.5 * (1 + erf(x / sqrt(2)))`
///
/// Uses the Abramowitz & Stegun polynomial approximation for erf:
/// `erf(x) = 1 - (a1*t + a2*t^2 + a3*t^3) * exp(-x^2)` with sign correction,
/// where `t = 1/(1 + p*|x|)`, p = 0.47047.
fn emit_gelu_body(ptx: &mut String) {
    let half = format_ptx_float(0.5);
    let one = format_ptx_float(1.0);
    let rsqrt2 = format_ptx_float(std::f32::consts::FRAC_1_SQRT_2); // 1/sqrt(2)
    let log2e = format_ptx_float(LOG2_E);
    // Abramowitz & Stegun constants
    let p = format_ptx_float(0.3275911);
    let a1 = format_ptx_float(0.254_829_6);
    let a2 = format_ptx_float(-0.284_496_72);
    let a3 = format_ptx_float(1.421_413_8);
    let a4 = format_ptx_float(-1.453_152_1);
    let a5 = format_ptx_float(1.061_405_4);

    ptx.push_str(&format!(
        "\x20   // GELU: x * 0.5 * (1 + erf(x / sqrt(2)))\n\
         \x20   // erf via Abramowitz & Stegun 5-term approximation\n\
         \x20   mul.f32       %f2, %f0, {rsqrt2};     // z = x / sqrt(2)\n\
         \x20   abs.f32       %f3, %f2;               // |z|\n\
         \x20   // t = 1/(1 + p*|z|)\n\
         \x20   mul.f32       %f4, %f3, {p};          // p * |z|\n\
         \x20   add.f32       %f4, %f4, {one};        // 1 + p*|z|\n\
         \x20   rcp.approx.f32 %f4, %f4;              // t = 1/(1+p*|z|)\n\
         \x20   // Horner: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t\n\
         \x20   mov.f32       %f5, {a5};\n\
         \x20   fma.rn.f32    %f5, %f5, %f4, {a4};    // a5*t + a4\n\
         \x20   fma.rn.f32    %f5, %f5, %f4, {a3};    // *t + a3\n\
         \x20   fma.rn.f32    %f5, %f5, %f4, {a2};    // *t + a2\n\
         \x20   fma.rn.f32    %f5, %f5, %f4, {a1};    // *t + a1\n\
         \x20   mul.f32       %f5, %f5, %f4;          // * t\n\
         \x20   // exp(-z^2) via 2^(-z^2 * log2(e))\n\
         \x20   mul.f32       %f6, %f3, %f3;          // z^2 (using |z|)\n\
         \x20   neg.f32       %f6, %f6;               // -z^2\n\
         \x20   mul.f32       %f6, %f6, {log2e};      // -z^2 * log2(e)\n\
         \x20   ex2.approx.f32 %f6, %f6;              // exp(-z^2)\n\
         \x20   mul.f32       %f5, %f5, %f6;          // poly * exp(-z^2)\n\
         \x20   sub.f32       %f7, {one}, %f5;        // erf(|z|) = 1 - poly*exp\n\
         \x20   // Apply sign: erf(z) = sign(z) * erf(|z|)\n\
         \x20   setp.lt.f32   %p1, %f2, {zero};       // z < 0?\n\
         \x20   neg.f32       %f8, %f7;               // -erf(|z|)\n\
         \x20   selp.f32      %f7, %f8, %f7, %p1;    // erf(z)\n\
         \x20   // gelu = x * 0.5 * (1 + erf(z))\n\
         \x20   add.f32       %f9, {one}, %f7;        // 1 + erf(z)\n\
         \x20   mul.f32       %f9, %f9, {half};       // 0.5 * (1+erf)\n\
         \x20   mul.f32       %f1, %f0, %f9;          // x * 0.5 * (1+erf)\n\n",
        zero = format_ptx_float(0.0),
    ));
}

/// GELU fast approximation: `result = x * sigmoid(1.702 * x)`
fn emit_gelu_fast_body(ptx: &mut String) {
    let coeff = format_ptx_float(1.702);
    let log2e = format_ptx_float(LOG2_E);
    let one = format_ptx_float(1.0);
    ptx.push_str(&format!(
        "\x20   // GELU fast: x * sigmoid(1.702 * x)\n\
         \x20   mul.f32       %f2, %f0, {coeff};      // 1.702 * x\n\
         \x20   neg.f32       %f3, %f2;               // -(1.702*x)\n\
         \x20   mul.f32       %f3, %f3, {log2e};      // -(1.702*x) * log2(e)\n\
         \x20   ex2.approx.f32 %f4, %f3;              // exp(-(1.702*x))\n\
         \x20   add.f32       %f5, %f4, {one};        // 1 + exp(-(1.702*x))\n\
         \x20   rcp.approx.f32 %f6, %f5;              // sigmoid(1.702*x)\n\
         \x20   mul.f32       %f1, %f0, %f6;          // x * sigmoid(1.702*x)\n\n"
    ));
}

/// Mish: `result = x * tanh(softplus(x))` where softplus(x) = ln(1+exp(x))
fn emit_mish_body(ptx: &mut String) {
    let log2e = format_ptx_float(LOG2_E);
    let one = format_ptx_float(1.0);
    let two = format_ptx_float(2.0);
    let rcp_log2e = format_ptx_float(1.0 / LOG2_E); // ln(2)
    ptx.push_str(&format!(
        "\x20   // Mish: x * tanh(softplus(x))\n\
         \x20   // softplus(x) = ln(1 + exp(x))\n\
         \x20   // exp(x) = 2^(x * log2(e))\n\
         \x20   mul.f32       %f2, %f0, {log2e};      // x * log2(e)\n\
         \x20   ex2.approx.f32 %f3, %f2;              // exp(x)\n\
         \x20   add.f32       %f4, %f3, {one};        // 1 + exp(x)\n\
         \x20   // ln(y) = lg2(y) * ln(2) = lg2(y) / log2(e)\n\
         \x20   lg2.approx.f32 %f5, %f4;              // lg2(1+exp(x))\n\
         \x20   mul.f32       %f5, %f5, {rcp_log2e};  // softplus = ln(1+exp(x))\n\
         \x20   // tanh(sp) = 2*sigmoid(2*sp) - 1\n\
         \x20   mul.f32       %f6, %f5, {two};        // 2*sp\n\
         \x20   neg.f32       %f7, %f6;               // -(2*sp)\n\
         \x20   mul.f32       %f7, %f7, {log2e};      // -(2*sp)*log2(e)\n\
         \x20   ex2.approx.f32 %f8, %f7;              // exp(-2*sp)\n\
         \x20   add.f32       %f9, %f8, {one};        // 1 + exp(-2*sp)\n\
         \x20   rcp.approx.f32 %f9, %f9;              // sigmoid(2*sp)\n\
         \x20   fma.rn.f32    %f10, %f9, {two}, {neg_one}; // 2*sigmoid(2*sp) - 1 = tanh(sp)\n\
         \x20   mul.f32       %f1, %f0, %f10;         // x * tanh(softplus(x))\n\n",
        neg_one = format_ptx_float(-1.0),
    ));
}

/// Snake: `result = x + (1/alpha) * sin(alpha*x)^2`
///
/// Alpha is loaded from `%f14` (param_alpha).
fn emit_snake_body(ptx: &mut String) {
    ptx.push_str(
        "\x20   // Snake: x + (1/alpha) * sin(alpha*x)^2\n\
         \x20   // alpha is in %f14\n\
         \x20   mul.f32       %f2, %f14, %f0;         // alpha * x\n\
         \x20   sin.approx.f32 %f3, %f2;              // sin(alpha*x)\n\
         \x20   mul.f32       %f4, %f3, %f3;          // sin(alpha*x)^2\n\
         \x20   rcp.approx.f32 %f5, %f14;             // 1/alpha\n\
         \x20   fma.rn.f32    %f1, %f5, %f4, %f0;    // x + (1/alpha)*sin^2\n\n",
    );
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Convenience: emit PTX activation with default sm_80 target and 256-thread blocks.
pub fn emit_ptx_activation_default(
    name: &str,
    activation: PtxActivation,
) -> Result<String, PtxCodegenError> {
    emit_ptx_activation(&PtxActivationConfig::new(name, activation))
}

/// Compute the grid and block dimensions for a PTX activation kernel.
///
/// Grid: `(ceil(n / block_size), 1, 1)`.
/// Block: `(block_size, 1, 1)`.
///
/// # Returns
///
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_activation_launch_config(n: usize, block_size: usize) -> ([usize; 3], [usize; 3]) {
    let bs = if block_size == 0 {
        ACTIVATION_BLOCK_SIZE
    } else {
        block_size
    };
    let grid_x = n.div_ceil(bs);
    ([grid_x, 1, 1], [bs, 1, 1])
}

/// Generate PTX for all supported activations with default settings.
///
/// Returns a `Vec` of `(activation_name, ptx_string)` pairs. Useful for
/// building a kernel library with all activations pre-compiled.
pub fn generate_all_activation_ptx() -> Vec<(&'static str, String)> {
    let activations = [
        PtxActivation::Gelu,
        PtxActivation::GeluFast,
        PtxActivation::Silu,
        PtxActivation::Mish,
        PtxActivation::Snake,
    ];
    activations
        .iter()
        .map(|&act| {
            let name = format!("ptx_{}_f32", act.name());
            let ptx =
                emit_ptx_activation_default(&name, act).expect("activation PTX generation failed");
            (act.name(), ptx)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Reference CPU implementations
// ---------------------------------------------------------------------------

/// Sigmoid: `1 / (1 + exp(-x))`.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU/Swish reference: `x * sigmoid(x)`.
pub fn silu_reference(x: f32) -> f32 {
    x * sigmoid(x)
}

/// GELU exact reference using erf.
///
/// `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`
///
/// Uses the Abramowitz & Stegun 5-term polynomial approximation for erf.
pub fn gelu_reference(x: f32) -> f32 {
    // erf approximation (Abramowitz & Stegun, formula 7.1.26)
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_7 + t * (-1.453_152 + t * 1.061_405_4))));
    let erf_abs = 1.0 - poly * (-z * z).exp();
    let erf_val = if z >= 0.0 { erf_abs } else { -erf_abs };
    x * 0.5 * (1.0 + erf_val)
}

/// GELU fast approximation reference: `x * sigmoid(1.702 * x)`.
pub fn gelu_fast_reference(x: f32) -> f32 {
    x * sigmoid(1.702 * x)
}

/// Mish reference: `x * tanh(softplus(x))` where softplus(x) = ln(1+exp(x)).
pub fn mish_reference(x: f32) -> f32 {
    let softplus = x.exp().ln_1p();
    x * softplus.tanh()
}

/// Snake reference: `x + (1/alpha) * sin(alpha*x)^2`.
pub fn snake_reference(x: f32, alpha: f32) -> f32 {
    let sin_val = (alpha * x).sin();
    x + (1.0 / alpha) * sin_val * sin_val
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_activations_tests.rs"]
mod ptx_activations_tests;
