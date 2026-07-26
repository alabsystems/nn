// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CUDA C++ kernel generation for Rotary Position Embeddings (RoPE).
//!
//! Generates CUDA C++ kernel source for applying rotary position embeddings
//! to query/key tensors in transformer attention layers. RoPE encodes
//! absolute position information by rotating pairs of dimensions in the
//! embedding space, following [Su et al., 2021](https://arxiv.org/abs/2104.09864).
//!
//! ## Algorithm
//!
//! For each position `pos` in `[0, seq_len)` and each dimension pair
//! `(2i, 2i+1)` in `[0, head_dim)`:
//!
//! ```text
//! theta_i = pos / (10000^(2i / head_dim))
//! cos_val = cos(theta_i)
//! sin_val = sin(theta_i)
//!
//! x0 = input[..., 2i]
//! x1 = input[..., 2i+1]
//! output[..., 2i]     = x0 * cos_val - x1 * sin_val
//! output[..., 2i+1]   = x0 * sin_val + x1 * cos_val
//! ```
//!
//! ## Kernel variants
//!
//! 1. **`generate_rope_ptx`** — computes sin/cos on the fly from position
//!    and dimension indices using `__sinf`/`__cosf` intrinsics.
//! 2. **`generate_rope_cached_ptx`** — reads precomputed sin/cos tables
//!    from device memory, avoiding transcendental computation in the kernel.
//!
//! ## Grid and block configuration
//!
//! Block: `(block_size, 1, 1)` — default 256.
//! Grid:  `(ceil(seq_len * head_dim / 2 / block_size), 1, 1)`.
//!
//! Each thread handles one dimension pair `(2i, 2i+1)` at one sequence
//! position, using a grid-stride loop for large inputs.
//!
//! ## Usage in transformers
//!
//! RoPE is used in Llama, Qwen3, GLM, Mistral, and most modern decoder-only
//! LLMs. It replaces learned position embeddings with a fixed rotation scheme
//! that generalizes to longer sequences than seen during training.

use crate::codegen_ptx::PtxCodegenError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for RoPE kernels (256 threads).
pub const ROPE_BLOCK_SIZE: u32 = 256;

/// Base frequency for RoPE (10000.0 by convention).
const ROPE_BASE: f32 = 10000.0;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for CUDA C++ RoPE kernel generation.
#[derive(Debug, Clone)]
pub struct PtxRopeConfig {
    /// Maximum sequence length (compile-time hint; actual from kernel args).
    pub seq_len: usize,
    /// Head dimension — must be even (dimension pairs).
    pub head_dim: usize,
    /// Thread block size (default: 256).
    pub block_size: usize,
    /// RoPE frequency base (default: 10000.0).
    pub base: f32,
}

impl PtxRopeConfig {
    /// Create a RoPE config with default block size and base.
    pub fn new(seq_len: usize, head_dim: usize) -> Self {
        Self {
            seq_len,
            head_dim,
            block_size: ROPE_BLOCK_SIZE as usize,
            base: ROPE_BASE,
        }
    }

    /// Set the thread block size.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Set the frequency base (default: 10000.0).
    #[must_use]
    pub fn with_base(mut self, base: f32) -> Self {
        self.base = base;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.seq_len == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "seq_len must be > 0".into(),
            ));
        }
        if self.head_dim == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "head_dim must be > 0".into(),
            ));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(PtxCodegenError::InvalidParameter(
                "head_dim must be even for RoPE dimension pairing".into(),
            ));
        }
        if self.block_size == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "block_size must be > 0".into(),
            ));
        }
        if !self.base.is_finite() || self.base <= 0.0 {
            return Err(PtxCodegenError::InvalidParameter(
                "base must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CUDA C++ kernel generation: on-the-fly sin/cos
// ---------------------------------------------------------------------------

/// Generate CUDA C++ kernel source for RoPE with on-the-fly sin/cos.
///
/// Each thread handles one dimension pair `(2i, 2i+1)` at one position,
/// computing `theta = pos / base^(2i / head_dim)` and applying the rotation.
///
/// # Kernel signature
///
/// ```c
/// __global__ void rope_apply(
///     const float* __restrict__ input,   // [seq_len][head_dim]
///     float*       __restrict__ output,  // [seq_len][head_dim]
///     const unsigned int seq_len,
///     const unsigned int head_dim
/// );
/// ```
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_rope::{generate_rope_ptx, PtxRopeConfig};
/// let config = PtxRopeConfig::new(512, 64);
/// let src = generate_rope_ptx(&config).unwrap();
/// assert!(src.contains("__global__"));
/// assert!(src.contains("rope_apply"));
/// ```
pub fn generate_rope_ptx(config: &PtxRopeConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let seq_len = config.seq_len;
    let head_dim = config.head_dim;
    let block_size = config.block_size;
    let base = config.base;

    let mut src = String::with_capacity(2048);

    // -- Header comment --
    src.push_str(&format!(
        "// Rotary Position Embedding (RoPE) kernel (CUDA C++)\n\
         // seq_len={seq_len}, head_dim={head_dim}, block_size={block_size}, base={base}\n\
         //\n\
         // For each position and dimension pair (2i, 2i+1):\n\
         //   theta = pos / base^(2i / head_dim)\n\
         //   out[2i]   = x[2i]*cos(theta) - x[2i+1]*sin(theta)\n\
         //   out[2i+1] = x[2i]*sin(theta) + x[2i+1]*cos(theta)\n\n"
    ));

    src.push_str("#include <math.h>\n\n");

    // -- Kernel definition --
    src.push_str(
        "__global__ void rope_apply(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float*       __restrict__ output,\n\
         \x20   const unsigned int seq_len,\n\
         \x20   const unsigned int head_dim\n\
         ) {\n",
    );

    // -- Grid-stride loop over (pos, pair) space --
    // Total pairs = seq_len * (head_dim / 2)
    src.push_str(
        "\x20   const unsigned int half_dim = head_dim / 2;\n\
         \x20   const unsigned int total_pairs = seq_len * half_dim;\n\
         \x20   for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20        idx < total_pairs;\n\
         \x20        idx += gridDim.x * blockDim.x) {\n",
    );

    // -- Decompose idx into (pos, pair_idx) --
    src.push_str(
        "\x20       // Decompose into position and dimension pair\n\
         \x20       const unsigned int pos      = idx / half_dim;\n\
         \x20       const unsigned int pair_idx = idx % half_dim;\n\n",
    );

    // -- Compute theta --
    src.push_str(&format!(
        "\x20       // Compute rotation angle\n\
         \x20       const float freq_exp = (2.0f * (float)pair_idx) / (float)head_dim;\n\
         \x20       const float freq = 1.0f / powf({base:.1}f, freq_exp);\n\
         \x20       const float theta = (float)pos * freq;\n\
         \x20       const float cos_val = __cosf(theta);\n\
         \x20       const float sin_val = __sinf(theta);\n\n"
    ));

    // -- Load input pair and apply rotation --
    src.push_str(
        "\x20       // Load dimension pair and apply rotation\n\
         \x20       const unsigned int offset = pos * head_dim + 2 * pair_idx;\n\
         \x20       const float x0 = input[offset];\n\
         \x20       const float x1 = input[offset + 1];\n\
         \x20       output[offset]     = x0 * cos_val - x1 * sin_val;\n\
         \x20       output[offset + 1] = x0 * sin_val + x1 * cos_val;\n",
    );

    // -- Close loop and kernel --
    src.push_str(
        "\x20   }\n\
         }\n",
    );

    Ok(src)
}

// ---------------------------------------------------------------------------
// CUDA C++ kernel generation: precomputed sin/cos tables
// ---------------------------------------------------------------------------

/// Generate CUDA C++ kernel source for RoPE with precomputed sin/cos tables.
///
/// The sin/cos tables are passed as device buffers, avoiding transcendental
/// math in the kernel hot path. Tables are indexed as
/// `cos_table[pos * half_dim + pair_idx]` and
/// `sin_table[pos * half_dim + pair_idx]`.
///
/// # Kernel signature
///
/// ```c
/// __global__ void rope_apply_cached(
///     const float* __restrict__ input,     // [seq_len][head_dim]
///     float*       __restrict__ output,    // [seq_len][head_dim]
///     const float* __restrict__ cos_table, // [seq_len][head_dim/2]
///     const float* __restrict__ sin_table, // [seq_len][head_dim/2]
///     const unsigned int seq_len,
///     const unsigned int head_dim
/// );
/// ```
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_rope::{generate_rope_cached_ptx, PtxRopeConfig};
/// let config = PtxRopeConfig::new(512, 64);
/// let src = generate_rope_cached_ptx(&config).unwrap();
/// assert!(src.contains("rope_apply_cached"));
/// assert!(src.contains("cos_table"));
/// ```
pub fn generate_rope_cached_ptx(config: &PtxRopeConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let seq_len = config.seq_len;
    let head_dim = config.head_dim;
    let block_size = config.block_size;
    let base = config.base;

    let mut src = String::with_capacity(2048);

    // -- Header comment --
    src.push_str(&format!(
        "// RoPE with precomputed sin/cos tables (CUDA C++)\n\
         // seq_len={seq_len}, head_dim={head_dim}, block_size={block_size}, base={base}\n\
         //\n\
         // Uses cached sin/cos tables to avoid transcendental computation.\n\n"
    ));

    // -- Kernel definition --
    src.push_str(
        "__global__ void rope_apply_cached(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float*       __restrict__ output,\n\
         \x20   const float* __restrict__ cos_table,\n\
         \x20   const float* __restrict__ sin_table,\n\
         \x20   const unsigned int seq_len,\n\
         \x20   const unsigned int head_dim\n\
         ) {\n",
    );

    // -- Grid-stride loop --
    src.push_str(
        "\x20   const unsigned int half_dim = head_dim / 2;\n\
         \x20   const unsigned int total_pairs = seq_len * half_dim;\n\
         \x20   for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20        idx < total_pairs;\n\
         \x20        idx += gridDim.x * blockDim.x) {\n",
    );

    // -- Decompose and load cached sin/cos --
    src.push_str(
        "\x20       const unsigned int pos      = idx / half_dim;\n\
         \x20       const unsigned int pair_idx = idx % half_dim;\n\
         \x20       const float cos_val = cos_table[idx];\n\
         \x20       const float sin_val = sin_table[idx];\n\n",
    );

    // -- Apply rotation --
    src.push_str(
        "\x20       const unsigned int offset = pos * head_dim + 2 * pair_idx;\n\
         \x20       const float x0 = input[offset];\n\
         \x20       const float x1 = input[offset + 1];\n\
         \x20       output[offset]     = x0 * cos_val - x1 * sin_val;\n\
         \x20       output[offset + 1] = x0 * sin_val + x1 * cos_val;\n",
    );

    // -- Close --
    src.push_str(
        "\x20   }\n\
         }\n",
    );

    Ok(src)
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Compute grid and block dimensions for a RoPE kernel.
///
/// Total work items = `seq_len * (head_dim / 2)` (one thread per dimension pair).
/// Grid:  `(ceil(total / block_size), 1, 1)`.
/// Block: `(block_size, 1, 1)`.
///
/// # Returns
///
/// `(grid_size, block_size)` as `(u32, u32)`.
#[must_use]
pub fn ptx_rope_launch_config(seq_len: usize, config: &PtxRopeConfig) -> (u32, u32) {
    let total_pairs = seq_len * (config.head_dim / 2);
    let bs = config.block_size.max(1) as u32;
    let grid = ((total_pairs as u64).div_ceil(u64::from(bs))).min(u64::from(u32::MAX)) as u32;
    (grid, bs)
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation for RoPE.
///
/// Applies rotary position embeddings to a flat input tensor of shape
/// `[seq_len][head_dim]`. Uses the standard formula with base 10000.0:
///
/// ```text
/// theta_i = pos / 10000^(2i / head_dim)
/// out[2i]   = x[2i]*cos(theta) - x[2i+1]*sin(theta)
/// out[2i+1] = x[2i]*sin(theta) + x[2i+1]*cos(theta)
/// ```
///
/// # Arguments
///
/// * `x` - Input tensor, flat `[seq_len * head_dim]`.
/// * `seq_len` - Number of positions.
/// * `head_dim` - Dimension of each head (must be even).
///
/// # Returns
///
/// Output tensor with rotary embeddings applied.
///
/// # Panics
///
/// Panics if `head_dim` is odd or if `x.len() != seq_len * head_dim`.
pub fn rope_reference(x: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    rope_reference_with_base(x, seq_len, head_dim, ROPE_BASE)
}

/// CPU reference implementation for RoPE with configurable base frequency.
pub fn rope_reference_with_base(x: &[f32], seq_len: usize, head_dim: usize, base: f32) -> Vec<f32> {
    assert_eq!(head_dim % 2, 0, "head_dim must be even");
    assert_eq!(
        x.len(),
        seq_len * head_dim,
        "input length mismatch: {} != {}",
        x.len(),
        seq_len * head_dim
    );

    let half_dim = head_dim / 2;
    let mut output = vec![0.0f32; x.len()];

    for pos in 0..seq_len {
        for i in 0..half_dim {
            let freq_exp = (2 * i) as f32 / head_dim as f32;
            let freq = 1.0 / base.powf(freq_exp);
            let theta = pos as f32 * freq;
            let cos_val = theta.cos();
            let sin_val = theta.sin();

            let offset = pos * head_dim + 2 * i;
            let x0 = x[offset];
            let x1 = x[offset + 1];
            output[offset] = x0 * cos_val - x1 * sin_val;
            output[offset + 1] = x0 * sin_val + x1 * cos_val;
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_rope_tests.rs"]
mod tests;
