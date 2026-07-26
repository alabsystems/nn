// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CUDA C++ kernel generation for embedding lookup.
//!
//! Generates CUDA C++ kernel source for f32 embedding table lookups, compiled
//! via `nvcc` to PTX/cubin. Given token IDs (`u32[]`) and an embedding table
//! (`float[vocab_size][embedding_dim]`), produces:
//!
//! `output[i] = embedding_table[token_ids[i]]`
//!
//! ## Algorithm
//!
//! Each thread handles one `(token, dim_offset)` element using a grid-stride
//! loop. Total work items = `num_tokens * embedding_dim`. The kernel:
//!
//! 1. Computes `token_idx = global_idx / embedding_dim`
//! 2. Computes `dim_idx   = global_idx % embedding_dim`
//! 3. Loads `tok = token_ids[token_idx]`
//! 4. Bounds-checks `tok < vocab_size`; out-of-bounds tokens produce zero
//! 5. Copies `output[token_idx * embedding_dim + dim_idx] =
//!           embedding_table[tok * embedding_dim + dim_idx]`
//!
//! ## Kernel interface
//!
//! ```c
//! __global__ void embedding_lookup(
//!     const unsigned int* __restrict__ token_ids,     // [num_tokens]
//!     const float*        __restrict__ embedding_table, // [vocab_size][embedding_dim]
//!     float*              __restrict__ output,         // [num_tokens][embedding_dim]
//!     const unsigned int vocab_size,
//!     const unsigned int embedding_dim,
//!     const unsigned int num_tokens
//! );
//! ```
//!
//! ## Grid and block configuration
//!
//! Block: `(block_size, 1, 1)` — default 256.
//! Grid:  `(ceil(num_tokens * embedding_dim / block_size), 1, 1)`.
//!
//! Parallel to Metal embedding in `dyn_tensor_metal_ops.rs` and raw PTX
//! generation patterns in `ptx_activations.rs`.

use crate::codegen_ptx::PtxCodegenError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for embedding kernels (256 threads).
const DEFAULT_EMBEDDING_BLOCK_SIZE: usize = 256;

/// Public block size constant for embedding kernels.
pub const EMBEDDING_BLOCK_SIZE: u32 = DEFAULT_EMBEDDING_BLOCK_SIZE as u32;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for CUDA C++ embedding lookup kernel generation.
#[derive(Debug, Clone)]
pub struct PtxEmbeddingConfig {
    /// Vocabulary size (number of rows in the embedding table).
    pub vocab_size: usize,
    /// Embedding dimension (number of columns / row length).
    pub embedding_dim: usize,
    /// Thread block size (default: 256).
    pub block_size: usize,
}

impl PtxEmbeddingConfig {
    /// Create a config with default 256-thread blocks.
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        Self {
            vocab_size,
            embedding_dim,
            block_size: DEFAULT_EMBEDDING_BLOCK_SIZE,
        }
    }

    /// Set the thread block size.
    #[must_use]
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), PtxCodegenError> {
        if self.vocab_size == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "vocab_size must be > 0".into(),
            ));
        }
        if self.embedding_dim == 0 {
            return Err(PtxCodegenError::InvalidParameter(
                "embedding_dim must be > 0".into(),
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
// CUDA C++ kernel generation — public API
// ---------------------------------------------------------------------------

/// Generate CUDA C++ kernel source for f32 embedding lookup.
///
/// Produces a complete CUDA C++ source string containing a `__global__`
/// kernel that performs embedding table lookup with a grid-stride loop.
/// Out-of-bounds token IDs produce zero-filled output rows.
///
/// The generated kernel uses `__restrict__` pointers to enable compiler
/// optimizations and documents the vocab/dim configuration in a header
/// comment.
///
/// # Arguments
///
/// * `config` — Kernel configuration (vocab_size, embedding_dim, block_size).
///
/// # Returns
///
/// CUDA C++ source string ready for `nvcc` compilation.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_embedding::{generate_embedding_ptx, PtxEmbeddingConfig};
/// let config = PtxEmbeddingConfig::new(50257, 768);
/// let src = generate_embedding_ptx(&config).unwrap();
/// assert!(src.contains("__global__"));
/// assert!(src.contains("embedding_lookup"));
/// ```
pub fn generate_embedding_ptx(config: &PtxEmbeddingConfig) -> Result<String, PtxCodegenError> {
    config.validate()?;

    let vocab_size = config.vocab_size;
    let embedding_dim = config.embedding_dim;
    let block_size = config.block_size;

    let mut src = String::with_capacity(2048);

    // -- Header comment --
    src.push_str(&format!(
        "// Embedding lookup kernel (CUDA C++)\n\
         // vocab_size={vocab_size}, embedding_dim={embedding_dim}, block_size={block_size}\n\
         //\n\
         // output[i] = embedding_table[token_ids[i]]\n\
         // Grid-stride loop over num_tokens * embedding_dim elements.\n\
         // Out-of-bounds token IDs produce zero output.\n\n"
    ));

    // -- Kernel definition --
    src.push_str(
        "__global__ void embedding_lookup(\n\
         \x20   const unsigned int* __restrict__ token_ids,\n\
         \x20   const float*        __restrict__ embedding_table,\n\
         \x20   float*              __restrict__ output,\n\
         \x20   const unsigned int vocab_size,\n\
         \x20   const unsigned int embedding_dim,\n\
         \x20   const unsigned int num_tokens\n\
         ) {\n",
    );

    // -- Grid-stride loop --
    src.push_str(
        "\x20   // Total work items = num_tokens * embedding_dim\n\
         \x20   const unsigned int total = num_tokens * embedding_dim;\n\
         \x20   for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20        idx < total;\n\
         \x20        idx += gridDim.x * blockDim.x) {\n",
    );

    // -- Decompose idx into (token_idx, dim_idx) --
    src.push_str(
        "\x20       // Decompose global index into (token, dimension)\n\
         \x20       const unsigned int token_idx = idx / embedding_dim;\n\
         \x20       const unsigned int dim_idx   = idx % embedding_dim;\n\n",
    );

    // -- Load token ID and bounds check --
    src.push_str(
        "\x20       // Load token ID and validate against vocab_size\n\
         \x20       const unsigned int tok = token_ids[token_idx];\n\
         \x20       if (tok < vocab_size) {\n\
         \x20           output[token_idx * embedding_dim + dim_idx] =\n\
         \x20               embedding_table[tok * embedding_dim + dim_idx];\n\
         \x20       } else {\n\
         \x20           // Out-of-bounds: zero fill\n\
         \x20           output[token_idx * embedding_dim + dim_idx] = 0.0f;\n\
         \x20       }\n",
    );

    // -- Close loop and kernel --
    src.push_str(
        "\x20   }\n\
         }\n",
    );

    Ok(src)
}

// ---------------------------------------------------------------------------
// Launch configuration
// ---------------------------------------------------------------------------

/// Compute grid and block dimensions for an embedding lookup kernel.
///
/// Total work items = `num_tokens * embedding_dim`.
/// Grid:  `(ceil(total / block_size), 1, 1)`.
/// Block: `(block_size, 1, 1)`.
///
/// # Returns
///
/// `(grid_size, block_size)` as `(u32, u32)`.
#[must_use]
pub fn ptx_embedding_launch_config(num_tokens: usize, config: &PtxEmbeddingConfig) -> (u32, u32) {
    let total = num_tokens * config.embedding_dim;
    let bs = config.block_size.max(1) as u32;
    let grid = ((total as u64).div_ceil(u64::from(bs))).min(u64::from(u32::MAX)) as u32;
    (grid, bs)
}

// ---------------------------------------------------------------------------
// CPU reference implementation
// ---------------------------------------------------------------------------

/// CPU reference implementation for embedding table lookup.
///
/// For each index in `indices`, copies the corresponding row from `table`
/// (of width `dim`) into the output. Out-of-bounds indices (>= table rows)
/// produce zero-filled rows.
///
/// # Arguments
///
/// * `indices` - Token IDs to look up.
/// * `table` - Flat embedding table, row-major `[vocab_size][dim]`.
/// * `dim` - Embedding dimension (row width).
///
/// # Returns
///
/// Flat output vector of length `indices.len() * dim`.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_embedding::embedding_reference;
/// let table = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2 rows, dim=3
/// let out = embedding_reference(&[1, 0], &table, 3);
/// assert_eq!(out, vec![0.4, 0.5, 0.6, 0.1, 0.2, 0.3]);
/// ```
pub fn embedding_reference(indices: &[u32], table: &[f32], dim: usize) -> Vec<f32> {
    let vocab_size = table.len().checked_div(dim).unwrap_or(0);
    let mut output = vec![0.0f32; indices.len() * dim];
    for (i, &tok) in indices.iter().enumerate() {
        let tok = tok as usize;
        if tok < vocab_size {
            let src_offset = tok * dim;
            let dst_offset = i * dim;
            output[dst_offset..dst_offset + dim]
                .copy_from_slice(&table[src_offset..src_offset + dim]);
        }
        // else: output stays zero (OOV)
    }
    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ptx_embedding_tests.rs"]
mod tests;
