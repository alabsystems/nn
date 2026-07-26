// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! PTX assembly generation for gather and scatter operations.
//!
//! Generates raw PTX (Parallel Thread Execution) assembly for f32 gather/scatter
//! operations. These are index-based memory access patterns common in embedding
//! lookups, sparse attention, and scatter-add aggregation.
//!
//! ## Gather
//!
//! `output[i] = data[indices[i]]` (1-D gather; generalized with `dim_size` for
//! multi-dimensional support).
//!
//! For a 2-D gather along dimension `dim` with `dim_size` columns:
//! - Row `i / dim_size`, column from `indices[i]`
//! - `output[i] = data[row * dim_size + indices[i]]`
//!
//! ## Scatter-add
//!
//! `output[indices[i]] += src[i]` (atomic addition to handle duplicate indices).
//!
//! For 2-D scatter along dimension `dim` with `dim_size` columns:
//! - `output[row * dim_size + indices[i]] += src[i]`
//!
//! ## Kernel interface
//!
//! Gather parameters:
//! - `param_data`    -- pointer to source data (f32)
//! - `param_indices` -- pointer to index array (u32)
//! - `param_output`  -- pointer to output (f32)
//! - `param_n`       -- u32, number of elements to gather
//! - `param_dim_size` -- u32, size of the gather dimension
//!
//! Scatter-add parameters:
//! - `param_src`     -- pointer to source values (f32)
//! - `param_indices` -- pointer to index array (u32)
//! - `param_output`  -- pointer to output (f32, pre-initialized)
//! - `param_n`       -- u32, number of elements to scatter
//! - `param_dim_size` -- u32, size of the scatter dimension
//!
//! ## Thread block configuration
//!
//! Block: `(256, 1, 1)`.
//! Grid:  `(ceil(n / 256), 1, 1)`.

use crate::codegen_ptx::ptx_prelude;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default block size for gather/scatter kernels (256 threads).
pub const GATHER_BLOCK_SIZE: u32 = 256;

/// SM target for gather/scatter kernels.
const SM_TARGET: &str = "sm_70";

// ---------------------------------------------------------------------------
// Gather
// ---------------------------------------------------------------------------

/// Generate PTX for gather along a dimension.
///
/// `output[i] = data[(i / dim_size) * dim_size + indices[i]]`
///
/// For 1-D gather, set `dim_size = n` (or use `dim = 0`).
///
/// # Arguments
/// * `n` -- total number of elements to gather
/// * `dim` -- dimension along which to gather (used in comments; the actual
///   layout is controlled by `dim_size` at launch time)
///
/// # Example
/// ```
/// use nn_cuda::ptx_gather::generate_gather_ptx;
/// let ptx = generate_gather_ptx(1024, 0);
/// assert!(ptx.contains(".entry ptx_gather_f32"));
/// ```
#[must_use]
pub fn generate_gather_ptx(n: u32, dim: u32) -> String {
    let block_size = GATHER_BLOCK_SIZE;

    let mut ptx = String::with_capacity(4096);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Gather f32: n={n}, dim={dim}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_gather_f32(\n\
         \x20   .param .u64 param_data,\n\
         \x20   .param .u64 param_indices,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n,\n\
         \x20   .param .u32 param_dim_size\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<12>;\n\
         \x20   .reg .f32  %f<2>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_data];\n\
         \x20   ld.param.u64  %rd1, [param_indices];\n\
         \x20   ld.param.u64  %rd2, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\
         \x20   ld.param.u32  %r1,  [param_dim_size];\n\n",
    );

    // Global thread index
    ptx.push_str(
        "\x20   mov.u32       %r2, %tid.x;\n\
         \x20   mov.u32       %r3, %ctaid.x;\n\
         \x20   mov.u32       %r4, %ntid.x;\n\
         \x20   mad.lo.u32    %r5, %r3, %r4, %r2;    // idx = blockIdx.x * blockDim.x + threadIdx.x\n\n",
    );

    // Grid-stride loop
    ptx.push_str("\x20   mov.u32       %r6, %nctaid.x;\n\
         \x20   mul.lo.u32    %r7, %r6, %r4;          // grid_stride\n\
         GATHER_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r5, %r0;          // idx >= n?\n\
         \x20   @%p0 bra      GATHER_EXIT;\n\
         \x20   // Compute row = idx / dim_size\n\
         \x20   div.u32       %r8, %r5, %r1;           // row = idx / dim_size\n\
         \x20   // Load index: indices[idx]\n\
         \x20   mul.wide.u32  %rd3, %r5, 4;            // byte offset for indices (u32)\n\
         \x20   add.u64       %rd4, %rd1, %rd3;\n\
         \x20   ld.global.u32 %r9, [%rd4];             // gathered_col = indices[idx]\n\
         \x20   // Compute source offset: row * dim_size + gathered_col\n\
         \x20   mul.lo.u32    %r10, %r8, %r1;          // row * dim_size\n\
         \x20   add.u32       %r10, %r10, %r9;         // + gathered_col\n\
         \x20   mul.wide.u32  %rd5, %r10, 4;           // byte offset\n\
         \x20   add.u64       %rd6, %rd0, %rd5;\n\
         \x20   ld.global.f32 %f0, [%rd6];             // val = data[row * dim_size + gathered_col]\n\
         \x20   // Store to output[idx]\n\
         \x20   mul.wide.u32  %rd7, %r5, 4;\n\
         \x20   add.u64       %rd8, %rd2, %rd7;\n\
         \x20   st.global.f32 [%rd8], %f0;\n\
         \x20   add.u32       %r5, %r5, %r7;           // idx += grid_stride\n\
         \x20   bra           GATHER_LOOP;\n\
         GATHER_EXIT:\n\
         \x20   ret;\n\
         }\n");

    ptx
}

// ---------------------------------------------------------------------------
// Scatter-add
// ---------------------------------------------------------------------------

/// Generate PTX for scatter-add along a dimension.
///
/// `output[(i / dim_size) * dim_size + indices[i]] += src[i]`
///
/// Uses `atom.global.add.f32` for correctness with duplicate indices.
///
/// # Arguments
/// * `n` -- total number of source elements to scatter
/// * `dim` -- dimension along which to scatter (used in comments)
///
/// # Example
/// ```
/// use nn_cuda::ptx_gather::generate_scatter_add_ptx;
/// let ptx = generate_scatter_add_ptx(1024, 0);
/// assert!(ptx.contains(".entry ptx_scatter_add_f32"));
/// assert!(ptx.contains("atom.global.add.f32"));
/// ```
#[must_use]
pub fn generate_scatter_add_ptx(n: u32, dim: u32) -> String {
    let block_size = GATHER_BLOCK_SIZE;

    let mut ptx = String::with_capacity(4096);
    ptx.push_str(&ptx_prelude(SM_TARGET));
    ptx.push_str(&format!(
        "// Scatter-add f32: n={n}, dim={dim}, block_size={block_size}\n\n"
    ));

    ptx.push_str(&format!(
        ".visible .entry ptx_scatter_add_f32(\n\
         \x20   .param .u64 param_src,\n\
         \x20   .param .u64 param_indices,\n\
         \x20   .param .u64 param_output,\n\
         \x20   .param .u32 param_n,\n\
         \x20   .param .u32 param_dim_size\n\
         )\n\
         .reqntid {block_size}\n{{\n"
    ));

    // Registers
    ptx.push_str(
        "\x20   .reg .u32  %r<12>;\n\
         \x20   .reg .f32  %f<2>;\n\
         \x20   .reg .u64  %rd<10>;\n\
         \x20   .reg .pred %p<2>;\n\n",
    );

    // Load parameters
    ptx.push_str(
        "\x20   ld.param.u64  %rd0, [param_src];\n\
         \x20   ld.param.u64  %rd1, [param_indices];\n\
         \x20   ld.param.u64  %rd2, [param_output];\n\
         \x20   ld.param.u32  %r0,  [param_n];\n\
         \x20   ld.param.u32  %r1,  [param_dim_size];\n\n",
    );

    // Global thread index
    ptx.push_str(
        "\x20   mov.u32       %r2, %tid.x;\n\
         \x20   mov.u32       %r3, %ctaid.x;\n\
         \x20   mov.u32       %r4, %ntid.x;\n\
         \x20   mad.lo.u32    %r5, %r3, %r4, %r2;\n\n",
    );

    // Grid-stride loop
    ptx.push_str("\x20   mov.u32       %r6, %nctaid.x;\n\
         \x20   mul.lo.u32    %r7, %r6, %r4;\n\
         SCATTER_LOOP:\n\
         \x20   setp.ge.u32   %p0, %r5, %r0;\n\
         \x20   @%p0 bra      SCATTER_EXIT;\n\
         \x20   // Compute row = idx / dim_size\n\
         \x20   div.u32       %r8, %r5, %r1;\n\
         \x20   // Load src[idx]\n\
         \x20   mul.wide.u32  %rd3, %r5, 4;\n\
         \x20   add.u64       %rd4, %rd0, %rd3;\n\
         \x20   ld.global.f32 %f0, [%rd4];\n\
         \x20   // Load index: indices[idx]\n\
         \x20   add.u64       %rd5, %rd1, %rd3;\n\
         \x20   ld.global.u32 %r9, [%rd5];\n\
         \x20   // Compute target offset: row * dim_size + indices[idx]\n\
         \x20   mul.lo.u32    %r10, %r8, %r1;\n\
         \x20   add.u32       %r10, %r10, %r9;\n\
         \x20   mul.wide.u32  %rd6, %r10, 4;\n\
         \x20   add.u64       %rd7, %rd2, %rd6;\n\
         \x20   // Atomic add: output[target] += src[idx]\n\
         \x20   atom.global.add.f32 %f1, [%rd7], %f0;\n\
         \x20   add.u32       %r5, %r5, %r7;\n\
         \x20   bra           SCATTER_LOOP;\n\
         SCATTER_EXIT:\n\
         \x20   ret;\n\
         }\n");

    ptx
}

// ---------------------------------------------------------------------------
// Launch config
// ---------------------------------------------------------------------------

/// Compute grid and block dimensions for a gather/scatter kernel.
///
/// Grid: `(ceil(n / 256), 1, 1)`.
/// Block: `(256, 1, 1)`.
///
/// # Returns
/// `(grid_dim, block_dim)` as `([x, y, z], [x, y, z])`.
#[must_use]
pub fn ptx_gather_launch_config(n: usize) -> ([usize; 3], [usize; 3]) {
    let bs = GATHER_BLOCK_SIZE as usize;
    let grid_x = n.div_ceil(bs);
    ([grid_x, 1, 1], [bs, 1, 1])
}

// ---------------------------------------------------------------------------
// Reference implementations
// ---------------------------------------------------------------------------

/// CPU reference for gather: `output[i] = data[(i / dim_size) * dim_size + indices[i]]`.
///
/// # Arguments
/// * `data` -- source data buffer
/// * `indices` -- index buffer (u32 values)
/// * `dim_size` -- size of the gather dimension
///
/// # Panics
/// Panics if any computed index is out of bounds for `data`.
#[must_use]
pub fn gather_reference(data: &[f32], indices: &[u32], dim_size: usize) -> Vec<f32> {
    indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| {
            let row = i / dim_size;
            let src_idx = row * dim_size + idx as usize;
            data[src_idx]
        })
        .collect()
}

/// CPU reference for scatter-add: `output[row * dim_size + indices[i]] += src[i]`.
///
/// # Arguments
/// * `src` -- source values
/// * `indices` -- index buffer (u32 values)
/// * `dim_size` -- size of the scatter dimension
/// * `output_len` -- total length of the output buffer
///
/// Returns a zero-initialized output with the scatter-add applied.
///
/// # Panics
/// Panics if any computed index is out of bounds.
#[must_use]
pub fn scatter_add_reference(
    src: &[f32],
    indices: &[u32],
    dim_size: usize,
    output_len: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; output_len];
    for (i, (&val, &idx)) in src.iter().zip(indices.iter()).enumerate() {
        let row = i / dim_size;
        let target = row * dim_size + idx as usize;
        output[target] += val;
    }
    output
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Gather PTX structure --

    #[test]
    fn test_gather_ptx_contains_entry() {
        let ptx = generate_gather_ptx(1024, 0);
        assert!(ptx.contains(".entry ptx_gather_f32"));
        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".target sm_70"));
    }

    #[test]
    fn test_gather_ptx_has_global_load() {
        let ptx = generate_gather_ptx(256, 0);
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("ld.global.u32"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn test_gather_ptx_dim1() {
        let ptx = generate_gather_ptx(512, 1);
        assert!(ptx.contains("dim=1"));
        assert!(ptx.contains("div.u32"));
    }

    // -- Scatter-add PTX structure --

    #[test]
    fn test_scatter_add_ptx_contains_entry() {
        let ptx = generate_scatter_add_ptx(1024, 0);
        assert!(ptx.contains(".entry ptx_scatter_add_f32"));
        assert!(ptx.contains(".target sm_70"));
    }

    #[test]
    fn test_scatter_add_ptx_uses_atomic() {
        let ptx = generate_scatter_add_ptx(256, 0);
        assert!(ptx.contains("atom.global.add.f32"));
    }

    #[test]
    fn test_scatter_add_ptx_dim1() {
        let ptx = generate_scatter_add_ptx(512, 1);
        assert!(ptx.contains("dim=1"));
    }

    // -- Gather reference --

    #[test]
    fn test_gather_reference_1d() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let indices = vec![2, 0, 4, 1];
        // 1-D gather: dim_size = data.len(), all elements in one row
        let result = gather_reference(&data, &indices, data.len());
        assert_eq!(result, vec![30.0, 10.0, 50.0, 20.0]);
    }

    #[test]
    fn test_gather_reference_2d() {
        // 2 rows of 3 elements: [[10, 20, 30], [40, 50, 60]]
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let indices = vec![2, 0, 1, 1, 2, 0]; // 2 rows of 3 indices
        let result = gather_reference(&data, &indices, 3);
        // row 0: data[0*3+2]=30, data[0*3+0]=10, data[0*3+1]=20
        // row 1: data[1*3+1]=50, data[1*3+2]=60, data[1*3+0]=40
        assert_eq!(result, vec![30.0, 10.0, 20.0, 50.0, 60.0, 40.0]);
    }

    #[test]
    fn test_gather_reference_single() {
        let data = vec![42.0];
        let indices = vec![0];
        let result = gather_reference(&data, &indices, 1);
        assert_eq!(result, vec![42.0]);
    }

    // -- Scatter-add reference --

    #[test]
    fn test_scatter_add_reference_basic() {
        let src = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 2, 1];
        let result = scatter_add_reference(&src, &indices, 3, 3);
        assert_eq!(result, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_scatter_add_reference_duplicate_indices() {
        // Multiple sources scatter to the same target => values add up
        let src = vec![1.0, 2.0, 3.0];
        let indices = vec![0, 0, 0]; // all scatter to index 0
        let result = scatter_add_reference(&src, &indices, 3, 3);
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_scatter_add_reference_2d() {
        // 2 rows of 2: src and output are both [2, 2]
        let src = vec![1.0, 2.0, 3.0, 4.0]; // row0=[1,2], row1=[3,4]
        let indices = vec![1, 0, 0, 1]; // row0: [1,0], row1: [0,1]
        let result = scatter_add_reference(&src, &indices, 2, 4);
        // row 0: out[0*2+1]+=1.0 => out[1]=1.0, out[0*2+0]+=2.0 => out[0]=2.0
        // row 1: out[1*2+0]+=3.0 => out[2]=3.0, out[1*2+1]+=4.0 => out[3]=4.0
        assert_eq!(result, vec![2.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn test_scatter_add_reference_single() {
        let src = vec![5.0];
        let indices = vec![0];
        let result = scatter_add_reference(&src, &indices, 1, 1);
        assert_eq!(result, vec![5.0]);
    }

    // -- Launch config --

    #[test]
    fn test_gather_launch_config() {
        let (grid, block) = ptx_gather_launch_config(1024);
        assert_eq!(block, [256, 1, 1]);
        assert_eq!(grid, [4, 1, 1]);
    }

    #[test]
    fn test_gather_launch_config_non_aligned() {
        let (grid, block) = ptx_gather_launch_config(300);
        assert_eq!(block, [256, 1, 1]);
        assert_eq!(grid, [2, 1, 1]); // ceil(300 / 256) = 2
    }
}
