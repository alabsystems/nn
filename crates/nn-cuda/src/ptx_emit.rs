// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CUDA C++ kernel emission for NVIDIA GPUs.
//!
//! Generates CUDA C++ kernel source for common ML operations (elementwise,
//! reduction, softmax, tiled matmul). Compiled via `nvcc` to PTX.
//!
//! For raw PTX assembly matmul generation (no `nvcc` needed), see
//! [`ptx_matmul`](super::ptx_matmul).

use crate::codegen_ptx::{PtxCodegenError, PTX_BLOCK_SIZE, REDUCE_BLOCK_SIZE};

/// CUDA C++ prelude for generated kernel files.
///
/// Includes CUDA runtime headers and bf16/f16 support.
pub const CUDA_PRELUDE: &str = "\
#include <cuda_runtime.h>\n\
#include <cuda_fp16.h>\n\
#include <cuda_bf16.h>\n\n";

/// Emit a CUDA C++ elementwise kernel.
///
/// Generates a `__global__` function that applies a scalar operation to each
/// element of the input buffer. The operation is specified as a CUDA C++
/// expression string where `x` is the input element.
///
/// # Arguments
///
/// * `kernel_name` — Name for the `__global__` function.
/// * `op_expr` — CUDA C++ expression for the scalar operation (e.g., `"x > 0.0f ? x : 0.0f"`).
/// * `total_elements` — Number of elements in the input buffer.
///
/// # Example
///
/// ```
/// use nn_cuda::ptx_emit::emit_elementwise_kernel;
/// let src = emit_elementwise_kernel("relu_kernel", "x > 0.0f ? x : 0.0f", 1024).unwrap();
/// assert!(src.contains("__global__"));
/// assert!(src.contains("relu_kernel"));
/// ```
pub fn emit_elementwise_kernel(
    kernel_name: &str,
    op_expr: &str,
    total_elements: usize,
) -> Result<String, PtxCodegenError> {
    if total_elements == 0 {
        return Err(PtxCodegenError::InvalidParameter(
            "total_elements must be > 0".into(),
        ));
    }

    let mut src = String::with_capacity(512);
    src.push_str(CUDA_PRELUDE);
    src.push_str(&format!(
        "__global__ void {kernel_name}(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int N\n\
         ) {{\n\
         \x20   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (idx >= N) return;\n\
         \x20   float x = input[idx];\n\
         \x20   output[idx] = {op_expr};\n\
         }}\n"
    ));
    Ok(src)
}

/// Emit CUDA C++ kernels for common activations.
///
/// Returns a single CUDA C++ source file containing `__global__` kernels for
/// relu, silu, sigmoid, tanh, and gelu.
pub fn emit_activation_kernels() -> String {
    let mut src = String::with_capacity(2048);
    src.push_str(CUDA_PRELUDE);

    // ReLU: max(0, x)
    src.push_str(
        "__global__ void relu_kernel(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int N\n\
         ) {\n\
         \x20   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (idx >= N) return;\n\
         \x20   float x = input[idx];\n\
         \x20   output[idx] = x > 0.0f ? x : 0.0f;\n\
         }\n\n",
    );

    // SiLU (Swish): x * sigmoid(x)
    src.push_str(
        "__global__ void silu_kernel(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int N\n\
         ) {\n\
         \x20   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (idx >= N) return;\n\
         \x20   float x = input[idx];\n\
         \x20   output[idx] = x / (1.0f + expf(-x));\n\
         }\n\n",
    );

    // Sigmoid: 1 / (1 + exp(-x))
    src.push_str(
        "__global__ void sigmoid_kernel(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int N\n\
         ) {\n\
         \x20   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (idx >= N) return;\n\
         \x20   float x = input[idx];\n\
         \x20   output[idx] = 1.0f / (1.0f + expf(-x));\n\
         }\n\n",
    );

    // Tanh
    src.push_str(
        "__global__ void tanh_kernel(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int N\n\
         ) {\n\
         \x20   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (idx >= N) return;\n\
         \x20   output[idx] = tanhf(input[idx]);\n\
         }\n\n",
    );

    // GELU (approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    src.push_str(
        "__global__ void gelu_kernel(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int N\n\
         ) {\n\
         \x20   unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;\n\
         \x20   if (idx >= N) return;\n\
         \x20   float x = input[idx];\n\
         \x20   float c = 0.7978845608f; // sqrt(2/pi)\n\
         \x20   float inner = c * (x + 0.044715f * x * x * x);\n\
         \x20   output[idx] = 0.5f * x * (1.0f + tanhf(inner));\n\
         }\n\n",
    );

    src
}

/// Emit a CUDA C++ softmax kernel (numerically stable, 3-phase).
///
/// Phase 1: find max (shared memory reduction).
/// Phase 2: compute exp(x - max) and sum (shared memory reduction).
/// Phase 3: normalize by sum.
///
/// Operates along the last axis. Each block processes one row.
pub fn emit_softmax_kernel(row_size: usize) -> Result<String, PtxCodegenError> {
    if row_size == 0 {
        return Err(PtxCodegenError::InvalidParameter(
            "row_size must be > 0".into(),
        ));
    }

    let block_size = REDUCE_BLOCK_SIZE.min(row_size.next_power_of_two());

    let mut src = String::with_capacity(2048);
    src.push_str(CUDA_PRELUDE);
    src.push_str(&format!(
        "// Softmax kernel: last-axis, row_size={row_size}\n\
         __global__ void softmax_kernel(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int row_size\n\
         ) {{\n\
         \x20   extern __shared__ float sdata[];\n\
         \x20   unsigned int tid = threadIdx.x;\n\
         \x20   unsigned int row = blockIdx.x;\n\
         \x20   const float* row_in = input + row * row_size;\n\
         \x20   float* row_out = output + row * row_size;\n\
         \n\
         \x20   // Phase 1: find max\n\
         \x20   float max_val = -HUGE_VALF;\n\
         \x20   for (unsigned int i = tid; i < row_size; i += {block_size}) {{\n\
         \x20       float v = row_in[i];\n\
         \x20       if (v > max_val) max_val = v;\n\
         \x20   }}\n\
         \x20   sdata[tid] = max_val;\n\
         \x20   __syncthreads();\n\
         \x20   for (unsigned int s = {block_size} / 2; s > 0; s >>= 1) {{\n\
         \x20       if (tid < s && sdata[tid + s] > sdata[tid])\n\
         \x20           sdata[tid] = sdata[tid + s];\n\
         \x20       __syncthreads();\n\
         \x20   }}\n\
         \x20   max_val = sdata[0];\n\
         \x20   __syncthreads();\n\
         \n\
         \x20   // Phase 2: exp(x - max) and sum\n\
         \x20   float sum_val = 0.0f;\n\
         \x20   for (unsigned int i = tid; i < row_size; i += {block_size}) {{\n\
         \x20       float v = expf(row_in[i] - max_val);\n\
         \x20       row_out[i] = v;\n\
         \x20       sum_val += v;\n\
         \x20   }}\n\
         \x20   sdata[tid] = sum_val;\n\
         \x20   __syncthreads();\n\
         \x20   for (unsigned int s = {block_size} / 2; s > 0; s >>= 1) {{\n\
         \x20       if (tid < s) sdata[tid] += sdata[tid + s];\n\
         \x20       __syncthreads();\n\
         \x20   }}\n\
         \x20   sum_val = sdata[0];\n\
         \x20   __syncthreads();\n\
         \n\
         \x20   // Phase 3: normalize\n\
         \x20   float inv_sum = 1.0f / sum_val;\n\
         \x20   for (unsigned int i = tid; i < row_size; i += {block_size}) {{\n\
         \x20       row_out[i] *= inv_sum;\n\
         \x20   }}\n\
         }}\n"
    ));
    Ok(src)
}

/// Emit a CUDA C++ reduction kernel (sum, max, min, mean).
///
/// Uses shared-memory tree reduction within each block.
/// For multi-block reductions, a second pass is needed (not emitted here).
pub fn emit_reduction_kernel(
    kernel_name: &str,
    op: ReductionOp,
    axis_size: usize,
) -> Result<String, PtxCodegenError> {
    if axis_size == 0 {
        return Err(PtxCodegenError::InvalidParameter(
            "axis_size must be > 0".into(),
        ));
    }

    let block_size = REDUCE_BLOCK_SIZE.min(axis_size.next_power_of_two());
    let (identity, combine, finalize) = match op {
        ReductionOp::Sum => ("0.0f", "sdata[tid] += sdata[tid + s];", ""),
        ReductionOp::Max => (
            "-HUGE_VALF",
            "if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];",
            "",
        ),
        ReductionOp::Min => (
            "HUGE_VALF",
            "if (sdata[tid + s] < sdata[tid]) sdata[tid] = sdata[tid + s];",
            "",
        ),
        ReductionOp::Mean => (
            "0.0f",
            "sdata[tid] += sdata[tid + s];",
            "    if (tid == 0) output[row] = sdata[0] / (float)axis_size;\n",
        ),
    };

    let write_output = if matches!(op, ReductionOp::Mean) {
        finalize.to_string()
    } else {
        "    if (tid == 0) output[row] = sdata[0];\n".to_string()
    };

    let mut src = String::with_capacity(1024);
    src.push_str(CUDA_PRELUDE);
    src.push_str(&format!(
        "// Reduction kernel: {op:?}, axis_size={axis_size}\n\
         __global__ void {kernel_name}(\n\
         \x20   const float* __restrict__ input,\n\
         \x20   float* __restrict__ output,\n\
         \x20   const unsigned int axis_size\n\
         ) {{\n\
         \x20   extern __shared__ float sdata[];\n\
         \x20   unsigned int tid = threadIdx.x;\n\
         \x20   unsigned int row = blockIdx.x;\n\
         \x20   const float* row_in = input + row * axis_size;\n\
         \n\
         \x20   float val = {identity};\n\
         \x20   for (unsigned int i = tid; i < axis_size; i += {block_size}) {{\n"
    ));

    match op {
        ReductionOp::Sum | ReductionOp::Mean => {
            src.push_str("        val += row_in[i];\n");
        }
        ReductionOp::Max => {
            src.push_str(
                "        float v = row_in[i];\n\
                 \x20       if (v > val) val = v;\n",
            );
        }
        ReductionOp::Min => {
            src.push_str(
                "        float v = row_in[i];\n\
                 \x20       if (v < val) val = v;\n",
            );
        }
    }

    src.push_str(&format!(
        "    }}\n\
         \x20   sdata[tid] = val;\n\
         \x20   __syncthreads();\n\
         \x20   for (unsigned int s = {block_size} / 2; s > 0; s >>= 1) {{\n\
         \x20       if (tid < s) {combine}\n\
         \x20       __syncthreads();\n\
         \x20   }}\n\
         {write_output}\
         }}\n"
    ));
    Ok(src)
}

/// Emit a CUDA C++ tiled matrix multiplication kernel.
///
/// Uses shared memory tiling for cache efficiency. Each block computes a
/// `TILE_SIZE x TILE_SIZE` output tile.
///
/// C[M, N] = A[M, K] * B[K, N]
pub fn emit_matmul_kernel(kernel_name: &str, tile_size: usize) -> Result<String, PtxCodegenError> {
    if tile_size == 0 || tile_size > 32 {
        return Err(PtxCodegenError::InvalidParameter(format!(
            "tile_size must be 1..=32, got {tile_size}"
        )));
    }

    let mut src = String::with_capacity(2048);
    src.push_str(CUDA_PRELUDE);
    src.push_str(&format!(
        "#define TILE_SIZE {tile_size}\n\n\
         // Tiled GEMM: C[M,N] = A[M,K] * B[K,N]\n\
         __global__ void {kernel_name}(\n\
         \x20   const float* __restrict__ A,\n\
         \x20   const float* __restrict__ B,\n\
         \x20   float* __restrict__ C,\n\
         \x20   const unsigned int M,\n\
         \x20   const unsigned int N,\n\
         \x20   const unsigned int K\n\
         ) {{\n\
         \x20   __shared__ float As[TILE_SIZE][TILE_SIZE];\n\
         \x20   __shared__ float Bs[TILE_SIZE][TILE_SIZE];\n\
         \n\
         \x20   unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;\n\
         \x20   unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;\n\
         \x20   float acc = 0.0f;\n\
         \n\
         \x20   for (unsigned int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {{\n\
         \x20       unsigned int a_col = t * TILE_SIZE + threadIdx.x;\n\
         \x20       unsigned int b_row = t * TILE_SIZE + threadIdx.y;\n\
         \n\
         \x20       As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)\n\
         \x20           ? A[row * K + a_col] : 0.0f;\n\
         \x20       Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)\n\
         \x20           ? B[b_row * N + col] : 0.0f;\n\
         \x20       __syncthreads();\n\
         \n\
         \x20       for (unsigned int i = 0; i < TILE_SIZE; i++) {{\n\
         \x20           acc += As[threadIdx.y][i] * Bs[i][threadIdx.x];\n\
         \x20       }}\n\
         \x20       __syncthreads();\n\
         \x20   }}\n\
         \n\
         \x20   if (row < M && col < N) {{\n\
         \x20       C[row * N + col] = acc;\n\
         \x20   }}\n\
         }}\n\n\
         #undef TILE_SIZE\n"
    ));
    Ok(src)
}

/// Reduction operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    Sum,
    Max,
    Min,
    Mean,
}

/// Compute the launch configuration for an elementwise kernel.
#[must_use]
pub fn elementwise_launch_config(total_elements: usize) -> (usize, usize) {
    let block_size = PTX_BLOCK_SIZE;
    let grid_size = total_elements.div_ceil(block_size);
    (grid_size, block_size)
}

/// Compute the launch configuration for a softmax/reduction kernel.
///
/// One block per row. Block size is min(REDUCE_BLOCK_SIZE, next_power_of_two(row_size)).
#[must_use]
pub fn reduction_launch_config(num_rows: usize, row_size: usize) -> (usize, usize) {
    let block_size = REDUCE_BLOCK_SIZE.min(row_size.next_power_of_two());
    (num_rows, block_size)
}

/// Compute the launch configuration for a tiled matmul kernel.
///
/// Grid: `(ceil(N/tile), ceil(M/tile))`. Block: `(tile, tile)`.
#[must_use]
pub fn matmul_launch_config(m: usize, n: usize, tile_size: usize) -> ([usize; 2], [usize; 2]) {
    let grid = [n.div_ceil(tile_size), m.div_ceil(tile_size)];
    let block = [tile_size, tile_size];
    (grid, block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emit_elementwise_kernel_relu() {
        let src = emit_elementwise_kernel("relu_kernel", "x > 0.0f ? x : 0.0f", 1024).unwrap();
        assert!(src.contains("__global__"));
        assert!(src.contains("relu_kernel"));
        assert!(src.contains("x > 0.0f ? x : 0.0f"));
        assert!(src.contains("#include <cuda_runtime.h>"));
    }

    #[test]
    fn test_emit_elementwise_kernel_zero_elements_rejected() {
        let result = emit_elementwise_kernel("k", "x", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_emit_activation_kernels() {
        let src = emit_activation_kernels();
        assert!(src.contains("relu_kernel"));
        assert!(src.contains("silu_kernel"));
        assert!(src.contains("sigmoid_kernel"));
        assert!(src.contains("tanh_kernel"));
        assert!(src.contains("gelu_kernel"));
        // Each kernel has __global__
        assert_eq!(src.matches("__global__").count(), 5);
    }

    #[test]
    fn test_emit_softmax_kernel() {
        let src = emit_softmax_kernel(512).unwrap();
        assert!(src.contains("softmax_kernel"));
        assert!(src.contains("__shared__"));
        assert!(src.contains("__syncthreads"));
        assert!(src.contains("expf"));
    }

    #[test]
    fn test_emit_reduction_kernel_sum() {
        let src = emit_reduction_kernel("sum_kernel", ReductionOp::Sum, 1024).unwrap();
        assert!(src.contains("sum_kernel"));
        assert!(src.contains("__shared__"));
        assert!(src.contains("sdata[tid] +="));
    }

    #[test]
    fn test_emit_reduction_kernel_max() {
        let src = emit_reduction_kernel("max_kernel", ReductionOp::Max, 256).unwrap();
        assert!(src.contains("max_kernel"));
        assert!(src.contains("-HUGE_VALF"));
    }

    #[test]
    fn test_emit_reduction_kernel_mean() {
        let src = emit_reduction_kernel("mean_kernel", ReductionOp::Mean, 128).unwrap();
        assert!(src.contains("(float)axis_size"));
    }

    #[test]
    fn test_emit_matmul_kernel() {
        let src = emit_matmul_kernel("gemm_kernel", 16).unwrap();
        assert!(src.contains("gemm_kernel"));
        assert!(src.contains("__shared__"));
        assert!(src.contains("TILE_SIZE"));
        assert!(src.contains("#define TILE_SIZE 16"));
    }

    #[test]
    fn test_emit_matmul_kernel_invalid_tile_rejected() {
        assert!(emit_matmul_kernel("k", 0).is_err());
        assert!(emit_matmul_kernel("k", 64).is_err());
    }

    #[test]
    fn test_elementwise_launch_config() {
        let (grid, block) = elementwise_launch_config(1024);
        assert_eq!(block, 256);
        assert_eq!(grid, 4);
    }

    #[test]
    fn test_elementwise_launch_config_not_multiple() {
        let (grid, block) = elementwise_launch_config(1000);
        assert_eq!(block, 256);
        assert_eq!(grid, 4); // ceil(1000/256) = 4
    }

    #[test]
    fn test_reduction_launch_config() {
        let (num_blocks, block_size) = reduction_launch_config(32, 512);
        assert_eq!(num_blocks, 32);
        assert_eq!(block_size, 256); // min(256, 512)
    }

    #[test]
    fn test_matmul_launch_config() {
        let (grid, block) = matmul_launch_config(128, 64, 16);
        assert_eq!(grid, [4, 8]); // [ceil(64/16), ceil(128/16)]
        assert_eq!(block, [16, 16]);
    }
}
