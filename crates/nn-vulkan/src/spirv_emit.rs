// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation helpers for Vulkan compute shaders.
//!
//! Generates SPIR-V binary modules (not GLSL text) for compute shader
//! dispatch. Two emission paths:
//!
//! - **GLSL compute shader strings**: Human-readable GLSL source compiled
//!   to SPIR-V via `glslangValidator` or `shaderc`. Used for prototyping
//!   and when readability matters (activation kernels, simple elementwise).
//!
//! - **Direct SPIR-V binary**: Programmatic construction of SPIR-V word
//!   streams for kernels where precise control is needed (matmul tiling,
//!   subgroup operations, shared memory layouts). Avoids the GLSL compiler
//!   as an intermediary.
//!
//! Both paths produce `Vec<u32>` (SPIR-V word stream) consumed by
//! [`super::dispatch::VulkanDispatcher`].

use crate::error::VulkanError;
use nn_dsl::ScalarType;

/// SPIR-V magic number (little-endian).
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// SPIR-V version 1.5 (Vulkan 1.2+).
pub const SPIRV_VERSION_1_5: u32 = 0x0001_0500;

/// Default workgroup size for compute shaders.
pub const DEFAULT_WORKGROUP_SIZE: u32 = 256;

/// GLSL version header for compute shaders.
pub const GLSL_COMPUTE_VERSION: &str = "#version 450\n";

/// Map `ScalarType` to GLSL type name.
///
/// SPIR-V compute shaders use GLSL-compatible types: `float` for f32,
/// and `float16_t` (via `GL_EXT_shader_explicit_arithmetic_types_float16`)
/// for f16.
pub fn glsl_type(dtype: ScalarType) -> Result<&'static str, VulkanError> {
    match dtype {
        ScalarType::F32 => Ok("float"),
        ScalarType::F16 => Ok("float16_t"),
        ScalarType::BF16 => {
            // bf16 has no native Vulkan/GLSL support in most implementations.
            // Emulate via uint16 + bitcast when needed.
            Err(VulkanError::UnsupportedType {
                type_desc: "bf16 — no native GLSL type, requires uint16 emulation",
            })
        }
        _ => Err(VulkanError::UnsupportedType {
            type_desc: "non-float ScalarType",
        }),
    }
}

/// Map `ScalarType` to byte size.
pub fn spirv_type_bytes(dtype: ScalarType) -> Result<usize, VulkanError> {
    match dtype {
        ScalarType::F32 => Ok(4),
        ScalarType::F16 | ScalarType::BF16 => Ok(2),
        _ => Err(VulkanError::UnsupportedType {
            type_desc: "non-float ScalarType",
        }),
    }
}

/// Emit a GLSL compute shader for an elementwise unary operation.
///
/// Generates a GLSL 450 compute shader that reads from `input_buffer`,
/// applies `op_glsl_expr` (where `x` is the input value), and writes to
/// `output_buffer`.
///
/// The result is a GLSL source string suitable for compilation to SPIR-V
/// via `glslangValidator` or `shaderc`.
///
/// # Arguments
///
/// * `kernel_name` — Used in comments only (GLSL entry point is always `main`).
/// * `op_glsl_expr` — GLSL expression using `x` as input (e.g., `"max(x, 0.0)"`).
/// * `workgroup_size` — Local workgroup size in X dimension.
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_emit::emit_elementwise_glsl;
/// let glsl = emit_elementwise_glsl("relu", "max(x, 0.0)", 256).unwrap();
/// assert!(glsl.contains("layout(local_size_x ="));
/// assert!(glsl.contains("max(x, 0.0)"));
/// ```
pub fn emit_elementwise_glsl(
    kernel_name: &str,
    op_glsl_expr: &str,
    workgroup_size: u32,
) -> Result<String, VulkanError> {
    if workgroup_size == 0 {
        return Err(VulkanError::InvalidParameter(
            "workgroup_size must be > 0".into(),
        ));
    }

    let mut src = String::with_capacity(512);
    src.push_str(GLSL_COMPUTE_VERSION);
    src.push_str(&format!("// Kernel: {kernel_name}\n\n"));
    src.push_str(&format!(
        "layout(local_size_x = {workgroup_size}, local_size_y = 1, local_size_z = 1) in;\n\n"
    ));
    src.push_str(
        "layout(std430, set = 0, binding = 0) readonly buffer InputBuf {\n\
         \x20   float data[];\n\
         } input_buffer;\n\n\
         layout(std430, set = 0, binding = 1) writeonly buffer OutputBuf {\n\
         \x20   float data[];\n\
         } output_buffer;\n\n\
         layout(push_constant) uniform PushConstants {\n\
         \x20   uint total_elements;\n\
         } params;\n\n\
         void main() {\n\
         \x20   uint idx = gl_GlobalInvocationID.x;\n\
         \x20   if (idx >= params.total_elements) return;\n\
         \x20   float x = input_buffer.data[idx];\n",
    );
    src.push_str(&format!(
        "\x20   output_buffer.data[idx] = {op_glsl_expr};\n}}\n"
    ));
    Ok(src)
}

/// Emit a GLSL compute shader for a reduction operation along the last axis.
///
/// Generates a shader that performs a parallel reduction within a workgroup
/// using shared memory. Each workgroup reduces one row of the input.
///
/// # Arguments
///
/// * `kernel_name` — Used in comments.
/// * `op` — Reduction operation: `"add"`, `"max"`, or `"min"`.
/// * `workgroup_size` — Local workgroup size (should be power of 2).
pub fn emit_reduction_glsl(
    kernel_name: &str,
    op: ReductionOp,
    workgroup_size: u32,
) -> Result<String, VulkanError> {
    if workgroup_size == 0 || !workgroup_size.is_power_of_two() {
        return Err(VulkanError::InvalidParameter(
            "workgroup_size must be a power of 2".into(),
        ));
    }

    let (identity, combine) = match op {
        ReductionOp::Sum => ("0.0", "a + b"),
        ReductionOp::Max => ("-1.0 / 0.0", "max(a, b)"),
        ReductionOp::Min => ("1.0 / 0.0", "min(a, b)"),
    };

    let mut src = String::with_capacity(1024);
    src.push_str(GLSL_COMPUTE_VERSION);
    src.push_str(&format!("// Reduction kernel: {kernel_name} ({op:?})\n\n"));
    src.push_str(&format!(
        "layout(local_size_x = {workgroup_size}, local_size_y = 1, local_size_z = 1) in;\n\n"
    ));
    src.push_str(&format!(
        "shared float sdata[{workgroup_size}];\n\n\
         layout(std430, set = 0, binding = 0) readonly buffer InputBuf {{\n\
         \x20   float data[];\n\
         }} input_buffer;\n\n\
         layout(std430, set = 0, binding = 1) writeonly buffer OutputBuf {{\n\
         \x20   float data[];\n\
         }} output_buffer;\n\n\
         layout(push_constant) uniform PushConstants {{\n\
         \x20   uint row_size;\n\
         \x20   uint num_rows;\n\
         }} params;\n\n\
         void main() {{\n\
         \x20   uint row = gl_WorkGroupID.x;\n\
         \x20   uint tid = gl_LocalInvocationID.x;\n\
         \x20   if (row >= params.num_rows) return;\n\
         \x20   uint base = row * params.row_size;\n\
         \x20   // Load phase: each thread accumulates a stripe\n\
         \x20   float acc = {identity};\n\
         \x20   for (uint i = tid; i < params.row_size; i += {workgroup_size}) {{\n\
         \x20       float a = acc;\n\
         \x20       float b = input_buffer.data[base + i];\n\
         \x20       acc = {combine};\n\
         \x20   }}\n\
         \x20   sdata[tid] = acc;\n\
         \x20   barrier();\n\
         \x20   // Tree reduction in shared memory\n"
    ));

    let mut stride = workgroup_size / 2;
    while stride > 0 {
        src.push_str(&format!(
            "\x20   if (tid < {stride}) {{\n\
             \x20       float a = sdata[tid];\n\
             \x20       float b = sdata[tid + {stride}];\n\
             \x20       sdata[tid] = {combine};\n\
             \x20   }}\n\
             \x20   barrier();\n"
        ));
        stride /= 2;
    }

    src.push_str(
        "\x20   if (tid == 0) {\n\
         \x20       output_buffer.data[row] = sdata[0];\n\
         \x20   }\n\
         }\n",
    );
    Ok(src)
}

/// Emit a GLSL compute shader for tiled matrix multiplication.
///
/// Generates a `C = A * B` kernel using shared-memory tiling. Each workgroup
/// computes a `TILE x TILE` block of the output matrix.
///
/// # Arguments
///
/// * `tile_size` — Tile dimension (both M and N tile). Must be power of 2.
pub fn emit_matmul_glsl(tile_size: u32) -> Result<String, VulkanError> {
    if tile_size == 0 || !tile_size.is_power_of_two() {
        return Err(VulkanError::InvalidParameter(
            "tile_size must be a power of 2".into(),
        ));
    }

    let mut src = String::with_capacity(2048);
    src.push_str(GLSL_COMPUTE_VERSION);
    src.push_str(&format!(
        "// Tiled matmul: C[M,N] = A[M,K] * B[K,N], tile={tile_size}\n\n"
    ));
    src.push_str(&format!(
        "layout(local_size_x = {tile_size}, local_size_y = {tile_size}, local_size_z = 1) in;\n\n"
    ));
    src.push_str(&format!(
        "shared float tileA[{tile_size}][{tile_size}];\n\
         shared float tileB[{tile_size}][{tile_size}];\n\n\
         layout(std430, set = 0, binding = 0) readonly buffer ABuf {{\n\
         \x20   float data[];\n\
         }} A;\n\n\
         layout(std430, set = 0, binding = 1) readonly buffer BBuf {{\n\
         \x20   float data[];\n\
         }} B;\n\n\
         layout(std430, set = 0, binding = 2) writeonly buffer CBuf {{\n\
         \x20   float data[];\n\
         }} C;\n\n\
         layout(push_constant) uniform PushConstants {{\n\
         \x20   uint M;\n\
         \x20   uint N;\n\
         \x20   uint K;\n\
         }} params;\n\n\
         void main() {{\n\
         \x20   uint row = gl_WorkGroupID.y * {tile_size} + gl_LocalInvocationID.y;\n\
         \x20   uint col = gl_WorkGroupID.x * {tile_size} + gl_LocalInvocationID.x;\n\
         \x20   float acc = 0.0;\n\
         \x20   uint numTiles = (params.K + {tile_size} - 1) / {tile_size};\n\
         \x20   for (uint t = 0; t < numTiles; t++) {{\n\
         \x20       uint tiledCol = t * {tile_size} + gl_LocalInvocationID.x;\n\
         \x20       uint tiledRow = t * {tile_size} + gl_LocalInvocationID.y;\n\
         \x20       tileA[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =\n\
         \x20           (row < params.M && tiledCol < params.K)\n\
         \x20               ? A.data[row * params.K + tiledCol] : 0.0;\n\
         \x20       tileB[gl_LocalInvocationID.y][gl_LocalInvocationID.x] =\n\
         \x20           (tiledRow < params.K && col < params.N)\n\
         \x20               ? B.data[tiledRow * params.N + col] : 0.0;\n\
         \x20       barrier();\n\
         \x20       for (uint k = 0; k < {tile_size}; k++) {{\n\
         \x20           acc += tileA[gl_LocalInvocationID.y][k] * tileB[k][gl_LocalInvocationID.x];\n\
         \x20       }}\n\
         \x20       barrier();\n\
         \x20   }}\n\
         \x20   if (row < params.M && col < params.N) {{\n\
         \x20       C.data[row * params.N + col] = acc;\n\
         \x20   }}\n\
         }}\n"
    ));
    Ok(src)
}

/// Emit a GLSL compute shader for softmax along the last axis.
///
/// Two-pass approach: first pass computes row-wise max, second pass
/// computes exp(x - max) / sum(exp(x - max)). Both passes are fused
/// into a single shader using shared memory.
pub fn emit_softmax_glsl(workgroup_size: u32) -> Result<String, VulkanError> {
    if workgroup_size == 0 || !workgroup_size.is_power_of_two() {
        return Err(VulkanError::InvalidParameter(
            "workgroup_size must be a power of 2".into(),
        ));
    }

    let mut src = String::with_capacity(2048);
    src.push_str(GLSL_COMPUTE_VERSION);
    src.push_str(&format!(
        "// Softmax along last axis (fused max + exp + sum + normalize)\n\n\
         layout(local_size_x = {workgroup_size}, local_size_y = 1, local_size_z = 1) in;\n\n\
         shared float smax[{workgroup_size}];\n\
         shared float ssum[{workgroup_size}];\n\n\
         layout(std430, set = 0, binding = 0) readonly buffer InputBuf {{\n\
         \x20   float data[];\n\
         }} input_buffer;\n\n\
         layout(std430, set = 0, binding = 1) writeonly buffer OutputBuf {{\n\
         \x20   float data[];\n\
         }} output_buffer;\n\n\
         layout(push_constant) uniform PushConstants {{\n\
         \x20   uint row_size;\n\
         \x20   uint num_rows;\n\
         }} params;\n\n\
         void main() {{\n\
         \x20   uint row = gl_WorkGroupID.x;\n\
         \x20   uint tid = gl_LocalInvocationID.x;\n\
         \x20   if (row >= params.num_rows) return;\n\
         \x20   uint base = row * params.row_size;\n\
         \x20   // Pass 1: find row max\n\
         \x20   float local_max = -1.0 / 0.0;\n\
         \x20   for (uint i = tid; i < params.row_size; i += {workgroup_size}) {{\n\
         \x20       local_max = max(local_max, input_buffer.data[base + i]);\n\
         \x20   }}\n\
         \x20   smax[tid] = local_max;\n\
         \x20   barrier();\n"
    ));

    // Tree reduction for max
    let mut stride = workgroup_size / 2;
    while stride > 0 {
        src.push_str(&format!(
            "\x20   if (tid < {stride}) smax[tid] = max(smax[tid], smax[tid + {stride}]);\n\
             \x20   barrier();\n"
        ));
        stride /= 2;
    }

    src.push_str(&format!(
        "\x20   float row_max = smax[0];\n\
         \x20   // Pass 2: compute exp(x - max) and sum\n\
         \x20   float local_sum = 0.0;\n\
         \x20   for (uint i = tid; i < params.row_size; i += {workgroup_size}) {{\n\
         \x20       local_sum += exp(input_buffer.data[base + i] - row_max);\n\
         \x20   }}\n\
         \x20   ssum[tid] = local_sum;\n\
         \x20   barrier();\n"
    ));

    // Tree reduction for sum
    stride = workgroup_size / 2;
    while stride > 0 {
        src.push_str(&format!(
            "\x20   if (tid < {stride}) ssum[tid] = ssum[tid] + ssum[tid + {stride}];\n\
             \x20   barrier();\n"
        ));
        stride /= 2;
    }

    src.push_str(&format!(
        "\x20   float row_sum = ssum[0];\n\
         \x20   // Pass 3: normalize\n\
         \x20   for (uint i = tid; i < params.row_size; i += {workgroup_size}) {{\n\
         \x20       output_buffer.data[base + i] = exp(input_buffer.data[base + i] - row_max) / row_sum;\n\
         \x20   }}\n\
         }}\n"
    ));
    Ok(src)
}

/// Reduction operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Summation.
    Sum,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}
