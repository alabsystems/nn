// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Safe Vulkan compute pipeline abstraction for dispatching SPIR-V compute shaders.
//!
//! This module provides a high-level API for loading SPIR-V binaries (raw bytes)
//! and dispatching them on a Vulkan device. It complements the existing
//! [`super::dispatch`] module (which works with SPIR-V word streams `Vec<u32>`)
//! by operating on raw byte buffers — the format produced by offline SPIR-V
//! compilers and file I/O.
//!
//! # Architecture
//!
//! ```text
//! SPIR-V bytes (Vec<u8>)
//!   → CompiledShader::new()       — validates magic, stores metadata
//!   → validate_dispatch()         — checks bindings, push constants, grid
//!   → (future) submit to VulkanDevice for GPU execution
//! ```
//!
//! # Relationship to existing modules
//!
//! - [`spirv_binary`](super::spirv_binary), [`spirv_matmul`](super::spirv_matmul),
//!   [`spirv_reduction`](super::spirv_reduction): Generate `Vec<u32>` word streams.
//!   Convert to bytes via `spirv_words_to_bytes()` before passing to [`CompiledShader`].
//! - [`dispatch::ComputePipeline`](super::dispatch::ComputePipeline): Works with
//!   `&[u32]` word streams directly. This module's [`CompiledShader`] stores raw
//!   bytes and adds richer validation (binding count, push constant size, grid dims).
//! - [`workgroup`](super::workgroup): Provides `workgroup_count_1d` etc. Use
//!   [`compute_grid_dims`] for a quick 1D grid calculation from total elements.

/// SPIR-V magic number as raw bytes (little-endian): `0x07230203`.
const SPIRV_MAGIC_BYTES: [u8; 4] = [0x03, 0x02, 0x23, 0x07];

/// Default maximum buffer size: 256 MiB.
const DEFAULT_MAX_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Default workgroup size (x dimension).
const DEFAULT_WORKGROUP_SIZE_X: u32 = 256;

/// Vulkan guaranteed minimum for `maxComputeWorkGroupInvocations`.
const MAX_WORKGROUP_INVOCATIONS_GUARANTEED: u32 = 128;

/// Vulkan compute pipeline configuration.
#[derive(Debug, Clone)]
pub struct VulkanComputeConfig {
    /// Maximum buffer size in bytes.
    pub max_buffer_size: usize,
    /// Preferred workgroup size (x dimension).
    pub workgroup_size_x: u32,
    /// Preferred workgroup size (y dimension).
    pub workgroup_size_y: u32,
    /// Preferred workgroup size (z dimension).
    pub workgroup_size_z: u32,
    /// Enable validation layers (debug builds).
    pub enable_validation: bool,
}

impl Default for VulkanComputeConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: DEFAULT_MAX_BUFFER_SIZE,
            workgroup_size_x: DEFAULT_WORKGROUP_SIZE_X,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            enable_validation: cfg!(debug_assertions),
        }
    }
}

impl VulkanComputeConfig {
    /// Total workgroup invocations (product of all dimensions).
    #[must_use]
    pub fn total_workgroup_invocations(&self) -> u32 {
        self.workgroup_size_x
            .saturating_mul(self.workgroup_size_y)
            .saturating_mul(self.workgroup_size_z)
    }
}

/// A compiled SPIR-V compute shader ready for dispatch.
#[derive(Debug, Clone)]
pub struct CompiledShader {
    /// SPIR-V binary (raw bytes).
    spirv: Vec<u8>,
    /// Entry point name.
    entry_point: String,
    /// Number of descriptor set bindings (buffers).
    num_bindings: u32,
    /// Push constant size in bytes (0 if none).
    push_constant_size: u32,
    /// Local workgroup size [x, y, z].
    workgroup_size: [u32; 3],
}

impl CompiledShader {
    /// Create a compiled shader from a SPIR-V binary.
    ///
    /// Validates the SPIR-V magic number (`0x07230203`) and that the binary
    /// is at least 20 bytes (5 SPIR-V header words).
    ///
    /// # Arguments
    ///
    /// * `spirv` -- Raw SPIR-V binary bytes.
    /// * `entry_point` -- Shader entry point name (usually `"main"`).
    /// * `num_bindings` -- Number of descriptor set bindings the shader expects.
    /// * `push_constant_size` -- Size of push constant block in bytes (0 if none).
    /// * `workgroup_size` -- Local workgroup size `[x, y, z]`.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanPipelineError::SpirvValidation`] if the binary is too
    /// short or has an invalid magic number.
    pub fn new(
        spirv: Vec<u8>,
        entry_point: &str,
        num_bindings: u32,
        push_constant_size: u32,
        workgroup_size: [u32; 3],
    ) -> Result<Self, VulkanPipelineError> {
        // SPIR-V header is 5 words = 20 bytes minimum.
        if spirv.len() < 20 {
            return Err(VulkanPipelineError::SpirvValidation {
                reason: format!(
                    "SPIR-V binary too short: {} bytes (minimum 20)",
                    spirv.len()
                ),
            });
        }

        // Validate magic number (first 4 bytes, little-endian).
        if spirv[0..4] != SPIRV_MAGIC_BYTES {
            return Err(VulkanPipelineError::SpirvValidation {
                reason: format!(
                    "invalid SPIR-V magic: expected {:02x}{:02x}{:02x}{:02x}, got {:02x}{:02x}{:02x}{:02x}",
                    SPIRV_MAGIC_BYTES[0], SPIRV_MAGIC_BYTES[1],
                    SPIRV_MAGIC_BYTES[2], SPIRV_MAGIC_BYTES[3],
                    spirv[0], spirv[1], spirv[2], spirv[3],
                ),
            });
        }

        // Validate workgroup size dimensions are non-zero.
        if workgroup_size[0] == 0 || workgroup_size[1] == 0 || workgroup_size[2] == 0 {
            return Err(VulkanPipelineError::WorkgroupSizeExceeded {
                product: 0,
                limit: MAX_WORKGROUP_INVOCATIONS_GUARANTEED,
            });
        }

        Ok(Self {
            spirv,
            entry_point: entry_point.to_owned(),
            num_bindings,
            push_constant_size,
            workgroup_size,
        })
    }

    /// Validate that a dispatch configuration is compatible with this shader.
    ///
    /// Checks:
    /// - Binding count matches `num_bindings`.
    /// - Push constant size does not exceed declared size.
    /// - Grid dimensions are all non-zero.
    ///
    /// # Errors
    ///
    /// Returns a [`VulkanPipelineError`] variant describing the mismatch.
    pub fn validate_dispatch(&self, config: &DispatchConfig) -> Result<(), VulkanPipelineError> {
        // Check binding count.
        let provided = config.bindings.len() as u32;
        if provided != self.num_bindings {
            return Err(VulkanPipelineError::BindingCountMismatch {
                required: self.num_bindings,
                provided,
            });
        }

        // Check each binding index is within range.
        for binding in &config.bindings {
            if binding.binding >= self.num_bindings {
                return Err(VulkanPipelineError::BindingOutOfRange {
                    index: binding.binding,
                    max: self.num_bindings.saturating_sub(1),
                });
            }
        }

        // Check push constants.
        if let Some(ref pc) = config.push_constants {
            let actual = pc.data.len() as u32;
            if actual > self.push_constant_size {
                return Err(VulkanPipelineError::PushConstantOverflow {
                    actual,
                    declared: self.push_constant_size,
                });
            }
        }

        // Check grid dimensions.
        if config.grid[0] == 0 {
            return Err(VulkanPipelineError::ZeroGridDimension { dim: "x" });
        }
        if config.grid[1] == 0 {
            return Err(VulkanPipelineError::ZeroGridDimension { dim: "y" });
        }
        if config.grid[2] == 0 {
            return Err(VulkanPipelineError::ZeroGridDimension { dim: "z" });
        }

        Ok(())
    }

    /// The raw SPIR-V binary bytes.
    #[must_use]
    pub fn spirv(&self) -> &[u8] {
        &self.spirv
    }

    /// The shader entry point name.
    #[must_use]
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Number of descriptor set bindings.
    #[must_use]
    pub fn num_bindings(&self) -> u32 {
        self.num_bindings
    }

    /// Push constant block size in bytes.
    #[must_use]
    pub fn push_constant_size(&self) -> u32 {
        self.push_constant_size
    }

    /// Local workgroup size `[x, y, z]`.
    #[must_use]
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }
}

/// Descriptor for a buffer binding in the dispatch.
#[derive(Debug, Clone)]
pub struct BufferBinding {
    /// Binding index in the descriptor set.
    pub binding: u32,
    /// Offset into the buffer in bytes.
    pub offset: u64,
    /// Size of the buffer region in bytes.
    pub size: u64,
    /// Whether this binding is read-only.
    pub read_only: bool,
}

/// Push constants for kernel dispatch.
///
/// Accumulates push constant data as a byte buffer. Values are written in
/// little-endian order (matching Vulkan spec for push constants).
#[derive(Debug, Clone)]
pub struct PushConstants {
    data: Vec<u8>,
}

impl PushConstants {
    /// Create an empty push constants builder.
    #[must_use]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Append a `u32` value (4 bytes, little-endian).
    pub fn push_u32(&mut self, value: u32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// Append an `f32` value (4 bytes, little-endian).
    pub fn push_f32(&mut self, value: f32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// Append an `i32` value (4 bytes, little-endian).
    pub fn push_i32(&mut self, value: i32) {
        self.data.extend_from_slice(&value.to_le_bytes());
    }

    /// The accumulated push constant data as raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Total size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

impl Default for PushConstants {
    fn default() -> Self {
        Self::new()
    }
}

/// A dispatch configuration for a compute shader invocation.
#[derive(Debug, Clone)]
pub struct DispatchConfig {
    /// Grid dimensions [x, y, z] (number of workgroups per dimension).
    pub grid: [u32; 3],
    /// Buffer bindings.
    pub bindings: Vec<BufferBinding>,
    /// Optional push constants.
    pub push_constants: Option<PushConstants>,
}

/// Compute pipeline error.
#[derive(Debug, thiserror::Error)]
pub enum VulkanPipelineError {
    /// SPIR-V binary failed validation (bad magic, too short, etc.).
    #[error("SPIR-V validation failed: {reason}")]
    SpirvValidation { reason: String },

    /// A binding index exceeds the shader's declared binding count.
    #[error("binding index {index} exceeds maximum {max}")]
    BindingOutOfRange { index: u32, max: u32 },

    /// Push constant data exceeds the shader's declared push constant size.
    #[error("push constant size {actual} exceeds declared {declared}")]
    PushConstantOverflow { actual: u32, declared: u32 },

    /// Workgroup size product exceeds the device limit.
    #[error("workgroup size product ({product}) exceeds device limit ({limit})")]
    WorkgroupSizeExceeded { product: u32, limit: u32 },

    /// Requested buffer size exceeds the configured maximum.
    #[error("buffer size {requested} exceeds maximum {max}")]
    BufferTooLarge { requested: u64, max: u64 },

    /// A grid dimension is zero (dispatch would be a no-op).
    #[error("grid dimension {dim} is zero")]
    ZeroGridDimension { dim: &'static str },

    /// No suitable Vulkan device found.
    #[error("no suitable Vulkan device found")]
    NoDevice,

    /// Dispatch requires a different number of bindings than provided.
    #[error("dispatch requires {required} bindings but only {provided} provided")]
    BindingCountMismatch { required: u32, provided: u32 },
}

/// Compute grid dimensions for a 1D dispatch.
///
/// Given a total number of elements and a workgroup size `[x, y, z]`, returns
/// the dispatch grid `[ceil(total / x), 1, 1]`. The y and z workgroup
/// dimensions are assumed to be 1 for this helper.
///
/// # Example
///
/// ```
/// use nn_vulkan::compute_pipeline::compute_grid_dims;
/// let grid = compute_grid_dims(1000, [256, 1, 1]);
/// assert_eq!(grid, [4, 1, 1]); // ceil(1000/256) = 4
/// ```
///
/// # Panics
///
/// Panics if `workgroup_size[0]` is 0.
#[must_use]
pub fn compute_grid_dims(total_elements: u32, workgroup_size: [u32; 3]) -> [u32; 3] {
    assert!(workgroup_size[0] > 0, "workgroup_size[0] must be > 0");
    // Intentional manual ceil-div, NOT `div_ceil`: the `+ workgroup_size[0] - 1`
    // overflow on inputs near u32::MAX is a documented limitation guarded by the
    // `max_u32_elements_*_overflows` tests. `div_ceil` would silently remove that
    // overflow and change the contract.
    #[allow(clippy::manual_div_ceil)]
    let groups_x = (total_elements + workgroup_size[0] - 1) / workgroup_size[0];
    [groups_x, 1, 1]
}

/// Convert a SPIR-V word stream (`Vec<u32>`) to raw bytes (`Vec<u8>`).
///
/// Each word is written in native-endian byte order, which on little-endian
/// systems matches the SPIR-V binary format directly.
///
/// # Example
///
/// ```
/// use nn_vulkan::compute_pipeline::spirv_words_to_bytes;
/// let words: Vec<u32> = vec![0x07230203, 0x00010500, 0, 0, 0];
/// let bytes = spirv_words_to_bytes(&words);
/// assert_eq!(bytes.len(), 20);
/// assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07]); // magic (little-endian)
/// ```
#[must_use]
pub fn spirv_words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
#[path = "compute_pipeline_tests.rs"]
mod tests;
