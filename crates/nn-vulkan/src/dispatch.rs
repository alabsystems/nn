// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Compute pipeline creation, descriptor sets, and command buffer dispatch.
//!
//! Orchestrates the Vulkan compute dispatch pipeline:
//!
//! 1. **Shader module**: SPIR-V binary loaded into a `VkShaderModule`.
//! 2. **Descriptor set layout**: Declares buffer bindings for the shader.
//! 3. **Pipeline layout**: Combines descriptor set layout + push constants.
//! 4. **Compute pipeline**: Compiled pipeline ready for dispatch.
//! 5. **Command buffer**: Records bind + dispatch + barrier commands.
//!
//! This module provides safe Rust types wrapping each Vulkan object.
//! When Vulkan hardware is unavailable, pipeline creation returns
//! [`VulkanError::NotAvailable`].

use crate::buffer::VulkanBuffer;
use crate::error::VulkanError;

/// Descriptor binding type for compute shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DescriptorType {
    /// Storage buffer (SSBO) — read/write from compute shaders.
    StorageBuffer,
    /// Uniform buffer — read-only small constant data.
    UniformBuffer,
}

impl DescriptorType {
    /// Convert to Vulkan `VkDescriptorType` numeric value.
    #[must_use]
    pub fn to_vk_type(self) -> u32 {
        match self {
            Self::StorageBuffer => 7, // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            Self::UniformBuffer => 6, // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        }
    }
}

/// A single descriptor binding specification.
#[derive(Debug, Clone)]
pub struct DescriptorBinding {
    /// Binding index in the shader.
    pub binding: u32,
    /// Descriptor type.
    pub descriptor_type: DescriptorType,
    /// Number of descriptors (usually 1 for buffers).
    pub count: u32,
}

/// Descriptor set layout: declares the buffer bindings for a compute shader.
#[derive(Debug)]
pub struct DescriptorSetLayout {
    bindings: Vec<DescriptorBinding>,
    /// Opaque handle (placeholder for VkDescriptorSetLayout — will be used when FFI is wired).
    _handle: u64,
}

impl DescriptorSetLayout {
    /// Create a descriptor set layout from binding specifications.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::DescriptorSetError`] if bindings are invalid.
    pub fn new(bindings: Vec<DescriptorBinding>) -> Result<Self, VulkanError> {
        if bindings.is_empty() {
            return Err(VulkanError::DescriptorSetError {
                reason: "descriptor set layout must have at least one binding".into(),
            });
        }

        // Validate no duplicate binding indices.
        let mut seen = std::collections::HashSet::new();
        for b in &bindings {
            if !seen.insert(b.binding) {
                return Err(VulkanError::DescriptorSetError {
                    reason: format!("duplicate binding index: {}", b.binding),
                });
            }
        }

        Ok(Self {
            bindings,
            _handle: 0,
        })
    }

    /// Number of bindings in this layout.
    #[must_use]
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Iterate over bindings.
    pub fn bindings(&self) -> &[DescriptorBinding] {
        &self.bindings
    }
}

/// Push constant range specification.
#[derive(Debug, Clone)]
pub struct PushConstantRange {
    /// Offset in bytes from the start of the push constant block.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
}

/// Pipeline layout combining descriptor set layouts and push constants.
#[derive(Debug)]
pub struct PipelineLayout {
    /// Opaque handle (placeholder for VkPipelineLayout — will be used when FFI is wired).
    _handle: u64,
    push_constant_size: u32,
}

impl PipelineLayout {
    /// Create a pipeline layout from a descriptor set layout and push constant range.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::PipelineCreation`] if parameters are invalid.
    pub fn new(
        _descriptor_layout: &DescriptorSetLayout,
        push_constant_size: u32,
    ) -> Result<Self, VulkanError> {
        // Vulkan spec: push constant size must be a multiple of 4.
        if !push_constant_size.is_multiple_of(4) {
            return Err(VulkanError::PipelineCreation {
                reason: format!(
                    "push constant size must be a multiple of 4, got {push_constant_size}"
                ),
            });
        }
        // Vulkan spec: max push constant size is 128 bytes (guaranteed minimum).
        if push_constant_size > 128 {
            return Err(VulkanError::PipelineCreation {
                reason: format!(
                    "push constant size {push_constant_size} exceeds 128-byte guaranteed minimum"
                ),
            });
        }

        Ok(Self {
            _handle: 0,
            push_constant_size,
        })
    }

    /// Push constant block size in bytes.
    #[must_use]
    pub fn push_constant_size(&self) -> u32 {
        self.push_constant_size
    }
}

/// A compiled Vulkan compute pipeline.
///
/// Created from a SPIR-V shader module + pipeline layout. Ready for
/// dispatch via [`VulkanDispatcher`].
#[derive(Debug, Clone)]
pub struct ComputePipeline {
    /// Opaque handle (placeholder for VkPipeline — will be used when FFI is wired).
    _handle: u64,
    /// Entry point name.
    entry_point: String,
    /// Workgroup size from shader reflection.
    workgroup_size: [u32; 3],
}

impl ComputePipeline {
    /// Create a compute pipeline from SPIR-V binary.
    ///
    /// # Arguments
    ///
    /// * `spirv_words` — SPIR-V binary as a word stream.
    /// * `entry_point` — Shader entry point name (usually `"main"`).
    /// * `layout` — Pipeline layout with descriptor set and push constant info.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::ShaderCompilationFailed`] if SPIR-V is invalid.
    /// Returns [`VulkanError::PipelineCreation`] if pipeline creation fails.
    pub fn new(
        spirv_words: &[u32],
        entry_point: &str,
        _layout: &PipelineLayout,
    ) -> Result<Self, VulkanError> {
        if spirv_words.is_empty() {
            return Err(VulkanError::ShaderCompilationFailed {
                reason: "empty SPIR-V binary".into(),
            });
        }
        if spirv_words[0] != crate::spirv_emit::SPIRV_MAGIC {
            return Err(VulkanError::ShaderCompilationFailed {
                reason: format!(
                    "invalid SPIR-V magic: expected {:#010x}, got {:#010x}",
                    crate::spirv_emit::SPIRV_MAGIC,
                    spirv_words[0]
                ),
            });
        }

        Ok(Self {
            _handle: 0,
            entry_point: entry_point.to_owned(),
            workgroup_size: [crate::spirv_emit::DEFAULT_WORKGROUP_SIZE, 1, 1],
        })
    }

    /// The shader entry point name.
    #[must_use]
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// The workgroup size `[x, y, z]`.
    #[must_use]
    pub fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup_size
    }
}

/// Vulkan compute dispatcher.
///
/// Records and submits compute dispatch commands. Manages command buffer
/// allocation, buffer binding, push constants, and execution barriers.
///
/// # Lifecycle
///
/// 1. Create via [`VulkanDispatcher::new`].
/// 2. Record dispatches via [`dispatch`](Self::dispatch).
/// 3. Submit and wait via [`submit_and_wait`](Self::submit_and_wait).
#[derive(Debug)]
pub struct VulkanDispatcher {
    /// Number of dispatches recorded.
    dispatch_count: u32,
    /// Opaque handle (placeholder for VkCommandBuffer — will be used when FFI is wired).
    _command_buffer_handle: u64,
}

impl VulkanDispatcher {
    /// Create a new dispatcher.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::NotAvailable`] if Vulkan is not initialized.
    pub fn new() -> Result<Self, VulkanError> {
        // Placeholder — real implementation allocates command pool + command buffer.
        Ok(Self {
            dispatch_count: 0,
            _command_buffer_handle: 0,
        })
    }

    /// Record a compute dispatch.
    ///
    /// Binds the pipeline, descriptor set (buffers), push constants, and
    /// records a `vkCmdDispatch` with the given workgroup counts.
    ///
    /// # Arguments
    ///
    /// * `pipeline` — Compiled compute pipeline.
    /// * `buffers` — Buffers to bind as descriptor set entries (in binding order).
    /// * `push_constants` — Push constant data (raw bytes, must match pipeline layout).
    /// * `group_count` — Number of workgroups to dispatch `[x, y, z]`.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::CommandBufferError`] if recording fails.
    pub fn dispatch(
        &mut self,
        _pipeline: &ComputePipeline,
        _buffers: &[&VulkanBuffer],
        _push_constants: &[u8],
        group_count: [u32; 3],
    ) -> Result<(), VulkanError> {
        if group_count.contains(&0) {
            return Err(VulkanError::CommandBufferError {
                reason: "workgroup count must be > 0 in all dimensions".into(),
            });
        }

        self.dispatch_count += 1;
        // Placeholder — real implementation records vkCmdBindPipeline,
        // vkCmdBindDescriptorSets, vkCmdPushConstants, vkCmdDispatch,
        // vkCmdPipelineBarrier.
        Ok(())
    }

    /// Submit recorded commands and wait for GPU completion.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::CommandBufferError`] if submission fails.
    pub fn submit_and_wait(&self) -> Result<(), VulkanError> {
        if self.dispatch_count == 0 {
            return Err(VulkanError::CommandBufferError {
                reason: "no dispatches recorded".into(),
            });
        }

        // Placeholder — real implementation calls vkEndCommandBuffer,
        // vkQueueSubmit, vkQueueWaitIdle.
        Ok(())
    }

    /// Number of dispatches recorded so far.
    #[must_use]
    pub fn dispatch_count(&self) -> u32 {
        self.dispatch_count
    }
}
