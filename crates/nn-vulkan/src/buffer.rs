// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vulkan GPU buffer management with staging buffer pattern.
//!
//! Vulkan memory is explicitly managed: device-local memory is fast but not
//! host-visible, so host-to-device transfers require a staging buffer in
//! host-visible memory followed by a copy command.
//!
//! [`VulkanBuffer`] represents a device-local buffer suitable for compute
//! shader access. [`StagingBuffer`] represents a host-visible buffer for
//! upload/download transfers.

use crate::error::VulkanError;

/// Access mode for Vulkan buffers (maps to `VkBufferUsageFlags`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BufferUsage {
    /// Read-only storage buffer (SSBO) for compute shaders.
    StorageRead,
    /// Read-write storage buffer (SSBO) for compute shaders.
    StorageReadWrite,
    /// Uniform buffer for small constant data.
    Uniform,
    /// Transfer source (staging upload).
    TransferSrc,
    /// Transfer destination (device-local, populated via copy).
    TransferDst,
}

impl BufferUsage {
    /// Convert to Vulkan `VkBufferUsageFlagBits` numeric value.
    #[must_use]
    pub fn to_vk_bits(self) -> u32 {
        match self {
            Self::StorageRead | Self::StorageReadWrite => 0x0000_0020, // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            Self::Uniform => 0x0000_0010, // VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            Self::TransferSrc => 0x0000_0001, // VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            Self::TransferDst => 0x0000_0002, // VK_BUFFER_USAGE_TRANSFER_DST_BIT
        }
    }
}

/// A device-local GPU buffer for compute shader access.
///
/// Vulkan buffers are allocated from device-local memory. Data must be
/// transferred via a staging buffer (see [`StagingBuffer`]).
///
/// Buffers are freed when dropped (RAII). When Vulkan runtime is connected,
/// the destructor calls `vkDestroyBuffer` and `vkFreeMemory`.
#[derive(Debug)]
pub struct VulkanBuffer {
    /// Size in bytes.
    size_bytes: usize,
    /// Buffer usage flags.
    usage: BufferUsage,
    /// Opaque handle (placeholder for VkBuffer — will be used when FFI is wired).
    _handle: u64,
    /// Opaque memory handle (placeholder for VkDeviceMemory — will be used when FFI is wired).
    _memory_handle: u64,
}

impl VulkanBuffer {
    /// Create a new device-local buffer.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::OutOfMemory`] if allocation fails.
    /// Returns [`VulkanError::InvalidParameter`] if `size_bytes` is 0.
    pub fn new(size_bytes: usize, usage: BufferUsage) -> Result<Self, VulkanError> {
        if size_bytes == 0 {
            return Err(VulkanError::InvalidParameter(
                "buffer size must be > 0".into(),
            ));
        }

        // Placeholder — real implementation calls vkCreateBuffer + vkAllocateMemory.
        Ok(Self {
            size_bytes,
            usage,
            _handle: 0,
            _memory_handle: 0,
        })
    }

    /// Buffer size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Buffer usage mode.
    #[must_use]
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Raw buffer handle (opaque; for FFI integration).
    #[must_use]
    pub fn handle(&self) -> u64 {
        self._handle
    }
}

/// A host-visible staging buffer for CPU-GPU data transfer.
///
/// Used in the staging buffer pattern: data is written to a `StagingBuffer`
/// (host-visible memory), then copied to a [`VulkanBuffer`] (device-local
/// memory) via a transfer command.
#[derive(Debug)]
pub struct StagingBuffer {
    /// Size in bytes.
    size_bytes: usize,
    /// Whether this is an upload (TransferSrc) or download (TransferDst) staging buffer.
    is_upload: bool,
    /// Opaque handle (placeholder for VkBuffer — will be used when FFI is wired).
    _handle: u64,
    /// Opaque memory handle (placeholder for VkDeviceMemory — will be used when FFI is wired).
    _memory_handle: u64,
}

impl StagingBuffer {
    /// Create a host-visible staging buffer for upload to device.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::OutOfMemory`] if allocation fails.
    /// Returns [`VulkanError::InvalidParameter`] if `size_bytes` is 0.
    pub fn new_upload(size_bytes: usize) -> Result<Self, VulkanError> {
        if size_bytes == 0 {
            return Err(VulkanError::InvalidParameter(
                "staging buffer size must be > 0".into(),
            ));
        }

        // Placeholder — real implementation creates host-visible + host-coherent buffer.
        Ok(Self {
            size_bytes,
            is_upload: true,
            _handle: 0,
            _memory_handle: 0,
        })
    }

    /// Create a host-visible staging buffer for download from device.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::OutOfMemory`] if allocation fails.
    /// Returns [`VulkanError::InvalidParameter`] if `size_bytes` is 0.
    pub fn new_download(size_bytes: usize) -> Result<Self, VulkanError> {
        if size_bytes == 0 {
            return Err(VulkanError::InvalidParameter(
                "staging buffer size must be > 0".into(),
            ));
        }

        Ok(Self {
            size_bytes,
            is_upload: false,
            _handle: 0,
            _memory_handle: 0,
        })
    }

    /// Staging buffer size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Whether this is an upload staging buffer.
    #[must_use]
    pub fn is_upload(&self) -> bool {
        self.is_upload
    }

    /// Write data from a host slice into the staging buffer.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::BufferSizeMismatch`] if `data.len() * 4` exceeds
    /// the staging buffer size.
    pub fn write_f32(&mut self, data: &[f32]) -> Result<(), VulkanError> {
        let byte_len = data
            .len()
            .checked_mul(4)
            .ok_or_else(|| VulkanError::InvalidParameter("data length overflow".into()))?;
        if byte_len > self.size_bytes {
            return Err(VulkanError::BufferSizeMismatch {
                expected: self.size_bytes,
                actual: byte_len,
            });
        }

        // Placeholder — real implementation maps memory and copies.
        Ok(())
    }

    /// Read data from the staging buffer into a host vec.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::BufferSizeMismatch`] if `count * 4` exceeds
    /// the staging buffer size.
    pub fn read_f32(&self, count: usize) -> Result<Vec<f32>, VulkanError> {
        let byte_len = count
            .checked_mul(4)
            .ok_or_else(|| VulkanError::InvalidParameter("read count overflow".into()))?;
        if byte_len > self.size_bytes {
            return Err(VulkanError::BufferSizeMismatch {
                expected: self.size_bytes,
                actual: byte_len,
            });
        }

        // Placeholder — real implementation maps memory and copies.
        Ok(vec![0.0_f32; count])
    }
}
