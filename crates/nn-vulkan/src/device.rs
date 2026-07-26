// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Vulkan device discovery and queue family selection.
//!
//! Provides [`VulkanDevice`] for selecting a Vulkan-capable GPU, identifying
//! a compute queue family, and discovering memory types suitable for GPU
//! buffer allocation.
//!
//! # Runtime gating
//!
//! All hardware interaction is gated behind [`is_vulkan_available`]. When
//! no Vulkan driver is present (e.g., macOS without MoltenVK, CI without
//! GPU), construction returns [`VulkanError::NotAvailable`]. SPIR-V code
//! generation works without hardware — only dispatch requires a live device.

use crate::error::VulkanError;

/// Check whether a Vulkan-capable driver is available on this system.
///
/// Returns `true` on Linux/Windows with Vulkan ICD installed, or macOS
/// with MoltenVK. Returns `false` when no Vulkan loader can be found.
///
/// This is a lightweight check (no device enumeration). Use
/// [`VulkanDevice::new`] for full device discovery.
#[must_use]
pub fn is_vulkan_available() -> bool {
    // Placeholder: real implementation will probe for Vulkan loader via
    // dlopen("libvulkan.so") / LoadLibrary("vulkan-1.dll") / MoltenVK.
    false
}

/// Vulkan memory type flags (mirrors `VkMemoryPropertyFlagBits`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MemoryPropertyFlags {
    /// Visible to the device only (fastest for GPU compute).
    DeviceLocal,
    /// Visible to both host and device (staging buffers, readback).
    HostVisible,
    /// Host-visible and coherent (no explicit flush needed).
    HostCoherent,
}

impl MemoryPropertyFlags {
    /// Convert to Vulkan `VkMemoryPropertyFlagBits` numeric value.
    #[must_use]
    pub fn to_vk_bits(self) -> u32 {
        match self {
            Self::DeviceLocal => 0x0000_0001,
            Self::HostVisible => 0x0000_0002,
            Self::HostCoherent => 0x0000_0004,
        }
    }
}

/// Metadata about a Vulkan physical device queue family.
#[derive(Debug, Clone)]
pub struct QueueFamilyInfo {
    /// Queue family index within the physical device.
    pub index: u32,
    /// Number of queues available in this family.
    pub queue_count: u32,
    /// Whether this family supports compute operations.
    pub supports_compute: bool,
    /// Whether this family supports transfer operations.
    pub supports_transfer: bool,
}

/// Vulkan device context.
///
/// Wraps physical device selection, logical device creation, compute queue
/// family identification, and memory type discovery. This is the entry
/// point for all Vulkan backend operations.
///
/// # Example (conceptual)
///
/// ```no_run
/// use nn_vulkan::device::VulkanDevice;
/// let device = VulkanDevice::new(0).expect("Vulkan device");
/// println!("Using: {} (driver {})", device.device_name(), device.driver_version());
/// ```
#[derive(Debug)]
pub struct VulkanDevice {
    /// Device ordinal (physical device index).
    device_index: u32,
    /// Human-readable device name.
    device_name: String,
    /// Driver version string.
    driver_version: String,
    /// Vulkan API version supported by the device.
    api_version: (u32, u32, u32),
    /// Selected compute queue family index.
    compute_queue_family: u32,
    /// Maximum workgroup size (x dimension).
    max_workgroup_size_x: u32,
    /// Maximum total invocations per workgroup.
    max_workgroup_invocations: u32,
    /// Maximum shared memory per workgroup (bytes).
    max_shared_memory: u32,
}

impl VulkanDevice {
    /// Create a Vulkan device context for the given physical device index.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::NotAvailable`] if no Vulkan driver is found.
    /// Returns [`VulkanError::NoDevices`] if no GPU is detected.
    /// Returns [`VulkanError::NoComputeQueue`] if no compute-capable queue
    /// family exists on the device.
    pub fn new(_device_ordinal: u32) -> Result<Self, VulkanError> {
        if !is_vulkan_available() {
            return Err(VulkanError::NotAvailable);
        }

        // Placeholder — real implementation enumerates VkPhysicalDevice,
        // queries queue families, creates VkDevice.
        Err(VulkanError::NotAvailable)
    }

    /// The selected physical device index.
    #[must_use]
    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    /// The device name (e.g., "AMD Radeon RX 7900 XTX").
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// The driver version string.
    #[must_use]
    pub fn driver_version(&self) -> &str {
        &self.driver_version
    }

    /// The Vulkan API version as `(major, minor, patch)`.
    #[must_use]
    pub fn api_version(&self) -> (u32, u32, u32) {
        self.api_version
    }

    /// The selected compute queue family index.
    #[must_use]
    pub fn compute_queue_family(&self) -> u32 {
        self.compute_queue_family
    }

    /// Maximum workgroup size in the X dimension.
    #[must_use]
    pub fn max_workgroup_size_x(&self) -> u32 {
        self.max_workgroup_size_x
    }

    /// Maximum total invocations per workgroup.
    #[must_use]
    pub fn max_workgroup_invocations(&self) -> u32 {
        self.max_workgroup_invocations
    }

    /// Maximum shared memory per workgroup in bytes.
    #[must_use]
    pub fn max_shared_memory(&self) -> u32 {
        self.max_shared_memory
    }

    /// Find a memory type index that satisfies the given property flags.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError::NoSuitableMemoryType`] if no memory type
    /// on this device matches.
    pub fn find_memory_type(
        &self,
        _type_filter: u32,
        flags: MemoryPropertyFlags,
    ) -> Result<u32, VulkanError> {
        // Placeholder — real implementation queries VkPhysicalDeviceMemoryProperties.
        Err(VulkanError::NoSuitableMemoryType {
            flags: flags.to_vk_bits(),
        })
    }
}
