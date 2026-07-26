// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Device types for nn

/// Compute device for tensor operations
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// CPU with optional SIMD
    #[default]
    Cpu,

    /// Metal GPU (Apple Silicon)
    Metal { device_id: u32 },

    /// CUDA GPU (NVIDIA)
    Cuda { device_id: u32 },

    /// Vulkan GPU (cross-platform: AMD, Intel, mobile)
    Vulkan { device_id: u32 },

    /// Apple Neural Engine (via CoreML)
    Ane,
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Metal { device_id } => write!(f, "Metal({device_id})"),
            Self::Cuda { device_id } => write!(f, "CUDA({device_id})"),
            Self::Vulkan { device_id } => write!(f, "Vulkan({device_id})"),
            Self::Ane => write!(f, "ANE"),
        }
    }
}

impl Device {
    /// Default Metal device (device 0).
    #[must_use]
    pub fn metal() -> Self {
        Self::Metal { device_id: 0 }
    }

    /// Default CUDA device (device 0).
    #[must_use]
    pub fn cuda() -> Self {
        Self::Cuda { device_id: 0 }
    }

    /// Default Vulkan device (device 0).
    #[must_use]
    pub fn vulkan() -> Self {
        Self::Vulkan { device_id: 0 }
    }

    /// Returns `true` if this device is a GPU (Metal, CUDA, or Vulkan).
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        match self {
            Self::Metal { .. } | Self::Cuda { .. } | Self::Vulkan { .. } => true,
            Self::Cpu | Self::Ane => false,
        }
    }

    /// Returns `true` if this device is CPU.
    #[must_use]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Returns `true` if this device is Metal (Apple Silicon GPU).
    #[must_use]
    pub fn is_metal(&self) -> bool {
        matches!(self, Self::Metal { .. })
    }

    /// Returns `true` if this device is CUDA (NVIDIA GPU).
    #[must_use]
    pub fn is_cuda(&self) -> bool {
        matches!(self, Self::Cuda { .. })
    }

    /// Returns `true` if this device is Vulkan (cross-platform GPU).
    #[must_use]
    pub fn is_vulkan(&self) -> bool {
        matches!(self, Self::Vulkan { .. })
    }

    /// Returns `true` if this device is Apple Neural Engine.
    #[must_use]
    pub fn is_ane(&self) -> bool {
        matches!(self, Self::Ane)
    }

    /// Returns `true` if this device has dedicated compute hardware
    /// (GPU or ANE). Useful for routing decisions without hardcoding
    /// device variant checks.
    #[must_use]
    pub fn is_accelerator(&self) -> bool {
        !self.is_cpu()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_default_is_cpu() {
        assert_eq!(Device::default(), Device::Cpu);
    }

    #[test]
    fn test_device_metal_constructor() {
        let d = Device::metal();
        assert!(matches!(d, Device::Metal { device_id: 0 }));
    }

    #[test]
    fn test_device_cuda_constructor() {
        let d = Device::cuda();
        assert!(matches!(d, Device::Cuda { device_id: 0 }));
    }

    #[test]
    fn test_device_vulkan_constructor() {
        let d = Device::vulkan();
        assert!(matches!(d, Device::Vulkan { device_id: 0 }));
    }

    #[test]
    fn test_is_gpu() {
        assert!(!Device::Cpu.is_gpu());
        assert!(Device::metal().is_gpu());
        assert!(Device::Metal { device_id: 1 }.is_gpu());
        assert!(Device::cuda().is_gpu());
        assert!(Device::Cuda { device_id: 2 }.is_gpu());
        assert!(Device::vulkan().is_gpu());
        assert!(Device::Vulkan { device_id: 3 }.is_gpu());
        assert!(!Device::Ane.is_gpu());
    }

    #[test]
    fn test_is_cpu() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::metal().is_cpu());
        assert!(!Device::cuda().is_cpu());
        assert!(!Device::vulkan().is_cpu());
        assert!(!Device::Ane.is_cpu());
    }

    #[test]
    fn test_is_metal() {
        assert!(!Device::Cpu.is_metal());
        assert!(Device::metal().is_metal());
        assert!(Device::Metal { device_id: 5 }.is_metal());
        assert!(!Device::cuda().is_metal());
        assert!(!Device::vulkan().is_metal());
        assert!(!Device::Ane.is_metal());
    }

    #[test]
    fn test_is_cuda() {
        assert!(!Device::Cpu.is_cuda());
        assert!(!Device::metal().is_cuda());
        assert!(Device::cuda().is_cuda());
        assert!(Device::Cuda { device_id: 3 }.is_cuda());
        assert!(!Device::vulkan().is_cuda());
        assert!(!Device::Ane.is_cuda());
    }

    #[test]
    fn test_is_vulkan() {
        assert!(!Device::Cpu.is_vulkan());
        assert!(!Device::metal().is_vulkan());
        assert!(!Device::cuda().is_vulkan());
        assert!(Device::vulkan().is_vulkan());
        assert!(Device::Vulkan { device_id: 2 }.is_vulkan());
        assert!(!Device::Ane.is_vulkan());
    }

    #[test]
    fn test_is_ane() {
        assert!(!Device::Cpu.is_ane());
        assert!(!Device::metal().is_ane());
        assert!(!Device::cuda().is_ane());
        assert!(!Device::vulkan().is_ane());
        assert!(Device::Ane.is_ane());
    }

    #[test]
    fn test_is_accelerator() {
        assert!(!Device::Cpu.is_accelerator());
        assert!(Device::metal().is_accelerator());
        assert!(Device::cuda().is_accelerator());
        assert!(Device::vulkan().is_accelerator());
        assert!(Device::Ane.is_accelerator());
    }

    #[test]
    fn test_display_all_variants() {
        assert_eq!(format!("{}", Device::Cpu), "CPU");
        assert_eq!(format!("{}", Device::metal()), "Metal(0)");
        assert_eq!(format!("{}", Device::Metal { device_id: 3 }), "Metal(3)");
        assert_eq!(format!("{}", Device::cuda()), "CUDA(0)");
        assert_eq!(format!("{}", Device::Cuda { device_id: 7 }), "CUDA(7)");
        assert_eq!(format!("{}", Device::vulkan()), "Vulkan(0)");
        assert_eq!(format!("{}", Device::Vulkan { device_id: 2 }), "Vulkan(2)");
        assert_eq!(format!("{}", Device::Ane), "ANE");
    }
}
