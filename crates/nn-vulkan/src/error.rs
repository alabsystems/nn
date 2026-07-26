// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for the Vulkan backend.

use thiserror::Error;

/// Errors from the Vulkan/SPIR-V backend.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VulkanError {
    #[error("Vulkan runtime not available — Vulkan-capable driver required")]
    NotAvailable,

    #[error("no Vulkan-capable GPU devices found")]
    NoDevices,

    #[error("no suitable queue family found with compute support")]
    NoComputeQueue,

    #[error("no suitable memory type found for flags {flags:#x}")]
    NoSuitableMemoryType { flags: u32 },

    #[error("GPU out of memory: requested {requested} bytes")]
    OutOfMemory { requested: usize },

    #[error("buffer size mismatch: expected {expected} bytes, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },

    #[error("SPIR-V code generation error: {reason}")]
    SpirVCodegen { reason: String },

    #[error("compute pipeline creation failed: {reason}")]
    PipelineCreation { reason: String },

    #[error("descriptor set layout error: {reason}")]
    DescriptorSetError { reason: String },

    #[error("command buffer recording error: {reason}")]
    CommandBufferError { reason: String },

    #[error("shader compilation failed: {reason}")]
    ShaderCompilationFailed { reason: String },

    #[error("unsupported dispatch step: {step_name}")]
    UnsupportedStep { step_name: &'static str },

    #[error("unsupported scalar type for Vulkan SPIR-V: {type_desc}")]
    UnsupportedType { type_desc: &'static str },

    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
