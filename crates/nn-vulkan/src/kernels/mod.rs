// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pre-built GLSL compute shader strings for common ML operations.
//!
//! These are GLSL 450 compute shader source strings that can be compiled
//! to SPIR-V via `glslangValidator` or `shaderc`. Each shader follows
//! the standard nn-vulkan buffer binding convention:
//!
//! - `set = 0, binding = 0`: input buffer(s)
//! - `set = 0, binding = 1`: output buffer
//! - Push constants for runtime parameters (sizes, strides)

pub mod activations;
