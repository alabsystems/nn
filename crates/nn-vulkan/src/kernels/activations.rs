// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GLSL compute shader strings for elementwise activation functions.
//!
//! Each function returns a complete GLSL 450 compute shader source string
//! suitable for compilation to SPIR-V. Shaders use the standard nn-vulkan
//! binding convention (input at binding 0, output at binding 1, push
//! constants for `total_elements`).

use crate::error::VulkanError;
use crate::spirv_emit::{emit_elementwise_glsl, DEFAULT_WORKGROUP_SIZE};

/// ReLU activation: `max(x, 0.0)`.
pub fn relu_glsl() -> Result<String, VulkanError> {
    emit_elementwise_glsl("relu", "max(x, 0.0)", DEFAULT_WORKGROUP_SIZE)
}

/// SiLU (Swish) activation: `x * (1.0 / (1.0 + exp(-x)))`.
pub fn silu_glsl() -> Result<String, VulkanError> {
    emit_elementwise_glsl("silu", "x / (1.0 + exp(-x))", DEFAULT_WORKGROUP_SIZE)
}

/// Sigmoid activation: `1.0 / (1.0 + exp(-x))`.
pub fn sigmoid_glsl() -> Result<String, VulkanError> {
    emit_elementwise_glsl("sigmoid", "1.0 / (1.0 + exp(-x))", DEFAULT_WORKGROUP_SIZE)
}

/// Tanh activation: `tanh(x)`.
pub fn tanh_glsl() -> Result<String, VulkanError> {
    emit_elementwise_glsl("tanh_act", "tanh(x)", DEFAULT_WORKGROUP_SIZE)
}

/// GELU activation (approximate): `0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
pub fn gelu_glsl() -> Result<String, VulkanError> {
    emit_elementwise_glsl(
        "gelu",
        "0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))",
        DEFAULT_WORKGROUP_SIZE,
    )
}

/// Snake activation: `x + (1.0 / alpha) * sin(alpha * x)^2`.
///
/// Note: `alpha` is passed as a push constant. This simplified version
/// hardcodes `alpha = 1.0` for the GLSL string. Parameterized alpha
/// requires a separate push constant binding (deferred to runtime integration).
pub fn snake_glsl() -> Result<String, VulkanError> {
    emit_elementwise_glsl("snake", "x + sin(x) * sin(x)", DEFAULT_WORKGROUP_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_glsl_generates_valid_shader() {
        let src = relu_glsl().expect("should generate relu");
        assert!(src.contains("#version 450"));
        assert!(src.contains("layout(local_size_x ="));
        assert!(src.contains("max(x, 0.0)"));
        assert!(src.contains("input_buffer"));
        assert!(src.contains("output_buffer"));
    }

    #[test]
    fn test_silu_glsl_generates_valid_shader() {
        let src = silu_glsl().expect("should generate silu");
        assert!(src.contains("exp(-x)"));
    }

    #[test]
    fn test_sigmoid_glsl_generates_valid_shader() {
        let src = sigmoid_glsl().expect("should generate sigmoid");
        assert!(src.contains("1.0 / (1.0 + exp(-x))"));
    }

    #[test]
    fn test_gelu_glsl_generates_valid_shader() {
        let src = gelu_glsl().expect("should generate gelu");
        assert!(src.contains("0.044715"));
    }

    #[test]
    fn test_snake_glsl_generates_valid_shader() {
        let src = snake_glsl().expect("should generate snake");
        assert!(src.contains("sin(x)"));
    }

    #[test]
    fn test_all_activations_contain_version_and_layout() {
        let generators: Vec<(&str, fn() -> Result<String, VulkanError>)> = vec![
            ("relu", relu_glsl),
            ("silu", silu_glsl),
            ("sigmoid", sigmoid_glsl),
            ("tanh", tanh_glsl),
            ("gelu", gelu_glsl),
            ("snake", snake_glsl),
        ];
        for (name, emit_fn) in generators {
            let src = emit_fn().unwrap_or_else(|_| panic!("failed to generate {name}"));
            assert!(
                src.contains("#version 450"),
                "{name} missing GLSL version header"
            );
            assert!(
                src.contains("gl_GlobalInvocationID"),
                "{name} missing global invocation ID"
            );
        }
    }
}
