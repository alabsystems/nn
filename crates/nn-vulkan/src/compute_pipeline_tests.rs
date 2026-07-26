// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive tests for [`super::compute_pipeline`].

use super::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a minimal valid SPIR-V binary (20 bytes = 5 header words).
/// Magic: 0x07230203, Version: 0x00010500, Generator: 0, Bound: 0, Schema: 0.
fn minimal_spirv() -> Vec<u8> {
    spirv_words_to_bytes(&[
        crate::spirv_emit::SPIRV_MAGIC,
        crate::spirv_emit::SPIRV_VERSION_1_5,
        0, // generator
        0, // bound
        0, // schema
    ])
}

/// Build a SPIR-V binary with `extra_words` additional words beyond the 5-word header.
fn spirv_with_extra(extra_words: usize) -> Vec<u8> {
    let mut words = vec![
        crate::spirv_emit::SPIRV_MAGIC,
        crate::spirv_emit::SPIRV_VERSION_1_5,
        0,
        0,
        0,
    ];
    words.extend(std::iter::repeat_n(0u32, extra_words));
    spirv_words_to_bytes(&words)
}

/// Create a valid `CompiledShader` with reasonable defaults for dispatch tests.
fn default_shader() -> CompiledShader {
    CompiledShader::new(minimal_spirv(), "main", 2, 16, [256, 1, 1])
        .expect("default shader must be valid")
}

/// Create a `DispatchConfig` that matches `default_shader()`.
fn matching_dispatch_config() -> DispatchConfig {
    let mut pc = PushConstants::new();
    pc.push_u32(1024);
    pc.push_f32(1.0);
    pc.push_u32(0);
    pc.push_u32(0);
    DispatchConfig {
        grid: [4, 1, 1],
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: 4096,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: 4096,
                read_only: false,
            },
        ],
        push_constants: Some(pc),
    }
}

// ===========================================================================
// 1. CompiledShader
// ===========================================================================

mod compiled_shader {
    use super::*;

    // -- Construction with valid SPIR-V --

    #[test]
    fn valid_spirv_minimal() {
        let shader = CompiledShader::new(minimal_spirv(), "main", 2, 8, [256, 1, 1]);
        assert!(shader.is_ok());
        let s = shader.unwrap();
        assert_eq!(s.entry_point(), "main");
        assert_eq!(s.num_bindings(), 2);
        assert_eq!(s.push_constant_size(), 8);
        assert_eq!(s.workgroup_size(), [256, 1, 1]);
        assert_eq!(s.spirv().len(), 20);
    }

    #[test]
    fn valid_spirv_larger_binary() {
        let bytes = spirv_with_extra(100);
        assert_eq!(bytes.len(), (5 + 100) * 4);
        let shader = CompiledShader::new(bytes, "cs_main", 4, 64, [64, 4, 1]).unwrap();
        assert_eq!(shader.spirv().len(), 420);
        assert_eq!(shader.entry_point(), "cs_main");
        assert_eq!(shader.num_bindings(), 4);
        assert_eq!(shader.push_constant_size(), 64);
        assert_eq!(shader.workgroup_size(), [64, 4, 1]);
    }

    #[test]
    fn valid_spirv_zero_bindings_and_push_constants() {
        let s = CompiledShader::new(minimal_spirv(), "main", 0, 0, [256, 1, 1]).unwrap();
        assert_eq!(s.num_bindings(), 0);
        assert_eq!(s.push_constant_size(), 0);
    }

    // -- Construction with invalid/empty bytes --

    #[test]
    fn reject_empty_bytes() {
        let result = CompiledShader::new(vec![], "main", 0, 0, [256, 1, 1]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("too short"), "unexpected error: {msg}");
    }

    #[test]
    fn reject_too_short_19_bytes() {
        let bytes = vec![
            0x03, 0x02, 0x23, 0x07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]; // 19 bytes
        let result = CompiledShader::new(bytes, "main", 0, 0, [256, 1, 1]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("19 bytes"), "unexpected error: {msg}");
    }

    #[test]
    fn reject_4_bytes_too_short() {
        let spirv = vec![0x03, 0x02, 0x23, 0x07]; // only 4 bytes, need 20
        let result = CompiledShader::new(spirv, "main", 0, 0, [1, 1, 1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VulkanPipelineError::SpirvValidation { .. }
        ));
    }

    #[test]
    fn reject_invalid_magic_first_byte() {
        let mut spirv = minimal_spirv();
        spirv[0] = 0xFF;
        let result = CompiledShader::new(spirv, "main", 1, 0, [64, 1, 1]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, VulkanPipelineError::SpirvValidation { .. }));
        assert!(err.to_string().contains("invalid SPIR-V magic"));
    }

    #[test]
    fn reject_all_zeros_20_bytes() {
        let bytes = vec![0u8; 20]; // all zeros => bad magic
        let result = CompiledShader::new(bytes, "main", 0, 0, [64, 1, 1]);
        assert!(result.is_err());
    }

    // -- Entry point detection --

    #[test]
    fn entry_point_custom_name() {
        let s = CompiledShader::new(minimal_spirv(), "nn_custom_entry", 0, 0, [32, 1, 1]).unwrap();
        assert_eq!(s.entry_point(), "nn_custom_entry");
    }

    #[test]
    fn entry_point_empty_string() {
        let s = CompiledShader::new(minimal_spirv(), "", 0, 0, [32, 1, 1]).unwrap();
        assert_eq!(s.entry_point(), "");
    }

    // -- Workgroup size extraction --

    #[test]
    fn workgroup_size_3d() {
        let s = CompiledShader::new(minimal_spirv(), "main", 0, 0, [8, 8, 4]).unwrap();
        assert_eq!(s.workgroup_size(), [8, 8, 4]);
    }

    #[test]
    fn workgroup_size_1_1_1() {
        let s = CompiledShader::new(minimal_spirv(), "main", 0, 0, [1, 1, 1]).unwrap();
        assert_eq!(s.workgroup_size(), [1, 1, 1]);
    }

    #[test]
    fn reject_zero_workgroup_x() {
        let result = CompiledShader::new(minimal_spirv(), "main", 0, 0, [0, 1, 1]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VulkanPipelineError::WorkgroupSizeExceeded { .. }
        ));
    }

    #[test]
    fn reject_zero_workgroup_y() {
        let result = CompiledShader::new(minimal_spirv(), "main", 0, 0, [256, 0, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn reject_zero_workgroup_z() {
        let result = CompiledShader::new(minimal_spirv(), "main", 0, 0, [256, 1, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn reject_all_zero_workgroup() {
        let result = CompiledShader::new(minimal_spirv(), "main", 0, 0, [0, 0, 0]);
        assert!(result.is_err());
    }

    // -- spirv() accessor --

    #[test]
    fn spirv_accessor_returns_original_bytes() {
        let original = minimal_spirv();
        let s = CompiledShader::new(original.clone(), "main", 0, 0, [64, 1, 1]).unwrap();
        assert_eq!(s.spirv(), original.as_slice());
    }

    // -- validate_dispatch --

    #[test]
    fn validate_dispatch_accepts_matching_config() {
        let shader = default_shader();
        assert!(shader
            .validate_dispatch(&matching_dispatch_config())
            .is_ok());
    }

    #[test]
    fn validate_dispatch_rejects_wrong_binding_count() {
        let shader = default_shader(); // expects 2
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![BufferBinding {
                binding: 0,
                offset: 0,
                size: 256,
                read_only: true,
            }],
            push_constants: None,
        };
        let err = shader.validate_dispatch(&config).unwrap_err();
        assert!(matches!(
            err,
            VulkanPipelineError::BindingCountMismatch {
                required: 2,
                provided: 1
            }
        ));
    }

    #[test]
    fn validate_dispatch_rejects_binding_out_of_range() {
        let shader = default_shader(); // 2 bindings => max index 1
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![
                BufferBinding {
                    binding: 0,
                    offset: 0,
                    size: 256,
                    read_only: true,
                },
                BufferBinding {
                    binding: 5,
                    offset: 0,
                    size: 256,
                    read_only: false,
                },
            ],
            push_constants: None,
        };
        let err = shader.validate_dispatch(&config).unwrap_err();
        assert!(matches!(
            err,
            VulkanPipelineError::BindingOutOfRange { index: 5, max: 1 }
        ));
    }

    #[test]
    fn validate_dispatch_rejects_push_constant_overflow() {
        let shader = default_shader(); // push_constant_size = 16
        let mut pc = PushConstants::new();
        for _ in 0..5 {
            pc.push_u32(0); // 20 bytes > 16
        }
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![
                BufferBinding {
                    binding: 0,
                    offset: 0,
                    size: 256,
                    read_only: true,
                },
                BufferBinding {
                    binding: 1,
                    offset: 0,
                    size: 256,
                    read_only: false,
                },
            ],
            push_constants: Some(pc),
        };
        let err = shader.validate_dispatch(&config).unwrap_err();
        assert!(matches!(
            err,
            VulkanPipelineError::PushConstantOverflow {
                actual: 20,
                declared: 16
            }
        ));
    }

    #[test]
    fn validate_dispatch_accepts_smaller_push_constants() {
        let shader = default_shader(); // 16 declared
        let mut pc = PushConstants::new();
        pc.push_u32(42); // 4 bytes < 16
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![
                BufferBinding {
                    binding: 0,
                    offset: 0,
                    size: 256,
                    read_only: true,
                },
                BufferBinding {
                    binding: 1,
                    offset: 0,
                    size: 256,
                    read_only: false,
                },
            ],
            push_constants: Some(pc),
        };
        assert!(shader.validate_dispatch(&config).is_ok());
    }

    #[test]
    fn validate_dispatch_accepts_exact_push_constant_size() {
        let shader = default_shader(); // 16 declared
        let mut pc = PushConstants::new();
        for _ in 0..4 {
            pc.push_u32(0); // exactly 16 bytes
        }
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![
                BufferBinding {
                    binding: 0,
                    offset: 0,
                    size: 256,
                    read_only: true,
                },
                BufferBinding {
                    binding: 1,
                    offset: 0,
                    size: 256,
                    read_only: false,
                },
            ],
            push_constants: Some(pc),
        };
        assert!(shader.validate_dispatch(&config).is_ok());
    }

    #[test]
    fn validate_dispatch_rejects_zero_grid_x() {
        let shader = CompiledShader::new(minimal_spirv(), "main", 0, 0, [64, 1, 1]).unwrap();
        let config = DispatchConfig {
            grid: [0, 1, 1],
            bindings: vec![],
            push_constants: None,
        };
        let err = shader.validate_dispatch(&config).unwrap_err();
        assert!(matches!(
            err,
            VulkanPipelineError::ZeroGridDimension { dim: "x" }
        ));
    }

    #[test]
    fn validate_dispatch_rejects_zero_grid_y() {
        let shader = CompiledShader::new(minimal_spirv(), "main", 0, 0, [64, 1, 1]).unwrap();
        let config = DispatchConfig {
            grid: [1, 0, 1],
            bindings: vec![],
            push_constants: None,
        };
        let err = shader.validate_dispatch(&config).unwrap_err();
        assert!(matches!(
            err,
            VulkanPipelineError::ZeroGridDimension { dim: "y" }
        ));
    }

    #[test]
    fn validate_dispatch_rejects_zero_grid_z() {
        let shader = CompiledShader::new(minimal_spirv(), "main", 0, 0, [64, 1, 1]).unwrap();
        let config = DispatchConfig {
            grid: [1, 1, 0],
            bindings: vec![],
            push_constants: None,
        };
        let err = shader.validate_dispatch(&config).unwrap_err();
        assert!(matches!(
            err,
            VulkanPipelineError::ZeroGridDimension { dim: "z" }
        ));
    }

    #[test]
    fn validate_dispatch_no_push_constants_when_none_declared() {
        let shader = CompiledShader::new(minimal_spirv(), "main", 1, 0, [256, 1, 1]).unwrap();
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![BufferBinding {
                binding: 0,
                offset: 0,
                size: 1024,
                read_only: false,
            }],
            push_constants: None,
        };
        assert!(shader.validate_dispatch(&config).is_ok());
    }

    #[test]
    fn validate_dispatch_error_priority_binding_count_before_grid() {
        // If binding count is wrong AND grid has zeros, binding count error comes first.
        let shader = default_shader(); // 2 bindings
        let config = DispatchConfig {
            grid: [0, 0, 0],
            bindings: vec![], // 0 bindings, shader expects 2
            push_constants: None,
        };
        let err = shader.validate_dispatch(&config).unwrap_err();
        assert!(matches!(
            err,
            VulkanPipelineError::BindingCountMismatch { .. }
        ));
    }
}

// ===========================================================================
// 2. DispatchConfig
// ===========================================================================

mod dispatch_config {
    use super::*;

    #[test]
    fn grid_dimensions_stored() {
        let config = DispatchConfig {
            grid: [8, 4, 2],
            bindings: vec![],
            push_constants: None,
        };
        assert_eq!(config.grid, [8, 4, 2]);
    }

    #[test]
    fn push_constant_sizing() {
        let mut pc = PushConstants::new();
        pc.push_u32(100);
        pc.push_f32(3.14);
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![],
            push_constants: Some(pc),
        };
        assert_eq!(config.push_constants.as_ref().unwrap().size(), 8);
    }

    #[test]
    fn no_push_constants() {
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![],
            push_constants: None,
        };
        assert!(config.push_constants.is_none());
    }

    #[test]
    fn multiple_bindings() {
        let config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![
                BufferBinding {
                    binding: 0,
                    offset: 0,
                    size: 512,
                    read_only: true,
                },
                BufferBinding {
                    binding: 1,
                    offset: 0,
                    size: 512,
                    read_only: false,
                },
                BufferBinding {
                    binding: 2,
                    offset: 256,
                    size: 256,
                    read_only: true,
                },
            ],
            push_constants: None,
        };
        assert_eq!(config.bindings.len(), 3);
    }

    #[test]
    fn zero_grid_dimensions_representable() {
        // DispatchConfig itself does not validate; validation happens on shader.
        let config = DispatchConfig {
            grid: [0, 0, 0],
            bindings: vec![],
            push_constants: None,
        };
        assert_eq!(config.grid, [0, 0, 0]);
    }

    #[test]
    fn max_grid_dimensions() {
        let config = DispatchConfig {
            grid: [u32::MAX, u32::MAX, u32::MAX],
            bindings: vec![],
            push_constants: None,
        };
        assert_eq!(config.grid[0], u32::MAX);
    }

    #[test]
    fn clone_preserves_all_fields() {
        let mut pc = PushConstants::new();
        pc.push_u32(42);
        let config = DispatchConfig {
            grid: [10, 20, 30],
            bindings: vec![BufferBinding {
                binding: 0,
                offset: 0,
                size: 1024,
                read_only: false,
            }],
            push_constants: Some(pc),
        };
        let cloned = config;
        assert_eq!(cloned.grid, [10, 20, 30]);
        assert_eq!(cloned.bindings.len(), 1);
        assert_eq!(cloned.push_constants.as_ref().unwrap().size(), 4);
    }
}

// ===========================================================================
// 3. PushConstants
// ===========================================================================

mod push_constants {
    use super::*;

    #[test]
    fn empty() {
        let pc = PushConstants::new();
        assert_eq!(pc.size(), 0);
        assert!(pc.as_bytes().is_empty());
    }

    #[test]
    fn default_is_empty() {
        let pc = PushConstants::default();
        assert_eq!(pc.size(), 0);
    }

    // -- 4 bytes --

    #[test]
    fn push_u32_serialization_known_value() {
        let mut pc = PushConstants::new();
        pc.push_u32(0x04030201);
        assert_eq!(pc.size(), 4);
        assert_eq!(pc.as_bytes(), &[0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn push_u32_roundtrip() {
        let mut pc = PushConstants::new();
        pc.push_u32(0xDEAD_BEEF);
        let b = pc.as_bytes();
        let recovered = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        assert_eq!(recovered, 0xDEAD_BEEF);
    }

    #[test]
    fn push_u32_zero() {
        let mut pc = PushConstants::new();
        pc.push_u32(0);
        assert_eq!(pc.as_bytes(), &[0, 0, 0, 0]);
    }

    #[test]
    fn push_u32_max() {
        let mut pc = PushConstants::new();
        pc.push_u32(u32::MAX);
        assert_eq!(pc.as_bytes(), &[0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn push_f32_serialization() {
        let mut pc = PushConstants::new();
        pc.push_f32(1.0f32);
        assert_eq!(pc.size(), 4);
        assert_eq!(pc.as_bytes(), &1.0f32.to_le_bytes());
    }

    #[test]
    fn push_f32_roundtrip_pi() {
        let mut pc = PushConstants::new();
        let value = std::f32::consts::PI;
        pc.push_f32(value);
        let b = pc.as_bytes();
        let recovered = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        assert_eq!(recovered, value);
    }

    #[test]
    fn push_f32_negative() {
        let mut pc = PushConstants::new();
        pc.push_f32(-3.14);
        let b = pc.as_bytes();
        let recovered = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        assert!((recovered - (-3.14)).abs() < 1e-6);
    }

    #[test]
    fn push_i32_serialization() {
        let mut pc = PushConstants::new();
        pc.push_i32(-7);
        assert_eq!(pc.size(), 4);
        let b = pc.as_bytes();
        assert_eq!(i32::from_le_bytes([b[0], b[1], b[2], b[3]]), -7);
    }

    #[test]
    fn push_i32_roundtrip_negative() {
        let mut pc = PushConstants::new();
        pc.push_i32(-42);
        let b = pc.as_bytes();
        assert_eq!(i32::from_le_bytes([b[0], b[1], b[2], b[3]]), -42);
    }

    #[test]
    fn push_i32_min() {
        let mut pc = PushConstants::new();
        pc.push_i32(i32::MIN);
        let b = pc.as_bytes();
        assert_eq!(i32::from_le_bytes([b[0], b[1], b[2], b[3]]), i32::MIN);
    }

    // -- 16 bytes (mixed types) --

    #[test]
    fn mixed_types_16_bytes() {
        let mut pc = PushConstants::new();
        pc.push_u32(1024);
        pc.push_f32(0.5);
        pc.push_i32(-7);
        pc.push_u32(0);
        assert_eq!(pc.size(), 16);

        let b = pc.as_bytes();
        assert_eq!(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), 1024);
        assert_eq!(f32::from_le_bytes([b[4], b[5], b[6], b[7]]), 0.5);
        assert_eq!(i32::from_le_bytes([b[8], b[9], b[10], b[11]]), -7);
        assert_eq!(u32::from_le_bytes([b[12], b[13], b[14], b[15]]), 0);
    }

    // -- 128 bytes (Vulkan guaranteed max) --

    #[test]
    fn push_128_bytes_maximum_vulkan_guaranteed() {
        let mut pc = PushConstants::new();
        for i in 0u32..32 {
            pc.push_u32(i);
        }
        assert_eq!(pc.size(), 128);

        // Verify first and last values.
        let b = pc.as_bytes();
        assert_eq!(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), 0);
        assert_eq!(u32::from_le_bytes([b[124], b[125], b[126], b[127]]), 31);
    }

    #[test]
    fn push_beyond_128_bytes_still_accumulates() {
        // PushConstants itself has no size limit; the limit is validated
        // when binding to a shader via validate_dispatch.
        let mut pc = PushConstants::new();
        for i in 0u32..64 {
            pc.push_u32(i);
        }
        assert_eq!(pc.size(), 256);
    }

    #[test]
    fn clone_preserves_data() {
        let mut pc = PushConstants::new();
        pc.push_u32(42);
        pc.push_f32(2.718);
        let cloned = pc.clone();
        assert_eq!(cloned.size(), pc.size());
        assert_eq!(cloned.as_bytes(), pc.as_bytes());
    }

    #[test]
    fn sequential_pushes_accumulate() {
        let mut pc = PushConstants::new();
        assert_eq!(pc.size(), 0);
        pc.push_u32(1);
        assert_eq!(pc.size(), 4);
        pc.push_f32(2.0);
        assert_eq!(pc.size(), 8);
        pc.push_i32(-3);
        assert_eq!(pc.size(), 12);
    }
}

// ===========================================================================
// 4. BufferBinding
// ===========================================================================

mod buffer_binding {
    use super::*;

    #[test]
    fn construction_and_fields() {
        let bb = BufferBinding {
            binding: 0,
            offset: 64,
            size: 4096,
            read_only: true,
        };
        assert_eq!(bb.binding, 0);
        assert_eq!(bb.offset, 64);
        assert_eq!(bb.size, 4096);
        assert!(bb.read_only);
    }

    #[test]
    fn read_write_binding() {
        let bb = BufferBinding {
            binding: 1,
            offset: 0,
            size: 256,
            read_only: false,
        };
        assert!(!bb.read_only);
    }

    #[test]
    fn multiple_bindings_different_indices() {
        let bindings = [BufferBinding {
                binding: 0,
                offset: 0,
                size: 1024,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: 2048,
                read_only: false,
            },
            BufferBinding {
                binding: 2,
                offset: 512,
                size: 512,
                read_only: true,
            }];
        assert_eq!(bindings.len(), 3);
        for (i, bb) in bindings.iter().enumerate() {
            assert_eq!(bb.binding, i as u32);
        }
        assert_eq!(bindings[2].offset, 512);
    }

    #[test]
    fn large_buffer_size_u64() {
        let bb = BufferBinding {
            binding: 0,
            offset: 0,
            size: 1 << 30, // 1 GiB
            read_only: false,
        };
        assert_eq!(bb.size, 1_073_741_824);
    }

    #[test]
    fn max_offset() {
        let bb = BufferBinding {
            binding: 0,
            offset: u64::MAX,
            size: 0,
            read_only: true,
        };
        assert_eq!(bb.offset, u64::MAX);
    }

    #[test]
    fn clone_preserves_all_fields() {
        let bb = BufferBinding {
            binding: 3,
            offset: 128,
            size: 8192,
            read_only: true,
        };
        let cloned = bb.clone();
        assert_eq!(cloned.binding, bb.binding);
        assert_eq!(cloned.offset, bb.offset);
        assert_eq!(cloned.size, bb.size);
        assert_eq!(cloned.read_only, bb.read_only);
    }

    #[test]
    fn binding_index_validated_via_shader() {
        let shader = CompiledShader::new(minimal_spirv(), "main", 2, 0, [64, 1, 1]).unwrap();

        // Valid: indices 0 and 1.
        let valid_config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![
                BufferBinding {
                    binding: 0,
                    offset: 0,
                    size: 256,
                    read_only: true,
                },
                BufferBinding {
                    binding: 1,
                    offset: 0,
                    size: 256,
                    read_only: false,
                },
            ],
            push_constants: None,
        };
        assert!(shader.validate_dispatch(&valid_config).is_ok());

        // Invalid: index 2 out of range.
        let invalid_config = DispatchConfig {
            grid: [1, 1, 1],
            bindings: vec![
                BufferBinding {
                    binding: 0,
                    offset: 0,
                    size: 256,
                    read_only: true,
                },
                BufferBinding {
                    binding: 2,
                    offset: 0,
                    size: 256,
                    read_only: false,
                },
            ],
            push_constants: None,
        };
        assert!(shader.validate_dispatch(&invalid_config).is_err());
    }

    #[test]
    fn zero_size_binding() {
        let bb = BufferBinding {
            binding: 0,
            offset: 0,
            size: 0,
            read_only: true,
        };
        assert_eq!(bb.size, 0);
    }
}

// ===========================================================================
// 5. compute_grid_dims
// ===========================================================================

mod grid_dims {
    use super::*;

    // -- 1D workloads --

    #[test]
    fn exact_multiple_256() {
        assert_eq!(compute_grid_dims(256, [256, 1, 1]), [1, 1, 1]);
        assert_eq!(compute_grid_dims(512, [256, 1, 1]), [2, 1, 1]);
        assert_eq!(compute_grid_dims(1024, [256, 1, 1]), [4, 1, 1]);
    }

    #[test]
    fn single_element() {
        assert_eq!(compute_grid_dims(1, [256, 1, 1]), [1, 1, 1]);
    }

    #[test]
    fn elements_equal_workgroup_size() {
        assert_eq!(compute_grid_dims(64, [64, 1, 1]), [1, 1, 1]);
    }

    #[test]
    fn one_more_than_workgroup() {
        assert_eq!(compute_grid_dims(257, [256, 1, 1]), [2, 1, 1]);
    }

    #[test]
    fn one_less_than_workgroup() {
        assert_eq!(compute_grid_dims(255, [256, 1, 1]), [1, 1, 1]);
    }

    #[test]
    fn large_element_count() {
        // ceil(1_000_000 / 256) = 3907
        assert_eq!(compute_grid_dims(1_000_000, [256, 1, 1]), [3907, 1, 1]);
    }

    #[test]
    fn workgroup_size_64() {
        // ceil(1000/64) = 16
        assert_eq!(compute_grid_dims(1000, [64, 1, 1]), [16, 1, 1]);
    }

    #[test]
    fn workgroup_size_32() {
        // ceil(100/32) = 4
        assert_eq!(compute_grid_dims(100, [32, 1, 1]), [4, 1, 1]);
    }

    #[test]
    fn workgroup_size_1() {
        assert_eq!(compute_grid_dims(42, [1, 1, 1]), [42, 1, 1]);
    }

    #[test]
    fn zero_elements() {
        assert_eq!(compute_grid_dims(0, [256, 1, 1]), [0, 1, 1]);
    }

    // -- Edge cases with workgroup alignment --

    #[test]
    fn ceiling_division_various_cases() {
        let cases: &[(u32, u32, u32)] = &[
            (1, 1, 1),
            (1, 256, 1),
            (256, 256, 1),
            (257, 256, 2),
            (512, 256, 2),
            (513, 256, 3),
            (1023, 256, 4),
            (1024, 256, 4),
            (1025, 256, 5),
            (0, 256, 0),
            (65, 64, 2),
            (128, 128, 1),
            (129, 128, 2),
        ];
        for &(total, wg, expected) in cases {
            let grid = compute_grid_dims(total, [wg, 1, 1]);
            assert_eq!(
                grid[0], expected,
                "compute_grid_dims({total}, [{wg},1,1]) = {}, expected {expected}",
                grid[0]
            );
        }
    }

    // -- 2D/3D: y and z always 1 for this 1D helper --

    #[test]
    fn y_z_always_one() {
        for n in [1, 100, 1024, 65535] {
            let grid = compute_grid_dims(n, [128, 1, 1]);
            assert_eq!(grid[1], 1);
            assert_eq!(grid[2], 1);
        }
    }

    #[test]
    fn nonunit_yz_workgroup_ignored() {
        // compute_grid_dims only uses workgroup_size[0].
        let grid = compute_grid_dims(512, [64, 8, 4]);
        assert_eq!(grid, [8, 1, 1]);
    }

    // -- Division with remainder (ceiling division) --

    #[test]
    fn large_but_safe_element_count() {
        // Safe max for wg=256: u32::MAX - 256 so that (total + 256 - 1)
        // does not overflow. (total + wg) evaluated first by left-to-right
        // associativity before the -1.
        let total = u32::MAX - 256;
        let grid = compute_grid_dims(total, [256, 1, 1]);
        let expected = u64::from(total).div_ceil(256);
        assert_eq!(u64::from(grid[0]), expected);
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn max_u32_elements_overflows_in_debug() {
        // Known limitation: compute_grid_dims uses (total + wg - 1) / wg
        // which overflows u32 when total is close to u32::MAX and wg > 1.
        // In debug mode this panics; in release it wraps.
        let _ = compute_grid_dims(u32::MAX, [256, 1, 1]);
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn max_u32_elements_workgroup_1_overflows() {
        // Even wg=1 overflows: (u32::MAX + 1 - 1) evaluates (u32::MAX + 1) first.
        let _ = compute_grid_dims(u32::MAX, [1, 1, 1]);
    }

    #[test]
    fn max_safe_elements_workgroup_1() {
        // Largest safe total for wg=1: u32::MAX - 1. (u32::MAX - 1 + 1 - 1) = u32::MAX - 1.
        let grid = compute_grid_dims(u32::MAX - 1, [1, 1, 1]);
        assert_eq!(grid[0], u32::MAX - 1);
    }

    #[test]
    #[should_panic(expected = "workgroup_size[0] must be > 0")]
    fn panics_on_zero_workgroup_x() {
        let _ = compute_grid_dims(100, [0, 1, 1]);
    }
}

// ===========================================================================
// 6. spirv_words_to_bytes
// ===========================================================================

mod words_to_bytes {
    use super::*;

    #[test]
    fn empty_input() {
        assert!(spirv_words_to_bytes(&[]).is_empty());
    }

    #[test]
    fn single_word_magic() {
        let bytes = spirv_words_to_bytes(&[crate::spirv_emit::SPIRV_MAGIC]);
        assert_eq!(bytes, vec![0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn single_word_one() {
        let bytes = spirv_words_to_bytes(&[0x00000001]);
        assert_eq!(bytes, vec![0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn known_conversion_header() {
        let words: Vec<u32> = vec![0x07230203, 0x00010500, 0x00000000];
        let bytes = spirv_words_to_bytes(&words);
        assert_eq!(bytes.len(), 12);
        assert_eq!(&bytes[0..4], &[0x03, 0x02, 0x23, 0x07]);
        assert_eq!(&bytes[4..8], &[0x00, 0x05, 0x01, 0x00]);
        assert_eq!(&bytes[8..12], &[0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn five_header_words_produce_20_bytes() {
        let words: Vec<u32> = vec![0x07230203, 0x00010500, 0, 0, 0];
        assert_eq!(spirv_words_to_bytes(&words).len(), 20);
    }

    #[test]
    fn roundtrip_words_to_bytes_and_back() {
        let original: Vec<u32> = vec![0x07230203, 0x00010500, 42, 99, 0xDEADBEEF];
        let bytes = spirv_words_to_bytes(&original);
        let recovered: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(recovered, original);
    }

    #[test]
    fn roundtrip_large_word_stream() {
        let original: Vec<u32> = (0..1000).collect();
        let bytes = spirv_words_to_bytes(&original);
        assert_eq!(bytes.len(), 4000);
        let recovered: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(recovered, original);
    }

    #[test]
    fn max_u32_word() {
        let bytes = spirv_words_to_bytes(&[u32::MAX]);
        assert_eq!(bytes, vec![0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn zero_word() {
        let bytes = spirv_words_to_bytes(&[0u32]);
        assert_eq!(bytes, vec![0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn length_is_always_4x_word_count() {
        for n in [0, 1, 5, 10, 100, 500] {
            let words = vec![0u32; n];
            assert_eq!(spirv_words_to_bytes(&words).len(), n * 4);
        }
    }

    #[test]
    fn byte_order_is_little_endian() {
        // 0xAABBCCDD in LE: [DD, CC, BB, AA]
        let bytes = spirv_words_to_bytes(&[0xAABBCCDD]);
        assert_eq!(bytes, vec![0xDD, 0xCC, 0xBB, 0xAA]);
    }
}

// ===========================================================================
// 7. VulkanPipelineError
// ===========================================================================

mod pipeline_error {
    use super::*;

    #[test]
    fn spirv_validation_display() {
        let err = VulkanPipelineError::SpirvValidation {
            reason: "bad magic".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("bad magic"));
        assert!(msg.contains("SPIR-V validation"));
    }

    #[test]
    fn binding_out_of_range_display() {
        let err = VulkanPipelineError::BindingOutOfRange { index: 5, max: 3 };
        let msg = err.to_string();
        assert!(msg.contains("5"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn push_constant_overflow_display() {
        let err = VulkanPipelineError::PushConstantOverflow {
            actual: 256,
            declared: 128,
        };
        let msg = err.to_string();
        assert!(msg.contains("256"));
        assert!(msg.contains("128"));
    }

    #[test]
    fn workgroup_size_exceeded_display() {
        let err = VulkanPipelineError::WorkgroupSizeExceeded {
            product: 1024,
            limit: 128,
        };
        let msg = err.to_string();
        assert!(msg.contains("1024"));
        assert!(msg.contains("128"));
    }

    #[test]
    fn buffer_too_large_display() {
        let err = VulkanPipelineError::BufferTooLarge {
            requested: 1 << 30,
            max: 256 * 1024 * 1024,
        };
        let msg = err.to_string();
        assert!(msg.contains("exceeds"));
    }

    #[test]
    fn zero_grid_dimension_display() {
        let err = VulkanPipelineError::ZeroGridDimension { dim: "x" };
        let msg = err.to_string();
        assert!(msg.contains("x"));
        assert!(msg.contains("zero"));
    }

    #[test]
    fn no_device_display() {
        let err = VulkanPipelineError::NoDevice;
        let msg = err.to_string();
        assert!(msg.contains("no suitable"));
    }

    #[test]
    fn binding_count_mismatch_display() {
        let err = VulkanPipelineError::BindingCountMismatch {
            required: 4,
            provided: 2,
        };
        let msg = err.to_string();
        assert!(msg.contains("4"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn all_variants_debug_does_not_panic() {
        let variants: Vec<VulkanPipelineError> = vec![
            VulkanPipelineError::SpirvValidation {
                reason: "test".into(),
            },
            VulkanPipelineError::BindingOutOfRange { index: 0, max: 0 },
            VulkanPipelineError::PushConstantOverflow {
                actual: 0,
                declared: 0,
            },
            VulkanPipelineError::WorkgroupSizeExceeded {
                product: 0,
                limit: 0,
            },
            VulkanPipelineError::BufferTooLarge {
                requested: 0,
                max: 0,
            },
            VulkanPipelineError::ZeroGridDimension { dim: "x" },
            VulkanPipelineError::NoDevice,
            VulkanPipelineError::BindingCountMismatch {
                required: 0,
                provided: 0,
            },
        ];
        for v in &variants {
            let display = format!("{v}");
            let debug = format!("{v:?}");
            assert!(!display.is_empty());
            assert!(!debug.is_empty());
        }
    }

    #[test]
    fn error_implements_std_error() {
        let err = VulkanPipelineError::NoDevice;
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn zero_grid_dimension_all_axes() {
        for dim in ["x", "y", "z"] {
            let err = VulkanPipelineError::ZeroGridDimension { dim };
            let msg = err.to_string();
            assert!(msg.contains(dim));
        }
    }
}

// ===========================================================================
// 8. VulkanComputeConfig
// ===========================================================================

mod vulkan_compute_config {
    use super::*;

    #[test]
    fn default_values() {
        let config = VulkanComputeConfig::default();
        assert_eq!(config.max_buffer_size, 256 * 1024 * 1024);
        assert_eq!(config.workgroup_size_x, 256);
        assert_eq!(config.workgroup_size_y, 1);
        assert_eq!(config.workgroup_size_z, 1);
    }

    #[test]
    fn total_invocations_default() {
        let config = VulkanComputeConfig::default();
        assert_eq!(config.total_workgroup_invocations(), 256);
    }

    #[test]
    fn total_invocations_3d() {
        let config = VulkanComputeConfig {
            workgroup_size_x: 8,
            workgroup_size_y: 8,
            workgroup_size_z: 4,
            ..Default::default()
        };
        assert_eq!(config.total_workgroup_invocations(), 256);
    }

    #[test]
    fn total_invocations_16x16x1() {
        let config = VulkanComputeConfig {
            workgroup_size_x: 16,
            workgroup_size_y: 16,
            workgroup_size_z: 1,
            ..Default::default()
        };
        assert_eq!(config.total_workgroup_invocations(), 256);
    }

    #[test]
    fn total_invocations_saturates() {
        let config = VulkanComputeConfig {
            workgroup_size_x: u32::MAX,
            workgroup_size_y: 2,
            workgroup_size_z: 1,
            ..Default::default()
        };
        assert_eq!(config.total_workgroup_invocations(), u32::MAX);
    }

    #[test]
    fn total_invocations_saturates_all_max() {
        let config = VulkanComputeConfig {
            workgroup_size_x: u32::MAX,
            workgroup_size_y: u32::MAX,
            workgroup_size_z: u32::MAX,
            ..Default::default()
        };
        assert_eq!(config.total_workgroup_invocations(), u32::MAX);
    }

    #[test]
    fn clone_preserves_fields() {
        let config = VulkanComputeConfig {
            max_buffer_size: 1024,
            workgroup_size_x: 32,
            workgroup_size_y: 4,
            workgroup_size_z: 2,
            enable_validation: false,
        };
        let cloned = config;
        assert_eq!(cloned.max_buffer_size, 1024);
        assert_eq!(cloned.workgroup_size_x, 32);
        assert_eq!(cloned.workgroup_size_y, 4);
        assert_eq!(cloned.workgroup_size_z, 2);
        assert!(!cloned.enable_validation);
    }

    #[test]
    fn custom_max_buffer_size() {
        let config = VulkanComputeConfig {
            max_buffer_size: 64 * 1024, // 64 KiB
            ..Default::default()
        };
        assert_eq!(config.max_buffer_size, 65536);
    }
}
