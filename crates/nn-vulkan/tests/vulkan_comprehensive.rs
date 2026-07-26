// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Comprehensive integration tests for the nn-vulkan crate.
//!
//! Tests GLSL emission, SPIR-V structure, buffer management, dispatch
//! pipeline, device/error handling. All tests validate code generation
//! and structural correctness — no live Vulkan GPU required.

use nn_vulkan::buffer::{BufferUsage, StagingBuffer, VulkanBuffer};
use nn_vulkan::device::{is_vulkan_available, MemoryPropertyFlags, QueueFamilyInfo};
use nn_vulkan::dispatch::{
    ComputePipeline, DescriptorBinding, DescriptorSetLayout, DescriptorType, PipelineLayout,
    PushConstantRange, VulkanDispatcher,
};
use nn_vulkan::error::VulkanError;
use nn_vulkan::kernels::activations;
use nn_vulkan::spirv_emit::{
    emit_elementwise_glsl, emit_matmul_glsl, emit_reduction_glsl, emit_softmax_glsl, glsl_type,
    spirv_type_bytes, ReductionOp, DEFAULT_WORKGROUP_SIZE, GLSL_COMPUTE_VERSION, SPIRV_MAGIC,
    SPIRV_VERSION_1_5,
};

// ============================================================================
// A. GLSL / SPIR-V Emission Tests (elementwise activations + reductions)
// ============================================================================

/// Helper: assert common structure of all elementwise GLSL shaders.
fn assert_elementwise_glsl_structure(src: &str, kernel_name: &str) {
    assert!(
        src.contains("#version 450"),
        "{kernel_name}: missing GLSL version header"
    );
    assert!(
        src.contains("layout(local_size_x ="),
        "{kernel_name}: missing workgroup layout"
    );
    assert!(
        src.contains("gl_GlobalInvocationID"),
        "{kernel_name}: missing global invocation ID"
    );
    assert!(
        src.contains("void main()"),
        "{kernel_name}: missing main entry point"
    );
    assert!(
        src.contains("input_buffer"),
        "{kernel_name}: missing input buffer binding"
    );
    assert!(
        src.contains("output_buffer"),
        "{kernel_name}: missing output buffer binding"
    );
    assert!(
        src.contains("push_constant"),
        "{kernel_name}: missing push constants"
    );
    assert!(
        src.contains("total_elements"),
        "{kernel_name}: missing total_elements parameter"
    );
}

#[test]
fn test_glsl_relu_emission() {
    let src = emit_elementwise_glsl("relu", "max(x, 0.0)", 256).expect("relu emission");
    assert_elementwise_glsl_structure(&src, "relu");
    assert!(src.contains("max(x, 0.0)"), "relu: missing max expression");
}

#[test]
fn test_glsl_gelu_emission() {
    let src = emit_elementwise_glsl(
        "gelu",
        "0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x)))",
        256,
    )
    .expect("gelu emission");
    assert_elementwise_glsl_structure(&src, "gelu");
    assert!(src.contains("0.044715"), "gelu: missing gelu constant");
    assert!(src.contains("tanh"), "gelu: missing tanh call");
}

#[test]
fn test_glsl_silu_emission() {
    let src = emit_elementwise_glsl("silu", "x / (1.0 + exp(-x))", 256).expect("silu emission");
    assert_elementwise_glsl_structure(&src, "silu");
    assert!(src.contains("exp(-x)"), "silu: missing exp(-x)");
}

#[test]
fn test_glsl_sigmoid_emission() {
    let src =
        emit_elementwise_glsl("sigmoid", "1.0 / (1.0 + exp(-x))", 256).expect("sigmoid emission");
    assert_elementwise_glsl_structure(&src, "sigmoid");
    assert!(
        src.contains("1.0 / (1.0 + exp(-x))"),
        "sigmoid: missing sigmoid expression"
    );
}

#[test]
fn test_glsl_tanh_emission() {
    let src = emit_elementwise_glsl("tanh_act", "tanh(x)", 256).expect("tanh emission");
    assert_elementwise_glsl_structure(&src, "tanh_act");
    assert!(src.contains("tanh(x)"), "tanh: missing tanh(x)");
}

#[test]
fn test_glsl_snake_emission() {
    let src = emit_elementwise_glsl("snake", "x + sin(x) * sin(x)", 256).expect("snake emission");
    assert_elementwise_glsl_structure(&src, "snake");
    assert!(src.contains("sin(x)"), "snake: missing sin(x)");
}

#[test]
fn test_glsl_leaky_relu_emission() {
    let src =
        emit_elementwise_glsl("leaky_relu", "x > 0.0 ? x : 0.01 * x", 256).expect("leaky_relu");
    assert_elementwise_glsl_structure(&src, "leaky_relu");
    assert!(
        src.contains("0.01 * x"),
        "leaky_relu: missing negative slope"
    );
}

#[test]
fn test_glsl_swish_emission() {
    let src = emit_elementwise_glsl("swish", "x / (1.0 + exp(-x))", 256).expect("swish emission");
    assert_elementwise_glsl_structure(&src, "swish");
    assert!(src.contains("exp(-x)"), "swish: missing exp(-x)");
}

#[test]
fn test_glsl_mish_emission() {
    let src =
        emit_elementwise_glsl("mish", "x * tanh(log(1.0 + exp(x)))", 256).expect("mish emission");
    assert_elementwise_glsl_structure(&src, "mish");
    assert!(src.contains("tanh"), "mish: missing tanh");
    assert!(src.contains("log"), "mish: missing log");
    assert!(src.contains("exp"), "mish: missing exp");
}

#[test]
fn test_glsl_softplus_emission() {
    let src = emit_elementwise_glsl("softplus", "log(1.0 + exp(x))", 256).expect("softplus");
    assert_elementwise_glsl_structure(&src, "softplus");
    assert!(
        src.contains("log(1.0 + exp(x))"),
        "softplus: missing log-exp expression"
    );
}

#[test]
fn test_glsl_hardswish_emission() {
    let src = emit_elementwise_glsl("hardswish", "x * clamp(x / 6.0 + 0.5, 0.0, 1.0)", 256)
        .expect("hardswish");
    assert_elementwise_glsl_structure(&src, "hardswish");
    assert!(src.contains("clamp"), "hardswish: missing clamp");
}

#[test]
fn test_glsl_elu_emission() {
    let src =
        emit_elementwise_glsl("elu", "x >= 0.0 ? x : 1.0 * (exp(x) - 1.0)", 256).expect("elu");
    assert_elementwise_glsl_structure(&src, "elu");
    assert!(src.contains("exp(x)"), "elu: missing exp(x)");
}

#[test]
fn test_glsl_elementwise_custom_workgroup_size() {
    for wg in [1, 32, 64, 128, 512, 1024] {
        let src = emit_elementwise_glsl("custom", "x", wg).expect("custom workgroup");
        assert!(
            src.contains(&format!("local_size_x = {wg}")),
            "missing workgroup size {wg}"
        );
    }
}

#[test]
fn test_glsl_elementwise_zero_workgroup_rejected() {
    let result = emit_elementwise_glsl("bad", "x", 0);
    assert!(result.is_err(), "zero workgroup should be rejected");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanError::InvalidParameter(_)),
        "expected InvalidParameter, got {err:?}"
    );
}

#[test]
fn test_glsl_softmax_emission() {
    let src = emit_softmax_glsl(256).expect("softmax emission");
    assert!(src.contains("#version 450"), "softmax: missing version");
    assert!(
        src.contains("layout(local_size_x ="),
        "softmax: missing workgroup layout"
    );
    assert!(src.contains("smax"), "softmax: missing shared max array");
    assert!(src.contains("ssum"), "softmax: missing shared sum array");
    assert!(
        src.contains("barrier()"),
        "softmax: missing barrier synchronization"
    );
    assert!(
        src.contains("exp("),
        "softmax: missing exp for normalization"
    );
    assert!(
        src.contains("row_max"),
        "softmax: missing row_max computation"
    );
    assert!(
        src.contains("row_sum"),
        "softmax: missing row_sum computation"
    );
    assert!(
        src.contains("num_rows"),
        "softmax: missing num_rows push constant"
    );
    assert!(
        src.contains("row_size"),
        "softmax: missing row_size push constant"
    );
}

#[test]
fn test_glsl_softmax_non_power_of_2_rejected() {
    let result = emit_softmax_glsl(100);
    assert!(result.is_err(), "non-power-of-2 workgroup should fail");
}

#[test]
fn test_glsl_softmax_zero_workgroup_rejected() {
    let result = emit_softmax_glsl(0);
    assert!(result.is_err(), "zero workgroup should fail");
}

#[test]
fn test_glsl_reduction_sum_emission() {
    let src = emit_reduction_glsl("sum_reduce", ReductionOp::Sum, 256).expect("sum reduction");
    assert!(src.contains("#version 450"), "sum_reduce: missing version");
    assert!(
        src.contains("shared float sdata"),
        "sum_reduce: missing shared memory"
    );
    assert!(src.contains("barrier()"), "sum_reduce: missing barrier");
    // Identity for sum is 0.0
    assert!(src.contains("0.0"), "sum_reduce: missing identity element");
}

#[test]
fn test_glsl_reduction_max_emission() {
    let src = emit_reduction_glsl("max_reduce", ReductionOp::Max, 128).expect("max reduction");
    assert!(src.contains("max(a, b)"), "max_reduce: missing max combine");
    // Identity for max is -inf
    assert!(
        src.contains("-1.0 / 0.0"),
        "max_reduce: missing -inf identity"
    );
}

#[test]
fn test_glsl_reduction_min_emission() {
    let src = emit_reduction_glsl("min_reduce", ReductionOp::Min, 64).expect("min reduction");
    assert!(src.contains("min(a, b)"), "min_reduce: missing min combine");
    // Identity for min is +inf
    assert!(
        src.contains("1.0 / 0.0"),
        "min_reduce: missing +inf identity"
    );
}

#[test]
fn test_glsl_reduction_non_power_of_2_rejected() {
    let result = emit_reduction_glsl("bad", ReductionOp::Sum, 100);
    assert!(
        result.is_err(),
        "non-power-of-2 workgroup should fail for reduction"
    );
}

#[test]
fn test_glsl_reduction_zero_workgroup_rejected() {
    let result = emit_reduction_glsl("bad", ReductionOp::Sum, 0);
    assert!(result.is_err(), "zero workgroup should fail for reduction");
}

#[test]
fn test_glsl_matmul_emission() {
    let src = emit_matmul_glsl(16).expect("matmul emission");
    assert!(src.contains("#version 450"), "matmul: missing version");
    assert!(
        src.contains("local_size_x = 16"),
        "matmul: wrong tile size in local_size_x"
    );
    assert!(
        src.contains("local_size_y = 16"),
        "matmul: wrong tile size in local_size_y"
    );
    assert!(src.contains("tileA"), "matmul: missing shared tileA");
    assert!(src.contains("tileB"), "matmul: missing shared tileB");
    assert!(
        src.contains("barrier()"),
        "matmul: missing barrier synchronization"
    );
    assert!(src.contains("params.M"), "matmul: missing M push constant");
    assert!(src.contains("params.N"), "matmul: missing N push constant");
    assert!(src.contains("params.K"), "matmul: missing K push constant");
    assert!(src.contains("acc"), "matmul: missing accumulator");
}

#[test]
fn test_glsl_matmul_various_tile_sizes() {
    for tile in [1, 2, 4, 8, 16, 32] {
        let src = emit_matmul_glsl(tile).unwrap_or_else(|e| panic!("tile={tile} failed: {e}"));
        assert!(
            src.contains(&format!("local_size_x = {tile}")),
            "tile={tile}: wrong local_size_x"
        );
    }
}

#[test]
fn test_glsl_matmul_non_power_of_2_rejected() {
    let result = emit_matmul_glsl(12);
    assert!(
        result.is_err(),
        "non-power-of-2 tile size should fail for matmul"
    );
}

#[test]
fn test_glsl_matmul_zero_tile_rejected() {
    let result = emit_matmul_glsl(0);
    assert!(result.is_err(), "zero tile size should fail for matmul");
}

// ============================================================================
// A.extra: Pre-built activation kernel tests (kernels/activations module)
// ============================================================================

#[test]
fn test_prebuilt_relu_uses_default_workgroup() {
    let src = activations::relu_glsl().expect("relu");
    assert!(src.contains(&format!("local_size_x = {DEFAULT_WORKGROUP_SIZE}")));
}

#[test]
fn test_prebuilt_silu_uses_default_workgroup() {
    let src = activations::silu_glsl().expect("silu");
    assert!(src.contains(&format!("local_size_x = {DEFAULT_WORKGROUP_SIZE}")));
}

#[test]
fn test_prebuilt_tanh_uses_default_workgroup() {
    let src = activations::tanh_glsl().expect("tanh");
    assert!(src.contains(&format!("local_size_x = {DEFAULT_WORKGROUP_SIZE}")));
}

// ============================================================================
// B. SPIR-V Structure Tests
// ============================================================================

#[test]
fn test_spirv_magic_number() {
    assert_eq!(SPIRV_MAGIC, 0x0723_0203, "SPIR-V magic number mismatch");
}

#[test]
fn test_spirv_version_1_5() {
    // SPIR-V version 1.5 is encoded as 0x0001_0500
    assert_eq!(SPIRV_VERSION_1_5, 0x0001_0500, "SPIR-V 1.5 encoding");
    // Major = 1
    let major = (SPIRV_VERSION_1_5 >> 16) & 0xFF;
    assert_eq!(major, 1);
    // Minor = 5
    let minor = (SPIRV_VERSION_1_5 >> 8) & 0xFF;
    assert_eq!(minor, 5);
}

#[test]
fn test_glsl_compute_version_string() {
    assert_eq!(GLSL_COMPUTE_VERSION, "#version 450\n");
}

#[test]
fn test_default_workgroup_size() {
    assert_eq!(DEFAULT_WORKGROUP_SIZE, 256);
    assert!(DEFAULT_WORKGROUP_SIZE.is_power_of_two());
}

#[test]
fn test_glsl_type_f32() {
    use nn_dsl::ScalarType;
    assert_eq!(glsl_type(ScalarType::F32).expect("f32"), "float");
}

#[test]
fn test_glsl_type_f16() {
    use nn_dsl::ScalarType;
    assert_eq!(glsl_type(ScalarType::F16).expect("f16"), "float16_t");
}

#[test]
fn test_glsl_type_bf16_unsupported() {
    use nn_dsl::ScalarType;
    let result = glsl_type(ScalarType::BF16);
    assert!(result.is_err(), "bf16 should not have a native GLSL type");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanError::UnsupportedType { .. }),
        "expected UnsupportedType, got {err:?}"
    );
}

#[test]
fn test_spirv_type_bytes_f32() {
    use nn_dsl::ScalarType;
    assert_eq!(spirv_type_bytes(ScalarType::F32).expect("f32"), 4);
}

#[test]
fn test_spirv_type_bytes_f16() {
    use nn_dsl::ScalarType;
    assert_eq!(spirv_type_bytes(ScalarType::F16).expect("f16"), 2);
}

#[test]
fn test_spirv_type_bytes_bf16() {
    use nn_dsl::ScalarType;
    assert_eq!(spirv_type_bytes(ScalarType::BF16).expect("bf16"), 2);
}

// ============================================================================
// C. Buffer Management Tests
// ============================================================================

#[test]
fn test_vulkan_buffer_creation_various_sizes() {
    for size in [1, 64, 1024, 1_048_576, 256 * 1024 * 1024] {
        let buf = VulkanBuffer::new(size, BufferUsage::StorageRead).expect("buffer creation");
        assert_eq!(buf.size_bytes(), size);
        assert_eq!(buf.usage(), BufferUsage::StorageRead);
    }
}

#[test]
fn test_vulkan_buffer_zero_size_rejected() {
    let result = VulkanBuffer::new(0, BufferUsage::StorageRead);
    assert!(result.is_err(), "zero-size buffer should be rejected");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanError::InvalidParameter(_)),
        "expected InvalidParameter, got {err:?}"
    );
}

#[test]
fn test_vulkan_buffer_usage_variants() {
    let usages = [
        BufferUsage::StorageRead,
        BufferUsage::StorageReadWrite,
        BufferUsage::Uniform,
        BufferUsage::TransferSrc,
        BufferUsage::TransferDst,
    ];
    for usage in usages {
        let buf = VulkanBuffer::new(256, usage).expect("buffer creation");
        assert_eq!(buf.usage(), usage);
    }
}

#[test]
fn test_vulkan_buffer_handle_initialized() {
    let buf = VulkanBuffer::new(128, BufferUsage::StorageRead).expect("buffer creation");
    // Placeholder handle is 0
    assert_eq!(buf.handle(), 0);
}

#[test]
fn test_buffer_usage_vk_bits() {
    // VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 0x00000020
    assert_eq!(BufferUsage::StorageRead.to_vk_bits(), 0x0000_0020);
    assert_eq!(BufferUsage::StorageReadWrite.to_vk_bits(), 0x0000_0020);
    // VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT = 0x00000010
    assert_eq!(BufferUsage::Uniform.to_vk_bits(), 0x0000_0010);
    // VK_BUFFER_USAGE_TRANSFER_SRC_BIT = 0x00000001
    assert_eq!(BufferUsage::TransferSrc.to_vk_bits(), 0x0000_0001);
    // VK_BUFFER_USAGE_TRANSFER_DST_BIT = 0x00000002
    assert_eq!(BufferUsage::TransferDst.to_vk_bits(), 0x0000_0002);
}

#[test]
fn test_staging_buffer_upload_creation() {
    let staging = StagingBuffer::new_upload(4096).expect("staging upload creation");
    assert_eq!(staging.size_bytes(), 4096);
    assert!(staging.is_upload());
}

#[test]
fn test_staging_buffer_download_creation() {
    let staging = StagingBuffer::new_download(4096).expect("staging download creation");
    assert_eq!(staging.size_bytes(), 4096);
    assert!(!staging.is_upload());
}

#[test]
fn test_staging_buffer_zero_size_rejected() {
    let upload = StagingBuffer::new_upload(0);
    assert!(upload.is_err(), "zero-size upload staging should fail");
    let download = StagingBuffer::new_download(0);
    assert!(download.is_err(), "zero-size download staging should fail");
}

#[test]
fn test_staging_buffer_write_f32_within_bounds() {
    let mut staging = StagingBuffer::new_upload(1024).expect("staging creation");
    let data = vec![1.0_f32; 256]; // 256 * 4 = 1024 bytes
    staging.write_f32(&data).expect("write_f32 should succeed");
}

#[test]
fn test_staging_buffer_write_f32_overflow() {
    let mut staging = StagingBuffer::new_upload(100).expect("staging creation");
    let data = vec![1.0_f32; 100]; // 100 * 4 = 400 bytes > 100
    let result = staging.write_f32(&data);
    assert!(result.is_err(), "write beyond capacity should fail");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanError::BufferSizeMismatch { .. }),
        "expected BufferSizeMismatch, got {err:?}"
    );
}

#[test]
fn test_staging_buffer_read_f32_within_bounds() {
    let staging = StagingBuffer::new_download(1024).expect("staging creation");
    let data = staging.read_f32(256).expect("read_f32 should succeed");
    assert_eq!(data.len(), 256);
    // Placeholder returns zeros
    assert!(data.iter().all(|&v| v == 0.0));
}

#[test]
fn test_staging_buffer_read_f32_overflow() {
    let staging = StagingBuffer::new_download(100).expect("staging creation");
    let result = staging.read_f32(100); // 100 * 4 = 400 > 100
    assert!(result.is_err(), "read beyond capacity should fail");
}

#[test]
fn test_staging_buffer_roundtrip_pattern() {
    // Simulate the staging buffer pattern: upload -> device -> download
    let upload_data = vec![3.14_f32; 64];
    let byte_size = upload_data.len() * 4;

    // Step 1: Create upload staging and write host data
    let mut upload_staging = StagingBuffer::new_upload(byte_size).expect("upload staging");
    upload_staging
        .write_f32(&upload_data)
        .expect("write to upload staging");

    // Step 2: Create device-local buffer (copy target)
    let device_buf =
        VulkanBuffer::new(byte_size, BufferUsage::StorageReadWrite).expect("device buffer");
    assert_eq!(device_buf.size_bytes(), byte_size);

    // Step 3: Create download staging and read back
    let download_staging = StagingBuffer::new_download(byte_size).expect("download staging");
    let readback = download_staging
        .read_f32(64)
        .expect("read from download staging");
    assert_eq!(readback.len(), 64);
    // Placeholder always returns zeros (no real GPU), but the API pattern works
}

#[test]
fn test_staging_buffer_write_empty_slice() {
    let mut staging = StagingBuffer::new_upload(64).expect("staging creation");
    // Empty write should succeed (0 bytes <= capacity)
    staging.write_f32(&[]).expect("empty write should succeed");
}

#[test]
fn test_staging_buffer_read_zero_count() {
    let staging = StagingBuffer::new_download(64).expect("staging creation");
    let data = staging.read_f32(0).expect("zero-count read should succeed");
    assert!(data.is_empty());
}

// ============================================================================
// D. Dispatch Tests
// ============================================================================

#[test]
fn test_descriptor_type_vk_values() {
    // VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
    assert_eq!(DescriptorType::StorageBuffer.to_vk_type(), 7);
    // VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER = 6
    assert_eq!(DescriptorType::UniformBuffer.to_vk_type(), 6);
}

#[test]
fn test_descriptor_set_layout_single_binding() {
    let layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("single binding layout");
    assert_eq!(layout.binding_count(), 1);
}

#[test]
fn test_descriptor_set_layout_multiple_bindings() {
    let layout = DescriptorSetLayout::new(vec![
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 1,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 2,
            descriptor_type: DescriptorType::UniformBuffer,
            count: 1,
        },
    ])
    .expect("multi-binding layout");
    assert_eq!(layout.binding_count(), 3);

    let bindings = layout.bindings();
    assert_eq!(bindings[0].binding, 0);
    assert_eq!(bindings[1].binding, 1);
    assert_eq!(bindings[2].binding, 2);
    assert_eq!(bindings[2].descriptor_type, DescriptorType::UniformBuffer);
}

#[test]
fn test_descriptor_set_layout_empty_rejected() {
    let result = DescriptorSetLayout::new(vec![]);
    assert!(result.is_err(), "empty layout should be rejected");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanError::DescriptorSetError { .. }),
        "expected DescriptorSetError, got {err:?}"
    );
}

#[test]
fn test_descriptor_set_layout_duplicate_binding_rejected() {
    let result = DescriptorSetLayout::new(vec![
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::UniformBuffer,
            count: 1,
        },
    ]);
    assert!(
        result.is_err(),
        "duplicate binding index should be rejected"
    );
}

#[test]
fn test_pipeline_layout_valid_push_constants() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");

    for size in [0, 4, 8, 16, 64, 128] {
        let pl = PipelineLayout::new(&ds_layout, size)
            .unwrap_or_else(|e| panic!("push constant size {size} should be valid: {e}"));
        assert_eq!(pl.push_constant_size(), size);
    }
}

#[test]
fn test_pipeline_layout_non_multiple_of_4_rejected() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");

    for bad_size in [1, 2, 3, 5, 7, 13, 127] {
        let result = PipelineLayout::new(&ds_layout, bad_size);
        assert!(
            result.is_err(),
            "push constant size {bad_size} (not multiple of 4) should fail"
        );
    }
}

#[test]
fn test_pipeline_layout_exceeds_128_byte_limit() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");

    let result = PipelineLayout::new(&ds_layout, 132);
    assert!(result.is_err(), "132 bytes exceeds 128-byte guarantee");
    let result = PipelineLayout::new(&ds_layout, 256);
    assert!(result.is_err(), "256 bytes exceeds 128-byte guarantee");
}

#[test]
fn test_compute_pipeline_creation_with_valid_spirv() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 8).expect("pipeline layout");

    // Minimal SPIR-V: magic + version + more words
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline creation");
    assert_eq!(pipeline.entry_point(), "main");
    assert_eq!(pipeline.workgroup_size(), [DEFAULT_WORKGROUP_SIZE, 1, 1]);
}

#[test]
fn test_compute_pipeline_empty_spirv_rejected() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 8).expect("pipeline layout");

    let result = ComputePipeline::new(&[], "main", &pl);
    assert!(result.is_err(), "empty SPIR-V should be rejected");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanError::ShaderCompilationFailed { .. }),
        "expected ShaderCompilationFailed, got {err:?}"
    );
}

#[test]
fn test_compute_pipeline_bad_magic_rejected() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 8).expect("pipeline layout");

    let bad_spirv = vec![0xDEAD_BEEF, 0, 0, 0, 0];
    let result = ComputePipeline::new(&bad_spirv, "main", &pl);
    assert!(result.is_err(), "invalid SPIR-V magic should be rejected");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanError::ShaderCompilationFailed { .. }),
        "expected ShaderCompilationFailed, got {err:?}"
    );
}

#[test]
fn test_vulkan_dispatcher_creation() {
    let dispatcher = VulkanDispatcher::new().expect("dispatcher creation");
    assert_eq!(dispatcher.dispatch_count(), 0);
}

#[test]
fn test_dispatcher_dispatch_increments_count() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 8).expect("pipeline layout");
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");

    let buf = VulkanBuffer::new(256, BufferUsage::StorageReadWrite).expect("buffer");
    let push = [0u8; 8];

    let mut dispatcher = VulkanDispatcher::new().expect("dispatcher");
    assert_eq!(dispatcher.dispatch_count(), 0);

    dispatcher
        .dispatch(&pipeline, &[&buf], &push, [1, 1, 1])
        .expect("dispatch 1");
    assert_eq!(dispatcher.dispatch_count(), 1);

    dispatcher
        .dispatch(&pipeline, &[&buf], &push, [4, 1, 1])
        .expect("dispatch 2");
    assert_eq!(dispatcher.dispatch_count(), 2);
}

#[test]
fn test_dispatcher_zero_group_count_rejected() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 8).expect("pipeline layout");
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");
    let buf = VulkanBuffer::new(256, BufferUsage::StorageReadWrite).expect("buffer");
    let push = [0u8; 8];

    let mut dispatcher = VulkanDispatcher::new().expect("dispatcher");

    // Zero in X dimension
    let result = dispatcher.dispatch(&pipeline, &[&buf], &push, [0, 1, 1]);
    assert!(result.is_err(), "zero workgroup X should fail");

    // Zero in Y dimension
    let result = dispatcher.dispatch(&pipeline, &[&buf], &push, [1, 0, 1]);
    assert!(result.is_err(), "zero workgroup Y should fail");

    // Zero in Z dimension
    let result = dispatcher.dispatch(&pipeline, &[&buf], &push, [1, 1, 0]);
    assert!(result.is_err(), "zero workgroup Z should fail");
}

#[test]
fn test_dispatcher_submit_without_dispatch_rejected() {
    let dispatcher = VulkanDispatcher::new().expect("dispatcher");
    let result = dispatcher.submit_and_wait();
    assert!(result.is_err(), "submit without dispatches should fail");
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanError::CommandBufferError { .. }),
        "expected CommandBufferError, got {err:?}"
    );
}

#[test]
fn test_dispatcher_submit_after_dispatch() {
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 8).expect("pipeline layout");
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");
    let buf = VulkanBuffer::new(256, BufferUsage::StorageReadWrite).expect("buffer");
    let push = [0u8; 8];

    let mut dispatcher = VulkanDispatcher::new().expect("dispatcher");
    dispatcher
        .dispatch(&pipeline, &[&buf], &push, [1, 1, 1])
        .expect("dispatch");
    dispatcher
        .submit_and_wait()
        .expect("submit should succeed after dispatch");
}

#[test]
fn test_workgroup_count_calculation_pattern() {
    // Demonstrate the standard workgroup count calculation
    let total_elements: u32 = 10_000;
    let workgroup_size: u32 = DEFAULT_WORKGROUP_SIZE;
    let group_count_x = total_elements.div_ceil(workgroup_size);
    assert_eq!(group_count_x, 40); // ceil(10000/256) = 40
}

#[test]
fn test_workgroup_count_exact_multiple() {
    let total = 1024_u32;
    let wg_size = 256_u32;
    let groups = total.div_ceil(wg_size);
    assert_eq!(groups, 4);
}

#[test]
fn test_workgroup_count_one_element() {
    let total = 1_u32;
    let wg_size = DEFAULT_WORKGROUP_SIZE;
    let groups = total.div_ceil(wg_size);
    assert_eq!(groups, 1);
}

// ============================================================================
// E. Device / Error Tests
// ============================================================================

#[test]
fn test_is_vulkan_available_returns_bool() {
    // On CI/macOS without MoltenVK this returns false — that is valid.
    let available = is_vulkan_available();
    // Just ensure it does not panic and returns a bool.
    assert!(!available || available);
}

#[test]
fn test_vulkan_device_not_available() {
    // Without Vulkan runtime, VulkanDevice::new should return NotAvailable.
    use nn_vulkan::device::VulkanDevice;
    let result = VulkanDevice::new(0);
    // On systems without Vulkan, this should be an error.
    if !is_vulkan_available() {
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, VulkanError::NotAvailable),
            "expected NotAvailable, got {err:?}"
        );
    }
}

#[test]
fn test_memory_property_flags_vk_bits() {
    // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 0x00000001
    assert_eq!(MemoryPropertyFlags::DeviceLocal.to_vk_bits(), 0x0000_0001);
    // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002
    assert_eq!(MemoryPropertyFlags::HostVisible.to_vk_bits(), 0x0000_0002);
    // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004
    assert_eq!(MemoryPropertyFlags::HostCoherent.to_vk_bits(), 0x0000_0004);
}

#[test]
fn test_queue_family_info_structure() {
    let qf = QueueFamilyInfo {
        index: 0,
        queue_count: 16,
        supports_compute: true,
        supports_transfer: true,
    };
    assert_eq!(qf.index, 0);
    assert_eq!(qf.queue_count, 16);
    assert!(qf.supports_compute);
    assert!(qf.supports_transfer);
}

#[test]
fn test_queue_family_info_compute_only() {
    let qf = QueueFamilyInfo {
        index: 2,
        queue_count: 4,
        supports_compute: true,
        supports_transfer: false,
    };
    assert!(qf.supports_compute);
    assert!(!qf.supports_transfer);
}

#[test]
fn test_vulkan_error_display() {
    let errors: Vec<VulkanError> = vec![
        VulkanError::NotAvailable,
        VulkanError::NoDevices,
        VulkanError::NoComputeQueue,
        VulkanError::NoSuitableMemoryType { flags: 0x1 },
        VulkanError::OutOfMemory { requested: 1024 },
        VulkanError::BufferSizeMismatch {
            expected: 100,
            actual: 200,
        },
        VulkanError::SpirVCodegen {
            reason: "test".into(),
        },
        VulkanError::PipelineCreation {
            reason: "test".into(),
        },
        VulkanError::DescriptorSetError {
            reason: "test".into(),
        },
        VulkanError::CommandBufferError {
            reason: "test".into(),
        },
        VulkanError::ShaderCompilationFailed {
            reason: "test".into(),
        },
        VulkanError::UnsupportedStep {
            step_name: "test_step",
        },
        VulkanError::UnsupportedType {
            type_desc: "test_type",
        },
        VulkanError::InvalidParameter("test".into()),
    ];

    for err in &errors {
        // Verify Display works and produces non-empty strings
        let msg = format!("{err}");
        assert!(
            !msg.is_empty(),
            "error display should not be empty: {err:?}"
        );
    }
}

#[test]
fn test_vulkan_error_debug() {
    let err = VulkanError::NotAvailable;
    let debug = format!("{err:?}");
    assert!(debug.contains("NotAvailable"));
}

#[test]
fn test_vulkan_error_io_conversion() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
    let vk_err: VulkanError = io_err.into();
    assert!(
        matches!(vk_err, VulkanError::Io(_)),
        "expected Io variant, got {vk_err:?}"
    );
    let msg = format!("{vk_err}");
    assert!(msg.contains("file not found"));
}

#[test]
fn test_push_constant_range_structure() {
    let range = PushConstantRange {
        offset: 0,
        size: 16,
    };
    assert_eq!(range.offset, 0);
    assert_eq!(range.size, 16);
}

#[test]
fn test_push_constant_range_with_offset() {
    let range = PushConstantRange {
        offset: 64,
        size: 8,
    };
    assert_eq!(range.offset, 64);
    assert_eq!(range.size, 8);
}

// ============================================================================
// F. End-to-end pattern tests (code generation + dispatch pipeline)
// ============================================================================

#[test]
fn test_end_to_end_relu_pipeline_setup() {
    // Generate GLSL
    let glsl = activations::relu_glsl().expect("relu glsl");
    assert!(glsl.contains("max(x, 0.0)"));

    // Set up descriptor layout (input + output buffers)
    let ds_layout = DescriptorSetLayout::new(vec![
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 1,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
    ])
    .expect("ds layout for relu");
    assert_eq!(ds_layout.binding_count(), 2);

    // Pipeline layout with push constants for total_elements (1 uint = 4 bytes)
    let pl = PipelineLayout::new(&ds_layout, 4).expect("pipeline layout");
    assert_eq!(pl.push_constant_size(), 4);

    // Create mock SPIR-V (in real usage this would be compiled from GLSL)
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");
    assert_eq!(pipeline.entry_point(), "main");

    // Create buffers
    let n_elements: usize = 1024;
    let byte_size = n_elements * 4; // f32
    let input_buf = VulkanBuffer::new(byte_size, BufferUsage::StorageRead).expect("input buffer");
    let output_buf =
        VulkanBuffer::new(byte_size, BufferUsage::StorageReadWrite).expect("output buffer");

    // Dispatch
    let total_elements = n_elements as u32;
    let group_count_x = total_elements.div_ceil(DEFAULT_WORKGROUP_SIZE);
    let push_data = total_elements.to_le_bytes();

    let mut dispatcher = VulkanDispatcher::new().expect("dispatcher");
    dispatcher
        .dispatch(
            &pipeline,
            &[&input_buf, &output_buf],
            &push_data,
            [group_count_x, 1, 1],
        )
        .expect("relu dispatch");
    assert_eq!(dispatcher.dispatch_count(), 1);

    dispatcher.submit_and_wait().expect("submit");
}

#[test]
fn test_end_to_end_matmul_pipeline_setup() {
    // Generate GLSL for matmul
    let tile = 16;
    let glsl = emit_matmul_glsl(tile).expect("matmul glsl");
    assert!(glsl.contains("tileA"));

    // Matmul needs 3 buffers: A, B, C
    let ds_layout = DescriptorSetLayout::new(vec![
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 1,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 2,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
    ])
    .expect("matmul ds layout");
    assert_eq!(ds_layout.binding_count(), 3);

    // Push constants: M, N, K (3 uints = 12 bytes)
    let pl = PipelineLayout::new(&ds_layout, 12).expect("matmul pipeline layout");
    assert_eq!(pl.push_constant_size(), 12);
}

#[test]
fn test_end_to_end_reduction_pipeline_setup() {
    let glsl = emit_reduction_glsl("nn_sum", ReductionOp::Sum, 256).expect("sum reduction glsl");
    assert!(glsl.contains("shared float sdata"));

    // Reduction: input + output, push constants for row_size and num_rows (8 bytes)
    let ds_layout = DescriptorSetLayout::new(vec![
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 1,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
    ])
    .expect("reduction ds layout");

    let pl = PipelineLayout::new(&ds_layout, 8).expect("reduction pipeline layout");
    assert_eq!(pl.push_constant_size(), 8);
}

#[test]
fn test_reduction_op_debug_formatting() {
    assert_eq!(format!("{:?}", ReductionOp::Sum), "Sum");
    assert_eq!(format!("{:?}", ReductionOp::Max), "Max");
    assert_eq!(format!("{:?}", ReductionOp::Min), "Min");
}

#[test]
fn test_reduction_op_clone_and_eq() {
    let a = ReductionOp::Sum;
    let b = a;
    assert_eq!(a, b);
    assert_ne!(ReductionOp::Sum, ReductionOp::Max);
    assert_ne!(ReductionOp::Max, ReductionOp::Min);
}

// ============================================================================
// G. Pipeline Cache Integration Tests
// ============================================================================

#[test]
fn test_pipeline_cache_end_to_end() {
    use nn_vulkan::pipeline_cache::{compile_or_cache, PipelineCache};

    let mut cache = PipelineCache::new();
    // Use unique GLSL to avoid L2 cache pollution from parallel unit tests.
    let glsl = format!(
        "// integration_test_{}\nvoid main() {{ }}",
        std::process::id()
    );
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let ds_layout = DescriptorSetLayout::new(vec![
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 1,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
    ])
    .expect("ds layout");

    // First compile: either L2 hit (from same-process test) or miss.
    let p1 =
        compile_or_cache(&mut cache, &glsl, "main", &spirv, &ds_layout, 4).expect("first compile");
    let stats1 = cache.stats();
    // First call must have exactly one non-L1-hit.
    assert_eq!(stats1.l1_hits, 0, "first call should not be an L1 hit");

    // Second compile: guaranteed L1 hit.
    let p2 =
        compile_or_cache(&mut cache, &glsl, "main", &spirv, &ds_layout, 4).expect("cached compile");
    assert_eq!(cache.stats().l1_hits, 1);
    assert_eq!(p1.entry_point(), p2.entry_point());
}

#[test]
fn test_pipeline_cache_different_kernels() {
    use nn_vulkan::pipeline_cache::{compile_or_cache, PipelineCache};

    let mut cache = PipelineCache::new();
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");

    // Use unique GLSL strings to avoid L2 cache pollution from parallel tests.
    let pid = std::process::id();
    let kernel_a = format!("// diff_test_a_{pid}\nvoid main() {{ float a = 1.0; }}");
    let kernel_b = format!("// diff_test_b_{pid}\nvoid main() {{ float b = 2.0; }}");

    compile_or_cache(&mut cache, &kernel_a, "main", &spirv, &ds_layout, 4)
        .expect("kernel_a compile");
    compile_or_cache(&mut cache, &kernel_b, "main", &spirv, &ds_layout, 4)
        .expect("kernel_b compile");

    // Both should be in L1 after compilation.
    assert_eq!(cache.l1_len(), 2);
    // Neither should have been an L1 hit (both are first-time lookups).
    assert_eq!(cache.stats().l1_hits, 0);
}

// ============================================================================
// H. Command Batch Integration Tests
// ============================================================================

#[test]
fn test_command_batch_multi_kernel_pipeline() {
    use nn_vulkan::command_batch::{BarrierStrategy, CommandBatch};

    // Simulate a multi-kernel pipeline: relu -> matmul -> softmax
    let ds_layout = DescriptorSetLayout::new(vec![
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 1,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
    ])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 4).expect("pipeline layout");
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");

    let buf_a = VulkanBuffer::new(4096, BufferUsage::StorageReadWrite).expect("buf_a");
    let buf_b = VulkanBuffer::new(4096, BufferUsage::StorageReadWrite).expect("buf_b");

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);

    // Step 1: relu (buf_a -> buf_b)
    batch
        .record(
            &pipeline,
            &[&buf_a, &buf_b],
            &1024_u32.to_le_bytes(),
            [4, 1, 1],
        )
        .expect("relu dispatch");

    // Step 2: another op (buf_b -> buf_a)
    batch
        .record(
            &pipeline,
            &[&buf_b, &buf_a],
            &1024_u32.to_le_bytes(),
            [4, 1, 1],
        )
        .expect("second dispatch");

    assert_eq!(batch.dispatch_count(), 2);
    assert_eq!(batch.barrier_count(), 2); // Auto mode inserts barriers.

    batch.submit_and_wait().expect("batch submit");
}

#[test]
fn test_command_batch_manual_barrier_chain() {
    use nn_vulkan::command_batch::{BarrierStrategy, CommandBatch};

    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 4).expect("pl");
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");
    let buf = VulkanBuffer::new(256, BufferUsage::StorageReadWrite).expect("buf");

    let mut batch = CommandBatch::new(BarrierStrategy::Manual);

    // Two independent dispatches (no barrier needed).
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("d1");
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("d2");

    // Barrier before dependent dispatch.
    batch.barrier();
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("d3");

    assert_eq!(batch.dispatch_count(), 3);
    assert_eq!(batch.barrier_count(), 1);
    assert!(!batch.has_barrier_after(0));
    assert!(batch.has_barrier_after(1)); // barrier() marks the last dispatch.
    assert!(!batch.has_barrier_after(2));

    batch.submit_and_wait().expect("submit");
}

#[test]
fn test_command_batch_async_submit() {
    use nn_vulkan::command_batch::{BarrierStrategy, CommandBatch};

    let ds_layout = DescriptorSetLayout::new(vec![DescriptorBinding {
        binding: 0,
        descriptor_type: DescriptorType::StorageBuffer,
        count: 1,
    }])
    .expect("ds layout");
    let pl = PipelineLayout::new(&ds_layout, 4).expect("pl");
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let pipeline = ComputePipeline::new(&spirv, "main", &pl).expect("pipeline");
    let buf = VulkanBuffer::new(256, BufferUsage::StorageReadWrite).expect("buf");

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);
    batch
        .record(&pipeline, &[&buf], &[0u8; 4], [1, 1, 1])
        .expect("record");

    let pending = batch.submit_async().expect("async submit");
    assert_eq!(pending.dispatch_count(), 1);
    assert!(pending.is_completed());
    pending.wait().expect("wait");
}

// ============================================================================
// I. Buffer Pool Integration Tests
// ============================================================================

#[test]
fn test_buffer_pool_acquire_release_cycle() {
    use nn_vulkan::buffer_pool::BufferPool;

    let mut pool = BufferPool::new();

    // Acquire.
    let buf = pool
        .acquire(1024, BufferUsage::StorageReadWrite)
        .expect("acquire");
    assert!(buf.size_bytes() >= 1024);
    assert_eq!(pool.stats().acquisitions, 1);

    // Release.
    pool.release(buf);

    // Re-acquire: should get a hit (if in the size class).
    let _buf2 = pool
        .acquire(1024, BufferUsage::StorageReadWrite)
        .expect("re-acquire");
    assert_eq!(pool.stats().acquisitions, 2);
}

#[test]
fn test_buffer_pool_clear_resets_state() {
    use nn_vulkan::buffer_pool::BufferPool;

    let mut pool = BufferPool::new();
    let _buf = pool
        .acquire(4096, BufferUsage::StorageReadWrite)
        .expect("acquire");
    pool.clear();
    assert_eq!(pool.stats().retained_bytes, 0);
    assert_eq!(pool.stats().buffer_count, 0);
}

// ============================================================================
// J. Workgroup Utility Integration Tests
// ============================================================================

#[test]
fn test_workgroup_utils_relu_dispatch() {
    use nn_vulkan::workgroup::{
        optimal_elementwise_workgroup, push_constants_1d, validate_dispatch, workgroup_count_1d,
    };

    let total_elements: u32 = 10_000;
    let max_invocations: u32 = 1024;

    // Choose workgroup size.
    let wg_size = optimal_elementwise_workgroup(total_elements, max_invocations);
    assert_eq!(wg_size, DEFAULT_WORKGROUP_SIZE);

    // Compute dispatch grid.
    let group_count_x = workgroup_count_1d(total_elements, wg_size);
    assert_eq!(group_count_x, 40); // ceil(10000/256) = 40

    // Validate dispatch.
    validate_dispatch(
        [group_count_x, 1, 1],
        [wg_size, 1, 1],
        65535,
        max_invocations,
    )
    .expect("valid dispatch");

    // Build push constants.
    let push = push_constants_1d(total_elements);
    assert_eq!(u32::from_le_bytes(push), total_elements);
}

#[test]
fn test_workgroup_utils_matmul_dispatch() {
    use nn_vulkan::workgroup::{push_constants_matmul, validate_dispatch, workgroup_count_2d};

    let m: u32 = 512;
    let n: u32 = 256;
    let k: u32 = 128;
    let tile: u32 = 16;

    let groups = workgroup_count_2d(n, m, tile);
    assert_eq!(groups, [16, 32, 1]); // ceil(256/16), ceil(512/16)

    validate_dispatch(groups, [tile, tile, 1], 65535, 1024).expect("valid matmul dispatch");

    let push = push_constants_matmul(m, n, k);
    assert_eq!(push.len(), 12);
}

#[test]
fn test_workgroup_utils_reduction_dispatch() {
    use nn_vulkan::workgroup::{
        push_constants_reduction, validate_dispatch, workgroup_count_row_reduce,
    };

    let num_rows: u32 = 64;
    let row_size: u32 = 512;

    let groups = workgroup_count_row_reduce(num_rows);
    assert_eq!(groups, [64, 1, 1]);

    validate_dispatch(groups, [DEFAULT_WORKGROUP_SIZE, 1, 1], 65535, 1024)
        .expect("valid reduction dispatch");

    let push = push_constants_reduction(row_size, num_rows);
    assert_eq!(push.len(), 8);
}

// ============================================================================
// K. End-to-end: full pipeline with cache + batch + workgroup utils
// ============================================================================

#[test]
fn test_full_pipeline_cached_batch_dispatch() {
    use nn_vulkan::command_batch::{BarrierStrategy, CommandBatch};
    use nn_vulkan::pipeline_cache::{compile_or_cache, PipelineCache};
    use nn_vulkan::workgroup::{push_constants_1d, workgroup_count_1d};

    // Step 1: compile kernels with caching.
    let mut cache = PipelineCache::new();
    let spirv = vec![SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0];
    let ds_layout = DescriptorSetLayout::new(vec![
        DescriptorBinding {
            binding: 0,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
        DescriptorBinding {
            binding: 1,
            descriptor_type: DescriptorType::StorageBuffer,
            count: 1,
        },
    ])
    .expect("ds layout");

    let relu_glsl = activations::relu_glsl().expect("relu");
    let silu_glsl = activations::silu_glsl().expect("silu");

    let relu_pipeline = compile_or_cache(&mut cache, &relu_glsl, "main", &spirv, &ds_layout, 4)
        .expect("relu compile");
    let silu_pipeline = compile_or_cache(&mut cache, &silu_glsl, "main", &spirv, &ds_layout, 4)
        .expect("silu compile");

    // Step 2: allocate buffers.
    let n_elements: u32 = 4096;
    let byte_size = (n_elements as usize) * 4;
    let buf_input = VulkanBuffer::new(byte_size, BufferUsage::StorageRead).expect("input");
    let buf_intermediate =
        VulkanBuffer::new(byte_size, BufferUsage::StorageReadWrite).expect("intermediate");
    let buf_output = VulkanBuffer::new(byte_size, BufferUsage::StorageReadWrite).expect("output");

    // Step 3: record dispatches in a batch.
    let group_count = workgroup_count_1d(n_elements, DEFAULT_WORKGROUP_SIZE);
    let push = push_constants_1d(n_elements);

    let mut batch = CommandBatch::new(BarrierStrategy::Auto);

    // relu: input -> intermediate
    batch
        .record(
            &relu_pipeline,
            &[&buf_input, &buf_intermediate],
            &push,
            [group_count, 1, 1],
        )
        .expect("relu dispatch");

    // silu: intermediate -> output
    batch
        .record(
            &silu_pipeline,
            &[&buf_intermediate, &buf_output],
            &push,
            [group_count, 1, 1],
        )
        .expect("silu dispatch");

    assert_eq!(batch.dispatch_count(), 2);
    assert_eq!(batch.barrier_count(), 2);

    // Step 4: submit.
    batch.submit_and_wait().expect("batch submit");

    // Verify cache was used for the second identical compile.
    let _relu_again = compile_or_cache(&mut cache, &relu_glsl, "main", &spirv, &ds_layout, 4)
        .expect("cached relu");
    assert_eq!(cache.stats().l1_hits, 1);
}
