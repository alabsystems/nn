// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Extended tests for SPIR-V kernel generation, normalization, and dispatch infrastructure.
//!
//! Covers:
//! 1. SPIR-V activation kernel generation: ReLU, GELU, SiLU output validity
//! 2. SPIR-V normalization: LayerNorm, RMSNorm code generation
//! 3. Elementwise binary ops: add, mul, sub, div SPIR-V generation
//! 4. Reduction ops: sum, mean, max SPIR-V generation
//! 5. Matrix transpose SPIR-V generation
//! 6. Embedding lookup SPIR-V generation
//! 7. Workgroup size configuration: valid sizes for different operations
//! 8. Buffer binding validation: correct descriptor set bindings
//! 9. Push constant layout: size and alignment requirements
//! 10. Shader module compatibility: SPIR-V version and capability requirements

use crate::compute_pipeline::{
    compute_grid_dims, spirv_words_to_bytes, BufferBinding, CompiledShader, DispatchConfig,
    PushConstants, VulkanComputeConfig, VulkanPipelineError,
};
use crate::spirv_activations::{
    gelu_reference, generate_gelu_spirv, generate_silu_spirv, generate_snake_spirv, silu_reference,
    ACTIVATION_WORKGROUP_SIZE,
};
use crate::spirv_binary::{
    emit_add_spirv, emit_mul_spirv, emit_relu_spirv, emit_scalar_mul_spirv, find_entry_point_name,
    find_workgroup_size, BINARY_WORKGROUP_SIZE,
};
use crate::spirv_embedding::{generate_embedding_spirv, EMBEDDING_WORKGROUP_SIZE};
use crate::spirv_emit::{SPIRV_MAGIC, SPIRV_VERSION_1_5};
use crate::spirv_layernorm::{
    generate_layernorm_spirv, generate_rmsnorm_spirv, LAYERNORM_WORKGROUP_SIZE,
};
use crate::spirv_reduction::{
    generate_max_spirv, generate_mean_spirv, generate_sum_spirv, REDUCTION_WORKGROUP_SIZE,
};
use crate::spirv_transpose::{
    generate_batch_transpose_spirv, generate_transpose_spirv, transpose_reference,
    TRANSPOSE_WORKGROUP_SIZE,
};
use crate::workgroup::{optimal_elementwise_workgroup, validate_dispatch, workgroup_count_1d};

// ---- SPIR-V structural constants ----

const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

// ---- Helpers ----

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(
        bytes.len() % 4,
        0,
        "SPIR-V byte length must be multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn assert_valid_spirv_header(words: &[u32], label: &str) {
    assert!(
        words.len() >= 5,
        "{label}: SPIR-V module too short ({} words)",
        words.len()
    );
    assert_eq!(words[0], SPIRV_MAGIC, "{label}: wrong magic number");
    assert_eq!(words[1], SPIRV_VERSION_1_0, "{label}: wrong SPIR-V version");
    assert_eq!(words[2], GENERATOR_MAGIC, "{label}: wrong generator magic");
    assert!(words[3] > 0, "{label}: bound must be > 0");
    assert_eq!(words[4], 0, "{label}: schema must be 0");
}

fn assert_valid_spirv_bytes(bytes: &[u8], label: &str) {
    let words = bytes_to_words(bytes);
    assert_valid_spirv_header(&words, label);
}

fn assert_entry_point_main(words: &[u32], label: &str) {
    let name =
        find_entry_point_name(words).unwrap_or_else(|| panic!("{label}: no entry point found"));
    assert_eq!(
        name, "main",
        "{label}: entry point should be 'main', got '{name}'"
    );
}

fn assert_workgroup_size_eq(words: &[u32], expected: [u32; 3], label: &str) {
    let wg =
        find_workgroup_size(words).unwrap_or_else(|| panic!("{label}: no workgroup size found"));
    assert_eq!(wg, expected, "{label}: wrong workgroup size");
}

fn minimal_spirv_bytes() -> Vec<u8> {
    spirv_words_to_bytes(&[SPIRV_MAGIC, SPIRV_VERSION_1_5, 0, 0, 0])
}

fn shader_with(num_bindings: u32, push_constant_size: u32, wg: [u32; 3]) -> CompiledShader {
    CompiledShader::new(
        minimal_spirv_bytes(),
        "main",
        num_bindings,
        push_constant_size,
        wg,
    )
    .expect("shader construction must succeed")
}

// ====================================================================
// 1. SPIR-V activation kernel generation: ReLU, GELU, SiLU
// ====================================================================

#[test]
fn test_relu_spirv_valid_header_and_entry() {
    let words = emit_relu_spirv().expect("ReLU SPIR-V generation must succeed");
    assert_valid_spirv_header(&words, "relu");
    assert_entry_point_main(&words, "relu");
}

#[test]
fn test_relu_spirv_workgroup_size() {
    let words = emit_relu_spirv().unwrap();
    assert_workgroup_size_eq(&words, [BINARY_WORKGROUP_SIZE, 1, 1], "relu_wg");
}

#[test]
fn test_gelu_spirv_valid_header_various_workgroups() {
    for wg in [64, 128, 256, 512] {
        let bytes = generate_gelu_spirv(wg);
        let words = bytes_to_words(&bytes);
        assert_valid_spirv_header(&words, &format!("gelu_wg{wg}"));
        assert_entry_point_main(&words, &format!("gelu_wg{wg}"));
        assert_workgroup_size_eq(&words, [wg, 1, 1], &format!("gelu_wg{wg}"));
    }
}

#[test]
fn test_silu_spirv_valid_header_various_workgroups() {
    for wg in [64, 128, 256, 512] {
        let bytes = generate_silu_spirv(wg);
        let words = bytes_to_words(&bytes);
        assert_valid_spirv_header(&words, &format!("silu_wg{wg}"));
        assert_entry_point_main(&words, &format!("silu_wg{wg}"));
        assert_workgroup_size_eq(&words, [wg, 1, 1], &format!("silu_wg{wg}"));
    }
}

#[test]
fn test_gelu_reference_known_values() {
    // GELU(0) = 0
    assert!(gelu_reference(0.0).abs() < 1e-6, "GELU(0) should be 0");
    // GELU(large positive) ~ x
    let g10 = gelu_reference(10.0);
    assert!((g10 - 10.0).abs() < 0.01, "GELU(10) ~ 10, got {g10}");
    // GELU(large negative) ~ 0
    let gn10 = gelu_reference(-10.0);
    assert!(gn10.abs() < 0.01, "GELU(-10) ~ 0, got {gn10}");
}

#[test]
fn test_silu_reference_known_values() {
    // SiLU(0) = 0 * sigmoid(0) = 0
    let s0 = silu_reference(0.0);
    assert!(s0.abs() < 1e-6, "SiLU(0) should be 0, got {s0}");
    // SiLU(x) > 0 for x > 0
    assert!(silu_reference(1.0) > 0.0, "SiLU(1) should be positive");
    assert!(silu_reference(5.0) > 0.0, "SiLU(5) should be positive");
    // SiLU(large negative) ~ 0
    assert!(silu_reference(-20.0).abs() < 1e-6, "SiLU(-20) ~ 0");
}

#[test]
fn test_relu_gelu_silu_all_produce_nonempty_spirv() {
    let relu = emit_relu_spirv().unwrap();
    assert!(
        relu.len() > 20,
        "ReLU SPIR-V should be substantial, got {} words",
        relu.len()
    );

    let gelu = generate_gelu_spirv(256);
    assert!(
        gelu.len() > 80,
        "GELU SPIR-V should be substantial, got {} bytes",
        gelu.len()
    );

    let silu = generate_silu_spirv(256);
    assert!(
        silu.len() > 80,
        "SiLU SPIR-V should be substantial, got {} bytes",
        silu.len()
    );
}

#[test]
fn test_snake_spirv_valid_header_and_entry() {
    let bytes = generate_snake_spirv(ACTIVATION_WORKGROUP_SIZE);
    let words = bytes_to_words(&bytes);
    assert_valid_spirv_header(&words, "snake");
    assert_entry_point_main(&words, "snake");
    assert_workgroup_size_eq(&words, [ACTIVATION_WORKGROUP_SIZE, 1, 1], "snake");
}

// ====================================================================
// 2. SPIR-V normalization: LayerNorm, RMSNorm
// ====================================================================

#[test]
fn test_layernorm_spirv_valid_header_various_shapes() {
    for shape in [64, 128, 256, 512, 768, 1024, 4096] {
        let bytes = generate_layernorm_spirv(shape, 1e-5);
        assert_valid_spirv_bytes(&bytes, &format!("layernorm_shape{shape}"));
    }
}

#[test]
fn test_layernorm_spirv_entry_point_main() {
    let bytes = generate_layernorm_spirv(768, 1e-5);
    let words = bytes_to_words(&bytes);
    assert_entry_point_main(&words, "layernorm_768");
}

#[test]
fn test_layernorm_spirv_workgroup_size() {
    let bytes = generate_layernorm_spirv(512, 1e-5);
    let words = bytes_to_words(&bytes);
    assert_workgroup_size_eq(&words, [LAYERNORM_WORKGROUP_SIZE, 1, 1], "layernorm_wg");
}

#[test]
fn test_rmsnorm_spirv_valid_header_various_shapes() {
    for shape in [64, 128, 256, 512, 768, 1024] {
        let bytes = generate_rmsnorm_spirv(shape, 1e-5);
        assert_valid_spirv_bytes(&bytes, &format!("rmsnorm_shape{shape}"));
    }
}

#[test]
fn test_rmsnorm_spirv_entry_point_and_workgroup() {
    let bytes = generate_rmsnorm_spirv(768, 1e-6);
    let words = bytes_to_words(&bytes);
    assert_entry_point_main(&words, "rmsnorm_768");
    assert_workgroup_size_eq(&words, [LAYERNORM_WORKGROUP_SIZE, 1, 1], "rmsnorm_wg");
}

#[test]
fn test_layernorm_vs_rmsnorm_different_modules() {
    let ln_bytes = generate_layernorm_spirv(256, 1e-5);
    let rms_bytes = generate_rmsnorm_spirv(256, 1e-5);
    // LayerNorm has mean + variance passes; RMSNorm has only RMS pass.
    // They should produce different SPIR-V binaries.
    assert_ne!(
        ln_bytes, rms_bytes,
        "LayerNorm and RMSNorm should produce different SPIR-V"
    );
}

#[test]
fn test_layernorm_spirv_different_eps_produces_different_binary() {
    let bytes_a = generate_layernorm_spirv(256, 1e-5);
    let bytes_b = generate_layernorm_spirv(256, 1e-8);
    // Different eps constants should produce different binaries.
    assert_ne!(
        bytes_a, bytes_b,
        "Different eps should yield different SPIR-V"
    );
}

#[test]
fn test_rmsnorm_spirv_shape_1_minimal() {
    let bytes = generate_rmsnorm_spirv(1, 1e-5);
    assert_valid_spirv_bytes(&bytes, "rmsnorm_shape1");
}

// ====================================================================
// 3. Elementwise binary ops: add, mul SPIR-V generation
// ====================================================================

#[test]
fn test_add_spirv_valid_header_and_entry() {
    let words = emit_add_spirv().expect("add SPIR-V generation must succeed");
    assert_valid_spirv_header(&words, "add");
    assert_entry_point_main(&words, "add");
}

#[test]
fn test_mul_spirv_valid_header_and_entry() {
    let words = emit_mul_spirv().expect("mul SPIR-V generation must succeed");
    assert_valid_spirv_header(&words, "mul");
    assert_entry_point_main(&words, "mul");
}

#[test]
fn test_scalar_mul_spirv_valid_header_and_entry() {
    let words = emit_scalar_mul_spirv().expect("scalar_mul SPIR-V generation must succeed");
    assert_valid_spirv_header(&words, "scalar_mul");
    assert_entry_point_main(&words, "scalar_mul");
}

#[test]
fn test_add_mul_produce_distinct_spirv() {
    let add = emit_add_spirv().unwrap();
    let mul = emit_mul_spirv().unwrap();
    assert_ne!(add, mul, "add and mul should produce different SPIR-V");
}

#[test]
fn test_binary_ops_workgroup_size() {
    for (label, words) in [
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
    ] {
        assert_workgroup_size_eq(&words, [BINARY_WORKGROUP_SIZE, 1, 1], label);
    }
}

#[test]
fn test_binary_ops_all_produce_nonempty_modules() {
    let ops: Vec<(&str, Vec<u32>)> = vec![
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
    ];
    for (name, words) in &ops {
        assert!(
            words.len() > 20,
            "{name}: SPIR-V module too small ({} words)",
            words.len()
        );
    }
}

// ====================================================================
// 4. Reduction ops: sum, mean, max SPIR-V generation
// ====================================================================

#[test]
fn test_sum_spirv_valid_header_various_sizes() {
    for n in [1, 16, 64, 256, 1024, 4096, 65536] {
        let bytes = generate_sum_spirv(n);
        assert_valid_spirv_bytes(&bytes, &format!("sum_n{n}"));
    }
}

#[test]
fn test_mean_spirv_valid_header_various_sizes() {
    for n in [1, 16, 256, 4096] {
        let bytes = generate_mean_spirv(n);
        assert_valid_spirv_bytes(&bytes, &format!("mean_n{n}"));
    }
}

#[test]
fn test_max_spirv_valid_header_various_sizes() {
    for n in [1, 64, 256, 1024] {
        let bytes = generate_max_spirv(n);
        assert_valid_spirv_bytes(&bytes, &format!("max_n{n}"));
    }
}

#[test]
fn test_reduction_ops_entry_points_all_main() {
    for (label, bytes) in [
        ("sum", generate_sum_spirv(256)),
        ("mean", generate_mean_spirv(256)),
        ("max", generate_max_spirv(256)),
    ] {
        let words = bytes_to_words(&bytes);
        assert_entry_point_main(&words, label);
    }
}

#[test]
fn test_reduction_ops_workgroup_size() {
    for (label, bytes) in [
        ("sum", generate_sum_spirv(1024)),
        ("mean", generate_mean_spirv(1024)),
        ("max", generate_max_spirv(1024)),
    ] {
        let words = bytes_to_words(&bytes);
        assert_workgroup_size_eq(&words, [REDUCTION_WORKGROUP_SIZE, 1, 1], label);
    }
}

#[test]
fn test_sum_mean_max_produce_distinct_spirv() {
    let sum = generate_sum_spirv(256);
    let mean = generate_mean_spirv(256);
    let max = generate_max_spirv(256);
    assert_ne!(sum, mean, "sum and mean should differ");
    assert_ne!(sum, max, "sum and max should differ");
    assert_ne!(mean, max, "mean and max should differ");
}

#[test]
fn test_sum_spirv_size_changes_with_n() {
    // Different n values should produce different binaries because n is baked in.
    let s64 = generate_sum_spirv(64);
    let s128 = generate_sum_spirv(128);
    assert_ne!(s64, s128, "sum(64) and sum(128) should differ");
}

// ====================================================================
// 5. Matrix transpose SPIR-V generation
// ====================================================================

#[test]
fn test_transpose_spirv_valid_header_various_dims() {
    for (r, c) in [(4, 4), (16, 32), (128, 256), (1, 1), (1024, 512)] {
        let words = generate_transpose_spirv(r, c);
        assert_valid_spirv_header(&words, &format!("transpose_{r}x{c}"));
    }
}

#[test]
fn test_transpose_spirv_entry_point() {
    let words = generate_transpose_spirv(64, 128);
    assert_entry_point_main(&words, "transpose_64x128");
}

#[test]
fn test_transpose_spirv_workgroup_size_2d() {
    let words = generate_transpose_spirv(128, 256);
    let wg = find_workgroup_size(&words).expect("transpose must have workgroup size");
    // Transpose uses 2D workgroup: TRANSPOSE_WORKGROUP_SIZE x TRANSPOSE_WORKGROUP_SIZE
    assert_eq!(
        wg,
        [TRANSPOSE_WORKGROUP_SIZE, TRANSPOSE_WORKGROUP_SIZE, 1],
        "transpose should use 2D workgroup"
    );
}

#[test]
fn test_batch_transpose_spirv_valid_header() {
    for (b, r, c) in [(2, 8, 16), (4, 32, 64), (1, 128, 128)] {
        let words = generate_batch_transpose_spirv(b, r, c);
        assert_valid_spirv_header(&words, &format!("btranspose_{b}x{r}x{c}"));
    }
}

#[test]
fn test_batch_transpose_spirv_entry_point() {
    let words = generate_batch_transpose_spirv(4, 32, 64);
    assert_entry_point_main(&words, "btranspose");
}

#[test]
fn test_transpose_reference_identity_square() {
    // Transposing a 3x3 identity matrix should return the same matrix.
    let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let result = transpose_reference(&data, 3, 3);
    assert_eq!(result, data);
}

#[test]
fn test_transpose_reference_2x3() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
    let result = transpose_reference(&data, 2, 3);
    // Expected: 3x2 = [1, 4, 2, 5, 3, 6]
    assert_eq!(result, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_reference_double_transpose_is_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect(); // 4x6
    let transposed = transpose_reference(&data, 4, 6); // 6x4
    let back = transpose_reference(&transposed, 6, 4); // 4x6
    for (i, (&a, &b)) in data.iter().zip(back.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "double transpose: pos {i}: {a} != {b}"
        );
    }
}

#[test]
fn test_transpose_reference_single_row() {
    let data = vec![1.0, 2.0, 3.0, 4.0]; // 1x4
    let result = transpose_reference(&data, 1, 4); // 4x1
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_reference_single_column() {
    let data = vec![1.0, 2.0, 3.0]; // 3x1
    let result = transpose_reference(&data, 3, 1); // 1x3
    assert_eq!(result, vec![1.0, 2.0, 3.0]);
}

// ====================================================================
// 6. Embedding lookup SPIR-V generation
// ====================================================================

#[test]
fn test_embedding_spirv_valid_header_various_configs() {
    for (vocab, dim) in [(100, 32), (1000, 128), (32000, 768), (50257, 1024)] {
        let bytes = generate_embedding_spirv(vocab, dim);
        assert_valid_spirv_bytes(&bytes, &format!("embedding_v{vocab}_d{dim}"));
    }
}

#[test]
fn test_embedding_spirv_entry_point() {
    let bytes = generate_embedding_spirv(32000, 768);
    let words = bytes_to_words(&bytes);
    assert_entry_point_main(&words, "embedding");
}

#[test]
fn test_embedding_spirv_workgroup_size() {
    let bytes = generate_embedding_spirv(50257, 1024);
    let words = bytes_to_words(&bytes);
    assert_workgroup_size_eq(&words, [EMBEDDING_WORKGROUP_SIZE, 1, 1], "embedding_wg");
}

#[test]
fn test_embedding_spirv_small_vocab() {
    let bytes = generate_embedding_spirv(1, 1);
    assert_valid_spirv_bytes(&bytes, "embedding_v1_d1");
}

#[test]
fn test_embedding_spirv_different_configs_produce_different_binaries() {
    let a = generate_embedding_spirv(1000, 128);
    let b = generate_embedding_spirv(2000, 128);
    let c = generate_embedding_spirv(1000, 256);
    assert_ne!(
        a, b,
        "different vocab sizes should produce different SPIR-V"
    );
    assert_ne!(
        a, c,
        "different embedding dims should produce different SPIR-V"
    );
}

// ====================================================================
// 7. Workgroup size configuration: valid sizes for different operations
// ====================================================================

#[test]
fn test_all_workgroup_sizes_are_powers_of_two() {
    let sizes = [
        ("BINARY", BINARY_WORKGROUP_SIZE),
        ("ACTIVATION", ACTIVATION_WORKGROUP_SIZE),
        ("REDUCTION", REDUCTION_WORKGROUP_SIZE),
        ("LAYERNORM", LAYERNORM_WORKGROUP_SIZE),
        ("EMBEDDING", EMBEDDING_WORKGROUP_SIZE),
    ];
    for (name, size) in sizes {
        assert!(
            size.is_power_of_two(),
            "{name}_WORKGROUP_SIZE={size} is not a power of two"
        );
    }
}

#[test]
fn test_all_workgroup_sizes_within_vulkan_guaranteed_limit() {
    // Vulkan spec guarantees maxComputeWorkGroupInvocations >= 128.
    // Most sizes are 256. TRANSPOSE is 16 (2D: 16*16=256).
    let sizes = [
        ("BINARY", BINARY_WORKGROUP_SIZE),
        ("ACTIVATION", ACTIVATION_WORKGROUP_SIZE),
        ("REDUCTION", REDUCTION_WORKGROUP_SIZE),
        ("LAYERNORM", LAYERNORM_WORKGROUP_SIZE),
        ("EMBEDDING", EMBEDDING_WORKGROUP_SIZE),
        ("TRANSPOSE", TRANSPOSE_WORKGROUP_SIZE),
    ];
    for (name, size) in sizes {
        assert!(
            (1..=1024).contains(&size),
            "{name}_WORKGROUP_SIZE={size} out of reasonable range [1, 1024]"
        );
    }
}

#[test]
fn test_transpose_workgroup_2d_product_is_256() {
    // TRANSPOSE uses 2D: 16x16 = 256 total invocations.
    let product = TRANSPOSE_WORKGROUP_SIZE * TRANSPOSE_WORKGROUP_SIZE;
    assert_eq!(product, 256, "transpose 2D workgroup product should be 256");
}

#[test]
fn test_optimal_workgroup_respects_vulkan_limits() {
    // optimal_elementwise_workgroup should never exceed max_invocations.
    for max_inv in [128, 256, 512, 1024] {
        for total in [1, 100, 1000, 100_000] {
            let wg = optimal_elementwise_workgroup(total, max_inv);
            assert!(
                wg <= max_inv,
                "optimal({total}, {max_inv})={wg} exceeds max"
            );
            assert!(wg.is_power_of_two(), "optimal must return power of two");
        }
    }
}

#[test]
fn test_workgroup_count_1d_covers_all_elements_for_activation_size() {
    let wg = ACTIVATION_WORKGROUP_SIZE;
    for total in [1, 255, 256, 257, 1000, 10000] {
        let count = workgroup_count_1d(total, wg);
        assert!(count * wg >= total, "coverage: {count}*{wg} < {total}");
    }
}

// ====================================================================
// 8. Buffer binding validation: correct descriptor set bindings
// ====================================================================

#[test]
fn test_binding_validation_exact_count_accepted() {
    let shader = shader_with(3, 0, [64, 1, 1]);
    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: 1024,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: 1024,
                read_only: true,
            },
            BufferBinding {
                binding: 2,
                offset: 0,
                size: 1024,
                read_only: false,
            },
        ],
        push_constants: None,
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_binding_validation_too_few_bindings_rejected() {
    let shader = shader_with(3, 0, [64, 1, 1]);
    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings: vec![BufferBinding {
            binding: 0,
            offset: 0,
            size: 1024,
            read_only: true,
        }],
        push_constants: None,
    };
    let err = shader.validate_dispatch(&config).unwrap_err();
    assert!(
        matches!(err, VulkanPipelineError::BindingCountMismatch { .. }),
        "expected BindingCountMismatch, got: {err:?}"
    );
}

#[test]
fn test_binding_validation_out_of_range_binding_index() {
    let shader = shader_with(2, 0, [64, 1, 1]);
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
                binding: 10,
                offset: 0,
                size: 512,
                read_only: false,
            },
        ],
        push_constants: None,
    };
    let err = shader.validate_dispatch(&config).unwrap_err();
    assert!(
        matches!(err, VulkanPipelineError::BindingOutOfRange { .. }),
        "expected BindingOutOfRange, got: {err:?}"
    );
}

#[test]
fn test_binding_validation_zero_bindings_accepted() {
    let shader = shader_with(0, 0, [64, 1, 1]);
    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings: vec![],
        push_constants: None,
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_binding_validation_read_only_and_readwrite_mix() {
    let shader = shader_with(4, 0, [128, 1, 1]);
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
                read_only: true,
            },
            BufferBinding {
                binding: 2,
                offset: 0,
                size: 256,
                read_only: true,
            },
            BufferBinding {
                binding: 3,
                offset: 0,
                size: 256,
                read_only: false,
            },
        ],
        push_constants: None,
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_binding_offset_nonzero_accepted() {
    let shader = shader_with(1, 0, [64, 1, 1]);
    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings: vec![BufferBinding {
            binding: 0,
            offset: 4096,
            size: 1024,
            read_only: true,
        }],
        push_constants: None,
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

// ====================================================================
// 9. Push constant layout: size and alignment requirements
// ====================================================================

#[test]
fn test_push_constants_u32_alignment() {
    let mut pc = PushConstants::new();
    pc.push_u32(42);
    assert_eq!(pc.size(), 4, "single u32 should be 4 bytes");
    assert_eq!(pc.as_bytes().len(), 4);
}

#[test]
fn test_push_constants_multiple_u32() {
    let mut pc = PushConstants::new();
    for i in 0..8 {
        pc.push_u32(i);
    }
    assert_eq!(pc.size(), 32, "8 u32 values should be 32 bytes");
    // Verify each value round-trips correctly.
    let bytes = pc.as_bytes();
    for i in 0u32..8 {
        let offset = (i as usize) * 4;
        let val = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        assert_eq!(val, i, "push constant value at index {i}");
    }
}

#[test]
fn test_push_constants_f32_round_trip() {
    let mut pc = PushConstants::new();
    let values = [0.0_f32, 1.0, -1.0, 3.14159, f32::MAX, f32::MIN];
    for &v in &values {
        pc.push_f32(v);
    }
    assert_eq!(pc.size(), values.len() * 4);
    let bytes = pc.as_bytes();
    for (i, &expected) in values.iter().enumerate() {
        let offset = i * 4;
        let val = f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        assert_eq!(val.to_bits(), expected.to_bits(), "f32 at index {i}");
    }
}

#[test]
fn test_push_constants_i32_round_trip() {
    let mut pc = PushConstants::new();
    let values = [0_i32, 1, -1, i32::MAX, i32::MIN, 42, -42];
    for &v in &values {
        pc.push_i32(v);
    }
    assert_eq!(pc.size(), values.len() * 4);
    let bytes = pc.as_bytes();
    for (i, &expected) in values.iter().enumerate() {
        let offset = i * 4;
        let val = i32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        assert_eq!(val, expected, "i32 at index {i}");
    }
}

#[test]
fn test_push_constants_overflow_rejected() {
    let shader = shader_with(0, 4, [64, 1, 1]);
    let mut pc = PushConstants::new();
    pc.push_u32(0);
    pc.push_u32(0); // 8 bytes > declared 4 bytes
    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings: vec![],
        push_constants: Some(pc),
    };
    let err = shader.validate_dispatch(&config).unwrap_err();
    assert!(
        matches!(err, VulkanPipelineError::PushConstantOverflow { .. }),
        "expected PushConstantOverflow, got: {err:?}"
    );
}

#[test]
fn test_push_constants_exact_size_accepted() {
    let shader = shader_with(0, 12, [64, 1, 1]);
    let mut pc = PushConstants::new();
    pc.push_u32(128);
    pc.push_u32(256);
    pc.push_u32(64);
    assert_eq!(pc.size(), 12);
    let config = DispatchConfig {
        grid: [1, 1, 1],
        bindings: vec![],
        push_constants: Some(pc),
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_push_constants_mixed_types() {
    let mut pc = PushConstants::new();
    pc.push_u32(1024); // n_elements
    pc.push_f32(1e-5); // eps
    pc.push_i32(-1); // some flag
    assert_eq!(pc.size(), 12);
    let bytes = pc.as_bytes();
    assert_eq!(
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        1024
    );
    let eps = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert!((eps - 1e-5).abs() < 1e-10);
    assert_eq!(
        i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]),
        -1
    );
}

// ====================================================================
// 10. Shader module compatibility: SPIR-V version and capability requirements
// ====================================================================

#[test]
fn test_all_generated_modules_use_spirv_1_0() {
    let modules: Vec<(&str, Vec<u8>)> = vec![
        ("gelu", generate_gelu_spirv(256)),
        ("silu", generate_silu_spirv(256)),
        ("snake", generate_snake_spirv(256)),
        ("layernorm", generate_layernorm_spirv(256, 1e-5)),
        ("rmsnorm", generate_rmsnorm_spirv(256, 1e-5)),
        ("sum", generate_sum_spirv(256)),
        ("mean", generate_mean_spirv(256)),
        ("max", generate_max_spirv(256)),
        ("embedding", generate_embedding_spirv(1000, 128)),
    ];
    for (name, bytes) in &modules {
        let words = bytes_to_words(bytes);
        assert_eq!(
            words[1], SPIRV_VERSION_1_0,
            "{name}: expected SPIR-V 1.0, got 0x{:08X}",
            words[1]
        );
    }
}

#[test]
fn test_all_generated_modules_use_nn_generator_magic() {
    let modules: Vec<(&str, Vec<u8>)> = vec![
        ("gelu", generate_gelu_spirv(256)),
        ("layernorm", generate_layernorm_spirv(512, 1e-5)),
        ("sum", generate_sum_spirv(128)),
        ("embedding", generate_embedding_spirv(500, 64)),
    ];
    for (name, bytes) in &modules {
        let words = bytes_to_words(bytes);
        assert_eq!(
            words[2], GENERATOR_MAGIC,
            "{name}: expected generator magic 0x4E4E0000, got 0x{:08X}",
            words[2]
        );
    }
}

#[test]
fn test_word_based_modules_use_spirv_1_0() {
    let modules: Vec<(&str, Vec<u32>)> = vec![
        ("add", emit_add_spirv().unwrap()),
        ("mul", emit_mul_spirv().unwrap()),
        ("relu", emit_relu_spirv().unwrap()),
        ("scalar_mul", emit_scalar_mul_spirv().unwrap()),
        ("transpose", generate_transpose_spirv(16, 16)),
    ];
    for (name, words) in &modules {
        assert_eq!(
            words[1], SPIRV_VERSION_1_0,
            "{name}: expected SPIR-V 1.0, got 0x{:08X}",
            words[1]
        );
    }
}

#[test]
fn test_spirv_schema_always_zero() {
    // SPIR-V spec: word[4] (schema) must be 0.
    let modules: Vec<(&str, Vec<u8>)> = vec![
        ("gelu", generate_gelu_spirv(256)),
        ("silu", generate_silu_spirv(256)),
        ("layernorm", generate_layernorm_spirv(768, 1e-5)),
        ("rmsnorm", generate_rmsnorm_spirv(768, 1e-5)),
        ("sum", generate_sum_spirv(512)),
        ("embedding", generate_embedding_spirv(32000, 768)),
    ];
    for (name, bytes) in &modules {
        let words = bytes_to_words(bytes);
        assert_eq!(words[4], 0, "{name}: SPIR-V schema must be 0");
    }
}

#[test]
fn test_spirv_bound_positive_and_reasonable() {
    // word[3] is the ID bound; must be > 0 and reasonable (< 100,000).
    let modules: Vec<(&str, Vec<u8>)> = vec![
        ("gelu", generate_gelu_spirv(256)),
        ("layernorm", generate_layernorm_spirv(256, 1e-5)),
        ("sum", generate_sum_spirv(256)),
    ];
    for (name, bytes) in &modules {
        let words = bytes_to_words(bytes);
        let bound = words[3];
        assert!(bound > 0, "{name}: bound must be > 0");
        assert!(bound < 100_000, "{name}: bound {bound} unreasonably large");
    }
}

// ====================================================================
// Cross-cutting: dispatch validation integration
// ====================================================================

#[test]
fn test_elementwise_dispatch_with_real_workgroup_sizes() {
    // Simulate dispatching an elementwise activation on 50000 elements.
    let total = 50_000u32;
    let wg = ACTIVATION_WORKGROUP_SIZE;
    let count = workgroup_count_1d(total, wg);
    let grid = compute_grid_dims(total, [wg, 1, 1]);
    assert_eq!(grid[0], count);

    // Validate dispatch parameters.
    let result = validate_dispatch([count, 1, 1], [wg, 1, 1], 65535, 1024);
    assert!(
        result.is_ok(),
        "elementwise dispatch should be valid: {result:?}"
    );
}

#[test]
fn test_reduction_dispatch_workflow_end_to_end() {
    let shader = shader_with(2, 8, [REDUCTION_WORKGROUP_SIZE, 1, 1]);
    let mut pc = PushConstants::new();
    pc.push_u32(1024); // row_size
    pc.push_u32(64); // rows
    let config = DispatchConfig {
        grid: [64, 1, 1],
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: 64 * 1024 * 4,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: 64 * 4,
                read_only: false,
            },
        ],
        push_constants: Some(pc),
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_embedding_dispatch_workflow_end_to_end() {
    // Embedding: 3 bindings (token_ids, table, output), 12 bytes push constants.
    let shader = shader_with(3, 12, [EMBEDDING_WORKGROUP_SIZE, 1, 1]);
    let num_tokens = 128u32;
    let vocab = 32000u32;
    let dim = 768u32;
    let total = num_tokens * dim;
    let count = workgroup_count_1d(total, EMBEDDING_WORKGROUP_SIZE);

    let mut pc = PushConstants::new();
    pc.push_u32(num_tokens);
    pc.push_u32(vocab);
    pc.push_u32(dim);

    let config = DispatchConfig {
        grid: [count, 1, 1],
        bindings: vec![
            BufferBinding {
                binding: 0,
                offset: 0,
                size: u64::from(num_tokens) * 4,
                read_only: true,
            },
            BufferBinding {
                binding: 1,
                offset: 0,
                size: u64::from(vocab) * u64::from(dim) * 4,
                read_only: true,
            },
            BufferBinding {
                binding: 2,
                offset: 0,
                size: u64::from(total) * 4,
                read_only: false,
            },
        ],
        push_constants: Some(pc),
    };
    assert!(shader.validate_dispatch(&config).is_ok());
}

#[test]
fn test_spirv_bytes_length_is_multiple_of_4() {
    // All SPIR-V byte streams must be multiples of 4 (u32 word alignment).
    let modules: Vec<(&str, Vec<u8>)> = vec![
        ("gelu", generate_gelu_spirv(256)),
        ("silu", generate_silu_spirv(256)),
        ("snake", generate_snake_spirv(256)),
        ("layernorm", generate_layernorm_spirv(768, 1e-5)),
        ("rmsnorm", generate_rmsnorm_spirv(768, 1e-5)),
        ("sum", generate_sum_spirv(256)),
        ("mean", generate_mean_spirv(256)),
        ("max", generate_max_spirv(256)),
        ("embedding", generate_embedding_spirv(1000, 128)),
    ];
    for (name, bytes) in &modules {
        assert_eq!(
            bytes.len() % 4,
            0,
            "{name}: SPIR-V byte length {} not multiple of 4",
            bytes.len()
        );
    }
}

#[test]
fn test_spirv_minimum_module_size() {
    // A valid SPIR-V module must have at least 5 header words = 20 bytes.
    let modules: Vec<(&str, Vec<u8>)> = vec![
        ("gelu", generate_gelu_spirv(256)),
        ("layernorm", generate_layernorm_spirv(128, 1e-5)),
        ("sum", generate_sum_spirv(64)),
        ("embedding", generate_embedding_spirv(100, 32)),
    ];
    for (name, bytes) in &modules {
        assert!(
            bytes.len() >= 20,
            "{name}: SPIR-V module too small ({} bytes), minimum is 20",
            bytes.len()
        );
    }
}

#[test]
fn test_compiled_shader_rejects_invalid_magic() {
    let mut bad_spirv = minimal_spirv_bytes();
    // Corrupt magic number.
    bad_spirv[0] = 0xFF;
    let result = CompiledShader::new(bad_spirv, "main", 0, 0, [64, 1, 1]);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, VulkanPipelineError::SpirvValidation { .. }),
        "expected SpirvValidation error, got: {err:?}"
    );
}

#[test]
fn test_compiled_shader_rejects_too_short_binary() {
    let too_short = vec![0x03, 0x02, 0x23, 0x07]; // Only 4 bytes, need 20.
    let result = CompiledShader::new(too_short, "main", 0, 0, [64, 1, 1]);
    assert!(result.is_err());
}

#[test]
fn test_vulkan_compute_config_default_values() {
    let config = VulkanComputeConfig::default();
    assert_eq!(config.workgroup_size_x, 256);
    assert_eq!(config.workgroup_size_y, 1);
    assert_eq!(config.workgroup_size_z, 1);
    assert_eq!(config.total_workgroup_invocations(), 256);
}

#[test]
fn test_zero_grid_dimension_rejected() {
    let shader = shader_with(0, 0, [64, 1, 1]);
    let config = DispatchConfig {
        grid: [0, 1, 1],
        bindings: vec![],
        push_constants: None,
    };
    let err = shader.validate_dispatch(&config).unwrap_err();
    assert!(
        matches!(err, VulkanPipelineError::ZeroGridDimension { .. }),
        "expected ZeroGridDimension, got: {err:?}"
    );
}
