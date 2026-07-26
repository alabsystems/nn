// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Part of #4186.
//!
//! Extended tests for `TensorError`, `DType`, `Device`, `BackendDomain`,
//! `BackendErrorKind`, the `Result` type alias, and error conversion traits.
//! Complements `error_tests.rs` and `error_dtype_tests.rs` with coverage for
//! `?`-propagation, pattern-matching recovery, edge cases, and property-based
//! invariant checks.

use std::collections::HashSet;
use std::error::Error;

use crate::error::{check_dim, BackendDomain, BackendErrorKind, TensorError};
use crate::{DType, Device};

// ===========================================================================
// 1. Result type alias — `?` propagation
// ===========================================================================

/// Helper that exercises `?` propagation through the `crate::Result` alias.
fn propagate_shape_mismatch() -> crate::Result<()> {
    let _ok: () = Ok::<(), TensorError>(())?;
    Err(TensorError::shape_mismatch(vec![2, 3], vec![4, 5]))
}

#[test]
fn test_result_alias_question_mark_propagation_returns_err() {
    let result = propagate_shape_mismatch();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Shape mismatch"));
}

/// Helper that returns Ok through the `crate::Result` alias.
fn propagate_ok() -> crate::Result<u32> {
    let val: u32 = Ok::<u32, TensorError>(42)?;
    Ok(val)
}

#[test]
fn test_result_alias_question_mark_propagation_returns_ok() {
    let result = propagate_ok();
    assert_eq!(result.unwrap(), 42);
}

/// Helper that exercises `?` propagation through an io::Error → TensorError
/// conversion via the `From` impl.
fn propagate_io_error() -> crate::Result<()> {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
    Err(io_err)?
}

#[test]
fn test_result_alias_io_error_propagation_via_from() {
    let result = propagate_io_error();
    let err = result.unwrap_err();
    match err {
        TensorError::IoError(ref inner) => {
            assert_eq!(inner.kind(), std::io::ErrorKind::NotFound);
        }
        other => panic!("expected IoError, got: {other:?}"),
    }
}

// ===========================================================================
// 2. TensorError — pattern matching for programmatic recovery
// ===========================================================================

#[test]
fn test_tensor_error_match_rank_mismatch_fields() {
    let err = TensorError::RankMismatch {
        expected: 4,
        actual: 2,
    };
    match err {
        TensorError::RankMismatch { expected, actual } => {
            assert_eq!(expected, 4);
            assert_eq!(actual, 2);
        }
        other => panic!("expected RankMismatch, got: {other:?}"),
    }
}

#[test]
fn test_tensor_error_match_data_length_mismatch_fields() {
    let err = TensorError::DataLengthMismatch {
        expected: 100,
        actual: 50,
    };
    match err {
        TensorError::DataLengthMismatch { expected, actual } => {
            assert_eq!(expected, 100);
            assert_eq!(actual, 50);
        }
        other => panic!("expected DataLengthMismatch, got: {other:?}"),
    }
}

#[test]
fn test_tensor_error_match_out_of_memory_fields() {
    let err = TensorError::OutOfMemory {
        requested: 1_000_000,
        available: 500,
    };
    match err {
        TensorError::OutOfMemory {
            requested,
            available,
        } => {
            assert_eq!(requested, 1_000_000);
            assert_eq!(available, 500);
        }
        other => panic!("expected OutOfMemory, got: {other:?}"),
    }
}

#[test]
fn test_tensor_error_match_conv_parameter_invalid_fields() {
    let err = TensorError::ConvParameterInvalid {
        param: "kernel_size",
        value: 0,
        reason: "must be >= 1",
    };
    match err {
        TensorError::ConvParameterInvalid {
            param,
            value,
            reason,
        } => {
            assert_eq!(param, "kernel_size");
            assert_eq!(value, 0);
            assert_eq!(reason, "must be >= 1");
        }
        other => panic!("expected ConvParameterInvalid, got: {other:?}"),
    }
}

#[test]
fn test_tensor_error_match_embedding_index_fields() {
    let err = TensorError::EmbeddingIndexOutOfRange {
        index: 99999,
        vocab_size: 50000,
    };
    match err {
        TensorError::EmbeddingIndexOutOfRange { index, vocab_size } => {
            assert_eq!(index, 99999);
            assert_eq!(vocab_size, 50000);
        }
        other => panic!("expected EmbeddingIndexOutOfRange, got: {other:?}"),
    }
}

#[test]
fn test_tensor_error_match_topology_error_fields() {
    let err = TensorError::TopologyError {
        node_name: "relu_7".to_string(),
        index: 12,
        missing_input: 99,
    };
    match err {
        TensorError::TopologyError {
            ref node_name,
            index,
            missing_input,
        } => {
            assert_eq!(node_name, "relu_7");
            assert_eq!(index, 12);
            assert_eq!(missing_input, 99);
        }
        other => panic!("expected TopologyError, got: {other:?}"),
    }
}

#[test]
fn test_tensor_error_match_non_finite_data_fields() {
    let err = TensorError::NonFiniteData {
        name: "layer.0.weight".to_string(),
        count: 42,
    };
    match err {
        TensorError::NonFiniteData { ref name, count } => {
            assert_eq!(name, "layer.0.weight");
            assert_eq!(count, 42);
        }
        other => panic!("expected NonFiniteData, got: {other:?}"),
    }
}

#[test]
fn test_tensor_error_match_weight_conversion_failed_fields() {
    let err = TensorError::WeightConversionFailed {
        dtype: DType::BF16,
        device: Device::metal(),
    };
    match err {
        TensorError::WeightConversionFailed { dtype, device } => {
            assert_eq!(dtype, DType::BF16);
            assert_eq!(device, Device::metal());
        }
        other => panic!("expected WeightConversionFailed, got: {other:?}"),
    }
}

// ===========================================================================
// 3. TensorError — edge cases
// ===========================================================================

#[test]
fn test_shape_mismatch_empty_shapes() {
    let err = TensorError::shape_mismatch(vec![], vec![]);
    assert_eq!(err.to_string(), "Shape mismatch: expected [], got []");
}

#[test]
fn test_shape_mismatch_high_rank_shapes() {
    let expected = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let actual = vec![8, 7, 6, 5, 4, 3, 2, 1];
    let err = TensorError::shape_mismatch(expected, actual);
    let msg = err.to_string();
    assert!(msg.contains("[1, 2, 3, 4, 5, 6, 7, 8]"));
    assert!(msg.contains("[8, 7, 6, 5, 4, 3, 2, 1]"));
}

#[test]
fn test_dimension_overflow_single_dim() {
    let err = TensorError::DimensionOverflow {
        dims: vec![usize::MAX],
    };
    let msg = err.to_string();
    assert!(msg.contains("Dimension product overflow"));
}

#[test]
fn test_invalid_shape_empty_message() {
    let err = TensorError::InvalidShape(String::new());
    assert_eq!(err.to_string(), "Invalid shape: ");
}

#[test]
fn test_invalid_bounds_empty_message() {
    let err = TensorError::InvalidBounds(String::new());
    assert_eq!(err.to_string(), "Invalid bounds: ");
}

#[test]
fn test_check_dim_zero_rank_always_fails() {
    let result = check_dim(0, 0);
    assert!(result.is_err());
    match result.unwrap_err() {
        TensorError::DimensionOutOfRange { dim: 0, rank: 0 } => {}
        other => panic!("expected DimensionOutOfRange, got: {other:?}"),
    }
}

// ===========================================================================
// 4. TensorError — Debug format
// ===========================================================================

#[test]
fn test_tensor_error_debug_contains_variant_name() {
    let err = TensorError::RankMismatch {
        expected: 3,
        actual: 2,
    };
    let debug = format!("{err:?}");
    assert!(
        debug.contains("RankMismatch"),
        "Debug should contain variant name, got: {debug}"
    );
}

#[test]
fn test_tensor_error_debug_invalid_shape_contains_message() {
    let err = TensorError::InvalidShape("bad shape data".to_string());
    let debug = format!("{err:?}");
    assert!(debug.contains("InvalidShape"));
    assert!(debug.contains("bad shape data"));
}

// ===========================================================================
// 5. TensorError — std::error::Error trait (source chain)
// ===========================================================================

#[test]
fn test_io_error_source_chain() {
    let io_err = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "pipe broken");
    let tensor_err: TensorError = io_err.into();
    // The IoError variant should have the original io::Error as its source.
    let source = tensor_err.source().expect("IoError should have a source");
    let downcasted = source
        .downcast_ref::<std::io::Error>()
        .expect("source should be io::Error");
    assert_eq!(downcasted.kind(), std::io::ErrorKind::BrokenPipe);
}

#[test]
fn test_simple_variants_have_no_source() {
    let variants: Vec<TensorError> = vec![
        TensorError::RankMismatch {
            expected: 1,
            actual: 2,
        },
        TensorError::InvalidShape("x".to_string()),
        TensorError::DimensionOutOfRange { dim: 0, rank: 0 },
        TensorError::ValueOutOfRange {
            description: "test",
        },
        TensorError::DataLengthMismatch {
            expected: 1,
            actual: 2,
        },
        TensorError::OutOfMemory {
            requested: 1,
            available: 0,
        },
        TensorError::InvalidBounds("x".to_string()),
        TensorError::Unsupported("x".to_string()),
        TensorError::TensorNotFound {
            name: "x".to_string(),
        },
        TensorError::NonFiniteData {
            name: "x".to_string(),
            count: 1,
        },
    ];
    for v in &variants {
        assert!(v.source().is_none(), "variant {v:?} should have no source");
    }
}

// ===========================================================================
// 6. BackendDomain — all variants in error messages
// ===========================================================================

#[test]
fn test_backend_domain_all_variants_in_error_message() {
    let domains = [
        (BackendDomain::Device, "Device error:"),
        (BackendDomain::Cpu, "Cpu error:"),
        (BackendDomain::Metal, "Metal error:"),
        (BackendDomain::Cuda, "Cuda error:"),
        (BackendDomain::Vulkan, "Vulkan error:"),
        (BackendDomain::Ane, "Ane error:"),
        (BackendDomain::Bounds, "Bounds error:"),
        (BackendDomain::Verification, "Verification error:"),
        (BackendDomain::Whisper, "Whisper error:"),
        (BackendDomain::Qwen3, "Qwen3 error:"),
        (BackendDomain::Glm5, "Glm5 error:"),
        (BackendDomain::Kokoro, "Kokoro error:"),
    ];
    for (domain, expected_prefix) in domains {
        let err = TensorError::backend_failure(
            domain,
            BackendErrorKind::Other,
            "test message".to_string(),
        );
        let msg = err.to_string();
        assert!(
            msg.starts_with(expected_prefix),
            "domain {domain:?}: expected prefix '{expected_prefix}', got: {msg}"
        );
    }
}

// ===========================================================================
// 7. BackendErrorKind — all variants accessible via accessor
// ===========================================================================

#[test]
fn test_backend_error_kind_all_variants_via_accessor() {
    let kinds = [
        BackendErrorKind::OutOfMemory,
        BackendErrorKind::KernelCompile,
        BackendErrorKind::Timeout,
        BackendErrorKind::DispatchFailed,
        BackendErrorKind::Other,
    ];
    for kind in kinds {
        let err = TensorError::backend_failure(BackendDomain::Metal, kind, "test".to_string());
        assert_eq!(
            err.backend_error_kind(),
            Some(kind),
            "accessor should return {kind:?}"
        );
    }
}

// ===========================================================================
// 8. BackendDomain and BackendErrorKind — Copy, Clone, PartialEq
// ===========================================================================

#[test]
fn test_backend_domain_copy_clone() {
    let d = BackendDomain::Metal;
    let copied = d;
    let cloned = d;
    assert_eq!(d, copied);
    assert_eq!(d, cloned);
}

#[test]
fn test_backend_error_kind_copy_clone() {
    let k = BackendErrorKind::Timeout;
    let copied = k;
    let cloned = k;
    assert_eq!(k, copied);
    assert_eq!(k, cloned);
}

// ===========================================================================
// 9. DType — size relationship invariants
// ===========================================================================

#[test]
fn test_dtype_f64_largest_size() {
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    for dt in all {
        assert!(
            DType::F64.size_bytes() >= dt.size_bytes(),
            "F64 (8) should be >= {dt:?} ({})",
            dt.size_bytes()
        );
    }
}

#[test]
fn test_dtype_u8_and_bool_smallest_size() {
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    for dt in all {
        assert!(
            dt.size_bytes() >= DType::U8.size_bytes(),
            "{dt:?} ({}) should be >= U8 (1)",
            dt.size_bytes()
        );
    }
}

#[test]
fn test_dtype_half_precisions_same_size() {
    assert_eq!(DType::F16.size_bytes(), DType::BF16.size_bytes());
}

#[test]
fn test_dtype_32bit_types_same_size() {
    assert_eq!(DType::F32.size_bytes(), DType::I32.size_bytes());
    assert_eq!(DType::F32.size_bytes(), DType::U32.size_bytes());
}

#[test]
fn test_dtype_64bit_types_same_size() {
    assert_eq!(DType::F64.size_bytes(), DType::I64.size_bytes());
}

// ===========================================================================
// 10. DType — float/int partition is exhaustive and exclusive
// ===========================================================================

#[test]
fn test_dtype_every_variant_is_float_or_int_or_bool() {
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    for dt in all {
        let is_something = dt.is_float() || dt.is_int() || matches!(dt, DType::Bool);
        assert!(is_something, "{dt:?} is not float, not int, and not Bool");
    }
}

#[test]
fn test_dtype_float_and_int_mutually_exclusive() {
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    for dt in all {
        assert!(
            !(dt.is_float() && dt.is_int()),
            "{dt:?} is both float and int"
        );
    }
}

// ===========================================================================
// 11. DType — Display roundtrip consistency
// ===========================================================================

#[test]
fn test_dtype_display_is_lowercase_name() {
    let all = [
        (DType::F32, "f32"),
        (DType::F16, "f16"),
        (DType::BF16, "bf16"),
        (DType::F64, "f64"),
        (DType::I32, "i32"),
        (DType::I64, "i64"),
        (DType::U32, "u32"),
        (DType::U8, "u8"),
        (DType::Bool, "bool"),
    ];
    for (dt, expected) in all {
        assert_eq!(
            dt.to_string(),
            expected,
            "{dt:?} Display should be {expected}"
        );
    }
}

#[test]
fn test_dtype_display_all_unique() {
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];
    let strings: HashSet<String> = all.iter().map(ToString::to_string).collect();
    assert_eq!(
        strings.len(),
        9,
        "all DType Display strings should be unique"
    );
}

// ===========================================================================
// 12. Device — constructor methods produce correct device_id
// ===========================================================================

#[test]
fn test_device_metal_constructor_device_id_zero() {
    match Device::metal() {
        Device::Metal { device_id } => assert_eq!(device_id, 0),
        other => panic!("expected Metal, got: {other:?}"),
    }
}

#[test]
fn test_device_cuda_constructor_device_id_zero() {
    match Device::cuda() {
        Device::Cuda { device_id } => assert_eq!(device_id, 0),
        other => panic!("expected Cuda, got: {other:?}"),
    }
}

#[test]
fn test_device_vulkan_constructor_device_id_zero() {
    match Device::vulkan() {
        Device::Vulkan { device_id } => assert_eq!(device_id, 0),
        other => panic!("expected Vulkan, got: {other:?}"),
    }
}

// ===========================================================================
// 13. Device — predicate methods are mutually exclusive on base type
// ===========================================================================

#[test]
fn test_device_predicates_exactly_one_true() {
    let devices = [
        Device::Cpu,
        Device::metal(),
        Device::cuda(),
        Device::vulkan(),
        Device::Ane,
    ];
    for d in devices {
        let count = [
            d.is_cpu(),
            d.is_metal(),
            d.is_cuda(),
            d.is_vulkan(),
            d.is_ane(),
        ]
        .iter()
        .filter(|&&b| b)
        .count();
        assert_eq!(
            count, 1,
            "exactly one predicate should be true for {d:?}, got {count}"
        );
    }
}

#[test]
fn test_device_is_accelerator_equals_not_cpu() {
    let devices = [
        Device::Cpu,
        Device::metal(),
        Device::Metal { device_id: 3 },
        Device::cuda(),
        Device::Cuda { device_id: 7 },
        Device::vulkan(),
        Device::Vulkan { device_id: 2 },
        Device::Ane,
    ];
    for d in devices {
        assert_eq!(
            d.is_accelerator(),
            !d.is_cpu(),
            "is_accelerator should be !is_cpu for {d:?}"
        );
    }
}

// ===========================================================================
// 14. Device — non-default device_id values
// ===========================================================================

#[test]
fn test_device_metal_nonzero_device_id() {
    let d = Device::Metal { device_id: 5 };
    assert!(d.is_metal());
    assert!(d.is_gpu());
    assert_eq!(d.to_string(), "Metal(5)");
}

#[test]
fn test_device_cuda_nonzero_device_id() {
    let d = Device::Cuda { device_id: 3 };
    assert!(d.is_cuda());
    assert!(d.is_gpu());
    assert_eq!(d.to_string(), "CUDA(3)");
}

#[test]
fn test_device_vulkan_nonzero_device_id() {
    let d = Device::Vulkan { device_id: 9 };
    assert!(d.is_vulkan());
    assert!(d.is_gpu());
    assert_eq!(d.to_string(), "Vulkan(9)");
}

// ===========================================================================
// 15. Device — different device_ids are not equal
// ===========================================================================

#[test]
fn test_device_same_variant_different_id_not_equal() {
    assert_ne!(
        Device::Metal { device_id: 0 },
        Device::Metal { device_id: 1 }
    );
    assert_ne!(Device::Cuda { device_id: 0 }, Device::Cuda { device_id: 1 });
    assert_ne!(
        Device::Vulkan { device_id: 0 },
        Device::Vulkan { device_id: 1 }
    );
}

// ===========================================================================
// 16. DtypeConversion error with all dtype combinations
// ===========================================================================

#[test]
fn test_dtype_conversion_error_uses_display_names() {
    let err = TensorError::DtypeConversion {
        source_dtype: DType::BF16,
        target_dtype: DType::U8,
        reason: "out of range".to_string(),
    };
    let msg = err.to_string();
    assert!(msg.contains("bf16"), "should contain source dtype display");
    assert!(msg.contains("u8"), "should contain target dtype display");
    assert!(msg.contains("out of range"), "should contain reason");
}

// ===========================================================================
// 17. DeviceAllocationUnavailable with all device variants
// ===========================================================================

#[test]
fn test_device_allocation_unavailable_all_devices() {
    let devices = [
        (Device::Cpu, "CPU"),
        (Device::metal(), "Metal(0)"),
        (Device::cuda(), "CUDA(0)"),
        (Device::vulkan(), "Vulkan(0)"),
        (Device::Ane, "ANE"),
    ];
    for (device, expected_name) in devices {
        let err = TensorError::DeviceAllocationUnavailable { device };
        let msg = err.to_string();
        assert!(
            msg.contains(expected_name),
            "message should contain device name '{expected_name}', got: {msg}"
        );
    }
}

// ===========================================================================
// 18. TensorError convenience constructors vs direct struct construction
// ===========================================================================

#[test]
fn test_shape_mismatch_constructor_matches_expected_display() {
    let err = TensorError::shape_mismatch(vec![10, 20], vec![10, 30]);
    let msg = err.to_string();
    assert_eq!(msg, "Shape mismatch: expected [10, 20], got [10, 30]");
}

#[test]
fn test_dtype_mismatch_constructor_matches_expected_display() {
    let err = TensorError::dtype_mismatch(DType::F16, DType::F64);
    let msg = err.to_string();
    assert_eq!(msg, "Data type mismatch: expected f16, got f64");
}

#[test]
fn test_device_transfer_constructor_matches_expected_display() {
    let err = TensorError::device_transfer(Device::metal(), Device::cuda());
    let msg = err.to_string();
    assert_eq!(
        msg,
        "Device transfer unavailable: Metal(0) -> CUDA(0) transfer not yet implemented"
    );
}

// ===========================================================================
// 19. ndarray::ShapeError → TensorError conversion
// ===========================================================================

#[test]
fn test_ndarray_shape_error_converts_to_invalid_shape() {
    // Trigger an ndarray ShapeError by reshaping with wrong element count.
    let arr = ndarray::Array1::<f32>::zeros(6);
    let result = arr.into_shape_with_order((2, 4));
    assert!(result.is_err());
    let shape_err = result.unwrap_err();
    let tensor_err: TensorError = shape_err.into();
    match tensor_err {
        TensorError::InvalidShape(ref msg) => {
            assert!(!msg.is_empty(), "message should be non-empty");
        }
        other => panic!("expected InvalidShape, got: {other:?}"),
    }
}

// ===========================================================================
// 20. BackendFailure with source — source chain walking
// ===========================================================================

#[derive(Debug, thiserror::Error)]
#[error("inner error: {0}")]
struct InnerError(String);

#[derive(Debug, thiserror::Error)]
#[error("outer error")]
struct OuterError {
    #[source]
    inner: InnerError,
}

#[test]
fn test_backend_failure_source_chain_walk() {
    let outer = OuterError {
        inner: InnerError("root cause".to_string()),
    };
    let err = TensorError::backend_failure_with_source(
        BackendDomain::Verification,
        BackendErrorKind::Other,
        "verification failed".to_string(),
        outer,
    );
    // Walk the chain: TensorError → OuterError → InnerError
    let source1 = err.source().expect("should have source (OuterError)");
    let outer_ref = source1
        .downcast_ref::<OuterError>()
        .expect("should downcast to OuterError");
    assert_eq!(outer_ref.to_string(), "outer error");

    let source2 = outer_ref
        .source()
        .expect("OuterError should have source (InnerError)");
    let inner_ref = source2
        .downcast_ref::<InnerError>()
        .expect("should downcast to InnerError");
    assert_eq!(inner_ref.0, "root cause");
}

// ===========================================================================
// 21. Device — Display for high device_id values
// ===========================================================================

#[test]
fn test_device_display_high_device_id() {
    let d = Device::Metal {
        device_id: u32::MAX,
    };
    let msg = d.to_string();
    assert_eq!(msg, format!("Metal({})", u32::MAX));
}

// ===========================================================================
// 22. DType — Debug and Display are different representations
// ===========================================================================

#[test]
fn test_dtype_debug_vs_display_different() {
    // Debug uses variant name (e.g. "F32"), Display uses lowercase (e.g. "f32")
    let all = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
    ];
    for dt in all {
        let debug = format!("{dt:?}");
        let display = format!("{dt}");
        assert_ne!(debug, display, "{dt:?}: Debug and Display should differ");
    }
}

// ===========================================================================
// 23. BackendDomain — all variants have distinct Debug strings
// ===========================================================================

#[test]
fn test_backend_domain_debug_all_distinct() {
    let domains = [
        BackendDomain::Device,
        BackendDomain::Cpu,
        BackendDomain::Metal,
        BackendDomain::Cuda,
        BackendDomain::Vulkan,
        BackendDomain::Ane,
        BackendDomain::Bounds,
        BackendDomain::Verification,
        BackendDomain::Whisper,
        BackendDomain::Qwen3,
        BackendDomain::Glm5,
        BackendDomain::Kokoro,
    ];
    let debug_strings: HashSet<String> = domains.iter().map(|d| format!("{d:?}")).collect();
    assert_eq!(
        debug_strings.len(),
        domains.len(),
        "all BackendDomain variants should have unique Debug strings"
    );
}

// ===========================================================================
// 24. BackendErrorKind — all variants have distinct Debug strings
// ===========================================================================

#[test]
fn test_backend_error_kind_debug_all_distinct() {
    let kinds = [
        BackendErrorKind::OutOfMemory,
        BackendErrorKind::KernelCompile,
        BackendErrorKind::Timeout,
        BackendErrorKind::DispatchFailed,
        BackendErrorKind::Other,
    ];
    let debug_strings: HashSet<String> = kinds.iter().map(|k| format!("{k:?}")).collect();
    assert_eq!(
        debug_strings.len(),
        kinds.len(),
        "all BackendErrorKind variants should have unique Debug strings"
    );
}

// ===========================================================================
// 25. TensorError — ZeroLengthDimension edge case
// ===========================================================================

#[test]
fn test_zero_length_dimension_axis_zero() {
    let err = TensorError::ZeroLengthDimension {
        axis: 0,
        operation: "batch_norm",
    };
    assert_eq!(
        err.to_string(),
        "Zero-length dimension: axis 0 has size 0 (operation: batch_norm)"
    );
}

// ===========================================================================
// 26. TensorError — DimensionOverflow with empty dims
// ===========================================================================

#[test]
fn test_dimension_overflow_empty_dims() {
    let err = TensorError::DimensionOverflow { dims: vec![] };
    let msg = err.to_string();
    assert!(msg.contains("Dimension product overflow"));
    assert!(msg.contains("[]"));
}

// ===========================================================================
// 27. Device — Hash consistency with Eq
// ===========================================================================

#[test]
fn test_device_hash_consistency_with_eq() {
    // Equal values must have equal hashes.
    use std::hash::{Hash, Hasher};
    let d1 = Device::Metal { device_id: 0 };
    let d2 = Device::metal();
    assert_eq!(d1, d2);

    let mut h1 = std::collections::hash_map::DefaultHasher::new();
    let mut h2 = std::collections::hash_map::DefaultHasher::new();
    d1.hash(&mut h1);
    d2.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

// ===========================================================================
// 28. DType — Hash consistency with Eq
// ===========================================================================

#[test]
fn test_dtype_hash_consistency_with_eq() {
    use std::hash::{Hash, Hasher};
    let dt1 = DType::F32;
    let dt2 = DType::F32;
    assert_eq!(dt1, dt2);

    let mut h1 = std::collections::hash_map::DefaultHasher::new();
    let mut h2 = std::collections::hash_map::DefaultHasher::new();
    dt1.hash(&mut h1);
    dt2.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

// ===========================================================================
// 29. TensorError is Send + Sync
// ===========================================================================

#[test]
fn test_tensor_error_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<TensorError>();
}

#[test]
fn test_tensor_error_is_sync() {
    fn assert_sync<T: Sync>() {}
    assert_sync::<TensorError>();
}

// ===========================================================================
// 30. Device — is_gpu vs specific predicates consistency
// ===========================================================================

#[test]
fn test_device_is_gpu_iff_metal_or_cuda_or_vulkan() {
    let devices = [
        Device::Cpu,
        Device::metal(),
        Device::Metal { device_id: 2 },
        Device::cuda(),
        Device::Cuda { device_id: 4 },
        Device::vulkan(),
        Device::Vulkan { device_id: 6 },
        Device::Ane,
    ];
    for d in devices {
        let expected_gpu = d.is_metal() || d.is_cuda() || d.is_vulkan();
        assert_eq!(
            d.is_gpu(),
            expected_gpu,
            "is_gpu should be (is_metal || is_cuda || is_vulkan) for {d:?}"
        );
    }
}
