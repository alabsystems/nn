// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests covering TensorError variant construction and Display, DType byte
//! size and classification methods, Device equality and Display format,
//! and error source chain / downcast behavior.

use std::collections::HashSet;
use std::error::Error;

use crate::error::{BackendDomain, BackendErrorKind, TensorError};
use crate::{DType, Device};

// ===========================================================================
// A. TensorError variant construction and Display
// ===========================================================================

#[test]
fn test_shape_mismatch_display() {
    let e = TensorError::shape_mismatch(vec![2, 3], vec![4, 5]);
    assert_eq!(e.to_string(), "Shape mismatch: expected [2, 3], got [4, 5]");
}

#[test]
fn test_rank_mismatch_display() {
    let e = TensorError::RankMismatch {
        expected: 3,
        actual: 2,
    };
    assert_eq!(e.to_string(), "Rank mismatch: expected 3 dimensions, got 2");
}

#[test]
fn test_invalid_shape_display() {
    let e = TensorError::InvalidShape("zero-length dimension".to_string());
    assert_eq!(e.to_string(), "Invalid shape: zero-length dimension");
}

#[test]
fn test_dimension_out_of_range_display() {
    let e = TensorError::DimensionOutOfRange { dim: 7, rank: 3 };
    assert_eq!(e.to_string(), "Dimension 7 out of range for rank 3");
}

#[test]
fn test_conv_parameter_invalid_display() {
    let e = TensorError::ConvParameterInvalid {
        param: "stride",
        value: 0,
        reason: "must be positive",
    };
    assert_eq!(
        e.to_string(),
        "Conv error: stride = 0 is invalid (must be positive)"
    );
}

#[test]
fn test_value_out_of_range_display() {
    let e = TensorError::ValueOutOfRange {
        description: "alpha must be positive",
    };
    assert_eq!(e.to_string(), "Value out of range: alpha must be positive");
}

#[test]
fn test_dtype_conversion_display() {
    let e = TensorError::DtypeConversion {
        source_dtype: DType::F32,
        target_dtype: DType::I32,
        reason: "cannot convert float to int".to_string(),
    };
    assert_eq!(
        e.to_string(),
        "Dtype conversion f32 \u{2192} i32: cannot convert float to int"
    );
}

#[test]
fn test_embedding_index_out_of_range_display() {
    let e = TensorError::EmbeddingIndexOutOfRange {
        index: 50000,
        vocab_size: 32000,
    };
    assert_eq!(
        e.to_string(),
        "Embedding index 50000 out of range for vocab size 32000"
    );
}

#[test]
fn test_zero_length_dimension_display() {
    let e = TensorError::ZeroLengthDimension {
        axis: 2,
        operation: "softmax",
    };
    assert_eq!(
        e.to_string(),
        "Zero-length dimension: axis 2 has size 0 (operation: softmax)"
    );
}

#[test]
fn test_device_allocation_unavailable_display() {
    let e = TensorError::DeviceAllocationUnavailable {
        device: Device::Ane,
    };
    assert_eq!(
        e.to_string(),
        "Device allocation unavailable: ANE backend not yet implemented"
    );
}

#[test]
fn test_device_transfer_unavailable_display() {
    let e = TensorError::device_transfer(Device::Cpu, Device::metal());
    assert_eq!(
        e.to_string(),
        "Device transfer unavailable: CPU -> Metal(0) transfer not yet implemented"
    );
}

#[test]
fn test_backend_failure_display() {
    let e = TensorError::backend_failure(
        BackendDomain::Cuda,
        BackendErrorKind::OutOfMemory,
        "allocation failed".to_string(),
    );
    assert_eq!(e.to_string(), "Cuda error: allocation failed");
}

#[test]
fn test_data_length_mismatch_display() {
    let e = TensorError::DataLengthMismatch {
        expected: 12,
        actual: 10,
    };
    assert_eq!(
        e.to_string(),
        "Data length mismatch: shape requires 12 elements, got 10"
    );
}

#[test]
fn test_dimension_overflow_display() {
    let e = TensorError::DimensionOverflow {
        dims: vec![usize::MAX, 2],
    };
    let msg = e.to_string();
    assert!(msg.contains("Dimension product overflow"));
    assert!(msg.contains("exceed usize::MAX"));
}

#[test]
fn test_dtype_mismatch_display() {
    let e = TensorError::dtype_mismatch(DType::F32, DType::I64);
    assert_eq!(e.to_string(), "Data type mismatch: expected f32, got i64");
}

#[test]
fn test_out_of_memory_display() {
    let e = TensorError::OutOfMemory {
        requested: 1024,
        available: 512,
    };
    assert_eq!(
        e.to_string(),
        "Out of memory: requested 1024 bytes, available 512"
    );
}

#[test]
fn test_invalid_bounds_display() {
    let e = TensorError::InvalidBounds("lower > upper".to_string());
    assert_eq!(e.to_string(), "Invalid bounds: lower > upper");
}

#[test]
fn test_unsupported_display() {
    let e = TensorError::Unsupported("int4 quantization".to_string());
    assert_eq!(e.to_string(), "Operation not supported: int4 quantization");
}

#[test]
fn test_tensor_not_found_display() {
    let e = TensorError::TensorNotFound {
        name: "encoder.layer.0.weight".to_string(),
    };
    assert_eq!(e.to_string(), "Tensor not found: encoder.layer.0.weight");
}

#[test]
fn test_non_finite_data_display() {
    let e = TensorError::NonFiniteData {
        name: "decoder.bias".to_string(),
        count: 5,
    };
    assert_eq!(
        e.to_string(),
        "Non-finite data: 5 NaN/Inf values in tensor 'decoder.bias'"
    );
}

#[test]
fn test_topology_error_display() {
    let e = TensorError::TopologyError {
        node_name: "add_3".to_string(),
        index: 7,
        missing_input: 42,
    };
    let msg = e.to_string();
    assert!(msg.contains("add_3"));
    assert!(msg.contains("index 7"));
    assert!(msg.contains("input_id 42"));
}

#[test]
fn test_weight_conversion_failed_display() {
    let e = TensorError::WeightConversionFailed {
        dtype: DType::I32,
        device: Device::Cpu,
    };
    assert_eq!(
        e.to_string(),
        "Weight conversion failed: dtype=i32, device=CPU"
    );
}

#[test]
fn test_io_error_display() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let e: TensorError = io_err.into();
    assert_eq!(e.to_string(), "IO error: file missing");
}

// ===========================================================================
// B. DType byte size and classification
// ===========================================================================

#[test]
fn test_dtype_size_bytes_all_variants() {
    assert_eq!(DType::F32.size_bytes(), 4);
    assert_eq!(DType::F16.size_bytes(), 2);
    assert_eq!(DType::BF16.size_bytes(), 2);
    assert_eq!(DType::F64.size_bytes(), 8);
    assert_eq!(DType::I32.size_bytes(), 4);
    assert_eq!(DType::I64.size_bytes(), 8);
    assert_eq!(DType::U32.size_bytes(), 4);
    assert_eq!(DType::U8.size_bytes(), 1);
    assert_eq!(DType::Bool.size_bytes(), 1);
}

#[test]
fn test_dtype_all_sizes_positive() {
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
        assert!(dt.size_bytes() > 0, "{dt:?} should have nonzero size");
    }
}

#[test]
fn test_dtype_is_float_classification() {
    assert!(DType::F32.is_float());
    assert!(DType::F16.is_float());
    assert!(DType::BF16.is_float());
    assert!(DType::F64.is_float());
    assert!(!DType::I32.is_float());
    assert!(!DType::I64.is_float());
    assert!(!DType::U32.is_float());
    assert!(!DType::U8.is_float());
    assert!(!DType::Bool.is_float());
}

#[test]
fn test_dtype_is_int_classification() {
    assert!(!DType::F32.is_int());
    assert!(!DType::F16.is_int());
    assert!(!DType::BF16.is_int());
    assert!(!DType::F64.is_int());
    assert!(DType::I32.is_int());
    assert!(DType::I64.is_int());
    assert!(DType::U32.is_int());
    assert!(DType::U8.is_int());
    assert!(!DType::Bool.is_int());
}

#[test]
fn test_dtype_bool_is_neither_float_nor_int() {
    assert!(!DType::Bool.is_float());
    assert!(!DType::Bool.is_int());
}

#[test]
fn test_dtype_display_all_variants() {
    assert_eq!(DType::F32.to_string(), "f32");
    assert_eq!(DType::F16.to_string(), "f16");
    assert_eq!(DType::BF16.to_string(), "bf16");
    assert_eq!(DType::F64.to_string(), "f64");
    assert_eq!(DType::I32.to_string(), "i32");
    assert_eq!(DType::I64.to_string(), "i64");
    assert_eq!(DType::U32.to_string(), "u32");
    assert_eq!(DType::U8.to_string(), "u8");
    assert_eq!(DType::Bool.to_string(), "bool");
}

#[test]
fn test_dtype_equality() {
    assert_eq!(DType::F32, DType::F32);
    assert_ne!(DType::F32, DType::F16);
    assert_ne!(DType::F32, DType::BF16);
    assert_ne!(DType::I32, DType::U32);
}

#[test]
fn test_dtype_hash_uniqueness() {
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
    let set: HashSet<DType> = all.iter().copied().collect();
    assert_eq!(set.len(), 9, "all DType variants should have unique hashes");
}

#[test]
fn test_dtype_copy_and_clone() {
    let dt = DType::F32;
    let copied = dt;
    let cloned = dt;
    assert_eq!(dt, copied);
    assert_eq!(dt, cloned);
}

#[test]
fn test_dtype_debug_format() {
    let debug = format!("{:?}", DType::BF16);
    assert_eq!(debug, "BF16");
}

// ===========================================================================
// C. Device equality, Display, hash
// ===========================================================================

#[test]
fn test_device_default_is_cpu() {
    assert_eq!(Device::default(), Device::Cpu);
}

#[test]
fn test_device_equality() {
    assert_eq!(Device::Cpu, Device::Cpu);
    assert_eq!(Device::metal(), Device::Metal { device_id: 0 });
    assert_ne!(Device::Cpu, Device::metal());
    assert_ne!(
        Device::Metal { device_id: 0 },
        Device::Metal { device_id: 1 }
    );
    assert_ne!(Device::metal(), Device::cuda());
}

#[test]
fn test_device_display_all_variants() {
    assert_eq!(Device::Cpu.to_string(), "CPU");
    assert_eq!(Device::metal().to_string(), "Metal(0)");
    assert_eq!(Device::Metal { device_id: 5 }.to_string(), "Metal(5)");
    assert_eq!(Device::cuda().to_string(), "CUDA(0)");
    assert_eq!(Device::Cuda { device_id: 3 }.to_string(), "CUDA(3)");
    assert_eq!(Device::vulkan().to_string(), "Vulkan(0)");
    assert_eq!(Device::Vulkan { device_id: 1 }.to_string(), "Vulkan(1)");
    assert_eq!(Device::Ane.to_string(), "ANE");
}

#[test]
fn test_device_hash_uniqueness() {
    let devices = [
        Device::Cpu,
        Device::Metal { device_id: 0 },
        Device::Metal { device_id: 1 },
        Device::Cuda { device_id: 0 },
        Device::Vulkan { device_id: 0 },
        Device::Ane,
    ];
    let set: HashSet<Device> = devices.iter().copied().collect();
    assert_eq!(set.len(), 6, "distinct devices should have unique hashes");
}

#[test]
fn test_device_copy_and_clone() {
    let d = Device::metal();
    let copied = d;
    let cloned = d;
    assert_eq!(d, copied);
    assert_eq!(d, cloned);
}

#[test]
fn test_device_is_gpu() {
    assert!(!Device::Cpu.is_gpu());
    assert!(Device::metal().is_gpu());
    assert!(Device::cuda().is_gpu());
    assert!(Device::vulkan().is_gpu());
    assert!(!Device::Ane.is_gpu());
}

#[test]
fn test_device_is_cpu() {
    assert!(Device::Cpu.is_cpu());
    assert!(!Device::metal().is_cpu());
    assert!(!Device::Ane.is_cpu());
}

#[test]
fn test_device_is_accelerator() {
    assert!(!Device::Cpu.is_accelerator());
    assert!(Device::metal().is_accelerator());
    assert!(Device::cuda().is_accelerator());
    assert!(Device::vulkan().is_accelerator());
    assert!(Device::Ane.is_accelerator());
}

#[test]
fn test_device_debug_format() {
    let debug = format!("{:?}", Device::Cuda { device_id: 7 });
    assert!(debug.contains("Cuda"));
    assert!(debug.contains("7"));
}

// ===========================================================================
// D. Error source chain and downcast
// ===========================================================================

#[derive(Debug, thiserror::Error)]
#[error("custom error: {msg}")]
struct CustomBackendErr {
    msg: String,
}

#[test]
fn test_backend_failure_no_source() {
    let e = TensorError::backend_failure(
        BackendDomain::Metal,
        BackendErrorKind::KernelCompile,
        "MSL compile failed".to_string(),
    );
    assert!(e.source().is_none());
}

#[test]
fn test_backend_failure_with_source_chain() {
    let inner = CustomBackendErr {
        msg: "OOM".to_string(),
    };
    let e = TensorError::backend_failure_with_source(
        BackendDomain::Metal,
        BackendErrorKind::OutOfMemory,
        "OOM".to_string(),
        inner,
    );
    let src = e.source().expect("should have source");
    let downcast = src
        .downcast_ref::<CustomBackendErr>()
        .expect("downcast should succeed");
    assert_eq!(downcast.msg, "OOM");
}

#[test]
fn test_backend_error_kind_accessor() {
    let e = TensorError::backend_failure(
        BackendDomain::Cuda,
        BackendErrorKind::DispatchFailed,
        "dispatch error".to_string(),
    );
    assert_eq!(
        e.backend_error_kind(),
        Some(BackendErrorKind::DispatchFailed)
    );
}

#[test]
fn test_backend_error_kind_none_for_non_backend() {
    let e = TensorError::Unsupported("test".to_string());
    assert_eq!(e.backend_error_kind(), None);
}

#[test]
fn test_backtrace_present_for_shape_mismatch() {
    let e = TensorError::shape_mismatch(vec![1], vec![2]);
    assert!(e.backtrace().is_some());
}

#[test]
fn test_backtrace_present_for_dtype_mismatch() {
    let e = TensorError::dtype_mismatch(DType::F32, DType::I32);
    assert!(e.backtrace().is_some());
}

#[test]
fn test_backtrace_present_for_device_transfer() {
    let e = TensorError::device_transfer(Device::Cpu, Device::metal());
    assert!(e.backtrace().is_some());
}

#[test]
fn test_backtrace_present_for_backend_failure() {
    let e = TensorError::backend_failure(
        BackendDomain::Metal,
        BackendErrorKind::Other,
        "test".to_string(),
    );
    assert!(e.backtrace().is_some());
}

#[test]
fn test_backtrace_none_for_simple_variants() {
    assert!(TensorError::InvalidShape("x".to_string())
        .backtrace()
        .is_none());
    assert!(TensorError::Unsupported("x".to_string())
        .backtrace()
        .is_none());
    assert!(TensorError::TensorNotFound {
        name: "x".to_string()
    }
    .backtrace()
    .is_none());
    assert!(TensorError::RankMismatch {
        expected: 1,
        actual: 2
    }
    .backtrace()
    .is_none());
}

#[test]
fn test_io_error_from_conversion() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let tensor_err: TensorError = io_err.into();
    match tensor_err {
        TensorError::IoError(ref inner) => {
            assert_eq!(inner.kind(), std::io::ErrorKind::PermissionDenied);
        }
        other => panic!("expected IoError, got: {other:?}"),
    }
}

#[test]
fn test_check_dim_valid_boundary() {
    // dim = rank - 1 is the last valid dimension.
    assert!(crate::error::check_dim(0, 1).is_ok());
    assert!(crate::error::check_dim(4, 5).is_ok());
}

#[test]
fn test_check_dim_at_rank_is_error() {
    let err = crate::error::check_dim(3, 3).unwrap_err();
    match err {
        TensorError::DimensionOutOfRange { dim: 3, rank: 3 } => {}
        other => panic!("expected DimensionOutOfRange, got: {other:?}"),
    }
}

#[test]
fn test_check_dim_beyond_rank_is_error() {
    let err = crate::error::check_dim(10, 2).unwrap_err();
    match err {
        TensorError::DimensionOutOfRange { dim: 10, rank: 2 } => {}
        other => panic!("expected DimensionOutOfRange, got: {other:?}"),
    }
}

// ===========================================================================
// E. BackendDomain and BackendErrorKind coverage
// ===========================================================================

#[test]
fn test_backend_domain_all_variants_debug() {
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
    for domain in domains {
        // Verify Debug is implemented and non-empty.
        let debug = format!("{domain:?}");
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_backend_domain_equality() {
    assert_eq!(BackendDomain::Metal, BackendDomain::Metal);
    assert_ne!(BackendDomain::Metal, BackendDomain::Cuda);
}

#[test]
fn test_backend_error_kind_all_variants() {
    let kinds = [
        BackendErrorKind::OutOfMemory,
        BackendErrorKind::KernelCompile,
        BackendErrorKind::Timeout,
        BackendErrorKind::DispatchFailed,
        BackendErrorKind::Other,
    ];
    for kind in kinds {
        let debug = format!("{kind:?}");
        assert!(!debug.is_empty());
    }
}

#[test]
fn test_backend_error_kind_equality() {
    assert_eq!(BackendErrorKind::OutOfMemory, BackendErrorKind::OutOfMemory);
    assert_ne!(BackendErrorKind::OutOfMemory, BackendErrorKind::Timeout);
}

#[test]
fn test_backend_domain_hash() {
    let set: HashSet<BackendDomain> = [
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
    ]
    .into_iter()
    .collect();
    assert_eq!(
        set.len(),
        12,
        "all BackendDomain variants should hash uniquely"
    );
}

#[test]
fn test_backend_error_kind_hash() {
    let set: HashSet<BackendErrorKind> = [
        BackendErrorKind::OutOfMemory,
        BackendErrorKind::KernelCompile,
        BackendErrorKind::Timeout,
        BackendErrorKind::DispatchFailed,
        BackendErrorKind::Other,
    ]
    .into_iter()
    .collect();
    assert_eq!(
        set.len(),
        5,
        "all BackendErrorKind variants should hash uniquely"
    );
}
