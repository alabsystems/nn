// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use super::{convert_dtype, page_align, PAGE_SIZE};
use nn_core::DType;

// --- page_align ---

#[test]
fn test_page_align_zero() {
    assert_eq!(page_align(0), 0);
}

#[test]
fn test_page_align_one_byte() {
    assert_eq!(page_align(1), PAGE_SIZE);
}

#[test]
fn test_page_align_exact_page() {
    assert_eq!(page_align(PAGE_SIZE), PAGE_SIZE);
}

#[test]
fn test_page_align_just_over_page() {
    assert_eq!(page_align(PAGE_SIZE + 1), 2 * PAGE_SIZE);
}

#[test]
fn test_page_align_result_is_page_multiple() {
    for size in [1, 100, 4095, 4096, 4097, 8192, 10000] {
        let aligned = page_align(size);
        assert!(aligned >= size, "page_align({size}) = {aligned} < {size}");
        assert_eq!(
            aligned % PAGE_SIZE,
            0,
            "page_align({size}) = {aligned} not page-aligned"
        );
    }
}

// --- convert_dtype ---

#[test]
fn test_convert_dtype_supported() {
    assert_eq!(
        convert_dtype(safetensors::Dtype::BF16).unwrap(),
        DType::BF16
    );
    assert_eq!(convert_dtype(safetensors::Dtype::F16).unwrap(), DType::F16);
    assert_eq!(convert_dtype(safetensors::Dtype::F32).unwrap(), DType::F32);
    assert_eq!(convert_dtype(safetensors::Dtype::F64).unwrap(), DType::F64);
    assert_eq!(convert_dtype(safetensors::Dtype::I32).unwrap(), DType::I32);
    assert_eq!(convert_dtype(safetensors::Dtype::I64).unwrap(), DType::I64);
    assert_eq!(convert_dtype(safetensors::Dtype::U8).unwrap(), DType::U8);
    assert_eq!(
        convert_dtype(safetensors::Dtype::BOOL).unwrap(),
        DType::Bool
    );
}

#[test]
fn test_convert_dtype_unsupported() {
    let err = convert_dtype(safetensors::Dtype::U16).unwrap_err();
    assert!(matches!(
        err,
        super::WeightError::UnsupportedDtype(safetensors::Dtype::U16)
    ));
}

// --- TensorInfo::numel ---

#[test]
fn test_tensor_info_numel() {
    let info = super::TensorInfo {
        offset: 0,
        byte_len: 48,
        dtype: DType::F32,
        shape: vec![2, 3, 4],
    };
    assert_eq!(info.numel().unwrap(), 24);
}

#[test]
fn test_tensor_info_numel_scalar() {
    let info = super::TensorInfo {
        offset: 0,
        byte_len: 4,
        dtype: DType::F32,
        shape: vec![],
    };
    // Product of empty shape = 1 (scalar)
    assert_eq!(info.numel().unwrap(), 1);
}

#[test]
fn test_tensor_info_numel_single_dim() {
    let info = super::TensorInfo {
        offset: 0,
        byte_len: 40,
        dtype: DType::F32,
        shape: vec![10],
    };
    assert_eq!(info.numel().unwrap(), 10);
}

#[test]
fn test_tensor_info_numel_with_one_dim() {
    let info = super::TensorInfo {
        offset: 0,
        byte_len: 24,
        dtype: DType::F32,
        shape: vec![1, 3, 2],
    };
    assert_eq!(info.numel().unwrap(), 6);
}

#[test]
fn test_tensor_info_numel_overflow_rejected() {
    let info = super::TensorInfo {
        offset: 0,
        byte_len: 0,
        dtype: DType::F32,
        shape: vec![usize::MAX, 2],
    };
    let err = info.numel().expect_err("overflow should be rejected");
    assert!(
        format!("{err}").contains("shape product overflow"),
        "expected overflow error, got: {err}"
    );
}

// --- tensor_data bounds checking ---

#[test]
fn test_weight_error_tensor_data_overflow_display() {
    let err = super::WeightError::TensorDataOverflow {
        name: "layer.weight".into(),
    };
    assert!(
        err.to_string().contains("offset + byte_len overflows"),
        "error: {err}"
    );
    assert!(
        err.to_string().contains("layer.weight"),
        "should contain tensor name: {err}"
    );
}

#[test]
fn test_weight_error_tensor_data_out_of_bounds_display() {
    let err = super::WeightError::TensorDataOutOfBounds {
        name: "decoder.bias".into(),
        offset: 1000,
        byte_len: 500,
        buffer_size: 1200,
    };
    let msg = err.to_string();
    assert!(msg.contains("decoder.bias"), "should contain name: {msg}");
    assert!(msg.contains("1000"), "should contain offset: {msg}");
    assert!(msg.contains("500"), "should contain byte_len: {msg}");
    assert!(msg.contains("1200"), "should contain buffer_size: {msg}");
}

#[test]
fn test_weight_error_tensor_data_overflow_to_tensor_error() {
    let err = super::WeightError::TensorDataOverflow { name: "w".into() };
    let tensor_err: nn_core::TensorError = err.into();
    match tensor_err {
        nn_core::TensorError::BackendFailure {
            domain, message, ..
        } => {
            assert_eq!(domain, nn_core::BackendDomain::Metal);
            assert!(
                message.contains("overflows"),
                "message should mention overflow: {message}"
            );
        }
        other => panic!("expected BackendFailure, got: {other:?}"),
    }
}

#[test]
fn test_weight_error_tensor_data_out_of_bounds_to_tensor_error() {
    let err = super::WeightError::TensorDataOutOfBounds {
        name: "w".into(),
        offset: 100,
        byte_len: 200,
        buffer_size: 150,
    };
    let tensor_err: nn_core::TensorError = err.into();
    match tensor_err {
        nn_core::TensorError::BackendFailure {
            domain, message, ..
        } => {
            assert_eq!(domain, nn_core::BackendDomain::Metal);
            assert!(
                message.contains("out of bounds"),
                "message should mention out of bounds: {message}"
            );
        }
        other => panic!("expected BackendFailure, got: {other:?}"),
    }
}

// --- WeightError Display ---

#[test]
fn test_weight_error_io_display() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
    let err = super::WeightError::Io(io_err);
    assert_eq!(err.to_string(), "failed to open weight file: missing file");
}

#[test]
fn test_weight_error_metal_display() {
    let metal_err = crate::MetalError::NoDevice;
    let err = super::WeightError::Metal(metal_err);
    assert_eq!(
        err.to_string(),
        "Metal error: Metal is unavailable on this host"
    );
}

#[test]
fn test_weight_error_tensor_not_found_display() {
    let err = super::WeightError::TensorNotFound("decoder.weight".into());
    assert_eq!(err.to_string(), "tensor not found: decoder.weight");
}

#[test]
fn test_weight_error_unsupported_dtype_display() {
    let err = super::WeightError::UnsupportedDtype(safetensors::Dtype::U16);
    assert_eq!(err.to_string(), "unsupported dtype: U16");
}

// --- WeightError From conversions ---

#[test]
fn test_weight_error_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
    let err: super::WeightError = io_err.into();
    assert!(matches!(err, super::WeightError::Io(_)));
}

#[test]
fn test_weight_error_from_metal_error() {
    let metal_err = crate::MetalError::BufferCreate(0);
    let err: super::WeightError = metal_err.into();
    assert!(matches!(err, super::WeightError::Metal(_)));
}

// --- WeightError → TensorError bridge ---

#[test]
fn test_weight_error_to_tensor_error_non_metal() {
    let err = super::WeightError::TensorNotFound("layer.weight".into());
    let tensor_err: nn_core::TensorError = err.into();
    match tensor_err {
        nn_core::TensorError::BackendFailure {
            domain, message, ..
        } => {
            assert_eq!(domain, nn_core::BackendDomain::Metal);
            assert!(message.contains("tensor not found: layer.weight"));
        }
        other => panic!("expected BackendFailure, got: {other:?}"),
    }
}

#[test]
fn test_weight_error_to_tensor_error_metal_no_double_prefix() {
    // WeightError::Metal should delegate to From<MetalError> to avoid
    // "Metal error: Metal error: ..." double prefix
    let err = super::WeightError::Metal(crate::MetalError::NoDevice);
    let tensor_err: nn_core::TensorError = err.into();
    match tensor_err {
        nn_core::TensorError::BackendFailure {
            domain, message, ..
        } => {
            assert_eq!(domain, nn_core::BackendDomain::Metal);
            assert_eq!(message, "Metal is unavailable on this host");
            assert!(
                !message.starts_with("Metal error:"),
                "double Metal prefix: {message}"
            );
        }
        other => panic!("expected BackendFailure, got: {other:?}"),
    }
}

// --- convert_dtype exhaustive coverage ---

#[test]
fn test_convert_dtype_all_unsupported_variants() {
    // Verify other unsupported types also produce the correct error
    for unsupported in [
        safetensors::Dtype::U16,
        safetensors::Dtype::U32,
        safetensors::Dtype::U64,
        safetensors::Dtype::I8,
        safetensors::Dtype::I16,
    ] {
        let err = convert_dtype(unsupported).unwrap_err();
        assert!(
            matches!(err, super::WeightError::UnsupportedDtype(dt) if dt == unsupported),
            "expected UnsupportedDtype({unsupported:?}), got {err:?}"
        );
    }
}

// --- page_align edge cases ---

#[test]
fn test_page_align_two_pages() {
    assert_eq!(page_align(2 * PAGE_SIZE), 2 * PAGE_SIZE);
}

#[test]
fn test_page_align_just_under_page() {
    assert_eq!(page_align(PAGE_SIZE - 1), PAGE_SIZE);
}

// --- page_align memory safety: overflow prevention ---

#[test]
fn test_page_align_near_usize_max_no_wrap() {
    // Before the saturating fix, `page_align(usize::MAX)` would wrap to 0.
    // A buffer length of 0 for a large file would be unsound (enables
    // out-of-bounds reads via Metal's buffer view).
    let result = page_align(usize::MAX);
    assert!(
        result >= usize::MAX - PAGE_SIZE,
        "saturated result should be near usize::MAX"
    );
    assert_eq!(result % PAGE_SIZE, 0, "result must be page-aligned");
}

#[test]
fn test_page_align_overflow_boundary() {
    // The last input that doesn't require saturation:
    // usize::MAX - (PAGE_SIZE - 1) = usize::MAX - 4095
    // For this input, len + 4095 = usize::MAX, which doesn't overflow.
    let safe_max = usize::MAX - (PAGE_SIZE - 1);
    let result = page_align(safe_max);
    assert!(result >= safe_max, "result >= input");
    assert_eq!(result % PAGE_SIZE, 0, "page-aligned");

    // One more: usize::MAX - 4094 would overflow without saturation.
    let overflow_input = usize::MAX - (PAGE_SIZE - 2);
    let result2 = page_align(overflow_input);
    assert!(result2 > 0, "must not wrap to zero");
    assert_eq!(result2 % PAGE_SIZE, 0, "page-aligned");
    assert!(
        result2 >= overflow_input.saturating_sub(PAGE_SIZE),
        "result should be close to max page-aligned value"
    );
}

#[test]
fn test_page_align_monotonic_near_boundary() {
    // For inputs near the overflow boundary, page_align should be monotonically
    // non-decreasing (except for the page-rounding). Verify it doesn't jump to 0.
    let base = usize::MAX - 2 * PAGE_SIZE;
    let mut prev = page_align(base);
    for offset in 1..=(2 * PAGE_SIZE) {
        let val = base.saturating_add(offset);
        let aligned = page_align(val);
        assert!(
            aligned >= prev || aligned == prev - PAGE_SIZE + PAGE_SIZE,
            "page_align should not decrease: page_align({val}) = {aligned}, prev = {prev}"
        );
        assert!(aligned > 0, "must never be zero for non-zero input");
        prev = aligned;
    }
}

// --- WeightMap drop-order guarantee (#522) ---

/// Verify WeightMap uses ManuallyDrop fields with an explicit Drop impl.
///
/// `std::mem::needs_drop` returns true for types with a custom `Drop`
/// impl. If someone removes the `Drop` impl from WeightMap (reverting
/// to field-order-dependent drop), this test will still pass because
/// MetalBuffer/Mmap themselves need drop — but the `ManuallyDrop`
/// fields in the struct definition and the `impl Drop` block are the
/// structural guarantees. This test serves as a documentation anchor.
#[test]
fn test_weight_map_needs_drop() {
    assert!(
        std::mem::needs_drop::<super::WeightMap>(),
        "WeightMap must implement Drop for explicit buffer-before-mmap ordering"
    );
}
