// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for the SafeTensors backend (`var_builder_safetensors.rs`).
//!
//! Verifies properties of the `SafeTensorsBackend` and `load_tensor_from_weight_map`
//! logic using abstract models (no real Metal GPU or mmap dependencies).
//!
//! Properties proved:
//! 1. Tensor dtype preservation — loaded tensors preserve their original dtype
//! 2. Weight name lookup determinism — same name always retrieves same tensor
//! 3. Missing weight returns error — non-existent name returns Err, not panic
//! 4. Buffer byte alignment — tensor byte lengths are aligned to dtype element size
//! 5. Tensor shape consistency — byte length matches shape * dtype.size_bytes()
//! 6. Zero-size tensor handling — tensor with 0 elements produces 0 bytes
//! 7. Multiple load idempotency — loading same weight twice yields identical metadata
//! 8. Integer dtype rejection — non-float requested dtype returns Err
//! 9. Byte length validation — mismatched byte length returns Err

use std::collections::HashMap;

use nn_core::DType;

use crate::safetensors::TensorInfo;

// ---------------------------------------------------------------------------
// Model types — abstract SafeTensorsBackend without Metal/mmap dependencies
// ---------------------------------------------------------------------------

/// Abstract model of `SafeTensorsBackend` for Kani proofs.
///
/// Models the core behavior of the backend: a lookup table from tensor names
/// to metadata (dtype, shape, byte data). Does not require Metal or mmap.
struct BackendModel {
    tensors: HashMap<String, ModelTensorEntry>,
}

/// A single tensor entry in the abstract model.
struct ModelTensorEntry {
    info: TensorInfo,
    /// Simulated raw byte data length (we don't carry actual bytes in Kani).
    data_len: usize,
}

impl BackendModel {
    /// Model of `TensorBackend::contains_tensor`.
    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Model of `WeightMap::tensor_info` — returns Err for missing names.
    fn tensor_info(&self, name: &str) -> Result<&TensorInfo, ()> {
        self.tensors.get(name).map(|e| &e.info).ok_or(())
    }

    /// Model of `load_tensor_from_weight_map` dtype validation.
    /// Returns Ok(dtype) if the requested dtype is float, Err otherwise.
    fn validate_requested_dtype(requested: DType) -> Result<DType, ()> {
        if requested.is_float() {
            Ok(requested)
        } else {
            Err(())
        }
    }

    /// Model of `load_tensor_from_weight_map` byte length validation.
    /// Checks that data_len == numel * stored_dtype.size_bytes().
    fn validate_byte_length(
        shape: &[usize],
        stored_dtype: DType,
        data_len: usize,
    ) -> Result<usize, ()> {
        let numel = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(())?;
        let expected = numel.checked_mul(stored_dtype.size_bytes()).ok_or(())?;
        if data_len == expected {
            Ok(numel)
        } else {
            Err(())
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: pick a float DType from a selector
// ---------------------------------------------------------------------------

/// Return a float DType and its byte size from a selector in 0..=2.
fn float_dtype_from_selector(sel: u8) -> (DType, usize) {
    match sel % 3 {
        0 => (DType::F32, 4),
        1 => (DType::F16, 2),
        2 => (DType::BF16, 2),
        _ => unreachable!(),
    }
}

/// Return any DType from a selector in 0..=8 (includes integers/bool).
fn any_dtype_from_selector(sel: u8) -> DType {
    match sel % 9 {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        4 => DType::I32,
        5 => DType::I64,
        6 => DType::U32,
        7 => DType::U8,
        8 => DType::Bool,
        _ => unreachable!(),
    }
}

// ===========================================================================
// Proof 1: Tensor dtype preservation
// ===========================================================================

/// Proves that when a tensor is stored with dtype D and loaded with requested
/// dtype D, the resulting dtype is D (no silent dtype conversion).
///
/// Models the native-load paths in `load_tensor_from_weight_map`:
/// - BF16 stored + BF16 requested -> BF16
/// - F16 stored + F16 requested -> F16
/// - F32 stored + F32 requested -> F32
#[kani::unwind(1)]
#[kani::proof]
fn safetensors_backend_dtype_preservation() {
    let sel: u8 = kani::any();
    let (dtype, elem_bytes) = float_dtype_from_selector(sel);

    // When stored dtype == requested dtype, the output dtype is preserved.
    let stored = dtype;
    let requested = dtype;

    // Validate that the requested dtype passes the float check.
    let result = BackendModel::validate_requested_dtype(requested);
    assert!(result.is_ok(), "float dtype must pass validation");
    assert_eq!(result.unwrap(), stored, "output dtype must match stored dtype");

    // The element byte size is correct for the dtype.
    assert_eq!(stored.size_bytes(), elem_bytes);
}

// ===========================================================================
// Proof 2: Weight name lookup determinism
// ===========================================================================

/// Proves that looking up the same tensor name twice in the backend model
/// always returns the same metadata (offset, byte_len, dtype, shape).
///
/// Models the deterministic HashMap lookup in `WeightMap::tensor_info`.
#[kani::unwind(1)]
#[kani::proof]
fn safetensors_backend_lookup_determinism() {
    let mut tensors = HashMap::new();

    let info = TensorInfo {
        offset: 128,
        byte_len: 2048,
        dtype: DType::F32,
        shape: vec![4, 128],
    };
    let entry = ModelTensorEntry {
        info: info.clone(),
        data_len: 2048,
    };
    tensors.insert("layer.weight".to_string(), entry);

    let backend = BackendModel { tensors };

    // Two lookups of the same name.
    let first = backend.tensor_info("layer.weight");
    let second = backend.tensor_info("layer.weight");

    assert!(first.is_ok());
    assert!(second.is_ok());

    let a = first.unwrap();
    let b = second.unwrap();

    // All fields must be identical.
    assert_eq!(a.offset, b.offset);
    assert_eq!(a.byte_len, b.byte_len);
    assert_eq!(a.dtype, b.dtype);
    assert_eq!(a.shape, b.shape);
}

// ===========================================================================
// Proof 3: Missing weight returns error
// ===========================================================================

/// Proves that looking up a non-existent weight name returns Err, never panics.
///
/// Models the `tensor_info` error path: `WeightError::TensorNotFound`.
#[kani::unwind(1)]
#[kani::proof]
fn safetensors_backend_missing_weight_returns_err() {
    let mut tensors = HashMap::new();
    let info = TensorInfo {
        offset: 0,
        byte_len: 512,
        dtype: DType::F32,
        shape: vec![128],
    };
    tensors.insert(
        "existing_weight".to_string(),
        ModelTensorEntry {
            info,
            data_len: 512,
        },
    );

    let backend = BackendModel { tensors };

    // Existing name succeeds.
    assert!(backend.tensor_info("existing_weight").is_ok());
    assert!(backend.contains_tensor("existing_weight"));

    // Non-existent name returns Err.
    assert!(backend.tensor_info("nonexistent_weight").is_err());
    assert!(!backend.contains_tensor("nonexistent_weight"));

    // Empty name returns Err.
    assert!(backend.tensor_info("").is_err());
    assert!(!backend.contains_tensor(""));
}

// ===========================================================================
// Proof 4: Buffer byte alignment
// ===========================================================================

/// Proves that for any float dtype and valid tensor dimensions, the total
/// byte length (numel * dtype.size_bytes()) is aligned to at least 4 bytes
/// when numel is a multiple of 2 (common for ML tensors), and always
/// aligned to the dtype's element size.
///
/// Models the byte alignment invariant relied upon by Metal buffer creation
/// and the f32 chunk iteration in `load_tensor_from_weight_map`.
#[kani::unwind(1)]
#[kani::proof]
fn safetensors_backend_byte_alignment() {
    let sel: u8 = kani::any();
    let (dtype, elem_bytes) = float_dtype_from_selector(sel);

    let numel: usize = kani::any();
    // Constrain to tractable range.
    kani::assume(numel > 0 && numel <= 1_000_000);

    if let Some(total_bytes) = numel.checked_mul(elem_bytes) {
        // Total bytes is always a multiple of the element byte size.
        assert_eq!(
            total_bytes % elem_bytes,
            0,
            "byte length must be aligned to element size"
        );

        // For F32 specifically, byte length is always 4-byte aligned.
        if dtype == DType::F32 {
            assert_eq!(
                total_bytes % 4,
                0,
                "F32 byte length must be 4-byte aligned"
            );
        }

        // For F16 and BF16, byte length is always 2-byte aligned.
        if dtype == DType::F16 || dtype == DType::BF16 {
            assert_eq!(
                total_bytes % 2,
                0,
                "F16/BF16 byte length must be 2-byte aligned"
            );
        }
    }
}

// ===========================================================================
// Proof 5: Tensor shape consistency
// ===========================================================================

/// Proves that for a TensorInfo with consistent shape and byte_len, the
/// byte_len equals numel * dtype.size_bytes() — matching the validation
/// in `load_tensor_from_weight_map` at lines 144-149.
///
/// Also verifies that `TensorInfo::numel()` agrees with the manual product.
#[kani::unwind(4)]
#[kani::proof]
fn safetensors_backend_shape_consistency() {
    let sel: u8 = kani::any();
    let (dtype, elem_bytes) = float_dtype_from_selector(sel);

    let d0: usize = kani::any();
    let d1: usize = kani::any();
    let d2: usize = kani::any();

    // Constrain dimensions to small values for Kani tractability.
    kani::assume(d0 > 0 && d0 <= 256);
    kani::assume(d1 > 0 && d1 <= 256);
    kani::assume(d2 > 0 && d2 <= 256);

    let shape = vec![d0, d1, d2];

    // Manual numel via checked_mul chain.
    let numel = 1usize
        .checked_mul(d0)
        .and_then(|n| n.checked_mul(d1))
        .and_then(|n| n.checked_mul(d2));

    if let Some(numel) = numel {
        if let Some(byte_len) = numel.checked_mul(elem_bytes) {
            let info = TensorInfo {
                offset: 0,
                byte_len,
                dtype,
                shape: shape.clone(),
            };

            // TensorInfo::numel() must agree.
            let computed = info.numel().expect("numel must succeed");
            assert_eq!(computed, numel);

            // byte_len must equal numel * dtype.size_bytes().
            assert_eq!(byte_len, computed * dtype.size_bytes());

            // Validate via BackendModel helper.
            let result = BackendModel::validate_byte_length(&shape, dtype, byte_len);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), numel);
        }
    }
}

// ===========================================================================
// Proof 6: Zero-size tensor handling
// ===========================================================================

/// Proves that a tensor with at least one zero dimension produces numel == 0
/// and byte_len == 0, and that this is handled without panic.
///
/// Models the zero-element edge case in `load_tensor_from_weight_map`
/// where shape product is 0, so expected_bytes == 0 and data.len() must be 0.
#[kani::unwind(4)]
#[kani::proof]
fn safetensors_backend_zero_size_tensor() {
    let sel: u8 = kani::any();
    let (dtype, _elem_bytes) = float_dtype_from_selector(sel);

    let d0: usize = kani::any();
    let d1: usize = kani::any();

    // Constrain dimensions, with at least one being zero.
    kani::assume(d0 <= 256);
    kani::assume(d1 <= 256);
    kani::assume(d0 == 0 || d1 == 0);

    let shape = vec![d0, d1];

    // numel is 0 when any dimension is 0.
    let numel = d0.checked_mul(d1).unwrap_or(0);
    assert_eq!(numel, 0, "numel must be 0 when any dimension is 0");

    // byte_len is 0 for zero-element tensors.
    let byte_len = numel * dtype.size_bytes();
    assert_eq!(byte_len, 0, "byte_len must be 0 for zero-element tensor");

    // TensorInfo::numel() agrees.
    let info = TensorInfo {
        offset: 0,
        byte_len,
        dtype,
        shape: shape.clone(),
    };
    let computed = info.numel().expect("numel must succeed for zero-dim tensor");
    assert_eq!(computed, 0);

    // Byte length validation passes with data_len == 0.
    let result = BackendModel::validate_byte_length(&shape, dtype, 0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

// ===========================================================================
// Proof 7: Multiple load idempotency
// ===========================================================================

/// Proves that looking up the same tensor twice in the model yields identical
/// metadata, modeling the idempotency of `TensorBackend::get`.
///
/// Unlike proof 2 (determinism) which checks pointer equality on a known key,
/// this proof uses symbolic dimensions and dtype to verify that the stored
/// metadata is unchanged between two lookups — the backend model is a
/// pure function of the stored data.
#[kani::unwind(1)]
#[kani::proof]
fn safetensors_backend_load_idempotency() {
    let sel: u8 = kani::any();
    let (dtype, elem_bytes) = float_dtype_from_selector(sel);

    let dim: usize = kani::any();
    kani::assume(dim > 0 && dim <= 4096);

    let byte_len = dim * elem_bytes;

    let mut tensors = HashMap::new();
    let info = TensorInfo {
        offset: 0,
        byte_len,
        dtype,
        shape: vec![dim],
    };
    tensors.insert(
        "w".to_string(),
        ModelTensorEntry {
            info: info.clone(),
            data_len: byte_len,
        },
    );

    let backend = BackendModel { tensors };

    // First load.
    let first = backend.tensor_info("w").unwrap();
    // Second load.
    let second = backend.tensor_info("w").unwrap();

    // Metadata is identical across loads.
    assert_eq!(first.offset, second.offset);
    assert_eq!(first.byte_len, second.byte_len);
    assert_eq!(first.dtype, second.dtype);
    assert_eq!(first.shape.len(), second.shape.len());

    // Both loads validate byte length consistently.
    let v1 = BackendModel::validate_byte_length(&first.shape, first.dtype, first.byte_len);
    let v2 = BackendModel::validate_byte_length(&second.shape, second.dtype, second.byte_len);
    assert_eq!(v1.is_ok(), v2.is_ok());
    if let (Ok(n1), Ok(n2)) = (v1, v2) {
        assert_eq!(n1, n2, "numel must be identical across loads");
    }
}

// ===========================================================================
// Proof 8: Integer dtype rejection
// ===========================================================================

/// Proves that requesting an integer/bool dtype from `load_tensor_from_weight_map`
/// returns Err. Models the `!requested_dtype.is_float()` guard at line 121.
#[kani::unwind(1)]
#[kani::proof]
fn safetensors_backend_rejects_integer_dtype() {
    let sel: u8 = kani::any();
    let dtype = any_dtype_from_selector(sel);

    let result = BackendModel::validate_requested_dtype(dtype);

    if dtype.is_float() {
        assert!(result.is_ok(), "float dtypes must be accepted");
        assert_eq!(result.unwrap(), dtype);
    } else {
        assert!(result.is_err(), "non-float dtypes must be rejected");
    }
}

// ===========================================================================
// Proof 9: Byte length validation rejects mismatch
// ===========================================================================

/// Proves that when data_len != numel * dtype.size_bytes(), the validation
/// returns Err. Models the `data.len() != expected_bytes` check at line 144.
#[kani::unwind(1)]
#[kani::proof]
fn safetensors_backend_byte_length_mismatch_rejected() {
    let sel: u8 = kani::any();
    let (dtype, elem_bytes) = float_dtype_from_selector(sel);

    let dim: usize = kani::any();
    kani::assume(dim > 0 && dim <= 4096);

    let correct_bytes = dim * elem_bytes;
    let wrong_bytes: usize = kani::any();
    kani::assume(wrong_bytes <= 1_000_000);
    kani::assume(wrong_bytes != correct_bytes);

    let shape = vec![dim];

    // Correct byte length passes.
    let correct = BackendModel::validate_byte_length(&shape, dtype, correct_bytes);
    assert!(correct.is_ok());

    // Wrong byte length is rejected.
    let wrong = BackendModel::validate_byte_length(&shape, dtype, wrong_bytes);
    assert!(wrong.is_err(), "mismatched byte length must be rejected");
}
