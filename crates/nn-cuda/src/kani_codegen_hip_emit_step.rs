// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for HIP codegen emit step helpers.
//!
//! Proves properties of type mapping (`hip_type`, `hip_accumulator_type`),
//! value validation (`safe_hip_uint`), float formatting (`format_float`),
//! and MXFP4 parameter validation. These prove correctness of the codegen
//! helper layer that all emit functions depend on.
//!
//! Part of #3719.

use super::codegen_hip::{
    format_float, hip_accumulator_type, hip_type, safe_hip_uint, HIP_BLOCK_SIZE, REDUCE_BLOCK_SIZE,
};
use super::codegen_hip_mxfp4::{
    mxfp4_num_scales, mxfp4_packed_bytes, MXFP4_BLOCK_BYTES, MXFP4_BLOCK_SIZE,
};
use nn_dsl::ScalarType;

// =========================================================================
// hip_type mapping proofs
// =========================================================================

/// Prove hip_type returns Ok for all supported scalar types.
#[kani::unwind(1)]
#[kani::proof]
fn prove_hip_type_f32_ok() {
    let result = hip_type(ScalarType::F32);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "float");
}

#[kani::unwind(1)]
#[kani::proof]
fn prove_hip_type_f16_ok() {
    let result = hip_type(ScalarType::F16);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "half");
}

#[kani::unwind(1)]
#[kani::proof]
fn prove_hip_type_bf16_ok() {
    let result = hip_type(ScalarType::BF16);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "hip_bfloat16");
}

// =========================================================================
// hip_accumulator_type proofs
// =========================================================================

/// Prove accumulator type is always "float" regardless of input dtype.
#[kani::unwind(1)]
#[kani::proof]
fn prove_accumulator_always_float_f32() {
    assert_eq!(hip_accumulator_type(ScalarType::F32), "float");
}

#[kani::unwind(1)]
#[kani::proof]
fn prove_accumulator_always_float_f16() {
    assert_eq!(hip_accumulator_type(ScalarType::F16), "float");
}

#[kani::unwind(1)]
#[kani::proof]
fn prove_accumulator_always_float_bf16() {
    assert_eq!(hip_accumulator_type(ScalarType::BF16), "float");
}

// =========================================================================
// safe_hip_uint proofs
// =========================================================================

/// Prove safe_hip_uint accepts all values <= u32::MAX.
#[kani::unwind(1)]
#[kani::proof]
fn prove_safe_hip_uint_valid_range() {
    let val: u32 = kani::any();
    let result = safe_hip_uint(val as usize);
    assert!(result.is_ok());
}

/// Prove safe_hip_uint rejects values > u32::MAX.
#[kani::unwind(1)]
#[kani::proof]
fn prove_safe_hip_uint_overflow() {
    let val: usize = kani::any();
    kani::assume(val > u32::MAX as usize);
    let result = safe_hip_uint(val);
    assert!(result.is_err());
}

/// Prove safe_hip_uint(0) returns "0".
#[kani::unwind(1)]
#[kani::proof]
fn prove_safe_hip_uint_zero() {
    let result = safe_hip_uint(0);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "0");
}

/// Prove safe_hip_uint(u32::MAX) succeeds.
#[kani::unwind(1)]
#[kani::proof]
fn prove_safe_hip_uint_max() {
    let result = safe_hip_uint(u32::MAX as usize);
    assert!(result.is_ok());
}

/// Prove safe_hip_uint output string parses back to the original value.
#[kani::unwind(1)]
#[kani::proof]
fn prove_safe_hip_uint_roundtrip() {
    let val: u16 = kani::any();
    let result = safe_hip_uint(val as usize).unwrap();
    let parsed: usize = result.parse().unwrap();
    assert_eq!(parsed, val as usize);
}

// =========================================================================
// format_float proofs
// =========================================================================

/// Prove format_float(INFINITY) returns the HIP infinity literal.
#[kani::unwind(1)]
#[kani::proof]
fn prove_format_float_inf() {
    let s = format_float(f32::INFINITY);
    assert_eq!(s, "HUGE_VALF");
}

/// Prove format_float(NEG_INFINITY) returns the HIP neg-infinity literal.
#[kani::unwind(1)]
#[kani::proof]
fn prove_format_float_neg_inf() {
    let s = format_float(f32::NEG_INFINITY);
    assert_eq!(s, "(-HUGE_VALF)");
}

/// Prove format_float(NaN) returns the HIP NaN literal.
#[kani::unwind(1)]
#[kani::proof]
fn prove_format_float_nan() {
    let s = format_float(f32::NAN);
    assert_eq!(s, "nanf(\"\")");
}

/// Prove format_float for a normal value is non-empty and does not
/// contain HIP special literals.
#[kani::unwind(1)]
#[kani::proof]
fn prove_format_float_normal_no_special() {
    let v: f32 = kani::any();
    kani::assume(v.is_finite());
    let s = format_float(v);
    assert!(!s.is_empty());
    assert!(s != "HUGE_VALF");
    assert!(s != "(-HUGE_VALF)");
    assert!(s != "nanf(\"\")");
}

// =========================================================================
// MXFP4 parameter validation proofs
// =========================================================================

/// Prove MXFP4 block constants are consistent.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_block_constants() {
    assert_eq!(MXFP4_BLOCK_SIZE, 32);
    assert_eq!(MXFP4_BLOCK_BYTES, MXFP4_BLOCK_SIZE / 2);
    assert_eq!(MXFP4_BLOCK_BYTES, 16);
}

/// Prove mxfp4_packed_bytes rejects odd element counts.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_packed_bytes_rejects_odd() {
    let n: u16 = kani::any();
    kani::assume(n % 2 != 0);
    let result = mxfp4_packed_bytes(n as usize);
    assert!(result.is_err());
}

/// Prove mxfp4_packed_bytes returns n/2 for even inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_packed_bytes_even() {
    let half: u16 = kani::any();
    let n = (half as usize) * 2;
    let result = mxfp4_packed_bytes(n);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), half as usize);
}

/// Prove mxfp4_num_scales rejects non-multiples of 32.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_num_scales_rejects_unaligned() {
    let n: u16 = kani::any();
    kani::assume(n as usize % MXFP4_BLOCK_SIZE != 0);
    let result = mxfp4_num_scales(n as usize);
    assert!(result.is_err());
}

/// Prove mxfp4_num_scales returns n/32 for aligned inputs.
#[kani::unwind(1)]
#[kani::proof]
fn prove_mxfp4_num_scales_aligned() {
    let blocks: u8 = kani::any();
    kani::assume(blocks > 0);
    let n = (blocks as usize) * MXFP4_BLOCK_SIZE;
    let result = mxfp4_num_scales(n);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), blocks as usize);
}

// =========================================================================
// Codegen type invariant proofs
// =========================================================================

/// Prove that f16 and bf16 types are distinct (HIP supports native bf16).
#[kani::unwind(1)]
#[kani::proof]
fn prove_f16_bf16_distinct() {
    let f16_t = hip_type(ScalarType::F16).unwrap();
    let bf16_t = hip_type(ScalarType::BF16).unwrap();
    assert!(f16_t != bf16_t);
}

/// Prove accumulator != type for f16 (requires cast in generated code).
#[kani::unwind(1)]
#[kani::proof]
fn prove_f16_needs_cast() {
    let t = hip_type(ScalarType::F16).unwrap();
    let acc = hip_accumulator_type(ScalarType::F16);
    assert!(t != acc);
}

/// Prove accumulator == type for f32 (no cast needed).
#[kani::unwind(1)]
#[kani::proof]
fn prove_f32_no_cast() {
    let t = hip_type(ScalarType::F32).unwrap();
    let acc = hip_accumulator_type(ScalarType::F32);
    assert_eq!(t, acc);
}

/// Prove accumulator != type for bf16 (requires cast in generated code).
#[kani::unwind(1)]
#[kani::proof]
fn prove_bf16_needs_cast() {
    let t = hip_type(ScalarType::BF16).unwrap();
    let acc = hip_accumulator_type(ScalarType::BF16);
    assert!(t != acc);
}

// =========================================================================
// Block size constant proofs
// =========================================================================

/// Prove HIP_BLOCK_SIZE and REDUCE_BLOCK_SIZE are both 256.
#[kani::unwind(1)]
#[kani::proof]
fn prove_block_size_values() {
    assert_eq!(HIP_BLOCK_SIZE, 256);
    assert_eq!(REDUCE_BLOCK_SIZE, 256);
}

/// Prove block sizes divide evenly into 1024 (max HIP block size).
#[kani::unwind(1)]
#[kani::proof]
fn prove_block_sizes_divide_max() {
    assert!(1024 % HIP_BLOCK_SIZE == 0);
    assert!(1024 % REDUCE_BLOCK_SIZE == 0);
}
