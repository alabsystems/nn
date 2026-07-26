// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`KernelBinding`] constant encoding.
//!
//! Proves:
//! - constant_u32 encodes exactly 4 bytes matching bytemuck::bytes_of
//! - constant_f32 encodes exactly 4 bytes matching bytemuck::bytes_of
//! - u32 and f32 constant values round-trip through encoding

use crate::compiled_model::kernel_spec::KernelBinding;

/// Proves: constant_u32 produces exactly 4 bytes for any u32 value.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_binding_constant_u32_size() {
    let val: u32 = kani::any();
    let binding = KernelBinding::constant_u32(val);
    if let KernelBinding::Constant(bytes) = binding {
        assert_eq!(bytes.len(), 4);
    } else {
        panic!("constant_u32 must produce Constant variant");
    }
}

/// Proves: constant_u32 encodes the value as little-endian bytes
/// (matching bytemuck::bytes_of which is the native representation).
#[kani::unwind(1)]
#[kani::proof]
fn kernel_binding_constant_u32_roundtrip() {
    let val: u32 = kani::any();
    let binding = KernelBinding::constant_u32(val);
    if let KernelBinding::Constant(bytes) = binding {
        let arr: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let decoded = u32::from_ne_bytes(arr);
        assert_eq!(decoded, val, "u32 constant must round-trip");
    } else {
        panic!("constant_u32 must produce Constant variant");
    }
}

/// Proves: constant_f32 produces exactly 4 bytes for any f32 value.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_binding_constant_f32_size() {
    let val: f32 = kani::any();
    let binding = KernelBinding::constant_f32(val);
    if let KernelBinding::Constant(bytes) = binding {
        assert_eq!(bytes.len(), 4);
    } else {
        panic!("constant_f32 must produce Constant variant");
    }
}

/// Proves: constant_f32 encodes the value as native-endian bytes that
/// round-trip through f32::from_ne_bytes, preserving bit pattern.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_binding_constant_f32_bits_roundtrip() {
    let val: f32 = kani::any();
    let binding = KernelBinding::constant_f32(val);
    if let KernelBinding::Constant(bytes) = binding {
        let arr: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        let decoded = f32::from_ne_bytes(arr);
        // Compare bit patterns because NaN != NaN.
        assert_eq!(decoded.to_bits(), val.to_bits(), "f32 constant must preserve bit pattern");
    } else {
        panic!("constant_f32 must produce Constant variant");
    }
}
