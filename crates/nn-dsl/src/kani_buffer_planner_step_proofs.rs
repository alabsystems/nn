// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `buffer_planner_bytes.rs` — the step-level
//! byte-size computation functions [`step_output_bytes`] and
//! [`step_output_bytes_typed`].
//!
//! Proves:
//! - InputForward, IdentityPassthrough, Passthrough, NarrowView, and
//!   RuntimeOp always return 0 bytes (no allocation).
//! - ConstantValue returns product(shape) * 4 bytes.
//! - step_output_bytes_typed with F16 returns half the F32 byte count.
//! - step_output_bytes_typed with BF16 returns half the F32 byte count.
//! - step_output_bytes_typed with None dtype matches step_output_bytes.
//! - ConstantValue with huge shape dimensions returns 0 (overflow
//!   protection via checked_mul), not a panic.

use super::bytes::{step_output_bytes, step_output_bytes_typed};
use crate::ir::ScalarType;
use crate::trace_compile::CompiledStep;

// ============================================================================
// Zero-byte alias steps
// ============================================================================

/// Proves: InputForward always returns 0 bytes.
///
/// SUBSTANTIVE: InputForward aliases an external input buffer — no new
/// allocation. If this returned non-zero, the planner would waste GPU
/// memory on a redundant copy.
#[kani::proof]
#[kani::unwind(1)]
fn proof_input_forward_zero_bytes() {
    assert_eq!(step_output_bytes(&CompiledStep::InputForward), 0);
}

/// Proves: IdentityPassthrough always returns 0 bytes.
///
/// SUBSTANTIVE: IdentityPassthrough aliases an existing buffer (e.g.,
/// Dropout at inference). Non-zero would cause double allocation.
#[kani::proof]
#[kani::unwind(1)]
fn proof_identity_passthrough_zero_bytes() {
    assert_eq!(step_output_bytes(&CompiledStep::IdentityPassthrough), 0);
}

/// Proves: Passthrough always returns 0 bytes.
///
/// SUBSTANTIVE: Passthrough (reshape/unsqueeze/squeeze) is metadata-only.
/// The underlying data buffer is shared with the input.
#[kani::proof]
#[kani::unwind(1)]
fn proof_passthrough_zero_bytes() {
    let step = CompiledStep::Passthrough {
        op_name: String::from("reshape"),
        output_shape: vec![2, 3, 4],
    };
    assert_eq!(step_output_bytes(&step), 0);
}

/// Proves: NarrowView always returns 0 bytes.
///
/// SUBSTANTIVE: NarrowView is a zero-copy byte-offset view into the
/// input buffer. If this allocated, it would defeat the zero-copy
/// narrow optimization from peephole pass 12 (BatchedLinearProjection).
#[kani::proof]
#[kani::unwind(1)]
fn proof_narrow_view_zero_bytes() {
    let step = CompiledStep::NarrowView {
        byte_offset: 128,
        output_shape: vec![1, 64],
        source_step: Some(5),
    };
    assert_eq!(step_output_bytes(&step), 0);
}

/// Proves: RuntimeOp always returns 0 bytes.
///
/// SUBSTANTIVE: RuntimeOp has data-dependent output shape that cannot
/// be pre-planned. The executor allocates at runtime. If the planner
/// returned non-zero, it would be a guess that wastes memory or is
/// too small.
#[kani::proof]
#[kani::unwind(1)]
fn proof_runtime_op_zero_bytes() {
    use crate::trace_compile::RuntimeOpKind;
    let step = CompiledStep::RuntimeOp {
        op: RuntimeOpKind::RepeatInterleave {
            dim: 1,
            input_shape: vec![1, 64],
            counts_shape: vec![1, 10],
        },
    };
    assert_eq!(step_output_bytes(&step), 0);
}

// ============================================================================
// ConstantValue byte computation
// ============================================================================

/// Proves: ConstantValue bytes = product(shape) * 4 for small shapes.
///
/// SUBSTANTIVE: The buffer planner allocates exactly product(shape) * 4
/// bytes for constant fills. Under- or over-allocation causes either
/// out-of-bounds GPU access or wasted memory.
#[kani::proof]
#[kani::unwind(8)]
fn proof_constant_value_bytes() {
    let d1: usize = kani::any();
    kani::assume(d1 > 0 && d1 <= 64);
    let d2: usize = kani::any();
    kani::assume(d2 > 0 && d2 <= 64);
    let step = CompiledStep::ConstantValue {
        value: 1.0,
        shape: vec![d1, d2],
    };
    let bytes = step_output_bytes(&step);
    assert_eq!(bytes, d1 * d2 * 4);
}

// ============================================================================
// Typed byte computation (F16/BF16/None)
// ============================================================================

/// Proves: step_output_bytes_typed with F16 returns half the F32 byte count.
///
/// SUBSTANTIVE: Mixed-precision executors store ConstantValue output in
/// F16 (2 bytes/elem) instead of F32 (4 bytes/elem). The planner must
/// allocate the correct half-size buffer.
#[kani::proof]
#[kani::unwind(8)]
fn proof_constant_value_f16_half_bytes() {
    let d1: usize = kani::any();
    kani::assume(d1 > 0 && d1 <= 64);
    let d2: usize = kani::any();
    kani::assume(d2 > 0 && d2 <= 64);
    let step = CompiledStep::ConstantValue {
        value: 0.5,
        shape: vec![d1, d2],
    };
    let f32_bytes = step_output_bytes(&step);
    let f16_bytes = step_output_bytes_typed(&step, Some(ScalarType::F16));
    assert_eq!(f16_bytes * 2, f32_bytes);
}

/// Proves: step_output_bytes_typed with BF16 returns half the F32 byte count.
///
/// SUBSTANTIVE: BF16 has the same byte width as F16 (2 bytes/elem).
/// The planner must treat BF16 identically to F16 for allocation sizing.
#[kani::proof]
#[kani::unwind(8)]
fn proof_constant_value_bf16_half_bytes() {
    let d1: usize = kani::any();
    kani::assume(d1 > 0 && d1 <= 64);
    let d2: usize = kani::any();
    kani::assume(d2 > 0 && d2 <= 64);
    let step = CompiledStep::ConstantValue {
        value: -1.0,
        shape: vec![d1, d2],
    };
    let f32_bytes = step_output_bytes(&step);
    let bf16_bytes = step_output_bytes_typed(&step, Some(ScalarType::BF16));
    assert_eq!(bf16_bytes * 2, f32_bytes);
}

/// Proves: step_output_bytes_typed with None dtype matches step_output_bytes.
///
/// SUBSTANTIVE: None dtype falls back to F32 (4 bytes/elem), which is
/// the same as step_output_bytes. If these diverged, the non-typed and
/// typed buffer plans would disagree on allocation sizes.
#[kani::proof]
#[kani::unwind(8)]
fn proof_typed_none_matches_untyped() {
    let d1: usize = kani::any();
    kani::assume(d1 > 0 && d1 <= 64);
    let d2: usize = kani::any();
    kani::assume(d2 > 0 && d2 <= 64);
    let step = CompiledStep::ConstantValue {
        value: 42.0,
        shape: vec![d1, d2],
    };
    let untyped = step_output_bytes(&step);
    let typed_none = step_output_bytes_typed(&step, None);
    assert_eq!(untyped, typed_none);
}

// ============================================================================
// Overflow protection
// ============================================================================

/// Proves: ConstantValue with huge shape dimensions returns 0 (overflow
/// protection), not a panic or a wrapped value.
///
/// SUBSTANTIVE: checked_mul in checked_shape_bytes returns 0 on overflow.
/// Without this, a malicious or buggy trace with enormous shapes would
/// either panic (release: wrapping overflow) or silently produce a
/// small allocation that the GPU kernel writes past.
#[kani::proof]
#[kani::unwind(8)]
fn proof_constant_value_overflow_returns_zero() {
    let step = CompiledStep::ConstantValue {
        value: 1.0,
        shape: vec![usize::MAX, usize::MAX],
    };
    let bytes = step_output_bytes(&step);
    assert_eq!(bytes, 0, "overflow must produce 0, not panic or wrap");
}
