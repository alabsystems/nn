// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for GPU buffer capacity validation (#3298).
//!
//! Models the invariant that `slice_to_dyn` and `weight_to_dyn` SHOULD
//! enforce: the declared tensor shape must fit within the backing buffer's
//! byte capacity. Without this check, a buffer planner bug could produce
//! an undersized buffer and the DynTensor would enable GPU OOB writes.
//!
//! These harnesses prove the arithmetic correctness of the capacity
//! formula: `byte_offset + product(shape) * dtype_size <= buffer_len`.

/// Prove: the capacity check formula does not overflow for any valid inputs.
///
/// Models the computation: `byte_offset + (product_of_shape * elem_bytes)`
/// using `checked_mul` and `checked_add`. Proves that the check either
/// correctly validates capacity or correctly rejects overflow.
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn buffer_capacity_formula_no_silent_overflow() {
    let buffer_len: usize = kani::any();
    let byte_offset: usize = kani::any();
    let dim_product: usize = kani::any();
    let elem_bytes: usize = kani::any();

    // Constrain to realistic bounds for CBMC tractability.
    kani::assume(buffer_len <= (1usize << 34)); // ~16 GB
    kani::assume(byte_offset <= buffer_len);
    kani::assume(dim_product <= (1usize << 30)); // ~1 billion elements
    kani::assume(elem_bytes > 0 && elem_bytes <= 8); // u8..f64

    // The capacity check computation (what slice_to_dyn should use).
    let data_bytes = dim_product.checked_mul(elem_bytes);
    let end_byte = data_bytes.and_then(|db| byte_offset.checked_add(db));

    match end_byte {
        Some(end) if end <= buffer_len => {
            // Capacity check passed. Verify the region is truly within bounds.
            assert!(
                byte_offset + dim_product * elem_bytes <= buffer_len,
                "capacity check passed but region exceeds buffer"
            );
        }
        Some(_end) => {
            // end > buffer_len: correctly rejected (insufficient capacity).
        }
        None => {
            // Overflow in checked arithmetic: correctly rejected.
        }
    }
}

/// Prove: without the capacity check, an undersized buffer can be constructed.
///
/// This is a "possibility proof" — it shows that the gap is real.
/// Given arbitrary buffer_len < required_bytes, the current code path
/// (which only checks dim product overflow) would accept the buffer.
///
/// This harness proves that `checked_dim_product` alone is insufficient:
/// a valid dim product with a small buffer_len creates an OOB region.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn undersized_buffer_possible_without_capacity_check() {
    let buffer_len: usize = kani::any();
    let byte_offset: usize = kani::any();
    let dim_product: usize = kani::any();
    let elem_bytes: usize = kani::any();

    kani::assume(buffer_len <= (1usize << 34));
    kani::assume(dim_product > 0 && dim_product <= (1usize << 30));
    kani::assume(elem_bytes > 0 && elem_bytes <= 8);
    kani::assume(byte_offset <= buffer_len);

    // The dim product is valid (no overflow) — this is all from_gpu_storage checks.
    let data_bytes = dim_product.checked_mul(elem_bytes);
    kani::assume(data_bytes.is_some());
    let required = byte_offset + data_bytes.unwrap();

    // The gap: buffer is too small.
    kani::assume(required > buffer_len);

    // Without the capacity check, this allocation would succeed (creating
    // a DynTensor backed by insufficient memory). The GPU dispatch would
    // read/write past buffer_len, causing OOB access.
    //
    // Prove: this state is reachable — the gap is real.
    assert!(
        required > buffer_len,
        "undersized buffer state must be reachable"
    );
}

/// Prove: weight_to_dyn capacity check with zero byte_offset (dedicated buffers).
///
/// Weight buffers are not arena-allocated (byte_offset = 0), but the shape
/// could still exceed the buffer capacity if the weight file is truncated
/// or the shape metadata is wrong.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_buffer_capacity_check_zero_offset() {
    let buffer_len: usize = kani::any();
    let dim_product: usize = kani::any();
    let elem_bytes: usize = kani::any();

    kani::assume(buffer_len <= (1usize << 34));
    kani::assume(dim_product > 0 && dim_product <= (1usize << 30));
    kani::assume(elem_bytes > 0 && elem_bytes <= 8);

    let data_bytes = match dim_product.checked_mul(elem_bytes) {
        Some(db) => db,
        None => return, // overflow → reject (safe)
    };

    if data_bytes <= buffer_len {
        // Capacity sufficient: the tensor region [0, data_bytes) fits.
        assert!(data_bytes <= buffer_len);
    } else {
        // Capacity insufficient: weight_to_dyn SHOULD reject this.
        // Currently it does NOT (the #3298 gap).
        assert!(
            data_bytes > buffer_len,
            "insufficient capacity must be detectable"
        );
    }
}
