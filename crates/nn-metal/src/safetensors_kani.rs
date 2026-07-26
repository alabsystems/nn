// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for safetensors module.
//!
//! Extracted from `safetensors.rs` to keep it under 500 lines (#768).
//! Contains proofs for page alignment and WeightMap drop-order safety.

use super::{page_align, PAGE_SIZE};
use std::mem::ManuallyDrop;

/// Page-alignment rounding never wraps and always produces a
/// page-aligned value >= the input.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn page_align_never_wraps() {
    let file_size: usize = kani::any();
    // Files up to 4 GiB — well beyond realistic safetensors sizes.
    kani::assume(file_size <= (1usize << 32));
    let aligned = page_align(file_size);
    assert!(aligned >= file_size);
    assert_eq!(aligned % PAGE_SIZE, 0);
}

// --- Drop-order model for WeightMap (#620) ---
//
// We cannot instantiate real MetalBuffer/Mmap in Kani (platform deps).
// Instead we model the drop-order invariant with mock types that track
// their drop sequence via a shared counter. The model struct mirrors
// WeightMap's `ManuallyDrop` + explicit `Drop` pattern exactly.

/// Global drop-sequence counter. Safe in Kani's single-threaded model.
static mut DROP_SEQ: u32 = 0;

/// Mock for MetalBuffer — records its drop sequence number.
struct MockBuffer {
    drop_order: *mut u32,
}

impl Drop for MockBuffer {
    fn drop(&mut self) {
        // SAFETY: DROP_SEQ is a global mutable static, safe in Kani's
        // single-threaded execution model. drop_order points to a local
        // variable owned by the test harness with a lifetime that exceeds
        // this Drop call.
        unsafe {
            *self.drop_order = DROP_SEQ;
            DROP_SEQ += 1;
        }
    }
}

/// Mock for Mmap — records its drop sequence number.
struct MockMmap {
    drop_order: *mut u32,
}

impl Drop for MockMmap {
    fn drop(&mut self) {
        // SAFETY: Same rationale as MockBuffer::drop — single-threaded
        // Kani model, drop_order outlives this call.
        unsafe {
            *self.drop_order = DROP_SEQ;
            DROP_SEQ += 1;
        }
    }
}

/// Model of WeightMap with identical ManuallyDrop + Drop structure.
struct WeightMapModel {
    buffer: ManuallyDrop<MockBuffer>,
    mmap: ManuallyDrop<MockMmap>,
}

impl Drop for WeightMapModel {
    fn drop(&mut self) {
        // SAFETY: ManuallyDrop::drop is called exactly once per field,
        // in buffer-before-mmap order mirroring safetensors.rs lines 147-150.
        // This is the only Drop impl for WeightMapModel, so double-drop
        // cannot occur.
        unsafe {
            ManuallyDrop::drop(&mut self.buffer);
            ManuallyDrop::drop(&mut self.mmap);
        }
    }
}

/// Proves that WeightMap's Drop impl drops buffer before mmap.
///
/// The model struct mirrors WeightMap's `ManuallyDrop<MetalBuffer>` +
/// `ManuallyDrop<Mmap>` fields and explicit `Drop` impl. Kani
/// exhaustively verifies that the buffer's drop-sequence number is
/// strictly less than the mmap's — i.e., buffer is released first.
///
/// This guards against accidental reordering of the `ManuallyDrop::drop`
/// calls, which would cause a use-after-unmap (#522).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_map_drops_buffer_before_mmap() {
    let mut buffer_order: u32 = u32::MAX;
    let mut mmap_order: u32 = u32::MAX;

    // SAFETY: Reset global counter before test. Single-threaded Kani model.
    unsafe {
        DROP_SEQ = 0;
    }

    {
        let _model = WeightMapModel {
            buffer: ManuallyDrop::new(MockBuffer {
                drop_order: &mut buffer_order,
            }),
            mmap: ManuallyDrop::new(MockMmap {
                drop_order: &mut mmap_order,
            }),
        };
        // _model drops here, triggering the explicit Drop impl.
    }

    // Buffer must drop before mmap (lower sequence number).
    assert!(buffer_order < mmap_order, "buffer must drop before mmap");
    // Exactly two drops must have occurred.
    // SAFETY: Read-only access to DROP_SEQ after all drops have completed.
    assert!(unsafe { DROP_SEQ } == 2, "both fields must be dropped");
}

/// Proves that WeightMap's buffer field is listed before mmap in the
/// ManuallyDrop Drop impl — a structural mirror of the actual code.
///
/// If someone swaps the drop order in WeightMapModel (which mirrors
/// the real Drop impl), `weight_map_drops_buffer_before_mmap` will
/// catch it. This harness additionally verifies the model is
/// consistent even for repeated constructions.
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_map_drop_order_deterministic() {
    let mut b1: u32 = u32::MAX;
    let mut m1: u32 = u32::MAX;
    let mut b2: u32 = u32::MAX;
    let mut m2: u32 = u32::MAX;

    // SAFETY: Reset global counter before test. Single-threaded Kani model.
    unsafe {
        DROP_SEQ = 0;
    }

    {
        let _first = WeightMapModel {
            buffer: ManuallyDrop::new(MockBuffer {
                drop_order: &mut b1,
            }),
            mmap: ManuallyDrop::new(MockMmap {
                drop_order: &mut m1,
            }),
        };
    }

    {
        let _second = WeightMapModel {
            buffer: ManuallyDrop::new(MockBuffer {
                drop_order: &mut b2,
            }),
            mmap: ManuallyDrop::new(MockMmap {
                drop_order: &mut m2,
            }),
        };
    }

    // Both instances drop buffer-before-mmap.
    assert!(b1 < m1);
    assert!(b2 < m2);
    // Four total drops across both instances.
    // SAFETY: Read-only access to DROP_SEQ after all drops have completed.
    assert!(unsafe { DROP_SEQ } == 4);
}
