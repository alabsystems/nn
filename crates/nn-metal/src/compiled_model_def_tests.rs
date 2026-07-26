// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for `CompiledModelDef` — the immutable, Arc-shareable model definition.

use std::sync::Arc;

use super::{CompiledModel, CompiledModelDef};

// ── Send + Sync compile-time assertions ─────────────────────────────
//
// `CompiledModelDef` must be `Send + Sync` so that `Arc<CompiledModelDef>`
// can be shared across threads (e.g., chorus voices on different threads).
// These are compile-time checks — if the struct gains a `!Send` or `!Sync`
// field, the build will fail here with a clear error message.

const _: () = {
    const fn assert_send<T: Send>() {}
    const fn assert_sync<T: Sync>() {}
    assert_send::<CompiledModelDef>();
    assert_sync::<CompiledModelDef>();
};

// ── share_def returns the same Arc ──────────────────────────────────

#[test]
fn share_def_returns_same_arc() {
    let model = CompiledModel::empty();
    let arc1 = model.share_def();
    let arc2 = model.share_def();
    // Both Arcs point to the same allocation.
    assert!(Arc::ptr_eq(&arc1, &arc2));
}

// ── from_shared creates independent execution state ─────────────────

#[test]
fn from_shared_creates_independent_execution_state() {
    let model = CompiledModel::empty();
    let shared = model.share_def();
    let instance = CompiledModel::from_shared(Arc::clone(&shared));

    // The definition is shared (same Arc).
    assert!(Arc::ptr_eq(&model.def, &instance.def));

    // But query methods work independently on both.
    assert_eq!(model.num_steps(), instance.num_steps());
    assert_eq!(model.num_inputs(), instance.num_inputs());
    assert_eq!(model.num_steps(), 0);
}

// ── from_shared has its own execution cache ─────────────────────────

#[test]
fn from_shared_has_own_cached_planned_buf() {
    let model = CompiledModel::empty();
    let shared = model.share_def();
    let instance = CompiledModel::from_shared(shared);

    // The cached_planned_buf RefCells are independent: borrowing one
    // does not affect the other.
    let _borrow_original = model.cached_planned_buf.borrow();
    let _borrow_instance = instance.cached_planned_buf.borrow();
    // If they shared the same RefCell, the second borrow would panic.
}
