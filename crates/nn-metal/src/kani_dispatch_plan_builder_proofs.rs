// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`DispatchPlan`] builder pattern preservation.
//!
//! Proves:
//! - Builder methods preserve unrelated fields
//! - with_output_elems correctly overrides output_elems
//! - with_use_threadgroups correctly overrides mode
//! - with_threadgroup_memory_bytes correctly overrides shared memory
//! - Elementwise plan builder chain preserves grid/threads

use crate::dispatch_plan::*;

/// Proves: with_output_elems overrides only output_elems, preserving grid.
#[kani::unwind(1)]
#[kani::proof]
fn dispatch_plan_with_output_elems_preserves_grid() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let plan = plan_elementwise(total).unwrap();
    let original_grid = plan.grid();
    let original_threads = plan.threads();

    let new_elems: usize = kani::any();
    kani::assume(new_elems <= 1_000_000);
    let modified = plan.with_output_elems(new_elems);

    assert_eq!(modified.output_elems(), new_elems);
    assert_eq!(modified.grid(), original_grid);
    assert_eq!(modified.threads(), original_threads);
}

/// Proves: with_use_threadgroups overrides dispatch mode, preserves output_elems.
#[kani::unwind(1)]
#[kani::proof]
fn dispatch_plan_with_use_threadgroups_preserves_output() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let plan = plan_elementwise(total).unwrap();
    let original_elems = plan.output_elems();

    let mode: bool = kani::any();
    let modified = plan.with_use_threadgroups(mode);

    assert_eq!(modified.use_threadgroups(), mode);
    assert_eq!(modified.output_elems(), original_elems);
}

/// Proves: with_threadgroup_memory_bytes overrides shared memory, preserves grid.
#[kani::unwind(1)]
#[kani::proof]
fn dispatch_plan_with_threadgroup_memory_preserves_grid() {
    let outer: u32 = kani::any();
    let reduce: u32 = kani::any();
    let threads: u32 = kani::any();
    let shared: u32 = kani::any();
    kani::assume(outer > 0 && reduce > 0 && threads > 0);

    let plan = plan_reduction(outer, reduce, threads, shared).unwrap();
    let original_grid = plan.grid();

    let new_bytes: u64 = kani::any();
    kani::assume(new_bytes <= 65536);
    let modified = plan.with_threadgroup_memory_bytes(Some(new_bytes));

    assert_eq!(modified.threadgroup_memory_bytes(), Some(new_bytes));
    assert_eq!(modified.grid(), original_grid);
}

/// Proves: with_constants overrides constants, preserves output_elems.
#[kani::unwind(1)]
#[kani::proof]
fn dispatch_plan_with_constants_preserves_output() {
    let total: u32 = kani::any();
    kani::assume(total > 0);

    let plan = plan_elementwise(total).unwrap();
    let original_elems = plan.output_elems();

    let c: u32 = kani::any();
    let modified = plan.with_constants(vec![c]);

    assert_eq!(modified.constants(), &[c]);
    assert_eq!(modified.output_elems(), original_elems);
}
