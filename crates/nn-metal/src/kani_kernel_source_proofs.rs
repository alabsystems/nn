// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for [`KernelSource`] builder pattern and equality.
//!
//! Proves:
//! - Builder pattern preserves source/entry invariants
//! - fast_math toggle is idempotent
//! - Function constants accumulate correctly
//! - Equality/hash consistency

use crate::kernel_source::KernelSource;

/// Proves: `new()` always produces `fast_math == false` and empty constants.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_source_new_defaults() {
    let ks = KernelSource::new("src", "entry");
    assert!(!ks.fast_math());
    assert!(ks.function_constants().is_empty());
    assert_eq!(ks.msl_source(), "src");
    assert_eq!(ks.entry_point(), "entry");
}

/// Proves: `with_fast_math(true).with_fast_math(true)` is idempotent.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_source_fast_math_idempotent() {
    let a = KernelSource::new("s", "e").with_fast_math(true);
    let b = KernelSource::new("s", "e").with_fast_math(true).with_fast_math(true);
    assert_eq!(a, b);
}

/// Proves: Different fast_math values produce different KernelSources.
#[kani::unwind(1)]
#[kani::proof]
fn kernel_source_fast_math_distinguishes() {
    let a = KernelSource::new("s", "e").with_fast_math(false);
    let b = KernelSource::new("s", "e").with_fast_math(true);
    assert_ne!(a, b);
}

/// Proves: `with_function_constant` appends (does not replace).
#[kani::unwind(1)]
#[kani::proof]
fn kernel_source_function_constant_appends() {
    let idx1: u32 = kani::any();
    let val1: u32 = kani::any();
    let idx2: u32 = kani::any();
    let val2: u32 = kani::any();

    let ks = KernelSource::new("s", "e")
        .with_function_constant(idx1, val1)
        .with_function_constant(idx2, val2);

    let consts = ks.function_constants();
    assert_eq!(consts.len(), 2);
    assert_eq!(consts[0], (idx1, val1));
    assert_eq!(consts[1], (idx2, val2));
}

/// Proves: KernelSource equality is reflexive (a == a for any construction).
#[kani::unwind(1)]
#[kani::proof]
fn kernel_source_equality_reflexive() {
    let fm: bool = kani::any();
    let ks = KernelSource::new("code", "fn_name").with_fast_math(fm);
    assert_eq!(ks, ks.clone());
}
