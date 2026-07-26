// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for `build_sigmoid_kernel`.
//!
//! Proves the builder does not panic.
//!
//! Part of #752 AC2.

// ---------------------------------------------------------------------------
// Deterministic structural tests (converted from tautological Kani harnesses)
// ---------------------------------------------------------------------------
//
// `build_sigmoid_kernel` has no parameters — exactly one configuration.
// Kani exhaustive model-checking adds nothing over a normal #[test] here.

#[cfg(test)]
mod tests {
    use super::build_sigmoid_kernel;
    use crate::ir::ScalarType;

    #[test]
    fn sigmoid_build_no_panic() {
        let def = build_sigmoid_kernel().expect("sigmoid kernel build must succeed");
        assert!(!def.name.is_empty(), "kernel name must not be empty");
    }

    #[test]
    fn sigmoid_param_count() {
        let def = build_sigmoid_kernel().expect("build must succeed");
        assert_eq!(def.params.len(), 1);
    }

    #[test]
    fn sigmoid_return_type() {
        let def = build_sigmoid_kernel().expect("build must succeed");
        assert_eq!(def.return_type, ScalarType::F32);
    }
}
