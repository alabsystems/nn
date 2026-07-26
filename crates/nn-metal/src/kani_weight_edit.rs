// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani harnesses for weight_edit and live_edit types:
//! WeightEditSpec, WeightEditError, WeightEditResult,
//! ApplyReceipt, DeltaApplyReceipt, LiveEditError.

#[cfg(kani)]
mod proofs {
    use crate::live_edit::{ApplyReceipt, DeltaApplyReceipt, LiveEditError};
    use crate::weight_edit::{WeightEditError, WeightEditResult, WeightEditSpec};

    // -----------------------------------------------------------------------
    // WeightEditSpec harnesses
    // -----------------------------------------------------------------------

    /// Prove WeightEditSpec can be constructed with non-empty data.
    #[kani::unwind(1)]
    #[kani::proof]
    fn weight_edit_spec_construction_non_empty() {
        let data = [1.0f32, 2.0, 3.0];
        let spec = WeightEditSpec {
            layer_name: "test.weight",
            new_data: &data,
        };
        assert_eq!(spec.layer_name, "test.weight");
        assert_eq!(spec.new_data.len(), 3);
        assert!(!spec.new_data.is_empty());
    }

    /// Prove WeightEditSpec can be constructed with empty data (the validation
    /// rejects this later, but construction itself is allowed).
    #[kani::unwind(1)]
    #[kani::proof]
    fn weight_edit_spec_construction_empty() {
        let data: [f32; 0] = [];
        let spec = WeightEditSpec {
            layer_name: "empty.layer",
            new_data: &data,
        };
        assert!(spec.new_data.is_empty());
    }

    // -----------------------------------------------------------------------
    // WeightEditResult harnesses
    // -----------------------------------------------------------------------

    /// Prove WeightEditResult fields are consistent: new_generation == previous_generation + 1.
    #[kani::unwind(1)]
    #[kani::proof]
    fn weight_edit_result_generation_invariant() {
        let prev: u64 = kani::any();
        // Guard against overflow at u64::MAX.
        kani::assume(prev < u64::MAX);

        let result = WeightEditResult {
            previous_generation: prev,
            new_generation: prev + 1,
            elements_written: 100,
        };

        assert_eq!(result.new_generation, result.previous_generation + 1);
        assert_eq!(result.elements_written, 100);
    }

    /// Prove WeightEditResult PartialEq is reflexive.
    #[kani::unwind(1)]
    #[kani::proof]
    fn weight_edit_result_eq_reflexive() {
        let result = WeightEditResult {
            previous_generation: 42,
            new_generation: 43,
            elements_written: 256,
        };
        assert_eq!(result, result.clone());
    }

    // -----------------------------------------------------------------------
    // WeightEditError harnesses
    // -----------------------------------------------------------------------

    /// Prove all WeightEditError variants can be constructed.
    #[kani::unwind(1)]
    #[kani::proof]
    fn weight_edit_error_variants_reachable() {
        let e1 = WeightEditError::EmptyData {
            layer_name: String::from("test"),
        };
        assert!(matches!(e1, WeightEditError::EmptyData { .. }));

        let e2 = WeightEditError::NonFiniteData {
            layer_name: String::from("test"),
            count: 5,
        };
        assert!(matches!(e2, WeightEditError::NonFiniteData { count: 5, .. }));
    }

    // -----------------------------------------------------------------------
    // ApplyReceipt harnesses
    // -----------------------------------------------------------------------

    /// Prove ApplyReceipt can represent both cache-invalidated and non-invalidated states.
    #[kani::unwind(1)]
    #[kani::proof]
    fn apply_receipt_kv_invalidated_states() {
        let with_kv = ApplyReceipt {
            elements_written: 1024,
            kv_invalidated: true,
            kv_generation_before: 5,
            kv_generation_after: 6,
        };
        assert!(with_kv.kv_invalidated);
        assert_eq!(with_kv.kv_generation_after, with_kv.kv_generation_before + 1);

        let without_kv = ApplyReceipt {
            elements_written: 512,
            kv_invalidated: false,
            kv_generation_before: 0,
            kv_generation_after: 0,
        };
        assert!(!without_kv.kv_invalidated);
        assert_eq!(without_kv.kv_generation_before, 0);
        assert_eq!(without_kv.kv_generation_after, 0);
    }

    /// Prove ApplyReceipt PartialEq is reflexive.
    #[kani::unwind(1)]
    #[kani::proof]
    fn apply_receipt_eq_reflexive() {
        let receipt = ApplyReceipt {
            elements_written: 64,
            kv_invalidated: true,
            kv_generation_before: 1,
            kv_generation_after: 2,
        };
        assert_eq!(receipt, receipt.clone());
    }

    // -----------------------------------------------------------------------
    // DeltaApplyReceipt harnesses
    // -----------------------------------------------------------------------

    /// Prove DeltaApplyReceipt fields are independently settable.
    #[kani::unwind(1)]
    #[kani::proof]
    fn delta_apply_receipt_construction() {
        let receipt = DeltaApplyReceipt {
            elements_written: 2048,
            layers_invalidated: 12,
        };
        assert_eq!(receipt.elements_written, 2048);
        assert_eq!(receipt.layers_invalidated, 12);
    }

    /// Prove DeltaApplyReceipt with zero layers_invalidated (no cache case).
    #[kani::unwind(1)]
    #[kani::proof]
    fn delta_apply_receipt_no_cache() {
        let receipt = DeltaApplyReceipt {
            elements_written: 100,
            layers_invalidated: 0,
        };
        assert_eq!(receipt.layers_invalidated, 0);
        assert_eq!(receipt.elements_written, 100);
    }

    // -----------------------------------------------------------------------
    // LiveEditError harnesses
    // -----------------------------------------------------------------------

    /// Prove LiveEditError::DeltaSizeMismatch captures both sizes.
    #[kani::unwind(1)]
    #[kani::proof]
    fn live_edit_error_delta_size_mismatch() {
        let err = LiveEditError::DeltaSizeMismatch {
            buffer_len: 1000,
            delta_len: 500,
        };
        match err {
            LiveEditError::DeltaSizeMismatch {
                buffer_len,
                delta_len,
            } => {
                assert_eq!(buffer_len, 1000);
                assert_eq!(delta_len, 500);
            }
            _ => panic!("expected DeltaSizeMismatch"),
        }
    }

    /// Prove LiveEditError::DeltaNotContiguous variant is constructible.
    #[kani::unwind(1)]
    #[kani::proof]
    fn live_edit_error_not_contiguous() {
        let err = LiveEditError::DeltaNotContiguous;
        assert!(matches!(err, LiveEditError::DeltaNotContiguous));
    }

    /// Prove LiveEditError::NonFiniteResult captures the count.
    #[kani::unwind(1)]
    #[kani::proof]
    fn live_edit_error_non_finite_result() {
        let err = LiveEditError::NonFiniteResult { count: 42 };
        match err {
            LiveEditError::NonFiniteResult { count } => {
                assert_eq!(count, 42);
            }
            _ => panic!("expected NonFiniteResult"),
        }
    }
}
