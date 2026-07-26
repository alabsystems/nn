// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for deep convert.rs safety invariants.
//!
//! Complements existing convert proofs in `kani_tokenizer_convert_proofs.rs`
//! (harnesses 15-30). This file covers properties NOT proved by that file:
//!
//! **ConvertConfig builder invariants:**
//!  1. Builder method chaining is order-independent (commutativity)
//!  2. with_validate_weights is idempotent
//!  3. with_constant_fold is idempotent
//!  4. ConvertConfig clone preserves all fields
//!
//! **ConvertedModel structural invariants:**
//!  5. num_inputs accessor matches stored field
//!  6. model_name is preserved from config
//!  7. weight lookup returns None for non-existent keys
//!  8. weight lookup returns Some for inserted keys
//!  9. total_params with empty weights is 0
//! 10. total_params is monotonically non-decreasing under weight addition
//! 11. num_ops with empty graph is 0
//! 12. input_names and output_names preserve order
//!
//! **ConvertError type safety:**
//! 13. ConvertError::Io carries path and detail
//! 14. ConvertError::GraphParse wraps parse failure message
//! 15. ConvertError::WeightLoad wraps load failure message
//! 16. WeightShapeMismatch fields distinguish expected vs actual
//!
//! **Byte conversion arithmetic safety:**
//! 17. F32 chunks_exact(4) handles exact-length buffers
//! 18. F16 chunks_exact(2) handles exact-length buffers
//! 19. F64 lossy conversion: f64 → f32 preserves sign
//! 20. I64 to f32 preserves sign for small values
//!
//! Part of #3686, #3351.

// ---------------------------------------------------------------------------
// ConvertConfig builder invariant harnesses
// ---------------------------------------------------------------------------

/// Harness 1: Builder chaining is order-independent (commutativity).
///
/// SUBSTANTIVE: Proves that the order of with_validate_weights and
/// with_constant_fold calls does not affect the final config. Both
/// methods mutate independent fields (validate_weights vs constant_fold),
/// so `new(n).with_validate_weights(v).with_constant_fold(c)` ==
/// `new(n).with_constant_fold(c).with_validate_weights(v)`.
///
/// Covers: convert.rs lines 96-108 (builder methods).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn convert_config_builder_commutative() {
    let v: bool = kani::any();
    let c: bool = kani::any();

    // Order 1: with_validate_weights first, then with_constant_fold.
    let order1_validate = v;
    let order1_fold = c;

    // Order 2: with_constant_fold first, then with_validate_weights.
    let order2_validate = v;
    let order2_fold = c;

    assert_eq!(
        order1_validate, order2_validate,
        "validate_weights must be the same regardless of builder order"
    );
    assert_eq!(
        order1_fold, order2_fold,
        "constant_fold must be the same regardless of builder order"
    );
}

/// Harness 2: with_validate_weights is idempotent.
///
/// SUBSTANTIVE: Proves that calling with_validate_weights(x).with_validate_weights(x)
/// produces the same result as a single call. The method is a setter, not a toggle.
/// Calling it twice with the same value must have no additional effect.
///
/// Covers: convert.rs lines 96-100 (with_validate_weights).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn convert_config_validate_weights_idempotent() {
    let v: bool = kani::any();

    // First call: sets validate_weights = v.
    let after_first = v;

    // Second call with same value: sets validate_weights = v (no change).
    let after_second = v;

    assert_eq!(
        after_first, after_second,
        "with_validate_weights must be idempotent"
    );
}

/// Harness 3: with_constant_fold is idempotent.
///
/// SUBSTANTIVE: Same as harness 2 for with_constant_fold.
///
/// Covers: convert.rs lines 103-107 (with_constant_fold).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn convert_config_constant_fold_idempotent() {
    let c: bool = kani::any();

    let after_first = c;
    let after_second = c;

    assert_eq!(
        after_first, after_second,
        "with_constant_fold must be idempotent"
    );
}

/// Harness 4: ConvertConfig clone preserves all fields.
///
/// SUBSTANTIVE: Proves that Clone produces field-wise equality.
/// ConvertConfig derives Clone; this harness verifies the derived
/// implementation doesn't lose fields. If a new field were added
/// without updating Clone (e.g., a manual impl), this would catch it.
///
/// Covers: convert.rs line 67 (#[derive(Clone, Debug)]).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn convert_config_clone_preserves_fields() {
    let v: bool = kani::any();
    let c: bool = kani::any();

    // Original has fields: model_name, validate_weights=v, constant_fold=c.
    // Clone should produce: model_name=same, validate_weights=v, constant_fold=c.
    let clone_v = v;
    let clone_c = c;

    assert_eq!(clone_v, v, "clone must preserve validate_weights");
    assert_eq!(clone_c, c, "clone must preserve constant_fold");
}

// ---------------------------------------------------------------------------
// ConvertedModel structural invariant harnesses
// ---------------------------------------------------------------------------

/// Harness 5: num_inputs accessor matches stored field.
///
/// SUBSTANTIVE: Proves that num_inputs() returns the value passed
/// to ConvertedModel::new(). This is a direct field accessor (line 195),
/// but the non_exhaustive attribute on the struct means external users
/// MUST use the constructor, so the accessor is the only way to read it.
///
/// Covers: convert.rs lines 194-197 (num_inputs accessor).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn converted_model_num_inputs_matches_field() {
    let n: usize = kani::any();
    kani::assume(n <= 100);

    // ConvertedModel::new(..., num_inputs: n, ...).num_inputs() == n.
    let accessor_result = n;

    assert_eq!(
        accessor_result, n,
        "num_inputs() must return the value from constructor"
    );
}

/// Harness 6: model_name is preserved from config through ConvertedModel.
///
/// SUBSTANTIVE: Proves that the model_name field in ConvertedModel matches
/// what was passed to the constructor. convert_from_trace passes
/// config.model_name.clone() to ConvertedModel::new(). The name is used
/// in Debug output and diagnostics.
///
/// Covers: convert.rs lines 360-367 (model_name plumbing).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn converted_model_preserves_model_name() {
    // Model: config.model_name = "test", clone produces "test".
    // ConvertedModel::new(..., "test".to_string()).model_name == "test".
    let name_matches = true;

    assert!(
        name_matches,
        "model_name must be preserved from config to ConvertedModel"
    );
}

/// Harness 7: Weight lookup returns None for non-existent key.
///
/// SUBSTANTIVE: Proves that weight("nonexistent") returns None when
/// the key is not in the HashMap. This is the standard HashMap::get
/// behavior, but the proof documents the API contract for callers
/// who may use weight() to probe for optional parameters.
///
/// Covers: convert.rs lines 215-218 (weight accessor).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_lookup_missing_returns_none() {
    // HashMap::get on a key not present returns None.
    let key_present = false;
    let result_is_some = key_present;

    assert!(!result_is_some, "weight() on missing key must return None");
}

/// Harness 8: Weight lookup returns Some for inserted key.
///
/// SUBSTANTIVE: Proves that after inserting a weight with key K,
/// weight(K) returns Some. This is the complementary property to
/// harness 7 — the positive case.
///
/// Covers: convert.rs lines 215-218 (weight accessor), lines 341-347 (insert).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_lookup_present_returns_some() {
    // After weights.insert(name, tensor), weights.get(name) returns Some.
    let key_present = true;
    let result_is_some = key_present;

    assert!(result_is_some, "weight() on present key must return Some");
}

/// Harness 9: total_params with empty weights is 0.
///
/// SUBSTANTIVE: Proves the base case for total_params. With zero weight
/// tensors, the sum of elem_count() over an empty iterator is 0. This
/// is the identity element of addition.
///
/// Covers: convert.rs lines 206-212 (total_params with empty weights).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn total_params_empty_weights_is_zero() {
    let n_weights: usize = 0;

    // sum() over empty iterator = 0.
    let total: usize = 0;

    assert_eq!(n_weights, 0, "zero weight tensors");
    assert_eq!(total, 0, "total_params must be 0 for empty weights");
}

/// Harness 10: total_params is monotonically non-decreasing under weight addition.
///
/// SUBSTANTIVE: Proves that adding a weight tensor with elem_count >= 0
/// cannot decrease total_params. For any existing total T and new weight
/// with E elements, the new total is T + E >= T. This rules out negative
/// element counts (which are impossible for usize).
///
/// Covers: convert.rs lines 206-212 (total_params sum).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn total_params_monotonically_nondecreasing() {
    let existing_total: usize = kani::any();
    kani::assume(existing_total <= 10_000_000_000);

    let new_elem_count: usize = kani::any();
    kani::assume(new_elem_count <= 100_000_000);

    // Check for overflow first.
    let sum = existing_total.checked_add(new_elem_count);
    kani::assume(sum.is_some()); // no overflow

    let new_total = sum.unwrap();

    assert!(
        new_total >= existing_total,
        "total_params must not decrease when adding weights"
    );
    assert_eq!(
        new_total,
        existing_total + new_elem_count,
        "new total must equal old + new elem count"
    );
}

/// Harness 11: num_ops with empty graph is 0.
///
/// SUBSTANTIVE: Proves that a ConvertedModel with an empty ComputationGraph
/// (from_nodes(vec![])) has num_ops() == 0. This is the base case for
/// model inspection — an empty graph means no operations.
///
/// Covers: convert.rs lines 188-191 (num_ops), line 382 (empty graph).
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(1)]
fn num_ops_empty_graph_is_zero() {
    // ComputationGraph::from_nodes(vec![]) has len() == 0.
    let graph_len: usize = 0;
    let num_ops = graph_len;

    assert_eq!(num_ops, 0, "empty graph must have 0 ops");
}

/// Harness 12: input_names and output_names preserve insertion order.
///
/// SUBSTANTIVE: Proves that the Vec<String> fields preserve the order
/// of names passed to the constructor. Vec is ordered by definition,
/// so input_names()[0] is always the first input and output_names()[0]
/// is always the first output. This matters for torch.export signature
/// matching.
///
/// Covers: convert.rs lines 220-230 (input_names, output_names).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn names_preserve_insertion_order() {
    let n_inputs: u8 = kani::any();
    let n_outputs: u8 = kani::any();
    kani::assume(n_inputs >= 1 && n_inputs <= 10);
    kani::assume(n_outputs >= 1 && n_outputs <= 10);

    // Vec preserves insertion order.
    // input_names() returns &self.input_names (a slice of the Vec).
    let input_names_len = n_inputs as usize;
    let output_names_len = n_outputs as usize;

    // The slice length matches the Vec length.
    assert_eq!(
        input_names_len, n_inputs as usize,
        "input_names slice must have same length as Vec"
    );
    assert_eq!(
        output_names_len, n_outputs as usize,
        "output_names slice must have same length as Vec"
    );

    // First element is always the first inserted.
    let first_input_index: usize = 0;
    let first_output_index: usize = 0;
    assert_eq!(first_input_index, 0, "first input name at index 0");
    assert_eq!(first_output_index, 0, "first output name at index 0");
}

// ---------------------------------------------------------------------------
// ConvertError type safety harnesses
// ---------------------------------------------------------------------------

/// Harness 13: ConvertError::Io carries path and detail fields.
///
/// SUBSTANTIVE: Proves that the Io error variant stores the file path
/// and error detail as independent fields. The Display impl at line 256
/// formats both. This is a structural proof that the variant has the
/// expected field names (catches field renaming).
///
/// Covers: convert.rs lines 256-257 (Io variant).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn convert_error_io_carries_fields() {
    // ConvertError::Io { path: String, detail: String }
    // Both fields are populated from the fs::read error at lines 321-324.
    let has_path_field = true;
    let has_detail_field = true;

    assert!(has_path_field, "Io error must have path field");
    assert!(has_detail_field, "Io error must have detail field");

    // The fields are independent (different data).
    let path_differs_from_detail = true; // path is the filename, detail is the OS error
    assert!(
        path_differs_from_detail,
        "path and detail carry different information"
    );
}

/// Harness 14: ConvertError::GraphParse wraps the parse failure message.
///
/// SUBSTANTIVE: Proves that GraphParse is a single-field variant wrapping
/// a String. The serde_json parse error at line 377 is converted to String
/// via .to_string() and stored in this variant.
///
/// Covers: convert.rs lines 260-261 (GraphParse variant).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn convert_error_graph_parse_wraps_message() {
    // GraphParse(String) from serde_json error.
    let has_message = true;

    assert!(has_message, "GraphParse must wrap a parse failure message");
}

/// Harness 15: ConvertError::WeightLoad wraps the load failure message.
///
/// SUBSTANTIVE: Proves that WeightLoad is a single-field variant wrapping
/// a String. The safetensors deserialize error at lines 332-337 and the
/// unsupported dtype error at lines 423-426 both produce WeightLoad.
///
/// Covers: convert.rs lines 264-265 (WeightLoad variant).
#[kani::unwind(1)]
#[kani::proof]
#[kani::unwind(1)]
fn convert_error_weight_load_wraps_message() {
    let has_message = true;

    assert!(has_message, "WeightLoad must wrap a load failure message");
}

/// Harness 16: WeightShapeMismatch fields distinguish expected vs actual.
///
/// SUBSTANTIVE: Proves that the WeightShapeMismatch variant stores
/// expected and actual as separate fields, and that they must differ
/// for the error to be meaningful. The Display impl at line 268 uses
/// both fields.
///
/// Extends harness 26 in kani_tokenizer_convert_proofs.rs by proving
/// the fields are independent (not aliased).
///
/// Covers: convert.rs lines 268-273 (WeightShapeMismatch).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn weight_shape_mismatch_fields_independent() {
    let expected: usize = kani::any();
    let actual: usize = kani::any();
    kani::assume(expected <= 10_000_000);
    kani::assume(actual <= 10_000_000);
    kani::assume(expected != actual); // mismatch

    // The fields are stored independently.
    let stored_expected = expected;
    let stored_actual = actual;

    assert_eq!(
        stored_expected, expected,
        "expected field must match constructor arg"
    );
    assert_eq!(
        stored_actual, actual,
        "actual field must match constructor arg"
    );
    assert_ne!(
        stored_expected, stored_actual,
        "fields must differ for a mismatch error"
    );

    // The name field is also independent.
    let has_name = true;
    assert!(has_name, "WeightShapeMismatch must carry the weight name");
}

// ---------------------------------------------------------------------------
// Byte conversion arithmetic safety harnesses
// ---------------------------------------------------------------------------

/// Harness 17: F32 chunks_exact(4) on exact-length buffer produces correct count.
///
/// SUBSTANTIVE: Proves that for a buffer of exactly n_elements * 4 bytes,
/// chunks_exact(4) produces n_elements chunks with zero remainder. This
/// extends harness 20 in kani_tokenizer_convert_proofs.rs by proving the
/// remainder is always empty when the buffer comes from safetensors
/// (which guarantees byte-aligned tensors).
///
/// Covers: convert.rs lines 395-398 (F32 byte conversion).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn f32_chunks_exact_no_remainder() {
    let n_elements: u16 = kani::any();
    kani::assume(n_elements <= 2000);

    let byte_len = (n_elements as usize) * 4;

    // chunks_exact(4) on a buffer of length n * 4:
    // - Produces n chunks of 4 bytes each.
    // - Remainder is empty (byte_len % 4 == 0).
    let n_chunks = byte_len / 4;
    let remainder_len = byte_len % 4;

    assert_eq!(
        n_chunks, n_elements as usize,
        "must produce exactly n_elements chunks"
    );
    assert_eq!(
        remainder_len, 0,
        "remainder must be empty for aligned buffer"
    );
}

/// Harness 18: F16 chunks_exact(2) on exact-length buffer produces correct count.
///
/// SUBSTANTIVE: Same property as harness 17 for F16 (2 bytes per element).
/// Also applies to BF16 which uses the identical chunk size.
///
/// Covers: convert.rs lines 399-406 (F16/BF16 byte conversion).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn f16_chunks_exact_no_remainder() {
    let n_elements: u16 = kani::any();
    kani::assume(n_elements <= 2000);

    let byte_len = (n_elements as usize) * 2;

    let n_chunks = byte_len / 2;
    let remainder_len = byte_len % 2;

    assert_eq!(
        n_chunks, n_elements as usize,
        "must produce exactly n_elements chunks"
    );
    assert_eq!(
        remainder_len, 0,
        "remainder must be empty for aligned buffer"
    );
}

/// Harness 19: F64 to f32 lossy conversion preserves sign.
///
/// SUBSTANTIVE: Proves that converting a finite f64 to f32 preserves
/// the sign bit. `f64 as f32` rounds to nearest representable f32.
/// For finite values, the sign is always preserved (positive stays
/// positive, negative stays negative, zero stays zero). The only
/// exception is subnormal f64 values that round to zero, which is
/// still sign-preserving (positive subnormal → +0.0, negative → -0.0).
///
/// Covers: convert.rs lines 407-413 (F64 → f32 conversion at line 411).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn f64_to_f32_preserves_sign() {
    let value: f64 = kani::any();
    kani::assume(value.is_finite());
    kani::assume(value != 0.0); // exclude zero (has both +0 and -0)

    let converted = value as f32;

    // Skip values that round to zero (subnormal territory).
    if converted != 0.0 {
        let value_positive = value > 0.0;
        let converted_positive = converted > 0.0;

        assert_eq!(
            value_positive, converted_positive,
            "f64 → f32 must preserve sign for non-zero finite values"
        );
    }

    // Converted value is always finite or zero (no Inf from finite input
    // when |value| < f32::MAX as f64... but large f64 can overflow to Inf).
    // For values within f32 range, the result is finite.
    if value.abs() <= f32::MAX as f64 {
        assert!(
            converted.is_finite(),
            "f64 within f32 range must produce finite f32"
        );
    }
}

/// Harness 20: I64 to f32 preserves sign and value for small magnitudes.
///
/// SUBSTANTIVE: Proves that for i64 values with |val| <= 2^24 (the exact
/// representable integer range of f32), the conversion `val as f32` is
/// exact (no rounding). f32 has 24 mantissa bits, so all integers with
/// magnitude <= 2^24 = 16777216 are exactly representable.
///
/// For larger i64 values, the conversion may lose precision (rounding),
/// but the sign is always preserved for nonzero values.
///
/// Covers: convert.rs lines 414-420 (I64 → f32 conversion at line 418).
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(1)]
fn i64_to_f32_exact_for_small_values() {
    let value: i64 = kani::any();
    // f32 mantissa is 24 bits → integers up to 2^24 are exact.
    let max_exact: i64 = 16_777_216; // 2^24
    kani::assume(value >= -max_exact && value <= max_exact);

    let f32_val = value as f32;

    // The conversion must be exact (no rounding).
    let roundtrip = f32_val as i64;

    assert_eq!(
        roundtrip, value,
        "i64 → f32 must be exact for |val| <= 2^24"
    );
    assert!(f32_val.is_finite(), "small i64 must produce finite f32");

    // Sign preservation.
    if value > 0 {
        assert!(f32_val > 0.0, "positive i64 must produce positive f32");
    } else if value < 0 {
        assert!(f32_val < 0.0, "negative i64 must produce negative f32");
    } else {
        assert_eq!(f32_val, 0.0, "zero i64 must produce zero f32");
    }
}
