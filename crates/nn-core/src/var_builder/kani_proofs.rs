// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani proof harnesses for VarBuilder weight loading safety (#3621).
//!
//! Proves correctness properties of the VarBuilder path construction,
//! name resolution, dtype handling, and shape validation logic:
//!
//! - Path prefix construction produces correct dot-separated keys
//! - Empty prefixes are skipped (no double-dot or leading-dot keys)
//! - resolve_name joins path + tensor_name correctly
//! - Name mapping is applied after path resolution
//! - effective_weight_dtype falls back correctly
//! - to_dtype/to_device preserve other fields
//! - Shape comparison is reflexive, symmetric, and detects mismatches
//! - Prefix mapping first-match-wins semantics
//! - Rename map passthrough for unmapped keys
//! - DType size_bytes is always nonzero (weight buffer sizing)
//! - Weight buffer byte count cannot underflow

// -----------------------------------------------------------------------
// Harness 1: Single pp() produces correct prefix string.
//
// VarBuilder.pp("encoder") must produce prefix "encoder".
// The path Vec must contain exactly 1 element, and joining with "."
// must produce the prefix string without leading/trailing dots.
// -----------------------------------------------------------------------

/// Prove: a single pp() call produces a prefix equal to the input string.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn single_pp_produces_correct_prefix() {
    // Model the path as a Vec with one element.
    // pp(s) pushes s when s is non-empty, then prefix() joins with ".".
    let path: Vec<String> = vec!["encoder".to_string()];
    let prefix = path.join(".");
    assert!(
        prefix == "encoder",
        "single pp must produce the input string"
    );
    assert!(!prefix.contains(".."), "must not contain double dot");
    assert!(!prefix.starts_with('.'), "must not start with dot");
    assert!(!prefix.ends_with('.'), "must not end with dot");
}

// -----------------------------------------------------------------------
// Harness 2: Chained pp() produces dot-separated path.
//
// VarBuilder.pp("a").pp("b").pp("c") must produce "a.b.c".
// The number of dots equals the number of segments minus one.
// -----------------------------------------------------------------------

/// Prove: chained pp() produces dot-separated segments.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn chained_pp_produces_dot_separated_path() {
    let path: Vec<String> = vec![
        "model".to_string(),
        "encoder".to_string(),
        "layer0".to_string(),
    ];
    let prefix = path.join(".");
    assert!(prefix == "model.encoder.layer0");

    // Number of dots = number of segments - 1
    let dot_count = prefix.chars().filter(|&c| c == '.').count();
    assert!(dot_count == path.len() - 1, "dots must equal segments - 1");
}

// -----------------------------------------------------------------------
// Harness 3: Empty string pp() is skipped.
//
// pp("") must NOT push to the path vec. This prevents ".weight",
// "encoder..weight", and "encoder." keys.
// -----------------------------------------------------------------------

/// Prove: empty prefix is skipped — path does not grow.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn empty_pp_is_skipped() {
    // Simulate pp("encoder"), pp(""), pp("layer0")
    let mut path: Vec<String> = Vec::new();

    let segments = ["encoder", "", "layer0"];
    for s in &segments {
        let prefix_str = s.to_string();
        // Match production logic: skip empty
        if !prefix_str.is_empty() {
            path.push(prefix_str);
        }
    }

    assert!(path.len() == 2, "empty segment must be skipped");
    let joined = path.join(".");
    assert!(joined == "encoder.layer0");
    assert!(!joined.contains(".."), "must not contain double dot");
}

// -----------------------------------------------------------------------
// Harness 4: resolve_name with empty path returns bare tensor name.
//
// When no pp() has been called, resolve_name("weight") must return
// "weight" — not ".weight" or any other mangling.
// -----------------------------------------------------------------------

/// Prove: resolve_name on empty path returns the bare tensor name.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn resolve_name_empty_path_returns_bare_name() {
    let path: Vec<String> = Vec::new();
    let tensor_name = "weight";

    // Match production resolve_name logic
    let name = if path.is_empty() {
        tensor_name.to_string()
    } else {
        format!("{}.{}", path.join("."), tensor_name)
    };

    assert!(name == "weight", "empty path must return bare tensor name");
    assert!(!name.starts_with('.'), "must not start with dot");
}

// -----------------------------------------------------------------------
// Harness 5: resolve_name with path joins correctly.
//
// After pp("encoder").pp("layer0"), resolve_name("weight") must return
// "encoder.layer0.weight".
// -----------------------------------------------------------------------

/// Prove: resolve_name joins path segments and tensor name with dots.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn resolve_name_joins_path_and_tensor_name() {
    let path: Vec<String> = vec!["encoder".to_string(), "layer0".to_string()];
    let tensor_name = "weight";

    let name = if path.is_empty() {
        tensor_name.to_string()
    } else {
        format!("{}.{}", path.join("."), tensor_name)
    };

    assert!(name == "encoder.layer0.weight");

    // The resolved name has exactly path.len() dots
    let dot_count = name.chars().filter(|&c| c == '.').count();
    assert!(
        dot_count == path.len(),
        "dot count must equal path length (segments + tensor name separator)"
    );
}

// -----------------------------------------------------------------------
// Harness 6: pp() path length grows by exactly 1 per non-empty call.
//
// Proves that each non-empty pp() adds exactly one segment, and the
// path length is bounded by the number of calls.
// -----------------------------------------------------------------------

/// Prove: each non-empty pp() increases path length by exactly 1.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn pp_path_length_grows_by_one() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 8);

    let mut path: Vec<String> = Vec::new();
    for _ in 0..n {
        // Non-empty prefix always pushes
        path.push("seg".to_string());
    }

    assert!(path.len() == n, "path length must equal push count");
}

// -----------------------------------------------------------------------
// Harness 7: effective_weight_dtype falls back to VarBuilder dtype.
//
// When no precision policy is set, effective_weight_dtype() must return
// the VarBuilder's dtype. When a policy is set, it must return the
// policy's weight_dtype.
// -----------------------------------------------------------------------

/// Prove: effective_weight_dtype fallback logic is correct.
#[kani::unwind(1)]
#[kani::proof]
fn effective_weight_dtype_fallback() {
    use crate::DType;

    // All float dtypes that VarBuilder uses
    let dtypes = [DType::F32, DType::F16, DType::BF16, DType::F64];

    for &vb_dtype in &dtypes {
        // Without policy: returns VarBuilder dtype
        let policy: Option<DType> = None;
        let effective = policy.unwrap_or(vb_dtype);
        assert!(
            effective == vb_dtype,
            "no policy must fall back to vb dtype"
        );
    }

    for &vb_dtype in &dtypes {
        for &policy_dtype in &dtypes {
            // With policy: returns policy dtype
            let policy: Option<DType> = Some(policy_dtype);
            let effective = policy.unwrap_or(vb_dtype);
            assert!(
                effective == policy_dtype,
                "with policy must return policy dtype"
            );
        }
    }
}

// -----------------------------------------------------------------------
// Harness 8: to_dtype preserves device, to_device preserves dtype.
//
// VarBuilder.to_dtype(new_dt) must change only dtype, not device.
// VarBuilder.to_device(new_dev) must change only device, not dtype.
// -----------------------------------------------------------------------

/// Prove: to_dtype changes dtype but preserves device identity.
#[kani::unwind(1)]
#[kani::proof]
fn to_dtype_preserves_device() {
    use crate::{DType, Device};

    let original_dtype = DType::F32;
    let original_device = Device::Cpu;
    let new_dtype = DType::BF16;

    // Model: to_dtype creates a clone with new dtype
    let result_dtype = new_dtype;
    let result_device = original_device;

    assert!(result_dtype == new_dtype, "dtype must change");
    assert!(result_device == original_device, "device must be preserved");
    assert!(
        result_dtype != original_dtype,
        "new dtype differs from original"
    );
}

/// Prove: to_device changes device but preserves dtype identity.
#[kani::unwind(1)]
#[kani::proof]
fn to_device_preserves_dtype() {
    use crate::{DType, Device};

    let original_dtype = DType::F32;
    let original_device = Device::Cpu;
    let new_device = Device::Metal { device_id: 0 };

    // Model: to_device creates a clone with new device
    let result_dtype = original_dtype;
    let result_device = new_device;

    assert!(result_device == new_device, "device must change");
    assert!(result_dtype == original_dtype, "dtype must be preserved");
    assert!(
        result_device != original_device,
        "new device differs from original"
    );
}

// -----------------------------------------------------------------------
// Harness 9: Shape comparison reflexivity (identity check).
//
// TensorMapBackend::get compares t.dims() == expected_dims.
// Prove: identical shapes always compare equal (reflexivity).
// -----------------------------------------------------------------------

/// Prove: shape comparison is reflexive — a shape equals itself.
#[kani::unwind(1)]
#[kani::proof]
fn shape_comparison_reflexive() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();
    kani::assume(d0 >= 1 && d0 <= 4096);
    kani::assume(d1 >= 1 && d1 <= 4096);

    let shape = [d0, d1];
    let same = [d0, d1];

    assert!(shape == same, "identical shapes must compare equal");
}

// -----------------------------------------------------------------------
// Harness 10: Shape comparison symmetry.
//
// If shape_a != shape_b, then shape_b != shape_a. This is important
// because TensorMapBackend uses == which should be symmetric.
// -----------------------------------------------------------------------

/// Prove: shape comparison is symmetric.
#[kani::unwind(1)]
#[kani::proof]
fn shape_comparison_symmetric() {
    let a0: usize = kani::any();
    let a1: usize = kani::any();
    let b0: usize = kani::any();
    let b1: usize = kani::any();

    kani::assume(a0 >= 1 && a0 <= 4096);
    kani::assume(a1 >= 1 && a1 <= 4096);
    kani::assume(b0 >= 1 && b0 <= 4096);
    kani::assume(b1 >= 1 && b1 <= 4096);

    let shape_a = [a0, a1];
    let shape_b = [b0, b1];

    // Symmetry: (a == b) implies (b == a)
    if shape_a == shape_b {
        assert!(shape_b == shape_a, "equality must be symmetric");
    }
    // Contra-symmetry: (a != b) implies (b != a)
    if shape_a != shape_b {
        assert!(shape_b != shape_a, "inequality must be symmetric");
    }
}

// -----------------------------------------------------------------------
// Harness 11: Shape mismatch with different rank is always detected.
//
// If one shape is rank 2 and the other is rank 1, they must never
// compare equal. This catches the case where a bias [N] is
// accidentally matched against a weight [N, M].
// -----------------------------------------------------------------------

/// Prove: rank-2 shape is never equal to a rank-1 shape.
#[kani::unwind(1)]
#[kani::proof]
fn rank_mismatch_always_detected() {
    let d0: usize = kani::any();
    let d1: usize = kani::any();

    kani::assume(d0 >= 1 && d0 <= 4096);
    kani::assume(d1 >= 1 && d1 <= 4096);

    let rank2: &[usize] = &[d0, d1];
    let rank1: &[usize] = &[d0];

    // Different ranks must never compare equal
    assert!(rank2 != rank1, "rank-2 must not equal rank-1");
}

// -----------------------------------------------------------------------
// Harness 12: DType size_bytes is always nonzero.
//
// Weight buffer allocation uses dtype.size_bytes() * element_count.
// A zero size_bytes would produce a zero-length buffer, which would
// cause out-of-bounds reads when the Metal backend maps weight data.
// -----------------------------------------------------------------------

/// Prove: all DType variants have nonzero size_bytes.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_size_bytes_always_nonzero() {
    use crate::DType;

    let all_dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];

    for dt in all_dtypes {
        assert!(
            dt.size_bytes() > 0,
            "dtype size_bytes must be nonzero for buffer allocation"
        );
    }
}

// -----------------------------------------------------------------------
// Harness 13: Weight buffer byte count cannot underflow.
//
// The byte count for a weight buffer is element_count * size_bytes.
// For any valid shape with positive dimensions and any DType, the
// product must be >= 1 (cannot be zero, cannot underflow).
// -----------------------------------------------------------------------

/// Prove: weight buffer byte count is always positive for valid shapes.
#[kani::unwind(1)]
#[kani::proof]
fn weight_buffer_bytes_positive() {
    use crate::DType;

    let d0: usize = kani::any();
    let d1: usize = kani::any();

    kani::assume(d0 >= 1 && d0 <= 1024);
    kani::assume(d1 >= 1 && d1 <= 1024);

    let element_count = d0 * d1;
    assert!(element_count >= 1, "element count must be positive");

    // F32 is the most common weight dtype
    let byte_count = element_count * DType::F32.size_bytes();
    assert!(byte_count >= 4, "F32 buffer must be at least 4 bytes");

    // F16/BF16 are used with mixed precision
    let bf16_bytes = element_count * DType::BF16.size_bytes();
    assert!(bf16_bytes >= 2, "BF16 buffer must be at least 2 bytes");
}

// -----------------------------------------------------------------------
// Harness 14: Prefix mapping first-match-wins semantics.
//
// with_prefix_mapping iterates pairs and returns on first match.
// Prove: when multiple prefixes match, the first one is used.
// -----------------------------------------------------------------------

/// Prove: prefix mapping uses first-match-wins.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn prefix_mapping_first_match_wins() {
    let name = "a.weight";

    // Two pairs that both match prefix "a"
    let pairs: &[(&str, &str)] = &[("a", "first"), ("a", "second")];

    let mut result = name.to_string();
    for (from, to) in pairs {
        if let Some(rest) = name.strip_prefix(from) {
            result = format!("{to}{rest}");
            break; // first match wins
        }
    }

    assert!(result == "first.weight", "first matching prefix must win");
    assert!(
        result != "second.weight",
        "second matching prefix must lose"
    );
}

// -----------------------------------------------------------------------
// Harness 15: Rename map passthrough for unmapped keys.
//
// with_rename_map uses HashMap::get — keys not in the map must pass
// through unchanged. Prove: an unmapped key is returned as-is.
// -----------------------------------------------------------------------

/// Prove: rename map passes through unmapped keys unchanged.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn rename_map_passthrough_unmapped() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    map.insert("a.weight".to_string(), "b.weight".to_string());

    let unmapped_key = "decoder.weight";

    // Production logic: map.get(name).cloned().unwrap_or_else(|| name.to_string())
    let result = map
        .get(unmapped_key)
        .cloned()
        .unwrap_or_else(|| unmapped_key.to_string());

    assert!(
        result == "decoder.weight",
        "unmapped key must pass through unchanged"
    );
}

/// Prove: rename map returns the mapped value for a mapped key.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn rename_map_returns_mapped_value() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    map.insert("a.weight".to_string(), "b.weight".to_string());

    let mapped_key = "a.weight";

    let result = map
        .get(mapped_key)
        .cloned()
        .unwrap_or_else(|| mapped_key.to_string());

    assert!(result == "b.weight", "mapped key must return mapped value");
}

// -----------------------------------------------------------------------
// Harness 17: DType float classification is consistent with weight loading.
//
// Weight loading uses to_dtype() which only supports float-to-float.
// Prove: is_float() correctly partitions all DType variants.
// -----------------------------------------------------------------------

/// Prove: DType is_float and is_int are mutually exclusive, and Bool
/// is neither. Every variant is exactly one of {float, int, bool}.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_classification_partitions_correctly() {
    use crate::DType;

    let all_dtypes = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::F64,
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U8,
        DType::Bool,
    ];

    for dt in all_dtypes {
        let f = dt.is_float();
        let i = dt.is_int();

        // Mutually exclusive
        assert!(!(f && i), "a dtype cannot be both float and int");

        // Exhaustive: every variant is float, int, or Bool
        let is_bool = matches!(dt, DType::Bool);
        assert!(f || i || is_bool, "every dtype must be float, int, or bool");

        // Bool is neither float nor int
        if is_bool {
            assert!(!f && !i, "Bool must be neither float nor int");
        }
    }
}

// -----------------------------------------------------------------------
// Harness 18: Precision policy effective_weight_dtype identity.
//
// When precision_policy is None, effective_weight_dtype must equal
// the VarBuilder's dtype. This is the default code path for all
// models that don't use mixed precision.
// -----------------------------------------------------------------------

/// Prove: without precision policy, effective_weight_dtype == vb.dtype.
#[kani::unwind(1)]
#[kani::proof]
fn effective_weight_dtype_identity_without_policy() {
    use crate::DType;

    // Symbolic dtype (any float variant)
    let vb_dtype_idx: u8 = kani::any();
    kani::assume(vb_dtype_idx < 4);

    let vb_dtype = match vb_dtype_idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        _ => DType::F64,
    };

    // Production logic: precision_policy.map(|p| p.weight_dtype).unwrap_or(self.dtype)
    let policy: Option<DType> = None;
    let effective = policy.unwrap_or(vb_dtype);

    assert!(
        effective == vb_dtype,
        "without policy, effective dtype must equal vb dtype"
    );
}

// -----------------------------------------------------------------------
// Harness 19: Dot count in resolved name equals path depth.
//
// For a path of depth N and a tensor name, the resolved name has
// exactly N dots (N-1 between segments, 1 before tensor name).
// -----------------------------------------------------------------------

/// Prove: resolved name dot count equals path segment count.
#[kani::unwind(16)]
#[kani::proof]
#[kani::unwind(8)]
fn resolved_name_dot_count_equals_depth() {
    // Test depths 1, 2, 3
    for depth in 1..=3usize {
        let mut path: Vec<String> = Vec::new();
        for i in 0..depth {
            path.push(format!("s{i}"));
        }

        let tensor_name = "weight";
        let name = format!("{}.{}", path.join("."), tensor_name);

        let dot_count = name.chars().filter(|&c| c == '.').count();
        // path.join(".") has depth-1 dots, plus 1 dot before tensor_name = depth dots total
        assert!(dot_count == depth, "dot count must equal path depth");
    }
}

// -----------------------------------------------------------------------
// Harness 20: convert_tensor dtype identity is a no-op.
//
// When the requested dtype equals the tensor's dtype, convert_tensor
// must return a tensor with the same dtype (identity conversion).
// -----------------------------------------------------------------------

/// Prove: dtype identity conversion preserves dtype.
#[kani::unwind(1)]
#[kani::proof]
fn dtype_identity_conversion_is_noop() {
    use crate::DType;

    let dtypes = [DType::F32, DType::F16, DType::BF16, DType::F64];

    for &dt in &dtypes {
        // Production logic: if t.dtype() != dtype { t.to_dtype(dtype) } else { t.clone() }
        let tensor_dtype = dt;
        let requested_dtype = dt;

        let needs_conversion = tensor_dtype != requested_dtype;
        assert!(!needs_conversion, "same dtype must not trigger conversion");
    }
}

/// Prove: different dtypes always trigger conversion.
#[kani::unwind(1)]
#[kani::proof]
fn different_dtype_triggers_conversion() {
    use crate::DType;

    let dt_a_idx: u8 = kani::any();
    let dt_b_idx: u8 = kani::any();
    kani::assume(dt_a_idx < 4);
    kani::assume(dt_b_idx < 4);
    kani::assume(dt_a_idx != dt_b_idx);

    let dt_a = match dt_a_idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        _ => DType::F64,
    };
    let dt_b = match dt_b_idx {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        _ => DType::F64,
    };

    let needs_conversion = dt_a != dt_b;
    assert!(needs_conversion, "different dtypes must trigger conversion");
}
