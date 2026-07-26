// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for where/clamp PTX generation and CPU reference implementations.

use super::*;

// =========================================================================
// Where reference: all-true condition
// =========================================================================

#[test]
fn test_where_reference_all_true() {
    let cond = vec![1u32, 1, 1, 1];
    let a = vec![10.0f32, 20.0, 30.0, 40.0];
    let b = vec![50.0f32, 60.0, 70.0, 80.0];
    let out = where_reference(&cond, &a, &b);
    assert_eq!(out, a, "all-true condition should select a entirely");
}

// =========================================================================
// Where reference: all-false condition
// =========================================================================

#[test]
fn test_where_reference_all_false() {
    let cond = vec![0u32, 0, 0, 0];
    let a = vec![10.0f32, 20.0, 30.0, 40.0];
    let b = vec![50.0f32, 60.0, 70.0, 80.0];
    let out = where_reference(&cond, &a, &b);
    assert_eq!(out, b, "all-false condition should select b entirely");
}

// =========================================================================
// Where reference: mixed condition
// =========================================================================

#[test]
fn test_where_reference_mixed() {
    let cond = vec![1u32, 0, 1, 0, 1];
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0, 50.0];
    let out = where_reference(&cond, &a, &b);
    assert_eq!(out, vec![1.0, 20.0, 3.0, 40.0, 5.0]);
}

// =========================================================================
// Where reference: nonzero values > 1 treated as true
// =========================================================================

#[test]
fn test_where_reference_nonzero_cond() {
    let cond = vec![0u32, 42, 0, 255];
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![5.0f32, 6.0, 7.0, 8.0];
    let out = where_reference(&cond, &a, &b);
    assert_eq!(out, vec![5.0, 2.0, 7.0, 4.0]);
}

// =========================================================================
// Where reference: empty input
// =========================================================================

#[test]
fn test_where_reference_empty() {
    let out = where_reference(&[], &[], &[]);
    assert!(out.is_empty());
}

// =========================================================================
// Where reference: length mismatch panics
// =========================================================================

#[test]
#[should_panic(expected = "condition and a must have the same length")]
fn test_where_reference_cond_a_mismatch() {
    let _ = where_reference(&[1], &[1.0, 2.0], &[3.0, 4.0]);
}

#[test]
#[should_panic(expected = "a and b must have the same length")]
fn test_where_reference_a_b_mismatch() {
    let _ = where_reference(&[1, 0], &[1.0, 2.0], &[3.0]);
}

// =========================================================================
// Clamp reference: within range unchanged
// =========================================================================

#[test]
fn test_clamp_reference_within_range() {
    let input = vec![0.5f32, 1.0, 2.0, 3.0, 4.5];
    let out = clamp_reference(&input, 0.0, 5.0);
    assert_eq!(out, input, "values within range should be unchanged");
}

// =========================================================================
// Clamp reference: below min clamped
// =========================================================================

#[test]
fn test_clamp_reference_below_min() {
    let input = vec![-10.0f32, -1.0, -0.01];
    let out = clamp_reference(&input, 0.0, 5.0);
    assert_eq!(
        out,
        vec![0.0, 0.0, 0.0],
        "values below min should be clamped to min"
    );
}

// =========================================================================
// Clamp reference: above max clamped
// =========================================================================

#[test]
fn test_clamp_reference_above_max() {
    let input = vec![6.0f32, 100.0, 999.0];
    let out = clamp_reference(&input, 0.0, 5.0);
    assert_eq!(
        out,
        vec![5.0, 5.0, 5.0],
        "values above max should be clamped to max"
    );
}

// =========================================================================
// Clamp reference: mixed values
// =========================================================================

#[test]
fn test_clamp_reference_mixed() {
    let input = vec![-5.0f32, 0.0, 2.5, 5.0, 10.0];
    let out = clamp_reference(&input, 0.0, 5.0);
    assert_eq!(out, vec![0.0, 0.0, 2.5, 5.0, 5.0]);
}

// =========================================================================
// Clamp reference: negative range
// =========================================================================

#[test]
fn test_clamp_reference_negative_range() {
    let input = vec![-10.0f32, -3.0, 0.0, 5.0];
    let out = clamp_reference(&input, -5.0, -1.0);
    assert_eq!(out, vec![-5.0, -3.0, -1.0, -1.0]);
}

// =========================================================================
// Clamp reference: empty
// =========================================================================

#[test]
fn test_clamp_reference_empty() {
    let out = clamp_reference(&[], 0.0, 1.0);
    assert!(out.is_empty());
}

// =========================================================================
// Clamp reference: edge values equal to bounds
// =========================================================================

#[test]
fn test_clamp_reference_exact_bounds() {
    let input = vec![0.0f32, 1.0];
    let out = clamp_reference(&input, 0.0, 1.0);
    assert_eq!(
        out,
        vec![0.0, 1.0],
        "values exactly at bounds should pass through"
    );
}

// =========================================================================
// Where PTX: valid structure
// =========================================================================

#[test]
fn test_where_ptx_valid_structure() {
    let ptx = generate_where_ptx(1024);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains(".entry ptx_where_f32"), "missing entry point");
}

// =========================================================================
// Where PTX: contains expected instructions
// =========================================================================

#[test]
fn test_where_ptx_contains_instructions() {
    let ptx = generate_where_ptx(512);
    assert!(
        ptx.contains("selp.f32"),
        "must contain selp.f32 for conditional select"
    );
    assert!(
        ptx.contains("setp.ne.u32"),
        "must contain setp.ne for condition check"
    );
    assert!(ptx.contains("ld.global.u32"), "must load condition as u32");
    assert!(ptx.contains("ld.global.f32"), "must load f32 values");
    assert!(ptx.contains("st.global.f32"), "must store f32 output");
}

// =========================================================================
// Where PTX: has 4 parameters (cond, a, b, output) + n
// =========================================================================

#[test]
fn test_where_ptx_has_params() {
    let ptx = generate_where_ptx(256);
    assert!(ptx.contains("param_cond"), "must have cond param");
    assert!(ptx.contains("param_a"), "must have a param");
    assert!(ptx.contains("param_b"), "must have b param");
    assert!(ptx.contains("param_output"), "must have output param");
    assert!(ptx.contains("param_n"), "must have n param");
}

// =========================================================================
// Where PTX: balanced braces
// =========================================================================

#[test]
fn test_where_ptx_balanced_braces() {
    let ptx = generate_where_ptx(1024);
    let open = ptx.matches('{').count();
    let close = ptx.matches('}').count();
    assert_eq!(open, close, "PTX must have balanced braces");
}

// =========================================================================
// Clamp PTX: valid structure
// =========================================================================

#[test]
fn test_clamp_ptx_valid_structure() {
    let ptx = generate_clamp_ptx(1024, -1.0, 1.0);
    assert!(ptx.contains(".version 6.5"), "missing PTX version");
    assert!(ptx.contains(".target sm_70"), "missing SM target");
    assert!(ptx.contains(".entry ptx_clamp_f32"), "missing entry point");
}

// =========================================================================
// Clamp PTX: contains expected instructions
// =========================================================================

#[test]
fn test_clamp_ptx_contains_instructions() {
    let ptx = generate_clamp_ptx(512, 0.0, 1.0);
    assert!(
        ptx.contains("max.f32"),
        "must contain max.f32 for lower bound"
    );
    assert!(
        ptx.contains("min.f32"),
        "must contain min.f32 for upper bound"
    );
    assert!(ptx.contains("ld.global.f32"), "must load f32 input");
    assert!(ptx.contains("st.global.f32"), "must store f32 output");
}

// =========================================================================
// Clamp PTX: balanced braces
// =========================================================================

#[test]
fn test_clamp_ptx_balanced_braces() {
    let ptx = generate_clamp_ptx(1024, -1.0, 1.0);
    let open = ptx.matches('{').count();
    let close = ptx.matches('}').count();
    assert_eq!(open, close, "PTX must have balanced braces");
}

// =========================================================================
// Clamp PTX: different bounds produce different output
// =========================================================================

#[test]
fn test_clamp_ptx_different_bounds_differ() {
    let ptx_a = generate_clamp_ptx(256, 0.0, 1.0);
    let ptx_b = generate_clamp_ptx(256, -1.0, 2.0);
    assert_ne!(
        ptx_a, ptx_b,
        "different clamp bounds must produce different PTX"
    );
}

// =========================================================================
// Launch config
// =========================================================================

#[test]
fn test_where_launch_config() {
    let lc = ptx_where_launch_config(1024);
    assert_eq!(lc.grid.x, 4); // 1024 / 256
    assert_eq!(lc.block.x, 256);
    assert_eq!(lc.shared_mem_bytes, 0);
}

#[test]
fn test_where_launch_config_non_multiple() {
    let lc = ptx_where_launch_config(300);
    assert_eq!(lc.grid.x, 2); // ceil(300 / 256) = 2
    assert_eq!(lc.block.x, 256);
}

// =========================================================================
// WHERE_BLOCK_SIZE constant
// =========================================================================

#[test]
fn test_where_block_size_constant() {
    assert_eq!(WHERE_BLOCK_SIZE, 256);
}
