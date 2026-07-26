// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for GEMV, dot product, and outer product PTX kernels.

use super::*;

// -----------------------------------------------------------------------
// GEMV: PTX generation
// -----------------------------------------------------------------------

#[test]
fn test_gemv_ptx_contains_entry_point() {
    let ptx = generate_gemv_ptx(64, 128);
    assert!(
        ptx.contains(".entry gemv_f32"),
        "GEMV PTX must contain kernel entry point"
    );
}

#[test]
fn test_gemv_ptx_has_shared_memory() {
    let ptx = generate_gemv_ptx(32, 64);
    assert!(
        ptx.contains(".shared .align 4 .f32 xs["),
        "GEMV PTX must declare shared memory for x vector"
    );
}

#[test]
fn test_gemv_ptx_has_required_params() {
    let ptx = generate_gemv_ptx(8, 16);
    assert!(ptx.contains("param_A"), "must have A pointer param");
    assert!(ptx.contains("param_x"), "must have x pointer param");
    assert!(ptx.contains("param_y"), "must have y pointer param");
    assert!(ptx.contains("param_M"), "must have M dimension param");
    assert!(ptx.contains("param_N"), "must have N dimension param");
}

#[test]
fn test_gemv_ptx_has_barrier_sync() {
    let ptx = generate_gemv_ptx(16, 32);
    assert!(
        ptx.contains("bar.sync"),
        "GEMV must synchronize after loading x into shared memory"
    );
}

#[test]
fn test_gemv_ptx_has_fma() {
    let ptx = generate_gemv_ptx(4, 4);
    assert!(
        ptx.contains("fma.rn.f32"),
        "GEMV must use fused multiply-add for dot product"
    );
}

#[test]
fn test_gemv_ptx_is_valid_structure() {
    let ptx = generate_gemv_ptx(128, 256);
    assert!(ptx.contains(".version"), "must have PTX version");
    assert!(ptx.contains(".target sm_70"), "must target sm_70");
    assert!(
        ptx.contains(".address_size 64"),
        "must use 64-bit addresses"
    );
    assert!(ptx.contains("ret;"), "must have return instruction");
    assert!(ptx.contains('{'), "must have opening brace");
    assert!(ptx.contains('}'), "must have closing brace");
}

#[test]
fn test_gemv_ptx_dimension_comments() {
    let ptx = generate_gemv_ptx(100, 200);
    assert!(ptx.contains("y[100]") || ptx.contains("100"));
    assert!(ptx.contains("x[200]") || ptx.contains("200"));
}

#[test]
fn test_gemv_ptx_uses_shared_and_global_memory() {
    let ptx = generate_gemv_ptx(32, 64);
    assert!(
        ptx.contains("ld.global.f32"),
        "must load from global memory"
    );
    assert!(ptx.contains("st.global.f32"), "must store to global memory");
    assert!(
        ptx.contains("ld.shared.f32"),
        "must load from shared memory"
    );
    assert!(ptx.contains("st.shared.f32"), "must store to shared memory");
}

// -----------------------------------------------------------------------
// GEMV: CPU reference
// -----------------------------------------------------------------------

#[test]
fn test_gemv_reference_identity_matrix() {
    // Identity matrix * x = x
    let identity = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let x = vec![3.0, 5.0, 7.0];
    let y = gemv_reference(&identity, &x, 3, 3);
    assert_eq!(y, vec![3.0, 5.0, 7.0]);
}

#[test]
fn test_gemv_reference_known_computation() {
    // A = [[1, 2], [3, 4]], x = [5, 6]
    // y[0] = 1*5 + 2*6 = 17
    // y[1] = 3*5 + 4*6 = 39
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let x = vec![5.0, 6.0];
    let y = gemv_reference(&a, &x, 2, 2);
    assert_eq!(y, vec![17.0, 39.0]);
}

#[test]
fn test_gemv_reference_zeros() {
    let a = vec![0.0; 6];
    let x = vec![1.0, 2.0, 3.0];
    let y = gemv_reference(&a, &x, 2, 3);
    assert_eq!(y, vec![0.0, 0.0]);
}

#[test]
fn test_gemv_reference_non_square() {
    // A[2,4] @ x[4] = y[2]
    let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let x = vec![10.0, 20.0, 30.0, 40.0];
    let y = gemv_reference(&a, &x, 2, 4);
    assert_eq!(y, vec![10.0, 40.0]);
}

#[test]
fn test_gemv_reference_single_row() {
    // A[1,3] @ x[3] = y[1]
    let a = vec![2.0, 3.0, 4.0];
    let x = vec![1.0, 1.0, 1.0];
    let y = gemv_reference(&a, &x, 1, 3);
    assert_eq!(y, vec![9.0]);
}

// -----------------------------------------------------------------------
// Dot product: PTX generation
// -----------------------------------------------------------------------

#[test]
fn test_dot_ptx_contains_entry_point() {
    let ptx = generate_dot_ptx(128);
    assert!(
        ptx.contains(".entry dot_f32"),
        "Dot PTX must contain kernel entry point"
    );
}

#[test]
fn test_dot_ptx_has_shared_memory_reduction() {
    let ptx = generate_dot_ptx(256);
    assert!(
        ptx.contains(".shared .align 4 .f32 partial["),
        "Dot PTX must have shared memory for partial sums"
    );
}

#[test]
fn test_dot_ptx_has_tree_reduction() {
    let ptx = generate_dot_ptx(64);
    assert!(
        ptx.contains("DOT_REDUCE:"),
        "Dot PTX must have reduction loop label"
    );
    assert!(
        ptx.contains("shr.u32"),
        "Dot PTX must halve stride via shift"
    );
}

#[test]
fn test_dot_ptx_has_required_params() {
    let ptx = generate_dot_ptx(32);
    assert!(ptx.contains("param_a"), "must have a pointer param");
    assert!(ptx.contains("param_b"), "must have b pointer param");
    assert!(
        ptx.contains("param_result"),
        "must have result pointer param"
    );
    assert!(ptx.contains("param_N"), "must have N dimension param");
}

#[test]
fn test_dot_ptx_is_valid_structure() {
    let ptx = generate_dot_ptx(512);
    assert!(ptx.contains(".version"), "must have PTX version");
    assert!(ptx.contains(".target sm_70"), "must target sm_70");
    assert!(ptx.contains("ret;"), "must have return instruction");
    assert!(ptx.contains("bar.sync"), "must have barrier for reduction");
}

// -----------------------------------------------------------------------
// Dot product: CPU reference
// -----------------------------------------------------------------------

#[test]
fn test_dot_reference_orthogonal_vectors() {
    // Orthogonal: [1, 0] . [0, 1] = 0
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    assert_eq!(dot_reference(&a, &b), 0.0);
}

#[test]
fn test_dot_reference_parallel_vectors() {
    // Parallel: [3, 4] . [3, 4] = 9 + 16 = 25
    let a = vec![3.0, 4.0];
    let b = vec![3.0, 4.0];
    assert_eq!(dot_reference(&a, &b), 25.0);
}

#[test]
fn test_dot_reference_known_values() {
    // [1, 2, 3] . [4, 5, 6] = 4 + 10 + 18 = 32
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    assert_eq!(dot_reference(&a, &b), 32.0);
}

#[test]
fn test_dot_reference_single_element() {
    assert_eq!(dot_reference(&[7.0], &[3.0]), 21.0);
}

#[test]
fn test_dot_reference_zeros() {
    let a = vec![0.0; 5];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(dot_reference(&a, &b), 0.0);
}

#[test]
fn test_dot_reference_negative_values() {
    // [1, -1] . [-1, 1] = -1 + -1 = -2
    let a = vec![1.0, -1.0];
    let b = vec![-1.0, 1.0];
    assert_eq!(dot_reference(&a, &b), -2.0);
}

// -----------------------------------------------------------------------
// Outer product: PTX generation
// -----------------------------------------------------------------------

#[test]
fn test_outer_ptx_contains_entry_point() {
    let ptx = generate_outer_ptx(4, 8);
    assert!(
        ptx.contains(".entry outer_f32"),
        "Outer PTX must contain kernel entry point"
    );
}

#[test]
fn test_outer_ptx_has_required_params() {
    let ptx = generate_outer_ptx(16, 32);
    assert!(ptx.contains("param_a"), "must have a pointer param");
    assert!(ptx.contains("param_b"), "must have b pointer param");
    assert!(ptx.contains("param_C"), "must have C pointer param");
    assert!(ptx.contains("param_M"), "must have M dimension param");
    assert!(ptx.contains("param_N"), "must have N dimension param");
}

#[test]
fn test_outer_ptx_uses_mul_not_fma() {
    let ptx = generate_outer_ptx(4, 4);
    assert!(
        ptx.contains("mul.f32"),
        "Outer product must use plain multiply (no accumulation)"
    );
}

#[test]
fn test_outer_ptx_is_valid_structure() {
    let ptx = generate_outer_ptx(64, 128);
    assert!(ptx.contains(".version"), "must have PTX version");
    assert!(ptx.contains(".target sm_70"), "must target sm_70");
    assert!(ptx.contains("ret;"), "must have return");
    assert!(ptx.contains("ld.global.f32"), "must load from global");
    assert!(ptx.contains("st.global.f32"), "must store to global");
}

#[test]
fn test_outer_ptx_has_2d_thread_indices() {
    let ptx = generate_outer_ptx(8, 8);
    assert!(ptx.contains("%tid.x"), "must use tid.x");
    assert!(ptx.contains("%tid.y"), "must use tid.y");
    assert!(ptx.contains("%ctaid.x"), "must use blockIdx.x");
    assert!(ptx.contains("%ctaid.y"), "must use blockIdx.y");
}

// -----------------------------------------------------------------------
// Outer product: CPU reference
// -----------------------------------------------------------------------

#[test]
fn test_outer_reference_shape_and_values() {
    // a = [1, 2], b = [3, 4, 5]
    // C = [[3, 4, 5], [6, 8, 10]]
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0, 5.0];
    let c = outer_reference(&a, &b);
    assert_eq!(c.len(), 6); // 2 * 3
    assert_eq!(c, vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
}

#[test]
fn test_outer_reference_unit_vectors() {
    // a = [1], b = [1] -> C = [[1]]
    let c = outer_reference(&[1.0], &[1.0]);
    assert_eq!(c, vec![1.0]);
}

#[test]
fn test_outer_reference_zeros() {
    let a = vec![0.0, 0.0];
    let b = vec![1.0, 2.0, 3.0];
    let c = outer_reference(&a, &b);
    assert_eq!(c, vec![0.0; 6]);
}

#[test]
fn test_outer_reference_symmetric() {
    // outer(a, a) should be symmetric
    let a = vec![1.0, 2.0, 3.0];
    let c = outer_reference(&a, &a);
    // C[0,1] == C[1,0], C[0,2] == C[2,0], C[1,2] == C[2,1]
    assert_eq!(c[1], c[3]);
    assert_eq!(c[2], c[2 * 3]);
    assert_eq!(c[3 + 2], c[2 * 3 + 1]);
}

// -----------------------------------------------------------------------
// GEMV_BLOCK_SIZE constant
// -----------------------------------------------------------------------

#[test]
fn test_gemv_block_size_value() {
    assert_eq!(GEMV_BLOCK_SIZE, 256);
}

#[test]
fn test_gemv_block_size_is_power_of_2() {
    assert!(
        (GEMV_BLOCK_SIZE as usize).is_power_of_two(),
        "GEMV_BLOCK_SIZE must be a power of 2 for reduction"
    );
}

// -----------------------------------------------------------------------
// Cross-kernel consistency
// -----------------------------------------------------------------------

#[test]
fn test_gemv_and_dot_agree_single_row() {
    // GEMV with a single row of A is equivalent to a dot product
    let a_row = vec![1.0, 2.0, 3.0, 4.0];
    let x = vec![5.0, 6.0, 7.0, 8.0];
    let y = gemv_reference(&a_row, &x, 1, 4);
    let dot_val = dot_reference(&a_row, &x);
    assert!(
        (y[0] - dot_val).abs() < 1e-6,
        "single-row GEMV must equal dot product: {} vs {}",
        y[0],
        dot_val
    );
}

#[test]
fn test_outer_then_gemv_equals_scaled_b() {
    // If C = outer(a, b) and we compute C @ b, the result should be
    // (a[i] * (b . b)) for each i — i.e., a scalar-multiply of a.
    let a = vec![1.0, 2.0];
    let b = vec![3.0, 4.0];
    let c = outer_reference(&a, &b);
    let result = gemv_reference(&c, &b, 2, 2);
    let b_dot_b = dot_reference(&b, &b); // 9 + 16 = 25
    let expected: Vec<f32> = a.iter().map(|ai| ai * b_dot_b).collect();
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!(
            (r - e).abs() < 1e-4,
            "outer then GEMV: got {r}, expected {e}"
        );
    }
}
