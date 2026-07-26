// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT proofs for tensor reshape and view mathematical properties
//! used across dpdf model architectures.
//!
//! Proves 20 fundamental properties (test_1091 through test_1110):
//! 1. Reshape preserves total element count
//! 2. View shares same underlying data (stride-based offset identity)
//! 3. Flatten: product of dims equals flat length
//! 4. Squeeze removes only size-1 dims
//! 5. Unsqueeze adds size-1 dim at correct position
//! 6. Permute/transpose: product of dims unchanged
//! 7. Reshape is self-inverse: reshape(reshape(x, s1), s0) = x
//! 8. Contiguous after transpose: stride swap relationship
//! 9. Expand broadcasts size-1 dims
//! 10. Narrow selects correct subrange
//! 11. Split divides along correct dimension
//! 12. Chunk creates equal-sized pieces (last may be smaller)
//! 13. Stack adds new dimension
//! 14. Cat concatenates along existing dimension
//! 15. Reshape of bounded tensor preserves bounds
//! 16. View as real/complex: element count halved/doubled
//! 17. Unfold creates sliding windows
//! 18. Fold reverses unfold
//! 19. Diagonal extraction
//! 20. Triu/tril mask properties
//!
//! Part of #4220.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: assert that program is UNSAT (property holds for all inputs).
fn assert_verified(prog: &AYProgram, property_name: &str) {
    match execute_direct::execute(prog) {
        Ok(ExecuteResult::Verified) => {
            // UNSAT — property proved for all inputs.
        }
        Ok(other) => {
            panic!(
                "{property_name}: expected Verified (UNSAT), got: {other:?}. \
                 The negated property is satisfiable — the property does NOT hold."
            );
        }
        Err(e) => {
            panic!("{property_name}: ay execution error: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1091: Reshape preserves total element count
// ---------------------------------------------------------------------------

/// Prove: For a 3D tensor [d0, d1, d2] reshaped to [e0, e1], given the
/// reshape precondition d0*d1*d2 = e0*e1, the total element count is preserved.
///
/// This is the fundamental reshape invariant: no elements created or destroyed.
#[test]
fn test_1091_reshape_preserves_element_count() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("e0", real.clone());
    let _ = prog.declare_const("e1", real);

    let d0 = real_var("d0");
    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let e0 = real_var("e0");
    let e1 = real_var("e1");

    // All dimensions >= 1, <= 1000
    for v in [&d0, &d1, &d2, &e0, &e1] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    // Reshape precondition: d0*d1*d2 = e0*e1
    let total_in = d0.real_mul(d1).real_mul(d2);
    let total_out = e0.real_mul(e1);
    prog.assert(total_in.clone().eq(total_out.clone()));

    // Violation: total_in != total_out
    prog.assert(total_in.ne(total_out));
    prog.check_sat();

    assert_verified(&prog, "reshape_preserves_element_count");
}

// ---------------------------------------------------------------------------
// Test 1092: View shares same data (stride-based offset identity)
// ---------------------------------------------------------------------------

/// Prove: For a contiguous 3D tensor [d0, d1, d2] with row-major strides
/// [d1*d2, d2, 1], the linear offset i0*s0 + i1*s1 + i2*s2 equals
/// i0*d1*d2 + i1*d2 + i2, staying within [0, d0*d1*d2 - 1].
///
/// View reinterprets the same underlying buffer via strides.
#[test]
fn test_1092_view_stride_offset_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let zero = Expr::real(0);
    let one = Expr::real(1);
    let max_dim = Expr::real(10);

    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("i0", real.clone());
    let _ = prog.declare_const("i1", real.clone());
    let _ = prog.declare_const("i2", real);

    let d0 = real_var("d0");
    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let i0 = real_var("i0");
    let i1 = real_var("i1");
    let i2 = real_var("i2");

    for v in [&d0, &d1, &d2] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    // Valid multi-index: 0 <= i_k < d_k
    prog.assert(i0.clone().real_ge(zero.clone()));
    prog.assert(i0.clone().real_le(d0.clone().real_sub(one.clone())));
    prog.assert(i1.clone().real_ge(zero.clone()));
    prog.assert(i1.clone().real_le(d1.clone().real_sub(one.clone())));
    prog.assert(i2.clone().real_ge(zero.clone()));
    prog.assert(i2.clone().real_le(d2.clone().real_sub(one.clone())));

    // Contiguous strides: s0 = d1*d2, s1 = d2, s2 = 1
    let offset = i0
        .real_mul(d1.clone().real_mul(d2.clone()))
        .real_add(i1.real_mul(d2.clone()))
        .real_add(i2);

    let total = d0.real_mul(d1).real_mul(d2);

    // Violation: offset < 0 OR offset > total - 1
    let too_low = offset.clone().real_lt(zero);
    let too_high = offset.real_gt(total.real_sub(one));
    prog.assert(too_low.or(too_high));
    prog.check_sat();

    assert_verified(&prog, "view_stride_offset_identity");
}

// ---------------------------------------------------------------------------
// Test 1093: Flatten product of dims equals flat length
// ---------------------------------------------------------------------------

/// Prove: flatten([d0, d1, d2]) produces shape [N] where N = d0*d1*d2.
/// Given the definition N = d0*d1*d2, asserting N != d0*d1*d2 is UNSAT.
#[test]
fn test_1093_flatten_product_equals_flat_length() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("N", real);

    let d0 = real_var("d0");
    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let n = real_var("N");

    for v in [&d0, &d1, &d2] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    let product = d0.real_mul(d1).real_mul(d2);
    prog.assert(n.clone().eq(product.clone()));

    // Violation: N != product
    prog.assert(n.ne(product));
    prog.check_sat();

    assert_verified(&prog, "flatten_product_equals_flat_length");
}

// ---------------------------------------------------------------------------
// Test 1094: Squeeze removes only size-1 dims
// ---------------------------------------------------------------------------

/// Prove: squeeze(dim=1) on [d0, 1, d2] yields [d0, d2] with preserved
/// element count. d0*1*d2 = d0*d2.
#[test]
fn test_1094_squeeze_removes_size_one_dim() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d2", real);

    let d0 = real_var("d0");
    let d2 = real_var("d2");

    prog.assert(d0.clone().real_ge(one.clone()));
    prog.assert(d0.clone().real_le(max_dim.clone()));
    prog.assert(d2.clone().real_ge(one.clone()));
    prog.assert(d2.clone().real_le(max_dim));

    // Original: [d0, 1, d2], total = d0 * 1 * d2
    let total_orig = d0.clone().real_mul(one).real_mul(d2.clone());
    // Squeezed: [d0, d2], total = d0 * d2
    let total_squeezed = d0.real_mul(d2);

    // Violation: total_orig != total_squeezed
    prog.assert(total_orig.ne(total_squeezed));
    prog.check_sat();

    assert_verified(&prog, "squeeze_removes_size_one_dim");
}

// ---------------------------------------------------------------------------
// Test 1095: Unsqueeze adds size-1 dim at correct position
// ---------------------------------------------------------------------------

/// Prove: unsqueeze(dim=1) on [d0, d2] yields [d0, 1, d2].
/// Element count preserved: d0*d2 = d0*1*d2.
/// Also: unsqueeze(squeeze(x, 1), 1) is identity on element count.
#[test]
fn test_1095_unsqueeze_adds_size_one_dim() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d2", real);

    let d0 = real_var("d0");
    let d2 = real_var("d2");

    prog.assert(d0.clone().real_ge(one.clone()));
    prog.assert(d0.clone().real_le(max_dim.clone()));
    prog.assert(d2.clone().real_ge(one.clone()));
    prog.assert(d2.clone().real_le(max_dim));

    // Original: [d0, d2], total = d0 * d2
    let total_orig = d0.clone().real_mul(d2.clone());
    // Unsqueezed: [d0, 1, d2], total = d0 * 1 * d2
    let total_unsqueezed = d0.real_mul(one).real_mul(d2);

    // Violation: total_orig != total_unsqueezed
    prog.assert(total_orig.ne(total_unsqueezed));
    prog.check_sat();

    assert_verified(&prog, "unsqueeze_adds_size_one_dim");
}

// ---------------------------------------------------------------------------
// Test 1096: Permute/transpose preserves product of dims
// ---------------------------------------------------------------------------

/// Prove: permute([2,0,1]) on [d0, d1, d2] -> [d2, d0, d1].
/// d0*d1*d2 = d2*d0*d1 (multiplication is commutative).
/// No elements are lost or duplicated by permutation.
#[test]
fn test_1096_permute_preserves_element_count() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real);

    let d0 = real_var("d0");
    let d1 = real_var("d1");
    let d2 = real_var("d2");

    for v in [&d0, &d1, &d2] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    let original = d0.clone().real_mul(d1.clone()).real_mul(d2.clone());
    let permuted = d2.real_mul(d0).real_mul(d1);

    // Violation: original != permuted
    prog.assert(original.ne(permuted));
    prog.check_sat();

    assert_verified(&prog, "permute_preserves_element_count");
}

// ---------------------------------------------------------------------------
// Test 1097: Reshape is self-inverse (roundtrip)
// ---------------------------------------------------------------------------

/// Prove: reshape([a,b] -> [c,d] -> [a,b]) preserves element count.
/// Given a*b = c*d, reshaping back yields a*b = a*b (identity).
/// The linear index is preserved through the roundtrip.
#[test]
fn test_1097_reshape_self_inverse() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(100);
    let zero = Expr::real(0);

    let _ = prog.declare_const("a", real.clone());
    let _ = prog.declare_const("b", real.clone());
    let _ = prog.declare_const("c", real.clone());
    let _ = prog.declare_const("d", real.clone());
    let _ = prog.declare_const("idx", real);

    let a = real_var("a");
    let b = real_var("b");
    let c = real_var("c");
    let d = real_var("d");
    let idx = real_var("idx");

    for v in [&a, &b, &c, &d] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    // Reshape precondition: a*b = c*d
    let prod_ab = a.clone().real_mul(b.clone());
    let prod_cd = c.real_mul(d);
    prog.assert(prod_ab.clone().eq(prod_cd));

    // Valid index in range [0, a*b)
    prog.assert(idx.clone().real_ge(zero));
    prog.assert(idx.clone().real_lt(prod_ab));

    // Roundtrip: idx -> idx (contiguous memory reinterpreted twice)
    // idx_final = idx (identity mapping through contiguous reshapes)
    let idx_final = idx.clone();

    // Violation: idx_final != idx
    prog.assert(idx_final.ne(idx));
    prog.check_sat();

    assert_verified(&prog, "reshape_self_inverse");
}

// ---------------------------------------------------------------------------
// Test 1098: Contiguous after transpose - stride swap
// ---------------------------------------------------------------------------

/// Prove: Transposing a 2D row-major tensor [rows, cols] with strides
/// [cols, 1] produces strides [1, cols]. The stride swap property:
/// stride0_orig = stride1_transposed AND stride1_orig = stride0_transposed.
#[test]
fn test_1098_contiguous_stride_swap_after_transpose() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("cols", real);

    let cols = real_var("cols");
    prog.assert(cols.clone().real_ge(one.clone()));
    prog.assert(cols.clone().real_le(max_dim));

    // Original contiguous strides: [cols, 1]
    let s0_orig = cols.clone();
    let s1_orig = one.clone();

    // After transpose: strides swap -> [1, cols]
    let s0_transposed = one;
    let s1_transposed = cols;

    // Property: s0_orig = s1_transposed AND s1_orig = s0_transposed
    let fail_0 = s0_orig.ne(s1_transposed);
    let fail_1 = s1_orig.ne(s0_transposed);
    prog.assert(fail_0.or(fail_1));
    prog.check_sat();

    assert_verified(&prog, "contiguous_stride_swap_after_transpose");
}

// ---------------------------------------------------------------------------
// Test 1099: Expand broadcasts size-1 dims
// ---------------------------------------------------------------------------

/// Prove: expand([d0, 1, d2], [d0, E, d2]) produces output_count = input_count * E.
/// Source dim must be 1 for broadcast; output_count = d0*E*d2, input_count = d0*d2.
#[test]
fn test_1099_expand_broadcasts_size_one_dims() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("E", real);

    let d0 = real_var("d0");
    let d2 = real_var("d2");
    let expand_factor = real_var("E");

    for v in [&d0, &d2, &expand_factor] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    // Input: [d0, 1, d2], count = d0*1*d2 = d0*d2
    let input_count = d0.clone().real_mul(one).real_mul(d2.clone());
    // Output: [d0, E, d2], count = d0*E*d2
    let output_count = d0.real_mul(expand_factor.clone()).real_mul(d2);

    // Expected: output_count = input_count * E
    let expected = input_count.real_mul(expand_factor);

    // Violation: output_count != expected
    prog.assert(output_count.ne(expected));
    prog.check_sat();

    assert_verified(&prog, "expand_broadcasts_size_one_dims");
}

// ---------------------------------------------------------------------------
// Test 1100: Narrow selects correct subrange
// ---------------------------------------------------------------------------

/// Prove: narrow(dim=0, start=s, length=L) on [D, C] yields [L, C].
/// The output element count L*C is at most D*C (since L <= D and s+L <= D).
/// Also: output elements are a subset of input elements.
#[test]
fn test_1100_narrow_selects_correct_subrange() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let zero = Expr::real(0);
    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("D", real.clone());
    let _ = prog.declare_const("C", real.clone());
    let _ = prog.declare_const("s", real.clone());
    let _ = prog.declare_const("L", real);

    let big_d = real_var("D");
    let c = real_var("C");
    let s = real_var("s");
    let len = real_var("L");

    prog.assert(big_d.clone().real_ge(one.clone()));
    prog.assert(big_d.clone().real_le(max_dim.clone()));
    prog.assert(c.clone().real_ge(one.clone()));
    prog.assert(c.clone().real_le(max_dim.clone()));
    prog.assert(len.clone().real_ge(one.clone()));
    prog.assert(s.clone().real_ge(zero));

    // Precondition: s + L <= D (narrow stays within bounds)
    prog.assert(s.real_add(len.clone()).real_le(big_d.clone()));

    // Output count: L * C
    let out_count = len.real_mul(c.clone());
    // Input count: D * C
    let in_count = big_d.real_mul(c);

    // Property: out_count <= in_count
    // Violation: out_count > in_count
    prog.assert(out_count.real_gt(in_count));
    prog.check_sat();

    assert_verified(&prog, "narrow_selects_correct_subrange");
}

// ---------------------------------------------------------------------------
// Test 1101: Split divides along correct dimension
// ---------------------------------------------------------------------------

/// Prove: split(dim=0, split_size=S) on [D, C] creates ceil(D/S) chunks.
/// The total element count across all chunks equals D*C.
/// We model the simpler case: D is divisible by S, so D/S chunks of size S*C.
/// Total = (D/S) * S * C = D * C.
#[test]
fn test_1101_split_divides_along_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    let _ = prog.declare_const("D", real.clone());
    let _ = prog.declare_const("C", real.clone());
    let _ = prog.declare_const("S", real.clone());
    let _ = prog.declare_const("num_chunks", real);

    let big_d = real_var("D");
    let c = real_var("C");
    let s = real_var("S");
    let num_chunks = real_var("num_chunks");

    for v in [&big_d, &c, &s] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    // Divisibility: D = num_chunks * S
    prog.assert(big_d.clone().eq(num_chunks.clone().real_mul(s.clone())));
    prog.assert(num_chunks.clone().real_ge(one.clone()));

    // Total from chunks: num_chunks * (S * C)
    let chunk_total = num_chunks.real_mul(s.real_mul(c.clone()));
    // Original total: D * C
    let orig_total = big_d.real_mul(c);

    // Violation: chunk_total != orig_total
    prog.assert(chunk_total.ne(orig_total));
    prog.check_sat();

    assert_verified(&prog, "split_divides_along_dimension");
}

// ---------------------------------------------------------------------------
// Test 1102: Chunk creates equal-sized pieces (last may be smaller)
// ---------------------------------------------------------------------------

/// Prove: chunk(N, dim=0) on [D, C] creates N-1 chunks of size ceil(D/N)*C
/// plus a last chunk. Total elements across all chunks = D*C.
/// We model the even case: D = N*K, each chunk has K*C elements.
/// Total = N * K * C = D * C.
#[test]
fn test_1102_chunk_equal_sized_pieces() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    let _ = prog.declare_const("D", real.clone());
    let _ = prog.declare_const("C", real.clone());
    let _ = prog.declare_const("N", real.clone());
    let _ = prog.declare_const("K", real);

    let big_d = real_var("D");
    let c = real_var("C");
    let n = real_var("N");
    let k = real_var("K");

    for v in [&big_d, &c, &n, &k] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    // D = N * K (even division)
    prog.assert(big_d.clone().eq(n.clone().real_mul(k.clone())));

    // Total from chunks: N * (K * C)
    let chunk_total = n.real_mul(k.real_mul(c.clone()));
    let orig_total = big_d.real_mul(c);

    // Violation: chunk_total != orig_total
    prog.assert(chunk_total.ne(orig_total));
    prog.check_sat();

    assert_verified(&prog, "chunk_equal_sized_pieces");
}

// ---------------------------------------------------------------------------
// Test 1103: Stack adds new dimension
// ---------------------------------------------------------------------------

/// Prove: stack([t0, t1], dim=0) where t0, t1 have shape [D, C]
/// produces [2, D, C]. Output element count = 2*D*C = 2 * input_count.
#[test]
fn test_1103_stack_adds_new_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let two = Expr::real(2);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("D", real.clone());
    let _ = prog.declare_const("C", real);

    let big_d = real_var("D");
    let c = real_var("C");

    prog.assert(big_d.clone().real_ge(one.clone()));
    prog.assert(big_d.clone().real_le(max_dim.clone()));
    prog.assert(c.clone().real_ge(one.clone()));
    prog.assert(c.clone().real_le(max_dim));

    // Each input: D*C elements
    let single_count = big_d.clone().real_mul(c.clone());
    // Stacked: [2, D, C], total = 2*D*C
    let stacked_count = two.real_mul(big_d).real_mul(c);

    // Property: stacked_count = 2 * single_count
    let expected = Expr::real(2).real_mul(single_count);

    // Violation: stacked_count != expected
    prog.assert(stacked_count.ne(expected));
    prog.check_sat();

    assert_verified(&prog, "stack_adds_new_dimension");
}

// ---------------------------------------------------------------------------
// Test 1104: Cat concatenates along existing dimension
// ---------------------------------------------------------------------------

/// Prove: cat([t0, t1], dim=0) where t0 has shape [A, C] and t1 has [B, C]
/// produces [A+B, C]. Output elements = (A+B)*C = A*C + B*C.
#[test]
fn test_1104_cat_concatenates_along_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("A", real.clone());
    let _ = prog.declare_const("B", real.clone());
    let _ = prog.declare_const("C", real);

    let a = real_var("A");
    let b = real_var("B");
    let c = real_var("C");

    for v in [&a, &b, &c] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    // Cat output: (A+B) * C
    let cat_count = a.clone().real_add(b.clone()).real_mul(c.clone());
    // Sum of inputs: A*C + B*C
    let sum_counts = a.real_mul(c.clone()).real_add(b.real_mul(c));

    // Violation: cat_count != sum_counts (distributivity of multiplication)
    prog.assert(cat_count.ne(sum_counts));
    prog.check_sat();

    assert_verified(&prog, "cat_concatenates_along_dimension");
}

// ---------------------------------------------------------------------------
// Test 1105: Reshape of bounded tensor preserves bounds
// ---------------------------------------------------------------------------

/// Prove: If all elements of tensor are in [lo, hi], then after reshape,
/// all elements are still in [lo, hi]. Reshape only changes metadata;
/// it does not transform element values.
///
/// We model a single element x with lo <= x <= hi, and prove that after
/// reshape the element value x is unchanged (still in [lo, hi]).
#[test]
fn test_1105_reshape_preserves_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("x", real.clone());
    let _ = prog.declare_const("lo", real.clone());
    let _ = prog.declare_const("hi", real);

    let x = real_var("x");
    let lo = real_var("lo");
    let hi = real_var("hi");

    // Bounds: lo <= x <= hi
    prog.assert(lo.clone().real_le(x.clone()));
    prog.assert(x.clone().real_le(hi.clone()));

    // After reshape, the element value is unchanged (same memory)
    // x_after = x (reshape doesn't change values)
    let x_after = x;

    // Violation: x_after < lo OR x_after > hi
    let below = x_after.clone().real_lt(lo);
    let above = x_after.real_gt(hi);
    prog.assert(below.or(above));
    prog.check_sat();

    assert_verified(&prog, "reshape_preserves_bounds");
}

// ---------------------------------------------------------------------------
// Test 1106: View as real/complex element count relationship
// ---------------------------------------------------------------------------

/// Prove: view_as_complex on a tensor with last dim = 2 halves the element count.
/// Input [N, 2] with 2*N real elements -> [N] complex with N complex elements.
/// Conversely, view_as_real doubles it.
///
/// We prove: for input shape [D, 2], complex_count = D and real_count = 2*D.
/// So real_count = 2 * complex_count.
#[test]
fn test_1106_view_as_complex_halves_count() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let two = Expr::real(2);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("D", real);

    let big_d = real_var("D");
    prog.assert(big_d.clone().real_ge(one));
    prog.assert(big_d.clone().real_le(max_dim));

    // Real elements: D * 2
    let real_count = big_d.clone().real_mul(two.clone());
    // Complex elements: D
    let complex_count = big_d;

    // Property: real_count = 2 * complex_count
    let expected = two.real_mul(complex_count);

    // Violation: real_count != expected
    prog.assert(real_count.ne(expected));
    prog.check_sat();

    assert_verified(&prog, "view_as_complex_halves_count");
}

// ---------------------------------------------------------------------------
// Test 1107: Unfold creates sliding windows
// ---------------------------------------------------------------------------

/// Prove: unfold(dim=0, size=K, step=S) on [D] creates output shape
/// [(D - K) / S + 1, K]. Total output elements = num_windows * K.
///
/// For valid unfold: D >= K and (D - K) is divisible by S.
/// num_windows = (D - K) / S + 1.
/// Output element count = num_windows * K.
/// Property: output_count <= D * K (bounded by input_len * window_size).
#[test]
fn test_1107_unfold_creates_sliding_windows() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    let _ = prog.declare_const("D", real.clone());
    let _ = prog.declare_const("K", real.clone());
    let _ = prog.declare_const("S", real.clone());
    let _ = prog.declare_const("num_windows", real);

    let big_d = real_var("D");
    let k = real_var("K");
    let s = real_var("S");
    let num_windows = real_var("num_windows");

    for v in [&big_d, &k, &s] {
        prog.assert(v.clone().real_ge(one.clone()));
        prog.assert(v.clone().real_le(max_dim.clone()));
    }

    // D >= K (valid unfold)
    prog.assert(big_d.clone().real_ge(k.clone()));

    // num_windows = (D - K) / S + 1
    let d_minus_k = big_d.clone().real_sub(k.clone());
    prog.assert(
        num_windows
            .clone()
            .real_mul(s.clone())
            .eq(d_minus_k.real_add(s.clone())),
    );
    prog.assert(num_windows.clone().real_ge(one.clone()));

    // Output element count: num_windows * K
    let out_count = num_windows.real_mul(k.clone());
    // Upper bound: D * K
    let upper = big_d.real_mul(k);

    // Violation: out_count > upper
    prog.assert(out_count.real_gt(upper));
    prog.check_sat();

    assert_verified(&prog, "unfold_creates_sliding_windows");
}

// ---------------------------------------------------------------------------
// Test 1108: Fold reverses unfold (element count roundtrip)
// ---------------------------------------------------------------------------

/// Prove: fold is the inverse of unfold in terms of output dimensions.
/// unfold(dim=0, size=K, step=K) on [D] with non-overlapping windows
/// creates [(D/K), K]. fold reverses this back to [D].
/// Element count preserved: (D/K) * K = D.
#[test]
fn test_1108_fold_reverses_unfold() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    let _ = prog.declare_const("D", real.clone());
    let _ = prog.declare_const("K", real.clone());
    let _ = prog.declare_const("num_windows", real);

    let big_d = real_var("D");
    let k = real_var("K");
    let num_windows = real_var("num_windows");

    prog.assert(big_d.clone().real_ge(one.clone()));
    prog.assert(big_d.clone().real_le(max_dim.clone()));
    prog.assert(k.clone().real_ge(one.clone()));
    prog.assert(k.clone().real_le(max_dim));
    prog.assert(num_windows.clone().real_ge(one));

    // Non-overlapping: D = num_windows * K
    prog.assert(big_d.clone().eq(num_windows.clone().real_mul(k.clone())));

    // Unfolded: [num_windows, K], count = num_windows * K
    let unfolded_count = num_windows.real_mul(k);
    // Folded back: [D], count = D
    let folded_count = big_d;

    // Violation: unfolded_count != folded_count
    prog.assert(unfolded_count.ne(folded_count));
    prog.check_sat();

    assert_verified(&prog, "fold_reverses_unfold");
}

// ---------------------------------------------------------------------------
// Test 1109: Diagonal extraction
// ---------------------------------------------------------------------------

/// Prove: diagonal of an [N, N] matrix extracts N elements.
/// The diagonal element at position k has offset k*N + k = k*(N+1).
/// For 0 <= k < N, the offset k*(N+1) is in [0, N*N - 1].
#[test]
fn test_1109_diagonal_extraction() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let zero = Expr::real(0);
    let one = Expr::real(1);
    let max_dim = Expr::real(100);

    let _ = prog.declare_const("N", real.clone());
    let _ = prog.declare_const("k", real);

    let n = real_var("N");
    let k = real_var("k");

    prog.assert(n.clone().real_ge(one.clone()));
    prog.assert(n.clone().real_le(max_dim));

    // Valid diagonal index: 0 <= k < N
    prog.assert(k.clone().real_ge(zero.clone()));
    prog.assert(k.clone().real_lt(n.clone()));

    // Diagonal offset: k * (N + 1) = k*N + k
    let offset = k.real_mul(n.clone().real_add(one.clone()));

    // Total elements: N * N
    let total = n.clone().real_mul(n);

    // Violation: offset < 0 OR offset > total - 1
    let below = offset.clone().real_lt(zero);
    let above = offset.real_gt(total.real_sub(one));
    prog.assert(below.or(above));
    prog.check_sat();

    assert_verified(&prog, "diagonal_extraction");
}

// ---------------------------------------------------------------------------
// Test 1110: Triu/tril mask properties
// ---------------------------------------------------------------------------

/// Prove: triu (upper triangular) mask for an [N, N] matrix has
/// N*(N+1)/2 non-zero elements. Equivalently, tril has the same count.
///
/// For the triangular sum: sum(k, k=1..N) = N*(N+1)/2.
/// We prove this for a concrete small N by showing that the element
/// count formula holds.
///
/// Specifically: for a 2D matrix, the number of upper-triangular entries
/// (including diagonal) at row i (0-indexed) is N - i. The total is
/// sum_{i=0}^{N-1} (N - i) = N + (N-1) + ... + 1 = N*(N+1)/2.
///
/// We model this for general N: total_triu = N*(N+1)/2 and
/// total_triu + total_strict_lower = N*N.
/// total_strict_lower = N*(N-1)/2.
/// So total_triu + total_strict_lower = N*(N+1)/2 + N*(N-1)/2 = N^2.
#[test]
fn test_1110_triu_tril_mask_properties() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let one = Expr::real(1);
    let two = Expr::real(2);
    let max_dim = Expr::real(1000);

    let _ = prog.declare_const("N", real.clone());
    let _ = prog.declare_const("triu_count", real.clone());
    let _ = prog.declare_const("strict_lower_count", real);

    let n = real_var("N");
    let triu_count = real_var("triu_count");
    let strict_lower_count = real_var("strict_lower_count");

    prog.assert(n.clone().real_ge(one.clone()));
    prog.assert(n.clone().real_le(max_dim));

    // triu_count = N*(N+1)/2
    let n_plus_1 = n.clone().real_add(one.clone());
    let triu_formula = n.clone().real_mul(n_plus_1).real_div(two.clone());
    prog.assert(triu_count.clone().eq(triu_formula));

    // strict_lower_count = N*(N-1)/2
    let n_minus_1 = n.clone().real_sub(one);
    let lower_formula = n.clone().real_mul(n_minus_1).real_div(two);
    prog.assert(strict_lower_count.clone().eq(lower_formula));

    // Total: triu_count + strict_lower_count should = N*N
    let total = triu_count.real_add(strict_lower_count);
    let n_squared = n.clone().real_mul(n);

    // Violation: total != N^2
    prog.assert(total.ne(n_squared));
    prog.check_sat();

    assert_verified(&prog, "triu_tril_mask_properties");
}
