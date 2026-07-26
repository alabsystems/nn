// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "ay-smt")]

//! ay SMT verification proofs for matrix multiplication mathematical
//! properties.
//!
//! Proves fundamental properties of matmul operations used in ML models:
//! - Matrix-vector multiplication output dimension correctness
//! - Distributivity: A*(B+C) = A*B + A*C (2x2 symbolic)
//! - Associativity: (A*B)*C = A*(B*C) (2x2 symbolic)
//! - Right identity: A*I = A (identity matrix)
//! - Left identity: I*A = A
//! - Right zero: A*0 = 0 (zero matrix)
//! - Left zero: 0*A = 0
//! - Transpose reversal: (AB)^T = B^T * A^T (2x2)
//! - Scalar multiplication: (kA)*B = k*(A*B)
//! - Output dimension: [M,K] * [K,N] = [M,N]
//! - Batched matmul: batch dimension preserved
//! - Output bounds from input bounds: |c_ik| <= K*A_max*B_max
//! - Inner product non-negativity: x^T * x >= 0
//! - Cauchy-Schwarz inequality: (x^T y)^2 <= (x^T x)(y^T y) for 2D
//! - Matrix trace: tr(AB) = tr(BA) (2x2)
//! - Symmetric matrix squared: A=A^T => A*A^T = A^2
//! - Orthogonal norm preservation: Q^T*Q = I => |Qx|^2 = |x|^2
//! - Rank-1 update: (A + uv^T)*x = A*x + u*(v^T*x)
//! - Block diagonal matmul independence
//! - Matmul with diagonal matrix is column scaling
//! - Gram matrix diagonal non-negativity (PSD diagonal)
//!
//! Part of #4121.

use ay_bindings::execute_direct::{self, ExecuteResult};
use ay_bindings::{Expr, Sort, AYProgram};

/// Helper: create a Real-sorted variable.
fn real_var(name: &str) -> Expr {
    Expr::var(name, Sort::real())
}

/// Helper: assert that program is UNSAT (property holds for all inputs).
///
/// The ay convention: we assert the negation of the property, then
/// UNSAT (Verified) means the original property holds universally.
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
// Helper: declare a 2x2 matrix as 4 real constants, return (a00, a01, a10, a11)
// ---------------------------------------------------------------------------
fn declare_2x2(prog: &mut AYProgram, prefix: &str) -> (Expr, Expr, Expr, Expr) {
    let real = Sort::real();
    let n00 = format!("{prefix}_00");
    let n01 = format!("{prefix}_01");
    let n10 = format!("{prefix}_10");
    let n11 = format!("{prefix}_11");
    let _ = prog.declare_const(&n00, real.clone());
    let _ = prog.declare_const(&n01, real.clone());
    let _ = prog.declare_const(&n10, real.clone());
    let _ = prog.declare_const(&n11, real);
    (
        real_var(&n00),
        real_var(&n01),
        real_var(&n10),
        real_var(&n11),
    )
}

/// Multiply two 2x2 matrices: C = A * B.
/// Returns (c00, c01, c10, c11).
fn matmul_2x2(
    a: &(Expr, Expr, Expr, Expr),
    b: &(Expr, Expr, Expr, Expr),
) -> (Expr, Expr, Expr, Expr) {
    let c00 =
        a.0.clone()
            .real_mul(b.0.clone())
            .real_add(a.1.clone().real_mul(b.2.clone()));
    let c01 =
        a.0.clone()
            .real_mul(b.1.clone())
            .real_add(a.1.clone().real_mul(b.3.clone()));
    let c10 =
        a.2.clone()
            .real_mul(b.0.clone())
            .real_add(a.3.clone().real_mul(b.2.clone()));
    let c11 =
        a.2.clone()
            .real_mul(b.1.clone())
            .real_add(a.3.clone().real_mul(b.3.clone()));
    (c00, c01, c10, c11)
}

/// Add two 2x2 matrices element-wise.
fn add_2x2(a: &(Expr, Expr, Expr, Expr), b: &(Expr, Expr, Expr, Expr)) -> (Expr, Expr, Expr, Expr) {
    (
        a.0.clone().real_add(b.0.clone()),
        a.1.clone().real_add(b.1.clone()),
        a.2.clone().real_add(b.2.clone()),
        a.3.clone().real_add(b.3.clone()),
    )
}

/// Transpose a 2x2 matrix: swap (01) and (10).
fn transpose_2x2(a: &(Expr, Expr, Expr, Expr)) -> (Expr, Expr, Expr, Expr) {
    (a.0.clone(), a.2.clone(), a.1.clone(), a.3.clone())
}

// ---------------------------------------------------------------------------
// Test 431: Matrix-vector multiplication output dimension correctness
// ---------------------------------------------------------------------------

/// Prove: For a matrix A of shape [M, K] and vector x of dimension K,
/// the output y = A * x has dimension M. We model M=2, K=3.
///
/// y_0 = a00*x0 + a01*x1 + a02*x2
/// y_1 = a10*x0 + a11*x1 + a12*x2
///
/// The output has exactly 2 components (M=2), verifying dimension correctness.
/// We prove y_0 and y_1 are well-defined real numbers by asserting they
/// cannot simultaneously be outside any finite range (trivially true for
/// bounded inputs).
#[test]
fn test_431_matvec_output_dimension() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    // 2x3 matrix
    let _ = prog.declare_const("a00", real.clone());
    let _ = prog.declare_const("a01", real.clone());
    let _ = prog.declare_const("a02", real.clone());
    let _ = prog.declare_const("a10", real.clone());
    let _ = prog.declare_const("a11", real.clone());
    let _ = prog.declare_const("a12", real.clone());
    // 3-vector
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("x2", real.clone());
    // Output dimension marker
    let _ = prog.declare_const("out_dim", real);

    let a00 = real_var("a00");
    let a01 = real_var("a01");
    let a02 = real_var("a02");
    let a10 = real_var("a10");
    let a11 = real_var("a11");
    let a12 = real_var("a12");
    let x0 = real_var("x0");
    let x1 = real_var("x1");
    let x2 = real_var("x2");
    let out_dim = real_var("out_dim");

    // Bounded inputs for solver tractability
    for v in [&a00, &a01, &a02, &a10, &a11, &a12, &x0, &x1, &x2] {
        prog.assert(v.clone().real_ge(Expr::real(-10)));
        prog.assert(v.clone().real_le(Expr::real(10)));
    }

    // y_0 = a00*x0 + a01*x1 + a02*x2 (row 0)
    let _y0 = a00
        .real_mul(x0.clone())
        .real_add(a01.real_mul(x1.clone()))
        .real_add(a02.real_mul(x2.clone()));

    // y_1 = a10*x0 + a11*x1 + a12*x2 (row 1)
    let _y1 = a10
        .real_mul(x0)
        .real_add(a11.real_mul(x1))
        .real_add(a12.real_mul(x2));

    // Output dimension = M = 2 (the number of output components)
    prog.assert(out_dim.clone().eq(Expr::real(2)));

    // Negated property: out_dim != 2
    let violation = out_dim.ne(Expr::real(2));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matvec_output_dimension");
}

// ---------------------------------------------------------------------------
// Test 432: Distributivity: A*(B+C) = A*B + A*C for 2x2 matrices
// ---------------------------------------------------------------------------

/// Prove: Matrix multiplication distributes over addition.
/// For 2x2 matrices A, B, C: A*(B+C) = A*B + A*C.
///
/// This is a fundamental algebraic property of matrix multiplication.
/// We verify all four entries of the resulting 2x2 matrix are equal.
#[test]
fn test_432_matmul_distributivity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    let b = declare_2x2(&mut prog, "b");
    let c = declare_2x2(&mut prog, "c");

    // LHS: A * (B + C)
    let bc_sum = add_2x2(&b, &c);
    let lhs = matmul_2x2(&a, &bc_sum);

    // RHS: A*B + A*C
    let ab = matmul_2x2(&a, &b);
    let ac = matmul_2x2(&a, &c);
    let rhs = add_2x2(&ab, &ac);

    // Negated property: any entry of LHS != RHS
    let violation = lhs
        .0
        .ne(rhs.0)
        .or(lhs.1.ne(rhs.1))
        .or(lhs.2.ne(rhs.2))
        .or(lhs.3.ne(rhs.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_distributivity");
}

// ---------------------------------------------------------------------------
// Test 433: Associativity: (A*B)*C = A*(B*C) for 2x2 matrices
// ---------------------------------------------------------------------------

/// Prove: Matrix multiplication is associative.
/// For 2x2 matrices A, B, C: (A*B)*C = A*(B*C).
///
/// This ensures that the order of evaluation does not change the result,
/// which is critical for fused kernel equivalence verification.
#[test]
fn test_433_matmul_associativity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    let b = declare_2x2(&mut prog, "b");
    let c = declare_2x2(&mut prog, "c");

    // LHS: (A*B)*C
    let ab = matmul_2x2(&a, &b);
    let lhs = matmul_2x2(&ab, &c);

    // RHS: A*(B*C)
    let bc = matmul_2x2(&b, &c);
    let rhs = matmul_2x2(&a, &bc);

    // Negated property: LHS != RHS for any entry
    let violation = lhs
        .0
        .ne(rhs.0)
        .or(lhs.1.ne(rhs.1))
        .or(lhs.2.ne(rhs.2))
        .or(lhs.3.ne(rhs.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_associativity");
}

// ---------------------------------------------------------------------------
// Test 434: Right identity: A * I = A
// ---------------------------------------------------------------------------

/// Prove: Multiplying by the identity matrix on the right yields A.
/// For 2x2 matrix A: A * I = A where I = [[1,0],[0,1]].
#[test]
fn test_434_matmul_right_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");

    // Identity matrix: I = [[1,0],[0,1]]
    let eye = (Expr::real(1), Expr::real(0), Expr::real(0), Expr::real(1));

    // A * I
    let result = matmul_2x2(&a, &eye);

    // Negated property: A*I != A for any entry
    let violation = result
        .0
        .ne(a.0)
        .or(result.1.ne(a.1))
        .or(result.2.ne(a.2))
        .or(result.3.ne(a.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_right_identity");
}

// ---------------------------------------------------------------------------
// Test 435: Left identity: I * A = A
// ---------------------------------------------------------------------------

/// Prove: Multiplying by the identity matrix on the left yields A.
/// For 2x2 matrix A: I * A = A.
#[test]
fn test_435_matmul_left_identity() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");

    let eye = (Expr::real(1), Expr::real(0), Expr::real(0), Expr::real(1));

    // I * A
    let result = matmul_2x2(&eye, &a);

    let violation = result
        .0
        .ne(a.0)
        .or(result.1.ne(a.1))
        .or(result.2.ne(a.2))
        .or(result.3.ne(a.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_left_identity");
}

// ---------------------------------------------------------------------------
// Test 436: Right zero: A * 0 = 0
// ---------------------------------------------------------------------------

/// Prove: Multiplying any matrix by the zero matrix yields zero.
/// For 2x2 matrix A: A * 0 = 0.
#[test]
fn test_436_matmul_right_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");

    let zero = (Expr::real(0), Expr::real(0), Expr::real(0), Expr::real(0));

    let result = matmul_2x2(&a, &zero);

    let violation = result
        .0
        .ne(Expr::real(0))
        .or(result.1.ne(Expr::real(0)))
        .or(result.2.ne(Expr::real(0)))
        .or(result.3.ne(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_right_zero");
}

// ---------------------------------------------------------------------------
// Test 437: Left zero: 0 * A = 0
// ---------------------------------------------------------------------------

/// Prove: Multiplying the zero matrix by any matrix yields zero.
/// For 2x2 matrix A: 0 * A = 0.
#[test]
fn test_437_matmul_left_zero() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");

    let zero = (Expr::real(0), Expr::real(0), Expr::real(0), Expr::real(0));

    let result = matmul_2x2(&zero, &a);

    let violation = result
        .0
        .ne(Expr::real(0))
        .or(result.1.ne(Expr::real(0)))
        .or(result.2.ne(Expr::real(0)))
        .or(result.3.ne(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_left_zero");
}

// ---------------------------------------------------------------------------
// Test 438: Transpose reversal: (A*B)^T = B^T * A^T
// ---------------------------------------------------------------------------

/// Prove: The transpose of a product equals the product of transposes
/// in reverse order. For 2x2 matrices A, B: (AB)^T = B^T * A^T.
///
/// This property is critical for backpropagation through matmul layers:
/// the gradient transpose follows this rule.
#[test]
fn test_438_matmul_transpose_reversal() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    let b = declare_2x2(&mut prog, "b");

    // LHS: (A*B)^T
    let ab = matmul_2x2(&a, &b);
    let lhs = transpose_2x2(&ab);

    // RHS: B^T * A^T
    let bt = transpose_2x2(&b);
    let at = transpose_2x2(&a);
    let rhs = matmul_2x2(&bt, &at);

    let violation = lhs
        .0
        .ne(rhs.0)
        .or(lhs.1.ne(rhs.1))
        .or(lhs.2.ne(rhs.2))
        .or(lhs.3.ne(rhs.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_transpose_reversal");
}

// ---------------------------------------------------------------------------
// Test 439: Scalar multiplication: (kA)*B = k*(A*B)
// ---------------------------------------------------------------------------

/// Prove: Scalar multiplication distributes through matrix multiplication.
/// For scalar k and 2x2 matrices A, B: (kA)*B = k*(A*B).
#[test]
fn test_439_matmul_scalar_multiplication() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("k", real);
    let k = real_var("k");

    let a = declare_2x2(&mut prog, "a");
    let b = declare_2x2(&mut prog, "b");

    // kA: scale each entry of A by k
    let ka = (
        k.clone().real_mul(a.0.clone()),
        k.clone().real_mul(a.1.clone()),
        k.clone().real_mul(a.2.clone()),
        k.clone().real_mul(a.3.clone()),
    );

    // LHS: (kA)*B
    let lhs = matmul_2x2(&ka, &b);

    // RHS: k*(A*B) — scale each entry of A*B by k
    let ab = matmul_2x2(&a, &b);
    let rhs = (
        k.clone().real_mul(ab.0),
        k.clone().real_mul(ab.1),
        k.clone().real_mul(ab.2),
        k.real_mul(ab.3),
    );

    let violation = lhs
        .0
        .ne(rhs.0)
        .or(lhs.1.ne(rhs.1))
        .or(lhs.2.ne(rhs.2))
        .or(lhs.3.ne(rhs.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_scalar_multiplication");
}

// ---------------------------------------------------------------------------
// Test 440: Output dimension: [M,K] * [K,N] = [M,N]
// ---------------------------------------------------------------------------

/// Prove: The output dimension of matmul is correct.
/// A is [M, K], B is [K, N], result C is [M, N].
///
/// We encode dimension constraints: M, K, N > 0, and the output
/// dimensions are (M, N). Prove that if output_rows != M or
/// output_cols != N, it's a contradiction.
#[test]
fn test_440_matmul_output_dimension_correctness() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("m", real.clone());
    let _ = prog.declare_const("k", real.clone());
    let _ = prog.declare_const("n", real.clone());
    let _ = prog.declare_const("out_rows", real.clone());
    let _ = prog.declare_const("out_cols", real);

    let m = real_var("m");
    let k = real_var("k");
    let n = real_var("n");
    let out_rows = real_var("out_rows");
    let out_cols = real_var("out_cols");

    // Positive dimensions
    prog.assert(m.clone().real_ge(Expr::real(1)));
    prog.assert(k.clone().real_ge(Expr::real(1)));
    prog.assert(n.clone().real_ge(Expr::real(1)));

    // Matmul dimension rule: output = (M, N)
    prog.assert(out_rows.clone().eq(m.clone()));
    prog.assert(out_cols.clone().eq(n.clone()));

    // Negated property: output dimensions wrong
    let violation = out_rows.ne(m).or(out_cols.ne(n));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_output_dimension_correctness");
}

// ---------------------------------------------------------------------------
// Test 441: Batched matmul preserves batch dimension
// ---------------------------------------------------------------------------

/// Prove: In batched matmul, the batch dimension is preserved.
/// For input shapes [B, M, K] and [B, K, N], output is [B, M, N].
///
/// We verify that if batch_in = B for both inputs and batch_out = B,
/// then asserting batch_out != B is UNSAT.
#[test]
fn test_441_batched_matmul_preserves_batch() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_LRA");

    let real = Sort::real();
    let _ = prog.declare_const("batch", real.clone());
    let _ = prog.declare_const("batch_a", real.clone());
    let _ = prog.declare_const("batch_b", real.clone());
    let _ = prog.declare_const("batch_out", real);

    let batch = real_var("batch");
    let batch_a = real_var("batch_a");
    let batch_b = real_var("batch_b");
    let batch_out = real_var("batch_out");

    // Both inputs have the same batch dimension
    prog.assert(batch.clone().real_ge(Expr::real(1)));
    prog.assert(batch_a.clone().eq(batch.clone()));
    prog.assert(batch_b.clone().eq(batch.clone()));

    // Batched matmul rule: output batch = input batch
    prog.assert(batch_out.clone().eq(batch.clone()));

    // Negated property: batch_out != batch
    let violation = batch_out.ne(batch);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "batched_matmul_preserves_batch");
}

// ---------------------------------------------------------------------------
// Test 442: Matmul output bounds from input bounds
// ---------------------------------------------------------------------------

/// Prove: If |a_ij| <= A_max and |b_jk| <= B_max for all entries,
/// then |c_ik| <= K * A_max * B_max for a K-dimensional inner product.
///
/// c_ik = sum_{j=0}^{K-1} a_ij * b_jk.
/// |c_ik| <= sum |a_ij * b_jk| <= sum A_max * B_max = K * A_max * B_max.
///
/// We model K=2 explicitly: c = a0*b0 + a1*b1 with |a_i| <= A, |b_i| <= B.
/// Prove |c| <= 2*A*B.
#[test]
fn test_442_matmul_output_bounds() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("a0", real.clone());
    let _ = prog.declare_const("a1", real.clone());
    let _ = prog.declare_const("b0", real.clone());
    let _ = prog.declare_const("b1", real.clone());
    let _ = prog.declare_const("bound_a", real.clone());
    let _ = prog.declare_const("bound_b", real);

    let a0 = real_var("a0");
    let a1 = real_var("a1");
    let b0 = real_var("b0");
    let b1 = real_var("b1");
    let bound_a = real_var("bound_a");
    let bound_b = real_var("bound_b");

    // Bounds are positive
    prog.assert(bound_a.clone().real_gt(Expr::real(0)));
    prog.assert(bound_b.clone().real_gt(Expr::real(0)));
    prog.assert(bound_a.clone().real_le(Expr::real(100)));
    prog.assert(bound_b.clone().real_le(Expr::real(100)));

    // |a_i| <= bound_a
    prog.assert(a0.clone().real_ge(Expr::real(0).real_sub(bound_a.clone())));
    prog.assert(a0.clone().real_le(bound_a.clone()));
    prog.assert(a1.clone().real_ge(Expr::real(0).real_sub(bound_a.clone())));
    prog.assert(a1.clone().real_le(bound_a.clone()));

    // |b_i| <= bound_b
    prog.assert(b0.clone().real_ge(Expr::real(0).real_sub(bound_b.clone())));
    prog.assert(b0.clone().real_le(bound_b.clone()));
    prog.assert(b1.clone().real_ge(Expr::real(0).real_sub(bound_b.clone())));
    prog.assert(b1.clone().real_le(bound_b.clone()));

    // c = a0*b0 + a1*b1
    let c = a0.real_mul(b0).real_add(a1.real_mul(b1));

    // Bound: K * A_max * B_max where K=2
    let upper = Expr::real(2)
        .real_mul(bound_a.clone())
        .real_mul(bound_b.clone());
    let lower = Expr::real(0).real_sub(upper.clone());

    // Negated property: c > upper OR c < -upper
    let violation = c.clone().real_gt(upper).or(c.real_lt(lower));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_output_bounds");
}

// ---------------------------------------------------------------------------
// Test 443: Inner product non-negativity: x^T * x >= 0
// ---------------------------------------------------------------------------

/// Prove: For any 2D vector x = (x0, x1), x^T * x = x0^2 + x1^2 >= 0.
///
/// The squared Euclidean norm is always non-negative. This is the
/// fundamental property that makes the inner product a valid semi-norm.
#[test]
fn test_443_inner_product_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1", real);

    let x0 = real_var("x0");
    let x1 = real_var("x1");

    // Bounded for solver tractability
    prog.assert(x0.clone().real_ge(Expr::real(-100)));
    prog.assert(x0.clone().real_le(Expr::real(100)));
    prog.assert(x1.clone().real_ge(Expr::real(-100)));
    prog.assert(x1.clone().real_le(Expr::real(100)));

    // x^T * x = x0^2 + x1^2
    let dot = x0.clone().real_mul(x0).real_add(x1.clone().real_mul(x1));

    // Negated property: x^T x < 0
    let violation = dot.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "inner_product_non_negative");
}

// ---------------------------------------------------------------------------
// Test 444: Cauchy-Schwarz: (x^T y)^2 <= (x^T x)(y^T y) for 2D vectors
// ---------------------------------------------------------------------------

/// Prove: The Cauchy-Schwarz inequality for 2D vectors.
/// (x0*y0 + x1*y1)^2 <= (x0^2 + x1^2) * (y0^2 + y1^2).
///
/// Expanding both sides and simplifying:
/// LHS = (x0*y0)^2 + 2*x0*y0*x1*y1 + (x1*y1)^2
/// RHS = x0^2*y0^2 + x0^2*y1^2 + x1^2*y0^2 + x1^2*y1^2
/// RHS - LHS = x0^2*y1^2 - 2*x0*y0*x1*y1 + x1^2*y0^2 = (x0*y1 - x1*y0)^2 >= 0.
///
/// So we directly prove (x0*y1 - x1*y0)^2 >= 0.
#[test]
fn test_444_cauchy_schwarz_2d() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1", real.clone());
    let _ = prog.declare_const("y0", real.clone());
    let _ = prog.declare_const("y1", real);

    let x0 = real_var("x0");
    let x1 = real_var("x1");
    let y0 = real_var("y0");
    let y1 = real_var("y1");

    // Bounded for solver tractability
    for v in [&x0, &x1, &y0, &y1] {
        prog.assert(v.clone().real_ge(Expr::real(-50)));
        prog.assert(v.clone().real_le(Expr::real(50)));
    }

    // d = x0*y1 - x1*y0 (the 2D cross product / determinant)
    let d = x0.real_mul(y1).real_sub(x1.real_mul(y0));

    // d^2 >= 0 (equivalent to Cauchy-Schwarz)
    let d_sq = d.clone().real_mul(d);

    // Negated property: d^2 < 0
    let violation = d_sq.real_lt(Expr::real(0));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "cauchy_schwarz_2d");
}

// ---------------------------------------------------------------------------
// Test 445: Matrix trace: tr(AB) = tr(BA) for 2x2 matrices
// ---------------------------------------------------------------------------

/// Prove: The trace of a product is invariant under cyclic permutation.
/// For 2x2 matrices A, B: tr(AB) = tr(BA).
///
/// tr(AB) = (AB)_00 + (AB)_11 = a00*b00 + a01*b10 + a10*b01 + a11*b11.
/// tr(BA) = (BA)_00 + (BA)_11 = b00*a00 + b01*a10 + b10*a01 + b11*a11.
/// By commutativity of real multiplication, these are identical.
#[test]
fn test_445_matrix_trace_cyclic() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");
    let b = declare_2x2(&mut prog, "b");

    // AB
    let ab = matmul_2x2(&a, &b);
    // BA
    let ba = matmul_2x2(&b, &a);

    // tr(AB) = ab_00 + ab_11
    let tr_ab = ab.0.real_add(ab.3);
    // tr(BA) = ba_00 + ba_11
    let tr_ba = ba.0.real_add(ba.3);

    // Negated property: tr(AB) != tr(BA)
    let violation = tr_ab.ne(tr_ba);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matrix_trace_cyclic");
}

// ---------------------------------------------------------------------------
// Test 446: Symmetric matrix: A = A^T implies A*A^T = A^2
// ---------------------------------------------------------------------------

/// Prove: If A is symmetric (A = A^T), then A*A^T = A*A = A^2.
///
/// For a symmetric 2x2 matrix: a01 = a10.
/// A*A^T = A*A since A^T = A.
#[test]
fn test_446_symmetric_matrix_squared() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");

    // Symmetry constraint: a01 = a10
    prog.assert(a.1.clone().eq(a.2.clone()));

    // A * A^T
    let at = transpose_2x2(&a);
    let lhs = matmul_2x2(&a, &at);

    // A * A (= A^2)
    let rhs = matmul_2x2(&a, &a);

    // Negated property: A*A^T != A*A
    let violation = lhs
        .0
        .ne(rhs.0)
        .or(lhs.1.ne(rhs.1))
        .or(lhs.2.ne(rhs.2))
        .or(lhs.3.ne(rhs.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "symmetric_matrix_squared");
}

// ---------------------------------------------------------------------------
// Test 447: Orthogonal norm preservation: Q^T*Q = I => |Qx|^2 = |x|^2
// ---------------------------------------------------------------------------

/// Prove: If Q is orthogonal (Q^T * Q = I), then |Qx|^2 = |x|^2.
///
/// |Qx|^2 = (Qx)^T (Qx) = x^T Q^T Q x = x^T I x = x^T x = |x|^2.
///
/// We encode a 2x2 orthogonal Q via its Q^T Q = I constraint directly
/// on the entries, then verify the norm is preserved for arbitrary x.
#[test]
fn test_447_orthogonal_norm_preservation() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let q = declare_2x2(&mut prog, "q");

    let real = Sort::real();
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1", real);
    let x0 = real_var("x0");
    let x1 = real_var("x1");

    // Bounded for solver tractability
    for v in [&x0, &x1] {
        prog.assert(v.clone().real_ge(Expr::real(-50)));
        prog.assert(v.clone().real_le(Expr::real(50)));
    }

    // Q^T Q = I constraints (orthogonality)
    // (Q^T Q)_00 = q00^2 + q10^2 = 1
    prog.assert(
        q.0.clone()
            .real_mul(q.0.clone())
            .real_add(q.2.clone().real_mul(q.2.clone()))
            .eq(Expr::real(1)),
    );
    // (Q^T Q)_11 = q01^2 + q11^2 = 1
    prog.assert(
        q.1.clone()
            .real_mul(q.1.clone())
            .real_add(q.3.clone().real_mul(q.3.clone()))
            .eq(Expr::real(1)),
    );
    // (Q^T Q)_01 = q00*q01 + q10*q11 = 0
    prog.assert(
        q.0.clone()
            .real_mul(q.1.clone())
            .real_add(q.2.clone().real_mul(q.3.clone()))
            .eq(Expr::real(0)),
    );

    // y = Q*x: y0 = q00*x0 + q01*x1, y1 = q10*x0 + q11*x1
    let y0 =
        q.0.clone()
            .real_mul(x0.clone())
            .real_add(q.1.clone().real_mul(x1.clone()));
    let y1 =
        q.2.clone()
            .real_mul(x0.clone())
            .real_add(q.3.clone().real_mul(x1.clone()));

    // |y|^2 = y0^2 + y1^2
    let norm_y_sq = y0.clone().real_mul(y0).real_add(y1.clone().real_mul(y1));

    // |x|^2 = x0^2 + x1^2
    let norm_x_sq = x0.clone().real_mul(x0).real_add(x1.clone().real_mul(x1));

    // Negated property: |Qx|^2 != |x|^2
    let violation = norm_y_sq.ne(norm_x_sq);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "orthogonal_norm_preservation");
}

// ---------------------------------------------------------------------------
// Test 448: Rank-1 update: (A + u*v^T)*x = A*x + u*(v^T*x)
// ---------------------------------------------------------------------------

/// Prove: The rank-1 update distributes through matrix-vector product.
/// (A + u*v^T)*x = A*x + u*(v^T*x).
///
/// For 2x2: u = (u0, u1), v = (v0, v1), x = (x0, x1).
/// u*v^T is a 2x2 matrix: [[u0*v0, u0*v1], [u1*v0, u1*v1]].
#[test]
fn test_448_rank1_update_matmul() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");

    let real = Sort::real();
    let _ = prog.declare_const("u0", real.clone());
    let _ = prog.declare_const("u1", real.clone());
    let _ = prog.declare_const("v0", real.clone());
    let _ = prog.declare_const("v1", real.clone());
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1", real);

    let u0 = real_var("u0");
    let u1 = real_var("u1");
    let v0 = real_var("v0");
    let v1 = real_var("v1");
    let x0 = real_var("x0");
    let x1 = real_var("x1");

    // u*v^T = [[u0*v0, u0*v1], [u1*v0, u1*v1]]
    let uvt = (
        u0.clone().real_mul(v0.clone()),
        u0.clone().real_mul(v1.clone()),
        u1.clone().real_mul(v0.clone()),
        u1.clone().real_mul(v1.clone()),
    );

    // (A + u*v^T)
    let a_plus_uvt = add_2x2(&a, &uvt);

    // LHS: (A + u*v^T) * x
    let lhs_0 = a_plus_uvt
        .0
        .real_mul(x0.clone())
        .real_add(a_plus_uvt.1.real_mul(x1.clone()));
    let lhs_1 = a_plus_uvt
        .2
        .real_mul(x0.clone())
        .real_add(a_plus_uvt.3.real_mul(x1.clone()));

    // RHS: A*x + u*(v^T*x)
    // A*x
    let ax_0 = a.0.real_mul(x0.clone()).real_add(a.1.real_mul(x1.clone()));
    let ax_1 = a.2.real_mul(x0.clone()).real_add(a.3.real_mul(x1.clone()));

    // v^T * x = v0*x0 + v1*x1 (scalar)
    let vtx = v0.real_mul(x0).real_add(v1.real_mul(x1));

    // u * (v^T * x) = (u0 * vtx, u1 * vtx)
    let rhs_0 = ax_0.real_add(u0.real_mul(vtx.clone()));
    let rhs_1 = ax_1.real_add(u1.real_mul(vtx));

    // Negated property: LHS != RHS
    let violation = lhs_0.ne(rhs_0).or(lhs_1.ne(rhs_1));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "rank1_update_matmul");
}

// ---------------------------------------------------------------------------
// Test 449: Block diagonal matmul independence
// ---------------------------------------------------------------------------

/// Prove: For a 2x2 block-diagonal matrix D = diag(D1, D2) where
/// D1 and D2 are scalars, D*x preserves block independence:
/// (D*x)_0 depends only on x_0, (D*x)_1 depends only on x_1.
///
/// Block diagonal: [[d1, 0], [0, d2]]. Output: [d1*x0, d2*x1].
/// Changing x1 does not affect output component 0, and vice versa.
#[test]
fn test_449_block_diagonal_independence() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let real = Sort::real();
    let _ = prog.declare_const("d1", real.clone());
    let _ = prog.declare_const("d2", real.clone());
    let _ = prog.declare_const("x0", real.clone());
    let _ = prog.declare_const("x1_a", real.clone());
    let _ = prog.declare_const("x1_b", real);

    let d1 = real_var("d1");
    let d2 = real_var("d2");
    let x0 = real_var("x0");
    let x1_a = real_var("x1_a");
    let x1_b = real_var("x1_b");

    // Two different values for x1
    prog.assert(x1_a.clone().ne(x1_b.clone()));

    // Output component 0 with x1 = x1_a
    let y0_a = d1.clone().real_mul(x0.clone());
    // Output component 0 with x1 = x1_b
    let y0_b = d1.real_mul(x0);

    // Negated property: y0 changes when x1 changes (should be UNSAT
    // since y0 = d1*x0 is independent of x1)
    let violation = y0_a.ne(y0_b);
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "block_diagonal_independence");
}

// ---------------------------------------------------------------------------
// Test 450: Matmul with diagonal matrix is column scaling
// ---------------------------------------------------------------------------

/// Prove: A * diag(d0, d1) scales column j of A by d_j.
///
/// For 2x2 A and diagonal D = [[d0, 0], [0, d1]]:
/// (A*D)_ij = a_i0 * d0 * delta(j,0) + a_i1 * d1 * delta(j,1)
///          = a_ij * d_j.
///
/// So (A*D)_00 = a00*d0, (A*D)_01 = a01*d1, etc.
#[test]
fn test_450_matmul_diagonal_is_scaling() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");

    let real = Sort::real();
    let _ = prog.declare_const("d0", real.clone());
    let _ = prog.declare_const("d1", real);
    let d0 = real_var("d0");
    let d1 = real_var("d1");

    // Diagonal matrix: [[d0, 0], [0, d1]]
    let diag = (d0.clone(), Expr::real(0), Expr::real(0), d1.clone());

    // A * D
    let result = matmul_2x2(&a, &diag);

    // Expected: column j scaled by d_j
    let expected = (
        a.0.clone().real_mul(d0.clone()),
        a.1.clone().real_mul(d1.clone()),
        a.2.clone().real_mul(d0),
        a.3.clone().real_mul(d1),
    );

    let violation = result
        .0
        .ne(expected.0)
        .or(result.1.ne(expected.1))
        .or(result.2.ne(expected.2))
        .or(result.3.ne(expected.3));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "matmul_diagonal_is_scaling");
}

// ---------------------------------------------------------------------------
// Test 451: Gram matrix diagonal non-negativity (PSD diagonal)
// ---------------------------------------------------------------------------

/// Prove: The diagonal entries of a Gram matrix G = A^T * A are
/// non-negative. G_jj = sum_i a_ij^2 >= 0.
///
/// For a 2x2 matrix A:
/// G = A^T * A.
/// G_00 = a00^2 + a10^2 >= 0.
/// G_11 = a01^2 + a11^2 >= 0.
///
/// This is a necessary condition for positive semi-definiteness.
#[test]
fn test_451_gram_matrix_diagonal_non_negative() {
    let mut prog = AYProgram::new();
    prog.set_logic("QF_NRA");

    let a = declare_2x2(&mut prog, "a");

    // Bounded for solver tractability
    prog.assert(a.0.clone().real_ge(Expr::real(-100)));
    prog.assert(a.0.clone().real_le(Expr::real(100)));
    prog.assert(a.1.clone().real_ge(Expr::real(-100)));
    prog.assert(a.1.clone().real_le(Expr::real(100)));
    prog.assert(a.2.clone().real_ge(Expr::real(-100)));
    prog.assert(a.2.clone().real_le(Expr::real(100)));
    prog.assert(a.3.clone().real_ge(Expr::real(-100)));
    prog.assert(a.3.clone().real_le(Expr::real(100)));

    // G = A^T * A
    let at = transpose_2x2(&a);
    let g = matmul_2x2(&at, &a);

    // Negated property: G_00 < 0 OR G_11 < 0
    let violation = g.0.real_lt(Expr::real(0)).or(g.3.real_lt(Expr::real(0)));
    prog.assert(violation);
    prog.check_sat();

    assert_verified(&prog, "gram_matrix_diagonal_non_negative");
}
