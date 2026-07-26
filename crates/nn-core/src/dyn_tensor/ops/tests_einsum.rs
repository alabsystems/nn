// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for einsum operation.

use crate::dyn_tensor::ops::einsum::einsum;
use crate::dyn_tensor::test_helpers::{cpu, t1d, t2d, tnd};
use crate::DynTensor;

/// Helper: assert two f32 slices are approximately equal.
fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: actual={a}, expected={e}, diff={}",
            (a - e).abs()
        );
    }
}

// ============================================================================
// 1. Basic contractions
// ============================================================================

// -- Matmul: "ij,jk->ik" ---------------------------------------------------

#[test]
fn test_einsum_matmul_ij_jk_ik() {
    // [2,3] x [3,2] -> [2,2]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
    // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
    assert_close(&flat, &[58.0, 64.0, 139.0, 154.0], 1e-5);
}

#[test]
fn test_einsum_matmul_square() {
    // [3,3] x [3,3] -> [3,3]
    let a = tnd(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
    let b = tnd(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 3]);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[3, 3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Identity times B = B
    assert_close(&flat, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], 1e-5);
}

// -- Batched matmul: "bij,bjk->bik" -----------------------------------------

#[test]
fn test_einsum_batched_matmul_bij_bjk_bik() {
    // [2, 2, 3] x [2, 3, 2] -> [2, 2, 2]
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(a_data, &[2, 2, 3], &cpu()).unwrap();
    let b_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let b = DynTensor::from_vec(b_data, &[2, 3, 2], &cpu()).unwrap();
    let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2]);

    // Verify against DynTensor::matmul for batch 0 and batch 1.
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Batch 0: [[1,2,3],[4,5,6]] x [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    assert_close(&flat[0..4], &[22.0, 28.0, 49.0, 64.0], 1e-5);
    // Batch 1: [[7,8,9],[10,11,12]] x [[7,8],[9,10],[11,12]]
    //   [0,0] = 7*7+8*9+9*11 = 220, [0,1] = 7*8+8*10+9*12 = 244
    //   [1,0] = 10*7+11*9+12*11 = 301, [1,1] = 10*8+11*10+12*12 = 334
    assert_close(&flat[4..8], &[220.0, 244.0, 301.0, 334.0], 1e-5);
}

#[test]
fn test_einsum_batched_matmul_single_batch() {
    // [1, 2, 2] x [1, 2, 2] -> [1, 2, 2]
    let a = tnd(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let b = tnd(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]);
    let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[1, 2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
    assert_close(&flat, &[19.0, 22.0, 43.0, 50.0], 1e-5);
}

// -- Inner product (dot product): "i,i->" -----------------------------------

#[test]
fn test_einsum_inner_product() {
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = einsum("i,i->", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // 1*4 + 2*5 + 3*6 = 32
    assert_close(&flat, &[32.0], 1e-5);
}

#[test]
fn test_einsum_inner_product_single_element() {
    let a = t1d(&[7.0]);
    let b = t1d(&[3.0]);
    let c = einsum("i,i->", &[&a, &b]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[21.0], 1e-5);
}

// -- Outer product: "i,j->ij" -----------------------------------------------

#[test]
fn test_einsum_outer_product() {
    let a = tnd(&[1.0, 2.0, 3.0], &[3]);
    let b = tnd(&[4.0, 5.0], &[2]);
    let c = einsum("i,j->ij", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // [[1*4,1*5],[2*4,2*5],[3*4,3*5]] = [[4,5],[8,10],[12,15]]
    assert_close(&flat, &[4.0, 5.0, 8.0, 10.0, 12.0, 15.0], 1e-5);
}

#[test]
fn test_einsum_outer_product_1x1() {
    let a = tnd(&[3.0], &[1]);
    let b = tnd(&[5.0], &[1]);
    let c = einsum("i,j->ij", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[1, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[15.0], 1e-5);
}

#[test]
fn test_einsum_outer_product_reversed_output() {
    // "i,j->ji" — outer product but with transposed output
    let a = tnd(&[1.0, 2.0], &[2]);
    let b = tnd(&[3.0, 4.0, 5.0], &[3]);
    let c = einsum("i,j->ji", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Transposed outer product: row j, col i = b[j]*a[i]
    // [[3*1,3*2],[4*1,4*2],[5*1,5*2]] = [[3,6],[4,8],[5,10]]
    assert_close(&flat, &[3.0, 6.0, 4.0, 8.0, 5.0, 10.0], 1e-5);
}

// -- Trace: "ii->" -----------------------------------------------------------

#[test]
fn test_einsum_trace_ii() {
    // trace of [[1,2],[3,4]] = 1+4 = 5
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let c = einsum("ii->", &[&a]).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[5.0], 1e-5);
}

#[test]
fn test_einsum_trace_3x3() {
    // trace of [[1,2,3],[4,5,6],[7,8,9]] = 1+5+9 = 15
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let c = einsum("ii->", &[&a]).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[15.0], 1e-5);
}

#[test]
fn test_einsum_trace_1x1() {
    let a = t2d(&[42.0], 1, 1);
    let c = einsum("ii->", &[&a]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[42.0], 1e-5);
}

// -- Transpose: "ij->ji" ----------------------------------------------------

#[test]
fn test_einsum_transpose_ij_ji() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let c = einsum("ij->ji", &[&a]).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Transpose of [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]
    assert_close(&flat, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5);
}

#[test]
fn test_einsum_transpose_square() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let c = einsum("ij->ji", &[&a]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // [[1,2],[3,4]] -> [[1,3],[2,4]]
    assert_close(&flat, &[1.0, 3.0, 2.0, 4.0], 1e-5);
}

#[test]
fn test_einsum_transpose_1x1() {
    let a = t2d(&[7.0], 1, 1);
    let c = einsum("ij->ji", &[&a]).unwrap();
    assert_eq!(c.dims(), &[1, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[7.0], 1e-5);
}

// -- Diagonal extraction: "ii->i" -------------------------------------------

#[test]
fn test_einsum_diagonal() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
    let c = einsum("ii->i", &[&a]).unwrap();
    assert_eq!(c.dims(), &[3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[1.0, 5.0, 9.0], 1e-5);
}

#[test]
fn test_einsum_diagonal_4x4() {
    // 4x4 matrix with known diagonal
    #[rustfmt::skip]
    let data = [
        10.0, 1.0, 2.0, 3.0,
         4.0,20.0, 5.0, 6.0,
         7.0, 8.0,30.0, 9.0,
        11.0,12.0,13.0,40.0,
    ];
    let a = t2d(&data, 4, 4);
    let c = einsum("ii->i", &[&a]).unwrap();
    assert_eq!(c.dims(), &[4]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[10.0, 20.0, 30.0, 40.0], 1e-5);
}

#[test]
fn test_einsum_diagonal_1x1() {
    let a = t2d(&[99.0], 1, 1);
    let c = einsum("ii->i", &[&a]).unwrap();
    assert_eq!(c.dims(), &[1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[99.0], 1e-5);
}

// ============================================================================
// 2. Multi-operand
// ============================================================================

// -- Three-tensor contraction: "ij,jk,kl->il" ------------------------------

#[test]
fn test_einsum_three_tensor_chain() {
    // A=[2,3], B=[3,4], C=[4,2] -> result=[2,2]
    // This is A*B*C via generic path.
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    #[rustfmt::skip]
    let b = t2d(&[
        1.0, 0.0, 0.0, 1.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ], 3, 4);
    #[rustfmt::skip]
    let c_mat = t2d(&[
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        0.0, 0.0,
    ], 4, 2);
    let result = einsum("ij,jk,kl->il", &[&a, &b, &c_mat]).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let flat = result.to_flat_vec::<f32>().unwrap();
    // AB = [[1,2,3,1],[4,5,6,4]]
    // ABC[0,0] = 1*1 + 2*0 + 3*1 + 1*0 = 4
    // ABC[0,1] = 1*0 + 2*1 + 3*1 + 1*0 = 5
    // ABC[1,0] = 4*1 + 5*0 + 6*1 + 4*0 = 10
    // ABC[1,1] = 4*0 + 5*1 + 6*1 + 4*0 = 11
    assert_close(&flat, &[4.0, 5.0, 10.0, 11.0], 1e-5);
}

#[test]
fn test_einsum_three_tensor_all_ones() {
    // All 2x2 identity-like: result should be identity too.
    let eye = t2d(&[1.0, 0.0, 0.0, 1.0], 2, 2);
    let result = einsum("ij,jk,kl->il", &[&eye, &eye, &eye]).unwrap();
    assert_eq!(result.dims(), &[2, 2]);
    let flat = result.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[1.0, 0.0, 0.0, 1.0], 1e-5);
}

// -- Attention-like: "bhqd,bhkd->bhqk" (Q*K^T) ----------------------------

#[test]
fn test_einsum_attention_qk_bhqd_bhkd_bhqk() {
    // batch=1, heads=1, q_len=2, k_len=3, d=4
    // Q: [1,1,2,4], K: [1,1,3,4]
    // Result: Q * K^T -> [1,1,2,3]
    let q_data: Vec<f32> = (1..=8).map(|x| x as f32).collect(); // 1..8
    let q = DynTensor::from_vec(q_data, &[1, 1, 2, 4], &cpu()).unwrap();
    let k_data: Vec<f32> = (1..=12).map(|x| x as f32).collect(); // 1..12
    let k = DynTensor::from_vec(k_data, &[1, 1, 3, 4], &cpu()).unwrap();
    let result = einsum("bhqd,bhkd->bhqk", &[&q, &k]).unwrap();
    assert_eq!(result.dims(), &[1, 1, 2, 3]);
    let flat = result.to_flat_vec::<f32>().unwrap();
    // Q[0,0] = [1,2,3,4], Q[0,1] = [5,6,7,8]
    // K[0,0] = [1,2,3,4], K[0,1] = [5,6,7,8], K[0,2] = [9,10,11,12]
    // [0,0] dot [0,0] = 1+4+9+16 = 30
    // [0,0] dot [0,1] = 5+12+21+32 = 70
    // [0,0] dot [0,2] = 9+20+33+48 = 110
    // [0,1] dot [0,0] = 5+12+21+32 = 70
    // [0,1] dot [0,1] = 25+36+49+64 = 174
    // [0,1] dot [0,2] = 45+60+77+96 = 278
    assert_close(&flat, &[30.0, 70.0, 110.0, 70.0, 174.0, 278.0], 1e-5);
}

#[test]
fn test_einsum_attention_multi_head() {
    // batch=1, heads=2, q_len=2, k_len=2, d=2
    // Q: [1,2,2,2], K: [1,2,2,2]
    let q = tnd(
        &[
            // head 0, q0=[1,0], q1=[0,1]
            1.0, 0.0, 0.0, 1.0, // head 1, q0=[1,1], q1=[2,0]
            1.0, 1.0, 2.0, 0.0,
        ],
        &[1, 2, 2, 2],
    );
    let k = tnd(
        &[
            // head 0, k0=[1,0], k1=[0,1]
            1.0, 0.0, 0.0, 1.0, // head 1, k0=[3,0], k1=[0,3]
            3.0, 0.0, 0.0, 3.0,
        ],
        &[1, 2, 2, 2],
    );
    let result = einsum("bhqd,bhkd->bhqk", &[&q, &k]).unwrap();
    assert_eq!(result.dims(), &[1, 2, 2, 2]);
    let flat = result.to_flat_vec::<f32>().unwrap();
    // Head 0: [[1,0],[0,1]] x [[1,0],[0,1]]^T = [[1,0],[0,1]] (identity)
    // Head 1: [[1,1],[2,0]] x [[3,0],[0,3]]^T = [[3,3],[6,0]]
    assert_close(&flat, &[1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 6.0, 0.0], 1e-5);
}

// ============================================================================
// 3. Broadcasting / batch dimensions
// ============================================================================

// -- Batched element-wise multiply: "bi,bi->bi" -----------------------------

#[test]
fn test_einsum_batched_elementwise_bi_bi_bi() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = tnd(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[2, 3]);
    let c = einsum("bi,bi->bi", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[7.0, 16.0, 27.0, 40.0, 55.0, 72.0], 1e-5);
}

// -- Sum over batch: "bi->i" ------------------------------------------------

#[test]
fn test_einsum_sum_over_batch_bi_i() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = einsum("bi->i", &[&a]).unwrap();
    assert_eq!(c.dims(), &[3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // col 0: 1+4=5, col 1: 2+5=7, col 2: 3+6=9
    assert_close(&flat, &[5.0, 7.0, 9.0], 1e-5);
}

// -- Sum over feature: "bi->b" ----------------------------------------------

#[test]
fn test_einsum_sum_over_feature_bi_b() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let c = einsum("bi->b", &[&a]).unwrap();
    assert_eq!(c.dims(), &[2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // row 0: 1+2+3=6, row 1: 4+5+6=15
    assert_close(&flat, &[6.0, 15.0], 1e-5);
}

// -- Repeat / broadcast: "i->bi" (output index not in input) ----------------

#[test]
fn test_einsum_repeat_i_bi_error() {
    // 'b' appears in output but not in any input, which should error.
    let a = t1d(&[1.0, 2.0, 3.0]);
    let result = einsum("i->bi", &[&a]);
    assert!(
        result.is_err(),
        "output index 'b' not in input should error"
    );
}

// ============================================================================
// 4. Existing tests (retained from original)
// ============================================================================

// -- Sum all: "ij->" ---------------------------------------------------------

#[test]
fn test_einsum_sum_all_ij() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let c = einsum("ij->", &[&a]).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[21.0], 1e-5);
}

// -- Row sum: "ij->i" --------------------------------------------------------

#[test]
fn test_einsum_row_sum_ij_i() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let c = einsum("ij->i", &[&a]).unwrap();
    assert_eq!(c.dims(), &[2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Row 0: 1+2+3=6, Row 1: 4+5+6=15
    assert_close(&flat, &[6.0, 15.0], 1e-5);
}

// -- Column sum: "ij->j" -----------------------------------------------------

#[test]
fn test_einsum_col_sum_ij_j() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let c = einsum("ij->j", &[&a]).unwrap();
    assert_eq!(c.dims(), &[3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Col 0: 1+4=5, Col 1: 2+5=7, Col 2: 3+6=9
    assert_close(&flat, &[5.0, 7.0, 9.0], 1e-5);
}

// -- Implicit output: "ij,jk" = "ij,jk->ik" --------------------------------

#[test]
fn test_einsum_implicit_output() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let c = einsum("ij,jk", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[58.0, 64.0, 139.0, 154.0], 1e-5);
}

// -- Dot product: "i,i->" ---------------------------------------------------

#[test]
fn test_einsum_dot_product() {
    let a = tnd(&[1.0, 2.0, 3.0], &[3]);
    let b = tnd(&[4.0, 5.0, 6.0], &[3]);
    let c = einsum("i,i->", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // 1*4 + 2*5 + 3*6 = 32
    assert_close(&flat, &[32.0], 1e-5);
}

// -- Element-wise multiply: "ij,ij->ij" -------------------------------------

#[test]
fn test_einsum_elementwise_mul() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = einsum("ij,ij->ij", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[5.0, 12.0, 21.0, 32.0], 1e-5);
}

// -- Batch diagonal: "bii->bi" ----------------------------------------------

#[test]
fn test_einsum_batch_diagonal_bii_bi() {
    // Batch of 2, each 3x3 matrix.
    // Batch 0: [[1,2,3],[4,5,6],[7,8,9]]  -> diag = [1,5,9]
    // Batch 1: [[10,11,12],[13,14,15],[16,17,18]] -> diag = [10,14,18]
    let data: Vec<f32> = (1..=18).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(data, &[2, 3, 3], &cpu()).unwrap();
    let c = einsum("bii->bi", &[&a]).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[1.0, 5.0, 9.0, 10.0, 14.0, 18.0], 1e-5);
}

// -- Batch trace: "bii->b" --------------------------------------------------

#[test]
fn test_einsum_batch_trace_bii_b() {
    // Batch of 2, each 3x3 matrix.
    // Batch 0: trace = 1+5+9 = 15
    // Batch 1: trace = 10+14+18 = 42
    let data: Vec<f32> = (1..=18).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(data, &[2, 3, 3], &cpu()).unwrap();
    let c = einsum("bii->b", &[&a]).unwrap();
    assert_eq!(c.dims(), &[2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[15.0, 42.0], 1e-5);
}

// -- Batch outer product: "bi,bj->bij" --------------------------------------

#[test]
fn test_einsum_batch_outer_product_bi_bj_bij() {
    // Batch of 2. a: [2,3], b: [2,2] -> result: [2,3,2]
    let a = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![7.0, 8.0, 9.0, 10.0], &[2, 2], &cpu()).unwrap();
    let c = einsum("bi,bj->bij", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 3, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Batch 0: outer([1,2,3], [7,8]) = [[7,8],[14,16],[21,24]]
    // Batch 1: outer([4,5,6], [9,10]) = [[36,40],[45,50],[54,60]]
    assert_close(
        &flat,
        &[
            7.0, 8.0, 14.0, 16.0, 21.0, 24.0, 36.0, 40.0, 45.0, 50.0, 54.0, 60.0,
        ],
        1e-5,
    );
}

// -- Matrix-vector: "ij,j->i" -----------------------------------------------

#[test]
fn test_einsum_matvec_ij_j_i() {
    // [2,3] x [3] -> [2]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = tnd(&[7.0, 8.0, 9.0], &[3]);
    let c = einsum("ij,j->i", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Row 0: 1*7+2*8+3*9 = 50
    // Row 1: 4*7+5*8+6*9 = 122
    assert_close(&flat, &[50.0, 122.0], 1e-5);
}

// -- Batch matrix-vector: "bij,bj->bi" --------------------------------------

#[test]
fn test_einsum_batch_matvec_bij_bj_bi() {
    // Batch of 2, matrix [2,3], vector [3] per batch -> [2,2]
    let a_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(a_data, &[2, 2, 3], &cpu()).unwrap();
    let b = DynTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();
    let c = einsum("bij,bj->bi", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Batch 0: [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    // Batch 1: [7*4+8*5+9*6, 10*4+11*5+12*6] = [122, 167]
    assert_close(&flat, &[14.0, 32.0, 122.0, 167.0], 1e-5);
}

// -- 3D Hadamard: "ijk,ijk->ijk" --------------------------------------------

#[test]
fn test_einsum_hadamard_3d() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = tnd(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 2, 2]);
    let c = einsum("ijk,ijk->ijk", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0], 1e-5);
}

// -- Matrix-vector with 1x1 edge case: "ij,j->i" ----------------------------

#[test]
fn test_einsum_matvec_1x1() {
    let a = t2d(&[5.0], 1, 1);
    let b = tnd(&[3.0], &[1]);
    let c = einsum("ij,j->i", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[15.0], 1e-5);
}

// -- Batch diagonal with 2x2 matrices: "bii->bi" ----------------------------

#[test]
fn test_einsum_batch_diagonal_2x2() {
    // Batch of 3, each 2x2.
    // [[1,2],[3,4]], [[5,6],[7,8]], [[9,10],[11,12]]
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(data, &[3, 2, 2], &cpu()).unwrap();
    let c = einsum("bii->bi", &[&a]).unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // diag of [[1,2],[3,4]] = [1,4]
    // diag of [[5,6],[7,8]] = [5,8]
    // diag of [[9,10],[11,12]] = [9,12]
    assert_close(&flat, &[1.0, 4.0, 5.0, 8.0, 9.0, 12.0], 1e-5);
}

// -- Single input sum (no contraction across inputs): "ijk->ij" -------------

#[test]
fn test_einsum_reduce_last_dim() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1]);
    let c = einsum("ijk->ij", &[&a]).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Each element is just the value (last dim is size 1).
    assert_close(&flat, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5);
}

// ============================================================================
// 5. Edge cases
// ============================================================================

// -- Single element tensors --------------------------------------------------

#[test]
fn test_einsum_single_element_matmul() {
    // [1,1] x [1,1] -> [1,1]
    let a = t2d(&[3.0], 1, 1);
    let b = t2d(&[7.0], 1, 1);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[1, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[21.0], 1e-5);
}

#[test]
fn test_einsum_single_element_sum() {
    // Scalar sum of a 1x1 tensor.
    let a = t2d(&[42.0], 1, 1);
    let c = einsum("ij->", &[&a]).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[42.0], 1e-5);
}

#[test]
fn test_einsum_single_element_dot() {
    let a = t1d(&[5.0]);
    let b = t1d(&[6.0]);
    let c = einsum("i,i->", &[&a, &b]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[30.0], 1e-5);
}

#[test]
fn test_einsum_single_element_outer() {
    let a = t1d(&[4.0]);
    let b = t1d(&[5.0]);
    let c = einsum("i,j->ij", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[1, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[20.0], 1e-5);
}

// -- Empty notation string (should error) ------------------------------------

#[test]
fn test_einsum_invalid_notation_empty() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(einsum("", &[&a]).is_err());
}

// -- Invalid notation: bad characters ----------------------------------------

#[test]
fn test_einsum_invalid_notation_bad_char() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(einsum("IJ,JK->IK", &[&a]).is_err());
}

#[test]
fn test_einsum_invalid_notation_digits() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(einsum("12->21", &[&a]).is_err());
}

#[test]
fn test_einsum_invalid_notation_special_chars() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(einsum("i+j->ij", &[&a]).is_err());
}

#[test]
fn test_einsum_invalid_output_not_in_input() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(einsum("ij->z", &[&a]).is_err());
}

#[test]
fn test_einsum_invalid_output_bad_char() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(einsum("ij->I", &[&a]).is_err());
}

// -- Mismatched dimensions ---------------------------------------------------

#[test]
fn test_einsum_shape_mismatch() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    // j is 3 in a but 2 in b
    assert!(einsum("ij,jk->ik", &[&a, &b]).is_err());
}

#[test]
fn test_einsum_shape_mismatch_dot() {
    // "i,i->" with different lengths.
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[1.0, 2.0]);
    assert!(einsum("i,i->", &[&a, &b]).is_err());
}

// -- Wrong number of tensors -------------------------------------------------

#[test]
fn test_einsum_wrong_tensor_count() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert!(einsum("ij,jk->ik", &[&a]).is_err());
}

#[test]
fn test_einsum_wrong_tensor_count_too_many() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let c = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    // Notation expects 2 inputs but 3 given.
    assert!(einsum("ij,jk->ik", &[&a, &b, &c]).is_err());
}

// -- Rank mismatch -----------------------------------------------------------

#[test]
fn test_einsum_rank_mismatch() {
    let a = tnd(&[1.0, 2.0, 3.0], &[3]);
    assert!(einsum("ij->i", &[&a]).is_err());
}

#[test]
fn test_einsum_rank_mismatch_3d_as_2d() {
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    assert!(einsum("ij->i", &[&a]).is_err());
}

// -- Very long notation strings ----------------------------------------------

#[test]
fn test_einsum_long_notation_many_indices() {
    // 26 dimensions (a-z), use 1-element dims so it's tractable.
    // "abcdefghij,abcdefghij->abcdefghij" (Hadamard, all 10 dims)
    // Keep it at 5 dims to not be unreasonably slow.
    let data = vec![2.0f32; 1]; // Each dim is size 1
    let a = DynTensor::from_vec(data.clone(), &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let b = DynTensor::from_vec(data, &[1, 1, 1, 1, 1], &cpu()).unwrap();
    let c = einsum("abcde,abcde->abcde", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[1, 1, 1, 1, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[4.0], 1e-5);
}

#[test]
fn test_einsum_notation_with_spaces() {
    // Spaces should be stripped.
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let c = einsum(" ij , jk -> ik ", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[58.0, 64.0, 139.0, 154.0], 1e-5);
}

// ============================================================================
// 6. Numerical accuracy
// ============================================================================

// -- Known values with hand-computed expected results ------------------------

#[test]
fn test_einsum_matmul_hand_computed() {
    // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
    // AB = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //    = [[19, 22], [43, 50]]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[19.0, 22.0, 43.0, 50.0], 1e-5);
}

#[test]
fn test_einsum_matmul_with_zeros() {
    // Multiplying by zero matrix should give all zeros.
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[0.0, 0.0, 0.0, 0.0], 2, 2);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[0.0, 0.0, 0.0, 0.0], 1e-5);
}

#[test]
fn test_einsum_matmul_identity() {
    // A * I = A
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
    let identity = t2d(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 3, 3);
    let c = einsum("ij,jk->ik", &[&a, &identity]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 1e-5);
}

#[test]
fn test_einsum_matmul_with_negatives() {
    // A = [[1, -1], [-1, 1]], B = [[1, -1], [-1, 1]]
    // AB = [[1*1+(-1)*(-1), 1*(-1)+(-1)*1], [(-1)*1+1*(-1), (-1)*(-1)+1*1]]
    //    = [[2, -2], [-2, 2]]
    let a = t2d(&[1.0, -1.0, -1.0, 1.0], 2, 2);
    let b = t2d(&[1.0, -1.0, -1.0, 1.0], 2, 2);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[2.0, -2.0, -2.0, 2.0], 1e-5);
}

#[test]
fn test_einsum_trace_hand_computed() {
    // trace of [[10, 20, 30], [40, 50, 60], [70, 80, 90]] = 10+50+90 = 150
    let a = tnd(
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        &[3, 3],
    );
    let c = einsum("ii->", &[&a]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[150.0], 1e-5);
}

#[test]
fn test_einsum_dot_product_orthogonal() {
    // Orthogonal vectors have zero dot product.
    let a = t1d(&[1.0, 0.0, 0.0]);
    let b = t1d(&[0.0, 1.0, 0.0]);
    let c = einsum("i,i->", &[&a, &b]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[0.0], 1e-5);
}

#[test]
fn test_einsum_outer_product_hand_computed() {
    // a=[2,3], b=[5,7]
    // Outer: [[10,14],[15,21]]
    let a = t1d(&[2.0, 3.0]);
    let b = t1d(&[5.0, 7.0]);
    let c = einsum("i,j->ij", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[10.0, 14.0, 15.0, 21.0], 1e-5);
}

// -- Associativity: (A*B)*C approx A*(B*C) ----------------------------------

#[test]
fn test_einsum_associativity() {
    // A=[2,3], B=[3,4], C=[4,2]
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    #[rustfmt::skip]
    let b = t2d(&[
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
    ], 3, 4);
    #[rustfmt::skip]
    let c_mat = t2d(&[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ], 4, 2);

    // (A*B)*C
    let ab = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let abc_left = einsum("ij,jk->ik", &[&ab, &c_mat]).unwrap();

    // A*(B*C)
    let bc = einsum("ij,jk->ik", &[&b, &c_mat]).unwrap();
    let abc_right = einsum("ij,jk->ik", &[&a, &bc]).unwrap();

    let flat_left = abc_left.to_flat_vec::<f32>().unwrap();
    let flat_right = abc_right.to_flat_vec::<f32>().unwrap();
    assert_eq!(abc_left.dims(), abc_right.dims());
    assert_close(&flat_left, &flat_right, 1e-4);
}

#[test]
fn test_einsum_three_tensor_vs_sequential() {
    // Compare three-tensor einsum with sequential two-tensor einsum.
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c_mat = t2d(&[9.0, 10.0, 11.0, 12.0], 2, 2);

    // Three-tensor einsum
    let abc_direct = einsum("ij,jk,kl->il", &[&a, &b, &c_mat]).unwrap();

    // Sequential: (A*B)*C
    let ab = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let abc_seq = einsum("ij,jk->ik", &[&ab, &c_mat]).unwrap();

    let flat_direct = abc_direct.to_flat_vec::<f32>().unwrap();
    let flat_seq = abc_seq.to_flat_vec::<f32>().unwrap();
    assert_eq!(abc_direct.dims(), abc_seq.dims());
    assert_close(&flat_direct, &flat_seq, 1e-4);
}

// -- Consistency: einsum matmul vs DynTensor::matmul -------------------------

#[test]
fn test_einsum_matmul_matches_dyntensor_matmul() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = t2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);
    let einsum_result = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let matmul_result = a.matmul(&b).unwrap();
    let flat_einsum = einsum_result.to_flat_vec::<f32>().unwrap();
    let flat_matmul = matmul_result.to_flat_vec::<f32>().unwrap();
    assert_close(&flat_einsum, &flat_matmul, 1e-5);
}

// ============================================================================
// 7. Additional patterns
// ============================================================================

// -- Bilinear-like: "ij,ik->jk" (A^T * B without explicit transpose) --------

#[test]
fn test_einsum_bilinear_ij_ik_jk() {
    // A=[3,2], B=[3,2] -> C=[2,2] where C[j,k] = sum_i A[i,j]*B[i,k]
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let b = tnd(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
    let c = einsum("ij,ik->jk", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // C[0,0] = 1*7+3*9+5*11 = 7+27+55 = 89
    // C[0,1] = 1*8+3*10+5*12 = 8+30+60 = 98
    // C[1,0] = 2*7+4*9+6*11 = 14+36+66 = 116
    // C[1,1] = 2*8+4*10+6*12 = 16+40+72 = 128
    assert_close(&flat, &[89.0, 98.0, 116.0, 128.0], 1e-5);
}

// -- Vector-matrix: "j,ij->i" (v * M^T, equivalent to M * v) ---------------

#[test]
fn test_einsum_vecmat_j_ij_i() {
    // v=[3], M=[2,3] -> result=[2]
    // result[i] = sum_j M[i,j] * v[j] (same as matvec but operand order swapped)
    let v = t1d(&[1.0, 2.0, 3.0]);
    let m = t2d(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 2, 3);
    let c = einsum("j,ij->i", &[&v, &m]).unwrap();
    assert_eq!(c.dims(), &[2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // [0]: 4*1+5*2+6*3=32, [1]: 7*1+8*2+9*3=50
    assert_close(&flat, &[32.0, 50.0], 1e-5);
}

// -- Implicit output for element-wise: "ij,ij" -> all indices unique = "ij" --

#[test]
fn test_einsum_implicit_hadamard() {
    // In implicit mode, 'i' appears 2x and 'j' appears 2x, so both are
    // contracted. "ij,ij" implicit -> empty output (scalar sum of products).
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let b = t2d(&[5.0, 6.0, 7.0, 8.0], 2, 2);
    let c = einsum("ij,ij", &[&a, &b]).unwrap();
    // All indices appear 2x -> all contracted -> scalar
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // sum of elementwise product: 1*5+2*6+3*7+4*8 = 5+12+21+32 = 70
    assert_close(&flat, &[70.0], 1e-5);
}

// -- Sum over specific 3D dimensions: "ijk->j" ------------------------------

#[test]
fn test_einsum_sum_3d_to_middle_dim() {
    // [2,3,2] -> [3], summing over i and k
    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(data, &[2, 3, 2], &cpu()).unwrap();
    let c = einsum("ijk->j", &[&a]).unwrap();
    assert_eq!(c.dims(), &[3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // j=0: a[0,0,0]+a[0,0,1]+a[1,0,0]+a[1,0,1] = 1+2+7+8 = 18
    // j=1: a[0,1,0]+a[0,1,1]+a[1,1,0]+a[1,1,1] = 3+4+9+10 = 26
    // j=2: a[0,2,0]+a[0,2,1]+a[1,2,0]+a[1,2,1] = 5+6+11+12 = 34
    assert_close(&flat, &[18.0, 26.0, 34.0], 1e-5);
}

// -- Permutation of higher-rank tensor: "ijkl->jilk" -----------------------

#[test]
fn test_einsum_4d_permutation() {
    // [2,3,1,1] -> [3,2,1,1] (swap first two dims)
    let data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(data, &[2, 3, 1, 1], &cpu()).unwrap();
    let c = einsum("ijkl->jikl", &[&a]).unwrap();
    assert_eq!(c.dims(), &[3, 2, 1, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Original [i,j,0,0]: [0,0]=1, [0,1]=2, [0,2]=3, [1,0]=4, [1,1]=5, [1,2]=6
    // Transposed [j,i,0,0]: [0,0]=1, [0,1]=4, [1,0]=2, [1,1]=5, [2,0]=3, [2,1]=6
    assert_close(&flat, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-5);
}

// -- Contraction to scalar with 3 inputs: "i,i,i->" -------------------------

#[test]
fn test_einsum_triple_dot() {
    // sum_i a[i]*b[i]*c[i]
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c_vec = t1d(&[7.0, 8.0, 9.0]);
    let result = einsum("i,i,i->", &[&a, &b, &c_vec]).unwrap();
    assert_eq!(result.dims(), &[] as &[usize]);
    let flat = result.to_flat_vec::<f32>().unwrap();
    // 1*4*7 + 2*5*8 + 3*6*9 = 28 + 80 + 162 = 270
    assert_close(&flat, &[270.0], 1e-5);
}

// -- Einsum with all-negative values -----------------------------------------

#[test]
fn test_einsum_all_negatives() {
    let a = t2d(&[-1.0, -2.0, -3.0, -4.0], 2, 2);
    let b = t2d(&[-5.0, -6.0, -7.0, -8.0], 2, 2);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    // [[-1*-5+-2*-7, -1*-6+-2*-8], [-3*-5+-4*-7, -3*-6+-4*-8]]
    // = [[5+14, 6+16], [15+28, 18+32]] = [[19,22],[43,50]]
    assert_close(&flat, &[19.0, 22.0, 43.0, 50.0], 1e-5);
}

// -- Einsum with fractional values -------------------------------------------

#[test]
fn test_einsum_fractional_values() {
    let a = t2d(&[0.5, 0.25, 0.125, 0.0625], 2, 2);
    let b = t2d(&[2.0, 4.0, 8.0, 16.0], 2, 2);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    // [0.5*2+0.25*8, 0.5*4+0.25*16] = [3.0, 6.0]
    // [0.125*2+0.0625*8, 0.125*4+0.0625*16] = [0.75, 1.5]
    assert_close(&flat, &[3.0, 6.0, 0.75, 1.5], 1e-5);
}

// -- Double transpose is identity: "ij->ji->ij" ------------------------------

#[test]
fn test_einsum_double_transpose_is_identity() {
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let transposed = einsum("ij->ji", &[&a]).unwrap();
    let double_transposed = einsum("ij->ji", &[&transposed]).unwrap();
    assert_eq!(double_transposed.dims(), &[2, 3]);
    let flat_orig = a.to_flat_vec::<f32>().unwrap();
    let flat_double = double_transposed.to_flat_vec::<f32>().unwrap();
    assert_close(&flat_double, &flat_orig, 1e-5);
}

// -- Trace of outer product: "i,j->ij" then "ii->" -------------------------

#[test]
fn test_einsum_trace_of_outer_product() {
    // outer(a, b) then trace should equal dot(a, b) when a, b same size
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let outer = einsum("i,j->ij", &[&a, &b]).unwrap();
    let trace = einsum("ii->", &[&outer]).unwrap();
    let dot = einsum("i,i->", &[&a, &b]).unwrap();
    let trace_val = trace.to_flat_vec::<f32>().unwrap();
    let dot_val = dot.to_flat_vec::<f32>().unwrap();
    assert_close(&trace_val, &dot_val, 1e-5);
}

// -- Large-ish contraction to stress generic path ----------------------------

#[test]
fn test_einsum_larger_matmul() {
    // [4,8] x [8,5] -> [4,5]
    let a_data: Vec<f32> = (0..32).map(|x| (x as f32) * 0.1).collect();
    let b_data: Vec<f32> = (0..40).map(|x| (x as f32) * 0.1).collect();
    let a = DynTensor::from_vec(a_data, &[4, 8], &cpu()).unwrap();
    let b = DynTensor::from_vec(b_data, &[8, 5], &cpu()).unwrap();
    let c_einsum = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    let c_matmul = a.matmul(&b).unwrap();
    assert_eq!(c_einsum.dims(), &[4, 5]);
    let flat_einsum = c_einsum.to_flat_vec::<f32>().unwrap();
    let flat_matmul = c_matmul.to_flat_vec::<f32>().unwrap();
    assert_close(&flat_einsum, &flat_matmul, 1e-4);
}

// ============================================================================
// 8. 3D contraction patterns (generic path)
// ============================================================================

// -- Contraction: "ijk,ikl->ijl" (contract over k) --------------------------

#[test]
fn test_einsum_contraction_ijk_ikl_ijl() {
    // A=[2,3,4], B=[2,4,5] -> C=[2,3,5]
    // C[i,j,l] = sum_k A[i,j,k] * B[i,k,l]
    // This is a batched matmul where i is batch, j is rows, k is contracted, l is cols.
    let a_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let a = DynTensor::from_vec(a_data, &[2, 3, 4], &cpu()).unwrap();
    let b_data: Vec<f32> = (1..=40).map(|x| x as f32).collect();
    let b = DynTensor::from_vec(b_data, &[2, 4, 5], &cpu()).unwrap();
    let c = einsum("ijk,ikl->ijl", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 3, 5]);

    let flat = c.to_flat_vec::<f32>().unwrap();

    // Batch 0: A[0] = [[1,2,3,4],[5,6,7,8],[9,10,11,12]], B[0] = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]]
    // C[0,0,:] = [1*1+2*6+3*11+4*16, 1*2+2*7+3*12+4*17, 1*3+2*8+3*13+4*18, 1*4+2*9+3*14+4*19, 1*5+2*10+3*15+4*20]
    //          = [1+12+33+64, 2+14+36+68, 3+16+39+72, 4+18+42+76, 5+20+45+80]
    //          = [110, 120, 130, 140, 150]
    assert_close(&flat[0..5], &[110.0, 120.0, 130.0, 140.0, 150.0], 1e-4);
}

// -- Contraction: "ijk,jl->ikl" (contract over j) ---------------------------

#[test]
fn test_einsum_contraction_ijk_jl_ikl() {
    // A=[2,3,2], B=[3,4] -> C=[2,2,4]
    // C[i,k,l] = sum_j A[i,j,k] * B[j,l]
    let a = tnd(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0: [[1,2],[3,4],[5,6]]
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1: [[7,8],[9,10],[11,12]]
        ],
        &[2, 3, 2],
    );
    #[rustfmt::skip]
    let b = tnd(
        &[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        &[3, 4],
    );
    let c = einsum("ijk,jl->ikl", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 2, 4]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // C[0,0,:] = sum_j A[0,j,0]*B[j,:] = 1*[1,0,0,0] + 3*[0,1,0,0] + 5*[0,0,1,0] = [1,3,5,0]
    assert_close(&flat[0..4], &[1.0, 3.0, 5.0, 0.0], 1e-5);
    // C[0,1,:] = sum_j A[0,j,1]*B[j,:] = 2*[1,0,0,0] + 4*[0,1,0,0] + 6*[0,0,1,0] = [2,4,6,0]
    assert_close(&flat[4..8], &[2.0, 4.0, 6.0, 0.0], 1e-5);
}

// -- Full contraction of 3D tensors: "ijk,ijk->" ----------------------------

#[test]
fn test_einsum_full_contraction_3d() {
    // Sum of element-wise products across all dimensions.
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = tnd(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 2, 2]);
    let c = einsum("ijk,ijk->", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // Sum of all elements in a: 1+2+3+4+5+6+7+8 = 36
    assert_close(&flat, &[36.0], 1e-5);
}

// ============================================================================
// 9. EinsumNotation::parse unit tests
// ============================================================================

#[test]
fn test_parse_explicit_notation() {
    use super::EinsumNotation;
    let n = EinsumNotation::parse("ij,jk->ik").unwrap();
    assert_eq!(n.input_subscripts, vec![vec!['i', 'j'], vec!['j', 'k']]);
    assert_eq!(n.output_subscripts, vec!['i', 'k']);
}

#[test]
fn test_parse_implicit_notation_matmul() {
    use super::EinsumNotation;
    // "ij,jk" implicit -> j appears 2x (contracted), i and k appear 1x -> output "ik"
    let n = EinsumNotation::parse("ij,jk").unwrap();
    assert_eq!(n.input_subscripts, vec![vec!['i', 'j'], vec!['j', 'k']]);
    assert_eq!(n.output_subscripts, vec!['i', 'k']);
}

#[test]
fn test_parse_implicit_single_input() {
    use super::EinsumNotation;
    // "ij" implicit -> i appears 1x, j appears 1x -> output "ij" (identity)
    let n = EinsumNotation::parse("ij").unwrap();
    assert_eq!(n.input_subscripts, vec![vec!['i', 'j']]);
    assert_eq!(n.output_subscripts, vec!['i', 'j']);
}

#[test]
fn test_parse_scalar_output() {
    use super::EinsumNotation;
    let n = EinsumNotation::parse("ii->").unwrap();
    assert_eq!(n.input_subscripts, vec![vec!['i', 'i']]);
    assert!(n.output_subscripts.is_empty());
}

#[test]
fn test_parse_spaces_stripped() {
    use super::EinsumNotation;
    let n = EinsumNotation::parse(" i j , j k -> i k ").unwrap();
    assert_eq!(n.input_subscripts, vec![vec!['i', 'j'], vec!['j', 'k']]);
    assert_eq!(n.output_subscripts, vec!['i', 'k']);
}

#[test]
fn test_parse_three_inputs() {
    use super::EinsumNotation;
    let n = EinsumNotation::parse("ij,jk,kl->il").unwrap();
    assert_eq!(n.input_subscripts.len(), 3);
    assert_eq!(n.output_subscripts, vec!['i', 'l']);
}

#[test]
fn test_parse_error_empty() {
    use super::EinsumNotation;
    assert!(EinsumNotation::parse("").is_err());
}

#[test]
fn test_parse_error_uppercase() {
    use super::EinsumNotation;
    assert!(EinsumNotation::parse("IJ->JI").is_err());
}

#[test]
fn test_parse_error_output_not_in_input() {
    use super::EinsumNotation;
    assert!(EinsumNotation::parse("ij->z").is_err());
}

// ============================================================================
// 10. Size-1 dimension edge cases
// ============================================================================

#[test]
fn test_einsum_matmul_size_1_inner_dim() {
    // [2,1] x [1,3] -> [2,3] (inner dimension is 1)
    let a = t2d(&[3.0, 7.0], 2, 1);
    let b = t2d(&[2.0, 4.0, 6.0], 1, 3);
    let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // [3*2, 3*4, 3*6, 7*2, 7*4, 7*6] = [6, 12, 18, 14, 28, 42]
    assert_close(&flat, &[6.0, 12.0, 18.0, 14.0, 28.0, 42.0], 1e-5);
}

#[test]
fn test_einsum_batch_matmul_batch_1() {
    // [1,2,3] x [1,3,2] -> [1,2,2] (batch dim is 1)
    let a = tnd(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let b = tnd(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[1, 3, 2]);
    let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[1, 2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // [[1,2,3],[4,5,6]] x [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    assert_close(&flat, &[58.0, 64.0, 139.0, 154.0], 1e-5);
}

#[test]
fn test_einsum_outer_product_size_1() {
    // [1] x [1] -> [1,1]
    let a = tnd(&[5.0], &[1]);
    let b = tnd(&[7.0], &[1]);
    let c = einsum("i,j->ij", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[1, 1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[35.0], 1e-5);
}

#[test]
fn test_einsum_trace_of_1x1() {
    // Trace of a 1x1 matrix.
    let a = t2d(&[99.0], 1, 1);
    let c = einsum("ii->", &[&a]).unwrap();
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[99.0], 1e-5);
}

#[test]
fn test_einsum_diagonal_of_1x1() {
    // Diagonal of a 1x1 matrix.
    let a = t2d(&[42.0], 1, 1);
    let c = einsum("ii->i", &[&a]).unwrap();
    assert_eq!(c.dims(), &[1]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[42.0], 1e-5);
}

// ============================================================================
// 11. Implicit mode edge cases
// ============================================================================

#[test]
fn test_einsum_implicit_single_input_identity() {
    // "ij" with single input -> all indices unique -> output = "ij" (identity)
    let a = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    let c = einsum("ij", &[&a]).unwrap();
    assert_eq!(c.dims(), &[2, 2]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[1.0, 2.0, 3.0, 4.0], 1e-5);
}

#[test]
fn test_einsum_implicit_dot_product() {
    // "i,i" implicit -> 'i' appears 2x -> contracted -> output = "" (scalar)
    let a = t1d(&[1.0, 2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = einsum("i,i", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[] as &[usize]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // 1*4 + 2*5 + 3*6 = 32
    assert_close(&flat, &[32.0], 1e-5);
}

#[test]
fn test_einsum_implicit_outer_product() {
    // "i,j" implicit -> i appears 1x, j appears 1x -> output = "ij"
    let a = t1d(&[2.0, 3.0]);
    let b = t1d(&[4.0, 5.0, 6.0]);
    let c = einsum("i,j", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 3]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    assert_close(&flat, &[8.0, 10.0, 12.0, 12.0, 15.0, 18.0], 1e-5);
}

// ============================================================================
// 12. Contraction with repeated indices across inputs
// ============================================================================

#[test]
fn test_einsum_contraction_ik_jk_ij() {
    // A=[2,3], B=[4,3] -> C=[2,4]
    // C[i,j] = sum_k A[i,k] * B[j,k]  (A * B^T)
    let a = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let b = tnd(
        &[
            1.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
            1.0, 1.0, 1.0, // row 3
        ],
        &[4, 3],
    );
    let c = einsum("ik,jk->ij", &[&a, &b]).unwrap();
    assert_eq!(c.dims(), &[2, 4]);
    let flat = c.to_flat_vec::<f32>().unwrap();
    // C[0,0] = 1*1+2*0+3*0 = 1
    // C[0,1] = 1*0+2*1+3*0 = 2
    // C[0,2] = 1*0+2*0+3*1 = 3
    // C[0,3] = 1*1+2*1+3*1 = 6
    // C[1,0] = 4*1+5*0+6*0 = 4
    // C[1,1] = 4*0+5*1+6*0 = 5
    // C[1,2] = 4*0+5*0+6*1 = 6
    // C[1,3] = 4*1+5*1+6*1 = 15
    assert_close(&flat, &[1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0], 1e-5);
}
