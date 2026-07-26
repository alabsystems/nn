// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! CPU matrix multiplication implementations for [`DynTensor`].
//!
//! Supports 2D, 3D, 3D×2D broadcast, 4D batched, and 4D×2D broadcast matmul.
//! Extracted from `dyn_tensor_ops.rs` for file-size compliance.

use crate::dyn_tensor::DynTensor;
use crate::{Result, TensorError};
use ndarray::{ArrayD, ArrayViewD, IxDyn};

/// CPU matmul dispatch by rank combination.
///
/// Matmul always accumulates in f32 for numerical precision (#1646 D3).
/// bf16/f16 inputs are promoted to f32, result is converted back to lhs dtype.
pub(crate) fn cpu_matmul(lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
    // Promote bf16/f16 to f32, compute, convert back (#1646 D3).
    let lhs_arr = lhs.to_f32_array()?;
    let rhs_arr = rhs.to_f32_array()?;
    let lhs_view = lhs_arr.view();
    let rhs_view = rhs_arr.view();

    let result = match (lhs.rank(), rhs.rank()) {
        (2, 2) => matmul_2d_2d(lhs_view, rhs_view),
        (3, 3) => matmul_3d_3d(lhs_view, rhs_view, lhs.dims(), rhs.dims()),
        (3, 2) => matmul_3d_2d(lhs_view, rhs_view, lhs.dims(), rhs.dims()),
        (4, 4) => matmul_4d_4d(lhs_view, rhs_view, lhs.dims(), rhs.dims()),
        (4, 2) => matmul_4d_2d(lhs_view, rhs_view, lhs.dims(), rhs.dims()),
        _ => Err(TensorError::Unsupported(format!(
            "matmul for ranks ({}, {}) not yet supported",
            lhs.rank(),
            rhs.rank()
        ))),
    }?;
    // Result dtype follows lhs (matching PyTorch convention).
    DynTensor::from_f32_result(result, lhs.dtype())
}

/// Standard 2D matmul: [M, K] × [K, N] → [M, N]
fn matmul_2d_2d(lhs: ArrayViewD<'_, f32>, rhs: ArrayViewD<'_, f32>) -> Result<ArrayD<f32>> {
    let a = lhs.into_dimensionality::<ndarray::Ix2>()?;
    let b = rhs.into_dimensionality::<ndarray::Ix2>()?;
    if a.ncols() != b.nrows() {
        return Err(TensorError::shape_mismatch(
            vec![a.nrows(), a.ncols()],
            vec![b.nrows(), b.ncols()],
        ));
    }
    Ok(a.dot(&b).into_dyn())
}

/// Batched 3D matmul: [B, M, K] × [B, K, N] → [B, M, N]
fn matmul_3d_3d(
    lhs: ArrayViewD<'_, f32>,
    rhs: ArrayViewD<'_, f32>,
    lhs_dims: &[usize],
    rhs_dims: &[usize],
) -> Result<ArrayD<f32>> {
    let batch = lhs_dims[0];
    if rhs_dims[0] != batch {
        return Err(TensorError::shape_mismatch(
            lhs_dims.to_vec(),
            rhs_dims.to_vec(),
        ));
    }
    let (m, k, n) = (lhs_dims[1], lhs_dims[2], rhs_dims[2]);
    if rhs_dims[1] != k {
        return Err(TensorError::shape_mismatch(
            vec![batch, k, n],
            rhs_dims.to_vec(),
        ));
    }
    let mut out = ArrayD::<f32>::zeros(IxDyn(&[batch, m, n]));
    for b in 0..batch {
        let a2 = lhs
            .slice(ndarray::s![b, .., ..])
            .into_dimensionality::<ndarray::Ix2>()?;
        let b2 = rhs
            .slice(ndarray::s![b, .., ..])
            .into_dimensionality::<ndarray::Ix2>()?;
        out.slice_mut(ndarray::s![b, .., ..]).assign(&a2.dot(&b2));
    }
    Ok(out)
}

/// Broadcast 3D×2D matmul: [B, M, K] × [K, N] → [B, M, N]
fn matmul_3d_2d(
    lhs: ArrayViewD<'_, f32>,
    rhs: ArrayViewD<'_, f32>,
    lhs_dims: &[usize],
    rhs_dims: &[usize],
) -> Result<ArrayD<f32>> {
    let (batch, m, k) = (lhs_dims[0], lhs_dims[1], lhs_dims[2]);
    let n = rhs_dims[1];
    if rhs_dims[0] != k {
        return Err(TensorError::shape_mismatch(vec![k, n], rhs_dims.to_vec()));
    }
    let b2 = rhs.into_dimensionality::<ndarray::Ix2>()?;
    let mut out = ArrayD::<f32>::zeros(IxDyn(&[batch, m, n]));
    for b in 0..batch {
        let a2 = lhs
            .slice(ndarray::s![b, .., ..])
            .into_dimensionality::<ndarray::Ix2>()?;
        out.slice_mut(ndarray::s![b, .., ..]).assign(&a2.dot(&b2));
    }
    Ok(out)
}

/// Broadcast 4D×2D matmul: [B, H, M, K] × [K, N] → [B, H, M, N]
///
/// The 2D weight is broadcast across batch and head dimensions, matching
/// PyTorch's `nn.Linear` semantics when applied to 4D attention-head tensors.
fn matmul_4d_2d(
    lhs: ArrayViewD<'_, f32>,
    rhs: ArrayViewD<'_, f32>,
    lhs_dims: &[usize],
    rhs_dims: &[usize],
) -> Result<ArrayD<f32>> {
    let (b0, b1, m, k) = (lhs_dims[0], lhs_dims[1], lhs_dims[2], lhs_dims[3]);
    let n = rhs_dims[1];
    if rhs_dims[0] != k {
        return Err(TensorError::shape_mismatch(vec![k, n], rhs_dims.to_vec()));
    }
    let w2 = rhs.into_dimensionality::<ndarray::Ix2>()?;
    let mut out = ArrayD::<f32>::zeros(IxDyn(&[b0, b1, m, n]));
    for i in 0..b0 {
        for j in 0..b1 {
            let a2 = lhs
                .slice(ndarray::s![i, j, .., ..])
                .into_dimensionality::<ndarray::Ix2>()?;
            out.slice_mut(ndarray::s![i, j, .., ..])
                .assign(&a2.dot(&w2));
        }
    }
    Ok(out)
}

/// Batched 4D matmul: [B, H, M, K] × [B, H, K, N] → [B, H, M, N]
fn matmul_4d_4d(
    lhs: ArrayViewD<'_, f32>,
    rhs: ArrayViewD<'_, f32>,
    lhs_dims: &[usize],
    rhs_dims: &[usize],
) -> Result<ArrayD<f32>> {
    let (b0, b1) = (lhs_dims[0], lhs_dims[1]);
    if rhs_dims[0] != b0 || rhs_dims[1] != b1 {
        return Err(TensorError::shape_mismatch(
            lhs_dims.to_vec(),
            rhs_dims.to_vec(),
        ));
    }
    let (m, k, n) = (lhs_dims[2], lhs_dims[3], rhs_dims[3]);
    if rhs_dims[2] != k {
        return Err(TensorError::shape_mismatch(
            vec![b0, b1, k, n],
            rhs_dims.to_vec(),
        ));
    }
    let mut out = ArrayD::<f32>::zeros(IxDyn(&[b0, b1, m, n]));
    for i in 0..b0 {
        for j in 0..b1 {
            let a2 = lhs
                .slice(ndarray::s![i, j, .., ..])
                .into_dimensionality::<ndarray::Ix2>()?;
            let b2 = rhs
                .slice(ndarray::s![i, j, .., ..])
                .into_dimensionality::<ndarray::Ix2>()?;
            out.slice_mut(ndarray::s![i, j, .., ..])
                .assign(&a2.dot(&b2));
        }
    }
    Ok(out)
}
