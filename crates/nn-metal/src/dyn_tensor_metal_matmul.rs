// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Metal GEMM kernel for DynTensor matmul.
//!
//! Uses the IR-based naive kernel (one GPU thread per output element with
//! sequential K reduction). This is the production dispatch since #1375 —
//! faster than tiled/simdgroup on Apple Silicon unified memory.
//!
//! Issue: #1289, #1375

use nn_core::dyn_tensor::DynTensor;
use nn_core::{Result, TensorError};

use super::MetalTensorData;

impl super::MetalDynBackend {
    /// IR-based matmul: one GPU thread per output element, sequential K reduction.
    ///
    /// Production dispatch since #1375 — faster than tiled/simdgroup on Apple Silicon
    /// unified memory. Also used as baseline for performance benchmarks (AC2 of #1289).
    ///
    /// Supports 2D (`[M,K] @ [K,N]`), 3D (`[B,M,K] @ [B,K,N]`), and
    /// 4D (`[B,H,M,K] @ [B,H,K,N]`) matmul. For 3D×2D (`[B,M,K] @ [K,N]`),
    /// the 2D matrix is broadcast across the batch dimension.
    pub(crate) fn gpu_matmul_naive(lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
        Self::validate_same_float_dtype(lhs, rhs, "gpu_matmul_naive")?;

        let l_shape = lhs.dims();
        let r_shape = rhs.dims();
        let l_ndim = l_shape.len();
        let r_ndim = r_shape.len();

        if l_ndim < 2 || r_ndim < 2 {
            return Err(TensorError::InvalidShape(format!(
                "matmul requires >= 2D, got {l_shape:?} and {r_shape:?}"
            )));
        }

        let k_lhs = l_shape[l_ndim - 1];
        let k_rhs = r_shape[r_ndim - 2];
        if k_lhs != k_rhs {
            return Err(TensorError::InvalidShape(format!(
                "matmul K mismatch: {l_shape:?} (K={k_lhs}) @ {r_shape:?} (K={k_rhs})"
            )));
        }

        let ndim = l_ndim;
        let m = l_shape[ndim - 2];
        let n = r_shape[r_ndim - 1];

        let mut out_shape: Vec<usize> = l_shape[..ndim - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let lhs_data = lhs.gpu_data::<MetalTensorData>()?;
        let rhs_data = rhs.gpu_data::<MetalTensorData>()?;

        let def = crate::kernel_def_cache::get_or_build(
            "matmul",
            &[l_shape, r_shape],
            &[],
            lhs.dtype(),
            || {
                let mut b = nn_dsl::TensorBlockBuilder::new("dyn_matmul");
                let lhs_node = b.add_input("lhs", l_shape);
                let rhs_node = b.add_input("rhs", r_shape);
                let out = b.add_matmul(lhs_node, rhs_node, false, None, &out_shape);
                crate::build_kernel(b, out)
            },
        )?;

        Self::dispatch_def(
            &def,
            &[
                ("lhs", lhs_data.as_gpu_slice()),
                ("rhs", rhs_data.as_gpu_slice()),
            ],
            &out_shape,
            lhs.dtype(),
        )
    }
}
