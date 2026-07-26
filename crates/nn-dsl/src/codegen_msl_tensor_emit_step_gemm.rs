// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! MSL emission for GEMM dispatch steps (Linear, MatMul, Simdgroup, Tiled).
//!
//! Extracted from `codegen_msl_tensor_emit_step.rs` to keep that file under
//! the 450-line limit. All functions are `pub(super)` — called from the
//! `emit_step_msl` match in the parent `step` module.

use crate::codegen_msl_tensor::{DispatchStep, TensorMSLCodegenError};

use super::super::complex::{emit_linear_kernel, emit_matmul_kernel};

/// Emit MSL for one of the 6 GEMM dispatch steps.
///
/// Covers naive Linear/MatMul, simdgroup variants, and tiled variants.
pub(super) fn emit_gemm_msl(step: &DispatchStep) -> Result<String, TensorMSLCodegenError> {
    match step {
        DispatchStep::Linear {
            kernel_name,
            dtype,
            in_features,
            out_features,
            bias,
            ..
        } => emit_linear_kernel(
            kernel_name,
            *dtype,
            *in_features,
            *out_features,
            bias.is_some(),
        ),
        DispatchStep::MatMul {
            kernel_name,
            dtype,
            m,
            k,
            n,
            transpose_right,
            broadcast_right,
            scale,
            ..
        } => emit_matmul_kernel(
            kernel_name,
            *dtype,
            *m,
            *k,
            *n,
            *transpose_right,
            *broadcast_right,
            *scale,
        ),
        DispatchStep::SimdgroupLinear(ref p) => {
            super::super::simdgroup::emit_simdgroup_linear_kernel(
                &p.kernel_name,
                p.dtype,
                p.in_features,
                p.out_features,
                p.batch_size,
                p.bias.is_some(),
            )
        }
        DispatchStep::SimdgroupMatMul(ref p) => {
            super::super::simdgroup::emit_simdgroup_matmul_kernel(
                &p.kernel_name,
                p.dtype,
                p.m,
                p.k,
                p.n,
                p.batch_size,
                p.transpose_right,
                p.broadcast_right,
                p.scale,
            )
        }
        DispatchStep::TiledLinear(ref p) => super::super::tiled::emit_tiled_linear_kernel(
            &p.kernel_name,
            p.dtype,
            p.in_features,
            p.out_features,
            p.batch_size,
            p.bias.is_some(),
        ),
        DispatchStep::TiledMatMul(ref p) => super::super::tiled::emit_tiled_matmul_kernel(
            &p.kernel_name,
            p.dtype,
            p.m,
            p.k,
            p.n,
            p.batch_size,
            p.transpose_right,
            p.broadcast_right,
            p.scale,
        ),
        _ => unreachable!("emit_gemm_msl called with non-GEMM step"),
    }
}
