// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU compare, where_cond, and matmul dispatch methods for [`MetalDynBackend`].
//!
//! Split from `dyn_tensor_metal_ops.rs` (#1544 D8) for 500-line compliance.
//! Contains gpu_matmul (routing to naive/simdgroup), gpu_compare,
//! gpu_compare_tensor, and gpu_where_cond.

use nn_core::dyn_tensor::{CompareOp, DynTensor};
use nn_core::{DType, Device, Result, TensorError};

use nn_dsl::ir::CompareOpKind;
use nn_dsl::TensorBlockBuilder;

use super::kernels::{
    make_compare_scalar_kernel, make_compare_tensor_kernel, make_where_cond_kernel,
};
use super::MetalTensorData;

impl super::MetalDynBackend {
    /// GPU-native matmul dispatch with simdgroup GEMM for all float dtypes.
    ///
    /// Routes to simdgroup kernel for large aligned matrices (f32 and bf16/f16),
    /// naive (dispatch_def) for smaller/non-aligned shapes.
    ///
    /// Simdgroup kernel uses hardware 8×8 matrix multiply-accumulate with 32×32
    /// output tiles. Requires all dims % 8 == 0, M×N ≥ 16,384, K ≥ 128.
    /// F32 uses `simd_gemm_f32` (float buffers), BF16/F16 uses `simd_gemm_f16`
    /// (half buffers, float accumulators for mixed-precision).
    ///
    /// Issue: #1518, #1289, #1294, #1375, #1670
    pub(super) fn gpu_matmul(lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
        let l_shape = lhs.dims();
        let r_shape = rhs.dims();
        let l_ndim = l_shape.len();
        let r_ndim = r_shape.len();

        // Route to simdgroup GEMM for all float dtypes (F32, BF16, F16).
        // F32 uses float-typed kernel, BF16/F16 uses half-typed kernel (#1670).
        // F16/BF16 require minimum occupancy to outperform F32 (#3315).
        let is_float = matches!(lhs.dtype(), DType::F32 | DType::BF16 | DType::F16);
        if l_ndim >= 2 && r_ndim >= 2 && is_float {
            let m = l_shape[l_ndim - 2];
            let k = l_shape[l_ndim - 1];
            let n = r_shape[r_ndim - 1];

            let use_simdgroup = if matches!(lhs.dtype(), DType::F16 | DType::BF16) {
                let batch: usize = if l_ndim > 2 {
                    l_shape[..l_ndim - 2].iter().product()
                } else {
                    1
                };
                super::matmul_simd::should_use_f16_simdgroup(m, k, n, batch)
            } else {
                super::matmul_simd::should_use_simdgroup(m, k, n)
            };

            if use_simdgroup {
                return Self::gpu_matmul_simdgroup(lhs, rhs);
            }
        }

        Self::gpu_matmul_naive(lhs, rhs)
    }

    /// GPU-native element-wise comparison against scalar.
    ///
    /// Returns f32 (0.0/1.0) directly on GPU — no CPU round-trip (#1323).
    /// `where_cond` accepts f32 masks on GPU for the fast path.
    pub(super) fn gpu_compare(x: &DynTensor, op: CompareOp, val: f64) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_compare")?;
        let shape = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;

        let ir_op = match op {
            CompareOp::Eq => CompareOpKind::Eq,
            CompareOp::Ne => CompareOpKind::Ne,
            CompareOp::Ge => CompareOpKind::Ge,
            CompareOp::Gt => CompareOpKind::Gt,
            CompareOp::Lt => CompareOpKind::Lt,
            CompareOp::Le => CompareOpKind::Le,
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_compare: unsupported op {op:?}"
                )))
            }
        };

        // Defense-in-depth: error on unknown variants instead of assigning
        // a wrong kernel name (was `_ => "cmp"` — silent wrong-name catch-all).
        let name = match op {
            CompareOp::Eq => "cmp_eq",
            CompareOp::Ne => "cmp_ne",
            CompareOp::Ge => "cmp_ge",
            CompareOp::Gt => "cmp_gt",
            CompareOp::Lt => "cmp_lt",
            CompareOp::Le => "cmp_le",
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_compare name: unsupported op {op:?}"
                )))
            }
        };

        let def = crate::kernel_def_cache::get_or_build(
            name,
            &[shape],
            &[val.to_bits()],
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new(name);
                let input = b.add_input("data", shape);
                let out = b.add_elementwise(
                    make_compare_scalar_kernel(name, ir_op, val),
                    &[input],
                    shape,
                );
                crate::build_kernel(b, out)
            },
        )?;

        // Return f32 (0.0/1.0) directly on GPU — no CPU round-trip (#1323).
        Self::dispatch_def(&def, &[("data", x_data.as_gpu_slice())], shape, x.dtype())
    }

    /// GPU-native element-wise comparison between two tensors (#1357 AC1).
    ///
    /// Both tensors must be f32 with the same shape. Returns f32 (0.0/1.0)
    /// directly on GPU, matching the scalar compare convention.
    pub(super) fn gpu_compare_tensor(
        lhs: &DynTensor,
        op: CompareOp,
        rhs: &DynTensor,
    ) -> Result<DynTensor> {
        Self::validate_same_float_dtype(lhs, rhs, "gpu_compare_tensor")?;
        let shape = lhs.dims();
        if shape != rhs.dims() {
            return Err(TensorError::shape_mismatch(
                shape.to_vec(),
                rhs.dims().to_vec(),
            ));
        }
        let lhs_data = lhs.gpu_data::<MetalTensorData>()?;
        let rhs_data = rhs.gpu_data::<MetalTensorData>()?;

        let ir_op = match op {
            CompareOp::Eq => CompareOpKind::Eq,
            CompareOp::Ne => CompareOpKind::Ne,
            CompareOp::Ge => CompareOpKind::Ge,
            CompareOp::Gt => CompareOpKind::Gt,
            CompareOp::Lt => CompareOpKind::Lt,
            CompareOp::Le => CompareOpKind::Le,
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_compare_tensor: unsupported op {op:?}"
                )))
            }
        };

        // Defense-in-depth: error on unknown variants instead of assigning
        // a wrong kernel name (was `_ => "cmp_tensor"` — silent wrong-name catch-all).
        let name = match op {
            CompareOp::Eq => "cmp_tensor_eq",
            CompareOp::Ne => "cmp_tensor_ne",
            CompareOp::Ge => "cmp_tensor_ge",
            CompareOp::Gt => "cmp_tensor_gt",
            CompareOp::Lt => "cmp_tensor_lt",
            CompareOp::Le => "cmp_tensor_le",
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_compare_tensor name: unsupported op {op:?}"
                )))
            }
        };

        let def =
            crate::kernel_def_cache::get_or_build(name, &[shape, shape], &[], lhs.dtype(), || {
                let mut b = TensorBlockBuilder::new(name);
                let lhs_node = b.add_input("lhs", shape);
                let rhs_node = b.add_input("rhs", shape);
                let out = b.add_elementwise(
                    make_compare_tensor_kernel(name, ir_op),
                    &[lhs_node, rhs_node],
                    shape,
                );
                crate::build_kernel(b, out)
            })?;

        Self::dispatch_def(
            &def,
            &[
                ("lhs", lhs_data.as_gpu_slice()),
                ("rhs", rhs_data.as_gpu_slice()),
            ],
            shape,
            lhs.dtype(),
        )
    }

    /// GPU-native where_cond: `if mask[i] != 0 { on_true[i] } else { on_false[i] }`.
    ///
    /// Accepts both F32 (0.0/1.0, from `gpu_compare`) and U8 masks (#1323).
    /// F32 masks stay on GPU with zero round-trips. U8 masks are converted
    /// to f32 via CPU round-trip (legacy path).
    pub(super) fn gpu_where_cond(
        mask: &DynTensor,
        on_true: &DynTensor,
        on_false: &DynTensor,
    ) -> Result<DynTensor> {
        // Validate on_true and on_false are float and same dtype.
        Self::validate_same_float_dtype(on_true, on_false, "gpu_where_cond")?;

        if mask.dtype() != DType::U8 && mask.dtype() != DType::F32 {
            return Err(TensorError::Unsupported(format!(
                "gpu_where_cond: mask must be U8 or F32, got {:?}",
                mask.dtype()
            )));
        }

        let shape = on_true.dims();

        // Validate all three tensor shapes are compatible.
        if on_false.dims() != shape {
            return Err(TensorError::shape_mismatch(
                shape.to_vec(),
                on_false.dims().to_vec(),
            ));
        }
        if mask.dims() != shape {
            return Err(TensorError::shape_mismatch(
                shape.to_vec(),
                mask.dims().to_vec(),
            ));
        }

        // F32 mask (from gpu_compare) stays on GPU — zero round-trips (#1323).
        // U8 mask needs CPU round-trip for dtype conversion (legacy path).
        let gpu_mask = if mask.dtype() == DType::F32 {
            mask.clone()
        } else {
            let f32_mask = mask.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
            f32_mask.to_device(&on_true.device())?
        };

        let mask_data = gpu_mask.gpu_data::<MetalTensorData>()?;
        let true_data = on_true.gpu_data::<MetalTensorData>()?;
        let false_data = on_false.gpu_data::<MetalTensorData>()?;

        let def = crate::kernel_def_cache::get_or_build(
            "where_cond",
            &[shape, shape, shape],
            &[],
            on_true.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("where_cond");
                let m_node = b.add_input("mask", shape);
                let t_node = b.add_input("on_true", shape);
                let f_node = b.add_input("on_false", shape);
                let out =
                    b.add_elementwise(make_where_cond_kernel(), &[m_node, t_node, f_node], shape);
                crate::build_kernel(b, out)
            },
        )?;

        Self::dispatch_def(
            &def,
            &[
                ("mask", mask_data.as_gpu_slice()),
                ("on_true", true_data.as_gpu_slice()),
                ("on_false", false_data.as_gpu_slice()),
            ],
            shape,
            on_true.dtype(),
        )
    }
}
