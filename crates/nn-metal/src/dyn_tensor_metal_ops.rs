// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU op dispatch methods for [`MetalDynBackend`].
//!
//! Extracted from `dyn_tensor_metal.rs` (#1065 D2) for 500-line compliance.
//! Contains the builder-based GPU dispatch implementations for binary ops,
//! unary ops, Silu, reduce, and matmul.

use nn_core::dyn_tensor::{BinaryOp, DynTensor, UnaryOp};
use nn_core::{Result, TensorError};

use nn_dsl::ir::{BinOpKind, UnaryFnKind};
use nn_dsl::TensorBlockBuilder;

use super::helpers::retype_kernel;
use super::kernels::{
    make_atan2_kernel, make_binop_kernel, make_clamp_kernel, make_clamp_max_kernel,
    make_clamp_min_kernel, make_gelu_erf_kernel, make_maximum_kernel, make_minimum_kernel,
    make_neg_kernel, make_sqr_kernel, make_unary_kernel,
};
use super::MetalTensorData;

impl super::MetalDynBackend {
    /// GPU-native binary op dispatch (Add, Sub, Mul, Div).
    pub(super) fn gpu_binary(op: BinaryOp, lhs: &DynTensor, rhs: &DynTensor) -> Result<DynTensor> {
        Self::validate_same_float_dtype(lhs, rhs, "gpu_binary")?;
        let out_shape = Self::broadcast_shape(lhs.dims(), rhs.dims())?;
        let lhs_data = lhs.gpu_data::<MetalTensorData>()?;
        let rhs_data = rhs.gpu_data::<MetalTensorData>()?;
        let stype = super::helpers::scalar_type_for_dtype(lhs.dtype());

        let op_tag = match op {
            BinaryOp::Add => "binary_add",
            BinaryOp::Mul => "binary_mul",
            BinaryOp::Sub => "binary_sub",
            BinaryOp::Div => "binary_div",
            BinaryOp::Maximum => "binary_max",
            BinaryOp::Minimum => "binary_min",
            BinaryOp::Atan2 => "binary_atan2",
            _ => {
                return Err(TensorError::backend_failure(
                    nn_core::BackendDomain::Metal,
                    nn_core::BackendErrorKind::Other,
                    format!("unsupported binary op: {op:?}"),
                ))
            }
        };
        let def = crate::kernel_def_cache::get_or_build(
            op_tag,
            &[lhs.dims(), rhs.dims()],
            &[],
            lhs.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_binary");
                let lhs_node = b.add_input("lhs", lhs.dims());
                let rhs_node = b.add_input("rhs", rhs.dims());

                let lhs_final = if lhs.dims() != out_shape.as_slice() {
                    b.add_broadcast(lhs_node, &out_shape)
                } else {
                    lhs_node
                };
                let rhs_final = if rhs.dims() != out_shape.as_slice() {
                    b.add_broadcast(rhs_node, &out_shape)
                } else {
                    rhs_node
                };

                let out = match op {
                    BinaryOp::Add => b.add_binary_add(lhs_final, rhs_final, &out_shape),
                    BinaryOp::Mul => b.add_binary_mul(lhs_final, rhs_final, &out_shape),
                    BinaryOp::Sub => b.add_elementwise(
                        retype_kernel(make_binop_kernel("sub", BinOpKind::Sub), stype),
                        &[lhs_final, rhs_final],
                        &out_shape,
                    ),
                    BinaryOp::Div => b.add_elementwise(
                        retype_kernel(make_binop_kernel("div", BinOpKind::Div), stype),
                        &[lhs_final, rhs_final],
                        &out_shape,
                    ),
                    BinaryOp::Maximum => b.add_elementwise(
                        retype_kernel(make_maximum_kernel(), stype),
                        &[lhs_final, rhs_final],
                        &out_shape,
                    ),
                    BinaryOp::Minimum => b.add_elementwise(
                        retype_kernel(make_minimum_kernel(), stype),
                        &[lhs_final, rhs_final],
                        &out_shape,
                    ),
                    BinaryOp::Atan2 => b.add_elementwise(
                        retype_kernel(make_atan2_kernel(), stype),
                        &[lhs_final, rhs_final],
                        &out_shape,
                    ),
                    other => {
                        return Err(TensorError::Unsupported(format!(
                            "gpu_binary builder: unsupported op {other:?}"
                        )))
                    }
                };

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

    /// GPU-native unary op dispatch.
    pub(super) fn gpu_unary(op: UnaryOp, x: &DynTensor) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_unary")?;
        let shape = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let stype = super::helpers::scalar_type_for_dtype(x.dtype());

        let op_tag = match op {
            UnaryOp::Relu => "unary_relu",
            UnaryOp::Gelu => "unary_gelu",
            UnaryOp::Sigmoid => "unary_sigmoid",
            UnaryOp::Tanh => "unary_tanh",
            UnaryOp::Exp => "unary_exp",
            UnaryOp::Sqrt => "unary_sqrt",
            UnaryOp::Abs => "unary_abs",
            UnaryOp::Recip => "unary_recip",
            UnaryOp::Sin => "unary_sin",
            UnaryOp::Cos => "unary_cos",
            UnaryOp::Log => "unary_log",
            UnaryOp::Floor => "unary_floor",
            UnaryOp::Round => "unary_round",
            UnaryOp::Fract => "unary_fract",
            UnaryOp::Neg => "unary_neg",
            UnaryOp::Sqr => "unary_sqr",
            UnaryOp::GeluErf => "unary_gelu_erf",
            _ => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_unary: unsupported op {op:?}"
                )))
            }
        };
        let def = crate::kernel_def_cache::get_or_build(op_tag, &[shape], &[], x.dtype(), || {
            let mut b = TensorBlockBuilder::new("dyn_unary");
            let input = b.add_input("data", shape);

            let out = match op {
                UnaryOp::Relu => b.add_relu(input, shape),
                UnaryOp::Gelu => b.add_gelu(input, shape),
                UnaryOp::Sigmoid => b.add_sigmoid(input, shape),
                UnaryOp::Tanh => b.add_tanh(input, shape),
                UnaryOp::Exp => b.add_elementwise(
                    retype_kernel(make_unary_kernel("exp", UnaryFnKind::Exp), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Sqrt => b.add_elementwise(
                    retype_kernel(make_unary_kernel("sqrt", UnaryFnKind::Sqrt), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Abs => b.add_elementwise(
                    retype_kernel(make_unary_kernel("abs", UnaryFnKind::Abs), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Recip => b.add_elementwise(
                    retype_kernel(make_unary_kernel("recip", UnaryFnKind::Recip), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Sin => b.add_elementwise(
                    retype_kernel(make_unary_kernel("sin", UnaryFnKind::Sin), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Cos => b.add_elementwise(
                    retype_kernel(make_unary_kernel("cos", UnaryFnKind::Cos), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Log => b.add_elementwise(
                    retype_kernel(make_unary_kernel("log", UnaryFnKind::Log), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Floor => b.add_elementwise(
                    retype_kernel(make_unary_kernel("floor", UnaryFnKind::Floor), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Round => b.add_elementwise(
                    retype_kernel(make_unary_kernel("round", UnaryFnKind::Round), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Fract => b.add_elementwise(
                    retype_kernel(make_unary_kernel("fract", UnaryFnKind::Fract), stype),
                    &[input],
                    shape,
                ),
                UnaryOp::Neg => {
                    b.add_elementwise(retype_kernel(make_neg_kernel(), stype), &[input], shape)
                }
                UnaryOp::Sqr => {
                    b.add_elementwise(retype_kernel(make_sqr_kernel(), stype), &[input], shape)
                }
                UnaryOp::GeluErf => b.add_elementwise(
                    retype_kernel(make_gelu_erf_kernel(), stype),
                    &[input],
                    shape,
                ),
                other => {
                    return Err(TensorError::Unsupported(format!(
                        "gpu_unary builder: unsupported op {other:?}"
                    )))
                }
            };

            crate::build_kernel(b, out)
        })?;

        Self::dispatch_def(&def, &[("data", x_data.as_gpu_slice())], shape, x.dtype())
    }

    /// GPU-native Silu: decomposed as sigmoid(x) * x (no scalar Silu in IR).
    pub(super) fn gpu_silu(x: &DynTensor) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_silu")?;
        let shape = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;

        let def = crate::kernel_def_cache::get_or_build("silu", &[shape], &[], x.dtype(), || {
            let mut b = TensorBlockBuilder::new("dyn_silu");
            let input = b.add_input("data", shape);
            let sig = b.add_sigmoid(input, shape);
            let out = b.add_binary_mul(input, sig, shape);
            crate::build_kernel(b, out)
        })?;

        Self::dispatch_def(&def, &[("data", x_data.as_gpu_slice())], shape, x.dtype())
    }

    /// GPU-native clamp: `max(lo, min(hi, x))` in a single dispatch.
    ///
    /// Replaces the 8-encoding relu decomposition (clamp_min=3 + clamp_max=5)
    /// with a single kernel containing compare+select IR nodes. (#1815 D2a)
    pub(super) fn gpu_clamp(x: &DynTensor, min: f64, max: f64) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_clamp")?;
        let shape = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let stype = super::helpers::scalar_type_for_dtype(x.dtype());
        let tag = format!("clamp_{min}_{max}");

        let def = crate::kernel_def_cache::get_or_build(&tag, &[shape], &[], x.dtype(), || {
            let mut b = TensorBlockBuilder::new("dyn_clamp");
            let input = b.add_input("data", shape);
            let kernel = make_clamp_kernel("clamp", min, max, stype);
            let out = b.add_elementwise(kernel, &[input], shape);
            crate::build_kernel(b, out)
        })?;

        Self::dispatch_def(&def, &[("data", x_data.as_gpu_slice())], shape, x.dtype())
    }

    /// GPU-native clamp_min: `max(lo, x)` in a single dispatch.
    ///
    /// Replaces the 3-encoding relu decomposition (sub_scalar + relu + add_scalar).
    /// (#1815 D2a)
    pub(super) fn gpu_clamp_min(x: &DynTensor, min: f64) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_clamp_min")?;
        let shape = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let stype = super::helpers::scalar_type_for_dtype(x.dtype());
        let tag = format!("clamp_min_{min}");

        let def = crate::kernel_def_cache::get_or_build(&tag, &[shape], &[], x.dtype(), || {
            let mut b = TensorBlockBuilder::new("dyn_clamp_min");
            let input = b.add_input("data", shape);
            let kernel = make_clamp_min_kernel("clamp_min", min, stype);
            let out = b.add_elementwise(kernel, &[input], shape);
            crate::build_kernel(b, out)
        })?;

        Self::dispatch_def(&def, &[("data", x_data.as_gpu_slice())], shape, x.dtype())
    }

    /// GPU-native clamp_max: `min(hi, x)` in a single dispatch.
    ///
    /// Replaces the 5-encoding relu decomposition (neg + add_scalar + relu + neg + add_scalar).
    /// (#1815 D2a)
    pub(super) fn gpu_clamp_max(x: &DynTensor, max: f64) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_clamp_max")?;
        let shape = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let stype = super::helpers::scalar_type_for_dtype(x.dtype());
        let tag = format!("clamp_max_{max}");

        let def = crate::kernel_def_cache::get_or_build(&tag, &[shape], &[], x.dtype(), || {
            let mut b = TensorBlockBuilder::new("dyn_clamp_max");
            let input = b.add_input("data", shape);
            let kernel = make_clamp_max_kernel("clamp_max", max, stype);
            let out = b.add_elementwise(kernel, &[input], shape);
            crate::build_kernel(b, out)
        })?;

        Self::dispatch_def(&def, &[("data", x_data.as_gpu_slice())], shape, x.dtype())
    }

    /// GPU scalar binary op: `x OP scalar` in a single dispatch.
    ///
    /// Bakes the scalar value as an inline `Literal` constant in the MSL kernel,
    /// eliminating `scalar_like()` CPU alloc, GPU transfer, broadcast step, and
    /// intermediate buffer. Single kernel, single buffer read/write.
    /// Same pattern as `gpu_clamp_min`/`gpu_clamp_max`. Part of #3230 (Gap 2).
    pub(super) fn gpu_scalar_binary(op: BinaryOp, x: &DynTensor, scalar: f64) -> Result<DynTensor> {
        use nn_dsl::ir::BinOpKind;

        let ir_op = match op {
            BinaryOp::Add => BinOpKind::Add,
            BinaryOp::Sub => BinOpKind::Sub,
            BinaryOp::Mul => BinOpKind::Mul,
            BinaryOp::Div => BinOpKind::Div,
            other => {
                return Err(TensorError::Unsupported(format!(
                    "gpu_scalar_binary: unsupported op {other:?}"
                )))
            }
        };
        Self::validate_f32(x, "gpu_scalar_binary")?;
        let shape = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;
        let stype = super::helpers::scalar_type_for_dtype(x.dtype());
        let tag = format!("scalar_{ir_op:?}_{scalar}");

        let def = crate::kernel_def_cache::get_or_build(&tag, &[shape], &[], x.dtype(), || {
            let op_name = format!("scalar_{ir_op:?}").to_lowercase();
            let mut b = TensorBlockBuilder::new("dyn_scalar_binary");
            let input = b.add_input("data", shape);
            let kernel = super::kernels::make_scalar_binop_kernel(&op_name, ir_op, scalar, stype);
            let out = b.add_elementwise(kernel, &[input], shape);
            crate::build_kernel(b, out)
        })?;

        Self::dispatch_def(&def, &[("data", x_data.as_gpu_slice())], shape, x.dtype())
    }

    // gpu_reduce and gpu_reduce_via_transpose extracted to
    // dyn_tensor_metal_ops_reduce.rs (#1276)

    // gpu_matmul, gpu_compare, gpu_compare_tensor, gpu_where_cond extracted to
    // dyn_tensor_metal_ops_compare.rs (#1544 D8)
}
