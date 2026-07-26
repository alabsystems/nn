// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU-native shape op implementations for [`MetalDynBackend`].
//!
//! These methods implement the optional `GpuBackend` trait methods (narrow,
//! transpose, permute, softmax, log_softmax) so `DynTensor` shape
//! operations dispatch directly on Metal GPU buffers without CPU round-trips.
//!
//! Concatenation (`gpu_cat`) lives in `dyn_tensor_metal_cat.rs`.
//! Convolution ops (conv1d, conv_transpose1d) and index_select are in
//! `dyn_tensor_metal_conv_ops.rs`.
//!
//! Part of #1084 (GPU shape ops).

use nn_core::dyn_tensor::DynTensor;
use nn_core::{check_dim, Result, TensorError};

use nn_dsl::ir::ScalarType;
use nn_dsl::{PrecisionContract, PrecisionTier, TensorBlockBuilder};

use super::MetalTensorData;

#[path = "dyn_tensor_metal_shape_ops_narrow.rs"]
mod narrow;

#[path = "dyn_tensor_metal_shape_ops_slice_set.rs"]
mod slice_set;

#[path = "dyn_tensor_metal_shape_ops_unfold.rs"]
mod unfold;

impl super::MetalDynBackend {
    /// GPU-native transpose (swap two dimensions).
    ///
    /// Builds a full permutation array from the two swapped dims and delegates
    /// to `add_transpose` which takes an axes permutation.
    pub(super) fn gpu_transpose(x: &DynTensor, d1: usize, d2: usize) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_transpose")?;
        let shape = x.dims();
        let rank = shape.len();

        // Defense-in-depth: validate dimension bounds before indexing.
        check_dim(d1, rank)?;
        check_dim(d2, rank)?;

        let x_data = x.gpu_data::<MetalTensorData>()?;

        let mut axes: Vec<usize> = (0..rank).collect();
        axes.swap(d1, d2);

        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape.swap(d1, d2);

        let def = crate::kernel_def_cache::get_or_build(
            "transpose",
            &[shape],
            &[d1 as u64, d2 as u64],
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_transpose");
                let input = b.add_input("data", shape);
                let out = b.add_transpose(input, &axes, &out_shape);
                crate::build_kernel(b, out)
            },
        )?;

        Self::dispatch_def(
            &def,
            &[("data", x_data.as_gpu_slice())],
            &out_shape,
            x.dtype(),
        )
    }

    /// GPU-native permute (arbitrary dimension reordering).
    ///
    /// Uses the same `add_transpose` builder method since it accepts a full
    /// axes permutation (not just a pair swap).
    pub(super) fn gpu_permute(x: &DynTensor, dims: &[usize]) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_permute")?;
        let shape = x.dims();
        let rank = shape.len();

        // Defense-in-depth: validate permutation before indexing.
        if dims.len() != rank {
            return Err(TensorError::RankMismatch {
                expected: rank,
                actual: dims.len(),
            });
        }
        let mut seen = vec![false; rank];
        for &d in dims {
            check_dim(d, rank)?;
            if seen[d] {
                return Err(TensorError::ValueOutOfRange {
                    description: "gpu_permute: duplicate axis in permutation",
                });
            }
            seen[d] = true;
        }

        let x_data = x.gpu_data::<MetalTensorData>()?;

        let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();

        let mut perm_params: Vec<u64> = dims.iter().map(|&d| d as u64).collect();
        perm_params.push(rank as u64); // sentinel to differentiate same-prefix permutations
        let def = crate::kernel_def_cache::get_or_build(
            "permute",
            &[shape],
            &perm_params,
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_permute");
                let input = b.add_input("data", shape);
                let out = b.add_transpose(input, dims, &out_shape);
                crate::build_kernel(b, out)
            },
        )?;

        Self::dispatch_def(
            &def,
            &[("data", x_data.as_gpu_slice())],
            &out_shape,
            x.dtype(),
        )
    }

    // gpu_cat lives in dyn_tensor_metal_cat.rs (extracted for file size).

    /// GPU-native softmax along a dimension.
    pub(super) fn gpu_softmax(x: &DynTensor, dim: usize) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_softmax")?;
        let shape = x.dims();
        check_dim(dim, shape.len())?;
        let x_data = x.gpu_data::<MetalTensorData>()?;

        let axis = dim as i32;

        let def = crate::kernel_def_cache::get_or_build(
            "softmax",
            &[shape],
            &[dim as u64],
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_softmax");
                let input = b.add_input("data", shape);
                let out = b.add_softmax(input, axis, shape);
                crate::build_kernel(b, out)
            },
        )?;

        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        Self::dispatch_def_with_contract(
            &def,
            &[("data", x_data.as_gpu_slice())],
            shape,
            x.dtype(),
            contract,
        )
    }

    /// GPU-native log-softmax along a dimension.
    ///
    /// Composes softmax + elementwise log as two dispatch steps within a single
    /// kernel graph, reducing the 6-step decomposed path (max→sub→exp→sum→log→sub)
    /// to 2 steps (softmax→log).
    pub(super) fn gpu_log_softmax(x: &DynTensor, dim: usize) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_log_softmax")?;
        let shape = x.dims();
        check_dim(dim, shape.len())?;
        let x_data = x.gpu_data::<MetalTensorData>()?;

        let axis = dim as i32;

        let def = crate::kernel_def_cache::get_or_build(
            "log_softmax",
            &[shape],
            &[dim as u64],
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_log_softmax");
                let input = b.add_input("data", shape);
                let softmax_out = b.add_softmax(input, axis, shape);
                let log_kernel = super::kernels::build_log_kernel();
                let out = b.add_elementwise(log_kernel, &[softmax_out], shape);
                crate::build_kernel(b, out)
            },
        )?;

        let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
        Self::dispatch_def_with_contract(
            &def,
            &[("data", x_data.as_gpu_slice())],
            shape,
            x.dtype(),
            contract,
        )
    }

    /// GPU-native expand (broadcast to larger shape).
    ///
    /// Dimensions of size 1 in the input can be expanded to any target size.
    /// Uses `add_broadcast` which generates an MSL kernel that maps each output
    /// element to the corresponding input element via stride calculations,
    /// avoiding the zero-tensor + add workaround.
    pub(super) fn gpu_expand(x: &DynTensor, new_dims: &[usize]) -> Result<DynTensor> {
        Self::validate_f32(x, "gpu_expand")?;
        let shape = x.dims();
        let x_data = x.gpu_data::<MetalTensorData>()?;

        let def = crate::kernel_def_cache::get_or_build(
            "expand",
            &[shape, new_dims],
            &[],
            x.dtype(),
            || {
                let mut b = TensorBlockBuilder::new("dyn_expand");
                let input = b.add_input("data", shape);
                let out = b.add_broadcast(input, new_dims);
                crate::build_kernel(b, out)
            },
        )?;

        Self::dispatch_def(
            &def,
            &[("data", x_data.as_gpu_slice())],
            new_dims,
            x.dtype(),
        )
    }
}
