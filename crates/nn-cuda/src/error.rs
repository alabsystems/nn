// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Error types for HIP code generation from TensorOpKind IR.

use nn_dsl::TensorNodeId;
use thiserror::Error;

/// Errors from HIP C++ code generation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum HipCodegenError {
    #[error("shape product overflow: {shape:?}")]
    ShapeProductOverflow { shape: Vec<usize> },

    #[error(
        "reduce node {node_id:?} uses axis {axis} for shape {shape:?}, \
         but HIP codegen currently supports only last-axis reductions"
    )]
    NonLastAxisReduce {
        node_id: TensorNodeId,
        axis: usize,
        shape: Vec<usize>,
    },

    #[error(
        "softmax node {node_id:?} uses axis {axis} for shape {shape:?}, \
         but HIP codegen currently supports only last-axis softmax"
    )]
    NonLastAxisSoftmax {
        node_id: TensorNodeId,
        axis: usize,
        shape: Vec<usize>,
    },

    #[error("unsupported dispatch step for HIP codegen: {step_name}")]
    UnsupportedStep { step_name: &'static str },

    #[error("stride value {value} exceeds u32::MAX ({max}) — HIP uint is 32-bit")]
    StrideExceedsU32 { value: usize, max: u32 },

    #[error("invalid convolution parameter: {0}")]
    InvalidParameter(String),

    #[error("axis {axis} out of bounds for shape with rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },

    #[error("emit_stack_kernel called with n_inputs=0")]
    EmptyStack,

    #[error("unsupported IR variant for HIP codegen: {variant_desc}")]
    UnsupportedIRVariant { variant_desc: &'static str },
}
