// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Unary activation dispatch step builder.
//!
//! Extracted from `codegen_msl_tensor_dispatch.rs` (#2575) to keep
//! the dispatch planner under 400 lines.

use crate::ir::ScalarType;
use crate::tensor_ir::{TensorKernelDef, TensorNode, TensorNodeId};

use super::{shape_total, DispatchStep, TensorMSLCodegenError};

/// Build a unary activation dispatch step (Sigmoid, Gelu, GeluErf, Relu, Tanh).
pub(super) fn build_unary_activation(
    effective: &TensorKernelDef,
    node: &TensorNode,
    input: TensorNodeId,
    name: &str,
    dtype: ScalarType,
) -> Result<Option<DispatchStep>, TensorMSLCodegenError> {
    let total_elements = shape_total(&node.shape)?;
    let step = match name {
        "sigmoid" => DispatchStep::Sigmoid {
            kernel_name: format!("{}_sigmoid_n{}", effective.name, node.id.index()),
            dtype,
            input,
            output: node.id,
            total_elements,
        },
        "gelu" => DispatchStep::Gelu {
            kernel_name: format!("{}_gelu_n{}", effective.name, node.id.index()),
            dtype,
            input,
            output: node.id,
            total_elements,
        },
        "gelu_erf" => DispatchStep::GeluErf {
            kernel_name: format!("{}_gelu_erf_n{}", effective.name, node.id.index()),
            dtype,
            input,
            output: node.id,
            total_elements,
        },
        "relu" => DispatchStep::Relu {
            kernel_name: format!("{}_relu_n{}", effective.name, node.id.index()),
            dtype,
            input,
            output: node.id,
            total_elements,
        },
        "tanh" => DispatchStep::Tanh {
            kernel_name: format!("{}_tanh_n{}", effective.name, node.id.index()),
            dtype,
            input,
            output: node.id,
            total_elements,
        },
        "exp" => DispatchStep::Exp {
            kernel_name: format!("{}_exp_n{}", effective.name, node.id.index()),
            dtype,
            input,
            output: node.id,
            total_elements,
        },
        "softplus" => DispatchStep::Softplus {
            kernel_name: format!("{}_softplus_n{}", effective.name, node.id.index()),
            dtype,
            input,
            output: node.id,
            total_elements,
        },
        _ => {
            return Err(TensorMSLCodegenError::InvalidParameter(format!(
                "unknown unary activation: {name}"
            )))
        }
    };
    Ok(Some(step))
}
