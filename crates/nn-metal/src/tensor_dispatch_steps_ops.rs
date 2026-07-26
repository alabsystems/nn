// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Binary, unary, padding, and softmax dispatch steps for tensor Metal pipeline.
//!
//! Extracted from `tensor_dispatch_steps.rs` (#1665) to keep files under 400 lines.
//! Contains dispatch arms for BinaryAdd, BinaryMul, MatMul, Sigmoid, Gelu, Relu,
//! Tanh, ZeroPad1d, and Softmax.

use nn_dsl::{DispatchStep, TensorKernelDef};

use super::super::helpers::{encode_elementwise_step, encode_softmax, EncodeContext};
use super::super::TensorDispatchError;

/// Dispatch a binary, unary, padding, or softmax step.
///
/// Returns `Some(Ok(()))` if the step was handled, `Some(Err(...))` on dispatch
/// failure, or `None` if the step is not one of the handled variants (caller
/// should fall through to the catch-all).
pub(super) fn dispatch_binary_unary_or_misc(
    step: &DispatchStep,
    enc: &mut EncodeContext<'_>,
    combined_msl: &str,
    kernel: &TensorKernelDef,
) -> Option<Result<(), TensorDispatchError>> {
    let _ = kernel; // used by callers for output_elems; not needed here
    match step {
        // Binary ops: left + right.
        DispatchStep::BinaryAdd {
            kernel_name,
            left,
            right,
            output,
            total_elements,
            ..
        }
        | DispatchStep::BinaryMul {
            kernel_name,
            left,
            right,
            output,
            total_elements,
            ..
        }
        | DispatchStep::MatMul {
            kernel_name,
            left,
            right,
            output,
            total_elements,
            ..
        } => {
            let r = encode_elementwise_step(
                enc,
                combined_msl,
                kernel_name,
                &[*left, *right],
                *output,
                *total_elements,
            );
            Some(r)
        }

        // Unary activations.
        DispatchStep::Sigmoid {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        }
        | DispatchStep::Gelu {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        }
        | DispatchStep::GeluErf {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        }
        | DispatchStep::Relu {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        }
        | DispatchStep::Tanh {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        }
        | DispatchStep::LeakyRelu {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        }
        | DispatchStep::Elu {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        }
        | DispatchStep::Exp {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        }
        | DispatchStep::Softplus {
            kernel_name,
            input,
            output,
            total_elements,
            ..
        } => {
            let r = encode_elementwise_step(
                enc,
                combined_msl,
                kernel_name,
                &[*input],
                *output,
                *total_elements,
            );
            Some(r)
        }

        DispatchStep::ZeroPad1d {
            kernel_name,
            input,
            output,
            channels,
            out_length,
            ..
        } => {
            let elems = match channels.checked_mul(*out_length) {
                Some(e) => e,
                None => {
                    return Some(Err(TensorDispatchError::ShapeOverflow {
                        shape: vec![*channels, *out_length],
                    }))
                }
            };
            let r =
                encode_elementwise_step(enc, combined_msl, kernel_name, &[*input], *output, elems);
            Some(r)
        }

        DispatchStep::Softmax {
            kernel_name,
            input,
            output,
            axis_size,
            outer_size,
            ..
        } => {
            let r = encode_softmax(
                enc,
                combined_msl,
                kernel_name,
                *input,
                *output,
                *outer_size,
                *axis_size,
            );
            Some(r)
        }

        _ => None,
    }
}
