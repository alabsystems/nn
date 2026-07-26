// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pool2d (AvgPool2d / MaxPool2d) tensor IR validators.
//!
//! Extracted alongside `tensor_ir_validate_layers_conv.rs` to keep
//! per-file line counts under the 500-line limit.

use super::super::{TensorIRConvError, TensorIRError, TensorKernelDef, TensorNodeId};

impl TensorKernelDef {
    /// Validate a Pool2d op (AvgPool2d or MaxPool2d).
    ///
    /// Checks: input rank >= 3, stride >= 1, kernel >= 1, kernel fits padded input.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_pool2d(
        &self,
        current: TensorNodeId,
        input: TensorNodeId,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Result<(), TensorIRError> {
        self.check_ref(current, input)?;

        let input_shape = &self.nodes[input.index()].shape;
        if input_shape.len() < 3 {
            return Err(TensorIRConvError::Pool2dInputRankTooLow {
                rank: input_shape.len(),
            }
            .into());
        }

        if stride_h == 0 || stride_w == 0 {
            return Err(TensorIRConvError::Pool2dZeroStride { stride_h, stride_w }.into());
        }

        if kernel_h == 0 || kernel_w == 0 {
            return Err(TensorIRConvError::Pool2dZeroKernelSize { kernel_h, kernel_w }.into());
        }

        let in_h = input_shape[input_shape.len() - 2];
        let in_w = input_shape[input_shape.len() - 1];
        let padded_h = in_h + 2 * padding_h;
        let padded_w = in_w + 2 * padding_w;
        if padded_h < kernel_h || padded_w < kernel_w {
            return Err(TensorIRConvError::Pool2dKernelTooLarge {
                kernel_h,
                kernel_w,
                padded_h,
                padded_w,
            }
            .into());
        }

        Ok(())
    }
}
