// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Causal Conv1d tensor kernel builder (pad-then-conv decomposition).
//!
//! Builds a `TensorKernelDef` for causal (left-pad-only) 1D convolution by
//! composing `ZeroPad1d` + `Conv1d(padding=0)`. This matches the dvoice
//! pattern used in CosyVoice3 vocoder ResBlocks:
//!
//! ```python
//! pad_left = (kernel_size - 1) * dilation
//! x = F.pad(x, (pad_left, 0))       # causal: all padding on left
//! x = conv1d(x, weight, stride=1, padding=0)
//! ```

use crate::tensor_ir::{
    TensorIRConvError, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build a causal Conv1d using pad-then-conv decomposition.
///
/// Creates a two-node graph: `ZeroPad1d(pad_left=(K-1)*D, pad_right=0)` followed
/// by `Conv1d(padding=0)`. The Conv1d sees a pre-padded input, so no symmetric
/// padding is applied by the convolution itself.
///
/// This decomposition preserves Conv1d's existing NY verification path
/// and adds only a trivial-bounds zero-padding node.
///
/// # Arguments
///
/// Same as [`super::build_conv1d_full`] except `padding` is replaced by causal
/// padding computed automatically from `kernel_size` and `dilation`.
///
/// # Errors
///
/// Returns `TensorIRError` if any parameter is zero when it must be >= 1,
/// if arithmetic overflows, or if `in_channels` is not divisible by `groups`.
pub fn build_causal_conv1d(
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    has_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    if stride == 0 {
        return Err(TensorIRConvError::Conv1dZeroStride.into());
    }
    if kernel_size == 0 {
        return Err(TensorIRConvError::Conv1dZeroKernelSize.into());
    }
    if dilation == 0 {
        return Err(TensorIRConvError::Conv1dZeroDilation.into());
    }
    if groups == 0 {
        return Err(TensorIRConvError::Conv1dZeroGroups.into());
    }
    if !in_channels.is_multiple_of(groups) {
        return Err(TensorIRConvError::Conv1dGroupsChannelMismatch {
            in_channels,
            groups,
        }
        .into());
    }

    // Causal padding: pad_left = (kernel_size - 1) * dilation
    // kernel_size >= 1 guaranteed above, so kernel_size - 1 is safe.
    let pad_left = dilation.checked_mul(kernel_size - 1).ok_or_else(|| {
        TensorIRConvError::Conv1dArithmeticOverflow {
            context: format!(
                "causal pad_left: dilation={dilation} * (kernel_size={kernel_size} - 1)"
            ),
        }
    })?;

    // Padded input length: in_length + pad_left (pad_right = 0 for causal)
    let padded_length = in_length.checked_add(pad_left).ok_or_else(|| {
        TensorIRConvError::Conv1dArithmeticOverflow {
            context: format!("padded_length: in_length={in_length} + pad_left={pad_left}"),
        }
    })?;

    // effective_kernel = dilation * (kernel_size - 1) + 1 = pad_left + 1
    let effective_kernel = pad_left + 1;

    if padded_length < effective_kernel {
        return Err(TensorIRConvError::Conv1dKernelTooLarge {
            kernel_size: effective_kernel,
            padded_len: padded_length,
            in_len: in_length,
            padding: 0,
        }
        .into());
    }

    // out_length = (padded_length - effective_kernel) / stride + 1
    let out_length = (padded_length - effective_kernel) / stride + 1;
    let weight_in_channels = in_channels / groups;

    let mut nodes = vec![
        // %0 = input data: [in_channels, in_length]
        TensorNode::new(
            TensorNodeId::new(0),
            TensorOpKind::Input {
                name: crate::input_names::DATA.into(),
                shape: vec![in_channels, in_length],
            },
            vec![in_channels, in_length],
        ),
        // %1 = weight: [out_channels, in_channels/groups, kernel_size]
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![out_channels, weight_in_channels, kernel_size],
            },
            vec![out_channels, weight_in_channels, kernel_size],
        ),
    ];

    let bias_node = if has_bias {
        // %2 = bias: [out_channels]
        nodes.push(TensorNode::new(
            TensorNodeId::new(2),
            TensorOpKind::Input {
                name: "bias".into(),
                shape: vec![out_channels],
            },
            vec![out_channels],
        ));
        Some(TensorNodeId::new(2))
    } else {
        None
    };

    // %N = ZeroPad1d: [in_channels, padded_length]
    let pad_id = TensorNodeId::new(nodes.len());
    nodes.push(TensorNode::new(
        pad_id,
        TensorOpKind::ZeroPad1d {
            input: TensorNodeId::new(0),
            pad_left,
            pad_right: 0,
        },
        vec![in_channels, padded_length],
    ));

    // %N+1 = Conv1d(padding=0): [out_channels, out_length]
    let conv_id = TensorNodeId::new(nodes.len());
    nodes.push(TensorNode::new(
        conv_id,
        TensorOpKind::Conv1d {
            input: pad_id,
            weight: TensorNodeId::new(1),
            bias: bias_node,
            stride,
            padding: 0,
            dilation,
            groups,
        },
        vec![out_channels, out_length],
    ));

    Ok(TensorKernelDef::new(name, nodes, conv_id))
}

#[cfg(test)]
#[path = "causal_conv1d_tests.rs"]
mod tests;
