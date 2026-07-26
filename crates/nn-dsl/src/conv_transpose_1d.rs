// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ConvTranspose1d tensor kernel builder.
//!
//! Builds a `TensorKernelDef` for transposed 1D convolution (upsampling) that
//! maps to NY's `Layer::ConvTranspose1d(ConvTranspose1dLayer)`. Weight
//! and bias inputs are treated as fixed parameters (not verified as variables).
//!
//! # Key differences from Conv1d
//!
//! - Kernel layout: `[in_channels, out_channels/groups, kernel_size]` (in/out swapped).
//! - Output formula: `(L-1)*stride - 2*pad + dilation*(K-1) + 1`.

use crate::tensor_ir::{
    TensorIRConvError, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build a ConvTranspose1d tensor kernel definition.
///
/// Inputs (in order): `[data, weight]` or `[data, weight, bias]`.
/// Weight and bias are treated as constant parameters during verification.
///
/// # Arguments
///
/// * `name` — Kernel name for diagnostics and node naming.
/// * `in_channels` — Number of input channels.
/// * `out_channels` — Number of output channels (= out_ch_per_group * groups).
/// * `kernel_size` — Spatial extent of the transposed convolution kernel.
/// * `in_length` — Spatial length of the input tensor.
/// * `stride` — Convolution stride (must be >= 1).
/// * `padding` — Zero-padding applied to both sides.
/// * `dilation` — Dilation (spacing between kernel elements, must be >= 1).
/// * `groups` — Number of channel groups (must be >= 1).
/// * `has_bias` — Whether to include a bias input node.
/// * `output_padding` — Extra size added to one side of the output (must be < stride).
///
/// # Errors
///
/// Returns `TensorIRError` if stride/dilation/groups is zero, if output_padding >= stride,
/// if arithmetic overflows, or if the output length is non-positive.
#[allow(clippy::too_many_arguments)]
pub fn build_conv_transpose_1d(
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    has_bias: bool,
    output_padding: usize,
) -> Result<TensorKernelDef, TensorIRError> {
    if stride == 0 {
        return Err(TensorIRConvError::ConvTranspose1dZeroStride.into());
    }
    if dilation == 0 {
        return Err(TensorIRConvError::ConvTranspose1dZeroDilation.into());
    }
    if groups == 0 {
        return Err(TensorIRConvError::ConvTranspose1dZeroGroups.into());
    }
    // PyTorch constraint: output_padding must be < stride.
    if stride > 0 && output_padding >= stride {
        return Err(TensorIRConvError::ConvTranspose1dArithmeticOverflow {
            context: format!("output_padding={output_padding} must be < stride={stride}"),
        }
        .into());
    }

    let out_ch_per_group = out_channels / groups;

    // out_length = (in_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    let expanded = in_length
        .checked_sub(1)
        .and_then(|v| v.checked_mul(stride))
        .and_then(|base| {
            dilation
                .checked_mul(kernel_size.checked_sub(1)?)
                .and_then(|dk| base.checked_add(dk))
        })
        .and_then(|v| v.checked_add(output_padding))
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::ConvTranspose1dArithmeticOverflow {
                context: format!(
                    "(in_length={in_length} - 1) * stride={stride} + dilation={dilation} \
                     * (kernel_size={kernel_size} - 1) + output_padding={output_padding} + 1"
                ),
            })
        })?;
    let double_pad = padding.checked_mul(2).ok_or_else(|| {
        TensorIRError::from(TensorIRConvError::ConvTranspose1dArithmeticOverflow {
            context: format!("2 * padding={padding}"),
        })
    })?;
    if expanded < double_pad || expanded - double_pad == 0 {
        return Err(TensorIRConvError::ConvTranspose1dOutputNonPositive {
            out_length: if expanded >= double_pad {
                (expanded - double_pad) as isize
            } else {
                expanded as isize - double_pad as isize
            },
            in_length,
            stride,
            kernel_size,
            padding,
        }
        .into());
    }
    let out_length = expanded - double_pad;

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
        // %1 = input weight: [in_channels, out_ch_per_group, kernel_size]
        TensorNode::new(
            TensorNodeId::new(1),
            TensorOpKind::Input {
                name: "weight".into(),
                shape: vec![in_channels, out_ch_per_group, kernel_size],
            },
            vec![in_channels, out_ch_per_group, kernel_size],
        ),
    ];

    let bias_node = if has_bias {
        // %2 = input bias: [out_channels]
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

    let conv_id = TensorNodeId::new(nodes.len());
    nodes.push(TensorNode::new(
        conv_id,
        TensorOpKind::ConvTranspose1d {
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: bias_node,
            stride,
            padding,
            dilation,
            groups,
            output_padding,
        },
        vec![out_channels, out_length],
    ));

    Ok(TensorKernelDef::new(name, nodes, conv_id))
}

#[cfg(test)]
#[path = "conv_transpose_1d_tests.rs"]
mod tests;

#[cfg(kani)]
mod kani_proofs {
    use super::build_conv_transpose_1d;

    /// Proves `build_conv_transpose_1d` never panics for any bounded parameter inputs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn conv_transpose_1d_build_no_panic() {
        let in_len: usize = kani::any();
        let kernel_size: usize = kani::any();
        let stride: usize = kani::any();
        let padding: usize = kani::any();
        let dilation: usize = kani::any();
        let groups: usize = kani::any();
        let in_channels: usize = kani::any();
        let out_channels: usize = kani::any();
        let output_padding: usize = kani::any();

        kani::assume(in_len >= 1 && in_len <= 8);
        kani::assume(kernel_size >= 1 && kernel_size <= 8);
        kani::assume(stride >= 1 && stride <= 8);
        kani::assume(padding <= 8);
        kani::assume(dilation >= 1 && dilation <= 4);
        kani::assume(groups >= 1 && groups <= 4);
        kani::assume(in_channels >= 1 && in_channels <= 8);
        kani::assume(out_channels >= 1 && out_channels <= 8);
        kani::assume(output_padding < stride);

        let _ = build_conv_transpose_1d(
            "kani_test",
            in_channels,
            out_channels,
            kernel_size,
            in_len,
            stride,
            padding,
            dilation,
            groups,
            false,
            output_padding,
        );
    }

    /// Proves that when `build_conv_transpose_1d` succeeds, output spatial dim >= 1.
    #[kani::unwind(1)]
    #[kani::proof]
    fn conv_transpose_1d_output_shape_positive() {
        let in_len: usize = kani::any();
        let kernel_size: usize = kani::any();
        let stride: usize = kani::any();
        let padding: usize = kani::any();

        kani::assume(in_len >= 1 && in_len <= 8);
        kani::assume(kernel_size >= 1 && kernel_size <= 8);
        kani::assume(stride >= 1 && stride <= 8);
        kani::assume(padding <= 8);

        if let Ok(def) = build_conv_transpose_1d(
            "kani_test",
            4,
            2,
            kernel_size,
            in_len,
            stride,
            padding,
            1,
            1,
            false,
            0,
        ) {
            let output_node = &def.nodes[def.nodes.len() - 1];
            let out_len = output_node.shape[1];
            assert!(out_len >= 1, "ConvTranspose1d output length must be >= 1");
        }
    }

    /// Proves the output length formula with dilation=1, groups=1, output_padding=0.
    #[kani::unwind(1)]
    #[kani::proof]
    fn conv_transpose_1d_output_length_formula() {
        let in_len: usize = kani::any();
        let kernel_size: usize = kani::any();
        let stride: usize = kani::any();
        let padding: usize = kani::any();

        kani::assume(in_len >= 1 && in_len <= 8);
        kani::assume(kernel_size >= 1 && kernel_size <= 8);
        kani::assume(stride >= 1 && stride <= 8);
        kani::assume(padding <= 8);

        // Independent formula: (in-1)*stride - 2*pad + 1*(K-1) + 0 + 1
        let base = (in_len - 1) * stride;
        let effective = base + (kernel_size - 1) + 1;
        let double_pad = 2 * padding;
        if effective <= double_pad {
            return; // invalid params — builder should return Err
        }
        let expected_out_len = effective - double_pad;

        if let Ok(def) = build_conv_transpose_1d(
            "kani_test",
            4,
            2,
            kernel_size,
            in_len,
            stride,
            padding,
            1,
            1,
            false,
            0,
        ) {
            let output_node = &def.nodes[def.nodes.len() - 1];
            let actual_out_len = output_node.shape[1];
            assert_eq!(
                actual_out_len, expected_out_len,
                "output length must match ConvTranspose1d formula"
            );
        }
    }
}
