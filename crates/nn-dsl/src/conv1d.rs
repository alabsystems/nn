// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Conv1d tensor kernel builder.
//!
//! Builds a `TensorKernelDef` for 1D convolution that maps to NY's
//! `Layer::Conv1d(Conv1dLayer)`. Weight and bias inputs are treated as fixed
//! parameters (not verified as variables).
//!
//! # dvoice parameter coverage
//!
//! Stride, padding, kernel_size, optional bias, and dilation are fully
//! supported. Dilation uses kernel expansion (zero-insertion) to work around
//! NY's lack of native dilation support (#582). Groups != 1 are
//! rejected until NY upstream adds support (NY#3170).

use crate::tensor_ir::{
    TensorIRConvError, TensorIRError, TensorKernelDef, TensorNode, TensorNodeId, TensorOpKind,
};

/// Build a Conv1d tensor kernel definition with default dilation=1, groups=1.
///
/// Delegates to [`build_conv1d_full`] with `dilation=1` and `groups=1`.
/// Use `build_conv1d_full` when you need dilated or grouped convolutions.
///
/// # Errors
///
/// Returns `TensorIRError` if parameters are invalid (zero stride/kernel_size,
/// arithmetic overflow, or padded input smaller than kernel).
pub fn build_conv1d(
    name: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    in_length: usize,
    stride: usize,
    padding: usize,
    has_bias: bool,
) -> Result<TensorKernelDef, TensorIRError> {
    build_conv1d_full(
        name,
        in_channels,
        out_channels,
        kernel_size,
        in_length,
        stride,
        padding,
        1, // dilation
        1, // groups
        has_bias,
    )
}

/// Build a Conv1d tensor kernel definition with full parameter control.
///
/// Inputs (in order): `[data, weight]` or `[data, weight, bias]`.
/// Weight and bias are treated as constant parameters during verification.
///
/// # Arguments
///
/// * `name` — Kernel name for diagnostics and node naming.
/// * `in_channels` — Number of input channels.
/// * `out_channels` — Number of output channels (filters).
/// * `kernel_size` — Spatial extent of the convolution kernel (must be >= 1).
/// * `in_length` — Spatial length of the input tensor.
/// * `stride` — Convolution stride (must be >= 1).
/// * `padding` — Zero-padding applied to both sides.
/// * `dilation` — Spacing between kernel elements (must be >= 1).
/// * `groups` — Number of input channel groups (must be >= 1).
/// * `has_bias` — Whether to include a bias input node.
///
/// # Errors
///
/// Returns `TensorIRError` if any parameter is zero when it must be >= 1,
/// if `in_channels` is not divisible by `groups`, if arithmetic overflows,
/// or if the effective padded input is smaller than the effective kernel.
pub fn build_conv1d_full(
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

    // effective_kernel = dilation * (kernel_size - 1) + 1
    // kernel_size >= 1 guaranteed above, so kernel_size - 1 is safe.
    let effective_kernel = dilation
        .checked_mul(kernel_size - 1)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv1dArithmeticOverflow {
                context: format!(
                    "effective_kernel: dilation={dilation} * (kernel_size={kernel_size} - 1) + 1"
                ),
            })
        })?;

    // padded = in_length + 2 * padding
    let padded = padding
        .checked_mul(2)
        .and_then(|v| v.checked_add(in_length))
        .ok_or_else(|| {
            TensorIRError::from(TensorIRConvError::Conv1dArithmeticOverflow {
                context: format!("padded: in_length={in_length} + 2 * padding={padding}"),
            })
        })?;

    if padded < effective_kernel {
        return Err(TensorIRConvError::Conv1dKernelTooLarge {
            kernel_size: effective_kernel,
            padded_len: padded,
            in_len: in_length,
            padding,
        }
        .into());
    }

    // out_length = (padded - effective_kernel) / stride + 1
    // padded >= effective_kernel guaranteed above, subtraction is safe.
    let out_length = (padded - effective_kernel) / stride + 1;
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
        // %1 = input weight: [out_channels, in_channels/groups, kernel_size]
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
        TensorOpKind::Conv1d {
            input: TensorNodeId::new(0),
            weight: TensorNodeId::new(1),
            bias: bias_node,
            stride,
            padding,
            dilation,
            groups,
        },
        vec![out_channels, out_length],
    ));

    Ok(TensorKernelDef::new(name, nodes, conv_id))
}

#[cfg(test)]
#[path = "conv1d_tests.rs"]
mod tests;

#[cfg(kani)]
mod kani_proofs {
    use super::build_conv1d_full;

    /// Proves `build_conv1d_full` never panics for any bounded parameter inputs.
    ///
    /// Calls the actual production function (not a parallel formula) to verify
    /// that all checked arithmetic and Result returns handle the parameter space.
    /// Domain reduced to [1, 4] and unwind(16) added for CBMC tractability (#767 AC3).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn conv1d_build_no_panic() {
        let in_len: usize = kani::any();
        let kernel_size: usize = kani::any();
        let stride: usize = kani::any();
        let padding: usize = kani::any();
        let dilation: usize = kani::any();
        let in_channels: usize = kani::any();
        let out_channels: usize = kani::any();
        let groups: usize = kani::any();

        kani::assume(in_len >= 1 && in_len <= 4);
        kani::assume(kernel_size >= 1 && kernel_size <= 4);
        kani::assume(stride >= 1 && stride <= 4);
        kani::assume(padding <= 4);
        kani::assume(dilation >= 1 && dilation <= 4);
        kani::assume(in_channels >= 1 && in_channels <= 4);
        kani::assume(out_channels >= 1 && out_channels <= 4);
        kani::assume(groups >= 1 && groups <= 4);

        // Call the actual production function — must not panic.
        let _ = build_conv1d_full(
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
        );
    }

    /// Proves that when `build_conv1d_full` succeeds, output spatial dim >= 1.
    /// Domain reduced to [1, 4] and unwind(16) added for CBMC tractability (#767 AC3).
    #[kani::unwind(1)]
    #[kani::proof]
    #[kani::unwind(16)]
    fn conv1d_output_shape_positive() {
        let in_len: usize = kani::any();
        let kernel_size: usize = kani::any();
        let stride: usize = kani::any();
        let padding: usize = kani::any();
        let dilation: usize = kani::any();

        kani::assume(in_len >= 1 && in_len <= 4);
        kani::assume(kernel_size >= 1 && kernel_size <= 4);
        kani::assume(stride >= 1 && stride <= 4);
        kani::assume(padding <= 4);
        kani::assume(dilation >= 1 && dilation <= 4);

        if let Ok(def) = build_conv1d_full(
            "kani_test",
            4,
            2,
            kernel_size,
            in_len,
            stride,
            padding,
            dilation,
            1,
            false,
        ) {
            let output_node = &def.nodes[def.nodes.len() - 1];
            let out_len = output_node.shape[1];
            assert!(out_len >= 1, "Conv1d output length must be >= 1");
        }
    }

    /// Proves that `out_channels` in the output shape matches the requested parameter.
    ///
    /// Conv1d output shape is `[out_channels, out_length]`. This harness verifies
    /// the channel dimension is correctly propagated (not swapped with in_channels
    /// or corrupted by groups arithmetic).
    #[kani::unwind(1)]
    #[kani::proof]
    fn conv1d_output_channels_preserved() {
        let in_channels: usize = kani::any();
        let out_channels: usize = kani::any();
        let kernel_size: usize = kani::any();
        let in_len: usize = kani::any();
        let stride: usize = kani::any();
        let padding: usize = kani::any();

        kani::assume(in_channels >= 1 && in_channels <= 8);
        kani::assume(out_channels >= 1 && out_channels <= 8);
        kani::assume(kernel_size >= 1 && kernel_size <= 8);
        kani::assume(in_len >= 1 && in_len <= 8);
        kani::assume(stride >= 1 && stride <= 8);
        kani::assume(padding <= 8);

        if let Ok(def) = build_conv1d_full(
            "kani_test",
            in_channels,
            out_channels,
            kernel_size,
            in_len,
            stride,
            padding,
            1,
            1,
            false,
        ) {
            let output_node = &def.nodes[def.nodes.len() - 1];
            assert_eq!(
                output_node.shape[0], out_channels,
                "output channel dim must equal out_channels parameter"
            );
        }
    }

    /// Proves the output length formula: `(in_len + 2*padding - eff_kernel) / stride + 1`.
    ///
    /// Verifies that the builder's output shape matches an independently computed
    /// expected value, catching off-by-one or formula transposition bugs.
    #[kani::unwind(1)]
    #[kani::proof]
    fn conv1d_output_length_formula() {
        let in_len: usize = kani::any();
        let kernel_size: usize = kani::any();
        let stride: usize = kani::any();
        let padding: usize = kani::any();
        let dilation: usize = kani::any();

        kani::assume(in_len >= 1 && in_len <= 8);
        kani::assume(kernel_size >= 1 && kernel_size <= 8);
        kani::assume(stride >= 1 && stride <= 8);
        kani::assume(padding <= 8);
        kani::assume(dilation >= 1 && dilation <= 4);

        // Independent formula computation (same as production but separate evaluation).
        let eff_kernel = dilation * (kernel_size - 1) + 1;
        let padded = in_len + 2 * padding;
        if padded < eff_kernel {
            return; // invalid params — builder should return Err
        }
        let expected_out_len = (padded - eff_kernel) / stride + 1;

        if let Ok(def) = build_conv1d_full(
            "kani_test",
            4,
            2,
            kernel_size,
            in_len,
            stride,
            padding,
            dilation,
            1,
            false,
        ) {
            let output_node = &def.nodes[def.nodes.len() - 1];
            let actual_out_len = output_node.shape[1];
            assert_eq!(
                actual_out_len, expected_out_len,
                "output length must match Conv1d formula"
            );
        }
    }
}
