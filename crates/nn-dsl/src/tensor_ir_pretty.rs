// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Pretty-print a `TensorKernelDef` as a human-readable IR dump.

#[path = "tensor_ir_pretty_layers.rs"]
mod pretty_layers;

use super::{ReduceOp, TensorKernelDef, TensorOpKind};

/// Pretty-print a `TensorKernelDef` as a human-readable IR dump.
///
/// Format:
/// ```text
/// tensor_kernel instance_norm {
///   %0 = input("x", [4, 32, 128])
///   %1 = reduce_mean(%0, axis=2) -> [4, 32]
///   %2 = elementwise(square, [%0]) -> [4, 32, 128]
///   ...
///   return %N
/// }
/// ```
#[must_use]
pub fn tensor_ir_pretty_print(kernel: &TensorKernelDef) -> String {
    let mut out = String::new();
    out.push_str(&format!("tensor_kernel {} {{\n", kernel.name));

    for node in &kernel.nodes {
        out.push_str(&format!("  %{} = ", node.id.index()));

        // Try layer-level formatting first (extracted to tensor_ir_pretty_layers.rs).
        if let Some(s) = pretty_layers::format_layer_node(&node.kind) {
            out.push_str(&s);
        } else {
            // Core structural ops.
            match &node.kind {
                TensorOpKind::Input { name, shape } => {
                    out.push_str(&format!("input(\"{name}\", {shape:?})"));
                }
                TensorOpKind::Reshape {
                    input,
                    target_shape,
                } => {
                    out.push_str(&format!("reshape(%{}, {:?})", input.index(), target_shape));
                }
                TensorOpKind::AxisSelect { input, axis, index } => {
                    out.push_str(&format!(
                        "axis_select(%{}, axis={}, index={})",
                        input.index(),
                        axis,
                        index
                    ));
                }
                TensorOpKind::Stack { inputs, axis } => {
                    let refs: Vec<String> =
                        inputs.iter().map(|id| format!("%{}", id.index())).collect();
                    out.push_str(&format!("stack([{}], axis={})", refs.join(", "), axis));
                }
                TensorOpKind::Concat { inputs, axis } => {
                    let refs: Vec<String> =
                        inputs.iter().map(|id| format!("%{}", id.index())).collect();
                    out.push_str(&format!("concat([{}], axis={})", refs.join(", "), axis));
                }
                TensorOpKind::Reduce {
                    op,
                    input,
                    axis,
                    keepdim,
                } => {
                    let op_name = match op {
                        ReduceOp::Sum => "reduce_sum",
                        ReduceOp::Mean => "reduce_mean",
                        ReduceOp::Max => "reduce_max",
                        ReduceOp::Min => "reduce_min",
                    };
                    if *keepdim {
                        out.push_str(&format!(
                            "{}(%{}, axis={}, keepdim=true)",
                            op_name,
                            input.index(),
                            axis
                        ));
                    } else {
                        out.push_str(&format!("{}(%{}, axis={})", op_name, input.index(), axis));
                    }
                }
                TensorOpKind::Elementwise { kernel, inputs } => {
                    let refs: Vec<String> =
                        inputs.iter().map(|id| format!("%{}", id.index())).collect();
                    out.push_str(&format!(
                        "elementwise({}, [{}])",
                        kernel.name,
                        refs.join(", ")
                    ));
                }
                TensorOpKind::Broadcast {
                    input,
                    target_shape,
                    alignment,
                } => {
                    out.push_str(&format!(
                        "broadcast(%{}, {:?}, {:?})",
                        input.index(),
                        target_shape,
                        alignment
                    ));
                }
                // Graceful fallback for unhandled #[non_exhaustive] variants.
                // Per design doc (#1424): unreachable!() on #[non_exhaustive] enums
                // must be error returns or safe fallbacks, not panics.
                other => {
                    out.push_str(&format!("unknown_op({:?})", std::mem::discriminant(other)));
                }
            }
        }
        out.push_str(&format!(" -> {:?}\n", node.shape));
    }

    out.push_str(&format!("  return %{}\n", kernel.output.index()));
    out.push_str("}\n");
    out
}

impl std::fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sum => write!(f, "sum"),
            Self::Mean => write!(f, "mean"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
        }
    }
}
