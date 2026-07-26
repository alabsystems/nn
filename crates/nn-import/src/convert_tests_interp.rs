// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Wave 6 op mapping tests: interpolate, scatter, reflection_pad2d, clamp_max.
//!
//! These ops are needed for dpdf model conversion (FPN upsample, embedding
//! scatter, conv preprocessing, activation clamping).

use std::collections::HashMap;

use nn_core::dyn_tensor::trace::TraceOp;

use crate::op_map::{map_node_to_trace_op, OpMapContext, ResolvedWeight};
use crate::parse::{
    Argument, ArgumentFloat, ArgumentInt, ArgumentInts, ArgumentString, ArgumentTensor,
    NamedArgument, Node, TensorArgument, TensorMeta,
};

// ---------------------------------------------------------------------------
// Helpers (same pattern as convert_tests_dpdf_ops.rs)
// ---------------------------------------------------------------------------

fn make_node(target: &str, inputs: Vec<NamedArgument>) -> Node {
    Node {
        target: target.to_string(),
        inputs,
        outputs: vec![],
        metadata: HashMap::default(),
    }
}

fn tensor_arg(name: &str, tensor_name: &str) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Tensor(ArgumentTensor {
            as_tensor: TensorArgument {
                name: tensor_name.to_string(),
            },
        }),
        kind: Some(1),
    }
}

fn int_arg(name: &str, val: i64) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Int(ArgumentInt { as_int: val }),
        kind: Some(1),
    }
}

fn float_arg(name: &str, val: f64) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Float(ArgumentFloat { as_float: val }),
        kind: Some(1),
    }
}

fn ints_arg(name: &str, vals: &[i64]) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Ints(ArgumentInts {
            as_ints: vals.to_vec(),
        }),
        kind: Some(1),
    }
}

fn string_arg(name: &str, val: &str) -> NamedArgument {
    NamedArgument {
        name: name.to_string(),
        arg: Argument::Str(ArgumentString {
            as_string: val.to_string(),
        }),
        kind: Some(1),
    }
}

fn empty_ctx() -> OpMapContext<'static> {
    let tensor_meta: &'static HashMap<String, TensorMeta> = Box::leak(Box::new(HashMap::new()));
    let weights: &'static HashMap<String, ResolvedWeight> = Box::leak(Box::new(HashMap::new()));
    OpMapContext {
        tensor_meta,
        weights,
    }
}

// ---------------------------------------------------------------------------
// Interpolate tests
// ---------------------------------------------------------------------------

#[test]
fn test_map_interpolate_nearest() {
    let node = make_node(
        "torch.ops.aten.interpolate.default",
        vec![
            tensor_arg("input", "x"),
            string_arg("mode", "nearest"),
            float_arg("scales_h", 3.0),
            float_arg("scales_w", 3.0),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { scale_h, scale_w, .. }
            if (scale_h - 3.0).abs() < 1e-6 && (scale_w - 3.0).abs() < 1e-6),
        "expected Upsample2d nearest with scale 3.0, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_interpolate_bilinear() {
    let node = make_node(
        "torch.ops.aten.interpolate.default",
        vec![
            tensor_arg("input", "feat"),
            string_arg("mode", "bilinear"),
            float_arg("scales_h", 2.0),
            float_arg("scales_w", 2.0),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { .. }),
        "expected Upsample2d bilinear, got {op:?}"
    );
    assert_eq!(inputs, vec!["feat"]);
}

#[test]
fn test_map_interpolate_vec_overload() {
    let node = make_node(
        "torch.ops.aten.interpolate.vec",
        vec![
            tensor_arg("input", "x"),
            string_arg("mode", "nearest"),
            float_arg("scale_factor", 4.0),
        ],
    );
    let ctx = empty_ctx();
    let (op, _inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { scale_h, scale_w, .. }
            if (scale_h - 4.0).abs() < 1e-6 && (scale_w - 4.0).abs() < 1e-6),
        "expected scale_factor=4.0 for both h/w, got {op:?}"
    );
}

#[test]
fn test_map_interpolate_bicubic() {
    let node = make_node(
        "torch.ops.aten.interpolate.default",
        vec![
            tensor_arg("input", "x"),
            string_arg("mode", "bicubic"),
            float_arg("scales_h", 2.0),
            float_arg("scales_w", 2.0),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { .. }),
        "expected Upsample2d bicubic, got {op:?}"
    );
    assert_eq!(inputs, vec!["x"]);
}

#[test]
fn test_map_interpolate_unsupported_mode() {
    let node = make_node(
        "torch.ops.aten.interpolate.default",
        vec![tensor_arg("input", "x"), string_arg("mode", "area")],
    );
    let ctx = empty_ctx();
    let result = map_node_to_trace_op(&node, &ctx, 4);
    assert!(result.is_err(), "area mode should be unsupported");
}

#[test]
fn test_map_interpolate_default_mode() {
    // No mode arg → defaults to "nearest"
    let node = make_node(
        "torch.ops.aten.interpolate.default",
        vec![
            tensor_arg("input", "x"),
            float_arg("scales_h", 2.0),
            float_arg("scales_w", 2.0),
        ],
    );
    let ctx = empty_ctx();
    let (op, _) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Upsample2d { .. }),
        "expected Upsample2d (default nearest), got {op:?}"
    );
}

// ---------------------------------------------------------------------------
// Scatter tests
// ---------------------------------------------------------------------------

#[test]
fn test_map_scatter_src() {
    let node = make_node(
        "torch.ops.aten.scatter.src",
        vec![
            tensor_arg("self", "base"),
            int_arg("dim", 1),
            tensor_arg("index", "idx"),
            tensor_arg("src", "values"),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Scatter { dim: 1 }),
        "expected Scatter dim=1, got {op:?}"
    );
    assert_eq!(inputs, vec!["base", "idx", "values"]);
}

#[test]
fn test_map_scatter_dim_0() {
    let node = make_node(
        "torch.ops.aten.scatter.src",
        vec![
            tensor_arg("self", "out"),
            int_arg("dim", 0),
            tensor_arg("index", "positions"),
            tensor_arg("src", "updates"),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Scatter { dim: 0 }),
        "expected Scatter dim=0, got {op:?}"
    );
    assert_eq!(inputs, vec!["out", "positions", "updates"]);
}

// ---------------------------------------------------------------------------
// Reflection pad 2D tests
// ---------------------------------------------------------------------------

#[test]
fn test_map_reflection_pad2d() {
    let node = make_node(
        "torch.ops.aten.reflection_pad2d.default",
        vec![
            tensor_arg("self", "feature"),
            ints_arg("padding", &[1, 1, 1, 1]),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReflectionPad2d {
                pad_left: 1,
                pad_right: 1,
                pad_top: 1,
                pad_bottom: 1,
            }
        ),
        "expected ReflectionPad2d(1,1,1,1), got {op:?}"
    );
    assert_eq!(inputs, vec!["feature"]);
}

#[test]
fn test_map_reflection_pad2d_asymmetric() {
    let node = make_node(
        "torch.ops.aten.reflection_pad2d.default",
        vec![
            tensor_arg("self", "img"),
            ints_arg("padding", &[2, 3, 1, 4]),
        ],
    );
    let ctx = empty_ctx();
    let (op, _) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReflectionPad2d {
                pad_left: 2,
                pad_right: 3,
                pad_top: 1,
                pad_bottom: 4,
            }
        ),
        "expected ReflectionPad2d(2,3,1,4), got {op:?}"
    );
}

// ---------------------------------------------------------------------------
// Clamp max tests
// ---------------------------------------------------------------------------

#[test]
fn test_map_clamp_max() {
    let node = make_node(
        "torch.ops.aten.clamp_max.default",
        vec![tensor_arg("self", "activation"), float_arg("max", 6.0)],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(op, TraceOp::Clamp { min: None, max: Some(m) } if (m - 6.0).abs() < 1e-6),
        "expected Clamp(None, Some(6.0)), got {op:?}"
    );
    assert_eq!(inputs, vec!["activation"]);
}

#[test]
fn test_map_clamp_max_positional() {
    // Second positional arg as max value
    let node = make_node(
        "torch.ops.aten.clamp_max.default",
        vec![tensor_arg("self", "x"), float_arg("value", 1.0)],
    );
    let ctx = empty_ctx();
    let (op, _) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    // Should pick up the float from positional fallback
    assert!(
        matches!(op, TraceOp::Clamp { min: None, .. }),
        "expected Clamp with min=None, got {op:?}"
    );
}

// ---------------------------------------------------------------------------
// map_pad routing tests (reflect 1D vs 2D)
// ---------------------------------------------------------------------------

#[test]
fn test_map_pad_reflect_2d() {
    let node = make_node(
        "torch.ops.aten.pad.default",
        vec![
            tensor_arg("self", "img"),
            ints_arg("pad", &[1, 1, 2, 2]),
            string_arg("mode", "reflect"),
        ],
    );
    let ctx = empty_ctx();
    let (op, inputs) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReflectionPad2d {
                pad_left: 1,
                pad_right: 1,
                pad_top: 2,
                pad_bottom: 2,
            }
        ),
        "expected ReflectionPad2d for 4-element reflect pad, got {op:?}"
    );
    assert_eq!(inputs, vec!["img"]);
}

#[test]
fn test_map_pad_reflect_1d() {
    let node = make_node(
        "torch.ops.aten.pad.default",
        vec![
            tensor_arg("self", "audio"),
            ints_arg("pad", &[3, 3]),
            string_arg("mode", "reflect"),
        ],
    );
    let ctx = empty_ctx();
    let (op, _) = map_node_to_trace_op(&node, &ctx, 4).unwrap();
    assert!(
        matches!(
            op,
            TraceOp::ReflectionPad1d {
                pad_left: 3,
                pad_right: 3,
            }
        ),
        "expected ReflectionPad1d for 2-element reflect pad, got {op:?}"
    );
}

// ---------------------------------------------------------------------------
// SUPPORTED_ATEN_OPS coverage
// ---------------------------------------------------------------------------

#[test]
fn test_wave6_ops_in_supported_ops_list() {
    let supported = crate::op_map::supported_ops();
    let wave6_ops = [
        "aten::interpolate",
        "aten::scatter",
        "aten::reflection_pad2d",
        "aten::clamp_max",
    ];
    for op in &wave6_ops {
        assert!(
            supported.contains(op),
            "Wave 6 op {op} missing from SUPPORTED_ATEN_OPS"
        );
    }
}
