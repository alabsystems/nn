// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Round-trip tests for decomposed ops: verify-path AND compile-path accept
//! the same trace graphs, and IBP bounds propagation produces valid results.
//!
//! Covers the 8 ops from #2329 that the compile path decomposes into primitives:
//!   Flip, WhereCond, Cumsum, PixelShuffle, PixelUnshuffle, Upsample2d,
//!   Expand (in shape_softmax), RepeatInterleave (in shape_softmax).
//!
//! Posture note (ny 3f91a42f2): RepeatInterleave and WhereCond BOTH translate.
//! The counts a RepeatInterleave lowering needs do ride the wire (input 1, a
//! captured constant), and WhereCond lowers to the native 3-input `Where`
//! layer whose union IBP never evaluates the mask. The tests here assert the
//! lowerings, not the retired refusals.
//!
//! PixelShuffle/PixelUnshuffle/WhereCond use programmatic `ComputationGraph`
//! construction because their DynTensor implementations decompose into
//! primitives at trace time (no single `TraceOp` recorded).
//!
//! Part of #2329 acceptance criterion 2: round-trip compile+verify agreement.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{record_input, trace_graph, ComputationGraph, TraceNode, TraceOp};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_verify::{trace_to_graph_model, trace_to_graph_model_multi_input};

fn cpu() -> Device {
    Device::Cpu
}

/// Count nodes carrying a given NY layer type.
fn count_layer_type(graph: &nn_verify::GraphNetwork, layer_type: &str) -> usize {
    graph
        .node_names()
        .iter()
        .filter(|name| {
            graph
                .node(name)
                .is_some_and(|n| n.layer().layer_type() == layer_type)
        })
        .count()
}

// -- Flip: verify + compile round-trip ----------------------------------------

#[test]
fn test_model_trace_flip_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.flip(1)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 3]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);

    let gn = trace_to_graph_model(&graph)
        .expect("Flip translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 3], 10.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -10.0 - 1e-6, "flip lo >= -10, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 10.0 + 1e-6, "flip hi <= 10, got {v}");
    }
}

#[test]
fn test_model_trace_flip_both_paths_accept() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.flip(1)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Flip: verify path")
        .graph;
    assert!(gn.num_nodes() >= 2);

    let steps = nn_dsl::compile_trace(&graph).expect("Flip: compile path");
    assert!(!steps.is_empty());
}

// -- Cumsum: verify + compile round-trip --------------------------------------

#[test]
fn test_model_trace_cumsum_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.cumsum(1)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 4]);
    let vals = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(vals, vec![1.0, 3.0, 6.0, 10.0]);

    let gn = trace_to_graph_model(&graph)
        .expect("Cumsum translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 4], 5.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v.is_finite(), "cumsum lo must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "cumsum hi must be finite, got {v}");
    }
}

#[test]
fn test_model_trace_cumsum_both_paths_accept() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 4], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.cumsum(1)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Cumsum: verify path")
        .graph;
    assert!(gn.num_nodes() >= 2);

    let steps = nn_dsl::compile_trace(&graph).expect("Cumsum: compile path");
    assert!(!steps.is_empty());
}

// -- PixelShuffle: programmatic graph (DynTensor decomposes at trace time) ----

#[test]
fn test_model_trace_pixel_shuffle_ibp() {
    // Programmatic: [1, 4, 2, 2] with r=2 -> [1, 1, 4, 4]
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 4, 2, 2],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "ps".into(),
            TraceOp::PixelShuffle { upscale_factor: 2 },
            vec![0],
            vec![1, 1, 4, 4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("PixelShuffle translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 4, 2, 2], 20.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -20.0 - 1e-6, "pixel_shuffle lo >= -20, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 20.0 + 1e-6, "pixel_shuffle hi <= 20, got {v}");
    }
}

// -- PixelUnshuffle: programmatic graph ---------------------------------------

#[test]
fn test_model_trace_pixel_unshuffle_ibp() {
    // Programmatic: [1, 1, 4, 4] with r=2 -> [1, 4, 2, 2]
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 1, 4, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pus".into(),
            TraceOp::PixelUnshuffle {
                downscale_factor: 2,
            },
            vec![0],
            vec![1, 4, 2, 2],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("PixelUnshuffle translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 4, 4], 20.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -20.0 - 1e-6, "pixel_unshuffle lo >= -20, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 20.0 + 1e-6, "pixel_unshuffle hi <= 20, got {v}");
    }
}

// -- Upsample2d: verify + compile round-trip ----------------------------------

#[test]
fn test_model_trace_upsample2d_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.upsample_nearest_2d(2, 2)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 1, 4, 4]);

    let gn = trace_to_graph_model(&graph)
        .expect("Upsample2d translation")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 2, 2], 5.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -5.0 - 1e-6, "upsample2d lo >= -5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 5.0 + 1e-6, "upsample2d hi <= 5, got {v}");
    }
}

#[test]
fn test_model_trace_upsample2d_both_paths_accept() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &cpu()).unwrap();

    let (_result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 2, 2], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.upsample_nearest_2d(2, 2)?;
        Ok(y)
    })
    .unwrap();

    let gn = trace_to_graph_model(&graph)
        .expect("Upsample2d: verify path")
        .graph;
    assert!(gn.num_nodes() >= 2);

    let steps = nn_dsl::compile_trace(&graph).expect("Upsample2d: compile path");
    assert!(!steps.is_empty());
}

// -- WhereCond: programmatic graph (3-input native Where) ---------------------
//
// WhereCond lowers to ny's native 3-input `Where` layer (ny 3f91a42f2). The
// old refusal argued against a `m·x + (1−m)·y` mask-arithmetic decomposition —
// genuinely unsound for a non-binary mask — but ny never used it. `WhereLayer`
// takes the per-element UNION of the two branch boxes and never reads the
// condition's VALUE, so the enclosure holds for any realizable mask.

/// WhereCond translates to one `Where` layer whose IBP is the branch union,
/// asserted here against a masked select evaluated independently in f64.
#[test]
fn test_model_trace_where_cond_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "cond".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "a".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "b".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "where".into(),
            TraceOp::WhereCond,
            vec![0, 1, 2],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("WhereCond must lower to the native Where layer")
        .graph;
    assert_eq!(
        count_layer_type(&gn, "Where"),
        1,
        "WhereCond must lower to exactly one native Where layer"
    );

    // Multi-input stacking follows trace order: [cond (8) | a (8) | b (8)].
    // Per-element branch values keep the union from collapsing to one constant.
    let a_at = |i: usize| 1.0 + i as f64; // a in 1.0 ..= 8.0
    let b_at = |i: usize| -2.0 * (i as f64) - 1.0; // b in -1.0 ..= -15.0
    let cond_at = |i: usize| f64::from(u8::from(i.is_multiple_of(2))); // exact 0/1 mask
    let point = |i: usize| {
        if i < 8 {
            cond_at(i)
        } else if i < 16 {
            a_at(i - 8)
        } else {
            b_at(i - 16)
        }
    };
    // Degenerate box at a single realizable point: the true output is then a
    // definite masked select, and the union must still contain it.
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[24]), |ix| point(ix[0]) as f32),
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[24]), |ix| point(ix[0]) as f32),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.len(), 8, "Where output keeps the branch shape [2, 4]");
    for (i, (&l, &u)) in lo.iter().zip(hi.iter()).enumerate() {
        // Union of the two (degenerate) branch intervals.
        let want_lo = a_at(i).min(b_at(i));
        let want_hi = a_at(i).max(b_at(i));
        assert!(
            (f64::from(l) - want_lo).abs() < 1e-5 && (f64::from(u) - want_hi).abs() < 1e-5,
            "Where[{i}] must be the branch union [{want_lo}, {want_hi}], got [{l}, {u}]"
        );
        // The true masked select at this point, computed independently.
        let truth = if cond_at(i) != 0.0 { a_at(i) } else { b_at(i) };
        assert!(
            f64::from(l) <= truth && truth <= f64::from(u),
            "Where[{i}] bounds [{l}, {u}] must enclose the true select {truth}"
        );
    }
}

// -- RepeatInterleave: lowered (uniform counts → Reshape + Tile) --------------

/// RepeatInterleave TRANSLATES (ny 3f91a42f2). The old refusal claimed the
/// per-element repeat counts were not serialized; they are — they ride input 1
/// as a captured constant, which the bridge reads at translate time. Uniform
/// counts take the `Reshape` + `Tile` + `Reshape` decomposition (general counts
/// take Slice-per-run + Concat); only an absent or non-constant counts operand
/// is still refused.
///
/// The IBP check uses a DEGENERATE input box at the traced point, so the graph
/// output must reproduce nn's own eager `repeat_interleave` result element for
/// element — which is what pins the interleave ORDER (`[a a b b c c]`, not the
/// tile order `[a b c a b c]`) rather than merely the output shape.
#[test]
fn test_repeat_interleave_uniform_counts_ibp() {
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[2, 3], DType::F32).unwrap();
        x.set_trace_id(id);
        let repeats = DynTensor::full(&[3], 2.0, DType::F32, &cpu())?;
        let y = x.repeat_interleave(1, &repeats)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 6]);
    let eager = result.to_flat_vec::<f32>().unwrap();
    assert_eq!(
        eager,
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        "eager repeat_interleave interleaves rather than tiles"
    );

    let gn = trace_to_graph_model(&graph)
        .expect("RepeatInterleave with constant counts must translate")
        .graph;
    assert_eq!(
        count_layer_type(&gn, "Tile"),
        1,
        "uniform counts take the Reshape + Tile decomposition"
    );
    assert!(
        count_layer_type(&gn, "Reshape") >= 2,
        "the Tile is bracketed by an unsqueeze and a flatten Reshape"
    );

    // Degenerate box on the traced point: the IBP output must reproduce the
    // eager result exactly, element for element.
    let src = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 3]), |ix| src[ix[0] * 3 + ix[1]]),
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[2, 3]), |ix| src[ix[0] * 3 + ix[1]]),
    )
    .expect("valid bounds");

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    assert_eq!(lo.len(), 12, "expected the [2, 6] interleaved output");
    for (i, (&l, &u)) in lo.iter().zip(hi.iter()).enumerate() {
        // Expected value, recomputed here from the source row rather than read
        // back from `eager`: row r, column c, repeat count 2 everywhere.
        let want = f64::from(src[(i / 6) * 3 + (i % 6) / 2]);
        assert!(
            f64::from(l) <= want && want <= f64::from(u),
            "repeat_interleave[{i}] bounds [{l}, {u}] must enclose {want}"
        );
        assert!(
            (f64::from(l) - want).abs() < 1e-6 && (f64::from(u) - want).abs() < 1e-6,
            "repeat_interleave[{i}] should be exact on a degenerate box, got [{l}, {u}] \
             for {want}"
        );
        assert!(
            (l - eager[i]).abs() < 1e-6 && (u - eager[i]).abs() < 1e-6,
            "repeat_interleave[{i}] must match nn's eager value {}, got [{l}, {u}]",
            eager[i]
        );
    }

    // Compile path agrees the graph is translatable.
    let steps = nn_dsl::compile_trace(&graph).expect("RepeatInterleave: compile path");
    assert!(!steps.is_empty());
}

// -- Unfold: verify path IBP (#3094) ------------------------------------------

#[test]
fn test_model_trace_unfold_ibp_3094() {
    // [1, 1, 8] unfold(dim=2, size=4, step=2) -> [1, 1, 3, 4]
    let x = DynTensor::new(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        &[1, 1, 8],
        &cpu(),
    )
    .unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[1, 1, 8], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.unfold(2, 4, 2)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[1, 1, 3, 4]);

    let gn = trace_to_graph_model(&graph)
        .expect("Unfold translation should succeed")
        .graph;
    let input_bounds = uniform_bounds(&[1, 1, 8], 10.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // Unfold just rearranges elements — bounds should be within input range.
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -10.0 - 1e-6, "unfold lo >= -10, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 10.0 + 1e-6, "unfold hi <= 10, got {v}");
    }
}

#[test]
fn test_model_trace_unfold_1d_ibp_3094() {
    // [6] unfold(dim=0, size=3, step=2) -> [2, 3]
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[6], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.unfold(0, 3, 2)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 3]);

    let gn = trace_to_graph_model(&graph)
        .expect("Unfold 1D translation should succeed")
        .graph;
    let input_bounds = uniform_bounds(&[6], 5.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -5.0 - 1e-6, "unfold 1d lo >= -5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 5.0 + 1e-6, "unfold 1d hi <= 5, got {v}");
    }
}

#[test]
fn test_model_trace_unfold_non_overlapping_ibp_3094() {
    // [8] unfold(dim=0, size=4, step=4) -> [2, 4] (non-overlapping)
    let x = DynTensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8], &cpu()).unwrap();

    let (result, graph) = trace_graph(|| {
        let mut x = x.clone();
        let id = record_input(&[8], DType::F32).unwrap();
        x.set_trace_id(id);
        let y = x.unfold(0, 4, 4)?;
        Ok(y)
    })
    .unwrap();

    assert_eq!(result.dims(), &[2, 4]);

    let gn = trace_to_graph_model(&graph)
        .expect("Unfold non-overlapping translation")
        .graph;
    let input_bounds = uniform_bounds(&[8], 20.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -20.0 - 1e-6, "unfold no-overlap lo >= -20, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 20.0 + 1e-6, "unfold no-overlap hi <= 20, got {v}");
    }
}

// -- SliceSet: verify path IBP (#3094) ----------------------------------------

#[test]
fn test_model_trace_slice_set_ibp_3094() {
    // Programmatic graph: self=[2, 6], src=[2, 2], slice_set(dim=1, start=2)
    // Result: self[:, 0:2] + src + self[:, 4:6]
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "self_in".into(),
            TraceOp::Input,
            vec![],
            vec![2, 6],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "src_in".into(),
            TraceOp::Input,
            vec![],
            vec![2, 2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "slice_set".into(),
            TraceOp::SliceSet { dim: 1, start: 2 },
            vec![0, 1],
            vec![2, 6],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("SliceSet translation should succeed")
        .graph;

    // Multi-input: stacked [2*6 + 2*2] = [16].
    let input_bounds = uniform_bounds(&[16], 10.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -10.0 - 1e-6, "slice_set lo >= -10, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 10.0 + 1e-6, "slice_set hi <= 10, got {v}");
    }
}

#[test]
fn test_model_trace_slice_set_at_start_ibp_3094() {
    // SliceSet at the beginning: self=[4], src=[2], start=0
    // Result: src + self[2:4]
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "self_in".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "src_in".into(),
            TraceOp::Input,
            vec![],
            vec![2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "slice_set".into(),
            TraceOp::SliceSet { dim: 0, start: 0 },
            vec![0, 1],
            vec![4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("SliceSet at start translation")
        .graph;

    // Stacked [4 + 2] = [6].
    let input_bounds = uniform_bounds(&[6], 5.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -5.0 - 1e-6, "slice_set_start lo >= -5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 5.0 + 1e-6, "slice_set_start hi <= 5, got {v}");
    }
}

#[test]
fn test_model_trace_slice_set_at_end_ibp_3094() {
    // SliceSet at the end: self=[4], src=[2], start=2
    // Result: self[0:2] + src
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "self_in".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "src_in".into(),
            TraceOp::Input,
            vec![],
            vec![2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "slice_set".into(),
            TraceOp::SliceSet { dim: 0, start: 2 },
            vec![0, 1],
            vec![4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("SliceSet at end translation")
        .graph;

    // Stacked [4 + 2] = [6].
    let input_bounds = uniform_bounds(&[6], 8.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= -8.0 - 1e-6, "slice_set_end lo >= -8, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 8.0 + 1e-6, "slice_set_end hi <= 8, got {v}");
    }
}

#[test]
fn test_model_trace_slice_set_full_replace_ibp_3094() {
    // Full replacement: self=[3], src=[3], start=0
    // Result: just src (no before or after parts).
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "self_in".into(),
            TraceOp::Input,
            vec![],
            vec![3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "src_in".into(),
            TraceOp::Input,
            vec![],
            vec![3],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "slice_set".into(),
            TraceOp::SliceSet { dim: 0, start: 0 },
            vec![0, 1],
            vec![3],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model_multi_input(&graph)
        .expect("SliceSet full replace translation")
        .graph;

    // Stacked [3 + 3] = [6].
    let input_bounds = uniform_bounds(&[6], 3.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v.is_finite(), "slice_set_full lo finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "slice_set_full hi finite, got {v}");
    }
}
