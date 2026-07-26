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
//! PixelShuffle/PixelUnshuffle/WhereCond use programmatic `ComputationGraph`
//! construction because their DynTensor implementations decompose into
//! primitives at trace time (no single `TraceOp` recorded).
//!
//! Part of #2329 acceptance criterion 2: round-trip compile+verify agreement.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{
    record_input, trace_graph, ComputationGraph, TraceNode, TraceOp,
};
use nn_core::dyn_tensor::DynTensor;
use nn_core::{DType, Device};
use nn_verify::{trace_to_graph_model, trace_to_graph_model_multi_input};

fn cpu() -> Device {
    Device::Cpu
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

// -- WhereCond: programmatic graph (3-input decomposition) --------------------
//
// INC-FINAL soundness fix: WhereCond is REFUSED. The deleted legacy
// `m·x + (1−m)·y` decomposition was unsound whenever a realizable mask value
// is not in {0, 1} (a constant 0.5 "mask" makes the true output exactly `x`,
// but the decomposition yields the x/y midpoint — bounds that can exclude
// the real output, i.e. a false "holds").

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

    // Sound refusal: WhereCond is Unsupported in the bridge's coverage
    // taxonomy — non-binary masks break the decomposition.
    {
        let err = trace_to_graph_model_multi_input(&graph)
            .expect_err("WhereCond must be refused (unsound legacy decomposition)");
        let msg = err.to_string();
        assert!(
            msg.contains("WhereCond") && msg.contains("not supported"),
            "refusal should name WhereCond, got: {msg}"
        );
    }

}

// -- RepeatInterleave: verify-path IBP (unblocked by #2399 Constant fix) ------

/// Verifies that RepeatInterleave works through the full verify path now that
/// Constant nodes are correctly translated as weight tensors.
///
/// The RepeatInterleave decomposition creates Constant nodes (from
/// `DynTensor::full()`) which previously caused dangling tensor references.
/// Part of #2399 acceptance criterion 2.
#[test]
fn test_repeat_interleave_verify_path_ibp() {
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

    // Verify path: trace_to_graph_model (previously failed with dangling ref).
    let gn = trace_to_graph_model(&graph)
        .expect("RepeatInterleave verify path should succeed after Constant fix")
        .graph;
    assert!(gn.num_nodes() > 0);

    // IBP propagation: x in [-1, 1].
    let input_bounds = uniform_bounds(&[2, 3], 1.0);
    let output = gn
        .propagate_ibp(&input_bounds)
        .expect("IBP should propagate through RepeatInterleave");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(
            v.is_finite(),
            "repeat_interleave lo must be finite, got {v}"
        );
    }
    for &v in hi.iter() {
        assert!(
            v.is_finite(),
            "repeat_interleave hi must be finite, got {v}"
        );
    }
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
