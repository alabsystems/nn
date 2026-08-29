// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for trace-to-graph translation gap-fill ops (#3557):
//!   - Powf (general exponents: -0.5, -1, 0, 3, 1/3 — the integer/finite
//!     exponent whitelist |p| <= 2^24 lowers to ny-propagate's parity-aware
//!     `PowConstantLayer`)
//!   - SwiGlu decomposition (Silu + Mul)
//!   - ScatterAdd / IndexAdd: lowered to ny's native additive accumulation
//!     layers (ny 3f91a42f2), whose endpoints round OUTWARD per endpoint
//!     (`add_down_f32` below, `add_up_f32` above)
//!   - GridSample: REFUSED fail-closed (ny hardening 25dee0c5e) — the former
//!     identity passthrough was unsound (coordinate remap + zero padding can
//!     produce values outside the input's per-element interval).
//!
//! Uses programmatic `ComputationGraph` construction to test TraceOp variants
//! that DynTensor may decompose at trace time.

use super::common::{assert_bounds_valid, uniform_bounds};
use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;
use nn_verify::trace_to_graph_model;

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

/// Build a bounded tensor over a flat multi-input stack from per-slot bounds.
fn stacked_bounds(len: usize, at: impl Fn(usize) -> (f32, f32)) -> nn_verify::BoundedTensor {
    nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[len]), |ix| at(ix[0]).0),
        ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[len]), |ix| at(ix[0]).1),
    )
    .expect("valid stacked bounds")
}

/// Assert `[got_lo, got_hi]` encloses the f64-exact `[want_lo, want_hi]` and is
/// not materially wider.
///
/// NEVER a bit-equality check: ny's additive-scatter layers round each endpoint
/// OUTWARD (`add_down_f32` for the lower output, `add_up_f32` for the upper),
/// so an accumulated endpoint may sit an ulp beyond an exact f32 recomputation.
/// That direction is the sound one; the tolerance below allows it while still
/// rejecting a bound that drifted the WRONG way or blew up.
fn assert_accumulation_encloses(
    label: &str,
    index: usize,
    got_lo: f32,
    got_hi: f32,
    want_lo: f64,
    want_hi: f64,
) {
    const EPS: f64 = 1e-5;
    assert!(
        got_lo.is_finite() && got_hi.is_finite(),
        "{label}[{index}] bounds must be finite, got [{got_lo}, {got_hi}]"
    );
    assert!(
        got_lo <= got_hi,
        "{label}[{index}] lower {got_lo} must be <= upper {got_hi}"
    );
    assert!(
        f64::from(got_lo) <= want_lo + EPS && f64::from(got_hi) >= want_hi - EPS,
        "{label}[{index}] bounds [{got_lo}, {got_hi}] must ENCLOSE the accumulated range \
         [{want_lo}, {want_hi}]"
    );
    assert!(
        f64::from(got_lo) >= want_lo - EPS && f64::from(got_hi) <= want_hi + EPS,
        "{label}[{index}] bounds [{got_lo}, {got_hi}] are looser than the exact accumulated \
         range [{want_lo}, {want_hi}]"
    );
}

// ---------------------------------------------------------------------------
// Powf: extended exponent support
// ---------------------------------------------------------------------------

/// Powf(0) = constant 1.0.
#[test]
fn test_powf_zero_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow0".into(),
            TraceOp::Powf { exponent: 0.0 },
            vec![0],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(0) translation")
        .graph;
    let input_bounds = uniform_bounds(&[2, 4], 10.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // x^0 = 1.0, so output bounds should be [1.0, 1.0].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!((v - 1.0).abs() < 1e-5, "Powf(0) lo should be ~1.0, got {v}");
    }
    for &v in hi.iter() {
        assert!((v - 1.0).abs() < 1e-5, "Powf(0) hi should be ~1.0, got {v}");
    }
}

/// Powf(-1) = Reciprocal.
#[test]
fn test_powf_neg1_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![2, 3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow_neg1".into(),
            TraceOp::Powf { exponent: -1.0 },
            vec![0],
            vec![2, 3],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(-1) translation")
        .graph;
    // Use positive-only bounds to avoid division by zero.
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 3]), 0.5_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 3]), 2.0_f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // 1/x for x in [0.5, 2.0] → output in [0.5, 2.0].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= 0.5 - 1e-4, "Powf(-1) lo >= 0.5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 2.0 + 1e-4, "Powf(-1) hi <= 2.0, got {v}");
    }
}

/// Powf(-0.5) = 1/sqrt(x).
#[test]
fn test_powf_neg_half_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow_neghalf".into(),
            TraceOp::Powf { exponent: -0.5 },
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(-0.5) translation")
        .graph;
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4]), 1.0_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4]), 4.0_f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // 1/sqrt(x) for x in [1, 4] → output in [0.5, 1.0].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= 0.5 - 1e-3, "Powf(-0.5) lo >= 0.5, got {v}");
    }
    for &v in hi.iter() {
        assert!(v <= 1.0 + 1e-3, "Powf(-0.5) hi <= 1.0, got {v}");
    }
}

/// Powf(3) = x^3 (general case via Exp(3*Log(Abs(x)))).
#[test]
fn test_powf_cube_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![2, 2],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow3".into(),
            TraceOp::Powf { exponent: 3.0 },
            vec![0],
            vec![2, 2],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(3) translation")
        .graph;
    // Use positive bounds to keep the decomposition in valid domain.
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), 1.0_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[2, 2]), 3.0_f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // x^3 for x in [1, 3] → output in [1, 27].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= 1.0 - 1e-2, "Powf(3) lo >= 1.0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "Powf(3) hi must be finite, got {v}");
    }
}

/// Powf(1/3) = cube root (general case).
#[test]
fn test_powf_third_exponent_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "pow_third".into(),
            TraceOp::Powf {
                exponent: 1.0 / 3.0,
            },
            vec![0],
            vec![3],
            DType::F32,
        ),
    ]);

    let gn = trace_to_graph_model(&graph)
        .expect("Powf(1/3) translation")
        .graph;
    let input_bounds = nn_verify::BoundedTensor::new(
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[3]), 1.0_f32),
        ndarray::ArrayD::from_elem(ndarray::IxDyn(&[3]), 8.0_f32),
    )
    .expect("valid bounds");
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    // x^(1/3) for x in [1, 8] → output in [1, 2].
    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v >= 1.0 - 1e-2, "Powf(1/3) lo >= 1.0, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "Powf(1/3) hi must be finite, got {v}");
    }
}

// ---------------------------------------------------------------------------
// SwiGlu: Silu(gate) * up
// ---------------------------------------------------------------------------

/// SwiGlu with two inputs: gate and up.
#[test]
fn test_swiglu_ibp() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "gate".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "up".into(),
            TraceOp::Input,
            vec![],
            vec![2, 4],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "swiglu".into(),
            TraceOp::SwiGlu,
            vec![0, 1],
            vec![2, 4],
            DType::F32,
        ),
    ]);

    let gn = nn_verify::trace_to_graph_model_multi_input(&graph)
        .expect("SwiGlu translation")
        .graph;

    // Multi-input: stacked [2*4 + 2*4] = [16].
    let input_bounds = uniform_bounds(&[16], 2.0);
    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    for &v in lo.iter() {
        assert!(v.is_finite(), "swiglu lo must be finite, got {v}");
    }
    for &v in hi.iter() {
        assert!(v.is_finite(), "swiglu hi must be finite, got {v}");
    }
}

/// SwiGlu with insufficient inputs should error.
#[test]
fn test_swiglu_one_input_returns_error() {
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "swiglu".into(),
            TraceOp::SwiGlu,
            vec![0],
            vec![4],
            DType::F32,
        ),
    ]);

    let err = trace_to_graph_model(&graph).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("SwiGlu") || msg.contains("2 inputs"),
        "SwiGlu with 1 input should error, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// ScatterAdd
// ---------------------------------------------------------------------------
//
// ScatterAdd and IndexAdd lower to ny's native `ScatterAddLayer` /
// `IndexAddLayer` (ny 3f91a42f2 added the LayerTypes, the ny-build arms and
// the bridge lowering for layers ny-propagate already had). The refusal these
// replace was aimed at an identity `Add(x, 0)` translator — correctly, since
// ACCUMULATION moves output values outside the destination's own bounds — but
// an identity was never the only option: the real layers accumulate the lower
// bounds into the lower output and the upper into the upper, which is the
// interval image of the sum.
//
// nn records both ops as `(self, index, src)`, which is exactly the operand
// order the layers resolve. A CONSTANT index (below) takes the exact
// per-element static path; a runtime index would take the layers' looser
// additive-hull path instead.

/// ScatterAdd with constant indices accumulates end to end, duplicates
/// included.
///
/// `index = [[0, 0], [1, 3], [2, 2]]` on `axis = 1` sends BOTH of row 0's src
/// elements into `out[0][0]` and both of row 2's into `out[2][2]`, so the
/// destination bound must widen by the SUM of the two contributions — the
/// precise thing the retired identity lowering got wrong.
#[test]
fn test_scatter_add_ibp() {
    // index[r][j] — the destination column for src[r][j] along axis 1.
    const INDEX: [[usize; 2]; 3] = [[0, 0], [1, 3], [2, 2]];
    // Traced point for the two variable operands.
    const DATA: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    const SRC: [f32; 6] = [0.5, -1.5, 2.0, 0.25, -3.0, 4.0];
    const SRC_RADIUS: f32 = 0.5;

    let index_data: Vec<f32> = INDEX.iter().flatten().map(|&i| i as f32).collect();
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "dest".into(),
            TraceOp::Input,
            vec![],
            vec![3, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "idx".into(),
            TraceOp::ConstantWeight {
                weight: WeightRef::new(index_data, vec![3, 2]).expect("valid index constant"),
            },
            vec![],
            vec![3, 2],
            DType::I64,
        ),
        TraceNode::new(
            2,
            "src".into(),
            TraceOp::Input,
            vec![],
            vec![3, 2],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "scatter".into(),
            TraceOp::ScatterAdd { dim: 1 },
            vec![0, 1, 2],
            vec![3, 4],
            DType::F32,
        ),
    ]);

    let gn = nn_verify::trace_to_graph_model_multi_input(&graph)
        .expect("ScatterAdd must lower to the native accumulation layer")
        .graph;
    assert_eq!(
        count_layer_type(&gn, "ScatterAdd"),
        1,
        "ScatterAdd must lower to exactly one native ScatterAdd layer"
    );

    // Multi-input stacking is trace order over the ACTIVATION inputs only —
    // the constant index carries no graph edge: [dest (12) | src (6)].
    let input_bounds = stacked_bounds(18, |i| {
        if i < 12 {
            (DATA[i], DATA[i])
        } else {
            (SRC[i - 12] - SRC_RADIUS, SRC[i - 12] + SRC_RADIUS)
        }
    });

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let got_lo: Vec<f32> = lo.iter().copied().collect();
    let got_hi: Vec<f32> = hi.iter().copied().collect();
    assert_eq!(
        got_lo.len(),
        12,
        "ScatterAdd preserves the [3, 4] data shape"
    );

    for r in 0..3 {
        for c in 0..4 {
            let flat = r * 4 + c;
            // Independent f64 accumulation over every src element routed here.
            let mut want_lo = f64::from(DATA[flat]);
            let mut want_hi = f64::from(DATA[flat]);
            for j in 0..2 {
                if INDEX[r][j] == c {
                    want_lo += f64::from(SRC[r * 2 + j] - SRC_RADIUS);
                    want_hi += f64::from(SRC[r * 2 + j] + SRC_RADIUS);
                }
            }
            assert_accumulation_encloses(
                "scatter_add",
                flat,
                got_lo[flat],
                got_hi[flat],
                want_lo,
                want_hi,
            );
        }
    }

    // The duplicate-destination cells must be strictly wider than a passthrough
    // of `data` — the identity lowering's exact failure mode.
    for (r, c) in [(0_usize, 0_usize), (2, 2)] {
        let flat = r * 4 + c;
        assert!(
            got_hi[flat] - got_lo[flat] > 1.5,
            "out[{r}][{c}] takes TWO src contributions, so its width must exceed one \
             src interval (2 * {SRC_RADIUS}); got [{}, {}]",
            got_lo[flat],
            got_hi[flat]
        );
    }
    // A destination no index names is untouched: exactly `data`. out[0][3] —
    // row 0's only index is column 0, twice.
    let untouched = 3;
    assert!(
        (got_lo[untouched] - DATA[untouched]).abs() < 1e-6
            && (got_hi[untouched] - DATA[untouched]).abs() < 1e-6,
        "an unnamed destination must pass `data` through exactly, got [{}, {}]",
        got_lo[untouched],
        got_hi[untouched]
    );
}

// ---------------------------------------------------------------------------
// IndexAdd
// ---------------------------------------------------------------------------

/// IndexAdd with constant indices accumulates whole slices end to end.
///
/// `index = [2, 0, 2]` on `axis = 0` covers all three shapes at once: row 1 of
/// `data` is untouched, row 0 receives one src row, and row 2 receives TWO —
/// the duplicate case the retired identity lowering could not enclose.
#[test]
fn test_index_add_ibp() {
    // index[i] — the destination row for src row i along axis 0.
    const INDEX: [usize; 3] = [2, 0, 2];
    const DATA: [f32; 12] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    const SRC: [f32; 9] = [0.5, -1.5, 2.0, 0.25, -3.0, 4.0, 1.25, -0.75, 6.0];
    const SRC_RADIUS: f32 = 0.5;

    let index_data: Vec<f32> = INDEX.iter().map(|&i| i as f32).collect();
    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "dest".into(),
            TraceOp::Input,
            vec![],
            vec![4, 3],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "idx".into(),
            TraceOp::ConstantWeight {
                weight: WeightRef::new(index_data, vec![3]).expect("valid index constant"),
            },
            vec![],
            vec![3],
            DType::I64,
        ),
        TraceNode::new(
            2,
            "src".into(),
            TraceOp::Input,
            vec![],
            vec![3, 3],
            DType::F32,
        ),
        TraceNode::new(
            3,
            "index_add".into(),
            TraceOp::IndexAdd { dim: 0 },
            vec![0, 1, 2],
            vec![4, 3],
            DType::F32,
        ),
    ]);

    let gn = nn_verify::trace_to_graph_model_multi_input(&graph)
        .expect("IndexAdd must lower to the native accumulation layer")
        .graph;
    assert_eq!(
        count_layer_type(&gn, "IndexAdd"),
        1,
        "IndexAdd must lower to exactly one native IndexAdd layer"
    );

    // Stacked activation inputs: [dest (12) | src (9)].
    let input_bounds = stacked_bounds(21, |i| {
        if i < 12 {
            (DATA[i], DATA[i])
        } else {
            (SRC[i - 12] - SRC_RADIUS, SRC[i - 12] + SRC_RADIUS)
        }
    });

    let output = gn.propagate_ibp(&input_bounds).expect("IBP propagation");
    assert_bounds_valid(&output);

    let (lo, hi) = output.lower_upper();
    let got_lo: Vec<f32> = lo.iter().copied().collect();
    let got_hi: Vec<f32> = hi.iter().copied().collect();
    assert_eq!(got_lo.len(), 12, "IndexAdd preserves the [4, 3] data shape");

    for r in 0..4 {
        for c in 0..3 {
            let flat = r * 3 + c;
            let mut want_lo = f64::from(DATA[flat]);
            let mut want_hi = f64::from(DATA[flat]);
            for (src_row, &dest_row) in INDEX.iter().enumerate() {
                if dest_row == r {
                    want_lo += f64::from(SRC[src_row * 3 + c] - SRC_RADIUS);
                    want_hi += f64::from(SRC[src_row * 3 + c] + SRC_RADIUS);
                }
            }
            assert_accumulation_encloses(
                "index_add",
                flat,
                got_lo[flat],
                got_hi[flat],
                want_lo,
                want_hi,
            );
        }
    }

    // Row 1 is named by no index: exact passthrough of `data`.
    for flat in 3..6 {
        assert!(
            (got_lo[flat] - DATA[flat]).abs() < 1e-6 && (got_hi[flat] - DATA[flat]).abs() < 1e-6,
            "row 1 is untouched and must pass `data` through exactly, got [{}, {}]",
            got_lo[flat],
            got_hi[flat]
        );
    }
    // Row 2 takes TWO src rows, so it is strictly wider than a single one.
    for flat in 6..9 {
        assert!(
            got_hi[flat] - got_lo[flat] > 1.5,
            "row 2 accumulates two src rows, so its width must exceed one src interval \
             (2 * {SRC_RADIUS}); got [{}, {}]",
            got_lo[flat],
            got_hi[flat]
        );
    }
}

// ---------------------------------------------------------------------------
// GridSample: refused fail-closed (ny 25dee0c5e)
// ---------------------------------------------------------------------------
//
// The former "conservative identity passthrough" was not a sound
// over-approximation: GridSample remaps coordinates (an output element can
// draw from ANY input element) and Zeros padding can introduce 0.0 values
// outside the input interval, so identity bounds could exclude the true
// output. The bridge refuses the op; these tests pin the refusal so a future
// un-refusal (a real sound lowering) is a conscious event.
//
// The refusal REASON matters, and it is not the one its sibling
// `ResizeBilinear` used to share. GridSample's conventions are complete on
// the wire — `padding_mode` and `align_corners` are both serialized fields —
// so nothing about coordinate conventions blocks it. What blocks it is
// STRUCTURAL: the sampling grid is a runtime tensor operand, so the
// output-to-input coordinate map is a function of a runtime value and cannot
// be frozen into a layer at translate time. (`ResizeBilinear`, whose geometry
// IS trace-time-known, is lowered — see
// `trace_to_graph_model_resize_moe.rs`.) Asserting the reason, not just the
// refusal, keeps a future porter from looking for the wrong fix.

/// Shared refusal assertion for the GridSample fixtures.
fn assert_grid_sample_refused(graph: &ComputationGraph) {
    let err = nn_verify::trace_to_graph_model_multi_input(graph).expect_err(
        "GridSample must be refused fail-closed (ny 25dee0c5e; \
         the former identity passthrough was unsound)",
    );
    match err {
        nn_verify::VerifyError::Ny(ny_api::NyError::UnsupportedOp(msg)) => {
            assert!(
                msg.contains("GridSample")
                    && msg.contains("runtime tensor operand")
                    && msg.contains("value-dependent"),
                "refusal must name GridSample and the runtime-grid structural reason \
                 (NOT a coordinate-convention reason: padding_mode and align_corners both \
                 ride the wire), got: {msg}"
            );
        }
        other => panic!("expected VerifyError::Ny(UnsupportedOp), got: {other:?}"),
    }
}

/// GridSample with Zeros padding + align_corners is refused fail-closed.
#[test]
fn test_grid_sample_zeros_padding_refused_fail_closed() {
    use nn_core::dyn_tensor::GridSamplePaddingMode;

    let graph = ComputationGraph::from_nodes(vec![
        TraceNode::new(
            0,
            "input".into(),
            TraceOp::Input,
            vec![],
            vec![1, 3, 4, 4],
            DType::F32,
        ),
        TraceNode::new(
            1,
            "grid".into(),
            TraceOp::Input,
            vec![],
            vec![1, 2, 2, 2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "sample".into(),
            TraceOp::GridSample {
                padding_mode: GridSamplePaddingMode::Zeros,
                align_corners: true,
            },
            vec![0, 1],
            vec![1, 3, 2, 2],
            DType::F32,
        ),
    ]);

    assert_grid_sample_refused(&graph);
}

/// GridSample with Border padding (no align_corners) is refused fail-closed —
/// the refusal is padding-mode-independent.
#[test]
fn test_grid_sample_border_padding_refused_fail_closed() {
    use nn_core::dyn_tensor::GridSamplePaddingMode;

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
            "grid".into(),
            TraceOp::Input,
            vec![],
            vec![1, 3, 3, 2],
            DType::F32,
        ),
        TraceNode::new(
            2,
            "sample".into(),
            TraceOp::GridSample {
                padding_mode: GridSamplePaddingMode::Border,
                align_corners: false,
            },
            vec![0, 1],
            vec![1, 1, 3, 3],
            DType::F32,
        ),
    ]);

    assert_grid_sample_refused(&graph);
}
