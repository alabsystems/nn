// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Embedding lookup tensor op translation to NY constant-fold bounds.
//!
//! Embedding is a table lookup: `output[*, d] = weight[input[*], d]` where the
//! variable input contains integer row indices into the fixed weight table
//! `[V, D]`. For verification the table is a constant and the indices are the
//! bounded variable.
//!
//! Since *any* row of the table could be selected, the tightest sound,
//! index-agnostic bound on the output is the per-dimension box over all rows:
//!   `lo[d] = min(weight[0..V, d])`, `hi[d] = max(weight[0..V, d])`.
//! Every reachable row's value `weight[i, d]` lies in `[lo[d], hi[d]]`, so this
//! box CONTAINS every possible embedded value and is a sound over-approximation.
//!
//! Crucially, this bound is INDEPENDENT of the index tensor's interval: the
//! output box must not inherit (and must not shrink with) the indices' bounds,
//! because the table is an arbitrary (non-affine) function of the index. The
//! previous translation emitted `AddConstant(midpoint[D])` directly on the
//! `[*index_dims]` index tensor, which (1) failed IBP with `ShapeMismatch`
//! (broadcasting `[*index_dims]` against `[D]`) and (2) was unsound on the
//! accidental `index_dims == D` path, where it merely SHIFTED the index
//! interval instead of emitting the table's `[lo, hi]` spread.
//!
//! The corrected translation emits a small subgraph whose output bounds have
//! the declared shape `[*index_dims, D]` and, per dimension `d`, the fixed
//! interval `[lo[d], hi[d]]`, discarding the index interval entirely:
//!   1. `Reshape([-1])`         — flatten the indices to seed a valid graph edge.
//!   2. `Linear(zeros, midpoint)` — collapse to the per-position/per-dim midpoint
//!      `mid[d] = (lo[d]+hi[d])/2`. The zero weight makes the output a degenerate
//!      point INDEPENDENT of the index bounds (same idiom as the LSTM/attention
//!      constant injection).
//!   3. `EpsilonExpand(eps[0..D])` — a certified additive `±eps` widening with
//!      PER-ELEMENT radii that turns the degenerate point into the table's box.
//!      `eps[d]` is the PER-DIM half-spread `(hi[d]-lo[d])/2`, rounded OUTWARD
//!      (see [`outward_half_spread`]) so `[mid[d]-eps[d], mid[d]+eps[d]]`
//!      contains `[lo[d], hi[d]]` for every `d`. This is the per-dim hull —
//!      the tightest sound index-agnostic box, up to outward rounding. (The
//!      retired `QdqPerturbation` widening — whose constructor is now disabled
//!      upstream — applied one GLOBAL radius `max_d (hi[d]-lo[d])/2` to every
//!      dim, inflating each narrow dim to the widest dim's spread.) The radii
//!      are tiled across positions in the same row-major `[*index_dims, D]`
//!      order as the midpoint bias, so they align element-for-element with the
//!      Linear's flattened `[num_positions * D]` output. For a degenerate table
//!      (all rows equal -> zero spread in every dim) this step is omitted and
//!      the exact constant point is emitted.
//!   4. `Reshape([*index_dims, D])` — restore the declared output shape.
//!
//! Soundness: every value-carrying weight is zero, so the output bounds do not
//! depend on the index interval; the emitted box is a fixed, index-independent
//! per-dim hull `[lo[d], hi[d]]` (enlarged only by outward rounding) that
//! provably contains every row of the table.
//!
//! ## Why this lives in nn at all, and whose rounding scheme it uses
//!
//! The identical surrogate also exists in NY, as
//! `ny_trace_bridge::translate::ops_conv_linear::translate_embedding`. That copy
//! is **authoritative** — per `ny/docs/TRACE_BRIDGE_MIGRATION.md`,
//! `ny_trace_bridge::translate` IS nn-verify's trace intake, so every
//! `TraceOp::Embedding` goes through it and never through this file.
//!
//! nn nevertheless needs its own emitter, for a reason the migration doc states
//! explicitly: "The scalar-kernel `KernelIR` path (`nn_dsl::ir`,
//! `kernel_to_graph`) is explicitly **not** touched." This module serves that
//! second, tensor-IR intake — `TensorOpKind::Embedding` inside a
//! `TensorKernelDef` — which the bridge does not see (it consumes
//! `ny_trace_bridge::schema::TraceOp`, a different IR) and could not be called
//! for anyway: `translate_embedding` there is a private `fn` on a private `Ctx`.
//!
//! What was NOT justified was the two copies computing the outward-rounded
//! radius by two different methods. NY's is a single directed-rounding step,
//! `sub_up_f32(mid, lo).max(sub_up_f32(hi, mid))`; nn's used to be a
//! round-to-nearest seed followed by a `next_up` bump loop with an f64
//! acceptance test. [`outward_half_spread`] now mirrors NY's, so the two
//! surrogates cannot drift on the one quantity that decides whether the hull
//! encloses. `tests::outward_half_spread_matches_the_ny_bridge_scheme` pins the
//! agreement, including on the retired loop's own extreme-magnitude fixture.

use ndarray::{Array1, Array2};
use nn_dsl::tensor_ir::TensorNodeId;
use ny_propagate::layers::{EpsilonExpandLayer, LinearLayer, ReshapeLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};

use crate::error::VerifyError;
use crate::util::get_value;

use super::TensorNodeValue;

/// Knuth's `TwoSum` in f32: `(hi, lo)` with `hi == fl(a + b)` and, for finite
/// non-overflowing `a`, `b`, `a + b == hi + lo` in EXACT real arithmetic.
///
/// A local mirror of `ny_tensor::rounding`'s private helper of the same name.
/// nn-verify does not depend on `ny-tensor` (only `ny-core`/`ny-propagate`/
/// `ny-api`, none of which re-export the rounding primitives), so the four
/// lines are reproduced rather than imported.
#[inline]
fn two_sum_f32(a: f32, b: f32) -> (f32, f32) {
    let hi = a + b;
    let a_virtual = hi - b;
    let b_virtual = hi - a_virtual;
    (hi, (a - a_virtual) + (b - b_virtual))
}

/// `a - b` rounded toward `+inf`: the SMALLEST f32 that is `>= a - b` exactly.
///
/// Mirrors `ny_tensor::sub_up_f32` (`add_up_f32(a, -b)`). `TwoSum` reports
/// exactly what the subtraction lost, so this steps up only when the operation
/// was inexact — an exactly-representable difference costs nothing, which is
/// what makes the result MINIMAL as well as sound.
#[inline]
fn sub_up_f32(a: f32, b: f32) -> f32 {
    let (hi, lo) = two_sum_f32(a, -b);
    if !hi.is_finite() {
        // Mirror of NY's overflow arm: a finite exact difference that rounded to
        // -inf must come back as the largest-magnitude finite f32, never -inf.
        return if hi == f32::NEG_INFINITY && a.is_finite() && b.is_finite() {
            f32::MIN
        } else {
            hi
        };
    }
    if lo > 0.0 {
        hi.next_up()
    } else {
        hi
    }
}

/// Per-dim widening radius: the half-spread `(hi - lo) / 2` rounded OUTWARD so
/// that `[mid - eps, mid + eps]` contains `[lo, hi]` in EXACT real arithmetic.
///
/// This is NY's scheme, verbatim in effect —
/// `sub_up_f32(mid, lo).max(sub_up_f32(hi, mid))` — the same expression
/// `ny_trace_bridge`'s `translate_embedding` uses for the same surrogate (see
/// the module docs on which copy is authoritative).
///
/// # Why it is sound, in one step
///
/// The enclosure requirement is exactly `eps >= mid - lo` AND `eps >= hi - mid`
/// in the reals. `sub_up_f32` returns the smallest f32 satisfying each of those
/// individually, so their max is the smallest f32 satisfying both: minimal AND
/// sufficient, with no iteration and no f64 detour. `f32::midpoint` rounds to
/// nearest, so it can land strictly inside one of the two gaps (e.g. `lo = -1`,
/// `hi = 2^26` rounds `mid` to `2^25`, where the round-to-nearest half-spread
/// `2^25` leaves `mid - eps = 0 > lo` and drops the table minimum) — taking the
/// max of the two SEPARATELY upward-rounded gaps covers whichever side the
/// midpoint rounded toward.
///
/// `EpsilonExpandLayer` then applies the radius with directed f32 rounding
/// (`sub_down_f32(mid, eps)` / `add_up_f32(mid, eps)`), which only ever moves
/// further outward — so the applied box still contains `[lo, hi]`.
///
/// Edges: a table wider than `f32::MAX` yields a non-finite radius, which
/// `EpsilonExpandLayer::try_new` rejects — fail closed, never a silently narrow
/// box. Never NaN: `lo`, `hi` are validated finite and
/// `mid = f32::midpoint(lo, hi)` lies in `[lo, hi]`. The `.max(0.0)` pins a
/// degenerate (zero-spread) dim to a true `0`.
fn outward_half_spread(mid: f32, lo: f32, hi: f32) -> f32 {
    debug_assert!(lo <= mid && mid <= hi, "midpoint must lie within [lo, hi]");
    // Fail closed on a non-finite triple. The callers validate every table entry
    // finite before deriving `lo`/`hi`/`mid`, so this is unreachable — but
    // `f32::max` returns the NON-NaN operand, so a NaN radius would otherwise be
    // swallowed by the `.max(0.0)` below and emitted as a ZERO-width box. NY's
    // copy cannot receive a non-finite triple either (`weight_f32` rejects
    // them), so this guard is a strictly-more-conservative addition, not a
    // divergence in behaviour on any reachable input.
    if !mid.is_finite() || !lo.is_finite() || !hi.is_finite() {
        return f32::INFINITY;
    }
    sub_up_f32(mid, lo).max(sub_up_f32(hi, mid)).max(0.0)
}

/// Translate a `TensorOpKind::Embedding` node to NY graph nodes.
///
/// The weight table is a constant `[V, D]` matrix and the input indices are a
/// bounded variable. The emitted subgraph produces output bounds of the declared
/// shape `[*index_dims, D]` whose per-dim interval is the table's `[lo[d], hi[d]]`
/// spread, independent of the index interval. See the module docs for the exact
/// node sequence and the soundness argument.
pub(super) fn translate_embedding(
    node_id: TensorNodeId,
    input: TensorNodeId,
    weight: TensorNodeId,
    output_shape: &[usize],
    node_values: &[TensorNodeValue],
    graph: &mut GraphNetwork,
) -> Result<TensorNodeValue, VerifyError> {
    // Input (indices) must be a Variable graph node.
    let input_name = match get_value(node_values, input.index(), "Embedding input")? {
        TensorNodeValue::Variable(name) => name.clone(),
        TensorNodeValue::Constant(_) => {
            return Err(VerifyError::UnsupportedOp(
                "Embedding input must be a variable tensor (indices), not a constant scalar".into(),
            ));
        }
        TensorNodeValue::WeightTensor(_) => {
            return Err(VerifyError::UnsupportedOp(
                "Embedding input must be a variable tensor (indices), not a weight tensor".into(),
            ));
        }
    };

    // Weight (embedding table) must be a 2-D WeightTensor `[V, D]`.
    let weight_array = match get_value(node_values, weight.index(), "Embedding weight")? {
        TensorNodeValue::WeightTensor(arr) => {
            if arr.shape().len() != 2 {
                return Err(VerifyError::WeightValidation {
                    op: "Embedding",
                    reason: format!(
                        "weight must be 2-D [num_embeddings, embedding_dim], got {}-D",
                        arr.shape().len()
                    ),
                });
            }
            arr.clone()
        }
        TensorNodeValue::Variable(_) => {
            return Err(VerifyError::WeightValidation {
                op: "Embedding",
                reason: "weight must be a ConstantTensor binding, not a variable".into(),
            });
        }
        TensorNodeValue::Constant(_) => {
            return Err(VerifyError::WeightValidation {
                op: "Embedding",
                reason: "weight must be a ConstantTensor binding, not a constant scalar".into(),
            });
        }
    };

    let weight_shape = weight_array.shape();
    let num_embeddings = weight_shape[0];
    let embedding_dim = weight_shape[1];

    // An empty table (no rows / no columns) has no rows to bound; reject up front
    // so the per-dim min/max loop below can never produce NaN/inf extrema.
    if num_embeddings == 0 || embedding_dim == 0 {
        return Err(VerifyError::WeightValidation {
            op: "Embedding",
            reason: format!(
                "weight table must be non-empty [num_embeddings, embedding_dim], got \
                 [{num_embeddings}, {embedding_dim}]"
            ),
        });
    }

    // The declared output shape must be `[*index_dims, embedding_dim]`; its last
    // axis is the embedding dimension. Validate so a malformed spec is rejected
    // up front rather than producing a mis-shaped node.
    if output_shape.last() != Some(&embedding_dim) {
        return Err(VerifyError::WeightValidation {
            op: "Embedding",
            reason: format!(
                "declared output shape {output_shape:?} last axis must equal embedding_dim \
                 {embedding_dim}"
            ),
        });
    }
    let out_total: usize = output_shape.iter().product();
    if out_total == 0 {
        return Err(VerifyError::WeightValidation {
            op: "Embedding",
            reason: format!("declared output shape {output_shape:?} has a zero dimension"),
        });
    }
    // Number of index positions = product of all output axes except the last.
    let num_positions = out_total / embedding_dim;

    // Per-dimension extrema over all rows: lo[d] = min, hi[d] = max. Every
    // reachable row `weight[i, d]` lies in `[lo[d], hi[d]]`, so this box is the
    // tightest sound index-agnostic bound. `eps[d]` is the outward-rounded
    // per-dim half-spread that recovers `[lo[d], hi[d]]` from the midpoint.
    let mut mid = vec![0.0f32; embedding_dim];
    let mut eps = vec![0.0f32; embedding_dim];
    let mut any_spread = false;
    for d in 0..embedding_dim {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in 0..num_embeddings {
            let val = weight_array[[row, d]];
            if !val.is_finite() {
                return Err(VerifyError::WeightValidation {
                    op: "Embedding",
                    reason: format!("weight contains non-finite value at [{row}, {d}]"),
                });
            }
            if val < lo {
                lo = val;
            }
            if val > hi {
                hi = val;
            }
        }
        mid[d] = f32::midpoint(lo, hi);
        eps[d] = outward_half_spread(mid[d], lo, hi);
        if eps[d] > 0.0 {
            any_spread = true;
        }
    }

    let base = format!("t{}", node_id.index());

    // 1. Flatten the indices to 1-D so the collapse Linear sees a known
    //    `in_features`. The reshape only seeds a valid graph edge; its values
    //    are discarded by the zero weight below. `in_features` here is the index
    //    tensor's total element count; it can differ from `num_positions` if the
    //    index tensor and the output share index dims but we always overwrite the
    //    values via the zero weight, so only its total matters for the edge.
    let flat_name = format!("{base}_idx_flat");
    graph.add_node(GraphNode::new(
        flat_name.clone(),
        Layer::Reshape(ReshapeLayer::new(vec![-1])),
        vec![input_name],
    ));

    // The flattened index length equals the index tensor's total element count.
    // For an embedding it is exactly `num_positions` (one integer per output
    // position), so the zero-weight Linear's `in_features` is `num_positions`.
    let in_features = num_positions;

    // 2. Collapse to the per-position/per-dim midpoint via a ZERO weight: the
    //    output is exactly `bias` (a degenerate point) regardless of the index
    //    bounds, so the result is fully decoupled from the index interval. The
    //    bias broadcasts the per-dim midpoint `mid[d]` across all positions in
    //    row-major `[*index_dims, D]` order.
    let mut midpoint_flat = Vec::with_capacity(out_total);
    for _pos in 0..num_positions {
        midpoint_flat.extend_from_slice(&mid);
    }
    let collapse = LinearLayer::new(
        Array2::zeros((out_total, in_features)),
        Some(Array1::from_vec(midpoint_flat)),
    )
    .map_err(|e| VerifyError::InternalTranslationError {
        context: format!("Embedding midpoint Linear construction failed: {e}"),
    })?;
    let mid_name = format!("{base}_mid");
    graph.add_node(GraphNode::new(
        mid_name.clone(),
        Layer::Linear(collapse),
        vec![flat_name],
    ));

    // 3. Widen the degenerate midpoint point into the per-dim hull via
    //    `EpsilonExpand`, a certified additive `x ± eps` widening with
    //    per-element radii: `[mid[d]-eps[d], mid[d]+eps[d]] ⊇ [lo[d], hi[d]]`
    //    per dim (outward rounding in `outward_half_spread`), the tightest
    //    sound index-agnostic box. The widening is independent of the input
    //    bounds, preserving the index-agnostic property. The radii are tiled
    //    per position in the SAME row-major `[*index_dims, D]` order as the
    //    midpoint bias, so the flat `[out_total]` epsilon tensor aligns
    //    element-for-element with the Linear's flattened output (its trailing
    //    shape exactly). For a degenerate table (zero spread in every dim) we
    //    skip this and emit the exact constant point.
    let widened_name =
        if any_spread {
            let mut eps_flat = Vec::with_capacity(out_total);
            for _pos in 0..num_positions {
                eps_flat.extend_from_slice(&eps);
            }
            let expand = EpsilonExpandLayer::try_new(Array1::from_vec(eps_flat).into_dyn())
                .map_err(|e| VerifyError::InternalTranslationError {
                    context: format!("Embedding per-dim epsilon widening construction failed: {e}"),
                })?;
            let box_name = format!("{base}_box");
            graph.add_node(GraphNode::new(
                box_name.clone(),
                Layer::EpsilonExpand(expand),
                vec![mid_name],
            ));
            box_name
        } else {
            mid_name
        };

    // 4. Restore the declared `[*index_dims, D]` output shape.
    let target: Vec<i64> = output_shape.iter().map(|&d| d as i64).collect();
    let node_name = base;
    graph.add_node(GraphNode::new(
        node_name.clone(),
        Layer::Reshape(ReshapeLayer::new(target)),
        vec![widened_name],
    ));

    Ok(TensorNodeValue::Variable(node_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundedTensor;
    use ndarray::{ArrayD, IxDyn};
    use ny_propagate::NETWORK_INPUT;

    /// Run `translate_embedding` on a `[V, D]` table with the network input as
    /// the index variable; returns the graph (output already set) and the
    /// emitted node's base name.
    fn translate(weight: ArrayD<f32>, output_shape: &[usize]) -> (GraphNetwork, String) {
        let node_values = vec![
            TensorNodeValue::Variable(NETWORK_INPUT.to_string()),
            TensorNodeValue::WeightTensor(weight),
        ];
        let mut graph = GraphNetwork::new();
        let result = translate_embedding(
            TensorNodeId::new(2),
            TensorNodeId::new(0),
            TensorNodeId::new(1),
            output_shape,
            &node_values,
            &mut graph,
        )
        .expect("embedding translation should succeed");
        let name = match result {
            TensorNodeValue::Variable(name) => name,
            other => panic!("expected Variable output, got {other:?}"),
        };
        graph.set_output(name.clone());
        (graph, name)
    }

    fn table(rows: &[&[f32]]) -> ArrayD<f32> {
        let v = rows.len();
        let d = rows[0].len();
        let data: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        ArrayD::from_shape_vec(IxDyn(&[v, d]), data).expect("valid table shape")
    }

    /// Index bounds are irrelevant to the emitted box (zero-weight collapse);
    /// any finite interval works.
    fn index_bounds(num_positions: usize) -> BoundedTensor {
        BoundedTensor::new(
            ArrayD::from_elem(IxDyn(&[num_positions]), 0.0f32),
            ArrayD::from_elem(IxDyn(&[num_positions]), 3.0f32),
        )
        .expect("valid index bounds")
    }

    // -- outward_half_spread --

    #[test]
    fn outward_half_spread_is_exact_on_dyadic_spreads() {
        // (-1, 3): mid = 1, half-spread 2 — all exactly representable, no bump.
        assert_eq!(outward_half_spread(1.0, -1.0, 3.0), 2.0);
        // Degenerate interval: zero radius, pinned exactly.
        assert_eq!(outward_half_spread(5.0, 5.0, 5.0), 0.0);
    }

    /// The RETIRED nn scheme, kept as the "before" oracle: a round-to-nearest
    /// seed plus a `next_up` bump loop gated on an f64 acceptance check.
    fn retired_loop_half_spread(mid: f32, lo: f32, hi: f32) -> f32 {
        let mut eps = f32::max(mid - lo, hi - mid);
        // The loop is unbounded in the original; cap it here so a regression
        // fails the test instead of hanging it.
        for _ in 0..4096 {
            if f64::from(mid) - f64::from(eps) <= f64::from(lo)
                && f64::from(mid) + f64::from(eps) >= f64::from(hi)
            {
                return eps;
            }
            eps = eps.next_up();
        }
        panic!("retired oracle failed to converge for mid={mid} lo={lo} hi={hi}");
    }

    /// BEFORE/AFTER measurement (task 5). nn's radius scheme is now NY's
    /// directed-rounding one (`sub_up_f32(mid, lo).max(sub_up_f32(hi, mid))`),
    /// the same expression `ny_trace_bridge`'s embedding surrogate uses. This
    /// pins that the swap changed NO bound: the radii are bit-identical to the
    /// retired loop's across the full f32 range, so the emitted hull is
    /// unchanged — the win is that the two intakes can no longer drift, and the
    /// soundness argument is one directed-rounding step instead of a loop plus
    /// an f64 ULP-accounting argument.
    #[test]
    fn outward_half_spread_matches_the_ny_bridge_scheme() {
        // Deterministic xorshift over raw f32 bit patterns, so the sweep covers
        // subnormals and near-overflow magnitudes, not just tame decimals.
        let mut state: u64 = 0x2026_0827_dead_beef;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        let mut checked = 0usize;
        let mut bumped = 0usize;
        for i in 0..60_000u64 {
            let (a, b) = if i % 2 == 0 {
                let mk = |bits: u32| f32::from_bits(bits);
                (mk(next() as u32), mk(next() as u32))
            } else {
                let tame = |r: u64| (r % 2_000_001) as f32 / 1000.0 - 1000.0;
                (tame(next()), tame(next()))
            };
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            let mid = f32::midpoint(lo, hi);
            if !mid.is_finite() {
                continue;
            }

            let now = outward_half_spread(mid, lo, hi);
            let before = retired_loop_half_spread(mid, lo, hi);
            assert_eq!(
                now.to_bits(),
                before.to_bits(),
                "radius drift at lo={lo:e} hi={hi:e}: NY scheme {now:e} vs retired loop {before:e}"
            );

            // Soundness, independently of the oracle: the radius must dominate
            // BOTH gaps, so the widened box contains the whole per-dim hull.
            assert!(
                f64::from(mid) - f64::from(now) <= f64::from(lo)
                    && f64::from(mid) + f64::from(now) >= f64::from(hi),
                "radius {now:e} fails to enclose [{lo:e}, {hi:e}] around mid={mid:e}"
            );

            let naive = f32::max(mid - lo, hi - mid);
            if now != naive {
                bumped += 1;
            }
            checked += 1;
        }
        assert!(checked > 10_000, "sweep must actually cover cases");
        assert!(
            bumped > 0,
            "the sweep must exercise at least one case where round-to-nearest under-covers"
        );
        println!("outward_half_spread: {checked} pairs agreed bit-for-bit ({bumped} needed an outward bump)");
    }

    #[test]
    fn outward_half_spread_bumps_at_extreme_magnitudes() {
        // lo = -1, hi = 2^26: mid rounds to 2^25 and the round-to-nearest
        // candidate max(mid - lo, hi - mid) rounds DOWN to exactly 2^25, so
        // mid - candidate = 0 > lo — the naive radius loses the lower endpoint.
        let lo = -1.0f32;
        let hi = (1u32 << 26) as f32;
        let mid = f32::midpoint(lo, hi);
        let naive = f32::max(mid - lo, hi - mid);
        assert!(
            f64::from(mid) - f64::from(naive) > f64::from(lo),
            "test premise: the naive radius must under-cover for this triple"
        );

        let eps = outward_half_spread(mid, lo, hi);
        assert!(
            eps > naive,
            "outward rounding must bump past the naive radius"
        );
        assert!(
            f64::from(mid) - f64::from(eps) <= f64::from(lo),
            "mid - eps = {} must reach lo = {lo}",
            f64::from(mid) - f64::from(eps)
        );
        assert!(
            f64::from(mid) + f64::from(eps) >= f64::from(hi),
            "mid + eps = {} must reach hi = {hi}",
            f64::from(mid) + f64::from(eps)
        );
    }

    // -- emitted subgraph structure --

    #[test]
    fn epsilon_node_carries_per_dim_radii_tiled_per_position() {
        // Per-dim hulls (all dyadic, so radii are exact):
        //   dim 0: [0, 100]  -> mid 50,   eps 50
        //   dim 1: [5, 5.5]  -> mid 5.25, eps 0.25
        //   dim 2: [7, 7]    -> mid 7,    eps 0 (degenerate dim, pinned)
        let weights = table(&[
            &[0.0, 5.0, 7.0],
            &[100.0, 5.5, 7.0],
            &[50.0, 5.25, 7.0],
            &[25.0, 5.0, 7.0],
        ]);
        let (graph, name) = translate(weights, &[2, 3]);
        assert_eq!(name, "t2");
        // Reshape -> Linear -> EpsilonExpand -> Reshape.
        assert_eq!(graph.num_nodes(), 4);

        let node = graph.node("t2_box").expect("widening node should exist");
        let Layer::EpsilonExpand(expand) = node.layer() else {
            panic!(
                "widening node should be EpsilonExpand, got {:?}",
                node.layer()
            );
        };
        // Radii align with the Linear's flat [num_positions * D] output: the
        // per-dim radii repeat per position in row-major [*index_dims, D] order
        // (the same tiling as the midpoint bias).
        assert_eq!(expand.epsilons().shape(), &[6]);
        let expected = [50.0f32, 0.25, 0.0];
        for pos in 0..2 {
            for d in 0..3 {
                assert_eq!(
                    expand.epsilons()[[pos * 3 + d]],
                    expected[d],
                    "pos {pos} dim {d}: per-dim radius tiled per position"
                );
            }
        }
    }

    /// The zero-weight Linear's own IBP applies outward directed rounding, so
    /// even an exactly-representable box comes back a few ULPs wide. Tightness
    /// assertions therefore use a small absolute tolerance; containment stays
    /// strict (outward slop only ever widens).
    const TIGHTNESS_TOL: f32 = 1e-3;

    #[test]
    fn degenerate_table_emits_exact_point_without_widening() {
        // All rows identical: zero spread in every dim -> no EpsilonExpand
        // node, and the IBP box is the constant row up to the Linear's own
        // outward rounding slop.
        let weights = table(&[&[3.5, -2.0], &[3.5, -2.0], &[3.5, -2.0]]);
        let (graph, _) = translate(weights, &[2, 2]);
        assert_eq!(
            graph.num_nodes(),
            3,
            "no widening node for a degenerate table"
        );
        assert!(graph.node("t2_box").is_none());

        let output = graph
            .propagate_ibp(&index_bounds(2))
            .expect("IBP through degenerate embedding");
        let (lo, hi) = output.lower_upper();
        assert_eq!(lo.shape(), &[2, 2]);
        let row = [3.5f32, -2.0];
        for p in 0..2 {
            for d in 0..2 {
                assert!(
                    lo[[p, d]] <= row[d] && hi[[p, d]] >= row[d],
                    "pos {p} dim {d}: [{}, {}] must contain the constant {}",
                    lo[[p, d]],
                    hi[[p, d]],
                    row[d]
                );
                assert!(
                    lo[[p, d]] >= row[d] - TIGHTNESS_TOL && hi[[p, d]] <= row[d] + TIGHTNESS_TOL,
                    "pos {p} dim {d}: [{}, {}] must be a near-exact point at {}",
                    lo[[p, d]],
                    hi[[p, d]],
                    row[d]
                );
            }
        }
    }

    // -- IBP box: per-dim hull, not the global-half-spread box --

    #[test]
    fn ibp_box_is_the_per_dim_hull() {
        // Deliberately unequal spreads: dim 0 spans 100, dim 1 spans 0.5,
        // dim 2 is a point. The retired GLOBAL widening would inflate dims 1-2
        // by the dim-0 radius (dim 1 lower would be 5.25 - 50 = -44.75); the
        // per-dim widening must reproduce each dim's own hull up to the
        // Linear's few-ULP outward rounding slop (TIGHTNESS_TOL is orders of
        // magnitude below the 50-wide global radius it guards against).
        let weights = table(&[
            &[0.0, 5.0, 7.0],
            &[100.0, 5.5, 7.0],
            &[50.0, 5.25, 7.0],
            &[25.0, 5.0, 7.0],
        ]);
        let (graph, _) = translate(weights, &[2, 3]);
        let output = graph
            .propagate_ibp(&index_bounds(2))
            .expect("IBP through embedding subgraph");
        let (lo, hi) = output.lower_upper();
        assert_eq!(lo.shape(), &[2, 3]);

        let table_lo = [0.0f32, 5.0, 7.0];
        let table_hi = [100.0f32, 5.5, 7.0];
        for p in 0..2 {
            for d in 0..3 {
                // Containment (soundness): the box holds the per-dim hull.
                assert!(
                    lo[[p, d]] <= table_lo[d] && hi[[p, d]] >= table_hi[d],
                    "pos {p} dim {d}: [{}, {}] must contain [{}, {}]",
                    lo[[p, d]],
                    hi[[p, d]],
                    table_lo[d],
                    table_hi[d]
                );
                // Tightness (the per-dim upgrade): the hull itself, not the
                // global-half-spread box.
                assert!(
                    lo[[p, d]] >= table_lo[d] - TIGHTNESS_TOL,
                    "pos {p} dim {d}: lower {} must be tight at {}",
                    lo[[p, d]],
                    table_lo[d]
                );
                assert!(
                    hi[[p, d]] <= table_hi[d] + TIGHTNESS_TOL,
                    "pos {p} dim {d}: upper {} must be tight at {}",
                    hi[[p, d]],
                    table_hi[d]
                );
            }
        }
    }

    #[test]
    fn ibp_box_contains_hull_at_extreme_magnitudes() {
        // The 2^26 table exercises the outward-rounding bump end-to-end: with
        // a round-to-nearest radius the emitted lower bound would be 0 > -1,
        // silently dropping the table minimum from the box.
        let lo_val = -1.0f32;
        let hi_val = (1u32 << 26) as f32;
        let weights = table(&[&[lo_val], &[hi_val]]);
        let (graph, _) = translate(weights, &[1, 1]);
        let output = graph
            .propagate_ibp(&index_bounds(1))
            .expect("IBP through extreme-magnitude embedding");
        let (lo, hi) = output.lower_upper();
        assert!(
            f64::from(lo[[0, 0]]) <= f64::from(lo_val),
            "lower {} must enclose table min {lo_val}",
            lo[[0, 0]]
        );
        assert!(
            f64::from(hi[[0, 0]]) >= f64::from(hi_val),
            "upper {} must enclose table max {hi_val}",
            hi[[0, 0]]
        );
    }
}
