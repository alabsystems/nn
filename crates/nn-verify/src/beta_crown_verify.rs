// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sound beta-CROWN (GenBaB branch-and-bound) verification path.
//!
//! IBP and alpha-CROWN *decorrelate* multiplicative graph nodes — `SiLU(gate) *
//! up` in SwiGLU, `Q @ Kᵀ` in attention — by bounding each factor
//! independently. For products of correlated terms this is a gross
//! over-approximation: the interval-arithmetic lower bound can be hundreds of
//! units below the *true* (Monte-Carlo-confirmed) minimum.
//!
//! ny's `BetaCrownVerifier` with the [`BranchingHeuristic::GenBaB`] heuristic
//! recovers tightness by *splitting* the nonlinear / bilinear nodes
//! (`MulBinary`, `BilinearCrown`) and recursing — GenBaB branch-and-bound. It
//! returns a **verdict**, not a tightened tensor: it answers "is
//! `objective · output ≥ threshold` for all inputs in the region?".
//!
//! # Soundness
//!
//! A [`BabVerificationStatus::Verified`] verdict is a **sound proof**: ny's
//! beta-CROWN is sound branch-and-bound (every leaf domain is closed with a
//! sound CROWN lower bound, and the union of leaf domains covers the input
//! region). There are no false positives — `Verified` implies the property
//! holds for *every* input in the box. Any other verdict
//! (`Unknown`, `Timeout`, `PotentialViolation`, `Violated`) is treated as "not
//! proven" and yields `false`; callers must NOT weaken their threshold to force
//! a pass.
//!
//! Memory is bounded soundly by ny's `NY_BAB_QUEUE_MEM_MB` (default 3072 MB):
//! when the domain queue exceeds the cap, lowest-priority domains are evicted,
//! which can only *lose* a `Verified` (turning it into `Unknown`) — never
//! produce a spurious `Verified`.
//!
//! # API
//!
//! - [`beta_crown_proves_lower_bound`] — proves `output_i ≥ threshold` for every
//!   output index `i`.
//! - [`beta_crown_proves_upper_bound`] — proves `output_i ≤ threshold` for every
//!   output index `i`.
//! - [`beta_crown_proves_width`] — proves every output coordinate lies within
//!   `[lower, upper]` (lower-bound + upper-bound combined).

use std::time::Duration;

use ny_propagate::beta_crown::NonlinearBranchingConfig;
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic, GraphNetwork,
};

use crate::BoundedTensor;

/// Default number of GenBaB branching candidates evaluated per split.
///
/// Mirrors ny's bilinear-BaB integration tests; 4 balances per-domain cost
/// against branch quality for the multiplicative SwiGLU/attention nodes.
const GENBAB_NUM_CANDIDATES: usize = 4;

/// Default ceiling on explored domains before beta-CROWN gives up (returns
/// `Unknown`). Sound: hitting the cap can only fail to prove, never falsely
/// prove.
const DEFAULT_MAX_DOMAINS: usize = 4000;

/// Build the shared GenBaB beta-CROWN config used by all helpers here.
///
/// - `use_alpha_crown`: seed each domain with alpha-CROWN bounds (~10x tighter
///   than IBP) before branching.
/// - `GenBaB` heuristic: split the nonlinear/bilinear nodes that IBP
///   decorrelates (this is the whole point — recover the correlation IBP drops).
/// - `timeout`: hard wall-clock budget; on expiry the verdict is `Timeout`.
fn genbab_config(timeout: Duration) -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(NonlinearBranchingConfig {
            num_candidates: GENBAB_NUM_CANDIDATES,
            ..Default::default()
        }),
        // false: skip the per-domain alpha-CROWN optimization. With the GenBaB
        // input-index fix the BaB descends and splits the MulBinary directly; the
        // alpha bootstrap's Graph IBP is too slow on these deep graphs to fit the
        // wall-clock budget, so a fast plain-CROWN seed + more BaB domains wins.
        use_alpha_crown: false,
        max_domains: DEFAULT_MAX_DOMAINS,
        timeout,
        ..Default::default()
    }
}

/// Run beta-CROWN once for a single objective and return whether it produced a
/// **sound** `Verified` verdict for `objective · output ≥ threshold`.
///
/// `Verified` ⇒ the property holds for every input in `input`'s box (sound).
/// Every other verdict ⇒ `false` (not proven within budget).
///
/// On a verifier error (malformed graph/objective), returns `false` — an error
/// is "not proven", never a spurious pass.
fn verify_one(
    verifier: &BetaCrownVerifier,
    graph: &GraphNetwork,
    input: &BoundedTensor,
    objective: &[f32],
    threshold: f32,
) -> bool {
    match verifier.verify_graph_relu_split(graph, input, objective, threshold) {
        Ok(result) => matches!(result.result, BabVerificationStatus::Verified),
        Err(_) => false,
    }
}

/// Soundly prove `output_i ≥ threshold` for **every** output index
/// `i ∈ [0, output_len)` via beta-CROWN GenBaB branch-and-bound.
///
/// For each index `i` the objective is the unit vector `e_i` (1.0 at `i`, 0.0
/// elsewhere), so `objective · output = output_i`; beta-CROWN then proves
/// `output_i ≥ threshold`. Returns `true` **only if every** index verifies.
///
/// `timeout` is the per-index wall-clock budget (each index gets its own
/// search). The total wall-clock is therefore up to `output_len * timeout` in
/// the worst case.
///
/// # Soundness
///
/// `true` is a sound proof that the lower bound holds for the whole input box;
/// `Verified` has no false positives (see the module docs). A single
/// `Unknown`/`Timeout`/violation on any index yields `false` — callers must not
/// relax `threshold` to force a pass.
#[must_use]
pub fn beta_crown_proves_lower_bound(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_len: usize,
    threshold: f32,
    timeout: Duration,
) -> bool {
    let verifier = BetaCrownVerifier::new(genbab_config(timeout));
    (0..output_len).all(|i| {
        let mut objective = vec![0.0_f32; output_len];
        objective[i] = 1.0;
        verify_one(&verifier, graph, input, &objective, threshold)
    })
}

/// Soundly prove `output_i ≤ threshold` for **every** output index
/// `i ∈ [0, output_len)` via beta-CROWN GenBaB branch-and-bound.
///
/// Implemented by proving `-output_i ≥ -threshold`: the objective is `-e_i`
/// (−1.0 at `i`, 0.0 elsewhere) with negated threshold, since beta-CROWN only
/// proves lower bounds on `objective · output`. Returns `true` **only if every**
/// index verifies.
///
/// `timeout` is the per-index wall-clock budget. See
/// [`beta_crown_proves_lower_bound`] for the soundness contract (identical).
#[must_use]
pub fn beta_crown_proves_upper_bound(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_len: usize,
    threshold: f32,
    timeout: Duration,
) -> bool {
    let verifier = BetaCrownVerifier::new(genbab_config(timeout));
    (0..output_len).all(|i| {
        let mut objective = vec![0.0_f32; output_len];
        objective[i] = -1.0;
        verify_one(&verifier, graph, input, &objective, -threshold)
    })
}

/// Soundly prove that every output coordinate lies within `[lower, upper]`:
/// `lower ≤ output_i ≤ upper` for all `i`.
///
/// Combines [`beta_crown_proves_lower_bound`] (with `lower`) and
/// [`beta_crown_proves_upper_bound`] (with `upper`); `true` only if both hold
/// for every index. `timeout` is the per-index, per-direction budget.
///
/// Same soundness contract as the directional helpers: `true` is a sound proof
/// of the box property; any unproven index/direction yields `false`.
#[must_use]
pub fn beta_crown_proves_width(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_len: usize,
    lower: f32,
    upper: f32,
    timeout: Duration,
) -> bool {
    beta_crown_proves_lower_bound(graph, input, output_len, lower, timeout)
        && beta_crown_proves_upper_bound(graph, input, output_len, upper, timeout)
}
