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
//! This adapter is additive — it does not touch NN's existing verify path. The
//! `*_with_config` variants and [`GenBabConfigBuilder`] only widen the config
//! surface; the original helpers keep their exact signatures and behavior.
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
//! a pass. A config rejected by ny's fail-closed `validate()` at the engine
//! ingress likewise surfaces as `false`, never as a pass.
//!
//! Memory is bounded soundly on both BaB lanes. The graph lanes these helpers
//! call cap their domain frontier with the typed
//! [`BetaCrownConfig::max_queue_size`] (domain count, default 500 000) and
//! [`BetaCrownConfig::max_queue_bytes`] (estimated payload bytes, `0` =
//! uncapped default) fields: on overflow the lowest-priority *unexplored*
//! domains are discarded and the run is forced to `Unknown` — eviction can only
//! *lose* a `Verified`, never mint one. ny's sequential-network lane instead
//! reads the `NY_BAB_QUEUE_MEM_MB` env lever (see the levers section below).
//!
//! # Engine behaviors at tip (ny `d3f5785fa`, 2026-08-21)
//!
//! Since NN's previous integration (2026-07-18), ny's sequential BaB lane
//! gained three behaviors that fire **without any config knob**:
//!
//! - **alpha/beta warm-start** — a child domain inherits its parent's optimized
//!   β for every inherited constraint (only the new split constraint starts at
//!   0), and, under `use_alpha_crown`, the parent's optimized α trajectory; the
//!   root itself is seeded with one joint-α optimization. Cold-starting both at
//!   every split (the historical behavior) kept every domain's bound at root
//!   quality regardless of depth.
//! - **root-tight boxes** — the BaB root upgrades raw forward-IBP per-layer
//!   boxes to collection-grade CROWN-IBP whenever the node budget allows,
//!   regardless of `use_crown_ibp`. The root's per-layer boxes become every
//!   domain's ReLU-relaxation triangles for the whole tree, so this is the
//!   difference between a closable and an unclosable search on deep FC nets.
//! - **monotone bound floor** — a child's lower (proof-side) bound is clamped
//!   up to its parent's: the child's feasible region is a subset of the
//!   parent's, so any sound parent lower bound is a sound child lower bound.
//!   The clamp is skipped when it would invert against the child's upper bound,
//!   leaving violation detection untouched.
//!
//! All three are warm-starts or shrink-only tightenings: none can produce a
//! spurious `Verified`. The graph relu-split lane these helpers call already
//! carried the equivalent parent-lower inheritance
//! (`inherit_parent_lower_only`).
//!
//! # Budget semantics (ny `dc068d74e`)
//!
//! [`BetaCrownConfig::timeout`] is both the hard budget for one search *and*
//! the effort horizon every phase sizes itself against (a zero duration expires
//! immediately). Historically BaB received only the *residue* of the root
//! window: each root phase (alpha bootstrap, interm sweeps, root objective
//! pass) claimed `min(fixed_cap, k × remaining)` of the same window, and on
//! hard instances the residue was zero — the BaB loop never started. At tip,
//! `split_root_window` can carve named shares out of ONE window with BaB's
//! reserve subtracted FIRST, reached through the typed fields
//! [`BetaCrownConfig::root_bab_reserve_frac`] (the arming gate),
//! [`BetaCrownConfig::root_spec_frac`], and
//! [`BetaCrownConfig::root_alpha_frac`]. All three are `Option<f64>`; `None`
//! (the default) means the arbitration does not exist and scheduling is
//! byte-identical to the legacy ladder. The `NY_BAB_RESERVE_FRAC`,
//! `NY_ROOT_SPEC_FRAC`, and `NY_ROOT_ALPHA_FRAC` env levers override the typed
//! values in both directions. Phase-level fractions (warmup cap, PGD
//! reservations, MIP clamps) are centralized in [`PhaseBudgetConfig`]
//! ([`BetaCrownConfig::phase_budget`]). Deadlines only schedule work — every
//! path falls back to sound reference bounds on expiry — so no budget setting
//! carries a soundness obligation.
//!
//! # In-process environment levers
//!
//! These fire for embedded API consumers (no CLI involved). Semantics below
//! are sourced from the current reader sites in `ny-propagate`, not docs:
//!
//! - `NY_BAB_QUEUE_MEM_MB` — memory budget (MB) for the *sequential-network*
//!   BaB domain queue (`beta_crown/engine/core.rs`); default 3072. Unset,
//!   unparsable, and `0` all fall back to 3072 — this lever has no "unlimited"
//!   spelling. Reaching the cap returns a sound `Unknown`. (The graph lanes
//!   used here are governed by the typed `max_queue_size` / `max_queue_bytes`
//!   fields instead.)
//! - `NY_F64_TAIL` — certified f64 tail re-bounding of knife-edge survivors on
//!   the grouped-disjunctive input-split lane: `1` force-arms, any other
//!   *present* value force-disarms, absent defers to the typed
//!   [`BetaCrownConfig::input_split_f64_tail_enabled`] (default dark). The
//!   value is read once per process and cached. Guard band via `NY_F64_TAIL_BAND`
//!   over [`BetaCrownConfig::input_split_f64_tail_band`] (legacy default 5e-3).
//! - `NY_SEQ_WEIGHTED_BUDGET` — **inverted opt-OUT polarity**: the reader
//!   tests `!= "0"`, so absence and every token other than the exact `0` keep
//!   cost-weighted per-candidate grants ENGAGED for the sequential CROWN-IBP
//!   collection budget (weight = objective rows × walk length). Exactly `0`
//!   restores the historical count-based equal shares byte-for-byte.
//!
//! # API
//!
//! - [`beta_crown_proves_lower_bound`] — proves `output_i ≥ threshold` for every
//!   output index `i`.
//! - [`beta_crown_proves_upper_bound`] — proves `output_i ≤ threshold` for every
//!   output index `i`.
//! - [`beta_crown_proves_width`] — proves every output coordinate lies within
//!   `[lower, upper]` (lower-bound + upper-bound combined).
//! - [`beta_crown_proves_lower_bound_with_config`] /
//!   [`beta_crown_proves_upper_bound_with_config`] /
//!   [`beta_crown_proves_width_with_config`] — the same proofs under a
//!   caller-supplied [`BetaCrownConfig`].
//! - [`default_genbab_config`] — the shared GenBaB base config the plain
//!   helpers use.
//! - [`GenBabConfigBuilder`] — builder-style access to the post-2026-07 config
//!   surface (queue byte caps, kFSB/conv/clip knobs, the f64-tail sub-config,
//!   root budget shares, phase budgets).
//!
//! ```rust,no_run
//! use std::time::Duration;
//! use nn_verify::beta_crown_verify::{
//!     beta_crown_proves_lower_bound_with_config, GenBabConfigBuilder,
//! };
//! # fn run(graph: &ny_propagate::GraphNetwork, input: &nn_verify::BoundedTensor) -> bool {
//! let cfg = GenBabConfigBuilder::new(Duration::from_secs(30))
//!     .max_queue_bytes(512 * 1024 * 1024)
//!     .f64_tail(true)
//!     .build();
//! beta_crown_proves_lower_bound_with_config(graph, input, 2, -1.0, &cfg)
//! # }
//! ```

use std::time::Duration;

use ny_propagate::beta_crown::NonlinearBranchingConfig;
use ny_propagate::{
    BabVerificationStatus, BetaCrownConfig, BetaCrownVerifier, BranchingHeuristic, ConvMode,
    GraphNetwork, InputClipType, KfsbReduceOp, PhaseBudgetConfig,
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

/// Build the shared GenBaB beta-CROWN config used by the plain helpers here.
///
/// - `GenBaB` heuristic: split the nonlinear/bilinear nodes that IBP
///   decorrelates (this is the whole point — recover the correlation IBP drops).
/// - `use_alpha_crown: false`: skip the per-domain alpha-CROWN optimization.
///   With the GenBaB input-index fix the BaB descends and splits the MulBinary
///   directly; the alpha bootstrap's Graph IBP is too slow on these deep graphs
///   to fit the wall-clock budget, so a fast plain-CROWN seed + more BaB
///   domains wins.
/// - `timeout`: hard wall-clock budget per search; on expiry the verdict is
///   `Timeout`.
///
/// Every other field keeps ny's [`BetaCrownConfig::default`], including the new
/// post-2026-07 surface: uncapped `max_queue_bytes`, dark f64 tail, no root
/// budget arbitration. Reach those through [`GenBabConfigBuilder`] or by
/// mutating the returned config directly.
#[must_use]
pub fn default_genbab_config(timeout: Duration) -> BetaCrownConfig {
    BetaCrownConfig {
        branching_heuristic: BranchingHeuristic::GenBaB(NonlinearBranchingConfig {
            num_candidates: GENBAB_NUM_CANDIDATES,
            ..Default::default()
        }),
        use_alpha_crown: false,
        max_domains: DEFAULT_MAX_DOMAINS,
        timeout,
        ..Default::default()
    }
}

/// Builder-style access to the beta-CROWN config surface NN callers need.
///
/// Starts from [`default_genbab_config`] and lets call sites reach the fields
/// ny grew since NN's 2026-07-18 integration — queue byte caps, kFSB/conv/clip
/// knobs, the input-split f64-tail sub-config, root budget shares, and phase
/// budgets — without spelling out an exhaustive [`BetaCrownConfig`] literal
/// (ny documents that struct as a growing pre-1.0 surface; exhaustive literals
/// break on every ny bump).
///
/// [`Self::build`] returns the plain [`BetaCrownConfig`]; fields without a
/// dedicated setter can be adjusted on the returned value before handing it to
/// a `*_with_config` helper.
///
/// ```rust,no_run
/// use std::time::Duration;
/// use nn_verify::beta_crown_verify::GenBabConfigBuilder;
///
/// let cfg = GenBabConfigBuilder::new(Duration::from_secs(60))
///     .max_domains(10_000)
///     .max_queue_bytes(256 * 1024 * 1024)
///     .root_bab_reserve_frac(0.25)
///     .build();
/// assert_eq!(cfg.max_queue_bytes, 256 * 1024 * 1024);
/// ```
#[derive(Debug, Clone)]
pub struct GenBabConfigBuilder {
    config: BetaCrownConfig,
}

impl GenBabConfigBuilder {
    /// Start from [`default_genbab_config`] with the given per-search timeout.
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            config: default_genbab_config(timeout),
        }
    }

    /// Hard wall-clock budget for one search; on expiry the verdict is
    /// `Timeout`. A zero duration expires immediately (ny treats zero as an
    /// instant deadline, not as "unbounded").
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Seed each domain with alpha-CROWN optimization (default `false` here —
    /// see [`default_genbab_config`] for why the plain-CROWN seed wins on NN's
    /// deep multiplicative graphs).
    #[must_use]
    pub fn use_alpha_crown(mut self, enabled: bool) -> Self {
        self.config.use_alpha_crown = enabled;
        self
    }

    /// Ceiling on explored domains before the search gives up (`Unknown`).
    /// Sound: hitting the cap can only fail to prove.
    #[must_use]
    pub fn max_domains(mut self, max_domains: usize) -> Self {
        self.config.max_domains = max_domains;
        self
    }

    /// Cap on simultaneously *stored* domains (count) before lowest-priority
    /// eviction. `0` disables the cap. Eviction forces `Unknown`, never a
    /// spurious `Verified`.
    #[must_use]
    pub fn max_queue_size(mut self, max_queue_size: usize) -> Self {
        self.config.max_queue_size = max_queue_size;
        self
    }

    /// Estimated payload cap in **bytes** for the graph BaB frontier
    /// (new since 2026-07; `#ml4acopf-bab-queue-mem`). `0` (the default)
    /// disables the cap. This is a logical-payload estimate, not a process-RSS
    /// limit; evicted domains are unexplored, so overflow forces `Unknown` —
    /// the bound stays sound.
    #[must_use]
    pub fn max_queue_bytes(mut self, max_queue_bytes: usize) -> Self {
        self.config.max_queue_bytes = max_queue_bytes;
        self
    }

    /// Replace the branching heuristic outright (e.g. [`BranchingHeuristic::Kfsb`]).
    ///
    /// The builder default is GenBaB with [`GenBabConfigBuilder::genbab_candidates`]
    /// candidates; use that setter instead to keep GenBaB and only tune the
    /// candidate count.
    #[must_use]
    pub fn branching_heuristic(mut self, heuristic: BranchingHeuristic) -> Self {
        self.config.branching_heuristic = heuristic;
        self
    }

    /// Number of GenBaB branching candidates evaluated per split (default 4).
    ///
    /// Keeps the current GenBaB sub-config when the heuristic is already
    /// GenBaB; otherwise reinstates GenBaB with ny's defaults plus this count.
    #[must_use]
    pub fn genbab_candidates(mut self, num_candidates: usize) -> Self {
        if let BranchingHeuristic::GenBaB(nl) = &mut self.config.branching_heuristic {
            nl.num_candidates = num_candidates;
        } else {
            self.config.branching_heuristic =
                BranchingHeuristic::GenBaB(NonlinearBranchingConfig {
                    num_candidates,
                    ..Default::default()
                });
        }
        self
    }

    /// Number of candidate neurons evaluated for FSB/kFSB branching (ny
    /// default 8). Only consulted by the FSB/kFSB heuristics.
    #[must_use]
    pub fn fsb_candidates(mut self, fsb_candidates: usize) -> Self {
        self.config.fsb_candidates = fsb_candidates;
        self
    }

    /// Reduce operation combining kFSB branch scores: `Min` (conservative,
    /// default), `Max` (optimistic), or `Mean` (balanced).
    #[must_use]
    pub fn kfsb_reduce_op(mut self, op: KfsbReduceOp) -> Self {
        self.config.kfsb_reduce_op = op;
        self
    }

    /// Convolution backward mode for CROWN propagation: `Auto` (default),
    /// `Patches`, or `Matrix`.
    #[must_use]
    pub fn conv_mode(mut self, mode: ConvMode) -> Self {
        self.config.conv_mode = mode;
        self
    }

    /// Input-domain clipping algorithm: `Relaxed` (closed-form 1D, default) or
    /// `Complete` (LP-optimal Lagrangian dual). Only observed when relaxed
    /// clipping is enabled ([`Self::enable_relaxed_clip`]).
    #[must_use]
    pub fn input_clip_type(mut self, clip_type: InputClipType) -> Self {
        self.config.input_clip_type = clip_type;
        self
    }

    /// Enable relaxed clipping (Clip-and-Verify): tighten child input bounds
    /// with CROWN linear constraints after each input split. Requires the
    /// `InputSplit` branching heuristic to have effect.
    #[must_use]
    pub fn enable_relaxed_clip(mut self, enabled: bool) -> Self {
        self.config.enable_relaxed_clip = enabled;
        self
    }

    /// Enable intermediate-domain clipping: use split-derived linear
    /// constraints to tighten intermediate (pre-activation) bounds. Requires
    /// ReLU branching (not input split).
    #[must_use]
    pub fn clip_interm_domain(mut self, enabled: bool) -> Self {
        self.config.enable_clip_interm_domain = enabled;
        self
    }

    /// Number of objective neurons per layer tightened by intermediate-domain
    /// clipping (ny default 3).
    #[must_use]
    pub fn clip_interm_topk(mut self, topk: usize) -> Self {
        self.config.clip_interm_topk = topk;
        self
    }

    /// Arm (`true`) or explicitly disarm (`false`) the certified f64 tail
    /// re-bounding of knife-edge survivors on the grouped-disjunctive
    /// input-split lane (`#cgan-f64-tail`). The builder default leaves the
    /// typed field `None` (dark), deferring to the `NY_F64_TAIL` env lever.
    /// Additive parity class: only certified `Verified` outcomes mutate
    /// anything, and only by monotonically raising certified lower bounds.
    #[must_use]
    pub fn f64_tail(mut self, enabled: bool) -> Self {
        self.config.input_split_f64_tail_enabled = Some(enabled);
        self
    }

    /// Escalation guard band for the f64 tail: a still-unverified domain is
    /// eligible iff its grouped f32 gap is `>= -band`. Unset keeps the legacy
    /// 5e-3 default; `NY_F64_TAIL_BAND` overrides when set and valid.
    #[must_use]
    pub fn f64_tail_band(mut self, band: f32) -> Self {
        self.config.input_split_f64_tail_band = Some(band);
        self
    }

    /// Hard per-run escalation cap shared by both f64-tail seams (`0` =
    /// unlimited, the legacy behavior). Config-only — this knob deliberately
    /// has no env override.
    #[must_use]
    pub fn f64_tail_max_escalations(mut self, max_escalations: usize) -> Self {
        self.config.input_split_f64_tail_max_escalations = Some(max_escalations);
        self
    }

    /// Hard wall-clock cap (seconds) on the root alpha-CROWN warmup. Unset
    /// leaves the warmup bounded only by the phase budget (the historical
    /// behavior). Scheduling only — no soundness obligation;
    /// `NY_ROOT_ALPHA_CAP_SECS` still overrides.
    #[must_use]
    pub fn root_alpha_cap_secs(mut self, secs: f64) -> Self {
        self.config.root_alpha_cap_secs = Some(secs);
        self
    }

    /// `#bab-floor` arming gate (ny `dc068d74e`): BaB's guaranteed share of
    /// the multi-objective root window, subtracted BEFORE any root phase sizes
    /// itself. Unset (the default) means no arbitration exists and every root
    /// phase keeps its legacy deadline. `NY_BAB_RESERVE_FRAC` overrides in
    /// both directions.
    #[must_use]
    pub fn root_bab_reserve_frac(mut self, frac: f64) -> Self {
        self.config.root_bab_reserve_frac = Some(frac);
        self
    }

    /// `#bab-floor`: the root objective pass's share of the root window, read
    /// only when [`Self::root_bab_reserve_frac`] armed the arbitration.
    /// `NY_ROOT_SPEC_FRAC` overrides.
    #[must_use]
    pub fn root_spec_frac(mut self, frac: f64) -> Self {
        self.config.root_spec_frac = Some(frac);
        self
    }

    /// `#bab-floor`: the bootstrap ascent's share, min-composed onto
    /// [`Self::root_alpha_cap_secs`]. `NY_ROOT_ALPHA_FRAC` overrides.
    #[must_use]
    pub fn root_alpha_frac(mut self, frac: f64) -> Self {
        self.config.root_alpha_frac = Some(frac);
        self
    }

    /// Phase-level time budget fractions (warmup cap, PGD reservations, MIP
    /// clamps) — ny's `#2206` centralization of the previously scattered
    /// timeout policy. The default reproduces ny's shipped behavior.
    #[must_use]
    pub fn phase_budget(mut self, phase_budget: PhaseBudgetConfig) -> Self {
        self.config.phase_budget = phase_budget;
        self
    }

    /// Finish, yielding the [`BetaCrownConfig`] to pass to the
    /// `*_with_config` helpers (or to tweak further for fields without a
    /// dedicated setter).
    #[must_use]
    pub fn build(self) -> BetaCrownConfig {
        self.config
    }
}

/// Run beta-CROWN once for a single objective and return whether it produced a
/// **sound** `Verified` verdict for `objective · output ≥ threshold`.
///
/// `Verified` ⇒ the property holds for every input in `input`'s box (sound).
/// Every other verdict ⇒ `false` (not proven within budget).
///
/// On a verifier error (malformed graph/objective, config rejected by ny's
/// fail-closed ingress validation), returns `false` — an error is "not
/// proven", never a spurious pass.
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
/// the worst case. Uses [`default_genbab_config`]; for the extended config
/// surface use [`beta_crown_proves_lower_bound_with_config`].
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
    beta_crown_proves_lower_bound_with_config(
        graph,
        input,
        output_len,
        threshold,
        &default_genbab_config(timeout),
    )
}

/// [`beta_crown_proves_lower_bound`] under a caller-supplied
/// [`BetaCrownConfig`] (build one with [`GenBabConfigBuilder`] or
/// [`default_genbab_config`]).
///
/// The per-index wall-clock budget is `config.timeout`; every field of
/// `config` — including the post-2026-07 surface (queue byte caps, f64 tail,
/// root budget shares, phase budgets) — reaches ny's engine unmodified.
///
/// Same soundness contract as [`beta_crown_proves_lower_bound`]: `true` is a
/// sound proof; any unproven index (or a config ny's ingress validation
/// rejects) yields `false`.
#[must_use]
pub fn beta_crown_proves_lower_bound_with_config(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_len: usize,
    threshold: f32,
    config: &BetaCrownConfig,
) -> bool {
    let verifier = BetaCrownVerifier::new(config.clone());
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
/// `timeout` is the per-index wall-clock budget. Uses
/// [`default_genbab_config`]; for the extended config surface use
/// [`beta_crown_proves_upper_bound_with_config`]. See
/// [`beta_crown_proves_lower_bound`] for the soundness contract (identical).
#[must_use]
pub fn beta_crown_proves_upper_bound(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_len: usize,
    threshold: f32,
    timeout: Duration,
) -> bool {
    beta_crown_proves_upper_bound_with_config(
        graph,
        input,
        output_len,
        threshold,
        &default_genbab_config(timeout),
    )
}

/// [`beta_crown_proves_upper_bound`] under a caller-supplied
/// [`BetaCrownConfig`].
///
/// The per-index wall-clock budget is `config.timeout`. Same soundness
/// contract as the directional helpers: `true` is a sound proof; any unproven
/// index yields `false`.
#[must_use]
pub fn beta_crown_proves_upper_bound_with_config(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_len: usize,
    threshold: f32,
    config: &BetaCrownConfig,
) -> bool {
    let verifier = BetaCrownVerifier::new(config.clone());
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

/// [`beta_crown_proves_width`] under a caller-supplied [`BetaCrownConfig`].
///
/// The per-index, per-direction budget is `config.timeout`. Same soundness
/// contract as [`beta_crown_proves_width`].
#[must_use]
pub fn beta_crown_proves_width_with_config(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    output_len: usize,
    lower: f32,
    upper: f32,
    config: &BetaCrownConfig,
) -> bool {
    beta_crown_proves_lower_bound_with_config(graph, input, output_len, lower, config)
        && beta_crown_proves_upper_bound_with_config(graph, input, output_len, upper, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ny_propagate::layers::{Layer, LinearLayer, ReLULayer};
    use ny_propagate::GraphNode;

    /// Tiny Linear → ReLU → Linear graph (2→2→2) for end-to-end BaB exercise.
    ///
    /// Mirrors the construction NN uses elsewhere (e.g. `ny_ladder` tests),
    /// keeping the network small and deterministic. On the unit box
    /// `[-1, 1]^2` both outputs equal `|x1 - x2| ∈ [0, 2]`; plain IBP already
    /// brackets them in `[0, 4]`.
    fn linear_relu_linear_graph() -> GraphNetwork {
        let w1 =
            ndarray::Array2::from_shape_vec((2, 2), vec![1.0, -1.0, -1.0, 1.0]).expect("2x2 w1");
        let linear1 = LinearLayer::new(w1, Some(ndarray::Array1::zeros(2))).expect("valid linear1");
        let w2 = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0]).expect("2x2 w2");
        let linear2 = LinearLayer::new(w2, Some(ndarray::Array1::zeros(2))).expect("valid linear2");

        let mut graph = GraphNetwork::new();
        graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
        graph.add_node(GraphNode::new(
            "relu",
            Layer::ReLU(ReLULayer::new()),
            vec!["linear1".to_string()],
        ));
        graph.add_node(GraphNode::new(
            "linear2",
            Layer::Linear(linear2),
            vec!["relu".to_string()],
        ));
        graph.set_output("linear2");
        graph
    }

    /// The unit input box `[-1, 1]^2`.
    fn unit_box_input() -> BoundedTensor {
        BoundedTensor::new(
            ndarray::arr1(&[-1.0_f32, -1.0]).into_dyn(),
            ndarray::arr1(&[1.0_f32, 1.0]).into_dyn(),
        )
        .expect("valid input box")
    }

    #[test]
    fn default_genbab_config_pins_the_shared_shape() {
        let cfg = default_genbab_config(Duration::from_secs(30));
        assert_eq!(cfg.timeout, Duration::from_secs(30));
        assert!(
            !cfg.use_alpha_crown,
            "the shared config seeds with plain CROWN (see module docs)"
        );
        assert_eq!(cfg.max_domains, DEFAULT_MAX_DOMAINS);
        match &cfg.branching_heuristic {
            BranchingHeuristic::GenBaB(nl) => {
                assert_eq!(nl.num_candidates, GENBAB_NUM_CANDIDATES);
            }
            other => panic!("expected GenBaB heuristic, got {other:?}"),
        }
        // The post-2026-07 surface keeps ny's defaults: uncapped byte budget,
        // dark f64 tail, no root budget arbitration.
        assert_eq!(cfg.max_queue_bytes, 0);
        assert_eq!(cfg.input_split_f64_tail_enabled, None);
        assert_eq!(cfg.root_bab_reserve_frac, None);
        assert_eq!(cfg.root_spec_frac, None);
        assert_eq!(cfg.root_alpha_frac, None);
    }

    #[test]
    fn builder_defaults_match_default_genbab_config() {
        let reference = default_genbab_config(Duration::from_secs(7));
        let built = GenBabConfigBuilder::new(Duration::from_secs(7)).build();
        assert_eq!(built.timeout, reference.timeout);
        assert_eq!(built.max_domains, reference.max_domains);
        assert_eq!(built.use_alpha_crown, reference.use_alpha_crown);
        assert_eq!(built.branching_heuristic, reference.branching_heuristic);
        assert_eq!(built.max_queue_size, reference.max_queue_size);
        assert_eq!(built.max_queue_bytes, reference.max_queue_bytes);
        assert_eq!(built.phase_budget, reference.phase_budget);
    }

    #[test]
    fn builder_setters_reach_the_ny_config() {
        let budget = PhaseBudgetConfig {
            initial_bounds_fraction: 0.25,
            ..PhaseBudgetConfig::default()
        };
        let cfg = GenBabConfigBuilder::new(Duration::from_secs(9))
            .max_domains(123)
            .max_queue_size(50_000)
            .max_queue_bytes(64 * 1024 * 1024)
            .genbab_candidates(2)
            .fsb_candidates(16)
            .kfsb_reduce_op(KfsbReduceOp::Mean)
            .conv_mode(ConvMode::Matrix)
            .input_clip_type(InputClipType::Complete)
            .enable_relaxed_clip(true)
            .clip_interm_domain(true)
            .clip_interm_topk(5)
            .f64_tail(true)
            .f64_tail_band(5e-4)
            .f64_tail_max_escalations(8)
            .root_alpha_cap_secs(40.0)
            .root_bab_reserve_frac(0.25)
            .root_spec_frac(0.10)
            .root_alpha_frac(0.20)
            .phase_budget(budget.clone())
            .build();

        assert_eq!(cfg.timeout, Duration::from_secs(9));
        assert_eq!(cfg.max_domains, 123);
        assert_eq!(cfg.max_queue_size, 50_000);
        assert_eq!(cfg.max_queue_bytes, 64 * 1024 * 1024);
        match &cfg.branching_heuristic {
            BranchingHeuristic::GenBaB(nl) => assert_eq!(nl.num_candidates, 2),
            other => panic!("expected GenBaB heuristic, got {other:?}"),
        }
        assert_eq!(cfg.fsb_candidates, 16);
        assert_eq!(cfg.kfsb_reduce_op, KfsbReduceOp::Mean);
        assert_eq!(cfg.conv_mode, ConvMode::Matrix);
        assert_eq!(cfg.input_clip_type, InputClipType::Complete);
        assert!(cfg.enable_relaxed_clip);
        assert!(cfg.enable_clip_interm_domain);
        assert_eq!(cfg.clip_interm_topk, 5);
        assert_eq!(cfg.input_split_f64_tail_enabled, Some(true));
        assert_eq!(cfg.input_split_f64_tail_band, Some(5e-4));
        assert_eq!(cfg.input_split_f64_tail_max_escalations, Some(8));
        assert_eq!(cfg.root_alpha_cap_secs, Some(40.0));
        assert_eq!(cfg.root_bab_reserve_frac, Some(0.25));
        assert_eq!(cfg.root_spec_frac, Some(0.10));
        assert_eq!(cfg.root_alpha_frac, Some(0.20));
        assert_eq!(cfg.phase_budget, budget);
    }

    #[test]
    fn builder_branching_heuristic_replacement_and_genbab_reinstatement() {
        let cfg = GenBabConfigBuilder::new(Duration::from_secs(1))
            .branching_heuristic(BranchingHeuristic::Kfsb)
            .build();
        assert_eq!(cfg.branching_heuristic, BranchingHeuristic::Kfsb);

        // genbab_candidates after a non-GenBaB heuristic reinstates GenBaB
        // with ny defaults plus the requested candidate count.
        let cfg = GenBabConfigBuilder::new(Duration::from_secs(1))
            .branching_heuristic(BranchingHeuristic::Kfsb)
            .genbab_candidates(3)
            .build();
        match &cfg.branching_heuristic {
            BranchingHeuristic::GenBaB(nl) => assert_eq!(nl.num_candidates, 3),
            other => panic!("expected GenBaB heuristic, got {other:?}"),
        }
    }

    #[test]
    fn beta_crown_proves_wide_box_on_linear_relu_linear() {
        let graph = linear_relu_linear_graph();
        let input = unit_box_input();
        let timeout = Duration::from_secs(10);

        // output_i = |x1 - x2| ∈ [0, 2]; IBP already brackets it in [0, 4], so
        // these wide thresholds must be soundly provable.
        assert!(
            beta_crown_proves_lower_bound(&graph, &input, 2, -1.0, timeout),
            "output_i >= -1.0 must verify (true range [0, 2])"
        );
        assert!(
            beta_crown_proves_upper_bound(&graph, &input, 2, 5.0, timeout),
            "output_i <= 5.0 must verify (true range [0, 2])"
        );
        assert!(
            beta_crown_proves_width(&graph, &input, 2, -1.0, 5.0, timeout),
            "the combined box property must verify"
        );
    }

    #[test]
    fn beta_crown_rejects_unprovable_threshold() {
        let graph = linear_relu_linear_graph();
        let input = unit_box_input();

        // The true minimum of output_i is 0 (attained at x1 == x2), so
        // `output_i >= 1.0` is FALSE — a sound engine must never verify it,
        // whatever the search returns (Unknown / PotentialViolation / Timeout).
        assert!(
            !beta_crown_proves_lower_bound(&graph, &input, 2, 1.0, Duration::from_secs(5)),
            "a false property must never be proven"
        );
    }

    #[test]
    fn with_config_carries_the_caller_config_into_the_engine() {
        let graph = linear_relu_linear_graph();
        let input = unit_box_input();
        let cfg = GenBabConfigBuilder::new(Duration::from_secs(10))
            .max_domains(64)
            .max_queue_bytes(64 * 1024 * 1024)
            .build();

        // The builder-made config drives a real sound verdict end-to-end.
        assert!(
            beta_crown_proves_lower_bound_with_config(&graph, &input, 2, -1.0, &cfg),
            "wide lower bound must verify under the builder config"
        );
        assert!(
            beta_crown_proves_width_with_config(&graph, &input, 2, -1.0, 5.0, &cfg),
            "the combined box property must verify under the builder config"
        );

        // Plumbing proof by refusal: ny validates the caller's config at the
        // engine ingress (`verify_graph_relu_split` → `config.validate()`),
        // and cut proof authority is quarantined at tip — `enable_cuts` fails
        // validation. The SAME call that verified above must come back false
        // when the config carries the quarantined flag, demonstrating that the
        // wrapper hands this exact config to ny and that engine errors surface
        // as "not proven", never as a pass.
        let mut quarantined = default_genbab_config(Duration::from_secs(10));
        quarantined.enable_cuts = true;
        assert!(
            !beta_crown_proves_lower_bound_with_config(&graph, &input, 2, -1.0, &quarantined),
            "a config rejected by ny's fail-closed validation must yield false"
        );
    }
}
