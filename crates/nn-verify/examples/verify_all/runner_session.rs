// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Session tier — load-once / verify-many through `ny_session`.
//!
//! # Why this tier exists
//!
//! The kernel tier answers "what is this kernel's output enclosure": it
//! propagates bounds and records them. A [`VerifierSession`] answers a different
//! question — "does this *spec* hold" — so the kernel tier's existing work
//! cannot simply be re-pointed at a session. What the kernel tier *does* have is
//! the session's defining workload shape: **many input regions over the same
//! graph**. `gelu` / `gelu_wide`, `sigmoid` / `sigmoid_wide`, `relu` /
//! `relu_wide` and `tanh_act` / `tanh_act_wide` are each one kernel with one set
//! of constant params verified over several regions, i.e. one network and
//! several specs.
//!
//! This tier turns that shape into spec verification: it groups the kernel
//! configs by graph identity, builds each graph **once**, and asks one session
//! every member's spec.
//!
//! # What the specs assert
//!
//! Each group gets an *envelope*: the union of the output boxes the kernel tier
//! just proved for its members, padded by a small documented slack. Each
//! member's spec then asks "over my input region, does the output stay inside
//! the group's envelope?".
//!
//! This is a real property, not a tautology — it fails if a region's bounds
//! escape the envelope its siblings established, which is exactly the shape of
//! an unsound tightening or a translation regression. It is deliberately **not**
//! fatal: the tier is a diagnostic, it does not touch `verify_all`'s exit code,
//! and it only runs when `NN_VERIFY_SESSION=1` is set.
//!
//! The session runs plain IBP, the same method the kernel tier's non-escalated
//! runs used to compute the envelope, so the envelope and the check are on the
//! same rung and a group's verdicts are comparable to one another.

use std::collections::BTreeMap;

use nn_dsl::ir::KernelDef;
use nn_verify::pipeline::session::{PropagationConfig, PropagationMethod};
use nn_verify::pipeline::verify_specs_in_session;
use nn_verify::{kernel_to_graph, Bound, VerificationSpec};

/// Relative slack added to a group envelope before it becomes a required
/// output box.
///
/// Absorbs `f32` round-off between the enclosure the kernel tier recorded and
/// the one the session recomputes, without absorbing a real regression: a
/// genuine bound blow-up is orders of magnitude larger than 0.1%.
const ENVELOPE_RELATIVE_SLACK: f32 = 1e-3;

/// Absolute slack floor, for envelopes whose width is ~0.
const ENVELOPE_ABSOLUTE_SLACK: f32 = 1e-6;

/// One kernel-tier result, retained so the session tier can re-ask it as a spec.
pub(crate) struct SessionCase {
    pub(crate) config_name: &'static str,
    pub(crate) kernel: KernelDef,
    pub(crate) constant_params: Vec<f32>,
    pub(crate) input_lower: f32,
    pub(crate) input_upper: f32,
    pub(crate) output_lower: f32,
    pub(crate) output_upper: f32,
}

/// Whether the session tier is enabled. Opt-in: `NN_VERIFY_SESSION=1`.
pub(crate) fn session_tier_enabled() -> bool {
    std::env::var("NN_VERIFY_SESSION")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Group key: same kernel name **and** same constant params means the same
/// translated graph, since constants are baked in at translation time.
fn graph_key(case: &SessionCase) -> (String, Vec<u32>) {
    (
        case.kernel.name.clone(),
        case.constant_params.iter().map(|v| v.to_bits()).collect(),
    )
}

/// Run the session tier over the kernel-tier results.
///
/// Returns `(groups_run, specs_verified, specs_unproven)`. Nothing here feeds
/// the process exit code — see the module docs.
pub(crate) fn run_session_verification(cases: Vec<SessionCase>) -> (usize, usize, usize) {
    let mut groups: BTreeMap<(String, Vec<u32>), Vec<SessionCase>> = BTreeMap::new();
    for case in cases {
        groups.entry(graph_key(&case)).or_default().push(case);
    }

    // Only groups with more than one region exercise load-once / verify-many.
    let multi: Vec<_> = groups
        .into_iter()
        .filter(|(_, members)| members.len() > 1)
        .collect();
    if multi.is_empty() {
        return (0, 0, 0);
    }

    println!(
        "\n{:<30} {:>8} {:>10} {:>10} {:>12}",
        "Session group", "Specs", "Verified", "CacheHits", "Envelope"
    );
    println!("{}", "-".repeat(74));

    let mut groups_run = 0;
    let mut verified = 0;
    let mut unproven = 0;

    for ((kernel_name, _), members) in multi {
        // Envelope: union of the members' proven output boxes, plus slack.
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for m in &members {
            lo = lo.min(m.output_lower);
            hi = hi.max(m.output_upper);
        }
        if !lo.is_finite() || !hi.is_finite() {
            eprintln!("  {kernel_name}: skipped — a member produced non-finite bounds");
            continue;
        }
        let slack = (hi - lo).abs() * ENVELOPE_RELATIVE_SLACK + ENVELOPE_ABSOLUTE_SLACK;
        let (lo, hi) = (lo - slack, hi + slack);

        let head = &members[0];
        let graph = match kernel_to_graph(&head.kernel, &head.constant_params) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  {kernel_name}: skipped — graph translation failed: {e}");
                continue;
            }
        };

        let specs: Vec<VerificationSpec> = match members
            .iter()
            .map(|m| {
                VerificationSpec::new(
                    vec![Bound::new(m.input_lower, m.input_upper)],
                    vec![Bound::new(lo, hi)],
                )
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  {kernel_name}: skipped — spec construction failed: {e}");
                continue;
            }
        };

        // Load once, verify many. The verdict cache is on: it is sound (network
        // fingerprint + full spec key) and identical regions do recur.
        let run = verify_specs_in_session(
            graph,
            &specs,
            PropagationConfig {
                method: PropagationMethod::Ibp,
                ..PropagationConfig::default()
            },
            true,
        );

        let group_verified = run.verified_count();
        verified += group_verified;
        unproven += specs.len() - group_verified;
        groups_run += 1;

        println!(
            "{kernel_name:<30} {:>8} {:>10} {:>10} {:>12}",
            specs.len(),
            group_verified,
            run.stats.cache_hits,
            format!("[{lo:.3}, {hi:.3}]"),
        );

        for (member, verdict) in members.iter().zip(run.verdicts.iter()) {
            match verdict {
                Ok(result) if result.is_verified() => {}
                Ok(result) => eprintln!(
                    "  {}: outputs not proven inside the group envelope: {result:?}",
                    member.config_name
                ),
                Err(e) => eprintln!("  {}: session error: {e}", member.config_name),
            }
        }
    }

    (groups_run, verified, unproven)
}
