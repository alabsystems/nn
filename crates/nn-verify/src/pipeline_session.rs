// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Load-once / verify-many session wiring — the [`ny_session`](crate::ny_session)
//! adapter reached from the verification pipeline.
//!
//! NN's batch workloads verify **many specs against one network**: `verify_all`
//! runs several input regions per kernel graph, sweeps re-ask the same graph at
//! many positions, and per-layer property checks re-enter the same DAG. Each
//! standalone `Verifier::verify_graph` call re-hands the network to a fresh
//! verifier; a [`VerifierSession`] owns the graph once and answers every spec
//! against it, with an opt-in verdict cache on top.
//!
//! # What is wired
//!
//! [`verify_specs_in_session`] is the pipeline-level entry point: hand it a
//! translated `GraphNetwork` and a batch of specs and it runs the whole batch
//! through one session, returning the per-spec verdicts alongside
//! [`SessionStats`]. `verify_all`'s session tier calls exactly this.
//!
//! # What stays opt-in
//!
//! Nothing routes through a session implicitly. The existing pipeline entry
//! points ([`crate::pipeline::verify_and_record_full`] and friends) still build
//! and propagate per call, because they answer "what is the output enclosure",
//! not "does this spec hold" — the session's unit of work is a
//! [`VerificationSpec`], so it has nothing to answer for a bounds-only run.
//! Warm start ([`VerifierSession::set_warm_start_enabled`]) is likewise left
//! off; it changes which queries run propagation and is a separate caller
//! decision.
//!
//! # Cache soundness
//!
//! The cache key folds a structural fingerprint of the owned network together
//! with the *entire* spec (input region, output bounds, input shape, timeout,
//! every output constraint), and the session owns its network immutably for its
//! whole lifetime. A hit can therefore only ever replay the verdict this exact
//! network already produced for this exact property — never a stronger claim,
//! and never across differing input regions. [`verdict_fingerprint`] exists so
//! callers (and tests) can assert that byte-for-byte identity directly.

use ny_api::{VerificationResult, VerificationSpec};

use crate::error::VerifyError;

pub use crate::ny_session::{SessionStats, VerifierSession};
pub use ny_api::verify::{PropagationConfig, PropagationMethod};

/// Outcome of one batch run through a [`VerifierSession`].
#[derive(Debug)]
#[non_exhaustive]
pub struct SessionRun {
    /// One verdict per input spec, in the order the specs were given.
    ///
    /// An `Err` entry is a verifier error for that spec alone; the batch keeps
    /// going, matching [`VerifierSession::verify_many`].
    pub verdicts: Vec<Result<VerificationResult, VerifyError>>,
    /// Session counters for the batch (queries, verified, cache hits/misses).
    pub stats: SessionStats,
    /// Whether the verdict cache was consulted for this batch.
    pub caching_enabled: bool,
}

impl SessionRun {
    /// Number of specs the batch proved.
    #[must_use]
    pub fn verified_count(&self) -> usize {
        self.verdicts
            .iter()
            .filter(|v| v.as_ref().is_ok_and(VerificationResult::is_verified))
            .count()
    }

    /// Number of specs whose verification errored.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.verdicts.iter().filter(|v| v.is_err()).count()
    }

    /// Per-verdict fingerprints, in spec order. `None` where the spec errored.
    ///
    /// Two runs of the same batch are verdict-identical exactly when their
    /// fingerprint vectors are equal — which is how the cached and uncached
    /// paths are held to producing the same answers.
    #[must_use]
    pub fn fingerprints(&self) -> Vec<Option<String>> {
        self.verdicts
            .iter()
            .map(|v| v.as_ref().ok().map(verdict_fingerprint))
            .collect()
    }
}

/// A canonical, byte-comparable rendering of a verification verdict.
///
/// [`VerificationResult`] has no `PartialEq`, but it is `Serialize`, and its
/// serialization carries everything that distinguishes one verdict from
/// another: the variant, the output bounds, the soundness provenance, the
/// method tag, and any proof channel. Comparing these strings is therefore a
/// full-verdict comparison, not a variant-only one.
///
/// Falls back to the `Debug` rendering if serialization ever fails, so this is
/// total and never panics.
#[must_use]
pub fn verdict_fingerprint(result: &VerificationResult) -> String {
    serde_json::to_string(result).unwrap_or_else(|_| format!("{result:?}"))
}

/// Verify a batch of specs against one network through a single
/// [`VerifierSession`] — load once, verify many.
///
/// The network is moved into the session and prepared once; every spec is then
/// answered against that same loaded graph. With `caching` enabled, a spec
/// byte-identical to one already answered in this batch is served from the
/// verdict cache instead of re-propagated.
///
/// `config` selects the propagation method the session runs for every spec.
/// Keep it fixed across a batch: a session is one network *and* one propagation
/// configuration, and mixing methods within a batch would make the per-spec
/// verdicts incomparable.
///
/// # Soundness
///
/// A cached verdict is the *same* [`VerificationResult`] the verifier produced,
/// stored verbatim with its provenance — a hit can never upgrade an `Unknown`
/// to a `Verified`, and a differing input region hashes to a different key.
/// Pass `caching = false` whenever a nondeterministic
/// [`GemmEngine`](ny_core::GemmEngine) is in play, so every query re-propagates.
#[must_use = "the batch verdicts are the point of the call"]
pub fn verify_specs_in_session(
    net: ny_propagate::GraphNetwork,
    specs: &[VerificationSpec],
    config: PropagationConfig,
    caching: bool,
) -> SessionRun {
    let mut session = VerifierSession::with_config(net, config);
    session.set_caching_enabled(caching);

    let verdicts = session
        .verify_many(specs)
        .into_iter()
        .map(|r| r.map_err(VerifyError::from))
        .collect();

    SessionRun {
        verdicts,
        stats: session.stats().clone(),
        caching_enabled: caching,
    }
}

/// Open a session over `net` for callers that need to interleave specs with
/// other work rather than handing over a prepared batch.
///
/// Thin pass-through to [`crate::ny_session::session_with_config`], re-exposed
/// here so the pipeline is a single import surface for session use.
#[must_use]
pub fn open_session(net: ny_propagate::GraphNetwork, config: PropagationConfig) -> VerifierSession {
    crate::ny_session::session_with_config(net, config)
}
