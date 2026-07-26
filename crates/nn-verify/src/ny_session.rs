// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Load-once / verify-many sessions via `ny_api::session`.
//!
//! NN's sweep and batch workloads (parallel position sweeps, epsilon searches,
//! per-layer property checks) re-verify the *same* graph against many specs. Each
//! standalone `verify` call re-walks and re-prepares the network. A
//! [`VerifierSession`] owns the [`GraphNetwork`] once and answers many specs
//! against it, with an opt-in verdict cache.
//!
//! # Cache soundness
//!
//! The cache key is the network fingerprint *plus the full spec*, so a hit can
//! only ever return the verdict that this exact network and this exact property
//! already produced. It cannot leak a verdict across differing inputs, and it is
//! opt-in ([`VerifierSession::set_caching_enabled`]). Disable it whenever a
//! [`ny_core::GemmEngine`] with nondeterministic accumulation is in play.
//!
//! ```rust,no_run
//! use nn_verify::ny_session::session;
//! # fn demo(net: ny_propagate::GraphNetwork, specs: &[ny_core::VerificationSpec]) {
//! let mut s = session(net);
//! let verdicts = s.verify_many(specs);
//! eprintln!("{} specs, {} cache entries", verdicts.len(), s.cache_len());
//! # }
//! ```

pub use ny_api::session::{SessionStats, VerifierSession};

/// Open a verification session over `net` with ny's default propagation config.
///
/// Caching follows [`VerifierSession`]'s default; flip it with
/// [`VerifierSession::set_caching_enabled`].
#[must_use]
pub fn session(net: ny_propagate::GraphNetwork) -> VerifierSession {
    VerifierSession::new(net)
}

/// Open a verification session over `net` with an explicit propagation config.
#[must_use]
pub fn session_with_config(
    net: ny_propagate::GraphNetwork,
    config: ny_api::verify::PropagationConfig,
) -> VerifierSession {
    VerifierSession::with_config(net, config)
}
