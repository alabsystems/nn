// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Parallel position verification for sequence-model graphs.
//!
//! Wraps NY's [`ParallelVerifier`] to verify different sequence
//! positions independently in parallel. Near-linear speedup with cores for
//! position-independent properties (e.g., output bounds at each timestep).
//!
//! # When to use
//!
//! - Verifying transformer/attention graphs where each sequence position has
//!   identical structure but independent bounds.
//! - Sequence length >= number of CPU cores (otherwise serial is faster).
//!
//! # Example
//!
//! ```rust,no_run
//! use nn_verify::parallel::{parallel_verify_positions, ParallelVerifyConfig};
//!
//! // Build graph from MHA kernel def + bindings, then:
//! // let result = parallel_verify_positions(&graph, &input, 0, None)?;
//! // assert!(result.output_bounds.lower().iter().all(|v| v.is_finite()));
//! ```
//!
//! Part of #813.

use std::sync::Arc;

use ny_core::GemmEngine;
use ny_propagate::{
    verify_parallel_with_method, GraphNetwork, ParallelConfig, ParallelVerificationResult,
    ParallelVerifier, PropagationMethod,
};

use crate::error::VerifyError;
use crate::verify_types::PropMethod;
use ny_api::BoundedTensor;

/// Backend selection for parallel CROWN verification.
///
/// Controls which GEMM engine is used for linear-layer bound propagation
/// during CROWN backward passes. CPU is the default (no acceleration).
/// `GpuEngine` threads a caller-provided [`GemmEngine`] through CROWN
/// so linear-layer matmuls can run on GPU.
///
/// # Example
///
/// ```rust,no_run
/// use nn_verify::parallel::{ParallelVerifyBackend, ParallelVerifyConfig};
///
/// // CPU-default (same as today):
/// let config = ParallelVerifyConfig::crown();
///
/// // With a caller-provided engine:
/// // let engine: Arc<dyn ny_core::GemmEngine> = ...;
/// // let config = ParallelVerifyConfig::crown().with_backend(ParallelVerifyBackend::GpuEngine(engine));
/// ```
///
/// Part of #2193.
#[derive(Clone, Default)]
pub enum ParallelVerifyBackend {
    /// CPU-default: no GEMM engine, same behaviour as before #2193.
    #[default]
    Cpu,
    /// Caller-provided GEMM engine for GPU-accelerated CROWN.
    GpuEngine(Arc<dyn GemmEngine>),
}

impl std::fmt::Debug for ParallelVerifyBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "Cpu"),
            Self::GpuEngine(_) => write!(f, "GpuEngine(...)"),
        }
    }
}

/// Configuration for parallel position verification.
///
/// Thin wrapper over NY's [`ParallelConfig`] with nn-specific
/// defaults and builder methods.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ParallelVerifyConfig {
    /// Propagation method (IBP or CROWN).
    pub method: PropMethod,
    /// Minimum positions before enabling parallelism (default: 4).
    pub min_positions: usize,
    /// Max threads. `None` = rayon default (num cores).
    pub max_threads: Option<usize>,
    /// GEMM backend for CROWN linear-layer propagation.
    pub backend: ParallelVerifyBackend,
}

impl Default for ParallelVerifyConfig {
    fn default() -> Self {
        Self {
            method: PropMethod::Ibp,
            min_positions: 4,
            max_threads: None,
            backend: ParallelVerifyBackend::Cpu,
        }
    }
}

impl ParallelVerifyConfig {
    /// Create a CROWN parallel config.
    #[must_use]
    pub fn crown() -> Self {
        Self {
            method: PropMethod::Crown,
            ..Default::default()
        }
    }

    /// Set max thread count.
    #[must_use]
    pub fn with_max_threads(mut self, n: usize) -> Self {
        self.max_threads = Some(n);
        self
    }

    /// Set the GEMM backend for CROWN propagation.
    #[must_use]
    pub fn with_backend(mut self, backend: ParallelVerifyBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Convert to NY's [`ParallelConfig`].
    fn to_gc_config(&self) -> ParallelConfig {
        ParallelConfig {
            method: to_gc_method(self.method),
            min_positions_for_parallel: self.min_positions,
            max_threads: self.max_threads,
            report_progress: false,
        }
    }

    /// Extract engine reference from backend, if any.
    fn engine(&self) -> Option<&dyn GemmEngine> {
        match &self.backend {
            ParallelVerifyBackend::Cpu => None,
            ParallelVerifyBackend::GpuEngine(e) => Some(e.as_ref()),
        }
    }
}

/// Convert nn `PropMethod` to NY `PropagationMethod`.
fn to_gc_method(method: PropMethod) -> PropagationMethod {
    match method {
        PropMethod::Ibp => PropagationMethod::Ibp,
        PropMethod::Crown => PropagationMethod::Crown,
        PropMethod::AlphaCrown => PropagationMethod::AlphaCrown,
        PropMethod::BetaCrown => PropagationMethod::BetaCrown,
        // Analytical bounds bypass NY; use IBP as a conservative fallback.
        PropMethod::Analytical => PropagationMethod::Ibp,
        // Mixed mode is orchestrated at a higher level; default to IBP for single-graph calls.
        PropMethod::MixedIbpCrown => PropagationMethod::Ibp,
    }
}

/// Verify sequence positions in parallel along `axis`.
///
/// Slices the input tensor along `axis`, verifies each slice independently
/// via NY's parallel verifier, and reassembles the output bounds.
///
/// When the config specifies a [`ParallelVerifyBackend::GpuEngine`], CROWN
/// propagation uses the engine for linear-layer GEMM acceleration.
/// When `Cpu` (the default), behaviour is identical to pre-#2193.
///
/// # Arguments
///
/// * `graph` — The NY `GraphNetwork` to verify.
/// * `input` — Input bounds (must have `axis` dimension >= 2).
/// * `axis` — Sequence axis to parallelize over (usually 0 for [T, D]).
/// * `config` — Optional config. `None` uses defaults (IBP, Cpu, auto threads).
///
/// # Errors
///
/// Returns [`VerifyError`] if propagation fails on any position.
pub fn parallel_verify_positions(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
    config: Option<&ParallelVerifyConfig>,
) -> Result<ParallelVerificationResult, VerifyError> {
    let default_config;
    let cfg = match config {
        Some(c) => c,
        None => {
            default_config = ParallelVerifyConfig::default();
            &default_config
        }
    };

    // When an engine is provided, NY's ParallelVerifier doesn't yet
    // accept it, so we delegate to our own engine-aware parallel loop instead.
    if cfg.engine().is_some() {
        return parallel_verify_with_engine(graph, input, axis, cfg);
    }

    let gc_config = cfg.to_gc_config();
    let verifier = ParallelVerifier::new(gc_config);
    Ok(verifier.verify_positions_parallel(graph, input, axis)?)
}

/// Engine-aware parallel verification loop.
///
/// Re-implements the position-slicing logic from NY's
/// `ParallelVerifier` but threads the `GemmEngine` into each CROWN call.
fn parallel_verify_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
    config: &ParallelVerifyConfig,
) -> Result<ParallelVerificationResult, VerifyError> {
    let start = std::time::Instant::now();
    let shape = input.shape();
    if axis >= shape.len() {
        return Err(ny_api::NyError::InvalidSpec(format!(
            "Axis {} out of bounds for tensor with {} dimensions",
            axis,
            shape.len()
        ))
        .into());
    }
    let num_positions = shape[axis];
    let engine = config.engine();

    let mut position_outputs = Vec::with_capacity(num_positions);
    for pos in 0..num_positions {
        let pos_input = input.slice_axis(axis, pos)?;
        let output = propagate_with_engine(graph, &pos_input, config.method, engine)?;
        position_outputs.push(output);
    }

    let output_bounds = BoundedTensor::stack(&position_outputs, axis)?;

    let total_time_ms = start.elapsed().as_millis() as u64;
    let avg_position_time_ms = if num_positions > 0 {
        total_time_ms as f64 / num_positions as f64
    } else {
        0.0
    };

    Ok(ParallelVerificationResult {
        output_bounds,
        num_positions,
        parallel_positions: 0, // serial for now — NY needs engine threading for rayon
        total_time_ms,
        avg_position_time_ms,
    })
}

/// Propagate a single position with an optional GEMM engine.
fn propagate_with_engine(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    method: PropMethod,
    engine: Option<&dyn GemmEngine>,
) -> Result<BoundedTensor, VerifyError> {
    Ok(match method {
        PropMethod::Ibp | PropMethod::Analytical | PropMethod::MixedIbpCrown => {
            graph.propagate_ibp(input)?
        }
        PropMethod::Crown => graph.propagate_crown_with_engine(input, engine)?,
        PropMethod::AlphaCrown | PropMethod::BetaCrown => {
            graph.propagate_alpha_crown_with_engine(input, engine)?
        }
    })
}

/// Verify positions in parallel with a specific propagation method.
///
/// Convenience wrapper for the common case of choosing IBP vs CROWN
/// without configuring other options.
pub fn parallel_verify_with_method(
    graph: &GraphNetwork,
    input: &BoundedTensor,
    axis: usize,
    method: PropMethod,
) -> Result<BoundedTensor, VerifyError> {
    Ok(verify_parallel_with_method(
        graph,
        input,
        axis,
        to_gc_method(method),
    )?)
}

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod tests;
