// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Shared helpers for Kokoro duration positivity parametric tests.
//!
//! Extracted from `compose_kokoro_duration_parametric.rs` for 500-line
//! compliance. Used by both `compose_kokoro_duration_parametric.rs`
//! (Groups A+B) and `compose_kokoro_duration_scaled_parametric.rs`
//! (Groups C+D+E).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::common::{assert_bounds_valid, uniform_bounds};
use nn_tts_verify::monotonicity::interpret_duration_positivity;
use nn_verify::tensor_kernel_to_graph;

use super::duration_scaled_helpers::KokoroDims;
use super::kokoro_prosody::{build_kokoro_prosody_single_block, build_kokoro_prosody_three_blocks};
use super::kokoro_prosody::{
    kokoro_prosody_bindings_with_weight_mag, kokoro_prosody_three_block_bindings_with_weight_mag,
    FLAT_INPUT_SIZE,
};
use super::prosody_scaled::{
    build_scaled_prosody, build_scaled_prosody_single_block, scaled_prosody_bindings,
    scaled_prosody_single_block_bindings, ProsodyDims,
};

// Thread-safe propagation caches. Tests run in parallel threads within the
// same binary, so many tests call run_scaled_proof with identical (dims, ib)
// pairs. Caching eliminates ~80+ redundant graph builds + propagations.
type ProofResult = (f64, &'static str, bool, bool);
type ProofKey = (usize, u32); // (d_model, ib.to_bits())

static SCALED_PROOF_CACHE: Mutex<Option<HashMap<ProofKey, ProofResult>>> = Mutex::new(None);
static SCALED_PROOF_1B_CACHE: Mutex<Option<HashMap<ProofKey, ProofResult>>> = Mutex::new(None);

type SensResult = (f64, &'static str, bool);
type SensKey = (u32, u32); // (wm.to_bits(), ib.to_bits())

static SENS_1B_CACHE: Mutex<Option<HashMap<SensKey, SensResult>>> = Mutex::new(None);
static SENS_3B_CACHE: Mutex<Option<HashMap<SensKey, SensResult>>> = Mutex::new(None);

/// Duration logits threshold: exp(-10) ~= 4.5e-5 > 0.
pub(super) const DURATION_THRESHOLD: f64 = -10.0;

pub(super) fn method_str(m: nn_verify::PropMethod) -> &'static str {
    match m {
        nn_verify::PropMethod::Crown => "CROWN",
        nn_verify::PropMethod::Ibp => "IBP",
        _ => "unknown",
    }
}

pub(super) fn lo_min_of(output: &nn_verify::BoundedTensor) -> f64 {
    f64::from(
        output
            .lower_upper()
            .0
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
    )
}

/// Core propagation: build graph → propagate → extract lo_min.
/// Returns `(lo_min, method_str, is_proven, is_finite)`.
pub(super) fn run_duration_proof(
    def: &nn_dsl::TensorKernelDef,
    bindings: &[nn_verify::TensorParamBinding],
    input_size: usize,
    input_bound: f32,
) -> (f64, &'static str, bool, bool) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("graph translation");
    let input = uniform_bounds(&[input_size], input_bound);
    let (method, output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("propagation");
    let lo_min = lo_min_of(&output);
    let method = method_str(method);
    let is_finite = lo_min.is_finite();
    (
        lo_min,
        method,
        is_finite && lo_min > DURATION_THRESHOLD,
        is_finite,
    )
}

/// Scaled three-block proof at given dimensions.
/// Results are cached by (d_model, ib) — eliminates ~80 redundant propagation
/// calls across parametric tests that share (dims, ib) pairs.
pub(super) fn run_scaled_proof(dims: &KokoroDims, ib: f32) -> (f64, &'static str, bool, bool) {
    let key = (dims.d_model, ib.to_bits());
    if let Some(&cached) = SCALED_PROOF_CACHE
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&key))
    {
        return cached;
    }
    let pd = ProsodyDims::from_kokoro(dims);
    let (def, _) = build_scaled_prosody(dims);
    let result = run_duration_proof(
        &def,
        &scaled_prosody_bindings(dims),
        pd.flat_input_size(),
        ib,
    );
    SCALED_PROOF_CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    result
}

/// Scaled single-block proof at given dimensions (cached).
pub(super) fn run_scaled_proof_single_block(
    dims: &KokoroDims,
    ib: f32,
) -> (f64, &'static str, bool, bool) {
    let key = (dims.d_model, ib.to_bits());
    if let Some(&cached) = SCALED_PROOF_1B_CACHE
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&key))
    {
        return cached;
    }
    let pd = ProsodyDims::from_kokoro(dims);
    let (def, _) = build_scaled_prosody_single_block(dims);
    let result = run_duration_proof(
        &def,
        &scaled_prosody_single_block_bindings(dims),
        pd.flat_input_size(),
        ib,
    );
    SCALED_PROOF_1B_CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    result
}

/// Fixed-D=8 single-block proof with custom weight magnitude (cached).
pub(super) fn run_sensitivity_single_block(wm: f32, ib: f32) -> (f64, &'static str, bool) {
    let key = (wm.to_bits(), ib.to_bits());
    if let Some(&cached) = SENS_1B_CACHE
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&key))
    {
        return cached;
    }
    let (def, _) = build_kokoro_prosody_single_block();
    let r = run_duration_proof(
        &def,
        &kokoro_prosody_bindings_with_weight_mag(wm),
        FLAT_INPUT_SIZE,
        ib,
    );
    let result = (r.0, r.1, r.2);
    SENS_1B_CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    result
}

/// Fixed-D=8 three-block proof with custom weight magnitude (cached).
pub(super) fn run_sensitivity_three_block(wm: f32, ib: f32) -> (f64, &'static str, bool) {
    let key = (wm.to_bits(), ib.to_bits());
    if let Some(&cached) = SENS_3B_CACHE
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|m| m.get(&key))
    {
        return cached;
    }
    let (def, _) = build_kokoro_prosody_three_blocks();
    let r = run_duration_proof(
        &def,
        &kokoro_prosody_three_block_bindings_with_weight_mag(wm),
        FLAT_INPUT_SIZE,
        ib,
    );
    let result = (r.0, r.1, r.2);
    SENS_3B_CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(key, result);
    result
}

/// Binary search for maximum provable input bound at given dimensions.
pub(super) fn crossover_sweep(dims: &KokoroDims, hi_init: f32, n_iter: usize) -> (f32, f32) {
    let mut lo = 0.1_f32;
    let mut hi = hi_init;
    for _ in 0..n_iter {
        let mid = f32::midpoint(lo, hi);
        let (_, _, is_proven, _) = run_scaled_proof(dims, mid);
        if is_proven {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo, hi)
}

/// Simple CROWN vs IBP comparison (shared by single-block and T=4 tests).
pub(super) fn crown_vs_ibp_simple(
    def: &nn_dsl::TensorKernelDef,
    bindings: &[nn_verify::TensorParamBinding],
    input_size: usize,
    label: &str,
) {
    let graph = tensor_kernel_to_graph(def, bindings).expect("graph translation");
    let input = uniform_bounds(&[input_size], 1.0);
    let ibp_output = graph.propagate_ibp(&input).expect("IBP baseline");
    let ibp_lo_min = lo_min_of(&ibp_output) as f32;
    let (method, crown_output, _) =
        nn_verify::propagate_with_crown_fallback(&graph, &input).expect("CROWN propagation");
    let crown_lo_min = lo_min_of(&crown_output) as f32;
    assert_bounds_valid(&crown_output);
    if matches!(method, nn_verify::PropMethod::Crown) {
        assert!(
            crown_lo_min >= ibp_lo_min - 1e-4,
            "CROWN lower {crown_lo_min:.6} should be >= IBP lower {ibp_lo_min:.6}"
        );
    }
    eprintln!(
        "{label}: IBP lo_min={ibp_lo_min:.6}, CROWN lo_min={crown_lo_min:.6}, method={method:?}"
    );
}

/// Print duration positivity certificates for a list of dimensions.
pub(super) fn print_certificates(dims_list: &[(KokoroDims, &str)], header: &str) {
    eprintln!("\n=== {header} ===");
    for (dims, label) in dims_list {
        let pd = ProsodyDims::from_kokoro(dims);
        let (lo_min, method, _, _) = run_scaled_proof(dims, 1.0);
        let cert =
            interpret_duration_positivity(lo_min, DURATION_THRESHOLD, 1.0, 1.0, pd.seq_len, method);
        eprintln!(
            "  {label}: proven={}, lower={:.6}",
            cert.is_proven, cert.lower_bound
        );
    }
}
