// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Weight remapping, loading, and status recording helpers for Kokoro
//! production-weight verification tests.
//!
//! Extracted from `compose_kokoro_production.rs` for 500-line compliance.
//! Part of #2633.

use ny_api::NyError;
use nn_core::dyn_tensor::trace::record_input;
use nn_core::dyn_tensor::DynTensor;
use nn_core::test_utils::cpu;
use nn_core::{load_safetensors, DType};
use nn_models::KokoroConfig;
use nn_verify::{
    model_for_kernel, model_status_path, BoundedTensor, LayerBoundRecord, PropMethod,
    VerificationSoundnessMode, VerifyStatus,
};
use ndarray::{ArrayD, Axis, IxDyn};
use std::collections::HashMap;
use std::path::Path;

// -- v1.0 weight norm decomposition ------------------------------------------

/// Decompose weight_norm parametrization: merge `weight_g` + `weight_v` → `weight`.
///
/// PyTorch weight_norm stores weights as:
///   - `name.weight_g`: `[out_ch, 1, ...]` magnitude
///   - `name.weight_v`: `[out_ch, in_ch, ...]` direction
///
/// The actual weight is: `weight = weight_v * (weight_g / ||weight_v||)`,
/// where the norm is over all dims except dim 0 (per output channel).
///
/// Matches Python `decompose_weight_norm` in `kokoro_reference_keyremap.py`.
fn decompose_weight_norm(tensors: HashMap<String, DynTensor>) -> HashMap<String, DynTensor> {
    let g_keys: Vec<String> = tensors
        .keys()
        .filter(|k| k.ends_with(".weight_g"))
        .cloned()
        .collect();

    let mut out = HashMap::with_capacity(tensors.len());
    let mut merged_bases = std::collections::HashSet::new();

    for gk in &g_keys {
        let base = &gk[..gk.len() - ".weight_g".len()];
        let vk = format!("{base}.weight_v");
        if let (Some(g), Some(v)) = (tensors.get(gk), tensors.get(&vk)) {
            match decompose_single_weight_norm(g, v) {
                Ok(weight) => {
                    let wk = format!("{base}.weight");
                    eprintln!("Weight norm decomposed: {base} ({:?})", weight.dims());
                    out.insert(wk, weight);
                    merged_bases.insert(base.to_string());
                }
                Err(e) => {
                    eprintln!("Weight norm decomposition failed for {base}: {e}");
                }
            }
        }
    }

    // Copy all non-weight_norm keys
    for (k, v) in tensors {
        if k.ends_with(".weight_g") || k.ends_with(".weight_v") {
            let base = if k.ends_with(".weight_g") {
                &k[..k.len() - ".weight_g".len()]
            } else {
                &k[..k.len() - ".weight_v".len()]
            };
            if merged_bases.contains(base) {
                continue;
            }
        }
        out.insert(k, v);
    }

    if !merged_bases.is_empty() {
        eprintln!(
            "Weight norm: {} pairs merged, {} → {} tensors",
            merged_bases.len(),
            merged_bases.len() * 2 + out.len() - merged_bases.len(),
            out.len()
        );
    }
    out
}

/// Decompose a single weight_norm pair into a merged weight tensor.
fn decompose_single_weight_norm(
    weight_g: &DynTensor,
    weight_v: &DynTensor,
) -> Result<DynTensor, String> {
    let g_arr = weight_g
        .to_f32_array()
        .map_err(|e| format!("weight_g to_f32_array: {e}"))?;
    let v_arr = weight_v
        .to_f32_array()
        .map_err(|e| format!("weight_v to_f32_array: {e}"))?;

    let out_ch = v_arr.shape()[0];
    let mut result = v_arr.clone();

    for oc in 0..out_ch {
        // L2 norm of weight_v[oc] over all dims except dim 0
        let v_slice = v_arr.index_axis(Axis(0), oc);
        let norm: f32 = v_slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);

        // g value for this output channel (scalar from [out_ch, 1, ...])
        let g_val = g_arr
            .index_axis(Axis(0), oc)
            .iter()
            .next()
            .copied()
            .unwrap_or(1.0);

        let scale = g_val / norm;
        result
            .index_axis_mut(Axis(0), oc)
            .mapv_inplace(|v| v * scale);
    }

    DynTensor::from_cpu_f32(result).map_err(|e| format!("from_cpu_f32: {e}"))
}

// -- v1.0 weight key remapping ------------------------------------------------

/// LSTM terminal key substitutions: `weight_ih.weight` → `weight_ih_l0`, etc.
///
/// Applied to the END of a key. Handles the v1.0 safetensors convention where
/// LSTM weights are split into `.weight` and `.bias` sub-keys instead of using
/// PyTorch's `weight_ih_l0` / `bias_ih_l0` flat naming.
const LSTM_TERMINAL: &[(&str, &str)] = &[
    ("weight_ih.weight", "weight_ih_l0"),
    ("weight_ih.bias", "bias_ih_l0"),
    ("weight_hh.weight", "weight_hh_l0"),
    ("weight_hh.bias", "bias_hh_l0"),
];

/// BiLstm::load direction-folding: `.forward.X.Y` → `.X_l0`, `.backward.X.Y` → `.X_l0_reverse`.
///
/// Applied to keys that end with `.forward.weight_ih.weight` etc. Strips the
/// direction prefix and folds it into `_reverse` suffix (BiLstm::load convention).
const LSTM_BILSTM: &[(&str, &str)] = &[
    (".forward.weight_ih.weight", ".weight_ih_l0"),
    (".forward.weight_ih.bias", ".bias_ih_l0"),
    (".forward.weight_hh.weight", ".weight_hh_l0"),
    (".forward.weight_hh.bias", ".bias_hh_l0"),
    (".backward.weight_ih.weight", ".weight_ih_l0_reverse"),
    (".backward.weight_ih.bias", ".bias_ih_l0_reverse"),
    (".backward.weight_hh.weight", ".weight_hh_l0_reverse"),
    (".backward.weight_hh.bias", ".bias_hh_l0_reverse"),
];

/// Generate all alias keys for a v1.0 safetensors key.
///
/// Each input key may produce multiple aliases (prefix remap + LSTM remap +
/// ResBlock path remap). VarBuilder will find whichever key matches its
/// naming convention.
fn generate_aliases(key: &str) -> Vec<String> {
    // Step 1a: prefix remaps produce additional keys
    let mut prefixed = Vec::new();
    if key.starts_with("bert.") {
        prefixed.push(key.replacen("bert.", "bert_encoder.", 1));
    }
    if key.starts_with("predictor.duration_lstm.") {
        prefixed.push(key.replacen("predictor.duration_lstm.", "prosody_predictor.lstm.", 1));
    } else if key.starts_with("predictor.duration.") {
        prefixed.push(key.replacen("predictor.duration.", "prosody_predictor.duration.", 1));
    }

    // Step 1b: decoder key remap — v1.0 safetensors uses `decoder.generator.*`
    // but tests load Generator directly at `vb.pp("decoder")` which resolves
    // to `decoder.*`. Add both directions:
    //   decoder.generator.X → decoder.X (for direct Generator::load)
    //   decoder.X → decoder.generator.X (for FullDecoder::load)
    // Matches Python `_remap_decoder_keys` in kokoro_reference_keyremap.py.
    const GENERATOR_SUBS: &[&str] = &[
        "ups.",
        "resblocks.",
        "noise_res.",
        "noise_convs.",
        "conv_post.",
        "m_source.",
    ];
    for &sub in GENERATOR_SUBS {
        let with_gen = format!("decoder.generator.{sub}");
        let without_gen = format!("decoder.{sub}");
        // decoder.generator.X → decoder.X
        if key.starts_with(&with_gen) {
            let rest = &key[with_gen.len()..];
            prefixed.push(format!("{without_gen}{rest}"));
            break;
        }
        // decoder.X → decoder.generator.X
        if key.starts_with(&without_gen) {
            let rest = &key[without_gen.len()..];
            prefixed.push(format!("{with_gen}{rest}"));
            break;
        }
    }

    // Step 2: apply LSTM remaps to ALL keys (original + prefix-remapped)
    let all_variants: Vec<String> = std::iter::once(key.to_string())
        .chain(prefixed.iter().cloned())
        .collect();
    let mut results = prefixed; // start with prefix remaps

    for k in &all_variants {
        let k = k.as_str();
        // Terminal remap: `.weight_ih.weight` → `.weight_ih_l0` (for manual pp("forward") loaders)
        for &(from, to) in LSTM_TERMINAL {
            if let Some(base) = k.strip_suffix(from) {
                results.push(format!("{base}{to}"));
            }
        }
        // BiLstm fold: `.forward.weight_ih.weight` → `.weight_ih_l0` (for BiLstm::load)
        for &(from, to) in LSTM_BILSTM {
            if let Some(base) = k.strip_suffix(from) {
                results.push(format!("{base}{to}"));
            }
        }
    }

    // Step 3: ResBlock paths remap (v1.0 uses paths.{i}.c1/c2/n1/n2/s1/s2,
    // Rust expects convs1.{i}/convs2.{i}/adain1.{i}/adain2.{i}/alpha1.{i}/alpha2.{i}).
    // Matches Python `_remap_resblock_paths` in kokoro_reference_keyremap.py.
    let mut path_aliases = Vec::new();
    for k in std::iter::once(key.to_string()).chain(results.iter().cloned()) {
        if let Some(alias) = remap_resblock_path(&k) {
            path_aliases.push(alias);
        }
    }
    results.extend(path_aliases);

    // Step 4: AdaIn layer name remap (v1.0 uses conv1/conv2/norm1/norm2/conv1x1,
    // Python remaps to c1/c2/n1/n2/skip before ResBlock path remap).
    let mut adain_aliases = Vec::new();
    for k in std::iter::once(key.to_string()).chain(results.iter().cloned()) {
        if let Some(alias) = remap_adain_layer_name(&k) {
            // Chain: adain remap may produce a key that also needs resblock path remap
            if let Some(chained) = remap_resblock_path(&alias) {
                adain_aliases.push(chained);
            }
            adain_aliases.push(alias);
        }
    }
    results.extend(adain_aliases);

    results
}

/// Remap ResBlock `paths.{i}.{layer}` to Rust naming convention.
///
/// - `paths.{i}.c1.*` → `convs1.{i}.*`
/// - `paths.{i}.c2.*` → `convs2.{i}.*`
/// - `paths.{i}.n1.*` → `adain1.{i}.*`
/// - `paths.{i}.n2.*` → `adain2.{i}.*`
/// - `paths.{i}.s1.alpha` → `alpha1.{i}`
/// - `paths.{i}.s2.alpha` → `alpha2.{i}`
fn remap_resblock_path(key: &str) -> Option<String> {
    // Match: prefix.paths.{idx}.{layer}.rest
    let paths_idx = key.find(".paths.")?;
    let after_paths = &key[paths_idx + ".paths.".len()..];
    let dot_after_idx = after_paths.find('.')?;
    let idx = &after_paths[..dot_after_idx];
    let after_idx = &after_paths[dot_after_idx + 1..];
    let prefix = &key[..paths_idx];

    // Conv remap: c1.rest → convs1.{idx}.rest, c2.rest → convs2.{idx}.rest
    if let Some(rest) = after_idx.strip_prefix("c1.") {
        return Some(format!("{prefix}.convs1.{idx}.{rest}"));
    }
    if let Some(rest) = after_idx.strip_prefix("c2.") {
        return Some(format!("{prefix}.convs2.{idx}.{rest}"));
    }
    // Norm remap: n1.rest → adain1.{idx}.rest, n2.rest → adain2.{idx}.rest
    if let Some(rest) = after_idx.strip_prefix("n1.") {
        return Some(format!("{prefix}.adain1.{idx}.{rest}"));
    }
    if let Some(rest) = after_idx.strip_prefix("n2.") {
        return Some(format!("{prefix}.adain2.{idx}.{rest}"));
    }
    // Snake alpha: s1.alpha → alpha1.{idx}, s2.alpha → alpha2.{idx}
    if after_idx == "s1.alpha" {
        return Some(format!("{prefix}.alpha1.{idx}"));
    }
    if after_idx == "s2.alpha" {
        return Some(format!("{prefix}.alpha2.{idx}"));
    }
    None
}

/// Remap AdaIn layer shorthand names from v1.0 PyTorch to Rust expectations.
///
/// - `conv1` → `c1`, `conv2` → `c2`
/// - `norm1` → `n1`, `norm2` → `n2`
/// - `conv1x1` → `skip`
///
/// Matches Python `_ADAIN_LAYER_RENAME` in kokoro_reference_keyremap.py.
fn remap_adain_layer_name(key: &str) -> Option<String> {
    // These patterns appear within F0/N AdainResBlk1d and generator ResBlocks
    const RENAMES: &[(&str, &str)] = &[
        (".conv1.", ".c1."),
        (".conv2.", ".c2."),
        (".norm1.", ".n1."),
        (".norm2.", ".n2."),
        (".conv1x1.", ".skip."),
    ];
    for &(from, to) in RENAMES {
        if key.contains(from) {
            return Some(key.replacen(from, to, 1));
        }
    }
    None
}

/// Add synthetic identity weights for layers missing in v1.0 architecture.
///
/// v1.0 Kokoro lacks:
/// - `text_encoder.lstm.linear` (Linear(d_en, d_en) projection after BiLSTM)
/// - `decoder.conv_pre` (Conv1d(ch, ch, 7) input projection for Generator)
///
/// Identity weights are mathematically neutral (output = input), so IBP bounds
/// propagate unchanged through these synthetic layers.
fn add_synthetic_weights(out: &mut HashMap<String, DynTensor>, d_en: usize, gen_ch: usize) {
    let dev = cpu();

    // text_encoder.lstm.linear: identity Linear(d_en, d_en)
    if !out.contains_key("text_encoder.lstm.linear.weight") {
        let mut identity_data = vec![0.0f32; d_en * d_en];
        for i in 0..d_en {
            identity_data[i * d_en + i] = 1.0;
        }
        let identity =
            DynTensor::from_vec(identity_data, &[d_en, d_en], &dev).expect("identity matrix");
        let zeros = DynTensor::zeros(&[d_en], DType::F32, &dev).expect("zeros");
        out.insert("text_encoder.lstm.linear.weight".into(), identity);
        out.insert("text_encoder.lstm.linear.bias".into(), zeros);
        eprintln!("Synthetic: text_encoder.lstm.linear (identity {d_en}x{d_en})");
    }

    // decoder.conv_pre: identity Conv1d(gen_ch, gen_ch, kernel_size=7, padding=3)
    // Center-tap kernel: weight[oc, ic, 3] = delta(oc==ic), all else 0.
    if !out.contains_key("decoder.conv_pre.weight") {
        let mut kernel_data = vec![0.0f32; gen_ch * gen_ch * 7];
        let center = 3; // kernel_size=7, center at index 3
        for c in 0..gen_ch {
            kernel_data[c * gen_ch * 7 + c * 7 + center] = 1.0;
        }
        let kernel = DynTensor::from_vec(kernel_data, &[gen_ch, gen_ch, 7], &dev)
            .expect("identity conv kernel");
        let zeros = DynTensor::zeros(&[gen_ch], DType::F32, &dev).expect("zeros");
        out.insert("decoder.conv_pre.weight".into(), kernel);
        out.insert("decoder.conv_pre.bias".into(), zeros);
        eprintln!("Synthetic: decoder.conv_pre (identity conv {gen_ch}x{gen_ch}x7)");
    }
}

/// Remap v1.0 Kokoro safetensors keys to match nn model loader expectations.
///
/// v1.0 differences from the Rust model loaders:
/// - Weight norm: `weight_g` + `weight_v` → merged `weight`
/// - `bert.` → `bert_encoder.` (top-level linear projection)
/// - `predictor.duration.` → `prosody_predictor.duration.` (duration encoder)
/// - `predictor.duration_lstm.` → `prosody_predictor.lstm.` (final duration BiLSTM)
/// - LSTM: `forward.weight_ih.weight` → `weight_ih_l0` (PyTorch l0 convention)
/// - LSTM: `backward.weight_ih.weight` → `weight_ih_l0_reverse`
/// - ResBlock: `paths.{i}.c1/c2` → `convs1.{i}/convs2.{i}`
/// - ResBlock: `paths.{i}.n1/n2` → `adain1.{i}/adain2.{i}`
/// - ResBlock: `paths.{i}.s1/s2.alpha` → `alpha1.{i}/alpha2.{i}`
/// - AdaIn: `conv1/conv2/norm1/norm2/conv1x1` → `c1/c2/n1/n2/skip`
/// - Missing `text_encoder.lstm.linear` → synthetic identity Linear
/// - Missing `decoder.conv_pre` → synthetic identity Conv1d
fn remap_v1_weights(
    tensors: HashMap<String, DynTensor>,
    d_en: usize,
    gen_ch: usize,
) -> HashMap<String, DynTensor> {
    let input_count = tensors.len();
    // Step 1: decompose weight_norm pairs (weight_g + weight_v → weight)
    let tensors = decompose_weight_norm(tensors);

    // Step 2: generate aliases for key naming differences
    let mut out = HashMap::with_capacity(tensors.len() * 2);
    for (key, val) in &tensors {
        out.insert(key.clone(), val.clone());
        for alias in generate_aliases(key) {
            out.insert(alias, val.clone());
        }
    }

    // Step 3: add synthetic identity weights for missing layers
    add_synthetic_weights(&mut out, d_en, gen_ch);
    eprintln!(
        "v1.0 remap: {} input keys → {} output keys (after weight_norm + aliases)",
        input_count,
        out.len()
    );
    out
}

// -- Weight loading helpers ---------------------------------------------------

/// Load production safetensors weights, or return None if unavailable.
///
/// Uses `KOKORO_WEIGHTS` env var (matches existing test convention).
/// Applies v1.0 key remapping to support both v0.19 and v1.0 weight naming.
///
/// **Prefer `require_production_weights()`** in tests gated behind
/// `#[cfg(feature = "production-weights")]` — it panics with a clear
/// message instead of silently passing (#2716).
pub(super) fn load_production_weights() -> Option<HashMap<String, DynTensor>> {
    let path = std::env::var("KOKORO_WEIGHTS").ok()?;
    if path.is_empty() || !Path::new(&path).exists() {
        return None;
    }
    let tensors = load_safetensors(&path)
        .map_err(|e| eprintln!("Failed to load safetensors: {e}"))
        .ok()?;
    let config = KokoroConfig::default();
    Some(remap_v1_weights(
        tensors,
        config.d_en,
        config.gen_initial_channels,
    ))
}

/// Load production weights, panicking if unavailable.
///
/// Use in tests gated behind `#[cfg(feature = "production-weights")]`.
/// Eliminates silent pass-on-skip pattern (#2716).
pub(super) fn require_production_weights() -> HashMap<String, DynTensor> {
    load_production_weights().expect(
        "KOKORO_WEIGHTS must be set when production-weights feature is active. \
         Set KOKORO_WEIGHTS=/path/to/kokoro_v1_0.safetensors",
    )
}

/// Register a trace input and return the tensor with trace ID set.
///
/// Uses the tensor's actual dtype (not hardcoded F32) so I64 token inputs
/// are recorded correctly in the trace graph (#2598 audit).
pub(super) fn trace_input(t: &DynTensor) -> DynTensor {
    let mut out = t.clone();
    out.set_trace_id(record_input(out.dims(), out.dtype()).expect("tracing active"));
    out
}

/// Build flat stacked bounds for multi-input graphs.
///
/// Each entry is `(shape, (lower, upper))`. Returns a flat 1D `BoundedTensor`
/// with all input elements concatenated (same layout used by multi-input IBP).
pub(super) fn build_multi_input_bounds(inputs: &[(&[usize], (f32, f32))]) -> BoundedTensor {
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for &(shape, (lo, hi)) in inputs {
        let flat: usize = shape.iter().product();
        lower.extend(vec![lo; flat]);
        upper.extend(vec![hi; flat]);
    }
    let total = lower.len();
    BoundedTensor::new(
        ArrayD::from_shape_vec(IxDyn(&[total]), lower).unwrap(),
        ArrayD::from_shape_vec(IxDyn(&[total]), upper).unwrap(),
    )
    .expect("valid bounds")
}

// -- Status recording helpers -------------------------------------------------

/// Record a segment's IBP verification result to the per-model status file.
///
/// Uses `record_pipeline` with `load_locked` + `save` for concurrent safety.
/// Persists results to `nn_verify_status_kokoro.json` so production weight
/// verification clears stale entries automatically (#2461).
pub(super) fn record_segment(
    status_key: &str,
    input_bounds: &BoundedTensor,
    output: &BoundedTensor,
) {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model = model_for_kernel(status_key);
    let model_path = model_status_path(ws, model);
    let mut locked = VerifyStatus::load_locked(&model_path).expect("load_locked");

    let (in_lo, in_hi) = super::common::bounds_min_max(input_bounds);
    let (out_lo, out_hi) = super::common::bounds_min_max(output);

    // Record as 1D flattened shape for the scalar summary.
    let (lo_arr, _) = output.lower_upper();
    let out_shape = [lo_arr.len()];

    locked
        .status
        .record_pipeline(
            status_key,
            PropMethod::Ibp,
            in_lo,
            in_hi,
            out_lo,
            out_hi,
            &out_shape,
            VerificationSoundnessMode::Heuristic,
            Some(input_bounds.shape()),
        )
        .expect("record_pipeline");
    locked
        .status
        .set_soundness_justification(
            status_key,
            "IBP through InstanceNorm/RMSNorm uses forward-pass midpoint statistics",
        )
        .expect("set justification");
    locked.save().expect("save status");
    eprintln!("Recorded {status_key} to status file (stale=false)");
}

pub(super) fn propagate_with_tight_crown_fallback(
    graph: &nn_verify::GraphNetwork,
    input_bounds: &BoundedTensor,
) -> Result<(PropMethod, BoundedTensor, Option<String>), nn_verify::VerifyError> {
    match graph.propagate_alpha_crown(input_bounds) {
        Ok(alpha_output) => Ok((PropMethod::AlphaCrown, alpha_output, None)),
        Err(alpha_err) => {
            if crown_error_must_propagate(&alpha_err) {
                return Err(alpha_err.into());
            }

            match graph.propagate_crown_with_provenance(input_bounds) {
                Ok(crown_result) => {
                    if crown_result.is_fallback() {
                        Ok((
                            PropMethod::Ibp,
                            crown_result.bounds,
                            Some(format!(
                                "alpha-CROWN failed: {alpha_err}; fixed-slope CROWN fell back to IBP internally"
                            )),
                        ))
                    } else {
                        Ok((
                            PropMethod::Crown,
                            crown_result.bounds,
                            Some(format!(
                                "alpha-CROWN failed: {alpha_err}; fixed-slope CROWN succeeded"
                            )),
                        ))
                    }
                }
                Err(crown_err) => {
                    if crown_error_must_propagate(&crown_err) {
                        return Err(crown_err.into());
                    }

                    let ibp_output = graph.propagate_ibp(input_bounds)?;
                    Ok((
                        PropMethod::Ibp,
                        ibp_output,
                        Some(format!(
                            "alpha-CROWN failed: {alpha_err}; fixed-slope CROWN failed: {crown_err}"
                        )),
                    ))
                }
            }
        }
    }
}

fn crown_error_must_propagate(error: &NyError) -> bool {
    matches!(
        error,
        NyError::SoundnessRefusal(_) | NyError::InternalError(_)
    )
}

pub(super) fn is_tight_crown_method(method: PropMethod) -> bool {
    matches!(
        method,
        PropMethod::Crown | PropMethod::AlphaCrown | PropMethod::BetaCrown
    )
}

pub(super) fn tight_crown_method_name(method: PropMethod) -> &'static str {
    match method {
        PropMethod::Crown => "CROWN",
        PropMethod::AlphaCrown => "AlphaCrown",
        PropMethod::BetaCrown => "BetaCrown",
        _ => "IBP-fallback",
    }
}

/// Record a segment's CROWN verification result to the per-model status file.
///
/// Records with the actual propagation method (AlphaCrown/CROWN/IBP fallback).
/// When a tight CROWN-family method succeeded, also logs the IBP comparison
/// width and tightening ratio.
#[allow(dead_code)] // Used by compose_kokoro_production.rs, not by all test binaries.
pub(super) fn record_segment_crown(
    status_key: &str,
    input_bounds: &BoundedTensor,
    crown_output: &BoundedTensor,
    method: PropMethod,
    ibp_width: Option<f32>,
) {
    let ws = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let model = model_for_kernel(status_key);
    let model_path = model_status_path(ws, model);
    let mut locked = VerifyStatus::load_locked(&model_path).expect("load_locked");

    let (in_lo, in_hi) = super::common::bounds_min_max(input_bounds);
    let (out_lo, out_hi) = super::common::bounds_min_max(crown_output);

    let (lo_arr, _) = crown_output.lower_upper();
    let out_shape = [lo_arr.len()];

    locked
        .status
        .record_pipeline(
            status_key,
            method,
            in_lo,
            in_hi,
            out_lo,
            out_hi,
            &out_shape,
            VerificationSoundnessMode::Heuristic,
            Some(input_bounds.shape()),
        )
        .expect("record_pipeline");
    let justification = match method {
        PropMethod::Crown => "CROWN through normalization layers uses heuristic linearization",
        PropMethod::AlphaCrown => {
            "AlphaCrown through normalization layers uses heuristic linearization"
        }
        PropMethod::BetaCrown => {
            "BetaCrown through normalization layers uses heuristic linearization"
        }
        PropMethod::Ibp => {
            "CROWN-family attempt was no tighter than IBP; recorded the tighter IBP bounds instead"
        }
        PropMethod::MixedIbpCrown => {
            "Mixed IBP/CROWN propagation recorded with heuristic normalization approximation"
        }
        _ => "Production segment recorded with heuristic bound propagation",
    };
    locked
        .status
        .set_soundness_justification(status_key, justification)
        .expect("set justification");
    locked.save().expect("save status");

    let crown_width = out_hi - out_lo;
    if let Some(ibp_w) = ibp_width {
        let ratio = if ibp_w > 0.0 {
            crown_width / ibp_w
        } else {
            1.0
        };
        eprintln!(
            "Recorded {status_key} (method={method:?}): CROWN_w={crown_width:.4}, IBP_w={ibp_w:.4}, ratio={ratio:.6}"
        );
    } else {
        eprintln!("Recorded {status_key} (method={method:?}): width={crown_width:.4}");
    }
}

/// Prefer the tighter recorded result between a CROWN-family method and the
/// IBP baseline.
///
/// Production normalization-heavy segments sometimes return a structurally
/// successful CROWN-family result that is wider than IBP. In that case,
/// persist the tighter IBP bounds instead of keeping a vacuous `_crown` entry.
pub(super) fn prefer_tighter_recorded_output(
    method: PropMethod,
    ibp_output: &BoundedTensor,
    crown_output: &BoundedTensor,
) -> (PropMethod, BoundedTensor, Option<f32>, &'static str) {
    let (ibp_lo, ibp_hi) = super::common::bounds_min_max(ibp_output);
    let ibp_width = ibp_hi - ibp_lo;

    if !is_tight_crown_method(method) {
        return (
            PropMethod::Ibp,
            ibp_output.clone(),
            None,
            "CROWN-family propagation fell back to IBP; recording the IBP baseline",
        );
    }

    let (crown_lo, crown_hi) = super::common::bounds_min_max(crown_output);
    let crown_width = crown_hi - crown_lo;
    if crown_width + 1e-4 < ibp_width {
        (
            method,
            crown_output.clone(),
            Some(ibp_width),
            "CROWN-family bounds are strictly tighter than IBP",
        )
    } else {
        (
            PropMethod::Ibp,
            ibp_output.clone(),
            None,
            "CROWN-family bounds were not strictly tighter than IBP; recording the IBP baseline",
        )
    }
}

/// Result from tracing a segment: layer records + input/output bounds.
#[derive(Debug)]
pub(super) struct SegmentResult {
    pub(super) records: Vec<LayerBoundRecord>,
    pub(super) input_bounds: BoundedTensor,
    pub(super) output_bounds: BoundedTensor,
}

/// Log bounds width statistics.
pub(super) fn log_bounds_width(label: &str, output: &BoundedTensor) {
    let (lo, hi) = output.lower_upper();
    let widths: Vec<f32> = lo.iter().zip(hi.iter()).map(|(l, h)| h - l).collect();
    let max_w = widths.iter().copied().fold(0.0f32, f32::max);
    let min_w = widths.iter().copied().fold(f32::INFINITY, f32::min);
    let n = widths.len();
    let avg_w = if n == 0 {
        0.0
    } else {
        widths.iter().sum::<f32>() / n as f32
    };
    eprintln!("{label} production IBP: {n} elements, min_w={min_w:.4}, avg_w={avg_w:.4}, max_w={max_w:.4}");
}
