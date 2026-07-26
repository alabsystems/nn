// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Load pre-computed CROWN bounds from external solvers (e.g., auto_LiRPA).
//!
//! The bridge accepts safetensors files containing output and per-layer
//! bounds produced by any sound overapproximation engine. The primary
//! use case is importing auto_LiRPA (alpha-beta-CROWN) bounds when
//! NY's CROWN propagation delivers IBP-quality results due to
//! un-optimized alpha parameters (#1927).
//!
//! Soundness mode: `Heuristic`. The bounds are mathematically sound within
//! auto_LiRPA's framework, but the Python execution is unverified. A
//! separate Kani proof or NY cross-check can upgrade to `Sound`.
//!
//! See `designs/2026-03-12-auto-lirpa-bounds-bridge.md` for full design.

use std::collections::BTreeMap;
use std::path::Path;

use crate::error::VerifyError;
use crate::soundness_compat::VerificationSoundnessMode;
use crate::status::{ParamInputRecord, VerifyStatus};
use crate::verify_types::{KernelVerification, OutputTensorBounds, PropMethod};

/// Metadata about the external bounds source.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ExternalBoundsSource {
    /// Propagation method (e.g., "CROWN-Optimized", "alpha-CROWN").
    pub method: String,
    /// Engine that produced the bounds (e.g., "auto_LiRPA").
    pub engine: String,
    /// Perturbation radius used for bounds computation.
    pub eps: f64,
    /// Input tensor shape.
    pub input_shape: Vec<usize>,
}

impl ExternalBoundsSource {
    /// Construct a new source descriptor.
    #[must_use]
    pub fn new(method: String, engine: String, eps: f64, input_shape: Vec<usize>) -> Self {
        Self {
            method,
            engine,
            eps,
            input_shape,
        }
    }
}

/// Per-layer bounds from an external solver.
#[derive(Debug, Clone)]
pub struct ExternalLayerBounds {
    /// Lower bounds (flattened, row-major).
    pub lower: Vec<f32>,
    /// Upper bounds (flattened, row-major).
    pub upper: Vec<f32>,
    /// Tensor shape.
    pub shape: Vec<usize>,
}

/// Pre-computed bounds loaded from a safetensors file.
#[derive(Debug, Clone)]
pub struct ExternalBounds {
    /// Metadata about the source engine and configuration.
    pub source: ExternalBoundsSource,
    /// Output bounds (final model output).
    pub output_lower: Vec<f32>,
    /// Output upper bounds.
    pub output_upper: Vec<f32>,
    /// Output tensor shape.
    pub output_shape: Vec<usize>,
    /// Per-layer intermediate bounds, keyed by layer name.
    pub layer_bounds: BTreeMap<String, ExternalLayerBounds>,
}

/// Load pre-computed bounds from a safetensors file.
///
/// The file must contain at least `output/lower` and `output/upper` tensors
/// (F32). Per-layer bounds are loaded from `layer/{name}/lower` and
/// `layer/{name}/upper` keys. Metadata (method, engine, eps, input_shape)
/// is read from the safetensors header.
///
/// # Errors
///
/// Returns `VerifyError::Io` for file read failures,
/// `VerifyError::InvalidInput` for missing tensors or non-finite values.
pub fn load_external_bounds(path: impl AsRef<Path>) -> Result<ExternalBounds, VerifyError> {
    let data = std::fs::read(path.as_ref())?;
    load_external_bounds_from_bytes(&data)
}

/// Load pre-computed bounds from in-memory safetensors data.
///
/// Same as [`load_external_bounds`] but operates on a byte slice.
pub fn load_external_bounds_from_bytes(data: &[u8]) -> Result<ExternalBounds, VerifyError> {
    let st = safetensors::SafeTensors::deserialize(data).map_err(|e| {
        VerifyError::InvalidInput(format!("safetensors deserialization failed: {e}"))
    })?;

    // Load output bounds (required).
    let output_lower = load_f32_tensor(&st, "output/lower")?;
    let output_upper = load_f32_tensor(&st, "output/upper")?;

    let output_shape = tensor_shape(&st, "output/lower")?;

    // Validate output bounds.
    if output_lower.len() != output_upper.len() {
        return Err(VerifyError::InvalidInput(format!(
            "output lower/upper length mismatch: {} vs {}",
            output_lower.len(),
            output_upper.len()
        )));
    }
    validate_finite(&output_lower, "output/lower")?;
    validate_finite(&output_upper, "output/upper")?;
    validate_ordering(&output_lower, &output_upper, "output")?;

    // Parse metadata from safetensors header (needs raw bytes for read_metadata).
    let source = parse_source_metadata(data)?;

    // Load per-layer bounds (optional).
    let layer_bounds = load_layer_bounds(&st)?;

    Ok(ExternalBounds {
        source,
        output_lower,
        output_upper,
        output_shape,
        layer_bounds,
    })
}

/// Build a [`KernelVerification`] from external bounds.
///
/// The verification uses `PropMethod::Crown` with `Heuristic` soundness mode.
/// The `status_key` is used as the kernel name in the verification record.
///
/// Input bounds (`input_lower`, `input_upper`) are scalar summaries of the
/// perturbation set used by the external solver (typically `center - eps`
/// and `center + eps`).
pub fn verification_from_external(
    external: &ExternalBounds,
    status_key: &str,
) -> KernelVerification {
    let lo_min = external
        .output_lower
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let hi_max = external
        .output_upper
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let width = hi_max - lo_min;
    let is_finite = lo_min.is_finite() && hi_max.is_finite();

    let finite_mask: Vec<bool> = external
        .output_lower
        .iter()
        .zip(external.output_upper.iter())
        .map(|(&lo, &hi)| lo.is_finite() && hi.is_finite())
        .collect();

    let output_tensor = OutputTensorBounds {
        lower: external.output_lower.clone(),
        upper: external.output_upper.clone(),
        shape: external.output_shape.clone(),
        finite_mask,
    };

    let mut verification = KernelVerification::new(
        status_key.to_string(),
        PropMethod::Crown,
        lo_min,
        hi_max,
        width,
        is_finite,
    )
    .with_soundness_mode(VerificationSoundnessMode::Heuristic);

    verification.output_tensor = Some(output_tensor);
    verification
}

/// Load external bounds from a file, build verification, and record in status.
///
/// Convenience entry point that chains [`load_external_bounds`] →
/// [`verification_from_external`] → [`VerifyStatus::record_with_variable_inputs`].
///
/// Input bounds are recorded as a single variable with range `[-eps, +eps]`
/// matching the L-infinity perturbation ball from the external solver.
///
/// # Errors
///
/// Returns an error if the file cannot be read, deserialization fails,
/// bounds contain non-finite values, or status recording fails.
pub fn verify_and_record_external(
    path: impl AsRef<Path>,
    status: &mut VerifyStatus,
    status_key: &str,
) -> Result<(ExternalBounds, KernelVerification), VerifyError> {
    let external = load_external_bounds(path)?;
    verify_and_record_external_from_loaded(external, status, status_key)
}

/// Build verification from already-loaded bounds and record in status.
///
/// Same as [`verify_and_record_external`] but accepts pre-loaded
/// [`ExternalBounds`], avoiding a redundant file read when bounds
/// are already in memory.
///
/// # Errors
///
/// Returns an error if status recording fails.
pub fn verify_and_record_external_from_loaded(
    external: ExternalBounds,
    status: &mut VerifyStatus,
    status_key: &str,
) -> Result<(ExternalBounds, KernelVerification), VerifyError> {
    let verification = verification_from_external(&external, status_key);

    // Record input bounds as a single variable with eps-ball range.
    #[allow(clippy::cast_possible_truncation)]
    let eps = external.source.eps as f32;
    let variable_inputs = [ParamInputRecord {
        param_index: 0,
        lower: -eps,
        upper: eps,
    }];
    status.record_with_variable_inputs(
        &verification,
        &variable_inputs,
        &[],
        Some(status_key),
        None, // external bounds — shape not available
    )?;

    Ok((external, verification))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Load a single F32 tensor from safetensors, using safe byte-level decoding.
fn load_f32_tensor(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f32>, VerifyError> {
    let view = st
        .tensor(name)
        .map_err(|e| VerifyError::InvalidInput(format!("missing tensor '{name}': {e}")))?;

    if view.dtype() != safetensors::Dtype::F32 {
        return Err(VerifyError::InvalidInput(format!(
            "tensor '{name}' has dtype {:?}, expected F32",
            view.dtype()
        )));
    }

    let raw = view.data();
    let numel: usize = view
        .shape()
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| {
            VerifyError::InvalidInput(format!("shape product overflow for tensor '{name}'"))
        })?;

    let expected_bytes = numel.checked_mul(4).ok_or_else(|| {
        VerifyError::InvalidInput(format!("byte count overflow for tensor '{name}'"))
    })?;

    if raw.len() != expected_bytes {
        return Err(VerifyError::InvalidInput(format!(
            "tensor '{name}': expected {expected_bytes} bytes, got {}",
            raw.len()
        )));
    }

    Ok(raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Extract the shape of a named tensor.
fn tensor_shape(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<usize>, VerifyError> {
    let view = st
        .tensor(name)
        .map_err(|e| VerifyError::InvalidInput(format!("missing tensor '{name}': {e}")))?;
    Ok(view.shape().to_vec())
}

/// Validate that all values in a slice are finite.
fn validate_finite(values: &[f32], context: &str) -> Result<(), VerifyError> {
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            return Err(VerifyError::InvalidInput(format!(
                "non-finite value at index {i} in '{context}': {v}"
            )));
        }
    }
    Ok(())
}

/// Validate that lower[i] <= upper[i] for all elements (defense-in-depth).
///
/// External solvers could produce inverted intervals. Catching them here
/// prevents unsound verification results downstream.
fn validate_ordering(lower: &[f32], upper: &[f32], context: &str) -> Result<(), VerifyError> {
    for (i, (&lo, &hi)) in lower.iter().zip(upper.iter()).enumerate() {
        if lo > hi {
            return Err(VerifyError::InvalidInput(format!(
                "inverted bounds at index {i} in '{context}': lower={lo} > upper={hi}"
            )));
        }
    }
    Ok(())
}

/// Parse source metadata from safetensors header.
///
/// Uses `read_metadata()` to access the `Metadata` struct which provides
/// the `metadata()` accessor. `SafeTensors` does not expose this directly.
fn parse_source_metadata(data: &[u8]) -> Result<ExternalBoundsSource, VerifyError> {
    let empty = std::collections::HashMap::new();
    let (_, header) = safetensors::SafeTensors::read_metadata(data)
        .map_err(|e| VerifyError::InvalidInput(format!("safetensors metadata read failed: {e}")))?;
    let meta = header.metadata().as_ref().unwrap_or(&empty);

    let method = meta
        .get("method")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let engine = meta
        .get("engine")
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let eps: f64 = meta
        .get("eps")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let input_shape: Vec<usize> = meta
        .get("input_shape")
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();

    Ok(ExternalBoundsSource::new(method, engine, eps, input_shape))
}

/// Load all per-layer bounds from the safetensors file.
fn load_layer_bounds(
    st: &safetensors::SafeTensors<'_>,
) -> Result<BTreeMap<String, ExternalLayerBounds>, VerifyError> {
    let mut layers = BTreeMap::new();
    let names: Vec<String> = st.names().into_iter().map(String::from).collect();
    // O(N) HashSet avoids O(N²) linear scan per upper_key lookup (#3020).
    let name_set: std::collections::HashSet<&str> = names.iter().map(String::as_str).collect();

    for name in &names {
        if let Some(layer_name) = name
            .strip_prefix("layer/")
            .and_then(|rest| rest.strip_suffix("/lower"))
        {
            let upper_key = format!("layer/{layer_name}/upper");
            // Only include layers that have both lower and upper.
            if name_set.contains(upper_key.as_str()) {
                let lower = load_f32_tensor(st, name)?;
                let upper = load_f32_tensor(st, &upper_key)?;
                let shape = tensor_shape(st, name)?;

                if lower.len() != upper.len() {
                    return Err(VerifyError::InvalidInput(format!(
                        "layer '{layer_name}' lower/upper length mismatch: {} vs {}",
                        lower.len(),
                        upper.len()
                    )));
                }

                validate_finite(&lower, &format!("layer/{layer_name}/lower"))?;
                validate_finite(&upper, &format!("layer/{layer_name}/upper"))?;
                validate_ordering(&lower, &upper, &format!("layer/{layer_name}"))?;

                layers.insert(
                    layer_name.to_string(),
                    ExternalLayerBounds {
                        lower,
                        upper,
                        shape,
                    },
                );
            }
        }
    }

    Ok(layers)
}

#[cfg(test)]
#[path = "external_bounds_tests.rs"]
mod tests;
