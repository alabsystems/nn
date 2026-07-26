// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Automatic weight quantization detection for the import pipeline.
//!
//! Analyzes safetensors weight files to produce a [`QuantizationReport`] with:
//! - Per-tensor dtype breakdown
//! - Estimated total size
//! - Recommended quantization levels for memory savings
//!
//! This is a metadata-only analysis — it reads the safetensors header and
//! tensor info without performing any dtype conversion.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use crate::error::ImportError;

// ---------------------------------------------------------------------------
// Detected dtype (superset of what safetensors exposes)
// ---------------------------------------------------------------------------

/// Dtype categories detected in safetensors weights.
///
/// Maps safetensors `Dtype` variants to broader quantization-relevant
/// categories. Sub-byte types (F4, F6) are grouped under `SubByte`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum DetectedDtype {
    /// 32-bit floating point.
    F32,
    /// 16-bit floating point (IEEE 754).
    F16,
    /// 16-bit brain floating point.
    BF16,
    /// 64-bit floating point.
    F64,
    /// 8-bit signed integer.
    I8,
    /// 8-bit unsigned integer.
    U8,
    /// 8-bit floating point (E5M2 or E4M3 or E8M0).
    F8,
    /// Sub-byte types (4-bit, 6-bit packed).
    SubByte,
    /// 16-bit integer (signed or unsigned).
    I16,
    /// 32-bit integer (signed or unsigned).
    I32,
    /// 64-bit integer (signed or unsigned).
    I64,
    /// Boolean tensors.
    Bool,
    /// Complex 64-bit (32-bit real + 32-bit imag).
    C64,
    /// Unknown or unrecognized dtype.
    Other,
}

impl DetectedDtype {
    /// Convert a safetensors `Dtype` to our detection category.
    pub(crate) fn from_safetensors(dtype: safetensors::Dtype) -> Self {
        use safetensors::Dtype as SD;
        match dtype {
            SD::F32 => Self::F32,
            SD::F16 => Self::F16,
            SD::BF16 => Self::BF16,
            SD::F64 => Self::F64,
            SD::I8 => Self::I8,
            SD::U8 => Self::U8,
            SD::F8_E5M2 | SD::F8_E4M3 | SD::F8_E8M0 => Self::F8,
            SD::F4 | SD::F6_E2M3 | SD::F6_E3M2 => Self::SubByte,
            SD::I16 | SD::U16 => Self::I16,
            SD::I32 | SD::U32 => Self::I32,
            SD::I64 | SD::U64 => Self::I64,
            SD::BOOL => Self::Bool,
            SD::C64 => Self::C64,
            _ => Self::Other,
        }
    }

    /// Bytes per element for this dtype category. Returns `None` for sub-byte.
    pub fn bytes_per_element(&self) -> Option<usize> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::BF16 | Self::I16 => Some(2),
            Self::F64 | Self::I64 | Self::C64 => Some(8),
            Self::I8 | Self::U8 | Self::F8 | Self::Bool => {
                Some(1)
            }
            Self::SubByte | Self::Other => None,
        }
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::U8 => "U8",
            Self::F8 => "F8",
            Self::SubByte => "SubByte",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Bool => "Bool",
            Self::C64 => "C64",
            Self::Other => "Other",
        }
    }
}

impl fmt::Display for DetectedDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

// ---------------------------------------------------------------------------
// Per-tensor info
// ---------------------------------------------------------------------------

/// Information about a single tensor's dtype and size.
#[derive(Debug, Clone)]
pub struct TensorQuantInfo {
    /// Tensor name (fully qualified).
    pub name: String,
    /// Detected dtype category.
    pub dtype: DetectedDtype,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Number of elements (product of shape dimensions).
    pub num_elements: usize,
    /// Size in bytes (from safetensors data offsets).
    pub size_bytes: usize,
}

// ---------------------------------------------------------------------------
// Dtype breakdown bucket
// ---------------------------------------------------------------------------

/// Aggregate statistics for a single dtype category.
#[derive(Debug, Clone)]
pub struct DtypeBreakdown {
    /// Dtype category.
    pub dtype: DetectedDtype,
    /// Number of tensors with this dtype.
    pub tensor_count: usize,
    /// Total number of parameters (elements) across all tensors of this dtype.
    pub total_parameters: usize,
    /// Total size in bytes across all tensors of this dtype.
    pub total_bytes: usize,
}

// ---------------------------------------------------------------------------
// Quantization recommendation
// ---------------------------------------------------------------------------

/// A recommended quantization action for a set of tensors.
#[derive(Debug, Clone)]
pub struct QuantRecommendation {
    /// Target dtype to quantize to.
    pub target_dtype: DetectedDtype,
    /// Tensor names that would benefit from this quantization.
    pub tensor_names: Vec<String>,
    /// Current total bytes for these tensors.
    pub current_bytes: usize,
    /// Estimated bytes after quantization.
    pub projected_bytes: usize,
    /// Estimated savings in bytes.
    pub savings_bytes: usize,
}

// ---------------------------------------------------------------------------
// Quantization report
// ---------------------------------------------------------------------------

/// Full quantization analysis report for a safetensors weight file.
#[derive(Debug, Clone)]
pub struct QuantizationReport {
    /// Per-tensor dtype and size information, sorted by name.
    pub tensors: Vec<TensorQuantInfo>,
    /// Aggregate breakdown by dtype.
    pub dtype_breakdown: Vec<DtypeBreakdown>,
    /// Total number of tensors.
    pub total_tensors: usize,
    /// Total number of parameters across all tensors.
    pub total_parameters: usize,
    /// Total size in bytes.
    pub total_bytes: usize,
    /// Recommended quantization actions for memory savings.
    pub recommendations: Vec<QuantRecommendation>,
}

impl QuantizationReport {
    /// Returns `true` if the model contains multiple dtypes.
    pub fn is_mixed_precision(&self) -> bool {
        self.dtype_breakdown.len() > 1
    }

    /// Returns the fraction of bytes stored in the given dtype.
    pub fn dtype_fraction(&self, dtype: DetectedDtype) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        let dtype_bytes: usize = self
            .dtype_breakdown
            .iter()
            .filter(|b| b.dtype == dtype)
            .map(|b| b.total_bytes)
            .sum();
        dtype_bytes as f64 / self.total_bytes as f64
    }

    /// Total estimated savings across all recommendations.
    pub fn total_savings_bytes(&self) -> usize {
        self.recommendations.iter().map(|r| r.savings_bytes).sum()
    }

    /// Format a human-readable summary.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Quantization Report: {} tensors, {} parameters, {}\n",
            self.total_tensors,
            format_count(self.total_parameters),
            format_bytes(self.total_bytes),
        ));
        out.push_str("\nDtype Breakdown:\n");
        for b in &self.dtype_breakdown {
            let pct = if self.total_bytes > 0 {
                (b.total_bytes as f64 / self.total_bytes as f64) * 100.0
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:>8}: {} tensors, {} params, {} ({:.1}%)\n",
                b.dtype.label(),
                b.tensor_count,
                format_count(b.total_parameters),
                format_bytes(b.total_bytes),
                pct,
            ));
        }
        if !self.recommendations.is_empty() {
            out.push_str("\nRecommendations:\n");
            for r in &self.recommendations {
                out.push_str(&format!(
                    "  Quantize {} tensors to {}: {} -> {} (save {})\n",
                    r.tensor_names.len(),
                    r.target_dtype.label(),
                    format_bytes(r.current_bytes),
                    format_bytes(r.projected_bytes),
                    format_bytes(r.savings_bytes),
                ));
            }
            out.push_str(&format!(
                "\nTotal potential savings: {}\n",
                format_bytes(self.total_savings_bytes()),
            ));
        } else {
            out.push_str("\nNo quantization recommendations (model is already compact).\n");
        }
        out
    }
}

impl fmt::Display for QuantizationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

// ---------------------------------------------------------------------------
// Core detection function
// ---------------------------------------------------------------------------

/// Analyze a safetensors weight file and produce a quantization report.
///
/// Reads the safetensors header to extract per-tensor dtype and shape
/// information. Does NOT load or convert tensor data — this is a
/// metadata-only analysis suitable for large models.
///
/// # Errors
///
/// Returns `ImportError::Io` if the file cannot be read or parsed.
pub fn detect_quantization(weights_path: &Path) -> Result<QuantizationReport, ImportError> {
    let data = std::fs::read(weights_path).map_err(|e| ImportError::Io {
        path: weights_path.display().to_string(),
        detail: e.to_string(),
    })?;
    detect_quantization_from_bytes(&data)
}

/// Analyze safetensors bytes (in-memory) and produce a quantization report.
///
/// This is the inner implementation used by both `detect_quantization` (file)
/// and tests (in-memory bytes).
pub fn detect_quantization_from_bytes(data: &[u8]) -> Result<QuantizationReport, ImportError> {
    let tensors_st = safetensors::SafeTensors::deserialize(data).map_err(|e| ImportError::Io {
        path: "<bytes>".to_string(),
        detail: format!("safetensors parse: {e}"),
    })?;

    let mut tensor_infos: Vec<TensorQuantInfo> = Vec::new();

    for (name, view) in tensors_st.tensors() {
        let dtype = DetectedDtype::from_safetensors(view.dtype());
        let shape: Vec<usize> = view.shape().to_vec();
        let num_elements: usize = shape.iter().copied().product();
        let size_bytes = view.data().len();

        tensor_infos.push(TensorQuantInfo {
            name,
            dtype,
            shape,
            num_elements,
            size_bytes,
        });
    }

    // Sort by name for deterministic output.
    tensor_infos.sort_by(|a, b| a.name.cmp(&b.name));

    build_report(tensor_infos)
}

/// Build the full report from collected tensor info.
fn build_report(tensors: Vec<TensorQuantInfo>) -> Result<QuantizationReport, ImportError> {
    // Aggregate by dtype.
    let mut by_dtype: BTreeMap<DetectedDtype, (usize, usize, usize)> = BTreeMap::new();
    for t in &tensors {
        let entry = by_dtype.entry(t.dtype).or_insert((0, 0, 0));
        entry.0 += 1; // tensor count
        entry.1 += t.num_elements; // parameter count
        entry.2 += t.size_bytes; // byte count
    }

    let dtype_breakdown: Vec<DtypeBreakdown> = by_dtype
        .into_iter()
        .map(
            |(dtype, (tensor_count, total_parameters, total_bytes))| DtypeBreakdown {
                dtype,
                tensor_count,
                total_parameters,
                total_bytes,
            },
        )
        .collect();

    let total_tensors = tensors.len();
    let total_parameters: usize = tensors.iter().map(|t| t.num_elements).sum();
    let total_bytes: usize = tensors.iter().map(|t| t.size_bytes).sum();

    // Generate recommendations.
    let recommendations = generate_recommendations(&tensors);

    Ok(QuantizationReport {
        tensors,
        dtype_breakdown,
        total_tensors,
        total_parameters,
        total_bytes,
        recommendations,
    })
}

/// Generate quantization recommendations.
///
/// Strategy:
/// - F32 float tensors with >= 1024 elements: recommend F16 (50% savings).
/// - F32 float tensors with >= 1024 elements: also recommend I8 (75% savings).
/// - F64 tensors: recommend F32 (50% savings).
/// - Small tensors (biases < 1024 elements) are left alone — savings are
///   negligible and quantization risk is higher for small buffers.
fn generate_recommendations(tensors: &[TensorQuantInfo]) -> Vec<QuantRecommendation> {
    let mut recommendations = Vec::new();

    // Collect F32 tensors large enough to benefit from quantization.
    let f32_large: Vec<&TensorQuantInfo> = tensors
        .iter()
        .filter(|t| t.dtype == DetectedDtype::F32 && t.num_elements >= 1024)
        .collect();

    if !f32_large.is_empty() {
        let current_bytes: usize = f32_large.iter().map(|t| t.size_bytes).sum();
        let names: Vec<String> = f32_large.iter().map(|t| t.name.clone()).collect();

        // F32 -> F16: 50% savings.
        let projected_f16 = current_bytes / 2;
        recommendations.push(QuantRecommendation {
            target_dtype: DetectedDtype::F16,
            tensor_names: names.clone(),
            current_bytes,
            projected_bytes: projected_f16,
            savings_bytes: current_bytes - projected_f16,
        });

        // F32 -> I8: 75% savings.
        let projected_i8 = current_bytes / 4;
        recommendations.push(QuantRecommendation {
            target_dtype: DetectedDtype::I8,
            tensor_names: names,
            current_bytes,
            projected_bytes: projected_i8,
            savings_bytes: current_bytes - projected_i8,
        });
    }

    // F64 tensors: recommend F32.
    let f64_tensors: Vec<&TensorQuantInfo> = tensors
        .iter()
        .filter(|t| t.dtype == DetectedDtype::F64 && t.num_elements >= 1)
        .collect();

    if !f64_tensors.is_empty() {
        let current_bytes: usize = f64_tensors.iter().map(|t| t.size_bytes).sum();
        let names: Vec<String> = f64_tensors.iter().map(|t| t.name.clone()).collect();
        let projected = current_bytes / 2;
        recommendations.push(QuantRecommendation {
            target_dtype: DetectedDtype::F32,
            tensor_names: names,
            current_bytes,
            projected_bytes: projected,
            savings_bytes: current_bytes - projected,
        });
    }

    recommendations
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn format_count(count: usize) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.2}K", count as f64 / 1_000.0)
    } else {
        format!("{count}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "quantization_tests.rs"]
mod tests;
