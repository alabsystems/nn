// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! ICB pre-compilation: build-time segment detection and codegen caching.
//!
//! `pre_compile_icb_segments()` runs at `CompiledModel` build time:
//! detects eligible contiguous step ranges, pre-compiles codegen outputs
//! (MSL + dispatch plans), and returns `IcbSegment` metadata.
//!
//! The actual `IndirectCommandBuffer` encoding is deferred to the first
//! forward pass, since activation buffers and the planned buffer are not
//! available at build time.
//!
//! Part of #3259 (D2).

use std::collections::HashMap;
use std::sync::Arc;

use nn_dsl::ir::ScalarType;
use nn_dsl::trace_compile::CompiledStep;
use nn_dsl::PrecisionContract;

use crate::msl_codegen_cache::CodegenOutput;

use super::analysis::{build_segment_starts, detect_icb_segments, IcbSegment};

/// Minimum contiguous ICB-eligible steps to form a segment.
/// Below this threshold, the ICB overhead exceeds per-dispatch savings.
const MIN_SEGMENT_LEN: usize = 4;

/// Pre-compile ICB segment metadata at model build time.
///
/// For each detected segment:
/// 1. Runs `codegen_for_kernel()` to cache MSL + dispatch plans per step.
/// 2. Stores codegen outputs on the `IcbSegment` for first-pass encoding.
///
/// Returns `(segments, segment_starts)` for storage on `CompiledModel`.
/// The actual `IndirectCommandBuffer` is lazily created on first forward pass.
///
/// Only called when `!autocast_active && !mixed_precision_active`
/// (ICB eligibility gate in `analyze_icb_eligibility`).
pub(crate) fn pre_compile_icb_segments(
    steps: &[CompiledStep],
    icb_eligible: &[bool],
    step_scalar_types: &[ScalarType],
    precision: Option<PrecisionContract>,
) -> (Vec<IcbSegment>, HashMap<usize, usize>) {
    let ranges = detect_icb_segments(icb_eligible, MIN_SEGMENT_LEN);
    if ranges.is_empty() {
        return (Vec::new(), HashMap::new());
    }

    let mut segments = Vec::with_capacity(ranges.len());
    for &(start, end) in &ranges {
        match build_one_segment(steps, start, end, step_scalar_types, precision) {
            Ok(seg) => segments.push(seg),
            Err(e) => {
                // Non-fatal: skip this segment, fall back to per-step dispatch.
                eprintln!("[nn-metal] ICB segment [{start}..={end}] skipped: {e}");
            }
        }
    }

    let starts = build_segment_starts(
        &segments
            .iter()
            .map(|s| (s.start, s.end))
            .collect::<Vec<_>>(),
    );
    (segments, starts)
}

/// Build one ICB segment by pre-compiling codegen outputs for each step.
fn build_one_segment(
    steps: &[CompiledStep],
    start: usize,
    end: usize,
    step_scalar_types: &[ScalarType],
    precision: Option<PrecisionContract>,
) -> Result<IcbSegment, String> {
    let mut step_codegen: Vec<Arc<CodegenOutput>> = Vec::with_capacity(end - start + 1);

    for (step_idx, step) in steps.iter().enumerate().take(end + 1).skip(start) {
        let kernel = match step {
            CompiledStep::Dispatch { kernel, .. } => kernel.def(),
            _ => {
                return Err(format!("step {step_idx} is not a Dispatch"));
            }
        };

        let dtype = step_scalar_types
            .get(step_idx)
            .copied()
            .unwrap_or(ScalarType::F32);
        // BF16 → F16 remap (Metal has no native bf16 compute).
        let effective_dtype = match dtype {
            ScalarType::BF16 => ScalarType::F16,
            other => other,
        };

        let codegen =
            crate::tensor_dispatch::codegen_for_kernel(kernel, effective_dtype, precision)
                .map_err(|e| format!("codegen for step {step_idx}: {e}"))?;

        step_codegen.push(codegen);
    }

    Ok(IcbSegment {
        start,
        end,
        step_codegen,
        external_bindings: Vec::new(),
        planned_bindings: Vec::new(),
    })
}
