// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Broadcast alignment inference and validation for the tensor IR.
//!
//! Extracted from `tensor_ir.rs` to keep that module under 500 lines.
//! This module owns the two broadcast functions:
//!
//! - [`infer_broadcast_alignment`]: infer the unique valid alignment from shapes.
//! - [`validate_broadcast_alignment`]: check that a declared alignment is compatible.
//!
//! Both use [`dims_compatible_at_offset`] internally to avoid duplicated
//! per-dimension compatibility checks.

use super::{BroadcastAlignment, TensorIRError};

/// Check whether `input_shape` is broadcast-compatible with `target_shape`
/// when input dims are placed at the given offset within the target dims.
///
/// Each input dimension must either equal 1 (broadcast) or equal the
/// corresponding target dimension at `offset + i`.
fn dims_compatible_at_offset(input_shape: &[usize], target_shape: &[usize], offset: usize) -> bool {
    input_shape
        .iter()
        .enumerate()
        .all(|(i, &input_dim)| input_dim == 1 || input_dim == target_shape[offset + i])
}

/// Validate that a declared broadcast alignment is compatible with the shapes.
///
/// Checks that the declared alignment (Left or Right) produces valid dim
/// matching for the given input and target shapes. This allows callers to
/// specify alignment explicitly without requiring the inference function.
pub(crate) fn validate_broadcast_alignment(
    input_shape: &[usize],
    target_shape: &[usize],
    alignment: BroadcastAlignment,
) -> Result<(), TensorIRError> {
    if input_shape.len() > target_shape.len() {
        return Err(TensorIRError::IncompatibleBroadcast {
            input: input_shape.to_vec(),
            target: target_shape.to_vec(),
        });
    }

    let offset = match alignment {
        BroadcastAlignment::Left => 0,
        BroadcastAlignment::Right => target_shape.len() - input_shape.len(),
    };

    if dims_compatible_at_offset(input_shape, target_shape, offset) {
        Ok(())
    } else {
        Err(TensorIRError::IncompatibleBroadcast {
            input: input_shape.to_vec(),
            target: target_shape.to_vec(),
        })
    }
}

/// Infer broadcast alignment from input and target shapes.
///
/// Returns the unique valid alignment, or an error if:
/// - The shapes are incompatible (neither alignment works).
/// - Both left and right alignment are valid and produce different mappings
///   (ambiguous — the caller must specify alignment explicitly).
///
/// For same-rank broadcasts, alignment is always `Left` (offset = 0 either way).
pub fn infer_broadcast_alignment(
    input_shape: &[usize],
    target_shape: &[usize],
) -> Result<BroadcastAlignment, TensorIRError> {
    if input_shape.len() > target_shape.len() {
        return Err(TensorIRError::IncompatibleBroadcast {
            input: input_shape.to_vec(),
            target: target_shape.to_vec(),
        });
    }

    // Same rank → offset is always 0, left and right are identical.
    if input_shape.len() == target_shape.len() {
        return if dims_compatible_at_offset(input_shape, target_shape, 0) {
            Ok(BroadcastAlignment::Left)
        } else {
            Err(TensorIRError::IncompatibleBroadcast {
                input: input_shape.to_vec(),
                target: target_shape.to_vec(),
            })
        };
    }

    // Try left-aligned (input aligns to prefix of target)
    let left_ok = dims_compatible_at_offset(input_shape, target_shape, 0);

    // Try right-aligned (input aligns to suffix of target, NumPy-style)
    let offset = target_shape.len() - input_shape.len();
    let right_ok = dims_compatible_at_offset(input_shape, target_shape, offset);

    match (left_ok, right_ok) {
        (true, true) => {
            // Both alignments match. If all input dims are 1, both produce
            // identical coordinate mappings — default to Left (not ambiguous).
            // Only flag ambiguity when at least one non-1 dim exists, making
            // the two alignments produce genuinely different index mappings.
            let all_ones = input_shape.iter().all(|&d| d == 1);
            if all_ones {
                Ok(BroadcastAlignment::Left)
            } else {
                Err(TensorIRError::AmbiguousBroadcast {
                    input: input_shape.to_vec(),
                    target: target_shape.to_vec(),
                })
            }
        }
        (true, false) => Ok(BroadcastAlignment::Left),
        (false, true) => Ok(BroadcastAlignment::Right),
        (false, false) => Err(TensorIRError::IncompatibleBroadcast {
            input: input_shape.to_vec(),
            target: target_shape.to_vec(),
        }),
    }
}
