// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the compiled Kokoro component registry (#2923).

use super::*;

/// Every kernel's expected_stages must be a subset of its kokoro_stages.
///
/// If this test fails, a kernel designed for a stage is not wired into that
/// stage's compiled segment. Either wire it or update expected_stages.
///
/// This would have caught #2910 (forward_leaky_relu unwired for weeks).
#[test]
fn test_no_unwired_fused_kernels() {
    for entry in KERNEL_REGISTRY {
        for &expected in entry.expected_stages {
            assert!(
                entry.kokoro_stages.contains(&expected),
                "Kernel '{}' is expected in stage '{}' but not wired. \
                 File issue or update expected_stages in compiled_kokoro_registry.rs. \
                 Dispatch file: {}",
                entry.kind,
                expected,
                entry.dispatch_file,
            );
        }
    }
}

/// Registry must document all NativeOpKind variants.
///
/// If a new variant is added to NativeOpKind without a registry entry,
/// this test fails. Bump NATIVE_OP_VARIANT_COUNT and add the entry.
#[test]
fn test_registry_covers_all_native_op_variants() {
    assert_eq!(
        KERNEL_REGISTRY.len(),
        NATIVE_OP_VARIANT_COUNT,
        "KERNEL_REGISTRY has {} entries but NATIVE_OP_VARIANT_COUNT is {}. \
         Add the new NativeOpKind variant to the registry.",
        KERNEL_REGISTRY.len(),
        NATIVE_OP_VARIANT_COUNT,
    );
}

/// Sync point count must match pipeline documentation.
///
/// compiled_kokoro_pipeline.rs documents N sync points. If the count
/// changes, update both the pipeline docs and EXPECTED_SYNC_POINTS.
#[test]
fn test_sync_point_count_matches_pipeline_docs() {
    assert_eq!(
        SYNC_POINT_REGISTRY.len(),
        EXPECTED_SYNC_POINTS,
        "Registry has {} sync points but pipeline expects {}. \
         Update EXPECTED_SYNC_POINTS or SYNC_POINT_REGISTRY.",
        SYNC_POINT_REGISTRY.len(),
        EXPECTED_SYNC_POINTS,
    );
}

/// Every CPU bridge must have a documented reason.
#[test]
fn test_no_undocumented_cpu_bridges() {
    for bridge in CPU_BRIDGE_REGISTRY {
        assert!(
            !bridge.reason.is_empty(),
            "CPU bridge '{}' at {} has no documented reason",
            bridge.name,
            bridge.file_line,
        );
        assert!(
            !bridge.file_line.is_empty(),
            "CPU bridge '{}' has no file:line reference",
            bridge.name,
        );
    }
}

/// Segment native_ops must reference kernels that exist in the registry.
#[test]
fn test_segment_native_ops_are_registered() {
    let kernel_kinds: Vec<&str> = KERNEL_REGISTRY.iter().map(|k| k.kind).collect();
    for seg in SEGMENT_REGISTRY {
        for &op in seg.native_ops {
            assert!(
                kernel_kinds.contains(&op),
                "Segment '{}' references NativeOp '{}' which is not in KERNEL_REGISTRY",
                seg.name,
                op,
            );
        }
    }
}

/// Kernel kokoro_stages must reference segments that exist in the registry.
#[test]
fn test_kernel_stages_reference_valid_segments() {
    // Map stage names to their segment registry entries.
    // Stage names in kernel entries use short names; map them to segment steps.
    let valid_stages = [
        "plbert",
        "text_encoder",
        "prosody",
        "f0_energy",
        "generator",
        "harmonic_source",
    ];
    for entry in KERNEL_REGISTRY {
        for &stage in entry.kokoro_stages {
            assert!(
                valid_stages.contains(&stage),
                "Kernel '{}' references unknown stage '{}'. \
                 Valid stages: {:?}",
                entry.kind,
                stage,
                valid_stages,
            );
        }
    }
}

/// No eliminable sync points should lack a replacement issue.
#[test]
fn test_eliminable_sync_points_have_issues() {
    for sp in SYNC_POINT_REGISTRY {
        if sp.eliminable {
            assert!(
                sp.replacement_issue.is_some(),
                "Eliminable sync point '{}' has no replacement_issue tracking its removal",
                sp.name,
            );
        }
    }
}
