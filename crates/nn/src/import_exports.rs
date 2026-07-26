// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Import re-exports for `nn::import`.
//!
//! Extracted from `lib.rs` for 450-line compliance.
//!
//! This module exposes exported-artifact import/build surfaces plus
//! `ConvertReport` provenance helpers and the current composition-bounds
//! classification enums. Callers can use
//! `ConvertReport::provenance_summary()` and
//! `ConvertReport::artifact_readiness_note()` to describe which exported
//! artifacts produced a report and what artifact kind it actually covers,
//! without implying raw PyTorch or ONNX intake support. When the current
//! composition-bounds verifier path runs, `VerificationCoverage` can also
//! record `ConvertCompositionMethod`, `ConvertSoundnessMode`, and
//! `ConvertProofStrength` for that run. It also re-exports the current
//! multi-segment imported-model / Metal-compile bridge for already segmented
//! exported-artifact bundles, not runtime orchestration.

// Convert / import
pub use nn_import::{
    check_composition_bounds, import_model, CompositionBoundsReport, ConvertError,
    EquivalenceProof, KaniSafetyReport, ParityReport,
};
#[cfg(feature = "import-metal")]
pub use nn_import::{convert, load_kokoro, ConvertResult};
// Report types and exported-artifact provenance/classification enums
// (available without Metal).
pub use nn_import::{
    ConvertArtifactKind, ConvertCompositionMethod, ConvertIntakePath, ConvertProofStrength,
    ConvertReport, ConvertSoundnessMode, FusionReport, PeepholeReport, VerificationCoverage,
};
// Builder API: `convert_build()` returns a ConvertBuilder for fluent configuration.
// Requires `import-metal` so `nn-import/metal` is enabled too.
#[cfg(feature = "import-metal")]
pub use nn_import::{
    convert_build, ConvertBuilder, ConvertResultWithReport, OptLevel, VerifyLevel,
};
// Graph building
pub use nn_import::{build_graph, build_weight_map, ImportedGraph};
// Multi-segment imported-model bridge for already-exported JSON segments.
pub use nn_import::{
    convert_multi_segment, convert_single_segment, MultiSegmentError, MultiSegmentModel,
};
// Multi-segment Metal compilation for already-imported or directly supplied
// exported-artifact segment bundles.
#[cfg(feature = "import-metal")]
pub use nn_import::{
    compile_multi_segment, convert_multi_segment_to_metal, CompiledMultiSegmentModel,
    MultiSegmentCompileError,
};
// Error
pub use nn_import::ImportError;
// Kokoro weights
pub use nn_import::{
    kokoro_name_mapping, map_pytorch_key, validate_kokoro_keys, validate_kokoro_safetensors,
};
// Op mapping
pub use nn_import::{map_node_to_trace_op, OpMapContext, ResolvedWeight};
// Parsing
pub use nn_import::{parse_exported_program, ExportedProgram, InputSpec, OutputSpec};
