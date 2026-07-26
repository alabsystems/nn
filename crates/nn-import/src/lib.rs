// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exported-artifact model import via `torch.export` JSON +
//! `safetensors` weights.
//!
//! # Overview
//!
//! `nn-import` consumes already-exported `torch.export` JSON graphs plus
//! `safetensors` weights, maps aten ops to nn `TraceOp` variants, and
//! produces import/build surfaces for downstream compilation. It does not
//! ingest raw PyTorch modules or raw ONNX graphs.
//!
//! For structured conversion reporting, [`ConvertReport`] exposes
//! [`ConvertReport::provenance_summary`] and
//! [`ConvertReport::artifact_readiness_note`] alongside
//! [`ConvertIntakePath`] and [`ConvertArtifactKind`]. Those helpers let callers
//! say which exported-artifact intake path produced a report and what artifact
//! kind it actually covers.
//!
//! # Usage
//!
//! ```rust,ignore
//! use nn_import::{parse_exported_program, build_graph, build_weight_map};
//!
//! let program = parse_exported_program(&json_bytes)?;
//! let weight_map = build_weight_map(
//!     &program.graph_module.signature.input_specs,
//!     &safetensors_weights,
//! );
//! let imported = build_graph(&program, &weight_map)?;
//! // imported.graph is a ComputationGraph ready for compile_trace_to_plan_with_fusion
//! ```
//!
//! ```rust,ignore
//! use nn_import::{convert_build, VerifyLevel};
//!
//! let result = convert_build(&graph_json, &weights, &cache)
//!     .verify(VerifyLevel::Bounds)
//!     .build()?;
//! let provenance = result.report.provenance_summary();
//! let readiness = result.report.artifact_readiness_note();
//! ```

pub(crate) mod convert;
pub mod convert_parity;
mod error;
pub(crate) mod graph_build;
#[cfg(kani)]
mod kani_convert;
#[cfg(kani)]
mod kani_convert_builder;
#[cfg(kani)]
mod kani_convert_builder_proofs;
#[cfg(kani)]
mod kani_convert_report_extended;
#[cfg(kani)]
mod kani_convert_report_proofs;
#[cfg(kani)]
mod kani_error_types_proofs;
#[cfg(kani)]
mod kani_graph_build_proofs;
#[cfg(kani)]
mod kani_import_safety;
#[cfg(kani)]
mod kani_import_wave11;
#[cfg(kani)]
mod kani_op_map_args_proofs;
#[cfg(kani)]
mod kani_op_map_dispatch;
#[cfg(kani)]
mod kani_op_map_expand_proofs;
#[cfg(kani)]
mod kani_op_map_impl;
#[cfg(kani)]
mod kani_op_map_impl_ext_extended;
#[cfg(kani)]
mod kani_op_map_impl_ext_proofs;
#[cfg(kani)]
mod kani_op_map_impl_kokoro_proofs;
#[cfg(kani)]
mod kani_op_map_impl_proofs;
#[cfg(kani)]
mod kani_op_map_proofs;
#[cfg(kani)]
mod kani_parse_accessors;
#[cfg(kani)]
mod kani_parse_extended;
#[cfg(kani)]
mod kani_parse_proofs;
#[cfg(kani)]
mod kani_weight_ops_proofs;
#[cfg(feature = "metal")]
mod kokoro_load;
pub mod kokoro_weights;
pub mod multi_segment;
pub(crate) mod op_map;
pub(crate) mod parse;
pub mod quantization;

// Exported-artifact builder surface. `ConvertBuilder::build()` returns a
// `ConvertReport` whose provenance helpers describe intake path and artifact
// kind without widening the accepted inputs beyond exported artifacts.
#[cfg(feature = "metal")]
pub use convert::builder::{
    convert as convert_build, ConvertBuilder, ConvertResultWithReport, OptLevel, VerifyLevel,
};
// Structured report surface for exported-artifact provenance, optimization,
// and verification coverage.
pub use convert::report::{
    ConvertArtifactKind, ConvertCompositionMethod, ConvertIntakePath, ConvertProofStrength,
    ConvertReport, ConvertSoundnessMode, FusionReport, PeepholeReport, VerificationCoverage,
};
pub use convert::{
    check_composition_bounds, import_model, CompositionBoundsReport, ConvertError,
    EquivalenceProof, KaniSafetyReport, ParityReport,
};
#[cfg(feature = "metal")]
pub use convert::{convert, ConvertResult};
pub use convert_parity::{
    compute_parity_metric, verify_parity, CheckStatus, ParityCheck, ParityLevel, ParityMetric,
    ParityThresholds, StructuralExpectation,
};
pub use error::ImportError;
pub use graph_build::{build_graph, build_weight_map, ImportedGraph};
#[cfg(feature = "metal")]
pub use kokoro_load::load_kokoro;
pub use kokoro_weights::{
    kokoro_name_mapping, map_pytorch_key, validate_kokoro_keys, validate_kokoro_safetensors,
};
#[cfg(feature = "metal")]
pub use multi_segment::{
    compile_multi_segment, convert_multi_segment_to_metal, CompiledMultiSegmentModel,
    MultiSegmentCompileError,
};
pub use multi_segment::{
    convert_multi_segment, convert_single_segment, MultiSegmentError, MultiSegmentModel,
};
pub use op_map::{map_node_to_trace_op, supported_ops, OpMapContext, ResolvedWeight};
pub use parse::{parse_exported_program, ExportedProgram, InputSpec, OutputSpec};
pub use quantization::{
    detect_quantization, detect_quantization_from_bytes, DetectedDtype, DtypeBreakdown,
    QuantRecommendation, QuantizationReport, TensorQuantInfo,
};

#[cfg(test)]
#[path = "gguf_expanded_tests.rs"]
mod gguf_expanded_tests;

#[cfg(test)]
#[path = "op_map_extended_tests.rs"]
mod op_map_extended_tests;

#[cfg(test)]
#[path = "quantization_extended_tests.rs"]
mod quantization_extended_tests;

#[cfg(test)]
#[path = "op_map_extended_tests2.rs"]
mod op_map_extended_tests2;

#[cfg(test)]
#[path = "graph_build_extended_tests.rs"]
mod graph_build_extended_tests;

#[cfg(test)]
#[path = "quantization_detection_tests.rs"]
mod quantization_detection_tests;

#[cfg(test)]
#[path = "aten_op_mapping_extended_tests.rs"]
mod aten_op_mapping_extended_tests;

#[cfg(test)]
#[path = "import_pipeline_extended_tests.rs"]
mod import_pipeline_extended_tests;

#[cfg(test)]
#[path = "multi_segment_extended_tests.rs"]
mod multi_segment_extended_tests;

#[cfg(test)]
#[path = "parse_graph_extended_tests.rs"]
mod parse_graph_extended_tests;
