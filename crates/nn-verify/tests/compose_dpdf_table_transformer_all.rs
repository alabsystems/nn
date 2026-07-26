// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated Table Transformer DETR pipeline composition verification tests.
//!
//! Verifies bounds propagation through the complete Table Transformer
//! architecture — a DETR-based model for table structure recognition
//! (Smock et al. 2022). This test binary consolidates all Table Transformer
//! subgraph tests into a single compilation unit to reduce link-time overhead.
//!
//! ## Module Summary
//!
//! **Backbone & Feature Extraction** (compose_dpdf_table_transformer.rs — 32 tests):
//! - ResNet basic block: Conv2d -> BN -> ReLU -> Conv2d -> BN + skip (IBP + CROWN)
//! - ResNet backbone level: Conv2d(stride=2) spatial downsampling
//! - ResNet 2-stage backbone: Cascaded stride-2 downsampling
//! - ResNet-18 full 4-stage backbone
//! - Backbone-to-transformer transition (reshape + linear projection)
//!
//! **Transformer Encoder** (compose_dpdf_table_transformer.rs):
//! - Self-attention -> LayerNorm -> FFN -> LayerNorm (CROWN)
//! - DETR encoder 2-layer and 4-layer stacks
//! - Position encoding + attention composition
//! - Encoder with final LayerNorm
//!
//! **Transformer Decoder** (compose_dpdf_table_transformer.rs):
//! - DETR decoder cross-attention: Object queries attend to encoder memory (CROWN)
//! - DETR decoder 2-layer and 4-layer stacks
//! - Encoder-decoder composition
//!
//! **Detection Heads** (compose_dpdf_table_transformer.rs):
//! - Classification head: Linear -> sigmoid (output in [0, 1])
//! - Box regression head: Linear -> sigmoid (normalized coordinates)
//! - DFL regression: Softmax -> weighted sum
//! - Multi-head detection (parallel sigmoid heads)
//!
//! **Table Structure** (compose_dpdf_table_structure.rs — 15 tests):
//! - Cell detection: classification, bbox regression, row/column separators
//! - Structure parsing: row/column count prediction, cell-to-row/column assignment
//! - Spanning cells: rowspan, colspan, span confidence
//! - Composed pipelines: detect-to-structure, full table parsing
//!
//! **Pipeline** (compose_dpdf_table_transformer_pipeline.rs — 10 tests):
//! - ResNet18 backbone feature extraction (IBP)
//! - Sinusoidal 2D position encoding (IBP)
//! - Transformer encoder self-attention (IBP + CROWN)
//! - DETR decoder cross-attention (IBP + CROWN)
//! - Object query init refinement (IBP)
//! - Table cell classification softmax (IBP)
//! - Table row/column regression sigmoid (IBP)
//! - Hungarian matching cost computation (IBP)
//! - Full encoder pipeline end-to-end (IBP + CROWN)
//! - Full decoder pipeline end-to-end (IBP + CROWN)
//!
//! **Deep Compositions** (compose_table_transformer_deep.rs — 12 tests):
//! - ResNet 2-stage with BN+ReLU (IBP + CROWN)
//! - Encoder layer tight-input CROWN analysis
//! - Decoder self+cross attention (IBP + CROWN)
//! - 3-layer encoder stack (IBP)
//! - Encoder + LayerNorm + classification (IBP + CROWN)
//! - Encoder + decoder + sigmoid heads (IBP)
//! - Widening analysis (1-layer vs 3-layer)
//! - Structure recognition heads (IBP + CROWN)
//! - ResNet backbone + input projection (IBP)
//! - Cross-attention with learned queries (IBP + CROWN)
//! - Verify-and-record entries
//!
//! Architecture references:
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - ResNet (He et al. 2016): Residual network backbone
//!
//! Part of #4237: Table Transformer DETR pipeline compose verification tests.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

/// Table Transformer DETR subgraph tests (32 tests): backbone, encoder, decoder,
/// detection heads, position encoding, DFL regression, full pipeline.
/// Part of #3883, #3915, #3945.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_transformer.rs"]
mod table_transformer;

/// Table structure recognition tests (15 tests): cell detection, row/column
/// parsing, spanning cells, composed detect-to-structure pipelines.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_structure.rs"]
mod table_structure;

/// Table Transformer DETR full pipeline tests (10 tests): production-scale
/// dimensions (d=256, heads=8), encoder/decoder end-to-end, Hungarian matching.
/// Part of #4177.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_transformer_pipeline.rs"]
mod table_transformer_pipeline;

/// Deep composition tests (12 tests): multi-layer stacks, cross-modal
/// compositions, widening analysis, verify-and-record entries.
/// Part of #4273.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_table_transformer_deep.rs"]
mod table_transformer_deep;

/// Full DETR pipeline tests (15+ tests): ResNet backbone, 6-layer encoder,
/// decoder cross-attention, object queries, FFN classification/bbox heads,
/// Hungarian matching, row/column detection, cell spanning, position encoding,
/// multi-scale features, layer norm, full encoder-decoder pipeline, NMS.
/// Part of #4237.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_detr_full.rs"]
mod table_detr_full;

/// Extended DETR pipeline tests (20 tests): CROWN variants for backbone,
/// encoder, bbox head, row/column detection, NMS confidence; ResNet residual
/// block; encoder+PE composition; monotone tightening; verification-recording
/// for backbone, classification head, bbox head, and full pipeline;
/// bipartite assignment score bounds; cell spanning prediction bounds;
/// multi-scale feature map bounds; decoder layer norm bounds.
/// Part of #4237.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_detr_extended.rs"]
mod table_detr_extended;
