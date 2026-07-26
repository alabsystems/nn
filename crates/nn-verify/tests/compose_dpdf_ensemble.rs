// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated compose verification tests for the dpdf 7-model ensemble
//! pipeline bounds.
//!
//! Wires 4 helper modules into a single test binary:
//!
//! - **ensemble** (15 tests): Per-model subgraph bounds (DocLayout-YOLO,
//!   Table Transformer, FireRed-OCR, Qwen3-VL, Granite-Docling, GLM-OCR,
//!   PaddleOCR) + cross-model cascades + multi-model pipelines + ensemble
//!   confidence aggregation + monotone tightening + 7-model dispatch routing.
//!
//! - **ensemble_7model** (14 tests): Pipeline-level composition — stage
//!   composition, parallel dispatch, weighted aggregation, confidence-weighted
//!   selection, fallback chains (2-model, 3-model), full page-to-structured-data
//!   pipeline, multi-page batch processing, detection-to-multi-OCR fan-out,
//!   OCR-to-language aggregation, table+OCR merge, ensemble monotone, 7-model
//!   confidence ensemble, multi-page attention aggregation.
//!
//! - **ensemble_7model_extended** (7 tests): Per-model standalone bounds —
//!   each of the 7 models verified individually (DocLayout-YOLO, Table
//!   Transformer, Granite-Docling, PaddleOCR-VL, FireRed-OCR, GLM-OCR,
//!   Qwen3-VL).
//!
//! - **ensemble_7model_interactions** (7 tests): Cross-model interaction
//!   patterns — feature fusion, majority voting, vision-to-LM cascade,
//!   ensemble monotone CROWN, E2E 7-head pipeline, hierarchical routing,
//!   confidence calibration.
//!
//! - **ensemble_pipeline** (20 tests): Full document-type-specific pipeline
//!   tests — individual model subnetworks, pairwise composition, full
//!   sequential + parallel pipeline, aggregation bounds, and document-type
//!   specialization (text-heavy, table-heavy, figure-heavy).
//!
//! Total: 63 tests covering individual model bounds, pairwise cascades,
//! multi-model pipeline composition, parallel dispatch, fallback chains,
//! ensemble aggregation, monotone tightening, and 7-model dispatch routing.
//!
//! Part of #4243: Compose tests for dpdf 7-model ensemble pipeline bounds.

#![allow(clippy::duplicate_mod)]

mod common;

/// Per-model subgraph bounds + cross-model cascades + multi-model pipelines +
/// confidence aggregation + monotone tightening + 7-model dispatch (15 tests).
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_ensemble.rs"]
mod ensemble;

/// Pipeline-level composition: stage composition, parallel dispatch, fallback
/// chains, full page-to-data pipeline, multi-page processing, fan-out,
/// aggregation, monotone, 7-model confidence ensemble, page attention (14 tests).
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_7model_ensemble.rs"]
mod ensemble_7model;

/// Per-model standalone bounds: each of the 7 models verified individually —
/// DocLayout-YOLO, Table Transformer, Granite-Docling, PaddleOCR-VL,
/// FireRed-OCR, GLM-OCR, Qwen3-VL (7 tests).
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_7model_ensemble_extended.rs"]
mod ensemble_7model_extended;

/// Cross-model interaction patterns: feature fusion, majority voting,
/// vision-to-LM cascade, ensemble monotone CROWN, E2E 7-head pipeline,
/// hierarchical routing, confidence calibration (7 tests).
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_7model_ensemble_interactions.rs"]
mod ensemble_7model_interactions;

/// Full document-type-specific pipeline: individual model subnetworks,
/// pairwise composition, full sequential + parallel pipeline, aggregation
/// bounds, document-type specialization (20 tests).
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_ensemble_pipeline.rs"]
mod ensemble_pipeline;
