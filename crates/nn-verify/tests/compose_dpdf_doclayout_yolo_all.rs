// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated DocLayout-YOLO multi-scale detection compose verification tests.
//!
//! Wires the multi-scale detection pipeline helper module into a test binary.
//! Tests verify IBP and CROWN bound propagation through multi-scale detection
//! subgraphs: backbone extraction (P3/P4/P5), FPN + PAN neck fusion,
//! per-scale detection heads, and end-to-end pipelines.
//!
//! Part of #4234: DocLayout-YOLO multi-scale detection compose tests.

#![allow(clippy::duplicate_mod)]

mod common;

/// DocLayout-YOLO multi-scale detection pipeline compose tests (16 tests).
/// Backbone multi-scale extraction, FPN + PAN neck fusion, per-scale detection
/// heads (P3 small-object, P5 large-object), dual-head DFL + sigmoid, end-to-end
/// pipelines, monotone tightening, and widening analysis across scales.
/// Part of #4234.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_doclayout_yolo_multiscale.rs"]
mod doclayout_yolo_multiscale;
