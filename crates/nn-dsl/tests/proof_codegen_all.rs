// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code, unreachable_pub)]

//! Consolidated proof and codegen tests: proof coverage, codegen coverage,
//! Kani precision properties, RoPE bounds, fusion equivalence, and error unification.

#[path = "proof_codegen/adain_fusion_equivalence.rs"]
mod adain_fusion_equivalence;
#[path = "proof_codegen/codegen_coverage.rs"]
mod codegen_coverage;
#[path = "proof_codegen/kani_precision_properties.rs"]
mod kani_precision_properties;
#[path = "proof_codegen/lower_sum_reduce.rs"]
mod lower_sum_reduce;
#[path = "proof_codegen/nn_dsl_error_unification.rs"]
mod nn_dsl_error_unification;
#[path = "proof_codegen/proof_coverage.rs"]
mod proof_coverage;
#[path = "proof_codegen/proof_coverage_codegen.rs"]
mod proof_coverage_codegen;
#[path = "proof_codegen/rope_bounds_soundness.rs"]
mod rope_bounds_soundness;
#[path = "proof_codegen/snake_metal_diff.rs"]
mod snake_metal_diff;
