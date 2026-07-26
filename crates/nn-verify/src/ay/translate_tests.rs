// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the ay KernelDef→AYProgram translator.
//!
//! Split into submodules to stay under 500 lines per file:
//! - `core`: basic translation, powi, abs, sin UF
//! - `encoding`: real_from_f64 encoding, constant param validation
//! - `encoding_adaptive`: adaptive denominator (#398) and param convention (#448) tests
//! - `verification`: SMT content verification, divisor guards, param counts
//! - `node_kinds`: structural encoding checks for Clamp, MinMax, Select, SumReduce
//! - `node_kinds_semantic`: round-trip semantic tests via ay direct execution (#415)
//! - `coverage_449`: Exp, Cos, Sub, Compare variant structural encoding tests (#449)
//! - `coverage_449_semantic`: semantic round-trip tests for Sub, Compare via ay execution (#449)
//! - `correctness`: end-to-end param binding and known-answer correctness tests (#451)

#[allow(unused_imports)]
use super::*;
#[allow(unused_imports)]
use nn_dsl::ir::{IRNode, NodeId, Param, ScalarType};

/// Build all-Variable bindings for a kernel (every param is symbolic).
fn all_variable(kernel: &KernelDef) -> Vec<ParamBinding> {
    vec![ParamBinding::Variable; kernel.params.len()]
}

use nn_dsl::test_kernels::identity_kernel;

/// Kernel: `fn add_one(x: f32) -> f32 { x + 1.0 }`
fn add_one_kernel() -> KernelDef {
    KernelDef::new(
        "add_one",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.0)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

/// Helper to build a kernel that computes `x.powi(exp)`.
fn powi_kernel(name: &str, exp: i32) -> KernelDef {
    KernelDef::new(
        name,
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(
                NodeId::new(1),
                IRNodeKind::Powi {
                    base: NodeId::new(0),
                    exp,
                },
            ),
        ],
        NodeId::new(1),
    )
}

/// Helper: build a kernel `fn f(x, y) -> f32 { x / y }`.
fn div_kernel() -> KernelDef {
    KernelDef::new(
        "div_xy",
        vec![
            Param::new("x", ScalarType::F32),
            Param::new("y", ScalarType::F32),
        ],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Param(1)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Div,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

#[path = "translate_tests_core.rs"]
mod core_tests;

#[path = "translate_tests_encoding.rs"]
mod encoding;

#[path = "translate_tests_encoding_adaptive.rs"]
mod encoding_adaptive;

#[path = "translate_tests_verification.rs"]
mod verification;

#[path = "translate_tests_node_kinds.rs"]
mod node_kinds;

#[path = "translate_tests_node_kinds_semantic.rs"]
mod node_kinds_semantic;

#[path = "translate_tests_coverage_449.rs"]
mod coverage_449;

#[path = "translate_tests_coverage_449_semantic.rs"]
mod coverage_449_semantic;

#[path = "translate_tests_correctness.rs"]
mod correctness;

#[path = "translate_tests_ground_folding.rs"]
mod ground_folding;
