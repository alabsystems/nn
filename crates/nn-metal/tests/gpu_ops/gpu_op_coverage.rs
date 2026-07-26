// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Coverage tests for GPU op enums (BinaryOp, UnaryOp, CompareOp, ReduceOp).
//!
//! These enums are `#[non_exhaustive]` in nn-core. Downstream dispatch code in
//! nn-metal uses `_ =>` catch-all arms that silently absorb new variants at
//! runtime. These tests maintain explicit variant lists with KNOWN_VARIANT_COUNT
//! assertions — when a new variant is added to any enum, the count fails,
//! forcing the developer to update both the dispatch code and this test.
//!
//! Part of #1516.

use nn_core::{BinaryOp, CompareOp, ReduceOp, UnaryOp};

// ---------------------------------------------------------------------------
// Known variant counts — bump when adding new enum variants.
// ---------------------------------------------------------------------------

const KNOWN_BINARY_OP_COUNT: usize = 7;
const KNOWN_UNARY_OP_COUNT: usize = 18;
const KNOWN_COMPARE_OP_COUNT: usize = 6;
const KNOWN_REDUCE_OP_COUNT: usize = 4;

// ---------------------------------------------------------------------------
// Tag functions — explicit match arms for every known variant.
// `_ => "UNKNOWN"` is required by #[non_exhaustive] but should never fire.
// ---------------------------------------------------------------------------

fn binary_op_tag(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "Add",
        BinaryOp::Sub => "Sub",
        BinaryOp::Mul => "Mul",
        BinaryOp::Div => "Div",
        BinaryOp::Maximum => "Maximum",
        BinaryOp::Minimum => "Minimum",
        BinaryOp::Atan2 => "Atan2",
        _ => "UNKNOWN",
    }
}

fn unary_op_tag(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Relu => "Relu",
        UnaryOp::Gelu => "Gelu",
        UnaryOp::Silu => "Silu",
        UnaryOp::Tanh => "Tanh",
        UnaryOp::Sigmoid => "Sigmoid",
        UnaryOp::Exp => "Exp",
        UnaryOp::Log => "Log",
        UnaryOp::Sqrt => "Sqrt",
        UnaryOp::Sqr => "Sqr",
        UnaryOp::Abs => "Abs",
        UnaryOp::Neg => "Neg",
        UnaryOp::Recip => "Recip",
        UnaryOp::Sin => "Sin",
        UnaryOp::Cos => "Cos",
        UnaryOp::GeluErf => "GeluErf",
        UnaryOp::Floor => "Floor",
        UnaryOp::Round => "Round",
        UnaryOp::Fract => "Fract",
        _ => "UNKNOWN",
    }
}

fn compare_op_tag(op: CompareOp) -> &'static str {
    match op {
        CompareOp::Eq => "Eq",
        CompareOp::Ne => "Ne",
        CompareOp::Ge => "Ge",
        CompareOp::Gt => "Gt",
        CompareOp::Lt => "Lt",
        CompareOp::Le => "Le",
        _ => "UNKNOWN",
    }
}

fn reduce_op_tag(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "Sum",
        ReduceOp::Mean => "Mean",
        ReduceOp::Max => "Max",
        ReduceOp::Min => "Min",
        _ => "UNKNOWN",
    }
}

// ---------------------------------------------------------------------------
// Coverage tests — enumerate all known variants, assert count + no UNKNOWN.
// ---------------------------------------------------------------------------

#[test]
fn coverage_binary_op_all_variants() {
    let variants = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
        BinaryOp::Atan2,
    ];
    let tags: Vec<_> = variants.iter().map(|v| binary_op_tag(*v)).collect();
    assert!(
        !tags.contains(&"UNKNOWN"),
        "unknown BinaryOp variant: {tags:?}"
    );
    assert_eq!(
        tags.len(),
        KNOWN_BINARY_OP_COUNT,
        "BinaryOp variant count mismatch: expected {KNOWN_BINARY_OP_COUNT}, got {}.\n\
         Bump KNOWN_BINARY_OP_COUNT and add dispatch arms when adding BinaryOp variants.",
        tags.len(),
    );
}

#[test]
fn coverage_unary_op_all_variants() {
    let variants = [
        UnaryOp::Relu,
        UnaryOp::Gelu,
        UnaryOp::Silu,
        UnaryOp::Tanh,
        UnaryOp::Sigmoid,
        UnaryOp::Exp,
        UnaryOp::Log,
        UnaryOp::Sqrt,
        UnaryOp::Sqr,
        UnaryOp::Abs,
        UnaryOp::Neg,
        UnaryOp::Recip,
        UnaryOp::Sin,
        UnaryOp::Cos,
        UnaryOp::GeluErf,
        UnaryOp::Floor,
        UnaryOp::Round,
        UnaryOp::Fract,
    ];
    let tags: Vec<_> = variants.iter().map(|v| unary_op_tag(*v)).collect();
    assert!(
        !tags.contains(&"UNKNOWN"),
        "unknown UnaryOp variant: {tags:?}"
    );
    assert_eq!(
        tags.len(),
        KNOWN_UNARY_OP_COUNT,
        "UnaryOp variant count mismatch: expected {KNOWN_UNARY_OP_COUNT}, got {}.\n\
         Bump KNOWN_UNARY_OP_COUNT and add dispatch arms when adding UnaryOp variants.",
        tags.len(),
    );
}

#[test]
fn coverage_compare_op_all_variants() {
    let variants = [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Ge,
        CompareOp::Gt,
        CompareOp::Lt,
        CompareOp::Le,
    ];
    let tags: Vec<_> = variants.iter().map(|v| compare_op_tag(*v)).collect();
    assert!(
        !tags.contains(&"UNKNOWN"),
        "unknown CompareOp variant: {tags:?}"
    );
    assert_eq!(
        tags.len(),
        KNOWN_COMPARE_OP_COUNT,
        "CompareOp variant count mismatch: expected {KNOWN_COMPARE_OP_COUNT}, got {}.\n\
         Bump KNOWN_COMPARE_OP_COUNT and add dispatch arms when adding CompareOp variants.",
        tags.len(),
    );
}

#[test]
fn coverage_reduce_op_all_variants() {
    let variants = [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min];
    let tags: Vec<_> = variants.iter().map(|v| reduce_op_tag(*v)).collect();
    assert!(
        !tags.contains(&"UNKNOWN"),
        "unknown ReduceOp variant: {tags:?}"
    );
    assert_eq!(
        tags.len(),
        KNOWN_REDUCE_OP_COUNT,
        "ReduceOp variant count mismatch: expected {KNOWN_REDUCE_OP_COUNT}, got {}.\n\
         Bump KNOWN_REDUCE_OP_COUNT and add dispatch arms when adding ReduceOp variants.",
        tags.len(),
    );
}
