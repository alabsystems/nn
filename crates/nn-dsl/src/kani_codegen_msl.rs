// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kani harnesses for codegen_msl structural invariants.
//!
//! These avoid heavy string-emission paths that are currently solver-hostile
//! under Kani (CBMC Exit 241), while still proving core codegen contracts.

use crate::codegen_msl::{
    compare_op, msl_fn, msl_type, wrapper_out_buffer_index, wrapper_total_buffer_index, MslUnaryOp,
    MSL_PRELUDE,
};
use crate::ir::{CompareOpKind, ScalarType, UnaryFnKind};
use crate::precision::PrecisionTier;

fn contains_ascii(haystack: &'static str, needle: &'static str) -> bool {
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() || n.len() > h.len() {
        return n.is_empty();
    }

    let mut start = 0usize;
    while start + n.len() <= h.len() {
        let mut i = 0usize;
        let mut matched = true;
        while i < n.len() {
            if h[start + i] != n[i] {
                matched = false;
                break;
            }
            i += 1;
        }
        if matched {
            return true;
        }
        start += 1;
    }
    false
}

/// Byte-by-byte string equality that avoids `memcmp` (which causes CBMC
/// loop unwinding timeouts). Both arguments must be `&'static str` for
/// Kani's bounded analysis.
fn str_eq(a: &'static str, b: &'static str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0usize;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

fn is_compare_symbol(symbol: &'static str) -> bool {
    str_eq(symbol, "==")
        || str_eq(symbol, "!=")
        || str_eq(symbol, "<")
        || str_eq(symbol, "<=")
        || str_eq(symbol, ">")
        || str_eq(symbol, ">=")
}

fn is_msl_scalar_type(mapped: &'static str) -> bool {
    str_eq(mapped, "float") || str_eq(mapped, "half")
}

fn any_compare_op() -> CompareOpKind {
    let raw: u8 = kani::any();
    kani::assume(raw < 6);
    match raw {
        0 => CompareOpKind::Eq,
        1 => CompareOpKind::Ne,
        2 => CompareOpKind::Lt,
        3 => CompareOpKind::Le,
        4 => CompareOpKind::Gt,
        5 => CompareOpKind::Ge,
        _ => unreachable!("assume(raw < 6) guarantees covered cases"),
    }
}

fn any_scalar_type() -> ScalarType {
    let raw: u8 = kani::any();
    kani::assume(raw < 3);
    match raw {
        0 => ScalarType::F32,
        1 => ScalarType::F16,
        2 => ScalarType::BF16,
        _ => unreachable!("assume(raw < 3) guarantees covered cases"),
    }
}

fn any_unary_fn() -> UnaryFnKind {
    let raw: u8 = kani::any();
    kani::assume(raw < 13);
    match raw {
        0 => UnaryFnKind::Sin,
        1 => UnaryFnKind::Cos,
        2 => UnaryFnKind::Sqrt,
        3 => UnaryFnKind::Rsqrt,
        4 => UnaryFnKind::Exp,
        5 => UnaryFnKind::Abs,
        6 => UnaryFnKind::Recip,
        7 => UnaryFnKind::Tanh,
        8 => UnaryFnKind::Log,
        9 => UnaryFnKind::Floor,
        10 => UnaryFnKind::Round,
        11 => UnaryFnKind::Fract,
        12 => UnaryFnKind::Neg,
        _ => unreachable!("assume(raw < 13) guarantees covered cases"),
    }
}

/// Proves the generated file prelude always includes required Metal imports.
#[kani::unwind(64)]
#[kani::proof]
fn kani_codegen_msl_includes_metal_prelude() {
    let probe_include: bool = kani::any();
    let required = if probe_include {
        "#include <metal_stdlib>"
    } else {
        "using namespace metal;"
    };
    assert!(contains_ascii(MSL_PRELUDE, required));

    let i: u8 = kani::any();
    kani::assume(usize::from(i) < MSL_PRELUDE.len());
    assert!(MSL_PRELUDE.as_bytes()[usize::from(i)] != 0);
}

/// Proves wrapper buffer index layout remains consistent for bounded arities.
#[kani::unwind(64)]
#[kani::proof]
fn kani_codegen_msl_wrapper_buffer_indices_consistent() {
    let arity: u8 = kani::any();
    kani::assume(arity <= 64);
    let arity = usize::from(arity);

    let out_idx = wrapper_out_buffer_index(arity);
    let total_idx = wrapper_total_buffer_index(arity);
    assert!(out_idx == arity);
    assert!(total_idx == out_idx + 1);
}

/// Proves compare/select codegen uses canonical MSL comparison symbols.
#[kani::unwind(64)]
#[kani::proof]
fn kani_codegen_msl_compare_select_structure() {
    let op = any_compare_op();
    let symbol = compare_op(op);

    assert!(is_compare_symbol(symbol));
    match op {
        CompareOpKind::Eq => assert!(str_eq(symbol, "==")),
        CompareOpKind::Ne => assert!(str_eq(symbol, "!=")),
        CompareOpKind::Lt => assert!(str_eq(symbol, "<")),
        CompareOpKind::Le => assert!(str_eq(symbol, "<=")),
        CompareOpKind::Gt => assert!(str_eq(symbol, ">")),
        CompareOpKind::Ge => assert!(str_eq(symbol, ">=")),
    }
}

/// Proves scalar type and intrinsic mappings remain stable for f16.
/// Unwind(64) bounds the string comparison loops in contains_ascii/str_eq
/// for MSL intrinsic name strings (max ~30 chars). (#767 AC3)
#[kani::unwind(8)]
#[kani::proof]
#[kani::unwind(64)]
fn kani_codegen_msl_f16_signature_uses_half() {
    let ty = any_scalar_type();
    let mapped = msl_type(ty);
    assert!(is_msl_scalar_type(mapped));
    match ty {
        ScalarType::F16 | ScalarType::BF16 => assert!(str_eq(mapped, "half")),
        ScalarType::F32 => assert!(str_eq(mapped, "float")),
    }

    let op = any_unary_fn();
    let strict = msl_fn(op, PrecisionTier::Strict);
    let relaxed = msl_fn(op, PrecisionTier::Relaxed);

    match (strict, relaxed) {
        (MslUnaryOp::Reciprocal, MslUnaryOp::Reciprocal) => {
            // Recip returns Reciprocal for all tiers — no named intrinsic.
            assert!(matches!(op, UnaryFnKind::Recip));
        }
        (MslUnaryOp::Negation, MslUnaryOp::Negation) => {
            assert!(matches!(op, UnaryFnKind::Neg));
        }
        (MslUnaryOp::Named(s), MslUnaryOp::Named(r)) => {
            assert!(!matches!(op, UnaryFnKind::Recip | UnaryFnKind::Neg));
            assert!(contains_ascii(s, "metal::"));
            assert!(contains_ascii(r, "metal::"));
            // Tier-independent ops: same intrinsic for Strict and Relaxed.
            if matches!(
                op,
                UnaryFnKind::Abs | UnaryFnKind::Floor | UnaryFnKind::Round | UnaryFnKind::Fract
            ) {
                assert!(str_eq(s, r));
                assert!(!contains_ascii(s, "precise::"));
            } else {
                // Tier-dependent: Strict uses metal::precise::*, Relaxed uses metal::*.
                assert!(contains_ascii(s, "precise::"));
                assert!(!contains_ascii(r, "precise::"));
            }
        }
        // Mixed Named/Reciprocal across tiers is structurally impossible.
        _ => unreachable!("msl_fn returns consistent variants across tiers"),
    }
}
