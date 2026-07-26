// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

use nn_dsl::{IRNodeKind, LowerError, Lowerer};

fn parse_and_lower(src: &str) -> nn_dsl::KernelDef {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse source");
    Lowerer::lower_fn(&func).expect("lower to IR")
}

fn parse_and_expect_err(src: &str) -> LowerError {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse source");
    Lowerer::lower_fn(&func).expect_err("lowering should fail")
}

#[test]
fn test_lower_sum_reduce_intrinsic() {
    let kernel = parse_and_lower(
        "fn sum3(a: f32, b: f32, c: f32) -> f32 {
            nn_dsl::sum_reduce([a, b, c])
        }",
    );
    assert_eq!(kernel.name, "sum3");
    assert!(matches!(
        kernel.nodes[kernel.output.index()].kind,
        IRNodeKind::SumReduce { .. }
    ));
    if let IRNodeKind::SumReduce { ref inputs } = kernel.nodes[kernel.output.index()].kind {
        assert_eq!(inputs.len(), 3);
    }
}

#[test]
fn test_sum_reduce_rejects_non_array_arg() {
    let err = parse_and_expect_err("fn bad(a: f32, b: f32) -> f32 { nn_dsl::sum_reduce(a + b) }");
    assert!(
        matches!(err, LowerError::UnsupportedExpr(ref msg) if msg.contains("array literal")),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sum_reduce_rejects_empty_array() {
    let err = parse_and_expect_err("fn bad() -> f32 { nn_dsl::sum_reduce([]) }");
    assert!(
        matches!(err, LowerError::UnsupportedExpr(ref msg) if msg.contains("at least one element")),
        "unexpected error: {err}"
    );
}

#[test]
fn test_sum_reduce_wrong_arg_count() {
    let err =
        parse_and_expect_err("fn bad(a: f32, b: f32) -> f32 { nn_dsl::sum_reduce([a], [b]) }");
    assert!(matches!(
        err,
        LowerError::WrongArgCount {
            method,
            expected: 1,
            got: 2
        } if method == "sum_reduce"
    ));
}
