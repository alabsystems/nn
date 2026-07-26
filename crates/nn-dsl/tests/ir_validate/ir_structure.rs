// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for IR lowering structure — verifies exact node counts
//! and output node kinds to catch regressions in the lowering pass.

use nn_dsl::{BinOpKind, IRNodeKind, LowerError, Lowerer};

fn parse_and_lower(src: &str) -> nn_dsl::KernelDef {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse source");
    Lowerer::lower_fn(&func).expect("lower to IR")
}

fn try_parse_and_lower(src: &str) -> Result<nn_dsl::KernelDef, LowerError> {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse source");
    Lowerer::lower_fn(&func)
}

#[test]
fn test_snake_kernel_exact_node_count() {
    let kernel = parse_and_lower(
        "fn snake(x: f32, alpha: f32) -> f32 {
            x + (1.0 / alpha) * (alpha * x).sin().powi(2)
        }",
    );
    // snake: x + (1/alpha) * sin(alpha*x)^2
    // Expected nodes: param(x), param(alpha), lit(1.0), div, mul, sin, powi, mul, add = 9
    assert_eq!(
        kernel.nodes.len(),
        9,
        "snake kernel should have exactly 9 IR nodes"
    );
    // Output node must be Add (the top-level x + ...)
    assert!(
        matches!(
            kernel.nodes[kernel.output.index()].kind,
            IRNodeKind::BinOp {
                op: BinOpKind::Add,
                ..
            }
        ),
        "snake output node must be Add, got: {:?}",
        kernel.nodes[kernel.output.index()].kind
    );
}

#[test]
fn test_let_binding_shares_nodes() {
    let kernel = parse_and_lower(
        "fn snake_v2(x: f32, alpha: f32) -> f32 {
            let sin_val = (alpha * x).sin();
            x + (1.0 / alpha) * sin_val * sin_val
        }",
    );
    // let sin_val reuses the same NodeId for both occurrences
    // Expected: param(x), param(alpha), mul, sin, lit(1.0), div, mul, mul, add = 9
    assert_eq!(
        kernel.nodes.len(),
        9,
        "let-binding snake should have 9 nodes (sin shared via let)"
    );
}

#[test]
fn test_negation_lowers_to_zero_minus_x() {
    let kernel = parse_and_lower(
        "fn neg(x: f32) -> f32 {
            -x
        }",
    );
    // Negation lowers to 0.0 - x: param(x), lit(0.0), sub
    assert_eq!(kernel.nodes.len(), 3, "negation: param, lit(0), sub");
    assert!(
        matches!(
            kernel.nodes[kernel.output.index()].kind,
            IRNodeKind::BinOp {
                op: BinOpKind::Sub,
                ..
            }
        ),
        "negation output must be Sub"
    );
}

#[test]
fn test_if_else_with_branch_locals_does_not_leak_bindings() {
    let kernel = parse_and_lower(
        "fn choose(x: f32, y: f32) -> f32 {
            if x > y {
                let shifted = x + 1.0;
                shifted
            } else {
                let shifted = y + 2.0;
                shifted
            }
        }",
    );

    assert!(
        matches!(
            kernel.nodes[kernel.output.index()].kind,
            IRNodeKind::Select { .. }
        ),
        "if/else output should be Select, got: {:?}",
        kernel.nodes[kernel.output.index()].kind
    );

    kernel
        .validate()
        .expect("if/else with branch-local lets should produce valid IR");
}

#[test]
fn test_deeply_nested_expr_rejected() {
    // Build x + 1.0 + 1.0 + ... with 140 additions (no explicit parens).
    // Syn parses left-to-right as ((x + 1.0) + 1.0) + ... creating depth 140.
    // This exceeds the MAX_EXPR_DEPTH=128 guard in our lowerer.
    let mut body = "x".to_string();
    for _ in 0..140 {
        body = format!("{body} + 1.0");
    }
    let src = format!("fn deep(x: f32) -> f32 {{ {body} }}");

    let err = try_parse_and_lower(&src).expect_err("deeply nested expr should fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("nesting depth"),
        "expected ExprTooDeep error, got: {msg}"
    );
}

#[test]
fn test_moderate_nesting_accepted() {
    // 100 additions: depth=100, well within the MAX_EXPR_DEPTH=128 limit.
    let mut body = "x".to_string();
    for _ in 0..100 {
        body = format!("{body} + 1.0");
    }
    let src = format!("fn moderate(x: f32) -> f32 {{ {body} }}");
    let kernel = parse_and_lower(&src);
    kernel.validate().expect("moderate nesting should be valid");
}

#[test]
fn test_validate_chain_node_count_formula() {
    // Verify the node count formula for chain kernels of varying sizes.
    // Node count: 1 param + chain_len literals + chain_len adds = 1 + 2*chain_len
    //
    // Note: validate() is O(n) by code inspection — single pass over nodes
    // plus single-pass type inference. This test checks correctness, not timing.
    for &chain_len in &[10usize, 50, 100] {
        let mut body = "x".to_string();
        for _ in 0..chain_len {
            body = format!("{body} + 1.0");
        }
        let src = format!("fn chain_{chain_len}(x: f32) -> f32 {{ {body} }}");
        let kernel = parse_and_lower(&src);
        kernel.validate().expect("chain kernel should be valid");
        assert_eq!(
            kernel.nodes.len(),
            1 + 2 * chain_len,
            "chain_{chain_len} should have 1 + 2*{chain_len} nodes"
        );
    }
}

#[test]
fn test_validate_scales_linearly_with_node_count() {
    // Property: validate() runs in O(n) time.
    //
    // validate() is two sequential O(n) passes: node reference checks, then
    // type inference. No nested loops over the full node set. This test
    // verifies that 10x more nodes does not cause >30x slowdown (generous
    // bound to avoid flakiness; true O(n^2) would show ~100x at these sizes).
    //
    // Uses let-binding chains to avoid MAX_EXPR_DEPTH=128 limit.
    // Each `let aK = aK_1 + 1.0;` adds 2 nodes (literal + add) at depth 1.
    let sizes = [50usize, 500];
    let mut durations = Vec::new();

    for &chain_len in &sizes {
        let mut stmts = String::new();
        stmts.push_str("let a0 = x + 1.0;\n");
        for i in 1..chain_len {
            stmts.push_str(&format!("let a{i} = a{} + 1.0;\n", i - 1));
        }
        let last = chain_len - 1;
        let src = format!("fn chain_{chain_len}(x: f32) -> f32 {{ {stmts} a{last} }}");
        let kernel = parse_and_lower(&src);

        // Warm up
        kernel.validate().expect("valid");

        // Measure
        let start = std::time::Instant::now();
        for _ in 0..10 {
            kernel.validate().expect("valid");
        }
        let elapsed = start.elapsed();
        durations.push(elapsed);
    }

    // sizes[1] / sizes[0] = 10x more nodes.
    // O(n) -> ~10x slower. O(n^2) -> ~100x slower.
    // Allow up to 30x for measurement noise and cache effects.
    let ratio = durations[1].as_nanos() as f64 / durations[0].as_nanos() as f64;
    assert!(
        ratio < 30.0,
        "validate() scaling ratio for 10x more nodes: {ratio:.1}x (expected <30x for O(n)); \
         durations: {:.2?} vs {:.2?}",
        durations[0],
        durations[1],
    );
}
