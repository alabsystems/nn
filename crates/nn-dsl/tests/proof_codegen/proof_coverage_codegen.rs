// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Proof coverage tests: IR validation, Display impls, MSL codegen emission,
//! and precision contract behaviour.
//!
//! Companion to `proof_coverage.rs` (lowerer error/operation tests).

use nn_dsl::ir::{
    BinOpKind, CompareOpKind, IRError, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType,
    UnaryFnKind,
};
use nn_dsl::{
    bootstrap_budget, differential_tolerance, emit_msl, emit_msl_with_contract, emit_scalar_fn,
    ir_pretty_print, within_differential_budget, Lowerer, PrecisionContract, PrecisionTier,
};

// ======================== Helpers ========================

fn parse_and_lower(src: &str) -> KernelDef {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse source");
    Lowerer::lower_fn(&func).expect("lower to IR")
}

fn simple_add_kernel() -> KernelDef {
    KernelDef::new(
        "add",
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
                    op: BinOpKind::Add,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    )
}

// ======================== ir.rs validation tests ========================

#[test]
fn test_validate_accepts_valid_kernel() {
    let kernel = simple_add_kernel();
    kernel
        .validate()
        .expect("valid kernel should pass validation");
}

#[test]
fn test_validate_rejects_out_of_bounds_node_ref() {
    let mut kernel = simple_add_kernel();
    kernel.nodes[2].kind = IRNodeKind::BinOp {
        op: BinOpKind::Add,
        lhs: NodeId::new(0),
        rhs: NodeId::new(99),
    };
    let err = kernel.validate().expect_err("should reject bad node ref");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(99)));
}

#[test]
fn test_validate_rejects_out_of_bounds_param_ref() {
    let mut kernel = simple_add_kernel();
    kernel.nodes[0].kind = IRNodeKind::Param(5);
    let err = kernel.validate().expect_err("should reject bad param ref");
    assert!(matches!(err, IRError::InvalidParamRef(5, 2)));
}

#[test]
fn test_validate_rejects_out_of_bounds_output() {
    let mut kernel = simple_add_kernel();
    kernel.output = NodeId::new(100);
    let err = kernel.validate().expect_err("should reject bad output ref");
    assert!(matches!(err, IRError::InvalidNodeRef(id) if id == NodeId::new(100)));
}

// ======================== ir.rs pretty-print and Display tests ========================

#[test]
fn test_ir_pretty_print_contains_kernel_header() {
    let kernel = simple_add_kernel();
    let pp = ir_pretty_print(&kernel);
    assert!(
        pp.contains("kernel add(x: f32, y: f32) -> f32"),
        "pretty print should contain kernel header, got:\n{pp}"
    );
    assert!(
        pp.contains("return %2"),
        "should contain return, got:\n{pp}"
    );
    assert!(
        pp.contains("param(x)"),
        "should contain param(x), got:\n{pp}"
    );
    assert!(
        pp.contains("param(y)"),
        "should contain param(y), got:\n{pp}"
    );
    assert!(
        pp.contains("add(%0, %1)"),
        "should contain add op, got:\n{pp}"
    );
}

#[test]
fn test_ir_pretty_print_literal() {
    let kernel = KernelDef::new(
        "const_fn",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        vec![
            IRNode::new(NodeId::new(0), IRNodeKind::Param(0)),
            IRNode::new(NodeId::new(1), IRNodeKind::Literal(1.23)),
            IRNode::new(
                NodeId::new(2),
                IRNodeKind::BinOp {
                    op: BinOpKind::Mul,
                    lhs: NodeId::new(0),
                    rhs: NodeId::new(1),
                },
            ),
        ],
        NodeId::new(2),
    );
    let pp = ir_pretty_print(&kernel);
    assert!(
        pp.contains("const(1.23)"),
        "should contain literal, got:\n{pp}"
    );
    assert!(pp.contains("mul(%0, %1)"), "should contain mul, got:\n{pp}");
}

#[test]
fn test_scalar_type_display() {
    assert_eq!(format!("{}", ScalarType::F32), "f32");
    assert_eq!(format!("{}", ScalarType::F16), "f16");
}

#[test]
fn test_binop_display() {
    assert_eq!(format!("{}", BinOpKind::Add), "+");
    assert_eq!(format!("{}", BinOpKind::Sub), "-");
    assert_eq!(format!("{}", BinOpKind::Mul), "*");
    assert_eq!(format!("{}", BinOpKind::Div), "/");
}

#[test]
fn test_compareop_display() {
    assert_eq!(format!("{}", CompareOpKind::Eq), "==");
    assert_eq!(format!("{}", CompareOpKind::Ne), "!=");
    assert_eq!(format!("{}", CompareOpKind::Lt), "<");
    assert_eq!(format!("{}", CompareOpKind::Le), "<=");
    assert_eq!(format!("{}", CompareOpKind::Gt), ">");
    assert_eq!(format!("{}", CompareOpKind::Ge), ">=");
}

#[test]
fn test_unaryfn_display() {
    assert_eq!(format!("{}", UnaryFnKind::Sin), "sin");
    assert_eq!(format!("{}", UnaryFnKind::Cos), "cos");
    assert_eq!(format!("{}", UnaryFnKind::Sqrt), "sqrt");
    assert_eq!(format!("{}", UnaryFnKind::Rsqrt), "rsqrt");
    assert_eq!(format!("{}", UnaryFnKind::Exp), "exp");
    assert_eq!(format!("{}", UnaryFnKind::Abs), "abs");
    assert_eq!(format!("{}", UnaryFnKind::Recip), "recip");
}

// ======================== codegen_msl.rs emission tests ========================

#[test]
fn test_cos_msl_emits_precise_cos() {
    let kernel = parse_and_lower("fn cosine(x: f32) -> f32 { x.cos() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::precise::cos("),
        "cos should emit metal::precise::cos, MSL:\n{msl}"
    );
}

#[test]
fn test_sqrt_msl_emits_precise_sqrt() {
    let kernel = parse_and_lower("fn root(x: f32) -> f32 { x.sqrt() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::precise::sqrt("),
        "sqrt should emit metal::precise::sqrt, MSL:\n{msl}"
    );
}

#[test]
fn test_rsqrt_msl_emits_precise_rsqrt() {
    let kernel = parse_and_lower("fn inv_sqrt(x: f32) -> f32 { x.rsqrt() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::precise::rsqrt("),
        "rsqrt should emit metal::precise::rsqrt, MSL:\n{msl}"
    );
}

#[test]
fn test_exp_msl_emits_precise_exp() {
    let kernel = parse_and_lower("fn exponential(x: f32) -> f32 { x.exp() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::precise::exp("),
        "exp should emit metal::precise::exp, MSL:\n{msl}"
    );
}

#[test]
fn test_abs_msl_emits_metal_abs() {
    let kernel = parse_and_lower("fn absolute(x: f32) -> f32 { x.abs() }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("metal::abs("),
        "abs should emit metal::abs (no precise:: prefix), MSL:\n{msl}"
    );
    assert!(
        !msl.contains("metal::precise::abs"),
        "abs should not use precise:: prefix, MSL:\n{msl}"
    );
}

#[test]
fn test_min_msl_emission() {
    let kernel = parse_and_lower("fn capped(x: f32) -> f32 { x.min(1.0) }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(msl.contains("min("), "min should emit min(), MSL:\n{msl}");
}

#[test]
fn test_if_expr_msl_emits_bool_compare_and_select() {
    let kernel = parse_and_lower("fn relu_if(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }");
    let msl = emit_msl(&kernel).expect("emit");
    assert!(
        msl.contains("bool"),
        "if/else should emit boolean compare temp, MSL:\n{msl}"
    );
    assert!(
        msl.contains(" > "),
        "MSL should include comparison op, MSL:\n{msl}"
    );
    assert!(
        msl.contains(" ? "),
        "if/else should emit ternary select, MSL:\n{msl}"
    );
}

#[test]
fn test_strict_precision_uses_precise_intrinsics() {
    let kernel = parse_and_lower("fn strict_math(x: f32) -> f32 { x.sin().cos() }");
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    let msl = emit_msl_with_contract(&kernel, contract).expect("emit");
    assert!(
        msl.contains("metal::precise::sin("),
        "strict tier should use metal::precise::sin, MSL:\n{msl}"
    );
    assert!(
        msl.contains("metal::precise::cos("),
        "strict tier should use metal::precise::cos, MSL:\n{msl}"
    );
}

#[test]
fn test_powi_2_inlines_square() {
    let kernel = parse_and_lower("fn sq(x: f32) -> f32 { x.powi(2) }");
    let scalar = emit_scalar_fn(&kernel).expect("emit");
    assert!(
        !scalar.contains("metal::precise::pow"),
        "powi(2) should inline as multiply, not use pow, scalar:\n{scalar}"
    );
}

// ======================== precision.rs tests ========================

#[test]
fn test_within_differential_budget_failing() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(!within_differential_budget(1.0, 1.01, contract));
}

#[test]
fn test_within_differential_budget_exact_match() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Strict, ScalarType::F32);
    assert!(within_differential_budget(42.0, 42.0, contract));
}

#[test]
fn test_precision_tier_as_str() {
    assert_eq!(PrecisionTier::Strict.as_str(), "strict");
    assert_eq!(PrecisionTier::Normal.as_str(), "normal");
    assert_eq!(PrecisionTier::Relaxed.as_str(), "relaxed");
}

#[test]
fn test_precision_tier_fast_math() {
    assert!(!PrecisionTier::Strict.fast_math());
    assert!(!PrecisionTier::Normal.fast_math());
    assert!(PrecisionTier::Relaxed.fast_math());
}

#[test]
fn test_bootstrap_budget_f16_all_tiers() {
    let (strict_abs, _) = bootstrap_budget(ScalarType::F16, PrecisionTier::Strict);
    let (normal_abs, _) = bootstrap_budget(ScalarType::F16, PrecisionTier::Normal);
    let (relaxed_abs, _) = bootstrap_budget(ScalarType::F16, PrecisionTier::Relaxed);
    assert!(
        strict_abs < normal_abs,
        "f16 strict should be tighter than normal"
    );
    assert!(
        normal_abs < relaxed_abs,
        "f16 normal should be tighter than relaxed"
    );
}

#[test]
fn test_bootstrap_budget_f32_strict_tightest() {
    let (strict_abs, _) = bootstrap_budget(ScalarType::F32, PrecisionTier::Strict);
    let (normal_abs, _) = bootstrap_budget(ScalarType::F32, PrecisionTier::Normal);
    assert!(
        strict_abs < normal_abs,
        "f32 strict should be tighter than normal"
    );
}

#[test]
fn test_differential_tolerance_grows_with_reference() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Normal, ScalarType::F32);
    let tol_small = differential_tolerance(1.0, contract);
    let tol_large = differential_tolerance(1000.0, contract);
    assert!(
        tol_large > tol_small,
        "tolerance should grow with reference magnitude"
    );
}

#[test]
fn test_precision_contract_fields() {
    let contract = PrecisionContract::bootstrap(PrecisionTier::Relaxed, ScalarType::F32);
    assert_eq!(contract.tier, PrecisionTier::Relaxed);
    assert!(contract.fast_math);
    assert!(contract.differential_abs_budget > 0.0);
    assert!(contract.differential_rel_budget > 0.0);
}
