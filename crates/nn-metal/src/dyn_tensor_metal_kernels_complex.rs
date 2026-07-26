// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Complex multi-node scalar kernel builders for DynTensor GPU Elementwise dispatch.
//!
//! Extracted from `dyn_tensor_metal_kernels.rs` for 500-line compliance.
//! Contains kernels that require >5 IR nodes (compare+select patterns, erf polynomial).

use nn_dsl::ir::{
    BinOpKind, BinaryFnKind, CompareOpKind, IRNode, IRNodeKind, KernelDef, NodeId as IrNodeId,
    Param, ScalarType, UnaryFnKind,
};

/// Helper for building multi-node scalar IR graphs without boilerplate.
struct IrBuilder {
    nodes: Vec<IRNode>,
    next_id: usize,
}

impl IrBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
        }
    }

    fn push(&mut self, kind: IRNodeKind) -> IrNodeId {
        let id = IrNodeId::new(self.next_id);
        self.next_id += 1;
        self.nodes.push(IRNode::new(id, kind));
        id
    }

    fn param(&mut self, idx: usize) -> IrNodeId {
        self.push(IRNodeKind::Param(idx))
    }

    fn lit(&mut self, val: f64) -> IrNodeId {
        self.push(IRNodeKind::Literal(val))
    }

    fn binop(&mut self, op: BinOpKind, lhs: IrNodeId, rhs: IrNodeId) -> IrNodeId {
        self.push(IRNodeKind::BinOp { op, lhs, rhs })
    }

    fn mul(&mut self, lhs: IrNodeId, rhs: IrNodeId) -> IrNodeId {
        self.binop(BinOpKind::Mul, lhs, rhs)
    }

    fn add(&mut self, lhs: IrNodeId, rhs: IrNodeId) -> IrNodeId {
        self.binop(BinOpKind::Add, lhs, rhs)
    }

    fn sub(&mut self, lhs: IrNodeId, rhs: IrNodeId) -> IrNodeId {
        self.binop(BinOpKind::Sub, lhs, rhs)
    }

    fn unary(&mut self, op: UnaryFnKind, input: IrNodeId) -> IrNodeId {
        self.push(IRNodeKind::UnaryFn { op, input })
    }

    fn binary_fn(&mut self, op: BinaryFnKind, lhs: IrNodeId, rhs: IrNodeId) -> IrNodeId {
        self.push(IRNodeKind::BinaryFn { op, lhs, rhs })
    }

    fn compare(&mut self, op: CompareOpKind, lhs: IrNodeId, rhs: IrNodeId) -> IrNodeId {
        self.push(IRNodeKind::Compare { op, lhs, rhs })
    }

    fn select(&mut self, cond: IrNodeId, then_val: IrNodeId, else_val: IrNodeId) -> IrNodeId {
        self.push(IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        })
    }

    fn into_nodes(self) -> Vec<IRNode> {
        self.nodes
    }
}

/// Build a comparison-against-scalar kernel:
/// `fn cmp(x: f32) -> f32 { if x OP val { 1.0 } else { 0.0 } }`
///
/// Used by GPU comparison ops (ge, gt, lt, le). The result is f32 (0.0/1.0)
/// to maintain the GPU f32-only invariant; callers convert to U8 after CPU
/// transfer.
pub(crate) fn make_compare_scalar_kernel(name: &str, op: CompareOpKind, val: f64) -> KernelDef {
    let mut b = IrBuilder::new();
    let x = b.param(0);
    let v = b.lit(val);
    let cmp = b.compare(op, x, v);
    let one = b.lit(1.0);
    let zero = b.lit(0.0);
    let result = b.select(cmp, one, zero);
    KernelDef::new(
        name,
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        b.into_nodes(),
        result,
    )
}

/// Build a compare-two-tensors kernel:
/// `fn cmp_tensor(lhs: f32, rhs: f32) -> f32 { if lhs OP rhs { 1.0 } else { 0.0 } }`
///
/// Used by GPU tensor-vs-tensor comparison ops (#1357 AC1). Returns f32 (0.0/1.0)
/// to match the scalar compare kernel convention.
pub(crate) fn make_compare_tensor_kernel(name: &str, op: CompareOpKind) -> KernelDef {
    let mut b = IrBuilder::new();
    let lhs = b.param(0);
    let rhs = b.param(1);
    let cmp = b.compare(op, lhs, rhs);
    let one = b.lit(1.0);
    let zero = b.lit(0.0);
    let result = b.select(cmp, one, zero);
    KernelDef::new(
        name,
        vec![
            Param::new("lhs", ScalarType::F32),
            Param::new("rhs", ScalarType::F32),
        ],
        ScalarType::F32,
        b.into_nodes(),
        result,
    )
}

/// Build a where_cond kernel:
/// `fn where_cond(mask: f32, on_true: f32, on_false: f32) -> f32`
/// `{ if mask != 0.0 { on_true } else { on_false } }`
///
/// Mask is passed as f32 (converted from U8 before dispatch).
pub(crate) fn make_where_cond_kernel() -> KernelDef {
    let mut b = IrBuilder::new();
    let mask = b.param(0);
    let on_true = b.param(1);
    let on_false = b.param(2);
    let zero = b.lit(0.0);
    let cmp = b.compare(CompareOpKind::Ne, mask, zero);
    let result = b.select(cmp, on_true, on_false);
    KernelDef::new(
        "where_cond",
        vec![
            Param::new("mask", ScalarType::F32),
            Param::new("on_true", ScalarType::F32),
            Param::new("on_false", ScalarType::F32),
        ],
        ScalarType::F32,
        b.into_nodes(),
        result,
    )
}

/// Build a maximum kernel: `fn maximum(a: f32, b: f32) -> f32 { if a > b { a } else { b } }`
pub(crate) fn make_maximum_kernel() -> KernelDef {
    make_minmax_kernel("maximum", CompareOpKind::Gt)
}

/// Build a minimum kernel: `fn minimum(a: f32, b: f32) -> f32 { if a < b { a } else { b } }`
pub(crate) fn make_minimum_kernel() -> KernelDef {
    make_minmax_kernel("minimum", CompareOpKind::Lt)
}

fn make_minmax_kernel(name: &str, op: CompareOpKind) -> KernelDef {
    let mut b = IrBuilder::new();
    let a = b.param(0);
    let bv = b.param(1);
    let cmp = b.compare(op, a, bv);
    let result = b.select(cmp, a, bv);
    KernelDef::new(
        name,
        vec![
            Param::new("a", ScalarType::F32),
            Param::new("b", ScalarType::F32),
        ],
        ScalarType::F32,
        b.into_nodes(),
        result,
    )
}

/// Build a clamp kernel: `fn clamp(x: f32) -> f32 { max(lo, min(hi, x)) }`
///
/// Single-dispatch replacement for the 8-encoding relu decomposition of
/// `clamp(min, max)`. Used by `gpu_clamp()` in the Metal backend.
pub(crate) fn make_clamp_kernel(name: &str, lo: f64, hi: f64, stype: ScalarType) -> KernelDef {
    let mut b = IrBuilder::new();
    let x = b.param(0);
    let lo_val = b.lit(lo);
    let hi_val = b.lit(hi);
    // max(lo, min(hi, x))
    let above_hi = b.compare(CompareOpKind::Gt, x, hi_val);
    let clamped_hi = b.select(above_hi, hi_val, x); // min(hi, x)
    let below_lo = b.compare(CompareOpKind::Lt, clamped_hi, lo_val);
    let result = b.select(below_lo, lo_val, clamped_hi); // max(lo, min(hi, x))
    KernelDef::new(
        name,
        vec![Param::new("x", stype)],
        stype,
        b.into_nodes(),
        result,
    )
}

/// Build a clamp_min kernel: `fn clamp_min(x: f32) -> f32 { max(lo, x) }`
///
/// Single-dispatch replacement for the 3-encoding relu decomposition of
/// `clamp_min(lo)` = `sub_scalar(lo) + relu() + add_scalar(lo)`.
pub(crate) fn make_clamp_min_kernel(name: &str, lo: f64, stype: ScalarType) -> KernelDef {
    let mut b = IrBuilder::new();
    let x = b.param(0);
    let lo_val = b.lit(lo);
    let below = b.compare(CompareOpKind::Lt, x, lo_val);
    let result = b.select(below, lo_val, x);
    KernelDef::new(
        name,
        vec![Param::new("x", stype)],
        stype,
        b.into_nodes(),
        result,
    )
}

/// Build a clamp_max kernel: `fn clamp_max(x: T) -> T { min(hi, x) }`
///
/// Single-dispatch replacement for the 5-encoding relu decomposition of
/// `clamp_max(hi)` = `neg() + add_scalar(hi) + relu() + neg() + add_scalar(hi)`.
pub(crate) fn make_clamp_max_kernel(name: &str, hi: f64, stype: ScalarType) -> KernelDef {
    let mut b = IrBuilder::new();
    let x = b.param(0);
    let hi_val = b.lit(hi);
    let above = b.compare(CompareOpKind::Gt, x, hi_val);
    let result = b.select(above, hi_val, x);
    KernelDef::new(
        name,
        vec![Param::new("x", stype)],
        stype,
        b.into_nodes(),
        result,
    )
}

/// Build an atan2 kernel: `fn atan2(y: f32, x: f32) -> f32 { atan2(y, x) }`
///
/// Uses `IRNodeKind::BinaryFn` with `BinaryFnKind::Atan2`, which emits the
/// MSL native `atan2(y, x)` intrinsic.
pub(crate) fn make_atan2_kernel() -> KernelDef {
    let mut b = IrBuilder::new();
    let y = b.param(0);
    let x = b.param(1);
    let result = b.binary_fn(BinaryFnKind::Atan2, y, x);
    KernelDef::new(
        "atan2",
        vec![
            Param::new("y", ScalarType::F32),
            Param::new("x", ScalarType::F32),
        ],
        ScalarType::F32,
        b.into_nodes(),
        result,
    )
}

/// Build `fn gelu_erf(x: f32) -> f32 { 0.5 * x * (1 + erf(x / sqrt(2))) }`
///
/// Uses Abramowitz & Stegun formula 7.1.26 for erf, matching the CPU
/// `erf_f32` implementation in `dyn_tensor/ops/math.rs:18`.
pub(crate) fn make_gelu_erf_kernel() -> KernelDef {
    let mut b = IrBuilder::new();
    let x = b.param(0);
    let erf_val = build_erf_graph(&mut b, x);
    // gelu_erf = 0.5 * x * (1 + erf)
    let half = b.lit(0.5);
    let one = b.lit(1.0);
    let one_plus_erf = b.add(one, erf_val);
    let hx = b.mul(half, x);
    let result = b.mul(hx, one_plus_erf);
    KernelDef::new(
        "gelu_erf",
        vec![Param::new("x", ScalarType::F32)],
        ScalarType::F32,
        b.into_nodes(),
        result,
    )
}

/// Build the erf approximation subgraph using Abramowitz & Stegun formula 7.1.26.
///
/// Input `x` is scaled by `1/sqrt(2)` inside this function.
/// Returns the node ID of `erf(x / sqrt(2))`.
fn build_erf_graph(b: &mut IrBuilder, x: IrNodeId) -> IrNodeId {
    // u = x * FRAC_1_SQRT_2
    let frac = b.lit(std::f64::consts::FRAC_1_SQRT_2);
    let u = b.mul(x, frac);
    // sign(u): +1 if u >= 0, -1 otherwise
    let zero = b.lit(0.0);
    let one = b.lit(1.0);
    let neg_one = b.lit(-1.0);
    let cond = b.compare(CompareOpKind::Ge, u, zero);
    let sign = b.select(cond, one, neg_one);
    // ax = abs(u)
    let ax = b.unary(UnaryFnKind::Abs, u);
    // t = 1 / (1 + p * ax)
    let p = b.lit(0.327_591_1);
    let p_ax = b.mul(p, ax);
    let denom = b.add(one, p_ax);
    let t = b.unary(UnaryFnKind::Recip, denom);
    // Horner polynomial: ((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t
    let poly = build_horner_poly(b, t);
    // exp(-u*u)
    let u2 = b.mul(u, u);
    let neg_u2 = b.sub(zero, u2);
    let exp_nu2 = b.unary(UnaryFnKind::Exp, neg_u2);
    // erf = sign * (1 - poly * exp(-u*u))
    let pe = b.mul(poly, exp_nu2);
    let om = b.sub(one, pe);
    b.mul(sign, om)
}

/// Horner-form evaluation of the 5-coefficient erf polynomial.
fn build_horner_poly(b: &mut IrBuilder, t: IrNodeId) -> IrNodeId {
    let coeffs: [f64; 5] = [
        1.061_405_4,
        -1.453_152,
        1.421_413_8,
        -0.284_496_74,
        0.254_829_6,
    ];
    let mut acc = b.lit(coeffs[0]);
    for &c in &coeffs[1..] {
        acc = b.mul(acc, t);
        let cn = b.lit(c);
        acc = b.add(acc, cn);
    }
    b.mul(acc, t)
}

/// Build a scalar binary op kernel: `fn op(x: T) -> T { x OP scalar }`.
///
/// Bakes the scalar value as an inline `Literal` constant in the MSL kernel,
/// eliminating buffer allocation + GPU transfer for the scalar operand.
/// Same pattern as `make_clamp_kernel`. Part of #3230 (Gap 2).
///
/// `stype` determines the MSL buffer types: F32 → `float*`, F16/BF16 → `half*`.
/// The scalar literal is emitted at the buffer precision; for F16 this means
/// half-precision compute, which is correct for simple binary ops (add/mul/sub/div).
pub(crate) fn make_scalar_binop_kernel(
    name: &str,
    op: BinOpKind,
    scalar: f64,
    stype: ScalarType,
) -> KernelDef {
    let mut b = IrBuilder::new();
    let x = b.param(0);
    let s = b.lit(scalar);
    let result = b.binop(op, x, s);
    KernelDef::new(
        name,
        vec![Param::new("x", stype)],
        stype,
        b.into_nodes(),
        result,
    )
}
