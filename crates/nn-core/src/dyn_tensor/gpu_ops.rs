// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! GPU operation discriminant enums.
//!
//! Extracted from `gpu.rs` (#1575) to keep files under 400 lines.

/// Binary operation discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Element-wise maximum: `max(lhs, rhs)`.
    Maximum,
    /// Element-wise minimum: `min(lhs, rhs)`.
    Minimum,
    /// Element-wise two-argument arctangent: `atan2(self, rhs)`.
    ///
    /// Returns the angle in radians between the positive x-axis and the
    /// point `(rhs, self)`, with range `(-π, π]`. Follows Rust `f32::atan2`
    /// and MSL `atan2(y, x)` convention: `self` is `y`, `rhs` is `x`.
    Atan2,
}

/// Unary operation discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum UnaryOp {
    Relu,
    Gelu,
    Silu,
    Tanh,
    Sigmoid,
    Exp,
    Log,
    Sqrt,
    Sqr,
    Abs,
    Neg,
    Recip,
    Sin,
    Cos,
    GeluErf,
    Floor,
    Round,
    Fract,
    /// Tangent (`f32::tan`).
    Tan,
    /// Ceiling (`f32::ceil`). Smallest integer >= x.
    Ceil,
    /// Sign function: -1 if x < 0, 0 if x == 0, 1 if x > 0.
    Sign,
}

/// Reduction operation discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Comparison operation discriminant for element-wise scalar comparisons.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompareOp {
    Eq,
    Ne,
    Ge,
    Gt,
    Lt,
    Le,
}
