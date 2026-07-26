// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Enum types for kernel IR node operations.
//!
//! Extracted from `ir.rs` for the 500-line limit.

/// Binary arithmetic operations on scalar values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum BinOpKind {
    /// Addition (`+`).
    Add,
    /// Subtraction (`-`).
    Sub,
    /// Multiplication (`*`).
    Mul,
    /// Division (`/`).
    Div,
}

/// Boolean comparison operations between two scalar values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum CompareOpKind {
    /// Equal (`==`).
    Eq,
    /// Not equal (`!=`).
    Ne,
    /// Less than (`<`).
    Lt,
    /// Less than or equal (`<=`).
    Le,
    /// Greater than (`>`).
    Gt,
    /// Greater than or equal (`>=`).
    Ge,
}

/// Unary math functions supported by the kernel subset.
///
/// These map 1:1 to both Rust `f32` methods and MSL `metal::precise::*` calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum UnaryFnKind {
    /// Sine (`f32::sin`).
    Sin,
    /// Cosine (`f32::cos`).
    Cos,
    /// Square root (`f32::sqrt`).
    Sqrt,
    /// Reciprocal square root (`1.0 / f32::sqrt`).
    Rsqrt,
    /// Exponential (`f32::exp`).
    Exp,
    /// Absolute value (`f32::abs`).
    Abs,
    /// Reciprocal (`f32::recip`).
    Recip,
    /// Hyperbolic tangent (`f32::tanh`).
    Tanh,
    /// Natural logarithm (`f32::ln`).
    Log,
    /// Floor (`f32::floor`). Largest integer ≤ x.
    Floor,
    /// Round ties to even (`f32::round_ties_even`). Matches `torch.round()`.
    Round,
    /// Fractional part: `x - floor(x)`. Matches MSL `fract()` / GLSL `fract()`.
    ///
    /// Note: differs from Rust `f32::fract()` which uses trunc (`x - trunc(x)`).
    /// For negative inputs: `fract(-1.7)` = 0.3 (floor-based), not -0.7 (trunc-based).
    Fract,
    /// Negation (`-x`). Emitted as `(-x)` in MSL, `f32::neg()` in Rust.
    Neg,
}

impl UnaryFnKind {
    const ALL: &[Self] = &[
        Self::Sin,
        Self::Cos,
        Self::Sqrt,
        Self::Rsqrt,
        Self::Exp,
        Self::Abs,
        Self::Recip,
        Self::Tanh,
        Self::Log,
        Self::Floor,
        Self::Round,
        Self::Fract,
        Self::Neg,
    ];

    /// Rust method name. Exhaustive match — new variants cause compile errors.
    #[must_use]
    pub fn method_name(self) -> &'static str {
        match self {
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Sqrt => "sqrt",
            Self::Rsqrt => "rsqrt",
            Self::Exp => "exp",
            Self::Abs => "abs",
            Self::Recip => "recip",
            Self::Tanh => "tanh",
            Self::Log => "ln",
            Self::Floor => "floor",
            Self::Round => "round_ties_even",
            Self::Fract => "fract",
            Self::Neg => "neg",
        }
    }

    /// Reverse lookup by Rust method name. Always in sync via [`method_name`](Self::method_name).
    #[must_use]
    pub fn from_method_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|k| k.method_name() == name)
    }
}

/// Min/max selection between two scalar values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum MinMaxKind {
    /// Select the smaller of two values (`f32::min`).
    Min,
    /// Select the larger of two values (`f32::max`).
    Max,
}

/// Two-input math functions supported by the kernel subset.
///
/// These map 1:1 to both Rust `f32` methods and MSL intrinsics.
/// Unlike [`BinOpKind`] (infix operators), these emit function-call
/// syntax: `fn_name(a, b)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum BinaryFnKind {
    /// Two-argument arctangent: `atan2(y, x)`.
    ///
    /// Returns the angle in radians between the positive x-axis and the
    /// point `(x, y)`, with range `(-π, π]`.
    Atan2,
}
