// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Operation-specific translation helpers for KernelIR -> NY mapping.
//!
//! Handles binary ops, unary functions, piecewise activations (MinMax, Compare,
//! Select), and pattern matching for specialized layers (ReLU, LeakyReLU).

mod binop;
mod clamp;
mod compare;
mod minmax;
mod select;
mod sum_reduce;
mod unary;

pub(crate) use binop::{translate_binary_fn, translate_binop};
pub(crate) use clamp::translate_clamp;
#[cfg(kani)]
pub(crate) use compare::evaluate_constant_compare;
pub(crate) use compare::translate_compare;
pub(crate) use minmax::translate_minmax;
pub(crate) use select::translate_select;
pub(crate) use sum_reduce::translate_sum_reduce;
pub(crate) use unary::translate_unary;

#[cfg(kani)]
#[path = "kani_graph_ops_extra.rs"]
mod kani_graph_ops_extra;

#[cfg(test)]
#[path = "../graph_ops_tests_arithmetic.rs"]
mod tests_arithmetic;

#[cfg(test)]
#[path = "../graph_ops_tests_piecewise.rs"]
mod tests_piecewise;

#[cfg(test)]
#[path = "../graph_ops_tests_reduce_error.rs"]
mod tests_reduce_error;
