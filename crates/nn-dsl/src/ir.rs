// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Kernel Intermediate Representation
//!
//! A DAG of typed nodes representing a scalar kernel function. Each node
//! produces exactly one scalar value. The graph is built by the lowering pass
//! (see `lower.rs`) and consumed by codegen backends (MSL, Kani, etc.).

#[path = "ir_error.rs"]
mod ir_error;
#[path = "ir_pretty.rs"]
mod ir_pretty;
#[path = "ir_type_check.rs"]
mod ir_type_check;
#[path = "ir_validate.rs"]
mod ir_validate;

pub use ir_error::IRError;

pub use ir_pretty::ir_pretty_print;

/// Maximum absolute exponent allowed for `Powi` nodes.
///
/// Binary exponentiation generates O(log n) temporaries, but exponents beyond
/// this limit produce unreasonably large MSL and likely indicate a bug.
pub const POWI_MAX_EXPONENT: u32 = 64;

/// Unique identifier for a node in the kernel IR graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct NodeId(usize);

impl NodeId {
    /// Create a new `NodeId` from a raw index.
    pub fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// Get the underlying index.
    pub fn index(self) -> usize {
        self.0
    }
}

/// Scalar types supported by kernel functions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ScalarType {
    F32,
    F16,
    /// Brain floating-point 16-bit. On Apple GPUs (no native bf16 compute),
    /// MSL emits `half` and precision matches F16. The Rust reference
    /// implementation uses `half::bf16` for correct rounding semantics.
    BF16,
}

impl ScalarType {
    const ALL: &[Self] = &[Self::F32, Self::F16, Self::BF16];

    /// Rust type name. Exhaustive match — new variants cause compile errors.
    #[must_use]
    pub fn type_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        }
    }

    /// Reverse lookup by Rust type name. Always in sync via [`type_name`](Self::type_name).
    #[must_use]
    pub fn from_type_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|k| k.type_name() == name)
    }

    /// MSL scalar type name for buffer declarations.
    ///
    /// BF16 maps to `"half"` — Apple GPUs have no native bf16 compute.
    /// The MetalElement impl converts bf16→f16 at the buffer boundary.
    #[must_use]
    pub fn msl_str(self) -> &'static str {
        match self {
            Self::F32 => "float",
            Self::F16 | Self::BF16 => "half",
        }
    }

    /// MSL accumulator type for reductions and dot products.
    ///
    /// F16/BF16 accumulate in F32 to avoid catastrophic precision loss.
    /// Matches PyTorch CUDA `opmath_t` (#1352).
    #[must_use]
    pub fn msl_accumulator_str(self) -> &'static str {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => "float",
        }
    }

    /// Byte size of one element in GPU buffer.
    #[must_use]
    pub fn byte_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

/// Internal node-level value type for type inference and validation.
///
/// Unlike [`ScalarType`] (which represents kernel signature types), `ValueType`
/// includes `Bool` to represent the output of comparison operations. This
/// enables the IR validator to enforce that `Select.cond` is boolean and that
/// arithmetic operands are numeric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ValueType {
    F32,
    F16,
    BF16,
    Bool,
}

impl ValueType {
    /// Returns `true` for numeric types (`F32`, `F16`, `BF16`), `false` for `Bool`.
    #[must_use]
    pub fn is_numeric(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }
}

impl From<ScalarType> for ValueType {
    fn from(ty: ScalarType) -> Self {
        match ty {
            ScalarType::F32 => Self::F32,
            ScalarType::F16 => Self::F16,
            ScalarType::BF16 => Self::BF16,
        }
    }
}

/// A kernel function parameter.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct Param {
    /// Parameter name as it appears in the Rust function signature.
    pub name: String,
    /// Scalar type of this parameter.
    pub ty: ScalarType,
}

impl Param {
    /// Create a new kernel parameter.
    #[must_use]
    pub fn new(name: impl Into<String>, ty: ScalarType) -> Self {
        Self {
            name: name.into(),
            ty,
        }
    }
}

/// Complete IR for a single kernel function.
///
/// The `nodes` vector is topologically ordered: every node references only
/// earlier nodes. `output` is the node whose value is the function return.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct KernelDef {
    /// Kernel function name (used as MSL entry point).
    pub name: String,
    /// Ordered list of function parameters.
    pub params: Vec<Param>,
    /// Scalar type of the kernel's return value.
    pub return_type: ScalarType,
    /// Topologically ordered IR nodes.
    pub nodes: Vec<IRNode>,
    /// Node whose value is the function return.
    pub output: NodeId,
}

impl KernelDef {
    /// Create a new kernel definition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        params: Vec<Param>,
        return_type: ScalarType,
        nodes: Vec<IRNode>,
        output: NodeId,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            return_type,
            nodes,
            output,
        }
    }

    /// Returns `true` if the kernel IR contains operations that are sensitive
    /// to GPU flush-to-zero (FTZ) mode: `rsqrt`, `recip`, or `div`.
    ///
    /// When denormal inputs flow through these operations, Metal GPUs may
    /// produce NaN or divergent results. FTZ flushes denormals to zero,
    /// which can create `rsqrt(0)→+inf` then `0*inf→NaN`, or `0/0→NaN`.
    /// Rust CPU handles denormals correctly. Denormal differential tests
    /// should be skipped for these kernels since the divergence is an expected
    /// hardware behavior, not a codegen bug.
    #[must_use]
    pub(crate) fn has_ftz_sensitive_op(&self) -> bool {
        self.nodes.iter().any(|node| {
            // Exhaustive match — new IRNodeKind variants cause compile errors,
            // forcing explicit FTZ-sensitivity classification.
            match &node.kind {
                IRNodeKind::UnaryFn { op, .. } => {
                    matches!(op, UnaryFnKind::Rsqrt | UnaryFnKind::Recip)
                }
                IRNodeKind::BinOp { op, .. } => matches!(op, BinOpKind::Div),
                IRNodeKind::Param(_)
                | IRNodeKind::Literal(_)
                | IRNodeKind::Compare { .. }
                | IRNodeKind::Powi { .. }
                | IRNodeKind::Clamp { .. }
                | IRNodeKind::MinMax { .. }
                | IRNodeKind::Select { .. }
                | IRNodeKind::SumReduce { .. }
                | IRNodeKind::BinaryFn { .. } => false,
            }
        })
    }
}

/// A single node in the IR graph.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct IRNode {
    /// Position-based identifier (must equal the node's index in `KernelDef::nodes`).
    pub id: NodeId,
    /// The operation this node performs.
    pub kind: IRNodeKind,
}

impl IRNode {
    /// Create a new IR node.
    #[must_use]
    pub fn new(id: NodeId, kind: IRNodeKind) -> Self {
        Self { id, kind }
    }
}

/// Node kinds in the kernel IR.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum IRNodeKind {
    /// Reference to a function parameter (by index into `KernelDef::params`).
    Param(usize),
    /// Floating-point literal constant.
    Literal(f64),
    /// Binary arithmetic operation.
    BinOp {
        op: BinOpKind,
        lhs: NodeId,
        rhs: NodeId,
    },
    /// Boolean comparison between two scalar values.
    Compare {
        op: CompareOpKind,
        lhs: NodeId,
        rhs: NodeId,
    },
    /// Unary math function (sin, cos, sqrt, etc.).
    UnaryFn { op: UnaryFnKind, input: NodeId },
    /// Integer power: `base.powi(exp)`.
    Powi { base: NodeId, exp: i32 },
    /// Clamp: `input.clamp(min, max)`.
    Clamp {
        input: NodeId,
        min: NodeId,
        max: NodeId,
    },
    /// Min or max of two values.
    MinMax {
        op: MinMaxKind,
        lhs: NodeId,
        rhs: NodeId,
    },
    /// Select one of two values based on a boolean condition.
    Select {
        cond: NodeId,
        then_val: NodeId,
        else_val: NodeId,
    },
    /// Sum-reduction over an explicit list of scalar inputs.
    SumReduce { inputs: Vec<NodeId> },
    /// Two-input math function (atan2, etc.).
    ///
    /// Emitted as `fn_name(lhs, rhs)` in MSL — function-call syntax for
    /// two-input operations that are not infix arithmetic.
    BinaryFn {
        op: BinaryFnKind,
        lhs: NodeId,
        rhs: NodeId,
    },
}

// Enum types extracted to ir_enums.rs (500-line limit).
#[path = "ir_enums.rs"]
mod ir_enums;
pub use ir_enums::{BinOpKind, BinaryFnKind, CompareOpKind, MinMaxKind, UnaryFnKind};

impl KernelDef {
    /// Validate that all node references are in bounds and topologically ordered.
    ///
    /// Topological order: every node may only reference nodes with strictly
    /// smaller indices. Self-references and forward references are rejected.
    /// After reference validation, runs type inference and checks type
    /// consistency across all nodes.
    ///
    /// Also validates that the kernel name and all parameter names are
    /// structurally valid identifiers (no special characters or empty names).
    /// Backend-specific reserved word checks run at codegen time.
    #[must_use = "returns a Result that may contain an error"]
    pub fn validate(&self) -> Result<(), IRError> {
        self.validate_identifiers()?;
        for (i, node) in self.nodes.iter().enumerate() {
            if node.id != NodeId::new(i) {
                return Err(IRError::MismatchedNodeId {
                    found: node.id,
                    expected_index: i,
                });
            }
        }
        for node in &self.nodes {
            self.validate_node(node)?;
        }
        self.check_ref_bounds(self.output)?;
        self.validate_types()?;
        Ok(())
    }
}

// Type conversions and Display impls extracted to ir_convert.rs (#557, 500-line limit).
#[path = "ir_convert.rs"]
mod ir_convert;

#[cfg(test)]
#[path = "ir_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "ir_tests_validate.rs"]
mod validate_tests;
