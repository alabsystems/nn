// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Reference validation for KernelIR nodes.
//!
//! Validates that all node references are in-bounds, topologically ordered
//! (no forward or self references), and structurally correct (non-empty
//! sum-reduce, valid param indices, bounded powi exponents).

use super::{IRError, IRNode, IRNodeKind, KernelDef, NodeId, POWI_MAX_EXPONENT};

/// Validate that a name is a structurally valid identifier.
///
/// Checks structural properties only: non-empty, starts with a letter or
/// underscore, contains only alphanumeric characters and underscores.
///
/// Backend-specific reserved word checks (MSL, PTX, SPIR-V) run at codegen
/// time, not here. This keeps the IR backend-agnostic. (Part of #586.)
pub(crate) fn validate_identifier(name: &str, context: &'static str) -> Result<(), IRError> {
    if name.is_empty() {
        return Err(IRError::InvalidIdentifier {
            name: name.to_string(),
            context,
            reason: "name is empty",
        });
    }
    let first = name.as_bytes()[0];
    if first.is_ascii_digit() {
        return Err(IRError::InvalidIdentifier {
            name: name.to_string(),
            context,
            reason: "starts with a digit",
        });
    }
    if !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
        return Err(IRError::InvalidIdentifier {
            name: name.to_string(),
            context,
            reason: "contains non-alphanumeric non-underscore characters",
        });
    }
    Ok(())
}

impl KernelDef {
    /// Validate kernel name and all parameter names as structurally valid identifiers.
    ///
    /// Checks structural properties only (non-empty, valid characters). Backend-specific
    /// reserved word checks run at codegen time. (Part of #586.)
    pub(super) fn validate_identifiers(&self) -> Result<(), IRError> {
        validate_identifier(&self.name, "kernel name")?;
        for param in &self.params {
            validate_identifier(&param.name, "parameter name")?;
        }
        Ok(())
    }

    pub(super) fn validate_node(&self, node: &IRNode) -> Result<(), IRError> {
        let current = node.id;
        match &node.kind {
            IRNodeKind::Param(idx) => {
                if *idx >= self.params.len() {
                    return Err(IRError::InvalidParamRef(*idx, self.params.len()));
                }
            }
            IRNodeKind::Literal(v) => {
                if !v.is_finite() {
                    return Err(IRError::NonFiniteLiteral(node.id, *v));
                }
            }
            IRNodeKind::BinOp { lhs, rhs, .. } => {
                self.check_ref(current, *lhs)?;
                self.check_ref(current, *rhs)?;
            }
            IRNodeKind::Compare { lhs, rhs, .. } => {
                self.check_ref(current, *lhs)?;
                self.check_ref(current, *rhs)?;
            }
            IRNodeKind::UnaryFn { input, .. } => {
                self.check_ref(current, *input)?;
            }
            IRNodeKind::Powi { base, exp } => {
                self.check_ref(current, *base)?;
                if exp.unsigned_abs() > POWI_MAX_EXPONENT {
                    return Err(IRError::PowiExponentTooLarge {
                        node: current,
                        exp: *exp,
                        max: POWI_MAX_EXPONENT,
                    });
                }
            }
            IRNodeKind::Clamp { input, min, max } => {
                self.check_ref(current, *input)?;
                self.check_ref(current, *min)?;
                self.check_ref(current, *max)?;
            }
            IRNodeKind::MinMax { lhs, rhs, .. } => {
                self.check_ref(current, *lhs)?;
                self.check_ref(current, *rhs)?;
            }
            IRNodeKind::Select {
                cond,
                then_val,
                else_val,
            } => {
                self.check_ref(current, *cond)?;
                self.check_ref(current, *then_val)?;
                self.check_ref(current, *else_val)?;
            }
            IRNodeKind::SumReduce { inputs } => {
                if inputs.is_empty() {
                    return Err(IRError::EmptySumReduce(node.id));
                }
                for input in inputs {
                    self.check_ref(current, *input)?;
                }
            }
            IRNodeKind::BinaryFn { lhs, rhs, .. } => {
                self.check_ref(current, *lhs)?;
                self.check_ref(current, *rhs)?;
            }
        }
        Ok(())
    }

    /// Check that `target` is a valid reference from `current`: must be in
    /// bounds AND strictly before `current` in topological order.
    fn check_ref(&self, current: NodeId, target: NodeId) -> Result<(), IRError> {
        if target.0 >= self.nodes.len() {
            return Err(IRError::InvalidNodeRef(target));
        }
        if target.0 >= current.0 {
            return Err(IRError::ForwardRef(current, target));
        }
        Ok(())
    }

    /// Check only that `id` is within the node array bounds (for output ref).
    pub(super) fn check_ref_bounds(&self, id: NodeId) -> Result<(), IRError> {
        if id.0 >= self.nodes.len() {
            return Err(IRError::InvalidNodeRef(id));
        }
        Ok(())
    }
}
