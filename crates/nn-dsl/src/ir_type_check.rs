// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Type inference and validation for KernelIR nodes.
//!
//! Computes a [`ValueType`] for every node and validates type consistency:
//! - `Select.cond` must be `Bool` (from a `Compare` node)
//! - Binary/compare/minmax operands must be the same numeric type
//! - Arithmetic and unary operands must be numeric (not `Bool`)
//! - Output node type must match the declared `return_type`

use super::{IRError, IRNode, IRNodeKind, KernelDef, NodeId, ValueType};

impl KernelDef {
    /// Infer the [`ValueType`] of every node in topological order.
    ///
    /// # Precondition
    ///
    /// All node references must be in-bounds and topologically ordered.
    /// This is guaranteed when called via [`KernelDef::validate()`], which runs
    /// [`validate_node()`] for every node before type inference.
    pub(super) fn infer_types(&self) -> Vec<ValueType> {
        assert!(
            self.nodes.iter().all(|n| self.validate_node(n).is_ok()),
            "infer_types precondition: all node references must be valid"
        );
        let mut types = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let ty = match &node.kind {
                IRNodeKind::Param(idx) => ValueType::from(self.params[*idx].ty),
                IRNodeKind::Literal(_) => ValueType::from(self.return_type),
                IRNodeKind::BinOp { lhs, .. } => types[lhs.0],
                IRNodeKind::Compare { .. } => ValueType::Bool,
                IRNodeKind::UnaryFn { input, .. } => types[input.0],
                IRNodeKind::Powi { base, .. } => types[base.0],
                IRNodeKind::Clamp { input, .. } => types[input.0],
                IRNodeKind::MinMax { lhs, .. } => types[lhs.0],
                IRNodeKind::BinaryFn { lhs, .. } => types[lhs.0],
                IRNodeKind::Select { then_val, .. } => types[then_val.0],
                IRNodeKind::SumReduce { inputs } => types[inputs[0].0],
            };
            types.push(ty);
        }
        types
    }

    /// Validate type consistency across all nodes and the output declaration.
    pub(super) fn validate_types(&self) -> Result<(), IRError> {
        let types = self.infer_types();
        for node in &self.nodes {
            validate_node_types(node, &types)?;
        }
        let output_type = types[self.output.index()];
        let expected = ValueType::from(self.return_type);
        if output_type != expected {
            return Err(IRError::OutputTypeMismatch {
                found: output_type,
                expected,
            });
        }
        Ok(())
    }
}

/// Assert that `operand` has a numeric type in `types`.
fn check_numeric(node: NodeId, operand: NodeId, types: &[ValueType]) -> Result<(), IRError> {
    let t = types[operand.0];
    if !t.is_numeric() {
        return Err(IRError::NonNumericOperand {
            node,
            operand,
            found: t,
        });
    }
    Ok(())
}

/// Assert that `lhs` and `rhs` are both numeric and the same type.
fn check_numeric_pair(
    node: NodeId,
    lhs: NodeId,
    rhs: NodeId,
    types: &[ValueType],
) -> Result<(), IRError> {
    check_numeric(node, lhs, types)?;
    check_numeric(node, rhs, types)?;
    let lt = types[lhs.0];
    let rt = types[rhs.0];
    if lt != rt {
        return Err(IRError::OperandTypeMismatch {
            node,
            lhs_type: lt,
            rhs_type: rt,
        });
    }
    Ok(())
}

fn validate_node_types(node: &IRNode, types: &[ValueType]) -> Result<(), IRError> {
    match &node.kind {
        IRNodeKind::Param(_) | IRNodeKind::Literal(_) => {}
        IRNodeKind::BinOp { lhs, rhs, .. }
        | IRNodeKind::Compare { lhs, rhs, .. }
        | IRNodeKind::MinMax { lhs, rhs, .. }
        | IRNodeKind::BinaryFn { lhs, rhs, .. } => {
            check_numeric_pair(node.id, *lhs, *rhs, types)?;
        }
        IRNodeKind::UnaryFn { input, .. } => {
            check_numeric(node.id, *input, types)?;
        }
        IRNodeKind::Powi { base, .. } => {
            check_numeric(node.id, *base, types)?;
        }
        IRNodeKind::Clamp { input, min, max } => {
            check_numeric(node.id, *input, types)?;
            check_numeric(node.id, *min, types)?;
            check_numeric(node.id, *max, types)?;
            let it = types[input.0];
            if types[min.0] != it {
                return Err(IRError::OperandTypeMismatch {
                    node: node.id,
                    lhs_type: it,
                    rhs_type: types[min.0],
                });
            }
            if types[max.0] != it {
                return Err(IRError::OperandTypeMismatch {
                    node: node.id,
                    lhs_type: it,
                    rhs_type: types[max.0],
                });
            }
        }
        IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        } => {
            let ct = types[cond.0];
            if ct != ValueType::Bool {
                return Err(IRError::SelectCondNotBool {
                    node: node.id,
                    found: ct,
                });
            }
            let tt = types[then_val.0];
            let et = types[else_val.0];
            if tt != et {
                return Err(IRError::SelectBranchTypeMismatch {
                    node: node.id,
                    then_type: tt,
                    else_type: et,
                });
            }
        }
        IRNodeKind::SumReduce { inputs } => {
            let first_type = types[inputs[0].0];
            check_numeric(node.id, inputs[0], types)?;
            for input in &inputs[1..] {
                let t = types[input.0];
                if t != first_type {
                    return Err(IRError::OperandTypeMismatch {
                        node: node.id,
                        lhs_type: first_type,
                        rhs_type: t,
                    });
                }
            }
        }
    }
    Ok(())
}
