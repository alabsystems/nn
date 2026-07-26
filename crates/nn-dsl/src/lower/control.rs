// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Control-flow lowering: `if/else` → `Select` node with scoped bindings.

use crate::ir::{IRNodeKind, NodeId};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(super) fn lower_if_expr(&mut self, if_expr: &syn::ExprIf) -> Result<NodeId, LowerError> {
        let cond = self.lower_expr(&if_expr.cond)?;
        let then_val = self.lower_block_scoped(&if_expr.then_branch)?;
        let else_expr = if_expr
            .else_branch
            .as_ref()
            .ok_or_else(|| LowerError::UnsupportedExpr("if expression requires else".to_string()))?
            .1
            .as_ref();
        let else_val = self.lower_expr_scoped(else_expr)?;
        Ok(self.add_node(IRNodeKind::Select {
            cond,
            then_val,
            else_val,
        }))
    }

    pub(super) fn lower_block_scoped(&mut self, block: &syn::Block) -> Result<NodeId, LowerError> {
        let bindings_snapshot = self.bindings.clone();
        let result = self.lower_block(block);
        self.bindings = bindings_snapshot;
        result
    }

    pub(super) fn lower_expr_scoped(&mut self, expr: &syn::Expr) -> Result<NodeId, LowerError> {
        let bindings_snapshot = self.bindings.clone();
        let result = self.lower_expr(expr);
        self.bindings = bindings_snapshot;
        result
    }
}
