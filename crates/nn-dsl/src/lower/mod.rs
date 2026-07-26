// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Lower a Rust function AST (syn) to KernelIR.
//!
//! Supports the kernel-safe subset: scalar arithmetic, f32 math methods,
//! let bindings, and a single return expression.

mod calls;
mod control;

use std::collections::HashMap;

use thiserror::Error;

use crate::ir::{
    BinOpKind, CompareOpKind, IRError, IRNode, IRNodeKind, KernelDef, NodeId, Param, ScalarType,
};
use crate::kernel_error::KernelError;
use crate::tensor_ir::TensorIRError;

/// Errors produced during AST-to-IR lowering.
///
/// Runtime/kernel validation errors (non-finite inputs, shape mismatches, etc.)
/// have moved to [`KernelError`]. The `Kernel` variant bridges them so that
/// `build_scalar_kernel()` and other functions that chain lowering + validation
/// can still use `?`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LowerError {
    #[error("unsupported binary operator")]
    UnsupportedBinOp,
    #[error("unsupported unary operator")]
    UnsupportedUnaryOp,
    #[error("unsupported method: {0}")]
    UnsupportedMethod(String),
    #[error("unsupported type: {0}")]
    UnsupportedType(String),
    #[error("unsupported pattern in let binding")]
    UnsupportedPattern,
    #[error("unsupported expression: {0}")]
    UnsupportedExpr(String),
    #[error("unsupported literal type")]
    UnsupportedLiteral,
    #[error("unsupported path expression")]
    UnsupportedPath,
    #[error("unsupported statement type")]
    UnsupportedStatement,
    #[error("self parameter not allowed in kernels")]
    SelfParam,
    #[error("kernel function must have a return type")]
    MissingReturnType,
    #[error("let binding must have an initializer")]
    UninitializedLet,
    #[error("function body must end with an expression")]
    EmptyBody,
    #[error("unknown variable: {0}")]
    UnknownVariable(String),
    #[error("invalid numeric literal")]
    InvalidLiteral,
    #[error("wrong argument count for {method}: expected {expected}, got {got}")]
    WrongArgCount {
        method: String,
        expected: usize,
        got: usize,
    },
    #[error("expression nesting depth exceeds limit ({0})")]
    ExprTooDeep(usize),
    #[error("failed to parse kernel source: {0}")]
    ParseError(#[from] syn::Error),
    #[error("IR validation failed: {0}")]
    IrValidation(#[from] IRError),
    #[error("tensor IR validation failed: {0}")]
    TensorIrValidation(#[from] TensorIRError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
}

/// Maximum expression nesting depth before rejecting a kernel.
///
/// Hand-written kernels rarely exceed 20 levels. This guard prevents
/// stack overflow from pathological or auto-generated inputs.
const MAX_EXPR_DEPTH: usize = 128;

/// Lowers a syn `ItemFn` to a `KernelDef`.
pub struct Lowerer {
    nodes: Vec<IRNode>,
    params: Vec<Param>,
    bindings: HashMap<String, NodeId>,
    depth: usize,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            params: Vec::new(),
            bindings: HashMap::new(),
            depth: 0,
        }
    }

    fn next_id(&self) -> NodeId {
        NodeId::new(self.nodes.len())
    }

    fn add_node(&mut self, kind: IRNodeKind) -> NodeId {
        let id = self.next_id();
        self.nodes.push(IRNode { id, kind });
        id
    }

    fn add_param(&mut self, name: String, ty: ScalarType) -> NodeId {
        let idx = self.params.len();
        let id = self.next_id();
        self.params.push(Param {
            name: name.clone(),
            ty,
        });
        self.nodes.push(IRNode {
            id,
            kind: IRNodeKind::Param(idx),
        });
        self.bindings.insert(name, id);
        id
    }

    /// Lower a complete function to IR.
    #[must_use = "returns a Result that may contain an error"]
    pub fn lower_fn(func: &syn::ItemFn) -> Result<KernelDef, LowerError> {
        let mut lowerer = Self::new();

        let name = func.sig.ident.to_string();

        for input in &func.sig.inputs {
            match input {
                syn::FnArg::Typed(pt) => {
                    let pname = match pt.pat.as_ref() {
                        syn::Pat::Ident(pi) => pi.ident.to_string(),
                        _ => return Err(LowerError::UnsupportedPattern),
                    };
                    let pty = Self::parse_scalar_type(&pt.ty)?;
                    lowerer.add_param(pname, pty);
                }
                syn::FnArg::Receiver(_) => return Err(LowerError::SelfParam),
            }
        }

        let return_type = match &func.sig.output {
            syn::ReturnType::Type(_, ty) => Self::parse_scalar_type(ty)?,
            syn::ReturnType::Default => return Err(LowerError::MissingReturnType),
        };

        let output = lowerer.lower_block(&func.block)?;

        let kernel = KernelDef {
            name,
            params: lowerer.params,
            return_type,
            nodes: lowerer.nodes,
            output,
        };
        kernel.validate()?;
        Ok(kernel)
    }

    pub(crate) fn lower_block(&mut self, block: &syn::Block) -> Result<NodeId, LowerError> {
        let mut last = None;
        for stmt in &block.stmts {
            match stmt {
                syn::Stmt::Local(local) => {
                    let var_name = match &local.pat {
                        syn::Pat::Ident(pi) => pi.ident.to_string(),
                        syn::Pat::Type(pt) => match pt.pat.as_ref() {
                            syn::Pat::Ident(pi) => pi.ident.to_string(),
                            _ => return Err(LowerError::UnsupportedPattern),
                        },
                        _ => return Err(LowerError::UnsupportedPattern),
                    };
                    let init = local.init.as_ref().ok_or(LowerError::UninitializedLet)?;
                    let node = self.lower_expr(&init.expr)?;
                    self.bindings.insert(var_name, node);
                }
                syn::Stmt::Expr(expr, _semi) => {
                    last = Some(self.lower_expr(expr)?);
                }
                _ => return Err(LowerError::UnsupportedStatement),
            }
        }
        last.ok_or(LowerError::EmptyBody)
    }

    pub(crate) fn lower_expr(&mut self, expr: &syn::Expr) -> Result<NodeId, LowerError> {
        self.depth += 1;
        if self.depth > MAX_EXPR_DEPTH {
            return Err(LowerError::ExprTooDeep(MAX_EXPR_DEPTH));
        }
        let result = match expr {
            syn::Expr::Binary(bin) => self.lower_binary_expr(bin),
            syn::Expr::MethodCall(mc) => self.lower_method_call(mc),
            syn::Expr::Call(call) => self.lower_call(call),
            syn::Expr::If(if_expr) => self.lower_if_expr(if_expr),
            syn::Expr::Path(ep) => self.lower_path_expr(ep),
            syn::Expr::Lit(el) => self.lower_literal_expr(el),
            syn::Expr::Paren(p) => self.lower_expr(&p.expr),
            syn::Expr::Group(g) => self.lower_expr(&g.expr),
            syn::Expr::Unary(u) => self.lower_unary_expr(u),
            syn::Expr::Block(b) => self.lower_block(&b.block),
            _ => Err(LowerError::UnsupportedExpr(format!(
                "{:?}",
                std::mem::discriminant(expr)
            ))),
        };
        self.depth -= 1;
        result
    }

    fn lower_binary_expr(&mut self, bin: &syn::ExprBinary) -> Result<NodeId, LowerError> {
        let lhs = self.lower_expr(&bin.left)?;
        let rhs = self.lower_expr(&bin.right)?;
        let kind = match &bin.op {
            syn::BinOp::Add(_) => IRNodeKind::BinOp {
                op: BinOpKind::Add,
                lhs,
                rhs,
            },
            syn::BinOp::Sub(_) => IRNodeKind::BinOp {
                op: BinOpKind::Sub,
                lhs,
                rhs,
            },
            syn::BinOp::Mul(_) => IRNodeKind::BinOp {
                op: BinOpKind::Mul,
                lhs,
                rhs,
            },
            syn::BinOp::Div(_) => IRNodeKind::BinOp {
                op: BinOpKind::Div,
                lhs,
                rhs,
            },
            syn::BinOp::Eq(_) => IRNodeKind::Compare {
                op: CompareOpKind::Eq,
                lhs,
                rhs,
            },
            syn::BinOp::Ne(_) => IRNodeKind::Compare {
                op: CompareOpKind::Ne,
                lhs,
                rhs,
            },
            syn::BinOp::Lt(_) => IRNodeKind::Compare {
                op: CompareOpKind::Lt,
                lhs,
                rhs,
            },
            syn::BinOp::Le(_) => IRNodeKind::Compare {
                op: CompareOpKind::Le,
                lhs,
                rhs,
            },
            syn::BinOp::Gt(_) => IRNodeKind::Compare {
                op: CompareOpKind::Gt,
                lhs,
                rhs,
            },
            syn::BinOp::Ge(_) => IRNodeKind::Compare {
                op: CompareOpKind::Ge,
                lhs,
                rhs,
            },
            _ => return Err(LowerError::UnsupportedBinOp),
        };
        Ok(self.add_node(kind))
    }

    fn lower_path_expr(&self, ep: &syn::ExprPath) -> Result<NodeId, LowerError> {
        let ident = ep.path.get_ident().ok_or(LowerError::UnsupportedPath)?;
        let name = ident.to_string();
        self.bindings
            .get(&name)
            .copied()
            .ok_or(LowerError::UnknownVariable(name))
    }

    fn lower_literal_expr(&mut self, el: &syn::ExprLit) -> Result<NodeId, LowerError> {
        let value = match &el.lit {
            syn::Lit::Float(f) => f.base10_parse().map_err(|_| LowerError::InvalidLiteral)?,
            syn::Lit::Int(i) => i.base10_parse().map_err(|_| LowerError::InvalidLiteral)?,
            _ => return Err(LowerError::UnsupportedLiteral),
        };
        Ok(self.add_node(IRNodeKind::Literal(value)))
    }

    fn lower_unary_expr(&mut self, unary: &syn::ExprUnary) -> Result<NodeId, LowerError> {
        match &unary.op {
            syn::UnOp::Neg(_) => {
                let operand = self.lower_expr(&unary.expr)?;
                let zero = self.add_node(IRNodeKind::Literal(0.0));
                Ok(self.add_node(IRNodeKind::BinOp {
                    op: BinOpKind::Sub,
                    lhs: zero,
                    rhs: operand,
                }))
            }
            _ => Err(LowerError::UnsupportedUnaryOp),
        }
    }

    fn parse_scalar_type(ty: &syn::Type) -> Result<ScalarType, LowerError> {
        match ty {
            syn::Type::Path(tp) => {
                if let Some(segment) = tp.path.segments.last() {
                    let name = segment.ident.to_string();
                    ScalarType::from_type_name(&name).ok_or(LowerError::UnsupportedType(name))
                } else {
                    Err(LowerError::UnsupportedType("empty path".to_string()))
                }
            }
            _ => Err(LowerError::UnsupportedType("non-path type".to_string())),
        }
    }
}
