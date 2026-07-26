// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

#![allow(deprecated)] // ModelDef IR is deprecated; this module is its lowering pass

//! Model-body lowering: Rust function AST → `ModelDef`.
//!
//! Walks the body of a `#[model]`-annotated function and extracts the
//! call graph into a [`ModelDef`]. Each `let x = f(a, b);` becomes a
//! [`ModelStep`], and the return expression becomes the [`ModelOutput`].
//!
//! # Supported patterns
//!
//! - `let x = foo(arg1, arg2);` — function call bound to a let variable
//! - Trailing expression: `foo(...)` or variable name → model output
//!
//! # Unsupported (returns error)
//!
//! - Complex expressions (arithmetic, closures, control flow)
//! - Method calls on values (`.method()`)
//! - Shadowed variables

use std::collections::{HashMap, HashSet};

use crate::model_ir::{ModelDef, ModelOutput, ModelParam, ModelStep, ModelStepId, ModelValueRef};
use thiserror::Error;

/// Errors from model lowering.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelLowerError {
    #[error("model body must be a block of let-bindings followed by a return expression")]
    UnsupportedBody,

    #[error("let binding must be a simple `let x = f(...)` call: {0}")]
    UnsupportedLetBinding(String),

    #[error("function call argument must be a variable name, got: {0}")]
    UnsupportedArgument(String),

    #[error("return expression must be a function call or variable name, got: {0}")]
    UnsupportedReturn(String),

    #[error("unknown variable `{0}` — not a parameter or let-binding")]
    UnknownVariable(String),

    #[error("shadowed variable `{0}` is not supported in #[model] lowering")]
    ShadowedVariable(String),

    #[error("model function must have a return type")]
    MissingReturnType,
}

/// Scope tracking for variable resolution.
struct ModelScope {
    /// Model parameter names (O(1) lookup).
    params: HashSet<String>,
    /// binding_name → step_index (O(1) lookup).
    bindings: HashMap<String, usize>,
}

impl ModelScope {
    fn new(params: Vec<String>) -> Self {
        Self {
            params: params.into_iter().collect(),
            bindings: HashMap::new(),
        }
    }

    fn resolve(&self, name: &str) -> Result<ModelValueRef, ModelLowerError> {
        if let Some(&idx) = self.bindings.get(name) {
            return Ok(ModelValueRef::StepOutput(ModelStepId(idx)));
        }
        if self.params.contains(name) {
            return Ok(ModelValueRef::Param(name.to_string()));
        }
        Err(ModelLowerError::UnknownVariable(name.to_string()))
    }

    fn add_binding(&mut self, name: String, step_idx: usize) -> Result<(), ModelLowerError> {
        if self.params.contains(&name) {
            return Err(ModelLowerError::ShadowedVariable(name));
        }
        if self.bindings.contains_key(&name) {
            return Err(ModelLowerError::ShadowedVariable(name));
        }
        self.bindings.insert(name, step_idx);
        Ok(())
    }
}

/// Extract a simple type name from a `syn::Type`.
///
/// For paths like `f32`, `Tensor`, `NnType`, returns the last segment's ident.
/// For complex types, returns `"<complex>"` as a placeholder — the proc-macro
/// caller (nn-macros) has `quote` and can provide better stringification.
fn type_to_string(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(tp) => tp
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_else(|| "<complex>".into()),
        _ => "<complex>".into(),
    }
}

/// Extract model parameters from a function signature.
fn extract_model_params(sig: &syn::Signature) -> Result<Vec<ModelParam>, ModelLowerError> {
    let mut params = Vec::new();
    for input in &sig.inputs {
        match input {
            syn::FnArg::Typed(pat_ty) => {
                let name = match &*pat_ty.pat {
                    syn::Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                    _ => {
                        return Err(ModelLowerError::UnsupportedLetBinding(
                            "non-identifier parameter pattern".into(),
                        ));
                    }
                };
                let ty_str = type_to_string(&pat_ty.ty);
                params.push(ModelParam::new(name, ty_str));
            }
            syn::FnArg::Receiver(_) => {
                return Err(ModelLowerError::UnsupportedBody);
            }
        }
    }
    Ok(params)
}

/// Lower a `syn::ItemFn` into a `ModelDef`.
#[must_use = "returns a Result that may contain an error"]
pub fn lower_model_fn(func: &syn::ItemFn) -> Result<ModelDef, ModelLowerError> {
    let model_name = func.sig.ident.to_string();
    let params = extract_model_params(&func.sig)?;

    let return_type = match &func.sig.output {
        syn::ReturnType::Default => "()".to_string(),
        syn::ReturnType::Type(_, ty) => type_to_string(ty),
    };

    let mut scope = ModelScope::new(params.iter().map(|p| p.name.clone()).collect());
    let mut steps = Vec::new();

    // Walk the body: expect a sequence of let-bindings + a trailing expression
    let block = &func.block;
    let stmt_count = block.stmts.len();

    for (i, stmt) in block.stmts.iter().enumerate() {
        let is_last = i == stmt_count - 1;

        match stmt {
            syn::Stmt::Local(local) => {
                if let Some((binding, step)) = lower_let_binding(local, &scope, steps.len())? {
                    scope.add_binding(binding, step.id.0)?;
                    steps.push(step);
                }
                // `let _ = expr;` (wildcard discard) is silently skipped
            }
            syn::Stmt::Expr(expr, _semi) if !is_last => {
                let step = lower_call_as_step(expr, &scope, steps.len())?;
                scope.add_binding(step.binding.clone(), step.id.0)?;
                steps.push(step);
            }
            syn::Stmt::Expr(expr, None) if is_last => {
                let output = lower_return_expr(expr, &scope, &mut steps)?;
                return Ok(ModelDef::new(
                    model_name,
                    params,
                    steps,
                    output,
                    return_type,
                ));
            }
            syn::Stmt::Expr(_, Some(_semi)) if is_last => {
                return Err(ModelLowerError::UnsupportedBody)
            }
            _ => {
                return Err(ModelLowerError::UnsupportedBody);
            }
        }
    }

    Err(ModelLowerError::UnsupportedBody)
}

fn lower_let_binding(
    local: &syn::Local,
    scope: &ModelScope,
    step_idx: usize,
) -> Result<Option<(String, ModelStep)>, ModelLowerError> {
    let binding_name = match &local.pat {
        syn::Pat::Ident(pat_ident) => {
            let name = pat_ident.ident.to_string();
            if name == "_" {
                return Ok(None); // wildcard discard: `let _ = expr;`
            }
            name
        }
        syn::Pat::Wild(_) => return Ok(None), // `let _ = expr;`
        syn::Pat::Type(pat_type) => match &*pat_type.pat {
            syn::Pat::Ident(pat_ident) => {
                let name = pat_ident.ident.to_string();
                if name == "_" {
                    return Ok(None); // typed wildcard discard: `let _: T = expr;`
                }
                name
            }
            syn::Pat::Wild(_) => return Ok(None), // typed wildcard discard: `let _: T = expr;`
            _ => {
                return Err(ModelLowerError::UnsupportedLetBinding(
                    "non-identifier let pattern".into(),
                ));
            }
        },
        _ => {
            return Err(ModelLowerError::UnsupportedLetBinding(
                "non-identifier let pattern".into(),
            ));
        }
    };

    let init_expr = local.init.as_ref().ok_or_else(|| {
        ModelLowerError::UnsupportedLetBinding("let binding without initializer".into())
    })?;

    let (callee, args) = extract_call(&init_expr.expr, scope)?;

    Ok(Some((
        binding_name.clone(),
        ModelStep::new(ModelStepId(step_idx), binding_name, callee, args),
    )))
}

fn lower_call_as_step(
    expr: &syn::Expr,
    scope: &ModelScope,
    step_idx: usize,
) -> Result<ModelStep, ModelLowerError> {
    let (callee, args) = extract_call(expr, scope)?;
    let binding = format!("__step_{step_idx}");
    Ok(ModelStep::new(ModelStepId(step_idx), binding, callee, args))
}

fn lower_return_expr(
    expr: &syn::Expr,
    scope: &ModelScope,
    steps: &mut Vec<ModelStep>,
) -> Result<ModelOutput, ModelLowerError> {
    match expr {
        // Variable name → resolve to param or step output
        syn::Expr::Path(path) => {
            if let Some(ident) = path.path.get_ident() {
                let name = ident.to_string();
                match scope.resolve(&name)? {
                    ModelValueRef::Param(p) => Ok(ModelOutput::Param(p)),
                    ModelValueRef::StepOutput(id) => Ok(ModelOutput::StepOutput(id)),
                }
            } else {
                Err(ModelLowerError::UnsupportedReturn(
                    "non-simple path expression".into(),
                ))
            }
        }
        // Function call → create a final step and reference it
        syn::Expr::Call(_) => {
            let step_idx = steps.len();
            let step = lower_call_as_step(expr, scope, step_idx)?;
            let step_id = step.id;
            steps.push(step);
            Ok(ModelOutput::StepOutput(step_id))
        }
        _ => Err(ModelLowerError::UnsupportedReturn(
            "unsupported return expression (expected variable or function call)".into(),
        )),
    }
}

fn extract_call(
    expr: &syn::Expr,
    scope: &ModelScope,
) -> Result<(String, Vec<ModelValueRef>), ModelLowerError> {
    let call = match expr {
        syn::Expr::Call(c) => c,
        _ => {
            return Err(ModelLowerError::UnsupportedLetBinding(
                "expected function call".into(),
            ));
        }
    };

    let callee = match call.func.as_ref() {
        syn::Expr::Path(path) => path
            .path
            .segments
            .last()
            .map(|seg| seg.ident.to_string())
            .ok_or_else(|| ModelLowerError::UnsupportedLetBinding("empty call path".to_string()))?,
        _ => {
            return Err(ModelLowerError::UnsupportedLetBinding(
                "call target must be a function name".into(),
            ));
        }
    };

    let mut args = Vec::with_capacity(call.args.len());
    for arg in &call.args {
        match arg {
            syn::Expr::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    args.push(scope.resolve(&ident.to_string())?);
                } else {
                    return Err(ModelLowerError::UnsupportedArgument(
                        "non-simple path argument".into(),
                    ));
                }
            }
            _ => {
                return Err(ModelLowerError::UnsupportedArgument(
                    "argument must be a variable name".into(),
                ));
            }
        }
    }

    Ok((callee, args))
}

#[cfg(test)]
#[path = "lower_model_tests.rs"]
mod tests;
