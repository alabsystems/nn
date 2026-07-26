// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Model-level IR — a graph of named kernel/function calls with tensor data flow.
//!
//! # Deprecated
//!
//! **This module is deprecated.** The trace-based compilation pipeline
//! (`trace.rs` -> `trace_compile.rs` -> `CompiledModel`) is the primary path
//! for model execution and verification. `ModelDef` captures only the call
//! graph (function names and data flow) but lacks tensor shapes, dtypes,
//! and GPU execution semantics. The trace-based `ComputationGraph` with
//! `TraceOp` variants captures tensor-level fidelity and is what
//! `CompiledModel` and NY verification consume.
//!
//! See `designs/2026-03-13-compile-time-graph-execution.md` (Option 3:
//! trace-first) and `designs/2026-03-14-compile-time-graph-execution-decomposition.md`.
//!
//! # Original design
//!
//! This is the **third IR level** in the nn hierarchy:
//!
//! - **Level 1 (`KernelDef`)**: scalar operations (add, sin, clamp, etc.)
//! - **Level 2 (`TensorKernelDef`)**: tensor operations (reduce, broadcast, elementwise)
//! - **Level 3 (`ModelDef`)**: model composition (named steps calling kernels and functions)
//!
//! A `ModelDef` captures the data-flow graph of a `#[model]`-annotated function.
//! Each [`ModelStep`] is a function or kernel call, with inputs referencing either
//! model parameters or outputs of previous steps. The graph is topologically
//! ordered: each step references only earlier steps or model inputs.
//!
//! The model IR intentionally does **not** contain the function bodies. It records
//! which functions are called, in what order, and how data flows between them.
//! Function-level semantics (scalar ops, tensor ops) are captured at Levels 1-2.

#![allow(deprecated)] // Internal uses of deprecated types within this module

use thiserror::Error;

/// Unique identifier for a step in the model graph.
#[deprecated(
    since = "0.1.0",
    note = "Use trace-based ComputationGraph (trace.rs -> trace_compile.rs -> CompiledModel) instead of ModelDef IR"
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ModelStepId(pub usize);

/// Reference to a value in the model data-flow graph.
///
/// Each call argument resolves to either a model input parameter or
/// the output of a previous model step.
#[deprecated(
    since = "0.1.0",
    note = "Use trace-based ComputationGraph (trace.rs -> trace_compile.rs -> CompiledModel) instead of ModelDef IR"
)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ModelValueRef {
    /// A model function parameter, referenced by name.
    Param(String),
    /// The output of a previous step, referenced by its step id.
    StepOutput(ModelStepId),
}

/// A single step in the model graph: one function or kernel invocation.
#[deprecated(
    since = "0.1.0",
    note = "Use trace-based ComputationGraph (trace.rs -> trace_compile.rs -> CompiledModel) instead of ModelDef IR"
)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ModelStep {
    /// Unique identifier for this step.
    pub id: ModelStepId,
    /// Name of the let-binding for this step's output (e.g., `"text_hidden"`).
    pub binding: String,
    /// Name of the called function or kernel (e.g., `"text_encoder_stub"`).
    pub callee: String,
    /// Arguments to the call, in order.
    pub args: Vec<ModelValueRef>,
}

impl ModelStep {
    /// Create a new model step.
    #[must_use]
    pub fn new(
        id: ModelStepId,
        binding: impl Into<String>,
        callee: impl Into<String>,
        args: Vec<ModelValueRef>,
    ) -> Self {
        Self {
            id,
            binding: binding.into(),
            callee: callee.into(),
            args,
        }
    }
}

/// A model input parameter.
#[deprecated(
    since = "0.1.0",
    note = "Use trace-based ComputationGraph (trace.rs -> trace_compile.rs -> CompiledModel) instead of ModelDef IR"
)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ModelParam {
    /// Parameter name as declared in the function signature.
    pub name: String,
    /// Rust type of the parameter (stringified).
    pub ty: String,
}

impl ModelParam {
    /// Create a new model parameter.
    #[must_use]
    pub fn new(name: impl Into<String>, ty: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ty: ty.into(),
        }
    }
}

/// The final output reference of the model.
///
/// The model's return value is either a direct forwarding of a parameter,
/// the output of a step, or a direct call expression (which is also a step).
#[deprecated(
    since = "0.1.0",
    note = "Use trace-based ComputationGraph (trace.rs -> trace_compile.rs -> CompiledModel) instead of ModelDef IR"
)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ModelOutput {
    /// The model returns the output of a specific step.
    StepOutput(ModelStepId),
    /// The model returns a parameter directly.
    Param(String),
}

/// Complete model-level definition extracted from a `#[model]` function.
///
/// **Deprecated:** Use the trace-based compilation pipeline instead.
/// The trace pipeline (`trace_graph()` -> `compile_trace()` -> `CompiledModel`)
/// captures tensor-level operation semantics (shapes, dtypes, GPU dispatch)
/// that `ModelDef` cannot represent. `ModelDef` only captures the call graph
/// (function names and data flow) without function body semantics.
///
/// Topologically ordered: each step references only earlier steps or model
/// parameters. The output identifies which step (or parameter) produces the
/// model's return value.
#[deprecated(
    since = "0.1.0",
    note = "Use trace-based ComputationGraph (trace.rs -> trace_compile.rs -> CompiledModel) instead of ModelDef IR"
)]
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ModelDef {
    /// Name of the model function.
    pub name: String,
    /// Model input parameters, in declaration order.
    pub params: Vec<ModelParam>,
    /// Ordered steps (function/kernel calls) in the model body.
    pub steps: Vec<ModelStep>,
    /// The model's output (return value).
    pub output: ModelOutput,
    /// Rust return type (stringified).
    pub return_type: String,
}

impl ModelDef {
    /// Create a new model definition.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        params: Vec<ModelParam>,
        steps: Vec<ModelStep>,
        output: ModelOutput,
        return_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            params,
            steps,
            output,
            return_type: return_type.into(),
        }
    }

    /// Validate the model definition.
    ///
    /// Checks:
    /// - Step ids are sequential starting from 0
    /// - All value references point to valid params or earlier steps
    /// - Output references a valid step or param
    #[must_use = "returns a Result that may contain an error"]
    pub fn validate(&self) -> Result<(), ModelIRError> {
        let param_names: std::collections::HashSet<&str> =
            self.params.iter().map(|p| p.name.as_str()).collect();

        for (i, step) in self.steps.iter().enumerate() {
            if step.id.0 != i {
                return Err(ModelIRError::StepIdMismatch {
                    expected: i,
                    found: step.id,
                });
            }

            for arg in &step.args {
                match arg {
                    ModelValueRef::Param(name) => {
                        if !param_names.contains(&name.as_str()) {
                            return Err(ModelIRError::UnknownParam {
                                step: step.id,
                                name: name.clone(),
                            });
                        }
                    }
                    ModelValueRef::StepOutput(ref_id) => {
                        if ref_id.0 >= i {
                            return Err(ModelIRError::ForwardRef {
                                step: step.id,
                                references: *ref_id,
                            });
                        }
                    }
                }
            }
        }

        match &self.output {
            ModelOutput::StepOutput(id) => {
                if id.0 >= self.steps.len() {
                    return Err(ModelIRError::InvalidOutputRef(*id));
                }
            }
            ModelOutput::Param(name) => {
                if !param_names.contains(&name.as_str()) {
                    return Err(ModelIRError::UnknownOutputParam(name.clone()));
                }
            }
        }

        Ok(())
    }
}

/// Errors from model IR construction or validation.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelIRError {
    #[error("step at index {expected} has id {found:?} (must be sequential)")]
    StepIdMismatch { expected: usize, found: ModelStepId },

    #[error("step {step:?} references unknown parameter `{name}`")]
    UnknownParam { step: ModelStepId, name: String },

    #[error("step {step:?} has forward reference to {references:?}")]
    ForwardRef {
        step: ModelStepId,
        references: ModelStepId,
    },

    #[error("model output references invalid step {0:?}")]
    InvalidOutputRef(ModelStepId),

    #[error("model output references unknown parameter `{0}`")]
    UnknownOutputParam(String),
}

/// Pretty-print a `ModelDef` for debugging.
#[deprecated(
    since = "0.1.0",
    note = "Use trace-based ComputationGraph (trace.rs -> trace_compile.rs -> CompiledModel) instead of ModelDef IR"
)]
#[must_use]
pub fn model_ir_pretty_print(model: &ModelDef) -> String {
    let mut out = format!("model {}(", model.name);
    for (i, p) in model.params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{}: {}", p.name, p.ty));
    }
    out.push_str(&format!(") -> {} {{\n", model.return_type));

    for step in &model.steps {
        out.push_str(&format!("  let {} = {}(", step.binding, step.callee));
        for (i, arg) in step.args.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            match arg {
                ModelValueRef::Param(name) => out.push_str(name),
                ModelValueRef::StepOutput(id) => match model.steps.get(id.0) {
                    Some(step) => out.push_str(&step.binding),
                    None => out.push_str(&format!("<invalid step {}>", id.0)),
                },
            }
        }
        out.push_str(");\n");
    }

    match &model.output {
        ModelOutput::StepOutput(id) => match model.steps.get(id.0) {
            Some(step) => out.push_str(&format!("  {}\n", step.binding)),
            None => out.push_str(&format!("  <invalid step {}>\n", id.0)),
        },
        ModelOutput::Param(name) => {
            out.push_str(&format!("  {name}\n"));
        }
    }
    out.push('}');
    out
}

#[cfg(test)]
#[path = "model_ir_tests.rs"]
mod tests;
