// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! InputSpec / OutputSpec types for classifying graph inputs and outputs.
//!
//! Extracted from `parse.rs` for file-size compliance.

use serde::Deserialize;

use super::{Argument, TensorArgument};

/// Classification of a graph input.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum InputSpec {
    /// `{"parameter": {...}}` — a lifted nn.Module parameter.
    Parameter(InputSpecParameter),
    /// `{"buffer": {...}}` — a lifted nn.Module buffer.
    Buffer(InputSpecBuffer),
    /// `{"user_input": {...}}` — a runtime user input.
    UserInput(InputSpecUserInput),
    /// `{"tensor_constant": {...}}` — a constant tensor.
    TensorConstant(InputSpecTensorConstant),
    /// `{"constant_input": {...}}` — a scalar/simple constant.
    ConstantInput(InputSpecConstantInput),
    /// `{"token": {...}}` — a control token.
    Token(InputSpecToken),
    /// Catch-all for unknown variants.
    Other(serde_json::Value),
}

#[derive(Debug, Deserialize)]
pub struct InputSpecParameter {
    pub parameter: ParameterSpec,
}

#[derive(Debug, Deserialize)]
pub struct ParameterSpec {
    pub arg: TensorArgument,
    pub parameter_name: String,
}

#[derive(Debug, Deserialize)]
pub struct InputSpecBuffer {
    pub buffer: BufferSpec,
}

#[derive(Debug, Deserialize)]
pub struct BufferSpec {
    pub arg: TensorArgument,
    pub buffer_name: String,
    #[serde(default)]
    pub persistent: bool,
}

#[derive(Debug, Deserialize)]
pub struct InputSpecUserInput {
    pub user_input: UserInputSpec,
}

#[derive(Debug, Deserialize)]
pub struct UserInputSpec {
    pub arg: Argument,
}

#[derive(Debug, Deserialize)]
pub struct InputSpecTensorConstant {
    pub tensor_constant: TensorConstantSpec,
}

#[derive(Debug, Deserialize)]
pub struct TensorConstantSpec {
    pub arg: TensorArgument,
    pub tensor_constant_name: String,
}

#[derive(Debug, Deserialize)]
pub struct InputSpecConstantInput {
    pub constant_input: ConstantInputSpec,
}

#[derive(Debug, Deserialize)]
pub struct ConstantInputSpec {
    pub name: String,
    pub value: Argument,
}

#[derive(Debug, Deserialize)]
pub struct InputSpecToken {
    pub token: TokenSpec,
}

#[derive(Debug, Deserialize)]
pub struct TokenSpec {
    pub arg: TensorArgument,
}

/// Classification of a graph output.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OutputSpec {
    /// `{"user_output": {...}}` — a user-visible output tensor.
    UserOutput(OutputSpecUserOutput),
    /// `{"buffer_mutation": {...}}` — a buffer mutation.
    BufferMutation(OutputSpecBufferMutation),
    /// Catch-all.
    Other(serde_json::Value),
}

#[derive(Debug, Deserialize)]
pub struct OutputSpecUserOutput {
    pub user_output: UserOutputSpec,
}

#[derive(Debug, Deserialize)]
pub struct UserOutputSpec {
    pub arg: Argument,
}

#[derive(Debug, Deserialize)]
pub struct OutputSpecBufferMutation {
    pub buffer_mutation: BufferMutationSpec,
}

#[derive(Debug, Deserialize)]
pub struct BufferMutationSpec {
    pub arg: TensorArgument,
    pub buffer_name: String,
}
