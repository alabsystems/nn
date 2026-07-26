// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! JSON serde structs for `torch.export` ExportedProgram format.
//!
//! torch.export serializes graphs as JSON with a versioned schema.
//! Union types use single-key JSON objects (e.g., `{"as_int": 42}`).
//! Enum types use integer values (e.g., `ScalarType::Float = 7`).

use std::collections::HashMap;

use serde::Deserialize;

/// Top-level ExportedProgram (the `models/model.json` file).
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub struct ExportedProgram {
    pub graph_module: GraphModule,
    pub schema_version: SchemaVersion,
    #[serde(default)]
    pub opset_version: HashMap<String, u64>,
    #[serde(default)]
    pub range_constraints: HashMap<String, RangeConstraint>,
    #[serde(default)]
    pub torch_version: Option<String>,
}

/// Schema version (major.minor).
#[derive(Debug, Deserialize)]
pub struct SchemaVersion {
    pub major: u64,
    pub minor: u64,
}

/// Range constraint for dynamic dimensions.
#[derive(Debug, Deserialize)]
pub struct RangeConstraint {
    pub min_val: i64,
    pub max_val: i64,
}

/// The GraphModule wrapping the computation graph and its signature.
#[derive(Debug, Deserialize)]
pub struct GraphModule {
    pub graph: Graph,
    pub signature: GraphSignature,
    #[serde(default)]
    pub module_call_graph: Vec<serde_json::Value>,
}

/// The computation graph: nodes + tensor metadata.
#[derive(Debug, Deserialize)]
pub struct Graph {
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub nodes: Vec<Node>,
    #[serde(default)]
    pub tensor_values: HashMap<String, TensorMeta>,
    #[serde(default)]
    pub is_single_tensor_return: bool,
}

/// A single computation node (one aten op call).
#[derive(Debug, Deserialize)]
pub struct Node {
    pub target: String,
    #[serde(default)]
    pub inputs: Vec<NamedArgument>,
    #[serde(default)]
    pub outputs: Vec<Argument>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A named argument to a node (positional or keyword).
#[derive(Debug, Deserialize)]
pub struct NamedArgument {
    pub name: String,
    pub arg: Argument,
    #[serde(default)]
    pub kind: Option<i32>,
}

/// Graph signature: input/output specs, parameter/buffer classification.
#[derive(Debug, Deserialize)]
pub struct GraphSignature {
    pub input_specs: Vec<InputSpec>,
    pub output_specs: Vec<OutputSpec>,
}

/// Tensor metadata: dtype, shape, strides.
#[derive(Debug, Deserialize)]
pub struct TensorMeta {
    pub dtype: i32,
    pub sizes: Vec<SymInt>,
    #[serde(default)]
    pub requires_grad: bool,
    #[serde(default)]
    pub strides: Vec<SymInt>,
    #[serde(default)]
    pub storage_offset: Option<SymInt>,
    #[serde(default)]
    pub device: Option<DeviceSpec>,
    #[serde(default)]
    pub layout: Option<i32>,
}

/// Device specification.
#[derive(Debug, Deserialize)]
pub struct DeviceSpec {
    #[serde(rename = "type")]
    pub device_type: String,
    pub index: Option<i32>,
}

// ---------------------------------------------------------------------------
// Tagged union types — torch.export serializes these as single-key JSON objects
// ---------------------------------------------------------------------------

/// The core argument type — a tagged union with many variants.
///
/// Serialized as single-key JSON objects: `{"as_int": 42}`, `{"as_tensor": {"name": "x"}}`.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum Argument {
    /// `{"as_tensor": {"name": "..."}}` — reference to a tensor by name.
    Tensor(ArgumentTensor),
    /// `{"as_int": N}` — integer literal.
    Int(ArgumentInt),
    /// `{"as_ints": [N, ...]}` — integer list.
    Ints(ArgumentInts),
    /// `{"as_float": F}` — float literal.
    Float(ArgumentFloat),
    /// `{"as_floats": [F, ...]}` — float list.
    Floats(ArgumentFloats),
    /// `{"as_bool": B}` — boolean literal.
    Bool(ArgumentBool),
    /// `{"as_bools": [B, ...]}` — boolean list.
    Bools(ArgumentBools),
    /// `{"as_string": "..."}` — string literal.
    Str(ArgumentString),
    /// `{"as_none": true}` — None/null.
    None(ArgumentNone),
    /// `{"as_tensors": [{...}, ...]}` — tensor list.
    Tensors(ArgumentTensors),
    /// `{"as_scalar_type": N}` — ScalarType enum value.
    ScalarType(ArgumentScalarType),
    /// `{"as_optional_tensors": [...]}` — optional tensor list.
    OptionalTensors(ArgumentOptionalTensors),
    /// `{"as_sym_int": {...}}` — symbolic integer.
    SymInt(ArgumentSymInt),
    /// `{"as_sym_ints": [...]}` — symbolic integer list.
    SymInts(ArgumentSymInts),
    /// `{"as_memory_format": N}` — memory format enum.
    MemoryFormat(ArgumentMemoryFormat),
    /// `{"as_device": {...}}` — device.
    Device(ArgumentDevice),
    /// Catch-all for unknown or complex argument types.
    Other(serde_json::Value),
}

// Wrapper structs for each Argument variant (single-key deserialization).

#[derive(Debug, Deserialize)]
pub struct ArgumentTensor {
    pub as_tensor: TensorArgument,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentInt {
    pub as_int: i64,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentInts {
    pub as_ints: Vec<i64>,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentFloat {
    pub as_float: f64,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentFloats {
    pub as_floats: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentBool {
    pub as_bool: bool,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentBools {
    pub as_bools: Vec<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentString {
    pub as_string: String,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentNone {
    pub as_none: bool,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentTensors {
    pub as_tensors: Vec<TensorArgument>,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentScalarType {
    pub as_scalar_type: i32,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentOptionalTensors {
    pub as_optional_tensors: Vec<Argument>,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentSymInt {
    pub as_sym_int: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentSymInts {
    pub as_sym_ints: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentMemoryFormat {
    pub as_memory_format: i32,
}

#[derive(Debug, Deserialize)]
pub struct ArgumentDevice {
    pub as_device: DeviceSpec,
}

/// A tensor reference by name.
#[derive(Debug, Deserialize)]
pub struct TensorArgument {
    pub name: String,
}

/// Symbolic integer — either concrete or a dynamic expression.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SymInt {
    /// Concrete integer: `{"as_int": N}`.
    Concrete(SymIntConcrete),
    /// Symbolic expression: `{"as_expr": {"expr_str": "...", "hint": {...}}}`.
    Symbolic(SymIntSymbolic),
}

#[derive(Debug, Deserialize)]
pub struct SymIntConcrete {
    pub as_int: i64,
}

#[derive(Debug, Deserialize)]
pub struct SymIntSymbolic {
    pub as_expr: SymIntExpr,
}

#[derive(Debug, Deserialize)]
pub struct SymIntExpr {
    pub expr_str: String,
    pub hint: Option<Box<SymInt>>,
}

// ---------------------------------------------------------------------------
// InputSpec / OutputSpec — classify graph inputs/outputs
// ---------------------------------------------------------------------------

#[path = "parse_specs.rs"]
mod specs;
pub use specs::*;

// ---------------------------------------------------------------------------
// Helpers for extracting concrete values from Argument
// ---------------------------------------------------------------------------

impl Argument {
    /// Extract tensor name if this is an `as_tensor` argument.
    pub fn as_tensor_name(&self) -> Option<&str> {
        match self {
            Self::Tensor(t) => Some(&t.as_tensor.name),
            _ => None,
        }
    }

    /// Extract integer value if this is an `as_int` argument.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(i.as_int),
            _ => None,
        }
    }

    /// Extract integer list if this is an `as_ints` argument.
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            Self::Ints(i) => Some(&i.as_ints),
            _ => None,
        }
    }

    /// Extract float value if this is an `as_float` argument.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(f.as_float),
            _ => None,
        }
    }

    /// Extract boolean value if this is an `as_bool` argument.
    pub fn as_bool_val(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(b.as_bool),
            _ => None,
        }
    }

    /// Extract string value if this is an `as_string` argument.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(&s.as_string),
            _ => None,
        }
    }

    /// Returns true if this is an `as_none` argument.
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None(_))
    }

    /// Extract tensor name list if this is an `as_tensors` argument.
    pub fn as_tensor_names(&self) -> Option<Vec<&str>> {
        match self {
            Self::Tensors(t) => Some(t.as_tensors.iter().map(|ta| ta.name.as_str()).collect()),
            _ => None,
        }
    }
}

impl SymInt {
    /// Extract a concrete integer value, ignoring symbolic expressions.
    ///
    /// Returns `None` for symbolic dimensions (dynamic shapes).
    pub fn as_concrete(&self) -> Option<i64> {
        match self {
            Self::Concrete(c) => Some(c.as_int),
            Self::Symbolic(_) => None,
        }
    }
}

impl TensorMeta {
    /// Extract concrete shape (all dimensions must be static integers).
    ///
    /// Returns `None` if any dimension is symbolic (dynamic shape).
    pub fn concrete_shape(&self) -> Option<Vec<usize>> {
        self.sizes
            .iter()
            .map(|s| s.as_concrete().and_then(|v| usize::try_from(v).ok()))
            .collect()
    }

    /// Convert `ScalarType` integer to nn `DType`.
    pub fn to_dtype(&self) -> Option<nn_core::DType> {
        scalar_type_to_dtype(self.dtype)
    }
}

/// Map torch ScalarType integer to nn DType.
fn scalar_type_to_dtype(st: i32) -> Option<nn_core::DType> {
    match st {
        1 => Some(nn_core::DType::U8),
        5 => Some(nn_core::DType::I64),
        6 => Some(nn_core::DType::F16),
        7 => Some(nn_core::DType::F32),
        8 => Some(nn_core::DType::F64),
        13 => Some(nn_core::DType::BF16),
        _ => None,
    }
}

/// Parse an `ExportedProgram` from JSON bytes.
pub fn parse_exported_program(json: &[u8]) -> Result<ExportedProgram, crate::ImportError> {
    let program: ExportedProgram = serde_json::from_slice(json)?;
    if program.schema_version.major != 8 {
        return Err(crate::ImportError::UnsupportedSchema {
            major: program.schema_version.major,
            minor: program.schema_version.minor,
        });
    }
    Ok(program)
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;
