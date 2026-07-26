// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for grouped Conv1d compute shaders.
//!
//! Extends the basic Conv1d kernel in [`super::spirv_conv`] with:
//!
//! - **Groups support** (standard and depthwise convolution)
//! - **Separate bias buffer** (binding 2, optional via bias=true in config)
//! - **Config struct** ([`Conv1dConfig`]) for convolution parameters
//! - **CPU reference** ([`conv1d_reference`]) for differential verification
//!
//! # Buffer layout
//!
//! - **Binding 0** (set 0): Input `float[batch * in_channels * length]` (readonly)
//! - **Binding 1** (set 0): Weight `float[out_channels * (in_channels/groups) * kernel_size]` (readonly)
//! - **Binding 2** (set 0): Bias `float[out_channels]` (readonly)
//! - **Binding 3** (set 0): Output `float[batch * out_channels * out_length]`
//!
//! # Push constants
//!
//! ```text
//! { uint batch, uint in_channels, uint out_channels, uint length,
//!   uint kernel_size, uint stride, uint padding, uint dilation, uint groups }
//! ```
//!
//! # Dispatch
//!
//! One thread per output element. Dispatch `ceil(total_output / WORKGROUP_SIZE)` workgroups.
//! Each thread computes one output value at `[b, oc, ox]`.

use crate::spirv_emit::SPIRV_MAGIC;

/// Default workgroup size for Conv1d kernels (1D dispatch).
pub const CONV1D_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V constants ----

const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

const fn op(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}

// Opcodes.
const OP_CAPABILITY: u16 = 17;
const OP_MEMORY_MODEL: u16 = 14;
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;
const OP_TYPE_VOID: u16 = 19;
const OP_TYPE_BOOL: u16 = 20;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_TYPE_FUNCTION: u16 = 33;
const OP_CONSTANT: u16 = 43;
const OP_VARIABLE: u16 = 59;
const OP_LOAD: u16 = 61;
const OP_STORE: u16 = 62;
const OP_ACCESS_CHAIN: u16 = 65;
const OP_FUNCTION: u16 = 54;
const OP_FUNCTION_END: u16 = 56;
const OP_LABEL: u16 = 248;
const OP_RETURN: u16 = 253;
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LOOP_MERGE: u16 = 246;
const OP_PHI: u16 = 245;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_FADD: u16 = 129;
const OP_FMUL: u16 = 133;
const OP_U_LESS_THAN: u16 = 176;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_IADD: u16 = 128;
const OP_IMUL: u16 = 132;
const OP_U_DIV: u16 = 134;
const OP_U_MOD: u16 = 137;

// Decorations.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_NON_WRITABLE: u32 = 24;

// Built-ins.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;

// Storage classes.
const STORAGE_CLASS_INPUT: u32 = 1;
const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

// Execution model / mode.
const EXECUTION_MODEL_GL_COMPUTE: u32 = 5;
const EXECUTION_MODE_LOCAL_SIZE: u32 = 17;

// Capability.
const CAPABILITY_SHADER: u32 = 1;

// Memory model.
const ADDRESSING_LOGICAL: u32 = 0;
const MEMORY_MODEL_GLSL450: u32 = 1;

// Function control.
const FUNCTION_CONTROL_NONE: u32 = 0;

/// Encode a string as SPIR-V literal words (null-terminated, padded to 4-byte boundary).
fn encode_string(s: &str) -> Vec<u32> {
    let bytes = s.as_bytes();
    let word_count = (bytes.len() + 1).div_ceil(4);
    let mut words = vec![0u32; word_count];
    for (i, &b) in bytes.iter().enumerate() {
        let word_idx = i / 4;
        let byte_idx = i % 4;
        words[word_idx] |= u32::from(b) << (byte_idx * 8);
    }
    words
}

/// Convert a `Vec<u32>` SPIR-V module to `Vec<u8>` (little-endian).
fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for &w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// SPIR-V module builder (local copy for module independence).
struct SpirVBuilder {
    bound: u32,
    capabilities: Vec<u32>,
    extensions: Vec<u32>,
    memory_model: Vec<u32>,
    entry_points: Vec<u32>,
    execution_modes: Vec<u32>,
    annotations: Vec<u32>,
    type_declarations: Vec<u32>,
    functions: Vec<u32>,
}

impl SpirVBuilder {
    fn new() -> Self {
        Self {
            bound: 1,
            capabilities: Vec::new(),
            extensions: Vec::new(),
            memory_model: Vec::new(),
            entry_points: Vec::new(),
            execution_modes: Vec::new(),
            annotations: Vec::new(),
            type_declarations: Vec::new(),
            functions: Vec::new(),
        }
    }

    fn id(&mut self) -> u32 {
        let id = self.bound;
        self.bound += 1;
        id
    }

    fn capability(&mut self, cap: u32) {
        self.capabilities.push(op(2, OP_CAPABILITY));
        self.capabilities.push(cap);
    }

    fn memory_model_decl(&mut self, addressing: u32, model: u32) {
        self.memory_model.push(op(3, OP_MEMORY_MODEL));
        self.memory_model.push(addressing);
        self.memory_model.push(model);
    }

    fn entry_point_compute(&mut self, func_id: u32, name: &str, interface_ids: &[u32]) {
        let name_words = encode_string(name);
        let wc = 3 + name_words.len() as u16 + interface_ids.len() as u16;
        self.entry_points.push(op(wc, OP_ENTRY_POINT));
        self.entry_points.push(EXECUTION_MODEL_GL_COMPUTE);
        self.entry_points.push(func_id);
        self.entry_points.extend_from_slice(&name_words);
        self.entry_points.extend_from_slice(interface_ids);
    }

    fn execution_mode_local_size(&mut self, func_id: u32, x: u32, y: u32, z: u32) {
        self.execution_modes.push(op(6, OP_EXECUTION_MODE));
        self.execution_modes.push(func_id);
        self.execution_modes.push(EXECUTION_MODE_LOCAL_SIZE);
        self.execution_modes.push(x);
        self.execution_modes.push(y);
        self.execution_modes.push(z);
    }

    fn decorate(&mut self, target: u32, decoration: u32, operands: &[u32]) {
        let wc = 3 + operands.len() as u16;
        self.annotations.push(op(wc, OP_DECORATE));
        self.annotations.push(target);
        self.annotations.push(decoration);
        self.annotations.extend_from_slice(operands);
    }

    fn member_decorate(
        &mut self,
        struct_type: u32,
        member: u32,
        decoration: u32,
        operands: &[u32],
    ) {
        let wc = 4 + operands.len() as u16;
        self.annotations.push(op(wc, OP_MEMBER_DECORATE));
        self.annotations.push(struct_type);
        self.annotations.push(member);
        self.annotations.push(decoration);
        self.annotations.extend_from_slice(operands);
    }

    // ---- Type declarations ----

    fn type_void(&mut self) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(2, OP_TYPE_VOID));
        self.type_declarations.push(result);
        result
    }

    fn type_bool(&mut self) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(2, OP_TYPE_BOOL));
        self.type_declarations.push(result);
        result
    }

    fn type_int(&mut self, width: u32, signedness: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_INT));
        self.type_declarations.push(result);
        self.type_declarations.push(width);
        self.type_declarations.push(signedness);
        result
    }

    fn type_float(&mut self, width: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(3, OP_TYPE_FLOAT));
        self.type_declarations.push(result);
        self.type_declarations.push(width);
        result
    }

    fn type_vector(&mut self, component_type: u32, count: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_VECTOR));
        self.type_declarations.push(result);
        self.type_declarations.push(component_type);
        self.type_declarations.push(count);
        result
    }

    fn type_runtime_array(&mut self, element_type: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(3, OP_TYPE_RUNTIME_ARRAY));
        self.type_declarations.push(result);
        self.type_declarations.push(element_type);
        result
    }

    fn type_struct(&mut self, member_types: &[u32]) -> u32 {
        let result = self.id();
        let wc = 2 + member_types.len() as u16;
        self.type_declarations.push(op(wc, OP_TYPE_STRUCT));
        self.type_declarations.push(result);
        self.type_declarations.extend_from_slice(member_types);
        result
    }

    fn type_pointer(&mut self, storage_class: u32, pointee_type: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_TYPE_POINTER));
        self.type_declarations.push(result);
        self.type_declarations.push(storage_class);
        self.type_declarations.push(pointee_type);
        result
    }

    fn type_function(&mut self, return_type: u32, param_types: &[u32]) -> u32 {
        let result = self.id();
        let wc = 3 + param_types.len() as u16;
        self.type_declarations.push(op(wc, OP_TYPE_FUNCTION));
        self.type_declarations.push(result);
        self.type_declarations.push(return_type);
        self.type_declarations.extend_from_slice(param_types);
        result
    }

    fn constant_u32(&mut self, type_id: u32, value: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value);
        result
    }

    fn constant_f32(&mut self, type_id: u32, value: f32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_CONSTANT));
        self.type_declarations.push(type_id);
        self.type_declarations.push(result);
        self.type_declarations.push(value.to_bits());
        result
    }

    fn variable_global(&mut self, ptr_type: u32, storage_class: u32) -> u32 {
        let result = self.id();
        self.type_declarations.push(op(4, OP_VARIABLE));
        self.type_declarations.push(ptr_type);
        self.type_declarations.push(result);
        self.type_declarations.push(storage_class);
        result
    }

    // ---- Function body instructions ----

    fn func_begin(&mut self, result_type: u32, func_id: u32, control: u32, func_type: u32) {
        self.functions.push(op(5, OP_FUNCTION));
        self.functions.push(result_type);
        self.functions.push(func_id);
        self.functions.push(control);
        self.functions.push(func_type);
    }

    fn func_end(&mut self) {
        self.functions.push(op(1, OP_FUNCTION_END));
    }

    fn label(&mut self) -> u32 {
        let result = self.id();
        self.functions.push(op(2, OP_LABEL));
        self.functions.push(result);
        result
    }

    fn label_with_id(&mut self, id: u32) {
        self.functions.push(op(2, OP_LABEL));
        self.functions.push(id);
    }

    fn op_return(&mut self) {
        self.functions.push(op(1, OP_RETURN));
    }

    fn branch(&mut self, target_label: u32) {
        self.functions.push(op(2, OP_BRANCH));
        self.functions.push(target_label);
    }

    fn branch_conditional(&mut self, condition: u32, true_label: u32, false_label: u32) {
        self.functions.push(op(4, OP_BRANCH_CONDITIONAL));
        self.functions.push(condition);
        self.functions.push(true_label);
        self.functions.push(false_label);
    }

    fn selection_merge(&mut self, merge_label: u32) {
        self.functions.push(op(3, OP_SELECTION_MERGE));
        self.functions.push(merge_label);
        self.functions.push(0);
    }

    fn loop_merge(&mut self, merge_label: u32, continue_label: u32) {
        self.functions.push(op(4, OP_LOOP_MERGE));
        self.functions.push(merge_label);
        self.functions.push(continue_label);
        self.functions.push(0);
    }

    fn phi(&mut self, result_type: u32, operands: &[(u32, u32)]) -> u32 {
        let result = self.id();
        let wc = 3 + (operands.len() as u16) * 2;
        self.functions.push(op(wc, OP_PHI));
        self.functions.push(result_type);
        self.functions.push(result);
        for &(val, parent) in operands {
            self.functions.push(val);
            self.functions.push(parent);
        }
        result
    }

    fn load(&mut self, result_type: u32, pointer: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(4, OP_LOAD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(pointer);
        result
    }

    fn store(&mut self, pointer: u32, object: u32) {
        self.functions.push(op(3, OP_STORE));
        self.functions.push(pointer);
        self.functions.push(object);
    }

    fn access_chain(&mut self, result_type: u32, base: u32, indices: &[u32]) -> u32 {
        let result = self.id();
        let wc = 4 + indices.len() as u16;
        self.functions.push(op(wc, OP_ACCESS_CHAIN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(base);
        self.functions.extend_from_slice(indices);
        result
    }

    fn composite_extract(&mut self, result_type: u32, composite: u32, index: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_COMPOSITE_EXTRACT));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(composite);
        self.functions.push(index);
        result
    }

    fn fadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn fmul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FMUL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn u_less_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_LESS_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn u_greater_than_equal(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_GREATER_THAN_EQUAL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn iadd(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IADD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn imul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IMUL));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn u_div(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_DIV));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn u_mod(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_MOD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn isub(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        let op_isub: u16 = 130;
        self.functions.push(op(5, op_isub));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(512);
        module.push(SPIRV_MAGIC);
        module.push(SPIRV_VERSION_1_0);
        module.push(GENERATOR_MAGIC);
        module.push(self.bound);
        module.push(0);
        module.extend_from_slice(&self.capabilities);
        module.extend_from_slice(&self.extensions);
        module.extend_from_slice(&self.memory_model);
        module.extend_from_slice(&self.entry_points);
        module.extend_from_slice(&self.execution_modes);
        module.extend_from_slice(&self.annotations);
        module.extend_from_slice(&self.type_declarations);
        module.extend_from_slice(&self.functions);
        module
    }
}

/// Fixup a phi instruction to add an additional (value, parent) operand.
fn fixup_phi(functions: &mut Vec<u32>, phi_id: u32, value: u32, parent: u32) {
    let mut pos = 0;
    while pos < functions.len() {
        let word = functions[pos];
        let word_count = (word >> 16) as usize;
        let opcode_val = (word & 0xFFFF) as u16;
        if word_count == 0 {
            break;
        }
        if opcode_val == OP_PHI && pos + 2 < functions.len() && functions[pos + 2] == phi_id {
            let insert_pos = pos + word_count;
            functions.insert(insert_pos, parent);
            functions.insert(insert_pos, value);
            let new_wc = word_count + 2;
            functions[pos] = op(new_wc as u16, OP_PHI);
            return;
        }
        pos += word_count;
    }
}

// ============================================================================
// Conv1d configuration
// ============================================================================

/// Configuration for a grouped Conv1d kernel.
///
/// Supports standard convolution, grouped convolution, and depthwise convolution.
/// Depthwise is the special case where `groups == in_channels == out_channels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dConfig {
    /// Number of input channels.
    pub in_channels: u32,
    /// Number of output channels.
    pub out_channels: u32,
    /// Convolution kernel size.
    pub kernel_size: u32,
    /// Stride of the convolution.
    pub stride: u32,
    /// Zero-padding added to both sides of the input.
    pub padding: u32,
    /// Spacing between kernel elements.
    pub dilation: u32,
    /// Number of groups for grouped convolution.
    /// `in_channels` and `out_channels` must both be divisible by `groups`.
    /// When `groups == in_channels == out_channels`, this is depthwise convolution.
    pub groups: u32,
}

impl Conv1dConfig {
    /// Create a new Conv1d configuration with default stride=1, padding=0,
    /// dilation=1, groups=1.
    pub fn new(in_channels: u32, out_channels: u32, kernel_size: u32) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
        }
    }

    /// Set stride.
    #[must_use]
    pub fn stride(mut self, stride: u32) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding.
    #[must_use]
    pub fn padding(mut self, padding: u32) -> Self {
        self.padding = padding;
        self
    }

    /// Set dilation.
    #[must_use]
    pub fn dilation(mut self, dilation: u32) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set groups.
    #[must_use]
    pub fn groups(mut self, groups: u32) -> Self {
        self.groups = groups;
        self
    }

    /// Validate the configuration. Returns `Err` with a description if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if self.in_channels == 0 {
            return Err("in_channels must be > 0".to_string());
        }
        if self.out_channels == 0 {
            return Err("out_channels must be > 0".to_string());
        }
        if self.kernel_size == 0 {
            return Err("kernel_size must be > 0".to_string());
        }
        if self.stride == 0 {
            return Err("stride must be > 0".to_string());
        }
        if self.dilation == 0 {
            return Err("dilation must be > 0".to_string());
        }
        if self.groups == 0 {
            return Err("groups must be > 0".to_string());
        }
        if !self.in_channels.is_multiple_of(self.groups) {
            return Err(format!(
                "in_channels ({}) must be divisible by groups ({})",
                self.in_channels, self.groups
            ));
        }
        if !self.out_channels.is_multiple_of(self.groups) {
            return Err(format!(
                "out_channels ({}) must be divisible by groups ({})",
                self.out_channels, self.groups
            ));
        }
        Ok(())
    }

    /// Compute the output length for a given input length.
    ///
    /// `out_length = (length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
    pub fn output_length(&self, length: usize) -> usize {
        let effective_ks = self.dilation as usize * (self.kernel_size as usize - 1) + 1;
        let padded = length + 2 * self.padding as usize;
        if padded < effective_ks {
            return 0;
        }
        (padded - effective_ks) / self.stride as usize + 1
    }

    /// Whether this is a depthwise configuration (groups == in_channels == out_channels).
    pub fn is_depthwise(&self) -> bool {
        self.groups == self.in_channels && self.groups == self.out_channels
    }
}

// ============================================================================
// SPIR-V generation
// ============================================================================

/// Generate a SPIR-V 1.0 binary module for grouped Conv1d with bias.
///
/// The kernel computes grouped 1D convolution with an explicit bias buffer.
/// Each thread computes one output element at `[batch, out_channel, out_pos]`.
///
/// # Layout
///
/// - **Binding 0** (set 0): Input buffer `float[]` (readonly)
/// - **Binding 1** (set 0): Weight buffer `float[]` (readonly)
/// - **Binding 2** (set 0): Bias buffer `float[]` (readonly)
/// - **Binding 3** (set 0): Output buffer `float[]`
/// - **Push constants**: 9 x uint32 (batch, in_channels, out_channels, length,
///   kernel_size, stride, padding, dilation, groups)
///
/// # Arguments
///
/// * `config` - Convolution configuration (validated before generation).
///
/// # Panics
///
/// Panics if `config.validate()` fails.
pub fn generate_conv1d_grouped_spirv(config: &Conv1dConfig) -> Vec<u8> {
    config.validate().expect("Conv1dConfig validation failed");

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays for buffers.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (readonly).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Weight buffer struct (readonly).
    let ty_struct_weight = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_weight, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_weight, 0, DECORATION_OFFSET, &[0]);

    // Bias buffer struct (readonly).
    let ty_struct_bias = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_bias, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_bias, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct.
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: 9 x uint.
    let pc_members = vec![ty_uint; 9];
    let ty_struct_pc = b.type_struct(&pc_members);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    for i in 0..9u32 {
        b.member_decorate(ty_struct_pc, i, DECORATION_OFFSET, &[i * 4]);
    }

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_weight = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_weight);
    let ptr_sb_bias = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_bias);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_3u = b.constant_u32(ty_uint, 3);
    let const_4u = b.constant_u32(ty_uint, 4);
    let const_5u = b.constant_u32(ty_uint, 5);
    let const_6u = b.constant_u32(ty_uint, 6);
    let const_7u = b.constant_u32(ty_uint, 7);
    let const_8u = b.constant_u32(ty_uint, 8);
    let _const_f0 = b.constant_f32(ty_float, 0.0);

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);
    b.decorate(var_input, DECORATION_NON_WRITABLE, &[]);

    let var_weight = b.variable_global(ptr_sb_weight, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_weight, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_weight, DECORATION_BINDING, &[1]);
    b.decorate(var_weight, DECORATION_NON_WRITABLE, &[]);

    let var_bias = b.variable_global(ptr_sb_bias, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_bias, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_bias, DECORATION_BINDING, &[2]);
    b.decorate(var_bias, DECORATION_NON_WRITABLE, &[]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[3]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, CONV1D_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load global invocation ID.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants.
    // 0: batch, 1: in_channels, 2: out_channels, 3: length,
    // 4: kernel_size, 5: stride, 6: padding, 7: dilation, 8: groups
    let pc_ptrs: Vec<u32> = (0..9u32)
        .map(|i| {
            let idx = match i {
                0 => const_0u,
                1 => const_1u,
                2 => const_2u,
                3 => const_3u,
                4 => const_4u,
                5 => const_5u,
                6 => const_6u,
                7 => const_7u,
                8 => const_8u,
                _ => unreachable!(),
            };
            b.access_chain(ptr_pc_uint, var_pc, &[idx])
        })
        .collect();

    let dim_batch = b.load(ty_uint, pc_ptrs[0]);
    let dim_in_ch = b.load(ty_uint, pc_ptrs[1]);
    let dim_out_ch = b.load(ty_uint, pc_ptrs[2]);
    let dim_length = b.load(ty_uint, pc_ptrs[3]);
    let dim_ks = b.load(ty_uint, pc_ptrs[4]);
    let dim_stride = b.load(ty_uint, pc_ptrs[5]);
    let dim_padding = b.load(ty_uint, pc_ptrs[6]);
    let dim_dilation = b.load(ty_uint, pc_ptrs[7]);
    let dim_groups = b.load(ty_uint, pc_ptrs[8]);

    // Compute out_length = (length + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
    let two_padding = b.imul(ty_uint, const_2u, dim_padding);
    let padded_len = b.iadd(ty_uint, dim_length, two_padding);
    let ks_minus_1 = b.isub(ty_uint, dim_ks, const_1u);
    let dil_times_ks_m1 = b.imul(ty_uint, dim_dilation, ks_minus_1);
    let effective_ks = b.iadd(ty_uint, dil_times_ks_m1, const_1u);
    let numerator = b.isub(ty_uint, padded_len, effective_ks);
    let out_length = b.u_div(ty_uint, numerator, dim_stride);
    let out_length = b.iadd(ty_uint, out_length, const_1u);

    // Compute total output elements = batch * out_channels * out_length
    let batch_times_oc = b.imul(ty_uint, dim_batch, dim_out_ch);
    let total_output = b.imul(ty_uint, batch_times_oc, out_length);

    // Bounds check: if gid_x >= total_output, return early.
    let oob = b.u_greater_than_equal(ty_bool, gid_x, total_output);
    let body_label = b.id();
    let return_label = b.id();
    b.selection_merge(return_label);
    b.branch_conditional(oob, return_label, body_label);

    b.label_with_id(body_label);

    // Decompose gid_x into [batch_idx, oc, ox]:
    //   ox = gid_x % out_length
    //   tmp = gid_x / out_length
    //   oc = tmp % out_channels
    //   batch_idx = tmp / out_channels
    let ox = b.u_mod(ty_uint, gid_x, out_length);
    let tmp = b.u_div(ty_uint, gid_x, out_length);
    let oc = b.u_mod(ty_uint, tmp, dim_out_ch);
    let batch_idx = b.u_div(ty_uint, tmp, dim_out_ch);

    // Grouped conv parameters:
    //   ic_per_group = in_channels / groups
    //   oc_per_group = out_channels / groups
    //   group_idx = oc / oc_per_group
    //   ic_start = group_idx * ic_per_group
    let ic_per_group = b.u_div(ty_uint, dim_in_ch, dim_groups);
    let oc_per_group = b.u_div(ty_uint, dim_out_ch, dim_groups);
    let group_idx = b.u_div(ty_uint, oc, oc_per_group);
    let ic_start = b.imul(ty_uint, group_idx, ic_per_group);

    // Load bias[oc].
    let bias_ptr = b.access_chain(ptr_sb_float, var_bias, &[const_0u, oc]);
    let bias_val = b.load(ty_float, bias_ptr);

    // Inner loop: accumulate over ic_per_group and kernel_size.
    // For each ic in [ic_start, ic_start + ic_per_group):
    //   For each k in [0, kernel_size):
    //     ix = ox * stride + k * dilation - padding
    //     if ix >= 0 && ix < length:
    //       input_idx = batch_idx * in_channels * length + (ic_start + ic_local) * length + ix
    //       weight_idx = oc * ic_per_group * kernel_size + ic_local * kernel_size + k
    //       acc += input[input_idx] * weight[weight_idx]

    // Outer loop over ic_local in [0, ic_per_group)
    let ic_loop_header = b.id();
    let ic_loop_body = b.id();
    let ic_loop_continue = b.id();
    let ic_loop_merge = b.id();

    b.branch(ic_loop_header);
    b.label_with_id(ic_loop_header);
    b.loop_merge(ic_loop_merge, ic_loop_continue);

    let phi_ic = b.phi(ty_uint, &[(const_0u, body_label)]);
    let phi_acc_outer = b.phi(ty_float, &[(bias_val, body_label)]);

    let ic_cond = b.u_less_than(ty_bool, phi_ic, ic_per_group);
    b.branch_conditional(ic_cond, ic_loop_body, ic_loop_merge);

    b.label_with_id(ic_loop_body);

    // Current input channel = ic_start + ic_local
    let ic_actual = b.iadd(ty_uint, ic_start, phi_ic);

    // Inner loop over k in [0, kernel_size)
    let k_loop_header = b.id();
    let k_loop_body = b.id();
    let k_loop_continue = b.id();
    let k_loop_merge = b.id();

    b.branch(k_loop_header);
    b.label_with_id(k_loop_header);
    b.loop_merge(k_loop_merge, k_loop_continue);

    let phi_k = b.phi(ty_uint, &[(const_0u, ic_loop_body)]);
    let phi_acc_inner = b.phi(ty_float, &[(phi_acc_outer, ic_loop_body)]);

    let k_cond = b.u_less_than(ty_bool, phi_k, dim_ks);
    b.branch_conditional(k_cond, k_loop_body, k_loop_merge);

    b.label_with_id(k_loop_body);

    // Compute ix = ox * stride + k * dilation - padding
    let ox_times_stride = b.imul(ty_uint, ox, dim_stride);
    let k_times_dilation = b.imul(ty_uint, phi_k, dim_dilation);
    let pos_sum = b.iadd(ty_uint, ox_times_stride, k_times_dilation);

    // Bounds check: ix = pos_sum - padding. We need pos_sum >= padding and ix < length.
    // Since we use unsigned arithmetic, check pos_sum >= padding first.
    let in_bounds_low = b.u_greater_than_equal(ty_bool, pos_sum, dim_padding);
    let ix = b.isub(ty_uint, pos_sum, dim_padding);
    let in_bounds_high = b.u_less_than(ty_bool, ix, dim_length);

    // Combined bounds: in_bounds_low AND in_bounds_high
    // SPIR-V doesn't have OpLogicalAnd directly in this builder, so we use
    // nested selection.
    let in_bounds_label = b.id();
    let check_high_label = b.id();
    let skip_label = b.id();

    b.selection_merge(skip_label);
    b.branch_conditional(in_bounds_low, check_high_label, skip_label);

    b.label_with_id(check_high_label);
    b.selection_merge(skip_label);
    b.branch_conditional(in_bounds_high, in_bounds_label, skip_label);

    b.label_with_id(in_bounds_label);

    // input_idx = batch_idx * in_channels * length + ic_actual * length + ix
    let batch_stride_in = b.imul(ty_uint, dim_in_ch, dim_length);
    let batch_offset = b.imul(ty_uint, batch_idx, batch_stride_in);
    let ch_offset = b.imul(ty_uint, ic_actual, dim_length);
    let input_idx = b.iadd(ty_uint, batch_offset, ch_offset);
    let input_idx = b.iadd(ty_uint, input_idx, ix);

    // weight_idx = oc * ic_per_group * kernel_size + ic_local * kernel_size + k
    let oc_weight_stride = b.imul(ty_uint, ic_per_group, dim_ks);
    let oc_weight_offset = b.imul(ty_uint, oc, oc_weight_stride);
    let ic_weight_offset = b.imul(ty_uint, phi_ic, dim_ks);
    let weight_idx = b.iadd(ty_uint, oc_weight_offset, ic_weight_offset);
    let weight_idx = b.iadd(ty_uint, weight_idx, phi_k);

    // Load input and weight, multiply-accumulate.
    let in_ptr = b.access_chain(ptr_sb_float, var_input, &[const_0u, input_idx]);
    let in_val = b.load(ty_float, in_ptr);
    let w_ptr = b.access_chain(ptr_sb_float, var_weight, &[const_0u, weight_idx]);
    let w_val = b.load(ty_float, w_ptr);
    let prod = b.fmul(ty_float, in_val, w_val);
    let new_acc = b.fadd(ty_float, phi_acc_inner, prod);

    b.branch(skip_label);

    // skip_label: phi to select updated or unchanged accumulator
    b.label_with_id(skip_label);
    let phi_acc_after_k = b.phi(
        ty_float,
        &[
            (new_acc, in_bounds_label),
            (phi_acc_inner, k_loop_body),
            (phi_acc_inner, check_high_label),
        ],
    );

    b.branch(k_loop_continue);

    b.label_with_id(k_loop_continue);
    let next_k = b.iadd(ty_uint, phi_k, const_1u);
    b.branch(k_loop_header);

    fixup_phi(&mut b.functions, phi_k, next_k, k_loop_continue);
    fixup_phi(
        &mut b.functions,
        phi_acc_inner,
        phi_acc_after_k,
        k_loop_continue,
    );

    b.label_with_id(k_loop_merge);

    // After inner k loop, continue to next ic
    b.branch(ic_loop_continue);

    b.label_with_id(ic_loop_continue);
    let next_ic = b.iadd(ty_uint, phi_ic, const_1u);
    b.branch(ic_loop_header);

    fixup_phi(&mut b.functions, phi_ic, next_ic, ic_loop_continue);
    fixup_phi(
        &mut b.functions,
        phi_acc_outer,
        phi_acc_inner,
        ic_loop_continue,
    );

    b.label_with_id(ic_loop_merge);

    // Store result to output[gid_x].
    let out_ptr = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(out_ptr, phi_acc_outer);

    b.branch(return_label);

    b.label_with_id(return_label);
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

// ============================================================================
// CPU reference implementation
// ============================================================================

/// Compute Conv1d on CPU for reference/verification.
///
/// # Arguments
///
/// * `input` - Flat `[batch, in_channels, length]` in row-major order.
/// * `weight` - Flat `[out_channels, in_channels/groups, kernel_size]` in row-major order.
/// * `bias` - Flat `[out_channels]` bias vector.
/// * `config` - Convolution configuration.
/// * `batch` - Batch size.
/// * `length` - Input spatial length.
///
/// # Returns
///
/// Flat `[batch, out_channels, out_length]` output in row-major order.
///
/// # Panics
///
/// Panics if config validation fails or input sizes are inconsistent.
pub fn conv1d_reference(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    config: &Conv1dConfig,
    batch: usize,
    length: usize,
) -> Vec<f32> {
    config.validate().expect("Conv1dConfig validation failed");

    let in_ch = config.in_channels as usize;
    let out_ch = config.out_channels as usize;
    let ks = config.kernel_size as usize;
    let stride = config.stride as usize;
    let padding = config.padding as usize;
    let dilation = config.dilation as usize;
    let groups = config.groups as usize;

    assert_eq!(input.len(), batch * in_ch * length);
    assert_eq!(weight.len(), out_ch * (in_ch / groups) * ks);
    assert_eq!(bias.len(), out_ch);

    let out_length = config.output_length(length);
    let ic_per_group = in_ch / groups;
    let oc_per_group = out_ch / groups;

    let mut output = vec![0.0f32; batch * out_ch * out_length];

    for b_idx in 0..batch {
        for oc in 0..out_ch {
            let group_idx = oc / oc_per_group;
            let ic_start = group_idx * ic_per_group;

            for ox in 0..out_length {
                let mut acc = bias[oc];

                for ic_local in 0..ic_per_group {
                    let ic = ic_start + ic_local;
                    for k in 0..ks {
                        let ix_pos = ox * stride + k * dilation;
                        if ix_pos >= padding {
                            let ix = ix_pos - padding;
                            if ix < length {
                                let in_idx = b_idx * in_ch * length + ic * length + ix;
                                let w_idx = oc * ic_per_group * ks + ic_local * ks + k;
                                acc += input[in_idx] * weight[w_idx];
                            }
                        }
                    }
                }

                let out_idx = b_idx * out_ch * out_length + oc * out_length + ox;
                output[out_idx] = acc;
            }
        }
    }

    output
}

#[cfg(test)]
#[path = "spirv_conv1d_tests.rs"]
mod spirv_conv1d_tests;
