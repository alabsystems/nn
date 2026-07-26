// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for 2D convolution compute shaders.
//!
//! Extends the 1D convolution infrastructure in [`super::spirv_conv`] and
//! [`super::spirv_conv1d`] with 2D spatial support for vision models.
//!
//! - **Groups support** (standard, grouped, and depthwise convolution)
//! - **Separate bias buffer** (binding 2, always present)
//! - **Config struct** ([`Conv2dConfig`]) for all convolution parameters
//! - **CPU reference** ([`conv2d_reference`]) for differential verification
//!
//! # Buffer layout
//!
//! - **Binding 0** (set 0): Input `float[batch * in_channels * in_h * in_w]` (readonly)
//! - **Binding 1** (set 0): Weight `float[out_channels * (in_channels/groups) * kh * kw]` (readonly)
//! - **Binding 2** (set 0): Bias `float[out_channels]` (readonly)
//! - **Binding 3** (set 0): Output `float[batch * out_channels * out_h * out_w]`
//!
//! # Push constants
//!
//! ```text
//! { uint batch, uint in_channels, uint out_channels, uint in_h, uint in_w,
//!   uint kernel_h, uint kernel_w, uint stride_h, uint stride_w,
//!   uint padding_h, uint padding_w, uint groups }
//! ```
//!
//! # Dispatch
//!
//! One thread per output element. Dispatch `ceil(total_output / WORKGROUP_SIZE)` workgroups.

use crate::spirv_emit::SPIRV_MAGIC;

/// Default workgroup size for Conv2d kernels (1D dispatch).
pub const CONV2D_WORKGROUP_SIZE: u32 = 64;

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
        let mut module = Vec::with_capacity(1024);
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
// Conv2d configuration
// ============================================================================

/// Configuration for a 2D convolution kernel.
///
/// Supports standard convolution, grouped convolution, and depthwise convolution.
/// Depthwise is the special case where `groups == in_channels == out_channels`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dConfig {
    /// Number of input channels.
    pub in_channels: u32,
    /// Number of output channels.
    pub out_channels: u32,
    /// Kernel height.
    pub kernel_h: u32,
    /// Kernel width.
    pub kernel_w: u32,
    /// Stride in the height dimension.
    pub stride_h: u32,
    /// Stride in the width dimension.
    pub stride_w: u32,
    /// Zero-padding added to both sides in height.
    pub padding_h: u32,
    /// Zero-padding added to both sides in width.
    pub padding_w: u32,
    /// Number of groups for grouped convolution.
    pub groups: u32,
}

impl Conv2dConfig {
    /// Create a new Conv2d configuration with default stride=1, padding=0, groups=1.
    pub fn new(in_channels: u32, out_channels: u32, kernel_h: u32, kernel_w: u32) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h: 1,
            stride_w: 1,
            padding_h: 0,
            padding_w: 0,
            groups: 1,
        }
    }

    /// Set stride (same in both dimensions).
    #[must_use]
    pub fn stride(mut self, stride_h: u32, stride_w: u32) -> Self {
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    /// Set padding (same in both dimensions).
    #[must_use]
    pub fn padding(mut self, padding_h: u32, padding_w: u32) -> Self {
        self.padding_h = padding_h;
        self.padding_w = padding_w;
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
        if self.kernel_h == 0 {
            return Err("kernel_h must be > 0".to_string());
        }
        if self.kernel_w == 0 {
            return Err("kernel_w must be > 0".to_string());
        }
        if self.stride_h == 0 {
            return Err("stride_h must be > 0".to_string());
        }
        if self.stride_w == 0 {
            return Err("stride_w must be > 0".to_string());
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

    /// Whether this is a depthwise configuration (groups == in_channels == out_channels).
    pub fn is_depthwise(&self) -> bool {
        self.groups == self.in_channels && self.groups == self.out_channels
    }
}

/// Compute the output spatial dimensions for a 2D convolution.
///
/// Returns `(out_h, out_w)`.
pub fn conv2d_output_size(in_h: usize, in_w: usize, config: &Conv2dConfig) -> (usize, usize) {
    let out_h = (in_h + 2 * config.padding_h as usize - config.kernel_h as usize)
        / config.stride_h as usize
        + 1;
    let out_w = (in_w + 2 * config.padding_w as usize - config.kernel_w as usize)
        / config.stride_w as usize
        + 1;
    (out_h, out_w)
}

// ============================================================================
// SPIR-V generation
// ============================================================================

/// Generate a SPIR-V 1.0 binary module (as `Vec<u32>`) for 2D convolution with bias.
///
/// The kernel computes grouped 2D convolution with an explicit bias buffer.
/// Each thread computes one output element at `[batch, out_channel, oy, ox]`.
///
/// # Layout
///
/// - **Binding 0** (set 0): Input buffer `float[]` (readonly)
/// - **Binding 1** (set 0): Weight buffer `float[]` (readonly)
/// - **Binding 2** (set 0): Bias buffer `float[]` (readonly)
/// - **Binding 3** (set 0): Output buffer `float[]`
/// - **Push constants**: 12 x uint32
///
/// # Panics
///
/// Panics if `config.validate()` fails.
pub fn generate_conv2d_spirv(config: &Conv2dConfig) -> Vec<u32> {
    config.validate().expect("Conv2dConfig validation failed");

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

    // Buffer structs.
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_weight = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_weight, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_weight, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_bias = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_bias, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_bias, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: 12 x uint.
    let pc_count = 12u32;
    let pc_members = vec![ty_uint; pc_count as usize];
    let ty_struct_pc = b.type_struct(&pc_members);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    for i in 0..pc_count {
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
    let const_u: Vec<u32> = (0..pc_count).map(|i| b.constant_u32(ty_uint, i)).collect();
    let _const_f0 = b.constant_f32(ty_float, 0.0);
    let const_0u = const_u[0];
    let const_1u = const_u[1];
    let const_2u = const_u[2];

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
    b.execution_mode_local_size(func_id, CONV2D_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load global invocation ID.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants:
    // 0: batch, 1: in_channels, 2: out_channels, 3: in_h, 4: in_w,
    // 5: kernel_h, 6: kernel_w, 7: stride_h, 8: stride_w,
    // 9: padding_h, 10: padding_w, 11: groups
    let pc_ptrs: Vec<u32> = (0..pc_count)
        .map(|i| b.access_chain(ptr_pc_uint, var_pc, &[const_u[i as usize]]))
        .collect();

    let dim_batch = b.load(ty_uint, pc_ptrs[0]);
    let dim_in_ch = b.load(ty_uint, pc_ptrs[1]);
    let dim_out_ch = b.load(ty_uint, pc_ptrs[2]);
    let dim_in_h = b.load(ty_uint, pc_ptrs[3]);
    let dim_in_w = b.load(ty_uint, pc_ptrs[4]);
    let dim_kh = b.load(ty_uint, pc_ptrs[5]);
    let dim_kw = b.load(ty_uint, pc_ptrs[6]);
    let dim_stride_h = b.load(ty_uint, pc_ptrs[7]);
    let dim_stride_w = b.load(ty_uint, pc_ptrs[8]);
    let dim_pad_h = b.load(ty_uint, pc_ptrs[9]);
    let dim_pad_w = b.load(ty_uint, pc_ptrs[10]);
    let dim_groups = b.load(ty_uint, pc_ptrs[11]);

    // Compute out_h = (in_h + 2*pad_h - kh) / stride_h + 1
    let two_pad_h = b.imul(ty_uint, const_2u, dim_pad_h);
    let padded_h = b.iadd(ty_uint, dim_in_h, two_pad_h);
    let num_h = b.isub(ty_uint, padded_h, dim_kh);
    let out_h = b.u_div(ty_uint, num_h, dim_stride_h);
    let out_h = b.iadd(ty_uint, out_h, const_1u);

    // Compute out_w = (in_w + 2*pad_w - kw) / stride_w + 1
    let two_pad_w = b.imul(ty_uint, const_2u, dim_pad_w);
    let padded_w = b.iadd(ty_uint, dim_in_w, two_pad_w);
    let num_w = b.isub(ty_uint, padded_w, dim_kw);
    let out_w = b.u_div(ty_uint, num_w, dim_stride_w);
    let out_w = b.iadd(ty_uint, out_w, const_1u);

    // total_output = batch * out_channels * out_h * out_w
    let batch_times_oc = b.imul(ty_uint, dim_batch, dim_out_ch);
    let boc_times_oh = b.imul(ty_uint, batch_times_oc, out_h);
    let total_output = b.imul(ty_uint, boc_times_oh, out_w);

    // Bounds check: if gid_x >= total_output, return early.
    let oob = b.u_greater_than_equal(ty_bool, gid_x, total_output);
    let body_label = b.id();
    let return_label = b.id();
    b.selection_merge(return_label);
    b.branch_conditional(oob, return_label, body_label);

    b.label_with_id(body_label);

    // Decompose gid_x into [batch_idx, oc, oy, ox]:
    //   ox = gid_x % out_w
    //   tmp1 = gid_x / out_w
    //   oy = tmp1 % out_h
    //   tmp2 = tmp1 / out_h
    //   oc = tmp2 % out_channels
    //   batch_idx = tmp2 / out_channels
    let ox = b.u_mod(ty_uint, gid_x, out_w);
    let tmp1 = b.u_div(ty_uint, gid_x, out_w);
    let oy = b.u_mod(ty_uint, tmp1, out_h);
    let tmp2 = b.u_div(ty_uint, tmp1, out_h);
    let oc = b.u_mod(ty_uint, tmp2, dim_out_ch);
    let batch_idx = b.u_div(ty_uint, tmp2, dim_out_ch);

    // Grouped conv parameters:
    let ic_per_group = b.u_div(ty_uint, dim_in_ch, dim_groups);
    let oc_per_group = b.u_div(ty_uint, dim_out_ch, dim_groups);
    let group_idx = b.u_div(ty_uint, oc, oc_per_group);
    let ic_start = b.imul(ty_uint, group_idx, ic_per_group);

    // Load bias[oc].
    let bias_ptr = b.access_chain(ptr_sb_float, var_bias, &[const_0u, oc]);
    let bias_val = b.load(ty_float, bias_ptr);

    // ---- Triple nested loop: ic_local, kh, kw ----
    // Outer loop over ic_local in [0, ic_per_group)
    let ic_loop_header = b.id();
    let ic_loop_body = b.id();
    let ic_loop_continue = b.id();
    let ic_loop_merge = b.id();

    b.branch(ic_loop_header);
    b.label_with_id(ic_loop_header);
    b.loop_merge(ic_loop_merge, ic_loop_continue);

    let phi_ic = b.phi(ty_uint, &[(const_0u, body_label)]);
    let phi_acc_ic = b.phi(ty_float, &[(bias_val, body_label)]);

    let ic_cond = b.u_less_than(ty_bool, phi_ic, ic_per_group);
    b.branch_conditional(ic_cond, ic_loop_body, ic_loop_merge);

    b.label_with_id(ic_loop_body);
    let ic_actual = b.iadd(ty_uint, ic_start, phi_ic);

    // Middle loop over kh_idx in [0, kernel_h)
    let kh_loop_header = b.id();
    let kh_loop_body = b.id();
    let kh_loop_continue = b.id();
    let kh_loop_merge = b.id();

    b.branch(kh_loop_header);
    b.label_with_id(kh_loop_header);
    b.loop_merge(kh_loop_merge, kh_loop_continue);

    let phi_kh = b.phi(ty_uint, &[(const_0u, ic_loop_body)]);
    let phi_acc_kh = b.phi(ty_float, &[(phi_acc_ic, ic_loop_body)]);

    let kh_cond = b.u_less_than(ty_bool, phi_kh, dim_kh);
    b.branch_conditional(kh_cond, kh_loop_body, kh_loop_merge);

    b.label_with_id(kh_loop_body);

    // Inner loop over kw_idx in [0, kernel_w)
    let kw_loop_header = b.id();
    let kw_loop_body = b.id();
    let kw_loop_continue = b.id();
    let kw_loop_merge = b.id();

    b.branch(kw_loop_header);
    b.label_with_id(kw_loop_header);
    b.loop_merge(kw_loop_merge, kw_loop_continue);

    let phi_kw = b.phi(ty_uint, &[(const_0u, kh_loop_body)]);
    let phi_acc_kw = b.phi(ty_float, &[(phi_acc_kh, kh_loop_body)]);

    let kw_cond = b.u_less_than(ty_bool, phi_kw, dim_kw);
    b.branch_conditional(kw_cond, kw_loop_body, kw_loop_merge);

    b.label_with_id(kw_loop_body);

    // Compute iy = oy * stride_h + kh_idx - pad_h
    let oy_times_sh = b.imul(ty_uint, oy, dim_stride_h);
    let pos_h = b.iadd(ty_uint, oy_times_sh, phi_kh);

    // Compute ix = ox * stride_w + kw_idx - pad_w
    let ox_times_sw = b.imul(ty_uint, ox, dim_stride_w);
    let pos_w = b.iadd(ty_uint, ox_times_sw, phi_kw);

    // Bounds check using nested selection (unsigned: pos >= pad && (pos - pad) < dim).
    let in_bounds_h_low = b.u_greater_than_equal(ty_bool, pos_h, dim_pad_h);
    let iy = b.isub(ty_uint, pos_h, dim_pad_h);
    let in_bounds_h_high = b.u_less_than(ty_bool, iy, dim_in_h);

    let in_bounds_w_low = b.u_greater_than_equal(ty_bool, pos_w, dim_pad_w);
    let ix = b.isub(ty_uint, pos_w, dim_pad_w);
    let in_bounds_w_high = b.u_less_than(ty_bool, ix, dim_in_w);

    // Nested selection for combined bounds check.
    let in_bounds_label = b.id();
    let check_h_high_label = b.id();
    let check_w_low_label = b.id();
    let check_w_high_label = b.id();
    let skip_label = b.id();

    b.selection_merge(skip_label);
    b.branch_conditional(in_bounds_h_low, check_h_high_label, skip_label);

    b.label_with_id(check_h_high_label);
    b.selection_merge(skip_label);
    b.branch_conditional(in_bounds_h_high, check_w_low_label, skip_label);

    b.label_with_id(check_w_low_label);
    b.selection_merge(skip_label);
    b.branch_conditional(in_bounds_w_low, check_w_high_label, skip_label);

    b.label_with_id(check_w_high_label);
    b.selection_merge(skip_label);
    b.branch_conditional(in_bounds_w_high, in_bounds_label, skip_label);

    b.label_with_id(in_bounds_label);

    // input_idx = batch_idx * in_ch * in_h * in_w + ic_actual * in_h * in_w + iy * in_w + ix
    let in_hw = b.imul(ty_uint, dim_in_h, dim_in_w);
    let in_chw = b.imul(ty_uint, dim_in_ch, in_hw);
    let batch_off = b.imul(ty_uint, batch_idx, in_chw);
    let ch_off = b.imul(ty_uint, ic_actual, in_hw);
    let row_off = b.imul(ty_uint, iy, dim_in_w);
    let input_idx = b.iadd(ty_uint, batch_off, ch_off);
    let input_idx = b.iadd(ty_uint, input_idx, row_off);
    let input_idx = b.iadd(ty_uint, input_idx, ix);

    // weight_idx = oc * ic_per_group * kh * kw + ic_local * kh * kw + kh_idx * kw + kw_idx
    let w_kh_kw = b.imul(ty_uint, dim_kh, dim_kw);
    let w_ic_kh_kw = b.imul(ty_uint, ic_per_group, w_kh_kw);
    let oc_woff = b.imul(ty_uint, oc, w_ic_kh_kw);
    let ic_woff = b.imul(ty_uint, phi_ic, w_kh_kw);
    let kh_woff = b.imul(ty_uint, phi_kh, dim_kw);
    let weight_idx = b.iadd(ty_uint, oc_woff, ic_woff);
    let weight_idx = b.iadd(ty_uint, weight_idx, kh_woff);
    let weight_idx = b.iadd(ty_uint, weight_idx, phi_kw);

    // Load input and weight, multiply-accumulate.
    let in_ptr = b.access_chain(ptr_sb_float, var_input, &[const_0u, input_idx]);
    let in_val = b.load(ty_float, in_ptr);
    let w_ptr = b.access_chain(ptr_sb_float, var_weight, &[const_0u, weight_idx]);
    let w_val = b.load(ty_float, w_ptr);
    let prod = b.fmul(ty_float, in_val, w_val);
    let new_acc = b.fadd(ty_float, phi_acc_kw, prod);

    b.branch(skip_label);

    // skip_label: phi to select updated or unchanged accumulator
    b.label_with_id(skip_label);
    let phi_acc_after_kw = b.phi(
        ty_float,
        &[
            (new_acc, in_bounds_label),
            (phi_acc_kw, kw_loop_body),
            (phi_acc_kw, check_h_high_label),
            (phi_acc_kw, check_w_low_label),
            (phi_acc_kw, check_w_high_label),
        ],
    );

    b.branch(kw_loop_continue);

    b.label_with_id(kw_loop_continue);
    let next_kw = b.iadd(ty_uint, phi_kw, const_1u);
    b.branch(kw_loop_header);

    fixup_phi(&mut b.functions, phi_kw, next_kw, kw_loop_continue);
    fixup_phi(
        &mut b.functions,
        phi_acc_kw,
        phi_acc_after_kw,
        kw_loop_continue,
    );

    b.label_with_id(kw_loop_merge);

    // After kw loop, continue to next kh
    b.branch(kh_loop_continue);

    b.label_with_id(kh_loop_continue);
    let next_kh = b.iadd(ty_uint, phi_kh, const_1u);
    b.branch(kh_loop_header);

    fixup_phi(&mut b.functions, phi_kh, next_kh, kh_loop_continue);
    fixup_phi(&mut b.functions, phi_acc_kh, phi_acc_kw, kh_loop_continue);

    b.label_with_id(kh_loop_merge);

    // After kh loop, continue to next ic
    b.branch(ic_loop_continue);

    b.label_with_id(ic_loop_continue);
    let next_ic = b.iadd(ty_uint, phi_ic, const_1u);
    b.branch(ic_loop_header);

    fixup_phi(&mut b.functions, phi_ic, next_ic, ic_loop_continue);
    fixup_phi(&mut b.functions, phi_acc_ic, phi_acc_kh, ic_loop_continue);

    b.label_with_id(ic_loop_merge);

    // Store result to output[gid_x].
    let out_ptr = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(out_ptr, phi_acc_ic);

    b.branch(return_label);

    b.label_with_id(return_label);
    b.op_return();
    b.func_end();

    b.build()
}

// ============================================================================
// CPU reference implementation
// ============================================================================

/// Compute Conv2d on CPU for reference/verification.
///
/// # Arguments
///
/// * `input` - Flat `[batch, in_channels, in_h, in_w]` in row-major order.
/// * `weight` - Flat `[out_channels, in_channels/groups, kernel_h, kernel_w]` in row-major order.
/// * `bias` - Optional flat `[out_channels]` bias vector.
/// * `config` - Convolution configuration.
/// * `in_h` - Input height.
/// * `in_w` - Input width.
///
/// # Returns
///
/// Flat `[batch, out_channels, out_h, out_w]` output in row-major order.
///
/// # Panics
///
/// Panics if config validation fails or input sizes are inconsistent.
pub fn conv2d_reference(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    config: &Conv2dConfig,
    in_h: usize,
    in_w: usize,
) -> Vec<f32> {
    config.validate().expect("Conv2dConfig validation failed");

    let in_ch = config.in_channels as usize;
    let out_ch = config.out_channels as usize;
    let kh = config.kernel_h as usize;
    let kw = config.kernel_w as usize;
    let sh = config.stride_h as usize;
    let sw = config.stride_w as usize;
    let ph = config.padding_h as usize;
    let pw = config.padding_w as usize;
    let groups = config.groups as usize;

    let batch = input.len() / (in_ch * in_h * in_w);
    assert_eq!(input.len(), batch * in_ch * in_h * in_w);

    let ic_per_group = in_ch / groups;
    let oc_per_group = out_ch / groups;
    assert_eq!(weight.len(), out_ch * ic_per_group * kh * kw);
    if let Some(b) = bias {
        assert_eq!(b.len(), out_ch);
    }

    let (out_h, out_w) = conv2d_output_size(in_h, in_w, config);
    let mut output = vec![0.0f32; batch * out_ch * out_h * out_w];

    for b_idx in 0..batch {
        for oc_idx in 0..out_ch {
            let group_idx = oc_idx / oc_per_group;
            let ic_start = group_idx * ic_per_group;
            let bias_val = bias.map_or(0.0, |b| b[oc_idx]);

            for oy in 0..out_h {
                for ox_idx in 0..out_w {
                    let mut acc = bias_val;

                    for ic_local in 0..ic_per_group {
                        let ic = ic_start + ic_local;
                        for kh_idx in 0..kh {
                            for kw_idx in 0..kw {
                                let iy_pos = oy * sh + kh_idx;
                                let ix_pos = ox_idx * sw + kw_idx;
                                if iy_pos >= ph && ix_pos >= pw {
                                    let iy = iy_pos - ph;
                                    let ix = ix_pos - pw;
                                    if iy < in_h && ix < in_w {
                                        let in_idx = b_idx * in_ch * in_h * in_w
                                            + ic * in_h * in_w
                                            + iy * in_w
                                            + ix;
                                        let w_idx = oc_idx * ic_per_group * kh * kw
                                            + ic_local * kh * kw
                                            + kh_idx * kw
                                            + kw_idx;
                                        acc += input[in_idx] * weight[w_idx];
                                    }
                                }
                            }
                        }
                    }

                    let out_idx = b_idx * out_ch * out_h * out_w
                        + oc_idx * out_h * out_w
                        + oy * out_w
                        + ox_idx;
                    output[out_idx] = acc;
                }
            }
        }
    }

    output
}

#[cfg(test)]
#[path = "spirv_conv2d_tests.rs"]
mod spirv_conv2d_tests;
