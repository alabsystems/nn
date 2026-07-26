// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for 2D pooling compute shaders (max and average).
//!
//! Extends the 1D pooling infrastructure in [`super::spirv_conv`] with
//! 2D spatial support for vision models.
//!
//! - [`generate_max_pool2d_spirv`]: Max pooling 2D
//! - [`generate_avg_pool2d_spirv`]: Average pooling 2D
//! - [`Pool2dConfig`]: Shared configuration for both pooling variants
//! - CPU references for differential verification
//!
//! # Buffer layout
//!
//! - **Binding 0** (set 0): Input `float[batch * channels * in_h * in_w]` (readonly)
//! - **Binding 1** (set 0): Output `float[batch * channels * out_h * out_w]`
//!
//! # Push constants
//!
//! ```text
//! { uint batch, uint channels, uint in_h, uint in_w,
//!   uint kernel_h, uint kernel_w, uint stride_h, uint stride_w,
//!   uint padding_h, uint padding_w }
//! ```
//!
//! # Dispatch
//!
//! One thread per output element. Dispatch `ceil(total_output / WORKGROUP_SIZE)` workgroups.

use crate::spirv_emit::SPIRV_MAGIC;

/// Default workgroup size for Pool2d kernels (1D dispatch).
pub const POOL2D_WORKGROUP_SIZE: u32 = 64;

// ---- SPIR-V constants ----

const SPIRV_VERSION_1_0: u32 = 0x0001_0000;
const GENERATOR_MAGIC: u32 = 0x4E4E_0000;

const fn op(word_count: u16, opcode: u16) -> u32 {
    (word_count as u32) << 16 | opcode as u32
}

// Opcodes.
const OP_CAPABILITY: u16 = 17;
const OP_EXT_INST_IMPORT: u16 = 11;
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
const OP_FDIV: u16 = 136;
const OP_U_LESS_THAN: u16 = 176;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_IADD: u16 = 128;
const OP_IMUL: u16 = 132;
const OP_U_DIV: u16 = 134;
const OP_U_MOD: u16 = 137;
const OP_EXT_INST: u16 = 12;
const OP_CONVERT_U_TO_F: u16 = 112;

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

// GLSL.std.450 extended instruction set.
const GLSL_STD_450_FMAX: u32 = 40;

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

    fn ext_inst_import(&mut self, name: &str) -> u32 {
        let result = self.id();
        let name_words = encode_string(name);
        let wc = 2 + name_words.len() as u16;
        self.extensions.push(op(wc, OP_EXT_INST_IMPORT));
        self.extensions.push(result);
        self.extensions.extend_from_slice(&name_words);
        result
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

    fn fdiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FDIV));
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

    fn ext_inst(&mut self, result_type: u32, set: u32, instruction: u32, operands: &[u32]) -> u32 {
        let result = self.id();
        let wc = 5 + operands.len() as u16;
        self.functions.push(op(wc, OP_EXT_INST));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(set);
        self.functions.push(instruction);
        self.functions.extend_from_slice(operands);
        result
    }

    fn convert_u_to_f(&mut self, result_type: u32, value: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(4, OP_CONVERT_U_TO_F));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(value);
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
// Pool2d configuration
// ============================================================================

/// Configuration for a 2D pooling kernel (max or average).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool2dConfig {
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
}

impl Pool2dConfig {
    /// Create a new Pool2d configuration with stride equal to kernel size and no padding.
    pub fn new(kernel_h: u32, kernel_w: u32) -> Self {
        Self {
            kernel_h,
            kernel_w,
            stride_h: kernel_h,
            stride_w: kernel_w,
            padding_h: 0,
            padding_w: 0,
        }
    }

    /// Set stride (both dimensions).
    #[must_use]
    pub fn stride(mut self, stride_h: u32, stride_w: u32) -> Self {
        self.stride_h = stride_h;
        self.stride_w = stride_w;
        self
    }

    /// Set padding (both dimensions).
    #[must_use]
    pub fn padding(mut self, padding_h: u32, padding_w: u32) -> Self {
        self.padding_h = padding_h;
        self.padding_w = padding_w;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
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
        Ok(())
    }
}

/// Compute the output spatial dimensions for 2D pooling.
pub fn pool2d_output_size(in_h: usize, in_w: usize, config: &Pool2dConfig) -> (usize, usize) {
    let out_h = (in_h + 2 * config.padding_h as usize - config.kernel_h as usize)
        / config.stride_h as usize
        + 1;
    let out_w = (in_w + 2 * config.padding_w as usize - config.kernel_w as usize)
        / config.stride_w as usize
        + 1;
    (out_h, out_w)
}

// ============================================================================
// Shared pool2d SPIR-V setup
// ============================================================================

/// Pool2d variant: max or average.
enum PoolVariant {
    Max,
    Avg,
}

/// Generate a SPIR-V 1.0 binary module (as `Vec<u32>`) for 2D pooling.
///
/// Internal function used by both max and avg pool generators.
fn generate_pool2d_spirv_inner(config: &Pool2dConfig, variant: PoolVariant) -> Vec<u32> {
    config.validate().expect("Pool2dConfig validation failed");

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);

    // For max pool we need GLSL.std.450 for FMax.
    let glsl_ext = match variant {
        PoolVariant::Max => Some(b.ext_inst_import("GLSL.std.450")),
        PoolVariant::Avg => None,
    };

    b.memory_model_decl(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // Types.
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer structs: input (readonly) + output.
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: 10 x uint.
    let pc_count = 10u32;
    let pc_members = vec![ty_uint; pc_count as usize];
    let ty_struct_pc = b.type_struct(&pc_members);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    for i in 0..pc_count {
        b.member_decorate(ty_struct_pc, i, DECORATION_OFFSET, &[i * 4]);
    }

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // Constants.
    let const_u: Vec<u32> = (0..pc_count).map(|i| b.constant_u32(ty_uint, i)).collect();
    let const_0u = const_u[0];
    let const_1u = const_u[1];
    let const_2u = const_u[2];

    let init_val = match variant {
        PoolVariant::Max => b.constant_f32(ty_float, f32::NEG_INFINITY),
        PoolVariant::Avg => b.constant_f32(ty_float, 0.0),
    };

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);
    b.decorate(var_input, DECORATION_NON_WRITABLE, &[]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, POOL2D_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants:
    // 0: batch, 1: channels, 2: in_h, 3: in_w,
    // 4: kernel_h, 5: kernel_w, 6: stride_h, 7: stride_w,
    // 8: padding_h, 9: padding_w
    let pc_ptrs: Vec<u32> = (0..pc_count)
        .map(|i| b.access_chain(ptr_pc_uint, var_pc, &[const_u[i as usize]]))
        .collect();

    let dim_batch = b.load(ty_uint, pc_ptrs[0]);
    let dim_channels = b.load(ty_uint, pc_ptrs[1]);
    let dim_in_h = b.load(ty_uint, pc_ptrs[2]);
    let dim_in_w = b.load(ty_uint, pc_ptrs[3]);
    let dim_kh = b.load(ty_uint, pc_ptrs[4]);
    let dim_kw = b.load(ty_uint, pc_ptrs[5]);
    let dim_stride_h = b.load(ty_uint, pc_ptrs[6]);
    let dim_stride_w = b.load(ty_uint, pc_ptrs[7]);
    let dim_pad_h = b.load(ty_uint, pc_ptrs[8]);
    let dim_pad_w = b.load(ty_uint, pc_ptrs[9]);

    // out_h = (in_h + 2*pad_h - kh) / stride_h + 1
    let two_pad_h = b.imul(ty_uint, const_2u, dim_pad_h);
    let padded_h = b.iadd(ty_uint, dim_in_h, two_pad_h);
    let num_h = b.isub(ty_uint, padded_h, dim_kh);
    let out_h = b.u_div(ty_uint, num_h, dim_stride_h);
    let out_h = b.iadd(ty_uint, out_h, const_1u);

    // out_w = (in_w + 2*pad_w - kw) / stride_w + 1
    let two_pad_w = b.imul(ty_uint, const_2u, dim_pad_w);
    let padded_w = b.iadd(ty_uint, dim_in_w, two_pad_w);
    let num_w = b.isub(ty_uint, padded_w, dim_kw);
    let out_w = b.u_div(ty_uint, num_w, dim_stride_w);
    let out_w = b.iadd(ty_uint, out_w, const_1u);

    // total = batch * channels * out_h * out_w
    let bc = b.imul(ty_uint, dim_batch, dim_channels);
    let bco = b.imul(ty_uint, bc, out_h);
    let total = b.imul(ty_uint, bco, out_w);

    // Bounds check.
    let oob = b.u_greater_than_equal(ty_bool, gid_x, total);
    let body_label = b.id();
    let return_label = b.id();
    b.selection_merge(return_label);
    b.branch_conditional(oob, return_label, body_label);

    b.label_with_id(body_label);

    // Decompose gid_x into [batch_idx, ch, oy, ox].
    let ox = b.u_mod(ty_uint, gid_x, out_w);
    let tmp1 = b.u_div(ty_uint, gid_x, out_w);
    let oy = b.u_mod(ty_uint, tmp1, out_h);
    let tmp2 = b.u_div(ty_uint, tmp1, out_h);
    let ch = b.u_mod(ty_uint, tmp2, dim_channels);
    let batch_idx = b.u_div(ty_uint, tmp2, dim_channels);

    // Double nested loop: kh, kw.
    let kh_loop_header = b.id();
    let kh_loop_body = b.id();
    let kh_loop_continue = b.id();
    let kh_loop_merge = b.id();

    b.branch(kh_loop_header);
    b.label_with_id(kh_loop_header);
    b.loop_merge(kh_loop_merge, kh_loop_continue);

    let phi_kh = b.phi(ty_uint, &[(const_0u, body_label)]);
    let phi_acc_kh = b.phi(ty_float, &[(init_val, body_label)]);
    // For avg pool, also track count of valid elements.
    let phi_cnt_kh = match variant {
        PoolVariant::Avg => Some(b.phi(ty_uint, &[(const_0u, body_label)])),
        PoolVariant::Max => None,
    };

    let kh_cond = b.u_less_than(ty_bool, phi_kh, dim_kh);
    b.branch_conditional(kh_cond, kh_loop_body, kh_loop_merge);

    b.label_with_id(kh_loop_body);

    // Inner kw loop.
    let kw_loop_header = b.id();
    let kw_loop_body = b.id();
    let kw_loop_continue = b.id();
    let kw_loop_merge = b.id();

    b.branch(kw_loop_header);
    b.label_with_id(kw_loop_header);
    b.loop_merge(kw_loop_merge, kw_loop_continue);

    let phi_kw = b.phi(ty_uint, &[(const_0u, kh_loop_body)]);
    let phi_acc_kw = b.phi(ty_float, &[(phi_acc_kh, kh_loop_body)]);
    let phi_cnt_kw = match variant {
        PoolVariant::Avg => {
            let cnt_kh = phi_cnt_kh.unwrap();
            Some(b.phi(ty_uint, &[(cnt_kh, kh_loop_body)]))
        }
        PoolVariant::Max => None,
    };

    let kw_cond = b.u_less_than(ty_bool, phi_kw, dim_kw);
    b.branch_conditional(kw_cond, kw_loop_body, kw_loop_merge);

    b.label_with_id(kw_loop_body);

    // Compute iy, ix with padding.
    let oy_sh = b.imul(ty_uint, oy, dim_stride_h);
    let pos_h = b.iadd(ty_uint, oy_sh, phi_kh);
    let ox_sw = b.imul(ty_uint, ox, dim_stride_w);
    let pos_w = b.iadd(ty_uint, ox_sw, phi_kw);

    let in_bounds_h_low = b.u_greater_than_equal(ty_bool, pos_h, dim_pad_h);
    let iy = b.isub(ty_uint, pos_h, dim_pad_h);
    let in_bounds_h_high = b.u_less_than(ty_bool, iy, dim_in_h);

    let in_bounds_w_low = b.u_greater_than_equal(ty_bool, pos_w, dim_pad_w);
    let ix = b.isub(ty_uint, pos_w, dim_pad_w);
    let in_bounds_w_high = b.u_less_than(ty_bool, ix, dim_in_w);

    // Nested bounds check.
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

    // input_idx = batch_idx * channels * in_h * in_w + ch * in_h * in_w + iy * in_w + ix
    let in_hw = b.imul(ty_uint, dim_in_h, dim_in_w);
    let in_chw = b.imul(ty_uint, dim_channels, in_hw);
    let batch_off = b.imul(ty_uint, batch_idx, in_chw);
    let ch_off = b.imul(ty_uint, ch, in_hw);
    let row_off = b.imul(ty_uint, iy, dim_in_w);
    let input_idx = b.iadd(ty_uint, batch_off, ch_off);
    let input_idx = b.iadd(ty_uint, input_idx, row_off);
    let input_idx = b.iadd(ty_uint, input_idx, ix);

    let in_ptr = b.access_chain(ptr_sb_float, var_input, &[const_0u, input_idx]);
    let in_val = b.load(ty_float, in_ptr);

    let new_acc = match variant {
        PoolVariant::Max => {
            // FMax via GLSL.std.450 extended instruction.
            b.ext_inst(
                ty_float,
                glsl_ext.unwrap(),
                GLSL_STD_450_FMAX,
                &[phi_acc_kw, in_val],
            )
        }
        PoolVariant::Avg => b.fadd(ty_float, phi_acc_kw, in_val),
    };
    let new_cnt = match variant {
        PoolVariant::Avg => Some(b.iadd(ty_uint, phi_cnt_kw.unwrap(), const_1u)),
        PoolVariant::Max => None,
    };

    b.branch(skip_label);

    // skip_label: phi for accumulator.
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
    let phi_cnt_after_kw = match variant {
        PoolVariant::Avg => {
            let cnt_kw = phi_cnt_kw.unwrap();
            Some(b.phi(
                ty_uint,
                &[
                    (new_cnt.unwrap(), in_bounds_label),
                    (cnt_kw, kw_loop_body),
                    (cnt_kw, check_h_high_label),
                    (cnt_kw, check_w_low_label),
                    (cnt_kw, check_w_high_label),
                ],
            ))
        }
        PoolVariant::Max => None,
    };

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
    if let (PoolVariant::Avg, Some(cnt_kw), Some(cnt_after_kw)) =
        (&variant, phi_cnt_kw, phi_cnt_after_kw)
    {
        fixup_phi(&mut b.functions, cnt_kw, cnt_after_kw, kw_loop_continue);
    }

    b.label_with_id(kw_loop_merge);

    b.branch(kh_loop_continue);

    b.label_with_id(kh_loop_continue);
    let next_kh = b.iadd(ty_uint, phi_kh, const_1u);
    b.branch(kh_loop_header);

    fixup_phi(&mut b.functions, phi_kh, next_kh, kh_loop_continue);
    fixup_phi(&mut b.functions, phi_acc_kh, phi_acc_kw, kh_loop_continue);
    if let (PoolVariant::Avg, Some(cnt_kh), Some(cnt_kw)) = (&variant, phi_cnt_kh, phi_cnt_kw) {
        fixup_phi(&mut b.functions, cnt_kh, cnt_kw, kh_loop_continue);
    }

    b.label_with_id(kh_loop_merge);

    // For avg pool: divide by count.
    let final_val = match variant {
        PoolVariant::Max => phi_acc_kh,
        PoolVariant::Avg => {
            let cnt_f = b.convert_u_to_f(ty_float, phi_cnt_kh.unwrap());
            b.fdiv(ty_float, phi_acc_kh, cnt_f)
        }
    };

    // Store output.
    let out_ptr = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(out_ptr, final_val);

    b.branch(return_label);

    b.label_with_id(return_label);
    b.op_return();
    b.func_end();

    b.build()
}

// ============================================================================
// Public API
// ============================================================================

/// Generate a SPIR-V 1.0 binary module (as `Vec<u32>`) for 2D max pooling.
///
/// # Panics
///
/// Panics if `config.validate()` fails.
pub fn generate_max_pool2d_spirv(config: &Pool2dConfig) -> Vec<u32> {
    generate_pool2d_spirv_inner(config, PoolVariant::Max)
}

/// Generate a SPIR-V 1.0 binary module (as `Vec<u32>`) for 2D average pooling.
///
/// Average pool divides by the number of valid (non-padded) elements in each window.
///
/// # Panics
///
/// Panics if `config.validate()` fails.
pub fn generate_avg_pool2d_spirv(config: &Pool2dConfig) -> Vec<u32> {
    generate_pool2d_spirv_inner(config, PoolVariant::Avg)
}

// ============================================================================
// CPU reference implementations
// ============================================================================

/// CPU reference for 2D max pooling.
///
/// Input is flat `[batch, channels, in_h, in_w]` in row-major order.
/// Returns flat `[batch, channels, out_h, out_w]`.
pub fn max_pool2d_reference(
    input: &[f32],
    config: &Pool2dConfig,
    in_h: usize,
    in_w: usize,
) -> Vec<f32> {
    config.validate().expect("Pool2dConfig validation failed");

    let kh = config.kernel_h as usize;
    let kw = config.kernel_w as usize;
    let sh = config.stride_h as usize;
    let sw = config.stride_w as usize;
    let ph = config.padding_h as usize;
    let pw = config.padding_w as usize;

    let (out_h, out_w) = pool2d_output_size(in_h, in_w, config);
    let spatial = in_h * in_w;

    // Infer channels from input size (batch * channels * spatial).
    // We treat input as [N, C, H, W] where N*C = input.len() / spatial.
    let nc = input.len() / spatial;
    assert_eq!(input.len(), nc * spatial);

    let mut output = vec![f32::NEG_INFINITY; nc * out_h * out_w];

    for nc_idx in 0..nc {
        for oy in 0..out_h {
            for ox_idx in 0..out_w {
                let mut max_val = f32::NEG_INFINITY;
                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        let iy_pos = oy * sh + kh_idx;
                        let ix_pos = ox_idx * sw + kw_idx;
                        if iy_pos >= ph && ix_pos >= pw {
                            let iy = iy_pos - ph;
                            let ix = ix_pos - pw;
                            if iy < in_h && ix < in_w {
                                let val = input[nc_idx * spatial + iy * in_w + ix];
                                if val > max_val {
                                    max_val = val;
                                }
                            }
                        }
                    }
                }
                output[nc_idx * out_h * out_w + oy * out_w + ox_idx] = max_val;
            }
        }
    }

    output
}

/// CPU reference for 2D average pooling.
///
/// Divides by count of valid (non-padded) elements per window.
/// Input is flat `[batch, channels, in_h, in_w]` in row-major order.
/// Returns flat `[batch, channels, out_h, out_w]`.
pub fn avg_pool2d_reference(
    input: &[f32],
    config: &Pool2dConfig,
    in_h: usize,
    in_w: usize,
) -> Vec<f32> {
    config.validate().expect("Pool2dConfig validation failed");

    let kh = config.kernel_h as usize;
    let kw = config.kernel_w as usize;
    let sh = config.stride_h as usize;
    let sw = config.stride_w as usize;
    let ph = config.padding_h as usize;
    let pw = config.padding_w as usize;

    let (out_h, out_w) = pool2d_output_size(in_h, in_w, config);
    let spatial = in_h * in_w;
    let nc = input.len() / spatial;
    assert_eq!(input.len(), nc * spatial);

    let mut output = vec![0.0f32; nc * out_h * out_w];

    for nc_idx in 0..nc {
        for oy in 0..out_h {
            for ox_idx in 0..out_w {
                let mut sum = 0.0f32;
                let mut count = 0u32;
                for kh_idx in 0..kh {
                    for kw_idx in 0..kw {
                        let iy_pos = oy * sh + kh_idx;
                        let ix_pos = ox_idx * sw + kw_idx;
                        if iy_pos >= ph && ix_pos >= pw {
                            let iy = iy_pos - ph;
                            let ix = ix_pos - pw;
                            if iy < in_h && ix < in_w {
                                sum += input[nc_idx * spatial + iy * in_w + ix];
                                count += 1;
                            }
                        }
                    }
                }
                if count > 0 {
                    output[nc_idx * out_h * out_w + oy * out_w + ox_idx] = sum / count as f32;
                }
            }
        }
    }

    output
}

#[cfg(test)]
#[path = "spirv_pool2d_tests.rs"]
mod spirv_pool2d_tests;
