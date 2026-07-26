// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for depthwise 1D convolution compute shaders.
//!
//! Depthwise convolution convolves each input channel independently with its own
//! filter (groups == channels). This is a common building block in efficient
//! architectures like MobileNet and EfficientNet.
//!
//! # Buffer layout
//!
//! - **Binding 0** (set 0): Input `float[N * C * L]` (readonly)
//! - **Binding 1** (set 0): Weight `float[C * 1 * K]` (readonly)
//! - **Binding 2** (set 0): Output `float[N * C * L_out]`
//!
//! # Push constants
//!
//! ```text
//! { uint n, uint channels, uint length, uint kernel_size, uint stride, uint padding }
//! ```
//!
//! # Dispatch
//!
//! One thread per output element. Dispatch `ceil(total_output / WORKGROUP_SIZE)` workgroups.

use crate::spirv_emit::SPIRV_MAGIC;

/// Default workgroup size for depthwise Conv1d kernels (1D dispatch).
pub const DEPTHWISE_CONV_WORKGROUP_SIZE: u32 = 256;

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
// SPIR-V generation
// ============================================================================

/// Generate a SPIR-V 1.0 binary module for depthwise 1D convolution.
///
/// Each channel is convolved independently with its own filter (groups == channels).
///
/// # Layout
///
/// - **Binding 0** (set 0): Input buffer `float[N * C * L]` (readonly)
/// - **Binding 1** (set 0): Weight buffer `float[C * 1 * K]` (readonly)
/// - **Binding 2** (set 0): Output buffer `float[N * C * L_out]`
/// - **Push constants**: 6 x uint32 (n, channels, length, kernel_size, stride, padding)
///
/// # Arguments
///
/// * `channels` - Number of input/output channels.
/// * `kernel_size` - Convolution kernel spatial extent.
/// * `stride` - Stride of the convolution.
/// * `padding` - Zero-padding added to both sides of the input.
///
/// # Returns
///
/// SPIR-V binary as a `Vec<u32>` word vector.
pub fn generate_depthwise_conv1d_spirv(
    channels: u32,
    kernel_size: u32,
    stride: u32,
    _padding: u32,
) -> Vec<u32> {
    assert!(channels > 0, "channels must be > 0");
    assert!(kernel_size > 0, "kernel_size must be > 0");
    assert!(stride > 0, "stride must be > 0");

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

    // Output buffer struct.
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: 6 x uint.
    let pc_members = vec![ty_uint; 6];
    let ty_struct_pc = b.type_struct(&pc_members);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    for i in 0..6u32 {
        b.member_decorate(ty_struct_pc, i, DECORATION_OFFSET, &[i * 4]);
    }

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_weight = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_weight);
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
    let const_f0 = b.constant_f32(ty_float, 0.0);

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);
    b.decorate(var_input, DECORATION_NON_WRITABLE, &[]);

    let var_weight = b.variable_global(ptr_sb_weight, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_weight, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_weight, DECORATION_BINDING, &[1]);
    b.decorate(var_weight, DECORATION_NON_WRITABLE, &[]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, DEPTHWISE_CONV_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load global invocation ID.
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants.
    // 0: n, 1: channels, 2: length, 3: kernel_size, 4: stride, 5: padding
    let pc_idx_consts = [const_0u, const_1u, const_2u, const_3u, const_4u, const_5u];
    let pc_ptrs: Vec<u32> = pc_idx_consts
        .iter()
        .map(|&idx| b.access_chain(ptr_pc_uint, var_pc, &[idx]))
        .collect();

    let dim_n = b.load(ty_uint, pc_ptrs[0]);
    let dim_channels = b.load(ty_uint, pc_ptrs[1]);
    let dim_length = b.load(ty_uint, pc_ptrs[2]);
    let dim_ks = b.load(ty_uint, pc_ptrs[3]);
    let dim_stride = b.load(ty_uint, pc_ptrs[4]);
    let dim_padding = b.load(ty_uint, pc_ptrs[5]);

    // Compute out_length = (length + 2*padding - kernel_size) / stride + 1
    let two_padding = b.imul(ty_uint, const_2u, dim_padding);
    let padded_len = b.iadd(ty_uint, dim_length, two_padding);
    let numerator = b.isub(ty_uint, padded_len, dim_ks);
    let out_length = b.u_div(ty_uint, numerator, dim_stride);
    let out_length = b.iadd(ty_uint, out_length, const_1u);

    // Compute total output elements = n * channels * out_length
    let n_times_c = b.imul(ty_uint, dim_n, dim_channels);
    let total_output = b.imul(ty_uint, n_times_c, out_length);

    // Bounds check: if gid_x >= total_output, return early.
    let oob = b.u_greater_than_equal(ty_bool, gid_x, total_output);
    let body_label = b.id();
    let return_label = b.id();
    b.selection_merge(return_label);
    b.branch_conditional(oob, return_label, body_label);

    b.label_with_id(body_label);

    // Decompose gid_x into [batch_idx, channel, ox]:
    //   ox = gid_x % out_length
    //   tmp = gid_x / out_length
    //   channel = tmp % channels
    //   batch_idx = tmp / channels
    let ox = b.u_mod(ty_uint, gid_x, out_length);
    let tmp = b.u_div(ty_uint, gid_x, out_length);
    let channel = b.u_mod(ty_uint, tmp, dim_channels);
    let batch_idx = b.u_div(ty_uint, tmp, dim_channels);

    // Inner loop: accumulate over kernel_size
    // For each k in [0, kernel_size):
    //   ix = ox * stride + k - padding
    //   if ix >= 0 && ix < length:
    //     input_idx = batch_idx * channels * length + channel * length + ix
    //     weight_idx = channel * kernel_size + k
    //     acc += input[input_idx] * weight[weight_idx]
    let k_loop_header = b.id();
    let k_loop_body = b.id();
    let k_loop_continue = b.id();
    let k_loop_merge = b.id();

    b.branch(k_loop_header);
    b.label_with_id(k_loop_header);
    b.loop_merge(k_loop_merge, k_loop_continue);

    let phi_k = b.phi(ty_uint, &[(const_0u, body_label)]);
    let phi_acc = b.phi(ty_float, &[(const_f0, body_label)]);

    let k_cond = b.u_less_than(ty_bool, phi_k, dim_ks);
    b.branch_conditional(k_cond, k_loop_body, k_loop_merge);

    b.label_with_id(k_loop_body);

    // Compute ix = ox * stride + k - padding
    let ox_times_stride = b.imul(ty_uint, ox, dim_stride);
    let pos_sum = b.iadd(ty_uint, ox_times_stride, phi_k);

    // Bounds check: pos_sum >= padding and (pos_sum - padding) < length
    let in_bounds_low = b.u_greater_than_equal(ty_bool, pos_sum, dim_padding);
    let ix = b.isub(ty_uint, pos_sum, dim_padding);
    let in_bounds_high = b.u_less_than(ty_bool, ix, dim_length);

    // Nested selection for AND of bounds checks.
    let in_bounds_label = b.id();
    let check_high_label = b.id();
    let skip_label = b.id();

    b.selection_merge(skip_label);
    b.branch_conditional(in_bounds_low, check_high_label, skip_label);

    b.label_with_id(check_high_label);
    b.selection_merge(skip_label);
    b.branch_conditional(in_bounds_high, in_bounds_label, skip_label);

    b.label_with_id(in_bounds_label);

    // input_idx = batch_idx * channels * length + channel * length + ix
    let batch_stride_in = b.imul(ty_uint, dim_channels, dim_length);
    let batch_offset = b.imul(ty_uint, batch_idx, batch_stride_in);
    let ch_offset = b.imul(ty_uint, channel, dim_length);
    let input_idx = b.iadd(ty_uint, batch_offset, ch_offset);
    let input_idx = b.iadd(ty_uint, input_idx, ix);

    // weight_idx = channel * kernel_size + k
    let weight_offset = b.imul(ty_uint, channel, dim_ks);
    let weight_idx = b.iadd(ty_uint, weight_offset, phi_k);

    // Load input and weight, multiply-accumulate.
    let in_ptr = b.access_chain(ptr_sb_float, var_input, &[const_0u, input_idx]);
    let in_val = b.load(ty_float, in_ptr);
    let w_ptr = b.access_chain(ptr_sb_float, var_weight, &[const_0u, weight_idx]);
    let w_val = b.load(ty_float, w_ptr);
    let prod = b.fmul(ty_float, in_val, w_val);
    let new_acc = b.fadd(ty_float, phi_acc, prod);

    b.branch(skip_label);

    // skip_label: phi to select updated or unchanged accumulator
    b.label_with_id(skip_label);
    let phi_acc_after_k = b.phi(
        ty_float,
        &[
            (new_acc, in_bounds_label),
            (phi_acc, k_loop_body),
            (phi_acc, check_high_label),
        ],
    );

    b.branch(k_loop_continue);

    b.label_with_id(k_loop_continue);
    let next_k = b.iadd(ty_uint, phi_k, const_1u);
    b.branch(k_loop_header);

    fixup_phi(&mut b.functions, phi_k, next_k, k_loop_continue);
    fixup_phi(&mut b.functions, phi_acc, phi_acc_after_k, k_loop_continue);

    b.label_with_id(k_loop_merge);

    // Store result to output[gid_x].
    let out_ptr = b.access_chain(ptr_sb_float, var_output, &[const_0u, gid_x]);
    b.store(out_ptr, phi_acc);

    b.branch(return_label);

    b.label_with_id(return_label);
    b.op_return();
    b.func_end();

    b.build()
}

// ============================================================================
// CPU reference implementation
// ============================================================================

/// Compute depthwise Conv1d on CPU for reference/verification.
///
/// Each channel is convolved independently with its own filter.
///
/// # Arguments
///
/// * `input` - Flat `[N, C, L]` in row-major order.
/// * `weight` - Flat `[C, 1, K]` in row-major order.
/// * `n` - Batch size.
/// * `c` - Number of channels.
/// * `l` - Input spatial length.
/// * `k` - Kernel size.
/// * `stride` - Stride of the convolution.
/// * `padding` - Zero-padding added to both sides of the input.
///
/// # Returns
///
/// Flat `[N, C, L_out]` output in row-major order.
///
/// # Panics
///
/// Panics if input or weight sizes are inconsistent with the given dimensions.
pub fn depthwise_conv1d_reference(
    input: &[f32],
    weight: &[f32],
    n: usize,
    c: usize,
    l: usize,
    k: usize,
    stride: usize,
    padding: usize,
) -> Vec<f32> {
    assert_eq!(input.len(), n * c * l, "input size mismatch");
    assert_eq!(weight.len(), c * k, "weight size mismatch");
    assert!(stride > 0, "stride must be > 0");
    assert!(k > 0, "kernel_size must be > 0");

    let out_len = (l + 2 * padding - k) / stride + 1;
    let mut output = vec![0.0f32; n * c * out_len];

    for b_idx in 0..n {
        for ch in 0..c {
            for ox in 0..out_len {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    let ix_pos = ox * stride + ki;
                    if ix_pos >= padding {
                        let ix = ix_pos - padding;
                        if ix < l {
                            let in_idx = b_idx * c * l + ch * l + ix;
                            let w_idx = ch * k + ki;
                            acc += input[in_idx] * weight[w_idx];
                        }
                    }
                }
                let out_idx = b_idx * c * out_len + ch * out_len + ox;
                output[out_idx] = acc;
            }
        }
    }

    output
}

#[cfg(test)]
#[path = "spirv_depthwise_conv_tests.rs"]
mod spirv_depthwise_conv_tests;
