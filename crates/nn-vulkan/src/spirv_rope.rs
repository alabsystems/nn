// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for rotary position embedding (RoPE) compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for applying rotary position embeddings
//! to query/key tensors in transformer attention layers.
//!
//! # Standard RoPE
//!
//! For each position `pos` and dimension pair `(2i, 2i+1)`:
//! ```text
//! theta = pos * base^(-2i / head_dim)
//! x_rot[2i]   = x[2i]   * cos(theta) - x[2i+1] * sin(theta)
//! x_rot[2i+1] = x[2i]   * sin(theta) + x[2i+1] * cos(theta)
//! ```
//!
//! # NeoX-style RoPE
//!
//! Splits the head dimension in half — first half paired with second half:
//! ```text
//! theta = pos * base^(-2i / head_dim)
//! x_rot[i]              = x[i]              * cos(theta) - x[i + head_dim/2] * sin(theta)
//! x_rot[i + head_dim/2] = x[i + head_dim/2] * cos(theta) + x[i]              * sin(theta)
//! ```
//!
//! # Buffer layout
//!
//! - Binding 0: Input `x` — `float[batch_heads * seq_len * head_dim]` (row-major)
//! - Binding 1: Output `x_rot` — `float[batch_heads * seq_len * head_dim]` (row-major)
//!
//! # Push constants
//!
//! - `uint batch_heads` at offset 0
//! - `uint seq_len` at offset 4
//! - `uint head_dim` at offset 8
//! - `float base_bits` at offset 12 (base frequency as f32 bit pattern, typically 10000.0)

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for RoPE kernels (1D dispatch).
pub const ROPE_WORKGROUP_SIZE: u32 = 64;

// ---- SPIR-V constants (duplicated to keep module independent) ----

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
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_SELECTION_MERGE: u16 = 247;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_IMUL: u16 = 132;
const OP_IADD: u16 = 128;
const OP_EXT_INST: u16 = 12;
const OP_UDIV: u16 = 134;
const OP_UMOD: u16 = 137;
const OP_CONVERT_U_TO_F: u16 = 112;
const OP_FDIV: u16 = 136;

// Decorations.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

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

// GLSL.std.450 extended instruction set opcodes.
const GLSL_STD_450_SIN: u32 = 13;
const GLSL_STD_450_COS: u32 = 14;
const GLSL_STD_450_EXP: u32 = 27;
const GLSL_STD_450_LOG: u32 = 28;

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

/// SPIR-V module builder (local to this module).
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

    fn memory_model(&mut self, addressing: u32, model: u32) {
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

    fn fsub(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FSUB));
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

    fn fdiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_FDIV));
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

    fn imul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_IMUL));
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

    fn udiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UDIV));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    fn umod(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UMOD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
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

    fn ext_inst(
        &mut self,
        result_type: u32,
        ext_set: u32,
        instruction: u32,
        operands: &[u32],
    ) -> u32 {
        let result = self.id();
        let wc = 5 + operands.len() as u16;
        self.functions.push(op(wc, OP_EXT_INST));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(ext_set);
        self.functions.push(instruction);
        self.functions.extend_from_slice(operands);
        result
    }

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(512);
        module.push(SPIRV_MAGIC);
        module.push(SPIRV_VERSION_1_0);
        module.push(GENERATOR_MAGIC);
        module.push(self.bound);
        module.push(0); // Reserved schema.
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

/// Generate a SPIR-V 1.0 binary for standard rotary position embedding (RoPE).
///
/// Each thread handles one element in the flattened `[batch_heads, seq_len, head_dim]`
/// tensor. Dimension pairs `(2i, 2i+1)` are rotated by angle `theta = pos * base^(-2i/head_dim)`.
///
/// # Arguments
///
/// * `seq_len` - Sequence length (compile-time hint; actual from push constants).
/// * `head_dim` - Dimension of each attention head (compile-time hint; must be even).
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>`, ready for Vulkan pipeline creation.
pub fn generate_rope_spirv(seq_len: u32, head_dim: u32) -> Vec<u32> {
    let _ = seq_len;
    let _ = head_dim;
    generate_rope_spirv_inner(false)
}

/// Generate a SPIR-V 1.0 binary for NeoX-style interleaved RoPE.
///
/// NeoX pairs the first half of head_dim with the second half:
/// - `x_rot[i] = x[i] * cos(theta) - x[i + half_dim] * sin(theta)`
/// - `x_rot[i + half_dim] = x[i + half_dim] * cos(theta) + x[i] * sin(theta)`
///
/// # Arguments
///
/// * `seq_len` - Sequence length (compile-time hint; actual from push constants).
/// * `head_dim` - Dimension of each attention head (compile-time hint; must be even).
///
/// # Returns
///
/// SPIR-V binary as `Vec<u32>`, ready for Vulkan pipeline creation.
pub fn generate_rope_neox_spirv(seq_len: u32, head_dim: u32) -> Vec<u32> {
    let _ = seq_len;
    let _ = head_dim;
    generate_rope_spirv_inner(true)
}

/// Internal implementation shared by standard and NeoX RoPE.
///
/// # Algorithm (standard, neox=false)
///
/// ```text
/// gid = gl_GlobalInvocationID.x
/// // Each thread handles one pair (2i, 2i+1)
/// pair_idx = gid   // indexes into half_dim pairs
/// total_pairs = batch_heads * seq_len * half_dim
/// if pair_idx >= total_pairs: return
///
/// half_dim = head_dim / 2
/// dim_i = pair_idx % half_dim
/// tmp = pair_idx / half_dim
/// pos = tmp % seq_len
/// bh = tmp / seq_len
///
/// // Compute theta = pos * base^(-2*dim_i / head_dim)
/// // = pos * exp(-2*dim_i/head_dim * log(base))
/// exponent = -2.0 * float(dim_i) / float(head_dim) * log(base)
/// theta = float(pos) * exp(exponent)
///
/// cos_t = cos(theta)
/// sin_t = sin(theta)
///
/// base_idx = bh * seq_len * head_dim + pos * head_dim
/// // Standard: pair (2*dim_i, 2*dim_i + 1)
/// idx_even = base_idx + 2 * dim_i
/// idx_odd  = base_idx + 2 * dim_i + 1
/// // NeoX: pair (dim_i, dim_i + half_dim)
/// idx_first  = base_idx + dim_i
/// idx_second = base_idx + dim_i + half_dim
///
/// x0 = input[idx_even/first]
/// x1 = input[idx_odd/second]
/// output[idx_even/first]  = x0 * cos_t - x1 * sin_t
/// output[idx_odd/second]  = x0 * sin_t + x1 * cos_t
/// ```
fn generate_rope_spirv_inner(neox: bool) -> Vec<u32> {
    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    let glsl_ext = b.ext_inst_import("GLSL.std.450");
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // ---- Types ----
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Input buffer struct (binding 0).
    let ty_struct_input = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_input, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_input, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct (binding 1).
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint batch_heads, uint seq_len, uint head_dim, float base }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint, ty_uint, ty_float]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);
    b.member_decorate(ty_struct_pc, 3, DECORATION_OFFSET, &[12]);

    // Pointer types.
    let ptr_sb_input = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_input);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);
    let ptr_pc_float = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_float);

    // Constants.
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_3u = b.constant_u32(ty_uint, 3);
    let const_f_neg2 = b.constant_f32(ty_float, -2.0);

    // Global variables.
    let var_input = b.variable_global(ptr_sb_input, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_input, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_input, DECORATION_BINDING, &[0]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[1]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[var_gid]);
    b.execution_mode_local_size(func_id, ROPE_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let loaded_gid = b.load(ty_uvec3, var_gid);
    let gid = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load push constants.
    let pc_bh_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let dim_bh = b.load(ty_uint, pc_bh_ptr);
    let pc_seq_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let dim_seq = b.load(ty_uint, pc_seq_ptr);
    let pc_hd_ptr = b.access_chain(ptr_pc_uint, var_pc, &[const_2u]);
    let dim_hd = b.load(ty_uint, pc_hd_ptr);
    let pc_base_ptr = b.access_chain(ptr_pc_float, var_pc, &[const_3u]);
    let base_val = b.load(ty_float, pc_base_ptr);

    // half_dim = head_dim / 2
    let half_dim = b.udiv(ty_uint, dim_hd, const_2u);

    // total_pairs = batch_heads * seq_len * half_dim
    let seq_times_half = b.imul(ty_uint, dim_seq, half_dim);
    let total_pairs = b.imul(ty_uint, dim_bh, seq_times_half);

    // Bounds check: gid >= total_pairs -> return.
    let cmp_oob = b.u_greater_than_equal(ty_bool, gid, total_pairs);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    b.label_with_id(return_label);
    b.op_return();

    b.label_with_id(body_label);

    // Decode pair_idx -> (bh, pos, dim_i)
    // dim_i = gid % half_dim
    let dim_i = b.umod(ty_uint, gid, half_dim);
    // tmp = gid / half_dim
    let tmp = b.udiv(ty_uint, gid, half_dim);
    // pos = tmp % seq_len
    let pos = b.umod(ty_uint, tmp, dim_seq);
    // bh = tmp / seq_len
    let bh = b.udiv(ty_uint, tmp, dim_seq);

    // Compute theta = pos * base^(-2*dim_i / head_dim)
    // Using: theta = pos * exp(-2 * dim_i / head_dim * log(base))
    let dim_i_f = b.convert_u_to_f(ty_float, dim_i);
    let head_dim_f = b.convert_u_to_f(ty_float, dim_hd);
    let pos_f = b.convert_u_to_f(ty_float, pos);

    // log_base = log(base)
    let log_base = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_LOG, &[base_val]);
    // exponent = -2.0 * dim_i_f / head_dim_f * log_base
    let neg2_dim_i = b.fmul(ty_float, const_f_neg2, dim_i_f);
    let ratio = b.fdiv(ty_float, neg2_dim_i, head_dim_f);
    let exponent = b.fmul(ty_float, ratio, log_base);
    // freq = exp(exponent)
    let freq = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_EXP, &[exponent]);
    // theta = pos * freq
    let theta = b.fmul(ty_float, pos_f, freq);

    // cos_t = cos(theta), sin_t = sin(theta)
    let cos_t = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_COS, &[theta]);
    let sin_t = b.ext_inst(ty_float, glsl_ext, GLSL_STD_450_SIN, &[theta]);

    // base_idx = bh * seq_len * head_dim + pos * head_dim
    let seq_times_hd = b.imul(ty_uint, dim_seq, dim_hd);
    let bh_offset = b.imul(ty_uint, bh, seq_times_hd);
    let pos_offset = b.imul(ty_uint, pos, dim_hd);
    let base_idx = b.iadd(ty_uint, bh_offset, pos_offset);

    // Compute element indices based on standard vs NeoX layout.
    let (idx0, idx1) = if neox {
        // NeoX: pair (dim_i, dim_i + half_dim)
        let idx0 = b.iadd(ty_uint, base_idx, dim_i);
        let dim_i_plus_half = b.iadd(ty_uint, dim_i, half_dim);
        let idx1 = b.iadd(ty_uint, base_idx, dim_i_plus_half);
        (idx0, idx1)
    } else {
        // Standard: pair (2*dim_i, 2*dim_i + 1)
        let two_dim_i = b.imul(ty_uint, const_2u, dim_i);
        let idx0 = b.iadd(ty_uint, base_idx, two_dim_i);
        let two_dim_i_plus_1 = b.iadd(ty_uint, two_dim_i, const_1u);
        let idx1 = b.iadd(ty_uint, base_idx, two_dim_i_plus_1);
        (idx0, idx1)
    };

    // Load x0, x1 from input.
    let ptr_x0 = b.access_chain(ptr_sb_float, var_input, &[const_0u, idx0]);
    let x0 = b.load(ty_float, ptr_x0);
    let ptr_x1 = b.access_chain(ptr_sb_float, var_input, &[const_0u, idx1]);
    let x1 = b.load(ty_float, ptr_x1);

    // output[idx0] = x0 * cos_t - x1 * sin_t
    let x0_cos = b.fmul(ty_float, x0, cos_t);
    let x1_sin = b.fmul(ty_float, x1, sin_t);
    let out0 = b.fsub(ty_float, x0_cos, x1_sin);

    // output[idx1] = x0 * sin_t + x1 * cos_t
    let x0_sin = b.fmul(ty_float, x0, sin_t);
    let x1_cos = b.fmul(ty_float, x1, cos_t);
    let out1 = b.fadd(ty_float, x0_sin, x1_cos);

    // Store results.
    let ptr_out0 = b.access_chain(ptr_sb_float, var_output, &[const_0u, idx0]);
    b.store(ptr_out0, out0);
    let ptr_out1 = b.access_chain(ptr_sb_float, var_output, &[const_0u, idx1]);
    b.store(ptr_out1, out1);

    b.op_return();
    b.func_end();

    b.build()
}

/// CPU reference implementation of standard RoPE.
///
/// Applies rotary position embeddings to the input tensor using the standard
/// pairing scheme: dimension pairs `(2i, 2i+1)` are rotated together.
///
/// # Arguments
///
/// * `x` - Input tensor, flattened as `[batch_heads * seq_len * head_dim]`.
/// * `seq_len` - Number of positions in the sequence.
/// * `head_dim` - Dimension of each attention head (must be even).
/// * `base` - Base frequency (typically 10000.0).
///
/// # Returns
///
/// Output tensor of same shape with rotary embeddings applied.
pub fn rope_reference(x: &[f32], seq_len: usize, head_dim: usize, base: f32) -> Vec<f32> {
    assert!(head_dim.is_multiple_of(2), "head_dim must be even");
    let total = x.len();
    let stride = seq_len * head_dim;
    let batch_heads = total / stride;
    assert_eq!(
        batch_heads * stride,
        total,
        "input length must be divisible by seq_len * head_dim"
    );

    let half_dim = head_dim / 2;
    let mut out = x.to_vec();

    for bh in 0..batch_heads {
        for pos in 0..seq_len {
            for i in 0..half_dim {
                let exponent = -2.0 * (i as f32) / (head_dim as f32);
                let freq = base.powf(exponent);
                let theta = (pos as f32) * freq;
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                let base_idx = bh * stride + pos * head_dim;
                let idx0 = base_idx + 2 * i;
                let idx1 = base_idx + 2 * i + 1;

                let x0 = x[idx0];
                let x1 = x[idx1];
                out[idx0] = x0 * cos_t - x1 * sin_t;
                out[idx1] = x0 * sin_t + x1 * cos_t;
            }
        }
    }

    out
}

#[cfg(test)]
#[path = "spirv_rope_tests.rs"]
mod spirv_rope_tests;
