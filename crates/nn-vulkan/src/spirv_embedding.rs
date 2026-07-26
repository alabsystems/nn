// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for embedding lookup compute shaders.
//!
//! Generates a SPIR-V 1.0 binary module for the embedding lookup operation:
//!
//! ```text
//! output[t * embedding_dim + d] = embedding_table[token_ids[t] * embedding_dim + d]
//! ```
//!
//! Each thread handles one `(token, dim)` pair. A grid-stride loop allows
//! dispatching fewer threads than total elements (`num_tokens * embedding_dim`).
//!
//! # Buffer layout
//!
//! - Binding 0: `token_ids` — `uint[]` of length `num_tokens`
//! - Binding 1: `embedding_table` — `float[vocab_size * embedding_dim]`
//! - Binding 2: `output` — `float[num_tokens * embedding_dim]`
//!
//! # Push constants
//!
//! - `uint num_tokens` at offset 0
//! - `uint vocab_size` at offset 4
//! - `uint embedding_dim` at offset 8
//!
//! # Algorithm
//!
//! ```text
//! gid = gl_GlobalInvocationID.x
//! total = num_tokens * embedding_dim
//! grid_size = gl_NumWorkGroups.x * WORKGROUP_SIZE
//!
//! for i in (gid..total).step_by(grid_size):
//!     t = i / embedding_dim
//!     d = i % embedding_dim
//!     tok = token_ids[t]
//!     if tok < vocab_size:
//!         output[i] = embedding_table[tok * embedding_dim + d]
//!     else:
//!         output[i] = 0.0   // out-of-vocab → zero
//! ```
//!
//! The bounds check on `vocab_size` prevents out-of-bounds reads into the
//! embedding table. Out-of-vocabulary tokens produce zero vectors.

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for embedding lookup kernels (1D dispatch).
pub const EMBEDDING_WORKGROUP_SIZE: u32 = 256;

// ---- SPIR-V constants (duplicated to keep module independent) ----

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
const OP_U_LESS_THAN: u16 = 176;
const OP_IADD: u16 = 128;
const OP_IMUL: u16 = 132;
const OP_UDIV: u16 = 134;
const OP_UMOD: u16 = 137;

// Decorations.
const DECORATION_BUILTIN: u32 = 11;
const DECORATION_BINDING: u32 = 33;
const DECORATION_DESCRIPTOR_SET: u32 = 34;
const DECORATION_OFFSET: u32 = 35;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_BLOCK: u32 = 2;

// Built-ins.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_NUM_WORKGROUPS: u32 = 24;

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

/// SPIR-V module builder (local to this module, mirrors spirv_layernorm.rs).
struct SpirVBuilder {
    bound: u32,
    capabilities: Vec<u32>,
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

    fn u_less_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_LESS_THAN));
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

    fn build(self) -> Vec<u32> {
        let mut module = Vec::with_capacity(256);
        module.push(SPIRV_MAGIC);
        module.push(SPIRV_VERSION_1_0);
        module.push(GENERATOR_MAGIC);
        module.push(self.bound);
        module.push(0);
        module.extend_from_slice(&self.capabilities);
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
        let opcode = (word & 0xFFFF) as u16;
        if word_count == 0 {
            break;
        }
        if opcode == OP_PHI && pos + 2 < functions.len() && functions[pos + 2] == phi_id {
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

/// Generate a SPIR-V 1.0 binary for embedding lookup.
///
/// Each thread processes one `(token, dim)` pair using a grid-stride loop,
/// looking up the token's embedding vector from the table and writing it
/// to the output buffer. Tokens outside `[0, vocab_size)` produce zero vectors.
///
/// # Arguments
///
/// * `vocab_size` - Number of vocabulary entries (compile-time hint; actual from push constants).
/// * `embedding_dim` - Dimension of each embedding vector (compile-time hint; actual from push constants).
///
/// # Buffers
///
/// - Binding 0: `token_ids` — `uint[]` of length `num_tokens`
/// - Binding 1: `embedding_table` — `float[vocab_size * embedding_dim]`
/// - Binding 2: `output` — `float[num_tokens * embedding_dim]`
///
/// # Push constants
///
/// - `uint num_tokens` at offset 0
/// - `uint vocab_size` at offset 4
/// - `uint embedding_dim` at offset 8
///
/// # Example
///
/// ```
/// use nn_vulkan::spirv_embedding::generate_embedding_spirv;
/// let spirv = generate_embedding_spirv(32000, 768);
/// assert_eq!(spirv.len() % 4, 0); // 4-byte aligned
/// let magic = u32::from_le_bytes([spirv[0], spirv[1], spirv[2], spirv[3]]);
/// assert_eq!(magic, 0x07230203);
/// ```
pub fn generate_embedding_spirv(vocab_size: u32, embedding_dim: u32) -> Vec<u8> {
    let _ = vocab_size; // hint; actual dim from push constants.
    let _ = embedding_dim; // hint; actual dim from push constants.

    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    // ---- Types ----
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    // Runtime arrays.
    let ty_rtarr_uint = b.type_runtime_array(ty_uint);
    b.decorate(ty_rtarr_uint, DECORATION_ARRAY_STRIDE, &[4]);

    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Buffer structs.
    // Binding 0: token_ids { uint data[]; }
    let ty_struct_token_ids = b.type_struct(&[ty_rtarr_uint]);
    b.decorate(ty_struct_token_ids, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_token_ids, 0, DECORATION_OFFSET, &[0]);

    // Binding 1: embedding_table { float data[]; }
    let ty_struct_embed_table = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_embed_table, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_embed_table, 0, DECORATION_OFFSET, &[0]);

    // Binding 2: output { float data[]; }
    let ty_struct_output = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_output, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_output, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint num_tokens; uint vocab_size; uint embedding_dim; }
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint, ty_uint]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);

    // ---- Pointer types ----
    let ptr_sb_token_ids = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_token_ids);
    let ptr_sb_embed_table = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_embed_table);
    let ptr_sb_output = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_output);
    let ptr_pc = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_struct_pc);
    let ptr_input_uvec3 = b.type_pointer(STORAGE_CLASS_INPUT, ty_uvec3);
    let ptr_sb_uint = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_uint);
    let ptr_sb_float = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_float);
    let ptr_pc_uint = b.type_pointer(STORAGE_CLASS_PUSH_CONSTANT, ty_uint);

    // ---- Constants ----
    let const_0u = b.constant_u32(ty_uint, 0);
    let const_1u = b.constant_u32(ty_uint, 1);
    let const_2u = b.constant_u32(ty_uint, 2);
    let const_f0 = b.constant_f32(ty_float, 0.0);
    let const_wg_size = b.constant_u32(ty_uint, EMBEDDING_WORKGROUP_SIZE);

    // ---- Global variables ----
    let var_token_ids = b.variable_global(ptr_sb_token_ids, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_token_ids, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_token_ids, DECORATION_BINDING, &[0]);

    let var_embed_table = b.variable_global(ptr_sb_embed_table, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_embed_table, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_embed_table, DECORATION_BINDING, &[1]);

    let var_output = b.variable_global(ptr_sb_output, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_output, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_output, DECORATION_BINDING, &[2]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_global_id = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(
        var_global_id,
        DECORATION_BUILTIN,
        &[BUILTIN_GLOBAL_INVOCATION_ID],
    );

    let var_num_wg = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_num_wg, DECORATION_BUILTIN, &[BUILTIN_NUM_WORKGROUPS]);

    // Entry point — list all Input interface variables.
    b.entry_point_compute(func_id, "main", &[var_global_id, var_num_wg]);
    b.execution_mode_local_size(func_id, EMBEDDING_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(ty_void, func_id, FUNCTION_CONTROL_NONE, ty_fn_void);
    let entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let loaded_gid = b.load(ty_uvec3, var_global_id);
    let gid_x = b.composite_extract(ty_uint, loaded_gid, 0);

    // Load gl_NumWorkGroups.x
    let loaded_nwg = b.load(ty_uvec3, var_num_wg);
    let nwg_x = b.composite_extract(ty_uint, loaded_nwg, 0);

    // grid_size = nwg_x * WORKGROUP_SIZE
    let grid_size = b.imul(ty_uint, nwg_x, const_wg_size);

    // Load push constants: num_tokens, vocab_size, embedding_dim.
    let pc_ptr_num_tokens = b.access_chain(ptr_pc_uint, var_pc, &[const_0u]);
    let num_tokens = b.load(ty_uint, pc_ptr_num_tokens);

    let pc_ptr_vocab_size = b.access_chain(ptr_pc_uint, var_pc, &[const_1u]);
    let pc_vocab_size = b.load(ty_uint, pc_ptr_vocab_size);

    let pc_ptr_embed_dim = b.access_chain(ptr_pc_uint, var_pc, &[const_2u]);
    let embed_dim = b.load(ty_uint, pc_ptr_embed_dim);

    // total = num_tokens * embedding_dim
    let total = b.imul(ty_uint, num_tokens, embed_dim);

    // ---- Grid-stride loop ----
    // Pre-allocate labels.
    let loop_header = b.id();
    let loop_body = b.id();
    let loop_continue = b.id();
    let loop_merge = b.id();

    b.branch(loop_header);

    // Loop header.
    b.label_with_id(loop_header);
    b.loop_merge(loop_merge, loop_continue);

    // phi: i = gid_x on entry, i_next from continue block.
    let phi_i = b.phi(ty_uint, &[(gid_x, entry_label)]);

    // Loop condition: i < total.
    let cmp_loop = b.u_less_than(ty_bool, phi_i, total);
    b.branch_conditional(cmp_loop, loop_body, loop_merge);

    // Loop body.
    b.label_with_id(loop_body);

    // t = i / embedding_dim
    let t = b.udiv(ty_uint, phi_i, embed_dim);
    // d = i % embedding_dim
    let d = b.umod(ty_uint, phi_i, embed_dim);

    // Load token_ids[t].
    let ptr_tok = b.access_chain(ptr_sb_uint, var_token_ids, &[const_0u, t]);
    let tok = b.load(ty_uint, ptr_tok);

    // Bounds check: tok < vocab_size.
    let cmp_vocab = b.u_less_than(ty_bool, tok, pc_vocab_size);
    let in_vocab_label = b.id();
    let oov_label = b.id();
    let store_label = b.id();
    b.selection_merge(store_label);
    b.branch_conditional(cmp_vocab, in_vocab_label, oov_label);

    // ---- In-vocab branch: load from embedding table ----
    b.label_with_id(in_vocab_label);
    // table_idx = tok * embedding_dim + d
    let tok_times_dim = b.imul(ty_uint, tok, embed_dim);
    let table_idx = b.iadd(ty_uint, tok_times_dim, d);
    let ptr_embed = b.access_chain(ptr_sb_float, var_embed_table, &[const_0u, table_idx]);
    let embed_val = b.load(ty_float, ptr_embed);
    b.branch(store_label);

    // ---- Out-of-vocab branch: zero ----
    b.label_with_id(oov_label);
    b.branch(store_label);

    // ---- Store merge: phi selects embed_val or 0.0 ----
    b.label_with_id(store_label);
    let result_val = b.phi(
        ty_float,
        &[(embed_val, in_vocab_label), (const_f0, oov_label)],
    );

    // Store to output[i].
    let ptr_out = b.access_chain(ptr_sb_float, var_output, &[const_0u, phi_i]);
    b.store(ptr_out, result_val);

    // ---- Continue block: i_next = i + grid_size ----
    b.branch(loop_continue);
    b.label_with_id(loop_continue);
    let i_next = b.iadd(ty_uint, phi_i, grid_size);
    b.branch(loop_header);

    // Fixup phi for the loop induction variable.
    fixup_phi(&mut b.functions, phi_i, i_next, loop_continue);

    // ---- Loop merge: return ----
    b.label_with_id(loop_merge);
    b.op_return();
    b.func_end();

    let words = b.build();
    words_to_bytes(&words)
}

#[cfg(test)]
#[path = "spirv_embedding_tests.rs"]
mod spirv_embedding_tests;
