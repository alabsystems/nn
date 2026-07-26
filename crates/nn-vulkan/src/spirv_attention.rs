// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! SPIR-V binary generation for scaled dot-product attention compute shaders.
//!
//! Generates SPIR-V 1.0 binary modules for the standard attention operation:
//!
//! ```text
//! score  = Q @ K^T * scale
//! if causal: score[i][j] = -inf  where j > i
//! weights = softmax(score, dim=-1)
//! output  = weights @ V
//! ```
//!
//! The [`generate_attention_spirv`] function produces a single fused kernel that
//! performs all three stages (score computation, optional causal masking + softmax,
//! and value weighting) without materializing intermediate tensors.
//!
//! # Buffer layout
//!
//! - Binding 0: Q \[batch * num\_heads, seq\_len, head\_dim\] (row-major float\[\])
//! - Binding 1: K \[batch * num\_heads, seq\_len, head\_dim\] (row-major float\[\])
//! - Binding 2: V \[batch * num\_heads, seq\_len, head\_dim\] (row-major float\[\])
//! - Binding 3: Output \[batch * num\_heads, seq\_len, head\_dim\] (row-major float\[\])
//!
//! # Push constants
//!
//! - `uint batch_heads` at offset 0  (batch * num\_heads)
//! - `uint seq_len` at offset 4
//! - `uint head_dim` at offset 8
//! - `float scale_bits` at offset 12  (scale as f32 bit pattern, typically 1/sqrt(head\_dim))
//!
//! # Algorithm
//!
//! Each invocation handles one output element at position `(bh, row, d)`:
//!
//! ```text
//! gid = gl_GlobalInvocationID.x
//! d   = gid % head_dim
//! row = (gid / head_dim) % seq_len
//! bh  = gid / (seq_len * head_dim)
//!
//! // Phase 1: compute row-max of scores for numerical stability
//! max_score = -inf
//! for col in 0..seq_len:
//!     score = 0.0
//!     for k in 0..head_dim:
//!         score += Q[bh, row, k] * K[bh, col, k]
//!     score *= scale
//!     if causal && col > row: score = -inf
//!     max_score = fmax(max_score, score)
//!
//! // Phase 2: compute exp-sum for softmax denominator
//! exp_sum = 0.0
//! for col in 0..seq_len:
//!     score = dot(Q[bh, row, :], K[bh, col, :]) * scale
//!     if causal && col > row: score = -inf
//!     exp_sum += exp(score - max_score)
//!
//! // Phase 3: output = sum_col softmax_weight[col] * V[bh, col, d]
//! output_val = 0.0
//! for col in 0..seq_len:
//!     score = dot(Q[bh, row, :], K[bh, col, :]) * scale
//!     if causal && col > row: score = -inf
//!     weight = exp(score - max_score) / exp_sum
//!     output_val += weight * V[bh, col, d]
//!
//! Output[bh, row, d] = output_val
//! ```
//!
//! This is a straightforward O(seq_len^2 * head_dim) implementation suitable for
//! moderate sequence lengths. For long sequences, tiled/flash-attention variants
//! would reduce memory bandwidth; those are future work.

use crate::spirv_emit::SPIRV_MAGIC;

/// Workgroup size for the attention kernel (1D dispatch).
pub const ATTENTION_WORKGROUP_SIZE: u32 = 256;

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
const OP_BRANCH: u16 = 249;
const OP_BRANCH_CONDITIONAL: u16 = 250;
const OP_SELECTION_MERGE: u16 = 247;
const OP_LOOP_MERGE: u16 = 246;
const OP_PHI: u16 = 245;
const OP_COMPOSITE_EXTRACT: u16 = 81;
const OP_FADD: u16 = 129;
const OP_FSUB: u16 = 131;
const OP_FMUL: u16 = 133;
const OP_FDIV: u16 = 136;
const OP_U_LESS_THAN: u16 = 176;
const OP_U_GREATER_THAN_EQUAL: u16 = 174;
const OP_IMUL: u16 = 132;
const OP_IADD: u16 = 128;
const OP_EXT_INST: u16 = 12;
const OP_SELECT: u16 = 169;

// Inline opcodes not in the builder.
const OP_UDIV: u16 = 134;
const OP_UMOD: u16 = 137;
const OP_U_GREATER_THAN: u16 = 172;

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
const GLSL_STD_450_EXP: u32 = 27;
const GLSL_STD_450_FMAX: u32 = 40;

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

/// SPIR-V module builder (local to this module, mirrors spirv_binary.rs).
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
        self.functions.push(0); // SelectionControl: None
    }

    fn loop_merge(&mut self, merge_label: u32, continue_label: u32) {
        self.functions.push(op(4, OP_LOOP_MERGE));
        self.functions.push(merge_label);
        self.functions.push(continue_label);
        self.functions.push(0); // LoopControl: None
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

    /// OpUDiv: unsigned integer divide.
    fn udiv(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UDIV));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    /// OpUMod: unsigned integer modulo.
    fn umod(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_UMOD));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    /// OpUGreaterThan: unsigned integer a > b.
    fn u_greater_than(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(5, OP_U_GREATER_THAN));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(a);
        self.functions.push(b);
        result
    }

    /// OpSelect: select between two values based on a boolean condition.
    fn select(&mut self, result_type: u32, condition: u32, true_val: u32, false_val: u32) -> u32 {
        let result = self.id();
        self.functions.push(op(6, OP_SELECT));
        self.functions.push(result_type);
        self.functions.push(result);
        self.functions.push(condition);
        self.functions.push(true_val);
        self.functions.push(false_val);
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
        let mut module = Vec::with_capacity(1024);
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

/// Fixup a phi instruction in a Vec (supports insertion).
fn fixup_phi_vec(functions: &mut Vec<u32>, phi_id: u32, value: u32, parent: u32) {
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

// ---- Type setup ----

/// Common types and variables for the attention shader.
struct AttentionSetup {
    ty_void: u32,
    ty_float: u32,
    ty_uint: u32,
    ty_bool: u32,
    ty_fn_void: u32,
    ptr_sb_float: u32,
    ptr_pc_uint: u32,
    ptr_pc_float: u32,
    glsl_ext: u32,
    const_0u: u32,
    const_1u: u32,
    const_2u: u32,
    const_3u: u32,
    const_neg_inf: u32,
    const_f0: u32,
    var_buf_q: u32,
    var_buf_k: u32,
    var_buf_v: u32,
    var_buf_out: u32,
    var_pc: u32,
    var_gid: u32,
}

/// Set up types, decorations, and global variables for the attention shader.
///
/// Push constant layout:
/// ```text
/// { uint batch_heads, uint seq_len, uint head_dim, float scale }
/// ```
///
/// Note: The last push constant member is float (not uint). We declare the
/// struct with 3 uint + 1 float members. The scale is loaded via a float
/// pointer at offset 12.
fn setup_attention_types(b: &mut SpirVBuilder) -> AttentionSetup {
    let ty_void = b.type_void();
    let ty_float = b.type_float(32);
    let ty_uint = b.type_int(32, 0);
    let ty_bool = b.type_bool();
    let ty_uvec3 = b.type_vector(ty_uint, 3);
    let ty_fn_void = b.type_function(ty_void, &[]);

    let glsl_ext = b.ext_inst_import("GLSL.std.450");

    // Runtime arrays of float for storage buffers.
    let ty_rtarr_float = b.type_runtime_array(ty_float);
    b.decorate(ty_rtarr_float, DECORATION_ARRAY_STRIDE, &[4]);

    // Q buffer struct.
    let ty_struct_q = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_q, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_q, 0, DECORATION_OFFSET, &[0]);

    // K buffer struct.
    let ty_struct_k = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_k, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_k, 0, DECORATION_OFFSET, &[0]);

    // V buffer struct.
    let ty_struct_v = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_v, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_v, 0, DECORATION_OFFSET, &[0]);

    // Output buffer struct.
    let ty_struct_out = b.type_struct(&[ty_rtarr_float]);
    b.decorate(ty_struct_out, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_out, 0, DECORATION_OFFSET, &[0]);

    // Push constant struct: { uint batch_heads, uint seq_len, uint head_dim, float scale }
    // We use uint for the first 3, float for the last. SPIR-V struct members
    // can be mixed types.
    let ty_struct_pc = b.type_struct(&[ty_uint, ty_uint, ty_uint, ty_float]);
    b.decorate(ty_struct_pc, DECORATION_BLOCK, &[]);
    b.member_decorate(ty_struct_pc, 0, DECORATION_OFFSET, &[0]);
    b.member_decorate(ty_struct_pc, 1, DECORATION_OFFSET, &[4]);
    b.member_decorate(ty_struct_pc, 2, DECORATION_OFFSET, &[8]);
    b.member_decorate(ty_struct_pc, 3, DECORATION_OFFSET, &[12]);

    // Pointer types.
    let ptr_sb_q = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_q);
    let ptr_sb_k = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_k);
    let ptr_sb_v = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_v);
    let ptr_sb_out = b.type_pointer(STORAGE_CLASS_STORAGE_BUFFER, ty_struct_out);
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
    let const_neg_inf = b.constant_f32(ty_float, f32::NEG_INFINITY);
    let const_f0 = b.constant_f32(ty_float, 0.0);

    // Global variables.
    let var_buf_q = b.variable_global(ptr_sb_q, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_q, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_q, DECORATION_BINDING, &[0]);

    let var_buf_k = b.variable_global(ptr_sb_k, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_k, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_k, DECORATION_BINDING, &[1]);

    let var_buf_v = b.variable_global(ptr_sb_v, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_v, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_v, DECORATION_BINDING, &[2]);

    let var_buf_out = b.variable_global(ptr_sb_out, STORAGE_CLASS_STORAGE_BUFFER);
    b.decorate(var_buf_out, DECORATION_DESCRIPTOR_SET, &[0]);
    b.decorate(var_buf_out, DECORATION_BINDING, &[3]);

    let var_pc = b.variable_global(ptr_pc, STORAGE_CLASS_PUSH_CONSTANT);

    let var_gid = b.variable_global(ptr_input_uvec3, STORAGE_CLASS_INPUT);
    b.decorate(var_gid, DECORATION_BUILTIN, &[BUILTIN_GLOBAL_INVOCATION_ID]);

    AttentionSetup {
        ty_void,
        ty_float,
        ty_uint,
        ty_bool,
        ty_fn_void,
        ptr_sb_float,
        ptr_pc_uint,
        ptr_pc_float,
        glsl_ext,
        const_0u,
        const_1u,
        const_2u,
        const_3u,
        const_neg_inf,
        const_f0,
        var_buf_q,
        var_buf_k,
        var_buf_v,
        var_buf_out,
        var_pc,
        var_gid,
    }
}

/// Generate a SPIR-V 1.0 binary for scaled dot-product attention.
///
/// Produces a fused kernel that computes:
/// ```text
/// score  = Q @ K^T * scale
/// if causal: score[i][j] = -inf where j > i
/// weights = softmax(score, dim=-1)
/// output  = weights @ V
/// ```
///
/// Each thread computes one element of the output at position `(bh, row, d)`.
/// The kernel iterates over the sequence dimension three times per output
/// element (for max, exp-sum, and weighted-sum), recomputing scores on the
/// fly to avoid materializing the full `[seq_len, seq_len]` attention matrix.
///
/// # Arguments
///
/// * `_head_dim` - Dimension of each attention head (compile-time hint; runtime
///   from push constants). Used only for documentation/dispatch planning.
/// * `causal` - If `true`, generates causal masking logic that sets scores
///   at positions `col > row` to `-inf` before softmax.
///
/// # Returns
///
/// SPIR-V binary as `Vec<u8>`, ready for Vulkan pipeline creation.
pub fn generate_attention_spirv(_head_dim: usize, causal: bool) -> Vec<u8> {
    let mut b = SpirVBuilder::new();
    let func_id = b.id();

    b.capability(CAPABILITY_SHADER);
    b.memory_model(ADDRESSING_LOGICAL, MEMORY_MODEL_GLSL450);

    let s = setup_attention_types(&mut b);

    // Entry point.
    b.entry_point_compute(func_id, "main", &[s.var_gid]);
    b.execution_mode_local_size(func_id, ATTENTION_WORKGROUP_SIZE, 1, 1);

    // ---- Function body ----
    b.func_begin(s.ty_void, func_id, FUNCTION_CONTROL_NONE, s.ty_fn_void);
    let _entry_label = b.label();

    // Load gl_GlobalInvocationID.x
    let ty_uvec3 = b.type_vector(s.ty_uint, 3);
    let loaded_gid = b.load(ty_uvec3, s.var_gid);
    let gid = b.composite_extract(s.ty_uint, loaded_gid, 0);

    // Load push constants.
    let pc_bh_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_0u]);
    let dim_bh = b.load(s.ty_uint, pc_bh_ptr);
    let pc_seq_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_1u]);
    let dim_seq = b.load(s.ty_uint, pc_seq_ptr);
    let pc_hd_ptr = b.access_chain(s.ptr_pc_uint, s.var_pc, &[s.const_2u]);
    let dim_hd = b.load(s.ty_uint, pc_hd_ptr);
    let pc_scale_ptr = b.access_chain(s.ptr_pc_float, s.var_pc, &[s.const_3u]);
    let scale = b.load(s.ty_float, pc_scale_ptr);

    // Compute total output elements = batch_heads * seq_len * head_dim.
    let seq_times_hd = b.imul(s.ty_uint, dim_seq, dim_hd);
    let total_out = b.imul(s.ty_uint, dim_bh, seq_times_hd);

    // Bounds check: gid >= total_out -> return.
    let cmp_oob = b.u_greater_than_equal(s.ty_bool, gid, total_out);
    let return_label = b.id();
    let body_label = b.id();
    b.selection_merge(body_label);
    b.branch_conditional(cmp_oob, return_label, body_label);

    // Early return for out-of-bounds threads.
    b.label_with_id(return_label);
    b.op_return();

    // Body: gid < total_out.
    b.label_with_id(body_label);

    // Decode (bh, row, d) from flat gid.
    // d   = gid % head_dim
    let d = b.umod(s.ty_uint, gid, dim_hd);
    // tmp = gid / head_dim
    let tmp = b.udiv(s.ty_uint, gid, dim_hd);
    // row = tmp % seq_len
    let row = b.umod(s.ty_uint, tmp, dim_seq);
    // bh  = tmp / seq_len
    let bh = b.udiv(s.ty_uint, tmp, dim_seq);

    // Precompute base offset for this batch-head: bh * seq_len * head_dim.
    let bh_offset = b.imul(s.ty_uint, bh, seq_times_hd);

    // row_offset_q = bh_offset + row * head_dim (base of Q[bh, row, :]).
    let row_times_hd = b.imul(s.ty_uint, row, dim_hd);
    let row_offset_q = b.iadd(s.ty_uint, bh_offset, row_times_hd);

    // ===========================================================
    // Phase 1: Find max score across all columns for numerical stability.
    // max_score = max over col of (dot(Q[bh,row,:], K[bh,col,:]) * scale)
    //   with causal masking applied if enabled.
    // ===========================================================

    let p1_header = b.id();
    let p1_body = b.id();
    let p1_continue = b.id();
    let p1_merge = b.id();

    b.branch(p1_header);

    // Phase 1 loop header.
    b.label_with_id(p1_header);
    b.loop_merge(p1_merge, p1_continue);
    let phi_col1 = b.phi(s.ty_uint, &[(s.const_0u, body_label)]);
    let phi_max = b.phi(s.ty_float, &[(s.const_neg_inf, body_label)]);
    let cmp_col1 = b.u_less_than(s.ty_bool, phi_col1, dim_seq);
    b.branch_conditional(cmp_col1, p1_body, p1_merge);

    // Phase 1 loop body: compute score for (row, col).
    b.label_with_id(p1_body);

    // col_offset_k = bh_offset + col * head_dim
    let col1_times_hd = b.imul(s.ty_uint, phi_col1, dim_hd);
    let col1_offset_k = b.iadd(s.ty_uint, bh_offset, col1_times_hd);

    // Inner dot product: score = sum_k Q[bh, row, k] * K[bh, col, k]
    let dot1 = emit_dot_product(&mut b, &s, row_offset_q, col1_offset_k, dim_hd, p1_body);

    // score = dot * scale
    let score1 = b.fmul(s.ty_float, dot1, scale);

    // Causal masking: if col > row, score = -inf.
    let masked_score1 = if causal {
        let col_gt_row = b.u_greater_than(s.ty_bool, phi_col1, row);
        b.select(s.ty_float, col_gt_row, s.const_neg_inf, score1)
    } else {
        score1
    };

    // max_score = fmax(max_score, masked_score)
    let new_max = b.ext_inst(
        s.ty_float,
        s.glsl_ext,
        GLSL_STD_450_FMAX,
        &[phi_max, masked_score1],
    );

    // Phase 1 continue: increment col.
    b.branch(p1_continue);
    b.label_with_id(p1_continue);
    let next_col1 = b.iadd(s.ty_uint, phi_col1, s.const_1u);
    // Fixup phi nodes with back-edge values.
    fixup_phi_vec(&mut b.functions, phi_col1, next_col1, p1_continue);
    fixup_phi_vec(&mut b.functions, phi_max, new_max, p1_continue);
    b.branch(p1_header);

    // Phase 1 merge.
    b.label_with_id(p1_merge);

    // ===========================================================
    // Phase 2: Compute exp-sum = sum over col of exp(score - max_score).
    // ===========================================================

    let p2_header = b.id();
    let p2_body = b.id();
    let p2_continue = b.id();
    let p2_merge = b.id();

    b.branch(p2_header);

    b.label_with_id(p2_header);
    b.loop_merge(p2_merge, p2_continue);
    let phi_col2 = b.phi(s.ty_uint, &[(s.const_0u, p1_merge)]);
    let phi_expsum = b.phi(s.ty_float, &[(s.const_f0, p1_merge)]);
    let cmp_col2 = b.u_less_than(s.ty_bool, phi_col2, dim_seq);
    b.branch_conditional(cmp_col2, p2_body, p2_merge);

    b.label_with_id(p2_body);

    // Recompute score for (row, col).
    let col2_times_hd = b.imul(s.ty_uint, phi_col2, dim_hd);
    let col2_offset_k = b.iadd(s.ty_uint, bh_offset, col2_times_hd);

    let dot2 = emit_dot_product(&mut b, &s, row_offset_q, col2_offset_k, dim_hd, p2_body);

    let score2 = b.fmul(s.ty_float, dot2, scale);

    let masked_score2 = if causal {
        let col_gt_row2 = b.u_greater_than(s.ty_bool, phi_col2, row);
        b.select(s.ty_float, col_gt_row2, s.const_neg_inf, score2)
    } else {
        score2
    };

    // exp(score - max_score)
    let diff2 = b.fsub(s.ty_float, masked_score2, phi_max);
    let exp_val2 = b.ext_inst(s.ty_float, s.glsl_ext, GLSL_STD_450_EXP, &[diff2]);
    let new_expsum = b.fadd(s.ty_float, phi_expsum, exp_val2);

    b.branch(p2_continue);
    b.label_with_id(p2_continue);
    let next_col2 = b.iadd(s.ty_uint, phi_col2, s.const_1u);
    fixup_phi_vec(&mut b.functions, phi_col2, next_col2, p2_continue);
    fixup_phi_vec(&mut b.functions, phi_expsum, new_expsum, p2_continue);
    b.branch(p2_header);

    b.label_with_id(p2_merge);

    // ===========================================================
    // Phase 3: Compute output = sum over col of softmax_weight * V[bh, col, d].
    //   softmax_weight = exp(score - max_score) / exp_sum
    // ===========================================================

    let p3_header = b.id();
    let p3_body = b.id();
    let p3_continue = b.id();
    let p3_merge = b.id();

    b.branch(p3_header);

    b.label_with_id(p3_header);
    b.loop_merge(p3_merge, p3_continue);
    let phi_col3 = b.phi(s.ty_uint, &[(s.const_0u, p2_merge)]);
    let phi_output = b.phi(s.ty_float, &[(s.const_f0, p2_merge)]);
    let cmp_col3 = b.u_less_than(s.ty_bool, phi_col3, dim_seq);
    b.branch_conditional(cmp_col3, p3_body, p3_merge);

    b.label_with_id(p3_body);

    // Recompute score for (row, col).
    let col3_times_hd = b.imul(s.ty_uint, phi_col3, dim_hd);
    let col3_offset_k = b.iadd(s.ty_uint, bh_offset, col3_times_hd);

    let dot3 = emit_dot_product(&mut b, &s, row_offset_q, col3_offset_k, dim_hd, p3_body);

    let score3 = b.fmul(s.ty_float, dot3, scale);

    let masked_score3 = if causal {
        let col_gt_row3 = b.u_greater_than(s.ty_bool, phi_col3, row);
        b.select(s.ty_float, col_gt_row3, s.const_neg_inf, score3)
    } else {
        score3
    };

    // weight = exp(score - max) / exp_sum
    let diff3 = b.fsub(s.ty_float, masked_score3, phi_max);
    let exp_val3 = b.ext_inst(s.ty_float, s.glsl_ext, GLSL_STD_450_EXP, &[diff3]);
    let weight3 = b.fdiv(s.ty_float, exp_val3, phi_expsum);

    // Load V[bh, col, d].
    let col3_offset_v = b.iadd(s.ty_uint, bh_offset, col3_times_hd);
    let v_idx = b.iadd(s.ty_uint, col3_offset_v, d);
    let v_ptr = b.access_chain(s.ptr_sb_float, s.var_buf_v, &[s.const_0u, v_idx]);
    let v_val = b.load(s.ty_float, v_ptr);

    // output_val += weight * V
    let weighted_v = b.fmul(s.ty_float, weight3, v_val);
    let new_output = b.fadd(s.ty_float, phi_output, weighted_v);

    b.branch(p3_continue);
    b.label_with_id(p3_continue);
    let next_col3 = b.iadd(s.ty_uint, phi_col3, s.const_1u);
    fixup_phi_vec(&mut b.functions, phi_col3, next_col3, p3_continue);
    fixup_phi_vec(&mut b.functions, phi_output, new_output, p3_continue);
    b.branch(p3_header);

    b.label_with_id(p3_merge);

    // Store output[bh, row, d] = output_val.
    let out_idx = b.iadd(s.ty_uint, row_offset_q, d);
    let out_ptr = b.access_chain(s.ptr_sb_float, s.var_buf_out, &[s.const_0u, out_idx]);
    b.store(out_ptr, phi_output);

    b.op_return();
    b.func_end();

    words_to_bytes(&b.build())
}

/// Emit a dot product between Q[row_offset..row_offset+dim] and K[col_offset..col_offset+dim].
///
/// Returns the SPIR-V ID holding the dot product result. Uses a loop with phi
/// for the accumulator.
fn emit_dot_product(
    b: &mut SpirVBuilder,
    s: &AttentionSetup,
    row_offset: u32,
    col_offset: u32,
    dim: u32,
    entry_from: u32,
) -> u32 {
    let dot_header = b.id();
    let dot_body = b.id();
    let dot_continue = b.id();
    let dot_merge = b.id();

    b.branch(dot_header);

    // Loop header.
    b.label_with_id(dot_header);
    b.loop_merge(dot_merge, dot_continue);
    let phi_k = b.phi(s.ty_uint, &[(s.const_0u, entry_from)]);
    let phi_acc = b.phi(s.ty_float, &[(s.const_f0, entry_from)]);
    let cmp_k = b.u_less_than(s.ty_bool, phi_k, dim);
    b.branch_conditional(cmp_k, dot_body, dot_merge);

    // Loop body: acc += Q[row_offset + k] * K[col_offset + k]
    b.label_with_id(dot_body);

    let q_idx = b.iadd(s.ty_uint, row_offset, phi_k);
    let q_ptr = b.access_chain(s.ptr_sb_float, s.var_buf_q, &[s.const_0u, q_idx]);
    let q_val = b.load(s.ty_float, q_ptr);

    let k_idx = b.iadd(s.ty_uint, col_offset, phi_k);
    let k_ptr = b.access_chain(s.ptr_sb_float, s.var_buf_k, &[s.const_0u, k_idx]);
    let k_val = b.load(s.ty_float, k_ptr);

    let prod = b.fmul(s.ty_float, q_val, k_val);
    let new_acc = b.fadd(s.ty_float, phi_acc, prod);

    // Continue: increment k.
    b.branch(dot_continue);
    b.label_with_id(dot_continue);
    let next_k = b.iadd(s.ty_uint, phi_k, s.const_1u);
    fixup_phi_vec(&mut b.functions, phi_k, next_k, dot_continue);
    fixup_phi_vec(&mut b.functions, phi_acc, new_acc, dot_continue);
    b.branch(dot_header);

    // Merge: phi_acc holds the final dot product.
    b.label_with_id(dot_merge);

    // Return the accumulator phi; at the merge block, phi_acc holds the result.
    phi_acc
}

#[cfg(test)]
#[path = "spirv_attention_tests.rs"]
mod spirv_attention_tests;
