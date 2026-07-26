// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Llama/Llama2/Llama3 architecture builder for GGUF models.
//!
//! Constructs a [`ComputationGraph`] from GGUF metadata that can be
//! compiled by `compile_trace_to_plan_with_fusion()`. Weight data is
//! loaded separately via [`GgufFile::read_tensor_f32()`]; this module
//! builds the graph structure with shape-only [`WeightRef`]s.

use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp, WeightRef};
use nn_core::DType;

use crate::error::GgufError;
use crate::reader::GgufFile;

/// Llama model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_base: f32,
    pub max_seq_len: usize,
}

impl LlamaConfig {
    /// Extract Llama config from GGUF metadata.
    ///
    /// Reads standard `llama.*` metadata keys. Falls back to deriving
    /// `vocab_size` from the `token_embd.weight` tensor shape when the
    /// metadata key is absent.
    pub fn from_gguf(file: &GgufFile) -> Result<Self, GgufError> {
        // Verify architecture is llama (if present).
        if let Some(arch) = file.architecture() {
            if arch != "llama" {
                return Err(GgufError::ArchitectureMismatch {
                    expected: "llama".to_string(),
                    found: arch.to_string(),
                });
            }
        }

        let hidden_dim = require_u32_meta(&file.metadata, "llama.embedding_length")? as usize;
        let num_layers = require_u32_meta(&file.metadata, "llama.block_count")? as usize;
        let num_heads = require_u32_meta(&file.metadata, "llama.attention.head_count")? as usize;
        let num_kv_heads = file
            .metadata
            .get_u32("llama.attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(num_heads);

        let head_dim = hidden_dim / num_heads;

        let ffn_dim = require_u32_meta(&file.metadata, "llama.feed_forward_length")? as usize;

        let rms_norm_eps = file
            .metadata
            .get("llama.attention.layer_norm_rms_epsilon")
            .and_then(super::metadata::GgufMetadataValue::as_f32)
            .unwrap_or(1e-5);

        let rope_base = file
            .metadata
            .get("llama.rope.freq_base")
            .and_then(super::metadata::GgufMetadataValue::as_f32)
            .unwrap_or(10000.0);

        let max_seq_len = file
            .metadata
            .get_u32("llama.context_length")
            .map(|v| v as usize)
            .unwrap_or(2048);

        // vocab_size: try metadata first, then derive from token_embd.weight.
        let vocab_size = file
            .metadata
            .get_u32("llama.vocab_size")
            .map(|v| v as usize)
            .or_else(|| {
                file.tensors
                    .get("token_embd.weight")
                    .map(|t| t.shape[0] as usize)
            })
            .ok_or_else(|| GgufError::MissingMetadata {
                key: "llama.vocab_size (and no token_embd.weight tensor)".to_string(),
            })?;

        Ok(Self {
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            ffn_dim,
            rms_norm_eps,
            rope_base,
            max_seq_len,
        })
    }
}

/// Build a [`ComputationGraph`] for a Llama model from config.
///
/// Uses `batch_size=1, seq_len=1` as the representative shape for
/// single-token inference. Weight data is not loaded -- all
/// [`WeightRef`]s are shape-only placeholders. Load actual weights
/// separately via [`GgufFile::read_tensor_f32()`].
///
/// The graph follows the standard Llama decoder-only transformer:
/// ```text
/// token_ids -> Embedding -> N x [RMSNorm -> QKV+SDPA -> Add ->
///   RMSNorm -> SiLU-gated FFN -> Add] -> RMSNorm -> Linear -> logits
/// ```
pub fn build_llama_graph(config: &LlamaConfig) -> ComputationGraph {
    let batch = 1;
    let seq = 1;
    let h = config.hidden_dim;
    let head_dim = config.head_dim;
    let nh = config.num_heads;
    let nkv = config.num_kv_heads;
    let ffn = config.ffn_dim;
    let vocab = config.vocab_size;
    let eps = f64::from(config.rms_norm_eps);

    let mut nodes: Vec<TraceNode> = Vec::new();
    let mut next_id: u64 = 0;

    let mut alloc_id = || {
        let id = next_id;
        next_id += 1;
        id
    };

    // Helper: shape-only weight ref.
    let weight_ref = |shape: Vec<usize>| WeightRef::from_shape(&shape);

    // ---- Input: token IDs [batch, seq] ----
    let input_id = alloc_id();
    nodes.push(TraceNode::new(
        input_id,
        "token_ids".to_string(),
        TraceOp::Input,
        vec![],
        vec![batch, seq],
        DType::U32,
    ));

    // ---- Token embedding [batch, seq, hidden] ----
    let embd_id = alloc_id();
    nodes.push(TraceNode::new(
        embd_id,
        "token_embd".to_string(),
        TraceOp::Embedding {
            weight: weight_ref(vec![vocab, h]),
        },
        vec![input_id],
        vec![batch, seq, h],
        DType::F32,
    ));

    let mut x_id = embd_id;

    // ---- Transformer blocks ----
    for layer in 0..config.num_layers {
        let residual_id = x_id;

        // 1. Pre-attention RMSNorm
        let attn_norm_id = alloc_id();
        nodes.push(TraceNode::new(
            attn_norm_id,
            format!("blk.{layer}.attn_norm"),
            TraceOp::RmsNorm {
                eps,
                weight: weight_ref(vec![h]),
            },
            vec![x_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // 2. Q projection [batch, seq, num_heads * head_dim]
        let q_id = alloc_id();
        nodes.push(TraceNode::new(
            q_id,
            format!("blk.{layer}.attn_q"),
            TraceOp::Linear {
                weight: weight_ref(vec![nh * head_dim, h]),
                bias: None,
            },
            vec![attn_norm_id],
            vec![batch, seq, nh * head_dim],
            DType::F32,
        ));

        // 3. K projection [batch, seq, num_kv_heads * head_dim]
        let k_id = alloc_id();
        nodes.push(TraceNode::new(
            k_id,
            format!("blk.{layer}.attn_k"),
            TraceOp::Linear {
                weight: weight_ref(vec![nkv * head_dim, h]),
                bias: None,
            },
            vec![attn_norm_id],
            vec![batch, seq, nkv * head_dim],
            DType::F32,
        ));

        // 4. V projection [batch, seq, num_kv_heads * head_dim]
        let v_id = alloc_id();
        nodes.push(TraceNode::new(
            v_id,
            format!("blk.{layer}.attn_v"),
            TraceOp::Linear {
                weight: weight_ref(vec![nkv * head_dim, h]),
                bias: None,
            },
            vec![attn_norm_id],
            vec![batch, seq, nkv * head_dim],
            DType::F32,
        ));

        // 5. Reshape Q -> [batch, seq, num_heads, head_dim]
        let q_reshape_id = alloc_id();
        nodes.push(TraceNode::new(
            q_reshape_id,
            format!("blk.{layer}.q_reshape"),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, nh, head_dim],
            },
            vec![q_id],
            vec![batch, seq, nh, head_dim],
            DType::F32,
        ));

        // Transpose Q -> [batch, num_heads, seq, head_dim]
        let q_t_id = alloc_id();
        nodes.push(TraceNode::new(
            q_t_id,
            format!("blk.{layer}.q_transpose"),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![q_reshape_id],
            vec![batch, nh, seq, head_dim],
            DType::F32,
        ));

        // 6. Reshape K -> [batch, seq, num_kv_heads, head_dim]
        let k_reshape_id = alloc_id();
        nodes.push(TraceNode::new(
            k_reshape_id,
            format!("blk.{layer}.k_reshape"),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, nkv, head_dim],
            },
            vec![k_id],
            vec![batch, seq, nkv, head_dim],
            DType::F32,
        ));

        // Transpose K -> [batch, num_kv_heads, seq, head_dim]
        let k_t_id = alloc_id();
        nodes.push(TraceNode::new(
            k_t_id,
            format!("blk.{layer}.k_transpose"),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![k_reshape_id],
            vec![batch, nkv, seq, head_dim],
            DType::F32,
        ));

        // 7. Reshape V -> [batch, seq, num_kv_heads, head_dim]
        let v_reshape_id = alloc_id();
        nodes.push(TraceNode::new(
            v_reshape_id,
            format!("blk.{layer}.v_reshape"),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, nkv, head_dim],
            },
            vec![v_id],
            vec![batch, seq, nkv, head_dim],
            DType::F32,
        ));

        // Transpose V -> [batch, num_kv_heads, seq, head_dim]
        let v_t_id = alloc_id();
        nodes.push(TraceNode::new(
            v_t_id,
            format!("blk.{layer}.v_transpose"),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![v_reshape_id],
            vec![batch, nkv, seq, head_dim],
            DType::F32,
        ));

        // 8. SDPA (causal) -> [batch, num_heads, seq, head_dim]
        let scale = 1.0 / (head_dim as f64).sqrt();
        let sdpa_id = alloc_id();
        nodes.push(TraceNode::new(
            sdpa_id,
            format!("blk.{layer}.sdpa"),
            TraceOp::SdpaCausal { scale },
            vec![q_t_id, k_t_id, v_t_id],
            vec![batch, nh, seq, head_dim],
            DType::F32,
        ));

        // 9. Transpose attn_out -> [batch, seq, num_heads, head_dim]
        let attn_t_id = alloc_id();
        nodes.push(TraceNode::new(
            attn_t_id,
            format!("blk.{layer}.attn_transpose"),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![sdpa_id],
            vec![batch, seq, nh, head_dim],
            DType::F32,
        ));

        // Reshape -> [batch, seq, hidden]
        let attn_reshape_id = alloc_id();
        nodes.push(TraceNode::new(
            attn_reshape_id,
            format!("blk.{layer}.attn_reshape"),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, h],
            },
            vec![attn_t_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // 10. Output projection [batch, seq, hidden]
        let o_id = alloc_id();
        nodes.push(TraceNode::new(
            o_id,
            format!("blk.{layer}.attn_output"),
            TraceOp::Linear {
                weight: weight_ref(vec![h, nh * head_dim]),
                bias: None,
            },
            vec![attn_reshape_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // 11. Residual add
        let attn_add_id = alloc_id();
        nodes.push(TraceNode::new(
            attn_add_id,
            format!("blk.{layer}.attn_residual"),
            TraceOp::Add,
            vec![residual_id, o_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // 12. Pre-FFN RMSNorm
        let ffn_norm_id = alloc_id();
        nodes.push(TraceNode::new(
            ffn_norm_id,
            format!("blk.{layer}.ffn_norm"),
            TraceOp::RmsNorm {
                eps,
                weight: weight_ref(vec![h]),
            },
            vec![attn_add_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // 13. Gate projection [batch, seq, ffn_dim]
        let gate_id = alloc_id();
        nodes.push(TraceNode::new(
            gate_id,
            format!("blk.{layer}.ffn_gate"),
            TraceOp::Linear {
                weight: weight_ref(vec![ffn, h]),
                bias: None,
            },
            vec![ffn_norm_id],
            vec![batch, seq, ffn],
            DType::F32,
        ));

        // 14. Up projection [batch, seq, ffn_dim]
        let up_id = alloc_id();
        nodes.push(TraceNode::new(
            up_id,
            format!("blk.{layer}.ffn_up"),
            TraceOp::Linear {
                weight: weight_ref(vec![ffn, h]),
                bias: None,
            },
            vec![ffn_norm_id],
            vec![batch, seq, ffn],
            DType::F32,
        ));

        // 15. SiLU(gate)
        let silu_id = alloc_id();
        nodes.push(TraceNode::new(
            silu_id,
            format!("blk.{layer}.ffn_silu"),
            TraceOp::Silu,
            vec![gate_id],
            vec![batch, seq, ffn],
            DType::F32,
        ));

        // 16. Mul(silu_gate, up) -> gated hidden
        let ffn_mul_id = alloc_id();
        nodes.push(TraceNode::new(
            ffn_mul_id,
            format!("blk.{layer}.ffn_mul"),
            TraceOp::Mul,
            vec![silu_id, up_id],
            vec![batch, seq, ffn],
            DType::F32,
        ));

        // 17. Down projection [batch, seq, hidden]
        let down_id = alloc_id();
        nodes.push(TraceNode::new(
            down_id,
            format!("blk.{layer}.ffn_down"),
            TraceOp::Linear {
                weight: weight_ref(vec![h, ffn]),
                bias: None,
            },
            vec![ffn_mul_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // 18. FFN residual add
        let ffn_add_id = alloc_id();
        nodes.push(TraceNode::new(
            ffn_add_id,
            format!("blk.{layer}.ffn_residual"),
            TraceOp::Add,
            vec![attn_add_id, down_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        x_id = ffn_add_id;
    }

    // ---- Final RMSNorm ----
    let output_norm_id = alloc_id();
    nodes.push(TraceNode::new(
        output_norm_id,
        "output_norm".to_string(),
        TraceOp::RmsNorm {
            eps,
            weight: weight_ref(vec![h]),
        },
        vec![x_id],
        vec![batch, seq, h],
        DType::F32,
    ));

    // ---- LM head (output projection) -> [batch, seq, vocab] ----
    let lm_head_id = alloc_id();
    nodes.push(TraceNode::new(
        lm_head_id,
        "output".to_string(),
        TraceOp::Linear {
            weight: weight_ref(vec![vocab, h]),
            bias: None,
        },
        vec![output_norm_id],
        vec![batch, seq, vocab],
        DType::F32,
    ));

    ComputationGraph::from_nodes(nodes)
}

/// Build a [`ComputationGraph`] with actual weight data from a GGUF file.
///
/// Same structure as [`build_llama_graph`] but `WeightRef`s contain the
/// dequantized f32 weight data read from the GGUF file. This graph is
/// ready for `CompiledModel::builder(graph, cache).build()`.
///
/// GGUF tensor name mapping:
/// - `token_embd.weight` → embedding weight
/// - `blk.{i}.attn_norm.weight` → pre-attention RMSNorm
/// - `blk.{i}.attn_q.weight` → Q projection
/// - `blk.{i}.attn_k.weight` → K projection
/// - `blk.{i}.attn_v.weight` → V projection
/// - `blk.{i}.attn_output.weight` → attention output projection
/// - `blk.{i}.ffn_norm.weight` → pre-FFN RMSNorm
/// - `blk.{i}.ffn_gate.weight` → gate projection (SwiGLU)
/// - `blk.{i}.ffn_up.weight` → up projection (SwiGLU)
/// - `blk.{i}.ffn_down.weight` → down projection
/// - `output_norm.weight` → final RMSNorm
/// - `output.weight` → LM head (falls back to `token_embd.weight` for tied embeddings)
pub fn build_llama_graph_with_weights<R: std::io::Read + std::io::Seek>(
    config: &LlamaConfig,
    gguf: &GgufFile,
    reader: &mut R,
) -> Result<ComputationGraph, GgufError> {
    let batch = 1;
    let seq = 1;
    let h = config.hidden_dim;
    let head_dim = config.head_dim;
    let nh = config.num_heads;
    let nkv = config.num_kv_heads;
    let ffn = config.ffn_dim;
    let vocab = config.vocab_size;
    let eps = f64::from(config.rms_norm_eps);

    let mut nodes: Vec<TraceNode> = Vec::new();
    let mut next_id: u64 = 0;

    let mut alloc_id = || {
        let id = next_id;
        next_id += 1;
        id
    };

    // Helper: load a weight tensor from GGUF, dequantize to f32, create WeightRef.
    let mut load_weight = |name: &str| -> Result<WeightRef, GgufError> {
        let (data, shape) = gguf.read_tensor_f32(reader, name)?;
        WeightRef::new(data, shape).map_err(|e| {
            GgufError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("weight shape mismatch for {name}: {e}"),
            ))
        })
    };

    // ---- Input ----
    let input_id = alloc_id();
    nodes.push(TraceNode::new(
        input_id,
        "token_ids".to_string(),
        TraceOp::Input,
        vec![],
        vec![batch, seq],
        DType::U32,
    ));

    // ---- Token embedding ----
    let embd_w = load_weight("token_embd.weight")?;
    let embd_id = alloc_id();
    nodes.push(TraceNode::new(
        embd_id,
        "token_embd".to_string(),
        TraceOp::Embedding { weight: embd_w },
        vec![input_id],
        vec![batch, seq, h],
        DType::F32,
    ));

    let mut x_id = embd_id;

    // ---- Transformer blocks ----
    for layer in 0..config.num_layers {
        let residual_id = x_id;

        let attn_norm_w = load_weight(&format!("blk.{layer}.attn_norm.weight"))?;
        let attn_norm_id = alloc_id();
        nodes.push(TraceNode::new(
            attn_norm_id,
            format!("blk.{layer}.attn_norm"),
            TraceOp::RmsNorm {
                eps,
                weight: attn_norm_w,
            },
            vec![x_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        let q_w = load_weight(&format!("blk.{layer}.attn_q.weight"))?;
        let q_id = alloc_id();
        nodes.push(TraceNode::new(
            q_id,
            format!("blk.{layer}.attn_q"),
            TraceOp::Linear {
                weight: q_w,
                bias: None,
            },
            vec![attn_norm_id],
            vec![batch, seq, nh * head_dim],
            DType::F32,
        ));

        let k_w = load_weight(&format!("blk.{layer}.attn_k.weight"))?;
        let k_id = alloc_id();
        nodes.push(TraceNode::new(
            k_id,
            format!("blk.{layer}.attn_k"),
            TraceOp::Linear {
                weight: k_w,
                bias: None,
            },
            vec![attn_norm_id],
            vec![batch, seq, nkv * head_dim],
            DType::F32,
        ));

        let v_w = load_weight(&format!("blk.{layer}.attn_v.weight"))?;
        let v_id = alloc_id();
        nodes.push(TraceNode::new(
            v_id,
            format!("blk.{layer}.attn_v"),
            TraceOp::Linear {
                weight: v_w,
                bias: None,
            },
            vec![attn_norm_id],
            vec![batch, seq, nkv * head_dim],
            DType::F32,
        ));

        // Reshape + Transpose Q → [batch, num_heads, seq, head_dim]
        let q_reshape_id = alloc_id();
        nodes.push(TraceNode::new(
            q_reshape_id,
            format!("blk.{layer}.q_reshape"),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, nh, head_dim],
            },
            vec![q_id],
            vec![batch, seq, nh, head_dim],
            DType::F32,
        ));
        let q_t_id = alloc_id();
        nodes.push(TraceNode::new(
            q_t_id,
            format!("blk.{layer}.q_transpose"),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![q_reshape_id],
            vec![batch, nh, seq, head_dim],
            DType::F32,
        ));

        // Reshape + Transpose K → [batch, num_kv_heads, seq, head_dim]
        let k_reshape_id = alloc_id();
        nodes.push(TraceNode::new(
            k_reshape_id,
            format!("blk.{layer}.k_reshape"),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, nkv, head_dim],
            },
            vec![k_id],
            vec![batch, seq, nkv, head_dim],
            DType::F32,
        ));
        let k_t_id = alloc_id();
        nodes.push(TraceNode::new(
            k_t_id,
            format!("blk.{layer}.k_transpose"),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![k_reshape_id],
            vec![batch, nkv, seq, head_dim],
            DType::F32,
        ));

        // Reshape + Transpose V → [batch, num_kv_heads, seq, head_dim]
        let v_reshape_id = alloc_id();
        nodes.push(TraceNode::new(
            v_reshape_id,
            format!("blk.{layer}.v_reshape"),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, nkv, head_dim],
            },
            vec![v_id],
            vec![batch, seq, nkv, head_dim],
            DType::F32,
        ));
        let v_t_id = alloc_id();
        nodes.push(TraceNode::new(
            v_t_id,
            format!("blk.{layer}.v_transpose"),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![v_reshape_id],
            vec![batch, nkv, seq, head_dim],
            DType::F32,
        ));

        // SDPA
        let scale = 1.0 / (head_dim as f64).sqrt();
        let sdpa_id = alloc_id();
        nodes.push(TraceNode::new(
            sdpa_id,
            format!("blk.{layer}.sdpa"),
            TraceOp::SdpaCausal { scale },
            vec![q_t_id, k_t_id, v_t_id],
            vec![batch, nh, seq, head_dim],
            DType::F32,
        ));

        // Transpose + Reshape back → [batch, seq, hidden]
        let attn_t_id = alloc_id();
        nodes.push(TraceNode::new(
            attn_t_id,
            format!("blk.{layer}.attn_transpose"),
            TraceOp::Transpose { dim0: 1, dim1: 2 },
            vec![sdpa_id],
            vec![batch, seq, nh, head_dim],
            DType::F32,
        ));
        let attn_reshape_id = alloc_id();
        nodes.push(TraceNode::new(
            attn_reshape_id,
            format!("blk.{layer}.attn_reshape"),
            TraceOp::Reshape {
                target_shape: vec![batch, seq, h],
            },
            vec![attn_t_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // Output projection
        let o_w = load_weight(&format!("blk.{layer}.attn_output.weight"))?;
        let o_id = alloc_id();
        nodes.push(TraceNode::new(
            o_id,
            format!("blk.{layer}.attn_output"),
            TraceOp::Linear {
                weight: o_w,
                bias: None,
            },
            vec![attn_reshape_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // Residual add
        let attn_add_id = alloc_id();
        nodes.push(TraceNode::new(
            attn_add_id,
            format!("blk.{layer}.attn_residual"),
            TraceOp::Add,
            vec![residual_id, o_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // Pre-FFN RMSNorm
        let ffn_norm_w = load_weight(&format!("blk.{layer}.ffn_norm.weight"))?;
        let ffn_norm_id = alloc_id();
        nodes.push(TraceNode::new(
            ffn_norm_id,
            format!("blk.{layer}.ffn_norm"),
            TraceOp::RmsNorm {
                eps,
                weight: ffn_norm_w,
            },
            vec![attn_add_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        // SwiGLU FFN
        let gate_w = load_weight(&format!("blk.{layer}.ffn_gate.weight"))?;
        let gate_id = alloc_id();
        nodes.push(TraceNode::new(
            gate_id,
            format!("blk.{layer}.ffn_gate"),
            TraceOp::Linear {
                weight: gate_w,
                bias: None,
            },
            vec![ffn_norm_id],
            vec![batch, seq, ffn],
            DType::F32,
        ));

        let up_w = load_weight(&format!("blk.{layer}.ffn_up.weight"))?;
        let up_id = alloc_id();
        nodes.push(TraceNode::new(
            up_id,
            format!("blk.{layer}.ffn_up"),
            TraceOp::Linear {
                weight: up_w,
                bias: None,
            },
            vec![ffn_norm_id],
            vec![batch, seq, ffn],
            DType::F32,
        ));

        let silu_id = alloc_id();
        nodes.push(TraceNode::new(
            silu_id,
            format!("blk.{layer}.ffn_silu"),
            TraceOp::Silu,
            vec![gate_id],
            vec![batch, seq, ffn],
            DType::F32,
        ));

        let ffn_mul_id = alloc_id();
        nodes.push(TraceNode::new(
            ffn_mul_id,
            format!("blk.{layer}.ffn_mul"),
            TraceOp::Mul,
            vec![silu_id, up_id],
            vec![batch, seq, ffn],
            DType::F32,
        ));

        let down_w = load_weight(&format!("blk.{layer}.ffn_down.weight"))?;
        let down_id = alloc_id();
        nodes.push(TraceNode::new(
            down_id,
            format!("blk.{layer}.ffn_down"),
            TraceOp::Linear {
                weight: down_w,
                bias: None,
            },
            vec![ffn_mul_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        let ffn_add_id = alloc_id();
        nodes.push(TraceNode::new(
            ffn_add_id,
            format!("blk.{layer}.ffn_residual"),
            TraceOp::Add,
            vec![attn_add_id, down_id],
            vec![batch, seq, h],
            DType::F32,
        ));

        x_id = ffn_add_id;
    }

    // ---- Final RMSNorm ----
    let final_norm_w = load_weight("output_norm.weight")?;
    let output_norm_id = alloc_id();
    nodes.push(TraceNode::new(
        output_norm_id,
        "output_norm".to_string(),
        TraceOp::RmsNorm {
            eps,
            weight: final_norm_w,
        },
        vec![x_id],
        vec![batch, seq, h],
        DType::F32,
    ));

    // ---- LM head ----
    // Some models tie output.weight = token_embd.weight.
    let lm_head_w = if gguf.tensors.contains_key("output.weight") {
        load_weight("output.weight")?
    } else {
        load_weight("token_embd.weight")?
    };
    let lm_head_id = alloc_id();
    nodes.push(TraceNode::new(
        lm_head_id,
        "output".to_string(),
        TraceOp::Linear {
            weight: lm_head_w,
            bias: None,
        },
        vec![output_norm_id],
        vec![batch, seq, vocab],
        DType::F32,
    ));

    Ok(ComputationGraph::from_nodes(nodes))
}

/// Read a required `u32` metadata key, returning a descriptive error if absent.
fn require_u32_meta(metadata: &crate::metadata::GgufMetadata, key: &str) -> Result<u32, GgufError> {
    metadata
        .get_u32(key)
        .ok_or_else(|| GgufError::MissingMetadata {
            key: key.to_string(),
        })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Small Llama config for testing: 2 layers, 4 heads, dim=64.
    fn small_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 256,
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16, // 64 / 4
            ffn_dim: 128,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 512,
        }
    }

    #[test]
    fn test_build_graph_node_count() {
        let config = small_config();
        let graph = build_llama_graph(&config);

        // Per layer: 20 nodes (attn_norm, q, k, v, q_reshape, q_transpose,
        //   k_reshape, k_transpose, v_reshape, v_transpose, sdpa,
        //   attn_transpose, attn_reshape, attn_output, attn_residual,
        //   ffn_norm, ffn_gate, ffn_up, ffn_silu, ffn_mul, ffn_down,
        //   ffn_residual) = 22 nodes per layer.
        // Global: input + embedding + output_norm + lm_head = 4.
        // Total: 4 + 22 * num_layers.
        let expected = 4 + 22 * config.num_layers;
        assert_eq!(
            graph.len(),
            expected,
            "expected {expected} nodes for {} layers, got {}",
            config.num_layers,
            graph.len()
        );
    }

    #[test]
    fn test_output_shape() {
        let config = small_config();
        let graph = build_llama_graph(&config);

        let output = graph.output_node().expect("graph should have output node");
        assert_eq!(
            output.output_shape(),
            &[1, 1, config.vocab_size],
            "output should be [batch=1, seq=1, vocab_size={}]",
            config.vocab_size
        );
    }

    #[test]
    fn test_topology_valid() {
        let config = small_config();
        let graph = build_llama_graph(&config);
        graph
            .validate_topology()
            .expect("graph should be in valid topological order");
    }

    #[test]
    fn test_single_layer() {
        let config = LlamaConfig {
            vocab_size: 32,
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 16,
            ffn_dim: 64,
            rms_norm_eps: 1e-6,
            rope_base: 10000.0,
            max_seq_len: 128,
        };
        let graph = build_llama_graph(&config);

        // 4 global + 22 * 1 = 26
        assert_eq!(graph.len(), 26);

        // Verify first node is input.
        let first = &graph.nodes()[0];
        assert_eq!(first.name(), "token_ids");
        assert!(matches!(first.op(), TraceOp::Input));

        // Verify second node is embedding.
        let second = &graph.nodes()[1];
        assert_eq!(second.name(), "token_embd");
        assert!(matches!(second.op(), TraceOp::Embedding { .. }));

        // Verify output shape.
        let output = graph.output_node().unwrap();
        assert_eq!(output.output_shape(), &[1, 1, 32]);
        assert_eq!(output.name(), "output");
    }

    #[test]
    fn test_gqa_shapes() {
        // GQA: fewer KV heads than Q heads.
        let config = LlamaConfig {
            vocab_size: 128,
            hidden_dim: 128,
            num_layers: 1,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 16, // 128 / 8
            ffn_dim: 256,
            rms_norm_eps: 1e-5,
            rope_base: 10000.0,
            max_seq_len: 256,
        };
        let graph = build_llama_graph(&config);

        // Find the K reshape node to verify GQA shapes.
        let k_reshape = graph
            .nodes()
            .iter()
            .find(|n| n.name() == "blk.0.k_reshape")
            .expect("should have k_reshape node");
        assert_eq!(k_reshape.output_shape(), &[1, 1, 2, 16]);

        // Q reshape should use full head count.
        let q_reshape = graph
            .nodes()
            .iter()
            .find(|n| n.name() == "blk.0.q_reshape")
            .expect("should have q_reshape node");
        assert_eq!(q_reshape.output_shape(), &[1, 1, 8, 16]);
    }

    #[test]
    fn test_input_nodes() {
        let config = small_config();
        let graph = build_llama_graph(&config);

        let inputs = graph.input_nodes();
        assert_eq!(inputs.len(), 1, "should have exactly one input node");
        assert_eq!(inputs[0].name(), "token_ids");
        assert_eq!(inputs[0].output_dtype(), DType::U32);
    }
}
