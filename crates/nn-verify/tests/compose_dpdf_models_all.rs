// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Consolidated dpdf model composition verification tests.
//!
//! Combines sub-block tests for document understanding models into a single
//! test binary to reduce link-time overhead from redundant NY linkage.
//!
//! **Granite-Docling** (compose_dpdf_granite_docling.rs — 15 tests):
//! - Vision patch embedding: Conv2d(3, D, P, stride=P) -> reshape -> transpose
//! - RMSNorm: root mean square normalization (Granite decoder layers)
//! - SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj
//! - Vision projection: Linear mapping vision features to LM embedding space
//!
//! **DocLayout-YOLO** (compose_dpdf_doclayout_yolo.rs — 20 tests):
//! - ConvBnAct: Conv2d -> BatchNorm -> SiLU (backbone building block)
//! - SPPF: MaxPool2d chain with channel concatenation (multi-scale pooling)
//! - Detection sigmoid: Sigmoid classification (output in [0, 1])
//! - DFL regression: Softmax -> weighted sum (Distribution Focal Loss decode)
//!
//! **GLM-OCR** (compose_dpdf_glm_ocr.rs — 46 tests):
//! - RMSNorm: IBP and CROWN bounds through root mean square normalization
//! - SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj with CROWN
//! - GQA attention: Grouped-query attention with softmax bounds
//! - Rotary embedding: cos/sin positional encoding bounded in [-1, 1]
//! - Decoder layer: Attention -> RMSNorm -> SwiGLU -> RMSNorm composition
//! - MTP head: Linear -> softmax output distribution in [0, 1]
//! - MTP multi-step: Chain of prediction heads (2-step and 3-step with CROWN)
//! - MTP normed chain: RMSNorm-gated prediction heads (IBP + CROWN)
//! - Embedding projection: Token embedding -> Linear bounds
//! - Causal mask attention: Causal mask preserves attention bounds
//! - Full decoder stack: 2-layer decoder -> LM head end-to-end (IBP + CROWN)
//! - Deep decoder stacks: 4-layer, 8-layer, 24-layer (IBP + CROWN)
//! - GQA at scale: 16 heads / 2 KV heads (8:1 ratio) (IBP + CROWN)
//! - Residual accumulation: Monotonic tightening through depth
//! - Embedding -> RoPE -> attention composition
//! - Decoder + MTP end-to-end: Decoder layer -> RMSNorm -> LM head (IBP + CROWN)
//! - First-layer full: Embedding -> RoPE -> GQA -> SwiGLU composition
//! - Multi-token parallel: Standard attention 2x sequence (IBP + CROWN)
//!
//! **Table Transformer** (compose_dpdf_table_transformer.rs — 32 tests):
//! - ResNet basic block: Conv2d -> BN -> ReLU -> Conv2d -> BN + skip (IBP + CROWN)
//! - ResNet backbone level: Conv2d(stride=2) spatial downsampling
//! - Transformer encoder layer: Self-attention -> LayerNorm -> FFN -> LayerNorm (CROWN)
//! - DETR decoder cross-attention: Object queries attend to encoder memory (CROWN)
//! - Classification head: Linear -> sigmoid output in [0, 1]
//! - Box regression head: Linear -> sigmoid for normalized coordinates
//! - Sinusoidal position encoding: sin/cos bounded in [-1, 1]
//! - DFL regression: Softmax -> weighted sum for box coordinates
//! - Full detection compose: End-to-end cls + box head pipeline
//! - DETR encoder 2-layer stack: Stacked self-attention + FFN (IBP + CROWN)
//! - DETR decoder 2-layer stack: Self-attn + cross-attn + FFN x2 (IBP + CROWN)
//! - ResNet 2-stage backbone: Cascaded stride-2 downsampling (IBP)
//! - Cross-attention with LayerNorm: Normalized queries + residual (IBP)
//! - Position encoding + attention: PE + LN + MHA + residual (IBP + CROWN)
//! - Full DETR pipeline: Encoder -> decoder proj -> dual sigmoid heads (IBP)
//! - DFL -> sigmoid end-to-end: Softmax -> weighted sum -> sigmoid (IBP)
//! - Transformer FFN residual: LN -> Linear -> ReLU -> Linear + skip (CROWN)
//! - Table detect + structure: Triple sigmoid heads for dual task (IBP + CROWN)
//! - Full pipeline: ResNet -> encoder -> decoder -> heads (IBP)
//!
//! **Qwen3-VL** (compose_dpdf_qwen3_vl.rs — 17 tests):
//! - Conv3D patch embedding: Conv2d(3, D, P, stride=P) spatial core (IBP)
//! - Window attention: Local self-attention over fixed-size patch windows (IBP)
//! - Interleaved M-RoPE: Multi-modal rotary embedding bounded in [-1, 1] (IBP)
//! - Vision encoder block: RMSNorm -> Attention -> FFN with residuals (CROWN)
//! - Deep stack fusion: 2 stacked vision encoder blocks (IBP)
//! - SwiGLU decoder FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj (CROWN)
//! - GQA KV-cache: Grouped-query attention with causal mask (IBP)
//! - MoE routing: Linear -> softmax expert gate in [0, 1] (IBP)
//! - RMSNorm: Root mean square normalization (IBP)
//! - Vision-language projection: Linear mapping vision features to LM space (IBP)
//! - Full VLM compose: Patch embed -> encoder -> projection -> decoder FFN (IBP)
//!
//! **Qwen3-VL Vision Encoder** (compose_dpdf_qwen3_vl_vision_encoder.rs — 25 tests):
//! - Patch embedding Conv2d: Conv2d(3, D, P, stride=P) spatial projection (IBP)
//! - Patch embed + RMSNorm: Conv2d -> reshape -> RMSNorm (IBP + CROWN)
//! - RoPE cos/sin bounded: cos/sin positional encoding in [-1, 1] (IBP)
//! - GQA attention: Q full-rank, K/V reduced-rank grouped attention (IBP + CROWN)
//! - SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj (IBP + CROWN)
//! - Vision encoder block: RMSNorm -> attn -> res -> RMSNorm -> SwiGLU -> res (IBP + CROWN)
//! - Multi-block stacks: 2-block and 4-block depth composition (IBP)
//! - Vision projection: Linear(encoder_dim -> lm_dim) mapping (IBP)
//! - Full pipeline: image -> patches -> encoder blocks -> projection (IBP + CROWN)
//! - Depth scaling: 1/2/4 block bound width analysis (IBP)
//! - Image normalization: (pixel - mean) / std per-channel bounds (IBP)
//! - Multi-resolution tiling: 2-tile additive merge through patch embed (IBP)
//! - Dynamic resolution padding: zero-padded image -> patch embed -> RMSNorm (IBP)
//! - Global average pooling: mean-reduce over sequence dim (IBP)
//! - Merged image-text sequence: additive vision + text fusion -> projection (IBP)
//! - Multi-scale features: shallow + deep encoder additive fusion (IBP)
//! - Patch position interpolation: linear interpolation of position embeds (IBP)
//!
//! **Qwen3-VL Vision Encoder Extended** (compose_dpdf_qwen3_vl_vision_encoder_ext.rs — 15 tests):
//! - RoPE Q/K projection: CROWN linearization through rotary embeddings (CROWN)
//! - Vision projection: CROWN through Linear encoder->LM mapping (CROWN)
//! - 2-block encoder stack: CROWN through deep stack composition (CROWN)
//! - RMSNorm isolation: standalone CROWN bounds (CROWN)
//! - Window attention: local window self-attention (IBP + CROWN)
//! - Temporal patch embedding: 3D patch temporal dim proxy (IBP)
//! - Image norm + patch embed: normalization + conv CROWN composition (CROWN)
//! - Verify-and-record: patch embed, encoder block, GQA, SwiGLU, full pipeline
//! - GAP + projection: mean reduce + linear CROWN composition (CROWN)
//! - Position embed + encoder block: position injection + block CROWN (CROWN)
//!
//! **Qwen3-VL Vision Encoder Pipeline** (compose_dpdf_qwen3_vl_encoder.rs — 12 tests):
//! - Patch embedding: Conv2d -> flatten -> Linear projection (IBP + CROWN)
//! - ViT block: LayerNorm -> Q/K/V -> attention -> residual -> FFN (IBP + CROWN)
//! - Multi-scale patch merge: Linear projection across 2 scales (IBP + CROWN)
//! - Visual token projection: LayerNorm -> Linear -> GELU -> Linear (IBP + CROWN)
//! - Full 2-layer vision encoder: patch_embed -> 2x ViT blocks -> projection (IBP + CROWN)
//! - Attention + FFN composition: bounds through attention then FFN (IBP + CROWN)
//!
//! **PaddleOCR** (compose_dpdf_paddle_ocr.rs — 13 tests):
//! - DB Conv backbone: Conv2d -> BatchNorm -> ReLU (text detector backbone)
//! - DB sigmoid output: Conv2d -> Sigmoid probability map in [0, 1]
//! - SVTR patch embedding: Conv2d -> reshape -> transpose (recognition encoder)
//! - SVTR attention block: LayerNorm -> Attention -> residual (CROWN)
//! - SVTR MLP GELU: LayerNorm -> Linear -> GELU -> Linear -> residual (CROWN)
//! - CTC linear head: Linear projection to vocabulary
//! - CTC softmax output: Linear -> Softmax character probabilities in [0, 1]
//! - Detection pipeline: Conv-BN-ReLU -> sigmoid end-to-end
//! - Recognition pipeline: Patch embed -> MLP -> CTC head
//! - Full OCR pipeline: Detection backbone -> recognition -> softmax
//!
//! **PaddleOCR Deep** (compose_paddle_ocr_deep.rs — 20 tests):
//! - Self-attention isolation: Q/K/V + softmax + out_proj (IBP + CROWN)
//! - Full SVTR encoder block: LN -> Attention -> residual -> LN -> MLP -> residual (IBP + CROWN)
//! - Patch embed + one block: Conv2d patch embedding + encoder block (IBP + CROWN)
//! - Two-block encoder + final LayerNorm: depth composition (IBP)
//! - Widening analysis: 1-block vs 2-block IBP width comparison
//! - Encoder + CTC softmax: 2 blocks + LN + CTC head + softmax (IBP)
//! - DB ResNet skip connection: Conv-BN-ReLU -> Conv-BN + skip -> ReLU (IBP + CROWN)
//! - DB 2-stage backbone + sigmoid: Multi-scale backbone -> sigmoid head (IBP)
//! - Tight-input attention: Narrow +-0.1 bounds for CROWN precision (IBP + CROWN)
//! - Full recognition pipeline: Patch embed -> 2 blocks -> LN -> CTC softmax (IBP)
//! - DB backbone widening: 1-stage vs 3-stage bounds growth (IBP)
//! - Attention with sinusoidal PE: Position encoding + CROWN linearization (CROWN)
//! - Verify-and-record: Full encoder block + ResNet skip block (IBP + CROWN)
//!
//! **FireRed-OCR** (compose_dpdf_firered_ocr.rs — 12 tests):
//! - Patch embedding: Conv2d(3, D, 14, stride=14) for 2B-scale patches (IBP)
//! - Small attention: 12-head attention at 1536 dims with residual (IBP)
//! - Encoder layer: RMSNorm -> Attention -> SwiGLU FFN with residuals (IBP + CROWN)
//! - OCR vocab projection: Linear(HIDDEN_DIM, VOCAB_SIZE) for CTC logits (IBP)
//! - CTC blank probability: Softmax output bounded in [0, 1] (IBP)
//! - CTC softmax output: Linear -> Softmax character probabilities (IBP)
//! - OCR pipeline: Patch embed -> encoder -> CTC head -> softmax (IBP)
//! - RMSNorm: Root mean square normalization at 2B-scale dims (IBP)
//! - SwiGLU FFN: gate_proj -> SiLU -> mul(up_proj) -> down_proj (CROWN)
//! - Two-layer encoder: 2 stacked encoder blocks (IBP)
//! - Line detection sigmoid: Sigmoid confidence output in [0, 1] (IBP)
//!
//! Architecture references:
//! - Granite-Docling: SigLIP2 vision encoder + Granite LLM decoder
//! - DocLayout-YOLO (Zhao et al. 2024): YOLOv10-based document layout detection
//! - DFL (Li et al. 2022): Distribution Focal Loss for box regression
//! - GLM-4V (THUDM): Vision-language model with GLM-4 decoder for OCR
//! - Table Transformer (Smock et al. 2022): DETR-based table structure recognition
//! - DETR (Carion et al. 2020): DEtection TRansformer
//! - Qwen2-VL / Qwen3-VL (Alibaba): Vision-language model with 3D patch embedding,
//!   M-RoPE, window attention, SwiGLU, GQA, and optional MoE routing
//! - PaddleOCR (Baidu): Production OCR with DB detector + SVTR recognizer
//! - DB (Liao et al. 2020): Differentiable Binarization for text detection
//! - SVTR (Du et al. 2022): Scene Text Recognition with a Single Visual Model
//! - FireRed-OCR: Qwen3-VL-2B variant fine-tuned for document OCR with CTC decoding
//!
//! **FireRed-OCR Deep** (compose_firered_ocr_deep.rs -- 13 tests):
//! - Full encoder block: RMSNorm -> 4-head Attention -> residual -> SwiGLU -> residual (IBP + CROWN)
//! - 2-layer encoder stack: Depth composition widening (IBP)
//! - Patch embed + encoder: Conv2d(2x2) + encoder block (IBP + CROWN)
//! - CTC pipeline: 2 blocks -> Linear(HIDDEN, VOCAB) -> Softmax (IBP)
//! - Full OCR pipeline: Patch embed -> 2 blocks -> CTC head -> softmax (IBP + CROWN)
//! - Tight-input CROWN: Narrow +-0.1 bounds for CROWN precision (IBP + CROWN)
//! - Widening analysis: 1-block vs 2-block IBP width comparison (IBP)
//! - RMSNorm -> SwiGLU -> RMSNorm sandwich (IBP)
//! - Patch embedding + positional encoding (IBP)
//!
//! **Granite-Docling SigLIP2 Deep** (compose_granite_docling_siglip2_deep.rs — 21 tests):
//! - Self-attention isolation: Q/K/V + softmax + out_proj (IBP + CROWN)
//! - Attention + residual: LayerNorm -> MHA -> skip (IBP + CROWN)
//! - LayerNorm + SiGLU FFN + residual (IBP + CROWN)
//! - Frontend + one block: patch proj + pos embed + encoder block (IBP + CROWN)
//! - Two-block encoder stack + widening analysis (IBP + CROWN)
//! - Encoder + post-LayerNorm (IBP)
//! - Encoder + post-LN + mean pooling (IBP + CROWN)
//! - Full encoder + vision projection (Granite-Docling bridge) (IBP + CROWN + verify)
//! - Three-block encoder stack (IBP + CROWN)
//! - Tight-input SiGLU FFN: narrow bounds for CROWN precision (IBP + CROWN)
//!
//! **Table Transformer Deep** (compose_table_transformer.rs — 14 tests):
//! - Decoder self-attention: Object queries self-attend (IBP + CROWN)
//! - Full DETR decoder layer: Self-attn + cross-attn + FFN with residuals (CROWN)
//! - Two-layer encoder stack: Depth composition bounds widening (IBP + CROWN)
//! - Input projection + PE: Conv2d(1x1) + sinusoidal positional encoding (IBP)
//! - Encoder-to-decoder pipeline: Encoder norm -> cross-attention (CROWN)
//! - Full pipeline compose: Encoder -> decoder -> cls + box heads (IBP)
//!
//! **Quantized INT4 Dequantization** (compose_dpdf_quantized.rs — 14 tests):
//! - Single group dequant: scale * (code - zero_point) bounds for one group (IBP)
//! - Multi-group dequant: bounds across group boundaries (IBP)
//! - Asymmetric vs symmetric: compare dequant bound widths (IBP)
//! - Group size impact: larger groups produce wider per-element bounds (IBP)
//! - INT4 dequant with bias: dequant -> add bias shift verification (IBP)
//! - Quantized linear layer: dequant -> matmul -> bias bounds (IBP)
//! - Quantized attention: Q/K/V through INT4 dequant layers (IBP)
//! - Quantized SwiGLU: gate/up projections through INT4 dequant (IBP)
//! - Quantized MoE expert: dequant -> expert FFN + residual (IBP)
//! - Full quantized decoder layer: all projections INT4 (IBP)
//! - Quantized vs FP32 comparison: bound width ratio analysis (IBP)
//! - Mixed precision: FP32 attention + INT4 FFN decoder (IBP)
//! - Quantized generation: decoder -> LM head -> softmax (IBP)
//! - 2-layer quantized decoder + LM head (CROWN)
//!
//! **Certification Properties** (compose_dpdf_certification.rs — 35+ tests):
//! - P1 Bounded outputs: Class sigmoid, box sigmoid, CTC logits all bounded
//! - P1 ext: Per-model heads for all 6 archetypes (Granite-Docling softmax,
//!   DocLayout-YOLO dual sigmoid, GLM-OCR MTP, PaddleOCR CTC, Table Transformer
//!   DETR heads, Qwen3-VL LM head)
//! - P2 Monotone confidence: Tighter input bounds -> tighter output bounds
//! - P2 multi-model: Deep MLP (Linear->ReLU->Linear->Sigmoid) monotonicity + CROWN
//! - P3 Quantization safety: INT4 dequant delta bounded
//! - P4 Pipeline composition: Multi-head sigmoid pipeline preserves bounds (IBP + CROWN)
//! - P4 ext: Detection (DocLayout-YOLO) -> OCR (FireRed-OCR) 3-stage pipeline (IBP + CROWN)
//! - P5 NMS stability: Small perturbation -> bounded box coordinate change
//! - P6 Softmax normalization: Softmax sum brackets 1.0 (IBP + CROWN)
//! - P6 multi-head: GQA attention softmax normalization (causal + standard masks,
//!   GQA -> softmax classification head composition) (IBP + CROWN)
//! - P7 Sigmoid boundedness: Deep MLP + sigmoid strictly in (0, 1) (IBP + CROWN)
//! - P8 Resolution invariance: Patch embedding bounds across spatial sizes
//! - P8 ext: 3D patch embedding (Qwen3-VL temporal conv) resolution invariance,
//!   2D vs 3D bounds comparison (IBP)
//!
//! **GLM-OCR Deep** (compose_glm_ocr_deep.rs — 14 tests):
//! - Full decoder layer: RMSNorm -> Attention -> residual -> SwiGLU FFN -> residual (IBP + CROWN)
//! - 2-layer decoder stack: Depth composition with widening analysis (IBP)
//! - MTP head chain: Linear -> softmax multi-step prediction heads (IBP + CROWN)
//! - Embedding + decoder: Token embedding -> Linear -> decoder layer (IBP)
//! - Full pipeline: Embedding -> 2-layer decoder -> LM head -> softmax (IBP)
//! - Tight-input analysis: Narrow +-0.1 bounds for CROWN precision (IBP + CROWN)
//! - Verify-and-record: Full decoder layer + full pipeline + tight decoder (IBP + CROWN)
//!
//! **DocLayout-YOLO Deep** (compose_doclayout_yolo_deep.rs — 14 tests):
//! - C2f block: Entry conv -> bottleneck + skip -> concat -> exit conv (IBP + CROWN)
//! - SPPF + detection head: MaxPool chain -> concat -> Conv -> sigmoid [0,1] (IBP)
//! - Backbone stage: ConvBnSiLU with stride-2 downsampling (IBP + CROWN)
//! - Neck FPN: Multi-scale feature concat + Conv channel reduction (IBP)
//! - Detection pipeline: Backbone -> SPPF -> dual heads (cls sigmoid + box DFL) (IBP)
//! - Widening analysis: 1-stage vs 2-stage backbone bounds comparison (IBP)
//! - 2-stage backbone: Two ConvBnSiLU stride-2 stages (IBP + CROWN)
//!
//! **Cross-Pipeline** (compose_dpdf_cross_pipeline.rs — 6 tests):
//! - Detection -> OCR: YOLO sigmoid -> bridge -> FireRed-OCR CTC softmax (IBP)
//! - Detection -> Table: YOLO bbox sigmoid -> Table Transformer queries + attention (IBP)
//! - OCR -> Language: CTC softmax -> token embedding -> decoder FFN + sigmoid (IBP)
//! - Multi-head pipeline: cls sigmoid + box sigmoid + CTC softmax simultaneously (IBP + CROWN)
//! - Tier routing: softmax gate dispatches to table/OCR heads, convex merge (IBP)
//!
//! **Adversarial Robustness** (compose_dpdf_adversarial.rs -- 16 tests):
//! - Per-model robustness: epsilon-ball perturbation -> bounded output change (7 tests)
//! - Monotone tightening: smaller eps -> tighter bounds for 5 activation types
//! - Cross-model robustness: pipeline, cascading, quantization, VLM-to-layout (4 tests)
//!
//! **Normalization Layer Variants** (compose_dpdf_normalization.rs -- 20 tests):
//! - LayerNorm: single layer IBP, affine IBP, CROWN linearization
//! - RMSNorm: single layer IBP, scaled weight IBP, CROWN
//! - BatchNorm: inference mode IBP, affine IBP, CROWN
//! - GroupNorm: G=4 IBP, G=1 (LayerNorm equivalent) IBP, G=1 CROWN
//! - LayerNorm -> Linear composition (IBP + CROWN)
//! - RMSNorm -> SwiGLU composition (IBP + CROWN)
//! - BatchNorm -> ReLU -> Conv2d composition (IBP)
//! - Normalization epsilon monotone tightening
//! - Verify and record for LayerNorm, RMSNorm, BatchNorm, GroupNorm
//!
//! **Spatial Operations** (compose_dpdf_spatial.rs — 17 tests):
//! - MaxPool2d: single layer, strided, CROWN bounds
//! - AvgPool2d: single layer, strided, CROWN bounds
//! - AdaptiveAvgPool2d: simulated via computed kernel size (IBP)
//! - Conv2d stride=2: spatial downsampling (IBP + CROWN)
//! - ConvTranspose2d: spatial upsampling (IBP)
//! - SPPF: multi-scale cascaded MaxPool2d + concat (IBP)
//! - MaxPool chain: 3 cascaded MaxPool2d 16x16 -> 2x2 (IBP)
//! - Conv -> Pool -> Conv: spatial processing pipeline (IBP + CROWN)
//! - Downsample -> upsample: round-trip spatial (IBP)
//! - Feature pyramid: stride 1, 2, 4 multi-resolution (IBP)
//! - Monotone tightening: smaller eps -> tighter spatial bounds
//! - AvgPool2d overlapping: stride != kernel (IBP)
//!
//! **Activation Function Variants** (compose_dpdf_activations.rs -- 16 tests):
//! - GELU (tanh approx + exact erf): IBP + CROWN bounds
//! - SiLU (Swish): IBP + CROWN + gate pattern (SwiGLU building block)
//! - Snake: IBP with scalar kernel, varying alpha comparison
//! - Mish: IBP + CROWN (x * tanh(softplus(x)))
//! - ReLU baseline, Sigmoid bounded in (0, 1)
//! - Composed Linear-Activation-Linear pipelines (IBP + CROWN)
//! - Monotone tightening: smaller eps -> tighter output for 5 activations
//! - Activation chain: GELU -> Linear -> SiLU -> Linear (IBP)
//!
//! **Embedding & Projection Layers** (compose_dpdf_embeddings.rs -- 14 tests):
//! - Token embedding: lookup-based IBP bounds
//! - Patch embedding: Conv2d(3, D, P, stride=P) -> reshape -> transpose (IBP)
//! - Learned positional embedding: embedding lookup IBP
//! - Token embed + positional addition: shifted bounds (IBP)
//! - Sinusoidal positional encoding: sin/cos PE bounded in [-1, 1] (IBP)
//! - RoPE: rotary positional embedding bounded IBP
//! - M-RoPE: multimodal rotary (temporal, height, width) IBP
//! - 2D sinusoidal PE: spatial grid position encoding (IBP)
//! - Vision-to-language projection: Linear mapping (IBP + CROWN)
//! - Cross-modal projection: Linear -> GELU (CROWN)
//! - Embedding + LayerNorm composition (IBP + CROWN)
//! - Patch embed + PE + attention composition (IBP)
//! - Embedding dimension scaling: smaller weights -> tighter bounds (IBP)
//! - Embedding monotone tightening: smaller eps -> tighter output (IBP)
//!
//! **Attention Mechanisms** (compose_dpdf_attention.rs -- 35 tests):
//! - Multi-head self-attention (MHA): IBP + CROWN bounds
//! - Grouped-query attention (GQA): reduced KV heads with residual (IBP + CROWN)
//! - Window (local) attention: fixed-size window MHA (IBP)
//! - Cross-attention: encoder-decoder attention (IBP + CROWN)
//! - Causal mask attention: causal MHA (IBP)
//! - Attention + RoPE: sinusoidal PE + MHA composition (IBP)
//! - Attention + LayerNorm + residual: pre-norm attention (CROWN)
//! - Softmax attention weights: bounded in [0, 1] (IBP)
//! - Attention scaling: 1/sqrt(d_k) vs unscaled (IBP)
//! - KV-cache attention: current query over cached context (IBP)
//! - Transformer block: LN -> MHA -> residual -> LN -> FFN -> residual (IBP + CROWN)
//! - Attention monotone tightening: smaller eps -> tighter bounds
//! - Causal mask triangular pattern: softmax -> sigmoid in [0, 1] (IBP + CROWN)
//! - Sliding window attention: partitioned SDPA with restricted span (IBP + CROWN)
//! - DETR cross-attention: object queries -> encoder memory -> sigmoid (IBP)
//! - Granite-Docling cross-attention: vision features -> LM decoder (IBP)
//! - GQA repeat_kv: KV head expansion preserves bounds (IBP + CROWN)
//! - GQA repeat_kv expansion monotone tightening (IBP)
//! - ViT window attention: partition -> local attention -> unpartition (IBP + CROWN)
//! - Window partition preserves total bound range (IBP)
//! - Deformable attention: learned sigmoid offsets -> bounded features (IBP + CROWN)
//! - Deformable offset magnitude bounded in [0, 1] (IBP)
//! - SageAttention INT8: tanh-quantized QK -> bounded attention (IBP + CROWN)
//! - SageAttention quantization error bounded in [-1, 1] (IBP)
//! - Multi-pattern attention: causal + cross composition (IBP)
//!
//! **MoE Routing and Expert Selection** (compose_dpdf_moe.rs -- 15 tests):
//! - MoE gate: Linear -> softmax output in [0, 1] (IBP)
//! - MoE gate sum-to-one: softmax probability feasibility (IBP)
//! - Top-1 expert selection: narrow(1) after softmax (IBP)
//! - Top-2 expert selection with load balancing (IBP)
//! - Expert FFN (SwiGLU): single expert with CROWN linearization
//! - MoE dispatch: gate -> select -> expert FFN composition (IBP)
//! - MoE residual: input + MoE(input) skip connection (IBP)
//! - MoE with shared expert: shared + routed expert combination (IBP)
//! - MoE routing monotone tightening: tighter input -> tighter gate (IBP)
//! - Expert capacity: per-expert probability bounds across input ranges (IBP)
//! - MoE vs dense FFN: bound width comparison (IBP)
//! - MoE auxiliary loss: router + load balance softmax bounded (IBP)
//! - Multi-layer MoE: 2-layer decoder with attention + MoE FFN (IBP)
//! - MoE + attention decoder block with CROWN linearization
//! - MoE quantized experts: INT4 tighter bounds vs FP32 (IBP)
//!
//! **Residual & Skip Connections** (compose_dpdf_residual.rs -- 15 tests):
//! - Pre-norm residual: x + Linear(LayerNorm(x)) (IBP)
//! - Post-norm residual: LayerNorm(x + Linear(x)) (IBP)
//! - RMSNorm residual: x + Linear(RMSNorm(x)) (IBP + CROWN)
//! - Dense residual: DenseNet-style concatenation (IBP)
//! - ResNet basic block: Conv-BN-ReLU-Conv-BN + skip -> ReLU (IBP + CROWN)
//! - ResNet bottleneck: 1x1-3x3-1x1 + skip -> ReLU (IBP)
//! - FPN lateral connection: 1x1 conv + add (IBP)
//! - Deep residual stack: 3-layer accumulation (IBP)
//! - Residual monotone tightening: smaller eps -> tighter output (IBP)
//! - Skip preserves bound width ordering (IBP)
//! - Stochastic depth: x + alpha * Linear(x) (IBP)
//! - Cross-attention residual: q + Attn(q, kv) (IBP + CROWN)
//! - Multi-scale residual fusion: 1x1 proj + add (IBP)
//! - Residual gradient stability: 4-layer deep stack (CROWN)
//! - Pre-norm vs post-norm bound width comparison (IBP)
//!
//! **Residual Stream Bounds** (compose_dpdf_residual_stream.rs -- 15 tests):
//! - Pre-norm residual stream 2-block (IBP + CROWN)
//! - Pre-norm residual stream 4-block depth (IBP)
//! - Residual accumulation monotonicity (IBP)
//! - Dense residual DenseNet concatenation 2-block (IBP + CROWN)
//! - Residual scaling alpha sweep (IBP + CROWN)
//! - Cross-attention residual stream q + attn(q, kv) (IBP + CROWN)
//! - Encoder-decoder full residual stream (IBP)
//! - RMSNorm residual stream 3-block (IBP + CROWN)
//! - Residual stream monotone tightening (IBP)
//! - Mixed norm residual stream: RMSNorm + LayerNorm (IBP)
//!
//! **Multi-Modal Fusion** (compose_dpdf_multimodal.rs -- 15 tests):
//! - Vision feature projection (Linear) IBP + CROWN
//! - Projection + LayerNorm composition IBP + CROWN
//! - Cross-modal attention (image queries, text memory) IBP + CROWN
//! - Cross-modal residual connection IBP
//! - Vision-language concatenation IBP
//! - Gated fusion (sigmoid gate * vision + complement * text) IBP
//! - Multi-scale vision feature fusion before projection IBP
//! - Interleaved vision-text token sequence IBP
//! - Vision encoder -> projection -> decoder attention pipeline IBP
//! - Vision-language alignment bounds IBP
//! - Multi-modal monotone tightening IBP
//! - Full VLM path: vision encode -> project -> decode IBP + CROWN
//!
//! Part of #3870: NY compose tests for dpdf models.
//! Part of #3883: deep NY compose tests for Table Transformer.
//! Part of #3902: deep NY compose tests for Granite-Docling SigLIP2.
//! Part of #3928: deep NY compose tests for PaddleOCR.
//! Part of #3938: dpdf deployment certification property proofs (P1-P8).
//! Part of #3961: Quantized Qwen3-VL INT4 compose tests.
//! Part of #3962: Adversarial robustness compose tests for dpdf models.
//! Part of #3968: Normalization layer variant compose tests.
//! Part of #3969: Activation function compose tests for dpdf models.
//! Part of #3973: Spatial operations compose tests for pooling, upsampling, convolution.
//! Part of #3974: Attention mechanism compose tests for MHA, GQA, window, cross-attention.
//! Part of #3975: Embedding and projection compose tests for token, patch, positional layers.
//! Part of #3980: Loss function and output head compose tests for focal, DFL, CTC, cross-entropy.
//! Part of #3981: Residual and skip connection compose tests across model architectures.
//! Part of #3985: MoE routing and expert selection compose tests for Qwen3-VL.
//! Part of #3986: Decoder stack compose tests for 2-layer, 4-layer, full-depth decoders.
//! Part of #3987: End-to-end model pipeline compose tests for detection, OCR, table, VLM.
//! Part of #3991: Multi-modal fusion compose tests for vision-language projection.
//! Part of #3992: Backbone architecture compose tests for ResNet, YOLO, ViT.
//! Part of #3996: Compose tests for table structure recognition.
//! Part of #3997: Text detection probability map compose tests for DB detector.
//! Part of #3998: Compose tests for CTC decoding paths.
//! Part of #4002: FPN lateral connections, top-down fusion, multi-scale output.
//! Part of #4003: Positional encoding variant compose tests.
//! Part of #4004: SwiGLU FFN variant compose tests for gate patterns and depth composition.
//! Part of #4008: DETR decoder patterns: object queries, cross-attention, bipartite matching.
//! Part of #4009: Vision encoder depth bound tracking compose tests.
//! Part of #4010: GQA KV-cache attention compose tests for document VLMs.
//! Part of #4014: NMS and detection postprocessing compose tests.
//! Part of #4015: Document reading order and layout spatial reasoning compose tests.
//! Part of #4016: MoE expert routing and sparse gating compose tests.
//! Part of #4020: Multi-scale feature aggregation (PAN/FPN fusion) compose tests.
//! Part of #4021: Compose tests for vision-language projection and token merging.
//! Part of #4022: Compose tests for RoPE and M-RoPE position encoding variants.
//! Part of #4026: Compose tests for quantized weight inference (INT4 GPTQ/AWQ).
//! Part of #4027: Compose tests for text recognition sequence decoder (CTC vs attention).
//! Part of #4028: Compose tests for depthwise separable convolution bounds (MobileNet patterns).
//! Part of #4032: Compose tests for normalization variant bounds.
//! Part of #4034: Compose tests for patch embedding and image tokenization bounds.
//! Part of #4033: Compose tests for cross-attention in encoder-decoder models.
//! Part of #4035: Compose tests for vocabulary projection and sampling head bounds.
//! Part of #4036: Compose tests for sliding window and local attention patterns.
//! Part of #4040: Compose tests for language model head bounds.
//! Part of #4042: Compose tests for multi-head prediction (MTP) patterns.
//! Part of #4043: Compose tests for attention mask patterns.
//! Part of #4047: Compose tests for pooling and spatial reduction patterns.
//! Part of #4048: Compose tests for weight initialization bounds.
//! Part of #4049: Compose tests for gradient flow and residual stream bounds.
//! Part of #4055: Compose tests for sequence length and position extrapolation bounds.
//! Part of #4056: Compose tests for dynamic shape and multi-resolution bounds.
//! Part of #4060: Compose tests for dropout and stochastic depth effects on bounds.
//! Part of #4066: Compose tests for mixed-precision inference bounds.
//! Part of #4061: Compose tests for activation function variant bounds.
//! Part of #4101: Compose tests for model weight initialization bounds (Xavier/Kaiming).
//! Part of #4112: Compose tests for residual stream bounds propagation.
//! Part of #4129: Compose tests for vision encoder feature extraction.
//! Part of #4273: Deep compose tests for TT structure, Granite decoder, audio/image/orchestration.
//!
//! **GQA KV-Cache Attention** (compose_dpdf_gqa_kvcache.rs -- 15 tests):
//! - GQA head grouping (8Q/2KV -> 4:1 ratio) IBP bounds
//! - KV-cache append pattern (decode phase) IBP bounds
//! - Sliding window with cached positions IBP bounds
//! - Output projection (KV_DIM -> DIM) IBP bounds
//! - Cross-attention between vision encoder and text decoder IBP bounds
//! - Prefill vs decode phase bound width comparison IBP
//! - GQA with RoPE position encoding IBP bounds
//! - KV-cache memory layout (interleaved heads) IBP bounds
//! - Causal mask interaction with cache offset IBP bounds
//! - GQA group ratio comparison (4:1 vs 8:1) IBP
//! - Cross-attention with encoder features (VLM pattern) IBP bounds
//! - KV-cache eviction/rotation bounds IBP
//! - GQA numerical stability (softmax temperature scaling) IBP
//! - Multi-layer GQA depth composition (2-layer) IBP + CROWN
//! - Full attention block: RMSNorm + QKV proj + GQA + output proj IBP + CROWN
//!
//! **SwiGLU FFN Variants** (compose_dpdf_swiglu_variants.rs -- 15 tests):
//! - Standard SwiGLU: gate_proj -> SiLU -> mul(up_proj) -> down_proj (IBP + CROWN)
//! - SwiGLU dimension ratio: 2/3 * 4h hidden dimension scaling (IBP)
//! - SwiGLU with RMSNorm: pre-norm -> SwiGLU composition (IBP + CROWN)
//! - SwiGLU residual: x + SwiGLU(RMSNorm(x)) (IBP)
//! - SwiGLU at different scales: 256, 512, 1024 hidden dims (IBP)
//! - SwiGLU depth 2: stacked FFN layers (IBP + CROWN)
//! - SwiGLU depth 4: deep FFN chain bound widening (IBP)
//! - SwiGLU gate analysis: sigmoid gate bounded in (0, 1) (IBP)
//! - SwiGLU vs GELU FFN: bound width comparison (IBP)
//! - SwiGLU with dropout: stochastic depth skip (IBP)
//! - Quantized SwiGLU: INT4 gate/up projections (IBP)
//! - SwiGLU monotone tightening: smaller eps -> tighter output (IBP)
//! - SwiGLU + attention: decoder block FFN component (IBP + CROWN)
//! - MoE SwiGLU: expert FFN with gated routing (IBP)
//! - SwiGLU numerical stability: large input range handling (IBP)
//!
//! **Vision Encoder Depth** (compose_dpdf_encoder_depth.rs -- 15 tests):
//! - 1-block encoder: LN -> MHA -> residual -> LN -> FFN -> residual (IBP + CROWN)
//! - 2-block encoder stack: bound width after 2 blocks (IBP + CROWN)
//! - 4-block encoder stack: bound width after 4 blocks (IBP)
//! - 8-block encoder stack: bound width after 8 blocks (IBP)
//! - Bound width vs depth curve: monotone widening tracked (IBP)
//! - Pre-norm encoder: RMSNorm -> attention -> residual pattern (IBP)
//! - Post-norm encoder: attention -> LayerNorm -> residual pattern (IBP)
//! - Encoder with window attention: local attention depth scaling (IBP)
//! - Encoder with cross-attention: depth impact on cross-attn bounds (IBP)
//! - SigLIP2 encoder depth: Granite-Docling ViT depth profile (IBP)
//! - Qwen3-VL encoder depth: window ViT depth profile (IBP)
//! - SVTR encoder depth: PaddleOCR recognition encoder (IBP)
//! - Depth vs CROWN tightness: CROWN advantage at increasing depth (CROWN)
//! - Encoder depth monotone: deeper -> wider bounds verified property (IBP)
//! - Encoder depth + head: full encoder -> projection -> sigmoid (IBP)
//!
//! **Table Structure Recognition** (compose_dpdf_table_structure.rs -- 15 tests):
//! - Cell classification sigmoid: output in (0, 1) (IBP + CROWN)
//! - Cell bbox regression sigmoid: normalized coordinates (IBP)
//! - Row separator detection: binary classification head (IBP)
//! - Column separator detection: binary classification head (IBP)
//! - Row count prediction: softmax over row count bins (IBP)
//! - Column count prediction: softmax over column count bins (IBP)
//! - Cell-to-row assignment: softmax probability per cell (IBP)
//! - Cell-to-column assignment: softmax probability per cell (IBP)
//! - Rowspan prediction: sigmoid bounded in (0, 1) (IBP)
//! - Colspan prediction: sigmoid bounded in (0, 1) (IBP)
//! - Span confidence: sigmoid gating for span detection (IBP + CROWN)
//! - Detection -> structure pipeline: end-to-end cls + row + col (IBP)
//! - Structure monotone tightening: smaller eps -> tighter bounds (IBP)
//! - Multi-head table: detection + structure + span combined (IBP + CROWN)
//! - Table -> HTML: confidence-weighted structure assembly (IBP)
//!
//! **Backbone Architectures** (compose_dpdf_backbone.rs -- 15 tests):
//! - ResNet BasicBlock: Conv-BN-ReLU-Conv-BN + skip -> ReLU (IBP + CROWN)
//! - ResNet stage: 2 BasicBlocks with stride-2 downsampling (IBP)
//! - 2-stage ResNet backbone: cascaded stride-2 stages (IBP)
//! - Feature map spatial halving: verify dimensions per stage (IBP)
//! - YOLO ConvBnAct stack: 3 cascaded Conv-BN-SiLU (IBP)
//! - C2f block: cross-stage partial with bottleneck (IBP + CROWN)
//! - SPPF integration: multi-scale MaxPool2d at backbone output (IBP)
//! - Backbone -> neck FPN: lateral 1x1 conv projection (IBP)
//! - ViT patch embed + encoder block composition (IBP)
//! - 2-block ViT encoder stack (IBP + CROWN)
//! - Window attention ViT: local MHA with partition (IBP)
//! - Deep stack fusion: multi-level feature combination (IBP)
//! - Backbone output dimension comparison across models (IBP)
//! - Backbone monotone tightening: smaller eps -> tighter features (IBP)
//! - Backbone -> head projection: AvgPool -> Linear -> sigmoid (IBP)
//!
//! **Feature Pyramid Networks** (compose_dpdf_fpn.rs -- 15 tests):
//! - 1x1 conv lateral: channel reduction preserves bounds (IBP + CROWN)
//! - Multi-scale lateral: stride 4, 8, 16 feature maps (IBP)
//! - Lateral + upsample: top-down pathway element addition (IBP)
//! - 2x upsample + add: nearest-neighbor upsample + lateral fusion (IBP)
//! - 3-level top-down: P5 -> P4 -> P3 cascaded fusion (IBP)
//! - Top-down with Conv smoothing: 3x3 conv after fusion (IBP + CROWN)
//! - Bottom-up pathway: P3 -> P4 -> P5 stride-2 downsampling (IBP)
//! - PAN bidirectional: top-down + bottom-up combined (IBP)
//! - PAN with C2f blocks: CSP bottleneck in neck (IBP + CROWN)
//! - Multi-scale detection: per-level detection heads (IBP)
//! - Feature map resolution: spatial dimensions halve per level (IBP)
//! - Channel alignment: all levels same channel count (IBP)
//! - FPN monotone tightening: smaller eps -> tighter multi-scale bounds (IBP)
//! - Cross-level feature consistency: adjacent levels bounded (IBP)
//! - Full neck pipeline: backbone features -> FPN -> detection heads (IBP)
//!
//! **Decoder Stack Patterns** (compose_dpdf_decoder_stacks.rs -- 16 tests):
//! - 1-layer decoder: single attention + SwiGLU FFN block (IBP + CROWN)
//! - 2-layer decoder stack: two stacked decoder layers (IBP + CROWN)
//! - 4-layer decoder stack: four stacked decoder layers (IBP)
//! - Deep decoder 8-layer: bound width tracking through depth (IBP)
//! - Decoder with causal mask: causal attention preserves bounds (IBP)
//! - Decoder with cross-attention (DETR): self-attn + cross-attn + FFN (IBP + CROWN)
//! - Decoder + LM head: decoder -> RMSNorm -> Linear -> softmax in [0,1] (IBP)
//! - Decoder + CTC head: decoder -> Linear -> softmax CTC output (IBP)
//! - Bound width vs depth: monotone widening through 1/2/4 layers (IBP)
//! - Pre-norm vs post-norm: bound width comparison (IBP)
//! - Mixed attention: self + cross attention decoder (IBP)
//! - Decoder with MoE FFN: gate -> softmax -> expert SwiGLU (IBP)
//! - Full GLM-OCR decoder: 2-layer -> RMSNorm -> LM head -> softmax (IBP)
//!
//! **Loss Function & Output Head Patterns** (compose_dpdf_loss_heads.rs -- 16 tests):
//! - Sigmoid classification head: output in (0, 1) (IBP + CROWN)
//! - Softmax output head: sum=1 probability distribution (IBP + CROWN)
//! - DFL regression: softmax -> weighted sum box decoding (IBP)
//! - DFL -> sigmoid: normalized box coordinates in [0, 1] (IBP)
//! - CTC blank probability: narrowed softmax class in [0, 1] (IBP)
//! - CTC softmax character probabilities in [0, 1] (IBP)
//! - Focal loss weighting: (1-p)^2 * p preserves [0, 1] bounds (IBP)
//! - Box regression sigmoid: normalized coordinates (IBP)
//! - Dual-head detection: cls + box sigmoid composition (IBP)
//! - Triple-head table detection: cls + box + structure sigmoid (IBP + CROWN)
//! - MTP head chain: multi-step prediction -> softmax (IBP)
//! - LM head: Linear -> softmax token probabilities (IBP + CROWN)
//! - Log-softmax output: all elements <= 0 (IBP)
//! - Output head monotone tightening: smaller eps -> tighter bounds
//!
//! **End-to-End Model Pipelines** (compose_dpdf_end_to_end.rs -- 15 tests):
//! - DocLayout-YOLO full: image -> backbone -> neck -> detect -> sigmoid (IBP)
//! - Table Transformer full: image -> ResNet -> encoder -> decoder -> heads (IBP)
//! - PaddleOCR detection: image -> Conv-BN -> sigmoid probability map (IBP)
//! - PaddleOCR recognition: image -> patch embed -> SVTR -> CTC softmax (IBP)
//! - GLM-OCR full: embeddings -> RMSNorm -> attention -> FFN -> MTP softmax (IBP)
//! - Granite-Docling full: image -> patch embed -> ViT -> projection -> decoder -> sigmoid (IBP + CROWN)
//! - Qwen3-VL full: image -> patch embed -> window attn -> projection -> decoder -> softmax (IBP)
//! - FireRed-OCR full: image -> patch embed -> encoder -> CTC softmax (IBP)
//! - Detection + recognition cascade: image -> sigmoid detection -> CTC softmax (IBP + CROWN)
//! - VLM + layout detection: VLM features -> GELU -> sigmoid layout (IBP)
//! - Multi-model ensemble: parallel detection heads -> additive fusion (IBP)
//! - Quantized decoder pipeline: INT4 dequant -> projection -> sigmoid (IBP)
//! - Pipeline monotone tightening: tighter input -> tighter output (IBP)
//!
//! **Positional Encoding Variants** (compose_dpdf_position_encoding.rs -- 15 tests):
//! - Sinusoidal PE fixed sin/cos: bounded in [-1, 1] for all positions (IBP)
//! - Sinusoidal PE frequency scaling: higher dims -> lower frequency (IBP)
//! - Sinusoidal PE position interpolation: fractional positions bounded (IBP)
//! - Learned PE lookup: bounded by weight matrix range (IBP)
//! - Learned PE position extrapolation: OOB positions handled (IBP)
//! - Learned vs sinusoidal: bound width comparison (IBP)
//! - RoPE rotation matrix: cos/sin bounded preserves norm (IBP + CROWN)
//! - RoPE at different positions: bounded rotation (IBP)
//! - RoPE + attention: QK dot product after rotation (IBP)
//! - M-RoPE 3-component: temporal, height, width rotations (IBP)
//! - M-RoPE interleaved: alternating component application (IBP)
//! - M-RoPE vision: spatial position encoding bounded (IBP)
//! - 2D grid encoding: row + column sinusoidal (IBP)
//! - 2D PE + Conv: spatial features with position (IBP)
//! - PE monotone tightening: smaller eps -> tighter position bounds (IBP)
//!
//! **Text Detection Probability Maps** (compose_dpdf_text_detection.rs -- 15 tests):
//! - Conv backbone -> sigmoid probability map in (0, 1) (IBP + CROWN)
//! - Threshold map: Conv -> sigmoid threshold in (0, 1) (IBP)
//! - Binary map: sigmoid(k * (P - T)) approximation (IBP + CROWN)
//! - Probability map spatial resolution: stride-preserving (IBP)
//! - FPN feature fusion: multi-scale probability maps (IBP)
//! - Upsampled probability map: bilinear upsampling preserves [0, 1] (IBP)
//! - Feature pyramid 3-level: stride 4, 8, 16 maps (IBP)
//! - Hard threshold: P > 0.3 classification boundary (IBP)
//! - Soft threshold: differentiable binarization approximation (IBP + CROWN)
//! - Threshold sensitivity: small threshold change -> bounded output change (IBP)
//! - Region confidence: max probability in region bounded (IBP)
//! - Region area: spatial extent bounded by input resolution (IBP)
//! - Probability monotone tightening: smaller eps -> tighter map bounds (IBP)
//! - Detection -> recognition handoff: cropped region bounds (IBP)
//! - Full pipeline: backbone -> FPN -> probability -> binarization (IBP)
//!
//! **CTC Decoding Paths** (compose_dpdf_ctc_decoding.rs -- 15 tests):
//! - CTC logit projection: Linear(hidden, vocab) bounds (IBP)
//! - CTC softmax: per-timestep probability in [0, 1] (IBP + CROWN)
//! - CTC blank probability: blank class bounded (IBP)
//! - CTC greedy decode: argmax-narrow over softmax (IBP)
//! - CTC beam search: top-k probabilities bounded (IBP)
//! - CTC prefix merge: adjacent class bounds preserved (IBP)
//! - CTC sequence length: blank probability constrains output length (IBP)
//! - Multi-timestep CTC: 2-step and 4-step probability chains (IBP)
//! - Encoder -> CTC composition: encoder proj -> CTC softmax (IBP + CROWN)
//! - CTC confidence score: averaged per-char probability bounded (IBP)
//! - CTC log probability: log-softmax <= 0 (IBP)
//! - CTC monotone tightening: smaller eps -> tighter bounds (IBP)
//! - CTC vocabulary scaling: larger vocab -> wider per-char bounds (IBP)
//! - PaddleOCR CTC: SVTR MLP -> CTC softmax pipeline (IBP)
//! - FireRed-OCR CTC: Qwen3-VL SwiGLU -> CTC softmax pipeline (IBP)
//!
//! **DETR Decoder Patterns** (compose_dpdf_detr_decoder.rs -- 15 tests):
//! - Object query initialization: learned queries bounded (IBP)
//! - Self-attention over queries: query-to-query attention (IBP + CROWN)
//! - Cross-attention: queries attend to encoder memory (IBP + CROWN)
//! - Decoder layer: self-attn -> cross-attn -> FFN (IBP)
//! - 2-layer decoder stack: stacked decoder layers (IBP + CROWN)
//! - Query refinement: queries refined through decoder depth (IBP)
//! - Classification head: query -> Linear -> sigmoid (IBP + CROWN)
//! - Box regression head: query -> Linear -> sigmoid coordinates (IBP)
//! - Dual head: classification + box heads from same queries (IBP)
//! - Object query count scaling: 10, 50, 100 queries (IBP)
//! - Encoder-decoder projection: feature dim alignment (IBP)
//! - Decoder with sinusoidal PE: position encoding for queries (IBP)
//! - No-object class: background class sigmoid bounded (IBP)
//! - Decoder monotone tightening: smaller eps -> tighter bounds (IBP)
//! - Full DETR pipeline: encoder -> decoder -> heads (IBP + CROWN)
//!
//! **Reading Order & Layout** (compose_dpdf_reading_order.rs -- 15 tests):
//! - Spatial position encoding: box (x, y, w, h) -> feature projection (IBP)
//! - Pairwise box relationship features: overlap/distance (IBP)
//! - Reading order classifier MLP: sigmoid output bounded (IBP + CROWN)
//! - Column detection: softmax column assignment (IBP)
//! - Table cell adjacency: sigmoid adjacency probability (IBP)
//! - Multi-column layout: spatial features -> column softmax (IBP)
//! - Page-level aggregation: mean pooling over boxes (IBP)
//! - Spatial self-attention: layout feature refinement (IBP)
//! - Box coordinate normalization: sigmoid to [0, 1] (IBP)
//! - Layout classification head: multi-label sigmoid (IBP + CROWN)
//! - Reading order pairwise comparison: sigmoid bounded (IBP)
//! - Spatial distance features: L1/L2 projection bounded (IBP)
//! - Layout region merging: adjacent region merge probability (IBP)
//! - Hierarchical layout: page -> column -> paragraph softmax (IBP)
//! - Full layout pipeline: coords -> attention -> classification (IBP + CROWN)
//!
//! **MoE Expert Routing & Sparse Gating** (compose_dpdf_moe_routing.rs -- 18 tests):
//! - Router softmax output bounds (sum to 1) IBP
//! - Top-k expert selection gate bounds IBP
//! - Expert capacity factor bounds across input ranges IBP
//! - Load balancing auxiliary loss bounds IBP
//! - Sparse gating: top-1 of 8 experts (most zeroed out) IBP
//! - Expert FFN output bounds per expert IBP
//! - Combined expert output (weighted sum) bounds IBP
//! - MoE vs dense FFN bound comparison IBP
//! - Expert dropout/jitter noise bounds IBP
//! - Router temperature scaling effect IBP
//! - 2-expert vs 8-expert routing bounds IBP
//! - MoE residual: x + MoE(norm(x)) bounds IBP
//! - Expert specialization: per-expert output range IBP
//! - Router z-loss: penalizes large router logits IBP
//! - MoE depth composition (2-layer stacked MoE) IBP
//! - MoE depth composition (2-layer stacked MoE) CROWN
//! - Full MoE block: router -> experts -> combine -> residual IBP
//! - Full MoE block: router -> experts -> combine -> residual CROWN
//!
//! **Vision-Language Projection & Token Merging** (compose_dpdf_vl_projection.rs -- 15 tests):
//! - Vision-to-language linear projection (VIS_DIM -> LLM_DIM) IBP bounds
//! - MLP projection (2-layer with GELU) IBP + CROWN bounds
//! - Token merging/pooling (spatial -> sequence) IBP bounds
//! - Cross-modal LayerNorm before projection IBP + CROWN bounds
//! - Projection with residual connection IBP bounds
//! - Dynamic resolution token count IBP bounds
//! - Spatial token flattening (H*W -> seq_len) IBP bounds
//! - Perceiver resampler (fixed-length output) IBP bounds
//! - Vision token compression ratio IBP bounds
//! - Multi-scale vision token fusion before projection IBP bounds
//! - Projection dimension alignment (vision_dim -> llm_dim) IBP + CROWN bounds
//! - Token type embedding addition IBP bounds
//! - Position embedding for projected tokens IBP bounds
//! - Projection + RoPE composition IBP bounds
//! - Full VL projection: vision encoder -> merge -> project -> LLM input IBP + CROWN
//!
//! **RoPE & M-RoPE Variants** (compose_dpdf_rope_variants.rs -- 15 tests):
//! - Standard RoPE (cos/sin rotation) bounds (IBP)
//! - M-RoPE temporal, height, width component bounds (IBP)
//! - 3-component M-RoPE combined bounds (IBP)
//! - Interleaved M-RoPE (Qwen3-VL) bounds (IBP)
//! - 2D sinusoidal PE (SigLIP2/Granite-Docling) bounds (IBP)
//! - RoPE frequency scaling bounds across base frequencies (IBP)
//! - RoPE with extended context (YaRN) bounds (IBP)
//! - RoPE applied to QK in attention bounds (IBP)
//! - Vision vs text position encoding comparison (IBP)
//! - Absolute vs relative position encoding bounds (IBP)
//! - RoPE numerical stability at large positions (IBP)
//! - Position encoding interpolation bounds (IBP)
//! - Full attention with RoPE: QKV proj + RoPE + SDPA bounds (IBP + CROWN)
//!
//! **Quantized Weight Inference (INT4 GPTQ/AWQ)** (compose_dpdf_quantized_inference.rs -- 15 tests):
//! - INT4 dequantization: scale * (q - zero_point) linear bounds (IBP)
//! - Group quantization: per-group scale/zero bounds (IBP)
//! - Quantized linear layer output bounds (IBP + CROWN)
//! - INT4 vs FP16 output bound comparison (IBP)
//! - GPTQ quantization error bounds (IBP)
//! - AWQ activation-aware quantization bounds (IBP)
//! - Quantized attention QKV projection bounds (IBP + CROWN)
//! - Quantized FFN (SwiGLU with INT4 weights) bounds (IBP)
//! - Mixed precision: INT4 weights + FP16 activations (IBP)
//! - Quantized residual connection bounds (IBP)
//! - Quantized LayerNorm interaction bounds (IBP + CROWN)
//! - INT8 vs INT4 quantization precision comparison (IBP)
//! - Quantized embedding lookup bounds (IBP)
//! - Quantized detection head output bounds (IBP)
//! - Full quantized decoder block: attention + FFN + residual (IBP + CROWN)
//!
//! **Quantization Preservation (INT4/INT8 vs FP32)** (compose_dpdf_quantization_preservation.rs -- 22 tests):
//! - GPTQ INT4 dequant bounds for Qwen3-VL MoE expert FFN (IBP)
//! - GPTQ INT4 per-group scale variation across experts (IBP)
//! - GPTQ INT4 expert gate + FFN end-to-end (IBP + CROWN)
//! - GPTQ INT4 MoE residual: quantized expert + skip (IBP)
//! - AWQ per-channel salient scale preserves output bounds (IBP)
//! - AWQ channel-wise vs uniform quantization comparison (IBP)
//! - AWQ quantized SwiGLU FFN preserves output range (IBP)
//! - AWQ quantized attention + FFN decoder block (IBP + CROWN)
//! - INT8 QK scoring bounds (SageAttention-style) (IBP)
//! - INT8 QK + FP32 PV accumulation split-precision (IBP)
//! - INT8 attention with smooth-K channel subtraction (IBP)
//! - INT8 attention vs FP32 attention bound width comparison (IBP)
//! - Per-layer quantization error: INT4 linear vs FP32 linear (IBP)
//! - Per-layer quantization error: INT8 linear vs FP32 linear (IBP)
//! - Quantization error accumulation: 2-layer stack margin growth (IBP)
//! - Quantized softmax output margin: bounded deviation from FP32 (IBP)
//! - Group-size partitioning preserves weight tensor shape (IBP)
//! - Group-size impact: smaller groups produce tighter per-element bounds (IBP)
//! - Non-aligned group boundary: partial last group (IBP)
//! - Group quantization monotone tightening: tighter input -> tighter output (IBP)
//!
//! **Sequence Decoder (CTC vs Attention)** (compose_dpdf_seq_decoder.rs -- 15 tests):
//! - CTC decoder softmax output: per-timestep probability in [0, 1] (IBP)
//! - Attention decoder autoregressive step: cross-attn + FFN + softmax (IBP + CROWN)
//! - CTC vs attention output bound comparison: width analysis (IBP)
//! - CTC blank token probability: blank class bounded in [0, 1] (IBP)
//! - Attention decoder cross-attention: encoder-decoder attention (IBP + CROWN)
//! - Hybrid CTC+attention joint decoding: weighted combination (IBP)
//! - CTC prefix beam search score: top-k probabilities bounded (IBP)
//! - Attention decoder teacher forcing vs inference: bound width comparison (IBP)
//! - CTC time-step independence: per-step softmax consistency (IBP)
//! - Attention decoder causal mask interaction: masked attention (IBP)
//! - CTC character-level output distribution: per-class in [0, 1] (IBP)
//! - Attention decoder vocabulary projection: Linear -> softmax (IBP + CROWN)
//! - CTC greedy decode vs beam search bound difference (IBP)
//! - Attention decoder with KV-cache: cached context bounds (IBP)
//! - Full recognition pipeline: encoder -> decoder -> output (IBP + CROWN)
//!
//! **Depthwise Separable Convolution** (compose_dpdf_depthwise_conv.rs -- 15 tests):
//! - Depthwise Conv2d (groups=channels) IBP bounds
//! - Pointwise Conv2d (1x1) IBP bounds
//! - Depthwise separable: depthwise -> pointwise composition (IBP)
//! - Depthwise + BatchNorm + ReLU pipeline (IBP)
//! - Inverted residual (MBConv): expand -> depthwise -> project + skip (IBP)
//! - Squeeze-and-Excitation after depthwise conv (IBP)
//! - MBConv with expansion ratio 4 (IBP)
//! - MBConv with expansion ratio 6 (IBP)
//! - Stride-2 depthwise for spatial downsampling (IBP)
//! - Depthwise conv with different kernel sizes (3x3, 5x5) (IBP)
//! - CROWN tightness for depthwise separable blocks (CROWN)
//! - Stacked MBConv blocks (2-layer) (IBP + CROWN)
//! - Depthwise separable monotone tightening (IBP)
//! - Depthwise separable vs standard conv bound comparison (IBP)
//! - Full MBConv stage: 3 MBConv blocks with stride (IBP)
//!
//! **Normalization Variant Bounds** (compose_dpdf_norm_variants.rs -- 15 tests):
//! - BatchNorm running stats offset effect on bounds (IBP)
//! - LayerNorm tighter input produces tighter output (IBP)
//! - RMSNorm tighter input produces tighter output (IBP)
//! - GroupNorm (G=32) production-scale group count (IBP)
//! - InstanceNorm standalone verification (IBP)
//! - BN vs LN bound width comparison (IBP)
//! - RMSNorm vs LN bound width comparison (IBP)
//! - GroupNorm group count effect (G=4/8/16) (IBP)
//! - BatchNorm large affine (gamma/beta) bounds (IBP)
//! - LayerNorm scaled affine bounds (IBP)
//! - Norm + activation: LayerNorm -> GELU composition (IBP)
//! - Pre-norm residual: x + Linear(LayerNorm(x)) (IBP)
//! - Pre-norm vs post-norm comparison (IBP)
//! - Normalization numerical stability (small variance inputs) (IBP)
//! - Full Conv -> BN -> SiLU block (YOLO backbone pattern) (IBP)
//!
//! **Patch Embedding & Image Tokenization** (compose_dpdf_patch_embed.rs -- 15 tests):
//! - Basic Conv2d patch projection (patch_size=14) IBP
//! - Patch projection with patch_size=16 IBP
//! - Patch projection with patch_size=32 IBP
//! - Patch flatten + transpose for ViT sequence format IBP
//! - Learnable position embedding addition bounds IBP
//! - Sinusoidal 2D position embedding bounds IBP
//! - CLS token prepend bounds propagation IBP
//! - Conv2d projection with different channels (3->768 vs 3->1024) IBP
//! - Patch embedding + LayerNorm composition (IBP + CROWN)
//! - Batch dimension handling in patch projection IBP
//! - CROWN tightness vs IBP for patch embedding CROWN
//! - Overlapping patch embedding (stride < patch_size) IBP
//! - Interpolated position embedding for variable resolution IBP
//! - Patch merging (downsampling via reshape+linear) IBP
//! - End-to-end image-to-tokens pipeline bounds (IBP + CROWN)
//!
//! **Sliding Window & Local Attention** (compose_dpdf_sliding_window.rs -- 15 tests):
//! - Basic sliding window mask generation bounds (IBP)
//! - Window partition + local attention + unpartition pipeline (IBP)
//! - Window size effect on attention bound tightness (IBP)
//! - Interleaved window/global attention pattern (Qwen3-VL) (IBP)
//! - Window attention with padding for non-divisible sequence lengths (IBP)
//! - Dilated/strided window attention bounds (IBP)
//! - 2D spatial window partitioning for vision features (IBP)
//! - Window attention + relative position bias composition (IBP)
//! - Cross-window information flow (shifted windows, Swin-style) (IBP)
//! - Window attention with causal mask (IBP)
//! - Multi-head window attention with GQA (IBP + CROWN)
//! - CROWN tightness for window vs global attention (CROWN)
//! - Window attention memory efficiency bounds (IBP)
//! - Overlapping windows for boundary continuity (IBP)
//! - End-to-end windowed ViT encoder block (IBP + CROWN)
//!
//! **Vocabulary Projection & Sampling Heads** (compose_dpdf_vocab_projection.rs -- 15 tests):
//! - Linear projection to vocabulary size (hidden_dim -> vocab_size) IBP
//! - Tied weight embedding projection IBP
//! - Vocabulary projection + softmax composition IBP + CROWN
//! - Vocabulary projection + log-softmax for CTC loss IBP
//! - Temperature-scaled logits bounds IBP
//! - Top-k filtering effect on output bounds IBP
//! - Top-p (nucleus) sampling threshold bounds IBP
//! - Greedy decoding (argmax) output bounds IBP
//! - Beam search score accumulation bounds IBP
//! - CTC blank token probability bounds IBP
//! - Large vocabulary (151k tokens for Qwen3) projection bounds IBP
//! - Multi-head CTC output (character + position) bounds IBP
//! - CROWN tightness for vocabulary projection layer CROWN
//! - Logit bias/mask application bounds IBP
//! - End-to-end hidden-to-token probability pipeline IBP + CROWN
//!
//! **Cross-Attention Patterns** (compose_dpdf_cross_attention.rs -- 15 tests):
//! - Basic cross-attention: query attends to key-value memory (IBP)
//! - Cross-attention with LayerNorm pre-processing (IBP + CROWN)
//! - Cross-attention with residual connection: q + Attn(q, kv) (IBP)
//! - Multi-head cross-attention with h=8 heads (IBP)
//! - Cross-attention softmax weights bounded in [0, 1] (IBP)
//! - Cross-attention with different Q and KV dimensions (IBP)
//! - Cross-attention + FFN decoder layer (IBP + CROWN)
//! - Self-attention -> cross-attention sequential (DETR decoder) (IBP)
//! - Cross-attention with position encoding added to queries (IBP)
//! - Cross-attention with position encoding added to keys (IBP)
//! - Stacked cross-attention (2-layer decoder) (IBP + CROWN)
//! - Cross-attention KV from vision encoder features (VLM pattern) (IBP)
//! - Cross-attention monotone tightening: smaller eps -> tighter bounds (IBP)
//! - CROWN tightness for cross-attention vs IBP (CROWN)
//! - Full decoder layer: LN + self-attn + LN + cross-attn + LN + FFN (IBP + CROWN)
//!
//! **Language Model Head (LM Head)** (compose_dpdf_lm_head.rs -- 15 tests):
//! - RMSNorm before LM head projection (IBP)
//! - Linear projection hidden_dim -> vocab_size (IBP)
//! - RMSNorm + Linear composition (IBP + CROWN)
//! - Softmax output in [0, 1] after LM head (IBP)
//! - Log-softmax output <= 0 after LM head (IBP)
//! - Temperature scaling effect on output bounds (IBP)
//! - Top-k logit masking effect on bounds (IBP)
//! - Repetition penalty application bounds (IBP)
//! - LM head with tied embeddings (weight sharing) (IBP)
//! - Multi-token prediction: 2-step LM head chain (IBP)
//! - LM head numerical stability (large logits) (IBP)
//! - CROWN tightness for RMSNorm + LM head (CROWN)
//! - LM head monotone tightening: smaller eps -> tighter bounds (IBP)
//! - LM head with different vocab sizes (32k, 64k, 151k) (IBP)
//! - Full decoder -> RMSNorm -> LM head -> softmax pipeline (IBP + CROWN)
//!
//! **Weight Initialization** (compose_dpdf_weight_init.rs — 15 tests):
//! - Xavier uniform / Kaiming normal weight range -> output bound width (IBP)
//! - Small / large weight initialization -> tighter / wider output bounds (IBP)
//! - Bias initialization effect on output shift, zero-bias symmetry (IBP)
//! - Weight scale factor, magnitude correlation, sparsity, clipping (IBP)
//! - Embedding weight range -> lookup bound width (IBP)
//! - Normalization gamma near 1 -> output bound width (IBP)
//! - CROWN tightness with different weight ranges (CROWN)
//! - Tied vs independent weights comparison (IBP)
//! - Full model: Linear -> SiLU -> Linear -> RMSNorm -> Linear (IBP + CROWN)
//!
//! Part of #4048.
//!
//! Part of #4063: Compose tests for KV-cache update and autoregressive generation bounds.
//! Part of #4062: Compose tests for token merging and spatial reduction bounds.
//!
//! **Token Merging and Spatial Reduction** (compose_dpdf_token_merging.rs — 12 tests):
//! - Adaptive average pooling spatial reduction (IBP)
//! - Spatial reshape (H*W -> seq_len) (IBP)
//! - Token concatenation from multi-scale features (IBP)
//! - Vision-to-text linear projection after pooling (IBP)
//! - Adaptive pooling CROWN bounds (CROWN)
//! - Multi-scale feature concatenation + projection (IBP)
//! - Token selection/sampling pattern (IBP)
//! - Spatial reduction via strided convolution (IBP)
//! - Token merging with attention weights (IBP)
//! - Vision-text projection after spatial reduction (CROWN)
//! - Monotone tightening for spatial reduction pipeline (IBP)
//! - Full pipeline: Conv features -> pool -> reshape -> project -> text embedding (IBP)
//!
//! **Trace-to-Graph Translation Fidelity** (compose_dpdf_trace_fidelity.rs — 20 tests):
//! - Linear layer trace: MatMul + Add maps to Linear (IBP + CROWN)
//! - RMSNorm decomposition: reshape -> instance_norm -> reshape -> affine (IBP + CROWN)
//! - Conv2d trace: Conv2d op maps to conv layer (IBP + CROWN)
//! - SiLU activation: x * sigmoid(x) decomposition (IBP)
//! - GELU activation: tanh-approx GELU bounds (IBP)
//! - Sigmoid activation: bounded in [0, 1] (IBP + CROWN)
//! - Softmax decomposition: exp -> sum -> div captured (IBP + CROWN)
//! - Residual connection: Add of two branches (IBP + CROWN)
//! - Reshape preservation: shape ops retain element bounds (IBP)
//! - Transpose preservation: permutation retains element bounds (IBP)
//! - Reshape + transpose chain: composed shape ops (IBP)
//! - Linear -> GELU -> linear pipeline: multi-op fidelity (IBP)
//! - RMSNorm -> linear -> sigmoid pipeline: norm + activation (IBP)
//! - Full pipeline: Conv2d -> reshape -> linear -> softmax end-to-end (IBP)
//!
//! Part of #4095.
//!
//! **Output Calibration** (compose_dpdf_output_calibration.rs — 20 tests):
//! - Temperature scaling: low T narrows, high T widens, T=1 identity (IBP)
//! - Temperature extreme values preserve softmax [0,1] (IBP + CROWN)
//! - Top-k masking: zeroes non-top-k paths, k=1 sharp, k=vocab unmasked (IBP)
//! - Top-k + temperature composition (IBP + CROWN)
//! - DocLayout-YOLO sigmoid detection confidence in [0,1] (IBP)
//! - Detection sigmoid monotonicity and CROWN tightness (IBP + CROWN)
//! - CTC softmax bounded for PaddleOCR (IBP)
//! - Autoregressive OCR log-softmax bounded in (-inf, 0] (IBP)
//! - CTC blank token dominance (IBP)
//! - MoE gate temperature, sum bounded near 1.0, CROWN tightness (IBP + CROWN)
//! - Large vocab projection range scaling (IBP)
//! - Vocab projection + softmax output in [0,1] (IBP)
//! - Vocab projection + temperature + softmax end-to-end (IBP + CROWN)
//!
//! Part of #4102.
//!
//! **Cross-Model Pipeline Chaining** (compose_dpdf_cross_model_pipeline.rs — 19 tests):
//! - Layout detection output shape: [N, num_classes+4] bounding boxes (IBP)
//! - OCR input from cropped layout regions: bounds preservation (IBP)
//! - OCR confidence score range: [0, 1] after sigmoid (IBP)
//! - Table structure detection input from layout crop (IBP)
//! - VLM input from OCR text + image: combined bounds (IBP)
//! - End-to-end image to detection boxes bounds (IBP)
//! - Detection to crop: coordinate clipping to image bounds (IBP)
//! - Crop to OCR: resized input bounds preservation (IBP)
//! - OCR to text: token embedding bounds (IBP)
//! - Multi-model dtype conversion: FP32 between models (IBP)
//! - Resolution scaling: 640x640 detection to variable OCR (IBP)
//! - Confidence threshold filtering: bounds after threshold (IBP)
//! - NMS output: non-overlapping box bounds (IBP)
//! - Batch pipeline: N pages processed independently (IBP)
//! - Pipeline error propagation: bounds widening per stage (IBP)
//! - Model output calibration through pipeline (IBP + CROWN)
//! - Table cell extraction from structure + OCR (IBP)
//! - Document understanding: layout + OCR + table composition (IBP + CROWN)
//!
//! Part of #4123.
//!
//! **Decoder Autoregressive Generation** (compose_dpdf_decoder_generation.rs — 18 tests):
//! - Single step logits: hidden -> Linear -> vocab logits (IBP)
//! - Logit bounds from bounded embeddings: embed -> decoder -> LM head (IBP)
//! - Greedy decoding: softmax probability distribution in [0, 1] (IBP)
//! - Temperature scaling: logits * (1/T) -> softmax for T=0.7, T=2.0 (IBP)
//! - Top-k filtering: projection to top-k -> softmax (IBP)
//! - Top-p nucleus sampling: projection to nucleus subset -> softmax (IBP)
//! - Softmax on filtered logits: temperature + top-k combined (IBP)
//! - Beam search: projection to beam-width candidates -> softmax (IBP)
//! - Beam score accumulation: summed log-softmax <= 0 (IBP)
//! - KV cache: attention with extended key/value sequence (IBP)
//! - KV cache growth rate: monotonic widening at cache lengths 2/4/8 (IBP)
//! - Causal mask: future positions masked in generation (IBP)
//! - Stop token detection: sigmoid on stop-token logit in [0, 1] (IBP)
//! - Max length enforcement: decoder at max position produces bounded logits (IBP)
//! - Repetition penalty: penalized logit multiply -> softmax (IBP)
//! - Cross-attention: encoder-decoder attention bounds propagation (IBP + CROWN)
//! - Multi-step generation: bound width widening at 1/2/4 layers (IBP)
//! - Full pipeline: embed -> decoder -> RMSNorm -> LM head -> softmax (IBP + CROWN)
//!
//! Part of #4130.
//!
//! **Vision Encoder Feature Extraction** (compose_dpdf_vision_encoder.rs — 18 tests):
//! - Patch embedding Conv2d output bounds (IBP)
//! - Position embedding addition bounds (IBP)
//! - Single encoder block: self-attention + MLP (IBP + CROWN)
//! - Multi-head attention bounds within encoder (IBP)
//! - LayerNorm before attention bounds (IBP)
//! - LayerNorm after FFN bounds (IBP)
//! - CLS token output bounds after full encoder (IBP)
//! - 2-layer ViT encoder bounds (IBP + CROWN)
//! - 4-layer ViT encoder bounds (IBP)
//! - Window attention bounds (Qwen3-VL WindowViT) (IBP)
//! - Multi-scale feature extraction (different resolutions) (IBP)
//! - Patch merging (reducing spatial resolution) (IBP)
//! - Feature pyramid from encoder layers (IBP)
//! - SigLIP2 encoder output bounds (IBP + CROWN)
//! - Global average pooling after encoder (IBP)
//! - Image resolution scaling effect on bounds (IBP)
//! - Encoder with different embedding dimensions (IBP)
//! - Skip connection from early to late encoder layers (IBP)
//!
//! Part of #4129.
//!
//! **OCR Pipeline E2E** (compose_dpdf_ocr_pipeline_e2e.rs — 18 tests):
//! - Layout detection crop bounds: bounding box within image bounds (IBP)
//! - Detection confidence filter: threshold preserves high-confidence boxes (IBP)
//! - Crop resize bounds: aspect-ratio resize preserves pixel range (IBP)
//! - CTC output length bounded: output <= feature_length (IBP)
//! - CTC beam log probability: beam scores <= 0 (IBP)
//! - Table cell within table: cell bbox inside table bbox (IBP)
//! - Table structure spanning: spanning cells bounded by table dims (IBP)
//! - Reading order permutation: ordering is permutation of regions (IBP)
//! - Page confidence bounded: page confidence in [min_region, max_region] (IBP)
//! - Pipeline latency additive: total <= sum of per-model latency (IBP)
//! - Pipeline memory sequential peak: peak = max of per-model peaks (IBP)
//! - Multipage independent bounds: per-page bounds independent (IBP)
//! - Detection miss propagation: miss rate -> recognition coverage (IBP)
//! - Ensemble voting narrows: 2+ model votes narrow output bounds (IBP)
//! - Fallback chain bounds: primary failure -> secondary bounds (IBP + CROWN)
//! - NMS IOU monotone: higher threshold -> fewer detections (IBP)
//! - Vocabulary constraint tightens: known vocab narrows recognition (IBP)
//! - Full pipeline output bounded: image -> JSON field count bounded (IBP + CROWN)
//!
//! Part of #4142.
//!
//! **GLM-OCR Cross-Modal** (compose_dpdf_glm_cross_modal.rs — 15 tests):
//! - Vision projection + RMSNorm: Linear(VISION, HIDDEN) -> RMSNorm (IBP + CROWN)
//! - Cross-attention GQA: Vision KV, decoder Q through grouped-query attention (IBP + CROWN)
//! - Cross-attention + SwiGLU FFN: Attention -> gated FFN + residual (IBP + CROWN)
//! - Vision projection + decoder block: Cross-modal -> full decoder layer (IBP + CROWN)
//! - MTP head from cross-attention: RMSNorm -> Linear -> softmax in [0, 1] (IBP)
//! - Vision + 2-layer decoder + LM head: End-to-end cross-modal (IBP + CROWN)
//! - Tight-input cross-attention: Narrow +-0.1 bounds for CROWN precision (CROWN)
//! - Cross-modal residual accumulation: Self-attn + cross-attn residual compose (IBP)
//!
//! Part of #4304.
//!
//! **PaddleOCR-VL Visual Encoder Deep** (compose_dpdf_paddle_visual_encoder_deep.rs — 15 tests):
//! - SVTR self-attention isolation: LN -> Q/K/V + softmax + out_proj + residual (IBP + CROWN)
//! - SVTR MLP GELU block: LN -> Linear -> GELU -> Linear + residual (IBP + CROWN)
//! - Full SVTR encoder block: LN+Attn+residual -> LN+MLP(GELU)+residual (IBP + CROWN)
//! - Patch embed + encoder block: Conv2d -> reshape -> encoder block (IBP + CROWN)
//! - 2-block SVTR encoder + CTC head: Depth -> LN -> Linear -> softmax (IBP + CROWN)
//! - Tight-input encoder block: Narrow +-0.1 for CROWN precision (IBP + CROWN)
//! - DB detector backbone + sigmoid: Conv-BN-ReLU -> 1x1 conv -> sigmoid (IBP + CROWN)
//!
//! Part of #4304.
//!
//! **FireRed-OCR Decoder Deep** (compose_dpdf_firered_decoder_deep.rs — 16 tests):
//! - RoPE Q/K application: cos/sin positional encoding on Q projection (IBP + CROWN)
//! - RoPE + GQA attention: RoPE-modulated Q/K through attention + softmax (IBP + CROWN)
//! - Decoder layer with RoPE: RMSNorm -> RoPE-GQA -> SwiGLU -> residuals (IBP + CROWN)
//! - 2-layer decoder stack: Depth composition with RoPE-attention (IBP)
//! - Decoder + CTC head: 2-layer -> RMSNorm -> Linear -> softmax (IBP + CROWN)
//! - Cross-attention decoder layer: RMSNorm -> Q/K/V attention -> RMSNorm (IBP + CROWN)
//! - Tight-input RoPE attention: Narrow +-0.1 bounds for CROWN precision (CROWN)
//! - Vision-decoder-CTC: Projection -> decoder -> CTC head -> softmax (IBP)
//!
//! Part of #4304.
//!
//! **FireRed-OCR Decoder Pipeline** (compose_dpdf_firered_decoder_pipeline.rs — 17 tests):
//! - Text decoder causal attention: RMSNorm -> causal self-attn -> residual (IBP + CROWN)
//! - Cross-attention vision-to-decoder: encoder features attend to decoder (IBP + CROWN)
//! - Autoregressive generation step: decoder -> RMSNorm -> LM head -> softmax (IBP)
//! - Multi-step generation accumulation: 2-step decoder -> softmax (IBP)
//! - Beam search score propagation: softmax + additive score accumulation (IBP)
//! - E2E vision-encoder-cross-attn-decoder: patch -> enc -> proj -> xattn -> dec -> softmax (IBP + CROWN)
//! - LM head probability bounds: RMSNorm -> Linear -> softmax in [0, 1] (IBP)
//! - Token embedding lookup: Linear(VOCAB, HIDDEN) embedding projection (IBP)
//! - Position encoding addition: learned PE + RMSNorm stability (IBP + CROWN)
//! - CROWN tightening decoder: narrow +-0.5 input decoder block (CROWN)
//! - Decoder + cross-attn + LM head: xattn -> decoder -> softmax pipeline (IBP)
//! - Verify-and-record: E2E pipeline + xattn+LM pipeline status recording (IBP)
//!
//! Part of #4240.
//!
//! **DocLayout-YOLO Backbone Deep** (compose_dpdf_doclayout_backbone_deep.rs — 14 tests):
//! - C2f entry + bottleneck: Conv-BN-SiLU entry -> bottleneck + skip (IBP + CROWN)
//! - C2f full block: Entry -> bottleneck -> concat -> exit conv (IBP + CROWN)
//! - Backbone stage: ConvBnAct stride-2 + C2f bottleneck (IBP + CROWN)
//! - Detection head sigmoid: 1x1 conv -> sigmoid in [0, 1] (IBP + CROWN)
//! - Backbone-to-detection: Stage + detection head cross-stage (IBP + CROWN)
//! - Image-to-detection: 2-stage backbone -> sigmoid end-to-end (IBP + CROWN)
//!
//! Part of #4304.
//!
//! **7-Model Ensemble Pipeline** (compose_dpdf_ensemble.rs — 15 tests):
//! - DocLayout-YOLO multi-scale detection: conv -> pool -> sigmoid head (IBP)
//! - Table Transformer DETR: attention -> sigmoid bbox regression (IBP + CROWN)
//! - FireRed-OCR: patch embed -> encoder -> CTC softmax (IBP)
//! - Qwen3-VL: patch embed -> MLP projection (IBP + CROWN)
//! - Granite-Docling: ViT -> linear vision-LM bridge (IBP + CROWN)
//! - GLM-OCR: FFN -> MTP head -> softmax (IBP)
//! - PaddleOCR: DB sigmoid + SVTR CTC softmax (IBP)
//! - Detection -> Table cascade (IBP)
//! - Detection -> FireRed-OCR cascade (IBP)
//! - Detection -> Qwen3-VL cascade (IBP)
//! - 3-model pipeline: layout -> table -> OCR (IBP)
//! - 4-model pipeline: layout -> table -> OCR -> language (IBP)
//! - Ensemble confidence aggregation: 3-head sigmoid average (IBP + CROWN)
//! - Ensemble monotone tightening: narrow input -> narrow output (IBP)
//! - 7-model dispatch routing: softmax gate -> heads -> sigmoid (IBP)
//!
//! Part of #4243.
//!
//! **Post-Processing Pipeline** (compose_dpdf_postprocess_nms_decode.rs — 14 tests):
//! - Box decoding center-to-corner: sigmoid -> affine scaling (IBP)
//! - Score thresholding via sigmoid: confidence in [0, 1] (IBP)
//! - NMS IoU threshold score: sigmoid -> threshold -> ReLU (IBP)
//! - Multi-class NMS pipeline: per-class filter + objectness gate (IBP)
//! - Softmax class probability: softmax output in [0, 1] (IBP)
//! - Text line horizontal merge: Linear -> sigmoid merge score (IBP)
//! - Text line vertical merge: Linear -> sigmoid merge score (IBP)
//! - OCR CTC decode: Linear -> softmax character probabilities (IBP)
//! - Table cell grid assignment: Linear -> sigmoid row/col assignment (IBP)
//! - Full detect -> NMS -> decode pipeline: cls + box + threshold (IBP)
//! - CROWN score thresholding: sigmoid -> threshold -> ReLU (IBP + CROWN)
//! - CROWN box decoding: sigmoid -> scale affine bounds (IBP + CROWN)
//! - DFL box regression: softmax -> weighted sum in [0, bins-1] (IBP)
//! - Detection -> OCR -> merge: 3-stage pipeline composition (IBP)
//!
//! Part of #4213.
//!
//! **Training Loop Pipeline** (compose_dpdf_training_loop_pipeline.rs -- 14 tests):
//! - Forward pass linear chain with activations (Linear -> ReLU -> Linear) (IBP)
//! - MSE loss computation bounds: (pred - target)^2 bounded (IBP)
//! - Cross-entropy style loss: log_softmax + neg-product bounded (IBP)
//! - Gradient flow through linear layers: transposed weight matmul (IBP + CROWN)
//! - Optimizer step learning rate scaling: weight - lr * gradient (IBP)
//! - Weight update bound preservation: updated weights stay bounded (IBP)
//! - Multi-step training stability: 2-step forward-backward chain (IBP + CROWN)
//! - Mixed precision training: BF16-range vs FP32 weight magnitudes (IBP)
//! - Batch normalization running stats update: BN inference + ReLU (IBP)
//! - Learning rate warmup schedule: linear warmup scaling (IBP)
//! - Learning rate cosine decay: midpoint annealing (IBP)
//! - Gradient clipping: tanh smooth clipping + scaled update (IBP)
//! - Full training step end-to-end: forward + softmax + backward proxy (IBP + CROWN)
//! - Verify-and-record: forward pipeline with status recording (IBP)
//!
//! Part of #4219.

// Shared helpers are #[path]-included by multiple child submodules.
#![allow(clippy::duplicate_mod)]

mod common;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_granite_docling.rs"]
mod granite_docling;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_doclayout_yolo.rs"]
mod doclayout_yolo;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_glm_ocr.rs"]
mod glm_ocr;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_transformer.rs"]
mod table_transformer;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_qwen3_vl.rs"]
mod qwen3_vl;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_paddle_ocr.rs"]
mod paddle_ocr;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_firered_ocr.rs"]
mod firered_ocr;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_granite_docling_siglip2_deep.rs"]
mod granite_docling_siglip2_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_paddle_ocr_deep.rs"]
mod paddle_ocr_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_cross_model.rs"]
mod cross_model;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_quantized.rs"]
mod quantized;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_certification.rs"]
mod certification;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_glm_ocr_deep.rs"]
mod glm_ocr_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_doclayout_yolo_deep.rs"]
mod doclayout_yolo_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_qwen3_vl_deep.rs"]
mod qwen3_vl_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_firered_ocr_deep.rs"]
mod firered_ocr_deep;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_cross_pipeline.rs"]
mod cross_pipeline;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_adversarial.rs"]
mod adversarial;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_normalization.rs"]
mod normalization;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_spatial.rs"]
mod spatial;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_activations.rs"]
mod activations;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_embeddings.rs"]
mod embeddings;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_quantization.rs"]
mod quantization;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_attention.rs"]
mod attention;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_loss_heads.rs"]
mod loss_heads;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_moe.rs"]
mod moe;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_residual.rs"]
mod residual;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_decoder_stacks.rs"]
mod decoder_stacks;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_multimodal.rs"]
mod multimodal;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_end_to_end.rs"]
mod end_to_end;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_structure.rs"]
mod table_structure_compose;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_text_detection.rs"]
mod text_detection;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_backbone.rs"]
mod backbone;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_position_encoding.rs"]
mod position_encoding;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_ctc_decoding.rs"]
mod ctc_decoding;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_fpn.rs"]
mod fpn;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_swiglu_variants.rs"]
mod swiglu_variants;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_encoder_depth.rs"]
mod encoder_depth;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_detr_decoder.rs"]
mod detr_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_nms_detection.rs"]
mod nms_detection;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_gqa_kvcache.rs"]
mod gqa_kvcache;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_reading_order.rs"]
mod reading_order;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_moe_routing.rs"]
mod moe_routing;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_vl_projection.rs"]
mod vl_projection;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_multiscale_fusion.rs"]
mod multiscale_fusion;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_rope_variants.rs"]
mod rope_variants;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_quantized_inference.rs"]
mod quantized_inference;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_seq_decoder.rs"]
mod seq_decoder;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_norm_variants.rs"]
mod norm_variants;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_patch_embed.rs"]
mod patch_embed;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_depthwise_conv.rs"]
mod depthwise_conv;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_sliding_window.rs"]
mod sliding_window;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_vocab_projection.rs"]
mod vocab_projection;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_cross_attention.rs"]
mod cross_attention;

/// Multi-Token Prediction (MTP) patterns: parallel/sequential prediction heads,
/// RMSNorm gating, tied weights, decoder composition, dropout masking.
/// Part of #4042.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_mtp_patterns.rs"]
mod mtp_patterns;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_lm_head.rs"]
mod lm_head;

/// Attention mask patterns: causal, padding, sliding window, block-sparse,
/// cross-attention masks, prefix LM, multi-head broadcasting, full block.
/// Part of #4043.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_attention_masks.rs"]
mod attention_masks;

/// Gradient flow and training stability bounds: linear backward, ReLU masking,
/// softmax Jacobian, residual identity, LayerNorm backward, attention backward,
/// SwiGLU gated flow, deep residual, gradient clipping, vanishing/exploding
/// gradient detection, CROWN tightness, monotone tightening, skip connections,
/// full forward-backward pipeline.
/// Part of #4049.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_gradient_flow.rs"]
mod gradient_flow;

/// Pooling and spatial reduction patterns: global avg pool, adaptive avg pool,
/// max pool, attention pooling, token pooling, SPP, pool+linear classifier,
/// CROWN tightness, monotone tightening, full classification pipeline.
/// Part of #4047.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_pooling_patterns.rs"]
mod pooling_patterns;
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_weight_init.rs"]
mod weight_init;

/// Sequence length and position extrapolation bounds: variable-length MHA,
/// PE extrapolation, RoPE extended positions, causal mask length effects,
/// encoder scaling, decoder generation length, KV-cache growing context,
/// truncation, padding, CROWN tightness, monotone tightening, full
/// transformer block at multiple lengths.
/// Part of #4055.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_seq_length.rs"]
mod seq_length;

/// Multi-resolution and dynamic shape handling: spatial resolutions (224, 384, 512),
/// padding to square, adaptive pooling, patch embedding, feature pyramid, conv at
/// non-standard sizes, batch independence, dynamic token count, resolution
/// interpolation, CROWN at different resolutions, monotone tightening, full pipeline.
/// Part of #4056.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_dynamic_shape.rs"]
mod dynamic_shape;

/// Dropout and stochastic depth effects on bounds: eval-mode identity,
/// stochastic depth alpha sweep, dropout mask scaling, attention dropout,
/// FFN dropout, layer drop, scale factor comparison, CROWN tightness,
/// monotone tightening, deep model, full block eval pipeline.
/// Part of #4060.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_dropout_effects.rs"]
mod dropout_effects;

/// KV-cache update and autoregressive generation bounds: cache append, cache-attended
/// cross-attention, prefill vs decode, multi-step growth, RoPE update, generation
/// length effects, CROWN tightness, monotone tightening, full autoregressive step.
/// Part of #4063.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_kv_cache_update.rs"]
mod kv_cache_update;

/// Mixed-precision inference bounds: BF16/F16 epsilon perturbation, precision loss
/// accumulation, softmax/normalization stability, attention score precision,
/// CROWN tightness, monotone tightening, full mixed-precision transformer block.
/// Part of #4066.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_mixed_precision.rs"]
mod mixed_precision;

/// Token merging and spatial reduction bounds: adaptive pooling, spatial reshape,
/// multi-scale concatenation, vision-to-text projection, strided reduction,
/// attention-based merging, CROWN tightness, monotone tightening, full pipeline.
/// Part of #4062.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_token_merging.rs"]
mod token_merging;

/// Activation function variant bounds: GELU, SiLU, Mish, Swish, hardswish,
/// approximate GELU, activation pipelines, CROWN tightness, monotone tightening.
/// Part of #4061.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_activation_variants.rs"]
mod activation_variants_v2;

/// Certified per-model output bounds for 7 dpdf architectures.
/// Part of #4078.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_certified_outputs.rs"]
mod certified_outputs;

/// Robustness certification under input perturbation for dpdf model subgraphs.
/// Part of #4084.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_robustness.rs"]
mod robustness;

/// Pipeline composition bounds: cross-model verification for dpdf pipelines.
/// Part of #4088.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_pipeline.rs"]
mod pipeline;
/// Quantization preservation: INT4/INT8 vs FP32 bounds preservation through
/// GPTQ, AWQ, SageAttention INT8, per-layer margin analysis, and group
/// quantization structure verification.
/// Part of #4087.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_quantization_preservation.rs"]
mod quantization_preservation;

/// Trace-to-graph translation fidelity: verifies that tensor_kernel_to_graph
/// produces correct GraphNetwork representations from DynTensor operation traces.
/// Covers linear, RMSNorm, Conv2d, activations (SiLU/GELU/sigmoid), softmax,
/// residual connections, reshape/transpose preservation, and multi-op pipelines.
/// Part of #4095.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_trace_fidelity.rs"]
mod trace_fidelity;

/// Model output calibration: temperature scaling, top-k masking, detection
/// confidence, OCR logit bounds, MoE routing calibration, vocabulary projection.
/// Verifies that calibration transforms preserve expected bound invariants
/// across dpdf model architectures.
/// Part of #4102.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_output_calibration.rs"]
mod output_calibration;
/// Weight initialization bounds (Xavier/Kaiming): Xavier uniform output width,
/// Kaiming uniform output width, fan ratio scaling, two-layer and four-layer
/// deep ReLU pipelines, sigmoid/softmax output boundedness, zero-bias structure
/// preservation, bias shift without widening, CROWN tightness at Xavier/Kaiming
/// scale, initialization scale monotonicity.
/// Part of #4101.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_weight_initialization.rs"]
mod weight_initialization;

/// Residual stream bounds propagation: multi-block pre-norm streams, dense
/// residual (DenseNet concatenation), residual scaling alpha sweep, cross-
/// attention residual, encoder-decoder full residual, RMSNorm streams,
/// monotone tightening, mixed-norm residual streams.
/// Residual stream bounds propagation: basic residual, feedforward comparison,
/// pre-norm RMSNorm, post-norm LayerNorm, dropout scaling, stacked depth,
/// U-Net skip, DenseNet concat, learned scaling, cross-attention residual,
/// parallel residual (GPT-J), norm growth rate, FPN lateral, gated highway,
/// projection shortcut, nested double residual, gradient flow, Demucs skip.
/// Part of #4112.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_residual_stream.rs"]
mod residual_stream;

/// Model depth scaling bounds growth: measures output bound width growth
/// with model depth (1/2/4/8 layer stacks). Verifies residual connection
/// effect, LayerNorm/RMSNorm normalization tightening, CROWN vs IBP depth
/// scaling, and sigmoid/softmax output activation capping.
/// Model depth scaling bounds growth: 1/2/4/8 layer stack bounds through
/// transformer, MLP, attention, conv, LSTM, and mixed architectures.
/// Residual connections, normalization, and activation comparison.
/// Part of #4111.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_depth_scaling.rs"]
mod depth_scaling;

#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_activation_proofs.rs"]
mod activation_proofs;
/// ay SMT proofs for embedding lookup bounds and vocabulary index safety.
/// Covers index bounds, output dimension, weight shape, out-of-bounds detection,
/// norm bounds, padding index, scale factor, position offsets, BPE/WordPiece,
/// gradient sparsity, tied weights, quantization error, and more.
/// Part of #4109.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_embedding_proofs.rs"]
mod embedding_proofs;
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_positional_encoding_proofs.rs"]
mod positional_encoding_proofs;

/// ay SMT proofs for loss function mathematical properties.
/// Covers cross-entropy, focal loss, KL divergence, MSE, L1, Huber,
/// CTC, gradient bounds, weighted CE, temperature scaling, contrastive,
/// and triplet loss.
/// Part of #4115.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_loss_function_proofs.rs"]
mod loss_function_proofs;
/// ay SMT proofs for softmax and cross-entropy loss mathematical properties.
/// Covers sum-to-one, positivity, (0,1) range, translation invariance,
/// log-softmax identity, CE non-negativity, CE one-hot formula, KL divergence,
/// temperature scaling (high/low), gradient Jacobian, numerical stability,
/// log-softmax gradient, label smoothing, focal loss, monotonicity, masking,
/// top-k concentration, logit ratio bound, binary CE, temperature-zero one-hot.
/// Part of #4232.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_softmax_loss_proofs.rs"]
mod softmax_loss_proofs;

/// ay SMT proofs for matrix multiplication mathematical properties.
/// Covers distributivity, associativity, identity, zero, transpose reversal,
/// scalar multiplication, output dimensions, batched matmul, output bounds,
/// inner product non-negativity, Cauchy-Schwarz, trace cyclicity,
/// symmetric matrix squared, orthogonal norm preservation, rank-1 update,
/// block diagonal independence, diagonal scaling, and Gram matrix PSD diagonal.
/// Part of #4121.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_matmul_proofs.rs"]
mod matmul_proofs;

/// Normalization stability through deep model stacks: LayerNorm, RMSNorm,
/// GroupNorm, BatchNorm, InstanceNorm single layers, stacked 2/4-layer
/// compositions, pre-norm vs post-norm comparison, double normalization
/// idempotency, large epsilon stability, CROWN backward bounds, activation
/// ordering (ReLU before/after norm), scale reset verification.
/// Part of #4117.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_normalization_stability.rs"]
mod normalization_stability;

/// Activation function bounds propagation through multi-layer architectures:
/// ReLU clipping, GELU/SiLU/Mish/sigmoid/tanh bounds, LayerNorm+activation,
/// stacked MLP depth scaling, mixed activations, residual connections,
/// softmax probability bounds, gradient bounds, GLU/SwiGLU gated activations.
/// Part of #4118.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_activation_bounds.rs"]
mod activation_bounds;

/// ay SMT proofs for convolution output dimension formulas: Conv1d/Conv2d
/// output length/height/width, same/valid/full padding, stride effects,
/// dilation effective kernel size, 1x1/3x3 spatial preservation, groups
/// divisibility, depthwise conv, output channel independence, transposed
/// conv output formula, output_padding < stride, spatial >= 1, conv+pool
/// composition, max pool dimension formula.
/// Part of #4127.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_conv_dimension_proofs.rs"]
mod conv_dimension_proofs;
/// Cross-model pipeline chaining: end-to-end bounds through multi-model
/// document understanding pipelines (layout -> OCR -> table -> VLM).
/// Detection output shape, crop bounds, confidence thresholding, NMS,
/// resolution scaling, dtype conversion, batch independence, error
/// propagation, calibration (CROWN), table cell extraction, full
/// document understanding compose.
/// Part of #4123.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_cross_model_pipeline.rs"]
mod cross_model_pipeline;
/// Decoder autoregressive generation bounds: greedy, beam search, sampling.
/// Single step logits, embedding-to-logit propagation, greedy softmax,
/// temperature scaling, top-k filtering, nucleus sampling, filtered softmax,
/// beam search, beam score accumulation, KV cache bounds, KV cache growth,
/// causal mask, stop token detection, max length enforcement, repetition
/// penalty, cross-attention, multi-step bounds widening, full pipeline
/// (IBP + CROWN).
/// Part of #4130.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_decoder_generation.rs"]
mod decoder_generation;
/// ay SMT proofs for normalization layer mathematical properties.
/// Covers LayerNorm (mean-zero, unit-variance, affine, epsilon), RMSNorm (formula,
/// positivity, unit magnitude), BatchNorm (running mean/var update, inference vs
/// training modes), GroupNorm (channel divisibility, per-group independence),
/// InstanceNorm (per-sample independence), scale invariance, idempotency,
/// gradient bounds, epsilon positivity, affine initialization, weight decay bounds.
/// Part of #4128.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_normalization_proofs.rs"]
mod normalization_proofs;
/// Quantization-aware inference bounds: INT4/INT8 vs FP32 equivalence margins,
/// symmetric/asymmetric quantization, GPTQ group-wise dequant, AWQ per-channel
/// scaling, INT8 attention QKV, quantized MLP (SwiGLU), error delta bounding,
/// mixed precision, dequant-compute-requant pipeline, quantized softmax/LayerNorm,
/// accumulator overflow safety, residual connections, per-token quantization,
/// weight-only vs activation quantization, end-to-end output difference bounds.
/// Part of #4124.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_quantization_inference.rs"]
mod quantization_inference_bounds;
/// ay SMT proofs for softmax and attention score numerical properties.
/// Covers softmax sum-to-one, positivity, ordering preservation, shift invariance,
/// log-softmax identity, temperature scaling, gradient Jacobian, scaled dot-product
/// attention, Cauchy-Schwarz score bounds, row-stochastic weights, causal mask,
/// attention output bounds, multi-head dimension split, GQA KV-repeat, self-attention
/// symmetry, ALiBi linear bias, top-k sparsity, sliding window, cross-attention.
/// Part of #4122.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_softmax_attention_proofs.rs"]
mod softmax_attention_proofs;

/// ay SMT proofs for dropout and regularization mathematical properties.
/// Covers dropout inverted scaling 1/(1-p), eval-mode identity, mask
/// element-wise application, output bounds, gradient pass-through,
/// L2 weight decay, L1 regularization, AdamW decoupled decay, label
/// smoothing (valid range, sum-to-one, identity at alpha=0), stochastic
/// depth (survival scaling, identity at s=1, dropped path), DropPath
/// (1/keep_prob scaling, batch independence), two-layer dropout composition.
/// Part of #4134.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_dropout_regularization_proofs.rs"]
mod dropout_regularization_proofs;
/// ay SMT proofs for pooling layer properties: max pool output bounded by
/// max input, avg pool between min/max, adaptive pool output size, global
/// average pool per-channel, stride/kernel dimension formulas, max pool
/// ordering preservation, convex combination, gradient sparsity/uniformity,
/// multi-scale pooling (SPP), Lp pool bounds, pool chain composition.
/// Part of #4133.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_pooling_proofs.rs"]
mod pooling_proofs;
/// Vision encoder feature extraction bounds (compose_dpdf_vision_encoder.rs — 18 tests).
///
/// Patch embedding Conv2d output bounds, position embedding addition,
/// single encoder block (LN+MHA+FFN), multi-head attention, LayerNorm
/// pre/post, CLS token extraction, 2-layer and 4-layer ViT stacks,
/// window attention (Qwen3-VL), multi-scale feature extraction, patch
/// merging, feature pyramid, SigLIP2 encoder output, global average
/// pooling, resolution scaling, embedding dimension comparison, skip
/// connections from early to late layers.
/// Part of #4129.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_vision_encoder.rs"]
mod vision_encoder;

/// KV cache bounds growth and memory safety (compose_dpdf_kv_cache_bounds.rs — 18 tests).
///
/// Cache shape after N steps, bounds bounded by activation range,
/// concatenation preserves entries, append order invariance, multi-step
/// concat chain, concat with projection, paged KV block allocation,
/// paged cross-block attention, GQA cache sharing, GQA with attention
/// output, RoPE with cache offset, RoPE offset monotone growth,
/// sliding window cache eviction, sliding window with attention,
/// full autoregressive step, multi-layer KV cache depth composition.
/// Part of #4136.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_kv_cache_bounds.rs"]
mod kv_cache_bounds;

/// Cross-architecture composition verification (compose_dpdf_cross_arch.rs -- 12 tests).
///
/// Detection→Recognition pipeline (sigmoid→linear→ReLU→softmax, IBP+CROWN+monotone),
/// Layout→Table pipeline (box coords→query proj→LayerNorm→sigmoid, IBP+CROWN),
/// Vision encoder comparison (SigLIP2 Conv2d+LN vs Qwen3-VL Conv2d+RMSNorm),
/// Multi-model softmax consistency (same and varying hidden dims),
/// Shared backbone invariant (conv+BN+activation, head independence, CROWN).
/// Part of #3956.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_cross_arch.rs"]
mod cross_arch;

/// ay SMT proofs for attention score scaling and causal masking properties.
/// Scaling factor positivity, ordering preservation, causal mask structure,
/// softmax of masked scores, attention weight sum-to-one and [0,1] range,
/// convex combination output bounds, multi-head dimension constraints,
/// GQA divisibility and repeat factor, sliding window sparsity, ALiBi
/// linear penalty and geometric slopes, cross-attention full access,
/// dropout expected value, numerically stable softmax, temperature scaling.
/// Part of #4139.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_attention_scaling_proofs.rs"]
mod attention_scaling_proofs;
/// ay SMT proofs for MoE expert routing and capacity factor properties.
/// Covers router softmax (sum-to-one, unit interval), top-k selection (exactly k,
/// weight bounds, renormalization), capacity factor formula, load balancing loss,
/// auxiliary loss independence, expert index bounds, token assignment, convex
/// combination output bounds, shared expert additive composition, token dispatch
/// count preservation, token combine order recovery, jitter noise bounds, expert
/// utilization fraction, router z-loss non-negativity, capacity overflow dropping.
/// Part of #4140.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_moe_routing_proofs.rs"]
mod moe_routing_proofs;

/// Compose tests for mixed-precision bf16/f16/f32 inference pipeline bounds.
/// Precision conversion chains: f32/bf16/f16 quantization, upcast preservation,
/// overflow clamping, bf16 matmul accumulator, mixed-precision attention/layernorm/
/// SwiGLU, rounding bounds, denormal flush, roundtrip error, loss scaling,
/// gradient unscaling, residual connections, int8 conv accumulation,
/// GPTQ/AWQ dequant, full pipeline composition (IBP + CROWN).
/// Part of #4141.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_precision_chain.rs"]
mod precision_chain;

/// Compose tests for end-to-end document OCR pipeline bounds composition.
/// Part of #4142.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_ocr_pipeline_e2e.rs"]
mod ocr_pipeline_e2e;

/// ay SMT proofs for beam search and CTC decoding mathematical properties.
/// Part of #4147.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_decoding_math_proofs.rs"]
mod decoding_math_proofs;
/// Compose tests for DETR decoder object query and cross-attention pipeline.
/// Part of #4148.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_detr_pipeline.rs"]
mod detr_pipeline;
/// ay SMT proofs for KV cache growth bounds and memory allocation properties.
/// Covers cache size linear in steps (N * head_dim * kv_heads), paged block count
/// (ceil formula), paged memory, append length, monotone memory growth, GQA kv_heads
/// and repeat factor, sliding window cache length and memory bound, pre-allocation
/// fixed memory and utilization fraction, multi-layer total, RoPE offset position,
/// cross-attention fixed cache, concatenation length, byte alignment, token-level
/// update, batch cache total, bf16/f32 element sizes, max cache bound.
/// Part of #4146.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_kv_cache_math_proofs.rs"]
mod kv_cache_math_proofs;

/// Compose tests for GLM-OCR reading order and text line detection pipeline.
/// Part of #4154.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_glm_reading_order.rs"]
mod glm_reading_order;
/// ay SMT proofs for image preprocessing and normalization math.
/// Part of #4152.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_image_preprocess_math_proofs.rs"]
mod image_preprocess_math_proofs;
/// ay SMT proofs for SwiGLU and gated FFN mathematical properties.
/// Part of #4158.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_swiglu_math_proofs.rs"]
mod swiglu_math_proofs;
/// ay SMT proofs for weight quantization and dequantization math.
/// Part of #4153.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_weight_quant_math_proofs.rs"]
mod weight_quant_math_proofs;

/// ay SMT proofs for RoPE and position encoding mathematical properties.
/// Part of #4159.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_rope_extended_proofs.rs"]
mod rope_extended_proofs;

/// ay SMT proofs for RoPE mathematical properties in VLM decoders.
/// Part of #4229.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_rope_math_proofs.rs"]
mod rope_math_proofs;

/// ay SMT proofs for batch normalization math. Part of #4166.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_batchnorm_math_proofs.rs"]
mod batchnorm_math_proofs;
/// Compose tests for FireRed-OCR encoder-decoder pipeline bounds.
/// Part of #4160.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_firered_pipeline.rs"]
mod firered_pipeline;
/// ay SMT proofs for FPN multi-scale fusion math.
/// Part of #4163.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_fpn_fusion_proofs.rs"]
mod fpn_fusion_proofs;
/// Compose tests for Qwen3-VL MoE pipeline. Part of #4168.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_qwen3_moe_pipeline.rs"]
mod qwen3_moe_pipeline;

/// ay SMT proofs for cross-attention decoder math. Part of #4170.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_cross_attention_proofs.rs"]
mod cross_attention_proofs;
/// ay SMT proofs for LayerNorm and RMSNorm math. Part of #4176.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_layernorm_math_proofs.rs"]
mod layernorm_math_proofs;
/// ay SMT proofs for residual connection math. Part of #4173.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_residual_connection_proofs.rs"]
mod residual_connection_proofs;
/// Compose tests for Table Transformer DETR pipeline. Part of #4177.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_transformer_pipeline.rs"]
mod table_transformer_pipeline;

/// Compose tests for Granite-Docling-258M pipeline. Part of #4180.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_granite_docling_pipeline.rs"]
mod granite_docling_pipeline;

/// ay SMT proofs for GQA math. Part of #4179.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_gqa_math_proofs.rs"]
mod gqa_math_proofs;

/// ay SMT proofs for depthwise separable conv math. Part of #4181.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_depthwise_conv_proofs.rs"]
mod depthwise_conv_proofs;
/// Compose tests for GLM-OCR decoder pipeline. Part of #4183.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_glm_ocr_decoder_pipeline.rs"]
mod glm_ocr_decoder_pipeline;

/// Compose tests for DocLayout-YOLO detection pipeline. Part of #4186.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_doclayout_yolo_pipeline.rs"]
mod doclayout_yolo_pipeline;

/// Compose tests for FireRed-OCR full pipeline. Part of #4196.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_firered_full_pipeline.rs"]
mod firered_full_pipeline;
/// Compose tests for full document pipeline. Part of #4199.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_full_document_pipeline.rs"]
mod full_document_pipeline;
/// ay SMT proofs for matmul/linear math. Part of #4197.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_matmul_linear_proofs.rs"]
mod matmul_linear_proofs;
/// ay SMT proofs for multi-scale fusion math. Part of #4184.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_multiscale_fusion_proofs.rs"]
mod multiscale_fusion_proofs;
/// Compose tests for PaddleOCR-VL pipeline. Part of #4194.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_paddle_pipeline.rs"]
mod paddle_pipeline;
/// Compose tests for Qwen3-VL MoE full pipeline. Part of #4192.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_qwen3_moe_full_pipeline.rs"]
mod qwen3_moe_full_pipeline;
/// ay SMT proofs for SwiGLU FFN math. Part of #4190.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_swiglu_ffn_proofs.rs"]
mod swiglu_ffn_proofs;

/// ay SMT proofs for dropout and regularization math. Part of #4202.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_dropout_proofs.rs"]
mod dropout_proofs;

/// Compose tests for audio pipeline (Silero VAD + Whisper). Part of #4273.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_audio_pipeline.rs"]
mod audio_pipeline;
/// Deep compose tests for Granite-Docling decoder pipeline. Part of #4273.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_granite_decoder_deep.rs"]
mod granite_decoder_deep;
/// Compose tests for image preprocessing pipeline. Part of #4273.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_image_preprocess.rs"]
mod image_preprocess;
/// Compose tests for multi-model orchestration pipeline. Part of #4273.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_orchestration_pipeline.rs"]
mod orchestration_pipeline;
/// Deep compose tests for Table Transformer structure recognition. Part of #4273.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_table_transformer_deep.rs"]
mod table_transformer_deep_v2;

/// GLM-OCR vision-to-decoder cross-modal composition: vision projection + RMSNorm,
/// cross-attention GQA, cross-attn + SwiGLU, vision + full decoder block, MTP head,
/// vision + 2-layer decoder + LM head, tight-input analysis, residual accumulation.
/// Part of #4304.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_glm_cross_modal.rs"]
mod glm_cross_modal;

/// PaddleOCR-VL visual encoder deep: SVTR self-attention, MLP GELU, full encoder
/// block, patch embed + encoder, 2-block SVTR + CTC, tight-input CROWN, DB detector
/// backbone + sigmoid, verify-and-record.
/// Part of #4304.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_paddle_visual_encoder_deep.rs"]
mod paddle_visual_encoder_deep;

/// FireRed-OCR Qwen3-VL decoder deep: RoPE Q/K application, RoPE + GQA attention,
/// decoder layer with RoPE, 2-layer stack, decoder + CTC head, cross-attention
/// decoder layer, tight-input RoPE, vision-decoder-CTC pipeline.
/// Part of #4304.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_firered_decoder_deep.rs"]
mod firered_decoder_deep;

/// DocLayout-YOLO backbone deep: C2f entry + bottleneck, C2f full block, backbone
/// stage with stride-2, detection head sigmoid, backbone-to-detection, image-to-
/// detection end-to-end, verify-and-record.
/// Part of #4304.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_doclayout_backbone_deep.rs"]
mod doclayout_backbone_deep;

/// dpdf 7-model ensemble pipeline compose tests: DocLayout-YOLO multi-scale
/// detection, Table Transformer DETR attention, FireRed-OCR CTC, Qwen3-VL
/// vision projection, Granite-Docling vision-LM bridge, GLM-OCR MTP head,
/// PaddleOCR detect+recognize, cross-model cascades (detection->table,
/// detection->OCR, detection->VLM), 3-model and 4-model pipelines, ensemble
/// confidence aggregation, monotone tightening, 7-model dispatch routing.
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_ensemble.rs"]
mod ensemble;

/// DocLayout-YOLO multi-scale detection pipeline compose tests (19 tests):
/// CSPDarkNet stem/stage, two-stage channel expansion, three-scale P3/P4/P5
/// extraction, backbone CROWN, FPN top-down path, PAN bottom-up path,
/// FPN+PAN combined neck, neck lateral CROWN, P3/P5 detection heads,
/// dual-head DFL+sigmoid, detection head CROWN, end-to-end P3/P5 detection,
/// monotone tightening, widening analysis, objectness x class confidence
/// scoring (sigmoid*softmax), multi-scale output merge (concat), NMS
/// confidence thresholding (sigmoid->threshold->ReLU, IBP+CROWN).
/// Part of #4234.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_doclayout_yolo_multiscale.rs"]
mod doclayout_yolo_multiscale;

/// FireRed-OCR vision-language pipeline compose tests.
/// Part of #4240.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_fire_red_ocr.rs"]
mod fire_red_ocr;

/// Qwen3-VL vision encoder pipeline compose tests.
/// Part of #4231.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_qwen3_vl_vision_encoder.rs"]
mod qwen3_vl_vision_encoder;

/// Qwen3-VL vision encoder extended compose tests: CROWN variants for
/// RoPE/projection/2-block stack, window attention, temporal patch embedding,
/// verify-and-record for patch embed/encoder block/GQA/SwiGLU/full pipeline.
/// Part of #4231.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_qwen3_vl_vision_encoder_ext.rs"]
mod qwen3_vl_vision_encoder_ext;

/// Qwen3-VL vision encoder pipeline compose tests: LayerNorm-based ViT blocks,
/// GELU-activated visual token projection, multi-scale patch merge, and
/// attention+FFN composition bounds (12 tests, IBP + CROWN).
/// Part of #4231.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_qwen3_vl_encoder.rs"]
mod qwen3_vl_encoder;

/// Granite-Docling encoder-decoder full pipeline compose tests.
/// Part of #4228.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_granite_docling_enc_dec.rs"]
mod granite_docling_enc_dec;

/// Extended Granite-Docling cross-attention compose tests: asymmetric
/// encoder/decoder dimensions, MHA head splitting, encoder re-projection,
/// full enc->xattn->dec pipeline, post-xattn LayerNorm, combined
/// self+cross attention, CROWN tightening, 3-layer decoder stack,
/// varying encoder sequence lengths, verify-and-record (10 tests).
/// Part of #4228.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_granite_docling_enc_dec_ext.rs"]
mod granite_docling_enc_dec_ext;

/// Table Transformer full DETR pipeline compose tests: ResNet backbone,
/// 6-layer encoder, decoder cross-attention, object query embeddings,
/// classification/bbox FFN heads, Hungarian matching, row/column detection,
/// cell spanning, position encoding, multi-scale features, layer norm,
/// full encoder-decoder pipeline, NMS confidence filtering.
/// Part of #4237.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_detr_full.rs"]
mod table_detr_full;

/// Extended Table Transformer DETR pipeline tests: CROWN variants for
/// backbone, encoder, bbox head, row/column, NMS; ResNet residual block;
/// encoder+PE composition; monotone tightening; verification-recording.
/// Part of #4237.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_detr_extended.rs"]
mod table_detr_extended;

/// PaddleOCR-VL text detection subpipeline compose tests.
/// Part of #4222.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_paddleocr_vl.rs"]
mod paddleocr_vl;

/// PaddleOCR-VL detection + recognition subpipeline compose tests:
/// PP-OCRv4 ResNet backbone, DB detection head [0,1], SVTR self-attention,
/// CTC decoder probability, box regression, NMS score filtering, FPN multi-scale,
/// recognition normalization, vocabulary probability distribution, batch detection,
/// orientation classifier, line grouping geometry, beam search, full pipeline.
/// Part of #4222.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_paddle_detection.rs"]
mod paddle_detection;

/// PaddleOCR-VL recognition pipeline compose tests:
/// ResNet encoder backbone, LayerNorm normalization (IBP + CROWN), multi-scale
/// FPN fusion, CTC linear/FFN decoders, cross-attention decoder, causal attention
/// with residual, variable-length padding, character embeddings, beam search top-k,
/// confidence scoring, detect-to-encoder composition, full recognition pipeline
/// end-to-end (IBP + CROWN).
/// Part of #4222.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_paddle_recognition.rs"]
mod paddle_recognition;

/// GLM-OCR full decoder pipeline bounds: embedding, RoPE, SwiGLU, causal attention,
/// RMSNorm contraction, 2-layer decoder stack, vocab projection, KV cache consistency.
/// Part of #4225.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_glm_ocr_full_decoder.rs"]
mod compose_dpdf_glm_ocr_full_decoder;

/// Quantization pipeline bound preservation (13 tests): F16/BF16/INT8 cast
/// precision, per-channel INT8 bounds, symmetric roundtrip, INT8 vs FP32
/// linear, mixed-precision attention, reduced-precision LayerNorm, full
/// transformer block INT8 vs FP32, asymmetric INT8 zero-point, per-channel
/// vs per-tensor comparison, quantized matmul error accumulation, quantized
/// softmax [0,1] bounds, full encoder-decoder mixed-precision pipeline.
/// Part of #4216.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_quantization_pipeline.rs"]
mod quantization_pipeline;

/// Granite-Docling encoder-decoder full pipeline compose tests:
/// ViT patch embedding, encoder self-attention, position embedding,
/// decoder cross-attention + causal self-attention + SwiGLU FFN,
/// token prediction head, structured output, table structure,
/// OCR text line detection, layout classification.
/// Part of #4228.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_granite_encoder_decoder.rs"]
mod granite_encoder_decoder;

/// dpdf 7-model ensemble pipeline composition tests: pipeline stage
/// composition, parallel dispatch, result aggregation, confidence-weighted
/// selection, fallback chains, full page-to-structured-data pipeline,
/// multi-page batch processing, detection-to-multi-OCR fan-out,
/// OCR-to-language aggregation, table+OCR parallel merge, ensemble
/// monotone tightening, confidence-weighted ensemble, page attention.
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_7model_ensemble.rs"]
mod ensemble_7model;

/// Per-model standalone bounds for the 7-model ensemble: DocLayout-YOLO
/// (conv -> relu -> sigmoid cls/bbox), Table Transformer (FFN + residual ->
/// sigmoid, IBP + CROWN), Granite-Docling (vision proj -> LM FFN -> softmax),
/// PaddleOCR-VL (SVTR -> softmax), FireRed-OCR (CTC -> softmax, IBP + CROWN),
/// GLM-OCR (SwiGLU FFN + residual -> LM head), Qwen3-VL (GELU MLP -> FFN).
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_7model_ensemble_extended.rs"]
mod ensemble_7model_extended;

/// Cross-model interaction and ensemble composition tests: feature fusion
/// (concat features -> MLP), majority voting (logit sum + temperature softmax),
/// vision-to-LM cascade (4-stage), ensemble monotone CROWN, end-to-end
/// realistic 7-head pipeline, hierarchical region-type routing, confidence
/// calibration with temperature scaling.
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_7model_ensemble_interactions.rs"]
mod ensemble_7model_interactions;

/// GLM-OCR full image-to-text pipeline bounds: vision encoder feature extraction,
/// cross-attention image-text, RoPE, SwiGLU FFN, causal attention mask, token
/// embedding, LM head logits, temperature scaling, full img2txt pipeline,
/// multi-turn conversation, system prompt, image patch embedding, KV projections,
/// output token probability distribution.
/// Part of #4225.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_glm_ocr_full_pipeline.rs"]
mod glm_ocr_full_pipeline;

/// DocLayout-YOLO multi-scale detection pipeline compose tests:
/// CSPDarknet backbone stages, C2f bottleneck, SPPF, FPN top-down cascade,
/// PAN bottom-up cascade, per-scale detection heads (P3/P4/P5),
/// DFL box regression, objectness scores, class probabilities,
/// NMS thresholding, multi-scale fusion, anchor-free decoding,
/// full backbone-to-detection pipeline.
/// Part of #4234.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_doclayout_multiscale.rs"]
mod doclayout_multiscale;

/// FireRed-OCR vision-language pipeline compose tests: ViT visual encoder,
/// visual token projection, decoder self-attention, cross-attention, RoPE,
/// SwiGLU FFN, RMSNorm sandwich, LM head, OCR char prediction, layout PE,
/// multi-resolution features, full pipeline, confidence scores, reading order.
/// Part of #4240.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_firered_vision_lang.rs"]
mod firered_vision_lang;

/// FireRed-OCR decoder pipeline compose tests: causal decoder attention,
/// vision-to-decoder cross-attention, autoregressive generation step,
/// multi-step generation accumulation, beam search score propagation,
/// end-to-end vision-encoder-cross-attn-decoder pipeline, LM head probs,
/// token embedding lookup, position encoding addition, CROWN tightening,
/// decoder + cross-attn + LM head pipeline, verify-and-record.
/// Part of #4240.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_firered_decoder_pipeline.rs"]
mod firered_decoder_pipeline;

/// ay SMT proofs for convolution stride and padding mathematical properties
/// (part 1, tests 1031-1040): output size formula, stride positivity,
/// same padding, dilation equivalence, transposed conv inverse, depthwise
/// decomposition, grouped weight partitioning, causal padding, linearity.
/// Part of #4226.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_conv_stride_proofs.rs"]
mod conv_stride_proofs;

/// ay SMT proofs for convolution stride and padding advanced properties
/// (part 2, tests 1041-1050): output boundedness, downsampling, odd kernel
/// symmetry, receptive field, stacked receptive field, separable decomposition,
/// bias bounds, group norm, deformable offset bounds, same padding consistency.
/// Part of #4226.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_conv_stride_advanced_proofs.rs"]
mod conv_stride_advanced_proofs;

/// Post-processing pipeline compose tests: NMS score propagation, box decoding
/// (center-to-corner), confidence filtering, multi-class NMS, text line merging,
/// OCR CTC decode, table cell grid assignment, DFL regression, and full
/// detect->NMS->decode->merge pipeline (14 tests). Part of #4213.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_postprocess_nms_decode.rs"]
mod postprocess_nms_decode;

/// Training loop pipeline bound verification (14 tests):
/// forward pass, MSE/cross-entropy loss, gradient flow, optimizer step,
/// weight update preservation, multi-step stability, mixed precision,
/// BatchNorm running stats, LR warmup/cosine decay, gradient clipping,
/// full training step E2E, verify-and-record.
/// Part of #4219.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_training_loop_pipeline.rs"]
mod training_loop_pipeline;

/// Extended GeLU re-verification compose tests across model pipelines:
/// standalone GeLU CROWN, GeLU FFN (Qwen3/GLM5 pattern), GeLU vs SwiGLU
/// comparison, LayerNorm+GeLU FFN, quantization sensitivity, 3-layer MLP
/// depth widening, residual block, RMSNorm+GeLU FFN, IBP vs CROWN sweep.
/// Part of #4314.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_gelu_reverify_extended.rs"]
mod gelu_reverify_extended;

/// GLM-OCR deep decoder pipeline compose tests (15 tests):
/// RMSNorm Conservative Sound, fused SwiGLU with narrow-based gate/up split,
/// GQA self-attention (IBP + CROWN), single/multi-layer decoder blocks with
/// Conservative NormBoundsMode, 2-layer and 3-layer decoder stacks (IBP + CROWN),
/// LM head + softmax output bounds, full end-to-end pipeline (IBP + CROWN),
/// tight-input precision analysis, monotone widening across depth,
/// large vocabulary projection scaling. Production-representative dimensions
/// (D_MODEL=32, FFN=64, N_HEADS=4, SEQ=8, VOCAB=64).
/// Part of #4225.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_glm_ocr_decoder_deep.rs"]
mod glm_ocr_decoder_deep;

/// Extended Whisper encoder-decoder compose verification tests (25 tests):
/// Mel spectrogram normalization + conv features (IBP + CROWN), encoder
/// self-attention with sinusoidal PE injection (IBP + CROWN), encoder FFN
/// with GELU (IBP + CROWN + verify), cross-attention + FFN decoder (IBP),
/// 3-layer encoder stack depth composition (IBP + CROWN + verify + widening),
/// decoder autoregressive single-step (IBP + verify), LM head with softmax
/// output bounded in [0,1] (IBP + CROWN + verify), full encoder-decoder E2E
/// pipeline mel->tokens (IBP + CROWN + verify), timestamp token prediction
/// head with narrow/softmax bounds (IBP + CROWN + verify + wide-input).
/// Part of #4560.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_whisper_encoder_decoder_extended.rs"]
mod whisper_encoder_decoder_extended;

/// Table and form extraction compose verification tests (18 tests).
/// Part of #4320.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_extraction.rs"]
mod compose_dpdf_table_extraction;

/// FireRed-OCR VLM pipeline compose tests (18 tests).
/// Part of #4240.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_firered_ocr_vlm.rs"]
mod firered_ocr_vlm;

/// Compose tests for Table Transformer DETR full pipeline (22 tests).
/// Part of #4237.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_table_transformer_detr_pipeline.rs"]
mod table_transformer_detr_pipeline;

/// dpdf 7-model ensemble pipeline composition tests (20 tests):
/// Individual model subnetwork bounds (layout detection, OCR recognition,
/// table structure DETR, figure classification, reading order, document
/// classification), pairwise compositions (layout->OCR, layout->table,
/// layout->figure, OCR->reading order, table->doc classification),
/// full pipeline (sequential, CROWN tightened, parallel dispatch),
/// aggregation layer (weighted sum, monotone tightening, softmax gating),
/// document type specialization (text-heavy, table-heavy, figure-heavy).
/// Part of #4243.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_ensemble_pipeline.rs"]
mod ensemble_pipeline;

/// Compose verification tests for GLM-OCR pipeline stages (18 tests):
/// Patch embedding (conv2d + flatten + PE), visual encoder (self-attention,
/// RMSNorm contraction, 2-layer stack), text decoder (causal self-attention,
/// cross-attention to visual features, SwiGLU FFN, full block, short/long
/// sequence lengths), character classification head (softmax, log_softmax),
/// full end-to-end pipeline (small/large image), and layer norm through
/// transformer depth (3-layer stack with inter-layer RMSNorm).
/// Part of #4225.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/compose_dpdf_glm_ocr_pipeline.rs"]
mod glm_ocr_pipeline;

/// ay SMT proofs for attention mask and position bias properties (20 tests).
/// Causal mask upper/lower triangle, padding mask, combined causal+padding,
/// ALiBi linear bias and geometric slopes, sliding window bandwidth,
/// global attention for special tokens, cross-attention no self-masking,
/// mask dtype compatibility (bool vs float), mask broadcasting across
/// heads and batch, additive mask softmax zeroing, multiplicative mask,
/// prefix mask, block-sparse pattern, BigBird local+global, dynamic
/// variable-length mask, mask inversion complement, mask union/intersection.
/// Part of #4217.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_attention_mask_proofs.rs"]
mod attention_mask_proofs;

/// ay SMT proofs for quantization error mathematical bounds (20 proofs).
/// INT8 symmetric/asymmetric, GPTQ INT4, AWQ, BF16, MXFP4, SmoothQuant, etc.
/// Part of #4238.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_quantization_error_proofs.rs"]
mod quantization_error_proofs;

/// ay SMT proofs for tensor reshape and view mathematical properties (20 tests).
/// Reshape element count preservation, view stride offset identity, flatten,
/// squeeze/unsqueeze, permute, reshape invertibility, stride swap after transpose,
/// expand broadcast, narrow subrange, split/chunk, stack/cat, reshape bounds
/// preservation, view_as_complex/real, unfold/fold, diagonal, triu/tril.
/// Part of #4220.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_reshape_proofs.rs"]
mod reshape_proofs;

/// ay SMT proofs for matrix decomposition mathematical properties (20 tests):
/// SVD reconstruction/orthogonality/singular values, eigenvalue decomposition,
/// symmetric real eigenvalues, PSD non-negative eigenvalues, Cholesky (PSD +
/// lower triangular), trace/determinant via eigenvalues, Frobenius/spectral/
/// nuclear norms, low-rank factorization residual, rank product bound,
/// QR decomposition, Moore-Penrose pseudoinverse, truncated SVD error bound.
/// Part of #4235.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_matrix_decomp_proofs.rs"]
mod matrix_decomp_proofs;

/// ay SMT proofs for gradient computation mathematical correctness (20 tests,
/// test_1131 through test_1150): matmul backward (dL/dA, dL/dB), conv backward,
/// ReLU backward, softmax JVP, LayerNorm backward, add/mul backward, attention
/// backward, embedding scatter_add, cross-entropy p-y, chain rule, sum/mean
/// backward, transpose/reshape backward, concatenate backward, sigmoid backward,
/// tanh backward, GELU backward, SiLU backward identity.
/// Part of #4241.
#[allow(dead_code, unreachable_pub)]
#[path = "helpers/ay_dpdf_gradient_proofs.rs"]
mod gradient_proofs;
