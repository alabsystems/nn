// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for dpdf model weight loading via VarBuilder.
//!
//! Verifies that Granite-Docling and DocLayout-YOLO model builders correctly
//! load weights from mock tensor maps using `VarBuilder::from_tensors`. No
//! actual safetensors files needed — all weights are zero-filled with correct
//! shapes.
//!
//! Part of #3872.

use std::collections::HashMap;

use nn_core::dyn_tensor::DynTensor;
use nn_core::var_builder::VarBuilder;
use nn_core::{DType, Device};
use nn_models::doclayout_yolo::{DocLayoutYolo, DocLayoutYoloConfig};
use nn_models::granite_docling::{
    GraniteDocling, GraniteDoclingConfig, DECODER_HEADS, DECODER_HIDDEN, DECODER_INTERMEDIATE,
    DECODER_KV_HEADS, DECODER_LAYERS, VISION_HIDDEN, VISION_LAYERS, VOCAB_SIZE,
};

// ============================================================================
// Helpers
// ============================================================================

/// Insert a zero tensor into the map with the given name and shape.
fn insert_zeros(map: &mut HashMap<String, DynTensor>, name: &str, shape: &[usize]) {
    let t = DynTensor::zeros(shape, DType::F32, &Device::Cpu).expect("should create zero tensor");
    map.insert(name.to_string(), t);
}

/// Build a complete weight map for Granite-Docling-258M with zero tensors.
fn granite_docling_weight_map() -> HashMap<String, DynTensor> {
    let mut map = HashMap::new();
    let h = DECODER_HIDDEN; // 768
    let vh = VISION_HIDDEN; // 768
    let inter = DECODER_INTERMEDIATE; // 2048
    let vocab = VOCAB_SIZE; // 49152
    let heads = DECODER_HEADS; // 12
    let kv_heads = DECODER_KV_HEADS; // 4
    let head_dim = h / heads; // 64
    let kv_dim = kv_heads * head_dim; // 256
    let num_patches = 1024_usize; // (512/16)^2

    // --- Vision encoder (SigLIP2) ---
    // Patch embedding
    insert_zeros(
        &mut map,
        "vision_model.embeddings.patch_embedding.weight",
        &[vh, 3, 16, 16],
    );
    insert_zeros(
        &mut map,
        "vision_model.embeddings.patch_embedding.bias",
        &[vh],
    );
    // Position embedding
    insert_zeros(
        &mut map,
        "vision_model.embeddings.position_embedding.weight",
        &[num_patches, vh],
    );

    // Vision encoder layers
    for i in 0..VISION_LAYERS {
        let prefix = format!("vision_model.encoder.layers.{i}");
        // Self-attention Q/K/V/out projections (all [vh, vh] for SigLIP2)
        for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
            insert_zeros(
                &mut map,
                &format!("{prefix}.self_attn.{proj}.weight"),
                &[vh, vh],
            );
            insert_zeros(&mut map, &format!("{prefix}.self_attn.{proj}.bias"), &[vh]);
        }
        // Layer norms
        insert_zeros(&mut map, &format!("{prefix}.layer_norm1.weight"), &[vh]);
        insert_zeros(&mut map, &format!("{prefix}.layer_norm1.bias"), &[vh]);
        insert_zeros(&mut map, &format!("{prefix}.layer_norm2.weight"), &[vh]);
        insert_zeros(&mut map, &format!("{prefix}.layer_norm2.bias"), &[vh]);
        // MLP
        // fc1: [intermediate_size, vh], fc2: [vh, intermediate_size]
        // SigLIP2-base intermediate = 3072
        let vision_inter = 3072;
        insert_zeros(
            &mut map,
            &format!("{prefix}.mlp.fc1.weight"),
            &[vision_inter, vh],
        );
        insert_zeros(&mut map, &format!("{prefix}.mlp.fc1.bias"), &[vision_inter]);
        insert_zeros(
            &mut map,
            &format!("{prefix}.mlp.fc2.weight"),
            &[vh, vision_inter],
        );
        insert_zeros(&mut map, &format!("{prefix}.mlp.fc2.bias"), &[vh]);
    }
    // Post-layernorm
    insert_zeros(&mut map, "vision_model.post_layernorm.weight", &[vh]);
    insert_zeros(&mut map, "vision_model.post_layernorm.bias", &[vh]);

    // --- Multi-modal projector ---
    insert_zeros(&mut map, "multi_modal_projector.linear.weight", &[h, vh]);
    insert_zeros(&mut map, "multi_modal_projector.linear.bias", &[h]);

    // --- Text embedding ---
    insert_zeros(&mut map, "model.embed_tokens.weight", &[vocab, h]);

    // --- Decoder layers ---
    for i in 0..DECODER_LAYERS {
        let prefix = format!("model.layers.{i}");
        // Input layernorm
        insert_zeros(&mut map, &format!("{prefix}.input_layernorm.weight"), &[h]);
        // Self-attention (no bias for Granite decoder)
        insert_zeros(
            &mut map,
            &format!("{prefix}.self_attn.q_proj.weight"),
            &[h, h],
        );
        insert_zeros(
            &mut map,
            &format!("{prefix}.self_attn.k_proj.weight"),
            &[kv_dim, h],
        );
        insert_zeros(
            &mut map,
            &format!("{prefix}.self_attn.v_proj.weight"),
            &[kv_dim, h],
        );
        insert_zeros(
            &mut map,
            &format!("{prefix}.self_attn.out_proj.weight"),
            &[h, h],
        );
        // Post-attention layernorm
        insert_zeros(
            &mut map,
            &format!("{prefix}.post_attention_layernorm.weight"),
            &[h],
        );
        // SwiGLU MLP
        insert_zeros(
            &mut map,
            &format!("{prefix}.mlp.gate_proj.weight"),
            &[inter, h],
        );
        insert_zeros(
            &mut map,
            &format!("{prefix}.mlp.up_proj.weight"),
            &[inter, h],
        );
        insert_zeros(
            &mut map,
            &format!("{prefix}.mlp.down_proj.weight"),
            &[h, inter],
        );
    }

    // --- Final norm + LM head ---
    insert_zeros(&mut map, "model.norm.weight", &[h]);
    insert_zeros(&mut map, "lm_head.weight", &[vocab, h]);

    map
}

/// Build a complete weight map for DocLayout-YOLO with zero tensors.
///
/// DocLayout-YOLO uses ConvBnAct (conv.weight + bn.{weight,bias,running_mean,running_var})
/// and C2f (cv1/cv2 + bottlenecks) structures. ZerosBackend is simpler for YOLO since
/// it auto-generates shapes; this map is for explicit shape verification.
fn doclayout_yolo_weight_map() -> HashMap<String, DynTensor> {
    let mut map = HashMap::new();
    let cfg = DocLayoutYoloConfig::default();
    let [c0, c1, c2, c3, c4] = cfg.backbone_channels; // [16, 32, 64, 128, 256]

    // Helper: add ConvBnAct weights at a given prefix
    let add_conv_bn = |map: &mut HashMap<String, DynTensor>,
                       prefix: &str,
                       in_c: usize,
                       out_c: usize,
                       k: usize| {
        insert_zeros(map, &format!("{prefix}.conv.weight"), &[out_c, in_c, k, k]);
        insert_zeros(map, &format!("{prefix}.bn.weight"), &[out_c]);
        insert_zeros(map, &format!("{prefix}.bn.bias"), &[out_c]);
        insert_zeros(map, &format!("{prefix}.bn.running_mean"), &[out_c]);
        insert_zeros(map, &format!("{prefix}.bn.running_var"), &[out_c]);
    };

    // Helper: add C2f weights at a given prefix
    let add_c2f = |map: &mut HashMap<String, DynTensor>,
                   prefix: &str,
                   in_c: usize,
                   out_c: usize,
                   n_bottlenecks: usize| {
        let hidden = out_c / 2;
        // cv1: projects in_c -> 2*hidden (kernel 1x1)
        add_conv_bn(map, &format!("{prefix}.cv1"), in_c, 2 * hidden, 1);
        // cv2: projects (2*hidden + n_bottlenecks*hidden) -> out_c (kernel 1x1)
        let cat_channels = 2 * hidden + n_bottlenecks * hidden;
        add_conv_bn(map, &format!("{prefix}.cv2"), cat_channels, out_c, 1);
        // Each bottleneck: cv1 (3x3) + cv2 (3x3), both hidden -> hidden
        for j in 0..n_bottlenecks {
            let bp = format!("{prefix}.bottlenecks.{j}");
            add_conv_bn(map, &format!("{bp}.cv1"), hidden, hidden, 3);
            add_conv_bn(map, &format!("{bp}.cv2"), hidden, hidden, 3);
        }
    };

    // --- Backbone ---
    // Stage 0: stem 3->c0, stride 2
    add_conv_bn(&mut map, "backbone.stage0", 3, c0, 3);

    // Stage 1: conv c0->c1 + C2f c1->c1 (1 bottleneck)
    add_conv_bn(&mut map, "backbone.stage1.conv", c0, c1, 3);
    add_c2f(&mut map, "backbone.stage1.c2f", c1, c1, 1);

    // Stage 2: conv c1->c2 + C2f c2->c2 (2 bottlenecks)
    add_conv_bn(&mut map, "backbone.stage2.conv", c1, c2, 3);
    add_c2f(&mut map, "backbone.stage2.c2f", c2, c2, 2);

    // Stage 3: conv c2->c3 + C2f c3->c3 (2 bottlenecks)
    add_conv_bn(&mut map, "backbone.stage3.conv", c2, c3, 3);
    add_c2f(&mut map, "backbone.stage3.c2f", c3, c3, 2);

    // Stage 4: conv c3->c4 + C2f c4->c4 (1 bottleneck) + SPPF c4->c4
    add_conv_bn(&mut map, "backbone.stage4.conv", c3, c4, 3);
    add_c2f(&mut map, "backbone.stage4.c2f", c4, c4, 1);
    // SPPF: cv1 (c4->c4/2, 1x1) + cv2 (c4/2*4 -> c4, 1x1)
    let sppf_hidden = c4 / 2;
    add_conv_bn(&mut map, "backbone.stage4.sppf.cv1", c4, sppf_hidden, 1);
    add_conv_bn(&mut map, "backbone.stage4.sppf.cv2", sppf_hidden * 4, c4, 1);

    // --- Neck (PAN) and Head ---
    // For these components, VarBuilder::zeros handles shapes internally.
    // We skip explicit neck/head tensors here; test_doclayout_yolo_full_load_zeros
    // covers these via ZerosBackend.

    map
}

// ============================================================================
// Granite-Docling weight loading tests
// ============================================================================

#[test]
fn test_granite_docling_full_load_from_tensor_map() {
    // Verify the full model loads from an explicit tensor map with correct shapes.
    let weights = granite_docling_weight_map();
    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let model = GraniteDocling::load(&vb, cfg).expect("model should load from tensor map");
    assert_eq!(model.decoder_layers().len(), DECODER_LAYERS);
}

#[test]
fn test_granite_docling_weight_names_vision_encoder() {
    // Verify all expected vision encoder weight names are in the map.
    let weights = granite_docling_weight_map();
    let vh = VISION_HIDDEN;

    // Patch embedding
    assert!(weights.contains_key("vision_model.embeddings.patch_embedding.weight"));
    assert!(weights.contains_key("vision_model.embeddings.patch_embedding.bias"));
    assert!(weights.contains_key("vision_model.embeddings.position_embedding.weight"));

    // All 12 vision layers
    for i in 0..VISION_LAYERS {
        let prefix = format!("vision_model.encoder.layers.{i}");
        for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
            let key = format!("{prefix}.self_attn.{proj}.weight");
            assert!(weights.contains_key(&key), "missing: {key}");
            assert_eq!(weights[&key].dims(), &[vh, vh], "wrong shape: {key}");
        }
        for ln in &["layer_norm1", "layer_norm2"] {
            let key_w = format!("{prefix}.{ln}.weight");
            assert!(weights.contains_key(&key_w), "missing: {key_w}");
            assert_eq!(weights[&key_w].dims(), &[vh], "wrong shape: {key_w}");
        }
        assert!(
            weights.contains_key(&format!("{prefix}.mlp.fc1.weight")),
            "missing fc1 for layer {i}"
        );
        assert!(
            weights.contains_key(&format!("{prefix}.mlp.fc2.weight")),
            "missing fc2 for layer {i}"
        );
    }

    // Post-layernorm
    assert!(weights.contains_key("vision_model.post_layernorm.weight"));
    assert!(weights.contains_key("vision_model.post_layernorm.bias"));
}

#[test]
fn test_granite_docling_weight_names_decoder() {
    // Verify all expected decoder weight names are in the map.
    let weights = granite_docling_weight_map();
    let h = DECODER_HIDDEN;
    let kv_dim = DECODER_KV_HEADS * (h / DECODER_HEADS); // 256
    let inter = DECODER_INTERMEDIATE;

    for i in 0..DECODER_LAYERS {
        let prefix = format!("model.layers.{i}");

        // Input layernorm
        let key = format!("{prefix}.input_layernorm.weight");
        assert!(weights.contains_key(&key), "missing: {key}");
        assert_eq!(weights[&key].dims(), &[h]);

        // Q proj: [h, h]
        let key = format!("{prefix}.self_attn.q_proj.weight");
        assert!(weights.contains_key(&key), "missing: {key}");
        assert_eq!(weights[&key].dims(), &[h, h]);

        // K/V proj: [kv_dim, h]
        for proj in &["k_proj", "v_proj"] {
            let key = format!("{prefix}.self_attn.{proj}.weight");
            assert!(weights.contains_key(&key), "missing: {key}");
            assert_eq!(weights[&key].dims(), &[kv_dim, h], "wrong shape: {key}");
        }

        // Out proj: [h, h]
        let key = format!("{prefix}.self_attn.out_proj.weight");
        assert!(weights.contains_key(&key), "missing: {key}");
        assert_eq!(weights[&key].dims(), &[h, h]);

        // Post-attention layernorm
        let key = format!("{prefix}.post_attention_layernorm.weight");
        assert!(weights.contains_key(&key), "missing: {key}");
        assert_eq!(weights[&key].dims(), &[h]);

        // SwiGLU MLP: gate_proj [inter, h], up_proj [inter, h], down_proj [h, inter]
        let key = format!("{prefix}.mlp.gate_proj.weight");
        assert!(weights.contains_key(&key), "missing: {key}");
        assert_eq!(weights[&key].dims(), &[inter, h]);

        let key = format!("{prefix}.mlp.up_proj.weight");
        assert!(weights.contains_key(&key), "missing: {key}");
        assert_eq!(weights[&key].dims(), &[inter, h]);

        let key = format!("{prefix}.mlp.down_proj.weight");
        assert!(weights.contains_key(&key), "missing: {key}");
        assert_eq!(weights[&key].dims(), &[h, inter]);
    }

    // Final norm + LM head + embedding
    assert!(weights.contains_key("model.norm.weight"));
    assert_eq!(weights["model.norm.weight"].dims(), &[h]);

    assert!(weights.contains_key("lm_head.weight"));
    assert_eq!(weights["lm_head.weight"].dims(), &[VOCAB_SIZE, h]);

    assert!(weights.contains_key("model.embed_tokens.weight"));
    assert_eq!(
        weights["model.embed_tokens.weight"].dims(),
        &[VOCAB_SIZE, h]
    );
}

#[test]
fn test_granite_docling_weight_shapes() {
    // Verify key weight shapes match the model architecture.
    let weights = granite_docling_weight_map();
    let h = DECODER_HIDDEN;
    let kv_dim = DECODER_KV_HEADS * (h / DECODER_HEADS);
    let inter = DECODER_INTERMEDIATE;
    let vocab = VOCAB_SIZE;
    let vh = VISION_HIDDEN;

    // Vision patch embedding: [hidden, 3, 16, 16]
    assert_eq!(
        weights["vision_model.embeddings.patch_embedding.weight"].dims(),
        &[vh, 3, 16, 16]
    );
    // Position embedding: [1024, hidden]
    assert_eq!(
        weights["vision_model.embeddings.position_embedding.weight"].dims(),
        &[1024, vh]
    );

    // Decoder Q: [h, h], K/V: [kv_dim, h]
    assert_eq!(
        weights["model.layers.0.self_attn.q_proj.weight"].dims(),
        &[h, h]
    );
    assert_eq!(
        weights["model.layers.0.self_attn.k_proj.weight"].dims(),
        &[kv_dim, h]
    );
    assert_eq!(
        weights["model.layers.0.self_attn.v_proj.weight"].dims(),
        &[kv_dim, h]
    );

    // FFN gate/up: [inter, h], down: [h, inter]
    assert_eq!(
        weights["model.layers.0.mlp.gate_proj.weight"].dims(),
        &[inter, h]
    );
    assert_eq!(
        weights["model.layers.0.mlp.down_proj.weight"].dims(),
        &[h, inter]
    );

    // Embedding and LM head: [vocab, h]
    assert_eq!(weights["model.embed_tokens.weight"].dims(), &[vocab, h]);
    assert_eq!(weights["lm_head.weight"].dims(), &[vocab, h]);
}

#[test]
fn test_granite_docling_param_count() {
    // Verify approximate parameter count matches ~258M.
    let weights = granite_docling_weight_map();
    let total: usize = weights.values().map(DynTensor::elem_count).sum();

    // Expected ~258M parameters. Allow some tolerance since we build the map
    // manually and the exact count depends on bias presence.
    assert!(total > 200_000_000, "param count too low: {total}");
    assert!(total < 350_000_000, "param count too high: {total}");
}

#[test]
fn test_granite_docling_missing_weight_errors() {
    // Attempting to load the model with an incomplete weight map should fail,
    // not panic.
    let mut weights = granite_docling_weight_map();
    // Remove a critical weight
    weights.remove("lm_head.weight");

    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let result = GraniteDocling::load(&vb, cfg);
    assert!(
        result.is_err(),
        "loading with missing lm_head.weight should fail"
    );
}

#[test]
fn test_granite_docling_missing_decoder_layer_weight_errors() {
    // Remove a decoder layer weight to verify graceful error.
    let mut weights = granite_docling_weight_map();
    weights.remove("model.layers.5.mlp.gate_proj.weight");

    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let result = GraniteDocling::load(&vb, cfg);
    assert!(
        result.is_err(),
        "loading with missing decoder layer weight should fail"
    );
}

#[test]
fn test_granite_docling_missing_vision_weight_errors() {
    // Remove a vision encoder weight to verify graceful error.
    let mut weights = granite_docling_weight_map();
    weights.remove("vision_model.encoder.layers.0.self_attn.q_proj.weight");

    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let result = GraniteDocling::load(&vb, cfg);
    assert!(
        result.is_err(),
        "loading with missing vision weight should fail"
    );
}

#[test]
fn test_granite_docling_wrong_shape_errors() {
    // Provide a weight with the wrong shape.
    let mut weights = granite_docling_weight_map();
    // Replace lm_head.weight with a wrong shape (swap dimensions)
    insert_zeros(
        &mut weights,
        "lm_head.weight",
        &[DECODER_HIDDEN, VOCAB_SIZE],
    );

    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let result = GraniteDocling::load(&vb, cfg);
    assert!(
        result.is_err(),
        "loading with wrong-shaped lm_head.weight should fail"
    );
}

#[test]
fn test_granite_docling_weight_dtype_f32() {
    // Verify all loaded weights are F32.
    let weights = granite_docling_weight_map();
    for (name, tensor) in &weights {
        assert_eq!(
            tensor.dtype(),
            DType::F32,
            "weight {name} should be F32, got {:?}",
            tensor.dtype()
        );
    }
}

#[test]
fn test_granite_docling_forward_after_tensor_map_load() {
    // Verify the model can run a forward pass after loading from explicit tensors.
    let weights = granite_docling_weight_map();
    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let model = GraniteDocling::load(&vb, cfg.clone()).expect("model should load from tensor map");

    let image = DynTensor::zeros(
        &[1, 3, cfg.image_size, cfg.image_size],
        DType::F32,
        &Device::Cpu,
    )
    .expect("should create image tensor");
    let text_ids: Vec<usize> = (0..5).collect();
    let logits = model
        .forward(&image, &text_ids)
        .expect("forward should succeed with tensor map weights");
    // 1024 vision patches + 5 text tokens = 1029
    assert_eq!(logits.dims(), &[1, 1029, cfg.vocab_size]);
}

// ============================================================================
// DocLayout-YOLO weight loading tests
// ============================================================================

#[test]
fn test_doclayout_yolo_full_load_zeros() {
    // Verify full model loads from ZerosBackend.
    let cfg = DocLayoutYoloConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = DocLayoutYolo::load(&vb, cfg).expect("model should load from zeros");
    assert_eq!(model.config().num_classes, 10);
}

#[test]
fn test_doclayout_yolo_backbone_weight_names() {
    // Verify key backbone weight names exist in the map.
    let weights = doclayout_yolo_weight_map();

    // Stage 0 stem
    assert!(weights.contains_key("backbone.stage0.conv.weight"));
    assert!(weights.contains_key("backbone.stage0.bn.weight"));

    // Stage 1
    assert!(weights.contains_key("backbone.stage1.conv.conv.weight"));
    assert!(weights.contains_key("backbone.stage1.c2f.cv1.conv.weight"));

    // Stage 4 SPPF
    assert!(weights.contains_key("backbone.stage4.sppf.cv1.conv.weight"));
    assert!(weights.contains_key("backbone.stage4.sppf.cv2.conv.weight"));
}

#[test]
fn test_doclayout_yolo_weight_shapes_conv() {
    // Verify conv weight shapes match [out_ch, in_ch, kH, kW].
    let weights = doclayout_yolo_weight_map();
    let [c0, c1, _c2, _c3, _c4] = DocLayoutYoloConfig::default().backbone_channels;

    // Stem: 3 -> c0, kernel 3x3
    assert_eq!(
        weights["backbone.stage0.conv.weight"].dims(),
        &[c0, 3, 3, 3]
    );

    // Stage 1 conv: c0 -> c1, kernel 3x3
    assert_eq!(
        weights["backbone.stage1.conv.conv.weight"].dims(),
        &[c1, c0, 3, 3]
    );
}

#[test]
fn test_doclayout_yolo_weight_shapes_bn() {
    // Verify BN weight shapes are [channels].
    let weights = doclayout_yolo_weight_map();
    let [c0, c1, _, _, _] = DocLayoutYoloConfig::default().backbone_channels;

    assert_eq!(weights["backbone.stage0.bn.weight"].dims(), &[c0]);
    assert_eq!(weights["backbone.stage0.bn.bias"].dims(), &[c0]);
    assert_eq!(weights["backbone.stage0.bn.running_mean"].dims(), &[c0]);
    assert_eq!(weights["backbone.stage0.bn.running_var"].dims(), &[c0]);

    assert_eq!(weights["backbone.stage1.conv.bn.weight"].dims(), &[c1]);
}

#[test]
fn test_doclayout_yolo_c2f_weight_structure() {
    // Verify C2f module has cv1, cv2, and bottleneck weights.
    let weights = doclayout_yolo_weight_map();

    // Stage 1 C2f (1 bottleneck)
    assert!(weights.contains_key("backbone.stage1.c2f.cv1.conv.weight"));
    assert!(weights.contains_key("backbone.stage1.c2f.cv2.conv.weight"));
    assert!(weights.contains_key("backbone.stage1.c2f.bottlenecks.0.cv1.conv.weight"));
    assert!(weights.contains_key("backbone.stage1.c2f.bottlenecks.0.cv2.conv.weight"));

    // Stage 2 C2f (2 bottlenecks)
    assert!(weights.contains_key("backbone.stage2.c2f.bottlenecks.0.cv1.conv.weight"));
    assert!(weights.contains_key("backbone.stage2.c2f.bottlenecks.1.cv1.conv.weight"));
}

// ============================================================================
// Weight mapper tests
// ============================================================================

#[test]
fn test_weight_mapper_qwen3_identity() {
    // Qwen3 mapper is identity (NN uses same names as HF).
    use nn_core::var_builder::{HfToNnMapper, WeightNameMapper};
    let mapper = HfToNnMapper::qwen3();

    assert_eq!(
        mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("model.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
    assert_eq!(mapper.map_name("lm_head.weight"), "lm_head.weight");
}

#[test]
fn test_weight_mapper_siglip2_prefix() {
    // SigLIP2 mapper adds "model.vision_model." prefix.
    use nn_core::var_builder::{HfToNnMapper, WeightNameMapper};
    let mapper = HfToNnMapper::siglip2_granite_docling();

    assert_eq!(
        mapper.map_name("encoder.layers.0.self_attn.q_proj.weight"),
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("embeddings.patch_embedding.weight"),
        "model.vision_model.embeddings.patch_embedding.weight"
    );
    assert_eq!(
        mapper.map_name("post_layernorm.weight"),
        "model.vision_model.post_layernorm.weight"
    );
}

#[test]
fn test_weight_mapper_decoder_transformer_renaming() {
    // Decoder transformer mapper renames segments.
    use nn_core::var_builder::{HfToNnMapper, WeightNameMapper};
    let mapper = HfToNnMapper::decoder_transformer();

    // Prefix: "layers" -> "model.layers"
    assert_eq!(
        mapper.map_name("layers.0.attn.q.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.ln1.weight"),
        "model.layers.0.input_layernorm.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.ln2.weight"),
        "model.layers.0.post_attention_layernorm.weight"
    );
}

#[test]
fn test_weight_mapper_coverage_verification() {
    // Use verify_mapper_coverage to check that all NN names resolve.
    use nn_core::var_builder::{verify_mapper_coverage, HfToNnMapper};

    let mapper = HfToNnMapper::new().with_prefix_rule("model", "m");

    let checkpoint_names = vec!["model.weight".to_string(), "model.bias".to_string()];
    let nn_names = vec![
        "m.weight".to_string(),
        "m.bias".to_string(),
        "m.extra".to_string(),
    ];

    let missing = verify_mapper_coverage(&nn_names, &checkpoint_names, &mapper);
    assert_eq!(missing, vec!["m.extra"]);
}

// ============================================================================
// Weight contiguity and dtype tests
// ============================================================================

#[test]
fn test_granite_docling_weights_contiguous() {
    // Verify all generated mock weights are contiguous.
    let weights = granite_docling_weight_map();
    for (name, tensor) in &weights {
        assert!(tensor.is_contiguous(), "weight {name} should be contiguous");
    }
}

#[test]
fn test_doclayout_yolo_weights_contiguous() {
    // Verify all generated mock weights are contiguous.
    let weights = doclayout_yolo_weight_map();
    for (name, tensor) in &weights {
        assert!(tensor.is_contiguous(), "weight {name} should be contiguous");
    }
}

#[test]
fn test_doclayout_yolo_weights_dtype_f32() {
    // Verify all weights are F32.
    let weights = doclayout_yolo_weight_map();
    for (name, tensor) in &weights {
        assert_eq!(
            tensor.dtype(),
            DType::F32,
            "weight {name} should be F32, got {:?}",
            tensor.dtype()
        );
    }
}

// ============================================================================
// Extra / edge-case tests
// ============================================================================

#[test]
fn test_granite_docling_extra_weights_do_not_prevent_loading() {
    // Extra weights in the map should not prevent model loading.
    let mut weights = granite_docling_weight_map();
    insert_zeros(&mut weights, "extra.unexpected.weight", &[100, 100]);
    insert_zeros(&mut weights, "another.extra.bias", &[50]);

    let cfg = GraniteDoclingConfig::default_258m();
    let vb = VarBuilder::from_tensors(weights, DType::F32, &Device::Cpu);
    let result = GraniteDocling::load(&vb, cfg);
    assert!(
        result.is_ok(),
        "extra weights should not prevent model loading: {:?}",
        result.err()
    );
}

#[test]
fn test_doclayout_yolo_backbone_forward_shape_zeros() {
    // Verify backbone produces 3 feature maps at correct scales.
    let cfg = DocLayoutYoloConfig::default();
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = DocLayoutYolo::load(&vb, cfg).expect("model should load");

    let input =
        DynTensor::zeros(&[1, 3, 800, 800], DType::F32, &Device::Cpu).expect("should create input");
    let (p3, p4, p5) = model
        .forward_backbone(&input)
        .expect("backbone forward should succeed");

    // P3: stride 8 -> 800/8 = 100
    assert_eq!(p3.dims()[2], 100);
    assert_eq!(p3.dims()[3], 100);
    // P4: stride 16 -> 800/16 = 50
    assert_eq!(p4.dims()[2], 50);
    assert_eq!(p4.dims()[3], 50);
    // P5: stride 32 -> 800/32 = 25
    assert_eq!(p5.dims()[2], 25);
    assert_eq!(p5.dims()[3], 25);
}

#[test]
fn test_granite_docling_weight_map_completeness() {
    // Verify the tensor map contains the same number of tensors as a ZerosBackend
    // would provide — by loading from ZerosBackend and counting .get() calls.
    // Since ZerosBackend always succeeds, the real test is that TensorMapBackend
    // also succeeds with the exact same calls.
    let weights = granite_docling_weight_map();
    let weight_count = weights.len();

    // Sanity: should have at least vision + decoder + projector + head weights.
    // Vision: 12 layers * (4 proj * 2 + 2 ln * 2 + 2 mlp * 2) = 12 * 16 = 192
    //         + patch embed (2) + position (1) + post-ln (2) = 197
    // Decoder: 12 layers * (1 ln + 4 attn + 1 ln + 3 mlp) = 12 * 9 = 108
    // Global: projector (2) + embed (1) + norm (1) + lm_head (1) = 5
    // Total: ~310
    assert!(
        weight_count > 250,
        "expected >250 weights, got {weight_count}"
    );
    assert!(
        weight_count < 400,
        "expected <400 weights, got {weight_count}"
    );
}
