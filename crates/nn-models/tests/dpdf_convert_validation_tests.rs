// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exhaustive validation tests for convert_dpdf weight key mapping functions.
//!
//! Covers every `map_*_key()` mapper: known key patterns, unknown key rejection,
//! bias vs weight keys, adversarial inputs, layer index extraction, and
//! collision detection (no two HF keys may map to the same nn key).
//!
//! Part of #3926.

use std::collections::{HashMap, HashSet};

use nn_models::convert::{map_weight_key, DpdfModelType};

// ============================================================================
// Helpers
// ============================================================================

/// Assert that every key in `hf_keys` produces a `Some` result for the given model.
fn assert_all_mapped(model: &DpdfModelType, hf_keys: &[&str]) {
    for key in hf_keys {
        let result = map_weight_key(model, key);
        assert!(
            result.is_some(),
            "Expected Some for key '{key}' with model {model:?}, got None"
        );
    }
}

/// Assert that every key in `hf_keys` produces `None` for the given model.
fn assert_all_none(model: &DpdfModelType, hf_keys: &[&str]) {
    for key in hf_keys {
        let result = map_weight_key(model, key);
        assert!(
            result.is_none(),
            "Expected None for key '{key}' with model {model:?}, got {result:?}"
        );
    }
}

/// Collision detection: given a set of HF keys, assert no two map to the same nn key.
fn assert_no_collisions(model: &DpdfModelType, hf_keys: &[&str]) {
    let mut seen: HashMap<String, &str> = HashMap::new();
    for key in hf_keys {
        if let Some(mapped) = map_weight_key(model, key) {
            if let Some(prev) = seen.get(&mapped) {
                panic!(
                    "Collision for model {model:?}: HF keys '{prev}' and '{key}' both map to '{mapped}'"
                );
            }
            seen.insert(mapped, key);
        }
    }
}

// ============================================================================
// Adversarial inputs: all models must return None and not panic
// ============================================================================

const ADVERSARIAL_INPUTS: &[&str] = &[
    "",
    " ",
    ".",
    "..",
    "...",
    "a",
    "0",
    "12345",
    "model",
    "model.",
    ".model",
    "model..",
    "a.b.c.d.e.f.g.h.i.j.k.l.m.n.o.p",
    "\0",
    "\n\t\r",
    "model.layers.NaN.self_attn.o_proj.weight",
    "model.layers.-1.self_attn.o_proj.weight",
    "model.layers.999999999999999999999.self_attn.o_proj.weight",
    "model.layers.0.self_attn.o_proj.weight.extra.trailing.stuff",
    "UPPER.CASE.KEY",
    "model.layers.0.self_attn.",
    "model.layers.0.",
    "model.layers.",
    "model.layers",
    "Student",
    "Student.",
    "Student2",
    "Student2.",
    "visual",
    "visual.",
    "vision_model",
    "vision_model.",
    "multi_modal_projector",
    "multi_modal_projector.",
    // Unicode stress
    "\u{FEFF}model.layers.0.weight",
    "model.\u{200B}layers.0.weight",
];

/// Lazily generate the very long key since const &str can't do heap alloc.
fn adversarial_inputs_with_long() -> Vec<String> {
    let mut inputs: Vec<String> = ADVERSARIAL_INPUTS.iter().map(ToString::to_string).collect();
    inputs.push("a".repeat(10_000));
    inputs
}

fn all_model_types() -> Vec<DpdfModelType> {
    vec![
        DpdfModelType::GraniteDocling,
        DpdfModelType::DocLayoutYolo,
        DpdfModelType::Qwen3VL,
        DpdfModelType::TableTransformer,
        DpdfModelType::GlmOcr,
        DpdfModelType::PaddleOcr,
        DpdfModelType::FireRedOcr,
    ]
}

#[test]
fn test_adversarial_inputs_no_panic_all_models() {
    let inputs = adversarial_inputs_with_long();
    for model in all_model_types() {
        for input in &inputs {
            // Must not panic. Result can be Some or None.
            let _ = map_weight_key(&model, input);
        }
    }
}

// ============================================================================
// Granite-Docling exhaustive tests
// ============================================================================

#[test]
fn test_granite_docling_vision_encoder_keys() {
    let model = DpdfModelType::GraniteDocling;
    let keys = &[
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_model.encoder.layers.0.self_attn.q_proj.bias",
        "vision_model.encoder.layers.0.self_attn.k_proj.weight",
        "vision_model.encoder.layers.0.self_attn.k_proj.bias",
        "vision_model.encoder.layers.0.self_attn.v_proj.weight",
        "vision_model.encoder.layers.0.self_attn.v_proj.bias",
        "vision_model.encoder.layers.0.self_attn.out_proj.weight",
        "vision_model.encoder.layers.0.self_attn.out_proj.bias",
        "vision_model.encoder.layers.11.self_attn.q_proj.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.position_embedding.weight",
        "vision_model.layernorm.weight",
        "vision_model.layernorm.bias",
    ];
    assert_all_mapped(&model, keys);

    // Vision keys are passthrough: output == input
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "Vision key should pass through unchanged: {key}"
        );
    }
}

#[test]
fn test_granite_docling_decoder_layer_patterns() {
    let model = DpdfModelType::GraniteDocling;
    // o_proj -> out_proj remapping
    for layer_idx in [0, 1, 5, 23] {
        let hf = format!("model.layers.{layer_idx}.self_attn.o_proj.weight");
        let expected = format!("model.layers.{layer_idx}.self_attn.out_proj.weight");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "o_proj -> out_proj remap failed for layer {layer_idx}"
        );

        let hf_bias = format!("model.layers.{layer_idx}.self_attn.o_proj.bias");
        let expected_bias = format!("model.layers.{layer_idx}.self_attn.out_proj.bias");
        assert_eq!(
            map_weight_key(&model, &hf_bias).as_deref(),
            Some(expected_bias.as_str()),
            "o_proj bias remap failed for layer {layer_idx}"
        );
    }

    // Non-o_proj keys pass through
    let passthrough_keys = &[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ];
    for key in passthrough_keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "Decoder passthrough failed for: {key}"
        );
    }
}

#[test]
fn test_granite_docling_top_level_passthrough() {
    let model = DpdfModelType::GraniteDocling;
    let keys = &[
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.norm.bias",
        "lm_head.weight",
        "lm_head.bias",
        "multi_modal_projector.linear.weight",
        "multi_modal_projector.linear.bias",
    ];
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "Top-level passthrough failed for: {key}"
        );
    }
}

#[test]
fn test_granite_docling_unknown_returns_none() {
    let model = DpdfModelType::GraniteDocling;
    assert_all_none(
        &model,
        &[
            "unknown.foo.bar",
            "encoder.layers.0.weight",
            "decoder.layers.0.weight",
            "backbone.stage0.conv.weight",
        ],
    );
}

#[test]
fn test_granite_docling_collision_detection() {
    let model = DpdfModelType::GraniteDocling;
    let keys: Vec<&str> = vec![
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_model.encoder.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "multi_modal_projector.linear.weight",
        "multi_modal_projector.linear.bias",
    ];
    assert_no_collisions(&model, &keys);
}

// ============================================================================
// DocLayout-YOLO exhaustive tests
// ============================================================================

#[test]
fn test_doclayout_yolo_all_backbone_indices() {
    let model = DpdfModelType::DocLayoutYolo;

    let expected_prefixes = [
        (0, "backbone.stage0."),
        (1, "backbone.stage1.conv."),
        (2, "backbone.stage1.c2f."),
        (3, "backbone.stage2.conv."),
        (4, "backbone.stage2.c2f."),
        (5, "backbone.stage3.conv."),
        (6, "backbone.stage3.c2f."),
        (7, "backbone.stage4.conv."),
        (8, "backbone.stage4.c2f."),
        (9, "backbone.stage4.sppf."),
    ];

    for (idx, expected_prefix) in &expected_prefixes {
        let hf = format!("model.{idx}.conv.weight");
        let mapped = map_weight_key(&model, &hf);
        let mapped_str = mapped.as_deref().unwrap_or("None");
        assert!(
            mapped_str.starts_with(expected_prefix),
            "Index {idx}: expected prefix '{expected_prefix}', got '{mapped_str}'"
        );
    }
}

#[test]
fn test_doclayout_yolo_all_neck_indices() {
    let model = DpdfModelType::DocLayoutYolo;
    // Neck indices 10-23
    for idx in 10..=23 {
        let hf = format!("model.{idx}.conv.weight");
        let expected = format!("neck.{}.conv.weight", idx - 10);
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "Neck index {idx} mapping failed"
        );
    }
}

#[test]
fn test_doclayout_yolo_head() {
    let model = DpdfModelType::DocLayoutYolo;
    let keys = &[
        "model.24.cls.0.weight",
        "model.24.cls.0.bias",
        "model.24.reg.0.weight",
        "model.24.dfl.conv.weight",
    ];
    for key in keys {
        let mapped = map_weight_key(&model, key);
        assert!(mapped.is_some(), "Head key should map: {key}");
        assert!(
            mapped.as_deref().unwrap().starts_with("head."),
            "Head key should start with 'head.': {key} -> {mapped:?}"
        );
    }
}

#[test]
fn test_doclayout_yolo_out_of_range() {
    let model = DpdfModelType::DocLayoutYolo;
    assert_all_none(
        &model,
        &[
            "model.25.conv.weight",
            "model.30.conv.weight",
            "model.100.conv.weight",
            "model.999.conv.weight",
        ],
    );
}

#[test]
fn test_doclayout_yolo_not_model_prefix() {
    let model = DpdfModelType::DocLayoutYolo;
    assert_all_none(
        &model,
        &[
            "backbone.stage0.conv.weight",
            "neck.0.conv.weight",
            "head.cls.weight",
            "0.conv.weight",
        ],
    );
}

#[test]
fn test_doclayout_yolo_non_numeric_index() {
    let model = DpdfModelType::DocLayoutYolo;
    assert_all_none(
        &model,
        &[
            "model.abc.conv.weight",
            "model.NaN.conv.weight",
            "model..conv.weight",
        ],
    );
}

#[test]
fn test_doclayout_yolo_collision_detection() {
    let model = DpdfModelType::DocLayoutYolo;
    let mut keys = Vec::new();
    for idx in 0..=24 {
        keys.push(format!("model.{idx}.conv.weight"));
        keys.push(format!("model.{idx}.conv.bias"));
    }
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    assert_no_collisions(&model, &key_refs);
}

#[test]
fn test_doclayout_yolo_weight_vs_bias() {
    let model = DpdfModelType::DocLayoutYolo;
    let weight = map_weight_key(&model, "model.0.conv.weight");
    let bias = map_weight_key(&model, "model.0.conv.bias");
    assert_eq!(weight.as_deref(), Some("backbone.stage0.conv.weight"));
    assert_eq!(bias.as_deref(), Some("backbone.stage0.conv.bias"));
    assert_ne!(weight, bias);
}

// ============================================================================
// Qwen3-VL exhaustive tests
// ============================================================================

#[test]
fn test_qwen3_vl_visual_keys() {
    let model = DpdfModelType::Qwen3VL;
    let keys = &[
        "visual.patch_embed.proj.weight",
        "visual.patch_embed.proj.bias",
        "visual.blocks.0.attn.qkv.weight",
        "visual.blocks.0.attn.qkv.bias",
        "visual.blocks.0.attn.proj.weight",
        "visual.blocks.0.attn.proj.bias",
        "visual.blocks.0.mlp.fc1.weight",
        "visual.blocks.0.mlp.fc1.bias",
        "visual.blocks.0.mlp.fc2.weight",
        "visual.blocks.0.mlp.fc2.bias",
        "visual.blocks.0.norm1.weight",
        "visual.blocks.0.norm2.weight",
        "visual.blocks.31.attn.qkv.weight",
        "visual.merger.weight",
        "visual.merger.bias",
    ];
    assert_all_mapped(&model, keys);

    // All visual keys pass through unchanged
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "Visual key should pass through: {key}"
        );
    }
}

#[test]
fn test_qwen3_vl_decoder_o_proj_across_layers() {
    let model = DpdfModelType::Qwen3VL;
    for layer_idx in [0, 1, 7, 15, 27] {
        let hf = format!("model.layers.{layer_idx}.self_attn.o_proj.weight");
        let expected = format!("model.layers.{layer_idx}.self_attn.out_proj.weight");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "o_proj remap failed for Qwen3-VL layer {layer_idx}"
        );
    }
}

#[test]
fn test_qwen3_vl_decoder_passthrough() {
    let model = DpdfModelType::Qwen3VL;
    let keys = &[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "Decoder passthrough failed for Qwen3-VL: {key}"
        );
    }
}

#[test]
fn test_qwen3_vl_unknown_returns_none() {
    let model = DpdfModelType::Qwen3VL;
    assert_all_none(
        &model,
        &[
            "unknown.weight",
            "encoder.layers.0.weight",
            "backbone.conv.weight",
        ],
    );
}

#[test]
fn test_qwen3_vl_collision_detection() {
    let model = DpdfModelType::Qwen3VL;
    let keys = &[
        "visual.patch_embed.proj.weight",
        "visual.blocks.0.attn.qkv.weight",
        "visual.merger.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.self_attn.o_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ];
    assert_no_collisions(&model, keys);
}

// ============================================================================
// Table Transformer exhaustive tests
// ============================================================================

#[test]
fn test_table_transformer_backbone_patterns() {
    let model = DpdfModelType::TableTransformer;
    let keys = &[
        "model.backbone.conv_encoder.model.layer1.0.conv1.weight",
        "model.backbone.conv_encoder.model.layer1.0.conv1.bias",
        "model.backbone.conv_encoder.model.layer1.0.bn1.weight",
        "model.backbone.conv_encoder.model.layer2.0.conv1.weight",
        "model.backbone.conv_encoder.model.layer3.0.conv1.weight",
        "model.backbone.conv_encoder.model.layer4.0.conv1.weight",
        "model.backbone.conv_encoder.model.conv1.weight",
        "model.backbone.conv_encoder.model.bn1.weight",
    ];
    assert_all_mapped(&model, keys);

    // Verify prefix stripping
    assert_eq!(
        map_weight_key(
            &model,
            "model.backbone.conv_encoder.model.layer1.0.conv1.weight"
        )
        .as_deref(),
        Some("backbone.layer1.0.conv1.weight")
    );
    assert_eq!(
        map_weight_key(&model, "model.backbone.conv_encoder.model.conv1.weight").as_deref(),
        Some("backbone.conv1.weight")
    );
}

#[test]
fn test_table_transformer_input_projection() {
    let model = DpdfModelType::TableTransformer;
    assert_eq!(
        map_weight_key(&model, "model.input_projection.weight").as_deref(),
        Some("input_proj.weight")
    );
    assert_eq!(
        map_weight_key(&model, "model.input_projection.bias").as_deref(),
        Some("input_proj.bias")
    );
}

#[test]
fn test_table_transformer_encoder_decoder_layers() {
    let model = DpdfModelType::TableTransformer;

    // Encoder layers across indices
    for idx in [0, 1, 3, 5] {
        let hf = format!("model.encoder.layers.{idx}.self_attn.out_proj.weight");
        let expected = format!("encoder.layers.{idx}.self_attn.out_proj.weight");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "Encoder layer {idx}"
        );
    }

    // Decoder layers
    for idx in [0, 1, 3, 5] {
        let hf = format!("model.decoder.layers.{idx}.norm1.weight");
        let expected = format!("decoder.layers.{idx}.norm1.weight");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "Decoder layer {idx}"
        );
    }
}

#[test]
fn test_table_transformer_class_bbox_heads() {
    let model = DpdfModelType::TableTransformer;
    let keys_and_expected = &[
        (
            "model.class_labels_classifier.weight",
            "class_labels_classifier.weight",
        ),
        (
            "model.class_labels_classifier.bias",
            "class_labels_classifier.bias",
        ),
        (
            "model.bbox_predictor.layers.0.weight",
            "bbox_predictor.layers.0.weight",
        ),
        (
            "model.bbox_predictor.layers.0.bias",
            "bbox_predictor.layers.0.bias",
        ),
    ];
    for (hf, expected) in keys_and_expected {
        assert_eq!(
            map_weight_key(&model, hf).as_deref(),
            Some(*expected),
            "Head key mapping: {hf}"
        );
    }
}

#[test]
fn test_table_transformer_unknown_returns_none() {
    let model = DpdfModelType::TableTransformer;
    assert_all_none(
        &model,
        &[
            "unknown.weight",
            "backbone.layer1.weight",
            "encoder.layers.0.weight",
            "vision_model.weight",
        ],
    );
}

#[test]
fn test_table_transformer_collision_detection() {
    let model = DpdfModelType::TableTransformer;
    let keys = &[
        "model.backbone.conv_encoder.model.layer1.0.conv1.weight",
        "model.backbone.conv_encoder.model.layer2.0.conv1.weight",
        "model.input_projection.weight",
        "model.input_projection.bias",
        "model.encoder.layers.0.self_attn.out_proj.weight",
        "model.decoder.layers.0.norm1.weight",
        "model.class_labels_classifier.weight",
        "model.bbox_predictor.layers.0.weight",
    ];
    assert_no_collisions(&model, keys);
}

// ============================================================================
// GLM-OCR exhaustive tests
// ============================================================================

#[test]
fn test_glm_ocr_vision_model_passthrough() {
    let model = DpdfModelType::GlmOcr;
    let keys = &[
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_model.encoder.layers.0.self_attn.q_proj.bias",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.layernorm.weight",
    ];
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "Vision passthrough failed: {key}"
        );
    }
}

#[test]
fn test_glm_ocr_vision_projection_passthrough() {
    let model = DpdfModelType::GlmOcr;
    let keys = &["vision_projection.weight", "vision_projection.bias"];
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "Vision projection passthrough failed: {key}"
        );
    }
}

#[test]
fn test_glm_ocr_mtp_heads_across_indices() {
    let model = DpdfModelType::GlmOcr;
    for idx in [0, 1, 2, 5] {
        let hf = format!("model.mtp_heads.{idx}.weight");
        let expected = format!("mtp.{idx}.weight");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "MTP head remap failed for index {idx}"
        );

        let hf_bias = format!("model.mtp_heads.{idx}.bias");
        let expected_bias = format!("mtp.{idx}.bias");
        assert_eq!(
            map_weight_key(&model, &hf_bias).as_deref(),
            Some(expected_bias.as_str()),
            "MTP head bias remap failed for index {idx}"
        );
    }
}

#[test]
fn test_glm_ocr_decoder_o_proj_across_layers() {
    let model = DpdfModelType::GlmOcr;
    for layer_idx in [0, 6, 12, 23] {
        let hf = format!("model.layers.{layer_idx}.self_attn.o_proj.weight");
        let expected = format!("model.layers.{layer_idx}.self_attn.out_proj.weight");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "o_proj remap failed for GLM-OCR layer {layer_idx}"
        );
    }
}

#[test]
fn test_glm_ocr_decoder_passthrough() {
    let model = DpdfModelType::GlmOcr;
    let keys = &[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "GLM-OCR decoder passthrough failed: {key}"
        );
    }
}

#[test]
fn test_glm_ocr_unknown_returns_none() {
    let model = DpdfModelType::GlmOcr;
    assert_all_none(
        &model,
        &[
            "unknown.foo",
            "backbone.conv.weight",
            "Student.backbone.stage0.weight",
        ],
    );
}

#[test]
fn test_glm_ocr_collision_detection() {
    let model = DpdfModelType::GlmOcr;
    let keys = &[
        "vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "vision_projection.weight",
        "model.mtp_heads.0.weight",
        "model.mtp_heads.1.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ];
    assert_no_collisions(&model, keys);
}

// ============================================================================
// PaddleOCR-VL-1.5 exhaustive tests (passthrough mapping)
// ============================================================================

#[test]
fn test_paddle_ocr_vl_vision_encoder_passthrough() {
    let model = DpdfModelType::PaddleOcr;
    let keys = &[
        "visual.vision_model.embeddings.patch_embedding.weight",
        "visual.vision_model.embeddings.patch_embedding.bias",
        "visual.vision_model.embeddings.position_embedding.weight",
        "visual.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "visual.vision_model.encoder.layers.0.self_attn.k_proj.weight",
        "visual.vision_model.encoder.layers.0.self_attn.v_proj.weight",
        "visual.vision_model.encoder.layers.0.self_attn.out_proj.weight",
        "visual.vision_model.encoder.layers.26.mlp.fc1.weight",
        "visual.vision_model.post_layernorm.weight",
        "visual.vision_model.post_layernorm.bias",
    ];
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "PaddleOCR-VL vision key should pass through: {key}"
        );
    }
}

#[test]
fn test_paddle_ocr_vl_spatial_merge_passthrough() {
    let model = DpdfModelType::PaddleOcr;
    let keys = &[
        "mlp_AR.pre_norm.weight",
        "mlp_AR.pre_norm.bias",
        "mlp_AR.linear_1.weight",
        "mlp_AR.linear_1.bias",
        "mlp_AR.linear_2.weight",
        "mlp_AR.linear_2.bias",
    ];
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "PaddleOCR-VL mlp_AR key should pass through: {key}"
        );
    }
}

#[test]
fn test_paddle_ocr_vl_decoder_passthrough() {
    let model = DpdfModelType::PaddleOcr;
    let keys = &[
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.17.self_attn.q_proj.weight",
        "model.norm.weight",
    ];
    for key in keys {
        assert_eq!(
            map_weight_key(&model, key).as_deref(),
            Some(*key),
            "PaddleOCR-VL decoder key should pass through: {key}"
        );
    }
}

#[test]
fn test_paddle_ocr_vl_lm_head_passthrough() {
    let model = DpdfModelType::PaddleOcr;
    assert_eq!(
        map_weight_key(&model, "lm_head.weight").as_deref(),
        Some("lm_head.weight")
    );
}

#[test]
fn test_paddle_ocr_vl_unrecognized_prefixes() {
    let model = DpdfModelType::PaddleOcr;
    assert_all_none(
        &model,
        &[
            "unknown.foo.bar",
            "Student.backbone.stage0.weight",
            "Student2.backbone.blocks.0.weight",
            "backbone.stage0.weight",
            "encoder.layers.0.weight",
        ],
    );
}

#[test]
fn test_paddle_ocr_vl_collision_detection() {
    let model = DpdfModelType::PaddleOcr;
    let keys = &[
        "visual.vision_model.embeddings.patch_embedding.weight",
        "visual.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "visual.vision_model.encoder.layers.26.mlp.fc1.weight",
        "mlp_AR.pre_norm.weight",
        "mlp_AR.linear_1.weight",
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.17.self_attn.q_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    ];
    assert_no_collisions(&model, keys);
}

// ============================================================================
// FireRed-OCR exhaustive tests
// ============================================================================

#[test]
fn test_firered_ocr_ctc_head() {
    let model = DpdfModelType::FireRedOcr;
    assert_eq!(
        map_weight_key(&model, "model.ctc_head.fc.weight").as_deref(),
        Some("ctc_head.fc.weight")
    );
    assert_eq!(
        map_weight_key(&model, "model.ctc_head.fc.bias").as_deref(),
        Some("ctc_head.fc.bias")
    );
}

#[test]
fn test_firered_ocr_line_detector_patterns() {
    let model = DpdfModelType::FireRedOcr;
    let keys_and_expected = &[
        (
            "model.line_detector.conv.weight",
            "line_detector.conv.weight",
        ),
        ("model.line_detector.conv.bias", "line_detector.conv.bias"),
        ("model.line_detector.fc.weight", "line_detector.fc.weight"),
        ("model.line_detector.fc.bias", "line_detector.fc.bias"),
        ("model.line_detector.bn.weight", "line_detector.bn.weight"),
    ];
    for (hf, expected) in keys_and_expected {
        assert_eq!(
            map_weight_key(&model, hf).as_deref(),
            Some(*expected),
            "FireRed line detector: {hf}"
        );
    }
}

#[test]
fn test_firered_ocr_visual_keys() {
    let model = DpdfModelType::FireRedOcr;
    let keys_and_expected = &[
        (
            "model.visual.blocks.0.attn.qkv.weight",
            "visual.blocks.0.attn.qkv.weight",
        ),
        (
            "model.visual.blocks.31.attn.proj.weight",
            "visual.blocks.31.attn.proj.weight",
        ),
        (
            "model.visual.patch_embed.proj.weight",
            "visual.patch_embed.proj.weight",
        ),
        (
            "model.visual.patch_embed.proj.bias",
            "visual.patch_embed.proj.bias",
        ),
        ("model.visual.merger.weight", "visual.merger.weight"),
        ("model.visual.merger.bias", "visual.merger.bias"),
    ];
    for (hf, expected) in keys_and_expected {
        assert_eq!(
            map_weight_key(&model, hf).as_deref(),
            Some(*expected),
            "FireRed visual key: {hf}"
        );
    }
}

#[test]
fn test_firered_ocr_lm_head() {
    let model = DpdfModelType::FireRedOcr;
    assert_eq!(
        map_weight_key(&model, "model.lm_head.weight").as_deref(),
        Some("lm_head.weight")
    );
    assert_eq!(
        map_weight_key(&model, "model.lm_head.bias").as_deref(),
        Some("lm_head.bias")
    );
}

#[test]
fn test_firered_ocr_decoder_o_proj_across_layers() {
    let model = DpdfModelType::FireRedOcr;
    for layer_idx in [0, 5, 15, 27] {
        let hf = format!("model.model.layers.{layer_idx}.self_attn.o_proj.weight");
        let expected = format!("model.layers.{layer_idx}.self_attn.out_proj.weight");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "FireRed o_proj remap layer {layer_idx}"
        );
    }
}

#[test]
fn test_firered_ocr_decoder_passthrough() {
    let model = DpdfModelType::FireRedOcr;
    // model.model.layers.N.* -> model.layers.N.* (passthrough via Qwen3-VL delegate)
    let passthrough_suffixes = &[
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];
    for suffix in passthrough_suffixes {
        let hf = format!("model.model.layers.0.{suffix}");
        let expected = format!("model.layers.0.{suffix}");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "FireRed decoder passthrough: {suffix}"
        );
    }
}

#[test]
fn test_firered_ocr_embed_and_norm() {
    let model = DpdfModelType::FireRedOcr;
    // model.model.embed_tokens.weight -> model.embed_tokens.weight
    assert_eq!(
        map_weight_key(&model, "model.model.embed_tokens.weight").as_deref(),
        Some("model.embed_tokens.weight")
    );
    // model.model.norm.weight -> model.norm.weight
    assert_eq!(
        map_weight_key(&model, "model.model.norm.weight").as_deref(),
        Some("model.norm.weight")
    );
}

#[test]
fn test_firered_ocr_unknown_returns_none() {
    let model = DpdfModelType::FireRedOcr;
    assert_all_none(
        &model,
        &[
            "unknown.foo.bar",
            "visual.blocks.0.weight",
            "lm_head.weight",
            "ctc_head.fc.weight",
            "line_detector.conv.weight",
        ],
    );
}

#[test]
fn test_firered_ocr_collision_detection() {
    let model = DpdfModelType::FireRedOcr;
    let keys = &[
        "model.ctc_head.fc.weight",
        "model.ctc_head.fc.bias",
        "model.line_detector.conv.weight",
        "model.line_detector.fc.weight",
        "model.visual.blocks.0.attn.qkv.weight",
        "model.visual.patch_embed.proj.weight",
        "model.visual.merger.weight",
        "model.lm_head.weight",
        "model.model.layers.0.self_attn.o_proj.weight",
        "model.model.layers.0.self_attn.q_proj.weight",
        "model.model.layers.1.self_attn.o_proj.weight",
        "model.model.embed_tokens.weight",
        "model.model.norm.weight",
    ];
    assert_no_collisions(&model, keys);
}

// ============================================================================
// Cross-model tests
// ============================================================================

#[test]
fn test_map_weight_key_dispatches_correctly_per_model() {
    // Verify that map_weight_key routes to the right mapper per DpdfModelType.
    // A key recognized by one model should NOT be recognized by another
    // (unless the patterns genuinely overlap).

    // PaddleOcr Student prefix is unique
    let paddle_key = "Student.backbone.stage0.0.conv1.weight";
    assert!(map_weight_key(&DpdfModelType::PaddleOcr, paddle_key).is_some());
    assert!(map_weight_key(&DpdfModelType::GraniteDocling, paddle_key).is_none());
    assert!(map_weight_key(&DpdfModelType::DocLayoutYolo, paddle_key).is_none());
    assert!(map_weight_key(&DpdfModelType::TableTransformer, paddle_key).is_none());

    // DocLayout-YOLO model.N.* pattern: other models with "model." prefix
    // will match their own branches or fall through.
    let yolo_key = "model.5.conv.weight";
    assert!(map_weight_key(&DpdfModelType::DocLayoutYolo, yolo_key).is_some());
    // Table Transformer: model. but no backbone/encoder/decoder sub-prefix -> None
    assert!(map_weight_key(&DpdfModelType::TableTransformer, yolo_key).is_none());
}

#[test]
fn test_empty_string_returns_none_all_models() {
    for model in all_model_types() {
        assert_eq!(
            map_weight_key(&model, ""),
            None,
            "Empty string should return None for {model:?}"
        );
    }
}

#[test]
fn test_all_models_none_for_totally_unrelated_key() {
    let unrelated = "some.completely.unrelated.key.that.no.model.handles";
    for model in all_model_types() {
        assert_eq!(
            map_weight_key(&model, unrelated),
            None,
            "Unrelated key should return None for {model:?}"
        );
    }
}

// ============================================================================
// Layer index extraction correctness
// ============================================================================

#[test]
fn test_layer_indices_preserved_granite_docling() {
    let model = DpdfModelType::GraniteDocling;
    // Verify that layer indices in the output match the input
    for idx in [0, 1, 10, 99, 255] {
        let hf = format!("model.layers.{idx}.self_attn.o_proj.weight");
        let mapped = map_weight_key(&model, &hf).unwrap();
        assert!(
            mapped.contains(&format!("layers.{idx}.")),
            "Layer index {idx} not preserved in output: {mapped}"
        );
    }
}

#[test]
fn test_layer_indices_preserved_doclayout_yolo() {
    let model = DpdfModelType::DocLayoutYolo;
    // Verify correct numeric-to-hierarchical translation
    assert_eq!(
        map_weight_key(&model, "model.0.foo").as_deref(),
        Some("backbone.stage0.foo")
    );
    assert_eq!(
        map_weight_key(&model, "model.9.foo").as_deref(),
        Some("backbone.stage4.sppf.foo")
    );
    assert_eq!(
        map_weight_key(&model, "model.10.foo").as_deref(),
        Some("neck.0.foo")
    );
    assert_eq!(
        map_weight_key(&model, "model.23.foo").as_deref(),
        Some("neck.13.foo")
    );
    assert_eq!(
        map_weight_key(&model, "model.24.foo").as_deref(),
        Some("head.foo")
    );
}

#[test]
fn test_mtp_head_indices_preserved_glm_ocr() {
    let model = DpdfModelType::GlmOcr;
    for idx in [0, 1, 5, 10, 100] {
        let hf = format!("model.mtp_heads.{idx}.fc.weight");
        let expected = format!("mtp.{idx}.fc.weight");
        assert_eq!(
            map_weight_key(&model, &hf).as_deref(),
            Some(expected.as_str()),
            "MTP index {idx} preservation"
        );
    }
}

// ============================================================================
// Comprehensive collision detection across full key sets
// ============================================================================

#[test]
fn test_granite_docling_full_model_no_collisions() {
    let model = DpdfModelType::GraniteDocling;
    let mut keys = Vec::new();
    // Vision encoder layers
    for layer in 0..12 {
        for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
            for suffix in &["weight", "bias"] {
                keys.push(format!(
                    "vision_model.encoder.layers.{layer}.self_attn.{proj}.{suffix}"
                ));
            }
        }
    }
    // Decoder layers
    for layer in 0..24 {
        keys.push(format!("model.layers.{layer}.self_attn.o_proj.weight"));
        keys.push(format!("model.layers.{layer}.self_attn.q_proj.weight"));
        keys.push(format!("model.layers.{layer}.mlp.gate_proj.weight"));
    }
    keys.push("model.embed_tokens.weight".to_string());
    keys.push("lm_head.weight".to_string());
    keys.push("multi_modal_projector.linear.weight".to_string());

    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    assert_no_collisions(&model, &key_refs);
}

#[test]
fn test_doclayout_yolo_full_model_no_collisions() {
    let model = DpdfModelType::DocLayoutYolo;
    let mut keys = Vec::new();
    for idx in 0..=24 {
        keys.push(format!("model.{idx}.conv.weight"));
        keys.push(format!("model.{idx}.conv.bias"));
        keys.push(format!("model.{idx}.bn.weight"));
        keys.push(format!("model.{idx}.bn.bias"));
    }
    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    assert_no_collisions(&model, &key_refs);
}

#[test]
fn test_paddle_ocr_full_model_no_collisions() {
    let model = DpdfModelType::PaddleOcr;
    let mut keys = Vec::new();
    // DB backbone
    for stage in 0..=3 {
        for block in 0..=2 {
            for conv in 1..=3 {
                keys.push(format!(
                    "Student.backbone.stage{stage}.{block}.conv{conv}.weight"
                ));
                keys.push(format!(
                    "Student.backbone.stage{stage}.{block}.conv{conv}.bias"
                ));
            }
        }
    }
    // DB neck
    for idx in 0..=3 {
        keys.push(format!("Student.neck.inner_channels.{idx}"));
        keys.push(format!("Student.neck.out_channels.{idx}"));
    }
    // DB head
    keys.push("Student.head.binarize.conv1.weight".to_string());
    keys.push("Student.head.binarize.conv2.weight".to_string());
    // SVTR
    for idx in 0..=7 {
        keys.push(format!("Student2.backbone.blocks.{idx}.attn.qkv.weight"));
        keys.push(format!("Student2.backbone.blocks.{idx}.mlp.fc1.weight"));
    }
    keys.push("Student2.backbone.patch_embed.proj.weight".to_string());
    keys.push("Student2.head.fc.weight".to_string());
    keys.push("Student2.head.fc.bias".to_string());

    let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
    assert_no_collisions(&model, &key_refs);
}

// ============================================================================
// Mapped output uniqueness (inverse collision: same input via different models)
// ============================================================================

#[test]
fn test_no_duplicate_outputs_within_single_model() {
    // For each model, generate a large set of plausible keys and verify
    // that no two produce the same output.
    let models_and_keys: Vec<(DpdfModelType, Vec<String>)> = vec![
        (DpdfModelType::GraniteDocling, {
            let mut k = Vec::new();
            for layer in 0..5 {
                for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                    k.push(format!("model.layers.{layer}.self_attn.{proj}.weight"));
                }
            }
            k
        }),
        (DpdfModelType::DocLayoutYolo, {
            (0..=24).map(|i| format!("model.{i}.conv.weight")).collect()
        }),
        (DpdfModelType::Qwen3VL, {
            let mut k = Vec::new();
            for layer in 0..5 {
                k.push(format!("model.layers.{layer}.self_attn.o_proj.weight"));
                k.push(format!("model.layers.{layer}.self_attn.q_proj.weight"));
            }
            k
        }),
    ];

    for (model, keys) in &models_and_keys {
        let mut seen = HashSet::new();
        for key in keys {
            if let Some(mapped) = map_weight_key(model, key) {
                assert!(
                    seen.insert(mapped.clone()),
                    "Duplicate output '{mapped}' for model {model:?}"
                );
            }
        }
    }
}
