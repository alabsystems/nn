// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for dpdf model weight mapping in the convert pipeline.
//!
//! Tests cover weight key mapping correctness, config detection from model
//! metadata, shape validation during mapping, and smoke tests for each
//! model's convert path.

use super::*;

// ---------------------------------------------------------------------------
// Granite-Docling weight mapping
// ---------------------------------------------------------------------------

#[test]
fn test_granite_docling_vision_key_passthrough() {
    let key = "vision_model.encoder.layers.0.self_attn.q_proj.weight";
    let mapped = map_weight_key(&DpdfModelType::GraniteDocling, key);
    assert_eq!(mapped.as_deref(), Some(key));
}

#[test]
fn test_granite_docling_decoder_o_proj_remapped() {
    let hf = "model.layers.5.self_attn.o_proj.weight";
    let expected = "model.layers.5.self_attn.out_proj.weight";
    let mapped = map_weight_key(&DpdfModelType::GraniteDocling, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_granite_docling_projector_passthrough() {
    let key = "multi_modal_projector.linear.weight";
    let mapped = map_weight_key(&DpdfModelType::GraniteDocling, key);
    assert_eq!(mapped.as_deref(), Some(key));
}

// ---------------------------------------------------------------------------
// DocLayout-YOLO weight mapping
// ---------------------------------------------------------------------------

#[test]
fn test_doclayout_yolo_backbone_stem() {
    let hf = "model.0.conv.weight";
    let expected = "backbone.stage0.conv.weight";
    let mapped = map_weight_key(&DpdfModelType::DocLayoutYolo, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_doclayout_yolo_neck_index() {
    let hf = "model.12.conv.weight";
    let expected = "neck.2.conv.weight";
    let mapped = map_weight_key(&DpdfModelType::DocLayoutYolo, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_doclayout_yolo_detect_head() {
    let hf = "model.24.cls.0.weight";
    let expected = "head.cls.0.weight";
    let mapped = map_weight_key(&DpdfModelType::DocLayoutYolo, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

// ---------------------------------------------------------------------------
// Qwen3-VL weight mapping
// ---------------------------------------------------------------------------

#[test]
fn test_qwen3_vl_visual_passthrough() {
    let key = "visual.patch_embed.proj.weight";
    let mapped = map_weight_key(&DpdfModelType::Qwen3VL, key);
    assert_eq!(mapped.as_deref(), Some(key));
}

#[test]
fn test_qwen3_vl_decoder_o_proj_remapped() {
    let hf = "model.layers.0.self_attn.o_proj.weight";
    let expected = "model.layers.0.self_attn.out_proj.weight";
    let mapped = map_weight_key(&DpdfModelType::Qwen3VL, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

// ---------------------------------------------------------------------------
// Table Transformer weight mapping
// ---------------------------------------------------------------------------

#[test]
fn test_table_transformer_backbone_remap() {
    let hf = "model.backbone.conv_encoder.model.layer1.0.conv1.weight";
    let expected = "backbone.layer1.0.conv1.weight";
    let mapped = map_weight_key(&DpdfModelType::TableTransformer, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_table_transformer_encoder_remap() {
    let hf = "model.encoder.layers.3.self_attn.out_proj.weight";
    let expected = "encoder.layers.3.self_attn.out_proj.weight";
    let mapped = map_weight_key(&DpdfModelType::TableTransformer, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_table_transformer_decoder_remap() {
    let hf = "model.decoder.layers.0.norm1.weight";
    let expected = "decoder.layers.0.norm1.weight";
    let mapped = map_weight_key(&DpdfModelType::TableTransformer, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

// ---------------------------------------------------------------------------
// GLM-OCR weight mapping
// ---------------------------------------------------------------------------

#[test]
fn test_glm_ocr_mtp_heads_remapped() {
    let hf = "model.mtp_heads.0.weight";
    let expected = "mtp.0.weight";
    let mapped = map_weight_key(&DpdfModelType::GlmOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_glm_ocr_decoder_o_proj_remapped() {
    let hf = "model.layers.12.self_attn.o_proj.weight";
    let expected = "model.layers.12.self_attn.out_proj.weight";
    let mapped = map_weight_key(&DpdfModelType::GlmOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

// ---------------------------------------------------------------------------
// PaddleOCR weight mapping
// ---------------------------------------------------------------------------

#[test]
fn test_paddle_ocr_backbone_conv_remapped() {
    let hf = "Student.backbone.stage0.0.conv1.weight";
    let expected = "db.backbone.stage0.block0.conv1.weight";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_backbone_stage3_block2() {
    let hf = "Student.backbone.stage3.2.conv2.bias";
    let expected = "db.backbone.stage3.block2.conv2.bias";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_neck_inner_channels() {
    let hf = "Student.neck.inner_channels.0";
    let expected = "db.neck.inner.0";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_neck_out_channels() {
    let hf = "Student.neck.out_channels.2";
    let expected = "db.neck.out.2";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_db_head_binarize() {
    let hf = "Student.head.binarize.conv1.weight";
    let expected = "db.head.binarize.conv1.weight";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_svtr_patch_embed() {
    let hf = "Student2.backbone.patch_embed.proj.weight";
    let expected = "svtr.patch_embed.proj.weight";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_svtr_attention_block() {
    let hf = "Student2.backbone.blocks.1.attn.qkv.weight";
    let expected = "svtr.blocks.1.attn.qkv.weight";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_svtr_mlp_block() {
    let hf = "Student2.backbone.blocks.0.mlp.fc1.bias";
    let expected = "svtr.blocks.0.mlp.fc1.bias";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_ctc_head() {
    let hf = "Student2.head.fc.weight";
    let expected = "ctc.head.fc.weight";
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_paddle_ocr_unrecognized_prefix() {
    let mapped = map_weight_key(&DpdfModelType::PaddleOcr, "Teacher.backbone.stage0.weight");
    assert_eq!(mapped, None);
}

// ---------------------------------------------------------------------------
// FireRed-OCR weight mapping
// ---------------------------------------------------------------------------

#[test]
fn test_firered_ocr_ctc_head_weight() {
    let hf = "model.ctc_head.fc.weight";
    let expected = "ctc_head.fc.weight";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_ctc_head_bias() {
    let hf = "model.ctc_head.fc.bias";
    let expected = "ctc_head.fc.bias";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_line_detector_conv() {
    let hf = "model.line_detector.conv.weight";
    let expected = "line_detector.conv.weight";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_line_detector_fc() {
    let hf = "model.line_detector.fc.bias";
    let expected = "line_detector.fc.bias";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_vision_encoder_passthrough() {
    let hf = "model.visual.blocks.3.attn.qkv.weight";
    let expected = "visual.blocks.3.attn.qkv.weight";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_vision_patch_embed() {
    let hf = "model.visual.patch_embed.proj.weight";
    let expected = "visual.patch_embed.proj.weight";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_decoder_o_proj_remapped() {
    let hf = "model.model.layers.5.self_attn.o_proj.weight";
    let expected = "model.layers.5.self_attn.out_proj.weight";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_decoder_mlp_passthrough() {
    let hf = "model.model.layers.2.mlp.gate_proj.weight";
    let expected = "model.layers.2.mlp.gate_proj.weight";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_lm_head() {
    let hf = "model.lm_head.weight";
    let expected = "lm_head.weight";
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, hf);
    assert_eq!(mapped.as_deref(), Some(expected));
}

#[test]
fn test_firered_ocr_unrecognized_key() {
    let mapped = map_weight_key(&DpdfModelType::FireRedOcr, "unknown.foo.bar");
    assert_eq!(mapped, None);
}

// ---------------------------------------------------------------------------
// Config detection from model metadata
// ---------------------------------------------------------------------------

#[test]
fn test_detect_paddle_ocr() {
    let detected = ConvertConfig::detect_model_type("PaddlePaddle/PaddleOCR-v4");
    assert_eq!(detected, Some(DpdfModelType::PaddleOcr));
}

#[test]
fn test_detect_granite_docling() {
    let detected = ConvertConfig::detect_model_type("ds4sd/Granite-Docling-258M-Preview");
    assert_eq!(detected, Some(DpdfModelType::GraniteDocling));
}

#[test]
fn test_detect_doclayout_yolo() {
    let detected = ConvertConfig::detect_model_type("juliozhao/DocLayout-YOLO-DocStructBench");
    assert_eq!(detected, Some(DpdfModelType::DocLayoutYolo));
}

#[test]
fn test_detect_qwen3_vl() {
    let detected = ConvertConfig::detect_model_type("Qwen/Qwen3-VL-2B");
    assert_eq!(detected, Some(DpdfModelType::Qwen3VL));
}

#[test]
fn test_detect_table_transformer() {
    let detected = ConvertConfig::detect_model_type("microsoft/table-transformer-detection");
    assert_eq!(detected, Some(DpdfModelType::TableTransformer));
}

#[test]
fn test_detect_glm_ocr() {
    let detected = ConvertConfig::detect_model_type("THUDM/glm-ocr-0.9B");
    assert_eq!(detected, Some(DpdfModelType::GlmOcr));
}

#[test]
fn test_detect_firered_ocr() {
    let detected = ConvertConfig::detect_model_type("yuyq96/FireRed-OCR-Qwen3-VL-2B");
    assert_eq!(detected, Some(DpdfModelType::FireRedOcr));
}

#[test]
fn test_detect_unknown_model() {
    let detected = ConvertConfig::detect_model_type("AI Provider/whisper-large-v3");
    assert_eq!(detected, None);
}

// ---------------------------------------------------------------------------
// Shape validation / remap_weight_keys integration
// ---------------------------------------------------------------------------

#[test]
fn test_remap_weight_keys_preserves_tensors() {
    let mut weights = HashMap::new();
    let t = DynTensor::from_vec(vec![1.0_f32; 6], &[2, 3], &Device::Cpu).expect("tensor creation");
    weights.insert(
        "model.layers.0.self_attn.o_proj.weight".to_string(),
        t.clone(),
    );
    weights.insert("lm_head.weight".to_string(), t);

    let remapped = remap_weight_keys(&DpdfModelType::GraniteDocling, weights);
    assert_eq!(remapped.len(), 2);
    // o_proj was remapped to out_proj
    assert!(remapped.contains_key("model.layers.0.self_attn.out_proj.weight"));
    // lm_head passed through
    assert!(remapped.contains_key("lm_head.weight"));
    // original key is gone
    assert!(!remapped.contains_key("model.layers.0.self_attn.o_proj.weight"));
}

#[test]
fn test_convert_config_with_model_type_builder() {
    let config =
        ConvertConfig::new("granite-docling").with_model_type(DpdfModelType::GraniteDocling);
    assert_eq!(config.model_type, Some(DpdfModelType::GraniteDocling));
    assert_eq!(config.model_name, "granite-docling");
}

#[test]
fn test_unrecognized_key_returns_none() {
    // Keys outside the model's known patterns return None (pass-through
    // in remap_weight_keys).
    let mapped = map_weight_key(&DpdfModelType::GraniteDocling, "unknown.foo.bar");
    assert_eq!(mapped, None);
}

#[test]
fn test_doclayout_yolo_out_of_range_index() {
    // Index 30 is beyond the known YOLO architecture indices.
    let mapped = map_weight_key(&DpdfModelType::DocLayoutYolo, "model.30.conv.weight");
    assert_eq!(mapped, None);
}

// ---------------------------------------------------------------------------
// RT-DETR HuggingFace weight mapping
// ---------------------------------------------------------------------------

#[test]
fn test_rt_detr_strip_model_prefix() {
    let mapped = map_weight_key(&DpdfModelType::RtDetr, "model.encoder.norm1.weight");
    assert_eq!(mapped, Some("encoder.norm1.weight".to_string()));
}

#[test]
fn test_rt_detr_stem_conv_mapping() {
    // HF: model.backbone.model.embedder.embedder.0.convolution.weight
    // nn: backbone.stem.0.conv.weight
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.backbone.model.embedder.embedder.0.convolution.weight",
    );
    assert_eq!(mapped, Some("backbone.stem.0.conv.weight".to_string()));
}

#[test]
fn test_rt_detr_stem_bn_mapping() {
    // HF: model.backbone.model.embedder.embedder.1.normalization.weight
    // nn: backbone.stem.1.bn.weight
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.backbone.model.embedder.embedder.1.normalization.weight",
    );
    assert_eq!(mapped, Some("backbone.stem.1.bn.weight".to_string()));

    // running_mean
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.backbone.model.embedder.embedder.2.normalization.running_mean",
    );
    assert_eq!(mapped, Some("backbone.stem.2.bn.running_mean".to_string()));
}

#[test]
fn test_rt_detr_stage_conv_mapping() {
    // HF: model.backbone.model.encoder.stages.0.layers.0.layer.0.convolution.weight
    // nn: backbone.layer1.0.conv1.weight (stage 0 -> layer1, conv 0 -> conv1)
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.backbone.model.encoder.stages.0.layers.0.layer.0.convolution.weight",
    );
    assert_eq!(mapped, Some("backbone.layer1.0.conv1.weight".to_string()));
}

#[test]
fn test_rt_detr_stage_bn_mapping() {
    // HF: model.backbone.model.encoder.stages.1.layers.0.layer.1.normalization.bias
    // nn: backbone.layer2.0.bn2.bias (stage 1 -> layer2, conv 1 -> bn2)
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.backbone.model.encoder.stages.1.layers.0.layer.1.normalization.bias",
    );
    assert_eq!(mapped, Some("backbone.layer2.0.bn2.bias".to_string()));
}

#[test]
fn test_rt_detr_shortcut_mapping() {
    // HF: model.backbone.model.encoder.stages.1.layers.0.shortcut.convolution.weight
    // nn: backbone.layer2.0.downsample.0.weight
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.backbone.model.encoder.stages.1.layers.0.shortcut.convolution.weight",
    );
    assert_eq!(
        mapped,
        Some("backbone.layer2.0.downsample.0.weight".to_string())
    );
}

#[test]
fn test_rt_detr_shortcut_norm_mapping() {
    // HF: model.backbone.model.encoder.stages.2.layers.0.shortcut.normalization.weight
    // nn: backbone.layer3.0.downsample.1.weight
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.backbone.model.encoder.stages.2.layers.0.shortcut.normalization.weight",
    );
    assert_eq!(
        mapped,
        Some("backbone.layer3.0.downsample.1.weight".to_string())
    );
}

#[test]
fn test_rt_detr_non_backbone_passthrough() {
    // Encoder, decoder, input_proj keys just strip model. prefix.
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.decoder.layers.0.self_attn.q_proj.weight",
    );
    assert_eq!(
        mapped,
        Some("decoder.layers.0.self_attn.q_proj.weight".to_string())
    );

    let mapped = map_weight_key(&DpdfModelType::RtDetr, "model.input_proj.0.conv.weight");
    assert_eq!(mapped, Some("input_proj.0.conv.weight".to_string()));
}

#[test]
fn test_rt_detr_stage3_block1() {
    // Stage 3 (the last), block 1, conv 0, normalization
    // HF: model.backbone.model.encoder.stages.3.layers.1.layer.0.normalization.running_var
    // nn: backbone.layer4.1.bn1.running_var
    let mapped = map_weight_key(
        &DpdfModelType::RtDetr,
        "model.backbone.model.encoder.stages.3.layers.1.layer.0.normalization.running_var",
    );
    assert_eq!(
        mapped,
        Some("backbone.layer4.1.bn1.running_var".to_string())
    );
}
