// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Exhaustive config validation and cross-model consistency tests for dpdf models.
//!
//! Validates that every dpdf model config preset satisfies internal invariants:
//! no zero dimensions, attention compatibility, FFN expansion, vocab/layer positivity,
//! vision patch alignment, quantization constraints, memory estimates, preprocessing
//! matches model input requirements, and cross-model VLM projection compatibility.
//!
//! Part of #3929.

use nn_models::doclayout_yolo::DocLayoutYoloConfig;
use nn_models::dpdf_image_preprocess::DpdfPreprocessConfig;
use nn_models::firered_ocr::FireRedOcrConfig;
use nn_models::glm_ocr::GlmOcrConfig;
use nn_models::granite_docling::GraniteDoclingConfig;
use nn_models::paddle_ocr::PaddleOcrVlConfig;
use nn_models::qwen3_vl::Qwen3VLConfig;
use nn_models::qwen3_vl_quantized::{QuantMethod, Qwen3VLQuantConfig};
use nn_models::table_transformer::TableTransformerConfig;

// ============================================================================
// 1. Default config validity: no zero dims, no negatives, validate() passes
// ============================================================================

#[test]
fn test_granite_docling_default_validates() {
    let cfg = GraniteDoclingConfig::default_258m();
    cfg.validate().expect("default_258m should validate");
    assert!(cfg.image_size > 0);
    assert!(cfg.patch_size > 0);
    assert!(cfg.vision_hidden > 0);
    assert!(cfg.vision_heads > 0);
    assert!(cfg.vision_layers > 0);
    assert!(cfg.decoder_hidden > 0);
    assert!(cfg.decoder_heads > 0);
    assert!(cfg.decoder_kv_heads > 0);
    assert!(cfg.decoder_intermediate > 0);
    assert!(cfg.decoder_layers > 0);
    assert!(cfg.vocab_size > 0);
    assert!(cfg.rms_norm_eps > 0.0);
}

#[test]
fn test_doclayout_yolo_default_validates() {
    let cfg = DocLayoutYoloConfig::default();
    assert!(cfg.input_channels > 0);
    assert!(cfg.num_classes > 0);
    assert!(cfg.reg_max > 0);
    assert!(cfg.conf_threshold > 0.0);
    assert!(cfg.iou_threshold > 0.0);
    for &ch in &cfg.backbone_channels {
        assert!(ch > 0, "backbone channel must be > 0");
    }
}

#[test]
fn test_glm_ocr_preset_validates() {
    let cfg = GlmOcrConfig::preset_900m();
    cfg.validate().expect("preset_900m should validate");
    assert!(cfg.hidden_size > 0);
    assert!(cfg.num_heads > 0);
    assert!(cfg.num_kv_heads > 0);
    assert!(cfg.intermediate_size > 0);
    assert!(cfg.num_layers > 0);
    assert!(cfg.vocab_size > 0);
    assert!(cfg.vision_hidden > 0);
    assert!(cfg.vision_heads > 0);
    assert!(cfg.vision_layers > 0);
    assert!(cfg.image_size > 0);
    assert!(cfg.patch_size > 0);
    assert!(cfg.rms_norm_eps > 0.0);
}

#[test]
fn test_table_transformer_detection_validates() {
    let cfg = TableTransformerConfig::preset_detection();
    cfg.validate().expect("preset_detection should validate");
    assert!(cfg.hidden_dim > 0);
    assert!(cfg.num_heads > 0);
    assert!(cfg.num_encoder_layers > 0);
    assert!(cfg.num_decoder_layers > 0);
    assert!(cfg.num_queries > 0);
    assert!(cfg.num_classes > 0);
    assert!(cfg.ffn_dim > 0);
}

#[test]
fn test_table_transformer_structure_validates() {
    let cfg = TableTransformerConfig::preset_structure();
    cfg.validate().expect("preset_structure should validate");
    assert!(cfg.num_classes > 0);
}

#[test]
fn test_qwen3_vl_2b_validates() {
    let cfg = Qwen3VLConfig::preset_2b();
    cfg.validate().expect("preset_2b should validate");
    assert!(cfg.hidden_size > 0);
    assert!(cfg.num_heads > 0);
    assert!(cfg.num_kv_heads > 0);
    assert!(cfg.intermediate_size > 0);
    assert!(cfg.num_layers > 0);
    assert!(cfg.vocab_size > 0);
    assert!(cfg.vision_hidden > 0);
    assert!(cfg.vision_heads > 0);
    assert!(cfg.vision_layers > 0);
    assert!(cfg.vision_patch_size > 0);
    assert!(cfg.vision_temporal_patch > 0);
    assert!(!cfg.is_moe(), "2B should not be MoE");
}

#[test]
fn test_qwen3_vl_7b_validates() {
    let cfg = Qwen3VLConfig::preset_7b();
    cfg.validate().expect("preset_7b should validate");
    assert!(!cfg.is_moe(), "7B should not be MoE");
}

#[test]
fn test_qwen3_vl_30b_a3b_validates() {
    let cfg = Qwen3VLConfig::preset_30b_a3b();
    cfg.validate().expect("preset_30b_a3b should validate");
    assert!(cfg.is_moe(), "30B-A3B should be MoE");
    assert!(cfg.num_experts > 0);
    assert!(cfg.active_experts > 0);
    assert!(cfg.active_experts <= cfg.num_experts);
}

#[test]
fn test_paddle_ocr_vl_validates() {
    let cfg = PaddleOcrVlConfig::default_vl();
    cfg.validate().expect("default_vl should validate");
    assert!(cfg.decoder_hidden > 0);
    assert!(cfg.decoder_intermediate > 0);
    assert!(cfg.num_decoder_layers > 0);
    assert!(cfg.num_heads > 0);
    assert!(cfg.num_kv_heads > 0);
    assert!(cfg.head_dim > 0);
    assert!(cfg.vocab_size > 0);
    assert!(cfg.vision.hidden_size > 0);
    assert!(cfg.vision.num_hidden_layers > 0);
    assert!(cfg.vision.num_attention_heads > 0);
}

#[test]
fn test_firered_ocr_2b_validates() {
    let cfg = FireRedOcrConfig::preset_2b();
    cfg.validate().expect("preset_2b should validate");
    assert!(cfg.hidden_size() > 0);
    assert!(cfg.num_layers() > 0);
    assert!(cfg.vocab_size() > 0);
    assert!(cfg.max_output_tokens > 0);
}

#[test]
fn test_qwen3_vl_quant_gptq_validates() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.validate().expect("GPTQ preset should validate");
    assert_eq!(cfg.bits, 4);
    assert!(cfg.group_size > 0);
    assert!(cfg.group_size.is_power_of_two());
    assert!(cfg.is_moe());
}

#[test]
fn test_qwen3_vl_quant_awq_validates() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    cfg.validate().expect("AWQ preset should validate");
    assert_eq!(cfg.bits, 4);
    assert!(!cfg.desc_act, "AWQ should not use desc_act");
    assert!(cfg.is_moe());
}

// ============================================================================
// 2. Attention compatibility: hidden_dim divisible by num_heads
// ============================================================================

#[test]
fn test_granite_docling_attention_compat() {
    let cfg = GraniteDoclingConfig::default_258m();
    assert_eq!(
        cfg.decoder_hidden % cfg.decoder_heads,
        0,
        "decoder_hidden ({}) must be divisible by decoder_heads ({})",
        cfg.decoder_hidden,
        cfg.decoder_heads
    );
    assert_eq!(
        cfg.decoder_heads % cfg.decoder_kv_heads,
        0,
        "decoder_heads ({}) must be divisible by decoder_kv_heads ({})",
        cfg.decoder_heads,
        cfg.decoder_kv_heads
    );
    assert!(cfg.head_dim() > 0);
}

#[test]
fn test_glm_ocr_attention_compat() {
    let cfg = GlmOcrConfig::preset_900m();
    assert_eq!(cfg.hidden_size % cfg.num_heads, 0);
    assert_eq!(cfg.num_heads % cfg.num_kv_heads, 0);
    assert!(cfg.head_dim() > 0);
    assert!(cfg.gqa_ratio() > 0);
}

#[test]
fn test_table_transformer_attention_compat() {
    let det = TableTransformerConfig::preset_detection();
    assert_eq!(det.hidden_dim % det.num_heads, 0);
    let stru = TableTransformerConfig::preset_structure();
    assert_eq!(stru.hidden_dim % stru.num_heads, 0);
}

#[test]
fn test_qwen3_vl_all_presets_attention_compat() {
    for (name, cfg) in [
        ("2B", Qwen3VLConfig::preset_2b()),
        ("7B", Qwen3VLConfig::preset_7b()),
        ("30B-A3B", Qwen3VLConfig::preset_30b_a3b()),
    ] {
        assert_eq!(
            cfg.hidden_size % cfg.num_heads,
            0,
            "{name}: hidden_size not divisible by num_heads"
        );
        assert_eq!(
            cfg.num_heads % cfg.num_kv_heads,
            0,
            "{name}: num_heads not divisible by num_kv_heads"
        );
        assert!(cfg.head_dim() > 0, "{name}: head_dim must be > 0");
        assert!(cfg.gqa_ratio() > 0, "{name}: gqa_ratio must be > 0");
    }
}

#[test]
fn test_paddle_ocr_vl_attention_compat() {
    let cfg = PaddleOcrVlConfig::default_vl();
    assert_eq!(
        cfg.num_heads % cfg.num_kv_heads,
        0,
        "num_heads must be divisible by num_kv_heads"
    );
    assert!(cfg.head_dim > 0);
    assert!(cfg.gqa_ratio() > 0);
}

#[test]
fn test_firered_ocr_attention_compat() {
    let cfg = FireRedOcrConfig::preset_2b();
    assert!(cfg.head_dim() > 0);
    assert!(cfg.gqa_ratio() > 0);
    assert_eq!(cfg.base_config.hidden_size % cfg.base_config.num_heads, 0);
    assert_eq!(cfg.base_config.num_heads % cfg.base_config.num_kv_heads, 0);
}

// ============================================================================
// 3. FFN expansion: intermediate_size >= hidden_size
// ============================================================================

#[test]
fn test_granite_docling_ffn_expansion() {
    let cfg = GraniteDoclingConfig::default_258m();
    assert!(
        cfg.decoder_intermediate >= cfg.decoder_hidden,
        "intermediate ({}) should be >= hidden ({})",
        cfg.decoder_intermediate,
        cfg.decoder_hidden
    );
}

#[test]
fn test_glm_ocr_ffn_expansion() {
    let cfg = GlmOcrConfig::preset_900m();
    assert!(
        cfg.intermediate_size >= cfg.hidden_size,
        "intermediate ({}) should be >= hidden ({})",
        cfg.intermediate_size,
        cfg.hidden_size
    );
}

#[test]
fn test_table_transformer_ffn_expansion() {
    let cfg = TableTransformerConfig::preset_detection();
    assert!(
        cfg.ffn_dim >= cfg.hidden_dim,
        "ffn_dim ({}) should be >= hidden_dim ({})",
        cfg.ffn_dim,
        cfg.hidden_dim
    );
}

#[test]
fn test_qwen3_vl_ffn_expansion() {
    for (name, cfg) in [
        ("2B", Qwen3VLConfig::preset_2b()),
        ("7B", Qwen3VLConfig::preset_7b()),
        ("30B-A3B", Qwen3VLConfig::preset_30b_a3b()),
    ] {
        assert!(
            cfg.intermediate_size >= cfg.hidden_size,
            "{name}: intermediate ({}) should be >= hidden ({})",
            cfg.intermediate_size,
            cfg.hidden_size
        );
    }
}

// ============================================================================
// 4. Vocab size > 0 for models with token output
// ============================================================================

#[test]
fn test_vocab_sizes_positive() {
    let granite = GraniteDoclingConfig::default_258m();
    assert!(granite.vocab_size > 0, "Granite-Docling vocab must be > 0");

    let glm = GlmOcrConfig::preset_900m();
    assert!(glm.vocab_size > 0, "GLM-OCR vocab must be > 0");

    let qwen_2b = Qwen3VLConfig::preset_2b();
    assert!(qwen_2b.vocab_size > 0, "Qwen3-VL-2B vocab must be > 0");

    let qwen_7b = Qwen3VLConfig::preset_7b();
    assert!(qwen_7b.vocab_size > 0, "Qwen3-VL-7B vocab must be > 0");

    let qwen_30b = Qwen3VLConfig::preset_30b_a3b();
    assert!(
        qwen_30b.vocab_size > 0,
        "Qwen3-VL-30B-A3B vocab must be > 0"
    );

    let paddle = PaddleOcrVlConfig::default_vl();
    assert!(paddle.vocab_size > 0, "PaddleOCR-VL vocab must be > 0");

    let firered = FireRedOcrConfig::preset_2b();
    assert!(firered.vocab_size() > 0, "FireRed-OCR vocab must be > 0");
}

// ============================================================================
// 5. Num layers > 0
// ============================================================================

#[test]
fn test_num_layers_positive() {
    let granite = GraniteDoclingConfig::default_258m();
    assert!(granite.decoder_layers > 0);
    assert!(granite.vision_layers > 0);

    let glm = GlmOcrConfig::preset_900m();
    assert!(glm.num_layers > 0);
    assert!(glm.vision_layers > 0);

    let tt_det = TableTransformerConfig::preset_detection();
    assert!(tt_det.num_encoder_layers > 0);
    assert!(tt_det.num_decoder_layers > 0);

    for (name, cfg) in [
        ("2B", Qwen3VLConfig::preset_2b()),
        ("7B", Qwen3VLConfig::preset_7b()),
        ("30B-A3B", Qwen3VLConfig::preset_30b_a3b()),
    ] {
        assert!(cfg.num_layers > 0, "{name}: decoder layers must be > 0");
        assert!(cfg.vision_layers > 0, "{name}: vision layers must be > 0");
    }

    let paddle = PaddleOcrVlConfig::default_vl();
    assert!(paddle.num_decoder_layers > 0);
    assert!(paddle.vision.num_hidden_layers > 0);

    let firered = FireRedOcrConfig::preset_2b();
    assert!(firered.num_layers() > 0);
}

// ============================================================================
// 6. Patch size > 0, image size divisible by patch size (vision models)
// ============================================================================

#[test]
fn test_granite_docling_patch_alignment() {
    let cfg = GraniteDoclingConfig::default_258m();
    assert!(cfg.patch_size > 0);
    assert_eq!(
        cfg.image_size % cfg.patch_size,
        0,
        "image_size ({}) must be divisible by patch_size ({})",
        cfg.image_size,
        cfg.patch_size
    );
    let expected_patches = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size);
    assert_eq!(cfg.num_patches(), expected_patches);
    assert!(cfg.num_patches() > 0);
}

#[test]
fn test_glm_ocr_patch_alignment() {
    let cfg = GlmOcrConfig::preset_900m();
    assert!(cfg.patch_size > 0);
    assert_eq!(
        cfg.image_size % cfg.patch_size,
        0,
        "image_size ({}) must be divisible by patch_size ({})",
        cfg.image_size,
        cfg.patch_size
    );
    let expected_patches = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size);
    assert_eq!(cfg.num_patches(), expected_patches);
    assert!(cfg.num_patches() > 0);
}

#[test]
fn test_qwen3_vl_patch_size_positive() {
    for (name, cfg) in [
        ("2B", Qwen3VLConfig::preset_2b()),
        ("7B", Qwen3VLConfig::preset_7b()),
        ("30B-A3B", Qwen3VLConfig::preset_30b_a3b()),
    ] {
        assert!(
            cfg.vision_patch_size > 0,
            "{name}: vision_patch_size must be > 0"
        );
        assert!(
            cfg.vision_temporal_patch > 0,
            "{name}: vision_temporal_patch must be > 0"
        );
    }
}

// ============================================================================
// 7. Quantization: group_size divides hidden_dim, bits in {2,3,4,8}
// ============================================================================

#[test]
fn test_quant_gptq_group_size_divides_hidden() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    assert_eq!(
        cfg.base.hidden_size % cfg.group_size,
        0,
        "hidden_size ({}) must be divisible by group_size ({})",
        cfg.base.hidden_size,
        cfg.group_size
    );
    assert_eq!(
        cfg.base.intermediate_size % cfg.group_size,
        0,
        "intermediate_size ({}) must be divisible by group_size ({})",
        cfg.base.intermediate_size,
        cfg.group_size
    );
    assert!(
        [2, 3, 4, 8].contains(&cfg.bits),
        "bits ({}) must be in {{2, 3, 4, 8}}",
        cfg.bits
    );
    assert!(cfg.group_size.is_power_of_two());
    assert_eq!(cfg.quant_method, QuantMethod::Gptq);
}

#[test]
fn test_quant_awq_group_size_divides_hidden() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    assert_eq!(cfg.base.hidden_size % cfg.group_size, 0);
    assert_eq!(cfg.base.intermediate_size % cfg.group_size, 0);
    assert!([2, 3, 4, 8].contains(&cfg.bits));
    assert!(cfg.group_size.is_power_of_two());
    assert_eq!(cfg.quant_method, QuantMethod::Awq);
    assert!(!cfg.desc_act, "AWQ must not use desc_act");
}

#[test]
fn test_quant_format_conversion() {
    let gptq_cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let gptq_fmt = gptq_cfg
        .to_gptq_format()
        .expect("GPTQ format conversion should succeed");
    assert_eq!(gptq_fmt.group_size, gptq_cfg.group_size);
    assert_eq!(gptq_fmt.bits, gptq_cfg.bits);
    assert_eq!(gptq_fmt.act_order, gptq_cfg.desc_act);

    let awq_cfg = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    let awq_fmt = awq_cfg
        .to_awq_format()
        .expect("AWQ format conversion should succeed");
    assert_eq!(awq_fmt.group_size, awq_cfg.group_size);
    assert_eq!(awq_fmt.bits, awq_cfg.bits);

    // Cross-method should fail
    assert!(gptq_cfg.to_awq_format().is_err());
    assert!(awq_cfg.to_gptq_format().is_err());
}

#[test]
fn test_quant_invalid_bits_rejected() {
    let mut cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    cfg.bits = 5;
    assert!(
        cfg.validate().is_err(),
        "bits=5 should fail validation (only 4 supported)"
    );
}

#[test]
fn test_quant_zero_group_size_rejected() {
    let cfg = Qwen3VLQuantConfig::new(
        Qwen3VLConfig::preset_30b_a3b(),
        QuantMethod::Gptq,
        4,
        0, // invalid
        false,
    );
    assert!(cfg.validate().is_err(), "group_size=0 should fail");
}

#[test]
fn test_quant_non_power_of_two_group_size_rejected() {
    let cfg = Qwen3VLQuantConfig::new(
        Qwen3VLConfig::preset_30b_a3b(),
        QuantMethod::Gptq,
        4,
        96, // not power of two
        false,
    );
    assert!(
        cfg.validate().is_err(),
        "group_size=96 (not power of 2) should fail"
    );
}

#[test]
fn test_quant_awq_desc_act_rejected() {
    let cfg = Qwen3VLQuantConfig::new(
        Qwen3VLConfig::preset_30b_a3b(),
        QuantMethod::Awq,
        4,
        128,
        true, // AWQ + desc_act is invalid
    );
    assert!(
        cfg.validate().is_err(),
        "AWQ with desc_act=true should fail"
    );
}

// ============================================================================
// 8. Memory estimate is positive
// ============================================================================

#[test]
fn test_quant_memory_estimate_positive() {
    let gptq = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let mem_gptq = gptq.estimated_memory_bytes();
    assert!(mem_gptq > 0, "GPTQ memory estimate must be > 0");

    let awq = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    let mem_awq = awq.estimated_memory_bytes();
    assert!(mem_awq > 0, "AWQ memory estimate must be > 0");

    // Both should give similar estimates since same architecture
    let ratio = mem_gptq as f64 / mem_awq as f64;
    assert!(
        (0.5..2.0).contains(&ratio),
        "GPTQ/AWQ memory ratio ({ratio:.2}) should be within 2x"
    );

    // Sanity check: 30B MoE model should be multi-GB even quantized
    let one_gb = 1_000_000_000;
    assert!(
        mem_gptq > one_gb,
        "30B-A3B GPTQ should exceed 1 GB, got {mem_gptq} bytes"
    );
}

#[test]
fn test_quant_memory_estimate_standalone_matches_method() {
    let cfg = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let method_result = cfg.estimated_memory_bytes();
    let standalone_result = nn_models::estimate_memory_bytes(&cfg);
    assert_eq!(method_result, standalone_result);
}

// ============================================================================
// 9. Preprocessing config matches model input requirements
// ============================================================================

#[test]
fn test_preprocess_granite_docling_target_dims() {
    let pp = DpdfPreprocessConfig::for_granite_docling();
    assert!(pp.target_height > 0);
    assert!(pp.target_width > 0);
    assert_eq!(
        pp.target_height, pp.target_width,
        "Granite expects square input"
    );
    assert!(pp.scale_factor > 0.0);
    // Std values must be non-zero (used as divisor)
    for &s in &pp.std {
        assert!(s > 0.0, "std must be > 0 to avoid div-by-zero");
    }
}

#[test]
fn test_preprocess_doclayout_yolo_letterbox() {
    let pp = DpdfPreprocessConfig::for_doclayout_yolo();
    assert!(pp.target_height > 0);
    assert!(pp.target_width > 0);
    assert!(pp.maintain_aspect);
    match &pp.padding_mode {
        nn_models::dpdf_image_preprocess::PaddingMode::Letterbox { fill_value } => {
            assert!(*fill_value >= 0.0);
        }
        other => panic!("expected Letterbox, got {other:?}"),
    }
}

#[test]
fn test_preprocess_paddle_ocr_detect() {
    let pp = DpdfPreprocessConfig::for_paddle_ocr_detect();
    assert!(pp.target_height > 0);
    assert!(pp.target_width > 0);
    assert!(pp.maintain_aspect);
    for &s in &pp.std {
        assert!(s > 0.0);
    }
}

#[test]
fn test_preprocess_paddle_ocr_recognize() {
    let pp = DpdfPreprocessConfig::for_paddle_ocr_recognize();
    // Legacy recognizer preprocess still exists for backward compatibility.
    // Validate it has positive dimensions.
    assert!(pp.target_height > 0);
    assert!(pp.target_width > 0);
    for &s in &pp.std {
        assert!(s > 0.0);
    }
}

#[test]
fn test_preprocess_table_transformer() {
    let pp = DpdfPreprocessConfig::for_table_transformer();
    assert!(pp.target_height > 0);
    assert!(pp.target_width > 0);
    assert!(pp.maintain_aspect);
    for &s in &pp.std {
        assert!(s > 0.0);
    }
}

#[test]
fn test_preprocess_qwen3_vl_dynamic_resolution() {
    let pp = DpdfPreprocessConfig::for_qwen3_vl();
    // Qwen3-VL uses dynamic resolution, so target_height/width are 0
    assert!(pp.min_pixels > 0, "dynamic resolution needs min_pixels > 0");
    assert!(pp.max_pixels > 0, "dynamic resolution needs max_pixels > 0");
    assert!(
        pp.max_pixels >= pp.min_pixels,
        "max_pixels ({}) must be >= min_pixels ({})",
        pp.max_pixels,
        pp.min_pixels
    );
    assert!(pp.patch_size > 0, "dynamic resolution needs patch_size > 0");
    for &s in &pp.std {
        assert!(s > 0.0);
    }
}

#[test]
fn test_preprocess_glm_ocr() {
    let pp = DpdfPreprocessConfig::for_glm_ocr();
    assert!(pp.target_height > 0);
    assert!(pp.target_width > 0);
    assert!(pp.maintain_aspect);
    for &s in &pp.std {
        assert!(s > 0.0);
    }
}

#[test]
fn test_all_preprocess_configs_nonzero_scale() {
    let configs = [
        (
            "granite_docling",
            DpdfPreprocessConfig::for_granite_docling(),
        ),
        ("doclayout_yolo", DpdfPreprocessConfig::for_doclayout_yolo()),
        (
            "paddle_detect",
            DpdfPreprocessConfig::for_paddle_ocr_detect(),
        ),
        (
            "paddle_recognize",
            DpdfPreprocessConfig::for_paddle_ocr_recognize(),
        ),
        (
            "table_transformer",
            DpdfPreprocessConfig::for_table_transformer(),
        ),
        ("qwen3_vl", DpdfPreprocessConfig::for_qwen3_vl()),
        ("glm_ocr", DpdfPreprocessConfig::for_glm_ocr()),
    ];
    for (name, cfg) in configs {
        assert!(
            cfg.scale_factor > 0.0,
            "{name}: scale_factor must be > 0, got {}",
            cfg.scale_factor
        );
        for (ch, &s) in cfg.std.iter().enumerate() {
            assert!(
                s > 0.0,
                "{name}: std[{ch}] must be > 0 to avoid div-by-zero"
            );
        }
    }
}

// ============================================================================
// 10. Cross-model: VLM projection output matches LM embedding dim
// ============================================================================

#[test]
fn test_granite_docling_vision_projection_dims() {
    let cfg = GraniteDoclingConfig::default_258m();
    // Vision projection: vision_hidden -> decoder_hidden
    // For the projection to work, vision_hidden must be the input dim
    // and decoder_hidden must be the output dim
    assert!(cfg.vision_hidden > 0);
    assert!(cfg.decoder_hidden > 0);
    // In Granite-Docling, vision_hidden == decoder_hidden (both 768)
    assert_eq!(
        cfg.vision_hidden, cfg.decoder_hidden,
        "Granite-Docling vision and decoder hidden dims must match for direct projection"
    );
}

#[test]
fn test_glm_ocr_vision_to_decoder_projection() {
    let cfg = GlmOcrConfig::preset_900m();
    // Vision projection maps vision_hidden -> hidden_size (decoder)
    assert!(cfg.vision_hidden > 0);
    assert!(cfg.hidden_size > 0);
    // In GLM-OCR the decoder hidden (1536) > vision hidden (768),
    // so the projection is a Linear(768 -> 1536)
    // Both must be positive for the Linear to be constructible
}

#[test]
fn test_qwen3_vl_vision_merger_dims() {
    for (name, cfg) in [
        ("2B", Qwen3VLConfig::preset_2b()),
        ("7B", Qwen3VLConfig::preset_7b()),
        ("30B-A3B", Qwen3VLConfig::preset_30b_a3b()),
    ] {
        // Vision merger: vision_hidden -> hidden_size
        assert!(cfg.vision_hidden > 0, "{name}: vision_hidden must be > 0");
        assert!(cfg.hidden_size > 0, "{name}: hidden_size must be > 0");
    }
}

#[test]
fn test_firered_ocr_inherits_qwen3_vl_architecture() {
    let firered = FireRedOcrConfig::preset_2b();
    let qwen_2b = Qwen3VLConfig::preset_2b();

    // FireRed-OCR is fine-tuned from Qwen3-VL-2B, so architecture must match
    assert_eq!(
        firered.base_config.hidden_size, qwen_2b.hidden_size,
        "FireRed hidden_size should match base Qwen3-VL-2B"
    );
    assert_eq!(firered.base_config.num_heads, qwen_2b.num_heads);
    assert_eq!(firered.base_config.num_kv_heads, qwen_2b.num_kv_heads);
    assert_eq!(firered.base_config.num_layers, qwen_2b.num_layers);
    assert_eq!(
        firered.base_config.intermediate_size,
        qwen_2b.intermediate_size
    );
    assert_eq!(firered.base_config.vision_hidden, qwen_2b.vision_hidden);
    assert_eq!(
        firered.base_config.vision_patch_size,
        qwen_2b.vision_patch_size
    );
    // Vocab size may differ (FireRed uses 151936 vs Qwen3 152064)
    assert!(firered.vocab_size() > 0);
}

#[test]
fn test_qwen3_vl_quant_matches_base_architecture() {
    let quant_gptq = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let base_30b = Qwen3VLConfig::preset_30b_a3b();

    // Quantized config should preserve the base architecture
    // (except that presets may use slightly different expert counts)
    assert_eq!(quant_gptq.base.hidden_size, base_30b.hidden_size);
    assert_eq!(quant_gptq.base.num_heads, base_30b.num_heads);
    assert_eq!(quant_gptq.base.num_kv_heads, base_30b.num_kv_heads);
    assert_eq!(quant_gptq.base.num_layers, base_30b.num_layers);
    assert_eq!(quant_gptq.base.vocab_size, base_30b.vocab_size);
    assert_eq!(quant_gptq.base.vision_hidden, base_30b.vision_hidden);
}

// ============================================================================
// 11. Validation rejects invalid configs
// ============================================================================

#[test]
fn test_granite_docling_rejects_zero_patch_size() {
    let mut cfg = GraniteDoclingConfig::default_258m();
    cfg.patch_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_granite_docling_rejects_non_divisible_image_size() {
    let mut cfg = GraniteDoclingConfig::default_258m();
    cfg.image_size = 513; // not divisible by 16
    assert!(cfg.validate().is_err());
}

#[test]
fn test_granite_docling_rejects_bad_attention() {
    let mut cfg = GraniteDoclingConfig::default_258m();
    cfg.decoder_heads = 7; // 768 not divisible by 7
    assert!(cfg.validate().is_err());
}

#[test]
fn test_glm_ocr_rejects_zero_heads() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.num_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_glm_ocr_rejects_zero_kv_heads() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.num_kv_heads = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_glm_ocr_rejects_non_divisible_heads() {
    let mut cfg = GlmOcrConfig::preset_900m();
    cfg.num_heads = 5; // 1536 not divisible by 5
    assert!(cfg.validate().is_err());
}

#[test]
fn test_table_transformer_rejects_zero_hidden() {
    let mut cfg = TableTransformerConfig::preset_detection();
    cfg.hidden_dim = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_table_transformer_rejects_bad_attention() {
    let mut cfg = TableTransformerConfig::preset_detection();
    cfg.num_heads = 7; // 256 not divisible by 7
    assert!(cfg.validate().is_err());
}

#[test]
fn test_table_transformer_rejects_zero_queries() {
    let mut cfg = TableTransformerConfig::preset_detection();
    cfg.num_queries = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen3_vl_rejects_moe_zero_active() {
    let mut cfg = Qwen3VLConfig::preset_30b_a3b();
    cfg.active_experts = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen3_vl_rejects_moe_active_exceeds_total() {
    let mut cfg = Qwen3VLConfig::preset_30b_a3b();
    cfg.active_experts = cfg.num_experts + 1;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen3_vl_rejects_zero_vision_patch() {
    let mut cfg = Qwen3VLConfig::preset_2b();
    cfg.vision_patch_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_qwen3_vl_rejects_zero_temporal_patch() {
    let mut cfg = Qwen3VLConfig::preset_2b();
    cfg.vision_temporal_patch = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_paddle_ocr_vl_rejects_zero_decoder_hidden() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.decoder_hidden = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_paddle_ocr_vl_rejects_indivisible_heads() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.num_heads = 16;
    cfg.num_kv_heads = 3; // 16 % 3 != 0
    assert!(cfg.validate().is_err());
}

#[test]
fn test_paddle_ocr_vl_rejects_zero_vocab() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.vocab_size = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_paddle_ocr_vl_rejects_zero_head_dim() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.head_dim = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_paddle_ocr_vl_rejects_zero_decoder_layers() {
    let mut cfg = PaddleOcrVlConfig::default_vl();
    cfg.num_decoder_layers = 0;
    assert!(cfg.validate().is_err());
}

#[test]
fn test_firered_ocr_rejects_zero_max_tokens() {
    let mut cfg = FireRedOcrConfig::preset_2b();
    cfg.max_output_tokens = 0;
    assert!(cfg.validate().is_err());
}

// ============================================================================
// 12. DocLayout-YOLO specific: backbone channels and neck channels
// ============================================================================

#[test]
fn test_doclayout_yolo_neck_channels() {
    let cfg = DocLayoutYoloConfig::default();
    let neck = cfg.neck_channels();
    // neck_channels returns [c2, c3, c4] = [64, 128, 256]
    assert_eq!(neck, [64, 128, 256]);
    for &ch in &neck {
        assert!(ch > 0);
    }
}

#[test]
fn test_doclayout_yolo_backbone_progressive_expansion() {
    let cfg = DocLayoutYoloConfig::default();
    // Each stage should have >= the previous stage's channels
    for i in 1..cfg.backbone_channels.len() {
        assert!(
            cfg.backbone_channels[i] >= cfg.backbone_channels[i - 1],
            "backbone channels should progressively expand: stage {} ({}) < stage {} ({})",
            i,
            cfg.backbone_channels[i],
            i - 1,
            cfg.backbone_channels[i - 1]
        );
    }
}

// ============================================================================
// 13. GLM-OCR MTP depth
// ============================================================================

#[test]
fn test_glm_ocr_mtp_depth() {
    let cfg = GlmOcrConfig::preset_900m();
    assert!(
        cfg.mtp_depth > 0,
        "GLM-OCR should have MTP depth > 0 for speculative decoding"
    );
}

// ============================================================================
// 14. Cross-model vocab consistency for shared tokenizers
// ============================================================================

#[test]
fn test_qwen3_family_vocab_consistent() {
    // All Qwen3-VL variants should share the same vocab size
    let v2b = Qwen3VLConfig::preset_2b().vocab_size;
    let v7b = Qwen3VLConfig::preset_7b().vocab_size;
    let v30b = Qwen3VLConfig::preset_30b_a3b().vocab_size;

    assert_eq!(v2b, v7b, "2B and 7B should share vocab");
    assert_eq!(v7b, v30b, "7B and 30B should share vocab");

    // Quant presets should preserve vocab
    let q_gptq = Qwen3VLQuantConfig::preset_30b_a3b_gptq();
    let q_awq = Qwen3VLQuantConfig::preset_30b_a3b_awq();
    assert_eq!(q_gptq.base.vocab_size, v30b);
    assert_eq!(q_awq.base.vocab_size, v30b);
}

// ============================================================================
// 15. Head dimension consistency
// ============================================================================

#[test]
fn test_head_dim_consistency() {
    // Granite-Docling
    let g = GraniteDoclingConfig::default_258m();
    assert_eq!(g.head_dim(), g.decoder_hidden / g.decoder_heads);

    // GLM-OCR
    let glm = GlmOcrConfig::preset_900m();
    assert_eq!(glm.head_dim(), glm.hidden_size / glm.num_heads);

    // Qwen3-VL
    for cfg in [
        Qwen3VLConfig::preset_2b(),
        Qwen3VLConfig::preset_7b(),
        Qwen3VLConfig::preset_30b_a3b(),
    ] {
        assert_eq!(cfg.head_dim(), cfg.hidden_size / cfg.num_heads);
    }

    // FireRed-OCR delegates to base
    let fr = FireRedOcrConfig::preset_2b();
    assert_eq!(fr.head_dim(), fr.base_config.head_dim());
}
