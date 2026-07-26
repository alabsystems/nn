// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for WeightNameMapper trait and HfToNnMapper.

use std::collections::HashMap;

use super::{HfToNnMapper, WeightNameMapper};
use crate::dyn_tensor::DynTensor;
use crate::var_builder::VarBuilder;
use crate::{DType, Device};

// -- HfToNnMapper basic rules ------------------------------------------------

#[test]
fn test_empty_mapper_is_identity() {
    let mapper = HfToNnMapper::new();
    assert_eq!(mapper.map_name("model.weight"), "model.weight");
    assert_eq!(mapper.map_name(""), "");
    assert_eq!(mapper.map_name("weight"), "weight");
}

#[test]
fn test_prefix_rule_replaces_prefix() {
    let mapper = HfToNnMapper::new().with_prefix_rule("model.layers", "encoder.layer");
    assert_eq!(
        mapper.map_name("encoder.layer.0.weight"),
        "model.layers.0.weight"
    );
}

#[test]
fn test_prefix_rule_requires_segment_boundary() {
    // "encoder.layer" should NOT match "encoder.layer_extra.0.weight"
    let mapper = HfToNnMapper::new().with_prefix_rule("model.layers", "encoder.layer");
    // "encoder.layer_extra" does not start with "encoder.layer" followed by "." or end
    assert_eq!(
        mapper.map_name("encoder.layer_extra.0.weight"),
        "encoder.layer_extra.0.weight"
    );
}

#[test]
fn test_prefix_rule_first_match_wins() {
    let mapper = HfToNnMapper::new()
        .with_prefix_rule("first", "enc")
        .with_prefix_rule("second", "enc");
    assert_eq!(mapper.map_name("enc.weight"), "first.weight");
}

#[test]
fn test_segment_rule_replaces_segment() {
    let mapper = HfToNnMapper::new().with_segment_rule("self_attn", "attention");
    assert_eq!(
        mapper.map_name("layer.0.attention.weight"),
        "layer.0.self_attn.weight"
    );
}

#[test]
fn test_segment_rule_multiple_segments() {
    let mapper = HfToNnMapper::new()
        .with_segment_rule("self_attn", "attention")
        .with_segment_rule("q_proj", "q");
    assert_eq!(
        mapper.map_name("layer.0.attention.q.weight"),
        "layer.0.self_attn.q_proj.weight"
    );
}

#[test]
fn test_segment_rule_no_partial_match() {
    // "attention_heads" should NOT be affected by a rule for "attention"
    let mapper = HfToNnMapper::new().with_segment_rule("self_attn", "attention");
    assert_eq!(
        mapper.map_name("layer.attention_heads.weight"),
        "layer.attention_heads.weight"
    );
}

#[test]
fn test_suffix_rule_appends_suffix() {
    let mapper = HfToNnMapper::new().with_suffix_rule("_proj", &["q", "k", "v", "o"]);
    assert_eq!(mapper.map_name("layer.q.weight"), "layer.q_proj.weight");
    assert_eq!(mapper.map_name("layer.k.weight"), "layer.k_proj.weight");
    assert_eq!(mapper.map_name("layer.v.weight"), "layer.v_proj.weight");
    assert_eq!(mapper.map_name("layer.o.weight"), "layer.o_proj.weight");
}

#[test]
fn test_suffix_rule_no_match_leaves_unchanged() {
    let mapper = HfToNnMapper::new().with_suffix_rule("_proj", &["q", "k"]);
    // "bias" is not in the base_segments list
    assert_eq!(mapper.map_name("layer.bias"), "layer.bias");
    // "weight" is not in the list either
    assert_eq!(mapper.map_name("layer.weight"), "layer.weight");
}

#[test]
fn test_exact_overrides_take_precedence() {
    let mut overrides = HashMap::new();
    overrides.insert(
        "special.weight".to_string(),
        "checkpoint.special_weight".to_string(),
    );
    let mapper = HfToNnMapper::new()
        .with_segment_rule("other", "special") // Would normally apply
        .with_exact_overrides(overrides);
    assert_eq!(
        mapper.map_name("special.weight"),
        "checkpoint.special_weight"
    );
}

#[test]
fn test_combined_prefix_and_segment_rules() {
    let mapper = HfToNnMapper::new()
        .with_prefix_rule("model.layers", "encoder.layer")
        .with_segment_rule("self_attn", "attention")
        .with_segment_rule("q_proj", "q");
    assert_eq!(
        mapper.map_name("encoder.layer.0.attention.q.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
}

// -- Pre-built mapper tests ---------------------------------------------------

#[test]
fn test_siglip2_granite_docling_mapper() {
    let mapper = HfToNnMapper::siglip2_granite_docling();
    // NN uses bare names (no "model.vision_model." prefix)
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
fn test_decoder_transformer_mapper() {
    let mapper = HfToNnMapper::decoder_transformer();
    // NN shorter names -> HF names
    assert_eq!(
        mapper.map_name("layers.0.attn.q.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.attn.k.weight"),
        "model.layers.0.self_attn.k_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.attn.v.weight"),
        "model.layers.0.self_attn.v_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.attn.o.weight"),
        "model.layers.0.self_attn.o_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.ln1.weight"),
        "model.layers.0.input_layernorm.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.ln2.weight"),
        "model.layers.0.post_attention_layernorm.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.mlp.gate.weight"),
        "model.layers.0.mlp.gate_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.mlp.up.weight"),
        "model.layers.0.mlp.up_proj.weight"
    );
    assert_eq!(
        mapper.map_name("layers.0.mlp.down.weight"),
        "model.layers.0.mlp.down_proj.weight"
    );
}

#[test]
fn test_qwen3_mapper_is_identity() {
    let mapper = HfToNnMapper::qwen3();
    // Qwen3 model code already uses HF naming
    assert_eq!(
        mapper.map_name("model.layers.0.self_attn.q_proj.weight"),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("model.embed_tokens.weight"),
        "model.embed_tokens.weight"
    );
}

// -- VarBuilder integration ---------------------------------------------------

#[test]
fn test_varbuilder_with_weight_name_mapper() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        DynTensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &Device::Cpu).unwrap(),
    );

    let mapper = HfToNnMapper::new()
        .with_prefix_rule("model.layers", "encoder.layer")
        .with_segment_rule("self_attn", "attention")
        .with_segment_rule("q_proj", "q");

    let vb =
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    // NN model requests with its own naming convention
    let t = vb
        .pp("encoder")
        .pp("layer")
        .pp("0")
        .pp("attention")
        .get(&[2, 2], "q.weight")
        .unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_varbuilder_with_weight_name_mapper_contains_tensor() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.vision_model.encoder.layers.0.weight".to_string(),
        DynTensor::new(&[1.0], &[1], &Device::Cpu).unwrap(),
    );

    let mapper = HfToNnMapper::siglip2_granite_docling();
    let vb =
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    assert!(vb
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .contains_tensor("weight"));
    assert!(!vb
        .pp("encoder")
        .pp("layers")
        .pp("0")
        .contains_tensor("bias"));
}

#[test]
fn test_varbuilder_mapper_propagates_through_pp() {
    let mut tensors = HashMap::new();
    tensors.insert(
        "model.layers.0.self_attn.weight".to_string(),
        DynTensor::new(&[5.0], &[1], &Device::Cpu).unwrap(),
    );

    let mapper = HfToNnMapper::new()
        .with_prefix_rule("model.layers", "layers")
        .with_segment_rule("self_attn", "attn");

    let vb =
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu).with_weight_name_mapper(mapper);

    let child = vb.pp("layers").pp("0").pp("attn");
    assert!(child.has_name_mapping());
    let t = child.get(&[1], "weight").unwrap();
    assert_eq!(t.to_flat_vec::<f32>().unwrap(), vec![5.0]);
}

// -- Real-world HF weight name examples ---------------------------------------

#[test]
fn test_hf_llama_style_weight_names() {
    // Llama/Qwen/Mistral all use this pattern
    let mapper = HfToNnMapper::decoder_transformer();

    let hf_names = [
        (
            "layers.0.attn.q.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ),
        (
            "layers.0.attn.k.weight",
            "model.layers.0.self_attn.k_proj.weight",
        ),
        (
            "layers.0.attn.v.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ),
        (
            "layers.0.attn.o.weight",
            "model.layers.0.self_attn.o_proj.weight",
        ),
        (
            "layers.0.mlp.gate.weight",
            "model.layers.0.mlp.gate_proj.weight",
        ),
        (
            "layers.0.mlp.up.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
        (
            "layers.0.mlp.down.weight",
            "model.layers.0.mlp.down_proj.weight",
        ),
        (
            "layers.0.ln1.weight",
            "model.layers.0.input_layernorm.weight",
        ),
        (
            "layers.0.ln2.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ),
    ];

    for (nn_name, expected_hf_name) in &hf_names {
        assert_eq!(
            mapper.map_name(nn_name),
            *expected_hf_name,
            "failed for NN name: {nn_name}"
        );
    }
}

#[test]
fn test_hf_granite_docling_weight_names() {
    let mapper = HfToNnMapper::siglip2_granite_docling();

    let pairs = [
        (
            "encoder.layers.0.self_attn.q_proj.weight",
            "model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        ),
        (
            "encoder.layers.0.self_attn.k_proj.weight",
            "model.vision_model.encoder.layers.0.self_attn.k_proj.weight",
        ),
        (
            "encoder.layers.0.layer_norm1.weight",
            "model.vision_model.encoder.layers.0.layer_norm1.weight",
        ),
        (
            "encoder.layers.0.mlp.fc1.weight",
            "model.vision_model.encoder.layers.0.mlp.fc1.weight",
        ),
        (
            "embeddings.patch_embedding.weight",
            "model.vision_model.embeddings.patch_embedding.weight",
        ),
        (
            "post_layernorm.weight",
            "model.vision_model.post_layernorm.weight",
        ),
    ];

    for (nn_name, expected_hf) in &pairs {
        assert_eq!(
            mapper.map_name(nn_name),
            *expected_hf,
            "failed for: {nn_name}"
        );
    }
}

#[test]
fn test_custom_mapper_for_whisper() {
    // Whisper uses encoder.layers.{i}.self_attn_layer_norm and similar
    let mapper = HfToNnMapper::new()
        .with_segment_rule("self_attn_layer_norm", "attn_ln")
        .with_segment_rule("final_layer_norm", "final_ln")
        .with_segment_rule("encoder_attn", "cross_attn")
        .with_segment_rule("encoder_attn_layer_norm", "cross_attn_ln");

    assert_eq!(
        mapper.map_name("encoder.layers.0.attn_ln.weight"),
        "encoder.layers.0.self_attn_layer_norm.weight"
    );
    assert_eq!(
        mapper.map_name("decoder.layers.0.cross_attn.q.weight"),
        "decoder.layers.0.encoder_attn.q.weight"
    );
}

// -- Description --------------------------------------------------------------

#[test]
fn test_description() {
    let mapper = HfToNnMapper::new().with_description("test mapper");
    assert_eq!(mapper.description(), "test mapper");
}

#[test]
fn test_default_description() {
    let mapper = HfToNnMapper::new();
    assert_eq!(mapper.description(), "HfToNnMapper");
}

// -- Edge cases ---------------------------------------------------------------

#[test]
fn test_single_segment_name() {
    let mapper = HfToNnMapper::new().with_segment_rule("bias_hf", "bias");
    assert_eq!(mapper.map_name("bias"), "bias_hf");
}

#[test]
fn test_empty_prefix_rule_adds_prefix() {
    // NN model has no prefix, HF has "model." prefix.
    // with_prefix_rule("model", "") means: nn_prefix is empty, hf_prefix is "model".
    // Any NN name gets "model." prepended.
    let mapper = HfToNnMapper::new().with_prefix_rule("model", "");
    assert_eq!(mapper.map_name("encoder.weight"), "model.encoder.weight");
    // Empty string maps to just the HF prefix:
    assert_eq!(mapper.map_name(""), "model");
}

#[test]
fn test_prefix_rule_exact_match_no_trailing() {
    let mapper = HfToNnMapper::new().with_prefix_rule("hf_embed", "embed");
    // "embed" matches and rest is "" (empty, which satisfies rest.is_empty())
    assert_eq!(mapper.map_name("embed"), "hf_embed");
    // "embed.weight" matches (rest starts with ".")
    assert_eq!(mapper.map_name("embed.weight"), "hf_embed.weight");
    // "embedding" does NOT match (rest is "ding", doesn't start with ".")
    assert_eq!(mapper.map_name("embedding"), "embedding");
}

#[test]
fn test_numeric_layer_indices_pass_through() {
    let mapper = HfToNnMapper::new().with_segment_rule("self_attn", "attn");
    // Numeric segments like "0", "1", "12" should pass through unchanged
    assert_eq!(
        mapper.map_name("layers.0.attn.weight"),
        "layers.0.self_attn.weight"
    );
    assert_eq!(
        mapper.map_name("layers.12.attn.weight"),
        "layers.12.self_attn.weight"
    );
}

#[test]
fn test_chaining_multiple_mappers_via_rules() {
    // A complex real-world scenario: model with custom naming
    let mapper = HfToNnMapper::new()
        .with_prefix_rule("backbone.encoder.layers", "enc.blocks")
        .with_segment_rule("self_attention", "sa")
        .with_segment_rule("feed_forward", "ff")
        .with_segment_rule("layer_norm_1", "ln1")
        .with_segment_rule("layer_norm_2", "ln2")
        .with_suffix_rule("_proj", &["q", "k", "v", "out"]);

    assert_eq!(
        mapper.map_name("enc.blocks.0.sa.q.weight"),
        "backbone.encoder.layers.0.self_attention.q_proj.weight"
    );
    assert_eq!(
        mapper.map_name("enc.blocks.3.ff.weight"),
        "backbone.encoder.layers.3.feed_forward.weight"
    );
    assert_eq!(
        mapper.map_name("enc.blocks.0.ln1.weight"),
        "backbone.encoder.layers.0.layer_norm_1.weight"
    );
}
