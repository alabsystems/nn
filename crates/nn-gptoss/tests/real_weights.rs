// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Real-weight loading and smoke tests for Context-1.
//!
//! These tests require:
//! - `CONTEXT1_WEIGHTS` env var pointing to `model.safetensors`
//! - `CONTEXT1_TOKENIZER` env var pointing to `tokenizer.json`
//! - Config file at `./nn/weights/context-1/config.json`
//!
//! Tests skip gracefully when env vars are unset or files are missing.

use std::env;
use std::path::Path;

fn weights_path() -> Option<String> {
    env::var("CONTEXT1_WEIGHTS").ok()
}

fn tokenizer_path() -> Option<String> {
    env::var("CONTEXT1_TOKENIZER").ok()
}

/// Load model weights from safetensors using mmap to avoid 39GB heap allocation.
fn load_model_mmap(
    path: &str,
    cfg: nn_gptoss::GptOssConfig,
) -> nn_core::Result<nn_gptoss::GptOssModel> {
    use memmap2::Mmap;
    use nn_core::var_builder::VarBuilder;
    use nn_core::{DType, Device};
    use std::fs::File;

    let file = File::open(path)
        .map_err(|e| nn_core::TensorError::Unsupported(format!("failed to open {path}: {e}")))?;
    // SAFETY: Read-only mmap, file held open for mmap duration.
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| nn_core::TensorError::Unsupported(format!("failed to mmap {path}: {e}")))?;
    let tensors = nn_core::load_safetensors_from_bytes(&mmap)?;
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    nn_gptoss::GptOssModel::load(&vb, cfg)
}

/// Verify our preset matches the downloaded config.json.
#[test]
fn test_config_matches_downloaded() {
    let config_path = "./nn/weights/context-1/config.json";
    if !Path::new(config_path).exists() {
        eprintln!("Skipping: config.json not found at {config_path}");
        return;
    }
    let config_str = std::fs::read_to_string(config_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let cfg = nn_gptoss::GptOssConfig::gptoss_20b();

    assert_eq!(
        cfg.hidden_size,
        json["hidden_size"].as_u64().unwrap() as usize
    );
    assert_eq!(
        cfg.num_hidden_layers,
        json["num_hidden_layers"].as_u64().unwrap() as usize
    );
    assert_eq!(
        cfg.num_key_value_heads,
        json["num_key_value_heads"].as_u64().unwrap() as usize
    );
    assert_eq!(
        cfg.vocab_size,
        json["vocab_size"].as_u64().unwrap() as usize
    );
    assert_eq!(
        cfg.num_local_experts,
        json["num_local_experts"].as_u64().unwrap() as usize
    );
    assert_eq!(
        cfg.experts_per_token,
        json["num_experts_per_tok"].as_u64().unwrap() as usize
    );
    assert_eq!(cfg.head_dim, json["head_dim"].as_u64().unwrap() as usize);
    assert_eq!(
        cfg.sliding_window,
        json["sliding_window"].as_u64().unwrap() as usize
    );
    assert_eq!(
        cfg.eos_token_id,
        json["eos_token_id"].as_u64().unwrap() as usize
    );
    assert_eq!(
        cfg.num_attention_heads,
        json["num_attention_heads"].as_u64().unwrap() as usize
    );
    assert_eq!(cfg.layer_types.len(), cfg.num_hidden_layers);
}

/// Verify tokenizer.json exists and has expected size (>1MB for 201K-token vocab).
#[test]
fn test_tokenizer_file_exists() {
    let path = match tokenizer_path() {
        Some(p) => p,
        None => {
            let default = "./nn/weights/context-1/tokenizer.json";
            if !Path::new(default).exists() {
                eprintln!("Skipping: CONTEXT1_TOKENIZER not set and default not found");
                return;
            }
            default.to_string()
        }
    };
    let metadata = std::fs::metadata(&path).unwrap();
    assert!(
        metadata.len() > 1_000_000,
        "tokenizer should be >1MB, got {} bytes",
        metadata.len()
    );
}

/// Verify model.safetensors exists and has expected size (>30GB for 20B params).
#[test]
fn test_weight_file_exists() {
    let Some(path) = weights_path() else {
        eprintln!("Skipping: CONTEXT1_WEIGHTS not set");
        return;
    };
    if !Path::new(&path).exists() {
        eprintln!("Skipping: path does not exist: {path}");
        return;
    }
    let metadata = std::fs::metadata(&path).unwrap();
    assert!(
        metadata.len() > 30_000_000_000,
        "should be >30GB, got {} bytes",
        metadata.len()
    );
}

/// Verify safetensors tensor names and shapes match our weight loading code.
#[test]
fn test_safetensors_tensor_names() {
    let Some(path) = weights_path() else {
        eprintln!("Skipping: CONTEXT1_WEIGHTS not set");
        return;
    };
    if !Path::new(&path).exists() {
        eprintln!("Skipping: path does not exist: {path}");
        return;
    }

    use memmap2::Mmap;
    use std::fs::File;

    let file = File::open(&path).unwrap();
    let mmap = unsafe { Mmap::map(&file) }.unwrap();
    let st = safetensors::SafeTensors::deserialize(&mmap).unwrap();
    let names: Vec<String> = st.names().iter().map(ToString::to_string).collect();
    eprintln!("Total tensors: {}", names.len());

    // Structural tensors
    assert!(names.contains(&"model.embed_tokens.weight".to_string()));
    assert!(names.contains(&"model.norm.weight".to_string()));
    assert!(names.contains(&"lm_head.weight".to_string()));

    // Layer 0 attention
    for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
        assert!(names.contains(&format!("model.layers.0.self_attn.{proj}.weight")));
        assert!(names.contains(&format!("model.layers.0.self_attn.{proj}.bias")));
    }
    assert!(names.contains(&"model.layers.0.self_attn.sinks".to_string()));

    // Layer 0 MoE (fused format)
    assert!(names.contains(&"model.layers.0.mlp.router.weight".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.router.bias".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.experts.gate_up_proj".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.experts.gate_up_proj_bias".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.experts.down_proj".to_string()));
    assert!(names.contains(&"model.layers.0.mlp.experts.down_proj_bias".to_string()));
    assert!(names.contains(&"model.layers.0.input_layernorm.weight".to_string()));
    assert!(names.contains(&"model.layers.0.post_attention_layernorm.weight".to_string()));

    // Shape verification
    assert_eq!(
        st.tensor("model.embed_tokens.weight").unwrap().shape(),
        &[201_088, 2880]
    );
    assert_eq!(
        st.tensor("lm_head.weight").unwrap().shape(),
        &[201_088, 2880]
    );
    assert_eq!(
        st.tensor("model.layers.0.self_attn.q_proj.weight")
            .unwrap()
            .shape(),
        &[4096, 2880]
    );
    assert_eq!(
        st.tensor("model.layers.0.self_attn.k_proj.weight")
            .unwrap()
            .shape(),
        &[512, 2880]
    );
    assert_eq!(
        st.tensor("model.layers.0.self_attn.v_proj.weight")
            .unwrap()
            .shape(),
        &[512, 2880]
    );
    assert_eq!(
        st.tensor("model.layers.0.self_attn.o_proj.weight")
            .unwrap()
            .shape(),
        &[2880, 4096]
    );
    assert_eq!(
        st.tensor("model.layers.0.self_attn.sinks").unwrap().shape(),
        &[64]
    );
    assert_eq!(
        st.tensor("model.layers.0.mlp.router.weight")
            .unwrap()
            .shape(),
        &[32, 2880]
    );
    assert_eq!(
        st.tensor("model.layers.0.mlp.router.bias").unwrap().shape(),
        &[32]
    );
    assert_eq!(
        st.tensor("model.layers.0.mlp.experts.gate_up_proj")
            .unwrap()
            .shape(),
        &[32, 2880, 5760]
    );
    assert_eq!(
        st.tensor("model.layers.0.mlp.experts.gate_up_proj_bias")
            .unwrap()
            .shape(),
        &[32, 5760]
    );
    assert_eq!(
        st.tensor("model.layers.0.mlp.experts.down_proj")
            .unwrap()
            .shape(),
        &[32, 2880, 2880]
    );
    assert_eq!(
        st.tensor("model.layers.0.mlp.experts.down_proj_bias")
            .unwrap()
            .shape(),
        &[32, 2880]
    );

    // All 24 layers present
    for i in 0..24 {
        assert!(names.contains(&format!("model.layers.{i}.input_layernorm.weight")));
    }
    eprintln!("All tensor names and shapes verified.");
}

/// Load model and run forward pass via mmap.
///
/// Requires >= 80GB system RAM (78GB for F32 weights + working memory).
/// On 64GB machines, this test will be killed by the OOM killer.
/// Set CONTEXT1_WEIGHTS env var to run.
#[test]
fn test_forward_with_real_weights() {
    let Some(path) = weights_path() else {
        eprintln!("Skipping: CONTEXT1_WEIGHTS not set");
        return;
    };
    if !Path::new(&path).exists() {
        eprintln!("Skipping: path does not exist: {path}");
        return;
    }

    eprintln!("Loading model via mmap from {path} ...");
    let cfg = nn_gptoss::GptOssConfig::gptoss_20b();
    let model =
        load_model_mmap(&path, cfg).expect("model should load from real safetensors via mmap");
    eprintln!("Model loaded successfully.");

    let input_ids = &[1_usize, 2, 3];
    let positions = &[0_usize, 1, 2];
    eprintln!("Running forward pass with {} tokens ...", input_ids.len());
    let logits = model
        .forward(input_ids, positions)
        .expect("forward pass should succeed");

    let dims = logits.dims();
    assert_eq!(dims, &[1, 3, 201_088]);
    eprintln!("Output shape: {dims:?}");

    let flat: Vec<f32> = logits.to_flat_vec().expect("flat vec");
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "found {non_finite} NaN/Inf values");
    eprintln!("All {} logit values are finite.", flat.len());

    assert!(
        !flat.iter().all(|&v| v == 0.0),
        "logits should not be all-zero"
    );

    let first = &flat[..201_088];
    let mut indexed: Vec<(usize, f32)> = first.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    eprintln!("Top-5 predictions for position 0:");
    for (i, (tok, score)) in indexed.iter().take(5).enumerate() {
        eprintln!("  {}: token={}, logit={:.4}", i + 1, tok, score);
    }
}

/// Load model onto GPU via mmap and run forward pass.
///
/// Requires CONTEXT1_WEIGHTS env var and Metal GPU availability.
/// On M4 Max, BF16 weights take ~39GB/2 = ~19.5GB GPU memory.
#[test]
fn test_forward_gpu_bf16() {
    let Some(path) = weights_path() else {
        eprintln!("Skipping: CONTEXT1_WEIGHTS not set");
        return;
    };
    if !Path::new(&path).exists() {
        eprintln!("Skipping: path does not exist: {path}");
        return;
    }

    use memmap2::Mmap;
    use nn_core::var_builder::VarBuilder;
    use nn_core::{DType, Device};
    use std::fs::File;

    let device = Device::metal();
    let dtype = DType::BF16;

    eprintln!("Loading model via mmap to GPU (BF16) from {path} ...");
    let file = File::open(&path).expect("open weights");
    // SAFETY: Read-only mmap, file held open for mmap duration.
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap");
    let tensors = nn_core::load_safetensors_from_bytes(&mmap).expect("parse safetensors");
    let vb = VarBuilder::from_tensors(tensors, dtype, &device);
    let cfg = nn_gptoss::GptOssConfig::gptoss_20b();
    let model = nn_gptoss::GptOssModel::load(&vb, cfg).expect("GPU BF16 load");

    assert_eq!(model.dtype(), dtype);
    assert_eq!(model.device(), device);
    eprintln!(
        "Model loaded on {:?} with dtype {:?}",
        model.device(),
        model.dtype()
    );

    let input_ids = &[1_usize, 2, 3];
    let positions = &[0_usize, 1, 2];
    eprintln!(
        "Running GPU forward pass with {} tokens ...",
        input_ids.len()
    );
    let logits = model.forward(input_ids, positions).expect("GPU forward");

    let dims = logits.dims();
    assert_eq!(dims, &[1, 3, 201_088]);
    eprintln!("GPU output shape: {dims:?}");

    let flat: Vec<f32> = logits.to_flat_vec().expect("flat vec");
    let non_finite = flat.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(
        non_finite, 0,
        "found {non_finite} NaN/Inf values in GPU output"
    );
    assert!(
        !flat.iter().all(|&v| v == 0.0),
        "GPU logits should not be all-zero"
    );
    eprintln!(
        "GPU BF16 forward pass: all {} logit values finite.",
        flat.len()
    );
}

/// Load model using the convenience load_safetensors_to_device API.
///
/// Smoke test for the new device-aware loading method.
#[test]
fn test_load_safetensors_to_device_api() {
    let Some(path) = weights_path() else {
        eprintln!("Skipping: CONTEXT1_WEIGHTS not set");
        return;
    };
    if !Path::new(&path).exists() {
        eprintln!("Skipping: path does not exist: {path}");
        return;
    }

    use nn_core::{DType, Device};

    let device = Device::metal();
    let dtype = DType::BF16;

    eprintln!("Testing load_safetensors_to_device API ...");
    let cfg = nn_gptoss::GptOssConfig::gptoss_20b();
    let model = nn_gptoss::GptOssModel::load_safetensors_to_device(&path, cfg, dtype, &device)
        .expect("load_safetensors_to_device");

    assert_eq!(model.dtype(), dtype);
    assert_eq!(model.device(), device);
    eprintln!(
        "load_safetensors_to_device: model on {:?} dtype {:?}",
        model.device(),
        model.dtype()
    );

    let logits = model.forward(&[1, 2, 3], &[0, 1, 2]).expect("forward");
    assert_eq!(logits.dims(), &[1, 3, 201_088]);
    eprintln!("load_safetensors_to_device: forward pass succeeded.");
}

/// Verify generation_config.json matches our eos_token_id.
#[test]
fn test_generation_config_eos() {
    let gen_config_path = "./nn/weights/context-1/generation_config.json";
    if !Path::new(gen_config_path).exists() {
        eprintln!("Skipping: generation_config.json not found");
        return;
    }
    let config_str = std::fs::read_to_string(gen_config_path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&config_str).unwrap();
    let cfg = nn_gptoss::GptOssConfig::gptoss_20b();

    if let Some(eos) = json["eos_token_id"].as_u64() {
        assert_eq!(cfg.eos_token_id, eos as usize);
    } else if let Some(eos_arr) = json["eos_token_id"].as_array() {
        let ids: Vec<usize> = eos_arr
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        assert!(ids.contains(&cfg.eos_token_id));
    }
}
