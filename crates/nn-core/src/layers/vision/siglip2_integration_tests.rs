// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the dpdf vision pipeline (#2439).
//!
//! These tests exercise end-to-end flows that cross module boundaries:
//! - SigLIP2 + VarBuilder name mapping (Granite-Docling weight pattern)
//! - SigLIP2 + DeepStack fusion pipeline
//! - SigLIP2 with non-trivial (random-seeded) weights for numerical validation
//! - VarBuilder sharp edges with SigLIP2 loading

use super::{SigLip2Config, SigLip2VisionEncoder};
use crate::dyn_tensor::DynTensor;
use crate::layers::vision::{DeepStackFusion, PoolingStrategy};
use crate::layers::{Linear, Module};
use crate::var_builder::VarBuilder;
use crate::{DType, Device};
use std::collections::HashMap;

// -- Helpers ------------------------------------------------------------------

/// Deterministic pseudo-random data for reproducible weight initialization.
fn det_data(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + seed) * 0.01).sin() * 0.1)
        .collect()
}

/// Build a SigLIP2 tensor map with deterministic non-zero weights.
/// Uses small dimensions for fast testing: hidden=32, 2 layers, 2 heads, patch=4, img=8.
fn build_siglip2_nonzero_tensors() -> (SigLip2Config, HashMap<String, DynTensor>) {
    let d = 32_usize;
    let inter = 64_usize;
    let (ch, ps, img, n_layers) = (3, 4, 8, 2);
    let num_patches = (img / ps) * (img / ps);

    let config = SigLip2Config::new(ch, d, n_layers, 2, inter, ps, img, 1e-6).unwrap();

    let make = |s: &[usize], seed: f32| {
        let n: usize = s.iter().product();
        DynTensor::from_vec(det_data(n, seed), s, &Device::Cpu).unwrap()
    };

    let mut t = HashMap::new();
    t.insert(
        "embeddings.patch_embedding.weight".into(),
        make(&[d, ch, ps, ps], 1.0),
    );
    t.insert("embeddings.patch_embedding.bias".into(), make(&[d], 2.0));
    t.insert(
        "embeddings.position_embedding.weight".into(),
        make(&[num_patches, d], 3.0),
    );

    for i in 0..n_layers {
        let pfx = format!("encoder.layers.{i}");
        // LayerNorm weights: near 1.0 to avoid instability
        let ln_w: Vec<f32> = (0..d)
            .map(|j| 1.0 + det_data(1, i as f32 * 100.0 + j as f32)[0] * 0.01)
            .collect();
        let ln_b: Vec<f32> = (0..d)
            .map(|j| det_data(1, i as f32 * 200.0 + j as f32)[0] * 0.01)
            .collect();
        for ln in &["layer_norm1", "layer_norm2"] {
            t.insert(
                format!("{pfx}.{ln}.weight"),
                DynTensor::from_vec(ln_w.clone(), &[d], &Device::Cpu).unwrap(),
            );
            t.insert(
                format!("{pfx}.{ln}.bias"),
                DynTensor::from_vec(ln_b.clone(), &[d], &Device::Cpu).unwrap(),
            );
        }
        for (pi, proj) in ["q_proj", "k_proj", "v_proj", "out_proj"]
            .iter()
            .enumerate()
        {
            t.insert(
                format!("{pfx}.self_attn.{proj}.weight"),
                make(&[d, d], 10.0 + i as f32 * 40.0 + pi as f32 * 10.0),
            );
            t.insert(
                format!("{pfx}.self_attn.{proj}.bias"),
                make(&[d], 50.0 + i as f32 * 40.0 + pi as f32 * 10.0),
            );
        }
        t.insert(
            format!("{pfx}.mlp.fc1.weight"),
            make(&[inter, d], 100.0 + i as f32 * 50.0),
        );
        t.insert(
            format!("{pfx}.mlp.fc1.bias"),
            make(&[inter], 150.0 + i as f32 * 50.0),
        );
        t.insert(
            format!("{pfx}.mlp.fc2.weight"),
            make(&[d, inter], 200.0 + i as f32 * 50.0),
        );
        t.insert(
            format!("{pfx}.mlp.fc2.bias"),
            make(&[d], 250.0 + i as f32 * 50.0),
        );
    }

    // Post-layernorm: near 1.0 weights
    let post_ln_w: Vec<f32> = (0..d)
        .map(|j| 1.0 + det_data(1, 999.0 + j as f32)[0] * 0.01)
        .collect();
    let post_ln_b: Vec<f32> = (0..d)
        .map(|j| det_data(1, 1999.0 + j as f32)[0] * 0.01)
        .collect();
    t.insert(
        "post_layernorm.weight".into(),
        DynTensor::from_vec(post_ln_w, &[d], &Device::Cpu).unwrap(),
    );
    t.insert(
        "post_layernorm.bias".into(),
        DynTensor::from_vec(post_ln_b, &[d], &Device::Cpu).unwrap(),
    );

    (config, t)
}

/// Add a "model.vision_model." prefix to all keys in a tensor map.
/// Simulates a HuggingFace checkpoint where vision weights are nested.
fn prefix_keys(tensors: HashMap<String, DynTensor>, prefix: &str) -> HashMap<String, DynTensor> {
    tensors
        .into_iter()
        .map(|(k, v)| (format!("{prefix}{k}"), v))
        .collect()
}

// -- SigLIP2 + VarBuilder name mapping (Granite-Docling pattern) --------------

/// End-to-end: load SigLIP2 from a tensor map where all keys have a
/// `model.vision_model.` prefix (as in Granite-Docling safetensors).
/// Uses `with_name_mapping` to add the prefix during lookup.
///
/// VarBuilder resolves names by: (1) build the full key from pp() + tensor name,
/// (2) apply the name map. So the map receives the NN name and must return
/// the checkpoint (HF) name.
#[test]
fn test_siglip2_load_with_name_mapping_granite_docling() {
    let (config, tensors) = build_siglip2_nonzero_tensors();
    let hf_tensors = prefix_keys(tensors, "model.vision_model.");

    // The SigLIP2 loader builds key "embeddings.patch_embedding.weight".
    // The checkpoint has "model.vision_model.embeddings.patch_embedding.weight".
    // The name map adds the prefix so the backend can find the key.
    let vb = VarBuilder::from_tensors(hf_tensors, DType::F32, &Device::Cpu)
        .with_name_mapping(|name| format!("model.vision_model.{name}"));

    // This is the key test: loading resolves all weight paths through the mapping.
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        DynTensor::from_vec(det_data(3 * 8 * 8, 42.0), &[1, 3, 8, 8], &Device::Cpu).unwrap();

    let output = encoder.forward(&input, PoolingStrategy::None).unwrap();
    assert_eq!(output.dims(), &[1, 4, 32]); // [B, num_patches, D]
    assert!(!output.any_non_finite().unwrap());
}

/// Same as above but using `vb.pp("model").pp("vision_model")` scoping
/// instead of name mapping. This is the alternative approach: if the
/// caller controls the VarBuilder prefix, they can scope before loading.
#[test]
fn test_siglip2_load_with_pp_scoping() {
    let (config, tensors) = build_siglip2_nonzero_tensors();
    let hf_tensors = prefix_keys(tensors, "model.vision_model.");

    let vb = VarBuilder::from_tensors(hf_tensors, DType::F32, &Device::Cpu);
    let vision_vb = vb.pp("model").pp("vision_model");

    let encoder = SigLip2VisionEncoder::load(&vision_vb, &config).unwrap();

    let input =
        DynTensor::from_vec(det_data(3 * 8 * 8, 42.0), &[1, 3, 8, 8], &Device::Cpu).unwrap();

    let output = encoder.forward(&input, PoolingStrategy::None).unwrap();
    assert_eq!(output.dims(), &[1, 4, 32]);
    assert!(!output.any_non_finite().unwrap());
}

// -- SigLIP2 + DeepStack end-to-end -------------------------------------------

/// Full vision pipeline: SigLIP2 encoder -> forward_deepstack -> DeepStackFusion.
/// This is the exact flow Granite-Docling and Qwen3-VL use.
#[test]
fn test_siglip2_deepstack_fusion_pipeline() {
    let (config, tensors) = build_siglip2_nonzero_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    // DeepStack: extract from both layers, project to output_dim=48
    let hidden = config.hidden_size; // 32
    let num_extract = 2;
    let output_dim = 48;
    let concat_dim = num_extract * hidden; // 64
    let proj_w = DynTensor::from_vec(
        det_data(output_dim * concat_dim, 500.0),
        &[output_dim, concat_dim],
        &Device::Cpu,
    )
    .unwrap();
    let proj_b =
        DynTensor::from_vec(det_data(output_dim, 600.0), &[output_dim], &Device::Cpu).unwrap();
    let proj = Linear::new(proj_w, Some(proj_b)).unwrap();
    let fusion = DeepStackFusion::new(proj, hidden, num_extract, output_dim).unwrap();

    let input =
        DynTensor::from_vec(det_data(2 * 3 * 8 * 8, 42.0), &[2, 3, 8, 8], &Device::Cpu).unwrap();

    // Extract intermediates from layers 0 and 1
    let intermediates = encoder.forward_deepstack(&input, &[0, 1]).unwrap();
    assert_eq!(intermediates.len(), 2);
    for t in &intermediates {
        assert_eq!(t.dims(), &[2, 4, 32]); // [B, num_patches, D]
    }

    // Fuse
    let fused = fusion.forward_multi(&intermediates).unwrap();
    assert_eq!(fused.dims(), &[2, 4, output_dim]);
    assert!(!fused.any_non_finite().unwrap());
}

// -- SigLIP2 numerical validation with non-zero weights -----------------------

/// Verify that SigLIP2 with non-zero weights produces non-trivial output.
/// Zero weights produce all-zero output; this test catches accidental
/// weight loading failures that silently produce zero tensors.
#[test]
fn test_siglip2_nonzero_weights_produce_nontrivial_output() {
    let (config, tensors) = build_siglip2_nonzero_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        DynTensor::from_vec(det_data(3 * 8 * 8, 42.0), &[1, 3, 8, 8], &Device::Cpu).unwrap();

    let output = encoder.forward(&input, PoolingStrategy::None).unwrap();
    let data = output.to_flat_vec::<f32>().unwrap();

    // Output should not be all zeros (would indicate weight loading failure).
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs > 1e-6,
        "SigLIP2 output is near-zero with non-zero weights (max_abs={max_abs})"
    );

    // Output should have variance (not a constant vector).
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: f32 = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
    assert!(
        variance > 1e-8,
        "SigLIP2 output has near-zero variance ({variance}), suggesting degenerate computation"
    );
}

/// Verify that different inputs produce different outputs.
/// Catches bugs where the forward pass ignores the input.
#[test]
fn test_siglip2_different_inputs_produce_different_outputs() {
    let (config, tensors) = build_siglip2_nonzero_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input_a =
        DynTensor::from_vec(det_data(3 * 8 * 8, 42.0), &[1, 3, 8, 8], &Device::Cpu).unwrap();
    let input_b =
        DynTensor::from_vec(det_data(3 * 8 * 8, 99.0), &[1, 3, 8, 8], &Device::Cpu).unwrap();

    let out_a = encoder
        .forward(&input_a, PoolingStrategy::None)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();
    let out_b = encoder
        .forward(&input_b, PoolingStrategy::None)
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap();

    let diff: f32 = out_a
        .iter()
        .zip(out_b.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-4,
        "Different inputs produced identical outputs (diff={diff})"
    );
}

/// Verify mean pooling numerically: mean of seq should match pooled output.
#[test]
fn test_siglip2_mean_pooling_numerical() {
    let (config, tensors) = build_siglip2_nonzero_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        DynTensor::from_vec(det_data(3 * 8 * 8, 42.0), &[1, 3, 8, 8], &Device::Cpu).unwrap();

    let full_output = encoder.forward(&input, PoolingStrategy::None).unwrap(); // [1, 4, 32]
    let pooled = encoder.forward(&input, PoolingStrategy::Mean).unwrap(); // [1, 32]

    assert_eq!(full_output.dims(), &[1, 4, 32]);
    assert_eq!(pooled.dims(), &[1, 32]);

    // Manual mean over the sequence dimension
    let full = full_output.to_flat_vec::<f32>().unwrap();
    let d = 32;
    let num_patches = 4;
    let manual_mean: Vec<f32> = (0..d)
        .map(|j| (0..num_patches).map(|p| full[p * d + j]).sum::<f32>() / num_patches as f32)
        .collect();

    let pooled_data = pooled.to_flat_vec::<f32>().unwrap();
    for (j, (m, p)) in manual_mean.iter().zip(pooled_data.iter()).enumerate() {
        assert!(
            (m - p).abs() < 1e-5,
            "Mean pooling mismatch at dim {j}: manual={m}, pooled={p}"
        );
    }
}

// -- SigLIP2 + Module trait consistency ---------------------------------------

/// Verify that Module::forward (PoolingStrategy::None) matches
/// explicit forward(..., PoolingStrategy::None).
#[test]
fn test_siglip2_module_trait_matches_explicit_forward() {
    let (config, tensors) = build_siglip2_nonzero_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        DynTensor::from_vec(det_data(3 * 8 * 8, 42.0), &[1, 3, 8, 8], &Device::Cpu).unwrap();

    let explicit = encoder.forward(&input, PoolingStrategy::None).unwrap();
    let module = Module::forward(&encoder, &input).unwrap();

    let a = explicit.to_flat_vec::<f32>().unwrap();
    let b = module.to_flat_vec::<f32>().unwrap();
    assert_eq!(a.len(), b.len());
    for (va, vb) in a.iter().zip(b.iter()) {
        assert!(
            (va - vb).abs() < 1e-7,
            "Module trait output differs: {va} vs {vb}"
        );
    }
}

// -- SigLIP2 batch consistency ------------------------------------------------

/// Verify that batch processing is consistent: processing inputs separately
/// should give the same results as batching them.
#[test]
fn test_siglip2_batch_consistency() {
    let (config, tensors) = build_siglip2_nonzero_tensors();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let data_a = det_data(3 * 8 * 8, 42.0);
    let data_b = det_data(3 * 8 * 8, 99.0);

    // Process separately
    let input_a = DynTensor::from_vec(data_a.clone(), &[1, 3, 8, 8], &Device::Cpu).unwrap();
    let input_b = DynTensor::from_vec(data_b.clone(), &[1, 3, 8, 8], &Device::Cpu).unwrap();
    let out_a = encoder.forward(&input_a, PoolingStrategy::None).unwrap();
    let out_b = encoder.forward(&input_b, PoolingStrategy::None).unwrap();

    // Process as a batch
    let mut batch_data = data_a;
    batch_data.extend(data_b);
    let input_batch = DynTensor::from_vec(batch_data, &[2, 3, 8, 8], &Device::Cpu).unwrap();
    let out_batch = encoder
        .forward(&input_batch, PoolingStrategy::None)
        .unwrap();
    assert_eq!(out_batch.dims(), &[2, 4, 32]);

    // Compare batch[0] vs separate_a and batch[1] vs separate_b
    let batch_flat = out_batch.to_flat_vec::<f32>().unwrap();
    let a_flat = out_a.to_flat_vec::<f32>().unwrap();
    let b_flat = out_b.to_flat_vec::<f32>().unwrap();

    let d = 32;
    let n_patches = 4;
    let per_sample = n_patches * d;

    for i in 0..per_sample {
        assert!(
            (batch_flat[i] - a_flat[i]).abs() < 1e-5,
            "batch[0] vs separate_a mismatch at {i}: {} vs {}",
            batch_flat[i],
            a_flat[i]
        );
        assert!(
            (batch_flat[per_sample + i] - b_flat[i]).abs() < 1e-5,
            "batch[1] vs separate_b mismatch at {i}: {} vs {}",
            batch_flat[per_sample + i],
            b_flat[i]
        );
    }
}

// -- VarBuilder sharp edges with SigLIP2 --------------------------------------

/// Verify that loading SigLIP2 with a rename map works end-to-end.
/// This tests the `with_rename_map` path (exact key remapping), which is
/// the recommended approach for HuggingFace weight loading.
#[test]
fn test_siglip2_load_with_rename_map() {
    let (config, raw_tensors) = build_siglip2_nonzero_tensors();

    // Build the rename map: each SigLIP2 key -> a differently-named checkpoint key
    let checkpoint_prefix = "checkpoint.vision.";
    let hf_tensors = prefix_keys(raw_tensors, checkpoint_prefix);

    let checkpoint_keys: Vec<String> = hf_tensors.keys().cloned().collect();
    let mut rename = HashMap::new();
    for ck in &checkpoint_keys {
        let nn_key = ck.strip_prefix(checkpoint_prefix).unwrap().to_string();
        rename.insert(nn_key, ck.clone());
    }

    let vb = VarBuilder::from_tensors(hf_tensors, DType::F32, &Device::Cpu).with_rename_map(rename);

    let encoder = SigLip2VisionEncoder::load(&vb, &config).unwrap();

    let input =
        DynTensor::from_vec(det_data(3 * 8 * 8, 42.0), &[1, 3, 8, 8], &Device::Cpu).unwrap();
    let output = encoder.forward(&input, PoolingStrategy::None).unwrap();
    assert_eq!(output.dims(), &[1, 4, 32]);
    assert!(!output.any_non_finite().unwrap());
}

/// Verify that SigLIP2 loading fails gracefully when a weight is missing
/// (even with name mapping, a missing tensor should be clearly reported).
#[test]
fn test_siglip2_load_partial_weights_errors_clearly() {
    let (config, mut tensors) = build_siglip2_nonzero_tensors();

    // Remove one critical weight
    tensors.remove("encoder.layers.0.self_attn.q_proj.weight");

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let err = SigLip2VisionEncoder::load(&vb, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("q_proj") || msg.contains("not found") || msg.contains("TensorNotFound"),
        "Expected clear error about missing q_proj weight, got: {msg}"
    );
}

/// Verify that SigLIP2 loading fails when a weight has the wrong shape.
#[test]
fn test_siglip2_load_wrong_shape_weight_errors() {
    let (config, mut tensors) = build_siglip2_nonzero_tensors();

    // Replace a weight with wrong shape
    tensors.insert(
        "encoder.layers.0.self_attn.q_proj.weight".into(),
        DynTensor::ones(&[16, 16], DType::F32, &Device::Cpu).unwrap(), // Should be [32, 32]
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
    let err = SigLip2VisionEncoder::load(&vb, &config).unwrap_err();
    let msg = format!("{err:?}");
    assert!(
        msg.contains("ShapeMismatch") || msg.contains("shape"),
        "Expected shape mismatch error, got: {msg}"
    );
}
