// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Tests for the Kokoro TTS pre-vocoder encoder dispatch plan.
//!
//! Part of #1739 AC3 and #1741 P5.

use super::*;
use nn_dsl::DispatchStep;

// --- Step count ---

#[test]
fn test_encoder_step_count() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    assert_eq!(
        plan.len(),
        TOTAL_EXPECTED_STEPS,
        "expected {} steps, got {}",
        TOTAL_EXPECTED_STEPS,
        plan.len()
    );
}

#[test]
fn test_step_count_constant_consistency() {
    assert_eq!(
        PLBERT_EMB_STEPS, 7,
        "PlBert embeddings: 3 Emb + 2 Add + LN + Linear"
    );
    assert_eq!(
        PLBERT_LAYER_STEPS, 14,
        "ALBERT layer: 6 Linear + 2 MatMul + 1 Softmax + 2 Add + 2 Sigmoid + 1 GELU"
    );
    assert_eq!(NUM_ALBERT_LAYERS, 12, "12 shared ALBERT layers");
    assert_eq!(BERT_ENCODER_STEPS, 1, "bert_encoder: 1 Linear");
    assert_eq!(
        TEXT_ENCODER_STEPS, 25,
        "TextEncoder: BiLSTM(24) + Linear(1)"
    );
    assert_eq!(PROSODY_PREDICTOR_STEPS, 52, "ProsodyPredictor: 3×17 + 1");
    assert_eq!(
        F0_ENERGY_PREDICTOR_STEPS, 72,
        "F0Energy: BiLSTM(24) + F0(24) + Energy(24)"
    );
    assert_eq!(
        TOTAL_EXPECTED_STEPS, 325,
        "7 + 168 + 1 + 25 + 52 + 72 = 325"
    );
}

// --- Stage 1: PlBert embeddings ---

#[test]
fn test_plbert_embeddings_structure() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();

    // First 3 steps should be Embedding lookups (word, pos, token_type)
    for (i, step) in plan.iter().enumerate().take(3) {
        assert!(
            matches!(step, DispatchStep::Embedding { .. }),
            "step {i} should be Embedding, got {:?}",
            std::mem::discriminant(step)
        );
    }

    // Word embedding
    if let DispatchStep::Embedding {
        embedding_dim,
        num_indices,
        kernel_name,
        ..
    } = &plan[0]
    {
        assert_eq!(*embedding_dim, ALBERT_EMB_DIM, "word emb dim");
        assert_eq!(*num_indices, 100, "word emb indices = seq_len=100");
        assert!(kernel_name.contains("word"), "kernel name: {kernel_name}");
    }

    // Position embedding
    if let DispatchStep::Embedding { embedding_dim, .. } = &plan[1] {
        assert_eq!(*embedding_dim, ALBERT_EMB_DIM, "pos emb dim");
    }

    // Token type embedding
    if let DispatchStep::Embedding { embedding_dim, .. } = &plan[2] {
        assert_eq!(*embedding_dim, ALBERT_EMB_DIM, "token_type emb dim");
    }

    // Steps 3-4: BinaryAdd (word+pos, then +token_type)
    assert!(
        matches!(&plan[3], DispatchStep::BinaryAdd { .. }),
        "step 3: BinaryAdd"
    );
    assert!(
        matches!(&plan[4], DispatchStep::BinaryAdd { .. }),
        "step 4: BinaryAdd"
    );

    // Step 5: LayerNorm (Sigmoid proxy)
    assert!(
        matches!(&plan[5], DispatchStep::Sigmoid { .. }),
        "step 5: LN Sigmoid"
    );

    // Step 6: Linear(128→768) factorized projection
    if let DispatchStep::Linear {
        in_features,
        out_features,
        ..
    } = &plan[6]
    {
        assert_eq!(*in_features, ALBERT_EMB_DIM, "emb proj in=128");
        assert_eq!(*out_features, ALBERT_HIDDEN, "emb proj out=768");
    } else {
        panic!("step 6 should be Linear (emb proj)");
    }
}

// --- Stage 1 continued: ALBERT layers ---

#[test]
fn test_plbert_layer_0_structure() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let layer_start = PLBERT_EMB_STEPS; // step 7

    // Q Linear
    if let DispatchStep::Linear {
        in_features,
        out_features,
        kernel_name,
        ..
    } = &plan[layer_start]
    {
        assert_eq!(*in_features, ALBERT_HIDDEN, "Q in=768");
        assert_eq!(*out_features, ALBERT_HIDDEN, "Q out=768");
        assert!(kernel_name.contains("_q"), "Q kernel: {kernel_name}");
    } else {
        panic!("layer 0 should start with Q Linear");
    }

    // K, V Linears follow
    assert!(
        matches!(&plan[layer_start + 1], DispatchStep::Linear { .. }),
        "K Linear"
    );
    assert!(
        matches!(&plan[layer_start + 2], DispatchStep::Linear { .. }),
        "V Linear"
    );

    // MatMul(Q, K^T)
    assert!(
        matches!(&plan[layer_start + 3], DispatchStep::MatMul { .. }),
        "QK MatMul"
    );

    // Softmax
    assert!(
        matches!(&plan[layer_start + 4], DispatchStep::Softmax { .. }),
        "Softmax"
    );

    // MatMul(attn, V)
    assert!(
        matches!(&plan[layer_start + 5], DispatchStep::MatMul { .. }),
        "AV MatMul"
    );

    // Dense output projection
    assert!(
        matches!(&plan[layer_start + 6], DispatchStep::Linear { .. }),
        "Dense"
    );

    // Residual + LN
    assert!(
        matches!(&plan[layer_start + 7], DispatchStep::BinaryAdd { .. }),
        "attn residual"
    );
    assert!(
        matches!(&plan[layer_start + 8], DispatchStep::Sigmoid { .. }),
        "attn LN"
    );
}

#[test]
fn test_plbert_layer_0_ffn_structure() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let layer_start = PLBERT_EMB_STEPS; // step 7

    // FFN: up + GELU + down + residual + LN
    if let DispatchStep::Linear {
        in_features,
        out_features,
        ..
    } = &plan[layer_start + 9]
    {
        assert_eq!(*in_features, ALBERT_HIDDEN, "FFN up in=768");
        assert_eq!(*out_features, ALBERT_FFN_DIM, "FFN up out=2048");
    } else {
        panic!("FFN up should be Linear");
    }
    assert!(
        matches!(&plan[layer_start + 10], DispatchStep::Gelu { .. }),
        "GELU"
    );
    assert!(
        matches!(&plan[layer_start + 11], DispatchStep::Linear { .. }),
        "FFN down"
    );
    // FFN residual + LN: BinaryAdd then Sigmoid
    assert!(
        matches!(&plan[layer_start + 12], DispatchStep::BinaryAdd { .. }),
        "FFN residual"
    );
    assert!(
        matches!(&plan[layer_start + 13], DispatchStep::Sigmoid { .. }),
        "FFN LN"
    );
}

#[test]
fn test_all_12_layers_present() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    // Each layer should have its Q Linear with a unique prefix
    for layer_idx in 0..NUM_ALBERT_LAYERS {
        let start = PLBERT_EMB_STEPS + layer_idx * PLBERT_LAYER_STEPS;
        if let DispatchStep::Linear { kernel_name, .. } = &plan[start] {
            let expected_prefix = format!("plbert_l{layer_idx}_q");
            assert_eq!(
                kernel_name, &expected_prefix,
                "layer {layer_idx} Q kernel name"
            );
        } else {
            panic!("layer {layer_idx} should start with Q Linear at step {start}");
        }
    }
}

// --- Stage 2: bert_encoder ---

#[test]
fn test_bert_encoder_structure() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let bert_start = PLBERT_EMB_STEPS + NUM_ALBERT_LAYERS * PLBERT_LAYER_STEPS; // 7 + 156 = 163

    if let DispatchStep::Linear {
        in_features,
        out_features,
        kernel_name,
        ..
    } = &plan[bert_start]
    {
        assert_eq!(*in_features, ALBERT_HIDDEN, "bert_enc in=768");
        assert_eq!(*out_features, D_EN, "bert_enc out=512");
        assert_eq!(kernel_name, "bert_encoder", "bert_encoder kernel name");
    } else {
        panic!("bert_encoder should be Linear at step {bert_start}");
    }
}

// --- Stage 3: TextEncoder ---

#[test]
fn test_text_encoder_starts_with_bilstm() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let te_start = PLBERT_EMB_STEPS + NUM_ALBERT_LAYERS * PLBERT_LAYER_STEPS + BERT_ENCODER_STEPS; // 164

    // BiLSTM forward: first step is Linear(ih)
    if let DispatchStep::Linear {
        in_features,
        out_features,
        kernel_name,
        ..
    } = &plan[te_start]
    {
        assert_eq!(*in_features, D_EN, "BiLSTM fwd ih in=512");
        assert_eq!(
            *out_features,
            4 * (D_EN / 2),
            "BiLSTM fwd ih out=4*256=1024"
        );
        assert!(
            kernel_name.contains("fwd") && kernel_name.contains("ih"),
            "kernel: {kernel_name}"
        );
    } else {
        panic!("TextEncoder should start with forward LSTM ih Linear");
    }
}

#[test]
fn test_text_encoder_ends_with_projection() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let te_end = PLBERT_EMB_STEPS
        + NUM_ALBERT_LAYERS * PLBERT_LAYER_STEPS
        + BERT_ENCODER_STEPS
        + TEXT_ENCODER_STEPS
        - 1; // last step

    if let DispatchStep::Linear {
        in_features,
        out_features,
        kernel_name,
        ..
    } = &plan[te_end]
    {
        assert_eq!(*in_features, D_EN, "text_enc proj in=512");
        assert_eq!(*out_features, D_EN, "text_enc proj out=512");
        assert!(
            kernel_name.contains("text_enc_proj"),
            "kernel: {kernel_name}"
        );
    } else {
        panic!("TextEncoder should end with projection Linear");
    }
}

// --- Stage 4: ProsodyPredictor ---

#[test]
fn test_prosody_predictor_starts_with_conv() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let prosody_start = PLBERT_EMB_STEPS
        + NUM_ALBERT_LAYERS * PLBERT_LAYER_STEPS
        + BERT_ENCODER_STEPS
        + TEXT_ENCODER_STEPS; // 189

    // First ProsodyBlock starts with Conv1d(512, 512, k=3, pad=1)
    if let DispatchStep::Conv1d(p) = &plan[prosody_start] {
        assert_eq!(p.in_channels, D_EN, "prosody_b0 conv in=512");
        assert_eq!(p.out_channels, D_EN, "prosody_b0 conv out=512");
        assert_eq!(p.kernel_size, 3, "prosody_b0 conv k=3");
        assert!(
            p.kernel_name.contains("prosody_b0"),
            "kernel: {}",
            p.kernel_name
        );
    } else {
        panic!("ProsodyPredictor should start with Conv1d");
    }
}

#[test]
fn test_prosody_ends_with_dur_proj() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let prosody_end = PLBERT_EMB_STEPS
        + NUM_ALBERT_LAYERS * PLBERT_LAYER_STEPS
        + BERT_ENCODER_STEPS
        + TEXT_ENCODER_STEPS
        + PROSODY_PREDICTOR_STEPS
        - 1;

    if let DispatchStep::Linear {
        in_features,
        out_features,
        kernel_name,
        ..
    } = &plan[prosody_end]
    {
        assert_eq!(*in_features, D_EN, "dur_proj in=512");
        assert_eq!(*out_features, 1, "dur_proj out=1");
        assert!(kernel_name.contains("dur_proj"), "kernel: {kernel_name}");
    } else {
        panic!("ProsodyPredictor should end with dur_proj Linear");
    }
}

// --- Stage 5: F0EnergyPredictor ---

#[test]
fn test_f0_energy_starts_with_bilstm() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let f0_start = PLBERT_EMB_STEPS
        + NUM_ALBERT_LAYERS * PLBERT_LAYER_STEPS
        + BERT_ENCODER_STEPS
        + TEXT_ENCODER_STEPS
        + PROSODY_PREDICTOR_STEPS; // 241

    // Shared BiLSTM: forward LSTM starts with Linear(ih)
    if let DispatchStep::Linear {
        in_features,
        out_features,
        kernel_name,
        ..
    } = &plan[f0_start]
    {
        assert_eq!(*in_features, D_EN, "f0 BiLSTM fwd ih in=512");
        assert_eq!(
            *out_features,
            4 * F0_HIDDEN,
            "f0 BiLSTM fwd ih out=4*256=1024"
        );
        assert!(
            kernel_name.contains("f0_energy_bilstm") && kernel_name.contains("fwd"),
            "kernel: {kernel_name}"
        );
    } else {
        panic!("F0EnergyPredictor should start with BiLSTM forward ih");
    }
}

#[test]
fn test_f0_energy_ends_with_energy_proj() {
    let (plan, _) = build_kokoro_encoder_dispatch_plan_default();
    let f0_end = TOTAL_EXPECTED_STEPS - 1;

    // Energy projection is Conv1d(k=1) since #3512 (eliminates transpose dispatches).
    if let DispatchStep::Conv1d(p) = &plan[f0_end] {
        assert_eq!(p.in_channels, F0_HIDDEN, "energy_proj in_channels=256");
        assert_eq!(p.out_channels, 1, "energy_proj out_channels=1");
        assert_eq!(p.kernel_size, 1, "energy_proj kernel_size=1");
        assert!(
            p.kernel_name.contains("energy"),
            "kernel: {}",
            p.kernel_name
        );
    } else {
        panic!("F0EnergyPredictor should end with energy proj Conv1d(k=1)");
    }
}

// --- Topology invariance ---

#[test]
fn test_different_seq_lens_same_step_count() {
    let (plan_10, _) = build_kokoro_encoder_dispatch_plan(10);
    let (plan_200, _) = build_kokoro_encoder_dispatch_plan(200);
    assert_eq!(
        plan_10.len(),
        plan_200.len(),
        "topology unchanged by text_tokens"
    );
    assert_eq!(plan_10.len(), TOTAL_EXPECTED_STEPS);
}

#[test]
fn test_default_is_100_tokens() {
    let (plan_default, nodes_default) = build_kokoro_encoder_dispatch_plan_default();
    let (plan_100, nodes_100) = build_kokoro_encoder_dispatch_plan(100);
    assert_eq!(nodes_default, nodes_100);
    assert_eq!(plan_default.len(), plan_100.len());
}

// Step-type distribution and profiling integration tests extracted to
// kokoro_encoder_dispatch_tests_profiling.rs via #[path] submodule.
