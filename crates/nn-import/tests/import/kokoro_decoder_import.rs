// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Integration test: import Kokoro decoder (vocoder trainable path) via `import_model()`.
//!
//! Requires pre-exported model files at `models/kokoro-82m/decoder/`:
//! - `graph.json` (torch.export JSON, 765 nodes, 17 aten op types)
//! - `weights.safetensors` (250 tensors from Generator with weight_norm removed)
//!
//! The vocoder was exported via `scripts/export_kokoro_decoder.py`.

use nn_core::dyn_tensor::trace::TraceOp;
use nn_import::import_model;

fn kokoro_decoder_dir() -> std::path::PathBuf {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    workspace.join("models").join("kokoro-82m").join("decoder")
}

#[test]
fn test_kokoro_decoder_import_model_structure() {
    let dir = kokoro_decoder_dir();
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");

    if !graph_path.exists() || !weights_path.exists() {
        eprintln!(
            "SKIP: Kokoro decoder model files not found at {}",
            dir.display()
        );
        return;
    }

    let imported = import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    // Basic structure: 3 inputs (x, s, har), 2 outputs (spec, phase).
    assert_eq!(
        imported.num_user_inputs, 3,
        "vocoder has 3 inputs (x, s, har)"
    );
    assert_eq!(
        imported.output_names.len(),
        2,
        "vocoder has 2 outputs (spec, phase)"
    );

    // Count ops by type.
    let nodes = imported.graph.nodes();
    let count = |pred: &dyn Fn(&TraceOp) -> bool| nodes.iter().filter(|n| pred(n.op())).count();

    // Conv1d: 51 in the graph (noise_convs + resblock convs + conv_post).
    let conv1d = count(&|op| matches!(op, TraceOp::Conv1d { .. }));
    assert!(conv1d >= 40, "expected >= 40 Conv1d ops, got {conv1d}");

    // ConvTranspose1d: 2 upsample layers.
    let conv_t = count(&|op| matches!(op, TraceOp::ConvTranspose1d { .. }));
    assert_eq!(
        conv_t, 2,
        "expected 2 ConvTranspose1d (2 upsampling stages)"
    );

    // InstanceNorm: 48 from AdaIN1d blocks.
    let instnorm = count(&|op| matches!(op, TraceOp::InstanceNorm { .. }));
    assert!(
        instnorm >= 40,
        "expected >= 40 InstanceNorm ops (AdaIN), got {instnorm}"
    );

    // Linear: 48 from AdaIN1d fc layers.
    let linear = count(&|op| matches!(op, TraceOp::Linear { .. }));
    assert!(
        linear >= 40,
        "expected >= 40 Linear ops (AdaIN fc), got {linear}"
    );

    // Narrow: 48*2 = 96 from chunk decomposition (each chunk(2) → 2 Narrows).
    let narrow = count(&|op| matches!(op, TraceOp::Narrow { .. }));
    assert!(
        narrow >= 80,
        "expected >= 80 Narrow ops (from chunk decomposition), got {narrow}"
    );

    // ReflectionPad1d: 1 from generator.reflection_pad.
    let refpad = count(&|op| matches!(op, TraceOp::ReflectionPad1d { .. }));
    assert_eq!(refpad, 1, "expected 1 ReflectionPad1d");

    // Sin: 49 (48 from Snake activation + 1 phase output).
    let sin = count(&|op| matches!(op, TraceOp::Sin));
    assert!(sin >= 48, "expected >= 48 Sin ops, got {sin}");

    eprintln!(
        "Kokoro decoder imported: {} nodes, {} Conv1d, {} ConvTranspose1d, \
         {} InstanceNorm, {} Narrow (from chunk)",
        nodes.len(),
        conv1d,
        conv_t,
        instnorm,
        narrow
    );
}
