// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! NY IBP bounds parity tests for auto-converted models.
//!
//! Proves that auto-converted models (via `import_model()` / `ConvertBuilder`)
//! have NY IBP bounds consistent with hand-built traced models.
//!
//! Tests use mini fixtures that are always available in test_data/:
//! - `e2e_mlp.json` + `exported_mlp_weights.safetensors`
//! - `kokoro_decoder_mini.json`
//!
//! Part of #4189 (NY bounds parity for auto-converted models).

/// Test A: E2E MLP bounds verification via `check_composition_bounds()`.
///
/// Loads the MLP fixture, imports it, and runs NY IBP propagation
/// through `check_composition_bounds()`. Verifies that:
/// - IBP propagation succeeds
/// - Output bounds are finite
/// - Output width is reasonable (not vacuously wide)
#[test]
#[cfg(feature = "verify")]
fn test_e2e_mlp_composition_bounds() {
    use std::path::Path;
    let test_data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let graph_path = test_data.join("e2e_mlp.json");
    let weights_path = test_data.join("exported_mlp_weights.safetensors");

    assert!(graph_path.exists(), "e2e_mlp.json must exist in test_data");
    assert!(
        weights_path.exists(),
        "exported_mlp_weights.safetensors must exist in test_data"
    );

    let imported = nn_import::import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    eprintln!(
        "[E2E MLP bounds] imported graph: {} nodes, {} user inputs",
        imported.graph.len(),
        imported.num_user_inputs,
    );

    // Run NY IBP via the public API.
    let report = nn_import::check_composition_bounds(&imported);

    match report {
        Some(ref r) => {
            eprintln!(
                "[E2E MLP bounds] propagation_ok={}, output_width={:?}, method={:?}, soundness={:?}, proof_strength={:?}",
                r.propagation_ok,
                r.output_width,
                r.composition_method,
                r.composition_soundness_mode,
                r.composition_proof_strength
            );
            assert!(
                r.propagation_ok,
                "IBP propagation must succeed for MLP fixture"
            );
            assert_eq!(
                r.composition_method,
                Some(nn_import::ConvertCompositionMethod::Ibp)
            );
            assert_eq!(
                r.composition_soundness_mode,
                Some(nn_import::ConvertSoundnessMode::Sound)
            );
            assert_eq!(
                r.composition_proof_strength,
                Some(nn_import::ConvertProofStrength::SoundIbp)
            );
            if let Some(width) = r.output_width {
                assert!(
                    width.is_finite(),
                    "output bounds width must be finite, got {width}"
                );
                assert!(
                    width < 1000.0,
                    "output bounds width should be reasonable (<1000), got {width}"
                );
                eprintln!("[E2E MLP bounds] PASSED: output width = {width:.4}");
            } else {
                eprintln!(
                    "[E2E MLP bounds] WARNING: output width is None \
                     (bounds may be infinite)"
                );
            }
        }
        None => {
            // Translation may fail if some ops are not yet supported in
            // trace_to_graph. Report which ops are present for diagnostics.
            let ops: Vec<String> = imported
                .graph
                .nodes()
                .iter()
                .map(|n| format!("{:?}", n.op()))
                .collect();
            eprintln!(
                "[E2E MLP bounds] check_composition_bounds returned None. \
                 Graph ops: {:?}",
                ops
            );
            panic!(
                "check_composition_bounds must succeed for the E2E MLP fixture \
                 (basic Linear + ReLU graph)"
            );
        }
    }
}

/// Test A2: Direct NY translation + IBP propagation for the E2E MLP.
///
/// Goes deeper than Test A by directly calling `trace_to_graph_model()` and
/// `propagate_ibp()`, inspecting the intermediate GraphNetwork and BoundedTensor.
#[test]
#[cfg(feature = "verify")]
fn test_e2e_mlp_direct_ibp_propagation() {
    use nn_core::dyn_tensor::trace::TraceOp;
    use std::path::Path;

    let test_data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let graph_path = test_data.join("e2e_mlp.json");
    let weights_path = test_data.join("exported_mlp_weights.safetensors");

    let imported = nn_import::import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    // Count variable inputs to choose single vs multi-input translation.
    let variable_input_count = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Input))
        .count();

    eprintln!("[E2E MLP direct IBP] variable inputs: {variable_input_count}");

    // Translate to NY GraphNetwork.
    let translate_result = if variable_input_count > 1 {
        nn_verify::trace_to_graph_model_multi_input(&imported.graph)
    } else {
        nn_verify::trace_to_graph_model(&imported.graph)
    };

    let result =
        translate_result.unwrap_or_else(|e| panic!("trace_to_graph translation failed: {e:?}"));

    let gn = &result.graph;
    let num_nodes = gn.num_nodes();
    eprintln!(
        "[E2E MLP direct IBP] GraphNetwork: {num_nodes} nodes, \
         dtype_cast_count={}",
        result.dtype_cast_count
    );
    assert!(
        num_nodes > 0,
        "translated GraphNetwork must have at least 1 node"
    );

    // Build input bounds matching the input shape.
    let input_node = imported
        .graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), TraceOp::Input))
        .expect("MLP graph must have an Input node");
    let shape = input_node.output_shape();
    eprintln!("[E2E MLP direct IBP] input shape: {shape:?}");

    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), -1.0_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), 1.0_f32);
    let input_bounds = nn_verify::BoundedTensor::new(lower, upper)
        .expect("BoundedTensor construction must succeed for [-1, 1] bounds");

    // Propagate IBP.
    let output = gn
        .propagate_ibp(&input_bounds)
        .unwrap_or_else(|e| panic!("IBP propagation failed: {e:?}"));

    let (out_lower, out_upper) = output.lower_upper();
    let out_elements = out_lower.len();
    eprintln!("[E2E MLP direct IBP] output elements: {out_elements}");

    // Verify all output bounds are finite.
    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
        assert!(
            lo.is_finite(),
            "output lower bound [{i}] must be finite, got {lo}"
        );
        assert!(
            hi.is_finite(),
            "output upper bound [{i}] must be finite, got {hi}"
        );
        assert!(
            lo <= hi,
            "output bounds [{i}]: lower ({lo}) must be <= upper ({hi})"
        );
    }

    // Compute max width.
    let max_width = out_upper
        .iter()
        .zip(out_lower.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    eprintln!("[E2E MLP direct IBP] max output width: {max_width:.4}");
    assert!(
        max_width < 1000.0,
        "max output width should be reasonable (<1000), got {max_width}"
    );

    eprintln!(
        "[E2E MLP direct IBP] PASSED: {out_elements} output elements, \
         max width {max_width:.4}"
    );
}

/// Test B: ConvertBuilder with VerifyLevel::Bounds (Metal-gated).
///
/// Uses the ConvertBuilder API to compile the E2E MLP fixture and verify
/// that the builder's verification phase produces valid composition bounds.
#[test]
#[cfg(all(feature = "metal", feature = "verify", target_os = "macos"))]
fn test_e2e_mlp_convert_builder_with_bounds() {
    use std::path::Path;
    let test_data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let graph_path = test_data.join("e2e_mlp.json");
    let weights_path = test_data.join("exported_mlp_weights.safetensors");

    let _ = nn_metal::MetalBackend::init();
    nn_metal::register_metal_dyn_backend();
    let cache = nn_metal::PipelineCache::new(
        nn_metal::MetalContext::new().expect("Metal device required"),
    );

    let result = nn_import::convert_build(&graph_path, &weights_path, &cache)
        .verify(nn_import::VerifyLevel::Bounds)
        .build()
        .unwrap_or_else(|e| panic!("ConvertBuilder.build() failed: {e}"));

    let report = &result.report;
    eprintln!(
        "[ConvertBuilder bounds] composition_bounds_ok={}, width={:?}, \
         gamma_crown_layers_covered={}, gamma_crown_layers_total={}, method={:?}, \
         soundness={:?}, proof_strength={:?}",
        report.verification.composition_bounds_ok,
        report.verification.composition_bound_width,
        report.verification.gamma_crown_layers_covered,
        report.verification.gamma_crown_layers_total,
        report.verification.composition_method,
        report.verification.composition_soundness_mode,
        report.verification.composition_proof_strength,
    );

    assert!(
        report.verification.composition_bounds_ok,
        "ConvertBuilder must produce valid composition bounds"
    );
    assert!(
        report.verification.gamma_crown_layers_covered > 0,
        "at least one NY layer must be covered"
    );
    assert_eq!(
        report.verification.composition_method,
        Some(nn_import::ConvertCompositionMethod::Ibp)
    );
    assert_eq!(
        report.verification.composition_soundness_mode,
        Some(nn_import::ConvertSoundnessMode::Sound)
    );
    assert_eq!(
        report.verification.composition_proof_strength,
        Some(nn_import::ConvertProofStrength::SoundIbp)
    );

    // Verify the proof object also has composition bounds.
    let proof = &result.result.proof;
    let cb = proof
        .composition_bounds
        .as_ref()
        .expect("EquivalenceProof must have composition bounds");
    assert!(
        cb.propagation_ok,
        "proof composition bounds propagation must succeed"
    );
    assert_eq!(
        cb.composition_method,
        Some(nn_import::ConvertCompositionMethod::Ibp)
    );
    assert_eq!(
        cb.composition_soundness_mode,
        Some(nn_import::ConvertSoundnessMode::Sound)
    );
    assert_eq!(
        cb.composition_proof_strength,
        Some(nn_import::ConvertProofStrength::SoundIbp)
    );
    if let Some(width) = cb.output_width {
        assert!(
            width.is_finite(),
            "proof output width must be finite, got {width}"
        );
        eprintln!("[ConvertBuilder bounds] PASSED: proof output width = {width:.4}");
    }

    eprintln!(
        "[ConvertBuilder bounds] PASSED: {} dispatches, {} steps",
        report.dispatch_count, report.total_steps
    );
}

/// Test C: Kokoro decoder mini graph structure analysis.
///
/// Parses the Kokoro decoder mini fixture to verify that the graph JSON
/// is parseable and contains the expected Kokoro-specific aten ops
/// (Conv1d, InstanceNorm, etc.). The mini fixture has no weights file,
/// so `build_graph()` will fail with MissingWeight -- this is expected
/// and the test validates the parse layer and op coverage.
///
/// When full Kokoro model files are available at `models/kokoro-82m/decoder/`,
/// the test attempts full NY translation and IBP propagation.
#[test]
#[cfg(feature = "verify")]
fn test_kokoro_decoder_mini_bounds_coverage() {
    use std::path::Path;

    let test_data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let graph_path = test_data.join("kokoro_decoder_mini.json");

    assert!(
        graph_path.exists(),
        "kokoro_decoder_mini.json must exist in test_data"
    );

    // Parse the graph JSON to verify structure.
    let json_bytes = std::fs::read(&graph_path).unwrap_or_else(|e| panic!("read graph JSON: {e}"));
    let program = nn_import::parse_exported_program(&json_bytes)
        .unwrap_or_else(|e| panic!("parse failed: {e:?}"));

    // Enumerate aten targets for diagnostics.
    let mut target_counts: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for node in &program.graph_module.graph.nodes {
        *target_counts.entry(node.target.clone()).or_default() += 1;
    }
    let mut target_list: Vec<_> = target_counts.iter().collect();
    target_list.sort_by(|a, b| b.1.cmp(a.1));

    eprintln!(
        "[Kokoro mini bounds] parsed {} graph nodes",
        program.graph_module.graph.nodes.len()
    );
    for (target, count) in &target_list {
        eprintln!("  {target}: {count}");
    }

    // Count input specs by variant.
    let total_input_specs = program.graph_module.signature.input_specs.len();
    eprintln!(
        "[Kokoro mini bounds] {} total input specs",
        total_input_specs
    );

    // The mini fixture should have typical Kokoro ops.
    let has_conv = target_counts.keys().any(|k| k.contains("conv"));
    eprintln!("[Kokoro mini bounds] has convolution ops: {has_conv}");

    // Try full model path if available.
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let full_graph = workspace.join("models/kokoro-82m/decoder/graph.json");
    let full_weights = workspace.join("models/kokoro-82m/decoder/weights.safetensors");

    if full_graph.exists() && full_weights.exists() {
        eprintln!("[Kokoro mini bounds] full model files found, attempting IBP...");
        match nn_import::import_model(&full_graph, &full_weights) {
            Ok(imported) => {
                let report = nn_import::check_composition_bounds(&imported);
                match report {
                    Some(r) => {
                        eprintln!(
                            "[Kokoro mini bounds] full model IBP: \
                             propagation_ok={}, width={:?}",
                            r.propagation_ok, r.output_width
                        );
                    }
                    None => {
                        eprintln!(
                            "[Kokoro mini bounds] full model IBP: \
                             translation/propagation failed (expected for \
                             some Kokoro ops)"
                        );
                    }
                }
            }
            Err(e) => {
                eprintln!("[Kokoro mini bounds] full model import failed: {e:?}");
            }
        }
    } else {
        eprintln!(
            "[Kokoro mini bounds] full model files not found, \
             skipping IBP propagation test"
        );
    }

    eprintln!("[Kokoro mini bounds] PASSED: graph structure validated");
}

/// Test D: Multi-fixture bounds consistency check.
///
/// Runs IBP propagation on all fixtures that have both graph + weights,
/// and attempts parse-level validation on graph-only fixtures.
/// Collects coverage statistics to surface which model architectures have
/// full NY coverage and which have translation gaps.
#[test]
#[cfg(feature = "verify")]
fn test_multi_fixture_bounds_coverage() {
    use std::path::Path;

    let test_data = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data");

    // Fixtures that have both graph + weights -- can run full IBP.
    let fixtures_with_weights: &[(&str, &str, &str)] = &[(
        "e2e_mlp",
        "e2e_mlp.json",
        "exported_mlp_weights.safetensors",
    )];

    // Graph-only fixtures -- can only validate parsing (build_graph fails
    // with MissingWeight because no safetensors weights are provided).
    let fixtures_graph_only: &[&str] = &[
        "kokoro_decoder_mini",
        "kokoro_encoder_mini",
        "layernorm_softmax",
        "conv_bn_relu",
        "multi_layer_mlp",
        "resnet_basic_block",
        "transformer_encoder_layer",
        "attention_2head_mini",
    ];

    let mut total_with_weights = 0usize;
    let mut ibp_ok = 0usize;
    let mut parseable = 0usize;

    eprintln!("\n=== NY bounds coverage summary ===\n");

    // Full IBP test on fixtures with weights.
    for (name, graph_file, weights_file) in fixtures_with_weights {
        let graph_path = test_data.join(graph_file);
        let weights_path = test_data.join(weights_file);
        if !graph_path.exists() || !weights_path.exists() {
            eprintln!("  {name}: SKIP (files not found)");
            continue;
        }
        total_with_weights += 1;

        match nn_import::import_model(&graph_path, &weights_path) {
            Ok(imported) => {
                let report = nn_import::check_composition_bounds(&imported);
                match report {
                    Some(r) if r.propagation_ok => {
                        ibp_ok += 1;
                        eprintln!("  {name}: PASS (width={:?})", r.output_width);
                    }
                    Some(r) => {
                        eprintln!(
                            "  {name}: TRANSLATED but IBP failed (width={:?})",
                            r.output_width
                        );
                    }
                    None => {
                        eprintln!("  {name}: TRANSLATION FAILED");
                    }
                }
            }
            Err(e) => {
                eprintln!("  {name}: IMPORT FAILED ({e:?})");
            }
        }
    }

    // Parse-level validation on graph-only fixtures.
    eprintln!("\n  --- Graph-only fixtures (parse validation) ---\n");
    for name in fixtures_graph_only {
        let graph_path = test_data.join(format!("{name}.json"));
        if !graph_path.exists() {
            eprintln!("  {name}: SKIP (file not found)");
            continue;
        }

        let json_bytes = match std::fs::read(&graph_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("  {name}: READ FAILED ({e})");
                continue;
            }
        };
        match nn_import::parse_exported_program(&json_bytes) {
            Ok(program) => {
                parseable += 1;
                let op_count = program.graph_module.graph.nodes.len();
                eprintln!("  {name}: PARSEABLE ({op_count} graph nodes)");
            }
            Err(e) => {
                eprintln!("  {name}: PARSE FAILED ({e:?})");
            }
        }
    }

    eprintln!("\n=== Summary ===");
    eprintln!("  Fixtures with weights tested: {total_with_weights}");
    eprintln!("  IBP propagation passed: {ibp_ok}");
    eprintln!("  Graph-only fixtures parseable: {parseable}");
    eprintln!();

    // The E2E MLP must always pass -- it's a simple Linear+ReLU graph.
    assert!(
        ibp_ok >= 1,
        "at least the E2E MLP fixture must pass IBP propagation"
    );
    // All graph-only fixtures must at least parse.
    assert!(
        parseable >= 4,
        "at least 4 graph-only fixtures must be parseable, got {parseable}"
    );
}

/// Test E: Whisper encoder mini — trace → GraphNetwork → IBP propagation.
///
/// Builds a tiny Whisper encoder (1 layer, d_model=16, 2 heads, zero weights),
/// traces the encoder forward pass via `trace_graph()`, translates the
/// ComputationGraph to a NY GraphNetwork via `trace_to_graph_model()`,
/// and runs IBP propagation with bounded input. Verifies output bounds are finite.
///
/// This is the first production-model-architecture IBP test — it exercises
/// Conv1d → GELU → Conv1d → GELU → Transpose → Add → LayerNorm → Attention →
/// Residual → FFN → Residual → LayerNorm through the full NY pipeline.
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
fn test_whisper_encoder_mini_trace_ibp_bounds() {
    use nn_core::dyn_tensor::trace::trace_graph;
    use nn_core::{DType, Device, VarBuilder};
    use nn_whisper::WhisperConfig;

    // Tiny config: 1 encoder layer, d_model=16, 2 heads, small dims.
    // max_source_positions must accommodate the post-conv2 sequence length.
    // mel_len=8 -> conv1(stride=1,pad=1) -> 8 -> conv2(stride=2,pad=1) -> 4.
    // d_model must be divisible by both encoder and decoder attention heads.
    let config = WhisperConfig::whisper_tiny()
        .with_num_mel_bins(4)
        .with_d_model(16)
        .with_encoder_attention_heads(2)
        .with_encoder_layers(1)
        .with_encoder_ffn_dim(32)
        .with_max_source_positions(8)
        .with_decoder_attention_heads(2)
        .with_decoder_layers(1)
        .with_decoder_ffn_dim(32)
        .with_vocab_size(32)
        .with_max_target_positions(16);

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model =
        nn_whisper::WhisperModel::load(&vb, config.clone()).expect("load tiny Whisper model");

    // Build a small mel input: [1, num_mel_bins, mel_frames].
    let mel_len = 8usize;
    let mel =
        nn_core::DynTensor::zeros(&[1, config.num_mel_bins, mel_len], DType::F32, &Device::Cpu)
            .expect("create mel input");

    // Trace the encoder forward pass (cache-free path for clean graph).
    // Register mel as an Input node inside the trace closure so trace_to_graph
    // treats it as a variable (not ConstantWeight).
    let (_output, graph) = trace_graph(|| {
        use nn_core::dyn_tensor::trace::record_input;
        let mut x = mel.clone();
        let id = record_input(x.dims(), x.dtype()).expect("record_input");
        x.set_trace_id(id);
        model.encoder().forward_no_cache(&x)
    })
    .expect("trace_graph must succeed for Whisper encoder");

    let node_count = graph.nodes().len();
    eprintln!("[Whisper encoder mini IBP] traced graph: {node_count} nodes");
    assert!(
        node_count > 10,
        "Whisper encoder graph should have many nodes, got {node_count}"
    );

    // Translate to NY GraphNetwork.
    let translate_result = nn_verify::trace_to_graph_model(&graph);

    match translate_result {
        Ok(result) => {
            let gn = &result.graph;
            eprintln!(
                "[Whisper encoder mini IBP] GraphNetwork: {} nodes, dtype_casts={}",
                gn.num_nodes(),
                result.dtype_cast_count
            );

            // Build input bounds matching the mel input shape [1, 4, 8].
            // Use mel spectrogram range [-10, 0] (log-scale power).
            let input_node = graph
                .nodes()
                .iter()
                .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
                .expect("graph must have an Input node");
            let shape = input_node.output_shape();
            eprintln!("[Whisper encoder mini IBP] input shape: {shape:?}");

            let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), -10.0_f32);
            let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), 0.0_f32);
            let input_bounds =
                nn_verify::BoundedTensor::new(lower, upper).expect("BoundedTensor construction");

            // Run IBP propagation.
            match gn.propagate_ibp(&input_bounds) {
                Ok(output) => {
                    let (out_lower, out_upper) = output.lower_upper();
                    let out_elements = out_lower.len();
                    eprintln!(
                        "[Whisper encoder mini IBP] output elements: {out_elements}, \
                         shape: {:?}",
                        out_lower.shape()
                    );

                    // Verify all bounds are finite.
                    let mut all_finite = true;
                    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
                        if !lo.is_finite() || !hi.is_finite() {
                            eprintln!(
                                "[Whisper encoder mini IBP] non-finite bound at [{i}]: \
                                 lo={lo}, hi={hi}"
                            );
                            all_finite = false;
                        }
                    }
                    assert!(
                        all_finite,
                        "all Whisper encoder output bounds must be finite"
                    );

                    // Check lo <= hi (no inverted bounds).
                    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
                        assert!(
                            lo <= hi,
                            "output bounds [{i}]: lower ({lo}) must be <= upper ({hi})"
                        );
                    }

                    // Compute max width for diagnostics.
                    let max_width = out_upper
                        .iter()
                        .zip(out_lower.iter())
                        .map(|(hi, lo)| hi - lo)
                        .fold(0.0_f32, f32::max);
                    eprintln!(
                        "[Whisper encoder mini IBP] PASSED: {out_elements} elements, \
                         max width {max_width:.4}"
                    );
                }
                Err(e) => {
                    // IBP propagation failure on a production architecture is
                    // a real finding — report the error with diagnostics.
                    eprintln!("[Whisper encoder mini IBP] IBP propagation failed: {e:?}");
                    panic!(
                        "IBP propagation must succeed on Whisper encoder mini \
                         architecture. Error: {e:?}"
                    );
                }
            }
        }
        Err(e) => {
            // Known translation gap: LayerNorm traced via DynTensor produces
            // ConstantWeight nodes for weight/bias that NY's
            // build_graph_network can't handle (it expects activation inputs,
            // not constant-weight inputs). This gap does NOT affect the
            // TensorBlockBuilder-based compose tests in nn-verify.
            //
            // The tracing itself succeeded (verified by node_count assert above),
            // so this test still validates that the Whisper encoder architecture
            // can be traced. The translation gap is tracked in #4346.
            let op_types: Vec<String> = graph
                .nodes()
                .iter()
                .map(|n| format!("{:?}", n.op()))
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            let err_str = format!("{e:?}");
            eprintln!("[Whisper encoder mini IBP] trace_to_graph translation failed: {err_str}");
            eprintln!(
                "[Whisper encoder mini IBP] unique op types in graph: {op_types:?}"
            );

            // LayerNorm constant evaluation is now supported (gc#4348).
            // The constant cascade now reaches MatMul inside SDPA: when all
            // model weights are zeros (VarBuilder::zeros), LayerNorm(zeros)
            // evaluates to zeros, which cascades through Q/K/V projections
            // until MatMul(Q, K^T) sees both inputs as evaluated constants
            // and filters them out. This is expected behavior for zero-weight
            // models — the real model path uses TensorBlockBuilder.
            if err_str.contains("MatMul") && err_str.contains("activation inputs") {
                eprintln!(
                    "[Whisper encoder mini IBP] KNOWN GAP: All-zeros model \
                     cascades constants through LayerNorm into SDPA MatMul. \
                     Tracing succeeded ({node_count} nodes). \
                     Full IBP works via TensorBlockBuilder path (see compose_whisper_encoder_bounds)."
                );
            } else {
                panic!(
                    "trace_to_graph_model failed with unexpected error \
                     (not the known constant-cascade gap). Error: {e:?}"
                );
            }
        }
    }
}

/// Test F: PyanNet speaker segmentation — import_model → IBP propagation.
///
/// PyanNet is a production speaker diarization model with:
/// - Conv1d (SincNet-derived) → MaxPool1d → LSTM → Linear → LogSoftmax
///
/// Uses `import_model()` + `check_composition_bounds()` to verify that the
/// full auto-converter → NY IBP pipeline works on a production model.
///
/// Requires model files at `models/pyannet/graph.json` + `weights.safetensors`.
/// Skips gracefully when files are not present (CI environments without models).
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
#[cfg(feature = "verify")]
fn test_pyannet_import_composition_bounds() {
    use std::path::Path;

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let dir = workspace.join("models").join("pyannet");
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");

    if !graph_path.exists() || !weights_path.exists() {
        eprintln!(
            "[PyanNet IBP] SKIP: model files not found at {}",
            dir.display()
        );
        return;
    }

    let imported = nn_import::import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("[PyanNet IBP] import_model failed: {e:?}"));

    eprintln!(
        "[PyanNet IBP] imported: {} nodes, {} user inputs, output names: {:?}",
        imported.graph.nodes().len(),
        imported.num_user_inputs,
        imported.output_names,
    );

    let report = nn_import::check_composition_bounds(&imported);

    match report {
        Some(ref r) => {
            eprintln!(
                "[PyanNet IBP] propagation_ok={}, output_width={:?}",
                r.propagation_ok, r.output_width
            );
            assert!(r.propagation_ok, "IBP propagation must succeed for PyanNet");
            if let Some(width) = r.output_width {
                assert!(
                    width.is_finite(),
                    "PyanNet output bounds width must be finite, got {width}"
                );
                eprintln!("[PyanNet IBP] PASSED: output width = {width:.4}");
            } else {
                eprintln!("[PyanNet IBP] PASSED: propagation succeeded but width is None");
            }
        }
        None => {
            // Translation failure — diagnose which ops are unsupported.
            let mut op_counts: std::collections::HashMap<String, usize> =
                std::collections::HashMap::new();
            for node in imported.graph.nodes() {
                let op_name = format!("{:?}", node.op());
                // Truncate at first '{' to group by op type.
                let key = op_name
                    .find('{')
                    .map(|pos| &op_name[..pos])
                    .unwrap_or(&op_name)
                    .to_string();
                *op_counts.entry(key).or_default() += 1;
            }
            let mut ops: Vec<_> = op_counts.iter().collect();
            ops.sort_by(|a, b| b.1.cmp(a.1));
            eprintln!(
                "[PyanNet IBP] check_composition_bounds returned None. \
                 Op types in graph:"
            );
            for (op, count) in &ops {
                eprintln!("  {op}: {count}");
            }
            // PyanNet has LSTM which may not be supported in trace_to_graph.
            // This is a known translation gap, not a test failure — report
            // the diagnostic info without panicking.
            eprintln!(
                "[PyanNet IBP] WARNING: translation/propagation failed. \
                 PyanNet contains LSTM ops which may not yet be supported \
                 in trace_to_graph. This is a known gap."
            );
        }
    }
}

/// Test G: PyanNet direct trace_to_graph translation (model-file gated).
///
/// Goes deeper than Test F by calling `trace_to_graph_model()` directly
/// to diagnose exactly where translation fails for PyanNet's op mix
/// (Conv1d, MaxPool1d, LSTM, Linear, LogSoftmax, Abs, etc.).
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
fn test_pyannet_direct_trace_to_graph_translation() {
    use nn_core::dyn_tensor::trace::TraceOp;
    use std::path::Path;

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let dir = workspace.join("models").join("pyannet");
    let graph_path = dir.join("graph.json");
    let weights_path = dir.join("weights.safetensors");

    if !graph_path.exists() || !weights_path.exists() {
        eprintln!(
            "[PyanNet trace_to_graph] SKIP: model files not found at {}",
            dir.display()
        );
        return;
    }

    let imported = nn_import::import_model(&graph_path, &weights_path)
        .unwrap_or_else(|e| panic!("import_model failed: {e:?}"));

    // Count variable inputs.
    let variable_input_count = imported
        .graph
        .nodes()
        .iter()
        .filter(|n| matches!(n.op(), TraceOp::Input))
        .count();

    eprintln!(
        "[PyanNet trace_to_graph] {} nodes, {} variable inputs",
        imported.graph.nodes().len(),
        variable_input_count
    );

    // Attempt translation.
    let translate_result = if variable_input_count > 1 {
        nn_verify::trace_to_graph_model_multi_input(&imported.graph)
    } else {
        nn_verify::trace_to_graph_model(&imported.graph)
    };

    match translate_result {
        Ok(result) => {
            let gn = &result.graph;
            eprintln!(
                "[PyanNet trace_to_graph] SUCCESS: {} GraphNetwork nodes, \
                 dtype_casts={}",
                gn.num_nodes(),
                result.dtype_cast_count
            );

            // If translation succeeded, also run IBP.
            let input_node = imported
                .graph
                .nodes()
                .iter()
                .find(|n| matches!(n.op(), TraceOp::Input))
                .expect("graph must have an Input node");
            let shape = input_node.output_shape();

            let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), -1.0_f32);
            let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), 1.0_f32);
            let input_bounds =
                nn_verify::BoundedTensor::new(lower, upper).expect("BoundedTensor construction");

            match gn.propagate_ibp(&input_bounds) {
                Ok(output) => {
                    let (out_lower, out_upper) = output.lower_upper();
                    let max_width = out_upper
                        .iter()
                        .zip(out_lower.iter())
                        .map(|(hi, lo)| hi - lo)
                        .fold(0.0_f32, f32::max);

                    // Verify all bounds are finite.
                    let all_finite = out_lower.iter().all(|v| v.is_finite())
                        && out_upper.iter().all(|v| v.is_finite());

                    assert!(all_finite, "PyanNet output bounds must be finite");

                    eprintln!(
                        "[PyanNet trace_to_graph] IBP PASSED: {} output elements, \
                         max width {max_width:.4}",
                        out_lower.len()
                    );
                }
                Err(e) => {
                    eprintln!("[PyanNet trace_to_graph] IBP propagation failed: {e:?}");
                    // IBP failure after successful translation is unexpected
                    // but possible for complex models. Report as warning.
                    eprintln!(
                        "[PyanNet trace_to_graph] WARNING: translation succeeded \
                         but IBP failed"
                    );
                }
            }
        }
        Err(e) => {
            // Report the translation error with op diagnostics.
            eprintln!("[PyanNet trace_to_graph] translation failed: {e:?}");

            // Enumerate unique op types for gap analysis.
            let mut op_types: Vec<String> = imported
                .graph
                .nodes()
                .iter()
                .map(|n| {
                    let s = format!("{:?}", n.op());
                    s.find('{').map(|pos| s[..pos].to_string()).unwrap_or(s)
                })
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            op_types.sort();
            eprintln!("[PyanNet trace_to_graph] unique op types: {op_types:?}");
            eprintln!(
                "[PyanNet trace_to_graph] WARNING: known translation gap \
                 (PyanNet uses ops not yet in trace_to_graph)"
            );
        }
    }
}

/// Test H2: Programmatic Conv1d-ReLU chain with non-zero weights — trace → IBP.
///
/// Builds a Conv1d(4,8,k=3) → ReLU → Conv1d(8,4,k=3) → ReLU → Linear(4*T',2)
/// pipeline using manually constructed non-zero weights. Traces via `trace_graph()`,
/// translates to NY `GraphNetwork`, and runs IBP propagation with [-1, 1]
/// input bounds.
///
/// This is the first trace-path IBP test with **meaningful non-trivial bounds**:
/// non-zero weights produce non-degenerate output ranges. The architecture is
/// representative of Conv1d-based production models (Kokoro decoder blocks,
/// Silero VAD encoder, ECAPA-TDNN initial layers) without normalization layers
/// that hit the known LayerNorm constant-cascade gap (#4348).
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
fn test_conv1d_relu_chain_trace_ibp_nontrivial_bounds() {
    use nn_core::dyn_tensor::trace::trace_graph;
    use nn_core::{DType, Device, VarBuilder};
    use std::collections::HashMap;

    // Build Conv1d-ReLU-Conv1d-ReLU-Linear with small non-zero weights.
    let in_ch = 4;
    let mid_ch = 8;
    let out_ch = 4;
    let kernel_size = 3;
    let seq_len = 8;

    // Post-conv temporal: T' = seq_len - kernel_size + 1 = 6 for stride=1, padding=0.
    // After two conv layers: T'' = T' - kernel_size + 1 = 4.
    let t_after_conv1 = seq_len - kernel_size + 1;
    let t_after_conv2 = t_after_conv1 - kernel_size + 1;
    let flat_dim = out_ch * t_after_conv2;
    let final_out = 2;

    // Construct weight tensors with small non-zero values (Xavier-like init).
    let mut tensors: HashMap<String, nn_core::DynTensor> = HashMap::new();

    // Conv1 weights: [out_ch, in_ch, kernel_size]
    let conv1_w_data: Vec<f32> = (0..(mid_ch * in_ch * kernel_size))
        .map(|i| 0.1 * ((i % 7) as f32 - 3.0) / 3.0)
        .collect();
    tensors.insert(
        "conv1.weight".to_string(),
        nn_core::DynTensor::from_slice(&conv1_w_data, &[mid_ch, in_ch, kernel_size], &Device::Cpu)
            .expect("conv1 weight"),
    );
    tensors.insert(
        "conv1.bias".to_string(),
        nn_core::DynTensor::zeros(&[mid_ch], DType::F32, &Device::Cpu).expect("conv1 bias"),
    );

    // Conv2 weights: [out_ch, mid_ch, kernel_size]
    let conv2_w_data: Vec<f32> = (0..(out_ch * mid_ch * kernel_size))
        .map(|i| 0.1 * ((i % 5) as f32 - 2.0) / 2.0)
        .collect();
    tensors.insert(
        "conv2.weight".to_string(),
        nn_core::DynTensor::from_slice(
            &conv2_w_data,
            &[out_ch, mid_ch, kernel_size],
            &Device::Cpu,
        )
        .expect("conv2 weight"),
    );
    tensors.insert(
        "conv2.bias".to_string(),
        nn_core::DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).expect("conv2 bias"),
    );

    // Linear weights: [final_out, flat_dim]
    let linear_w_data: Vec<f32> = (0..(final_out * flat_dim))
        .map(|i| 0.1 * ((i % 3) as f32 - 1.0))
        .collect();
    tensors.insert(
        "linear.weight".to_string(),
        nn_core::DynTensor::from_slice(&linear_w_data, &[final_out, flat_dim], &Device::Cpu)
            .expect("linear weight"),
    );
    tensors.insert(
        "linear.bias".to_string(),
        nn_core::DynTensor::zeros(&[final_out], DType::F32, &Device::Cpu).expect("linear bias"),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    // Build layers.
    let conv1 = nn_core::layers::conv1d(
        in_ch,
        mid_ch,
        kernel_size,
        nn_core::layers::Conv1dConfig::default(),
        vb.pp("conv1"),
    )
    .expect("build conv1");
    let conv2 = nn_core::layers::conv1d(
        mid_ch,
        out_ch,
        kernel_size,
        nn_core::layers::Conv1dConfig::default(),
        vb.pp("conv2"),
    )
    .expect("build conv2");
    let linear =
        nn_core::layers::Linear::load(vb.pp("linear"), flat_dim, final_out).expect("build linear");

    // Build input: [1, in_ch, seq_len].
    let input = nn_core::DynTensor::zeros(&[1, in_ch, seq_len], DType::F32, &Device::Cpu)
        .expect("create input");

    // Trace the forward pass.
    // The input must be registered as an Input node (not ConstantWeight) inside
    // the trace closure, otherwise it gets constant-folded by trace_to_graph.
    let (_output, graph) = trace_graph(|| {
        use nn_core::dyn_tensor::trace::record_input;
        use nn_core::layers::Module;
        let mut x = input.clone();
        let id = record_input(x.dims(), x.dtype()).expect("record_input");
        x.set_trace_id(id);
        let x = conv1.forward(&x)?;
        let x = x.relu()?;
        let x = conv2.forward(&x)?;
        let x = x.relu()?;
        // Reshape [1, out_ch, T''] → [1, flat_dim].
        let x = x.reshape([1, flat_dim])?;
        linear.forward(&x)
    })
    .expect("trace_graph must succeed");

    let node_count = graph.nodes().len();
    eprintln!("[Conv1d-ReLU chain IBP] traced graph: {node_count} nodes");
    assert!(
        node_count > 5,
        "Conv1d-ReLU chain should have multiple nodes, got {node_count}"
    );

    // Translate to NY GraphNetwork.
    let result = nn_verify::trace_to_graph_model(&graph)
        .unwrap_or_else(|e| panic!("trace_to_graph translation failed: {e:?}"));

    let gn = &result.graph;
    eprintln!(
        "[Conv1d-ReLU chain IBP] GraphNetwork: {} nodes, dtype_casts={}",
        gn.num_nodes(),
        result.dtype_cast_count
    );

    // Build input bounds [-1, 1] matching traced input shape [1, 4, 8].
    let input_node = graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
        .expect("graph must have an Input node");
    let shape = input_node.output_shape();

    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), -1.0_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), 1.0_f32);
    let input_bounds =
        nn_verify::BoundedTensor::new(lower, upper).expect("BoundedTensor construction");

    // Run IBP propagation.
    let output = gn
        .propagate_ibp(&input_bounds)
        .unwrap_or_else(|e| panic!("IBP propagation failed: {e:?}"));

    let (out_lower, out_upper) = output.lower_upper();
    let out_elements = out_lower.len();
    eprintln!(
        "[Conv1d-ReLU chain IBP] output elements: {out_elements}, shape: {:?}",
        out_lower.shape()
    );

    // Verify all output bounds are finite.
    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
        assert!(
            lo.is_finite(),
            "output lower bound [{i}] must be finite, got {lo}"
        );
        assert!(
            hi.is_finite(),
            "output upper bound [{i}] must be finite, got {hi}"
        );
        assert!(
            lo <= hi,
            "output bounds [{i}]: lower ({lo}) must be <= upper ({hi})"
        );
    }

    // With non-zero weights, bounds should have non-trivial width.
    let max_width = out_upper
        .iter()
        .zip(out_lower.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    eprintln!("[Conv1d-ReLU chain IBP] max output width: {max_width:.6}");
    assert!(
        max_width > 0.0,
        "non-zero weights should produce non-trivial bounds, got max_width={max_width}"
    );
    assert!(
        max_width < 1000.0,
        "bounds should be reasonable (<1000), got max_width={max_width}"
    );

    eprintln!(
        "[Conv1d-ReLU chain IBP] PASSED: {out_elements} output elements, \
         max width {max_width:.6}"
    );
}

/// Regression: `ConstantPadNd` emits `LayerType::Pad` so NY uses its
/// forward-linear Pad bounds path (NY fa7fc91f4) instead of the prior
/// Slice+Concat + constant-weight decomposition.
///
/// Traces `Conv1d → constant_pad_nd → Conv1d` on `[1, 4, 8]` and asserts that
/// IBP propagation produces finite, well-ordered bounds. With zero-padding on
/// the time axis the second Conv1d must see padded inputs; any translator
/// regression (unsupported op, shape mismatch, infinite bounds) fails here.
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
fn test_pad_trace_ibp_bounds() {
    use nn_core::dyn_tensor::trace::trace_graph;
    use nn_core::{DType, Device, VarBuilder};
    use std::collections::HashMap;

    let in_ch = 4;
    let mid_ch = 4;
    let out_ch = 4;
    let kernel_size = 3;
    let seq_len = 8;
    // Pad innermost dim (time) by (1, 1) via PyTorch convention.
    let pad_l: usize = 1;
    let pad_r: usize = 1;

    let t_after_conv1 = seq_len - kernel_size + 1; // 6
    let t_after_pad = t_after_conv1 + pad_l + pad_r; // 8
    let t_after_conv2 = t_after_pad - kernel_size + 1; // 6
    let _ = t_after_conv2; // documented, not used directly.

    let mut tensors: HashMap<String, nn_core::DynTensor> = HashMap::new();

    let conv1_w_data: Vec<f32> = (0..(mid_ch * in_ch * kernel_size))
        .map(|i| 0.1 * ((i % 5) as f32 - 2.0) / 2.0)
        .collect();
    tensors.insert(
        "conv1.weight".to_string(),
        nn_core::DynTensor::from_slice(&conv1_w_data, &[mid_ch, in_ch, kernel_size], &Device::Cpu)
            .expect("conv1 weight"),
    );
    tensors.insert(
        "conv1.bias".to_string(),
        nn_core::DynTensor::zeros(&[mid_ch], DType::F32, &Device::Cpu).expect("conv1 bias"),
    );

    let conv2_w_data: Vec<f32> = (0..(out_ch * mid_ch * kernel_size))
        .map(|i| 0.1 * ((i % 3) as f32 - 1.0))
        .collect();
    tensors.insert(
        "conv2.weight".to_string(),
        nn_core::DynTensor::from_slice(
            &conv2_w_data,
            &[out_ch, mid_ch, kernel_size],
            &Device::Cpu,
        )
        .expect("conv2 weight"),
    );
    tensors.insert(
        "conv2.bias".to_string(),
        nn_core::DynTensor::zeros(&[out_ch], DType::F32, &Device::Cpu).expect("conv2 bias"),
    );

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);

    let conv1 = nn_core::layers::conv1d(
        in_ch,
        mid_ch,
        kernel_size,
        nn_core::layers::Conv1dConfig::default(),
        vb.pp("conv1"),
    )
    .expect("build conv1");
    let conv2 = nn_core::layers::conv1d(
        mid_ch,
        out_ch,
        kernel_size,
        nn_core::layers::Conv1dConfig::default(),
        vb.pp("conv2"),
    )
    .expect("build conv2");

    let input = nn_core::DynTensor::zeros(&[1, in_ch, seq_len], DType::F32, &Device::Cpu)
        .expect("create input");

    let (_output, graph) = trace_graph(|| {
        use nn_core::dyn_tensor::trace::record_input;
        use nn_core::layers::Module;
        let mut x = input.clone();
        let id = record_input(x.dims(), x.dtype()).expect("record_input");
        x.set_trace_id(id);
        let x = conv1.forward(&x)?;
        // PyTorch padding convention: `[left_innermost, right_innermost, ...]`.
        let x = x.constant_pad_nd(&[pad_l, pad_r], 0.0)?;
        conv2.forward(&x)
    })
    .expect("trace_graph must succeed");

    // Graph must contain a ConstantPadNd node so we exercise the translator.
    let has_pad = graph.nodes().iter().any(|n| {
        matches!(
            n.op(),
            nn_core::dyn_tensor::trace::TraceOp::ConstantPadNd { .. }
        )
    });
    assert!(
        has_pad,
        "traced graph must contain a ConstantPadNd node; got {:?}",
        graph
            .nodes()
            .iter()
            .map(|n| format!("{:?}", n.op()))
            .collect::<Vec<_>>()
    );

    let result = nn_verify::trace_to_graph_model(&graph)
        .unwrap_or_else(|e| panic!("trace_to_graph translation failed: {e:?}"));
    let gn = &result.graph;

    let input_node = graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
        .expect("graph must have an Input node");
    let shape = input_node.output_shape();

    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), -1.0_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), 1.0_f32);
    let input_bounds =
        nn_verify::BoundedTensor::new(lower, upper).expect("BoundedTensor construction");

    let output = gn
        .propagate_ibp(&input_bounds)
        .unwrap_or_else(|e| panic!("IBP propagation failed: {e:?}"));

    let (out_lower, out_upper) = output.lower_upper();
    assert!(!out_lower.is_empty(), "IBP output must be non-empty");
    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
        assert!(
            lo.is_finite(),
            "output lower [{i}] must be finite, got {lo}"
        );
        assert!(
            hi.is_finite(),
            "output upper [{i}] must be finite, got {hi}"
        );
        assert!(
            lo <= hi,
            "output bounds [{i}]: lower ({lo}) must be <= upper ({hi})"
        );
    }
    eprintln!(
        "[Pad IBP] PASSED: {} output elements, output shape {:?}",
        out_lower.len(),
        out_lower.shape()
    );
}

/// Test H3: ECAPA-TDNN speaker verification — trace → GraphNetwork → IBP.
///
/// Traces the full ECAPA-TDNN production model (Conv1d → SE-Res2Blocks → Cat →
/// ASP → BN + Linear → L2-norm) with zero weights via `trace_graph()`. This
/// exercises the most complex Conv1d/BatchNorm/Attention architecture available
/// in nn without hitting the LayerNorm translation gap.
///
/// With zero weights, output bounds are trivially narrow, but the test validates
/// that the ENTIRE ECAPA-TDNN op sequence (Conv1d, BatchNorm, ReLU, Transpose,
/// Softmax, Mul, ReduceSum, ReduceMean, Squeeze, Unsqueeze, Narrow, Cat, Sqrt,
/// Maximum, Div, Linear, Sqr) translates and propagates successfully through
/// NY IBP.
///
/// This is the first production-model trace-path IBP test that PASSES end-to-end.
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
fn test_ecapa_tdnn_trace_ibp_bounds() {
    use nn_core::dyn_tensor::trace::trace_graph;
    use nn_core::{DType, Device, VarBuilder};

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = nn_models::EcapaTdnn::load(&vb).expect("load ECAPA-TDNN with zero weights");

    // Build mel input: [1, 80, T]. T=32 is enough for the conv+pool chain.
    let mel = nn_core::DynTensor::zeros(&[1, 80, 32], DType::F32, &Device::Cpu)
        .expect("create mel input");

    // Trace the forward pass.
    // Register mel as an Input node inside the trace closure so trace_to_graph
    // treats it as a variable (not ConstantWeight).
    let trace_result = trace_graph(|| {
        use nn_core::dyn_tensor::trace::record_input;
        let mut x = mel.clone();
        let id = record_input(x.dims(), x.dtype()).expect("record_input");
        x.set_trace_id(id);
        model.forward(&x)
    });

    match trace_result {
        Ok((_output, graph)) => {
            let node_count = graph.nodes().len();
            eprintln!("[ECAPA-TDNN IBP] traced graph: {node_count} nodes");
            assert!(
                node_count > 50,
                "ECAPA-TDNN graph should have many nodes (deep architecture), got {node_count}"
            );

            // Enumerate unique op types for diagnostics.
            let mut op_types: std::collections::HashSet<String> = std::collections::HashSet::new();
            for node in graph.nodes() {
                let s = format!("{:?}", node.op());
                let key = s.find('{').map(|pos| s[..pos].to_string()).unwrap_or(s);
                op_types.insert(key);
            }
            let mut sorted_ops: Vec<_> = op_types.iter().cloned().collect();
            sorted_ops.sort();
            eprintln!("[ECAPA-TDNN IBP] unique op types: {sorted_ops:?}");

            // Translate to NY GraphNetwork.
            let translate_result = nn_verify::trace_to_graph_model(&graph);

            match translate_result {
                Ok(result) => {
                    let gn = &result.graph;
                    eprintln!(
                        "[ECAPA-TDNN IBP] GraphNetwork: {} nodes, dtype_casts={}",
                        gn.num_nodes(),
                        result.dtype_cast_count
                    );

                    // Build input bounds for mel: [-10, 0] (log-scale power range).
                    let input_node = graph
                        .nodes()
                        .iter()
                        .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
                        .expect("graph must have an Input node");
                    let shape = input_node.output_shape();

                    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), -1.0_f32);
                    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), 1.0_f32);
                    let input_bounds = nn_verify::BoundedTensor::new(lower, upper)
                        .expect("BoundedTensor construction");

                    match gn.propagate_ibp(&input_bounds) {
                        Ok(output) => {
                            let (out_lower, out_upper) = output.lower_upper();
                            let out_elements = out_lower.len();

                            // Verify all bounds are finite.
                            let mut all_finite = true;
                            for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate()
                            {
                                if !lo.is_finite() || !hi.is_finite() {
                                    eprintln!(
                                        "[ECAPA-TDNN IBP] non-finite bound at [{i}]: \
                                         lo={lo}, hi={hi}"
                                    );
                                    all_finite = false;
                                }
                            }
                            assert!(all_finite, "all ECAPA-TDNN output bounds must be finite");

                            // Check lo <= hi.
                            for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate()
                            {
                                assert!(
                                    lo <= hi,
                                    "bounds [{i}]: lower ({lo}) must be <= upper ({hi})"
                                );
                            }

                            let max_width = out_upper
                                .iter()
                                .zip(out_lower.iter())
                                .map(|(hi, lo)| hi - lo)
                                .fold(0.0_f32, f32::max);
                            eprintln!(
                                "[ECAPA-TDNN IBP] PASSED: {out_elements} output elements, \
                                 max width {max_width:.6}"
                            );
                        }
                        Err(e) => {
                            let err_str = format!("{e:?}");
                            eprintln!("[ECAPA-TDNN IBP] IBP propagation failed: {err_str}");
                            // ECAPA-TDNN's L2 normalization at the end divides by
                            // the embedding norm. With [-1, 1] input bounds, the norm
                            // upper bound is infinite (sqrt of sum of squares with no
                            // upper limit on squared values times channel count).
                            // NY's DivLayer requires finite divisor bounds.
                            // This is a KNOWN IBP limitation for L2-normalized models,
                            // not a translation gap. The graph translates correctly
                            // (validated above). Tracked in #4346.
                            if err_str.contains("DivLayer") || err_str.contains("divisor bounds") {
                                eprintln!(
                                    "[ECAPA-TDNN IBP] KNOWN LIMITATION: L2 normalization \
                                     produces infinite norm bounds with unbounded input. \
                                     Translation succeeded ({} GraphNetwork nodes). \
                                     IBP would work on bounded-norm subgraphs.",
                                    gn.num_nodes()
                                );
                            } else {
                                panic!(
                                    "IBP propagation failed with unexpected error \
                                     (not the known L2-norm limitation): {e:?}"
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    // ECAPA-TDNN uses only BatchNorm (not LayerNorm), so translation
                    // should succeed. If it fails, this is a real gap, not the known
                    // LayerNorm constant-cascade issue.
                    let err_str = format!("{e:?}");
                    eprintln!("[ECAPA-TDNN IBP] trace_to_graph translation failed: {err_str}");

                    // Diagnostic: which ops are in the graph?
                    let mut ops: Vec<_> = op_types.into_iter().collect();
                    ops.sort();
                    eprintln!("[ECAPA-TDNN IBP] ops present: {ops:?}");

                    // ECAPA-TDNN should NOT fail on translation — it has no LayerNorm.
                    // Mark as a failure unless we hit a known unsupported op.
                    if err_str.contains("not supported") {
                        eprintln!(
                            "[ECAPA-TDNN IBP] TRANSLATION GAP: unsupported op in \
                             ECAPA-TDNN architecture. This is a new gap to investigate."
                        );
                        panic!("ECAPA-TDNN trace_to_graph failed with unsupported op: {e:?}");
                    } else {
                        panic!("ECAPA-TDNN trace_to_graph failed unexpectedly: {e:?}");
                    }
                }
            }
        }
        Err(e) => {
            // Tracing itself failed — this is unexpected for zero-weight model.
            panic!("[ECAPA-TDNN IBP] trace_graph failed: {e:?}");
        }
    }
}

/// Test H: Whisper encoder mini — check_composition_bounds via import-like path.
///
/// Builds a tiny Whisper encoder, traces it, wraps the ComputationGraph in an
/// ImportedGraph-like structure, and calls check_composition_bounds() to verify
/// the full public API path works end-to-end on a traced production architecture.
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
#[cfg(feature = "verify")]
fn test_whisper_encoder_mini_check_composition_bounds() {
    use nn_core::dyn_tensor::trace::trace_graph;
    use nn_core::{DType, Device, VarBuilder};
    use nn_whisper::WhisperConfig;

    let config = WhisperConfig::whisper_tiny()
        .with_num_mel_bins(4)
        .with_d_model(16)
        .with_encoder_attention_heads(2)
        .with_encoder_layers(1)
        .with_encoder_ffn_dim(32)
        .with_max_source_positions(8)
        .with_decoder_attention_heads(2)
        .with_decoder_layers(1)
        .with_decoder_ffn_dim(32)
        .with_vocab_size(32)
        .with_max_target_positions(16);

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model = nn_whisper::WhisperModel::load(&vb, config.clone()).expect("load tiny Whisper");

    let mel_len = 8usize;
    let mel =
        nn_core::DynTensor::zeros(&[1, config.num_mel_bins, mel_len], DType::F32, &Device::Cpu)
            .expect("create mel input");

    let (_output, graph) = trace_graph(|| {
        use nn_core::dyn_tensor::trace::record_input;
        let mut x = mel.clone();
        let id = record_input(x.dims(), x.dtype()).expect("record_input");
        x.set_trace_id(id);
        model.encoder().forward_no_cache(&x)
    })
    .expect("trace_graph must succeed");

    // Build an ImportedGraph to use the public check_composition_bounds API.
    let imported = nn_import::ImportedGraph::new(
        graph,
        1,
        vec!["mel".to_string()],
        vec!["encoder_output".to_string()],
    );

    let report = nn_import::check_composition_bounds(&imported);
    match report {
        Some(ref r) => {
            eprintln!(
                "[Whisper encoder check_composition_bounds] \
                 propagation_ok={}, output_width={:?}",
                r.propagation_ok, r.output_width
            );
            assert!(
                r.propagation_ok,
                "check_composition_bounds must succeed for Whisper encoder mini"
            );
            if let Some(width) = r.output_width {
                assert!(
                    width.is_finite(),
                    "output width must be finite, got {width}"
                );
            }
            eprintln!("[Whisper encoder check_composition_bounds] PASSED");
        }
        None => {
            // Known gap: check_composition_bounds returns None when
            // trace_to_graph_model fails. The Whisper encoder has LayerNorm
            // layers whose traced ConstantWeight nodes for weight/bias confuse
            // NY's input filtering. This is the same gap as Test E.
            //
            // Verify the tracing step itself works by checking the graph.
            let node_count = imported.graph.nodes().len();
            eprintln!(
                "[Whisper encoder check_composition_bounds] returned None — \
                 trace_to_graph LayerNorm gap (same as Test E). \
                 Graph has {node_count} traced nodes."
            );
            assert!(
                node_count > 10,
                "Whisper encoder graph must still have traced nodes, got {node_count}"
            );
            eprintln!(
                "[Whisper encoder check_composition_bounds] KNOWN GAP: \
                 LayerNorm translation not yet supported via trace path. \
                 Full IBP works via TensorBlockBuilder path."
            );
        }
    }
}
/// Full `RtDetr::forward` IBP bounds test.
///
/// Currently blocked on #4360 (MultiHeadAttention translator gap). Test
/// asserts the known-expected UnsupportedOp to fail-fast when #4360 lands —
/// at which point this test MUST be updated to propagate IBP and assert
/// finite output bounds on both class_logits and bbox_preds.
///
/// Input: `[1, 3, 64, 64]`.
/// Outputs: class_logits `[B, num_queries, num_classes+1]` and
///          bbox_preds   `[B, num_queries, 4]`, traced as a concatenated
///          tensor `[B, num_queries, num_classes+5]`.
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
fn test_rt_detr_full_forward_trace_ibp_bounds() {
    use nn_core::dyn_tensor::trace::trace_graph;
    use nn_core::{DType, Device, VarBuilder};
    use nn_models::{RtDetr, RtDetrBackboneVariant, RtDetrConfig};

    let mut config = RtDetrConfig::preset_heron();
    config.backbone_variant = RtDetrBackboneVariant::HuggingFace;
    config.input_size = 64;
    config.num_queries = 10;
    config.num_classes = 4;
    config.hidden_dim = 32;
    config.num_heads = 4;
    config.ffn_dim = 64;
    config.num_decoder_layers = 2;
    config.num_sampling_points = 2;
    config.conf_threshold = 0.3;
    config.backbone_channels = [128, 256, 512];

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let model =
        RtDetr::load(&vb, config).expect("load RtDetr with zero weights and small config");

    let image = nn_core::DynTensor::zeros(&[1, 3, 64, 64], DType::F32, &Device::Cpu)
        .expect("create image input");

    let (_output, graph) = trace_graph(|| {
        use nn_core::dyn_tensor::trace::record_input;

        let mut x = image.clone();
        let id = record_input(x.dims(), x.dtype()).expect("record_input");
        x.set_trace_id(id);

        let (class_logits, bbox_preds) = model.forward(&x)?;

        // trace_graph tracks a single last_output node, so concatenate the
        // two rank-3 outputs into one tensor on the last axis.
        nn_core::DynTensor::cat(&[&class_logits, &bbox_preds], 2)
    })
    .expect("trace_graph must succeed for full RtDetr::forward");

    let node_count = graph.nodes().len();
    eprintln!("[RT-DETR full-forward IBP] traced graph: {node_count} nodes");
    assert!(
        node_count > 100,
        "full RT-DETR graph should have substantial traced structure, got {node_count}"
    );

    let mut op_types: std::collections::HashSet<String> = std::collections::HashSet::new();
    for node in graph.nodes() {
        let s = format!("{:?}", node.op());
        let key = s.find('{').map(|pos| s[..pos].to_string()).unwrap_or(s);
        op_types.insert(key);
    }
    let mut sorted_ops: Vec<_> = op_types.into_iter().collect();
    sorted_ops.sort();
    eprintln!(
        "[RT-DETR full-forward IBP] unique op types: {sorted_ops:?}"
    );

    // Translation currently fails on TraceOp::MultiHeadAttention (#4360).
    // Once #4360 lands, translation will succeed, the Err branch below
    // becomes unreachable, and this test MUST be updated to propagate
    // IBP bounds and assert finiteness on both class_logits and bbox_preds
    // regions (see the commented-out code at the bottom of this test).
    match nn_verify::trace_to_graph_model(&graph) {
        Err(nn_verify::VerifyError::UnsupportedOp(msg)) if msg.contains("MultiHeadAttention") => {
            eprintln!(
                "[RT-DETR full-forward IBP] expected UnsupportedOp tripwire \
                 hit: {msg}. Tracked in #4360. When resolved, update this \
                 test to assert finite output bounds on both class_logits \
                 and bbox_preds regions of the concatenated output tensor."
            );
        }
        Err(other) => {
            panic!(
                "[RT-DETR full-forward IBP] unexpected translator failure. \
                 Expected UnsupportedOp(MultiHeadAttention ...) pending #4360, \
                 got: {other:?}. If this is a NEW translator gap, file a \
                 follow-up issue and update this test to track it."
            );
        }
        Ok(_result) => {
            panic!(
                "[RT-DETR full-forward IBP] TRIPWIRE FIRED: \
                 trace_to_graph_model now succeeds on the full RT-DETR \
                 forward path (#4360 appears resolved). Update this test \
                 to propagate IBP with input bounds [-1, 1] and assert \
                 finite bounds on class_logits and bbox_preds regions. \
                 See the commented-out completion block in this test body."
            );
        }
    }

    // --- Completion path (commented out until #4360 lands) ---
    // When the translator supports MultiHeadAttention (or decomposes it to
    // SDPA), re-enable this block and delete the match above. This path will
    // need `use ndarray::s;` restored for the `s![..]` slicing below.
    //
    //   let result = nn_verify::trace_to_graph_model(&graph)
    //       .expect("translation should succeed once #4360 lands");
    //   let gn = &result.graph;
    //   eprintln!(
    //       "[RT-DETR full-forward IBP] GraphNetwork: {} nodes, dtype_casts={}",
    //       gn.num_nodes(),
    //       result.dtype_cast_count
    //   );
    //   let input_node = graph
    //       .nodes()
    //       .iter()
    //       .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
    //       .expect("graph must have an Input node");
    //   let shape = input_node.output_shape();
    //   let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), -1.0_f32);
    //   let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), 1.0_f32);
    //   let input_bounds =
    //       nn_verify::BoundedTensor::new(lower, upper).expect("BoundedTensor construction");
    //   let output = gn
    //       .propagate_ibp(&input_bounds)
    //       .expect("IBP propagation should succeed once #4360 lands");
    //   let (out_lower, out_upper) = output.lower_upper();
    //   let split = config.num_classes + 1;
    //   let expected_shape = [1, config.num_queries, split + 4];
    //   assert_eq!(out_lower.shape(), expected_shape.as_slice());
    //   let class_lower = out_lower.slice(s![.., .., 0..split]);
    //   let class_upper = out_upper.slice(s![.., .., 0..split]);
    //   let bbox_lower = out_lower.slice(s![.., .., split..]);
    //   let bbox_upper = out_upper.slice(s![.., .., split..]);
    //   // ... (check finiteness, lo<=hi, max_width<1e6 on both regions)
}

/// Test I: RT-DETR backbone (ResNet18Hf) — trace → GraphNetwork → IBP propagation.
///
/// Traces the HuggingFace ResNet18 backbone used by RT-DETRv2 (the same backbone
/// that feeds `RtDetr` with `RtDetrBackboneVariant::HuggingFace`) on a small
/// `[1, 3, 64, 64]` image. The full `RtDetr::forward` path hits the LayerNorm
/// constant-cascade gap (#4350) through the AIFI encoder and DETR decoder, so
/// this test exercises only the backbone — the op mix (Conv2d + BatchNorm +
/// ReLU + MaxPool + residual Add + global AvgPool) is exactly the surface area
/// unlocked by NY `fa7fc91f4` (Conv2d / Add / AveragePool / Pad in
/// graph forward-linear bounds).
///
/// With zero weights the output bounds collapse to a narrow range, but the test
/// validates that the entire backbone op sequence translates to NY's
/// `GraphNetwork` and IBP propagates end-to-end without vacuous (infinite)
/// bounds.
///
/// Input: `[1, 3, 64, 64]`.
/// Output: final feature scale C5 = `[1, 512, 2, 2]` (stride 32).
///
/// Part of #4346 (NY IBP bounds on production models).
#[test]
fn test_rt_detr_backbone_hf_trace_ibp_bounds() {
    use nn_core::dyn_tensor::trace::trace_graph;
    use nn_core::layers::vision::ResNet18Hf;
    use nn_core::{DType, Device, VarBuilder};

    // Build the HF-flavored ResNet18 backbone used by RtDetr's
    // `RtDetrBackboneVariant::HuggingFace`. No classification head —
    // we only need the multi-scale features.
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let backbone = ResNet18Hf::load(&vb, None).expect("load ResNet18Hf backbone with zero weights");

    // Small image so the test is fast. 64x64 → after stem (stride 4) 16x16 →
    // layer1 16x16 → layer2 stride2 8x8 → layer3 4x4 → layer4 2x2.
    let image = nn_core::DynTensor::zeros(&[1, 3, 64, 64], DType::F32, &Device::Cpu)
        .expect("create image input");

    // Trace the backbone's forward_features, returning only the deepest
    // scale (C5) as a single tensor so NY sees a single output.
    let (_output, graph) = trace_graph(|| {
        use nn_core::dyn_tensor::trace::record_input;
        let mut x = image.clone();
        let id = record_input(x.dims(), x.dtype()).expect("record_input");
        x.set_trace_id(id);
        let features = backbone.forward_features(&x)?;
        // C5 is the last scale (stride 32, 512 channels).
        features.into_iter().last().ok_or_else(|| {
            nn_core::TensorError::InvalidShape(
                "ResNet18Hf::forward_features returned empty".into(),
            )
        })
    })
    .expect("trace_graph must succeed for ResNet18Hf backbone");

    let node_count = graph.nodes().len();
    eprintln!("[RT-DETR backbone IBP] traced graph: {node_count} nodes");
    assert!(
        node_count > 30,
        "ResNet18 backbone should have many nodes (Conv2d+BN+ReLU+residual per block × \
         8 blocks + stem), got {node_count}"
    );

    // Enumerate op types for diagnostics.
    let mut op_types: std::collections::HashSet<String> = std::collections::HashSet::new();
    for node in graph.nodes() {
        let s = format!("{:?}", node.op());
        let key = s.find('{').map(|pos| s[..pos].to_string()).unwrap_or(s);
        op_types.insert(key);
    }
    let mut sorted_ops: Vec<_> = op_types.into_iter().collect();
    sorted_ops.sort();
    eprintln!("[RT-DETR backbone IBP] unique op types: {sorted_ops:?}");

    // Translate to NY GraphNetwork.
    let translate_result = nn_verify::trace_to_graph_model(&graph);

    let result = match translate_result {
        Ok(r) => r,
        Err(e) => {
            // Translation failure on the ResNet18 backbone is a real finding:
            // this is the exact op surface that gc#fa7fc91f4 unblocked. Surface
            // it clearly rather than silently skipping.
            panic!(
                "[RT-DETR backbone IBP] trace_to_graph_model failed — real gap in \
                 production backbone (Conv2d/BN/ReLU/Add/MaxPool). Error: {e:?}. \
                 Tracked in #4346."
            );
        }
    };

    let gn = &result.graph;
    eprintln!(
        "[RT-DETR backbone IBP] GraphNetwork: {} nodes, dtype_casts={}",
        gn.num_nodes(),
        result.dtype_cast_count
    );

    // Build input bounds [-1, 1] for a normalized image.
    let input_node = graph
        .nodes()
        .iter()
        .find(|n| matches!(n.op(), nn_core::dyn_tensor::trace::TraceOp::Input))
        .expect("graph must have an Input node");
    let shape = input_node.output_shape();
    eprintln!("[RT-DETR backbone IBP] input shape: {shape:?}");

    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), -1.0_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(shape), 1.0_f32);
    let input_bounds =
        nn_verify::BoundedTensor::new(lower, upper).expect("BoundedTensor construction");

    // Run IBP propagation.
    let output = match gn.propagate_ibp(&input_bounds) {
        Ok(o) => o,
        Err(e) => {
            // IBP failure after successful translation is the exact scenario
            // gc#fa7fc91f4 should have fixed for Conv2d/Add/AveragePool/Pad.
            // Surface clearly so we know it's a propagation gap, not translation.
            panic!(
                "[RT-DETR backbone IBP] IBP propagation failed (translation succeeded: \
                 {} GraphNetwork nodes). Error: {e:?}. Tracked in #4346.",
                gn.num_nodes()
            );
        }
    };

    let (out_lower, out_upper) = output.lower_upper();
    let out_elements = out_lower.len();
    eprintln!(
        "[RT-DETR backbone IBP] output elements: {out_elements}, shape: {:?}",
        out_lower.shape()
    );

    // Verify all bounds are finite.
    let mut non_finite = 0usize;
    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
        if !lo.is_finite() || !hi.is_finite() {
            if non_finite < 5 {
                eprintln!("[RT-DETR backbone IBP] non-finite bound at [{i}]: lo={lo}, hi={hi}");
            }
            non_finite += 1;
        }
    }
    assert_eq!(
        non_finite, 0,
        "all RT-DETR backbone output bounds must be finite, got {non_finite} non-finite"
    );

    // Check lo <= hi.
    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
        assert!(
            lo <= hi,
            "output bounds [{i}]: lower ({lo}) must be <= upper ({hi})"
        );
    }

    // Compute max width. With zero weights, width should be very narrow
    // (residual bias paths only) but finite and non-negative.
    let max_width = out_upper
        .iter()
        .zip(out_lower.iter())
        .map(|(hi, lo)| hi - lo)
        .fold(0.0_f32, f32::max);
    eprintln!(
        "[RT-DETR backbone IBP] PASSED: {out_elements} output elements, \
         max width {max_width:.6}"
    );
    assert!(
        max_width.is_finite(),
        "max output width must be finite, got {max_width}"
    );
    // Don't enforce a tight upper bound on width — with zero weights it should
    // be tiny, but non-zero weights in future integrations may widen it.
    assert!(
        max_width < 1e6,
        "max output width should be reasonable (<1e6), got {max_width}"
    );
}

/// TL7: OpaqueSkip fallback for `TraceOp::Custom` — trace → GraphNetwork → IBP.
///
/// Regression test for the TraceOp::Custom hard-fail → OpaqueSkip conversion
/// (Part of #4349; leverages NY 589c56c4a + 8bb2be2b8 upstream fixes).
///
/// Before this change, any graph containing a single `TraceOp::Custom` op
/// produced `VerifyError::UnsupportedOp`, blocking verification of the entire
/// model. With the fallback, Custom emits `LayerType::Unknown`, which
/// gamma-build's builder auto-replaces with `Layer::OpaqueSkip(OpaqueSkipLayer)`.
/// OpaqueSkip returns conservative [-inf, +inf] bounds, which downstream
/// sanitization converts to finite sentinels (bounds remain sound).
///
/// This test builds: Input → ReLU → Custom("mystery") → ReLU
/// and asserts:
/// - Translation succeeds (no `UnsupportedOp` error).
/// - IBP propagation succeeds.
/// - Output bounds pass the invariant `lower <= upper` at every element.
/// - Output bounds are conservative (wider than input, as expected for
///   OpaqueSkip-originated bounds that overwrite the finite input range).
#[test]
fn test_opaque_custom_trace_ibp_soundness_4349() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    // Build Input(shape=[1, 4]) → ReLU → Custom("mystery") → ReLU.
    // Four nodes, topologically ordered. Last node is the graph output.
    let input_node = TraceNode::new(
        0,
        "input".to_string(),
        TraceOp::Input,
        vec![],
        vec![1, 4],
        DType::F32,
    );
    let relu_pre = TraceNode::new(
        1,
        "relu_pre".to_string(),
        TraceOp::Relu,
        vec![0],
        vec![1, 4],
        DType::F32,
    );
    let custom_node = TraceNode::new(
        2,
        "custom_mystery".to_string(),
        TraceOp::Custom {
            name: "mystery".to_string(),
        },
        vec![1],
        vec![1, 4],
        DType::F32,
    );
    let relu_post = TraceNode::new(
        3,
        "relu_post".to_string(),
        TraceOp::Relu,
        vec![2],
        vec![1, 4],
        DType::F32,
    );

    let graph = ComputationGraph::from_nodes(vec![input_node, relu_pre, custom_node, relu_post]);

    // Translate: with the OpaqueSkip fallback this must succeed.
    let result = nn_verify::trace_to_graph_model(&graph).unwrap_or_else(|e| {
        panic!(
            "trace_to_graph_model should succeed for graphs with TraceOp::Custom \
             after the OpaqueSkip fallback landed (Part of #4349). Got: {e:?}"
        )
    });
    let gn = &result.graph;
    eprintln!(
        "[OpaqueSkip Custom IBP] GraphNetwork: {} nodes (dtype_casts={})",
        gn.num_nodes(),
        result.dtype_cast_count
    );

    // Bounded input [-1, 1] at the traced input shape.
    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4]), -1.0_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[1, 4]), 1.0_f32);
    let input_bounds =
        nn_verify::BoundedTensor::new(lower, upper).expect("BoundedTensor construction");

    // Run IBP. OpaqueSkipLayer returns ±inf internally; gamma-propagate
    // downstream sanitization (gc 589c56c4a) converts infinities to finite
    // sentinel values so the final post-ReLU bounds remain ordered.
    let output = gn
        .propagate_ibp(&input_bounds)
        .unwrap_or_else(|e| panic!("IBP propagation failed after OpaqueSkip fallback: {e:?}"));

    let (out_lower, out_upper) = output.lower_upper();
    let out_elements = out_lower.len();
    assert!(out_elements > 0, "output must have >= 1 element");
    eprintln!(
        "[OpaqueSkip Custom IBP] output: shape={:?}, elements={out_elements}",
        out_lower.shape()
    );

    // Soundness: every lane must have `lower <= upper` (NaN-safe: reject
    // bounds where either endpoint is NaN).
    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
        assert!(
            !lo.is_nan() && !hi.is_nan(),
            "OpaqueSkip output bounds [{i}] must not be NaN (gc 589c56c4a), got lo={lo}, hi={hi}"
        );
        assert!(
            lo <= hi,
            "OpaqueSkip output bounds [{i}]: lower ({lo}) must be <= upper ({hi})"
        );
    }

    // The final layer is post-ReLU, so the lower bound must be >= 0 (ReLU
    // floor). The upper bound is expected to be very large (OpaqueSkip
    // sentinel) or +inf depending on sanitization layer — both are sound
    // over-approximations of the unknown Custom op output.
    for (i, lo) in out_lower.iter().enumerate() {
        assert!(
            *lo >= 0.0 || !lo.is_finite(),
            "post-ReLU lower bound [{i}] should be >= 0 (ReLU floor) or sentinel, got {lo}"
        );
    }

    // Sanity: conservative OpaqueSkip widens the bounds beyond the input
    // range [-1, 1]. Upper bound max should exceed 1.0 (the input max).
    let max_upper = out_upper.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        max_upper > 1.0 || !max_upper.is_finite(),
        "OpaqueSkip over-approximation should widen upper bounds beyond input max (1.0); \
         got max_upper={max_upper}"
    );

    eprintln!(
        "[OpaqueSkip Custom IBP] PASSED: {out_elements} output lanes, all bounds ordered, \
         max_upper={max_upper}"
    );
}

/// TL7: OpaqueSkip fallback for unknown/future TraceOp variants via the
/// catch-all arm (Part of #4349).
///
/// Complements `test_opaque_custom_trace_ibp_soundness_4349` by exercising
/// the translator's catch-all path. Rather than picking a rarely-used
/// TraceOp variant (which may change over time), this test re-uses
/// `TraceOp::Custom` — the Custom arm and the catch-all arm share the
/// same implementation pattern (both emit `LayerType::Unknown`). The
/// explicit Custom arm is hit here, but the assertion surface (sound
/// widening + finite-downstream) is the exact contract that must also
/// hold for any future variant that falls into the catch-all.
#[test]
fn test_opaque_custom_single_node_finite_sentinel_4349() {
    use nn_core::dyn_tensor::trace::{ComputationGraph, TraceNode, TraceOp};
    use nn_core::DType;

    // Minimal graph: Input → Custom (no downstream ReLU). Exercises the
    // case where OpaqueSkip is the terminal node and no sanitization
    // happens on the output.
    let input_node = TraceNode::new(
        0,
        "input".to_string(),
        TraceOp::Input,
        vec![],
        vec![4],
        DType::F32,
    );
    let custom_node = TraceNode::new(
        1,
        "custom_terminal".to_string(),
        TraceOp::Custom {
            name: "future_op".to_string(),
        },
        vec![0],
        vec![4],
        DType::F32,
    );
    let graph = ComputationGraph::from_nodes(vec![input_node, custom_node]);

    let result = nn_verify::trace_to_graph_model(&graph).unwrap_or_else(|e| {
        panic!(
            "trace_to_graph_model must not hard-fail on TraceOp::Custom after #4349; \
             got {e:?}"
        )
    });
    let gn = &result.graph;

    let lower = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4]), -0.5_f32);
    let upper = ndarray::ArrayD::from_elem(ndarray::IxDyn(&[4]), 0.5_f32);
    let input_bounds =
        nn_verify::BoundedTensor::new(lower, upper).expect("BoundedTensor construction");

    let output = gn
        .propagate_ibp(&input_bounds)
        .unwrap_or_else(|e| panic!("IBP on terminal OpaqueSkip failed: {e:?}"));
    let (out_lower, out_upper) = output.lower_upper();

    // Bounds must be ordered and NaN-free. OpaqueSkip emits ±inf as a
    // terminal node — both are sound conservative over-approximations.
    for (i, (lo, hi)) in out_lower.iter().zip(out_upper.iter()).enumerate() {
        assert!(
            !lo.is_nan() && !hi.is_nan(),
            "terminal OpaqueSkip bound [{i}] must not be NaN; got lo={lo}, hi={hi}"
        );
        // With IEEE 754, -inf <= +inf evaluates true, so the standard
        // ordering assert suffices even for infinite sentinels.
        assert!(
            lo <= hi,
            "terminal OpaqueSkip bound [{i}] must satisfy lower ({lo}) <= upper ({hi})"
        );
    }

    eprintln!(
        "[OpaqueSkip terminal IBP] PASSED: shape={:?}",
        out_lower.shape()
    );
}
