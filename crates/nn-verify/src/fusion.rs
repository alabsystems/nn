// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Fusion equivalence verification via NY diamond DAG diff.
//!
//! Proves that a fused kernel produces the same output as the sequential
//! composition of its component kernels, for all inputs within given bounds.
//!
//! # Diamond DAG approach
//!
//! To verify `|K_fused(x) - K_sequential(x)| < epsilon`, we build a single
//! `GraphNetwork` where the shared input fans out to both computation paths,
//! with a `SubLayer` at the end computing the diff:
//!
//! ```text
//!          shared_input
//!         /            \
//!   fused_path    sequential_path
//!         \            /
//!        subtract (diff)
//! ```
//!
//! This avoids the Minkowski difference overestimate of independent propagation.
//! NY's CROWN relaxation captures the input correlation because both
//! paths share the same `NETWORK_INPUT` node.
//!
//! # References
//!
//! - Design: `designs/2026-02-25-verify-gamma-propagate-integration.md` (Step 5)
//! - NY diamond DAG support: `NY/tests/graph/alpha_crown_dag.rs`

use ny_api::BoundedTensor;
use ny_core::VerificationSoundnessMode;
use ny_propagate::layers::{AddConstantLayer, MulConstantLayer, SubLayer};
use ny_propagate::{GraphNetwork, GraphNode, Layer};
use nn_dsl::ir::KernelDef;

use crate::error::{StructuralError, VerifyError};
use crate::fusion_spec::validate_fusion_params;
use crate::graph::{scalar_array, translate_node, NodeValue, ParamBinding, TranslationContext};
use crate::soundness::soundness_for_graph;
use crate::util::get_value;
use crate::verify::{PropMethod, VerifyConfig};
use crate::verify_input::multi_scalar_input_bounds;

// Re-export types so `use crate::fusion::FusionSpec` etc. still works.
pub use crate::fusion_adain::{
    verify_ada_layer_norm_fusion, verify_ada_layer_norm_fusion_with_config,
    verify_adain_leaky_relu_fusion, verify_adain_leaky_relu_fusion_with_config,
    verify_adain_snake_fusion, verify_adain_snake_fusion_with_config, verify_all_named_fusions,
    verify_layer_norm_gelu_fusion, verify_layer_norm_gelu_fusion_with_config,
    verify_rms_norm_silu_mul_fusion, verify_rms_norm_silu_mul_fusion_with_config,
    NamedFusionBounds,
};
pub use crate::fusion_spec::{FusionSpec, FusionVerification};

/// Build a diamond DAG for verifying fusion equivalence of a kernel that
/// composes two sequential kernels.
///
/// The fused kernel takes `N` parameters. The sequential path runs `first`
/// (whose output feeds into `second`). Parameters are shared via indices:
///
/// - `first_param_indices[i]` maps `first.params[i]` to a shared input index
/// - `second_param_indices[i]` maps `second.params[i]` to a shared input index,
///   EXCEPT index `second_input_from_first` which takes `first`'s output.
///
/// # Arguments
///
/// * `fused` — The fused kernel (N params, all become variables in the diamond DAG)
/// * `first` — The first kernel in the sequential path (e.g., AdaIN K3)
/// * `second` — The second kernel (e.g., Snake K1)
/// * `num_shared_inputs` — Total number of shared input variables
/// * `first_param_indices` — Maps each param of `first` to a shared input index
/// * `second_param_indices` — Maps each param of `second` to a shared input index
///   (the entry at `second_input_from_first` is ignored)
/// * `second_input_from_first` — Which param of `second` receives `first`'s output
///
/// # Errors
///
/// Returns [`VerifyError`] if parameter index arrays have wrong lengths,
/// indices are out of range, or any kernel IR fails validation.
#[must_use = "returns a Result that may contain an error"]
pub fn build_fusion_diff_graph(spec: &FusionSpec<'_>) -> Result<GraphNetwork, VerifyError> {
    validate_fusion_params(spec)?;
    spec.fused.validate()?;
    spec.first.validate()?;
    spec.second.validate()?;

    let mut graph = GraphNetwork::new();

    // Create shared input SliceLayer nodes for multi-variable input.
    let input_names: Vec<String> = (0..spec.num_shared_inputs)
        .map(|i| format!("in_{i}"))
        .collect();
    for (i, name) in input_names.iter().enumerate() {
        graph.add_node(GraphNode::from_input(
            name.clone(),
            Layer::Slice(ny_propagate::layers::SliceLayer::new(0, i, i + 1)),
        ));
    }

    // --- Fused path (prefix "f_") ---
    let fused_param_names: Vec<Option<String>> =
        input_names.iter().map(|n| Some(n.clone())).collect();
    let fused_out = translate_kernel_path("f_", spec.fused, &fused_param_names, &mut graph)?;

    // --- Sequential path: first kernel (prefix "s1_") ---
    let first_param_names: Vec<Option<String>> = spec
        .first_param_indices
        .iter()
        .map(|&idx| Some(input_names[idx].clone()))
        .collect();
    let first_out = translate_kernel_path("s1_", spec.first, &first_param_names, &mut graph)?;

    // --- Sequential path: second kernel (prefix "s2_") ---
    // Wire second's inputs: most from shared inputs, one from first's output.
    let second_param_names: Vec<Option<String>> = spec
        .second_param_indices
        .iter()
        .enumerate()
        .map(|(i, &idx)| {
            if i == spec.second_input_from_first {
                Some(first_out.clone())
            } else {
                Some(input_names[idx].clone())
            }
        })
        .collect();
    let seq_out = translate_kernel_path("s2_", spec.second, &second_param_names, &mut graph)?;

    // --- Diff: fused - sequential ---
    graph.add_node(GraphNode::binary(
        "diff".to_string(),
        Layer::Sub(SubLayer),
        fused_out,
        seq_out,
    ));
    graph.set_output("diff".to_string());

    Ok(graph)
}

/// Translate all nodes of a kernel into a graph, returning the output node name.
///
/// Each kernel parameter is mapped to an existing graph node via `param_names`.
/// `None` entries should not be referenced by the kernel IR; they represent
/// parameters that are not wired (e.g., replaced by a previous kernel's output).
/// All node names in the emitted subgraph are prefixed with `prefix` to avoid
/// collisions when multiple kernels share a graph.
pub(crate) fn translate_kernel_path(
    prefix: &str,
    kernel: &KernelDef,
    param_names: &[Option<String>],
    graph: &mut GraphNetwork,
) -> Result<String, VerifyError> {
    let bindings = vec![ParamBinding::Variable; kernel.params.len()];
    let ctx = TranslationContext {
        prefix,
        bindings: &bindings,
        num_variables: kernel.params.len(),
        param_node_names: param_names,
        all_nodes: &kernel.nodes,
    };
    let mut values = Vec::with_capacity(kernel.nodes.len());
    for node in &kernel.nodes {
        let value = translate_node(&ctx, node.id.index(), &values, graph)?;
        values.push(value);
    }
    resolve_output(
        get_value(&values, kernel.output.index(), "fusion kernel output")?,
        &format!("{prefix}const_out"),
        graph,
    )
}

/// Resolve an output `NodeValue` to a graph node name.
/// If the output is a constant, create a two-node constant subgraph:
/// `MulConstant(0)` → `AddConstant(c)`, which maps any input to the constant `c`.
///
/// Returns `Err` if the constant value is NaN or Inf (per design doc #46:
/// all constant-fold paths must reject non-finite values).
pub(crate) fn resolve_output(
    value: &NodeValue,
    const_name: &str,
    graph: &mut GraphNetwork,
) -> Result<String, VerifyError> {
    match value {
        NodeValue::Variable(name) => Ok(name.clone()),
        NodeValue::Constant(val) => {
            // FiniteF32 guarantees val is finite — no NaN/Inf check needed.
            let v = val.get();
            // MulConstant(0) zeroes out the input, AddConstant(c) produces the constant.
            // This matches the pattern in graph_ops::ensure_variable_node.
            let zero_name = format!("{const_name}_zero");
            graph.add_node(GraphNode::from_input(
                zero_name.clone(),
                Layer::MulConstant(MulConstantLayer::scalar(0.0)),
            ));
            graph.add_node(GraphNode::new(
                const_name.to_string(),
                Layer::AddConstant(AddConstantLayer::new(scalar_array(v)?)),
                vec![zero_name],
            ));
            Ok(const_name.to_string())
        }
    }
}

/// Propagate bounds through a graph, trying CROWN first with IBP fallback.
///
/// Returns `(method, output_bounds, crown_fallback_reason)`:
/// - `PropMethod::Crown` if CROWN succeeded
/// - `PropMethod::Ibp` with the CROWN error message if CROWN failed and IBP succeeded
///
/// # Errors
///
/// Returns `VerifyError` if both CROWN and IBP fail (IBP error propagated).
#[must_use = "returns a Result that may contain an error"]
pub fn propagate_with_crown_fallback(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
) -> Result<(PropMethod, BoundedTensor, Option<String>), VerifyError> {
    // Mirror GraphNetwork::propagate_crown (alpha-CROWN, then fixed-slope CROWN,
    // then IBP) but thread a wall-clock deadline into the alpha optimizer so deep
    // models bail to sound best-bounds-so-far instead of running it unbounded.
    match graph
        .propagate_alpha_crown_with_config(input_bounds, &crate::verify::alpha_config_with_deadline())
    {
        Ok(crown_output) => Ok((PropMethod::Crown, floor_with_ibp(graph, input_bounds, crown_output), None)),
        Err(alpha_err) => match graph.propagate_crown_fixed_slope(input_bounds) {
            Ok(crown_output) => Ok((PropMethod::Crown, floor_with_ibp(graph, input_bounds, crown_output), None)),
            Err(_) => {
                // CROWN failed — fall back to IBP (will be loose for diff graphs)
                let ibp_output = graph.propagate_ibp(input_bounds)?;
                Ok((PropMethod::Ibp, ibp_output, Some(alpha_err.to_string())))
            }
        },
    }
}

/// Intersect a CROWN-family result with IBP elementwise so a deadline-truncated
/// or otherwise locally-loose CROWN bound is never *looser* than the (always
/// sound, always available) IBP bound.
///
/// CROWN's linear relaxation can be looser than IBP on individual output
/// coordinates — e.g. for bilinear/softmax (attention) ops, or when the
/// alpha-CROWN wall-clock deadline truncates optimization before the
/// best-so-far bounds reach IBP quality. Both CROWN and IBP are sound
/// over-approximations of the *same* quantity, so their per-element
/// intersection (`lower' = max(crown_lo, ibp_lo)`, `upper' = min(crown_hi,
/// ibp_hi)`) is still a sound enclosure and can only be tighter. This restores
/// the "CROWN is at least as tight as IBP" invariant the verifier relies on.
///
/// If IBP fails or the intersection is impossible (shape mismatch / NaN — which
/// would itself indicate corruption), the original CROWN result is returned
/// unchanged: this function only ever *tightens*, never weakens, the bound.
fn floor_with_ibp(
    graph: &GraphNetwork,
    input_bounds: &BoundedTensor,
    crown_output: BoundedTensor,
) -> BoundedTensor {
    let Ok(ibp_output) = graph.propagate_ibp(input_bounds) else {
        return crown_output;
    };
    match crown_output.intersection_per_element(&ibp_output) {
        Some((tightened, _disjoint)) => tightened,
        None => crown_output,
    }
}

/// Verify fusion equivalence: |K_fused(x) - K_sequential(x)| < epsilon.
///
/// Builds the diamond DAG and propagates bounds. The diff bounds should be
/// near zero for mathematically equivalent fusions. Non-zero diff indicates
/// either different formulas or floating-point reordering effects.
///
/// # Arguments
///
/// * `fused` — The fused kernel
/// * `first` — First kernel in the sequential path
/// * `second` — Second kernel in the sequential path
/// * `num_shared_inputs` — Number of shared input variables
/// * `first_param_indices` — Maps first's params to shared input indices
/// * `second_param_indices` — Maps second's params to shared input indices
/// * `second_input_from_first` — Which param of second receives first's output
/// * `variable_bounds` — Bounds for each shared input variable
/// * `epsilon` — Maximum tolerable absolute difference (must be finite and non-negative)
///
/// # Errors
///
/// Returns [`VerifyError::InvalidThreshold`] if `epsilon` is NaN, infinite,
/// or negative. Returns other [`VerifyError`] variants if bounds count
/// mismatches `num_shared_inputs`, diamond DAG construction fails, or bound
/// propagation produces non-finite results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_fusion_equivalence(
    spec: &FusionSpec<'_>,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
) -> Result<FusionVerification, VerifyError> {
    verify_fusion_equivalence_with_config(spec, variable_bounds, epsilon, &VerifyConfig::default())
}

/// Verify fusion equivalence with custom configuration.
///
/// The `config.require_sound()` flag is applied to the fusion result: if the
/// propagation used heuristic approximations, verification fails with
/// [`VerifyError::SoundnessRequired`]. The `config.escalation_threshold`
/// is not used because fusion always attempts CROWN first (see module docs).
///
/// # Errors
///
/// Returns [`VerifyError::InvalidThreshold`] if `epsilon` is NaN, infinite,
/// or negative. Returns other [`VerifyError`] variants if bounds count
/// mismatches `num_shared_inputs`, diamond DAG construction fails, or bound
/// propagation produces non-finite results.
#[must_use = "fusion verification result is computed but not used"]
pub fn verify_fusion_equivalence_with_config(
    spec: &FusionSpec<'_>,
    variable_bounds: &[(f32, f32)],
    epsilon: f32,
    config: &VerifyConfig,
) -> Result<FusionVerification, VerifyError> {
    if variable_bounds.len() != spec.num_shared_inputs {
        return Err(VerifyError::VariableBoundsMismatch {
            variable_count: spec.num_shared_inputs,
            bounds_count: variable_bounds.len(),
        });
    }
    // Epsilon must be finite and non-negative, matching the contract of
    // VerifyConfig::with_threshold. NaN silently bypasses comparisons
    // (design doc #66), +Inf trivially passes, negative trivially fails.
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(VerifyError::InvalidThreshold { value: epsilon });
    }

    let graph = build_fusion_diff_graph(spec)?;

    let input_bounds = multi_scalar_input_bounds(variable_bounds)?;

    // Fusion diff verification ALWAYS uses CROWN, not IBP.
    //
    // IBP computes intervals independently at each node and cannot capture
    // that both diamond DAG paths share the same input. This produces the
    // Minkowski difference [a-b, b-a] even for identical functions, which
    // can be arbitrarily wide.
    //
    // CROWN's linear relaxation propagates through the shared input node,
    // capturing the input correlation and producing tight diff bounds.
    //
    // We still run IBP as a fallback if CROWN fails.
    let (method, output_bounds, crown_fallback_reason) =
        propagate_with_crown_fallback(&graph, &input_bounds)?;

    let (diff_lower, diff_upper) = crate::util::bounds_min_max(&output_bounds);

    // IEEE 754 guard: NaN comparisons silently return false, bypassing validation.
    // Check finiteness explicitly before relational comparisons (design doc #66).
    if !diff_lower.is_finite() || !diff_upper.is_finite() {
        return Err(StructuralError::NonFiniteBounds {
            lower: diff_lower,
            upper: diff_upper,
        }
        .into());
    }

    let max_abs_diff = diff_lower.abs().max(diff_upper.abs());

    // Fusion equivalence graphs don't introduce comparison approximations —
    // they only contain arithmetic/transcendental ops from the kernel pair.
    let provenance = soundness_for_graph(&graph, &method, Some(&input_bounds), false)?;
    let soundness_mode = provenance.mode();
    if config.require_sound() && soundness_mode == VerificationSoundnessMode::Heuristic {
        return Err(VerifyError::SoundnessRequired {
            kernel_name: spec.fused.name.clone(),
        });
    }

    Ok(FusionVerification {
        fused_kernel_name: spec.fused.name.clone(),
        diff_lower,
        diff_upper,
        max_abs_diff,
        within_epsilon: max_abs_diff <= epsilon,
        epsilon,
        method,
        crown_fallback_reason,
        soundness_mode,
    })
}

#[cfg(test)]
#[path = "fusion_tests.rs"]
mod tests;
