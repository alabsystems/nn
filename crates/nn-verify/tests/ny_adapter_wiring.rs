// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the `ny_api` capability adapters wired into nn's real
//! verification paths.
//!
//! Each adapter is exercised through the pipeline entry point that now reaches
//! it — not through the adapter module's own unit tests — on a real nn fixture
//! (a lowered `KernelDef` or a `TensorBlockBuilder` model), so these fail if the
//! wiring regresses even when the adapter itself still works.
//!
//! * `conformance` — [`nn_verify::pipeline::verify_tensor_and_record_with_coverage`]
//!   and [`nn_verify::pipeline::verify_and_record_full_with_coverage`].
//! * `ladder` — [`nn_verify::VerifyRequest::verify_spec_laddered`], including
//!   the "never looser than the single-method path" soundness comparison.
//! * `session` — [`nn_verify::pipeline::verify_specs_in_session`], including the
//!   cached-vs-uncached verdict identity proof.
//! * `equivalence` — [`nn_verify::pipeline::verify_tensor_kernels_equivalent`]
//!   against the hand-rolled proof in `resblock_equivalence` on a shared
//!   fixture.
//!
//! All status recording goes to an in-memory `VerifyStatus`; nothing here
//! touches the workspace `nn_verify_status_*.json` files.

use ndarray::{ArrayD, IxDyn};
use nn_dsl::ir::KernelDef;
use nn_dsl::tensor_block_builder::TensorBlockBuilder;
use nn_dsl::tensor_ir::TensorKernelDef;
use nn_verify::pipeline::{
    verdict_fingerprint, verify_and_record_full_with_coverage, verify_specs_in_session,
    verify_tensor_and_record_with_coverage, verify_tensor_kernels_equivalent, OpCoverageMode,
};
use nn_verify::{
    scalar_input_bounds, Bound, BoundedTensor, TensorParamBinding, VerificationSpec, VerifyConfig,
    VerifyRequest, VerifyStatus,
};

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

/// Lower a Rust source function to nn's kernel IR — the same path `verify_all`
/// kernel configs go through.
fn parse_kernel(src: &str) -> KernelDef {
    let func: syn::ItemFn = syn::parse_str(src).expect("parse Rust source");
    nn_dsl::lower::Lowerer::lower_fn(&func).expect("lower to KernelDef")
}

/// A GELU kernel: a real nn activation whose IBP bounds are loose enough that a
/// tight output spec is not IBP-provable, which is what makes it a ladder
/// fixture rather than a trivially-decided one.
fn gelu_kernel() -> KernelDef {
    nn_dsl::build_gelu_kernel().expect("build gelu kernel")
}

/// A scalar kernel whose lowering emits a data-dependent `Select` on a variable
/// condition. The NN→ny translator has no sound relaxation for that shape and
/// emits a `Where` layer, which ny's conformance gate classifies `Unsupported`.
///
/// Deliberately not a ReLU/LeakyReLU shape: those are pattern-matched into
/// native layers by `graph_ops::select` and would be soundly covered.
fn hard_gate_kernel() -> KernelDef {
    parse_kernel("fn hard_gate(x: f32, t: f32) -> f32 { if x > t { 3.0 } else { -2.0 } }")
}

/// A small Linear -> ReLU -> Linear tensor model built through nn's own
/// `TensorBlockBuilder`, with every weight bound as a constant tensor.
fn fc_relu_fc_model() -> (TensorKernelDef, Vec<TensorParamBinding>, BoundedTensor) {
    let mut b = TensorBlockBuilder::new("fc_relu_fc");
    let x = b.add_input("x", &[4]);
    let w1 = b.add_input("w1", &[3, 4]);
    let b1 = b.add_input("b1", &[3]);
    let h = b.add_linear(x, w1, Some(b1), &[3]);
    let a = b.add_relu(h, &[3]);
    let w2 = b.add_input("w2", &[2, 3]);
    let b2 = b.add_input("b2", &[2]);
    let out = b.add_linear(a, w2, Some(b2), &[2]);
    let def = b.build(out).expect("valid fc_relu_fc model");

    let bindings = vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[3, 4]), 0.25f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[3]), 0.1f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[2, 3]), -0.5f32)),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[2]), 0.0f32)),
    ];

    let input = BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[4]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[4]), 1.0f32),
    )
    .expect("valid input bounds");

    (def, bindings, input)
}

// ---------------------------------------------------------------------------
// ny_conformance: op soundness coverage reachable from the pipeline
// ---------------------------------------------------------------------------

#[test]
fn coverage_off_is_the_historical_behavior() {
    let (def, bindings, input) = fc_relu_fc_model();
    let mut status = VerifyStatus::default();

    let result = nn_verify::verify_tensor_and_record_with_config(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("fc_relu_fc_off"),
        &VerifyConfig::default(),
    )
    .expect("tensor pipeline");

    assert!(
        result.op_coverage.is_none(),
        "the default entry point must not run the conformance gate"
    );
    let entry = status.kernel("fc_relu_fc_off").expect("status entry");
    assert!(
        entry
            .soundness_justification
            .as_deref()
            .is_none_or(|j| !j.contains("ny conformance")),
        "no coverage summary should be recorded when the gate is off"
    );
}

#[test]
fn tensor_pipeline_reports_and_records_op_coverage() {
    let (def, bindings, input) = fc_relu_fc_model();
    let mut status = VerifyStatus::default();

    let result = verify_tensor_and_record_with_coverage(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("fc_relu_fc_report"),
        &VerifyConfig::default(),
        OpCoverageMode::Report,
    )
    .expect("tensor pipeline with coverage");

    let coverage = result
        .op_coverage
        .as_ref()
        .expect("Report mode must attach a coverage tally");
    assert!(
        coverage.total() >= 3,
        "a Linear->ReLU->Linear model has at least 3 classified ops, got {coverage:?}"
    );
    assert_eq!(
        coverage.unsupported, 0,
        "this model has no soundness holes, got {:?}",
        coverage.unsupported_layers
    );
    assert!(coverage.is_fully_verifiable());

    // The tally is persisted next to the verdict, which is the point: the
    // report is visible in the status output, not only in a return value.
    let justification = status
        .kernel("fc_relu_fc_report")
        .expect("status entry")
        .soundness_justification
        .as_deref()
        .expect("coverage summary recorded on the status entry");
    assert!(
        justification.contains("ny conformance:"),
        "justification should carry the coverage summary, got {justification:?}"
    );
    assert!(
        justification.contains("unsupported 0"),
        "justification should spell out the unsupported bucket, got {justification:?}"
    );
}

#[test]
fn scalar_pipeline_surfaces_an_unsupported_op_in_the_status_output() {
    let kernel = hard_gate_kernel();
    let bounds = nn_verify::ScalarInputBounds::new(-5.0, 5.0).expect("valid bounds");
    let mut status = VerifyStatus::default();

    let result = verify_and_record_full_with_coverage(
        &mut status,
        &kernel,
        &[0.0],
        bounds,
        Some("hard_gate_report"),
        &VerifyConfig::default(),
        OpCoverageMode::Report,
    )
    .expect("scalar pipeline with coverage");

    let coverage = result
        .op_coverage
        .as_ref()
        .expect("Report mode must attach a coverage tally");
    assert!(
        coverage.unsupported > 0,
        "a variable-condition Select lowers to a `Where` layer, which is an \
         Unsupported-class op; got {coverage:?}"
    );
    assert!(
        coverage.unsupported_layers.iter().any(|l| l == "Where"),
        "the hole should be named, got {:?}",
        coverage.unsupported_layers
    );

    // The whole point of the wiring: the hole is legible in the status file.
    let justification = status
        .kernel("hard_gate_report")
        .expect("status entry")
        .soundness_justification
        .as_deref()
        .expect("coverage summary recorded on the status entry");
    assert!(
        justification.contains("Where"),
        "the unsupported op must be named in the status output, got {justification:?}"
    );
}

#[test]
fn coverage_gate_refuses_before_anything_is_recorded() {
    let kernel = hard_gate_kernel();
    let bounds = nn_verify::ScalarInputBounds::new(-5.0, 5.0).expect("valid bounds");
    let mut status = VerifyStatus::default();

    let err = verify_and_record_full_with_coverage(
        &mut status,
        &kernel,
        &[0.0],
        bounds,
        Some("hard_gate_gated"),
        &VerifyConfig::default(),
        OpCoverageMode::Gate,
    )
    .expect_err("Gate must refuse a model with an Unsupported-class op");

    let text = err.to_string();
    assert!(
        text.contains("Where"),
        "the refusal must name the offending op, got {text:?}"
    );
    // Fail-closed: no verdict may survive a refused run.
    assert!(
        status.kernel("hard_gate_gated").is_none(),
        "Gate refusal must not leave a recorded verdict behind"
    );
}

#[test]
fn coverage_gate_passes_a_fully_covered_model() {
    let (def, bindings, input) = fc_relu_fc_model();
    let mut status = VerifyStatus::default();

    let result = verify_tensor_and_record_with_coverage(
        &mut status,
        &def,
        &bindings,
        &input,
        Some("fc_relu_fc_gated"),
        &VerifyConfig::default(),
        OpCoverageMode::Gate,
    )
    .expect("Gate must admit a model with no Unsupported ops");

    assert!(result
        .op_coverage
        .as_ref()
        .expect("coverage attached")
        .is_fully_verifiable());
    assert!(status.kernel("fc_relu_fc_gated").is_some());
}

// ---------------------------------------------------------------------------
// ny_ladder: escalating ladder from VerifyRequest
// ---------------------------------------------------------------------------

/// Widest output-bound width a verdict carries, or `None` when it carries none.
fn max_result_width(result: &nn_verify::VerificationResult) -> Option<f32> {
    result_bounds(result).and_then(|bounds| {
        bounds
            .iter()
            .map(Bound::width)
            .fold(None, |acc: Option<f32>, w| {
                Some(acc.map_or(w, |a| a.max(w)))
            })
    })
}

/// Enclosing box (min lower, max upper) of a verdict's output bounds.
fn result_envelope(result: &nn_verify::VerificationResult) -> Option<(f32, f32)> {
    result_bounds(result).and_then(|bounds| {
        bounds.iter().fold(None, |acc: Option<(f32, f32)>, b| {
            Some(acc.map_or((b.lower(), b.upper()), |(lo, hi)| {
                (lo.min(b.lower()), hi.max(b.upper()))
            }))
        })
    })
}

fn result_bounds(result: &nn_verify::VerificationResult) -> Option<&[Bound]> {
    use nn_verify::VerificationResult as R;
    match result {
        R::Verified { output_bounds, .. } => Some(output_bounds),
        R::Unknown { bounds, .. } => Some(bounds),
        R::Timeout { partial_bounds, .. } => partial_bounds.as_deref(),
        R::Violated { .. } => None,
    }
}

/// A two-variable nn kernel whose lowering has the shape IBP cannot handle: the
/// same value flows down two paths that must cancel.
///
/// `(a + b) - b` is the residual/skip pattern — add a tensor in, subtract it
/// back out. IBP treats the two occurrences of `b` as independent intervals and
/// widens to `[-3, 3]` over `a, b ∈ [-1, 1]`; a CROWN-family rung carries the
/// linear dependence and recovers the true `[-1, 1]`.
fn residual_cancel_kernel() -> KernelDef {
    parse_kernel("fn residual_cancel(a: f32, b: f32) -> f32 { (a + b) - b }")
}

/// The gate-mix shape: `x*g + x*(1-g)`, algebraically `x`, but split across two
/// correlated multiplicative paths. CROWN tightens it without fully recovering
/// `x`, which is the "escalated and tightened but still unproven" case.
fn gate_mix_kernel() -> KernelDef {
    parse_kernel("fn gate_mix(x: f32, g: f32) -> f32 { x * g + x * (1.0 - g) }")
}

/// The ladder config the opt-in tests use: force the ladder to keep climbing
/// while the property is unproven (the default `1e6` threshold is far above
/// these fixtures' bound widths, so with defaults every rung after IBP is
/// skipped as "not worth the cost").
fn climbing_ladder() -> nn_verify::verify_request::LadderConfig {
    nn_verify::verify_request::LadderConfig {
        escalation_width_threshold: 0.0,
        timeout_ms: Some(20_000),
        ..nn_verify::verify_request::LadderConfig::default()
    }
}

#[test]
fn ladder_is_never_looser_than_the_single_method_path() {
    let both_vars = [
        nn_verify::ParamBinding::Variable,
        nn_verify::ParamBinding::Variable,
    ];

    // (kernel, variable bounds, required output box, closed form, sample range)
    let fixtures: Vec<(KernelDef, Vec<(f32, f32)>, Vec<Bound>, fn(f32, f32) -> f32)> = vec![
        (
            residual_cancel_kernel(),
            vec![(-1.0, 1.0), (-1.0, 1.0)],
            vec![Bound::new(-2.5, 2.5)],
            |a, b| (a + b) - b,
        ),
        (
            gate_mix_kernel(),
            vec![(-2.0, 2.0), (0.0, 1.0)],
            vec![Bound::new(-2.5, 2.5)],
            |x, g| x * g + x * (1.0 - g),
        ),
    ];

    for (kernel, vb, required, closed_form) in fixtures {
        let name = kernel.name.clone();

        let single = VerifyRequest::new(&kernel)
            .bindings(&both_vars)
            .variable_bounds(&vb)
            .required_output_bounds(&required)
            .verify_spec()
            .unwrap_or_else(|e| panic!("{name}: single-method spec verification: {e}"));

        let laddered = VerifyRequest::new(&kernel)
            .bindings(&both_vars)
            .variable_bounds(&vb)
            .required_output_bounds(&required)
            .ladder(climbing_ladder())
            .verify_spec_laddered()
            .unwrap_or_else(|e| panic!("{name}: laddered spec verification: {e}"));

        assert_eq!(
            laddered.rungs[0].method,
            nn_verify::verify_request::MethodUsed::Ibp,
            "{name}: the first rung is always IBP, matching the single-method path's first step"
        );
        assert!(
            laddered.summary.contains("ladder:"),
            "{name}: summary should be the ny_ladder rendering, got {:?}",
            laddered.summary
        );

        // NOT WEAKER: the ladder cannot lose a property the single-method path
        // proved.
        if single.result.is_verified() {
            assert!(
                laddered.is_verified(),
                "{name}: the ladder must not lose a property the single-method path proved"
            );
        }

        // NOT LOOSER: the laddered enclosure is at least as tight.
        let single_w = max_result_width(&single.result)
            .unwrap_or_else(|| panic!("{name}: single verdict carries bounds"));
        let ladder_w = max_result_width(&laddered.result)
            .unwrap_or_else(|| panic!("{name}: laddered verdict carries bounds"));
        assert!(
            ladder_w <= single_w * (1.0 + 1e-5) + 1e-6,
            "{name}: laddered enclosure must not be looser: ladder {ladder_w} vs single {single_w}"
        );

        // SOUND: the laddered enclosure still contains real evaluations of the
        // kernel over the same region. A tighter bound is only correct if it
        // still encloses.
        let envelope = result_envelope(&laddered.result)
            .unwrap_or_else(|| panic!("{name}: laddered verdict carries bounds"));
        let steps = 40;
        for i in 0..=steps {
            for j in 0..=steps {
                let a = vb[0].0 + (vb[0].1 - vb[0].0) * (i as f32) / (steps as f32);
                let b = vb[1].0 + (vb[1].1 - vb[1].0) * (j as f32) / (steps as f32);
                let y = closed_form(a, b);
                assert!(
                    y >= envelope.0 - 1e-4 && y <= envelope.1 + 1e-4,
                    "{name}: laddered bounds {envelope:?} must enclose f({a}, {b}) = {y}"
                );
            }
        }
    }
}

#[test]
fn ladder_escalates_past_ibp_and_names_the_deciding_rung() {
    let kernel = residual_cancel_kernel();
    let both_vars = [
        nn_verify::ParamBinding::Variable,
        nn_verify::ParamBinding::Variable,
    ];
    let vb = [(-1.0f32, 1.0f32), (-1.0, 1.0)];
    // IBP widens `(a + b) - b` to [-3, 3] and cannot prove this box; a CROWN
    // rung recovers [-1, 1] and can.
    let required = vec![Bound::new(-2.5, 2.5)];

    let single = VerifyRequest::new(&kernel)
        .bindings(&both_vars)
        .variable_bounds(&vb)
        .required_output_bounds(&required)
        .verify_spec()
        .expect("single-method spec verification");
    assert!(
        !single.result.is_verified(),
        "precondition: the fixed single-method path leaves this spec unproven"
    );

    let laddered = VerifyRequest::new(&kernel)
        .bindings(&both_vars)
        .variable_bounds(&vb)
        .required_output_bounds(&required)
        .ladder(climbing_ladder())
        .verify_spec_laddered()
        .expect("laddered spec verification");

    assert!(
        laddered.is_verified(),
        "the ladder must decide what the single fixed method could not; rungs = {:?}",
        laddered.rungs
    );
    assert!(
        laddered.rungs_run() >= 2,
        "the ladder must climb past IBP here; rungs = {:?}",
        laddered.rungs
    );
    assert!(
        !laddered.rungs[0].verified,
        "the IBP rung must be the one that failed (precondition of this fixture)"
    );
    // The deciding rung is reported, and mapped into nn's own PropMethod
    // vocabulary when it is a propagation method nn exposes.
    assert_eq!(
        laddered.method_used,
        laddered.rungs.last().expect("at least one rung").method,
        "method_used must name the rung that decided"
    );
    assert_eq!(
        laddered.method,
        Some(nn_verify::PropMethod::AlphaCrown),
        "the alpha-CROWN rung decides this fixture; got {:?}",
        laddered.method_used
    );
}

#[test]
fn ladder_default_config_leaves_verify_spec_untouched() {
    let kernel = gelu_kernel();
    let input = scalar_input_bounds(-5.0, 5.0).expect("input bounds");
    let required = vec![Bound::new(-10.0, 10.0)];

    // Arming the ladder must not change what `verify_spec` / `verify_bounds` do.
    let bounds_with_ladder = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input)
        .ladder_default()
        .verify_bounds()
        .expect("bounds path ignores the ladder setting");
    let bounds_plain = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input)
        .verify_bounds()
        .expect("bounds path");
    assert_eq!(bounds_with_ladder.output_lower, bounds_plain.output_lower);
    assert_eq!(bounds_with_ladder.output_upper, bounds_plain.output_upper);
    assert_eq!(bounds_with_ladder.method, bounds_plain.method);

    // And a wide spec is decided on the very first rung, with no escalation.
    let laddered = VerifyRequest::new(&kernel)
        .constant_params(&[])
        .input_bounds(&input)
        .required_output_bounds(&required)
        .ladder_default()
        .verify_spec_laddered()
        .expect("laddered spec verification");
    assert!(laddered.is_verified(), "[-10, 10] is IBP-provable for GELU");
    assert_eq!(laddered.rungs_run(), 1, "no escalation needed");
    assert_eq!(laddered.method, Some(nn_verify::PropMethod::Ibp));
}

// ---------------------------------------------------------------------------
// ny_session: load-once / verify-many, cached == uncached
// ---------------------------------------------------------------------------

/// Build the same batch `verify_all`'s session tier builds: one network, many
/// specs over different input regions of the same graph.
fn session_batch() -> (ny_propagate::GraphNetwork, Vec<VerificationSpec>) {
    let kernel = gelu_kernel();
    let net = nn_verify::kernel_to_graph(&kernel, &[]).expect("translate gelu");

    let regions = [(-5.0f32, 5.0f32), (-10.0, 10.0), (-1.0, 1.0), (-5.0, 5.0)];
    let specs = regions
        .iter()
        .map(|&(lo, hi)| {
            VerificationSpec::new(vec![Bound::new(lo, hi)], vec![Bound::new(-12.0, 12.0)])
                .expect("valid spec")
        })
        .collect();

    (net, specs)
}

#[test]
fn session_cached_path_returns_identical_verdicts_to_uncached() {
    let (net_cached, specs) = session_batch();
    let (net_uncached, _) = session_batch();
    let config = nn_verify::pipeline::session::PropagationConfig::default();

    let cached = verify_specs_in_session(net_cached, &specs, config.clone(), true);
    let uncached = verify_specs_in_session(net_uncached, &specs, config, false);

    assert_eq!(cached.stats.queries, specs.len());
    assert_eq!(uncached.stats.queries, specs.len());
    assert_eq!(
        uncached.stats.cache_hits, 0,
        "caching disabled must never serve a cached verdict"
    );
    // The batch repeats (-5, 5), so the cache is genuinely exercised.
    assert!(
        cached.stats.cache_hits >= 1,
        "the repeated spec should hit the verdict cache, stats = {:?}",
        cached.stats
    );

    // IDENTICAL VERDICTS: full serialized verdicts, not just variants — this
    // covers the output bounds, the soundness provenance and the method tag.
    assert_eq!(
        cached.fingerprints(),
        uncached.fingerprints(),
        "the cached path must return byte-identical verdicts to the uncached one"
    );
    assert_eq!(cached.verified_count(), uncached.verified_count());
    assert_eq!(cached.error_count(), 0, "gelu specs should all decide");
}

#[test]
fn session_repeated_spec_replays_the_same_verdict() {
    let (net, specs) = session_batch();
    let config = nn_verify::pipeline::session::PropagationConfig::default();
    let run = verify_specs_in_session(net, &specs, config, true);

    // Specs 0 and 3 are the same region; their verdicts must be identical, and
    // the second of them was the cache hit.
    let first = run.verdicts[0].as_ref().expect("spec 0 decided");
    let repeat = run.verdicts[3].as_ref().expect("spec 3 decided");
    assert_eq!(
        verdict_fingerprint(first),
        verdict_fingerprint(repeat),
        "a cache hit must replay the stored verdict verbatim"
    );
}

#[test]
fn session_matches_the_one_shot_verifier_verdict() {
    // Load-once/verify-many must not change the answer relative to verifying
    // each spec against a freshly loaded network.
    let (net, specs) = session_batch();
    let config = nn_verify::pipeline::session::PropagationConfig::default();
    let batched = verify_specs_in_session(net, &specs, config.clone(), true);

    let one_shot: Vec<Option<String>> = specs
        .iter()
        .map(|spec| {
            let (net, _) = session_batch();
            let run =
                verify_specs_in_session(net, std::slice::from_ref(spec), config.clone(), false);
            run.fingerprints().into_iter().next().flatten()
        })
        .collect();

    assert_eq!(
        batched.fingerprints(),
        one_shot,
        "session verdicts must match per-spec one-shot verification"
    );
}

// ---------------------------------------------------------------------------
// ny_equivalence: pipeline-level difference-network proving
// ---------------------------------------------------------------------------

/// Reproduces `resblock_equivalence`'s style-projection fixture: a single
/// `Linear(style) -> [2*C]`, which is what both the sequential and the absorbed
/// forms compute (absorption is an affine identity).
fn style_projection(style_dim: usize, channels: usize) -> TensorKernelDef {
    let output_dim = 2 * channels;
    let mut b = TensorBlockBuilder::new("style_proj");
    let style = b.add_input("style", &[style_dim]);
    let weight = b.add_input("weight", &[output_dim, style_dim]);
    let bias = b.add_input("bias", &[output_dim]);
    let out = b.add_linear(style, weight, Some(bias), &[output_dim]);
    b.build(out).expect("valid style projection graph")
}

const STYLE_DIM: usize = 8;
const STYLE_CHANNELS: usize = 4;
const STYLE_WEIGHT_MAG: f32 = 0.5;

fn style_bindings() -> Vec<TensorParamBinding> {
    let output_dim = 2 * STYLE_CHANNELS;
    vec![
        TensorParamBinding::Variable,
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(
            IxDyn(&[output_dim, STYLE_DIM]),
            STYLE_WEIGHT_MAG,
        )),
        TensorParamBinding::ConstantTensor(ArrayD::from_elem(IxDyn(&[output_dim]), 0.0f32)),
    ]
}

fn style_input_bounds() -> BoundedTensor {
    BoundedTensor::new(
        ArrayD::from_elem(IxDyn(&[STYLE_DIM]), -1.0f32),
        ArrayD::from_elem(IxDyn(&[STYLE_DIM]), 1.0f32),
    )
    .expect("valid style input bounds")
}

#[test]
fn facade_equivalence_matches_the_hand_rolled_resblock_proof() {
    let epsilon = 1e-3f32;

    // The specialized proof, untouched and still the reference.
    let hand_rolled = nn_verify::resblock_equivalence::verify_style_absorption_equivalence(
        STYLE_DIM,
        STYLE_CHANNELS,
        STYLE_WEIGHT_MAG,
        epsilon,
    )
    .expect("hand-rolled style absorption proof");

    // The same fixture through the pipeline-level ny_equivalence routing, on
    // the same propagation rung (plain CROWN) the hand-rolled proof uses.
    let f = style_projection(STYLE_DIM, STYLE_CHANNELS);
    let g = style_projection(STYLE_DIM, STYLE_CHANNELS);
    let routed = verify_tensor_kernels_equivalent(
        &f,
        &style_bindings(),
        &g,
        &style_bindings(),
        &style_input_bounds(),
        epsilon,
        // The documented default rung, which is exactly what the hand-rolled
        // proof propagates through its difference network.
        nn_verify::pipeline::default_equivalence_method(),
    )
    .expect("routed equivalence proof");

    assert!(
        hand_rolled.within_epsilon,
        "precondition: the hand-rolled proof proves this fixture equivalent"
    );
    assert!(
        routed.within_epsilon,
        "the routed facade must prove the same fixture equivalent, got {}",
        routed.summary
    );

    // NOT LOOSER: the routed worst-case difference is no larger than the
    // hand-rolled one (both are `max(|lo|, |hi|)` over the difference network).
    assert!(
        routed.max_abs_diff <= hand_rolled.max_abs_diff + 1e-9,
        "routed diff bound {} must not be looser than hand-rolled {}",
        routed.max_abs_diff,
        hand_rolled.max_abs_diff
    );

    // SOUND: both enclose the true difference, which is exactly 0 for two
    // identical graphs.
    assert!(
        routed.max_abs_diff >= 0.0 && routed.max_abs_diff <= epsilon,
        "routed diff bound {} must enclose the true 0 difference and stay within \
         epsilon {epsilon}",
        routed.max_abs_diff
    );
    assert!(routed.summary.contains("equivalent"), "{}", routed.summary);
}

#[test]
fn facade_equivalence_refuses_a_real_output_gap() {
    // Same structure, different constants: `g`'s projection weights are 0.25
    // larger per element. This is the shape the routed facade exists for — an
    // optimized/re-parameterized graph against its reference — and it is why
    // `f` and `g` get their own bindings.
    let f = style_projection(STYLE_DIM, STYLE_CHANNELS);
    let g = style_projection(STYLE_DIM, STYLE_CHANNELS);
    let output_dim = 2 * STYLE_CHANNELS;

    let mut g_bindings = style_bindings();
    g_bindings[1] = TensorParamBinding::ConstantTensor(ArrayD::from_elem(
        IxDyn(&[output_dim, STYLE_DIM]),
        STYLE_WEIGHT_MAG + 0.25,
    ));

    let routed = verify_tensor_kernels_equivalent(
        &f,
        &style_bindings(),
        &g,
        &g_bindings,
        &style_input_bounds(),
        1e-3,
        nn_verify::pipeline::default_equivalence_method(),
    )
    .expect("routed equivalence run on differing graphs");

    assert!(
        !routed.within_epsilon,
        "a 0.25-per-weight gap cannot be proven equivalent at eps=1e-3: {}",
        routed.summary
    );
    assert!(
        routed.summary.contains("not proven"),
        "summary should say the proof did not go through, got {}",
        routed.summary
    );

    // SOUND: the reported bound must ENCLOSE the true worst-case difference.
    // Each of the 8 outputs differs by 0.25 * sum(style), and style ranges over
    // [-1, 1]^8, so the true worst case is 0.25 * 8 = 2.0.
    let true_worst = 0.25 * STYLE_DIM as f32;
    assert!(
        routed.max_abs_diff >= true_worst - 1e-4,
        "reported diff bound {} must enclose the true worst-case {true_worst}",
        routed.max_abs_diff
    );

    // The same fixture verifies once epsilon is loose enough — the refusal
    // above is about the size of the gap, not a broken wiring.
    let loose = verify_tensor_kernels_equivalent(
        &f,
        &style_bindings(),
        &g,
        &g_bindings,
        &style_input_bounds(),
        true_worst * 2.0,
        nn_verify::pipeline::default_equivalence_method(),
    )
    .expect("routed equivalence run at a loose epsilon");
    assert!(loose.within_epsilon, "{}", loose.summary);
}
