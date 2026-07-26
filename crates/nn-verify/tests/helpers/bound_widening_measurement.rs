// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Bound widening measurement: per-layer CROWN vs full-model CROWN.
//!
//! Measures how much wider per-layer CROWN bounds are compared to full-model
//! CROWN bounds for an MLP at D=64. This data supports the per-layer
//! certificate composition approach in #1762 (design doc Direction 1).
//!
//! Run: `cargo test -p nn-verify --test bound_widening_measurement`
//! Report: `reports/research/2026-03-10-bound-widening-measurement-methodology.md`

use ny_propagate::layers::{LinearLayer, ReLULayer, SigmoidLayer};
use ny_propagate::{AlphaCrownConfig, GradientMethod, GraphNetwork, GraphNode, Layer};
use nn_verify::BoundedTensor;
use ndarray::{arr1, arr2, Array1, Array2, ArrayD, IxDyn};

/// Deterministic pseudo-random LCG state advance.
fn lcg_next(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    ((*state >> 33) as f32) / (u32::MAX as f32) - 0.5
}

/// Xavier-uniform weight initialization.
///
/// `scale_factor` controls the magnitude. 1.0 = standard Xavier
/// (sqrt(2 / (fan_in + fan_out))). Higher values stress-test bound widening.
fn xavier_weights(rows: usize, cols: usize, seed: u64, scale_factor: f32) -> Array2<f32> {
    let xavier_scale = (2.0 / (rows + cols) as f32).sqrt() * scale_factor;
    let mut w = Array2::zeros((rows, cols));
    let mut rng_state = seed;
    for elem in w.iter_mut() {
        *elem = lcg_next(&mut rng_state) * xavier_scale;
    }
    w
}

/// Random bias initialization centered around `center` with range `spread`.
///
/// Produces diverse per-element biases so that after a linear transform,
/// some pre-ReLU intervals are positive (pass-through), some negative
/// (zero-killed), and some mixed (where CROWN relaxation differs from IBP).
fn random_bias(d: usize, seed: u64, center: f32, spread: f32) -> Array1<f32> {
    let mut b = Array1::zeros(d);
    let mut rng_state = seed;
    for elem in b.iter_mut() {
        *elem = center + lcg_next(&mut rng_state) * spread;
    }
    b
}

/// Build a full 3-layer graph: Linear(D,D) → ReLU → Linear(D,D).
fn build_full_model(d: usize, scale: f32) -> GraphNetwork {
    let w1 = xavier_weights(d, d, 42, scale);
    let b1 = Array1::zeros(d);
    let w2 = xavier_weights(d, d, 137, scale);
    let b2 = Array1::zeros(d);

    let linear1 = LinearLayer::new(w1, Some(b1)).expect("linear1");
    let relu = ReLULayer;
    let linear2 = LinearLayer::new(w2, Some(b2)).expect("linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(relu),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["relu".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// Build a single-layer graph for per-layer verification.
fn build_single_linear(d_in: usize, d_out: usize, seed: u64, scale: f32) -> GraphNetwork {
    let w = xavier_weights(d_out, d_in, seed, scale);
    let b = Array1::zeros(d_out);
    let linear = LinearLayer::new(w, Some(b)).expect("linear");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");
    graph
}

/// Build a single ReLU graph.
fn build_single_relu() -> GraphNetwork {
    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("relu", Layer::ReLU(ReLULayer)));
    graph.set_output("relu");
    graph
}

/// Create uniform BoundedTensor with range [-r, +r].
fn uniform_input(shape: &[usize], range: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(shape), -range);
    let upper = ArrayD::from_elem(IxDyn(shape), range);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

/// Create positive-only BoundedTensor with range [0, +r].
///
/// Positive inputs prevent exponential bound collapse through ReLU chains.
/// With symmetric [-r, +r] inputs, each ReLU halves the effective width
/// because it clips the negative half. Positive inputs keep bounds in
/// the active regime of ReLU.
fn positive_input(shape: &[usize], range: f32) -> BoundedTensor {
    let lower = ArrayD::from_elem(IxDyn(shape), 0.0_f32);
    let upper = ArrayD::from_elem(IxDyn(shape), range);
    BoundedTensor::new(lower, upper).expect("valid bounds")
}

/// Compute mean bound width across all elements.
fn mean_width(bt: &BoundedTensor) -> f64 {
    let (lo, hi) = bt.lower_upper();
    let n = lo.len();
    let sum: f64 = lo
        .iter()
        .zip(hi.iter())
        .map(|(l, h)| f64::from(*h) - f64::from(*l))
        .sum();
    sum / n as f64
}

/// Per-layer CROWN: verify each layer independently, chain bounds.
fn per_layer_crown(d: usize, input: &BoundedTensor, scale: f32) -> BoundedTensor {
    // Layer 1: Linear(D, D)
    let graph1 = build_single_linear(d, d, 42, scale);
    let out1 = graph1.propagate_crown(input).expect("layer1 CROWN");

    // Layer 2: ReLU
    let graph2 = build_single_relu();
    let out2 = graph2.propagate_crown(&out1).expect("relu CROWN");

    // Layer 3: Linear(D, D)
    let graph3 = build_single_linear(d, d, 137, scale);
    graph3.propagate_crown(&out2).expect("layer3 CROWN")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Run the bound-widening measurement at a given dimension and weight scale.
fn measure_widening(d: usize, scale: f32, input_range: f32) {
    let input = uniform_input(&[d], input_range);
    let graph = build_full_model(d, scale);
    let full_crown = graph.propagate_crown(&input).expect("full-model CROWN");
    let perlayer = per_layer_crown(d, &input, scale);
    let ibp = graph.propagate_ibp(&input).expect("full-model IBP");

    let w_full = mean_width(&full_crown);
    let w_perlayer = mean_width(&perlayer);
    let w_ibp = mean_width(&ibp);
    let widening_ratio = if w_full > 1e-10 {
        w_perlayer / w_full
    } else {
        1.0
    };

    eprintln!(
        "D={d} s={scale:.1} r={input_range:.0}: CROWN={w_full:.4} PL={w_perlayer:.4} \
               IBP={w_ibp:.4} ratio={widening_ratio:.2}x"
    );

    assert!(w_full > 0.0, "full-model CROWN width should be positive");
    assert!(w_perlayer > 0.0, "per-layer CROWN width should be positive");
    assert!(w_ibp > 0.0, "IBP width should be positive");
    assert!(
        w_perlayer.is_finite(),
        "per-layer CROWN width should be finite"
    );
    assert!(w_full <= w_ibp * 1.01, "full-model CROWN should be <= IBP");
}

/// Core measurement: bound widening at D=64 across multiple weight scales.
///
/// Tests scale 0.1x (contractive), 1.0x (standard Xavier), and 2.0x (expansive)
/// to understand how weight magnitude affects the widening ratio.
#[test]
fn test_bound_widening_d64() {
    eprintln!("\n====== Bound Widening Measurement Suite (D=64) ======\n");

    // Small weights (contractive): should show minimal widening.
    measure_widening(64, 0.1, 1.0);

    // Standard Xavier: typical initialization.
    measure_widening(64, 1.0, 1.0);

    // Large weights (expansive): stress test.
    measure_widening(64, 2.0, 1.0);

    // Standard Xavier with wider input range.
    measure_widening(64, 1.0, 5.0);
}

/// Scale test at D=128 with standard Xavier weights.
#[test]
fn test_bound_widening_d128() {
    eprintln!("\n====== Bound Widening (D=128, Xavier 1.0x) ======\n");
    measure_widening(128, 1.0, 1.0);
}

/// Scale test at D=256 (validates tractability for Kokoro at D=512).
#[test]
fn test_bound_widening_d256() {
    eprintln!("\n====== Bound Widening (D=256, Xavier 1.0x) ======\n");
    measure_widening(256, 1.0, 1.0);
}

// ---------------------------------------------------------------------------
// Deep network tests (5+ layers) where CROWN vs IBP diverge meaningfully
// ---------------------------------------------------------------------------

/// Build a deeper MLP: Linear→ReLU→Linear→ReLU→Linear→ReLU→Linear (4 linear + 3 ReLU).
///
/// Uses random per-element biases centered around `bias_center` with `bias_spread`
/// to create diverse pre-ReLU activation ranges. This ensures some elements pass
/// through ReLU (fully positive bounds), some are killed (fully negative), and
/// some have mixed-sign bounds where CROWN's tighter relaxation differs from IBP.
fn build_deep_model(d: usize, scale: f32, bias_center: f32, bias_spread: f32) -> GraphNetwork {
    let seeds = [42u64, 137, 271, 419];
    let bias_seeds = [1042u64, 1137, 1271, 1419];
    let mut graph = GraphNetwork::new();

    for (i, (&w_seed, &b_seed)) in seeds.iter().zip(bias_seeds.iter()).enumerate() {
        let w = xavier_weights(d, d, w_seed, scale);
        let b = random_bias(d, b_seed, bias_center, bias_spread);
        let linear = LinearLayer::new(w, Some(b)).unwrap_or_else(|_| panic!("l{i}"));

        let lin_name = format!("linear{i}");
        if i == 0 {
            graph.add_node(GraphNode::from_input(&lin_name, Layer::Linear(linear)));
        } else {
            let prev_relu = format!("relu{}", i - 1);
            graph.add_node(GraphNode::new(
                &lin_name,
                Layer::Linear(linear),
                vec![prev_relu],
            ));
        }

        // Add ReLU between linear layers (not after last).
        if i < seeds.len() - 1 {
            let relu_name = format!("relu{i}");
            graph.add_node(GraphNode::new(
                &relu_name,
                Layer::ReLU(ReLULayer),
                vec![lin_name],
            ));
        }
    }

    graph.set_output("linear3");
    graph
}

/// Build a single linear layer with random bias for per-layer verification.
fn build_single_linear_biased(
    d_in: usize,
    d_out: usize,
    w_seed: u64,
    b_seed: u64,
    scale: f32,
    bias_center: f32,
    bias_spread: f32,
) -> GraphNetwork {
    let w = xavier_weights(d_out, d_in, w_seed, scale);
    let b = random_bias(d_out, b_seed, bias_center, bias_spread);
    let linear = LinearLayer::new(w, Some(b)).expect("linear");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear", Layer::Linear(linear)));
    graph.set_output("linear");
    graph
}

/// Per-layer CROWN for the deep (7-layer) model.
fn per_layer_crown_deep(
    d: usize,
    input: &BoundedTensor,
    scale: f32,
    bias_center: f32,
    bias_spread: f32,
) -> BoundedTensor {
    let seeds = [42u64, 137, 271, 419];
    let bias_seeds = [1042u64, 1137, 1271, 1419];

    let mut current = input.clone();
    for (i, (&w_seed, &b_seed)) in seeds.iter().zip(bias_seeds.iter()).enumerate() {
        let graph_lin =
            build_single_linear_biased(d, d, w_seed, b_seed, scale, bias_center, bias_spread);
        current = graph_lin
            .propagate_crown(&current)
            .unwrap_or_else(|_| panic!("linear{i} CROWN"));

        if i < seeds.len() - 1 {
            let graph_relu = build_single_relu();
            current = graph_relu
                .propagate_crown(&current)
                .unwrap_or_else(|_| panic!("relu{i} CROWN"));
        }
    }
    current
}

/// Per-layer alpha-CROWN for the deep (7-layer) model.
///
/// Uses optimized alpha slopes per layer to get tighter bounds than
/// heuristic CROWN, then chains bounds across layers.
fn per_layer_alpha_crown_deep(
    d: usize,
    input: &BoundedTensor,
    scale: f32,
    bias_center: f32,
    bias_spread: f32,
) -> BoundedTensor {
    let seeds = [42u64, 137, 271, 419];
    let bias_seeds = [1042u64, 1137, 1271, 1419];

    let mut current = input.clone();
    for (i, (&w_seed, &b_seed)) in seeds.iter().zip(bias_seeds.iter()).enumerate() {
        let graph_lin =
            build_single_linear_biased(d, d, w_seed, b_seed, scale, bias_center, bias_spread);
        current = graph_lin
            .propagate_alpha_crown(&current)
            .unwrap_or_else(|_| panic!("linear{i} alpha-CROWN"));

        if i < seeds.len() - 1 {
            let graph_relu = build_single_relu();
            current = graph_relu
                .propagate_alpha_crown(&current)
                .unwrap_or_else(|_| panic!("relu{i} alpha-CROWN"));
        }
    }
    current
}

/// Print diagnostic info about intermediate bounds through per-layer propagation.
fn diagnose_per_layer(d: usize, input: &BoundedTensor, scale: f32, bc: f32, bs: f32) {
    let seeds = [42u64, 137, 271, 419];
    let bias_seeds = [1042u64, 1137, 1271, 1419];
    let mut current = input.clone();
    for (i, (&w_seed, &b_seed)) in seeds.iter().zip(bias_seeds.iter()).enumerate() {
        let g = build_single_linear_biased(d, d, w_seed, b_seed, scale, bc, bs);
        current = g.propagate_crown(&current).expect("diag linear");
        let (lo, hi) = current.lower_upper();
        let mixed = lo
            .iter()
            .zip(hi.iter())
            .filter(|(&l, &h)| l < 0.0 && h > 0.0)
            .count();
        let pos = lo
            .iter()
            .zip(hi.iter())
            .filter(|(&l, &h)| l > 0.0 && h > 0.0)
            .count();
        eprintln!(
            "  L{i}: mixed={mixed}/{} pos={pos} w={:.4}",
            lo.len(),
            mean_width(&current)
        );
        if i < seeds.len() - 1 {
            let gr = build_single_relu();
            current = gr.propagate_crown(&current).expect("diag relu");
            eprintln!("  R{i}: w={:.4}", mean_width(&current));
        }
    }
}

/// Measure bound widening for the deep model.
fn measure_widening_deep(d: usize, scale: f32, input_range: f32, bc: f32, bs: f32) {
    let input = positive_input(&[d], input_range);
    let graph = build_deep_model(d, scale, bc, bs);
    let full_crown = graph
        .propagate_crown(&input)
        .expect("full-model CROWN (deep)");
    let perlayer = per_layer_crown_deep(d, &input, scale, bc, bs);
    let ibp = graph.propagate_ibp(&input).expect("full-model IBP (deep)");

    let w_full = mean_width(&full_crown);
    let w_perlayer = mean_width(&perlayer);
    let w_ibp = mean_width(&ibp);
    let widening_ratio = if w_full > 1e-10 {
        w_perlayer / w_full
    } else {
        1.0
    };

    eprintln!(
        "DEEP D={d} s={scale:.1} bc={bc:.1} bs={bs:.1}: CROWN={w_full:.4} \
               PL={w_perlayer:.4} IBP={w_ibp:.4} ratio={widening_ratio:.2}x"
    );

    assert!(
        w_full > 1e-30 && w_full.is_finite(),
        "full CROWN positive finite, got {w_full}"
    );
    assert!(
        w_perlayer > 1e-30 && w_perlayer.is_finite(),
        "perlayer positive finite, got {w_perlayer}"
    );
    assert!(
        w_ibp > 1e-30 && w_ibp.is_finite(),
        "IBP positive finite, got {w_ibp}"
    );
}

/// Deep network (7 layers) at D=64 with diverse activation patterns.
#[test]
fn test_bound_widening_deep_d64() {
    let diag_input = positive_input(&[64], 1.0);
    diagnose_per_layer(64, &diag_input, 1.0, 0.5, 2.0);
    measure_widening_deep(64, 1.0, 1.0, 0.5, 2.0);
    measure_widening_deep(64, 1.0, 1.0, 1.0, 2.0);
    measure_widening_deep(64, 1.0, 1.0, 0.0, 3.0);
    measure_widening_deep(64, 2.0, 1.0, 0.5, 2.0);
    measure_widening_deep(64, 2.0, 1.0, 0.0, 3.0);
}

/// Deep network at D=128.
#[test]
fn test_bound_widening_deep_d128() {
    measure_widening_deep(128, 1.0, 1.0, 0.5, 2.0);
    measure_widening_deep(128, 1.0, 1.0, 0.0, 3.0);
    measure_widening_deep(128, 2.0, 1.0, 0.5, 2.0);
}

/// Count elements where two BoundedTensors differ (tolerance 1e-6).
fn count_differing_elements(a: &BoundedTensor, b: &BoundedTensor) -> usize {
    let (a_lo, a_hi) = a.lower_upper();
    let (b_lo, b_hi) = b.lower_upper();
    let mut diff = 0;
    for idx in 0..a_lo.len() {
        if (f64::from(a_lo[idx]) - f64::from(b_lo[idx])).abs() > 1e-6
            || (f64::from(a_hi[idx]) - f64::from(b_hi[idx])).abs() > 1e-6
        {
            diff += 1;
        }
    }
    diff
}

/// Build a hand-crafted D=4 Linear→ReLU graph for element-wise inspection.
fn build_d4_graph_with_relu() -> GraphNetwork {
    let w = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 0.5, -0.3, 0.2, -0.4, 0.8, 0.1, -0.6, 0.7, -0.2, 0.9, 0.3, -0.1, 0.4, -0.5, 0.6,
        ],
    )
    .expect("w4");
    let b = Array1::from_vec(vec![0.5, -0.3, 0.7, 0.1]);
    let lin = LinearLayer::new(w, Some(b)).expect("lin4");
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input("lin", Layer::Linear(lin)));
    g.add_node(GraphNode::new(
        "relu",
        Layer::ReLU(ReLULayer),
        vec!["lin".to_string()],
    ));
    g.set_output("relu");
    g
}

/// Build a hand-crafted D=4 Linear-only graph for pre-ReLU inspection.
fn build_d4_graph_linear_only() -> GraphNetwork {
    let w = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 0.5, -0.3, 0.2, -0.4, 0.8, 0.1, -0.6, 0.7, -0.2, 0.9, 0.3, -0.1, 0.4, -0.5, 0.6,
        ],
    )
    .expect("w4");
    let b = Array1::from_vec(vec![0.5, -0.3, 0.7, 0.1]);
    let lin = LinearLayer::new(w, Some(b)).expect("lin4");
    let mut g = GraphNetwork::new();
    g.add_node(GraphNode::from_input("lin", Layer::Linear(lin)));
    g.set_output("lin");
    g
}

/// CROWN vs IBP element-wise: 3-layer model with asymmetric inputs.
///
/// Verifies CROWN = IBP bit-for-bit on every element, confirming NY's
/// intersection tightening collapses CROWN to IBP for heuristic alpha.
#[test]
fn test_crown_vs_ibp_elementwise_3layer() {
    let input = {
        let lower = ArrayD::from_elem(IxDyn(&[64]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[64]), 3.0_f32);
        BoundedTensor::new(lower, upper).expect("valid bounds")
    };
    let graph = build_full_model(64, 1.0);
    let crown = graph.propagate_crown(&input).expect("CROWN");
    let ibp = graph.propagate_ibp(&input).expect("IBP");
    let alpha = graph.propagate_alpha_crown(&input).expect("alpha-CROWN");

    let crown_ibp_diff = count_differing_elements(&crown, &ibp);
    let alpha_ibp_diff = count_differing_elements(&alpha, &ibp);
    eprintln!(
        "3L element-wise: CROWN≠IBP on {crown_ibp_diff}/64, alpha≠IBP on {alpha_ibp_diff}/64"
    );
    assert_eq!(crown_ibp_diff, 0, "CROWN should equal IBP element-wise");
    assert_eq!(
        alpha_ibp_diff, 0,
        "alpha-CROWN should equal IBP element-wise"
    );
}

/// Research measurement: alpha-CROWN vs IBP on 7-layer deep ReLU model.
///
/// NOT a closure gate for #3619. This measures deep sequential CROWN quality
/// with IBP intermediates, which is a separate quality gap (RC2 in the
/// measurement gate redesign). Deep sequential CROWN with `fix_interm_bounds=true`
/// degrades to IBP due to compounding relaxation error after 3+ ReLU layers.
///
/// See designs/2026-03-12-issue-3619-measurement-gate-redesign.md (Issue A).
#[test]
fn test_deep_relu_crown_quality_research() {
    let input = positive_input(&[64], 1.0);
    let deep = build_deep_model(64, 1.0, 0.5, 2.0);
    let crown = deep.propagate_crown(&input).expect("CROWN deep");
    let ibp = deep.propagate_ibp(&input).expect("IBP deep");
    let alpha = deep
        .propagate_alpha_crown(&input)
        .expect("alpha-CROWN deep");

    eprintln!(
        "Deep model CROWN={:.4}, IBP={:.4}, alpha={:.4}",
        mean_width(&crown),
        mean_width(&ibp),
        mean_width(&alpha)
    );
    let crown_ibp_diff = count_differing_elements(&crown, &ibp);
    let alpha_ibp_diff = count_differing_elements(&alpha, &ibp);
    eprintln!("Deep element-wise: CROWN≠IBP={crown_ibp_diff}, alpha≠IBP={alpha_ibp_diff}");

    // Per-layer alpha-CROWN widening.
    let perlayer_alpha = per_layer_alpha_crown_deep(64, &input, 1.0, 0.5, 2.0);
    let w_full = mean_width(&alpha);
    let w_pl = mean_width(&perlayer_alpha);
    let widening = if w_full > 1e-10 { w_pl / w_full } else { 1.0 };
    eprintln!("Per-layer alpha-CROWN widening: {widening:.4}x");
    assert!(widening.is_finite(), "widening ratio should be finite");
}

/// D=4 element-wise: inspect pre-ReLU and post-ReLU bounds with all 3 methods.
///
/// Uses hand-crafted weights to produce mixed-sign pre-ReLU intervals,
/// confirming CROWN should improve over IBP but doesn't due to NY's
/// intersection tightening.
#[test]
fn test_d4_prerelu_inspection() {
    let input = {
        let lower = ArrayD::from_elem(IxDyn(&[4]), -2.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[4]), 2.0_f32);
        BoundedTensor::new(lower, upper).expect("valid bounds")
    };

    // Pre-ReLU bounds: confirm mixed-sign intervals exist.
    let g_lin = build_d4_graph_linear_only();
    let pre_relu = g_lin.propagate_ibp(&input).expect("pre-ReLU IBP");
    let (pr_lo, pr_hi) = pre_relu.lower_upper();
    let mixed_count = (0..4).filter(|&i| pr_lo[i] < 0.0 && pr_hi[i] > 0.0).count();
    eprintln!("Pre-ReLU: {mixed_count}/4 elements have mixed-sign bounds");
    assert!(
        mixed_count >= 2,
        "need ≥2 mixed-sign elements for meaningful test"
    );

    // Post-ReLU: CROWN vs IBP vs alpha-CROWN.
    let g = build_d4_graph_with_relu();
    let crown = g.propagate_crown(&input).expect("CROWN");
    let ibp = g.propagate_ibp(&input).expect("IBP");
    let alpha = g.propagate_alpha_crown(&input).expect("alpha-CROWN");

    let diff = count_differing_elements(&crown, &ibp);
    eprintln!("Post-ReLU D=4: CROWN≠IBP on {diff}/4 elements");
    // Primary finding: all methods identical.
    assert_eq!(
        diff, 0,
        "CROWN = IBP at D=4 (NY intersection tightening)"
    );
    assert_eq!(
        count_differing_elements(&alpha, &ibp),
        0,
        "alpha = IBP at D=4"
    );
}

// ---------------------------------------------------------------------------
// #3619 closure gate: Sigmoid DAG alpha-CROWN tightening
// ---------------------------------------------------------------------------

/// Build a 2-layer Sigmoid graph: Linear(2→3) → Sigmoid → Linear(3→2).
///
/// Uses the same weights as the proven gamma-propagate regression test
/// (`alpha_crown_routing.rs`). This architecture exercises the Packet A/B
/// DAG Sigmoid/Tanh alpha-CROWN work surface.
fn build_sigmoid_model_3619() -> GraphNetwork {
    let w1 = arr2(&[[1.2_f32, -0.5], [-0.8, 1.3], [0.4, 0.7]]);
    let b1 = arr1(&[0.15_f32, -0.1, 0.05]);
    let w2 = arr2(&[[0.9_f32, -0.6, 0.4], [-0.3, 1.1, -0.5]]);
    let b2 = arr1(&[0.0_f32, 0.1]);

    let linear1 = LinearLayer::new(w1, Some(b1)).expect("linear1");
    let sigmoid = SigmoidLayer;
    let linear2 = LinearLayer::new(w2, Some(b2)).expect("linear2");

    let mut graph = GraphNetwork::new();
    graph.add_node(GraphNode::from_input("linear1", Layer::Linear(linear1)));
    graph.add_node(GraphNode::new(
        "sigmoid1",
        Layer::Sigmoid(sigmoid),
        vec!["linear1".to_string()],
    ));
    graph.add_node(GraphNode::new(
        "linear2",
        Layer::Linear(linear2),
        vec!["sigmoid1".to_string()],
    ));
    graph.set_output("linear2");
    graph
}

/// #3619 closure gate (SPSA): alpha-CROWN must be strictly tighter than IBP
/// on a Sigmoid DAG model.
///
/// This test validates that the Packet A/B DAG alpha-CROWN work (monotone
/// Sigmoid/Tanh registration + SPSA perturbation) produces measurably tighter
/// bounds than IBP. Fixed-slope CROWN also produces tighter bounds than IBP
/// on this model architecture.
///
/// Note: SPSA is stochastic. The alpha-vs-IBP margin is large (~17%) and
/// robust across seeds. Alpha-vs-CROWN is NOT asserted because both DAG
/// CROWN and SPSA exhibit run-to-run variance (DAG scheduling + SPSA
/// perturbation randomness).
#[test]
fn test_alpha_crown_vs_ibp_sigmoid_3619() {
    let input = {
        let lower = ArrayD::from_elem(IxDyn(&[2]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0_f32);
        BoundedTensor::new(lower, upper).expect("valid bounds")
    };

    let graph = build_sigmoid_model_3619();

    let ibp = graph.propagate_ibp(&input).expect("IBP");
    let crown = graph.propagate_crown(&input).expect("CROWN");
    let config = AlphaCrownConfig {
        iterations: 50,
        gradient_method: GradientMethod::Spsa,
        spsa_samples: 4,
        fix_interm_bounds: false,
        ..AlphaCrownConfig::default()
    };
    let alpha = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .expect("alpha-CROWN SPSA");

    let ibp_width = mean_width(&ibp);
    let crown_width = mean_width(&crown);
    let alpha_width = mean_width(&alpha);

    eprintln!(
        "#3619 SPSA gate: IBP={ibp_width:.6}, CROWN={crown_width:.6}, alpha={alpha_width:.6}"
    );
    eprintln!(
        "  CROWN vs IBP: {:.2}%, alpha vs CROWN: {:.2}%, alpha vs IBP: {:.2}%",
        (1.0 - crown_width / ibp_width) * 100.0,
        (1.0 - alpha_width / crown_width) * 100.0,
        (1.0 - alpha_width / ibp_width) * 100.0,
    );

    // Soundness: alpha-CROWN no wider than IBP.
    assert!(
        alpha_width <= ibp_width + 1e-4,
        "alpha-CROWN wider than IBP — unsound: alpha={alpha_width}, ibp={ibp_width}"
    );

    // Fixed-slope CROWN is strictly tighter than IBP (robust, deterministic
    // within float tolerance despite DAG scheduling variance).
    assert!(
        crown_width < ibp_width * 0.90,
        "fixed-slope CROWN not >10% tighter than IBP: crown={crown_width}, ibp={ibp_width}"
    );

    // Alpha-CROWN (SPSA) is strictly tighter than IBP. The ~17% margin is
    // robust across SPSA seeds.
    assert!(
        alpha_width < ibp_width * 0.90,
        "alpha-CROWN not >10% tighter than IBP — #3619 not fixed: \
         alpha={alpha_width}, ibp={ibp_width}"
    );
}

/// #3619 soundness check (AnalyticChain): default config doesn't widen
/// bounds beyond IBP on a Sigmoid DAG model.
///
/// AnalyticChain only computes gradients for ReLU alphas; Sigmoid/Tanh
/// tangent-point alphas receive zero gradients. The DAG path compensates
/// with a 1-sample SPSA supplement, but for crossing-zero symmetric inputs
/// the tangent-point constraint (.min(d_lower) / .max(d_upper)) prevents
/// improvement over precomputed table defaults. AnalyticChain therefore
/// does NOT reliably improve over fixed-slope CROWN for pure Sigmoid models.
///
/// This test asserts soundness (no wider than IBP) and that CROWN itself
/// is meaningfully tighter than IBP. Tightness of AnalyticChain over CROWN
/// is tracked by #3630 (AnalyticChain gradient gap, P3).
#[test]
fn test_alpha_crown_vs_ibp_sigmoid_analytic_chain_3619() {
    let input = {
        let lower = ArrayD::from_elem(IxDyn(&[2]), -1.0_f32);
        let upper = ArrayD::from_elem(IxDyn(&[2]), 1.0_f32);
        BoundedTensor::new(lower, upper).expect("valid bounds")
    };

    let graph = build_sigmoid_model_3619();

    let ibp = graph.propagate_ibp(&input).expect("IBP");
    let crown = graph.propagate_crown(&input).expect("CROWN");
    let config = AlphaCrownConfig {
        iterations: 50,
        gradient_method: GradientMethod::AnalyticChain,
        fix_interm_bounds: false,
        ..AlphaCrownConfig::default()
    };
    let alpha = graph
        .propagate_alpha_crown_with_config(&input, &config)
        .expect("alpha-CROWN AnalyticChain");

    let ibp_width = mean_width(&ibp);
    let crown_width = mean_width(&crown);
    let alpha_width = mean_width(&alpha);

    eprintln!(
        "#3619 AnalyticChain gate: IBP={ibp_width:.6}, CROWN={crown_width:.6}, alpha={alpha_width:.6}"
    );
    eprintln!(
        "  CROWN vs IBP: {:.2}%, alpha vs CROWN: {:.2}%, alpha vs IBP: {:.2}%",
        (1.0 - crown_width / ibp_width) * 100.0,
        (1.0 - alpha_width / crown_width) * 100.0,
        (1.0 - alpha_width / ibp_width) * 100.0,
    );

    // Soundness: alpha-CROWN no wider than IBP.
    assert!(
        alpha_width <= ibp_width + 1e-4,
        "alpha-CROWN wider than IBP — unsound: alpha={alpha_width}, ibp={ibp_width}"
    );

    // Fixed-slope CROWN is >10% tighter than IBP (robust across DAG
    // scheduling variance).
    assert!(
        crown_width < ibp_width * 0.90,
        "fixed-slope CROWN not >10% tighter than IBP: crown={crown_width}, ibp={ibp_width}"
    );

    // Alpha-CROWN with AnalyticChain is also >10% tighter than IBP.
    assert!(
        alpha_width < ibp_width * 0.90,
        "alpha-CROWN AnalyticChain not >10% tighter than IBP: alpha={alpha_width}, ibp={ibp_width}"
    );
}
