// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! LSTM numeric correctness tests for Silero VAD.
//!
//! Extracted from `silero_vad_tests.rs` to stay under the 500-line limit.
//! Tests LSTM cell decomposition on Metal dispatch with hand-computed expected
//! values covering the full gate equation:
//!   gates = x @ W_ih^T + h @ W_hh^T + bias
//!   i = sigmoid(gates[0:H]),  f = sigmoid(gates[H:2H])
//!   g = tanh(gates[2H:3H]),   o = sigmoid(gates[3H:4H])
//!   c_new = f * c_old + i * g
//!   h_new = o * tanh(c_new)

use std::collections::HashMap;

use nn_dsl::lstm_decomposed::build_lstm_cell_decomposed_dual;

use super::super::*;

/// Configurable LSTM test fixture for Metal dispatch.
struct LstmTestFixture {
    cache: PipelineCache,
    def: TensorKernelDef,
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias: Vec<f32>,
    h0: Vec<f32>,
    c0: Vec<f32>,
}

const LSTM_TEST_DIM: usize = 4;

/// Build a [4*H, H] weight matrix with `scale * identity` in each gate block.
fn scaled_identity_weights(h: usize, scale: f32) -> Vec<f32> {
    let mut w = vec![0.0f32; 4 * h * h];
    for gate in 0..4 {
        for j in 0..h {
            w[gate * h * h + j * h + j] = scale;
        }
    }
    w
}

impl LstmTestFixture {
    /// Default fixture: identity W_ih, zero W_hh, zero bias, zero state.
    fn new() -> Option<Self> {
        Self::with_weights_and_state(
            scaled_identity_weights(LSTM_TEST_DIM, 1.0),
            vec![0.0f32; 4 * LSTM_TEST_DIM * LSTM_TEST_DIM],
            vec![0.0f32; 4 * LSTM_TEST_DIM],
            vec![0.0f32; LSTM_TEST_DIM],
            vec![0.0f32; LSTM_TEST_DIM],
        )
    }

    /// Fixture with configurable weights and initial state.
    fn with_weights_and_state(
        weight_ih: Vec<f32>,
        weight_hh: Vec<f32>,
        bias: Vec<f32>,
        h0: Vec<f32>,
        c0: Vec<f32>,
    ) -> Option<Self> {
        let backend = crate::metal_backend::MetalBackend::init().ok()?;
        let cache = PipelineCache::new(backend.context().clone());
        let def = build_lstm_cell_decomposed_dual(LSTM_TEST_DIM, LSTM_TEST_DIM, 1, true)
            .expect("valid dims");
        Some(Self {
            cache,
            def,
            weight_ih,
            weight_hh,
            bias,
            h0,
            c0,
        })
    }

    fn run(&self, input: &[f32]) -> Vec<f32> {
        let mut inputs = HashMap::new();
        inputs.insert(nn_dsl::input_names::DATA, input);
        inputs.insert(nn_dsl::input_names::HIDDEN_STATE, self.h0.as_slice());
        inputs.insert(nn_dsl::input_names::CELL_STATE, self.c0.as_slice());
        inputs.insert(nn_dsl::input_names::WEIGHT_IH, self.weight_ih.as_slice());
        inputs.insert(nn_dsl::input_names::WEIGHT_HH, self.weight_hh.as_slice());
        inputs.insert(nn_dsl::input_names::BIAS, self.bias.as_slice());
        execute_tensor_dispatch(&self.cache, &self.def, ScalarType::F32, &inputs)
            .expect("LSTM dispatch")
    }

    /// Run and return separate (h_new, c_new) vectors.
    fn run_split(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let out = self.run(input);
        let (h, c) = out.split_at(LSTM_TEST_DIM);
        (h.to_vec(), c.to_vec())
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn assert_close(actual: f32, expected: f32, tol: f32, label: &str) {
    assert!(
        (actual - expected).abs() <= tol,
        "{label}: actual={actual}, expected={expected}, diff={}",
        (actual - expected).abs()
    );
}

/// Zero input, zero state, identity W_ih, zero W_hh → c_new=0, h_new=0.
#[test]
fn test_lstm_numeric_zero_input() {
    let fix = match LstmTestFixture::new() {
        Some(f) => f,
        None => return,
    };
    let out = fix.run(&[0.0f32; LSTM_TEST_DIM]);
    assert_eq!(out.len(), 2 * LSTM_TEST_DIM);

    let (h_new, c_new) = out.split_at(LSTM_TEST_DIM);
    for j in 0..LSTM_TEST_DIM {
        assert_close(c_new[j], 0.0, 1e-5, &format!("c_new[{j}]"));
        assert_close(h_new[j], 0.0, 1e-5, &format!("h_new[{j}]"));
    }
}

/// Input=[1,1,1,1], zero state → gates from sigmoid(1)/tanh(1).
///
/// i=σ(1), f=σ(1), g=tanh(1), o=σ(1)
/// c_new = f*0 + i*g = σ(1)*tanh(1) ≈ 0.557
/// h_new = o*tanh(c_new) ≈ 0.369
#[test]
fn test_lstm_numeric_ones_input() {
    let fix = match LstmTestFixture::new() {
        Some(f) => f,
        None => return,
    };
    let (h_new, c_new) = fix.run_split(&[1.0f32; LSTM_TEST_DIM]);

    let sig1 = sigmoid(1.0);
    let tanh1 = 1.0_f32.tanh();
    let expected_c = sig1 * tanh1;
    let expected_h = sig1 * expected_c.tanh();

    for j in 0..LSTM_TEST_DIM {
        assert_close(c_new[j], expected_c, 0.01, &format!("c[{j}]"));
        assert_close(h_new[j], expected_h, 0.01, &format!("h[{j}]"));
    }
}

/// AC1 (#1080): Non-zero W_hh exercises hidden-to-hidden recurrence.
///
/// W_ih = identity, W_hh = 0.5*identity, bias = 0
/// h0 = [1, -1, 0.5, 0], c0 = [0, 0, 0, 0], input = [0, 0, 0, 0]
///
/// Gate pre-activations = x@W_ih^T + h@W_hh^T = 0 + [0.5, -0.5, 0.25, 0]
/// (same pre-activation for all 4 gates since W_hh has the same block per gate)
#[test]
fn test_lstm_numeric_nonzero_whh() {
    let fix = match LstmTestFixture::with_weights_and_state(
        scaled_identity_weights(LSTM_TEST_DIM, 1.0),
        scaled_identity_weights(LSTM_TEST_DIM, 0.5),
        vec![0.0f32; 4 * LSTM_TEST_DIM],
        vec![1.0, -1.0, 0.5, 0.0],
        vec![0.0f32; LSTM_TEST_DIM],
    ) {
        Some(f) => f,
        None => return,
    };

    let (h_new, c_new) = fix.run_split(&[0.0f32; LSTM_TEST_DIM]);

    let pre = [0.5, -0.5, 0.25, 0.0];
    for j in 0..LSTM_TEST_DIM {
        let i_gate = sigmoid(pre[j]);
        let f_gate = sigmoid(pre[j]);
        let g_gate = pre[j].tanh();
        let o_gate = sigmoid(pre[j]);

        let exp_c = f_gate * 0.0 + i_gate * g_gate;
        let exp_h = o_gate * exp_c.tanh();

        assert_close(c_new[j], exp_c, 1e-4, &format!("c_new[{j}]"));
        assert_close(h_new[j], exp_h, 1e-4, &format!("h_new[{j}]"));
    }
}

/// AC2 (#1080): Non-zero c_old exercises the forget gate.
///
/// Identity W_ih, zero W_hh, zero bias, zero input, zero h0.
/// c0 = [2.0, -1.0, 0.5, 0.0]
///
/// All gate pre-activations = 0 (x=0, h=0, bias=0)
/// i = f = o = σ(0) = 0.5,  g = tanh(0) = 0
/// c_new = 0.5 * c0 + 0.5 * 0 = 0.5 * c0
/// h_new = 0.5 * tanh(c_new)
#[test]
fn test_lstm_numeric_nonzero_c_old() {
    let c0 = vec![2.0, -1.0, 0.5, 0.0];
    let fix = match LstmTestFixture::with_weights_and_state(
        scaled_identity_weights(LSTM_TEST_DIM, 1.0),
        vec![0.0f32; 4 * LSTM_TEST_DIM * LSTM_TEST_DIM],
        vec![0.0f32; 4 * LSTM_TEST_DIM],
        vec![0.0f32; LSTM_TEST_DIM],
        c0.clone(),
    ) {
        Some(f) => f,
        None => return,
    };

    let (h_new, c_new) = fix.run_split(&[0.0f32; LSTM_TEST_DIM]);

    for j in 0..LSTM_TEST_DIM {
        let exp_c = 0.5 * c0[j]; // f=0.5, i*g=0
        let exp_h = 0.5 * exp_c.tanh(); // o=0.5

        assert_close(c_new[j], exp_c, 1e-4, &format!("c_new[{j}]"));
        assert_close(h_new[j], exp_h, 1e-4, &format!("h_new[{j}]"));
    }
}

/// AC3 (#1080): Two-step state propagation — h_new/c_new from step 1 feed step 2.
///
/// Step 1: input=[1,1,1,1], h0=0, c0=0 (same as ones_input test)
/// Step 2: input=[0,0,0,0], h0=h_new_1, c0=c_new_1, W_hh=0.5*I
///
/// Step 2 exercises W_hh with realistic h_new values from step 1,
/// plus forget gate with realistic c_new from step 1.
#[test]
fn test_lstm_numeric_two_step_propagation() {
    // Step 1: identity W_ih, zero W_hh, input=1
    let fix1 = match LstmTestFixture::new() {
        Some(f) => f,
        None => return,
    };
    let (h1, c1) = fix1.run_split(&[1.0f32; LSTM_TEST_DIM]);

    // Verify step 1 matches expected (same as ones_input test)
    let sig1 = sigmoid(1.0);
    let exp_c1 = sig1 * 1.0_f32.tanh();
    let exp_h1 = sig1 * exp_c1.tanh();
    for j in 0..LSTM_TEST_DIM {
        assert_close(c1[j], exp_c1, 0.01, &format!("step1 c[{j}]"));
        assert_close(h1[j], exp_h1, 0.01, &format!("step1 h[{j}]"));
    }

    // Step 2: feed h1/c1 as initial state, W_hh = 0.5*I, input = 0
    let fix2 = match LstmTestFixture::with_weights_and_state(
        scaled_identity_weights(LSTM_TEST_DIM, 1.0),
        scaled_identity_weights(LSTM_TEST_DIM, 0.5),
        vec![0.0f32; 4 * LSTM_TEST_DIM],
        h1,
        c1,
    ) {
        Some(f) => f,
        None => return,
    };
    let (h2, c2) = fix2.run_split(&[0.0f32; LSTM_TEST_DIM]);

    // Step 2: gate pre = x@W_ih^T + h1@W_hh^T = 0 + 0.5*h1
    let pre = 0.5 * exp_h1;
    let i2 = sigmoid(pre);
    let f2 = sigmoid(pre);
    let g2 = pre.tanh();
    let o2 = sigmoid(pre);

    let exp_c2 = f2 * exp_c1 + i2 * g2;
    let exp_h2 = o2 * exp_c2.tanh();

    for j in 0..LSTM_TEST_DIM {
        assert_close(c2[j], exp_c2, 0.01, &format!("step2 c[{j}]"));
        assert_close(h2[j], exp_h2, 0.01, &format!("step2 h[{j}]"));
    }
}

/// Combined W_hh + non-zero state + non-zero input — full gate equation.
///
/// W_ih = identity, W_hh = 0.3*identity, bias = 0
/// h0 = [0.5, -0.5, 0, 0], c0 = [1.0, -0.5, 0, 0], input = [0.2, -0.3, 0.1, 0]
///
/// gate_pre[j] = input[j]*1.0 + h0[j]*0.3
/// All 4 gates share the same pre-activation per element.
#[test]
fn test_lstm_numeric_full_gate_equation() {
    let h0 = vec![0.5, -0.5, 0.0, 0.0];
    let c0 = vec![1.0, -0.5, 0.0, 0.0];
    let input = [0.2_f32, -0.3, 0.1, 0.0];

    let fix = match LstmTestFixture::with_weights_and_state(
        scaled_identity_weights(LSTM_TEST_DIM, 1.0),
        scaled_identity_weights(LSTM_TEST_DIM, 0.3),
        vec![0.0f32; 4 * LSTM_TEST_DIM],
        h0.clone(),
        c0.clone(),
    ) {
        Some(f) => f,
        None => return,
    };
    let (h_new, c_new) = fix.run_split(&input);

    for j in 0..LSTM_TEST_DIM {
        let pre = input[j] * 1.0 + h0[j] * 0.3;
        let i_gate = sigmoid(pre);
        let f_gate = sigmoid(pre);
        let g_gate = pre.tanh();
        let o_gate = sigmoid(pre);

        let exp_c = f_gate * c0[j] + i_gate * g_gate;
        let exp_h = o_gate * exp_c.tanh();

        assert_close(c_new[j], exp_c, 1e-3, &format!("c_new[{j}]"));
        assert_close(h_new[j], exp_h, 1e-3, &format!("h_new[{j}]"));
    }
}
