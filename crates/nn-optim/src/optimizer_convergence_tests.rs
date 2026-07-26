// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Convergence and stress tests for all optimizer implementations.
//!
//! Covers:
//! 1. SGD convergence on f(x)=x^2
//! 2. SGD with momentum acceleration
//! 3. AdamW convergence on quadratic loss
//! 4. AdamW bias correction in early steps
//! 5. AdaFactor convergence
//! 6. Learning rate schedules (Warmup, Cosine)
//! 7. Gradient clipping (norm and value)
//! 8. GradScaler round-trip
//! 9. LoRA parameter count
//! 10. step_with_schedule integration

use std::sync::Arc;

use nn_autodiff::{backward, GradStore, TrackedTensor, Var};
use nn_core::dyn_tensor::DynTensor;
use nn_core::layers::Linear;
use nn_core::test_utils::cpu;
use nn_core::DType;

use crate::adafactor::{AdaFactor, AdaFactorConfig};
use crate::adam::{AdamConfig, AdamW};
use crate::grad_clip::{clip_grad_norm, clip_grad_value};
use crate::grad_scaler::{GradScaler, GradScalerConfig};
use crate::lora::LoraLinear;
use crate::lr_schedule::{step_with_schedule, CosineSchedule, LrSchedule, WarmupSchedule};
use crate::optimizer::Optimizer;
use crate::sgd::{Sgd, SgdConfig};

/// Create a Var from a flat vec with given shape.
fn make_var(vals: &[f32], shape: &[usize]) -> Var {
    Var::new(DynTensor::from_vec(vals.to_vec(), shape, &cpu()).unwrap())
}

/// Compute loss = sum(x^2) and return (loss_tracked, loss_scalar, grads).
fn quadratic_loss(var: &Var) -> (Arc<TrackedTensor>, f64, GradStore) {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    let mut loss = sq;
    for d in 0..var.data().unwrap().dims().len() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    let loss_val: f64 = loss
        .tensor()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v))
        .sum();
    let grads = backward(&loss).unwrap();
    (loss, loss_val, grads)
}

/// Scalar loss value for a Var (sum of squares).
fn loss_of(var: &Var) -> f64 {
    var.data()
        .unwrap()
        .to_flat_vec::<f32>()
        .unwrap()
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum()
}

/// Build a GradStore by running backward on sum(x^2), then overwrite
/// the gradient with `grad_vals`.
fn manual_grads(var: &Var, grad_vals: &[f32]) -> GradStore {
    let t = Arc::new(TrackedTensor::from_var(var).unwrap());
    let sq = t.sqr().unwrap();
    let mut loss = sq;
    for d in 0..var.data().unwrap().dims().len() {
        loss = loss.sum_keepdim(d).unwrap();
    }
    let mut grads = backward(&loss).unwrap();
    let grad_tensor =
        DynTensor::from_vec(grad_vals.to_vec(), var.data().unwrap().dims(), &cpu()).unwrap();
    for (_, g) in grads.var_grads_mut() {
        *g = grad_tensor.clone();
    }
    grads
}

// ============================================================================
// 1. SGD convergence on f(x) = x^2
// ============================================================================

#[test]
fn test_sgd_converges_on_quadratic() {
    // f(x) = x^2, grad = 2x. With lr=0.1, SGD should converge toward 0.
    let var = make_var(&[5.0], &[1]);
    let config = SgdConfig {
        lr: 0.1,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let mut opt = Sgd::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var);
    for _ in 0..50 {
        let (_, _, grads) = quadratic_loss(&var);
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);

    // After 50 steps of lr=0.1 on f(x)=x^2, loss should be near zero.
    // Each step: x_{n+1} = x_n - 0.1 * 2 * x_n = 0.8 * x_n
    // After 50 steps: x = 5.0 * 0.8^50 ~ 7.2e-5, loss ~ 5.2e-9
    assert!(
        final_loss < 1e-5,
        "SGD should converge to near-zero loss, got {final_loss}"
    );
    assert!(
        final_loss < initial_loss * 1e-6,
        "SGD final loss ({final_loss}) should be many orders smaller than initial ({initial_loss})"
    );
}

#[test]
fn test_sgd_converges_multidim() {
    // f(x) = sum(x_i^2) for a 4-element vector.
    let var = make_var(&[3.0, -4.0, 1.0, -2.0], &[4]);
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let mut opt = Sgd::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var); // 9 + 16 + 1 + 4 = 30
    for _ in 0..80 {
        let (_, _, grads) = quadratic_loss(&var);
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);

    assert!(
        final_loss < 1e-4,
        "SGD on multi-dim quadratic should converge, got {final_loss}"
    );
    assert!(initial_loss > 29.0, "initial loss should be ~30");
}

// ============================================================================
// 2. SGD with momentum accelerates convergence
// ============================================================================

#[test]
fn test_sgd_momentum_accelerates_convergence() {
    // Run SGD with and without momentum on the same quadratic.
    // Momentum should converge faster (lower loss after same number of steps).
    let init = vec![10.0, -8.0, 5.0, -3.0];
    let var_no_mom = make_var(&init, &[4]);
    let var_mom = make_var(&init, &[4]);

    let config_no = SgdConfig {
        lr: 0.01,
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let config_mom = SgdConfig {
        lr: 0.01,
        momentum: 0.9,
        weight_decay: 0.0,
    };

    let mut opt_no = Sgd::new(vec![var_no_mom.clone()], config_no).unwrap();
    let mut opt_mom = Sgd::new(vec![var_mom.clone()], config_mom).unwrap();

    for _ in 0..40 {
        let (_, _, g_no) = quadratic_loss(&var_no_mom);
        opt_no.step(&g_no).unwrap();

        let (_, _, g_mom) = quadratic_loss(&var_mom);
        opt_mom.step(&g_mom).unwrap();
    }

    let loss_no = loss_of(&var_no_mom);
    let loss_mom = loss_of(&var_mom);

    // Both should converge, but momentum should be strictly faster.
    assert!(
        loss_mom < loss_no,
        "momentum should converge faster: mom_loss={loss_mom}, no_mom_loss={loss_no}"
    );
    // And both should be less than initial (~198).
    let initial_loss = init.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    assert!(loss_no < initial_loss);
    assert!(loss_mom < initial_loss);
}

// ============================================================================
// 3. AdamW convergence on quadratic loss
// ============================================================================

#[test]
fn test_adamw_converges_on_quadratic() {
    let var = make_var(&[7.0, -5.0, 3.0], &[3]);
    let config = AdamConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut opt = AdamW::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var); // 49 + 25 + 9 = 83
    for _ in 0..100 {
        let (_, _, grads) = quadratic_loss(&var);
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);

    assert!(
        final_loss < 1.0,
        "AdamW should converge to near-zero on quadratic, got {final_loss}"
    );
    assert!(
        final_loss < initial_loss * 0.01,
        "final loss ({final_loss}) should be <1% of initial ({initial_loss})"
    );
}

#[test]
fn test_adamw_converges_with_weight_decay() {
    // Weight decay should not prevent convergence on quadratic (it helps).
    let var = make_var(&[4.0, -3.0], &[2]);
    let config = AdamConfig {
        lr: 0.05,
        weight_decay: 0.01,
        ..AdamConfig::default()
    };
    let mut opt = AdamW::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var); // 16 + 9 = 25
    for _ in 0..80 {
        let (_, _, grads) = quadratic_loss(&var);
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);

    assert!(
        final_loss < 1.0,
        "AdamW+WD should converge, got {final_loss}"
    );
    assert!(initial_loss > 24.0);
}

// ============================================================================
// 4. AdamW bias correction in early steps
// ============================================================================

#[test]
fn test_adamw_bias_correction_early_steps() {
    // Bias correction compensates for zero-initialized moments.
    // Without it, the first step's effective learning rate would be
    // much smaller. We verify that even the first step produces a
    // meaningful update.
    let var = make_var(&[10.0], &[1]);
    let config = AdamConfig {
        lr: 0.5,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut opt = AdamW::new(vec![var.clone()], config).unwrap();

    let before = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let (_, _, grads) = quadratic_loss(&var);
    opt.step(&grads).unwrap();
    let after = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    // The update should be substantial (~lr=0.5) because bias correction
    // divides by (1 - beta^1) which is 0.1 for beta1 and 0.001 for beta2.
    // With bias correction: m_hat = m/(1-0.9) = 10*m, v_hat = v/(1-0.999) = 1000*v
    // The step size in Adam is approximately lr per coordinate for the first step.
    let delta = (after - before).abs();
    assert!(
        delta > 0.1,
        "bias correction should produce meaningful first step, delta={delta}"
    );
    assert!(
        after < before,
        "parameter should decrease toward minimum (x=0), before={before}, after={after}"
    );

    // Verify step count incremented.
    assert_eq!(opt.step_count(), 1);
}

#[test]
fn test_adamw_bias_correction_moment_scaling() {
    // Verify that at step 1, the bias-corrected moments differ significantly
    // from the raw moments. At step 1:
    //   bc1 = 1/(1 - 0.9^1) = 10.0
    //   bc2 = 1/(1 - 0.999^1) = 1000.0
    // So m_hat = 10 * m and v_hat = 1000 * v.
    //
    // We test this indirectly: two steps should produce updates of similar
    // magnitude (not wildly different) because bias correction normalizes them.
    let var = make_var(&[5.0], &[1]);
    let config = AdamConfig {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut opt = AdamW::new(vec![var.clone()], config).unwrap();

    let v0 = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];
    let (_, _, g1) = quadratic_loss(&var);
    opt.step(&g1).unwrap();
    let v1 = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    let (_, _, g2) = quadratic_loss(&var);
    opt.step(&g2).unwrap();
    let v2 = var.data().unwrap().to_flat_vec::<f32>().unwrap()[0];

    let delta1 = (v1 - v0).abs();
    let delta2 = (v2 - v1).abs();

    // With bias correction, both deltas should be in the same order of magnitude.
    // Without bias correction, delta1 would be orders of magnitude smaller.
    assert!(
        delta1 > 0.01,
        "first step should have meaningful magnitude: {delta1}"
    );
    assert!(
        delta2 > 0.01,
        "second step should have meaningful magnitude: {delta2}"
    );
    // The ratio should be reasonable (within 100x).
    let ratio = delta1.max(delta2) / delta1.min(delta2);
    assert!(
        ratio < 100.0,
        "bias correction should keep step sizes comparable, ratio={ratio}"
    );
}

// ============================================================================
// 5. AdaFactor convergence
// ============================================================================

#[test]
fn test_adafactor_converges_on_quadratic() {
    let var = make_var(&[6.0, -4.0, 2.0], &[3]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var); // 36 + 16 + 4 = 56
    for _ in 0..40 {
        let (_, _, grads) = quadratic_loss(&var);
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);

    assert!(
        final_loss < initial_loss * 0.5,
        "AdaFactor should converge: initial={initial_loss}, final={final_loss}"
    );
}

#[test]
fn test_adafactor_2d_factored_converges() {
    // 2D parameter triggers factored second moments.
    let var = make_var(&[3.0, -2.0, 1.0, -4.0, 5.0, -1.0], &[2, 3]);
    let config = AdaFactorConfig {
        lr: 0.1,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();

    let initial_loss = loss_of(&var);
    for _ in 0..30 {
        let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
        let sq = t.sqr().unwrap();
        let loss = sq.sum_keepdim(0).unwrap().sum_keepdim(1).unwrap();
        let grads = backward(&loss).unwrap();
        opt.step(&grads).unwrap();
    }
    let final_loss = loss_of(&var);

    assert!(
        final_loss < initial_loss * 0.5,
        "2D factored AdaFactor should converge: init={initial_loss}, final={final_loss}"
    );
}

// ============================================================================
// 6. Learning rate schedules
// ============================================================================

#[test]
fn test_warmup_schedule_ramps_linearly() {
    let sched = WarmupSchedule::new(0.1, 10).unwrap();

    // Step 0: lr = 0
    assert!((sched.lr_at_step(0) - 0.0).abs() < 1e-15);

    // Step 1: lr = 0.01
    assert!((sched.lr_at_step(1) - 0.01).abs() < 1e-10);

    // Step 5 (halfway): lr = 0.05
    assert!((sched.lr_at_step(5) - 0.05).abs() < 1e-10);

    // Step 10 (end of warmup): lr = base_lr = 0.1
    assert!((sched.lr_at_step(10) - 0.1).abs() < 1e-10);

    // Step 15 (after warmup): lr stays at base_lr
    assert!((sched.lr_at_step(15) - 0.1).abs() < 1e-10);

    // Step 100 (well after warmup): still base_lr
    assert!((sched.lr_at_step(100) - 0.1).abs() < 1e-10);

    // Linearity: lr(k) = base_lr * k / warmup_steps for k < warmup_steps
    for k in 0..10 {
        let expected = 0.1 * (k as f64) / 10.0;
        let actual = sched.lr_at_step(k);
        assert!(
            (actual - expected).abs() < 1e-12,
            "step {k}: expected {expected}, got {actual}"
        );
    }
}

#[test]
fn test_cosine_schedule_decays() {
    let sched = CosineSchedule::new(0.1, 0.001, 10, 110).unwrap();

    // During warmup (step < 10): linear ramp
    assert!((sched.lr_at_step(0) - 0.0).abs() < 1e-15);
    assert!((sched.lr_at_step(5) - 0.05).abs() < 1e-10);

    // At warmup end: lr = base_lr
    let lr_at_warmup = sched.lr_at_step(10);
    assert!(
        (lr_at_warmup - 0.1).abs() < 1e-10,
        "at warmup end, lr should be base_lr=0.1, got {lr_at_warmup}"
    );

    // Mid-schedule: between min and max
    let lr_mid = sched.lr_at_step(60);
    assert!(
        lr_mid > 0.001 && lr_mid < 0.1,
        "mid-schedule lr should be between min and max, got {lr_mid}"
    );

    // Near end: approaches min_lr
    let lr_end = sched.lr_at_step(109);
    assert!(
        lr_end < 0.005,
        "near-end lr should approach min_lr, got {lr_end}"
    );

    // Past total: clamps to min_lr
    let lr_past = sched.lr_at_step(200);
    assert!(
        (lr_past - 0.001).abs() < 1e-15,
        "past total_steps, lr should be min_lr, got {lr_past}"
    );

    // Monotonically decreasing in cosine phase
    let mut prev = sched.lr_at_step(10);
    for step in 11..110 {
        let current = sched.lr_at_step(step);
        assert!(
            current <= prev + 1e-12,
            "cosine should be monotonically decreasing: step {step}, prev={prev}, current={current}"
        );
        prev = current;
    }
}

// ============================================================================
// 7. Gradient clipping
// ============================================================================

#[test]
fn test_clip_grad_norm_bounds_magnitude() {
    let var = make_var(&[3.0, 4.0], &[2]);
    // Gradient = [6.0, 8.0] (from 2*x), L2 norm = 10.0
    let mut grads = manual_grads(&var, &[6.0, 8.0]);

    let original_norm = clip_grad_norm(&mut grads, 5.0).unwrap();

    // Original norm should be 10.0
    assert!(
        (original_norm - 10.0).abs() < 1e-5,
        "original norm should be 10.0, got {original_norm}"
    );

    // After clipping to max_norm=5.0, gradient should be scaled to norm 5.0
    let clipped: Vec<f32> = grads
        .var_grads()
        .next()
        .unwrap()
        .1
        .to_flat_vec::<f32>()
        .unwrap();
    let clipped_norm: f64 = clipped
        .iter()
        .map(|&v| f64::from(v).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        (clipped_norm - 5.0).abs() < 1e-4,
        "clipped norm should be 5.0, got {clipped_norm}"
    );

    // Direction should be preserved: [3, 4] direction
    let ratio = clipped[0] / clipped[1];
    assert!(
        (ratio - 0.75).abs() < 1e-5,
        "direction should be preserved (3:4 ratio), got {ratio}"
    );
}

#[test]
fn test_clip_grad_norm_no_clip_when_within_bound() {
    let var = make_var(&[1.0, 0.0], &[2]);
    // Gradient = [2.0, 0.0], L2 norm = 2.0
    let mut grads = manual_grads(&var, &[2.0, 0.0]);

    let original_norm = clip_grad_norm(&mut grads, 10.0).unwrap();

    assert!(
        (original_norm - 2.0).abs() < 1e-5,
        "original norm should be 2.0, got {original_norm}"
    );

    // Gradient should be unchanged (norm 2.0 < max_norm 10.0)
    let vals: Vec<f32> = grads
        .var_grads()
        .next()
        .unwrap()
        .1
        .to_flat_vec::<f32>()
        .unwrap();
    assert!(
        (vals[0] - 2.0).abs() < 1e-6,
        "gradient should be unchanged, got {vals:?}"
    );
}

#[test]
fn test_clip_grad_value_clamps_elements() {
    let var = make_var(&[1.0, 2.0, 3.0], &[3]);
    // Gradient with large and small values
    let mut grads = manual_grads(&var, &[10.0, -20.0, 0.5]);

    clip_grad_value(&mut grads, 5.0).unwrap();

    let vals: Vec<f32> = grads
        .var_grads()
        .next()
        .unwrap()
        .1
        .to_flat_vec::<f32>()
        .unwrap();

    // 10.0 should be clamped to 5.0
    assert!(
        (vals[0] - 5.0).abs() < 1e-6,
        "10.0 should clamp to 5.0, got {}",
        vals[0]
    );
    // -20.0 should be clamped to -5.0
    assert!(
        (vals[1] - (-5.0)).abs() < 1e-6,
        "-20.0 should clamp to -5.0, got {}",
        vals[1]
    );
    // 0.5 should be unchanged (within [-5, 5])
    assert!(
        (vals[2] - 0.5).abs() < 1e-6,
        "0.5 should be unchanged, got {}",
        vals[2]
    );
}

// ============================================================================
// 8. GradScaler integration
// ============================================================================

#[test]
fn test_grad_scaler_scale_unscale_roundtrip() {
    // Verify that scale_loss + backward + unscale produces gradients
    // with the same direction as direct backward.
    let var = make_var(&[3.0, -2.0], &[2]);
    let scale = 256.0;

    // Direct backward (no scaling)
    let (_, _, direct_grads) = quadratic_loss(&var);
    let direct_grad_vals: Vec<f32> = direct_grads
        .var_grads()
        .next()
        .unwrap()
        .1
        .to_flat_vec::<f32>()
        .unwrap();

    // Scaled backward
    let scaler_config = GradScalerConfig {
        init_scale: scale,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(scaler_config).unwrap();

    let t = Arc::new(TrackedTensor::from_var(&var).unwrap());
    let sq = t.sqr().unwrap();
    let loss = sq.sum_keepdim(0).unwrap();
    let scaled_loss = scaler.scale_loss(&loss).unwrap();
    let mut scaled_grads = backward(&scaled_loss).unwrap();

    // Before unscaling, gradients should be ~scale times larger
    let pre_unscale: Vec<f32> = scaled_grads
        .var_grads()
        .next()
        .unwrap()
        .1
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (&pre, &direct)) in pre_unscale.iter().zip(direct_grad_vals.iter()).enumerate() {
        let expected = direct * scale as f32;
        assert!(
            (pre - expected).abs() < 1e-2,
            "pre-unscale grad[{i}] should be ~{expected}, got {pre}"
        );
    }

    // After unscaling, gradients should match direct
    let ok = scaler.unscale_and_check(&mut scaled_grads).unwrap();
    assert!(ok, "unscale should report finite gradients");
    assert!(!scaler.found_inf(), "should not find inf/NaN");

    let unscaled: Vec<f32> = scaled_grads
        .var_grads()
        .next()
        .unwrap()
        .1
        .to_flat_vec::<f32>()
        .unwrap();
    for (i, (&u, &d)) in unscaled.iter().zip(direct_grad_vals.iter()).enumerate() {
        assert!(
            (u - d).abs() < 1e-3,
            "unscaled grad[{i}] should match direct: unscaled={u}, direct={d}"
        );
    }
}

#[test]
fn test_grad_scaler_update_grows_scale() {
    let config = GradScalerConfig {
        init_scale: 100.0,
        growth_factor: 2.0,
        growth_interval: 3,
        min_scale: 1.0,
        max_scale: 1e6,
        ..Default::default()
    };
    let mut scaler = GradScaler::new(config).unwrap();

    assert!((scaler.scale_factor() - 100.0).abs() < 1e-10);

    // Simulate 3 clean steps: scale should grow
    for _ in 0..3 {
        // Mark as clean (not found_inf)
        scaler.update();
    }
    assert!(
        (scaler.scale_factor() - 200.0).abs() < 1e-10,
        "after growth_interval clean steps, scale should double, got {}",
        scaler.scale_factor()
    );
}

// ============================================================================
// 9. LoRA parameter count
// ============================================================================

#[test]
fn test_lora_has_fewer_trainable_params() {
    // Linear: weight [64, 128] = 8192 parameters
    let weight = DynTensor::zeros(&[64, 128], DType::F32, &cpu()).unwrap();
    let linear = Linear::new(weight, None).unwrap();

    // LoRA with rank 4: A [4, 128] + B [64, 4] = 512 + 256 = 768 params
    let lora = LoraLinear::from_linear(&linear, 4, 4.0).unwrap();
    let trainable_vars = lora.trainable_vars();
    assert_eq!(
        trainable_vars.len(),
        2,
        "LoRA should have 2 trainable vars (A and B)"
    );

    let total_lora_params: usize = trainable_vars
        .iter()
        .map(|v| v.data().unwrap().elem_count())
        .sum();
    let full_params = 64 * 128; // 8192

    assert_eq!(
        total_lora_params, 768,
        "LoRA params should be rank*(in+out) = 4*(128+64) = 768, got {total_lora_params}"
    );
    assert!(
        total_lora_params < full_params,
        "LoRA ({total_lora_params}) should have fewer params than full linear ({full_params})"
    );

    // Verify shapes
    let a_dims = trainable_vars[0].data().unwrap().dims().to_vec();
    let b_dims = trainable_vars[1].data().unwrap().dims().to_vec();
    assert_eq!(
        a_dims,
        vec![4, 128],
        "A shape should be [rank, in_features]"
    );
    assert_eq!(
        b_dims,
        vec![64, 4],
        "B shape should be [out_features, rank]"
    );
}

#[test]
fn test_lora_scaling_factor() {
    let weight = DynTensor::zeros(&[32, 64], DType::F32, &cpu()).unwrap();
    let linear = Linear::new(weight, None).unwrap();

    let lora = LoraLinear::from_linear(&linear, 8, 16.0).unwrap();
    // scaling = alpha / rank = 16.0 / 8 = 2.0
    assert!(
        (lora.scaling() - 2.0).abs() < 1e-10,
        "scaling should be alpha/rank = 2.0, got {}",
        lora.scaling()
    );
}

// ============================================================================
// 10. step_with_schedule integration
// ============================================================================

#[test]
fn test_step_with_schedule_sgd_warmup() {
    let var = make_var(&[8.0, -6.0], &[2]);
    let config = SgdConfig {
        lr: 0.0, // overwritten by schedule
        momentum: 0.0,
        weight_decay: 0.0,
    };
    let mut opt = Sgd::new(vec![var.clone()], config).unwrap();
    let sched = WarmupSchedule::new(0.1, 10).unwrap();

    // During warmup, LR ramps. After warmup, LR is constant at 0.1.
    for step in 0..20 {
        let (_, _, grads) = quadratic_loss(&var);
        step_with_schedule(&mut opt, &grads, &sched, step).unwrap();
    }

    // After warmup (step >= 10), LR should be at base_lr.
    assert!(
        (opt.learning_rate() - 0.1).abs() < 1e-10,
        "after warmup, LR should be 0.1, got {}",
        opt.learning_rate()
    );

    // Parameters should have moved toward zero.
    let vals = var.data().unwrap().to_flat_vec::<f32>().unwrap();
    assert!(vals[0].abs() < 8.0, "param should move toward 0");
    assert!(vals[1].abs() < 6.0, "param should move toward 0");
}

#[test]
fn test_step_with_schedule_adam_cosine() {
    let var = make_var(&[5.0, -3.0, 2.0], &[3]);
    let config = AdamConfig {
        lr: 0.0,
        ..AdamConfig::default()
    };
    let mut opt = AdamW::new(vec![var.clone()], config).unwrap();
    // base_lr=0.1 only reduces the loss to ~20% over 50 cosine-decayed steps
    // (Adam moves ~lr/step and the peak window is short). base_lr=0.2 gives a
    // genuine 10x+ reduction while still decaying to near min_lr by the end.
    let sched = CosineSchedule::new(0.2, 0.001, 5, 50).unwrap();

    let initial_loss = loss_of(&var);
    for step in 0..50 {
        let (_, _, grads) = quadratic_loss(&var);
        step_with_schedule(&mut opt, &grads, &sched, step).unwrap();
    }
    let final_loss = loss_of(&var);

    // Should converge despite cosine decay.
    assert!(
        final_loss < initial_loss * 0.1,
        "AdamW + cosine should converge: init={initial_loss}, final={final_loss}"
    );

    // LR should be near min_lr at the end.
    assert!(
        opt.learning_rate() < 0.005,
        "at end of cosine schedule, lr should be near min_lr, got {}",
        opt.learning_rate()
    );
}

#[test]
fn test_step_with_schedule_adafactor_warmup() {
    let var = make_var(&[4.0, -4.0], &[2]);
    let config = AdaFactorConfig {
        lr: 0.0,
        relative_step: false,
        ..Default::default()
    };
    let mut opt = AdaFactor::new(vec![var.clone()], config).unwrap();
    let sched = WarmupSchedule::new(0.1, 5).unwrap();

    let initial_loss = loss_of(&var);
    for step in 0..20 {
        let (_, _, grads) = quadratic_loss(&var);
        step_with_schedule(&mut opt, &grads, &sched, step).unwrap();
    }
    let final_loss = loss_of(&var);

    assert!(
        final_loss < initial_loss,
        "AdaFactor + warmup should converge: init={initial_loss}, final={final_loss}"
    );
    assert!(
        (opt.learning_rate() - 0.1).abs() < 1e-10,
        "LR should be at base after warmup"
    );
}
