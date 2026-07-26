// Copyright 2026 Andrew Yates
// Author: Andrew Yates
// SPDX-License-Identifier: Apache-2.0

//! Sound global Lipschitz certification via `ny_api::lipschitz`.
//!
//! Certifies an exact-rational upper bound on a sequential network's global
//! ℓ₂→ℓ₂ Lipschitz constant: `‖f(a) − f(b)‖₂ ≤ L · ‖a − b‖₂` for all inputs.
//! The bound is a product of per-layer operator-norm bounds computed in exact
//! rationals ([`Rat`]), so it carries no floating-point round-off.
//!
//! # This is the *sound* Lipschitz path
//!
//! ny exposes two Lipschitz surfaces and they are not interchangeable:
//!
//! - [`ny_api::probabilistic::estimate_lipschitz_from_network`] — an *optimistic
//!   estimate* obtained by sampling. It can under-report and must never back a
//!   safety claim.
//! - [`ny_api::lipschitz::certify_upper_bound`] (this module) — a **certified
//!   upper bound**. It fails closed: any layer outside the certified fragment
//!   (Linear, Conv, ReLU, Reshape, Transpose) makes it return an error rather
//!   than a bound.
//!
//! NN previously consumed neither. Use [`certify_lipschitz`] wherever a Lipschitz
//! constant feeds a robustness radius, a perturbation budget, or a composition
//! bound — anywhere an under-estimate would be unsound.
//!
//! ```rust,no_run
//! use nn_verify::ny_lipschitz::certify_lipschitz;
//! # fn demo(net: &ny_propagate::Network) -> ny_core::Result<()> {
//! let cert = certify_lipschitz(net)?;
//! // `cert.bound` is an exact rational: Lip(f) <= bound, no round-off.
//! # Ok(())
//! # }
//! ```

pub use ny_api::lipschitz::{LayerLipschitzBound, NormBoundKind, SoundLipschitz};

/// Certify an exact-rational global ℓ₂ Lipschitz upper bound for `net`.
///
/// Thin pass-through to [`ny_api::lipschitz::certify_upper_bound`]. The result's
/// `bound` satisfies `Lip(net) ≤ bound` exactly; `per_layer` records the
/// contributing operator-norm bound for each layer, in network order.
///
/// # Errors
///
/// Fails closed rather than returning a loose or unsound number: any layer
/// outside the certified fragment (Linear, Conv, ReLU, Reshape, Transpose), a
/// non-finite weight, or inconsistent conv metadata yields a [`ny_core::NyError`].
pub fn certify_lipschitz(net: &ny_propagate::Network) -> ny_core::Result<SoundLipschitz> {
    ny_api::lipschitz::certify_upper_bound(net)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ny_propagate::layers::{LinearLayer, ReLULayer};
    use ny_propagate::{Layer, Network};

    /// `Linear(2->4) -> ReLU -> Linear(4->1)`.
    ///
    /// `W1ᵀW1 = [[5,1],[1,5]]` has eigenvalues 6 and 4, so `‖W1‖₂ = √6`.
    /// `W2 = [1,1,1,1]` gives `‖W2‖₂ = 2`. ReLU is 1-Lipschitz, so the layerwise
    /// product bound is `2√6 ≈ 4.89898`.
    fn fc_relu_net() -> Network {
        let mut n = Network::new();
        let w1 = arr2(&[[2.0, 0.0], [0.0, -2.0], [1.0, 1.0], [0.0, 0.0]]);
        let b1 = arr1(&[3.0, -3.0, 0.0, 5.0]);
        n.add_layer(Layer::Linear(
            LinearLayer::new(w1, Some(b1)).expect("valid linear layer 1"),
        ));
        n.add_layer(Layer::ReLU(ReLULayer::new()));
        let w2 = arr2(&[[1.0, 1.0, 1.0, 1.0]]);
        n.add_layer(Layer::Linear(
            LinearLayer::new(w2, None).expect("valid linear layer 2"),
        ));
        n
    }

    #[test]
    fn certifies_the_layerwise_product_bound() {
        let cert = certify_lipschitz(&fc_relu_net()).expect("net is in the certified fragment");
        let bound = cert.bound_approx();

        // Sound: never below the true Lipschitz constant. A lower bound is
        // witnessed by f(0,0)=8 and f(1,0)=11, giving |Δf|/‖Δx‖ = 3.
        assert!(bound >= 3.0, "certified bound {bound} is below a witnessed slope");
        // Tight: the layerwise product 2√6, not a wildly loose over-approximation.
        assert!(
            (bound - 2.0 * 6f64.sqrt()).abs() < 1e-6,
            "expected the 2*sqrt(6) product bound, got {bound}",
        );
    }

    /// The certifier must fail closed outside its fragment rather than guess.
    #[test]
    fn fails_closed_on_an_unsupported_layer() {
        let mut n = Network::new();
        n.add_layer(Layer::Sigmoid(ny_propagate::layers::SigmoidLayer::new()));
        assert!(
            certify_lipschitz(&n).is_err(),
            "Sigmoid is outside the certified fragment; must error, not return a bound",
        );
    }
}
