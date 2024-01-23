pub mod budget;
pub mod rdp_alphas_accounting;
pub mod rdpopt;

use crate::dprivacy::rdp_alphas_accounting::*;
use crate::dprivacy::AccountingType::*;
use crate::dprivacy::RdpAlphas::*;
use float_cmp::{ApproxEq, F64Margin};
use rug::{ops::Pow, Float};
use serde::{Deserialize, Serialize};
use serde_aux::prelude::*;
use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

// Structs and Enums

/// Holds an eps-dp, adp or rdp budget or cost. See [Accounting], [AdpAccounting] and
/// [RdpAccounting] for methods to call on this enum.
#[derive(Deserialize, Clone, Debug, PartialEq, PartialOrd, Serialize)]
pub enum AccountingType {
    EpsDp {
        #[serde(deserialize_with = "deserialize_number_from_string")]
        eps: f64,
    },
    EpsDeltaDp {
        #[serde(deserialize_with = "deserialize_number_from_string")]
        eps: f64,
        #[serde(deserialize_with = "deserialize_number_from_string")]
        delta: f64,
    },
    Rdp {
        eps_values: RdpAlphas,
    },
}

// Custom Traits

/// Contains methods which take do any [AccountingType] as input, or take no [AccountingType] as
/// input.
pub trait Accounting: Sized {
    /// assigns the minimum of two AccountingTypes
    fn min_assign(&mut self, rhs: &Self);
    /// assigns the maximum of two AccountingTypes
    fn max_assign(&mut self, rhs: &Self);
    /// checks if given cost would satisfy budged
    fn in_budget(&self, cost: &Self) -> bool;

    /// checks if AccountingType is approximately less or equal to rhs
    /// (< and > are exact comparisons prone to numerical instability)
    fn approx_le(&self, rhs: &Self) -> bool;

    /// Get rdp cost of releasing result of function with sensitivity one and gaussian noise with
    /// standard deviation stddev
    fn rdp_cost_gaussian_noise(stddev: f64, alphas: &RdpAlphas) -> Self;

    /// Get adp cost of releasing result of function with sensitivity one and gaussian noise with
    /// standard deviation stddev, for a certain value of delta
    fn adp_cost_gaussian_noise(stddev: f64, delta: f64) -> Self;

    /// check that the accounting type is valid (non-negative budget for DP and ADP) (at least one non-negative coefficient in RDP)
    fn is_valid(&self) -> bool;

    /// create a new accounting type with zeros everywhere, but of the same type as other
    fn zero_clone(other: &Self) -> Self;

    /// Applies a function for each value (either eps, or eps and delta, each eps_alpha)
    fn apply_func(&mut self, f: &dyn Fn(f64) -> f64);

    /// checks whether the same type of accounting is used
    fn check_same_type(&self, other: &Self) -> bool;

    /// Returns how much budget is still left for the epsilon / delta values as a ratio.
    /// self should be the current budget which is left, while original_budget should be the budget
    /// as it was in the beginning, without any subtractions. Also does
    /// some checks to verify that the passed budgets are plausible.
    fn remaining_ratios(&self, original_budget: &Self) -> AccountingTypeRatio;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AccountingTypeRatio(pub AccountingType);

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AlphaIndex(pub usize);

/// Contains methods which should only be applied to accounting types which are rdp, or to check
/// whether an accounting type is rdp
pub trait RdpAccounting {
    /// Returns whether the accounting type is a1, i.e., rdp with only a single alpha value.
    /// Used in conjunction with [Self::a1_to_eps_dp]
    fn is_a1(&self) -> bool;

    /// If the [AccountingType] is A1 Rdp (alpha_1, eps_1), then it returns eps_dp with parameter
    /// eps_1 (else panics). Note that if only basic sequential composition is used for eps dp, then we can
    /// transform A1 to eps dp this way, and can just use constraints for eps dp. Used in the ilp
    /// formulation, as the formulation for rdp is more complicated, and no advanced composition
    /// is used for eps dp.
    fn a1_to_eps_dp(&self) -> AccountingType;

    /// check whether [AccountingType] is an rdp type
    fn is_rdp(&self) -> bool;

    /// If the [AccountingType] is Rdp, then it returns the contained values as vector, else panics
    fn get_rdp_vec(&self) -> Vec<f64>;

    /// Get an vec of all the AlphaIndices in the [AccountingType]
    fn get_alpha_indices(&self) -> Vec<AlphaIndex> {
        self.get_rdp_vec()
            .iter()
            .enumerate()
            .map(|(i, _)| AlphaIndex(i))
            .collect()
    }

    /// Get the cost or budget for a certain alpha
    fn get_value_for_alpha(&self, index: AlphaIndex) -> f64;
}

/// Contains methods which should only be applied to accounting types which are adp, or to check
/// whether an accounting type is adp
pub trait AdpAccounting {
    /// Converts cost given in approximate (epsilon, delta) differential privacy to rdp. As
    /// rdp is a strictly stronger notion of differential privacy than approximate differential
    /// privacy, this is not generally possible. To make it possible, we assume that the
    /// differential privacy cost in adp given to this function results from the release of a result
    /// of a function with sensitivity one, to which gaussian noise was added.
    fn adp_to_rdp_cost_gaussian(&self, alphas: &RdpAlphas) -> AccountingType;

    /// Converts cost given in (pure) epsilon differential privacy to rdp. If given adp, the delta
    /// is ignored. As rdp is a strictly stronger notion of differential privacy than approximate
    /// differential privacy, this is not generally possible. To make it possible, we assume that
    /// the differential privacy cost in eps-dp given to this function results from the release of a
    /// result of a function with l1 sensitivity one, to which laplacian noise was added.
    fn adp_to_rdp_cost_laplacian(&self, alphas: &RdpAlphas) -> AccountingType;

    /// converts a budget in adp to rdp, by using that (alpha, eps) rdp implies
    /// (eps + (log(1/delta))/(alpha - 1), delta) (proposition 3, renyi differential privacy paper)
    /// as also used in privacy budget scheduling paper (section 5.2). In other words, if the
    /// computed rdp constraints hold, we know that the original constraints in approximate
    /// differential privacy also must hold.
    fn adp_to_rdp_budget(&self, alphas: &RdpAlphas) -> AccountingType;
}

// Trait implementations for AccountingType

impl Accounting for AccountingType {
    fn min_assign(&mut self, rhs: &AccountingType) {
        match (self, rhs) {
            (EpsDp { eps: eps1 }, EpsDp { eps: eps2 }) => {
                *eps1 = eps1.min(*eps2);
            }
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => {
                *eps1 = eps1.min(*eps2);
                *delta1 = delta1.min(*delta2);
            }
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => {
                vals1.min_assign(vals2);
            }
            (x, y) => panic!(
                "Tried to take max of different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }

    fn max_assign(&mut self, rhs: &AccountingType) {
        match (self, rhs) {
            (EpsDp { eps: eps1 }, EpsDp { eps: eps2 }) => {
                *eps1 = eps1.max(*eps2);
            }
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => {
                *eps1 = eps1.max(*eps2);
                *delta1 = delta1.max(*delta2);
            }
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => {
                vals1.max_assign(vals2);
            }
            (x, y) => panic!(
                "Tried to take max of different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }

    fn in_budget(&self, cost: &AccountingType) -> bool {
        let margin = F64Margin::default();
        match (self, cost) {
            (EpsDp { eps: eps_budget }, EpsDp { eps: eps_cost }) => {
                eps_cost < eps_budget || eps_cost.approx_eq(*eps_budget, margin)
            }
            (
                EpsDeltaDp {
                    eps: eps_budget,
                    delta: delta_budget,
                },
                EpsDeltaDp {
                    eps: eps_cost,
                    delta: delta_cost,
                },
            ) => {
                (eps_cost < eps_budget || eps_cost.approx_eq(*eps_budget, margin)) &&
                    (delta_cost < delta_budget || delta_cost.approx_eq(*delta_budget, margin))
            }
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => {
                vals1.in_budget(vals2, margin)
            }
            (x, y) => panic!(
                "Tried to take compute in budget of different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }

    fn approx_le(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (EpsDp { eps: eps1 }, EpsDp { eps: eps2 }) => {
                eps1.approx_eq(*eps2, F64Margin::default()) || eps1 < eps2
            }
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => {
                (eps1.approx_eq(*eps2, F64Margin::default()) || eps1 < eps2) && (delta1.approx_eq(*delta2, F64Margin::default()) || delta1 < delta2)
            }
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => {
                vals1.approx_le(vals2)
            }
            (x, y) => panic!(
                "Tried to take compute in budget of different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }

    fn rdp_cost_gaussian_noise(stddev: f64, alphas: &RdpAlphas) -> AccountingType {
        let to_rdp_cost = |alpha: f64| alpha / (2. * stddev.powf(2.)); // via "Renyi Differential Privacy" paper, Corollary 3
        Rdp {
            eps_values: (RdpAlphas::init_from_func(alphas, &to_rdp_cost)),
        }
    }

    fn adp_cost_gaussian_noise(stddev: f64, delta: f64) -> AccountingType {
        // via "The Algorithmic Foundations of Differential Privacy", Theorem A.1
        let c = (2. * (1.25 / delta).ln()).sqrt();
        EpsDeltaDp {
            eps: c / stddev,
            delta,
        }
    }

    fn is_valid(&self) -> bool {
        self.in_budget(&AccountingType::zero_clone(self))
    }

    fn zero_clone(other: &Self) -> Self {
        match other {
            EpsDp { eps: _ } => EpsDp { eps: 0.0 },
            EpsDeltaDp { eps: _, delta: _ } => EpsDeltaDp {
                eps: 0.0,
                delta: 0.0,
            },
            Rdp { eps_values } => Rdp {
                eps_values: eps_values.zero_clone(),
            },
        }
    }

    fn apply_func(&mut self, f: &dyn Fn(f64) -> f64) {
        match self {
            EpsDp { eps } => {
                *eps = f(*eps);
            }
            EpsDeltaDp { eps, delta } => {
                *eps = f(*eps);
                *delta = f(*delta);
            }
            Rdp { eps_values } => {
                eps_values.apply_func(f);
            }
        }
    }

    fn check_same_type(&self, other: &Self) -> bool {
        match (self, other) {
            (EpsDp { .. }, EpsDp { .. }) => true,
            (EpsDeltaDp { .. }, EpsDeltaDp { .. }) => true,
            (Rdp { eps_values: r1 }, Rdp { eps_values: r2 }) => r1.check_same_type(r2),
            (_, _) => false,
        }
    }

    fn remaining_ratios(&self, original_budget: &Self) -> AccountingTypeRatio {
        match (self, original_budget) {
            (EpsDp { eps: eps_new }, EpsDp { eps: eps_orig }) => {
                let eps_ratio = eps_new / eps_orig;
                assert!(
                    (0. <= eps_ratio || (0.).approx_eq(eps_ratio, F64Margin::default()))
                        && (eps_ratio <= 1. || eps_ratio.approx_eq(1., F64Margin::default())),
                    "The budget ratios needs to be between 0 and 1"
                );
                AccountingTypeRatio(EpsDp { eps: eps_ratio })
            }
            (
                EpsDeltaDp {
                    eps: eps_new,
                    delta: delta_new,
                },
                EpsDeltaDp {
                    eps: eps_orig,
                    delta: delta_orig,
                },
            ) => {
                let eps_ratio = eps_new / eps_orig;
                assert!(
                    (0. <= eps_ratio || (0.).approx_eq(eps_ratio, F64Margin::default()))
                        && (eps_ratio <= 1. || eps_ratio.approx_eq(1., F64Margin::default())),
                    "The budget ratios needs to be between 0 and 1"
                );
                let delta_ratio = delta_new / delta_orig;
                assert!(
                    (0. <= delta_ratio || (0.).approx_eq(delta_ratio, F64Margin::default()))
                        && (delta_ratio <= 1. || delta_ratio.approx_eq(1., F64Margin::default())),
                    "The budget ratios needs to be between 0 and 1"
                );

                AccountingTypeRatio(EpsDeltaDp {
                    eps: eps_ratio,
                    delta: delta_new / delta_orig,
                })
            }
            (
                Rdp {
                    eps_values: eps_values_new,
                },
                Rdp {
                    eps_values: eps_values_orig,
                },
            ) => AccountingTypeRatio(Rdp {
                eps_values: eps_values_new.remaining_ratios(eps_values_orig).0,
            }),
            (x, y) => panic!(
                "Tried to compute remaining percentage of {:?} and {:?}, which are incompatible",
                x, y
            ),
        }
    }
}

impl RdpAccounting for AccountingType {
    fn is_a1(&self) -> bool {
        matches!(self, Rdp { eps_values: A1(_) })
    }

    fn a1_to_eps_dp(&self) -> AccountingType {
        if let Rdp { eps_values: A1(x) } = self {
            EpsDp { eps: x[0] }
        } else {
            panic!("Tried to convert a1 to eps, but {:?} given", self)
        }
    }

    fn is_rdp(&self) -> bool {
        match self {
            EpsDp { .. } => false,
            EpsDeltaDp { .. } => false,
            Rdp { .. } => true,
        }
    }

    fn get_rdp_vec(&self) -> Vec<f64> {
        if let Rdp { eps_values } = self {
            eps_values.to_vec()
        } else {
            panic!("Cannot get rdp vec on non-rdp accounting type");
        }
    }

    fn get_value_for_alpha(&self, index: AlphaIndex) -> f64 {
        if let Rdp { eps_values } = self {
            eps_values.get_value_for_alpha(index)
        } else {
            panic!("Cannot get rdp cost for alpha on non-rdp accounting type");
        }
    }
}

impl AdpAccounting for AccountingType {
    fn adp_to_rdp_cost_gaussian(&self, alphas: &RdpAlphas) -> AccountingType {
        match self {
            EpsDeltaDp { eps, delta } => {
                // We first calculate the parameter sigma determining the scale of gaussian noise via
                // Theorem A.1 from "The Algorithmic Foundations of Differential Privacy"
                let c = (2. * (1.25 / delta).ln()).sqrt();
                let sigma = c / eps;

                AccountingType::rdp_cost_gaussian_noise(sigma, alphas)
            }
            _ => {
                panic!("Tried to converted non-adp cost {:?} to rdp", self)
            }
        }
    }

    fn adp_to_rdp_cost_laplacian(&self, alphas: &RdpAlphas) -> AccountingType {
        let eps = match self {
            EpsDp { eps } => eps,
            EpsDeltaDp { eps, .. } => {
                // delta is ignored
                eps
            }
            Rdp { .. } => {
                panic!("No need to convert rdp to rdp, as is already rdp")
            }
        };

        // since we assume sensitivity is one, the noise must be drawn from Lap(sensitivity f / eps).
        // since we assumed sensitivity 1, we therefore have Lap(1 / eps)
        // Source: Definition 3.3 / Theorem 3.6, the algorithmic foundations of differential privacy.

        let lambda = 1. / eps;

        let rdp_costs = AccountingType::rdp_cost_laplace(lambda, alphas.clone());

        Rdp {
            eps_values: rdp_costs,
        }
    }

    fn adp_to_rdp_budget(&self, alphas: &RdpAlphas) -> AccountingType {
        // Note: uses minimum 0, as negative budget and budget 0 are semantically equivalent
        // (i.e., no more requests are granted based upon this negative value),
        // but negative values are problematic with dpf and the ilp.

        match self {
            EpsDeltaDp { eps, delta } => match alphas {
                A1(alpha_vals) => Rdp {
                    eps_values: A1(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A2(alpha_vals) => Rdp {
                    eps_values: A2(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A3(alpha_vals) => Rdp {
                    eps_values: A3(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A4(alpha_vals) => Rdp {
                    eps_values: A4(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A5(alpha_vals) => Rdp {
                    eps_values: A5(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A7(alpha_vals) => Rdp {
                    eps_values: A7(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A10(alpha_vals) => Rdp {
                    eps_values: A10(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A13(alpha_vals) => Rdp {
                    eps_values: A13(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A14(alpha_vals) => Rdp {
                    eps_values: A14(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
                A15(alpha_vals) => Rdp {
                    eps_values: A15(AccountingType::get_rdp_limit_multiple(
                        eps, delta, alpha_vals,
                    )),
                },
            },
            _ => {
                panic!("Tried to convert non ADP budget ({:?}) to RDP", self)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct PerAlphaCostAndBudget {
    pub(crate) alpha: f64,
    costs: Vec<f64>,
    budget: f64,
}

/// Used to set which kinds of alpha reductions in
/// [calc_needed_alphas](rdpopt::RdpOptimization::calc_needed_alphas) should be applied.
/// * budget_reduction: Removes alphas where the budget is zero
/// * ratio_reduction: Removes any alpha where the cost/budget ratios are all worse than for some
/// other alpha (see paper for correctness proof)
/// * combinatorial_reduction: Removes any alpha which does not enable any new allocation
/// possibility
#[derive(Copy, Clone, Debug)]
pub struct AlphaReductions {
    pub(crate) budget_reduction: bool,
    pub(crate) ratio_reduction: bool,
    pub(crate) combinatorial_reduction: bool,
}

/// Used in [calc_needed_alphas](rdpopt::RdpOptimization::calc_needed_alphas) to return how many
/// alphas were removed by which optimization (if the optimization was enabled)
#[derive(Copy, Clone, Debug)]
pub struct AlphaReductionsResult {
    pub(crate) budget_reduction: Option<usize>,
    pub(crate) ratio_reduction: Option<usize>,
    pub(crate) combinatorial_reduction: Option<usize>,
}

impl AccountingType {
    /// Converts a budget in adp to rdp. See [AccountingType::adp_to_rdp_budget] for more
    /// information.
    fn get_rdp_limit_multiple<const N: usize>(
        eps: &f64,
        delta: &f64,
        alpha_vals: &[f64; N],
    ) -> [f64; N] {
        fn to_eps_rdp_limit(eps_dp: f64, delta_dp: f64, alpha: f64) -> f64 {
            (0f64).max(eps_dp - ((1. / delta_dp).ln() / (alpha - 1.)))
        }

        let mut res = [0.; N];
        for i in 0..alpha_vals.len() {
            res[i] = to_eps_rdp_limit(*eps, *delta, alpha_vals[i]);
        }
        res
    }

    /// Computes rdp cost of releasing result of function f with sensitivity 1, to which laplacian
    /// noise with scale lambda has been added. Tries computation using f64, and switches to
    /// arbitrary precision if that does not work out. Source: Corollary 2, renyi differential
    /// privacy paper.
    fn rdp_cost_laplace(lambda: f64, mut alphas: RdpAlphas) -> RdpAlphas {
        let converter = move |alpha: f64| {
            let res: f64 = AccountingType::rdp_cost_laplace_single_alpha(lambda, alpha);
            if res.is_finite() {
                res
            } else {
                AccountingType::rdp_cost_laplace_single_alpha_high_prec(lambda, alpha, 100)
            }
        };

        alphas.apply_func(&converter);

        alphas
    }

    /// Computes rdp cost for specific value of alpha of releasing result of function f with
    /// sensitivity 1, to which laplacian noise with scale lambda has been added
    fn rdp_cost_laplace_single_alpha(lambda: f64, alpha: f64) -> f64 {
        1. / (alpha - 1.)
            * (alpha / (2. * alpha - 1.) * std::f64::consts::E.powf((alpha - 1.) / lambda)
                + (alpha - 1.) / (2. * alpha - 1.) * std::f64::consts::E.powf(-alpha / lambda))
            .ln()
    }

    /// Computes rdp cost for specific value of alpha of releasing result of function f with
    /// sensitivity 1, to which laplacian noise with scale lambda has been added. In contrast to
    /// [AccountingType::rdp_cost_laplace_single_alpha], has arbitrary precision, but is a lot
    /// slower.
    fn rdp_cost_laplace_single_alpha_high_prec(lambda: f64, alpha: f64, precision: u32) -> f64 {
        let e = Float::with_val(precision, std::f64::consts::E);
        let alpha = Float::with_val(precision, alpha);
        (1_f64 / (alpha.clone() - 1_f64)
            * (alpha.clone() / (2_f64 * alpha.clone() - 1_f64)
                * e.clone().pow((alpha.clone() - 1_f64) / lambda)
                + (alpha.clone() - 1_f64) / (2_f64 * alpha.clone() - 1_f64)
                    * e.pow(-alpha / lambda))
            .ln())
        .to_f64()
    }
}

impl ApproxEq for &AccountingType {
    type Margin = F64Margin;

    #[inline]
    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        match (self, other) {
            (EpsDp { eps: eps1 }, EpsDp { eps: eps2 }) => eps1.approx_eq(*eps2, margin),
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => eps1.approx_eq(*eps2, margin) && delta1.approx_eq(*delta2, margin),
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => {
                vals1.approx_eq(vals2, margin)
            }
            (x, y) => panic!(
                "Tried to subtract different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }
}

impl Add<&AccountingType> for AccountingType {
    type Output = AccountingType;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl Add for AccountingType {
    type Output = AccountingType;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Sub for AccountingType {
    type Output = AccountingType;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub for &AccountingType {
    type Output = AccountingType;
    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (EpsDp { eps: eps1 }, EpsDp { eps: eps2 }) => EpsDp { eps: eps1 - eps2 },
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => EpsDeltaDp {
                eps: eps1 - eps2,
                delta: delta1 - delta2,
            },
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => Rdp {
                eps_values: vals1 - vals2,
            },
            (x, y) => panic!(
                "Tried to subtract different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }
}

impl Add for &AccountingType {
    // can use "+" syntax
    type Output = AccountingType;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (EpsDp { eps: eps1 }, EpsDp { eps: eps2 }) => EpsDp { eps: eps1 + eps2 },
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => EpsDeltaDp {
                eps: eps1 + eps2,
                delta: delta1 + delta2,
            },
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => Rdp {
                eps_values: vals1 + vals2,
            },
            (x, y) => panic!(
                "Tried to add different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }
}

impl AddAssign<&AccountingType> for AccountingType {
    fn add_assign(&mut self, rhs: &AccountingType) {
        match (self, rhs) {
            (EpsDp { eps: eps1 }, EpsDp { eps: eps2 }) => {
                *eps1 += *eps2;
            }
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => {
                *eps1 += *eps2;
                *delta1 += *delta2;
            }
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => {
                *vals1 += vals2;
            }
            (x, y) => panic!(
                "Tried to add different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }
}

impl SubAssign<&AccountingType> for AccountingType {
    fn sub_assign(&mut self, rhs: &AccountingType) {
        match (self, rhs) {
            (EpsDp { eps: eps1 }, EpsDp { eps: eps2 }) => {
                *eps1 -= *eps2;
            }
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => {
                *eps1 -= *eps2;
                *delta1 -= *delta2;
            }
            (Rdp { eps_values: vals1 }, Rdp { eps_values: vals2 }) => {
                *vals1 -= vals2;
            }
            (x, y) => panic!(
                "Tried to subtract different accounting types. Lhs: {:?}, Rhs: {:?}",
                x, y
            ),
        }
    }
}

impl fmt::Display for AccountingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EpsDp { eps } => write!(f, "ε={}", eps),
            EpsDeltaDp { eps, delta } => write!(f, "ε={} δ={}", eps, delta),
            Rdp { eps_values } => write!(f, "rdp={}", eps_values),
        }
    }
}

// TODO [later] can we do this better e.g., with a macro?
impl fmt::Display for RdpAlphas {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            A1(v) => write!(f, "[{}]", v[0]),
            A2(v) => write!(f, "[{}, {}]", v[0], v[1]),
            A3(v) => write!(f, "[{}, {}, {}]", v[0], v[1], v[2]),
            A4(v) => write!(f, "[{}, {}, {}, {}]", v[0], v[1], v[2], v[3]),
            A5(v) => write!(f, "[{}, {}, {}, {}, {}]", v[0], v[1], v[2], v[3], v[4]),
            A7(v) => write!(
                f,
                "[{}, {}, {}, {}, {}, {}, {}]",
                v[0], v[1], v[2], v[3], v[4], v[5], v[6]
            ),
            A10(v) => write!(
                f,
                "[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]",
                v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]
            ),
            A13(v) => write!(
                f,
                "[{}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}]",
                v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], v[12]
            ),
            A14(v) => write!(
                f,
                "[{}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}]",
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6],
                v[7],
                v[8],
                v[9],
                v[10],
                v[11],
                v[12],
                v[13]
            ),
            A15(v) => write!(
                f,
                "[{}, {}, {}, {}, {}, {}, {}, {}, {}, {},{}, {}, {}, {}, {}]",
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6],
                v[7],
                v[8],
                v[9],
                v[10],
                v[11],
                v[12],
                v[13],
                v[14]
            ),
        }
    }
}

//tests

#[cfg(test)]
mod tests {
    use crate::dprivacy::rdp_alphas_accounting::PubRdpAccounting;
    use crate::dprivacy::AccountingType::{EpsDeltaDp, EpsDp, Rdp};
    use crate::dprivacy::RdpAlphas::*;
    use crate::dprivacy::{Accounting, AccountingType, AdpAccounting, RdpAccounting};
    use float_cmp::{ApproxEq, F64Margin};

    #[test]
    fn test_get_rdp_vector_and_indices() {
        let rdp_cost: AccountingType = Rdp {
            eps_values: A3([1.0, 2.0, 3.0]),
        };
        let sol_vec = vec![1.0, 2.0, 3.0];
        assert_eq!(rdp_cost.get_rdp_vec(), sol_vec);
        for (i, a) in rdp_cost.get_alpha_indices().into_iter().enumerate() {
            assert_eq!(a.0, i);
            assert_eq!(rdp_cost.get_value_for_alpha(a), sol_vec[i]);
        }
    }

    #[test]
    fn test_remaining_percentage() {
        let new_budget = EpsDp { eps: 0.4 };
        let original_budget = EpsDp { eps: 1.2 };
        let ratio = new_budget.remaining_ratios(&original_budget);
        match ratio.0 {
            EpsDp { eps } => {
                assert!(eps.approx_eq(1. / 3., F64Margin::default()));
            }
            _ => panic!("Wrong return type"),
        }

        let new_budget = EpsDeltaDp {
            eps: 0.3,
            delta: 1e-8,
        };
        let original_budget = EpsDeltaDp {
            eps: 0.9,
            delta: 1e-7,
        };
        let ratio = new_budget.remaining_ratios(&original_budget);
        match ratio.0 {
            EpsDeltaDp { eps, delta } => {
                assert!(eps.approx_eq(1. / 3., F64Margin::default()));
                assert!(delta.approx_eq(1. / 10., F64Margin::default()));
            }
            _ => panic!("Wrong return type"),
        }

        let new_budget = Rdp {
            eps_values: A3([0.2, 0.4, 0.6]),
        };
        let original_budget = Rdp {
            eps_values: A3([0.6, 0.8, 0.6]),
        };
        let ratio = new_budget.remaining_ratios(&original_budget);
        match ratio.0 {
            Rdp {
                eps_values: A1(vals),
            } => {
                assert!(vals[0].approx_eq(1., F64Margin::default()));
            }
            _ => panic!("Wrong return type"),
        }
    }

    #[test]
    #[should_panic(expected = "The budget ratios needs to be between 0 and 1")]
    fn test_remaining_percentage_panic() {
        let new_budget = EpsDp { eps: 1.4 };
        let original_budget = EpsDp { eps: 1.2 };
        let _ratio = new_budget.remaining_ratios(&original_budget);
    }

    #[test]
    fn test_rdp_cost_laplace() {
        let alphas = [1.5, 2., 4., 8., 16., 32., 64.];
        let acc_alphas = A7(alphas);
        let lambdas = [0.001, 0.01, 0.1, 1., 10., 100.];
        for lambda in lambdas {
            let mut res_vec = Vec::new();
            for alpha in alphas {
                let res1 = AccountingType::rdp_cost_laplace_single_alpha(lambda, alpha);
                let res2 =
                    AccountingType::rdp_cost_laplace_single_alpha_high_prec(lambda, alpha, 64);
                let res3 =
                    AccountingType::rdp_cost_laplace_single_alpha_high_prec(lambda, alpha, 128);
                assert!(res2.is_finite());
                assert!(res2.approx_eq(res3, F64Margin::default()));
                assert!(res1.is_infinite() || res1.approx_eq(res2, F64Margin::default()));
                res_vec.push(res2);
            }
            let res_vec_2 = AccountingType::rdp_cost_laplace(lambda, acc_alphas.clone()).to_vec();
            assert_eq!(res_vec.len(), res_vec_2.len());
            for i in 0..res_vec.len() {
                assert!(res_vec[i].approx_eq(res_vec_2[i], F64Margin::default()));
                if i > 0 {
                    assert!(res_vec[i] >= res_vec[i - 1])
                }
            }
        }

        let adp_cost = EpsDeltaDp {
            eps: 1.0,
            delta: 0.1,
        };

        let epsdp_cost = EpsDp { eps: 1.0 };

        let rdp_cost_1 = adp_cost
            .adp_to_rdp_cost_laplacian(&acc_alphas)
            .get_rdp_vec();
        let rdp_cost_2 = epsdp_cost
            .adp_to_rdp_cost_laplacian(&acc_alphas)
            .get_rdp_vec();

        assert!(rdp_cost_1.approx_eq(&rdp_cost_2, F64Margin::default()));
    }

    #[test]
    fn a1_to_eps_dp() {
        let a1 = EpsDp { eps: 1.0 };
        let a2 = EpsDeltaDp {
            eps: 1.0,
            delta: 1.0,
        };
        let a3 = Rdp {
            eps_values: A1([1.0]),
        };
        let a4 = Rdp {
            eps_values: A5([1.0, 1.0, 1.0, 1.0, 1.0]),
        };
        assert!(!a1.is_a1());
        assert!(!a2.is_a1());
        assert!(a3.is_a1());
        assert!(!a4.is_a1());
        a3.a1_to_eps_dp();
    }

    #[test]
    #[should_panic(
        expected = "Tried to convert a1 to eps, but Rdp { eps_values: A5([1.0, 1.0, 1.0, 1.0, 1.0]) } given"
    )]
    fn a1_to_eps_dp_panic() {
        let a = Rdp {
            eps_values: A5([1.0, 1.0, 1.0, 1.0, 1.0]),
        };
        a.a1_to_eps_dp();
    }

    #[test]
    fn adp_to_rdp_cost_conversion() {
        let alpha_vals = [2., 3., 4., 6., 8.];
        let alphas = A5(alpha_vals);
        let adp_cost = EpsDeltaDp {
            eps: 1.0,
            delta: 1e-6,
        };
        let rdp_cost = adp_cost.adp_to_rdp_cost_gaussian(&alphas);
        match rdp_cost {
            Rdp { eps_values } => {
                match eps_values {
                    A5(arr) => {
                        // since we use gaussian noise, the values should be proportional to alpha
                        let reference_val: f64 = arr[0] / alpha_vals[0];
                        for i in 1..5 {
                            assert!(reference_val
                                .approx_eq(arr[i] / alpha_vals[i], F64Margin::default()))
                        }
                    }
                    _ => panic!("adp_to_rdp_cost did not return correct number of rdp values"),
                }
            }
            _ => panic!("adp_to_rdp_cost did not return rdp costs"),
        }
    }

    #[test]
    fn test_rdp_addition() {
        let mut rdp5_1 = A5([0.5, 0.6, 0.7, 0.8, 0.9]);
        let rdp5_2 = A5([1.5, 1.6, 1.7, 1.8, 1.9]);
        let rdp7_1 = A7([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]);
        let rdp7_2 = A7([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]);

        assert!((&rdp5_1 + &rdp5_2).approx_eq(&A5([2.0, 2.2, 2.4, 2.6, 2.8]), F64Margin::default()));
        assert!((&rdp7_1 + &rdp7_2).approx_eq(
            &A7([2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2]),
            F64Margin::default()
        ));

        rdp5_1 += &rdp5_2;
        assert!(rdp5_1.approx_eq(&A5([2.0, 2.2, 2.4, 2.6, 2.8]), F64Margin::default()));
        rdp5_1 += &rdp5_2;
        assert!(rdp5_1.approx_eq(&A5([3.5, 3.8, 4.1, 4.4, 4.7]), F64Margin::default()));
    }

    #[test]
    fn test_rdp_subtraction() {
        let mut rdp5_1 = A5([0.5, 0.6, 0.7, 0.8, 0.9]);
        let rdp5_2 = A5([1.5, 1.6, 1.7, 1.8, 1.9]);

        assert!((&rdp5_1 - &rdp5_2)
            .approx_eq(&A5([-1.0, -1.0, -1.0, -1.0, -1.0]), F64Margin::default()));
        assert!(!(&rdp5_1 - &rdp5_2)
            .approx_eq(&A5([-1.0, -1.1, -1.0, -1.0, -1.0]), F64Margin::default()));

        rdp5_1 -= &rdp5_2;
        assert!(rdp5_1.approx_eq(&A5([-1.0, -1.0, -1.0, -1.0, -1.0]), F64Margin::default()));
        rdp5_1 -= &rdp5_2;
        assert!(rdp5_1.approx_eq(&A5([-2.5, -2.6, -2.7, -2.8, -2.9]), F64Margin::default()));
    }

    #[test]
    #[should_panic]
    fn test_rdp_panic() {
        let rdp5_1 = A5([0.5, 0.6, 0.7, 0.8, 0.9]);
        let rdp7_1 = A7([1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1]);

        let _ = rdp5_1 + rdp7_1;
    }

    #[test]
    fn test_accounting_type_approx_eq() {
        let eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.0 };
        let eps_dp_3 = EpsDp { eps: 2.0 };

        assert!(eps_dp_1.approx_eq(&eps_dp_2, F64Margin::default()));
        assert!(!eps_dp_1.approx_eq(&eps_dp_3, F64Margin::default()));

        let eps_delta_dp_1 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.1,
        };
        let eps_delta_dp_2 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.1,
        };
        let eps_delta_dp_3 = EpsDeltaDp {
            eps: 3.0,
            delta: 0.3,
        };

        assert!(eps_delta_dp_1.approx_eq(&eps_delta_dp_2, F64Margin::default()));
        assert!(!eps_delta_dp_1.approx_eq(&eps_delta_dp_3, F64Margin::default()));

        let rdp_1 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([1.0, 1.0, 5.0, 7.0, 5.0]),
        };

        assert!(rdp_1.approx_eq(&rdp_2, F64Margin::default()));
        assert!(!rdp_1.approx_eq(&rdp_3, F64Margin::default()));
    }

    #[test]
    fn test_accounting_type_addition() {
        let mut eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };
        let eps_dp_3 = EpsDp { eps: 2.5 };

        assert!((&eps_dp_1 + &eps_dp_2).approx_eq(&eps_dp_3, F64Margin::default()));
        eps_dp_1 += &eps_dp_2;
        assert!(eps_dp_1.approx_eq(&eps_dp_3, F64Margin::default()));

        let mut eps_delta_dp_1 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.1,
        };
        let eps_delta_dp_2 = EpsDeltaDp {
            eps: 2.0,
            delta: 0.2,
        };
        let eps_delta_dp_3 = EpsDeltaDp {
            eps: 3.0,
            delta: 0.3,
        };

        assert!(
            (&eps_delta_dp_1 + &eps_delta_dp_2).approx_eq(&eps_delta_dp_3, F64Margin::default())
        );
        eps_delta_dp_1 += &eps_delta_dp_2;
        assert!(eps_delta_dp_1.approx_eq(&eps_delta_dp_3, F64Margin::default()));

        let mut rdp_1 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, -1.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([1.0, 1.0, 5.0, 7.0, 5.0]),
        };

        assert!((&rdp_1 + &rdp_2).approx_eq(&rdp_3, F64Margin::default()));
        rdp_1 += &rdp_2;
        assert!(rdp_1.approx_eq(&rdp_3, F64Margin::default()));
    }

    #[test]
    fn test_accounting_type_subtraction() {
        let mut eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };
        let eps_dp_3 = EpsDp { eps: -0.5 };

        assert!((&eps_dp_1 - &eps_dp_2).approx_eq(&eps_dp_3, F64Margin::default()));
        eps_dp_1 -= &eps_dp_2;
        assert!(eps_dp_1.approx_eq(&eps_dp_3, F64Margin::default()));

        let mut eps_delta_dp_1 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.1,
        };
        let eps_delta_dp_2 = EpsDeltaDp {
            eps: 2.0,
            delta: 0.2,
        };
        let eps_delta_dp_3 = EpsDeltaDp {
            eps: -1.0,
            delta: -0.1,
        };

        assert!(
            (&eps_delta_dp_1 - &eps_delta_dp_2).approx_eq(&eps_delta_dp_3, F64Margin::default())
        );
        eps_delta_dp_1 -= &eps_delta_dp_2;
        assert!(eps_delta_dp_1.approx_eq(&eps_delta_dp_3, F64Margin::default()));

        let mut rdp_1 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, -1.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([1.0, 3.0, 1.0, 1.0, 5.0]),
        };

        assert_eq!(rdp_3.get_rdp_vec(), [1.0, 3.0, 1.0, 1.0, 5.0].to_vec());
        assert!((&rdp_1 - &rdp_2).approx_eq(&rdp_3, F64Margin::default()));
        rdp_1 -= &rdp_2;
        assert!(rdp_1.approx_eq(&rdp_3, F64Margin::default()));
    }

    #[test]
    fn test_accounting_type_min() {
        let mut eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };
        let eps_dp_3 = EpsDp { eps: 1.0 };
        eps_dp_1.min_assign(&eps_dp_2);
        assert!(eps_dp_1.approx_eq(&eps_dp_3, F64Margin::default()));

        let mut eps_delta_dp_1 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.2,
        };
        let eps_delta_dp_2 = EpsDeltaDp {
            eps: 2.0,
            delta: 0.1,
        };
        let eps_delta_dp_3 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.1,
        };

        eps_delta_dp_1.min_assign(&eps_delta_dp_2);
        assert!(eps_delta_dp_1.approx_eq(&eps_delta_dp_3, F64Margin::default()));

        let mut rdp_1 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, -1.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([0.0, -1.0, 2.0, 3.0, 0.0]),
        };

        rdp_1.min_assign(&rdp_2);
        assert!(rdp_1.approx_eq(&rdp_3, F64Margin::default()));
    }

    #[test]
    fn test_accounting_type_max() {
        let mut eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };
        let eps_dp_3 = EpsDp { eps: 1.5 };
        eps_dp_1.max_assign(&eps_dp_2);
        assert!(eps_dp_1.approx_eq(&eps_dp_3, F64Margin::default()));

        let mut eps_delta_dp_1 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.2,
        };
        let eps_delta_dp_2 = EpsDeltaDp {
            eps: 2.0,
            delta: 0.1,
        };
        let eps_delta_dp_3 = EpsDeltaDp {
            eps: 2.0,
            delta: 0.2,
        };

        eps_delta_dp_1.max_assign(&eps_delta_dp_2);
        assert!(eps_delta_dp_1.approx_eq(&eps_delta_dp_3, F64Margin::default()));

        let mut rdp_1 = Rdp {
            eps_values: A5([1.0, 2.0, 1.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, -1.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([1.0, 2.0, 2.0, 4.0, 5.0]),
        };

        rdp_1.max_assign(&rdp_2);
        assert!(rdp_1.approx_eq(&rdp_3, F64Margin::default()));
    }

    #[test]
    fn test_in_budget() {
        let eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };
        assert!(!eps_dp_1.is_rdp());
        assert!(eps_dp_2.in_budget(&eps_dp_1));
        assert!(eps_dp_1.in_budget(&eps_dp_1));
        assert!(!eps_dp_1.in_budget(&eps_dp_2));

        let eps_delta_dp_1 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.2,
        };
        let eps_delta_dp_2 = EpsDeltaDp {
            eps: 2.0,
            delta: 0.1,
        };
        let eps_delta_dp_3 = EpsDeltaDp {
            eps: 3.0,
            delta: 0.1,
        };

        assert!(!eps_delta_dp_1.is_rdp());
        assert!(eps_delta_dp_1.in_budget(&eps_delta_dp_1));
        assert!(!eps_delta_dp_1.in_budget(&eps_delta_dp_2));
        assert!(!eps_delta_dp_2.in_budget(&eps_delta_dp_1));
        assert!(eps_delta_dp_3.in_budget(&eps_delta_dp_2));
        assert!(!eps_delta_dp_3.in_budget(&eps_delta_dp_1));

        let rdp_1 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 6.0, 5.0]),
        };

        assert!(rdp_1.is_rdp());
        assert!(rdp_1.in_budget(&rdp_1));
        assert!(rdp_2.in_budget(&rdp_1));
        assert!(!rdp_1.in_budget(&rdp_2));
        assert!(rdp_3.in_budget(&rdp_1));
        assert!(rdp_1.in_budget(&rdp_3));
        assert!(rdp_3.in_budget(&rdp_2));
        assert!(rdp_2.in_budget(&rdp_3));
    }

    #[test]
    fn test_approx_le() {
        let eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };
        let eps_dp_3 = EpsDp { eps: 1.5 };
        assert!(eps_dp_1.approx_le(&eps_dp_1));
        assert!(eps_dp_1.approx_le(&eps_dp_2));
        assert!(!eps_dp_2.approx_le(&eps_dp_1));
        assert!(eps_dp_3.approx_le(&eps_dp_2));

        let eps_delta_dp_1 = EpsDeltaDp {
            eps: 1.0,
            delta: 0.2,
        };
        let eps_delta_dp_2 = EpsDeltaDp {
            eps: 2.0,
            delta: 0.1,
        };
        let eps_delta_dp_3 = EpsDeltaDp {
            eps: 3.0,
            delta: 0.1,
        };

        assert!(eps_delta_dp_1.approx_le(&eps_delta_dp_1));
        assert!(!eps_delta_dp_1.approx_le(&eps_delta_dp_2));
        assert!(!eps_delta_dp_2.approx_le(&eps_delta_dp_1));
        assert!(eps_delta_dp_2.approx_le(&eps_delta_dp_3));
        assert!(!eps_delta_dp_1.approx_le(&eps_delta_dp_3));

        let rdp_1 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([1.0, 2.0, 3.0, 6.0, 5.0]),
        };

        assert!(rdp_1.approx_le(&rdp_1));
        assert!(rdp_1.approx_le(&rdp_2));
        assert!(!rdp_2.approx_le(&rdp_1));
        assert!(!rdp_3.approx_le(&rdp_1));
        assert!(rdp_1.approx_le(&rdp_3));
        assert!(!rdp_3.approx_le(&rdp_2));
        assert!(!rdp_2.approx_le(&rdp_3));
    }

    #[test]
    fn test_gaussian_cost() {
        let alpha_arr = [
            1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 8., 16., 32., 64.,
            1000000., // From Privacy Budget Scheduling
        ];

        let alphas = A13(alpha_arr);

        let adp_delta_low = AccountingType::adp_cost_gaussian_noise(10., 0.01);
        let adp_delta_high = AccountingType::adp_cost_gaussian_noise(10., 0.0001);
        match (&adp_delta_low, &adp_delta_high) {
            (
                EpsDeltaDp {
                    eps: eps1,
                    delta: delta1,
                },
                EpsDeltaDp {
                    eps: eps2,
                    delta: delta2,
                },
            ) => {
                assert!(
                    *delta1 == 0.01 && *delta2 == 0.0001,
                    "Adp cost initialized with incorrect deltas"
                );
                assert!(
                    *eps1 < *eps2,
                    "cost in eps was higher with higher delta: {:?}, {:?}",
                    &adp_delta_low,
                    &adp_delta_high
                );
            }
            (x, y) => {
                panic!(
                    "adp_cost of gaussian noise did not return EpsDeltaDp: {:?}, adp_delta_high: {:?}",
                    x, y
                )
            }
        }

        let rdp_cost = AccountingType::rdp_cost_gaussian_noise(10., &alphas);
        match &rdp_cost {
            Rdp { eps_values } => {
                match eps_values {
                    A13(eps_arr) => {
                        for i in 1..13 {
                            // gaussian noise cost is proportional to alpha value
                            assert!((eps_arr[i] / alpha_arr[i])
                                .approx_eq(eps_arr[0] / alpha_arr[0], F64Margin::default()), "rdp_cost_gaussian_noise incorrectly computed costs for alpha values");
                        }
                    }
                    x => {
                        panic!("rdp_cost_gaussian_noise returned rdp costs with different length than supplied values of alpha: {:?}", x)
                    }
                }
            }
            _ => {
                panic!(
                    "rdp_cost_gaussian_noise did not return rdp costs, returned {:?}",
                    rdp_cost
                )
            }
        }
    }
}
