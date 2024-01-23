use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
use crate::dprivacy::AlphaIndex;
use float_cmp::{ApproxEq, F64Margin};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Sub, SubAssign};

#[derive(Deserialize, Clone, Debug, PartialEq, PartialOrd, Serialize)]
#[serde(untagged)]
pub enum RdpAlphas {
    A1([f64; 1]),
    A2([f64; 2]),
    A3([f64; 3]),
    A4([f64; 4]),
    A5([f64; 5]),
    A7([f64; 7]),
    A10([f64; 10]),
    A13([f64; 13]),
    A14([f64; 14]),
    A15([f64; 15]),
}

/// Contains the various methods that can be used on/with the RdpAlphas struct.
/// Methods that take two or more RdpAlphas containing epsilon values assume that the same alpha
/// values in the same order are implicitly associated with these epsilon values. If the number
/// of alpha/epsilon values is different, a panic will be thrown, otherwise, the methods will
/// spuriously succeed,
pub(super) trait RdpAlphasAccounting: Sized {
    /// "Merges" two rdp accounting types by assigning the minimum epsilon value for each alpha
    /// value.
    fn min_assign(&mut self, rhs: &Self);

    /// Same as [min_assign](Self::min_assign), but assigns the maximum instead.
    fn max_assign(&mut self, rhs: &Self);

    /// Checks if the the given cost is in budget, where self functions as a budget.
    /// According to the properties of rdp, this means that the cost must be <= the budget
    /// (in epsilon) for one or more values of alpha. Note that this is different from
    /// [approx_le](Self::approx_le) where lhs <= rhs must hold for each alpha value.
    fn in_budget(&self, cost: &Self, margin: F64Margin) -> bool;

    /// Checks if for each alpha value, the epsilon value of lhs is <= rhs. Note that this should
    /// NOT be used to check if a cost is admissible under a certain budget, use
    /// [in_budget](Self::in_budget) for this purpose instead.
    fn approx_le(&self, rhs: &Self) -> bool;

    /// Applies a function for each (alpha, eps_alpha) pair: Takes alpha and eps_alpha,
    /// returns new eps_alpha. Same as [apply_func](Self::apply_func), but
    /// with the function taking the alpha value as additional input.
    fn apply_rdp_func(&mut self, alphas: &RdpAlphas, f: &dyn Fn(f64, f64) -> f64);

    /// Applies a function for each eps_alpha. Same as [apply_rdp_func](Self::apply_rdp_func), but
    /// without the function taking the alpha value as input.
    fn apply_func(&mut self, f: &dyn Fn(f64) -> f64);

    /// Returns a new RdpAlpha with zeros everywhere, with same number of alphas/eps as given argument
    fn zero_clone(&self) -> Self;

    /// Returns a new RdpAlpha with values initialized depending on value of alpha.
    /// Same as [apply_func](Self::apply_func), but
    /// with the function taking any prior epsilon value as additional input.
    fn init_from_func(alphas: &RdpAlphas, f: &dyn Fn(f64) -> f64) -> Self;

    /// Checks whether the same type of rdp is used (i.e., how many alpha values there are).
    /// Note that as the accounting types do not have individually saved which alpha values
    /// correspond to the present epsilon values, it is not possible to check that the
    /// alpha values of the inputs are indeed the same.
    fn check_same_type(&self, other: &Self) -> bool;

    /// Only keeps the values at positions where the mask is true
    fn reduce_alphas(&self, mask: &[bool]) -> Option<Self>;

    /// The implementation of [link](../trait.Accounting.html#tymethod.remaining_percentage) for
    /// rdp.
    fn remaining_ratios(&self, original_budget: &Self) -> RdpAlphasRatios;
}

pub(super) struct RdpAlphasRatios(pub(super) RdpAlphas);

pub trait PubRdpAccounting: Sized {
    /// returns the contained values as vec (constructed from the array)
    fn to_vec(&self) -> Vec<f64>;

    /// Given a vec of floats, tries to construct an RdpAlphas object from it.
    fn from_vec(vec: Vec<f64>) -> Option<Self>;

    /// Get the cost or budget for a certain alpha
    fn get_value_for_alpha(&self, index: AlphaIndex) -> f64;
}

// Trait implementations for RdpAlphas
impl RdpAlphasAccounting for RdpAlphas {
    fn min_assign(&mut self, rhs: &RdpAlphas) {
        match (self, rhs) {
            (A1(v1), A1(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A2(v1), A2(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A3(v1), A3(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A4(v1), A4(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A5(v1), A5(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A7(v1), A7(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A10(v1), A10(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A13(v1), A13(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A14(v1), A14(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (A15(v1), A15(v2)) => {
                element_wise_min_in_place(v1, v2);
            }
            (x, y) => {
                panic!("Computing min these RDP budgets is not supported (likely of different length): {:?} and {:?}", x, y)
            }
        }
    }

    fn max_assign(&mut self, rhs: &RdpAlphas) {
        match (self, rhs) {
            (A1(v1), A1(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A2(v1), A2(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A3(v1), A3(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A4(v1), A4(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A5(v1), A5(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A7(v1), A7(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A10(v1), A10(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A13(v1), A13(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A14(v1), A14(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (A15(v1), A15(v2)) => {
                element_wise_max_in_place(v1, v2);
            }
            (x, y) => {
                panic!("Computing min these RDP budgets is not supported (likely of different length): {:?} and {:?}", x, y)
            }
        }
    }

    fn in_budget(&self, cost: &RdpAlphas, margin: F64Margin) -> bool {
        match (self, cost) {
            (A1(v1), A1(v2)) => rdp_in_budget(v1, v2, margin),
            (A2(v1), A2(v2)) => rdp_in_budget(v1, v2, margin),
            (A3(v1), A3(v2)) => rdp_in_budget(v1, v2, margin),
            (A4(v1), A4(v2)) => rdp_in_budget(v1, v2, margin),
            (A5(v1), A5(v2)) => rdp_in_budget(v1, v2, margin),
            (A7(v1), A7(v2)) => rdp_in_budget(v1, v2, margin),
            (A10(v1), A10(v2)) => rdp_in_budget(v1, v2, margin),
            (A13(v1), A13(v2)) => rdp_in_budget(v1, v2, margin),
            (A14(v1), A14(v2)) => rdp_in_budget(v1, v2, margin),
            (A15(v1), A15(v2)) => rdp_in_budget(v1, v2, margin),
            (x, y) => {
                panic!("Computing in_budget of these RDP budgets is not supported (likely of different length): {:?} and {:?}", x, y)
            }
        }
    }

    fn approx_le(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (A1(v1), A1(v2)) => rdp_approx_le(v1, v2),
            (A2(v1), A2(v2)) => rdp_approx_le(v1, v2),
            (A3(v1), A3(v2)) => rdp_approx_le(v1, v2),
            (A4(v1), A4(v2)) => rdp_approx_le(v1, v2),
            (A5(v1), A5(v2)) => rdp_approx_le(v1, v2),
            (A7(v1), A7(v2)) => rdp_approx_le(v1, v2),
            (A10(v1), A10(v2)) => rdp_approx_le(v1, v2),
            (A13(v1), A13(v2)) => rdp_approx_le(v1, v2),
            (A14(v1), A14(v2)) => rdp_approx_le(v1, v2),
            (A15(v1), A15(v2)) => rdp_approx_le(v1, v2),
            (x, y) => {
                panic!("Computing approx_le of these RDP budgets is not supported (likely of different length): {:?} and {:?}", x, y)
            }
        }
    }

    fn apply_rdp_func(&mut self, alphas: &RdpAlphas, f: &dyn Fn(f64, f64) -> f64) {
        match (self, alphas) {
            (A1(eps), A1(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A2(eps), A2(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A3(eps), A3(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A4(eps), A4(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A5(eps), A5(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A7(eps), A7(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A10(eps), A10(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A13(eps), A13(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A14(eps), A14(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (A15(eps), A15(alphas)) => rdp_arr_apply_func(alphas, eps, f),
            (_, _) => {
                panic!("Error: alphas and given rdp accounting type have different lengths")
            }
        }
    }

    fn apply_func(&mut self, f: &dyn Fn(f64) -> f64) {
        let f_extended = |_: f64, eps: f64| f(eps);
        self.apply_rdp_func(&self.zero_clone(), &f_extended);
    }

    fn zero_clone(&self) -> Self {
        match self {
            A1(_) => A1([0.; 1]),
            A2(_) => A2([0.; 2]),
            A3(_) => A3([0.; 3]),
            A4(_) => A4([0.; 4]),
            A5(_) => A5([0.; 5]),
            A7(_) => A7([0.; 7]),
            A10(_) => A10([0.; 10]),
            A13(_) => A13([0.; 13]),
            A14(_) => A14([0.; 14]),
            A15(_) => A15([0.; 15]),
        }
    }

    fn init_from_func(alphas: &RdpAlphas, f: &dyn Fn(f64) -> f64) -> Self {
        let mut res = alphas.zero_clone();
        let f_expanded = |alpha: f64, _: f64| f(alpha);
        res.apply_rdp_func(alphas, &f_expanded);
        res
    }

    fn check_same_type(&self, other: &Self) -> bool {
        match (self, other) {
            (A1(_), A1(_)) => true,
            (A2(_), A2(_)) => true,
            (A3(_), A3(_)) => true,
            (A4(_), A4(_)) => true,
            (A5(_), A5(_)) => true,
            (A7(_), A7(_)) => true,
            (A10(_), A10(_)) => true,
            (A13(_), A13(_)) => true,
            (A14(_), A14(_)) => true,
            (A15(_), A15(_)) => true,
            (_, _) => false,
        }
    }

    fn reduce_alphas(&self, mask: &[bool]) -> Option<Self> {
        let self_vec = self.to_vec();
        assert_eq!(self_vec.len(), mask.len());
        let reduced_self_vec = self_vec
            .into_iter()
            .zip(mask.iter())
            .filter_map(|(alpha, keep)| if *keep { Some(alpha) } else { None })
            .collect::<Vec<_>>();
        Self::from_vec(reduced_self_vec)
    }

    fn remaining_ratios(&self, original_budget: &Self) -> RdpAlphasRatios {
        assert!(
            self.to_vec()
                .iter()
                .any(|x| *x >= 0. || x.approx_eq(0., F64Margin::default())),
            "To be a valid rdp budget, at least for one alpha value, the corresponding epsilon \
            value must be non-negative. But given was {:?}",
            self
        );
        let res: Vec<f64> = self
            .to_vec()
            .into_iter()
            .zip_eq(original_budget.to_vec().into_iter())
            .map(|(current_b, original_b)| f64::max(current_b / original_b, 0.))
            .collect();
        for ratio in res.iter() {
            assert!(
                (0. <= *ratio || (0.).approx_eq(*ratio, F64Margin::default()))
                    && (*ratio <= 1. || ratio.approx_eq(1., F64Margin::default())),
                "The budget ratios needs to be between 0 and 1, got {:?}",
                &res
            );
        }
        RdpAlphasRatios(A1([res.into_iter().reduce(f64::max).unwrap()]))
    }
}

impl PubRdpAccounting for RdpAlphas {
    fn to_vec(&self) -> Vec<f64> {
        match self {
            A1(arr) => arr.to_vec(),
            A2(arr) => arr.to_vec(),
            A3(arr) => arr.to_vec(),
            A4(arr) => arr.to_vec(),
            A5(arr) => arr.to_vec(),
            A7(arr) => arr.to_vec(),
            A10(arr) => arr.to_vec(),
            A13(arr) => arr.to_vec(),
            A14(arr) => arr.to_vec(),
            A15(arr) => arr.to_vec(),
        }
    }

    fn from_vec(vec: Vec<f64>) -> Option<Self> {
        match vec.len() {
            1 => Some(A1(vec.try_into().unwrap())),
            2 => Some(A2(vec.try_into().unwrap())),
            3 => Some(A3(vec.try_into().unwrap())),
            4 => Some(A4(vec.try_into().unwrap())),
            5 => Some(A5(vec.try_into().unwrap())),
            7 => Some(A7(vec.try_into().unwrap())),
            10 => Some(A10(vec.try_into().unwrap())),
            13 => Some(A13(vec.try_into().unwrap())),
            14 => Some(A14(vec.try_into().unwrap())),
            15 => Some(A15(vec.try_into().unwrap())),
            _ => None,
        }
    }

    fn get_value_for_alpha(&self, index: AlphaIndex) -> f64 {
        match self {
            A1(arr) => arr[index.0],
            A2(arr) => arr[index.0],
            A3(arr) => arr[index.0],
            A4(arr) => arr[index.0],
            A5(arr) => arr[index.0],
            A7(arr) => arr[index.0],
            A10(arr) => arr[index.0],
            A13(arr) => arr[index.0],
            A14(arr) => arr[index.0],
            A15(arr) => arr[index.0],
        }
    }
}

impl ApproxEq for &RdpAlphas {
    type Margin = F64Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        match (self, other) {
            (A1(v1), A1(v2)) => arr_approx_eq(v1, v2, margin),
            (A2(v1), A2(v2)) => arr_approx_eq(v1, v2, margin),
            (A3(v1), A3(v2)) => arr_approx_eq(v1, v2, margin),
            (A4(v1), A4(v2)) => arr_approx_eq(v1, v2, margin),
            (A5(v1), A5(v2)) => arr_approx_eq(v1, v2, margin),
            (A7(v1), A7(v2)) => arr_approx_eq(v1, v2, margin),
            (A10(v1), A10(v2)) => arr_approx_eq(v1, v2, margin),
            (A13(v1), A13(v2)) => arr_approx_eq(v1, v2, margin),
            (A14(v1), A14(v2)) => arr_approx_eq(v1, v2, margin),
            (A15(v1), A15(v2)) => arr_approx_eq(v1, v2, margin),
            (x, y) => {
                panic!("Comparing these RDP budgets is not supported (likely of different length): {:?} and {:?}", x, y)
            }
        }
    }
}

impl Add for RdpAlphas {
    type Output = RdpAlphas;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl Sub for RdpAlphas {
    type Output = RdpAlphas;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl Sub for &RdpAlphas {
    type Output = RdpAlphas;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (A1(v1), A1(v2)) => A1(element_wise_subtraction(v1, v2)),
            (A2(v1), A2(v2)) => A2(element_wise_subtraction(v1, v2)),
            (A3(v1), A3(v2)) => A3(element_wise_subtraction(v1, v2)),
            (A4(v1), A4(v2)) => A4(element_wise_subtraction(v1, v2)),
            (A5(v1), A5(v2)) => A5(element_wise_subtraction(v1, v2)),
            (A7(v1), A7(v2)) => A7(element_wise_subtraction(v1, v2)),
            (A10(v1), A10(v2)) => A10(element_wise_subtraction(v1, v2)),
            (A13(v1), A13(v2)) => A13(element_wise_subtraction(v1, v2)),
            (A14(v1), A14(v2)) => A14(element_wise_subtraction(v1, v2)),
            (A15(v1), A15(v2)) => A15(element_wise_subtraction(v1, v2)),
            (_, _) => {
                panic!("Subtracting these RDP budgets is not supported (likely of different length): {:?} and {:?}", self, rhs)
            }
        }
    }
}

impl Add for &RdpAlphas {
    type Output = RdpAlphas;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (A1(v1), A1(v2)) => A1(element_wise_addition(v1, v2)),
            (A2(v1), A2(v2)) => A2(element_wise_addition(v1, v2)),
            (A3(v1), A3(v2)) => A3(element_wise_addition(v1, v2)),
            (A4(v1), A4(v2)) => A4(element_wise_addition(v1, v2)),
            (A5(v1), A5(v2)) => A5(element_wise_addition(v1, v2)),
            (A7(v1), A7(v2)) => A7(element_wise_addition(v1, v2)),
            (A10(v1), A10(v2)) => A10(element_wise_addition(v1, v2)),
            (A13(v1), A13(v2)) => A13(element_wise_addition(v1, v2)),
            (A14(v1), A14(v2)) => A14(element_wise_addition(v1, v2)),
            (A15(v1), A15(v2)) => A15(element_wise_addition(v1, v2)),
            (_, _) => {
                panic!("Adding these RDP budgets is not supported (likely of different length): {:?} and {:?}", self, rhs)
            }
        }
    }
}

impl AddAssign<&RdpAlphas> for RdpAlphas {
    fn add_assign(&mut self, rhs: &RdpAlphas) {
        match (self, &rhs) {
            (A1(ref mut v1), A1(v2)) => element_wise_addition_in_place(v1, v2),
            (A2(ref mut v1), A2(v2)) => element_wise_addition_in_place(v1, v2),
            (A3(ref mut v1), A3(v2)) => element_wise_addition_in_place(v1, v2),
            (A4(ref mut v1), A4(v2)) => element_wise_addition_in_place(v1, v2),
            (A5(ref mut v1), A5(v2)) => element_wise_addition_in_place(v1, v2),
            (A7(ref mut v1), A7(v2)) => element_wise_addition_in_place(v1, v2),
            (A10(ref mut v1), A10(v2)) => element_wise_addition_in_place(v1, v2),
            (A13(ref mut v1), A13(v2)) => element_wise_addition_in_place(v1, v2),
            (A14(ref mut v1), A14(v2)) => element_wise_addition_in_place(v1, v2),
            (A15(ref mut v1), A15(v2)) => element_wise_addition_in_place(v1, v2),
            (x, y) => {
                panic!("Adding these RDP budgets is not supported (likely of different length): {:?} and {:?}", x, y)
            }
        }
    }
}

impl SubAssign<&RdpAlphas> for RdpAlphas {
    fn sub_assign(&mut self, rhs: &RdpAlphas) {
        match (self, &rhs) {
            (A1(ref mut v1), A1(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A2(ref mut v1), A2(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A3(ref mut v1), A3(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A4(ref mut v1), A4(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A5(ref mut v1), A5(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A7(ref mut v1), A7(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A10(ref mut v1), A10(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A13(ref mut v1), A13(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A14(ref mut v1), A14(v2)) => element_wise_subtraction_in_place(v1, v2),
            (A15(ref mut v1), A15(v2)) => element_wise_subtraction_in_place(v1, v2),
            (x, y) => {
                panic!("Adding these RDP budgets is not supported (likely of different length): {:?} and {:?}", x, y)
            }
        }
    }
}

// helper methods

fn rdp_in_budget<const LENGTH: usize>(
    budget: &[f64; LENGTH],
    cost: &[f64; LENGTH],
    margin: F64Margin,
) -> bool {
    budget
        .iter()
        .zip(cost.iter())
        .any(|(bi, ci)| bi > ci || bi.approx_eq(*ci, margin))
}

fn rdp_approx_le<const LENGTH: usize>(lhs: &[f64; LENGTH], rhs: &[f64; LENGTH]) -> bool {
    lhs.iter()
        .zip(rhs.iter())
        .all(|(l, r)| (l < r || l.approx_eq(*r, F64Margin::default())))
}

fn element_wise_min_in_place<const LENGTH: usize>(vec1: &mut [f64; LENGTH], vec2: &[f64; LENGTH]) {
    vec1.iter_mut()
        .zip(vec2)
        .for_each(|(e1, e2)| *e1 = e1.min(*e2));
}

fn element_wise_max_in_place<const LENGTH: usize>(vec1: &mut [f64; LENGTH], vec2: &[f64; LENGTH]) {
    vec1.iter_mut()
        .zip(vec2)
        .for_each(|(e1, e2)| *e1 = e1.max(*e2));
}

fn element_wise_addition<const LENGTH: usize>(
    vec1: &[f64; LENGTH],
    vec2: &[f64; LENGTH],
) -> [f64; LENGTH] {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(e1, e2)| e1 + e2)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn element_wise_subtraction<const LENGTH: usize>(
    vec1: &[f64; LENGTH],
    vec2: &[f64; LENGTH],
) -> [f64; LENGTH] {
    vec1.iter()
        .zip(vec2.iter())
        .map(|(e1, e2)| e1 - e2)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn element_wise_addition_in_place<const LENGTH: usize>(
    vec1: &mut [f64; LENGTH],
    vec2: &[f64; LENGTH],
) {
    vec1.iter_mut()
        .zip(vec2.iter())
        .for_each(|(e1, e2)| *e1 += e2);
}

fn element_wise_subtraction_in_place<const LENGTH: usize>(
    vec1: &mut [f64; LENGTH],
    vec2: &[f64; LENGTH],
) {
    vec1.iter_mut()
        .zip(vec2.iter())
        .for_each(|(e1, e2)| *e1 -= e2);
}

fn arr_approx_eq<const LENGTH: usize>(
    arr1: &[f64; LENGTH],
    arr2: &[f64; LENGTH],
    margin: F64Margin,
) -> bool {
    arr1.iter()
        .zip(arr2.iter())
        .all(|(e1, e2)| e1.approx_eq(*e2, margin))
}

fn rdp_arr_apply_func<const LENGTH: usize>(
    alphas: &[f64; LENGTH],
    arr: &mut [f64; LENGTH],
    f: &dyn Fn(f64, f64) -> f64,
) {
    alphas
        .iter()
        .zip(arr.iter_mut())
        .for_each(|(alpha, e)| *e = f(*alpha, *e));
}
