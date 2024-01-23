use crate::dprivacy::rdp_alphas_accounting::{RdpAlphas, RdpAlphasAccounting};
use crate::dprivacy::{AlphaReductionsResult, PerAlphaCostAndBudget, RdpAccounting};
use crate::{Accounting, AccountingType, AlphaReductions, Rdp};
use float_cmp::{ApproxEq, F64Margin};
use itertools::{Itertools, MultiProduct};
use log::warn;
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Contains the function needed to optimize rdp costs and budgets by eliminating eps_alpha values
/// that are unnecessary.
pub trait RdpOptimization: Sized {
    /// Takes as input some alpha values, and costs and a budget in terms of these alpha values.
    /// Returns which alpha values are actually needed and which are not, in the sense of that
    /// by only considering costs and budgets in terms of the returned alpha values, every possible
    /// allocation before is still possible afterwards. Which reductions are performed is controlled
    /// via the alpha_reductions argument (see [AlphaReductions] for more details).
    fn calc_needed_alphas(
        alphas: &Self,
        costs: &[Self],
        budget: &Self,
        alpha_reductions: AlphaReductions,
    ) -> (Vec<PerAlphaCostAndBudget>, AlphaReductionsResult);

    /// Using the given mask, reduced the number of rdp eps-values. If the accounting type is not
    /// rdp, this method will panic.
    fn reduce_alphas(&self, mask: &[bool]) -> Self;
}

impl RdpOptimization for AccountingType {
    fn calc_needed_alphas(
        alphas: &AccountingType,
        costs: &[AccountingType],
        budget: &AccountingType,
        alpha_reductions: AlphaReductions,
    ) -> (Vec<PerAlphaCostAndBudget>, AlphaReductionsResult) {
        assert!(alphas.is_rdp(), "Alphas need to be rdp");
        assert!(
            alphas.check_same_type(budget),
            "Budget and alphas are not compatible"
        );
        assert!(
            costs.iter().all(|cost| cost.check_same_type(alphas)),
            "costs and alphas are not compatible"
        );

        assert!(
            costs
                .iter()
                .all(|cost| cost.get_rdp_vec().iter().all(|c| *c > 0.0)),
            "any cost for any alpha must be > 0"
        );

        assert_eq!(
            HashSet::<u64>::from_iter(
                alphas
                    .get_rdp_vec()
                    .into_iter()
                    .map(|alpha| alpha.to_bits())
            )
            .len(),
            alphas.get_rdp_vec().len(),
            "Alphas need to be distinct"
        );

        let alpha_vec = alphas.get_rdp_vec();
        let cost_vecs = costs
            .iter()
            .map(|cost| cost.get_rdp_vec())
            .collect::<Vec<_>>();
        let budget_vec = budget.get_rdp_vec();

        let mut per_alpha_costs_and_budgets: Vec<PerAlphaCostAndBudget> = (0..alpha_vec.len())
            .map(|i| PerAlphaCostAndBudget {
                alpha: alpha_vec[i],
                costs: cost_vecs.iter().map(|cost| cost[i]).collect(),
                budget: budget_vec[i],
            })
            .collect();

        let mut n_starting_alphas = per_alpha_costs_and_budgets.len();
        let mut budget_reduction_alphas = None;
        let mut ratio_reduction_alphas = None;
        let mut combinatorial_reduction_alphas = None;
        if alpha_reductions.budget_reduction {
            RdpAlphas::calc_needed_alphas_budget(&mut per_alpha_costs_and_budgets);
            budget_reduction_alphas = Some(n_starting_alphas - per_alpha_costs_and_budgets.len());
            n_starting_alphas = per_alpha_costs_and_budgets.len();
        }

        if alpha_reductions.ratio_reduction {
            RdpAlphas::calc_needed_alphas_ratios(&mut per_alpha_costs_and_budgets);
            ratio_reduction_alphas = Some(n_starting_alphas - per_alpha_costs_and_budgets.len());
            n_starting_alphas = per_alpha_costs_and_budgets.len();
        }

        if alpha_reductions.combinatorial_reduction {
            if costs.len() >= 15 {
                warn!(
                    "Combinatorial reduction of the number of alphas is exponential in the number \
                    of costs - since the number of costs is {}, this might take a while",
                    costs.len()
                )
            }
            RdpAlphas::calc_needed_alphas_combinatorial(&mut per_alpha_costs_and_budgets);
            combinatorial_reduction_alphas =
                Some(n_starting_alphas - per_alpha_costs_and_budgets.len());
        }

        (
            per_alpha_costs_and_budgets,
            AlphaReductionsResult {
                budget_reduction: budget_reduction_alphas,
                ratio_reduction: ratio_reduction_alphas,
                combinatorial_reduction: combinatorial_reduction_alphas,
            },
        )
    }

    fn reduce_alphas(&self, mask: &[bool]) -> Self {
        if let Rdp { eps_values } = self {
            Rdp {
                eps_values: RdpAlphas::reduce_alphas(eps_values, mask)
                    .expect("Reducing alphas failed"),
            }
        } else {
            panic!("Tried to reduce a non-rdp accounting type")
        }
    }
}

impl RdpAlphas {
    /// calculated the needed alphas based open if the budget is positive.
    fn calc_needed_alphas_budget(per_alpha_costs_and_budgets: &mut Vec<PerAlphaCostAndBudget>) {
        per_alpha_costs_and_budgets
            .retain(|x| x.budget > 0.0 && !x.budget.approx_eq(0.0, F64Margin::default()))
    }

    /// We reduce the number of needed alphas according to the cost/budget ratios as described
    /// in the paper.
    fn calc_needed_alphas_ratios(per_alpha_costs_and_budgets: &mut Vec<PerAlphaCostAndBudget>) {
        assert!(
            per_alpha_costs_and_budgets.iter().all(|x| x.budget > 0.0),
            "To reduce the needed alpha with ratios, all budgets must be > 0"
        );

        impl PerAlphaCostAndBudget {
            // checks if an alpha value (self) is unnecessary according to the cost/budget ratio, if
            // the other alpha value is present.
            fn is_unnecessary_ratio(&self, other: &Self) -> bool {
                let self_ratios = self.costs.iter().map(|c_j| c_j / self.budget);
                let other_ratios = other.costs.iter().map(|c_j| c_j / other.budget);
                self_ratios
                    .zip_eq(other_ratios)
                    .all(|(self_ratio, other_ratio)| {
                        self_ratio > other_ratio
                            || self_ratio.approx_eq(other_ratio, F64Margin::default())
                    })
            }
        }

        let mut needed_alphas = Vec::new();
        while !per_alpha_costs_and_budgets.is_empty() {
            let next_candidate = per_alpha_costs_and_budgets.pop().unwrap();

            let is_unnecessary = per_alpha_costs_and_budgets
                .iter()
                .chain(needed_alphas.iter())
                .any(|other| next_candidate.is_unnecessary_ratio(other));

            if !is_unnecessary {
                needed_alphas.push(next_candidate);
            }
        }

        per_alpha_costs_and_budgets.extend(needed_alphas.into_iter());
    }

    /// We reduce the number of needed alphas by checking if an alpha enables additional assignment
    /// possibilities compared to an other alpha. More expensive than the other two methods, but
    /// should reduce the number of alphas as far as possible before allocation. Should only be
    /// called for a sufficiently small number of costs, as the runtime
    /// is exponential in the number of costs.
    fn calc_needed_alphas_combinatorial(
        per_alpha_costs_and_budgets: &mut Vec<PerAlphaCostAndBudget>,
    ) {
        if per_alpha_costs_and_budgets.is_empty() {
            return;
        }
        // eventually, this contains only mappings from alpha value to alpha value where the first
        // alpha value is unnecessary if the second alpha value is present.
        let mut unnecessary_alphas_if: BTreeMap<u64, HashSet<u64>> = per_alpha_costs_and_budgets
            .iter()
            .map(|pacb| {
                (
                    pacb.alpha.to_bits(),
                    per_alpha_costs_and_budgets
                        .iter()
                        .filter_map(|pacb2| {
                            if pacb2.alpha != pacb.alpha {
                                Some(pacb2.alpha.to_bits())
                            } else {
                                None
                            }
                        })
                        .collect::<HashSet<u64>>(),
                )
            })
            .collect();

        let n_costs = per_alpha_costs_and_budgets[0].costs.len();
        // contains how many allocations can be made at most with requests with just this cost over
        // all alpha values
        let max_individual_allocations = (0..n_costs).map(|i| {
            per_alpha_costs_and_budgets
                .iter()
                .map(|pacb| (pacb.budget / pacb.costs[i]).floor() as usize)
                .reduce(usize::max)
                .expect("Could not reduce number of individual allocations")
        });
        let possible_allocations: MultiProduct<std::ops::RangeInclusive<usize>> =
            max_individual_allocations
                .map(|i: usize| 0usize..=i)
                .multi_cartesian_product();
        for possible_allocation in possible_allocations {
            // calculate for which alphas this allocation is possible
            let mask: BTreeSet<u64> = per_alpha_costs_and_budgets
                .iter()
                .filter_map(|pacb| {
                    let total_cost: f64 = pacb
                        .costs
                        .iter()
                        .zip(possible_allocation.iter())
                        .map(|(cost, times)| *cost * (*times as f64))
                        .sum();

                    if total_cost <= pacb.budget
                        || total_cost.approx_eq(pacb.budget, F64Margin::default())
                    {
                        Some(pacb.alpha.to_bits())
                    } else {
                        None
                    }
                })
                .collect();
            // if this allocation is possible for alpha_1 but not for alpha_2, need to remove
            // alpha_2 from unnecessary_alphas_if[alpha_1]
            for alpha_1 in mask.iter() {
                for alpha_2 in per_alpha_costs_and_budgets.iter().map(|pacb| pacb.alpha) {
                    if !mask.contains(&alpha_2.to_bits()) {
                        unnecessary_alphas_if
                            .get_mut(alpha_1)
                            .unwrap()
                            .remove(&alpha_2.to_bits());
                    }
                }
            }
        }

        // now, we can iteratively remove any alpha_1, where unnecessary_alphas_if[alpha_1] is not
        // empty, since that means that alpha_1 is unnecessary due to the presence of the entry in
        // unnecessary_alphas_if[alpha_1]
        'outer: loop {
            for alpha in per_alpha_costs_and_budgets
                .iter()
                .map(|pacb| pacb.alpha.to_bits())
            {
                if unnecessary_alphas_if.contains_key(&alpha)
                    && !unnecessary_alphas_if[&alpha].is_empty()
                {
                    unnecessary_alphas_if.remove(&alpha);
                    for vals in unnecessary_alphas_if.values_mut() {
                        vals.remove(&alpha);
                    }
                    continue 'outer;
                }
            }
            break;
        }

        per_alpha_costs_and_budgets
            .retain(|pacb| unnecessary_alphas_if.contains_key(&pacb.alpha.to_bits()));
    }
}

#[cfg(test)]
mod tests {
    use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
    use crate::dprivacy::rdpopt::RdpOptimization;
    use crate::{AccountingType, AlphaReductions, Rdp};
    use float_cmp::Ulps;

    #[test]
    fn test_calc_needed_alphas_combinatorial() {
        let alphas = Rdp {
            eps_values: A2([2., 3.]),
        };

        let budget = Rdp {
            eps_values: A2([1.0, 1.0]),
        };

        let costs = [
            Rdp {
                eps_values: A2([0.6, 0.55]),
            },
            Rdp {
                eps_values: A2([0.55, 0.6]),
            },
        ];

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: false,
                    ratio_reduction: false,
                    combinatorial_reduction: true
                }
            )
            .0
            .len(),
            1
        );

        let alphas = Rdp {
            eps_values: A2([10., 20.]),
        };

        let costs = [
            Rdp {
                eps_values: A2([0.6, 0.45]),
            },
            Rdp {
                eps_values: A2([0.45, 0.6]),
            },
        ];

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: false,
                    ratio_reduction: false,
                    combinatorial_reduction: true
                }
            )
            .0
            .len(),
            2
        );

        let alphas = Rdp {
            eps_values: A3([2., 324., 4143245.]),
        };

        let budget = Rdp {
            eps_values: A3([1.0, 1.0, 1.0]),
        };

        let costs = [
            Rdp {
                eps_values: A3([0.6, 0.45, 0.5]),
            },
            Rdp {
                eps_values: A3([0.45, 0.6, 0.55]),
            },
        ];

        let res = AccountingType::calc_needed_alphas(
            &alphas,
            &costs,
            &budget,
            AlphaReductions {
                budget_reduction: false,
                ratio_reduction: true,
                combinatorial_reduction: true,
            },
        );
        assert_eq!(res.0.len(), 2);
        assert!(res.0.iter().any(|pacb| pacb.alpha == 2.0));
    }

    #[test]
    fn test_calc_needed_alphas_ratio() {
        let alphas = Rdp {
            eps_values: A2([1.5, 2.0]),
        };

        let budget = Rdp {
            eps_values: A2([0.8, 1.0]),
        };

        let costs = [
            Rdp {
                eps_values: A2([0.45, 0.5]),
            },
            Rdp {
                eps_values: A2([0.48, 0.53]),
            },
        ];

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: false,
                    ratio_reduction: true,
                    combinatorial_reduction: false
                }
            )
            .0
            .into_iter()
            .map(|pacb| pacb.alpha)
            .collect::<Vec<f64>>(),
            vec![2.0]
        );

        let budget = Rdp {
            eps_values: A2([0.999999, 1.0]),
        };

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: false,
                    ratio_reduction: true,
                    combinatorial_reduction: false
                }
            )
            .0
            .into_iter()
            .map(|pacb| pacb.alpha)
            .collect::<Vec<f64>>(),
            vec![1.5]
        );

        let budget = Rdp {
            eps_values: A2([1.0, 1.0]),
        };

        let costs = [
            Rdp {
                eps_values: A2([0.45, 0.45]),
            },
            Rdp {
                eps_values: A2([0.53, 0.53]),
            },
        ];

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: false,
                    ratio_reduction: true,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            1
        );

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: false,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            2
        );

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: true,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            2
        );

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: false,
                    ratio_reduction: true,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            1
        );

        let budget = Rdp {
            eps_values: A2([0.0, -1.0]),
        };

        assert!(AccountingType::calc_needed_alphas(
            &alphas,
            &costs,
            &budget,
            AlphaReductions {
                budget_reduction: true,
                ratio_reduction: true,
                combinatorial_reduction: false
            }
        )
        .0
        .is_empty());
    }

    #[test]
    #[should_panic(expected = "To reduce the needed alpha with ratios, all budgets must be > 0")]
    fn test_calc_needed_alphas_should_panic_ratio() {
        let alphas = Rdp {
            eps_values: A2([2., 3.]),
        };

        let budget = Rdp {
            eps_values: A2([3.0, 0.0]),
        };

        let costs = [
            Rdp {
                eps_values: A2([0.5, 0.7]),
            },
            Rdp {
                eps_values: A2([0.7, 0.5]),
            },
        ];

        AccountingType::calc_needed_alphas(
            &alphas,
            &costs,
            &budget,
            AlphaReductions {
                budget_reduction: false,
                ratio_reduction: true,
                combinatorial_reduction: false,
            },
        );
    }

    #[test]
    #[should_panic(expected = "Alphas need to be distinct")]
    fn test_calc_needed_alphas_should_panic() {
        let alphas = Rdp {
            eps_values: A2([2.0, 2.0]),
        };

        let costs = [
            Rdp {
                eps_values: A2([0.45, 0.5]),
            },
            Rdp {
                eps_values: A2([0.48, 0.53]),
            },
        ];

        let budget = Rdp {
            eps_values: A2([0.8, 1.0]),
        };

        AccountingType::calc_needed_alphas(
            &alphas,
            &costs,
            &budget,
            AlphaReductions {
                budget_reduction: true,
                ratio_reduction: true,
                combinatorial_reduction: true,
            },
        );
    }

    #[test]
    fn test_calc_needed_alphas_budget() {
        let alphas = Rdp {
            eps_values: A2([1.5, 2.0]),
        };

        let costs = [
            Rdp {
                eps_values: A2([0.45, 0.5]),
            },
            Rdp {
                eps_values: A2([0.48, 0.53]),
            },
        ];

        let budget = Rdp {
            eps_values: A2([0.8, 1.0]),
        };

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: true,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            2
        );

        let budget = Rdp {
            eps_values: A2([0.0, 1.0]),
        };

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: true,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            1
        );

        let budget = Rdp {
            eps_values: A2([-0.0, 1.0]),
        };

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: true,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            1
        );

        let budget = Rdp {
            eps_values: A2([-0.1, 1.0]),
        };

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: true,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            1
        );

        let budget = Rdp {
            eps_values: A2([-0.1, 0.0]),
        };

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: true,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            0
        );

        let budget = Rdp {
            eps_values: A2([-0.1, 0.0.next()]),
        };

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: true,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            0
        );

        let budget = Rdp {
            eps_values: A2([0.8, 1.0]),
        };

        let costs = [
            Rdp {
                eps_values: A2([0.45, 0.5]),
            },
            Rdp {
                eps_values: A2([0.48, 0.53]),
            },
        ];

        assert_eq!(
            AccountingType::calc_needed_alphas(
                &alphas,
                &costs,
                &budget,
                AlphaReductions {
                    budget_reduction: true,
                    ratio_reduction: false,
                    combinatorial_reduction: false
                }
            )
            .0
            .len(),
            2
        );
    }
}
