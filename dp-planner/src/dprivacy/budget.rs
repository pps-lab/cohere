use crate::dprivacy::budget::BudgetType::{DpAdpBudget, RdpBudget};
use crate::dprivacy::{Accounting, AccountingType, RdpAccounting};

use super::{EpsDeltaDp, EpsDp, Rdp};
use float_cmp::{ApproxEq, F64Margin};
use std::fmt::{self, Debug};

pub trait SegmentBudget: Debug + fmt::Display + Send + Sync + Clone {
    /// Returns a new budget without any constraints yet
    fn new() -> Self;

    /// Returns a new budget with the initial budget as constraint.
    /// Same as calling new(), and then add_budget_constraint()
    fn new_with_budget_constraint(initial_budget_constraint: &AccountingType) -> Self;

    /// Returns a new budget with several constraints
    fn new_with_budget_constraints(budget_constraints: Vec<&AccountingType>) -> Self;

    /// Adds a constraint to the budget
    fn add_budget_constraint(&mut self, other_budget: &AccountingType) -> &mut Self;

    /// checks if the budget is sufficient for the given cost
    fn is_budget_sufficient(&self, cost: &AccountingType) -> bool;

    /// get all constraints currently part of the merger
    fn get_budget_constraints(&self) -> Vec<&AccountingType>;

    /// Set budget constraints: For EpsDp and EpsDeltaDp, only a single constraint is accepted
    /// while for Rdp, the OptimalMerger may have multiple constraints.
    fn set_budget_constraints(&mut self, budget_constraints: Vec<&AccountingType>) -> &mut Self;

    /// Merge two budgets, meaning all constraints are merged
    fn merge_assign(&mut self, other: &Self) -> &mut Self {
        for budget in other.get_budget_constraints() {
            self.add_budget_constraint(budget);
        }
        self
    }

    fn approx_eq(&self, other: &Self, margin: F64Margin) -> bool {
        let b1 = self.get_budget_constraints();
        let b2 = other.get_budget_constraints();
        // due to numerical instability, one budget having more constraints than the other might actually be ok
        // just want that each constraint in one budget is ca. equal to a constraint in the other
        let b1_in_b2: bool = b1.iter().all(|x| b2.iter().any(|y| x.approx_eq(y, margin)));
        let b2_in_b1: bool = b2.iter().all(|x| b1.iter().any(|y| x.approx_eq(y, margin)));
        b1_in_b2 && b2_in_b1
    }

    // fn info_for_building_ilp(&self) -> X;
    // fn subtract_cost(&mut self, cost: &AccountingType);
    fn subtract_cost(&mut self, cost: &AccountingType);

    /// apply a function to eps / eps and delta / each eps_alpha individually
    fn apply_func(&mut self, f: &dyn Fn(f64) -> f64) {
        // TODO [later]: Implement directly instead of via getter and setter
        let mut new_constraints = self
            .get_budget_constraints()
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        for constraint in new_constraints.iter_mut() {
            constraint.apply_func(f);
        }
        self.set_budget_constraints(new_constraints.iter().collect());
    }

    fn is_rdp(&self) -> bool;

    /// Returns whether the accounting type is a1, i.e., rdp with only a single alpha value.
    /// Used in conjunction with [Self::a1_to_eps_dp]
    fn is_a1(&self) -> bool {
        self.get_budget_constraints().iter().all(|x| x.is_a1())
    }

    /// Tried to converted the budget to eps dp, which will only succeed if the current budget is
    /// A1 (else panics). See [AccountingType::a1_to_eps_dp] for more information.
    fn a1_to_eps_dp(&self) -> Self;
}

#[derive(Debug, Clone, PartialEq)]
enum BudgetType {
    // if we have RDP, we need different data structures than with standard DP or ADP
    DpAdpBudget(AccountingType),
    RdpBudget(Vec<AccountingType>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimalBudget {
    budget: Option<BudgetType>, // None if not initialized, Some(_) if initialized
}

impl fmt::Display for OptimalBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.budget {
            Some(DpAdpBudget(budget)) => write!(f, "{}", budget),
            Some(RdpBudget(budgets)) => write!(f, "{:?}", budgets),
            None => write!(f, ""),
        }
    }
}

impl SegmentBudget for OptimalBudget {
    fn new() -> Self {
        OptimalBudget { budget: None }
    }

    fn new_with_budget_constraint(initial_budget_constraint: &AccountingType) -> Self {
        match initial_budget_constraint {
            EpsDp { .. } => OptimalBudget {
                budget: Some(DpAdpBudget(initial_budget_constraint.clone())),
            },
            EpsDeltaDp { .. } => OptimalBudget {
                budget: Some(DpAdpBudget(initial_budget_constraint.clone())),
            },
            Rdp { .. } => OptimalBudget {
                budget: Some(RdpBudget(vec![initial_budget_constraint.clone()])),
            },
        }
    }

    fn new_with_budget_constraints(budget_constraints: Vec<&AccountingType>) -> Self {
        let mut res = Self::new();
        for budget in budget_constraints {
            res.add_budget_constraint(budget);
        }
        res
    }

    fn add_budget_constraint(&mut self, new_constraint: &AccountingType) -> &mut Self {
        match &mut self.budget {
            None => {
                self.budget = Self::new_with_budget_constraint(new_constraint).budget;
            }
            Some(ref mut budget_type) => match budget_type {
                DpAdpBudget(budget) => {
                    budget.min_assign(new_constraint);
                }
                RdpBudget(ref mut budgets) => {
                    let mut append_candidate = true; // if true, want to append candidate, else not
                    let candidate = new_constraint; // just a renaming
                    budgets.retain(|x| {
                        // Note: Retain can safely modify external state
                        let replace_x = candidate.approx_le(x); // Since candidate <= x, we don't need x if we have the candidate
                        let ignore_candidate = x.approx_le(candidate); // since x <= candidate, don't need candidate if we have x
                                                                       // Invariant: If append_candidate == false, an x where x <= candidate is retained
                                                                       // If append_candidate == true at the end, candidate will be appended.
                                                                       // Other than this one x, all x' where candidate <= x' are dropped (note: consider x' == candidate)
                        match (ignore_candidate, append_candidate) {
                            (true, true) => {
                                // can only happen once
                                append_candidate = false;
                                true // keep x, since we drop the candidate
                            }
                            (_, _) => !replace_x,
                            // if replace_x == true, then x <= candidate
                            // and our invariance guarantees that either the candidate or
                            // an x' with x' <= candidate is retained, so can safely remove x
                            // Note: Might not necessarily have x' <= x, due to numerical inaccuracies,
                            // even when x' <= candidate && candidate <= x
                            // though logically x' almost implies x, so is safe to remove.
                        }
                    });

                    if append_candidate {
                        budgets.push(candidate.clone());
                    }
                }
            },
        }
        self
    }

    fn is_budget_sufficient(&self, cost: &AccountingType) -> bool {
        // check if cost satisfies each constraint
        self.get_budget_constraints()
            .iter()
            .all(|budget| budget.in_budget(cost))
    }

    fn get_budget_constraints(&self) -> Vec<&AccountingType> {
        match &self.budget {
            None => Vec::new(),
            Some(budget_type) => match budget_type {
                DpAdpBudget(budget) => {
                    vec![budget]
                }
                RdpBudget(budgets) => budgets.iter().collect(),
            },
        }
    }

    fn set_budget_constraints(&mut self, budget_constraints: Vec<&AccountingType>) -> &mut Self {
        self.budget = Self::new_with_budget_constraints(budget_constraints).budget;
        self
    }

    fn subtract_cost(&mut self, cost: &AccountingType) {
        match &mut self.budget {
            None => panic!("no budget constraint set"),
            Some(DpAdpBudget(budget)) => {
                *budget -= cost;
                assert!(
                    budget.is_valid(),
                    "budget {:?} is invalid after subtraction (epsilon and/or delta < 0)",
                    budget
                );
            }
            Some(RdpBudget(budgets)) => budgets.iter_mut().for_each(|budget| {
                *budget -= cost;
                assert!(
                    budget.is_valid(),
                    "budget is invalid after subtraction (all epsilon < 0)"
                );
            }),
        }
    }

    fn is_rdp(&self) -> bool {
        match self
            .budget
            .as_ref()
            .expect("Called is_rdp on budget without constraint")
        {
            DpAdpBudget(_) => false,
            RdpBudget(_) => true,
        }
    }

    fn a1_to_eps_dp(&self) -> Self {
        let new_constraints = self
            .get_budget_constraints()
            .iter()
            .map(|x| x.a1_to_eps_dp())
            .collect::<Vec<_>>();
        Self::new_with_budget_constraints(new_constraints.iter().collect())
    }
}

#[derive(Debug, Clone)]
pub struct RdpMinBudget {
    budget: Option<AccountingType>,
}

impl fmt::Display for RdpMinBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.budget {
            Some(budget) => write!(f, "rdp={}", budget),
            None => write!(f, ""),
        }
    }
}

impl SegmentBudget for RdpMinBudget {
    fn new() -> Self {
        RdpMinBudget { budget: None }
    }

    fn new_with_budget_constraint(initial_budget_constraint: &AccountingType) -> Self {
        if let Rdp { eps_values: _ } = initial_budget_constraint {
            RdpMinBudget {
                budget: Some(initial_budget_constraint.clone()),
            }
        } else {
            panic!(
                "RdpMinBudget only works with rdp, but constraint {:?} given ",
                initial_budget_constraint
            )
        }
    }

    fn new_with_budget_constraints(budget_constraints: Vec<&AccountingType>) -> Self {
        assert!(!budget_constraints.is_empty());
        match budget_constraints[0] {
            Rdp { .. } => {}
            _ => {
                panic!(
                    "RdpMinBudget only works with rdp, but constraint {:?} given ",
                    budget_constraints[0]
                )
            }
        }
        let mut res = Self::new();
        for constraint in budget_constraints {
            res.add_budget_constraint(constraint);
        }
        res
    }

    fn add_budget_constraint(&mut self, other_budget: &AccountingType) -> &mut Self {
        match other_budget {
            Rdp { .. } => {}
            _ => {
                panic!(
                    "RdpMinBudget only works with rdp, but constraint {:?} given ",
                    other_budget
                )
            }
        }
        match self.budget {
            None => {
                self.budget = Some(other_budget.clone());
            }
            Some(ref mut budget) => {
                budget.min_assign(other_budget);
            }
        }
        self
    }

    fn is_budget_sufficient(&self, cost: &AccountingType) -> bool {
        match &self.budget {
            None => true,
            Some(budget) => budget.in_budget(cost),
        }
    }

    fn get_budget_constraints(&self) -> Vec<&AccountingType> {
        match &self.budget {
            None => Vec::new(),
            Some(budget) => {
                vec![budget]
            }
        }
    }

    fn set_budget_constraints(&mut self, budget_constraints: Vec<&AccountingType>) -> &mut Self {
        self.budget = Self::new_with_budget_constraints(budget_constraints).budget;
        self
    }

    fn subtract_cost(&mut self, cost: &AccountingType) {
        match &mut self.budget {
            None => panic!("no budget constraint set"),
            Some(budget) => {
                *budget -= cost;
                assert!(
                    budget.is_valid(),
                    "budget is invalid after subtraction (all epsilon < 0)"
                );
            }
        }
    }

    fn is_rdp(&self) -> bool {
        self.budget
            .as_ref()
            .expect("Called is_rdp on budget without constraint")
            .is_rdp()
    }

    fn a1_to_eps_dp(&self) -> Self {
        panic!("Cannot convert rdp min budget from a1 to eps dp, as rdp min budget only works with rdp")
    }
}

#[cfg(test)]
mod tests {
    use crate::dprivacy::budget::{OptimalBudget, RdpMinBudget, SegmentBudget};
    use crate::dprivacy::*;
    use float_cmp::F64Margin;

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

        let ob1 = OptimalBudget::new_with_budget_constraint(&a1);
        let ob2 = OptimalBudget::new_with_budget_constraint(&a2);
        let ob3 = OptimalBudget::new_with_budget_constraint(&a3);
        let ob4 = OptimalBudget::new_with_budget_constraint(&a4);

        let mb3 = RdpMinBudget::new_with_budget_constraint(&a3);
        let mb4 = RdpMinBudget::new_with_budget_constraint(&a4);

        assert!(!ob1.is_a1());
        assert!(!ob2.is_a1());
        assert!(ob3.is_a1());
        assert!(!ob4.is_a1());

        assert!(mb3.is_a1());
        assert!(!mb4.is_a1());

        ob3.a1_to_eps_dp();
    }

    #[test]
    #[should_panic(
        expected = "Cannot convert rdp min budget from a1 to eps dp, as rdp min budget only works with rdp"
    )]
    fn a1_to_eps_dp_panic() {
        let a4 = Rdp {
            eps_values: A5([1.0, 1.0, 1.0, 1.0, 1.0]),
        };
        let mb4 = RdpMinBudget::new_with_budget_constraint(&a4);
        mb4.a1_to_eps_dp();
    }

    #[test]
    fn test_optimal_budget() {
        let mut optimal_budget = OptimalBudget::new();

        let eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };
        let eps_dp_3 = EpsDp { eps: 0.5 };
        optimal_budget.add_budget_constraint(&eps_dp_2);
        assert!(optimal_budget.is_budget_sufficient(&eps_dp_2));
        assert!(optimal_budget.is_budget_sufficient(&eps_dp_1));
        optimal_budget.add_budget_constraint(&eps_dp_1);
        assert!(!optimal_budget.is_budget_sufficient(&eps_dp_2));
        assert!(optimal_budget.is_budget_sufficient(&eps_dp_1));
        assert!(!optimal_budget.is_rdp());

        optimal_budget.apply_func(&|x: f64| 0.5 * x);
        assert_eq!(optimal_budget.get_budget_constraints(), vec![&eps_dp_3]);

        let eps_delta_dp_1 = EpsDeltaDp {
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
        let eps_delta_dp_4 = EpsDeltaDp {
            eps: 0.5,
            delta: 0.05,
        };

        let mut optimal_budget = OptimalBudget::new();
        optimal_budget.add_budget_constraint(&eps_delta_dp_2);
        assert!(!optimal_budget.is_rdp());
        assert!(!optimal_budget.is_budget_sufficient(&eps_delta_dp_1));
        assert!(optimal_budget.is_budget_sufficient(&eps_delta_dp_2));
        assert!(optimal_budget.is_budget_sufficient(&eps_delta_dp_3));

        optimal_budget.add_budget_constraint(&eps_delta_dp_1);
        assert!(!optimal_budget.is_budget_sufficient(&eps_delta_dp_1));
        assert!(!optimal_budget.is_budget_sufficient(&eps_delta_dp_2));
        assert!(optimal_budget.is_budget_sufficient(&eps_delta_dp_3));

        optimal_budget.apply_func(&|x: f64| 0.5 * x);
        assert_eq!(
            optimal_budget.get_budget_constraints(),
            vec![&eps_delta_dp_4]
        );

        let rdp_1 = Rdp {
            eps_values: A5([1.0, 0.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, 5.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([6.0, 5.0, 6.0, 6.0, 5.0]),
        };
        let rdp_4 = Rdp {
            eps_values: A5([6.0, 5.0, 6.0, 6.0, 6.0]),
        };
        let rdp_5 = Rdp {
            eps_values: A5([0.5, 0.0, 1.5, 2.0, 2.5]),
        };
        let rdp_6 = Rdp {
            eps_values: A5([0.0, 2.5, 1.0, 1.5, 0.0]),
        };

        assert_eq!(rdp_1.get_rdp_vec(), vec![1.0, 0.0, 3.0, 4.0, 5.0]);
        let mut optimal_budget = OptimalBudget::new();
        optimal_budget.add_budget_constraint(&rdp_2);
        assert!(optimal_budget.is_rdp());
        assert!(optimal_budget.is_budget_sufficient(&rdp_1));
        assert!(optimal_budget.is_budget_sufficient(&rdp_2));
        assert!(optimal_budget.is_budget_sufficient(&rdp_3));
        assert!(optimal_budget.is_budget_sufficient(&rdp_4));

        optimal_budget.add_budget_constraint(&rdp_1);
        assert!(optimal_budget.is_budget_sufficient(&rdp_1));
        assert!(optimal_budget.is_budget_sufficient(&rdp_2));
        assert!(optimal_budget.is_budget_sufficient(&rdp_3));
        assert!(!optimal_budget.is_budget_sufficient(&rdp_4));

        let optimal_budget1 = OptimalBudget::new_with_budget_constraint(&rdp_1);
        let optimal_budget2 = OptimalBudget::new_with_budget_constraint(&rdp_1);
        let optimal_budget3 = OptimalBudget::new_with_budget_constraint(&rdp_2);
        assert!(optimal_budget1.approx_eq(&optimal_budget2, F64Margin::default()));
        assert!(!optimal_budget1.approx_eq(&optimal_budget3, F64Margin::default()));

        optimal_budget.apply_func(&|x: f64| 0.5 * x);
        let mut optimal_budget4 = OptimalBudget::new_with_budget_constraint(&rdp_5);
        optimal_budget4.add_budget_constraint(&rdp_6);
        assert!(
            optimal_budget.approx_eq(&optimal_budget4, F64Margin::default()),
            "left: {:?} right: {:?}",
            optimal_budget,
            optimal_budget4
        );
    }

    // TODO: Test Rdp Min Budget, and add tests for is_rdp()
    #[test]
    fn test_rdp_min_budget() {
        let mut min_budget = RdpMinBudget::new();

        let rdp_1 = Rdp {
            eps_values: A5([1.0, 0.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, 5.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([6.0, 5.0, 6.0, 6.0, 5.0]),
        };
        let rdp_4 = Rdp {
            eps_values: A5([6.0, 5.0, 6.0, 6.0, 6.0]),
        };
        let rdp_5 = Rdp {
            eps_values: A5([0.0, 0.0, 1.0, 1.5, 0.0]),
        };

        assert_eq!(rdp_1.get_rdp_vec(), vec![1.0, 0.0, 3.0, 4.0, 5.0]);

        min_budget.add_budget_constraint(&rdp_2);
        assert!(min_budget.is_rdp());

        // same as optimal budget, as only a single constraint added so far
        assert!(min_budget.is_budget_sufficient(&rdp_1));
        assert!(min_budget.is_budget_sufficient(&rdp_2));
        assert!(min_budget.is_budget_sufficient(&rdp_3));
        assert!(min_budget.is_budget_sufficient(&rdp_4));

        min_budget.add_budget_constraint(&rdp_1);

        // should be [0.0, 0.0, 2.0, 3.0, 0.0] in total
        assert_eq!(
            min_budget.get_budget_constraints(),
            vec![&Rdp {
                eps_values: A5([0.0, 0.0, 2.0, 3.0, 0.0])
            }]
        );

        assert!(min_budget.is_budget_sufficient(&rdp_1));
        assert!(min_budget.is_budget_sufficient(&rdp_2));
        assert!(!min_budget.is_budget_sufficient(&rdp_3));
        assert!(!min_budget.is_budget_sufficient(&rdp_4));

        let min_budget1 = RdpMinBudget::new_with_budget_constraint(&rdp_1);
        let min_budget2 = RdpMinBudget::new_with_budget_constraint(&rdp_1);
        let min_budget3 = RdpMinBudget::new_with_budget_constraint(&rdp_2);
        assert!(min_budget1.approx_eq(&min_budget2, F64Margin::default()));
        assert!(!min_budget1.approx_eq(&min_budget3, F64Margin::default()));

        min_budget.apply_func(&|x: f64| 0.5 * x);
        let min_budget4 = RdpMinBudget::new_with_budget_constraint(&rdp_5);
        assert!(
            min_budget.approx_eq(&min_budget4, F64Margin::default()),
            "left: {:?} right: {:?}",
            min_budget,
            min_budget4
        );
    }
    /*
    #[test]
    fn test_min_merger() {
        let min_merger = MinMergerOld::new();
        assert_eq!(min_merger.get(), Vec::<&AccountingType>::new());

        let eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&eps_dp_2);
        assert_eq!(min_merger.get(), vec![&eps_dp_2.clone()]);

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&eps_dp_1);
        min_merger.merge_assign(&eps_dp_2);
        assert_eq!(min_merger.get(), vec![&eps_dp_1.clone()]);

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&eps_dp_2);
        min_merger.merge_assign(&eps_dp_1);
        assert_eq!(min_merger.get(), vec![&eps_dp_1.clone()]);

        let eps_delta_dp_1 = EpsDeltaDp {
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

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&eps_delta_dp_1);
        assert_eq!(min_merger.get(), vec![&eps_delta_dp_1.clone()]);

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&eps_delta_dp_1);
        min_merger.merge_assign(&eps_delta_dp_2);
        assert_eq!(min_merger.get(), vec![&eps_delta_dp_3.clone()]);

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&eps_delta_dp_2);
        min_merger.merge_assign(&eps_delta_dp_1);
        assert_eq!(min_merger.get(), vec![&eps_delta_dp_3.clone()]);

        let rdp_1 = Rdp {
            eps_values: A5([1.0, 0.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, 5.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([0.0, 0.0, 2.0, 3.0, 0.0]),
        };

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&rdp_1);
        assert_eq!(min_merger.get(), vec![&rdp_1.clone()]);

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&rdp_1);
        min_merger.merge_assign(&rdp_2);
        assert_eq!(min_merger.get(), vec![&rdp_3.clone()]);

        let mut min_merger = MinMergerOld::new();
        min_merger.merge_assign(&rdp_2);
        min_merger.merge_assign(&rdp_1);
        assert_eq!(min_merger.get(), vec![&rdp_3.clone()]);
    }

    #[test]
    fn test_full_merger() {
        let optimal_merger = OptimalMergerOld::new();
        assert_eq!(optimal_merger.get(), Vec::<&AccountingType>::new());

        let eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&eps_dp_2);
        assert_eq!(optimal_merger.get(), vec![&eps_dp_2.clone()]);

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&eps_dp_1);
        optimal_merger.merge_assign(&eps_dp_2);
        assert_eq!(optimal_merger.get(), vec![&eps_dp_1.clone()]);

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&eps_dp_2);
        optimal_merger.merge_assign(&eps_dp_1);
        assert_eq!(optimal_merger.get(), vec![&eps_dp_1.clone()]);

        let eps_delta_dp_1 = EpsDeltaDp {
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

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&eps_delta_dp_1);
        assert_eq!(optimal_merger.get(), vec![&eps_delta_dp_1.clone()]);

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&eps_delta_dp_1);
        optimal_merger.merge_assign(&eps_delta_dp_2);
        assert_eq!(optimal_merger.get(), vec![&eps_delta_dp_3.clone()]);

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&eps_delta_dp_2);
        optimal_merger.merge_assign(&eps_delta_dp_1);
        assert_eq!(optimal_merger.get(), vec![&eps_delta_dp_3.clone()]);

        let rdp_1 = Rdp {
            eps_values: A5([1.0, 0.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, 5.0, 2.0, 3.0, 0.0]),
        };

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&rdp_1);
        assert_eq!(optimal_merger.get(), vec![&rdp_1.clone()]);

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&rdp_1);
        optimal_merger.merge_assign(&rdp_2);
        let res = optimal_merger.get();
        assert!(
            res == vec![&rdp_2.clone(), &rdp_1.clone()]
                || res == vec![&rdp_1.clone(), &rdp_2.clone()]
        );

        let mut optimal_merger = OptimalMergerOld::new();
        optimal_merger.merge_assign(&rdp_2);
        optimal_merger.merge_assign(&rdp_1);
        let res = optimal_merger.get();
        assert!(
            res == vec![&rdp_2.clone(), &rdp_1.clone()]
                || res == vec![&rdp_1.clone(), &rdp_2.clone()]
        );
    }

    #[test]
    #[should_panic]
    fn test_optimal_budget_empty_panic() {
        let optimal_budget = OptimalBudgetOld::new();
        let eps_dp_1 = EpsDp { eps: 1.0 };
        optimal_budget.is_budget_sufficient(&eps_dp_1);
    }

    #[test]
    fn test_optimal_budget() {
        let mut optimal_budget = OptimalBudgetOld::new();

        let eps_dp_1 = EpsDp { eps: 1.0 };
        let eps_dp_2 = EpsDp { eps: 1.5 };
        optimal_budget.add_budget_constraint(&eps_dp_2);
        assert!(optimal_budget.is_budget_sufficient(&eps_dp_2));
        assert!(optimal_budget.is_budget_sufficient(&eps_dp_1));
        optimal_budget.add_budget_constraint(&eps_dp_1);
        assert!(!optimal_budget.is_budget_sufficient(&eps_dp_2));
        assert!(optimal_budget.is_budget_sufficient(&eps_dp_1));

        let eps_delta_dp_1 = EpsDeltaDp {
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

        let mut optimal_budget = OptimalBudgetOld::new();
        optimal_budget.add_budget_constraint(&eps_delta_dp_2);
        assert!(!optimal_budget.is_budget_sufficient(&eps_delta_dp_1));
        assert!(optimal_budget.is_budget_sufficient(&eps_delta_dp_2));
        assert!(optimal_budget.is_budget_sufficient(&eps_delta_dp_3));

        optimal_budget.add_budget_constraint(&eps_delta_dp_1);
        assert!(!optimal_budget.is_budget_sufficient(&eps_delta_dp_1));
        assert!(!optimal_budget.is_budget_sufficient(&eps_delta_dp_2));
        assert!(optimal_budget.is_budget_sufficient(&eps_delta_dp_3));

        let rdp_1 = Rdp {
            eps_values: A5([1.0, 0.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, 5.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([6.0, 5.0, 6.0, 6.0, 5.0]),
        };
        let rdp_4 = Rdp {
            eps_values: A5([6.0, 5.0, 6.0, 6.0, 6.0]),
        };

        let mut optimal_budget = OptimalBudgetOld::new();
        optimal_budget.add_budget_constraint(&rdp_2);
        assert!(optimal_budget.is_budget_sufficient(&rdp_1));
        assert!(optimal_budget.is_budget_sufficient(&rdp_2));
        assert!(optimal_budget.is_budget_sufficient(&rdp_3));
        assert!(optimal_budget.is_budget_sufficient(&rdp_4));

        optimal_budget.add_budget_constraint(&rdp_1);
        assert!(optimal_budget.is_budget_sufficient(&rdp_1));
        assert!(optimal_budget.is_budget_sufficient(&rdp_2));
        assert!(optimal_budget.is_budget_sufficient(&rdp_3));
        assert!(!optimal_budget.is_budget_sufficient(&rdp_4));
    }

    #[test]
    #[should_panic]
    fn test_rdp_min_budget_different_accounting_type() {
        let mut rdp_min_budget = RdpMinBudgetOld::new();

        let eps_dp_2 = EpsDp { eps: 1.5 };
        rdp_min_budget.add_budget_constraint(&eps_dp_2);
        assert!(rdp_min_budget.is_budget_sufficient(&eps_dp_2));
    }

    #[test]
    fn test_rdp_min_budget() {
        let rdp_1 = Rdp {
            eps_values: A5([1.0, 0.0, 3.0, 4.0, 5.0]),
        };
        let rdp_2 = Rdp {
            eps_values: A5([0.0, 5.0, 2.0, 3.0, 0.0]),
        };
        let rdp_3 = Rdp {
            eps_values: A5([6.0, 5.0, 6.0, 6.0, 5.0]),
        };
        let rdp_4 = Rdp {
            eps_values: A5([6.0, 5.0, 6.0, 6.0, 6.0]),
        };

        let mut rdp_min_budget = RdpMinBudgetOld::new();
        rdp_min_budget.add_budget_constraint(&rdp_2);
        assert!(rdp_min_budget.is_budget_sufficient(&rdp_1));
        assert!(rdp_min_budget.is_budget_sufficient(&rdp_2));
        assert!(rdp_min_budget.is_budget_sufficient(&rdp_3));
        assert!(rdp_min_budget.is_budget_sufficient(&rdp_4));

        rdp_min_budget.add_budget_constraint(&rdp_1);
        assert!(rdp_min_budget.is_budget_sufficient(&rdp_1));
        assert!(rdp_min_budget.is_budget_sufficient(&rdp_2));
        assert!(!rdp_min_budget.is_budget_sufficient(&rdp_3));
        assert!(!rdp_min_budget.is_budget_sufficient(&rdp_4));
    }
     */
}
