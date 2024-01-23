//! Contains approximate knapsack (KP) solvers and helper methods
//!
//! Algorithms are based on the book "Knapsack problems" by Kellerer, Pferschy, and Pisinger (2004),
//! henceforth referred to as KPBook.

use crate::allocation::efficiency_based::knapsack::knapsack_private::KPApproxSolverInt;

use crate::allocation::ilp::ILP_INTEGRALITY_MARGIN;
use crate::request::RequestId;
use float_cmp::ApproxEq;
use grb::expr::LinExpr;
use grb::prelude::*;
use itertools::Itertools;
use log::log;
use serde::{Deserialize, Serialize};
use std::cmp::min;
use std::collections::HashMap;
use std::convert::From;
use std::fmt;
use std::fmt::Debug;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KPItem {
    /// The id of a KP Item (corresponding to the id of the request it represents)
    pub id: ItemId,
    /// The weight of the KP Item (corresponding to the privacy cost of the request for a specific alpha)
    /// Needs to be >= 0
    pub weight: f64,
    /// The profit of the KP Item (corresponding to the utility of the request). Needs to be >= 0
    pub profit: f64,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub struct ItemId(pub usize);

impl From<RequestId> for ItemId {
    fn from(req_id: RequestId) -> Self {
        ItemId(req_id.0)
    }
}

impl From<ItemId> for RequestId {
    fn from(item_id: ItemId) -> RequestId {
        RequestId(item_id.0)
    }
}

impl fmt::Display for ItemId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// All approximate knapsack solvers implement this trait
pub trait KPApproxSolver: KPApproxSolverInt + Debug {
    /// Returns an approximation to the optimal solution to the KP problem
    ///
    /// In contrast to [solve](KPApproxSolverInt::solve) can accept zero-weights/profits and
    /// weights > capacity. Will preprocess the inputs and the call solve
    fn preprocess_and_solve(
        &self,
        items: Vec<KPItem>,
        capacity: f64,
        approx_factor: f64,
    ) -> Vec<KPItem> {
        // negative capacity does not make sense here, and capacity zero is at least unusual
        assert!(capacity >= 0.0, "Capacity must be >= 0");
        if capacity == 0.0 {
            log!(log::Level::Debug, "KPApproxSolver: Capacity is 0.");
        }

        // no duplicate ids
        {
            let id_set = items
                .iter()
                .map(|item| item.id)
                .collect::<std::collections::HashSet<ItemId>>();
            assert_eq!(id_set.len(), items.len(), "Some item ids are duplicated.");
        }

        // approx factor needs to be between 0 and 1
        assert!(
            (0.0..=1.0).contains(&approx_factor),
            "Approximation factor must be between 0 and 1"
        );

        // negative weights and profits do not make sense here and indicate malformed input
        // will warn if any weights or profits are zero
        {
            let mut zero_weight_items: Vec<ItemId> = Vec::new();
            let mut zero_profit_items: Vec<ItemId> = Vec::new();
            for item in items.iter() {
                assert!(item.weight >= 0.0, "Item weight must be >= 0");
                assert!(item.profit >= 0.0, "Item profit must be >= 0");
                if item.weight == 0.0 {
                    zero_weight_items.push(item.id)
                }
                if item.profit == 0.0 {
                    zero_profit_items.push(item.id);
                }
            }
            if !zero_weight_items.is_empty() {
                log!(
                    log::Level::Warn,
                    "KPApproxSolver: Zero-weight items detected. Will add them to solution. \
                    Id's: {:?} (corresponds to request ids)",
                    zero_weight_items
                );
            }
            if !zero_profit_items.is_empty() {
                log!(
                    log::Level::Warn,
                    "KPApproxSolver: Zero-profit items detected. Will drop them (even if they have \
                    no weight). Id's: {:?} (corresponds to request ids)",
                    zero_profit_items
                );
            }
        }

        // Can drop items with zero profit and weight > capacity
        let (free_items, other_items): (Vec<KPItem>, Vec<KPItem>) = items
            .into_iter()
            .filter(|item| item.weight <= capacity && item.profit > 0.0)
            .partition(|item| item.weight == 0.0);

        if capacity == 0.0 || other_items.is_empty() {
            return free_items;
        }

        let mut approx_sol = KPApproxSolverInt::solve(self, other_items, capacity, approx_factor);
        approx_sol.extend(free_items);
        approx_sol
    }
}

mod knapsack_private {
    //! Contains the internal trait for all (approximate) knapsack solvers and some helper methods,
    //! which should not be used outside of the parent module.
    use crate::allocation::efficiency_based::knapsack::KPItem;

    /// The internal trait for all approximate knapsack solvers
    pub trait KPApproxSolverInt {
        /// Returns an approximation to the optimal solution to the KP problem
        ///
        /// * `items` - The KPItems from which to select to maximize profit while staying under capacity.
        /// The items are assumed to have profit and weight strictly greater than 0.
        /// * `capacity` - The maximum total weight that can be selected, which is assumed to be
        /// strictly greater than 0.
        /// * `approx_factor` - The approximation factor of the solution. An exception will be thrown
        /// if the approximation factor is set to a value not supported by the current solver. Should
        /// always be set to a value between 0 and 1. 0 Means no guarantee, 1 means exact solution.
        fn solve(&self, items: Vec<KPItem>, capacity: f64, approx_factor: f64) -> Vec<KPItem>;
    }
}

/// A greedy KP solver that selects items in decreasing order of profit/weight ratio
/// (KPBook, Figure 2.1).
///
/// Note that this does not guarantee any approximation factor, so needs to
/// be set to 0.
#[derive(Debug)]
pub struct GreedyKPSolver {}

impl KPApproxSolverInt for GreedyKPSolver {
    fn solve(&self, mut items: Vec<KPItem>, capacity: f64, approx_factor: f64) -> Vec<KPItem> {
        assert_eq!(
            approx_factor, 0.0,
            "GreedyKPSolver does not guarantee any approximation factor"
        );

        debug_assert!(
            items.iter().all(|item| item.weight <= capacity),
            "Weight of item exceeds capacity. Did you use preprocess_and_solve(..)?"
        );

        items.sort_by(|a, b| {
            let a_ratio = a.profit / a.weight;
            let b_ratio = b.profit / b.weight;
            a_ratio
                .partial_cmp(&b_ratio)
                .expect("Could not compare profit/weight ratios")
                .reverse()
        });

        let mut curr_weight: f64 = 0.0;
        let mut knapsack: Vec<KPItem> = Vec::new();
        for item in items.into_iter() {
            if curr_weight + item.weight <= capacity {
                curr_weight += item.weight;
                knapsack.push(item);
            } else {
                break;
            }
        }
        knapsack
    }
}

impl KPApproxSolver for GreedyKPSolver {}

/// A extended greedy KP solver that always returns at least an 1/2-approximation
/// (KPBook, Theorem 2.5.4).
#[derive(Debug)]
pub struct ExtendedGreedyKPSolver {}

impl KPApproxSolverInt for ExtendedGreedyKPSolver {
    fn solve(&self, items: Vec<KPItem>, capacity: f64, approx_factor: f64) -> Vec<KPItem> {
        assert!(
            approx_factor <= 0.5,
            "ExtendedGreedyKPSolver only guarantees 1/2-approximation"
        );

        debug_assert!(
            items.iter().all(|item| item.weight <= capacity),
            "Weight of item exceeds capacity. Did you use preprocess_and_solve(..)?"
        );

        if items.is_empty() {
            return Vec::new();
        }
        let max_val_item = items
            .iter()
            .max_by(|a, b| a.profit.partial_cmp(&b.profit).unwrap())
            .cloned()
            .unwrap();
        let greedy_sol = GreedyKPSolver {}.solve(items, capacity, 0.0);
        let greedy_sol_val: f64 = greedy_sol.iter().map(|item| item.profit).sum();

        if greedy_sol_val < max_val_item.profit {
            vec![max_val_item]
        } else {
            greedy_sol
        }
    }
}

impl KPApproxSolver for ExtendedGreedyKPSolver {}

/// A solver for the knapsack problem using Gurobi
#[derive(Debug)]
pub struct GurobiKPSolver {}

impl KPApproxSolverInt for GurobiKPSolver {
    fn solve(&self, items: Vec<KPItem>, capacity: f64, approx_factor: f64) -> Vec<KPItem> {
        debug_assert!(
            items.iter().all(|item| item.weight <= capacity),
            "Weight of item exceeds capacity. Did you use preprocess_and_solve(..)?"
        );

        match self.solve_inner(items, capacity, approx_factor) {
            Ok(sol) => sol,
            Err(e) => {
                panic!(
                    "GurobiKPSolver: Error while solving knapsack problem: {:?}",
                    e
                )
            }
        }
    }
}

impl GurobiKPSolver {
    fn solve_inner(
        &self,
        items: Vec<KPItem>,
        capacity: f64,
        approx_factor: f64,
    ) -> Result<Vec<KPItem>, grb::Error> {
        let mut model = Model::new("knapsack").unwrap();

        // Note: The MIPGap is defined as |(UB - LB)| / |LB|, therefore we need to invert the approximation
        // factor and subtract 1 to get the correct value for the gap
        // (assuming all values are positive, which is the case here, this implies our bound)
        // examples:
        //      approx_factor: 1/2 -> gap = 1/(1/2) - 1 = 1
        //      approx_factor: 1 -> gap = 1/1 - 1 = 0
        //      approx_factor: 0 -> gap = 1/0 - 1 = inf

        model
            .set_param(param::MIPGap, (1.0 / approx_factor) - 1.0)
            .unwrap();
        model.set_param(param::Threads, 1).unwrap();

        let mut goal = LinExpr::new();
        let mut constraint = LinExpr::new();
        let mut var_map: HashMap<ItemId, Var> = HashMap::new();
        for item in items.iter() {
            let var = add_binvar!(model, name: &item.id.to_string())?;
            goal.add_term(item.profit, var);
            constraint.add_term(item.weight, var);
            var_map.insert(item.id, var);
        }

        model.add_constr("capacity_constraint", c!(constraint <= capacity))?;
        model.set_objective(goal, Maximize)?;

        model.optimize().unwrap();
        assert_eq!(model.status()?, Status::Optimal);
        Ok(items
            .into_iter()
            .filter(|item| {
                let var = var_map.get(&item.id).unwrap();
                let assigned_val = model
                    .get_obj_attr(attr::X, var)
                    .expect("Could not read variable value from gurobi model");
                assigned_val.approx_eq(1.0, ILP_INTEGRALITY_MARGIN)
            })
            .collect())
    }
}

impl KPApproxSolver for GurobiKPSolver {}

/// The FPTAS algorithm H^eps for the 1d KP problem (KPBook, Fig. 2.11), which is also be used in
/// "Packing Privacy Budget Efficiently" <https://arxiv.org/abs/2212.13228> (it is not explicitly
/// mentioned which FPTAS was used in the implementation, but the mentioned asymptotic running time
/// O(n^3/eps) in the proof of property 2 matches the one of this algorithm).
#[derive(Debug)]
pub struct HepsFPTASKPSolver {}

impl KPApproxSolverInt for HepsFPTASKPSolver {
    fn solve(&self, items: Vec<KPItem>, capacity: f64, approx_factor: f64) -> Vec<KPItem> {
        assert!(
            approx_factor > 0.5,
            "HepsFPTASKPSolver assumes more than 1/2-approximation, \
            use ExtendedGreedyKPSolver or 1/2 for lower approximation factors"
        );

        debug_assert!(
            items.iter().all(|item| item.weight <= capacity),
            "Weight of item exceeds capacity. Did you use preprocess_and_solve(..)?"
        );

        let eps = 1.0 - approx_factor;
        let l = min((1.0 / eps).ceil() as usize - 2, items.len() as usize);

        let n_items = items.len();

        // for all subsets of size 1 until l-1, we find the subset with the highest profit sum with a sum of weights below the capacity
        // for subsets of size l, we have an additonal step see below
        let (_profit_sum, idx_set) = (1..l + 1)
            .flat_map(|k| (0..n_items).combinations(k))
            .map(|idx_set| {
                //aggregate weights
                let weight_sum: f64 = idx_set.iter().map(|idx| items[*idx].weight).sum();
                (weight_sum, idx_set)
            })
            .filter(|(weight_sum, _idx_set)| *weight_sum <= capacity) // filter out subsets exceeding budget
            .map(|(weight_sum, idx_set)| {
                // aggregate profits
                let profit_sum: f64 = idx_set.iter().map(|idx| items[*idx].profit).sum();
                (profit_sum, weight_sum, idx_set)
            })
            .map(|(mut profit_sum, weight_sum, mut idx_set)| {
                // For subsets of length l, we consider all other items that have a smaller profit than the minimum profit of the subset. And we try to additionally add them via ExtGreedy.
                // i.e., 2nd part of the algorithm
                if idx_set.len() == l {
                    let min_profit: f64 = idx_set
                        .iter()
                        .map(|idx| items[*idx].profit)
                        .reduce(f64::min)
                        .unwrap();

                    // TODO: could probably make faster by not collecting
                    let greedy_n: Vec<KPItem> = items // N in the pseudocode
                        .iter()
                        .enumerate()
                        .filter(|(idx, item)| {
                            item.profit < min_profit
                                || (item.profit == min_profit && !idx_set.contains(idx))
                        })
                        .map(|(idx, item)| KPItem {
                            id: ItemId(idx),
                            weight: item.weight,
                            profit: item.profit,
                        }) // we use idx instead of the id
                        .collect();

                    // run extended greedy on remaining items (with smaller than min_profit)
                    let greedy_sol = ExtendedGreedyKPSolver {}.preprocess_and_solve(
                        greedy_n,
                        capacity - weight_sum,
                        0.5,
                    );
                    let greedy_sol_val: f64 = greedy_sol.iter().map(|item| item.profit).sum();

                    // update solution
                    idx_set.extend(greedy_sol.iter().map(|item| item.id.0));
                    profit_sum += greedy_sol_val;
                }
                (profit_sum, idx_set)
            })
            .reduce(|(profit_sum_a, idx_set_a), (profit_sum_b, idx_set_b)| {
                if profit_sum_a > profit_sum_b {
                    (profit_sum_a, idx_set_a)
                } else {
                    (profit_sum_b, idx_set_b)
                }
            })
            .unwrap(); // find set with highest profits

        let solution: Vec<KPItem> = idx_set.iter().map(|idx| items[*idx].clone()).collect();
        solution
    }
}

impl KPApproxSolver for HepsFPTASKPSolver {}

#[cfg(test)]
mod tests {
    use crate::allocation::efficiency_based::knapsack::knapsack_private::KPApproxSolverInt;
    use crate::allocation::efficiency_based::knapsack::{ItemId, KPApproxSolver, KPItem};
    use float_cmp::{ApproxEq, F64Margin};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use serde::{Deserialize, Serialize};
    use serial_test::serial;
    use std::collections::{HashMap, HashSet};
    use std::io::Write;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct KPProblemInstance {
        items: Vec<KPItem>,
        capacity: f64,
        optimal_profit: f64,
    }

    struct ProfitWeight {
        profit: f64,
        weight: f64,
    }

    /// returns a list of solvers and the corresponding approximation factors to be tested except
    /// for FPTAS (since that one takes longer)
    ///
    /// Used in various tests, so any new solver should be added here, except if the running time
    /// is too long
    fn get_fast_solvers() -> Vec<(Box<dyn KPApproxSolver>, Vec<f64>)> {
        vec![
            (Box::new(super::GreedyKPSolver {}), vec![0.0]),
            (Box::new(super::ExtendedGreedyKPSolver {}), vec![0.5]),
            // (Box::new(super::GurobiKPSolver {}), vec![0.6, 0.8]),
        ]
    }

    /// returns a list of solvers and the corresponding approximation factors to be tested
    ///
    /// any solver that takes too long on many examples should be added here
    fn get_slower_solvers() -> Vec<(Box<dyn KPApproxSolver>, Vec<f64>)> {
        vec![(Box::new(super::HepsFPTASKPSolver {}), vec![0.6, 0.8])]
    }

    fn profit_weights_to_kp_items(weight_profits: Vec<ProfitWeight>) -> Vec<KPItem> {
        weight_profits
            .into_iter()
            .enumerate()
            .map(|(id, profit_weight)| KPItem {
                id: ItemId(id),
                weight: profit_weight.weight,
                profit: profit_weight.profit,
            })
            .collect()
    }

    fn get_problem_instances() -> Vec<KPProblemInstance> {
        let mut res = get_manual_problem_instances();
        res.append(&mut get_auto_problem_instances());
        res
    }

    #[allow(dead_code)]
    fn generate_auto_problem_instances() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut problem_instances = vec![];
        for _ in 0..200 {
            let num_items = rng.gen_range(1..100);
            let capacity = rng.gen_range(1.0..100.0);
            let items: Vec<KPItem> = (0..num_items)
                .map(|id| KPItem {
                    id: ItemId(id),
                    weight: rng.gen_range(1.0..100.0),
                    profit: rng.gen_range(1.0..100.0),
                })
                .collect();
            let optimal_profit = super::GurobiKPSolver {}
                .solve(items.clone(), capacity, 0.999999)
                .iter()
                .map(|item| item.profit)
                .sum();
            problem_instances.push(KPProblemInstance {
                items,
                capacity,
                optimal_profit,
            });
        }
        let serialized = serde_json::to_string(&problem_instances).unwrap();
        let mut file = std::fs::File::create("./resources/test/knapsack/kp_problem_instances.json")
            .expect("Could not create file");
        file.write_all(serialized.as_bytes())
            .expect("Could not write to file");
    }

    /// returns a list of randomly generated KP problem instances with the optimal solution found
    /// via gurobi.
    fn get_auto_problem_instances() -> Vec<KPProblemInstance> {
        // uncomment the following line to generate new problem instances (only if gurobi correctly installed on system)
        // generate_auto_problem_instances();
        let file = std::fs::File::open("./resources/test/knapsack/kp_problem_instances.json")
            .expect("Could not open file");
        let problem_instances: Vec<KPProblemInstance> =
            serde_json::from_reader(file).expect("Could not read file");
        // println!("{}", problem_instances.len());
        problem_instances
    }

    /// returns a list of handcrafted KP problem instances and the profit of the optimal solution
    fn get_manual_problem_instances() -> Vec<KPProblemInstance> {
        vec![
            KPProblemInstance {
                items: profit_weights_to_kp_items(vec![
                    ProfitWeight {
                        profit: 1.0,
                        weight: 1.0,
                    },
                    ProfitWeight {
                        profit: 2.0,
                        weight: 1.5,
                    },
                    ProfitWeight {
                        profit: 0.2,
                        weight: 0.1,
                    },
                ]),
                capacity: 1.0,
                optimal_profit: 1.0,
            },
            KPProblemInstance {
                // Example on page 16 of KPBook
                items: profit_weights_to_kp_items(vec![
                    ProfitWeight {
                        profit: 6.0,
                        weight: 2.0,
                    },
                    ProfitWeight {
                        profit: 5.0,
                        weight: 3.0,
                    },
                    ProfitWeight {
                        profit: 8.0,
                        weight: 6.0,
                    },
                    ProfitWeight {
                        profit: 9.0,
                        weight: 7.0,
                    },
                    ProfitWeight {
                        profit: 6.0,
                        weight: 5.0,
                    },
                    ProfitWeight {
                        profit: 7.0,
                        weight: 9.0,
                    },
                    ProfitWeight {
                        profit: 3.0,
                        weight: 4.0,
                    },
                ]),
                capacity: 9.0,
                optimal_profit: 15.0,
            },
        ]
    }

    /// Checks if two arrays of KPItems are equal using their id's. Will panic if there are duplicate
    /// ids in either array.
    fn unordered_array_equals(arr1: &[KPItem], arr2: &[KPItem]) -> bool {
        let set1: HashSet<ItemId> = HashSet::from_iter(arr1.iter().map(|item| item.id));
        let set2: HashSet<ItemId> = HashSet::from_iter(arr2.iter().map(|item| item.id));
        assert_eq!(set1.len(), arr1.len(), "duplicate ids in arr1");
        assert_eq!(set2.len(), arr2.len(), "duplicate ids in arr2");
        set1 == set2
    }

    /// The main test which checks that a solver returns a solution with the promised approximation ratio
    /// and that the solution is correct for the faster solvers
    #[test]
    #[serial(kp_problem_file)]
    fn test_all_fast_solvers() {
        test_solvers(get_fast_solvers, None);
    }

    /// The main test which checks that a solver returns a solution with the promised approximation ratio
    /// and that the solution is correct for the slower solvers
    ///
    /// Increase the max_instances to test more instances
    #[test]
    #[serial(kp_problem_file)]
    fn test_all_other_solvers() {
        test_solvers(get_slower_solvers, Some(5));
    }

    #[allow(clippy::type_complexity)]
    fn test_solvers(
        solvers: fn() -> Vec<(Box<dyn KPApproxSolver>, Vec<f64>)>,
        max_instances: Option<usize>,
    ) {
        let problem_instances;
        if let Some(max_instances) = max_instances {
            problem_instances = get_problem_instances()
                .into_iter()
                .take(max_instances)
                .collect();
        } else {
            problem_instances = get_problem_instances();
        }

        for problem_instance in problem_instances {
            let item_map = problem_instance
                .items
                .iter()
                .map(|item| (item.id, item))
                .collect::<HashMap<ItemId, &KPItem>>();
            assert_eq!(
                item_map.len(),
                problem_instance.items.len(),
                "duplicate ids in problem instance"
            );
            for (solver, approx_factors) in solvers() {
                for approx_factor in approx_factors {
                    //println!(
                    //    "Current solver: {:?}, approx_factor: {}",
                    //    solver, approx_factor
                    //);
                    let sol = solver.preprocess_and_solve(
                        problem_instance.items.clone(),
                        problem_instance.capacity,
                        approx_factor,
                    );

                    // check that anything in the solution is contained in the original selection
                    for sol_item in sol.iter() {
                        assert!(
                            item_map.contains_key(&sol_item.id),
                            "Solver {:?} returned solution with item id {} which is not in the original problem instance",
                            solver,
                            sol_item.id
                        );
                        assert_eq!(
                            item_map.get(&sol_item.id).unwrap().weight,
                            sol_item.weight,
                            "Solver {:?} returned solution with item id {} which has weight {} \
                                which is different from the original problem instance",
                            solver,
                            sol_item.id,
                            sol_item.weight
                        );
                        assert_eq!(
                            item_map.get(&sol_item.id).unwrap().profit,
                            sol_item.profit,
                            "Solver {:?} returned solution with item id {} which has profit {} \
                                   which is different from the original problem instance",
                            solver,
                            sol_item.id,
                            sol_item.profit
                        );
                    }

                    let sol_weight: f64 = sol.iter().map(|item| item.weight).sum();
                    assert!(
                        sol_weight <= problem_instance.capacity || sol_weight.approx_eq(problem_instance.capacity, F64Margin::default()),
                        "Solver {:?} returned solution with weight {} which is greater than the capacity {}",
                        solver,
                        sol_weight,
                        problem_instance.capacity
                    );

                    let sol_profit: f64 = sol.iter().map(|item| item.profit).sum();
                    let expected_profit = problem_instance.optimal_profit * approx_factor;
                    assert!(
                        sol_profit >= expected_profit || sol_profit.approx_eq(expected_profit, F64Margin::default()),
                        "Solver {:?} returned solution with profit {} which is less than the promised approximation factor {} of the optimal solution {}",
                        solver,
                        sol_profit,
                        approx_factor,
                        problem_instance.optimal_profit
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "Item weight must be >= 0")]
    fn negative_weight() {
        let items: Vec<KPItem> = vec![KPItem {
            id: ItemId(0),
            weight: -1.0,
            profit: 0.0,
        }];
        let capacity: f64 = 0.0;
        let approx_factor: f64 = 0.0;
        let solver = super::GreedyKPSolver {};
        let _sol = solver.preprocess_and_solve(items, capacity, approx_factor);
    }

    #[test]
    #[should_panic(expected = "Item profit must be >= 0")]
    fn negative_profit() {
        let items: Vec<KPItem> = vec![KPItem {
            id: ItemId(0),
            weight: 0.0,
            profit: -1.0,
        }];
        let capacity: f64 = 0.0;
        let approx_factor: f64 = 0.0;
        let solver = super::GreedyKPSolver {};
        let _sol = solver.preprocess_and_solve(items, capacity, approx_factor);
    }

    #[test]
    #[should_panic(expected = "Capacity must be >= 0")]
    fn negative_capacity() {
        let items: Vec<KPItem> = vec![KPItem {
            id: ItemId(0),
            weight: 0.0,
            profit: 0.0,
        }];
        let capacity: f64 = -1.0;
        let approx_factor: f64 = 0.0;
        let solver = super::GreedyKPSolver {};
        let _sol = solver.preprocess_and_solve(items, capacity, approx_factor);
    }

    #[test]
    #[should_panic(expected = "Approximation factor must be between 0 and 1")]
    fn negative_approx_factor() {
        let items: Vec<KPItem> = vec![KPItem {
            id: ItemId(0),
            weight: 0.0,
            profit: 0.0,
        }];
        let capacity: f64 = 0.0;
        let approx_factor: f64 = -0.1;
        let solver = super::GreedyKPSolver {};
        let _sol = solver.preprocess_and_solve(items, capacity, approx_factor);
    }

    #[test]
    #[should_panic(expected = "Approximation factor must be between 0 and 1")]
    fn approx_factor_gt_one() {
        let items: Vec<KPItem> = vec![KPItem {
            id: ItemId(0),
            weight: 0.0,
            profit: 0.0,
        }];
        let capacity: f64 = 0.0;
        let approx_factor: f64 = 1.1;
        let solver = super::GreedyKPSolver {};
        let _sol = solver.preprocess_and_solve(items, capacity, approx_factor);
    }

    #[test]
    fn no_items() {
        let items: Vec<KPItem> = vec![];
        let capacity: f64 = 0.0;
        run_solvers(&items, &vec![], capacity);
    }

    #[test]
    fn no_weight_items() {
        let items: Vec<KPItem> = vec![
            KPItem {
                // should be selected
                id: ItemId(0),
                weight: 0.0,
                profit: 1.0,
            },
            KPItem {
                // should be selected
                id: ItemId(1),
                weight: 0.0,
                profit: 2.0,
            },
            KPItem {
                // should not be selected
                id: ItemId(2),
                weight: 0.0,
                profit: 0.0,
            },
        ];
        let capacity: f64 = 0.0;
        let expected = items.iter().take(2).cloned().collect::<Vec<KPItem>>();
        run_solvers(&items, &expected, capacity);
    }

    #[test]
    fn no_profit_items() {
        let items: Vec<KPItem> = vec![
            KPItem {
                // should not be selected
                id: ItemId(0),
                weight: 0.0,
                profit: 0.0,
            },
            KPItem {
                // should not be selected
                id: ItemId(1),
                weight: 1.0,
                profit: 0.0,
            },
        ];
        let expected: Vec<KPItem> = vec![];
        let capacity: f64 = 0.0;
        run_solvers(&items, &expected, capacity);
    }

    #[test]
    #[should_panic(expected = "Some item ids are duplicated.")]
    fn duplicate_ids() {
        let items: Vec<KPItem> = vec![
            KPItem {
                id: ItemId(0),
                weight: 1.0,
                profit: 1.0,
            },
            KPItem {
                id: ItemId(0),
                weight: 1.0,
                profit: 1.0,
            },
        ];
        let expected: Vec<KPItem> = vec![];
        let capacity: f64 = 1.0;
        run_solvers(&items, &expected, capacity);
    }

    /// Tests that the solver returns the expected solution for all solvers. Note that this method
    /// is not suitable if results are not deterministic.
    fn run_solvers(items: &[KPItem], expected: &Vec<KPItem>, capacity: f64) {
        for (solver, approx_factors) in get_fast_solvers() {
            for approx_factor in approx_factors {
                //println!(
                //    "Current solver: {:?}, approx_factor: {}",
                //    solver, approx_factor
                //);
                let sol = solver.preprocess_and_solve(
                    Vec::from_iter(items.iter().cloned()),
                    capacity,
                    approx_factor,
                );
                assert!(
                    unordered_array_equals(&sol, expected),
                    "left: {:?}, right: {:?}",
                    sol,
                    expected
                )
            }
        }
    }
}
