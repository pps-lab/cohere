//! Contains the implementation of the DPK algorithm from "Packing Privacy Budget Efficiently"
//! from Tholoniat et al.

mod knapsack;

use crate::block::Block;
use crate::composition::{CompositionConstraint, ProblemFormulation, StatusResult};
use crate::dprivacy::budget::SegmentBudget;
use std::cmp::Ordering;

use crate::schema::Schema;

use crate::block::BlockId;
use crate::request::{Request, RequestId};

use crate::allocation::efficiency_based::knapsack::{
    GurobiKPSolver, HepsFPTASKPSolver, KPApproxSolver, KPItem,
};
use crate::allocation::utils::try_allocation;
use crate::allocation::{AllocationStatus, BlockCompWrapper};
use crate::config::{EfficiencyBasedAlgo, KPSolverType};
use crate::dprivacy::{AlphaIndex, RdpAccounting};
use crate::logging::{DpfStats, RuntimeKind, RuntimeMeasurement};
use crate::AccountingType;
use itertools::Itertools;
use rand::prelude::IteratorRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::Deref;

use super::ResourceAllocation;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentId {
    pub block_id: BlockId,
    pub segment: BTreeSet<RequestId>,
}

impl From<(BlockId, BTreeSet<RequestId>)> for SegmentId {
    fn from((block_id, segment): (BlockId, BTreeSet<RequestId>)) -> Self {
        SegmentId { block_id, segment }
    }
}

/// Contains any persistent state needed to execute the efficiency-based algorithms
#[derive(Debug)]
pub struct EfficiencyBased {
    /// maps each request seen so far to selected block ids
    request_cache: HashMap<RequestId, HashSet<BlockId>>,

    /// The rng used to decide the random assignment of blocks to requests
    rng: StdRng,

    algo: EfficiencyBasedAlgo,
}

impl EfficiencyBased {
    /// # Arguments
    /// * `seed` - The seed used to initialize the random number generator
    /// * `eta` - determines the optimality of the knapsack solver. See [eta](#field.eta) for more
    /// details

    pub fn construct_allocator(seed: u64, algo: EfficiencyBasedAlgo) -> Self {
        #[allow(irrefutable_let_patterns)]
        if let EfficiencyBasedAlgo::Dpk { eta, .. } = algo {
            assert!(
                eta > 0.0 && eta < 0.75,
                "Eta should be between 0 and 0.75 (ends not included)."
            );
        }
        EfficiencyBased {
            request_cache: HashMap::new(),
            rng: StdRng::seed_from_u64(seed),
            algo,
        }
    }

    pub fn round<M: SegmentBudget>(
        &mut self,
        candidate_requests: &HashMap<RequestId, Request>,
        request_history: &HashMap<RequestId, Request>,
        available_blocks: &HashMap<BlockId, Block>,
        schema: &Schema,
        block_comp_wrapper: &BlockCompWrapper,
        runtime_measurements: &mut Vec<RuntimeMeasurement>,
    ) -> (ResourceAllocation, AllocationStatus) {
        // build problem formulation
        // TODO: Could choose blocks before building problem formulation and take this into account when building the problem formulation
        let mut pf: ProblemFormulation<M> = block_comp_wrapper.build_problem_formulation::<M>(
            available_blocks,
            candidate_requests,
            request_history,
            schema,
            runtime_measurements,
        );

        let num_contested_segments_initially = pf.contested_constraints().count();

        let mut alloc_meas = RuntimeMeasurement::start(RuntimeKind::RunAllocationAlgorithm);

        // for each new candidate request, we randomly select blocks which we want to allocate
        for (rid, request) in candidate_requests.iter() {
            if !self.request_cache.contains_key(rid) {
                assert!(
                    available_blocks.len() >= request.n_users,
                    "Not enough blocks to fulfill request"
                );
                self.request_cache.insert(
                    *rid,
                    HashSet::from_iter(
                        available_blocks
                            .keys()
                            .copied()
                            .choose_multiple(&mut self.rng, request.n_users),
                    ),
                );
            }
        }

        // calculate efficiency score for each request if there any contested segments,
        // depending on the chosen algo
        let efficiency_scores = if pf.contested_constraints().next().is_none() {
            // assign all requests the same efficiency of 1.0 if there are no contested segments
            candidate_requests
                .iter()
                .map(|(rid, _)| (*rid, 1.0))
                .collect()
        } else {
            match self.algo {
                EfficiencyBasedAlgo::Dpk { eta, kp_solver, .. } => {
                    self.calculate_dpk_efficiency(candidate_requests, eta, kp_solver, &pf)
                }
            }
        };

        // sort candidate requests based on their efficiency
        let sorted_candidates: Vec<RequestId> = candidate_requests
            .keys()
            .copied()
            .sorted_by(|rid1, rid2| {
                // if a request does not touch any contested segments, we assign it the highest efficiency
                let c1: f64 = efficiency_scores.get(rid1).copied().unwrap_or(f64::MAX);
                let c2: f64 = efficiency_scores.get(rid2).copied().unwrap_or(f64::MAX);
                let cost_cmp = c1
                    .partial_cmp(&c2)
                    .expect("Some efficiency scores could not be compared")
                    .reverse(); // we want to sort in descending order (higher efficiency first)
                if let Ordering::Equal = cost_cmp {
                    // if costs are equal, we order by request id instead
                    let rid_cmp = Ord::cmp(rid1, rid2);
                    if let Ordering::Equal = rid_cmp {
                        panic!("Found two requests with same efficiency score and same rid");
                    } else {
                        return rid_cmp;
                    }
                }
                cost_cmp
            })
            .collect();

        let mut resource_allocation = ResourceAllocation {
            accepted: HashMap::new(),
            rejected: HashSet::new(),
        };

        // try to allocate requests greedily
        for rid in sorted_candidates {
            let selected_blocks = self.request_cache[&rid].clone();
            match pf.request_status(rid, None, candidate_requests) {
                StatusResult::Acceptable {
                    acceptable: acc_bids,
                    contested: con_bids,
                } => try_allocation(
                    candidate_requests,
                    &mut pf,
                    &mut resource_allocation,
                    rid,
                    selected_blocks,
                    acc_bids,
                    con_bids,
                ),
                StatusResult::Contested {
                    acceptable: acc_bids,
                    contested: con_bids,
                } => try_allocation(
                    candidate_requests,
                    &mut pf,
                    &mut resource_allocation,
                    rid,
                    selected_blocks,
                    acc_bids,
                    con_bids,
                ),
                StatusResult::Rejected => {
                    resource_allocation.rejected.insert(rid);
                }
            }
        }

        runtime_measurements.push(alloc_meas.stop());

        (
            resource_allocation,
            AllocationStatus::DpfStatus(DpfStats {
                num_contested_segments_initially,
            }),
        )
    }

    /// Computes the efficiency score for each request that has at least one contested segment
    /// as described in "Packing Privacy Budget Efficiently" by Tholoniat et al. for the dpk
    /// allocation algorithm
    ///
    /// If a different efficiency score is desired, this function can be replaced.
    fn calculate_dpk_efficiency<M: SegmentBudget>(
        &self,
        candidate_requests: &HashMap<RequestId, Request>,
        eta: f64,
        solver: KPSolverType,
        pf: &ProblemFormulation<M>,
    ) -> HashMap<RequestId, f64> {
        let best_alphas_and_capacities = self.calc_best_alphas(candidate_requests, eta, solver, pf);
        let request_demand_capacity_ratios: HashMap<RequestId, Vec<f64>> =
            best_alphas_and_capacities
                .into_iter()
                .flat_map(|(segment_id, (alpha_index, capacity))| {
                    // compute dij / cj for each segment for each request (if that request applies to the segment)
                    segment_id
                        .segment
                        .iter()
                        .filter(move |rid| {
                            self.request_cache
                                .get(rid)
                                .unwrap()
                                .contains(&segment_id.block_id)
                        })
                        .map(move |rid| {
                            let request_cost = candidate_requests
                                .get(rid)
                                .unwrap()
                                .request_cost
                                .get_value_for_alpha(alpha_index);
                            (*rid, request_cost / capacity)
                        })
                        .collect::<Vec<_>>()
                })
                .fold(HashMap::new(), |mut acc, (a, b)| {
                    acc.entry(a).or_default().push(b);
                    acc
                });

        request_demand_capacity_ratios
            .into_iter()
            .map(|(rid, ratios)| {
                let ratio_sum = ratios.into_iter().sum::<f64>();
                let request_profit = candidate_requests.get(&rid).unwrap().profit as f64;
                (rid, request_profit / ratio_sum)
            })
            .collect()
    }

    /// Calculates the best alpha for each partition, similar to as described in "Packing Privacy Budget Efficiently"
    /// by Tholoniat et al. Returns the alpha index and the associated budget (a 1d RDP type)
    ///
    /// # Arguments
    /// * `candidate_requests` - The requests that are currently being considered for allocation
    /// * `eta` - The parameter eta as described in the paper
    /// * `solver` - The (approximate) solver to use for the 1d knapsack problem.
    /// * `pf` - The problem formulation which contains the contested constraints
    fn calc_best_alphas<M: SegmentBudget>(
        &self,
        candidate_requests: &HashMap<RequestId, Request>,
        eta: f64,
        solver: KPSolverType,
        pf: &ProblemFormulation<M>,
    ) -> HashMap<SegmentId, (AlphaIndex, f64)> {
        assert!(
            eta < 0.75 && eta > 0.0,
            "eta must be smaller than 0.75, else the FPTAS does not apply \
        (it can only be used with approximation factors for which 1 > approximation factor > 0.5, \
        which is the case iff 0 < eta < 0.75)"
        );

        let contested_segments: Vec<(SegmentId, Vec<&AccountingType>)> = pf
            .contested_constraints()
            .map(|(bid, rids, budget)| {
                (
                    SegmentId::from((bid, rids)),
                    budget.get_budget_constraints(),
                )
            })
            .collect();

        let alpha_indices = contested_segments
            .first()
            .map(|(_, constraints)| {
                constraints
                    .first()
                    .expect("no budget constraints in segment")
                    .deref()
                    .get_alpha_indices()
            })
            .expect("No contested segments");

        #[allow(irrefutable_let_patterns)]
        let num_threads = if let EfficiencyBasedAlgo::Dpk { num_threads, .. } = self.algo {
            let num_threads = num_threads.unwrap_or_else(num_cpus::get);
            assert!(
                num_threads > 0 && num_threads <= num_cpus::get(),
                "num_threads must be greater than 0 and smaller than or equal \
            to the number of virtual cores ({}) (given number of threads was {})",
                num_cpus::get(),
                num_threads
            );
            num_threads
        } else {
            panic!("calc_best_alphas should only be called for the dpk algorithm");
        };
        crate::util::create_pool(num_threads)
            .expect("Failed to create thread pool")
            .install(|| {
                contested_segments
                    .into_par_iter()
                    .map(|(segment_id, constraints)| {
                        // for each segment, we determine the best alpha by solving 1d knapsack (approximately)

                        // get the requests that apply to this segment
                        let requests: Vec<_> = segment_id
                            .segment
                            .iter()
                            .map(|rid| {
                                candidate_requests
                                    .get(rid)
                                    .expect("request not found in candidate requests")
                            })
                            .filter(|r| {
                                // note that the pf considers that each request can use every block,
                                // so we need to filter out the segments in blocks that are not actually
                                // requested by the request
                                self.request_cache
                                    .get(&r.request_id)
                                    .unwrap()
                                    .contains(&segment_id.block_id)
                            })
                            .collect();

                        // for each segment, we determine the best alpha index by running the 1d knapsack
                        // solver for each alpha index
                        let (best_alpha_index, best_alpha_capacity): (AlphaIndex, f64) =
                            alpha_indices
                                .iter()
                                .map(|alpha_index| {
                                    // if we only consider a single alpha, we need to take the minimum budget across all constraints
                                    let capacity = constraints
                                        .iter()
                                        .map(|c| c.get_value_for_alpha(*alpha_index))
                                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                                        .expect("no budget constraints in segment")
                                        .max(0.0);

                                    // Each request is an allocation item with weight equal to the cost of the request
                                    // in the knapsack problem
                                    let items: Vec<KPItem> = requests
                                        .iter()
                                        .map(|r| KPItem {
                                            id: r.request_id.into(),
                                            weight: r
                                                .request_cost
                                                .get_value_for_alpha(*alpha_index),
                                            profit: r.profit as f64,
                                        })
                                        .collect::<Vec<_>>();

                                    // run the 1d knapsack solver
                                    let kp_sol = match solver {
                                        KPSolverType::Gurobi => GurobiKPSolver {}
                                            .preprocess_and_solve(
                                                items,
                                                capacity,
                                                1.0 - 2.0 * eta / 3.0,
                                            ),
                                        KPSolverType::FPTAS => HepsFPTASKPSolver {}
                                            .preprocess_and_solve(
                                                items,
                                                capacity,
                                                1.0 - 2.0 * eta / 3.0,
                                            ),
                                    };

                                    // profit using this alpha index
                                    let index_profit =
                                        kp_sol.into_iter().map(|i| i.profit).sum::<f64>();

                                    (*alpha_index, index_profit, capacity)
                                })
                                .max_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap())
                                .map(|(alpha_index, _, capacity)| (alpha_index, capacity))
                                .unwrap();
                        (segment_id, (best_alpha_index, best_alpha_capacity))
                    })
                    .collect()
            })
    }
}

#[cfg(test)]
mod tests {
    use crate::allocation::efficiency_based::{EfficiencyBased, KPSolverType, SegmentId};
    use crate::allocation::BlockCompWrapper;
    use crate::block::BlockId::User;
    use crate::block::{Block, BlockId};
    use crate::composition::{
        block_composition, block_composition_pa, CompositionConstraint, ProblemFormulation,
    };
    use crate::config::{EfficiencyBasedAlgo, SegmentationAlgo};
    use crate::dprivacy::budget::OptimalBudget;
    use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::A3;
    use crate::dprivacy::AccountingType::Rdp;
    use crate::dprivacy::{AccountingType, AlphaIndex};
    use crate::request::{Request, RequestId};
    use crate::schema::Schema;
    use crate::util::{
        build_dummy_requests_no_pa, build_dummy_requests_with_pa, build_dummy_schema,
        generate_blocks, get_dummy_composition,
    };
    use float_cmp::{ApproxEq, F64Margin};
    use itertools::Itertools;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::{BTreeMap, HashMap, HashSet};

    #[test]
    fn test_dpk_allocation_score_no_pa() {
        let n_blocks = 100;

        let block_comp_wrapper = BlockCompWrapper::BlockCompositionPartAttributesVariant(
            block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray),
        );

        let (_bids, candidate_requests, _pf, mut dpk, schema, blocks) =
            setup_dpk_test_no_pa(n_blocks);
        // println!("block len: {}", blocks.len());

        let resource_allocation = dpk
            .round::<OptimalBudget>(
                &candidate_requests,
                &HashMap::new(),
                &blocks,
                &schema,
                &block_comp_wrapper,
                &mut Vec::new(),
            )
            .0;
        // As with pa, request costs and budget ensure we can have at most 2 requests per block.
        // The requests with the highest profit are r5 and r6, so the chosen alpha index will be 2,
        // as then both r5 and r6 can run.
        // therefore, we expected that r5 and r6 are accepted, and r0, r1, r2, r3, r4 are rejected.

        let expected_accepted = vec![RequestId(5), RequestId(6)];
        let expected_rejected = vec![
            RequestId(0),
            RequestId(1),
            RequestId(2),
            RequestId(3),
            RequestId(4),
        ];
        assert_eq!(resource_allocation.accepted.len(), expected_accepted.len());
        assert_eq!(resource_allocation.rejected.len(), expected_rejected.len());
        assert!(expected_accepted
            .iter()
            .all(|r| resource_allocation.accepted.keys().contains(r)));
        assert!(expected_rejected
            .iter()
            .all(|r| resource_allocation.rejected.contains(r)));
    }

    #[test]
    fn test_dpk_allocation_score_with_pa() {
        let n_blocks = 100;

        let block_comp_wrapper = BlockCompWrapper::BlockCompositionPartAttributesVariant(
            block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray),
        );

        let (_bids, candidate_requests, _pf, mut dpk, schema, blocks) =
            setup_dpk_test_with_pa(n_blocks);
        // println!("block len: {}", blocks.len());

        let resource_allocation = dpk
            .round::<OptimalBudget>(
                &candidate_requests,
                &HashMap::new(),
                &blocks,
                &schema,
                &block_comp_wrapper,
                &mut Vec::new(),
            )
            .0;
        // Requests ordered according to efficiency:
        // r2, r0, r6, r5, r4, r3, r1
        // Request which should run:
        // r2, r0, r6, r5, r4
        let expected_accepted = vec![
            RequestId(2),
            RequestId(0),
            RequestId(6),
            RequestId(5),
            RequestId(4),
        ];
        let expected_rejected = vec![RequestId(3), RequestId(1)];
        assert_eq!(resource_allocation.accepted.len(), expected_accepted.len());
        assert_eq!(resource_allocation.rejected.len(), expected_rejected.len());
        assert!(expected_accepted
            .iter()
            .all(|r| resource_allocation.accepted.keys().contains(r)));
        assert!(expected_rejected
            .iter()
            .all(|r| resource_allocation.rejected.contains(r)));
    }

    #[test]
    fn test_calculate_efficiency_scores() {
        let n_blocks = 100;

        let (_bids, candidate_requests, pf, dpk, _, _) = setup_dpk_test_with_pa(n_blocks);

        let expected_efficiencies: BTreeMap<RequestId, f64> = [
            (RequestId(0), 100.0 / ((0.4 / 1.15) * n_blocks as f64)),
            (
                RequestId(1),
                110.0 / ((0.4 / 1.15 + 1.0 / 1.05 + 0.4 / 1.10) * n_blocks as f64),
            ),
            (RequestId(2), 120.0 / ((0.4 / 1.15) * n_blocks as f64)),
            (
                RequestId(3),
                130.0 / ((0.4 / 1.05 + 1.0 / 1.10 + 0.4 / 1.15) * n_blocks as f64),
            ),
            (
                RequestId(4),
                140.0 / ((0.4 / 1.05 + 0.4 / 1.10) * n_blocks as f64),
            ),
            (
                RequestId(5),
                150.0 / ((0.4 / 1.15 + 0.4 / 1.05) * n_blocks as f64),
            ),
            (
                RequestId(6),
                160.0 / ((0.4 / 1.10 + 0.4 / 1.15) * n_blocks as f64),
            ),
        ]
        .into_iter()
        .collect();

        // for 100 blocks: {
        // RequestId(0): 2.8749999999999996,
        // RequestId(1): 0.6611199095022624,
        // RequestId(2): 3.4499999999999993,
        // RequestId(3): 0.7937140887152377,
        // RequestId(4): 1.8802325581395347,
        // RequestId(5): 2.0582386363636367,
        // RequestId(6): 2.2488888888888887}

        // println!("expected_efficiencies: {:?}", expected_efficiencies);

        let efficiencies =
            dpk.calculate_dpk_efficiency(&candidate_requests, 0.01, KPSolverType::FPTAS, &pf);

        #[derive(Debug)]
        struct RequestDifference {
            #[allow(dead_code)]
            request_id: RequestId,
            #[allow(dead_code)]
            actual: f64,
            #[allow(dead_code)]
            expected: f64,
            #[allow(dead_code)]
            absolute_difference: f64,
        }

        let large_differences: Vec<_> = efficiencies
            .iter()
            .filter_map(|(r, e)| {
                let expected = expected_efficiencies.get(r).unwrap();
                if e.approx_eq(
                    *expected,
                    F64Margin {
                        epsilon: 1e-14,
                        ulps: 0,
                    },
                ) {
                    None
                } else {
                    Some(RequestDifference {
                        request_id: *r,
                        actual: *e,
                        expected: *expected,
                        absolute_difference: (e - expected).abs(),
                    })
                }
            })
            .collect();

        assert!(
            large_differences.is_empty(),
            "efficiencies were not as expected. For the following request, efficiencies deviated: {:?}",
            large_differences
        );
    }

    #[test]
    fn test_calc_best_alphas() {
        let (bids, candidate_requests, pf, dpk, _, _) = setup_dpk_test_with_pa(100);

        // println!("pf: {:?}", pf.contested_constraints().collect::<Vec<_>>());

        let expected_solution: HashMap<SegmentId, (AlphaIndex, f64)> = [
            (
                bids.clone(),
                vec![RequestId(1), RequestId(3), RequestId(4), RequestId(5)],
                (AlphaIndex(0), 1.05),
            ),
            (
                bids.clone(),
                vec![RequestId(1), RequestId(3), RequestId(4), RequestId(6)],
                (AlphaIndex(1), 1.1),
            ),
            (
                bids.clone(),
                vec![RequestId(2), RequestId(3), RequestId(6)],
                (AlphaIndex(2), 1.15),
            ),
            (
                bids,
                vec![RequestId(0), RequestId(1), RequestId(5)],
                (AlphaIndex(2), 1.15),
            ),
        ]
        .into_iter()
        .flat_map(
            |(bids, rids, val): (HashSet<BlockId>, Vec<RequestId>, (AlphaIndex, f64))| {
                bids.into_iter()
                    .map(move |bid| (SegmentId::from((bid, rids.iter().cloned().collect())), val))
            },
        )
        .collect();

        //let best_alphas_gurobi =
        //    dpk.calc_best_alphas(&candidate_requests, 0.01, KPSolverType::Gurobi, &pf);
        let best_alphas_fptas =
            dpk.calc_best_alphas(&candidate_requests, 0.01, KPSolverType::FPTAS, &pf);

        /*
        let difference_sols_gurobi = best_alphas_gurobi
            .iter()
            .filter(|(sid, (ai, val))| {
                expected_solution
                    .get(sid)
                    .map(|(expected_ai, expected_val)| expected_ai != ai || expected_val != val)
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();

        let difference_sols_fptas = best_alphas_fptas
            .iter()
            .filter(|(sid, (ai, val))| {
                expected_solution
                    .get(sid)
                    .map(|(expected_ai, expected_val)| expected_ai != ai || expected_val != val)
                    .unwrap_or(true)
            })
            .collect::<Vec<_>>();


        assert_eq!(
            expected_solution, best_alphas_gurobi,
            "alphas computed by gurobi knapsack solver are not as expected"
        );
        */

        assert_eq!(
            expected_solution, best_alphas_fptas,
            "alphas computed by fptas knapsack solver are not as expected"
        );
    }

    #[allow(clippy::type_complexity)]
    fn setup_dpk_test_with_pa(
        n_blocks: usize,
    ) -> (
        HashSet<BlockId>,
        HashMap<RequestId, Request>,
        ProblemFormulation<OptimalBudget>,
        EfficiencyBased,
        Schema,
        HashMap<BlockId, Block>,
    ) {
        let rdp_0: AccountingType = Rdp {
            eps_values: A3([1.0, 0.4, 0.4]),
        };
        let rdp_1: AccountingType = Rdp {
            eps_values: A3([0.4, 1.0, 0.4]),
        };
        let rdp_2: AccountingType = Rdp {
            eps_values: A3([0.4, 0.4, 1.0]),
        };
        let rdp_budget: AccountingType = Rdp {
            eps_values: A3([1.05, 1.1, 1.15]),
        };

        let bids = HashSet::from_iter((0..n_blocks).map(User));

        let schema = build_dummy_schema(rdp_budget.clone());
        let mut candidate_requests = build_dummy_requests_with_pa(&schema, 100, rdp_0.clone(), 7);
        let request_history = HashMap::new();
        let blocks = generate_blocks(0, n_blocks, rdp_budget);
        let block_comp_wrapper = BlockCompWrapper::BlockCompositionPartAttributesVariant(
            block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray),
        );

        // set custom request costs, which ensure that for each virtual block (identified by the
        // value of a0 and a1), the best solutions (alpha indices) are as follows, assuming
        // that more requests always gives more profit than less requests:
        // (0,0) => 1 or 2
        // (0,1) => 0 or 2
        // (0,2) => 1 or 2
        // (1,2) => 0 or 2
        // (the other two virtual blocks are not contested, so there will not be a segment for those)
        let request_costs: HashMap<RequestId, AccountingType> = [
            (0, &rdp_0),
            (1, &rdp_0),
            (2, &rdp_1),
            (3, &rdp_1),
            (4, &rdp_2),
            (5, &rdp_1),
            (6, &rdp_0),
        ]
        .into_iter()
        .map(|(id, rdp)| (RequestId(id), rdp.clone()))
        .collect();

        // set request costs individually
        #[allow(clippy::if_same_then_else)]
        for request in candidate_requests.values_mut() {
            request.request_cost = request_costs
                .get(&request.request_id)
                .expect("request cost not found")
                .clone();
            // set profit to prefer requests with higher id. Note that the property that more requests
            // is always better still holds, as we can allocate at most two requests per virtual block
            // and no single request has more >= 200 profit
            request.profit = 100 + request.request_id.0 as u64 * 10;
        }

        let pf: ProblemFormulation<OptimalBudget> = block_comp_wrapper.build_problem_formulation(
            &blocks,
            &candidate_requests,
            &request_history,
            &schema,
            &mut Vec::new(),
        );

        let dpk: EfficiencyBased = EfficiencyBased {
            request_cache: candidate_requests
                .values()
                .map(|r| (r.request_id, bids.clone()))
                .collect(),
            rng: StdRng::seed_from_u64(42),
            algo: EfficiencyBasedAlgo::Dpk {
                eta: 0.015,
                kp_solver: KPSolverType::FPTAS,
                num_threads: None,
                composition: get_dummy_composition(),
            },
        };
        (bids, candidate_requests, pf, dpk, schema, blocks)
    }

    #[allow(clippy::type_complexity)]
    fn setup_dpk_test_no_pa(
        n_blocks: usize,
    ) -> (
        HashSet<BlockId>,
        HashMap<RequestId, Request>,
        ProblemFormulation<OptimalBudget>,
        EfficiencyBased,
        Schema,
        HashMap<BlockId, Block>,
    ) {
        let rdp_0: AccountingType = Rdp {
            eps_values: A3([1.0, 0.4, 0.4]),
        };
        let rdp_1: AccountingType = Rdp {
            eps_values: A3([0.4, 1.0, 0.4]),
        };
        let rdp_2: AccountingType = Rdp {
            eps_values: A3([0.4, 0.4, 1.0]),
        };
        let rdp_budget: AccountingType = Rdp {
            eps_values: A3([1.05, 1.1, 1.15]),
        };

        let bids = HashSet::from_iter((0..n_blocks).map(User));

        let schema = build_dummy_schema(rdp_budget.clone());
        let mut candidate_requests = build_dummy_requests_no_pa(&schema, 100, rdp_0.clone(), 7);
        let request_history = HashMap::new();
        let blocks = generate_blocks(0, n_blocks, rdp_budget);
        let block_comp_wrapper =
            BlockCompWrapper::BlockCompositionVariant(block_composition::build_block_composition());

        // same request costs as in version with PA
        let request_costs: HashMap<RequestId, AccountingType> = [
            (0, &rdp_0),
            (1, &rdp_0),
            (2, &rdp_1),
            (3, &rdp_1),
            (4, &rdp_2),
            (5, &rdp_1),
            (6, &rdp_0),
        ]
        .into_iter()
        .map(|(id, rdp)| (RequestId(id), rdp.clone()))
        .collect();

        // set request costs individually
        #[allow(clippy::if_same_then_else)]
        for request in candidate_requests.values_mut() {
            request.request_cost = request_costs
                .get(&request.request_id)
                .expect("request cost not found")
                .clone();
            // set profit to prefer requests with higher id. Note that the property that more requests
            // is always better still holds, as we can allocate at most two requests per virtual block
            // and no single request has more >= 200 profit
            request.profit = 100 + request.request_id.0 as u64 * 10;
        }

        let pf: ProblemFormulation<OptimalBudget> = block_comp_wrapper.build_problem_formulation(
            &blocks,
            &candidate_requests,
            &request_history,
            &schema,
            &mut Vec::new(),
        );

        let dpk: EfficiencyBased = EfficiencyBased {
            request_cache: candidate_requests
                .values()
                .map(|r| (r.request_id, bids.clone()))
                .collect(),
            rng: StdRng::seed_from_u64(42),
            algo: EfficiencyBasedAlgo::Dpk {
                eta: 0.015,
                kp_solver: KPSolverType::FPTAS,
                num_threads: None,
                composition: get_dummy_composition(),
            },
        };
        (bids, candidate_requests, pf, dpk, schema, blocks)
    }
}
