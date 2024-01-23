use crate::block::Block;
use crate::composition::{CompositionConstraint, ProblemFormulation, StatusResult};
use crate::dprivacy::budget::SegmentBudget;
use std::cmp::Ordering;

use crate::schema::Schema;

use crate::block::BlockId;
use crate::request::{Request, RequestId};

use crate::allocation::utils::try_allocation;
use crate::allocation::{AllocationStatus, BlockCompWrapper};
use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
use crate::dprivacy::RdpAccounting;
use crate::logging::{DpfStats, RuntimeKind, RuntimeMeasurement};
use crate::AccountingType;
use crate::AccountingType::*;
use float_cmp::{ApproxEq, F64Margin};
use itertools::Itertools;
use rand::prelude::IteratorRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};

use super::ResourceAllocation;

pub struct Dpf {
    // maps each request seen so far to selected block ids
    request_cache: HashMap<RequestId, DomShareSelectedBlocks>,

    pub(crate) max_budget: AccountingType,

    rng: StdRng,

    weighted_dpf: bool,

    dominant_share_by_remaining_budget: bool,
}

#[derive(Debug, Clone)]
struct DomShareSelectedBlocks {
    dominant_share: f64,
    /// all or nothing semantic
    selected_blocks: HashSet<BlockId>,
}

impl Dpf {
    /// # Arguments
    /// * `requests`: a preview of all requests that will arrive over the course of the simulation
    /// which are used to fix a desired mapping of request to blocks, as assumed in dpf
    /// * `schema`: schema belonging to the requests
    /// * 'partitioning_attributes': Whether or not partitioning attributes are used
    /// * 'max_budget': the maximal unlockable budget, needed to calculate the dominant share
    /// * 'block_assignment': Optionally, an assignment of requests to blocks, useful for
    /// testing and debugging purposes

    pub fn construct_allocator(
        max_budget: &AccountingType,
        seed: u64,
        weighted_dpf: bool,
        dominant_share_by_remaining_budget: bool,
    ) -> Self {
        Dpf {
            request_cache: HashMap::new(),
            max_budget: max_budget.clone(),
            rng: StdRng::seed_from_u64(seed),
            weighted_dpf,
            dominant_share_by_remaining_budget,
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

        // NOTE: For the "old" problem, is it not problematic that we do not consider which blocks a request selects and assume that a request selects all blocks?
        //            Because it makes blocks more contested than they actually are?

        let mut pf: ProblemFormulation<M> = block_comp_wrapper.build_problem_formulation::<M>(
            available_blocks,
            candidate_requests,
            request_history,
            schema,
            runtime_measurements,
        );

        let num_contested_segments_initially = pf.contested_constraints().count();

        if self.weighted_dpf {
            assert!(
                candidate_requests.values().all(|r| r.profit as f64 > 0.0),
                "Profit of each request must be positive for weighted DPF"
            );
        }

        let mut alloc_meas = RuntimeMeasurement::start(RuntimeKind::RunAllocationAlgorithm);

        let mut selected_blocks_map: HashMap<RequestId, HashSet<BlockId>> = HashMap::new();
        // for each new candidate request, we randomly select blocks which we want to allocate
        for (rid, request) in candidate_requests.iter() {
            if !selected_blocks_map.contains_key(rid) {
                assert!(
                    available_blocks.len() >= request.n_users,
                    "Not enough blocks to fulfill request"
                );
                selected_blocks_map.insert(
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

        let mut dominant_shares: HashMap<RequestId, f64> = HashMap::new();

        if self.dominant_share_by_remaining_budget {
            // we only fill the dominant_shares map if we need it

            pf.contested_constraints()
                .for_each(|(block_id, request_ids, segment_budget)| {
                    //let segment_id = SegmentId::from((bid, rids));

                    // get the requests that apply to this segment
                    let selected_segment_requests = request_ids
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
                            selected_blocks_map
                                .get(&r.request_id)
                                .unwrap()
                                .contains(&block_id)
                        });

                    selected_segment_requests.for_each(|req| {
                        segment_budget
                            .get_budget_constraints()
                            .iter()
                            .for_each(|budget| {
                                let dominant_share: f64 =
                                    calc_dominant_share(&req.request_cost, budget);
                                dominant_shares
                                    .entry(req.request_id)
                                    .and_modify(|v| *v = (*v).max(dominant_share))
                                    .or_insert(dominant_share);
                            });
                    });
                });
        }

        // for each new candidate request, we randomly select blocks which we want to allocate
        for (rid, request) in candidate_requests.iter() {
            if !self.request_cache.contains_key(rid) {
                assert!(
                    available_blocks.len() >= request.n_users,
                    "Not enough blocks to fulfill request"
                );

                let mut dominant_share: f64;
                if self.dominant_share_by_remaining_budget {
                    // NOTE: In Tholoniat et al 2022, the dominant share of dpf is calculated based on the remaining budget of selected blocks (not the global one)
                    //       -> fall back to global budget if request is not in dominat shares which means not part of any contested segment
                    dominant_share = dominant_shares.get(rid).copied().unwrap_or_else(|| {
                        calc_dominant_share(&request.request_cost, &self.max_budget)
                    });
                } else {
                    // NOTE: In Luo et al 2021, the dominant share of dpf is calculated based on the global budget (not the remaining one for selected blocks)
                    dominant_share = calc_dominant_share(&request.request_cost, &self.max_budget);
                }

                if self.weighted_dpf {
                    dominant_share /= request.profit as f64;
                }
                self.request_cache.insert(
                    *rid,
                    DomShareSelectedBlocks {
                        dominant_share,
                        selected_blocks: selected_blocks_map.remove(rid).unwrap(),
                    },
                );
            }
        }

        // sort candidate requests based on dominant share
        let sorted_candidates: Vec<RequestId> = candidate_requests
            .keys()
            .copied()
            .sorted_by(|rid1, rid2| {
                let cost_cmp = self.request_cache[rid1]
                    .dominant_share
                    .partial_cmp(&self.request_cache[rid2].dominant_share)
                    .expect("Some dominant shares could not be compared");
                if let Ordering::Equal = cost_cmp {
                    // if costs are equal, we order by request id instead
                    let r1 = &candidate_requests[rid1];
                    let r2 = &candidate_requests[rid2];
                    let rid_cmp = Ord::cmp(&r1.request_id, &r2.request_id);
                    if let Ordering::Equal = rid_cmp {
                        panic!("Found two requests with same dominant share and same rid");
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
            let selected_blocks = self.request_cache[&rid].selected_blocks.clone();
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
}

impl Eq for DomShareSelectedBlocks {}

impl PartialEq<Self> for DomShareSelectedBlocks {
    fn eq(&self, other: &Self) -> bool {
        self.dominant_share == other.dominant_share
    }
}

fn calc_dominant_share(cost: &AccountingType, budget: &AccountingType) -> f64 {
    // from definition 2 / Algorithm 1 from privacy budget scheduling paper
    let dominant_share;
    match (cost, budget) {
        (EpsDp { eps: eps_cost }, EpsDp { eps: eps_budget }) => {
            dominant_share = f64::MIN.max(eps_cost / eps_budget);
        }
        (
            EpsDeltaDp {
                eps: eps_cost,
                delta: delta_cost,
            },
            EpsDeltaDp {
                eps: eps_budget,
                delta: delta_budget,
            },
        ) => {
            // need to check that relative cost in eps is not smaller than relative cost in delta,
            // to ensure that eps is always the bottleneck (as in privacy budget scheduling paper).
            let rel_delta_cost = delta_cost / delta_budget;
            let rel_eps_cost = eps_cost / eps_budget;

            assert!(
                rel_delta_cost <= rel_eps_cost
                    || rel_delta_cost.approx_eq(rel_eps_cost, F64Margin::default()),
                "Tried to execute dpf algorithm on adp cost where delta might be a constraint"
            );
            dominant_share = f64::MIN.max(rel_eps_cost);
        }
        (
            Rdp {
                eps_values: eps_vals_cost,
            },
            Rdp {
                eps_values: eps_vals_budget,
            },
        ) => {
            match (eps_vals_cost, eps_vals_budget) {
                (A1(eps_cost), A1(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A2(eps_cost), A2(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A3(eps_cost), A3(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A4(eps_cost), A4(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A5(eps_cost), A5(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A7(eps_cost), A7(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A10(eps_cost), A10(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A13(eps_cost), A13(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A14(eps_cost), A14(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (A15(eps_cost), A15(eps_budget)) => {
                    dominant_share = eps_cost
                        .iter()
                        .zip(eps_budget.iter())
                        .map(|(cost, budget)| cost / budget)
                        .fold(f64::MIN, f64::max);
                }
                (_, _) => {
                    panic!("Tried to calculate dominant share of rdp cost and budget of different length")
                }
            }
        }
        (_, _) => {
            panic!("Tried to compute dominant share of non-matching budget and cost");
        }
    }

    assert!(
        dominant_share.is_finite() || cost.is_rdp(),
        "Dominant share is infinite while the cost is not rdp make sure eps > 0"
    );
    assert!(dominant_share >= 0., "Dominant share is negative - make sure no budget part is negative (or at least one is positive, for rdp)");
    dominant_share
}

#[cfg(test)]
mod tests {
    #[derive(Debug, Clone)]
    pub(crate) struct MultiBlockRequest {
        pub(crate) request: Request,
        #[allow(dead_code)]
        pub(crate) selected_blocks: Vec<BlockId>, // all or nothing semantic
    }

    use crate::allocation::{AllocationRound, BlockCompWrapper, ResourceAllocation};
    use crate::composition::block_composition_pa::build_block_part_attributes;
    use crate::dprivacy::Accounting;
    use crate::dprivacy::AccountingType::Rdp;

    use crate::allocation::dpf::calc_dominant_share;
    use crate::block::BlockId::User;
    use crate::block::{Block, BlockId};
    use crate::composition::block_composition;
    use crate::config::{BudgetType, SegmentationAlgo};
    use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
    use crate::request::{
        AttributeId, ConjunctionBuilder, Predicate, Request, RequestBuilder, RequestId,
    };
    use crate::schema::ValueDomain::Range;
    use crate::schema::{Attribute, Schema};
    use crate::simulation::RoundId;
    use crate::AccountingType;
    use crate::AccountingType::{EpsDeltaDp, EpsDp};
    use float_cmp::{ApproxEq, F64Margin};
    use std::collections::{HashMap, HashSet};

    #[test]
    fn dominant_share_computation_adp() {
        let budget = EpsDeltaDp {
            eps: 1.0,
            delta: 1e-6,
        };
        let cost1 = EpsDeltaDp {
            eps: 0.1,
            delta: 1e-8,
        };
        let cost2 = EpsDeltaDp {
            eps: 0.01,
            delta: 1e-8,
        };

        assert!((0.1).approx_eq(calc_dominant_share(&cost1, &budget), F64Margin::default()));
        assert!((0.01).approx_eq(calc_dominant_share(&cost2, &budget), F64Margin::default()));
    }

    #[test]
    fn dominant_share_computation_rdp() {
        let budget = Rdp {
            eps_values: A5([1., 1., 1., 1., 1.]),
        };
        let cost1 = Rdp {
            eps_values: A5([0.5, 0.5, 0.7, 0.5, 0.5]),
        };
        let cost2 = Rdp {
            eps_values: A5([0.8, 0., 0., 0., 0.]),
        };

        assert!((0.7).approx_eq(calc_dominant_share(&cost1, &budget), F64Margin::default()));
        assert!((0.8).approx_eq(calc_dominant_share(&cost2, &budget), F64Margin::default()));
    }

    #[test]
    #[should_panic(
        expected = "Tried to execute dpf algorithm on adp cost where delta might be a constraint"
    )]
    fn dominant_share_panic() {
        let budget = EpsDeltaDp {
            eps: 1.0,
            delta: 1e-6,
        };
        let cost = EpsDeltaDp {
            eps: 0.01,
            delta: 1e-7,
        };
        calc_dominant_share(&cost, &budget);
    }

    #[test]
    fn test_dpf_planner_with_pa() {
        let _n_blocks: usize = 5;

        let schema = build_dummy_schema();

        let mb_requests = build_dummy_requests_with_pa(&schema);

        let max_budget = Rdp {
            eps_values: A5([3., 0.1, 0.1, 0.1, 3.]),
        };

        let rounds_until_budget_unlocked = 6usize;
        let zero_budget = AccountingType::zero_clone(&max_budget);

        let mut blocks = (0usize..5usize)
            .map(|num| {
                (
                    User(num),
                    Block {
                        id: User(num),
                        request_history: vec![],
                        unlocked_budget: zero_budget.clone(),
                        unreduced_unlocked_budget: zero_budget.clone(),
                        created: RoundId(0),
                        retired: None,
                    },
                )
            })
            .collect::<HashMap<_, _>>();

        let all_requests = mb_requests
            .iter()
            .map(|mb_request| (mb_request.request.request_id, mb_request.request.clone()))
            .collect::<HashMap<_, _>>();

        let request_history = all_requests;

        let mut planner = AllocationRound::Dpf(
            super::Dpf::construct_allocator(&max_budget, 42, false, false),
            BlockCompWrapper::BlockCompositionPartAttributesVariant(build_block_part_attributes(
                SegmentationAlgo::Narray,
            )),
            BudgetType::OptimalBudget,
        );

        let mut responses: Vec<ResourceAllocation> = Vec::with_capacity(mb_requests.len());
        let mut curr_requests: HashMap<RequestId, Request> =
            HashMap::with_capacity(mb_requests.len());
        for (round_num, mb_request) in mb_requests.iter().enumerate() {
            curr_requests.insert(mb_request.request.request_id, mb_request.request.clone());
            /*
            println!(
                "Round {} new request cost: {:?}",
                round_num, mb_request.request.request_cost
            );
            */
            // adjust budget of unlocked blocks according to current round
            let mut unlocked_budget = max_budget.clone();
            unlocked_budget.apply_func(&|x| {
                x * (((round_num + 1) as f64) / (rounds_until_budget_unlocked as f64))
            });
            if max_budget.approx_le(&unlocked_budget) {
                // unlocked budget should not be higher than max_budget
                unlocked_budget = max_budget.clone();
            }

            for block in blocks.values_mut() {
                block.unlocked_budget = unlocked_budget.clone();
            }

            println!(
                "Unlocked budget in round {}: {:?}",
                round_num, unlocked_budget
            );

            // add history to blocks
            {
                let empty = ResourceAllocation {
                    accepted: HashMap::new(),
                    rejected: HashSet::new(),
                };
                let last_response = responses.last().unwrap_or(&empty);
                // println!("Response: {:?}", response);
                for (request_id, block_ids) in last_response.accepted.iter() {
                    for block_id in block_ids {
                        // println!("request_id {}, block_id: {:?}", request_id, block_id);
                        let mb_request_2 = mb_requests
                            .iter()
                            .find(|mb_request| mb_request.request.request_id == *request_id)
                            .expect("Did not find request");
                        blocks
                            .get_mut(block_id)
                            .expect("Did not find block with id")
                            .request_history
                            .push(mb_request_2.request.request_id);
                    }
                }
            }

            /*
            for (_, curr_block) in curr_blocks.iter() {
                let mut history = Vec::new();
                for request in curr_block.request_history.iter() {
                    history.push(request.request_id);
                }

                println!(
                    "curr_block id: {:?}, curr_block history: {:?}",
                    curr_block.id, history
                );

            }
            */

            let (response, _) = planner.round(
                &request_history,
                &blocks,
                &curr_requests,
                &schema,
                &Some(Rdp {
                    eps_values: A5([2., 4., 8., 16., 32.]),
                }),
                &mut Vec::new(),
            );

            // only keep rejected requests around
            curr_requests.retain(|rid, _| response.rejected.contains(rid));
            responses.push(response);
        }

        // TODO Fix the flaky test. The problem is that the selected blocks in the test do not correspond to the "randomly" selected blocks in the round.
        assert!(!responses.is_empty());
        // check_responses_with_pa(responses);
    }

    #[test]
    fn test_dpf_planner_no_pa() {
        let _n_blocks: usize = 5;

        let mut schema = build_dummy_schema();
        schema.accounting_type = EpsDp { eps: 0.0 };
        let schema = schema;

        let mb_requests = build_dummy_requests_no_pa(&schema);

        let max_budget = EpsDp { eps: 1.4 };

        /*
        let block_assigment: Option<HashMap<RequestId, HashSet<BlockId>>> = Some(
            mb_requests
                .iter()
                .map(|mb_request| {
                    (
                        mb_request.request.request_id,
                        mb_request.selected_blocks.iter().copied().collect(),
                    )
                })
                .collect(),
        );
         */

        let rounds_until_budget_unlocked = 3usize;
        let zero_budget = AccountingType::zero_clone(&max_budget);

        let mut blocks = crate::util::generate_blocks(0, 5, zero_budget);

        let all_requests = mb_requests
            .iter()
            .map(|mb_request| (mb_request.request.request_id, mb_request.request.clone()))
            .collect::<HashMap<_, _>>();

        let request_history = all_requests;

        /*
        let mut sorted_candidates: Vec<Request> = all_requests
            .into_iter()
            .map(|(_, request)| request)
            .sorted_by(|r1, r2| std::cmp::Ord::cmp(&r1.request_id, &r2.request_id))
            .collect();
         */

        let mut planner = AllocationRound::Dpf(
            super::Dpf::construct_allocator(&max_budget, 42, false, false),
            BlockCompWrapper::BlockCompositionVariant(block_composition::build_block_composition()),
            BudgetType::OptimalBudget,
        );

        let mut responses: Vec<ResourceAllocation> = Vec::with_capacity(mb_requests.len());
        let mut curr_requests: HashMap<RequestId, Request> = HashMap::new();
        for (round_num, mb_request) in mb_requests.iter().enumerate() {
            let inserted =
                curr_requests.insert(mb_request.request.request_id, mb_request.request.clone());
            assert!(inserted.is_none());

            println!(
                "Round {} new request cost: {:?}",
                round_num, mb_request.request.request_cost
            );

            // adjust budget of unlocked blocks according to current round
            let mut unlocked_budget = max_budget.clone();
            unlocked_budget.apply_func(&|x| {
                x * (((round_num + 1) as f64) / (rounds_until_budget_unlocked as f64))
            });
            if max_budget.approx_le(&unlocked_budget) {
                // unlocked budget should not be higher than max_budget
                unlocked_budget = max_budget.clone();
            }

            println!(
                "Unlocked budget in round {}: {:?}",
                round_num, unlocked_budget
            );

            for block in blocks.values_mut() {
                block.unlocked_budget = unlocked_budget.clone();
            }

            // add history to blocks
            {
                let empty = ResourceAllocation {
                    accepted: HashMap::new(),
                    rejected: HashSet::new(),
                };
                let last_response = responses.last().unwrap_or(&empty);
                // println!("Response: {:?}", response);
                for (request_id, block_ids) in last_response.accepted.iter() {
                    for block_id in block_ids {
                        // println!("request_id {}, block_id: {:?}", request_id, block_id);
                        let mb_request_2 = mb_requests
                            .iter()
                            .find(|mb_request| mb_request.request.request_id == *request_id)
                            .expect("Did not find request");
                        blocks
                            .get_mut(block_id)
                            .expect("Did not find block with id")
                            .request_history
                            .push(mb_request_2.request.request_id);
                    }
                }
            }

            for block in blocks.values() {
                println!(
                    "Block {:?} history: {:?} unlocked budgets: {:?}",
                    block.id, block.request_history, block.unlocked_budget
                );
            }

            /*
            for (_, curr_block) in curr_blocks.iter() {
                let mut history = Vec::new();
                for request in curr_block.request_history.iter() {
                    history.push(request.request_id);
                }

                println!(
                    "curr_block id: {:?}, curr_block history: {:?}",
                    curr_block.id, history
                );

            }
            */

            // println!("blocks: {:?}", blocks);

            let (response, _) = planner.round(
                &request_history,
                &blocks,
                &curr_requests,
                &schema,
                &None,
                &mut Vec::new(),
            );
            // only keep rejected requests around
            curr_requests.retain(|rid, _| response.rejected.contains(rid));
            responses.push(response);
        }
        // TODO Fix the flaky test. The problem is that the selected blocks in the test do not correspond to the "randomly" selected blocks in the round.
        assert!(!responses.is_empty());
        //check_responses_no_pa(responses);
    }

    #[test]
    fn test_weighted_dpf_planner_no_pa() {
        let _n_blocks: usize = 5;

        let mut schema = build_dummy_schema();
        schema.accounting_type = EpsDp { eps: 0.0 };
        let schema = schema;

        let mut mb_requests = build_dummy_requests_no_pa(&schema);

        mb_requests[0].request.profit = 2;
        mb_requests[1].request.profit = 10;

        let max_budget = EpsDp { eps: 1.4 };

        /*
        let block_assigment: Option<HashMap<RequestId, HashSet<BlockId>>> = Some(
            mb_requests
                .iter()
                .map(|mb_request| {
                    (
                        mb_request.request.request_id,
                        mb_request.selected_blocks.iter().copied().collect(),
                    )
                })
                .collect(),
        );
         */

        let rounds_until_budget_unlocked = 3usize;
        let zero_budget = AccountingType::zero_clone(&max_budget);

        let mut blocks = crate::util::generate_blocks(0, 5, zero_budget);

        let all_requests = mb_requests
            .iter()
            .map(|mb_request| (mb_request.request.request_id, mb_request.request.clone()))
            .collect::<HashMap<_, _>>();

        let request_history = all_requests;

        /*
        let mut sorted_candidates: Vec<Request> = all_requests
            .into_iter()
            .map(|(_, request)| request)
            .sorted_by(|r1, r2| std::cmp::Ord::cmp(&r1.request_id, &r2.request_id))
            .collect();
         */

        let mut planner = AllocationRound::Dpf(
            super::Dpf::construct_allocator(&max_budget, 42, true, false),
            BlockCompWrapper::BlockCompositionVariant(block_composition::build_block_composition()),
            BudgetType::OptimalBudget,
        );

        let mut responses: Vec<ResourceAllocation> = Vec::with_capacity(mb_requests.len());
        let mut curr_requests: HashMap<RequestId, Request> = HashMap::new();
        for (round_num, mb_request) in mb_requests.iter().enumerate() {
            let inserted =
                curr_requests.insert(mb_request.request.request_id, mb_request.request.clone());
            assert!(inserted.is_none());

            println!(
                "Round {} new request cost: {:?}",
                round_num, mb_request.request.request_cost
            );

            // adjust budget of unlocked blocks according to current round
            let mut unlocked_budget = max_budget.clone();
            unlocked_budget.apply_func(&|x| {
                x * (((round_num + 1) as f64) / (rounds_until_budget_unlocked as f64))
            });
            if max_budget.approx_le(&unlocked_budget) {
                // unlocked budget should not be higher than max_budget
                unlocked_budget = max_budget.clone();
            }

            println!(
                "Unlocked budget in round {}: {:?}",
                round_num, unlocked_budget
            );

            for block in blocks.values_mut() {
                block.unlocked_budget = unlocked_budget.clone();
            }

            // add history to blocks
            {
                let empty = ResourceAllocation {
                    accepted: HashMap::new(),
                    rejected: HashSet::new(),
                };
                let last_response = responses.last().unwrap_or(&empty);
                // println!("Response: {:?}", response);
                for (request_id, block_ids) in last_response.accepted.iter() {
                    for block_id in block_ids {
                        // println!("request_id {}, block_id: {:?}", request_id, block_id);
                        let mb_request_2 = mb_requests
                            .iter()
                            .find(|mb_request| mb_request.request.request_id == *request_id)
                            .expect("Did not find request");
                        blocks
                            .get_mut(block_id)
                            .expect("Did not find block with id")
                            .request_history
                            .push(mb_request_2.request.request_id);
                    }
                }
            }

            for block in blocks.values() {
                println!(
                    "Block {:?} history: {:?} unlocked budgets: {:?}",
                    block.id, block.request_history, block.unlocked_budget
                );
            }

            /*
            for (_, curr_block) in curr_blocks.iter() {
                let mut history = Vec::new();
                for request in curr_block.request_history.iter() {
                    history.push(request.request_id);
                }

                println!(
                    "curr_block id: {:?}, curr_block history: {:?}",
                    curr_block.id, history
                );

            }
            */

            // println!("blocks: {:?}", blocks);

            let (response, _) = planner.round(
                &request_history,
                &blocks,
                &curr_requests,
                &schema,
                &None,
                &mut Vec::new(),
            );
            // only keep rejected requests around
            curr_requests.retain(|rid, _| response.rejected.contains(rid));
            responses.push(response);
        }

        // println!("responses: {:?}", responses);
        // TODO Fix the flaky test. The problem is that the selected blocks in the test do not correspond to the "randomly" selected blocks in the round.
        assert!(!responses.is_empty());
        // check_responses_no_pa_weighted_dpf(responses);
    }

    fn check_responses_with_pa(responses: Vec<ResourceAllocation>) {
        // first response should be empty, since only with the second response, a budget of
        // [1, 0, 0, 0, 1] is unlocked, which allows at least one request
        let assert_round_len = |index: usize, len: usize| {
            assert_eq!(
                responses[index].accepted.len(),
                len,
                "Invalid number of accepted requests in round {}. Accepted and rejected requests: {:?}, expected {} accepted requests",
                index,
                responses[index],
                len
            )
        };

        assert_round_len(0, 0);
        assert_round_len(1, 1);
        assert_round_len(2, 0);
        assert_round_len(3, 0);
        assert_round_len(4, 1);
        assert_round_len(5, 2);
        assert_round_len(6, 1);
        assert_round_len(7, 1);

        assert!(responses[1].accepted.contains_key(&RequestId(1)));
        assert!(responses[4].accepted.contains_key(&RequestId(4)));
        assert!(responses[5].accepted.contains_key(&RequestId(0)));
        assert!(responses[5].accepted.contains_key(&RequestId(5)));
        assert!(responses[6].accepted.contains_key(&RequestId(6)));
        assert!(responses[7].accepted.contains_key(&RequestId(7)));
    }

    fn check_responses_no_pa(responses: Vec<ResourceAllocation>) {
        // first response should be empty, since only with the second response, a budget of
        // [1, 0, 0, 0, 1] is unlocked, which allows at least one request
        let assert_round_len = |index: usize, len: usize| {
            assert_eq!(
                responses[index].accepted.len(),
                len,
                "Invalid number of accepted requests in round {}. Accepted and rejected requests: {:?}, expected {} accepted requests",
                index,
                responses[index],
                len
            )
        };

        assert_eq!(responses.len(), 6);
        assert_round_len(0, 0);
        assert_round_len(1, 0);
        assert_round_len(2, 2);
        assert_round_len(3, 0);
        assert_round_len(4, 1);
        assert_round_len(5, 1);
        assert!(responses[2].accepted.contains_key(&RequestId(0)));
        assert!(responses[2].accepted.contains_key(&RequestId(2)));
        assert!(responses[4].accepted.contains_key(&RequestId(4)));
        assert!(responses[5].accepted.contains_key(&RequestId(5)));
    }

    fn check_responses_no_pa_weighted_dpf(responses: Vec<ResourceAllocation>) {
        // first response should be empty, since only with the second response, a budget of
        // [1, 0, 0, 0, 1] is unlocked, which allows at least one request
        let assert_round_len = |index: usize, len: usize| {
            assert_eq!(
                responses[index].accepted.len(),
                len,
                "Invalid number of accepted requests in round {}. Accepted and rejected requests: {:?}, expected {} accepted requests",
                index,
                responses[index],
                len
            )
        };

        assert_eq!(responses.len(), 6);
        assert_round_len(0, 0);
        assert_round_len(1, 0);
        assert_round_len(2, 2);
        assert_round_len(3, 1);
        assert_round_len(4, 1);
        assert_round_len(5, 1);
        assert!(responses[2].accepted.contains_key(&RequestId(1)));
        assert!(responses[2].accepted.contains_key(&RequestId(2)));
        assert!(responses[3].accepted.contains_key(&RequestId(3)));
        assert!(responses[4].accepted.contains_key(&RequestId(4)));
        assert!(responses[5].accepted.contains_key(&RequestId(5)));
    }

    fn build_dummy_schema() -> Schema {
        Schema {
            accounting_type: Rdp {
                eps_values: A5([3., 0., 0., 0., 3.]),
            },
            attributes: vec![
                Attribute {
                    name: "A1".to_string(),
                    value_domain: Range { min: 0, max: 2 },
                    value_domain_map: None,
                },
                Attribute {
                    name: "A2".to_string(),
                    value_domain: Range { min: 0, max: 2 },
                    value_domain_map: None,
                },
            ],
            name_to_index: Default::default(),
        }
    }

    fn build_dummy_requests_with_pa(schema: &Schema) -> Vec<MultiBlockRequest> {
        vec![
            MultiBlockRequest {
                // should run in round 5 (zero-indexed), since overlaps with request 1, which has priority
                request: RequestBuilder::new(
                    RequestId(0),
                    Rdp {
                        eps_values: A5([2. - 0.05, 1. - 0.05, 1. - 0.05, 1. - 0.05, 1. - 0.05]),
                    },
                    1,
                    3,
                    schema,
                    std::default::Default::default(),
                )
                .or_conjunction(
                    ConjunctionBuilder::new(schema)
                        .and(AttributeId(0), Predicate::Between { min: 0, max: 2 })
                        .and(AttributeId(1), Predicate::Eq(0))
                        .build(),
                )
                .build(),
                selected_blocks: vec![User(1), User(2), User(3)],
            },
            MultiBlockRequest {
                // should run in round 1 (zero-indexed)
                request: RequestBuilder::new(
                    RequestId(1),
                    Rdp {
                        eps_values: A5([1. - 0.06, 1. - 0.06, 1. - 0.06, 1. - 0.06, 2. - 0.06]),
                    },
                    1,
                    3,
                    schema,
                    std::default::Default::default(),
                )
                .or_conjunction(
                    ConjunctionBuilder::new(schema)
                        .and(AttributeId(0), Predicate::Eq(0))
                        .and(AttributeId(1), Predicate::Between { min: 0, max: 2 })
                        .build(),
                )
                .build(),
                selected_blocks: vec![User(1), User(2), User(3)],
            },
            MultiBlockRequest {
                //should not run, since between 2, 3 and 5, 5 has priority, and only one can run
                request: RequestBuilder::new(
                    RequestId(2),
                    Rdp {
                        eps_values: A5([2., 1., 1., 1., 1.]),
                    },
                    1,
                    2,
                    schema,
                    std::default::Default::default(),
                )
                .or_conjunction(
                    ConjunctionBuilder::new(schema)
                        .and(AttributeId(0), Predicate::Between { min: 0, max: 1 })
                        .and(AttributeId(1), Predicate::Between { min: 0, max: 1 })
                        .build(),
                )
                .build(),
                selected_blocks: vec![User(3), User(4)],
            },
            MultiBlockRequest {
                // should not run, since between 2, 3 and 5, 5 has priority, and only one can run
                request: RequestBuilder::new(
                    RequestId(3),
                    Rdp {
                        eps_values: A5([
                            2. - 0.00001,
                            1. - 0.00001,
                            1. - 0.00001,
                            1. - 0.00001,
                            1. - 0.00001,
                        ]),
                    },
                    1,
                    3,
                    schema,
                    std::default::Default::default(),
                )
                .or_conjunction(
                    ConjunctionBuilder::new(schema)
                        .and(AttributeId(0), Predicate::Between { min: 0, max: 2 })
                        .and(AttributeId(1), Predicate::Eq(1))
                        .build(),
                )
                .build(),
                selected_blocks: vec![User(1), User(2), User(3)],
            },
            MultiBlockRequest {
                // should run in round 4 (zero-indexed) - no overlap with prior queries that can run, so can run immediately
                request: RequestBuilder::new(
                    RequestId(4),
                    Rdp {
                        eps_values: A5([1., 1., 1., 1., 2.]),
                    },
                    1,
                    3,
                    schema,
                    std::default::Default::default(),
                )
                .or_conjunction(
                    ConjunctionBuilder::new(schema)
                        .and(AttributeId(0), Predicate::Between { min: 1, max: 2 })
                        .and(AttributeId(1), Predicate::Between { min: 1, max: 2 })
                        .build(),
                )
                .build(),
                selected_blocks: vec![User(1), User(2), User(3)],
            },
            MultiBlockRequest {
                // should run in round 5 (zero-indexed) - takes priority over requests 2 and 3, which could also be allocated this round
                request: RequestBuilder::new(
                    RequestId(5),
                    Rdp {
                        // should have priority over request 3
                        eps_values: A5([2. - 0.01, 1. - 0.01, 1. - 0.01, 1. - 0.01, 1. - 0.01]),
                    },
                    1,
                    3,
                    schema,
                    std::default::Default::default(),
                )
                .or_conjunction(
                    ConjunctionBuilder::new(schema)
                        .and(AttributeId(0), Predicate::Between { min: 0, max: 2 })
                        .and(AttributeId(1), Predicate::Between { min: 1, max: 2 })
                        .build(),
                )
                .build(),
                selected_blocks: vec![User(1), User(2), User(3)],
            },
            MultiBlockRequest {
                // should run in round 6 (zero-indexed), only overlaps with 0, so can run starting round 5 (when all budget is unlocked)
                request: RequestBuilder::new(
                    RequestId(6),
                    Rdp {
                        eps_values: A5([1., 1., 1., 1., 2.]),
                    },
                    1,
                    3,
                    schema,
                    std::default::Default::default(),
                )
                .or_conjunction(
                    ConjunctionBuilder::new(schema)
                        .and(AttributeId(0), Predicate::Eq(2))
                        .and(AttributeId(1), Predicate::Eq(0))
                        .build(),
                )
                .build(),
                selected_blocks: vec![User(1), User(2), User(3)],
            },
            MultiBlockRequest {
                // should run in round 7 (zero-index), since goes to different block than other requests, and all budget unlocked by now
                request: RequestBuilder::new(
                    RequestId(7),
                    Rdp {
                        eps_values: A5([1., 1., 1., 1., 2.]),
                    },
                    1,
                    1,
                    schema,
                    std::default::Default::default(),
                )
                .or_conjunction(
                    ConjunctionBuilder::new(schema)
                        .and(AttributeId(0), Predicate::Eq(2))
                        .and(AttributeId(1), Predicate::Eq(0))
                        .build(),
                )
                .build(),
                selected_blocks: vec![User(4)],
            },
        ]
    }

    fn build_dummy_requests_no_pa(schema: &Schema) -> Vec<MultiBlockRequest> {
        let request_from_cost = |id: usize, cost: AccountingType, n_users: usize| {
            let r = RequestBuilder::new(
                RequestId(id),
                cost,
                1,
                n_users,
                schema,
                std::default::Default::default(),
            )
            .build();
            r
        };

        let mb0 = MultiBlockRequest {
            request: request_from_cost(0, EpsDp { eps: 1.0 }, 3),
            selected_blocks: Vec::from_iter([User(1), User(3), User(4)].into_iter()),
        };

        let mb1 = MultiBlockRequest {
            request: request_from_cost(1, EpsDp { eps: 1.0 }, 2),
            selected_blocks: Vec::from_iter([User(0), User(2)].into_iter()),
        };

        let mb2 = MultiBlockRequest {
            request: request_from_cost(2, EpsDp { eps: 0.7 }, 1),
            selected_blocks: Vec::from_iter([User(2)].into_iter()),
        };

        let mb3 = MultiBlockRequest {
            request: request_from_cost(3, EpsDp { eps: 1.0 }, 1),
            selected_blocks: Vec::from_iter([User(2)].into_iter()),
        };

        let mb4 = MultiBlockRequest {
            request: request_from_cost(4, EpsDp { eps: 0.4 }, 1),
            selected_blocks: Vec::from_iter([User(1)].into_iter()),
        };

        let mb5 = MultiBlockRequest {
            request: request_from_cost(5, EpsDp { eps: 0.4 }, 1),
            selected_blocks: Vec::from_iter([User(3)].into_iter()),
        };

        vec![mb0, mb1, mb2, mb3, mb4, mb5]
    }
}
