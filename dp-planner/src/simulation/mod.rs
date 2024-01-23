//! This module most importantly offers the method [run_simulation] which contains high-level
//! management code which is the same for all types of allocation, as well as some helper methods.
//! Methods from the [logging module](crate::logging) are also called here to produce the
//! various logs.

pub mod util;

use log::{info, trace};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ops::{Add, AddAssign, Sub};
use std::time::Instant;

use crate::allocation::AllocationStatus;
use crate::block::BlockId;
use crate::config::OutputPaths;
use crate::logging::{RuntimeKind, RuntimeMeasurement};
use crate::{
    allocation::{AllocationRound, ResourceAllocation},
    block::Block,
    config::Budget,
    global_reduce_alphas, logging,
    request::{Request, RequestId},
    schema::Schema,
    AlphaReductionsResult, Cli, Rdp, RdpAccounting,
};
use serde::{Deserialize, Serialize};

/// Contains various information needed to run the simulation. See the documentation about the
/// individual fields for more information.
pub struct SimulationConfig {
    /// The batching strategy define whether we will batch with a fixed batch size or batch based on the created field present on each request.
    pub(crate) batching_strategy: BatchingStrategy,
    /// Defines in how many rounds a request is considered for allocation at most. Note that if this
    /// number is greater than 1, more than batch_size many requests may be present in a single
    /// round, potentially slowing down allocation considerably. If a request still has "rounds left"
    /// after the whole simulation, it is counted as a request that is rejected, but is still available
    /// for allocation later. The information how many times it was already considered is lost in
    /// this case.
    pub(crate) timeout_rounds: usize,
    /// The same as [budget_config](struct.SimulationConfig.html#structfield.budget_config), except
    /// that the alphas are not reduced from global alpha reduction.
    pub(crate) unreduced_budget_config: Budget,
    /// The round in which the simulation starts. This is important if the [budget](Budget) is an
    /// [unlocking budget](Budget::UnlockingBudget), as how much budget is unlocked depends on the
    /// current round and on when a block was created.
    pub(crate) start_round: RoundId,
    /// The paths for the various outputs during/at the end of the simulation. See [OutputPaths] for
    /// more details.
    pub(crate) output_paths: OutputPaths,
    /// Whether we want to include request rejections that were not final (only matters if
    /// [keep rejected requests](struct.SimulationConfig.html#structfield.keep_rejected_requests)
    /// is enabled)
    pub(crate) log_nonfinal_rejections: bool,
}

/// Also contains information to run the simulation like [SimulationConfig], but in contrast
/// this information might be changed during the simulation due to global alpha reduction, while
/// [SimulationConfig] is not mutable.
#[derive(Clone)]
pub struct ConfigAndSchema {
    /// The schema matching the passed requests
    pub(crate) schema: Schema,
    /// The input given via the command line, possibly modified by
    /// [global alpha reduction](crate::global_reduce_alphas).
    pub(crate) config: Cli,
}

/// Contains various datastructures containing requests, which are necessary to run the simulation
pub struct RequestCollection {
    pub(crate) sorted_candidates: Vec<Request>,
    pub(crate) request_history: HashMap<RequestId, Request>,
    pub(crate) rejected_requests: BTreeMap<RequestId, Request>,
    pub(crate) accepted: BTreeMap<RequestId, HashSet<BlockId>>,
    pub(crate) remaining_requests: BTreeMap<RequestId, Request>,
}

#[derive(Deserialize, Debug, Clone, Copy, Hash, Serialize)]
pub enum BatchingStrategy {
    ByBatchSize(usize),
    ByRequestCreated,
}

#[derive(Deserialize, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct RoundId(pub usize);

impl AddAssign<usize> for RoundId {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs
    }
}

impl Add for RoundId {
    type Output = RoundId;

    fn add(self, rhs: Self) -> Self::Output {
        RoundId(self.0 + rhs.0)
    }
}

impl Sub for RoundId {
    type Output = RoundId;

    fn sub(self, rhs: Self) -> Self::Output {
        RoundId(self.0 - rhs.0)
    }
}

impl RequestCollection {
    /// This method resets the unlocked budget to contain again all alphas
    pub fn unreduce_alphas(&mut self) {
        for request in self.sorted_candidates.iter_mut() {
            request.unreduce_alphas();
        }
        for request in self.request_history.values_mut() {
            request.unreduce_alphas();
        }
        for request in self.rejected_requests.values_mut() {
            request.unreduce_alphas();
        }
        for request in self.remaining_requests.values_mut() {
            request.unreduce_alphas();
        }
    }
}

pub fn run_simulation(
    request_collection: &mut RequestCollection,
    blocks: &mut HashMap<BlockId, Block>,
    allocator: &mut AllocationRound,
    simulation_config: &SimulationConfig,
    config_and_schema: &mut ConfigAndSchema,
) {
    assert!(
        simulation_config.timeout_rounds > 0,
        "timeout_rounds must be > 0"
    );
    // initialize logger
    let mut req_logger =
        csv::Writer::from_path(&simulation_config.output_paths.req_log_output_path)
            .expect("Couldn't open request logger output file");
    let mut round_logger =
        csv::Writer::from_path(&simulation_config.output_paths.round_log_output_path)
            .expect("Couldn't open round logger output file");

    let mut runtime_measurements: Vec<RuntimeMeasurement> = Vec::new();
    let mut runtime_logger =
        csv::Writer::from_path(&simulation_config.output_paths.runtime_log_output_path)
            .expect("Couldn't open runtime logger output file");

    // The current round in the simulation. Note that for online allocation methods (like greedy),
    // one simulation round might correspond to multiple allocation rounds.
    let mut simulation_round = simulation_config.start_round;
    let mut request_start_rounds: HashMap<RequestId, RoundId> = HashMap::new();

    // for each round > start_round, contains block ids of blocks that joined this round
    // while all blocks that are currently part of the system are under the current_round
    let mut blocks_per_round: BTreeMap<RoundId, Vec<(BlockId, Block)>> = BTreeMap::new();

    let block_ids_to_remove: Vec<BlockId> = blocks
        .iter()
        .filter(|(_block_id, block)| block.created >= simulation_round)
        .map(|(block_id, _block)| *block_id)
        .collect();

    for block_id in block_ids_to_remove {
        let block = blocks.remove(&block_id).unwrap();
        blocks_per_round
            .entry(block.created)
            .or_default()
            .push((block_id, block));
    }
    // blocks: contains all blocks with a created field < simulation_round
    // blocks_per_round contains all blocks with a created field >= simulation_round (grouped by round)

    // check that candidate requests each need >= 1 block
    assert!(request_collection
        .sorted_candidates
        .iter()
        .all(|req| req.n_users > 0));

    let orig_n_candidates = request_collection.sorted_candidates.len();
    let orig_history_size = request_collection.request_history.len();
    let orig_config_and_schema = config_and_schema.clone();

    let mut curr_candidate_requests: BTreeMap<RequestId, Request> = BTreeMap::new();

    let mut round_instant: Option<Instant> = None;
    let start_instant = Instant::now();
    let rounds_instant: Instant = Instant::now();

    let mut total_profit = 0u64;

    loop {
        let mut round_total_meas = RuntimeMeasurement::start(RuntimeKind::TotalRound);
        let mut round_setup_meas = RuntimeMeasurement::start(RuntimeKind::RoundSetup);

        assert!(
            runtime_measurements.is_empty(),
            "runtime_measurements should be empty at the beginning of each round"
        );

        util::log_remaining_requests(
            &*request_collection,
            simulation_config,
            orig_n_candidates,
            &mut round_instant,
            &start_instant,
        );

        // Check if we have already processed all request - in which case we can stop the simulation.
        if request_collection.sorted_candidates.is_empty() {
            break;
        }

        // remove global alpha reduction from prior round
        *config_and_schema = orig_config_and_schema.clone();
        let budget = config_and_schema.config.total_budget().budget();
        request_collection.unreduce_alphas();
        for (_, req) in curr_candidate_requests.iter_mut() {
            req.unreduce_alphas();
        }
        for block in blocks.values_mut() {
            block.unreduce_alphas();
        }

        // move the new batch of requests from request.collection.sorted_candidates to curr_candidate_requests
        let (newly_available_requests, is_final_round) = util::pre_round_request_batch_update(
            simulation_round,
            simulation_config.batching_strategy,
            request_collection,
            &mut curr_candidate_requests,
            &mut request_start_rounds,
        );

        // activate and retire blocks so that blocks only contain blocks that are active in the current round
        util::pre_round_blocks_update(simulation_round, blocks, &mut blocks_per_round);

        // update the budget of the active blocks in `blocks` (budget unlocking)
        util::update_block_unlocked_budget(
            blocks,
            config_and_schema.config.allocation().budget_config(),
            &simulation_config.unreduced_budget_config,
            newly_available_requests.len(),
            simulation_round,
        );

        // Determine which alphas are actually needed, if we have rdp
        let global_alpha_red_res: Option<AlphaReductionsResult> = if budget.is_rdp()
            && config_and_schema
                .config
                .total_budget()
                .global_alpha_reduction()
        {
            let res = global_reduce_alphas(
                &mut config_and_schema.config,
                &budget,
                &mut config_and_schema.schema,
                &mut curr_candidate_requests,
                &mut request_collection.request_history,
                blocks,
            );
            Some(res)
        } else {
            None
        };

        // Adjust max budget used to calculate dominant share
        let budget = config_and_schema.config.total_budget().budget();
        if let AllocationRound::Dpf(dpf, _, _) = allocator {
            dpf.max_budget = budget;
        }

        trace!(
            "Current candidate requests: {:?}",
            &curr_candidate_requests.keys()
        );

        info!("Starting allocation in round={:?} n_blocks={:?} n_candidate_requests={:?}   n_request_allocated_until_now={:?}", simulation_round, blocks.len(), curr_candidate_requests.len(), request_collection.request_history.len());

        runtime_measurements.push(round_setup_meas.stop());

        let mut round_allocation_meas = RuntimeMeasurement::start(RuntimeKind::RoundAllocation);

        // Run a round of allocation
        let (assignment, allocation_status): (ResourceAllocation, AllocationStatus) = allocator
            .round(
                &request_collection.request_history,
                blocks,
                &curr_candidate_requests
                    .iter()
                    .map(|(rid, req)| (*rid, req.clone()))
                    .collect(),
                &config_and_schema.schema,
                &config_and_schema
                    .config
                    .total_budget()
                    .alphas()
                    .map(|alphas| Rdp { eps_values: alphas }),
                &mut runtime_measurements,
            );
        runtime_measurements.push(round_allocation_meas.stop());
        runtime_measurements.push(round_total_meas.stop());

        util::process_round_results(
            request_collection,
            blocks,
            simulation_config,
            &mut req_logger,
            &mut curr_candidate_requests,
            simulation_round,
            &mut total_profit,
            is_final_round,
            &assignment,
            &*config_and_schema,
            &allocation_status,
            &request_start_rounds,
        );

        logging::write_round_log_row(
            &mut round_logger,
            simulation_round,
            simulation_config,
            config_and_schema,
            &orig_config_and_schema,
            newly_available_requests,
            BTreeSet::from_iter(assignment.accepted.keys().copied()),
            allocation_status,
            config_and_schema.config.allocation(),
            &request_collection.request_history,
            &*blocks,
            global_alpha_red_res,
        );

        logging::write_runtime_log(
            &mut runtime_logger,
            simulation_round,
            &mut runtime_measurements,
        );

        simulation_round += 1;
    }

    // Add all blocks again to blocks (necessary if there were no candidate requests, or not enough
    // rounds to add all blocks)
    for (_, block) in blocks_per_round.into_values().flatten() {
        let inserted = blocks.insert(block.id, block);
        assert!(inserted.is_none());
    }

    util::process_simulation_results(
        request_collection,
        &*blocks,
        simulation_config,
        &*config_and_schema,
        orig_n_candidates,
        orig_history_size,
        curr_candidate_requests,
        total_profit,
    );

    info!(
        "Total Time to run simulation: {} seconds",
        rounds_instant.elapsed().as_millis() as f64 / 1000f64,
    );
}

#[cfg(test)]
mod tests {
    use crate::config::Mode::Simulate;
    use crate::config::{
        AllocationConfig, Budget, BudgetTotal, BudgetType, CompositionConfig, Input, OutputConfig,
        OutputPaths, RequestAdapterConfig, UnlockingBudgetTrigger,
    };
    use crate::request::RequestBuilder;
    use crate::schema::Schema;
    use crate::simulation::RequestCollection;
    use crate::util::{self, generate_blocks};
    use crate::AccountingType::EpsDp;
    use crate::{
        allocation, simulation, AccountingType, Cli, ConfigAndSchema, Request, RequestId, RoundId,
        SimulationConfig,
    };
    use itertools::Itertools;
    use std::collections::{BTreeMap, HashMap};
    use std::path::PathBuf;
    use std::str::FromStr;

    use super::BatchingStrategy;

    fn to_request_id(vec: Vec<usize>) -> Vec<RequestId> {
        vec.into_iter().map(RequestId).collect()
    }

    const EPS1_4: BudgetTotal = BudgetTotal {
        budget_file: None,
        epsilon: Some(1.4),
        delta: None,
        rdp1: None,
        rdp2: None,
        rdp3: None,
        rdp4: None,
        rdp5: None,
        rdp7: None,
        rdp10: None,
        rdp13: None,
        rdp14: None,
        rdp15: None,
        alphas: None,
        no_global_alpha_reduction: false,
        convert_candidate_request_costs: false,
        convert_history_request_costs: false,
        convert_block_budgets: false,
    };

    #[test]
    fn test_dpf_ordered_dpf_no_pa_keep_rejected_unlocking_budget_1() {
        let budgets_and_blocks = [
            (EpsDp { eps: 0.9 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 0.4 }, 3),
            (EpsDp { eps: 1.0 }, 3),
        ];
        let candidate_requests = build_dummy_requests_no_pa(budgets_and_blocks);
        let accepted_gt = to_request_id(vec![0, 4]);

        let budget_total = EPS1_4.clone();

        let budget = Budget::UnlockingBudget {
            trigger: UnlockingBudgetTrigger::Request,
            n_steps: 3,
            budget: budget_total,
            slack: None,
        };

        let timeout_rounds = candidate_requests.len();

        run_dpf_simulation_no_pa(candidate_requests, &accepted_gt, timeout_rounds, budget);
    }

    #[test]
    fn test_dpf_ordered_dpf_no_pa_keep_rejected_unlocking_budget_2() {
        let budgets_and_blocks = [
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 0.9 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 0.4 }, 3),
            (EpsDp { eps: 1.0 }, 3),
        ];
        let candidate_requests = build_dummy_requests_no_pa(budgets_and_blocks);
        let accepted_gt = to_request_id(vec![2, 4]);

        let budget_total = EPS1_4.clone();

        let budget = Budget::UnlockingBudget {
            trigger: UnlockingBudgetTrigger::Request,
            n_steps: 3,
            budget: budget_total,
            slack: None,
        };

        let timeout_rounds = candidate_requests.len();

        run_dpf_simulation_no_pa(candidate_requests, &accepted_gt, timeout_rounds, budget);
    }

    #[test]
    fn test_dpf_ordered_dpf_no_pa_throw_rejected_unlocking_budget_1() {
        let budgets_and_blocks = [
            (EpsDp { eps: 0.9 }, 3),
            (EpsDp { eps: 0.95 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 0.4 }, 3),
            (EpsDp { eps: 1.0 }, 3),
        ];
        let candidate_requests = build_dummy_requests_no_pa(budgets_and_blocks);
        let accepted_gt = to_request_id(vec![2, 4]);

        let budget_total = EPS1_4.clone();

        let budget = Budget::UnlockingBudget {
            trigger: UnlockingBudgetTrigger::Request,
            n_steps: 3,
            budget: budget_total,
            slack: None,
        };

        run_dpf_simulation_no_pa(candidate_requests, &accepted_gt, 1, budget);
    }

    #[test]
    fn test_dpf_ordered_dpf_no_pa_throw_rejected_unlocking_budget_2() {
        let budgets_and_blocks = [
            (EpsDp { eps: 1.5 }, 3),
            (EpsDp { eps: 1.5 }, 3),
            (EpsDp { eps: 1.5 }, 3),
            (EpsDp { eps: 1.5 }, 3),
            (EpsDp { eps: 0.4 }, 3),
            (EpsDp { eps: 1.5 }, 3),
        ];
        let candidate_requests = build_dummy_requests_no_pa(budgets_and_blocks);
        let accepted_gt = to_request_id(vec![4]);

        let budget_total = EPS1_4.clone();

        let budget = Budget::UnlockingBudget {
            trigger: UnlockingBudgetTrigger::Request,
            n_steps: 3,
            budget: budget_total,
            slack: None,
        };

        run_dpf_simulation_no_pa(candidate_requests, &accepted_gt, 1, budget);
    }

    #[test]
    fn test_dpf_ordered_dpf_no_pa_keep_rejected_fix_budget_1() {
        let budgets_and_blocks = [
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 0.9 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 1.0 }, 3),
            (EpsDp { eps: 0.4 }, 3),
            (EpsDp { eps: 1.0 }, 3),
        ];
        let candidate_requests = build_dummy_requests_no_pa(budgets_and_blocks);
        let accepted_gt = to_request_id(vec![0, 4]);

        let budget_total = EPS1_4.clone();

        let budget = Budget::FixBudget {
            budget: budget_total,
        };

        let timeout_rounds = candidate_requests.len();

        run_dpf_simulation_no_pa(candidate_requests, &accepted_gt, timeout_rounds, budget);
    }

    fn run_dpf_simulation_no_pa(
        candidate_requests: HashMap<RequestId, Request>,
        accepted_gt: &[RequestId],
        timeout_rounds: usize,
        budget: Budget,
    ) {
        let empty_schema = Schema {
            accounting_type: EpsDp { eps: -1.0 },
            attributes: vec![],
            name_to_index: Default::default(),
        };

        let output_paths = OutputPaths {
            req_log_output_path: PathBuf::from_str("results/requests.csv")
                .expect("Constructing PathBuf failed"),
            round_log_output_path: PathBuf::from_str("results/rounds.csv")
                .expect("Constructing PathBuf failed"),
            runtime_log_output_path: PathBuf::from_str("results/runtime.csv")
                .expect("Constructing PathBuf failed"),
            stats_output_path: PathBuf::from_str("results/stats.json")
                .expect("Constructing PathBuf failed"),
            history_output_directory_path: None,
        };

        let allocation_config: AllocationConfig = AllocationConfig::Dpf {
            block_selector_seed: 42u64,
            weighted_dpf: false,
            dominant_share_by_remaining_budget: false,
            composition: CompositionConfig::BlockComposition {
                budget: budget.clone(),
                budget_type: BudgetType::OptimalBudget,
            },
        };

        let config = Cli {
            mode: Simulate {
                allocation: allocation_config.clone(),
                batch_size: Some(1),
                timeout_rounds,
                max_requests: None,
            },
            input: Input {
                schema: Default::default(),
                blocks: Default::default(),
                requests: Default::default(),
                request_adapter_config: RequestAdapterConfig {
                    request_adapter: None,
                    request_adapter_seed: None,
                },
                history: Default::default(),
            },
            output_config: OutputConfig {
                req_log_output: Default::default(),
                round_log_output: Default::default(),
                runtime_log_output: Default::default(),
                log_remaining_budget: false,
                log_nonfinal_rejections: false,
                stats_output: Default::default(),
                history_output_directory: None,
            },
        };

        let simulation_config = SimulationConfig {
            batching_strategy: BatchingStrategy::ByBatchSize(1),
            timeout_rounds,
            unreduced_budget_config: budget,
            start_round: RoundId(0),
            output_paths,
            log_nonfinal_rejections: false,
        };

        let mut config_and_schema = ConfigAndSchema {
            config,
            schema: empty_schema,
        };

        let mut blocks = generate_blocks(0, 5, EpsDp { eps: 0.0 });

        let accepted = BTreeMap::new();
        let rejected_requests = BTreeMap::new();
        let request_history = HashMap::new();

        let sorted_candidates: Vec<Request> = candidate_requests
            .into_iter()
            .map(|(_, request)| request)
            .sorted_by(|r1, r2| Ord::cmp(&r1.request_id, &r2.request_id))
            .collect();

        let mut allocator = allocation::construct_allocator(&allocation_config);

        let rejected_gt = &sorted_candidates
            .iter()
            .filter_map(|req| {
                if !accepted_gt.contains(&req.request_id) {
                    Some(req.request_id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut request_collection = RequestCollection {
            sorted_candidates,
            request_history,
            rejected_requests,
            accepted,
            remaining_requests: BTreeMap::new(),
        };

        simulation::run_simulation(
            &mut request_collection,
            &mut blocks,
            &mut allocator,
            &simulation_config,
            &mut config_and_schema,
        );

        check_requests(
            &request_collection
                .accepted
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            accepted_gt,
            &request_collection
                .rejected_requests
                .keys()
                .copied()
                .collect::<Vec<_>>(),
            rejected_gt,
        );
    }

    fn check_requests(
        accepted: &[RequestId],
        accepted_gt: &[RequestId],
        rejected: &[RequestId],
        rejected_gt: &[RequestId],
    ) {
        assert_eq!(
            accepted.iter().sorted().collect::<Vec<_>>(),
            accepted_gt.iter().sorted().collect::<Vec<_>>(),
            "Accepted requests deviate from ground truth (left: calculated, right: ground truth)"
        );
        assert_eq!(
            rejected.iter().sorted().collect::<Vec<_>>(),
            rejected_gt.iter().sorted().collect::<Vec<_>>(),
            "Rejected requests deviate from ground truth (left: calculated, right: ground truth)"
        );
    }

    fn build_dummy_requests_no_pa<const N: usize>(
        budgets_and_blocks: [(AccountingType, usize); N],
    ) -> HashMap<RequestId, Request> {
        let schema = util::build_dummy_schema(EpsDp { eps: 1.0 });
        let mut res: HashMap<RequestId, Request> = HashMap::with_capacity(N);
        for (i, (cost, n_users)) in budgets_and_blocks.into_iter().enumerate() {
            let req = RequestBuilder::new(
                RequestId(i),
                cost.clone(),
                1,
                n_users,
                &schema,
                Default::default(),
            )
            .build();

            let inserted = res.insert(req.request_id, req);
            assert!(inserted.is_none());
        }
        res
    }
}
