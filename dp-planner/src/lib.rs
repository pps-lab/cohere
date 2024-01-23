//! This crate offers a variety of methods assigning requests to blocks, such that the
//! resulting allocation fulfills a certain differential privacy constraint.

use crate::block::{Block, BlockId};
use crate::config::Cli;
use crate::AccountingType::Rdp;
use clap::Parser;
use float_cmp::{ApproxEq, F64Margin};
use itertools::Itertools;
use log::{info, trace};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::dprivacy::budget::OptimalBudget;
use crate::dprivacy::rdp_alphas_accounting::PubRdpAccounting;
use crate::dprivacy::rdpopt::RdpOptimization;
use crate::dprivacy::{
    Accounting, AccountingType, AlphaReductions, AlphaReductionsResult, RdpAccounting,
};
use crate::request::adapter::RequestAdapter;
use crate::request::{Request, RequestId};
use crate::schema::Schema;
use crate::simulation::{
    BatchingStrategy, ConfigAndSchema, RequestCollection, RoundId, SimulationConfig,
};

pub mod allocation;
pub mod block;
pub mod composition;
pub mod config;
pub mod dprivacy;
pub mod logging;
pub mod request;
pub mod schema;
pub mod simulation;
pub mod util;

pub fn run_program() {
    #[cfg(debug_assertions)]
    info!("Debug mode enabled");

    let config: Cli = config::Cli::parse();

    trace!("Input config: {:?}", config);

    // Check that the output paths are valid
    let output_paths = config::check_output_paths(&config);

    // check that there are no misconfigurations
    config.check_config();

    let seed = config
        .input
        .request_adapter_config
        .request_adapter_seed
        .unwrap_or(1848);
    // initialize request adapter
    let mut request_adapter: RequestAdapter = config
        .input
        .request_adapter_config
        .request_adapter
        .clone()
        .map(|path| RequestAdapter::new(path, seed))
        .unwrap_or_else(RequestAdapter::get_empty_adapter);
    // 1. extract requests, schema, etc.

    let budget = config.total_budget().budget();
    info!("Budget: {:?}", &budget);

    // load and init schema
    let schema = schema::load_schema(config.input.schema.clone(), &config.total_budget().budget())
        .expect("loading schema failed");

    trace!("Loading candidate requests...");
    // loads candidate requests and converts them to internal format
    let candidate_requests = request::load_requests(
        config.input.requests.clone(),
        &schema,
        &mut request_adapter,
        &config.total_budget().candidate_request_conversion_alphas(),
    )
    .expect("loading requests failed");
    trace!("Loaded {} candidate requests", candidate_requests.len());

    assert!(candidate_requests
        .values()
        .all(|req| req.unreduced_cost == req.request_cost));

    let batching_strategy = match config.mode {
        config::Mode::Simulate {
            allocation: _,
            batch_size: Some(batch_size),
            ..
        } => BatchingStrategy::ByBatchSize(batch_size),
        config::Mode::Simulate {
            allocation: _,
            batch_size: None,
            ..
        } => BatchingStrategy::ByRequestCreated,
        config::Mode::Round { .. } => BatchingStrategy::ByBatchSize(candidate_requests.len()),
    };

    //let b_size = match config.mode {
    //    config::Mode::Simulate {
    //        allocation: _,
    //        batch_size,
    //        ..
    //    } => batch_size,
    //    _ => candidate_requests.len(),
    //};

    trace!("Loading history requests...");
    let request_history: HashMap<RequestId, Request> = match &config.input.history {
        Some(path) => request::load_requests(
            path.clone(),
            &schema,
            &mut RequestAdapter::get_empty_adapter(),
            &config.total_budget().history_request_conversion_alphas(),
        )
        .expect("loading history failed"),
        None => HashMap::new(),
    };

    trace!("Loaded {} history requests", request_history.len());

    assert!(request_history
        .values()
        .all(|req| req.unreduced_cost == req.request_cost));

    // TODO: Check that history is feasible, and does not violate any budget constraint

    // make sure request ids are unique also between candidate_requests and request_history
    for (rid, _request) in candidate_requests.iter() {
        assert!(
            !request_history.contains_key(rid),
            "A request id was used for candidate requests and the request history"
        )
    }

    trace!("Loading blocks...");
    let mut blocks: HashMap<BlockId, Block> = block::load_blocks(
        config.input.blocks.clone(),
        &request_history,
        &schema,
        &config.total_budget().block_conversion_alphas(),
    )
    .expect("loading of blocks failed");
    trace!("Loaded {} blocks", blocks.len());

    let unreduced_budget_config = config.allocation().budget_config().clone();

    // requires:
    // - assignment-> greedy,ilp,dps
    // - composition -> block, block-part-attributes(segmentation algo)

    let rejected_requests: BTreeMap<RequestId, Request> = BTreeMap::new();
    // request ids that were accepted during the run of this program
    // invariant: every request id in accepted in a key in history_requests
    let accepted: BTreeMap<RequestId, HashSet<BlockId>> = BTreeMap::new();

    let mut sorted_candidates: Vec<Request> = candidate_requests
        .into_iter()
        .map(|(_, request)| request)
        .sorted_by(|r1, r2| Ord::cmp(&r1.request_id, &r2.request_id))
        .collect();

    let is_sorted_by_created = sorted_candidates
        .windows(2)
        .all(|w| w[0].created <= w[1].created);
    assert!(
        is_sorted_by_created,
        "Sorting requests by request id leads to requests that are not sorted by the created field"
    );

    let mut n_candidates = sorted_candidates.len();

    if let BatchingStrategy::ByBatchSize(batch_size) = batching_strategy {
        for (num, candidate) in sorted_candidates.iter_mut().enumerate() {
            assert!(candidate.created.is_none(), "When using the batch size to determine the assignment round, the created field of the request must not be set");
            candidate.created = Some(RoundId(num / batch_size));
        }
    }

    let mut allocator = allocation::construct_allocator(config.allocation());

    let simulation_config: SimulationConfig;
    let mut config_and_schema: ConfigAndSchema;

    let mut request_collection: RequestCollection;

    let config_clone = config.clone();

    match config.mode {
        config::Mode::Simulate {
            allocation: _allocation,
            batch_size,
            timeout_rounds,
            max_requests,
        } => {
            let mut remaining_requests: BTreeMap<RequestId, Request> = BTreeMap::new();
            if let Some(max_req) = max_requests {
                assert!(
                    max_req <= n_candidates,
                    "max_requests needs to be <= the number of candidate requests"
                );
                remaining_requests.extend(
                    sorted_candidates
                        .drain(max_req..n_candidates)
                        .map(|req| (req.request_id, req)),
                );
                n_candidates = max_req
            }

            let batching_strategy = match batch_size {
                Some(batch_size) => BatchingStrategy::ByBatchSize(batch_size),
                None => BatchingStrategy::ByRequestCreated,
            };

            simulation_config = SimulationConfig {
                batching_strategy,
                timeout_rounds,
                unreduced_budget_config,
                start_round: sorted_candidates[0].created.unwrap_or(RoundId(0)),
                output_paths,
                log_nonfinal_rejections: config.output_config.log_nonfinal_rejections,
            };

            config_and_schema = ConfigAndSchema {
                schema,
                config: config_clone,
            };

            request_collection = RequestCollection {
                sorted_candidates,
                request_history,
                rejected_requests,
                accepted,
                remaining_requests,
            };

            simulation::run_simulation(
                &mut request_collection,
                &mut blocks,
                &mut allocator,
                &simulation_config,
                &mut config_and_schema,
            )
        }

        config::Mode::Round {
            allocation: _allocation,
            i,
            ..
        } => {
            for request in sorted_candidates.iter() {
                assert!(
                    request.created.map(|x| x <= RoundId(i)).unwrap_or(true),
                    "Tried to run round {}, but request with id {} is created only at time {:?}",
                    i,
                    request.request_id.0,
                    request.created
                );
            }

            simulation_config = SimulationConfig {
                batching_strategy: BatchingStrategy::ByBatchSize(sorted_candidates.len()), // If we are in round mode, we form a batch by taking all the candidate requests
                timeout_rounds: 1,
                unreduced_budget_config,
                start_round: RoundId(i),
                output_paths,
                log_nonfinal_rejections: config.output_config.log_nonfinal_rejections,
            };

            config_and_schema = ConfigAndSchema {
                schema,
                config: config_clone,
            };

            request_collection = RequestCollection {
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
        }
    }

    assert_eq!(
        request_collection.accepted.len() + request_collection.rejected_requests.len(),
        n_candidates,
        "Lost some requests while allocating"
    );

    info!(
        "Accepted {} requests, rejected {} requests",
        request_collection.accepted.len(),
        request_collection.rejected_requests.len()
    );
}

/// Reduces which alpha values are globally taken into account, and changes all the costs and
/// budgets appropriately
pub fn global_reduce_alphas(
    config: &mut Cli,
    budget: &AccountingType,
    schema: &mut Schema,
    candidate_requests: &mut BTreeMap<RequestId, Request>,
    request_history: &mut HashMap<RequestId, Request>,
    blocks: &mut HashMap<BlockId, Block>,
) -> AlphaReductionsResult {
    assert!(budget.is_rdp());
    trace!("Starting global alpha reduction. Budget: {:?}", budget);
    // first, find the unique costs
    let mut unique_costs: Vec<AccountingType> = Vec::new();

    'outer: for request in candidate_requests.values().chain(request_history.values()) {
        for unique_cost in unique_costs.iter() {
            if request
                .request_cost
                .approx_eq(unique_cost, F64Margin::default())
            {
                continue 'outer;
            }
        }
        unique_costs.push(request.request_cost.clone());
    }
    trace!("unique costs: {:?}", unique_costs);

    let all_alphas = config
        .total_budget()
        .alphas()
        .expect("Currently, specifying a budget directly in rdp is not supported");

    // given the unique costs and the budget, see which alpha values are actually needed
    let (needed_alphas, alpha_red_res) = dprivacy::rdpopt::RdpOptimization::calc_needed_alphas(
        &Rdp {
            eps_values: all_alphas.clone(),
        },
        &unique_costs,
        budget,
        AlphaReductions {
            budget_reduction: true,
            ratio_reduction: true,
            combinatorial_reduction: false,
        },
    );

    let needed_alphas_set = needed_alphas
        .iter()
        .map(|pacb| pacb.alpha.to_bits())
        .collect::<HashSet<u64>>();
    trace!(
        "alpha values after removal by global alpha reduction: {:?}",
        all_alphas
            .to_vec()
            .into_iter()
            .filter(|alpha| needed_alphas_set.contains(&alpha.to_bits()))
            .collect::<Vec<_>>(),
    );

    let mask: Vec<bool> = all_alphas
        .to_vec()
        .into_iter()
        .map(|alpha| needed_alphas_set.contains(&alpha.to_bits()))
        .collect();

    assert!(mask.iter().any(|b| *b), "empty mask");

    // now, actually remove the unnecessary values using the mask
    // first from the requests
    for req in candidate_requests.values_mut() {
        req.request_cost = req.request_cost.reduce_alphas(&mask);
    }

    for req in request_history.values_mut() {
        req.request_cost = req.request_cost.reduce_alphas(&mask);
    }
    // then change the alphas in the config
    config.budget_total_mut().alphas = Some(
        all_alphas
            .to_vec()
            .into_iter()
            .filter(|alpha| needed_alphas_set.contains(&alpha.to_bits()))
            .collect(),
    );
    // and in the schema
    schema.accounting_type = schema.accounting_type.reduce_alphas(&mask);
    // Finally, change it in the blocks
    for block in blocks.values_mut() {
        block.unlocked_budget = block.unlocked_budget.reduce_alphas(&mask);
    }

    alpha_red_res
}
