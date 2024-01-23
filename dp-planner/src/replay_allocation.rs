//! This module can be used to debug issues that occur during simulation, the most important part
//! of the first round of allocation.
//!
fn main() {
    println!("Not implemented at the moment!");
}
//use clap::Parser;
//use dp_planner_lib::block::{Block, BlockId};
//use dp_planner_lib::config::Cli;
//use dp_planner_lib::dprivacy::rdp_alphas_accounting::PubRdpAccounting;
//use dp_planner_lib::dprivacy::{Accounting, AccountingType, AlphaReductionsResult, RdpAccounting};
//use dp_planner_lib::request::adapter::RequestAdapter;
//use dp_planner_lib::request::{Request, RequestId};
//use dp_planner_lib::schema::DataValueLookup;
//use dp_planner_lib::simulation::RoundId;
//use dp_planner_lib::{block, request, schema};
//use log::{info, trace};
//use rayon::prelude::*;
//use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
//use std::fs::File;
//use std::io::BufReader;
//use std::ops::AddAssign;
//use std::path::PathBuf;
//
//
//
//fn main() {
//    // init logger
//    let mut builder = env_logger::Builder::from_default_env();
//    builder.target(env_logger::Target::Stdout);
//    builder.init();
//
//    let accepted_requests = get_accepted_requests_with_blocks(PathBuf::from(
//        "resources/specific_setup/accepted_requests.json",
//    ));
//    let ilp_allocated_requests = get_ilp_allocated(PathBuf::from(
//        "resources/specific_setup/ilp_allocated_requests.json",
//    ));
//
//    info!(
//        "Accepted requests which were not allocated by ilp: {:?}",
//        &BTreeSet::from_iter(accepted_requests.keys().copied())
//            .difference(&BTreeSet::from_iter(ilp_allocated_requests.iter().copied()))
//            .collect::<Vec<_>>()
//    );
//
//    assert!(ilp_allocated_requests
//        .iter()
//        .all(|request_id| { accepted_requests.contains_key(request_id) }));
//
//    let mut config: Cli = Cli::parse();
//
//    trace!("Input config: {:?}", config);
//
//    // check that there are no misconfigurations
//    config.check_config();
//
//    let seed = config
//        .input
//        .request_adapter_config
//        .request_adapter_seed
//        .unwrap_or(1848);
//    // initialize request adapter
//    let mut request_adapter: RequestAdapter = config
//        .input
//        .request_adapter_config
//        .request_adapter
//        .clone()
//        .map(|path| RequestAdapter::new(path, seed))
//        .unwrap_or_else(RequestAdapter::get_empty_adapter);
//    // 1. extract requests, schema, etc.
//
//    let budget = config.total_budget().budget();
//    info!("Budget: {:?}", &budget);
//
//    // load and init schema
//    let mut schema =
//        schema::load_schema(config.input.schema.clone(), &config.total_budget().budget())
//            .expect("loading schema failed");
//
//    trace!("Loading candidate requests");
//    // loads candidate requests and converts them to internal format
//    let candidate_requests = request::load_requests(
//        config.input.requests.clone(),
//        &schema,
//        &mut request_adapter,
//        &config.total_budget().candidate_request_conversion_alphas(),
//    )
//    .expect("loading requests failed");
//
//    assert!(candidate_requests
//        .values()
//        .all(|req| req.unreduced_cost == req.request_cost));
//
//    trace!("Loading history requests");
//    let mut request_history: HashMap<RequestId, Request> = request::load_requests(
//        config.input.history.clone(),
//        &schema,
//        &mut RequestAdapter::get_empty_adapter(),
//        &config.total_budget().history_request_conversion_alphas(),
//    )
//    .expect("loading history failed");
//
//    assert!(request_history
//        .values()
//        .all(|req| req.unreduced_cost == req.request_cost));
//
//    // make sure request ids are unique also between candidate_requests and request_history
//    for (rid, _request) in candidate_requests.iter() {
//        assert!(
//            !request_history.contains_key(rid),
//            "A request id was used for candidate requests and the request history"
//        )
//    }
//
//    trace!("Loading blocks");
//    let mut blocks: HashMap<BlockId, Block> = block::load_blocks(
//        config.input.blocks.clone(),
//        &request_history,
//        &schema,
//        &config.total_budget().block_conversion_alphas(),
//    )
//    .expect("loading of blocks failed");
//
//    for block in blocks.values() {
//        assert_eq!(
//            block.created,
//            RoundId(0),
//            "Currently, blocks with created > 0 are not supported"
//        )
//    }
//
//    let b_size = match config.mode {
//        dp_planner_lib::config::Mode::Simulate {
//            allocation: _,
//            batch_size,
//            ..
//        } => batch_size,
//        _ => candidate_requests.len(),
//    };
//
//    let unreduced_budget_config = config.allocation().budget_config().clone();
//    let _n_initial_alphas = config
//        .total_budget()
//        .alphas()
//        .map(|alphas| alphas.to_vec().len());
//
//    let mut candidate_requests = candidate_requests.into_iter().collect();
//
//    // Determine which alphas are actually needed, if we have rdp
//    let _global_alpha_red_res: Option<AlphaReductionsResult> =
//        if budget.is_rdp() && config.total_budget().global_alpha_reduction() {
//            let res = dp_planner_lib::global_reduce_alphas(
//                &mut config,
//                &budget,
//                &mut schema,
//                &mut candidate_requests,
//                &mut request_history,
//                &mut blocks,
//            );
//            Some(res)
//        } else {
//            None
//        };
//
//    let candidate_requests = candidate_requests;
//    let request_history = request_history;
//    let config = config;
//    let schema = schema;
//
//    let budget = config.total_budget().budget();
//
//    // Give the blocks some budget
//    dp_planner_lib::simulation::util::update_block_unlocked_budget(
//        &mut blocks,
//        config.allocation().budget_config(),
//        &unreduced_budget_config,
//        1,
//        b_size,
//        RoundId(0),
//    );
//
//    trace!("Finished loading requests, blocks etc. - starting segmentation");
//
//    let candidate_requests_hashmap: HashMap<RequestId, Request> = candidate_requests
//        .iter()
//        .map(|(rid, req)| (*rid, req.clone()))
//        .collect();
//
//    dp_planner_lib::util::construct_pf_and_replay_allocation(
//        &blocks,
//        &candidate_requests_hashmap,
//        &request_history,
//        &schema,
//        &ilp_allocated_requests,
//        &accepted_requests,
//    );
//
//    drop(request_history);
//
//    for (rid, blockset) in accepted_requests.iter() {
//        for bid in blockset {
//            blocks
//                .get_mut(bid)
//                .expect("Did not find allocated block")
//                .request_history
//                .push(*rid);
//        }
//    }
//
//    assert!(!blocks.is_empty(), "No blocks");
//    let zero_budget = AccountingType::zero_clone(&schema.accounting_type);
//    // We first calculate the remaining budget in the current simulation round.
//    let summed_costs = blocks
//        .par_iter()
//        .map(|(_, block)| {
//            // For each virtual block, initialize a budget with zero
//            let mut virtual_block_costs: HashMap<Vec<usize>, AccountingType> = schema
//                .virtual_block_id_iterator()
//                .map(|virtual_block_id| (virtual_block_id, zero_budget.clone()))
//                .collect();
//
//            // Now go through block history, and get allocated requests, and add up costs
//            for rid in block.request_history.iter() {
//                let request = &candidate_requests[rid];
//                for virtual_block_id in request.dnf().repeating_iter(&schema) {
//                    virtual_block_costs
//                        .get_mut(&virtual_block_id)
//                        .expect("Did not find virtual block")
//                        .add_assign(&request.request_cost);
//                    if !budget.in_budget(&virtual_block_costs[&virtual_block_id]) {
//                        panic!(
//                            "Cost {} is too large for budget {}",
//                            virtual_block_costs[&virtual_block_id], &budget
//                        );
//                    }
//                }
//            }
//
//            virtual_block_costs
//        })
//        .collect::<Vec<_>>();
//    trace!("some budget: {:?}", summed_costs[0].iter().next().unwrap());
//}
//
//fn get_accepted_requests_with_blocks(filepath: PathBuf) -> BTreeMap<RequestId, HashSet<BlockId>> {
//    let file = File::open(filepath).expect("Could not open file");
//    let reader = BufReader::new(file);
//
//    serde_json::from_reader(reader).expect("Could not parse file")
//}
//
//fn get_ilp_allocated(filepath: PathBuf) -> Vec<RequestId> {
//    let file = File::open(filepath).expect("Could not open file");
//    let reader = BufReader::new(file);
//
//    serde_json::from_reader(reader).expect("Could not parse file")
//}
//
