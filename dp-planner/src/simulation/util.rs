//! Contains various functions useful to run the simulation

use crate::allocation::{AllocationStatus, ResourceAllocation};
use crate::config::{Budget, UnlockingBudgetTrigger};
use crate::dprivacy::AccountingType;
use crate::{
    logging, Accounting, Block, BlockId, ConfigAndSchema, Request, RequestCollection, RequestId,
    RoundId, SimulationConfig,
};
use csv::Writer;
use float_cmp::{ApproxEq, F64Margin};
use log::info;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use super::BatchingStrategy;

/// This function updates the unlocked budget of the passed blocks.
///
/// # Arguments
/// * `available_blocks` - the blocks whose budget is to be updated
/// * `budget_config` - the input from the clip, containing the original budget
/// * `n_rounds` - how many rounds have passed since the budget was last updated (should be 1, if
///  the budgets are updated each round)
/// * `n_requests` - how many requests have been added since the last time the budget has been
/// updated (should be equal to the batch size, if the budgets are updated each round)
/// * `curr_round` - Which round it currently is - used to check if a block joined the system
/// this round, by comparing the blocks created field with this argument
pub fn update_block_unlocked_budget(
    available_blocks: &mut HashMap<BlockId, Block>,
    budget_config: &Budget,
    unreduced_budget_config: &Budget,
    n_requests: usize,
    simulation_round: RoundId,
) {
    //TODO possibly introduce counter and compute anew each time from max budget (If there are a lot of update steps, will get numerically inaccurate result)
    let total = budget_config.budget();
    let unreduced_total = unreduced_budget_config.budget();
    for (_, block) in available_blocks.iter_mut() {
        assert!(total.check_same_type(&block.unlocked_budget));
        assert!(unreduced_total.check_same_type(&block.unreduced_unlocked_budget));
    }

    match budget_config {
        Budget::FixBudget { .. } => {
            assert!(budget_config
                .budget()
                .approx_eq(&total, F64Margin::default())); // no blocks update
            for (_, block) in available_blocks.iter_mut() {
                block.unlocked_budget = total.clone();
                block.unreduced_unlocked_budget = unreduced_total.clone();
            }
        }
        Budget::UnlockingBudget {
            trigger: UnlockingBudgetTrigger::Round,
            n_steps: n_total_steps,
            slack,
            ..
        } => {
            // The reason for splitting unlocking budget into two cases is that the slack factor currently only works for the by round case
            // In request unlocking, we do not currently track how many steps we have already unlocked => cannot determine the slack factor

            let slack = slack.unwrap_or(0.0); // use 0.0 i.e., no slack as default value
            assert!(
                (0.0..=1.0).contains(&slack),
                "slack must be between 0 and 1"
            );

            // apply the update step on each block
            for (_, curr_block) in available_blocks.iter_mut() {
                unlock_budget_by_round(
                    simulation_round,
                    slack,
                    *n_total_steps,
                    curr_block,
                    &total,
                    &unreduced_total,
                );
            }
        }
        Budget::UnlockingBudget {
            trigger: UnlockingBudgetTrigger::Request,
            n_steps: n_total_steps,
            slack,
            ..
        } => {
            assert!(
                slack.is_none(),
                "slack is not supported for request unlocking"
            );

            let n_my_steps = n_requests;

            // TODO: for EpsDeltaDP delta should be available immediately fully.
            // reworked to always use clone instead of update_step, due to numerical stability
            let mut update_step = total.clone();
            update_step.apply_func(&|x: f64| x / *n_total_steps as f64 * n_my_steps as f64);
            assert!(update_step.approx_le(&total));

            let mut unreduced_update_step = unreduced_total.clone();
            unreduced_update_step
                .apply_func(&|x: f64| x / *n_total_steps as f64 * n_my_steps as f64);
            assert!(unreduced_update_step.approx_le(&unreduced_total));

            // apply the update step on each block
            for (_, curr_block) in available_blocks.iter_mut() {
                if curr_block.created == simulation_round {
                    // if the block joined this round, want to make sure it starts with correct budget
                    curr_block.unlocked_budget = update_step.clone();
                    curr_block.unreduced_unlocked_budget = unreduced_update_step.clone();
                } else {
                    let new_budget = &curr_block.unlocked_budget + &update_step;
                    let unreduced_new_budget =
                        &curr_block.unreduced_unlocked_budget + &unreduced_update_step;

                    // update the unlocked block budget (don't exceed total budget)
                    if new_budget.approx_le(&total) {
                        curr_block.unlocked_budget = new_budget;
                        curr_block.unreduced_unlocked_budget = unreduced_new_budget;
                    } else {
                        curr_block.unlocked_budget = total.clone();
                        curr_block.unreduced_unlocked_budget = unreduced_total.clone();
                    }
                }
            }
        }
    }
}

fn unlock_budget_by_round(
    simulation_round: RoundId,
    slack: f64,
    n_total_steps: usize,
    block: &mut Block,
    total_budget: &AccountingType,
    unreduced_total_budget: &AccountingType,
) {
    let block_age = ((simulation_round - block.created).0 + 1).min(n_total_steps);
    let mid = (n_total_steps as f64 / 2_f64).ceil() as usize;

    let slack_factor = if block_age < mid {
        block_age
    } else {
        n_total_steps - block_age
    };

    let unlocked_budget_factor =
        (block_age as f64 + slack_factor as f64 * slack) / n_total_steps as f64;

    // set unlocked budget
    let mut unlocked_budget = total_budget.clone();
    unlocked_budget.apply_func(&|total_budget: f64| total_budget * unlocked_budget_factor);
    assert!(unlocked_budget.approx_le(total_budget));
    block.unlocked_budget = unlocked_budget;

    // set unreduced unlocked budget
    let mut unreduced_unlocked_budget = unreduced_total_budget.clone();
    unreduced_unlocked_budget
        .apply_func(&|total_budget: f64| total_budget * unlocked_budget_factor);
    assert!(unreduced_unlocked_budget.approx_le(unreduced_total_budget));
    block.unreduced_unlocked_budget = unreduced_unlocked_budget;
}

/// This function updates the history of the passed blocks
/// # Arguments
/// * `available_blocks` - the blocks whose history is to be updated
/// * `allocation` - the allocation of candidate requests to blocks. The id of any request in the
/// allocation should not overlap with any id in a block history - the id can be added multiple
/// times to the request history in the current implementation, but the meaning of this is undefined
/// and the implementation may change.
pub fn update_block_history(
    available_blocks: &mut HashMap<BlockId, Block>,
    allocation: &ResourceAllocation,
) {
    for (request_id, block_ids) in allocation.accepted.iter() {
        for block_id in block_ids.iter() {
            available_blocks
                .get_mut(block_id)
                .expect("Assigned accepted request unavailable block")
                .request_history
                .push(*request_id);
        }
    }
}

/// Processes the results of running all simulation rounds. Most importantly, this function
/// updates the request collection with the results of the simulation, writes the results to the
/// output files if enabled (history and block files) and writes simulation statistics to a
/// stats file.
#[allow(clippy::too_many_arguments)]
pub fn process_simulation_results(
    request_collection: &mut RequestCollection,
    blocks: &HashMap<BlockId, Block>,
    simulation_config: &SimulationConfig,
    config_and_schema: &ConfigAndSchema,
    orig_n_candidates: usize,
    orig_history_size: usize,
    curr_candidate_requests: BTreeMap<RequestId, Request>,
    total_profit: u64,
) {
    // put the remaining candidates back into remaining requests and into rejected requests (note
    // that process_round_results has already removed requests that timed out from curr_candidate_requests)
    for (rid, request) in curr_candidate_requests.into_iter() {
        let inserted = request_collection
            .remaining_requests
            .insert(rid, request.clone());
        assert!(inserted.is_none());

        let inserted = request_collection.rejected_requests.insert(rid, request);
        assert!(inserted.is_none())
    }

    logging::write_global_stats(
        &config_and_schema.schema,
        simulation_config,
        orig_n_candidates,
        orig_history_size,
        blocks.len(),
        total_profit,
    );

    // TODO I think at the moment those files are not written if no request is accepted (which is a bit unfortunate)
    if let Some(history_path) = simulation_config
        .output_paths
        .history_output_directory_path
        .clone()
    {
        logging::write_history_and_blocks(
            &history_path,
            &request_collection
                .request_history
                .iter()
                .map(|(rid, req)| (*rid, req.clone()))
                .collect(),
            &request_collection.remaining_requests,
            &blocks
                .iter()
                .map(|(bid, block)| (*bid, block.clone()))
                .collect(),
            &config_and_schema.schema,
        )
    }
}

pub fn pre_round_blocks_update(
    simulation_round: RoundId,
    blocks: &mut HashMap<BlockId, Block>,
    blocks_per_round: &mut BTreeMap<RoundId, Vec<(BlockId, Block)>>,
) {
    // retire block that have a retirement field with a retirement round that is smaller than the current round
    blocks.retain(|&_blockid, block| {
        block
            .retired
            .map(|retired| retired > simulation_round)
            .unwrap_or(true)
    });

    let new_blocks = blocks_per_round
        .remove(&simulation_round)
        .unwrap_or_default();

    // add new block ids to set of active block ids
    blocks.extend(new_blocks);

    assert!(
        blocks
            .iter()
            .all(|(_, block)| block.created <= simulation_round
                && block
                    .retired
                    .map(|retired| retired > simulation_round)
                    .unwrap_or(true)),
        "active blocks are not working properly"
    );
    assert!(!blocks.is_empty());
}

pub fn pre_round_request_batch_update(
    simulation_round: RoundId,
    batching_strategy: BatchingStrategy,
    request_collection: &mut RequestCollection,
    curr_candidate_requests: &mut BTreeMap<RequestId, Request>,
    request_start_rounds: &mut HashMap<RequestId, RoundId>,
) -> (BTreeSet<RequestId>, bool) {
    // based on the batching strategy, we determine how many requests we want to process in this round
    //   -> the request_collection.sorted_candidates are guaranteed to be sorted by request_id and created
    let n_requests = match batching_strategy {
        BatchingStrategy::ByBatchSize(batch_size) => {
            // if we batch by batchsize, then we take the first batch_size requests
            batch_size.min(request_collection.sorted_candidates.len())
        }
        BatchingStrategy::ByRequestCreated => {
            // if we batch by request created, then we take all requests that were created in the current round

            let n_requests = request_collection
                .sorted_candidates
                .iter()
                .position(|req| req.created.unwrap() > simulation_round)
                .unwrap_or(request_collection.sorted_candidates.len()); // all request

            assert!(
                request_collection.sorted_candidates[n_requests - 1].created
                    == Some(simulation_round),
                "The first `n_requests` request need to be from this simulation round."
            );

            n_requests
        }
    };

    // The first `n_requests` requests are the ones that are newly available in this round.
    // -> they need to be moved to the curr_candidate_requests (and marked accordingly)
    let mut newly_available_requests: BTreeSet<RequestId> = BTreeSet::new();
    request_collection
        .sorted_candidates
        .drain(0..n_requests)
        .for_each(|req| {
            newly_available_requests.insert(req.request_id);
            request_start_rounds.insert(req.request_id, simulation_round);
            curr_candidate_requests.insert(req.request_id, req);
        });

    let is_final_round: bool = request_collection.sorted_candidates.is_empty();

    (newly_available_requests, is_final_round)
}

/// Logs how many requests are still waiting to be processed, and provides an estimate
/// for how long these remaining requests will take to be processed, and resets the time-counting
/// for the next round.
pub fn log_remaining_requests(
    request_collection: &RequestCollection,
    _simulation_config: &SimulationConfig,
    orig_n_candidates: usize,
    round_instant: &mut Option<Instant>,
    start_instant: &Instant,
) {
    if let Some(_inst) = round_instant {
        info!(
                "{} requests remaining out of {}, ca. {} seconds remaining                                     ",
                request_collection.sorted_candidates.len(),
                orig_n_candidates,
                (request_collection.sorted_candidates.len() as f64 * start_instant.elapsed().as_millis() as f64)
                / (1000_f64 * (orig_n_candidates - request_collection.sorted_candidates.len()) as f64)
            );
    } else {
        info!(
            "{} requests remaining out of {}",
            request_collection.sorted_candidates.len(),
            orig_n_candidates,
        );
    }
    std::io::stdout().flush().unwrap();
    *round_instant = Some(Instant::now());
}

/// Processes the result of a simulation round: Update all necessary data structures,
/// and write the request logs.
#[allow(clippy::too_many_arguments)]
pub fn process_round_results(
    request_collection: &mut RequestCollection,
    blocks: &mut HashMap<BlockId, Block>,
    simulation_config: &SimulationConfig,
    req_logger: &mut Writer<File>,
    curr_candidate_requests: &mut BTreeMap<RequestId, Request>,
    simulation_round: RoundId,
    total_profit: &mut u64,
    is_final_round: bool,
    assignment: &ResourceAllocation,
    config_and_schema: &ConfigAndSchema,
    allocation_status: &AllocationStatus,
    request_start_rounds: &HashMap<RequestId, RoundId>,
) {
    *total_profit += assignment
        .accepted
        .keys()
        .map(|rid| curr_candidate_requests[rid].profit)
        .sum::<u64>();

    // update the block history based on the accepted
    update_block_history(blocks, assignment);
    // add the accepted requests to the history, and rejected requests to currently rejected requests
    assert_eq!(
        curr_candidate_requests.len(),
        assignment.accepted.len() + assignment.rejected.len()
    );

    let mut curr_rejected_requests: HashMap<RequestId, Request> = HashMap::new();
    // Since curr_candidate_requests is a BTreeMap, can use .enumerate to find out what position
    // a certain request is in regarding request id.
    for (pos, (rid, request)) in curr_candidate_requests.iter_mut().enumerate() {
        if assignment.accepted.contains_key(rid) {
            // record acceptance in log
            let assigned_blocks = BTreeSet::from_iter(assignment.accepted[rid].iter().copied());

            logging::write_request_log_row(
                "Accepted",
                req_logger,
                pos,
                simulation_round,
                request,
                true,
                &assigned_blocks,
                simulation_config,
                config_and_schema,
                allocation_status.get_ilp_stats(),
            );

            // add to request history
            assert!(assigned_blocks.len() >= request.n_users);
            let inserted = request_collection
                .accepted
                .insert(*rid, assignment.accepted[rid].clone());
            assert!(inserted.is_none());

            let inserted = request_collection
                .request_history
                .insert(*rid, request.clone());
            assert!(inserted.is_none());
        } else if assignment.rejected.contains(rid) {
            // record rejectance in log (note: request may be rejected in multiple rounds)

            // Note that the simulation round has not yet changed, that happens only after this
            // function returns. So we need to add 1 to the difference between the current round
            // and the round in which the request was first available.
            let decision_is_final: bool = is_final_round
                || (usize::abs_diff(request_start_rounds[rid].0, simulation_round.0) + 1
                    >= simulation_config.timeout_rounds);
            if simulation_config.log_nonfinal_rejections || decision_is_final {
                logging::write_request_log_row(
                    "Rejected",
                    req_logger,
                    pos,
                    simulation_round,
                    request,
                    decision_is_final,
                    &BTreeSet::<BlockId>::new(),
                    simulation_config,
                    config_and_schema,
                    allocation_status.get_ilp_stats(),
                );
            }

            let inserted = curr_rejected_requests.insert(*rid, request.clone());
            assert!(inserted.is_none());
        } else {
            panic!("A request was neither accepted nor rejected")
        }
    }

    *curr_candidate_requests = BTreeMap::new();
    for (rid, request) in curr_rejected_requests.into_iter() {
        // Note that the simulation round has not yet changed, that happens only after this
        // function returns. So we need to add 1 to the difference between the current round
        // and the round in which the request was first available.
        if usize::abs_diff(simulation_round.0, request_start_rounds[&rid].0) + 1
            < simulation_config.timeout_rounds
        {
            let inserted = curr_candidate_requests.insert(rid, request);
            assert!(inserted.is_none())
        } else {
            let inserted = request_collection.rejected_requests.insert(rid, request);
            assert!(inserted.is_none())
        }
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::{ApproxEq, F64Margin};

    use crate::{
        block::{Block, BlockId},
        dprivacy::{rdp_alphas_accounting::RdpAlphas, AccountingType},
        simulation::{util::unlock_budget_by_round, RoundId},
    };

    fn _create_block(round: usize) -> Block {
        Block {
            id: BlockId::User(round),
            created: RoundId(round),
            retired: None,
            unlocked_budget: AccountingType::Rdp {
                eps_values: RdpAlphas::A2([0.0, 0.0]),
            },
            unreduced_unlocked_budget: AccountingType::Rdp {
                eps_values: RdpAlphas::A3([0.0, 0.0, 0.0]),
            },
            request_history: vec![],
        }
    }

    fn _rdp_a3(x1: f64, x2: f64, x3: f64) -> AccountingType {
        AccountingType::Rdp {
            eps_values: RdpAlphas::A3([x1, x2, x3]),
        }
    }

    fn _rdp_a2(x1: f64, x2: f64) -> AccountingType {
        AccountingType::Rdp {
            eps_values: RdpAlphas::A2([x1, x2]),
        }
    }

    #[test]
    fn test_budget_unlocking_by_round_with_noslack() {
        let slack = 0.0;

        for n_total_steps in [6, 7] {
            // test even and odd case
            let simulation_round = RoundId(n_total_steps);

            let share_a = 0.5;
            let share_b = 1.0;
            let share_c = 2.0;

            let total_budget = _rdp_a2(
                n_total_steps as f64 * share_a,
                n_total_steps as f64 * share_b,
            );
            let unreduced_total_budget = _rdp_a3(
                n_total_steps as f64 * share_a,
                n_total_steps as f64 * share_b,
                n_total_steps as f64 * share_c,
            );

            for i in 0..n_total_steps {
                let mut block = _create_block(n_total_steps - i);
                unlock_budget_by_round(
                    simulation_round,
                    slack,
                    n_total_steps,
                    &mut block,
                    &total_budget,
                    &unreduced_total_budget,
                );
                let x = (i + 1) as f64;
                assert!(block
                    .unlocked_budget
                    .approx_eq(&_rdp_a2(x * share_a, x * share_b), F64Margin::default()));
                assert!(block.unreduced_unlocked_budget.approx_eq(
                    &_rdp_a3(x * share_a, x * share_b, x * share_c),
                    F64Margin::default()
                ));
            }
        }
    }

    #[test]
    fn test_budget_unlocking_by_round_with_slack() {
        let slack = 0.2;

        let share_a = 0.5;
        let share_b = 1.0;
        let share_c = 2.0;

        // check even slack
        let even_slack = [
            1_f64 + slack,
            1_f64 + slack,
            1_f64 + slack,
            1_f64 - slack,
            1_f64 - slack,
            1_f64 - slack,
        ];

        let n_total_steps = even_slack.len();
        let simulation_round = RoundId(n_total_steps);
        let total_budget = _rdp_a2(
            n_total_steps as f64 * share_a,
            n_total_steps as f64 * share_b,
        );
        let unreduced_total_budget = _rdp_a3(
            n_total_steps as f64 * share_a,
            n_total_steps as f64 * share_b,
            n_total_steps as f64 * share_c,
        );
        for i in 0..n_total_steps {
            let mut block = _create_block(n_total_steps - i);
            unlock_budget_by_round(
                simulation_round,
                slack,
                n_total_steps,
                &mut block,
                &total_budget,
                &unreduced_total_budget,
            );
            let x: f64 = even_slack[0..=i].iter().sum();
            assert!(block
                .unlocked_budget
                .approx_eq(&_rdp_a2(x * share_a, x * share_b), F64Margin::default()));
            assert!(block.unreduced_unlocked_budget.approx_eq(
                &_rdp_a3(x * share_a, x * share_b, x * share_c),
                F64Margin::default()
            ));
        }

        // check odd slack
        let odd_slack = [
            1_f64 + slack,
            1_f64 + slack,
            1_f64 + slack,
            1.0,
            1_f64 - slack,
            1_f64 - slack,
            1_f64 - slack,
        ];

        let n_total_steps = odd_slack.len();
        let simulation_round = RoundId(n_total_steps);
        let total_budget = _rdp_a2(
            n_total_steps as f64 * share_a,
            n_total_steps as f64 * share_b,
        );
        let unreduced_total_budget = _rdp_a3(
            n_total_steps as f64 * share_a,
            n_total_steps as f64 * share_b,
            n_total_steps as f64 * share_c,
        );
        for i in 0..n_total_steps {
            let mut block = _create_block(n_total_steps - i);
            unlock_budget_by_round(
                simulation_round,
                slack,
                n_total_steps,
                &mut block,
                &total_budget,
                &unreduced_total_budget,
            );
            let x: f64 = odd_slack[0..=i].iter().sum();
            assert!(block
                .unlocked_budget
                .approx_eq(&_rdp_a2(x * share_a, x * share_b), F64Margin::default()));
            assert!(block.unreduced_unlocked_budget.approx_eq(
                &_rdp_a3(x * share_a, x * share_b, x * share_c),
                F64Margin::default()
            ));
        }
    }

    #[test]
    fn test_budget_unlocking_by_round_not_exceeding() {
        let share_a = 0.5;
        let share_b = 1.0;
        let share_c = 2.0;

        let n_total_steps = 6;

        let slack = 0.2;

        let simulation_round = RoundId(10);

        let total_budget = _rdp_a2(
            n_total_steps as f64 * share_a,
            n_total_steps as f64 * share_b,
        );
        let unreduced_total_budget = _rdp_a3(
            n_total_steps as f64 * share_a,
            n_total_steps as f64 * share_b,
            n_total_steps as f64 * share_c,
        );

        let mut block = _create_block(0);

        unlock_budget_by_round(
            simulation_round,
            slack,
            n_total_steps,
            &mut block,
            &total_budget,
            &unreduced_total_budget,
        );

        assert!(block
            .unlocked_budget
            .approx_eq(&total_budget, F64Margin::default()));
        assert!(block
            .unreduced_unlocked_budget
            .approx_eq(&unreduced_total_budget, F64Margin::default()));
    }
}
