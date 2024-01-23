//! This module contains various structs and functions that are used in the
//! [simulation module](crate::simulation) for logging

use serde::Serialize;
use std::time::Instant;

use crate::{AlphaReductionsResult, ConfigAndSchema, PubRdpAccounting, Request, RequestId, Schema};

use crate::allocation::ilp::stats::IlpStats;
use crate::allocation::AllocationStatus;
use crate::block::external::ExternalBlock;
use crate::config::AllocationConfig;
use crate::dprivacy::AccountingTypeRatio;
use crate::schema::DataValueLookup;
use crate::{Accounting, AccountingType, Block, BlockId, RoundId, SimulationConfig};
use csv::Writer;
use log::trace;
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::ops::AddAssign;
use std::path::Path;

/// Contains per-request information, written to req-log-output
#[derive(Serialize)]
pub struct RequestLogRow<'a> {
    decision_round: RoundId,
    decision: &'a str,
    request_id: RequestId,
    request_cost: &'a str,
    n_requested_users: usize,
    profit: u64,
    assigned_blocks: &'a str,
    adapter_info: &'a str,
    decision_is_final: bool,
    num_virtual_blocks: usize,
    /// Indicates if the request was accepted by the ilp, but then was rejected when
    /// trying to allocate it using the [crate::composition::ProblemFormulation]
    allocation_failed_after_accepted: Option<bool>,
    /// If true, indicates that this request was not accepted by the ilp, but was allocated
    /// by the greedy allocation which is run if any request accepted by the ilp could not be
    /// allocated (because of numeric issues)
    allocated_greedily_during_ilp_allocation: Option<bool>,
}

/// Contains per-round information, written to round-log-output
#[derive(Serialize, Debug, Clone)]
pub struct RuntimeMeasurement {
    round: Option<RoundId>,
    kind: RuntimeKind,
    measurement_millis: Option<u128>,

    #[serde(skip)]
    instant: Instant,
}

impl RuntimeMeasurement {
    pub fn start(kind: RuntimeKind) -> Self {
        Self {
            round: None,
            kind,
            measurement_millis: None,
            instant: Instant::now(),
        }
    }

    pub fn stop(&mut self) -> Self {
        self.measurement_millis = Some(self.instant.elapsed().as_millis());
        self.clone()
    }
}

#[derive(Serialize, Debug, Clone, Eq, PartialEq, Hash)]
pub enum RuntimeKind {
    TotalRound,      // total round time ~ (RoundSetup + RoundAllocation)
    RoundSetup,      // time to setup round (e.g., update blocks, global alpha reduction, ...)
    RoundAllocation, // ~(BuildProblemFormulation + RunAllocationAlgorithm)

    // general
    BuildProblemFormulation,
    RunAllocationAlgorithm,

    // block composition pa (part of BuildProblemFormulation)
    Segmentation, // only for block-composition-pa: find segments based on request batch
    PostSegmentation, // applying history to segments

    // ilp allocation (part of RunAllocationAlgorithm)
    BuildIlp,
    SolveIlp,
    PostIlp,
}

/// Contains per-round information, written to round-log-output
#[derive(Serialize, Debug)]
pub struct RoundLogRow<'a> {
    round: RoundId,
    newly_available_requests: &'a str,
    newly_accepted_requests: &'a str,
    allocation_status: &'a str,
    remaining_budget: Option<String>,
    /// The number of contested segments in the problem formulation when first constructed. Note
    /// that this can differ from [Self::ilp_num_contested_segments] (even if both are available)
    /// due to the fact that before constructing the ilp, any acceptable requests will be accepted,
    /// which ay reduce the number of contested segments.
    num_contested_segments_initially: Option<usize>,
    ilp_num_vars: Option<i32>,
    ilp_num_int_vars: Option<i32>,
    ilp_num_bin_vars: Option<i32>,
    ilp_num_constr: Option<i32>,
    ilp_num_nz_coeffs: Option<f64>,
    ilp_num_blocks: Option<usize>,
    ilp_num_contested_segments: Option<usize>,
    ilp_contested_segments_per_block_mean: Option<f64>,
    ilp_contested_segments_per_block_stddev: Option<f64>,
    n_alphas_before_global_reduction: Option<usize>,
    n_alphas_eliminated_global_budget_reduction: Option<usize>,
    n_alphas_eliminated_global_ratio_reduction: Option<usize>,
    n_alphas_eliminated_global_combinatorial_reduction: Option<usize>,
    n_alphas_after_global_before_local_reduction: Option<usize>,
    n_alphas_after_local_reduction: Option<usize>,
    n_alphas_eliminated_budget_reduction_mean: Option<f64>,
    n_alphas_eliminated_budget_reduction_stddev: Option<f64>,
    n_alphas_eliminated_ratio_reduction_mean: Option<f64>,
    n_alphas_eliminated_ratio_reduction_stddev: Option<f64>,
    n_alphas_eliminated_combinatorial_reduction_mean: Option<f64>,
    n_alphas_eliminated_combinatorial_reduction_stddev: Option<f64>,
}

/// Contains global information, written to stats-output
#[derive(Serialize)]
pub struct GlobalStats {
    schema_num_virtual_blocks: usize,
    n_candidate_requests: usize,
    n_history_requests: usize,
    max_available_blocks: usize,
    total_profit: u64,
}

/// Contains statistics about the greedy allocation
/// [alpha reduction](crate::dprivacy::rdpopt::RdpOptimization::calc_needed_alphas))
#[derive(Clone, Debug, Serialize, Copy)]
pub struct GreedyStats {
    /// How many contested segments there were in total
    pub num_contested_segments_initially: usize,
}

/// Contains statistics about the ilp model, the size of the problem in terms of privacy resources,
/// and  optimizations done during model building (e.g.,
/// [alpha reduction](crate::dprivacy::rdpopt::RdpOptimization::calc_needed_alphas))
#[derive(Clone, Debug, Serialize, Copy)]
pub struct DpfStats {
    /// How many contested segments there were in total
    pub num_contested_segments_initially: usize,
}

pub fn write_history_and_blocks(
    history_output_directory_path: &Path,
    request_history: &BTreeMap<RequestId, Request>,
    remaining_requests: &BTreeMap<RequestId, Request>,
    blocks: &BTreeMap<BlockId, Block>,
    schema: &Schema,
) {
    trace!("Writing history and blocks to disk");
    let block_path = history_output_directory_path.join("block_history.json");
    let request_history_path = history_output_directory_path.join("request_history.json");
    let remaining_requests_path = history_output_directory_path.join("remaining_requests.json");

    let mut block_file = File::create(block_path).expect("Creating block file failed");
    let mut request_history_file =
        File::create(request_history_path).expect("Creating request history file failed");
    let mut remaining_requests_file =
        File::create(remaining_requests_path).expect("Creating remaining requests file failed");

    // Note: We take the unreduced unlocked budget here since the history is meant
    // to be reused, and it should easily be possible to add new requests, without
    // needing to know how many alpha values were reduced away by global alpha reduction.
    trace!("Writing {} blocks to disk", blocks.len());
    serde_json::to_writer(
        &mut block_file,
        &blocks
            .iter()
            .map(|(_, block)| ExternalBlock {
                id: block.id,
                request_ids: block.request_history.clone(),
                unlocked_budget: block.unreduced_unlocked_budget.clone(),
                created: block.created,
                retired: block.retired,
            })
            .collect::<Vec<ExternalBlock>>(),
    )
    .expect("Writing blocks to file failed");

    trace!("Writing {} history requests to disk", request_history.len());
    serde_json::to_writer(
        &mut request_history_file,
        &request_history
            .iter()
            .map(|(_, request)| request.to_external(schema))
            .collect::<Vec<crate::request::external::ExternalRequest>>(),
    )
    .expect("Writing request history to file failed");

    trace!(
        "Writing {} remaining requests to disk",
        remaining_requests.len()
    );
    serde_json::to_writer(
        &mut remaining_requests_file,
        &remaining_requests
            .iter()
            .map(|(_, request)| request.to_external(schema))
            .collect::<Vec<crate::request::external::ExternalRequest>>(),
    )
    .expect("Writing remaining requests to file failed");
}

pub(crate) fn write_runtime_log(
    runtime_logger: &mut Writer<File>,
    simulation_round: RoundId,
    measurements: &mut Vec<RuntimeMeasurement>,
) {
    let mut kinds = HashSet::new();

    for mut m in measurements.drain(..) {
        let row = match m {
            RuntimeMeasurement {
                round: Some(round), ..
            } if round == simulation_round => m,
            RuntimeMeasurement { round: None, .. } => {
                m.round = Some(simulation_round);
                m
            }
            RuntimeMeasurement {
                round: other_round, ..
            } => {
                panic!(
                    "Measurement with round {:?} found, but expected round {:?}",
                    other_round, simulation_round
                );
            }
        };

        assert!(
            kinds.insert(row.kind.clone()),
            "Duplicate runtime measurement kind found: {:?}",
            row.kind
        );

        runtime_logger
            .serialize(row)
            .expect("Appending to runtime log failed");
    }

    runtime_logger.flush().expect("failed to flush runtime log");
}

/// Writes information about the current round to the given `round_logger`. Note that for greedy and
/// dpf, a round in simulation might corresponds to multiple rounds of allocation, so this methods
/// needs to recreate these allocation rounds from the given information. This recreation might not
/// contain all statistics that might be available, so
/// [batch_size](../config/enum.Mode.html#variant.Simulate.field.batch_size) should be set to 1
/// (or a round should only contain a single request) to be able get all statistics.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_round_log_row(
    round_logger: &mut Writer<File>,
    simulation_round: RoundId,
    _simulation_config: &SimulationConfig,
    config_and_schema: &ConfigAndSchema,
    orig_config_and_schema: &ConfigAndSchema,
    newly_available_requests: BTreeSet<RequestId>,
    newly_accepted_requests: BTreeSet<RequestId>,
    allocation_status: AllocationStatus,
    _allocation_config: &AllocationConfig,
    request_history: &HashMap<RequestId, Request>,
    blocks: &HashMap<BlockId, Block>,
    global_reduction_results: Option<AlphaReductionsResult>,
) {
    let remaining_budget: Option<String> = {
        if config_and_schema.config.output_config.log_remaining_budget {
            let ratio_mean = get_remaining_budget(config_and_schema, request_history, blocks);
            let remaining_budget_str =
                serde_json::to_string(&ratio_mean).expect("Serializing ratio mean failed");
            Some(remaining_budget_str)
        } else {
            None
        }
    };

    // If the allocation method is ilp, need just one round log entry per simulation round. The
    // same holds for dpf due to the assertion above. Else, if we have greedy allocation, a
    // simulation round might correspond to multiple allocation rounds, so might need to write
    // multiple log entries
    let ilp_stats: Option<&IlpStats> = allocation_status.get_ilp_stats();
    let round_log_row = RoundLogRow {
        round: simulation_round,
        newly_available_requests: &serde_json::to_string(&newly_available_requests)
            .expect("Serializing newly available requests failed"),
        newly_accepted_requests: &serde_json::to_string(&newly_accepted_requests)
            .expect("Serializing newly allocated requests failed"),
        allocation_status: allocation_status.to_string(),
        remaining_budget,
        num_contested_segments_initially: allocation_status.get_initial_num_contested_segments(),
        ilp_num_vars: ilp_stats.map(|x| x.num_vars),
        ilp_num_int_vars: ilp_stats.map(|x| x.num_int_vars),
        ilp_num_bin_vars: ilp_stats.map(|x| x.num_bin_vars),
        ilp_num_constr: ilp_stats.map(|x| x.num_constr),
        ilp_num_nz_coeffs: ilp_stats.map(|x| x.num_nz_coeffs),
        ilp_num_blocks: ilp_stats.map(|x| x.num_blocks),
        ilp_num_contested_segments: ilp_stats.map(|x| x.ilp_num_contested_segments),
        ilp_contested_segments_per_block_mean: ilp_stats
            .map(|x| x.contested_segments_per_block.mean),
        ilp_contested_segments_per_block_stddev: ilp_stats
            .map(|x| x.contested_segments_per_block.stddev),
        n_alphas_before_global_reduction: orig_config_and_schema
            .config
            .total_budget()
            .alphas()
            .map(|x| x.to_vec().len()),
        n_alphas_eliminated_global_budget_reduction: global_reduction_results
            .and_then(|x| x.budget_reduction),
        n_alphas_eliminated_global_ratio_reduction: global_reduction_results
            .and_then(|x| x.ratio_reduction),
        n_alphas_eliminated_global_combinatorial_reduction: global_reduction_results
            .and_then(|x| x.combinatorial_reduction),
        n_alphas_after_global_before_local_reduction: ilp_stats
            .and_then(|x| x.rdp_stats.map(|y| y.n_alphas_no_local_reduction)),
        n_alphas_after_local_reduction: ilp_stats
            .and_then(|x| x.rdp_stats.map(|y| y.n_alphas_after_local_reduction)),
        n_alphas_eliminated_budget_reduction_mean: ilp_stats.and_then(|x| {
            x.rdp_stats
                .and_then(|y| y.n_alphas_eliminated_budget_reduction.map(|z| z.mean))
        }),
        n_alphas_eliminated_budget_reduction_stddev: ilp_stats.and_then(|x| {
            x.rdp_stats
                .and_then(|y| y.n_alphas_eliminated_budget_reduction.map(|z| z.stddev))
        }),
        n_alphas_eliminated_ratio_reduction_mean: ilp_stats.and_then(|x| {
            x.rdp_stats
                .and_then(|y| y.n_alphas_eliminated_ratio_reduction.map(|z| z.mean))
        }),
        n_alphas_eliminated_ratio_reduction_stddev: ilp_stats.and_then(|x| {
            x.rdp_stats
                .and_then(|y| y.n_alphas_eliminated_ratio_reduction.map(|z| z.stddev))
        }),
        n_alphas_eliminated_combinatorial_reduction_mean: ilp_stats.and_then(|x| {
            x.rdp_stats.and_then(|y| {
                y.n_alphas_eliminated_combinatorial_reduction
                    .map(|z| z.mean)
            })
        }),
        n_alphas_eliminated_combinatorial_reduction_stddev: ilp_stats.and_then(|x| {
            x.rdp_stats.and_then(|y| {
                y.n_alphas_eliminated_combinatorial_reduction
                    .map(|z| z.stddev)
            })
        }),
    };
    round_logger
        .serialize(round_log_row)
        .expect("Appending to log failed");
    round_logger.flush().expect("failed to flush log");
}

pub fn get_remaining_budget(
    config_and_schema: &ConfigAndSchema,
    request_history: &HashMap<RequestId, Request>,
    blocks: &HashMap<BlockId, Block>,
) -> AccountingTypeRatio {
    assert!(!blocks.is_empty(), "No blocks");
    let zero_budget = AccountingType::zero_clone(&config_and_schema.schema.accounting_type);
    // We first calculate the remaining budget in the current simulation round.
    let remaining_budgets = blocks.par_iter().map(|(_, block)| {
        // For each virtual block, initialize a budget with zero
        let mut virtual_block_costs: HashMap<Vec<usize>, AccountingType> = config_and_schema
            .schema
            .virtual_block_id_iterator()
            .map(|virtual_block_id| (virtual_block_id, zero_budget.clone()))
            .collect();

        // Now go through block history, and get allocated requests, and add up costs
        for rid in block.request_history.iter() {
            let request = &request_history[rid];
            for virtual_block_id in request.dnf().repeating_iter(&config_and_schema.schema) {
                virtual_block_costs
                    .get_mut(&virtual_block_id)
                    .expect("Did not find virtual block")
                    .add_assign(&request.request_cost);
            }
        }

        // Now, calculate ratios of budget used and add them up
        virtual_block_costs
            .values()
            .map(|cost| {
                (&block.unlocked_budget - cost)
                    .remaining_ratios(&config_and_schema.config.total_budget().budget())
            })
            .fold(None, |acc, ratio| match acc {
                None => Some(ratio),
                Some(c_acc) => Some(AccountingTypeRatio(c_acc.0 + ratio.0)),
            })
            .expect("Couldn't calculate sum of remaining budget")
    });
    // Now, add up costs across different blocks
    let mut ratio_mean: AccountingTypeRatio = remaining_budgets.reduce(
        || AccountingTypeRatio(zero_budget.clone()),
        |acc, ratio| AccountingTypeRatio(acc.0 + ratio.0),
    );
    // Finally, divide by the number of budgets to get
    let n_budgets = config_and_schema.schema.num_virtual_blocks() as f64 * blocks.len() as f64;
    ratio_mean.0.apply_func(&|sum| sum / n_budgets);
    ratio_mean
}

pub(crate) fn write_global_stats(
    schema: &Schema,
    simulation_config: &SimulationConfig,
    n_candidate_requests: usize,
    n_history_requests: usize,
    max_available_blocks: usize,
    total_profit: u64,
) {
    let global_stats = GlobalStats {
        schema_num_virtual_blocks: schema.num_virtual_blocks(),
        n_candidate_requests,
        n_history_requests,
        max_available_blocks,
        total_profit,
    };

    let mut file = File::create(&simulation_config.output_paths.stats_output_path)
        .expect("Creating global stats file failed");

    serde_json::to_writer(&mut file, &global_stats).expect("Serializing global stats failed");
}

/// `candidate_request_position` contains in which (0-indexed) position regarding request id the
/// given request is in the current batch, and needed for greedy allocation with batch size > 1
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_request_log_row(
    decision: &str,
    logger: &mut Writer<File>,
    _candidate_request_position: usize,
    simulation_round: RoundId,
    request: &Request,
    decision_is_final: bool,
    assigned_blocks: &BTreeSet<BlockId>,
    _simulation_config: &SimulationConfig,
    config_and_schema: &ConfigAndSchema,
    ilp_stats: Option<&IlpStats>,
) {
    let decision_round = simulation_round;
    let log_row: RequestLogRow = RequestLogRow {
        decision_round,
        decision,
        request_id: request.request_id,
        request_cost: &serde_json::to_string(&request.request_cost).expect("Serialization failed"),
        n_requested_users: request.n_users,
        profit: request.profit,
        assigned_blocks: &serde_json::to_string(assigned_blocks).expect("Serialization failed"),
        adapter_info: &serde_json::to_string(&request.adapter_info).expect("Serialization failed"),
        decision_is_final,
        num_virtual_blocks: request.dnf().num_virtual_blocks(&config_and_schema.schema),
        allocation_failed_after_accepted: ilp_stats.map(|stats| {
            stats
                .failed_and_retried
                .as_ref()
                .expect("Could not get failed and retried")
                .failed_allocations
                .contains(&request.request_id)
        }),

        allocated_greedily_during_ilp_allocation: ilp_stats.map(|stats| {
            stats
                .failed_and_retried
                .as_ref()
                .expect("Could not get failed and retried")
                .greedily_allocated
                .contains(&request.request_id)
        }),
    };

    logger.serialize(log_row).expect("Appending to log failed");
}
