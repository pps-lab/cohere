//! Contains methods and structs needed for ILP allocation.

pub mod stats;

use crate::allocation::ilp::stats::FailedAndRetried;
use crate::allocation::{AllocationStatus, BlockCompWrapper, ResourceAllocation};
use crate::block::{Block, BlockId};
use crate::composition::{
    AllocationError, BlockOrderStrategy, CompositionConstraint, ProblemFormulation, StatusResult,
};
use crate::dprivacy::budget::OptimalBudget;
use crate::dprivacy::budget::SegmentBudget;
use crate::dprivacy::RdpAccounting;
use crate::logging::{RuntimeKind, RuntimeMeasurement};
use crate::request::{Request, RequestId};
use crate::schema::Schema;
use crate::{composition, dprivacy, AccountingType};
use float_cmp::{ApproxEq, F64Margin};
use grb::expr::LinExpr;
use grb::prelude::*;
use grb::Error;
use itertools::Itertools;
use log::{trace, warn};
use stats::{
    ConstraintRdpInfo, IlpRdpStats, IlpStats, MeanAndStddev, SegmentInfos, SegmentRdpInfo,
};
use std::collections::{btree_map, BTreeMap, BTreeSet, HashMap, HashSet};
use std::process::Command;
use std::time::Instant;

pub struct Ilp {}

/// When assigning values to binary variables, gurobi doesn't always assign 0 or 1, it may also
/// assign a value "close" to one of these values, e.g., 1e-5 is fine. To take this into account,
/// we define an integrality margin with the same limits as used by gurobi, to check the resulting
/// assignments by gurobi.
pub static ILP_INTEGRALITY_MARGIN: F64Margin = F64Margin {
    epsilon: 1e-5,
    ulps: 0,
};

struct ModelAndVars {
    model: Model,
    y_vars: BTreeMap<RequestId, grb::Var>,
    x_vars: BTreeMap<(RequestId, BlockId), grb::Var>,
}

/// Contains all information to uniquely identify a segment
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct SegmentId {
    /// Which block the segment belongs to
    block_id: BlockId,
    /// Which requests are part of this segment
    request_ids: BTreeSet<RequestId>,
}

impl Ilp {
    pub fn construct_allocator() -> Self {
        Ilp {}
    }

    #[allow(clippy::too_many_arguments)]
    pub fn round<M: SegmentBudget>(
        &mut self,
        candidate_requests: &HashMap<RequestId, Request>,
        request_history: &HashMap<RequestId, Request>,
        available_blocks: &HashMap<BlockId, Block>,
        schema: &Schema,
        block_comp_wrapper: &BlockCompWrapper,
        alphas: &Option<AccountingType>,
        runtime_measurements: &mut Vec<RuntimeMeasurement>,
    ) -> (ResourceAllocation, AllocationStatus) {
        trace!("Building problem formulation");
        trace!(
            "{}",
            Self::get_free_memory().unwrap_or_else(|| "Couldn't get free memory".to_string())
        );
        let mut pf: ProblemFormulation<M> = block_comp_wrapper.build_problem_formulation::<M>(
            available_blocks,
            candidate_requests,
            request_history,
            schema,
            runtime_measurements,
        );

        let num_contested_segments_initially = pf.contested_constraints().count();

        let mut alloc_meas = RuntimeMeasurement::start(RuntimeKind::RunAllocationAlgorithm);

        let mut resource_allocation = ResourceAllocation {
            accepted: HashMap::new(),
            rejected: HashSet::new(),
        };

        trace!("Allocating acceptable requests");
        trace!(
            "{}",
            Self::get_free_memory().unwrap_or_else(|| "Couldn't get free memory".to_string())
        );
        // first, allocate acceptable requests
        while let Some(composition::AcceptableRequest {
            request_id: rid,
            acceptable,
            contested: _,
        }) = pf.next_acceptable(
            Some(BlockOrderStrategy::BlockCreation {
                block_lookup: available_blocks,
            }),
            candidate_requests,
        ) {
            let request_n_users = candidate_requests[&rid].n_users;
            let selected_blocks = acceptable.into_iter().take(request_n_users).collect();
            pf.allocate_request(rid, &selected_blocks, candidate_requests)
                .expect("Allocating request failed");
            resource_allocation.accepted.insert(rid, selected_blocks);
        }

        let mut ilp_build_meas = RuntimeMeasurement::start(RuntimeKind::BuildIlp);

        trace!("Building Model");
        trace!(
            "{}",
            Self::get_free_memory().unwrap_or_else(|| "Couldn't get free memory".to_string())
        );
        // build up the model
        let (mut model_vars, segment_infos) =
            Ilp::build_model::<M>(&pf, candidate_requests, alphas).expect("Building Ilp failed");

        // Adding variables and constraints is lazy - the call to update ensures that all
        // datastructures are in place, so all operations to model can be applied
        model_vars.model.update().expect("Updating model failed");

        let mut ilp_stats = Self::get_ilp_stats(
            &model_vars,
            available_blocks,
            segment_infos,
            num_contested_segments_initially,
        );

        runtime_measurements.push(ilp_build_meas.stop());
        let mut ilp_solve_meas = RuntimeMeasurement::start(RuntimeKind::SolveIlp);

        trace!("Optimizing Model");
        trace!(
            "{}",
            Self::get_free_memory().unwrap_or_else(|| "Couldn't get free memory".to_string())
        );
        // optimize the model
        Ilp::optimize_model(&mut model_vars.model).expect("Optimizing model failed");

        runtime_measurements.push(ilp_solve_meas.stop());
        let mut ilp_post_meas = RuntimeMeasurement::start(RuntimeKind::PostIlp);

        trace!("Updating resource allocation");
        trace!(
            "{}",
            Self::get_free_memory().unwrap_or_else(|| "Couldn't get free memory".to_string())
        );
        // update resource_allocation
        let failed_and_retried = Ilp::update_resource_allocation::<M>(
            &model_vars,
            &mut resource_allocation,
            candidate_requests,
            &mut pf,
        )
        .expect("Updating resource allocation failed");

        ilp_stats.failed_and_retried = Some(failed_and_retried);

        let ilp_stats = ilp_stats;

        // all requests which are not yet in the resource_allocation must be rejected
        for rid in candidate_requests.keys() {
            if !resource_allocation.accepted.contains_key(rid)
                && !resource_allocation.rejected.contains(rid)
            {
                resource_allocation.rejected.insert(*rid);
            }
        }

        let status: AllocationStatus = {
            let res = std::panic::catch_unwind(|| {
                model_vars
                    .model
                    .status()
                    .expect("Couldn't retrieve model status")
            });
            match res {
                Ok(status) => AllocationStatus::IlpRegularStatus(status, ilp_stats),
                Err(_err) => AllocationStatus::IlpOtherStatus("Other Status", ilp_stats),
            }
        };

        runtime_measurements.push(ilp_post_meas.stop());
        runtime_measurements.push(alloc_meas.stop());

        (resource_allocation, status)
    }

    fn get_ilp_stats(
        model_vars: &ModelAndVars,
        available_blocks: &HashMap<BlockId, Block>,
        segment_infos: SegmentInfos,
        num_contested_segments_initially: usize,
    ) -> IlpStats {
        let num_contested_segments: usize = match &segment_infos {
            SegmentInfos::NoSegments => 0,
            SegmentInfos::RdpSegInfos(map) => map.len(),
            SegmentInfos::NonRdpSegInfos(set) => set.len(),
        };
        let contested_segments_per_block: MeanAndStddev = match &segment_infos {
            SegmentInfos::NoSegments => MeanAndStddev {
                mean: 0.0,
                stddev: 0.0,
            },
            SegmentInfos::RdpSegInfos(map) => {
                let mut segments_per_block: BTreeMap<BlockId, usize> = BTreeMap::new();
                for seg_id in map.keys() {
                    *segments_per_block.entry(seg_id.block_id).or_insert(0) += 1;
                }
                MeanAndStddev::new(
                    &segments_per_block
                        .values()
                        .map(|x| *x as f64)
                        .collect::<Vec<_>>(),
                )
            }
            SegmentInfos::NonRdpSegInfos(set) => {
                let mut segments_per_block: BTreeMap<BlockId, usize> = BTreeMap::new();
                for seg_id in set {
                    *segments_per_block.entry(seg_id.block_id).or_insert(0) += 1;
                }
                MeanAndStddev::new(
                    &segments_per_block
                        .values()
                        .map(|x| *x as f64)
                        .collect::<Vec<_>>(),
                )
            }
        };

        let rdp_stats: Option<IlpRdpStats> = match &segment_infos {
            SegmentInfos::RdpSegInfos(segment_info_map) => Some(IlpRdpStats {
                n_alphas_no_local_reduction: {
                    let info_type = |cstr_info: &ConstraintRdpInfo| cstr_info.alphas_no_red;
                    Self::aggregate_constraint_infos(segment_info_map, info_type)
                },
                n_alphas_after_local_reduction: {
                    let info_type = |cstr_info: &ConstraintRdpInfo| cstr_info.alphas_after_red;
                    Self::aggregate_constraint_infos(segment_info_map, info_type)
                },
                n_alphas_eliminated_budget_reduction: {
                    let info_type =
                        |cstr_info: &ConstraintRdpInfo| cstr_info.alpha_removed_budget_red;
                    Self::aggregate_optional_constraint_infos(segment_info_map, info_type)
                },
                n_alphas_eliminated_ratio_reduction: {
                    let info_type =
                        |cstr_info: &ConstraintRdpInfo| cstr_info.alpha_removed_ratio_red;
                    Self::aggregate_optional_constraint_infos(segment_info_map, info_type)
                },
                n_alphas_eliminated_combinatorial_reduction: {
                    let info_type =
                        |cstr_info: &ConstraintRdpInfo| cstr_info.alpha_removed_combinatorial_red;
                    Self::aggregate_optional_constraint_infos(segment_info_map, info_type)
                },
            }),
            _ => None,
        };
        IlpStats {
            num_vars: model_vars
                .model
                .get_attr(grb::attribute::attr::NumVars)
                .expect("Couldn't get number of variables"),
            num_int_vars: model_vars
                .model
                .get_attr(grb::attribute::attr::NumIntVars)
                .expect("Couldn't get number of integer variables"),
            num_bin_vars: model_vars
                .model
                .get_attr(grb::attribute::attr::NumBinVars)
                .expect("Couldn't get number of binary variables"),
            num_constr: model_vars
                .model
                .get_attr(grb::attribute::attr::NumConstrs)
                .expect("Couldn't get number of constraints"),
            num_nz_coeffs: model_vars
                .model
                .get_attr(grb::attribute::attr::DNumNZs)
                .expect("Couldn't get number of non-zero coefficients"),
            num_blocks: available_blocks.len(),
            num_contested_segments_initially,
            ilp_num_contested_segments: num_contested_segments,
            contested_segments_per_block,
            rdp_stats,
            failed_and_retried: None,
        }
    }

    /// Used to aggregate information over all segment rdp infos contained in the passed argument
    fn aggregate_constraint_infos(
        segment_info_map: &BTreeMap<SegmentId, SegmentRdpInfo>,
        info_type: fn(&ConstraintRdpInfo) -> usize,
    ) -> usize {
        segment_info_map
            .values()
            .map(|seg_rdp_info| -> usize {
                seg_rdp_info.constraint_infos.iter().map(info_type).sum()
            })
            .sum::<usize>()
    }

    /// Similar to [aggregate_constraint_infos](Self::aggregate_constraint_infos), but works with
    /// optional values and returns a [MeanAndStddev] instead of a usize
    fn aggregate_optional_constraint_infos(
        segment_info_map: &BTreeMap<SegmentId, SegmentRdpInfo>,
        info_type: fn(&ConstraintRdpInfo) -> Option<usize>,
    ) -> Option<MeanAndStddev> {
        let optional_vals = segment_info_map
            .values()
            .flat_map(|seg_rdp_info| seg_rdp_info.constraint_infos.iter().map(info_type))
            .collect::<Vec<_>>();
        let original_length = optional_vals.len();
        let vals: Vec<f64> = optional_vals
            .into_iter()
            .filter_map(|opt| opt.map(|x| x as f64))
            .collect();
        let filtered_length = vals.len();
        if filtered_length >= 1 && filtered_length < original_length {
            warn!(
                "When aggregating optional SegmentRdpInfo fields, some of those fields were None \
            and some were Some"
            )
        }
        if filtered_length >= 1 {
            let res = MeanAndStddev::new(&vals);
            Some(res)
        } else {
            None
        }
    }

    fn get_free_memory() -> Option<String> {
        if let Ok(command_output) = Command::new("sh")
            .arg("-c")
            .arg("cat /proc/meminfo | grep MemFree: | tr -d '\n'")
            .output()
        {
            if let Ok(stdout_string) = String::from_utf8(command_output.stdout) {
                return Some(stdout_string);
            }
        }

        None
    }

    fn update_resource_allocation<M: SegmentBudget>(
        model_vars: &ModelAndVars,
        resource_allocation: &mut ResourceAllocation,
        candidate_requests: &HashMap<RequestId, Request>,
        pf: &mut ProblemFormulation<M>,
    ) -> Result<FailedAndRetried, Error> {
        // first, see which yis are 1, i.e., which requests were accepted, and add those to the
        // problem formulation, together with all their acceptable blocks
        for (request_id, y_var) in model_vars.y_vars.iter() {
            let result: f64 = model_vars.model.get_obj_attr(attr::X, y_var)?;
            if result.approx_eq(1.0, ILP_INTEGRALITY_MARGIN) {
                let acceptable_blocks: HashSet<BlockId> = {
                    let status_result = pf.request_status(*request_id, None, candidate_requests);
                    match status_result {
                        StatusResult::Contested {
                            acceptable,
                            contested: _,
                        } => HashSet::<BlockId>::from_iter(acceptable),
                        _ => {
                            panic!("Did assign a non-contested request in ilp")
                        }
                    }
                };

                let inserted = resource_allocation
                    .accepted
                    .insert(*request_id, acceptable_blocks);
                assert!(inserted.is_none());
            } else {
                let inserted = resource_allocation.rejected.insert(*request_id);
                assert!(inserted);
            }
        }

        // then, add contested blocks to the accepted requests, as assigned by the ilp using the
        // xij variables
        for ((request_id, block_id), x_var) in model_vars.x_vars.iter() {
            let request = &candidate_requests[request_id];
            let result: f64 = model_vars.model.get_obj_attr(attr::X, x_var)?;
            if result.approx_eq(1.0, ILP_INTEGRALITY_MARGIN) {
                // xij might be 1 for a yi where yi is 0 in principle
                if resource_allocation.accepted.contains_key(request_id)
                    && resource_allocation.accepted[request_id].len() < request.n_users
                {
                    resource_allocation
                        .accepted
                        .get_mut(request_id)
                        .unwrap()
                        .insert(*block_id);
                }
            }
        }

        let check_start = Instant::now();
        trace!("Starting ilp allocation check");
        let failed_and_retried = Ilp::check_resource_allocation(
            resource_allocation,
            model_vars,
            candidate_requests,
            pf,
        )?;
        trace!(
            "Finished ilp allocation check in {:?} ms",
            check_start.elapsed().as_millis()
        );

        Ok(failed_and_retried)
    }

    /// For each request allocated by the ilp, we check if we can also allocate it using the
    /// problem formulation, which includes various checks regarding the validity of the assignment.
    /// Due to numerical tolerances differing, some requests allocated in the ilp might not be
    /// allocatable in the problem formulation. This function checks for those, and removes them
    /// from the resource allocation.
    fn check_resource_allocation<M: SegmentBudget>(
        resource_allocation: &mut ResourceAllocation,
        model_vars: &ModelAndVars,
        candidate_requests: &HashMap<RequestId, Request>,
        pf: &mut ProblemFormulation<M>,
    ) -> Result<FailedAndRetried, grb::Error> {
        // try to allocate in the problem formulation all requests allocated by ilp
        // note that this in != resource_allocation.accepted, because only contested segments
        // were assigned in the ilp, while accepted ones are already accounted for in this problem
        // formulation

        // Contains all requests where allocation failed even though they were allocated by the ilp
        let failed_allocations: BTreeSet<RequestId> = BTreeSet::new();
        // Contains all requests that were allocated greedily (if one or more requests allocated
        // by the ilp could not be allocated by the problem formulation)
        let greedily_allocated: BTreeSet<RequestId> = BTreeSet::new();

        let mut failed_and_retried = FailedAndRetried {
            failed_allocations,
            greedily_allocated,
        };

        /*
        trace!(
            "Resource allocation before checking: {:?}",
            resource_allocation
        );
         */

        let mut any_allocation_failed: bool = false;

        // we sort the requests by profit to avoid not being able to allocate a high-profit
        // request due to numeric tolerance differences between ilp solver and this program.
        let sorted_requests = model_vars
            .y_vars
            .iter()
            .sorted_by(|a, b| {
                candidate_requests[a.0]
                    .profit
                    .cmp(&candidate_requests[b.0].profit)
            })
            .collect::<Vec<_>>();

        trace!("Order of request allocation: {:?}", sorted_requests);

        for (request_id, y_var) in sorted_requests {
            let result: f64 = model_vars.model.get_obj_attr(attr::X, y_var)?;
            if result.approx_eq(1.0, ILP_INTEGRALITY_MARGIN) {
                assert!(
                    resource_allocation.accepted.contains_key(request_id),
                    "Request {} was assigned in the ilp, but not in the resource allocation",
                    request_id
                );
                match pf.allocate_request(
                    *request_id,
                    &resource_allocation.accepted[request_id],
                    candidate_requests,
                ) {
                    Ok(_) => {}
                    Err(e) => {
                        any_allocation_failed = true;
                        resource_allocation
                            .accepted
                            .remove(request_id)
                            .expect("Removing request from resource allocation failed");
                        resource_allocation.rejected.insert(*request_id);
                        let inserted = failed_and_retried.failed_allocations.insert(*request_id);
                        assert!(
                            inserted,
                            "Request {} was already in the failed allocations set",
                            request_id
                        );
                        match e {
                            AllocationError::IllegalRequestId(_) => {
                                warn!(
                                    "Could not allocate request {} in the problem formulation: {:?}",
                                    request_id, e
                                )
                            }
                            AllocationError::IllegalBlockAssignment(_) => {
                                warn!(
                                    "Could not allocate request {} in the problem formulation: {:?}",
                                    request_id, e
                                )
                            }
                        }
                    }
                }
            }
        }

        // if any allocation failed, greedily try to allocate failed requests
        if any_allocation_failed {
            let greedy_candidates = resource_allocation
                .rejected
                .drain()
                .sorted_by(|a, b| {
                    candidate_requests[a]
                        .profit
                        .cmp(&candidate_requests[b].profit)
                })
                .collect::<Vec<_>>();
            warn!(
                "Some requests could not be allocated in the problem formulation, likely due \
                    to numeric issues. Trying to greedily allocate rejected requests: {:?}",
                &greedy_candidates
            );
            // Order requests by id, and greedily try to allocate by profit
            for rid in greedy_candidates.into_iter() {
                let mut selected_blocks: HashSet<BlockId>;
                match pf.request_status(rid, Some(BlockOrderStrategy::Id), candidate_requests) {
                    StatusResult::Acceptable {
                        acceptable: acc_bids,
                        contested: _,
                    } => {
                        selected_blocks = acc_bids
                            .into_iter()
                            .take(candidate_requests[&rid].n_users)
                            .collect();
                    }
                    StatusResult::Contested {
                        acceptable: acc_bids,
                        contested: con_bids,
                    } => {
                        selected_blocks = acc_bids
                            .into_iter()
                            .take(candidate_requests[&rid].n_users)
                            .collect();
                        let still_needed_blocks =
                            candidate_requests[&rid].n_users - selected_blocks.len();
                        selected_blocks.extend(con_bids.into_iter().take(still_needed_blocks));
                    }
                    StatusResult::Rejected => {
                        resource_allocation.rejected.insert(rid);
                        continue;
                    }
                }
                let inserted = failed_and_retried.greedily_allocated.insert(rid);
                assert!(
                    inserted,
                    "Could not insert request {} into greedily_allocated",
                    rid
                );
                trace!("Allocated request {} greedily", rid);
                assert_eq!(
                    selected_blocks.len(),
                    candidate_requests[&rid].n_users,
                    "Not enough blocks to accept"
                );
                pf.allocate_request(rid, &selected_blocks, candidate_requests)
                    .expect("Allocating request failed");
                resource_allocation.accepted.insert(rid, selected_blocks);
            }
        }
        // assert!(pf.next_acceptable(None, candidate_requests).is_none());
        // assert!(pf.contested_constraints().next().is_none());

        Ok(failed_and_retried)
    }

    fn optimize_model(model: &mut Model) -> Result<(), Error> {
        model.optimize()?;
        // assert_eq!(model.status()?, Status::Optimal);
        Ok(())
    }

    /// builds the ilp model, and returns the model and the contained variables in [ModelAndVars],
    /// as well as information about the segments.
    fn build_model<M: SegmentBudget>(
        pf: &ProblemFormulation<M>,
        candidate_requests: &HashMap<RequestId, Request>,
        alphas: &Option<AccountingType>,
    ) -> Result<(ModelAndVars, SegmentInfos), grb::Error> {
        // init model
        let mut model = Model::new("ilp-allocator")?;

        // For each request i, contains yi
        let mut y_vars: BTreeMap<RequestId, grb::Var> = BTreeMap::new();
        // For each request i and block j, contains xij
        let mut x_vars: BTreeMap<(RequestId, BlockId), grb::Var> = BTreeMap::new();

        // Note: Here we use optimal budget to not delete any constraints.
        let mut budget_constraints: BTreeMap<(BlockId, BTreeSet<RequestId>), OptimalBudget> =
            BTreeMap::new();
        // go through contested segments, add all yi and xij variables to model, and build up
        // some key datastructures
        {
            for (bid, rids, budget) in pf.contested_constraints() {
                // Add new variables yi if it does not exist already);
                for rid in rids.iter() {
                    if !y_vars.contains_key(rid) {
                        // for each such "new" contested requests i, we create a new variable y_i
                        let varname = format!("y[{}]", rid);
                        let yi = add_binvar!(model, name: &varname)?;
                        y_vars.insert(*rid, yi);
                    }
                }

                // add new variables xij if it does not exist already (same request may appear
                // in multiple segments, and segment merging may not be enabled, so need to
                // check for duplicates)
                for rid in rids.iter() {
                    if let btree_map::Entry::Vacant(e) = x_vars.entry((*rid, bid)) {
                        let varname = format!("x[{},{}]", rid, bid);
                        let xij = add_binvar!(model, name: &varname)?;
                        e.insert(xij);
                    }
                }

                // Add all budget constraints to map
                let e: &mut OptimalBudget = budget_constraints
                    .entry((bid, rids))
                    .or_insert_with(OptimalBudget::new);
                for budget_constr in budget.get_budget_constraints() {
                    e.add_budget_constraint(budget_constr);
                }
            }
        }

        // in case no variables were added, we can return safely.
        if y_vars.is_empty() && x_vars.is_empty() {
            return Ok((
                ModelAndVars {
                    model,
                    y_vars,
                    x_vars,
                },
                SegmentInfos::NoSegments,
            ));
        }

        trace!("Contested requests: {:?}", y_vars.keys());

        // Construct the constraint to adhere to the specified minimum number of blocks for each
        // request
        {
            struct LhsRhs {
                lhs: LinExpr,
                rhs: LinExpr,
            }

            let mut min_blocks_constraints: BTreeMap<RequestId, LhsRhs> = BTreeMap::new();
            for (rid, yi) in y_vars.iter() {
                let n_blocks_needed =
                    candidate_requests[rid].n_users - pf.n_acceptable_blocks(*rid);
                let mut rhs = LinExpr::new();
                rhs.add_term(n_blocks_needed as f64, *yi);
                min_blocks_constraints.insert(
                    *rid,
                    LhsRhs {
                        lhs: LinExpr::new(),
                        rhs,
                    },
                );
            }

            for ((rid, _bid), xij) in x_vars.iter() {
                min_blocks_constraints
                    .get_mut(rid)
                    .expect("Did not find min_block constraint by rid")
                    .lhs
                    .add_term(1., *xij);
            }

            for (rid, mut lhs_rhs) in min_blocks_constraints.into_iter() {
                let name = &format!("min_n_blocks_{}", rid);
                assert_eq!(lhs_rhs.rhs.num_terms(), 1);
                // Need to scale both lhs and rhs, where rhs is also a LinExpr
                let rhs_coeff = *lhs_rhs.rhs.iter_terms().next().unwrap().1;
                Ilp::scale_rhs_and_lhs(&mut lhs_rhs.lhs, &mut rhs_coeff.clone());
                Ilp::scale_rhs_and_lhs(&mut lhs_rhs.rhs, &mut rhs_coeff.clone());

                let constr = c!(lhs_rhs.lhs >= lhs_rhs.rhs);
                model.add_constr(name, constr)?;
            }
        }
        assert!(
            !budget_constraints.is_empty(),
            "No budget constraints added"
        );

        let segment_info: SegmentInfos;

        // if candidate request is a1, we try to convert all budget constraints and request costs to eps dp
        if candidate_requests
            .values()
            .next()
            .expect("Candidate requests empty")
            .request_cost
            .is_a1()
        {
            let budget_constraints_eps_dp: BTreeMap<(BlockId, BTreeSet<RequestId>), OptimalBudget> =
                budget_constraints
                    .into_iter()
                    .map(|(index, budget)| (index, budget.a1_to_eps_dp()))
                    .collect();

            let candidate_requests_eps_dp: &HashMap<RequestId, Request> = &candidate_requests
                .iter()
                .map(|(rid, req)| {
                    let mut new_req = req.clone();
                    new_req.request_cost = req.request_cost.a1_to_eps_dp();
                    (*rid, new_req)
                })
                .collect();
            let res = Ilp::add_budget_constraints_no_rdp(
                &mut model,
                &x_vars,
                budget_constraints_eps_dp,
                candidate_requests_eps_dp,
            )?;
            segment_info = SegmentInfos::NonRdpSegInfos(res);
        } else {
            let all_rdp = budget_constraints.values().all(|budget| budget.is_rdp());
            let no_rdp = budget_constraints.values().all(|budget| !budget.is_rdp());

            assert!(all_rdp || no_rdp, "Some budgets have rdp and others don't");
            // now add budget constraints to the model, depending on if we have rdp or not
            if all_rdp {
                let res = Ilp::add_budget_constraints_with_rdp(
                    &mut model,
                    &x_vars,
                    budget_constraints,
                    candidate_requests,
                    alphas.as_ref().expect("Need alphas to build ilp with rdp"),
                )?;
                segment_info = SegmentInfos::RdpSegInfos(res);
            } else {
                let res = Ilp::add_budget_constraints_no_rdp(
                    &mut model,
                    &x_vars,
                    budget_constraints,
                    candidate_requests,
                )?;
                segment_info = SegmentInfos::NonRdpSegInfos(res);
            }
        }

        // Add the objective to the model
        {
            let mut objective = LinExpr::new();
            // for new each variable yi, add term (profit_i * y_i) to objective
            let mut total_profit = 0f64;
            for (rid, yi) in y_vars.iter() {
                let profit = candidate_requests[rid].profit as f64;
                total_profit += profit;
                objective.add_term(profit, *yi);
            }

            Ilp::scale_rhs_and_lhs(&mut objective, &mut total_profit);

            model.set_objective(objective, grb::ModelSense::Maximize)?;
        }

        assert_eq!(model.get_attr(attr::IsMIP)?, 1);

        Ok((
            ModelAndVars {
                model,
                y_vars,
                x_vars,
            },
            segment_info,
        ))
    }

    fn add_budget_constraints_no_rdp(
        model: &mut Model,
        x_vars: &BTreeMap<(RequestId, BlockId), grb::Var>,
        budget_constraints: BTreeMap<(BlockId, BTreeSet<RequestId>), OptimalBudget>,
        candidate_requests: &HashMap<RequestId, Request>,
    ) -> Result<BTreeSet<SegmentId>, grb::Error> {
        let mut segments: BTreeSet<SegmentId> = BTreeSet::new();
        // since there is no rdp, each kind of cost (eps and delta, if both present) must be
        // observed
        for (seg_id, ((bid, rids), budget)) in budget_constraints.into_iter().enumerate() {
            segments.insert(SegmentId {
                block_id: bid,
                request_ids: rids.clone(),
            });
            let mut eps_constr_lhs = LinExpr::new();
            let mut del_constr_lhs = LinExpr::new();

            // For eps (and delta, if present), we construct an expression summing up the cost
            // of all allocated requests in a segment
            for rid in rids.iter() {
                let cost = &candidate_requests[rid].request_cost;
                let xij = x_vars.get(&(*rid, bid)).expect("Did not find xij");
                match cost {
                    AccountingType::EpsDp { eps } => {
                        eps_constr_lhs.add_term(*eps, *xij);
                    }
                    AccountingType::EpsDeltaDp { eps, delta } => {
                        eps_constr_lhs.add_term(*eps, *xij);
                        del_constr_lhs.add_term(*delta, *xij);
                    }
                    AccountingType::Rdp { .. } => {
                        panic!("Some requests have rdp, but budgets don't")
                    }
                }
            }

            // and then add constraints making sure, that for each segment, the costs are not larger
            // than any budget in that segment.
            for (constr_id, budget_constr) in
                budget.get_budget_constraints().into_iter().enumerate()
            {
                let constr_name = format!("budget_constr_seg_{}_constr_{}", seg_id, constr_id);
                match budget_constr {
                    AccountingType::EpsDp { eps } => {
                        assert!(del_constr_lhs.is_empty());

                        let mut lhs = eps_constr_lhs.clone();
                        let mut rhs = *eps;

                        Ilp::scale_rhs_and_lhs(&mut lhs, &mut rhs);

                        model.add_constr(&format!("Eps_{}", constr_name), c!(lhs <= rhs))?;
                    }
                    AccountingType::EpsDeltaDp { eps, delta } => {
                        assert_eq!(eps_constr_lhs.num_terms(), del_constr_lhs.num_terms());

                        let mut lhs_eps = eps_constr_lhs.clone();
                        let mut rhs_eps = *eps;

                        Ilp::scale_rhs_and_lhs(&mut lhs_eps, &mut rhs_eps);

                        model
                            .add_constr(&format!("Eps_{}", constr_name), c!(lhs_eps <= rhs_eps))?;

                        let mut lhs_del = del_constr_lhs.clone();
                        let mut rhs_del = *delta;

                        Ilp::scale_rhs_and_lhs(&mut lhs_del, &mut rhs_del);

                        model.add_constr(
                            &format!("Delta_{}", constr_name),
                            c!(lhs_del <= rhs_del),
                        )?;
                    }
                    AccountingType::Rdp { .. } => {
                        panic!("Wrongly called ilp constructor without rdp, but budget has rdp")
                    }
                }
            }
        }
        Ok(segments)
    }

    fn add_budget_constraints_with_rdp(
        model: &mut Model,
        x_vars: &BTreeMap<(RequestId, BlockId), grb::Var>,
        budget_constraints: BTreeMap<(BlockId, BTreeSet<RequestId>), OptimalBudget>,
        candidate_requests: &HashMap<RequestId, Request>,
        alphas: &AccountingType,
    ) -> Result<BTreeMap<SegmentId, SegmentRdpInfo>, grb::Error> {
        let mut segment_infos: BTreeMap<SegmentId, SegmentRdpInfo> = BTreeMap::new();

        if budget_constraints.is_empty() {
            return Ok(BTreeMap::new());
        }
        for (seg_id, ((bid, rids), budget)) in budget_constraints.into_iter().enumerate() {
            // Contains the needed information for a budget constraints after removing unneeded
            // alphas
            struct BudgetConstr {
                id: usize,
                budget: Vec<f64>,
                costs: Vec<LinExpr>,
            }
            // calculate the trimmed constraints, by removing unnecessary alpha values
            let trimmed_budget_constrs_and_info: Vec<(Option<BudgetConstr>, ConstraintRdpInfo)> =
                budget
                    .get_budget_constraints()
                    .into_iter()
                    .enumerate()
                    .map(|(constr_id, budget_constr)| {
                        let costs: Vec<AccountingType> = rids
                            .iter()
                            .map(|rid| candidate_requests[rid].request_cost.clone())
                            .collect();
                        let needed_alphas_and_info =
                            dprivacy::rdpopt::RdpOptimization::calc_needed_alphas(
                                alphas,
                                &costs,
                                budget_constr,
                                crate::dprivacy::AlphaReductions {
                                    budget_reduction: true,
                                    ratio_reduction: true,
                                    combinatorial_reduction: false,
                                },
                            );
                        let alpha_red_result = needed_alphas_and_info.1;
                        let needed_alphas: Vec<f64> = needed_alphas_and_info
                            .0
                            .into_iter()
                            .map(|pacb| pacb.alpha)
                            .collect();
                        let constraint_info = ConstraintRdpInfo {
                            alphas_no_red: budget_constr.get_rdp_vec().len(),
                            alphas_after_red: needed_alphas.len(),
                            alpha_removed_budget_red: alpha_red_result.budget_reduction,
                            alpha_removed_ratio_red: alpha_red_result.ratio_reduction,
                            alpha_removed_combinatorial_red: alpha_red_result
                                .combinatorial_reduction,
                        };
                        if needed_alphas.is_empty() {
                            return (None, constraint_info);
                        }
                        let mask: Vec<bool> = alphas
                            .get_rdp_vec()
                            .into_iter()
                            .map(|alpha| {
                                needed_alphas.iter().any(|needed_alpha| {
                                    needed_alpha.approx_eq(alpha, F64Margin::default())
                                })
                            })
                            .collect();
                        let filtered_budget = budget_constr
                            .get_rdp_vec()
                            .into_iter()
                            .zip_eq(&mask)
                            .filter_map(|(bj, needed)| if *needed { Some(bj) } else { None })
                            .collect::<Vec<_>>();
                        let mut filtered_costs: Vec<LinExpr> =
                            (0..needed_alphas.len()).map(|_| LinExpr::new()).collect();
                        for rid in rids.iter() {
                            let cost_vec = candidate_requests[rid].request_cost.get_rdp_vec();
                            let filtered_cost_vec = cost_vec
                                .into_iter()
                                .zip_eq(&mask)
                                .filter_map(|(ci, needed)| if *needed { Some(ci) } else { None });
                            let xij = x_vars.get(&(*rid, bid)).expect("Did not find xij");
                            for (ci, acc) in filtered_cost_vec
                                .into_iter()
                                .zip_eq(filtered_costs.iter_mut())
                            {
                                acc.add_term(ci, *xij);
                            }
                        }

                        (
                            Some(BudgetConstr {
                                id: constr_id,
                                budget: filtered_budget,
                                costs: filtered_costs,
                            }),
                            constraint_info,
                        )
                    })
                    .collect();

            let inserted = segment_infos.insert(
                SegmentId {
                    block_id: bid,
                    request_ids: rids,
                },
                SegmentRdpInfo {
                    constraint_infos: trimmed_budget_constrs_and_info
                        .iter()
                        .map(|(_, constr_info)| constr_info)
                        .copied()
                        .collect(),
                },
            );

            assert!(inserted.is_none(), "Already inserted the segment info");

            let trimmed_budget_constrs: Vec<BudgetConstr> = trimmed_budget_constrs_and_info
                .into_iter()
                .filter_map(|(budget_constr_option, _)| budget_constr_option)
                .collect();

            // Now we add constraints to the model, making sure that if the constraints are
            // satisfied, then for each budget constraint in the current segment, at least for one
            // alpha value the costs are <= than the budget.
            for budget_constr in trimmed_budget_constrs {
                let num_alpha_values: usize = budget_constr.budget.len();
                assert!(num_alpha_values > 0);
                // Add variables a_jvlk to the model
                let mut a_vec: Vec<grb::Var> = Vec::with_capacity(num_alpha_values);
                if num_alpha_values > 1 {
                    for k in 0..num_alpha_values {
                        let varname = format!("a[{},{},{}]", seg_id, budget_constr.id, k);
                        // j: block id, v: segment on block (together: seg_id),
                        // l: constr_id, k: which alpha
                        let ajvlk = add_binvar!(model, name: &varname)?;
                        a_vec.push(ajvlk);
                    }
                }

                // Add constraint to ensure that we need to observe the budget constraint for at
                // least one value of alpha (i.e., for each j, v and l, for one k we must
                // have ajvlk = 0)
                let constr_name =
                    format!("alphas_constr_seg_{}_constr_{}", seg_id, budget_constr.id);
                let mut lhs = LinExpr::new();

                let mut lhs_expanded_constrs = budget_constr.costs.clone();
                // no need to add the as for this constraint, if there is only one a anyways
                if num_alpha_values > 1 {
                    for ajvlk in a_vec.iter() {
                        lhs.add_term(1., *ajvlk);
                    }

                    model.add_constr(&constr_name, c!(lhs == num_alpha_values - 1))?;

                    // use the a's to expand budget constraint
                    for (linexpr, ajvlk) in lhs_expanded_constrs.iter_mut().zip_eq(a_vec) {
                        // sum up costs of all requests on this segment and alpha_value
                        let total_cost = linexpr.iter_terms().fold(0., |acc, term| acc + term.1);
                        // if ajvlk = 1, we can basically ignore this constraint
                        linexpr.add_term(-total_cost, ajvlk);
                    }
                }

                let budget_constr_vec = budget_constr.budget;
                for (k, (mut lhs, rhs_part)) in lhs_expanded_constrs
                    .into_iter()
                    .zip_eq(budget_constr_vec.into_iter())
                    .enumerate()
                {
                    let constr_name = format!(
                        "budget_constr_seg_{}_constr_{}_alpha_ind_{}",
                        seg_id, budget_constr.id, k
                    );

                    let mut rhs = rhs_part.max(0.);
                    Ilp::scale_rhs_and_lhs(&mut lhs, &mut rhs);

                    model.add_constr(&constr_name, c!(lhs <= rhs))?;
                }
            }
        }
        Ok(segment_infos)
    }

    /// Following the recommendations from
    /// <https://www.gurobi.com/documentation/9.5/refman/recommended_ranges_for_var.html>
    /// we want to make sure that the rhs is somewhere between 1 and 10^4, and else scale
    /// rhs and lhs appropriately
    fn scale_rhs_and_lhs(lhs: &mut LinExpr, rhs: &mut f64) {
        assert!(*rhs >= 0.0, "rhs must not be negative");
        if *rhs < 0.001 {
            warn!("rhs is very small, ilp may produce inaccurate results");
        }
        if rhs.approx_eq(0.0, F64Margin::default()) {
        } else if *rhs < 1.0 {
            let scale_factor = 1.0 / *rhs;
            *rhs *= scale_factor;
            lhs.mul_scalar(scale_factor);
        } else if *rhs > 1e4 {
            let scale_factor = 1e4 / *rhs;
            *rhs *= scale_factor;
            lhs.mul_scalar(scale_factor);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::allocation::BlockCompWrapper;
    use crate::composition::{block_composition, block_composition_pa};
    use crate::config::SegmentationAlgo;
    use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
    use crate::AccountingType::{EpsDp, Rdp};
    use crate::{OptimalBudget, RequestId};
    use float_cmp::{ApproxEq, F64Margin};
    use grb::add_binvar;
    use grb::expr::LinExpr;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_scale_rhs_and_lhs() {
        let margin = F64Margin::default();
        let mut model = grb::Model::new("scaling test").unwrap();
        let var1 = add_binvar!(model, name: "var1").unwrap();
        let var2 = add_binvar!(model, name: "var2").unwrap();
        let mut lhs = LinExpr::new();
        lhs.add_term(1.0, var1);
        lhs.add_term(4.0, var2);

        //---------------------------------------

        let mut rhs1 = 0.01;
        let mut lhs1 = lhs.clone();

        super::Ilp::scale_rhs_and_lhs(&mut lhs1, &mut rhs1);

        assert!(rhs1.approx_eq(1.00, margin));
        for (var, val) in lhs1.iter_terms() {
            if var == &var1 {
                assert!(val.approx_eq(100.0, margin));
            } else if var == &var2 {
                assert!(val.approx_eq(400.0, margin));
            } else {
                panic!("Unexpected term in LinExpr")
            }
        }

        //---------------------------------------

        let mut rhs2 = 3.0;
        let mut lhs2 = lhs.clone();

        super::Ilp::scale_rhs_and_lhs(&mut lhs2, &mut rhs2);

        assert_eq!(rhs2, 3.0);
        for (var, val) in lhs2.iter_terms() {
            if var == &var1 {
                assert_eq!(*val, 1.0);
            } else if var == &var2 {
                assert_eq!(*val, 4.0);
            } else {
                panic!("Unexpected term in LinExpr")
            }
        }

        //---------------------------------------

        let mut rhs3 = 21000.0;
        let mut lhs3 = lhs.clone();

        super::Ilp::scale_rhs_and_lhs(&mut lhs3, &mut rhs3);

        assert!(rhs3.approx_eq(10000.0, margin));
        for (var, val) in lhs3.iter_terms() {
            if var == &var1 {
                assert!(val.approx_eq(1.0 / 2.1, margin));
            } else if var == &var2 {
                assert!(val.approx_eq(4.0 / 2.1, margin));
            } else {
                panic!("Unexpected term in LinExpr")
            }
        }

        // ------------------------------------------
        let mut rhs4 = 0.0;
        let mut lhs4 = lhs.clone();

        super::Ilp::scale_rhs_and_lhs(&mut lhs4, &mut rhs4);

        assert!(rhs4.approx_eq(0.0, margin));
        for (var, val) in lhs4.iter_terms() {
            if var == &var1 {
                assert!(val.approx_eq(1.0, margin));
            } else if var == &var2 {
                assert!(val.approx_eq(4.0, margin));
            } else {
                panic!("Unexpected term in LinExpr")
            }
        }
    }

    #[test]
    #[should_panic(expected = "rhs must not be negative")]
    fn test_scale_rhs_and_lhs_panic() {
        let mut model = grb::Model::new("scaling test").unwrap();
        let var1 = add_binvar!(model, name: "var1").unwrap();
        let var2 = add_binvar!(model, name: "var2").unwrap();
        let mut lhs = LinExpr::new();
        lhs.add_term(1.0, var1);
        lhs.add_term(4.0, var2);

        //---------------------------------------

        let mut rhs = -0.01;
        super::Ilp::scale_rhs_and_lhs(&mut lhs, &mut rhs);
    }

    #[test]
    fn test_ilp_with_pa_eps_dp() {
        let schema = crate::util::build_dummy_schema(EpsDp { eps: 1.0 });
        let request_batch =
            crate::util::build_dummy_requests_with_pa(&schema, 10, EpsDp { eps: 0.4 }, 7);
        let request_history = HashMap::new();
        let blocks = crate::util::generate_blocks(0, 10, EpsDp { eps: 1.0 });
        let block_comp_wrapper = BlockCompWrapper::BlockCompositionPartAttributesVariant(
            block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray),
        );

        let mut ilp = crate::allocation::ilp::Ilp::construct_allocator();
        let (resource_allocation, _) = ilp.round::<OptimalBudget>(
            &request_batch,
            &request_history,
            &blocks,
            &schema,
            &block_comp_wrapper,
            &None,
            &mut Vec::new(),
        );

        let accepted_requests_gt: HashSet<RequestId> =
            HashSet::from_iter([0, 2, 4, 5, 6].into_iter().map(RequestId));
        let rejected_requests_gt: HashSet<RequestId> =
            HashSet::from_iter([1, 3].into_iter().map(RequestId));
        assert_eq!(
            HashSet::from_iter(resource_allocation.accepted.keys().copied()),
            accepted_requests_gt
        );
        assert_eq!(resource_allocation.rejected, rejected_requests_gt);
    }

    #[test]
    fn test_ilp_with_pa_rdp() {
        let mut schema = crate::util::build_dummy_schema(EpsDp { eps: 1.0 });
        schema.accounting_type = Rdp {
            eps_values: A5([0.; 5]),
        };
        let request_batch = crate::util::build_dummy_requests_with_pa(
            &schema,
            10,
            Rdp {
                eps_values: A5([0.4; 5]),
            },
            7,
        );
        let request_history = HashMap::new();
        let blocks = crate::util::generate_blocks(
            0,
            10,
            Rdp {
                eps_values: A5([1.0, 0., 0., 0., 0.]),
            },
        );
        let block_comp_wrapper = BlockCompWrapper::BlockCompositionPartAttributesVariant(
            block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray),
        );

        let mut ilp = crate::allocation::ilp::Ilp::construct_allocator();
        let (resource_allocation, _) = ilp.round::<OptimalBudget>(
            &request_batch,
            &request_history,
            &blocks,
            &schema,
            &block_comp_wrapper,
            &Some(Rdp {
                eps_values: A5([2., 4., 8., 16., 32.]),
            }),
            &mut Vec::new(),
        );

        let accepted_requests_gt: HashSet<RequestId> =
            HashSet::from_iter([0, 2, 4, 5, 6].into_iter().map(RequestId));
        let rejected_requests_gt: HashSet<RequestId> =
            HashSet::from_iter([1, 3].into_iter().map(RequestId));
        assert_eq!(
            HashSet::from_iter(resource_allocation.accepted.keys().copied()),
            accepted_requests_gt
        );
        assert_eq!(resource_allocation.rejected, rejected_requests_gt);
    }

    #[test]
    fn test_ilp_no_pa_eps_dp() {
        let schema = crate::util::build_dummy_schema(EpsDp { eps: 1.0 });
        let mut request_batch =
            crate::util::build_dummy_requests_with_pa(&schema, 10, EpsDp { eps: 0.4 }, 7);
        let request_history = HashMap::new();
        let blocks = crate::util::generate_blocks(0, 15, EpsDp { eps: 1.0 });
        let block_comp_wrapper =
            BlockCompWrapper::BlockCompositionVariant(block_composition::build_block_composition());

        // set profit of these 3 requests higher than others -> should prioritize those
        for rid in [3, 4, 6].into_iter().map(RequestId) {
            request_batch.get_mut(&rid).expect("missing request").profit = 2;
        }

        let mut ilp = crate::allocation::ilp::Ilp::construct_allocator();
        let (resource_allocation, _) = ilp.round::<OptimalBudget>(
            &request_batch,
            &request_history,
            &blocks,
            &schema,
            &block_comp_wrapper,
            &None,
            &mut Vec::new(),
        );

        let accepted_requests_gt: HashSet<RequestId> =
            HashSet::from_iter([3, 4, 6].into_iter().map(RequestId));
        let rejected_requests_gt: HashSet<RequestId> =
            HashSet::from_iter([0, 1, 2, 5].into_iter().map(RequestId));
        assert_eq!(
            HashSet::from_iter(resource_allocation.accepted.keys().copied()),
            accepted_requests_gt
        );
        assert_eq!(resource_allocation.rejected, rejected_requests_gt);
    }

    #[test]
    fn test_ilp_no_pa_rdp() {
        let mut schema = crate::util::build_dummy_schema(EpsDp { eps: 1.0 });
        schema.accounting_type = Rdp {
            eps_values: A5([0.; 5]),
        };
        let mut request_batch =
            crate::util::build_dummy_requests_with_pa(&schema, 10, EpsDp { eps: 0.4 }, 7);
        let request_history = HashMap::new();
        let blocks = crate::util::generate_blocks(
            0,
            20,
            Rdp {
                eps_values: A5([1., 0., 0., 0., 1.]),
            },
        );
        let block_comp_wrapper =
            BlockCompWrapper::BlockCompositionVariant(block_composition::build_block_composition());

        let set1: HashSet<RequestId> = [3, 4, 6].into_iter().map(RequestId).collect();
        let set2: HashSet<RequestId> = [0, 1, 2, 5].into_iter().map(RequestId).collect();

        // set request costs such that 4 requests can be run in total, with the constraint
        // that an even number of requests must be from each set (i.e., not 3 requests from
        // set 1 and one request from set 2)
        for (rid, req) in request_batch.iter_mut() {
            if set1.contains(rid) {
                req.request_cost = Rdp {
                    eps_values: A5([0.5, 1., 1., 1., 1.]),
                };
            } else {
                assert!(set2.contains(rid));
                req.request_cost = Rdp {
                    eps_values: A5([1., 1., 1., 1., 0.5]),
                }
            }
        }

        // set profit of these 4 requests higher than others -> should prioritize those
        for rid in [0, 1, 3, 4].into_iter().map(RequestId) {
            request_batch.get_mut(&rid).expect("missing request").profit = 2;
        }

        let mut ilp = crate::allocation::ilp::Ilp::construct_allocator();
        let (resource_allocation, _) = ilp.round::<OptimalBudget>(
            &request_batch,
            &request_history,
            &blocks,
            &schema,
            &block_comp_wrapper,
            &Some(Rdp {
                eps_values: A5([2., 4., 8., 16., 32.]),
            }),
            &mut Vec::new(),
        );

        let accepted_requests_gt: HashSet<RequestId> =
            HashSet::from_iter([0, 1, 3, 4].into_iter().map(RequestId));
        let rejected_requests_gt: HashSet<RequestId> =
            HashSet::from_iter([2, 5, 6].into_iter().map(RequestId));
        assert_eq!(
            HashSet::from_iter(resource_allocation.accepted.keys().copied()),
            accepted_requests_gt
        );
        assert_eq!(resource_allocation.rejected, rejected_requests_gt);
    }
}
