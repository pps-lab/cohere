use crate::allocation::{AllocationStatus, BlockCompWrapper, ResourceAllocation};
use crate::block::{Block, BlockId};
use crate::composition::{
    BlockOrderStrategy, CompositionConstraint, ProblemFormulation, StatusResult,
};
use crate::dprivacy::budget::SegmentBudget;
use crate::logging::{GreedyStats, RuntimeKind, RuntimeMeasurement};
use crate::request::{Request, RequestId};
use crate::schema::Schema;
use itertools::Itertools;
use std::collections::{HashMap, HashSet};

/// Implements a greedy algorithm, that strictly prioritizes requests with lower request id (i.e.,
/// requests that came first) and, if given the choice of which blocks to allocate, prioritizes
/// allocating blocks with lower block id
pub struct Greedy {}

impl Greedy {
    pub fn construct_allocator() -> Self {
        Greedy {}
    }

    pub fn round<M: SegmentBudget>(
        &mut self,
        candidate_requests: &HashMap<RequestId, Request>,
        request_history: &HashMap<RequestId, Request>,
        available_blocks: &HashMap<BlockId, Block>,
        schema: &Schema,
        block_comp_wrapper: &BlockCompWrapper,
        runtime_measurements: &mut Vec<RuntimeMeasurement>,
    ) -> (super::ResourceAllocation, AllocationStatus) {
        let mut pf: ProblemFormulation<M> = block_comp_wrapper.build_problem_formulation::<M>(
            available_blocks,
            candidate_requests,
            request_history,
            schema,
            runtime_measurements,
        );

        let num_contested_segments_initially = pf.contested_constraints().count();

        let mut resource_allocation = ResourceAllocation {
            accepted: HashMap::new(),
            rejected: HashSet::new(),
        };

        let mut alloc_meas = RuntimeMeasurement::start(RuntimeKind::RunAllocationAlgorithm);

        let sorted_candidates = candidate_requests.keys().copied().sorted();

        // Order requests by id, and greedily try to allocate
        for rid in sorted_candidates {
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
            assert_eq!(
                selected_blocks.len(),
                candidate_requests[&rid].n_users,
                "Not enough blocks to accept"
            );
            pf.allocate_request(rid, &selected_blocks, candidate_requests)
                .expect("Allocating request failed");
            resource_allocation.accepted.insert(rid, selected_blocks);
        }

        runtime_measurements.push(alloc_meas.stop());

        //println!("{:?}", resource_allocation);
        (
            resource_allocation,
            AllocationStatus::GreedyStatus(GreedyStats {
                num_contested_segments_initially,
            }),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::allocation::greedy::Greedy;
    use crate::allocation::BlockCompWrapper;
    use crate::composition::{block_composition, block_composition_pa};
    use crate::config::SegmentationAlgo;
    use crate::util::{build_dummy_requests_with_pa, build_dummy_schema, generate_blocks};
    use crate::AccountingType::EpsDp;
    use crate::BlockId::User;
    use crate::{BlockId, OptimalBudget, RequestId};
    use std::collections::{HashMap, HashSet};

    #[test]
    fn test_greedy_no_pa() {
        let schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let request_batch = build_dummy_requests_with_pa(&schema, 10, EpsDp { eps: 0.4 }, 6);
        let request_history = HashMap::new();
        let blocks = generate_blocks(0, 19, EpsDp { eps: 1.0 });
        let block_comp_wrapper =
            BlockCompWrapper::BlockCompositionVariant(block_composition::build_block_composition());

        let mut greedy = Greedy::construct_allocator();
        let resource_allocation = greedy
            .round::<OptimalBudget>(
                &request_batch,
                &request_history,
                &blocks,
                &schema,
                &block_comp_wrapper,
                &mut Vec::new(),
            )
            .0;

        // since we have 19 blocks and each request needs 10 blocks, we could allocate 3 requests,
        // but greedy greedily takes blocks with lower block id, therefore we can only run 2 requests

        let accepted_gt: HashMap<RequestId, HashSet<BlockId>> = {
            let first_ten_blocks = HashSet::from_iter((0..10).map(User));
            HashMap::from([
                (RequestId(0), first_ten_blocks.clone()),
                (RequestId(1), first_ten_blocks),
            ])
        };
        let rejected_gt: HashSet<RequestId> = HashSet::from_iter((2..6).map(RequestId));
        assert_eq!(resource_allocation.accepted, accepted_gt);
        assert_eq!(resource_allocation.rejected, rejected_gt);
    }

    #[test]
    fn test_greedy_with_pa() {
        let schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let request_batch = build_dummy_requests_with_pa(&schema, 10, EpsDp { eps: 0.4 }, 6);
        let request_history = HashMap::new();
        let blocks = generate_blocks(0, 15, EpsDp { eps: 1.0 });
        let block_comp_wrapper = BlockCompWrapper::BlockCompositionPartAttributesVariant(
            block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray),
        );

        let mut greedy = Greedy::construct_allocator();
        let resource_allocation = greedy
            .round::<OptimalBudget>(
                &request_batch,
                &request_history,
                &blocks,
                &schema,
                &block_comp_wrapper,
                &mut Vec::new(),
            )
            .0;

        // after the first three requests, the last 5 blocks are now acceptable for 3 and 4, while
        // the other 10 blocks are still contested -> 3 will take the acceptable and the first 5
        // contested blocks, while 3 will then take the remaining blocks (which are now all acceptable,
        // since 5 was rejected after 0 and 1 were allocated, so 4 is the only request left,
        // and there are 10 blocks with enough budget)

        let accepted_gt: HashMap<RequestId, HashSet<BlockId>> = {
            let first_ten_blocks = HashSet::from_iter((0..10).map(User));
            let first_and_last_five: HashSet<BlockId> =
                HashSet::<BlockId>::from_iter((0..5).map(User))
                    .union(&HashSet::from_iter((10..15).map(User)))
                    .copied()
                    .collect();
            let last_ten_blocks = HashSet::from_iter((5..15).map(User));
            HashMap::from([
                (RequestId(0), first_ten_blocks.clone()),
                (RequestId(1), first_ten_blocks.clone()),
                (RequestId(2), first_ten_blocks),
                (RequestId(3), first_and_last_five),
                (RequestId(4), last_ten_blocks),
            ])
        };
        let rejected_gt: HashSet<RequestId> = HashSet::from_iter([5].map(RequestId));
        println!(
            "Accepted Requests: {:?}",
            resource_allocation.accepted.keys()
        );
        assert_eq!(resource_allocation.accepted, accepted_gt);
        assert_eq!(resource_allocation.rejected, rejected_gt);
    }
}
