use crate::composition::BlockConstraints;
use crate::composition::BlockSegment;
use crate::dprivacy::budget::SegmentBudget;
use crate::dprivacy::{Accounting, AccountingType};
use crate::request::{Request, RequestId};
use crate::schema::{DataValueLookup, Schema};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

use super::Segmentation;

pub struct HashmapSegmentor<'r, 's> {
    /// maps request id to request
    new_requests: HashMap<RequestId, &'r Request>,
    /// maps virtual block ids to (flat_id, segment_id (which is a vector of request ids))
    virtual_blocks: HashMap<Vec<usize>, (usize, Vec<RequestId>)>,
    /// The schema for the requests
    schema: &'s Schema,
}

impl<'r, 's> Segmentation<'r, 's> for HashmapSegmentor<'r, 's> {
    fn new(request_batch: Vec<&'r Request>, schema: &'s Schema) -> Self {
        // maps request ids of new requests to the request
        let new_requests: HashMap<RequestId, &'r Request> = request_batch
            .iter()
            .map(|request| (request.request_id, *request))
            .collect();

        // maps virtual block id to applicable segment (n)
        let mut virtual_block_id_to_segment_id: HashMap<Vec<usize>, (usize, Vec<RequestId>)> =
            schema
                .virtual_block_id_iterator()
                .enumerate()
                .map(|(flat_id, full_id)| (full_id, (flat_id, Vec::new())))
                .collect();

        for new_request in request_batch {
            for virtual_block_id in new_request.dnf().repeating_iter(schema) {
                let virtual_block = &mut virtual_block_id_to_segment_id
                    .get_mut(&virtual_block_id)
                    .expect("New request specifies invalid virtual block")
                    .1;
                if virtual_block.is_empty()
                    || virtual_block[virtual_block.len() - 1] != new_request.request_id
                {
                    virtual_block.push(new_request.request_id)
                }
            }
        }

        HashmapSegmentor {
            new_requests,
            virtual_blocks: virtual_block_id_to_segment_id,
            schema,
        }
    }

    fn compute_block_constraints<M: SegmentBudget + Debug>(
        &self,
        request_history: Vec<&Request>,
        initial_block_budget: &AccountingType,
    ) -> BlockConstraints<M> {
        #[derive(Debug)]
        struct InternalSegment<M: SegmentBudget> {
            budget: M,
        }

        struct VirtualBlockBudget {
            budget: AccountingType,
            last_old_request: Option<RequestId>,
        }

        // initialize the initial_budget for each virtual block
        let mut virtual_blocks_budget: HashMap<usize, VirtualBlockBudget> = self
            .virtual_blocks
            .iter()
            .map(|(_full_id, (flat_id, _segment_id))| {
                (
                    *flat_id,
                    VirtualBlockBudget {
                        budget: initial_block_budget.clone(),
                        last_old_request: None,
                    },
                )
            })
            .collect();

        // subtract all costs from all virtual blocks
        for request in request_history {
            for virtual_block_id in request.dnf().repeating_iter(self.schema) {
                let vb_budget = virtual_blocks_budget
                    .get_mut(
                        self.virtual_blocks
                            .get(&virtual_block_id)
                            .map(|(flat_id, _segment_id)| flat_id)
                            .expect("Did not find virtual block flat id"),
                    )
                    .expect("Did not find virtual block by id");
                match vb_budget.last_old_request {
                    Some(request_id) => {
                        if request_id != request.request_id {
                            vb_budget.last_old_request = Some(request.request_id);
                            vb_budget.budget -= &request.request_cost;
                        }
                    }
                    None => {
                        vb_budget.last_old_request = Some(request.request_id);
                        vb_budget.budget -= &request.request_cost;
                    }
                }
            }
        }

        // construct a segment_budget for each segment
        let mut segment_map: HashMap<&Vec<RequestId>, InternalSegment<M>> = self
            .virtual_blocks
            .iter()
            .map(|(_, (_, segment_id))| (segment_id, InternalSegment { budget: M::new() }))
            .collect();

        // merge all budgets in a segment
        for (vb_flat_id, segment_id) in self.virtual_blocks.values() {
            let virtual_block_budget = virtual_blocks_budget
                .get(vb_flat_id)
                .expect("Did not find virtual block budget");
            segment_map
                .get_mut(segment_id)
                .expect("Did not find segment by id")
                .budget
                .add_budget_constraint(&virtual_block_budget.budget);
        }

        let rejected: HashSet<RequestId> = self
            .new_requests
            .iter()
            .filter(
                |(_, request)| // check if budget is not sufficient in any segment
                request
                    .dnf()
                    .repeating_iter(self.schema)
                    .map(
                        |virtual_block_id|
                            segment_map
                                .get(& self
                                    .virtual_blocks
                                    .get(&virtual_block_id)
                                    .unwrap_or_else(|| panic!("No virtual block matches virtual block id {:?}", &virtual_block_id))
                                    .1)
                                .expect("Did not find segment")
                    )
                    .any(|segment| !segment
                              .budget
                              .is_budget_sufficient(&request.request_cost)),
            )
            .map(|(request_id, _request)| *request_id)
            .collect();

        #[allow(clippy::type_complexity)]
        let (uncontested_map, contested_map): (
            HashMap<&Vec<RequestId>, InternalSegment<M>>,
            HashMap<&Vec<RequestId>, InternalSegment<M>>,
        ) = segment_map.into_iter().partition(|(segment_id, segment)| {
            segment_id.is_empty()
                || segment.budget.is_budget_sufficient(
                    &segment_id
                        .iter()
                        .filter(|request_id| !rejected.contains(request_id)) // do not take into account rejected requests
                        .map(|request_id| {
                            &self
                                .new_requests
                                .get(request_id)
                                .expect("did not find request by id")
                                .request_cost
                        })
                        .fold(
                            AccountingType::zero_clone(
                                &self
                                    .new_requests
                                    .values()
                                    .take(1)
                                    .collect::<Vec<&&Request>>()[0]
                                    .request_cost,
                            ),
                            |acc, cost| acc + cost,
                        ),
                )
        });

        let acceptable: HashSet<RequestId> = self
            .new_requests
            .iter()
            .filter(|(request_id, request)| {
                !rejected.contains(request_id) // no need to check rejected requests
                    && (request.dnf().repeating_iter(self.schema).all(|virtual_block_id| { // all segments need to be uncontested
                    uncontested_map.contains_key(
                        &self.virtual_blocks
                        .get(&virtual_block_id)
                        .expect("Virtual Block by id not found")
                        .1)
                    })
                )
            })
            .map(|(request_id, _request)| *request_id)
            .collect();

        let contested: HashSet<RequestId> = self // everything else must be undecided
            .new_requests
            .iter()
            .filter(|(request_id, _request)| {
                !rejected.contains(request_id) && !acceptable.contains(request_id)
            })
            .map(|(request_id, _request)| *request_id)
            .collect();

        // For output, merge segments in contested by removing accepted and rejected requests
        let mut contested_segments: HashMap<Vec<RequestId>, InternalSegment<M>> = HashMap::new();
        for (segment_id, internal_segment) in contested_map {
            let stripped_segment_id: Vec<RequestId> = segment_id // only want requests which are undecided
                .iter()
                .filter(|request_id| contested.contains(request_id))
                .cloned()
                .collect();

            #[allow(clippy::map_entry)]
            if !contested_segments.contains_key(&stripped_segment_id) {
                contested_segments.insert(stripped_segment_id, internal_segment);
            } else {
                contested_segments
                    .get_mut(&stripped_segment_id)
                    .unwrap()
                    .budget
                    .merge_assign(&internal_segment.budget);
            }
        }
        // and remove the segment with no more requests applying (if there is one)
        contested_segments.remove(&Vec::new());

        // println!("Contested Segments: {:?}", contested_segments);

        let contested_segments: Vec<BlockSegment<M>> = contested_segments
            .into_iter()
            .enumerate()
            .map(|(flat_id, (full_id, internal_segment))| BlockSegment {
                id: flat_id,
                request_ids: Some(full_id),
                remaining_budget: internal_segment.budget,
            })
            .collect();

        BlockConstraints {
            acceptable,
            rejected,
            contested,
            contested_segments,
        }
    }
}
