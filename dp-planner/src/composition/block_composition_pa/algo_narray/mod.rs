use crate::composition::{BlockConstraints, BlockSegment};
use crate::dprivacy::budget::SegmentBudget;
use crate::request::{Conjunction, Request, RequestId};
use crate::schema::Schema;

use super::Segmentation;

use float_cmp::{ApproxEq, F64Margin};

use itertools::Itertools;

use fasthash::{sea::Hash64, FastHash};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use std::collections::{HashMap, HashSet};

mod narray;

use crate::dprivacy::{Accounting, AccountingType};
use narray::{Dimension, Index, NArray};

// TODO [nku] [later] write tests
pub struct NArraySegmentation<'r, 's> {
    dimension: Dimension,
    requested_budget: NArray<VirtualBlockRequested>,
    request_batch: Vec<&'r Request>,
    schema: &'s Schema,
}

#[derive(Clone)]
struct VirtualBlockBudget {
    budget: AccountingType,
    prev_request_id: Option<RequestId>,
}

#[derive(Clone)]
struct VirtualBlockRequested {
    request_hash: u64,
    request_count: u32,
    cost: AccountingType,
    prev_request_id: Option<RequestId>,
}

impl<'r, 's> Segmentation<'r, 's> for NArraySegmentation<'r, 's> {
    fn new(request_batch: Vec<&'r Request>, schema: &'s Schema) -> Self {
        let dimension: Vec<usize> = schema.attributes.iter().map(|attr| attr.len()).collect();
        let dimension = Dimension::new(&dimension);

        // calculate requested budget
        let block = VirtualBlockRequested::new(schema);
        let mut requested_budget = narray::build(&dimension, block);
        calculate_requested_budget(&request_batch, &mut requested_budget, schema);

        NArraySegmentation {
            dimension,
            requested_budget,
            request_batch,
            schema,
        }
    }

    fn compute_block_constraints<M: SegmentBudget>(
        &self,
        request_history: Vec<&Request>,
        initial_block_budget: &AccountingType,
    ) -> BlockConstraints<M> {
        // calculate remaining budget
        let budget = VirtualBlockBudget {
            budget: initial_block_budget.clone(),
            prev_request_id: None,
        };

        // TODO [nku] [later]: as alternative could also remove cost / budget from virtualblock and only focus on hash id.
        // afterwards, compute unique hash id and reverse all, then compute for each id cost.
        // also for request history could do this segmentation with hash id.
        // (would potentially save a lot of duplicated computation on adding up accounting_types)

        let mut remaining_budget = narray::build(&self.dimension, budget);
        calculate_remaining_budget(&request_history, &mut remaining_budget, self.schema);

        // calculate min remaining budget per segment
        // TODO [nku][later] old hash function with xor of request ids combined in a tuple  -> or xor of requestids and hash value
        let mut budget_by_segment: HashMap<u64, SegmentWrapper<M>> = HashMap::new();
        calculate_remaining_budget_per_segment(
            &mut budget_by_segment,
            &remaining_budget,
            &self.requested_budget,
        );

        budget_by_segment.retain(|_, v| v.is_contested());

        // reconstruct request ids from segment id with first_index
        reconstruct_request_ids(&mut budget_by_segment, &self.request_batch);

        // find rejected request ids (r.cost > budget)
        //  + remove cost of rejected request ids from cost sums
        //  + retain only congested (after subtraction of rejected requests)
        let rejected_request_ids =
            reject_infeasible_requests(&mut budget_by_segment, &self.request_batch);

        // find accepted (all \ rejected \ congested)
        build_block_constraints(
            &rejected_request_ids,
            budget_by_segment,
            &self.request_batch,
        )
    }
}

fn calculate_remaining_budget(
    request_history: &[&Request],
    remaining_budget: &mut NArray<VirtualBlockBudget>,
    schema: &Schema,
) {
    let update_budget_closure = |virtual_block_budget: &mut VirtualBlockBudget,
                                 request: &Request| {
        virtual_block_budget.update(request.request_id, &request.request_cost);
    };

    for request in request_history.iter() {
        let request = *request;
        for vec in request.dnf().repeating_iter(schema) {
            let idx = Index::new(&vec);
            remaining_budget.update(&idx, request, update_budget_closure);
        }
    }
}

fn calculate_requested_budget(
    request_batch: &[&Request],
    requested_budget: &mut NArray<VirtualBlockRequested>,
    schema: &Schema,
) {
    let update_virtual_block = |block: &mut VirtualBlockRequested, request: &Request| {
        block.update(request.request_id, &request.request_cost);
    };

    for request in request_batch.iter() {
        let request = *request;
        for vec in request.dnf().repeating_iter(schema) {
            let idx = Index::new(&vec);
            requested_budget.update(&idx, request, update_virtual_block);
        }
    }
}

// could be alternative approach for "calculate_remaining_budget_per_segment"
//fn compute_segments(requests: &[Request], schema: &Schema) -> Vec<Segment> {
//    let dimension: Vec<usize> = schema.attributes.iter().map(|attr| attr.len()).collect();
//    let dimension = Dimension::new(&dimension);
//
//    // calculate requested budget
//    let block = VirtualBlockRequested::new(schema);
//    let mut requested_budget = narray::build(&dimension, block);
//    calculate_requested_budget(requests, &mut requested_budget, schema);
//
//    // reduce to segments
//    let segments = requested_budget
//        .iter()
//        .enumerate()
//        .into_grouping_map_by(|(_i, virtual_block)| virtual_block.request_hash)
//        .aggregate(|acc: Option<Segment>, request_hash, (i, virtual_block)| {
//            match acc {
//                Some(Segment {
//                    id: _,
//                    request_ids: _,
//                    accounting: SegmentAccounting::Cost(cost),
//                }) => {
//                    cost += &virtual_block.cost;
//                    // TODO [nku] [later] verify that the change was done on return value
//                    acc
//                }
//                None => Some(Segment {
//                    id: *request_hash,
//                    request_ids: None,
//                    accounting: SegmentAccounting::Cost(virtual_block.cost),
//                }),
//                _ => panic!("illegal x"),
//            }
//        });
//
//    // TODO [nku] [later] could also think about returning HashMap of Segments
//    segments.into_iter().map(|(k, v)| v).collect()
//}

fn calculate_remaining_budget_per_segment<M: SegmentBudget>(
    budget_by_segment: &mut HashMap<u64, SegmentWrapper<M>>,
    remaining_budget: &NArray<VirtualBlockBudget>,
    requested_budget: &NArray<VirtualBlockRequested>,
) {
    for (i, virtual_block_requested_cost) in requested_budget.iter().enumerate() {
        let virtual_block_budget = remaining_budget.get_by_flat(i);

        budget_by_segment
            .entry(virtual_block_requested_cost.request_hash)
            .or_insert_with(|| {
                SegmentWrapper::new(
                    virtual_block_requested_cost.request_hash,
                    narray::from_idx(i, &requested_budget.dim),
                )
            })
            .update_segment(virtual_block_budget, virtual_block_requested_cost);
    }
}

trait Contains {
    fn contains(&self, idx: &Index) -> bool;
}

impl Contains for Conjunction {
    fn contains(&self, idx: &Index) -> bool {
        let idx_vec = idx.to_vec();

        if idx_vec.len() != self.predicates().len() {
            panic!(
                "incompatible index for conjunction idx_vec={:?}   pred length={:?}",
                idx_vec.len(),
                self.predicates().len()
            );
        }

        self.predicates()
            .iter()
            .zip(idx_vec.iter())
            .all(|(pred, i)| pred.contains(i))
    }
}

fn reconstruct_request_ids<M: SegmentBudget>(
    segments: &mut HashMap<u64, SegmentWrapper<M>>,
    requests: &[&Request],
) {
    for (_id, segment_wrapper) in segments.iter_mut() {
        let first_idx = &segment_wrapper.first_idx;

        segment_wrapper.segment.request_ids = Some(
            requests
                .iter()
                .filter(|r| {
                    // keep only requests which have one conjunction that contains the first_idx
                    r.dnf()
                        .conjunctions
                        .iter()
                        .any(|conj| conj.contains(first_idx))
                })
                .map(|r| r.request_id)
                .collect(),
        );
    }
}

fn reject_infeasible_requests<M: SegmentBudget>(
    segments: &mut HashMap<u64, SegmentWrapper<M>>,
    requests: &[&Request],
) -> HashSet<RequestId> {
    let request_cost_map: HashMap<RequestId, &AccountingType> = requests
        .iter()
        .map(|r| (r.request_id, &r.request_cost))
        .collect();

    let mut rejected_request_ids: HashSet<RequestId> = HashSet::new();

    for (_id, segment_wrapper) in segments.iter() {
        // loop over request_ids -> see if r.cost > segment.remaining_budget -> if yes => remove from request_ids, subtract cost from sum_requested_cost, put into rejected ids list
        let request_ids = segment_wrapper
            .segment
            .request_ids
            .as_ref()
            .expect("must be there: request ids");

        let iter = request_ids
            .iter()
            .filter(|request_id| {
                let request_cost = *request_cost_map
                    .get(*request_id)
                    .expect("must be there: request_cost_map");

                let is_budget_sufficient = segment_wrapper
                    .segment
                    .remaining_budget
                    .is_budget_sufficient(request_cost);

                !is_budget_sufficient
            })
            .copied();

        rejected_request_ids.extend(iter);
    }

    // after identifying rejected requests -> need to update all segments
    if !rejected_request_ids.is_empty() {
        segments.retain(|_id, segment_wrapper| {
            // remove rejected requests from segment and adapt request cost sum

            match &mut segment_wrapper.segment.request_ids {
                Some(request_ids) => request_ids.retain(|request_id| {
                    let contains = rejected_request_ids.contains(request_id);

                    if contains {
                        // segment contains rejected request => subtract rejected request cost from sum

                        if let Some(sum_requested_cost) = &mut segment_wrapper.sum_requested_cost {
                            let cost = *request_cost_map.get(request_id).unwrap();
                            *sum_requested_cost -= cost;
                        } else {
                            panic!("sum request cost must be set");
                        }
                    }

                    !contains
                }),
                None => panic!("request ids must be defined"),
            }

            segment_wrapper.is_contested()
        });
    }

    rejected_request_ids
}

fn build_block_constraints<'a, M: SegmentBudget>(
    rejected_request_ids: &HashSet<RequestId>,
    contested_segments: HashMap<u64, SegmentWrapper<M>>,
    request_batch: &[&'a Request],
) -> BlockConstraints<M> {
    let request_map: HashMap<RequestId, &Request> =
        request_batch.iter().map(|r| (r.request_id, *r)).collect();

    /*
    println!("request_map={:?}", request_map.keys());
    println!("contested_requests={:?}", contested_segments.keys());
    */

    let contested: HashSet<RequestId> = contested_segments
        .values()
        .flat_map(|segment_wrapper| {
            segment_wrapper
                .segment
                .request_ids
                .as_ref()
                .expect("request ids must be set")
        })
        .copied()
        .collect();

    let acceptable = request_batch
        .iter()
        .filter(|request| {
            (!rejected_request_ids.contains(&request.request_id))
                && (!contested.contains(&request.request_id))
        })
        .map(|r| r.request_id)
        .collect();

    let rejected = rejected_request_ids
        .iter()
        .map(|r_id| *request_map.get(r_id).expect("unknown request"))
        .map(|r| r.request_id)
        .collect();

    // TODO [nku] THIS can be replaced with own implmentation of problemformulation (or new BlockCOnstraints)
    let contested_segments = contested_segments
        .into_iter()
        .map(|(_segment_id, segment_wrapper)| segment_wrapper.segment)
        .into_grouping_map_by(|segment| {
            let mut hasher = DefaultHasher::new();
            Hash::hash_slice(segment.request_ids.as_ref().unwrap(), &mut hasher);
            hasher.finish()
        })
        .fold_first(|mut acc_segment, _key, val_segment| {
            // merge both of them
            acc_segment
                .remaining_budget
                .merge_assign(&val_segment.remaining_budget);

            acc_segment
        })
        .into_values()
        .filter(|segment| !segment.request_ids.as_ref().unwrap().is_empty())
        .collect_vec();

    BlockConstraints {
        acceptable,
        rejected,
        contested,
        contested_segments,
    }
}

// TODO [nku] [later] test with better hash function
/*
const K: usize = 0x517cc1b727220a95;
fn fx_hasher(start: usize, new: usize) -> usize {
    let tmp = start.rotate_left(5) ^ new;
    tmp.wrapping_mul(K)
}
 */

impl VirtualBlockBudget {
    fn update(&mut self, request_id: RequestId, privacy_cost: &AccountingType) {
        match self.prev_request_id {
            Some(prev_request_id) if prev_request_id == request_id => (), // do nothing
            // ignore (because we use repeating iter -> can happen that we select same block twice)
            _ => {
                // subtract the privacy cost from the budget and update request id to ensure that we only do it once
                self.prev_request_id = Some(request_id);
                self.budget -= privacy_cost
            }
        }
    }
}

// TODO [nku] [later]: The hash function is not the bottleneck -> with a better hash function we can ignore request_count logic
impl VirtualBlockRequested {
    fn new(schema: &Schema) -> VirtualBlockRequested {
        VirtualBlockRequested {
            request_hash: 0,
            request_count: 0,
            cost: AccountingType::zero_clone(&schema.accounting_type),
            prev_request_id: None,
        }
    }

    fn update(&mut self, request_id: RequestId, privacy_cost: &AccountingType) {
        match self.prev_request_id {
            Some(prev_request_id) if prev_request_id == request_id => (), // do nothing if prev request id is the same (deal with repeating iter)
            _ => {
                // TODO [nku] [later] could bring back request_hash with fx_hasher(...) -> Problem can observe hash collisions
                //self.request_hash = fx_hasher(self.request_hash, request_id);

                self.request_hash = Hash64::hash_with_seed(
                    request_id.0.to_ne_bytes(),
                    (self.request_hash, 0, 0, 0),
                );

                self.request_count += 1;
                self.cost += privacy_cost;
                self.prev_request_id = Some(request_id);
            }
        }
    }
}

pub struct SegmentWrapper<M: SegmentBudget> {
    segment: BlockSegment<M>,
    first_idx: Index,
    sum_requested_cost: Option<AccountingType>,
    request_count: Option<u32>,
}

impl<M: SegmentBudget> SegmentWrapper<M> {
    fn new(segment_id: u64, first_idx: Index) -> SegmentWrapper<M> {
        SegmentWrapper {
            segment: BlockSegment::new(segment_id.try_into().unwrap()),
            first_idx,
            sum_requested_cost: None,
            request_count: None,
        }
    }

    // at the moment also for rdp we return one budget per segment (the element-wise minimum)
    fn update_segment(
        &mut self,
        virtual_block_budget: &VirtualBlockBudget,
        virtual_block_requested_cost: &VirtualBlockRequested,
    ) {
        self.segment
            .remaining_budget
            .add_budget_constraint(&virtual_block_budget.budget);

        // 5914157414052549729

        match self.request_count {
            // set the cost if not set previously
            None => self.request_count = Some(virtual_block_requested_cost.request_count),

            // assert that n_requests is the same for all blocks in segment
            Some(request_count) => {
                assert_eq!(request_count, virtual_block_requested_cost.request_count)
            } // TODO [nku] [later] want some assert which is not present in --release
        }

        match &self.sum_requested_cost {
            // set the cost if not set previously
            None => self.sum_requested_cost = Some(virtual_block_requested_cost.cost.clone()),

            // assert that cost is the same for all blocks in segment
            Some(sum_requested_cost) => assert!(
                sum_requested_cost
                    .approx_eq(&virtual_block_requested_cost.cost, F64Margin::default()),
                "segment_id={}     my_sum={:?}  other_sum={:?}",
                self.segment.id,
                sum_requested_cost,
                virtual_block_requested_cost.cost
            ),
            // TODO [nku] [later] want some assert which is not present in --release
        }
    }

    fn is_contested(&self) -> bool {
        let cost = self
            .sum_requested_cost
            .as_ref()
            .expect("requested cost must be set");

        !self.segment.remaining_budget.is_budget_sufficient(cost)
    }
}
