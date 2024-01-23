//! This module provides a graph implementation to track the current state of the system, including
//! blocks, virtual blocks and requests with all relevant properties.
//!
//! This graph implementation exposes methods to enable efficient updates when a request is
//! allocated or rejected, and through that allows a greatly simplified implementation of
//! the allocation algorithms discussed in the paper.
//!
//! Such a graph datastructure can be initialized using
//! [build_problem_formulation](CompositionConstraint::build_problem_formulation)
//! (which abstracts over whether we have partitioning attributes or not)

use concurrent_queue::ConcurrentQueue;
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::collections::{HashMap, HashSet};
use std::fmt;

use itertools::{Either, Itertools};
use petgraph::dot::Dot;
use petgraph::graph::NodeIndex;
use petgraph::stable_graph::StableGraph;
use petgraph::Undirected;
use rayon::prelude::*;

use crate::logging::RuntimeMeasurement;
use crate::{
    block::{Block, BlockId},
    dprivacy::budget::SegmentBudget,
    request::{Request, RequestId},
    schema::Schema,
};

pub mod block_composition;
pub mod block_composition_pa;

/// New problem formulation based on Graph implementation. A variety of methods are implemented on
/// this struct to offer convenient and efficient operations on the graph data structure.
pub struct ProblemFormulation<M: SegmentBudget> {
    graph: StableGraph<Node<M>, Edge, Undirected>,

    requests: HashMap<RequestId, HashSet<NodeIndex>>, //<RequestId, RequestBlockNode>
}

/// Return value of [ProblemFormulation::request_status()].
///
/// It contains the current status of a request, taking into account all prior updates to the
/// [ProblemFormulation].
#[derive(PartialEq, Eq, Debug)]
pub enum StatusResult {
    /// This status means that a request can be accepted no matter what allocation decision is made
    /// for any other request. However, not all requests that fit this description necessarily
    /// return this status, as determining that would likely be a hard problem in some cases.
    Acceptable {
        /// Which blocks are acceptable, i.e., there is enough budget for this request on that block
        /// regardless of other allocation decisions.
        acceptable: Vec<BlockId>,
        /// Which blocks are contested, i.e., this block may be allocated to the specified request,
        /// depending on the allocation decisions of other requests.
        contested: Vec<BlockId>,
    },
    /// This status is returned for any request that can be allocated, but where it has not been
    /// determined that the request can be allocated regardless of allocation decisions for
    /// other requests.
    Contested {
        /// Which blocks are acceptable, i.e., there is enough budget for this request on that block
        /// regardless of other allocation decisions.
        acceptable: Vec<BlockId>,
        /// Which blocks are contested, i.e., this block may be allocated to the specified request,
        /// depending on the allocation decisions of other requests.
        contested: Vec<BlockId>,
    },
    /// Finally, this status is returned for requests that cannot be allocated anymore, taking
    /// all prior updates into account.
    Rejected,
}

/// This struct contains an acceptable request, as well as the blocks that are acceptable
/// (and contested).
///
/// This returns some information regarding the next acceptable request as determined by the
/// linked method
pub struct AcceptableRequest {
    /// The id of the next acceptable request.
    pub(crate) request_id: RequestId,
    /// Which blocks are acceptable, i.e., there is enough budget for this request on that block
    /// regardless of other allocation decisions.
    pub(crate) acceptable: Vec<BlockId>,

    /// Which blocks are contested, i.e., this block may be allocated to the specified request,
    /// depending on the allocation decisions of other requests.
    #[allow(dead_code)]
    pub(crate) contested: Vec<BlockId>,
}

/// Defines a strategy for how blocks are ordered when calling
/// [request_status](ProblemFormulation::request_status). This can then be used to allocate
/// blocks easily according to the chosen order.
pub enum BlockOrderStrategy<'a> {
    /// Blocks are ordered by block id
    Id,
    /// Blocks are ordered by block creation date
    BlockCreation {
        /// Should contain an entry for each block passed when constructing [ProblemFormulation]
        block_lookup: &'a HashMap<BlockId, Block>,
    },
    /// Blocks are ordered by how much budget they have left
    #[allow(dead_code)]
    RemainingBudget {
        /// Should contain an entry for each block passed to [ProblemFormulation]
        block_lookup: &'a HashMap<BlockId, Block>,
    },
}

impl<'a> BlockOrderStrategy<'a> {
    /// This methods returns the order two blocks are in according to the given
    /// [BlockOrderStrategy]. Used for sorting blocks returned by
    /// [request_status](ProblemFormulation::request_status)
    fn compare(&self, a: &BlockId, b: &BlockId) -> Ordering {
        match self {
            BlockOrderStrategy::Id => a.cmp(b),
            BlockOrderStrategy::BlockCreation { block_lookup } => {
                let a_block = block_lookup.get(a).expect("missing block in lookup");
                let b_block = block_lookup.get(b).expect("missing block in lookup");
                let res1 = a_block.created.cmp(&b_block.created);
                // to make things deterministic, order by blockid if created timestamp is not
                // sufficient. note that for the same block id, this still returns equal
                if let Ordering::Equal = res1 {
                    a.cmp(b)
                } else {
                    res1
                }
            }
            BlockOrderStrategy::RemainingBudget {
                block_lookup: _block_lookup,
            } => {
                // TODO [later] implement remaining budget sorting -> could do min_merge of all segments
                todo!("unsupported strategy at the moment")
            }
        }
    }
}

/// Unifying interface which allows to construct a [ProblemFormulation] with or without
/// partitioning attributes
pub trait CompositionConstraint {
    fn build_problem_formulation<M: SegmentBudget>(
        &self,
        blocks: &HashMap<BlockId, Block>,
        candidate_requests: &HashMap<RequestId, Request>,
        history_requests: &HashMap<RequestId, Request>,
        schema: &Schema,
        runtime_measurements: &mut Vec<RuntimeMeasurement>,
    ) -> ProblemFormulation<M>;
}

/// One node in the graph datastructure which is part of [ProblemFormulation]
///
/// The graph is bipartite, with the (or a) bipartition given by the two classes of nodes.
#[derive(Debug)]
enum Node<M: SegmentBudget> {
    /// A node that corresponds to a request wanting to access a certain block.
    RequestBlock {
        request_id: RequestId,
        block_id: BlockId,
    },
    /// A node that corresponds to a segment of a block (without partitioning attributes, there
    /// is exactly one segment per node, else there may be multiple)
    BlockSegmentConstraint { budget: M },
}

struct Edge {}

impl<M: SegmentBudget> fmt::Display for Node<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::RequestBlock {
                request_id,
                block_id,
            } => write!(f, "request: {}\nblock: {}", request_id, block_id),
            Node::BlockSegmentConstraint { budget } => write!(f, "budget:\n{}", budget),
        }
    }
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "constraint")
    }
}

#[derive(Debug, Clone)]
pub enum AllocationError {
    IllegalRequestId(RequestId),
    IllegalBlockAssignment(String),
}

impl<M: SegmentBudget> ProblemFormulation<M> {
    fn insert_request_block(&mut self, request_id: RequestId, block_id: BlockId) -> NodeIndex {
        let idx = self.graph.add_node(Node::RequestBlock {
            request_id,
            block_id,
        });

        self.requests
            .entry(request_id)
            .and_modify(|e| {
                let inserted = e.insert(idx);
                assert!(inserted, "request block was already present");
            })
            .or_insert_with(|| HashSet::from([idx]));

        idx
    }

    fn visualize(&self) -> Dot<&StableGraph<Node<M>, Edge, Undirected>> {
        // TODO [later] can improve visualization (maybe different node styles for our different node types + use request_lookup to add cost of requests to request_blocks)
        //let request_lookup: HashMap<RequestId, String> = HashMap::new();
        //
        //let get_node_attributes = |g, node_ref: (NodeIndex, &Node<M>)| {
        //    let node = node_ref.1;
        //    let x = match node {
        //        Node::RequestBlock {
        //            request_id,
        //            block_id: _,
        //        } => request_lookup.get(request_id).unwrap(),
        //
        //        _ => "Hello",
        //    };
        //
        //    x.to_string()
        //};
        //
        //let get_edge_attributes = |g, node_ref| "Hello".to_string();
        //
        //let v1 =
        //    Dot::with_attr_getters(&self.graph, &[], &get_edge_attributes, &get_node_attributes);

        Dot::new(&self.graph)
    }

    #[allow(dead_code)]
    pub(crate) fn print_graph(&self) {
        println!("{}", self.visualize());
    }

    pub(crate) fn new(
        mut block_constraints: HashMap<BlockId, BlockConstraints<M>>,
        request_lookup: &HashMap<RequestId, Request>,
    ) -> Self {
        let mut pf = ProblemFormulation {
            graph: StableGraph::default(),
            requests: HashMap::new(),
        };
        // println!("bbbbb {:?}", block_constraints);
        for (block_id, constraint) in block_constraints.drain() {
            // insert nodes for accepted request blocks into graph + data structure
            for request_id in constraint.acceptable {
                pf.insert_request_block(request_id, block_id);
            }

            // insert request block nodes for contested
            let request_block_indexes: HashMap<RequestId, NodeIndex> = constraint
                .contested
                .iter()
                .map(|request_id| (*request_id, pf.insert_request_block(*request_id, block_id)))
                .collect();

            for segment in constraint.contested_segments.into_iter() {
                // insert node for each segment
                let segment_idx = pf.graph.add_node(Node::BlockSegmentConstraint {
                    budget: segment.remaining_budget,
                });

                // add edges between segments and request_block nodes
                for request_id in segment.request_ids.expect("missing request ids on segment") {
                    let request_block_idx = request_block_indexes
                        .get(&request_id)
                        .expect("missing request in request_block_indexes");

                    pf.graph.add_edge(*request_block_idx, segment_idx, Edge {});
                }
            }
        }

        //  eliminate requests which don't have sufficient number of blocks
        let request_block_idx_removal = pf
            .requests
            .iter()
            .filter(|(request_id, blocks)| blocks.len() < request_lookup[request_id].n_users)
            .flat_map(|(_request_id, blocks)| blocks)
            .copied()
            .collect();

        pf.remove_request_blocks(request_block_idx_removal, request_lookup);

        // TODO [later]: at the moment, for acceptable requests, we don't reject additional contested blocks which could make other requests acceptable
        // in the way we use the allocator, it should not make a difference.

        pf
    }

    pub fn allocate_request(
        &mut self,
        request_id: RequestId,
        allocated_blocks: &HashSet<BlockId>,
        request_lookup: &HashMap<RequestId, Request>,
    ) -> Result<(), AllocationError> {
        let request = request_lookup
            .get(&request_id)
            .expect("request missing in lookup");

        assert!(
            request.n_users <= allocated_blocks.len(),
            "too few blocks allocated for request"
        );

        if self.requests.get(&request_id).is_none() {
            return Err(AllocationError::IllegalRequestId(request_id));
        }

        let nodes = self.requests.get(&request_id).unwrap_or_else(|| {
            panic!(
                "missing request (illegal id or request previously allocated or rejected). Rid: {}",
                request_id
            )
        });

        let allocated_request_blocks: Vec<NodeIndex> = nodes
            .iter()
            .filter_map(|node_idx| match self.graph.node_weight(*node_idx) {
                Some(Node::RequestBlock { block_id, .. })
                    if allocated_blocks.contains(block_id) =>
                {
                    Some(*node_idx)
                }
                Some(Node::RequestBlock { .. }) => None, // ignore request block from a different block
                _ => panic!("must be request block + weight defined"),
            })
            .collect();

        if allocated_request_blocks.len() != allocated_blocks.len() {
            return Err(AllocationError::IllegalBlockAssignment(format!(
                "Not all allocated blocks are valid. Rid {}, allocated blocks: {:?}",
                request_id, allocated_blocks,
            )));
        }

        let segments_to_subtract_costs: Vec<NodeIndex> = allocated_request_blocks
            .into_iter()
            .flat_map(|node_idx| self.graph.neighbors(node_idx))
            .collect();

        // subtract the cost from each connected segment
        for segment_idx in segments_to_subtract_costs {
            match self.graph.node_weight_mut(segment_idx) {
                Some(Node::BlockSegmentConstraint { budget }) => {
                    budget.subtract_cost(&request.request_cost)
                }
                _ => panic!("something went wrong"),
            }
        }

        // remove the node and all edges from the request node
        let nodes_to_delete: Vec<NodeIndex> = nodes.iter().copied().collect();
        self.remove_request_blocks(nodes_to_delete, request_lookup);
        Ok(())
    }

    /// returns the status of the passed request id: Acceptable, contested, or rejected. See
    /// [Result] for more information on the return value. Strategy determines the order in which
    /// the blocks are returned.
    pub fn request_status(
        &self,
        request_id: RequestId,
        block_order_strategy: Option<BlockOrderStrategy>,
        request_lookup: &HashMap<RequestId, Request>,
    ) -> StatusResult {
        if !self.requests.contains_key(&request_id) {
            return StatusResult::Rejected;
        }

        let (mut acceptable_blocks, mut contested_blocks): (Vec<BlockId>, Vec<BlockId>) = self
            .requests
            .get(&request_id)
            .unwrap()
            .iter()
            .partition_map(|node_idx| {
                let n_constraints = self.graph.neighbors(*node_idx).count();

                let (_, block_id) = self.node_request_block(node_idx);

                if n_constraints == 0 {
                    Either::Left(block_id)
                } else {
                    Either::Right(block_id)
                }
            });

        // sort the blocks according to specific sort order (if strategy provided)
        if let Some(block_order_strategy) = block_order_strategy {
            acceptable_blocks.sort_by(|a, b| block_order_strategy.compare(a, b));

            contested_blocks.sort_by(|a, b| block_order_strategy.compare(a, b));
        }

        let request = request_lookup.get(&request_id).expect("missing request");
        if acceptable_blocks.len() >= request.n_users {
            StatusResult::Acceptable {
                acceptable: acceptable_blocks,
                contested: contested_blocks,
            }
        } else {
            StatusResult::Contested {
                acceptable: acceptable_blocks,
                contested: contested_blocks,
            }
        }
    }

    /// Returns the first acceptable request according to the strategy (if any request is still
    /// acceptable)
    pub(crate) fn next_acceptable(
        &self,
        block_order_strategy: Option<BlockOrderStrategy>,
        request_lookup: &HashMap<RequestId, Request>,
    ) -> Option<AcceptableRequest> {
        let min_acceptable_request_id = self
            .requests
            .keys()
            .filter(|&request_id| {
                self.n_acceptable_blocks(*request_id)
                    >= request_lookup
                        .get(request_id)
                        .expect("missing request in lookup")
                        .n_users
            })
            .min();

        // convert request id into Acceptable struct by reusing the request_status logic
        min_acceptable_request_id.and_then(|request_id| {
            match self.request_status(*request_id, block_order_strategy, request_lookup) {
                StatusResult::Acceptable {
                    acceptable,
                    contested,
                } => Some(AcceptableRequest {
                    request_id: *request_id,
                    acceptable,
                    contested,
                }),
                _ => panic!("conflicting decision whether acceptable or not between request_status and next_acceptable")
            }
        })
    }

    /// Returns how many blocks are acceptable for this request
    pub(crate) fn n_acceptable_blocks(&self, request_id: RequestId) -> usize {
        // if the request_id is not in self.requests -> means the request is not acceptable and thus has 0 acceptable blocks
        let n_acceptable_blocks = self.requests.get(&request_id).map_or(0, |blocks| {
            blocks
                .iter()
                .filter(|node_idx| {
                    let n_constraints = self.graph.neighbors(**node_idx).count();
                    n_constraints == 0 // blocks without constraints (i.e., without neighbors) are acceptable
                })
                .count()
        });
        n_acceptable_blocks
    }

    /*
    fn to_neighbor_set(&self, idx: &NodeIndex) -> BTreeSet<NodeIndex> {
        self.graph.neighbors(*idx).collect()
    }
     */

    #[allow(dead_code)]
    fn collapse(&mut self, _request_lookup: &HashMap<RequestId, Request>) {
        // TODO [later] implement collapse
        todo!("NOT IMPLEMENTED YET")
        // Goal: Identify Segments with the same incoming edges

        //let mut group_map = self
        //    .graph
        //    .node_indices()
        //    .filter(|node_idx| {
        //        matches!(
        //            self.graph.node_weight(*node_idx),
        //            Some(Node::BlockSegmentConstraint { .. })
        //        )
        //    })
        //    .into_group_map_by(|node_idx| {
        //        // group blocks by block_id
        //        let first_neighbor_idx = self
        //            .graph
        //            .neighbors(*node_idx)
        //            .next()
        //            .expect("segment node without neighbor");
        //        let (_request_id, block_id) = self.node_request_block(&first_neighbor_idx);
        //
        //        let (neighbor_count, neighbor_xor) =
        //            self.graph
        //                .neighbors(*node_idx)
        //                .fold((0 as usize, 0), |mut agg, idx| {
        //                    agg.0 += 1; // increment count
        //                    agg.1 = agg.1 ^ idx.index();
        //                    agg
        //                });
        //
        //        (block_id, neighbor_count, neighbor_xor)
        //    });
        //
        //let mut node_index_to_merge: Vec<Vec<NodeIndex>> = group_map
        //    .drain()
        //    .filter_map(|(k, index_vec)| {
        //        if index_vec.len() > 1 {
        //            let node_index_to_merge: Vec<Vec<NodeIndex>> = index_vec
        //                .into_iter()
        //                .map(|node_idx| (node_idx, self.to_neighbor_set(&node_idx)))
        //                .into_grouping_map_by(|(_a, neighbor_set)| neighbor_set)
        //                .fold(Vec::new(), |mut acc, _key, (node_idx, _neighbor_set)| {
        //                    acc.push(node_idx);
        //                    acc
        //                })
        //                .iter()
        //                .filter_map(|(k, v)| if v.len() > 1 { Some(*v) } else { None })
        //                .collect();
        //
        //            if node_index_to_merge.is_empty() {
        //                None
        //            } else {
        //                Some(node_index_to_merge)
        //            }
        //        } else {
        //            None
        //        }
        //    })
        //    .flatten()
        //    .collect();
        //
        //let mut request_blocks_delete = Vec::new();
        //
        //for node_index_set in node_index_to_merge.drain(..) {
        //    let mut iter = node_index_set.iter();
        //
        //    let first = iter.next().unwrap();
        //
        //    if let Some(Node::BlockSegmentConstraint { budget }) =
        //        self.graph.node_weight_mut(*first)
        //    {
        //        for other in iter {
        //            let other_budget = self.node_block_segment_constraint(other);
        //            budget.merge_assign(other_budget);
        //
        //            // delete the segment (after merging the budget into the main node)
        //            self.graph.remove_node(*other);
        //        }
        //
        //        let iter_delete_request_blocks =
        //            self.graph.neighbors(*first).filter(|neighbor_idx| {
        //                let (request_id, _block_id) = self.node_request_block(neighbor_idx);
        //
        //                let request = request_lookup
        //                    .get(&request_id)
        //                    .expect("missing request in lookup");
        //
        //                !budget.is_budget_sufficient(&request.request_cost)
        //            });
        //
        //        request_blocks_delete.extend(iter_delete_request_blocks);
        //    } else {
        //        panic!("something went wrong")
        //    }
        //}
        //
        //if !request_blocks_delete.is_empty() {
        //    self.remove_request_blocks(request_blocks_delete, request_lookup);
        //
        //    // recursively repeat collapsing because by removing certain request blocks, it's possible that we need to merge further segments
        //    self.collapse(request_lookup);
        //}
    }

    /// Iterates through all segments (combination of block and set of request ids)
    /// and returns the available budget.
    pub(crate) fn contested_constraints(
        &self,
    ) -> impl Iterator<Item = (BlockId, BTreeSet<RequestId>, &M)> {
        // TODO [later]: we could combine budgets of segments that have the same incoming request_ids  (maybe have version of contested constraints which first collapses)

        //self.collapse(request_lookup)

        // TODO [later] instead of iterating over all nodes of the graph, we could introduce an index to find all segment nodes and iterate over this -> problem is we would also have to delete segments there

        let iter = self.graph.node_indices().sorted().filter_map(|node_idx| {
            match self.graph.node_weight(node_idx) {
                Some(Node::BlockSegmentConstraint { budget }) => {
                    let block_id = self
                        .graph
                        .neighbors(node_idx)
                        .next()
                        .map(|idx| self.node_request_block(&idx).1)
                        .expect("segment must have at least one neighbor"); // At least I think this should be true

                    let request_ids: BTreeSet<RequestId> = self
                        .graph
                        .neighbors(node_idx)
                        .map(|idx| self.node_request_block(&idx).0)
                        .collect();

                    Some((block_id, request_ids, budget))
                }
                _ => None,
            }
        });

        iter
    }

    // ---------------------------------
    // helper functions
    // ---------------------------------

    // TODO: add comments for pre- und post-conditions
    fn remove_request_blocks(
        &mut self,
        request_block_idx_set: Vec<NodeIndex>,
        request_lookup: &HashMap<RequestId, Request>,
    ) {
        let concurrent_remove_queue: ConcurrentQueue<NodeIndex> = ConcurrentQueue::unbounded();

        // init the remove queue
        // let mut remove_queue: VecDeque<NodeIndex> = VecDeque::new();

        for request_block_idx in request_block_idx_set.into_iter() {
            concurrent_remove_queue
                .push(request_block_idx)
                .expect("queue push failed");
        }

        while let Ok(node_idx) = concurrent_remove_queue.pop() {
            if !self.graph.contains_node(node_idx) {
                // node was previously already deleted => ignore
                continue;
            }

            // when deleting a request, all of it's neighbors (segment constraints) can possibly be affected.
            //    (on delete it can happen that minimum number of blocks is violated => insert all others as deletion request)
            //      case 1: because of removing a request -> there is now enough budget and the segment is not contested anymore => delete segment node
            //      case 2: because of reduced budget (i.e., allocate before) -> there is not enough budget for a request -> reject request => insert new deletion request

            // identify segment nodes that are not contested (requested cost <= budget) and mark them for deletion
            let segment_idx_to_delete: Vec<NodeIndex> = self
                .graph
                .neighbors(node_idx)
                .par_bridge()
                .filter(|segment_idx| {
                    !self.is_contested_segment(
                        *segment_idx,
                        Some(node_idx),
                        &concurrent_remove_queue,
                        request_lookup,
                    )
                })
                .collect();

            for segment_idx in segment_idx_to_delete {
                self.graph.remove_node(segment_idx);
            }

            // finally delete the original request:

            let (request_id, _block_id) = self.node_request_block(&node_idx);
            let request = request_lookup.get(&request_id).expect("missing request");

            // remove the node from the graph
            self.graph.remove_node(node_idx);

            // update requests datastructure + detect if a request must be rejected due to a lack of blocks
            if let Some(request_block_idx_set) = self.requests.get_mut(&request_id) {
                // if not in requests -> we previously removed it because not enough blocks to fullfil request

                let is_removed = request_block_idx_set.remove(&node_idx);
                assert!(is_removed, "node idx not found for removal");

                if request_block_idx_set.len() < request.n_users {
                    //println!("remove all  request_id={}", request_id);
                    // because of removal -> not enough blocks remaining for request => delete all remaining request blocks
                    for idx in request_block_idx_set.iter() {
                        concurrent_remove_queue
                            .push(*idx)
                            .expect("queue push failed");
                    }

                    // delete the complete request entry
                    self.requests.remove(&request_id);
                }
            }
        }
    }

    fn is_contested_segment(
        &self,
        segment_idx: NodeIndex,
        ignore_request_block_idx: Option<NodeIndex>,
        remove_request_block_queue: &ConcurrentQueue<NodeIndex>,
        request_lookup: &HashMap<RequestId, Request>,
    ) -> bool {
        let budget = self.node_block_segment_constraint(&segment_idx);

        // sum request cost of incoming edges (!= ignore) and also ignore incoming requests where individual request costs already exceed the budget -> they need to be deleted
        let request_cost_sum = self
            .graph
            .neighbors(segment_idx)
            .filter_map(|request_block_idx| {
                // ignore request_blocks that we delete
                if Some(request_block_idx) == ignore_request_block_idx {
                    return None;
                }

                let (request_id, _block_id) = self.node_request_block(&request_block_idx);

                let request = request_lookup.get(&request_id).expect("missing request");

                if budget.is_budget_sufficient(&request.request_cost) {
                    Some(&request.request_cost)
                } else {
                    remove_request_block_queue
                        .push(request_block_idx)
                        .expect("queue push in is_contested_segment failed");
                    None
                }
            })
            .fold(None, |agg, cost| match agg {
                None => Some(cost.clone()),
                Some(agg) => Some(agg + cost),
            });

        // determine if contested
        match request_cost_sum {
            Some(request_cost_sum) if budget.is_budget_sufficient(&request_cost_sum) => false, // sufficient budget for all requests => not contested
            Some(_) => true, // else: not enough budget => contested
            None => false,   // no further incoming edges (i.e., zero request constraint)
        }
    }

    fn node_request_block(&self, node_idx: &NodeIndex) -> (RequestId, BlockId) {
        match self.graph.node_weight(*node_idx) {
            Some(Node::RequestBlock {
                request_id,
                block_id,
            }) => (*request_id, *block_id),
            weight => panic!(
                "failed: node is not a request block: {:?}  weight={:?}   contains={}",
                node_idx,
                weight,
                self.graph.contains_node(*node_idx)
            ),
        }
    }

    fn node_block_segment_constraint(&self, node_idx: &NodeIndex) -> &M {
        match self.graph.node_weight(*node_idx) {
            Some(Node::BlockSegmentConstraint { budget }) => budget,
            _ => panic!("not found"),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BlockConstraints<M: SegmentBudget> {
    /// requests that can be accepted on this block without considering the decision for other requests
    pub acceptable: HashSet<RequestId>,

    /// requests that exceed the available budget of the block individually and thus cannot run (independent of whether other requests run)
    pub rejected: HashSet<RequestId>,

    /// requests that could potentially run on the block but contest with other requests (i.e., only a subset of them can run at the same time)
    pub contested: HashSet<RequestId>,

    /// constraints that need to be fulfilled among the contested requests
    pub contested_segments: Vec<BlockSegment<M>>, // segment can be contested, uncontested, invalid
}

// TODO [later] Think about how to define and manage group of blocks, and what methods and datastructures are needed to support them
//pub(crate) type BlockGroupId = BlockId;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockSegment<M: SegmentBudget> {
    id: usize,
    pub request_ids: Option<Vec<RequestId>>,
    pub remaining_budget: M, // note: all must be satisfied
}

impl<M: SegmentBudget> BlockSegment<M> {
    pub fn new(id: usize) -> BlockSegment<M> {
        BlockSegment {
            id,
            request_ids: None,
            remaining_budget: M::new(),
        }
    }

    /*
    /// A shortcut for
    /// ```
    /// self.request_ids.as_mut().unwrap()
    /// ```
    pub(crate) fn get_rids_mut(&mut self) -> &mut Vec<RequestId> {
        self.request_ids.as_mut().unwrap()
    }

    /// A shortcut for
    /// ```
    /// self.request_ids.as_ref().unwrap()
    /// ```
    pub(crate) fn get_rids(&self) -> &Vec<RequestId> {
        self.request_ids.as_ref().unwrap()
    }

    /// A shortcut for
    /// ```
    /// self.request_ids = Some(new_rids);
    /// ```
    pub(crate) fn set_rids(&mut self, new_rids: Vec<RequestId>) {
        self.request_ids = Some(new_rids);
    }
     */
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap, HashSet};

    use crate::composition::{block_composition, block_composition_pa, CompositionConstraint};
    use crate::config::SegmentationAlgo;
    use crate::request::RequestBuilder;
    use crate::util::{self, build_dummy_schema};
    use crate::AccountingType::EpsDp;
    use crate::BlockId::User;
    use crate::{
        block::BlockId,
        composition::ProblemFormulation,
        dprivacy::{
            budget::{OptimalBudget, SegmentBudget},
            AccountingType,
        },
        request::{Request, RequestId},
    };

    use super::{BlockConstraints, BlockSegment, StatusResult};

    #[test]
    fn test_single_block_problem_formulation() {
        // TODO also have to test multi block setup (much more complicated because there we run reduction)
        let candidate_requests = build_mock_requests(4);

        // r0: acceptable
        // r1: rejected
        // r2 + r3: contested

        let mut block_constraints = HashMap::new();

        let remaining_budget =
            OptimalBudget::new_with_budget_constraint(&AccountingType::EpsDp { eps: 1.0 });

        let contested_segments = vec![BlockSegment {
            id: 1,
            request_ids: Some(vec![RequestId(2), RequestId(3)]),
            remaining_budget: remaining_budget.clone(),
        }];

        let block_constraint = BlockConstraints {
            acceptable: HashSet::from([RequestId(0)]),
            rejected: HashSet::from([RequestId(1)]),
            contested: HashSet::from([RequestId(2), RequestId(3)]),
            contested_segments,
        };

        let block_id = BlockId::User(1);

        block_constraints.insert(block_id, block_constraint);

        let problem_formulation: ProblemFormulation<OptimalBudget> =
            ProblemFormulation::new(block_constraints, &candidate_requests);

        println!("{}", problem_formulation.visualize());

        assert_eq!(
            problem_formulation.request_status(RequestId(0), None, &candidate_requests,),
            StatusResult::Acceptable {
                acceptable: vec![block_id],
                contested: vec![],
            },
            "acceptable in problem formulation does not match"
        );

        assert_eq!(
            problem_formulation.request_status(RequestId(1), None, &candidate_requests),
            StatusResult::Rejected,
            "rejected in problem formulation does not match"
        );

        assert_eq!(
            problem_formulation.request_status(RequestId(2), None, &candidate_requests),
            StatusResult::Contested {
                acceptable: vec![],
                contested: vec![block_id],
            },
            "contested in problem formulation does not match (R2)"
        );

        assert_eq!(
            problem_formulation.request_status(RequestId(3), None, &candidate_requests),
            StatusResult::Contested {
                acceptable: vec![],
                contested: vec![block_id],
            },
            "contested in problem formulation does not match (R3)"
        );

        let mut constraint_iter = problem_formulation.contested_constraints();

        let actual = constraint_iter
            .next()
            .expect("missing contested constraint");

        assert_eq!(
            (
                block_id,
                BTreeSet::from([RequestId(2), RequestId(3)]),
                &remaining_budget
            ),
            actual,
        );

        assert_eq!(
            constraint_iter.next(),
            None,
            "more than one contested constraint"
        );
    }

    fn request_mock(id: usize) -> Request {
        let schema = util::build_dummy_schema(EpsDp { eps: 1.0 });

        RequestBuilder::new(
            RequestId(id),
            AccountingType::EpsDp { eps: 0.5 },
            1,
            1,
            &schema,
            std::default::Default::default(),
        )
        .build()
    }

    fn build_mock_requests(n: usize) -> HashMap<RequestId, Request> {
        (0..n)
            .map(|x| {
                let r = request_mock(x);
                (r.request_id, r)
            })
            .collect()
    }

    #[test]
    fn test_problem_formulation_with_pa_unordered() {
        let (requests, mut pf) = get_problem_formulation_with_pa(10);

        // only r2 should be acceptable, and every other request should be contested
        // check that r2 is acceptable
        for rid in to_request_ids([2]) {
            match pf.request_status(rid, None, &requests) {
                StatusResult::Acceptable { .. } => {}
                _ => panic!("Request {} should be acceptable", rid),
            }
        }

        // check that all other requests are contested
        for rid in to_request_ids([0, 1, 3, 4, 5]) {
            match pf.request_status(rid, None, &requests) {
                StatusResult::Contested { .. } => {}
                _ => panic!("Request {} should be contested", rid),
            }
        }

        // there should be 3 contested segments per block and 10 blocks -> 30 contested segments
        assert_eq!(pf.contested_constraints().count(), 30);

        // allocate all acceptable requests
        let accepted_rids = accept_all_acceptable(&requests, &mut pf);
        // only r2 was acceptable, and was therefore allocated
        // since all other requests are part of >= 1 segment with >= 3 requests -> contested
        assert_eq!(accepted_rids, HashSet::from([RequestId(2)]));

        fn to_request_ids<const N: usize>(inp: [usize; N]) -> [RequestId; N] {
            let mut outp = [RequestId(0); N];
            for i in 0..N {
                outp[i] = RequestId(inp[i]);
            }
            outp
        }

        assert_eq!(pf.contested_constraints().count(), 30);

        // check that r2 is rejected
        for rid in to_request_ids([2]) {
            match pf.request_status(rid, None, &requests) {
                StatusResult::Rejected { .. } => {}
                _ => panic!("Request {} should be rejected", rid),
            }
        }

        // check that all other requests are contested
        for rid in to_request_ids([0, 1, 3, 4, 5]) {
            match pf.request_status(rid, None, &requests) {
                StatusResult::Contested { .. } => {}
                _ => panic!("Request {} should be contested", rid),
            }
        }

        // allocate r0 and r1, making r5 rejected
        pf.allocate_request(
            RequestId(0),
            &HashSet::from_iter((0..10).map(User)),
            &requests,
        )
        .expect("Allocating request failed");

        // still 30 contested segments, on each block (r1, r5), (r1, r3, r4, r5) and (r1, r3, r4)
        assert_eq!(pf.contested_constraints().count(), 30);

        pf.allocate_request(
            RequestId(1),
            &HashSet::from_iter((0..10).map(User)),
            &requests,
        )
        .expect("Allocating request failed");

        // Depending on if there is merging or not, should have one or two
        // contested constraints per segment left (both have budget ofr one more request, but
        // needed by r3 and r4).
        let cont_constr_count = pf.contested_constraints().count();
        assert!(cont_constr_count == 20 || cont_constr_count == 10);

        // check that the accept requests are rejected

        for rid in to_request_ids([0, 1, 2, 5]) {
            match pf.request_status(rid, None, &requests) {
                StatusResult::Rejected { .. } => {}
                _ => panic!("Request {} should be rejected", rid),
            }
        }

        // check that all other requests are contested
        for rid in to_request_ids([3, 4]) {
            match pf.request_status(rid, None, &requests) {
                StatusResult::Contested { .. } => {}
                _ => panic!("Request {} should be contested", rid),
            }
        }

        // allocate r3, making r4 rejected -> all request rejected

        pf.allocate_request(
            RequestId(3),
            &HashSet::from_iter((0..10).map(User)),
            &requests,
        )
        .expect("Allocating request failed");

        for rid in to_request_ids([0, 1, 2, 3, 4, 5]) {
            match pf.request_status(rid, None, &requests) {
                StatusResult::Rejected { .. } => {}
                _ => panic!("Request {} should be rejected", rid),
            }
        }

        // as all requests are rejected, there should be no more contested segments
        assert_eq!(pf.contested_constraints().count(), 0);
    }

    #[test]
    fn test_problem_formulation_no_pa_all_acceptable_unordered() {
        let (requests, mut pf) = get_problem_formulation_no_pa(2, 10);

        // No contested segments
        assert_eq!(pf.contested_constraints().count(), 0);

        let accepted_rids = accept_all_acceptable(&requests, &mut pf);

        assert_eq!(
            accepted_rids.len(),
            2,
            "Only exactly two requests were acceptable and should have been inserted"
        );

        assert_eq!(
            accepted_rids,
            HashSet::from_iter(requests.keys().copied()),
            "Did allocate a request which was not a candidate"
        );

        // check that the status of all requests is rejected (allocated requests also become rejected
        // in problem formulation)
        for rid in requests.keys() {
            match pf.request_status(*rid, None, &requests) {
                StatusResult::Rejected => {}
                _ => panic!("All requests should be allocated, and therefore rejected in problem formulation"),
            }
        }
    }

    #[test]
    fn test_problem_formulation_no_pa_all_contested_unordered() {
        let (requests, mut pf) = get_problem_formulation_no_pa(6, 10);

        // 1 contested segment per block since no pa
        assert_eq!(pf.contested_constraints().count(), 10);

        let accepted_rids = accept_all_acceptable(&requests, &mut pf);

        assert!(
            accepted_rids.is_empty(),
            "A request was acceptable, even though all should be contested"
        );

        // check that the status of all requests is contested
        for rid in requests.keys() {
            match pf.request_status(*rid, None, &requests) {
                StatusResult::Contested { .. } => {}
                _ => panic!("All requests should be contested"),
            }
        }

        pf.allocate_request(RequestId(0), &(0..10).map(User).collect(), &requests)
            .expect("Allocating request failed");

        // check that the status of all requests (except 0) is contested
        for rid in requests.keys() {
            if *rid != RequestId(0) {
                match pf.request_status(*rid, None, &requests) {
                    StatusResult::Contested { .. } => {}
                    _ => panic!("Request {} should be contested", rid),
                }
            }
        }

        // still one contested segment per block (all requests except 0 still contested on each block)
        assert_eq!(pf.contested_constraints().count(), 10);
        // status of request 0 should be rejected
        match pf.request_status(RequestId(0), None, &requests) {
            StatusResult::Rejected { .. } => {}
            _ => panic!("Request 0 should be rejected"),
        }

        pf.allocate_request(RequestId(1), &(0..10).map(User).collect(), &requests)
            .expect("Allocating request failed");

        // no more contested segments, as all requests rejected
        assert_eq!(pf.contested_constraints().count(), 0);
        // check that the status of all requests is rejected now
        for rid in requests.keys() {
            match pf.request_status(*rid, None, &requests) {
                StatusResult::Rejected { .. } => {}
                _ => panic!("All requests should be Rejected"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "too few blocks allocated for request")]
    fn test_try_allocate_fewer_blocks() {
        let (requests, mut pf) = get_problem_formulation_no_pa(2, 10);

        // two requests should be acceptable, but no more
        match pf.next_acceptable(None, &requests) {
            None => {
                panic!("Problem formulation did not return an acceptable request, but there must be at least one");
            }
            Some(acceptable_request) => {
                pf.allocate_request(
                    acceptable_request.request_id,
                    &acceptable_request.acceptable.into_iter().take(9).collect(),
                    &requests,
                )
                .expect("Allocating request failed");
            }
        }
    }

    #[test]
    #[should_panic(expected = "request missing in lookup")]
    fn test_try_allocate_nonexistent_request() {
        let (requests, mut pf) = get_problem_formulation_no_pa(6, 10);

        pf.allocate_request(
            RequestId(42),
            &HashSet::from_iter((0..10).map(User)),
            &requests,
        )
        .expect("Allocating request failed");
    }

    #[test]
    #[should_panic(expected = "Allocating request failed: IllegalRequestId")]
    fn test_try_double_allocation() {
        let (requests, mut pf) = get_problem_formulation_no_pa(6, 10);

        // any two requests should be allocatable, but no more (and not the same twice)
        for rid in requests.keys().take(1) {
            pf.allocate_request(*rid, &HashSet::from_iter((0..10).map(User)), &requests)
                .expect("Allocating request failed");
            pf.allocate_request(*rid, &HashSet::from_iter((0..10).map(User)), &requests)
                .expect("Allocating request failed");
        }
    }

    #[test]
    #[should_panic(expected = "Allocating request failed: IllegalRequestId")]
    fn test_try_allocate_over_budget() {
        let (requests, mut pf) = get_problem_formulation_no_pa(6, 10);

        // any two requests should be allocatable, but no more
        for rid in requests.keys().take(3) {
            pf.allocate_request(*rid, &HashSet::from_iter((0..10).map(User)), &requests)
                .expect("Allocating request failed");
        }
    }

    #[test]
    #[should_panic(expected = "Not all allocated blocks are valid")]
    fn test_try_allocate_nonexistent_blocks() {
        let (requests, mut pf) = get_problem_formulation_no_pa(6, 10);

        // any two requests should be allocatable, but no more
        for rid in requests.keys().take(2) {
            pf.allocate_request(*rid, &HashSet::from_iter((10..20).map(User)), &requests)
                .expect("Allocating request failed");
        }
    }

    #[test]
    fn test_n_acceptable_with_pa() {
        let (_requests, pf) = get_problem_formulation_with_pa(10);

        // For request 2, all blocks should be acceptable
        assert_eq!(pf.n_acceptable_blocks(RequestId(2)), 10);
        // and none for the other requests
        assert_eq!(pf.n_acceptable_blocks(RequestId(0)), 0);
        assert_eq!(pf.n_acceptable_blocks(RequestId(1)), 0);
        assert_eq!(pf.n_acceptable_blocks(RequestId(3)), 0);
        assert_eq!(pf.n_acceptable_blocks(RequestId(4)), 0);
        assert_eq!(pf.n_acceptable_blocks(RequestId(5)), 0);
    }

    #[test]
    fn test_n_acceptable_no_pa() {
        let (requests, mut pf) = get_problem_formulation_no_pa(3, 11);

        pf.allocate_request(
            RequestId(0),
            &HashSet::from_iter((0..10).map(User)),
            &requests,
        )
        .expect("Allocating request failed");

        // Block 11 should now be acceptable for the two remaining requests
        assert_eq!(pf.n_acceptable_blocks(RequestId(1)), 1);
        assert_eq!(pf.n_acceptable_blocks(RequestId(2)), 1);
    }

    /// greedily allocates acceptable requests
    fn accept_all_acceptable(
        requests: &HashMap<RequestId, Request>,
        pf: &mut ProblemFormulation<OptimalBudget>,
    ) -> HashSet<RequestId> {
        let mut accepted_rids = HashSet::new();
        while let Some(acceptable_request) = pf.next_acceptable(None, requests) {
            let inserted = accepted_rids.insert(acceptable_request.request_id);
            assert!(
                inserted,
                "next_acceptable returned a request id which is already allocated"
            );
            pf.allocate_request(
                acceptable_request.request_id,
                &acceptable_request.acceptable.into_iter().take(10).collect(),
                requests,
            )
            .expect("Allocating request failed");
        }
        accepted_rids
    }

    /// helper function for tests to initialize a common problem formulation.
    fn get_problem_formulation_no_pa(
        n_requests: usize,
        n_blocks: usize,
    ) -> (
        HashMap<RequestId, Request>,
        ProblemFormulation<OptimalBudget>,
    ) {
        let _n_blocks: usize = 5;

        let schema = build_dummy_schema(EpsDp { eps: 1.0 });

        let mut requests =
            crate::util::build_dummy_requests_with_pa(&schema, 10, EpsDp { eps: 0.4 }, 6);

        assert!(n_requests <= requests.len());
        requests.retain(|rid, _| *rid < RequestId(n_requests));

        let blocks = crate::util::generate_blocks(0, n_blocks, EpsDp { eps: 1.0 });

        let block_composition = block_composition::build_block_composition();

        let pf = block_composition.build_problem_formulation::<OptimalBudget>(
            &blocks,
            &requests,
            &HashMap::new(),
            &schema,
            &mut Vec::new(),
        );
        (requests, pf)
    }

    /// helper function for tests to initialize a common problem formulation with PA.
    fn get_problem_formulation_with_pa(
        n_blocks: usize,
    ) -> (
        HashMap<RequestId, Request>,
        ProblemFormulation<OptimalBudget>,
    ) {
        let _n_blocks: usize = 5;

        let schema = build_dummy_schema(EpsDp { eps: 1.0 });

        let requests =
            crate::util::build_dummy_requests_with_pa(&schema, 10, EpsDp { eps: 0.4 }, 6);

        let blocks = crate::util::generate_blocks(0, n_blocks, EpsDp { eps: 1.0 });

        let block_composition =
            block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray);

        let pf = block_composition.build_problem_formulation::<OptimalBudget>(
            &blocks,
            &requests,
            &HashMap::new(),
            &schema,
            &mut Vec::new(),
        );
        (requests, pf)
    }
}
