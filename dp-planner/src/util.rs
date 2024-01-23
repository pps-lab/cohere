//! Contains a variety of functions and some static variables which may be useful to have in
//! different modules, but are usually only part of the test modules, i.e., they are normally
//! only invoked when running the unit tests.

use crate::composition::{CompositionConstraint, ProblemFormulation};
use crate::config::{Budget, BudgetTotal, BudgetType, CompositionConfig, SegmentationAlgo};
use crate::request::{
    AttributeId, ConjunctionBuilder, Predicate, Request, RequestBuilder, RequestId,
};
use crate::schema::{Attribute, Schema, ValueDomain};
use crate::BlockId::User;
use crate::{AccountingType, Block, BlockId, OptimalBudget, RoundId};
use itertools::Itertools;
use log::trace;
use rayon::ThreadPoolBuildError;
use std::cmp::min;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// The path of the census schema relative to resources/test
#[allow(dead_code)]
pub static CENSUS_SCHEMA: &str = "schema_files/census_schema.json";
/// The path of the census requests relative to resources/test
#[allow(dead_code)]
pub static CENSUS_REQUESTS: &str = "request_files/census_requests.json";

/*
pub fn demo() {
    let hasher_size = mem::size_of::<seahash::SeaHasher>();
    println!("hasher_size= {} bytes", hasher_size);

    let size = mem::size_of::<AccountingType>();
    println!("accouting type size= {} bytes", size);
}
*/

/// Builds a schema with two attributes a0 and a1, with 2 and 3 possible values respectively, and
/// and EpsDp accounting type schema.
#[allow(dead_code)]
pub fn build_dummy_schema(dp_type: AccountingType) -> Schema {
    // test with two partitioning attributes one with two values and the other with three
    let attributes: Vec<Attribute> = vec![
        Attribute {
            name: "a0".to_owned(),
            value_domain: ValueDomain::Range { min: 0, max: 1 },
            value_domain_map: None,
        },
        Attribute {
            name: "a1".to_owned(),
            value_domain: ValueDomain::Range { min: 0, max: 2 },
            value_domain_map: None,
        },
    ];

    Schema {
        accounting_type: dp_type,
        attributes,
        name_to_index: HashMap::new(),
    }
}

/// Returns max(7, n_requests) requests with two partitioning attributes, with n_users and request
/// cost as specified. The requests access the attributes with attribute ids 0 and 1 from the
/// schema, which should have values 0, 1 and 2 (or more) for the attribute with id 1, and 0 and 1
/// for the attribute with id 0, as the requests are defined to match to a combination of these
/// attributes and values.
///
/// See the following table for which requests apply to which virtual block. Note that which segments
/// are created depends on things like the request cost and available budget.
///
/// |           | a1 = 0        | a1 = 1            | a1 = 2            |
/// | :---:     | :---:         | :---:             | :---:             |
/// | a0 = 0    | (r0, r1, r5)  | (r1, r3, r4, r5)  | (r1, r3, r4, r6)  |
/// | a0 = 1    | (r0)          | (r3, r4)          | (r2, r3, r6)      |
#[allow(dead_code)]
pub fn build_dummy_requests_with_pa(
    schema: &Schema,
    n_users: usize,
    request_cost: AccountingType,
    n_requests: usize,
) -> HashMap<RequestId, Request> {
    let mut requests = vec![
        RequestBuilder::new(
            RequestId(0),
            request_cost.clone(),
            1,
            n_users,
            schema,
            Default::default(),
        )
        .or_conjunction(
            ConjunctionBuilder::new(schema)
                .and(AttributeId(1), Predicate::Eq(0))
                .build(),
        )
        .build(),
        RequestBuilder::new(
            RequestId(1),
            request_cost.clone(),
            1,
            n_users,
            schema,
            Default::default(),
        )
        .or_conjunction(
            ConjunctionBuilder::new(schema)
                .and(AttributeId(0), Predicate::Eq(0))
                .build(),
        )
        .build(),
        RequestBuilder::new(
            RequestId(2),
            request_cost.clone(),
            1,
            n_users,
            schema,
            Default::default(),
        )
        .or_conjunction(
            ConjunctionBuilder::new(schema)
                .and(AttributeId(0), Predicate::Eq(1))
                .and(AttributeId(1), Predicate::Eq(2))
                .build(),
        )
        .build(),
        RequestBuilder::new(
            RequestId(3),
            request_cost.clone(),
            1,
            n_users,
            schema,
            Default::default(),
        )
        .or_conjunction(
            ConjunctionBuilder::new(schema)
                .and(AttributeId(1), Predicate::Between { min: 1, max: 2 })
                .build(),
        )
        .build(),
        RequestBuilder::new(
            RequestId(4),
            request_cost.clone(),
            1,
            n_users,
            schema,
            Default::default(),
        )
        .or_conjunction(
            ConjunctionBuilder::new(schema)
                .and(AttributeId(1), Predicate::Eq(1))
                .build(),
        )
        .or_conjunction(
            ConjunctionBuilder::new(schema)
                .and(AttributeId(0), Predicate::Eq(0))
                .and(AttributeId(1), Predicate::Eq(2))
                .build(),
        )
        .build(),
        RequestBuilder::new(
            RequestId(5),
            request_cost.clone(),
            1,
            n_users,
            schema,
            Default::default(),
        )
        .or_conjunction(
            ConjunctionBuilder::new(schema)
                .and(AttributeId(0), Predicate::Eq(0))
                .and(AttributeId(1), Predicate::Between { min: 0, max: 1 })
                .build(),
        )
        .build(),
        RequestBuilder::new(
            RequestId(6),
            request_cost,
            1,
            n_users,
            schema,
            Default::default(),
        )
        .or_conjunction(
            ConjunctionBuilder::new(schema)
                .and(AttributeId(1), Predicate::Eq(2))
                .build(),
        )
        .build(),
    ];

    assert!(n_requests <= requests.len());
    requests = requests.into_iter().take(n_requests).collect();

    requests
        .into_iter()
        .map(|request| (request.request_id, request))
        .collect()
}

/// Similar to [build_dummy_requests_with_pa], but requests always apply to all virtual blocks.
///
/// Note that at most 7 requests are generated, to conform to the semantics of [build_dummy_requests_with_pa]
#[allow(dead_code)]
pub fn build_dummy_requests_no_pa(
    schema: &Schema,
    n_users: usize,
    request_cost: AccountingType,
    n_requests: usize,
) -> HashMap<RequestId, Request> {
    (0..min(7, n_requests))
        .map(|i| {
            RequestBuilder::new(
                RequestId(i),
                request_cost.clone(),
                1,
                n_users,
                schema,
                Default::default(),
            )
            .build()
        })
        .map(|request| (request.request_id, request))
        .collect()
}

/// Generates blocks via [generate_blocks], and then writes these blocks to the given path.
pub fn generate_and_write_blocks(
    start: usize,
    end: usize,
    budget: AccountingType,
    path: &Path,
) -> Result<(), std::io::Error> {
    let blocks = generate_blocks(start, end, budget)
        .into_iter()
        .map(|(_, block)| crate::block::external::ExternalBlock {
            id: block.id,
            request_ids: block.request_history,
            unlocked_budget: block.unlocked_budget,
            created: block.created,
            retired: block.retired,
        })
        .collect::<Vec<_>>();
    let stringify = if (end - start) == 1 {
        serde_json::to_string_pretty(&blocks)?
    } else {
        serde_json::to_string(&blocks)?
    };
    let mut file = File::create(path)?;
    file.write_all(stringify.as_ref())?;
    Ok(())
}

/// Generates blocks with ids start..end, given budget as unlocked_budget, empty request histories
/// and created time 0.
pub fn generate_blocks(
    start: usize,
    end: usize,
    unlocked_budget: AccountingType,
) -> HashMap<BlockId, Block> {
    (start..end)
        .map(|num| {
            (
                User(num),
                Block {
                    id: User(num),
                    request_history: vec![],
                    unlocked_budget: unlocked_budget.clone(),
                    unreduced_unlocked_budget: unlocked_budget.clone(),
                    created: RoundId(0),
                    retired: None,
                },
            )
        })
        .collect::<HashMap<_, _>>()
}

/// Constructs a problem formulation and then replays the given allocation on top of it
pub fn construct_pf_and_replay_allocation(
    blocks: &HashMap<BlockId, Block>,
    candidate_requests: &HashMap<RequestId, Request>,
    history_requests: &HashMap<RequestId, Request>,
    schema: &Schema,
    ilp_allocated_requests: &Vec<RequestId>,
    accepted_requests: &BTreeMap<RequestId, HashSet<BlockId>>,
) {
    let block_composition = crate::composition::block_composition_pa::build_block_part_attributes(
        SegmentationAlgo::Narray,
    );

    let mut pf: ProblemFormulation<OptimalBudget> = block_composition
        .build_problem_formulation::<OptimalBudget>(
            blocks,
            candidate_requests,
            history_requests,
            schema,
            &mut Vec::new(),
        );

    for rid in ilp_allocated_requests {
        trace!(
            "Replaying allocation for request {}. Num virtual blocks: {}, Status: {:?}, assigned {:?}",
            rid,
            candidate_requests[rid].dnf().num_virtual_blocks(schema),
            pf.request_status(*rid, Some(crate::composition::BlockOrderStrategy::Id), candidate_requests),
            accepted_requests[rid].iter().sorted().collect::<Vec<_>>()
        );
        pf.allocate_request(*rid, &accepted_requests[rid], candidate_requests)
            .expect("Allocating request failed");
    }
}

/// Can be used in unit tests where a composition is needed (currently EfficiencyBased allocation)
#[allow(dead_code)]
pub fn get_dummy_composition() -> CompositionConfig {
    CompositionConfig::BlockComposition {
        budget: Budget::FixBudget {
            budget: BudgetTotal {
                budget_file: None,
                epsilon: None,
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
            },
        },
        budget_type: BudgetType::OptimalBudget,
    }
}

/// Use this to change  the number of rayon threads if parallelization is desired, but the default
/// number of threads (= number of virtual cores) is too much.
/// via https://stackoverflow.com/questions/59205184/how-can-i-change-the-number-of-threads-rayon-uses
pub fn create_pool(num_threads: usize) -> Result<rayon::ThreadPool, ThreadPoolBuildError> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
}
