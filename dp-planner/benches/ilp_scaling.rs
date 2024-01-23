use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dp_planner_lib::allocation::{BlockCompWrapper, ResourceAllocation};
use dp_planner_lib::block::{Block, BlockId};
use dp_planner_lib::composition::block_composition_pa;
use dp_planner_lib::config::SegmentationAlgo;
use dp_planner_lib::dprivacy::budget::OptimalBudget;
use dp_planner_lib::dprivacy::rdp_alphas_accounting::RdpAlphas;
use dp_planner_lib::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
use dp_planner_lib::dprivacy::AccountingType::{EpsDeltaDp, Rdp};
use dp_planner_lib::dprivacy::{AccountingType, AdpAccounting};
use dp_planner_lib::request::adapter::RequestAdapter;
use dp_planner_lib::request::{load_requests, resource_path, Request, RequestId};
use dp_planner_lib::schema::{load_schema, Schema};
use dp_planner_lib::util::{CENSUS_REQUESTS, CENSUS_SCHEMA};
use itertools::Itertools;
use std::collections::HashMap;
use std::path::PathBuf;

#[allow(dead_code)]
const ALPHAS13: RdpAlphas = A13([
    1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 8., 16., 32., 64., 1000000.,
]);

const ALPHAS2: RdpAlphas = A2([32., 64.]);

/// A struct which contains all necessary information to run a round of allocation.
struct RoundConfig {
    alphas: Option<AccountingType>,
    #[allow(dead_code)]
    budget: AccountingType,
    schema: Schema,
    request_history: HashMap<RequestId, Request>,
    candidate_requests: HashMap<RequestId, Request>,
    available_blocks: HashMap<BlockId, Block>,
    block_comp_wrapper: BlockCompWrapper,
}

impl RoundConfig {
    fn run_ilp_round(&self) -> ResourceAllocation {
        let mut ilp_allocator = dp_planner_lib::allocation::ilp::Ilp::construct_allocator();
        let (ra, _as) = ilp_allocator.round::<OptimalBudget>(
            &self.candidate_requests,
            &self.request_history,
            &self.available_blocks,
            &self.schema,
            &self.block_comp_wrapper,
            &self.alphas,
            &mut Vec::new(),
        );
        ra
    }
}

fn get_adapter(seed: u128, new_n_blocks_and_probs: &[(usize, f64)]) -> RequestAdapter {
    let mut adapter = RequestAdapter::new(
        PathBuf::from(
            "resources/test/adapter_configs/adapter_config_hard_assignment_10_blocks.json",
        ),
        seed,
    );
    adapter.change_n_blocks(new_n_blocks_and_probs);
    adapter
}

/// Tests the scaling of the ilp allocation, but in contrast to the other bench,
/// only considers a single round of ilp allocation, and not the whole allocation procedure.
fn ilp_allocation(c: &mut Criterion) {
    let seed: u128 = 42;
    let num_block_sizes: Vec<usize> = vec![1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    let history_sizes: Vec<usize> = vec![0, 512, 1024];
    let batch_sizes: Vec<usize> = vec![64, 128, 256, 512];

    for history_size in history_sizes {
        for batch_size in batch_sizes.iter() {
            let group_name = format!(
                "history_size_{}_batch_size_{}_seed_{}",
                history_size, batch_size, seed
            );
            let mut group = c.benchmark_group(group_name);
            group.sample_size(10);
            group.noise_threshold(0.05);
            for num_blocks in num_block_sizes.iter() {
                let round_config =
                    prepare_run(&ALPHAS2, *batch_size, history_size, *num_blocks, seed);
                group.bench_with_input(
                    BenchmarkId::new("ilp_scaling", num_blocks),
                    num_blocks,
                    |b, _num_blocks| {
                        b.iter(|| {
                            black_box(&round_config.run_ilp_round());
                        })
                    },
                );
            }
            group.finish();
        }
    }
}

fn prepare_run(
    alphas: &RdpAlphas,
    batch_size: usize,
    history_size: usize,
    num_blocks: usize,
    seed: u128,
) -> RoundConfig {
    let n_total_req = batch_size + history_size;
    assert_eq!(
        history_size % batch_size,
        0,
        "History size must be zero or a multiple of batch size"
    );
    assert!(
        n_total_req <= 3415,
        "History size + batch size must be <= 3415"
    );
    assert!(
        num_blocks == 1 || (num_blocks > 1 && num_blocks % 2 == 0),
        "num blocks needs to be either 1 or even and > 1"
    );

    let optional_alphas = Some(Rdp {
        eps_values: alphas.clone(),
    });

    let budget = EpsDeltaDp {
        eps: 1.,
        delta: 1e-7,
    }
    .adp_to_rdp_budget(alphas);

    let census_schema =
        load_schema(resource_path(CENSUS_SCHEMA), &budget).expect("Loading schema failed");

    let new_n_blocks_and_probs = if num_blocks == 1 {
        vec![(1, 1.0 / 3.0), (1, 1.0 / 3.0), (1, 1.0 / 3.0)]
    } else {
        vec![
            (1, 1.0 / 3.0),
            (num_blocks / 2, 1.0 / 3.0),
            (num_blocks, 1.0 / 3.0),
        ]
    };

    let mut adapter = get_adapter(seed, &new_n_blocks_and_probs);

    // loads requests and converts them to internal format
    let census_requests = load_requests(
        resource_path(CENSUS_REQUESTS),
        &census_schema,
        &mut adapter,
        &Some(alphas.clone()),
    )
    .expect("Loading requests failed");

    let mut census_requests: Vec<(RequestId, Request)> = census_requests
        .into_iter()
        .sorted_by(|(id1, _), (id2, _)| Ord::cmp(id1, id2))
        .take(n_total_req)
        .collect();

    let mut available_blocks = dp_planner_lib::util::generate_blocks(
        0,
        num_blocks,
        EpsDeltaDp {
            eps: 1.0,
            delta: 1e-7,
        }
        .adp_to_rdp_budget(alphas),
    );

    let block_comp_wrapper = BlockCompWrapper::BlockCompositionPartAttributesVariant(
        block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray),
    );

    let mut request_history: HashMap<RequestId, Request> = HashMap::new();

    // run simulation until only batch_size remain (i.e., generate history)
    while census_requests.len() > batch_size {
        let mut ilp = dp_planner_lib::allocation::ilp::Ilp::construct_allocator();

        let request_batch: HashMap<RequestId, Request> =
            census_requests.drain(0..batch_size).collect();

        let (resource_allocation, _as) = ilp.round::<OptimalBudget>(
            &request_batch,
            &request_history,
            &available_blocks,
            &census_schema,
            &block_comp_wrapper,
            &optional_alphas,
            &mut Vec::new(),
        );
        dp_planner_lib::simulation::util::update_block_history(
            &mut available_blocks,
            &resource_allocation,
        );
        for rid in resource_allocation.accepted.keys() {
            let inserted = request_history.insert(*rid, request_batch[rid].clone());
            assert!(inserted.is_none())
        }
    }

    RoundConfig {
        alphas: optional_alphas,
        budget,
        schema: census_schema,
        request_history,
        candidate_requests: census_requests.into_iter().collect(),
        available_blocks,
        block_comp_wrapper,
    }
}

criterion_group!(segmentations, ilp_allocation);
criterion_main!(segmentations);
