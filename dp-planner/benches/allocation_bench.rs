use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use dp_planner_lib::allocation::BlockCompWrapper;
use dp_planner_lib::composition::block_composition_pa;
use dp_planner_lib::config::SegmentationAlgo;
use dp_planner_lib::dprivacy::budget::OptimalBudget;
use dp_planner_lib::dprivacy::rdp_alphas_accounting::RdpAlphas;
use dp_planner_lib::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
use dp_planner_lib::dprivacy::AccountingType::{EpsDeltaDp, Rdp};
use dp_planner_lib::dprivacy::{Accounting, AdpAccounting, RdpAccounting};
use dp_planner_lib::request::adapter::RequestAdapter;
use dp_planner_lib::request::{load_requests, resource_path, Request, RequestId};
use dp_planner_lib::schema::load_schema;
use dp_planner_lib::util::{CENSUS_REQUESTS, CENSUS_SCHEMA};
use itertools::Itertools;
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;

const ALPHAS13: RdpAlphas = A13([
    1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 8., 16., 32., 64., 1000000.,
]);

const ALPHAS2: RdpAlphas = A2([32., 64.]);

/// New bench to benchmark the speed of ilp allocation. Unfortunately, need to do at least
/// 10 repetitions, so not suited to benchmark larger allocations.
pub fn allocations_outer(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocations");

    group.measurement_time(std::time::Duration::new(20, 0));
    group.noise_threshold(0.05);
    ilp_allocation(&mut group);
}

fn ilp_allocation(group: &mut BenchmarkGroup<WallTime>) {
    two_ilp_allocations(group, &ALPHAS13);
    two_ilp_allocations(group, &ALPHAS2);

    /*
    let resource_allocation = ilp.round::<OptimalBudget>(
        &request_batch,
        &request_history,
        &blocks,
        &census_schema,
        &block_comp_wrapper,
        &Some(Rdp { eps_values: ALPHAS }),
    );

     */
}

fn two_ilp_allocations(group: &mut BenchmarkGroup<WallTime>, alphas: &RdpAlphas) {
    let round_size = 150;
    let n_total_req = 300;

    let alpha_acc_type = Rdp {
        eps_values: alphas.clone(),
    };

    let budget = EpsDeltaDp {
        eps: 1.,
        delta: 1e-7,
    }
    .adp_to_rdp_budget(alphas);

    let census_schema =
        load_schema(resource_path(CENSUS_SCHEMA), &budget).expect("Loading schema failed");

    let adapter_pathbuf = PathBuf::from_str(
        "./resources/test/adapter_configs/adapter_config_hard_assignment_10_blocks.json",
    )
    .expect("Getting adapter pathbuf failed");

    // loads requests and converts them to internal format
    let census_requests = load_requests(
        resource_path(CENSUS_REQUESTS),
        &census_schema,
        &mut RequestAdapter::new(adapter_pathbuf, 42),
        &Some(alphas.clone()),
    )
    .expect("Loading requests failed");

    let mut census_requests: Vec<(RequestId, Request)> = census_requests
        .into_iter()
        .sorted_by(|(id1, _), (id2, _)| Ord::cmp(id1, id2))
        .take(n_total_req)
        .collect();

    let blocks = dp_planner_lib::util::generate_blocks(
        0,
        10,
        EpsDeltaDp {
            eps: 1.0,
            delta: 1e-7,
        }
        .adp_to_rdp_budget(alphas),
    );

    let block_comp_wrapper = BlockCompWrapper::BlockCompositionPartAttributesVariant(
        block_composition_pa::build_block_part_attributes(SegmentationAlgo::Narray),
    );

    let mut round = 0;
    while !census_requests.is_empty() {
        let request_batch: Vec<(RequestId, Request)> =
            census_requests.drain(0..round_size).collect();

        let name = format!(
            "greedy_request_batch_{}_alphas_{}",
            round,
            alpha_acc_type.get_rdp_vec().len()
        );

        group.bench_function(&name, |b| {
            b.iter(|| {
                let mut request_history: HashMap<RequestId, Request> = HashMap::new();
                let mut greedy = dp_planner_lib::allocation::greedy::Greedy::construct_allocator();
                let mut available_blocks = blocks.clone();
                for (rid, request) in request_batch.iter().cloned() {
                    let curr_batch = HashMap::from_iter(vec![(rid, request.clone())].into_iter());
                    let resource_allocation = greedy.round::<OptimalBudget>(
                        &curr_batch,
                        &request_history,
                        &blocks,
                        &census_schema,
                        &block_comp_wrapper,
                        &mut Vec::new(),
                    );
                    dp_planner_lib::simulation::util::update_block_history(
                        &mut available_blocks,
                        &resource_allocation.0,
                    );
                    let inserted = request_history.insert(rid, request);
                    assert!(inserted.is_none())
                }
                black_box(&request_history);
            });
        });

        let name = format!(
            "dpf_request_batch_{}_alphas_{}",
            round,
            alpha_acc_type.get_rdp_vec().len()
        );

        group.bench_function(&name, |b| {
            b.iter(|| {
                let mut request_history: HashMap<RequestId, Request> = HashMap::new();
                let mut dpf = dp_planner_lib::allocation::dpf::Dpf::construct_allocator(
                    &budget, 42, false, true,
                );
                let mut available_blocks = blocks.clone();
                let request_batch_size = request_batch.len();
                for (subround, (rid, request)) in request_batch.iter().cloned().enumerate() {
                    let curr_batch = HashMap::from_iter(vec![(rid, request.clone())].into_iter());
                    for (_, block) in available_blocks.iter_mut() {
                        block.unlocked_budget = budget.clone();
                        block.unlocked_budget.apply_func(&|b: f64| {
                            b * (subround as f64) / (request_batch_size as f64)
                        })
                    }
                    let resource_allocation = dpf.round::<OptimalBudget>(
                        &curr_batch,
                        &request_history,
                        &blocks,
                        &census_schema,
                        &block_comp_wrapper,
                        &mut Vec::new(),
                    );
                    dp_planner_lib::simulation::util::update_block_history(
                        &mut available_blocks,
                        &resource_allocation.0,
                    );
                    let inserted = request_history.insert(rid, request);
                    assert!(inserted.is_none())
                }
                black_box(&request_history);
            })
        });

        let name = format!(
            "ilp_{}_request_batch_{}_alphas_{}",
            request_batch.len(),
            round,
            alpha_acc_type.get_rdp_vec().len()
        );

        let request_batch_map: HashMap<RequestId, Request> = request_batch.into_iter().collect();

        group.bench_function(&name, |b| {
            b.iter(|| {
                let request_history: HashMap<RequestId, Request> = HashMap::new();
                let mut ilp = dp_planner_lib::allocation::ilp::Ilp::construct_allocator();
                let _resource_allocation = ilp.round::<OptimalBudget>(
                    &request_batch_map,
                    &request_history,
                    &blocks,
                    &census_schema,
                    &block_comp_wrapper,
                    &Some(Rdp {
                        eps_values: alphas.clone(),
                    }),
                    &mut Vec::new(),
                );
                black_box(&request_history);
            })
        });
        round += 1;
    }
}

criterion_group!(segmentations, allocations_outer);
criterion_main!(segmentations);
