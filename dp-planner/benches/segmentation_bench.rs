use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use dp_planner_lib::composition::block_composition_pa::algo_hashmap::HashmapSegmentor;
use dp_planner_lib::composition::block_composition_pa::algo_narray::NArraySegmentation;
use dp_planner_lib::composition::block_composition_pa::Segmentation;
use dp_planner_lib::composition::BlockConstraints;
use dp_planner_lib::dprivacy::budget::OptimalBudget;
use dp_planner_lib::dprivacy::rdp_alphas_accounting::RdpAlphas;
use dp_planner_lib::dprivacy::rdp_alphas_accounting::RdpAlphas::A13;
use dp_planner_lib::dprivacy::AccountingType::EpsDeltaDp;
use dp_planner_lib::dprivacy::{AccountingType, AdpAccounting};
use dp_planner_lib::request::adapter::RequestAdapter;
use dp_planner_lib::request::{load_requests, resource_path, Request, RequestId};
use dp_planner_lib::schema::{load_schema, Schema};
use dp_planner_lib::util::{CENSUS_REQUESTS, CENSUS_SCHEMA};
use itertools::Itertools;
use std::collections::BTreeMap;

const ALPHAS: RdpAlphas = A13([
    1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 8., 16., 32., 64., 1000000.,
]);

/// Another criterion benchmark, this time to measure the performance of the segmentation.
/// Again, an issue is that there need to be at least 10 repetitions, which can be very long for
/// larger, more interesting examples.

pub fn segmentations_outer(c: &mut Criterion) {
    let mut group = c.benchmark_group("segmentations");

    /*
    let epsilons = [2., 4., 8., 16., 32.];
    let deltas = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10];
    let history_frac = 0.1;

     */
    let epsilons = [2.];
    let deltas = [1e-6];
    for eps in epsilons {
        for delta in deltas {
            group.measurement_time(std::time::Duration::new(20, 0));
            group.noise_threshold(0.05);
            narray_segmentation_particular(&mut group, eps, delta);
            group.measurement_time(std::time::Duration::new(60, 0));
            group.noise_threshold(0.02);
            hashmap_segmentation_particular(&mut group, eps, delta);
        }
    }
}

pub fn narray_segmentation_particular(group: &mut BenchmarkGroup<WallTime>, eps: f64, delta: f64) {
    let (census_schema, request_history, request_batch, initial_budget) =
        segmentation_setup(eps, delta);

    let name = format!("narray_eps_{}_delta_{}", eps, delta);

    group.bench_function(&name, |b| {
        b.iter(|| {
            let algo_narray =
                NArraySegmentation::new(request_batch.iter().collect(), &census_schema);
            let problem_formulation_algo_narray: BlockConstraints<OptimalBudget> = algo_narray
                .compute_block_constraints(request_history.iter().collect(), &initial_budget);
            black_box(&problem_formulation_algo_narray);
        })
    });
}

pub fn hashmap_segmentation_particular(group: &mut BenchmarkGroup<WallTime>, eps: f64, delta: f64) {
    let (census_schema, request_history, request_batch, initial_budget) =
        segmentation_setup(eps, delta);

    let name = format!("hashseg_eps_{}_delta_{}", eps, delta);

    group.bench_function(&name, |b| {
        b.iter(|| {
            let algo_hash = HashmapSegmentor::new(request_batch.iter().collect(), &census_schema);
            let problem_formulation_algo_hashmap: BlockConstraints<OptimalBudget> = algo_hash
                .compute_block_constraints(request_history.iter().collect(), &initial_budget);
            black_box(&problem_formulation_algo_hashmap);
        })
    });
}

fn segmentation_setup(
    eps: f64,
    delta: f64,
) -> (Schema, Vec<Request>, Vec<Request>, AccountingType) {
    let history_frac = 0.1;

    let census_schema = load_schema(
        resource_path(CENSUS_SCHEMA),
        &EpsDeltaDp { eps: 0., delta: 0. }.adp_to_rdp_budget(&ALPHAS),
    )
    .expect("Loading schema failed");

    // loads requests and converts them to internal format
    let census_requests = load_requests(
        resource_path(CENSUS_REQUESTS),
        &census_schema,
        &mut RequestAdapter::get_empty_adapter(),
        &None,
    )
    .expect("Loading requests failed");

    let n_total_requests = census_requests.len();
    let census_requests: BTreeMap<RequestId, Request> = census_requests
        .into_iter()
        .sorted_by(|(id1, _), (id2, _)| Ord::cmp(id1, id2))
        .take(n_total_requests / 7)
        .collect();

    let n_history_requests = (census_requests.len() as f64 * history_frac) as usize;
    let request_history: Vec<Request> = census_requests
        .iter()
        .take(n_history_requests)
        .map(|(_rid, request)| request.clone())
        .collect();
    let request_batch: Vec<Request> = census_requests
        .iter()
        .skip(n_history_requests)
        .map(|(_rid, request)| request.clone())
        .collect();

    let initial_adp_budget = EpsDeltaDp { eps, delta };

    let initial_budget = initial_adp_budget.adp_to_rdp_budget(&ALPHAS);
    (
        census_schema,
        request_history,
        request_batch,
        initial_budget,
    )
}

criterion_group!(segmentations, segmentations_outer);
criterion_main!(segmentations);
