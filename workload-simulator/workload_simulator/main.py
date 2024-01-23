

import time
import os
from datetime import timedelta
import argparse

import workload_simulator.request_generator.mechanism as mlib
from workload_simulator.request_generator import (calibration, mode,
                                                  partitioning, request,
                                                  sampling, utility)
from workload_simulator.request_generator.request import Category as Cat
from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import (Schema,
                                              create_single_attribute_schema)
from workload_simulator.simulation import Simulation, WorkloadVariationConfig

import workload_simulator.report

from multiprocessing import Pool
import tqdm
from copy import deepcopy


def all_mode_encoders(scenario):
    encoders = [
        mode.ActivePoissonSubsamplingModeEncoder(scenario=scenario),
        mode.ActiveBlockSubsetModeEncoder(user_group_size=5000, scenario=scenario),
    ]
    return encoders


def default_utility_assigners():

    ncd = utility.NormalizedCobbDouglasUtilityAssigner(
            privacy_cost_elasticity=2,
            data_elasticity=1,
            scaling_beta_a= 0.25, # with 0.25, the scaling factor is drawn from a "u-shape", so that most values are close to 0 or 1
            scaling_beta_b=0.25,
            use_balance_epsilon=True,
        )

    utility_assigners = [
        utility.ConstantUtilityShadowAssigner(shadow_assigner=ncd),
        ncd,
    ]
    return utility_assigners


def default_scenario():
    """
    Uses the Cohere schema, and runs a scenario with weekly allocations with an active user window of 12 weeks.
    In expectation, we have 786k active users in 3 months, and 504 requests per allocation in expectation (batch).
    We simulate 50 allocations.
    """

    schema = create_single_attribute_schema(domain_size=204800) # what cohere was using

    scenario = ScenarioConfig(
        name="40-1w-12w",
        allocation_interval=timedelta(weeks=1),
        active_time_window=timedelta(weeks=12), # ~3 months (quartal)
        user_expected_interarrival_time=timedelta(seconds=10), # 786k active users in 3 months
        request_expected_interarrival_time=timedelta(minutes=20), # resulting in 504 requests per week in expectation (batch)
        n_allocations=40,
        schema=schema,
    )

    return scenario


def lineargaussian_workload(schema: Schema, name="gm:GM-sub25100-defpa", request_type_dist= None, partition=None, subsampling=None):

    """Corresponds to (W1:GM) in the paper.
    """

    if partition is None:
        partition = partitioning.Defaults.new_highly_partitioned(schema)

    if subsampling is None:
        subsampling = sampling.Defaults.equal_p25_p100()

    if request_type_dist is None:
        cost = calibration.Defaults.mice_hare_elephant_v2()
    else:
        cost = calibration.Defaults.mice_hare_elephant_v2(**request_type_dist)

    workload_cfg = request.Workload(
        name,
        schema,
        [
            Cat(partition, mlib.GaussianMechanism(cost, subsampling), 1),
        ],
    )

    return workload_cfg


def basic_workload(schema: Schema, name="basic:GM-LM-RR-LSVT-sub25100-defpa", subsampling=None):

    """Corresponds to (W2:Mix) in the paper.
    """

    highly_partition = partitioning.Defaults.new_highly_partitioned(schema)
    no_partition = partitioning.Defaults.new_low_partitioned(schema)
    if subsampling is None:
        subsampling = sampling.Defaults.equal_p25_p100()
    cost = calibration.Defaults.mice_hare_elephant_v2()

    cost_cheap = calibration.Defaults.mice_hare_elephant_cheap()

    workload_cfg = request.Workload(
        name,
        schema,
        [
            Cat(highly_partition, mlib.GaussianMechanism(cost, subsampling), 1),
            Cat(highly_partition, mlib.LaplaceMechanism(cost_cheap, subsampling), 1),
            Cat(no_partition, mlib.SVTLaplaceMechanism(cost_cheap, subsampling), 1),
            Cat(no_partition, mlib.RandResponseMechanism(cost_cheap, subsampling), 1),
        ],
    )

    return workload_cfg


def ml_workload(schema: Schema, name="ml:SGD-PATE-sub25100-defpa", subsampling=None):

    """Corresponds to (W3:ML) in the paper.
    """

    norm_partition = partitioning.Defaults.new_normal_partitioned(schema)

    if subsampling is None:
        subsampling = sampling.Defaults.equal_p25_p100()
    cost = calibration.Defaults.mice_hare_elephant_v2()

    mechanism_distribution = [
        Cat(norm_partition, mlib.MLNoisySGDMechanism(cost, subsampling), 1,),
        Cat(norm_partition, mlib.MLPateGaussianMechanism(cost, subsampling), 1,),
    ]

    workload_cfg = request.Workload(name, schema, mechanism_distribution)

    return workload_cfg


def mixed_workload(schema: Schema, name="mixed:GM-LM-RR-LSVT-SGD-PATE-sub25100-defpa", subsampling=None):

    """Corresponds to (W4:All) in the paper.
    """


    highly_partition = partitioning.Defaults.new_highly_partitioned(schema)
    no_partition = partitioning.Defaults.new_low_partitioned(schema)
    norm_partition = partitioning.Defaults.new_normal_partitioned(schema)

    sub = sampling.Defaults.equal_p25_p100() if subsampling is None else subsampling

    cost = calibration.Defaults.mice_hare_elephant_v2()

    cheap_cost = calibration.Defaults.mice_hare_elephant_cheap()

    mechanism_distribution = [
        Cat(highly_partition, mlib.GaussianMechanism(cost, sub), 1),
        Cat(highly_partition, mlib.LaplaceMechanism(cheap_cost, sub), 1),
        Cat(no_partition,mlib.SVTLaplaceMechanism(cheap_cost, sub), 1),
        Cat(no_partition, mlib.RandResponseMechanism(cheap_cost, sub), 1),
        Cat(norm_partition,mlib.MLNoisySGDMechanism(cost, sub), 1),
        Cat(norm_partition, mlib.MLPateGaussianMechanism(cost, sub), 1),
    ]

    workload_cfg = request.Workload(
        name,
        schema,
        mechanism_distribution
    )
    return workload_cfg



def workload_simulation(output_dir, n_repetitions):
    scenario = default_scenario()

    workload_variations = WorkloadVariationConfig.product(
        default_utility_assigners(), all_mode_encoders(scenario)
    )

    workloads = [
        lineargaussian_workload(scenario.schema), # (W1:GM) in paper
        basic_workload(scenario.schema), # (W2:Mix) in paper
        ml_workload(scenario.schema), # (W3:ML) in paper
        mixed_workload(scenario.schema), # (W4:All) in paper
    ]

    simulation = Simulation(
            scenario=scenario,
            workloads=workloads,
            workload_variations=workload_variations,
            output_dir=output_dir,
        )

    run_in_parallel(simulation, n_repetitions=n_repetitions)



def effect_subsampling_simulation(output_dir, n_repetitions):
    scenario = default_scenario()

    # only maximize for the number of requests
    workload_variations = WorkloadVariationConfig.product(
        default_utility_assigners(), all_mode_encoders(scenario)
    )

    workloads = []
    for sub_prob in [0.05, 0.25, 0.45, 0.65, 0.85]:
        subsampling = sampling.Defaults.single(sub_prob)

        workloads_sub = [
            lineargaussian_workload(scenario.schema, f"gm:GM-sub{int(100*sub_prob)}-defpa", subsampling=subsampling),
            basic_workload(scenario.schema, f"basic:GM-LM-RR-LSVT-sub{int(100*sub_prob)}-defpa", subsampling=subsampling),
            ml_workload(scenario.schema, f"ml:SGD-PATE-sub{int(100*sub_prob)}-defpa", subsampling=subsampling),
            mixed_workload(scenario.schema, f"mixed:GM-LM-RR-LSVT-SGD-PATE-sub{int(100*sub_prob)}-defpa", subsampling=subsampling),
        ]
        workloads += workloads_sub

    simulation = Simulation(
            scenario=scenario,
            workloads=workloads,
            workload_variations=workload_variations,
            output_dir=output_dir,
            )

    run_in_parallel(simulation, n_repetitions=n_repetitions)


def simulation_runner(info):
    i, simulation = info
    simulation.run(repetition_start=i, n_repetitions=1)


def run_in_parallel(simulation, n_repetitions):

    lst = []
    for i in range(n_repetitions):
        copy = deepcopy(simulation)
        lst.append((i, copy))


    with Pool(processes=n_repetitions) as p:
        for _ in tqdm.tqdm(p.imap_unordered(simulation_runner, lst), total=len(lst)):
            pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create Workload')

    parser.add_argument('-o', '--output-dir', type=str, default='output', help='Output directory path')

    parser.add_argument('-n', '--n-repetition', type=int, default=5, help='Number of repetitions for each workload')

    args = parser.parse_args()
    output_dir = args.output_dir

    # create directory if not exists
    os.makedirs(output_dir, exist_ok=True)


    workload_simulation(output_dir, n_repetitions=args.n_repetition)

    # create request workloads to look at advantage of subsampling
    effect_subsampling_simulation(output_dir, n_repetitions=args.n_repetition)



    # create a workload report for each scenario
    for scn in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, scn)):
            workload_simulator.report.create_workload_report(os.path.join(output_dir, scn), slacks=[0.0], skip_pa_overlap=True)
