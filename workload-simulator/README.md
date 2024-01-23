# Workload Generator
<a name="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## About The Project

Benchmarking DP systems remains a major challenge and this difficulty is exacerbated when dealing with mixed workloads.
We provide a configurable DP workload generator that can accommodate various request types and workload characteristics.
The key to its flexibility lies in recognizing that any DP application can be represented as a combination of a small number of fundamental DP mechanisms.
For example, training an ML model with DP-SGD can be represented as a combination of Gaussian mechanisms.
By doing so, we are able to abstract a diverse range of real-world use cases into a more generalized setting.
We make our workload generator available as an open-source tool, as we believe better benchmarks for complex mixed DP workloads are of independent interest.


### Built With

* [![Python][python-shield]][python-url]
* [![Poetry][poetry-shield]][poetry-url]
* [![AutoDP][autodp-shield]][autodp-url]
* [![SimPy][simpy-shield]][simpy-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python Poetry](https://python-poetry.org/)
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

* [Make](https://www.gnu.org/software/make/)

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/cohere.git
    ```


### Installation


To re-create the four workloads from the evaluation section in the paper:
```sh
# takes ~170 mins
make create-workloads
```



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

In [workload-simulator/workload_simulator/main.py](workload_simulator/main.py), we provide the workload configurations used in Cohere's evaluation.
The function `default_scenario()` contains Cohere's evaluation scenario, while `lineargaussian_workload(..)`, `basic_workload(..)`, `ml_workload(..)`, and `mixed_workload(..)` represent the four distinct workload configurations.

In the following, we outline the individual components to construct additional workloads.



### Workload Configuration

At the core of the workload generator is the request distribution, which models diverse request families as a categorical distribution over request types.
Request types (see `Cat`) are parameterized with data and privacy requirements, allowing for granular customization and fine-grained management of workload dynamics.
Data requirements are sampled from a distribution over partitioning attributes (see e.g., `IntervalPopulationSelection`), defining a target population, and a categorical distribution over the percentage of that population to be requested (see `CategoricalSampling`).
For example, a request type might require either 25% or 50% of all active users selected by the `highly_partitioned` selection strategy with equal probability.
Privacy requirements specify a DP mechanism (e.g., Laplace Mechanism) with privacy costs sampled from a categorical distribution of possible costs (see `CategoricalCostCalibration`).
For example, a request type might specify the Gaussian mechanism and an equal split between low and high privacy costs.


```Python
# define schema of partitioning attributes
schema = create_single_attribute_schema(domain_size=204800)

# define target population (via selection of partitioning attributes)
highly_partitioned = IntervalPopulationSelection(schema, beta_a=1, beta_b=10)

# request type requires 25% or 100% of all active users with equal probability
subsampling = CategoricalSampling([
    {"name": "25%", "prob": 0.5, "fraction": 0.25},
    {"name": "100%", "prob": 0.5, "fraction": 1},
])

# request type has privacy costs calibrated to eps=0.2 or eps=0.75 with equal probability
cost = CategoricalCostCalibration([
    {"name": "cost-1", "prob": 0.5, "epsilon": 0.2, "delta": 1.0e-9},
    {"name": "cost-2", "prob": 0.5, "epsilon": 0.75, "delta": 1.0e-9},
])

# workload consists of an equal combination (weight 1) of the Gaussian and Laplace Mechanism
workload = request.Workload(
    name,
    schema,
    [
        Cat(highly_partitioned, mlib.GaussianMechanism(cost, subsampling), 1),
        Cat(highly_partitioned, mlib.LaplaceMechanism(cost, subsampling), 1),
    ],
)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Scenario Configuration


Alongside the workload, the generator expects a deployment scenario that defines aspects such as the frequency of allocation rounds, arrival rates of users and requests.

As each system expects a certain request format, we provide adapters (see `WorkloadVariationConfig`) designed to transform samples into concrete requests for specific budget management systems (e.g., Cohere or [PrivateKube][privatekube-url]).
In addition, these workload variations can also be used to choose different allocation objectives.

Based on these configurations, the generator employs a discrete event simulator to produce random instances of workloads aligned with a specified deployment scenario.

```Python

# define a scenario
scenario = ScenarioConfig(
    name="40-1w-12w",
    allocation_interval=timedelta(weeks=1),
    active_time_window=timedelta(weeks=12), # ~3 months (quartal)
    user_expected_interarrival_time=timedelta(seconds=10), # 786k active users in 3 months
    request_expected_interarrival_time=timedelta(minutes=20), # resulting in 504 requests per week in expectation (batch)
    n_allocations=40,
    schema=schema,
)


# create workload variants with different utilities per request, and formatted for different systems (Cohere / PrivateKube)
workload_variations = WorkloadVariationConfig.product(
    default_utility_assigners(), all_mode_encoders(scenario)
)


# generate arriving users / requests according to the configuration, and create a json with all requests
simulation = Simulation(
        scenario=scenario,
        workloads=[workload],
        workload_variations=workload_variations,
        output_dir=output_dir,
)
run_in_parallel(simulation, n_repetitions=5)

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Workload Generator Output


The workload simulator generates the following output files:

- `schema.json`: Defines the schema of partitioning attributes and the accounting configuration. Its format corresponds to the expected structure for the [resource planner](./../dp-planner) in the `--schema` argument.

- `blocks_<rep>.json`:  Here, `<rep>` is an integer representing the repetition.
This file contains the stream of arriving blocks (i.e., groups in Cohere).
The format matches the expected structure by the [resource planner](./../dp-planner) for the `--blocks` argument.

- `requests_<rep>.json`: Similarly,  `<rep>` is an integer corresponding to the repetition.
This file contains the set of candidate requests for each round.
The format of these requests aligns with the expectations of the [resource planner](./../dp-planner) in the `--requests` argument.


#### Example Schema

```YAML
# schema.json
{
    # a single partitioning attributes with a domain of size 204800
   "attributes":[
      {
         "name":"attr0",
         "value_domain":{"min":0, "max":204799}
      }
   ],
   "accounting_type":{
      "Rdp":{  #  rdp accounting with 14 orders
         "eps_values":[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      }
   }
}
```


#### Example Blocks

```YAML
# blocks.json
[
    {
    "id": 0, # block id
    "unlocked_budget":{
        "Rdp":{ # initially no budget is unlocked
            "eps_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
    },
    "n_users": 60037, # info how many users are in this block
    "created": 0,  # round id in which block is added
    "retired": 12, # round id in which block is retired
    "request_ids":[] # no allocation history in the past
    },
    ...
]


```


#### Requests

```YAML
# requests.json
[
    {
        "request_id": 0,    # unique id
        "created": 11,      # request round
        "profit": 1,        # utility if request accepted
        "n_users": 12,      # called n_users for legacy reasons, it's the number of active groups requested

        # selecting a population with PA
        "dnf": {"conjunctions": [{"predicates": {"attr0": {"Between": {"min": 119529, "max": 119763}}}}]},

        # rdp cost vector
        "request_cost": {"Rdp": {"eps_values": [...]}}

        # additional information about the request
        "request_info": {
            "mechanism": {"mechanism": {"name": "GaussianMechanism", "sigma": 7.235376900721338, "l2_sensitivity": 1}},
            "sampling_info": {"name": "5%", "prob": 0.05},
            "cost_original": {"name": "elephant", "adp": {"EpsDeltaDp": {"eps": 0.75, "delta": 1e-09}}, "rdp": {"Rdp": {"eps_values": [...]}}},
            ...
        },

        # additional information about the workload
        "workload_info": {...},
    },
    ...
]

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->

[python-shield]: https://img.shields.io/badge/python-grey?style=for-the-badge&logo=python
[python-url]: https://www.python.org/

[poetry-shield]: https://img.shields.io/badge/poetry-grey?style=for-the-badge&logo=poetry
[poetry-url]: https://python-poetry.org/


[autodp-shield]: https://img.shields.io/badge/autodp-grey?style=for-the-badge&logo=github
[autodp-url]: https://github.com/yuxiangw/autodp


[simpy-shield]: https://img.shields.io/badge/simpy-grey?style=for-the-badge&logo=pypi
[simpy-url]: https://simpy.readthedocs.io/en/latest/


[privatekube-url]: https://arxiv.org/abs/2106.15335
