# DP Planner

<a name="readme-top"></a>


<!-- ABOUT THE PROJECT -->
## About The Project


The main component of Cohere's architecture is the `DP Resource Planner`.
The planner takes as input a set of candidate requests alongside an inventory of available privacy resources represented as blocks.
Based on these, the planner determines an allocation while adhering to the configured privacy budget.
The planner supports experimentation with different allocation strategies and algorithms.


### Built With

* [![Rust][rust-shield]][rust-url]
* [![Cargo][cargo-shield]][cargo-url]
* [![Gurobi][gurobi-shield]][gurobi-url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

*  Installed `curl`, `git`, and `m4`:
    ```
    sudo apt-get install curl git m4
    ```

* [Rustup](https://rustup.rs/)
    ```sh
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

* Local clone of the repository (with submodules)
    ```sh
    git clone --recurse-submodules git@github.com:pps-lab/cohere.git
    ```

* Installed [Gurobi License](https://www.gurobi.com/features/academic-named-user-license/) and activated with `grbgetkey`.



### Installation


1. Install [Rust Version 1.65](https://blog.rust-lang.org/2022/11/03/Rust-1.65.0.html):
    ```sh
    rustup install 1.65 && rustup override set 1.65
    ```

2. Install [Gurobi Version 9.5.1](https://support.gurobi.com/hc/en-us/articles/4429974969105-Gurobi-9-5-1-released)

    For Ubuntu 22.04 LTS:
    ```sh
    curl -o gurobi951.tar.gz https://packages.gurobi.com/9.5/gurobi9.5.1_linux64.tar.gz
    ```

    ```sh
    sudo tar -xzf gurobi951.tar.gz -C /opt
    ```

    Gurobi should now be installed under: `/opt/gurobi951`

3. Set environment variables:

    ```sh
    # or to other location chosen in previous step
    export GUROBI_HOME=/opt/gurobi951/linux64

    export LD_LIBRARY_PATH=/opt/gurobi951/linux64/lib
    ```

    [!TIP]
    You can use [Direnv](https://direnv.net/) to create project-specific `.envrc` files that set environment variables when you enter the project directory.


4. Build the DP-Planner (from the root of the repository):
    ```sh
    cargo build --release
    ```



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

The `DP Planner` contains multiple binaries.
We focus on the main binary `dp_planner`as the other binaries are only auxiliary support tools.

The `dp_planner` can either run a single allocation round (i.e., `round` as the mode) or a simulation over multiple allocation rounds (i.e., `simulate` as the mode).
Moreover, the `dp_planner` offers a variety of configurations via additional options and subcommands.
There are subcommands for the choice of allocation algorithm (`<dpf | efficiency-based | greedy | ilp>`), whether the composition should consider partitioning attributes (`<block-composition | block-composition-pa>`), and for the budget configuration of each block (`<fix-budget | unlocking-budget>`).

The complete command has the following structure:
```sh
cargo run --release --bin dp_planner -- [BASE-OPTIONS]            # schema, requests, blocks, etc.
    <simulate | round> [MODE-OPTIONS]                             # single round or multiple rounds
    <dpf | efficiency-based | greedy | ilp> [ALGO-OPTIONS]        # allocation algorithm
    <block-composition-pa | block-composition> [COMP-OPTIONS]     # w/ PAs or w/o PAs
    <fix-budget | unlocking-budget> [BUDGET-OPTIONS]              # budget and unlocking
```

<details>
<summary>Show [BASE-OPTIONS]</summary>

```sh
BASE-OPTIONS:
    -A, --request-adapter <FILE>
            Sets the file which contains the request adapter (if not set, empty adapter is used)

    -B, --blocks <FILE>
            Existing blocks with request history

    -H, --history <FILE>
            Previously accepted requests

        --history-output-directory <FILE>
            Optionally define a directory where the generated history and blocks is saved. The files
            will have paths history_output_directory/block_history.json,
            history_output_directory/request_history.json and
            history_output_directory/remaining_requests.json

        --log-nonfinal-rejections
            Whether or not nonfinal rejections are logged

        --log-remaining-budget
            Whether or not the remaining budget is logged as part of the round log. Warning: This
            can be expensive, especially with a small batch size

    -R, --requests <FILE>
            Candidate requests for allocation

        --req-log-output <FILE>
            Sets the path for the log file, containing information about each request [default:
            ./results/requests.csv]

        --request-adapter-seed <REQUEST_ADAPTER_SEED>
            Sets the seed which is used for the request adapter

        --round-log-output <FILE>
            Sets the path for the log file, containing information about each round [default:
            ./results/rounds.csv]

        --runtime-log-output <FILE>
            Sets the path for the log file, containing information about the round runtime [default:
            ./results/runtime.csv]

    -S, --schema <FILE>
            Schema file of partitioning attributes

        --stats-output <FILE>
            Sets the path to the stats file, containing summary metrics of the current run [default:
            ./results/stats.json]

    -V, --version
            Print version information
```
</details>


<details>
<summary>Show [MODE-OPTIONS]</summary>

```sh
simulate [MODE-OPTIONS]

MODE-OPTIONS:
    -b, --batch-size <BATCH_SIZE>
            Request batch size per simulation round. Depending on the chosen allocation method, this
            has different effects:

    -m, --max-requests <MAX_REQUESTS>
            If set, this option limits how many requests can be processed. Useful to generate a
            history with some remaining requests

    -t, --timeout-rounds <TIMEOUT_ROUNDS>
            If keep_rejected_requests is set, this option limits long requests are kept. Set to a
            number higher than the number of rounds to keep all requests [default: 1]
```

```sh
round <I>

ARGS:
    <I>    Round number
```
</details>



<details>
<summary>Show [ALGO-OPTIONS]</summary>

Dominant Private Block Fairness allocation algorithm from the Privacy Budget Scheduling paper:
```sh
dpf [ALGO-OPTIONS]

ALGO-OPTIONS:
    --block-selector-seed <BLOCK_SELECTOR_SEED>
        The seed used in deciding which blocks are desired by each request [default: 42]

    --dominant-share-by-remaining-budget
        If set, the dpf (and weighted dpf) consider the remaining budget of the selected blocks
        to determine the dominant share. In the original Luo et al 2021 paper, the share is
        determined by the global budget. In "Packing Privacy Budget Efficiently" by Tholoniat et
        al 2022, the share is determined by the remaining budget of the selected blocks

    --weighted-dpf
        If set, the weighted dpf algorithm is used, which is a modification of the original dpf
        as described in "Packing Privacy Budget Efficiently" by Tholoniat et al

```

Any efficiency-based allocation algorithms (currently only Dpk) except dpf, for which a separate, optimized implementation exists:
```sh
efficiency-based [ALGO-OPTIONS] dpk [DPK-OPTIONS]

ALGO-OPTIONS:
    --block-selector-seed <BLOCK_SELECTOR_SEED>
        The seed used in deciding which blocks are desired by each request [default: 42]

DPK-OPTIONS:
    --eta <ETA>                    determines how close to the optimal solution the knapsack
                                    solver should be. Lower values result in better
                                    approximations, but also in longer runtimes. Should be
                                    between 0 and 0.75 (ends not included) [default: 0.05]
    --kp-solver <KP_SOLVER>        Which solver should be used to (approximately) solve Knapsack
                                    [default: fptas] [possible values: fptas, gurobi]
    --num-threads <NUM_THREADS>    How many parallel instances of
                                    [kp_solver](enum.EfficiencyBasedAlgo.html#variant.Dpk.field.kp_solver)
                                    should run in parallel at most at any time

```

Greedy allocation algorithm (prioritizes lower request id):
```sh
greedy [ALGO-OPTIONS]

ALGO-OPTIONS: -
```

Solve a profit optimization problem formulated as an integer linear program (ILP):
```sh
ilp [ALGO-OPTIONS]

ALGO-OPTIONS: -
```
</details>


<details>
<summary>Show [COMP-OPTIONS]</summary>

Without partitioning attributes:
```sh
block-composition [COMP-OPTIONS]

COMP-OPTIONS:
    -b, --budget-type <BUDGET_TYPE>    [default: optimal-budget] [possible values: optimal-budget,
                                       rdp-min-budget]
```


With partitioning attributes:
```sh
block-composition-pa [COMP-OPTIONS]

COMP-OPTIONS:
    -a, --algo <ALGO>                  The segmentation algo to split the request batch into
                                       segments and compute the remaining budget [default: narray]
                                       [possible values: narray, hashmap]
    -b, --budget-type <BUDGET_TYPE>    [default: optimal-budget] [possible values: optimal-budget,
                                       rdp-min-budget]
```

</details>


<details>
<summary>Show [BUDGET-OPTIONS]</summary>

 The complete budget is already unlocked in the first round:
```sh
fix-budget [BUDGET-OPTIONS]

BUDGET-OPTIONS:
# Budget Configuration:

    --budget-file <FILE>
        Read the budget from a file. The format is defined by how serde (de-)serializes an
        accounting type, and should be used only on files generated by this program earlier


    --epsilon <EPSILON>
        differential privacy epsilon budget (for DP and ADP)

    --delta <DELTA>
        differential privacy delta budget (for ADP)

    --alphas <ALPHAS>...
        converts epsilon, delta approximate differential privacy budget to renyi differential
        privacy budget, using the given alpha values. Only 1, 2, 3, 4, 5, 7, 10, 13, 14 or 15
        values are supported. See [AdpAccounting::adp_to_rdp_budget] for more details



    --rdp1 <RDP1>...
        renyi differential privacy with one alpha value

    --rdp2 <RDP2>...
        renyi differential privacy with two alpha values

    --rdp3 <RDP3>...
        renyi differential privacy with three alpha values

    --rdp4 <RDP4>...
        renyi differential privacy with four alpha values

    --rdp5 <RDP5>...
        renyi differential privacy with 5 alpha values

    --rdp7 <RDP7>...
        renyi differential privacy with 7 alpha values

    --rdp10 <RDP10>...
        renyi differential privacy with 10 alpha values

    --rdp13 <RDP13>...
        renyi differential privacy with 13 alpha values

    --rdp14 <RDP14>...
        renyi differential privacy with 14 alpha values

    --rdp15 <RDP15>...
        renyi differential privacy with 15 alpha values



# Configuration Options:

    --no-global-alpha-reduction
        If set to true, alpha values are not globally reduced. Note that this will not affect
        the history output, which always shows unreduced costs/budgets

    --convert-block-budgets
        If set to true, converts unlocked budgets of blocks from adp to rdp, same as the budget
        passed by the command line. See [AdpAccounting::adp_to_rdp_budget] for more details

    --convert-candidate-request-costs
        If set to true, converts cost of candidate requests from adp to rdp, by assuming the adp
        cost stems from the release of a result of a function with sensitivity one, to which
        gaussian noise was applied. See also [AdpAccounting::adp_to_rdp_cost_gaussian]. Uses the
        alpha values supplied by alphas field

    --convert-history-request-costs
        If set to true, converts cost of history requests from adp to rdp, by assuming the adp
        cost stems from the release of a result of a function with sensitivity one, to which
        gaussian noise was applied. See also [AdpAccounting::adp_to_rdp_cost_gaussian]. Uses the
        alpha values supplied by alphas field
```


The budget is gradually unlocked over time (i.e., requests in the first round cannot consume the complete budget):
```sh
unlocking-budget [BUDGET-OPTIONS]

BUDGET-OPTIONS:

# Budget Configuration:

    --budget-file <FILE>
        Read the budget from a file. The format is defined by how serde (de-)serializes an
        accounting type, and should be used only on files generated by this program earlier


    --epsilon <EPSILON>
        differential privacy epsilon budget (for DP and ADP)

    --delta <DELTA>
        differential privacy delta budget (for ADP)

    --alphas <ALPHAS>...
        converts epsilon, delta approximate differential privacy budget to renyi differential
        privacy budget, using the given alpha values. Only 1, 2, 3, 4, 5, 7, 10, 13, 14 or 15
        values are supported. See [AdpAccounting::adp_to_rdp_budget] for more details



    --rdp1 <RDP1>...
        renyi differential privacy with one alpha value

    --rdp2 <RDP2>...
        renyi differential privacy with two alpha values

    --rdp3 <RDP3>...
        renyi differential privacy with three alpha values

    --rdp4 <RDP4>...
        renyi differential privacy with four alpha values

    --rdp5 <RDP5>...
        renyi differential privacy with 5 alpha values

    --rdp7 <RDP7>...
        renyi differential privacy with 7 alpha values

    --rdp10 <RDP10>...
        renyi differential privacy with 10 alpha values

    --rdp13 <RDP13>...
        renyi differential privacy with 13 alpha values

    --rdp14 <RDP14>...
        renyi differential privacy with 14 alpha values

    --rdp15 <RDP15>...
        renyi differential privacy with 15 alpha values


# Budget Unlocking Configuration:

    --n-steps <N_STEPS>
        The total number of unlocking steps

    --slack <SLACK>
        The slack \in [0, 1] unlocks slightly more budget in the first n_steps/2 unlocking
        steps:  (1 + slack) * budget/n_steps and then (1 - slack) * budget/n_steps in the 2nd
        part of the unlocking steps. Currently, slack can only be used if the trigger is set to
        round (slack default = 0.0)

    --trigger <TRIGGER>
        The trigger of a budget unlocking step [possible values: round, request]


# Configuration Options:

    --no-global-alpha-reduction
        If set to true, alpha values are not globally reduced. Note that this will not affect
        the history output, which always shows unreduced costs/budgets

    --convert-block-budgets
        If set to true, converts unlocked budgets of blocks from adp to rdp, same as the budget
        passed by the command line. See [AdpAccounting::adp_to_rdp_budget] for more details

    --convert-candidate-request-costs
        If set to true, converts cost of candidate requests from adp to rdp, by assuming the adp
        cost stems from the release of a result of a function with sensitivity one, to which
        gaussian noise was applied. See also [AdpAccounting::adp_to_rdp_cost_gaussian]. Uses the
        alpha values supplied by alphas field

    --convert-history-request-costs
        If set to true, converts cost of history requests from adp to rdp, by assuming the adp
        cost stems from the release of a result of a function with sensitivity one, to which
        gaussian noise was applied. See also [AdpAccounting::adp_to_rdp_cost_gaussian]. Uses the
        alpha values supplied by alphas field

```

</details>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Example Command

The command below demonstrates how to invoke the `dp-planner`.
This example corresponds to the minimal example experiment design: [doe-suite-config/designs/minimal.yml](./../doe-suite-config/designs/minimal.yml).
The command executes a simulation comprising 40 rounds while employing the DPK algorithm.
Additionally, the `dp-planner` is configured to use partitioning attributes and budget unlocking.

(Please ensure this command is executed from within the dp-planner directory.)

```sh
cargo run --release --bin dp_planner --                             \
        --schema resources/applications/minimal/schema.json         \
        --requests resources/applications/minimal/requests.json     \
        --blocks resources/applications/minimal/blocks.json         \
        --req-log-output results/request_log.csv                    \
        --round-log-output results/round_log.csv                    \
        --runtime-log-output results/runtime_log.csv                \
        --stats-output results/stats.json                           \
                                                                    \
    simulate --timeout-rounds 1                                     \
                                                                    \
    efficiency-based                                                \
        --block-selector-seed 1000                                  \
        dpk                                                         \
            --eta 0.05                                              \
            --kp-solver gurobi                                      \
                                                                    \
    block-composition-pa                                            \
    unlocking-budget                                                \
        --trigger round                                             \
        --slack 0.4                                                 \
        --n-steps 12                                                \
        --epsilon 3.0                                               \
        --delta 1e-07                                               \
        --alphas 1.5 1.75 2 2.5 3 4 5 6 8 16 32 64 1e6 1e10
```


### Experiment Commands


Additional command examples are available from Cohere's evaluation ([configuration](../doe-suite-config)).
All the commands can be listed by executing the commands below from the root directory of the repository.
Please note that for these commands, it is necessary to update the file paths (schema, requests, blocks, ...) to match your local environment.



List all commands underlying Figure 5, which compares selecting a subset of active blocks compared to subsampling:
```sh
cmd-subsampling
```

List all commands underlying Figure 6, which compares different budget-unlocking strategies:
```sh
cmd-unlocking
```

List all commands underlying Figure 7, which compares different allocation algorithms and compositions:
```sh
cmd-comparison
```



<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Tests

The `DP-Planner` contains a test suite designed to assess individual components' functionality.

To execute the test suite:
```sh
cargo test --release
```

To execute an additional, more long-running test (~ 9 minutes):
```sh
cargo test --release -- --ignored
```


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->

[rust-shield]: https://img.shields.io/badge/rust-grey?style=for-the-badge&logo=rust
[rust-url]: https://www.rust-lang.org/


[cargo-shield]: https://img.shields.io/badge/cargo-grey?style=for-the-badge&logo=rust
[cargo-url]: https://doc.rust-lang.org/stable/cargo/


[gurobi-shield]: https://img.shields.io/badge/gurobi-grey?style=for-the-badge&logo=gurobi
[gurobi-url]: https://www.gurobi.com/
