use crate::dprivacy::rdp_alphas_accounting::{PubRdpAccounting, RdpAlphas};
use crate::dprivacy::{AccountingType, AdpAccounting};
use clap::{Parser, Subcommand};
use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

#[derive(Parser, Debug, Clone)]
#[clap(author, version, about, long_about = None)]

pub struct Cli {
    #[clap(subcommand)]
    pub mode: Mode,

    #[clap(flatten)]
    pub input: Input,

    #[clap(flatten)]
    pub output_config: OutputConfig,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Mode {
    Simulate {
        #[clap(subcommand)]
        allocation: AllocationConfig,

        /// Request batch size per simulation round. Depending on the chosen allocation method, this
        /// has different effects:
        ///
        /// * For ilp allocation, this directly controls how big the batches are and with this,
        /// which allocations the optimizer can find. This also influences the logs, as the number
        /// of rounds is ceil(#requests / batch_size)
        /// * For greedy allocation, this enables some computational optimizations which allow for
        /// faster computation of the final greedy allocation. Greedy allocation is an online
        /// allocation method, and batch_size > 1 does not change this, the result will be the same,
        /// it will just be calculated a lot faster, and the logs may be missing some statistics
        /// which would otherwise be available. The number of rounds in the logs is always equal to
        /// #requests for greedy.
        /// * For dpf, setting the batch size to anything else but one will throw an error.
        ///
        /// In case we use the created field of the requests, we have variable sized batches, and thus cannot set a fixed batch size
        #[clap(long, short)]
        batch_size: Option<usize>,

        /// If keep_rejected_requests is set, this option limits long requests are kept. Set to a
        /// number higher than the number of rounds to keep all requests.
        #[clap(long, short, default_value("1"))]
        timeout_rounds: usize,

        /// If set, this option limits how many requests can be processed. Useful to generate a
        /// history with some remaining requests.
        #[clap(long, short)]
        max_requests: Option<usize>,
    },
    Round {
        #[clap(subcommand)]
        allocation: AllocationConfig,

        /// Round number
        i: usize,
    },
}

#[derive(Subcommand, Debug, Clone)]
pub enum AllocationConfig {
    /// Greedy allocation algorithm (prioritizes lower request id).
    Greedy {
        #[clap(subcommand)]
        composition: CompositionConfig,
    },

    /// Solve a profit optimization problem formulated as an integer linear program (ilp).
    Ilp {
        #[clap(subcommand)]
        composition: CompositionConfig,
    },

    /// Dominant Private Block Fairness allocation algorithm from the Privacy Budget Scheduling paper.
    Dpf {
        /// The seed used in deciding which blocks are desired by each request
        #[clap(long, default_value("42"))]
        block_selector_seed: u64,

        /// If set, the weighted dpf algorithm is used, which is a modification of the original dpf
        /// as described in "Packing Privacy Budget Efficiently" by Tholoniat et al
        #[clap(long)]
        weighted_dpf: bool,

        /// If set, the dpf (and weighted dpf) consider the remaining budget of the selected blocks to determine the dominant share.
        /// In the original Luo et al 2021 paper, the share is determined by the global budget.
        /// In "Packing Privacy Budget Efficiently" by Tholoniat et al 2022, the share is determined by the remaining budget of the selected blocks.
        #[clap(long)]
        dominant_share_by_remaining_budget: bool,

        #[clap(subcommand)]
        composition: CompositionConfig,
    },

    /// Any efficiency-based allocation algorithms (currently only Dpk) except dpf, for which a
    /// separate, optimized implementation exists.
    EfficiencyBased {
        /// The type of efficiency-based algorithm to use
        #[clap(subcommand)]
        algo_type: EfficiencyBasedAlgo,

        /// The seed used in deciding which blocks are desired by each request
        #[clap(long, default_value("42"))]
        block_selector_seed: u64,
    },
}

/// The type of efficiency-based algorithm to use.
///
/// Add any new efficiency-based algos here, and fix any compiler errors with match clauses to allow
/// access to the new algo via the CLI.
#[derive(Subcommand, Debug, Clone)]
pub enum EfficiencyBasedAlgo {
    /// use Dpk
    Dpk {
        /// determines how close to the optimal solution the knapsack solver should be. Lower values
        /// result in better approximations, but also in longer runtimes. Should be between 0 and 0.75
        /// (ends not included).
        #[clap(long, default_value("0.05"))]
        eta: f64,

        /// Which solver should be used to (approximately) solve Knapsack.
        #[clap(long, arg_enum, default_value("fptas"))]
        kp_solver: KPSolverType,

        /// How many parallel instances of
        /// [kp_solver](enum.EfficiencyBasedAlgo.html#variant.Dpk.field.kp_solver) should run in
        /// parallel at most at any time
        #[clap(long)]
        num_threads: Option<usize>,

        #[clap(subcommand)]
        composition: CompositionConfig,
    },
}

impl EfficiencyBasedAlgo {
    pub fn get_composition(&self) -> &CompositionConfig {
        match self {
            Self::Dpk { composition, .. } => composition,
        }
    }
}

/// Which solver should be used to (approximately) solve Knapsack.
/// See [allocation::efficiency_based::knapsack::KPApproxSolver] for more details.
#[derive(clap::ArgEnum, Debug, Clone, Copy)]
pub enum KPSolverType {
    FPTAS,
    Gurobi,
}

#[derive(clap::ArgEnum, Debug, Clone)]
pub enum BudgetType {
    /// use OptimalBudget
    OptimalBudget,
    /// use RdpMinBudget
    RdpMinBudget,
}

#[derive(Subcommand, Debug, Clone)]
pub enum CompositionConfig {
    /// Block composition with Partitioning Attributes.
    BlockCompositionPa {
        #[clap(subcommand)]
        budget: Budget,

        /// The segmentation algo to split the request batch into segments and compute the remaining budget.
        #[clap(short, long, arg_enum, default_value("narray"))]
        algo: SegmentationAlgo,

        #[clap(short, long, arg_enum, default_value("optimal-budget"))]
        budget_type: BudgetType,
    },
    /// Regular block composition (without partitioning attributes)
    BlockComposition {
        #[clap(subcommand)]
        budget: Budget,

        #[clap(short, long, arg_enum, default_value("optimal-budget"))]
        budget_type: BudgetType,
    },
}

#[derive(clap::ArgEnum, Debug, Clone, Copy)]
pub enum SegmentationAlgo {
    Narray,
    Hashmap,
}

#[derive(Subcommand, Debug, Clone)]
pub enum Budget {
    /// The complete budget is already unlocked in the first round.
    FixBudget {
        #[clap(flatten)]
        budget: BudgetTotal,
    },

    /// The budget is gradually unlocked over time (i.e., requests in the first round cannot
    /// consume the complete budget). Note that for both greedy and dpf, selecting
    /// this option will throw an error if the
    /// [batch_size](enum.Mode.html#variant.Simulate.field.batch_size) is > 1.
    /// Therefore, for both greedy and dpf, the budget unlocked per round is total_budget / n_steps,
    /// regardless of if the trigger is set to round or request.
    /// For ilp, it works as expected: in each round, the amount of budget unlocked is either
    /// total_budget / n_steps if the trigger is round, or batch_size / n_steps if the trigger is
    /// request.
    UnlockingBudget {
        /// The trigger of a budget unlocking step.
        #[clap(short, long, arg_enum)]
        trigger: UnlockingBudgetTrigger,

        // would require additional state on block
        //#[clap(short, long)]
        //every: usize,
        /// The total number of unlocking steps.
        #[clap(short, long)]
        n_steps: usize,

        /// The slack \in [0, 1] unlocks slightly more budget in the first n_steps/2 unlocking steps:  (1 + slack) * budget/n_steps
        /// and then (1 - slack) * budget/n_steps in the 2nd part of the unlocking steps.
        /// Currently, slack can only be used if the trigger is set to round (slack default = 0.0).
        #[clap(short, long)]
        slack: Option<f64>,

        /// The total amount of budget available over all unlocking steps.
        #[clap(flatten)]
        budget: BudgetTotal,
    },
}

static EXCLUSIVE_BUDGET_OPTIONS: [&str; 11] = [
    "budget-file",
    "rdp1",
    "rdp2",
    "rdp3",
    "rdp4",
    "rdp5",
    "rdp7",
    "rdp10",
    "rdp13",
    "rdp14",
    "rdp15",
];

#[derive(clap::Args, Debug, Clone)]
pub struct BudgetTotal {
    /// Read the budget from a file. The format is defined by how serde (de-)serializes an
    /// accounting type, and should be used only on files generated by this program earlier.
    #[clap(long, exclusive(true), parse(from_os_str), value_name = "FILE")]
    pub(crate) budget_file: Option<PathBuf>,

    /// differential privacy epsilon budget (for DP and ADP)
    #[clap(
        long,
        required_unless_present_any(&EXCLUSIVE_BUDGET_OPTIONS)
    )]
    pub(crate) epsilon: Option<f64>,

    /// differential privacy delta budget (for ADP)
    #[clap(long, requires("epsilon"))]
    pub(crate) delta: Option<f64>,

    /// renyi differential privacy with one alpha value.
    #[clap(long, exclusive(true), min_values(1), max_values(1))]
    pub(crate) rdp1: Option<Vec<f64>>,

    /// renyi differential privacy with two alpha values.
    #[clap(long, exclusive(true), min_values(2), max_values(2))]
    pub(crate) rdp2: Option<Vec<f64>>,

    /// renyi differential privacy with three alpha values.
    #[clap(long, exclusive(true), min_values(3), max_values(3))]
    pub(crate) rdp3: Option<Vec<f64>>,

    /// renyi differential privacy with four alpha values.
    #[clap(long, exclusive(true), min_values(4), max_values(4))]
    pub(crate) rdp4: Option<Vec<f64>>,

    /// renyi differential privacy with 5 alpha values.
    #[clap(long, exclusive(true), min_values(5), max_values(5))]
    pub(crate) rdp5: Option<Vec<f64>>,

    /// renyi differential privacy with 7 alpha values.
    #[clap(long, exclusive(true), min_values(7), max_values(7))]
    pub(crate) rdp7: Option<Vec<f64>>,

    /// renyi differential privacy with 10 alpha values.
    #[clap(long, exclusive(true), min_values(10), max_values(10))]
    pub(crate) rdp10: Option<Vec<f64>>,

    /// renyi differential privacy with 13 alpha values.
    #[clap(long, exclusive(true), min_values(13), max_values(13))]
    pub(crate) rdp13: Option<Vec<f64>>,

    /// renyi differential privacy with 14 alpha values.
    #[clap(long, exclusive(true), min_values(14), max_values(14))]
    pub(crate) rdp14: Option<Vec<f64>>,

    /// renyi differential privacy with 15 alpha values.
    #[clap(long, exclusive(true), min_values(15), max_values(15))]
    pub(crate) rdp15: Option<Vec<f64>>,

    /// converts epsilon, delta approximate differential privacy budget to renyi differential privacy
    /// budget, using the given alpha values. Only 1, 2, 3, 4, 5, 7, 10, 13, 14 or 15 values are supported.
    /// See [AdpAccounting::adp_to_rdp_budget] for more details
    #[clap(long, requires_all(&["epsilon", "delta"]), conflicts_with_all(&EXCLUSIVE_BUDGET_OPTIONS), min_values(1), max_values(15))]
    pub(crate) alphas: Option<Vec<f64>>,

    /// If set to true, alpha values are not globally reduced. Note that this will not affect the
    /// history output, which always shows unreduced costs/budgets.
    #[clap(long)]
    pub(crate) no_global_alpha_reduction: bool,

    // TODO [later] this flag with the conversion of request costs should be part of the request cost adapter -> (the conversion assumes a gaussian mechanism with sensitivity 1)
    /// If set to true, converts cost of candidate requests from adp to rdp, by assuming the adp cost
    /// stems from the release of a result of a function with sensitivity one, to which gaussian
    /// noise was applied. See also [AdpAccounting::adp_to_rdp_cost_gaussian]. Uses the alpha values
    /// supplied by alphas field
    #[clap(long, requires("alphas"))]
    pub(crate) convert_candidate_request_costs: bool,

    /// If set to true, converts cost of history requests from adp to rdp, by assuming the adp cost
    /// stems from the release of a result of a function with sensitivity one, to which gaussian
    /// noise was applied. See also [AdpAccounting::adp_to_rdp_cost_gaussian]. Uses the alpha values
    /// supplied by alphas field
    #[clap(long, requires("alphas"), conflicts_with_all(&["rdp5", "rdp7", "rdp10", "rdp13", "rdp14", "rdp15"]))]
    pub(crate) convert_history_request_costs: bool,

    // TODO [later] for converting blocks + history need a better solution
    /// If set to true, converts unlocked budgets of blocks from adp to rdp, same as the budget passed
    /// by the command line. See [AdpAccounting::adp_to_rdp_budget] for more details
    #[clap(long, requires("alphas"))]
    pub(crate) convert_block_budgets: bool,
}

impl Budget {
    pub fn budget(&self) -> AccountingType {
        match self {
            Budget::FixBudget { budget } => budget.budget(),
            Budget::UnlockingBudget { budget, .. } => budget.budget(),
        }
    }

    /// Returns whether or not the budget is unlocking or not
    pub fn unlocking_budget(&self) -> bool {
        match self {
            Budget::FixBudget { .. } => false,
            Budget::UnlockingBudget { .. } => true,
        }
    }
}

impl BudgetTotal {
    /// Returns whether or not global alpha reduction is enabled
    pub fn global_alpha_reduction(&self) -> bool {
        !self.no_global_alpha_reduction
    }

    /// Returns the alphas passed to the Cli if they were passed and block budgets should be
    /// converted
    pub fn block_conversion_alphas(&self) -> Option<RdpAlphas> {
        if let BudgetTotal {
            convert_block_budgets: true,
            ..
        } = self
        {
            self.alphas()
        } else {
            None
        }
    }

    /// Returns the alphas passed to the Cli if they were passed and candidate request costs
    /// should be converted
    pub fn candidate_request_conversion_alphas(&self) -> Option<RdpAlphas> {
        if let BudgetTotal {
            convert_candidate_request_costs: true,
            ..
        } = self
        {
            self.alphas()
        } else {
            None
        }
    }

    /// Returns the alphas passed to the Cli if they were passed and history request costs should be
    /// converted
    pub fn history_request_conversion_alphas(&self) -> Option<RdpAlphas> {
        if let BudgetTotal {
            convert_history_request_costs: true,
            ..
        } = self
        {
            self.alphas()
        } else {
            None
        }
    }

    /// Returns the alphas passed to the Cli if they were passed
    pub fn alphas(&self) -> Option<RdpAlphas> {
        if let BudgetTotal {
            alphas: Some(alpha_vals),
            ..
        } = self
        {
            Some(RdpAlphas::from_vec(
                alpha_vals.clone()).expect(
                "Supplied an unsupported number of alpha values (supported: 1, 2, 3, 4, 5, 7, 10, 13, 14, 15)"
            )
            )
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub(crate) fn unconverted_budget(&self) -> AccountingType {
        match self {
            BudgetTotal {
                epsilon: Some(epsilon),
                delta: None,
                ..
            } => AccountingType::EpsDp { eps: *epsilon },
            BudgetTotal {
                epsilon: Some(epsilon),
                delta: Some(delta),
                ..
            } => AccountingType::EpsDeltaDp {
                eps: *epsilon,
                delta: *delta,
            },
            BudgetTotal {
                rdp1: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A1(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp2: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A2(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp3: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A3(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp4: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A4(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp5: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A5(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp7: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A7(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp10: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A10(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp13: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A13(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp14: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A14(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                rdp15: Some(rdp), ..
            } => AccountingType::Rdp {
                eps_values: RdpAlphas::A15(rdp.clone().try_into().unwrap()),
            },
            BudgetTotal {
                budget_file: Some(path_buf),
                ..
            } => {
                let file = File::open(path_buf)
                    .unwrap_or_else(|_| panic!("Couldn't open file {:?}", path_buf));
                let reader = BufReader::new(file);
                serde_json::from_reader(reader).expect("Couldn't deserialize budget")
            }
            _ => unreachable!("Did not enter a budget or at least not a budget that is supported"),
        }
    }

    pub fn budget(&self) -> AccountingType {
        // first, check if conversion to rdp is needed, and if yes, get alphas as AccountingType
        let alphas = self.alphas();

        let unconverted = self.unconverted_budget();

        if let Some(rdp_alphas) = alphas {
            unconverted.adp_to_rdp_budget(&rdp_alphas)
        } else {
            unconverted
        }
    }
}

#[derive(clap::ArgEnum, Debug, Clone)]
pub enum UnlockingBudgetTrigger {
    Round,
    Request,
}

#[derive(clap::Args, Debug, Clone)]
pub struct Input {
    /// Schema file of partitioning attributes
    #[clap(short = 'S', long, parse(from_os_str), value_name = "FILE")]
    pub schema: PathBuf,

    /// Existing blocks with request history
    #[clap(short = 'B', long, parse(from_os_str), value_name = "FILE")]
    pub blocks: PathBuf,

    /// Candidate requests for allocation
    #[clap(short = 'R', long, parse(from_os_str), value_name = "FILE")]
    pub requests: PathBuf,

    /// Config for request adapter to set request cost, block demand, and profit.
    #[clap(flatten)]
    pub request_adapter_config: RequestAdapterConfig,

    /// Previously accepted requests
    #[clap(short = 'H', long, parse(from_os_str), value_name = "FILE")]
    pub history: Option<PathBuf>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct RequestAdapterConfig {
    /// Sets the file which contains the request adapter (if not set, empty adapter is used)
    #[clap(short = 'A', long, parse(from_os_str), value_name = "FILE")]
    pub request_adapter: Option<PathBuf>,

    /// Sets the seed which is used for the request adapter
    #[clap(long, requires("request-adapter"))]
    pub request_adapter_seed: Option<u128>,
}

#[derive(clap::Args, Debug, Clone)]
pub struct OutputConfig {
    /// Sets the path for the log file, containing information about each request
    #[clap(
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value("./results/requests.csv")
    )]
    pub req_log_output: PathBuf,

    /// Sets the path for the log file, containing information about each round
    #[clap(
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value("./results/rounds.csv")
    )]
    pub round_log_output: PathBuf,

    /// Sets the path for the log file, containing information about the round runtime
    #[clap(
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value("./results/runtime.csv")
    )]
    pub runtime_log_output: PathBuf,

    /// Whether or not the remaining budget is logged as part of the round log. Warning: This can
    /// be expensive, especially with a small batch size.
    #[clap(long)]
    pub log_remaining_budget: bool,

    /// Whether or not nonfinal rejections are logged
    #[clap(long)]
    pub log_nonfinal_rejections: bool,

    /// Sets the path to the stats file, containing summary metrics of the current run
    #[clap(
        long,
        parse(from_os_str),
        value_name = "FILE",
        default_value("./results/stats.json")
    )]
    pub stats_output: PathBuf,

    /// Optionally define a directory where the generated history and blocks is saved. The files
    /// will have paths history_output_directory/block_history.json,
    /// history_output_directory/request_history.json and
    /// history_output_directory/remaining_requests.json
    #[clap(long, parse(from_os_str), value_name = "FILE")]
    pub history_output_directory: Option<PathBuf>,
}

// provide top-level access for different parts of the config
impl Cli {
    pub fn total_budget(&self) -> &BudgetTotal {
        self.mode.budget()
    }

    pub(crate) fn budget_total_mut(&mut self) -> &mut BudgetTotal {
        self.mode.budget_total_mut()
    }
    /*
    pub(crate) fn composition(&self) -> &CompositionConfig {
        self.mode.composition()
    }
     */

    pub fn allocation(&self) -> &AllocationConfig {
        self.mode.allocation()
    }

    /// Get the current batch size (returns None if the [mode](Mode) is round)
    pub fn batch_size(&self) -> Option<usize> {
        match self.mode {
            Mode::Simulate { batch_size, .. } => batch_size,
            Mode::Round { .. } => None,
        }
    }

    /// Checks that the combination of selected features is currently supported (where this is not
    /// already handled via [attributes](https://doc.rust-lang.org/reference/attributes.html))
    pub fn check_config(&self) {
        // check that the mode is simulate - round is not currently supported
        match self.mode {
            Mode::Simulate { timeout_rounds, .. } => {
                assert!(
                    0 < timeout_rounds,
                    "Timeout rounds must be strictly positive"
                );
            }
            Mode::Round { .. } => {
                panic!("Round mode is not currently supported, use simulate instead")
            }
        }

        if let Mode::Simulate { timeout_rounds, .. } = self.mode {
            // Due to numeric issues, ilp might allocate a request that cannot be allocated
            // by the problem formulation. To prevent ilp from allocating the same request in the
            // next rounds and effectively blocking progress, timeout_rounds must be set to 1.
            assert!(
                !(self.allocation().is_ilp() && timeout_rounds > 1),
                "ILP allocation is currently not supported with timeout_rounds > 1"
            )
        }
    }
}

impl Mode {
    fn budget(&self) -> &BudgetTotal {
        match self {
            Mode::Simulate { allocation, .. } => allocation.budget(),
            Mode::Round { allocation, .. } => allocation.budget(),
        }
    }

    pub(crate) fn budget_total_mut(&mut self) -> &mut BudgetTotal {
        match self {
            Mode::Simulate { allocation, .. } => allocation.budget_total_mut(),
            Mode::Round { allocation, .. } => allocation.budget_total_mut(),
        }
    }

    /*
    fn composition(&self) -> &CompositionConfig {
        match self {
            Mode::Simulate { allocation, .. } => allocation.composition(),
            Mode::Round { allocation, .. } => allocation.composition(),
        }
    }
     */

    fn allocation(&self) -> &AllocationConfig {
        match self {
            Mode::Simulate { allocation, .. } => allocation,
            Mode::Round { allocation, .. } => allocation,
        }
    }
}

impl EfficiencyBasedAlgo {
    pub(crate) fn budget(&self) -> &BudgetTotal {
        match self {
            EfficiencyBasedAlgo::Dpk { composition, .. } => composition.budget(),
        }
    }

    pub(crate) fn budget_total_mut(&mut self) -> &mut BudgetTotal {
        match self {
            EfficiencyBasedAlgo::Dpk { composition, .. } => composition.budget_total_mut(),
        }
    }

    pub fn budget_config(&self) -> &Budget {
        match self {
            EfficiencyBasedAlgo::Dpk { composition, .. } => composition.budget_config(),
        }
    }
}

impl AllocationConfig {
    pub fn is_greedy(&self) -> bool {
        matches!(self, AllocationConfig::Greedy { .. })
    }

    pub fn is_dpf(&self) -> bool {
        matches!(self, AllocationConfig::Dpf { .. })
    }

    pub fn is_ilp(&self) -> bool {
        matches!(self, AllocationConfig::Ilp { .. })
    }

    pub(crate) fn budget(&self) -> &BudgetTotal {
        match self {
            AllocationConfig::Dpf {
                block_selector_seed: _,
                composition,
                ..
            } => composition.budget(),
            AllocationConfig::Greedy { composition } => composition.budget(),
            AllocationConfig::Ilp { composition } => composition.budget(),
            AllocationConfig::EfficiencyBased { algo_type, .. } => algo_type.budget(),
        }
    }

    pub(crate) fn budget_total_mut(&mut self) -> &mut BudgetTotal {
        match self {
            AllocationConfig::Dpf {
                block_selector_seed: _,
                composition,
                ..
            } => composition.budget_total_mut(),
            AllocationConfig::Greedy { composition } => composition.budget_total_mut(),
            AllocationConfig::Ilp { composition } => composition.budget_total_mut(),
            AllocationConfig::EfficiencyBased { algo_type, .. } => algo_type.budget_total_mut(),
        }
    }

    pub fn budget_config(&self) -> &Budget {
        match self {
            AllocationConfig::Dpf {
                block_selector_seed: _,
                composition,
                ..
            } => composition.budget_config(),
            AllocationConfig::Greedy { composition } => composition.budget_config(),
            AllocationConfig::Ilp { composition } => composition.budget_config(),
            AllocationConfig::EfficiencyBased { algo_type, .. } => algo_type.budget_config(),
        }
    }

    /*
    fn composition(&self) -> &CompositionConfig {
        match self {
            AllocationConfig::Dpf {
                block_selector_seed: _,
                composition,
            } => composition,
            AllocationConfig::Greedy { composition } => composition,
            AllocationConfig::Ilp { composition } => composition,
        }
    }
     */
}

impl CompositionConfig {
    pub(crate) fn budget(&self) -> &BudgetTotal {
        let temp = match self {
            CompositionConfig::BlockCompositionPa { budget, .. } => budget,
            CompositionConfig::BlockComposition { budget, .. } => budget,
        };
        match temp {
            Budget::FixBudget { budget } => budget,
            Budget::UnlockingBudget { budget, .. } => budget,
        }
    }

    pub(crate) fn budget_total_mut(&mut self) -> &mut BudgetTotal {
        let temp = match self {
            CompositionConfig::BlockCompositionPa { budget, .. } => budget,
            CompositionConfig::BlockComposition { budget, .. } => budget,
        };
        match temp {
            Budget::FixBudget { budget } => budget,
            Budget::UnlockingBudget { budget, .. } => budget,
        }
    }

    pub(crate) fn budget_config(&self) -> &Budget {
        match self {
            CompositionConfig::BlockCompositionPa { budget, .. } => budget,
            CompositionConfig::BlockComposition { budget, .. } => budget,
        }
    }
}

pub fn check_output_paths(config: &Cli) -> OutputPaths {
    let req_log_output_path = config.output_config.req_log_output.clone();
    {
        // did not supply empty output_path
        let mut copy = req_log_output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .csv (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            req_log_output_path.extension().and_then(OsStr::to_str),
            Some("csv"),
            "output file needs to have \".csv\" extension (no capital letters)"
        );
    }

    let round_log_output_path = config.output_config.round_log_output.clone();
    {
        // did not supply empty output_path
        let mut copy = round_log_output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .csv (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            round_log_output_path.extension().and_then(OsStr::to_str),
            Some("csv"),
            "output file needs to have \".csv\" extension (no capital letters)"
        );
    }

    let runtime_log_output_path = config.output_config.runtime_log_output.clone();
    {
        // did not supply empty output_path
        let mut copy = runtime_log_output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .csv (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            runtime_log_output_path.extension().and_then(OsStr::to_str),
            Some("csv"),
            "output file needs to have \".csv\" extension (no capital letters)"
        );
    }

    let stats_output_path = config.output_config.stats_output.clone();
    {
        // did not supply empty output_path
        let mut copy = stats_output_path.clone();
        assert!(copy.pop(), "Empty output path was supplied");
        // all parent directories exist
        assert!(
            copy.exists(),
            "A directory on the supplied output path either does not exist or is inaccessible"
        );
        // check that file ends in .csv (via
        // https://stackoverflow.com/questions/45291832/extracting-a-file-extension-from-a-given-path-in-rust-idiomatically)
        assert_eq!(
            stats_output_path.extension().and_then(OsStr::to_str),
            Some("json"),
            "output file needs to have \".json\" extension (no capital letters)"
        );
    }

    let history_output_directory_path = config.output_config.history_output_directory.clone();
    if let Some(history_path) = history_output_directory_path.as_ref() {
        // check that the given path is indeed pointing at a directory
        assert!(history_path.is_dir());
    }

    OutputPaths {
        req_log_output_path,
        round_log_output_path,
        runtime_log_output_path,
        stats_output_path,
        history_output_directory_path,
    }
}

pub struct OutputPaths {
    pub req_log_output_path: PathBuf,
    pub round_log_output_path: PathBuf,
    pub runtime_log_output_path: PathBuf,
    pub stats_output_path: PathBuf,
    pub history_output_directory_path: Option<PathBuf>,
}
