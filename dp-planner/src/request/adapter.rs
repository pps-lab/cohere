use float_cmp::{ApproxEq, F64Margin};
use int_conv::Split;
use itertools::Itertools;
use log::trace;
use probability::distribution::{Categorical, Sample};
use probability::source::Source;
//use rand::seq::SliceRandom;
//use rand::prelude::StdRng;
//use rand::{RngCore, SeedableRng};
use serde::Deserialize;
use serde_aux::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;

use crate::dprivacy::rdp_alphas_accounting::RdpAlphas;
use crate::dprivacy::{AccountingType, AdpAccounting};
use crate::request::adapter::AdapterProfitOption::{DependentProfit, IndependentProfit};
use crate::request::external::ExternalRequest;
use crate::request::{AdapterInfo, ModificationStatus};

/// Contains the instructions supplied by the file, as well as a seeded source of randomness.
pub struct RequestAdapter {
    /// A source of randomness. If this source is seeded to the same value, and the same adapter
    /// as well as the same request input is given, then the result of applying an adapter should
    /// be the same.
    pub source: probability::source::Default,
    /// The instructions for how to change requests
    instructions: RequestAdapterInstructions,
}

/// Contains information about how to change given requests. All of these are optional - if one is
/// not set, that means that a certain aspect of the given requests will remain unchanged.
#[derive(Deserialize, Clone, Debug)]
struct RequestAdapterInstructions {
    /// How to change the privacy cost
    privacy_cost: Option<AdapterCostOption>,
    /// How to change the requested number of blocks
    n_blocks: Option<AdapterBlocksOption>,
    /// How to change the profit achieved by fulfilling a request
    profit: Option<AdapterProfitOption>,
    /// If this is set to some value n, then every n-th request will have their predicates
    /// regarding which partitioning attributes are demanded by this request replaced by
    /// predicates that always return true. In other words, every n-th requests wants access
    /// to all virtual blocks.
    no_pa_inverse_frac: Option<u64>,
    /// How to convert a request to rdp.
    rdp_conversion: Option<RdpConversionOptions>,
}
/// A certain profit category.
///
/// Different profits might be assigned to the same category according to the specified
/// probabilities.
#[derive(Deserialize, Clone, Debug)]
pub struct Profit {
    #[allow(dead_code)]
    /// The name of the profit category
    name: Option<String>,
    /// Which privacy costs match this profit category ("*" matches everything)
    privacy_cost_name_pattern: String,
    /// How many required blocks requests need to have to mach this profit category
    /// ("*" matches everything)
    n_blocks_name_pattern: String,
    /// The options for this profit category (either one value or multiple values, for which
    /// an option is randomly chosen by probabilities defined in [AdapterOption])
    options: AdapterOption<u64>,
}

/// Which types of noise should be assumed for adp to rdp conversion
#[derive(Deserialize, Clone, Debug)]
pub struct RdpConversionOptions(Vec<ConversionOption>);
/// The probabilities with which gaussian or laplacian noise is assumed when converting adp to
/// rdp costs. The probabilities need to sum to one.
#[derive(Deserialize, Clone, Debug)]
pub enum ConversionOption {
    /// Assume there is laplacian noise with the given probability.
    Laplacian { prob: f64 },
    /// Assume there is gaussian noise with the given probability.
    Gaussian { prob: f64 },
}

/// Whether the profit may depend on cost and/or profit set by this adapter.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum AdapterProfitOption {
    /// The profit is independent of cost and profit
    IndependentProfit(AdapterOption<u64>),
    /// The profit depends on cost and/or profit.
    DependentProfit(Vec<Profit>),
}

/// Note: This variant is needed in addition to [Variant](Variant) to enable deserializing a number
/// from a string, which is not possible if the number is a generic (as the generic might also be
/// something else)
#[derive(Deserialize, Clone, Debug)]
pub struct VariantUsize {
    /// The name of this possibility
    pub name: Option<String>,
    /// The probability that this possibility happens
    pub prob: f64,
    /// The value that is assigned when this possibility happens
    #[serde(deserialize_with = "deserialize_number_from_string")]
    pub value: usize,
}

/// Defines a possibility for a certain field to assume a certain value
#[derive(Deserialize, Clone, Debug)]
pub struct Variant<T> {
    /// The name of this possibility
    pub name: Option<String>,
    /// The probability that this possibility happens
    pub prob: f64,
    /// The value that is assigned when this possibility happens
    pub value: T,
}

/// The different options for a field to be assigned a value.
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum AdapterOption<T> {
    /// Always assign the same value to the field
    Constant(T),
    /// Assign a value randomly from among the contained [Variant]
    Categorical(Vec<Variant<T>>),
}

/// The different options for a privacy cost field to assume
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum AdapterCostOption {
    /// Assign each request the given cost
    Constant(AccountingType),
    /// Assign each request one of the given costs according to the probabilities in
    /// [Variant]
    Categorical(Vec<Variant<AccountingType>>),
}

/// The different options for how the request amount of blocks for a request is assigned
#[derive(Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum AdapterBlocksOption {
    /// The same value is assigned for each request
    #[serde(deserialize_with = "deserialize_number_from_string")]
    Constant(usize),
    /// Assign each request one of the given number of blocks according to the probabilities in
    /// [Variant]
    Categorical(Vec<VariantUsize>),
}

/// A combination of a certain cost name and a number of blocks.
///
/// For each such combination, a profit can be defined (though one profit can apply to multiple of
/// these by using "*".)
#[derive(Eq, PartialEq, Hash)]
struct CostBlockCount(String, usize);

/// Same as [Variant], but the value is option, in which case the value from the request file will
/// be used
#[derive(Debug, Clone)]
struct OptionalVariant<T> {
    /// The probability of this variant (between 0 (exclusive) and 1 (inclusive))
    prob: f64,
    /// The value that is assigned when this variant happens (if None, the existing value will be
    /// used)
    val: Option<T>,
    /// The name of this variant (if specified)
    name: Option<String>,
}

/// Combines one choice of privacy cost and one choice of n_blocks with a profit wrapper. Any
/// request with privacy cost and n_blocks according to this variant will get a profit assigned
/// according to the [ProfitWrapper]
///
/// Note that if the values are none, this means that we do not set the corresponding category
/// (privacy cost or n_blocks) on any requests.
#[derive(Clone)]
struct VariantCombo<'pw, 'p> {
    /// The name of the privacy cost variant
    pcost_name: Option<String>,
    /// The value of the privacy cost variant
    pcost_val: Option<AccountingType>,
    /// The name of the n_blocks variant
    n_blocks_name: Option<String>,
    /// The amount of blocks in the n_blocks variant
    n_blocks_val: Option<usize>,
    /// The profit wrapper to assign the profit to fitting requests.
    profit_wrapper: &'pw ProfitWrapper<'p>,
}

/// For a given combination of privacy cost and n_users (= category), provides a get_profit method to
/// get a profit and a name.
struct ProfitWrapper<'p> {
    /// The profit values which may be assigned to the given category
    profit: &'p Profit,
    /// Used if multiple profit values are present to sample one according to the probabilities
    /// specified in the adapter
    categorical: Option<Categorical>,
    /// A function which returns whether or or not the given privacy cost and n_blocks
    /// match the given [Profit]
    #[allow(clippy::type_complexity)]
    adapter_option_selector: Box<dyn Fn(&str, usize) -> bool>,
    /// If there is only a singe possibility for the profit amount, then this will be Some(x),
    /// where x is the profit to be assigned
    const_profit: Option<u64>,
}

/// A helper construct to identify which part of [AdapterInfo] should be updated
enum UpdateTarget {
    /// The cost of a request was updated
    Cost,
    /// The number of users demanded by a request was updated
    NUsers,
    /// The profit generated by allocating a request was updated.
    Profit,
}

impl RdpConversionOptions {
    /// Get probabilities and names of conversion options
    fn get_probs_and_names(&self) -> Vec<(f64, String)> {
        self.0
            .iter()
            .map(|x| match x {
                ConversionOption::Laplacian { prob } => (*prob, "laplacian".to_string()),
                ConversionOption::Gaussian { prob } => (*prob, "gaussian".to_string()),
            })
            .collect()
    }

    /// Only get probabilities
    fn get_probs(&self) -> Vec<f64> {
        self.get_probs_and_names()
            .into_iter()
            .map(|(prob, _)| prob)
            .collect()
    }

    /// Check that conversion options are valid (probabilities sum to 1)
    fn check(&self) {
        assert!(
            self.get_probs()
                .into_iter()
                .sum::<f64>()
                .approx_eq(1.0, F64Margin::default()),
            "rdp conversion options need to sum to 1"
        )
    }

    /// Given a source of randomness, returns name of an option according to the specified
    /// probabilities
    fn get_random_option_name(&self, source: &mut probability::source::Default) -> String {
        let (probs, names): (Vec<f64>, Vec<String>) =
            self.get_probs_and_names().into_iter().unzip();
        let categorical = Categorical::new(&probs);
        names[categorical.sample(source)].clone()
    }
}

impl RequestAdapter {
    /// Initializes an adapter using the file at the given path and the given seed.
    /// The adapter is deterministic in the presence of a fixed seed, meaning that given the same
    /// adapter file and the same requests, the same transformed requests will be generated by
    /// applying the adapter to the requests.
    pub fn new(adapter_file_pathbuf: PathBuf, seed: u128) -> Self {
        let instructions =
            RequestAdapterInstructions::load_request_adapter_instructions(adapter_file_pathbuf);
        let mut source = probability::source::default();
        let split_seed = [seed.hi(), seed.lo()];
        assert_ne!(split_seed, [0u64, 0u64], "Seed needs to be nonzero");
        source = source.seed(split_seed);
        RequestAdapter {
            source,
            instructions,
        }
    }

    /// Retrieve the inverse fraction of how many requests "lose" their partitioning attributes when
    /// applying the adapter (in the sense that these requests demand access to all virtual
    /// blocks of a block)
    pub(crate) fn get_no_pa_inverse_frac(&self) -> Option<u64> {
        self.instructions.no_pa_inverse_frac
    }

    /// Changes the number of blocks in the adapter to the given argument. Fails if the number of
    /// options for blocks is not the same as the given argument.
    pub fn change_n_blocks(&mut self, new_n_blocks_and_prob: &[(usize, f64)]) {
        assert!(new_n_blocks_and_prob
            .iter()
            .map(|(_val, prob)| prob)
            .sum::<f64>()
            .approx_eq(1.0, F64Margin::default()));

        // map old to new
        let mut value_map: BTreeMap<usize, usize> = BTreeMap::new();

        let curr_block_instr = &mut self
            .instructions
            .n_blocks
            .as_mut()
            .expect("Tried to change n_blocks in adapter, but none present");

        match curr_block_instr {
            AdapterBlocksOption::Constant(n_blocks) => {
                assert_eq!(
                    new_n_blocks_and_prob.len(),
                    1,
                    "Adapter has one block option, but passed argument has {}",
                    new_n_blocks_and_prob.len()
                );

                value_map.insert(*n_blocks, new_n_blocks_and_prob[0].0);
                *n_blocks = new_n_blocks_and_prob[0].0;
            }
            AdapterBlocksOption::Categorical(n_blocks_vec) => {
                for (new, old) in new_n_blocks_and_prob.iter().zip_eq(n_blocks_vec.iter_mut()) {
                    let inserted = value_map.insert(old.value, new.0);
                    assert!(inserted.is_none());
                    old.prob = new.1;
                    old.value = new.0;
                }
            }
        }

        if let Some(DependentProfit(profits)) = &mut self.instructions.profit.as_mut() {
            for profit in profits.iter_mut() {
                if profit.n_blocks_name_pattern != "*" {
                    let n_block_option = usize::from_str(&profit.n_blocks_name_pattern);
                    if let Ok(n_block_val) = n_block_option {
                        profit.n_blocks_name_pattern = value_map[&n_block_val].to_string();
                    } else {
                        panic!(
                            "Malformated selection criteria for profit: {}",
                            &profit.n_blocks_name_pattern
                        );
                    }
                }
            }
        }
    }

    /// Returns an adapter which does not modify any requests.
    pub fn get_empty_adapter() -> Self {
        let instructions = RequestAdapterInstructions::get_empty_request_adapter_instructions();
        let source = probability::source::default();
        RequestAdapter {
            source,
            instructions,
        }
    }

    /// Apply the adapter to the given external requests, possibly changing cost, number of blocks
    /// and/or profits of the requests.
    /// Additionally, if alphas are not None, request cost is converted to rdp.
    pub(crate) fn apply(
        &mut self,
        external_requests: &mut Vec<ExternalRequest>,
        alphas: &Option<RdpAlphas>,
    ) {
        for request in external_requests.iter_mut() {
            request.adapter_info = Some(AdapterInfo {
                privacy_cost: std::default::Default::default(),
                n_users: std::default::Default::default(),
                profit: std::default::Default::default(),
            });
        }

        // apply adapter without worrying about conversion to rdp
        self.instructions.apply(external_requests, &mut self.source);

        // if conversion to rdp is demanded, do conversion now
        Self::convert_to_rdp(
            external_requests,
            alphas,
            &mut self.source,
            &self.instructions.rdp_conversion,
        );
    }

    /// If alphas are Some(..), then the given requests' costs will be converted to RDP.
    ///
    /// Conversion_options specify the type of conversion (see [RdpConversionOptions]),
    /// and the source is used to select the noise in the end (as the conversion_options
    /// specify probabilities)
    fn convert_to_rdp(
        external_requests: &mut Vec<ExternalRequest>,
        alphas: &Option<RdpAlphas>,
        source: &mut probability::source::Default,
        conversion_options: &Option<RdpConversionOptions>,
    ) {
        // only can convert if alpha vals are given
        if let Some(alpha_vals) = alphas {
            // check if conversion options given - if not, just use gaussian:
            let conversion_options = match conversion_options {
                None => RdpConversionOptions(vec![ConversionOption::Gaussian { prob: 1.0 }]),
                Some(x) => x.clone(),
            };

            // check that conversion option is valid
            conversion_options.check();

            let mut unique_costs: Vec<AccountingType> = Vec::new();
            for request in external_requests {
                let conversion_name = conversion_options.get_random_option_name(source);
                match conversion_name.as_str() {
                    "gaussian" => {
                        request.request_cost = Some(
                            request
                                .request_cost
                                .as_ref()
                                .unwrap()
                                .adp_to_rdp_cost_gaussian(alpha_vals),
                        );
                    }
                    "laplacian" => {
                        request.request_cost = Some(
                            request
                                .request_cost
                                .as_ref()
                                .unwrap()
                                .adp_to_rdp_cost_laplacian(alpha_vals),
                        );
                    }
                    _ => {
                        panic!("Unknown rdp conversion option")
                    }
                }
                if unique_costs.iter().all(|x| {
                    !x.approx_eq(request.request_cost.as_ref().unwrap(), F64Margin::default())
                }) {
                    unique_costs.push(request.request_cost.as_ref().unwrap().clone())
                }
            }
            trace!("Request costs after conversion to rdp: {:?}", unique_costs);
        }
    }
}

impl<'p> ProfitWrapper<'p> {
    /// Given a [Profit], returns a [ProfitWrapper] which offer the convenient
    /// [ProfitWrapper::get_profit] method.
    fn new(profit: &'p Profit) -> Self {
        let adapter_option_selector = RequestAdapterInstructions::adapter_option_selector(profit);
        match &profit.options {
            AdapterOption::Constant(val) => ProfitWrapper {
                categorical: None,
                adapter_option_selector,
                const_profit: Some(*val),
                profit,
            },
            AdapterOption::Categorical(opts) => ProfitWrapper {
                categorical: Some(Categorical::new(
                    &opts.iter().map(|variant| variant.prob).collect::<Vec<_>>(),
                )),
                adapter_option_selector,
                const_profit: None,
                profit,
            },
        }
    }

    /// Given a profit wrapper, returns a profit and a corresponding name (if available)
    fn get_profit<T: Source>(&self, source: &mut T) -> (u64, Option<String>) {
        if let Some(val) = self.const_profit {
            assert!(self.categorical.is_none());
            (val, None)
        } else {
            let i = self
                .categorical
                .as_ref()
                .expect("Categorical not set")
                .sample(source);
            if let AdapterOption::Categorical(possible_profits) = &self.profit.options {
                (possible_profits[i].value, possible_profits[i].name.clone())
            } else {
                panic!("Did not find multiple options in ProfitWrapper")
            }
        }
    }

    fn get_correct_wrapper<'w>(
        cost_name: &str,
        n_blocks: usize,
        wrappers: &'w [ProfitWrapper<'p>],
    ) -> &'w ProfitWrapper<'p> {
        wrappers
            .iter()
            .find(|wrapper| (wrapper.adapter_option_selector)(cost_name, n_blocks))
            .expect("Did not find matching wrapper")
    }
}

impl RequestAdapterInstructions {
    fn get_empty_request_adapter_instructions() -> RequestAdapterInstructions {
        RequestAdapterInstructions {
            privacy_cost: None,
            n_blocks: None,
            profit: None,
            no_pa_inverse_frac: None,
            rdp_conversion: None,
        }
    }

    fn load_request_adapter_instructions(path_buf: PathBuf) -> RequestAdapterInstructions {
        let file = File::open(&path_buf).unwrap_or_else(|_| {
            panic!(
                "Could not open adapter file at {:?}",
                fs::canonicalize(&path_buf).unwrap_or(path_buf)
            )
        });
        let reader = BufReader::new(file);

        // Parse with serde
        let adapter: RequestAdapterInstructions =
            serde_json::from_reader(reader).expect("Parsing Failed");

        // check that names of privacy costs are defined and unique
        if let Some(AdapterCostOption::Categorical(privacy_costs)) = &adapter.privacy_cost {
            let mut name_set = HashSet::new();
            for privacy_cost in privacy_costs {
                assert!(
                    privacy_cost.name.is_some(),
                    "Defined multiple privacy costs without naming (some of) them"
                );
                let inserted = name_set.insert(privacy_cost.name.as_ref().unwrap());
                assert!(inserted, "Need to use unique names for privacy costs")
            }
        }

        // check that n_blocks are unique
        if let Some(AdapterBlocksOption::Categorical(n_blocks)) = &adapter.n_blocks {
            let mut block_number_set = HashSet::new();
            for n_block in n_blocks {
                let inserted = block_number_set.insert(n_block.value);
                assert!(
                    inserted,
                    "Need to use unique numbers for different possibilities of n_blocks"
                )
            }
        }

        // enforce all probabilities sum to one if AdapterOption::Categorical is used
        if let Some(AdapterCostOption::Categorical(cost_vec)) = &adapter.privacy_cost {
            assert!(
                cost_vec
                    .iter()
                    .map(|cost| cost.prob)
                    .sum::<f64>()
                    .approx_eq(1., F64Margin::default()),
                "Probabilities of privacy costs need to sum to one"
            );

            assert!(
                cost_vec
                    .iter()
                    .all(|variant| variant.prob >= 0. && variant.prob <= 1.),
                "Probabilities of privacy costs need to be >= 0 and <= 1"
            );
        }
        if let Some(AdapterBlocksOption::Categorical(n_block_vec)) = &adapter.n_blocks {
            assert!(
                n_block_vec
                    .iter()
                    .map(|cost| cost.prob)
                    .sum::<f64>()
                    .approx_eq(1., F64Margin::default()),
                "Probabilities of n_blocks need to sum to one"
            );
            assert!(
                n_block_vec
                    .iter()
                    .all(|variant| variant.prob >= 0. && variant.prob <= 1.),
                "Probabilities of n_blocks need to be >= 0 and <= 1"
            );
        }
        if let Some(DependentProfit(profits)) = &adapter.profit {
            for profit in profits.iter() {
                if let AdapterOption::Categorical(profit_option) = &profit.options {
                    assert!(
                        profit_option
                            .iter()
                            .map(|variant| variant.prob)
                            .sum::<f64>()
                            .approx_eq(1., F64Margin::default()),
                        "Probabilities of each kind of profit need to sum to one"
                    )
                }
            }
        }

        // check that profit covers all combinations of n_block vals and privacy cost names
        if let Some(adapter_profit_option) = &adapter.profit {
            match adapter_profit_option {
                // if profit is not categorical, then no further checks are necessary
                IndependentProfit(_profit) => {}
                // check that if profit is categorical, then it partitions (privacy cost names x values of adapter.n_blocks)
                // i.e., each combination can be mapped to exactly one profit
                DependentProfit(profits) => {
                    let mut profit_counts =
                        RequestAdapterInstructions::get_empty_profit_counts(&adapter);
                    assert!(!profits.is_empty(), "Empty array of profits given");
                    for profit in profits {
                        let adapter_selector =
                            RequestAdapterInstructions::adapter_option_selector(profit);
                        for (CostBlockCount(privacy_cost_name, n_blocks_val), count) in
                            profit_counts.iter_mut()
                        {
                            if adapter_selector(privacy_cost_name, *n_blocks_val) {
                                *count += 1;
                            }
                        }
                    }
                    for (CostBlockCount(privacy_cost_name, n_blocks_val), count) in
                        profit_counts.iter_mut()
                    {
                        assert_ne!(*count, 0, "The combination of privacy cost name {} and n_blocks_val {} was not assigned a profit", privacy_cost_name, n_blocks_val);
                        assert!(*count <= 1, "The combination of privacy cost name {} and n_blocks_val {} was assigned a profit more than once", privacy_cost_name,n_blocks_val);
                    }
                }
            }
        }

        adapter
    }

    #[allow(clippy::type_complexity)]
    fn adapter_option_selector(profit: &Profit) -> Box<dyn Fn(&str, usize) -> bool> {
        let n_block_selector: Box<dyn Fn(usize) -> bool>;
        if profit.n_blocks_name_pattern == "*" {
            n_block_selector = Box::new(|_: usize| true);
        } else {
            let n_block_option = usize::from_str(&profit.n_blocks_name_pattern);
            if let Ok(n_block_val) = n_block_option {
                n_block_selector = Box::new(move |x: usize| x == n_block_val);
            } else {
                panic!(
                    "Malformated selection criteria for profit {}",
                    &profit.n_blocks_name_pattern
                );
            }
        }
        let privacy_cost_name: String;
        let privacy_cost_name_selector: Box<dyn Fn(&str) -> bool>;
        if profit.privacy_cost_name_pattern == "*" {
            privacy_cost_name_selector = Box::new(|_| true);
        } else {
            privacy_cost_name = profit.privacy_cost_name_pattern.to_owned();
            privacy_cost_name_selector = Box::new(move |str| str == privacy_cost_name);
        }
        Box::new(move |privacy_cost_name: &str, n_blocks: usize| {
            privacy_cost_name_selector(privacy_cost_name) && n_block_selector(n_blocks)
        })
    }

    fn get_empty_profit_counts(
        adapter: &RequestAdapterInstructions,
    ) -> HashMap<CostBlockCount, usize> {
        // collect all names of privacy costs
        let privacy_cost_names: Vec<String>;
        match &adapter.privacy_cost {
            None => privacy_cost_names = vec!["default".to_string()],
            Some(adapter_option) => match adapter_option {
                AdapterCostOption::Constant(_p_cost) => {
                    privacy_cost_names = vec!["unit_cost".to_string()]
                }
                AdapterCostOption::Categorical(p_costs) => {
                    privacy_cost_names = p_costs
                        .iter()
                        .map(|p_cost| p_cost.name.as_ref().unwrap().clone())
                        .collect();
                }
            },
        }

        // collect all values for the n_blocks constraint
        let n_block_vals: Vec<usize>;
        match &adapter.n_blocks {
            None => n_block_vals = vec![0usize],
            Some(adapter_option) => match adapter_option {
                AdapterBlocksOption::Constant(val) => n_block_vals = vec![*val],
                AdapterBlocksOption::Categorical(variants) => {
                    n_block_vals = variants.iter().map(|variant| variant.value).collect();
                }
            },
        }

        let mut empty_profit_counts: HashMap<CostBlockCount, usize> = HashMap::new();
        for privacy_cost_name in privacy_cost_names {
            for n_block_val in n_block_vals.iter() {
                empty_profit_counts
                    .insert(CostBlockCount(privacy_cost_name.clone(), *n_block_val), 0);
            }
        }
        empty_profit_counts
    }

    /// applies the request adapter, possibly changing profit, n_users and/or request cost
    fn apply(
        &self,
        external_requests: &mut [ExternalRequest],
        source: &mut probability::source::Default,
    ) {
        let mut update_privacy_costs_and_blocks = || {
            let pc = |request: &mut ExternalRequest,
                      cost: &AccountingType,
                      cost_name: Option<String>| {
                request.request_cost = Some(cost.clone());
                Self::update_adapter_info(request, cost_name, UpdateTarget::Cost);
            };
            Self::update_costs(external_requests, &self.privacy_cost, pc, source);

            // update blocks
            let blocks =
                |request: &mut ExternalRequest, n_blocks: &usize, blocks_name: Option<String>| {
                    request.n_users = Some(*n_blocks);
                    Self::update_adapter_info(request, blocks_name, UpdateTarget::NUsers);
                };
            Self::update_nblocks(external_requests, &self.n_blocks, blocks, source);
        };

        if self.profit.is_none() {
            update_privacy_costs_and_blocks();
        } else {
            let profit_updater =
                |request: &mut ExternalRequest, profit: &u64, profit_name: Option<String>| {
                    request.profit = Some(*profit);
                    Self::update_adapter_info(request, profit_name, UpdateTarget::Profit);
                };
            match self.profit.as_ref().unwrap() {
                IndependentProfit(profit) => {
                    update_privacy_costs_and_blocks();

                    Self::update_requests(
                        external_requests,
                        &Some(profit.clone()),
                        profit_updater,
                        source,
                    );
                }
                DependentProfit(profits) => {
                    assert!(!profits.is_empty());
                    if profits.len() == 1 {
                        update_privacy_costs_and_blocks();

                        Self::update_requests(
                            external_requests,
                            &Some(profits[0].options.clone()),
                            profit_updater,
                            source,
                        );
                    } else {
                        RequestAdapterInstructions::joint_update(
                            external_requests,
                            &self.privacy_cost,
                            &self.n_blocks,
                            profits,
                            source,
                        );
                    }
                }
            }
        }
    }

    /// Updates adapter_info, should be used after updating request cost etc.
    fn update_adapter_info(
        request: &mut ExternalRequest,
        name: Option<String>,
        update_target: UpdateTarget,
    ) {
        match update_target {
            UpdateTarget::Cost => {
                if let Some(name_str) = name {
                    request
                        .adapter_info
                        .as_mut()
                        .expect("Adapter info not initialized")
                        .privacy_cost = ModificationStatus::Named(name_str);
                } else {
                    request
                        .adapter_info
                        .as_mut()
                        .expect("Adapter info not initialized")
                        .privacy_cost = ModificationStatus::Unnamed;
                }
            }
            UpdateTarget::NUsers => {
                if let Some(name_str) = name {
                    request
                        .adapter_info
                        .as_mut()
                        .expect("Adapter info not initialized")
                        .n_users = ModificationStatus::Named(name_str);
                } else {
                    request
                        .adapter_info
                        .as_mut()
                        .expect("Adapter info not initialized")
                        .n_users = ModificationStatus::Unnamed;
                }
            }
            UpdateTarget::Profit => {
                if let Some(name_str) = name {
                    request
                        .adapter_info
                        .as_mut()
                        .expect("Adapter info not initialized")
                        .profit = ModificationStatus::Named(name_str);
                } else {
                    request
                        .adapter_info
                        .as_mut()
                        .expect("Adapter info not initialized")
                        .profit = ModificationStatus::Unnamed;
                }
            }
        }
    }

    fn adapter_cost_option_to_probs(
        opt: &Option<AdapterCostOption>,
    ) -> Vec<OptionalVariant<AccountingType>> {
        let mut probs = vec![OptionalVariant {
            prob: 1.,
            val: None,
            name: None,
        }];
        if let Some(AdapterCostOption::Categorical(variants)) = opt {
            probs = variants
                .iter()
                .map(|variant| OptionalVariant {
                    prob: variant.prob,
                    val: Some(variant.value.clone()),
                    name: variant.name.clone(),
                })
                .collect();
            assert!(probs
                .iter()
                .map(|o_var| o_var.prob)
                .sum::<f64>()
                .approx_eq(1., F64Margin::default()));
        } else if let Some(AdapterCostOption::Constant(val)) = opt {
            probs = vec![OptionalVariant {
                prob: 1.0,
                val: Some(val.clone()),
                name: None,
            }];
        };
        probs
    }

    fn adapter_blocks_option_to_probs(
        opt: &Option<AdapterBlocksOption>,
    ) -> Vec<OptionalVariant<usize>> {
        let mut probs = vec![OptionalVariant {
            prob: 1.,
            val: None,
            name: None,
        }];
        if let Some(AdapterBlocksOption::Categorical(variants)) = opt {
            probs = variants
                .iter()
                .map(|variant| OptionalVariant {
                    prob: variant.prob,
                    val: Some(variant.value),
                    name: variant.name.clone(),
                })
                .collect();
            assert!(probs
                .iter()
                .map(|o_var| o_var.prob)
                .sum::<f64>()
                .approx_eq(1., F64Margin::default()));
        } else if let Some(AdapterBlocksOption::Constant(val)) = opt {
            probs = vec![OptionalVariant {
                prob: 1.0,
                val: Some(*val),
                name: None,
            }];
        };
        probs
    }

    /// call this function if profit is dependent on cost and/or n_blocks
    fn joint_update<S>(
        requests: &mut [ExternalRequest],
        opt_privacy_cost: &Option<AdapterCostOption>,
        opt_n_blocks: &Option<AdapterBlocksOption>,
        profits: &[Profit],
        source: &mut S,
    ) where
        S: Source,
    {
        let privacy_cost_probs =
            RequestAdapterInstructions::adapter_cost_option_to_probs(opt_privacy_cost);
        let n_blocks_probs =
            RequestAdapterInstructions::adapter_blocks_option_to_probs(opt_n_blocks);
        let profit_wrappers: Vec<ProfitWrapper> = profits.iter().map(ProfitWrapper::new).collect();
        assert!(
            profits.len() > 1,
            "Profits not dependent on cost and/or n_blocks, no need to update jointly"
        );

        // determine joint probabilities, and construct a variant_combo object for each such joint probability
        let mut joint_probs: Vec<f64> =
            Vec::with_capacity(privacy_cost_probs.len() * n_blocks_probs.len());
        let mut flat_index_to_variant_combo: Vec<VariantCombo> =
            Vec::with_capacity(joint_probs.len());

        for (pcost_index, pcost_o_var) in privacy_cost_probs.iter().enumerate() {
            for (block_index, block_o_var) in n_blocks_probs.iter().enumerate() {
                let flat_index = pcost_index * n_blocks_probs.len() + block_index;
                joint_probs.push(pcost_o_var.prob * block_o_var.prob);
                assert_eq!(joint_probs.len() - 1, flat_index);
                flat_index_to_variant_combo.push(VariantCombo {
                    pcost_name: pcost_o_var.name.clone(),
                    pcost_val: pcost_o_var.val.clone(),
                    n_blocks_name: block_o_var.name.clone(),
                    n_blocks_val: block_o_var.val,
                    profit_wrapper: ProfitWrapper::get_correct_wrapper(
                        pcost_o_var
                            .name
                            .as_ref()
                            .unwrap_or(&"default default".to_string()),
                        block_o_var.val.unwrap_or(usize::MAX),
                        &profit_wrappers,
                    ),
                });
                assert_eq!(flat_index_to_variant_combo.len() - 1, flat_index);
            }
        }

        // update each request
        let categorical = Categorical::new(&joint_probs);
        for request in requests.iter_mut() {
            // sample from joint distribution and get VariantCombo
            let variant_combo = &flat_index_to_variant_combo[categorical.sample(source)];
            // get corresponding profit (possibly constant, possibly sampled)
            let (profit, profit_name) = variant_combo.profit_wrapper.get_profit(source);
            // update request
            if let Some(pcost_val) = &variant_combo.pcost_val {
                request.request_cost = Some(pcost_val.clone());
                Self::update_adapter_info(
                    request,
                    variant_combo.pcost_name.clone(),
                    UpdateTarget::Cost,
                );
            }
            if let Some(n_blocks_val) = &variant_combo.n_blocks_val {
                request.n_users = Some(*n_blocks_val);
                Self::update_adapter_info(
                    request,
                    variant_combo.n_blocks_name.clone(),
                    UpdateTarget::NUsers,
                );
            }
            request.profit = Some(profit);
            Self::update_adapter_info(request, profit_name, UpdateTarget::Profit);
        }
    }

    fn update_requests<T, F, S>(
        requests: &mut [ExternalRequest],
        opt: &Option<AdapterOption<T>>,
        update_func: F,
        source: &mut S,
    ) where
        F: Fn(&mut ExternalRequest, &T, Option<String>),
        S: Source,
    {
        match opt {
            Some(AdapterOption::Constant(x)) => requests
                .iter_mut()
                .for_each(|request| update_func(request, x, None)),
            Some(AdapterOption::Categorical(variants)) => {
                let probs: Vec<f64> = variants.iter().map(|x| x.prob).collect();

                let categorical = Categorical::new(&probs);

                requests.iter_mut().for_each(|request| {
                    let i = categorical.sample(source);
                    update_func(request, &variants[i].value, variants[i].name.clone());
                });
            }

            None => (), // do nothing
        }
    }

    fn update_costs<F, S>(
        requests: &mut [ExternalRequest],
        opt: &Option<AdapterCostOption>,
        update_func: F,
        source: &mut S,
    ) where
        F: Fn(&mut ExternalRequest, &AccountingType, Option<String>),
        S: Source,
    {
        match opt {
            Some(AdapterCostOption::Constant(x)) => requests
                .iter_mut()
                .for_each(|request| update_func(request, x, None)),
            Some(AdapterCostOption::Categorical(variants)) => {
                let probs: Vec<f64> = variants.iter().map(|x| x.prob).collect();

                let categorical = Categorical::new(&probs);

                requests.iter_mut().for_each(|request| {
                    let i = categorical.sample(source);
                    update_func(request, &variants[i].value, variants[i].name.clone());
                });
            }

            None => (), // do nothing
        }
    }

    fn update_nblocks<F, S>(
        requests: &mut [ExternalRequest],
        opt: &Option<AdapterBlocksOption>,
        update_func: F,
        source: &mut S,
    ) where
        F: Fn(&mut ExternalRequest, &usize, Option<String>),
        S: Source,
    {
        match opt {
            Some(AdapterBlocksOption::Constant(x)) => requests
                .iter_mut()
                .for_each(|request| update_func(request, x, None)),
            Some(AdapterBlocksOption::Categorical(variants)) => {
                let probs: Vec<f64> = variants.iter().map(|x| x.prob).collect();

                let categorical = Categorical::new(&probs);

                requests.iter_mut().for_each(|request| {
                    let i = categorical.sample(source);
                    update_func(request, &variants[i].value, variants[i].name.clone());
                });
            }

            None => (), // do nothing
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
    use crate::request::adapter::RequestAdapter;
    use crate::request::external::{convert_requests, ExternalRequest};
    use crate::request::ModificationStatus;
    use crate::schema::Schema;
    use crate::util::{build_dummy_requests_with_pa, build_dummy_schema};
    use crate::AccountingType::{EpsDp, Rdp};
    use crate::{request, AccountingType, RequestId};
    use float_cmp::{ApproxEq, F64Margin};
    use std::path::PathBuf;

    static SEED: u128 = 1848;

    #[test]
    #[should_panic(expected = "rdp conversion options need to sum to 1")]
    fn adapter_rdp_conversion_should_panic() {
        let schema = Schema {
            accounting_type: Rdp {
                eps_values: A5([0., 0., 0., 0., 0.]),
            },
            attributes: vec![],
            name_to_index: Default::default(),
        };

        let external_requests: Vec<ExternalRequest> = (0..10)
            .map(|i| ExternalRequest {
                request_id: RequestId(i),
                request_cost: None,
                profit: Some(1),
                dnf: Default::default(),
                n_users: None,
                created: None,
                adapter_info: None,
            })
            .collect();

        let mut adapter = RequestAdapter::new(
            PathBuf::from(
                "resources/test/adapter_configs/adapter_config_rdp_conversion_should_panic.json",
            ),
            SEED,
        );

        let _converted = convert_requests(
            external_requests,
            &schema,
            &mut adapter,
            &Some(A5([2., 4., 8., 16., 32.])),
        )
        .expect("Conversion failed");
    }

    #[test]
    fn adapter_rdp_conversion() {
        let schema = Schema {
            accounting_type: Rdp {
                eps_values: A5([0., 0., 0., 0., 0.]),
            },
            attributes: vec![],
            name_to_index: Default::default(),
        };

        let external_requests: Vec<ExternalRequest> = (0..20)
            .map(|i| ExternalRequest {
                request_id: RequestId(i),
                request_cost: None,
                profit: Some(1),
                dnf: Default::default(),
                n_users: None,
                created: None,
                adapter_info: None,
            })
            .collect();

        let mut adapter = RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config_rdp_conversion.json"),
            SEED,
        );

        let converted = convert_requests(
            external_requests,
            &schema,
            &mut adapter,
            &Some(A5([2., 4., 8., 16., 32.])),
        )
        .expect("Conversion failed");

        let mut unique_costs: Vec<AccountingType> = Vec::new();

        for (_, request) in converted.into_iter() {
            if unique_costs
                .iter()
                .all(|x| !x.approx_eq(&request.request_cost, F64Margin::default()))
            {
                unique_costs.push(request.request_cost)
            }
        }

        assert_eq!(unique_costs.len(), 4)
    }

    #[test]
    fn adapter_rdp_conversion_unspecified() {
        let schema = Schema {
            accounting_type: Rdp {
                eps_values: A5([0., 0., 0., 0., 0.]),
            },
            attributes: vec![],
            name_to_index: Default::default(),
        };

        let external_requests: Vec<ExternalRequest> = (0..20)
            .map(|i| ExternalRequest {
                request_id: RequestId(i),
                request_cost: None,
                profit: Some(1),
                dnf: Default::default(),
                n_users: None,
                created: None,
                adapter_info: None,
            })
            .collect();

        let mut adapter = RequestAdapter::new(
            PathBuf::from(
                "resources/test/adapter_configs/adapter_config_rdp_conversion_unspecified.json",
            ),
            SEED,
        );

        let converted = convert_requests(
            external_requests,
            &schema,
            &mut adapter,
            &Some(A5([2., 4., 8., 16., 32.])),
        )
        .expect("Conversion failed");

        let mut unique_costs: Vec<AccountingType> = Vec::new();

        for (_, request) in converted.into_iter() {
            if unique_costs
                .iter()
                .all(|x| !x.approx_eq(&request.request_cost, F64Margin::default()))
            {
                unique_costs.push(request.request_cost)
            }
        }

        assert_eq!(unique_costs.len(), 2)
    }

    #[test]
    fn adapter_info_no_joint_update() {
        let schema = Schema {
            accounting_type: EpsDp { eps: 0.0 },
            attributes: vec![],
            name_to_index: Default::default(),
        };

        let external_requests: Vec<ExternalRequest> = (0..10)
            .map(|i| ExternalRequest {
                request_id: RequestId(i),
                request_cost: None,
                profit: Some(1),
                dnf: Default::default(),
                n_users: None,
                created: None,
                adapter_info: None,
            })
            .collect();

        let mut adapter = RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/set_cost_and_nblocks.json"),
            SEED,
        );

        let converted = convert_requests(external_requests, &schema, &mut adapter, &None)
            .expect("Conversion failed");

        for request in converted.into_values() {
            assert!(matches!(
                request.adapter_info.profit,
                ModificationStatus::Unchanged
            ));

            assert!(matches!(
                request.adapter_info.n_users,
                ModificationStatus::Unnamed
            ));

            match request.adapter_info.privacy_cost {
                ModificationStatus::Named(cost_name) => {
                    assert!(cost_name == "mice" || cost_name == "elephant")
                }
                _ => panic!("privacy cost adapter info should be named"),
            }
        }
    }

    #[test]
    fn adapter_info_joint_update() {
        let schema = Schema {
            accounting_type: EpsDp { eps: 0.0 },
            attributes: vec![],
            name_to_index: Default::default(),
        };

        let external_requests: Vec<ExternalRequest> = (0..10)
            .map(|i| ExternalRequest {
                request_id: RequestId(i),
                request_cost: None,
                profit: None,
                dnf: Default::default(),
                n_users: None,
                created: None,
                adapter_info: None,
            })
            .collect();

        let mut adapter = RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config.json"),
            SEED,
        );

        let converted = convert_requests(external_requests, &schema, &mut adapter, &None)
            .expect("Conversion failed");

        for request in converted.into_values() {
            assert!(!matches!(
                request.adapter_info.n_users,
                ModificationStatus::Unchanged
            ));

            assert!(matches!(
                request.adapter_info.n_users,
                ModificationStatus::Unnamed
            ));

            match request.adapter_info.profit {
                ModificationStatus::Named(profit_name) => {
                    assert!(profit_name == "low" || profit_name == "mid" || profit_name == "high")
                }
                _ => panic!("privacy cost adapter info should be named"),
            }

            match request.adapter_info.privacy_cost {
                ModificationStatus::Named(cost_name) => {
                    assert!(cost_name == "mice" || cost_name == "elephant")
                }
                _ => panic!("privacy cost adapter info should be named"),
            }
        }
    }

    #[test]
    fn test_adapter_parsing_adapter_config() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config.json"),
            SEED,
        );
    }

    #[test]
    fn test_adapter_parsing_with_inverse_frac() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config_with_inverse_frac.json"),
            SEED,
        );
    }

    #[test]
    fn test_adapter_parsing_adapter_config_rdp() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config_rdp.json"),
            SEED,
        );
    }

    #[test]
    fn test_adapter_parsing_1() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config_should_not_panic_1.json"),
            SEED,
        );
    }

    #[test]
    fn test_adapter_parsing_constant_profit() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/constant_profit.json"),
            SEED,
        );
    }

    #[test]
    #[should_panic]
    fn test_adapter_parsing_panic_1() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config_should_panic_1.json"),
            SEED,
        );
    }

    #[test]
    #[should_panic]
    fn test_adapter_parsing_panic_2() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config_should_panic_2.json"),
            SEED,
        );
    }

    #[test]
    #[should_panic]
    fn test_adapter_parsing_panic_3() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config_should_panic_3.json"),
            SEED,
        );
    }

    #[test]
    #[should_panic]
    fn test_adapter_parsing_panic_4() {
        RequestAdapter::new(
            PathBuf::from("resources/test/adapter_configs/adapter_config_should_panic_4.json"),
            SEED,
        );
    }

    #[test]
    fn request_joint_update() {
        let schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let mut requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6)
            .into_iter()
            .map(|(_, req)| ExternalRequest {
                request_id: req.request_id,
                request_cost: Some(req.request_cost),
                profit: Some(req.profit),
                dnf: request::external::Dnf {
                    conjunctions: vec![],
                },
                n_users: Some(req.n_users),
                created: None,
                adapter_info: None,
            })
            .collect();
        let seeds: [u128; 18] = [
            42111, 42222, 42333, 42444, 42555, 42666, 42777, 42888, 42999, 421111, 421222, 421333,
            421444, 421555, 421666, 421777, 421888, 421999,
        ];
        for seed in seeds {
            let mut joint_adapter = RequestAdapter::new(
                PathBuf::from("resources/test/adapter_configs/adapter_config.json"),
                seed,
            );

            joint_adapter.apply(&mut requests, &None);
            for request in requests.iter() {
                assert_eq!(request.n_users.unwrap(), 1);
                if request.profit.unwrap() == 100
                    || request.profit.unwrap() == 200
                    || request.profit.unwrap() == 1000
                {
                    // elephant
                    assert_eq!(request.request_cost.clone().unwrap(), EpsDp { eps: 0.8 })
                } else if request.profit.unwrap() == 10 || request.profit.unwrap() == 20 {
                    assert_eq!(request.request_cost.clone().unwrap(), EpsDp { eps: 0.1 })
                }
                println!(
                    "Request id: {:?} Profit: {:04} N_blocks: {:04} Privacy_Cost: {:?} ",
                    request.request_id,
                    request.profit.unwrap(),
                    request.n_users.unwrap(),
                    request.request_cost.clone().unwrap()
                );
            }
        }
    }
}
