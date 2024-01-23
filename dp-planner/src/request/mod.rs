//! Contains structs and methods to define and handle requests.
//!
//! Functions to load and convert external requests, as well as the
//! external request definition is part of [external], everything relating to request adapters in
//! [adapter], and [internal] contains methods for handling requests inside this program

pub mod external;
pub mod internal;

pub mod adapter;

use std::{
    collections::{HashMap, HashSet},
    fmt,
    path::{Path, PathBuf},
};

use itertools::MultiProduct;
use serde::{Deserialize, Serialize};

use crate::schema::{DataValue, DataValueLookup};
use crate::{dprivacy::rdp_alphas_accounting::RdpAlphas, simulation::RoundId};
use crate::{
    schema::{Schema, SchemaError},
    AccountingType, RequestAdapter,
};

use self::internal::PredicateWithSchemaIntoIterator;

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, Deserialize, PartialOrd, Ord, Serialize)]
pub struct RequestId(pub usize);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct AttributeId(pub usize);

#[allow(dead_code)]
pub fn resource_path(filename: &str) -> PathBuf {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"));
    path.join("resources").join("test").join(filename)
}

pub fn load_requests(
    request_path: PathBuf,
    schema: &Schema,
    request_adapter: &mut RequestAdapter,
    alphas: &Option<RdpAlphas>,
) -> Result<HashMap<RequestId, Request>, SchemaError> {
    let external_requests =
        external::parse_requests(request_path).expect("Failed to open or parse requests");
    external::convert_requests(external_requests, schema, request_adapter, alphas)
}

impl Dnf {
    pub fn repeating_iter<'a, 'b>(&'a self, schema: &'b Schema) -> DNFRepeatingIterator<'a, 'b> {
        let iters: Vec<_> = self
            .conjunctions
            .iter()
            .map(|conj| conj.prod_iter(schema))
            .collect();
        assert!(!iters.is_empty());
        DNFRepeatingIterator {
            conj_index: 0,
            iterators: iters,
        }
    }

    pub fn num_virtual_blocks(&self, schema: &Schema) -> usize {
        let virtual_blocks: HashSet<Vec<usize>> = self.repeating_iter(schema).collect();
        virtual_blocks.len()
    }
}

#[derive(Clone, Debug)]
pub struct Request {
    /// A unique identifier for this request. Often used as the key for some hash- or tree-based
    /// map to access a request object.
    pub request_id: RequestId,
    /// How much cost this request incurs when it is allocated some blocks / segments of a block.
    pub request_cost: AccountingType,
    /// The request cost before global alpha reduction, but after the request adapter
    /// possibly changed request costs.
    pub unreduced_cost: AccountingType,
    /// How important this request is compared to other requests. Often times, the goal of
    /// [crate::allocation] is to maximize the total profit
    pub profit: u64,
    dnf: Dnf,
    pub n_users: usize,
    /// identifies the round the request joins the system. Default: 0
    pub created: Option<RoundId>,
    pub(crate) adapter_info: AdapterInfo,
}

/// Records whether certain fields in a request were changed by an adapter, and if these changes
/// were named, also records that name. Needed for evaluation purposes.
#[derive(Default, Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct AdapterInfo {
    privacy_cost: ModificationStatus,
    n_users: ModificationStatus,
    profit: ModificationStatus,
}

/// Records whether a certain field in the record was changed by the privacy adapter, as well as
/// the name of the applied adapter option, if this was specified
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ModificationStatus {
    /// The field was not changed by the adapter
    Unchanged,
    /// The field was changed by the adapter, but the rule that changed it does not
    /// have a name.
    Unnamed,
    /// The field was changed by the adapter, and the rule is named as specified.
    Named(String),
}

impl Default for ModificationStatus {
    fn default() -> Self {
        Self::Unchanged
    }
}

/// Used to start building a request. Usually initialized via [RequestBuilder::new],
/// and then further modified via [RequestBuilder::or_conjunction], and the finalized request
/// is extracted via [RequestBuilder::build]
pub struct RequestBuilder<'a> {
    /// The schema to which the request adheres. The attributes as well as their
    /// respective ranges should be a superset of the ones in the request.
    schema: &'a Schema,
    /// The current state of the request which is being built.
    request: Request,
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<'a> RequestBuilder<'a> {
    /// This initializes a [RequestBuilder] with the given arguments. Calling this method and
    /// then adding conjunctions via [RequestBuilder::or_conjunction] is preferable to manually
    /// creating requests, as this is less error prone and easier to read.
    pub fn new(
        request_id: RequestId,
        request_cost: AccountingType,
        profit: u64,
        n_users: usize,
        schema: &'a Schema,
        adapter_info: AdapterInfo,
    ) -> Self {
        let request = Request {
            request_id,
            request_cost: request_cost.clone(),
            unreduced_cost: request_cost,
            profit,
            dnf: Dnf {
                conjunctions: Vec::new(),
            },
            n_users,
            created: None,
            adapter_info,
        };

        RequestBuilder { request, schema }
    }

    pub fn new_full(
        request_id: RequestId,
        request_cost: AccountingType,
        profit: u64,
        n_users: usize,
        created: Option<RoundId>,
        schema: &'a Schema,
        adapter_info: AdapterInfo,
    ) -> Self {
        let request = Request {
            request_id,
            request_cost: request_cost.clone(),
            unreduced_cost: request_cost,
            profit,
            dnf: Dnf {
                conjunctions: Vec::new(),
            },
            n_users,
            created,
            adapter_info,
        };

        RequestBuilder { request, schema }
    }

    /// Add another conjunction to the dnf in the request in the [RequestBuilder]
    pub fn or_conjunction(mut self, c: Conjunction) -> Self {
        self.request.dnf.conjunctions.push(c);
        self
    }

    /// Construct the request from the input given to [Self::new] and [Self::or_conjunction].
    pub fn build(mut self) -> Request {
        if self.request.dnf.conjunctions.is_empty() {
            // request with empty dnf -> want all blokcs => need to insert all blocks conjunction
            let all_conjunction = ConjunctionBuilder::new(self.schema).build();
            self.request.dnf.conjunctions.push(all_conjunction);
        }

        self.request
    }
}

/// The disjunctive normal form of the predicates attached to a request.
#[derive(Clone, Debug)]
pub struct Dnf {
    /// Each entry is a conjunction, together making up the disjunctive normal form of the request
    /// predicates.
    pub conjunctions: Vec<Conjunction>,
}

/// An iterator which visits each virtual field that is part of a requests demand. Note that this
/// iterator may visit a virtual field multiple times.
#[derive(Clone)]
pub struct DNFRepeatingIterator<'a, 'b> {
    /// Which conjunction we are currently looking at
    conj_index: usize,
    /// An iterator for each conjunction
    iterators: Vec<MultiProduct<PredicateWithSchemaIntoIterator<'a, 'b>>>,
}

/// The basic building block for the disjunctive normal form which defines request predicates.
///
/// Is part of [Dnf]
#[derive(Clone, Debug)]
pub struct Conjunction {
    // TODO [later]: a small inefficiency is that each conjunction needs to contain a predicate for all attributes.
    // We could change this and store the "full" predicate once per schema.
    // However, this results to problems with lifetimes in Conjunction::prod_iter(..)
    /// A vector of predicates, which for each attribute defines the values that match to the given
    /// request.
    ///
    /// That each conjunction needs to contain a predicate for all attributes
    /// us a small inefficiency.
    /// We could change this and store the "full" predicate once per schema.
    /// However, this results to problems with lifetimes in Conjunction::prod_iter(..)
    predicates: Vec<Predicate>, // must always include a predicate for all values in schema
}

/// Preferred way to initialise conjunctions manually.
///
/// First, a [ConjunctionBuilder] should be initialized with [ConjunctionBuilder::new], then
/// new predicates should be appended with [ConjunctionBuilder::and], and then finally, the
/// conjunction should be built with [ConjunctionBuilder::build]
pub struct ConjunctionBuilder<'a> {
    schema: &'a Schema,
    predicates: HashMap<AttributeId, Predicate>,
}

/// A predicate defines a set of acceptable values for a certain attribute.
///
/// Note that the attribute to which a predicate belongs is not stored here, and is part of the
/// datastructure which the predicate is part of.
#[derive(Clone, Debug)]
pub enum Predicate {
    // TODO [later]: Expand with Gt, Le etc
    /// The predicate is only true if the attribute takes the given value.
    Eq(usize),
    /// The predicate is only true if the attribute does NOT take the given value
    Neq(usize),
    /// The predicate is only true if the attribute is between min and max (including both ends)
    Between { min: usize, max: usize }, //including both ends
    /// The predicate is only true if the attribute takes a value which is part of the given set.
    In(HashSet<usize>),
}

impl Predicate {
    /// This method "translates" a predicate from a [Request] to a predicate of an
    /// [external::ExternalRequest].
    ///
    /// This is useful if one wants to serialize a [Request], which requires transforming the it to
    /// an [external::ExternalRequest] first.
    fn to_external(
        &self,
        attribute_id: AttributeId,
        schema: &Schema,
    ) -> Result<external::Predicate, SchemaError> {
        match self {
            Predicate::Eq(x) => {
                let val = schema.attribute_value(attribute_id.0, *x)?;
                Ok(external::Predicate::Eq(val))
            }
            Predicate::Neq(x) => {
                let val = schema.attribute_value(attribute_id.0, *x)?;
                Ok(external::Predicate::Neq(val))
            }
            Predicate::Between { min, max } => {
                let min = schema.attribute_value(attribute_id.0, *min)?;
                let max = schema.attribute_value(attribute_id.0, *max)?;
                Ok(external::Predicate::Between { min, max })
            }
            Predicate::In(x) => Ok(external::Predicate::In(
                x.iter()
                    .map(|y| schema.attribute_value(attribute_id.0, *y))
                    .collect::<Result<HashSet<DataValue>, SchemaError>>()?,
            )),
        }
    }
}

/// This datastructure provides context for a predicate. Provides an
/// into_iter method to iterate over all values part of the schema for which
/// the predicate evaluates to true.
pub struct PredicateWithSchema<'a, 'b> {
    /// The id of the attribute to which the predicate belongs
    attribute_id: AttributeId,
    /// A reference to the predicate in question
    predicate: &'a Predicate,
    /// A reference to the schema which defines the current partitioning attributes.
    schema: &'b Schema,
}
