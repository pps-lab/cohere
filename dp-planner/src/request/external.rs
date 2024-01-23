use crate::dprivacy::rdp_alphas_accounting::RdpAlphas;
use crate::dprivacy::Accounting;
use crate::request::AdapterInfo;
use crate::simulation::RoundId;
use crate::{AccountingType, RequestAdapter};
use probability::distribution::Bernoulli;
use probability::distribution::Sample;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Error};
use std::path::PathBuf;

use super::super::schema;
use super::super::schema::DataValueLookup;
use super::{ConjunctionBuilder, RequestBuilder, RequestId};

/// ExternalRequest is the serialized format of [super::Request].
/// However, there also a few semantic
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ExternalRequest {
    pub request_id: RequestId,
    pub request_cost: Option<AccountingType>,
    pub profit: Option<u64>,
    pub dnf: Dnf,
    pub n_users: Option<usize>,
    // Note: Adapter info supplied from file will be ignored
    pub(crate) adapter_info: Option<AdapterInfo>,
    /// identifies the round the request joins the system
    pub created: Option<RoundId>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Dnf {
    pub(super) conjunctions: Vec<Conjunction>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Conjunction {
    pub(crate) predicates: HashMap<String, Predicate>, //Key: Name of attribute, value: Predicate on that attribute
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Predicate {
    // TODO [later]: Expand with Gt, Le etc
    Eq(schema::DataValue),
    Neq(schema::DataValue),
    // TODO [later] not sure whether min max should be doable with non number
    Between {
        min: schema::DataValue,
        max: schema::DataValue,
    }, //including both ends
    In(HashSet<schema::DataValue>), // Note: Can contain different variants of DataValue at the same time, e.g., String("xy") and Integer(3)
}

trait ExternalPredicate<'a> {
    fn to_internal(
        &self,
        attr_name: &str,
        schema: &schema::Schema,
    ) -> Result<(super::AttributeId, super::Predicate), schema::SchemaError>;
}

impl<'a> ExternalPredicate<'a> for Predicate {
    fn to_internal(
        &self,
        attr_name: &str,
        schema: &schema::Schema,
    ) -> Result<(super::AttributeId, super::Predicate), schema::SchemaError> {
        let attr_id = schema.attribute_id(attr_name)?;
        match self {
            Predicate::Eq(val) => {
                let val = schema.attribute_idx(attr_id, val)?;
                Ok((super::AttributeId(attr_id), super::Predicate::Eq(val)))
            }
            Predicate::Neq(val) => {
                let val = schema.attribute_idx(attr_id, val)?;
                Ok((super::AttributeId(attr_id), super::Predicate::Neq(val)))
            }
            Predicate::Between { min, max } => {
                let min = schema.attribute_idx(attr_id, min)?;
                let max = schema.attribute_idx(attr_id, max)?;
                Ok((
                    super::AttributeId(attr_id),
                    super::Predicate::Between { min, max },
                ))
            }
            Predicate::In(vals) => {
                let values: Result<HashSet<usize>, schema::SchemaError> = vals
                    .iter()
                    .map(|x| schema.attribute_idx(attr_id, x))
                    .collect();
                match values {
                    Ok(values) => Ok((super::AttributeId(attr_id), super::Predicate::In(values))),
                    Err(e) => Err(e),
                }
            }
        }
    }
}

impl ExternalRequest {
    /// Returns the converted request, and a list of attributes which could not be converted
    fn convert(&self, schema: &schema::Schema) -> (super::Request, Vec<String>) {
        assert!(
            self.n_users.is_some() && self.profit.is_some() && self.request_cost.is_some(),
            "A request did not have n_users, a profit or request_cost set"
        );

        let mut builder = RequestBuilder::new_full(
            self.request_id,
            self.request_cost.as_ref().unwrap().clone(),
            self.profit.unwrap(),
            self.n_users.unwrap(),
            self.created,
            schema,
            self.adapter_info
                .clone()
                .expect("External request did not have adapter_info"),
        );

        let mut missing_attributes: Vec<String> = Vec::new();
        // convert external dnf -> internal dnf
        for external_conjunction in self.dnf.conjunctions.iter() {
            let mut conj_builder = ConjunctionBuilder::new(schema);

            // convert predicates from external -> internal
            for (name, external_pred) in external_conjunction.predicates.iter() {
                match external_pred.to_internal(name, schema) {
                    Ok(sol) => {
                        let (attr_id, pred) = sol;
                        conj_builder = conj_builder.and(attr_id, pred);
                    }
                    Err(_) => missing_attributes.push(name.to_string()),
                }
            }

            builder = builder.or_conjunction(conj_builder.build());
        }

        (builder.build(), missing_attributes)
    }
}

pub fn parse_requests(filepath: PathBuf) -> Result<Vec<ExternalRequest>, Error> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let requests: Vec<ExternalRequest> = serde_json::from_reader(reader).expect("Parsing Failed");

    //println!("{:?}", request);

    Ok(requests)
}

pub fn convert_requests(
    mut requests: Vec<ExternalRequest>,
    schema: &schema::Schema,
    request_adapter: &mut RequestAdapter,
    alphas: &Option<RdpAlphas>,
) -> Result<HashMap<RequestId, super::Request>, schema::SchemaError> {
    // first, we apply the adapter to potentially "fill" any values that are still none
    request_adapter.apply(&mut requests, alphas);

    // then we convert it to internal representation
    let mut internal_requests: HashMap<RequestId, super::Request> = HashMap::new();
    let mut all_missing_attributes: BTreeSet<String> = BTreeSet::new();

    // bernoulli distribution with p=1/no_inverse_frac -> approx every no_inverse_frac request will have no pa
    let bernoulli = request_adapter
        .get_no_pa_inverse_frac()
        .map(|inverse_frac| Bernoulli::new(1.0 / inverse_frac as f64));

    for r in requests.into_iter() {
        let (mut converted, missing_attributes) = r.convert(schema);

        if let Some(bernoulli) = bernoulli {
            // roughly every inverse_frac-th request does not have any dnf
            // -> sample from bernoulli
            if bernoulli.sample(&mut request_adapter.source) == 1 {
                converted.dnf = super::Dnf {
                    conjunctions: vec![ConjunctionBuilder::new(schema).build()],
                }
            }
        }

        let inserted = internal_requests.insert(converted.request_id, converted);
        assert!(inserted.is_none());
        all_missing_attributes.extend(missing_attributes.into_iter());
    }

    // finally, check if cost in schema and the requests is of the same type
    assert!(
        internal_requests
            .values()
            .all(|req| schema.accounting_type.check_same_type(&req.request_cost)),
        "Request and schema have different types of DP. Schema: {}, first request: {}",
        schema.accounting_type,
        internal_requests.values().next().unwrap().request_cost
    );

    if !all_missing_attributes.is_empty() {
        println!(
            "Requests had attributes which were not part of schema: {:?}",
            all_missing_attributes
        )
    }

    Ok(internal_requests)
}

#[cfg(test)]
mod tests {
    use crate::dprivacy::rdp_alphas_accounting::RdpAlphas::*;
    use crate::request::{external::parse_requests, load_requests, resource_path};
    use crate::AccountingType::{EpsDp, Rdp};
    use crate::{AccountingType, RequestAdapter};
    use std::path::PathBuf;
    use std::str::FromStr;

    static DEMO_REQUESTS: &str = "request_files/demo_requests.json";
    static CENSUS_REQUESTS: &str = "request_files/census_requests.json";
    static DEMO_SCHEMA: &str = "schema_files/demo_schema.json";
    static CENSUS_SCHEMA: &str = "schema_files/census_schema.json";
    static REQUEST_DIRECTORY: &str = "./resources/test/request_files/";
    static SEED: u128 = 1848;
    static RDP_FIVE_ALPHAS: AccountingType = Rdp {
        eps_values: A5([0f64; 5]),
    };

    #[test]
    fn test_parse_demo_requests() {
        let demo_requests = parse_requests(resource_path(DEMO_REQUESTS));
        assert!(demo_requests.is_ok());
    }

    #[test]
    fn test_parse_census_requests() {
        let census_requests = parse_requests(resource_path(CENSUS_REQUESTS));
        assert!(census_requests.is_ok());
    }

    #[test]
    fn test_convert_demo_requests() {
        let demo_schema =
            crate::schema::load_schema(resource_path(DEMO_SCHEMA), &RDP_FIVE_ALPHAS).unwrap();
        let converted_request = load_requests(
            resource_path(DEMO_REQUESTS),
            &demo_schema,
            &mut RequestAdapter::get_empty_adapter(),
            &None,
        );
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_add_cost_later() {
        let demo_schema =
            crate::schema::load_schema(resource_path(DEMO_SCHEMA), &EpsDp { eps: 0.0 }).unwrap();

        let requests_path_buf =
            PathBuf::from_str(&(REQUEST_DIRECTORY.to_owned() + "demo_requests_no_cost.json"))
                .expect("Constructing PathBuf Failed");
        let adapter_path_buf =
            PathBuf::from_str("./resources/test/adapter_configs/set_cost_only.json")
                .expect("Constructing PathBuf Failed");
        let mut adapter = RequestAdapter::new(adapter_path_buf, SEED);
        let converted_request = load_requests(requests_path_buf, &demo_schema, &mut adapter, &None);
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    #[should_panic(expected = "A request did not have n_users, a profit or request_cost set")]
    fn test_no_cost_added() {
        let demo_schema =
            crate::schema::load_schema(resource_path(DEMO_SCHEMA), &RDP_FIVE_ALPHAS).unwrap();

        let requests_path_buf =
            PathBuf::from_str(&(REQUEST_DIRECTORY.to_owned() + "demo_requests_no_cost.json"))
                .expect("Constructing PathBuf Failed");
        let adapter_path_buf =
            PathBuf::from_str("./resources/test/adapter_configs/set_profit_only.json")
                .expect("Constructing PathBuf Failed");
        let mut adapter = RequestAdapter::new(adapter_path_buf, SEED);
        let converted_request = load_requests(requests_path_buf, &demo_schema, &mut adapter, &None);
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_add_profit_later() {
        let demo_schema =
            crate::schema::load_schema(resource_path(DEMO_SCHEMA), &RDP_FIVE_ALPHAS).unwrap();

        let requests_path_buf =
            PathBuf::from_str(&(REQUEST_DIRECTORY.to_owned() + "demo_requests_no_profit.json"))
                .expect("Constructing PathBuf Failed");
        let adapter_path_buf =
            PathBuf::from_str("./resources/test/adapter_configs/set_profit_only.json")
                .expect("Constructing PathBuf Failed");
        let mut adapter = RequestAdapter::new(adapter_path_buf, SEED);
        let converted_request = load_requests(requests_path_buf, &demo_schema, &mut adapter, &None);
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_add_nusers_later() {
        let demo_schema =
            crate::schema::load_schema(resource_path(DEMO_SCHEMA), &RDP_FIVE_ALPHAS).unwrap();

        let requests_path_buf =
            PathBuf::from_str(&(REQUEST_DIRECTORY.to_owned() + "demo_requests_no_nusers.json"))
                .expect("Constructing PathBuf Failed");
        let adapter_path_buf =
            PathBuf::from_str("./resources/test/adapter_configs/set_nblocks_only.json")
                .expect("Constructing PathBuf Failed");
        let mut adapter = RequestAdapter::new(adapter_path_buf, SEED);
        let converted_request = load_requests(requests_path_buf, &demo_schema, &mut adapter, &None);
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_add_nusers_profit_and_cost_later() {
        let demo_schema =
            crate::schema::load_schema(resource_path(DEMO_SCHEMA), &EpsDp { eps: 0.0 }).unwrap();

        let requests_path_buf = PathBuf::from_str(
            &(REQUEST_DIRECTORY.to_owned() + "demo_requests_no_profit_cost_nusers.json"),
        )
        .expect("Constructing PathBuf Failed");
        let adapter_path_buf =
            PathBuf::from_str("./resources/test/adapter_configs/set_profit_cost_and_nblocks.json")
                .expect("Constructing PathBuf Failed");
        let mut adapter = RequestAdapter::new(adapter_path_buf, SEED);
        let converted_request = load_requests(requests_path_buf, &demo_schema, &mut adapter, &None);
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    #[should_panic(expected = "A request did not have n_users, a profit or request_cost set")]
    fn test_dont_add_cost_and_profit() {
        let demo_schema =
            crate::schema::load_schema(resource_path(DEMO_SCHEMA), &RDP_FIVE_ALPHAS).unwrap();

        let requests_path_buf = PathBuf::from_str(
            &(REQUEST_DIRECTORY.to_owned() + "demo_requests_no_profit_cost_nusers.json"),
        )
        .expect("Constructing PathBuf Failed");
        let adapter_path_buf =
            PathBuf::from_str("./resources/test/adapter_configs/set_nblocks_only.json")
                .expect("Constructing PathBuf Failed");
        let mut adapter = RequestAdapter::new(adapter_path_buf, SEED);
        let converted_request = load_requests(requests_path_buf, &demo_schema, &mut adapter, &None);
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_convert_census_requests() {
        let census_schema = crate::schema::load_schema(
            resource_path(CENSUS_SCHEMA),
            &Rdp {
                eps_values: A13([0.; 13]),
            },
        )
        .unwrap();
        let converted_request = load_requests(
            resource_path(CENSUS_REQUESTS),
            &census_schema,
            &mut RequestAdapter::get_empty_adapter(),
            &None,
        );
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }

    #[test]
    #[should_panic(expected = "Request and schema have different types of DP")]
    fn test_convert_census_requests_wrong_budget() {
        let census_schema = crate::schema::load_schema(
            resource_path(CENSUS_SCHEMA),
            &Rdp {
                eps_values: A10([0.; 10]),
            },
        )
        .unwrap();
        let converted_request = load_requests(
            resource_path(CENSUS_REQUESTS),
            &census_schema,
            &mut RequestAdapter::get_empty_adapter(),
            &None,
        );
        assert!(
            converted_request.is_ok(),
            "{}",
            converted_request.err().unwrap().to_string()
        );
    }
}
