use super::super::schema::Schema;
use super::{
    AttributeId, Conjunction, ConjunctionBuilder, DNFRepeatingIterator, Predicate,
    PredicateWithSchema,
};
use crate::request::external::ExternalRequest;
use crate::request::{external, Dnf};
use crate::schema::DataValueLookup;
use crate::Request;
use itertools::{Itertools, MultiProduct};
use std::collections::HashMap;

/// An iterator which iterates over all possible values of the specified attribute
/// which are legal according to the current schema and for which the predicate evaluates to true.
#[derive(Clone)]
pub struct PredicateWithSchemaIntoIterator<'a, 'b> {
    /// The id of the attribute in question
    attribute_id: AttributeId,
    /// The predicate in question. Note that this predicate may allow more values than contained
    /// in the schema
    predicate: &'a Predicate,
    /// The schema containing the current partitioning attributes.
    schema: &'b Schema,
    /// Used to iterate over values for certain predicates, e.g., [Predicate::Neq]
    index: usize,
    /// Used to iterate over the predicate if it's a set predicate.
    iter: Option<std::collections::hash_set::Iter<'a, usize>>,
}

impl<'a, 'b> PredicateWithSchema<'a, 'b> {
    /// Construct a new [PredicateWithSchema]
    pub fn new(
        attribute_id: AttributeId,
        predicate: &'a Predicate,
        schema: &'b Schema,
    ) -> PredicateWithSchema<'a, 'b> {
        // TODO [later]: we could verify here that the predicate is compatible with the schema (basically -> avoid out of bounds etc.) -> but maybe this should be done on the conversion on the internal representation
        PredicateWithSchema {
            attribute_id,
            predicate,
            schema,
        }
    }
}

impl<'a, 'b> IntoIterator for &PredicateWithSchema<'a, 'b> {
    type Item = usize;
    type IntoIter = PredicateWithSchemaIntoIterator<'a, 'b>;

    fn into_iter(self) -> Self::IntoIter {
        let iter = match self.predicate {
            Predicate::In(values, ..) => Some(values.iter()),
            _ => None,
        };

        PredicateWithSchemaIntoIterator {
            attribute_id: self.attribute_id,
            predicate: self.predicate,
            schema: self.schema,
            index: 0,
            iter,
        }
    }
}

impl<'a, 'b> Iterator for PredicateWithSchemaIntoIterator<'a, 'b> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let next = match (&self.predicate, self.index, &mut self.iter) {
            (Predicate::Eq(val, ..), 0, None) => Some(*val),
            (Predicate::Neq(val, ..), idx, None) => {
                let attribute = self
                    .schema
                    .attributes
                    .get(self.attribute_id.0)
                    .expect("missing attribute_id");
                let size = attribute.len();

                if idx == *val {
                    // skip and increase by one
                    self.index += 1;
                }

                if self.index < size {
                    Some(self.index)
                } else {
                    None
                }
            }
            (Predicate::Between { min, max, .. }, idx, None) if *min + idx <= *max => {
                Some(min + idx)
            }
            (Predicate::In { .. }, _idx, Some(iter)) => iter.next().copied(),
            _ => None,
        };

        self.index += 1;

        next
    }
}

impl<'a> ConjunctionBuilder<'a> {
    /// Initializes a new ConjunctionBuilder.
    ///
    /// See [ConjunctionBuilder] for how this is used to instantiate a conjunction.
    pub fn new(schema: &'a Schema) -> ConjunctionBuilder {
        ConjunctionBuilder {
            schema,
            predicates: HashMap::new(),
        }
    }

    /// Add a new predicate to the given [ConjunctionBuilder].
    ///
    /// See [ConjunctionBuilder] for more information on how to instantiate a Conjunction.
    pub fn and(mut self, attr_id: AttributeId, p: Predicate) -> ConjunctionBuilder<'a> {
        self.predicates.insert(attr_id, p);
        self
    }

    /// Finalizes and returns the Conjunction from the given [ConjunctionBuilder]
    ///
    /// See [ConjunctionBuilder] for more information on how to instantiate a Conjunction.
    pub fn build(self) -> Conjunction {
        Conjunction::new(self.schema, &self.predicates)
    }
}

impl Conjunction {
    /// Instantiate a new conjunction using the given schema and the mappig from
    /// attribute ids to predicates.
    fn new(schema: &Schema, predicates: &HashMap<AttributeId, Predicate>) -> Conjunction {
        let predicates = schema
            .attributes
            .iter()
            .enumerate()
            .map(
                |(attr_id, attr)| match predicates.get(&AttributeId(attr_id)) {
                    // TODO [later]: there might be a more elegant way to pass predicates by reference and then clone them here
                    Some(predicate) => predicate.clone(),
                    None => Predicate::Between {
                        min: 0,
                        max: attr.len() - 1,
                    },
                },
            )
            .collect();

        Conjunction { predicates }
    }

    /// Returns an Iterator that iterates over all combinations of attribute values allowed by the
    /// current conjunction and schema. Note that in contrast to [Dnf::repeating_iter], there should
    /// be no repeating elements in this iterator.
    pub fn prod_iter<'a, 'b>(
        &'a self,
        schema: &'b Schema,
    ) -> MultiProduct<PredicateWithSchemaIntoIterator<'a, 'b>> {
        self.predicates
            .iter()
            .enumerate()
            .map(|(attr_id, pred)| {
                PredicateWithSchema::new(AttributeId(attr_id), pred, schema).into_iter()
            })
            .into_iter()
            .multi_cartesian_product()
    }

    /// Returns the predicates which are part of the conjunction.
    pub fn predicates(&self) -> &Vec<Predicate> {
        &self.predicates
    }
}

impl<'a, 'b> Iterator for DNFRepeatingIterator<'a, 'b> {
    type Item = Vec<usize>;
    fn next(&mut self) -> Option<Self::Item> {
        while self.conj_index < self.iterators.len() {
            match self.iterators[self.conj_index].next() {
                None => {
                    // end of iterator of one conjunction
                    self.conj_index += 1; // switch to iterator of next conjunction
                }
                Some(x) => {
                    // case we found element to return (break out of loop)
                    return Some(x);
                }
            }
        }

        // end of DNFRepeatingIterator
        None
    }
}

impl Predicate {
    /// Checks if the given value is contained in theself given predicate.
    pub fn contains(&self, idx: &usize) -> bool {
        match &self {
            Predicate::Eq(v) => v == idx,
            Predicate::Neq(v) => v != idx,
            Predicate::Between { min, max } => idx >= min && idx <= max, //including both ends
            Predicate::In(values) => values.contains(idx),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::request::RequestId;
    use crate::request::{Dnf, Request};

    use super::*;

    use super::super::super::schema::*;

    use super::super::super::dprivacy::*;

    use std::collections::{HashMap, HashSet};

    fn build_single_range_attribute_schema(n: usize) -> Schema {
        let mut schema = Schema {
            accounting_type: AccountingType::EpsDp { eps: 1.0 },
            attributes: vec![Attribute {
                name: "demo".to_owned(),
                value_domain: ValueDomain::Range {
                    min: 0,
                    max: (n - 1) as isize,
                },
                value_domain_map: None,
            }],
            name_to_index: HashMap::new(),
        };

        schema.init();

        schema
    }

    fn build_single_collection_attribute_schema(n: usize) -> Schema {
        let data = (0..n).map(|i| DataValue::String(i.to_string())).collect();

        let mut schema = Schema {
            accounting_type: AccountingType::EpsDp { eps: 1.0 },
            attributes: vec![Attribute {
                name: "demo".to_owned(),
                value_domain: ValueDomain::Collection(data),
                value_domain_map: None,
            }],
            name_to_index: HashMap::new(),
        };

        schema.init();

        schema
    }

    fn build_multi_attribute_schema(attribute_sizes: Vec<usize>) -> Schema {
        let attributes = attribute_sizes
            .iter()
            .enumerate()
            .map(|(idx, size)| Attribute {
                name: idx.to_string(),
                value_domain: ValueDomain::Range {
                    min: 0,
                    max: (size - 1) as isize,
                },
                value_domain_map: None,
            })
            .collect();

        let mut schema = Schema {
            accounting_type: AccountingType::EpsDp { eps: 1.0 },
            attributes,
            name_to_index: HashMap::new(),
        };

        schema.init();
        schema
    }

    fn build_base_request(
        schema: &Schema,
        conjunctions_desc: &[&HashMap<usize, Vec<usize>>],
    ) -> Request {
        let mut conjunctions = Vec::new();

        for map in conjunctions_desc {
            let predicates = map
                .iter()
                .map(|(attr_id, values)| {
                    let values_set: HashSet<usize> = values.iter().copied().collect(); // convert to HashSet
                    if values.len() != values_set.len() {
                        panic!("no support for duplicates in values");
                    }

                    println!("attr_id={:?}   values set = {:?}", attr_id, values_set);

                    (AttributeId(*attr_id), Predicate::In(values_set))
                })
                .collect();

            conjunctions.push(Conjunction::new(schema, &predicates));
        }

        Request {
            request_id: RequestId(0),
            request_cost: schema.accounting_type.clone(),
            unreduced_cost: schema.accounting_type.clone(),
            profit: 1,
            dnf: Dnf { conjunctions },
            n_users: 1,
            created: None,
            adapter_info: Default::default(),
        }
    }

    #[test]
    fn test_dnf_repeating_iter() {
        let schema = build_multi_attribute_schema(vec![3, 3, 3]);

        // a0 \in {0, 1}   a1 \in {0, 1, 2}    a2 \in {1, 2}
        let mut map = HashMap::new();
        map.insert(0, vec![0, 1]);
        map.insert(1, vec![0, 1, 2]);
        map.insert(2, vec![1, 2]);

        // request with a single conjunction
        let r1 = build_base_request(&schema, &[&map]);
        let mut actual: Vec<Vec<usize>> = r1.dnf.repeating_iter(&schema).collect();
        let actual_num = r1.dnf.num_virtual_blocks(&schema);
        assert_eq!(12, actual_num);

        let mut expected = Vec::new();
        for a0 in map.get(&0).unwrap() {
            for a1 in map.get(&1).unwrap() {
                for a2 in map.get(&2).unwrap() {
                    expected.push(vec![*a0, *a1, *a2]);
                }
            }
        }

        actual.sort();
        expected.sort();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_eq_iterator() {
        let schemas = vec![
            build_single_range_attribute_schema(10),
            build_single_collection_attribute_schema(10),
        ];
        for schema in schemas.iter() {
            let p1 = Predicate::Eq(5);
            let mut iter1 = PredicateWithSchema::new(AttributeId(0), &p1, schema).into_iter();
            assert_eq!(iter1.next(), Some(5));
            assert_eq!(iter1.next(), None);
        }
    }

    #[test]
    fn test_neq_iterator() {
        let schemas = vec![
            build_single_range_attribute_schema(5),
            build_single_collection_attribute_schema(5),
        ];
        for schema in schemas.iter() {
            // start
            let p_start = Predicate::Neq(0);
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p_start, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![1, 2, 3, 4]);

            // middle
            let p_mid = Predicate::Neq(3);
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p_mid, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![0, 1, 2, 4]);

            // last
            let p_mid = Predicate::Neq(4);
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p_mid, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn test_between_iterator() {
        let schemas = vec![
            build_single_range_attribute_schema(10),
            build_single_collection_attribute_schema(10),
        ];
        for schema in schemas.iter() {
            // start
            let p = Predicate::Between { min: 0, max: 4 };
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![0, 1, 2, 3, 4]);

            // middle
            let p = Predicate::Between { min: 1, max: 3 };
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![1, 2, 3]);

            // end
            let p = Predicate::Between { min: 8, max: 9 };
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![8, 9]);

            // single
            let p = Predicate::Between { min: 1, max: 1 };
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![1]);

            // single end
            let p = Predicate::Between { min: 9, max: 9 };
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![9]);

            // full
            let p = Predicate::Between { min: 0, max: 9 };
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

            // malformed
            let p = Predicate::Between { min: 2, max: 1 };
            let mut iter = PredicateWithSchema::new(AttributeId(0), &p, schema).into_iter();
            assert_eq!(iter.next(), None);

            // out of range -> expected behaviour is that there is no error (we don't check the schema at the moment)
            let p = Predicate::Between { min: 20, max: 23 };
            let res: Vec<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            assert_eq!(res, vec![20, 21, 22, 23]);
        }
    }

    #[test]
    fn test_in_iterator() {
        let schemas = vec![
            build_single_range_attribute_schema(10),
            build_single_collection_attribute_schema(10),
        ];
        for schema in schemas.iter() {
            // single value
            let p = Predicate::In(vec![4].into_iter().collect());
            let res: HashSet<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            let expected: HashSet<usize> = vec![4].into_iter().collect();
            assert_eq!(res, expected);

            // some values
            let p = Predicate::In(vec![1, 5, 7].into_iter().collect());
            let res: HashSet<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            let expected: HashSet<usize> = vec![1, 5, 7].into_iter().collect();
            assert_eq!(res, expected);

            // all values
            let p = Predicate::In(vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0].into_iter().collect());
            let res: HashSet<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            let expected: HashSet<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_iter().collect();
            assert_eq!(res, expected);

            // out of range -> at the moment there is no schema check => expected that it returns all of them
            let p = Predicate::In(vec![100, 34].into_iter().collect());
            let res: HashSet<usize> = PredicateWithSchema::new(AttributeId(0), &p, schema)
                .into_iter()
                .collect();
            let expected: HashSet<usize> = vec![34, 100].into_iter().collect();
            assert_eq!(res, expected);
        }
    }
}

impl Request {
    pub fn unreduce_alphas(&mut self) {
        self.request_cost = self.unreduced_cost.clone();
    }

    pub fn dnf(&self) -> &Dnf {
        &self.dnf
    }

    /// Converts an internal request to external representation, to enable serialization
    ///
    /// Note: The request cost used is the unreduced request cost, otherwise requests
    /// converted back to external couldn't be combined with other external requests anymore.
    pub fn to_external(&self, schema: &Schema) -> ExternalRequest {
        ExternalRequest {
            request_id: self.request_id,
            request_cost: Some(self.unreduced_cost.clone()),
            profit: Some(self.profit),
            dnf: external::Dnf {
                conjunctions: self
                    .dnf
                    .conjunctions
                    .iter()
                    .map(|conj| {
                        let predicates: HashMap<String, external::Predicate> = conj
                            .predicates
                            .iter()
                            .enumerate()
                            .map(|(attr_id, pred)| {
                                (
                                    schema
                                        .attribute_name(attr_id)
                                        .expect("Couldn't get attribute name")
                                        .to_string(),
                                    pred.to_external(AttributeId(attr_id), schema)
                                        .expect("Couldn't convert predicate to external"),
                                )
                            })
                            .collect();

                        external::Conjunction { predicates }
                    })
                    .collect(),
            },
            n_users: Some(self.n_users),
            adapter_info: Some(self.adapter_info.clone()),
            created: self.created,
        }
    }
}
