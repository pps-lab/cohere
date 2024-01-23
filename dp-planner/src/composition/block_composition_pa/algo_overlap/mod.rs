// TODO [nku] [later] consider new segmentation algorithm that splits conjunctions

//struct SplitResult {
//    a_only: Vec<Conjunction>,
//    b_only: Vec<Conjunction>,
//    ab_both: Vec<Conjunction>,
//}
//
//
//use std::cmp::Ordering;
//
//use crate::{
//    request::internal::{Conjunction, Predicate},
//    schema::{Attribute, Schema},
//};
//
//impl PartialEq for Predicate {
//    fn eq(&self, other: &Predicate) -> bool {
//        match (self, other) {
//            (Predicate::Eq(v1), Predicate::Eq(v2)) => v1 == v2,
//            (Predicate::Neq(v1), Predicate::Neq(v2)) => v1 == v2,
//            (
//                Predicate::Between {
//                    min: min1,
//                    max: max1,
//                },
//                Predicate::Between {
//                    min: min2,
//                    max: max2,
//                },
//            ) => min1 == min2 && max1 == max2,
//            (Predicate::In(set1), Predicate::In(set2)) => set1.eq(set2),
//            // TODO [nku] [later] considering schema there could be some theoretical eq (neq(0) == between(1, max-1))
//            _ => false,
//        }
//    }
//}
//
//impl PartialOrd for Predicate {

//
//    // idea is to introduce an ordering between enum variants of predicate eq < neq < between < in
//    fn partial_cmp(&self, other: &Predicate) -> Option<Ordering> {
//        match (self, other) {
//            (Predicate::Eq(v1), Predicate::Eq(v2)) => v1.partial_cmp(v2),
//            (Predicate::Eq(_), _) => Some(Ordering::Less),
//            (_, Predicate::Eq(_)) => Some(Ordering::Greater),
//            (Predicate::Neq(v1), Predicate::Neq(v2)) => v1.partial_cmp(v2),
//            (Predicate::Neq(_), _) => Some(Ordering::Less),
//            (_, Predicate::Neq(_)) => Some(Ordering::Greater),
//            (
//                Predicate::Between {
//                    min: min1,
//                    max: max1,
//                },
//                Predicate::Between {
//                    min: min2,
//                    max: max2,
//                },
//            ) if min1 == min2 && max1 == max2 => Some(Ordering::Equal),
//            (
//                Predicate::Between { min: min1, max: _ },
//                Predicate::Between { min: min2, max: _ },
//            ) if min1 != min2 => min1.partial_cmp(min2),
//            (
//                Predicate::Between {
//                    min: min1,
//                    max: max1,
//                },
//                Predicate::Between {
//                    min: min2,
//                    max: max2,
//                },
//            ) if min1 == min2 && max1 != max2 => max1.partial_cmp(max2),
//            (Predicate::Between { min: _, max: _ }, _) => Some(Ordering::Less),
//            (_, Predicate::Between { min: _, max: _ }) => Some(Ordering::Greater),
//            (Predicate::In(set1), Predicate::In(set2)) if set1.eq(set2) => Some(Ordering::Equal),
//            (Predicate::In(_), _) => Some(Ordering::Less),
//        }
//    }
//}
//
//fn is_overlap_predicate(p1: &Predicate, p2: &Predicate, schema_attribute: &Attribute) -> bool {
//    let mut pred_tuple = (p1, p2);
//
//    if let Some(Ordering::Greater) = p1.partial_cmp(p2) {
//        pred_tuple = (p2, p1);
//    }
//
//    match pred_tuple {
//        (Predicate::Eq(v1), Predicate::Eq(v2)) => v1 == v2,
//        (Predicate::Eq(v1), Predicate::Neq(v2)) => v1 != v2,
//        (Predicate::Eq(v1), Predicate::Between { min, max }) => v1 >= min && v1 <= max,
//        (Predicate::Eq(v1), Predicate::In(set)) => set.contains(v1),
//
//        (Predicate::Neq(v1), Predicate::Neq(v2)) => {
//            match schema_attribute.len() {
//                0 | 1 => panic!("schema attribute with neq needs more than 1 possible value"),
//                2 => v1 == v2, // for attribute with two possible values (case v1==v2 => they overlap, case v1!=v2 -> they don't overlap)
//                _ => true, // for larger domain  > 2: neq cannot be non-overlapping because we can only exclude one
//            }
//        }
//        (Predicate::Neq(v1), Predicate::Between { min, max }) => panic!("need schema"), // TODO [nku] [later]: need schema
//        (Predicate::Neq(v1), Predicate::In(set)) => { match schema_attribute.len() {
//            0 | 1 => panic!("schema attribute with neq needs more than 1 possible value"),
//
//            // TODO [nku] [later] should it be the set len()?
//            //match schema_attribute.len() {
//            //    0 | 1 => panic!("schema attribute with neq needs more than 1 possible value"),
//            //    2 => ,
//            //    _ => true
//
//        }panic!("need schema"), // TODO [nku] [later]: need schema
//
//        // we can assume ordering between between ranges
//        (Predicate::Between { min: _, max: max1 }, Predicate::Between { min: min2, max: _ }) => {
//            min2 >= max1
//        }
//
//        (Predicate::Between { min, max }, Predicate::In(set)) => {
//            set.iter().any(|x| x >= min && x <= max)
//        }
//        (Predicate::In(set1), Predicate::In(set2)) => set1.iter().any(|x| set2.contains(x)),
//        _ => panic!("unexpected combination in predicate overlap"),
//    }
//}
//
//fn is_overlap(c1: &Conjunction, c2: &Conjunction) -> bool {
//    false
//}
//
//fn split(c1: &Conjunction, c2: &Conjunction) -> SplitResult {
//    SplitResult {
//        a_only: Vec::new(),
//        b_only: Vec::new(),
//        ab_both: Vec::new(),
//    }
//}
//
