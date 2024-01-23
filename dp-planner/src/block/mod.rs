use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::block::external::load_and_convert_blocks;
use crate::dprivacy::rdp_alphas_accounting::RdpAlphas;
use crate::dprivacy::{Accounting, AdpAccounting};
use crate::schema::Schema;
use crate::simulation::RoundId;
use crate::{dprivacy::AccountingType, request::Request, RequestId};

pub(crate) mod external;

#[derive(Debug, Clone, Serialize)]
pub struct Block {
    /// a unique identifier for this block, sometimes referred to as privacy id
    pub id: BlockId,
    /// which requests where executed on this block
    pub request_history: Vec<RequestId>,
    /// the budget that is unlocked for this block - note that this ignores the cost of executed
    /// requests but may take into account the number of executed requests
    pub unlocked_budget: AccountingType,
    /// Same budget as above, but not affected by global alpha reduction. Needed if the blocks
    /// are serialized, since then want unreduced budget (same type of dp as input)
    pub unreduced_unlocked_budget: AccountingType,
    /// round when this block joins the system
    pub created: RoundId,
    /// round when this block leaves the system (i.e., is retired)
    pub retired: Option<RoundId>,
}

#[derive(PartialOrd, Ord, Hash, Eq, PartialEq, Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(untagged)]
pub enum BlockId {
    User(usize),
    UserTime(usize, usize),
}

impl Block {
    pub fn unreduce_alphas(&mut self) {
        self.unlocked_budget = self.unreduced_unlocked_budget.clone();
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BlockId::User(uid) => {
                write!(f, "{}", uid)
            }
            BlockId::UserTime(uid, tid) => {
                write!(f, "({},{})", uid, tid)
            }
        }
    }
}

/// Loads blocks from the specified pathbuf, and converts the budgets to rdp if alphas are set.
/// Returns both the blocks as hashmap
pub fn load_blocks(
    filepath: PathBuf,
    request_history: &HashMap<RequestId, Request>,
    schema: &Schema,
    alphas: &Option<RdpAlphas>,
) -> Result<HashMap<BlockId, Block>, std::io::Error> {
    let mut blocks = load_and_convert_blocks(request_history, filepath)?;

    // if conversion to rdp is desired
    if let Some(alpha_vals) = alphas {
        for block in blocks.values_mut() {
            block.unlocked_budget = block.unlocked_budget.adp_to_rdp_budget(alpha_vals);
            block.unreduced_unlocked_budget = block
                .unreduced_unlocked_budget
                .adp_to_rdp_budget(alpha_vals);
        }
    }

    assert!(
        blocks
            .values()
            .all(|bl| schema.accounting_type.check_same_type(&bl.unlocked_budget)),
        "Blocks did not have same type of DP as schema"
    );
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use crate::dprivacy::Accounting;
    use crate::dprivacy::AccountingType::EpsDp;
    use crate::util::{build_dummy_requests_with_pa, build_dummy_schema};
    use crate::AccountingType;
    use crate::AccountingType::EpsDeltaDp;
    use std::path::PathBuf;

    #[test]
    #[should_panic(expected = "Blocks did not have same type of DP as schema")]
    fn test_loading_blocks_wrong_budget() {
        let mut schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let requests = build_dummy_requests_with_pa(
            &schema,
            1,
            EpsDeltaDp {
                eps: 0.4,
                delta: 0.4,
            },
            6,
        );
        schema.accounting_type = AccountingType::zero_clone(&EpsDeltaDp {
            eps: 0.4,
            delta: 0.4,
        });

        let _blocks = super::load_blocks(
            PathBuf::from("resources/test/block_files/block_test_1.json"),
            &requests,
            &schema,
            &None,
        );
    }
}
