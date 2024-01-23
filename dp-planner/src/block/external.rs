use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Error, ErrorKind};
use std::path::PathBuf;

use crate::dprivacy::{Accounting, AccountingType};
use crate::simulation::RoundId;

use super::BlockId;

use crate::request::{Request, RequestId};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct ExternalBlock {
    pub(crate) id: BlockId,
    /// the requests which applied to this block
    pub(crate) request_ids: Vec<RequestId>,
    /// the budget that is unlocked for this block - note that this ignores the cost of executed
    /// requests but may take into account the number of executed requests
    pub(crate) unlocked_budget: AccountingType, // ignores cost of request history -> i.e., needs to be subtracted
    /// round when this block joins the system
    pub(crate) created: RoundId,
    /// round when this block leaves the system (i.e., is retired)
    pub(crate) retired: Option<RoundId>,
}

pub(super) fn load_and_convert_blocks(
    request_history: &HashMap<RequestId, Request>,
    filepath: PathBuf,
) -> Result<HashMap<BlockId, super::Block>, Error> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    let blocks: Vec<ExternalBlock> = serde_json::from_reader(reader)?;

    // check that all blocks have the same budget type
    let first_budget = blocks[0].unlocked_budget.clone();
    if blocks
        .iter()
        .any(|block| !block.unlocked_budget.check_same_type(&first_budget))
    {
        return Err(std::io::Error::new(
            ErrorKind::Other,
            "blocks have incompatible budgets",
        ));
    }

    // check that there are no duplicate block ids
    let sorted_ids = blocks
        .iter()
        .map(|block| block.id)
        .sorted()
        .collect::<Vec<_>>();

    let deduped = {
        let mut temp = sorted_ids.clone();
        temp.dedup();
        temp
    };

    if deduped.len() != sorted_ids.len() {
        return Err(std::io::Error::new(ErrorKind::Other, "duplicate block ids"));
    }

    // check that all requests in each blocks history are actually in request_history
    for block in blocks.iter() {
        for request_id in block.request_ids.iter() {
            assert!(
                request_history.contains_key(request_id),
                "A request in a blocks request history was not present in overall request history"
            );
        }
    }

    Ok(blocks
        .into_iter()
        .map(|external_block| {
            (
                external_block.id,
                super::Block {
                    id: external_block.id,
                    request_history: external_block.request_ids,
                    unlocked_budget: external_block.unlocked_budget.clone(),
                    unreduced_unlocked_budget: external_block.unlocked_budget,
                    created: external_block.created,
                    retired: external_block.retired,
                },
            )
        })
        .collect())
}

#[cfg(test)]

mod tests {
    use crate::block::BlockId::User;
    use crate::request::RequestId;
    use crate::util::{build_dummy_requests_with_pa, build_dummy_schema};
    use crate::AccountingType::EpsDp;
    use crate::{BlockId, RoundId};
    use std::path::PathBuf;

    #[test]
    fn test_loading_blocks_1() {
        let schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            PathBuf::from("resources/test/block_files/block_test_1.json"),
        );
        assert!(blocks.is_ok(), "{:?}", blocks.err());
        let block = &blocks.unwrap()[&BlockId::User(1)];
        assert_eq!(block.created, RoundId(0));
        assert_eq!(block.unlocked_budget, EpsDp { eps: 1.0 });
        assert_eq!(block.id, User(1));
        assert_eq!(
            block.request_history,
            vec![RequestId(1), RequestId(2), RequestId(3)]
        )
    }

    #[test]
    fn test_loading_blocks_2() {
        let schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            PathBuf::from("resources/test/block_files/block_test_2.json"),
        );
        assert!(blocks.is_ok(), "{:?}", blocks.err());
        assert_eq!(blocks.unwrap().len(), 3);
    }

    #[test]
    #[should_panic]
    fn test_loading_blocks_invalid_request_id() {
        let schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            PathBuf::from("resources/test/block_files/block_test_invalid_request_ids.json"),
        );
        assert!(blocks.is_ok());
    }

    #[test]
    #[should_panic]
    fn test_loading_blocks_incompatible_budgets() {
        let schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            PathBuf::from("resources/test/block_files/block_test_incompatible_budgets.json"),
        );
        assert!(blocks.is_ok());
    }

    #[test]
    #[should_panic]
    fn test_loading_blocks_duplicate_ids() {
        let schema = build_dummy_schema(EpsDp { eps: 1.0 });
        let requests = build_dummy_requests_with_pa(&schema, 1, EpsDp { eps: 0.4 }, 6);
        let blocks = super::load_and_convert_blocks(
            &requests,
            PathBuf::from("resources/test/block_files/block_test_duplicate_ids.json"),
        );
        assert!(blocks.is_ok());
    }
}
