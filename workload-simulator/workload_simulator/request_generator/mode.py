from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

from workload_simulator.block_generator.block import Block, UserUpdateInfo
from workload_simulator.request_generator.request import Request
from workload_simulator.schema.scenario import ScenarioConfig
from workload_simulator.schema.schema import Schema
@dataclass
class BaseModeEncoder(ABC):

    scenario: ScenarioConfig

    @abstractmethod
    def assign_mode(self, requests: List[Request], user_updates: List[UserUpdateInfo]) -> Tuple[List[Block], List[Request]]:
        pass

    def config(self):
        return {
            "name": self.__class__.__name__,
        }

    @abstractmethod
    def short(self) -> str:
        pass

    def n_rounds_active(self):
        # for how many rounds is a block active
        return self.scenario.active_time_window // self.scenario.allocation_interval


def n_active_blocks(req_created: int, blocks: List[Block]):
    return sum(req_created >= b.created and req_created < b.retired for b in blocks)

class ActivePoissonSubsamplingModeEncoder(BaseModeEncoder):

    def assign_mode(self, requests: List[Request], user_updates: List[UserUpdateInfo]):

        # convert user updates into blocks
        blocks = []
        for user_update in user_updates:
            blk = Block(
                id=user_update.round_id,
                unlocked_budget=self.scenario.schema.empty_cost(),
                n_users=user_update.n_new_users,
                created=user_update.round_id,
                retired=user_update.round_id + self.n_rounds_active(),
            )
            blocks.append(blk)


        # update privacy + data demands of requests
        n_active_cache = {}

        for req in requests:
            if req.created not in n_active_cache:
                n_active_cache[req.created] = n_active_blocks(req.created, blocks)

            req.set_n_users(n_active_cache[req.created])
            req.set_request_cost(req.request_info.cost_poisson_amplified.rdp)

        return blocks, requests

    def short(self):
        return "poisson"

@dataclass
class ActiveBlockSubsetModeEncoder(BaseModeEncoder):

    user_group_size: int


    def assign_mode(self, requests: List[Request], user_updates: List[UserUpdateInfo]):

        # convert user updates into blocks
        blocks = []
        block_id_ctr = 0
        spillover = 0
        for update in user_updates:
            n_users = spillover + update.n_new_users
            n_groups = n_users // self.user_group_size
            spillover = n_users % self.user_group_size

            for _ in range(n_groups):
                blk = Block(
                    id=block_id_ctr,
                    unlocked_budget=self.scenario.schema.empty_cost(),
                    n_users=self.user_group_size,
                    created=update.round_id,
                    retired=update.round_id + self.n_rounds_active(),
                )
                blocks.append(blk)
                block_id_ctr += 1


        # update privacy + data demands of requests
        n_active_cache = {}

        for req in requests:
            if req.created not in n_active_cache:
                n_active_cache[req.created] = n_active_blocks(req.created, blocks)

            # the number of users requested corresponds to the fraction of selected active blocks at request time
            n_users =  int(n_active_cache[req.created] * req.request_info.sampling_info.prob)
            req.set_n_users(n_users)

            # we use non-amplified cost
            req.set_request_cost(req.request_info.cost_original.rdp)

        return blocks, requests

    def short(self):
        return "upc" # user parallel composition