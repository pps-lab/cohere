from dataclasses import dataclass, field


@dataclass
class UserUpdateInfo:
    round_id: int
    n_new_users: int


@dataclass
class Block:
    id: int
    unlocked_budget: list
    n_users: int
    created: int
    retired: int
    request_ids: list = field(default_factory=list)