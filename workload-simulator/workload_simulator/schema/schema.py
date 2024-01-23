

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class RdpAccountingType:

    eps_values: List[float] = None

    def __post_init__(self):
        self.eps_values = len(self.alphas()) * [0.0]


    def alphas(self):
        return [1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 16, 32, 64, 1e6, 1e10] #, 1e16 1e16  TODO [nku] maybe bring back?

    def empty_cost(self):
        return {"Rdp": {"eps_values": [0.0] * len(self.eps_values)}}


    def cost_config(self):
        return {
            "alphas": self.alphas()
        }


@dataclass
class AccountingType:

    Rdp: RdpAccountingType = RdpAccountingType()

    def empty_cost(self):
        return self.Rdp.empty_cost()

    def cost_config(self):
        return self.Rdp.cost_config()



@dataclass
class DomainRange:
    min: int
    max: int

@dataclass
class Attribute:
    name: str
    value_domain: DomainRange

@dataclass
class Schema:
    attributes: List[Attribute]
    accounting_type: AccountingType

    def empty_cost(self):
        return self.accounting_type.empty_cost()

    def cost_config(self):
        return self.accounting_type.cost_config()


def create_single_attribute_schema(domain_size, name="attr0"):
    return Schema(attributes=[Attribute(name=name, value_domain=DomainRange(min=0, max=domain_size-1))], accounting_type=AccountingType())