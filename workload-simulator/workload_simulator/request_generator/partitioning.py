from abc import ABC, abstractmethod
from dataclasses import dataclass

import random

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import beta

from workload_simulator.schema.schema import Schema

class Defaults:

    @staticmethod
    def new_ultra_highly_partitioned(schema: Schema):
        """
        In almost all cases, the requests select only a single slot.
        """
        return IntervalPopulationSelection(schema, beta_a=1, beta_b=100)


    @staticmethod
    def new_highly_partitioned(schema: Schema):
        """
        """
        return IntervalPopulationSelection(schema, beta_a=1, beta_b=10)


    @staticmethod
    def new_ultra_low_partitioned(schema: Schema):
        """
        In almost all cases, the requests select all slots.
        """
        return IntervalPopulationSelection(schema, beta_a=2**20, beta_b=2)

    @staticmethod
    def new_low_partitioned(schema: Schema):
        """
        In expectation, requests select 0.66 of the slots.
        (function looks like an exponetial)

        P(X<0.25)=0.13
        P(X<0.5)=0.29
        P(X<0.75)=0.5
        """
        return IntervalPopulationSelection(schema, beta_a=1, beta_b=0.5)

    @staticmethod
    def new_quadratic_partitioned(schema: Schema):
        """
        In expectation, requests select 0.5 of the slots.
        (function looks like a quadratic function and thus most of the time selects either a very high or a very low percentage)
        """
        return IntervalPopulationSelection(schema, beta_a=0.5, beta_b=0.5)


    @staticmethod
    def new_normal_partitioned(schema: Schema):
        """
        In expectation, requests select 0.5 of the slots.
        (function looks like a normal distribution and thus most of the time selects a percentage around 0.5)
        """
        return IntervalPopulationSelection(schema, beta_a=2, beta_b=2)



@dataclass
class BasePopulationSelection(ABC):

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def info(self):
        pass

@dataclass
class IntervalPopulationSelection(BasePopulationSelection):

    """
    This class implements the population selection mechanism for interval-based selections.
    Each request receives a center uniformly at random from the domain of the attribute.
    In addition, we sample a percentage of the partitioning attributes that this request covers from a Beta distribution.
    This allows us to control the selectivity of the request and thus the overlap between requests.

    Args:
        schema (dict): The schema of the population with a single attribute.
        beta_a (float): The alpha parameter for the Beta distribution used to determine the selection percentage.
        beta_b (float): The beta parameter for the Beta distribution used to determine the selection percentage.

    Attributes:
        attribute_name (str): The name of the attribute in the schema.
        domain (dict): The value domain of the attribute, containing "min" and "max" values.
        n_slots (int): The number of slots in the attribute's domain.
        beta_a (float): The alpha parameter for the Beta distribution.
        beta_b (float): The beta parameter for the Beta distribution.

    Methods:
        show_selection_percentage_pdf(): Plot and save the probability density function (PDF) of the selection percentage.
        generate(): Generate a single disjunctive normal form (DNF) selection for a request.
    """


    schema: Schema
    beta_a: float
    beta_b: float

    def __post_init__(self):

        assert len(self.schema.attributes) == 1, f"at the moment, only schemas with one attribute are supported (has {len(self.schema.attributes)} attributes)"

        self.attribute_name = self.schema.attributes[0].name
        self.domain = self.schema.attributes[0].value_domain
        self.n_slots = self.domain.max - self.domain.min + 1





    def show_selection_percentage_pdf(self):
        x = np.linspace(0.0, 1.0, 100000)

        plt.plot(x, beta.pdf(x, self.beta_a, self.beta_b), label=f"beta(a={self.beta_a}, b={self.beta_b})")
        plt.legend()
        plt.xlabel("selection percentage")
        plt.savefig(f"selection_percentage_a{self.beta_a}_b{self.beta_b}.png")


    def generate(self):
        # generate a single dnf selection for a request

        selection_percentage = beta.rvs(self.beta_a, self.beta_b, size=1)[0]

        # since diff is +/- from the random center, we need to divide by 2
        diff = round(selection_percentage * self.n_slots / 2)


        slot = random.randint(0, self.n_slots-1) # select a random slot

        selection, selection_size = self._selection(slot, diff)
        dnf = self._intervals2dnf(selection)

        return dnf, selection_size

    def info(self):
        return {
            "name": __class__.__name__,
            "beta_a": self.beta_a,
            "beta_b": self.beta_b,
            "n_slots": self.n_slots,
        }



    def compute_stats(self, selections):

        # materialize every possible slot and go over the batch of selections
        # and assign them to the respective slots

        slots_per_request_distribution = {}

        assignments = {slot: set() for slot in range(self.n_slots)}
        for i, dnf in enumerate(selections):
            intervals = self._dnf2intervals(dnf)
            slot_count = 0
            for interval in intervals:
                for slot in range(interval[0], interval[1] + 1):
                    assignments[slot].add(i)
                    slot_count += 1

            if slot_count not in slots_per_request_distribution:
                slots_per_request_distribution[slot_count] = 0
            slots_per_request_distribution[slot_count] += 1

        # go over the assignment and form groups (i.e., segments) for slots with the same set of requests
        segments = {}
        for slot, request_id_set in assignments.items():

            x = frozenset(request_id_set)

            if x not in segments:
                segments[x] = []
            segments[x].append(slot)

        # build a distribution of segment sizes (how many requests per segment)
        segment_size_distribution = {}

        for seg, slots in segments.items():
            size = len(seg)
            if size not in segment_size_distribution:
                segment_size_distribution[size] = 0
            segment_size_distribution[size] += 1



        n_segments = len(segments)
        max_segment_size = max(segment_size_distribution.keys())

        segment_size_distribution = dict(sorted(segment_size_distribution.items()))

        return {"n_segments": n_segments, "max_segment_size": max_segment_size, "segment_size_distribution": segment_size_distribution, "slots_per_request_distribution": slots_per_request_distribution}


    def _selection(self, slot, diff):

        if 2 * diff + 1 >= self.n_slots:
            selection = [(0, self.n_slots - 1)]

        else:

            if slot + diff < self.n_slots and slot - diff >= 0:
                # no wrapping occurs
                selection = [(slot - diff, slot + diff)]

            elif slot + diff >= self.n_slots:
                # wrapping on the right side occurs
                assert slot - diff >= 0, "slot - diff must be >= 0"
                p1 = (slot - diff, self.n_slots - 1)
                p2 = (0, slot + diff - self.n_slots)
                selection = [p2, p1]
            elif slot - diff < 0:
                # wrapping on the left side occurs

                p1 = (0, slot + diff)
                p2 = (self.n_slots + slot - diff, self.n_slots - 1)
                selection = [p1, p2]

        def selection_size(interval):
            return interval[1] - interval[0] + 1

        selection_size = sum(selection_size(x) for x in selection)

        return selection, selection_size

    def _intervals2dnf(self, intervals):

        dnf = {
            "conjunctions": []
        }

        for interval in intervals:

            d = {
                "predicates":{
                    self.attribute_name: {
                        "Between":{
                            "min": self.domain.min + interval[0],
                            "max": self.domain.min + interval[1]
                        }
                    }
                }
            }

            dnf["conjunctions"].append(d)

        return dnf

    def _dnf2intervals(self, dnf):
        intervals = []
        for x in dnf["conjunctions"]:
            between = x["predicates"][self.attribute_name]["Between"]
            interval = (between["min"] - self.domain["min"], between["max"] - self.domain["min"])
            intervals.append(interval)
        return intervals




#if __name__ == "__main__":
#
#    schema = {"attributes": [{"name": "a", "value_domain": {"min": 0, "max": 9999}}]}
#    schema_size = schema["attributes"][0]["value_domain"]["max"] - schema["attributes"][0]["value_domain"]["min"] + 1
#    n_requests = 1000
#
#    # This is a good set of candidate values for the beta parameters (they have extreme values where basically eaxch request only selects a single slot and where each request selects all slots)
#    # The case a=b=2 centers the distribution around 0.5.
#    vary_b = [(2, 2**x) for x in [1, 2, 4, 20]]
#    vary_a = [(x[1], x[0]) for x in vary_b]
#
#
#    for beta_a, beta_b in sorted(set(vary_a + vary_b)):
#
#        selection = IntervalPopulationSelection(schema=schema, beta_a=beta_a, beta_b=beta_b)
#
#        if beta_a < 2**10 and beta_b < 2**10:
#            selection.show_selection_percentage_pdf()
#
#        dnfs = []
#        for request_id in range(n_requests):
#            dnf, selection_size = selection.generate()
#            dnfs.append(dnf)
#
#        stats = selection.compute_stats(dnfs)
#        print(f"beta_a={beta_a}, beta_b={beta_b}, n_requests: {n_requests}, schema_size: {schema_size}")
#        print(f"      n_segments: {stats['n_segments']}")
#        print(f"      max_segment_size: {stats['max_segment_size']}")