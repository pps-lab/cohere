from abc import ABC, abstractmethod
import random
import numpy as np

from workload_simulator.request_generator.sampling import BaseSampling, SamplingInfo
from workload_simulator.request_generator.calibration import BaseCostCalibration

from autodp import mechanism_zoo, calibrator_zoo, transformer_zoo
from autodp.autodp_core import Mechanism as AutoDPMechanism

from dataclasses import dataclass


cache = {} # {str_mechanism: {mech: budget}}



@dataclass
class MechanismCost:
    name: str
    adp: dict = None
    rdp: dict = None

    def set_cost_adp(self, epsilon, delta):
        self.adp = {"EpsDeltaDp": {"eps": epsilon, "delta": delta}}

    def set_cost_rdp(self, mechanism, alphas):
        costs = [mechanism.get_RDP(alpha) for alpha in alphas]
        self.rdp = {"Rdp": {"eps_values": costs}}


@dataclass
class MechanismInfo:
    mechanism: dict

    cost_poisson_amplified: MechanismCost # amplified cost
    cost_original: MechanismCost # not amplified cost

    sampling_info: SamplingInfo

    selection: dict = None
    utility_info: dict = None




class MechanismWrapper(ABC):

    improved_bound_flag = False


    def info(self):
        return {
            "name": self.__class__.__name__,
            "cost_calibration": self.cost_calibration.info(),
            "sampling": self.sampling.info(),
        }


    def __init__(self, cost_calibration: BaseCostCalibration = None, sampling: BaseSampling = None):

            assert cost_calibration is None or isinstance(cost_calibration, BaseCostCalibration)
            assert sampling is None or isinstance(sampling, BaseSampling)

            self.cost_calibration = cost_calibration
            self.sampling = sampling


    def generate(self, alphas) -> MechanismInfo:

        epsilon, delta, cost_name = self.cost_calibration.sample()

        mechanism, mech_info, cost_original = self.cache_calibrate(epsilon, delta, cost_name, alphas)

        sampling_info = self.sampling.sample()
        mechanism_amplified, cost_amplified = self.poisson_amplify(mechanism, amplify_poisson_lambda=sampling_info.prob, cost_name=cost_name, epsilon=epsilon, delta=delta, alphas=alphas)

        return MechanismInfo(mechanism=mech_info,
                      cost_poisson_amplified=cost_amplified,
                      cost_original=cost_original,
                      sampling_info=sampling_info)


    def cache_calibrate(self, epsilon, delta, cost_name, alphas):
        """
        caches the calibration of the mechanism
        """

        key = self.__class__.__name__

        if key not in cache:
            cache[key] = {}

        if (epsilon, delta, 1.0) in cache[key]:
            mechanism, info, mcost = cache[key][(epsilon, delta, 1.0)]
        else:
            # print(f"Cache Miss Original: {key} eps={epsilon} delta={delta}")
            mechanism, info = self.calibrate(epsilon, delta)

            mcost = MechanismCost(name=cost_name)
            mcost.set_cost_adp(epsilon, delta)
            mcost.set_cost_rdp(mechanism, alphas)

            cache[key][(epsilon, delta, 1.0)] = (mechanism, info, mcost)

        return mechanism, info, mcost

    @abstractmethod
    def calibrate(self, epsilon:float, delta: float):
        pass

    def get_poisson_amplified_mechanism(self, mechanism: AutoDPMechanism, amplify_poisson_lambda: float):
        if amplify_poisson_lambda == 1.0:
            return mechanism
        subsampling = transformer_zoo.AmplificationBySampling(PoissonSampling=True)
        mechanism_amplified = subsampling(mechanism, amplify_poisson_lambda, improved_bound_flag=self.improved_bound_flag)
        return mechanism_amplified


    def poisson_amplify(self, mechanism: AutoDPMechanism, amplify_poisson_lambda: float, cost_name, alphas, epsilon=None, delta=None) -> AutoDPMechanism:

        key = self.__class__.__name__

        skip_cache = epsilon is None or delta is None

        if (not skip_cache) and (epsilon, delta, amplify_poisson_lambda) in cache[key]:
            mechanism_amplified, _info, mcost =  cache[key][(epsilon, delta, amplify_poisson_lambda)]
        else:
            # print(f"Cache Miss Amplify: {key} eps={epsilon} delta={delta} lambda={amplify_poisson_lambda}")
            mechanism_amplified = self.get_poisson_amplified_mechanism(mechanism, amplify_poisson_lambda)

            mcost = MechanismCost(name=cost_name)
            mcost.set_cost_rdp(mechanism_amplified, alphas)

            if not skip_cache:
                cache[key][(epsilon, delta, amplify_poisson_lambda)] = (mechanism_amplified, None, mcost)


        return mechanism_amplified, mcost



class LaplaceMechanism(MechanismWrapper):

    improved_bound_flag = True

    def create(self, noise_param):
        l1_sensitivity = 1
        mechanism = mechanism_zoo.LaplaceMechanism(b=noise_param)
        info = {"mechanism": {"name": __class__.__name__, "b": mechanism.params["b"], "l1_sensitivity": l1_sensitivity}}
        return mechanism, info


    def calibrate(self, epsilon, delta):

        l1_sensitivity = 1
        # b = float(l1_sensitivity) / float(epsilon)

        cal = calibrator_zoo.generalized_eps_delta_calibrator()

        mechanism = cal(mechanism_zoo.LaplaceMechanism, epsilon, delta, [0, 1000], name=__class__.__name__)

        info = {"mechanism": {"name": __class__.__name__, "b": mechanism.params["b"], "l1_sensitivity": l1_sensitivity}}
        # "cost_name": cost_name, "adp_cost": request_cost_adp(epsilon, delta), "rdp_cost": request_cost_rdp(rdp_cost)
        return mechanism, info


class GaussianMechanism(MechanismWrapper):

    improved_bound_flag = True

    def create(self, sigma):
        l2_sensitivity = 1
        mechanism = mechanism_zoo.ExactGaussianMechanism(sigma=sigma)
        info = {"mechanism": {"name": __class__.__name__, "sigma": mechanism.params["sigma"], "l2_sensitivity": l2_sensitivity}}
        return mechanism, info


    def calibrate(self, epsilon, delta):

        l2_sensitivity = 1
        # sigma = np.sqrt(2 * np.log(1.25 / delta)) * float(l2_sensitivity) / float(epsilon)

        cal = calibrator_zoo.generalized_eps_delta_calibrator()
        mechanism = cal(mechanism_zoo.ExactGaussianMechanism, epsilon, delta, [0, 1000], name=__class__.__name__)


        info = {"mechanism": {"name": __class__.__name__, "sigma": mechanism.params["sigma"], "l2_sensitivity": l2_sensitivity}}
        #        "cost_name": cost_name, "adp_cost": request_cost_adp(epsilon, delta), "rdp_cost": request_cost_rdp(rdp_cost)}
        return mechanism, info

class DiscreteGaussianMechanism(MechanismWrapper):

    improved_bound_flag = True

    def create(self, sigma):
        l2_sensitivity = 1
        mechanism = mechanism_zoo.DiscreteGaussianMechanism(sigma=sigma)
        info = {"mechanism": {"name": __class__.__name__, "sigma": sigma, "rho": mechanism.params["rho"], "l2_sensitivity": l2_sensitivity}}
        return mechanism, info


    def calibrate(self, epsilon, delta):
        l2_sensitivity = 1
        cal = calibrator_zoo.generalized_eps_delta_calibrator()
        mechanism = cal(mechanism_zoo.DiscreteGaussianMechanism, epsilon, delta, [0, 1000], name=__class__.__name__)

        info = {"mechanism": {"name": __class__.__name__, "sigma": mechanism.params["sigma"], "l2_sensitivity": l2_sensitivity}}
        return mechanism, info


class RandResponseMechanism(MechanismWrapper):

    improved_bound_flag = False

    def calibrate(self, epsilon, delta):

        cal = calibrator_zoo.generalized_eps_delta_calibrator()
        mechanism = cal(mechanism_zoo.RandresponseMechanism, epsilon, delta, [0.0, 1.0], name=__class__.__name__)

        info = {"mechanism": {"name": __class__.__name__, "bernoulli_p": mechanism.params["p"]}}
        #        "cost_name": cost_name, "adp_cost": request_cost_adp(epsilon, delta), "rdp_cost": request_cost_rdp(rdp_cost)}
        return mechanism, info


from autodp.mechanism_zoo import zCDP_Mechanism

class ExponentialMechanism(MechanismWrapper):

    improved_bound_flag = False

    class MyExponentialMechanism(zCDP_Mechanism):
        # NOTE: we use a custom Exponential mechanism because the the one from autodp may suffer from numerical problems for large alphas
        def __init__(self, eps, name='ExpMech'):
            zCDP_Mechanism.__init__(self, eps**2/8, name=name)
            # the zCDP bound is from here: https://arxiv.org/pdf/2004.07223.pdf
            self.params['eps'] = eps


    def create(self, epsilon):
        mechanism = ExponentialMechanism.MyExponentialMechanism(eps=epsilon) #, RDP_off=False
        info = {"mechanism": {"name": __class__.__name__, "epsilon": mechanism.params["eps"], "all": mechanism.params}}
        return mechanism, info


    def calibrate(self, epsilon, delta):

        mechanism = ExponentialMechanism.MyExponentialMechanism(eps=epsilon) # , RDP_off=False

        info = {"mechanism": {"name": __class__.__name__, "epsilon": mechanism.params["eps"]}}
        #        "cost_name": cost_name, "adp_cost": request_cost_adp(epsilon, delta), "rdp_cost": request_cost_rdp(rdp_cost)}
        return mechanism, info



class SVTLaplaceMechanism(MechanismWrapper):

    improved_bound_flag = False

    def create(self, noise_param):
        raise ValueError("not implemented yet")


    def calibrate(self, epsilon, delta):

        cutoff = 5
        k = 100
        l1_sensitivity = 1

        calibrate = calibrator_zoo.generalized_eps_delta_calibrator()
        mechanism = calibrate(mechanism_zoo.LaplaceSVT_Mechanism, epsilon, delta, [0, 2000], params={"k": k, "c": cutoff}, para_name="b", name=__class__.__name__)


        info = {"mechanism": {"name": __class__.__name__, "b": mechanism.params["b"], "cutoff": cutoff, "k": k, "l1_sensitivity": l1_sensitivity}}
        #        "cost_name": cost_name, "adp_cost": request_cost_adp(epsilon, delta), "rdp_cost": request_cost_rdp(rdp_cost)}
        return mechanism, info


class SVTGaussianMechanism(MechanismWrapper):

    improved_bound_flag = False

    def create(self, noise_param):
        raise ValueError("not implemented yet")


    def calibrate(self, epsilon, delta):

        cutoff = random.randint(1, 10)
        k = random.randint(cutoff, 200)
        l2_sensitivity = 1

        calibrate = calibrator_zoo.generalized_eps_delta_calibrator()
        margin = 1
        if cutoff == 1:
            mechanism = calibrate(mechanism_zoo.GaussianSVT_Mechanism, epsilon, delta, [0, 20000], params={"k": k, "margin": margin}, para_name="sigma", name=__class__.__name__)
        elif cutoff > 1:
            class Wrapper(mechanism_zoo.GaussianSVT_Mechanism):
                def __init__(self, params,name='GaussianSVT'):
                    super().__init__(params, name, rdp_c_1=False)
            mechanism = calibrate(Wrapper, epsilon, delta, [0, 10000], params={"k": k, "c": cutoff}, para_name="sigma", name=__class__.__name__)
        else:
            raise ValueError("fail")

        info = {"mechanism": {"name": __class__.__name__, "sigma": mechanism.params["sigma"], "cutoff": cutoff, "k": k, "l2_sensitivity": l2_sensitivity}}
        #        "cost_name": cost_name, "adp_cost": request_cost_adp(epsilon, delta), "rdp_cost": request_cost_rdp(rdp_cost)}
        return mechanism, info



class MLNoisySGDMechanism(MechanismWrapper):

    def create(self, noise_param):
        raise ValueError("not implemented yet")

    class NoisySGD_Mechanism(AutoDPMechanism):
        def __init__(self, params, name='NoisySGD'):
            AutoDPMechanism.__init__(self)
            self.name = name
            self.params = params

            # create such a mechanism as in previously
            subsample = transformer_zoo.AmplificationBySampling(PoissonSampling=True)
            # by default this is using poisson sampling

            mech = mechanism_zoo.ExactGaussianMechanism(sigma=params["sigma"])

            # Create subsampled Gaussian mechanism
            SubsampledGaussian_mech = subsample(mech, params["prob"], improved_bound_flag=True)
            # for Gaussian mechanism the improved bound always applies

            # Now run this for niter iterations
            compose = transformer_zoo.Composition()
            mech = compose([SubsampledGaussian_mech], [params["niter"]])

            # Now we get it and let's extract the RDP function and assign it to the current mech being constructed
            rdp_total = mech.RenyiDP
            self.propagate_updates(rdp_total, type_of_update='RDP')


    def calibrate(self, epsilon, delta):

        params = {
            "prob": 0.005, #random.uniform(0.005, 0.01),
            "niter": 1000, #random.randint(100, 1000),
            "sigma": None
        }

        calibrate = calibrator_zoo.generalized_eps_delta_calibrator()
        mechanism = calibrate(MLNoisySGDMechanism.NoisySGD_Mechanism, epsilon, delta, [0.1, 500], params=params, para_name="sigma", name=__class__.__name__)


        info = {"mechanism": {"name": __class__.__name__, "prob": mechanism.params["prob"], "niter": mechanism.params["niter"], "sigma": mechanism.params["sigma"]}}
        #                "cost_name": cost_name, "adp_cost": request_cost_adp(eps, delta), "rdp_cost": request_cost_rdp(rdp_cost), "adp_cost_target": request_cost_adp(epsilon, delta)}
        return mechanism, info




class MLPateGaussianMechanism(MechanismWrapper):

    improved_bound_flag = True

    def create(self, noise_param):
        raise ValueError("not implemented yet")


    def calibrate(self, epsilon, delta):

        class PATE(mechanism_zoo.ExactGaussianMechanism):
            def __init__(self, params, name='PATE'):
                # sigma is the std of the Gaussian noise added to the voting scores
                if params["binary"]:
                    # This is a binary classification task
                    sensitivity = 1
                else: # for the multiclass case, the L2 sensitivity is sqrt(2)
                    sensitivity = np.sqrt(2)

                self.params = params
                super().__init__(sigma=params["sigma"]/sensitivity/np.sqrt(params["m"]),name=name)


        calibrate = calibrator_zoo.generalized_eps_delta_calibrator()
        dataset_size = 5000
        binary = True
        mechanism = calibrate(PATE, epsilon, delta, [0.1, 50000], params={"m": dataset_size, "binary": binary}, para_name="sigma", name=__class__.__name__)

        info = {"mechanism": {"name": __class__.__name__, "dataset_size": dataset_size, "binary_pate": binary, "sigma": mechanism.params["sigma"]}}

        return mechanism, info