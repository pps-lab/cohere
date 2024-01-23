from workload_simulator.request_generator import mechanism

from autodp import transformer_zoo

def convert_to_autodp(optimizer, n_batches: int, epochs: int, population_prob: float=1.0):

    sampling_rate = 1.0 / n_batches
    n_steps = int(epochs / sampling_rate)

    sigma = optimizer.noise_multiplier

    wrapper = mechanism.GaussianMechanism()

    mech_raw, info = wrapper.create(sigma=sigma)

    subsampling = transformer_zoo.AmplificationBySampling(PoissonSampling=True)
    mech  = subsampling(mech_raw, sampling_rate, improved_bound_flag=True)

    mech_amplified  = subsampling(mech_raw, sampling_rate * population_prob, improved_bound_flag=True)

    compose = transformer_zoo.Composition()
    mech_composed = compose(n_steps * [mech], n_steps * [1])

    mech_amplified_composed = compose(n_steps * [mech_amplified], n_steps * [1])

    mechanism_info = {
        "n_steps": n_steps,
        "mech_per_step": info
    }
    return mech_composed, mech_amplified_composed, mechanism_info