import numpy as np
import copy

from Core.LocalEnvironmentCalculator import NeighborCountingEnvironmentCalculator
from MCMC.RandomExchangeOperator import RandomExchangeOperator


def setup_monte_carlo(start_particle, energy_calculator, local_feature_classifier):
    symbols = start_particle.get_all_symbols()
    energy_key = energy_calculator.get_energy_key()

    local_env_calculator = NeighborCountingEnvironmentCalculator(symbols)
    local_env_calculator.compute_local_environments(start_particle)

    local_feature_classifier.compute_feature_vector(start_particle)
    energy_calculator.compute_energy(start_particle)

    exchange_operator = RandomExchangeOperator(0.5)
    exchange_operator.bind_particle(start_particle)

    return energy_key, local_env_calculator, exchange_operator


def update_atomic_features(exchanges, local_env_calculator, local_feature_classifier, particle):
    neighborhood = set()
    for exchange in exchanges:
        index1, index2 = exchange
        neighborhood.add(index1)
        neighborhood.add(index2)

        neighborhood = neighborhood.union(particle.neighbor_list[index1])
        neighborhood = neighborhood.union(particle.neighbor_list[index2])

    for index in neighborhood:
        local_env_calculator.compute_local_environment(particle, index)
        local_feature_classifier.compute_atom_feature(particle, index)

    local_feature_classifier.compute_feature_vector(particle, recompute_atom_features=False)
    return particle, neighborhood


def run_monte_carlo(beta, max_steps, start_particle, energy_calculator, local_feature_classifier):
    energy_key, local_env_calculator, exchange_operator = setup_monte_carlo(start_particle, energy_calculator,
                                                                            local_feature_classifier)

    start_energy = start_particle.get_energy(energy_key)
    lowest_energy = start_energy
    accepted_energies = [(lowest_energy, 0)]

    found_new_solution = False
    fields = ['energies', 'symbols']
    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))

    total_steps = 0
    no_improvement = 0
    accepted_steps = []
    while no_improvement < max_steps:
        total_steps += 1
        if total_steps % 2000 == 0:
            print("Step: {}".format(total_steps))
            print("Lowest energy: {}".format(lowest_energy))

        exchanges = exchange_operator.random_exchange(start_particle)

        start_particle, neighborhood = update_atomic_features(exchanges, local_env_calculator, local_feature_classifier,
                                                              start_particle)

        accepted_particle = copy.deepcopy(start_particle)
        energy_calculator.compute_energy(start_particle)
        new_energy = start_particle.get_energy(energy_key)

        delta_e = new_energy - start_energy

        acceptance_rate = min(1, np.exp(-beta * delta_e))
        if np.random.random() < acceptance_rate:
            if found_new_solution:
                if new_energy > start_energy:
                    start_particle.swap_symbols(exchanges)
                    best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                    best_particle['energies'][energy_key] = start_energy
                    start_particle.swap_symbols(exchanges)

            start_energy = new_energy
            accepted_energies.append((new_energy, total_steps, accepted_particle))

            if new_energy < lowest_energy:
                no_improvement = 0
                lowest_energy = new_energy
                found_new_solution = True
            else:
                no_improvement += 1
                found_new_solution = False

        else:
            no_improvement += 1

            # roll back exchanges and make sure features and environments are up-to-date
            start_particle.swap_symbols(exchanges)
            start_particle.set_energy(energy_key, start_energy)
            for index in neighborhood:
                local_env_calculator.compute_local_environment(start_particle, index)
                local_feature_classifier.compute_atom_feature(start_particle, index)

            if found_new_solution:
                best_particle = copy.deepcopy(start_particle.get_as_dictionary(fields))
                best_particle['energies'][energy_key] = copy.deepcopy(start_energy)
                found_new_solution = False

    accepted_energies.append((accepted_energies[-1][0], total_steps))

    return [best_particle, accepted_energies], accepted_steps
