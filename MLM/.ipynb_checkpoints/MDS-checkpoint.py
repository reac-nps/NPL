import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy

import Core.Nanoparticle as NP
from ase.visualize import view
from Core.GlobalFeatureClassifier import SimpleFeatureClassifier


class CustomFeatureClassifier(SimpleFeatureClassifier):
    def __init__(self, symbols):
        SimpleFeatureClassifier.__init__(self, symbols)
        self.feature_key = 'TFC'
        self.bond_scaling_factor = 0.1

    def compute_feature_vector(self, particle):
        n_atoms = particle.get_n_atoms()
        n_aa_bonds, n_bb_bonds, n_ab_bonds = self.compute_respective_bond_counts(particle)

        coordinated_atoms_a = [len(particle.get_atom_indices_from_coordination_number([cn], symbol=self.symbol_a)) for cn in range(13)]
        coordinated_atoms_b = [len(particle.get_atom_indices_from_coordination_number([cn], symbol=self.symbol_b)) for cn in range(13)]

        feature_vector = np.array([n_aa_bonds*self.bond_scaling_factor, n_bb_bonds*self.bond_scaling_factor,
                                               n_ab_bonds*self.bond_scaling_factor] + coordinated_atoms_a + coordinated_atoms_b)

        particle.set_feature_vector(self.feature_key, feature_vector)

def compute_distance_matrix(training_set, key):
    
    import matplotlib.pyplot as plt
    print("Computing distance_matrix...")
    N = len(training_set)
    dist_matrix = np.empty((N, N))

    for i in range(N):
        #if i % 100 == 0:
         #   print(i)
        for j in range(i, N):
            distance_vector = training_set[i].data[key] - training_set[j].data[key]
            dist_matrix[i][j] = dist_matrix[j][i] = np.linalg.norm(distance_vector)
    plt.imshow(dist_matrix)
    plt.colorbar()
    plt.show()
    return dist_matrix

def maximum_distance_sampling(original_set, key, n_structures):
    dist_matrix = compute_distance_matrix(original_set, key)
    print("computed distance_matrix")
    print('Selecting indices...')
    seed = np.random.randint(len(original_set))
    selected_indices = [seed, np.argmax(dist_matrix[seed])]

    for i in range(n_structures - 2):
        samples_rem = np.delete(np.arange(len(original_set)), selected_indices)
                        
        dists = dist_matrix[selected_indices][:, samples_rem]
        min_dists = np.min(dists, axis=0)
        largest_min_distance = np.argmax(min_dists)
        selected_indices.append(samples_rem[largest_min_distance])

    print('Indices selected!')
    training_set = [original_set[i] for i in selected_indices]
    return training_set, selected_indices
