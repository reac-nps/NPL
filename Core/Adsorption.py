import Core.MathModules as math

from ase import Atoms

import copy
import itertools
import numpy as np


class Adsorption():

    def __init__(self):
        pass
        
class FindAdsorptionSites():
    """ Class that identify and place add atoms based on the Generalized coordination Numbers of the nanoparticles"""
    def __init__(self):
        self.ontop = []
        self.bridge_positions = []
        self.hollow_positions = []
    
    def get_ontop_sites(self, particle):
        self.ontop = particle.get_atom_indices_from_coordination_number(range(10))
    
    def get_bridge_sites(self, particle):
        shell_atoms = set(particle.get_atom_indices_from_coordination_number(range(10)))
        for central_atom_index in shell_atoms:
            central_atom_nearest_neighbors = set(particle.get_coordination_atoms(central_atom_index))
            for nearest_neighbor in shell_atoms.intersection(central_atom_nearest_neighbors):
                pair = sorted([central_atom_index, nearest_neighbor])
                if pair not in self.bridge_positions:
                    self.bridge_positions.append(pair)
                    
    def get_hollow_sites(self, particle):      
        for pair in self.bridge_positions:
            for third_atom in self.find_plane_for_bridge_atoms(particle, pair):
                triplet = copy.copy(pair)
                triplet.append(third_atom)
                triplet = sorted(triplet)
                if triplet not in self.hollow_positions:
                    self.hollow_positions.append(triplet)
    
    def find_plane_for_bridge_atoms(self, particle, indices):
        uncoordinated_atoms = particle.get_atom_indices_from_coordination_number(range(10))
        uncoordinated_atoms = set(uncoordinated_atoms)
        shell_1 = set(particle.get_coordination_atoms(indices[0]))
        shell_2 = set(particle.get_coordination_atoms(indices[1]))
        shared_atoms = uncoordinated_atoms.intersection(shell_1.intersection(shell_2))
        return list(shared_atoms)
    
class PlaceAddAtoms():
    """Class that plance add atoms on positions identified by FindAdsorptionSites"""
    def __init__(self, symbols):
        self.adsorption_sites = FindAdsorptionSites()
        self.symbols = sorted(symbols)
        self.sites_list = []
        self.ontop_positions = {site[0] : [] for site in itertools.combinations_with_replacement(self.symbols,1)}
        self.bridge_positions = {''.join(list(site)) : [] for site in itertools.combinations_with_replacement(self.symbols,2)}
        self.hollow_positions = {''.join(list(site)) : [] for site in itertools.combinations_with_replacement(self.symbols,3)}

    def bind_particle(self, particle):
        particle.atoms.atoms.translate(-particle.atoms.atoms.get_center_of_mass())
        self.adsorption_sites.get_ontop_sites(particle)
        self.adsorption_sites.get_bridge_sites(particle)
        self.adsorption_sites.get_hollow_sites(particle)  
        self.get_ontop_sites(particle)
        self.get_bridge_sties(particle)
        self.get_hollow_sties(particle)
    
    def get_ontop_sites(self, particle):
        for atom in self.adsorption_sites.ontop:
            self.ontop_positions[particle.get_symbol(atom)].append(atom)

    def get_bridge_sties(self, particle):
        for pairs in self.adsorption_sites.bridge_positions:
            self.bridge_positions[''.join(sorted(particle.get_symbols(pairs)))].append(pairs)
 
    def get_hollow_sties(self, particle):
        for triplet in self.adsorption_sites.hollow_positions:
            self.hollow_positions[''.join(sorted(particle.get_symbols(triplet)))].append(triplet)
            
    def get_total_adsorption_sites_xyz(self, particle):
        self.sites_list += self.adsorption_sites.ontop_positions
        self.sites_list += self.adsorption_sites.bridge_positions
        self.sites_list += self.adsorption_sites.hollow_positions
        
    def get_xyz_site_from_atom_indices(self, particle, site):
        
        if isinstance(site, (np.ndarray, np.generic)) or isinstance(site, int):
            unit_vector1, length1 = math.get_unit_vector(particle.get_position(site))
            xyz_site = (unit_vector1*(length1+2))
        else:
            xyz_atoms = [particle.get_position(atom_index) for atom_index in site]
            xyz_site_plane = math.find_middle_point(xyz_atoms)
            unit_vector1, length1 = math.get_unit_vector(xyz_site_plane)
            if len(site) == 3:
                normal_vector = math.get_normal_vector(xyz_atoms) 
            if len(site) == 2:
                third_atom = list(self.adsorption_sites.find_plane_for_bridge_atoms(particle, site))[0]
                xyz_third_atom = particle.get_position(third_atom)
                xyz_atoms.append(xyz_third_atom)
                normal_vector = math.get_normal_vector(xyz_atoms)
                
            unit_vector2, length2 = math.get_unit_vector(normal_vector)
            dot_prod = np.dot(unit_vector1, unit_vector2)
            direction = dot_prod/abs(dot_prod)
            xyz_site = (unit_vector1*(length1))+(direction*unit_vector2*(1.4))
   
        return xyz_site

    def place_add_atom(self, particle, add_atom_symbol, sites):
        add_atom_list = []
        for site in sites:
            xyz_site = self.get_xyz_site_from_atom_indices(particle, site)
            add_atom = Atoms(add_atom_symbol)   
            add_atom.translate(xyz_site)
            add_atom_list.append(add_atom)
            
        for add_atom in add_atom_list:
            particle.add_atoms(add_atom)
            
        return particle    





