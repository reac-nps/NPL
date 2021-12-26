import Core.MathModules as math

from ase import Atoms

import itertools
import numpy as np


class FindAdsorptionSites():
    """ Class that identify and place add atoms based on the Generalized coordination Numbers of the nanoparticles"""
    def __init__(self):
        self.ontop = []
        self.bridge_positions = []
        self.hollow_positions = []

    def get_bridge_sites(self, particle):
        shell_atoms = set(particle.get_atom_indices_from_coordination_number(range(10)))
        for central_atom_index in shell_atoms:
            central_atom_nearest_neighbors = set(particle.get_coordination_atoms(central_atom_index))
            for nearest_neighbor in shell_atoms.intersection(central_atom_nearest_neighbors):
                self.bridge_positions.append([central_atom_index, nearest_neighbor])

    def get_hollow_sites(self, particle):
        dict_bridge_positions = {index_1 : [] for index_1, index_2 in self.bridge_positions}
        for index_1, index_2 in self.bridge_positions:
            dict_bridge_positions[index_1].append(index_2)

        triplets = []
        for index_1, indices in dict_bridge_positions.items():
            for index_2 in indices:
                for index_3 in indices:
                    triplet = [index_1, index_2]
                    if index_3 in dict_bridge_positions[index_2]:
                        triplet.append(index_3)
                        triplet = sorted(triplet)
                        triplets.append(triplet)
        triplets.sort()
        self.hollow_positions = list(triplet for triplet, _ in itertools.groupby(triplets))

    def addatom_ontop(self, indices, distance):
        atoms_addatoms = self.get_ase_atoms()
        
        for index in indices:
            cn = self.get_coordination_number(index)
            position = self.get_position(index)
            unit, length = math.get_unit_vector(position)
            tilted_vector, _ = math.get_unit_vector(unit + math.get_perpendicular_vector(unit))
        
            if cn == 4 or cn == 6:
                C_distance = unit*(length + distance)
                O_distance = C_distance + (tilted_vector*1.15)

            if cn == 7:
                edge_perp_vec = math.get_perpendicular_edge_vector(position)
                perp_vector = math.get_perpendicular_vector(edge_perp_vec)
                tilted_vector,_  = math.get_unit_vector(edge_perp_vec + perp_vector)                                   
                C_distance = (unit * length) + (edge_perp_vec * distance)
                O_distance = C_distance + (tilted_vector*1.15)

            if cn == 9:
                around = self.get_coordination_atoms(index)
                plane = [self.get_position(x) for x in around if self.get_coordination_number(x) < 12]
                normal = math.get_normal_vector(plane)
                dot_prod = math.get_dot_product(unit, normal)
                direction = dot_prod/abs(dot_prod)
                normal = normal * direction
                perp_vector = math.get_perpendicular_vector(normal)
                tilted_vector, _ = math.get_unit_vector(normal + perp_vector)
                C_distance = (unit*length) + (normal*distance)
                O_distance = C_distance + (tilted_vector*1.15)

            add_atom1 = Atoms('O')
            #add_atom2 = Atoms('O')    
            add_atom1.translate(C_distance)
            #add_atom2.translate(O_distance)
            atoms_addatoms += add_atom1
            #atoms_addatoms += add_atom2
        
        return atoms_addatoms
    
    def find_plane_np(self, tolerance = 1):
        indices = self.get_atoms_in_the_surface_plane(7, edges_corner=True)
        normal_vector, d = math.get_plane([self.get_position(x) for x in indices[:3]])
    
        plane = []
        for indx in self.get_indices():
            dot_prod = np.dot(self.get_position(indx), normal_vector)
            if dot_prod > d - tolerance and dot_prod < d + tolerance:
                plane.append(indx)
                
        return plane, normal_vector
    
    def add_atom_bridge(self, particle, tolerance, add_atom):
        plane, normal_vector = self.find_plane_np(tolerance)
    
        for bridge_position in bridge_positions:
            atom_positions = [particle.get_position(index) for index in bridge_position]
            point = math.find_middle_point(atom_positions) 
            unit_vector1, length1 = math.get_unit_vector(point)
            unit_vector2, length2 = math.get_unit_vector(normal_vector)
            dot_prod = np.dot(unit_vector1, unit_vector2)
            direction = dot_prod/abs(dot_prod)
            ads = Atoms(add_atom)
            bridge_position = (unit_vector1*(length1))+(direction*unit_vector2*(1.4))
            ads.translate(bridge_position)
            particle.add_atoms(ads)

        return particle

    def add_atom_hollow(self, particle, add_atom):
    
        for hollow_position in self.hollow_positions:
            atom_positions = [particle.get_position(index) for index in hollow_position]
            middle_point = math.find_middle_point(atom_positions)
            normal_vector = math.get_normal_vector(atom_positions) 
            unit_vector1, length1 = math.get_unit_vector(middle_point)
            unit_vector2, length2 = math.get_unit_vector(normal_vector)
            dot_prod = np.dot(unit_vector1, unit_vector2)
            direction = dot_prod/abs(dot_prod)
            ads = Atoms(add_atom)
            bridge_position = (unit_vector1*(length1))+(direction*unit_vector2*(1.4))
            ads.translate(bridge_position)
            particle.add_atoms(ads)

        return particle

class PlaceAddAtoms():
    """Class that plance add atoms on positions identified by FindAdsorptionSites"""
    def __init__(self, symbols):
        self.adsorption_sites = FindAdsorptionSites()
        self.symbols = sorted(symbols)
        self.bridge_positions = {''.join(list(site)) : [] for site in itertools.combinations_with_replacement(self.symbols,2)}
        self.hollow_positions = {''.join(list(site)) : [] for site in itertools.combinations_with_replacement(self.symbols,3)}

    def bind_particle(self, particle):
        self.adsorption_sites.get_bridge_sites(particle)
        self.adsorption_sites.get_hollow_sites(particle)  
        self.get_bridge_sties(particle)
        self.get_hollow_sties(particle)

    def get_on_top_sites(self, particle):
        return NotImplemented

    def get_bridge_sties(self, particle):
        for pairs in self.adsorption_sites.bridge_positions:
            self.bridge_positions[''.join(sorted(particle.get_symbols(pairs)))].append(pairs)
 
    def get_hollow_sties(self, particle):
        for triplet in self.adsorption_sites.hollow_positions:
            self.hollow_positions[''.join(sorted(particle.get_symbols(triplet)))].append(triplet)

    def place_add_atom(self, particle, add_atom_symbol, sites):
        
        for site in sites:
            xyz_atoms = [particle.get_position(atom_index) for atom_index in site]
            xyz_site_plane = math.find_middle_point(xyz_atoms)
            unit_vector1, length1 = math.get_unit_vector(xyz_site_plane)
            if len(site) == 3:
                normal_vector = math.get_normal_vector(xyz_atoms) 
            if len(site) == 2:
                center_of_mass = particle.atoms.atoms.get_center_of_mass()
                normal_vector = math.get_bridge_perpendicular_line(xyz_atoms, center_of_mass)
            unit_vector2, length2 = math.get_unit_vector(normal_vector)
            dot_prod = np.dot(unit_vector1, unit_vector2)
            direction = dot_prod/abs(dot_prod)
            
            add_atom = Atoms(add_atom_symbol)   
            xyz_site = (unit_vector1*(length1))+(direction*unit_vector2*(1.4))
            add_atom.translate(xyz_site)
            particle.add_atoms(add_atom)
            
        return particle





