from Core.Nanoparticle import Nanoparticle
import Core.MathModules as math

from ase.cluster import Octahedron
from ase import Atoms

class Octhaedral(Nanoparticle):
    def __init__(self):
        Nanoparticle.__init__(self)
        
    def octahedron(self, height, cutoff, stoichiometry, lattice_constant=3.9, alloy=False):
        octa = Octahedron('Cu', height, cutoff=0, latticeconstant=lattice_constant, alloy=alloy)
        atoms = Atoms(octa.symbols, octa.positions)
        com = atoms.get_center_of_mass()
        atoms.positions -= com

        self.add_atoms(atoms, recompute_neighbor_list=False)
        #self.random_ordering(stoichiometry)
        self.construct_neighbor_list()
        
    def addatom_ontop(self, indices, distance):
        atoms_addatoms = self.get_ase_atoms()
        
        for index in indices:
            cn = self.get_coordination_number(index)
            position = self.get_position(index)
            unit, length = math.get_unit_vector(position)
            tilted_vector, _ = math.get_unit_vector(unit + math.get_perpendicular_vector(unit))
        
            if cn == 4:
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

            add_atom1 = Atoms('C')
            add_atom2 = Atoms('O')    
            add_atom1.translate(C_distance)
            add_atom2.translate(O_distance)
            atoms_addatoms += add_atom1
            atoms_addatoms += add_atom2
        
        return atoms_addatoms
