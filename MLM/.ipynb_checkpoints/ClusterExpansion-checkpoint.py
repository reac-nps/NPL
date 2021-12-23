#! /usr/bin/env python3.8
from ase.neighborlist import build_neighbor_list
from Core.BaseNanoparticle import BaseNanoparticle as BN
import numpy as np


def get_new_top(hom):
    bn = hom
    bn.neighbor_list.construct
    n_atoms = bn.get_n_atoms()
    coord_pos = [6,7,8,9,12]
    desc_dict = {'Au': {cn : {top :0 for top in range(0,cn+1) } for cn in coord_pos } ,
            'Pt': {cn : {top : 0 for top in range(0,cn+1)} for cn in coord_pos }} 

    for index in range(n_atoms):
        element = bn.get_symbol(index)
        symbols = [bn.get_symbol(index) for index in bn.neighbor_list.list[index]] 
        coord = len(symbols)
        n_pt = symbols.count('Pt')
        desc_dict[element][coord][n_pt] += 1

    flat_desc = []

    for elements in desc_dict.keys():
        elements = desc_dict[elements]
        for coordination in elements.keys():
            coordination = elements[coordination]
            for values in coordination.values():
                flat_desc.append(values)
    
    flat_desc = np.array(flat_desc)

    return flat_desc
