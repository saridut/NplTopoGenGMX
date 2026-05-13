#!/usr/bin/env python

"""
Class implementing a single NPL crystal.

"""

import warnings
import os
import copy
import math
import numpy as np
from _system import System
from _geom_utils import rotate_vector_axis_angle


class SolventBox(System):
    def __init__(self, name, fns_ff):
        """
        The simulation box = ?

        Parameters
        ----------
        name : str
            Name of the system
        fns_ff : list
            
        """
        super().__init__(name, fns_ff)



    def add_solvent(self, solvent, box_params, packmol_tol=2.0,
                packmol_sidemax=1.0e3, packmol_path=None):
        """
        Adds solvent molecules around a grafted ligand. For a slab positioned at
        the middle, there are two ligands. For a slab positioned at the bottom,
        there is a single ligand.

        Parameters
        ----------
        solvent : tuple
            Solvent molecule to add. The tuple is (Molecule, density), where density
            is in g/mL.
        box_params : dict
            Parameters for changing the simulation box.
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_sidemax : float
            Parameter for Packmol. Default is 1000 angstrom.
        packmol_path : str or pathlib.Path or None
            Path to the packmol executable. If None, Packmol will not be used. In
            this case all added molecules will have their atom positions set to
            zero.

        """
        delta = 0.2 #Small gap of 0.2 nm between periodic images 
                    #(See Packmol manual)
        #Change simulation box size
        if 'boxx' in box_params:
            self.simbox[0,1] = box_params['boxx']/2
            self.simbox[0,0] = -self.simbox[0,1]
        if 'boxy' in box_params:
            self.simbox[1,1] = box_params['boxy']/2
            self.simbox[1,0] = -self.simbox[1,1]
        if 'boxz' in box_params:
            self.simbox[2,1] = box_params['boxz']/2
            self.simbox[2,0] = -self.simbox[2,1]

        #Add solvent molecule type
        self.add_molecule_type(mt=solvent[0])

        #Solvent volume = box volume
        volume = np.prod(self.simbox[:,1]-self.simbox[:,0])

        #Population of molecules 
        molwt = solvent[0].get_total_mass() # in g/mol
        dens = solvent[1] #Density in g/mL
        pop = (602.3*volume*dens)/molwt
        pop = math.floor(pop)

        aid_beg = self.num_atoms + 1 #First solvent atom id
        aid_end = aid_beg + pop*solvent[0].num_atoms - 1 #Last solvent atom id
        mid_beg = self.num_molecules + 1 #First solvent molecule id
        mid_end = mid_beg + pop - 1#Last solvent molecule id

        #Define bounding boxes & add molecules
        bbox_lo = self.simbox[:,0] + delta #Overall bounding box
        bbox_hi = self.simbox[:,1] - delta #Overall bounding box

        print(f"Packing solvents \n"
              f" inside box ({' '.join(['%g'%v for v in bbox_lo])})"
              f" ({' '.join(['%g'%v for v in bbox_hi])}) \n"
              )
        g_in = 'inside box ' \
            + ' '.join(['%g'%(10*v) for v in bbox_lo]) + ' ' \
            + ' '.join(['%g'%(10*v) for v in bbox_hi])

        #Add solvent molecules
        mids = self.add_molecules(solvent[0].name, pop, 1, np.zeros((3,)))
        elem = {'name': solvent[0].name, 'mids': mids, 'constraints': [g_in]}
        mols_to_pack = [elem]

        self.pack_molecules(mols_to_pack, packmol_tol, packmol_sidemax,
                      packmol_path)

        #Create new group of solvent atoms
        self.set_group('Solvent', atom_ids=range(aid_beg, aid_end+1), 
                            molecule_ids=range(mid_beg, mid_end+1))


