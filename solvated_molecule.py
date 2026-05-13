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


class SolvatedMolecule(System):
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



    def add_mol_one(self, mol_one, lx=0.0, ly=0.0, lz=0.0, restrain=False):
        """
        Adds a single molecule to the system. 

        Parameters
        ----------
        mol_one : Molecule
            An instance of `Molecule`
        lx : float, optional
            Extent of the simulation box along the x-direction
        ly : float, optional
            Extent of the simulation box along the y-direction
        lz : float, optional
            Extent of the simulation box along the z-direction

        """
        #Add molecule type
        _ = self.add_molecule_type(mol_one)

        #Add molecule
        mids = self.add_molecules(mol_one.name, 1, 1, [0.0, 0.0, 0.0])

        #Create group for the added molecule
        self.set_group('Mol_one', molecule_ids=mids)

        #Update simulation box
        com = self.get_com()
        self.translate(r=-com, only_atoms=True) #Move com to the origin
        self.fit_simbox(sep=1.0)
        hmbxl = self.simbox[:,1].max()
        self.simbox[0,1] = max(hmbxl, self.simbox[0,1], lx/2)
        self.simbox[1,1] = max(hmbxl, self.simbox[1,1], ly/2)
        self.simbox[2,1] = max(hmbxl, self.simbox[2,1], lz/2)
        self.simbox[:,0] = -self.simbox[:,1]
        self.translate(r=-self.simbox[:,0]) #Move to the first octant


    def solvate_one(self, solvent, box_params, xtal_offset=0.0, packmol_tol=2.0,
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
        xtal_offset : float
            Distance from the crystal surface beyond which the solvent molecules
            will be placed.
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_sidemax : float
            Parameter for Packmol. Default is 1000 angstrom.
        packmol_path : str or pathlib.Path or None
            Path to the packmol executable. If None, Packmol will not be used. In
            this case all added molecules will have their atom positions set to
            zero.

        """
        raise NotImplementedError
        delta = 0.2 #Small gap of 0.2 nm between periodic images 
                    #(See Packmol manual)
        #Change simulation box size
        if 'boxx' in box_params:
            if not self.is_slab:
                self.simbox[0,1] = box_params['boxx']/2
                self.simbox[0,0] = -self.simbox[0,1]
        if 'boxy' in box_params:
            if not self.is_slab:
                self.simbox[1,1] = box_params['boxy']/2
                self.simbox[1,0] = -self.simbox[1,1]
        if 'boxz' in box_params:
            if self.is_slab and self.slab_pos=='bot':
                self.simbox[2,1] = box_params['boxz']
            else:
                self.simbox[2,1] = box_params['boxz']/2
                self.simbox[2,0] = -self.simbox[2,1]


        #Add solvent molecule type
        self.add_molecule_type(mt=solvent[0])

        #Solvent volume = box volume - crystal volume (in angstrom^3)
        volume = np.prod(self.simbox[:,1]-self.simbox[:,0])
        if 'Xtal' in self.groups.keys():
            xtal_vol = (xtal_hi-xtal_lo).prod()
            volume -= xtal_vol

        #Population of molecules 
        molwt = solvent[0].get_total_mass() # in g/mol
        dens = solvent[1] #Density in g/mL
        nonsolvent_mass = 0.0
        if 'Ligands' in self.groups.keys():
            nonsolvent_atoms = self.groups['Ligands']['atom_ids']
            for each in nonsolvent_atoms:
                nonsolvent_mass += self.atoms[each].mass
        pop = (602.3*volume*dens - nonsolvent_mass)/molwt
        pop = math.floor(pop)

        aid_beg = self.num_atoms + 1 #First solvent atom id
        aid_end = aid_beg + pop*solvent[0].num_atoms - 1 #Last solvent atom id
        mid_beg = self.num_molecules + 1 #First solvent molecule id
        mid_end = mid_beg + pop - 1#Last solvent molecule id

        #Enlarge simulation box if xtal_offset > 0
        if xtal_offset > 0:
            if self.is_slab:
                if self.slab_pos == 'bot':
                    self.simbox[2,1] += xtal_offset + 1
                else:
                    self.simbox[2,0] -= (xtal_offset + 1)
                    self.simbox[2,1] += (xtal_offset + 1)

        #Define bounding boxes & add molecules
        oa_bbox_lo = self.simbox[:,0] + delta #Overall bounding box
        oa_bbox_hi = self.simbox[:,1] - delta #Overall bounding box

        if self.is_slab:
            if self.slab_pos == 'bot':
                #Ligand bounding box (Inner bounding box for solvent molecules)
                lig_lo, lig_hi = self.get_bbox(self.groups['Ligands']['atom_ids'])
                lig_lo[0:2] -= delta
                lig_hi += delta
                #Outer bounding box for solvent molecules
                bbox_lo = [oa_bbox_lo[0], oa_bbox_lo[1], xtal_hi[2]+xtal_offset]
                bbox_hi = oa_bbox_hi
                print(f"Packing solvents \n"
                      f" inside box ({' '.join(['%g'%(10*v) for v in bbox_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in bbox_hi])}) \n"
                      f" outside box ({' '.join(['%g'%(10*v) for v in lig_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in lig_hi])})"
                      )
                g_in = 'inside box ' \
                    + ' '.join(['%g'%(10*v) for v in bbox_lo]) + ' ' \
                    + ' '.join(['%g'%(10*v) for v in bbox_hi])
                g_out = 'outside box ' \
                    + ' '.join(['%g'%(10*v) for v in lig_lo]) + ' ' \
                    + ' '.join(['%g'%(10*v) for v in lig_hi])
                #Add solvent molecules
                mids = self.add_molecules(solvent[0].name, pop, 1, np.zeros((3,)))
                elem = {'name': solvent[0].name, 'mids': mids, 
                        'constraints': [g_in, g_out]}
                mols_to_pack = [elem]
            else:
                #Bounding boxes for the top & bottom halves
                pop_top = int(0.5*pop); pop_bot = pop - pop_top

                top_bbox_hi = oa_bbox_hi
                top_bbox_lo = [ oa_bbox_lo[0], oa_bbox_lo[1],
                               xtal_hi[2]+xtal_offset ]

                bot_bbox_lo = oa_bbox_lo
                bot_bbox_hi = [ oa_bbox_hi[0], oa_bbox_hi[1],
                               xtal_lo[2]-xtal_offset ]

                #Add solvent molecules
                mids_top = self.add_molecules(solvent[0].name, pop_top, 1,
                                              np.zeros((3,)))
                mids_bot = self.add_molecules(solvent[0].name, pop_bot, 1,
                                              np.zeros((3,)))
                #Ligand bounding boxes
                mids_lig = self.groups['Ligands']['molecule_ids']
                #Atom id of the first ligand molecule
                abeg = self.molecules[mids_lig[0]].beg 
                posz = self.atoms[abeg].coords[2]
                if posz > 0:
                    mid_top_lig = mids_lig[0]
                    mid_bot_lig = mids_lig[1]
                else:
                    mid_top_lig = mids_lig[1]
                    mid_bot_lig = mids_lig[0]

                aids_top_lig = range(self.molecules[mid_top_lig].beg,
                                    self.molecules[mid_top_lig].end+1)
                top_lig_lo, top_lig_hi = self.get_bbox(aids_top_lig)
                top_lig_lo[0:2] -= delta; top_lig_lo[2] += delta
                top_lig_hi += delta

                aids_bot_lig = range(self.molecules[mid_bot_lig].beg,
                                    self.molecules[mid_bot_lig].end+1)
                bot_lig_lo, bot_lig_hi = self.get_bbox(aids_bot_lig)
                bot_lig_lo -= delta
                bot_lig_hi[0:2] += delta; bot_lig_hi[2] -= delta

                print(f"Packing solvents \n"
                      f"Top part \n"
                      f" inside box ({' '.join(['%g'%(10*v) for v in top_bbox_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in top_bbox_hi])}) \n"
                      f" outside box ({' '.join(['%g'%(10*v) for v in top_lig_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in top_lig_hi])}) \n"
                      f"\n"
                      f"Bottom part \n"
                      f" inside box ({' '.join(['%g'%(10*v) for v in bot_bbox_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in bot_bbox_hi])}) \n"
                      f" outside box ({' '.join(['%g'%(10*v) for v in bot_lig_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in bot_lig_hi])})"
                      )
                g_top_in = 'inside box ' \
                        + ' '.join(['%g'%(10*v) for v in top_bbox_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in top_bbox_hi])
                g_top_out = 'outside box ' \
                        + ' '.join(['%g'%(10*v) for v in top_lig_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in top_lig_hi])
                elem = {'name': solvent[0].name, 'mids': mids_top,
                        'constraints': [g_top_in, g_top_out]}
                mols_to_pack = [elem]

                g_bot_in = 'inside box ' \
                        + ' '.join(['%g'%(10*v) for v in bot_bbox_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in bot_bbox_hi])
                g_bot_out = 'outside box ' \
                        + ' '.join(['%g'%(10*v) for v in bot_lig_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in bot_lig_hi])
                elem = {'name': solvent[0].name, 'mids': mids_bot,
                        'constraints': [g_bot_in, g_bot_out]}
                mols_to_pack.append(elem)

        self.pack_molecules(mols_to_pack, packmol_tol, packmol_sidemax,
                      packmol_path)
        #Create new group of solvent atoms
        self.set_group('Solvent', atom_ids=range(aid_beg, aid_end+1), 
                            molecule_ids=range(mid_beg, mid_end+1))


