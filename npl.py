#!/usr/bin/env python

"""
Class implementing a single NPL crystal.

"""

import warnings
import os
import copy
import math
import numpy as np
from crystal import get_lattice_points
from _system import System
from _geom_utils import rotate_vector_axis_angle


class NanoPlatelet(System):
    def __init__(self, name, fns_ff, is_slab, slab_pos):
        """
        The simulation box is centered in the xy-plane with its bottom surface
        at z = 0.

        Parameters
        ----------
        is_slab : bool
            Whether the crystal is a slab or not. A slab is periodic in the x &
            y directions and parallel to the xy-plane.
        slab_pos : {'bot', 'mid'}
            Whether the slab is at the bottom or at the middle of the box. If
            `bot`, the lower z-bound of the simulation box will be at z = 0. For
            all other cases, the simulation box is centered at the origin =
            (0,0,0) and the crystal is placed at the center of the box.
            
        """
        super().__init__(name, fns_ff)
        self.is_slab = is_slab
        self.slab_pos = slab_pos
        self.apl = None   #Area per ligand
        self.xlpar = None



    def add_xtal(self, lx, ly, lz, xlpar, xmol, out_dir=None):
        """
        Adds a crystal normal to the z-axis onto whose surface(s) the ligands will
        be grafted.

        Parameters
        ----------
        lx : float
            Extent of the crystal along the x-direction
        ly : float
            Extent of the crystal along the y-direction
        lz : float
            Extent of the crystal along the z-direction
        xlpar : float
            Lattice parameter of the crystal
        xmol : Molecule
            Wall molecule types

        """
        if out_dir is None:
            odir = os.getcwd()
        else:
            path = os.path.expanduser(out_dir)
            if os.path.exists(path):
                odir = path
            else:
                warnings.warn(f"Directory {out_dir} does not exist, using"
                              " current working directory.") 
                odir = os.getcwd()

        self.xlpar = xlpar
        self.add_molecule_type(xmol)
        hlx = 0.5*self.xlpar*math.ceil(lx/self.xlpar)
        hly = 0.5*self.xlpar*math.ceil(ly/self.xlpar)
        hlz = 0.5*self.xlpar*math.ceil(lz/self.xlpar)
        if self.is_slab and (self.slab_pos == 'bot'):
            lo = [-hlx, -hly, 0]; hi = [hlx, hly, 2*hlz]
        else:
            lo = [-hlx, -hly, -hlz]; hi = [hlx, hly, hlz]

        xa_coords = get_lattice_points('fcc', self.xlpar, lo, hi, boundary='ppc')

        if self.is_slab:
            self.add_simbox(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
            print(f"Surface area = {(hi[0]-lo[0])*(hi[1]-lo[1]):g} nm^2",
                    file=open(f"{odir}/info.txt", 'w'))
        else:
            center = xa_coords.mean(axis=0)
            xa_coords[:,:] -= center
            lb = xa_coords.min(axis=0)
            ub = xa_coords.max(axis=0)
            self.add_simbox(lb[0], ub[0], lb[1], ub[1], lb[2], ub[2])

        #Crystal atoms types: Assuming only a single atom type
        _ = self.add_molecule_type(mt=xmol)

        #Crystal atoms ids
        aid_beg = self.num_atoms + 1 # First crystal atom id
        aid_end = aid_beg + xa_coords.shape[0] - 1 #Last crystal atom id

        #Add atom positions
        self.add_molecules(xmol.name, xa_coords.shape[0], 1, xa_coords)

        #Atoms ids of the top & bottom crystal surface
        xa_zcoords = xa_coords[:,2]
        zlo = xa_zcoords.min(); zhi = xa_zcoords.max()

        top_mask = np.isclose(xa_zcoords, zhi)
        aids_xst = (aid_beg + np.nonzero(top_mask)[0]).tolist()

        bot_mask = np.isclose(xa_zcoords, zlo)
        aids_xsb = (aid_beg + np.nonzero(bot_mask)[0]).tolist()

        #Add groups
        atoms = range(aid_beg, aid_end+1)
        self.set_group('Xtal', atom_ids=atoms)
        self.set_group('XtalSurfTop', atom_ids=aids_xst)
        self.set_group('XtalSurfBot', atom_ids=aids_xsb)



    def add_ligand_one(self, ligand, r0, intermolecular_bond_func=None, 
                    intermolecular_bond_params=None, out_dir=None):
        """
        Adds a single ligand to the top & bottom surface (for a slab positioned
        at the middle). For a slab positioned at the bottom of the box, a single
        ligand is added only to the top surface. The ligand(s) is (are)
        positioned at the center of the surface(s). The resulting configuration
        will have an enlarged simulation box to include the ligands.

        Parameters
        ----------

        ligand: LigandMolecule
            Ligand molecule to add.
        r0 : float
            Distance of the ligand heads from the crystal surface

        """
        if out_dir is None:
            odir = os.getcwd()
        else:
            path = os.path.expanduser(out_dir)
            if os.path.exists(path):
                odir = path
            else:
                warnings.warn(f"Directory {out_dir} does not exist, using"
                              " current working directory.") 
                odir = os.getcwd()

        rng = np.random.default_rng()

        #Add ligand molecule types
        self.add_molecule_type(ligand)

        #Crystal bounding box 
        xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atom_ids'])

        #Add ligand molecules
        mid_beg = self.num_molecules + 1
        aids_head = []
        if self.is_slab:
            if self.slab_pos == 'bot':
                normal = np.array([0, 0, 1])
                pop = 1
                gps = np.array([(self.simbox[0,0]+self.simbox[0,1])/2,
                                (self.simbox[1,0]+self.simbox[1,1])/2, 
                                 xtal_hi[2]+r0 ])
                mids = self.add_molecules(ligand.name, pop, ligand.head, gps)
                mid_end = mids[-1]
                #Single molecule: only one element in mids
                aid_beg = self.molecules[mid_end].beg
                aid_end = self.molecules[mid_end].end
                aid_head = aid_beg + ligand.head - 1
                aid_tail = aid_beg + ligand.tail - 1
                aids_head.append(aid_head)
                self.align(aid_head, aid_tail, axis=normal,
                           atm_ids=range(aid_beg, aid_end+1))
                angle = 2*math.pi*rng.random()
                pivot = self.atoms[aid_head].coords
                self.rotate(angle, normal, pivot,
                            atm_ids=range(aid_beg, aid_end+1))
            else:
                #Top surface
                normal = np.array([0, 0, 1])
                pop = 1
                gps = np.array([(self.simbox[0,0]+self.simbox[0,1])/2,
                                (self.simbox[1,0]+self.simbox[1,1])/2, 
                                 xtal_hi[2]+r0 ])
                mids = self.add_molecules(ligand.name, pop, ligand.head, gps)
                mid_end = mids[-1]
                #Single molecule: only one element in mids
                aid_beg = self.molecules[mid_end].beg
                aid_end = self.molecules[mid_end].end
                aid_head = aid_beg + ligand.head - 1
                aid_tail = aid_beg + ligand.tail - 1
                aids_head.append(aid_head)
                self.align(aid_head, aid_tail, axis=normal,
                           atm_ids=range(aid_beg, aid_end+1))
                angle = 2*math.pi*rng.random()
                pivot = self.atoms[aid_head].coords
                self.rotate(angle, normal, pivot,
                            atm_ids=range(aid_beg, aid_end+1))

                #Bottom surface
                normal = np.array([0, 0, -1])
                pop = 1
                gps = np.array([(self.simbox[0,0]+self.simbox[0,1])/2,
                                (self.simbox[1,0]+self.simbox[1,1])/2, 
                                 xtal_lo[2]-r0 ])
                mids = self.add_molecules(ligand.name, pop, ligand.head, gps)
                mid_end = mids[-1]
                #Single molecule: only one element in mids
                aid_beg = self.molecules[mid_end].beg
                aid_end = self.molecules[mid_end].end
                aid_head = aid_beg + ligand.head - 1
                aid_tail = aid_beg + ligand.tail - 1
                aids_head.append(aid_head)
                self.align(aid_head, aid_tail, axis=normal,
                           atm_ids=range(aid_beg, aid_end+1))
                angle = 2*math.pi*rng.random()
                pivot = self.atoms[aid_head].coords
                self.rotate(angle, normal, pivot,
                            atm_ids=range(aid_beg, aid_end+1))

        #Create group for ligands
        self.set_group('Ligands', molecule_ids=range(mid_beg, mid_end+1))
        #Create group for ligand heads
        self.set_group('LigandHeads', atom_ids=aids_head)
        #self.write_group(['LigandHeads'], f"{odir}/GrpLigHead.ndx")

        #Add intermolecular bonds
        if self.is_slab:
            if self.slab_pos == 'bot':
                aids_xtal = self.groups['XtalSurfTop']['atom_ids']
            else:
                aids_xtal = self.groups['XtalSurfTop']['atom_ids'] + \
                            self.groups['XtalSurfBot']['atom_ids']

        for aid_head in self.groups['LigandHeads']['atom_ids']:
            ri = self.atoms[aid_head].coords
            rijmagsq_min = math.inf
            for aid_xtal in aids_xtal:
                rj = self.atoms[aid_xtal].coords
                rij = rj - ri
                rijmagsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                if rijmagsq < rijmagsq_min:
                    rijmagsq_min = rijmagsq
                    aid_xtal_min = aid_xtal
            self.add_intermolecular_bond(aid_head, aid_xtal_min,
                                        func=intermolecular_bond_func,
                                        params=intermolecular_bond_params)
        #self.reindex() #Single ligand type, reindexing not necessary
        #Update simulation box
        if self.is_slab:
            lo, hi = self.get_bbox(self.groups['Ligands']['atom_ids'])
            if self.slab_pos == 'bot':
                self.simbox[2,1] = hi[2]
            else:
                self.simbox[2,1] = hi[2]; self.simbox[2,0] = -self.simbox[2,1]
        else:
            self.fit_simbox()



    def add_ligands(self, ligands, pop_ratio, r0, balance_charge, lattice, apl,
                    intermolecular_bond_func=None, 
                    intermolecular_bond_params=None, out_dir=None):
        """
        Adds ligands to a bare crystal. The resulting configuration will have
        an enlarged simulation box to include the ligands.

        Parameters
        ----------

        ligands: list of LigandMolecule
            Ligand molecules to add.
        pop_ratio : list of int
            In case of multiple ligand molecules, the ratio of ligand population
            for each type. Only considers monovalent ligands. E.g., for two
            ligand types in 2:1 ratio, use [2, 1].
        r0 : float
            Distance of the ligand heads from the crystal surface
        balance_charge : bool
            Determine ligand population from neutralizing the total charge on
            the crystal.
        lattice : {'sc', 'bcc', None}
            Lattice for arranging the ligand molecule graft points. If None,
            ligands are distributed randomly.
        apl : float or None
            Area per ligand in angstrom^2. Ignored if `balance_charge` is True.

        """
        if out_dir is None:
            odir = os.getcwd()
        else:
            path = os.path.expanduser(out_dir)
            if os.path.exists(path):
                odir = path
            else:
                warnings.warn(f"Directory {out_dir} does not exist, using"
                              " current working directory.") 
                odir = os.getcwd()

        rng = np.random.default_rng()

        #Add ligand molecule types
        for each in ligands:
            self.add_molecule_type(mt=each)

        #Ligand population
        if balance_charge:
            pop = self._ligpop_from_charge(ligands, pop_ratio)
        else:
            pop = self._ligpop_from_area(ligands, pop_ratio, apl, out_dir)

        #Assign ligands to surfaces
        surfaces = self._ligands_to_surfaces(ligands, pop, r0, lattice)

        #Add ligand molecules
        mid_beg = self.num_molecules + 1
        aids_head = []
        for surf in surfaces.values():
            normal = surf['normal']
            pop = surf['pop']
            gp_indx = self._choose(pop)
            for i, lg in enumerate(ligands):
                gps = surf['graft_points'][gp_indx[i]] #Graft points
                mids = self.add_molecules(lg.name, pop[i], lg.head, gps)
                mid_end = mids[-1]
                for mid in mids: 
                    aid_beg = self.molecules[mid].beg
                    aid_end = self.molecules[mid].end
                    aid_head = aid_beg + lg.head - 1
                    aid_tail = aid_beg + lg.tail - 1
                    aids_head.append(aid_head)
                    self.align(aid_head, aid_tail, axis=normal,
                               atm_ids=range(aid_beg, aid_end+1))
                    angle = 2*math.pi*rng.random()
                    pivot = self.atoms[aid_head].coords
                    self.rotate(angle, normal, pivot,
                                atm_ids=range(aid_beg, aid_end+1))
        #Create group for ligands
        self.set_group('Ligands', molecule_ids=range(mid_beg, mid_end+1))
        #Create group for ligand heads
        self.set_group('LigandHeads', atom_ids=aids_head)
        #self.write_group(['LigandHeads'], f"{odir}/GrpLigHead.ndx")

        #Add intermolecular bonds
        if self.is_slab:
            if self.slab_pos == 'bot':
                aids_xtal = self.groups['XtalSurfTop']['atom_ids']
            else:
                aids_xtal = self.groups['XtalSurfTop']['atom_ids'] + \
                            self.groups['XtalSurfBot']['atom_ids']
        else:
            aids_xtal = self.groups['Xtal']['atom_ids']

        for aid_head in self.groups['LigandHeads']['atom_ids']:
            ri = self.atoms[aid_head].coords
            rijmagsq_min = math.inf
            for aid_xtal in aids_xtal:
                rj = self.atoms[aid_xtal].coords
                rij = rj - ri
                rijmagsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                if rijmagsq < rijmagsq_min:
                    rijmagsq_min = rijmagsq
                    aid_xtal_min = aid_xtal
            self.add_intermolecular_bond(aid_head, aid_xtal_min,
                                        func=intermolecular_bond_func,
                                        params=intermolecular_bond_params)
        self.reindex()
        #Update simulation box
        if self.is_slab:
            lo, hi = self.get_bbox(self.groups['Ligands']['atom_ids'])
            if self.slab_pos == 'bot':
                self.simbox[2,1] = hi[2]
            else:
                self.simbox[2,1] = hi[2]; self.simbox[2,0] = -self.simbox[2,1]
        else:
            self.fit_simbox()


    def _choose(self, pop):
        """
        Assign types to each ligand molecule graft point (needed for random
        mixture of different ligand molecules)

        Parameters
        ----------
        pop : int, array-like
            Population of each ligand species

        Returns
        -------
        List of 1-D numpy array of ints

        """
        rng = np.random.default_rng()
        pop_tot = sum(pop)
        indices = list(range(pop_tot))
        chosen = []
        for i,n in enumerate(pop):
            indx = rng.choice(indices, size=n, replace=False, shuffle=False)
            chosen.append(indx) 
            for j in indx:
                indices.remove(j)
        return chosen


    def _ligpop_from_area(self, ligands, pop_ratio, apl, out_dir):
        """
        Determines population of each ligand species based on surface area.

        """
        #Crystal bounding box 
        lo, hi = self.get_bbox(self.groups['Xtal']['atom_ids'])
        #Total surface area
        if self.is_slab:
            area_tot = (hi[0]-lo[0])*(hi[1]-lo[1])
            if self.slab_pos != 'bot':
                area_tot *= 2
        else:
            dr = hi - lo
            area_tot = 2*(dr[0]*dr[1] + dr[1]*dr[2] + dr[2]*dr[0])
        #Population of each ligand species
        num_ligands = area_tot/apl #This is a float
        s = sum(pop_ratio); pop_frac = [x/s for x in pop_ratio]
        pop = [round(num_ligands*f) for f in pop_frac]
        self.apl = area_tot/sum(pop)
        print(f"Area/ligand = {self.apl:f} nm^2\n"
              f"Brush density = {1/self.apl:f}/nm^2", 
              file=open(f"{out_dir}/info.txt", 'a'))
        return pop


    def _ligpop_from_charge(self, ligands, pop_ratio):
        """
        Determines population of each ligand species to balance the charge of
        the crystal.

        """
        #Crystal bounding box 
        lo, hi = self.get_bbox(self.groups['Xtal']['atom_ids'])
        #Total surface area
        if self.is_slab:
            area_tot = (hi[0]-lo[0])*(hi[1]-lo[1])
            if self.slab_pos != 'bot':
                area_tot *= 2
        else:
            dr = hi - lo
            area_tot = 2*(dr[0]*dr[1] + dr[1]*dr[2] + dr[2]*dr[0])

        totchg = self.get_total_charge()
        ligchg = [each.get_total_charge() for each in ligands]
        s = sum(pop_ratio); pop_frac = [x/s for x in pop_ratio]
        pop_tot = abs(totchg)/sum([abs(f*c) for f,c in zip(pop_frac,ligchg)])
        pop = [round(pop_tot*f) for f in pop_frac]

        residual_charge = totchg + sum([p*c for p,c in zip(pop,ligchg)])
        print (f"Residual charge after adding ligands = {residual_charge}")

        self.apl = area_tot/sum(pop)
        print(f"Area/ligand = {self.apl:f} A^2\n"
              f"Brush density = {1/self.apl:f}/A^2")
        return pop


    def _get_graft_points(self, n, lo, hi, lattice, boundary):
        """

        """
        rng = np.random.default_rng()

        if math.isclose(hi[0], lo[0]):
            area = (hi[1]-lo[1])*(hi[2]-lo[2])
        elif math.isclose(hi[1], lo[1]):
            area = (hi[2]-lo[2])*(hi[0]-lo[0])
        elif math.isclose(hi[2], lo[2]):
            area = (hi[0]-lo[0])*(hi[1]-lo[1])

        apl = area/n
        if lattice == 'sc':
            lpar = math.sqrt(apl)
        elif lattice == 'bcc':
            lpar = math.sqrt(2*apl)

        sites = get_lattice_points(lattice, lpar, lo, hi, boundary='ppp',
                                    num_sites=n)
        graft_points = rng.choice(sites, size=n, replace=False)
        return graft_points


    def _ligands_to_surfaces(self, ligands, pop, r0, lattice):
        """

        """
        surfaces = {} #Keys: 'name', Values= {'normal', 'pop', 'graft_points'}
        #Crystal bounding box 
        xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atom_ids'])
        #Bounding box for ligand surface area
        if self.is_slab:
            if self.slab_pos == 'bot':
                lo = [self.simbox[0,0], self.simbox[1,0], xtal_hi[2]+r0]
                hi = [self.simbox[0,1], self.simbox[1,1], xtal_hi[2]+r0]
                gp = self._get_graft_points(sum(pop), lo, hi, lattice, 'ppp')
                surfaces['top'] = {'normal': np.array([0,0,1]), 'pop': pop,
                                    'graft_points': gp}
            else:
                #Top surface
                lo = [self.simbox[0,0], self.simbox[1,0], xtal_hi[2]+r0]
                hi = [self.simbox[0,1], self.simbox[1,1], xtal_hi[2]+r0]
                pop_top = [x//2 for x in pop]
                n = sum(pop_top)
                gp = self._get_graft_points(n, lo, hi, lattice, 'ppp')
                surfaces['top'] = {'normal': np.array([0,0,1]),
                                   'pop': pop_top, 'graft_points': gp}
                #Bottom surface
                lo = [self.simbox[0,0], self.simbox[1,0], xtal_lo[2]-r0]
                hi = [self.simbox[0,1], self.simbox[1,1], xtal_lo[2]-r0]
                pop_bot = [x-y for x,y in zip(pop,pop_top)]
                n = sum(pop_bot)
                gp = self._get_graft_points(n, lo, hi, lattice, 'ppp')
                surfaces['bot'] = {'normal': np.array([0,0,-1]),
                                   'pop': pop_bot, 'graft_points': gp}
        else:
            dx = xtal_hi[0] - xtal_lo[0] + 2*r0
            dy = xtal_hi[1] - xtal_lo[1] + 2*r0
            dz = xtal_hi[2] - xtal_lo[2] + 2*r0
            areas = [2*dy*dz, 2*dx*dz, 2*dx*dy]
            area_tot = sum(areas)

            ind = np.argsort(areas)
            pop_ = [[], [], []]
            pop_[ind[0]] = [int(areas[ind[0]]*x/area_tot) for x in pop]
            pop_[ind[1]] = [int(areas[ind[1]]*x/area_tot) for x in pop]
            pop_[ind[2]] = [z-x-y for 
                            x,y,z in zip(pop_[ind[0]], pop_[ind[1]], pop)]
            popx = pop_[0]; popy = pop_[1]; popz = pop_[2]

            #Surface normal to the x-axis: Left 
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_lo[0]-r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_left = [x//2 for x in popx]
            n = sum(pop_left)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['left'] = {'normal': np.array([-1,0,0]),
                                'pop': pop_left, 'graft_points': gp}

            #Surface normal to the x-axis: Right 
            lo = [xtal_hi[0]+r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_right = [x-y for x,y in zip(popx,pop_left)]
            n = sum(pop_right)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['right'] = {'normal': np.array([1,0,0]),
                                'pop': pop_right, 'graft_points': gp}

            #Surface normal to the y-axis: Front
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_lo[1]-r0, xtal_hi[2]+r0]
            pop_front = [x//2 for x in popy]
            n = sum(pop_front)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['front'] = {'normal': np.array([0,-1,0]),
                                'pop': pop_front, 'graft_points': gp}

            #Surface normal to the y-axis: Back
            lo = [xtal_lo[0]-r0, xtal_hi[1]+r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_back = [x-y for x,y in zip(popy,pop_front)]
            n = sum(pop_back)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['back'] = {'normal': np.array([0,1,0]),
                                'pop': pop_back, 'graft_points': gp}

            #Surface normal to the z-axis: Bottom
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_lo[2]-r0]
            pop_bot = [x//2 for x in popz]
            n = sum(pop_bot)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['bot'] = {'normal': np.array([0,0,-1]),
                               'pop': pop_bot, 'graft_points': gp}

            #Surface normal to the z-axis: Top
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_hi[2]+r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_top = [x-y for x,y in zip(popz,pop_bot)]
            n = sum(pop_top)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['top'] = {'normal': np.array([0,0,1]),
                               'pop': pop_top, 'graft_points': gp}
        return surfaces



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

        if 'Xtal' in self.groups:
            xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atom_ids'])
            if 'sepx' in box_params:
                if not self.is_slab:
                    self.simbox[0,1] = xtal_hi[0] + box_params['sepx']
                    self.simbox[0,0] = -self.simbox[0,1]
            if 'sepy' in box_params:
                if not self.is_slab:
                    self.simbox[1,1] = xtal_hi[1] + box_params['sepy']
                    self.simbox[1,0] = -self.simbox[1,1]
            if 'sepz' in box_params:
                if self.is_slab and self.slab_pos=='bot':
                    self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                else:
                    self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                    self.simbox[2,0] = -self.simbox[2,1]
            if 'make_cubic' in box_params and box_params['make_cubic']:
                if self.is_slab and \
                        math.isclose(self.simbox[0,1], self.simbox[1,1]):
                    if self.slab_pos == 'bot':
                        self.simbox[2,1] = 2*self.simbox[0,1]
                    else:
                        self.simbox[2,1] = self.simbox[0,1]
                        self.simbox[2,0] = -self.simbox[2,1]
                else:
                    v = self.simbox[:,1].max()
                    self.simbox[:,1] = v
                    self.simbox[:,0] = -self.simbox[:,1]

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



    def solvate(self, solvent, box_params, xtal_offset=0.0, packmol_tol=2.0,
                packmol_sidemax=1.0e3, packmol_path=None):
        """
        Adds solvent molecules around a nanoplatelet.

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

        if 'Xtal' in self.groups:
            xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atom_ids'])
            if 'sepx' in box_params:
                if not self.is_slab:
                    self.simbox[0,1] = xtal_hi[0] + box_params['sepx']
                    self.simbox[0,0] = -self.simbox[0,1]
            if 'sepy' in box_params:
                if not self.is_slab:
                    self.simbox[1,1] = xtal_hi[1] + box_params['sepy']
                    self.simbox[1,0] = -self.simbox[1,1]
            if 'sepz' in box_params:
                if self.is_slab and self.slab_pos=='bot':
                    self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                else:
                    self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                    self.simbox[2,0] = -self.simbox[2,1]
            if 'make_cubic' in box_params and box_params['make_cubic']:
                if self.is_slab and \
                        math.isclose(self.simbox[0,1], self.simbox[1,1]):
                    if self.slab_pos == 'bot':
                        self.simbox[2,1] = 2*self.simbox[0,1]
                    else:
                        self.simbox[2,1] = self.simbox[0,1]
                        self.simbox[2,0] = -self.simbox[2,1]
                else:
                    v = self.simbox[:,1].max()
                    self.simbox[:,1] = v
                    self.simbox[:,0] = -self.simbox[:,1]

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
                    self.simbox[2,1] += xtal_offset
                else:
                    self.simbox[2,0] -= (xtal_offset+0)
                    self.simbox[2,1] += (xtal_offset+0)
            else:
                self.simbox[:,0] -= xtal_offset
                self.simbox[:,1] += xtal_offset

        #Define bounding boxes & add molecules
        oa_bbox_lo = self.simbox[:,0] + delta #Overall bounding box
        oa_bbox_hi = self.simbox[:,1] - delta #Overall bounding box

        if self.is_slab:
            if self.slab_pos == 'bot':
                #A single bounding box for solvent molecules
                bbox_lo = [oa_bbox_lo[0], oa_bbox_lo[1], xtal_hi[2]+xtal_offset]
                bbox_hi = oa_bbox_hi
                print(f"Packing solvents \n"
                      f" inside box ({' '.join(['%g'%(10*v) for v in bbox_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in bbox_hi])})"
                      )
                g = 'inside box ' \
                    + ' '.join(['%g'%(10*v) for v in bbox_lo]) + ' ' \
                    + ' '.join(['%g'%(10*v) for v in bbox_hi])
                #Add solvent molecules
                mids = self.add_molecules(solvent[0].name, pop, 1, np.zeros((3,)))
                elem = {'name': solvent[0].name, 'mids': mids, 'constraints': [g]}
                mols_to_pack = [elem]
            else:
                #Two bounding boxes for the top & bottom halves
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

                print(f"Packing solvents \n"
                      f" inside box ({' '.join(['%g'%(10*v) for v in top_bbox_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in top_bbox_hi])}) \n"
                      f" inside box ({' '.join(['%g'%(10*v) for v in bot_bbox_lo])})"
                      f" ({' '.join(['%g'%(10*v) for v in bot_bbox_hi])})"
                      )
                g_top = 'inside box ' \
                    + ' '.join(['%g'%(10*v) for v in top_bbox_lo]) + ' ' \
                    + ' '.join(['%g'%(10*v) for v in top_bbox_hi])
                elem = {'name': solvent[0].name, 'mids': mids_top,
                        'constraints': [g_top]}
                mols_to_pack = [elem]

                g_bot = 'inside box ' \
                    + ' '.join(['%g'%(10*v) for v in bot_bbox_lo]) + ' ' \
                    + ' '.join(['%g'%(10*v) for v in bot_bbox_hi])
                elem = {'name': solvent[0].name, 'mids': mids_bot,
                        'constraints': [g_bot]}
                mols_to_pack.append(elem)
        else:
            #Add solvent molecules to the region between two bounding boxes
            in_bbox_lo = xtal_lo - xtal_offset
            in_bbox_hi = xtal_hi + xtal_offset
            out_bbox_lo = oa_bbox_lo
            out_bbox_hi = oa_bbox_hi

            #Add solvent molecules
            mids = self.add_molecules(solvent[0].name, pop, 1, np.zeros((3,)))
            
            print(f"Packing solvents \n"
                  f" inside box ({' '.join(['%g'%(10*v) for v in out_bbox_lo])})"
                  f" ({' '.join(['%g'%(10*v) for v in out_bbox_hi])}) \n"
                  f" outside box ({' '.join(['%g'%(10*v) for v in in_bbox_lo])})"
                  f" ({' '.join(['%g'%(10*v) for v in in_bbox_hi])})"
                  )
            g_in = 'inside box ' \
                + ' '.join(['%g'%(10*v) for v in out_bbox_lo]) + ' ' \
                + ' '.join(['%g'%(10*v) for v in out_bbox_hi])

            g_out = 'outside box ' \
                + ' '.join(['%g'%(10*v) for v in in_bbox_lo]) + ' ' \
                + ' '.join(['%g'%(10*v) for v in in_bbox_hi])

            elem = {'name': solvent[0].name, 'mids': mids,
                    'constraints': [g_in, g_out]}
            mols_to_pack = [elem]

        self.pack_molecules(mols_to_pack, packmol_tol, packmol_sidemax,
                      packmol_path)
        #Create new group of solvent atoms
        self.set_group('Solvent', atom_ids=range(aid_beg, aid_end+1), 
                            molecule_ids=range(mid_beg, mid_end+1))


