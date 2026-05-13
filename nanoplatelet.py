#!/usr/bin/env python

"""
Class implementing a single NPL crystal.

"""

import warnings
import os
import sys
import copy
import math
import numpy as np
from contextlib import nullcontext
from crystal_new import get_lattice_points, make_npl_xtal
from _system import System
from _geom_utils import rotate_vector_axis_angle


class Nanoplatelet(System):
    def __init__(self, name, fns_ff, form, slab_pos='mid'):
        """
        The simulation box is centered in the xy-plane with its bottom surface
        at z = 0.

        Parameters
        ----------
        form : {'ribbon', 'slab', 'finite'}
            Whether the crystal is a ribbon (periodic along y), slab (periodic
            along x & y) or a finite object (nonperiodic). 
        slab_pos : {'bot', 'mid'}
            Whether the slab is at the bottom or at the middle of the box. If
            `bot`, the lower z-bound of the simulation box will be at z = 0. For
            all other cases, the simulation box is centered at the origin =
            (0,0,0) and the crystal is placed at the center of the box.
            
        """
        if form not in ['ribbon', 'slab', 'finite']:
            raise ValueError("self.form must be {'ribbon' | 'slab' | 'finite'}")
        if form == 'slab' and slab_pos not in ['bot', 'mid']:
            raise ValueError("self.slab_pos must be {'bot' | 'mid'}")
        super().__init__(name, fns_ff)
        self.form = form
        self.slab_pos = slab_pos



    def add_xtal(self, lx, ly, lz, lattice, species, a, c=None,  
                 orient=['1_0_0', '0_1_0', '0_0_1'], unit=['nm', 'nm', 'nm'],
                 mlcp=[], atomsk_path='atomsk', fn_info=None):
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
        lattice : str
            Lattice type. 
        species : list of atomic species
            At most two atoms.
        a : float
            Lattice parameter (in nm)
        c : float, optional
            Lattice parameter (in nm)
        orient : Sequence of 3 str
            Lattice orientation in Miller indices.
        unit : List of three strings
            {'nm'|'LU'| 'ML'}
        mlcp : list of floats
            Locations of the monolayer cutting planes in fractional coordinates
            of a unit cell.
        atomsk_path : Pathlike
            Path to the atomsk binary
        fn_info : Pathlike
            Path to info file

        """
        for each in species:
            _ = self.add_molecule_type(each)

        species_names = [x.name for x in species]
        if self.form == 'ribbon':
            boundary = 'pcc'
        elif self.form == 'slab':
            boundary = 'ppc'
        elif self.form == 'finite':
            boundary = 'ccc'
         
        H0, xa_names, xa_coords = make_npl_xtal(lx, ly, lz, lattice, 
                                    species_names, a, c=c, orient=orient,
                                    unit=unit, mlcp=mlcp, boundary=boundary,
                                    atomsk_path=atomsk_path)

        #Reposition crystal and get simulation box upper & lower bounds
        if self.form == 'ribbon':
            cx = H0[0,0]/2
            cy = (xa_coords[:,1].min() + xa_coords[:,1].max())/2
            cz = (xa_coords[:,2].min() + xa_coords[:,2].max())/2
            center = np.array([cx, cy, cz])
            xa_coords[:,:] -= center
            lb = np.array([ -cx, xa_coords[:,1].min(), xa_coords[:,2].min() ])
            ub = -lb
        elif self.form == 'slab':
            if self.slab_pos == 'bot':
                dr = np.array([ H0[0,0]/2, H0[1,1]/2, xa_coords[:,2].min() ])
                xa_coords[:,:] -= dr
                lb = np.array([ -H0[0,0]/2, -H0[1,1]/2, 0 ])
                ub = np.array([ -lb[0], -lb[1], xa_coords[:,2].max() ])
            elif self.slab_pos == 'mid':
                cz = (xa_coords[:,2].min() + xa_coords[:,2].max())/2
                center = np.array([ H0[0,0]/2, H0[1,1]/2, cz ])
                xa_coords[:,:] -= center
                lb = np.array([ -H0[0,0]/2, -H0[1,1]/2, xa_coords[:,2].min() ])
                ub = -lb
        elif self.form == 'finite':
            center = (xa_coords.min(axis=0) + xa_coords.max(axis=0))/2
            xa_coords[:,:] -= center
            lb = xa_coords.min(axis=0)
            ub = -lb

        with open(fn_info, 'w') if fn_info \
            else nullcontext(sys.stdout) as fh:
            buf  = f"AreaX = {(ub[1]-lb[1])*(ub[2]-lb[2]):g} nm^2\n"
            buf += f"AreaY = {(ub[0]-lb[0])*(ub[2]-lb[2]):g} nm^2\n"
            buf += f"AreaZ = {(ub[0]-lb[0])*(ub[1]-lb[1]):g} nm^2\n"
            fh.write(buf)

        #Add simulation box
        self.add_simbox(lb[0], ub[0], lb[1], ub[1], lb[2], ub[2])

        #Crystal molecule ids (molecules consist of a single atom)
        mid_beg = self.num_molecules + 1 # First crystal molecule id
        mid_end = mid_beg + xa_coords.shape[0] - 1 #Last crystal molecule id

        #Molecule (monoatomic) ids of the top & bottom surfaces
        #(along the thickness)
        xa_zcoords = xa_coords[:,2]
        zlo = xa_zcoords.min(); zhi = xa_zcoords.max()

        top_mask = np.isclose(xa_zcoords, zhi)
        mids_xst = (mid_beg + np.nonzero(top_mask)[0]).tolist()

        bot_mask = np.isclose(xa_zcoords, zlo)
        mids_xsb = (mid_beg + np.nonzero(bot_mask)[0]).tolist()

        #Add all crystal molecules (molecules are monoatomic) & species groups
        for each in species:
            name = each.name
            mask = [y==name for y in xa_names]
            n = np.count_nonzero(mask)
            mids = (mid_beg + np.nonzero(mask)[0]).tolist()
            _ = self.add_molecules(name, n, 1, xa_coords[mask])
            self.set_group(name, molecule_ids=mids)

        #Make groups
        self.set_group('Xtal', molecule_ids=range(mid_beg, mid_end+1))
        self.set_group('XtalSurfTop', molecule_ids=mids_xst)
        self.set_group('XtalSurfBot', molecule_ids=mids_xsb)




    def add_ligand_one(self, ligand, r0, intermolecular_bond_func=None, 
                    intermolecular_bond_params=None):
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
        intermolecular_bond_func : int or None
            Integer designating the GROMACS bond function type for
            intermolecular bonds.
        intermolecular_bond_params : list
            Parameters corresponding to the intermolecular bond type.

        """
        if self.form != 'slab':
            raise SystemExit('A single ligand can be added only for a slab.')

        rng = np.random.default_rng()

        #Add ligand molecule types
        self.add_molecule_type(ligand)

        #Add ligand molecules
        mid_beg = self.num_molecules + 1
        aids_head = []
        #Top surface (common to both bottom & mid-positioned slabs)
        normal = np.array([0, 0, 1])
        pop = 1
        gps = np.array([(self.simbox[0,0]+self.simbox[0,1])/2,
                        (self.simbox[1,0]+self.simbox[1,1])/2, 
                         self.simbox[2,1]+r0 ])
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

        #Bottom surface for mid-positioned slab
        if self.slab_pos == 'mid':
            normal = np.array([0, 0, -1])
            pop = 1
            gps = np.array([(self.simbox[0,0]+self.simbox[0,1])/2,
                            (self.simbox[1,0]+self.simbox[1,1])/2, 
                             self.simbox[2,0]-r0 ])
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

        #Add intermolecular bonds
        if self.slab_pos == 'bot':
            aids_xtal = self.groups['XtalSurfTop']['atom_ids']
        elif self.slab_pos == 'mid':
            aids_xtal = self.groups['XtalSurfTop']['atom_ids'] + \
                        self.groups['XtalSurfBot']['atom_ids']

        if intermolecular_bond_func:
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
        #Single ligand type, reindexing not necessary
        #Update simulation box
        lo, hi = self.get_bbox(self.groups['Ligands']['atom_ids'])
        if self.slab_pos == 'bot':
            self.simbox[2,1] = hi[2]
        elif self.slab_pos == 'mid':
            self.simbox[2,1] = max(-lo[2], hi[2])
            self.simbox[2,0] = -self.simbox[2,1]



    def add_ligands(self, ligands, pop_ratio, r0, balance_charge, lattice, apl,
                    intermolecular_bond_func=None, 
                    intermolecular_bond_params=None, fn_info=None):
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
            Whether to determine the ligand population from neutralizing the
            total charge on the crystal.
        lattice : {'sc', 'bcc', None}
            Lattice for arranging the ligand molecule graft points. If None,
            ligands are distributed randomly.
        apl : float or None
            Area per ligand in nm^2. Ignored if `balance_charge` is True.
        intermolecular_bond_func : int or None
            Integer designating the GROMACS bond function type for
            intermolecular bonds.
        intermolecular_bond_params : list
            Parameters corresponding to the intermolecular bond type.
        fn_info : Pathlike
            Path to info file

        """
        rng = np.random.default_rng()

        #Add ligand molecule types
        for each in ligands:
            self.add_molecule_type(mt=each)

        #Ligand population
        if balance_charge:
            pop = self._ligpop_from_charge(ligands, pop_ratio, fn_info)
        else:
            pop = self._ligpop_from_area(ligands, pop_ratio, apl, fn_info)

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

        #Add intermolecular bonds
        if self.form == 'slab':
            if self.slab_pos == 'bot':
                aids_xtal = self.groups['XtalSurfTop']['atom_ids']
            else:
                aids_xtal = self.groups['XtalSurfTop']['atom_ids'] + \
                            self.groups['XtalSurfBot']['atom_ids']
        else:
            aids_xtal = self.groups['Xtal']['atom_ids']

        if intermolecular_bond_func:
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
        print('reindexing')
        self.reindex()
        #Update simulation box
        if self.form == 'ribbon':
            lo, hi = self.get_bbox(self.groups['Ligands']['atom_ids'])
            self.simbox[1:3,1] = np.maximum(-lo[1:3], hi[1:3])
            self.simbox[1:3,0] = -self.simbox[1:3,1]
        elif self.form == 'slab':
            lo, hi = self.get_bbox(self.groups['Ligands']['atom_ids'])
            if self.slab_pos == 'bot':
                self.simbox[2,1] = hi[2]
            else:
                self.simbox[2,1] = max(-lo[2], hi[2])
                self.simbox[2,0] = -self.simbox[2,1]
        elif self.form == 'finite':
            self.fit_simbox()
            self.simbox[:,1] = np.maximum(-self.simbox[:,0], self.simbox[:,1])
            self.simbox[:,0] = -self.simbox[:,1]
        #Make the system charge neutral
        chge = self.get_total_charge()
        if chge != 0:
            aids_charged = []
            for i in range(1, self.num_atoms+1):
                q = self.atoms[i].charge
                if not math.isclose(q, 0.0):
                    aids_charged.append(i)
            na_charged = len(aids_charged)
            cpa = chge/na_charged
            for i in aids_charged:
                self.atoms[i].charge -= cpa
            for mt in self.molecule_types.values():
                for atm in mt.atoms.values():
                    atm.charge -= cpa
        print("Total charge after adjustment = %g"%self.get_total_charge())


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


    def _ligpop_from_area(self, ligands, pop_ratio, apl, fn_info):
        """
        Determines population of each ligand species based on surface area.

        """
        area_tot = self._get_total_surface_area()
        num_ligands = area_tot/apl #This is a float
        s = sum(pop_ratio); pop_frac = [x/s for x in pop_ratio]
        pop = [round(num_ligands*f) for f in pop_frac]
        true_apl = area_tot/sum(pop)
        with open(fn_info, 'a') if fn_info else nullcontext(sys.stdout) as fh:
            buf = f"Area/ligand = {true_apl:f} nm^2\n"
            buf += f"Brush density = {1/true_apl:f}/nm^2\n"
            fh.write(buf)
        return pop


    def _ligpop_from_charge(self, ligands, pop_ratio, fn_info):
        """
        Determines population of each ligand species to balance the charge of
        the crystal.

        """
        area_tot = self._get_total_surface_area()
        totchg = self.get_total_charge()
        ligchg = [each.get_total_charge() for each in ligands]
        s = sum(pop_ratio); pop_frac = [x/s for x in pop_ratio]
        pop_tot = abs(totchg)/sum([abs(f*c) for f,c in zip(pop_frac,ligchg)])
        pop = [round(pop_tot*f) for f in pop_frac]

        residual_charge = totchg + sum([p*c for p,c in zip(pop,ligchg)])
        print (f"Residual charge after adding ligands = {residual_charge}")

        true_apl = area_tot/sum(pop)
        with open(fn_info, 'a') if fn_info else nullcontext(sys.stdout) as fh:
            buf = f"Area/ligand = {true_apl:f} nm^2\n"
            buf += f"Brush density = {1/true_apl:f}/nm^2\n"
            fh.write(buf)
        return pop


    def _get_total_surface_area(self):
        """
        Returns the total surface area of the crystal.

        """
        #Crystal bounding box 
        xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atom_ids'])

        #Total surface area
        if self.form == 'ribbon':
            dx = self.simbox[0,1] - self.simbox[0,0]
            dy = xtal_hi[1] - xtal_lo[0]
            dz = xtal_hi[2] - xtal_lo[2]
            area_tot = 2*(dx*dy + dx*dz)
        elif self.form == 'slab':
            area_tot = (self.simbox[0:2,1] - self.simbox[0:2,0]).prod()
            if self.slab_pos != 'bot':
                area_tot *= 2
        elif self.form == 'finite':
            dr = xtal_hi - xtal_lo
            area_tot = 2*(dr[0]*dr[1] + dr[1]*dr[2] + dr[2]*dr[0])
        return area_tot


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

        if self.form == 'ribbon':
            dx = xtal_hi[0] - xtal_lo[0] + 2*r0
            dy = xtal_hi[1] - xtal_lo[1] + 2*r0
            dz = xtal_hi[2] - xtal_lo[2] + 2*r0
            areas = [2*dx*dz, 2*dx*dy]
            area_tot = sum(areas)

            if areas[0] <= areas[1]:
                popy = [int(areas[0]*w/area_tot) for w in pop]
                popz = [w-v for v,w in zip(popy,pop)]
            else:
                popz = [int(areas[1]*w/area_tot) for w in pop]
                popy = [w-v for v,w in zip(popz,pop)]

            #Surface normal to the y-axis: Front
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_lo[1]-r0, xtal_hi[2]+r0]
            pop_front = [v//2 for v in popy]
            n = sum(pop_front)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['front'] = {'normal': np.array([0,-1,0]),
                                'pop': pop_front, 'graft_points': gp}

            #Surface normal to the y-axis: Back
            lo = [xtal_lo[0]-r0, xtal_hi[1]+r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_back = [w-v for w,v in zip(popy,pop_front)]
            n = sum(pop_back)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['back'] = {'normal': np.array([0,1,0]),
                                'pop': pop_back, 'graft_points': gp}

            #Surface normal to the z-axis: Bottom
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_lo[2]-r0]
            pop_bot = [v//2 for v in popz]
            n = sum(pop_bot)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['bot'] = {'normal': np.array([0,0,-1]),
                               'pop': pop_bot, 'graft_points': gp}

            #Surface normal to the z-axis: Top
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_hi[2]+r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_top = [w-v for w,v in zip(popz,pop_bot)]
            n = sum(pop_top)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['top'] = {'normal': np.array([0,0,1]),
                               'pop': pop_top, 'graft_points': gp}

        elif self.form == 'slab':
            if self.slab_pos == 'bot':
                lo = [self.simbox[0,0], self.simbox[1,0], xtal_hi[2]+r0]
                hi = [self.simbox[0,1], self.simbox[1,1], xtal_hi[2]+r0]
                gp = self._get_graft_points(sum(pop), lo, hi, lattice, 'ppp')
                surfaces['top'] = {'normal': np.array([0,0,1]), 'pop': pop,
                                    'graft_points': gp}
            elif self.slab_pos == 'mid':
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
        elif self.form == 'finite':
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



    def solvate_one(self, solvent, box_params, xtal_offset=0.0, packmol_tol=0.2,
                packmol_sidemax=100, packmol_path=None):
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
            Tolerance for Packmol. Default is 0.2 nm.
        packmol_sidemax : float
            Parameter for Packmol. Default is 100 nm.
        packmol_path : str or pathlib.Path or None
            Path to the packmol executable. If None, Packmol will not be used. In
            this case all added molecules will have their atom positions set to
            zero.

        """
        if self.form != 'slab':
            raise SystemExit(
                    'Solvating with a single ligand allowed only for a slab.')

        delta = 0.2 #Small gap of 0.2 nm between periodic images 
                    #(See Packmol manual)
        #Change simulation box size
        if 'boxx' in box_params:
            raise ValueError('Box size may not be changed along x.')
        if 'boxy' in box_params:
            raise ValueError('Box size may not be changed along y.')
        if 'boxz' in box_params:
            if self.slab_pos=='bot':
                self.simbox[2,1] = box_params['boxz']
            elif self.slab_pos == 'mid':
                self.simbox[2,1] = box_params['boxz']/2
                self.simbox[2,0] = -self.simbox[2,1]

        if 'Xtal' in self.groups:
            xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atom_ids'])
            if 'sepx' in box_params:
                raise ValueError('Box size may not be changed along x.')
            if 'sepy' in box_params:
                raise ValueError('Box size may not be changed along y.')
            if 'sepz' in box_params:
                if self.slab_pos=='bot':
                    self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                elif self.slab_pos=='mid':
                    self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                    self.simbox[2,0] = -self.simbox[2,1]
            if 'make_cubic' in box_params and box_params['make_cubic']:
                if not math.isclose(self.simbox[0,1], self.simbox[1,1]):
                    raise ValueError('Box size unequal along x and y.'+
                                     ' Cannot make a cubic box')
                else:
                    if self.slab_pos == 'bot':
                        self.simbox[2,1] = 2*self.simbox[0,1]
                    elif self.slab_pos=='mid':
                        self.simbox[2,1] = self.simbox[0,1]
                        self.simbox[2,0] = -self.simbox[2,1]

        #Add solvent molecule type
        self.add_molecule_type(mt=solvent[0])

        #Solvent volume = box volume - crystal volume (in nm^3)
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
            if self.slab_pos == 'bot':
                self.simbox[2,1] += xtal_offset
            elif self.slab_pos=='mid':
                self.simbox[2,0] -= xtal_offset
                self.simbox[2,1] += xtal_offset

        #Define bounding boxes & add molecules
        oa_bbox_lo = self.simbox[:,0] + delta #Overall bounding box
        oa_bbox_hi = self.simbox[:,1] - delta #Overall bounding box

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
        elif self.slab_pos=='mid':
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



    def solvate(self, solvent, density, ratio, box_params, offset=0.0, 
                offset_from='xtal', rbuf=0.0, packmol_tol=0.2, packmol_sidemax=100,
                packmol_path=None):
        """
        Adds solvent molecules around a nanoplatelet.

        Parameters
        ----------
        solvent : list
            List of solvent molecules.
        density : list
            Densities of the solvent components in g/mL.
        ratio : list of int
            The volumetric ratio between the pure solvent components. E.g., if
            the solvent contains two volumes of pure component A and one volume
            of pure component in 2:1 volume ratio, `ratio = [2, 1]`. For a
            single component, `ratio = [1]`.
        box_params : dict
            Parameters for changing the simulation box.
        offset : float
            Distance from the outer ligand boundary beyond which the solvent molecules
            will be placed, assuming the crystal to be undeformed.
        offset_from : {'xtal' | 'lig'}
            Distance from the crystal surface beyond which the solvent molecules
            will be placed, assuming the crystal to be undeformed. Ignored if
            `lig_offset` is non-zero.
        rbuf : float
            Add this distance in addition to ligand or crystal offset to help
            packing with Packmol (not included in solvent volume calculation)
        packmol_tol : float
            Tolerance for Packmol. Default is 0.2 nm.
        packmol_sidemax : float
            Parameter for Packmol. Default is 100 nm.
        packmol_path : str or pathlib.Path or None
            Path to the packmol executable. If None, Packmol will not be used. In
            this case all added molecules will have their atom positions set to
            zero.

        """
        delta = 0.2 #Small gap between periodic images (See Packmol manual)
        #Change simulation box size
        if 'boxx' in box_params:
            if self.form == 'finite':
                self.simbox[0,1] = box_params['boxx']/2
                self.simbox[0,0] = -self.simbox[0,1]
            else:
                if box_params['boxx'] != 0:
                    raise ValueError('Box size may not be changed along x.')
        if 'boxy' in box_params:
            if self.form == 'finite' or self.form == 'ribbon':
                self.simbox[1,1] = box_params['boxy']/2
                self.simbox[1,0] = -self.simbox[1,1]
            else:
                if box_params['boxy'] != 0:
                    raise ValueError('Box size may not be changed along y.')
        if 'boxz' in box_params:
            if self.form == 'slab' and self.slab_pos=='bot':
                self.simbox[2,1] = box_params['boxz']
            else:
                self.simbox[2,1] = box_params['boxz']/2
                self.simbox[2,0] = -self.simbox[2,1]

        xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atom_ids'])
        if 'sepx' in box_params:
            if self.form == 'finite':
                self.simbox[0,1] = xtal_hi[0] + box_params['sepx']
                self.simbox[0,0] = -self.simbox[0,1]
            else:
                if box_params['sepx'] != 0:
                    raise ValueError('Box size may not be changed along x.')
        if 'sepy' in box_params:
            if self.form == 'finite' or self.form == 'ribbon':
                self.simbox[1,1] = xtal_hi[1] + box_params['sepy']
                self.simbox[1,0] = -self.simbox[1,1]
            else:
                if box_params['sepy'] != 0:
                    raise ValueError('Box size may not be changed along y.')
        if 'sepz' in box_params:
            if self.form == 'slab' and self.slab_pos=='bot':
                self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
            else:
                self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                self.simbox[2,0] = -self.simbox[2,1]

        if 'make_cubic' in box_params and box_params['make_cubic']:
            if self.is_slab:
                if not math.isclose(self.simbox[0,1], self.simbox[1,1]):
                    raise ValueError('Box size unequal along x and y.'+
                                     ' Cannot make a cubic box')
                else:
                    if self.slab_pos == 'bot':
                        self.simbox[2,1] = 2*self.simbox[0,1]
                    else:
                        self.simbox[2,1] = self.simbox[0,1]
                        self.simbox[2,0] = -self.simbox[2,1]
            else:
                v = self.simbox[:,1].max()
                self.simbox[:,1] = v
                self.simbox[:,0] = -self.simbox[:,1]

        #Add solvent molecule types
        num_components = len(solvent)
        assert num_components == len(density)
        for each in solvent:
            self.add_molecule_type(mt=each)

        #Solvent volume = box volume - crystal volume (in nm^3)
        volume = np.prod(self.simbox[:,1]-self.simbox[:,0])
        xtal_vol = (xtal_hi-xtal_lo).prod()
        volume -= xtal_vol

        if 'Ligands' in self.groups.keys():
            nonsolvent_atoms = self.groups['Ligands']['atom_ids']
            nonsolvent_mass = 0.0
            for each in nonsolvent_atoms:
                nonsolvent_mass += self.atoms[each].mass

        #Total population of solvent molecules 
        s = sum(ratio); vol_frac = [x/s for x in ratio]
        vol_comps = [round(volume*f) for f in vol_frac] #Volume of each component
        pop_comps = [0]*num_components
        for i in range(num_components):
            molwt = solvent[i].get_total_mass() # in g/mol
            dens = density[i] #Density in g/mL
            pop_comps[i] = (602.3*vol_comps[i]*dens \
                            - nonsolvent_mass*vol_frac[i])/molwt
            pop_comps[i] = math.floor(pop_comps[i])

        #Split population for each subvolume
        if self.form == 'ribbon':
            voly = 2*(xtal_hi[0]-xtal_lo[0])*(xtal_hi[2]-xtal_lo[2]) \
                    *(self.simbox[1,1]-xtal_hi[1])
            popy = [round(x*voly/volume)for x in pop_comps]
            popz = [x-y for x,y in zip(pop_comps, popy)]

            pop_comps_front = [x//2 for x in popy]
            pop_comps_back = [x-y for x,y in zip(popy, pop_comps_front)]

            pop_comps_top = [x//2 for x in popz]
            pop_comps_bot = [x-y for x,y in zip(popz, pop_comps_top)]
        elif self.form == 'slab':
            if self.slab_pos == 'bot':
                pass #Single subvolume
            elif self.slab_pos == 'mid':
                pop_comps_top = [x//2 for x in pop_comps]
                pop_comps_bot = [x-y for x,y in zip(pop_comps, pop_comps_top)]
        elif self.form == 'finite':
            pass #Single subvolume

        aid_beg = self.num_atoms + 1 #First solvent atom id
        #Last solvent atom id
        aid_end = aid_beg + sum([x*y.num_atoms 
                                 for x,y in zip(pop_comps, solvent)]) - 1
        mid_beg = self.num_molecules + 1 #First solvent molecule id
        mid_end = mid_beg + sum(pop_comps) - 1 #Last solvent molecule id

        #Pack solvent molecules
        lxtal_lo, lxtal_hi = self.get_bbox(self.groups['Ligands']['atom_ids'])
        if offset_from == 'xtal':
            ibxmin = xtal_lo[0] - offset; ibxmax = xtal_hi[0] + offset
            ibymin = xtal_lo[1] - offset; ibymax = xtal_hi[1] + offset
            ibzmin = xtal_lo[2] - offset; ibzmax = xtal_hi[2] + offset
        elif offset_from == 'lig':
            ibxmin = lxtal_lo[0] - offset; ibxmax = lxtal_hi[0] + offset
            ibymin = lxtal_lo[1] - offset; ibymax = lxtal_hi[1] + offset
            ibzmin = lxtal_lo[2] - offset; ibzmax = lxtal_hi[2] + offset

        if self.form == 'ribbon':
            #Enlarge simulation box to include offset
            self.simbox[1:3,0] -= (offset+rbuf)
            self.simbox[1:3,1] += (offset+rbuf)

            bbox_front_lo = [self.simbox[0,0], self.simbox[1,0], ibzmin]
            bbox_front_hi = [self.simbox[0,1], ibymin, ibzmax]
            bbox_back_lo = [self.simbox[0,0], ibymax, ibzmin]
            bbox_back_hi = [self.simbox[0,1], self.simbox[1,1], ibzmax]
            bbox_bottom_lo = self.simbox[:,0].tolist()
            bbox_bottom_hi = [self.simbox[0,1], self.simbox[1,1], ibzmin]
            bbox_top_lo = [self.simbox[0,0], self.simbox[1,0], ibzmax]
            bbox_top_hi = self.simbox[:,1].tolist()

            #Adjust box bounds by adding a small gap of delta
            for i in range(3):
                bbox_front_lo[i] += delta;  bbox_front_hi[i] -= delta
                bbox_back_lo[i] += delta;   bbox_back_hi[i] -= delta
                bbox_bottom_lo[i] += delta; bbox_bottom_hi[i] -= delta
                bbox_top_lo[i] += delta;    bbox_top_hi[i] -= delta

            #Add molecules and pack
            mols_to_pack = []
            for i in range(num_components): 
                print(f"Packing solvents \n"
                      f" inside box ({' '.join(['%g'%v for v in bbox_front_lo])})"
                      f" ({' '.join(['%g'%v for v in bbox_front_hi])}) \n"
                      f" inside box ({' '.join(['%g'%v for v in bbox_back_lo])})"
                      f" ({' '.join(['%g'%v for v in bbox_back_hi])}) \n"
                      f" inside box ({' '.join(['%g'%v for v in bbox_bottom_lo])})"
                      f" ({' '.join(['%g'%v for v in bbox_bottom_hi])}) \n"
                      f" inside box ({' '.join(['%g'%v for v in bbox_top_lo])})"
                      f" ({' '.join(['%g'%v for v in bbox_top_hi])}) "
                      )
                if pop_comps_front[i] > 0:
                    g = 'inside box ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_front_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_front_hi])
                    #Add solvent molecules
                    mids = self.add_molecules(solvent[i].name, pop_comps_front[i],
                                              1, np.zeros((3,)))
                    elem = {'name': solvent[i].name, 'mids': mids, 'constraints': [g]}
                    mols_to_pack.append(elem)

                if pop_comps_back[i] > 0:
                    g = 'inside box ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_back_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_back_hi])
                    #Add solvent molecules
                    mids = self.add_molecules(solvent[i].name, pop_comps_back[i],
                                              1, np.zeros((3,)))
                    elem = {'name': solvent[i].name, 'mids': mids, 'constraints': [g]}
                    mols_to_pack.append(elem)

                if pop_comps_bot[i] > 0:
                    g = 'inside box ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_bottom_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_bottom_hi])
                    #Add solvent molecules
                    mids = self.add_molecules(solvent[i].name, pop_comps_bot[i],
                                              1, np.zeros((3,)))
                    elem = {'name': solvent[i].name, 'mids': mids, 'constraints': [g]}
                    mols_to_pack.append(elem)

                if pop_comps_top[i] > 0:
                    g = 'inside box ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_top_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_top_hi])
                    #Add solvent molecules
                    mids = self.add_molecules(solvent[i].name, pop_comps_top[i],
                                              1, np.zeros((3,)))
                    elem = {'name': solvent[i].name, 'mids': mids, 'constraints': [g]}
                    mols_to_pack.append(elem)

        elif self.form == 'slab':
            if self.slab_pos == 'bot':
                self.simbox[2,1] += (offset+rbuf)
                #A single bounding box for solvent molecules
                bbox_lo = [self.simbox[0,0]+delta, self.simbox[1,0]+delta,
                           ibzmax+delta]
                bbox_hi = (self.simbox[:,1]-delta).tolist()
                #Add molecules and pack
                mols_to_pack = []
                for i in range(num_components): 
                    print(f"Packing solvents \n"
                          f" inside box ({' '.join(['%g'%v for v in bbox_lo])})"
                          f" ({' '.join(['%g'%v for v in bbox_hi])})"
                          )
                    if pop_comps[i] > 0:
                        g = 'inside box ' \
                            + ' '.join(['%g'%(10*v) for v in bbox_lo]) + ' ' \
                            + ' '.join(['%g'%(10*v) for v in bbox_hi])
                        #Add solvent molecules
                        mids = self.add_molecules(solvent[i].name, pop_comps[i],
                                                  1, np.zeros((3,)))
                        elem = {'name': solvent[i].name, 'mids': mids, 'constraints': [g]}
                        mols_to_pack.append(elem)
            else:
                self.simbox[2,0] -= (offset+rbuf)
                self.simbox[2,1] += (offset+rbuf)

                bbox_bottom_lo = (self.simbox[:,0]+delta).tolist()
                bbox_bottom_hi = [self.simbox[0,1]-delta, self.simbox[1,1]-delta,
                                  ibzmin-delta]
                bbox_top_lo = [self.simbox[0,0]+delta, self.simbox[1,0]+delta,
                               ibzmax+delta]
                bbox_top_hi = (self.simbox[:,1]-delta).tolist()
                #Add molecules and pack
                mols_to_pack = []
                for i in range(num_components): 
                    print(f"Packing solvents \n"
                          f" inside box ({' '.join(['%g'%v for v in bbox_bottom_lo])})"
                          f" ({' '.join(['%g'%v for v in bbox_bottom_hi])}) \n"
                          f" inside box ({' '.join(['%g'%v for v in bbox_top_lo])})"
                          f" ({' '.join(['%g'%v for v in bbox_top_hi])}) "
                          )
                    if pop_comps_bot[i] > 0:
                        g = 'inside box ' \
                            + ' '.join(['%g'%(10*v) for v in bbox_bottom_lo]) + ' ' \
                            + ' '.join(['%g'%(10*v) for v in bbox_bottom_hi])
                        #Add solvent molecules
                        mids = self.add_molecules(solvent[i].name, pop_comps_bot[i],
                                                  1, np.zeros((3,)))
                        elem = {'name': solvent[i].name, 'mids': mids, 'constraints': [g]}
                        mols_to_pack.append(elem)

                    if pop_comps_top[i] > 0:
                        g = 'inside box ' \
                            + ' '.join(['%g'%(10*v) for v in bbox_top_lo]) + ' ' \
                            + ' '.join(['%g'%(10*v) for v in bbox_top_hi])
                        #Add solvent molecules
                        mids = self.add_molecules(solvent[i].name, pop_comps_top[i],
                                                  1, np.zeros((3,)))
                        elem = {'name': solvent[i].name, 'mids': mids, 'constraints': [g]}
                        mols_to_pack.append(elem)
        elif self.form == 'finite':
            self.simbox[:,0] -= (offset+rbuf)
            self.simbox[:,1] += (offset+rbuf)

            bbox_out_lo = (self.simbox[:,0]+delta).tolist()
            bbox_out_hi = (self.simbox[:,1]-delta).tolist()
            bbox_in_lo = [ibxmin-delta, ibymin-delta, ibzmin-delta]
            bbox_in_hi = [ibxmax+delta, ibymax+delta, ibzmax+delta]
            #Add molecules and pack
            mols_to_pack = []
            for i in range(num_components): 
                print(f"Packing solvents \n"
                      f" inside box ({' '.join(['%g'%v for v in bbox_out_lo])})"
                      f" ({' '.join(['%g'%v for v in bbox_out_hi])}) \n"
                      f" outside box ({' '.join(['%g'%v for v in bbox_in_lo])})"
                      f" ({' '.join(['%g'%v for v in bbox_in_hi])})"
                      )
                if pop_comps[i] > 0:
                    g_in = 'inside box ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_out_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_out_hi])
                    g_out = 'outside box ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_in_lo]) + ' ' \
                        + ' '.join(['%g'%(10*v) for v in bbox_in_hi])
                    #Add solvent molecules
                    mids = self.add_molecules(solvent[i].name, pop_comps[i],
                                              1, np.zeros((3,)))
                    elem = {'name': solvent[i].name, 'mids': mids,
                            'constraints': [g_in, g_out]}
                    mols_to_pack.append(elem)

        self.pack_molecules(mols_to_pack, packmol_tol, packmol_sidemax,
                      packmol_path)

        #Create new group of solvent atoms
        self.set_group('Solvent', atom_ids=range(aid_beg, aid_end+1), 
                            molecule_ids=range(mid_beg, mid_end+1))
        if num_components > 1:
            mbeg = mid_beg
            for i,each in enumerate(solvent):
                name = each.name
                mids = range(mbeg, mbeg+pop_comps[i])
                self.set_group(f"Solvent.{name}", molecule_ids=mids)
                mbeg += pop_comps[i]
