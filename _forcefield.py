#!/usr/bin/env python

import warnings
import math
import copy
import numpy as np

class ForceField(object):
    def __init__(self):
        self.atom_style = None
        self.pair_style = None
        self.bond_style = None
        self.angle_style = None
        self.dihedral_style = None
        self.improper_style = None

        self.pair_mixing_rule = None
        self.pair14_gen = None
        self.pair_weights = None

        self.num_atom_types = 0
        self.num_bond_types = 0
        self.num_angle_types = 0
        self.num_dihedral_types = 0
        self.num_improper_types = 0
        
        self.atom_types = {}
        self.pair14_types = {}
        self.pairij_types = {}
        self.bond_types = {}
        self.angle_types = {}
        self.dihedral_types = {}
        self.improper_types = {}


    def clear(self):
        '''
        Clears all data.

        '''
        self.num_atom_types = 0
        self.num_bond_types = 0
        self.num_angle_types = 0
        self.num_dihedral_types = 0
        self.num_improper_types = 0
        
        self.atom_types.clear()
        self.bond_types.clear()
        self.angle_types.clear()
        self.dihedral_types.clear()
        self.improper_types.clear()


    def add_atom_type(self, ptype, name, mass, charge, epsilon, sigma, 
                      bonded_atom_type=None, atomic_number=None):
        '''
        Adds a new atom type.

        Parameters
        ----------
        ptype : {'A', 'D', 'V', 'S'}
            Particle type. 'A': atomic, 'D': virtual site, 'V': virtual site,
            'S': shell.
        name : str
            Name of the atom type. 
        mass : float
            Mass of the atom (a.m.u). 
        charge : float
            Charge of the atom (elementary charge).
        bonded_atom_type : str, optional
            Name of the mapped bonded atom type. Must be <= 5 characters, longer
            strings will be truncated to five characters. Must contain at least
            one non-digit character.
        atomic_number : int, optional
            Atomic number of the atom.

        '''
        self.num_atom_types += 1
        iat = self.num_atom_types
        self.atom_types[name] = \
            {'id': iat, 'ptype': ptype, 'mass': mass, 'charge': charge,
             'epsilon': epsilon, 'sigma': sigma}
        if len(bonded_atom_type) > 5:
            print(f"`bonded_atom_type` = {bonded_atom_type} is longer than "
                    "5 characters. Truncating to the first 5 characters")
            assert not bonded_atom_type[:5].isdigit()
            self.atom_types[name]['bonded_atom_type'] = bonded_atom_type[:5]
        else:
            assert not bonded_atom_type.isdigit()
            self.atom_types[name]['bonded_atom_type'] = bonded_atom_type 
        self.atom_types[name]['atomic_number'] = atomic_number 


    def add_bond_type(self, at_i, at_j, params):
        '''
        Adds a new bond type.

        Parameters
        ----------
        at_i : str
            Atom type of the first atom (must exist in `self.atomtypes`).
        at_j : str
            Atom type of the second atom (must exist in `self.atomtypes`).
        params : sequence
            Parameters for this bond type.

        '''
        bat = [x['bonded_atom_type'] for x in self.atom_types.keys()]
        if not all(bat):
            bat = self.atom_types.keys()
        assert at_i in bat
        assert at_j in bat
        self.num_bond_types += 1
        ibt = self.num_bond_types
        self.bond_types[ibt] = {'at_i': at_i, 'at_j': at_j,
                                'params': list(params)}


    def add_angle_type(self, at_i, at_j, at_k, params):
        '''
        Adds a new angle type.

        Parameters
        ----------
        at_i : str
            Atom type of the first atom (must exist in `self.atomtypes`).
        at_j : str
            Atom type of the second atom (must exist in `self.atomtypes`).
        at_k : str
            Atom type of the third atom (must exist in `self.atomtypes`).
        params : sequence
            Parameters for this angle type.

        '''
        bat = [x['bonded_atom_type'] for x in self.atom_types.keys()]
        if not all(bat):
            bat = self.atom_types.keys()
        assert at_i in bat
        assert at_j in bat
        assert at_k in bat
        self.num_angle_types += 1
        iat = self.num_angle_types
        self.angle_types[iat] = {'at_i': at_i, 'at_j': at_j, 'at_k': at_k,
                                'params': list(params)}


    def add_dihedral_type(self, at_i, at_j, at_k, at_l, params):
        '''
        Adds a new dihedral type.

        Parameters
        ----------
        at_i : str
            Atom type of the first atom (must exist in `self.atomtypes`).
        at_j : str
            Atom type of the second atom (must exist in `self.atomtypes`).
        at_k : str
            Atom type of the third atom (must exist in `self.atomtypes`).
        at_l : str
            Atom type of the fourth atom (must exist in `self.atomtypes`).
        params : sequence
            Parameters for this dihedral type.

        '''
        bat = [x['bonded_atom_type'] for x in self.atom_types.keys()]
        if not all(bat):
            bat = self.atom_types.keys()
        assert at_i in bat
        assert at_j in bat
        assert at_k in bat
        assert at_l in bat
        self.num_dihedral_types += 1
        idt = self.num_dihedral_types
        self.dihedral_types[idt] = {'at_i': at_i, 'at_j': at_j, 'at_k': at_k,
                                    'at_l': at_l, 'params': list(params)}


    def add_improper_type(self, at_i, at_j, at_k, at_l, params):
        '''
        Adds a new improper type.

        Parameters
        ----------
        at_i : str
            Atom type of the first atom (must exist in `self.atomtypes`).
        at_j : str
            Atom type of the second atom (must exist in `self.atomtypes`).
        at_k : str
            Atom type of the third atom (must exist in `self.atomtypes`).
        at_l : str
            Atom type of the fourth atom (must exist in `self.atomtypes`).
        params : sequence
            Parameters for this dihedral type.

        '''
        bat = [x['bonded_atom_type'] for x in self.atom_types.keys()]
        if not all(bat):
            bat = self.atom_types.keys()
        assert at_i in bat
        assert at_j in bat
        assert at_k in bat
        assert at_l in bat
        self.num_improper_types += 1
        iit = self.num_improper_types
        self.improper_types[iit] = {'at_i': at_i, 'at_j': at_j, 'at_k': at_k,
                                    'at_l': at_l, 'params': list(params)}


    def remove_duplicate_types(self):
        self.remove_duplicate_atom_types()
        for x in ['bond', 'angle', 'dihedral', 'improper']:
            self.remove_duplicate_x_types(x)


    def remove_duplicate_atom_types(self):
        assert len(self.atom_names) == self.num_atom_types
        assert len(self.atom_mass) == self.num_atom_types
        assert len(self.pair_coeffs) == self.num_atom_types
        types = np.zeros((self.num_atom_types,2), dtype=np.int32)
        types[:,0] = np.arange(1,self.num_atom_types+1)
        for typ_i in range(1, self.num_atom_types):
            if types[typ_i-1,0] < typ_i:
                continue
            mass_i = self.atom_mass[typ_i]
            coeff_i = self.pair_coeffs[typ_i]
            for typ_j in range(typ_i+1, self.num_atom_types+1):
                if types[typ_j-1,0] <= typ_i:
                    continue
                if mass_i == self.atom_mass[typ_j] and \
                        coeffs_equal(coeff_i, self.pair_coeffs[typ_j]):
                    types[typ_j-1,0] = typ_i
        if np.any(types[:,0] != np.arange(1,self.num_atom_types+1)):
            uniqtyps = np.unique(types[:,0])
            for i in range(self.num_atom_types):
                m = np.abs(uniqtyps-types[i,0]).argmin()
                types[i,1] = m + 1
            for val in self.atoms.values():
                m = val['type']; val['type'] = types[m-1,1]
        
            atm_mass_tmp = copy.deepcopy(self.atom_mass)
            atm_nam_tmp = copy.deepcopy(self.atom_names)
            pair_coeff_tmp = copy.deepcopy(self.pair_coeffs)
            self.atom_mass = {}; self.atom_names = {}; self.pair_coeffs = {}
            for i,v in enumerate(uniqtyps):
                self.atom_mass[i+1] = atm_mass_tmp.pop(v)
                self.atom_names[i+1] = atm_nam_tmp.pop(v)
                self.pair_coeffs[i+1] = pair_coeff_tmp.pop(v)

        self.num_atom_types = len(self.atom_mass)
        assert len(self.atom_names) == self.num_atom_types
        assert len(self.pair_coeffs) == self.num_atom_types


    def remove_duplicate_x_types(self, x):
        if x == 'bond':
            num_x_types = self.num_bond_types
            x_coeffs = self.bond_coeffs
            x_topo = self.bonds
        elif x == 'angle':
            num_x_types = self.num_angle_types
            x_coeffs = self.angle_coeffs
            x_topo = self.angles
        elif x == 'dihedral':
            num_x_types = self.num_dihedral_types
            x_coeffs = self.dihedral_coeffs
            x_topo = self.dihedrals
        elif x == 'improper':
            num_x_types = self.num_improper_types
            x_coeffs = self.improper_coeffs
            x_topo = self.impropers
        else:
            raise ValueError('Unknown input: %s'%x)

        if num_x_types == 0:
            return #No type information, nothing to do 

        assert len(x_coeffs) == num_x_types
        types = np.zeros((num_x_types,2), dtype=np.int32)
        types[:,0] = np.arange(1,num_x_types+1)
        for typ_i in range(1, num_x_types):
            if types[typ_i-1,0] < typ_i:
                continue
            coeff_i = x_coeffs[typ_i]
            for typ_j in range(typ_i+1, num_x_types+1):
                if types[typ_j-1,0] <= typ_i:
                    continue
                if coeffs_equal(coeff_i, x_coeffs[typ_j]):
                    types[typ_j-1,0] = typ_i
        if np.any(types[:,0] != np.arange(1,num_x_types+1)):
            uniqtyps = np.unique(types[:,0])
            for i in range(num_x_types):
                m = np.abs(uniqtyps-types[i,0]).argmin()
                types[i,1] = m + 1
            if len(x_topo) > 0:
                for val in x_topo.values():
                    m = val['type']; val['type'] = types[m-1,1]
        
            x_coeff_tmp = copy.deepcopy(x_coeffs)
            x_coeffs.clear()
            for i,v in enumerate(uniqtyps):
                x_coeffs[i+1] = x_coeff_tmp.pop(v)

        num_x_types = len(x_coeffs)

        if x == 'bond':
            self.num_bond_types = num_x_types
        elif x == 'angle':
            self.num_angle_types = num_x_types
        elif x == 'dihedral':
            self.num_dihedral_types = num_x_types
        elif x == 'improper':
            self.num_improper_types = num_x_types

    @staticmethod
    def coeffs_equal(coeffa, coeffb):
        if len(coeffa) != len(coeffb):
            return False
        out = True
        for a,b in zip(coeffa,coeffb):
            if a != b:
                try:
                    out = math.isclose(float(a), float(b), rel_tol=1e-6)
                except ValueError:
                    out = False
            if not out:
                break
        return out


