#!/usr/bin/env python

import copy
import math
import warnings
import numpy as np
from numpy.typing import NDArray
from dataclasses import field, dataclass
from _geom_utils import rotate_vectors_random, get_ransphere, get_frc_link

@dataclass
class AtomRec(object):
    typ: str|int = 0
    molid: int = 0
    name: str = ''
    mass: float = 0.0
    charge: float = 0.0
    cgnr: int = 0
    residnum: int = 0
    residue: str = ''
    coords: NDArray|None = None
    imgflags: NDArray|None = None


@dataclass
class BondRec(object):
    ai: int|str = 0
    aj: int|str = 0
    typ: int = 0
    func: int = 0
    params: list|None = None


@dataclass
class AngleRec(object):
    ai: int|str = 0
    aj: int|str = 0
    ak: int|str = 0
    typ: int = 0
    func: int = 0
    params: list|None = None


@dataclass
class DihedralRec(object):
    ai: int|str = 0
    aj: int|str = 0
    ak: int|str = 0
    al: int|str = 0
    typ: int = 0
    func: int = 0
    params: list|None = None


class Topology(object):
    def __init__(self):
        self.atoms = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}


    def clear(self):
        '''
        Clears all data.

        '''
        self.atoms.clear()
        self.bonds.clear()
        self.angles.clear()
        self.dihedrals.clear()
        self.impropers.clear()


    @property
    def num_atoms(self):
        return len(self.atoms)

    @property
    def num_bonds(self):
        return len(self.bonds)

    @property
    def num_angles(self):
        return len(self.angles)

    @property
    def num_dihedrals(self):
        return len(self.dihedrals)

    @property
    def num_impropers(self):
        return len(self.impropers)


    def add_atom(self, typ=0, molid=0, name='', mass=0.0, charge=0.0, cgnr=0,
                residnum=0, residue='', coords=[0, 0, 0], imgflags=[0, 0, 0],
                atm_id=None):
        """
        Adds an atom.

        """
        if atm_id is None:
            atm_id = len(self.atoms) + 1
        self.atoms[atm_id] = AtomRec(typ, molid, name, mass, charge, cgnr,
                                        residnum, residue, 
                                        np.array(coords[0:3],dtype=np.float64),
                                        np.array(imgflags[0:3], dtype=np.int32))
        return atm_id


    def append_atom_unbonded(self, box, sep=None, typ=0, molid=0, name='', 
                            mass=0.0, charge=0.0, cgnr=0, residnum=0,
                            residue='', rng=None):
        """
        Adds an atom at a position chosen randomly within a box, possibly
        checking for overlap.

        """
        rng_ = rng if rng else np.random.default_rng()
        atm_id = len(self.atoms) + 1
        itr = 0; maxitr = 100
        if not sep:
            ri = rng_.random((3,))
            ri = box[:,0] + ri*(box[:,1]-box[:,0])
        else:
            while itr < maxitr:
                #Choose any random position within the box. 
                ri = rng_.random((3,))
                ri = box[:,0] + ri*(box[:,1]-box[:,0])
                if not self.has_overlap(ri.reshape((1,3)), 'all', None, sep):
                    break
                itr += 1
            else:
                warnings.warn('Maximum iteration count reached.')
        self.atoms[atm_id] = AtomRec(typ, molid, name, mass, charge, cgnr,
                                    residnum, residue, ri)
        return atm_id


    def append_atom_bonded(self, len_bond, method, im1, im2=None, theta=None,
                        uhat=None, sep=None, typ=0, name='', mass=0.0,
                        charge=0.0, rng=None):
        """
        Adds an atom bonded to another existing atom, possibly checking for
        overlap.

        """
        atm_id = len(self.atoms) + 1

        itr = 0; maxitr = 100 #For method = 'efjc'|'efrc'

        if method == 'alignx':
            rim1 = self.atoms[im1].coords
            ri = rim1 + np.array([len_bond, 0.0, 0.0])
        elif method == 'aligny':
            rim1 = self.atoms[im1].coords
            ri = rim1 + np.array([0.0, len_bond, 0.0])
        elif method == 'alignz':
            rim1 = self.atoms[im1].coords
            ri = rim1 + np.array([0.0, 0.0, len_bond])
        elif method == 'align':
            rim1 = self.atoms[im1].coords
            ri = rim1 + len_bond*uhat
        elif method == 'fjc':
            rim1 = self.atoms[im1].coords
            ri = rim1 + get_ransphere(len_bond, rng)
        elif method == 'frc':
            rim2 = self.atoms[im2].coords
            rim1 = self.atoms[im1].coords
            ri = rim1 + len_bond*get_frc_link(rim1-rim2, theta, rng)
        elif method == 'efjc':
            rim1 = self.atoms[im1].coords
            while itr < maxitr:
                ri = rim1 + get_ransphere(len_bond, rng)
                if not self.has_overlap(ri.reshape((1,3)), 'all', None, sep):
                    break
                itr += 1
            else:
                warnings.warn('Maximum iteration count reached.')
        elif method == 'efrc':
            rim2 = self.atoms[im2].coords
            rim1 = self.atoms[im1].coords
            while itr < maxitr:
                ri = rim1 + len_bond*get_frc_link(rim1-rim2, theta, rng)
                if not self.has_overlap(ri.reshape((1,3)), 'all', None, sep):
                    break
                itr += 1
            else:
                warnings.warn('Maximum iteration count reached.')
        else:
            raise ValueError('Unknown method "%s"'%method)

        molid = self.atoms[im1].molid
        cgnr = self.atoms[im1].cgnr
        residnum = self.atoms[im1].residnum
        residue = self.atoms[im1].residue
        self.atoms[atm_id] = AtomRec(typ, molid, name, mass, charge, cgnr, 
                                     residnum, residue, ri)
        return atm_id

    
    def set_atom_coords(self, atm_ids, coords):
        for i,aid in enumerate(atm_ids):
            self.atoms[aid].coords[:] = coords[i,:]


    def get_atom_coords(self, atm_ids=None, out=None):
        if atm_ids is None:
            n = self.num_atoms; aids = range(1, n+1)
        else:
            n = len(atm_ids); aids = atm_ids
        if out is None:
            coords = np.zeros((n,3))
        else:
            nr, nc = out.shape
            assert nr >= n
            assert nc >= 3
            coords = out 
        for i in range(n):
            aid = aids[i]
            coords[i,:] = self.atoms[aid].coords
        if out is None:
            return np.squeeze(coords)


    def add_bond(self, ai, aj, typ=0, func=0, params=None, bnd_id=None):
        if typ > 0:
            assert typ >= 1 and typ <= self.num_bond_types
        assert ai in self.atoms
        assert aj in self.atoms
        if bnd_id is None:
            bnd_id = len(self.bonds) + 1
        self.bonds[bnd_id] = BondRec(ai, aj, typ, func, copy.deepcopy(params))
        return bnd_id


    def add_angle(self, ai, aj, ak, typ=0, func=0, params=None, ang_id=None):
        if typ > 0:
            assert typ >= 1 and typ <= self.num_angle_types
        assert ai in self.atoms
        assert aj in self.atoms
        assert ak in self.atoms
        if ang_id is None:
            ang_id = len(self.angles) + 1
        self.angles[ang_id] = AngleRec(ai, aj, ak, typ, func, 
                                       copy.deepcopy(params))
        return ang_id


    def add_dihedral(self, ai, aj, ak, al, typ=0, func=0, params=None, 
                     dhd_id=None):
        if typ > 0:
            assert typ >= 1 and typ <= self.num_dihedral_types
        assert ai in self.atoms
        assert aj in self.atoms
        assert ak in self.atoms
        assert al in self.atoms
        if dhd_id is None:
            dhd_id = len(self.dihedrals) + 1
        self.dihedrals[dhd_id] = DihedralRec(ai, aj, ak, al, typ, func, 
                                             copy.deepcopy(params))
        return dhd_id


    def add_improper(self, ai, aj, ak, al, typ=0, func=0, params=None, 
                     imp_id=None):
        if typ > 0:
            assert typ >= 1 and typ <= self.num_improper_types
        assert ai in self.atoms
        assert aj in self.atoms
        assert ak in self.atoms
        assert al in self.atoms
        if imp_id is None:
            imp_id = len(self.impropers) + 1
        self.impropers[imp_id] = DihedralRec(ai, aj, ak, al, typ, func, 
                                             copy.deepcopy(params))
        return imp_id

    
    def has_overlap(self, pos, key, val, sep):
        '''
        Checks atomwise overlap of a set of atoms `new` with an existing set 
        of atoms `old` within a separation distance of `sep`.

        pos : (n,3) ndarray of atom positions
        key : 'atm_id'/'typ'/'molid'/'name'/'all'
        val : sequence of arguments

        '''
        out = False
        sepsq = sep*sep
        
        #Atom ids of the set of existing atoms against which overlap is to be
        #tested
        if key == 'atm_id':
            set_ext = val
        elif key == 'typ':
            set_ext = [k for k,v in self.atoms.items() if v.typ in val]
        elif key == 'name':
            set_ext = [k for k,v in self.atoms.items() if v.name in val]
        elif key == 'all':
            set_ext = np.arange(1, self.num_atoms+1, 1, dtype=np.int32)
        else:
            raise ValueError(f"Unknown key {key}")

        for rj in pos:
            for i in set_ext:
                ri = self.atoms[i].coords
                rij = rj - ri
                rijmsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                if rijmsq < sepsq:
                    out = True
                    return out
        return out


    def write_xyz(self, fn, title='', file_unit='nm'):
        """
        Writes atom coordinates to an XYZ file.  
    
        Parameters
        ---------
        file_unit : 'nm' | 'angstrom'

        """
        if file_unit == 'nm':
            factor = 1
        elif file_unit == 'angstrom': 
            factor = 10
        else:
            raise ValueError()

        na = self.num_atoms
        with open(fn,'w') as fh:
            fh.write(str(na) + '\n')
            fh.write(title + '\n')
    
            for i in range(1, na+1):
                atm_nam = self.atoms[i].name
                coords = factor*self.atoms[i].coords
                fh.write( '%s  '%atm_nam 
                    + '  '.join(['% .15g'%x for x in coords]) + '\n')
    
    
    def read_xyz(self, fn, offset=0, aids=None, file_unit='nm'):
        """
        Reads atom positions from an XYZ file.
    
        """
        if file_unit == 'nm':
            factor = 1
        elif file_unit == 'angstrom': 
            factor = 0.1
        else:
            raise ValueError()        

        with open(fn, 'r') as fh:
            lines = fh.readlines()
    
        na = int(lines[0].strip('\n'))
        assert na <= self.num_atoms
        if aids:
            aids_ = aids
        else:
            aids_ = [offset+i for i in range(1, na+1)]
        for i in range(2, 2+na):
            words = lines[i].strip('\n').split()
            coords = np.array([float(x) for x in words[1:4]])
            iatm = aids_[i-2]
            self.atoms[iatm].coords[:] = factor*coords
