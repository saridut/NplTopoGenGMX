#!/usr/bin/env python

import os
import math
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from _geom_utils import fix_axis_angle, rotate_vector_axis_angle
from _configuration import Configuration

@dataclass
class PairRec(object):
    ai: int|str = 0
    aj: int|str = 0
    func: int = 0
    params: list|None = None

@dataclass
class PosresRec(object):
    ai: int|str = 0
    func: int = 0
    g: int = 0
    r: float = 0.0
    k: float = 0.0
    kx: float = 0.0
    ky: float = 0.0
    kz: float = 0.0


class Molecule(Configuration):
    def __init__(self, fns, orient=False) :
        """
        Parameters
        ----------
        name : str
            Name of the solvent molecule.
        fns : tuple of str or pathlib.Path
            Path of the file containing itp and gro files.

        """
        super().__init__()
        self.name = ''
        self.nexcl = 0
        self.pairs = {}
        self.exclusions = {}
        self.settles = {}
        self.constraints = {}
        self.position_restraints = {}
        self.distance_restraints = {}
        self.dihedral_restraints = {}
        self.orientation_restraints = {}
        self.angle_restraints = {}
        self.angle_restraints_z = {}
 
        self.from_file(fns[0], fns[1])
        #Find an OBB for the atoms
        na = self.num_atoms
        if orient and na > 1:
            positions = np.zeros((na,3))
            for i in range(1, na+1):
                positions[i-1,:] = self.atoms[i].coords
            com = positions.mean(axis=0)
            positions -= np.tile(com, (na,1))
            cov = np.matmul(positions.T, positions)/na
            eigvals, eigvecs = np.linalg.eigh(cov, UPLO='L')
            positions = positions @ eigvecs
            for i in range(1, na+1):
                self.atoms[i].coords[:] = positions[i-1,:]
        #Place the molecule in the positive quadrant, with its bottom, left, back
        #corner to the origin.
        self.fit_simbox(sep=0.0)
        r = self.simbox[:,0]
        self.translate(-r)

    
    def clear(self):
        self.name = ''
        self.nexcl = 0
        self.pairs.clear()
        self.exclusions.clear()
        self.settles.clear()
        self.constraints.clear()
        self.position_restraints.clear()
        self.distance_restraints.clear()
        self.dihedral_restraints.clear()
        self.orientation_restraints.clear()
        self.angle_restraints.clear()
        self.angle_restraints_z.clear()
        super().clear()


    @property
    def num_pairs(self):
        return len(self.pairs)

    @property
    def num_exclusions(self):
        return len(self.exclusions)

    @property
    def num_settles(self):
        return len(self.settles)

    @property
    def num_constraints(self):
        return len(self.constraints)

    @property
    def num_position_restraints(self):
        return len(self.position_restraints)

    @property
    def num_distance_restraints(self):
        return len(self.distance_restraints)

    @property
    def num_dihedral_restraints(self):
        return len(self.dihedral_restraints)

    @property
    def num_orientation_restraints(self):
        return len(self.orientation_restraints)

    @property
    def num_angle_restraints(self):
        return len(self.angle_restraints)

    @property
    def num_angle_restraints_z(self):
        return len(self.angle_restraints_z)


    def add_pair(self, ai, aj, func=0, params=None, pair_id=None):
        assert ai in self.atoms
        assert aj in self.atoms
        if pair_id is None:
            pair_id = self.num_pairs + 1
        self.pairs[pair_id] = PairRec(ai, aj, func, params)
        return pair_id


    def add_position_restraint(self, ai, func=0, g=0, r=0.0, k=0.0,
                               kx=0.0, ky=0.0, kz=0.0, posrec_id=None):
        assert ai in self.atoms
        if func == 1:
            if (kx <= 0.0):
                raise ValueError(f"kx (={kx}) must be > 0.0.")
            if (ky <= 0.0):
                raise ValueError(f"ky (={ky}) must be > 0.0.")
            if (kz <= 0.0):
                raise ValueError(f"kz (={kz}) must be > 0.0.")
        elif func == 2:
            if g not in range(1, 9):
                raise ValueError(f"g (={g}) must be one of: {list(range(1,9))}")
            if (k <= 0.0):
                raise ValueError(f"k (={k}) must be > 0.0.")
        else:
            raise ValueError(f"func (={func}) must be 1 or 2.")
        if posrec_id is None:
            posrec_id = self.num_position_restraints + 1
        self.position_restraints[posrec_id] = \
                PosresRec(ai, func, g, r, k, kx, ky, kz)
        return posrec_id


    def get_heavy_atoms(self):
        """
        Return atom ids of all heavy atoms.

        """
        hmass = 1.008
        aids_heavy = set()
        for aid, rec in self.atoms.items():
            if not math.isclose(rec.mass, hmass):
                aids_heavy.add(aid)
        return aids_heavy


    def reindex(self):
        """
        Renumber atoms such that hydrogen atoms are placed immediately after the
        heavy atom they are bonded to.

        """
        hmass = 1.008
        aids_heavy = set(); aids_hydrogen = set()
        for aid, rec in self.atoms.items():
            if math.isclose(rec.mass, hmass):
                aids_hydrogen.add(aid)
            else:
                aids_heavy.add(aid)
        assert aids_heavy.isdisjoint(aids_hydrogen)
        #Sort atoms in each category
        aids_heavy = sorted(list(aids_heavy))
        aids_hydrogen = sorted(list(aids_hydrogen))
        #Assign hydrogens to each heavy atom based on bond list
        map_aids_heavy2hydrogen = {x:set() for x in aids_heavy}
        for brec in self.bonds.values():
            ai = brec.ai; aj = brec.aj
            if ai in aids_heavy and aj in aids_hydrogen:
                map_aids_heavy2hydrogen[ai].add(aj)
            elif ai in aids_hydrogen and aj in aids_heavy:
                map_aids_heavy2hydrogen[aj].add(ai)
        ###Convert sets to sorted lists
        for k, aids in map_aids_heavy2hydrogen.items():
            map_aids_heavy2hydrogen[k] = sorted(list(aids))

        #Assign new atom ids
        map_aids_new2old = {}
        map_aids_old2new = {}
        atoms = {}
        for k, aids in map_aids_heavy2hydrogen.items():
            key = len(map_aids_new2old) + 1
            map_aids_new2old[key] = k
            for i,each in enumerate(aids):
                map_aids_new2old[key+i+1] = each
        for k,v in map_aids_new2old.items():
            map_aids_old2new[v] = k #Invert to get old2new map
        #print('map_aids_old2new')
        #for k in sorted(list(map_aids_old2new.keys())):
        #    print(f"    {k}: {map_aids_old2new[k]}")
        #print('map_aids_new2old')
        #for k in sorted(list(map_aids_new2old.keys())):
        #    print(f"    {k}: {map_aids_new2old[k]}")

        for k,v in map_aids_new2old.items():
            atoms[k] = deepcopy(self.atoms[v])
        #Update atoms list
        self.atoms.update(atoms)
        #Update atom ids in bonds, angles, etc
        for brec in self.bonds.values():
            brec.ai = map_aids_old2new[brec.ai]
            brec.aj = map_aids_old2new[brec.aj]
        for arec in self.angles.values():
            arec.ai = map_aids_old2new[arec.ai]
            arec.aj = map_aids_old2new[arec.aj]
            arec.ak = map_aids_old2new[arec.ak]
        for drec in self.dihedrals.values():
            drec.ai = map_aids_old2new[drec.ai]
            drec.aj = map_aids_old2new[drec.aj]
            drec.ak = map_aids_old2new[drec.ak]
            drec.al = map_aids_old2new[drec.al]
        for prec in self.pairs.values():
            prec.ai = map_aids_old2new[prec.ai]
            prec.aj = map_aids_old2new[prec.aj]
        for prec in self.position_restraints.values():
            prec.ai = map_aids_old2new[prec.ai]

        #Return atom id maps
        return (map_aids_old2new, map_aids_new2old)


    def from_file(self, fn_itp, fn_gro):
        with open(fn_itp, 'r') as fh:
            lns = [y for s in fh 
                   if len(y := s.strip(" \n").partition(";")[0]) > 0]
        for line in lns:
            wrds = line.split()
            if (wrds[0]=='[') and (wrds[2]==']') and (len(wrds) == 3):
                this_section = wrds[1]
                continue
            if this_section == 'moleculetype':
                self.name = wrds[0]
                self.nexcl = int(wrds[1])

            elif this_section == 'atoms':
                typ = wrds[1]
                residnum = int(wrds[2])
                residue = wrds[3]
                name = wrds[4]
                cgnr = int(wrds[5])
                charge = float(wrds[6])
                mass = float(wrds[7])
                _ = self.add_atom(typ=typ, name=name, mass=mass, charge=charge,
                              cgnr=cgnr, residnum=residnum, residue=residue)

            elif this_section == 'bonds':
                ai = int(wrds[0]); aj = int(wrds[1])
                func = int(wrds[2])
                params = [float(x) for x in wrds[3:]] if len(wrds) > 3 else None
                _ = self.add_bond(ai, aj, func=func, params=params)

            elif this_section == 'angles':
                ai = int(wrds[0]); aj = int(wrds[1]); ak = int(wrds[2])
                func = int(wrds[3])
                params = [float(x) for x in wrds[4:]] \
                            if len(wrds) > 4 else None
                _ = self.add_angle(ai, aj, ak, func=func, params=params)

            elif this_section == 'dihedrals':
                ai = int(wrds[0]); aj = int(wrds[1])
                ak = int(wrds[2]); al = int(wrds[3])
                func = int(wrds[4])
                params = [float(x) for x in wrds[5:]] \
                            if len(wrds) > 5 else None
                _ = self.add_dihedral(ai, aj, ak, al, func=func, params=params)

            elif this_section == 'pairs':
                ai = int(wrds[0]); aj = int(wrds[1])
                func = int(wrds[2])
                params = [float(x) for x in wrds[3:]] if len(wrds) > 3 else None
                _ = self.add_pair(ai, aj, func=func, params=params)

            elif this_section == 'position_restraints':
                ai = int(wrds[0])
                func = int(wrds[1])
                if func == 1:
                    kx = float(wrds[2]); ky = float(wrds[3]); kz = float(wrds[4]) 
                    _ = self.add_position_restraint(ai, func, kx=kx, ky=ky, kz=kz)
                elif func == 2:
                    g = int(wrds[2]); r = float(wrds[3]); k = float(wrds[4]) 
                    _ = self.add_position_restraint(ai, func, g=g, r=r, k=k)
                else:
                    raise ValueError(f"func (={func}) must be 1 or 2.")
            else:
                raise KeyError(f"Unknown directive {this_section}.")

        #Read in atom positions
        with open(fn_gro, 'r') as fh:
            _ = fh.readline() #Title line
            n = int(fh.readline()) #Number of atoms
            assert self.num_atoms == n
            atm_ids = []; coords = np.empty((n,3))
            for i in range(n):
                line = fh.readline().rstrip('\n')
                atm_ids.append( i+1 )
                coords[i,0] = float(line[20:28])
                coords[i,1] = float(line[28:36])
                coords[i,2] = float(line[36:44])
            self.set_atom_coords(atm_ids, coords)


    def to_file(self, fn_itp, fn_gro='', title=''):
        with open(fn_itp, 'w') as fh:
            qtot = 0.0
            fh.write('[ moleculetype ]\n')
            fh.write('; Name  nrexcl\n')
            fh.write(f"{self.name} {self.nexcl}\n")

            fh.write('[ atoms ]\n')
            fh.write('; nr type resnr residue atom cgnr charge mass\n')
            for k in range(1, self.num_atoms+1):
                val = self.atoms[k]
                qtot += val.charge
                buf = '%6d %8s %8d %8s %8s %4d  % .15f  % .4f'%(
                        k, val.typ, val.residnum, val.residue, val.name,
                        val.cgnr, val.charge, val.mass)
                fh.write(buf + f"; qtot {qtot}" + '\n')

            if self.num_bonds > 0:
                fh.write('[ bonds ]\n')
                for k in range(1, self.num_bonds+1):
                    val = self.bonds[k]
                    buf = '%6d %6d %6d  '%(val.ai, val.aj, val.func)
                    if val.params:
                        buf += '  '.join(['%.4f'%x for x in val.params])
                    fh.write(buf + '\n')

            if self.num_angles > 0:
                fh.write('[ angles ]\n')
                for k in range(1, self.num_angles+1):
                    val = self.angles[k]
                    buf = ''.join(['%6d  '%x for x in 
                                   [val.ai, val.aj, val.ak, val.func]])
                    if val.params:
                        buf += '  '.join(['%.4f'%x for x in val.params])
                    fh.write(buf + '\n')

            if self.num_dihedrals > 0:
                fh.write('[ dihedrals ]\n')
                for k in range(1, self.num_dihedrals+1):
                    val = self.dihedrals[k]
                    buf = ''.join(['%6d  '%x for x in 
                                   [val.ai, val.aj, val.ak, val.al, val.func]])
                    if val.params:
                        buf += '  '.join(['% .4f'%x for x in val.params])
                    fh.write(buf + '\n')

            if self.num_pairs > 0:
                fh.write('[ pairs ]\n')
                for k in range(1, self.num_pairs+1):
                    val = self.pairs[k]
                    buf = '%6d %6d %6d  '%(val.ai, val.aj, val.func)
                    if val.params:
                        buf += '  '.join(['% .4f'%x for x in val.params])
                    fh.write(buf + '\n')

            if self.num_position_restraints > 0:
                fh.write('#ifdef POSRES\n')
                fh.write('[ position_restraints ]\n')
                for k in range(1, self.num_position_restraints+1):
                    val = self.position_restraints[k]
                    buf = '%6d %6d  '%(val.ai, val.func)
                    if val.func == 1:
                        buf += '% .4f% .4f% .4f'%(val.kx, val.ky, val.kz)
                    elif val.func == 2:
                        buf += '%6d % .4f% .4f'%(val.g, val.r, val.k)
                    else:
                        raise ValueError(f"self.position_restraints[{k}] "
                                        f"(={val.func}) must be 1 or 2.")
                    fh.write(buf + '\n')
                fh.write('#endif\n')

        #Write atom positions
        if fn_gro:
            with open(fn_gro, 'w') as fh:
                fh.write(title + '\n') #Title line
                fh.write(f"{self.num_atoms}" + '\n') #Number of atoms
                for i in range(1, self.num_atoms+1):
                    v = self.atoms[i]
                    #Renumbering atoms from 0 after 99,999. To renumber from 1
                    #use 1+(i-1)%99999.
                    buf = '%5d%5s%5s%5d%8.3f%8.3f%8.3f\n'%(
                            v.residnum, v.residue, v.name, i%100000,
                            v.coords[0], v.coords[1], v.coords[2])
                    fh.write(buf) 
                boxv = self.simbox[:,1] - self.simbox[:,0]
                fh.write(f"{boxv[0]} {boxv[1]} {boxv[2]}\n")




class LigandMolecule(Molecule):
    """
    Class implementing a ligand molecule.

    """
    def __init__(self, fns, head, tail, bind_group, orient=True) :
        """
        Parameters
        ----------
        name : str
            Name of the ligand molecule.
        fns : tuple of str or pathlib.Path
            Path to itp and gro files.
        head : int
            Atom id of the ligand head. The id is local, i.e. with respect to
            the ligand molecule. Ligand head is the first atom after the binding
            group in the ligand main chain.
        tail : int
            Atom id of the ligand tail. The id is local, i.e. with respect to
            the ligand molecule. Ligand tail is the last atom in the ligand main
            chain.
        bind_group : sequence of ints
            Atom ids of the binding group of the ligand. The ids are local, i.e.
            with respect to the ligand molecule.

        """
        super().__init__(fns, orient)
        self.head = head
        self.tail = tail
        self.bind_group = deepcopy(bind_group)


    def reindex(self):
        """
        Renumber atoms such that hydrogen atoms are placed immediately after the
        heavy atom they are bonded to.

        """
        map_aids_old2new, map_aids_new2old = super().reindex()
        self.head = map_aids_old2new[self.head]
        self.tail = map_aids_old2new[self.tail]
        for k in range(len(self.bind_group)):
            self.bind_group[k] = map_aids_old2new[self.bind_group[k]]

        return (map_aids_old2new, map_aids_new2old)
