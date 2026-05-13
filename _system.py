#!/usr/bin/env python

import os
import warnings
import math
import copy
import pprint
import subprocess
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from _topology import BondRec, AngleRec
from _configuration import Configuration
from _molecule import Molecule

@dataclass
class MolRec(object):
    name: str = ''
    beg: int = 0
    end: int = 0


class System(Configuration):
    def __init__(self, name, fns_ff):
        super().__init__()
        self.name = name
        self.fns_ff = copy.deepcopy(fns_ff)
        self.molecule_types = {}
        self.molecule_pop = {}
        self.molecules = {}
        self.groups = {}
        self.intermolecular = {'bonds':{}, 'angles':{}}


    def clear(self):
        '''
        Clears all data.

        '''
        self.name = ''
        self.fns_ff = []
        self.num_molecule_types = 0
        self.molecule_types.clear()
        self.molecule_pop.clear()
        self.molecules.clear()
        self.groups.clear()
        for v in self.intermolecular.values():
            v.clear()
        super().clear()

    @property
    def num_molecule_types(self):
        return len(self.molecule_types)

    @property
    def num_molecules(self):
        return len(self.molecules)

    @property
    def num_groups(self):
        return len(self.groups)

    @property
    def num_intermolecular_bonds(self):
        return len(self.intermolecular['bonds'])

    @property
    def num_intermolecular_angles(self):
        return len(self.intermolecular['angles'])


    def add_molecule_type(self, mt=None, fns=None, orient=False):
        '''
        Adds a new molecule type.  Returns the name of the molecule type added.

        '''
        if mt:
            m = mt
        else:
            m = Molecule(fns, orient)
        name = m.name
        self.molecule_types[name] = copy.deepcopy(m)
        self.molecule_pop[name] = 0
        return name


    def add_molecules(self, name, num, anchor_atoms, anchor_coords):
        '''
        Adds a molecule named `name`.

        Parameters
        ----------
        num : int
            Number of molecules to add
        anchor_atoms : int | sequence of ints
            Anchoring atom of a molecule
        anchor_coords : ndarray-like
            Coordinates of the anchor atom for each moleule added

        Returns
        -------
        mids : int | sequence of ints
            List of molecule ids.

        '''
        if isinstance(anchor_atoms, int):
            anchors_ = [anchor_atoms]*num
        else:
            anchors_ = anchor_atoms

        if len(anchor_coords) == 3:
            anchor_coords_ = np.tile(np.array(anchor_coords), [num,1])
        else:
            anchor_coords_ = anchor_coords

        na_in_mol = self.molecule_types[name].num_atoms
        atoms_in_mol = self.molecule_types[name].atoms
        mids = []
        for k in range(num):
            dr = anchor_coords_[k,:] - atoms_in_mol[anchors_[k]].coords
            mid = self.num_molecules + 1
            self.molecules[mid] = MolRec(name, self.num_atoms+1,
                                        self.num_atoms+na_in_mol)
            mids.append(mid)
            for each in atoms_in_mol.values():
                coords = each.coords + dr
                self.add_atom(typ=each.typ, molid=mid, name=each.name, 
                              mass=each.mass, charge=each.charge,
                              cgnr=each.cgnr, residnum=mid, residue=each.residue,
                              coords=coords)
            self.molecule_pop[name] += 1
        return mids


    def add_intermolecular_bond(self, ai, aj, typ=0, func=0, params=None, 
                                bnd_id=None):
        if typ > 0:
            assert typ >= 1 and typ <= self.num_bond_types
        assert ai in self.atoms
        assert aj in self.atoms
        if bnd_id is None:
            bnd_id = self.num_intermolecular_bonds + 1
        self.intermolecular['bonds'][bnd_id] = \
                BondRec(ai, aj, typ, func, copy.deepcopy(params))
        return bnd_id

    
    def pack_molecules(self, molecules, packmol_tol=0.2, packmol_sidemax=100,
                  packmol_path=''):
        """
        Packs, i.e. changes the atom positions, of specified molecule_types.

        molecules : list
            List of molecules. 

            Each element of `molecules` is a dict with keys 'name', 'mids',
            and 'constraints'.

            `name` : name of a molecule_type.

            `mids` : list of molecule ids.
           
            `constaints` is a list of strings specifying the constrains as required
            by packmol,
            e.g., `constraints = ['inside cube xmin ymin zmin d',
            'outside sphere a b c d', ...]`.
            An empty constaint list indicates inside the entire simulation box.

            An example of `molecules` may be 
            [{'name': acetone, 'mids': [1,2, 4, ...], 
            'constraints': ['inside cube x y z d', 
                'outside box xmin ymin zmin xmax ymax zmax', ... ]},
             {'name': glycol, 'mids': [10, 12, ...],
             'constraints': ['outside cube x y z d',
                'inside sphere a b c d', ...]},
            ...]
        packmol_tol : float
            Tolerance for Packmol. Default is 0.2 nm.
        packmol_sidemax : float
            Parameter for Packmol. Default is 100 nm.
        packmol_path : str or pathlib.Path or None
            Path to the packmol executable. If None, Packmol will not be used. In
            this case all added molecules will have their atom positions set to
            zero.

        """

        #Update atom positions with packmol
        #Write out packmol file
        assert os.path.isfile(packmol_path)

        fn_pm_in = Path('inp_pm.txt') #Packmol input file
        fn_pm_out = Path('out_pm.xyz')
        fns_xyz = []
        aids = []
        with open(fn_pm_in, 'w') as fh:
            fh.write(f"tolerance {10*packmol_tol:g}\n") #nm to angstrom
            fh.write(f"sidemax {10*packmol_sidemax:g}\n") #nm to angstrom
            fh.write("seed -1\n")
            fh.write("randominitialpoint\n")
            fh.write("movebadrandom yes\n")
            fh.write(f"output {fn_pm_out}\n")
            fh.write("filetype xyz\n")
            fh.write('\n')
            for each in molecules:
                if len(each['mids']) < 1:
                    continue
                name = each['name']
                fn_xyz = Path(f"_tmp_{name}.xyz")
                fns_xyz.append(fn_xyz)
                self.molecule_types[name].write_xyz(fn_xyz, title=name,
                                                    file_unit='angstrom')
                fh.write(f"structure {fn_xyz}\n")
                fh.write(f"  number {len(each['mids'])}\n")
                for constraint in each['constraints']:
                    fh.write(f"  {constraint}\n")
                fh.write('end structure\n')
                for mid in each['mids']:
                    mrec = self.molecules[mid]
                    aids += list(range(mrec.beg, mrec.end+1))
        
        #Run packmol
        args_run = [f"{packmol_path} < {fn_pm_in} > /dev/null 2>&1"]
        subprocess.run(args_run, shell=True)
        #Read back the packmol output
        self.read_xyz(fn_pm_out, offset=0, aids=aids, file_unit='angstrom')

        fn_pm_in.unlink(missing_ok=True)
        fn_pm_out.unlink(missing_ok=True)
        Path(str(fn_pm_out)+'_FORCED').unlink(missing_ok=True)
        for each in fns_xyz:
            each.unlink(missing_ok=True)


    def write_molecules(self, fn, title=''):
        with open(fn, 'w') as fh:
            fh.write('%s\n'%title)
            fh.write('MOLECULES %d\n'%self.num_molecules)
            for key,val in self.molecules.items():
                fh.write(f"{key} {val.name} {val.beg} {val.end}\n")

    
    def reindex(self):
        """
        Reindexes atoms and molecules such that molecules of the same type
        remain together.

        """
        #New molecules dict
        molecules = {} 
        #New atoms dict
        atoms = {} 
        #New atoms list
        aids_new = []; aids_new2mids = []
        mids_new = []
        molecule_names = [m for m in self.molecule_types.keys()]
        tmp = {name: [] for name in molecule_names}
        for k, m in self.molecules.items():
            tmp[m.name].append(k)
        #Sort by molecule ids
        for name in tmp.keys():
            tmp[name].sort()
            for mid in tmp[name]:
                beg = self.molecules[mid].beg
                end = self.molecules[mid].end
                aids_new += list(range(beg, end+1))
                aids_new2mids += [mid]*(end + 1 - beg)
                mids_new.append(mid)

        #Update new molecules dict
        imol = 1
        for name in tmp.keys():
            for mid in tmp[name]:
                beg = self.molecules[mid].beg
                end = self.molecules[mid].end
                beg_new = aids_new.index(beg) + 1
                end_new = aids_new.index(end) + 1
                molecules[imol] = MolRec(name, beg_new, end_new)
                imol += 1
        self.molecules = molecules
        #Update new atoms dict
        for i, aid in enumerate(aids_new):
            atoms[i+1] = copy.deepcopy(self.atoms[aid])
            atoms[i+1].molid = aids_new2mids[i]
        self.atoms = atoms
        #Update groups
        for gn, v in self.groups.items():
            if v['atom_ids']:
                for i, x in enumerate(v['atom_ids']):
                    v['atom_ids'][i] = aids_new.index(x) + 1
            if v['molecule_ids']:
                for i, x in enumerate(v['molecule_ids']):
                    v['molecule_ids'][i] = mids_new.index(x) + 1
        #Update intermolecular bonds
        for brec in self.intermolecular['bonds'].values():
            brec.ai = aids_new.index(brec.ai) + 1
            brec.aj = aids_new.index(brec.aj) + 1
            

    def set_group(self, name, atom_names=None, atom_ids=None,
                  molecule_names=None, molecule_ids=None):
        """
        Parameters
        ----------
        atom_names : list
        atom_ids : range | list
        molecule_names : list
        molecule_ids : range | list

        Returns
        -------
        None

        """
        if name in self.groups.keys():
            s = input("Group name must be unique. Enter a different name: ")
            name_ = str(s)
        else:
            name_ = str(name)
        self.groups[name_] = {'atom_names': [], 'atom_ids': [],
                              'molecule_names': [], 'molecule_ids': []}
        if atom_ids:
            self.groups[name_]['atom_ids'] = list(atom_ids)

        if atom_names:
            self.groups[name_]['atom_names'] = list(atom_names)
            if not atom_ids:
                for aid, v in self.atoms.items():
                    if v.name in atom_names:
                        self.groups[name_]['atom_ids'].append(aid)

        if molecule_names:
            self.groups[name_]['molecule_names'] = list(molecule_names)
            if not atom_ids:
                for mid, v in self.molecules.items():
                    if v.name in molecule_names:
                        beg = v.beg; end = v.end
                        self.groups[name_]['atom_ids'] += list(range(beg,end+1))

        if molecule_ids:
            self.groups[name_]['molecule_ids'] = list(molecule_ids)
            if not atom_ids:
                for mid in molecule_ids:
                    beg = self.molecules[mid].beg
                    end = self.molecules[mid].end
                    self.groups[name_]['atom_ids'] += list(range(beg,end+1))



    def write_group(self, group_names, fn):
        """
        Write atom groups to a file.

        Parameters
        ----------
        group_names : list | `None`
            List of group names. If `None`, all groups will be written.
        fn : str | pathlib.path
            Output file name

        Returns
        -------
        None

        """
        if not group_names:
            gns = self.groups.keys()
        else:
            gns = group_names
        with open(fn, 'w') as fh:
            for k, name in enumerate(gns):
                if k == 0:
                    fh.write(f"[ {name} ]\n")
                else:
                    fh.write(f"\n[ {name} ]\n")
                gcontent = self.groups[name]
                if 'atom_ids' in gcontent.keys():
                    lbuf = gcontent['atom_ids']
                elif 'atoms_names' in gcontent.keys():
                    ans = gcontent['atom_names']
                    lbuf = [k for k,v in self.atoms.items() if v.name in ans]
                elif 'molecule_ids' in gcontent.keys():
                    mids = gcontent['molecule_ids']
                    lbuf = []
                    for mid in mids:
                        v = self.molecules[mid]
                        lbuf += list(range(v.beg, v.end+1))
                elif 'molecule_names' in gcontent.keys():
                    mns = gcontent['molecule_names']
                    lbuf = []
                    for v in self.molecules.values():
                        if v.name in mns:
                            lbuf += list(range(v.beg, v.end+1))
                buf = ' '.join([str(x) for x in lbuf])
                fbuf = pprint.pformat(buf, width=80, compact=True)
                fh.write(fbuf[1:-1].replace("'", ""))
            fh.write('\n')


    def to_file(self, dn):
        dn_ = os.path.expanduser(dn)

        #Write molecule files
        for k, v in self.molecule_types.items():
            fn_itp = os.path.join(dn_, f"{k}.itp")
            v.to_file(fn_itp)

        #Write structure (.gro) file
        fn_gro = os.path.join(dn_, f"{self.name}.gro")
        with open(fn_gro, 'w') as fh:
            fh.write(f"{self.name}\n") #Title line
            fh.write(f"{self.num_atoms}" + '\n') #Number of atoms
            for i in range(1, self.num_atoms+1):
                v = self.atoms[i]
                #Renumbering atoms from 0 after 99,999. To renumber from 1 use 
                #1+(i-1)%99999.
                buf = '%5d%5s%5s%5d%8.3f%8.3f%8.3f\n'%(
                        v.residnum, v.residue, v.name, i%100000,
                        v.coords[0], v.coords[1], v.coords[2])
                fh.write(buf)
            boxv = self.simbox[:,1] - self.simbox[:,0]
            fh.write(f"{boxv[0]} {boxv[1]} {boxv[2]}\n")

        #Write topology (.top) file
        fn_top = os.path.join(dn_, f"{self.name}.top")
        with open(fn_top, 'w') as fh:
            for each in self.fns_ff:
                fh.write(f"#include \"{each}\"\n")
            fh.write('\n')
            for k in self.molecule_types.keys():
                fn_mitp = os.path.join(dn_, f"{k}.itp")
                fh.write(f"#include \"{k}.itp\"\n")
            fh.write(f"\n[ system ]\n")
            fh.write(f"{self.name}\n")
            fh.write(f"\n[ molecules ]\n")
            for k in self.molecule_types.keys():
                fh.write(f"{k}  {self.molecule_pop[k]}\n")
            if self.num_intermolecular_bonds > 0:
                fh.write(f"\n[ intermolecular_interactions ]\n")
                fh.write(f"[ bonds ]\n")
                for k in range(1, self.num_intermolecular_bonds+1):
                    val = self.intermolecular['bonds'][k]
                    buf = '%6d %6d %6d  '%(val.ai, val.aj, val.func)
                    if val.params:
                        buf += '  '.join(['% .4f'%x for x in val.params])
                    fh.write(buf + '\n') 
