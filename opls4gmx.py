#!/usr/bin/env python

import sys
import os
import shutil
import math
import numpy as np
from _molecule import Molecule
from _topology import BondRec, AngleRec, DihedralRec

def fourier_to_rb(v1, v2, v3, v4):
    c = [0.0]*6
    c[0] = v2 + 0.5*(v1 + v3)
    c[1] = 0.5*(-v1 + 3*v3)
    c[2] = -v2 + 4*v4
    if v3 != 0:
        c[3] = -2*v3
    if v4 != 0:
        c[4] = -4*v4
    #Convertion factor from kcal/mol to kJ/mol
    cf = 4.1858
    return [cf*x for x in c]


#Parameter files from BOSS
bossdir = os.path.expanduser('~/soft/boss')
fn_vd = os.path.join(bossdir, 'oplsaa.par') #Nonbonded & dihedrals
fn_ba = os.path.join(bossdir, 'oplsaa.sb') #Bonds & angles

#Dicts storing BOSS parameters
bond_types_boss = {}
angle_types_boss = {}
dihedral_types_boss = {}

#Read in dihedral parameters
with open(fn_vd, 'r') as fh:
    _ = fh.readline() #Ignore title line
    this_section = 'vdw_params'
    while (line := fh.readline()) != '':
        if line.strip().startswith('#'):
            continue #Skip comments
        if line.startswith('Type'):
            this_section = 'dihedral_params'
            continue
        if this_section == 'dihedral_params':
            words = line.split(maxsplit=5)
            if len(words) < 6:
                continue #Skip unassigned types
            typ = int(words[0])
            v1 = float(words[1]); v2 = float(words[2])
            v3 = float(words[3]); v4 = float(words[4])
            #Bonded atom types
            ai = words[5][0:2].strip()
            aj = words[5][3:5].strip()
            ak = words[5][6:8].strip()
            al = words[5][9:11].strip()
            rb_params = fourier_to_rb(v1, v2, v3, v4)
            ityp = len(dihedral_types_boss) + 1
            dihedral_types_boss[ityp] = DihedralRec(ai, aj, ak, al, typ,
                                                    func=3, params=rb_params)

#Read in bond & angle parameters
with open(fn_ba, 'r') as fh:
    this_section = 'bond_params'
    while (line := fh.readline()) != '':
        if line.strip().startswith('*'):
            continue #Skip comments
        if line.strip().startswith('#'):
            continue #Skip comments
        if line == '\n':
            this_section = 'angle_params'
            continue
        if this_section == 'bond_params':
            #Bonded atom types
            ai = line[0:2].strip()
            aj = line[3:5].strip()
            words = line[5:].split(maxsplit=2)
            #kcal/mol.A^2 to kJ/mol.nm^2. Gmx has a factor of 1/2 for kb.
            kb = 2*418.58*float(words[0])
            req = 0.1*float(words[1])  # A to nm
            ityp = len(bond_types_boss) + 1
            bond_types_boss[ityp] = BondRec(ai, aj, typ=0, func=1,
                                            params=[req, kb])
        if this_section == 'angle_params':
            #Angle atom types
            ai = line[0:2].strip()
            aj = line[3:5].strip()
            ak = line[6:8].strip()
            words = line[8:].split(maxsplit=2)
            #kcal/mol.deg^2 to kJ/mol.deg^2. Gmx has a factor of 1/2 for ktheta.
            ktheta = 2*4.1858*float(words[0]) 
                                        
            thetaeq = float(words[1])
            ityp = len(angle_types_boss) + 1
            angle_types_boss[ityp] = AngleRec(ai, aj, ak, typ=0, func=1,
                                            params=[thetaeq, ktheta])


molecules = [ Molecule(['molecules/C8T.itp', 'molecules/C8T.gro']),
              Molecule(['molecules/C12T.itp', 'molecules/C12T.gro']),
              Molecule(['molecules/C16T.itp', 'molecules/C16T.gro']),
              Molecule(['molecules/C18T.itp', 'molecules/C18T.gro']),
              Molecule(['molecules/C18UT.itp', 'molecules/C18UT.gro']),
              Molecule(['molecules/C20T.itp', 'molecules/C20T.gro']),
              Molecule(['molecules/HEXN.itp', 'molecules/HEXN.gro']),
              Molecule(['molecules/TOL.itp', 'molecules/TOL.gro']),
              Molecule(['molecules/THF.itp', 'molecules/THF.gro'])
             ]

bond_types = {}
angle_types = {}
dihedral_types = {}

for echmol in molecules:
    #Check bonds
    for bond in echmol.bonds.values():
        ai = echmol.atoms[bond.ai].name
        aj = echmol.atoms[bond.aj].name
        is_stored = False; is_listed = False
        for sbnd in bond_types.values():
            if ((ai == sbnd.ai and aj == sbnd.aj) or 
                (ai == sbnd.aj and aj == sbnd.ai)):
                is_stored = True
                break
        if not is_stored:
            for lbnd in bond_types_boss.values():
                if ((ai == lbnd.ai and aj == lbnd.aj) or 
                    (ai == lbnd.aj and aj == lbnd.ai)):
                    ityp = len(bond_types) + 1
                    bond_types[ityp] = BondRec(ai, aj, lbnd.typ, lbnd.func,
                                               params=lbnd.params)
                    is_listed = True
                    break
        if not (is_stored or is_listed) :
            print(f"Bond {ai} - {aj} not found.")

    #Check angles
    for angle in echmol.angles.values():
        ai = echmol.atoms[angle.ai].name
        aj = echmol.atoms[angle.aj].name
        ak = echmol.atoms[angle.ak].name
        is_stored = False; is_listed = False
        for sang in angle_types.values():
            if ((aj == sang.aj) and 
                ((ai == sang.ai and ak == sang.ak) or 
                 (ai == sang.ak and ak == sang.ai))):
                is_stored = True
                break
        if not is_stored:
            for lang in angle_types_boss.values():
                if ((aj == lang.aj) and
                    ((ai == lang.ai and ak == lang.ak) or 
                     (ai == lang.ak and ak == lang.ai))):
                    ityp = len(angle_types) + 1
                    angle_types[ityp] = AngleRec(ai, aj, ak, lang.typ,
                                            lang.func, params=lang.params)
                    is_listed = True
                    break
        if not (is_stored or is_listed) :
            print(f"Angle {ai} - {aj} - {ak} not found.")

    #Check dihedrals
#   for dihedral in echmol.dihedrals.values():
    for key in range(1,echmol.num_dihedrals+1):
        dihedral = echmol.dihedrals[key]
        ai = echmol.atoms[dihedral.ai].name
        aj = echmol.atoms[dihedral.aj].name
        ak = echmol.atoms[dihedral.ak].name
        al = echmol.atoms[dihedral.al].name
        if echmol.name == 'TOL':
            print(dihedral)
            #print(dihedral.ai, dihedral.aj, dihedral.ak, dihedral.al)
        is_stored = False; is_listed = False
        for sdhd in dihedral_types.values():
            if (ai == sdhd.ai and aj == sdhd.aj and ak == sdhd.ak \
                and al == sdhd.al) or \
                (ai == sdhd.al and aj == sdhd.ak and ak == sdhd.aj \
                and al == sdhd.ai):
                is_stored = True
                break
        if not is_stored:
            for ldhd in dihedral_types_boss.values():
                if (ai == ldhd.ai and aj == ldhd.aj and ak == ldhd.ak \
                    and al == ldhd.al) or \
                    (ai == ldhd.al and aj == ldhd.ak and ak == ldhd.aj \
                    and al == ldhd.ai) :
                    ityp = len(dihedral_types) + 1
                    dihedral_types[ityp] = DihedralRec(ai, aj, ak, al, ldhd.typ,
                                            ldhd.func, params=ldhd.params)
                    is_listed = True
                    break
        if not (is_stored or is_listed) :
            print(f"Dihedral {ai} - {aj} - {ak} - {al} not found.")
            ityp = len(dihedral_types) + 1
            dihedral_types[ityp] = DihedralRec(ai, aj, ak, al, 0, func=3,
                                               params=[])

#Write out all interactions
fn_bonded = 'oplslpg.ff/ffbonded.itp'
with open(fn_bonded, 'w') as fh:
    fh.write('[ bondtypes ]\n')
    fh.write(';  i   j  func       b0          kb\n')
    for bt in bond_types.values():
        fh.write('  %2s%4s%6d%9.5f%12.2f\n'%(bt.ai, bt.aj, bt.func,
                                             bt.params[0], bt.params[1]))
    fh.write('\n[ angletypes ]\n')
    fh.write(';  i   j   k  func      th0       cth\n')
    for at in angle_types.values():
        fh.write('  %2s%4s%4s%6d%9.3f%10.3f\n'%(at.ai, at.aj, at.ak, at.func,
                                             at.params[0], at.params[1]))
    fh.write('\n[ dihedraltypes ]\n')
    fh.write(';  i   j   k   l  func   coefficients\n')
    for dt in dihedral_types.values():
        buf = '  %2s%4s%4s%4s%6d'%(dt.ai, dt.aj, dt.ak, dt.al, dt.func)
        buf += ''.join(['% 10.5f'%x for x in dt.params])
        fh.write(buf + '\n')
