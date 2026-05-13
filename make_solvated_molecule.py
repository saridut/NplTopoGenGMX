#!/usr/bin/env python

import sys
import os
import shutil
import math
import numpy as np
from _molecule import Molecule, LigandMolecule
from solvated_molecule import SolvatedMolecule
from solvents import Hexane, Toluene, Thf

packmol_path=os.path.expanduser('~/soft/packmol/packmol')

dn_ff = 'oplslpg.ff'
fns_ff = ['forcefield.itp']

nC = "18U"
lig_cx = LigandMolecule(fns=[f"molecules/C{nC}T.itp", f"molecules/C{nC}T.gro"],
        head=2, tail=51, bind_group=[2])
lig_cx.reindex()

#lig_cx.to_file('C12T.itp', 'C12T.gro')
#raise SystemExit()
#for i in range(1,12+1):
#    lig_cx.add_position_restraint(i, func=2, g=1, r=0.1, k=1000)
#lig_cx.adjust_charge()
#print( lig_cx.get_total_charge() )

#solvent = Hexane
#solvent[0].reindex()
#solvent[0].adjust_charge()
#print( solvent[0].get_total_charge() )

tag = f"mol_{lig_cx.name}"
dn = f"mol_{lig_cx.name}"
if not os.path.isdir(dn):
    os.makedirs(dn)

#Copy forcefield files
for each in fns_ff:
    fn = os.path.join(dn_ff, each)
    shutil.copy2(fn, dn)


sm = SolvatedMolecule(name=tag, fns_ff=fns_ff)
sm.add_mol_one(lig_cx, restrain=False)

sm.apply_pbc(directions='xyz')

sm.set_group('System', atom_ids=range(1, sm.num_atoms+1))
sm.write_group(None, f"{dn}/groups.ndx")

sm.to_file(dn)
