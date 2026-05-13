#!/usr/bin/env python

import sys
import os
import shutil
import math
import numpy as np
from _molecule import Molecule
from solvent_box import SolventBox
from solvents import Hexane, Toluene, Thf

packmol_path=os.path.expanduser('~/soft/packmol/packmol')

dn_ff = 'oplslpg.ff'
fns_ff = ['forcefield.itp']

solvent = Hexane
solvent[0].reindex()

box_params = {'boxx': 6.0, 'boxy': 6.0, 'boxz': 18.0}

tag = f"slvnt_{solvent[0].name}"
dn = f"minimize/{solvent[0].name}"
if not os.path.isdir(dn):
    os.makedirs(dn)

#Copy forcefield files
for each in fns_ff:
    fn = os.path.join(dn_ff, each)
    shutil.copy2(fn, dn)

sb = SolventBox(name=tag, fns_ff=fns_ff)

#Solvent
sb.add_solvent(solvent, box_params, packmol_tol=0.2, packmol_sidemax=100,
               packmol_path=packmol_path)

sb.translate(r=sb.simbox[:,1])

sb.apply_pbc(directions='xyz')

sb.set_group('System', atom_ids=range(1, sb.num_atoms+1))
sb.write_group(None, f"{dn}/groups.ndx")

sb.to_file(dn)
