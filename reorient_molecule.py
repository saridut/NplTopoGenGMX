#!/usr/bin/env python

import sys
import os
import shutil
import math
import numpy as np
from _molecule import Molecule, LigandMolecule

dn_ff = 'oplslpg.ff'
fns_ff = ['forcefield.itp']

nC = "14"
lig = LigandMolecule(fns=[f"molecules/C{nC}T.itp", f"molecules/C{nC}T.gro"],
        head=15, tail=14, bind_group=[15], orient=True)
#lig.align(15,14, np.array([0,0,1]))
lig.reindex()
lig.align(1,8, np.array([0,0,1]))
#lig.translate(-lig.atoms[lig.head].coords)
#lig.fit_simbox(sep=0.0)

print(f"Ligand bbox = {lig.get_bbox()[1]}")

dn_out = "minimize/molecules"
lig.to_file(f"{dn_out}/C{nC}T.itp", f"{dn_out}/C{nC}T.gro", f"Ligand C{nC}")
