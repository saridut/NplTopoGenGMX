#!/usr/bin/env python

import sys
import os
import shutil
import math
import numpy as np
from _molecule import Molecule, LigandMolecule
from brush_gmx import Brush
from solvents import Hexane, Toluene, Thf

packmol_path=os.path.expanduser('~/soft/packmol/packmol')

dn_ff = 'oplslpg.ff'
fns_ff = ['forcefield.itp']

xmol = Molecule(['molecules/Au.itp', 'molecules/Au.gro'])
#xmol.add_position_restraint(1, func=2, g=1, r=0.1, k=1000)

nC = "20"
lig_cx = LigandMolecule(fns=[f"minimize/molecules/C{nC}T.itp",
                             f"minimize/molecules/C{nC}T.gro"],
        head=2, tail=59, bind_group=[2])

#Flat-bottomed position restraint (layer with z-normal)
#lig_cx.add_position_restraint(lig_cx.head, func=2, g=8, r=4.1, k=1000)

#heavy_atoms = lig_cx.get_heavy_atoms()
#for i in sorted(list(heavy_atoms)):
#    lig_cx.add_position_restraint(i, func=2, g=5, r=0.5, k=1000)

xlpar = 4.0778e-1 # in nm
xtal_lx = 7.8
xtal_ly = xtal_lx
xtal_lz = 11*xlpar
box_params = {'sepz': 7.0}

apl = 0.32 #int(sys.argv[1]) #Area per ligand in nm^2
llpar = math.sqrt(2*apl)
r0 = 0.20 # in nm
#solvent_offset = 2.5 # in nm

ligands = [lig_cx]
#ligands = [lig_cx, lig_cy]
ligand_pop_ratio = np.array([1])
#ligand_pop_ratio = np.array([1,1])

tag = f"ml_C{nC}_apl_{apl:.2f}" #_{solvent[0].name}"
dn = f"minimize/C{nC}/apl_{apl}"
#dn = f"C12-{solvent[0].name}/apl_{apl:.2g}"
if not os.path.isdir(dn):
    os.makedirs(dn)

#Copy forcefield files
for each in fns_ff:
    fn = os.path.join(dn_ff, each)
    shutil.copy2(fn, dn)

brush = Brush(name=tag, fns_ff=fns_ff, is_slab=True, slab_pos='mid')

#Add crystal
brush.add_xtal(xtal_lx, xtal_ly, xtal_lz, xlpar, xmol, out_dir=dn)
#brush.to_file(dn)
#brush.write_group(None, f"{dn}/groups.ndx")

#Add ligands
brush.add_ligands(ligands, ligand_pop_ratio, r0, balance_charge=False,
                  lattice='bcc', llpar=llpar, intermolecular_bond_func=10,
                  intermolecular_bond_params=[0.1, 0.288, 0.6, 2e5],
                  out_dir=dn)
print(len(brush.groups['LigandHeads']['atom_ids'])/2)
#brush.to_file(dn)
#brush.write_group(None, f"{dn}/groups.ndx")
#raise SystemExit()
#
#brush.adjust_charge()
brush.apply_pbc(directions='xy')


#Solvent
#brush.solvate(solvent, box_params, xtal_offset=solvent_offset, packmol_tol=0.2,
#              packmol_sidemax=100, packmol_path=packmol_path)
#brush.adjust_charge()

if brush.slab_pos == 'mid':
    brush.simbox[2,1] += box_params['sepz']
    brush.simbox[2,0] -= box_params['sepz']
    brush.translate(r=brush.simbox[:,1])
else:
    brush.simbox[2,1] += box_params['sepz']
    r = np.array([brush.simbox[0,1], brush.simbox[1,1], 0.0])
    brush.translate(r=r)

#brush.apply_pbc(directions='xyz')

aids = brush.groups['Xtal']['atom_ids'] + brush.groups['Ligands']['atom_ids']
brush.set_group('LigXtal', atom_ids=aids)
brush.set_group('System', atom_ids=range(1, brush.num_atoms+1))
brush.write_group(None, f"{dn}/groups.ndx")

brush.to_file(dn)
