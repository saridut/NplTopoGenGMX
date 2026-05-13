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

#lig_cx = Molecule(fns=['molecules/C20T.itp', 'molecules/C20T.gro'])
#lig_cx.reindex()
#raise SystemExit()

nC = "18"
lig_cx = LigandMolecule(fns=[f"molecules/C{nC}T.itp", f"molecules/C{nC}T.gro"],
        head=2, tail=53, bind_group=[2])
lig_cx.reindex()
#lig_cx.to_file('C12T.itp', 'C12T.gro')
#raise SystemExit()
heavy_atoms = lig_cx.get_heavy_atoms()
for i in sorted(list(heavy_atoms)):
    lig_cx.add_position_restraint(i, func=2, g=1, r=0.1, k=1000)
#lig_cx.adjust_charge()
#print( lig_cx.get_total_charge() )

solvent = Thf
solvent[0].reindex()
#solvent[0].adjust_charge()
#print( solvent[0].get_total_charge() )
#raise SystemExit()

xlpar = 4.0778e-1 # in nm
xtal_lx = 5.0
xtal_ly = xtal_lx
xtal_lz = 11*xlpar
box_params = {'sepz': 6.0}

apl = 0 #0.32 #int(sys.argv[1]) #Area per ligand in nm^2
r0 = 0.20 # in nm
solvent_offset = 0.2 # in nm

#ligands = [lig_cx]
#ligand_pop_ratio = np.array([1])

tag = f"slvbrush_C{nC}_apl_{apl}_{solvent[0].name}"
dn = f"C{nC}-{solvent[0].name}/apl_{apl}"
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
brush.add_ligand_one(lig_cx, r0, intermolecular_bond_func=10,
                  intermolecular_bond_params=[0.1, 0.288, 0.6, 2e5],
                  out_dir=dn)
#print(len(brush.groups['LigandHeads']['atom_ids'])/2)
#brush.to_file(dn)
#brush.write_group(None, f"{dn}/groups.ndx")
#raise SystemExit()
#
#brush.adjust_charge()
#brush.apply_pbc(directions='xy', add_img_flag=True)


#Solvent
brush.solvate_one(solvent, box_params, xtal_offset=solvent_offset, packmol_tol=2.0,
              packmol_sidemax=1.0e3, packmol_path=packmol_path)
#print(brush.num_atoms)
#raise SystemExit()
#brush.adjust_charge()

if brush.slab_pos == 'mid':
    brush.translate(r=brush.simbox[:,1])
else:
    r = np.array([brush.simbox[0,1], brush.simbox[1,1], 0.0])
    brush.translate(r=r)

brush.apply_pbc(directions='xyz')

aids = brush.groups['Xtal']['atom_ids'] + brush.groups['Ligands']['atom_ids']
brush.set_group('LigXtal', atom_ids=aids)
brush.set_group('System', atom_ids=range(1, brush.num_atoms+1))
brush.write_group(None, f"{dn}/groups.ndx")

brush.to_file(dn)
