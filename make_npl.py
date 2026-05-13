#!/usr/bin/env python

import sys
import os
import shutil
import math
import numpy as np
from _molecule import Molecule, LigandMolecule #Remove this
from nanoplatelet import Nanoplatelet
from solvents import Hexane, Toluene, Thf

packmol = os.path.expanduser('~/soft/packmol/packmol')
atomsk = os.path.expanduser('~/soft/atomsk/0.13.1/bin/atomsk')

dn_ff = 'oplslpg.ff'
fns_ff = ['forcefield.itp']

xtal_species = [Molecule(['molecules/Cd.itp', 'molecules/Cd.gro']),
                Molecule(['molecules/Se.itp', 'molecules/Se.gro'])
                ]
for species in xtal_species:
    species.add_position_restraint(1, func=2, g=1, r=0.1, k=1000)

#nC = "18U"
#lig_cx = LigandMolecule(fns=[f"molecules/C{nC}T.itp", f"molecules/C{nC}T.gro"],
#        head=2, tail=51, bind_group=[2])
ligands = [
           LigandMolecule(fns=[f"molecules/LGAC.itp", f"molecules/LGAC.gro"],
                head=5, tail=1, bind_group=[5, 6, 7]),
            LigandMolecule(fns=[f"molecules/LGOL.itp", f"molecules/LGOL.gro"],
                head=51, tail=1, bind_group=[51, 52, 53])
           ]
for each in ligands:
    each.reindex()
ligand_pop_ratio = np.array([1,1])
#raise SystemExit()
#lig_cy.reindex()

#lig_cx.to_file('C12T.itp', 'C12T.gro')
#raise SystemExit()
#for i in range(1,12+1):
#    lig_cx.add_position_restraint(i, func=2, g=1, r=0.1, k=1000)
#lig_cx.adjust_charge()
#print( lig_cx.get_total_charge() )
#raise SystemExit()

solvent = [Hexane[0], Thf[0]]
for each in solvent:
    each.reindex()
solvent_density = [Hexane[1], Thf[1]]
solvent_ratio=[2,1]
#raise SystemExit()

xtal_lx = 40
xtal_ly = 10
xtal_lz = 3
box_params = {'sepy': 0.0, 'sepz': 5.0}

apl = 0.40 #int(sys.argv[1]) #Area per ligand in nm^2
r0 = 0.50 # in nm
solvent_offset = 0.2 # in nm

#ligands = [lig_cx]
#ligands = [lig_cx, lig_cy]
#ligand_pop_ratio = np.array([1])

tag = f"slvnpl_ACOL_{solvent[0].name}"
#tag = 'xtal'
dn = f"ACOL" #Output files directory
if not os.path.isdir(dn):
    os.makedirs(dn)

#Copy forcefield files
for each in fns_ff:
    fn = os.path.join(dn_ff, each)
    shutil.copy2(fn, dn)

npl = Nanoplatelet(name=tag, fns_ff=fns_ff, form='slab', slab_pos='bot')

#Add crystal
npl.add_xtal(lx=xtal_lx, ly=xtal_ly, lz=xtal_lz, lattice='zb',
             species=xtal_species, a=0.614, c=None, 
             orient=['1_1_0', '-1_1_0', '0_0_1'], unit=['nm', 'nm', 'ML'], 
             mlcp=[0.0, 0.5], atomsk_path=atomsk, fn_info=f"{dn}/info.txt")
#npl.to_file(dn)
#npl.write_group(None, f"{dn}/groups.ndx")

#Add ligands
npl.add_ligands(ligands, ligand_pop_ratio, r0, balance_charge=True,
                  lattice='bcc', apl=apl, intermolecular_bond_func=None, #10
                  intermolecular_bond_params=[0.1, 0.288, 0.6, 2e5],
                  fn_info=f"{dn}/info.txt")
#print(len(npl.groups['LigandHeads']['atom_ids'])/2)
#npl.add_ligand_one(lig_cx, r0, intermolecular_bond_func=10, 
#                    intermolecular_bond_params=[0.1, 0.288, 0.6, 2e5],
#                   fn_info=f"{dn}/info.txt")
#npl.translate(r=npl.simbox[:,1]) 
#r = np.array([npl.simbox[0,1], npl.simbox[1,1], 0.0])
#npl.translate(r=r)
#npl.to_file(dn)
#npl.write_group(None, f"{dn}/groups.ndx")
#raise SystemExit()
#
#brush.adjust_charge()
#brush.apply_pbc(directions='xy', add_img_flag=True)


#Solvent
npl.solvate(solvent, solvent_density, solvent_ratio, box_params,
            offset=solvent_offset, offset_from='lig', rbuf=0.0, 
            packmol_tol=0.2, packmol_sidemax=100, packmol_path=packmol)
#brush.adjust_charge()

if npl.form == 'slab' and npl.slab_pos == 'bot':
    r = np.array([npl.simbox[0,1], npl.simbox[1,1], 0.0])
    npl.translate(r=r)
else:
    npl.translate(r=npl.simbox[:,1])

npl.apply_pbc(directions='xyz')

aids = npl.groups['Xtal']['atom_ids'] + npl.groups['Ligands']['atom_ids']
npl.set_group('LigXtal', atom_ids=aids)
npl.set_group('System', atom_ids=range(1, npl.num_atoms+1))
npl.write_group(None, f"{dn}/groups.ndx")

npl.to_file(dn)
