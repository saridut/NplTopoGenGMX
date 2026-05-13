#!/usr/bin/env python

from _molecule import Molecule

#Solvents (with density at 25 C)
#Ipa = (Molecule('Ipa', 'solvents/ipa/slvnt_ipa.lmp'), 0.78149)
#Mch = (Molecule('Mch', 'solvents/mch/slvnt_mch.lmp'), 0.766)
Hexane = (Molecule(['molecules/HEXN.itp', 'molecules/HEXN.gro'], orient=True), 0.65485)
#Methanol = (Molecule('Methanol', 'solvents/methanol/slvnt_methanol.lmp'), 0.78633)
#Butanol = (Molecule('Butanol', 'solvents/butanol/slvnt_butanol.lmp'), 0.80577)
#Acetone = (Molecule('Acetone', 'solvents/acetone/slvnt_acetone.lmp'), 0.78658)
Toluene = (Molecule(['molecules/TOL.itp', 'molecules/TOL.gro'], orient=True), 0.86224)
Thf = (Molecule(['molecules/THF.itp', 'molecules/THF.gro'], orient=True), 0.8833)
#OleicAcid = (Molecule('OleicAcid', 
#            'ligands/carboxy_acid/C18_usat/C18_usat_acid.lmp'), 0.895)
