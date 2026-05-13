#!/usr/bin/env python

import os
import warnings
import subprocess
from pathlib import Path
import numpy as np
import pprint


def get_lines_from_file(fn):
    """
    Return all non-blank lines from a file after stripping comments.

    """
    with open(fn, 'r') as fh:
        lns = [y for s in fh if len(y := s.strip(" \n").partition(";")[0]) > 0]
    return lns


def resolve_include(lines):
    """
    Resolves include directives.

    """
    indxval = [(i,s) for i,s in enumerate(lines) if s.startswith("#include")]
    print(f"# include directives = {len(indxval)}")
    if indxval:
        i = indxval[0][0]; s = indxval[0][1].split()[1]
        print(indxval)
        if s.startswith('<'):
            fn_inc = s.strip('<>')
        elif s.startswith('\"'):
            fn_inc = s.strip('\"')
        else:
            raise ValueError(f"Include file name quoted with ({s[0]} {s[-1]}), \
                    quote characters must be (< >) or (\" \").")
        lines_fn_inc = get_lines_from_file(fn_inc)
        lines_winc = lines[0:i] + lines_fn_inc + lines[(i+1):]
        lines = resolve_include(lines_winc)
    return lines


def resolve_ifdef(lines):
    """
    Resolves #define, #undef, #ifdef, #ifndef, #else, and #endif directives.

    """
    defined = set()
    lstat = True
    ifcond = []
    delete_lines = [] #Indices of lines to be deleted
    for i,line in enumerate(lines):
        if line.startswith('#ifdef'):
            ifcond.append(line.split()[1] in defined)
            delete_lines.append(i)
            lstat = all(ifcond)
        elif line.startswith('#ifndef'):
            ifcond.append(line.split()[1] not in defined)
            delete_lines.append(i)
            lstat = all(ifcond)
        elif line.startswith('#else'):
            ifcond[-1] = not ifcond[-1]
            delete_lines.append(i)
            lstat = all(ifcond)
        elif line.startswith('#endif'):
            _ = ifcond.pop()
            lstat = all(ifcond)
            delete_lines.append(i)
        elif line.startswith('#define'):
            if lstat: 
                defined.add(line.split()[1])
            delete_lines.append(i)
        elif line.startswith('#undef'):
            if lstat:
                if (key := line.split()[1]) in defined:
                    defined.remove(key)
            delete_lines.append(i)
        else:
            if not lstat:
                delete_lines.append(i)

        for i in delete_lines.reverse():
            lines.pop(i)
        return lines


def write_top(config, fn, title=''):
    '''
    Writes a gromacs topology file (.top).  

    '''

    na = config.num_atoms

    with open(fn,'w') as fh:
        fh.write(str(na) + '\n')
        fh.write(title + '\n')

        for i in range(1, na+1):
            at = config.atoms[i]['type']
            atm_nam = config.atom_names[at]
            coords = config.atoms[i]['coords']
            fh.write( '%s  '%atm_nam 
                + '  '.join(['% .15g'%x for x in coords]) + '\n')


def read_top(config, fn):
    '''
    Reads from a gromacs topology file (.top) or include file (.itp).  

    '''
    config.clear()
    fn_pp = 'tmp.txt'
    lines = get_lines_from_file(fn)
    lines = resolve_include(lines)
    lines = resolve_ifdef(lines)
#   with open(fn_pp, 'w') as fh_pp:
#       for s in lines:
#           fh_pp.write(s + '\n')
    for line in lines:
        wrds = line.split()
        if (wrds[0]=='[') and (wrds[2]==']') and (len(wrds) == 3):
            this_section = wrds[1]
            continue
        if this_section == 'defaults':
            nbfunc = int(wrds[0])
            comb_rule = int(wrds[1])
            gen_pairs = wrds[2]
            fudgeLJ = float(wrds[3])
            fudgeQQ = float(wrds[4])
        elif this_section == 'atomtypes':
            at = wrds[0]
            if len(words) == 8:
                abt_name = wrds[1]
                ano = int(wrds[2])
                mass = float(wrds[3])
                charge = float(wrds[4])
                pt = wrds[5]
                V = float(wrds[6])
                W = float(wrds[7])
            elif len(words) == 7:
                abt_name = wrds[1]
                mass = float(wrds[2])
                charge = float(wrds[3])
                pt = wrds[4]
                V = float(wrds[5])
                W = float(wrds[6])
            else:
                mass = float(wrds[1])
                charge = float(wrds[2])
                pt = wrds[3]
                V = float(wrds[4])
                W = float(wrds[5])
        elif this_section == 'pairtypes':
            pass
        elif this_section == 'nonbond_params':
            pass
        elif this_section == 'bondtypes':
            pass
        elif this_section == 'angletypes':
            pass
        elif this_section == 'dihedraltypes':
            pass
        elif this_section == 'constrainttypes':
            pass
        elif this_section == 'molecule_type':
            pass
        elif this_section == 'atoms':
            pass
        elif this_section == 'pairs':
            pass
        elif this_section == 'pairs_nb':
            pass
        elif this_section == 'bonds':
            pass
        elif this_section == 'angles':
            pass
        elif this_section == 'dihedrals':
            pass
        elif this_section == 'constraints':
            pass
        elif this_section == 'exclusions':
            pass
        elif this_section == 'settles':
            pass
        elif this_section == 'position_restraints':
            pass
        elif this_section == 'distance_restraints':
            pass
        elif this_section == 'angle_restraints':
            pass
        elif this_section == 'angle_restraints_z':
            pass
        elif this_section == 'dihedral_restraints':
            pass
        elif this_section == 'orientation_restraints':
            pass
        elif this_section == 'system':
            pass
        elif this_section == 'molecules':
            pass
        elif this_section == 'intermolecular_interactions':
            pass



#-------------------------------------------------------------------------------

def write_ldf(config, fn, title='', with_pc=True):
    '''
    Writes configuration to a LAMMPS data file (can be imported in Ovito).

    with_pc : bool
        Whether to write pair force field parameters

    '''

    with open(fn,'w') as fh:
        fh.write('#' + title + '\n')

        fh.write('\n')
        fh.write('%d atoms\n'%config.num_atoms)
        fh.write('%d atom types\n'%config.num_atom_types)

        if config.num_bonds > 0:
            fh.write('\n')
            fh.write('%d bonds\n'%config.num_bonds)
            fh.write('%d bond types\n'%config.num_bond_types)

        if config.num_angles > 0:
            fh.write('\n')
            fh.write('%d angles\n'%config.num_angles)
            fh.write('%d angle types\n'%config.num_angle_types)

        if config.num_dihedrals > 0:
            fh.write('\n')
            fh.write('%d dihedrals\n'%config.num_dihedrals)
            fh.write('%d dihedral types\n'%config.num_dihedral_types)

        if config.num_impropers > 0:
            fh.write('\n')
            fh.write('%d impropers\n'%config.num_impropers)
            fh.write('%d improper types\n'%config.num_improper_types)

        fh.write('\n')
        fh.write('%.15g  %.15g  xlo xhi\n'%(config.simbox[0,0], config.simbox[0,1]))
        fh.write('%.15g  %.15g  ylo yhi\n'%(config.simbox[1,0], config.simbox[1,1]))
        fh.write('%.15g  %.15g  zlo zhi\n'%(config.simbox[2,0], config.simbox[2,1]))
        #fh.write('%.15g  %.15g  %.15g  xy xz yz\n'%(config.tilt_factors[0],
        #    config.tilt_factors[1], config.tilt_factors[2])) #Tilt factors

        fh.write('\n')
        fh.write('Masses\n')
        fh.write('\n')
        for iat in range(1,config.num_atom_types+1):
            fh.write('%d  %.8g\n'%(iat, config.atom_mass[iat]))

        if len(config.pair_coeffs) > 0 and with_pc:
            fh.write('\n')
            fh.write('Pair Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.pair_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.pair_coeffs[i])
                fh.write(buf+'\n')

        if len(config.bond_coeffs) > 0 :
            fh.write('\n')
            fh.write('Bond Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.bond_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.bond_coeffs[i])
                fh.write(buf+'\n')

        if len(config.angle_coeffs) > 0 :
            fh.write('\n')
            fh.write('Angle Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.angle_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.angle_coeffs[i])
                fh.write(buf+'\n')

        if len(config.dihedral_coeffs) > 0 :
            fh.write('\n')
            fh.write('Dihedral Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.dihedral_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.dihedral_coeffs[i])
                fh.write(buf+'\n')

        if len(config.improper_coeffs) > 0 :
            fh.write('\n')
            fh.write('Improper Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.improper_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.improper_coeffs[i])
                fh.write(buf+'\n')

        fh.write('\n')
        fh.write('Atoms # full\n')
        fh.write('\n')

        #Check if image flags are present (must be present for all atoms or
        #absent for all atoms)
        if 'img_flag' in config.atoms[1]:
            has_img_flag = True
        else :
            has_img_flag = False
        #Write out atom data
        for iatm in range(1, config.num_atoms+1):
            at = config.atoms[iatm]['type']
            chge = config.atoms[iatm]['charge']
            imol = config.atoms[iatm]['imol']
            coords = config.atoms[iatm]['coords']
            buf = '%d  %d  %d  % .15g  '%(iatm, imol, at, chge) \
                + '  '.join( ['% .15g '%x for x in coords] )
            if has_img_flag: 
                img_flag = config.atoms[iatm]['img_flag']
                buf += '  '.join( ['% d'%x for x in img_flag] )
            fh.write(buf+'\n')

        if config.num_bonds > 0:
            fh.write('\n')
            fh.write('Bonds\n')
            fh.write('\n')
            for i in range(1, config.num_bonds+1):
                bt = config.bonds[i]['type']
                atm_i = config.bonds[i]['atm_i']
                atm_j = config.bonds[i]['atm_j']
                buf = '%d  %d  %d  %d\n'%(i, bt, atm_i, atm_j)
                fh.write(buf)

        if config.num_angles > 0:
            fh.write('\n')
            fh.write('Angles\n')
            fh.write('\n')
            for i in range(1, config.num_angles+1):
                ant   = config.angles[i]['type']
                atm_i = config.angles[i]['atm_i']
                atm_j = config.angles[i]['atm_j']
                atm_k = config.angles[i]['atm_k']
                buf = '%d  %d  %d  %d  %d\n'%(i, ant, atm_i, atm_j, atm_k)
                fh.write(buf)

        if config.num_dihedrals > 0:
            fh.write('\n')
            fh.write('Dihedrals\n')
            fh.write('\n')
            for i in range(1, config.num_dihedrals+1):
                dt    = config.dihedrals[i]['type']
                atm_i = config.dihedrals[i]['atm_i']
                atm_j = config.dihedrals[i]['atm_j']
                atm_k = config.dihedrals[i]['atm_k']
                atm_l = config.dihedrals[i]['atm_l']
                buf = '%d  %d  %d  %d  %d  %d\n'%(i, dt, atm_i, atm_j, atm_k, atm_l)
                fh.write(buf)

        if config.num_impropers > 0:
            fh.write('\n')
            fh.write('Impropers\n')
            fh.write('\n')
            for i in range(1, config.num_impropers+1):
                it    = config.impropers[i]['type']
                atm_i = config.impropers[i]['atm_i']
                atm_j = config.impropers[i]['atm_j']
                atm_k = config.impropers[i]['atm_k']
                atm_l = config.impropers[i]['atm_l']
                buf = '%d  %d  %d  %d  %d  %d\n'%(i, it, atm_i, atm_j, atm_k, atm_l)
                fh.write(buf)

        if len(config.velocities) > 0:
            fh.write('\n')
            fh.write('Velocities\n')
            fh.write('\n')
            for i in range(1, len(config.velocities)+1):
                atm_id = i
                v = config.velocities[atm_id]
                buf = '%d  % .15g  % .15g  % .15g\n'%(atm_id, v[0], v[1], v[2])
                fh.write(buf)

#-------------------------------------------------------------------------------

def read_ldf(config, fn):
    '''
    Reads configuration from a LAMMPS data file.

    '''
    atm_toff = config.num_atom_types
    bnd_toff = config.num_bond_types
    ang_toff = config.num_angle_types
    dhd_toff = config.num_dihedral_types
    imp_toff = config.num_improper_types

    atm_idoff = len(config.atoms)
    bnd_idoff = len(config.bonds)
    ang_idoff = len(config.angles)
    dhd_idoff = len(config.dihedrals)
    imp_idoff = len(config.impropers)
    mol_idoff = config.num_molecules

    fh = open(fn,'r')
    #Skip the first line
    line = fh.readline()
    is_header = True #header flag

    while True:
        line = fh.readline()
        #Check EOF
        if not line:
            break
        line = line.strip(' \n')
        #Remove comments 
        m = line.find('#')
        if m != -1:
            line = line[0:m] 
        #Remove blank spaces after removing comment substring
        line = line.strip()
        #Skip blank lines
        if not line:
            continue

        #print(line)
        if line.endswith('atom types'):
            num_atom_types = int(line.split(maxsplit=1)[0])
            for i in range(num_atom_types):
                config.add_atom_type()
        elif line.endswith('bond types'):
            num_bond_types = int(line.split(maxsplit=1)[0])
            for i in range(num_bond_types):
                config.add_bond_type()
        elif line.endswith('angle types'):
            num_angle_types = int(line.split(maxsplit=1)[0])
            for i in range(num_angle_types):
                config.add_angle_type()
        elif line.endswith('dihedral types'):
            num_dihedral_types = int(line.split(maxsplit=1)[0])
            for i in range(num_dihedral_types):
                config.add_dihedral_type()
        elif line.endswith('improper types'):
            num_improper_types = int(line.split(maxsplit=1)[0])
            for i in range(num_improper_types):
                config.add_improper_type()

        elif line.endswith('atoms'):
            num_atoms = int(line.split(maxsplit=1)[0])
        elif line.endswith('bonds'):
            num_bonds = int(line.split(maxsplit=1)[0])
        elif line.endswith('angles'):
            num_angles = int(line.split(maxsplit=1)[0])
        elif line.endswith('dihedrals'):
            num_dihedrals = int(line.split(maxsplit=1)[0])
        elif line.endswith('impropers'):
            num_impropers = int(line.split(maxsplit=1)[0])

        elif line.endswith('xlo xhi'):
            words = line.split()
            config.simbox[0,0] = min(config.simbox[0,0], float(words[0]))
            config.simbox[0,1] = max(config.simbox[0,1], float(words[1]))
        elif line.endswith('ylo yhi'):
            words = line.split()
            config.simbox[1,0] = min(config.simbox[1,0], float(words[0]))
            config.simbox[1,1] = max(config.simbox[1,1], float(words[1]))
        elif line.endswith('zlo zhi'):
            words = line.split()
            config.simbox[2,0] = min(config.simbox[2,0], float(words[0]))
            config.simbox[2,1] = max(config.simbox[2,1], float(words[1]))
        #Tricilinic boxes not considered
        elif line.endswith('xy xz yz'):
            pass
            #config.tilt_factors[0] = float(words[0])
            #config.tilt_factors[1] = float(words[1])
            #config.tilt_factors[2] = float(words[2])

        #Read body section
        elif line == 'Masses':
            fh.readline() #Skip the line following a section header
            for i in range(num_atom_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = atm_toff + int(wrds[0])
                config.set_atom_mass(typ, float(wrds[1]))
                
        elif line == 'Atoms':
            fh.readline() #Skip the line following a section header
            for i in range(num_atoms):
                wrds = fh.readline().rstrip(' \n').split()
                atm_id = atm_idoff + int(wrds[0])
                imol = int(wrds[1])
                if imol !=0:
                    imol += mol_idoff
                typ = atm_toff + int(wrds[2])
                charge = float(wrds[3])
                coords = np.array([float(wrds[4]), float(wrds[5]), float(wrds[6])])
                config.add_atom(typ, charge, coords, imol, atm_id)
                if len(wrds) == 10:
                    img_flag = np.array([int(wrds[7]), int(wrds[8]), int(wrds[9])],
                                        dtype=np.int32)
                    config.add_img_flag(atm_id, img_flag)

        elif line == 'Velocities':
            fh.readline() #Skip the line following a section header
            for i in range(num_atoms):
                wrds = fh.readline().rstrip(' \n').split()
                atm_id = atm_idoff + int(wrds[0])
                vel = np.array([float(wrds[1]), float(wrds[2]), float(wrds[3])])
                config.add_velocity(atm_id, vel)

        elif line == 'Bonds':
            fh.readline() #Skip the line following a section header
            for i in range(num_bonds):
                wrds = fh.readline().rstrip(' \n').split()
                bnd_id = bnd_idoff + int(wrds[0])
                typ = bnd_toff + int(wrds[1])
                config.add_bond(typ, atm_idoff+int(wrds[2]), 
                        atm_idoff+int(wrds[3]), bnd_id)

        elif line == 'Angles':
            fh.readline() #Skip the line following a section header
            for i in range(num_angles):
                wrds = fh.readline().rstrip(' \n').split()
                ang_id = ang_idoff + int(wrds[0])
                typ = ang_toff + int(wrds[1])
                config.add_angle(typ, atm_idoff+int(wrds[2]), 
                    atm_idoff+int(wrds[3]), atm_idoff+int(wrds[4]), ang_id)

        elif line == 'Dihedrals':
            fh.readline() #Skip the line following a section header
            for i in range(num_dihedrals):
                wrds = fh.readline().rstrip(' \n').split()
                dhd_id = dhd_idoff + int(wrds[0])
                typ = dhd_toff + int(wrds[1])
                config.add_dihedral(typ, atm_idoff+int(wrds[2]), 
                    atm_idoff+int(wrds[3]), atm_idoff+int(wrds[4]),
                    atm_idoff+int(wrds[5]), dhd_id)

        elif line == 'Impropers':
            fh.readline() #Skip the line following a section header
            for i in range(num_impropers):
                wrds = fh.readline().rstrip(' \n').split()
                imp_id = imp_idoff + int(wrds[0])
                typ = imp_toff + int(wrds[1])
                config.add_improper(typ, atm_idoff+int(wrds[2]), 
                    atm_idoff+int(wrds[3]), atm_idoff+int(wrds[4]), 
                    atm_idoff+int(wrds[5]), imp_id)

        elif line == 'Pair Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_atom_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = atm_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_pair_coeff(typ, coeffs)

        elif line == 'Bond Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_bond_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = bnd_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_bond_coeff(typ, coeffs)

        elif line == 'Angle Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_angle_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = ang_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_angle_coeff(typ, coeffs)
            
        elif line == 'Dihedral Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_dihedral_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = dhd_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_dihedral_coeff(typ, coeffs)
            
        elif line == 'Improper Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_improper_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = imp_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_improper_coeff(typ, coeffs)
        else:
            raise IOError('Unrecognized header line: "%s"'%line)
            
    fh.close()

#-------------------------------------------------------------------------------

def ldf_to_xyz(fn_ldf, fn_xyz, atom_names=None):
    """
    Converts a LAMMPS data file to XYZ file.

    Parameters
    ---------
    fn_ldf : str or pathlib.Path
        Name of the LAMMPS data file to read.
    fn_xyz : str or pathlib.Path
        Name of the XYZ file to write to.
    atom_names : list of tuples
        Mapping of atom type to atom names. E.g. [(1, 'H'), (2, 'Se'), ...].
        If None, the default name is `Xi`, where `i` is the corresponding atom
        type.

    Returns
    -------
    None

    """
    config = Configuration()
    read_ldf(config, fn_ldf)
    config.set_atom_names(atoms_names)
    write_xyz(config, fn_xyz, title='')





def write_grp_lammps(config, fn, gname, lmp_gid):
    """
    Write atoms ids of a group for including in a Lammps script.

    """
    with open(fn, 'w') as fh:
        buf = f"group {lmp_gid} id "
        v = config.groups[gname]['atoms']
        if isinstance(v, range):
            buf += f" {v[0]}:{v[-1]}:{v.step}"
        else:
            buf += f" {' '.join([str(x) for x in v])}"
        fbuf = pprint.pformat(buf, width=80, compact=True)
        lines = fbuf[1:-1].splitlines()
        nlines = len(lines)
        #Write first line
        if nlines == 1:
            fh.write(lines[0].strip(" '") + '\n')
        else:
            fh.write(lines[0].strip(" '") + ' &\n')
        #Lines following the first are indented
        if nlines > 2:
            for each in lines[1:-1]:
                fh.write('  ' + each.strip(" '") + ' &\n')
        #Write last line
        if nlines > 1:
            fh.write('  ' + lines[-1].strip(" '") + '\n')


def write_mol_grp(config, fn, title=''):
    """
    Write molecule and group records.

    """
    with open(fn, 'w') as fh:
        fh.write('%s\n'%title)
        fh.write('GROUPS %d\n'%config.num_groups)
        for key,val in config.groups.items():
            buf = f"{key}"
            for k, v in val.items():
                buf += f" {k}"
                if isinstance(v, range):
                    buf += f" {v[0]}:{v[-1]}:{v.step}"
                else:
                    buf += f" {len(v)} {' '.join([str(x) for x in v])}"
            fbuf = pprint.pformat(buf, width=80, compact=True)
            lines = fbuf[1:-1].splitlines()
            nlines = len(lines)
            #Write first line
            if nlines == 1:
                fh.write(lines[0].strip(" '") + '\n')
            else:
                fh.write(lines[0].strip(" '") + ' &\n')
            #Lines following the first are indented
            if nlines > 2:
                for each in lines[1:-1]:
                    fh.write('  ' + each.strip(" '") + ' &\n')
            #Write last line
            if nlines > 1:
                fh.write('  ' + lines[-1].strip(" '") + '\n')
        fh.write('\n')
        fh.write('MOLECULES %d\n'%config.num_molecules)
        for key,val in config.molecules.items():
            fh.write(f"  {key} {val['name']} {val['atm_beg']} {val['atm_end']}\n")



def read_mol_grp(config, fn):
    """
    Reads molecule and group records from a file.

    """
    with open(fn, 'r') as fh:
        fh.readline() #Skip title line
        num_groups = int( fh.readline().strip(' \n').split()[1] )
        for i in range(num_groups):
            line = fh.readline().strip(' \n')
            words = line.rstrip('&').split()
            while line.endswith('&'):
                line = fh.readline().strip(' \n')
                words += line.rstrip('&').split()

            num_words = len(words)
            gname = words[0]
            atom_types=None; atoms=None; molecules=None
            indx = 1
            while indx < num_words:
                key = words[indx]
                if ':' in words[indx+1]:
                    fields = [int(x) for x in words[indx+1].split(':')]
                    values = range(fields[0], fields[1]+1, fields[2])
                    indx += 2
                else:
                    n = int(words[indx+1])
                    values = [int(x) for x in words[indx+2:indx+2+n]] 
                    indx += (n+2)
                if key == 'atom_types':
                    atom_types = values
                elif key == 'atoms':
                    atoms = values
                elif key == 'molecules':
                    molecules = values

            config.set_group(gname, atom_types=atom_types, atoms=atoms,
                             molecules=molecules)

        fh.readline() #Skip blank line
        num_molecules = int( fh.readline().strip(' \n').split()[1] )
        for i in range(num_molecules):
            words = fh.readline().strip(' \n').split()
            config.molecules[i+1] = {'name': words[1], 'atm_beg': int(words[2]),
                                     'atm_end': int(words[3])}
