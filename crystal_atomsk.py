#!/usr/bin/env python

import math
import subprocess
import numpy as np
from pathlib import Path
from copy import deepcopy


def _get_lattice_points_2d(lattice, normal_axis, normal_pos,
                           lpar, lo, hi, boundary='pp', num_sites=0):
    """
    Generates points on a 2D lattice embedded in 3D space.

    Parameters
    ----------
    lattice : str
        {'sc', 'bcc'}
    normal_axis : {0, 1, 2}
        Axis along the positive normal. If `normal_axis = 0`, the elements of
        `lpar`, `lo`, & `hi` contain the corresponding values along axes 1 & 2,
        respectively.  Likewise, for `normal_axis = 1` the elemental values are
        along axes 0 & 2 and for `normal_axis = 2` the elemental values are
        along axes 1 & 2. 
    normal_pos : float
        Position of the lattice plane along the normal_axis
    lpar : scalar/list/tuple/ndarray of size (2,)
        Lattice parameter
    lo : (2,) array-like
        Lower bounds along two directions
    hi : (2,) array-like
        Upper bounds along two directions
    boundary : str
        Two characters specifying the nature of the boundary along the two
        axes. Allowed values are {'p', 'n', 'c'}. 'p' denotes periodic
        boundary, 'n' denotes non-periodic boundary, and 'c' denotes an
        end-capped boundary. For 'p' & 'c', `lpar` will be modified if
        necessary.
    num_sites : int
        Minimum number of lattice points. If zero, this parameter is ignored.

    Returns
    -------
    coords : ndarray of size (n,3)
        array of the coordinates of the lattice points

    """
    #Check boundary flags to see if the lattice parameter needs to be modified
    if np.isscalar(lpar):
        lpar_ = np.ones((2,))*lpar
    else:
        lpar_ = np.array(lpar)

    taxes = [0,1,2]; taxes.remove(normal_axis)
    nuc = np.zeros((2,), dtype=np.int32) #Number of unit cells along each direction
    l = np.asarray(hi) - np.asarray(lo)
    while True:
        for i in range(2):
            if boundary[i]=='p':
                nuc[i] = round(l[i]/lpar_[i])
                if nuc[i] < 1: nuc[i] = 1
                lpar_[i] = l[i]/nuc[i]
            elif boundary[i]=='c':
                nuc[i] = round(l[i]/lpar_[i])
                if nuc[i] < 1: nuc[i] = 1
                lpar_[i] = l[i]/nuc[i]
                nuc[i] += 1
            elif boundary[i]=='n':
                nuc[i] = math.floor(1 + l[i]/lpar_[i])

        coords = []
        origin = np.zeros((3,))
        origin[normal_axis] = normal_pos
        origin[taxes[0]] = lo[0]; origin[taxes[1]] = lo[1]
        coord = np.zeros((3,))
        a = lpar_[0]; b = lpar_[1]
        for j in range(nuc[1]):
            for i in range(nuc[0]):
                if lattice == 'sc':
                    coord[taxes[0]] = i*a; coord[taxes[1]] = j*b
                    coords.append(origin+coord)
                elif lattice == 'bcc':
                    coord[taxes[0]] = i*a;
                    coord[taxes[1]] = j*b
                    coords.append(origin+coord)

                    coord[taxes[0]] = (i+0.5)*a
                    coord[taxes[1]] = (j+0.5)*b
                    coords.append(origin+coord)

        #Check if boundary conditions are satisfied
        coords_final = []
        for each in coords:
            add = True
            for i in range(2):
                if (boundary[i] in 'cn') and (each[taxes[i]] > hi[i]):
                    add = False; break
            if add: 
                coords_final.append(each)

        #Number of lattice points
        ns = len(coords_final)
        if ns >= num_sites:
            break
        else:
            #Updated lattice parameters
            m = np.argmax(l); p = [0, 1]; p.remove(m)
            nuc[m] += 1
            if l[m] == l[p[0]]: nuc[p[0]] += 1
            lpar_[:] = (np.asarray(hi) - np.asarray(lo))/nuc

    print(f"Lattice parameters = ({lpar_[0]}, {lpar_[1]})")
    return np.array(coords_final)



def get_lattice_points(lattice, lpar, lo, hi, boundary='ppp', num_sites=0):
    """
    Generates points on a lattice.

    Parameters
    ----------
    lattice : str
        {'sc', 'bcc', 'fcc'}
    lpar : scalar/list/tuple/ndarray of size (3,)
        Lattice parameter
    lo : (3,) array-like
        Lower bounds along x, y, & z directions
    hi : (3,) array-like
        Upper bounds along x, y, & z directions
    boundary : str
        Three characters specifying the nature of the boundary along the x, y,
        & z axes. Allowed values are {'p', 'n', 'c'}. 'p' denotes periodic
        boundary, 'n' denotes non-periodic boundary, and 'c' denotes an
        end-capped boundary. For 'p' & 'c', `lpar` will be modified if
        necessary.
    num_sites : int
        Minimum number of lattice points. If zero, this parameter is ignored.

    Returns
    -------
    coords : ndarray of size (n,3)
        array of the coordinates of the lattice points

    Notes
    -----
    It is possible to set the lower and upper bounds equal to each other along
    only one of the three dimensions -- resulting in a 2D lattice. In this case
    `lattice` must be either `sc` or `bcc`.

    """
    #Check if lattice is 2D. If so, set the number of unit cells along the
    #third dimension equal to one.
    if np.any(np.array(lo) == np.array(hi)):
        assert np.count_nonzero(np.array(lo) == np.array(hi)) <= 1
        assert lattice in ['sc', 'bcc']
        normal_axis = np.nonzero(np.array(lo) == np.array(hi))[0][0]
        taxes = [0,1,2]; taxes.remove(normal_axis)
        normal_pos = lo[normal_axis]
        if np.isscalar(lpar):
            lpar_ = lpar
        else:
            lpar_ = np.asarray([ lpar[taxes[0]], lpar[taxes[1]] ])
        lo_ = np.asarray([ lo[taxes[0]], lo[taxes[1]] ])
        hi_ = np.asarray([ hi[taxes[0]], hi[taxes[1]] ])
        bndry = boundary[taxes[0]] + boundary[taxes[1]]
        coords = _get_lattice_points_2d(lattice, normal_axis, normal_pos,
                                        lpar_, lo_, hi_, bndry, num_sites)
        return coords

    #Check boundary flags to see if the lattice parameter needs to be modified
    if np.isscalar(lpar):
        lpar_ = np.ones((3,))*lpar
    else:
        lpar_ = np.asarray(lpar)
    nuc = np.zeros((3,), dtype=np.int32) #Number of unit cells along each direction
    l = np.asarray(hi) - np.asarray(lo)
    while True:
        for i in range(3):
            if boundary[i]=='p':
                nuc[i] = round(l[i]/lpar_[i])
                if nuc[i] < 1: nuc[i] = 1
                lpar_[i] = l[i]/nuc[i]
            elif boundary[i]=='c':
                nuc[i] = round(l[i]/lpar_[i])
                if nuc[i] < 1: nuc[i] = 1
                lpar_[i] = l[i]/nuc[i]
                nuc[i] += 1
            elif boundary[i]=='n':
                nuc[i] = math.ceil(l[i]/lpar_[i])

        coords = []
        origin = np.array(lo)
        coord = np.zeros((3,))
        a = lpar_[0]; b = lpar_[1]; c = lpar_[2]
        for k in range(nuc[2]):
            for j in range(nuc[1]):
                for i in range(nuc[0]):
                    if lattice == 'sc':
                        coord[0] = i*a; coord[1] = j*b; coord[2] = k*c
                        coords.append(origin+coord)
                    elif lattice == 'bcc':
                        coord[0] = i*a;
                        coord[1] = j*b
                        coord[2] = k*c
                        coords.append(origin+coord)

                        coord[0] = (i+0.5)*a
                        coord[1] = (j+0.5)*b
                        coord[2] = (k+0.5)*c
                        coords.append(origin+coord)
                    elif lattice == 'fcc':
                        coord[0] = i*a;
                        coord[1] = j*b
                        coord[2] = k*c
                        coords.append(origin+coord)

                        coord[0] = (i+0.5)*a
                        coord[1] = (j+0.5)*b
                        coord[2] = k*c
                        coords.append(origin+coord)

                        coord[0] = (i+0.5)*a
                        coord[1] = j*b
                        coord[2] = (k+0.5)*c
                        coords.append(origin+coord)

                        coord[0] = i*a
                        coord[1] = (j+0.5)*b
                        coord[2] = (k+0.5)*c
                        coords.append(origin+coord)

        #Check if boundary conditions are satisfied
        coords_final = []
        for each in coords:
            add = True
            for i in range(3):
                if (boundary[i] in 'cn') and (each[i] > hi[i]):
                    add = False; break
            if add: 
                coords_final.append(each)

        #Number of lattice points
        ns = len(coords_final)
        if ns >= num_sites:
            break
        else:
            #Updated lattice parameters
            m = np.argmax(l); p = [0, 1, 2]; p.remove(m)
            nuc[m] += 1
            if l[m] == l[p[0]]: nuc[p[0]] += 1
            if l[m] == l[p[1]]: nuc[p[1]] += 1
            lpar_[:] = (np.asarray(hi) - np.asarray(lo))/nuc

    print(f"Lattice parameters = ({lpar_[0]}, {lpar_[1]}, {lpar_[2]})")
    return np.array(coords_final)


def read_cfg(fn):
    """
    Extract box vectors from a cfg file.

    """
    length_scale = 1.0 #in Angstrom
    rate_scale = 1.0   #in ns^-1
    H0 = np.zeros((3,3)); transform = np.identity(3); eta = np.zeros((3,3))
    atom_section = False
    with open(fn, 'r') as fh:
        while (linein := fh.readline()):
            #Remove blank and comment lines
            line = linein.strip('\n ')
            if len(line)==0 or line.startswith('#'):
                continue
            if line.startswith('Number of particles') :
                N = int(line.split()[4])
                coords = np.empty((N,3)); atom_names = []; iatm = 0
            elif line.startswith('A') and len(words := line.split()) > 2:
                length_scale = float(words[2])
            elif line.startswith('R') and len(words := line.split()) > 2:
                rate_scale = float(words[2])
            elif line.startswith('H0('):
                words = line.split()
                i = int(words[0][3]) - 1; j = int(words[0][5]) - 1
                H0[i,j] = float(words[2])*length_scale
            elif line.startswith('Transform('):
                words = line.split()
                i = int(words[0][10]) - 1; j = int(words[0][12]) - 1
                transform[i,j] = float(words[2])
            elif line.startswith('eta('):
                words = line.split()
                i = int(words[0][4]) - 1; j = int(words[0][6]) - 1
                eta[i,j] = float(words[2])
            elif line.startswith('.NO_VELOCITY.'):
                velocity_data = False
            elif line.startswith('entry_count'):
                entry_count = int(line.split()[2])
                if velocity_data:
                    num_aux = entry_count - 6
                else:
                    num_aux = entry_count - 3
            elif line.startswith('auxiliary'):
                continue
            else:
                #Atom data records
                if not atom_section:
                    atom_section = True
                    H = H0 @ transform
                words = line.split(); nwords = len(words)
                if nwords == 1 :
                    if words[0].isnumeric():
                        mass = float(words[0])
                    else:
                        nam = words[0]
                else:
                    atom_names.append(nam) 
                    s = [float(x) for x in words[0:3]]
                    coords[iatm,0] = s[0]*H[0,0] + s[1]*H[1,0] + s[2]*H[2,0]
                    coords[iatm,1] = s[0]*H[0,1] + s[1]*H[1,1] + s[2]*H[2,1]
                    coords[iatm,2] = s[0]*H[0,2] + s[1]*H[1,2] + s[2]*H[2,2]
                    iatm += 1

    return H0, atom_names, coords



def make_npl_xtal(lx, ly, lz, lattice, species_names, a, c=None, 
                  orient=['1_0_0', '0_1_0', '0_0_1'], unit=['nm', 'nm', 'nm'],
                  orthogonalize=True, mlcp=[], boundary='ppp',
                  atomsk_path='atomsk'):

    if unit[0] not in ['nm', 'LU']:
        raise ValueError("unit[0] must be {'nm' | 'LU'}.")
    if unit[1] not in ['nm', 'LU']:
        raise ValueError("unit[1] must be {'nm' | 'LU'}.")
    if unit[2] not in ['nm', 'LU', 'ML']:
        raise ValueError("unit[2] must be {'nm' | 'LU' | 'ML'}.")
    for i in range(3):
        if boundary[i] not in ['p', 'c', 'n']:
            raise ValueError(f"boundary[{i}] must be {'p' | 'c' | 'n'}.")
    #Create unit cell
    fn_uc = Path('tmp.ucell.cfg')
    fn_sc = Path('tmp.scell.cfg')
    fn_uc.unlink(missing_ok=True) 
    fn_sc.unlink(missing_ok=True) 

    args = [f"{atomsk_path}"]
    args.extend(f"--create {lattice} {a*10}".split()) #Convert `a` to angstrom
    if c:
        args.append(f"{c*10}") #Convert `c` to angstrom
    for each in species_names:
        args.append(f"{each}")
    if orient:
        args.extend(['orient'] + orient)
    if orthogonalize:
        args.append('-orthogonal-cell')
    args.append('-ow') #Overwrite preexisting output file
    args.append(fn_uc) #Output file name

    cp = subprocess.run(args, capture_output=True, text=True,
                        shell=False, check=True)
    print('Creating unit cell ...')
    print(cp.stdout)
    if 'ERROR' in cp.stdout:
        raise SystemExit()

    H0, _, uc_coords = read_cfg(fn_uc)
    Hx = H0[0,0]; Hy = H0[1,1]; Hz = H0[2,2]

    if unit[0] == 'LU':
        if boundary[0] == 'p' or boundary[0] == 'n':
            Nx = int(math.trunc(lx)); cpxl = 0.0; cpxh = Nx*Hx
        elif boundary[0] == 'c':
            Nx = int(math.trunc(lx)) + 1; cpxl = 0.0; cpxh = (Nx-1)*Hx
    elif unit[0] == 'nm':
        if boundary[0] == 'p':
            Nx = int(round(10*lx/Hx)); cpxl = 0.0; cpxh = Nx*Hx
        elif boundary[0] == 'c':
            Nx = int(round(10*lx/Hx)) + 1; cpxl = 0.0; cpxh = (Nx-1)*Hx
        elif boundary[0] == 'n':
            Nx = int(round(10*lx/Hx)); cpxl = 0.0; cpxh = 10*lx
    else:
        raise ValueError("unit[0] must be {'nm' | 'LU'}.")

    if unit[1] == 'LU':
        if boundary[1] == 'p' or boundary[1] == 'n':
            Ny = int(math.trunc(ly)); cpyl = 0.0; cpyh = Ny*Hy
        elif boundary[1] == 'c':
            Ny = int(math.trunc(ly)) + 1; cpyl = 0.0; cpyh = (Ny-1)*Hy
    elif unit[1] == 'nm':
        if boundary[1] == 'p':
            Ny = int(round(10*ly/Hy)); cpyl = 0.0; cpyh = Ny*Hy
        elif boundary[1] == 'c':
            Ny = int(round(10*ly/Hy)) + 1; cpyl = 0.0; cpyh = (Ny-1)*Hy
        elif boundary[1] == 'n':
            Ny = int(round(10*ly/Hy)); cpyl = 0.0; cpyh = 10*ly
    else:
        raise ValueError("unit[1] must be {'nm' | 'LU'}.")

    if unit[2] == 'LU':
        if boundary[2] == 'p' or boundary[2] == 'n':
            Nz = int(math.trunc(lz)); cpzl = 0.0; cpzh = Nz*Hz
        elif boundary[2] == 'c':
            Nz = int(math.trunc(lz)) + 1; cpzl = 0.0; cpzh = (Nz-1)*Hz
    elif unit[2] == 'nm':
        if boundary[2] == 'p':
            Nz = int(round(10*lz/Hz)); cpzl = 0.0; cpzh = Nz*Hz
        elif boundary[2] == 'c':
            Nz = int(round(10*lz/Hz)) + 1; cpzl = 0.0; cpzh = (Nz-1)*Hz
        elif boundary[2] == 'n':
            Nz = int(round(10*lz/Hz)); cpzl = 0.0; cpzh = 10*lz
    elif unit[2] == 'ML':
        ncp = int(math.trunc(lz)) + 1 #Number of cutting planes
        print(f"ncp = {ncp}")
        Nz = int(math.ceil(ncp/len(mlcp)))
        print(f"Nz = {Nz}")
        cpzl = mlcp[0]*Hz
        cpzh = (Nz - 1 + mlcp[ncp%len(mlcp)-1])*Hz
        print(f"cpzl = {cpzl}, cpzh = {cpzh}")
    else:
        raise ValueError("unit[2] must be {'nm' | 'LU' | 'ML'}.")

    #Replicate to create supercell and cut if necessary
    args = [f"{atomsk_path}"]
    args.append(fn_uc) 
    args.extend(f"-dup {Nx} {Ny} {Nz}".split())
    args.extend(f"-cut above {cpxh} x".split())     
    args.extend(f"-cut above {cpyh} y".split())     
    args.extend(f"-cut above {cpzh} z".split())     
    args.extend(f"-cut below {cpzl} z".split())     
    args.extend(f"-shift 0.0 0.0 -{cpzl}".split()) 
    args.extend(f"-cell set {cpzh-cpzl} H3".split())

    args.extend('-sort s pack'.split()) #Pack similar atoms together
    args.append('-ow')  #Overwrite preexisting output file
    args.append(fn_sc) #Output file name

    cp = subprocess.run(args, capture_output=True, text=True, shell=False,
                        check=False)
    print('Creating super cell ...')
    print(cp.stdout)
    if 'ERROR' in cp.stdout:
        raise SystemExit()

    H0, atom_names, coords = read_cfg(fn_sc)
    H0 /= 10; coords /= 10 #Converting from angstrom to nm
    return H0, atom_names, coords


