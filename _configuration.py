#!/usr/bin/env python

import copy
import math
import numpy as np
from _topology import Topology
from _geom_utils import fix_axis_angle, rotate_vector_axis_angle


class Configuration(Topology):
    def __init__(self):
        super().__init__()
        self.simbox = np.zeros((3,2))
        self.tilt_factors = np.zeros((3,))


    def clear(self):
        '''
        Clears all data.

        '''
        self.simbox[:,:] = 0.0
        self.tilt_factors[:] = 0.0
        super().clear()


    def add_simbox(self, xlo, xhi, ylo, yhi, zlo, zhi):
        '''
        Adds a simulation box. x-dir is toward the right, y is up, and z points
        out of the screen.

        '''
        self.simbox = np.array([[xlo,xhi],[ylo,yhi],[zlo,zhi]])


    def fit_simbox(self, sep=0.0):
        '''
        Changes the simulation box dimensions to the extent of the atom
        positions, i.e., it is a minimal bounding box for the atoms.
        Does not change boundary conditions as set before.
        `self.add_simbox` must be called and all atoms already added 
        before calling this function.
        Useful for setting the box size when it is hard to guess or calculate
        box size, e.g. for single molecules.

        '''
        rlo, rhi = self.get_bbox()
        #Update box dimensions
        self.simbox[:,0] = rlo - sep
        self.simbox[:,1] = rhi + sep


    def get_bbox(self, atm_ids=None):
        '''
        Returns an axis-aligned bounding box around a set of atoms.

        Parameters
        ----------
        atm_ids : iterable | None
            Iterable object with atom ids. If None, all atoms are included.

        Returns
        -------
        rlo, rhi : Tuple of ndarrays
            Lower and upper bounds of the bounding box.

        '''
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids
        rlo = np.array([np.inf, np.inf, np.inf])
        rhi = np.array([-np.inf, -np.inf, -np.inf])
        for iatm in aids:
            coords = self.atoms[iatm].coords
            rlo = np.minimum(rlo, coords)
            rhi = np.maximum(rhi, coords)
        return (rlo, rhi)


    def translate(self, r, only_atoms=False, atm_ids=None, out_coords=None):
        '''
        Translates a group of atoms by a vector `r`. If all atoms are included,
        the simulation box may be translated as well.

        Parameters
        ----------
        r : ndarray
            Atoms will be translated by the vector `r`, i.e. `r` will be added
            to the coordinates of all atoms with ids in `atm_ids`.
        only_atoms : bool
            Whether to translate the simulation box by `r`.
        atm_ids : iterable|None
            Ids of atoms to translate. If `None` all atoms will be translated.
        out_coords : ndarray
            If not `None`, the translated coordinates will be placed here. The
            original coordinates will be be changed.

        Returns
        -------
        `None`

        '''
        dr = np.copy(r)
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids

        if out_coords is None:
            for i in aids:
                self.atoms[i].coords += dr
            if not only_atoms:
                self.simbox[:,0] += dr
                self.simbox[:,1] += dr
        else:
            nr, nc = out_coords.shape
            assert nr >= len(aids)
            assert nc >= 3
            for i,k in enumerate(aids):
                out_coords[i,:] = self.atoms[k].coords + dr


    def translate_atom(self, atm_id, pos, atm_ids=None, out_coords=None):
        """
        Translate a group of atoms such that an atom of that group is brought
        to a given point. Note that the simulation box remains unchanged. 

        Parameters
        ----------
        atm_id : int
            Atom id 
        pos : (3,) ndarray of floats
            Point where atom `atm_id` should be moved.
        atm_ids : iterable
            Ids of the atoms to translate. If `None` all atoms will be translated.
        out_coords : (n,3) ndarray of floats
            If not `None`, the atom coordinates after translation will be placed
            here. The original coordinates will remain unchanged.

        Returns
        -------
        `None`

        """
        assert atm_id in self.atoms.keys()
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids
        dr = pos - self.atoms[atm_id].coords
        self.translate(dr, only_atoms=True, atm_ids=atm_ids,
                       out_coords=out_coords)


    def rotate(self, angle, axis, pivot, atm_ids=None, out_coords=None):
        """
        Rotates a group of atoms by an angle about an axis.

        Parameters
        ----------
        angle : float
            Angle of rotation (radian) 
        axis : (3,) ndarray of floats
            Axis of rotation (need not be a unit vector)
        pivot : (3,) ndarray of floats
            Point in space about which to rotate
        atm_ids : iterable
            Ids of the atoms to rotate. If `None` all atoms will be rotated.
        out_coords : (n,3) ndarray of floats
            If not `None`, the atom coordinates after rotation will be placed
            here. The original coordinates will remain unchanged.

        Returns
        -------
        `None`

        """
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids

        uhat, theta = fix_axis_angle(axis, angle, normalize=True)
        n = len(aids)
        coords = np.zeros((n,3))
        for i,k in enumerate(aids):
            coords[i,:] = self.atoms[k].coords - pivot

        if out_coords is None:
            rotated_coords = rotate_vector_axis_angle(coords, uhat, theta)
            for i,k in enumerate(aids):
                self.atoms[k].coords = rotated_coords[i,:] + pivot
        else:
            out_coords[:,:] = rotate_vector_axis_angle(coords, uhat, theta)
            for i in range(n):
                out_coords[i,:] += pivot


    def align(self, atm_1, atm_2, axis, atm_ids=None, out_coords=None):
        """
        Rotates a group of atoms so as to align the line through two atoms along
        a direction. The two atoms need not be bonded. The pivot point for
        rotation will be the position of `atm_1`.

        Parameters
        ----------
        atm_1 : int
            Atom id
        atm_2 : int
            Atom id
        axis : (3,) ndarray of floats
            Direction along which to align (need not be a unit vector)
        atm_ids : iterable
            Ids of the atoms to rotate. If `None` all atoms will be rotated.
        out_coords : (n,3) ndarray of floats
            If not `None`, the atom coordinates after rotation will be placed
            here. The original coordinates will remain unchanged.

        Returns
        -------
        `None`

        """
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids

        pivot = self.atoms[atm_1].coords
        phat = axis/np.linalg.norm(axis) #Direction unit vector
        v = self.atoms[atm_2].coords - self.atoms[atm_1].coords
        vhat = v/np.linalg.norm(v) #Line unit vector
        ctheta = np.dot(vhat, phat)
        if ctheta == 1:
            #Nothing to do
            return
        elif ctheta == -1:
            #Reflect
            for aid in aids:
                self.atoms[aid].coords = -(self.atoms[aid].coords - pivot)
        else:
            #Rotate
            theta = math.acos(ctheta)
            u = np.cross(vhat, phat)
            umag = np.linalg.norm(u)
            uhat = u/umag
            self.rotate(theta, uhat, pivot, atm_ids, out_coords)


    def get_barycenter(self, atm_ids=None):
        '''
        Returns the barycenter for all (or a subset) of atoms.

        '''
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids
        bc = np.zeros((3,))
        for i in aids:
            bc += self.atoms[i].coords
        bc /= len(aids)
        return bc


    def get_com(self, atm_ids=None):
        '''
        Returns the center of mass for all (or a subset) of atoms.

        '''
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids
        com = np.zeros((3,)); totmass = 0.0
        for i in aids:
            mass = self.atoms[i].mass
            coords = self.atoms[i].coords
            com += (mass*coords); totmass += mass
        com /= totmass
        return com


    def get_gyration_radius(self, atm_ids=None):
        """
        Parameters
        ----------
        atm_ids : None | iterable of ints
            Atom ids. If ``None``, calculate for all atoms.

        Returns
        -------
        dict

        """
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids
        gyr_ten = np.zeros((3,3))
        c = self.get_com(aids); totmass = 0.0
        for i in aids:
            mass = self.atoms[at].mass
            totmass += mass
            r = self.atoms[i].coords - c
            gyr_ten[0,0] += (mass*r[0]*r[0]) 
            gyr_ten[0,1] += (mass*r[0]*r[1]) 
            gyr_ten[0,2] += (mass*r[0]*r[2]) 
            gyr_ten[1,1] += (mass*r[1]*r[1]) 
            gyr_ten[1,2] += (mass*r[1]*r[2]) 
            gyr_ten[2,2] += (mass*r[2]*r[2]) 
        gyr_ten /= totmass
        gyr_ten[1,0] = gyr_ten[0,1]
        gyr_ten[2,0] = gyr_ten[0,2]
        gyr_ten[2,1] = gyr_ten[1,2]
        rg = math.sqrt(gyr_ten[0,0]+gyr_ten[1,1]+gyr_ten[2,2])
        return {'rg': rg, 'rgxx': gyr_ten[0,0]**0.5, 'rgyy': gyr_ten[1,1]**0.5,
                'rgzz': gyr_ten[2,2]**0.5, 'tensor': gyr_ten}


    def get_total_charge(self, atm_ids=None):
        '''
        Returns the total charge for all (or a subset) of atoms.

        '''
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids
        charges = [self.atoms[k].charge for k in aids]
        return math.fsum(charges)


    def get_total_mass(self, atm_ids=None):
        '''
        Returns the total mass for all (or a subset) of atoms.

        '''
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids
        masses = [self.atoms[k].mass for k in aids]
        return math.fsum(masses)


    def apply_pbc_atoms(self, pos, directions='xyz'):
        '''
        Applies PBC along all directions to a set of atom coordinates.

        '''
        flag = np.array([1,1,1], dtype=np.int32)
        if 'x' not in directions: flag[0] = 0
        if 'y' not in directions: flag[1] = 0
        if 'z' not in directions: flag[2] = 0

        boxl = self.simbox[:,1] - self.simbox[:,0]
        coords = copy.deepcopy(pos)
        for i in range(coords.shape[0]):
            dr = coords[i,:] - self.simbox[:,0] #Origin at left lower back corner
            pb = np.floor(dr/boxl) #Index of the periodic box
            pb = np.where(flag==0, flag, pb)
            dr -= boxl*pb   # PBC
            coords[i,:] = self.simbox[:,0] + dr #Back to origin
        return coords



    def apply_pbc(self, directions='xyz', atm_ids=None, update_img_flag=False):
        '''
        Applies PBC along all directions.

        '''
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids

        flag = np.array([1,1,1], dtype=np.int32)
        if 'x' not in directions: flag[0] = 0
        if 'y' not in directions: flag[1] = 0
        if 'z' not in directions: flag[2] = 0

        boxl = self.simbox[:,1] - self.simbox[:,0]
        for i in aids:
            coords = self.atoms[i].coords
            #Shift origin to the left lower back corner
            dr = coords - self.simbox[:,0] 
            pb = np.floor(dr/boxl) #Index of the periodic box
            pb = np.where(flag==0, flag, pb)
            dr -= boxl*pb   # PBC
            coords[:] = self.simbox[:,0] + dr #Back to the origin
            if update_img_flag:
                self.atoms[i].img_flag[:] = pb


    def unwrap_pbc(self):
        '''
        Unwraps PBC along all directions by traversing all bonds. Note that only
        bonded atoms can be unwrapped (that too in the absence of rings). Image
        flags remain same as before unwrapping.

        '''
        if len(self.bonds) == 0:
            print("Cannot unwrap without any bonds. Returning ...")
            return None

        boxl = self.simbox[:,1] - self.simbox[:,0]
        for i in range(1, self.num_bonds+1):
            atm_i = self.bonds[i].ai
            atm_j = self.bonds[i].aj
            coords_i = self.atoms[atm_i].coords
            coords_j = self.atoms[atm_j].coords
            #Atom i has higher atm_id
            dr = coords_i - coords_j
            dr -= boxl*np.rint(dr/boxl)   # PBC
            coords_i[:] = coords_j + dr


    def adjust_charge(self):
        """
        Tweak charges to make the system electroneutral.

        """
        chge = self.get_total_charge()
        if chge != 0:
            aids_charged = []
            for i in range(1, self.num_atoms+1):
                q = self.atoms[i].charge
                if not math.isclose(q, 0.0):
                    aids_charged.append(i)
            na_charged = len(aids_charged)
            cpa = chge/na_charged
            for i in aids_charged:
                self.atoms[i].charge -= cpa
        print("Total charge after adjustment = %g"%self.get_total_charge())

