# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.interpolate import interp1d
from ctypes import cdll, windll, wintypes
from ctypes import c_double, c_int, c_void_p

kernel32 = windll.LoadLibrary('kernel32')
kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]

pkg_path = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(pkg_path, "bin", "smatrix.dll")


class SimpleLayer(object):
    """Single layer of a simple material with no dispersion"""

    def __init__(self, n, d):
        """
        Input:
        n : Refractive index
        d : Thickness of layer (m)
        """
        self.n = n

        self.N = np.array(self.n, dtype=np.complex)
        self.d = c_double(d)

    def generate_N_for_wvl(self, wvl):
        """Generate refractive index array for new wavelength array"""
        self.wvl = wvl.astype(np.double)
        self.N = self.n*np.ones(len(self.wvl), dtype=np.complex)


class PolarLayer(object):
    """Single layer of a polar material with dispersion relation"""

    def __init__(self, wvl, N, d):
        """
        Input:
        wvl : Wavelengths array for dispersion relation
        N   : Complex refractive index array for dispersion relation
        d   : Thickness of layer (m)
        """
        self.unit_cell = []           # List of layers in unit cell

        self.wvl_disp = wvl.astype(np.double)
        self.N_disp = N.astype(np.complex)

        self.wvl = self.wvl_disp
        self.N = self.N_disp
        self.d = c_double(d)
        self.num_wvl = c_int(len(self.wvl))

    def generate_N_for_wvl(self, wvl):
        """Interpolate refractive index for new wavelength array using cubic splines"""
        n_interp = interp1d(self.wvl_disp, self.N_disp.real, kind="cubic")
        k_interp = interp1d(self.wvl_disp, self.N_disp.imag, kind="cubic")

        self.wvl = wvl.astype(np.double)
        self.N = n_interp(self.wvl) + 1j*k_interp(self.wvl)

        # TODO : Raise error of interpolation range is larger than dispersion data


class MultiLayer(object):

    """Multilayer made of a polar or simple material layers
    """

    def __init__(self):

        self.unit_cell = []           # List of layers in unit cell
        self.num_uc = c_int(1)        # Number of repetitions of unit cell

    def add_to_unit_cell(self, *layers):
        """Add a layer to unit cell
        """
        for layer in layers:
            self.unit_cell.append(layer)

    def clear_unit_cell(self):
        """Remove all layers from unit cell
        """
        del self.unit_cell[:]

    def number_repetitions(self, num):
        """Define number of repetitons of unit cell
        """
        self.num_uc.value = num


class Fields(object):

    """Properties of incident EM fields"""

    def __init__(self, wvl, pTE, pTM, theta, phi):
        """Polarization vector components and angles of incidence

        Input:
        wvl     : Wavelength array of incident fields (m)
        pTE     : TE component of incident electric field
        pTM     : TM component of incident electric field
        theta   : Angle of incidence with respect to normal (degrees)
        phi     : Angle of incidence on surface plane (degrees)
        """
        self.wvl = wvl.astype(np.double)
        self.pTE = c_double(pTE)
        self.pTM = c_double(pTM)
        self.theta = c_double(theta)
        self.phi = c_double(phi)


class ScatteringMatrix(object):

    """Scattering matrix for multilayer
    """

    def __init__(self, multilayer, fields, N_inc, N_sub):
        """Define multilayer and fields properties"""
        self.ml = multilayer
        self.fd = fields

        self.N_inc = c_double(N_inc)  # Refractive index of medium of incidence
        self.N_sub = c_double(N_sub)  # Refractive index of substrate

        # Make a list of all single layers
        self.single_layers = self.find_single_layers(self.ml)

        # Find most effective layer order for computation
        #self.steps = []  # Each element is a step of computation
        #self.steps.append(self.single_layers)  # First step is the single layers
        #self.uc_hierarchy = self.order_layers(self.ml, self.single_layers)
        self.uc = self.find_unit_cells(self.ml)

        # Create pointers to pass arrays to cpp library
        #self.wvl_p = c_void_p(self.wvl.ctypes.data)
        #self.N_polar_p = c_void_p(self.N_polar.ctypes.data)

    def find_single_layers(self, multilayer):
        """Find all single layers in multilayer"""
        single_layers = []
        for layer in multilayer.unit_cell:
            name = type(layer).__name__
            if name == "SimpleLayer" or name == "PolarLayer":
                if layer not in single_layers:
                    single_layers.append(layer)
            else:
                single_layers.extend(self.find_single_layers(layer))
        return single_layers

    def find_unit_cells(self, multilayer):
        """Find set of all unit cells to compute."""
        unit_cells = [multilayer]
        for layer in multilayer.unit_cell:
            if layer not in unit_cells:
                unit_cells.append(unit_cells)
            if type(layer).__name__ == "MultiLayer":
                unit_cells.extend(self.find_unit_cells(layer))
        return unit_cells

#    def find_matrices(self, multilayer):
#        """Find order in which to compute matrices."""
#        matrices = []
#        for layer in multilayer.unit_cell:
#            if layer not in matrices:
#                matrices.append(matrices)
#            else:
#                matrices.extend(self.find_matrices(layer))
#        return matrices


#    def order_layers(self, multilayer, single_layers):
#        """Find hierarchy of unit cells.
#        """
#        hierarchy = []
#        for layer in multilayer.unit_cell:
#            name = type(layer).__name__
#            if name == "MultiLayer":
#                hierarchy.append(self.order_layers(layer, single_layers))
#            else:
#                hierarchy.append(single_layers.index(layer))
#        return hierarchy

    def solve(self):
        """Solve scattering matrix problem to get reflection and
           transmission coefficients of multilayer
        """
        self.R = np.zeros(self.ml.num_wvl.value, dtype=np.double)
        self.T = np.zeros(self.ml.num_wvl.value, dtype=np.double)
        self.R_p = c_void_p(self.R.ctypes.data)
        self.T_p = c_void_p(self.T.ctypes.data)

        # Connect to cpp library
        lib = cdll.LoadLibrary(lib_path)

        lib.solve(self.fd.pTE, self.fd.pTM, self.fd.theta, self.fd.phi,
                  self.ml.wvl_p, self.ml.N_polar_p, self.ml.num_wvl,
                  self.ml.L_polar, self.ml.N_A, self.ml.N_B, self.ml.L_A,
                  self.ml.L_B, self.ml.num_uc, self.ml.N_inc, self.ml.N_sub,
                  self.R_p, self.T_p)

        # Disconnect from cpp library
        handle = lib._handle
        kernel32.FreeLibrary(handle)
