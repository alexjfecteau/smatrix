# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.interpolate import interp1d
from ctypes import cdll, windll, wintypes
from ctypes import c_double, c_int, c_void_p
from ctypes import byref

kernel32 = windll.LoadLibrary('kernel32')
kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]

# Find absolute path to cpp library
pkg_path = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(pkg_path, "bin", "smatrix.dll")


def open_session():
    """Connect to cpp library"""
    global lib
    lib = cdll.LoadLibrary(lib_path)


def close_session():
    """Disconnect from cpp library"""
    global lib
    handle = lib._handle
    kernel32.FreeLibrary(handle)
    del lib


class Fields(object):

    """Properties of EM fields"""

    def __init__(self, wvl, pTE, pTM, theta, phi, N_inc, N_sub):
        """
        Input:
        wvl     : Wavelength array of incident fields (m)
        pTE     : TE component of incident electric field
        pTM     : TM component of incident electric field
        theta   : Angle of incidence with respect to normal (degrees)
        phi     : Angle of incidence on surface plane (degrees)
        N_inc   : Refractive index of medium of incidence
        N_sub   : Refractive index of substrate
        """
        self.wvl = wvl.astype(np.double)
        self.wvl_p = c_void_p(self.wvl.ctypes.data)
        self.num_wvl = c_int(len(self.wvl))

        self.pTE = c_double(pTE)
        self.pTM = c_double(pTM)
        self.theta = c_double(theta)
        self.phi = c_double(phi)

        self.N_inc = c_double(N_inc)
        self.N_sub = c_double(N_sub)

        self.c_fields = lib.NewFields(self.pTE, self.pTM, self.theta, self.phi, self.wvl_p, self.num_wvl, self.N_inc, self.N_sub)


class SingleLayerNoDisp(object):
    """Single layer of a simple material with no dispersion"""

    def __init__(self, n, d, wvl):
        """
        Input:
        n   : Refractive index
        d   : Thickness of layer (m)
        wvl : Wavelength of incident field (m)
        """
        self.wvl = wvl.astype(np.double)
        self.N = n*np.ones(len(self.wvl), dtype=np.complex)
        self.d = c_double(d)
        self.num_wvl = c_int(len(self.wvl))
        self.num_z = 0

        self.c_layer = lib.NewSingleLayer(self.wvl.ctypes.data, self.N.ctypes.data, self.num_wvl, self.d)


class SingleLayerDisp(object):
    """Single layer of a polar material with dispersion relation"""

    def __init__(self, wvl_disp, N_disp, d, wvl):
        """
        Input:
        wvl_disp : Wavelengths array for dispersion relation
        N_disp   : Complex refractive index array for dispersion relation
        d        : Thickness of layer (m)
        """
        self.wvl_disp = wvl_disp.astype(np.double)
        self.N_disp = N_disp.astype(np.complex)

        self.wvl = wvl_disp
        self.N = self.N_disp
        self.generate_N_for_wvl(wvl)

        self.d = c_double(d)
        self.num_wvl = c_int(len(self.wvl))
        self.num_z = 0

        self.c_layer = lib.NewSingleLayer(self.wvl.ctypes.data, self.N.ctypes.data, self.num_wvl, self.d)

    def generate_N_for_wvl(self, wvl):
        """Interpolate refractive index for new wavelength array using cubic splines"""
        n_interp = interp1d(self.wvl_disp, self.N_disp.real, kind="cubic")
        k_interp = interp1d(self.wvl_disp, self.N_disp.imag, kind="cubic")

        self.wvl = wvl.astype(np.double)
        self.N = n_interp(wvl) + 1j*k_interp(wvl)

        # TODO : Raise error of interpolation range is larger than dispersion data


class MultiLayer(object):

    """Multilayer made of single layers or other multilayers
    """

    def __init__(self, *layers):

        self.unit_cell = []      # List of layers in unit cell
        self.num_uc = c_int(1)   # Number of repetitions of unit cell
        self.num_z = 0           # Number of positions inside multilayer

        self.c_layer = lib.NewMultiLayer()
        self.add_to_unit_cell(*layers)

    def add_to_unit_cell(self, *layers):
        """Add a layer to unit cell
        """
        for layer in layers:
            self.unit_cell.append(layer)
            lib.AddToUnitCell(self.c_layer, layer.c_layer)

    def clear_unit_cell(self):
        """Remove all layers from unit cell
        """
        del self.unit_cell[:]
        lib.ClearUnitCell(self.c_layer)

    def set_num_repetitions(self, num):
        """Define number of repetitons of unit cell
        """
        self.num_uc.value = num
        lib.SetNumRepetitions(self.c_layer, num)

    def get_thickness(self):
        """Compute total thickness of multilayer
        """
        self.thk = c_double()
        lib.GetThkLayer(byref(self.thk))


class ScatteringMatrix(object):

    """Scattering matrix for multilayer
    """

    def __init__(self, multilayer, fields):
        """Define multilayer and fields properties"""
        self.ml = multilayer
        self.fd = fields
        self.wvl = self.fd.wvl
        self.R = np.empty(self.fd.num_wvl.value, dtype=np.double)
        self.T = np.empty(self.fd.num_wvl.value, dtype=np.double)
        self.R_p = c_void_p(self.R.ctypes.data)
        self.T_p = c_void_p(self.T.ctypes.data)
        self.sm = lib.NewScatteringMatrix(self.ml.c_layer, self.fd.c_fields)

    def solve(self):
        """Solve scattering matrix problem to get reflection and
           transmission coefficients of multilayer
        """
        lib.ComputeRT(self.sm, self.R_p, self.T_p)

    def compute_E(self, wvl_index):
        """Compute electric field components and squared norm as a
           function of wavelength and depth in multilayer
        """
        # Try if solve() was apply. If not, run solve().
        self.E2 = np.empty(10, dtype=np.double)
        self.E2_p = c_void_p(self.E2.ctypes.data)
        lib.ComputeE(self.sm, wvl_index, self.E2_p)

        # Find number of positions in multilayer
