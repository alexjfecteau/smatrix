# -*- coding: utf-8 -*-

import os
import numpy as np
from smatrix import solver
from ctypes import cdll, windll, wintypes
from ctypes import c_double, c_int, c_void_p, POINTER
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


class Fields:

    """Properties of EM fields"""

    def __init__(self, wvl, pTE, pTM, theta, phi):
        """
        Input:
        wvl     : Wavelength array of incident fields (m)
        pTE     : TE component of incident electric field
        pTM     : TM component of incident electric field
        theta   : Angle of incidence with respect to normal (degrees)
        phi     : Angle of incidence on surface plane (degrees)
        """
        self.wvl = wvl.astype(np.double)
        self.num_wvl = c_int(len(self.wvl))

        self.pTE = c_double(pTE)
        self.pTM = c_double(pTM)
        self.theta = c_double(np.pi*theta/180.)
        self.phi = c_double(np.pi*phi/180.)

        lib.NewFields.argtypes = [c_double, c_double, c_double, c_double, np.ctypeslib.ndpointer(np.double, ndim=1, flags="C"), c_int]
        lib.NewFields.restype = c_void_p
        self.c_fields = c_void_p(lib.NewFields(self.pTE, self.pTM, self.theta, self.phi, self.wvl, self.num_wvl))


class SemiInfMedNoDisp:
    """Semi-infinite medium with no dispersion"""

    def __init__(self, n, fields):
        """
        Input:
        n      : Refractive index
        fields : Incident electric field object
        """
        self.wvl = fields.wvl
        self.N = n*np.ones(len(self.wvl), dtype=np.complex)
        self.num_wvl = c_int(len(self.wvl))

        lib.NewSemiInfMed.argtypes = [np.ctypeslib.ndpointer(np.complex, ndim=1, flags="C"), c_int]
        lib.NewSemiInfMed.restype = c_void_p
        self.c_med = c_void_p(lib.NewSemiInfMed(self.N, self.num_wvl))


class SemiInfMedDisp:
    """Semi-infinite medium with dispersion"""

    def __init__(self, N_func, fields):
        """
        Input:
        N_func   : Complex refractive index function
        fields   : Incident electric field object
        """
        self.wvl = fields.wvl
        self.N = N_func(self.wvl).astype(np.complex)

        self.num_wvl = c_int(len(self.wvl))

        lib.NewSemiInfMed.argtypes = [np.ctypeslib.ndpointer(np.complex, ndim=1, flags="C"), c_int]
        lib.NewSemiInfMed.restype = c_void_p
        self.c_med = c_void_p(lib.NewSemiInfMed(self.N, self.num_wvl))


class SingleLayerNoDisp:
    """Single layer of a simple material with no dispersion"""

    def __init__(self, n, d, fields):
        """
        Input:
        n      : Refractive index
        d      : Thickness of layer (m)
        fields : Incident electric field object
        """
        self.wvl = fields.wvl
        self.N = n*np.ones(len(self.wvl), dtype=np.complex)
        self.d = c_double(d)

        self.num_wvl = c_int(len(self.wvl))
        self.num_z = 0

        lib.NewSingleLayer.argtypes = [np.ctypeslib.ndpointer(np.complex, ndim=1, flags="C"), c_int, c_double]
        lib.NewSingleLayer.restype = c_void_p
        self.c_layer = c_void_p(lib.NewSingleLayer(self.N, self.num_wvl, self.d))


class SingleLayerDisp:
    """Single layer of a polar material with dispersion relation"""

    def __init__(self, N_func, d, fields):
        """
        Input:
        N_func   : Complex refractive index function
        d        : Thickness of layer (m)
        fields   : Incident electric field object
        """
        self.wvl = fields.wvl
        self.N = N_func(self.wvl).astype(np.complex)
        self.d = c_double(d)

        self.num_wvl = c_int(len(self.wvl))
        self.num_z = 0

        lib.NewSingleLayer.argtypes = [np.ctypeslib.ndpointer(np.complex, ndim=1, flags="C"), c_int, c_double]
        lib.NewSingleLayer.restype = c_void_p
        self.c_layer = c_void_p(lib.NewSingleLayer(self.N, self.num_wvl, self.d))


class MultiLayer:

    """Multilayer made of single layers or other multilayers
    """

    def __init__(self, *layers):

        self.unit_cell = []      # List of layers in unit cell
        self.num_uc = c_int(1)   # Number of repetitions of unit cell
        self.num_z = 0           # Number of positions inside multilayer

        lib.NewMultiLayer.restype = c_void_p
        self.c_layer = c_void_p(lib.NewMultiLayer())
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
        """Define number of repetitions of unit cell
        """
        self.num_uc.value = num
        lib.SetNumRepetitions(self.c_layer, num)

    def get_thickness(self):
        """Compute total thickness of multilayer
        """
        self.thk = c_double()
        lib.GetThkLayer(byref(self.thk))

    def get_layer_positions(self):
        """Compute position of each layers in the multilayer
        """
        return solver.layer_positions(self)


class ScatteringMatrix:

    """Scattering matrix for multilayer
    """

    def __init__(self, multilayer, fields, inc_med, sub_med):
        """Define multilayer and fields properties"""
        self.ml = multilayer
        self.fd = fields
        self.incm = inc_med
        self.subm = sub_med
        self.wvl = self.fd.wvl

        self.r_TE = np.empty(self.fd.num_wvl.value, dtype=np.complex)
        self.t_TE = np.empty(self.fd.num_wvl.value, dtype=np.complex)
        self.r_TM = np.empty(self.fd.num_wvl.value, dtype=np.complex)
        self.t_TM = np.empty(self.fd.num_wvl.value, dtype=np.complex)
        self.R = np.empty(self.fd.num_wvl.value, dtype=np.double)
        self.T = np.empty(self.fd.num_wvl.value, dtype=np.double)

        lib.NewScatteringMatrix.argtypes = [c_void_p, c_void_p, c_void_p, c_void_p]
        lib.NewScatteringMatrix.restype = c_void_p
        self.sm = c_void_p(lib.NewScatteringMatrix(self.ml.c_layer, self.fd.c_fields, self.incm.c_med, self.subm.c_med))

    def solve(self):
        """Solve scattering matrix problem to get reflection and
           transmission coefficients of multilayer
        """
        lib.ComputeRT.argtypes = [c_void_p,
                                  np.ctypeslib.ndpointer(np.complex, ndim=1, flags="C"),
                                  np.ctypeslib.ndpointer(np.complex, ndim=1, flags="C"),
                                  np.ctypeslib.ndpointer(np.complex, ndim=1, flags="C"),
                                  np.ctypeslib.ndpointer(np.complex, ndim=1, flags="C"),
                                  np.ctypeslib.ndpointer(np.double, ndim=1, flags="C"),
                                  np.ctypeslib.ndpointer(np.double, ndim=1, flags="C")]

        lib.ComputeRT(self.sm, self.r_TE, self.t_TE, self.r_TM, self.t_TM, self.R, self.T)

    def solve_py(self):
        return solver.computeRT(self.ml, self.fd, self.incm, self.subm)

    def computeE2(self, wvl_id, z_res=5e-8):
        """Compute squared norm of electric field inside multilayer"""
        return solver.computeE2(self.ml, self.fd, self.incm, self.subm, wvl_id, z_res)
