# -*- coding: utf-8 -*-

import os
import numpy as np
from ctypes import cdll, windll, wintypes
from ctypes import c_double, c_int, c_void_p

kernel32 = windll.LoadLibrary('kernel32')
kernel32.FreeLibrary.argtypes = [wintypes.HMODULE]

pkg_path = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join(pkg_path, "bin", "smatrix.dll")


class Multilayer(object):

    """Multilayer made of a polar material on top of a
       distributed Bragg reflector.
    """

    def __init__(self, wvl, N_polar, L_polar, N_A, N_B, L_A, L_B, num_uc, N_inc, N_sub):
        """Initialize structure parameters.

        Input:
        wvl     : (numpy array) Wavelength of incident beam (m).
        N_polar : (numpy array) Complex refractive index of polar material.
        N_A     : (real number) Refractive index of first layer in unit cell.
        N_B     : (real number) Refractive index of second layer in unit cell.
        L_A     : (real number) Thickness of first layer in unit cell (m).
        L_B     : (real number) Thickness of second layer in unit cell (m).
        num_uc  : (integer)     Number of unit cells.
        N_inc   : (real number) Refractive index of medium of incidence.
        N_sub   : (real number) Refractive index of substrate.
        """
        self.wvl = wvl.astype(np.double)
        self.N_polar = N_polar.astype(np.complex)
        self.L_polar = c_double(L_polar)
        self.N_A = c_double(N_A)
        self.N_B = c_double(N_B)
        self.L_A = c_double(L_A)
        self.L_B = c_double(L_B)
        self.num_uc = c_int(num_uc)
        self.N_inc = c_double(N_inc)
        self.N_sub = c_double(N_sub)

        self.num_wvl = c_int(len(self.wvl))

        # Create pointers to pass arrays to cpp library.
        self.wvl_p = c_void_p(self.wvl.ctypes.data)
        self.N_polar_p = c_void_p(self.N_polar.ctypes.data)


class Fields(object):

    """Fields properties"""

    def __init__(self, pTE, pTM, theta, phi):
        """Polarization vector components and angles of incidence.

        Input:
        pTE     : (real number) TE component of incident electric field.
        pTM     : (real number) TM component of incident electric field.
        theta   : (real number) Angle of incidence with respect to normal (degrees).
        phi     : (real number) Angle of incidence on surface plane (degrees).
        """
        self.pTE = c_double(pTE)
        self.pTM = c_double(pTM)
        self.theta = c_double(theta)
        self.phi = c_double(phi)


class ScatteringMatrix(object):

    """Scattering matrices for each layer in multilayer.
    """

    def __init__(self, multilayer, fields):
        """Define multilayer and fields properties."""
        self.ml = multilayer
        self.fd = fields

    def solve(self, ):
        """Solve scattering matrix problem to get reflection and
           transmission coefficients."""
        self.R = np.zeros(self.ml.num_wvl.value, dtype=np.double)
        self.T = np.zeros(self.ml.num_wvl.value, dtype=np.double)
        self.R_p = c_void_p(self.R.ctypes.data)
        self.T_p = c_void_p(self.T.ctypes.data)

        # Connect to cpp library.
        self.lib = cdll.LoadLibrary(lib_path)

        self.lib.solve(self.fd.pTE, self.fd.pTM, self.fd.theta, self.fd.phi,
                       self.ml.wvl_p, self.ml.N_polar_p, self.ml.num_wvl,
                       self.ml.L_polar, self.ml.N_A, self.ml.N_B, self.ml.L_A,
                       self.ml.L_B, self.ml.num_uc, self.ml.N_inc, self.ml.N_sub,
                       self.R_p, self.T_p)

        # Disconnect from cpp library.
        handle = self.lib._handle
        kernel32.FreeLibrary(handle)
