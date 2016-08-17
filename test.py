# -*- coding: utf-8 -*-

import numpy as np
from ctypes import cdll
from ctypes import c_double, c_int, c_void_p

lib = cdll.LoadLibrary('./bin/smatrix.dll')

pTE = c_double(1.0)
pTM = c_double(0.0)
theta = c_double(30.0)
phi = c_double(0.0)

wvl = np.array([4.0e-6, 10e-6, 15e-6], dtype=np.double)
N_polar = np.array([1.0+0.1j, 1.0+0.1j, 1.0+0.1j], dtype=np.complex)
wvl_p = c_void_p(wvl.ctypes.data)
N_polar_p = c_void_p(N_polar.ctypes.data)
num_wvl = c_int(len(wvl))

L_polar = c_double(1e-6)
NA = c_double(2.50)
NB = c_double(1.50)
LA = c_double(0.920e-6)
LB = c_double(1.53e-6)
num_uc = c_int(10)
N_inc = c_double(1.0)
N_sub = c_double(3.4)

R = np.zeros(6, dtype=np.double)
T = np.zeros(6, dtype=np.double)
R_p = c_void_p(R.ctypes.data)
T_p = c_void_p(T.ctypes.data)

lib.solve(pTE, pTM, theta, phi, wvl_p, N_polar_p, num_wvl, L_polar, NA, NB, LA, LB, num_uc, N_inc, N_sub, R_p, T_p)

print wvl
print R
