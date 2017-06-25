# -*- coding: utf-8 -*-

import pkg_resources
import numpy as np
import scipy.constants as cst
from scipy.interpolate import interp1d
from io import BytesIO


#---------------------------------------------------------#
# SiC


def diel_SiC(w):
    """Dielectric function of SiC"""
    eps_inf = 6.7
    w_LO = 1.8253e14  # rad/s
    w_TO = 1.4937e14  # rad/s
    G = 8.966e11  # s^-1

    num = w**2 - w_LO**2 + 1j*G*w
    den = w**2 - w_TO**2 + 1j*G*w
    return eps_inf*(num/den)


#---------------------------------------------------------#
# SiO2


pkg_name = __package__

n_SiO2_pth = "/".join(("data", "n_SiO2_Palik.dat"))
rsc_n_SiO2 = pkg_resources.resource_string(pkg_name, n_SiO2_pth)
data_n_SiO2 = np.genfromtxt(BytesIO(rsc_n_SiO2), delimiter=",")
wvl_n_SiO2 = 1e-6*data_n_SiO2[:, 0]  # m
omeg_n_SiO2 = 2*cst.pi*cst.c/wvl_n_SiO2  # rad/s
n_SiO2_Palik = data_n_SiO2[:, 1]

k_SiO2_pth = "/".join(("data", "k_SiO2_Palik.dat"))
rsc_k_SiO2 = pkg_resources.resource_string(pkg_name, k_SiO2_pth)
data_k_SiO2 = np.genfromtxt(BytesIO(rsc_k_SiO2), delimiter=",")
wvl_k_SiO2 = 1e-6*data_k_SiO2[:, 0]  # m
omeg_k_SiO2 = 2*cst.pi*cst.c/wvl_k_SiO2  # rad/s
k_SiO2_Palik = data_k_SiO2[:, 1]


def diel_SiO2(w):
    """Dielectric function of SiO2"""

    # Interpolate n&k with cubic spline
    n_SiO2_itp_func = interp1d(omeg_n_SiO2, n_SiO2_Palik, kind="cubic")
    k_SiO2_itp_func = interp1d(omeg_k_SiO2, k_SiO2_Palik, kind="cubic")

    return (n_SiO2_itp_func(w) + 1j*k_SiO2_itp_func(w))**2

#---------------------------------------------------------#
# Vacuum


def diel_vac(w):
    """Dielectric function of vacuum"""
    return np.ones(len(w))
