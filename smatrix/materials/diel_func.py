# -*- coding: utf-8 -*-

import pkg_resources
import numpy as np
import scipy.constants as cst
from scipy.interpolate import interp1d
from io import BytesIO

pkg_name = __package__

#---------------------------------------------------------#
# Vacuum


def diel_vac(wvl):
    """Dielectric function of vacuum"""
    return np.ones(len(wvl))


#---------------------------------------------------------#
# SiC


def diel_SiC(wvl):
    """Dielectric function of SiC"""

    w = 2*cst.pi*cst.c/wvl  # rad/s

    eps_inf = 6.7
    w_LO = 1.8253e14  # rad/s
    w_TO = 1.4937e14  # rad/s
    G = 8.966e11  # s^-1

    num = w**2 - w_LO**2 + 1j*G*w
    den = w**2 - w_TO**2 + 1j*G*w
    return eps_inf*(num/den)


#---------------------------------------------------------#
# AlN

def diel_AlN(wvl):
    """
    Dielectric function of AlN
    Lorentz model with parameters from Akasaki et al. (1967)
    """

    w = 2*cst.pi*cst.c/wvl  # rad/s

    eps_0 = 8.50
    eps_inf = 4.68
    w_TO = 1.25e14  # s^-1
    G = 0.01

    eps = eps_inf + (eps_0 - eps_inf)/(1. - (w/w_TO)**2 - 1j*G*(w/w_TO))
    return eps


#---------------------------------------------------------#
# SiO2


def diel_SiO2(wvl):
    """
    Dielectric function of SiO2
    Interpolated from Palik et al.
    """

    n_pth = "/".join(("data", "n_SiO2_Palik.dat"))
    rsc_n = pkg_resources.resource_string(pkg_name, n_pth)
    data_n = np.genfromtxt(BytesIO(rsc_n), delimiter=",")
    wvl_n_dat = 1e-6*data_n[:, 0]  # m
    n_dat = data_n[:, 1]

    k_pth = "/".join(("data", "k_SiO2_Palik.dat"))
    rsc_k = pkg_resources.resource_string(pkg_name, k_pth)
    data_k = np.genfromtxt(BytesIO(rsc_k), delimiter=",")
    wvl_k_dat = 1e-6*data_k[:, 0]  # m
    k_dat = data_k[:, 1]

    # Interpolate n&k with cubic spline
    n_itp_func = interp1d(wvl_n_dat, n_dat, kind="cubic")
    k_itp_func = interp1d(wvl_k_dat, k_dat, kind="cubic")

    return (n_itp_func(wvl) + 1j*k_itp_func(wvl))**2
