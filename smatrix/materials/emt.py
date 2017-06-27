# -*- coding: utf-8 -*-

import numpy as np


def diel_MG(diel_h, diel_i, P):
    """
    Return effective dielectric function of composite medium
    using Maxwell Garnett model

    Inputs:
    e_h : dielectric function of host medium
    e_i : dielectric function of inclusions
    P   : porosity
    """

    def diel_eff(wvl):
        eps_h = diel_h(wvl)
        eps_i = diel_i(wvl)
        num = 1 + 2*P*(eps_i-eps_h)/(eps_i+2*eps_h)
        den = 1 - P*(eps_i-eps_h)/(eps_i+2*eps_h)
        return eps_h*num/den

    return diel_eff


def N_MG(N_h, N_i, P):
    """
    Return effective refractive index of composite medium
    using Maxwell Garnett model

    Inputs:
    N_h : complex refractive index function of host medium
    N_i : complex refractive index function of inclusions
    P   : porosity
    """

    def N_eff(wvl):
        eps_h = np.sqrt(N_h(wvl))
        eps_i = np.sqrt(N_i(wvl))
        num = 1 + 2*P*(eps_i-eps_h)/(eps_i+2*eps_h)
        den = 1 - P*(eps_i-eps_h)/(eps_i+2*eps_h)
        return (eps_h*num/den)**2

    return N_eff
