# -*- coding: utf-8 -*-

import numpy as np

# Identity matrix
Id = np.mat([[1, 0], [0, 1]])


def redheffer(SA, SB):
    """Redheffer star product"""
    D = SA[0:2, 2:4]*(Id-SB[0:2, 0:2]*SA[2:4, 2:4]).I
    F = SB[2:4, 0:2]*(Id-SA[2:4, 2:4]*SB[0:2, 0:2]).I
    SG = np.mat(np.empty([4, 4]), dtype=np.complex)
    SG[0:2, 0:2] = SA[0:2, 0:2] + D*SB[0:2, 0:2]*SA[2:4, 0:2]
    SG[0:2, 2:4] = D*SB[0:2, 2:4]
    SG[2:4, 0:2] = F*SA[2:4, 0:2]
    SG[2:4, 2:4] = SB[2:4, 2:4] + F*SA[2:4, 2:4]*SB[0:2, 2:4]
    return SG


def layer_list(input_layer):
    """
    Input  : Multilayer
    Output : List of layers in multilayer
    """
    if type(input_layer).__name__ == "MultiLayer":
        input_layers = input_layer.num_uc.value*input_layer.unit_cell
    else:
        input_layers = [input_layer]

    output_layers = []
    for layer in input_layers:
        if type(layer).__name__ == "MultiLayer":
            output_layers.extend(layer_list(layer))
        else:
            output_layers.append(layer)
    return output_layers


def polarization(theta, phi, pTE, pTM):
    """Polarization vector of incident fields"""
    k_inc = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    az = np.array([0., 0., 1.])

    if theta == 0:
        aTE = np.array([0., 1., 0.])
    else:
        aTE = np.cross(az, k_inc)
        aTE = aTE/np.linalg.norm(aTE)

    aTM = np.cross(aTE, k_inc)
    aTM = aTM/np.linalg.norm(aTM)

    P = pTE*aTE + pTM*aTM
    P = P/np.linalg.norm(P)
    return P


# Scattering matrices
def S_layer(kx, ky, kz, wvl, V_h, d):
    """Scattering matrix for single layer"""
    Q_i = np.mat([[kx*ky, kz**2+ky**2], [-(kz**2+kx**2), -kx*ky]])
    Om_i = 1j*kz*Id
    V_i = Q_i*Om_i.I
    A_i = Id + V_i.I*V_h
    A_inv = A_i.I
    B_i = Id - V_i.I*V_h
    k0 = 2*np.pi/wvl
    X_i = Id * np.exp(1j*kz*k0*d)
    D_inv = (A_i - X_i*B_i*A_inv*X_i*B_i).I
    S11 = D_inv*(X_i*B_i*A_inv*X_i*A_i - B_i)
    S12 = D_inv*X_i*(A_i - B_i*A_inv*B_i)
    S21 = S12
    S22 = S11
    S = np.bmat([[S11, S12], [S21, S22]])
    return S


def S_ref(kx, ky, kz, V_h):
    """Reflection side matrix"""
    Q_ref = np.mat([[kx*ky, kz**2+ky**2], [-(kz**2+kx**2), -kx*ky]])
    Om_ref = 1j*kz*Id
    V_ref = Q_ref*Om_ref.I
    A_ref = Id + V_h.I*V_ref
    B_ref = Id - V_h.I*V_ref
    S11 = -A_ref.I*B_ref
    S12 = 2*A_ref.I
    S21 = 0.5*(A_ref-B_ref*A_ref.I*B_ref)
    S22 = B_ref*A_ref.I
    S = np.bmat([[S11, S12], [S21, S22]])
    return S


def S_trn(kx, ky, kz, V_h):
    """Transmission side matrix"""
    Q_trn = np.mat([[kx*ky, kz**2+ky**2], [-(kz**2+kx**2), -kx*ky]])
    Om_trn = 1j*kz*Id
    V_trn = Q_trn*Om_trn.I
    A_trn = Id + V_h.I*V_trn
    B_trn = Id - V_h.I*V_trn
    S11 = B_trn*A_trn.I
    S12 = 0.5*(A_trn-B_trn*A_trn.I*B_trn)
    S21 = 2*A_trn.I
    S22 = -A_trn.I*B_trn
    S = np.bmat([[S11, S12], [S21, S22]])
    return S


def computeE2(ml, fd, incm, subm, wvl_id):
    """
    Input:
    ml      : Multilayer
    fd      : Incident fields
    incm    : Medium of incidence
    subm    : Medium of reflection
    wvl_id  : Index of selected wavelength
    """

    # List of layers from multilayer object
    layers = layer_list(ml)

    # Convert degrees to radians
    theta = np.pi*fd.theta.value/180.
    phi = np.pi*fd.phi.value/180.

    # Polarization vector
    P = polarization(theta, phi, fd.pTE.value, fd.pTM.value)

    # Wavevector in vacuum
    k0 = 2*np.pi/fd.wvl[wvl_id]

    # Transverse component of wavevector
    kx = incm.N[wvl_id]*np.sin(theta)*np.cos(phi)
    ky = incm.N[wvl_id]*np.sin(theta)*np.sin(phi)

    # Homogeneous gap properties
    Q_h = np.mat([[kx*ky, 1.+ky**2], [-(1.+kx**2), -kx*ky]])
    V_h = -1j*Q_h

    # Reflection matrix
    kz_ref = np.sqrt(incm.N[wvl_id]**2 - kx**2 - ky**2)
    S_glob = S_ref(kx, ky, kz_ref, V_h)

    # List of scattering matrices
    S_list = []
    for layer in layers:
        kz = np.sqrt(layer.N[wvl_id]**2 - kx**2 - ky**2)
        S_i = S_layer(kx, ky, kz, fd.wvl[wvl_id], V_h, layer.d.value)
        S_list.append(S_i)
        S_glob = redheffer(S_glob, S_i)

    # Transmission matrix
    kz_trn = np.sqrt(subm.N[wvl_id]**2 - kx**2 - ky**2)
    S_t = S_trn(kx, ky, kz_trn, V_h)
    S_glob = redheffer(S_glob, S_t)

    # Transverse components of incident, reflected and transmitted fields
    e_src = np.mat([[P[0]], [P[1]]])
    e_ref = S_glob[0:2, 0:2]*e_src
    e_trn = S_glob[2:4, 0:2]*e_src

    # Longitudinal components of reflected and transmitted fields
    #ez_ref = -(kx*e_ref.item(0) + ky*e_ref.item(1))/kz_ref
    #ez_trn = -(kx*e_trn.item(0) + ky*e_trn.item(1))/kz_trn

    # Transverse fields coefficients in incident medium
    c_incm = np.bmat([[e_src], [e_ref]])

    # Transverse fields coefficients in first gap
    M_h = np.bmat([[Id, Id], [V_h, -V_h]])
    Q_ref = np.mat([[kx*ky, kz_ref**2+ky**2], [-(kz_ref**2+kx**2), -kx*ky]])
    Om_ref = 1j*kz_ref*Id
    V_ref = Q_ref*Om_ref.I
    M_ref = np.bmat([[Id, Id], [V_ref, -V_ref]])
    D_ref = M_h.I*M_ref
    c_h_0 = D_ref*c_incm

    # Electric field inside layers
    c_h_list = [c_h_0]
    z_list = []
    E2_list = []
    z_offset = 0.
    for m in range(len(layers)):
        kz_i = np.sqrt(layers[m].N[wvl_id]**2 - kx**2 - ky**2)

        # Fields coefficients inside layer
        Q_i = np.mat([[kx*ky, kz_i**2+ky**2], [-(kz_i**2+kx**2), -kx*ky]])
        Om_i = 1j*kz_i*Id
        V_i = Q_i*Om_i.I
        M_i = np.bmat([[Id, Id], [V_i, -V_i]])
        M_h = np.bmat([[Id, Id], [V_h, -V_h]])
        D_i = M_i.I*M_h
        c_i = D_i*c_h_list[m]

        # Gap coefficients for next layer
        cp_1 = c_h_list[m][0:2]
        cm_1 = c_h_list[m][2:4]
        S_i = S_list[m]
        cm_2 = S_i[0:2, 2:4].I*(cm_1-S_i[0:2, 0:2]*cp_1)
        cp_2 = S_i[2:4, 0:2]*cp_1+S_i[2:4, 2:4]*cm_2
        c_h_2 = np.bmat([[cp_2], [cm_2]])
        c_h_list.append(c_h_2)

        # Positions inside layer
        z_i = np.arange(0., layers[m].d.value, 5e-8)

        # Squared electric field
        E2_i = []
        for z in z_i:
            Exy = np.exp(1j*k0*kz_i*z)*c_i[0:2] + np.exp(-1j*k0*kz_i*z)*c_i[2:4]
            Ex = Exy.item(0)
            Ey = Exy.item(1)
            Ez = -(kx*Ex+ky*Ey)/kz_i
            E2_i.append(np.absolute(Ex)**2 + np.absolute(Ey)**2 + np.absolute(Ez)**2)

        z_list.extend(z_i+z_offset)
        E2_list.extend(E2_i)

        z_offset += layers[m].d.value

    z_arr = np.array(z_list)
    E2_arr = np.array(E2_list)

    return z_arr, E2_arr


def computeRT(ml, fd, incm, subm):
    """
    Input:
    ml      : Multilayer
    fd      : Incident fields
    incm    : Medium of incidence
    subm    : Medium of reflection
    """

    # List of layers from multilayer object
    layers = layer_list(ml)

    # Convert degrees to radians
    theta = np.pi*fd.theta.value/180.
    phi = np.pi*fd.phi.value/180.

    # Polarization vector
    P = polarization(theta, phi, fd.pTE.value, fd.pTM.value)

    # Reflectance and transmittance
    R = np.empty(fd.num_wvl.value)
    T = np.empty(fd.num_wvl.value)

    # Loop on all wavelengths
    for q in range(fd.num_wvl.value):

        # Transverse components of wavevector
        kx = incm.N[q]*np.sin(theta)*np.cos(phi)
        ky = incm.N[q]*np.sin(theta)*np.sin(phi)

        # Homogeneous gap properties
        Q_h = np.mat([[kx*ky, 1.+ky**2], [-(1.+kx**2), -kx*ky]])
        V_h = -1j*Q_h

        # Reflection matrix
        kz_ref = np.sqrt(incm.N[q]**2 - kx**2 - ky**2)
        S_glob = S_ref(kx, ky, kz_ref, V_h)

        # Loop on all layers
        for layer in layers:
            kz = np.sqrt(layer.N[q]**2 - kx**2 - ky**2)
            S_i = S_layer(kx, ky, kz, fd.wvl[q], V_h, layer.d.value)
            S_glob = redheffer(S_glob, S_i)

        # Transmission matrix
        kz_trn = np.sqrt(subm.N[q]**2 - kx**2 - ky**2)
        S_t = S_trn(kx, ky, kz_trn, V_h)
        S_glob = redheffer(S_glob, S_t)

        # Transverse components of incident, reflected and transmitted fields
        e_src = np.mat([[P[0]], [P[1]]])
        e_ref = S_glob[0:2, 0:2]*e_src
        e_trn = S_glob[2:4, 0:2]*e_src

        # Longitudinal components of reflected and transmitted fields
        ez_ref = -(kx*e_ref.item(0) + ky*e_ref.item(1))/kz_ref
        ez_trn = -(kx*e_trn.item(0) + ky*e_trn.item(1))/kz_trn

        # Reflected and transmitted fields
        E_ref = np.array([e_ref.item(0), e_ref.item(1), ez_ref])
        E_trn = np.array([e_trn.item(0), e_trn.item(1), ez_trn])

        R[q] = np.linalg.norm(E_ref)**2
        T[q] = np.real(kz_trn/kz_ref)*np.linalg.norm(E_trn)**2

    return R, T
