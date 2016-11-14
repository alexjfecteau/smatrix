
#define EIGEN_USE_MKL_ALL
#include "fields.h"

using namespace Eigen;
using namespace std::complex_literals;

Fields::Fields(double pTE, double pTM, double theta, double phi, double* wvl_p, int size_wvl, double N_inc, double N_sub):
pTE(pTE), pTM(pTM), theta(theta), phi(phi), wvl(wvl_p, size_wvl), size_wvl(size_wvl), N_inc(N_inc), N_sub(N_sub)
{
    // Convert angles to radians
    theta = M_PI*theta/180;
    phi = M_PI*phi/180;

    // Incident wave vector amplitude
    k0 = 2.0*M_PI/wvl;

    // Transverse wave vectors
    kx = N_inc*sin(theta)*cos(phi);
    ky = N_inc*sin(theta)*sin(phi);

    V_h << kx*ky, 1.0 + ky*ky,
           -1.0 - kx*kx, -kx*ky;
    V_h = -1i*V_h;

    // Permittivities and longitudinal wavevectors in incident medium and substrate
    eps_r_inc = pow(N_inc, 2);
    eps_r_sub = pow(N_sub, 2);
    kz_inc = sqrt(eps_r_inc - pow(kx, 2) - pow(ky, 2));
    kz_sub = sqrt(eps_r_sub - pow(kx, 2) - pow(ky, 2));

    // Polarization vector of incident fields
    k_inc << sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta);
    az << 0, 0, 1;

    if (theta == 0)
    {
        aTE << 0, 1, 0;
    }
    else
    {
        aTE = az.cross(k_inc);
        aTE.normalize();
    }

    aTM = aTE.cross(k_inc);
    aTM.normalize();

    P = pTE*aTE + pTM*aTM;
    P.normalize();
}

void Fields::E_refl_tran(SMatrix SGlob)
{
    // Compute reflected and transmitted fields

    // Transverse field components
    Vector2cd e_src(P[0], P[1]), e_refl, e_tran;

    e_refl = SGlob.S_11*e_src;
    e_tran = SGlob.S_21*e_src;

    // Longitudinal field components
    dcomp ez_refl = -(kx*e_refl(0) + ky*e_refl(1))/kz_inc;
    dcomp ez_tran = -(kx*e_tran(0) + ky*e_tran(1))/kz_sub;

    // Update reflected and transmitted fields
    E_refl << e_refl(0), e_refl(1), ez_refl;
    E_tran << e_tran(0), e_tran(1), ez_tran;
}

//void Fields::E_slayer(Layer* l_p, int wvl_index)
//{
//    // Compute electric field in single layer at every 100 nm
//
//    // Longitudinal wavevector in single layer
//    dcomp kz = sqrt(l_p->eps_r[wvl_index] - pow(kx, 2) - pow(ky, 2));
//
//    // Amplitude coefficients in layer
//    l_p->cp_i = 0.5*l_p->V_inv*( (l_p->V + V_h) * l_p->cp_1 + (l_p->V - V_h) * l_p->cm_1 );
//    l_p->cm_i = 0.5*l_p->V_inv*( (l_p->V - V_h) * l_p->cp_1 + (l_p->V + V_h) * l_p->cm_1 );
//
//    // Loop over all positions in layer
//    Vector2cd Exy;
//    dcomp Ez;
//
//    for (int k=0; k<layer->single_p->size_z; k+=1)
//    {
//        // Transverse electric fields
//        Exy = (1i*kz*I*l_p->z(k)).exp()*l_p->cp_i + (-1i*kz*I*l_p->z(k)).exp()*l_p->cm_i;
//        l_p->Ex(wvl_index, k) = Exy(0);
//        l_p->Ey(wvl_index, k) = Exy(1);
//
//        // Longitudinal electric field
//        Ez = -(kx*Exy(0) + ky*Exy(1))/kz;
//        l_p->Ez(wvl_index, k) = Ez;
//
//        // Squared norm
//        Vector3cd E(Exy(0), Exy(1), Ez);
//        l_p->E2(k) = E.squaredNorm();
//    }
//
//    // Update amplitude coefficients for next layer
//    SMatrix S = layer->single_p->S;
//    l_p->cm_1 = l_p->S.S_12.inverse()*(l_p->cm_1 - l_p->S.S_11*l_p->cp_1);
//    l_p->cp_1 = l_p->S.S_21*l_p->cp_1 + l_p->S.S_22*l_p->cm_1;
//}
//
//void Fields::E_mlayer(Layer* layer, int wvl_index)
//{
//    // Compute electric field inside multilayer
//
//    if (layer->type == Layer::multi)
//    {
//        for (int k=0; k<layer->multi_p->num_uc; k+=1)
//        {
//            // Scattering matrix for unit cell of multilayer
//            for (int l=0; l<layer->multi_p->num_layers; l+=1)
//            {
//                E_mlayer(layer->multi_p->unit_cell[l], wvl_index);
//            }
//        }
//    }
//    else
//    {
//        // Electric field for single layer
//        E_slayer(layer, wvl_index);
//    }
//}
//
//void Fields::compute_E(Layer* multilayer)
//{
//    // Loop through all wavelengths
//    for (int q=0; q<size_wvl; q++)
//    {
//        E_mlayer(multilayer, q);
//    }
//}
//