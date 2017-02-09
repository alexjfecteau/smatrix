
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

void Fields::Compute_E_refl_tran(SMatrix SGlob)
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
