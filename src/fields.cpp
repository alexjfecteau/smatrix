
#define EIGEN_USE_MKL_ALL
#include "fields.h"
#include "sinfmed.h"

using namespace Eigen;
using namespace std::complex_literals;

Fields::Fields(double pTE, double pTM, double theta, double phi, double* wvl_p, int size_wvl):
pTE(pTE), pTM(pTM), theta(theta), phi(phi), wvl(wvl_p, size_wvl), size_wvl(size_wvl)
{
    // Convert angles to radians
    theta = M_PI*theta/180;
    phi = M_PI*phi/180;

    // Incident wave vector amplitude
    k0 = 2.0*M_PI/wvl;

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

void Fields::compute_kx_ky(SemiInfMed* inc_p)
{
    // Compute transverse components of the wavevector
    kx = inc_p->N*sin(theta)*cos(phi);
    ky = inc_p->N*sin(theta)*sin(phi);
}

void Fields::compute_E_refl_tran(SMatrix SGlob, SemiInfMed* inc_p, SemiInfMed* sub_p, int wvl_index)
{
    // Compute reflected and transmitted fields

    // Transverse field components
    Vector2cd e_src(P[0], P[1]), e_refl, e_tran;

    e_refl = SGlob.S_11*e_src;
    e_tran = SGlob.S_21*e_src;

    // Longitudinal field components
    dcomp ez_refl = -(kx[wvl_index]*e_refl(0) + ky[wvl_index]*e_refl(1))/inc_p->kz[wvl_index];
    dcomp ez_tran = -(kx[wvl_index]*e_tran(0) + ky[wvl_index]*e_tran(1))/sub_p->kz[wvl_index];

    // Update reflected and transmitted fields
    E_refl << e_refl(0), e_refl(1), ez_refl;
    E_tran << e_tran(0), e_tran(1), ez_tran;
}
