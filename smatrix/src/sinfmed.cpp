
#include "sinfmed.h"
#include "fields.h"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std::complex_literals;

SemiInfMed::SemiInfMed(dcomp* N_p, int wvl_s): N(N_p, wvl_s)
{
    // Permittivity of semi-infinite medium
    eps_r = N.pow(2);
}

void SemiInfMed::compute_kz(Fields* f_p)
{
    // Longitudinal component of the wavevector in semi-infinite medium
    kz = (eps_r - f_p->kx.pow(2) - f_p->ky.pow(2)).sqrt();
}

void SemiInfMed::compute_matrices(Fields* f_p, int wvl_index)
{
    // Compute all matrices used to find scattering matrix

    Matrix2cd Q, Omega, V_h, V, Vp, X, M, D, D_inv;

    // Transverse components of the wavevector in semi-infinite medium
    kx = f_p->kx[wvl_index];
    ky = f_p->ky[wvl_index];

    V_h << kx*ky, 1.0 + ky*ky,
          -1.0 - kx*kx, -kx*ky;
    V_h = -1i*V_h;

    Q << kx*ky, eps_r[wvl_index] - pow(kx, 2),
         pow(ky, 2) - eps_r[wvl_index], -kx*ky;

    Omega = 1i*kz[wvl_index]*I;
    V = Q*Omega.inverse();
    Vp = V_h.inverse()*V;
    A = I + Vp;
    B = I - Vp;
    A_inv = A.inverse();
}

SMatrix SemiInfMed::compute_S_refl(Fields* f_p, int wvl_index)
{
    // Scattering matrix if semi-infinite medium is on reflection side
    compute_matrices(f_p, wvl_index);

    SMatrix SM;
    SM.S_11 = -A_inv*B;
    SM.S_12 = 2*A_inv;
    SM.S_21 = 0.5*(A - B*A_inv*B);
    SM.S_22 = B*A_inv;

    return SM;
}

SMatrix SemiInfMed::compute_S_tran(Fields* f_p, int wvl_index)
{
    // Scattering matrix if semi-infinite medium is on transmission side
    compute_matrices(f_p, wvl_index);

    SMatrix SM;
    SM.S_11 = B*A_inv;
    SM.S_12 = 0.5*(A - B*A_inv*B);
    SM.S_21 = 2*A_inv;
    SM.S_22 = -A_inv*B;

    return SM;
}
