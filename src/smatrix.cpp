
#define EIGEN_USE_MKL_ALL
#include "smatrix.h"
#include "layers.h"
#include "fields.h"

using namespace Eigen;
using namespace std::complex_literals;

Matrix2cd I = Matrix2cd::Identity();
Matrix2cd Z = Matrix2cd::Zero();

SMatrix redheffer(SMatrix SA, SMatrix SB)
{
    // Compute Redheffer star product

    Matrix2cd E, F;
    E = SA.S_12*(I - SB.S_11*SA.S_22).inverse();
    F = SB.S_21*(I - SA.S_22*SB.S_11).inverse();

    SMatrix SAB;
    SAB.S_11 = SA.S_11 + E*SB.S_11*SA.S_21;
    SAB.S_12 = E*SB.S_12;
    SAB.S_21 = F*SA.S_21;
    SAB.S_22 = SB.S_22 + F*SA.S_22*SB.S_12;
    return SAB;
}

ScatteringMatrix::ScatteringMatrix(Layer* multilayer, Fields* fields):
m_p(multilayer), f_p(fields) {}

SMatrix ScatteringMatrix::S_refl()
{
    // Compute scattering matrix for semi-infinite medium on reflection side

    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B;

    Q << f_p->kx*f_p->ky, f_p->eps_r_inc - pow(f_p->kx, 2),
         pow(f_p->ky, 2) - f_p->eps_r_inc, -f_p->kx*f_p->ky;

    Omega = 1i*f_p->kz_inc*I;
    V = Q*Omega.inverse();
    Vp = f_p->V_h.inverse()*V;
    A = I + Vp;
    B = I - Vp;
    A_inv = A.inverse();

    SMatrix SM;
    SM.S_11 = -A_inv*B;
    SM.S_12 = 2*A_inv;
    SM.S_21 = 0.5*(A - B*A_inv*B);
    SM.S_22 = B*A_inv;

    return SM;
}

SMatrix ScatteringMatrix::S_tran()
{
    // Compute scattering matrix for semi-infinite medium on transmission side

    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B;

    Q << f_p->kx*f_p->ky, f_p->eps_r_sub - pow(f_p->kx, 2),
         pow(f_p->ky, 2) - f_p->eps_r_sub, -f_p->kx*f_p->ky;

    Omega = 1i*f_p->kz_sub*I;
    V = Q*Omega.inverse();
    Vp = f_p->V_h.inverse()*V;
    A = I + Vp;
    B = I - Vp;
    A_inv = A.inverse();

    SMatrix SM;
    SM.S_11 = B*A_inv;
    SM.S_12 = 0.5*(A - B*A_inv*B);
    SM.S_21 = 2*A_inv;
    SM.S_22 = -A_inv*B;

    return SM;
}

void ScatteringMatrix::compute_R_T(dcomp* r, dcomp* t, double* R, double* T)
{
    // Compute Fresnel coefficients, reflectance and transmittance
    r_p = r;
    t_p = t;
    R_p = R;
    T_p = T;

    SMatrix SGlob, SRefl, STran, SMulti;

    // Scattering matrix for reflection side
    SRefl = S_refl();

    // Scattering matrix for transmission side
    STran = S_tran();

    // Loop through all wavelengths
    for (int q=0; q<f_p->size_wvl; q++)
    {
        // Scattering matrix on reflection side
        SGlob = SRefl;

        // Multiply global scattering matrix by scattering matrix
        // of multilayer using Redheffer star product
        m_p->compute_S(f_p, q);
        SMulti = m_p->S;
        SGlob = redheffer(SGlob, SMulti);

        // Multiply global scattering matrix by scattering matrix
        // on transmission side using Redheffer star product
        SGlob = redheffer(SGlob, STran);

        // Compute reflected and transmitted electric fields
        f_p->Compute_E_refl_tran(SGlob);

        // Compute Fresnel coefficients
        r_p[q] = f_p->E_refl(1)/f_p->P(1);
        t_p[q] = f_p->E_tran(1)/f_p->P(1);

        // Compute reflectance and transmittance
        R_p[q] = f_p->E_refl.squaredNorm();
        T_p[q] = (f_p->kz_sub/f_p->kz_inc).real() * f_p->E_tran.squaredNorm();
    }
}

void ScatteringMatrix::compute_E(int wvl_index, double* E2_p)
{
    // Compute electric field in structure

    // Initial field coefficients
    m_p->cp_1 << f_p->P(0), f_p->P(1);
    m_p->cm_1 << 0, 0;

    m_p->compute_E(f_p, wvl_index);

    //Eigen::Map<Eigen::ArrayXcd> Ex, Ey, Ez;
    Eigen::Map<Eigen::ArrayXd> E2(E2_p, m_p->size_z);

    E2 << m_p->E2;
}
