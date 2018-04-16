
#include "smatrix.h"
#include "layers.h"
#include "sinfmed.h"
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

ScatteringMatrix::ScatteringMatrix(Layer* multilayer, Fields* fields, SemiInfMed* inc_med, SemiInfMed* sub_med):
m_p(multilayer), f_p(fields), inc_p(inc_med), sub_p(sub_med) {}

void ScatteringMatrix::compute_R_T(dcomp* r_TE_p, dcomp* t_TE_p, dcomp* r_TM_p, dcomp* t_TM_p, double* R_p, double* T_p)
{
    // Fresnel coefficients, reflectance and transmittance

    // Compute wavevector components
    f_p->compute_kx_ky(inc_p);
    inc_p->compute_kz(f_p);
    sub_p->compute_kz(f_p);

    // Loop through all wavelengths
    SMatrix SGlob, STran, SMulti;
    for (int q=0; q<f_p->size_wvl; q++)
    {
        // Set SGlob as scattering matrix on reflection side
        SGlob = inc_p->compute_S_refl(f_p, q);

        // Multiply global scattering matrix by scattering matrix
        // of multilayer using Redheffer star product
        m_p->compute_S(f_p, q);
        SMulti = m_p->S;
        SGlob = redheffer(SGlob, SMulti);

        // Multiply global scattering matrix by scattering matrix
        // on transmission side using Redheffer star product
        STran = sub_p->compute_S_tran(f_p, q);
        SGlob = redheffer(SGlob, STran);

        // Compute reflected and transmitted electric fields
        f_p->compute_E_refl_tran(SGlob, inc_p, sub_p, q);

        // Compute Fresnel coefficients for TE polarization
        r_TE_p[q] = f_p->aTE.dot(f_p->E_refl);
        t_TE_p[q] = f_p->aTE.dot(f_p->E_tran);

        // Compute Fresnel coefficients for TM polarization
        r_TM_p[q] = f_p->aTM.dot(f_p->E_refl);
        t_TM_p[q] = f_p->aTM.dot(f_p->E_tran);

        // Compute reflectance and transmittance
        R_p[q] = f_p->E_refl.squaredNorm();
        T_p[q] = (sub_p->kz[q]/inc_p->kz[q]).real() * f_p->E_tran.squaredNorm();
    }
}
