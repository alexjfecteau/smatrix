
#define EIGEN_USE_MKL_ALL
#include "layers.h"
#include "fields.h"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std::complex_literals;

SingleLayer::SingleLayer(dcomp* N_p, int wvl_s, double d): N(N_p, wvl_s)
{
    // Thickness of layer
    thk = d;

    // Permittivity of layer
    eps_r = N.pow(2);
}

void SingleLayer::compute_S(Fields* f_p, int wvl_index)
{
    // Compute scattering matrix for layer of finite size

    Matrix2cd Q, Omega, V_h, V, Vp, X, A, A_inv, B, M, D, D_inv;

    // Transverse components of the wavevector in the layer
    kx = f_p->kx[wvl_index];
    ky = f_p->ky[wvl_index];

    // Longitudinal component of the wavevector in the layer
    kz = sqrt(eps_r[wvl_index] - pow(kx, 2) - pow(ky, 2));

    V_h << kx*ky, 1.0 + ky*ky,
          -1.0 - kx*kx, -kx*ky;
    V_h = -1i*V_h;

    Q << kx*ky, eps_r[wvl_index] - pow(kx, 2),
         pow(ky, 2) - eps_r[wvl_index], -kx*ky;

    Omega = 1i*kz*I;
    V = Q*Omega.inverse();
    X = (f_p->k0[wvl_index]*thk*Omega).exp();
    Vp = V.inverse()*V_h;
    A = I + Vp;
    B = I - Vp;
    A_inv = A.inverse();
    M = X*B*A_inv*X;
    D = A - M*B;
    D_inv = D.inverse();

    S.S_11 = D_inv*(M*A - B);
    S.S_22 = S.S_11;
    S.S_12 = D_inv*X*(A - B*A_inv*B);
    S.S_21 = S.S_12;
}

MultiLayer::MultiLayer()
{
    thk_uc = 0;
    num_uc = 1;
    num_layers = 0;
}

void MultiLayer::add_to_unit_cell(Layer* l_p)
{
    num_layers += 1;
    unit_cell.resize(num_layers, l_p);

    thk_uc += l_p->thk;
    thk = num_uc * thk_uc;
}

void MultiLayer::clear_unit_cell()
{
    unit_cell.clear();
}

void MultiLayer::set_num_repetitions(int n)
{
    num_uc = n;
    thk = num_uc * thk_uc;
}

void MultiLayer::compute_S(Fields* f_p, int wvl_index)
{
    S.reset();
    S_uc.reset();

    // Scattering matrix for unit cell of multilayer
    for (int l=0; l<num_layers; l+=1)
    {
        unit_cell[l]->compute_S(f_p, wvl_index);
        S_uc = redheffer(S_uc, unit_cell[l]->S);
    }

    // Scattering matrix for multilayer
    for (int k=0; k<num_uc; k+=1)
    {
        S = redheffer(S, S_uc);
    }
}
