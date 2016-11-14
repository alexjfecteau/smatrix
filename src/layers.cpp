
#define EIGEN_USE_MKL_ALL
#include "layers.h"
#include "fields.h"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std::complex_literals;

SingleLayer::SingleLayer(double* wvl_p, dcomp* N_p, int wvl_s, double d):
wvl(wvl_p, wvl_s), N(N_p, wvl_s)
{
    // Number of wavelengths
    size_wvl = wvl_s;
    // Thickness of layer
    thk = d;
    // Permittivity of layer
    eps_r = N.pow(2);

    // Discretize space inside layer with one point every 100 nm
    size_z = thk/0.1;
    z = ArrayXd::LinSpaced(size_z, 0, thk);

    // Electric field components and norm
    //Ex(size_wvl, size_z);
    //Ey(size_wvl, size_z);
    //Ez(size_wvl, size_z);
    //E2(size_wvl, size_z);
}

void SingleLayer::compute_S(Fields* f_p, int wvl_index)
{
    // Compute scattering matrix for layer of finite size

    // Longitudinal wavevector in single layer
    dcomp kz = sqrt(eps_r[wvl_index] - pow(f_p->kx, 2) - pow(f_p->ky, 2));

    Q << f_p->kx*f_p->ky, eps_r[wvl_index] - pow(f_p->kx, 2),
         pow(f_p->ky, 2) - eps_r[wvl_index], -f_p->kx*f_p->ky;

    Omega = 1i*kz*I;
    V = Q*Omega.inverse();
    X = (f_p->k0[wvl_index]*thk*Omega).exp();
    V_inv = V.inverse();
    Vp = V_inv*f_p->V_h;
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
    size_z_uc = 0;
    num_uc = 1;
    num_layers = 0;
}

void MultiLayer::add_to_unit_cell(Layer* layer)
{
    num_layers += 1;
    unit_cell.resize(num_layers, layer);

    thk_uc += layer->thk;
    size_z_uc += layer->size_z;

    thk = num_uc * thk_uc;
    size_z = num_uc * size_z_uc;
}

void MultiLayer::clear_unit_cell()
{
    unit_cell.clear();
}

void MultiLayer::set_num_repetitions(int n)
{
    num_uc = n;
    thk = num_uc * thk_uc;
    size_z = num_uc * size_z_uc;
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
