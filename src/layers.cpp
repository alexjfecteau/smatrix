
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

    // Discretize space inside layer in 10 points
    size_z = 10;
    z = ArrayXd::LinSpaced(size_z, 0, thk);
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

void SingleLayer::compute_E(Fields* f_p, int wvl_index)
{
    // Compute electric field in single layer

    // Electric field components and norm
    Ex = ArrayXcd::Zero(size_z);
    Ey = ArrayXcd::Zero(size_z);
    Ez = ArrayXcd::Zero(size_z);
    E2 = ArrayXd::Zero(size_z);

    // Scattering matrix for specified wavelength
    compute_S(f_p, wvl_index);

    // Wavevector norm for specified wavelength
    double k0 = f_p->k0[wvl_index];

    // Longitudinal wavevector for specified wavelength
    dcomp kz = sqrt(eps_r[wvl_index] - pow(f_p->kx, 2) - pow(f_p->ky, 2));

    // Amplitude coefficients in layer
    cp_i = 0.5*V_inv*((V + f_p->V_h) * cp_1 + (V - f_p->V_h) * cm_1 );
    cm_i = 0.5*V_inv*((V - f_p->V_h) * cp_1 + (V + f_p->V_h) * cm_1 );

    // Loop over all positions in layer
    Vector2cd Exy;

    for (int q=0; q<size_z; q+=1)
    {
        // Transverse electric fields
        Exy = (1i*k0*kz*I*z[q]).exp()*cp_i + (-1i*k0*kz*I*z[q]).exp()*cm_i;
        Ex(q) = Exy(0);
        Ey(q) = Exy(1);

        // Longitudinal electric field
        Ez(q) = -(f_p->kx*Exy(0) + f_p->ky*Exy(1))/kz;

        // Squared norm
        Vector3cd E(Ex(q), Ey(q), Ez(q));
        E2(q) = E.squaredNorm();
    }

    // Update amplitude coefficients for next layer
    cm_1_next = S.S_12.inverse()*(cm_1 - S.S_11*cp_1);
    cp_1_next = S.S_21*cp_1 + S.S_22*cm_1;
}

MultiLayer::MultiLayer()
{
    thk_uc = 0;
    size_z_uc = 0;
    num_uc = 1;
    num_layers = 0;
    z = ArrayXd::Zero(size_z);
}

void MultiLayer::add_to_unit_cell(Layer* l_p)
{
    num_layers += 1;
    unit_cell.resize(num_layers, l_p);

    thk_uc += l_p->thk;
    size_z_uc += l_p->size_z;

    thk = num_uc * thk_uc;
    size_z = num_uc * size_z_uc;
    z.resize(size_z);
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
    z.resize(size_z);
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

void MultiLayer::compute_E(Fields* f_p, int wvl_index)
{
    // Compute electric field in multilayer

    // Scattering matrix for specified wavelength
    compute_S(f_p, wvl_index);

    // First layer
    unit_cell[0]->cp_1 = cp_1;
    unit_cell[0]->cm_1 = cm_1;

    unit_cell[0]->compute_E(f_p, wvl_index);
    Ex = unit_cell[0]->Ex;
    Ey = unit_cell[0]->Ey;
    Ez = unit_cell[0]->Ez;
    E2 = unit_cell[0]->E2;

    int offset = 0;
    for (int l=1; l<num_layers; l+=1)
    {
        //for (int m=0; m<unit_cell[l]->size_z; m+=1)
        //{
        //    z[m+offset] = unit_cell[l]->z[m];
        //}
        //offset += unit_cell[l]->size_z;
        unit_cell[l]->compute_E(f_p, wvl_index);
        Ex = unit_cell[l]->Ex;
        Ey = unit_cell[l]->Ey;
        Ez = unit_cell[l]->Ez;
        E2 = unit_cell[l]->E2;

        unit_cell[l]->cp_1 = unit_cell[l-1]->cp_1_next;
        unit_cell[l]->cm_1 = unit_cell[l-1]->cm_1_next;
    }

    // Update amplitude coefficients for next layer
    cm_1_next = S.S_12.inverse()*(cm_1 - S.S_11*cp_1);
    cp_1_next = S.S_21*cp_1 + S.S_22*cm_1;

    //std::cout << unit_cell[1]->unit_cell[0]->z << std::endl;
    //std::cout << z << std::endl;
}
