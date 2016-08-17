
#define EIGEN_USE_MKL_ALL
#define DLLEXPORT  __declspec(dllexport)

#include <iostream>
#include <complex>
#include <cmath>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::Matrix2cd;
using Eigen::Vector3d;
using Eigen::Vector3cd;
using Eigen::Vector2cd;
using Eigen::ArrayXd;
using Eigen::ArrayXcd;
using Eigen::Array2d;
using Eigen::Array2cd;
using Eigen::Map;
using namespace std::complex_literals;

typedef std::complex<double> dcomp;

Matrix2cd I = Matrix2cd::Identity();
Matrix2cd Z = Matrix2cd::Zero();
Matrix2cd Omega, Q, V, V_h, Vp, X, A, A_inv, B, M, D, D_inv;

// Function prototypes for ctype import in python.

extern "C" {
DLLEXPORT void solve(double pTE, double pTM, double theta, double phi, double* wvl_p, dcomp* N_polar_p, int num_wvl,
                     double L_polar, double N_A, double N_B, double L_A, double L_B, int num_uc, double N_inc, double N_sub,
                     double* R_p, double* T_p);
}

// Functions to compute EM scattering in multilayer.

void redheffer(Matrix2cd& SA_11, Matrix2cd& SA_12, Matrix2cd& SA_21, Matrix2cd& SA_22,
               const Matrix2cd& SB_11, const Matrix2cd& SB_12, const Matrix2cd& SB_21, const Matrix2cd& SB_22)
{
    // Compute Redheffer star product.

    Matrix2cd E, F;
    E = SA_12*(I - SB_11*SA_22).inverse();
    F = SB_21*(I - SA_22*SB_11).inverse();

    SA_11 = SA_11 + E*SB_11*SA_21;
    SA_12 = E*SB_12;
    SA_21 = F*SA_21;
    SA_22 = SB_22 + F*SA_22*SB_12;
}

void s_inf(bool ref, double kx, double ky, dcomp kz, dcomp eps_r, Matrix2cd& V_h,
           Matrix2cd& S_11, Matrix2cd& S_12, Matrix2cd& S_21, Matrix2cd& S_22)
{
    // Compute scattering matrix for semi-infinite medium.
    // bool ref = true : Reflection side.
    // bool ref = false : Transmission side.

    Q << kx*ky, eps_r - pow(kx, 2),
         pow(ky, 2) - eps_r, -kx*ky;

    Omega = 1i*kz*I;
    V = Q*Omega.inverse();
    Vp = V_h.inverse()*V;
    A = I + Vp;
    B = I - Vp;
    A_inv = A.inverse();

    if (ref)
    {
        S_11 = -A_inv*B;
        S_12 = 2*A_inv;
        S_21 = 0.5*(A - B*A_inv*B);
        S_22 = B*A_inv;
    }
    else
    {
        S_11 = B*A_inv;
        S_12 = 0.5*(A - B*A_inv*B);
        S_21 = 2*A_inv;
        S_22 = -A_inv*B;
    }

}

void s_layer(double k0, double kx, double ky, dcomp kz, dcomp eps_r, Matrix2cd& V_h, double L,
             Matrix2cd& S_11, Matrix2cd& S_12, Matrix2cd& S_21, Matrix2cd& S_22)
{
    // Compute scattering matrix for layer of finite size.

    Q << kx*ky, eps_r - pow(kx, 2),
         pow(ky, 2) - eps_r, -kx*ky;

    Omega = 1i*kz*I;
    V = Q*Omega.inverse();
    X = (k0*L*Omega).exp();
    Vp = V.inverse()*V_h;
    A = I + Vp;
    B = I - Vp;
    A_inv = A.inverse();
    M = X*B*A_inv*X;
    D = A - M*B;
    D_inv = D.inverse();

    S_11 = D_inv*(M*A - B);
    S_22 = S_11;
    S_12 = D_inv*X*(A - B*A_inv*B);
    S_21 = S_12;
}

void R_T_coeffs(double pTE, double pTM, double theta, double phi,
                double kx, double ky, dcomp kz_ref, dcomp kz_trn,
                Matrix2cd& S_11, Matrix2cd& S_12, Matrix2cd& S_21, Matrix2cd& S_22,
                double& R, double& T)
{
    // Compute polarization vector of incident fields.
    Vector3d k_inc(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
    Vector3d az(0, 0, 1), aTE, aTM, P;

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

    // Compute reflected and transmitted fields.
    Vector2cd e_src(P(0), P(1)), e_ref, e_trn;

    e_ref = S_11*e_src;
    e_trn = S_21*e_src;

    // Compute normal field components.
    dcomp ez_ref = -(kx*e_ref(0) + ky*e_ref(1))/kz_ref;
    dcomp ez_trn = -(kx*e_trn(0) + ky*e_trn(1))/kz_trn;

    Vector3cd E_ref(e_ref(0), e_ref(1), ez_ref);
    Vector3cd E_trn(e_trn(0), e_trn(1), ez_trn);

    // Compute reflectance and transmittance coefficients.
    R = E_ref.squaredNorm();
    T = (kz_trn/kz_ref).real() * E_trn.squaredNorm();
}

void solve(double pTE, double pTM, double theta, double phi, double* wvl_p, dcomp* N_polar_p, int num_wvl,
           double L_polar, double N_A, double N_B, double L_A, double L_B, int num_uc, double N_inc, double N_sub,
           double* R, double* T)
{
    // Incident wave vector amplitude.
    Map<ArrayXd> wvl(wvl_p, num_wvl);
    ArrayXd k0 = 2.0*M_PI/wvl;

    // Transverse wave vectors.
    double kx = N_inc*sin(M_PI*theta/180.0)*cos(M_PI*phi/180.0);
    double ky = N_inc*sin(M_PI*theta/180.0)*sin(M_PI*phi/180.0);

    // Gap parameters.
    V_h << kx*ky, 1.0 + ky*ky,
           -1.0 - kx*kx, -kx*ky;
    V_h = -1i*V_h;

    // Normal wave vectors in incident medium and substrate.
    dcomp eps_r_inc = pow(N_inc, 2);
    dcomp eps_r_sub = pow(N_sub, 2);
    dcomp kz_inc = sqrt(eps_r_inc - pow(kx, 2) - pow(ky, 2));
    dcomp kz_sub = sqrt(eps_r_sub - pow(kx, 2) - pow(ky, 2));

    // Normal wave vector in polar material.
    Map<ArrayXcd> N_polar(N_polar_p, num_wvl);
    ArrayXcd eps_r_polar = N_polar.square();
    ArrayXcd kz_polar = (eps_r_polar - pow(kx, 2) - pow(ky, 2)).sqrt();

    // Normal wave vector in Bragg reflector.
    dcomp eps_r_A = pow(N_A, 2);
    dcomp eps_r_B = pow(N_B, 2);
    dcomp kz_A = sqrt(eps_r_A - pow(kx, 2) - pow(ky, 2));
    dcomp kz_B = sqrt(eps_r_B - pow(kx, 2) - pow(ky, 2));

    // Define matrices for scattering computation.
    Matrix2cd Sglob_11, Sglob_12, Sglob_21, Sglob_22;
    Matrix2cd S_11, S_12, S_21, S_22;
    Matrix2cd Sref_11, Sref_12, Sref_21, Sref_22;
    Matrix2cd Strn_11, Strn_12, Strn_21, Strn_22;
    Matrix2cd Suc_11, Suc_12, Suc_21, Suc_22;

    // Compute scattering matrix for reflection side and
    // update global scattering matrix using Redheffer star product.
    s_inf(true, kx, ky, kz_inc, eps_r_inc, V_h, Sref_11, Sref_12, Sref_21, Sref_22);

    // Compute scattering matrix for transmission side.
    s_inf(false, kx, ky, kz_sub, eps_r_sub, V_h, Strn_11, Strn_12, Strn_21, Strn_22);

    // Loop through all wavelengths.
    for (int q=0; q<num_wvl; q++)
    {
        // Start with scattering matrix on reflection side.
        Sglob_11 = Sref_11;
        Sglob_12 = Sref_12;
        Sglob_21 = Sref_21;
        Sglob_22 = Sref_22;

        // Compute scattering matrix for polar material and
        // update global scattering matrix.
        s_layer(k0(q), kx, ky, kz_polar(q), eps_r_polar(q), V_h, L_polar,
                S_11, S_12, S_21, S_22);
        redheffer(Sglob_11, Sglob_12, Sglob_21, Sglob_22,
                  S_11, S_12, S_21, S_22);

        // Compute scattering matrix for unit cell of Bragg reflector and
        // update global scattering matrix.
        s_layer(k0(q), kx, ky, kz_A, eps_r_A, V_h, L_A,
                Suc_11, Suc_12, Suc_21, Suc_22);
        s_layer(k0(q), kx, ky, kz_B, eps_r_B, V_h, L_B,
                S_11, S_12, S_21, S_22);
        redheffer(Suc_11, Suc_12, Suc_21, Suc_22,
                  S_11, S_12, S_21, S_22);

        // Loop through all unit cells.
        for (int p=0; p<num_uc; p++)
        {
            // Update global scattering matrix using Redheffer star product.
            redheffer(Sglob_11, Sglob_12, Sglob_21, Sglob_22,
                      Suc_11, Suc_12, Suc_21, Suc_22);
        }

        // Finish by multiplying global scattering matrix by scattering matrix
        // on transmission side using Redheffer star product.
        redheffer(Sglob_11, Sglob_12, Sglob_21, Sglob_22, Strn_11, Strn_12, Strn_21, Strn_22);

        // Compute reflection and transmission coefficients.
        R_T_coeffs(pTE, pTM, theta, phi, kx, ky, kz_inc, kz_sub,
                   Sglob_11, Sglob_12, Sglob_21, Sglob_22,
                   R[q], T[q]);

        //std::cout << Sglob_11 << std::endl;
        //std::cout << Sglob_12 << std::endl;
        //std::cout << Sglob_21 << std::endl;
        //std::cout << Sglob_22 << std::endl;
        //std::cin.get();
    }

}