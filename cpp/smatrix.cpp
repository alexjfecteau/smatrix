
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

class SingleLayer
{
public:
    Map<ArrayXd> wvl;
    Map<ArrayXcd> N;
    const double thickness;
    const int size_array;
    SingleLayer(double* wvl_p, dcomp* N_p, int size, double d) : wvl(wvl_p, size), N(N_p, size), size_array(size), thickness(d) {}
};

class MultiLayer
{
public:
    const int num_uc;
    MultiLayer() : num_uc(10) {}
};

extern "C" {

// Functions for ctype import in python.

DLLEXPORT SingleLayer* NewSingleLayer(double* wvl_p, dcomp* N_p, int size, double d)
    {
        return new SingleLayer(wvl_p, N_p, size, d);
    }

// TODO : Function to destroy instances of SingleLayer

DLLEXPORT MultiLayer* NewMultiLayer(void)
    {
        return new MultiLayer();
    }

DLLEXPORT void PrintWvl(SingleLayer* layer)
    {
        std::cout << layer->size_array << std::endl;
    }

DLLEXPORT void solve(double pTE, double pTM, double theta, double phi, double* wvl_p, dcomp* N_polar_p, int num_wvl,
                     double L_polar, double N_A, double N_B, double L_A, double L_B, int num_uc, double N_inc, double N_sub,
                     double* R_p, double* T_p);
}

class SMatrix
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix2cd S_11;
    Matrix2cd S_12;
    Matrix2cd S_21;
    Matrix2cd S_22;
};

SMatrix redheffer(SMatrix SA, SMatrix SB)
{
    // Compute Redheffer star product.

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

SMatrix s_inf(bool ref, double kx, double ky, dcomp kz, dcomp eps_r, Matrix2cd V_h)
{
    // Compute scattering matrix for semi-infinite medium.
    // bool ref = true : Reflection side.
    // bool ref = false : Transmission side.

    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B;

    Q << kx*ky, eps_r - pow(kx, 2),
         pow(ky, 2) - eps_r, -kx*ky;

    Omega = 1i*kz*I;
    V = Q*Omega.inverse();
    Vp = V_h.inverse()*V;
    A = I + Vp;
    B = I - Vp;
    A_inv = A.inverse();

    SMatrix SM;

    if (ref)
    {
        SM.S_11 = -A_inv*B;
        SM.S_12 = 2*A_inv;
        SM.S_21 = 0.5*(A - B*A_inv*B);
        SM.S_22 = B*A_inv;
    }
    else
    {
        SM.S_11 = B*A_inv;
        SM.S_12 = 0.5*(A - B*A_inv*B);
        SM.S_21 = 2*A_inv;
        SM.S_22 = -A_inv*B;
    }

    return SM;
}

SMatrix s_layer(double k0, double kx, double ky, dcomp kz, dcomp eps_r, Matrix2cd& V_h, double L)
{
    // Compute scattering matrix for layer of finite size.

    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B, M, D, D_inv;

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

    SMatrix SM;
    SM.S_11 = D_inv*(M*A - B);
    SM.S_22 = SM.S_11;
    SM.S_12 = D_inv*X*(A - B*A_inv*B);
    SM.S_21 = SM.S_12;

    return SM;
}

void R_T_coeffs(double pTE, double pTM, double theta, double phi,
                double kx, double ky, dcomp kz_refl, dcomp kz_tran,
                SMatrix SGlob, double& R, double& T)
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
    Vector2cd e_src(P(0), P(1)), e_refl, e_tran;

    e_refl = SGlob.S_11*e_src;
    e_tran = SGlob.S_21*e_src;

    // Compute normal field components.
    dcomp ez_refl = -(kx*e_refl(0) + ky*e_refl(1))/kz_refl;
    dcomp ez_tran = -(kx*e_tran(0) + ky*e_tran(1))/kz_tran;

    Vector3cd E_refl(e_refl(0), e_refl(1), ez_refl);
    Vector3cd E_tran(e_tran(0), e_tran(1), ez_tran);

    // Compute reflectance and transmittance coefficients.
    R = E_refl.squaredNorm();
    T = (kz_tran/kz_refl).real() * E_tran.squaredNorm();
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
    Matrix2cd V_h;
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

    // Scattering matrices.
    SMatrix SGlob, SRefl, STran, SPolar, SBragg, SLayerA, SLayerB;

    // Compute scattering matrix for reflection side and
    SRefl = s_inf(true, kx, ky, kz_inc, eps_r_inc, V_h);

    // Compute scattering matrix for transmission side.
    STran = s_inf(false, kx, ky, kz_sub, eps_r_sub, V_h);

    // Loop through all wavelengths.
    for (int q=0; q<num_wvl; q++)
    {
        // Start with scattering matrix on reflection side.
        SGlob.S_11 = SRefl.S_11;
        SGlob.S_12 = SRefl.S_12;
        SGlob.S_21 = SRefl.S_21;
        SGlob.S_22 = SRefl.S_22;

        // Compute scattering matrix for polar material and
        // update global scattering matrix.
        SPolar = s_layer(k0(q), kx, ky, kz_polar(q), eps_r_polar(q), V_h, L_polar);
        SGlob = redheffer(SGlob, SPolar);

        // Compute scattering matrix for unit cell of Bragg reflector and
        // update global scattering matrix.
        SLayerA = s_layer(k0(q), kx, ky, kz_A, eps_r_A, V_h, L_A);
        SLayerB = s_layer(k0(q), kx, ky, kz_B, eps_r_B, V_h, L_B);
        SBragg = redheffer(SLayerA, SLayerB);

        // Loop through all unit cells.
        for (int p=0; p<num_uc; p++)
        {
            // Update global scattering matrix using Redheffer star product.
            SGlob = redheffer(SGlob, SBragg);
        }

        // Finish by multiplying global scattering matrix by scattering matrix
        // on transmission side using Redheffer star product.
        SGlob = redheffer(SGlob, STran);

        // Compute reflection and transmission coefficients.
        R_T_coeffs(pTE, pTM, theta, phi, kx, ky, kz_inc, kz_sub, SGlob, R[q], T[q]);

        //std::cout << SGlob.S_11 << std::endl;
        //std::cout << SGlob.S_12 << std::endl;
        //std::cout << SGlob.S_21 << std::endl;
        //std::cout << SGlob.S_22 << std::endl;
        //std::cin.get();
    }
}
