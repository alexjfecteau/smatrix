
#define EIGEN_USE_MKL_ALL
#define DLLEXPORT  __declspec(dllexport)

#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std::complex_literals;

typedef std::complex<double> dcomp;

Matrix2cd I = Matrix2cd::Identity();
Matrix2cd Z = Matrix2cd::Zero();

class Fields
{
public:
    // Angles are converted to radians in constructor
    double pTE, pTM, theta, phi;
    double* wvl_p;
    int num_wvl;
    Fields(double pTE, double pTM, double theta, double phi, double* wvl_p, int num_wvl):
    pTE(pTE), pTM(pTM), theta(M_PI*theta/180), phi(M_PI*phi/180), wvl_p(wvl_p), num_wvl(num_wvl) {}
};

class Layer;

class SingleLayer
{
public:
    Map<ArrayXd> wvl;
    Map<ArrayXcd> N;
    double thickness;
    int size_array;
    SingleLayer(double* wvl_p, dcomp* N_p, int size, double d):
    wvl(wvl_p, size), N(N_p, size), size_array(size), thickness(d) {}
};

class MultiLayer
{
public:
    int num_uc;
    int num_layers;
    std::vector<Layer*> unit_cell;
    MultiLayer()
    {
        num_uc = 1;
        num_layers = 0;
    }
    void add_to_unit_cell(Layer* layer_p)
    {
        num_layers += 1;
        unit_cell.resize(num_layers, layer_p);
    }
    void clear_unit_cell()
    {
        unit_cell.clear();
    }
    void set_num_repetitions(int n)
    {
        num_uc = n;
    }
};

class Layer
{
public:
    enum LayerType {single, multi} type;
    SingleLayer* single_p;
    MultiLayer* multi_p;
    Layer(LayerType t) : type(t) {}
};


extern "C" {

// Fields class
DLLEXPORT Fields* NewFields(double pTE, double pTM, double theta, double phi, double* wvl_p, int num_wvl)
    {
        Fields* fields_p = new Fields(pTE, pTM, theta, phi, wvl_p, num_wvl);
        return fields_p;
    }

// SingleLayer class
DLLEXPORT Layer* NewSingleLayer(double* wvl_p, dcomp* N_p, int size, double d)
    {
        SingleLayer* single_p = new SingleLayer(wvl_p, N_p, size, d);
        Layer* layer_p = new Layer(Layer::single);
        layer_p->single_p = single_p;
        return layer_p;
    }

// TODO : Function to destroy instances of SingleLayer

// MultiLayer class
DLLEXPORT Layer* NewMultiLayer()
    {
        MultiLayer* multi_p = new MultiLayer();
        Layer* layer_p = new Layer(Layer::multi);
        layer_p->multi_p = multi_p;
        return layer_p;
    }

DLLEXPORT void AddToUnitCell(Layer* multilayer, Layer* single_layer)
    {
        multilayer->multi_p->add_to_unit_cell(single_layer);
    }

DLLEXPORT void ClearUnitCell(Layer* multilayer)
    {
        multilayer->multi_p->clear_unit_cell();
    }

DLLEXPORT void SetNumRepetitions(Layer* multilayer, int n)
    {
        multilayer->multi_p->set_num_repetitions(n);
    }

DLLEXPORT void solve(Fields* fields, Layer* multilayer, double N_inc, double N_sub, double* R, double* T);
}

class SMatrix
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix2cd S_11 = Z;
    Matrix2cd S_12 = I;
    Matrix2cd S_21 = I;
    Matrix2cd S_22 = Z;
};

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

SMatrix s_inf(bool ref, double kx, double ky, Matrix2cd V_h, dcomp kz, dcomp eps_r)
{
    // Compute scattering matrix for semi-infinite medium
    // bool ref = true : Reflection side
    // bool ref = false : Transmission side

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

SMatrix s_slayer(double k0, double kx, double ky, Matrix2cd V_h, int wvl_index, Layer* layer)
{
    // Compute scattering matrix for layer of finite size

    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B, M, D, D_inv;

    // Permittivity and normal wavevector in single layer
    dcomp eps_r = pow(layer->single_p->N[wvl_index], 2);
    dcomp kz = sqrt(eps_r - pow(kx, 2) - pow(ky, 2));

    Q << kx*ky, eps_r - pow(kx, 2),
         pow(ky, 2) - eps_r, -kx*ky;

    double L = layer->single_p->thickness;

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

SMatrix s_mlayer(double k0, double kx, double ky, Matrix2cd V_h, int wvl_index, Layer* layer)
{
    SMatrix SLayer, SUnit, STot;
    if (layer->type == Layer::multi)
    {
        // Scattering matrix for unit cell of multilayer
        for (int l=0; l<layer->multi_p->num_layers; l+=1)
        {
            SLayer = s_mlayer(k0, kx, ky, V_h, wvl_index, layer->multi_p->unit_cell[l]);
            SUnit = redheffer(SUnit, SLayer);
        }

        // Scattering matrix for multilayer
        for (int k=0; k<layer->multi_p->num_uc; k+=1)
        {
            STot = redheffer(STot, SUnit);
        }
    }
    else
    {
        // Scattering matrix for single layer
        STot = s_slayer(k0, kx, ky, V_h, wvl_index, layer);
    }
    return STot;
}

void R_T_coeffs(Fields* fields, double kx, double ky, dcomp kz_refl, dcomp kz_tran,
                SMatrix SGlob, double& R, double& T)
{
    // Compute polarization vector of incident fields
    Vector3d k_inc(sin(fields->theta)*cos(fields->phi), sin(fields->theta)*sin(fields->phi), cos(fields->theta));
    Vector3d az(0, 0, 1), aTE, aTM, P;

    if (fields->theta == 0)
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

    P = fields->pTE*aTE + fields->pTM*aTM;
    P.normalize();

    // Compute reflected and transmitted fields
    Vector2cd e_src(P(0), P(1)), e_refl, e_tran;

    e_refl = SGlob.S_11*e_src;
    e_tran = SGlob.S_21*e_src;

    // Compute normal field components
    dcomp ez_refl = -(kx*e_refl(0) + ky*e_refl(1))/kz_refl;
    dcomp ez_tran = -(kx*e_tran(0) + ky*e_tran(1))/kz_tran;

    Vector3cd E_refl(e_refl(0), e_refl(1), ez_refl);
    Vector3cd E_tran(e_tran(0), e_tran(1), ez_tran);

    // Compute reflectance and transmittance coefficients
    R = E_refl.squaredNorm();
    T = (kz_tran/kz_refl).real() * E_tran.squaredNorm();
}

void solve(Fields* fields, Layer* multilayer, double N_inc, double N_sub, double* R, double* T)
{
    // Incident wave vector amplitude
    Map<ArrayXd> wvl(fields->wvl_p, fields->num_wvl);
    ArrayXd k0 = 2.0*M_PI/wvl;

    // Transverse wave vectors
    double kx = N_inc*sin(fields->theta)*cos(fields->phi);
    double ky = N_inc*sin(fields->theta)*sin(fields->phi);

    Matrix2cd V_h;
    V_h << kx*ky, 1.0 + ky*ky,
           -1.0 - kx*kx, -kx*ky;
    V_h = -1i*V_h;

    // Permittivities and normal wavevectors in incident medium and substrate.
    dcomp eps_r_inc = pow(N_inc, 2);
    dcomp eps_r_sub = pow(N_sub, 2);
    dcomp kz_inc = sqrt(eps_r_inc - pow(kx, 2) - pow(ky, 2));
    dcomp kz_sub = sqrt(eps_r_sub - pow(kx, 2) - pow(ky, 2));

    // Scattering matrices
    SMatrix SGlob, SRefl, STran, SMulti;

    // Compute scattering matrix for reflection side
    SRefl = s_inf(true, kx, ky, V_h, kz_inc, eps_r_inc);

    // Compute scattering matrix for transmission side
    STran = s_inf(false, kx, ky, V_h, kz_sub, eps_r_sub);

    // Loop through all wavelengths
    for (int q=0; q<fields->num_wvl; q++)
    {
        // Start with scattering matrix on reflection side
        SGlob = SRefl;

        // Compute scattering matrix for multilayer and
        // update global scattering matrix
        SMulti = s_mlayer(k0[q], kx, ky, V_h, q, multilayer);
        SGlob = redheffer(SGlob, SMulti);

        // Multiply global scattering matrix by scattering matrix
        // on transmission side using Redheffer star product
        SGlob = redheffer(SGlob, STran);

        // Compute reflection and transmission coefficients
        R_T_coeffs(fields, kx, ky, kz_inc, kz_sub, SGlob, R[q], T[q]);

        //std::cout << SGlob.S_11 << std::endl;
        //std::cout << SGlob.S_12 << std::endl;
        //std::cout << SGlob.S_21 << std::endl;
        //std::cout << SGlob.S_22 << std::endl;
        //std::cin.get();
    }
}
