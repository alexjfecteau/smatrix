
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

class SMatrix
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Matrix2cd S_11 = Z;
    Matrix2cd S_12 = I;
    Matrix2cd S_21 = I;
    Matrix2cd S_22 = Z;
};

class Layer;

class SingleLayer
{
public:
    Map<ArrayXd> wvl;
    Map<ArrayXcd> N;
    double thickness;
    int size_array;
    Matrix2cd V;
    SMatrix S;
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

class Fields
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double pTE, pTM, theta, phi, kx, ky, N_inc, N_sub;
    dcomp eps_r_inc, eps_r_sub, kz_inc, kz_sub;
    Vector3d k_inc, az, aTE, aTM, P;
    Vector3cd E_refl, E_tran;
    Map<ArrayXd> wvl;
    int num_wvl;
    ArrayXd k0;
    Matrix2cd V_h;

    Fields(double pTE, double pTM, double theta, double phi, double* wvl_p, int num_wvl, double N_inc, double N_sub):
    pTE(pTE), pTM(pTM), theta(theta), phi(phi), wvl(wvl_p, num_wvl), num_wvl(num_wvl), N_inc(N_inc), N_sub(N_sub)
    {
        // Convert angles to radians
        theta = M_PI*theta/180;
        phi = M_PI*phi/180;

        // Incident wave vector amplitude
        k0 = 2.0*M_PI/wvl;

        // Transverse wave vectors
        kx = N_inc*sin(theta)*cos(phi);
        ky = N_inc*sin(theta)*sin(phi);

        V_h << kx*ky, 1.0 + ky*ky,
               -1.0 - kx*kx, -kx*ky;
        V_h = -1i*V_h;

        // Permittivities and longitudinal wavevectors in incident medium and substrate
        eps_r_inc = pow(N_inc, 2);
        eps_r_sub = pow(N_sub, 2);
        kz_inc = sqrt(eps_r_inc - pow(kx, 2) - pow(ky, 2));
        kz_sub = sqrt(eps_r_sub - pow(kx, 2) - pow(ky, 2));

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
    void E_refl_tran(SMatrix SGlob);
    SMatrix s_refl();
    SMatrix s_tran();
    Vector3cd E_slayer(Layer* layer, Vector2cd& cp_1, Vector2cd& cm_1, double z, int wvl_index);
    Vector3cd E_mlayer(Layer* layer, Vector2cd& cp_1, Vector2cd& cm_1, double z, int wvl_index);
    void E2_field(Layer* multilayer, double z, double* E2);
};

void Fields::E_refl_tran(SMatrix SGlob)
{
    // Compute reflected and transmitted fields for given scattering matrix

    // Transverse field components
    Vector2cd e_src(P(0), P(1)), e_refl, e_tran;

    e_refl = SGlob.S_11*e_src;
    e_tran = SGlob.S_21*e_src;

    // Longitudinal field components
    dcomp ez_refl = -(kx*e_refl(0) + ky*e_refl(1))/kz_inc;
    dcomp ez_tran = -(kx*e_tran(0) + ky*e_tran(1))/kz_sub;

    // Update reflected and transmitted fields
    E_refl << e_refl(0), e_refl(1), ez_refl;
    E_tran << e_tran(0), e_tran(1), ez_tran;
}

SMatrix Fields::s_refl()
{
    // Compute scattering matrix for semi-infinite medium on reflection side

    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B;

    Q << kx*ky, eps_r_inc - pow(kx, 2),
         pow(ky, 2) - eps_r_inc, -kx*ky;

    Omega = 1i*kz_inc*I;
    V = Q*Omega.inverse();
    Vp = V_h.inverse()*V;
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

SMatrix Fields::s_tran()
{
    // Compute scattering matrix for semi-infinite medium on transmission side

    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B;

    Q << kx*ky, eps_r_sub - pow(kx, 2),
         pow(ky, 2) - eps_r_sub, -kx*ky;

    Omega = 1i*kz_sub*I;
    V = Q*Omega.inverse();
    Vp = V_h.inverse()*V;
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

Vector3cd Fields::E_slayer(Layer* layer, Vector2cd& cp_1, Vector2cd& cm_1, double z, int wvl_index)
{
    // Compute electric field in single layer

    // Permittivity and longitudinal wavevector in single layer
    dcomp eps_r = pow(layer->single_p->N[wvl_index], 2);
    dcomp kz = sqrt(eps_r - pow(kx, 2) - pow(ky, 2));

    // Amplitude coefficients in layer
    Matrix2cd V_i = layer->single_p->V;
    Matrix2cd V_i_inv = V_i.inverse();
    Vector2cd cp_i = 0.5*V_i_inv*( (V_i + V_h)*cp_1 + (V_i - V_h)*cm_1 );
    Vector2cd cm_i = 0.5*V_i_inv*( (V_i - V_h)*cp_1 + (V_i + V_h)*cm_1 );

    // Transverse electric fields
    Vector2cd Exy_i = (1i*kz*I*z).exp() * cp_i + (-1i*kz*I*z).exp() * cm_i;

    // Longitudinal electric field
    dcomp Ez_i = -(kx*Exy_i(0) + ky*Exy_i(1))/kz;
    Vector3cd E_i(Exy_i(0), Exy_i(1), Ez_i);

    // Update amplitude coefficients for next layer
    SMatrix S = layer->single_p->S;
    cm_1 = S.S_12.inverse()*(cm_1 - S.S_11*cp_1);
    cp_1 = S.S_21*cp_1 + S.S_22*cm_1;

    return E_i;
}

Vector3cd Fields::E_mlayer(Layer* layer, Vector2cd& cp_1, Vector2cd& cm_1, double z, int wvl_index)
{
    // Compute electric field inside multilayer

    Vector3cd ELayer;
    if (layer->type == Layer::multi)
    {
        for (int k=0; k<layer->multi_p->num_uc; k+=1)
        {
            // Scattering matrix for unit cell of multilayer
            for (int l=0; l<layer->multi_p->num_layers; l+=1)
            {
                ELayer = E_mlayer(layer->multi_p->unit_cell[l], cp_1, cm_1, z, wvl_index);
            }
        }
    }
    else
    {
        // Electric field for single layer
        ELayer = E_slayer(layer, cp_1, cm_1, z, wvl_index);
    }
    return ELayer;
}

void Fields::E2_field(Layer* multilayer, double z, double* E2)
{
    // Compute electric field squared norm inside multilayer

    // Loop through all wavelengths
    for (int q=0; q<num_wvl; q++)
    {
        // Amplitude coefficients
        Vector2cd cp(P(0)), cm(P(1));

        Vector3cd E = E_mlayer(multilayer, cp, cm, z, q);
        E2[q] = E.squaredNorm();
    }
}

extern "C" {

// Fields class
DLLEXPORT Fields* NewFields(double pTE, double pTM, double theta, double phi, double* wvl_p, int num_wvl, double N_inc, double N_sub)
    {
        Fields* fields_p = new Fields(pTE, pTM, theta, phi, wvl_p, num_wvl, N_inc, N_sub);
        return fields_p;
    }

DLLEXPORT void E2_field(Fields* fields, Layer* multilayer, double z, double* E2)
    {
        fields->E2_field(multilayer, z, E2);
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

DLLEXPORT void solve(Fields* fields, Layer* multilayer, double* R, double* T);
}

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

SMatrix s_slayer(Fields* fields, Layer* layer, int wvl_index)
{
    // Compute scattering matrix for layer of finite size

    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B, M, D, D_inv;

    // Permittivity and longitudinal wavevector in single layer
    dcomp eps_r = pow(layer->single_p->N[wvl_index], 2);
    dcomp kz = sqrt(eps_r - pow(fields->kx, 2) - pow(fields->ky, 2));

    Q << fields->kx*fields->ky, eps_r - pow(fields->kx, 2),
         pow(fields->ky, 2) - eps_r, -fields->kx*fields->ky;

    double L = layer->single_p->thickness;

    Omega = 1i*kz*I;
    V = Q*Omega.inverse();
    X = (fields->k0[wvl_index]*L*Omega).exp();
    Vp = V.inverse()*fields->V_h;
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

    // Save V and S matrices for further use
    layer->single_p->V = V;
    layer->single_p->S = SM;

    return SM;
}

SMatrix s_mlayer(Fields* fields, Layer* layer, int wvl_index)
{
    SMatrix SLayer, SUnit, STot;
    if (layer->type == Layer::multi)
    {
        // Scattering matrix for unit cell of multilayer
        for (int l=0; l<layer->multi_p->num_layers; l+=1)
        {
            SLayer = s_mlayer(fields, layer->multi_p->unit_cell[l], wvl_index);
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
        STot = s_slayer(fields, layer, wvl_index);
    }
    return STot;
}

void solve(Fields* fields, Layer* multilayer, double* R, double* T)
{
    // Scattering matrices
    SMatrix SGlob, SRefl, STran, SMulti;

    // Scattering matrix for reflection side
    SRefl = fields->s_refl();

    // Scattering matrix for transmission side
    STran = fields->s_tran();

    // Loop through all wavelengths
    for (int q=0; q<fields->num_wvl; q++)
    {
        // Scattering matrix on reflection side
        SGlob = SRefl;

        // Multiply global scattering matrix by scattering matrix
        // of multilayer using Redheffer star product
        SMulti = s_mlayer(fields, multilayer, q);
        SGlob = redheffer(SGlob, SMulti);

        // Multiply global scattering matrix by scattering matrix
        // on transmission side using Redheffer star product
        SGlob = redheffer(SGlob, STran);

        // Compute reflected and transmitted electric fields
        fields->E_refl_tran(SGlob);

        // Compute reflection and transmission coefficients
        R[q] = fields->E_refl.squaredNorm();
        T[q] = (fields->kz_sub/fields->kz_inc).real() * fields->E_tran.squaredNorm();
    }
}

