
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

class SingleLayer;
class MultiLayer;

class Layer
{
public:
    enum LayerType {single, multi} type;
    SingleLayer* single_p;
    MultiLayer* multi_p;
    Layer(LayerType t) : type(t) {}
};

class SingleLayer
{
public:
    Map<ArrayXd> wvl;
    Map<ArrayXcd> N;
    ArrayXcd eps_r;
    double size_wvl;
    double thk;
    double size_z;
    ArrayXd z;
    ArrayXXcd Ex;
    ArrayXXcd Ey;
    ArrayXXcd Ez;
    ArrayXXd E2;
    Vector2cd cp_1, cm_1, cp_i, cm_i;
    Matrix2cd V, V_inv;
    SMatrix S;
    SingleLayer(double* wvl_p, dcomp* N_p, int wvl_s, double d):
    wvl(wvl_p, wvl_s), N(N_p, wvl_s), size_wvl(wvl_s), thk(d)
    {
    // Permittivity of layer
    eps_r = N.pow(2);

    // Discretize space inside layer with one point every 100 nm
    size_z = thk/0.1;
    z = ArrayXd::LinSpaced(size_z, 0, thk);

    // Electric field components and norm
    Ex(size_wvl, size_z);
    Ey(size_wvl, size_z);
    Ez(size_wvl, size_z);
    E2(size_wvl, size_z);
    }
};

class MultiLayer
{
    double thk_uc;
    int size_z_uc;
public:
    int num_uc;
    int num_layers;
    double thk;
    double size_z;
    std::vector<Layer*> unit_cell;
    MultiLayer()
    {
        thk_uc = 0;
        size_z_uc = 0;
        num_uc = 1;
        num_layers = 0;
    }
    void add_to_unit_cell(Layer* layer)
    {
        num_layers += 1;
        unit_cell.resize(num_layers, layer);
        if (layer->type == Layer::multi)
        {
            thk_uc += layer->multi_p->thk;
            size_z_uc += layer->multi_p->size_z;
        }
        else
        {
            thk_uc += layer->single_p->thk;
            size_z_uc += layer->single_p->size_z;
        }
        thk = num_uc * thk_uc;
        size_z = num_uc * size_z_uc;
    }
    void clear_unit_cell()
    {
        unit_cell.clear();
    }
    void set_num_repetitions(int n)
    {
        num_uc = n;
        thk = num_uc * thk_uc;
        size_z = num_uc * size_z_uc;
    }
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
    int size_wvl;
    ArrayXd k0;
    Matrix2cd V_h;

    Fields(double pTE, double pTM, double theta, double phi, double* wvl_p, int size_wvl, double N_inc, double N_sub):
    pTE(pTE), pTM(pTM), theta(theta), phi(phi), wvl(wvl_p, size_wvl), size_wvl(size_wvl), N_inc(N_inc), N_sub(N_sub)
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
    void E_slayer(Layer* layer, int wvl_index);
    void E_mlayer(Layer* layer, int wvl_index);
    void compute_E(Layer* multilayer);
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

void Fields::E_slayer(Layer* layer, int wvl_index)
{
    // Compute electric field in single layer at every 100 nm

    // Single layer pointer
    SingleLayer* l_p = layer->single_p;

    // Longitudinal wavevector in single layer
    dcomp kz = sqrt(l_p->eps_r[wvl_index] - pow(kx, 2) - pow(ky, 2));

    // Amplitude coefficients in layer
    l_p->cp_i = 0.5*l_p->V_inv*( (l_p->V + V_h) * l_p->cp_1 + (l_p->V - V_h) * l_p->cm_1 );
    l_p->cm_i = 0.5*l_p->V_inv*( (l_p->V - V_h) * l_p->cp_1 + (l_p->V + V_h) * l_p->cm_1 );

    // Loop over all positions in layer
    Vector2cd Exy;
    dcomp Ez;

    for (int k=0; k<layer->single_p->size_z; k+=1)
    {
        // Transverse electric fields
        Exy = (1i*kz*I*l_p->z(k)).exp()*l_p->cp_i + (-1i*kz*I*l_p->z(k)).exp()*l_p->cm_i;
        l_p->Ex(wvl_index, k) = Exy(0);
        l_p->Ey(wvl_index, k) = Exy(1);

        // Longitudinal electric field
        Ez = -(kx*Exy(0) + ky*Exy(1))/kz;
        l_p->Ez(wvl_index, k) = Ez;

        // Squared norm
        Vector3cd E(Exy(0), Exy(1), Ez);
        l_p->E2(k) = E.squaredNorm();
    }

    // Update amplitude coefficients for next layer
    SMatrix S = layer->single_p->S;
    l_p->cm_1 = l_p->S.S_12.inverse()*(l_p->cm_1 - l_p->S.S_11*l_p->cp_1);
    l_p->cp_1 = l_p->S.S_21*l_p->cp_1 + l_p->S.S_22*l_p->cm_1;
}

void Fields::E_mlayer(Layer* layer, int wvl_index)
{
    // Compute electric field inside multilayer

    if (layer->type == Layer::multi)
    {
        for (int k=0; k<layer->multi_p->num_uc; k+=1)
        {
            // Scattering matrix for unit cell of multilayer
            for (int l=0; l<layer->multi_p->num_layers; l+=1)
            {
                E_mlayer(layer->multi_p->unit_cell[l], wvl_index);
            }
        }
    }
    else
    {
        // Electric field for single layer
        E_slayer(layer, wvl_index);
    }
}

void Fields::compute_E(Layer* multilayer)
{
    // Loop through all wavelengths
    for (int q=0; q<size_wvl; q++)
    {
        E_mlayer(multilayer, q);
    }
}

extern "C" {

// Fields functions
DLLEXPORT Fields* NewFields(double pTE, double pTM, double theta, double phi, double* wvl_p, int size_wvl, double N_inc, double N_sub)
    {
        Fields* fields_p = new Fields(pTE, pTM, theta, phi, wvl_p, size_wvl, N_inc, N_sub);
        return fields_p;
    }

DLLEXPORT void ComputeE(Fields* fields, Layer* multilayer)
    {
        fields->compute_E(multilayer);
    }

// SingleLayer functions
DLLEXPORT Layer* NewSingleLayer(double* wvl_p, dcomp* N_p, int size, double d)
    {
        SingleLayer* single_p = new SingleLayer(wvl_p, N_p, size, d);
        Layer* layer_p = new Layer(Layer::single);
        layer_p->single_p = single_p;
        return layer_p;
    }

// TODO : Function to destroy instances of SingleLayer

// MultiLayer functions
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

    // Single layer pointer
    SingleLayer* l_p = layer->single_p;

    // Matrices declarations
    Matrix2cd Q, Omega, V, Vp, X, A, A_inv, B, M, D, D_inv;

    // Longitudinal wavevector in single layer
    dcomp kz = sqrt(l_p->eps_r[wvl_index] - pow(fields->kx, 2) - pow(fields->ky, 2));

    Q << fields->kx*fields->ky, l_p->eps_r[wvl_index] - pow(fields->kx, 2),
         pow(fields->ky, 2) - l_p->eps_r[wvl_index], -fields->kx*fields->ky;

    Omega = 1i*kz*I;
    V = Q*Omega.inverse();
    X = (fields->k0[wvl_index]*l_p->thk*Omega).exp();
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
    l_p->V = V;
    l_p->V_inv = V.inverse();
    l_p->S = SM;

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
    for (int q=0; q<fields->size_wvl; q++)
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
