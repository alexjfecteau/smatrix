
//=================================
// include guard
#ifndef __FIELDS_H_INCLUDED__
#define __FIELDS_H_INCLUDED__

//=================================
// forward declared dependencies
class Layer;

//=================================
// included dependencies
#include "smatrix.h"

//=================================
// type definitions
typedef std::complex<double> dcomp;

//=================================

class Fields
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double pTE, pTM, theta, phi, kx, ky, N_inc, N_sub;
    dcomp eps_r_inc, eps_r_sub, kz_inc, kz_sub;
    Eigen::Vector3d k_inc, az, aTE, aTM, P;
    Eigen::Vector3cd E_refl, E_tran;
    Eigen::Map<Eigen::ArrayXd> wvl;
    int size_wvl;
    Eigen::ArrayXd k0;
    Eigen::Matrix2cd V_h;

    Fields(double pTE, double pTM, double theta, double phi, double* wvl_p, int size_wvl, double N_inc, double N_sub);
    void E_refl_tran(SMatrix SGlob);
    void E_slayer(Layer* layer, int wvl_index);
    void E_mlayer(Layer* layer, int wvl_index);
    void compute_E(Layer* multilayer);
};

#endif