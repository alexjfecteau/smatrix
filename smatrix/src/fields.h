
//=================================
// include guard
#ifndef __FIELDS_H_INCLUDED__
#define __FIELDS_H_INCLUDED__

//=================================
// forward declared dependencies
class Layer;
class SemiInfMed;

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
    Eigen::Map<Eigen::ArrayXd> wvl;
    int size_wvl;
    double pTE, pTM, theta, phi;
    Eigen::ArrayXd k0;
    Eigen::ArrayXcd kx, ky;
    Eigen::Vector3d k_inc, az, aTE, aTM, P;
    Eigen::Vector3cd E_refl, E_tran;

    Fields(double pTE, double pTM, double theta, double phi, double* wvl_p, int size_wvl);
    void compute_kx_ky(SemiInfMed* inc_p);
    void compute_E_refl_tran(SMatrix SGlob, SemiInfMed* inc_p, SemiInfMed* sub_p, int wvl_index);
};

#endif