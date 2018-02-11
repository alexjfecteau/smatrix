
//=================================
// include guard
#ifndef __SINFMED_H_INCLUDED__
#define __SINFMED_H_INCLUDED__

//=================================
// forward declared dependencies
class Fields;

//=================================
// included dependencies
#include <Eigen/Dense>
#include <vector>
#include "smatrix.h"

//=================================
// type definitions
typedef std::complex<double> dcomp;

//=================================

class SemiInfMed
{
    Eigen::Matrix2cd A, A_inv, B;
    void compute_matrices(Fields* f_p, int wvl_index);
public:
    Eigen::Map<Eigen::ArrayXcd> N;
    Eigen::ArrayXcd eps_r, kz;
    dcomp kx, ky;

    SemiInfMed(dcomp* N_p, int wvl_s);
    void compute_kz(Fields* f_p);
    SMatrix compute_S_refl(Fields* f_p, int wvl_index);
    SMatrix compute_S_tran(Fields* f_p, int wvl_index);
};

#endif