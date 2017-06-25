
//=================================
// include guard
#ifndef __SMATRIX_H_INCLUDED__
#define __SMATRIX_H_INCLUDED__

//=================================
// forward declared dependencies
class Fields;
class SemiInfMed;
class Layer;

//=================================
// included dependencies
#include <Eigen/Dense>

//=================================
// type definitions
typedef std::complex<double> dcomp;

//=================================

extern Eigen::Matrix2cd I;
extern Eigen::Matrix2cd Z;

struct SMatrix
{
    Eigen::Matrix2cd S_11 = Z;
    Eigen::Matrix2cd S_12 = I;
    Eigen::Matrix2cd S_21 = I;
    Eigen::Matrix2cd S_22 = Z;
    void reset()
    {
        S_11 = Z;
        S_12 = I;
        S_21 = I;
        S_22 = Z;
    }
};

SMatrix redheffer(SMatrix SA, SMatrix SB);

class ScatteringMatrix
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Layer* m_p;
    Fields* f_p;
    SemiInfMed* inc_p;
    SemiInfMed* sub_p;
    dcomp* r_p;
    dcomp* t_p;
    double* R_p;
    double* T_p;

    ScatteringMatrix(Layer* multilayer, Fields* fields, SemiInfMed* inc_med, SemiInfMed* sub_med);
    void compute_R_T(dcomp* r_p, dcomp* t_p, double* R_p, double* T_p);
};

#endif