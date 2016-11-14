
//=================================
// include guard
#ifndef __LAYERS_H_INCLUDED__
#define __LAYERS_H_INCLUDED__

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

class LayerTest
{
public:
    SMatrix S;
    double thk;
    double size_z;
    Eigen::Vector2cd cp_1, cm_1, cp_i, cm_i;
    //virtual ~Layer() = 0;
};


class SingleLayerTest : public LayerTest
{
public:
    SMatrix S;
    double thk;
    double size_z;
    Eigen::Vector2cd cp_1, cm_1, cp_i, cm_i;
    //virtual ~Layer() = 0;
};

class Layer
{
public:
    SMatrix S;
    double thk;
    double size_z;
    virtual void compute_S(Fields* fields, int wvl_index) = 0;
    virtual void add_to_unit_cell(Layer* layer) = 0;
    virtual void clear_unit_cell() = 0;
    virtual void set_num_repetitions(int n) = 0;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //virtual ~Layer() = 0;
};

class SingleLayer : public Layer
{
public:
    Eigen::Map<Eigen::ArrayXd> wvl;
    Eigen::Map<Eigen::ArrayXcd> N;
    Eigen::ArrayXcd eps_r;
    double size_wvl;
    Eigen::ArrayXd z;
    Eigen::ArrayXXcd Ex;
    Eigen::ArrayXXcd Ey;
    Eigen::ArrayXXcd Ez;
    Eigen::ArrayXXd E2;
    Eigen::Vector2cd cp_1, cm_1, cp_i, cm_i;
    Eigen::Matrix2cd Q, Omega, V, V_inv, Vp, X, A, A_inv, B, M, D, D_inv;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SingleLayer(double* wvl_p, dcomp* N_p, int wvl_s, double d);
    void compute_S(Fields* fields, int wvl_index);
    void add_to_unit_cell(Layer* layer) {};
    void clear_unit_cell() {};
    void set_num_repetitions(int n) {};
    //~SingleLayer() {}
};

class MultiLayer : public Layer
{
public:
    int num_uc;
    int num_layers;
    double thk_uc;
    int size_z_uc;
    SMatrix S_uc;
    std::vector<Layer*> unit_cell;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    MultiLayer();
    void compute_S(Fields* fields, int wvl_index);
    void add_to_unit_cell(Layer* layer);
    void clear_unit_cell();
    void set_num_repetitions(int n);
    //~MultiLayer() {}
};

#endif