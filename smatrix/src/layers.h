
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

class Layer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SMatrix S;
    double thk;
    virtual void compute_S(Fields* fields, int wvl_index) = 0;
    virtual void add_to_unit_cell(Layer* layer) = 0;
    virtual void clear_unit_cell() = 0;
    virtual void set_num_repetitions(int n) = 0;
    //virtual ~Layer() = 0;
};

class SingleLayer : public Layer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Map<Eigen::ArrayXcd> N;
    Eigen::ArrayXcd eps_r;
    dcomp kx, ky, kz;

    SingleLayer(dcomp* N_p, int wvl_s, double d);
    void compute_S(Fields* f_p, int wvl_index);
    void add_to_unit_cell(Layer* layer) {};
    void clear_unit_cell() {};
    void set_num_repetitions(int n) {};
    //~SingleLayer() {}
};

class MultiLayer : public Layer
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int num_uc;
    int num_layers;
    double thk_uc;
    SMatrix S_uc;
    std::vector<Layer*> unit_cell;

    MultiLayer();
    void compute_S(Fields* f_p, int wvl_index);
    void add_to_unit_cell(Layer* layer);
    void clear_unit_cell();
    void set_num_repetitions(int n);
    //~MultiLayer() {}
};

#endif