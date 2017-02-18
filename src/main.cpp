
#define DLLEXPORT  __declspec(dllexport)

#include <iostream>
//#include <complex>
//#include <cmath>

#include "smatrix.h"
#include "layers.h"
#include "sinfmed.h"
#include "fields.h"

extern "C" {

// Fields
DLLEXPORT Fields* NewFields(double pTE, double pTM, double theta, double phi, double* wvl_p, int size_wvl, double N_inc, double N_sub)
    {
        Fields* f_p = new Fields(pTE, pTM, theta, phi, wvl_p, size_wvl);
        return f_p;
    }

// Semi-infinite medium
DLLEXPORT SemiInfMed* NewSemiInfMed(dcomp* N_p, int size)
    {
        SemiInfMed* sinf_p = new SemiInfMed(N_p, size);
        return sinf_p;
    }

// SingleLayer
DLLEXPORT Layer* NewSingleLayer(dcomp* N_p, int size, double d)
    {
        Layer* l_p = new SingleLayer(N_p, size, d);
        return l_p;
    }

// TODO : Function to destroy instances of SingleLayer

// MultiLayer
DLLEXPORT Layer* NewMultiLayer()
    {
        Layer* l_p = new MultiLayer();
        return l_p;
    }

DLLEXPORT void AddToUnitCell(Layer* multilayer, Layer* single_layer)
    {
        multilayer->add_to_unit_cell(single_layer);
    }

DLLEXPORT void ClearUnitCell(Layer* multilayer)
    {
        multilayer->clear_unit_cell();
    }

DLLEXPORT void SetNumRepetitions(Layer* multilayer, int n)
    {
        multilayer->set_num_repetitions(n);
    }

// Scattering matrix

DLLEXPORT ScatteringMatrix* NewScatteringMatrix(Layer* multilayer, Fields* fields, SemiInfMed* inc_med, SemiInfMed* sub_med)
    {
        ScatteringMatrix* sm_p = new ScatteringMatrix(multilayer, fields, inc_med, sub_med);
        return sm_p;
    }

DLLEXPORT void ComputeRT(ScatteringMatrix* sm, dcomp* r_p, dcomp* t_p, double* R_p, double* T_p)
    {
        sm->compute_R_T(r_p, t_p, R_p, T_p);
    }
}
