
#define DLLEXPORT  __declspec(dllexport)

//#include <iostream>
//#include <complex>
//#include <cmath>

#include "smatrix.h"
#include "layers.h"
#include "fields.h"

extern "C" {

// Fields functions
DLLEXPORT Fields* NewFields(double pTE, double pTM, double theta, double phi, double* wvl_p, int size_wvl, double N_inc, double N_sub)
    {
        Fields* f_p = new Fields(pTE, pTM, theta, phi, wvl_p, size_wvl, N_inc, N_sub);
        return f_p;
    }

// SingleLayer functions
DLLEXPORT Layer* NewSingleLayer(double* wvl_p, dcomp* N_p, int size, double d)
    {
        Layer* l_p = new SingleLayer(wvl_p, N_p, size, d);
        return l_p;
    }

// TODO : Function to destroy instances of SingleLayer

// MultiLayer functions
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

// Scattering matrix functions

DLLEXPORT ScatteringMatrix* NewScatteringMatrix(Layer* multilayer, Fields* fields)
    {
        ScatteringMatrix* sm_p = new ScatteringMatrix(multilayer, fields);
        return sm_p;
    }

DLLEXPORT void ComputeRT(ScatteringMatrix* sm, double* R, double* T)
    {
        sm->compute_R_T(R, T);
    }

DLLEXPORT void ComputeE(ScatteringMatrix* sm, int wvl_index, double* E2)
    {
        sm->compute_E(wvl_index, E2);
    }
}
