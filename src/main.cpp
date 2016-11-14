
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

//DLLEXPORT void ComputeE(Fields* fields, Layer* multilayer)
//    {
//        fields->compute_E(multilayer);
//    }

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

DLLEXPORT ScatteringMatrix* NewScatteringMatrix(Layer* multilayer, Fields* fields, double* R, double* T)
    {
        ScatteringMatrix* sm_p = new ScatteringMatrix(multilayer, fields, R, T);
        return sm_p;
    }


DLLEXPORT void SolveSM(ScatteringMatrix* sm)
    {
        sm->solve();
    }

}
