#pragma once

struct returnStats
{
    double ratio95control; 
    double ratio99control; 
    double ratio05; 
    double ratio95; 
    double ratio99; 
    double totalMI; 
    double gaussMI; 
};

extern "C"
{
    double pair_mutual_information(double *x, double* y, int size, int binNo);
    void total_mutual_information(double *data, int times, int regions, int binNo, double *out);
    returnStats statistics (double *data, int numPairs, int numSurrogates, double *estim, double *actual, int bins);
    void correct_vector (double *data, int numValues, double *estim, double *actual, int bins, double *out);
    void quantile_vector (double *data, int numPairs, int numSurrogates, double* quant, int nquant, double *out);
}