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
    returnStats statistics (double *data, int numPairs, int numSurrogates);
}