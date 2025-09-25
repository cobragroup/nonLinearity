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
    double sigmaGaussMI;
};

extern "C"
{
    double binning_pair_mutual_information(double *x, double *y, int size, int binNo);
    void binning_total_mutual_information(double *data, int times, int regions, int binNo, double *out);
    double pair_Chatterjee(double *vec1, double *vec2, int n, int d1, int d2, bool distance);
    void total_Chatterjee(double *data, int n, int s, int d, bool distance, double *out);
    returnStats statistics(double *data, int numPairs, int numSurrogates, double *estim, double *actual, int bins, int numThreads, bool extended_stats);
    void correct_vector(double *data, int numValues, double *estim, double *actual, int bins, double *out);
    void quantile_vector(double *data, int numPairs, int numSurrogates, double *quant, int nquant, double *out);
}