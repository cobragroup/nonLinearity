#pragma once


extern "C"

{
    double pair_mutual_information(double *x, double* y, int size, int binNo);
    void total_mutual_information(double *data, int times, int regions, int binNo, double *out);
}