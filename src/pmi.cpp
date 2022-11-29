#include "lib/pmi.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>

std::vector<double> bin_loc (double* x, int size, int binNo){
    std::vector<double> vec(x, x + size);
    double k = 1.*size/binNo;
    std::vector<int> loc;
    std::vector<double> ret;
    loc.reserve(binNo+1);
    ret.reserve(binNo+1);
    std::sort(vec.begin(), vec.end());
    for (auto i=0; i<binNo+1; i++) loc.push_back((k*i+0.5));
    loc[binNo]-=1;
    for (auto i=0; i<binNo+1; i++)ret.push_back(vec[loc[i]]);
    return ret;
}

double entropy (std::vector<int> vec, double norm){
    double entr=0;
    for (auto p: vec){
        if (p>0){
            auto t=p/norm;
            entr-=t*log(t);
        }
    }
    return entr;
}

double pair_mutual_information(double *x, double* y, int size, int binNo){
    std::vector<double> xbins=bin_loc(x, size, binNo), ybins=bin_loc(y, size, binNo);
    std::vector<int> px(binNo,0), py(binNo,0), pxy(binNo*binNo,0);

    for (auto i=0; i<size; i++){
        auto idx = std::distance(xbins.begin(), std::upper_bound(xbins.begin(), xbins.end()-1, x[i]))-1;
        auto idy = std::distance(ybins.begin(), std::upper_bound(ybins.begin(), ybins.end()-1, y[i]))-1;
        px[idx]++;
        py[idy]++;
        pxy[idy*binNo+idx]++;
    }
    auto entr = entropy(px, size)+entropy(py, size)-entropy(pxy, size);
    return entr;
}

void total_mutual_information(double *data, int times, int regions, int binNo, double *out){
    std::vector<std::vector<int>> idx;
    std::vector<double> ex(regions,0);
    std::vector<double> mi;
    for (auto i=0; i<regions; i++){
        idx.push_back(std::vector<int>(times,0));
        std::vector<int>px(binNo,0);
        auto xbins = bin_loc(data+i*times, times, binNo);
        for (auto j=0; j<times; j++){
            idx[i][j]=std::distance(xbins.begin(), std::upper_bound(xbins.begin(), xbins.end()-1, data[i*times+j]))-1;
            px[idx[i][j]]++;
        }
        ex[i]=entropy(px, times);
    }
    auto index = 0;
    for (auto i=0; i<regions; i++) for (auto j=i+1; j<regions; j++){
        std::vector<int>pxy(binNo*binNo,0);
        for (auto k=0; k<times; k++) pxy[idx[i][k]*binNo+idx[j][k]]++;
        auto te=ex[i]+ex[j]-entropy(pxy, times);
        // std::cerr << te <<"["<<index<<"]"<<std::endl;
        (out)[index] = te;
        index++;
    }
    // std::cerr << "fatto!" <<std::endl;
    return;
}


returnStats statistics (double *data, int numPairs, int numSurrogates){
    double correctedperc95pointer = (numSurrogates * (0.95) - 0.5) / (numSurrogates - 1);
    double correctedperc99pointer = (numSurrogates * (0.99) - 0.5) / (numSurrogates - 1);
    double correctedperc05pointer = (numSurrogates * (0.05) - 0.5) / (numSurrogates - 1);
}
