#include "lib/pmi.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

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

double entropy (std::vector<double> vec, double norm){
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
    std::vector<double> px(binNo,0), py(binNo,0), pxy(binNo*binNo,0);

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