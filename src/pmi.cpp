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
double pair_mutual_information(double *x, double* y, int times, int binNo){
    std::vector<double> xbins=bin_loc(x, times, binNo), ybins=bin_loc(y, times, binNo);
    std::vector<int> px(binNo,0), py(binNo,0), pxy(binNo*binNo,0);
    
    for (auto i=0; i<times; i++){
        auto idx = std::distance(xbins.begin(), std::upper_bound(xbins.begin(), xbins.end()-1, x[i]))-1;
        auto idy = std::distance(ybins.begin(), std::upper_bound(ybins.begin(), ybins.end()-1, y[i]))-1;
        px[idx]++;
        py[idy]++;
        pxy[idy*binNo+idx]++;
    }
    auto entr = entropy(px, times)+entropy(py, times)-entropy(pxy, times);
    return entr;
}

void total_mutual_information(double *data, int times, int regions, int binNo, double *out){
    std::vector<std::vector<int>> idx;
    std::vector<double> ex(regions,0);
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
        (out)[index] = te;
        index++;
    }
    return;
}

double quantile (std::vector<double>::iterator it, int len, double quant){
    double v = quant*len;
    int j = v-1;
    double fact = v-j-1;
    return it[j]*(1-fact)+it[j+1]*fact;
}

size_t find_correct (std::vector<double>& vec, double val){
    auto pos=std::upper_bound(vec.begin(), vec.end()-1, val);
    if (pos == vec.begin()) return 0;
    size_t ind = std::distance(vec.begin(), pos);
    if (pos[0]-val<val-pos[-1]) return ind;
    return ind-1;
}

returnStats statistics (double *data, int numPairs, int numSurrogates, double *estim, double *actual, int bins){
    returnStats result;
    double correctedpercpointer[3], fractions[3] = {0.05,0.95,0.99};
    double ratioContr[3]={0}, ratio[3]={0};
    double meanData = 0, meanSurr = 0, sigma2 = 0;
    std::vector<double> estimated(estim, estim+bins), deriv(numPairs, 0);
    std::vector<std::vector<double>> tmp_toSigma(numPairs, std::vector<double>(numSurrogates));
    
    for (auto i=0; i<3; i++) correctedpercpointer[i] = (numSurrogates * fractions[i] - 0.5) / (numSurrogates - 1);

    for (auto j=0; j<numPairs; j++){
        auto firstPos = data+j*(numSurrogates+1);
        double tmp_meanSurr = 0, tmp_sigma2=0, tmp_sigma;
        std::vector<double> perc(firstPos, firstPos+numSurrogates+1);
        std::sort(perc.begin()+1, perc.end());
        for (auto i=0; i<3; i++){
            auto quant = quantile(perc.begin()+1, numSurrogates, correctedpercpointer[i]);
            ratio[i]+=perc[0]>quant;
        }
        for (auto i=1; i<3; i++){
            double small = std::distance(perc.begin()+1, std::upper_bound(perc.begin()+1, perc.end(), perc[0]));
            ratioContr[i]+=(1-small/(numSurrogates+1))<(1-fractions[i]+1e-6);
        }
        meanData+=actual[find_correct(estimated, perc[0])];
        
        for (auto i=1; i<numSurrogates+1; i++) tmp_meanSurr+=perc[i];
        tmp_meanSurr/=numSurrogates;
        meanSurr += actual[find_correct(estimated, tmp_meanSurr)];

        for (auto i=1; i<numSurrogates+1; i++) tmp_toSigma[j][i-1] = (perc[i]-tmp_meanSurr);
        for (auto i=0; i<numSurrogates; i++) tmp_sigma2 += tmp_toSigma[j][i]*tmp_toSigma[j][i];
        tmp_sigma = sqrt(tmp_sigma2)/numSurrogates;

        size_t lower = find_correct(estimated, std::max(0.,tmp_meanSurr-tmp_sigma));
        size_t upper = find_correct(estimated, tmp_meanSurr+tmp_sigma);
        if (lower != upper){
            deriv[j]= (actual[upper]-actual[lower])/(estimated[upper]-estimated[lower]);
            sigma2 += tmp_sigma2 * deriv[j] * deriv[j];
        }
    }

    for (auto j=0; j<numPairs; j++){
        for (auto k=j+1; k<numPairs; k++){
            double cov = 0;//, s1=0, s2=0;
            for (auto i=0; i<numSurrogates; i++) {
                cov += tmp_toSigma[j][i]*tmp_toSigma[k][i];
                // s1 += tmp_toSigma[j][i]*tmp_toSigma[j][i];
                // s2 += tmp_toSigma[k][i]*tmp_toSigma[k][i];
                }
            sigma2 += 2 * cov * deriv[j] * deriv[k];
            // *(wai++)=cov/sqrt(s1*s2);
        }
    }
    result.ratio05= 1 - ratio[0]/numPairs;
    result.ratio95= ratio[1]/numPairs;
    result.ratio99= ratio[2]/numPairs;
    result.ratio95control=ratioContr[1]/numPairs;
    result.ratio99control=ratioContr[2]/numPairs;
    result.totalMI=meanData/numPairs;
    result.gaussMI=meanSurr/numPairs/numSurrogates;
    result.sigmaGaussMI=sqrt(sigma2)/numSurrogates/numPairs;

    return result;
}


void correct_vector (double *data, int numValues, double *estim, double *actual, int bins, double *out){
    std::vector<double> estimated(estim, estim+bins);
    for (auto i=0; i<numValues; i++)out[i]=actual[find_correct(estimated, data[i])];
}


void quantile_vector (double *data, int numPairs, int numSurrogates, double* quant, int nquant, double *out){
    std::vector<double> correctedpercpointer;
    for (auto i=0; i<nquant; i++) correctedpercpointer.push_back((numSurrogates * quant[i] - 0.5) / (numSurrogates - 1));
    for (auto j=0; j<numPairs; j++){
        auto firstPos = data+j*(numSurrogates+1);
        std::vector<double> perc(firstPos, firstPos+numSurrogates+1);
        std::sort(perc.begin()+1, perc.end());
        for (auto i=0; i<nquant; i++)out[numPairs*i+j]=quantile(perc.begin()+1, numSurrogates, correctedpercpointer[i]);
    }
}