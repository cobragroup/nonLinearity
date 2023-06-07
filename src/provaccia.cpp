#include "lib/pmi.hpp"
#include <iostream>
int main (){
int numPairs=1072380, bins=200, numSurrogates=99, numThreads=10;
double actual[bins]={0}, estimate[bins]={0}, *data, tmp[numSurrogates+1]={0};
for (auto i=0; i<bins; i++){
    actual[i]=1.*i/bins;
    estimate[i]=1.*i/bins;
}
for (auto i=0; i<numSurrogates+1; i++) tmp[i]=1.*u_short(i*0xeeffdeacf02+0xeeffdeacf03)/u_short(-1);
for (auto i=0; i<numSurrogates+1; i++) std::cout << tmp[i] << std::endl;
data = (double*)malloc((numSurrogates+1)*numPairs*sizeof(double**));
for (auto i=0; i<numPairs; i++) for (auto j=0; j<numSurrogates+1; j++)data[i*(numSurrogates+1)+j]=tmp[j];
auto stats = statistics (data, numPairs, numSurrogates, estimate, actual, bins, numThreads);
std::cout<<stats.totalMI<<std::endl;
return 0;
}
