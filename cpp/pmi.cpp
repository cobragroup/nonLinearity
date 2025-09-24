#include "pmi.hpp"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>
#include <future>
#include <thread>
#include <chrono>

#define FULL_SIGMA_LIMIT 20000 // corresponds to ~200 series in the original data
class list_manager
{
private:
    std::mutex lock; /**< A \c mutex protecting from simultaneous accesses to the queue. */
    int now = -1, tot_pairs = -1;

public:
    list_manager() = default;
    list_manager(list_manager &&) = default;
    void init(int tot)
    {
        tot_pairs = tot;
        now = 0;
    };
    int pop()
    {
        std::unique_lock<std::mutex> l(lock);
        if (now == tot_pairs)
            return -1;
        return now++;
    }
    int remaining()
    {
        std::unique_lock<std::mutex> l(lock);
        return tot_pairs - now;
    }
};

std::atomic_int threadCount = 0;
list_manager tasks;

std::vector<double> bin_loc(double *x, int size, int binNo)
{
    std::vector<double> vec(x, x + size);
    double k = 1. * size / binNo;
    std::vector<int> loc;
    std::vector<double> ret;
    loc.reserve(binNo + 1);
    ret.reserve(binNo + 1);
    std::sort(vec.begin(), vec.end());
    for (auto i = 0; i < binNo + 1; i++)
        loc.push_back((k * i + 0.5));
    loc[binNo] -= 1;
    for (auto i = 0; i < binNo + 1; i++)
        ret.push_back(vec[loc[i]]);
    return ret;
}

double entropy(std::vector<int> vec, double norm)
{
    double entr = 0;
    for (auto p : vec)
    {
        if (p > 0)
        {
            auto t = p / norm;
            entr -= t * log(t);
        }
    }
    return entr;
}
double pair_mutual_information(double *x, double *y, int times, int binNo)
{
    std::vector<double> xbins = bin_loc(x, times, binNo), ybins = bin_loc(y, times, binNo);
    std::vector<int> px(binNo, 0), py(binNo, 0), pxy(binNo * binNo, 0);

    for (auto i = 0; i < times; i++)
    {
        auto idx = std::distance(xbins.begin(), std::upper_bound(xbins.begin(), xbins.end() - 1, x[i])) - 1;
        auto idy = std::distance(ybins.begin(), std::upper_bound(ybins.begin(), ybins.end() - 1, y[i])) - 1;
        px[idx]++;
        py[idy]++;
        pxy[idy * binNo + idx]++;
    }
    auto entr = entropy(px, times) + entropy(py, times) - entropy(pxy, times);
    return entr;
}

void total_mutual_information(double *data, int times, int regions, int binNo, double *out)
{
    std::vector<std::vector<int>> idx;
    std::vector<double> ex(regions, 0);
    for (auto i = 0; i < regions; i++)
    {
        idx.push_back(std::vector<int>(times, 0));
        std::vector<int> px(binNo, 0);
        auto xbins = bin_loc(data + i * times, times, binNo);
        for (auto j = 0; j < times; j++)
        {
            idx[i][j] = std::distance(xbins.begin(), std::upper_bound(xbins.begin(), xbins.end() - 1, data[i * times + j])) - 1;
            px[idx[i][j]]++;
        }
        ex[i] = entropy(px, times);
    }
    auto index = 0;
    for (auto i = 0; i < regions; i++)
        for (auto j = i + 1; j < regions; j++)
        {
            std::vector<int> pxy(binNo * binNo, 0);
            for (auto k = 0; k < times; k++)
                pxy[idx[i][k] * binNo + idx[j][k]]++;
            auto te = ex[i] + ex[j] - entropy(pxy, times);
            (out)[index] = te;
            index++;
        }
    return;
}

double quantile(std::vector<double>::iterator it, int len, double quant)
{
    double v = quant * len;
    int j = v - 1;
    double fact = v - j - 1;
    return it[j] * (1 - fact) + it[j + 1] * fact;
}

size_t find_correct(const std::vector<double> &vec, const double val)
{
    auto pos = std::upper_bound(vec.begin(), vec.end() - 1, val);
    if (pos == vec.begin())
        return 0;
    size_t ind = std::distance(vec.begin(), pos);
    if (pos[0] - val < val - pos[-1])
        return ind;
    return ind - 1;
}

void series_stats(double *data, int numSurrogates, const double correctedpercpointer[3], const double fractions[3], std::array<std::vector<double>, 3> &to_ratio, std::array<std::vector<double>, 3> &to_ratioContr)
{
    threadCount++;
    int j = tasks.pop();
    while (j >= 0)
    {
        auto firstPos = data + j * (numSurrogates + 1);
        std::vector<double> perc(firstPos, firstPos + numSurrogates + 1);
        std::sort(perc.begin() + 1, perc.end());
        for (auto i = 0; i < 3; i++)
        {
            auto quant = quantile(perc.begin() + 1, numSurrogates, correctedpercpointer[i]);
            to_ratio[i][j] = perc[0] > quant;
        }
        for (auto i = 1; i < 3; i++)
        {
            double small = std::distance(perc.begin() + 1, std::upper_bound(perc.begin() + 1, perc.end(), perc[0]));
            to_ratioContr[i][j] = (1 - small / (numSurrogates + 1)) < (1 - fractions[i] + 1e-6);
        }

        j = tasks.pop();
    }
    threadCount--;
}

void vertical_stats(double *data, int numPairs, int numSurrogates, std::vector<double> &to_meanData, const std::vector<double> &estimated, double *actual)
{
    threadCount++;
    int j = tasks.pop();
    while (j >= 0)
    {
        for (auto i = 0; i < numPairs; i++)
        {
            to_meanData[j] += actual[find_correct(estimated, *(data + j + i * (numSurrogates + 1)))];
        }

        j = tasks.pop();
    }
    threadCount--;
}

returnStats statistics(double *data, int numPairs, int numSurrogates, double *estim, double *actual, int bins, int numThreads, bool extended_stats)
{
    returnStats result;
    double meanSurr = 0, sigma2 = 0;
    std::vector<double> to_meanData(numSurrogates + 1, 0);
    std::vector<double> estimated(estim, estim + bins);
    std::vector<std::thread> workers;

    if (extended_stats)
    {
        double correctedpercpointer[3], fractions[3] = {0.05, 0.95, 0.99};
        double ratioContr[3] = {0}, ratio[3] = {0};
        std::array<std::vector<double>, 3> to_ratioContr = {std::vector<double>(numPairs, 0), std::vector<double>(numPairs, 0), std::vector<double>(numPairs, 0)}, to_ratio = {std::vector<double>(numPairs, 0), std::vector<double>(numPairs, 0), std::vector<double>(numPairs, 0)};
        for (auto i = 0; i < 3; i++)
            correctedpercpointer[i] = (numSurrogates * fractions[i] - 0.5) / (numSurrogates - 1);
        tasks.init(numPairs);
        for (auto j = 0; j < numThreads; j++)
        {
            workers.push_back(std::thread(&series_stats, data, numSurrogates, correctedpercpointer, fractions, std::ref(to_ratio), std::ref(to_ratioContr)));
        }
        while (tasks.remaining() || threadCount)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        for (auto &t : workers)
            if (t.joinable())
                t.join();

        workers.clear();

        for (auto j = 0; j < numPairs; j++)
        {
            for (auto i = 0; i < 3; i++)
                ratio[i] += to_ratio[i][j];
            for (auto i = 1; i < 3; i++)
                ratioContr[i] += to_ratioContr[i][j];
        }

        result.ratio05 = 1 - ratio[0] / numPairs;
        result.ratio95 = ratio[1] / numPairs;
        result.ratio99 = ratio[2] / numPairs;
        result.ratio95control = ratioContr[1] / numPairs;
        result.ratio99control = ratioContr[2] / numPairs;
    }
    else
    {
        result.ratio05 = NAN;
        result.ratio95 = NAN;
        result.ratio99 = NAN;
        result.ratio95control = NAN;
        result.ratio99control = NAN;
    }

    tasks.init(numSurrogates + 1);
    for (auto j = 0; j < numThreads; j++)
    {
        workers.push_back(std::thread(&vertical_stats, data, numPairs, numSurrogates, std::ref(to_meanData), std::ref(estimated), actual));
    }
    while (tasks.remaining() || threadCount)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    for (auto &t : workers)
        if (t.joinable())
            t.join();

    for (auto j = 1; j < numSurrogates + 1; j++)
    {
        meanSurr += to_meanData[j];
    }
    meanSurr /= numSurrogates;

    for (auto j = 1; j < numSurrogates + 1; j++)
    {
        sigma2 += (to_meanData[j] - meanSurr) * (to_meanData[j] - meanSurr);
    }

    result.totalMI = to_meanData[0] / numPairs;
    result.gaussMI = meanSurr / numPairs;
    result.sigmaGaussMI = sqrt(sigma2) / numSurrogates / numPairs;

    return result;
}

void correct_vector(double *data, int numValues, double *estim, double *actual, int bins, double *out)
{
    std::vector<double> estimated(estim, estim + bins);
    for (auto i = 0; i < numValues; i++)
        out[i] = actual[find_correct(estimated, data[i])];
}

void quantile_vector(double *data, int numPairs, int numSurrogates, double *quant, int nquant, double *out)
{
    std::vector<double> correctedpercpointer;
    for (auto i = 0; i < nquant; i++)
        correctedpercpointer.push_back((numSurrogates * quant[i] - 0.5) / (numSurrogates - 1));
    for (auto j = 0; j < numPairs; j++)
    {
        auto firstPos = data + j * (numSurrogates + 1);
        std::vector<double> perc(firstPos, firstPos + numSurrogates + 1);
        std::sort(perc.begin() + 1, perc.end());
        for (auto i = 0; i < nquant; i++)
            out[numPairs * i + j] = quantile(perc.begin() + 1, numSurrogates, correctedpercpointer[i]);
    }
}