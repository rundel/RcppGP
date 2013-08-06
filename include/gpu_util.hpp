#ifndef GPU_UTIL_HPP
#define GPU_UTIL_HPP

#ifdef USE_GPU
void symmat(double* M, const int n, char type, const int n_threads);
void trimat(double* M, const int n, const char type, const double val, const int n_threads);
#endif

#endif