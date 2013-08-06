#ifndef COV_FUNCS_HPP
#define COV_FUNCS_HPP

#include "enum_util.hpp"
#include "enums.hpp"

typedef enum_map<cov_func> cov_func_map;
typedef enum_property<int> cov_func_nparams;

RcppExport SEXP valid_cov_funcs();
RcppExport SEXP valid_nparams(SEXP func_r);

arma::mat cov_func(int type, arma::mat const& d, arma::vec const& params);
void cov_func_gpu(int type, double* d, double* cov, int n, int m, int n_threads, arma::vec const& p);

template<int i> arma::mat cov_func(arma::mat const& d, arma::vec const& params);

#ifdef USE_GPU
void nugget_cov_gpu(double* dist, double* cov,
                    const int n, const int m,
                    double sigma2, int n_threads);


void constant_cov_gpu(double* dist, double* cov,
                      const int n, const int m,
                      double sigma2, int n_threads);


void exponential_cov_gpu(double* dist, double* cov,
                         const int n, const int m,
                         double sigma2, double phi, 
                         int n_threads);


void gaussian_cov_gpu(double* dist, double* cov,
                      const int n, const int m,
                      double sigma2, double phi, 
                      int n_threads);


void powered_exponential_cov_gpu(double* dist, double* cov,
                                 const int n, const int m,
                                 double sigma2, double phi, 
                                 double kappa, int n_threads);


void spherical_cov_gpu(double* dist, double* cov,
                       const int n, const int m,
                       double sigma2, double phi, 
                       int n_threads);


void rational_quadratic_cov_gpu(double* dist, double* cov,
                                const int n, const int m,
                                double sigma2, double phi, 
                                double alpha, int n_threads);


void periodic_cov_gpu(double* dist, double* cov,
                      const int n, const int m,
                      double sigma2, double phi, 
                      double gamma, int n_threads);


void exp_periodic_cov_gpu(double* dist, double* cov,
                          const int n, const int m,
                          double sigma2, double phi1, 
                          double gamma, double phi2, 
                          int n_threads);

#endif

#endif