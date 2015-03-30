#ifndef COV_FUNCS_HPP
#define COV_FUNCS_HPP

#include "enum_util.hpp"
#include "enums.hpp"

typedef enum_map<cov_func> cov_func_map;
typedef enum_property<int> cov_func_nparams;

arma::mat cov_func(int type, arma::mat const& d, arma::vec const& params);
void cov_func_gpu(int type, double const* d, double* cov, int n, int m, int n_threads, arma::vec const& p);

template<int i> arma::mat cov_func(arma::mat const& d, arma::vec const& params);

#endif