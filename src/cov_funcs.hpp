#ifndef COV_FUNCS_HPP
#define COV_FUNCS_HPP

#include "enum_util.hpp"

enum cov_func {nugget_cov, const_cov, exp_cov, gauss_cov, powexp_cov, sphere_cov, matern_cov, rq_cov, periodic_cov}; 

typedef enum_map<cov_func> cov_func_map;
typedef enum_property<int> cov_func_nparams;

RcppExport SEXP valid_cov_funcs();
RcppExport SEXP valid_nparams(SEXP func_r);

arma::mat cov_func(int type, arma::mat const& d, arma::vec const& params);

template<int i> arma::mat cov_func(arma::mat const& d, arma::vec const& params);

#endif