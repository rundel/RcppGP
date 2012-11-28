#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include "enum_util.hpp"

enum param_dists 
{
    fixed_dist,
    uniform_dist, 
    invgamma_dist, 
    normal_dist
}; 
typedef enum_map<param_dists> param_dists_map;
RcppExport SEXP valid_param_dists();


template<int i> double log_likelihood(double x, arma::vec const& hyperparams);
double log_likelihood(int dist, double x, arma::vec const& hyperparams);


double binomial_logpost(arma::vec const& Y, arma::vec const& eta, arma::vec const& w, arma::vec const& weights);
double poisson_logpost(arma::vec const& Y, arma::vec const& eta, arma::vec const& w);

#endif
