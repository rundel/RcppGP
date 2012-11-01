#include <RcppArmadillo.h>
#include <boost/assign/list_of.hpp>
#include "distributions.hpp"


template <> std::string param_dists_map::name = "transformation";
template <> std::map<std::string, int> 
param_dists_map::map = boost::assign::map_list_of("uniform",       uniform_dist)
                                                 ("inverse gamma", invgamma_dist)
                                                 ("ig",            invgamma_dist)
                                                 ("normal",        normal_dist);
               
SEXP valid_param_dists()
{
    return Rcpp::wrap(param_dists_map::valid_keys());
}

template<> double log_likelihood<uniform_dist>(double x, arma::vec hyperparams)
{
    return 0.0;
}

template<> double log_likelihood<invgamma_dist>(double x, arma::vec hyperparams)
{
    double a = hyperparams(0);
    double b = hyperparams(1);
    return -(a+1)*log(x) - b / x;
}

template<> double log_likelihood<normal_dist>(double x, arma::vec hyperparams)
{
    double mu = hyperparams(0);
    double sd = hyperparams(1);
    return -log(sd) - pow((x-mu)/sd,2)/2;
}

double log_likelihood(int dist, double x, arma::vec hyperparams) 
{
    if      (dist == uniform_dist)  return log_likelihood<uniform_dist>(x,hyperparams);
    else if (dist == invgamma_dist) return log_likelihood<invgamma_dist>(x,hyperparams);
    else if (dist == normal_dist)   return log_likelihood<normal_dist>(x,hyperparams);
    else    throw std::range_error("Unknown hyperprior distribution.");
}

double binomial_logpost(arma::vec const& Y, arma::vec const& eta, arma::vec const& w, arma::vec const& weights)
{
    arma::vec p = 1.0/(1+arma::exp(-eta-w));
    arma::vec loglik = Y * arma::log(p) + (weights-Y) * arma::log(1.0-p);
  
    return arma::accu(loglik);
}


double poisson_logpost(arma::vec const& Y, arma::vec const& eta, arma::vec const& w)
{
    arma::vec loglik = -arma::exp(eta+w) + Y*(eta+w);

    return arma::accu(loglik);
}