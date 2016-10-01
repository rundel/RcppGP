#include <RcppArmadillo.h>
#include <boost/assign/list_of.hpp>
#include "transforms.hpp"
#include <boost/assert.hpp>


template <> std::string param_trans_map::name = "transformation";
template <> std::map<std::string, int> 
param_trans_map::map = boost::assign::map_list_of("identity", identity_trans)
                                                 ("log",      log_trans)
                                                 ("logit",    logit_trans);
               
SEXP valid_param_trans()
{
    return Rcpp::wrap(param_trans_map::valid_keys());
}

template<> double param_update<identity_trans>(double cur, double jump, arma::vec hyperparams)
{
    return cur+jump;
}

template<> double param_update<log_trans>(double cur, double jump, arma::vec hyperparams)
{
    return cur*exp(jump);
}

template<> double param_update<logit_trans>(double cur, double jump, arma::vec hyperparams)
{
    double a = hyperparams(0);
    double b = hyperparams(1);

    BOOST_ASSERT_MSG(a <= cur && cur <= b, "Param value outside bounds.");
    BOOST_ASSERT_MSG(a != b, "Hyperparameters cannot be equal. Use fixed distribution instead.");
    
    return logitInv(logit(cur,a,b) + jump, a, b);
}

double param_update(int type, double cur, double jump, arma::vec hyperparams) 
{
    if      (type == identity_trans) return param_update<identity_trans>(cur,jump,hyperparams);
    else if (type == log_trans)      return param_update<log_trans>(cur,jump,hyperparams);
    else if (type == logit_trans)    return param_update<logit_trans>(cur,jump,hyperparams);
    else    throw std::range_error("Unknown transformation.");
}


template<> double jacobian_adj<identity_trans>(double cur, arma::vec hyperparams)
{
    return 0.0;
}

template<> double jacobian_adj<log_trans>(double cur, arma::vec hyperparams)
{
    return log(cur);
}

template<> double jacobian_adj<logit_trans>(double cur, arma::vec hyperparams)
{
    double a = hyperparams(0);
    double b = hyperparams(1);

    BOOST_ASSERT_MSG(a <= cur && cur <= b, "Param value outside bounds.");
    BOOST_ASSERT_MSG(a != b, "Hyperparameters cannot be equal. Use fixed distribution instead.");
    
    return log(cur-a) + log(b-cur);
}

double jacobian_adj(int type, double cur, arma::vec hyperparams) 
{
    if      (type == identity_trans) return jacobian_adj<identity_trans>(cur,hyperparams);
    else if (type == log_trans)      return jacobian_adj<log_trans>(cur,hyperparams);
    else if (type == logit_trans)    return jacobian_adj<logit_trans>(cur,hyperparams);
    else    throw std::range_error("Unknown transformation.");
}