#include <RcppArmadillo.h>
#include <boost/assign/list_of.hpp>

#include "assert.hpp"
#include "cov_funcs.hpp"


template <> std::string cov_func_map::name = "covariance function";
template <> std::map<std::string, int> 
cov_func_map::map = boost::assign::map_list_of("nugget"             , nugget_cov)
                                              ("constant"           , const_cov)
                                              ("exponential"        , exp_cov)
                                              ("gaussian"           , gauss_cov)
                                              ("powered_exponential", powexp_cov)
                                              ("spherical"          , sphere_cov)
                                              ("matern"             , matern_cov)
                                              ("rational_quadratic" , rq_cov)
                                              ("periodic"           , periodic_cov);

template <> std::string cov_func_nparams::name = "covariance function";
template <> std::map<int, int>
cov_func_nparams::map = boost::assign::map_list_of(nugget_cov,   1)
                                                  (const_cov,    1)
                                                  (exp_cov,      2)
                                                  (gauss_cov,    2)
                                                  (powexp_cov,   3)
                                                  (sphere_cov,   3)
                                                  (matern_cov,   3)
                                                  (rq_cov,       3)
                                                  (periodic_cov, 3);

SEXP valid_cov_funcs()
{
    return Rcpp::wrap(cov_func_map::valid_keys());
}

SEXP valid_nparams(SEXP func_r)
{
// Possibility of bad input
BEGIN_RCPP

    int f = cov_func_map::from_string(Rcpp::as<std::string>(func_r));
    return Rcpp::wrap( cov_func_nparams::value(f) );

END_RCPP
}

template<> arma::mat cov_func<nugget_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem == 1, "Nugget cov - requires 1 parameter.");
    RT_ASSERT(params[0] >= 0,     "Nugget cov - requires params[0] >= 0.");
    
    double tauSq = params[0];
    arma::mat res;
    if (arma::accu(arma::diagvec(d)) == 0.0) {
        res = tauSq * arma::eye<arma::mat>(d.n_rows, d.n_cols);
    } else {
        res = arma::zeros<arma::mat>(d.n_rows, d.n_cols);
        arma::uvec idx = arma::find(d == 0);
        res.elem(idx) = arma::zeros<arma::vec>(idx.n_elem);
    }

    return res;
}

template<> arma::mat cov_func<const_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem == 1, "Constant cov - requires 1 parameter.");
    RT_ASSERT(params[0] >= 0,     "Constant cov - requires params[0] >= 0.");
    
    arma::mat res(d.n_rows, d.n_cols);
    res.fill(params[0]);

    return res;
}

template<> arma::mat cov_func<exp_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem == 2, "Exponential cov - requires 2 parameters.");
    RT_ASSERT(params[0] >= 0,     "Exponential cov - requires params[0] >= 0.");
    RT_ASSERT(params[1] >= 0,     "Exponential cov - requires params[1] >= 0.");

    double sigmaSq = params[0];
    double phi = params[1];
    return sigmaSq * exp(-phi*d);
}

template<> arma::mat cov_func<gauss_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem == 2, "Gaussian cov - requires 2 parameters.");
    RT_ASSERT(params[0] >= 0,     "Gaussian cov - requires params[0] >= 0.");

    double sigmaSq = params[0];
    double phi = params[1];
    
    return sigmaSq * exp(-pow(phi*d,2.0));
}

template<> arma::mat cov_func<powexp_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem != 3, "Powered exponential cov - requires 3 parameters.");
    RT_ASSERT(params[0] >= 0,     "Powered exponential cov - requires params[0] >= 0.");
    RT_ASSERT(params[1] >= 0,     "Powered exponential cov - requires params[1] >= 0.");
    RT_ASSERT(params[2] >= 0 && params[2] <= 2, "Powered exponential cov - requires 0 <= params[1] <= 2.");

    double sigmaSq = params[0];
    double phi     = params[1];
    double nu      = params[2];
    
    return sigmaSq * exp(-pow(phi*d,nu));
}

template<> arma::mat cov_func<sphere_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem == 2, "Spherical cov - requires 2 parameters.");
    RT_ASSERT(params[0] >= 0,     "Spherical cov - requires params[0] >= 0.");
    RT_ASSERT(params[1] >= 0,     "Spherical cov - requires params[1] >= 0.");

    double sigmaSq = params(0);
    double phi = params(1);
    
    arma::mat r(d.n_rows, d.n_cols);
    if (d.n_rows == d.n_cols) {
        for(int i=0; i!=r.n_rows; ++i) {
            for(int j=0; j!=i+1; ++j) {
                r(i,j) = (d(i,j) <= 1.0/phi) ? sigmaSq * (1.0 - 1.5*phi*d(i,j) + 0.5*pow(phi*d(i,j),3.0)) : 0; 
                r(j,i) = r(i,j);
            }
        }
    } else {
        for(int i=0; i!=r.n_rows; ++i) {
            for(int j=0; j!=r.n_cols; ++j) {
                r(i,j) = (d(i,j) <= 1.0/phi) ? sigmaSq * (1.0 - 1.5*phi*d(i,j) + 0.5*pow(phi*d(i,j),3.0)) : 0; 
            }
        }    
    }

    return r;  
}

template<> arma::mat cov_func<matern_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem == 3, "Matern cov - requires 3 parameters.");
    
    double sigmaSq = params(0);
    double phi = params(1);
    double nu = params(2);

    arma::mat r(d.n_rows, d.n_cols);
    if (d.n_rows == d.n_cols) {
        for(int i=0; i!=r.n_rows; ++i) {
            for(int j=0; j!=i+1; ++j) {
                r(i,j) = sigmaSq * pow( phi*d(i,j), nu ) * Rf_bessel_k( phi*d(i,j), nu, 1.0) / (pow(2, nu-1) * Rf_gammafn(nu));
                r(j,i) = r(i,j);
            }
        }
    } else {
        for(int i=0; i!=r.n_rows; ++i) {
            for(int j=0; j!=r.n_cols; ++j) {
                r(i,j) = sigmaSq * pow( phi*d(i,j), nu ) * Rf_bessel_k( phi*d(i,j), nu, 1.0) / (pow(2, nu-1) * Rf_gammafn(nu));
            }
        }    
    }  
    
    return r;
}

template<> arma::mat cov_func<rq_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem == 3, "Rational quadratic cov - requires 3 parameters.");
    
    double sigmaSq = params(0);
    double phi = params(1);
    double alpha = params(2);
 
    return sigmaSq * pow(1+pow(phi*d,2)/alpha, -alpha);
}

template<> arma::mat cov_func<periodic_cov>(arma::mat const& d, arma::vec const& params)
{
    RT_ASSERT(params.n_elem == 3, "Periodic cov - requires 3 parameters.");
    
    double sigmaSq = params(0);
    double phi = params(1);
    double gamma = params(2);
 
    return sigmaSq * exp(-2*pow(phi*sin(arma::datum::pi*d/gamma),2));
}

arma::mat cov_func(int type, arma::mat const& d, arma::vec const& params)
{   
    if      (type == nugget_cov)    return cov_func<nugget_cov>(d,params);
    else if (type == const_cov)     return cov_func<const_cov>(d,params);
    else if (type == exp_cov)       return cov_func<exp_cov>(d,params);
    else if (type == gauss_cov)     return cov_func<gauss_cov>(d,params);
    else if (type == powexp_cov)    return cov_func<powexp_cov>(d,params);
    else if (type == sphere_cov)    return cov_func<sphere_cov>(d,params);
    else if (type == matern_cov)    return cov_func<matern_cov>(d,params);
    else if (type == rq_cov)        return cov_func<rq_cov>(d,params);
    else if (type == periodic_cov)  return cov_func<periodic_cov>(d,params);
    else    RT_ASSERT(false, "Unknown covariance function.");

    return arma::mat();
}

