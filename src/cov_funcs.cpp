#include <RcppArmadillo.h>
#include <boost/assign/list_of.hpp>

#include <boost/assert.hpp>
#include "cov_funcs.hpp"
#include "cov_funcs_gpu.hpp"


template <> std::string
cov_func_map::name = "covariance function";
template <> std::map<std::string, int>
cov_func_map::map = boost::assign::map_list_of("noop"                , noop)
                                              ("nugget"              , nugget_cov)
                                              ("constant"            , const_cov)
                                              ("exponential"         , exp_cov)
                                              ("gaussian"            , gauss_cov)
                                              ("powered_exponential" , powexp_cov)
                                              ("spherical"           , sphere_cov)
                                              ("matern"              , matern_cov)
                                              ("rational_quadratic"  , rq_cov)
                                              ("periodic"            , periodic_cov)
                                              ("periodic_exponential", periodic_exp_cov);

template <> std::string
cov_func_nparams::name = "covariance function";
template <> std::map<int, int>
cov_func_nparams::map = boost::assign::map_list_of(noop,             1)
                                                  (nugget_cov,       1)
                                                  (const_cov,        1)
                                                  (exp_cov,          2)
                                                  (gauss_cov,        2)
                                                  (powexp_cov,       3)
                                                  (sphere_cov,       3)
                                                  (matern_cov,       3)
                                                  (rq_cov,           3)
                                                  (periodic_cov,     3)
                                                  (periodic_exp_cov, 4);
// [[Rcpp::export]]
std::vector<std::string> valid_cov_funcs()
{
    return cov_func_map::valid_keys();
}

// [[Rcpp::export]]
int valid_nparams(std::string func_r)
{
    int f = cov_func_map::from_string(func_r);
    return cov_func_nparams::value(f);
}

template<> arma::mat cov_func<nugget_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem == 1, "Nugget cov - requires 1 parameter.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Nugget cov - requires params[0] (tauSq) >= 0.");

    double tauSq = params[0];
    arma::mat res;
    if (d.n_rows == d.n_cols && arma::accu(arma::diagvec(d)) == 0.0) {
        res = tauSq * arma::eye<arma::mat>(d.n_rows, d.n_cols);
    } else {
        res = arma::zeros<arma::mat>(d.n_rows, d.n_cols);
        arma::uvec idx = arma::find(d == 0.0);
        res.elem(idx) = tauSq * arma::ones<arma::vec>(idx.n_elem);
    }

    return res;
}

template<> arma::mat cov_func<const_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem == 1, "Constant cov - requires 1 parameter.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Constant cov - requires params[0] (sigmaSq) >= 0.");

    arma::mat res(d.n_rows, d.n_cols);
    res.fill(params[0]);

    return res;
}

template<> arma::mat cov_func<exp_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem == 2, "Exponential cov - requires 2 parameters.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Exponential cov - requires params[0] (sigmaSq) >= 0.");
    BOOST_ASSERT_MSG(params[1] >= 0,     "Exponential cov - requires params[1] (phi) >= 0.");

    double sigmaSq = params[0];
    double phi = params[1];

    return sigmaSq * arma::exp(-phi*d);
}

template<> arma::mat cov_func<gauss_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem == 2, "Gaussian cov - requires 2 parameters.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Gaussian cov - requires params[0] (sigmaSq) >= 0.");
    BOOST_ASSERT_MSG(params[1] >= 0,     "Gaussian cov - requires params[1] (phi) >= 0.");

    double sigmaSq = params[0];
    double phi = params[1];

    return sigmaSq * arma::exp(-0.5 * arma::square(phi*d));
}

template<> arma::mat cov_func<powexp_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem != 3, "Powered exponential cov - requires 3 parameters.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Powered exponential cov - requires params[0] (sigmaSq) >= 0.");
    BOOST_ASSERT_MSG(params[1] >= 0,     "Powered exponential cov - requires params[1] (phi) >= 0.");
    BOOST_ASSERT_MSG(params[2] >= 0 && params[2] <= 2, "Powered exponential cov - requires 0 <= params[1] (nu) <= 2.");

    double sigmaSq = params[0];
    double phi     = params[1];
    double nu      = params[2];

    return sigmaSq * exp(-pow(phi*d,nu));
}

template<> arma::mat cov_func<sphere_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem == 2, "Spherical cov - requires 2 parameters.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Spherical cov - requires params[0] (sigmaSq) >= 0.");
    BOOST_ASSERT_MSG(params[1] >= 0,     "Spherical cov - requires params[1] (phi) >= 0.");

    double sigmaSq = params(0);
    double phi = params(1);

    arma::mat r(d.n_rows, d.n_cols);
    if (d.n_rows == d.n_cols)
    {
        for(int i=0; i!=r.n_rows; ++i)
        {
            for(int j=0; j!=i+1; ++j)
            {
                r(i,j) = (d(i,j) <= 1.0/phi) ? sigmaSq * (1.0 - 1.5*phi*d(i,j) + 0.5*pow(phi*d(i,j),3.0)) : 0;
                r(j,i) = r(i,j);
            }
        }
    }
    else
    {
        for(int i=0; i!=r.n_rows; ++i)
        {
            for(int j=0; j!=r.n_cols; ++j)
                r(i,j) = (d(i,j) <= 1.0/phi) ? sigmaSq * (1.0 - 1.5*phi*d(i,j) + 0.5*pow(phi*d(i,j),3.0)) : 0;
        }
    }

    return r;
}

template<> arma::mat cov_func<matern_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem == 3, "Matern cov - requires 3 parameters.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Spherical cov - requires params[0] (sigmaSq) >= 0.");
    BOOST_ASSERT_MSG(params[1] >= 0,     "Spherical cov - requires params[1] (phi) >= 0.");
    BOOST_ASSERT_MSG(params[2] >= 0,     "Spherical cov - requires params[1] (nu) >= 0.");

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
    BOOST_ASSERT_MSG(params.n_elem == 3, "Rational quadratic cov - requires 3 parameters.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Rational quadratic cov - requires params[0] (sigmaSq) >= 0.");
    BOOST_ASSERT_MSG(params[1] >= 0,     "Rational quadratic cov - requires params[1] (phi) > 0.");
    BOOST_ASSERT_MSG(params[2] >  0,     "Rational quadratic cov - requires params[1] (alpha) >= 0.");


    double sigmaSq = params(0);
    double phi = params(1);
    double alpha = params(2);

    return sigmaSq * arma::pow(1 + 0.5 * arma::square(phi * d)/alpha, -alpha);
}

template<> arma::mat cov_func<periodic_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem == 3, "Periodic cov - requires 3 parameters.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Periodic cov - requires params[0] (sigmaSq) >= 0.");
    BOOST_ASSERT_MSG(params[1] >= 0,     "Periodic cov - requires params[1] (phi) >= 0.");
    BOOST_ASSERT_MSG(params[2] >  0,     "Periodic cov - requires params[1] (gamma) > 0.");


    double sigmaSq = params(0);
    double phi = params(1);
    double gamma = params(2);

    return sigmaSq * arma::exp(-2.0 * arma::square(phi * arma::sin(arma::datum::pi * d / gamma)));
}

template<> arma::mat cov_func<periodic_exp_cov>(arma::mat const& d, arma::vec const& params)
{
    BOOST_ASSERT_MSG(params.n_elem == 4, "Periodic cov - requires 3 parameters.");
    BOOST_ASSERT_MSG(params[0] >= 0,     "Periodic cov - requires params[0] (sigmaSq) >= 0.");
    BOOST_ASSERT_MSG(params[1] >= 0,     "Periodic cov - requires params[1] (phi1) >= 0.");
    BOOST_ASSERT_MSG(params[2] >  0,     "Periodic cov - requires params[2] (gamma) > 0.");
    BOOST_ASSERT_MSG(params[3] >= 0,     "Periodic cov - requires params[3] (phi2) >= 0.");

    double sigmaSq = params(0);
    double phi1 = params(1);
    double gamma = params(2);
    double phi2 = params(3);

    return sigmaSq * arma::exp( -2.0 * arma::square(phi1 * arma::sin(arma::datum::pi * d / gamma))
                                -0.5 * arma::square(phi2 * d)
                              );
}

arma::mat cov_func(int type, arma::mat const& d, arma::vec const& params)
{
    if      (type == nugget_cov)       return cov_func<nugget_cov>(d,params);
    else if (type == const_cov)        return cov_func<const_cov>(d,params);
    else if (type == exp_cov)          return cov_func<exp_cov>(d,params);
    else if (type == gauss_cov)        return cov_func<gauss_cov>(d,params);
    else if (type == powexp_cov)       return cov_func<powexp_cov>(d,params);
    else if (type == sphere_cov)       return cov_func<sphere_cov>(d,params);
    else if (type == matern_cov)       return cov_func<matern_cov>(d,params);
    else if (type == rq_cov)           return cov_func<rq_cov>(d,params);
    else if (type == periodic_cov)     return cov_func<periodic_cov>(d,params);
    else if (type == periodic_exp_cov) return cov_func<periodic_exp_cov>(d,params);
    else if (type == noop)             return arma::zeros<arma::mat>(d.n_rows, d.n_cols);
    else    {BOOST_ASSERT_MSG(false, "Unknown covariance function.");}

    return arma::mat();
}

#ifdef USE_GPU

void cov_func_gpu(int type, double const* d, double* cov, int n, int m, int n_threads, arma::vec const& p)
{
    BOOST_ASSERT_MSG(cov_func_nparams::value(type) == p.n_elem, "Incorrect number of parameters.");

    if      (type == nugget_cov)       nugget_cov_gpu(d, cov, n, m, p(0), n_threads);
    else if (type == const_cov)        constant_cov_gpu(d, cov, n, m, p(0), n_threads);
    else if (type == exp_cov)          exponential_cov_gpu(d, cov, n, m, p(0), p(1), n_threads);
    else if (type == gauss_cov)        gaussian_cov_gpu(d, cov, n, m, p(0), p(1), n_threads);
    else if (type == powexp_cov)       powered_exponential_cov_gpu(d, cov, n, m, p(0), p(1), p(2), n_threads);
    else if (type == sphere_cov)       spherical_cov_gpu(d, cov, n, m, p(0), p(1), n_threads);
    else if (type == rq_cov)           rational_quadratic_cov_gpu(d, cov, n, m, p(0), p(1), p(2), n_threads);
    else if (type == periodic_cov)     periodic_cov_gpu(d, cov, n, m, p(0), p(1), p(2), n_threads);
    else if (type == periodic_exp_cov) exp_periodic_cov_gpu(d, cov, n, m, p(0), p(1), p(2), p(3), n_threads);
    else if (type == noop)             {/* NOOP */}
    else if (type == matern_cov)       {BOOST_ASSERT_MSG(false, "Matern is currently unsupported on the GPU.");}
    else                               {BOOST_ASSERT_MSG(false, "Unknown covariance function.");}
}

#else 

void cov_func_gpu(int type, double const* d, double* cov, int n, int m, int n_threads, arma::vec const& p)
{
    BOOST_ASSERT_MSG(cov_func_nparams::value(type) == p.n_elem, "Incorrect number of parameters.");

    arma::mat res = cov_func(type, arma::mat(d, n, m), p);

    cov = res.memptr();
}

#endif